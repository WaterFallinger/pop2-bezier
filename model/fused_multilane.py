"""Exact Multi-Lane history+self attention using a fused FlexAttention kernel.

This exact fused implementation is used by the v3.3 production trainer.
Input tensors use [batch, heads, sequence, head_dim]. The single self K/V
belonging to query i is placed at key position T+i; every other self key is
masked. The history mask is the same boolean tensor used by the v3.2 trainer.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

try:  # Triton is required by torch.compile(flex_attention); fall back on CPU-only boxes.
    import triton as _triton  # noqa: F401
    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False


def reference_attention(q, self_k, self_v, history_k, history_v, allowed_history):
    """Literal v3.2 Multi-Lane reference, including its softmax scaling."""
    history_scores = torch.matmul(q, history_k.transpose(-2, -1))
    history_scores = history_scores.masked_fill(~allowed_history, float("-inf"))
    self_scores = (q * self_k).sum(dim=-1, keepdim=True)
    scores = torch.cat((history_scores, self_scores), dim=-1)
    weights = torch.softmax(scores * (q.shape[-1] ** -0.5), dim=-1)
    return (
        torch.matmul(weights[..., :history_k.size(2)], history_v)
        + weights[..., history_k.size(2):] * self_v
    )


def validate_inputs(q, self_k, self_v, history_k, history_v, allowed_history):
    if q.ndim != 4 or self_k.shape != q.shape or self_v.shape != q.shape:
        raise ValueError("q/self_k/self_v must all be [B,H,N,D]")
    b, h, n, d = q.shape
    if history_k.shape != history_v.shape or history_k.shape[:2] != (b, h):
        raise ValueError("history K/V must both be [B,H,T,D]")
    if history_k.size(-1) != d or history_k.size(2) < 1:
        raise ValueError("history head width or length is invalid")
    if allowed_history.shape not in ((1, 1, n, history_k.size(2)),
                                     (b, 1, n, history_k.size(2))):
        raise ValueError("allowed_history must be [1 or B,1,N,T]")
    if allowed_history.dtype != torch.bool:
        raise ValueError("allowed_history must be boolean")
    if any(item.device != q.device for item in
           (self_k, self_v, history_k, history_v, allowed_history)):
        raise ValueError("all tensors must be on the same device")
    return b, n, history_k.size(2)


def build_block_mask(allowed_history, batch_size, query_length, history_length):
    """Build once per forward, then reuse this BlockMask across all layers."""
    if allowed_history.shape[0] not in (1, batch_size):
        raise ValueError("history mask batch dimension must be 1 or batch size")
    if allowed_history.shape[-2:] != (query_length, history_length):
        raise ValueError("history mask Q/T dimensions do not match")

    flattened_history_mask = allowed_history.reshape(-1)

    def mask_mod(batch, head, query_index, key_index):
        del head
        mask_batch = batch if allowed_history.shape[0] == batch_size else torch.zeros_like(batch)
        safe_history_index = key_index.clamp(max=history_length - 1)
        flat_index = ((mask_batch * query_length + query_index)
                      * history_length + safe_history_index)
        history_allowed = flattened_history_mask[flat_index]
        own_self = key_index == history_length + query_index
        return torch.where(key_index < history_length, history_allowed, own_self)

    return create_block_mask(
        mask_mod, B=batch_size, H=None, Q_LEN=query_length,
        KV_LEN=history_length + query_length,
        device=allowed_history.device,
    )


def build_fixed_grid_block_mask(query_length, history_length, lanes, device):
    """Specialize the v3.2 fixed grid: C_j visible iff j < 1 + floor(q/lanes).

    Unlike the general builder, this requires no dense captured mask. The
    resulting BlockMask is reusable across all layers and training steps with
    the same fixed grid.
    """
    if query_length < 1 or history_length < 1 or lanes < 1:
        raise ValueError("fixed-grid dimensions must be positive")
    if query_length > (history_length - 1) * lanes:
        raise ValueError("query grid exceeds its causal history")

    def mask_mod(batch, head, query_index, key_index):
        del batch, head
        history_visible = ((key_index < history_length)
                           & (key_index < 1 + query_index // lanes))
        own_self = key_index == history_length + query_index
        return history_visible | own_self

    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=query_length,
        KV_LEN=history_length + query_length, device=device,
    )


_compiled_flex = torch.compile(flex_attention) if _HAS_TRITON else flex_attention


def fused_attention(q, self_k, self_v, history_k, history_v, allowed_history,
                    block_mask=None):
    """Exact attention with autograd; only execution order/rounding can differ."""
    b, n, t = validate_inputs(
        q, self_k, self_v, history_k, history_v, allowed_history
    )
    if not _HAS_TRITON:
        # Same semantics as the v3.2 reference path; slower but dependency-free.
        return reference_attention(q, self_k, self_v, history_k, history_v, allowed_history)
    if block_mask is None:
        block_mask = build_block_mask(allowed_history, b, n, t)
    key = torch.cat((history_k, self_k), dim=2)
    value = torch.cat((history_v, self_v), dim=2)
    return _compiled_flex(q, key, value, block_mask=block_mask)
