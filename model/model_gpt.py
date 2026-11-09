import torch
import torch.nn as nn
from torch.nn import functional as F

from block_state import PRIMITIVE_TOKEN_WIDTH, VALUES_PER_BLOCK
from fused_multilane import build_fixed_grid_block_mask, fused_attention
from token_layout import TOKEN_LAYOUT


class _RMSNormFunction(torch.autograd.Function):
    """Memory-aware RMSNorm: retain BF16 x, not an FP32 activation-sized copy."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        x_float = x.float()
        inverse_rms = torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + eps
        )
        ctx.save_for_backward(x, weight, inverse_rms)
        return (x_float * inverse_rms * weight.float()).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inverse_rms = ctx.saved_tensors
        x_float = x.float()
        grad_float = grad_output.float()
        scaled_grad = grad_float * weight.float()
        projection = (scaled_grad * x_float).mean(dim=-1, keepdim=True)
        grad_x = inverse_rms * (
            scaled_grad - x_float * projection * inverse_rms.square()
        )
        reduce_dims = tuple(range(grad_float.ndim - 1))
        grad_weight = (grad_float * x_float * inverse_rms).sum(dim=reduce_dims)
        return grad_x.to(x.dtype), grad_weight.to(weight.dtype), None


class RMSNorm(nn.Module):
    """RMSNorm with FP32 accumulation and an input-dtype activation stream."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        if width <= 0 or eps <= 0:
            raise ValueError("RMSNorm width 和 eps 必须为正数。")
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps
        self.width = width
        self.use_fused = True

    def forward(self, x):
        # PyTorch's CUDA/HIP RMSNorm dispatches to its fused kernel when input
        # and affine weight share dtype. Keep an FP32 master parameter while the
        # cast node returns its gradient to that master parameter in FP32.
        if self.use_fused and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            return F.rms_norm(
                x,
                (self.width,),
                self.weight.to(dtype=x.dtype),
                self.eps,
            )
        return _RMSNormFunction.apply(x, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    """RoPE over the per-head feature axis using explicit block position ids."""

    def __init__(self, head_width, base=10000.0):
        super().__init__()
        if head_width % 2 != 0:
            raise ValueError("RoPE head width 必须是偶数。")
        if base <= 0:
            raise ValueError("RoPE base 必须为正数。")
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_width, 2, dtype=torch.float32) / head_width)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)
        self.base = float(base)

    def cos_sin(self, position_ids, dtype):
        """Build reusable RoPE factors for one or more leading position axes."""
        angles = position_ids.float().unsqueeze(-1) * self.inverse_frequency
        cos = angles.cos().unsqueeze(-2).to(dtype)
        sin = angles.sin().unsqueeze(-2).to(dtype)
        return cos, sin

    @staticmethod
    def apply_cos_sin(q, k, cos, sin):
        """Apply precomputed factors; singleton leading axes may broadcast."""
        if q.shape != k.shape:
            raise ValueError("RoPE 的 Q/K 形状必须一致。")

        def rotate(x):
            even = x[..., 0::2]
            odd = x[..., 1::2]
            return torch.stack(
                (even * cos - odd * sin, even * sin + odd * cos), dim=-1
            ).flatten(-2)

        return rotate(q), rotate(k)

    def forward(self, q, k, position_ids):
        expected_shape = q.shape[:-2]
        if q.shape != k.shape:
            raise ValueError("RoPE 的 Q/K 形状必须一致。")
        if position_ids.shape != expected_shape:
            raise ValueError(
                f"RoPE position_ids 应为 {expected_shape}，实际为 {position_ids.shape}。"
            )
        return self.apply_cos_sin(q, k, *self.cos_sin(position_ids, q.dtype))


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_width):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, d_model, bias=False)
        self.down.RESIDUAL_SCALE_INIT = True
        self.hidden_width = hidden_width

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, norm_eps=1e-6, rope_base=10000.0):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model {d_model} 必须能被 n_head {n_head} 整除。")
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.c_proj.RESIDUAL_SCALE_INIT = True
        self.n_head = n_head
        self.d_model = d_model
        self.head_width = d_model // n_head
        self.scale = self.head_width ** -0.5
        self.q_norm = RMSNorm(self.head_width, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_width, eps=norm_eps)
        self.rope = RotaryEmbedding(self.head_width, base=rope_base)

    def project_qkv(self, x, position_ids=None, rotary_cos_sin=None):
        """Project any [..., d_model] layout without changing leading axes."""
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"attention 输入末维应为 {self.d_model}，实际为 {x.size(-1)}。"
            )
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        projected_shape = (*x.shape[:-1], self.n_head, self.head_width)
        q = self.q_norm(q.reshape(projected_shape))
        k = self.k_norm(k.reshape(projected_shape))
        if rotary_cos_sin is not None:
            if position_ids is not None:
                raise ValueError("position_ids 与预计算 RoPE 不能同时提供。")
            q, k = self.rope.apply_cos_sin(q, k, *rotary_cos_sin)
        else:
            if position_ids is None:
                raise ValueError("QKV projection 必须提供位置或预计算 RoPE。")
            q, k = self.rope(q, k, position_ids)
        v = v.reshape(projected_shape)
        return q, k, v

    def forward(
        self,
        x,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
        position_ids=None,
    ):
        batch_size, query_length, channels = x.size()
        if position_ids is None:
            raise ValueError("RoPE attention 必须显式提供 position_ids。")
        q, k, v = self.project_qkv(x, position_ids)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise ValueError("branch attention mask 必须是 bool 张量。")
            expected_keys = past_length + query_length
            if (
                attention_mask.size(-2) != query_length
                or attention_mask.size(-1) != expected_keys
            ):
                raise ValueError(
                    "attention mask 尾部形状应为 "
                    f"({query_length}, {expected_keys})，实际为 "
                    f"{tuple(attention_mask.shape[-2:])}。"
                )
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=False
            )
        elif past_length == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif query_length == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            query_positions = torch.arange(query_length, device=x.device).unsqueeze(1)
            key_positions = torch.arange(
                past_length + query_length, device=x.device
            ).unsqueeze(0)
            causal_mask = key_positions <= (past_length + query_positions)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, is_causal=False
            )

        y = y.transpose(1, 2).contiguous().view(batch_size, query_length, channels)
        y = self.c_proj(y)
        if use_cache:
            return y, (k, v)
        return y


class GPTBlock(nn.Module):
    def __init__(
        self, d_model, n_head, ffn_hidden, norm_eps=1e-6, rope_base=10000.0
    ):
        super().__init__()
        self.norm_1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = CausalSelfAttention(
            d_model, n_head, norm_eps=norm_eps, rope_base=rope_base
        )
        self.norm_2 = RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLU(d_model, ffn_hidden)

    def forward(
        self,
        x,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
        position_ids=None,
    ):
        attn_output = self.attn(
            self.norm_1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if use_cache:
            attn_output, present_key_value = attn_output
        x = x + attn_output
        x = x + self.mlp(self.norm_2(x))
        if use_cache:
            return x, present_key_value
        return x

    def _validate_branched_inputs(self, x, history_key_value, first_query_block):
        if x.ndim != 4:
            raise ValueError(f"branched hidden 必须是 [B, U, K, D]，实际为 {x.shape}。")
        batch_size, query_block_count, queries_per_block, channels = x.shape
        if not 1 <= queries_per_block <= VALUES_PER_BLOCK:
            raise ValueError(
                f"branched K 必须位于 [1, {VALUES_PER_BLOCK}]，"
                f"实际为 {queries_per_block}。"
            )
        history_k, history_v = history_key_value
        if history_k.shape != history_v.shape:
            raise ValueError("history K/V 形状必须一致。")
        if (
            history_k.ndim != 4
            or history_k.size(0) != batch_size
            or history_k.size(1) != self.attn.n_head
            or history_k.size(3) != self.attn.head_width
        ):
            raise ValueError(
                "history K/V 必须为 [B, H, T, Dh]，实际为 "
                f"{tuple(history_k.shape)}。"
            )
        if not 0 <= first_query_block < history_k.size(2):
            raise ValueError("first_query_block 超出完整历史范围。")
        if first_query_block + query_block_count > history_k.size(2):
            raise ValueError("query block 范围超过完整历史长度。")
        return (
            batch_size,
            query_block_count,
            queries_per_block,
            channels,
            history_k,
            history_v,
        )

    def _project_branched_qkv(self, x, query_block_ids):
        batch_size, query_block_count, queries_per_block, _ = x.shape
        query_positions = query_block_ids.view(1, query_block_count, 1).expand(
            batch_size, -1, queries_per_block
        )
        normalized = self.norm_1(x)
        q, self_k, self_v = self.attn.project_qkv(normalized, query_positions)
        # [B, U, K, H, Dh] -> [B, U, H, K, Dh]
        q = q.permute(0, 1, 3, 2, 4)
        self_k = self_k.permute(0, 1, 3, 2, 4)
        self_v = self_v.permute(0, 1, 3, 2, 4)
        return q, self_k, self_v

    def _finish_branched_layer(self, x, attention_output):
        x = x + self.attn.c_proj(attention_output)
        x = x + self.mlp(self.norm_2(x))
        return x

    def _multi_lane_attention_projected(
        self,
        q,
        self_k,
        self_v,
        history_k,
        history_v,
        allowed_history,
    ):
        """Multi-Lane attention for already projected [B,H,N,Dh] tensors."""
        history_scores = torch.matmul(q, history_k.transpose(-2, -1))
        if allowed_history.shape not in (
            (1, 1, q.size(2), history_k.size(2)),
            (q.size(0), 1, q.size(2), history_k.size(2)),
        ):
            raise ValueError(
                "allowed_history 应为 [1或B,1,Nq,T]，实际为 "
                f"{tuple(allowed_history.shape)}。"
            )
        history_scores = history_scores.masked_fill(
            ~allowed_history, float("-inf")
        )
        self_scores = (q * self_k).sum(dim=-1, keepdim=True)
        joint_scores = torch.cat((history_scores, self_scores), dim=-1)
        joint_weights = torch.softmax(joint_scores * self.attn.scale, dim=-1)
        history_output = torch.matmul(
            joint_weights[..., : history_k.size(2)], history_v
        )
        self_output = joint_weights[..., history_k.size(2) :] * self_v
        return history_output + self_output

    def forward_shared_history_multi_lane(
        self,
        history_hidden,
        query_hidden,
        rotary_cos_sin,
        allowed_history,
        history_attention_mask=None,
        fused_block_mask=None,
    ):
        """Advance C and Q together while keeping their attention DAG separate.

        RMSNorm, QKV/c_proj, and SwiGLU are pointwise over token positions, so C
        and Q can share larger GEMMs. C still performs causal self-attention and
        Q still performs only history+self Multi-Lane attention.
        """
        if history_hidden.ndim != 3 or query_hidden.ndim != 4:
            raise ValueError("联合层要求 C=[B,T,D]、Q=[B,U,K,D]。")
        batch_size, history_length, channels = history_hidden.shape
        query_batch, query_block_count, queries_per_block, query_channels = (
            query_hidden.shape
        )
        if query_batch != batch_size or query_channels != channels:
            raise ValueError("联合层的 C/Q batch 或 hidden width 不一致。")
        query_length = query_block_count * queries_per_block
        query_flat = query_hidden.reshape(batch_size, query_length, channels)
        joint_hidden = torch.cat((history_hidden, query_flat), dim=1)

        q, k, v = self.attn.project_qkv(
            self.norm_1(joint_hidden), rotary_cos_sin=rotary_cos_sin
        )
        history_q, query_q = q.split((history_length, query_length), dim=1)
        history_k, query_k = k.split((history_length, query_length), dim=1)
        history_v, query_v = v.split((history_length, query_length), dim=1)
        history_q = history_q.transpose(1, 2)
        history_k = history_k.transpose(1, 2)
        history_v = history_v.transpose(1, 2)
        query_q = query_q.transpose(1, 2)
        query_k = query_k.transpose(1, 2)
        query_v = query_v.transpose(1, 2)

        history_attention = F.scaled_dot_product_attention(
            history_q, history_k, history_v,
            attn_mask=history_attention_mask,
            is_causal=history_attention_mask is None,
        )
        if fused_block_mask is None:
            query_attention = self._multi_lane_attention_projected(
                query_q, query_k, query_v, history_k, history_v,
                allowed_history,
            )
        else:
            query_attention = fused_attention(
                query_q, query_k, query_v, history_k, history_v,
                allowed_history, block_mask=fused_block_mask,
            )
        history_attention = history_attention.transpose(1, 2).reshape(
            batch_size, history_length, channels
        )
        query_attention = query_attention.transpose(1, 2).reshape(
            batch_size, query_length, channels
        )
        joint_attention = torch.cat((history_attention, query_attention), dim=1)
        joint_hidden = joint_hidden + self.attn.c_proj(joint_attention)
        joint_hidden = joint_hidden + self.mlp(self.norm_2(joint_hidden))

        history_hidden, query_flat = joint_hidden.split(
            (history_length, query_length), dim=1
        )
        query_hidden = query_flat.reshape(
            batch_size, query_block_count, queries_per_block, channels
        )
        return history_hidden, query_hidden

    def branched_attention_reference(self, x, history_key_value, first_query_block):
        """Per-block correctness reference with one attention group per C prefix."""
        (
            batch_size,
            query_block_count,
            queries_per_block,
            channels,
            history_k,
            history_v,
        ) = self._validate_branched_inputs(
            x, history_key_value, first_query_block
        )
        query_block_ids = torch.arange(
            first_query_block,
            first_query_block + query_block_count,
            dtype=torch.long,
            device=x.device,
        )
        q, self_k, self_v = self._project_branched_qkv(x, query_block_ids)

        attention_outputs = []
        for query_group_index in range(query_block_count):
            history_length = first_query_block + query_group_index
            query = q[:, query_group_index]
            query_k = self_k[:, query_group_index]
            query_v = self_v[:, query_group_index]
            if history_length == 0:
                # A one-element softmax is exactly one; Q/K gradients are zero.
                attention_outputs.append(query_v)
                continue

            history_k_prefix = history_k[:, :, :history_length, :]
            history_v_prefix = history_v[:, :, :history_length, :]
            history_scores = torch.matmul(
                query, history_k_prefix.transpose(-2, -1)
            )
            self_scores = (query * query_k).sum(dim=-1, keepdim=True)
            scores = torch.cat((history_scores, self_scores), dim=-1)
            weights = torch.softmax(scores * self.attn.scale, dim=-1)
            history_output = torch.matmul(
                weights[..., :history_length], history_v_prefix
            )
            self_output = weights[..., history_length:] * query_v
            attention_outputs.append(history_output + self_output)

        attention_output = torch.stack(attention_outputs, dim=1)
        attention_output = attention_output.permute(0, 1, 3, 2, 4)
        attention_output = attention_output.contiguous().reshape(
            batch_size, query_block_count, queries_per_block, channels
        )
        return attention_output

    def multi_lane_attention(
        self,
        x,
        history_key_value,
        query_block_ids,
        first_query_block,
    ):
        """All-Q x All-C attention with one physically shared history K/V.

        The score domain is [B, H, U*K, T]. A broadcastable strict block
        causal mask permits C_j iff j < query_block_id. Each Q's own K/V is
        appended as one additional score before the single joint softmax.
        """
        (
            batch_size,
            query_block_count,
            queries_per_block,
            channels,
            history_k,
            history_v,
        ) = self._validate_branched_inputs(
            x, history_key_value, first_query_block
        )
        if query_block_ids.shape != (query_block_count,):
            raise ValueError("multi-lane query_block_ids 形状不正确。")

        q, self_k, self_v = self._project_branched_qkv(x, query_block_ids)
        # Block-major flattening over the sampled K queries in each block.
        q = q.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            self.attn.n_head,
            query_block_count * queries_per_block,
            self.attn.head_width,
        )
        self_k = self_k.permute(0, 2, 1, 3, 4).reshape_as(q)
        self_v = self_v.permute(0, 2, 1, 3, 4).reshape_as(q)

        # history_k.transpose is a view. No repeat/expand/clone/contiguous is
        # applied to history K/V anywhere in this production attention path.
        history_scores = torch.matmul(q, history_k.transpose(-2, -1))
        history_length = history_k.size(2)
        history_scores = history_scores.view(
            batch_size,
            self.attn.n_head,
            query_block_count,
            queries_per_block,
            history_length,
        )
        history_indices = torch.arange(
            history_length, dtype=torch.long, device=x.device
        ).view(1, 1, 1, 1, history_length)
        allowed_history = history_indices < query_block_ids.view(
            1, 1, query_block_count, 1, 1
        )
        history_scores = history_scores.masked_fill(
            ~allowed_history, float("-inf")
        ).view(
            batch_size,
            self.attn.n_head,
            query_block_count * queries_per_block,
            history_length,
        )

        self_scores = (q * self_k).sum(dim=-1, keepdim=True)
        joint_scores = torch.cat((history_scores, self_scores), dim=-1)
        joint_weights = torch.softmax(
            joint_scores * self.attn.scale, dim=-1
        )
        history_output = torch.matmul(
            joint_weights[..., :history_length], history_v
        )
        self_output = joint_weights[..., history_length:] * self_v
        attention_output = history_output + self_output
        attention_output = attention_output.view(
            batch_size,
            self.attn.n_head,
            query_block_count,
            queries_per_block,
            self.attn.head_width,
        ).permute(0, 2, 3, 1, 4)
        return attention_output.contiguous().reshape(
            batch_size, query_block_count, queries_per_block, channels
        )

    def forward_branched(self, x, history_key_value, first_query_block):
        """Advance Q through the per-block correctness-reference attention."""
        attention_output = self.branched_attention_reference(
            x, history_key_value, first_query_block
        )
        return self._finish_branched_layer(x, attention_output)

    def forward_multi_lane(
        self, x, history_key_value, query_block_ids, first_query_block
    ):
        """Advance Q through production all-Q multi-lane attention."""
        attention_output = self.multi_lane_attention(
            x,
            history_key_value,
            query_block_ids=query_block_ids,
            first_query_block=first_query_block,
        )
        return self._finish_branched_layer(x, attention_output)


class GeometrizeGPT(nn.Module):
    """One-primitive discrete-embedding GPT with the v2 Multi-Lane body."""

    ARCHITECTURE_VERSION = (
        "text_prefix_primitive_embedding_512_multilane_v1"
    )

    def __init__(
        self,
        vocab_size=TOKEN_LAYOUT.vocab_size,
        d_model=1536,
        n_layer=36,
        n_head=24,
        ffn_hidden=4352,
        max_context_len=448,
        max_block_positions=448,
        norm_eps=1e-6,
        rope_base=10000.0,
        text_vocab_size=1500,
        max_text_tokens=224,
    ):
        super().__init__()
        if max_context_len <= 0 or max_block_positions <= 0:
            raise ValueError("上下文长度和 block 位置容量必须为正整数。")
        if ffn_hidden is None:
            # Parameter-matched SwiGLU width: ceil(8D/3) to a 64-wide GEMM tile.
            ffn_hidden = ((8 * d_model + 3 * 64 - 1) // (3 * 64)) * 64
        if ffn_hidden <= 0:
            raise ValueError("ffn_hidden 必须为正整数。")
        self.max_context_len = max_context_len
        self.max_block_positions = max_block_positions
        self.primitive_token_width = PRIMITIVE_TOKEN_WIDTH
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.ffn_hidden = ffn_hidden
        self.norm_eps = float(norm_eps)
        self.rope_base = float(rope_base)
        self.text_vocab_size = int(text_vocab_size)
        self.max_text_tokens = int(max_text_tokens)
        self.h = nn.ModuleList(
            [
                GPTBlock(
                    d_model,
                    n_head,
                    ffn_hidden,
                    norm_eps=norm_eps,
                    rope_base=rope_base,
                )
                for _ in range(n_layer)
            ]
        )
        self.final_norm = RMSNorm(d_model, eps=norm_eps)

        # 普通图元求和九个 lookup；两种结束事件各只使用自己的 lookup。
        self.wte = nn.Embedding(vocab_size, d_model)
        # Text IDs are independent of the geometric, output-tied vocabulary.
        self.text_wte = nn.Embedding(text_vocab_size, d_model, padding_idx=0)
        self.termination_head = nn.Linear(d_model, 2, bias=True)
        self.apply(self._init_weights)
        with torch.no_grad():
            self.text_wte.weight[0].zero_()
        self._compiled_multi_lane_forward = None
        # Runtime-only mask cache; no parameter or checkpoint-format change.
        self._fused_fixed_grid_masks = {}

    def encode_primitive_tokens(self, primitive_tokens):
        """Look up nine discrete fields and sum them into one model token."""
        if (
            primitive_tokens.ndim < 2
            or primitive_tokens.size(-1) != self.primitive_token_width
        ):
            raise ValueError(
                "primitive_tokens 末维必须为 "
                f"{self.primitive_token_width}，实际为 {tuple(primitive_tokens.shape)}。"
            )
        if primitive_tokens.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("primitive_tokens 必须是整型离散 token。")
        primitive_tokens = primitive_tokens.to(torch.long)
        embedded = self.wte(primitive_tokens).sum(dim=-2)
        marker_end = primitive_tokens[..., 0] == TOKEN_LAYOUT.true_end_token
        interrupt_end = primitive_tokens[..., 0] == TOKEN_LAYOUT.interrupt_end_token
        geometry_bos = primitive_tokens[..., 0] == TOKEN_LAYOUT.bos_token
        marker_end_embedding = self.wte.weight[TOKEN_LAYOUT.true_end_token]
        interrupt_end_embedding = self.wte.weight[TOKEN_LAYOUT.interrupt_end_token]
        embedded = torch.where(
            marker_end.unsqueeze(-1), marker_end_embedding, embedded
        )
        embedded = torch.where(
            interrupt_end.unsqueeze(-1), interrupt_end_embedding, embedded
        )
        embedded = torch.where(
            geometry_bos.unsqueeze(-1), self.wte.weight[TOKEN_LAYOUT.bos_token], embedded
        )
        # Embedding lookup is not autocast-eligible. Match the established v2
        # BF16 activation stream without changing the literal nine-vector sum.
        if embedded.is_cuda and torch.is_autocast_enabled():
            embedded = embedded.to(torch.get_autocast_dtype("cuda"))
        return embedded

    def encode_text_tokens(self, text_ids):
        if text_ids.ndim != 2 or not 1 <= text_ids.size(1) <= self.max_text_tokens:
            raise ValueError("文字 token 应为 [B,L]，且 L 不超过 max_text_tokens。")
        if text_ids.dtype != torch.long:
            text_ids = text_ids.long()
        embedded = self.text_wte(text_ids)
        if embedded.is_cuda and torch.is_autocast_enabled():
            embedded = embedded.to(torch.get_autocast_dtype("cuda"))
        return embedded

    def project_attribute_logits(self, hidden):
        """Tied projection over output-capable attribute rows only."""
        return F.linear(
            hidden, self.wte.weight[: TOKEN_LAYOUT.attribute_vocab_size]
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE_INIT", False):
                std *= (2 * len(self.h)) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_torch_compile(
        self,
        enabled,
        backend="inductor",
        mode="default",
        dynamic=False,
        fullgraph=False,
        fixed_grid=False,
    ):
        """Compile only the production training callable; state_dict stays unchanged."""
        self._compiled_multi_lane_forward = None
        self._compiled_fixed_grid_forward = None
        if enabled:
            target = self.forward_fixed_grid if fixed_grid else self.forward_multi_lane
            compiled = torch.compile(
                target,
                backend=backend,
                mode=mode,
                dynamic=dynamic,
                fullgraph=fullgraph,
            )
            if fixed_grid:
                self._compiled_fixed_grid_forward = compiled
            else:
                self._compiled_multi_lane_forward = compiled

    def training_forward_multi_lane(self, *args, **kwargs):
        forward = self._compiled_multi_lane_forward
        if forward is None:
            forward = self.forward_multi_lane
        return forward(*args, **kwargs)

    def training_forward_fixed_grid(self, *args, **kwargs):
        forward = getattr(self, "_compiled_fixed_grid_forward", None)
        if forward is None:
            forward = self.forward_fixed_grid
        return forward(*args, **kwargs)

    def forward_fixed_grid(
        self,
        text_ids,
        text_mask,
        geometry_tokens,
        query_tokens,
        query_fields,
    ):
        """Fixed-size text+geometry history and masked Multi-Lane query slots."""
        batch_size = text_ids.size(0)
        context = self.max_context_len
        if text_mask.shape != text_ids.shape or text_mask.dtype != torch.bool:
            raise ValueError("固定网格需要同形状的 bool text_mask。")
        if geometry_tokens.shape != (batch_size, context - 2, self.primitive_token_width):
            raise ValueError("固定网格的图元历史形状不正确。")
        if query_tokens.ndim != 4 or query_tokens.shape[:2] != (batch_size, context - 1):
            raise ValueError("固定网格的查询位置数必须等于总上下文长度减一。")
        lanes = query_tokens.size(2)
        if query_tokens.size(3) != self.primitive_token_width or query_fields.shape != (
            batch_size, context - 1, lanes
        ):
            raise ValueError("固定网格的查询字段形状不正确。")

        text_lengths = text_mask.sum(dim=1).long()
        text_hidden = self.encode_text_tokens(text_ids)
        geometry_hidden = self.encode_primitive_tokens(geometry_tokens)
        query_hidden = self.encode_primitive_tokens(query_tokens)

        history_position = torch.arange(context, device=text_ids.device).unsqueeze(0).expand(batch_size, -1)
        geometry_index = (history_position - text_lengths.unsqueeze(1)).clamp(0, context - 3)
        text_index = history_position.clamp_max(text_ids.size(1) - 1)
        text_at_position = torch.gather(
            text_hidden, 1, text_index.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        geometry_at_position = torch.gather(
            geometry_hidden, 1, geometry_index.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        history_hidden = torch.where(
            (history_position < text_lengths.unsqueeze(1)).unsqueeze(-1),
            text_at_position,
            geometry_at_position,
        )

        query_position = torch.arange(1, context, device=text_ids.device).unsqueeze(0).expand(batch_size, -1)
        query_position_ids = query_position.repeat_interleave(lanes, dim=1)
        joint_position_ids = torch.cat((history_position, query_position_ids), dim=1)
        rotary_cos_sin = self.h[0].attn.rope.cos_sin(joint_position_ids, history_hidden.dtype)
        allowed_history = history_position[:, None, None, :] < query_position_ids[:, None, :, None]

        mask_key = (str(text_ids.device), context, lanes)
        if mask_key not in self._fused_fixed_grid_masks:
            self._fused_fixed_grid_masks[mask_key] = build_fixed_grid_block_mask(
                (context - 1) * lanes, context, lanes, text_ids.device,
            )
        fused_block_mask = self._fused_fixed_grid_masks[mask_key]

        for block in self.h:
            history_hidden, query_hidden = block.forward_shared_history_multi_lane(
                history_hidden,
                query_hidden,
                rotary_cos_sin=rotary_cos_sin,
                allowed_history=allowed_history,
                history_attention_mask=None,
                fused_block_mask=fused_block_mask,
            )

        previous_hidden = history_hidden[:, :-1]
        selected_hidden = torch.where(
            (query_fields == 0).unsqueeze(-1),
            previous_hidden.unsqueeze(2).expand(-1, -1, lanes, -1),
            query_hidden,
        )
        attribute_logits = self.project_attribute_logits(self.final_norm(selected_hidden))
        termination_logits = self.termination_head(self.final_norm(previous_hidden))
        return attribute_logits, termination_logits

    def forward(
        self,
        primitive_tokens,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        return_hidden=False,
        return_intermediates=False,
        return_termination=False,
        text_ids=None,
        text_mask=None,
    ):
        """Predict global vocabulary logits from [B,T,9] primitive tokens."""
        if (
            primitive_tokens.ndim != 3
            or primitive_tokens.size(-1) != self.primitive_token_width
        ):
            raise ValueError(
                f"primitive_tokens 必须为 [B,T,{self.primitive_token_width}]，"
                f"实际收到 {tuple(primitive_tokens.shape)}。"
            )
        batch_size, sequence_length, _ = primitive_tokens.shape
        if past_key_values is not None and len(past_key_values) != len(self.h):
            raise ValueError(
                f"past_key_values 应有 {len(self.h)} 层，"
                f"实际收到 {len(past_key_values)} 层。"
            )
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(2)
        text_length = 0 if text_mask is None else text_mask.size(1)
        if text_ids is not None:
            if past_key_values is not None:
                raise ValueError("文字前缀只能在首次 forward 时写入 KV cache。")
            if text_mask is None or text_ids.shape != text_mask.shape:
                raise ValueError("text_ids 与 text_mask 必须具有相同形状。")
        if text_mask is not None and (
            text_mask.ndim != 2 or text_mask.size(0) != batch_size
            or text_mask.dtype != torch.bool
        ):
            raise ValueError("text_mask 应为 [B,L] bool。")
        total_length = past_length + sequence_length + (text_length if text_ids is not None else 0)
        if total_length > self.max_context_len:
            raise ValueError(
                f"缓存及输入长度 {total_length} 超过 "
                f"max_context_len={self.max_context_len}。"
            )

        if position_ids is None:
            position_ids = torch.arange(
                past_length + (text_length if text_ids is not None else 0),
                total_length,
                dtype=torch.long,
                device=primitive_tokens.device,
            ).unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape != (batch_size, sequence_length):
            raise ValueError(
                f"position_ids 应为 {(batch_size, sequence_length)}，"
                f"实际为 {tuple(position_ids.shape)}。"
            )
        elif text_mask is not None:
            # Callers express image positions relative to the image sequence.
            position_ids = position_ids + text_length
        # CPU 测试保留清晰错误；GPU 热路径的位置由 packer/state builder 构造，
        # 避免为范围检查引入逐 forward 的 device synchronization。
        if not position_ids.is_cuda and (
            bool(torch.any(position_ids < 0))
            or bool(torch.any(position_ids >= self.max_block_positions))
        ):
            raise ValueError("position_ids 超过 RoPE block 位置容量。")

        x = self.encode_primitive_tokens(primitive_tokens)
        if text_ids is not None:
            x = torch.cat((self.encode_text_tokens(text_ids), x), dim=1)
            text_positions = torch.arange(
                text_length, dtype=torch.long, device=x.device
            ).unsqueeze(0).expand(batch_size, -1)
            position_ids = torch.cat((text_positions, position_ids), dim=1)
        if text_mask is not None:
            query_length = x.size(1)
            key_length = past_length + query_length
            key_positions = torch.arange(key_length, device=x.device)
            query_positions = past_length + torch.arange(query_length, device=x.device)
            causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            key_valid = torch.cat((
                text_mask.to(x.device),
                torch.ones(batch_size, key_length - text_length, dtype=torch.bool, device=x.device),
            ), dim=1)
            attention_mask = causal.view(1, 1, query_length, key_length) & key_valid[:, None, None, :]
        present_key_values = []
        layer_hidden_states = []
        for layer_index, block in enumerate(self.h):
            layer_past = None if past_key_values is None else past_key_values[layer_index]
            if use_cache:
                x, layer_present = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=True,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                present_key_values.append(layer_present)
            else:
                x = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=False,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            if return_intermediates:
                layer_hidden_states.append(x)

        if text_ids is not None:
            x = x[:, text_length:]
            if return_intermediates:
                layer_hidden_states = [state[:, text_length:] for state in layer_hidden_states]
        final_hidden = self.final_norm(x)
        logits = self.project_attribute_logits(final_hidden)
        termination_logits = self.termination_head(final_hidden)
        if return_intermediates:
            if use_cache:
                raise ValueError("return_intermediates 与 use_cache 不能同时启用。")
            if return_termination:
                return logits, termination_logits, final_hidden, tuple(layer_hidden_states)
            return logits, final_hidden, tuple(layer_hidden_states)
        if return_hidden:
            if use_cache:
                raise ValueError("return_hidden 与 use_cache 不能同时启用。")
            if return_termination:
                return logits, termination_logits, final_hidden
            return logits, final_hidden
        if use_cache:
            if return_termination:
                return logits, termination_logits, tuple(present_key_values)
            return logits, tuple(present_key_values)
        if return_termination:
            return logits, termination_logits
        return logits

    def _forward_shared_history(
        self,
        full_tokens,
        query_tokens,
        query_block_ids,
        query_fields,
        first_query_block,
        attention_implementation,
        return_hidden=False,
        return_intermediates=False,
        return_termination=False,
    ):
        """Shared validation/embedding path for reference and production branches."""
        if (
            full_tokens.ndim != 3
            or full_tokens.size(-1) != self.primitive_token_width
        ):
            raise ValueError(
                f"full_tokens 必须为 [B,T,{self.primitive_token_width}]，"
                f"实际为 {tuple(full_tokens.shape)}。"
            )
        if (
            query_tokens.ndim != 4
            or not 1 <= query_tokens.size(2) <= VALUES_PER_BLOCK
            or query_tokens.size(-1) != self.primitive_token_width
        ):
            raise ValueError(
                f"query_tokens 必须为 [B,U,K,{self.primitive_token_width}]，"
                f"其中 1 <= K <= {VALUES_PER_BLOCK}，"
                f"实际为 {tuple(query_tokens.shape)}。"
            )
        batch_size, history_length, _ = full_tokens.shape
        query_batch, query_block_count, queries_per_block, _ = query_tokens.shape
        if query_batch != batch_size:
            raise ValueError("full_tokens 与 query_tokens 的 batch size 必须一致。")
        if query_fields.shape != (
            batch_size, query_block_count, queries_per_block
        ):
            raise ValueError("query_fields 必须与 query_tokens 的 [B,U,K] 一致。")
        if history_length > self.max_context_len:
            raise ValueError(
                f"完整历史长度 {history_length} 超过 max_context_len={self.max_context_len}。"
            )
        if query_block_ids.shape != (query_block_count,):
            raise ValueError(
                f"query_block_ids 应为 {(query_block_count,)}，"
                f"实际为 {tuple(query_block_ids.shape)}。"
            )
        expected_block_ids = torch.arange(
            first_query_block,
            first_query_block + query_block_count,
            dtype=torch.long,
            device=query_block_ids.device,
        )
        if not query_block_ids.is_cuda and not torch.equal(
            query_block_ids, expected_block_ids
        ):
            raise ValueError("query_block_ids 必须是连续递增的完整 block 位置。")
        if first_query_block + query_block_count > history_length:
            raise ValueError("query block 范围超过完整历史长度。")
        if history_length > self.max_block_positions:
            raise ValueError("完整历史超过 RoPE block 位置容量。")

        history_positions = torch.arange(
            history_length, dtype=torch.long, device=full_tokens.device
        ).unsqueeze(0).expand(batch_size, -1)
        history_hidden = self.encode_primitive_tokens(full_tokens)
        query_hidden = self.encode_primitive_tokens(query_tokens)

        layer_query_hidden_states = []
        for block in self.h:
            # Returned K/V were projected from this layer's C input and retain
            # their complete autograd graph. Q cannot affect the C update.
            history_hidden, history_key_value = block(
                history_hidden,
                use_cache=True,
                position_ids=history_positions,
            )
            if attention_implementation == "per_block_reference":
                query_hidden = block.forward_branched(
                    query_hidden,
                    history_key_value=history_key_value,
                    first_query_block=first_query_block,
                )
            elif attention_implementation == "multi_lane":
                query_hidden = block.forward_multi_lane(
                    query_hidden,
                    history_key_value=history_key_value,
                    query_block_ids=query_block_ids,
                    first_query_block=first_query_block,
                )
            else:
                raise ValueError(
                    f"未知 branched attention 实现：{attention_implementation}。"
                )
            if return_intermediates:
                layer_query_hidden_states.append(query_hidden)

        previous_history = history_hidden[:, query_block_ids - 1].unsqueeze(2)
        previous_history = previous_history.expand(-1, -1, queries_per_block, -1)
        selected_hidden = torch.where(
            (query_fields == 0).unsqueeze(-1),
            previous_history,
            query_hidden,
        )
        final_hidden = self.final_norm(selected_hidden)
        logits = self.project_attribute_logits(final_hidden)
        previous_final_hidden = self.final_norm(
            history_hidden[:, query_block_ids - 1]
        )
        termination_logits = self.termination_head(previous_final_hidden)
        if return_intermediates:
            if return_termination:
                return logits, termination_logits, final_hidden, tuple(layer_query_hidden_states)
            return logits, final_hidden, tuple(layer_query_hidden_states)
        if return_hidden:
            if return_termination:
                return logits, termination_logits, final_hidden
            return logits, final_hidden
        if return_termination:
            return logits, termination_logits
        return logits

    def _forward_shared_history_optimized(
        self,
        full_tokens,
        query_tokens,
        query_block_ids,
        query_fields,
        first_query_block,
        return_hidden=False,
        return_intermediates=False,
        return_termination=False,
        text_ids=None,
        text_mask=None,
    ):
        """Joint C/Q production path with shared RoPE and block mask metadata."""
        if (
            full_tokens.ndim != 3
            or full_tokens.size(-1) != self.primitive_token_width
        ):
            raise ValueError(
                f"full_tokens 必须为 [B,T,{self.primitive_token_width}]，"
                f"实际为 {tuple(full_tokens.shape)}。"
            )
        if (
            query_tokens.ndim != 4
            or not 1 <= query_tokens.size(2) <= VALUES_PER_BLOCK
            or query_tokens.size(-1) != self.primitive_token_width
        ):
            raise ValueError(
                f"query_tokens 必须为 [B,U,K,{self.primitive_token_width}]，"
                f"实际为 {tuple(query_tokens.shape)}。"
            )
        batch_size, history_length, _ = full_tokens.shape
        query_batch, query_block_count, queries_per_block, _ = query_tokens.shape
        if query_batch != batch_size:
            raise ValueError("full_tokens 与 query_tokens 的 batch size 必须一致。")
        if query_fields.shape != (
            batch_size, query_block_count, queries_per_block
        ):
            raise ValueError("query_fields 必须与 query_tokens 的 [B,U,K] 一致。")
        text_lengths = None
        if text_ids is not None:
            if text_mask is None or text_mask.shape != text_ids.shape or text_mask.dtype != torch.bool:
                raise ValueError("训练文字前缀需要同形状的 bool text_mask。")
            if text_ids.size(0) != batch_size:
                raise ValueError("文字与图元的 batch size 不一致。")
            text_lengths = text_mask.sum(dim=1).to(torch.long)
            if not text_mask.is_cuda and bool(torch.any(text_lengths < 1)):
                raise ValueError("每个文字前缀至少包含 BOS。")
        elif history_length > self.max_context_len:
            raise ValueError("完整历史超过 max_context_len。")
        if query_block_ids.shape != (query_block_count,):
            raise ValueError("query_block_ids 形状不正确。")
        if first_query_block + query_block_count > history_length:
            raise ValueError("query block 范围超过完整历史长度。")
        if text_ids is None and history_length > self.max_block_positions:
            raise ValueError("完整历史超过 RoPE block 位置容量。")

        image_hidden = self.encode_primitive_tokens(full_tokens)
        query_hidden = self.encode_primitive_tokens(query_tokens)

        # Pack each sample as [its own non-padding text][its own geometry].
        # This keeps every physical attention row within the same total budget
        # without letting another sample's longer caption truncate its image.
        history_attention_mask = None
        if text_ids is not None:
            lengths = text_lengths.tolist()
            packed_length = min(
                self.max_context_len, history_length + max(lengths)
            )
            if history_length + min(lengths) == self.max_context_len:
                packed_length = self.max_context_len
            if packed_length > self.max_block_positions:
                raise ValueError("打包后的位置超过 RoPE 容量。")
            text_hidden = self.encode_text_tokens(text_ids)
            history_hidden = image_hidden.new_zeros((batch_size, packed_length, self.d_model))
            for sample_index, length in enumerate(lengths):
                if length >= packed_length:
                    raise ValueError("文字前缀没有留下图元 BOS 位置。")
                take = min(history_length, packed_length - length)
                history_hidden[sample_index, :length] = text_hidden[sample_index, :length]
                history_hidden[sample_index, length:length + take] = image_hidden[sample_index, :take]
            query_position_ids = (
                query_block_ids.unsqueeze(0) + text_lengths.unsqueeze(1)
            ).repeat_interleave(queries_per_block, dim=1)
            query_position_ids = query_position_ids.clamp_max(packed_length - 1)
            joint_position_ids = torch.cat((
                torch.arange(packed_length, device=full_tokens.device).unsqueeze(0).expand(batch_size, -1),
                query_position_ids,
            ), dim=1)
            rotary_cos_sin = self.h[0].attn.rope.cos_sin(
                joint_position_ids, history_hidden.dtype
            )
            history_indices = torch.arange(packed_length, device=full_tokens.device)
            allowed_history = (
                history_indices.view(1, 1, 1, -1)
                < query_position_ids[:, None, :, None]
            )
            previous_indices = (
                query_block_ids.unsqueeze(0) + text_lengths.unsqueeze(1) - 1
            ).clamp_max(packed_length - 1)
        else:
            history_hidden = image_hidden
            query_position_ids = query_block_ids.repeat_interleave(queries_per_block)
            joint_position_ids = torch.cat((
                torch.arange(history_length, device=full_tokens.device), query_position_ids
            ))
            rotary_cos_sin = tuple(
                value.unsqueeze(0) for value in self.h[0].attn.rope.cos_sin(
                    joint_position_ids, history_hidden.dtype
                )
            )
            history_indices = torch.arange(history_length, device=full_tokens.device)
            allowed_history = (
                history_indices.unsqueeze(0) < query_position_ids.unsqueeze(1)
            ).view(1, 1, query_block_count * queries_per_block, history_length)
            previous_indices = (query_block_ids - 1).unsqueeze(0).expand(batch_size, -1)

        layer_query_hidden_states = []
        for block in self.h:
            history_hidden, query_hidden = block.forward_shared_history_multi_lane(
                history_hidden,
                query_hidden,
                rotary_cos_sin=rotary_cos_sin,
                allowed_history=allowed_history,
                history_attention_mask=history_attention_mask,
            )
            if return_intermediates:
                layer_query_hidden_states.append(query_hidden)

        previous_history = torch.gather(
            history_hidden, 1, previous_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        ).unsqueeze(2)
        previous_history = previous_history.expand(-1, -1, queries_per_block, -1)
        selected_hidden = torch.where(
            (query_fields == 0).unsqueeze(-1),
            previous_history,
            query_hidden,
        )
        final_hidden = self.final_norm(selected_hidden)
        logits = self.project_attribute_logits(final_hidden)
        previous_final_hidden = self.final_norm(
            torch.gather(history_hidden, 1, previous_indices.unsqueeze(-1).expand(-1, -1, self.d_model))
        )
        termination_logits = self.termination_head(previous_final_hidden)
        if return_intermediates:
            if return_termination:
                return logits, termination_logits, final_hidden, tuple(layer_query_hidden_states)
            return logits, final_hidden, tuple(layer_query_hidden_states)
        if return_hidden:
            if return_termination:
                return logits, termination_logits, final_hidden
            return logits, final_hidden
        if return_termination:
            return logits, termination_logits
        return logits

    def forward_branched(
        self,
        full_tokens,
        query_tokens,
        query_block_ids,
        query_fields,
        first_query_block,
        return_hidden=False,
        return_intermediates=False,
        return_termination=False,
    ):
        """Run the previous per-block implementation as a correctness reference."""
        return self._forward_shared_history(
            full_tokens,
            query_tokens,
            query_block_ids,
            query_fields,
            first_query_block,
            attention_implementation="per_block_reference",
            return_hidden=return_hidden,
            return_intermediates=return_intermediates,
            return_termination=return_termination,
        )

    def forward_multi_lane_reference(
        self,
        full_tokens,
        query_tokens,
        query_block_ids,
        query_fields,
        first_query_block,
        return_hidden=False,
        return_intermediates=False,
        return_termination=False,
    ):
        """Unoptimized Multi-Lane path retained as a numerical reference."""
        return self._forward_shared_history(
            full_tokens,
            query_tokens,
            query_block_ids,
            query_fields,
            first_query_block,
            attention_implementation="multi_lane",
            return_hidden=return_hidden,
            return_intermediates=return_intermediates,
            return_termination=return_termination,
        )

    def forward_multi_lane(
        self,
        full_tokens,
        query_tokens,
        query_block_ids,
        query_fields,
        first_query_block,
        return_hidden=False,
        return_intermediates=False,
        return_termination=False,
        text_ids=None,
        text_mask=None,
    ):
        """Production path with joint pointwise/linear C+Q execution."""
        return self._forward_shared_history_optimized(
            full_tokens,
            query_tokens,
            query_block_ids,
            query_fields,
            first_query_block,
            return_hidden=return_hidden,
            return_intermediates=return_intermediates,
            return_termination=return_termination,
            text_ids=text_ids,
            text_mask=text_mask,
        )
