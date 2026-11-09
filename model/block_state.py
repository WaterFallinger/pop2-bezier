"""Discrete one-primitive token construction shared by training and inference."""

from dataclasses import dataclass

import torch

from token_layout import TOKEN_LAYOUT


ATTRIBUTE_NAMES = (
    "x", "y", "angle", "width", "height", "shape", "red", "green", "blue",
    "edge0", "edge1", "edge2", "edge3",
)
ATTRIBUTES_PER_PRIMITIVE = len(ATTRIBUTE_NAMES)
PRIMITIVES_PER_BLOCK = 1
VALUES_PER_BLOCK = ATTRIBUTES_PER_PRIMITIVE
PRIMITIVE_TOKEN_WIDTH = ATTRIBUTES_PER_PRIMITIVE


def _field_offset(field_name):
    return TOKEN_LAYOUT.field_slice(field_name).start


def _field_bins(field_name):
    span = TOKEN_LAYOUT.field_slice(field_name)
    return span.stop - span.start


ATTRIBUTE_OFFSETS = tuple(_field_offset(name) for name in ATTRIBUTE_NAMES)
ATTRIBUTE_BINS = tuple(_field_bins(name) for name in ATTRIBUTE_NAMES)
EMPTY_TOKEN_IDS = tuple(
    TOKEN_LAYOUT.empty_token(field_name) for field_name in ATTRIBUTE_NAMES[1:]
)
# The leading value is never used as an x empty token: x queries are replaced
# by the preceding full primitive, while every p>=1 query already knows x.
PARTIAL_FILL_TOKEN_IDS = (TOKEN_LAYOUT.pad_token, *EMPTY_TOKEN_IDS)


@dataclass(frozen=True)
class BranchedBlockBatch:
    """Fixed-width sampled queries over a shared full primitive history.

    full_tokens is [B,T,9] and query_tokens is [B,U,K,9].
    Query groups correspond to supervised primitive positions. A query for
    field zero is a compute placeholder: its logits are selected from the
    preceding complete history position by the model. Fields one through
    eight contain the known prefix plus field-specific empty tokens.
    """

    full_tokens: torch.Tensor
    query_tokens: torch.Tensor
    query_block_ids: torch.Tensor
    query_partial_ids: torch.Tensor
    query_valid_mask: torch.Tensor
    targets: torch.Tensor
    query_fields: torch.Tensor
    first_query_block: int
    marker_end_mask: torch.Tensor
    interrupt_end_mask: torch.Tensor
    termination_targets: torch.Tensor
    termination_supervised_mask: torch.Tensor
    padding_mask: torch.Tensor


def _as_primitive_tokens(input_ids):
    if input_ids.ndim == 2:
        if input_ids.size(1) % ATTRIBUTES_PER_PRIMITIVE != 0:
            raise ValueError("token 序列长度必须能被每图元 9 个属性整除。")
        return input_ids.reshape(
            input_ids.size(0), -1, ATTRIBUTES_PER_PRIMITIVE
        )
    if input_ids.ndim == 3 and input_ids.size(-1) == ATTRIBUTES_PER_PRIMITIVE:
        return input_ids
    raise ValueError(
        "input_ids 必须为 [B, steps*9] 或 [B, steps, 9]，"
        f"实际收到 {tuple(input_ids.shape)}。"
    )


def canonical_marker_end_tokens(batch_size, device, dtype=torch.long):
    """Return the fixed-width carrier whose embedding is one marker-end vector."""
    tokens = torch.full(
        (int(batch_size), ATTRIBUTES_PER_PRIMITIVE),
        TOKEN_LAYOUT.pad_token,
        dtype=dtype,
        device=device,
    )
    tokens[:, 0] = TOKEN_LAYOUT.marker_end_token
    return tokens


def canonical_interrupt_end_tokens(batch_size, device, dtype=torch.long):
    """Return the fixed-width carrier for an externally injected interruption."""
    tokens = torch.full(
        (int(batch_size), ATTRIBUTES_PER_PRIMITIVE),
        TOKEN_LAYOUT.pad_token,
        dtype=dtype,
        device=device,
    )
    tokens[:, 0] = TOKEN_LAYOUT.interrupt_end_token
    return tokens


def canonical_bos_tokens(batch_size, device, dtype=torch.long):
    """An input-only geometry BOS carrier, never a predicted attribute."""
    tokens = torch.full(
        (int(batch_size), ATTRIBUTES_PER_PRIMITIVE),
        TOKEN_LAYOUT.pad_token, dtype=dtype, device=device,
    )
    tokens[:, 0] = TOKEN_LAYOUT.bos_token
    return tokens


def bos_steps_from_tokens(primitive_tokens):
    primitive_tokens = _as_primitive_tokens(primitive_tokens)
    components = primitive_tokens == TOKEN_LAYOUT.bos_token
    canonical = components[..., 0] & torch.all(
        primitive_tokens[..., 1:] == TOKEN_LAYOUT.pad_token, dim=-1
    )
    if bool(torch.any(components.any(dim=-1) & ~canonical)):
        raise ValueError("图元 BOS 必须是首位 BOS 加八个 pad token。")
    return canonical


def marker_begin_steps_from_tokens(primitive_tokens):
    """Return [B,T] marker-begin flags and reject malformed RGB triplets."""
    primitive_tokens = _as_primitive_tokens(primitive_tokens)
    rgb = primitive_tokens[..., 6:9]
    marker_components = rgb == TOKEN_LAYOUT.marker_begin_token
    any_marker = marker_components.any(dim=-1)
    all_marker = marker_components.all(dim=-1)
    if bool(torch.any(any_marker & ~all_marker)):
        raise ValueError("marker_begin 的三个 RGB token 必须同时出现。")
    begin_shape_token = TOKEN_LAYOUT.shape_offset + 3
    if bool(torch.any(all_marker & (primitive_tokens[..., 5] != begin_shape_token))):
        raise ValueError("marker_begin 必须使用 shape_type=2 的 shape token。")
    return all_marker


def marker_end_steps_from_tokens(primitive_tokens):
    """Return [B,T] standalone marker-end flags and reject partial carriers."""
    primitive_tokens = _as_primitive_tokens(primitive_tokens)
    end_components = primitive_tokens == TOKEN_LAYOUT.marker_end_token
    any_end = end_components.any(dim=-1)
    canonical = end_components[..., 0] & torch.all(
        primitive_tokens[..., 1:] == TOKEN_LAYOUT.pad_token, dim=-1
    )
    if bool(torch.any(any_end & ~canonical)):
        raise ValueError("marker_end 必须是首位 end token 加八个 pad token。")
    return canonical


def interrupt_end_steps_from_tokens(primitive_tokens):
    """Return [B,T] externally injected interruption flags."""
    primitive_tokens = _as_primitive_tokens(primitive_tokens)
    components = primitive_tokens == TOKEN_LAYOUT.interrupt_end_token
    any_interrupt = components.any(dim=-1)
    canonical = components[..., 0] & torch.all(
        primitive_tokens[..., 1:] == TOKEN_LAYOUT.pad_token, dim=-1
    )
    if bool(torch.any(any_interrupt & ~canonical)):
        raise ValueError("interrupt_end 必须是首位 interrupt token 加八个 pad token。")
    return canonical


def padding_steps_from_tokens(primitive_tokens):
    """Return [B,T] padding flags and require padding to be a right suffix."""
    primitive_tokens = _as_primitive_tokens(primitive_tokens)
    padding = torch.all(primitive_tokens == TOKEN_LAYOUT.pad_token, dim=-1)
    marker_end = marker_end_steps_from_tokens(primitive_tokens)
    interrupt_end = interrupt_end_steps_from_tokens(primitive_tokens)
    bos = bos_steps_from_tokens(primitive_tokens)
    contains_pad = torch.any(primitive_tokens == TOKEN_LAYOUT.pad_token, dim=-1)
    if bool(torch.any(contains_pad & ~(padding | marker_end | interrupt_end | bos))):
        raise ValueError("完整图元不能混入局部 pad token。")
    seen_padding = padding.to(torch.int32).cumsum(dim=-1) > 0
    if bool(torch.any(seen_padding & ~padding)):
        raise ValueError("padding 必须是序列的连续右侧后缀。")
    return padding


def marker_steps_from_tokens(primitive_tokens):
    """Backward-compatible combined marker-begin/marker-end flags."""
    return (
        marker_begin_steps_from_tokens(primitive_tokens)
        | marker_end_steps_from_tokens(primitive_tokens)
        | interrupt_end_steps_from_tokens(primitive_tokens)
    )


def primitive_tokens_to_full_inputs(input_ids):
    """Validate and return complete discrete primitive records [B,T,9]."""
    primitive_tokens = _as_primitive_tokens(input_ids)
    marker_begin_steps_from_tokens(primitive_tokens)
    marker_end_steps_from_tokens(primitive_tokens)
    interrupt_end_steps_from_tokens(primitive_tokens)
    padding_steps_from_tokens(primitive_tokens)
    return primitive_tokens


def _sample_query_fields(all_query_valid, rand_k, sampling_seed):
    batch_size, query_count, field_count = all_query_valid.shape
    if rand_k == field_count:
        return torch.arange(
            field_count, dtype=torch.long, device=all_query_valid.device
        ).view(1, 1, -1).expand(batch_size, query_count, -1)

    weights = all_query_valid.reshape(-1, field_count).to(torch.float32)
    empty_rows = weights.sum(dim=-1) == 0
    safe_weights = weights.clone()
    safe_weights[empty_rows, 0] = 1.0
    generator = torch.Generator(device=all_query_valid.device)
    generator.manual_seed(int(sampling_seed) & ((1 << 63) - 1))
    return torch.multinomial(
        safe_weights,
        num_samples=rand_k,
        replacement=True,
        generator=generator,
    ).view(batch_size, query_count, rand_k)


def build_branched_block_batch(
    input_ids,
    prefix_steps,
    rand_k=VALUES_PER_BLOCK,
    sampling_seed=0,
    sampling_mode="random",
):
    """Build complete primitive history and sampled discrete partial queries."""
    if not isinstance(rand_k, int) or not 1 <= rand_k <= VALUES_PER_BLOCK:
        raise ValueError(
            f"rand_k 必须是 [1, {VALUES_PER_BLOCK}] 内的整数，收到 {rand_k!r}。"
        )
    if sampling_mode != "random":
        raise ValueError(
            f"当前仅支持 sampling_mode='random'（均匀有放回），收到 {sampling_mode!r}。"
        )

    primitive_tokens = primitive_tokens_to_full_inputs(input_ids)
    _, step_count, _ = primitive_tokens.shape
    if not 1 <= prefix_steps < step_count:
        raise ValueError(
            f"prefix_steps 必须位于 [1, {step_count})，收到 {prefix_steps}。"
        )
    bos_steps = bos_steps_from_tokens(primitive_tokens)
    if not bool(torch.all(bos_steps[:, 0])) or bool(torch.any(bos_steps[:, 1:])):
        raise ValueError("训练序列必须以唯一的图元 BOS 开始。")

    marker_begin_steps = marker_begin_steps_from_tokens(primitive_tokens)
    marker_end_steps = marker_end_steps_from_tokens(primitive_tokens)
    interrupt_end_steps = interrupt_end_steps_from_tokens(primitive_tokens)
    padding_steps = padding_steps_from_tokens(primitive_tokens)
    next_is_begin = torch.cat(
        (
            marker_begin_steps[:, 1:],
            torch.zeros_like(marker_begin_steps[:, :1]),
        ),
        dim=-1,
    )
    non_terminal_interrupts = interrupt_end_steps.clone()
    non_terminal_interrupts[:, -1] = False
    next_is_padding = torch.cat(
        (padding_steps[:, 1:], torch.ones_like(padding_steps[:, :1])), dim=-1
    )
    if bool(torch.any(non_terminal_interrupts & ~(next_is_begin | next_is_padding))):
        raise ValueError("每个 interrupt_end 后必须立即跟一个 marker_begin。")
    event_delta = (
        marker_begin_steps.to(torch.int16)
        - marker_end_steps.to(torch.int16)
        - interrupt_end_steps.to(torch.int16)
    )
    active_depth = event_delta.cumsum(dim=-1)
    if bool(torch.any((active_depth < 0) | (active_depth > 1))):
        raise ValueError("限制块 begin/end 状态机不合法。")
    active_before = (active_depth - event_delta) > 0
    query_block_ids = torch.arange(
        prefix_steps, step_count, dtype=torch.long, device=input_ids.device
    )
    selected_tokens = primitive_tokens[:, query_block_ids]
    selected_begins = marker_begin_steps[:, query_block_ids]
    selected_ends = marker_end_steps[:, query_block_ids]
    selected_interrupts = interrupt_end_steps[:, query_block_ids]
    selected_padding = padding_steps[:, query_block_ids]
    selected_active_before = active_before[:, query_block_ids]
    ordinary_steps = ~(
        selected_begins | selected_ends | selected_interrupts | selected_padding
    )
    all_query_valid = ordinary_steps.unsqueeze(-1).expand(
        -1, -1, VALUES_PER_BLOCK
    )
    query_partial_ids = _sample_query_fields(
        all_query_valid, rand_k, sampling_seed
    )
    query_valid_mask = ordinary_steps.unsqueeze(-1).expand(-1, -1, rand_k)
    termination_supervised = selected_active_before & (
        ordinary_steps | selected_ends
    )
    termination_targets = selected_ends.to(torch.long)

    field_positions = torch.arange(
        VALUES_PER_BLOCK, dtype=torch.long, device=input_ids.device
    ).view(1, 1, 1, -1)
    known = field_positions < query_partial_ids.unsqueeze(-1)
    empty_template = torch.tensor(
        PARTIAL_FILL_TOKEN_IDS,
        dtype=primitive_tokens.dtype,
        device=input_ids.device,
    ).view(1, 1, 1, -1)
    query_tokens = torch.where(
        known,
        selected_tokens.unsqueeze(2),
        empty_template,
    )

    # x_j is predicted by C_(j-1). Keep a valid discrete placeholder in the
    # fixed-width query tensor; the model replaces these lanes with history
    # hidden states before the tied output head.
    previous_tokens = primitive_tokens[:, query_block_ids - 1].unsqueeze(2)
    predicts_x = query_partial_ids == 0
    query_tokens = torch.where(
        predicts_x.unsqueeze(-1), previous_tokens, query_tokens
    )

    targets = torch.gather(
        selected_tokens, dim=-1, index=query_partial_ids
    )
    targets = targets.masked_fill(~query_valid_mask, -100)

    return BranchedBlockBatch(
        full_tokens=primitive_tokens,
        query_tokens=query_tokens,
        query_block_ids=query_block_ids,
        query_partial_ids=query_partial_ids,
        query_valid_mask=query_valid_mask,
        targets=targets,
        query_fields=query_partial_ids,
        first_query_block=prefix_steps,
        marker_end_mask=selected_ends,
        interrupt_end_mask=selected_interrupts,
        termination_targets=termination_targets,
        termination_supervised_mask=termination_supervised,
        padding_mask=selected_padding,
    )


def field_token_slice(field_index):
    return TOKEN_LAYOUT.field_slice(ATTRIBUTE_NAMES[int(field_index)])


class AutoregressivePrimitiveState:
    """Inference state for one complete or partially generated primitive."""

    def __init__(self, batch_size, device):
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.completed_primitives = []
        self.partial_tokens = None
        self.active_attribute_count = 0

    @property
    def total_primitives(self):
        return len(self.completed_primitives)

    @property
    def current_position(self):
        return self.total_primitives

    @property
    def next_field_index(self):
        return self.active_attribute_count

    def current_partial_tokens(self):
        if self.partial_tokens is None or self.active_attribute_count == 0:
            raise RuntimeError("当前没有已开始生成的 partial 图元。")
        return self.partial_tokens

    def _validate_field_token(self, token_ids, field_index, allow_marker_end=True):
        if token_ids.shape != (self.batch_size,):
            raise ValueError(
                f"token_ids 应为 ({self.batch_size},)，实际为 {tuple(token_ids.shape)}。"
            )
        offset = ATTRIBUTE_OFFSETS[field_index]
        bins = ATTRIBUTE_BINS[field_index]
        valid = (token_ids >= offset) & (token_ids < offset + bins)
        if field_index == 0 and allow_marker_end:
            valid = valid | (token_ids == TOKEN_LAYOUT.marker_end_token)
        # GPU 推理从已经按字段裁剪过的候选中采样。这里若逐属性读取
        # valid 会强制设备同步；CPU 调用仍保留严格诊断。
        if not token_ids.is_cuda and bool(torch.any(~valid)):
            raise ValueError(
                f"推理字段 {ATTRIBUTE_NAMES[field_index]} 收到范围外 token。"
            )

    def append_token(self, token_ids, allow_marker_end=True):
        field_index = self.next_field_index
        if not 0 <= field_index < ATTRIBUTES_PER_PRIMITIVE:
            raise RuntimeError("当前图元属性计数无效。")
        self._validate_field_token(token_ids, field_index, allow_marker_end)

        if field_index == 0 and allow_marker_end:
            marker_end = token_ids == TOKEN_LAYOUT.marker_end_token
            if bool(torch.any(marker_end)):
                if not bool(torch.all(marker_end)):
                    raise ValueError(
                        "同一 AutoregressivePrimitiveState 不允许普通 x 与 "
                        "marker_end 分叉；请为各样本使用独立状态。"
                    )
                completed = canonical_marker_end_tokens(
                    self.batch_size, self.device, dtype=token_ids.dtype
                )
                self.completed_primitives.append(completed)
                return completed
        if field_index == 0:
            self.partial_tokens = torch.tensor(
                PARTIAL_FILL_TOKEN_IDS,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0).expand(self.batch_size, -1).clone()
        self.partial_tokens[:, field_index] = token_ids
        self.active_attribute_count += 1
        if self.active_attribute_count < ATTRIBUTES_PER_PRIMITIVE:
            return None

        completed = self.partial_tokens.clone()
        self.completed_primitives.append(completed)
        self.partial_tokens = None
        self.active_attribute_count = 0
        return completed

    def append_marker_begin(self, marker_tokens):
        marker_tokens = _as_primitive_tokens(marker_tokens)
        if marker_tokens.shape != (self.batch_size, 1, ATTRIBUTES_PER_PRIMITIVE):
            raise ValueError("marker_tokens 必须恰好包含每个 batch 一个完整 marker 图元。")
        if self.active_attribute_count != 0:
            raise RuntimeError("marker 只能在完整图元边界原子注入。")
        if not bool(torch.all(marker_begin_steps_from_tokens(marker_tokens))):
            raise ValueError("append_marker_begin 只能接收 marker_begin。")
        completed = marker_tokens[:, 0].clone()
        self.completed_primitives.append(completed)
        return completed

    def append_true_end(self):
        if self.active_attribute_count != 0:
            raise RuntimeError("true_end 只能在完整图元边界追加。")
        completed = canonical_marker_end_tokens(
            self.batch_size, self.device, dtype=torch.long
        )
        self.completed_primitives.append(completed)
        return completed

    def append_interrupt_end(self):
        if self.active_attribute_count != 0:
            raise RuntimeError("interrupt_end 只能在完整图元边界追加。")
        completed = canonical_interrupt_end_tokens(
            self.batch_size, self.device, dtype=torch.long
        )
        self.completed_primitives.append(completed)
        return completed

    def append_marker(self, marker_tokens):
        """Compatibility alias; injected markers are always marker_begin."""
        return self.append_marker_begin(marker_tokens)

    def append_primitive_tokens(self, primitive_tokens):
        primitive_tokens = primitive_tokens_to_full_inputs(primitive_tokens)
        if primitive_tokens.size(0) != self.batch_size:
            raise ValueError("primitive token batch 大小与状态对象不一致。")
        if self.active_attribute_count != 0:
            raise RuntimeError("不能在 partial 图元中间追加完整图元。")
        marker_steps = marker_steps_from_tokens(primitive_tokens)
        padding_steps = padding_steps_from_tokens(primitive_tokens)
        if bool(torch.any(padding_steps)):
            raise ValueError("推理状态不能追加 padding 图元。")
        for step_index in range(primitive_tokens.size(1)):
            step_markers = marker_steps[:, step_index]
            if bool(torch.any(step_markers)) and not bool(torch.all(step_markers)):
                raise ValueError("同一推理 batch 的 marker 位置必须一致。")
            self.completed_primitives.append(
                primitive_tokens[:, step_index].clone()
            )
