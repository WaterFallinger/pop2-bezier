"""Strict v3 checkpoint loading and semantic epoch-157 migration."""

import math

import torch

from model_gpt import GeometrizeGPT
from token_layout import TOKEN_LAYOUT, TOKEN_LAYOUT_256


NEW_CHECKPOINT_KIND = "primitive_embedding_true_interrupt_end_head_v3"
LEGACY_CHECKPOINT_KIND = "primitive_embedding_marker_end_v2"


def strip_module_prefix(state_dict):
    result = state_dict
    for prefix in ("module.", "_orig_mod."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix):]: value for key, value in result.items()}
    return result


def _raw_state_for(checkpoint, state_key):
    try:
        return strip_module_prefix(checkpoint[state_key])
    except KeyError as error:
        raise KeyError(f"checkpoint 缺少 {state_key}。") from error


def checkpoint_metadata(config):
    return {
        "architecture_version": GeometrizeGPT.ARCHITECTURE_VERSION,
        "block_primitives": config.BLOCK_PRIMITIVES,
        "prefix_steps": config.PREFIX_STEPS,
        "prediction_steps": config.PREDICTION_STEPS,
        "context_steps": config.CONTEXT_STEPS,
        "context_blocks": config.CONTEXT_BLOCKS,
        "max_block_positions": config.MAX_BLOCK_POSITIONS,
        "model_d_model": config.MODEL_D_MODEL,
        "model_n_layer": config.MODEL_N_LAYER,
        "model_n_head": config.MODEL_N_HEAD,
        "model_ffn_hidden": config.MODEL_FFN_HIDDEN,
        "model_norm_eps": config.MODEL_NORM_EPS,
        "model_rope_base": config.MODEL_ROPE_BASE,
        "position_encoding": "rope_text_prefix_then_primitive_id",
        "input_representation": "independent_text_embedding_prefix_then_geometry_bos_then_sum_9_or_control",
        "compression_ratio": "text_1:1_plus_geometry_9:1",
        "empty_token_ids": list(TOKEN_LAYOUT.empty_token_ids),
        "attribute_vocab_size": TOKEN_LAYOUT.attribute_vocab_size,
        "vocab_size": TOKEN_LAYOUT.vocab_size,
        "text_vocab_size": config.TEXT_VOCAB_SIZE,
        "text_max_tokens": config.TEXT_MAX_TOKENS,
        "total_context_tokens": config.TOTAL_CONTEXT_TOKENS,
        "text_dropout_probability": config.TEXT_DROPOUT_PROBABILITY,
        "text_sampling_seed": config.TEXT_SAMPLING_SEED,
        "caption_tokenizer": "caption_spm_1500_full_100k",
        "token_layout_version": TOKEN_LAYOUT.version,
        "token_layout": TOKEN_LAYOUT.to_metadata(),
        "termination_head": {
            "classes": ["continue", "true_end"],
            "loss": "unweighted_cross_entropy",
            "loss_weight": 1.0,
        },
        "canvas_size": config.CANVAS_SIZE,
        "max_shape_size": config.MAX_SHAPE_SIZE,
        "data_token_layout_version": config.DATA_TOKEN_LAYOUT_VERSION,
    }


def checkpoint_format(checkpoint, config, state_key="model_state_dict"):
    state_dict = _raw_state_for(checkpoint, state_key)
    required = (
        "h.0.attn.q_norm.weight",
        "h.0.attn.k_norm.weight",
        "h.0.mlp.gate_up.weight",
        "h.0.mlp.down.weight",
        "final_norm.weight",
        "wte.weight",
        "text_wte.weight",
        "termination_head.weight",
        "termination_head.bias",
    )
    for key in required:
        if key not in state_dict:
            raise RuntimeError(f"checkpoint 缺少 v3 参数 {key}。")
    if "lm_head.weight" in state_dict:
        raise RuntimeError("v3 checkpoint 不应保存独立 lm_head.weight。")
    if any(key.startswith("block_encoder.") for key in state_dict):
        raise RuntimeError("检测到旧 108D block_encoder；checkpoint 架构不兼容。")

    for name, expected in checkpoint_metadata(config).items():
        actual = checkpoint.get(name)
        if actual != expected:
            raise RuntimeError(
                f"checkpoint 元数据 {name}={actual!r}，当前要求 {expected!r}；"
                "不允许把旧架构当作续训断点。"
            )
    if tuple(state_dict["wte.weight"].shape) != (
        TOKEN_LAYOUT.vocab_size,
        config.MODEL_D_MODEL,
    ):
        raise RuntimeError("checkpoint 的输入 embedding 形状不兼容。")
    if tuple(state_dict["text_wte.weight"].shape) != (
        config.TEXT_VOCAB_SIZE, config.MODEL_D_MODEL
    ):
        raise RuntimeError("checkpoint 的文字 embedding 形状不兼容。")
    if tuple(state_dict["termination_head.weight"].shape) != (
        2,
        config.MODEL_D_MODEL,
    ):
        raise RuntimeError("checkpoint 的终止分类头形状不兼容。")
    return NEW_CHECKPOINT_KIND


def legacy_checkpoint_format(checkpoint, config, state_key="model_state_dict"):
    state_dict = _raw_state_for(checkpoint, state_key)
    required = (
        "h.0.attn.q_norm.weight",
        "h.0.attn.k_norm.weight",
        "h.0.mlp.gate_up.weight",
        "h.0.mlp.down.weight",
        "final_norm.weight",
        "wte.weight",
        "lm_head.weight",
    )
    for key in required:
        if key not in state_dict:
            raise RuntimeError(f"epoch-157 源 checkpoint 缺少参数 {key}。")
    if "termination_head.weight" in state_dict:
        raise RuntimeError("迁移入口收到的已经是带终止头的新 checkpoint。")
    expected_shape = (TOKEN_LAYOUT_256.vocab_size, config.MODEL_D_MODEL)
    if tuple(state_dict["wte.weight"].shape) != expected_shape:
        raise RuntimeError(
            f"epoch-157 embedding 形状 {tuple(state_dict['wte.weight'].shape)} "
            f"不等于 {expected_shape}。"
        )
    if tuple(state_dict["lm_head.weight"].shape) != expected_shape:
        raise RuntimeError("epoch-157 lm_head 形状不兼容。")
    if not torch.equal(state_dict["wte.weight"], state_dict["lm_head.weight"]):
        raise RuntimeError("epoch-157 的 tied wte/lm_head 内容不一致，拒绝迁移。")
    for name, expected in (
        ("model_d_model", config.MODEL_D_MODEL),
        ("model_n_layer", config.MODEL_N_LAYER),
        ("model_n_head", config.MODEL_N_HEAD),
        ("model_ffn_hidden", config.MODEL_FFN_HIDDEN),
        ("model_norm_eps", config.MODEL_NORM_EPS),
        ("model_rope_base", config.MODEL_ROPE_BASE),
    ):
        if name in checkpoint and checkpoint[name] != expected:
            raise RuntimeError(
                f"epoch-157 元数据 {name}={checkpoint[name]!r}，要求 {expected!r}。"
            )
    return LEGACY_CHECKPOINT_KIND


def _copy_field_rows(target, source, field_name):
    old_slice = TOKEN_LAYOUT_256.field_slice(field_name)
    new_slice = TOKEN_LAYOUT.field_slice(field_name)
    row_count = old_slice.stop - old_slice.start
    target[new_slice.start:new_slice.start + row_count].copy_(source[old_slice])


def migrate_legacy_state(model, checkpoint, config, state_key="model_state_dict"):
    """Create and strictly load a v3 state while preserving every matching row."""
    legacy_checkpoint_format(checkpoint, config, state_key)
    source = _raw_state_for(checkpoint, state_key)
    initialized = model.state_dict()
    migrated = {}
    excluded_new = {"wte.weight", "termination_head.weight", "termination_head.bias"}
    expected_body = set(initialized) - excluded_new
    source_body = set(source) - {"wte.weight", "lm_head.weight"}
    if source_body != expected_body:
        missing = sorted(expected_body - source_body)
        unexpected = sorted(source_body - expected_body)
        raise RuntimeError(
            f"Transformer 主体键不完全一致；missing={missing[:8]} "
            f"unexpected={unexpected[:8]}。"
        )
    for key in sorted(expected_body):
        if source[key].shape != initialized[key].shape:
            raise RuntimeError(
                f"主体参数 {key} 形状不一致: {tuple(source[key].shape)} vs "
                f"{tuple(initialized[key].shape)}。"
            )
        migrated[key] = source[key]

    source_embedding = source["wte.weight"]
    new_embedding = initialized["wte.weight"].clone()
    for field_name in (
        "x", "y", "angle", "width", "height", "shape", "red", "green", "blue"
    ):
        _copy_field_rows(new_embedding, source_embedding, field_name)
    new_embedding[
        TOKEN_LAYOUT.special_offset:TOKEN_LAYOUT.special_offset + TOKEN_LAYOUT_256.special_bins
    ].copy_(
        source_embedding[
            TOKEN_LAYOUT_256.special_offset:
            TOKEN_LAYOUT_256.special_offset + TOKEN_LAYOUT_256.special_bins
        ]
    )
    for field_name in TOKEN_LAYOUT.empty_fields:
        new_embedding[TOKEN_LAYOUT.empty_token(field_name)].copy_(
            source_embedding[TOKEN_LAYOUT_256.empty_token(field_name)]
        )
    old_end = source_embedding[TOKEN_LAYOUT_256.marker_end_token]
    new_embedding[TOKEN_LAYOUT.true_end_token].copy_(old_end)
    new_embedding[TOKEN_LAYOUT.interrupt_end_token].copy_(old_end)
    migrated["wte.weight"] = new_embedding

    source_output = source["lm_head.weight"]
    termination_weight = initialized["termination_head.weight"].clone()
    termination_bias = initialized["termination_head.bias"].clone()
    termination_weight[0].copy_(source_output[TOKEN_LAYOUT_256.field_slice("x")].mean(dim=0))
    termination_weight[1].copy_(source_output[TOKEN_LAYOUT_256.marker_end_token])
    termination_bias.zero_()
    termination_bias[0] = math.log(TOKEN_LAYOUT_256.x_bins)
    migrated["termination_head.weight"] = termination_weight
    migrated["termination_head.bias"] = termination_bias

    model.load_state_dict(migrated, strict=True)
    return migrated


def load_checkpoint_state(model, checkpoint, config, state_key="model_state_dict"):
    state_dict = _raw_state_for(checkpoint, state_key)
    kind = checkpoint_format(checkpoint, config, state_key)
    model.load_state_dict(state_dict, strict=True)
    return kind


def load_image_only_initialization(model, checkpoint, config):
    """Keep every old geometric weight; initialize only the new text embedding."""
    if checkpoint.get("architecture_version") != (
        "primitive_embedding_512_true_interrupt_end_head_rope_rms_swiglu_multilane_v3"
    ):
        raise RuntimeError("初始化权重不是预期的 v3 图元模型。")
    for name in ("model_d_model", "model_n_layer", "model_n_head", "model_ffn_hidden",
                 "model_norm_eps", "model_rope_base", "context_steps", "vocab_size"):
        expected = {
            "model_d_model": config.MODEL_D_MODEL,
            "model_n_layer": config.MODEL_N_LAYER,
            "model_n_head": config.MODEL_N_HEAD,
            "model_ffn_hidden": config.MODEL_FFN_HIDDEN,
            "model_norm_eps": config.MODEL_NORM_EPS,
            "model_rope_base": config.MODEL_ROPE_BASE,
            "context_steps": config.MAX_IMAGE_STEPS,
            "vocab_size": config.MODEL_VOCAB_SIZE - 1,
        }[name]
        if checkpoint.get(name) != expected:
            raise RuntimeError(f"图元初始化权重 {name} 与当前模型不兼容。")
    state = dict(_raw_state_for(checkpoint, "model_state_dict"))
    if "text_wte.weight" in state:
        raise RuntimeError("图元初始化权重意外包含文字 embedding。")
    if tuple(state["wte.weight"].shape) != (config.MODEL_VOCAB_SIZE - 1, config.MODEL_D_MODEL):
        raise RuntimeError("旧图元 embedding 尺寸与新增 BOS 前的词表不一致。")
    expanded_embedding = model.state_dict()["wte.weight"].clone()
    expanded_embedding[:-1].copy_(state["wte.weight"])
    state["wte.weight"] = expanded_embedding
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != {"text_wte.weight"} or result.unexpected_keys:
        raise RuntimeError(
            f"初始化参数不完全匹配：missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


# ---------------------------------------------------------------------------
# Bezier edge migration: 7069-vocab (9 attributes) -> 8097-vocab (13 fields).
# 旧布局为 geometrize_512_v2（9 属性，7069）；新布局为 v3（+edge0..3，8097）。
# ---------------------------------------------------------------------------

_LEGACY_ATTRIBUTE_VOCAB = 7054
_LEGACY_SPECIAL_RANGE = (7054, 7058)
_LEGACY_EMPTY_RANGE = (7058, 7066)
_LEGACY_CONTROL_RANGE = (7066, 7069)  # true_end, interrupt_end, BOS
_LEGACY_VOCAB_SIZE = 7069
_LEGACY_EMPTY_COUNT = 8


def load_bezier_initialization(model, checkpoint, state_key="model_state_dict"):
    """把 7069 词表的 POP2 权重迁移到带贝塞尔边字段的 8097 词表。

    映射（下标均为 token 行号）：
      新[0:7054]        ← 旧[0:7054]      9 个原属性，偏移完全一致
      新[7054:8078]     ← 0               edge0..3（4 条边的控制点），零初始化
      新[8078:8082]     ← 旧[7054:7058]   special（pad / region_marker 等）
      新[8082:8090]     ← 旧[7058:7066]   8 个字段专属 empty
      新[8090:8094]     ← 0               edge0..3 的 empty，零初始化
      新[8094:8097]     ← 旧[7066:7069]   true_end / interrupt_end / BOS

    edge 段零初始化保证第 0 步模型行为与原模型一致（9 项求和不变，新增项为 0）。
    其余参数（35 层、ln_f、终止头、text_wte）形状不变，直接加载。
    """
    layout = TOKEN_LAYOUT
    if layout.edge_bins <= 0:
        raise RuntimeError("当前 token_layout 不含贝塞尔边字段，无法执行 bezier 初始化。")

    state = dict(_raw_state_for(checkpoint, state_key))
    if "wte.weight" not in state:
        raise KeyError("checkpoint 缺少 wte.weight。")
    source = state["wte.weight"]
    target = model.state_dict()["wte.weight"]
    if tuple(source.shape) != (_LEGACY_VOCAB_SIZE, target.shape[1]):
        raise RuntimeError(
            f"bezier 初始化要求旧词表 {_LEGACY_VOCAB_SIZE} 行，实际 {tuple(source.shape)}。"
        )
    if target.shape[0] != layout.vocab_size:
        raise RuntimeError(
            f"目标 wte 行数 {target.shape[0]} 与 token_layout.vocab_size "
            f"{layout.vocab_size} 不一致。"
        )

    expanded = torch.zeros_like(target)
    expanded[: _LEGACY_ATTRIBUTE_VOCAB] = source[: _LEGACY_ATTRIBUTE_VOCAB]
    special_slice = slice(layout.special_offset, layout.special_offset + layout.special_bins)
    expanded[special_slice] = source[_LEGACY_SPECIAL_RANGE[0]: _LEGACY_SPECIAL_RANGE[1]]
    empty_slice = slice(layout.empty_offset, layout.empty_offset + _LEGACY_EMPTY_COUNT)
    expanded[empty_slice] = source[_LEGACY_EMPTY_RANGE[0]: _LEGACY_EMPTY_RANGE[1]]
    expanded[layout.marker_end_token] = source[_LEGACY_CONTROL_RANGE[0]]
    expanded[layout.interrupt_end_token] = source[_LEGACY_CONTROL_RANGE[0] + 1]
    expanded[layout.bos_token] = source[_LEGACY_CONTROL_RANGE[0] + 2]

    state["wte.weight"] = expanded
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"bezier 初始化存在未匹配参数：{sorted(result.unexpected_keys)}")
    if result.missing_keys:
        raise RuntimeError(f"bezier 初始化缺少参数：{sorted(result.missing_keys)}")

    # 自检：迁移后的行必须与映射表一致。
    with torch.no_grad():
        if not torch.equal(expanded[: _LEGACY_ATTRIBUTE_VOCAB], source[: _LEGACY_ATTRIBUTE_VOCAB]):
            raise RuntimeError("bezier 初始化：属性段复制校验失败。")
        if torch.count_nonzero(expanded[layout.edge_offset: layout.special_offset]) != 0:
            raise RuntimeError("bezier 初始化：edge 段应为零。")
        if not torch.equal(expanded[special_slice], source[_LEGACY_SPECIAL_RANGE[0]: _LEGACY_SPECIAL_RANGE[1]]):
            raise RuntimeError("bezier 初始化：special 段复制校验失败。")
    return state
