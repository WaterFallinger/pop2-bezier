import os
import math
import json
import glob
import re
import gc
import threading
from contextvars import ContextVar
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from numbers import Integral, Real
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import sentencepiece as spm
import matplotlib.pyplot as plt
from torch.nn import functional as F

from block_state import (
    ATTRIBUTE_BINS,
    ATTRIBUTE_NAMES,
    ATTRIBUTES_PER_PRIMITIVE,
    PARTIAL_FILL_TOKEN_IDS,
    canonical_interrupt_end_tokens,
    canonical_marker_end_tokens,
    canonical_bos_tokens,
    field_token_slice,
)
from checkpoint_compat import checkpoint_format, load_checkpoint_state
from config import TrainConfig
from model_gpt import GeometrizeGPT
from token_layout import (
    ANGLE_BINS_PER_DEGREE,
    SIZE_BINS_PER_PIXEL,
    TOKEN_LAYOUT,
    XY_BINS_PER_PIXEL,
)

VISUALIZATION_DIR = Path(__file__).resolve().parent / "visualizations"


# ==========================================
# 🔴 独立测试配置区
# ==========================================
config = TrainConfig()
config.validate_sequence_config()
config.validate_model_config()


def get_checkpoint_sort_key(file_path):
    path=Path(file_path)
    if path.name=='model.pt' and (path.parent/'READY.json').is_file():
        metadata=json.loads((path.parent/'READY.json').read_text())
        batch=int(metadata.get('batch_idx',-1))
        return int(metadata['epoch']),float('inf') if batch<0 else batch
    match = re.search(r"epoch_(\d+)(?:_batch_(\d+))?\.pt$", file_path)
    if not match:
        return -1, -1
    epoch = int(match.group(1))
    batch = int(match.group(2)) if match.group(2) else float("inf")
    return epoch, batch


checkpoint_candidates = glob.glob(
    os.path.join(config.CKPT_DIR, "geometrize_gpt_pretrain_epoch_*.pt")
)
_checkpoint_roots=[Path(config.CKPT_DIR)]
if os.environ.get('PAINTER_CHECKPOINT_ROOT'):_checkpoint_roots.append(Path(os.environ['PAINTER_CHECKPOINT_ROOT']))
_archive_root=Path(os.environ.get('PAINTER_CHECKPOINT_ROOT', config.CKPT_DIR))
if _archive_root.is_dir():_checkpoint_roots.append(_archive_root)
for _root in _checkpoint_roots:
    for _folder in _root.glob('geometrize_gpt_pretrain_epoch_*.ckpt'):
        if not (_folder/'READY.json').is_file():continue
        _manifest=json.loads((_folder/'READY.json').read_text())
        if _manifest.get('model')!='model.pt':continue
        if all((_folder/n).is_file() and (_folder/n).stat().st_size==size
               for n,size in _manifest['file_sizes'].items()):checkpoint_candidates.append(str(_folder/'model.pt'))
MODEL_CKPT_PATH = (
    max(checkpoint_candidates, key=get_checkpoint_sort_key)
    if checkpoint_candidates
    else None
)
if os.environ.get('PAINTER_MODEL_CKPT'):MODEL_CKPT_PATH=os.environ['PAINTER_MODEL_CKPT']
_validation_path = os.path.join(config.DATA_DIR, "validation")
TEST_DATA_PATH = os.environ.get(
    "PAINTER_VALIDATION_DATA",
    _validation_path if os.path.isdir(_validation_path) else os.path.join(config.DATA_DIR, "v1"),
)

NUM_TESTS = 32            # 测试多少张不同的原图
TARGET_SHAPE_STEPS = config.CONTEXT_STEPS

# 仅提供独立的图元 BOS；不把真实图元作为推理前缀。
START_STEPS = config.PREFIX_STEPS

# 候选数：每个候选独立维护状态/KV cache，允许 end 时间不同。
INFERENCE_BATCH_SIZE = 4
# 每张设备一次并发推进的独立序列数；显存不足时可调低。
SEQUENCES_PER_DEVICE_BATCH = 16  # 每卡16条；每条保留实际文字位置与独立图元预算。
# 启用后自动探测所有可用 CUDA/ROCm GPU，并将候选生成任务分摊到各卡。
# 只有一张可用 GPU 时自动退化为单卡；没有可用 GPU 时退化为 CPU。
ENABLE_MULTI_GPU_INFERENCE = True
# GPU 推理默认使用训练时相同的 BF16（不支持时 FP16）autocast 路径。
ENABLE_INFERENCE_AUTOCAST = True
# 正式采样方式："renyi" 为默认；改成 "legacy" 可恢复下方动态温度/top-k 曲线。
SAMPLING_MODE = "renyi"
# Rényi 熵阶数，直接修改这一行即可；必须是大于 1 的有限数。
RENYI_ALPHA = 4.0
ENABLE_OPTIMIZED_INFERENCE = os.environ.get('PAINTER_OPTIMIZED_INFERENCE','1') == '1'
OPTIMIZED_INFERENCE_COMPILE = os.environ.get('PAINTER_INFERENCE_COMPILE','1') == '1'
OPTIMIZED_INFERENCE_GRAPH = os.environ.get('PAINTER_INFERENCE_GRAPH','1') == '1'
OPTIMIZED_COMPILE_THREADS = 4
_OPTIMIZED_SAMPLER = ContextVar('painter_optimized_sampler',default=None)
_LEGAL_SUPPORT_MASK = ContextVar('painter_legal_support_mask',default=None)
ENABLE_REGION_LEGAL_ENTROPY = os.environ.get('PAINTER_REGION_LEGAL_ENTROPY','1')=='1'
_ENGINE_CREATION_LOCK = threading.Lock()
if SAMPLING_MODE not in {"renyi", "legacy"}:
    raise ValueError("SAMPLING_MODE 必须是 renyi 或 legacy。")
if not math.isfinite(RENYI_ALPHA) or RENYI_ALPHA < 1:
    raise ValueError("RENYI_ALPHA 必须是大于等于 1 的有限数；1使用Shannon熵极限。")

# 以下曲线仅在 SAMPLING_MODE="legacy" 时生效。
# 采样参数曲线：[完整序列中的零基图元序号, 参数值]。
# GT 前缀、marker_begin 与 marker_end 都占图元序号；相邻特征点之间
# 线性插值，曲线范围外保持首/末端点值。
# TEMPERATURE_SCHEDULES = {
#     "x": [[0, 0.3]],
#     "y": [[0, 0.7]],
#     "angle": [[0, 0.3]],
#     "width": [[0, 0.3]],
#     "height": [[0, 0.5]],
#     "shape": [[0, 0.4]],
#     "red": [[0, 0.3]],
#     "green": [[0, 0.6]],
#     "blue": [[0, 0.6]],
# }
TEMPERATURE_SCHEDULES = {
    "x": [[0, 1.05],[85, 1.55],[86, 4]],
    "y": [[0, 0.85],[85, 1.05],[86, 2]],
    "angle": [[0, 1.05],[85, 1.15],[86, 5]],
    "width": [[0, 1.05],[85, 1.15],[86, 4.5]],
    "height": [[0, 1.05],[85, 1.4],[86, 3]],
    "shape": [[0, 1.05],[85, 1.3],[86, 1]],
    "red": [[0, 0.55],[85, 1],[86, 1]],
    "green": [[0, 0.75],[85, 1],[86, 1]],
    "blue": [[0, 0.85],[85, 1],[86, 1]],
}
# Top-k 的插值结果按 .5 向上四舍五入；终止头始终直接 argmax。
TOP_K_SCHEDULES = {
    "x": [[0, 80],[85, 150]],
    "y": [[0, 100]],
    "angle": [[0, 40]],
    "width": [[0, 90],[85, 150]],
    "height": [[0, 90]],
    "shape": [[0, 1]],
    "red": [[0, 32],[85, 64]],
    "green": [[0, 64]],
    "blue": [[0, 64]],
}

# 三阶段控制器：85 个 global，16 个 coarse 块，64 个 fine 块。
GLOBAL_SHAPE_STEPS = 86  # 背景现在也由模型预测，占第一个全局图元位置。
STAGE_SPECS = {
    "coarse": {"grid_side": 4, "region_count": 16, "shape_cap": 20},
    "fine": {"grid_side": 8, "region_count": 64, "shape_cap": 10},
}
MAX_REGIONS = sum(spec["region_count"] for spec in STAGE_SPECS.values())
MARKER_THETA = 0.0
MARKER_SHAPE_TYPE = 2
INTERVENTION_POLICY = os.environ.get("PAINTER_INTERVENTION_POLICY", "mixed").lower()
PERTURBED_INTERRUPT_PROBABILITY = 0.05
# CSV 按行块流式读取；一旦拿到 NUM_TESTS 张完整图像便停止后续磁盘读取。
CSV_READ_CHUNK_ROWS = 2_048
# 与数据生成器 WGSL 中的 SHAPE_ALPHA 保持一致。
SHAPE_ALPHA = np.float32(0.5)

# ==========================================
# 工具函数区 (包含渲染钳位防崩机制)
# ==========================================
def _validate_schedule_mapping(schedules, parameter_name, *, top_k=False):
    expected_fields = set(ATTRIBUTE_NAMES)
    actual_fields = set(schedules)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ValueError(
            f"{parameter_name} 必须且只能配置 9 个属性；"
            f"缺少={missing}，多余={extra}。"
        )

    for field_index, field_name in enumerate(ATTRIBUTE_NAMES):
        points = schedules[field_name]
        if not isinstance(points, (list, tuple)) or not points:
            raise ValueError(f"{parameter_name}.{field_name} 至少需要一个特征点。")

        previous_t = None
        for point_index, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(
                    f"{parameter_name}.{field_name}[{point_index}] 必须是 [t, n]。"
                )
            t, value = point
            if isinstance(t, bool) or not isinstance(t, Integral) or t < 0:
                raise ValueError(
                    f"{parameter_name}.{field_name} 的 t 必须是非负整数，收到 {t!r}。"
                )
            if previous_t is not None and t <= previous_t:
                raise ValueError(
                    f"{parameter_name}.{field_name} 的 t 必须严格递增。"
                )
            previous_t = t

            if top_k:
                max_k = ATTRIBUTE_BINS[field_index]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Integral)
                    or not 1 <= value <= max_k
                ):
                    raise ValueError(
                        f"{parameter_name}.{field_name} 必须是 [1, {max_k}] 内的整数，"
                        f"收到 {value!r}。"
                    )
            elif (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(
                    f"{parameter_name}.{field_name} 必须是有限正数，收到 {value!r}。"
                )


def validate_sampling_schedules(
    temperature_schedules=None,
    top_k_schedules=None,
):
    if temperature_schedules is None:
        temperature_schedules = TEMPERATURE_SCHEDULES
    if top_k_schedules is None:
        top_k_schedules = TOP_K_SCHEDULES
    _validate_schedule_mapping(temperature_schedules, "TEMPERATURE_SCHEDULES")
    _validate_schedule_mapping(top_k_schedules, "TOP_K_SCHEDULES", top_k=True)


def interpolate_schedule(points, t):
    """Linearly interpolate a validated schedule and clamp outside its range."""
    if t <= points[0][0]:
        return float(points[0][1])
    if t >= points[-1][0]:
        return float(points[-1][1])

    for (left_t, left_value), (right_t, right_value) in zip(points, points[1:]):
        if t <= right_t:
            ratio = (t - left_t) / (right_t - left_t)
            return float(left_value + ratio * (right_value - left_value))
    raise RuntimeError("无法定位采样参数插值区间。")


def sampling_parameters(field_index, primitive_index):
    if SAMPLING_MODE == "renyi":
        # Rényi 模式从温度 1 的完整合法候选分布计算熵，不使用人工曲线。
        return 1.0, None
    field_name = ATTRIBUTE_NAMES[int(field_index)]
    temperature = interpolate_schedule(
        TEMPERATURE_SCHEDULES[field_name], primitive_index
    )
    interpolated_top_k = interpolate_schedule(
        TOP_K_SCHEDULES[field_name], primitive_index
    )
    top_k = int(math.floor(interpolated_top_k + 0.5))
    return temperature, top_k


def sample_renyi_candidate_token(field_logits, candidate_ids):
    """At T=1, uniformly sample the top round(exp(H_alpha)) legal candidates."""
    optimized_sampler=_OPTIMIZED_SAMPLER.get()
    legal_mask=_LEGAL_SUPPORT_MASK.get()
    if optimized_sampler is not None:return optimized_sampler(field_logits,candidate_ids,legal_mask)
    logits = field_logits.to(torch.float32)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Rényi 采样收到非有限 logits。")
    if legal_mask is not None and not bool(legal_mask.any(-1).all()):raise ValueError('Empty legal candidate set')
    from optimized_decode import renyi_entropy_and_support
    _,support = renyi_entropy_and_support(logits,RENYI_ALPHA,legal_mask)
    selected=logits if legal_mask is None else logits.masked_fill(~legal_mask,float('-inf'))
    _, top_positions = torch.topk(selected, k=int(support.max().item()), dim=-1)
    uniform_positions = torch.floor(
        torch.rand(support.shape, device=field_logits.device) * support
    ).to(torch.long).unsqueeze(-1)
    chosen_positions = top_positions.gather(1, uniform_positions).squeeze(-1)
    return candidate_ids[chosen_positions]


def sample_candidate_token(field_logits, candidate_ids, temperature, top_k):
    """Use the configured Rényi sampler or the retained legacy sampler."""
    if SAMPLING_MODE == "renyi":
        return sample_renyi_candidate_token(field_logits, candidate_ids)
    candidate_count = field_logits.size(-1)
    effective_top_k = min(top_k, candidate_count)
    if effective_top_k < candidate_count:
        selected_logits, selected_positions = torch.topk(
            field_logits, k=effective_top_k, dim=-1
        )
        probs = F.softmax(selected_logits / temperature, dim=-1)
        sampled_positions = torch.multinomial(probs, num_samples=1)
        sampled_positions = torch.gather(
            selected_positions, dim=-1, index=sampled_positions
        )
    else:
        probs = F.softmax(field_logits / temperature, dim=-1)
        sampled_positions = torch.multinomial(probs, num_samples=1)
    return candidate_ids[sampled_positions[:, 0]]


def _probe_cuda_device(device_id):
    """Run a real allocation/kernel/synchronization probe on one visible GPU."""
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    probe = torch.ones(1, device=device)
    probe.add_(1)
    torch.cuda.synchronize(device)
    del probe
    torch.cuda.empty_cache()
    return device


def discover_usable_cuda_devices():
    """Return every visible CUDA/ROCm device that passes an execution probe."""
    if not torch.cuda.is_available():
        return []

    usable_devices = []
    for device_id in range(torch.cuda.device_count()):
        try:
            device = _probe_cuda_device(device_id)
            device_name = torch.cuda.get_device_name(device_id)
            usable_devices.append(device)
            print(f"  ✅ GPU {device_id} 可用: {device_name}")
        except Exception as error:
            print(f"  ⚠️ 跳过 GPU {device_id}：探测失败 ({error})")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return usable_devices


def resolve_inference_devices(enable_multi_gpu=None):
    if enable_multi_gpu is None:
        enable_multi_gpu = ENABLE_MULTI_GPU_INFERENCE
    if not isinstance(enable_multi_gpu, bool):
        raise ValueError("ENABLE_MULTI_GPU_INFERENCE 必须是 bool。")
    if not enable_multi_gpu:
        return [
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        ]

    print("🔎 正在自动探测可用于推理的 GPU...")
    devices = discover_usable_cuda_devices()
    if len(devices) > 1:
        print(f"🚀 已启用自动多卡推理，共使用 {len(devices)} 张 GPU。")
        return devices
    if len(devices) == 1:
        print("ℹ️ 仅检测到一张可用 GPU，自动退化为单卡推理。")
        return devices
    print("⚠️ 没有检测到可用 GPU，自动退化为 CPU 推理。")
    return [torch.device("cpu")]


def iter_complete_image_groups(csv_path, dtype_dict, chunk_rows):
    """按 image_name 的连续分组流式读取 CSV，避免加载整份文件。"""
    yield from iter_complete_image_groups_from_parts(
        [csv_path], dtype_dict, chunk_rows
    )


def iter_complete_image_groups_from_parts(csv_paths, dtype_dict, chunk_rows):
    """跨 CSV 分片保留末尾图像，避免在文件边界截断一张图。"""
    carry = None
    for csv_path in csv_paths:
        for chunk in pd.read_csv(
            csv_path, dtype=dtype_dict, engine='c', chunksize=chunk_rows
        ):
            if carry is not None:
                chunk = pd.concat([carry, chunk], ignore_index=True)
            if chunk.empty:
                continue

            # 最后一组可能跨 chunk 或 CSV 文件；等读到下一组后才交付。
            last_image_name = chunk['image_name'].iloc[-1]
            complete_rows = chunk[chunk['image_name'].ne(last_image_name)]
            carry = chunk[chunk['image_name'].eq(last_image_name)]

            if not complete_rows.empty:
                yield from complete_rows.groupby('image_name', sort=False)

    if carry is not None and not carry.empty:
        yield from carry.groupby('image_name', sort=False)


def encode_primitive_tokens(
    cx, cy, width, height, shape_type, theta, red, green, blue,
    is_region_marker=False,
    edge_values=(0.0, 0.0, 0.0, 0.0),
):
    """Encode one primitive; a marker only changes its three RGB fields.

    edge_values are normalized [-1, 1] bezier edge offsets (0 = straight edge).
    """
    theta_deg = (theta * 180.0 / math.pi) % 180.0
    if theta_deg >= 90.0:
        theta_deg -= 90.0
        width, height = height, width
    tokens = [
        int(np.clip(cx * XY_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.x_bins - 1))
        + TOKEN_LAYOUT.x_offset,
        int(np.clip(cy * XY_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.y_bins - 1))
        + TOKEN_LAYOUT.y_offset,
        int(np.clip(theta_deg * ANGLE_BINS_PER_DEGREE, 0, TOKEN_LAYOUT.angle_bins - 1))
        + TOKEN_LAYOUT.angle_offset,
        int(np.clip(width * SIZE_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.width_bins - 1))
        + TOKEN_LAYOUT.width_offset,
        int(np.clip(height * SIZE_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.height_bins - 1))
        + TOKEN_LAYOUT.height_offset,
        int(np.clip(shape_type + 1, 0, TOKEN_LAYOUT.shape_bins - 1))
        + TOKEN_LAYOUT.shape_offset,
        int(np.clip(red // 2, 0, TOKEN_LAYOUT.color_bins - 1))
        + TOKEN_LAYOUT.red_offset,
        int(np.clip(green // 2, 0, TOKEN_LAYOUT.color_bins - 1))
        + TOKEN_LAYOUT.green_offset,
        int(np.clip(blue // 2, 0, TOKEN_LAYOUT.color_bins - 1))
        + TOKEN_LAYOUT.blue_offset,
    ]
    for index in range(TOKEN_LAYOUT.edge_count):
        value = float(edge_values[index]) if index < len(edge_values) else 0.0
        bin_value = int(round((value + 1.0) * 0.5 * (TOKEN_LAYOUT.edge_bins - 1)))
        bin_value = int(np.clip(bin_value, 0, TOKEN_LAYOUT.edge_bins - 1))
        tokens.append(
            bin_value + TOKEN_LAYOUT.edge_offset + index * TOKEN_LAYOUT.edge_bins
        )
    if is_region_marker:
        tokens[6:9] = [TOKEN_LAYOUT.region_marker_token] * 3
    return tokens


def encode_image_group(group_df):
    """Encode one committed three-stage sequence."""
    seq = []
    for row in group_df.iloc[:TARGET_SHAPE_STEPS].itertuples(index=False):
        if int(row.is_region_marker) == 1 and int(row.shape_type) == 3:
            seq.extend(
                canonical_marker_end_tokens(1, "cpu")[0].tolist()
            )
            continue
        if int(row.is_region_marker) == 1 and int(row.shape_type) == 4:
            seq.extend(canonical_interrupt_end_tokens(1, "cpu")[0].tolist())
            continue
        seq.extend(
            encode_primitive_tokens(
                row.cx, row.cy, row.w, row.h, row.shape_type, row.theta,
                row.r, row.g, row.b,
                is_region_marker=int(row.is_region_marker) == 1,
            )
        )
    return torch.tensor(seq, dtype=torch.long)


def load_test_samples(data_path, num_samples):
    """从合并版 CSV 或旧版 sequence.csv 树中流式读取前 N 张图。"""
    print(f"📥 正在提取前 {num_samples} 张三阶段验证序列...")
    dtype_dict = {
        'image_name': 'string', 'cx': 'float32', 'cy': 'float32', 'w': 'float32', 'h': 'float32',
        'shape_type': 'int8', 'theta': 'float32', 'r': 'uint8', 'g': 'uint8', 'b': 'uint8',
        'is_region_marker': 'uint8', 'stage': 'string', 'region_id': 'int16',
    }
    root = Path(data_path)
    if root.is_file():
        csv_paths = [root]
        merged_parts = True
    elif root.is_dir():
        numbered_parts = sorted(
            (
                (int(match.group(1)), path)
                for path in root.iterdir()
                if path.is_file()
                and (match := re.fullmatch(r"data_part_(\d+)\.csv", path.name))
            ),
            key=lambda item: item[0],
        )
        if numbered_parts:
            indices = [index for index, _ in numbered_parts]
            if indices != list(range(1, len(indices) + 1)):
                raise ValueError(f"CSV 分片编号不连续: {indices}")
            csv_paths = [path for _, path in numbered_parts]
            merged_parts = True
        else:
            csv_paths = sorted(root.glob("**/sequence.csv"))
            merged_parts = False
    else:
        raise FileNotFoundError(f"测试数据路径不存在: {root}")

    if not csv_paths:
        raise FileNotFoundError(f"{root} 中没有 data_part_N.csv 或 sequence.csv")

    samples = []
    if merged_parts:
        groups = iter_complete_image_groups_from_parts(
            csv_paths, dtype_dict, CSV_READ_CHUNK_ROWS
        )
    else:
        groups = (
            (csv_path.stem, pd.read_csv(csv_path, dtype=dtype_dict, engine="c"))
            for csv_path in csv_paths
        )
    for _, group_df in groups:
        if len(group_df) >= START_STEPS:
            encoded = encode_image_group(group_df)
            samples.append(torch.cat((canonical_bos_tokens(1, "cpu")[0], encoded)))
        if len(samples) >= num_samples:
            print(f"✅ 已取得 {len(samples)} 张样本。")
            return samples

    print(f"⚠️ 验证集读取结束，仅取得 {len(samples)} / {num_samples} 张有效样本。")
    return samples

def decode_tokens_to_render_data(tokens):
    data = []
    tokens_list = tokens.cpu().numpy().tolist()
    width_tokens = ATTRIBUTES_PER_PRIMITIVE
    for i in range(0, len(tokens_list) - (width_tokens - 1), width_tokens):
        record = tokens_list[i:i + width_tokens]
        if record[0] == TOKEN_LAYOUT.bos_token:
            continue
        if record[0] in (TOKEN_LAYOUT.true_end_token, TOKEN_LAYOUT.interrupt_end_token):
            continue
        if all(token == TOKEN_LAYOUT.pad_token for token in record):
            continue
        rgb_tokens = tokens_list[i + 6:i + 9]
        has_marker_component = any(
            token == TOKEN_LAYOUT.region_marker_token for token in rgb_tokens
        )
        is_marker = all(
            token == TOKEN_LAYOUT.region_marker_token for token in rgb_tokens
        )
        if has_marker_component and not is_marker:
            raise ValueError("无法渲染 RGB 不完整的 region marker token。")
        if is_marker:
            continue
        x_bin = tokens_list[i + 0] - TOKEN_LAYOUT.x_offset
        y_bin = tokens_list[i + 1] - TOKEN_LAYOUT.y_offset
        theta_bin = tokens_list[i + 2] - TOKEN_LAYOUT.angle_offset
        w_bin = tokens_list[i + 3] - TOKEN_LAYOUT.width_offset
        h_bin = tokens_list[i + 4] - TOKEN_LAYOUT.height_offset
        shape_bin = tokens_list[i + 5] - TOKEN_LAYOUT.shape_offset
        r_bin = tokens_list[i + 6] - TOKEN_LAYOUT.red_offset
        g_bin = tokens_list[i + 7] - TOKEN_LAYOUT.green_offset
        b_bin = tokens_list[i + 8] - TOKEN_LAYOUT.blue_offset

        w_val = w_bin / SIZE_BINS_PER_PIXEL
        h_val = h_bin / SIZE_BINS_PER_PIXEL

        edge_values = []
        for index in range(TOKEN_LAYOUT.edge_count):
            edge_bin = (
                tokens_list[i + 9 + index]
                - TOKEN_LAYOUT.edge_offset
                - index * TOKEN_LAYOUT.edge_bins
            )
            edge_values.append(
                2.0 * float(edge_bin) / max(1, TOKEN_LAYOUT.edge_bins - 1) - 1.0
            )

        data.append([
            x_bin / XY_BINS_PER_PIXEL,
            y_bin / XY_BINS_PER_PIXEL,
            w_val,
            h_val,
            shape_bin - 1,
            (theta_bin / ANGLE_BINS_PER_DEGREE) * math.pi / 180.0,
            r_bin * 2, g_bin * 2, b_bin * 2,
            *edge_values,
        ])
    return np.array(data, dtype=np.float32)


def grid_marker_tokens(stage, grid_index=None):
    """Build one externally injected marker_begin for coarse or fine."""
    if grid_index is None:
        grid_index = stage
        stage = "coarse"
    grid_side = STAGE_SPECS[stage]["grid_side"]
    marker_width = config.CANVAS_SIZE / grid_side
    marker_height = config.CANVAS_SIZE / grid_side
    row, column = divmod(int(grid_index), grid_side)
    cx = (column + 0.5) * marker_width - 0.5
    cy = (row + 0.5) * marker_height - 0.5
    return encode_primitive_tokens(
        cx, cy, marker_width, marker_height, MARKER_SHAPE_TYPE,
        MARKER_THETA, 0, 0, 0, is_region_marker=True,
    )

def _wgsl_smoothstep_coverage(distance):
    """复现数据生成器 smoothstep(-0.5, 0.5, distance) 的抗锯齿。"""
    t = np.clip(distance + np.float32(0.5), 0.0, 1.0).astype(np.float32)
    return np.float32(1.0) - t * t * (
        np.float32(3.0) - np.float32(2.0) * t
    )


def _composite_shape(canvas, row, alpha):
    """按原始 WGSL 的 SDF coverage 和逐通道截断规则合成一个图形。

    row 前 9 项为原属性，第 10-13 项为贝塞尔边控制点（归一化 -1..1）。
    TODO(bezier): 用 edge 值把矩形四条边弯成二次贝塞尔后再算 coverage。
    """
    cx, cy, width, height, shape_type, theta, red, green, blue = row[:9]
    edge_values = (
        np.asarray(row[9:13], dtype=np.float32)
        if len(row) > 9
        else np.zeros(4, dtype=np.float32)
    )
    cx = np.float32(cx)
    cy = np.float32(cy)
    half_width = np.float32(width) / np.float32(2.0)
    half_height = np.float32(height) / np.float32(2.0)
    theta = np.float32(theta)
    shape_type = int(shape_type)
    if shape_type not in (0, 1):
        raise ValueError(f"无法渲染未知 shape_type={shape_type}。")

    cos_theta = np.float32(np.cos(theta))
    sin_theta = np.float32(np.sin(theta))
    if shape_type == 0:
        extent_x = np.float32(
            abs(cos_theta) * half_width + abs(sin_theta) * half_height
        )
        extent_y = np.float32(
            abs(sin_theta) * half_width + abs(cos_theta) * half_height
        )
    else:
        half_width_cos = np.float32(half_width * cos_theta)
        half_height_sin = np.float32(half_height * sin_theta)
        half_width_sin = np.float32(half_width * sin_theta)
        half_height_cos = np.float32(half_height * cos_theta)
        extent_x = np.float32(np.sqrt(
            half_width_cos * half_width_cos
            + half_height_sin * half_height_sin
        ))
        extent_y = np.float32(np.sqrt(
            half_width_sin * half_width_sin
            + half_height_cos * half_height_cos
        ))

    canvas_height, canvas_width, _ = canvas.shape
    min_x = max(0, math.floor(float(cx - extent_x - np.float32(1.0))))
    max_x = min(
        canvas_width - 1,
        math.ceil(float(cx + extent_x + np.float32(1.0))),
    )
    min_y = max(0, math.floor(float(cy - extent_y - np.float32(1.0))))
    max_y = min(
        canvas_height - 1,
        math.ceil(float(cy + extent_y + np.float32(1.0))),
    )
    if min_x > max_x or min_y > max_y:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float32)
    ys = np.arange(min_y, max_y + 1, dtype=np.float32)
    delta_x = xs[None, :] - cx
    delta_y = ys[:, None] - cy
    local_x = delta_x * cos_theta + delta_y * sin_theta
    local_y = -delta_x * sin_theta + delta_y * cos_theta

    if shape_type == 0:
        qx = np.abs(local_x) - half_width
        qy = np.abs(local_y) - half_height
        outside = np.sqrt(
            np.maximum(qx, np.float32(0.0)) ** 2
            + np.maximum(qy, np.float32(0.0)) ** 2
        )
        distance = outside + np.minimum(
            np.maximum(qx, qy), np.float32(0.0)
        )
    else:
        inverse_half_width_squared = np.float32(1.0) / max(
            np.float32(half_width * half_width), np.float32(0.0001)
        )
        inverse_half_height_squared = np.float32(1.0) / max(
            np.float32(half_height * half_height), np.float32(0.0001)
        )
        implicit_distance = (
            local_x * local_x * inverse_half_width_squared
            + local_y * local_y * inverse_half_height_squared
            - np.float32(1.0)
        )
        gradient_x = (
            np.float32(2.0) * local_x * inverse_half_width_squared
        )
        gradient_y = (
            np.float32(2.0) * local_y * inverse_half_height_squared
        )
        gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
        distance = implicit_distance.copy()
        np.divide(
            implicit_distance,
            gradient,
            out=distance,
            where=gradient > np.float32(1e-5),
        )

    coverage = _wgsl_smoothstep_coverage(distance.astype(np.float32))
    mask = coverage > np.float32(0.0)
    if not np.any(mask):
        return

    region = canvas[min_y:max_y + 1, min_x:max_x + 1]
    effective_alpha = np.float32(alpha) * coverage
    shape_rgb = np.clip(
        np.asarray([red, green, blue], dtype=np.float32), 0.0, 255.0
    )
    mixed = (
        shape_rgb[None, None, :] * effective_alpha[:, :, None]
        + region.astype(np.float32)
        * (np.float32(1.0) - effective_alpha[:, :, None])
    )
    # WGSL 的 u32(f32) 会截断非负小数；这里保持相同规则。
    converted = mixed.astype(np.uint32).astype(np.uint8)
    region[mask] = converted[mask]


def rasterize_render_data(
    render_data, canvas_size=None, alpha=SHAPE_ALPHA
):
    """把解码图元重放为 RGB 像素，不经过 SVG/Matplotlib patch。"""
    if canvas_size is None:
        canvas_size = config.CANVAS_SIZE
    canvas_size = int(canvas_size)
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    if len(render_data) == 0:
        return canvas

    background_rgb = np.clip(
        np.asarray(render_data[0][6:9], dtype=np.float32), 0.0, 255.0
    ).astype(np.uint8)
    canvas[:, :] = background_rgb
    for row in render_data[1:]:
        _composite_shape(canvas, row, alpha)
    return canvas


def render_single_image(render_data, ax):
    """显示无描边的原始 SDF 重放结果，并在图下标出普通图形数。"""
    canvas = rasterize_render_data(render_data)
    canvas_size = canvas.shape[0]
    ax.imshow(
        canvas,
        origin="upper",
        interpolation="nearest",
        extent=(-0.5, canvas_size - 0.5, canvas_size - 0.5, -0.5),
    )
    ax.set_xlim(-0.5, canvas_size - 0.5)
    ax.set_ylim(canvas_size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    shape_count = sum(int(row[4]) in (0, 1) for row in render_data)
    ax.set_xlabel(f"Shapes: {shape_count}", fontsize=9, labelpad=4)
    return canvas

# ==========================================
# 🚀 dev2 独立状态推理引擎
# ==========================================
def _field_candidate_ids(field_index, device):
    valid_slice = field_token_slice(field_index)
    if SAMPLING_MODE == "renyi" and ATTRIBUTE_NAMES[int(field_index)] == "shape":
        # shape bin 0 解码为 -1，其余非图形 bin 也不可渲染。
        # 普通图形只有 bin 1/2（rectangle/ellipse）；region marker 由控制器注入。
        return torch.arange(
            valid_slice.start + 1,
            valid_slice.start + 3,
            dtype=torch.long,
            device=device,
        )
    return torch.arange(
        valid_slice.start,
        valid_slice.stop,
        dtype=torch.long,
        device=device,
    )


def _sample_rows(
    logits, row_indices, candidate_ids, temperature, top_k
):
    """Sample selected batch rows without splitting their shared KV cache."""
    if not row_indices:
        return torch.empty(0, dtype=torch.long, device=logits.device)
    index_tensor = torch.tensor(
        row_indices, dtype=torch.long, device=logits.device
    )
    selected_logits = logits.index_select(0, index_tensor)
    selected_logits = selected_logits.index_select(1, candidate_ids)
    return sample_candidate_token(
        selected_logits, candidate_ids, temperature, top_k
    )


_BUILTIN_SAMPLE_ROWS=_sample_rows


def _sample_region_rows(logits,rows,candidates,temperature,top_k,legal_mask=None):
    if legal_mask is None or not rows:
        return _sample_rows(logits,rows,candidates,temperature,top_k)
    if _sample_rows is not _BUILTIN_SAMPLE_ROWS and not getattr(_sample_rows,'supports_region_constraints',False):
        raise ValueError('Custom sampler must consume the legal-mask context before enabling region constraints')
    index=torch.tensor(rows,device=logits.device,dtype=torch.long)
    token=_LEGAL_SUPPORT_MASK.set(legal_mask.index_select(0,index))
    try:return _sample_rows(logits,rows,candidates,temperature,top_k)
    finally:_LEGAL_SUPPORT_MASK.reset(token)


def _generate_candidate_batch_legacy(
    model,
    base_sequences,
    device,
    random_generator,
    intervention_policy=INTERVENTION_POLICY,
    text_ids=None,
    text_mask=None,
):
    """Run the 85-global, 16-coarse, 64-fine controller with batched KV cache."""
    if isinstance(base_sequences, (list, tuple)):
        if not base_sequences:
            return []
        base_sequences = torch.stack(base_sequences)
    if base_sequences.ndim == 2:
        if base_sequences.size(1) % config.TOKENS_PER_STEP != 0:
            raise ValueError("前缀 token 数必须能被每图元 token 数整除。")
        base_tokens = base_sequences.reshape(
            base_sequences.size(0), -1, config.TOKENS_PER_STEP
        )
    elif (
        base_sequences.ndim == 3
        and base_sequences.size(2) == config.TOKENS_PER_STEP
    ):
        base_tokens = base_sequences
    else:
        raise ValueError("base_sequences 必须为 [B,T*9] 或 [B,T,9]。")
    if base_tokens.size(0) < 1 or base_tokens.size(1) < 1:
        raise ValueError("至少需要一个包含完整前缀的候选。")

    base_tokens = base_tokens.to(device=device, dtype=torch.long)
    batch_size, prefix_steps, _ = base_tokens.shape
    if text_ids is None:
        text_ids = torch.tensor([2, 3], dtype=torch.long, device=device).expand(batch_size, -1)
        text_mask = torch.ones((batch_size, 2), dtype=torch.bool, device=device)
    else:
        text_ids = text_ids.to(device=device, dtype=torch.long)
        text_mask = text_mask.to(device=device, dtype=torch.bool)
    if prefix_steps > TARGET_SHAPE_STEPS:
        raise ValueError("前缀长度超过推理上下文上限。")
    packed_positions=bool(getattr(model,'packed_text_positions',False))
    if packed_positions:
        # One small host read per batch, not one synchronization per attribute.
        lane_target_steps=[min(TARGET_SHAPE_STEPS,config.TOTAL_CONTEXT_TOKENS-int(n))
                           for n in text_mask.sum(dim=1).tolist()]
        target_steps=max(lane_target_steps)
    else:
        target_steps=min(TARGET_SHAPE_STEPS,config.TOTAL_CONTEXT_TOKENS-text_ids.size(1))
        lane_target_steps=[target_steps]*batch_size

    prefix_positions = torch.arange(
        prefix_steps, dtype=torch.long, device=device
    ).unsqueeze(0).expand(batch_size, -1)
    prefix_logits, prefix_termination_logits, past_key_values = model(
        base_tokens,
        position_ids=prefix_positions,
        use_cache=True,
        return_termination=True,
        text_ids=text_ids,
        text_mask=text_mask,
    )
    next_attribute_logits = prefix_logits[:, -1]
    next_termination_logits = prefix_termination_logits[:, -1]

    numeric_candidate_ids = [
        _field_candidate_ids(field_index, device)
        for field_index in range(len(ATTRIBUTE_NAMES))
    ]
    region_masks=None
    if ENABLE_REGION_LEGAL_ENTROPY and SAMPLING_MODE=='renyi':
        from region_constraints import RegionCandidateMasks
        region_masks=RegionCandidateMasks(device,config.CANVAS_SIZE)
    partial_template = torch.tensor(
        PARTIAL_FILL_TOKEN_IDS, dtype=torch.long, device=device
    )

    if intervention_policy not in {"normal", "perturbed", "mixed"}:
        raise ValueError("intervention_policy 必须是 normal、perturbed 或 mixed。")
    lane_policies = [
        (
            "normal" if random_generator.random() < 0.30 else "perturbed"
        )
        if intervention_policy == "mixed"
        else intervention_policy
        for _ in range(batch_size)
    ]
    perturb_regions = []
    for lane_policy in lane_policies:
        selected = {"coarse": set(), "fine": set()}
        if lane_policy == "perturbed":
            for stage, maximum in (("coarse", 3), ("fine", 12)):
                maximum = min(maximum, STAGE_SPECS[stage]["region_count"])
                count = int(random_generator.integers(0, maximum + 1))
                if count:
                    selected[stage] = set(
                        int(value)
                        for value in random_generator.choice(
                            STAGE_SPECS[stage]["region_count"],
                            size=count,
                            replace=False,
                        ).tolist()
                    )
        perturb_regions.append(selected)
    stages = ["global"] * batch_size
    closed_regions = [
        {"coarse": set(), "fine": set()} for _ in range(batch_size)
    ]
    region_counts = [
        {
            "coarse": [0] * STAGE_SPECS["coarse"]["region_count"],
            "fine": [0] * STAGE_SPECS["fine"]["region_count"],
        }
        for _ in range(batch_size)
    ]
    active_regions = [None] * batch_size
    region_visit_orders = [[] for _ in range(batch_size)]
    global_shapes = [START_STEPS - 1] * batch_size  # 第一步是背景。
    finished = [False] * batch_size
    output_lengths = [None] * batch_size
    generated_batches = []
    current_position = prefix_steps

    while current_position < target_steps:
        for lane in range(batch_size):
            if not finished[lane] and current_position>=lane_target_steps[lane]:
                finished[lane]=True;output_lengths[lane]=current_position
        # Compute termination decisions once; true_end is never top-k sampled.
        termination_predictions = next_termination_logits.argmax(dim=-1).tolist()
        begin_lanes = []
        ordinary_lanes = []
        global_lanes = []
        true_end_lanes = []
        interrupt_lanes = []
        for lane in range(batch_size):
            if finished[lane]:
                continue
            if stages[lane] == "global" and global_shapes[lane] < GLOBAL_SHAPE_STEPS:
                global_lanes.append(lane)
                ordinary_lanes.append(lane)
                continue
            if stages[lane] == "global":
                stages[lane] = "coarse"

            while active_regions[lane] is None:
                stage = stages[lane]
                spec = STAGE_SPECS[stage]
                if len(closed_regions[lane][stage]) == spec["region_count"]:
                    if stage == "coarse":
                        stages[lane] = "fine"
                        continue
                    stages[lane] = "done"
                    finished[lane] = True
                    output_lengths[lane] = current_position
                    break
                available = [
                    region_id
                    for region_id in range(spec["region_count"])
                    if region_id not in closed_regions[lane][stage]
                ]
                lane_rng = (random_generator.for_lane(lane)
                            if hasattr(random_generator, "for_lane") else random_generator)
                region_id = int(lane_rng.choice(available))
                active_regions[lane] = region_id
                region_visit_orders[lane].append((stage, region_id))
                begin_lanes.append(lane)
                break
            if finished[lane] or lane in begin_lanes:
                continue

            stage = stages[lane]
            region_id = active_regions[lane]
            count = region_counts[lane][stage][region_id]
            if count >= STAGE_SPECS[stage]["shape_cap"]:
                true_end_lanes.append(lane)
            elif (
                lane_policies[lane] == "perturbed"
                and region_id in perturb_regions[lane][stage]
                and count > 0
                and random_generator.random() < PERTURBED_INTERRUPT_PROBABILITY
            ):
                interrupt_lanes.append(lane)
            elif int(termination_predictions[lane]) == 1:
                true_end_lanes.append(lane)
            else:
                ordinary_lanes.append(lane)
        if all(finished):
            break
        coordinate_masks=region_masks.for_lanes(stages,active_regions) if region_masks is not None else (None,None)

        # true_end is a harmless placeholder for lanes whose output was already
        # truncated; all live lanes overwrite it with their actual next record.
        completed_tokens = canonical_marker_end_tokens(
            batch_size, device, dtype=torch.long
        )
        if begin_lanes:
            marker_tokens = torch.tensor(
                [
                    grid_marker_tokens(
                        stages[lane], active_regions[lane]
                    )
                    for lane in begin_lanes
                ],
                dtype=torch.long,
                device=device,
            )
            completed_tokens[begin_lanes] = marker_tokens

        if interrupt_lanes:
            completed_tokens[interrupt_lanes] = canonical_interrupt_end_tokens(
                len(interrupt_lanes), device, dtype=torch.long
            )
            for lane in interrupt_lanes:
                active_regions[lane] = None
        for lane in true_end_lanes:
            stage = stages[lane]
            region_id = active_regions[lane]
            closed_regions[lane][stage].add(region_id)
            active_regions[lane] = None

        partial_tokens = partial_template.unsqueeze(0).expand(
            batch_size, -1
        ).clone()
        temperature, top_k = sampling_parameters(0, current_position)
        sampled_x = _sample_region_rows(
            next_attribute_logits,
            ordinary_lanes,
            numeric_candidate_ids[0],
            temperature,
            top_k,
            coordinate_masks[0],
        )
        if ordinary_lanes:
            partial_tokens[ordinary_lanes, 0] = sampled_x
        position_ids = torch.full(
            (batch_size, 1),
            current_position,
            dtype=torch.long,
            device=device,
        )
        for field_index in range(1, len(ATTRIBUTE_NAMES)):
            if not ordinary_lanes:
                break
            logits = model(
                partial_tokens.unsqueeze(1),
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                text_mask=text_mask,
            )
            temperature, top_k = sampling_parameters(
                field_index, current_position
            )
            field_candidates = numeric_candidate_ids[field_index]
            if current_position == 1 and ATTRIBUTE_NAMES[field_index] == "shape":
                # Preserve the latest bug fix: record1 is the training canvas
                # background (shape-bin0), not an ordinary rectangle/ellipse.
                # Keep the sampler call so even the one-choice field draws RNG.
                field_candidates = torch.tensor(
                    [TOKEN_LAYOUT.shape_offset],dtype=torch.long,device=device
                )
            next_tokens = _sample_region_rows(
                logits[:, -1],
                ordinary_lanes,
                field_candidates,
                temperature,
                top_k,
                coordinate_masks[1] if field_index==1 else None,
            )
            partial_tokens[ordinary_lanes, field_index] = next_tokens

        if ordinary_lanes:
            completed_tokens[ordinary_lanes] = partial_tokens[ordinary_lanes]
        for lane in global_lanes:
            global_shapes[lane] += 1
        for lane in ordinary_lanes:
            if lane not in global_lanes:
                stage = stages[lane]
                region_id = active_regions[lane]
                region_counts[lane][stage][region_id] += 1

        logits, termination_logits, past_key_values = model(
            completed_tokens.unsqueeze(1),
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_termination=True,
            text_mask=text_mask,
        )
        next_attribute_logits = logits[:, -1]
        next_termination_logits = termination_logits[:, -1]
        generated_batches.append(completed_tokens.clone())
        current_position += 1

    for lane in range(batch_size):
        if output_lengths[lane] is None:
            output_lengths[lane] = current_position
    if generated_batches:
        history = torch.cat(
            (base_tokens, torch.stack(generated_batches, dim=1)), dim=1
        )
    else:
        history = base_tokens

    results = []
    for lane in range(batch_size):
        sequence = history[lane, :output_lengths[lane]].reshape(-1).cpu()
        results.append(
            (
                sequence,
                sum(len(value) for value in closed_regions[lane].values()),
                region_visit_orders[lane],
                active_regions[lane] is not None,
            )
        )
    return results


def get_optimized_inference_info(model):
    engine=getattr(model,'_visualization_inference_engine',None)
    result=engine.stats() if engine is not None else {'optimized':False}
    result.update(region_legal_entropy=ENABLE_REGION_LEGAL_ENTROPY,
                  region_constraint_version='renderer_center_bounds_v1')
    return result


def generate_candidate_batch(model,base_sequences,device,random_generator,
                             intervention_policy=INTERVENTION_POLICY,text_ids=None,text_mask=None):
    """Official entry, retaining CPU/explicitly disabled legacy compatibility."""
    if not ENABLE_OPTIMIZED_INFERENCE or torch.device(device).type!='cuda':
        return _generate_candidate_batch_legacy(model,base_sequences,device,random_generator,
            intervention_policy=intervention_policy,text_ids=text_ids,text_mask=text_mask)
    from optimized_inference import VisualizationInferenceEngine
    with _ENGINE_CREATION_LOCK:
        engine=getattr(model,'_visualization_inference_engine',None)
        if engine is None:
            engine=VisualizationInferenceEngine(model,OPTIMIZED_INFERENCE_COMPILE,
                OPTIMIZED_INFERENCE_GRAPH,OPTIMIZED_COMPILE_THREADS,RENYI_ALPHA)
            model._visualization_inference_engine=engine
    with engine.generation_lock:
        engine.sampler.alpha=RENYI_ALPHA;engine.sampler.reset()
        token=_OPTIMIZED_SAMPLER.set(engine.sampler) if SAMPLING_MODE=='renyi' else None
        try:
            result=_generate_candidate_batch_legacy(engine,base_sequences,device,random_generator,
                intervention_policy=intervention_policy,text_ids=text_ids,text_mask=text_mask)
            if token is not None:engine.sampler.check()
            return result
        finally:
            if token is not None:_OPTIMIZED_SAMPLER.reset(token)


def generate_candidate(model, base_seq, device, random_generator):
    """Backward-compatible one-candidate wrapper around batched generation."""
    return generate_candidate_batch(
        model, base_seq.unsqueeze(0), device, random_generator
    )[0]


def create_inference_model(device):
    return GeometrizeGPT(
        vocab_size=config.MODEL_VOCAB_SIZE,
        d_model=config.MODEL_D_MODEL,
        n_layer=config.MODEL_N_LAYER,
        n_head=config.MODEL_N_HEAD,
        ffn_hidden=config.MODEL_FFN_HIDDEN,
        max_context_len=config.TOTAL_CONTEXT_TOKENS,
        max_block_positions=config.MAX_BLOCK_POSITIONS,
        norm_eps=config.MODEL_NORM_EPS,
        rope_base=config.MODEL_ROPE_BASE,
        text_vocab_size=config.TEXT_VOCAB_SIZE,
        max_text_tokens=config.TEXT_MAX_TOKENS,
    ).to(device)


def resolve_checkpoint_path():
    checkpoint_path = MODEL_CKPT_PATH
    if checkpoint_path is None:
        raise FileNotFoundError("尚无文字条件训练 checkpoint；不能用旧图元权重假装文字模型。")
    return checkpoint_path


def encode_inference_prompts(prompts):
    processor = spm.SentencePieceProcessor(model_file=config.CAPTION_TOKENIZER_PATH)
    encoded = []
    for prompt in prompts:
        ids = [processor.bos_id(), *processor.encode(prompt), processor.eos_id()] if prompt.strip() else [processor.bos_id(), processor.eos_id()]
        if len(ids) > config.TEXT_MAX_TOKENS:
            raise ValueError(f"推理提示词超过 {config.TEXT_MAX_TOKENS} token。")
        encoded.append(ids)
    return encoded


def load_inference_models(devices):
    """Load one independent model replica on every usable inference device."""
    checkpoint_path = resolve_checkpoint_path()
    checkpoint = None
    state_key = None
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
        state_key = (
            "ema_state_dict" if "ema_state_dict" in checkpoint else "model_state_dict"
        )
        checkpoint_format(checkpoint, config, state_key)
    else:
        raise FileNotFoundError("No real checkpoint provided; refusing random inference weights")

    loaded_models = []
    for device in devices:
        model = None
        try:
            if device.type == "cuda":
                torch.cuda.set_device(device)
            model = create_inference_model(device)
            if checkpoint is not None:
                load_checkpoint_state(
                    model, checkpoint, config, state_key=state_key
                )
            model.eval()
            loaded_models.append((device, model))
            print(f"--> [成功] 模型已装载到 {device}")
        except Exception as error:
            if len(devices) == 1 or device.type != "cuda":
                raise
            print(f"  ⚠️ 跳过 {device}：模型装载失败 ({error})")
            del model
            gc.collect()
            try:
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()
            except Exception:
                pass

    del checkpoint
    gc.collect()
    if not loaded_models:
        raise RuntimeError("没有设备能够成功装载推理模型。")
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"--> [成功] 已装载权重: {checkpoint_path}")
    if len(loaded_models) < len(devices):
        print(f"ℹ️ 实际使用 {len(loaded_models)} / {len(devices)} 个推理设备。")
    return loaded_models


def partition_candidate_jobs(num_tests, candidates_per_test, worker_count):
    """Round-robin independent candidate jobs across device workers."""
    if worker_count < 1:
        raise ValueError("worker_count 必须至少为 1。")
    assignments = [[] for _ in range(worker_count)]
    jobs = (
        (row_index, candidate_index)
        for row_index in range(num_tests)
        for candidate_index in range(candidates_per_test)
    )
    for job_index, job in enumerate(jobs):
        assignments[job_index % worker_count].append(job)
    return assignments


def _run_device_jobs(
    device,
    model,
    jobs,
    gt_samples,
    candidates_per_test=INFERENCE_BATCH_SIZE,
    sequences_per_device_batch=SEQUENCES_PER_DEVICE_BATCH,
    prompt_token_ids=None,
):
    if prompt_token_ids is None:
        prompt_token_ids = [[2, 3] for _ in gt_samples]
    if device.type == "cuda":
        torch.cuda.set_device(device)
    random_generator = np.random.default_rng()
    completed_jobs = []
    autocast_enabled = device.type == "cuda" and ENABLE_INFERENCE_AUTOCAST
    autocast_dtype = (
        torch.bfloat16
        if autocast_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.inference_mode(), torch.amp.autocast(
        "cuda", enabled=autocast_enabled, dtype=autocast_dtype
    ):
        for batch_start in range(0, len(jobs), sequences_per_device_batch):
            batch_jobs = jobs[
                batch_start:batch_start + sequences_per_device_batch
            ]
            base_sequences = torch.stack(
                [
                    gt_samples[row_index][
                        : START_STEPS * config.TOKENS_PER_STEP
                    ]
                    for row_index, _ in batch_jobs
                ]
            )
            selected_prompts = [prompt_token_ids[row_index] for row_index, _ in batch_jobs]
            text_width = max(map(len, selected_prompts))
            text_ids = torch.zeros((len(batch_jobs), text_width), dtype=torch.long)
            text_mask = torch.zeros((len(batch_jobs), text_width), dtype=torch.bool)
            for index, ids in enumerate(selected_prompts):
                text_ids[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                text_mask[index, :len(ids)] = True
            batch_predictions = generate_candidate_batch(
                model, base_sequences, device, random_generator,
                text_ids=text_ids, text_mask=text_mask,
            )
            for (row_index, candidate_index), prediction in zip(
                batch_jobs, batch_predictions
            ):
                predicted, completed_regions, region_order, active_region = (
                    prediction
                )
                completed_jobs.append(
                    (
                        row_index,
                        candidate_index,
                        (
                            predicted,
                            completed_regions,
                            region_order,
                            active_region,
                        ),
                    )
                )
            print(
                f"  [{device}] 并发完成 {len(batch_jobs)} 条序列 | "
                f"进度={min(batch_start + len(batch_jobs), len(jobs))}/"
                f"{len(jobs)}"
            )
    return completed_jobs


def _run_gpu_process(
    device_id,
    jobs,
    gt_arrays,
    candidates_per_test,
    sequences_per_device_batch,
    prompt_token_ids,
):
    """Spawn-safe worker: own one GPU, model replica, interpreter and GIL."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    gt_samples = [torch.from_numpy(array) for array in gt_arrays]
    loaded_models = load_inference_models([device])
    completed_jobs = _run_device_jobs(
        device,
        loaded_models[0][1],
        jobs,
        gt_samples,
        candidates_per_test,
        sequences_per_device_batch,
        prompt_token_ids,
    )

    serialized_jobs = []
    for row_index, candidate_index, prediction in completed_jobs:
        predicted, completed_regions, region_order, active_region = prediction
        serialized_jobs.append(
            (
                row_index,
                candidate_index,
                (
                    predicted.numpy(),
                    completed_regions,
                    region_order,
                    active_region,
                ),
            )
        )
    del loaded_models
    gc.collect()
    torch.cuda.empty_cache()
    return serialized_jobs


def assemble_predictions(worker_results, num_tests, candidates_per_test):
    predictions = [
        [None for _ in range(candidates_per_test)]
        for _ in range(num_tests)
    ]
    for results in worker_results:
        for row_index, candidate_index, prediction in results:
            predicted, completed_regions, region_order, active_region = prediction
            if isinstance(predicted, np.ndarray):
                predicted = torch.from_numpy(predicted)
            predictions[row_index][candidate_index] = (
                predicted,
                completed_regions,
                region_order,
                active_region,
            )
    if any(prediction is None for row in predictions for prediction in row):
        raise RuntimeError("多设备推理结果不完整。")
    return predictions


def generate_all_predictions(
    devices,
    gt_samples,
    candidates_per_test=INFERENCE_BATCH_SIZE,
    sequences_per_device_batch=SEQUENCES_PER_DEVICE_BATCH,
    prompts=None,
):
    if (
        isinstance(candidates_per_test, bool)
        or not isinstance(candidates_per_test, Integral)
        or candidates_per_test < 1
    ):
        raise ValueError("candidates_per_test 必须是正整数。")
    candidates_per_test = int(candidates_per_test)
    if (
        isinstance(sequences_per_device_batch, bool)
        or not isinstance(sequences_per_device_batch, Integral)
        or sequences_per_device_batch < 1
    ):
        raise ValueError("sequences_per_device_batch 必须是正整数。")
    sequences_per_device_batch = int(sequences_per_device_batch)
    if prompts is None:
        prompts = [""] * len(gt_samples)
    if len(prompts) != len(gt_samples):
        raise ValueError("提示词数量必须等于测试图像数量。")
    prompt_token_ids = encode_inference_prompts(prompts)
    worker_count = len(devices)
    assignments = partition_candidate_jobs(
        len(gt_samples), candidates_per_test, worker_count
    )
    print(
        f"⚡ 批量自回归：{len(gt_samples) * candidates_per_test} 条序列 | "
        f"{worker_count} 个设备 | 每设备并发上限={sequences_per_device_batch}"
    )

    if worker_count == 1:
        loaded_models = load_inference_models(devices)
        device, model = loaded_models[0]
        worker_results = [
            _run_device_jobs(
                device,
                model,
                assignments[0],
                gt_samples,
                candidates_per_test,
                sequences_per_device_batch,
                prompt_token_ids,
            )
        ]
        del loaded_models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
    else:
        worker_results = []
        gt_arrays = [sample.numpy() for sample in gt_samples]
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=spawn_context
        ) as executor:
            futures = [
                executor.submit(
                    _run_gpu_process,
                    device.index,
                    jobs,
                    gt_arrays,
                    candidates_per_test,
                    sequences_per_device_batch,
                    prompt_token_ids,
                )
                for device, jobs in zip(devices, assignments)
                if jobs
            ]
            for future in as_completed(futures):
                worker_results.append(future.result())
    return assemble_predictions(
        worker_results, len(gt_samples), candidates_per_test
    )


def main():
    validate_sampling_schedules()
    assert START_STEPS >= 1, "❌ START_STEPS 必须至少为 1"
    assert START_STEPS <= GLOBAL_SHAPE_STEPS + 1
    assert TARGET_SHAPE_STEPS <= config.CONTEXT_STEPS, "❌ 推理长度超过模型训练视界"
    print("-" * 60)
    print(
        f"采样模式={SAMPLING_MODE}"
        + (f" | Rényi α={RENYI_ALPHA:g} | 温度=1 | top-N 内均匀抽样"
           if SAMPLING_MODE == "renyi" else " | 原动态温度/top-k 曲线")
    )
    print(
        f" 🎨 启动 v3 三阶段推理 | 起步={START_STEPS}步 | "
        f"上下文上限={TARGET_SHAPE_STEPS} | 候选数={INFERENCE_BATCH_SIZE} | "
        f"每设备并发={SEQUENCES_PER_DEVICE_BATCH} | "
        f"全局图形={GLOBAL_SHAPE_STEPS} | 最多区域={MAX_REGIONS}"
    )
    print("-" * 60)

    devices = resolve_inference_devices()
    gt_samples = load_test_samples(TEST_DATA_PATH, NUM_TESTS)
    if len(gt_samples) < NUM_TESTS:
        raise RuntimeError(f"有效测试样本不足：需要 {NUM_TESTS} 张，实际 {len(gt_samples)} 张。")
    prompts_file = os.environ.get("PAINTER_PROMPTS_FILE")
    if prompts_file:
        prompts = Path(prompts_file).read_text(encoding="utf-8").splitlines()
    else:
        prompts = [os.environ.get("PAINTER_TEXT_PROMPT", "")] * len(gt_samples)
    all_predictions = generate_all_predictions(devices, gt_samples, prompts=prompts)

    fig, axes = plt.subplots(
        NUM_TESTS,
        1 + INFERENCE_BATCH_SIZE,
        figsize=(3 * (1 + INFERENCE_BATCH_SIZE), 3 * NUM_TESTS),
    )
    if NUM_TESTS == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx in range(NUM_TESTS):
        gt_tokens = gt_samples[row_idx]
        predictions = all_predictions[row_idx]
        render_single_image(decode_tokens_to_render_data(gt_tokens), axes[row_idx, 0])
        if row_idx == 0:
            axes[row_idx, 0].set_title("GT v3")
        for candidate_index, prediction in enumerate(predictions):
            predicted, completed_regions, region_order, active = prediction
            render_single_image(
                decode_tokens_to_render_data(predicted),
                axes[row_idx, candidate_index + 1],
            )
            if row_idx == 0:
                axes[row_idx, candidate_index + 1].set_title(
                    f"Pred {candidate_index + 1} ({completed_regions}/{MAX_REGIONS})"
                )
            print(
                f"  candidate={candidate_index + 1} | "
                f"regions={completed_regions}/{MAX_REGIONS} | "
                f"active_at_cap={active} | "
                f"visits={[(stage, value + 1) for stage, value in region_order]}"
            )
        print(f"✅ 图像 [{row_idx + 1}/{NUM_TESTS}] 推演完毕！")

    plt.tight_layout()
    save_path = str(VISUALIZATION_DIR / "full_sequence_batch_inference_test.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"--> 🎯 [成功] v3 推理结果已保存至: {save_path}")

if __name__ == "__main__":
    raise SystemExit("This is the reference decoder library; use python model/infer.py --help")
