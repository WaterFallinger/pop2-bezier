import gc
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import TrainConfig
from caption_data import CaptionStore
from token_layout import (
    ANGLE_BINS_PER_DEGREE,
    SIZE_BINS_PER_PIXEL,
    TOKEN_LAYOUT,
    XY_BINS_PER_PIXEL,
)


# v7: 13-token primitive records (bezier edge fields). Old caches are rejected.
CACHE_FORMAT_VERSION = 7
CACHE_DTYPE = np.dtype("<u2")
TOKENS_PER_STEP = TrainConfig.TOKENS_PER_STEP
CACHE_SEQUENCES_PER_SHARD = 1000
PACKED_CSV_ROWS_PER_CHUNK = 131_072
PACKED_PART_RE = re.compile(r"data_part_([0-9]+)\.csv\Z")
PACKED_STORAGE_LAYOUTS = {"large_csv_v1", "merged_csv_shards_v1"}

CSV_DTYPE = {
    "image_name": "string",
    "cx": "float32",
    "cy": "float32",
    "w": "float32",
    "h": "float32",
    "shape_type": "int8",
    "theta": "float32",
    "r": "uint8",
    "g": "uint8",
    "b": "uint8",
    "is_region_marker": "uint8",
    "stage": "string",
    "region_id": "int16",
}


def _encode_complete_groups(
    frame, target_steps, apply_flip, canvas_size=256
):
    """编码完整 image_name 分组，超长截断、短序列右侧 padding。"""
    if frame is None or frame.empty:
        return np.empty((0, target_steps * TOKENS_PER_STEP), dtype=CACHE_DTYPE)

    image_names = frame["image_name"].to_numpy(copy=False)
    group_starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(image_names[1:] != image_names[:-1]).astype(np.int64)
            + 1,
        )
    )
    group_ends = np.concatenate(
        (group_starts[1:], np.array([len(frame)], dtype=np.int64))
    )
    group_sizes = group_ends - group_starts
    if np.any(group_sizes > target_steps):
        raise ValueError(f"A complete sequence exceeds {target_steps} raw records; refusing silent truncation")
    clipped_sizes = group_sizes
    step_indices = np.arange(target_steps, dtype=np.int64)
    populated = step_indices[None, :] < clipped_sizes[:, None]
    selected_rows = (
        group_starts[:, None] + step_indices[None, :]
    )[populated]

    cx = frame["cx"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    cy = frame["cy"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    width = frame["w"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    height = frame["h"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    shape_type = frame["shape_type"].to_numpy(
        dtype=np.int32, copy=False
    )[selected_rows]
    theta = frame["theta"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    red = frame["r"].to_numpy(dtype=np.int32, copy=False)[selected_rows]
    green = frame["g"].to_numpy(dtype=np.int32, copy=False)[selected_rows]
    blue = frame["b"].to_numpy(dtype=np.int32, copy=False)[selected_rows]
    edge_columns = ("e0", "e1", "e2", "e3")
    if all(name in frame.columns for name in edge_columns):
        edge_raw = [
            frame[name].to_numpy(dtype=np.float32, copy=False)[selected_rows]
            for name in edge_columns
        ]
    else:
        # 兼容旧版无 edge 列的 CSV：默认无弯曲（归一化偏移 0）。
        edge_raw = [np.zeros_like(cx) for _ in edge_columns]
    is_region_marker = frame["is_region_marker"].to_numpy(
        dtype=np.uint8, copy=False
    )[selected_rows]
    if np.any((is_region_marker != 0) & (is_region_marker != 1)):
        raise ValueError("is_region_marker 只能取 0 或 1。")
    marker_begin = (is_region_marker == 1) & (shape_type == 2)
    marker_end = (is_region_marker == 1) & (shape_type == 3)
    interrupt_end = (is_region_marker == 1) & (shape_type == 4)
    invalid_marker = (is_region_marker == 1) & ~(
        marker_begin | marker_end | interrupt_end
    )
    invalid_primitive = (is_region_marker == 0) & ~np.isin(
        shape_type, np.array([-1, 0, 1], dtype=np.int32)
    )
    if np.any(invalid_marker):
        raise ValueError(
            "region marker 必须使用 shape_type=2(begin)、3(true_end) 或 4(interrupt_end)。"
        )
    if np.any(invalid_primitive):
        raise ValueError("非 marker 行的 shape_type 只能是 -1、0 或 1。")

    if apply_flip:
        # 像素中心坐标位于 [0, canvas_size - 1]。
        cx = (float(canvas_size) - 1.0) - cx
        theta = np.where(shape_type != -1, np.pi - theta, theta)
        theta = np.mod(theta, np.pi)

    theta_deg = np.mod(theta * 180.0 / np.pi, 180.0)
    swap_mask = theta_deg >= 90.0
    theta_deg = np.where(swap_mask, theta_deg - 90.0, theta_deg)
    canonical_width = np.where(swap_mask, height, width)
    canonical_height = np.where(swap_mask, width, height)

    encoded_red = (
        np.clip(red // 2, 0, TOKEN_LAYOUT.color_bins - 1).astype(np.int32)
        + TOKEN_LAYOUT.red_offset
    )
    encoded_green = (
        np.clip(green // 2, 0, TOKEN_LAYOUT.color_bins - 1).astype(np.int32)
        + TOKEN_LAYOUT.green_offset
    )
    encoded_blue = (
        np.clip(blue // 2, 0, TOKEN_LAYOUT.color_bins - 1).astype(np.int32)
        + TOKEN_LAYOUT.blue_offset
    )
    encoded_red = np.where(
        marker_begin, TOKEN_LAYOUT.marker_begin_token, encoded_red
    )
    encoded_green = np.where(
        marker_begin, TOKEN_LAYOUT.marker_begin_token, encoded_green
    )
    encoded_blue = np.where(
        marker_begin, TOKEN_LAYOUT.marker_begin_token, encoded_blue
    )

    encoded_rows = np.stack(
        [
            np.clip(
                cx * XY_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.x_bins - 1
            ).astype(np.int32)
            + TOKEN_LAYOUT.x_offset,
            np.clip(
                cy * XY_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.y_bins - 1
            ).astype(np.int32)
            + TOKEN_LAYOUT.y_offset,
            np.clip(
                theta_deg * ANGLE_BINS_PER_DEGREE,
                0,
                TOKEN_LAYOUT.angle_bins - 1,
            ).astype(np.int32)
            + TOKEN_LAYOUT.angle_offset,
            np.clip(
                canonical_width * SIZE_BINS_PER_PIXEL,
                0,
                TOKEN_LAYOUT.width_bins - 1,
            ).astype(np.int32)
            + TOKEN_LAYOUT.width_offset,
            np.clip(
                canonical_height * SIZE_BINS_PER_PIXEL,
                0,
                TOKEN_LAYOUT.height_bins - 1,
            ).astype(np.int32)
            + TOKEN_LAYOUT.height_offset,
            np.clip(
                shape_type + 1, 0, TOKEN_LAYOUT.shape_bins - 1
            ).astype(np.int32)
            + TOKEN_LAYOUT.shape_offset,
            encoded_red,
            encoded_green,
            encoded_blue,
            *[
                np.clip(
                    (value + 1.0) * 0.5 * float(TOKEN_LAYOUT.edge_bins - 1),
                    0,
                    TOKEN_LAYOUT.edge_bins - 1,
                ).astype(np.int32)
                + TOKEN_LAYOUT.edge_offset
                + index * TOKEN_LAYOUT.edge_bins
                for index, value in enumerate(edge_raw)
            ],
        ],
        axis=1,
    ).astype(CACHE_DTYPE, copy=False)

    # marker_end 不重复 begin 的几何属性；它在模型中只占一条 embedding。
    encoded_rows[marker_end] = TOKEN_LAYOUT.pad_token
    encoded_rows[marker_end, 0] = TOKEN_LAYOUT.true_end_token
    encoded_rows[interrupt_end] = TOKEN_LAYOUT.pad_token
    encoded_rows[interrupt_end, 0] = TOKEN_LAYOUT.interrupt_end_token

    encoded = np.full(
        (group_starts.size, target_steps, TOKENS_PER_STEP),
        TOKEN_LAYOUT.pad_token,
        dtype=CACHE_DTYPE,
    )
    encoded.reshape(-1, TOKENS_PER_STEP)[populated.reshape(-1)] = encoded_rows
    return encoded.reshape(group_starts.size, target_steps * TOKENS_PER_STEP)


def _split_complete_prefix(frame):
    """留下最后一个可能跨 CSV chunk 的图像分组，其余分组可安全编码。"""
    if frame.empty:
        return frame, None
    image_names = frame["image_name"].to_numpy(copy=False)
    changes = np.flatnonzero(image_names[1:] != image_names[:-1]) + 1
    if changes.size == 0:
        return None, frame.copy()
    last_group_start = int(changes[-1])
    return frame.iloc[:last_group_start], frame.iloc[last_group_start:].copy()


def _records_fingerprint(records):
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_bounds(frame):
    """返回连续 image_name 分组的 [start, end) 边界。"""
    if frame is None or frame.empty:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    image_names = frame["image_name"].to_numpy(copy=False)
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(image_names[1:] != image_names[:-1]).astype(np.int64) + 1,
        )
    )
    ends = np.concatenate(
        (starts[1:], np.array([len(frame)], dtype=np.int64))
    )
    return starts, ends


def _validate_packed_sequence_frame(frame, image_name, source_label, max_total_records):
    """
    校验一个从大 CSV 中切出的完整 image_name 序列。

    新的大块格式没有逐图 summary.json，因此这里保留旧 loader 对 CSV 结构本身
    能验证的全部约束；valid_for_training 已由最终生成器在写入大 CSV 前保证。
    """
    required = set(CSV_DTYPE)
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{source_label} 缺少字段: {sorted(required-set(frame.columns))}"
        )
    if frame.empty:
        raise ValueError(f"{source_label}:{image_name} 是空序列。")
    if len(frame) > int(max_total_records):
        raise ValueError(
            f"{source_label}:{image_name} 记录数 {len(frame)} "
            f"超过 MAX_IMAGE_STEPS={max_total_records}。"
        )
    if not bool((frame["image_name"].astype(str) == str(image_name)).all()):
        raise ValueError(f"{source_label}:{image_name} 分组内混入其他 image_name。")

    shape = frame["shape_type"].to_numpy(dtype=np.int32, copy=False)
    stage = frame["stage"].astype(str).to_numpy(copy=False)
    region = frame["region_id"].to_numpy(dtype=np.int32, copy=False)
    marker = frame["is_region_marker"].to_numpy(dtype=np.int32, copy=False)

    if (
        shape[0] != -1
        or stage[0] != "global"
        or region[0] != -1
        or marker[0] != 0
    ):
        raise ValueError(
            f"{source_label}:{image_name} 第一行必须是非 marker 的 global 背景。"
        )
    if np.any((shape >= 2) != (marker == 1)):
        raise ValueError(
            f"{source_label}:{image_name} marker 标志与 shape_type 不一致。"
        )
    if np.any(~np.isin(stage, np.array(["global", "coarse", "fine"]))):
        raise ValueError(f"{source_label}:{image_name} 包含未知 stage。")
    if np.any((stage == "global") & (region != -1)):
        raise ValueError(
            f"{source_label}:{image_name} global 记录必须使用 region_id=-1。"
        )
    if np.any((stage == "coarse") & ((region < 0) | (region >= 16))):
        raise ValueError(
            f"{source_label}:{image_name} coarse region_id 越界。"
        )
    if np.any((stage == "fine") & ((region < 0) | (region >= 64))):
        raise ValueError(
            f"{source_label}:{image_name} fine region_id 越界。"
        )

    interrupt_indices = np.flatnonzero(shape == 4)
    if interrupt_indices.size:
        non_terminal_interrupts = interrupt_indices[
            interrupt_indices + 1 < len(frame)
        ]
        if (
            interrupt_indices[-1] + 1 == len(frame)
            and len(frame) != int(max_total_records)
        ):
            raise ValueError(
                f"{source_label}:{image_name} 只有达到硬预算 "
                f"{max_total_records} 条时才允许以 interrupt_end 截断。"
            )
        next_indices = non_terminal_interrupts + 1
        if np.any(shape[next_indices] != 2) or np.any(
            stage[next_indices] != stage[non_terminal_interrupts]
        ):
            raise ValueError(
                f"{source_label}:{image_name} 每个 interrupt_end 后必须立即跟同阶段 marker_begin。"
            )

    active = None
    closed = {"coarse": set(), "fine": set()}
    stage_rank = {"global": 0, "coarse": 1, "fine": 2}
    highest_stage = 0

    for row_index, (row_shape, row_stage, row_region) in enumerate(
        zip(shape.tolist(), stage.tolist(), region.tolist())
    ):
        rank = stage_rank[row_stage]
        if rank < highest_stage:
            raise ValueError(
                f"{source_label}:{image_name}:{row_index + 2} stage 顺序倒退。"
            )
        highest_stage = max(highest_stage, rank)

        if row_stage == "global":
            if row_shape not in (-1, 0, 1):
                raise ValueError(
                    f"{source_label}:{image_name}:{row_index + 2} global 不允许控制事件。"
                )
            continue

        key = (row_stage, int(row_region))

        if row_shape == 2:
            if active is not None or row_region in closed[row_stage]:
                raise ValueError(
                    f"{source_label}:{image_name}:{row_index + 2} 非法 begin/re-entry。"
                )
            active = key
        elif row_shape in (-1, 0, 1):
            if active != key:
                raise ValueError(
                    f"{source_label}:{image_name}:{row_index + 2} 图形不属于当前活动块。"
                )
        elif row_shape in (3, 4):
            if active != key:
                raise ValueError(
                    f"{source_label}:{image_name}:{row_index + 2} 结束事件不匹配活动块。"
                )
            if row_shape == 3:
                closed[row_stage].add(int(row_region))
            active = None


def _load_completed_names(path):
    """读取大 CSV 版本目录自己的完成清单，并返回集合及内容 SHA-256。"""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"{path} 最后一行不完整。")

    names = []
    seen = set()
    for line_number, raw_line in enumerate(data.decode("utf-8").splitlines(), 1):
        name = raw_line.strip()
        if not name:
            raise ValueError(f"{path}:{line_number} 出现空 image_name。")
        if name in seen:
            raise ValueError(f"{path}:{line_number} 重复 image_name: {name}")
        seen.add(name)
        names.append(name)

    return seen, hashlib.sha256(data).hexdigest()


def _iter_packed_csv_frames(csv_paths):
    """
    流式读取多个 data_part_N.csv。

    最后一个 image_name 分组会保留到下一 chunk（乃至下一 part），因此即便未来
    某种打包器恰好让一个序列跨 chunk/part，也不会把半条序列送入编码器。
    """
    carry = None

    for csv_path in csv_paths:
        for chunk in pd.read_csv(
            csv_path,
            dtype=CSV_DTYPE,
            engine="c",
            chunksize=PACKED_CSV_ROWS_PER_CHUNK,
        ):
            if carry is not None and not carry.empty:
                chunk = pd.concat((carry, chunk), ignore_index=True)
                carry = None

            complete, carry = _split_complete_prefix(chunk)
            if complete is not None and not complete.empty:
                yield complete

    if carry is not None and not carry.empty:
        yield carry


def _atomic_json(path, value):
    """原子写入小型 JSON 元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")

    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

    os.replace(temporary, path)


class GeometrizeFullSequenceDataset(Dataset):
    """固定长度序列数据集；CSV 仅首次流式转换，训练时由 mmap 按需读取。"""

    def __init__(self, config=None):
        self.config = config if config is not None else TrainConfig()
        self.config.validate_sequence_config()
        if self.config.MAX_IMAGE_STEPS < self.config.CONTEXT_STEPS - 1:
            raise ValueError(
                "MAX_IMAGE_STEPS 必须覆盖 BOS 后的全部预测位置。"
            )
        if self.config.CONTEXT_TOKENS != (
            self.config.CONTEXT_STEPS * TOKENS_PER_STEP
        ):
            raise ValueError("缓存数据格式要求每个图元恰好包含 9 个 token。")

        self.data_root = Path(self.config.DATA_DIR)
        self.dataset_manifest = self._validate_packed_dataset_root(self.data_root)
        self.caption_store = CaptionStore(self.config)
        self.current_epoch = None

        self.sequence_indices = np.empty(0, dtype=np.int32)
        self.sequence_names = []
        self.token_shards = []
        self.shard_starts = np.empty(0, dtype=np.int64)
        self.shard_ends = np.empty(0, dtype=np.int64)
        self.shard_metadata = []
        self.cache_prefetch_thread = None
        self.cache_prefetch_epoch = None
        self.cache_prefetch_error = None

    def _version_for_epoch(self, epoch):
        version_idx = ((epoch - 1) % self.config.VERSION_CYCLE_EPOCHS) + 1
        apply_flip = (
            self.config.ENABLE_FLIP_AUGMENTATION
            and epoch > self.config.VERSION_CYCLE_EPOCHS
        )
        return version_idx, apply_flip

    def set_epoch(self, epoch):
        version_idx, apply_flip = self._version_for_epoch(epoch)
        if (
            self.cache_prefetch_thread is not None
            and self.cache_prefetch_epoch == epoch
        ):
            print(f"⏳ [Cache] 等待 Epoch {epoch} 的后台缓存任务收尾...")
            self.cache_prefetch_thread.join()
            if self.cache_prefetch_error is not None:
                raise RuntimeError(
                    f"Epoch {epoch} 的后台缓存构建失败"
                ) from self.cache_prefetch_error
        print(
            f"\n🔄 [Dataset] 准备 Epoch {epoch} -> 锁定版本: "
            f"[v{version_idx}] | 水平翻转: [{apply_flip}]"
        )
        self._open_version_cache(version_idx, apply_flip, epoch)

    def start_next_epoch_cache_prefetch(self, next_epoch):
        """训练当前 epoch 时，在后台用多核构建下一版本的磁盘分片。"""
        if (
            self.cache_prefetch_thread is not None
            and self.cache_prefetch_thread.is_alive()
        ):
            return False

        version_idx, apply_flip = self._version_for_epoch(next_epoch)
        self.cache_prefetch_epoch = next_epoch
        self.cache_prefetch_error = None

        def prepare():
            try:
                print(
                    f"🔮 [Cache] 后台准备 Epoch {next_epoch} "
                    f"(v{version_idx}, flip={apply_flip})...",
                    flush=True,
                )
                self._ensure_version_cache(version_idx, apply_flip)
                print(
                    f"✅ [Cache] Epoch {next_epoch} 的磁盘分片已提前就绪。",
                    flush=True,
                )
            except BaseException as error:
                self.cache_prefetch_error = error
                print(
                    f"❌ [Cache] Epoch {next_epoch} 后台准备失败: {error}",
                    flush=True,
                )

        self.cache_prefetch_thread = threading.Thread(
            target=prepare,
            name=f"gpt-cache-epoch-{next_epoch}",
            daemon=True,
        )
        self.cache_prefetch_thread.start()
        return True

    def _cache_signature(self, apply_flip):
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "source_layout": "packed_csv",
            "cache_sequences_per_shard": CACHE_SEQUENCES_PER_SHARD,
            # Empty input tokens extend only the model vocabulary; serialized
            # nine-field records retain their existing IDs and cache format.
            "token_layout_version": self.config.DATA_TOKEN_LAYOUT_VERSION,
            "context_steps": self.config.CONTEXT_STEPS,
            "context_tokens": self.config.CONTEXT_TOKENS,
            "apply_flip": bool(apply_flip),
            "canvas_size": self.config.CANVAS_SIZE,
            "max_shape_size": self.config.MAX_SHAPE_SIZE,
            "attribute_vocab_size": TOKEN_LAYOUT.attribute_vocab_size,
            "input_vocab_size": TOKEN_LAYOUT.vocab_size,
        }

    def _cache_variant_dir(self, version_idx, apply_flip):
        safe_layout_version = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", self.config.DATA_TOKEN_LAYOUT_VERSION
        )
        variant_name = (
            f"{safe_layout_version}_steps{self.config.CONTEXT_STEPS}_"
            f"flip{int(apply_flip)}_cache{CACHE_FORMAT_VERSION}"
        )
        return (
            Path(self.config.DATA_CACHE_DIR)
            / f"v{version_idx}"
            / variant_name
        )

    @staticmethod
    def _cache_paths(cache_dir, shard_index):
        base_name = f"shard_{int(shard_index):05d}"
        return (
            cache_dir / f"{base_name}.tokens.bin",
            cache_dir / f"{base_name}.metadata.json",
        )

    def _read_valid_metadata(
        self, token_path, metadata_path, signature, source_fingerprint
    ):
        if not token_path.is_file() or not metadata_path.is_file():
            return None
        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            for key, expected_value in signature.items():
                if metadata.get(key) != expected_value:
                    return None
            if (
                metadata.get("source_fingerprint") != source_fingerprint
                or metadata.get("context_tokens") != self.config.CONTEXT_TOKENS
                or metadata.get("dtype") != CACHE_DTYPE.str
            ):
                return None
            sequence_count = int(metadata["sequence_count"])
            expected_bytes = (
                sequence_count
                * self.config.CONTEXT_TOKENS
                * CACHE_DTYPE.itemsize
            )
            if token_path.stat().st_size != expected_bytes:
                return None
            metadata["token_path"] = str(token_path)
            metadata["metadata_path"] = str(metadata_path)
            return metadata
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _validate_packed_dataset_root(data_root):
        if not data_root.is_dir():
            raise FileNotFoundError(f"❌ 数据目录不存在: {data_root}")

        manifest_path = data_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"❌ 新版大 CSV 数据集必须包含 manifest.json: {manifest_path}"
            )

        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"❌ 无法读取数据集 manifest: {manifest_path}: {error}"
            ) from error

        if manifest.get("format") != "geometrize_512_three_stage_interrupt_v1":
            raise ValueError(
                f"❌ 不兼容的数据格式: {manifest.get('format')!r}"
            )

        storage_layout = manifest.get("storage_layout")
        if storage_layout not in PACKED_STORAGE_LAYOUTS:
            raise ValueError(
                f"❌ dataset.py 只接受新版大 CSV 布局 {sorted(PACKED_STORAGE_LAYOUTS)}，"
                f"实际 storage_layout={storage_layout!r}"
            )

        expected_header = ",".join(CSV_DTYPE)
        manifest_header = manifest.get("csv_header")
        if manifest_header is not None and manifest_header != expected_header:
            raise ValueError(
                "❌ manifest 中 csv_header 与训练程序要求不一致。"
            )

        if storage_layout == "merged_csv_shards_v1" and manifest.get("complete") is not True:
            raise ValueError("❌ merged 数据集尚未标记 complete=true。")

        return manifest

    @staticmethod
    def _discover_packed_source(version_dir):
        checkpoint_path = version_dir / ".completed_images.txt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"❌ 大 CSV 版本缺少完成清单: {checkpoint_path}"
            )

        for marker_name in (
            ".csv_write_transaction",
            ".completed_images.dirty",
        ):
            marker = version_dir / marker_name
            if marker.exists():
                raise RuntimeError(
                    f"❌ 发现未完成写盘事务 {marker}；请先让生成器完成恢复。"
                )

        parts = []
        unexpected_csv = []
        for path in version_dir.iterdir():
            if not path.is_file():
                continue

            match = PACKED_PART_RE.fullmatch(path.name)
            if match:
                parts.append((int(match.group(1)), path))
            elif path.suffix.lower() == ".csv":
                unexpected_csv.append(path.name)

        if unexpected_csv:
            raise ValueError(
                f"❌ {version_dir} 含非 data_part_N.csv 文件: {unexpected_csv[:8]}"
            )

        parts.sort(key=lambda item: item[0])
        if not parts:
            raise FileNotFoundError(f"❌ 找不到大 CSV 分片: {version_dir}")

        expected_indices = list(range(1, len(parts) + 1))
        actual_indices = [index for index, _ in parts]
        if actual_indices != expected_indices:
            raise ValueError(
                f"❌ {version_dir} 的 data_part_N.csv 编号必须从 1 连续递增，"
                f"实际为 {actual_indices[:16]}"
            )

        expected_header = ",".join(CSV_DTYPE)
        part_records = []

        for index, path in parts:
            before = path.stat()

            with path.open("rb") as stream:
                first_line = stream.readline().decode("utf-8").rstrip("\r\n")
                if before.st_size > 0:
                    stream.seek(-1, os.SEEK_END)
                    final_byte = stream.read(1)
                else:
                    final_byte = b""

            after = path.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise RuntimeError(f"❌ 检查期间 CSV 发生变化: {path}")

            if first_line != expected_header:
                raise ValueError(f"❌ CSV 表头不匹配: {path}")

            if before.st_size <= len(expected_header) + 1:
                raise ValueError(f"❌ CSV 分片没有数据行: {path}")

            if final_byte != b"\n":
                raise ValueError(f"❌ CSV 分片最后一行不完整: {path}")

            part_records.append(
                {
                    "part_index": index,
                    "relative_path": path.name,
                    "csv_path": str(path.resolve()),
                    "csv_size": before.st_size,
                    "csv_mtime_ns": before.st_mtime_ns,
                }
            )

        checkpoint_before = checkpoint_path.stat()
        completed_names, completed_sha256 = _load_completed_names(checkpoint_path)
        checkpoint_after = checkpoint_path.stat()
        if (
            checkpoint_before.st_size != checkpoint_after.st_size
            or checkpoint_before.st_mtime_ns != checkpoint_after.st_mtime_ns
        ):
            raise RuntimeError(
                f"❌ 检查期间完成清单发生变化: {checkpoint_path}"
            )
        checkpoint_stat = checkpoint_after

        descriptor = {
            "parts": part_records,
            "checkpoint": {
                "relative_path": checkpoint_path.name,
                "size": checkpoint_stat.st_size,
                "mtime_ns": checkpoint_stat.st_mtime_ns,
                "sha256": completed_sha256,
                "sequence_count": len(completed_names),
            },
        }

        return descriptor, completed_names

    def _read_committed_cache_manifest(
        self,
        cache_dir,
        signature,
        source_fingerprint,
        expected_sequence_count,
    ):
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            return None

        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)

            for key, expected_value in signature.items():
                if manifest.get(key) != expected_value:
                    return None

            if (
                manifest.get("source_fingerprint") != source_fingerprint
                or int(manifest.get("sequence_count", -1))
                != int(expected_sequence_count)
            ):
                return None

            shards = manifest.get("shards")
            if not isinstance(shards, list) or not shards:
                return None

            metadata_list = []
            total_sequences = 0

            for shard_index, shard_record in enumerate(shards):
                token_path, metadata_path = self._cache_paths(
                    cache_dir, shard_index
                )
                if (
                    shard_record.get("token_file") != token_path.name
                    or shard_record.get("metadata_file") != metadata_path.name
                ):
                    return None

                metadata = self._read_valid_metadata(
                    token_path,
                    metadata_path,
                    signature,
                    source_fingerprint,
                )
                if metadata is None:
                    return None

                if int(shard_record.get("sequence_count", -1)) != int(
                    metadata["sequence_count"]
                ):
                    return None

                total_sequences += int(metadata["sequence_count"])
                metadata_list.append(metadata)

            if total_sequences != int(expected_sequence_count):
                return None

            names_path = cache_dir / "sequence_names.txt"
            if manifest.get("names_file") != names_path.name or not names_path.is_file():
                return None
            if len(names_path.read_text(encoding="utf-8").splitlines()) != total_sequences:
                return None

            return metadata_list

        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    def _ensure_version_cache(self, version_idx, apply_flip):
        version_dir = self.data_root / f"v{version_idx}"
        if not version_dir.is_dir():
            raise FileNotFoundError(f"❌ 找不到数据版本目录: {version_dir}")

        source_descriptor, completed_names = self._discover_packed_source(
            version_dir
        )
        csv_paths = [
            Path(record["csv_path"])
            for record in source_descriptor["parts"]
        ]
        source_fingerprint = _records_fingerprint(source_descriptor)
        expected_sequence_count = len(completed_names)

        if expected_sequence_count == 0:
            raise FileNotFoundError(
                f"❌ 完成清单为空，没有可训练序列: {version_dir}"
            )

        cache_dir = self._cache_variant_dir(version_idx, apply_flip)
        signature = self._cache_signature(apply_flip)

        committed_metadata = self._read_committed_cache_manifest(
            cache_dir,
            signature,
            source_fingerprint,
            expected_sequence_count,
        )
        if committed_metadata is not None:
            return committed_metadata

        cache_dir.mkdir(parents=True, exist_ok=True)

        # 可复用完整的 1000-sequence shard。最后一个不足 1000 的 shard 只有在
        # manifest 完整有效时才直接复用；否则重建它以避免把旧尾巴误当成当前尾巴。
        reusable_full_shards = {}
        max_candidate_shards = (
            expected_sequence_count + CACHE_SEQUENCES_PER_SHARD - 1
        ) // CACHE_SEQUENCES_PER_SHARD

        for shard_index in range(max_candidate_shards):
            token_path, metadata_path = self._cache_paths(
                cache_dir, shard_index
            )
            metadata = self._read_valid_metadata(
                token_path,
                metadata_path,
                signature,
                source_fingerprint,
            )
            if (
                metadata is not None
                and int(metadata["sequence_count"])
                == CACHE_SEQUENCES_PER_SHARD
            ):
                reusable_full_shards[shard_index] = metadata

        print(
            f"🏗️ [Cache] 流式构建 v{version_idx} 大 CSV 缓存 | "
            f"parts={len(csv_paths)} | images={expected_sequence_count} | "
            f"rows_per_chunk={PACKED_CSV_ROWS_PER_CHUNK} | "
            f"sequences_per_shard={CACHE_SEQUENCES_PER_SHARD} | "
            f"可复用完整分片={len(reusable_full_shards)} | 目录={cache_dir}",
            flush=True,
        )

        seen_names = set()
        names_in_source_order = []
        metadata_in_source_order = []
        shard_index = 0
        shard_sequence_count = 0
        shard_first_name = None
        shard_last_name = None

        token_file = None
        temporary_token_path = None
        current_token_path = None
        current_metadata_path = None
        current_reusable_metadata = None
        build_started = time.perf_counter()

        def start_shard():
            nonlocal token_file
            nonlocal temporary_token_path
            nonlocal current_token_path
            nonlocal current_metadata_path
            nonlocal current_reusable_metadata

            current_token_path, current_metadata_path = self._cache_paths(
                cache_dir, shard_index
            )
            current_reusable_metadata = reusable_full_shards.get(shard_index)

            if current_reusable_metadata is not None:
                token_file = None
                temporary_token_path = None
                return

            temporary_token_path = Path(
                str(current_token_path) + f".tmp.{os.getpid()}"
            )
            temporary_token_path.unlink(missing_ok=True)
            token_file = temporary_token_path.open("wb")

        def discard_open_shard():
            nonlocal token_file
            if token_file is not None:
                try:
                    token_file.close()
                finally:
                    token_file = None
            if temporary_token_path is not None:
                temporary_token_path.unlink(missing_ok=True)

        def finish_shard(is_final):
            nonlocal token_file
            nonlocal shard_sequence_count
            nonlocal shard_first_name
            nonlocal shard_last_name

            if shard_sequence_count == 0:
                return

            if current_reusable_metadata is not None:
                if shard_sequence_count != CACHE_SEQUENCES_PER_SHARD:
                    raise RuntimeError(
                        "内部错误：不足 1000 条的尾 shard 不应走复用路径。"
                    )
                metadata = current_reusable_metadata
            else:
                token_file.flush()
                os.fsync(token_file.fileno())
                token_file.close()
                token_file = None

                expected_bytes = (
                    shard_sequence_count
                    * self.config.CONTEXT_TOKENS
                    * CACHE_DTYPE.itemsize
                )
                actual_bytes = temporary_token_path.stat().st_size
                if actual_bytes != expected_bytes:
                    raise RuntimeError(
                        f"缓存大小不一致: {temporary_token_path}，"
                        f"实际 {actual_bytes}，期望 {expected_bytes}"
                    )

                os.replace(temporary_token_path, current_token_path)

                metadata = {
                    **signature,
                    "source_fingerprint": source_fingerprint,
                    "first_source": shard_first_name,
                    "last_source": shard_last_name,
                    "sequence_count": shard_sequence_count,
                    "context_tokens": self.config.CONTEXT_TOKENS,
                    "dtype": CACHE_DTYPE.str,
                    "token_file": current_token_path.name,
                }
                _atomic_json(current_metadata_path, metadata)

                metadata = self._read_valid_metadata(
                    current_token_path,
                    current_metadata_path,
                    signature,
                    source_fingerprint,
                )
                if metadata is None:
                    raise RuntimeError(
                        f"缓存 shard {shard_index} 写入后校验失败。"
                    )

            metadata_in_source_order.append(metadata)

            print(
                f"✅ [Cache] shard_{shard_index:05d} 完成 | "
                f"sequences={shard_sequence_count}"
                + (" | final" if is_final else ""),
                flush=True,
            )

        try:
            start_shard()

            for complete_frame in _iter_packed_csv_frames(csv_paths):
                starts, ends = _group_bounds(complete_frame)
                group_offset = 0

                while group_offset < len(starts):
                    remaining_in_shard = (
                        CACHE_SEQUENCES_PER_SHARD - shard_sequence_count
                    )
                    take = min(
                        remaining_in_shard,
                        len(starts) - group_offset,
                    )

                    first_group = group_offset
                    last_group_exclusive = group_offset + take
                    row_start = int(starts[first_group])
                    row_end = int(ends[last_group_exclusive - 1])
                    subframe = complete_frame.iloc[row_start:row_end]

                    sub_starts, sub_ends = _group_bounds(subframe)
                    sub_names = (
                        subframe["image_name"]
                        .to_numpy(copy=False)[sub_starts]
                        .astype(str)
                        .tolist()
                    )

                    for name in sub_names:
                        if name in seen_names:
                            raise ValueError(
                                f"❌ v{version_idx} 大 CSV 中 image_name 重复: {name}"
                            )
                        seen_names.add(name)
                    names_in_source_order.extend(sub_names)

                    if shard_first_name is None:
                        shard_first_name = sub_names[0]
                    shard_last_name = sub_names[-1]

                    if current_reusable_metadata is None:
                        for start_row, end_row, name in zip(
                            sub_starts.tolist(),
                            sub_ends.tolist(),
                            sub_names,
                        ):
                            _validate_packed_sequence_frame(
                                subframe.iloc[int(start_row):int(end_row)],
                                name,
                                f"v{version_idx}",
                                self.config.MAX_IMAGE_STEPS,
                            )

                        encoded = _encode_complete_groups(
                            subframe,
                            self.config.MAX_IMAGE_STEPS,
                            apply_flip,
                            self.config.CANVAS_SIZE,
                        )
                        if encoded.shape[0] != take:
                            raise RuntimeError(
                                f"❌ v{version_idx} 编码组数 {encoded.shape[0]} != {take}"
                            )
                        bos = np.full((take, TOKENS_PER_STEP), TOKEN_LAYOUT.pad_token, dtype=CACHE_DTYPE)
                        bos[:, 0] = TOKEN_LAYOUT.bos_token
                        encoded = np.concatenate((bos, encoded), axis=1)
                        encoded.tofile(token_file)

                    shard_sequence_count += take
                    group_offset += take

                    if shard_sequence_count == CACHE_SEQUENCES_PER_SHARD:
                        finish_shard(is_final=False)

                        shard_index += 1
                        shard_sequence_count = 0
                        shard_first_name = None
                        shard_last_name = None
                        start_shard()

            if shard_sequence_count > 0:
                # 末 shard 总是重建，因此这里不应命中 reusable_full_shards。
                finish_shard(is_final=True)
                shard_index += 1
                shard_sequence_count = 0
            else:
                # 如果刚好整除 1000，前一个 full shard 已完成；当前刚打开的空临时
                # shard 需要清理。
                discard_open_shard()

        except BaseException:
            discard_open_shard()
            raise

        if seen_names != completed_names:
            missing = sorted(completed_names - seen_names)[:8]
            extra = sorted(seen_names - completed_names)[:8]
            raise RuntimeError(
                f"❌ v{version_idx} CSV 与 .completed_images.txt 不一致："
                f"CSV={len(seen_names)} checkpoint={len(completed_names)} | "
                f"missing={missing} extra={extra}"
            )
        if len(names_in_source_order) != expected_sequence_count:
            raise RuntimeError("缓存的图片名索引与图元序列数量不一致。")

        # 确保缓存构建期间源 CSV/完成清单没有变化。
        final_descriptor, final_completed_names = self._discover_packed_source(
            version_dir
        )
        final_fingerprint = _records_fingerprint(final_descriptor)
        if (
            final_fingerprint != source_fingerprint
            or final_completed_names != completed_names
        ):
            raise RuntimeError(
                f"❌ v{version_idx} 缓存构建期间源大 CSV 发生变化，放弃发布。"
            )

        # 移除本次有效 shard 范围之外遗留的旧 cache 文件。
        valid_shard_count = len(metadata_in_source_order)
        for path in cache_dir.iterdir():
            match = re.fullmatch(
                r"shard_(\d{5})\.(tokens\.bin|metadata\.json)",
                path.name,
            )
            if match and int(match.group(1)) >= valid_shard_count:
                path.unlink(missing_ok=True)

        names_path = cache_dir / "sequence_names.txt"
        temporary_names = names_path.with_name(f".{names_path.name}.tmp.{os.getpid()}")
        with temporary_names.open("w", encoding="utf-8") as stream:
            stream.write("\n".join(names_in_source_order) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_names, names_path)

        manifest = {
            **signature,
            "version": int(version_idx),
            "source_fingerprint": source_fingerprint,
            "sequence_count": len(completed_names),
            "names_file": names_path.name,
            "source_parts": len(csv_paths),
            "shards": [
                {
                    "token_file": Path(metadata["token_path"]).name,
                    "metadata_file": Path(metadata["metadata_path"]).name,
                    "sequence_count": int(metadata["sequence_count"]),
                    "source_fingerprint": metadata["source_fingerprint"],
                }
                for metadata in metadata_in_source_order
            ],
        }

        _atomic_json(cache_dir / "manifest.json", manifest)

        print(
            f"✅ [Cache] v{version_idx} 大 CSV 缓存完成 | "
            f"images={len(completed_names)} | shards={len(metadata_in_source_order)} | "
            f"耗时={time.perf_counter() - build_started:.2f}s",
            flush=True,
        )

        return metadata_in_source_order

    def _open_version_cache(self, version_idx, apply_flip, epoch):
        metadata_list = self._ensure_version_cache(version_idx, apply_flip)

        self.token_shards.clear()
        self.shard_metadata = metadata_list
        gc.collect()

        sequence_counts = np.array(
            [int(metadata["sequence_count"]) for metadata in metadata_list],
            dtype=np.int64,
        )
        self.shard_starts = np.zeros(len(sequence_counts), dtype=np.int64)
        if len(sequence_counts) > 1:
            np.cumsum(sequence_counts[:-1], out=self.shard_starts[1:])
        self.shard_ends = self.shard_starts + sequence_counts
        total_sequence_count = int(sequence_counts.sum())
        self.sequence_names = (self._cache_variant_dir(version_idx, apply_flip) / "sequence_names.txt").read_text(encoding="utf-8").splitlines()
        if len(self.sequence_names) != total_sequence_count:
            raise RuntimeError("图片名索引与图元缓存长度不一致。")
        missing_captions = [name for name in self.sequence_names if name not in self.caption_store.name_to_row]
        if missing_captions:
            raise RuntimeError(f"图元序列缺少对应提示词：{missing_captions[:8]}")
        self.current_epoch = epoch

        for metadata, sequence_count in zip(
            metadata_list, sequence_counts.tolist()
        ):
            token_shard = np.memmap(
                metadata["token_path"],
                dtype=CACHE_DTYPE,
                mode="r",
                shape=(sequence_count, self.config.CONTEXT_TOKENS),
            )
            self.token_shards.append(token_shard)

        visible_sequence_count = total_sequence_count
        if self.config.LIMIT_TRAIN_SEQUENCES > 0:
            visible_sequence_count = min(
                visible_sequence_count,
                self.config.LIMIT_TRAIN_SEQUENCES,
            )
            print(
                f"🛑 [Dataset] 应用全局序列数量限制: "
                f"{visible_sequence_count}/{total_sequence_count}"
            )

        self.sequence_indices = np.arange(
            visible_sequence_count, dtype=np.int32
        )
        np.random.RandomState(epoch * 42).shuffle(self.sequence_indices)

        cache_bytes = sum(
            Path(metadata["token_path"]).stat().st_size
            for metadata in metadata_list
        )
        resident_index_mb = (
            self.sequence_indices.nbytes
            + self.shard_starts.nbytes
            + self.shard_ends.nbytes
        ) / (1024 * 1024)
        print(
            f"📥 [Dataset] mmap 分片已就绪 | 有效图像={visible_sequence_count} | "
            f"分片={len(self.token_shards)} | 磁盘缓存={cache_bytes / 2**30:.2f} GiB"
        )
        print(
            f"⚡ [Dataset] 常驻索引内存约 {resident_index_mb:.2f} MiB；"
            "token 页由操作系统按需读取和回收。"
        )

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        sequence_index = int(self.sequence_indices[idx])
        shard_index = int(
            np.searchsorted(self.shard_ends, sequence_index, side="right")
        )
        if shard_index >= len(self.token_shards):
            raise IndexError(
                f"序列索引 {sequence_index} 超出缓存总量 "
                f"{int(self.shard_ends[-1]) if len(self.shard_ends) else 0}"
            )
        local_index = sequence_index - int(self.shard_starts[shard_index])
        tokens = self.token_shards[shard_index][local_index]
        return torch.tensor(tokens, dtype=torch.long)

    def get_numpy_batch(self, start_index, end_index):
        """纯 NumPy/mmap 批量读取；可安全地在后台线程运行。"""
        batch = np.empty(
            (end_index - start_index, self.config.CONTEXT_TOKENS),
            dtype=np.int64,
        )
        self.fill_numpy_batch(start_index, end_index, batch)
        return batch

    def fill_numpy_batch(self, start_index, end_index, output):
        """将一个 batch 直接写入调用方提供的 NumPy/pinned-memory 视图。"""
        selected_sequence_indices = self.sequence_indices[
            start_index:end_index
        ].astype(np.int64, copy=False)
        expected_shape = (
            len(selected_sequence_indices),
            self.config.CONTEXT_TOKENS,
        )
        if output.shape != expected_shape or output.dtype != np.int64:
            raise ValueError(
                f"output 应为 int64 {expected_shape}，"
                f"实际为 {output.dtype} {output.shape}"
            )
        if len(selected_sequence_indices) == 0:
            return

        shard_indices = np.searchsorted(
            self.shard_ends, selected_sequence_indices, side="right"
        )
        for shard_index in np.unique(shard_indices):
            batch_positions = np.flatnonzero(shard_indices == shard_index)
            local_indices = (
                selected_sequence_indices[batch_positions]
                - self.shard_starts[shard_index]
            )
            output[batch_positions] = self.token_shards[shard_index][
                local_indices
            ]

    def fill_caption_batch(self, start_index, end_index):
        if self.current_epoch is None:
            raise RuntimeError("必须先调用 set_epoch 再读取文字。")
        sequence_indices = self.sequence_indices[start_index:end_index]
        names = [self.sequence_names[int(index)] for index in sequence_indices]
        return self.caption_store.batch(
            names, self.current_epoch, self.config.TEXT_SAMPLING_SEED
        )

    def iter_numpy_batches(self, start_index, batch_size, drop_last):
        """按 dataset 索引顺序产生 NumPy batch，不调用 torch。"""
        if not 0 <= start_index <= len(self):
            raise ValueError(
                f"start_index 必须位于 [0, {len(self)}]，收到 {start_index}"
            )
        for batch_start in range(start_index, len(self), batch_size):
            batch_end = min(batch_start + batch_size, len(self))
            if drop_last and batch_end - batch_start < batch_size:
                break
            yield self.get_numpy_batch(batch_start, batch_end)
