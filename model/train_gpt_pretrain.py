import os

# PyTorch/Inductor 默认在每个 rank 启动最多 32 个编译 worker。
# 三卡 DDP 冷编译会因此耗尽 30 GiB 主机内存。
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
# /tmp/torchinductor_<user> 会在重启或系统清理后丢失。使用
# setdefault 保留用户通过环境变量指定其他高速缓存盘的能力。
_DEFAULT_COMPILE_CACHE_DIR = os.path.join(
    os.environ.get("A800_RUNTIME_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")),
    "compile_cache",
)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _DEFAULT_COMPILE_CACHE_DIR)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    os.path.join(os.environ["TORCHINDUCTOR_CACHE_DIR"], "triton"),
)
# 不强制 ROCBLAS_USE_HIPBLASLT：ROCm 默认会按 GEMM 形状自动选择
# hipBLASLt 或 Tensile。首次懒加载由训练前的多卡并行预热显式完成。

import copy
import ctypes
import gc
import glob
import json
import math
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel

from block_state import ATTRIBUTE_OFFSETS, build_branched_block_batch, canonical_bos_tokens
from config import TrainConfig
from checkpoint_compat import (
    NEW_CHECKPOINT_KIND,
    checkpoint_format,
    checkpoint_metadata,
    load_checkpoint_state,
    load_image_only_initialization,
    migrate_legacy_state,
)
from dataset_gpt_full_sequence import GeometrizeFullSequenceDataset
from fused_multilane import build_fixed_grid_block_mask
from model_gpt import GeometrizeGPT
from gpu_master_adamw import GPUMasterAdamW
from token_layout import DATA_TOKEN_LAYOUT_VERSION, TOKEN_LAYOUT


RESET_OPTIMIZER_STATE = False
CONSOLIDATED_OPTIMIZER_FORMAT = "consolidated_adamw_v1"
CPU_MASTER_SHARDED_FORMAT = "sharded_cpu_fp32_master_adamw_v1"
GPU_MASTER_SHARDED_FORMAT = "sharded_gpu_fp32_master_adamw_v1"
SUPPORTED_MASTER_SHARDED_FORMATS = {CPU_MASTER_SHARDED_FORMAT, GPU_MASTER_SHARDED_FORMAT}

LR_SCHEDULE_BY_EPOCH = [
    [1, 1e-10], [1.25,2e-4],[120, 2.0e-4], [140, 5e-5], [140, 5e-5],
]


class MMapBatchIterator:
    """用两个 pinned buffer 交替执行 mmap 预取与 GPU 消费。"""

    def __init__(
        self,
        dataset,
        start_index,
        batch_size,
        drop_last,
        pin_memory,
        sample_stride=None,
        max_batches=None,
    ):
        self.dataset = dataset
        if not 0 <= start_index <= len(dataset):
            raise ValueError(
                f"start_index 必须位于 [0, {len(dataset)}]，收到 {start_index}"
            )
        self.next_index = start_index
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sample_stride = batch_size if sample_stride is None else sample_stride
        if self.sample_stride < batch_size:
            raise ValueError("sample_stride 不能小于 batch_size。")
        self.max_batches = max_batches
        self.scheduled_batches = 0
        self.last_wait_seconds = 0.0
        self.last_fill_seconds = 0.0
        self.tensors = [
            torch.empty(
                (batch_size, dataset.config.CONTEXT_TOKENS),
                dtype=torch.long,
                pin_memory=pin_memory,
            )
            for _ in range(2)
        ]
        self.numpy_views = [tensor.numpy() for tensor in self.tensors]
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mmap-batch-prefetch",
        )
        self.future = None
        self.future_buffer_index = None
        self.closed = False
        self._schedule(buffer_index=0)

    def _get_next_bounds(self):
        if self.next_index >= len(self.dataset):
            return None
        batch_end = min(
            self.next_index + self.batch_size, len(self.dataset)
        )
        item_count = batch_end - self.next_index
        if self.drop_last and item_count < self.batch_size:
            return None
        return self.next_index, batch_end, item_count

    def _fill_buffer(
        self, buffer_index, batch_start, batch_end, item_count
    ):
        fill_started_at = time.perf_counter()
        self.dataset.fill_numpy_batch(
            batch_start,
            batch_end,
            self.numpy_views[buffer_index][:item_count],
        )
        text_ids, text_mask = self.dataset.fill_caption_batch(batch_start, batch_end)
        return (
            batch_start,
            item_count,
            time.perf_counter() - fill_started_at,
            text_ids,
            text_mask,
        )

    def _schedule(self, buffer_index):
        if self.max_batches is not None and self.scheduled_batches >= self.max_batches:
            self.future = None
            self.future_buffer_index = None
            return
        bounds = self._get_next_bounds()
        if bounds is None:
            self.future = None
            self.future_buffer_index = None
            return
        batch_start, batch_end, item_count = bounds
        self.future_buffer_index = buffer_index
        self.future = self.executor.submit(
            self._fill_buffer,
            buffer_index,
            batch_start,
            batch_end,
            item_count,
        )
        self.scheduled_batches += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.future is None:
            self.close()
            raise StopIteration

        wait_started_at = time.perf_counter()
        batch_start, item_count, self.last_fill_seconds, text_ids, text_mask = self.future.result()
        self.last_wait_seconds = time.perf_counter() - wait_started_at
        ready_buffer_index = self.future_buffer_index
        self.next_index = batch_start + self.sample_stride

        # 当前 buffer 交给训练线程后，只写另一个 buffer。下一次 __next__
        # 发生在当前 Batch 的 H2D/前反向完成后，届时再安全地交换回来。
        self._schedule(buffer_index=1 - ready_buffer_index)
        return (
            self.tensors[ready_buffer_index][:item_count],
            torch.from_numpy(text_ids),
            torch.from_numpy(text_mask),
        )

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass


def validate_data_dir(config):
    """在分配模型/GPU 资源前确认所有 epoch 版本的大 CSV 数据均可读取。"""

    data_dir = os.path.abspath(
        os.path.expanduser(
            os.fspath(config.DATA_DIR)
        )
    )

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"❌ 找不到训练数据根目录: {data_dir}\n"
            "可通过环境变量 ANIME_PAINTER_DATA_DIR 指定三阶段 dataset 目录。"
        )

    # ============================================================
    # 1. 检查根 manifest
    # ============================================================

    manifest_path = os.path.join(
        data_dir,
        "manifest.json",
    )

    try:
        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as stream:
            manifest = json.load(stream)

    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"❌ 无法读取数据 manifest: {manifest_path}"
        ) from error

    if manifest.get("format") != DATA_TOKEN_LAYOUT_VERSION:
        raise RuntimeError(
            f"❌ 数据格式 {manifest.get('format')!r} 不兼容，"
            f"要求 {DATA_TOKEN_LAYOUT_VERSION!r}。"
        )

    # 新版 dataset 支持这两种大 CSV 布局：
    #
    # large_csv_v1
    #     正常生成器直接生成的大 CSV
    #
    # merged_csv_shards_v1
    #     rescued + resume 最终合并后的数据集
    #
    storage_layout = manifest.get("storage_layout")

    supported_layouts = {
        "large_csv_v1",
        "merged_csv_shards_v1",
    }

    if storage_layout not in supported_layouts:
        raise RuntimeError(
            f"❌ 不支持的数据存储布局 {storage_layout!r}；"
            f"要求以下之一: {sorted(supported_layouts)}"
        )

    manifest_versions = manifest.get("versions")

    if manifest_versions is not None:
        try:
            manifest_versions = int(manifest_versions)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"❌ manifest versions 无效: {manifest_versions!r}"
            ) from error

        if manifest_versions < config.VERSION_CYCLE_EPOCHS:
            raise RuntimeError(
                f"❌ 数据集只有 {manifest_versions} 个版本，"
                f"但训练配置需要 {config.VERSION_CYCLE_EPOCHS} 个版本。"
            )

    # merged 数据集只有 complete=true 才允许训练。
    if (
        storage_layout == "merged_csv_shards_v1"
        and manifest.get("complete") is not True
    ):
        raise RuntimeError(
            "❌ merged 数据集尚未标记 complete=true，"
            "拒绝使用未完成的数据集训练。"
        )

    # ============================================================
    # 2. 检查所有训练版本
    # ============================================================

    missing_versions = []
    broken_versions = []

    for version_idx in range(
        1,
        config.VERSION_CYCLE_EPOCHS + 1,
    ):
        version_name = f"v{version_idx}"

        version_dir = os.path.join(
            data_dir,
            version_name,
        )

        if not os.path.isdir(version_dir):
            missing_versions.append(
                version_name
            )
            continue

        # --------------------------------------------------------
        # checkpoint 必须存在。
        #
        # 新版 dataset loader 会使用它确认这个版本完整包含哪些 image_name。
        # --------------------------------------------------------

        checkpoint_path = os.path.join(
            version_dir,
            ".completed_images.txt",
        )

        if not os.path.isfile(checkpoint_path):
            broken_versions.append(
                f"{version_name}: 缺少 .completed_images.txt"
            )
            continue

        # --------------------------------------------------------
        # 如果生成器留下未提交事务标记，说明这个版本可能处于
        # CSV/checkpoint 不一致状态，不能直接训练。
        # --------------------------------------------------------

        unfinished_markers = [
            ".csv_write_transaction",
            ".completed_images.dirty",
        ]

        found_marker = None

        for marker_name in unfinished_markers:
            marker_path = os.path.join(
                version_dir,
                marker_name,
            )

            if os.path.exists(marker_path):
                found_marker = marker_name
                break

        if found_marker is not None:
            broken_versions.append(
                f"{version_name}: 存在未完成写盘标记 {found_marker}"
            )
            continue

        # --------------------------------------------------------
        # 找 data_part_N.csv
        # --------------------------------------------------------

        csv_parts = []

        try:
            with os.scandir(version_dir) as entries:
                for entry in entries:

                    if not entry.is_file():
                        continue

                    name = entry.name

                    if (
                        not name.startswith("data_part_")
                        or not name.endswith(".csv")
                    ):
                        continue

                    number_text = name[
                        len("data_part_"):-len(".csv")
                    ]

                    try:
                        part_index = int(
                            number_text
                        )
                    except ValueError:
                        continue

                    if part_index <= 0:
                        continue

                    csv_parts.append(
                        (
                            part_index,
                            entry.path,
                        )
                    )

        except OSError as error:
            broken_versions.append(
                f"{version_name}: 无法扫描目录 ({error})"
            )
            continue

        if not csv_parts:
            broken_versions.append(
                f"{version_name}: 找不到 data_part_N.csv"
            )
            continue

        csv_parts.sort(
            key=lambda item: item[0]
        )

        # --------------------------------------------------------
        # 分片编号必须是 1,2,3,... 连续。
        # --------------------------------------------------------

        actual_indices = [
            index
            for index, _ in csv_parts
        ]

        expected_indices = list(
            range(
                1,
                len(csv_parts) + 1,
            )
        )

        if actual_indices != expected_indices:
            broken_versions.append(
                f"{version_name}: CSV 分片编号不连续，"
                f"实际={actual_indices[:16]}"
            )
            continue

        # --------------------------------------------------------
        # 快速验证每个 CSV：
        # - 文件非空
        # - 第一行 header 正确
        # - 最后一字节是换行，排除明显的半行写入
        #
        # 不扫描整个 GB 级 CSV。
        # --------------------------------------------------------

        expected_header = (
            "image_name,cx,cy,w,h,shape_type,theta,"
            "r,g,b,is_region_marker,stage,region_id"
        )

        bad_part = None

        for part_index, csv_path in csv_parts:
            try:
                size = os.path.getsize(
                    csv_path
                )

                if size <= len(expected_header) + 1:
                    bad_part = (
                        f"data_part_{part_index}.csv 为空或没有数据"
                    )
                    break

                with open(
                    csv_path,
                    "rb",
                ) as stream:

                    first_line = (
                        stream.readline()
                        .decode("utf-8")
                        .rstrip("\r\n")
                    )

                    if first_line != expected_header:
                        bad_part = (
                            f"data_part_{part_index}.csv 表头不匹配"
                        )
                        break

                    stream.seek(
                        -1,
                        os.SEEK_END,
                    )

                    if stream.read(1) != b"\n":
                        bad_part = (
                            f"data_part_{part_index}.csv "
                            "末行可能未完整写入"
                        )
                        break

            except (
                OSError,
                UnicodeDecodeError,
            ) as error:
                bad_part = (
                    f"data_part_{part_index}.csv 无法读取: {error}"
                )
                break

        if bad_part is not None:
            broken_versions.append(
                f"{version_name}: {bad_part}"
            )

    # ============================================================
    # 3. 汇总报错
    # ============================================================

    if missing_versions:
        raise FileNotFoundError(
            f"❌ 数据目录 {data_dir} 缺少版本目录: "
            f"{', '.join(missing_versions)}"
        )

    if broken_versions:
        preview = "\n".join(
            f"  - {message}"
            for message in broken_versions[:20]
        )

        if len(broken_versions) > 20:
            preview += (
                f"\n  ... 还有 "
                f"{len(broken_versions) - 20} 个问题"
            )

        raise RuntimeError(
            "❌ 大 CSV 数据集完整性检查失败:\n"
            f"{preview}"
        )

    # ============================================================
    # 4. 发布规范化后的 DATA_DIR
    # ============================================================

    config.DATA_DIR = data_dir

    print(
        f"📁 [Dataset] 使用数据目录: {data_dir}"
    )

    print(
        f"✅ [Dataset] 大 CSV 数据布局检查通过 | "
        f"layout={storage_layout} | "
        f"versions={config.VERSION_CYCLE_EPOCHS}"
    )


class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        if device is not None:
            self.ema.to(device=device)
        for parameter in self.ema.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_parameter, model_parameter in zip(self.ema.parameters(), model.parameters()):
            ema_parameter.data.mul_(self.decay).add_(model_parameter.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


def get_current_lr(epoch, batch_idx, batches_per_epoch, global_step, schedule_array):
    current_fractional_epoch = epoch + (batch_idx / batches_per_epoch)
    first_epoch, first_lr = schedule_array[0]
    if current_fractional_epoch < first_epoch:
        return (current_fractional_epoch / first_epoch) * first_lr

    last_epoch, last_lr = schedule_array[-1]
    if current_fractional_epoch >= last_epoch:
        return last_lr

    for index in range(len(schedule_array) - 1):
        epoch_start, lr_start = schedule_array[index]
        epoch_end, lr_end = schedule_array[index + 1]
        if epoch_start <= current_fractional_epoch < epoch_end:
            progress = (current_fractional_epoch - epoch_start) / (epoch_end - epoch_start)
            return lr_start + progress * (lr_end - lr_start)
    return last_lr


def print_model_parameter_count(model):
    """打印单个主模型的参数量；共享参数、EMA 和多卡副本不重复计数。"""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"模型参数量: {total_parameters:,} ({total_parameters / 1e6:.2f} M) | "
        f"可训练参数量: {trainable_parameters:,} "
        f"({trainable_parameters / 1e6:.2f} M)"
    )


def get_effective_accumulation(local_accumulation, device_count):
    """由每卡累计次数计算一次更新的全局 micro-batch 数。"""
    if device_count < 1:
        raise ValueError("device_count 必须至少为 1。")
    if local_accumulation < 1:
        raise ValueError("local_accumulation 必须至少为 1。")
    return local_accumulation * device_count


def get_ckpt_sort_key(file_path):
    match = re.search(r"epoch_(\d+)(?:_batch_(\d+))?\.pt", file_path)
    if match:
        epoch = int(match.group(1))
        batch = int(match.group(2)) if match.group(2) else float("inf")
        return epoch, batch
    return -1, -1


def prune_intermediate_checkpoints(checkpoint_dir, keep=2):
    """Retain every completed epoch and the two newest new-run midpoints.

    Historical Epoch 1 files remain untouched. Temporary files, symlinks,
    and epoch-final checkpoints are never deletion candidates.
    """
    if keep < 1:
        raise ValueError("至少保留一个中途 checkpoint。")
    root = Path(checkpoint_dir).resolve(strict=True)
    candidates = []
    for path in root.iterdir():
        match = re.fullmatch(
            r"geometrize_gpt_pretrain_epoch_(\d+)_batch_(\d+)\.pt",
            path.name,
        )
        if match is None or int(match.group(1)) < 2:
            continue
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            continue
        candidates.append((int(match.group(1)), int(match.group(2)), path))
    candidates.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    for _, _, path in candidates[keep:]:
        try:
            path.unlink()
            print(f"🧹 [CheckpointRotation] 已移除旧中途权重: {path}", flush=True)
        except OSError as error:
            print(f"⚠️ [CheckpointRotation] 无法移除 {path}: {error}", flush=True)


# Control files live on ext4: the Windows data mount cannot enforce private
# permissions for the long-running four-GPU training process.
GUARD_CONTROL_DIR = os.environ.get(
    "V33_TRAINING_CONTROL_DIR",
    os.path.join(os.environ.get("A800_RUNTIME_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")), "control"),
)


def guard_atomic_status(status):
    os.makedirs(GUARD_CONTROL_DIR, mode=0o700, exist_ok=True)
    os.chmod(GUARD_CONTROL_DIR, 0o700)
    path = os.path.join(GUARD_CONTROL_DIR, "status.json")
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(status, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def guard_oom_stop(rank, epoch, batch_idx, device, error):
    """Hold the faulting rank for forensics; never claim in-place DDP recovery."""
    os.makedirs(GUARD_CONTROL_DIR, mode=0o700, exist_ok=True)
    marker = {
        "rank": rank, "pid": os.getpid(), "epoch": epoch,
        "batch_idx": batch_idx, "reason": f"{type(error).__name__}: {error}",
        "utc": time.time(), "torch_memory": cuda_memory_snapshot(device),
    }
    path = os.path.join(GUARD_CONTROL_DIR, f"oom_rank_{rank}.json")
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(marker, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    print(f"[GuardControl] rank={rank} OOM at Epoch={epoch} Batch={batch_idx}; snapshot={path}", flush=True)
    if os.environ.get("A800_FREEZE_ON_OOM", "0") == "1":
        os.kill(os.getpid(), signal.SIGSTOP)
    # A rented CUDA instance has no source-host forensic supervisor. Default
    # to a recorded fatal error so torchrun stops the other rank as well.
    raise RuntimeError("CUDA OOM recorded; restart from the newest complete checkpoint") from error


def guard_read_request(run_id, last_sequence):
    path = os.path.join(GUARD_CONTROL_DIR, "request.json")
    try:
        with open(path, "r", encoding="utf-8") as stream:
            request = json.load(stream)
    except FileNotFoundError:
        return 0, 0
    except (OSError, ValueError) as error:
        print(f"[GuardControl] 忽略无效请求: {error}", flush=True)
        return 0, 0
    if request.get("run_id") != run_id:
        return 0, 0
    try:
        sequence = int(request["sequence"])
    except (KeyError, TypeError, ValueError):
        return 0, 0
    if sequence <= last_sequence:
        return 0, 0
    action = {"save": 1, "pause": 2, "resume": 3}.get(request.get("action"), 0)
    return action, sequence


def guard_external_vram_snapshot(device):
    """Compare all VRAM on this physical card to PyTorch's reserved pool."""
    try:
        if torch.version.hip is None:
            free, total = torch.cuda.mem_get_info(device)
            reserved = torch.cuda.memory_reserved(device)
            return {
                "backend": "cuda_mem_get_info", "device": str(device),
                "used_bytes": total - free, "total_bytes": total,
                "torch_reserved_bytes": reserved,
                "outside_torch_estimate_bytes": max(0, total - free - reserved),
            }
        bus = int(torch.cuda.get_device_properties(device.index).pci_bus_id)
        gpu = f"0000:{bus:02x}:00.0"
        base = f"/sys/bus/pci/devices/{gpu}"
        with open(f"{base}/mem_info_vram_used", encoding="ascii") as stream:
            used = int(stream.read().strip())
        with open(f"{base}/mem_info_vram_total", encoding="ascii") as stream:
            total = int(stream.read().strip())
        reserved = torch.cuda.memory_reserved(device)
        return {
            "bdf": gpu, "used_bytes": used, "total_bytes": total,
            "torch_reserved_bytes": reserved,
            "outside_torch_estimate_bytes": max(0, used - reserved),
        }
    except (OSError, ValueError, AttributeError):
        return None


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module.model
    return model.module if isinstance(model, nn.DataParallel) else model


class MultiLaneTrainingModule(nn.Module):
    """Give DDP a conventional forward while preserving GeometrizeGPT APIs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        full_tokens,
        query_tokens,
        query_block_ids,
        query_fields,
        first_query_block,
        text_ids,
        text_mask,
        fixed_grid=False,
    ):
        if fixed_grid:
            return self.model.training_forward_fixed_grid(
                text_ids, text_mask, full_tokens, query_tokens, query_fields
            )
        return self.model.training_forward_multi_lane(
            full_tokens,
            query_tokens,
            query_block_ids=query_block_ids,
            query_fields=query_fields,
            first_query_block=first_query_block,
            return_termination=True,
            text_ids=text_ids,
            text_mask=text_mask,
        )


def iter_parameter_segment_slices(parameters, segment_start, segment_length):
    """将扁平同步分段映射到各参数内连续的切片。"""
    segment_end = segment_start + segment_length
    parameter_start = 0
    for parameter in parameters:
        parameter_end = parameter_start + parameter.numel()
        overlap_start = max(segment_start, parameter_start)
        overlap_end = min(segment_end, parameter_end)
        if overlap_start < overlap_end:
            yield (
                parameter,
                overlap_start - parameter_start,
                overlap_start - segment_start,
                overlap_end - overlap_start,
            )
        parameter_start = parameter_end


class ManualMultiGPUSynchronizer:
    """主卡更新，分段汇总任意数量副卡的梯度，并向所有副卡回传权重。"""

    def __init__(self, master_model, worker_models, mode, chunk_mb):
        self.master_parameters = list(master_model.parameters())
        self.worker_parameter_groups = [
            list(worker_model.parameters()) for worker_model in worker_models
        ]
        if not self.worker_parameter_groups:
            raise ValueError("手动多卡同步至少需要一个副卡模型。")
        for worker_parameters in self.worker_parameter_groups:
            if len(self.master_parameters) != len(worker_parameters):
                raise RuntimeError("主卡和副卡模型参数数量不一致。")
            if any(
                master.shape != worker.shape
                for master, worker in zip(
                    self.master_parameters, worker_parameters
                )
            ):
                raise RuntimeError("主卡和副卡模型参数形状不一致。")

        self.mode = mode
        self.total_numel = sum(parameter.numel() for parameter in self.master_parameters)
        self.cpu_buffer = None
        self.p2p_buffer = None
        buffer_bytes = chunk_mb * 1024 * 1024
        if mode == "manual_cpu_sync":
            buffer_numel = min(self.total_numel, buffer_bytes // 4)
            if buffer_numel <= 0:
                raise ValueError("CPU 同步缓冲区大小必须至少容纳一个 float32。")
            self.cpu_buffer = torch.empty(buffer_numel, dtype=torch.float32)
        elif mode == "manual_pcie_sync":
            buffer_dtype = self.master_parameters[0].dtype
            buffer_numel = min(
                self.total_numel,
                buffer_bytes // self.master_parameters[0].element_size(),
            )
            if buffer_numel <= 0:
                raise ValueError("P2P 同步缓冲区不能为零。")
            self.p2p_buffer = torch.empty(
                buffer_numel,
                dtype=buffer_dtype,
                device=self.master_parameters[0].device,
            )
        else:
            raise ValueError(f"未知的手动同步模式: {mode}")

    def iter_segments(self):
        if self.cpu_buffer is not None:
            buffer_numel = self.cpu_buffer.numel()
        elif self.p2p_buffer is not None:
            buffer_numel = self.p2p_buffer.numel()
        else:
            buffer_numel = self.total_numel
        for segment_start in range(0, self.total_numel, buffer_numel):
            yield segment_start, min(buffer_numel, self.total_numel - segment_start)

    @torch.no_grad()
    def synchronize_worker_parameters(self):
        """主卡更新后，把新权重同步到所有副卡。"""
        for segment_start, segment_length in self.iter_segments():
            master_slices = list(
                iter_parameter_segment_slices(
                    self.master_parameters, segment_start, segment_length
                )
            )
            if self.mode == "manual_cpu_sync":
                for parameter, parameter_offset, buffer_offset, length in master_slices:
                    source = parameter.detach().reshape(-1).narrow(
                        0, parameter_offset, length
                    )
                    self.cpu_buffer.narrow(0, buffer_offset, length).copy_(source)

            for worker_parameters in self.worker_parameter_groups:
                worker_slices = list(
                    iter_parameter_segment_slices(
                        worker_parameters, segment_start, segment_length
                    )
                )
                if self.mode == "manual_cpu_sync":
                    for parameter, parameter_offset, buffer_offset, length in worker_slices:
                        destination = parameter.detach().reshape(-1).narrow(
                            0, parameter_offset, length
                        )
                        source = self.cpu_buffer.narrow(
                            0, buffer_offset, length
                        ).to(device=destination.device, dtype=destination.dtype)
                        destination.copy_(source)
                else:
                    for master_slice, worker_slice in zip(
                        master_slices, worker_slices
                    ):
                        master, master_offset, _, length = master_slice
                        worker, worker_offset, _, _ = worker_slice
                        source = master.detach().reshape(-1).narrow(
                            0, master_offset, length
                        )
                        destination = worker.detach().reshape(-1).narrow(
                            0, worker_offset, length
                        )
                        # peer-access 可用时由 HIP 执行 GPU0 -> 副卡 PCIe DMA。
                        destination.copy_(source, non_blocking=True)

    @torch.no_grad()
    def average_worker_gradients_into_master(self):
        """所有卡的梯度等权平均；只在每轮本地累计完成后调用一次。"""
        model_count = 1 + len(self.worker_parameter_groups)
        for segment_start, segment_length in self.iter_segments():
            master_slices = list(
                iter_parameter_segment_slices(
                    self.master_parameters, segment_start, segment_length
                )
            )
            for parameter, _, _, _ in master_slices:
                if parameter.grad is None:
                    raise RuntimeError("主卡存在未参与反向传播的参数。")

            for worker_parameters in self.worker_parameter_groups:
                worker_slices = list(
                    iter_parameter_segment_slices(
                        worker_parameters, segment_start, segment_length
                    )
                )
                sync_buffer = (
                    self.cpu_buffer
                    if self.mode == "manual_cpu_sync"
                    else self.p2p_buffer
                )
                for parameter, parameter_offset, buffer_offset, length in worker_slices:
                    if parameter.grad is None:
                        raise RuntimeError("副卡存在未参与反向传播的参数。")
                    worker_grad = parameter.grad.detach().reshape(-1).narrow(
                        0, parameter_offset, length
                    )
                    sync_buffer.narrow(0, buffer_offset, length).copy_(
                        worker_grad,
                        non_blocking=self.mode == "manual_pcie_sync",
                    )
                for parameter, parameter_offset, buffer_offset, length in master_slices:
                    master_grad = parameter.grad.detach().reshape(-1).narrow(
                        0, parameter_offset, length
                    )
                    worker_grad = sync_buffer.narrow(
                        0, buffer_offset, length
                    ).to(device=master_grad.device, dtype=master_grad.dtype)
                    master_grad.add_(worker_grad)

            for parameter, parameter_offset, _, length in master_slices:
                parameter.grad.detach().reshape(-1).narrow(
                    0, parameter_offset, length
                ).div_(model_count)


def require_gpus(device_ids):
    if not torch.cuda.is_available():
        raise RuntimeError("多卡模式需要 CUDA/HIP GPU。")
    if torch.cuda.device_count() <= max(device_ids):
        raise RuntimeError(
            f"检测到 {torch.cuda.device_count()} 张 GPU，但设备编号 {device_ids} 不可用。"
        )


def _parse_linux_cpu_list(cpu_list):
    cpus = set()
    for part in cpu_list.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus


def bind_current_process_to_gpu_numa(device_id):
    """Pin one DDP rank to CPUs local to its GPU's PCIe NUMA node."""
    if os.name == "nt" or not hasattr(os, "sched_setaffinity"):
        return None
    properties = torch.cuda.get_device_properties(device_id)
    try:
        pci_address = (
            f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
            f"{properties.pci_device_id:02x}.0"
        )
    except (AttributeError, TypeError):
        return None
    device_path = os.path.join("/sys/bus/pci/devices", pci_address)
    numa_path = os.path.join(device_path, "numa_node")
    try:
        with open(numa_path, "r", encoding="utf-8") as numa_file:
            numa_node = int(numa_file.read().strip())
        if numa_node < 0:
            return None
        cpu_list_path = (
            f"/sys/devices/system/node/node{numa_node}/cpulist"
        )
        with open(cpu_list_path, "r", encoding="utf-8") as cpu_list_file:
            local_cpus = _parse_linux_cpu_list(cpu_list_file.read())
        # Respect an outer taskset/cgroup restriction rather than broadening it.
        allowed_cpus = set(os.sched_getaffinity(0))
        target_cpus = local_cpus & allowed_cpus
        if not target_cpus:
            return None
        os.sched_setaffinity(0, target_cpus)
        return pci_address, numa_node, tuple(sorted(target_cpus))
    except (OSError, ValueError):
        return None


def peer_access_available(device_ids):
    if not hasattr(torch.cuda, "can_device_access_peer"):
        return False
    try:
        master_device_id = device_ids[0]
        return all(
            torch.cuda.can_device_access_peer(master_device_id, worker_device_id)
            and torch.cuda.can_device_access_peer(worker_device_id, master_device_id)
            for worker_device_id in device_ids[1:]
        )
    except RuntimeError:
        return False


def report_rocm_power_profile(device_ids):
    """提示会让 RDNA3 训练吞吐剧烈波动的 Linux 自动功耗档。"""
    if torch.version.hip is None or os.name == "nt":
        return
    level_paths = sorted(
        glob.glob(
            "/sys/class/drm/card*/device/"
            "power_dpm_force_performance_level"
        )
    )
    levels = []
    for level_path in level_paths:
        try:
            with open(level_path, "r", encoding="utf-8") as level_file:
                levels.append(level_file.read().strip())
        except OSError:
            continue
    if levels and any(level.lower() == "auto" for level in levels):
        device_id_arguments = " ".join(str(device_id) for device_id in device_ids)
        print(
            "⚠️ [ROCm Power] 检测到 AMDGPU 性能档仍为 AUTO。"
            "本机实测训练时 GFXCLK 会跌到约 100 MHz，造成窗口从 "
            "57s 波动到 60/66/99s。建议训练前执行：\n"
            "    sudo amd-smi set -P COMPUTE_MASK -l HIGH -g "
            f"{device_id_arguments}\n"
            "训练结束可恢复：\n"
            "    sudo amd-smi set -P BOOTUP_DEFAULT -l AUTO -g "
            f"{device_id_arguments}"
        )


def require_rocm_numa_balancing_disabled():
    """避免自动 NUMA 页迁移驱逐 KFD/HSA 计算队列。"""
    if torch.version.hip is None or os.name == "nt":
        return
    numa_balancing_path = "/proc/sys/kernel/numa_balancing"
    try:
        with open(
            numa_balancing_path, "r", encoding="utf-8"
        ) as balancing_file:
            balancing_enabled = balancing_file.read().strip() != "0"
    except OSError as error:
        print(
            "⚠️ [ROCm NUMA] 无法读取 "
            f"{numa_balancing_path}: {error}"
        )
        return
    if balancing_enabled:
        raise RuntimeError(
            "检测到 kernel.numa_balancing=1。当前训练进程已经出现多卡 "
            "KFD 队列被驱逐约 728 秒；自动 NUMA 页迁移会使 ROCm GPU "
            "在前向/反向中无响应，直到 userptr/SVM 恢复。\n"
            "请先执行：sudo sysctl kernel.numa_balancing=0\n"
            "确认命令：cat /proc/sys/kernel/numa_balancing  # 应输出 0"
        )


class KFDEvictionMonitor:
    """读取当前进程每个 KFD 节点累计的队列驱逐毫秒数。"""

    def __init__(self):
        self.previous = self._read()

    @staticmethod
    def _read():
        values = {}
        pattern = (
            f"/sys/class/kfd/kfd/proc/{os.getpid()}/"
            "stats_*/evicted_ms"
        )
        for evicted_path in glob.glob(pattern):
            node_id = os.path.basename(
                os.path.dirname(evicted_path)
            ).removeprefix("stats_")
            try:
                with open(
                    evicted_path, "r", encoding="utf-8"
                ) as evicted_file:
                    values[node_id] = int(evicted_file.read().strip())
            except (OSError, ValueError):
                continue
        return values

    def consume_deltas(self):
        current = self._read()
        deltas = {
            node_id: value - self.previous.get(node_id, value)
            for node_id, value in current.items()
            if value > self.previous.get(node_id, value)
        }
        self.previous = current
        return deltas


@dataclass(frozen=True)
class RandKBatchStats:
    partial_histogram: torch.Tensor
    sampled_target_count: torch.Tensor
    padding_count: torch.Tensor
    query_position_count: int
    true_end_count: torch.Tensor
    interrupt_end_count: torch.Tensor
    supervised_termination_count: torch.Tensor
    attribute_loss: torch.Tensor
    termination_loss: torch.Tensor


@dataclass
class NonFiniteEventCounter:
    """Track skipped numerical failures and enforce a bounded retry policy."""

    max_consecutive: int
    max_per_epoch: int
    consecutive: int = 0
    epoch_total: int = 0

    def record_failure(self):
        self.consecutive += 1
        self.epoch_total += 1
        return (
            self.consecutive >= self.max_consecutive
            or self.epoch_total >= self.max_per_epoch
        )

    def record_successful_update(self):
        self.consecutive = 0


def synchronize_nonfinite_flag(local_nonfinite, device, is_ddp):
    """Return True on every rank when any rank reports a numerical failure."""
    flag = torch.tensor(
        int(bool(local_nonfinite)), dtype=torch.uint8, device=device
    )
    if is_ddp:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def parse_nonfinite_injection_spec(spec):
    """Parse optional EPOCH:BATCH:RANK diagnostic NaN injection."""
    spec = str(spec or "").strip()
    if not spec:
        return None
    try:
        epoch, batch, rank = (int(value) for value in spec.split(":"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ANIME_PAINTER_TEST_NONFINITE 必须是 EPOCH:BATCH:RANK，"
            f"实际为 {spec!r}。"
        ) from error
    if epoch < 0 or batch < 0 or rank < 0:
        raise ValueError("数值防护注入的 epoch、batch、rank 必须为非负整数。")
    return epoch, batch, rank


def complete_discarded_ddp_backward(
    scaled_loss,
    scaler,
    is_ddp,
    backward_already_run,
):
    """Finish DDP reducer state and release saved activations for a bad loss."""
    if backward_already_run or not is_ddp:
        return False
    if scaled_loss is None:
        raise RuntimeError("DDP 非有限 loss 分支缺少 scaled_loss。")
    scaler.scale(scaled_loss).backward()
    # Keep GradScaler state identical across ranks even when only one local
    # loss was non-finite and this accumulation step used no_sync.
    if scaler.is_enabled():
        scaler.update(
            new_scale=scaler.get_scale() * scaler.get_backoff_factor()
        )
    return True


def cuda_memory_snapshot(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "reserved_mib": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "peak_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
    }


def release_cuda_caches(devices):
    """Synchronize discarded work but retain reusable allocator cache.

    Keep the historical name for compatibility with numerical-guard tests.
    Eagerly emptying the cache after a NaN caused near-capacity VRAM churn
    during the very next fixed-shape training batch.
    """
    if not torch.cuda.is_available():
        return
    unique_devices = []
    for device in devices:
        device = torch.device(device)
        if device.type == "cuda" and device not in unique_devices:
            unique_devices.append(device)
    for device in unique_devices:
        torch.cuda.synchronize(device)


def floating_tensors_from_optimizer(optimizer):
    """Yield the local AdamW shard tensors without consolidating ZeRO-1."""
    local_optimizer = (
        optimizer.optim
        if isinstance(optimizer, ZeroRedundancyOptimizer)
        else optimizer
    )
    for state in local_optimizer.state.values():
        for state_name, value in state.items():
            # AdamW's scalar step may live on CPU while its moments live on GPU;
            # it is an integer-like counter and not part of the numerical stream.
            if state_name == "step":
                continue
            if torch.is_tensor(value) and value.is_floating_point():
                yield value


def collection_total_norm(tensors, device):
    tensors = [tensor for tensor in tensors if tensor is not None]
    if not tensors:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.nn.utils.get_total_norm(
        tensors,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=None,
    )


def local_training_state_norms(model, optimizer, device):
    """Cheap finite-state sentinels for model parameters and local Adam state."""
    parameter_norm = collection_total_norm(model.parameters(), device)
    optimizer_norm = collection_total_norm(
        floating_tensors_from_optimizer(optimizer), device
    )
    # Both GPU state and the legacy CPU diagnostic path produce a scalar;
    # keep the sentinel on the device used by the distributed finite check.
    return parameter_norm.to(device), optimizer_norm.to(device)


def nonfinite_named_tensors(named_tensors, limit=12):
    """List a bounded set of exact bad tensors; used only after a failure."""
    bad_names = []
    for name, tensor in named_tensors:
        if tensor is None or not tensor.is_floating_point():
            continue
        if not bool(torch.isfinite(tensor).all()):
            bad_names.append(name)
            if len(bad_names) >= limit:
                break
    return bad_names


def nonfinite_parameter_names(model, gradients=False, limit=12):
    named_tensors = (
        (name, parameter.grad)
        if gradients
        else (name, parameter)
        for name, parameter in model.named_parameters()
    )
    return nonfinite_named_tensors(named_tensors, limit=limit)


def nonfinite_optimizer_state_names(optimizer, limit=12):
    local_optimizer = (
        optimizer.optim
        if isinstance(optimizer, ZeroRedundancyOptimizer)
        else optimizer
    )
    parameter_indices = {}
    for group in local_optimizer.param_groups:
        for parameter in group["params"]:
            parameter_indices[parameter] = len(parameter_indices)
    named_tensors = (
        (f"param_{parameter_indices.get(parameter, -1)}.{state_name}", value)
        for parameter, state in local_optimizer.state.items()
        for state_name, value in state.items()
        if torch.is_tensor(value)
    )
    return nonfinite_named_tensors(named_tensors, limit=limit)


def validate_checkpoint_scalars(smoothed_loss, epoch_total_loss):
    values = {
        "smoothed_loss": smoothed_loss,
        "epoch_total_loss": epoch_total_loss,
    }
    bad = [
        name
        for name, value in values.items()
        if value is None or not math.isfinite(float(value))
    ]
    if bad:
        raise FloatingPointError(
            "拒绝保存非有限 checkpoint 标量: " + ", ".join(bad)
        )


def derive_rand_k_sampling_seed(base_seed, epoch, batch_idx, device_rank):
    """Stable splitmix64 seed; independent of Python hash and RNG state."""
    mask = (1 << 64) - 1
    value = int(base_seed) & mask
    for component in (epoch, batch_idx, device_rank):
        value = (value + 0x9E3779B97F4A7C15 + (int(component) & mask)) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
    return value & ((1 << 63) - 1)


def fit_geometry_to_total_context(input_ids, text_ids, config, text_mask=None):
    """Fit geometry independently for each caption, then batch-pad geometry."""
    if text_mask is None:
        text_mask = torch.ones_like(text_ids, dtype=torch.bool)
    text_lengths = text_mask.sum(dim=1).tolist()
    image_budgets = [config.TOTAL_CONTEXT_TOKENS - int(length) for length in text_lengths]
    if min(image_budgets) <= config.PREFIX_STEPS:
        raise ValueError("提示词过长，没有剩余图元预测位置。")
    primitive_tokens = input_ids.reshape(input_ids.size(0), -1, config.TOKENS_PER_STEP)
    output = torch.full(
        (input_ids.size(0), max(image_budgets), config.TOKENS_PER_STEP),
        TOKEN_LAYOUT.pad_token, dtype=primitive_tokens.dtype, device=primitive_tokens.device,
    )
    for batch_index, budget in enumerate(image_budgets):
        take = min(primitive_tokens.size(1), budget)
        output[batch_index, :take] = primitive_tokens[batch_index, :take]
    return output


@dataclass(frozen=True)
class FixedGridTrainingInputs:
    geometry_tokens: torch.Tensor
    query_tokens: torch.Tensor
    query_fields: torch.Tensor
    targets: torch.Tensor
    termination_targets: torch.Tensor
    termination_supervised_mask: torch.Tensor
    text_ids: torch.Tensor
    text_mask: torch.Tensor


def make_fixed_grid_training_inputs(input_ids, branched, text_ids, text_mask, config):
    """Preserve legacy valid queries while adding masked text/PAD slots."""
    context = config.TOTAL_CONTEXT_TOKENS
    batch_size = input_ids.size(0)
    lanes = branched.query_tokens.size(2)
    text_lengths = text_mask.sum(dim=1)
    if bool(torch.any((text_lengths < 2) | (text_lengths > config.TEXT_MAX_TOKENS))):
        raise ValueError("文字前缀长度必须包含 BOS/EOS 且不超过 TEXT_MAX_TOKENS。")
    if text_ids.size(1) > config.TEXT_MAX_TOKENS:
        raise ValueError("文字 batch 宽度超过 TEXT_MAX_TOKENS。")
    text_pad = config.TEXT_MAX_TOKENS - text_ids.size(1)
    text_ids_fixed = torch.nn.functional.pad(text_ids, (0, text_pad), value=0)
    text_mask_fixed = torch.nn.functional.pad(text_mask, (0, text_pad), value=False)

    raw_geometry = input_ids.reshape(batch_size, -1, config.TOKENS_PER_STEP)
    geometry_tokens = torch.full(
        (batch_size, context - 2, config.TOKENS_PER_STEP),
        TOKEN_LAYOUT.pad_token, dtype=raw_geometry.dtype, device=raw_geometry.device,
    )
    take = min(raw_geometry.size(1), context - 2)
    geometry_tokens[:, :take] = raw_geometry[:, :take]
    query_tokens = torch.full(
        (batch_size, context - 1, lanes, config.TOKENS_PER_STEP),
        TOKEN_LAYOUT.pad_token, dtype=raw_geometry.dtype, device=raw_geometry.device,
    )
    query_fields = torch.zeros((batch_size, context - 1, lanes), dtype=torch.long, device=input_ids.device)
    targets = torch.full((batch_size, context - 1, lanes), -100, dtype=torch.long, device=input_ids.device)
    termination_targets = torch.full((batch_size, context - 1), -100, dtype=torch.long, device=input_ids.device)
    termination_mask = torch.zeros((batch_size, context - 1), dtype=torch.bool, device=input_ids.device)
    for sample in range(batch_size):
        positions = text_lengths[sample] + branched.query_block_ids
        in_context = positions < context
        slots = positions[in_context] - 1
        query_tokens[sample, slots] = branched.query_tokens[sample, in_context]
        query_fields[sample, slots] = branched.query_fields[sample, in_context]
        targets[sample, slots] = branched.targets[sample, in_context]
        valid_termination = branched.termination_supervised_mask[sample, in_context]
        termination_mask[sample, slots] = valid_termination
        termination_targets[sample, slots] = branched.termination_targets[sample, in_context].masked_fill(
            ~valid_termination, -100
        )
    return FixedGridTrainingInputs(
        geometry_tokens, query_tokens, query_fields, targets,
        termination_targets, termination_mask, text_ids_fixed, text_mask_fixed,
    )


def compute_loss(
    model,
    input_ids,
    config,
    criterion,
    sampling_seed=0,
    return_sampling_stats=False,
    return_loss_components=False,
    text_ids=None,
    text_mask=None,
):
    if text_ids is None:
        text_ids = torch.tensor([2, 3], dtype=torch.long, device=input_ids.device).expand(input_ids.size(0), -1)
        text_mask = torch.ones_like(text_ids, dtype=torch.bool)
    primitive_tokens = fit_geometry_to_total_context(input_ids, text_ids, config, text_mask)
    branched = build_branched_block_batch(
        primitive_tokens,
        config.PREFIX_STEPS,
        rand_k=config.RAND_K,
        sampling_seed=sampling_seed,
        sampling_mode=config.RAND_K_SAMPLING,
    )
    if getattr(config, "FIXED_QUERY_GRID", False):
        fixed = make_fixed_grid_training_inputs(input_ids, branched, text_ids, text_mask, config)
        if isinstance(model, DistributedDataParallel):
            query_logits, termination_logits = model(
                fixed.geometry_tokens, fixed.query_tokens, None,
                fixed.query_fields, 0, fixed.text_ids, fixed.text_mask, True,
            )
        else:
            query_logits, termination_logits = model.training_forward_fixed_grid(
                fixed.text_ids, fixed.text_mask, fixed.geometry_tokens,
                fixed.query_tokens, fixed.query_fields,
            )
        targets = fixed.targets
        termination_mask = fixed.termination_supervised_mask
        termination_targets = fixed.termination_targets
    else:
        if isinstance(model, DistributedDataParallel):
            query_logits, termination_logits = model(
                branched.full_tokens,
                branched.query_tokens,
                branched.query_block_ids,
                branched.query_fields,
                branched.first_query_block,
                text_ids,
                text_mask,
            )
        else:
            query_logits, termination_logits = model.training_forward_multi_lane(
                branched.full_tokens,
                branched.query_tokens,
                query_block_ids=branched.query_block_ids,
                query_fields=branched.query_fields,
                first_query_block=branched.first_query_block,
                return_termination=True,
                text_ids=text_ids,
                text_mask=text_mask,
            )
        targets = branched.targets
        termination_mask = branched.termination_supervised_mask
        termination_targets = branched.termination_targets.masked_fill(
            ~termination_mask, -100
        )
    attribute_loss = criterion(
        query_logits.reshape(-1, query_logits.size(-1)),
        targets.reshape(-1),
    )
    termination_loss = nn.functional.cross_entropy(
        termination_logits.reshape(-1, 2),
        termination_targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / termination_mask.sum().clamp_min(1)
    loss = attribute_loss + termination_loss
    if not return_sampling_stats:
        if return_loss_components:
            return loss, attribute_loss.detach(), termination_loss.detach()
        return loss
    sampled_ids = branched.query_partial_ids[branched.query_valid_mask]
    histogram = torch.bincount(
        sampled_ids, minlength=config.BLOCK_VALUE_COUNT
    )
    sampled_count = branched.query_valid_mask.sum()
    stats = RandKBatchStats(
        partial_histogram=histogram,
        sampled_target_count=sampled_count,
        padding_count=branched.query_valid_mask.numel() - sampled_count,
        query_position_count=branched.query_valid_mask.numel(),
        true_end_count=branched.marker_end_mask.sum(),
        interrupt_end_count=branched.interrupt_end_mask.sum(),
        supervised_termination_count=termination_mask.sum(),
        attribute_loss=attribute_loss.detach(),
        termination_loss=termination_loss.detach(),
    )
    return loss, stats


def build_warmup_tokens(batch_size, config, device):
    """Construct valid ordinary primitives without depending on dataset I/O."""
    primitive = torch.tensor(
        ATTRIBUTE_OFFSETS, dtype=torch.long, device=device
    ).view(1, 1, -1)
    tokens = primitive.expand(batch_size, config.CONTEXT_STEPS, -1).clone()
    tokens[:, 0] = canonical_bos_tokens(batch_size, device)
    return tokens.reshape(batch_size, config.CONTEXT_TOKENS)


def warm_up_training_backend(
    config,
    model,
    master_model,
    worker_models,
    primary_device,
    device_ids,
    dtype,
    criterion,
    autocast_enabled,
    is_manual_parallel,
    is_data_parallel,
):
    """在统计训练窗口前加载 Linear/SDPA 的前向与反向 GPU kernel。"""
    if primary_device.type != "cuda":
        return

    print(
        "🔥 [Backend Warmup] 开始预热；仅执行前向/反向，不更新模型参数..."
    )
    for device_id in (
        device_ids if (is_manual_parallel or is_data_parallel) else [primary_device]
    ):
        torch.cuda.synchronize(device_id)
    warmup_started_at = time.perf_counter()

    def warm_up_one(target_model, device, batch_size):
        torch.cuda.set_device(device)
        input_ids = build_warmup_tokens(batch_size, config, device)
        started_at = time.perf_counter()
        with torch.amp.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            loss = compute_loss(target_model, input_ids, config, criterion)
        loss.backward()
        torch.cuda.synchronize(device)
        target_model.zero_grad(set_to_none=True)
        return time.perf_counter() - started_at

    if is_manual_parallel:
        # 各卡的 hipBLASLt 上下文相互独立；并行预热可重叠代码对象加载。
        warmup_models = [master_model, *worker_models]
        with ThreadPoolExecutor(
            max_workers=len(warmup_models), thread_name_prefix="rocm-warmup"
        ) as executor:
            futures = [
                executor.submit(
                    warm_up_one,
                    target_model,
                    torch.device(f"cuda:{device_id}"),
                    config.MICRO_BATCH_SIZE,
                )
                for target_model, device_id in zip(
                    warmup_models, device_ids
                )
            ]
            device_times = [future.result() for future in futures]
    else:
        warmup_batch_size = config.MICRO_BATCH_SIZE * (
            len(device_ids) if is_data_parallel else 1
        )
        device_times = [
            warm_up_one(model, primary_device, warmup_batch_size)
        ]

    print(
        f"✅ [ROCm Warmup] 后端预热完成 | wall="
        f"{time.perf_counter() - warmup_started_at:.3f}s | "
        f"device_times={[round(value, 3) for value in device_times]} | "
        "参数未更新"
    )


def run_manual_gpu_microbatch(
    model,
    host_input_ids,
    device,
    config,
    criterion,
    dtype,
    autocast_enabled,
    local_accum_steps,
    sampling_seed,
):
    """在固定 GPU 工作线程上完成一个 micro-batch 的前向和反向。"""
    torch.cuda.set_device(device)
    input_ids = host_input_ids.to(device, non_blocking=True)
    with torch.amp.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        loss, attribute_loss, termination_loss = compute_loss(
            model,
            input_ids,
            config,
            criterion,
            sampling_seed=sampling_seed,
            return_loss_components=True,
        )
        scaled_loss = loss / local_accum_steps
    scaled_loss.backward()
    # 这只保证当前 pinned buffer 可安全复用，不传递梯度/参数。
    # 真正的跨卡同步仍只在完整 local_accum_steps 窗口末尾发生一次。
    return (
        loss.detach().item(),
        attribute_loss.item(),
        termination_loss.item(),
    )


def save_checkpoint(path, epoch, batch_idx, global_step, smoothed_loss, epoch_total_loss,
                    model, ema, optimizer, scaler, parallel_mode, config,
                    optimizer_state_dict=None, attribute_loss=None,
                    termination_loss=None,
                    optimizer_state_format=CONSOLIDATED_OPTIMIZER_FORMAT):
    validate_checkpoint_scalars(smoothed_loss, epoch_total_loss)
    saved_world_size = (
        optimizer.world_size
        if isinstance(optimizer, ZeroRedundancyOptimizer)
        else 1
    )
    checkpoint = {
        **checkpoint_metadata(config),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "smoothed_loss": smoothed_loss,
        "epoch_total_loss": epoch_total_loss,
        "attribute_loss": attribute_loss,
        "termination_loss": termination_loss,
        "total_loss": smoothed_loss,
        "parallel_mode": parallel_mode,
        "prefix_steps": config.PREFIX_STEPS,
        "prediction_steps": config.PREDICTION_STEPS,
        "context_steps": config.CONTEXT_STEPS,
        "context_blocks": config.CONTEXT_BLOCKS,
        "max_block_positions": config.MAX_BLOCK_POSITIONS,
        "fixed_query_grid": bool(getattr(config, "FIXED_QUERY_GRID", False)),
        "model_d_model": config.MODEL_D_MODEL,
        "model_n_layer": config.MODEL_N_LAYER,
        "model_n_head": config.MODEL_N_HEAD,
        "model_ffn_hidden": config.MODEL_FFN_HIDDEN,
        "model_norm_eps": config.MODEL_NORM_EPS,
        "model_rope_base": config.MODEL_ROPE_BASE,
        "position_encoding": "rope_text_prefix_then_primitive_id",
        "block_primitives": config.BLOCK_PRIMITIVES,
        "canvas_size": config.CANVAS_SIZE,
        "max_shape_size": config.MAX_SHAPE_SIZE,
        "model_state_dict": unwrap_model(model).state_dict(),
        "ema_enabled": ema is not None,
        "optimizer_state_format": optimizer_state_format,
        # 仅用于诊断；consolidated state 不绑定保存时的卡数。
        "optimizer_saved_world_size": saved_world_size,
        "gradient_accumulation_steps_per_device": (
            config.GRADIENT_ACCUMULATION_STEPS
        ),
        "micro_batch_size_per_device": config.MICRO_BATCH_SIZE,
        "effective_global_batch_size": (
            config.MICRO_BATCH_SIZE * saved_world_size * config.GRADIENT_ACCUMULATION_STEPS
        ),
        "numerical_guard_enabled": config.ENABLE_NUMERICAL_GUARD,
        "nonfinite_max_consecutive": config.NONFINITE_MAX_CONSECUTIVE,
        "nonfinite_max_per_epoch": config.NONFINITE_MAX_PER_EPOCH,
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer_state_dict is None
            else optimizer_state_dict
        ),
        "scaler_state_dict": scaler.state_dict(),
    }
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()
    temporary_path = f"{path}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def clear_zero_redundancy_transient_state(optimizer):
    """释放 ZeRO-1 恢复所需、但训练时不使用的外层状态。"""
    if not isinstance(optimizer, ZeroRedundancyOptimizer):
        return
    # 兼容手动或旧路径调用 consolidate_state_dict() 后的暂存。
    optimizer._all_state_dicts.clear()
    # load_state_dict() 还会在 wrapper.state 中放一份本 rank state；
    # 实际 step() 只使用 optimizer.optim.state。
    optimizer.state.clear()


def zero_optimizer_shard_path(checkpoint_path, rank):
    return f"{checkpoint_path}.zero1_rank_{rank}.tmp"


def build_consolidated_zero_state_dict(optimizer, checkpoint_path):
    """由 mmap 本地 shard 构建标准、与 world size 无关的 AdamW state。"""
    if not isinstance(optimizer, ZeroRedundancyOptimizer):
        raise TypeError("optimizer 必须是 ZeroRedundancyOptimizer")
    if optimizer.rank != 0:
        raise RuntimeError("只能在 ZeRO rank 0 上构建完整 state")

    # 外层 optimizer 持有全局 param_groups，正常训练时 state 为空。
    consolidated = torch.optim.Optimizer.state_dict(optimizer)
    consolidated["state"] = {}
    partitioned_groups = optimizer._partition_parameters()
    mapped_shards = []
    for shard_rank in range(optimizer.world_size):
        shard = load_checkpoint_file(
            zero_optimizer_shard_path(checkpoint_path, shard_rank)
        )
        mapped_shards.append(shard)
        local_groups = shard["param_groups"]
        global_groups = partitioned_groups[shard_rank]
        if len(local_groups) != len(global_groups):
            raise RuntimeError(
                f"ZeRO rank {shard_rank} 的 optimizer param group 数不匹配"
            )
        for local_group, global_group in zip(local_groups, global_groups):
            local_indices = local_group["params"]
            global_parameters = global_group["params"]
            if len(local_indices) != len(global_parameters):
                raise RuntimeError(
                    f"ZeRO rank {shard_rank} 的 optimizer 参数分区不匹配"
                )
            for local_index, global_parameter in zip(
                local_indices, global_parameters
            ):
                if local_index in shard["state"]:
                    global_index = optimizer._param_to_index[global_parameter]
                    consolidated["state"][global_index] = shard["state"][
                        local_index
                    ]
    consolidated["state"] = dict(sorted(consolidated["state"].items()))
    # 必须让 mapped_shards 活到 torch.save() 结束，否则 mmap storage
    # 的生命周期将变得不明确。
    return consolidated, mapped_shards


def remove_zero_optimizer_shards(checkpoint_path, world_size):
    for shard_rank in range(world_size):
        shard_path = zero_optimizer_shard_path(checkpoint_path, shard_rank)
        try:
            os.remove(shard_path)
        except FileNotFoundError:
            pass


def training_checkpoint_path(config, epoch, batch_idx=None):
    stem = f"geometrize_gpt_pretrain_epoch_{epoch}"
    if batch_idx is not None:
        stem += f"_batch_{batch_idx}"
    suffix = ".ckpt" if config.SHARDED_GPU_MASTER_CHECKPOINTS else ".pt"
    return os.path.join(config.CKPT_DIR, stem + suffix)


def next_intermediate_save_batch(resume_batch_idx, interval):
    """Return the next one-based micro-batch count strictly after a save."""
    if interval <= 0 or resume_batch_idx < -1:
        raise ValueError("Invalid intermediate checkpoint interval or resume index")
    return ((resume_batch_idx + 1) // interval + 1) * interval


def completed_sharded_checkpoints(checkpoint_root, expected_world_size=2):
    """Ignore incomplete checkpoint directories; return verified ready ones."""
    candidates = []
    root = Path(checkpoint_root)
    if not root.is_dir():
        return candidates
    for directory in root.glob("geometrize_gpt_pretrain_epoch_*.ckpt"):
        if directory.is_symlink() or not directory.is_dir():
            continue
        manifest_path = directory / "READY.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("schema_version") != 1
                    or manifest.get("checkpoint_format") not in SUPPORTED_MASTER_SHARDED_FORMATS
                    or manifest.get("world_size") != expected_world_size):
                continue
            shard_names = [
                f"optimizer_rank_{rank}.pt" for rank in range(expected_world_size)
            ]
            if manifest["optimizer_shards"] != shard_names:
                continue
            names = [manifest["model"], *shard_names]
            if len(names) != expected_world_size + 1 or len(set(names)) != len(names):
                continue
            if any(Path(name).name != name or not (directory / name).is_file()
                   for name in names):
                continue
            sizes = manifest.get("file_sizes", {})
            if any((directory / name).stat().st_size != sizes.get(name)
                   for name in names):
                continue
            epoch = int(manifest["epoch"])
            batch_idx = int(manifest["batch_idx"])
            expected_name = f"geometrize_gpt_pretrain_epoch_{epoch}"
            if batch_idx >= 0:
                expected_name += f"_batch_{batch_idx}"
            if epoch < 1 or batch_idx < -1 or directory.name != expected_name + ".ckpt":
                continue
            ordering = (epoch, batch_idx == -1, batch_idx)
            candidates.append((ordering, directory, manifest))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def reserve_three_a800_checkpoint_slot(root, world_size):
    """Keep one previous version while staging next; ACK gate final Epochs."""
    if os.environ.get("POP_REQUIRE_ARCHIVE_ACK", "0") != "1":
        return  # Local/open-source default: retain complete epoch checkpoints.
    root = Path(root).resolve()
    if not os.environ.get("V33_TRAINING_CONTROL_DIR"):
        raise ValueError("POP_REQUIRE_ARCHIVE_ACK=1 requires V33_TRAINING_CONTROL_DIR")
    ack_root = Path(os.environ['V33_TRAINING_CONTROL_DIR']) / 'archive_acks' 
    while True:
        complete = completed_sharded_checkpoints(root, expected_world_size=world_size)
        if len(complete) < 2: return
        _, oldest, manifest = complete[-1]
        if int(manifest['batch_idx']) < 0:
            try:
                ack = json.loads((ack_root / (oldest.name + '.json')).read_text())
                acknowledged = ack.get('checkpoint') == oldest.name and ack.get('file_sizes') == manifest['file_sizes']
            except (OSError, ValueError): acknowledged = False
            if not acknowledged:
                print(f'[ArchiveWait] oldest Epoch has no archive ACK: {oldest}', flush=True)
                time.sleep(30); continue
        if oldest.is_symlink() or oldest.resolve().parent != root:
            raise RuntimeError('Unsafe checkpoint retention target.')
        shutil.rmtree(oldest)
        print(f'[CheckpointRetention] removed archived/old relay: {oldest.name}', flush=True)


def prune_sharded_intermediate_checkpoints(
    checkpoint_root, keep=2, expected_world_size=4
):
    """Keep only newest complete intermediate dirs; never touch Epoch finals."""
    if keep < 1:
        raise ValueError("At least one complete intermediate must remain")
    root = Path(checkpoint_root).resolve()
    intermediates = [
        (directory, manifest)
        for _, directory, manifest in completed_sharded_checkpoints(
            root, expected_world_size=expected_world_size
        )
        if int(manifest["batch_idx"]) >= 0
    ]
    for directory, manifest in intermediates[keep:]:
        if (directory.is_symlink() or directory.resolve().parent != root
                or not re.fullmatch(
                    r"geometrize_gpt_pretrain_epoch_\d+_batch_\d+\.ckpt",
                    directory.name,
                )
                or int(manifest["batch_idx"]) < 0):
            raise RuntimeError(f"Refusing unsafe checkpoint rotation: {directory}")
        shutil.rmtree(directory)
        print(f"[CheckpointRotation] 已移除最旧中途分片目录: {directory}", flush=True)


def wait_for_sharded_checkpoint_space(root, is_main_process, device, is_ddp):
    """Pause together at a safe update boundary until checkpoint headroom is free."""
    minimum_bytes = 16 * 1024 ** 3  # 13.1 GiB complete checkpoint + safety margin.
    while True:
        free = shutil.disk_usage(root).free if is_main_process else 0
        ready = torch.tensor(int(free >= minimum_bytes),
                             dtype=torch.uint8, device=device)
        if is_ddp:
            dist.broadcast(ready, src=0)
        if bool(ready.item()):
            return
        if is_main_process:
            print(
                "[Checkpoint] 磁盘余量低于 16 GiB，已在安全更新点暂停；"
                f"当前空闲 {free / 1024**3:.2f} GiB。请清理空间，保留全部历史权重。",
                flush=True,
            )
        time.sleep(30)


def save_gpu_master_sharded_checkpoint(
    path, epoch, batch_idx, global_step, smoothed_loss, epoch_total_loss,
    model, ema, optimizer, scaler, parallel_mode, config, is_main_process,
    attribute_loss=None, termination_loss=None,
):
    if not isinstance(optimizer, ZeroRedundancyOptimizer) or parallel_mode != "ddp":
        raise RuntimeError("GPU master checkpoint requires ZeRO-1 DDP")
    target = Path(path)
    root = Path(config.CKPT_DIR).resolve()
    if target.suffix != ".ckpt" or target.parent.resolve() != root or target.is_symlink():
        raise RuntimeError(f"Unsafe checkpoint directory: {target}")
    device = next(unwrap_model(model).parameters()).device
    if target.exists():
        raise FileExistsError(f"Completed checkpoint already exists: {target}")
    # A third intermediate would exceed the two-version limit. Retain the
    # newest previous copy while staging this one; standard Epoch finals are
    # never pruned. The incomplete new directory is never a resume target.
    if is_main_process:
        reserve_three_a800_checkpoint_slot(root, optimizer.world_size)
    dist.barrier()
    wait_for_sharded_checkpoint_space(root, is_main_process, device, True)
    rank0_pid = torch.tensor([os.getpid() if is_main_process else 0],
                             dtype=torch.int64, device=device)
    dist.broadcast(rank0_pid, src=0)
    temporary = Path(f"{target}.incomplete.{int(rank0_pid.item())}")
    if is_main_process:
        temporary.mkdir(mode=0o700, parents=False, exist_ok=False)
    dist.barrier()
    names = [f"optimizer_rank_{rank}.pt" for rank in range(optimizer.world_size)]
    for rank, name in enumerate(names):
        if optimizer.rank == rank:
            temporary_file = temporary / f"{name}.tmp"
            torch.save(optimizer.optim.state_dict(), temporary_file)
            os.replace(temporary_file, temporary / name)
            with (temporary / name).open("r+b") as stream:
                os.fsync(stream.fileno())
        dist.barrier()
    if is_main_process:
        save_checkpoint(
            str(temporary / "model.pt"), epoch, batch_idx, global_step,
            smoothed_loss, epoch_total_loss, model, ema, optimizer, scaler,
            parallel_mode, config,
            optimizer_state_dict={"sharded_optimizer": True},
            optimizer_state_format=GPU_MASTER_SHARDED_FORMAT,
            attribute_loss=attribute_loss,
            termination_loss=termination_loss,
        )
        with (temporary / "model.pt").open("r+b") as stream:
            os.fsync(stream.fileno())
        all_names = ["model.pt", *names]
        manifest = {
            "schema_version": 1,
            "checkpoint_format": GPU_MASTER_SHARDED_FORMAT,
            "epoch": epoch, "batch_idx": batch_idx,
            "global_step": global_step, "world_size": optimizer.world_size,
            "model": "model.pt", "optimizer_shards": names,
            "file_sizes": {
                name: (temporary / name).stat().st_size for name in all_names
            },
        }
        ready_tmp = temporary / "READY.json.tmp"
        with ready_tmp.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(ready_tmp, temporary / "READY.json")
        os.replace(temporary, target)
        if batch_idx >= 0:
            prune_sharded_intermediate_checkpoints(
                root, keep=2, expected_world_size=optimizer.world_size
            )
        print(
            f"[Checkpoint] {optimizer.world_size} rank 分片目录已完整发布: {target}",
            flush=True,
        )
    dist.barrier()


def save_training_checkpoint(
    path,
    epoch,
    batch_idx,
    global_step,
    smoothed_loss,
    epoch_total_loss,
    model,
    ema,
    optimizer,
    scaler,
    parallel_mode,
    config,
    is_main_process,
    attribute_loss=None,
    termination_loss=None,
):
    """在安全更新点保存与 world size 无关的单文件 checkpoint。"""
    is_ddp = parallel_mode == "ddp"
    uses_zero = isinstance(optimizer, ZeroRedundancyOptimizer)
    validate_checkpoint_scalars(smoothed_loss, epoch_total_loss)
    checkpoint_model = unwrap_model(model)
    checkpoint_device = next(checkpoint_model.parameters()).device
    parameter_norm, optimizer_norm = local_training_state_norms(
        checkpoint_model, optimizer, checkpoint_device
    )
    local_nonfinite = not bool(
        torch.isfinite(parameter_norm) & torch.isfinite(optimizer_norm)
    )
    if synchronize_nonfinite_flag(
        local_nonfinite, checkpoint_device, is_ddp
    ):
        bad_parameters = nonfinite_parameter_names(checkpoint_model)
        bad_optimizer = nonfinite_optimizer_state_names(optimizer)
        raise FloatingPointError(
            "拒绝保存含 NaN/Inf 的 checkpoint | "
            f"rank={dist.get_rank() if is_ddp else 0} | "
            f"parameter_norm={parameter_norm.item()} | "
            f"optimizer_norm={optimizer_norm.item()} | "
            f"bad_parameters={bad_parameters} | "
            f"bad_optimizer={bad_optimizer}"
        )
    if is_ddp:
        dist.barrier()
    if getattr(config, "SHARDED_GPU_MASTER_CHECKPOINTS", False):
        save_gpu_master_sharded_checkpoint(
            path, epoch, batch_idx, global_step, smoothed_loss,
            epoch_total_loss, model, ema, optimizer, scaler,
            parallel_mode, config, is_main_process,
            attribute_loss=attribute_loss,
            termination_loss=termination_loss,
        )
        return
    optimizer_state_dict = None
    mapped_shards = []
    if uses_zero:
        # 官方 consolidate_state_dict() 会同时构造 BytesIO、bytearray
        # 和 GPU byte tensor；1.3B AdamW 在 30 GiB 主机内存上会 OOM。
        # 各 rank 串行写临时 shard，rank 0 再通过 mmap 流式合并。
        for shard_rank in range(optimizer.world_size):
            if optimizer.rank == shard_rank:
                torch.save(
                    optimizer.optim.state_dict(),
                    zero_optimizer_shard_path(path, shard_rank),
                )
            dist.barrier()
        if is_main_process:
            optimizer_state_dict, mapped_shards = (
                build_consolidated_zero_state_dict(optimizer, path)
            )
    try:
        if is_main_process:
            save_checkpoint(
                path,
                epoch,
                batch_idx,
                global_step,
                smoothed_loss,
                epoch_total_loss,
                model,
                ema,
                optimizer,
                scaler,
                parallel_mode,
                config,
                optimizer_state_dict=optimizer_state_dict,
                attribute_loss=attribute_loss,
                termination_loss=termination_loss,
            )
    finally:
        optimizer_state_dict = None
        mapped_shards.clear()
        clear_zero_redundancy_transient_state(optimizer)
        if is_main_process:
            gc.collect()
            if uses_zero:
                remove_zero_optimizer_shards(path, optimizer.world_size)
    if is_ddp:
        dist.barrier()


def load_checkpoint_file(path):
    """使 DDP 进程共享文件页，避免各自复制整个 CPU checkpoint。"""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except (RuntimeError, ValueError):
        # 兼容旧式非 zip checkpoint。
        return torch.load(path, map_location="cpu", weights_only=False)


def load_local_gpu_master_shard(checkpoint_dir, optimizer, model, device):
    """Restore a CPU- or GPU-saved shard to its owning GPU without BF16 casts."""
    for loading_rank in range(optimizer.world_size):
        if optimizer.rank == loading_rank:
            shard_path = Path(checkpoint_dir) / f"optimizer_rank_{loading_rank}.pt"
            shard = load_checkpoint_file(str(shard_path))
            optimizer.optim.load_state_dict(shard)
            del shard
            gc.collect()
            if sys.platform.startswith("linux"):
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            local_parameters = [p for group in optimizer.optim.param_groups
                                for p in group["params"]]
            for parameter in local_parameters:
                state = optimizer.optim.state.get(parameter, {})
                for name in ("master_param", "exp_avg", "exp_avg_sq"):
                    value = state.get(name)
                    if (value is None or value.device != parameter.device
                            or value.dtype != torch.float32
                            or value.shape != parameter.shape):
                        raise RuntimeError(
                            f"rank {loading_rank} checkpoint {name} 缺失或精度错误"
                        )
                if not torch.equal(
                    state["master_param"].to(device=device, dtype=torch.bfloat16),
                    parameter.detach(),
                ):
                    raise RuntimeError("恢复的 GPU FP32 master 与 BF16 工作权重不一致")
            print(
                f"[A800 Resume] rank={loading_rank} GPU FP32 master/Adam 分片已恢复",
                flush=True,
            )
        dist.barrier()


def align_resume_batch(checkpoint, batch_idx, epoch_total_loss, world_size, config):
    """Preserve consumed-image position and global batch when micro-batch changes."""
    saved_world = int(checkpoint["optimizer_saved_world_size"])
    if saved_world != world_size and checkpoint.get("optimizer_state_format") in SUPPORTED_MASTER_SHARDED_FORMATS:
        raise RuntimeError("Sharded checkpoint rank count differs; explicit resharding is required.")
    old_micro = checkpoint.get("micro_batch_size_per_device")
    if old_micro is None:
        raise RuntimeError("Checkpoint is missing its saved micro-batch size; use an explicit migration, never guess the image position.")
    old_micro = int(old_micro)
    old_accumulation = int(checkpoint["gradient_accumulation_steps_per_device"])
    old_global = old_micro * saved_world * old_accumulation
    new_global = config.MICRO_BATCH_SIZE * world_size * config.GRADIENT_ACCUMULATION_STEPS
    if old_global != new_global:
        raise RuntimeError(f"Effective global batch changed: {old_global} -> {new_global}.")
    if batch_idx < 0 or (old_micro == config.MICRO_BATCH_SIZE and saved_world == world_size):
        return batch_idx, epoch_total_loss
    consumed = (batch_idx + 1) * saved_world * old_micro
    new_micro_global = world_size * config.MICRO_BATCH_SIZE
    if consumed % new_micro_global:
        raise RuntimeError("Saved image position does not align with the requested micro-batch.")
    return consumed // new_micro_global - 1, epoch_total_loss * (old_micro * saved_world) / new_micro_global


def run_training():
    config = TrainConfig()
    config.validate_sequence_config()
    config.validate_model_config()
    config.validate_parallel_config()
    parallel_mode = config.PARALLEL_MODE
    is_manual_parallel = parallel_mode in {"manual_cpu_sync", "manual_pcie_sync"}
    is_data_parallel = parallel_mode == "data_parallel"
    is_ddp = parallel_mode == "ddp"
    is_multi_gpu = is_manual_parallel or is_data_parallel or is_ddp
    gpu_master_enabled = bool(config.GPU_FP32_MASTER_WEIGHTS)
    if gpu_master_enabled and (not is_ddp or not config.SHARDED_GPU_MASTER_CHECKPOINTS):
        raise RuntimeError("GPU FP32 master requires DDP and sharded checkpoints")

    distributed_rank = 0
    local_rank = 0
    distributed_world_size = 1
    if is_ddp:
        missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ]
        if missing:
            raise RuntimeError(
                "DDP 训练必须由 torchrun 启动，缺少环境变量: "
                + ", ".join(missing)
            )
        distributed_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        distributed_world_size = int(os.environ["WORLD_SIZE"])

    validate_data_dir(config)
    if parallel_mode not in {"single", "ddp"}:
        raise RuntimeError("文字前缀训练目前只支持 single 或 DDP；拒绝在其它模式下丢失文字条件。")

    device_ids = list(config.PARALLEL_DEVICE_IDS) if is_multi_gpu else []
    if is_multi_gpu:
        require_gpus(device_ids)
        if is_ddp:
            if distributed_world_size != len(device_ids):
                raise RuntimeError(
                    f"torchrun WORLD_SIZE={distributed_world_size}，但配置了 "
                    f"{len(device_ids)} 张卡。"
                )
            if not 0 <= local_rank < len(device_ids):
                raise RuntimeError("LOCAL_RANK 超出 PARALLEL_DEVICE_IDS。")
        require_rocm_numa_balancing_disabled()
        if distributed_rank == 0:
            report_rocm_power_profile(device_ids)
        if parallel_mode == "manual_pcie_sync" and not peer_access_available(device_ids):
            raise RuntimeError(
                "manual_pcie_sync 需要主卡与每张副卡双向互访；"
                "当前不可用，请改用 manual_cpu_sync。"
            )
        if parallel_mode == "manual_pcie_sync":
            print(
                "🔗 [P2P] 主卡与所有副卡的双向 GPU peer-access 已确认；"
                "梯度与参数通过 PCIe DMA 直传，CPU staging 已禁用。"
            )
        primary_device_id = device_ids[local_rank] if is_ddp else device_ids[0]
        primary_device = torch.device(f"cuda:{primary_device_id}")
        torch.cuda.set_device(primary_device)
        if is_ddp:
            numa_binding = bind_current_process_to_gpu_numa(primary_device_id)
            if numa_binding is None:
                print(
                    f"⚠️ [DDP/NUMA] rank={distributed_rank} 无法自动绑定 "
                    f"cuda:{primary_device_id} 的本地 CPU；继续使用继承的 affinity。"
                )
            else:
                pci_address, numa_node, bound_cpus = numa_binding
                print(
                    f"🧭 [DDP/NUMA] rank={distributed_rank} cuda:{primary_device_id} "
                    f"PCI={pci_address} NUMA={numa_node} CPUs={bound_cpus}"
                )
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                # 串行冷编译时其他 rank 会在 barrier 等待。
                timeout=timedelta(minutes=60),
            )
    else:
        primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    is_main_process = distributed_rank == 0
    nonfinite_injection = parse_nonfinite_injection_spec(
        os.environ.get("ANIME_PAINTER_TEST_NONFINITE")
    )
    if nonfinite_injection is not None:
        if is_manual_parallel:
            raise RuntimeError(
                "ANIME_PAINTER_TEST_NONFINITE 诊断注入仅支持 single、"
                "data_parallel 或 DDP。"
            )
        if nonfinite_injection[2] >= distributed_world_size:
            raise ValueError(
                "数值防护注入 rank 超出 WORLD_SIZE："
                f"{nonfinite_injection[2]} >= {distributed_world_size}。"
            )
        if is_main_process:
            print(
                "🧪 [NumericalGuard] 已启用一次性 NaN 注入："
                f"epoch={nonfinite_injection[0]} "
                f"batch={nonfinite_injection[1]} "
                f"rank={nonfinite_injection[2]}",
                flush=True,
            )

    print(
        f"========== 启动文字前缀 + 图元 BOS 条件训练 | 并行模式: {parallel_mode} | "
        f"GPU 数: {torch.cuda.device_count()} | 使用设备: "
        f"{device_ids if is_multi_gpu else [primary_device]} =========="
    )
    print(
        f"序列配置: 前缀 {config.PREFIX_STEPS} 步 + 预测 {config.PREDICTION_STEPS} 步 "
        f"= 上下文 {config.CONTEXT_STEPS} 步 ({config.CONTEXT_TOKENS} tokens) | "
        f"压缩为 {config.CONTEXT_BLOCKS} 个单图元 token | "
        f"branched Q={config.BRANCH_QUERY_BLOCKS}×{config.RAND_K} "
        f"(rand_K={config.RAND_K}, {config.RAND_K_SAMPLING}, replacement=True)，"
        "历史 C 每层只投影一次"
    )
    if is_main_process:
        print(
            "🛡️ 数值防护: "
            f"enabled={config.ENABLE_NUMERICAL_GUARD} | "
            "nonfinite窗口先跳过，"
            f"连续{config.NONFINITE_MAX_CONSECUTIVE}次或单轮"
            f"{config.NONFINITE_MAX_PER_EPOCH}次后停止 | "
            "checkpoint保存前强制有限性检查"
        )
    if torch.version.hip is not None:
        print(
            f"ROCm 启动配置: MIOPEN_FIND_MODE="
            f"{os.environ.get('MIOPEN_FIND_MODE')} | "
            f"ROCBLAS_USE_HIPBLASLT="
            f"{os.environ.get('ROCBLAS_USE_HIPBLASLT', 'AUTO')} | "
            "本模型主体为 Linear/SDPA，主要后端是 rocBLAS/AOTriton。"
        )
    if is_main_process:
        os.makedirs(config.CKPT_DIR, exist_ok=True)
    if is_ddp:
        dist.barrier()

    epochs = int(os.environ.get('POP_EPOCHS', os.environ.get('A800_TARGET_EPOCHS', '40')))  # Resume complete checkpoints; configurable target.
    weight_decay = 2e-2
    start_epoch = 1
    resume_batch_idx = -1
    global_step = 0
    smoothed_loss = None
    smoothed_attribute_loss = None
    smoothed_termination_loss = None
    resume_total_loss = 0.0

    dataset = GeometrizeFullSequenceDataset(config=config)
    # Four concurrent FP32 1B model builds can exhaust the 30 GiB host.
    # Build ranks 0/1 together, then 2/3. Each rank retains only its ZeRO-1
    # GPU FP32 master shard after copying BF16 working weights to its GPU.
    if gpu_master_enabled:
        initialization_parallelism = min(distributed_world_size, max(1, int(os.environ.get("POP_INITIALIZATION_PARALLELISM", "1"))))  # Bound per-rank CPU model construction on ordinary hosts.
        initialization_group = distributed_rank // initialization_parallelism
        initialization_group_count = (
            distributed_world_size + initialization_parallelism - 1
        ) // initialization_parallelism
        for _ in range(initialization_group):
            dist.barrier()
        # ZeRO owns disjoint CPU master shards before DDP's initial weight
        # broadcast. All ranks must therefore start from identical FP32 bits.
        torch.manual_seed(20261001)
    master_model = GeometrizeGPT(
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
    )
    if not gpu_master_enabled:
        master_model = master_model.to(primary_device)

    checkpoint = None
    checkpoint_kind = None
    checkpoint_path = None
    checkpoint_is_initialization = False
    checkpoint_dir = None
    resume_source = None
    initialization_source = os.environ.get("POP_INIT_WEIGHTS")
    if initialization_source:
        if not Path(initialization_source).is_file():
            raise FileNotFoundError(initialization_source)
        checkpoint_path = initialization_source
        checkpoint_is_initialization = True
    elif gpu_master_enabled:
        complete = completed_sharded_checkpoints(
            config.CKPT_DIR, expected_world_size=distributed_world_size
        )
        imported_root = getattr(config, "RESUME_CKPT_DIR", "")
        if imported_root and os.path.abspath(imported_root) != os.path.abspath(config.CKPT_DIR):
            complete.extend(completed_sharded_checkpoints(imported_root, expected_world_size=distributed_world_size))
            complete.sort(key=lambda item: item[0], reverse=True)
        if complete:
            _, checkpoint_dir, _ = complete[0]
            checkpoint_path = str(checkpoint_dir / "model.pt")
        elif config.RANDOM_INITIALIZATION_IF_NO_CHECKPOINT:
            print("[V3.3] 新 1B 架构从 CPU FP32 随机初始化；旧 3.12 亿权重不加载。",
                  flush=True)
        else:
            raise FileNotFoundError("没有完整的双卡 FP32-master 分片 checkpoint")
    else:
        ckpt_files = glob.glob(os.path.join(config.CKPT_DIR, "geometrize_gpt_pretrain_epoch_*.pt"))
        resume_source = os.environ.get("V32_RESUME_CHECKPOINT_PATH")
        if ckpt_files:
            checkpoint_path = max(ckpt_files, key=get_ckpt_sort_key)
        elif resume_source:
            if not os.path.isfile(resume_source):
                raise FileNotFoundError(f"指定的三卡恢复 checkpoint 不存在: {resume_source}")
            checkpoint_path = resume_source
        elif config.RANDOM_INITIALIZATION_IF_NO_CHECKPOINT:
            checkpoint_path = None
        elif os.path.isfile(config.IMAGE_INIT_CHECKPOINT_PATH):
            checkpoint_path = config.IMAGE_INIT_CHECKPOINT_PATH
            checkpoint_is_initialization = True
        else:
            raise FileNotFoundError(
                f"找不到文字训练断点，也找不到图元初始化权重：{config.IMAGE_INIT_CHECKPOINT_PATH}"
            )
    if checkpoint_path is not None:
        print(f"--> 正在读取 checkpoint：{checkpoint_path}...")
        checkpoint = load_checkpoint_file(checkpoint_path)
        is_image_only_source = os.path.abspath(checkpoint_path) == os.path.abspath(
            config.IMAGE_INIT_CHECKPOINT_PATH
        )
        if is_image_only_source:
            checkpoint_kind = "image_only_initialization"
            checkpoint_is_initialization = True
            load_image_only_initialization(master_model, checkpoint, config)
            print("✅ 已迁移图元主体权重；文字 embedding 随机初始化，优化器状态不复用。")
        else:
            if os.environ.get("POP_BEZIER_INIT") == "1":
                from checkpoint_compat import load_bezier_initialization
                load_bezier_initialization(
                    master_model, checkpoint, "model_state_dict",
                )
                checkpoint_kind = "bezier_edge_initialization"
                print(
                    "[bezier] 已按 7069→8097 词表映射迁移权重；"
                    "edge0..3 与 edge empty 零初始化。",
                    flush=True,
                )
            else:
                checkpoint_kind = checkpoint_format(checkpoint, config)
                load_checkpoint_state(
                    master_model, checkpoint, config,
                    state_key="model_state_dict",
                )
            checkpoint_is_initialization = bool(
                checkpoint.get("initialization_only", False) or initialization_source
            )
        if gpu_master_enabled:
            if not checkpoint_is_initialization and checkpoint.get("optimizer_state_format") not in SUPPORTED_MASTER_SHARDED_FORMATS:
                raise RuntimeError("checkpoint 缺少兼容的 FP32 master 分片状态")
            checkpoint = {
                key: value for key, value in checkpoint.items()
                if key not in {"model_state_dict", "optimizer_state_dict", "ema_state_dict"}
            }
            gc.collect()

    cpu_initial_masters = None
    if gpu_master_enabled:
        if checkpoint_path is None or checkpoint_is_initialization:
            planner = ZeroRedundancyOptimizer(
                master_model.parameters(), optimizer_class=GPUMasterAdamW,
                process_group=dist.group.WORLD,
                parameters_as_bucket_view=False, overlap_with_ddp=False,
                lr=1e-4, weight_decay=2e-2, betas=(0.9, 0.95),
                foreach=False, fused=False,
            )
            owned = {id(p) for group in planner.optim.param_groups
                     for p in group["params"]}
            cpu_initial_masters = {
                name: p.detach().clone()
                for name, p in master_model.named_parameters()
                if id(p) in owned
            }
            del planner
        master_model = master_model.to(primary_device, dtype=torch.bfloat16)
        gc.collect()
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        for _ in range(initialization_group, initialization_group_count):
            dist.barrier()

    worker_models = []
    synchronizer = None
    if is_manual_parallel:
        worker_models = [
            GeometrizeGPT(
                vocab_size=config.MODEL_VOCAB_SIZE,
                d_model=config.MODEL_D_MODEL,
                n_layer=config.MODEL_N_LAYER,
                n_head=config.MODEL_N_HEAD,
                ffn_hidden=config.MODEL_FFN_HIDDEN,
                max_context_len=config.CONTEXT_BLOCKS,
                max_block_positions=config.MAX_BLOCK_POSITIONS,
                norm_eps=config.MODEL_NORM_EPS,
                rope_base=config.MODEL_ROPE_BASE,
            ).to(torch.device(f"cuda:{device_id}"))
            for device_id in device_ids[1:]
        ]
        synchronizer = ManualMultiGPUSynchronizer(
            master_model,
            worker_models,
            parallel_mode,
            config.PARALLEL_SYNC_BUFFER_CHUNK_MB,
        )
        synchronizer.synchronize_worker_parameters()
        # 初始化回传使用 non_blocking P2P copy；预热线程读取副本前必须完成交接。
        for device_id in device_ids:
            torch.cuda.synchronize(device_id)

    model = master_model
    if is_data_parallel:
        model = nn.DataParallel(
            master_model, device_ids=device_ids, output_device=device_ids[0]
        )

    # EMA is optional checkpoint-only state. When enabled, rank 0 alone owns the
    # extra model copy; when disabled, no EMA tensor is allocated on any GPU.
    ema = (
        ModelEMA(
            master_model,
            decay=config.EMA_DECAY,
            device=primary_device,
        )
        if is_main_process and config.ENABLE_EMA
        else None
    )
    optimizer_kwargs = {
        "lr": 1e-4,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.95),
        "foreach": False,
        "fused": bool(config.GPU_ADAMW_FUSED),
    }
    if is_ddp:
        optimizer = ZeroRedundancyOptimizer(
            master_model.parameters(),
            optimizer_class=(GPUMasterAdamW if gpu_master_enabled else torch.optim.AdamW),
            process_group=dist.group.WORLD,
            parameters_as_bucket_view=False,
            overlap_with_ddp=False,
            **optimizer_kwargs,
        )
    else:
        optimizer = torch.optim.AdamW(
            master_model.parameters(),
            **optimizer_kwargs,
        )
    if gpu_master_enabled and cpu_initial_masters is not None:
        names_by_id = {id(p): name for name, p in master_model.named_parameters()}
        local_parameters = [p for group in optimizer.optim.param_groups
                            for p in group["params"]]
        expected_names = {names_by_id[id(p)] for p in local_parameters}
        if expected_names != set(cpu_initial_masters):
            raise RuntimeError("CPU FP32 master 分片与最终 ZeRO-1 分区不一致")
        master_bytes = 0
        for parameter in local_parameters:
            source = cpu_initial_masters.pop(names_by_id[id(parameter)])
            if not torch.equal(source.to(primary_device, dtype=torch.bfloat16),
                               parameter.detach()):
                raise RuntimeError("GPU BF16 初始权重与 CPU FP32 master 不一致")
            master_bytes += source.numel() * source.element_size()
            optimizer.optim.prime_master(parameter, source, take_ownership=True)
        del cpu_initial_masters
        gc.collect()
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        print(
            f"[A800 GPU master] rank={distributed_rank} "
            f"local_fp32_master={master_bytes / 1024**3:.3f} GiB | "
            "BF16 GPU 工作权重已核对",
            flush=True,
        )
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(primary_device)
    autocast_enabled = primary_device.type == "cuda"
    dtype = torch.bfloat16 if autocast_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(
        primary_device.type, enabled=(autocast_enabled and dtype == torch.float16)
    )

    compile_models = [master_model, *worker_models]
    compile_level = config.TORCH_COMPILE_LEVEL
    configured_compile_threads = int(
        os.environ.get(
            "TORCHINDUCTOR_COMPILE_THREADS",
            config.TORCH_COMPILE_THREADS,
        )
    )
    compile_cache_dir = os.path.abspath(
        os.path.expanduser(
            os.environ.get(
                "TORCHINDUCTOR_CACHE_DIR",
                config.TORCH_COMPILE_CACHE_DIR,
            )
        )
    )
    if compile_level > 0:
        os.makedirs(compile_cache_dir, exist_ok=True)
    for compile_model in compile_models:
        if compile_level > 0 and config.FIXED_QUERY_GRID:
            # FlexAttention's create_block_mask() inspects a Python closure.
            # It must run eagerly before Dynamo traces the fullgraph forward.
            compile_device = next(compile_model.parameters()).device
            mask_key = (
                str(compile_device), config.TOTAL_CONTEXT_TOKENS, config.RAND_K
            )
            if mask_key not in compile_model._fused_fixed_grid_masks:
                compile_model._fused_fixed_grid_masks[mask_key] = (
                    build_fixed_grid_block_mask(
                        (config.TOTAL_CONTEXT_TOKENS - 1) * config.RAND_K,
                        config.TOTAL_CONTEXT_TOKENS,
                        config.RAND_K,
                        compile_device,
                    )
                )
        compile_model.configure_torch_compile(
            enabled=compile_level > 0,
            backend=config.TORCH_COMPILE_BACKEND,
            mode=config.TORCH_COMPILE_MODE,
            dynamic=config.TORCH_COMPILE_DYNAMIC,
            fullgraph=compile_level == 2,
            fixed_grid=config.FIXED_QUERY_GRID,
        )
    backend_warmed = False
    if is_ddp and compile_level > 0:
        if is_main_process:
            print(
                "🧵 [torch.compile/DDP] 首次编译按 rank 串行，"
                f"每 rank 最多 {configured_compile_threads} 个编译 worker；"
                "后续 rank 复用共享 Inductor 缓存。"
            )
        # 必须在 DDP 注册 autograd reducer hook 前预热。否则只让
        # 一个 rank 执行 backward 会在 AllReduce 中等待其他 rank。
        for compile_rank in range(distributed_world_size):
            if distributed_rank == compile_rank:
                print(
                    f"🧵 [torch.compile/DDP] rank {distributed_rank} "
                    "开始独占冷编译..."
                )
                warm_up_training_backend(
                    config=config,
                    model=master_model,
                    master_model=master_model,
                    worker_models=[],
                    primary_device=primary_device,
                    device_ids=[primary_device.index],
                    dtype=dtype,
                    criterion=criterion,
                    autocast_enabled=autocast_enabled,
                    is_manual_parallel=False,
                    is_data_parallel=False,
                )
            dist.barrier()
        backend_warmed = True
    if is_ddp:
        model = DistributedDataParallel(
            MultiLaneTrainingModule(master_model),
            device_ids=[primary_device.index],
            output_device=primary_device.index,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    print(
        "🧩 torch.compile: "
        + (
            f"level={compile_level} backend={config.TORCH_COMPILE_BACKEND} "
            f"mode={config.TORCH_COMPILE_MODE} dynamic={config.TORCH_COMPILE_DYNAMIC} "
            f"fullgraph={compile_level == 2} "
            f"compile_threads={configured_compile_threads} "
            f"persistent_cache={compile_cache_dir}"
            if compile_level > 0
            else "level=0 disabled (eager)"
        )
    )

    if checkpoint is not None:
        if ema is not None and not checkpoint_is_initialization:
            if "ema_state_dict" in checkpoint:
                load_checkpoint_state(
                    ema.ema, checkpoint, config,
                    state_key="ema_state_dict",
                )
            else:
                # ModelEMA was constructed after model_state_dict was restored,
                # so its initial value is already an exact copy of the resumed
                # training weights. No decay update is needed here.
                print(
                    "ℹ️ checkpoint 不含 EMA；从本次恢复后的模型权重开始构建 EMA。"
                )
        elif is_main_process and "ema_state_dict" in checkpoint and not checkpoint_is_initialization:
            print("ℹ️ 当前已关闭 EMA；忽略 checkpoint 中已有的 EMA 权重。")
        if checkpoint_kind == NEW_CHECKPOINT_KIND and not checkpoint_is_initialization:
            start_epoch = checkpoint["epoch"]
            resume_batch_idx = checkpoint.get("batch_idx", -1)
            global_step = checkpoint.get("global_step", 0)
            smoothed_loss = checkpoint.get("smoothed_loss", 10.0)
            smoothed_attribute_loss = checkpoint.get("attribute_loss")
            smoothed_termination_loss = checkpoint.get("termination_loss")
            resume_total_loss = checkpoint.get("epoch_total_loss", 0.0)
            old_batch_idx = resume_batch_idx
            resume_batch_idx, resume_total_loss = align_resume_batch(
                checkpoint, resume_batch_idx, resume_total_loss,
                distributed_world_size, config,
            )
            if old_batch_idx != resume_batch_idx:
                print(f"[Resume] micro-batch alignment {old_batch_idx} -> {resume_batch_idx}; global_step={global_step} unchanged", flush=True)
            if resume_batch_idx == -1:
                start_epoch += 1
                resume_total_loss = 0.0
            if gpu_master_enabled:
                load_local_gpu_master_shard(
                    checkpoint_dir, optimizer, master_model, primary_device
                )
                if "scaler_state_dict" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
            elif not RESET_OPTIMIZER_STATE and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                clear_zero_redundancy_transient_state(optimizer)
                if "scaler_state_dict" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print(
                f"--> 将从 Epoch {start_epoch}, Batch {resume_batch_idx + 1} 继续。"
                f"历史平滑 Loss: {smoothed_loss:.4f}"
            )
        elif checkpoint_is_initialization:
            print("--> 使用 v3 初始化权重，从 Epoch 1 / global_step 0 重新训练。")

    print_model_parameter_count(master_model)
    if checkpoint is not None:
        del checkpoint
        gc.collect()
        print("🧹 已释放 checkpoint 的 CPU 临时副本。")

    # 若当前 Epoch 尚无缓存，让多核 CSV 转换尽早与 GPU 后端预热重叠。
    if is_main_process:
        dataset.start_next_epoch_cache_prefetch(start_epoch)
    if not backend_warmed:
        warm_up_training_backend(
            config=config,
            model=model,
            master_model=master_model,
            worker_models=worker_models,
            primary_device=primary_device,
            device_ids=device_ids,
            dtype=dtype,
            criterion=criterion,
            autocast_enabled=autocast_enabled,
            is_manual_parallel=is_manual_parallel,
            is_data_parallel=is_data_parallel,
        )
    kfd_eviction_monitor = (
        KFDEvictionMonitor() if torch.version.hip is not None else None
    )
    print("\nStarting Training Loop...")
    first_optimizer_step_verified = False
    first_training_batch_verified = False
    device_count_for_accumulation = len(device_ids) if is_multi_gpu else 1
    global_loader_batch_size = (
        config.MICRO_BATCH_SIZE * device_count_for_accumulation
    )
    iterator_batch_size = (
        config.MICRO_BATCH_SIZE if is_ddp else global_loader_batch_size
    )
    manual_step_executor = None
    if is_manual_parallel and not scaler.is_enabled():
        manual_step_executor = ThreadPoolExecutor(
            max_workers=device_count_for_accumulation,
            thread_name_prefix="manual-multi-step",
        )
        print(
            f"⚙️ [Parallel] {device_count_for_accumulation} 卡微批前向/反向"
            "由持久线程并发提交；"
            f"梯度同步方式仍为 {parallel_mode}。"
        )
        if synchronizer is not None and synchronizer.p2p_buffer is not None:
            print(
                "🔁 [P2P] 每张卡独立累计完整窗口后才跨卡同步一次 | "
                f"reusable_gpu_buffer="
                f"{synchronizer.p2p_buffer.numel() * synchronizer.p2p_buffer.element_size() / 2**20:.0f} MiB"
            )
    elif is_manual_parallel:
        print(
            "⚠️ [Parallel] FP16 GradScaler 已启用，为保证缩放状态安全，"
            "多卡微批沿用主线程顺序提交。"
        )
    manual_models = [master_model, *worker_models]
    manual_devices = [
        torch.device(f"cuda:{device_id}") for device_id in device_ids
    ] if is_manual_parallel else []
    if config.DATALOADER_NUM_WORKERS != 0:
        print(
            "⚠️ [DataPipeline] 当前 mmap 管线不使用 DataLoader 子进程；"
            f"忽略 DATALOADER_NUM_WORKERS={config.DATALOADER_NUM_WORKERS}。"
        )

    guard_run_id = f"{os.getpid()}-{time.time_ns()}" if is_main_process else None
    guard_last_sequence = 0
    guard_last_checkpoint = None
    if is_main_process:
        guard_atomic_status({
            "run_id": guard_run_id, "phase": "running", "epoch": start_epoch,
            "batch_idx": resume_batch_idx, "global_step": global_step,
            "last_checkpoint": guard_last_checkpoint,
            "last_completed_sequence": guard_last_sequence,
            "updated_at": time.time(),
        })

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        if is_ddp:
            # Rank 0 alone may build a missing cache. Other ranks open it only
            # after the filesystem state is complete.
            if is_main_process:
                dataset.set_epoch(epoch)
            dist.barrier()
            if not is_main_process:
                dataset.set_epoch(epoch)
            dist.barrier()
        else:
            dataset.set_epoch(epoch)
        if is_main_process:
            cache_root = Path(config.DATA_CACHE_DIR).resolve()
            current_version = ((epoch - 1) % config.VERSION_CYCLE_EPOCHS) + 1
            keep_versions = {current_version, current_version % config.VERSION_CYCLE_EPOCHS + 1}
            for old_cache in cache_root.glob('v*'):
                match = re.fullmatch(r'v(\d+)', old_cache.name)
                if (match and int(match.group(1)) not in keep_versions
                        and old_cache.is_dir() and not old_cache.is_symlink()
                        and old_cache.resolve().parent == cache_root):
                    shutil.rmtree(old_cache)
                    print(f'[CacheWindow] released closed old version {old_cache.name}', flush=True)
        if is_ddp: dist.barrier()
        drop_last = is_multi_gpu
        if drop_last:
            batches_per_epoch = len(dataset) // global_loader_batch_size
        else:
            batches_per_epoch = (
                len(dataset) + global_loader_batch_size - 1
            ) // global_loader_batch_size

        epoch_start_batch = (
            resume_batch_idx + 1
            if epoch == start_epoch and resume_batch_idx >= 0
            else 0
        )
        if epoch_start_batch > batches_per_epoch:
            raise RuntimeError(
                f"checkpoint 要求从 Batch {epoch_start_batch} 恢复，"
                f"但当前 Epoch 只有 {batches_per_epoch} 个 Batch。"
            )

        start_sample_index = min(
            epoch_start_batch * global_loader_batch_size, len(dataset)
        )
        if epoch_start_batch > 0:
            print(
                f"⏩ [DataPipeline] 直接定位到 Batch {epoch_start_batch}/"
                f"{batches_per_epoch}（样本偏移 {start_sample_index}），"
                f"跳过构造前 {epoch_start_batch} 个已训练 Batch。"
            )

        remaining_sample_count = len(dataset) - start_sample_index
        if drop_last:
            remaining_batches = remaining_sample_count // global_loader_batch_size
        else:
            remaining_batches = (
                remaining_sample_count + global_loader_batch_size - 1
            ) // global_loader_batch_size
        expected_remaining_batches = batches_per_epoch - epoch_start_batch
        if remaining_batches != expected_remaining_batches:
            raise RuntimeError(
                "恢复后的批次数计算错误："
                f"实际 {remaining_batches}，期望 {expected_remaining_batches}。"
            )
        print(
            f"🚚 [DataPipeline] 已创建 | remaining_batches={remaining_batches} | "
            f"total_batches={batches_per_epoch} | "
            f"global_batch={global_loader_batch_size} | "
            f"rank_batch={iterator_batch_size} | mmap_on_demand=True | "
            f"reusable_pinned_buffers=2 | async_ping_pong=True | "
            f"background_next_epoch_cache=True"
        )

        model.train()
        for worker_model in worker_models:
            worker_model.train()
        total_loss = resume_total_loss if epoch == start_epoch else 0.0
        total_attribute_loss = 0.0
        total_termination_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for worker_model in worker_models:
            worker_model.zero_grad(set_to_none=True)
        nonfinite_counter = NonFiniteEventCounter(
            max_consecutive=config.NONFINITE_MAX_CONSECUTIVE,
            max_per_epoch=config.NONFINITE_MAX_PER_EPOCH,
        )
        accum_counter = 0
        local_accum_steps = 1
        effective_global_accum = 1
        accumulation_window_start_batch = epoch_start_batch
        last_log_time = time.perf_counter()
        last_log_batch_idx = epoch_start_batch - 1
        window_data_wait_seconds = 0.0
        window_async_fill_seconds = 0.0
        window_update_seconds = 0.0
        window_update_count = 0
        next_epoch_cache_prefetch_started = False
        resume_for_schedule = (
            resume_batch_idx if epoch == start_epoch else -1
        )
        next_save_batch = next_intermediate_save_batch(
            resume_for_schedule, config.SAVE_INTERVAL_BATCHES
        )

        def guard_control_step(safe_batch_idx):
            """All ranks poll together after optimizer.step; never save partial grads."""
            nonlocal guard_last_sequence, guard_last_checkpoint
            paused = False
            pause_reason = None
            vram = guard_external_vram_snapshot(primary_device)
            local_pressure = bool(
                vram is not None and
                vram["outside_torch_estimate_bytes"] >= 5 * 1024 ** 3
            )
            pressure = torch.tensor(
                int(local_pressure), dtype=torch.uint8, device=primary_device
            )
            if is_ddp:
                dist.all_reduce(pressure, op=dist.ReduceOp.MAX)
            auto_pause = bool(pressure.item())
            if local_pressure:
                print(f"[GuardControl] 外部显存占用触发安全暂停: rank={distributed_rank} {vram}", flush=True)
            while True:
                action, sequence = (
                    guard_read_request(guard_run_id, guard_last_sequence)
                    if is_main_process else (0, 0)
                )
                token = torch.tensor(
                    [action, sequence], dtype=torch.int64, device=primary_device
                )
                if is_ddp:
                    dist.broadcast(token, src=0)
                action, sequence = (int(value) for value in token.tolist())
                if auto_pause and not paused:
                    action = 2
                    pause_reason = "outside_torch_vram_threshold"
                    auto_pause = False
                if action in (1, 2):
                    checkpoint_path = training_checkpoint_path(
                        config, epoch, safe_batch_idx
                    )
                    # A save request during a pause refers to the same model
                    # state, so there is no reason to overwrite its checkpoint.
                    if not (paused and checkpoint_path == guard_last_checkpoint):
                        save_training_checkpoint(
                            checkpoint_path, epoch, safe_batch_idx, global_step,
                            smoothed_loss, total_loss, model, ema, optimizer,
                            scaler, parallel_mode, config, is_main_process,
                            attribute_loss=smoothed_attribute_loss,
                            termination_loss=smoothed_termination_loss,
                        )
                        guard_last_checkpoint = checkpoint_path
                        if is_main_process:
                            print(f"[GuardControl] 安全检查点已保存: {checkpoint_path}", flush=True)
                            if not config.SHARDED_GPU_MASTER_CHECKPOINTS:
                                prune_intermediate_checkpoints(config.CKPT_DIR, keep=2)
                    if sequence:
                        guard_last_sequence = sequence
                    if action == 2:
                        paused = True
                        if pause_reason is None:
                            pause_reason = "manual_request"
                elif action == 3 and paused:
                    guard_last_sequence = sequence
                    paused = False
                    pause_reason = None
                if is_main_process:
                    guard_atomic_status({
                        "run_id": guard_run_id,
                        "phase": "paused" if paused else "running",
                        "pause_reason": pause_reason,
                        "epoch": epoch, "batch_idx": safe_batch_idx,
                        "global_step": global_step,
                        "last_checkpoint": guard_last_checkpoint,
                        "last_completed_sequence": guard_last_sequence,
                        "updated_at": time.time(),
                    })
                if not paused:
                    return
                time.sleep(1)

        first_batch_wait_started = time.perf_counter()
        first_batch_received = False
        batch_iterator = MMapBatchIterator(
            dataset,
            start_sample_index + (
                distributed_rank * config.MICRO_BATCH_SIZE if is_ddp else 0
            ),
            iterator_batch_size,
            drop_last,
            pin_memory=primary_device.type == "cuda",
            sample_stride=global_loader_batch_size,
            max_batches=remaining_batches,
        )
        for batch_idx, (input_ids, text_ids, text_mask) in enumerate(
            batch_iterator, start=epoch_start_batch
        ):
            window_data_wait_seconds += batch_iterator.last_wait_seconds
            window_async_fill_seconds += batch_iterator.last_fill_seconds
            if not first_batch_received:
                print(
                    f"✅ [DataPipeline] 首批数据已就绪 | shape={tuple(input_ids.shape)} | "
                    f"等待 {time.perf_counter() - first_batch_wait_started:.3f}s"
                )
                first_batch_received = True
            first_training_batch_started_at = (
                time.perf_counter()
                if not first_training_batch_verified
                else None
            )
            if input_ids.size(1) != config.CONTEXT_TOKENS:
                raise RuntimeError(
                    f"训练序列长度应为 {config.CONTEXT_TOKENS}，"
                    f"实际收到 {input_ids.size(1)}"
                )
            if is_multi_gpu and input_ids.size(0) != iterator_batch_size:
                raise RuntimeError(
                    "多卡训练批次未被均分；请检查数据管线的 drop_last 设置。"
                )

            if accum_counter == 0:
                accumulation_window_start_batch = batch_idx
                local_accum_steps = min(
                    config.GRADIENT_ACCUMULATION_STEPS,
                    batches_per_epoch - batch_idx,
                )
                effective_global_accum = get_effective_accumulation(
                    local_accum_steps,
                    device_count_for_accumulation,
                )

            current_base_lr = get_current_lr(
                epoch,
                batch_idx,
                batches_per_epoch,
                global_step,
                LR_SCHEDULE_BY_EPOCH,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_base_lr

            backward_already_run = False
            # Loop iterations share one Python scope. Drop references from the
            # preceding iteration before allocating a new forward graph.
            loss = None
            microbatch_loss = None
            scaled_microbatch_loss = None
            batch_input = None
            device_input = None
            reduced_loss = None
            mean_attribute_loss = None
            mean_termination_loss = None
            gradients = None
            microbatch_losses = []
            scaled_microbatch_losses = []
            scaled_loss = None
            if manual_step_executor is not None:
                microbatch_inputs = input_ids.split(
                    config.MICRO_BATCH_SIZE, dim=0
                )
                futures = [
                    manual_step_executor.submit(
                        run_manual_gpu_microbatch,
                        target_model,
                        host_input,
                        device,
                        config,
                        criterion,
                        dtype,
                        autocast_enabled,
                        local_accum_steps,
                        derive_rand_k_sampling_seed(
                            config.RAND_K_SEED, epoch, batch_idx, device_rank
                        ),
                    )
                    for device_rank, (target_model, host_input, device) in enumerate(
                        zip(manual_models, microbatch_inputs, manual_devices)
                    )
                ]
                microbatch_results = [
                    future.result() for future in futures
                ]
                microbatch_losses = [result[0] for result in microbatch_results]
                mean_loss = (
                    sum(microbatch_losses) / len(microbatch_losses)
                )
                mean_attribute_loss = sum(
                    result[1] for result in microbatch_results
                ) / len(microbatch_results)
                mean_termination_loss = sum(
                    result[2] for result in microbatch_results
                ) / len(microbatch_results)
                backward_already_run = True
            else:
                if is_ddp:
                    # Equivalent to DDP.no_sync(): accumulated micro-batches do
                    # not launch AllReduce until the last local batch in window.
                    model.require_backward_grad_sync = (
                        accum_counter + 1 >= local_accum_steps
                    )
                with torch.amp.autocast(
                    device_type=primary_device.type,
                    dtype=dtype,
                    enabled=autocast_enabled,
                ):
                    if is_manual_parallel:
                        microbatch_inputs = input_ids.split(
                            config.MICRO_BATCH_SIZE, dim=0
                        )
                        microbatch_losses = []
                        scaled_microbatch_losses = []
                        for device_rank, (target_model, host_input, device) in enumerate(
                            zip(manual_models, microbatch_inputs, manual_devices)
                        ):
                            device_input = host_input.to(
                                device, non_blocking=True
                            )
                            # criterion 无参数和缓冲，可安全复用于所有 GPU。
                            (
                                microbatch_loss,
                                microbatch_attribute_loss,
                                microbatch_termination_loss,
                            ) = compute_loss(
                                target_model,
                                device_input,
                                config,
                                criterion,
                                sampling_seed=derive_rand_k_sampling_seed(
                                    config.RAND_K_SEED,
                                    epoch,
                                    batch_idx,
                                    device_rank,
                                ),
                                return_loss_components=True,
                            )
                            microbatch_losses.append(microbatch_loss)
                            mean_attribute_loss = (
                                (0.0 if mean_attribute_loss is None else mean_attribute_loss)
                                + microbatch_attribute_loss.item()
                            )
                            mean_termination_loss = (
                                (0.0 if mean_termination_loss is None else mean_termination_loss)
                                + microbatch_termination_loss.item()
                            )
                            scaled_microbatch_losses.append(
                                microbatch_loss / local_accum_steps
                            )
                        mean_loss = sum(
                            value.detach().item()
                            for value in microbatch_losses
                        ) / len(microbatch_losses)
                        mean_attribute_loss /= len(microbatch_losses)
                        mean_termination_loss /= len(microbatch_losses)
                    else:
                        batch_input = input_ids.to(
                            primary_device, non_blocking=True
                        )
                        batch_text_ids = text_ids.to(primary_device, non_blocking=True)
                        batch_text_mask = text_mask.to(primary_device, non_blocking=True)
                        try:
                            loss, attribute_loss, termination_loss = compute_loss(
                                model,
                                batch_input,
                                config,
                                criterion,
                                sampling_seed=derive_rand_k_sampling_seed(
                                    config.RAND_K_SEED, epoch, batch_idx, 0
                                ),
                                return_loss_components=True,
                                text_ids=batch_text_ids,
                                text_mask=batch_text_mask,
                            )
                        except torch.OutOfMemoryError as error:
                            guard_oom_stop(distributed_rank, epoch, batch_idx,
                                           primary_device, error)
                            raise
                        if nonfinite_injection == (
                            epoch, batch_idx, distributed_rank
                        ):
                            print(
                                "🧪 [NumericalGuard] 手动注入 NaN loss | "
                                f"rank={distributed_rank} epoch={epoch} "
                                f"batch={batch_idx}",
                                flush=True,
                            )
                            loss = loss * torch.full_like(loss, float("nan"))
                        scaled_loss = loss / local_accum_steps
                        mean_loss = loss.detach().item()
                        mean_attribute_loss = attribute_loss.item()
                        mean_termination_loss = termination_loss.item()

            local_mean_loss = mean_loss
            if is_ddp:
                # This scalar collective already existed for logging. Perform it
                # before backward so one bad rank can make every rank discard the
                # same accumulation window without entering mismatched DDP hooks.
                reduced_loss = torch.tensor(
                    [mean_loss, mean_attribute_loss, mean_termination_loss],
                    dtype=torch.float32,
                    device=primary_device,
                )
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss = reduced_loss / distributed_world_size
                mean_loss, mean_attribute_loss, mean_termination_loss = (
                    value.item() for value in reduced_loss
                )

            loss_nonfinite = (
                config.ENABLE_NUMERICAL_GUARD
                and not math.isfinite(mean_loss)
            )
            if loss_nonfinite:
                memory_before_cleanup = cuda_memory_snapshot(primary_device)
                if not math.isfinite(local_mean_loss):
                    print(
                        "🚨 [NumericalGuard] 本 rank 检测到非有限 loss | "
                        f"rank={distributed_rank} | epoch={epoch} | "
                        f"batch={batch_idx} | local_loss={local_mean_loss}",
                        flush=True,
                    )
                should_abort = nonfinite_counter.record_failure()
                completed_discard_backward = complete_discarded_ddp_backward(
                    scaled_loss=scaled_loss,
                    scaler=scaler,
                    is_ddp=is_ddp,
                    backward_already_run=backward_already_run,
                )
                optimizer.zero_grad(set_to_none=True)
                for worker_model in worker_models:
                    worker_model.zero_grad(set_to_none=True)
                accum_counter = 0

                # zero_grad only disconnects Parameter.grad. Explicitly release
                # every local that can still own the skipped forward graph.
                loss = None
                scaled_loss = None
                batch_input = None
                microbatch_loss = None
                scaled_microbatch_loss = None
                device_input = None
                reduced_loss = None
                microbatch_losses.clear()
                scaled_microbatch_losses.clear()
                gc.collect()
                cleanup_devices = (
                    [primary_device]
                    if is_ddp or not is_multi_gpu
                    else [torch.device(f"cuda:{device_id}") for device_id in device_ids]
                )
                release_cuda_caches(cleanup_devices)
                memory_after_cleanup = cuda_memory_snapshot(primary_device)
                if primary_device.type == "cuda":
                    print(
                        "🧹 [NumericalGuard] 异常窗口活跃显存已释放（缓存保留） | "
                        f"rank={distributed_rank} | "
                        f"ddp_backward_completed={completed_discard_backward} | "
                        f"before={memory_before_cleanup} | "
                        f"after={memory_after_cleanup}",
                        flush=True,
                    )
                if is_main_process:
                    print(
                        "⚠️ [NumericalGuard] 已跳过含非有限 loss 的完整梯度累计窗口 | "
                        f"Epoch={epoch} | Batch={batch_idx} | "
                        f"Window={accumulation_window_start_batch}:{batch_idx} | "
                        f"LR={current_base_lr:.3e} | "
                        f"连续={nonfinite_counter.consecutive}/"
                        f"{config.NONFINITE_MAX_CONSECUTIVE} | "
                        f"本轮累计={nonfinite_counter.epoch_total}/"
                        f"{config.NONFINITE_MAX_PER_EPOCH}",
                        flush=True,
                    )
                if should_abort:
                    raise FloatingPointError(
                        "非有限 loss 超过容忍阈值；所有 rank 同步停止。"
                    )
                continue

            if not backward_already_run:
                try:
                    if is_manual_parallel:
                        for scaled_microbatch_loss in scaled_microbatch_losses:
                            scaler.scale(scaled_microbatch_loss).backward()
                    else:
                        scaler.scale(scaled_loss).backward()
                except torch.OutOfMemoryError as error:
                    guard_oom_stop(distributed_rank, epoch, batch_idx,
                                   primary_device, error)
                    raise

            if not first_training_batch_verified:
                if primary_device.type == "cuda":
                    verification_device_ids = (
                        [primary_device]
                        if is_ddp
                        else (device_ids if is_multi_gpu else [primary_device])
                    )
                    for verification_device_id in verification_device_ids:
                        torch.cuda.synchronize(verification_device_id)
                print(
                    "✅ 首个实际训练 Batch 已完成前向与反向 | "
                    f"Batch={batch_idx}/{batches_per_epoch} | "
                    f"Accum(Local)={1}/{local_accum_steps} | "
                    f"耗时={time.perf_counter() - first_training_batch_started_at:.3f}s"
                )
                first_training_batch_verified = True

            smoothed_loss = (
                mean_loss
                if smoothed_loss is None
                else 0.99 * smoothed_loss + 0.01 * mean_loss
            )
            smoothed_attribute_loss = (
                mean_attribute_loss
                if smoothed_attribute_loss is None
                else 0.99 * smoothed_attribute_loss + 0.01 * mean_attribute_loss
            )
            smoothed_termination_loss = (
                mean_termination_loss
                if smoothed_termination_loss is None
                else 0.99 * smoothed_termination_loss + 0.01 * mean_termination_loss
            )
            total_loss += mean_loss
            total_attribute_loss += mean_attribute_loss
            total_termination_loss += mean_termination_loss
            accum_counter += 1
            will_step = accum_counter >= local_accum_steps

            if will_step:
                update_started_at = time.perf_counter()
                if synchronizer is not None:
                    synchronizer.average_worker_gradients_into_master()
                scaler.unscale_(optimizer)
                if config.ENABLE_NUMERICAL_GUARD:
                    gradients = [
                        parameter.grad
                        for parameter in master_model.parameters()
                        if parameter.grad is not None
                    ]
                    grad_norm = collection_total_norm(
                        gradients, primary_device
                    )
                    local_grad_nonfinite = not bool(torch.isfinite(grad_norm))
                    grad_nonfinite = synchronize_nonfinite_flag(
                        local_grad_nonfinite, primary_device, is_ddp
                    )
                    if grad_nonfinite:
                        memory_before_cleanup = cuda_memory_snapshot(primary_device)
                        bad_gradients = nonfinite_parameter_names(
                            master_model, gradients=True
                        )
                        if local_grad_nonfinite:
                            print(
                                "🚨 [NumericalGuard] 本 rank 检测到非有限梯度 | "
                                f"rank={distributed_rank} | epoch={epoch} | "
                                f"batch={batch_idx} | grad_norm={grad_norm.item()} | "
                                f"bad_gradients={bad_gradients}",
                                flush=True,
                            )
                        should_abort = nonfinite_counter.record_failure()
                        # FP16 GradScaler needs this update after unscale_ to
                        # reduce its scale. BF16 uses a disabled scaler.
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        for worker_model in worker_models:
                            worker_model.zero_grad(set_to_none=True)
                        accum_counter = 0
                        gradients.clear()
                        gradients = None
                        loss = None
                        scaled_loss = None
                        batch_input = None
                        microbatch_loss = None
                        scaled_microbatch_loss = None
                        device_input = None
                        microbatch_losses.clear()
                        scaled_microbatch_losses.clear()
                        gc.collect()
                        cleanup_devices = (
                            [primary_device]
                            if is_ddp or not is_multi_gpu
                            else [
                                torch.device(f"cuda:{device_id}")
                                for device_id in device_ids
                            ]
                        )
                        release_cuda_caches(cleanup_devices)
                        memory_after_cleanup = cuda_memory_snapshot(primary_device)
                        window_update_seconds += (
                            time.perf_counter() - update_started_at
                        )
                        if primary_device.type == "cuda":
                            print(
                                "🧹 [NumericalGuard] 非有限梯度活跃显存已释放（缓存保留） | "
                                f"rank={distributed_rank} | "
                                f"before={memory_before_cleanup} | "
                                f"after={memory_after_cleanup}",
                                flush=True,
                            )
                        if is_main_process:
                            print(
                                "⚠️ [NumericalGuard] 已跳过非有限梯度累计窗口，"
                                "AdamW 未执行 | "
                                f"Epoch={epoch} | Batch={batch_idx} | "
                                f"Window={accumulation_window_start_batch}:{batch_idx} | "
                                f"LR={current_base_lr:.3e} | "
                                f"连续={nonfinite_counter.consecutive}/"
                                f"{config.NONFINITE_MAX_CONSECUTIVE} | "
                                f"本轮累计={nonfinite_counter.epoch_total}/"
                                f"{config.NONFINITE_MAX_PER_EPOCH}",
                                flush=True,
                            )
                        if should_abort:
                            raise FloatingPointError(
                                "非有限梯度超过容忍阈值；所有 rank 同步停止。"
                            )
                        continue
                    torch.nn.utils.clip_grads_with_norm_(
                        master_model.parameters(),
                        max_norm=0.5,
                        total_norm=grad_norm,
                        foreach=None,
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        master_model.parameters(), max_norm=0.5
                    )
                scaler.step(optimizer)
                scaler.update()
                if config.ENABLE_NUMERICAL_GUARD:
                    parameter_norm, optimizer_norm = local_training_state_norms(
                        master_model, optimizer, primary_device
                    )
                    local_state_nonfinite = not bool(
                        torch.isfinite(parameter_norm)
                        & torch.isfinite(optimizer_norm)
                    )
                    state_nonfinite = synchronize_nonfinite_flag(
                        local_state_nonfinite, primary_device, is_ddp
                    )
                    if state_nonfinite:
                        bad_parameters = nonfinite_parameter_names(master_model)
                        bad_optimizer = nonfinite_optimizer_state_names(optimizer)
                        raise FloatingPointError(
                            "optimizer.step() 产生了非有限状态，无法安全回滚；"
                            "拒绝继续训练或保存 checkpoint | "
                            f"rank={distributed_rank} | epoch={epoch} | "
                            f"batch={batch_idx} | parameter_norm={parameter_norm.item()} | "
                            f"optimizer_norm={optimizer_norm.item()} | "
                            f"bad_parameters={bad_parameters} | "
                            f"bad_optimizer={bad_optimizer}"
                        )
                    nonfinite_counter.record_successful_update()
                if ema is not None:
                    ema.update(master_model)
                if synchronizer is not None:
                    synchronizer.synchronize_worker_parameters()
                if primary_device.type == "cuda":
                    # 多线程模式下，副卡参数由主线程回传、下一批由固定
                    # worker 线程消费；必须在所有设备都完成后再跨线程交接。
                    update_device_ids = (
                        [primary_device]
                        if is_ddp
                        else (device_ids if is_multi_gpu else [primary_device])
                    )
                    for update_device_id in update_device_ids:
                        torch.cuda.synchronize(update_device_id)
                optimizer.zero_grad(set_to_none=True)
                for worker_model in worker_models:
                    worker_model.zero_grad(set_to_none=True)
                accum_counter = 0
                global_step += 1
                window_update_seconds += (
                    time.perf_counter() - update_started_at
                )
                window_update_count += 1

                if not first_optimizer_step_verified:
                    print(
                        "✅ 首个梯度累计窗口已完成 optimizer.step 验证 | "
                        f"每卡分别累计 {local_accum_steps} 个小批量；"
                        "micro-batch 间不传递梯度/参数，窗口末同步梯度"
                        "并广播 ZeRO-1 更新参数；"
                        f"RoPE 无可训练位置表；EMA={'开启' if ema is not None else '关闭'} | "
                        f"模式={parallel_mode} | 设备="
                        f"{device_ids if is_multi_gpu else [primary_device]}"
                    )
                    first_optimizer_step_verified = True

                if batch_idx + 1 >= next_save_batch:
                    ckpt_path = training_checkpoint_path(config, epoch, batch_idx)
                    save_training_checkpoint(
                        ckpt_path,
                        epoch,
                        batch_idx,
                        global_step,
                        smoothed_loss,
                        total_loss,
                        model,
                        ema,
                        optimizer,
                        scaler,
                        parallel_mode,
                        config,
                        is_main_process,
                        attribute_loss=smoothed_attribute_loss,
                        termination_loss=smoothed_termination_loss,
                    )
                    guard_last_checkpoint = ckpt_path
                    if is_main_process:
                        print(f"--> 💾 [安全检查点] 进度存档已建立: {ckpt_path}\n")
                        if not config.SHARDED_GPU_MASTER_CHECKPOINTS:
                            prune_intermediate_checkpoints(config.CKPT_DIR, keep=2)
                    next_save_batch += config.SAVE_INTERVAL_BATCHES

                guard_control_step(batch_idx)

            if batch_idx % 500 == 0 and is_main_process:
                if primary_device.type == "cuda":
                    timing_device_ids = (
                        [primary_device]
                        if is_ddp
                        else (device_ids if is_multi_gpu else [primary_device])
                    )
                    for timing_device_id in timing_device_ids:
                        torch.cuda.synchronize(timing_device_id)
                elapsed_seconds = time.perf_counter() - last_log_time
                window_batches = batch_idx - last_log_batch_idx
                train_compute_seconds = max(
                    0.0,
                    elapsed_seconds
                    - window_data_wait_seconds
                    - window_update_seconds,
                )
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Batch {batch_idx}/{batches_per_epoch} | "
                    f"Loss(total/attr/term): {smoothed_loss:.4f}/"
                    f"{smoothed_attribute_loss:.4f}/"
                    f"{smoothed_termination_loss:.4f} | "
                    f"Accum(Global/Local): {effective_global_accum}/{local_accum_steps} | "
                    f"Eq_BS: {effective_global_accum * config.MICRO_BATCH_SIZE} | "
                    f"LR_GPT: {current_base_lr:.2e} | "
                    f"Window: {window_batches} batches / {elapsed_seconds:.2f}s "
                    f"({elapsed_seconds / max(window_batches, 1):.3f}s per batch)"
                )
                if nonfinite_counter.epoch_total:
                    print(
                        "  ⚠️ NumericalGuard: "
                        f"epoch_skips={nonfinite_counter.epoch_total} | "
                        f"consecutive={nonfinite_counter.consecutive}"
                    )
                print(
                    f"  ↳ Pipeline: data_wait={window_data_wait_seconds:.3f}s | "
                    f"async_mmap_fill={window_async_fill_seconds:.3f}s(overlapped) | "
                    f"train_compute={train_compute_seconds:.3f}s | "
                    f"optimizer_sync={window_update_seconds:.3f}s/"
                    f"{window_update_count} updates "
                    f"({window_update_seconds / max(window_update_count, 1):.3f}s/update) | "
                    "（墙钟分解，无逐 Batch GPU event 扰动）"
                )
                if kfd_eviction_monitor is not None:
                    eviction_deltas = (
                        kfd_eviction_monitor.consume_deltas()
                    )
                    if eviction_deltas:
                        formatted_deltas = ", ".join(
                            f"KFD{node_id}=+{delta_ms / 1000:.3f}s"
                            for node_id, delta_ms in sorted(
                                eviction_deltas.items()
                            )
                        )
                        print(
                            "  🚨 [KFD Queue Eviction] "
                            f"{formatted_deltas} | 这段时间 GPU 队列未执行；"
                            "不是模型计算或数据加载耗时。"
                        )
                if not next_epoch_cache_prefetch_started:
                    next_epoch_cache_prefetch_started = (
                        dataset.start_next_epoch_cache_prefetch(epoch + 1)
                    )
                last_log_time = time.perf_counter()
                last_log_batch_idx = batch_idx
                window_data_wait_seconds = 0.0
                window_async_fill_seconds = 0.0
                window_update_seconds = 0.0
                window_update_count = 0
        batch_iterator.close()

        if is_main_process and not next_epoch_cache_prefetch_started:
            dataset.start_next_epoch_cache_prefetch(epoch + 1)
        avg_loss = total_loss / max(batches_per_epoch, 1)
        avg_attribute_loss = total_attribute_loss / max(batches_per_epoch, 1)
        avg_termination_loss = total_termination_loss / max(batches_per_epoch, 1)
        if is_main_process:
            print(
                f"==== Epoch {epoch} Completed | Avg Loss(total/attr/term): "
                f"{avg_loss:.4f}/{avg_attribute_loss:.4f}/"
                f"{avg_termination_loss:.4f} ====\n"
            )
        ckpt_path = training_checkpoint_path(config, epoch)
        save_training_checkpoint(
            ckpt_path,
            epoch,
            -1,
            global_step,
            smoothed_loss,
            total_loss,
            model,
            ema,
            optimizer,
            scaler,
            parallel_mode,
            config,
            is_main_process,
            attribute_loss=smoothed_attribute_loss,
            termination_loss=smoothed_termination_loss,
        )
        if is_main_process:
            print(f"--> Saved checkpoint: {ckpt_path}\n")
            if not config.SHARDED_GPU_MASTER_CHECKPOINTS:
                prune_intermediate_checkpoints(config.CKPT_DIR, keep=2)
        guard_last_checkpoint = ckpt_path
        resume_batch_idx = -1

    if is_main_process:
        guard_atomic_status({
            "run_id": guard_run_id, "phase": "complete", "epoch": last_epoch,
            "batch_idx": -1, "global_step": global_step,
            "last_checkpoint": guard_last_checkpoint,
            "last_completed_sequence": guard_last_sequence,
            "updated_at": time.time(),
        })
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    launch_config = TrainConfig()
    if launch_config.PARALLEL_MODE == "ddp" and "LOCAL_RANK" not in os.environ:
        process_count = len(launch_config.PARALLEL_DEVICE_IDS)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={process_count}",
            os.path.abspath(__file__),
        ]
        raise SystemExit(subprocess.call(command))
    run_training()
