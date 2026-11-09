# config.py
import os
from pathlib import Path

from token_layout import (
    DATA_TOKEN_LAYOUT_VERSION,
    TOKEN_LAYOUT,
    TOKEN_LAYOUT_VERSION,
)
from block_state import (
    ATTRIBUTES_PER_PRIMITIVE,
    PRIMITIVES_PER_BLOCK,
    VALUES_PER_BLOCK,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("A800_DATA_ROOT", str(PROJECT_DIR / "data")))
SUPPORT_ROOT = Path(os.environ.get("A800_SUPPORT_ROOT", str(PROJECT_DIR / "assets")))
RUNTIME_ROOT = Path(os.environ.get("A800_RUNTIME_ROOT", str(PROJECT_DIR / "outputs")))


def resolve_data_dir():
    """解析训练数据目录，支持环境变量、项目相对路径和旧 Windows 路径。"""
    env_data_dir = os.environ.get("ANIME_PAINTER_DATA_DIR")
    if env_data_dir:
        return os.path.abspath(os.path.expanduser(env_data_dir))

    return str(DATA_ROOT / "sequences")


def resolve_data_cache_dir():
    """缓存放在项目目录内，也可通过环境变量迁移到其他高速磁盘。"""
    env_cache_dir = os.environ.get("ANIME_PAINTER_DATA_CACHE_DIR")
    if env_cache_dir:
        return os.path.abspath(os.path.expanduser(env_cache_dir))
    return str(RUNTIME_ROOT / "geometry_cache")


class TrainConfig:
    # ==========================================
    # 📁 路径配置
    # ==========================================
    # 可用 ANIME_PAINTER_DATA_DIR 显式覆盖；Linux 默认使用项目旁的数据目录。
    # DATA_DIR = resolve_data_dir()
    DATA_DIR = resolve_data_dir()
    DATA_CACHE_DIR = resolve_data_cache_dir()
    CAPTION_TOKENIZER_PATH = str(PROJECT_DIR / "assets" / "caption_spm_1500.model")
    CAPTION_SELECTION_PATH = str(DATA_ROOT / "selected_images.tsv")
    CAPTION_PROMPTS_DIR = str(DATA_ROOT / "prompts")
    CAPTION_CACHE_DIR = str(RUNTIME_ROOT / "caption_cache")
    CAPTION_VARIANTS = 20
    CAPTION_EXPECTED_IMAGES = None  # Any positive explicit selection size.
    TEXT_VOCAB_SIZE = 1500
    TEXT_MAX_TOKENS = 224
    TOTAL_CONTEXT_TOKENS = 1024  # 文字 + 图元 BOS + 可见图元 + 右侧 padding
    FIXED_QUERY_GRID = True  # 固定 1024 历史位和 1023 组查询；文字/输入/PAD 不计 loss
    TEXT_DROPOUT_PROBABILITY = 0.30
    TEXT_SAMPLING_SEED = 20260929
    # Public CLI chooses explicit scratch, model-only initialization, or complete resume.
    CKPT_DIR = os.environ.get("A800_CHECKPOINT_ROOT", str(RUNTIME_ROOT / "checkpoints"))
    RESUME_CKPT_DIR = os.environ.get("A800_RESUME_ROOT", str(RUNTIME_ROOT / "checkpoints"))
    RANDOM_INITIALIZATION_IF_NO_CHECKPOINT = False
    GPU_FP32_MASTER_WEIGHTS = False
    SHARDED_GPU_MASTER_CHECKPOINTS = False
    GPU_ADAMW_FUSED = True
    IMAGE_INIT_CHECKPOINT_PATH = str(
        Path(__file__).resolve().parent
        / "checkpoints_512_interrupt_896ctx_24l_rope_rms_swiglu"
        / "geometrize_gpt_pretrain_epoch_42.pt"
    )
    INITIAL_CHECKPOINT_PATH = str(
        Path(__file__).resolve().parent
        / "checkpoints_dev2_marker_end_448ctx_24l_rope_rms_swiglu"
        / "geometrize_gpt_pretrain_epoch_157.pt"
    )
    MIGRATED_INITIAL_CHECKPOINT_PATH = str(
        Path(CKPT_DIR) / "geometrize_gpt_pretrain_init_from_epoch_157.pt"
    )
    CANVAS_SIZE = 512
    MAX_SHAPE_SIZE = 512
    TOKEN_LAYOUT_VERSION = TOKEN_LAYOUT_VERSION
    DATA_TOKEN_LAYOUT_VERSION = DATA_TOKEN_LAYOUT_VERSION
    
    # ==========================================
    # 📐 自回归上下文与位置嵌入容量
    # ==========================================
    # 每个图元固定由 13 个离散 token 表示（9 个原属性 + 4 个贝塞尔边控制点）。
    # 仅保留独立 BOS；背景及全部真实图元都必须由模型预测。
    PREFIX_STEPS = 1
    PREDICTION_STEPS = 896
    CONTEXT_STEPS = 897
    TOKENS_PER_STEP = ATTRIBUTES_PER_PRIMITIVE

    BLOCK_PRIMITIVES = PRIMITIVES_PER_BLOCK
    ATTRIBUTES_PER_PRIMITIVE = ATTRIBUTES_PER_PRIMITIVE
    BLOCK_VALUE_COUNT = VALUES_PER_BLOCK
    CONTEXT_BLOCKS = CONTEXT_STEPS // BLOCK_PRIMITIVES
    FIRST_QUERY_BLOCK = PREFIX_STEPS // BLOCK_PRIMITIVES
    BRANCH_QUERY_BLOCKS = CONTEXT_BLOCKS - FIRST_QUERY_BLOCK
    # 仅供诊断估算容量；生产训练使用固定宽度 Multi-Lane query。
    MAX_PACKED_TOKENS = CONTEXT_BLOCKS + PREDICTION_STEPS * TOKENS_PER_STEP

    # 原始几何序列：独立图元 BOS + 896 步候选；文字前缀另行拼接。
    CONTEXT_TOKENS = CONTEXT_STEPS * TOKENS_PER_STEP
    PREFIX_TOKENS = PREFIX_STEPS * TOKENS_PER_STEP

    # C_i 与图元 i 的所有 partial Q 共用 RoPE 图元位置 i。
    MAX_BLOCK_POSITIONS = TOTAL_CONTEXT_TOKENS
    # v3 生成器的硬预算为 896 条（包含背景与控制事件）。
    MAX_IMAGE_STEPS = 896

    def validate_sequence_config(self):
        if not isinstance(self.FIXED_QUERY_GRID, bool):
            raise ValueError("FIXED_QUERY_GRID 必须是 bool")
        if self.PREFIX_STEPS <= 0 or self.PREDICTION_STEPS <= 0:
            raise ValueError("PREFIX_STEPS 和 PREDICTION_STEPS 必须为正整数")
        if self.PREFIX_STEPS + self.PREDICTION_STEPS != self.CONTEXT_STEPS:
            raise ValueError(
                "PREFIX_STEPS + PREDICTION_STEPS 必须等于 CONTEXT_STEPS，"
                f"当前为 {self.PREFIX_STEPS} + {self.PREDICTION_STEPS} "
                f"!= {self.CONTEXT_STEPS}"
            )
        if self.CONTEXT_TOKENS != self.CONTEXT_STEPS * self.TOKENS_PER_STEP:
            raise ValueError("CONTEXT_TOKENS 与 CONTEXT_STEPS * TOKENS_PER_STEP 不一致")
        if self.TOKENS_PER_STEP != self.ATTRIBUTES_PER_PRIMITIVE:
            raise ValueError("每图元 token 数与离散图元属性数不一致")
        if self.CONTEXT_STEPS % self.BLOCK_PRIMITIVES != 0:
            raise ValueError("CONTEXT_STEPS 必须能被每 block 图元数整除")
        if self.CONTEXT_BLOCKS != self.CONTEXT_STEPS // self.BLOCK_PRIMITIVES:
            raise ValueError("CONTEXT_BLOCKS 配置不一致")
        if self.FIRST_QUERY_BLOCK != self.PREFIX_STEPS // self.BLOCK_PRIMITIVES:
            raise ValueError("FIRST_QUERY_BLOCK 配置不一致")
        if self.BRANCH_QUERY_BLOCKS != self.CONTEXT_BLOCKS - self.FIRST_QUERY_BLOCK:
            raise ValueError("BRANCH_QUERY_BLOCKS 配置不一致")
        if self.PREFIX_TOKENS != self.PREFIX_STEPS * self.TOKENS_PER_STEP:
            raise ValueError("PREFIX_TOKENS 与 PREFIX_STEPS * TOKENS_PER_STEP 不一致")
        supervised_tokens = self.CONTEXT_TOKENS - self.PREFIX_TOKENS
        if supervised_tokens != self.PREDICTION_STEPS * self.TOKENS_PER_STEP:
            raise ValueError("预测区 token 数与 PREDICTION_STEPS 不一致")
        if self.MAX_BLOCK_POSITIONS < self.CONTEXT_BLOCKS:
            raise ValueError("RoPE block 位置容量不足")
        if self.MAX_BLOCK_POSITIONS < self.TOTAL_CONTEXT_TOKENS:
            raise ValueError("RoPE 位置容量小于文字与图元总上下文。")
        if self.TEXT_MAX_TOKENS + self.PREFIX_STEPS >= self.TOTAL_CONTEXT_TOKENS:
            raise ValueError("文字前缀挤占了全部可预测图元位置。")

    # ==========================================
    # 🧠 模型尺寸
    # ==========================================
    MODEL_VOCAB_SIZE = TOKEN_LAYOUT.vocab_size
    MODEL_ATTRIBUTE_VOCAB_SIZE = TOKEN_LAYOUT.attribute_vocab_size
    # Transformer 的隐藏宽度，也是 Q/K/V 投影后的总注意力维度。
    # 均衡扩容：1,004,182,914 参数；每注意力头 64 维。
    MODEL_D_MODEL = 1536
    MODEL_N_LAYER = 35
    MODEL_N_HEAD = 24
    MODEL_FFN_HIDDEN = 4096
    MODEL_NORM_EPS = 1e-6
    MODEL_ROPE_BASE = 10000.0

    # EMA 只影响训练状态和 checkpoint，不改变模型参数或推理结构。
    # 关闭时不分配 EMA 模型，也不会在新 checkpoint 中保存 EMA 权重。
    ENABLE_EMA = False
    EMA_DECAY = 0.999

    # torch.compile 级别：
    # 0 = eager，不编译；
    # 1 = 已验证路径，编译 Multi-Lane forward 且允许 graph break；
    # 2 = 完整图路径，编译同一 forward 但要求 fullgraph=True。
    TORCH_COMPILE_LEVEL = int(os.environ.get("A800_COMPILE_LEVEL", "0"))
    TORCH_COMPILE_BACKEND = "inductor"
    TORCH_COMPILE_MODE = "default"
    TORCH_COMPILE_DYNAMIC = False
    # 单个 DDP rank 的 Inductor 异步编译 worker 数。默认 32 会在
    # 三 rank 冷编译时制造近 96 个子进程并耗尽主机内存。
    TORCH_COMPILE_THREADS = 4
    # 持久化 FX/AOTAutograd/Inductor/Triton 产物，避免 /tmp 被清理或
    # 重启后重新冷编译。图、shape、dtype、版本或设备改变时，
    # Inductor 会自动生成新缓存键，不会误用旧产物。
    TORCH_COMPILE_CACHE_DIR = str(RUNTIME_ROOT / "compile_cache")

    def validate_model_config(self):
        if (
            self.MODEL_D_MODEL <= 0
            or self.MODEL_N_LAYER <= 0
            or self.MODEL_N_HEAD <= 0
            or self.MODEL_FFN_HIDDEN <= 0
        ):
            raise ValueError("模型宽度、层数、head 数和 FFN hidden 必须为正整数")
        if self.MODEL_VOCAB_SIZE != TOKEN_LAYOUT.vocab_size:
            raise ValueError(
                f"MODEL_VOCAB_SIZE={self.MODEL_VOCAB_SIZE} 与集中式词表布局 "
                f"{TOKEN_LAYOUT.vocab_size} 不一致"
            )
        if self.MODEL_ATTRIBUTE_VOCAB_SIZE != TOKEN_LAYOUT.attribute_vocab_size:
            raise ValueError(
                "MODEL_ATTRIBUTE_VOCAB_SIZE 与属性输出词表布局不一致"
            )
        if self.TEXT_VOCAB_SIZE != 1500 or not 2 <= self.TEXT_MAX_TOKENS <= 512:
            raise ValueError("文字词表或最大长度不符合当前分词器。")
        if not 0.0 <= self.TEXT_DROPOUT_PROBABILITY < 1.0:
            raise ValueError("文字空条件比例必须位于 [0,1)。")
        if self.MODEL_D_MODEL % self.MODEL_N_HEAD != 0:
            raise ValueError(
                f"MODEL_D_MODEL ({self.MODEL_D_MODEL}) 必须能被 "
                f"MODEL_N_HEAD ({self.MODEL_N_HEAD}) 整除"
            )
        if self.MODEL_NORM_EPS <= 0 or self.MODEL_ROPE_BASE <= 0:
            raise ValueError("MODEL_NORM_EPS 和 MODEL_ROPE_BASE 必须为正数")
        if not isinstance(self.ENABLE_EMA, bool):
            raise ValueError("ENABLE_EMA 必须是 bool")
        if not 0.0 < self.EMA_DECAY < 1.0:
            raise ValueError("EMA_DECAY 必须在 (0, 1) 内")
        if (
            not isinstance(self.TORCH_COMPILE_LEVEL, int)
            or isinstance(self.TORCH_COMPILE_LEVEL, bool)
            or self.TORCH_COMPILE_LEVEL not in {0, 1, 2}
        ):
            raise ValueError("TORCH_COMPILE_LEVEL 必须是 0、1 或 2")
        if not self.TORCH_COMPILE_BACKEND or not self.TORCH_COMPILE_MODE:
            raise ValueError("torch.compile backend 和 mode 不能为空")
        if (
            not isinstance(self.TORCH_COMPILE_THREADS, int)
            or isinstance(self.TORCH_COMPILE_THREADS, bool)
            or self.TORCH_COMPILE_THREADS < 1
        ):
            raise ValueError("TORCH_COMPILE_THREADS 必须是正整数")
        if not self.TORCH_COMPILE_CACHE_DIR:
            raise ValueError("TORCH_COMPILE_CACHE_DIR 不能为空")
        if (
            not isinstance(self.RAND_K, int)
            or not 1 <= self.RAND_K <= self.BLOCK_VALUE_COUNT
        ):
            raise ValueError(
                f"RAND_K 必须是 [1, {self.BLOCK_VALUE_COUNT}] 内的整数，"
                f"收到 {self.RAND_K!r}"
            )
        if self.RAND_K_SAMPLING != "random":
            raise ValueError(
                "首版 RAND_K_SAMPLING 仅支持 'random'（均匀有放回采样）"
            )
        if not isinstance(self.RAND_K_SEED, int):
            raise ValueError("RAND_K_SEED 必须是整数")
    # ==========================================
    # 🔄 数据平权与增强策略
    # ==========================================
    VERSION_CYCLE_EPOCHS = 20
    ENABLE_FLIP_AUGMENTATION = False 
     # 🔴 [新增] 训练集序列极速截断限制。0表示使用全部数据，N表示只取前N张图。
    LIMIT_TRAIN_SEQUENCES = 0
    # ==========================================
    # 🚀 训练超参
    # ==========================================
    # BF16 GPU 工作权重 + ZeRO-1 分片的 GPU FP32 master/AdamW。
    MICRO_BATCH_SIZE = int(os.environ.get("A800_MICRO_BATCH", "1"))
    # 每张 GPU 在一次 optimizer.step() 前累计的 micro-batch 数。
    # 等效全局 batch = 此值 * GPU 数 * MICRO_BATCH_SIZE。
    GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("A800_ACCUMULATION", "1"))
    # 每个可训练图元实际送入完整 Transformer 的 partial query 数量。
    # 1..8 使用合法位置内均匀有放回采样；9 保持完整回归路径。
    RAND_K = 2
    RAND_K_SAMPLING = "random"
    RAND_K_SEED = 42
    # 数值安全闸门：任何 rank 的 loss/gradient 非有限时，都禁止本次
    # optimizer.step()。偶发异常丢弃整个累计窗口；重复异常则停止训练。
    ENABLE_NUMERICAL_GUARD = True
    NONFINITE_MAX_CONSECUTIVE = 30
    NONFINITE_MAX_PER_EPOCH = 30
    # 首次遇到某个 CSV 时，以分块方式并行转换为 uint16 磁盘缓存。
    # 当前机器有 24 个物理核；限制为最多 12 个进程，避免并发解析撑爆内存。
    DATA_CACHE_BUILD_WORKERS = int(os.environ.get("A800_CACHE_BUILD_WORKERS", "4"))
    DATA_CACHE_CSV_CHUNK_ROWS = 250_000
    # ROCm 在主进程已经初始化 GPU 后再 fork DataLoader worker 可能永久阻塞。
    # Batch 直接由 mmap 填入循环复用的 pinned buffer，不创建 DataLoader 子进程。
    # 多核进程仅用于首次将 CSV 构建为分片缓存；下一 Epoch 缓存在后台提前构建。
    DATALOADER_NUM_WORKERS = 0

    # ==========================================
    # 🖥️ 并行训练策略
    # ==========================================
    # "manual_cpu_sync"：单进程多模型，经 CPU 分段合并梯度。
    # "manual_pcie_sync"：同一手动累计方案，但直接在多张 GPU 间传递分段张量。
    # "data_parallel"：使用 nn.DataParallel，由 PyTorch 在单进程内管理多卡。
    # "ddp"：推荐生产模式；torchrun 为每张 GPU 启动独立 Python 进程，
    # 梯度累计期间 no_sync，仅在窗口最后一个 micro-batch 执行 AllReduce。
    # "single"：单卡训练。
    # 完整保留 manual_cpu_sync 的计算、累计和数据流水，仅把累计窗口末尾的
    # 各副卡→GPU0 梯度合并及 GPU0→各副卡参数回传使用 PCIe P2P DMA。
    # 每张卡仍各自累计完整的 local_accum_steps，micro-batch 间不跨卡传输。
    PARALLEL_MODE = "single"
    PARALLEL_DEVICE_IDS = [0]
    # CPU staging 或 GPU P2P 的可复用同步缓冲区大小。
    PARALLEL_SYNC_BUFFER_CHUNK_MB = 48
    
    # ==========================================
    # 💾 断点保护机制 (随时可热更新)
    # ==========================================
    # 每隔多少个 Batch 执行一次强制保存？
    # 注意：底层会在达到该批次后，自动寻找最近的一个梯度更新点(accum==0)进行安全保存。
    SAVE_INTERVAL_BATCHES = 1_000_000_000  # 40 optimizer updates / 24000 global samples.

    def validate_parallel_config(self):
        if (
            not isinstance(self.GRADIENT_ACCUMULATION_STEPS, int)
            or isinstance(self.GRADIENT_ACCUMULATION_STEPS, bool)
            or self.GRADIENT_ACCUMULATION_STEPS < 1
        ):
            raise ValueError("GRADIENT_ACCUMULATION_STEPS 必须是正整数")
        if not isinstance(self.ENABLE_NUMERICAL_GUARD, bool):
            raise ValueError("ENABLE_NUMERICAL_GUARD 必须是 bool")
        for name in (
            "NONFINITE_MAX_CONSECUTIVE",
            "NONFINITE_MAX_PER_EPOCH",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} 必须是正整数")
        valid_modes = {
            "single",
            "manual_cpu_sync",
            "manual_pcie_sync",
            "data_parallel",
            "ddp",
        }
        if self.PARALLEL_MODE not in valid_modes:
            raise ValueError(
                f"PARALLEL_MODE 必须是以下之一: {sorted(valid_modes)}"
            )
        if self.PARALLEL_MODE == "single":
            return
        if (
            not isinstance(self.PARALLEL_DEVICE_IDS, (list, tuple))
            or len(self.PARALLEL_DEVICE_IDS) < 2
            or any(
                not isinstance(device_id, int) or device_id < 0
                for device_id in self.PARALLEL_DEVICE_IDS
            )
            or len(set(self.PARALLEL_DEVICE_IDS)) != len(self.PARALLEL_DEVICE_IDS)
        ):
            raise ValueError(
                "多卡模式要求 PARALLEL_DEVICE_IDS 至少包含两个不重复的非负整数"
            )
        if self.PARALLEL_SYNC_BUFFER_CHUNK_MB <= 0:
            raise ValueError("PARALLEL_SYNC_BUFFER_CHUNK_MB 必须大于 0")
