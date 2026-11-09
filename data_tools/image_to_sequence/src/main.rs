mod cli;
mod engine;
mod scheduling;
mod types;
mod v3;
use scheduling::{build_dispatches, ExecutionSettings, SubmissionWindow};
use v3::*;

use engine::GpuEngine;
use image::imageops::FilterType;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::io::{
    BufRead,
    BufReader,
    BufWriter,
    Write,
};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use types::{GpuShape, ImageState};
use wgpu::util::DeviceExt;

// ==========================================
// ⚙️ 全局生产力配置
// ==========================================
/// ① 每卡数据批量：一次读取并保留多少张图像（影响 I/O 和内存）。
const DEFAULT_IMAGES_PER_GPU_BATCH: usize = 16;
/// ② 单次 GPU 提交最多处理多少张图；小于数据批量时自动分片。
const DEFAULT_IMAGES_PER_GPU_DISPATCH: usize = 16;
/// ③ 每卡最多允许多少个未完成绘制提交；1 表示每次提交完成后再继续。
const DEFAULT_MAX_IN_FLIGHT_SUBMISSIONS: usize = 1;

const W: u32 = 512;
const H: u32 = 512;
const ALPHA: f32 = 0.5;
const PENALTY_FACTOR: f32 = 1.8;
/// true 时用原图 RGB 均值作为背景；false 时直接从纯白画布开始。
const INITIALIZE_WITH_MEAN_COLOR: bool = true;

/// 全局固定成功绘制 85 个图形，之后依次进入粗网格和细网格。
const PHASE_TWO_RANDOM_STEP_MIN: u32 = 85;
const PHASE_TWO_RANDOM_STEP_MAX: u32 = 85;
const GRID_SIDE: usize = 4;
const GRID_COUNT: usize = GRID_SIDE * GRID_SIDE;
/// 普通图形和 begin/end marker 共用的历史记录硬上限。
const DEFAULT_MAX_HISTORY_RECORDS: usize = 895;
const DEFAULT_MAX_SEARCH_ATTEMPTS: usize = 2000;
const DEFAULT_MAX_ASPECT_RATIO: f32 = 8.0;
const DEFAULT_REGION_SIMILARITY_THRESHOLD: f32 = 0.95;
const DEFAULT_GRID_MIN_SUCCESSFUL_SHAPES: u32 = 0;
const DEFAULT_GRID_MAX_SUCCESSFUL_SHAPES: u32 = 20;
const DEFAULT_IMAGE_LIMIT: usize = 16;

// ==========================================
// 🧬 GPU 搜索与变异配置
// 修改这里的值后重新执行 cargo run --release。
// 程序启动时会把这些值生成为 WGSL const，每张 GPU 只编译一次。
// ==========================================
/// 每一步先随机生成多少个图形参加海选；计算量近似随该值线性增加。
const INITIAL_CANDIDATE_COUNT: u32 = 50;
/// 海选后保留多少个候选继续爬山；必须不大于 INITIAL_CANDIDATE_COUNT。
const SURVIVOR_COUNT: u32 = 5;
/// 连续多少次变异没有改良后，采纳当前候选迄今找到的最佳参数。
const MUTATION_STAGNATION_LIMIT: u32 = 100;
/// 单个候选允许尝试的变异硬上限。
const MAX_MUTATIONS_PER_SURVIVOR: u32 = 10000;

const POSITION_MUTATION_RANGE: f32 = 56.0;
const WIDTH_MUTATION_RANGE: f32 = 56.0;
const HEIGHT_MUTATION_RANGE: f32 = 56.0;
/// 供人编辑时使用角度制；生成 WGSL 时转换成弧度。
const ANGLE_MUTATION_RANGE_DEGREES: f32 = 16.0;
const MIN_HALF_SIZE: f32 = 1.0;
const MAX_HALF_SIZE: f32 = 256.0;

// 六类变异的相对权重。五类直边变异保持 1；贝塞尔边变异设为 2，
// 让搜索器有足够机会发现弯曲的收益。设为 0 可完全关闭弯曲（A/B 基线）。
const POSITION_MUTATION_WEIGHT: u32 = 1;
const WIDTH_MUTATION_WEIGHT: u32 = 1;
const HEIGHT_MUTATION_WEIGHT: u32 = 1;
const ANGLE_MUTATION_WEIGHT: u32 = 1;
const SHAPE_TYPE_MUTATION_WEIGHT: u32 = 1;
const EDGE_MUTATION_WEIGHT: u32 = 2;
/// 单次贝塞尔边变异的归一化步长（edge 取值域 [-1, 1]）。
const EDGE_MUTATION_RANGE: f32 = 0.20;

// 只控制进度回读和日志频率；不再控制在途提交数量。
const PROGRESS_READBACK_INTERVAL: usize = 25;
// 历史始终完整保存在显存中，只在最终输出时按图片块映射到 CPU。
const DEFAULT_HISTORY_READBACK_IMAGES: usize = 128;

const NUM_VERSIONS: usize = 1;
/// 默认同时使用的硬件 GPU 数。可用 SHAPE_RENDERER_GPU_COUNT 覆盖，例如设为 1 回退单卡。
const DEFAULT_GPU_COUNT: usize = 1;
/// 每个 data_part_N.csv 最大约 1 GiB。
const MAX_CSV_SIZE_BYTES: u64 =
    1024 * 1024 * 1024;

/// 当前输出目录自身真正已经写入 CSV 的图片。
const COMPLETED_IMAGES_FILE: &str =
    ".completed_images.txt";

/// 一个后台写盘批次的事务标记。
///
/// 只要它存在，就说明上一次进程可能在
///
///     CSV write
///       -> checkpoint write
///       -> transaction commit
///
/// 中途退出。
const WRITE_TRANSACTION_FILE: &str =
    ".csv_write_transaction";

// ==========================================
// 🧩 辅助结构体
// ==========================================
#[derive(Copy, Clone, Debug, PartialEq)]
struct ShaderConstants {
    canvas_width: u32,
    canvas_height: u32,
    shape_alpha: f32,
    penalty_factor: f32,
    max_search_attempts: usize,
    max_history_records: usize,
    phase_two_random_step_min: u32,
    phase_two_random_step_max: u32,
    max_aspect_ratio: f32,
    region_similarity_threshold: f32,
    grid_min_successful_shapes: u32,
    grid_max_successful_shapes: u32,
    initial_candidate_count: u32,
    survivor_count: u32,
    mutation_stagnation_limit: u32,
    max_mutations_per_survivor: u32,
    position_mutation_range: f32,
    width_mutation_range: f32,
    height_mutation_range: f32,
    angle_mutation_range_degrees: f32,
    min_half_size: f32,
    max_half_size: f32,
    edge_mutation_range: f32,
    mutation_weights: [u32; 6],
}

#[derive(Clone)]
struct HostImgData {
    target_pixels: Vec<u32>,
    bg_color: (u8, u8, u8),
    bg_packed: u32,
    initial_sse: f64,
}

#[derive(Clone, Copy, Debug)]
struct CsvState {
    file_idx: usize,
    current_size: u64,
}

struct CsvWriteJob {
    version: usize,
    image_groups: Vec<Vec<RenderedImage>>,
    completed_image_names: Vec<String>,
}

fn shader_constants(
    max_history_records: usize,
    max_search_attempts: usize,
    max_aspect_ratio: f32,
    region_similarity_threshold: f32,
    grid_min_successful_shapes: u32,
    grid_max_successful_shapes: u32,
) -> ShaderConstants {
    ShaderConstants {
        canvas_width: W,
        canvas_height: H,
        shape_alpha: ALPHA,
        penalty_factor: PENALTY_FACTOR,
        max_search_attempts,
        max_history_records,
        phase_two_random_step_min: PHASE_TWO_RANDOM_STEP_MIN,
        phase_two_random_step_max: PHASE_TWO_RANDOM_STEP_MAX,
        max_aspect_ratio,
        region_similarity_threshold,
        grid_min_successful_shapes,
        grid_max_successful_shapes,
        initial_candidate_count: INITIAL_CANDIDATE_COUNT,
        survivor_count: SURVIVOR_COUNT,
        mutation_stagnation_limit: MUTATION_STAGNATION_LIMIT,
        max_mutations_per_survivor: MAX_MUTATIONS_PER_SURVIVOR,
        position_mutation_range: POSITION_MUTATION_RANGE,
        width_mutation_range: WIDTH_MUTATION_RANGE,
        height_mutation_range: HEIGHT_MUTATION_RANGE,
        angle_mutation_range_degrees: ANGLE_MUTATION_RANGE_DEGREES,
        min_half_size: MIN_HALF_SIZE,
        max_half_size: MAX_HALF_SIZE,
        edge_mutation_range: EDGE_MUTATION_RANGE,
        mutation_weights: [
            POSITION_MUTATION_WEIGHT,
            WIDTH_MUTATION_WEIGHT,
            HEIGHT_MUTATION_WEIGHT,
            ANGLE_MUTATION_WEIGHT,
            SHAPE_TYPE_MUTATION_WEIGHT,
            EDGE_MUTATION_WEIGHT,
        ],
    }
}

#[cfg(test)]
fn default_shader_constants() -> ShaderConstants {
    shader_constants(
        DEFAULT_MAX_HISTORY_RECORDS,
        DEFAULT_MAX_SEARCH_ATTEMPTS,
        DEFAULT_MAX_ASPECT_RATIO,
        DEFAULT_REGION_SIMILARITY_THRESHOLD,
        DEFAULT_GRID_MIN_SUCCESSFUL_SHAPES,
        DEFAULT_GRID_MAX_SUCCESSFUL_SHAPES,
    )
}

fn validate_shader_constants(constants: &ShaderConstants) -> Result<u32, String> {
    if constants.canvas_width == 0 {
        return Err("❌ 参数 W=0 无效：画布宽度必须大于 0".to_string());
    }
    if constants.canvas_height == 0 {
        return Err("❌ 参数 H=0 无效：画布高度必须大于 0".to_string());
    }
    if !constants.shape_alpha.is_finite()
        || constants.shape_alpha <= 0.0
        || constants.shape_alpha > 1.0
    {
        return Err(format!(
            "❌ 参数 ALPHA={} 无效：必须是 (0.0, 1.0] 内的有限数",
            constants.shape_alpha
        ));
    }
    if !constants.penalty_factor.is_finite() || constants.penalty_factor <= 0.0 {
        return Err(format!(
            "❌ 参数 PENALTY_FACTOR={} 无效：必须是大于 0 的有限数",
            constants.penalty_factor
        ));
    }
    if constants.max_history_records == 0 || constants.max_history_records > u32::MAX as usize {
        return Err(format!(
            "❌ 参数 MAX_HISTORY_RECORDS={} 无效：合法范围为 1..={}",
            constants.max_history_records,
            u32::MAX
        ));
    }
    if constants.max_search_attempts == 0
        || constants.max_search_attempts > u32::MAX as usize - GRID_COUNT * 2
    {
        return Err(format!(
            "❌ 参数 MAX_SEARCH_ATTEMPTS={} 无效：合法范围为 1..={}",
            constants.max_search_attempts,
            u32::MAX as usize - GRID_COUNT * 2
        ));
    }
    if constants.phase_two_random_step_min == 0
        || constants.phase_two_random_step_min > constants.phase_two_random_step_max
        || constants.phase_two_random_step_max > constants.max_history_records as u32
    {
        return Err(format!(
            "❌ 阶段二切换范围 {}..={} 无效：须满足 1 <= 最小值 <= 最大值 <= MAX_HISTORY_RECORDS({})",
            constants.phase_two_random_step_min,
            constants.phase_two_random_step_max,
            constants.max_history_records,
        ));
    }
    if !constants.max_aspect_ratio.is_finite() || constants.max_aspect_ratio < 1.0 {
        return Err(format!(
            "❌ 参数 MAX_ASPECT_RATIO={} 无效：必须是大于等于 1.0 的有限数",
            constants.max_aspect_ratio
        ));
    }
    if !constants.region_similarity_threshold.is_finite()
        || constants.region_similarity_threshold < 0.0
        || constants.region_similarity_threshold > 1.0
    {
        return Err(format!(
            "❌ 参数 REGION_SIMILARITY_THRESHOLD={} 无效：必须是 [0.0, 1.0] 内的有限数",
            constants.region_similarity_threshold
        ));
    }
    if constants.grid_max_successful_shapes == 0
        || constants.grid_min_successful_shapes > constants.grid_max_successful_shapes
    {
        return Err(format!(
            "❌ 每宫格成功图形范围 {}..={} 无效：须满足 0 <= 最小值 <= 最大值且最大值 >= 1",
            constants.grid_min_successful_shapes, constants.grid_max_successful_shapes
        ));
    }
    if constants.initial_candidate_count == 0 {
        return Err("❌ 参数 INITIAL_CANDIDATE_COUNT=0 无效：初始候选数必须大于 0".to_string());
    }
    if constants.survivor_count == 0 || constants.survivor_count > constants.initial_candidate_count
    {
        return Err(format!(
            "❌ 参数 SURVIVOR_COUNT={} 无效：合法范围为 1..=INITIAL_CANDIDATE_COUNT({})",
            constants.survivor_count, constants.initial_candidate_count
        ));
    }
    if constants.mutation_stagnation_limit == 0 {
        return Err(
            "❌ 参数 MUTATION_STAGNATION_LIMIT=0 无效：无改良截止次数必须大于 0".to_string(),
        );
    }
    if constants.max_mutations_per_survivor == 0 {
        return Err("❌ 参数 MAX_MUTATIONS_PER_SURVIVOR=0 无效：变异硬上限必须大于 0".to_string());
    }
    if constants.mutation_stagnation_limit > constants.max_mutations_per_survivor {
        return Err(format!(
            "❌ 参数 MUTATION_STAGNATION_LIMIT={} 无效：不得大于 MAX_MUTATIONS_PER_SURVIVOR({})",
            constants.mutation_stagnation_limit, constants.max_mutations_per_survivor
        ));
    }
    if !constants.min_half_size.is_finite()
        || !constants.max_half_size.is_finite()
        || constants.min_half_size < 1.0
        || constants.min_half_size > constants.max_half_size
        || constants.max_half_size > 256.0
    {
        return Err(format!(
            "❌ 参数半尺寸范围 [{}, {}] 无效：必须满足 1.0 <= MIN_HALF_SIZE <= MAX_HALF_SIZE <= 256.0",
            constants.min_half_size, constants.max_half_size
        ));
    }

    for (name, value) in [
        ("POSITION_MUTATION_RANGE", constants.position_mutation_range),
        ("WIDTH_MUTATION_RANGE", constants.width_mutation_range),
        ("HEIGHT_MUTATION_RANGE", constants.height_mutation_range),
        (
            "ANGLE_MUTATION_RANGE_DEGREES",
            constants.angle_mutation_range_degrees,
        ),
        ("EDGE_MUTATION_RANGE", constants.edge_mutation_range),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "❌ 参数 {name}={value} 无效：必须是大于等于 0 的有限数"
            ));
        }
    }

    let total_weight = constants
        .mutation_weights
        .iter()
        .try_fold(0u32, |total, &weight| total.checked_add(weight))
        .ok_or_else(|| {
            format!(
                "❌ 五类变异权重 {:?} 的总和超过 u32 上限 {}",
                constants.mutation_weights,
                u32::MAX
            )
        })?;
    if total_weight == 0 {
        return Err(format!(
            "❌ 五类变异权重 {:?} 无效：至少一个权重必须大于 0",
            constants.mutation_weights
        ));
    }

    Ok(total_weight)
}

fn wgsl_f32_literal(value: f32) -> String {
    let mut literal = format!("{value:?}");
    if !literal.contains('.') && !literal.contains('e') && !literal.contains('E') {
        literal.push_str(".0");
    }
    literal
}

fn build_shader_source(constants: &ShaderConstants) -> Result<String, String> {
    let total_mutation_weight = validate_shader_constants(constants)?;
    let angle_radians = constants.angle_mutation_range_degrees.to_radians();
    let [
        position_weight,
        width_weight,
        height_weight,
        angle_weight,
        shape_type_weight,
        edge_weight,
    ] = constants.mutation_weights;

    let prefix = format!(
        "// 此区域由 main.rs 的启动参数自动生成，请勿直接编辑。\n\
const CANVAS_WIDTH: u32 = {}u;\n\
const CANVAS_HEIGHT: u32 = {}u;\n\
const SHAPE_ALPHA: f32 = {};\n\
const PENALTY_FACTOR: f32 = {};\n\
const MAX_SEARCH_ATTEMPTS: u32 = {}u;\n\
const MAX_HISTORY_RECORDS: u32 = {}u;\n\
const GRID_SIDE: u32 = {}u;\n\
const GRID_COUNT: u32 = {}u;\n\
const MAX_ASPECT_RATIO: f32 = {};\n\
const REGION_SIMILARITY_THRESHOLD: f32 = {};\n\
const GRID_MIN_SUCCESSFUL_SHAPES: u32 = {}u;\n\
const GRID_MAX_SUCCESSFUL_SHAPES: u32 = {}u;\n\
const INITIAL_CANDIDATE_COUNT: u32 = {}u;\n\
const SURVIVOR_COUNT: u32 = {}u;\n\
const MUTATION_STAGNATION_LIMIT: u32 = {}u;\n\
const MAX_MUTATIONS_PER_SURVIVOR: u32 = {}u;\n\
const POSITION_MUTATION_RANGE: f32 = {};\n\
const WIDTH_MUTATION_RANGE: f32 = {};\n\
const HEIGHT_MUTATION_RANGE: f32 = {};\n\
const ANGLE_MUTATION_RANGE_RADIANS: f32 = {};\n\
const EDGE_MUTATION_RANGE: f32 = {};\n\
const MIN_HALF_SIZE: f32 = {};\n\
const MAX_HALF_SIZE: f32 = {};\n\
const POSITION_MUTATION_WEIGHT: u32 = {}u;\n\
const WIDTH_MUTATION_WEIGHT: u32 = {}u;\n\
const HEIGHT_MUTATION_WEIGHT: u32 = {}u;\n\
const ANGLE_MUTATION_WEIGHT: u32 = {}u;\n\
const SHAPE_TYPE_MUTATION_WEIGHT: u32 = {}u;\n\
const EDGE_MUTATION_WEIGHT: u32 = {}u;\n\
const TOTAL_MUTATION_WEIGHT: u32 = {}u;\n\n",
        constants.canvas_width,
        constants.canvas_height,
        wgsl_f32_literal(constants.shape_alpha),
        wgsl_f32_literal(constants.penalty_factor),
        constants.max_search_attempts,
        constants.max_history_records,
        GRID_SIDE,
        GRID_COUNT,
        wgsl_f32_literal(constants.max_aspect_ratio),
        wgsl_f32_literal(constants.region_similarity_threshold),
        constants.grid_min_successful_shapes,
        constants.grid_max_successful_shapes,
        constants.initial_candidate_count,
        constants.survivor_count,
        constants.mutation_stagnation_limit,
        constants.max_mutations_per_survivor,
        wgsl_f32_literal(constants.position_mutation_range),
        wgsl_f32_literal(constants.width_mutation_range),
        wgsl_f32_literal(constants.height_mutation_range),
        wgsl_f32_literal(angle_radians),
        wgsl_f32_literal(constants.edge_mutation_range),
        wgsl_f32_literal(constants.min_half_size),
        wgsl_f32_literal(constants.max_half_size),
        position_weight,
        width_weight,
        height_weight,
        angle_weight,
        shape_type_weight,
        edge_weight,
        total_mutation_weight,
    );

    Ok(prefix + include_str!("shader.wgsl"))
}

fn print_shader_constants(constants: &ShaderConstants) {
    println!(
        "🧬 WGSL 固定参数 | 画布: {}x{} | 历史记录上限（不含背景）: {} | 搜索上限: {} | alpha: {} | 惩罚: {}",
        constants.canvas_width,
        constants.canvas_height,
        constants.max_history_records,
        constants.max_search_attempts,
        constants.shape_alpha,
        constants.penalty_factor
    );
    println!(
        "🎨 初始画布: {}",
        if INITIALIZE_WITH_MEAN_COLOR {
            "原图 RGB 均值"
        } else {
            "纯白色 (#FFFFFF)"
        }
    );
    println!(
        "🗺️ 三阶段绘制 | 全局成功图形数: {}..={} | 粗网格: 4x4（20个/块）| 细网格: 8x8（10个/块） | 宫格相似度阈值: {} | 每宫格成功图形: {}..={} ",
        constants.phase_two_random_step_min,
        constants.phase_two_random_step_max,
        constants.region_similarity_threshold,
        constants.grid_min_successful_shapes,
        constants.grid_max_successful_shapes,
    );
    println!(
        "🧬 搜索参数 | 初筛: {} | 保留: {} | 无改良截止: {} | 单候选硬上限: {}",
        constants.initial_candidate_count,
        constants.survivor_count,
        constants.mutation_stagnation_limit,
        constants.max_mutations_per_survivor
    );
    println!(
        "🧬 变异参数 | 位置/宽/高: ±{}/±{}/±{} px | 角度: ±{}° | 半尺寸: {}..={} | 最大长宽比: {} | 权重: {:?}",
        constants.position_mutation_range,
        constants.width_mutation_range,
        constants.height_mutation_range,
        constants.angle_mutation_range_degrees,
        constants.min_half_size,
        constants.max_half_size,
        constants.max_aspect_ratio,
        constants.mutation_weights
    );
}

/// 一个批次中属于某张 GPU 的连续图片分片及其常驻目标缓冲区。
struct GpuBatchShard {
    range: Range<usize>,
    target_buf: wgpu::Buffer,
    background_colors: Vec<u32>,
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp"
            )
        })
        .unwrap_or(false)
}

/// 递归发现图片并排序；不会把数据集中的进度文本误当成图片。
fn collect_image_paths(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut pending_dirs = vec![root.to_path_buf()];
    let mut paths = Vec::new();

    while let Some(dir) = pending_dirs.pop() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let path = entry.path();

            if file_type.is_dir() {
                pending_dirs.push(path);
            } else if (file_type.is_file() || (file_type.is_symlink() && path.is_file()))
                && is_supported_image(&path)
            {
                paths.push(path);
            }
        }
    }

    paths.sort_unstable();
    Ok(paths)
}

fn limit_image_paths(mut paths: Vec<PathBuf>, limit: usize) -> Vec<PathBuf> {
    paths.truncate(limit);
    paths
}

/// 将 total 个元素尽可能均匀地切成不超过 worker_count 个非空连续区间。
fn balanced_ranges(total: usize, worker_count: usize) -> Vec<Range<usize>> {
    if total == 0 || worker_count == 0 {
        return Vec::new();
    }

    let active_workers = worker_count.min(total);
    let base_size = total / active_workers;
    let remainder = total % active_workers;
    let mut start = 0;

    (0..active_workers)
        .map(|index| {
            let size = base_size + usize::from(index < remainder);
            let range = start..start + size;
            start += size;
            range
        })
        .collect()
}

/// 将连续数据切成固定上限的小块，供 staging buffer 分块回读使用。
fn chunked_ranges(total: usize, max_chunk_size: usize) -> Vec<Range<usize>> {
    if total == 0 || max_chunk_size == 0 {
        return Vec::new();
    }

    (0..total)
        .step_by(max_chunk_size)
        .map(|start| start..(start + max_chunk_size).min(total))
        .collect()
}

fn history_readback_images() -> usize {
    positive_usize_env("SHAPE_RENDERER_HISTORY_READBACK_IMAGES")
        .unwrap_or(DEFAULT_HISTORY_READBACK_IMAGES)
}

fn requested_gpu_count() -> usize {
    positive_usize_env("SHAPE_RENDERER_GPU_COUNT").unwrap_or(DEFAULT_GPU_COUNT)
}

fn image_limit() -> Result<usize, String> {
    positive_usize_env_or("SHAPE_RENDERER_IMAGE_LIMIT", DEFAULT_IMAGE_LIMIT)
}

fn validate_requested_gpu_count(requested: usize, available: usize) -> Result<(), String> {
    if requested > available {
        return Err(format!(
            "❌ 请求使用 {requested} 张 GPU，但仅成功初始化 {available} 张。请连接更多 GPU，或设置 SHAPE_RENDERER_GPU_COUNT={available}（单卡则设为 1）后重试。"
        ));
    }
    Ok(())
}

fn positive_usize_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

fn positive_usize_env_or(name: &str, default: usize) -> Result<usize, String> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(default);
    };
    value
        .parse::<usize>()
        .ok()
        .filter(|&parsed| parsed > 0)
        .ok_or_else(|| format!("❌ {name} 必须是正整数"))
}

fn nonnegative_usize_env_or(name: &str, default: usize) -> Result<usize, String> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(default);
    };
    value
        .parse::<usize>()
        .map_err(|_| format!("❌ {name} 必须是非负整数"))
}

fn finite_f32_env(name: &str, default: f32) -> Result<f32, String> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(default);
    };
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("❌ {name} 必须是有限浮点数"))?;
    if !parsed.is_finite() {
        return Err(format!("❌ {name} 必须是有限浮点数"));
    }
    Ok(parsed)
}

fn phase_two_random_step_bounds(max_history_records: usize) -> Result<(u32, u32), String> {
    let min = std::env::var("SHAPE_RENDERER_PHASE_TWO_RANDOM_MIN")
        .ok()
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|_| "❌ SHAPE_RENDERER_PHASE_TWO_RANDOM_MIN 必须是正整数".to_string())?
        .unwrap_or(PHASE_TWO_RANDOM_STEP_MIN);
    let max = std::env::var("SHAPE_RENDERER_PHASE_TWO_RANDOM_MAX")
        .ok()
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|_| "❌ SHAPE_RENDERER_PHASE_TWO_RANDOM_MAX 必须是正整数".to_string())?
        .unwrap_or(PHASE_TWO_RANDOM_STEP_MAX);

    if min == 0 || min > max || max as usize > max_history_records {
        return Err(format!(
            "❌ 阶段二切换范围 {min}..={max} 无效：须满足 1 <= 最小值 <= 最大值 <= MAX_HISTORY_RECORDS({max_history_records})"
        ));
    }
    Ok((min, max))
}

fn input_dir() -> PathBuf {
    std::env::var_os("SHAPE_RENDERER_INPUT_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .expect("input path validated before GPU initialization")
}

fn output_dir() -> PathBuf {
    std::env::var_os("SHAPE_RENDERER_OUTPUT_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .expect("output path validated before GPU initialization")
}

fn version_output_dir(output_dir: &Path, version: usize) -> PathBuf {
    output_dir.join(format!("v{}", version + 1))
}

fn resume_seed_dir() -> Option<PathBuf> {
    std::env::var_os(
        "SHAPE_RENDERER_RESUME_SEED_DIR"
    )
    .filter(|value| !value.is_empty())
    .map(PathBuf::from)
}


fn load_resume_seed_counts(
    seed_dir: &Path,
    versions: usize,
) -> Result<Vec<usize>, String> {
    let path =
        seed_dir.join("snapshot_complete.json");

    let value: serde_json::Value =
        serde_json::from_slice(
            &fs::read(&path).map_err(|err| {
                format!(
                    "❌ 无法读取外部续生元数据 {}: {err}",
                    path.display()
                )
            })?,
        )
        .map_err(|err| {
            format!(
                "❌ 无法解析外部续生元数据 {}: {err}",
                path.display()
            )
        })?;

    if value["complete"] != true {
        return Err(format!(
            "❌ 外部续生快照未标记 complete=true: {}",
            path.display()
        ));
    }

    if value["format"].as_str() != Some(FORMAT) {
        return Err(format!(
            "❌ 外部续生快照 format 不匹配: {}",
            path.display()
        ));
    }

    if value["versions"].as_u64()
        != Some(versions as u64)
    {
        return Err(format!(
            "❌ 外部续生快照版本数不匹配: {}",
            path.display()
        ));
    }

    let counts = value["counts"]
        .as_array()
        .ok_or_else(|| {
            format!(
                "❌ 外部续生快照缺少 counts: {}",
                path.display()
            )
        })?;

    if counts.len() != versions {
        return Err(format!(
            "❌ 外部续生 counts 长度 {} != versions {}",
            counts.len(),
            versions
        ));
    }

    counts
        .iter()
        .enumerate()
        .map(|(version, value)| {
            value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    format!(
                        "❌ 外部续生 v{} count 无效",
                        version + 1
                    )
                })
        })
        .collect()
}


fn read_name_set(
    path: &Path,
) -> std::io::Result<HashSet<String>> {
    let file = fs::File::open(path)?;

    let mut names =
        HashSet::new();

    for line in BufReader::new(file).lines() {
        let name = line?;
        let name = name.trim();

        if !name.is_empty() {
            names.insert(
                name.to_string()
            );
        }
    }

    Ok(names)
}


fn completed_images_from_resume_seed(
    seed_dir: &Path,
    version: usize,
) -> Result<HashSet<String>, String> {
    let path =
        seed_dir.join(
            format!("v{}.txt", version + 1)
        );

    read_name_set(&path).map_err(|err| {
        format!(
            "❌ 无法读取外部续生清单 {}: {err}",
            path.display()
        )
    })
}


fn csv_part_paths(
    version_dir: &Path,
) -> std::io::Result<Vec<(usize, PathBuf)>> {
    let mut parts = Vec::new();

    for entry in fs::read_dir(version_dir)? {
        let entry = entry?;

        let name =
            entry.file_name();

        let name =
            name.to_string_lossy();

        let Some(index) = name
            .strip_prefix("data_part_")
            .and_then(|name| {
                name.strip_suffix(".csv")
            })
            .and_then(|number| {
                number.parse::<usize>().ok()
            })
        else {
            continue;
        };

        parts.push(
            (index, entry.path())
        );
    }

    parts.sort_unstable_by_key(
        |(index, _)| *index
    );

    Ok(parts)
}


fn completed_images_from_csv(
    version_dir: &Path,
) -> std::io::Result<HashSet<String>> {
    let mut completed =
        HashSet::new();

    for (_, csv_path)
        in csv_part_paths(version_dir)?
    {
        let file =
            fs::File::open(&csv_path)?;

        let mut lines =
            BufReader::new(file).lines();

        let Some(first_line) =
            lines.next().transpose()?
        else {
            continue;
        };

        if first_line
            != HEADER.trim_end_matches('\n')
        {
            return Err(
                std::io::Error::other(
                    format!(
                        "{} has wrong CSV header",
                        csv_path.display()
                    )
                )
            );
        }

        for line in lines {
            let line = line?;

            let mut fields =
                line.split(',');

            let Some(image_name) =
                fields.next()
            else {
                continue;
            };

            // 已取走 image_name。
            //
            // nth(4):
            // cx=0
            // cy=1
            // w=2
            // h=3
            // shape_type=4
            let shape_type =
                fields.nth(4);

            if shape_type == Some("-1")
                && !image_name.is_empty()
            {
                completed.insert(
                    image_name.to_string()
                );
            }
        }
    }

    Ok(completed)
}


#[cfg(unix)]
fn sync_directory(
    path: &Path,
) -> std::io::Result<()> {
    fs::File::open(path)?.sync_all()
}


#[cfg(not(unix))]
fn sync_directory(
    _path: &Path,
) -> std::io::Result<()> {
    Ok(())
}


fn rebuild_checkpoint(
    version_dir: &Path,
    completed: &HashSet<String>,
) -> std::io::Result<()> {
    let checkpoint =
        version_dir.join(
            COMPLETED_IMAGES_FILE
        );

    let temporary =
        version_dir.join(
            ".completed_images.rebuild.tmp"
        );

    let mut names: Vec<&str> =
        completed
            .iter()
            .map(String::as_str)
            .collect();

    names.sort_unstable();

    {
        let file =
            fs::File::create(
                &temporary
            )?;

        let mut writer =
            BufWriter::new(file);

        for name in names {
            writeln!(
                writer,
                "{name}"
            )?;
        }

        writer.flush()?;

        writer
            .get_ref()
            .sync_all()?;
    }

    if checkpoint.exists() {
        fs::remove_file(
            &checkpoint
        )?;
    }

    fs::rename(
        temporary,
        checkpoint,
    )?;

    sync_directory(
        version_dir
    )
}


fn completed_images_for_version(
    output_dir: &Path,
    version: usize,
) -> Result<HashSet<String>, String> {
    let version_dir =
        version_output_dir(
            output_dir,
            version
        );

    let checkpoint =
        version_dir.join(
            COMPLETED_IMAGES_FILE
        );

    // 正常快速路径。
    //
    // 即使 checkpoint 是空文件也可信。
    if checkpoint.is_file() {
        return read_name_set(
            &checkpoint
        )
        .map_err(|err| {
            format!(
                "❌ 无法读取 v{} checkpoint {}: {err}",
                version + 1,
                checkpoint.display()
            )
        });
    }

    // 兼容路径：
    // 有 CSV、没有 checkpoint 时，
    // 从 background 行恢复一次。
    let completed =
        completed_images_from_csv(
            &version_dir
        )
        .map_err(|err| {
            format!(
                "❌ 无法扫描 v{} CSV 恢复 checkpoint: {err}",
                version + 1
            )
        })?;

    rebuild_checkpoint(
        &version_dir,
        &completed,
    )
    .map_err(|err| {
        format!(
            "❌ 无法重建 v{} checkpoint: {err}",
            version + 1
        )
    })?;

    Ok(completed)
}


fn csv_state_for_version(
    output_dir: &Path,
    version: usize,
) -> std::io::Result<CsvState> {
    let version_dir =
        version_output_dir(
            output_dir,
            version
        );

    let Some(
        (file_idx, path)
    ) = csv_part_paths(
        &version_dir
    )?
    .pop()
    else {
        return Ok(
            CsvState {
                file_idx: 1,
                current_size: 0,
            }
        );
    };

    Ok(CsvState {
        file_idx,
        current_size:
            fs::metadata(path)?.len(),
    })
}


fn transaction_path(
    version_dir: &Path,
) -> PathBuf {
    version_dir.join(
        WRITE_TRANSACTION_FILE
    )
}


fn begin_csv_transaction(
    output_dir: &Path,
    version: usize,
    state: CsvState,
) -> std::io::Result<()> {
    let version_dir =
        version_output_dir(
            output_dir,
            version
        );

    let marker =
        transaction_path(
            &version_dir
        );

    if marker.exists() {
        return Err(
            std::io::Error::other(
                format!(
                    "transaction marker already exists: {}",
                    marker.display()
                )
            )
        );
    }

    let checkpoint =
        version_dir.join(
            COMPLETED_IMAGES_FILE
        );

    let checkpoint_size =
        checkpoint
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0);

    let temporary =
        version_dir.join(
            ".csv_write_transaction.tmp"
        );

    if temporary.exists() {
        fs::remove_file(
            &temporary
        )?;
    }

    {
        let mut file =
            fs::File::create(
                &temporary
            )?;

        writeln!(
            file,
            "{} {} {}",
            state.file_idx,
            state.current_size,
            checkpoint_size
        )?;

        file.sync_all()?;
    }

    fs::rename(
        temporary,
        marker,
    )?;

    sync_directory(
        &version_dir
    )
}


fn parse_transaction_marker(
    path: &Path,
) -> std::io::Result<(usize, u64, u64)> {
    let text =
        fs::read_to_string(path)?;

    let mut values =
        text.split_whitespace();

    let file_idx =
        values
            .next()
            .ok_or_else(|| {
                std::io::Error::other(
                    "missing transaction file_idx"
                )
            })?
            .parse::<usize>()
            .map_err(|_| {
                std::io::Error::other(
                    "bad transaction file_idx"
                )
            })?;

    let csv_size =
        values
            .next()
            .ok_or_else(|| {
                std::io::Error::other(
                    "missing transaction csv_size"
                )
            })?
            .parse::<u64>()
            .map_err(|_| {
                std::io::Error::other(
                    "bad transaction csv_size"
                )
            })?;

    let checkpoint_size =
        values
            .next()
            .ok_or_else(|| {
                std::io::Error::other(
                    "missing transaction checkpoint_size"
                )
            })?
            .parse::<u64>()
            .map_err(|_| {
                std::io::Error::other(
                    "bad transaction checkpoint_size"
                )
            })?;

    if values.next().is_some() {
        return Err(
            std::io::Error::other(
                "transaction marker contains extra fields"
            )
        );
    }

    Ok((
        file_idx,
        csv_size,
        checkpoint_size,
    ))
}


fn recover_incomplete_csv_write(
    output_dir: &Path,
    version: usize,
) -> Result<(), String> {
    let version_dir =
        version_output_dir(
            output_dir,
            version
        );

    let marker =
        transaction_path(
            &version_dir
        );

    let temporary =
        version_dir.join(
            ".csv_write_transaction.tmp"
        );

    if !marker.exists() {
        // tmp 只能产生在正式 marker 写入之前。
        // 此时 CSV 尚未开始修改，可以安全删除。
        if temporary.exists() {
            fs::remove_file(
                &temporary
            )
            .map_err(|err| {
                format!(
                    "❌ 无法清理 v{} stale transaction tmp: {err}",
                    version + 1
                )
            })?;
        }

        return Ok(());
    }

    let (
        file_idx,
        csv_size,
        checkpoint_size,
    ) =
        parse_transaction_marker(
            &marker
        )
        .map_err(|err| {
            format!(
                "❌ v{} transaction marker 损坏: {err}",
                version + 1
            )
        })?;

    let parts =
        csv_part_paths(
            &version_dir
        )
        .map_err(|err| {
            format!(
                "❌ v{} 无法扫描 CSV part: {err}",
                version + 1
            )
        })?;

    let mut baseline_part_found =
        false;

    for (index, path)
        in parts
    {
        if index > file_idx {
            fs::remove_file(
                &path
            )
            .map_err(|err| {
                format!(
                    "❌ v{} 删除回滚 part {} 失败: {err}",
                    version + 1,
                    path.display()
                )
            })?;

            continue;
        }

        if index == file_idx {
            baseline_part_found =
                true;

            let actual_size =
                fs::metadata(
                    &path
                )
                .map_err(|err| {
                    format!(
                        "❌ v{} 无法读取 {}: {err}",
                        version + 1,
                        path.display()
                    )
                })?
                .len();

            if actual_size < csv_size {
                return Err(format!(
                    "❌ v{} CSV {} 比事务基线还短：{} < {}",
                    version + 1,
                    path.display(),
                    actual_size,
                    csv_size
                ));
            }

            let file =
                fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .map_err(|err| {
                        format!(
                            "❌ 无法打开 {} 回滚: {err}",
                            path.display()
                        )
                    })?;

            file.set_len(
                csv_size
            )
            .map_err(|err| {
                format!(
                    "❌ 无法截断 {}: {err}",
                    path.display()
                )
            })?;

            file.sync_all()
                .map_err(|err| {
                    format!(
                        "❌ 无法同步 {}: {err}",
                        path.display()
                    )
                })?;
        }
    }

    if csv_size > 0
        && !baseline_part_found
    {
        return Err(format!(
            "❌ v{} 回滚失败：data_part_{}.csv 不存在，但事务基线大小为 {}",
            version + 1,
            file_idx,
            csv_size
        ));
    }

    let checkpoint =
        version_dir.join(
            COMPLETED_IMAGES_FILE
        );

    if checkpoint.exists() {
        let actual_size =
            fs::metadata(
                &checkpoint
            )
            .map_err(|err| {
                format!(
                    "❌ 无法读取 {}: {err}",
                    checkpoint.display()
                )
            })?
            .len();

        if actual_size
            < checkpoint_size
        {
            return Err(format!(
                "❌ v{} checkpoint 比事务基线还短：{} < {}",
                version + 1,
                actual_size,
                checkpoint_size
            ));
        }

        let file =
            fs::OpenOptions::new()
                .write(true)
                .open(&checkpoint)
                .map_err(|err| {
                    format!(
                        "❌ 无法打开 {} 回滚: {err}",
                        checkpoint.display()
                    )
                })?;

        file.set_len(
            checkpoint_size
        )
        .map_err(|err| {
            format!(
                "❌ 无法截断 checkpoint: {err}"
            )
        })?;

        file.sync_all()
            .map_err(|err| {
                format!(
                    "❌ 无法同步 checkpoint: {err}"
                )
            })?;
    } else if checkpoint_size > 0 {
        return Err(format!(
            "❌ v{} checkpoint 丢失，但事务基线大小为 {}",
            version + 1,
            checkpoint_size
        ));
    }

    fs::remove_file(
        &marker
    )
    .map_err(|err| {
        format!(
            "❌ 无法删除 v{} transaction marker: {err}",
            version + 1
        )
    })?;

    if temporary.exists() {
        let _ =
            fs::remove_file(
                temporary
            );
    }

    sync_directory(
        &version_dir
    )
    .map_err(|err| {
        format!(
            "❌ 无法同步 v{} 目录: {err}",
            version + 1
        )
    })?;

    println!(
        "♻️ v{}：检测到未提交写盘，已回滚至 data_part_{} @ {} bytes",
        version + 1,
        file_idx,
        csv_size
    );

    Ok(())
}


fn append_completed_images(
    output_dir: &Path,
    version: usize,
    names: &[String],
) -> std::io::Result<()> {
    if names.is_empty() {
        return Ok(());
    }

    let checkpoint =
        version_output_dir(
            output_dir,
            version
        )
        .join(
            COMPLETED_IMAGES_FILE
        );

    let file =
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(checkpoint)?;

    let mut writer =
        BufWriter::new(file);

    for name in names {
        writeln!(
            writer,
            "{name}"
        )?;
    }

    writer.flush()?;

    writer
        .get_ref()
        .sync_all()
}


fn write_version_csv(
    output_dir: &Path,
    version: usize,
    image_groups:
        Vec<Vec<RenderedImage>>,
    state: &mut CsvState,
) -> std::io::Result<()> {
    let version_dir =
        version_output_dir(
            output_dir,
            version
        );

    let open_part =
        |index: usize|
         -> std::io::Result<fs::File> {
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(
                    version_dir.join(
                        format!(
                            "data_part_{index}.csv"
                        )
                    )
                )
        };

    let mut writer =
        BufWriter::with_capacity(
            4 * 1024 * 1024,
            open_part(
                state.file_idx
            )?,
        );

    if state.current_size == 0 {
        writer.write_all(
            HEADER.as_bytes()
        )?;

        state.current_size =
            HEADER.len() as u64;
    }

    for image in
        image_groups
            .into_iter()
            .flatten()
    {
        let bytes =
            image.csv.as_bytes();

        let bytes_len =
            bytes.len() as u64;

        if state.current_size
            + bytes_len
            > MAX_CSV_SIZE_BYTES
        {
            writer.flush()?;

            writer
                .get_ref()
                .sync_all()?;

            state.file_idx += 1;

            writer =
                BufWriter::with_capacity(
                    4 * 1024 * 1024,
                    open_part(
                        state.file_idx
                    )?,
                );

            writer.write_all(
                HEADER.as_bytes()
            )?;

            state.current_size =
                HEADER.len() as u64;
        }

        writer.write_all(
            bytes
        )?;

        state.current_size +=
            bytes_len;
    }

    writer.flush()?;

    writer
        .get_ref()
        .sync_all()
}


fn commit_csv_transaction(
    output_dir: &Path,
    version: usize,
) -> std::io::Result<()> {
    let version_dir =
        version_output_dir(
            output_dir,
            version
        );

    let marker =
        transaction_path(
            &version_dir
        );

    fs::remove_file(
        marker
    )?;

    sync_directory(
        &version_dir
    )
}


fn spawn_csv_writer(
    output_dir: PathBuf,
    mut csv_states: Vec<CsvState>,
) -> (
    mpsc::SyncSender<CsvWriteJob>,
    thread::JoinHandle<
        Result<(), String>
    >,
) {
    // 两个待写任务足以覆盖磁盘延迟，
    // 同时防止大量 CSV 文本堆积在 RAM。
    let (sender, receiver) =
        mpsc::sync_channel::<CsvWriteJob>(2);

    let handle =
        thread::spawn(
            move || -> Result<(), String> {
        while let Ok(job) =
            receiver.recv()
        {
            // 在碰磁盘前先验证：
            //
            // 1. GPU 返回数量必须等于 checkpoint 数量。
            // 2. 顺序必须一致。
            // 3. 不允许把 invalid-for-training
            //    样本混进不可逆的大 CSV。
            {
                let images =
                    job.image_groups
                        .iter()
                        .flatten()
                        .collect::<Vec<_>>();

                if images.len()
                    != job.completed_image_names.len()
                {
                    return Err(format!(
                        "❌ v{} 写盘数量不一致：RenderedImage={} checkpoint={}",
                        job.version + 1,
                        images.len(),
                        job.completed_image_names.len()
                    ));
                }

                for (image, expected_name)
                    in images
                        .iter()
                        .zip(
                            job.completed_image_names.iter()
                        )
                {
                    if image.name
                        != *expected_name
                    {
                        return Err(format!(
                            "❌ v{} 写盘顺序不一致：{} != {}",
                            job.version + 1,
                            image.name,
                            expected_name
                        ));
                    }

                    if image.summary[
                        "valid_for_training"
                    ] != true
                    {
                        return Err(format!(
                            "❌ v{} 图片 {} valid_for_training=false，拒绝写入训练 CSV",
                            job.version + 1,
                            image.name
                        ));
                    }
                }
            }

            begin_csv_transaction(
                &output_dir,
                job.version,
                csv_states[
                    job.version
                ],
            )
            .map_err(|err| {
                format!(
                    "❌ v{} 无法开始 CSV transaction: {err}",
                    job.version + 1
                )
            })?;

            write_version_csv(
                &output_dir,
                job.version,
                job.image_groups,
                &mut csv_states[
                    job.version
                ],
            )
            .map_err(|err| {
                format!(
                    "❌ v{} CSV 写入失败: {err}",
                    job.version + 1
                )
            })?;

            append_completed_images(
                &output_dir,
                job.version,
                &job.completed_image_names,
            )
            .map_err(|err| {
                format!(
                    "❌ v{} checkpoint 写入失败: {err}",
                    job.version + 1
                )
            })?;

            commit_csv_transaction(
                &output_dir,
                job.version,
            )
            .map_err(|err| {
                format!(
                    "❌ v{} transaction 提交失败: {err}",
                    job.version + 1
                )
            })?;

            println!(
                "  💾 v{}：{} 张图片已提交到大 CSV",
                job.version + 1,
                job.completed_image_names.len()
            );
        }

        Ok(())
    });

    (sender, handle)
}

fn preprocess_image(path: &Path) -> HostImgData {
    let img = image::open(path)
        .unwrap_or_else(|err| panic!("❌ 无法读取图片 {}: {err}", path.display()))
        .resize_exact(W, H, FilterType::Lanczos3)
        .into_rgb8();
    let mut target_pixels = vec![0u32; (W * H) as usize];
    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;

    for (i, pixel) in img.pixels().enumerate() {
        let r = pixel[0] as u32;
        let g = pixel[1] as u32;
        let b = pixel[2] as u32;
        target_pixels[i] = (r << 16) | (g << 8) | b;
        if INITIALIZE_WITH_MEAN_COLOR {
            sum_r += r as u64;
            sum_g += g as u64;
            sum_b += b as u64;
        }
    }

    let count = (W * H) as u64;
    let (bg_r, bg_g, bg_b) = if INITIALIZE_WITH_MEAN_COLOR {
        (
            (sum_r / count) as u8,
            (sum_g / count) as u8,
            (sum_b / count) as u8,
        )
    } else {
        (255, 255, 255)
    };
    let bg_packed = ((bg_r as u32) << 16) | ((bg_g as u32) << 8) | bg_b as u32;

    let mut initial_sse = 0.0f64;
    for target in &target_pixels {
        let tr = ((target >> 16) & 255) as f64;
        let tg = ((target >> 8) & 255) as f64;
        let tb = (target & 255) as f64;
        initial_sse +=
            (tr - bg_r as f64).powi(2) + (tg - bg_g as f64).powi(2) + (tb - bg_b as f64).powi(2);
    }
    if initial_sse == 0.0 {
        initial_sse = 1.0;
    }

    HostImgData {
        target_pixels,
        bg_color: (bg_r, bg_g, bg_b),
        bg_packed,
        initial_sse,
    }
}

fn create_gpu_shards(engines: &[GpuEngine], host_data: &[HostImgData]) -> Vec<GpuBatchShard> {
    balanced_ranges(host_data.len(), engines.len())
        .into_iter()
        .enumerate()
        .map(|(engine_index, range)| {
            let num_images = range.end - range.start;
            let pixel_count = num_images * (W * H) as usize;
            let mut flat_targets = Vec::with_capacity(pixel_count);
            let mut background_colors = Vec::with_capacity(num_images);

            for data in &host_data[range.clone()] {
                flat_targets.extend_from_slice(&data.target_pixels);
                background_colors.push(data.bg_packed);
            }

            let target_buf = engines[engine_index].device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Targets"),
                    contents: bytemuck::cast_slice(&flat_targets),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            );

            println!(
                "  🛠️ GPU {} ({}) 分配图片区间 [{}..{})，共 {} 张",
                engines[engine_index].gpu_index,
                engines[engine_index].name,
                range.start,
                range.end,
                num_images
            );

            GpuBatchShard {
                range,
                target_buf,
                background_colors,
            }
        })
        .collect()
}

async fn process_version_on_gpu(
    engine: &GpuEngine,
    shard: &GpuBatchShard,
    batch_paths: &[PathBuf],
    host_data: &[HostImgData],
    version: usize,
    phase_two_random_step_bounds: (u32, u32),
    shader_constants: &ShaderConstants,
    execution: &ExecutionSettings,
) -> Result<Vec<RenderedImage>, String> {
    let max_dispatch_steps = shader_constants
        .max_search_attempts
        .checked_add(shader_constants.max_history_records)
        .ok_or_else(|| "❌ 最大 dispatch 数溢出".to_string())?;

    let paths = &batch_paths[shard.range.clone()];
    let image_data = &host_data[shard.range.clone()];
    let num_images = paths.len();
    let (phase_two_random_step_min, _phase_two_random_step_max) = phase_two_random_step_bounds;

    println!(
        "  🔄 GPU {} ({}) 开始生成 v{}，图片数: {}",
        engine.gpu_index,
        engine.name,
        version + 1,
        num_images
    );

    let initial_states: Vec<ImageState> = image_data
        .iter()
        .zip(paths)
        .map(|(data, path)| {
            make_state(
                data.initial_sse as f32,
                image_seed(path, version),
                phase_two_random_step_min,
            )
        })
        .collect();

    let canvas_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Canvas"),
        size: (num_images * (W * H) as usize * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // 每个版本只上传每张图一个背景色（4 bytes），完整画布由 GPU 初始化，
    // 避免把 num_images * W * H 个像素从 CPU 重传到显存。
    let background_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Background Colors"),
            contents: bytemuck::cast_slice(&shard.background_colors),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let states_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("States"),
            contents: bytemuck::cast_slice(&initial_states),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let history_byte_size = (num_images
        * shader_constants.max_history_records
        * std::mem::size_of::<GpuShape>()) as u64;
    let states_byte_size = (num_images * std::mem::size_of::<ImageState>()) as u64;

    let history_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("History"),
        size: history_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dispatches = build_dispatches(
        engine,
        num_images,
        execution.images_per_dispatch,
        [
            &shard.target_buf,
            &canvas_buf,
            &states_buf,
            &history_buf,
            &background_buf,
        ],
    );
    println!(
        "  ⚙️ GPU {}: {} 张数据 / 每步 {} 次提交 / 在途上限 {}",
        engine.gpu_index,
        num_images,
        dispatches.len(),
        execution.max_in_flight
    );
    engine.dispatch_initialize_canvas(&dispatches[0].bind_group, num_images as u32 * W * H);
    let mut submissions = SubmissionWindow::new(execution.max_in_flight);

    let mut dispatched_steps = 0usize;
    for step in 0..max_dispatch_steps {
        submissions.submit_step(engine, &dispatches)?;
        dispatched_steps = step + 1;

        if (step + 1) % PROGRESS_READBACK_INTERVAL == 0 || step + 1 == max_dispatch_steps {
            submissions.drain(engine)?;
            let synced_states = engine
                .read_buffer_range::<ImageState>(
                    &states_buf,
                    0,
                    states_byte_size,
                    "Progress States Readback",
                )
                .await?;
            let completed = synced_states
                .iter()
                .filter(|state| state.is_done == 1)
                .count();
            println!(
                "  ⏳ GPU {} v{}: 已 dispatch {}/{} 步，图片完成 {}/{}",
                engine.gpu_index,
                version + 1,
                step + 1,
                max_dispatch_steps,
                completed,
                num_images
            );
            if completed == num_images {
                break;
            }
        }
    }

    let final_states = engine
        .read_buffer_range::<ImageState>(&states_buf, 0, states_byte_size, "States Readback")
        .await?;
    if let Some((index, state)) = final_states.iter().enumerate().find(|(_, state)| {
        state.is_done != 1
            || state.history_count as usize > shader_constants.max_history_records
            || !(1..=3).contains(&state.termination_reason)
    }) {
        return Err(format!(
            "GPU {} v{} 图片局部索引 {} 的最终状态异常: done={}, reason={}, history={}/{}",
            engine.gpu_index,
            version + 1,
            index,
            state.is_done,
            state.termination_reason,
            state.history_count,
            shader_constants.max_history_records
        ));
    }
    let all_grids = final_states
        .iter()
        .filter(|state| state.termination_reason == 1)
        .count();
    let history_limit = final_states
        .iter()
        .filter(|state| state.termination_reason == 2)
        .count();
    let attempt_limit = final_states
        .iter()
        .filter(|state| state.termination_reason == 3)
        .count();
    println!(
        "  🛑 GPU {} v{}: dispatch {} 步 | 三阶段完成 {} | {} 历史记录上限（不含背景） {} | {} 搜索上限 {}",
        engine.gpu_index,
        version + 1,
        dispatched_steps,
        all_grids,
        shader_constants.max_history_records,
        history_limit,
        shader_constants.max_search_attempts,
        attempt_limit
    );
    if attempt_limit > 0 {
        eprintln!(
            "  ⚠️ GPU {} v{}: {} 张图片在历史记录不足 {} 条时耗尽 {} 次搜索，已导出现有结果",
            engine.gpu_index,
            version + 1,
            attempt_limit,
            shader_constants.max_history_records,
            shader_constants.max_search_attempts
        );
    }

    let history_items_per_image = shader_constants.max_history_records;
    let history_bytes_per_image =
        (history_items_per_image * std::mem::size_of::<GpuShape>()) as u64;
    let readback_images = history_readback_images().min(num_images).max(1);
    let mut all_history = Vec::with_capacity(num_images * history_items_per_image);
    let readback_ranges = chunked_ranges(num_images, readback_images);
    let total_chunks = readback_ranges.len();
    println!(
        "  📥 GPU {} v{}: 历史保留在显存，按每块最多 {} 张图回读（共 {} 块）",
        engine.gpu_index,
        version + 1,
        readback_images,
        total_chunks
    );

    for (chunk_index, image_range) in readback_ranges.into_iter().enumerate() {
        let image_count = image_range.end - image_range.start;
        let source_offset = image_range.start as u64 * history_bytes_per_image;
        let chunk_byte_size = image_count as u64 * history_bytes_per_image;
        let label = format!(
            "History Readback GPU {} v{} chunk {}/{}",
            engine.gpu_index,
            version + 1,
            chunk_index + 1,
            total_chunks
        );
        let mut chunk = engine
            .read_buffer_range::<GpuShape>(&history_buf, source_offset, chunk_byte_size, &label)
            .await?;
        all_history.append(&mut chunk);
    }
    debug_assert_eq!(
        all_history.len(),
        history_byte_size as usize / std::mem::size_of::<GpuShape>()
    );

    let image_strings: Vec<RenderedImage> = paths
        .par_iter()
        .enumerate()
        .map(|(i, path)| {
            let state = &final_states[i];
            let begin = i * history_items_per_image;
            export_image(
                path,
                version,
                image_data[i].bg_color,
                &all_history[begin..begin + state.history_count as usize],
                state,
            )
        })
        .collect();

    println!(
        "  ✅ GPU {} ({}) 已完成 v{}",
        engine.gpu_index,
        engine.name,
        version + 1
    );
    Ok(image_strings)
}

async fn run() -> Result<(), String> {
    let execution = ExecutionSettings::from_env()?;
    println!(
        "⚙️ 每卡数据批量 {} | 单次提交图像上限 {} | 最大在途提交 {}",
        execution.images_per_gpu_batch, execution.images_per_dispatch, execution.max_in_flight
    );
    // 在访问数据集和创建 GPU 前完成参数校验与 WGSL 生成。
    let max_history_records = positive_usize_env_or(
        "SHAPE_RENDERER_MAX_HISTORY_RECORDS",
        DEFAULT_MAX_HISTORY_RECORDS,
    )?;
    if max_history_records > 895 {
        return Err(
            "❌ SHAPE_RENDERER_MAX_HISTORY_RECORDS must be <=895 (plus background =896)"
                .to_string(),
        );
    }
    let max_search_attempts = positive_usize_env_or(
        "SHAPE_RENDERER_MAX_SEARCH_ATTEMPTS",
        DEFAULT_MAX_SEARCH_ATTEMPTS,
    )?;
    let max_aspect_ratio =
        finite_f32_env("SHAPE_RENDERER_MAX_ASPECT_RATIO", DEFAULT_MAX_ASPECT_RATIO)?;
    let region_similarity_threshold = finite_f32_env(
        "SHAPE_RENDERER_REGION_SIMILARITY_THRESHOLD",
        DEFAULT_REGION_SIMILARITY_THRESHOLD,
    )?;
    let grid_min_successful_shapes = nonnegative_usize_env_or(
        "SHAPE_RENDERER_GRID_MIN_SUCCESSFUL_SHAPES",
        DEFAULT_GRID_MIN_SUCCESSFUL_SHAPES as usize,
    )?
    .try_into()
    .map_err(|_| "❌ SHAPE_RENDERER_GRID_MIN_SUCCESSFUL_SHAPES 超过 u32 上限".to_string())?;
    let grid_max_successful_shapes = positive_usize_env_or(
        "SHAPE_RENDERER_GRID_MAX_SUCCESSFUL_SHAPES",
        DEFAULT_GRID_MAX_SUCCESSFUL_SHAPES as usize,
    )?
    .try_into()
    .map_err(|_| "❌ SHAPE_RENDERER_GRID_MAX_SUCCESSFUL_SHAPES 超过 u32 上限".to_string())?;
    let (phase_two_random_step_min, phase_two_random_step_max) =
        phase_two_random_step_bounds(max_history_records)?;
    let mut shader_constants = shader_constants(
        max_history_records,
        max_search_attempts,
        max_aspect_ratio,
        region_similarity_threshold,
        grid_min_successful_shapes,
        grid_max_successful_shapes,
    );
    shader_constants.phase_two_random_step_min = phase_two_random_step_min;
    shader_constants.phase_two_random_step_max = phase_two_random_step_max;
    let shader_source = build_shader_source(&shader_constants)?;
    print_shader_constants(&shader_constants);
    println!(
        "🎯 阶段切换成功图形数（可由 SHAPE_RENDERER_PHASE_TWO_RANDOM_MIN/MAX 覆盖）: {}..={}",
        phase_two_random_step_min, phase_two_random_step_max
    );

    let requested_gpu_count = requested_gpu_count();
    let mut engines = GpuEngine::discover_all(&shader_source).await?;
    validate_requested_gpu_count(requested_gpu_count, engines.len())?;
    engines.truncate(requested_gpu_count);
    println!(
        "🎮 本次使用 {} 张 GPU（可由 SHAPE_RENDERER_GPU_COUNT 覆盖）",
        engines.len()
    );
    for engine in &engines {
        let capacity = execution
            .validate_device_capacity(&engine.device.limits(), W, H, max_history_records)
            .map_err(|error| format!("GPU {} ({}): {}", engine.gpu_index, engine.name, error))?;
        println!(
            "  📐 GPU {}: 像素缓冲区 {} 字节 / 单缓冲区上限 {} 字节 / 当前配置最多 {} 张/卡",
            engine.gpu_index,
            capacity.pixel_buffer_bytes,
            capacity.buffer_limit_bytes,
            capacity.max_images_per_gpu
        );
    }

    let input_dir = input_dir();
    let output_dir = output_dir();
    let num_versions = positive_usize_env_or("SHAPE_RENDERER_NUM_VERSIONS", NUM_VERSIONS)?;
    let resume_seed_dir =
    resume_seed_dir();

    let resume_seed_counts =
        match resume_seed_dir.as_ref() {
            Some(seed_dir) => {
                println!(
                    "🧩 外部续生快照: {}",
                    seed_dir.display()
                );

                Some(
                    load_resume_seed_counts(
                        seed_dir,
                        num_versions,
                    )?
                )
            }

            None => None,
        };
    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("❌ 无法创建输出目录 {}: {err}", output_dir.display()))?;
    let _output_lock = OutputLock::acquire(&output_dir)?;
    validate_manifest(
        &output_dir,
        &shader_constants,
        num_versions,
        MAX_CSV_SIZE_BYTES,
        resume_seed_dir.as_deref(),
    )?;
    for version in 0..num_versions {
        fs::create_dir_all(version_output_dir(&output_dir, version))
            .map_err(|err| format!("❌ 无法创建版本输出目录: {err}"))?;
    }

    println!("🔎 正在递归扫描输入目录: {}", input_dir.display());
    let paths = collect_image_paths(&input_dir)
        .map_err(|err| format!("❌ 无法扫描输入目录 {}: {err}", input_dir.display()))?;
    if paths.is_empty() {
        return Err("No supported input images found".into());
    }
    let discovered_image_count = paths.len();
    let selected_image_limit = image_limit()?;
    let paths = limit_image_paths(paths, selected_image_limit);
    println!(
        "🖼️ 共发现 {} 张输入图片，按排序选择前 {} 张（上限 {}）",
        discovered_image_count,
        paths.len(),
        selected_image_limit
    );

    let selected_image_names:
    HashSet<String> =
    paths
        .iter()
        .map(|path| {
            path.file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string()
        })
        .collect();


    let mut completed_images_by_version =
        Vec::with_capacity(
            num_versions
        );

    let mut csv_states =
        Vec::with_capacity(
            num_versions
        );

    let mut total_pending:
        usize = 0;


    for version in 0..num_versions {
        println!(
            "♻️ 正在恢复断点 v{} / {}...",
            version + 1,
            num_versions
        );

        // 必须在读取 checkpoint 和 CsvState
        // 之前完成事务回滚。
        recover_incomplete_csv_write(
            &output_dir,
            version,
        )?;

        let mut completed =
            completed_images_for_version(
                &output_dir,
                version
            )?;

        let local_count =
            completed.len();

        let mut imported_count =
            0usize;

        if let Some(seed_dir) =
            resume_seed_dir.as_ref()
        {
            let imported =
                completed_images_from_resume_seed(
                    seed_dir,
                    version,
                )?;

            let expected =
                resume_seed_counts
                    .as_ref()
                    .unwrap()[
                        version
                    ];

            if imported.len()
                != expected
            {
                return Err(format!(
                    "❌ 外部续生 v{} 数量不一致：文件={} snapshot_complete={}",
                    version + 1,
                    imported.len(),
                    expected
                ));
            }

            if let Some(unknown) =
                imported
                    .iter()
                    .find(|name| {
                        !selected_image_names
                            .contains(*name)
                    })
            {
                return Err(format!(
                    "❌ 外部续生 v{} 包含当前输入集合不存在的图片：{}",
                    version + 1,
                    unknown
                ));
            }

            imported_count =
                imported.len();

            completed.extend(
                imported
            );
        }

        let effective_completed =
            completed
                .iter()
                .filter(|name| {
                    selected_image_names
                        .contains(*name)
                })
                .count();

        let pending =
            paths.len()
                .saturating_sub(
                    effective_completed
                );

        total_pending +=
            pending;

        println!(
            "♻️ v{}：本地 CSV={}，外部续生={}，有效跳过={}，待生成={}",
            version + 1,
            local_count,
            imported_count,
            effective_completed,
            pending
        );

        completed_images_by_version
            .push(completed);

        csv_states.push(
            csv_state_for_version(
                &output_dir,
                version,
            )
            .map_err(|err| {
                format!(
                    "❌ 无法恢复 v{} CSV 写入位置: {err}",
                    version + 1
                )
            })?
        );
    }


    println!(
        "📊 当前运行总待生成 image-version: {}",
        total_pending
    );

    println!(
        "🧬 计划生成版本数: {} | 画布: {}x{} | 历史记录上限（不含背景）: {} | 搜索上限: {} | 大 CSV 分片上限: {} MiB",
        num_versions,
        W,
        H,
        shader_constants.max_history_records,
        shader_constants.max_search_attempts,
        MAX_CSV_SIZE_BYTES / 1024 / 1024
    );

    println!(
        "📂 输出目录: {}",
        output_dir.display()
    );

    let (
        csv_writer_sender,
        csv_writer_handle
    ) =
        spawn_csv_writer(
            output_dir.clone(),
            csv_states,
        );
    let per_gpu_batch_size = execution.images_per_gpu_batch;
    let gpu_batch_size = per_gpu_batch_size * engines.len();
    println!(
        "📦 每批最多 {} 张 | 每张 GPU 约 {} 张",
        gpu_batch_size, per_gpu_batch_size
    );

    for (batch_index, batch_paths) in paths.chunks(gpu_batch_size).enumerate() {
        let batch_has_pending_work = (0..num_versions).any(|version| {
            batch_paths.iter().any(|path| {
                !completed_images_by_version[version]
                    .contains(&path.file_stem().unwrap().to_string_lossy().to_string())
            })
        });
        if !batch_has_pending_work {
            println!(
                "♻️ 第 {} 批次已在全部 {} 个版本完成，跳过。",
                batch_index + 1,
                num_versions
            );
            continue;
        }
        let num_images = batch_paths.len();
        println!("📦 ====================================================");
        println!(
            "📦 开始处理第 {} 批次 | 包含 {} 张图片",
            batch_index + 1,
            num_images
        );
        let batch_start = Instant::now();

        let host_data: Vec<HostImgData> = batch_paths
            .par_iter()
            .map(|path| preprocess_image(path))
            .collect();
        // 完整批次的目标图缓冲区在所有版本间复用；以前续跑改造会在每个版本
        // 重新上传同一批目标图，这里恢复为一次上传、多版本使用。
        let mut full_batch_gpu_shards: Option<Vec<GpuBatchShard>> = None;

        for version in 0..num_versions {
            let pending_indices: Vec<usize> = batch_paths
                .iter()
                .enumerate()
                .filter_map(|(index, path)| {
                    (!completed_images_by_version[version]
                        .contains(&path.file_stem().unwrap().to_string_lossy().to_string()))
                    .then_some(index)
                })
                .collect();
            if pending_indices.is_empty() {
                println!("  ♻️ v{}：本批次均已完成，跳过。", version + 1);
                continue;
            }

            println!(
                "  🔄 并行生成变体 v{} / {}（续跑 {} 张，跳过 {} 张）",
                version + 1,
                num_versions,
                pending_indices.len(),
                batch_paths.len() - pending_indices.len()
            );

            let pending_storage = (pending_indices.len() != batch_paths.len()).then(|| {
                let paths = pending_indices
                    .iter()
                    .map(|&index| batch_paths[index].clone())
                    .collect::<Vec<_>>();
                let data = pending_indices
                    .iter()
                    .map(|&index| host_data[index].clone())
                    .collect::<Vec<_>>();
                (paths, data)
            });
            let (paths_to_process, host_data_to_process): (&[PathBuf], &[HostImgData]) =
                match pending_storage.as_ref() {
                    Some((paths, data)) => (paths, data),
                    None => (batch_paths, &host_data),
                };
            let partial_gpu_shards = pending_storage
                .as_ref()
                .map(|(_, data)| create_gpu_shards(&engines, data));
            let version_gpu_shards: &[GpuBatchShard] = match partial_gpu_shards.as_ref() {
                Some(shards) => shards,
                None => full_batch_gpu_shards
                    .get_or_insert_with(|| create_gpu_shards(&engines, &host_data)),
            };
            let version_active_engines = &engines[..version_gpu_shards.len()];
            let completed_image_names: Vec<String> = paths_to_process
                .iter()
                .map(|path| path.file_stem().unwrap().to_string_lossy().to_string())
                .collect();

            // IndexedParallelIterator::collect 保留 GPU/分片顺序，因此 CSV 顺序稳定且无并发写入。
            let image_groups: Result<Vec<Vec<RenderedImage>>, String> = version_active_engines
                .par_iter()
                .zip(version_gpu_shards.par_iter())
                .map(|(engine, shard)| {
                    pollster::block_on(process_version_on_gpu(
                        engine,
                        shard,
                        paths_to_process,
                        host_data_to_process,
                        version,
                        (phase_two_random_step_min, phase_two_random_step_max),
                        &shader_constants,
                        &execution,
                    ))
                })
                .collect();
            let image_groups = image_groups.map_err(|err| {
                format!(
                    "❌ v{} GPU 计算或回读失败，未写入该版本 CSV: {}",
                    version + 1,
                    err
                )
            })?;

            csv_writer_sender
                .send(CsvWriteJob {
                    version,
                    image_groups,
                    completed_image_names,
                })
                .map_err(|err| format!("❌ 后台写盘线程不可用: {err}"))?;
            println!(
                "  📨 v{} 结果已交给后台写盘，GPU 可继续处理下一任务",
                version + 1
            );
        }

        println!(
            "✅ 第 {} 批次的 {} 个版本已全部完成！批次耗时: {:?}",
            batch_index + 1,
            num_versions,
            batch_start.elapsed()
        );
    }

    drop(csv_writer_sender);
    csv_writer_handle
        .join()
        .map_err(|_| "❌ 后台 CSV 写盘线程异常退出".to_string())??;

    println!("🎉 所有图片及其变体版本均已生成、落盘并登记断点！");
    Ok(())
}

#[cfg(windows)]
fn configure_utf8_console() {
    const CP_UTF8: u32 = 65001;

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn SetConsoleCP(code_page: u32) -> i32;
        fn SetConsoleOutputCP(code_page: u32) -> i32;
    }

    // 设置失败时仍继续运行；也可以在终端中手动执行 `chcp 65001`。
    unsafe {
        let _ = SetConsoleCP(CP_UTF8);
        let _ = SetConsoleOutputCP(CP_UTF8);
    }
}

#[cfg(not(windows))]
fn configure_utf8_console() {}

fn main() {
    configure_utf8_console();
    match cli::configure() {
        Ok(false) => return,
        Ok(true) => {},
        Err(error) => {eprintln!("{error}");std::process::exit(2);}
    }
    if let Err(err) = pollster::block_on(run()) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod gpu_tests;
