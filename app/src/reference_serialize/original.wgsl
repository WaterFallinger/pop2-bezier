struct Config {
    image_offset: u32, dispatch_images: u32, num_images: u32, pad: u32,
};

struct GpuShape {
    cx: f32, cy: f32, hw: f32, hh: f32, theta: f32, shape_type: u32,
    r: u32, g: u32, b: u32, delta_err: f32, pad1: u32, pad2: u32,
};

struct ImageState {
    current_sse: f32, initial_sse: f32, region_sse: f32, phase_two_start_count: u32,
    is_done: u32, phase: u32, first_phase_shape_count: u32, successful_shape_count: u32,
    search_attempt_count: u32, history_count: u32, active_grid: u32, stage: u32,
    termination_reason: u32, grid_shape_count: u32, perturbed: u32, rng: u32,
    counts: array<u32,80>, closed: array<u32,80>, perturb: array<u32,80>,
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> target_tex: array<u32>;
@group(0) @binding(2) var<storage, read_write> canvas_tex: array<u32>;
@group(0) @binding(3) var<storage, read_write> states: array<ImageState>;
@group(0) @binding(4) var<storage, read_write> history: array<GpuShape>;
@group(0) @binding(5) var<storage, read> background_colors: array<u32>;

/// 每个版本在 GPU 内创建初始画布，背景数组每张图仅占 4 bytes。
const INITIALIZE_WORKGROUPS_PER_ROW: u32 = 65535u;

@compute @workgroup_size(256)
fn initialize_canvas(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let workgroup_index = group_id.y * INITIALIZE_WORKGROUPS_PER_ROW + group_id.x;
    let pixel_index = workgroup_index * 256u + local_id.x;
    let pixels_per_image = CANVAS_WIDTH * CANVAS_HEIGHT;
    let total_pixels = config.num_images * pixels_per_image;
    if (pixel_index < total_pixels) {
        canvas_tex[pixel_index] = background_colors[pixel_index / pixels_per_image];
    }
}

var<private> rng_state: u32;
fn pcg_hash() -> u32 {
    rng_state = rng_state * 747796405u + 2891336453u;
    let word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn rand_f32() -> f32 { return f32(pcg_hash()) / 4294967296.0; }
fn rand_range(min_v: f32, max_v: f32) -> f32 { return min_v + rand_f32() * (max_v - min_v); }

const PHASE_FULL_CANVAS: u32 = 0u;
const PHASE_GRID_BEGIN: u32 = 1u;
const PHASE_GRID_DRAWING: u32 = 2u;
const PHASE_GRID_END: u32 = 3u;
const REGION_BEGIN_MARKER_SHAPE_TYPE: u32 = 2u;
const REGION_END_MARKER_SHAPE_TYPE: u32 = 3u;
const TERMINATION_ALL_GRIDS: u32 = 1u;
const TERMINATION_HISTORY_LIMIT: u32 = 2u;
const TERMINATION_ATTEMPT_LIMIT: u32 = 3u;

// Encoded grid IDs: coarse 0..15, fine 16..79.
fn active_grid_id(state: ImageState) -> u32 { return state.active_grid; }
fn grid_side(id: u32) -> u32 { return select(4u, 8u, id >= 16u); }
fn local_grid(id: u32) -> u32 { return select(id, id - 16u, id >= 16u); }
fn grid_x_start(id: u32) -> u32 { return (local_grid(id) % grid_side(id)) * CANVAS_WIDTH / grid_side(id); }
fn grid_x_end(id: u32) -> u32 { return ((local_grid(id) % grid_side(id)) + 1u) * CANVAS_WIDTH / grid_side(id); }
fn grid_y_start(id: u32) -> u32 { return (local_grid(id) / grid_side(id)) * CANVAS_HEIGHT / grid_side(id); }
fn grid_y_end(id: u32) -> u32 { return ((local_grid(id) / grid_side(id)) + 1u) * CANVAS_HEIGHT / grid_side(id); }
fn grid_limit(id: u32) -> u32 { return select(GRID_MAX_SUCCESSFUL_SHAPES, 10u, id >= 16u); }
const PHASE_GRID_INTERRUPT: u32 = 4u;

// Per-image scheduling random stream, independent from candidate-search draws.
fn schedule_random(img: u32) -> u32 {
    var value = states[img].rng;
    value ^= value << 13u; value ^= value >> 17u; value ^= value << 5u;
    states[img].rng = value;
    return value;
}
fn bounded_random(img: u32, n: u32) -> u32 {
    let threshold = (0u - n) % n;
    var x = schedule_random(img);
    while (x < threshold) { x = schedule_random(img); }
    return x % n;
}
fn schedule_float(img: u32) -> f32 {
    return f32(schedule_random(img) >> 8u) / 16777216.0;
}
fn select_next_grid(img: u32) {
    var start = select(0u, 16u, states[img].stage == 2u);
    var end = select(16u, 80u, states[img].stage == 2u);
    var count = 0u;
    for (var id=start; id<end; id++) { if (states[img].closed[id] == 0u) { count++; } }
    if (count == 0u) {
        if (states[img].stage == 2u) {
            states[img].is_done = 1u;
            states[img].termination_reason = TERMINATION_ALL_GRIDS;
            return;
        }
        states[img].stage = 2u; start = 16u; end = 80u; count = 64u;
    }
    var choice = bounded_random(img, count);
    for (var id=start; id<end; id++) {
        if (states[img].closed[id] == 0u) {
            if (choice == 0u) {
                states[img].active_grid = id;
                states[img].phase = PHASE_GRID_BEGIN;
                return;
            }
            choice--;
        }
    }
}

// 阶段二仅限制图形中心；图形本身可以跨越宫格边界。
fn clamp_shape_center_to_grid(base: GpuShape, grid_id: u32) -> GpuShape {
    var shape = base;
    shape.cx = clamp(shape.cx, f32(grid_x_start(grid_id)), f32(grid_x_end(grid_id) - 1u));
    shape.cy = clamp(shape.cy, f32(grid_y_start(grid_id)), f32(grid_y_end(grid_id) - 1u));
    return shape;
}

fn gen_random_shape(restricted_to_grid: bool, grid_id: u32) -> GpuShape {
    var s: GpuShape;
    s.cx = rand_range(0.0, f32(CANVAS_WIDTH - 1u));
    s.cy = rand_range(0.0, f32(CANVAS_HEIGHT - 1u));
    // 随机选择一个锚定轴，再在合法区间内生成另一个轴，避免偏向宽或高。
    if ((pcg_hash() & 1u) == 0u) {
        s.hw = rand_range(MIN_HALF_SIZE, MAX_HALF_SIZE);
        s.hh = rand_range(
            max(MIN_HALF_SIZE, s.hw / MAX_ASPECT_RATIO),
            min(MAX_HALF_SIZE, s.hw * MAX_ASPECT_RATIO),
        );
    } else {
        s.hh = rand_range(MIN_HALF_SIZE, MAX_HALF_SIZE);
        s.hw = rand_range(
            max(MIN_HALF_SIZE, s.hh / MAX_ASPECT_RATIO),
            min(MAX_HALF_SIZE, s.hh * MAX_ASPECT_RATIO),
        );
    }
    s.theta = rand_range(0.0, 3.1415926535);
    s.shape_type = pcg_hash() % 2u;
    if (restricted_to_grid) {
        s = clamp_shape_center_to_grid(s, grid_id);
    }
    return s;
}

fn mutate_shape(
    base: GpuShape,
    restricted_to_grid: bool,
    grid_id: u32,
) -> GpuShape {
    var clone = base;
    let choice = pcg_hash() % TOTAL_MUTATION_WEIGHT;
    let width_start = POSITION_MUTATION_WEIGHT;
    let height_start = width_start + WIDTH_MUTATION_WEIGHT;
    let angle_start = height_start + HEIGHT_MUTATION_WEIGHT;
    let shape_type_start = angle_start + ANGLE_MUTATION_WEIGHT;
    if (choice < width_start) {
        clone.cx = clamp(clone.cx + rand_range(-POSITION_MUTATION_RANGE, POSITION_MUTATION_RANGE), 0.0, f32(CANVAS_WIDTH - 1u));
        clone.cy = clamp(clone.cy + rand_range(-POSITION_MUTATION_RANGE, POSITION_MUTATION_RANGE), 0.0, f32(CANVAS_HEIGHT - 1u));
    } else if (choice < height_start) {
        clone.hw = clamp(
            clone.hw + rand_range(-WIDTH_MUTATION_RANGE, WIDTH_MUTATION_RANGE),
            max(MIN_HALF_SIZE, clone.hh / MAX_ASPECT_RATIO),
            min(MAX_HALF_SIZE, clone.hh * MAX_ASPECT_RATIO),
        );
    } else if (choice < angle_start) {
        clone.hh = clamp(
            clone.hh + rand_range(-HEIGHT_MUTATION_RANGE, HEIGHT_MUTATION_RANGE),
            max(MIN_HALF_SIZE, clone.hw / MAX_ASPECT_RATIO),
            min(MAX_HALF_SIZE, clone.hw * MAX_ASPECT_RATIO),
        );
    } else if (choice < shape_type_start) {
        let pi = 3.1415926535;
        let mut_theta = clone.theta + rand_range(-ANGLE_MUTATION_RANGE_RADIANS, ANGLE_MUTATION_RANGE_RADIANS);
        
        // 🔄 核心修复：环形折叠 (Wrap-around)
        // 将角度完美地限制在 [0.0, PI) 的区间内。
        // 如果 mut_theta 是 -0.1，算出来会自动变成 3.04 (PI - 0.1)
        // 如果 mut_theta 是 3.24，算出来会自动变成 0.1 (3.24 - PI)
        clone.theta = mut_theta - pi * floor(mut_theta / pi);
    } else {
        clone.shape_type = 1u - clone.shape_type;
    }
    if (restricted_to_grid) {
        clone = clamp_shape_center_to_grid(clone, grid_id);
    }
    return clone;
}

// 🚀 核心黑魔法：SDF (符号距离场) 覆盖率计算
// 它会根据像素中心点到图形边缘的几何距离，计算出 0.0 到 1.0 的完美平滑覆盖率
fn get_coverage(
    shape_type: u32,
    hw: f32,
    hh: f32,
    dx: f32,
    dy: f32,
    cos_t: f32,
    sin_t: f32,
    inv_hw2: f32,
    inv_hh2: f32,
) -> f32 {
    let lx = dx * cos_t + dy * sin_t;
    let ly = -dx * sin_t + dy * cos_t;
    var d: f32 = 0.0;

    if (shape_type == 0u) {
        // 矩形的精确解析距离场
        let d_vec = abs(vec2<f32>(lx, ly)) - vec2<f32>(hw, hh);
        d = length(max(d_vec, vec2<f32>(0.0))) + min(max(d_vec.x, d_vec.y), 0.0);
    } else {
        // 椭圆的梯度近似解析距离场（利用一阶泰勒展开，极其高效）
        let f = (lx * lx) * inv_hw2 + (ly * ly) * inv_hh2 - 1.0;
        let gx = 2.0 * lx * inv_hw2;
        let gy = 2.0 * ly * inv_hh2;
        let g_len = length(vec2<f32>(gx, gy));
        
        if (g_len > 1e-5) {
            d = f / g_len;
        } else {
            d = f; // 中心点特殊处理
        }
    }
    
    // smoothstep 将边界 -0.5(内部) 到 0.5(外部) 的距离映射为 0.0 到 1.0
    // 1.0 - smoothstep 即反转为：内部为1，外部为0，边缘平滑渐变
    return 1.0 - smoothstep(-0.5, 0.5, d);
}

const WG_SIZE = 256u;
var<workgroup> sum_opt_r: array<f32, WG_SIZE>;
var<workgroup> sum_opt_g: array<f32, WG_SIZE>;
var<workgroup> sum_opt_b: array<f32, WG_SIZE>;
var<workgroup> sum_weight: array<f32, WG_SIZE>;
var<workgroup> sum_k2: array<f32, WG_SIZE>;
var<workgroup> sum_b_r: array<f32, WG_SIZE>;
var<workgroup> sum_b_g: array<f32, WG_SIZE>;
var<workgroup> sum_b_b: array<f32, WG_SIZE>;
var<workgroup> sum_c_r: array<f32, WG_SIZE>;
var<workgroup> sum_c_g: array<f32, WG_SIZE>;
var<workgroup> sum_c_b: array<f32, WG_SIZE>;

var<workgroup> shared_opt_r: f32;
var<workgroup> shared_opt_g: f32;
var<workgroup> shared_opt_b: f32;
var<workgroup> out_err: f32;
var<workgroup> current_shape: GpuShape;
var<workgroup> shared_cos: f32;
var<workgroup> shared_sin: f32;
var<workgroup> shared_inv_hw2: f32;
var<workgroup> shared_inv_hh2: f32;
var<workgroup> shared_min_x: i32;
var<workgroup> shared_max_x: i32;
var<workgroup> shared_min_y: i32;
var<workgroup> shared_max_y: i32;
var<workgroup> shared_box_w: u32;
var<workgroup> shared_box_h: u32;
var<workgroup> wg_age: u32;
var<workgroup> wg_apply_flag: u32;
var<workgroup> wg_skip_flag: u32;
var<workgroup> wg_action: u32;
var<workgroup> sum_actual_delta_sse: array<f32, WG_SIZE>;
var<workgroup> sum_region_delta_sse: array<f32, WG_SIZE>;

// 每个候选只由线程 0 计算一次三角函数、尺寸倒数和精确旋转包围盒。
// 1 像素余量覆盖 SDF 的抗锯齿过渡区。
fn prepare_current_shape(tid: u32) {
    if (tid == 0u) {
        shared_cos = cos(current_shape.theta);
        shared_sin = sin(current_shape.theta);
        shared_inv_hw2 = 1.0 / max(current_shape.hw * current_shape.hw, 0.0001);
        shared_inv_hh2 = 1.0 / max(current_shape.hh * current_shape.hh, 0.0001);

        let abs_cos = abs(shared_cos);
        let abs_sin = abs(shared_sin);
        var extent_x: f32;
        var extent_y: f32;

        if (current_shape.shape_type == 0u) {
            // 旋转矩形的精确轴对齐包围盒。
            extent_x = abs_cos * current_shape.hw + abs_sin * current_shape.hh;
            extent_y = abs_sin * current_shape.hw + abs_cos * current_shape.hh;
        } else {
            // 旋转椭圆的精确轴对齐包围盒。
            let hw_cos = current_shape.hw * shared_cos;
            let hh_sin = current_shape.hh * shared_sin;
            let hw_sin = current_shape.hw * shared_sin;
            let hh_cos = current_shape.hh * shared_cos;
            extent_x = sqrt(hw_cos * hw_cos + hh_sin * hh_sin);
            extent_y = sqrt(hw_sin * hw_sin + hh_cos * hh_cos);
        }

        let aa_margin = 1.0;
        shared_min_x = max(0, i32(floor(current_shape.cx - extent_x - aa_margin)));
        shared_max_x = min(
            i32(CANVAS_WIDTH - 1u),
            i32(ceil(current_shape.cx + extent_x + aa_margin)),
        );
        shared_min_y = max(0, i32(floor(current_shape.cy - extent_y - aa_margin)));
        shared_max_y = min(
            i32(CANVAS_HEIGHT - 1u),
            i32(ceil(current_shape.cy + extent_y + aa_margin)),
        );
        shared_box_w = u32(max(0, shared_max_x - shared_min_x + 1));
        shared_box_h = u32(max(0, shared_max_y - shared_min_y + 1));
    }
    workgroupBarrier();
}

fn eval_current_shape(img_offset: u32, tid: u32) {
    sum_opt_r[tid] = 0.0;
    sum_opt_g[tid] = 0.0;
    sum_opt_b[tid] = 0.0;
    sum_weight[tid] = 0.0;
    sum_k2[tid] = 0.0;
    sum_b_r[tid] = 0.0;
    sum_b_g[tid] = 0.0;
    sum_b_b[tid] = 0.0;
    sum_c_r[tid] = 0.0;
    sum_c_g[tid] = 0.0;
    sum_c_b[tid] = 0.0;
    prepare_current_shape(tid);

    // ==========================================
    // 单遍扫描：同时聚合当前颜色解与误差二次式的充分统计量。
    //
    // 对每个通道，k = alpha * coverage，d = target - canvas：
    // delta = (d - k * (opt - canvas))^2 - d^2
    //       = A * opt^2 + B * opt + C
    // 扫描结束后将当前算法的 opt 代入即可，无需第二次计算 SDF。
    // ==========================================
    var idx = tid;
    let total_pixels = shared_box_w * shared_box_h;
    let inv_alpha = 1.0 / SHAPE_ALPHA;
    while (idx < total_pixels) {
        let x = u32(shared_min_x) + (idx % shared_box_w);
        let y = u32(shared_min_y) + (idx / shared_box_w);
        let px_idx = img_offset + y * CANVAS_WIDTH + x;
        let dx = f32(x) - current_shape.cx;
        let dy = f32(y) - current_shape.cy;

        let coverage = get_coverage(
            current_shape.shape_type,
            current_shape.hw,
            current_shape.hh,
            dx,
            dy,
            shared_cos,
            shared_sin,
            shared_inv_hw2,
            shared_inv_hh2,
        );

        if (coverage > 0.0) {
            let trgb = target_tex[px_idx];
            let crgb = canvas_tex[px_idx];
            let tr = f32((trgb >> 16u) & 255u);
            let tg = f32((trgb >> 8u) & 255u);
            let tb = f32(trgb & 255u);
            let cr = f32((crgb >> 16u) & 255u);
            let cg = f32((crgb >> 8u) & 255u);
            let cb = f32(crgb & 255u);
            let dr = tr - cr;
            let dg = tg - cg;
            let db = tb - cb;
            let k = SHAPE_ALPHA * coverage;
            let k2 = k * k;

            sum_opt_r[tid] += (dr * inv_alpha + cr) * coverage;
            sum_opt_g[tid] += (dg * inv_alpha + cg) * coverage;
            sum_opt_b[tid] += (db * inv_alpha + cb) * coverage;
            sum_weight[tid] += coverage;
            sum_k2[tid] += k2;

            sum_b_r[tid] += -2.0 * k * dr - 2.0 * k2 * cr;
            sum_b_g[tid] += -2.0 * k * dg - 2.0 * k2 * cg;
            sum_b_b[tid] += -2.0 * k * db - 2.0 * k2 * cb;
            sum_c_r[tid] += 2.0 * k * dr * cr + k2 * cr * cr;
            sum_c_g[tid] += 2.0 * k * dg * cg + k2 * cg * cg;
            sum_c_b[tid] += 2.0 * k * db * cb + k2 * cb * cb;
        }
        idx += WG_SIZE;
    }
    workgroupBarrier();

    for (var s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sum_opt_r[tid] += sum_opt_r[tid + s];
            sum_opt_g[tid] += sum_opt_g[tid + s];
            sum_opt_b[tid] += sum_opt_b[tid + s];
            sum_weight[tid] += sum_weight[tid + s];
            sum_k2[tid] += sum_k2[tid + s];
            sum_b_r[tid] += sum_b_r[tid + s];
            sum_b_g[tid] += sum_b_g[tid + s];
            sum_b_b[tid] += sum_b_b[tid + s];
            sum_c_r[tid] += sum_c_r[tid + s];
            sum_c_g[tid] += sum_c_g[tid + s];
            sum_c_b[tid] += sum_c_b[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        if (sum_weight[0] > 0.0) {
            shared_opt_r = f32(u32(clamp(sum_opt_r[0] / sum_weight[0], 0.0, 255.0)));
            shared_opt_g = f32(u32(clamp(sum_opt_g[0] / sum_weight[0], 0.0, 255.0)));
            shared_opt_b = f32(u32(clamp(sum_opt_b[0] / sum_weight[0], 0.0, 255.0)));
        } else {
            shared_opt_r = 0.0;
            shared_opt_g = 0.0;
            shared_opt_b = 0.0;
        }

        var raw_err =
            sum_k2[0] * (
                shared_opt_r * shared_opt_r +
                shared_opt_g * shared_opt_g +
                shared_opt_b * shared_opt_b
            ) +
            sum_b_r[0] * shared_opt_r +
            sum_b_g[0] * shared_opt_g +
            sum_b_b[0] * shared_opt_b +
            sum_c_r[0] +
            sum_c_g[0] +
            sum_c_b[0];

        // 与原版相同的越界惩罚。
        let inside_area = sum_weight[0];
        var total_area = 0.0;
        if (current_shape.shape_type == 0u) {
            total_area = 4.0 * current_shape.hw * current_shape.hh;
        } else {
            total_area = 3.1415926535 * current_shape.hw * current_shape.hh;
        }

        let base_inside_ratio = inside_area / total_area;
        var oob_ratio = 1.0 - base_inside_ratio;
        if (oob_ratio < 0.0) { oob_ratio = 0.0; }

        let eff_oob = oob_ratio * PENALTY_FACTOR;
        var final_ratio = 1.0 - eff_oob;
        if (final_ratio < 0.001) { final_ratio = 0.001; }
        if (final_ratio > 1.0) { final_ratio = 1.0; }

        if (raw_err < 0.0) { out_err = raw_err * final_ratio; } else { out_err = raw_err / final_ratio; }
    }
    workgroupBarrier();
}

var<workgroup> top_shapes: array<GpuShape, SURVIVOR_COUNT>;
var<workgroup> top_errs: array<f32, SURVIVOR_COUNT>;

fn region_similarity(state: ImageState) -> f32 {
    let grid_id = active_grid_id(state);
    let pixel_count =
        (grid_x_end(grid_id) - grid_x_start(grid_id)) *
        (grid_y_end(grid_id) - grid_y_start(grid_id));
    let rmse = sqrt(max(0.0, state.region_sse) / f32(pixel_count * 3u));
    return 1.0 - rmse / 255.0;
}

fn is_pixel_in_grid(x: u32, y: u32, grid_id: u32) -> bool {
    return x >= grid_x_start(grid_id) && x < grid_x_end(grid_id) &&
        y >= grid_y_start(grid_id) && y < grid_y_end(grid_id);
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let dispatch_image = group_id.y * 256u + group_id.x;
    let img_idx = config.image_offset + dispatch_image; 

    // action: 0=跳过，1=写 begin marker 并初始化宫格 SSE，2=写 end marker，3=搜索。
    if (tid == 0u) {
        if (dispatch_image >= config.dispatch_images || img_idx >= config.num_images) {
            wg_skip_flag = 1u;
            wg_action = 0u;
        } else {
            wg_skip_flag = states[img_idx].is_done;
            wg_action = 0u;
            if (states[img_idx].is_done == 0u &&
                states[img_idx].history_count >= MAX_HISTORY_RECORDS) {
                states[img_idx].is_done = 1u;
                states[img_idx].termination_reason = TERMINATION_HISTORY_LIMIT;
                wg_skip_flag = 1u;
            }
            if (states[img_idx].is_done == 0u) {
                if (states[img_idx].phase == PHASE_FULL_CANVAS &&
                    states[img_idx].first_phase_shape_count >= states[img_idx].phase_two_start_count) {
                    states[img_idx].stage = 1u;
                    select_next_grid(img_idx);
                }

                // 至少成功绘制规定数量后才允许按相似度结束；达到宫格硬上限则无条件结束。
                if (states[img_idx].phase == PHASE_GRID_DRAWING &&
                    ((states[img_idx].grid_shape_count >= GRID_MIN_SUCCESSFUL_SHAPES &&
                      region_similarity(states[img_idx]) >= REGION_SIMILARITY_THRESHOLD) ||
                     states[img_idx].grid_shape_count >= grid_limit(states[img_idx].active_grid))) {
                    states[img_idx].phase = PHASE_GRID_END;
                }

                if (states[img_idx].phase == PHASE_GRID_END || states[img_idx].phase == PHASE_GRID_INTERRUPT) {
                    wg_action = 2u;
                } else if (states[img_idx].search_attempt_count >= MAX_SEARCH_ATTEMPTS) {
                    states[img_idx].is_done = 1u;
                    states[img_idx].termination_reason = TERMINATION_ATTEMPT_LIMIT;
                    wg_skip_flag = 1u;
                } else if (states[img_idx].phase == PHASE_GRID_BEGIN) {
                    wg_action = 1u;
                } else {
                    wg_action = 3u;
                }
            }
        }
    }
    let skip_workgroup = workgroupUniformLoad(&wg_skip_flag);
    if (skip_workgroup == 1u) { return; }
    let action = workgroupUniformLoad(&wg_action);

    let img_offset = img_idx * CANVAS_WIDTH * CANVAS_HEIGHT;

    if (action == 1u) {
        let grid_id = active_grid_id(states[img_idx]);
        let x_start = grid_x_start(grid_id);
        let x_end = grid_x_end(grid_id);
        let y_start = grid_y_start(grid_id);
        let y_end = grid_y_end(grid_id);
        let grid_width = x_end - x_start;
        let pixel_count = grid_width * (y_end - y_start);

        sum_region_delta_sse[tid] = 0.0;
        var region_index = tid;
        while (region_index < pixel_count) {
            let x = x_start + region_index % grid_width;
            let y = y_start + region_index / grid_width;
            let pixel_index = img_offset + y * CANVAS_WIDTH + x;
            let trgb = target_tex[pixel_index];
            let crgb = canvas_tex[pixel_index];
            let dr = f32((trgb >> 16u) & 255u) - f32((crgb >> 16u) & 255u);
            let dg = f32((trgb >> 8u) & 255u) - f32((crgb >> 8u) & 255u);
            let db = f32(trgb & 255u) - f32(crgb & 255u);
            sum_region_delta_sse[tid] += dr * dr + dg * dg + db * db;
            region_index += WG_SIZE;
        }
        workgroupBarrier();
        for (var s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
            if (tid < s) {
                sum_region_delta_sse[tid] += sum_region_delta_sse[tid + s];
            }
            workgroupBarrier();
        }

        if (tid == 0u) {
            let hist_idx = img_idx * MAX_HISTORY_RECORDS + states[img_idx].history_count;
            var marker: GpuShape;
            marker.cx = (f32(x_start) + f32(x_end - 1u)) * 0.5;
            marker.cy = (f32(y_start) + f32(y_end - 1u)) * 0.5;
            marker.hw = f32(x_end - x_start) * 0.5;
            marker.hh = f32(y_end - y_start) * 0.5;
            marker.theta = 0.0;
            marker.shape_type = REGION_BEGIN_MARKER_SHAPE_TYPE;
            marker.r = 0u;
            marker.g = 0u;
            marker.b = 0u;
            marker.delta_err = -1.0;
            marker.pad1 = states[img_idx].stage;
            marker.pad2 = local_grid(grid_id);
            history[hist_idx] = marker;
            states[img_idx].history_count += 1u;
            if (states[img_idx].history_count >= MAX_HISTORY_RECORDS) {
                states[img_idx].is_done = 1u;
                states[img_idx].termination_reason = TERMINATION_HISTORY_LIMIT;
            } else {
                states[img_idx].region_sse = sum_region_delta_sse[0];
                states[img_idx].grid_shape_count = states[img_idx].counts[grid_id];
                states[img_idx].phase = PHASE_GRID_DRAWING;
            }
        }
        return;
    }

    if (action == 2u) {
        if (tid == 0u) {
            let grid_id = active_grid_id(states[img_idx]);
            let x_start = grid_x_start(grid_id);
            let x_end = grid_x_end(grid_id);
            let y_start = grid_y_start(grid_id);
            let y_end = grid_y_end(grid_id);
            let hist_idx = img_idx * MAX_HISTORY_RECORDS + states[img_idx].history_count;
            var marker: GpuShape;
            marker.cx = (f32(x_start) + f32(x_end - 1u)) * 0.5;
            marker.cy = (f32(y_start) + f32(y_end - 1u)) * 0.5;
            marker.hw = f32(x_end - x_start) * 0.5;
            marker.hh = f32(y_end - y_start) * 0.5;
            marker.theta = 0.0;
            marker.shape_type = select(REGION_END_MARKER_SHAPE_TYPE, 4u, states[img_idx].phase == PHASE_GRID_INTERRUPT);
            marker.r = 0u;
            marker.g = 0u;
            marker.b = 0u;
            marker.delta_err = -1.0;
            marker.pad1 = states[img_idx].stage;
            marker.pad2 = local_grid(grid_id);
            history[hist_idx] = marker;
            if (states[img_idx].phase == PHASE_GRID_END) { states[img_idx].closed[grid_id] = 1u; }
            states[img_idx].history_count += 1u;
            if (states[img_idx].history_count >= MAX_HISTORY_RECORDS) {
                states[img_idx].is_done = 1u;
                states[img_idx].termination_reason = TERMINATION_HISTORY_LIMIT;
            } else {
                if (states[img_idx].phase == PHASE_GRID_END) {
                    states[img_idx].closed[grid_id] = 1u;
                }
                select_next_grid(img_idx);
            }
        }
        return;
    }

    if (tid == 0u) { 
        rng_state = schedule_random(img_idx) ^ states[img_idx].search_attempt_count;
        states[img_idx].search_attempt_count += 1u;
        for (var i=0u; i<SURVIVOR_COUNT; i++) { top_errs[i] = 99999999.0; }
        wg_apply_flag = 0u; 
    }
    workgroupBarrier();

    let restricted_to_grid = states[img_idx].phase == PHASE_GRID_DRAWING;
    let grid_id = active_grid_id(states[img_idx]);

    // 🌊 海选
    for (var i = 0u; i < INITIAL_CANDIDATE_COUNT; i++) {
        if (tid == 0u) {
            current_shape = gen_random_shape(restricted_to_grid, grid_id);
        }
        workgroupBarrier(); 
        eval_current_shape(img_offset, tid);

        if (tid == 0u) {
            var e = out_err; var s = current_shape;
            s.r = u32(shared_opt_r); s.g = u32(shared_opt_g); s.b = u32(shared_opt_b); s.delta_err = e;
            for (var j = 0u; j < SURVIVOR_COUNT; j++) {
                if (e < top_errs[j]) {
                    for (var k = SURVIVOR_COUNT - 1u; k > j; k--) { top_errs[k] = top_errs[k-1]; top_shapes[k] = top_shapes[k-1]; }
                    top_errs[j] = e; top_shapes[j] = s;
                    break;
                }
            }
        }
        workgroupBarrier();
    }

    // 🧬 爬山
    var global_best_shape: GpuShape;
    var global_best_err = 99999999.0;

    for (var survivor = 0u; survivor < SURVIVOR_COUNT; survivor++) {
        var local_best_shape = top_shapes[survivor];
        var local_best_err = top_errs[survivor];

        if (tid == 0u) { wg_age = 0u; }
        workgroupBarrier();
        
        var iters = 0u;
        while (iters < MAX_MUTATIONS_PER_SURVIVOR) {
            // wg_age 只由线程 0 更新；显式进行 uniform load，保证所有线程
            // 一起退出循环，不会出现部分线程越过循环内 barrier 的情况。
            let uniform_age = workgroupUniformLoad(&wg_age);
            if (uniform_age >= MUTATION_STAGNATION_LIMIT) { break; }

            if (tid == 0u) {
                current_shape = mutate_shape(local_best_shape, restricted_to_grid, grid_id);
            }
            workgroupBarrier();
            eval_current_shape(img_offset, tid);

            if (tid == 0u) {
                if (out_err < local_best_err) {
                    local_best_err = out_err; 
                    local_best_shape = current_shape;
                    local_best_shape.r = u32(shared_opt_r); local_best_shape.g = u32(shared_opt_g); local_best_shape.b = u32(shared_opt_b);
                    local_best_shape.delta_err = out_err;
                    wg_age = 0u; 
                } else {
                    wg_age += 1u;
                }
            }
            workgroupBarrier(); 
            iters += 1u;
        }

        if (tid == 0u) {
            if (local_best_err < global_best_err) {
                global_best_err = local_best_err;
                global_best_shape = local_best_shape;
            }
        }
        workgroupBarrier();
    } 

    // 🎨 判定结果并记录
    if (tid == 0u) {
        if (global_best_err < 0.0) {
            wg_apply_flag = 1u;
            current_shape = global_best_shape;
            let hist_idx = img_idx * MAX_HISTORY_RECORDS + states[img_idx].history_count;
            global_best_shape.pad1 = states[img_idx].stage;
            global_best_shape.pad2 = local_grid(grid_id);
            history[hist_idx] = global_best_shape;
            states[img_idx].history_count += 1u;
        }
    }
    workgroupBarrier();

    // ==========================================
    // SDF 平滑上色涂抹，并按实际改变的 RGB 像素增量维护全局 SSE。
    // 这与 Geometrize 的 partial difference 相同：无需每一步重扫整张画布。
    // ==========================================
    let apply_flag = workgroupUniformLoad(&wg_apply_flag);
    sum_actual_delta_sse[tid] = 0.0;
    sum_region_delta_sse[tid] = 0.0;
    if (apply_flag == 1u) {
        prepare_current_shape(tid);

        var idx = tid;
        while (idx < shared_box_w * shared_box_h) {
            let x = u32(shared_min_x) + (idx % shared_box_w);
            let y = u32(shared_min_y) + (idx / shared_box_w);
            let px_idx = img_offset + y * CANVAS_WIDTH + x;
            let dx = f32(x) - current_shape.cx;
            let dy = f32(y) - current_shape.cy;

            let coverage = get_coverage(
                current_shape.shape_type,
                current_shape.hw,
                current_shape.hh,
                dx,
                dy,
                shared_cos,
                shared_sin,
                shared_inv_hw2,
                shared_inv_hh2,
            );

            if (coverage > 0.0) {
                let trgb = target_tex[px_idx];
                let tr = f32((trgb >> 16u) & 255u);
                let tg = f32((trgb >> 8u) & 255u);
                let tb = f32(trgb & 255u);
                let crgb = canvas_tex[px_idx];
                let cr = f32((crgb >> 16u) & 255u);
                let cg = f32((crgb >> 8u) & 255u);
                let cb = f32(crgb & 255u);
                
                let eff_alpha = SHAPE_ALPHA * coverage;
                let nr = u32(f32(current_shape.r) * eff_alpha + cr * (1.0 - eff_alpha));
                let ng = u32(f32(current_shape.g) * eff_alpha + cg * (1.0 - eff_alpha));
                let nb = u32(f32(current_shape.b) * eff_alpha + cb * (1.0 - eff_alpha));

                // 必须使用与真正写入画布完全一致的整数颜色，才会得到精确的局部 SSE 差值。
                let before_sse =
                    (tr - cr) * (tr - cr) + (tg - cg) * (tg - cg) + (tb - cb) * (tb - cb);
                let after_sse =
                    (tr - f32(nr)) * (tr - f32(nr)) +
                    (tg - f32(ng)) * (tg - f32(ng)) +
                    (tb - f32(nb)) * (tb - f32(nb));
                sum_actual_delta_sse[tid] += after_sse - before_sse;
                if (restricted_to_grid && is_pixel_in_grid(x, y, grid_id)) {
                    sum_region_delta_sse[tid] += after_sse - before_sse;
                }
                
                canvas_tex[px_idx] = (nr << 16u) | (ng << 8u) | nb;
            }
            idx += WG_SIZE;
        }
    }

    workgroupBarrier();
    for (var s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sum_actual_delta_sse[tid] += sum_actual_delta_sse[tid + s];
            sum_region_delta_sse[tid] += sum_region_delta_sse[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let step_delta_sse = sum_actual_delta_sse[0];
        states[img_idx].current_sse = max(0.0, states[img_idx].current_sse + step_delta_sse);

        if (apply_flag == 1u) {
            states[img_idx].successful_shape_count += 1u;
            if (states[img_idx].phase == PHASE_FULL_CANVAS) {
                states[img_idx].first_phase_shape_count += 1u;
                if (states[img_idx].first_phase_shape_count >= states[img_idx].phase_two_start_count) {
                    states[img_idx].stage = 1u;
                    select_next_grid(img_idx);
                }
            } else if (states[img_idx].phase == PHASE_GRID_DRAWING) {
                states[img_idx].grid_shape_count += 1u;
                states[img_idx].counts[grid_id] += 1u;
                states[img_idx].region_sse = max(
                    0.0,
                    states[img_idx].region_sse + sum_region_delta_sse[0],
                );
            }
        }

        // 普通图形和 marker 共用同一个历史上限；达到时立即停止，不补写记录。
        if (states[img_idx].history_count >= MAX_HISTORY_RECORDS) {
            states[img_idx].is_done = 1u;
            states[img_idx].termination_reason = TERMINATION_HISTORY_LIMIT;
        } else if (states[img_idx].phase == PHASE_GRID_DRAWING &&
            ((states[img_idx].grid_shape_count >= GRID_MIN_SUCCESSFUL_SHAPES &&
              region_similarity(states[img_idx]) >= REGION_SIMILARITY_THRESHOLD) ||
             states[img_idx].grid_shape_count >= grid_limit(states[img_idx].active_grid))) {
            states[img_idx].phase = PHASE_GRID_END;
        } else if (apply_flag == 1u && states[img_idx].phase == PHASE_GRID_DRAWING &&
                   states[img_idx].perturbed == 1u && states[img_idx].perturb[grid_id] == 1u) {
            let probability = schedule_float(img_idx) * 0.10;
            if (schedule_float(img_idx) < probability) { states[img_idx].phase = PHASE_GRID_INTERRUPT; }
        } else if (states[img_idx].search_attempt_count >= MAX_SEARCH_ATTEMPTS) {
            states[img_idx].is_done = 1u;
            states[img_idx].termination_reason = TERMINATION_ATTEMPT_LIMIT;
        }
    }
}
