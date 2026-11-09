//! Real, local Rust inference. The UI worker owns this engine on its thread.

use crate::model::{
    self, is_marker, region_marker, validate_visible, KvCache, OutputScope, Prediction,
    PrimitiveModel, BOS, BOS_RECORD, END_RECORD, INTERRUPT, INTERRUPT_RECORD, MAX_CONTEXT, OFFSETS,
    PAD, PARTIAL, TRUE_END,
};
use crate::region_state::{DrawingRegion, Stage};
pub use crate::region_state::PendingStageRecord;
use anyhow::{bail, Context, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sentencepiece_rs::SentencePieceProcessor;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

fn prepare_optional(operation: impl FnOnce() -> Result<()>) -> Result<()> {
    prepare_optional_value(operation)
}
fn prepare_optional_value<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => {
            let reason = payload.downcast_ref::<String>().cloned()
                .or_else(||payload.downcast_ref::<&str>().map(|s|s.to_string())).unwrap_or_else(||"可选组件初始化失败".into());
            bail!("可选加速组件准备失败，普通执行仍可使用：{reason}")
        }
    }
}

pub(crate) fn uncertain_device_execution(message: &str) -> bool {
    let upper = message.to_ascii_uppercase();
    ["EXECUTION_UNCERTAIN", "CAPTURE_STATE_UNCERTAIN", "DEVICE_ASSERT", "ILLEGAL_ADDRESS", "ILLEGAL MEMORY ACCESS",
        "LAUNCH_FAILED", "CONTEXT_IS_DESTROYED", "DEVICE_LOST", "DEVICE LOST",
        "CUDA_ERROR_ASSERT", "CUDAERRORASSERT", "CUDA_ERROR_UNKNOWN", "CUDAERRORUNKNOWN",
        "HIPERRORUNKNOWN", "HIP_ERROR_UNKNOWN", "DEVICE-SIDE ASSERT", "DEVICE SIDE ASSERT",
        "CUDA_ERROR_MISALIGNED_ADDRESS", "CUDAERRORMISALIGNEDADDRESS",
        "CUDA_ERROR_LAUNCH_TIMEOUT", "CUDAERRORLAUNCHTIMEOUT",
        "CUDA_ERROR_HARDWARE_STACK_ERROR", "CUDAERRORHARDWARESTACKERROR",
        "CUDA_ERROR_INVALID_ADDRESS_SPACE", "CUDAERRORINVALIDADDRESSSPACE",
        "CUDA_ERROR_ILLEGAL_INSTRUCTION", "CUDAERRORILLEGALINSTRUCTION",
        "CUDA_ERROR_INVALID_PC", "CUDAERRORINVALIDPC",
        "CUDA_ERROR_ECC_UNCORRECTABLE", "CUDAERRORECCUNCORRECTABLE",
        "CUDAERRORLAUNCHFAILURE", "CUDAERRORILLEGALADDRESS", "CUDAERRORCONTEXTISDESTROYED",
        "HIPERRORASSERT", "HIPERRORLAUNCHFAILURE", "HIPERRORILLEGALADDRESS",
        "HIPERRORMISALIGNEDADDRESS", "HIPERRORLAUNCHTIMEOUT", "HIPERRORHARDWARESTACKERROR",
        "HIPERRORINVALIDADDRESSSPACE", "HIPERRORILLEGALINSTRUCTION", "HIPERRORINVALIDPC",
        "HIPERRORECCNOTCORRECTABLE", "HIPERRORCONTEXTISDESTROYED",
        "CUBLAS_STATUS_EXECUTION_FAILED", "CUBLAS_STATUS_INTERNAL_ERROR",
        "HIPBLAS_STATUS_EXECUTION_FAILED", "HIPBLAS_STATUS_INTERNAL_ERROR"]
        .iter().any(|word| upper.contains(word))
}

fn known_safe_optional_gpu_failure(message: &str) -> bool {
    if uncertain_device_execution(message) { return false; }
    let upper=message.to_ascii_uppercase();
    let safe_codes=["CUDA_ERROR_OUT_OF_MEMORY", "CUDAERRORMEMORYALLOCATION",
        "HIPERROROUTOFMEMORY", "HIP_ERROR_OUT_OF_MEMORY",
        "CUDA_ERROR_NOT_SUPPORTED", "CUDAERRORNOTSUPPORTED", "HIPERRORNOTSUPPORTED",
        "CUDA_ERROR_NO_BINARY_FOR_GPU", "CUDAERRORNOKERNELIMAGEFORDEVICE", "HIPERRORNOBINARYFORGPU",
        "CUDA_ERROR_INVALID_DEVICE_FUNCTION", "CUDAERRORINVALIDDEVICEFUNCTION", "HIPERRORINVALIDDEVICEFUNCTION",
        "CUDA_ERROR_INVALID_PTX", "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
        "CUBLAS_STATUS_NOT_SUPPORTED", "HIPBLAS_STATUS_NOT_SUPPORTED",
        "NVRTC_ERROR_COMPILATION", "NVRTC_ERROR_INVALID_OPTION", "NVRTC_ERROR_OUT_OF_MEMORY"];
    let mut known_code=false;
    for word in upper.split(|c:char|!c.is_ascii_alphanumeric()&&c!='_') {
        if ["CUDA_ERROR_", "CUDAERROR", "HIP_ERROR_", "HIPERROR", "CUBLAS_STATUS_",
            "HIPBLAS_STATUS_", "NVRTC_ERROR_"].iter().any(|prefix|word.starts_with(prefix)) {
            if !safe_codes.contains(&word) { return false; }
            known_code=true;
        }
    }
    known_code || upper.contains("OUT OF MEMORY")
        || message.contains("当前模型后端不支持此融合解码路径")
        || upper.contains("UNSUPPORTED ATTENTION KV DTYPE")
        || (upper.contains("NVRTC")&&["FAILED TO LOAD", "UNABLE TO LOAD", "UNABLE TO DYNAMICALLY LOAD",
            "COULD NOT LOAD", "CANNOT LOAD", "NOT FOUND", "COMPILER IS MISSING"]
            .iter().any(|marker|upper.contains(marker)))
}

fn safe_int8_preparation_fallback(message: &str) -> bool {
    if uncertain_device_execution(message) { return false; }
    known_safe_optional_gpu_failure(message)
        || message.contains("KV INT8 quantization self-check failed")
        || message.contains("KV INT8 attention self-check failed")
        || message.contains("无法读取当前可选NVRTC支持的计算目标")
        || message.contains("无法读取当前可选NVRTC的计算目标列表")
        || message.contains("当前可选CUDA编译器没有适合该设备的通用PTX目标")
}

fn resolve_generation_plan(
    settings: &crate::acceleration::AccelerationSettings,
    capabilities: &crate::acceleration::BackendCapabilities,
    sampling: crate::acceleration::SamplingPreference,
    failures: &crate::acceleration::SessionFailures,
) -> Result<crate::acceleration::AccelerationPlan> {
    use crate::acceleration::{resolve, AccelerationMode, ComputeBackend, FeatureSet};
    let plan = resolve(settings, capabilities, sampling, failures)?;
    if capabilities.backend == ComputeBackend::Hip && settings.mode == AccelerationMode::Auto
        && capabilities.auto_recommended != FeatureSet::default()
        && plan.features != capabilities.auto_recommended {
        let mut ordinary = capabilities.clone();
        ordinary.auto_recommended = FeatureSet::default();
        return Ok(resolve(settings, &ordinary, sampling, failures)?);
    }
    Ok(plan)
}

#[path = "inference/batch.rs"]
mod batch;
#[path = "inference/context_window.rs"]
mod context_window;
#[path = "inference/redraw_scope.rs"]
mod redraw_scope;
use redraw_scope::{RedrawScope, activate_generation_pending, generation_region_target, request_settings,
    begin_generation_region,guarded_scoped_review_full};
pub use batch::{parallel_seed, ParallelSummary, SamplerBenchmarkCase};
#[path = "inference/sampling.rs"]
mod sampling;
use sampling::renyi_uniform_index;
#[path = "inference/device.rs"]
mod device;
#[path = "inference/device_sampling.rs"]
mod device_sampling;
#[path = "inference/fused_sampling.rs"]
mod fused_sampling;
#[cfg(feature = "rocm")]
#[path = "inference/hip_sampling.rs"]
mod hip_sampling;
#[path = "inference/sampling_workspace.rs"]
mod sampling_workspace;
#[cfg(feature = "aten-hip")]
#[path = "inference/aten_engine.rs"]
pub mod aten_engine;
#[cfg(feature = "rocm")]
#[path = "inference/backend_provider.rs"]
pub mod backend_provider;
#[path = "prompt_validation.rs"]
pub mod prompt_validation;
#[path = "inference/graph.rs"]
mod graph;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub use_cuda: bool,
}

/// Load-time preparation has no dependency on the editor/document types.
#[derive(Clone, Debug)]
pub struct LoadPreparationSettings {
    pub lanes: usize,
    pub cfg_scale: f64,
    pub renyi_alpha: f64,
    pub sampling_override: Option<bool>,
    pub acceleration: crate::acceleration::AccelerationSettings,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LoadPreparationReport {
    /// True for a supported preparation attempt, including compilation-cache hits.
    pub jit_attempted: bool,
    pub ready: bool,
    pub summary: String,
    pub details: Vec<String>,
    /// Input-dependent capture is deliberately not claimed as load-time work.
    pub graph_deferred: bool,
}

fn load_preparation_plan(
    capabilities: &crate::acceleration::BackendCapabilities,
    failures: &crate::acceleration::SessionFailures,
    settings: &LoadPreparationSettings,
    jit_implementation_available: bool,
) -> Result<crate::acceleration::AccelerationPlan> {
    use crate::acceleration::{AccelerationMode, FeatureSet, SamplingPreference};
    if !settings.cfg_scale.is_finite() || settings.cfg_scale < 0.
        || !settings.renyi_alpha.is_finite() || settings.renyi_alpha <= 1.
    {
        bail!("优化准备的并行数、CFG或Rényi参数无效");
    }
    model::checked_batch_rows(settings.lanes, if (settings.cfg_scale - 1.).abs() > 1e-12 { 2 } else { 1 })?;
    let mut capabilities = capabilities.clone();
    if !jit_implementation_available {
        capabilities.available = FeatureSet::default();
    }
    let sampler_scalars_supported = (settings.renyi_alpha as f32).is_finite()
        && (settings.renyi_alpha as f32) > 1.
        && (settings.cfg_scale as f32).is_finite();
    capabilities.available.gpu_sampling &= sampler_scalars_supported;
    capabilities.available.primitive_graph &= sampler_scalars_supported;
    let mut preferences = settings.acceleration.clone();
    if preferences.mode != AccelerationMode::Compatible {
        // Precompile supported optional code even when conservative Auto has
        // not selected it yet. Explicit Disabled choices remain respected;
        // the original execution preferences are restored before returning.
        preferences.mode = AccelerationMode::Optimized;
    }
    let sampling = match settings.sampling_override {
        Some(true) => SamplingPreference::Gpu,
        Some(false) => SamplingPreference::Cpu,
        None => SamplingPreference::Auto,
    };
    Ok(crate::acceleration::resolve(&preferences, &capabilities, sampling, failures)?)
}

#[derive(Clone, Debug)]
pub struct InferenceSettings {
    pub cfg_scale: f64,
    pub renyi_alpha: f64,
    pub seed: u64,
    /// Maximum new, visible primitives for this particular generation request.
    pub max_primitives: usize,
    pub context_overclock: bool,
    /// Finish the current global/coarse stage, then return before the next
    /// stage's marker or prefill. A new request may continue across that boundary.
    pub pause_at_stage_end: bool,
    /// Suspend regional visits once at their halfway point, then choose a
    /// different unfinished cell. Global drawing is unaffected.
    pub half_region_switch: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferencePrimitive {
    pub tokens: [u32; 9],
}

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub prompt: String,
    pub settings: InferenceSettings,
    pub primitives: Vec<InferencePrimitive>,
    /// Model-only control records associated with the selected history node.
    /// A manual canvas modification invalidates this context; the current
    /// visible canvas is then re-prefilled and is never silently truncated.
    pub history_records: Option<Vec<[u32; 9]>>,
    /// User-edited cells need a fresh termination decision. Their previous
    /// permanent ends have been invalidated in the associated raw history.
    pub manual_review_regions: Vec<(usize, usize)>,
    /// Deferred coarse edits are read-only fine-stage conditioning. They are
    /// never counted as generated fine shapes or appended to authoritative raw.
    pub extra_fine_context: Vec<[u32;9]>,
    /// Actual model progress, independent of the stage owned by edited shapes.
    pub stage_floor:Option<usize>,
    pub pending_stage_context:Vec<PendingStageRecord>,
    /// Independent local redraw. Empty preserves the ordinary stage scheduler.
    /// Selected yellow contents are queued and reread on each fresh visit.
    pub redraw_regions: Vec<(usize, usize)>,
}

fn manual_reviews(request: &GenerationRequest) -> Result<HashSet<(usize, usize)>> {
    let mut reviews=HashSet::new();
    for &(side,index) in &request.manual_review_regions {
        if ![4,8].contains(&side) || index >= side*side { bail!("人工复查区域编号无效"); }
        reviews.insert((side,index));
    }
    Ok(reviews)
}

fn remaining_region_records(history:&History,active:&RegionBlock,review:bool)->usize {
    let count=history.visit_counts.get(&(active.side,active.index)).copied().unwrap_or(0);
    active.cap().saturating_sub(count).max(usize::from(review))+1
}
fn guarded_review_full(settings:&InferenceSettings,history:&History,active:&RegionBlock,review:bool)->bool{
    settings.pause_at_stage_end && review && history.visit_counts.get(&(active.side,active.index)).copied().unwrap_or(0)>=active.cap()
}

#[derive(Clone, Debug)]
pub struct GeneratedPrimitive {
    pub primitive: InferencePrimitive,
    pub history_records: Vec<[u32; 9]>,
}

#[derive(Clone, Debug)]
pub struct GenerationSummary {
    pub generated: usize,
    pub stopped: bool,
    pub context_limit: bool,
    pub completed: bool,
    pub paused_after_stage: Option<Stage>,
    pub history_records: Vec<[u32; 9]>,
    pub elapsed_seconds: f64,
}

impl GenerationSummary {
    fn paused_after(mut self, stage: Stage) -> Self {
        self.paused_after_stage = Some(stage);
        self
    }

    fn with_region(self, callback: &mut impl FnMut(DrawingRegion)) -> Result<Self> {
        emit_region(&self.history_records, callback)?;
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct ForwardProbe {
    pub text_ids: Vec<u32>,
    pub attributes: Vec<f32>,
    pub termination: [f32; 2],
}

pub struct NativeEngine {
    model: PrimitiveModel,
    tokenizer: SentencePieceProcessor,
    pub model_description: String,
    gpu_sampling: bool,
    fused_sampling: bool,
    stop_after_device_enqueue: bool,
    sampler_override: Option<bool>,
    context_overclock: bool,
    fine_aux_blocks: Vec<RegionBlock>,
    diagnostic_cache_forecast: Option<usize>,
    diagnostic_full_attribute_vocab: bool,
    device_workspace: std::sync::Mutex<sampling_workspace::SamplingWorkspaceCache>,
    acceleration_settings: crate::acceleration::AccelerationSettings,
    kv_int8_failure: Option<String>,
    capabilities: crate::acceleration::BackendCapabilities,
    #[cfg(feature="cuda")]
    cuda_auto_probe: Option<CudaAutoProbe>,
    provider_preparation_attempted: bool,
    provider_preparation_details: Vec<String>,
    #[cfg(feature = "rocm")]
    hip_fingerprint: Option<crate::acceleration::BackendRuntimeFingerprint>,
    acceleration_failures: crate::acceleration::SessionFailures,
    primitive_graph_enabled: bool,
    primitive_graph: std::sync::Mutex<Option<graph::PrimitiveGraphSession>>,
    acceleration_fault: std::sync::Mutex<Option<String>>,
    acceleration_notices: std::sync::Mutex<Vec<String>>,
    reject_graph_preparation: AtomicBool,
    primitive_graph_captures: std::sync::atomic::AtomicU64,
    primitive_graph_replays: std::sync::atomic::AtomicU64,
    graph_stage_notified: AtomicBool,
    context_window_stats: std::sync::Mutex<ContextWindowStats>,
}

#[derive(Default, serde::Serialize)]
struct ContextWindowStats {
    append_boundary_records: u64,
    reorders: u64,
    /// Visual-token work per logical lane; CFG does not multiply these counters.
    prefilled_visual_tokens: u64,
    reused_prefix_tokens: u64,
}

#[cfg(feature="cuda")]
struct CudaAutoProbe {
    lanes:usize,cfg_bits:u64,alpha_bits:u64,fp32_kv:bool,kv_int8:bool,
    features:crate::acceleration::FeatureSet,report:serde_json::Value,
}
#[cfg(feature="cuda")]
impl CudaAutoProbe {
    fn matches(&self,lanes:usize,cfg:f64,alpha:f64,fp32:bool,kv_int8:bool)->bool {
        self.lanes==lanes && self.cfg_bits==cfg.to_bits() && self.alpha_bits==alpha.to_bits() && self.fp32_kv==fp32 && self.kv_int8==kv_int8
    }
}

#[derive(Clone)]
struct RegionBlock {
    side: usize,
    index: usize,
    records: Vec<[u32; 9]>,
}

impl RegionBlock {
    fn cap(&self) -> usize {
        if self.side == 4 {
            20
        } else {
            10
        }
    }
    fn center(&self) -> (f64, f64) {
        let cell = 512. / self.side as f64;
        (
            ((self.index % self.side) as f64 + 0.5) * cell - 0.5,
            ((self.index / self.side) as f64 + 0.5) * cell - 0.5,
        )
    }
}

struct History {
    fixed: Vec<[u32; 9]>,
    /// Completed visits are retained as bounded context blocks even if a human
    /// later explicitly reopens the same grid cell for a new visit.
    complete: Vec<RegionBlock>,
    /// Unfinished visit fragments ending in INTERRUPT, never counted complete.
    interrupted: Vec<RegionBlock>,
    active: Option<RegionBlock>,
    /// Counts persist across interruptions of a visit, and reset only when an
    /// explicitly marked new visit reopens an effectively closed cell.
    visit_counts: HashMap<(usize, usize), usize>,
    closed: HashSet<(usize, usize)>,
    pending_interrupt_side: Option<usize>,
    highest_side: Option<usize>,
}

fn parse_request_history(records:&[[u32;9]],floor:Option<usize>)->Result<History>{
    if floor.is_some_and(|side|![1,4,8].contains(&side)){bail!("绘制阶段状态无效");}
    let mut history=parse_history(records)?;
    if floor==Some(8) && (history.pending_interrupt_side==Some(4)||history.active.as_ref().is_some_and(|active|active.side==4)){
        bail!("细网格阶段不能恢复低阶段的活动块，请检查阶段上下文");
    }
    history.highest_side=history.highest_side.max(floor.filter(|side|*side>1));
    Ok(history)
}

fn activate_pending_stage(history:&mut History,full:&mut Vec<[u32;9]>,pending:&[PendingStageRecord],
    activated:&mut HashSet<usize>,reviews:&mut HashSet<(usize,usize)>)->Result<bool>{
    if history.active.is_some(){return Ok(false);}
    let Some(side)=next_stage(history)else{return Ok(false)};
    if activated.contains(&side){return Ok(false);}
    let mut groups=std::collections::BTreeMap::<usize,Vec<[u32;9]>>::new();
    for entry in pending{
        if ![4,8].contains(&entry.side)||entry.index>=entry.side*entry.side{bail!("待启用图元的绘制区域无效");}
        validate_visible(&entry.tokens)?;
        if entry.tokens[5]==OFFSETS[5]{bail!("背景不能作为后续阶段图元");}
        if entry.side==side {groups.entry(entry.index).or_default().push(entry.tokens);}
    }
    if groups.is_empty(){return Ok(false);}
    let floor=history.highest_side;
    let mut next=full.clone();
    for (&index,records) in &groups{
        next.push(region_marker(side,index));next.extend_from_slice(records);next.push(INTERRUPT_RECORD);
    }
    let parsed=parse_request_history(&next,floor)?;
    for &index in groups.keys(){reviews.insert((side,index));}
    *full=next;*history=parsed;activated.insert(side);
    Ok(true)
}

/// Arm only the stage that is incomplete when this request begins. Persisted
/// boundary histories therefore continue on the next explicit generation,
/// rather than repeatedly pausing without making progress.
fn stage_pause_target(history: &History, enabled: bool) -> Option<Stage> {
    if !enabled {
        return None;
    }
    if global_incomplete(history) {
        Some(Stage::Global)
    } else if history.active.as_ref().is_some_and(|active| active.side == 4)
        || next_stage(history) == Some(4)
    {
        Some(Stage::Coarse)
    } else {
        None
    }
}

fn global_incomplete(history: &History) -> bool {
    global_ordinary_count(history) < 85
        && history.highest_side.is_none()
        && history.complete.is_empty()
        && history.interrupted.is_empty()
        && history.active.is_none()
}
fn global_ordinary_count(history:&History)->usize {
    history.fixed.iter().skip(1).filter(|row|visible(row)&&row[5]!=OFFSETS[5]).count()
}

fn reached_stage_pause(history: &History, target: Option<Stage>) -> Option<Stage> {
    match target {
        Some(Stage::Global) if global_ordinary_count(history) >= 85 => Some(Stage::Global),
        Some(Stage::Coarse)
            if history.active.is_none()
                && history.pending_interrupt_side.is_none()
                && history.closed.iter().filter(|(side, _)| *side == 4).count() == 16 =>
        {
            Some(Stage::Coarse)
        }
        _ => None,
    }
}

/// A capped final coarse visit can close without another model query. Do this
/// before a per-request primitive limit returns, preserving its real TRUE_END.
fn capped_final_coarse_visit(history: &History, active: &RegionBlock, target: Option<Stage>) -> bool {
    target == Some(Stage::Coarse)
        && active.side == 4
        && !history.closed.contains(&(4, active.index))
        && history.closed.iter().filter(|(side, _)| *side == 4).count() == 15
        && history.visit_counts.get(&(4, active.index)).copied().unwrap_or(0) >= active.cap()
}

#[derive(Clone)]
struct GuidedCache {
    conditional: KvCache,
    unconditional: Option<KvCache>,
    conditional_pending: Prediction,
    unconditional_pending: Option<Prediction>,
    device_pending: Option<model::DevicePrediction>,
    /// Actual visual records represented by the KV, independently of the
    /// complete chronological canvas history after a rolling-window reorder.
    visual_records: Vec<[u32; 9]>,
}

impl NativeEngine {
    #[cfg(feature="cuda")]
    fn known_cuda_graph_candidate(&self)->bool {
        use crate::acceleration::{FeaturePreference,OptimizationFeature};
        !self.model.int8_kv_enabled() && self.capabilities.auto_recommended.primitive_graph
            &&self.acceleration_settings.primitive_graph!=FeaturePreference::Disabled
            &&self.acceleration_settings.fusion!=FeaturePreference::Disabled
            &&self.sampler_override!=Some(false)
            &&!self.acceleration_failures.failures().iter().any(|failure|
                [OptimizationFeature::PrimitiveGraph,OptimizationFeature::Fusion,OptimizationFeature::GpuSampling].contains(&failure.feature))
    }
    pub fn load_configured(config: &EngineConfig, settings: &LoadPreparationSettings,
        progress: impl FnMut(String)) -> Result<Self> {
        #[cfg(feature = "rocm")]
        {
            if config.use_cuda { bail!("此程序是HIP计算构建，无法替换CUDA请求"); }
            return backend_provider::load_rocm_with_preparation(config, settings, progress);
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = settings;
            Self::load(config, progress)
        }
    }

    pub fn load(config: &EngineConfig, mut progress: impl FnMut(String)) -> Result<Self> {
        #[cfg(feature = "rocm")]
        {
            if config.use_cuda { bail!("此程序是HIP计算构建，无法把CUDA请求替换为AMD或CPU"); }
            return backend_provider::load_rocm(config, progress);
        }
        #[cfg(not(feature = "rocm"))]
        {
        progress("加载原生 SentencePiece 分词器".into());
        let tokenizer = SentencePieceProcessor::open(&config.tokenizer_path).map_err(|e| {
            anyhow::anyhow!("无法加载分词器 {}：{e}", config.tokenizer_path.display())
        })?;
        let model = PrimitiveModel::load(&config.model_path, config.use_cuda, &mut progress)?;
        Self::finish_loading(config, tokenizer, model, None, progress)
        }
    }

    /// Provider-validation entry point. The caller supplies an actual device
    /// and verified capabilities; ordinary application loading stays unchanged.
    pub fn load_on_device(
        config: &EngineConfig,
        device: candle_core::Device,
        dtype: candle_core::DType,
        capabilities: crate::acceleration::BackendCapabilities,
        mut progress: impl FnMut(String),
    ) -> Result<Self> {
        let actual_backend = if device.is_cuda() {
            crate::acceleration::ComputeBackend::Cuda
        } else if device.is_cpu() {
            crate::acceleration::ComputeBackend::Cpu
        } else {
            #[cfg(feature = "rocm")]
            if device.is_rocm() {
                crate::acceleration::ComputeBackend::Hip
            } else { bail!("此构建没有实现所选模型计算设备") }
            #[cfg(not(feature = "rocm"))]
            bail!("此构建没有实现所选模型计算设备")
        };
        if capabilities.backend != actual_backend {
            bail!("模型实际计算设备与能力配置不一致；不会切换显卡或退回CPU");
        }
        progress("加载原生 SentencePiece 分词器".into());
        let tokenizer = SentencePieceProcessor::open(&config.tokenizer_path)
            .map_err(|error| anyhow::anyhow!("无法加载分词器 {}：{error}", config.tokenizer_path.display()))?;
        let model = PrimitiveModel::load_on_device(&config.model_path, device, dtype, &mut progress)?;
        Self::finish_loading(config, tokenizer, model, Some(capabilities), progress)
    }

    fn finish_loading(
        config: &EngineConfig,
        tokenizer: SentencePieceProcessor,
        model: PrimitiveModel,
        provider_capabilities: Option<crate::acceleration::BackendCapabilities>,
        mut progress: impl FnMut(String),
    ) -> Result<Self> {
        let ready = config.model_path.parent().map(|p| p.join("READY.json"));
        let metadata = ready
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        let epoch = metadata.as_ref().and_then(|m| m["epoch"].as_u64());
        let step = metadata.as_ref().and_then(|m| m["global_step"].as_u64());
        let arch = &model.architecture;
        let model_description = format!(
            "{} · {}层 / {}宽 / {}头 / FFN{} · {} {}{}",
            epoch.map_or_else(|| "本地权重".into(), |v| format!("Epoch {v}")),
            arch.layers,
            arch.width,
            arch.heads,
            arch.ffn_hidden,
            arch.device,
            arch.dtype,
            step.map_or_else(String::new, |s| format!(" · step {s}"))
        );
        progress(model_description.clone());
        let capabilities = if let Some(capabilities) = provider_capabilities {
            capabilities
        } else if model.device.is_cuda() {
            let mut caps = crate::acceleration::BackendCapabilities::baseline(
                crate::acceleration::GpuVendor::Nvidia, crate::acceleration::ComputeBackend::Cuda,
                format!("cuda:{:?}",model.device.location()),
            );
            caps.available = crate::acceleration::FeatureSet { gpu_sampling: true,
                fusion: model.architecture.head_width==64, split_k: model.architecture.head_width==64,
                primitive_graph: model.architecture.head_width==64 };
            #[cfg(feature="cuda")]
            {
                let tested_device=crate::model::cuda_compat::compiled_kernel_profile()==crate::model::cuda_compat::CudaKernelProfile::BlackwellSm120a && model.device.as_cuda_device().ok()
                    .and_then(|device|device.cuda_stream().context().compute_capability().ok())==Some((12,0));
                if tested_device && model.architecture.layers==35 && model.architecture.width==1536
                    && model.architecture.ffn_hidden==4096 && model.architecture.head_width==64 {
                    // Verified current NVIDIA family. Auto Graph is further
                    // restricted to the long global visit by device_frame.
                    // Split-K is available explicitly, not auto-recommended:
                    // its isolated measurements did not establish a gain.
                    caps.auto_recommended.fusion=true;
                    caps.auto_recommended.primitive_graph=true;
                }
            }
            caps
        } else {
            crate::acceleration::BackendCapabilities::baseline(crate::acceleration::GpuVendor::Cpu,
                crate::acceleration::ComputeBackend::Cpu,"cpu:local")
        };
        let acceleration_failures = crate::acceleration::SessionFailures::new(&capabilities);
        Ok(Self {
            model,
            tokenizer,
            model_description,
            gpu_sampling: false,
            fused_sampling: true,
            stop_after_device_enqueue: false,
            sampler_override: None,
            context_overclock: false,
            fine_aux_blocks: Vec::new(),
            diagnostic_cache_forecast: None,
            diagnostic_full_attribute_vocab: false,
            acceleration_settings: Default::default(),
            kv_int8_failure: None,
            capabilities,
            #[cfg(feature="cuda")]
            cuda_auto_probe:None,
            provider_preparation_attempted: false,
            provider_preparation_details: Vec::new(),
            #[cfg(feature = "rocm")]
            hip_fingerprint: None,
            acceleration_failures,
            primitive_graph_enabled: false,
            primitive_graph: std::sync::Mutex::new(None),
            acceleration_fault: std::sync::Mutex::new(None),
            acceleration_notices: std::sync::Mutex::new(Vec::new()),
            reject_graph_preparation: AtomicBool::new(false),
            primitive_graph_captures: std::sync::atomic::AtomicU64::new(0),
            primitive_graph_replays: std::sync::atomic::AtomicU64::new(0),
            graph_stage_notified: AtomicBool::new(false),
            context_window_stats: std::sync::Mutex::new(ContextWindowStats::default()),
            device_workspace: std::sync::Mutex::new(
                sampling_workspace::SamplingWorkspaceCache::default(),
            ),
        })
    }

    pub fn set_gpu_sampling_override(&mut self, mode: Option<bool>) {
        self.sampler_override = mode;
    }
    pub fn set_acceleration_preferences(&mut self, preferences: crate::acceleration::AccelerationSettings) {
        self.acceleration_settings = preferences;
    }
    fn configure_int8_kv(&mut self, on_status: &mut impl FnMut(String)) -> Result<()> {
        use crate::acceleration::{AccelerationMode,FeaturePreference};
        let requested=self.acceleration_settings.mode!=AccelerationMode::Compatible
            && self.acceleration_settings.kv_int8!=FeaturePreference::Disabled;
        if requested && self.kv_int8_failure.is_some() {
            self.model.prepare_int8_kv(false)?;
            return Ok(());
        }
        match prepare_optional_value(||self.model.prepare_int8_kv(requested)) {
            Ok(status)=>{if requested{on_status(format!("KV INT8：{}",status.reason));}},
            Err(error)=>{
                let text=format!("{error:#}");
                if !safe_int8_preparation_fallback(&text){
                    self.poison_acceleration(text.clone());
                    bail!("KV INT8准备的设备状态不确定，请重新加载模型；已保留画布：{text}");
                }
                self.model.prepare_int8_kv(false)?;
                self.kv_int8_failure=Some(format!("准备未通过，保留同设备浮点KV：{text}"));
                on_status(format!("KV INT8未启用：{}",self.kv_int8_failure.as_deref().unwrap_or_default()));
            }
        }
        Ok(())
    }
    fn int8_kv_diagnostic(&self)->serde_json::Value {
        use crate::acceleration::{AccelerationMode,FeaturePreference};
        let status=self.model.int8_kv_status();
        let compatible=self.acceleration_settings.mode==AccelerationMode::Compatible;
        let requested=match self.acceleration_settings.kv_int8 {FeaturePreference::Auto=>"auto",FeaturePreference::Enabled=>"enabled",FeaturePreference::Disabled=>"disabled"};
        let reason=if compatible{"普通兼容模式使用浮点KV".to_string()}
            else if requested=="disabled"{"用户已关闭KV INT8，使用浮点KV".to_string()}
            else{self.kv_int8_failure.clone().unwrap_or(status.reason.clone())};
        let storage=if status.enabled{"int8_per_token_head"}else if self.model.current_fp32_kv(){"F32"}else{status.format.as_str()};
        serde_json::json!({"requested":requested,"available":status.available&&self.kv_int8_failure.is_none(),"enabled":status.enabled,
            "ready":status.ready,"storage":storage,
            "reason":reason,"scale_dtype":"fp32","quantization":"symmetric_per_token_head64",
            "packed_bytes_per_token_head":status.packed_bytes_per_token_head,
            "scale_bytes_per_token_head":status.scale_bytes_per_token_head,
            "graph_disabled_for_int8":status.enabled})
    }
    pub fn acceleration_capabilities(&self) -> &crate::acceleration::BackendCapabilities { &self.capabilities }
    /// Read-only observations from the actual model provider. This does not
    /// initialize optional compilers or infer a model device from its renderer.
    pub fn diagnostic_info(&self) -> serde_json::Value {
        let mut info=serde_json::json!({
            "provider":if self.model.device.is_cuda(){"native_candle_cuda"}else if self.model.device.is_cpu(){"native_candle_cpu"}else{"native_other"},
            "architecture":self.model.architecture,
            "capabilities":self.capabilities,
            "active_features":self.active_acceleration_features(),
            "kv_int8":self.int8_kv_diagnostic(),
            "context_window":self.context_window_stats.lock().ok().map(|stats|serde_json::to_value(&*stats).unwrap_or_default()),
            "sampling":self.sampler_name(),
            "compiled_cuda_profile":if cfg!(feature="cuda"){Some(crate::model::cuda_compat::compiled_kernel_profile())}else{None},
            "embedded_ptx_target":if cfg!(feature="cuda"){option_env!("PAINTER_EMBEDDED_PTX_TARGET")}else{None},
            "compiled_cuda_toolkit":if cfg!(feature="cuda"){option_env!("PAINTER_CUDA_TOOLKIT_VERSION")}else{None},
            "actual_cuda_ordinal":serde_json::Value::Null,
            "basic_cuda_math_verified":self.model.device.is_cuda(),
            "compute_capability":serde_json::Value::Null,
            "driver_api_version":serde_json::Value::Null,
            "runtime_api_version":serde_json::Value::Null,
        });
        #[cfg(feature="cuda")]
        if let Ok(device)=self.model.device.as_cuda_device() {
            use candle_core::cuda_backend::cudarc;
            let stream=device.cuda_stream();
            let context=stream.context();
            info["actual_cuda_ordinal"]=serde_json::json!(context.ordinal());
            info["prepared_nvrtc_target"]=serde_json::json!(crate::model::cuda_compat::cached_optional_jit_target(device));
            info["cuda_private_auto_probe"]=self.cuda_auto_probe.as_ref().map_or(serde_json::Value::Null,|p|p.report.clone());
            info["device_name"]=serde_json::json!(context.name().ok());
            info["compute_capability"]=serde_json::json!(context.compute_capability().ok());
            info["total_device_memory_bytes"]=serde_json::json!(context.total_mem().ok());
            info["cuda_uuid"]=serde_json::json!(context.uuid().ok().map(|u|u.bytes.iter().map(|b|format!("{:02x}",*b as u8)).collect::<String>()));
            let mut version=0;
            if unsafe{cudarc::driver::sys::cuDriverGetVersion(&mut version)}==cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                info["driver_api_version"]=serde_json::json!(version);
            }
            let mut pci=[0i8;64];
            if unsafe{cudarc::driver::sys::cuDeviceGetPCIBusId(pci.as_mut_ptr(),pci.len() as i32,context.cu_device())}==cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                info["pci_bus_id"]=serde_json::json!(unsafe{std::ffi::CStr::from_ptr(pci.as_ptr())}.to_string_lossy().into_owned());
            }
            // No runtime API is imported here; its actual version is unknown.
            // Report loaded library paths without loading optional NVRTC.
            #[cfg(windows)]
            {
                use std::ffi::c_void;
                #[link(name="kernel32")]
                extern "system" {
                    fn GetModuleHandleW(name:*const u16)->*mut c_void;
                    fn GetModuleFileNameW(module:*mut c_void,path:*mut u16,size:u32)->u32;
                }
                let mut paths=serde_json::Map::new();
                for name in ["nvcuda.dll","cublas64_13.dll","cublasLt64_13.dll","curand64_10.dll","nvrtc64_130_0.dll","nvrtc-builtins64_131.dll"] {
                    let wide=name.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
                    let module=unsafe{GetModuleHandleW(wide.as_ptr())};
                    if module.is_null(){continue;}
                    let mut path=vec![0u16;32768];
                    let length=unsafe{GetModuleFileNameW(module,path.as_mut_ptr(),path.len() as u32)};
                    if length>0 && (length as usize)<path.len(){
                        paths.insert(name.into(),serde_json::json!(String::from_utf16_lossy(&path[..length as usize])));
                    }
                }
                info["loaded_cuda_libraries"]=serde_json::Value::Object(paths);
            }
        }
        info
    }
    pub fn active_acceleration_features(&self) -> crate::acceleration::FeatureSet {
        let (fusion, split_k) = self.model.optional_decode_features();
        crate::acceleration::FeatureSet { fusion, split_k, gpu_sampling: self.gpu_sampling,
            primitive_graph: self.primitive_graph_enabled }
    }
    #[cfg(feature = "rocm")]
    pub fn backend_fingerprint(&self) -> Option<&crate::acceleration::BackendRuntimeFingerprint> {
        self.hip_fingerprint.as_ref()
    }
    /// Compile/validate supported optional kernels on private scratch buffers.
    /// No text, user KV cache, RNG stream or Graph capture is constructed here.
    /// Ordinary compatibility and unsupported providers return an honest skip.
    pub fn prepare_after_load(
        &mut self,
        settings: &LoadPreparationSettings,
        mut on_progress: impl FnMut(String),
    ) -> Result<LoadPreparationReport> {
        use crate::acceleration::{AccelerationMode, OptimizationFeature};
        self.set_acceleration_preferences(settings.acceleration.clone());
        self.configure_int8_kv(&mut on_progress)?;
        let plan = load_preparation_plan(&self.capabilities, &self.acceleration_failures,
            settings, self.model.supports_native_decode_kernels())?;
        let features = plan.features;
        let jit_attempted = self.model.int8_kv_enabled() || self.provider_preparation_attempted
            || features.fusion || features.split_k || features.gpu_sampling;
        let mut details = self.provider_preparation_details.clone();
        details.extend(plan.decisions.iter()
            .map(|decision| format!("{}：{}", decision.feature.label(), decision.reason)));
        self.set_acceleration_preferences(settings.acceleration.clone());
        self.set_gpu_sampling_override(settings.sampling_override);
        if !jit_attempted {
            self.model.prepare_optional_kernels(false, false)?;
            self.gpu_sampling = false;
            self.primitive_graph_enabled = false;
            let reason = if settings.acceleration.mode == AccelerationMode::Compatible {
                "普通兼容模式不需要可选JIT编译，模型已就绪"
            } else {
                "当前执行后端没有需要准备的可选JIT配置，模型已就绪"
            };
            details.push("未执行可选编译；模型设备与用户选择保持不变".into());
            return Ok(LoadPreparationReport { jit_attempted: false, ready: true,
                summary: reason.into(), details, graph_deferred: false });
        }

        on_progress("正在为当前设备准备加速配置，请稍候…".into());
        let mut warm_preferences = settings.acceleration.clone();
        warm_preferences.mode = AccelerationMode::Optimized;
        self.set_acceleration_preferences(warm_preferences);
        // Reuse the same capability resolution and safe optional-failure
        // handling as generation. This only configures cache policy and tests
        // tiny scratch tensors; it does not allocate a full request's KV cache.
        let prepared = self.configure_generation_cache(settings.lanes, settings.cfg_scale,
            settings.renyi_alpha, 1, usize::MAX, &mut |message| {
                details.push(message.clone());
                on_progress(message);
            });
        self.set_acceleration_preferences(settings.acceleration.clone());
        self.set_gpu_sampling_override(settings.sampling_override);
        prepared?;
        // Restore the requested Auto/Compatible/explicit-feature policy without
        // discarding compiled modules or forcing enabled optimizations on Auto.
        self.configure_generation_cache(settings.lanes, settings.cfg_scale,
            settings.renyi_alpha, 1, usize::MAX, &mut |message| {details.push(message.clone());on_progress(message);})?;
        let graph_deferred = self.primitive_graph_enabled;
        let failures = self.acceleration_failures.failures();
        for feature in [OptimizationFeature::Fusion, OptimizationFeature::SplitK,
            OptimizationFeature::GpuSampling] {
            if let Some(failure) = failures.iter().find(|failure| failure.feature == feature) {
                details.push(format!("{}准备失败，已保留同设备普通路径：{}",
                    feature.label(), failure.reason));
            }
        }
        details.push("已准备的编译模块可以命中缓存；不会为了重复加载而强制清除缓存".into());
        if graph_deferred {
            details.push("整图元Graph已启用时，仍需按实际提示词、并行数与上下文判断并捕获；加载时没有执行Graph捕获".into());
        }
        let fallback = failures.iter().any(|failure| matches!(failure.feature,
            OptimizationFeature::Fusion | OptimizationFeature::SplitK | OptimizationFeature::GpuSampling));
        Ok(LoadPreparationReport { jit_attempted, ready: true, graph_deferred,
            summary: if fallback { "加速准备已结束，部分功能使用普通兼容路径" }
                else { "当前设备的加速准备完成，可以开始生成" }.into(), details })
    }
    pub fn gpu_sampling_active(&self) -> bool { self.gpu_sampling }
    pub fn prepare_acceleration_for_diagnostics(&mut self, lanes:usize,cfg:f64,alpha:f64,prefix:usize,max_new:usize)->Result<Vec<String>> {
        let mut messages=Vec::new();
        self.configure_generation_cache(lanes,cfg,alpha,prefix,max_new,&mut |message| messages.push(message))?;
        Ok(messages)
    }
    pub fn diagnostic_reject_graph_preparation(&self, reject:bool) {
        self.reject_graph_preparation.store(reject,Ordering::Relaxed);
    }
    pub(super) fn poison_acceleration(&self, reason: impl Into<String>) {
        if let Ok(mut fault) = self.acceleration_fault.lock() { *fault = Some(reason.into()); }
    }
    pub fn execution_error_message(&self, error: &anyhow::Error) -> String {
        let message = format!("{error:#}");
        if uncertain_device_execution(&message) {
            self.poison_acceleration(message.clone());
            format!("设备执行状态不确定，请重新加载模型；已保留完整画布：{message}")
        } else { message }
    }
    pub(super) fn acceleration_notice(&self, reason: impl Into<String>) {
        if let Ok(mut notices) = self.acceleration_notices.lock() { notices.push(reason.into()); }
    }
    pub(super) fn flush_acceleration_notices(&self, callback: &mut impl FnMut(String)) {
        let notices = self.acceleration_notices.lock().ok().map(|mut values| std::mem::take(&mut *values)).unwrap_or_default();
        for notice in notices { callback(notice); }
    }
    fn prepare_device_sampler(&self) -> Result<()> {
        use candle_core::{Tensor,DType};
        let device=&self.model.device;
        let logits=Tensor::zeros((1,128),DType::F32,device)?;
        let uniform=Tensor::full(0.25f32,(1,1),device)?;
        let valid=if self.fused_sampling {
            let bounds=Tensor::from_vec(vec![0u32,128],(1,2),device)?;
            fused_sampling::sample(&logits,&uniform,&bounds,4.,1.,1,0)?.1
        } else { device_sampling::sample(&logits,&[(0,128)],&uniform,4.,0)?.valid };
        if valid.to_vec2::<u32>()? != vec![vec![1]] { bail!("GPU采样准备验证失败"); }
        Ok(())
    }
    /// Match the unlimited production cache plan while timing fixed work.
    pub fn set_diagnostic_cache_forecast(&mut self, forecast: Option<usize>) {
        self.diagnostic_cache_forecast = forecast;
    }
    pub fn sampler_name(&self) -> &'static str {
        if self.gpu_sampling {
            if self.fused_sampling {
                "gpu-fused-fp32-renyi-v2"
            } else {
                "gpu-tensor-fp32-renyi-v1"
            }
        } else {
            "cpu-f64-renyi-v1"
        }
    }
    pub fn set_fused_gpu_sampling(&mut self, enabled: bool) {
        self.fused_sampling = enabled;
    }
    pub fn diagnostic_stop_after_device_enqueue(&mut self, enabled: bool) {
        self.stop_after_device_enqueue = enabled;
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        prompt_validation::validate_with_processor(&self.tokenizer, text)
            .into_token_ids()
            .map_err(anyhow::Error::msg)
    }

    /// Exposes a real initial/prefix prediction for native CPU/CUDA comparison.
    /// It does not generate, sample, or alter an editor document.
    pub fn forward_probe(
        &self,
        prompt: &str,
        prefix: &[InferencePrimitive],
    ) -> Result<ForwardProbe> {
        let text_ids = self.tokenize(prompt)?;
        let mut records = vec![BOS_RECORD];
        for p in prefix {
            validate_visible(&p.tokens)?;
            records.push(p.tokens);
        }
        let cache = self
            .prefill_scoped(
                &records,
                &text_ids,
                1.,
                &AtomicBool::new(false),
                &mut |_| {},
                OutputScope::All,
            )?
            .context("探测意外停止")?;
        Ok(ForwardProbe {
            text_ids,
            attributes: cache.conditional_pending.attributes,
            termination: cache.conditional_pending.termination,
        })
    }

    /// Independent single-position prefill schedule for checking chunk masks,
    /// positions, control-token embeddings, and cache commits with real weights.
    pub fn forward_probe_sequential(
        &self,
        prompt: &str,
        prefix: &[InferencePrimitive],
    ) -> Result<ForwardProbe> {
        let text_ids = self.tokenize(prompt)?;
        let mut cache = self.model.new_cache();
        for &id in &text_ids {
            self.model.forward_text(id, &mut cache)?;
        }
        let mut pending = self.model.forward_record(&BOS_RECORD, &mut cache, true)?;
        for p in prefix {
            validate_visible(&p.tokens)?;
            pending = self.model.forward_record(&p.tokens, &mut cache, true)?;
        }
        Ok(ForwardProbe {
            text_ids,
            attributes: pending.attributes,
            termination: pending.termination,
        })
    }

    pub fn generate(
        &mut self,
        request: &GenerationRequest,
        stop: &AtomicBool,
        on_primitive: impl FnMut(GeneratedPrimitive),
        on_status: impl FnMut(String),
    ) -> Result<GenerationSummary> {
        self.generate_with_regions(request, stop, on_primitive, on_status, |_| {})
    }

    /// Region notifications carry machine-readable state and never constitute
    /// a visible canvas operation or create an editor history node themselves.
    pub fn generate_with_regions(
        &mut self,
        request: &GenerationRequest,
        stop: &AtomicBool,
        mut on_primitive: impl FnMut(GeneratedPrimitive),
        mut on_status: impl FnMut(String),
        mut on_region: impl FnMut(DrawingRegion),
    ) -> Result<GenerationSummary> {
        let started = Instant::now();
        let effective_settings = request_settings(request);
        let settings = &effective_settings;
        self.context_overclock = settings.context_overclock;
        self.fine_aux_blocks=edited_context_blocks(&request.extra_fine_context)?;
        if !settings.cfg_scale.is_finite() || settings.cfg_scale < 0. {
            bail!("CFG 必须是有限的非负数");
        }
        if !settings.renyi_alpha.is_finite() || settings.renyi_alpha <= 1. {
            bail!("Rényi α 必须大于 1 且为有限数");
        }
        if settings.max_primitives == 0 {
            bail!("本次生成图元上限必须大于零");
        }
        for primitive in &request.primitives {
            validate_visible(&primitive.tokens)?;
        }
        let text = self.tokenize(&request.prompt)?;
        let mut full = if let Some(records) = &request.history_records {
            let visible: Vec<_> = records.iter().filter(|r| visible(r)).copied().collect();
            if visible
                != request
                    .primitives
                    .iter()
                    .map(|p| p.tokens)
                    .collect::<Vec<_>>()
            {
                bail!("该节点的模型上下文与画布不一致，拒绝使用过期缓存");
            }
            records.clone()
        } else {
            let mut records = vec![BOS_RECORD];
            records.extend(request.primitives.iter().map(|p| p.tokens));
            records
        };
        if full.first() != Some(&BOS_RECORD) {
            bail!("模型上下文必须以唯一图元 BOS 开始");
        }
        let mut history = parse_request_history(&full,request.stage_floor)?;
        let mut redraw_scope = RedrawScope::initialize(request, &mut history)?;
        let mut manual_review_regions = manual_reviews(request)?;
        let mut activated_stages=HashSet::new();
        let pause_target = stage_pause_target(&history, settings.pause_at_stage_end);
        self.configure_generation_cache(
            1,
            settings.cfg_scale,
            settings.renyi_alpha,
            full.len(),
            settings.max_primitives,
            &mut on_status,
        )?;
        self.reset_context_window_stats();
        emit_region(&full, &mut on_region)?;
        if redraw_scope.is_some() {
            on_status("局部修改：保留黄色区域已有图形并续画，红色区域须已清空；先粗后细".into());
        }
        if settings.context_overclock {
            on_status("上下文超频已开启：保留全局与粗网格历史，按距离选择细网格；模型总上下文仍为1024".into());
        }
        if history.fixed.len() > 87 {
            on_status(if settings.context_overclock {
                format!("全局前缀含 {} 个既有图元；优先保留可装入窗口的全局图元，画布与原始历史保持完整", history.fixed.len() - 1)
            } else {
                format!("编辑画布含 {} 个既有图元；续画将按区域邻近选择有限上下文，画布与历史保持完整", history.fixed.len() - 1)
            });
        }
        let mut generated = 0;
        let mut rng = StdRng::seed_from_u64(settings.seed);
        let mut live_cache = None;
        let mut fine_aux_seen=false;
        let mut raw_cursor = 0usize;
        let summary = |full: Vec<[u32; 9]>, generated, stopped, completed| GenerationSummary {
            generated,
            stopped,
            context_limit: false,
            completed,
            paused_after_stage: None,
            history_records: full,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        };
        if stop.load(Ordering::Relaxed) {
            return summary(full, generated, true, false).with_region(&mut on_region);
        }

        // The model was trained with one background plus 85 global primitives.
        // Existing, manually supplied primitives remain exactly as supplied.
        if global_incomplete(&history) {
            on_status(format!(
                "准备文字与 {} 个已有图元",
                request.primitives.len()
            ));
            live_cache = self.prefill(
                &history.fixed,
                &text,
                settings.cfg_scale,
                stop,
                &mut on_status,
            )?;
            let Some(cache) = live_cache.as_mut() else {
                return summary(full, generated, true, false).with_region(&mut on_region);
            };
            raw_cursor = full.len();
            while global_incomplete(&history) {
                let background = history.fixed.len() == 1;
                let Some(record) =
                    self.next_primitive(cache, settings, background, None, stop, &mut rng)?
                else {
                    return summary(full, generated, true, false).with_region(&mut on_region);
                };
                full.push(record);
                raw_cursor = full.len();
                self.flush_acceleration_notices(&mut on_status);
                history.fixed.push(record);
                generated += 1;
                on_primitive(GeneratedPrimitive {
                    primitive: InferencePrimitive { tokens: record },
                    history_records: full.clone(),
                });
                emit_region(&full, &mut on_region)?;
                if let Some(stage) = reached_stage_pause(&history, pause_target) {
                    return summary(full, generated, false, false)
                        .paused_after(stage).with_region(&mut on_region);
                }
                if generated >= settings.max_primitives {
                    return summary(full, generated, false, false).with_region(&mut on_region);
                }
            }
        }

        loop {
            if let Some(stage) = reached_stage_pause(&history, pause_target) {
                return summary(full, generated, false, false)
                    .paused_after(stage).with_region(&mut on_region);
            }
            if stop.load(Ordering::Relaxed) {
                return summary(full, generated, true, false).with_region(&mut on_region);
            }
            if activate_generation_pending(&mut history,&mut full,&request.pending_stage_context,&mut activated_stages,&mut manual_review_regions,&mut redraw_scope)?{
                emit_region(&full,&mut on_region)?;
                continue;
            }
            let mut active = if let Some(active) = history.active.take() {
                active
            } else {
                let Some((side,index)) = generation_region_target(&history, redraw_scope.as_ref(), &mut rng)? else {
                    on_status(format!(
                        "当前区域绘制已完成 · 粗区域 {}/16 · 细区域 {}/64",
                        history.closed.iter().filter(|(side, _)| *side == 4).count(),
                        history.closed.iter().filter(|(side, _)| *side == 8).count()
                    ));
                    return summary(full, generated, false, true).with_region(&mut on_region);
                };
                begin_generation_region(&mut history,&mut full,side,index,&request.pending_stage_context,
                    redraw_scope.as_ref(),&mut manual_review_regions)?
            };
            emit_region(&full, &mut on_region)?;
            if let Some(index) = half_region_switch_target(&history, &active,
                settings.half_region_switch && !manual_review_regions.contains(&(active.side,active.index)), &mut rng) {
                let from = active.index;
                let side = active.side;
                history.active = Some(interrupt_and_select_region(&mut history, &mut full, active, index));
                on_status(format!("半程自动换块：{}区域 {} → {}", if side == 4 { "粗" } else { "细" }, from + 1, index + 1));
                emit_region(&full, &mut on_region)?;
                continue;
            }
            // Human edits are never rejected or discarded for exceeding an
            // automatic generation cap. Close that visit before any prefill,
            // then select another region without adding an automatic shape.
            if !manual_review_regions.contains(&(active.side,active.index)) && *history
                .visit_counts
                .get(&(active.side, active.index))
                .unwrap_or(&0)
                >= active.cap()
            {
                full.push(END_RECORD);
                finish_visit(&mut history, active)?;
                emit_region(&full, &mut on_region)?;
                continue;
            }
            if !settings.context_overclock { live_cache = None; raw_cursor = 0; }
            let context = self.region_context_window(live_cache.as_ref(), &full, raw_cursor,
                &history, &active, text.len(), settings.context_overclock, manual_review_regions.contains(&(active.side,active.index)),!fine_aux_seen)?;
            let stage_name = if active.side == 4 { "粗" } else { "细" };
            on_status(format!(
                "{}区域 {}/{} · 已完成 {} 个区域 · 上下文 {}+{}",
                stage_name,
                active.index + 1,
                active.side * active.side,
                history.closed.len(),
                text.len(),
                context.len()
            ));
            live_cache = self.retarget_visual_context(live_cache.take(), &context, &text,
                settings.cfg_scale, stop, &mut on_status)?;
            let Some(cache) = live_cache.as_mut() else {
                return summary(full, generated, true, false).with_region(&mut on_region);
            };
            raw_cursor = full.len();
            if active.side==8 {fine_aux_seen=true;}
            loop {
                let count = *history
                    .visit_counts
                    .get(&(active.side, active.index))
                    .unwrap_or(&0);
                let review=manual_review_regions.contains(&(active.side,active.index));
                if let Some(index) = half_region_switch_target(&history, &active,
                    settings.half_region_switch && (!review || !should_end(cache,settings.cfg_scale)), &mut rng) {
                    let from = active.index;
                    let side = active.side;
                    history.active = Some(interrupt_and_select_region(&mut history, &mut full, active, index));
                    on_status(format!("半程自动换块：{}区域 {} → {}", if side == 4 { "粗" } else { "细" }, from + 1, index + 1));
                    emit_region(&full, &mut on_region)?;
                    break;
                }
                if (!review && count >= active.cap()) || should_end(cache, settings.cfg_scale) {
                    full.push(END_RECORD);
                    manual_review_regions.remove(&(active.side,active.index));
                    finish_visit(&mut history, active)?;
                    emit_region(&full, &mut on_region)?;
                    break;
                }
                if guarded_scoped_review_full(settings,&history,&active,review,redraw_scope.as_ref()){
                    on_status("阶段检查：当前块已达到上下文上限。模型已重新读取该块，但尚未预测结束；请先删除或调整图形后继续。".into());
                    return summary(full,generated,true,false).with_region(&mut on_region);
                }
                let remaining=remaining_region_records(&history,&active,true);
                if review && (cache.conditional.length + remaining > MAX_CONTEXT
                    || (!settings.context_overclock && cache.visual_records.len()+remaining>897)) {
                    history.active=Some(active);
                    break;
                }
                let Some(record) = self.next_primitive(
                    cache,
                    settings,
                    false,
                    Some((active.side, active.index)),
                    stop,
                    &mut rng,
                )?
                else {
                    return summary(full, generated, true, false).with_region(&mut on_region);
                };
                full.push(record);
                raw_cursor = full.len();
                self.flush_acceleration_notices(&mut on_status);
                active.records.push(record);
                *history
                    .visit_counts
                    .entry((active.side, active.index))
                    .or_default() += 1;
                generated += 1;
                on_primitive(GeneratedPrimitive {
                    primitive: InferencePrimitive { tokens: record },
                    history_records: full.clone(),
                });
                emit_region(&full, &mut on_region)?;
                if !manual_review_regions.contains(&(active.side,active.index)) && capped_final_coarse_visit(&history, &active, pause_target) {
                    full.push(END_RECORD);
                    finish_visit(&mut history, active)?;
                    return summary(full, generated, false, false)
                        .paused_after(Stage::Coarse).with_region(&mut on_region);
                }
                if generated >= settings.max_primitives {
                    return summary(full, generated, false, false).with_region(&mut on_region);
                }
            }
        }
    }

    fn prefill(
        &self,
        records: &[[u32; 9]],
        text: &[u32],
        cfg: f64,
        stop: &AtomicBool,
        on_status: &mut impl FnMut(String),
    ) -> Result<Option<GuidedCache>> {
        self.prefill_scoped(
            records,
            text,
            cfg,
            stop,
            on_status,
            OutputScope::NextPrimitive,
        )
    }

    fn configure_generation_cache(
        &mut self,
        lanes: usize,
        cfg: f64,
        alpha: f64,
        prefix_records: usize,
        max_new: usize,
        on_status: &mut impl FnMut(String),
    ) -> Result<()> {
        if let Some(reason) = self.acceleration_fault.lock().map_err(|_|anyhow::anyhow!("加速状态锁不可用"))?.as_ref() {
            bail!("加速执行状态不确定，请重新加载模型；已保留完整画布：{reason}");
        }
        let retired = self.primitive_graph.lock().map_err(|_|anyhow::anyhow!("Graph会话锁不可用"))?.take();
        if let Some(graph::PrimitiveGraphSession::Disabled(reason)) = retired.as_ref() {
            self.acceleration_failures.record(crate::acceleration::OptimizationFeature::PrimitiveGraph,
                crate::acceleration::FailureClass::Preparation,reason.clone());
        }
        drop(retired);
        self.configure_int8_kv(on_status)?;
        self.primitive_graph_captures.store(0,Ordering::Relaxed);
        self.primitive_graph_replays.store(0,Ordering::Relaxed);
        self.graph_stage_notified.store(false,Ordering::Relaxed);
        let mut plan = self.model.configure_cache_for_request(
            lanes,
            (cfg - 1.).abs() > 1e-12,
            prefix_records,
            self.diagnostic_cache_forecast.unwrap_or(max_new),
        )?;
        #[cfg(feature="cuda")]
        if self.model.device.is_cuda() && self.acceleration_settings.mode==crate::acceleration::AccelerationMode::Auto {
            use crate::acceleration::{FeaturePreference,OptimizationFeature};
            let known_graph=self.known_cuda_graph_candidate()&&!self.model.int8_kv_enabled();
            let matched=self.cuda_auto_probe.as_ref().is_some_and(|p|p.matches(lanes,cfg,alpha,self.model.current_fp32_kv(),self.model.int8_kv_enabled()));
            if !known_graph && self.acceleration_settings.primitive_graph!=FeaturePreference::Enabled && !matched {
                on_status("正在为本次硬件和并行配置验证、选择普通GPU加速组合，请稍候…".into());
                let requested=self.acceleration_settings.clone();
                self.acceleration_settings.mode=crate::acceleration::AccelerationMode::Optimized;
                let prepared=self.configure_generation_cache(lanes,cfg,alpha,prefix_records,max_new,on_status);
                self.acceleration_settings=requested;
                prepared?;
                let (fusion,_)=self.model.optional_decode_features();
                let gpu=self.gpu_sampling;
                let fp32=self.model.current_fp32_kv();
                let measured=if fusion||gpu {
                    prepare_optional_value(||self.private_cuda_auto_probe(lanes,cfg,alpha,fusion,gpu))
                }else{Ok(serde_json::json!({"status":"skipped","reason":"本次没有可验证的可选GPU功能","features":crate::acceleration::FeatureSet::default()}))};
                let report=match measured {
                    Ok(report)=>report,
                    Err(error)=>{
                        let text=format!("{error:#}");
                        if uncertain_device_execution(&text){self.poison_acceleration(text.clone());bail!("CUDA私有验证执行状态不确定，请重新加载模型：{text}");}
                        if text.contains("普通实际模型") || text.contains("CUDA私有测速预填充") {
                            return Err(error);
                        }else if known_safe_optional_gpu_failure(&text) {
                            serde_json::json!({"status":"skipped","reason":format!("可选组件不可用或资源不足，跳过私有测速并保留普通GPU路径：{text}"),"features":crate::acceleration::FeatureSet::default()})
                        }else{
                            let reason=format!("EXECUTION_UNCERTAIN: CUDA私有验证发生未分类异常，已停止GPU提交：{text}");
                            self.poison_acceleration(reason.clone());
                            bail!(reason);
                        }
                    }
                };
                let features=serde_json::from_value::<crate::acceleration::FeatureSet>(report["features"].clone()).unwrap_or_default();
                on_status(if features==crate::acceleration::FeatureSet::default(){
                    "本机短对照未确认可选组合有稳定收益，保留普通GPU路径".into()
                }else{"已根据本机短对照选择有收益的GPU采样／融合组合；未为未知设备自动启用Graph".into()});
                self.cuda_auto_probe=Some(CudaAutoProbe{lanes,cfg_bits:cfg.to_bits(),alpha_bits:alpha.to_bits(),fp32_kv:fp32,kv_int8:self.model.int8_kv_enabled(),features,report});
                self.model.device.synchronize()?;
                plan=self.model.configure_cache_for_request(lanes,(cfg-1.).abs()>1e-12,prefix_records,self.diagnostic_cache_forecast.unwrap_or(max_new))?;
            }
        }
        let supported = self.model.supports_native_decode_kernels()
            && (alpha as f32).is_finite()
            && (alpha as f32) > 1.
            && (cfg as f32).is_finite();
        use crate::acceleration::{FailureClass,OptimizationFeature,SamplingPreference};
        self.primitive_graph_enabled = false;
        let mut capabilities = self.capabilities.clone();
        #[cfg(feature="cuda")]
        let private_cuda_recommendation=if self.model.device.is_cuda()&&!self.known_cuda_graph_candidate(){
            self.cuda_auto_probe.as_ref().filter(|p|p.matches(lanes,cfg,alpha,self.model.current_fp32_kv(),self.model.int8_kv_enabled())).map(|p|p.features)
        }else{None};
        #[cfg(feature="cuda")]
        if let Some(features)=private_cuda_recommendation{capabilities.auto_recommended=features;}
        #[cfg(feature = "rocm")]
        if let Some(fingerprint) = self.hip_fingerprint.as_ref() {
            let guidance = if (cfg - 1.).abs() > 1e-12 { 2 } else { 1 };
            capabilities.auto_recommended = backend_provider::workload_recommendations(
                fingerprint, lanes * guidance, guidance);
        }
        capabilities.available.gpu_sampling &= supported;
        capabilities.available.primitive_graph &= supported && !self.model.int8_kv_enabled();
        let sampling = match self.sampler_override { Some(true)=>SamplingPreference::Gpu,Some(false)=>SamplingPreference::Cpu,None=>SamplingPreference::Auto };
        #[cfg(feature="cuda")]
        if self.model.device.is_cuda() && self.acceleration_settings.mode==crate::acceleration::AccelerationMode::Auto
            &&private_cuda_recommendation.is_some() {
            let attempted=crate::acceleration::resolve(&self.acceleration_settings,&capabilities,sampling,&self.acceleration_failures)?;
            if attempted.features!=capabilities.auto_recommended{capabilities.auto_recommended=Default::default();}
        }
        let mut acceleration = resolve_generation_plan(&self.acceleration_settings,&capabilities,sampling,&self.acceleration_failures)?;
        // Preparation uses private throw-away buffers, before any user KV or
        // random generator is touched. Device faults must never be retried.
        for feature in [OptimizationFeature::Fusion,OptimizationFeature::SplitK] {
            let enabled = match feature {OptimizationFeature::Fusion=>acceleration.features.fusion,_=>acceleration.features.split_k};
            if !enabled { continue; }
            let fusion = acceleration.features.fusion;
            let split = acceleration.features.split_k;
            if let Err(error) = prepare_optional(||self.model.prepare_optional_kernels(fusion,split)) {
                let message = format!("{error:#}");
                let fatal = uncertain_device_execution(&message);
                self.acceleration_failures.record(feature,if fatal{FailureClass::ExecutionOrDevice}else{FailureClass::Preparation},message.clone());
                if fatal {
                    self.poison_acceleration(message.clone());
                    bail!("加速设备执行异常，请重新加载模型；已保留完整画布：{message}");
                }
                on_status(format!("{}不可用，切回已验证的普通路径：{message}",feature.label()));
                acceleration = resolve_generation_plan(&self.acceleration_settings,&capabilities,sampling,&self.acceleration_failures)?;
            }
        }
        if acceleration.features.gpu_sampling {
            if let Err(error)=prepare_optional(||self.prepare_device_sampler()) {
                let message=format!("{error:#}");
                let fatal=uncertain_device_execution(&message);
                self.acceleration_failures.record(OptimizationFeature::GpuSampling,if fatal{FailureClass::ExecutionOrDevice}else{FailureClass::Preparation},message.clone());
                if fatal {
                    self.poison_acceleration(message.clone());
                    bail!("加速设备执行异常，请重新加载模型：{message}");
                }
                on_status("GPU采样准备不可用，模型保留GPU并改用普通CPU采样".into());
                acceleration=resolve_generation_plan(&self.acceleration_settings,&capabilities,sampling,&self.acceleration_failures)?;
            }
        }
        prepare_optional(||self.model.prepare_optional_kernels(acceleration.features.fusion,acceleration.features.split_k))?;
        self.gpu_sampling = acceleration.features.gpu_sampling;
        self.primitive_graph_enabled = acceleration.features.primitive_graph;
        #[cfg(feature="cuda")]
        {
            let measured=self.acceleration_settings.mode==crate::acceleration::AccelerationMode::Auto
                &&self.acceleration_settings.fusion!=crate::acceleration::FeaturePreference::Enabled
                &&self.acceleration_settings.split_k!=crate::acceleration::FeaturePreference::Enabled
                &&!self.primitive_graph_enabled
                &&self.cuda_auto_probe.as_ref().is_some_and(|p|p.matches(lanes,cfg,alpha,self.model.current_fp32_kv(),self.model.int8_kv_enabled()));
            self.model.restrict_cuda_auto_attention(measured.then_some(lanes*if (cfg-1.).abs()>1e-12{2}else{1}));
        }
        #[cfg(feature = "rocm")]
        {
            let automatic = self.acceleration_settings.mode == crate::acceleration::AccelerationMode::Auto
                && self.acceleration_settings.fusion != crate::acceleration::FeaturePreference::Enabled
                && self.acceleration_settings.split_k != crate::acceleration::FeaturePreference::Enabled;
            self.model.restrict_hip_auto_attention(automatic.then_some(lanes * if (cfg-1.).abs()>1e-12 {2}else{1}));
        }
        on_status(format!("推理路径：{}",acceleration.label()));
        on_status(format!("采样后端：{}", self.sampler_name()));
        on_status(if self.model.int8_kv_enabled() {
            "历史KV使用INT8与FP32缩放系数；当前图元浮点，整图元Graph关闭".into()
        } else if plan.use_fp32 {
            "长序列任务：已启用快速缓存，会占用更多显存".into()
        } else {
            "本次使用紧凑缓存".into()
        });
        Ok(())
    }

    fn retarget_visual_context(
        &self,
        previous: Option<GuidedCache>,
        records: &[[u32; 9]],
        text: &[u32],
        cfg: f64,
        stop: &AtomicBool,
        on_status: &mut impl FnMut(String),
    ) -> Result<Option<GuidedCache>> {
        let Some(mut cache) = previous else { return self.prefill(records, text, cfg, stop, on_status); };
        if stop.load(Ordering::Relaxed) { return Ok(None); }
        if records.is_empty() || records.len().checked_add(text.len()).is_none_or(|n| n > MAX_CONTEXT) {
            bail!("incremental context is empty or exceeds the model window");
        }
        let guidance = (cfg - 1.).abs() > 1e-12;
        if cache.unconditional.is_some() != guidance
            || cache.conditional.length != text.len() + cache.visual_records.len()
            || cache.unconditional.as_ref().is_some_and(|u| u.length != 2 + cache.visual_records.len()
                || u.absolute_position() != cache.conditional.absolute_position()) {
            bail!("incremental context does not match its conditional/unconditional KV lengths");
        }
        if cache.visual_records == records { return Ok(Some(cache)); }
        let keep = context_window::reusable_visual_prefix(&cache.visual_records, records);
        if let Ok(mut stats) = self.context_window_stats.lock() {
            stats.reused_prefix_tokens = stats.reused_prefix_tokens.saturating_add(keep as u64);
        }
        cache.conditional.validate_truncate_committed(text.len() + keep)?;
        if let Some(u) = &cache.unconditional { u.validate_truncate_committed(2 + keep)?; }
        cache.conditional.truncate_committed(text.len() + keep)?;
        if let Some(u) = &mut cache.unconditional { u.truncate_committed(2 + keep)?; }
        cache.visual_records.truncate(keep);
        cache.device_pending = None;
        let tail = &records[keep..];
        if tail.is_empty() { bail!("a shortened context must recompute a final token to refresh its pending head"); }
        on_status(format!("复用上下文前缀：保留 {keep} 个图元记录，仅计算 {} 个后缀记录", tail.len()));
        // A short END/marker tail can use the fused INT8 single-query path;
        // expanding the historical prefix for a two-token prefill is wasteful.
        let chunk_size = if self.model.int8_kv_enabled() && tail.len() <= 4 { 1 } else { 64 };
        let use_device = self.gpu_sampling;
        let mut c_device = None;
        let mut u_device = None;
        for (index, chunk) in tail.chunks(chunk_size).enumerate() {
            if stop.load(Ordering::Relaxed) { return Ok(None); }
            let last = (index + 1) * chunk_size >= tail.len();
            if use_device && last {
                c_device = Some(self.model.prefill_records_device(chunk, &mut cache.conditional, OutputScope::NextPrimitive)?);
            } else if let Some(prediction) = self.model.prefill_records_for(chunk, &mut cache.conditional,
                last.then_some(OutputScope::NextPrimitive))? {
                cache.conditional_pending = prediction;
            }
            if let Some(u) = &mut cache.unconditional {
                if use_device && last {
                    u_device = Some(self.model.prefill_records_device(chunk, u, OutputScope::NextPrimitive)?);
                } else if let Some(prediction) = self.model.prefill_records_for(chunk, u,
                    last.then_some(OutputScope::NextPrimitive))? {
                    cache.unconditional_pending = Some(prediction);
                }
            }
            cache.visual_records.extend_from_slice(chunk);
            if let Ok(mut stats) = self.context_window_stats.lock() {
                stats.prefilled_visual_tokens = stats.prefilled_visual_tokens.saturating_add(chunk.len() as u64);
            }
        }
        if use_device {
            let c = c_device.context("incremental conditional GPU head absent")?;
            let mut parts = vec![&c];
            if let Some(u) = &u_device { parts.push(u); }
            let pending = model::DevicePrediction::concat_rows(&parts)?;
            let terms = device::read_device_terms(&pending)?;
            cache.install_device(pending, &terms)?;
        }
        Ok(Some(cache))
    }

    fn reset_context_window_stats(&self) {
        if let Ok(mut stats) = self.context_window_stats.lock() { *stats = ContextWindowStats::default(); }
    }

    fn region_context_window(
        &self, previous: Option<&GuidedCache>, raw: &[[u32; 9]], raw_cursor: usize,
        history: &History, active: &RegionBlock, text_len: usize, overclock: bool, review: bool, fine_aux_unseen:bool,
    ) -> Result<Vec<[u32; 9]>> {
        let remaining=remaining_region_records(history,active,review);
        let inject_aux=active.side==8&&fine_aux_unseen&&!self.fine_aux_blocks.is_empty();
        if overclock || review || inject_aux {
            if let Some(records) = fine_context_candidate(
                previous.map(|cache| cache.visual_records.as_slice()),raw,
                if previous.is_some(){raw_cursor}else{0},active,remaining,text_len,overclock,
                &self.fine_aux_blocks,inject_aux)? {
                if previous.is_some() {
                    if let Ok(mut stats) = self.context_window_stats.lock() {
                        stats.append_boundary_records = stats.append_boundary_records.saturating_add((raw.len() - raw_cursor) as u64);
                    }
                }
                return Ok(records);
            }
            if let Ok(mut stats) = self.context_window_stats.lock() { stats.reorders = stats.reorders.saturating_add(1); }
        }
        let mut pool = Vec::new();
        for block in &history.complete { pool.extend(bounded_context_blocks(block)); }
        for fragment in history.interrupted.iter().filter(|block| block.side == active.side) {
            pool.extend(bounded_context_blocks(fragment));
        }
        if active.side==8 {pool.extend(self.fine_aux_blocks.iter().cloned());}
        let mut records = pack_context_for_mode_with_remaining(&history.fixed, &pool, active, text_len, overclock, remaining)?;
        records.extend_from_slice(&active.records);
        Ok(records)
    }

    fn prefill_scoped(
        &self,
        records: &[[u32; 9]],
        text: &[u32],
        cfg: f64,
        stop: &AtomicBool,
        on_status: &mut impl FnMut(String),
        scope: OutputScope,
    ) -> Result<Option<GuidedCache>> {
        if records.len() + text.len() > MAX_CONTEXT || (!self.context_overclock && records.len() > 897) {
            bail!("当前画布/区域前缀占用 {} 个图元记录和 {} 个文本 token，超过模型上下文；不会删除用户图元", records.len(), text.len());
        }
        let mut conditional = self.model.new_cache();
        for chunk in text.chunks(64) {
            if stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            self.model.prefill_text(chunk, &mut conditional)?;
        }
        let mut unconditional = if (cfg - 1.).abs() > 1e-12 {
            // The empty text BOS/EOS occupy the last two conditional text slots.
            // Padding is absent from the KV, but RoPE positions are identical.
            let mut cache = self.model.new_cache_offset(text.len() - 2);
            self.model.prefill_text(&[2, 3], &mut cache)?;
            Some(cache)
        } else {
            None
        };
        let mut conditional_pending = None;
        let mut unconditional_pending = None;
        let use_device = self.gpu_sampling && matches!(scope, OutputScope::NextPrimitive);
        let mut conditional_device = None;
        let mut unconditional_device = None;
        for (i, chunk) in records.chunks(64).enumerate() {
            if stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let output = ((i + 1) * 64 >= records.len()).then_some(scope);
            if use_device && output.is_some() {
                conditional_device = Some(self.model.prefill_records_device(
                    chunk,
                    &mut conditional,
                    scope,
                )?);
            } else {
                conditional_pending =
                    self.model
                        .prefill_records_for(chunk, &mut conditional, output)?;
            }
            if let Some(cache) = &mut unconditional {
                if use_device && output.is_some() {
                    unconditional_device =
                        Some(self.model.prefill_records_device(chunk, cache, scope)?);
                } else {
                    unconditional_pending = self.model.prefill_records_for(chunk, cache, output)?;
                }
            }
            if i > 0 {
                on_status(format!(
                    "重建当前节点的模型上下文：{}/{}",
                    ((i + 1) * 64).min(records.len()),
                    records.len()
                ));
            }
            if let Ok(mut stats) = self.context_window_stats.lock() {
                stats.prefilled_visual_tokens = stats.prefilled_visual_tokens.saturating_add(chunk.len() as u64);
            }
        }
        if use_device {
            let c = conditional_device.context("missing conditional GPU prefill")?;
            let mut parts = vec![&c];
            if let Some(u) = &unconditional_device {
                parts.push(u);
            }
            let prediction = model::DevicePrediction::concat_rows(&parts)?;
            let terms = device::read_device_terms(&prediction)?;
            let mut cache = GuidedCache {
                conditional,
                unconditional,
                conditional_pending: Prediction {
                    attributes: Vec::new(),
                    termination: [0., 0.],
                    attribute_offset: 0,
                },
                unconditional_pending: None,
                device_pending: None,
                visual_records: records.to_vec(),
            };
            cache.install_device(prediction, &terms)?;
            return Ok(Some(cache));
        }
        Ok(Some(GuidedCache {
            conditional,
            unconditional,
            conditional_pending: conditional_pending.context("图元上下文为空")?,
            unconditional_pending,
            device_pending: None,
            visual_records: records.to_vec(),
        }))
    }

    fn next_primitive(
        &self,
        cache: &mut GuidedCache,
        settings: &InferenceSettings,
        background: bool,
        region: Option<(usize, usize)>,
        stop: &AtomicBool,
        rng: &mut StdRng,
    ) -> Result<Option<[u32; 9]>> {
        if self.gpu_sampling {
            let record = self.next_primitive_device(cache, settings, background, region, stop, rng)?;
            if let Some(record) = record { cache.visual_records.push(record); }
            return Ok(record);
        }
        let mut record = PARTIAL;
        for field in 0..9 {
            if stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let candidates = field_candidates(field, background, region)?;
            if candidates.len() == 1 {
                // No model query is needed for a forced shape, but retain the
                // exact old sampler/RNG transition for reproducible seeds.
                let index = renyi_uniform_index(&[0.], settings.renyi_alpha, rng)?;
                record[field] = candidates[index];
                continue;
            }
            let (conditional, unconditional) = if field == 0 {
                (
                    &cache.conditional_pending,
                    cache.unconditional_pending.as_ref(),
                )
            } else {
                // Partial queries use the same uncommitted position. They never
                // advance the history and never become a visible editor node.
                self.query_guided(&record, cache, false, OutputScope::Field(field))?;
                (
                    &cache.conditional_pending,
                    cache.unconditional_pending.as_ref(),
                )
            };
            let logits: Vec<f64> = candidates
                .iter()
                .map(|&id| {
                    let c = conditional.attribute(id)?;
                    Ok(if let Some(u) = unconditional {
                        let u = u.attribute(id)?;
                        u + settings.cfg_scale * (c - u)
                    } else {
                        c
                    })
                })
                .collect::<Result<_>>()?;
            let index = renyi_uniform_index(&logits, settings.renyi_alpha, rng)?;
            record[field] = candidates[index];
        }
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        validate_visible(&record)?;
        self.query_guided(&record, cache, true, OutputScope::NextPrimitive)?;
        Ok(Some(record))
    }

    fn query_guided(
        &self,
        record: &[u32; 9],
        cache: &mut GuidedCache,
        commit: bool,
        scope: OutputScope,
    ) -> Result<()> {
        if let Some(unconditional) = &mut cache.unconditional {
            let mut predictions = self
                .model
                .forward_records_batched_for(
                    &[*record, *record],
                    &mut [&mut cache.conditional, unconditional],
                    commit,
                    scope,
                )?
                .into_iter();
            cache.conditional_pending = predictions.next().context("missing conditional row")?;
            cache.unconditional_pending =
                Some(predictions.next().context("missing unconditional row")?);
        } else {
            cache.conditional_pending =
                self.model
                    .forward_record_for(record, &mut cache.conditional, commit, scope)?;
        }
        if commit { cache.visual_records.push(*record); }
        Ok(())
    }
}

fn visible(row: &[u32; 9]) -> bool {
    row[0] != BOS && row[0] != TRUE_END && row[0] != INTERRUPT && row[0] != PAD && !is_marker(row)
}

fn emit_region(records: &[[u32; 9]], callback: &mut impl FnMut(DrawingRegion)) -> Result<()> {
    callback(DrawingRegion::from_history(records).map_err(anyhow::Error::msg)?);
    Ok(())
}

/// Exact renderer_center_bounds_v1 contract from region_constraints.py:
/// centers are clamped to inclusive pixel [start,end-1], at two bins/pixel.
/// In particular the last half-pixel bin of each grid interval is excluded.
fn center_bin_bounds(region: Option<(usize, usize)>) -> Result<(u32, u32, u32, u32)> {
    let Some((side, index)) = region else {
        return Ok((0, 1022, 0, 1022));
    };
    if ![4, 8].contains(&side) || index >= side * side {
        bail!("区域坐标范围非法：side={side}, index={index}");
    }
    let size = 512 / side;
    let row = index / side;
    let col = index % side;
    Ok((
        (col * size * 2) as u32,
        (((col + 1) * size - 1) * 2) as u32,
        (row * size * 2) as u32,
        (((row + 1) * size - 1) * 2) as u32,
    ))
}

fn field_candidates(
    field: usize,
    background: bool,
    region: Option<(usize, usize)>,
) -> Result<Vec<u32>> {
    if field >= 9 {
        bail!("属性下标超出范围");
    }
    if field == 0 || field == 1 {
        let (x0, x1, y0, y1) = center_bin_bounds(region)?;
        let (lo, hi) = if field == 0 { (x0, x1) } else { (y0, y1) };
        return Ok((OFFSETS[field] + lo..=OFFSETS[field] + hi).collect());
    }
    if field == 5 {
        return Ok(if background {
            vec![OFFSETS[5]]
        } else {
            vec![OFFSETS[5] + 1, OFFSETS[5] + 2]
        });
    }
    Ok((OFFSETS[field]..OFFSETS[field] + model::BINS[field]).collect())
}

fn should_end(cache: &GuidedCache, cfg: f64) -> bool {
    let c = cache.conditional_pending.termination;
    if let Some(u) = &cache.unconditional_pending {
        let u = u.termination;
        (u[1] as f64 + cfg * (c[1] - u[1]) as f64) > (u[0] as f64 + cfg * (c[0] - u[0]) as f64)
    } else {
        c[1] > c[0]
    }
}

fn marker_identity(record: &[u32; 9]) -> Result<(usize, usize)> {
    let width = (record[3].checked_sub(OFFSETS[3]).context("区域宽度非法")? as usize) / 4;
    let side = match width {
        128 => 4,
        64 => 8,
        _ => bail!("无法识别区域宽度 {width}"),
    };
    for index in 0..side * side {
        if &region_marker(side, index) == record {
            return Ok((side, index));
        }
    }
    bail!("区域 marker 不在固定的 4×4 / 8×8 网格上")
}

fn parse_history(records: &[[u32; 9]]) -> Result<History> {
    let fixed_end = records.iter().position(is_marker).unwrap_or(records.len());
    let fixed = records[..fixed_end].to_vec();
    if fixed.is_empty() || fixed[0] != BOS_RECORD {
        bail!("缺少图元 BOS");
    }
    for row in fixed.iter().skip(1) {
        validate_visible(row)?;
    }
    let mut history = History {
        fixed,
        complete: Vec::new(),
        interrupted: Vec::new(),
        active: None,
        visit_counts: HashMap::new(),
        closed: HashSet::new(),
        pending_interrupt_side: None,
        highest_side: None,
    };
    for row in records.iter().skip(fixed_end) {
        if history.pending_interrupt_side.is_some() && !is_marker(row) {
            bail!("interrupt_end 后必须立即跟同阶段 marker_begin");
        }
        if is_marker(row) {
            if history.active.is_some() {
                bail!("区域开始记录嵌套");
            }
            let (side, index) = marker_identity(row)?;
            if let Some(previous_side) = history.pending_interrupt_side {
                if previous_side != side {
                    bail!("interrupt_end 后必须立即跟同阶段 marker_begin");
                }
            }
            if history.highest_side == Some(8) && side == 4 {
                bail!("细区域阶段不能退回粗区域");
            }
            history.pending_interrupt_side = None;
            history.highest_side = Some(side);
            // Explicit human re-entry after true_end opens a new visit. A
            // suspended visit has no effective closed key, so its count stays.
            if history.closed.remove(&(side, index)) {
                history.visit_counts.insert((side, index), 0);
            }
            history.visit_counts.entry((side, index)).or_default();
            history.active = Some(RegionBlock {
                side,
                index,
                records: vec![*row],
            });
        } else if row == &END_RECORD {
            let block = history
                .active
                .take()
                .context("没有活动区域却出现 true_end")?;
            finish_visit(&mut history, block)?;
        } else if row == &INTERRUPT_RECORD {
            let mut block = history
                .active
                .take()
                .context("没有活动区域却出现 interrupt_end")?;
            block.records.push(INTERRUPT_RECORD);
            history.pending_interrupt_side = Some(block.side);
            history.interrupted.push(block);
        } else {
            validate_visible(row)?;
            let block = history.active.as_mut().context("区域外出现图元")?;
            block.records.push(*row);
            *history
                .visit_counts
                .entry((block.side, block.index))
                .or_default() += 1;
        }
    }
    Ok(history)
}

/// Close only the current visit. Interrupted fragments of that visit become a
/// single canonical completed context block; the full raw history is untouched.
fn finish_visit(history: &mut History, active: RegionBlock) -> Result<()> {
    let key = (active.side, active.index);
    let mut records = vec![region_marker(active.side, active.index)];
    let mut remaining = Vec::with_capacity(history.interrupted.len());
    for fragment in history.interrupted.drain(..) {
        if (fragment.side, fragment.index) == key {
            records.extend(fragment.records.into_iter().filter(visible));
        } else {
            remaining.push(fragment);
        }
    }
    history.interrupted = remaining;
    records.extend(active.records.into_iter().filter(visible));
    let count = history.visit_counts.remove(&key).unwrap_or(0);
    if records.len() - 1 != count {
        bail!("区域 visit 累计图元计数与记录不一致");
    }
    records.push(END_RECORD);
    history.complete.push(RegionBlock {
        side: key.0,
        index: key.1,
        records,
    });
    history.closed.insert(key);
    Ok(())
}

fn next_stage(history: &History) -> Option<usize> {
    if let Some(side) = history.pending_interrupt_side {
        return Some(side);
    }
    if history.highest_side == Some(8) {
        return (history.closed.iter().filter(|(side, _)| *side == 8).count() < 64).then_some(8);
    }
    if history.closed.iter().filter(|(side, _)| *side == 4).count() < 16 {
        Some(4)
    } else if history.closed.iter().filter(|(side, _)| *side == 8).count() < 64 {
        Some(8)
    } else {
        None
    }
}

/// Counts are visit-wide, including earlier suspended fragments. A fragment
/// which reached halfway proves this visit has already used its one switch,
/// including after Stop/resume or a saved-project roundtrip.
fn half_region_switch_target(
    history: &History, active: &RegionBlock, enabled: bool, rng: &mut StdRng,
) -> Option<usize> {
    if !enabled { return None; }
    let count = history.visit_counts.get(&(active.side, active.index)).copied().unwrap_or(0);
    let half = active.cap() / 2;
    let suspended_count: usize = history.interrupted.iter()
        .filter(|b| (b.side, b.index) == (active.side, active.index))
        .map(|b| b.records.iter().filter(|r| visible(r)).count()).sum();
    if count < half || count >= active.cap() || suspended_count >= half { return None; }
    let available: Vec<_> = (0..active.side * active.side)
        .filter(|&i| i != active.index && !history.closed.contains(&(active.side, i))).collect();
    if available.is_empty() { return None; }
    Some(available[rng.random_range(0..available.len())])
}

/// Keep the interrupted visit open; its shapes are never removed, duplicated,
/// or reset. Commit both controls together so no observer sees a dangling
/// interruption followed by a marker from a different drawing stage.
fn interrupt_and_select_region(
    history: &mut History, full: &mut Vec<[u32; 9]>, mut active: RegionBlock, index: usize,
) -> RegionBlock {
    let side = active.side;
    let from = active.index;
    active.records.push(INTERRUPT_RECORD);
    history.interrupted.push(active);
    let marker = region_marker(side, index);
    full.extend([INTERRUPT_RECORD, marker]);
    history.pending_interrupt_side = None;
    history.highest_side = Some(side);
    history.visit_counts.entry((side, index)).or_default();
    crate::diagnostics::event("half_region_switch", serde_json::json!({
        "side":side,"from":from,"to":index,
        "visit_count":history.visit_counts.get(&(side,from)).copied().unwrap_or(0)}));
    RegionBlock { side, index, records: vec![marker] }
}

/// Oversized manual visits remain intact in raw history, but their input-only
/// projection uses bounded complete pieces so an entire large visit is not
/// discarded merely because it cannot fit in one context window.
fn bounded_context_blocks(block: &RegionBlock) -> Vec<RegionBlock> {
    let visible_records: Vec<_> = block
        .records
        .iter()
        .filter(|r| visible(r))
        .copied()
        .collect();
    if visible_records.len() <= block.cap() {
        return vec![block.clone()];
    }
    let ending = if block.records.last() == Some(&INTERRUPT_RECORD) {
        INTERRUPT_RECORD
    } else {
        END_RECORD
    };
    visible_records
        .chunks(block.cap())
        .map(|chunk| {
            let mut records = vec![region_marker(block.side, block.index)];
            records.extend_from_slice(chunk);
            records.push(ending);
            RegionBlock {
                side: block.side,
                index: block.index,
                records,
            }
        })
        .collect()
}

/// Project an edited canvas's extra visible primitives into temporary,
/// complete fine-region context blocks. This projection is model input only:
/// it neither alters the full raw history nor marks any generation region done.
/// Each bucket keeps the canvas's chronological order; a bucket exceeding the
/// trained fine-region cap is split into independently bounded input blocks.
fn edited_context_blocks(extra_visible: &[[u32; 9]]) -> Result<Vec<RegionBlock>> {
    let mut buckets = vec![Vec::new(); 64];
    for row in extra_visible {
        validate_visible(row)?;
        let column = ((row[0] - OFFSETS[0]) as usize / 128).min(7);
        let r = ((row[1] - OFFSETS[1]) as usize / 128).min(7);
        buckets[r * 8 + column].push(*row);
    }
    let mut blocks = Vec::new();
    for (index, bucket) in buckets.into_iter().enumerate() {
        for chunk in bucket.chunks(10) {
            let mut records = Vec::with_capacity(chunk.len() + 2);
            records.push(region_marker(8, index));
            records.extend_from_slice(chunk);
            records.push(END_RECORD);
            blocks.push(RegionBlock {
                side: 8,
                index,
                records,
            });
        }
    }
    Ok(blocks)
}

/// Deferred coarse edits are immutable, read-only conditioning. Insert their
/// bounded fine blocks before the active marker only once per lane. The raw
/// cursor and authoritative visit counters never include this input-only data.
fn fine_context_candidate(
    previous:Option<&[[u32;9]]>,raw:&[[u32;9]],raw_cursor:usize,active:&RegionBlock,
    remaining:usize,text_len:usize,overclock:bool,aux:&[RegionBlock],inject_aux:bool,
) -> Result<Option<Vec<[u32;9]>>> {
    let Some(mut candidate)=context_window::reuse_or_repack_plan(previous,raw,raw_cursor,remaining,text_len)? else{return Ok(None)};
    if inject_aux && active.side==8 {
        if !candidate.ends_with(&active.records){bail!("只读细网格信息缺少正确的当前区域后缀");}
        let added=aux.iter().try_fold(0usize,|n,block|n.checked_add(block.records.len()).context("只读细网格信息过大"))?;
        let needed=candidate.len().checked_add(added).and_then(|n|n.checked_add(remaining)).context("细网格上下文容量溢出")?;
        if needed>MAX_CONTEXT.saturating_sub(text_len)||(!overclock&&needed>897){return Ok(None);}
        let split=candidate.len()-active.records.len();
        let tail=candidate.split_off(split);
        for block in aux {candidate.extend_from_slice(&block.records);}
        candidate.extend(tail);
    }
    if !overclock && candidate.len().checked_add(remaining).is_none_or(|needed|needed>897){return Ok(None);}
    Ok(Some(candidate))
}

fn pack_context_for_mode(
    fixed: &[[u32; 9]],
    complete: &[RegionBlock],
    target: &RegionBlock,
    text_length: usize,
    overclock: bool,
) -> Result<Vec<[u32; 9]>> {
    let count=target.records.iter().filter(|r|visible(r)).count();
    pack_context_for_mode_with_remaining(fixed,complete,target,text_length,overclock,target.cap().saturating_sub(count)+1)
}

fn pack_context_for_mode_with_remaining(
    fixed:&[[u32;9]], complete:&[RegionBlock], target:&RegionBlock,
    text_length:usize, overclock:bool, remaining:usize,
) -> Result<Vec<[u32;9]>> {
    if overclock {
        pack_overclock_context_reserved(fixed, complete, target, text_length, remaining)
    } else {
        pack_context_reserved(fixed, complete, target, text_length, remaining)
    }
}

/// Input-only rolling projection. The text and visual BOS remain fixed; all
/// available global prefix records take priority over complete coarse visits, then fine visits
/// ordered nearest first. The active visit is appended by the caller, and its
/// entire remaining generation footprint is reserved before selecting history.
/// We never extend RoPE beyond 1024 or alter full chronological canvas/history.
fn pack_overclock_context(
    fixed: &[[u32; 9]],
    complete: &[RegionBlock],
    target: &RegionBlock,
    text_length: usize,
) -> Result<Vec<[u32; 9]>> {
    let count=target.records.iter().filter(|r|visible(r)).count();
    pack_overclock_context_reserved(fixed,complete,target,text_length,target.cap().saturating_sub(count)+1)
}
fn pack_overclock_context_reserved(
    fixed:&[[u32;9]], complete:&[RegionBlock], target:&RegionBlock, text_length:usize, remaining:usize,
) -> Result<Vec<[u32;9]>> {
    let budget = MAX_CONTEXT.checked_sub(text_length).context("文字已经占满模型上下文")?;
    if fixed.first() != Some(&BOS_RECORD) {
        bail!("上下文超频缺少规范图元BOS");
    }
    let active_reserve = target.records.len().checked_add(remaining).context("区域上下文容量溢出")?;
    let fine_footprint = 10 + 2; // marker + at most ten primitives + true_end.
    let reserve = active_reserve.max(fine_footprint);
    let Some(prefix_budget) = budget.checked_sub(reserve) else {
        bail!("当前完整区域无法放入1024上下文；原画布与历史已保留");
    };
    if prefix_budget < 1 {
        bail!("上下文超频无法同时保留图元BOS及当前区域");
    }
    // Even an oversized user-authored global prefix stays the highest priority.
    // Select only its leading whole records if necessary; do not relabel the
    // remainder as fine data or allow lower-priority grids to displace it.
    let fixed = &fixed[..fixed.len().min(prefix_budget)];
    let history_budget = prefix_budget - fixed.len();

    let center = target.center();
    let distance = |block: &RegionBlock| {
        let point = block.center();
        (point.0 - center.0).powi(2) + (point.1 - center.1).powi(2)
    };
    let mut coarse = Vec::new();
    let mut fine = Vec::new();
    for block in complete {
        match block.side {
            4 => coarse.push(block),
            8 => fine.push(block),
            _ => bail!("上下文超频发现非法网格尺寸"),
        }
    }
    // Stable sorting preserves chronology among repeated visits at one cell.
    fine.sort_by(|a, b| distance(a).total_cmp(&distance(b)).then(a.index.cmp(&b.index)));
    let mut selected = Vec::new();
    let mut used = 0;
    for block in coarse.into_iter().chain(fine) {
        if history_budget - used < fine_footprint {
            break;
        }
        // Preserve priority rather than skipping an important block to make
        // room for a less important one. Whole context blocks are indivisible.
        if used + block.records.len() > history_budget {
            break;
        }
        used += block.records.len();
        selected.push(block);
    }
    let mut context = fixed.to_vec();
    // Preserve the control-token contract: an interrupt is followed immediately
    // by a same-stage begin. Selected interrupted fragments therefore follow
    // closed blocks, directly before the caller appends its active begin.
    for block in selected.iter().filter(|block| block.records.last() != Some(&INTERRUPT_RECORD)) {
        context.extend_from_slice(&block.records);
    }
    for block in selected.iter().filter(|block| block.records.last() == Some(&INTERRUPT_RECORD)) {
        if block.side != target.side {
            bail!("不能将不同阶段的中断片段接到当前区域");
        }
        context.extend_from_slice(&block.records);
    }
    if context.len() + reserve > budget {
        bail!("上下文超频打包超过1024上限");
    }
    Ok(context)
}

/// Complete-region packing: prioritize distance while retaining enough space
/// for the target region. A suffix subset-sum table avoids a greedy size gap.
/// The normal, unedited prefix follows its original path without projection.
fn pack_context(
    fixed: &[[u32; 9]],
    complete: &[RegionBlock],
    target: &RegionBlock,
    text_length: usize,
) -> Result<Vec<[u32; 9]>> {
    let count=target.records.iter().filter(|r|visible(r)).count();
    pack_context_reserved(fixed,complete,target,text_length,target.cap().saturating_sub(count)+1)
}
fn pack_context_reserved(
    fixed:&[[u32;9]], complete:&[RegionBlock], target:&RegionBlock, text_length:usize, remaining:usize,
) -> Result<Vec<[u32;9]>> {
    // A manually edited state can have arbitrarily more visible primitives
    // than a single model window. Preserve all of them in `full/history.fixed`,
    // but only retain BOS + the first 86 visible primitives unconditionally in
    // this model input. The other primitives compete as spatial context blocks.
    let projected = if fixed.len() > 87 {
        edited_context_blocks(&fixed[87..])?
    } else {
        Vec::new()
    };
    let fixed = if fixed.len() > 87 {
        &fixed[..87]
    } else {
        fixed
    };
    let budget = 897usize.min(MAX_CONTEXT.saturating_sub(text_length));
    let one = target.records.len().checked_add(remaining).context("区域上下文容量溢出")?;
    let two = one + target.cap() + 2;
    if fixed.len() + one > budget {
        bail!(
            "模型固定保留前缀有 {} 个记录，剩余上下文不足以放入一个完整区域；原画布已保留",
            fixed.len()
        );
    }
    let capacity = budget - fixed.len();
    let upper = capacity - one;
    let mut lower = (capacity + 1).saturating_sub(two);
    let center = target.center();
    let distance = |r: &RegionBlock| {
        let p = r.center();
        (p.0 - center.0).powi(2) + (p.1 - center.1).powi(2)
    };
    let mut ordered: Vec<_> = complete.iter().chain(projected.iter()).collect();
    ordered.sort_by(|a, b| {
        distance(a)
            .total_cmp(&distance(b))
            .then(a.side.cmp(&b.side))
            .then(a.index.cmp(&b.index))
    });
    let mut suffix = vec![vec![false; upper + 1]; ordered.len() + 1];
    suffix[ordered.len()][0] = true;
    for i in (0..ordered.len()).rev() {
        let n = ordered[i].records.len();
        for used in 0..=upper {
            suffix[i][used] = suffix[i + 1][used] || (used >= n && suffix[i + 1][used - n]);
        }
    }
    let mut chosen_upper = upper;
    if !(lower..=upper).any(|n| suffix[0][n]) {
        chosen_upper = (0..=upper).rev().find(|&n| suffix[0][n]).unwrap_or(0);
        lower = chosen_upper;
    }
    let reachable = |bits: &[bool], lo: isize, hi: isize| -> bool {
        if hi < 0 || hi < lo {
            return false;
        }
        (lo.max(0) as usize..=(hi as usize).min(upper)).any(|n| bits[n])
    };
    let mut selected = Vec::new();
    let mut used = 0;
    for (i, region) in ordered.iter().enumerate() {
        if used >= lower && used <= chosen_upper {
            break;
        }
        let n = region.records.len();
        if reachable(
            &suffix[i + 1],
            lower as isize - used as isize - n as isize,
            chosen_upper as isize - used as isize - n as isize,
        ) {
            selected.push(*region);
            used += n;
        }
    }
    if used < lower || used > chosen_upper {
        bail!("区域上下文打包失败");
    }
    let mut context = fixed.to_vec();
    // Suspended fragments must be followed immediately by a same-stage begin.
    // Keep selected closed blocks first, then the selected same-stage suspended
    // chain, and finally the caller's active marker. Ordinary, non-interrupted
    // contexts retain their original distance order unchanged.
    for block in selected
        .iter()
        .filter(|b| b.records.last() != Some(&INTERRUPT_RECORD))
    {
        context.extend_from_slice(&block.records);
    }
    for block in selected
        .iter()
        .filter(|b| b.records.last() == Some(&INTERRUPT_RECORD))
    {
        if block.side != target.side {
            bail!("不能把异阶段中断片段接到当前区域");
        }
        context.extend_from_slice(&block.records);
    }
    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_window_engine(terminate: bool) -> NativeEngine {
        use crate::acceleration::{BackendCapabilities, ComputeBackend, GpuVendor, SessionFailures};
        let mut model = model::small_test_model();
        model.force_test_termination(terminate).unwrap();
        let capabilities = BackendCapabilities::baseline(GpuVendor::Cpu, ComputeBackend::Cpu, "cpu:window-regression");
        let failures = SessionFailures::new(&capabilities);
        NativeEngine {
            model, tokenizer: SentencePieceProcessor::from_serialized_model(include_bytes!("../assets/caption_spm_1500.model")).unwrap(),
            model_description: "tiny CPU context-window regression".into(), gpu_sampling: false, fused_sampling: false,
            stop_after_device_enqueue: false, sampler_override: Some(false), context_overclock: true,fine_aux_blocks:Vec::new(),
            diagnostic_cache_forecast: None, diagnostic_full_attribute_vocab: false,
            acceleration_settings: Default::default(), kv_int8_failure: None, capabilities,
            #[cfg(feature="cuda")]
            cuda_auto_probe: None,
            provider_preparation_attempted: false, provider_preparation_details: Vec::new(),
            #[cfg(feature="rocm")]
            hip_fingerprint: None,
            acceleration_failures: failures, primitive_graph_enabled: false,
            primitive_graph: std::sync::Mutex::new(None), acceleration_fault: std::sync::Mutex::new(None),
            acceleration_notices: std::sync::Mutex::new(Vec::new()), reject_graph_preparation: AtomicBool::new(false),
            primitive_graph_captures: std::sync::atomic::AtomicU64::new(0),
            primitive_graph_replays: std::sync::atomic::AtomicU64::new(0), graph_stage_notified: AtomicBool::new(false),
            context_window_stats: std::sync::Mutex::new(ContextWindowStats::default()),
            device_workspace: std::sync::Mutex::new(sampling_workspace::SamplingWorkspaceCache::default()),
        }
    }

    fn window_global_prefix(count: usize) -> Vec<[u32; 9]> {
        let ordinary = [510,1500,2049,2500,4500,6415,6700,6800,7000];
        let background = [0,1024,2048,2318,4366,6414,6700,6800,7000];
        let mut full = vec![BOS_RECORD, background];
        full.extend(std::iter::repeat(ordinary).take(count - 2)); full
    }

    fn window_request(full: Vec<[u32; 9]>, cfg: f64, maximum: usize) -> GenerationRequest {
        GenerationRequest { prompt: "金发蓝眼".into(), settings: InferenceSettings {
            cfg_scale: cfg, renyi_alpha: 4., seed: 20261011, max_primitives: maximum,
            context_overclock: true, half_region_switch: false, pause_at_stage_end: false },
            primitives: full.iter().filter(|row| visible(row)).map(|row|InferencePrimitive { tokens:*row }).collect(),
            manual_review_regions: Vec::new(),extra_fine_context:Vec::new(),stage_floor:None,pending_stage_context:Vec::new(),redraw_regions:Vec::new(), history_records: Some(full) }
    }

    #[test]
    fn tiny_cpu_global_redraw_finishes_global_before_selected_coarse_and_fine(){
        for lanes in [1,2]{for global_count in [84,85]{
            let initial=window_global_prefix(global_count+2);
            let mut request=window_request(initial.clone(),if lanes==1{1.}else{2.},usize::MAX);
            request.stage_floor=Some(1);request.redraw_regions=vec![(8,9),(1,0),(4,3)];
            request.settings.pause_at_stage_end=true;request.settings.half_region_switch=true;
            let mut emitted=Vec::new();
            let summaries=if lanes==1{
                vec![tiny_window_engine(true).generate_with_regions(&request,&AtomicBool::new(false),
                    |event|emitted.push((0,event)),|_|{},|_|{}).unwrap()]
            }else{
                tiny_window_engine(true).generate_parallel(&request,lanes,&AtomicBool::new(false),
                    |lane,event|emitted.push((lane,event)),|_|{},|_,_|{},|_,_|{}).unwrap().lane_summaries
            };
            assert_eq!(emitted.len(),lanes*(85-global_count));
            for (_,event)in emitted{
                assert!(!event.history_records.iter().any(is_marker));
                assert_ne!(event.primitive.tokens[5],OFFSETS[5],"retained background is never regenerated");
            }
            for summary in summaries{
                assert!(summary.completed);assert!(!summary.stopped);assert_eq!(summary.paused_after_stage,None);
                assert!(summary.history_records.starts_with(&initial));
                assert_eq!(summary.history_records.iter().filter_map(|r|marker_identity(r).ok()).collect::<Vec<_>>(),[(4,3),(8,9)]);
                let history=parse_history(&summary.history_records).unwrap();assert_eq!(history.fixed.len(),87);
                assert_eq!(history.closed,HashSet::from([(4,3),(8,9)]));
            }
            assert!(request.settings.pause_at_stage_end&&request.settings.half_region_switch);
        }}
    }

    #[test]
    fn tiny_cpu_blank_global_redraw_creates_background_then_stops_at_a_complete_primitive(){
        for lanes in [1,2]{
            let mut request=window_request(vec![BOS_RECORD],1.,1);request.stage_floor=Some(1);
            request.redraw_regions=vec![(1,0)];let mut emitted=Vec::new();
            let result=tiny_window_engine(false).generate_parallel(&request,lanes,&AtomicBool::new(false),
                |lane,event|emitted.push((lane,event)),|_|{},|_,_|{},|_,_|{}).unwrap();
            assert_eq!(emitted.len(),lanes);
            for (_,event)in emitted{assert_eq!(event.primitive.tokens[5],OFFSETS[5]);assert_eq!(event.history_records.len(),2);}
            for summary in result.lane_summaries{assert!(!summary.completed);assert_eq!(summary.history_records.len(),2);}
        }
    }

    #[test]
    fn tiny_cpu_redraw_single_and_cfg_batch_require_real_end_at_the_region_cap() {
        // The real two-layer CPU model and normal tokenizer/sampler execute;
        // only its termination head is pinned false to exercise hard quotas.
        for lanes in [1,2] {
            let mut request=window_request(vec![BOS_RECORD],if lanes==1{1.}else{2.},usize::MAX);
            request.stage_floor=Some(4);request.redraw_regions=vec![(8,55),(4,7)];
            request.settings.pause_at_stage_end=true;request.settings.half_region_switch=true;
            let mut a=ordinary();a[0]=32;let mut b=ordinary();b[0]=40;let mut c=ordinary();c[0]=48;
            request.pending_stage_context=vec![PendingStageRecord{side:8,index:9,tokens:a},
                PendingStageRecord{side:8,index:1,tokens:b},PendingStageRecord{side:8,index:9,tokens:c}];
            let mut engine=tiny_window_engine(false);let stop=AtomicBool::new(false);
            let mut callbacks=Vec::<(usize,GeneratedPrimitive)>::new();
            let summaries=if lanes==1 {
                vec![engine.generate_with_regions(&request,&stop,|event|callbacks.push((0,event)),|_|{},|_|{}).unwrap()]
            } else {
                let result=engine.generate_parallel(&request,lanes,&stop,|lane,event|callbacks.push((lane,event)),|_|{},|_,_|{},|_,_|{}).unwrap();
                assert!(result.batched_decode_calls>0);assert_eq!(result.maximum_tensor_batch,4,"two real CFG lanes must batch four rows");
                result.lane_summaries
            };
            assert!(request.settings.pause_at_stage_end&&request.settings.half_region_switch,"request preferences must stay unchanged");
            assert_eq!(callbacks.len(),lanes*20);
            for (_,event) in &callbacks {
                let parsed=parse_history(&event.history_records).unwrap();let active=parsed.active.unwrap();
                assert_eq!((active.side,active.index),(4,7),"a false END cannot advance to the next region");
                let (x0,x1,y0,y1)=center_bin_bounds(Some((active.side,active.index))).unwrap();let row=event.primitive.tokens;
                assert!((x0..=x1).contains(&row[0]));assert!((y0..=y1).contains(&(row[1]-OFFSETS[1])));
                assert_ne!(row[5],OFFSETS[5],"BOS-only redraw must not synthesize background");
                assert_eq!(parsed.fixed,vec![BOS_RECORD],"no global geometry may be generated");
            }
            for summary in summaries {
                assert!(!summary.completed&&summary.stopped);assert_eq!(summary.paused_after_stage,None);assert_eq!(summary.generated,20);
                let parsed=parse_history(&summary.history_records).unwrap();let active=parsed.active.unwrap();
                assert_eq!((active.side,active.index),(4,7));assert_eq!(active.records.len(),21);
                assert!(!parsed.closed.contains(&(4,7)));assert!(!summary.history_records.contains(&END_RECORD));
            }
        }
    }

    #[test]
    fn tiny_cpu_redraw_real_termination_closes_only_selected_empty_regions_without_global_output() {
        for (lanes,fine_only) in [(1,false),(2,false),(1,true),(2,true)] {
            let mut request=window_request(vec![BOS_RECORD],1.,usize::MAX);
            request.stage_floor=Some(if fine_only{8}else{4});
            request.redraw_regions=if fine_only{vec![(8,42)]}else{vec![(8,42),(4,9),(4,2)]};
            if fine_only{request.pending_stage_context=vec![PendingStageRecord{side:8,index:1,tokens:ordinary()}];}
            request.settings.pause_at_stage_end=true;request.settings.half_region_switch=true;
            let mut engine=tiny_window_engine(true);let mut emitted=0;
            let result=engine.generate_parallel(&request,lanes,&AtomicBool::new(false),|_,_|emitted+=1,|_|{},|_,_|{},|_,_|{}).unwrap();
            assert_eq!(emitted,0);assert_eq!(result.generated,0);
            for summary in result.lane_summaries {
                assert!(summary.completed);assert_eq!(summary.paused_after_stage,None);
                let expected=if fine_only{vec![BOS_RECORD,region_marker(8,1),ordinary(),END_RECORD,region_marker(8,42),END_RECORD]}
                    else{vec![BOS_RECORD,region_marker(4,2),END_RECORD,region_marker(4,9),END_RECORD,region_marker(8,42),END_RECORD]};
                assert_eq!(summary.history_records,expected,"Fine-only request floor must not suppress pending Fine conditioning");
                assert_eq!(parse_history(&summary.history_records).unwrap().fixed,vec![BOS_RECORD]);
            }
        }
    }
    #[test]
    fn tiny_cpu_imported_paused_coarse_closes_before_deferred_fine_is_read(){
        use crate::document::{Document,Primitive,PrimitiveStageAssignment,ShapeKind,StageTier};
        use crate::{history_context::RawHistoryStore,redraw::RedrawSelection};
        let owned=|stage:StageTier,cell:usize,order:u64|{
            let side=stage.side();let mut p=Primitive::new(ShapeKind::Rectangle,(cell%side)as f32*512./side as f32+20.,
                (cell/side)as f32*512./side as f32+20.,8.,10.,[60,100,140]);
            p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:order});p
        };
        let mut bg=Primitive::background([240;3]);bg.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});
        let mut source=vec![bg];source.extend((0..85).map(|n|owned(StageTier::Global,0,n+1)));
        let mut raw=vec![BOS_RECORD];raw.extend(source.iter().map(|p|p.to_tokens().unwrap().map(u32::from)));
        raw.push(region_marker(4,2));for n in 0..20{let p=owned(StageTier::Coarse,2,100+n);raw.push(p.to_tokens().unwrap().map(u32::from));source.push(p);}
        raw.push(END_RECORD);let coarse=owned(StageTier::Coarse,3,130);
        raw.extend([region_marker(4,3),coarse.to_tokens().unwrap().map(u32::from),INTERRUPT_RECORD]);source.push(coarse);
        for n in 0..10{source.push(owned(StageTier::Fine,63,140+n));}source.push(owned(StageTier::Fine,9,160));
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let imported=crate::reference_import::import_reference(&mut doc,&mut store,0,&source,&raw).unwrap();
        let selection=RedrawSelection::from_reference_regions(&imported.resume_regions).unwrap();
        let plan=crate::redraw_context::prepare_redraw(&doc,&store,doc.current,&selection).unwrap();
        assert_eq!(plan.stage_floor,Some(4));assert!(plan.pending_stage_context.iter().any(|p|p.side==8));
        for lanes in [1,2]{
            let mut request=window_request(plan.records.clone(),if lanes==1{1.}else{2.},usize::MAX);
            request.redraw_regions=plan.regions.clone();request.pending_stage_context=plan.pending_stage_context.clone();request.stage_floor=plan.stage_floor;
            let mut emitted=0;let result=tiny_window_engine(true).generate_parallel(&request,lanes,&AtomicBool::new(false),|_,_|emitted+=1,|_|{},|_,_|{},|_,_|{}).unwrap();
            assert_eq!(emitted,0);
            for summary in result.lane_summaries{
                assert!(summary.completed);let mut active=None;let mut closed_coarse=HashSet::new();
                for row in &summary.history_records{
                    if is_marker(row){let key=marker_identity(row).unwrap();
                        if key.0==8{assert_eq!(closed_coarse.len(),16,"Fine reference conditioning must wait for all real coarse completions");}
                        active=Some(key);
                    }else if row==&END_RECORD{if let Some((4,index))=active.take(){closed_coarse.insert(index);}}
                    else if row==&INTERRUPT_RECORD{active=None;}
                }
                let mut actual=summary.history_records.iter().filter(|row|visible(row)).copied().collect::<Vec<_>>();actual.sort_unstable();
                let mut expected=source.iter().map(|p|p.to_tokens().unwrap().map(u32::from)).collect::<Vec<_>>();expected.sort_unstable();
                assert_eq!(actual,expected,"every existing reference shape must be read once, without deleting or duplicating future fine content");
                let parsed=parse_history(&summary.history_records).unwrap();assert_eq!(parsed.closed.len(),80);
            }
        }
        assert_eq!(doc.primitives.len(),source.len());assert_eq!(store.stage_floor(&doc,doc.current).unwrap(),Some(4));
    }

    #[test]
    fn tiny_cpu_yellow_contents_are_preserved_and_only_a_fresh_model_end_can_close(){
        for side in [4,8]{for lanes in [1,2]{for model_end in [false,true]{
            let cap=if side==4{20}else{10};let mut request=window_request(vec![BOS_RECORD],if lanes==1{1.}else{2.},usize::MAX);
            request.stage_floor=Some(side);request.redraw_regions=vec![(side,3)];
            request.pending_stage_context=std::iter::repeat_n(PendingStageRecord{side,index:3,tokens:ordinary()},cap).collect();
            let original=request.pending_stage_context.clone();let mut emitted=0;
            let result=tiny_window_engine(model_end).generate_parallel(&request,lanes,&AtomicBool::new(false),|_,_|emitted+=1,|_|{},|_,_|{},|_,_|{}).unwrap();
            assert_eq!(emitted,0,"existing yellow contents at the quota cannot be overwritten or exceeded");
            for summary in result.lane_summaries{
                assert_eq!(summary.completed,model_end);assert_eq!(summary.stopped,!model_end);
                let parsed=parse_history(&summary.history_records).unwrap();assert_eq!(parsed.closed.contains(&(side,3)),model_end);
                assert_eq!(summary.history_records.iter().filter(|r|visible(r)).count(),cap);
                assert_eq!(summary.history_records.last()==Some(&END_RECORD),model_end);
            }
            assert_eq!(request.pending_stage_context.len(),original.len());
            assert!(request.pending_stage_context.iter().zip(original).all(|(a,b)|a.tokens==b.tokens&&a.side==b.side&&a.index==b.index));
        }}}
    }

    #[test]
    fn tiny_cpu_redraw_occupied_targets_fail_before_callbacks_and_stop_keeps_completed_shape_only() {
        let occupied=vec![BOS_RECORD,region_marker(4,3),ordinary(),END_RECORD];
        for lanes in [1,2] {
            let mut request=window_request(occupied.clone(),1.,usize::MAX);request.stage_floor=Some(4);request.redraw_regions=vec![(4,3)];
            let mut emitted=0;let error=tiny_window_engine(false).generate_parallel(&request,lanes,&AtomicBool::new(false),|_,_|emitted+=1,|_|{},|_,_|{},|_,_|{}).unwrap_err();
            assert!(error.to_string().contains("旧的结束标记"));assert_eq!(emitted,0);assert_eq!(request.history_records,Some(occupied.clone()));
        }
        let mut request=window_request(vec![BOS_RECORD],1.,usize::MAX);request.stage_floor=Some(4);request.redraw_regions=vec![(4,3),(8,63)];
        let stop=AtomicBool::new(false);let mut emitted=0;
        let result=tiny_window_engine(false).generate(&request,&stop,|_|{emitted+=1;stop.store(true,Ordering::Relaxed);},|_|{}).unwrap();
        assert!(result.stopped&&!result.completed);assert_eq!(emitted,1);assert_eq!(result.generated,1);
        assert_eq!(result.history_records.len(),3);assert_eq!(result.history_records[1],region_marker(4,3));
        validate_visible(&result.history_records[2]).unwrap();
        assert!(!result.history_records.iter().any(|row|is_marker(row)&&marker_identity(row).unwrap().0==8));
    }

    #[test]
    fn half_switch_is_once_per_visit_excludes_self_and_resumes_without_reset() {
        for side in [4, 8] {
            let cap = if side == 4 {20} else {10};
            let mut full = window_global_prefix(87);
            if side == 8 { for i in 0..16 {full.extend([region_marker(4,i),END_RECORD]);} }
            full.push(region_marker(side,0));
            full.extend(std::iter::repeat([510,1500,2049,2500,4500,6415,6700,6800,7000]).take(cap/2));
            let before = full.clone();
            let mut history = parse_history(&full).unwrap();
            let active = history.active.take().unwrap();
            assert!(half_region_switch_target(&history,&active,false,&mut StdRng::seed_from_u64(4)).is_none());
            let mut rng = StdRng::seed_from_u64(4);
            let other = half_region_switch_target(&history,&active,true,&mut rng).unwrap();
            assert_ne!(other,0);
            history.active = Some(interrupt_and_select_region(&mut history,&mut full,active,other));
            assert_eq!(&full[..before.len()],before);
            assert_eq!(&full[before.len()..],&[INTERRUPT_RECORD,region_marker(side,other)]);
            assert_eq!(DrawingRegion::from_history(&full).unwrap().completed_fine,0);
            // A genuine model end in the other cell does not close cell zero.
            let other_active=history.active.take().unwrap();
            full.push(END_RECORD);finish_visit(&mut history,other_active).unwrap();
            full.push(region_marker(side,0));
            let resumed=parse_history(&full).unwrap();
            assert_eq!(resumed.visit_counts[&(side,0)],cap/2);
            assert!(!resumed.closed.contains(&(side,0)));
            assert!(half_region_switch_target(&resumed,resumed.active.as_ref().unwrap(),true,&mut rng).is_none());
            let mut resumed=resumed;
            let mut active=resumed.active.take().unwrap();
            for _ in cap/2..cap {
                let row=[510,1500,2049,2500,4500,6415,6700,6800,7000];
                full.push(row);active.records.push(row);*resumed.visit_counts.get_mut(&(side,0)).unwrap()+=1;
                assert!(half_region_switch_target(&resumed,&active,true,&mut rng).is_none());
            }
            full.push(END_RECORD);finish_visit(&mut resumed,active).unwrap();
            assert!(resumed.closed.contains(&(side,0)));
            assert_eq!(resumed.complete.last().unwrap().records.iter().filter(|r|visible(r)).count(),cap);
            assert!(parse_history(&full).unwrap().closed.contains(&(side,0)));
        }
    }

    #[test]
    fn half_switch_waits_for_half_and_keeps_last_unfinished_region() {
        let mut full=window_global_prefix(87);
        for i in 1..16 {full.extend([region_marker(4,i),END_RECORD]);}
        full.push(region_marker(4,0));
        full.extend(std::iter::repeat([510,1500,2049,2500,4500,6415,6700,6800,7000]).take(10));
        let history=parse_history(&full).unwrap();
        assert!(half_region_switch_target(&history,history.active.as_ref().unwrap(),true,&mut StdRng::seed_from_u64(9)).is_none());
        let mut full=window_global_prefix(87);full.push(region_marker(4,0));
        full.extend(std::iter::repeat([510,1500,2049,2500,4500,6415,6700,6800,7000]).take(9));
        let history=parse_history(&full).unwrap();
        assert!(half_region_switch_target(&history,history.active.as_ref().unwrap(),true,&mut StdRng::seed_from_u64(9)).is_none());
    }

    #[test]
    fn readonly_deferred_coarse_data_enters_fine_input_once_without_marking_canvas_cells_done(){
        let mut raw=window_global_prefix(87);
        for index in 0..16 {raw.extend([region_marker(4,index),END_RECORD]);}
        let marker=region_marker(8,7);raw.push(marker);
        let active=RegionBlock{side:8,index:7,records:vec![marker]};
        let extra=vec![[510,1500,2049,2500,4500,6415,6700,6800,7000];23];
        let aux=edited_context_blocks(&extra).unwrap();let original=raw.clone();
        let input=fine_context_candidate(None,&raw,0,&active,11,2,true,&aux,true).unwrap().unwrap();
        assert_eq!(raw,original);assert_eq!(input.last(),Some(&marker));
        assert_eq!(input.iter().filter(|row|visible(row)).count(),86+23);
        assert!(parse_history(&raw).unwrap().closed.iter().all(|(side,_)|*side!=8));
        let cursor=raw.len();raw.push(extra[0]);raw.push(END_RECORD);let marker2=region_marker(8,5);raw.push(marker2);
        let active2=RegionBlock{side:8,index:5,records:vec![marker2]};
        let continued=fine_context_candidate(Some(&input),&raw,cursor,&active2,11,2,true,&aux,false).unwrap().unwrap();
        assert_eq!(continued.len(),input.len()+3);assert_eq!(&continued[..input.len()],&input);
        assert!(fine_context_candidate(None,&original,0,&active,11,990,true,&aux,true).unwrap().is_none());
    }

    #[test]
    fn future_owned_shapes_activate_only_at_their_stage_with_real_counts_and_without_new_model_outputs(){
        let mut full=window_global_prefix(87);let mut history=parse_request_history(&full,None).unwrap();
        let ordinary=[510,1500,2049,2500,4500,6415,6700,6800,7000];
        let pending=vec![PendingStageRecord{side:4,index:2,tokens:ordinary},PendingStageRecord{side:8,index:3,tokens:ordinary}];
        let mut activated=HashSet::new();let mut reviews=HashSet::new();
        assert!(activate_pending_stage(&mut history,&mut full,&pending,&mut activated,&mut reviews).unwrap());
        assert!(reviews.contains(&(4,2)));assert!(!reviews.contains(&(8,3)));assert_eq!(history.visit_counts[&(4,2)],1);
        assert!(!full.contains(&region_marker(8,3)));let after_coarse=full.clone();
        assert!(!activate_pending_stage(&mut history,&mut full,&pending,&mut activated,&mut reviews).unwrap());assert_eq!(full,after_coarse);
        for index in 0..16 {full.extend([region_marker(4,index),END_RECORD]);}
        history=parse_request_history(&full,Some(4)).unwrap();
        assert!(activate_pending_stage(&mut history,&mut full,&pending,&mut activated,&mut reviews).unwrap());
        assert!(reviews.contains(&(8,3)));assert_eq!(history.visit_counts[&(8,3)],1);
        assert_eq!(history.pending_interrupt_side,Some(8));
    }

    #[test]
    fn stage_floor_never_restarts_global_and_no_background_still_has_only_eighty_five_global_shapes(){
        let row=[510,1500,2049,2500,4500,6415,6700,6800,7000];
        let raw=[vec![BOS_RECORD],vec![row;84]].concat();
        assert!(global_incomplete(&parse_request_history(&raw,None).unwrap()));
        for side in [4,8]{let history=parse_request_history(&raw,Some(side)).unwrap();assert!(!global_incomplete(&history));assert_eq!(next_stage(&history),Some(side));}
        let full=[vec![BOS_RECORD],vec![row;85]].concat();let history=parse_history(&full).unwrap();
        assert!(!global_incomplete(&history));assert_eq!(reached_stage_pause(&history,Some(Stage::Global)),Some(Stage::Global));
        assert_eq!(DrawingRegion::from_history(&full).unwrap().progress.global_shape_count,85);
    }

    #[test]
    fn manual_over_cap_region_is_re_read_and_continues_until_model_predicts_end() {
        let mut full=window_global_prefix(87);full.push(region_marker(4,0));
        full.extend(std::iter::repeat([510,1500,2049,2500,4500,6415,6700,6800,7000]).take(21));
        let mut request=window_request(full.clone(),1.,1);
        request.manual_review_regions=vec![(4,0)];
        let mut engine=tiny_window_engine(false);
        let result=engine.generate(&request,&AtomicBool::new(false),|_|{},|_|{}).unwrap();
        assert_eq!(result.generated,1);
        assert!(result.history_records.starts_with(&full));
        let parsed=parse_history(&result.history_records).unwrap();
        assert_eq!(parsed.active.as_ref().unwrap().index,0);
        assert_eq!(parsed.visit_counts[&(4,0)],22);
        assert!(!parsed.closed.contains(&(4,0)));
        // The same over-cap input may end, but only after a real fresh prefill
        // and the learned termination decision, never by its old shape count.
        let mut engine=tiny_window_engine(true);
        let result=engine.generate(&request,&AtomicBool::new(false),|_|{},|_|{}).unwrap();
        assert_eq!(result.generated,0);
        assert_eq!(result.history_records[full.len()],END_RECORD);
        assert!(engine.diagnostic_info()["context_window"]["prefilled_visual_tokens"].as_u64().unwrap()>0);
    }

    #[test]
    fn parallel_manual_over_cap_reviews_keep_independent_model_decisions() {
        let mut full=window_global_prefix(87);full.push(region_marker(8,1));
        full.extend(std::iter::repeat([510,1500,2049,2500,4500,6415,6700,6800,7000]).take(12));
        let mut request=window_request(full.clone(),2.,1);
        request.manual_review_regions=vec![(8,1)];
        let mut engine=tiny_window_engine(false);
        let result=engine.generate_parallel(&request,2,&AtomicBool::new(false),|_,_|{},|_|{},|_,_|{},|_,_|{}).unwrap();
        assert_eq!(result.generated,2);
        for lane in result.lane_summaries {
            assert_eq!(lane.generated,1);
            assert!(lane.history_records.starts_with(&full));
            let parsed=parse_history(&lane.history_records).unwrap();
            assert_eq!(parsed.visit_counts[&(8,1)],13);
            assert!(!parsed.closed.contains(&(8,1)));
        }
    }

    #[test]
    fn incremental_context_append_reorder_and_strict_prefix_match_fresh_cfg_prefill() {
        let engine = tiny_window_engine(false);
        let stop = AtomicBool::new(false);
        let text = [2,17,19,3];
        let mut initial = window_global_prefix(4);
        initial.extend([region_marker(4,0), [510,1500,2049,2500,4500,6415,6700,6800,7000], END_RECORD]);
        let original = engine.prefill(&initial, &text, 2., &stop, &mut |_|{}).unwrap().unwrap();
        let mut appended = initial.clone(); appended.push(region_marker(4,1));
        let mut reordered = appended.clone(); reordered.swap(2,3);
        reordered[3][0] = 700;
        for records in [appended, reordered, initial[..4].to_vec()] {
            let keep = context_window::reusable_visual_prefix(&initial, &records);
            engine.reset_context_window_stats();
            let incremental = engine.retarget_visual_context(Some(original.clone()), &records, &text, 2., &stop, &mut |_|{}).unwrap().unwrap();
            let counters = engine.diagnostic_info()["context_window"].clone();
            let fresh = engine.prefill(&records, &text, 2., &stop, &mut |_|{}).unwrap().unwrap();
            assert_eq!(incremental.visual_records, records);
            assert_eq!(incremental.conditional.length, text.len()+records.len());
            assert_eq!(incremental.unconditional.as_ref().unwrap().length, 2+records.len());
            assert_eq!(incremental.conditional.absolute_position(), incremental.unconditional.as_ref().unwrap().absolute_position());
            for (a,b) in [(&incremental.conditional_pending,&fresh.conditional_pending),
                (incremental.unconditional_pending.as_ref().unwrap(),fresh.unconditional_pending.as_ref().unwrap())] {
                assert!(a.attributes.iter().zip(&b.attributes).all(|(a,b)|(a-b).abs()<1e-4));
                assert!(a.termination.iter().zip(&b.termination).all(|(a,b)|(a-b).abs()<1e-4));
            }
            assert_eq!(counters["prefilled_visual_tokens"], (records.len()-keep) as u64);
            assert_eq!(counters["reused_prefix_tokens"], keep as u64);
            assert_eq!(original.visual_records, initial);
            assert_eq!(original.conditional.length, text.len()+initial.len());
        }
    }

    #[test]
    fn global_to_coarse_boundary_keeps_live_kv_and_appends_only_the_marker() {
        let mut engine = tiny_window_engine(false);
        let request = window_request(window_global_prefix(86), 1., 2);
        let result = engine.generate(&request, &AtomicBool::new(false), |_|{}, |_|{}).unwrap();
        assert_eq!(result.generated, 2);
        let stats = engine.diagnostic_info()["context_window"].clone();
        assert_eq!(stats["reorders"], 0);
        assert_eq!(stats["append_boundary_records"], 1);
        assert_eq!(stats["prefilled_visual_tokens"], 87);
        assert_eq!(stats["reused_prefix_tokens"], 87);
        assert!(parse_history(&result.history_records).unwrap().active.is_some());
    }

    #[test]
    fn parallel_empty_fine_regions_append_controls_without_rebuilding_or_reordering() {
        let mut engine = tiny_window_engine(true);
        let mut full = window_global_prefix(87);
        for cell in 0..16 { full.extend([region_marker(4,cell),END_RECORD]); }
        full.push(region_marker(8,0));
        let initial = full.len();
        let request = window_request(full, 2., usize::MAX);
        let result = engine.generate_parallel(&request, 2, &AtomicBool::new(false), |_,_|{}, |_|{}, |_,_|{}, |_,_|{}).unwrap();
        assert_eq!(result.generated, 0);
        assert_eq!(result.actual_prefill_calls, 1);
        assert_eq!(result.prefill_reuse_hits, 1);
        assert!(result.lane_summaries.iter().all(|summary|summary.completed
            &&parse_history(&summary.history_records).unwrap().closed.len()==80));
        let stats = engine.diagnostic_info()["context_window"].clone();
        assert_eq!(stats["reorders"], 0);
        assert_eq!(stats["append_boundary_records"], 2*63*2);
        assert_eq!(stats["prefilled_visual_tokens"], (initial+2*63*2) as u64);
    }
    #[test]
    fn hip_auto_does_not_run_an_unmeasured_partial_bundle_after_preference_or_failure() {
        use crate::acceleration::*;
        let mut caps = BackendCapabilities::baseline(GpuVendor::Amd, ComputeBackend::Hip, "hip:exact");
        caps.available = FeatureSet { fusion: true, gpu_sampling: true, ..Default::default() };
        caps.auto_recommended = caps.available;
        let mut failures = SessionFailures::new(&caps);
        let mut settings = AccelerationSettings::default();
        assert_eq!(resolve_generation_plan(&settings, &caps, SamplingPreference::Auto, &failures).unwrap().features, caps.available);
        settings.fusion = FeaturePreference::Disabled;
        assert_eq!(resolve_generation_plan(&settings, &caps, SamplingPreference::Auto, &failures).unwrap().features, FeatureSet::default());
        settings.fusion = FeaturePreference::Auto;
        failures.record(OptimizationFeature::GpuSampling, FailureClass::Preparation, "optional compiler failed");
        assert_eq!(resolve_generation_plan(&settings, &caps, SamplingPreference::Auto, &failures).unwrap().features, FeatureSet::default());
        settings.fusion = FeaturePreference::Enabled;
        assert!(resolve_generation_plan(&settings, &caps, SamplingPreference::Auto, &failures).unwrap().features.fusion);
    }

    #[test]
    fn device_execution_faults_stop_instead_of_replaying_partial_primitives() {
        assert!(uncertain_device_execution("HIP_EXECUTION_UNCERTAIN: kernel launch returned 719"));
        assert!(uncertain_device_execution("copy completion: hipErrorIllegalAddress"));
        assert!(uncertain_device_execution("CUDA_ERROR_ILLEGAL_ADDRESS"));
        for fatal in ["CUDA_ERROR_ASSERT", "CUDA_ERROR_MISALIGNED_ADDRESS", "CUDA_ERROR_LAUNCH_TIMEOUT",
            "CUDA_ERROR_HARDWARE_STACK_ERROR", "CUDA_ERROR_INVALID_ADDRESS_SPACE",
            "cudaErrorIllegalInstruction", "cudaErrorLaunchFailure", "hipErrorLaunchTimeOut",
            "CUDA error: device-side assert triggered", "CUDA_ERROR_UNKNOWN", "hipErrorUnknown", "CUBLAS_STATUS_EXECUTION_FAILED",
            "CUBLAS_STATUS_INTERNAL_ERROR", "HIPBLAS_STATUS_EXECUTION_FAILED"] {
            assert!(uncertain_device_execution(fatal), "must stop after {fatal}");
        }
        assert!(!uncertain_device_execution("HIP_PREPARATION: compiler is missing"));
        assert!(!uncertain_device_execution("HIP_PREPARATION: source compilation failed"));
        assert!(!uncertain_device_execution("out of memory before kernel launch"));
        assert!(!uncertain_device_execution("host assertion failed: invalid configuration"));
        assert!(!uncertain_device_execution("CUDA_ERROR_INVALID_VALUE before kernel launch"));
        assert!(!uncertain_device_execution("CUBLAS_STATUS_NOT_SUPPORTED"));
    }

    #[test]
    fn optional_gpu_trial_only_falls_back_for_known_safe_unavailability() {
        for safe in ["CUDA_ERROR_OUT_OF_MEMORY", "out of memory before allocation",
            "NVRTC_ERROR_COMPILATION", "CUDA_ERROR_NO_BINARY_FOR_GPU", "CUBLAS_STATUS_NOT_SUPPORTED",
            "Unable to dynamically load the nvrtc shared library"] {
            assert!(known_safe_optional_gpu_failure(safe), "safe candidate failure: {safe}");
        }
        for unknown in ["CUDA_ERROR_UNKNOWN", "hipErrorUnknown", "CUDA_ERROR_FUTURE_DEVICE_FAULT",
            "unrecognized kernel result", "host assertion failed", "NVRTC_ERROR_INTERNAL_ERROR",
            "CUDA_ERROR_NOT_SUPPORTED after CUDA_ERROR_FUTURE_DEVICE_FAULT",
            "out of memory after CUDA_ERROR_ASSERT", "NVRTC_ERROR_COMPILATION after CUDA_ERROR_UNKNOWN"] {
            assert!(!known_safe_optional_gpu_failure(unknown), "must stop for unclassified failure: {unknown}");
        }
    }

    #[test]
    fn int8_optional_compiler_skip_preserves_floating_but_device_faults_stop() {
        for reason in ["无法读取当前可选NVRTC支持的计算目标；保留普通GPU路径",
            "无法读取当前可选NVRTC的计算目标列表；保留普通GPU路径",
            "当前可选CUDA编译器没有适合该设备的通用PTX目标；保留普通GPU路径",
            "KV INT8 attention self-check failed", "NVRTC_ERROR_COMPILATION"] {
            assert!(safe_int8_preparation_fallback(reason));
            assert!(!safe_int8_preparation_fallback(&format!("{reason}: CUDA_ERROR_UNKNOWN")));
        }
        assert!(!safe_int8_preparation_fallback("unclassified launch status"));
    }

    fn preparation_settings() -> LoadPreparationSettings {
        LoadPreparationSettings { lanes: 8, cfg_scale: 1., renyi_alpha: 4.,
            sampling_override: None, acceleration: Default::default() }
    }

    fn preparation_capabilities() -> crate::acceleration::BackendCapabilities {
        use crate::acceleration::*;
        let mut capabilities = BackendCapabilities::baseline(GpuVendor::Nvidia,
            ComputeBackend::Cuda, "test:capability-driven-preparation");
        capabilities.available = FeatureSet { fusion: true, split_k: true,
            gpu_sampling: true, primitive_graph: true };
        capabilities
    }

    #[test]
    fn load_preparation_warms_supported_code_without_mutating_conservative_auto_preferences() {
        let settings = preparation_settings();
        let capabilities = preparation_capabilities();
        let failures = crate::acceleration::SessionFailures::new(&capabilities);
        let warm = load_preparation_plan(&capabilities, &failures, &settings, true).unwrap();
        assert_eq!(warm.features, capabilities.available);
        assert_eq!(settings.acceleration.mode, crate::acceleration::AccelerationMode::Auto);
        let actual = crate::acceleration::resolve(&settings.acceleration, &capabilities,
            crate::acceleration::SamplingPreference::Auto, &failures).unwrap();
        assert_eq!(actual.features, crate::acceleration::FeatureSet::default(),
            "warming available code must not claim Auto selected unverified speedups");
    }

    #[test]
    fn load_preparation_skips_compatible_cpu_and_unimplemented_provider_features() {
        use crate::acceleration::*;
        let capabilities = preparation_capabilities();
        let mut settings = preparation_settings();
        settings.acceleration.mode = AccelerationMode::Compatible;
        assert_eq!(load_preparation_plan(&capabilities, &SessionFailures::new(&capabilities),
            &settings, true).unwrap().features, FeatureSet::default());
        settings.acceleration.mode = AccelerationMode::Optimized;
        assert_eq!(load_preparation_plan(&capabilities, &SessionFailures::new(&capabilities),
            &settings, false).unwrap().features, FeatureSet::default(),
            "a capability declaration cannot replace a missing compiled implementation");
        for (vendor, backend) in [(GpuVendor::Cpu, ComputeBackend::Cpu),
            (GpuVendor::Amd, ComputeBackend::Hip), (GpuVendor::Intel, ComputeBackend::Vulkan)] {
            let baseline = BackendCapabilities::baseline(vendor, backend, "test:ordinary-provider");
            let plan = load_preparation_plan(&baseline, &SessionFailures::new(&baseline),
                &settings, true).unwrap();
            assert_eq!(plan.features, FeatureSet::default());
            assert_eq!((plan.vendor, plan.backend), (vendor, backend));
        }
    }

    #[test]
    fn load_preparation_respects_explicit_feature_disables_and_cpu_sampling() {
        use crate::acceleration::*;
        let capabilities = preparation_capabilities();
        let mut settings = preparation_settings();
        settings.acceleration.fusion = FeaturePreference::Disabled;
        settings.acceleration.split_k = FeaturePreference::Disabled;
        settings.acceleration.primitive_graph = FeaturePreference::Enabled;
        settings.sampling_override = Some(false);
        let plan = load_preparation_plan(&capabilities, &SessionFailures::new(&capabilities),
            &settings, true).unwrap();
        assert_eq!(plan.features, FeatureSet::default(),
            "a requested Graph must not secretly re-enable explicitly disabled dependencies");
    }

    #[test]
    fn load_preparation_failures_keep_same_provider_and_disable_only_optional_code() {
        use crate::acceleration::*;
        let capabilities = preparation_capabilities();
        let mut failures = SessionFailures::new(&capabilities);
        failures.record(OptimizationFeature::Fusion, FailureClass::Preparation, "compiler unavailable");
        let first = load_preparation_plan(&capabilities, &failures, &preparation_settings(), true).unwrap();
        assert!(!first.features.fusion && !first.features.primitive_graph);
        assert!(first.features.split_k && first.features.gpu_sampling);
        for feature in [OptimizationFeature::SplitK, OptimizationFeature::GpuSampling] {
            failures.record(feature, FailureClass::Preparation, "optional compiler rejected");
        }
        let ordinary = load_preparation_plan(&capabilities, &failures, &preparation_settings(), true).unwrap();
        assert_eq!(ordinary.features, FeatureSet::default());
        assert_eq!(ordinary.path, ExecutionPath::OrdinaryGpu);
        assert_eq!(ordinary.backend, capabilities.backend);
        assert_eq!(ordinary.vendor, capabilities.vendor);
        failures.record(OptimizationFeature::Fusion, FailureClass::ExecutionOrDevice, "device lost");
        assert!(load_preparation_plan(&capabilities, &failures, &preparation_settings(), true).is_err(),
            "uncertain device state must not be retried as a normal compile failure");
    }

    #[test]
    fn load_preparation_validates_workload_and_masks_unsupported_sampler_scalars() {
        let capabilities = preparation_capabilities();
        let failures = crate::acceleration::SessionFailures::new(&capabilities);
        let mut settings = preparation_settings();
        for lanes in [0, usize::MAX] {
            settings.lanes = lanes;
            assert!(load_preparation_plan(&capabilities, &failures, &settings, true).is_err());
        }
        for lanes in [17, 32, 64, 96] {
            settings.lanes = lanes;
            assert!(load_preparation_plan(&capabilities, &failures, &settings, true).is_ok());
        }
        settings.lanes = 8;
        settings.renyi_alpha = f64::NAN;
        assert!(load_preparation_plan(&capabilities, &failures, &settings, true).is_err());
        settings.renyi_alpha = 4.;
        settings.cfg_scale = f64::MAX;
        let plan = load_preparation_plan(&capabilities, &failures, &settings, true).unwrap();
        assert!(!plan.features.gpu_sampling && !plan.features.primitive_graph);
        assert!(plan.features.fusion);
    }
    #[test]
    fn optional_component_initialization_panic_preserves_the_ordinary_path() {
        let result=prepare_optional(||panic!("simulated optional compiler unavailable"));
        assert!(result.unwrap_err().to_string().contains("普通执行仍可使用"));
        assert!(prepare_optional(||Ok(())).is_ok());
    }

    #[test]
    fn entropy_uniform_and_concentrated_support() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..100 {
            assert_eq!(
                renyi_uniform_index(&[90., 0., -90.], 4., &mut rng).unwrap(),
                0
            );
        }
        let mut seen = HashSet::new();
        for _ in 0..100 {
            seen.insert(renyi_uniform_index(&[0.; 4], 4., &mut rng).unwrap());
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn unequal_probabilities_are_uniform_only_inside_effective_support() {
        // Softmax is strongly unequal, yet alpha=4 retains the top two. The
        // intended policy is uniform over those two, not temperature sampling.
        let mut rng = StdRng::seed_from_u64(91);
        let mut counts = [0usize; 3];
        for _ in 0..5000 {
            counts[renyi_uniform_index(&[10., 9., 0.], 4., &mut rng).unwrap()] += 1;
        }
        assert_eq!(counts[2], 0);
        assert!((2300..2700).contains(&counts[0]));
        assert!((2300..2700).contains(&counts[1]));
        assert!(renyi_uniform_index(&[1., 0.], 1., &mut rng).is_err());
        assert!(renyi_uniform_index(&[f64::NAN, 0.], 4., &mut rng).is_err());
    }

    fn ordinary() -> [u32; 9] {
        [512, 1536, 2048, 2518, 4566, 6415, 6702, 6830, 6958]
    }

    fn global_prefix() -> Vec<[u32; 9]> {
        let mut out = vec![BOS_RECORD];
        let mut bg = ordinary();
        bg[5] = OFFSETS[5];
        out.push(bg);
        out.extend(vec![ordinary(); 85]);
        out
    }

    fn complete_block(side: usize, index: usize, count: usize) -> RegionBlock {
        let mut records = vec![region_marker(side, index)];
        records.extend(vec![ordinary(); count]);
        records.push(END_RECORD);
        RegionBlock {
            side,
            index,
            records,
        }
    }

    #[test]
    fn stage_pause_arms_only_the_unfinished_request_stage_and_resume_advances() {
        let mut full = global_prefix();
        full.pop();
        let before = parse_history(&full).unwrap();
        let target = stage_pause_target(&before, true);
        assert_eq!(target, Some(Stage::Global));
        assert_eq!(reached_stage_pause(&before, target), None);
        full.push(ordinary());
        let global_done = parse_history(&full).unwrap();
        assert_eq!(reached_stage_pause(&global_done, target), Some(Stage::Global));
        assert_eq!(full.len(), 87);
        assert!(!full.iter().any(is_marker), "pause must precede the first coarse marker");
        assert_eq!(stage_pause_target(&global_done, true), Some(Stage::Coarse),
            "an explicit continuation must not pause again at the global boundary");
        assert_eq!(stage_pause_target(&global_done, false), None);
        let oversized = parse_history(&[full, vec![ordinary(); 5]].concat()).unwrap();
        assert_eq!(stage_pause_target(&oversized, true), Some(Stage::Coarse));
    }

    #[test]
    fn coarse_pause_requires_all_sixteen_real_ends_including_empty_visits() {
        let mut full = global_prefix();
        let target = stage_pause_target(&parse_history(&full).unwrap(), true);
        for index in 0..16 {
            full.push(region_marker(4, index));
            let open = parse_history(&full).unwrap();
            assert_eq!(reached_stage_pause(&open, target), None);
            full.push(END_RECORD); // Empty TRUE_END visits still complete a grid cell.
            let closed = parse_history(&full).unwrap();
            assert_eq!(reached_stage_pause(&closed, target),
                (index == 15).then_some(Stage::Coarse));
        }
        let completed = parse_history(&full).unwrap();
        assert_eq!(stage_pause_target(&completed, true), None,
            "a fresh request must enter fine grids after the coarse boundary");
        assert_eq!(next_stage(&completed), Some(8));
        assert_eq!(full.last(), Some(&END_RECORD));
        assert_eq!(completed.highest_side, Some(4), "no fine marker was appended");
    }

    #[test]
    fn capped_final_coarse_visit_closes_even_at_the_request_primitive_limit() {
        let mut full = global_prefix();
        for index in 0..15 { full.extend(complete_block(4, index, 0).records); }
        full.push(region_marker(4, 15));
        full.extend(vec![ordinary(); 19]);
        let mut history = parse_history(&full).unwrap();
        let target = stage_pause_target(&history, true);
        assert_eq!(target, Some(Stage::Coarse));
        assert!(!capped_final_coarse_visit(&history, history.active.as_ref().unwrap(), target));
        full.push(ordinary());
        history.active.as_mut().unwrap().records.push(ordinary());
        *history.visit_counts.get_mut(&(4, 15)).unwrap() += 1;
        assert!(capped_final_coarse_visit(&history, history.active.as_ref().unwrap(), target));
        assert!(!capped_final_coarse_visit(&history, history.active.as_ref().unwrap(), None));
        full.push(END_RECORD);
        let active = history.active.take().unwrap();
        finish_visit(&mut history, active).unwrap();
        assert_eq!(reached_stage_pause(&history, target), Some(Stage::Coarse));
        assert_eq!(parse_history(&full).unwrap().closed.len(), 16);
    }

    #[test]
    fn reopened_and_interrupted_cells_use_effective_closed_membership_for_pause() {
        let mut full = global_prefix();
        for index in 0..16 { full.extend(complete_block(4, index, 0).records); }
        full.extend([region_marker(4, 2), INTERRUPT_RECORD, region_marker(4, 3)]);
        let history = parse_history(&full).unwrap();
        let target = stage_pause_target(&history, true);
        assert_eq!(target, Some(Stage::Coarse));
        assert_eq!(history.closed.len(), 14);
        full.push(END_RECORD);
        assert_eq!(reached_stage_pause(&parse_history(&full).unwrap(), target), None);
        full.extend([region_marker(4, 2), END_RECORD]);
        let resumed = parse_history(&full).unwrap();
        assert_eq!(reached_stage_pause(&resumed, target), Some(Stage::Coarse));
        assert!(resumed.interrupted.is_empty());
        full.push(region_marker(8, 0));
        assert_eq!(stage_pause_target(&parse_history(&full).unwrap(), true), None,
            "fine regions must not acquire a stage pause");
    }

    #[test]
    fn stopped_region_history_retains_exact_resume_position() {
        let mut full = global_prefix();
        full.extend(complete_block(4, 5, 3).records);
        full.push(region_marker(4, 9));
        full.extend(vec![ordinary(); 4]);
        let parsed = parse_history(&full).unwrap();
        assert_eq!(parsed.fixed.len(), 87);
        assert_eq!(parsed.complete.len(), 1);
        let active = parsed.active.unwrap();
        assert_eq!((active.side, active.index), (4, 9));
        assert_eq!(active.records.len() - 1, 4);
        assert_eq!(parsed.complete[0].records.len(), 5);
    }

    #[test]
    fn explicit_reopen_starts_a_new_visit_and_manual_over_cap_is_retained() {
        let mut full = global_prefix();
        full.extend(complete_block(4, 1, 20).records.clone());
        full.push(region_marker(4, 1));
        let reopened = parse_history(&full).unwrap();
        assert_eq!(reopened.closed.len(), 0);
        assert_eq!(reopened.visit_counts[&(4, 1)], 0);
        assert_eq!(reopened.complete.len(), 1);
        full.push(ordinary());
        assert_eq!(parse_history(&full).unwrap().visit_counts[&(4, 1)], 1);
        let mut full = global_prefix();
        full.extend(complete_block(8, 0, 11).records);
        let manual = parse_history(&full).unwrap();
        assert_eq!(manual.closed.len(), 1);
        assert_eq!(manual.complete[0].records.len(), 13);
        let projected = bounded_context_blocks(&manual.complete[0]);
        assert_eq!(
            projected.iter().map(|r| r.records.len() - 2).sum::<usize>(),
            11
        );
        assert!(projected.iter().all(|r| r.records.len() <= 12));
    }

    #[test]
    fn interrupts_preserve_unfinished_visit_counts_and_only_true_end_completes() {
        let mut raw = global_prefix();
        raw.push(region_marker(4, 0));
        raw.extend(vec![ordinary(); 18]);
        raw.push(INTERRUPT_RECORD);
        raw.push(region_marker(4, 5));
        raw.push(ordinary());
        raw.push(INTERRUPT_RECORD);
        raw.push(region_marker(4, 0));
        raw.extend(vec![ordinary(); 2]);
        let mut parsed = parse_history(&raw).unwrap();
        assert_eq!(parsed.closed.len(), 0);
        assert_eq!(parsed.visit_counts[&(4, 0)], 20);
        assert_eq!(parsed.visit_counts[&(4, 5)], 1);
        assert_eq!(parsed.interrupted.len(), 2);
        assert_eq!(parsed.active.as_ref().unwrap().records.len() - 1, 2);
        let event = DrawingRegion::from_history(&raw).unwrap();
        assert_eq!(event.generated_in_region, 20);
        assert_eq!(event.completed_coarse, 0);
        // Automatic generation sees its cumulative cap already reached and
        // closes this visit rather than appending a twenty-first primitive.
        let active = parsed.active.take().unwrap();
        assert!(parsed.visit_counts[&(4, 0)] >= active.cap());
        finish_visit(&mut parsed, active).unwrap();
        raw.push(END_RECORD);
        assert_eq!(parsed.closed.len(), 1);
        assert_eq!(parsed.complete[0].records.len(), 22);
        assert_eq!(parsed.interrupted.len(), 1);
        assert_eq!(parsed.visit_counts[&(4, 5)], 1);
        assert!(!parsed.visit_counts.contains_key(&(4, 0)));
        assert_eq!(
            DrawingRegion::from_history(&raw).unwrap().completed_coarse,
            1
        );
    }

    #[test]
    fn interrupt_pause_and_same_stage_marker_order_are_strict() {
        let mut paused = global_prefix();
        paused.push(region_marker(4, 2));
        paused.push(ordinary());
        paused.push(INTERRUPT_RECORD);
        let parsed = parse_history(&paused).unwrap();
        assert_eq!(parsed.pending_interrupt_side, Some(4));
        assert_eq!(next_stage(&parsed), Some(4));
        assert_eq!(parsed.closed.len(), 0);
        assert_eq!(parsed.visit_counts[&(4, 2)], 1);
        let event = DrawingRegion::from_history(&paused).unwrap();
        assert!(event.awaiting_region);
        let mut wrong = paused.clone();
        wrong.push(ordinary());
        assert!(parse_history(&wrong).is_err());
        let mut wrong = paused.clone();
        wrong.push(region_marker(8, 2));
        assert!(parse_history(&wrong).is_err());
        let mut resumed = paused;
        resumed.push(region_marker(4, 2));
        resumed.push(ordinary());
        assert_eq!(parse_history(&resumed).unwrap().visit_counts[&(4, 2)], 2);
        let mut reversed = global_prefix();
        reversed.extend(complete_block(8, 1, 0).records);
        reversed.push(region_marker(4, 1));
        assert!(parse_history(&reversed).is_err());
    }

    #[test]
    fn many_interrupts_do_not_reset_automatic_visit_limit_or_count_as_completion() {
        let mut raw = global_prefix();
        raw.push(region_marker(8, 3));
        for count in 0..14 {
            raw.push(ordinary());
            if count < 13 {
                raw.push(INTERRUPT_RECORD);
                raw.push(region_marker(8, 3));
            }
        }
        let parsed = parse_history(&raw).unwrap();
        assert_eq!(parsed.visit_counts[&(8, 3)], 14);
        assert_eq!(parsed.closed.len(), 0);
        assert_eq!(
            DrawingRegion::from_history(&raw)
                .unwrap()
                .generated_in_region,
            14
        );
        assert!(parsed.visit_counts[&(8, 3)] >= parsed.active.as_ref().unwrap().cap());
        assert_eq!(visible_records_count(&raw), 86 + 14);
    }

    fn visible_records_count(raw: &[[u32; 9]]) -> usize {
        raw.iter().filter(|r| visible(r)).count()
    }

    #[test]
    fn packed_suspended_fragments_precede_same_stage_active_marker() {
        let fixed = global_prefix();
        let mut fragment = complete_block(8, 0, 4);
        *fragment.records.last_mut().unwrap() = INTERRUPT_RECORD;
        let complete = vec![complete_block(4, 15, 2), fragment, complete_block(8, 1, 2)];
        let target = RegionBlock {
            side: 8,
            index: 2,
            records: vec![region_marker(8, 2), ordinary()],
        };
        let mut context = pack_context(&fixed, &complete, &target, 224).unwrap();
        context.extend_from_slice(&target.records);
        let mut interrupted = 0;
        for (i, row) in context.iter().enumerate() {
            if row == &INTERRUPT_RECORD {
                interrupted += 1;
                assert!(is_marker(&context[i + 1]));
                assert_eq!(marker_identity(&context[i + 1]).unwrap().0, 8);
            }
        }
        assert_eq!(interrupted, 1);
        assert!(context.len() + 224 + target.cap() - 1 <= 1024);
    }

    #[test]
    fn malformed_controls_and_nested_markers_are_still_rejected() {
        let mut nested = global_prefix();
        nested.push(region_marker(4, 0));
        nested.push(region_marker(4, 1));
        assert!(parse_history(&nested).is_err());
        let mut malformed = global_prefix();
        malformed.push(region_marker(4, 0));
        let mut bad = INTERRUPT_RECORD;
        bad[1] = 0;
        malformed.push(bad);
        assert!(parse_history(&malformed).is_err());
    }

    #[test]
    fn region_markers_roundtrip() {
        for side in [4, 8] {
            for i in 0..side * side {
                assert_eq!(marker_identity(&region_marker(side, i)).unwrap(), (side, i));
            }
        }
    }

    #[test]
    fn legal_center_candidates_match_renderer_inclusive_bounds_for_all_regions() {
        let mut contexts = vec![None];
        contexts.extend((0..16).map(|index| Some((4, index))));
        contexts.extend((0..64).map(|index| Some((8, index))));
        for region in contexts {
            let (x0, x1, y0, y1) = center_bin_bounds(region).unwrap();
            let expected = match region {
                None => 1023,
                Some((4, _)) => 255,
                Some((8, _)) => 127,
                _ => unreachable!(),
            };
            for (field, lo, hi) in [(0, x0, x1), (1, y0, y1)] {
                let candidates = field_candidates(field, false, region).unwrap();
                assert_eq!(candidates.len(), expected);
                assert_eq!(candidates[0], OFFSETS[field] + lo);
                assert_eq!(*candidates.last().unwrap(), OFFSETS[field] + hi);
                assert!(candidates.iter().all(|&v| v - OFFSETS[field] <= 1022));
                assert_eq!(
                    candidates.windows(2).filter(|w| w[1] != w[0] + 1).count(),
                    0
                );
            }
        }
        assert_eq!(
            center_bin_bounds(Some((4, 15))).unwrap(),
            (768, 1022, 768, 1022)
        );
        assert_eq!(
            center_bin_bounds(Some((8, 63))).unwrap(),
            (896, 1022, 896, 1022)
        );
        assert!(center_bin_bounds(Some((8, 64))).is_err());
        assert!(center_bin_bounds(Some((3, 0))).is_err());
        assert_eq!(field_candidates(5, true, None).unwrap(), vec![6414]);
        assert_eq!(
            field_candidates(5, false, Some((8, 63))).unwrap(),
            vec![6415, 6416]
        );
        assert_eq!(
            field_candidates(2, false, Some((8, 63))).unwrap().len(),
            270
        );
    }

    #[test]
    fn overwhelming_illegal_coordinate_logits_never_enter_entropy_or_sampling() {
        let mut rng = StdRng::seed_from_u64(17);
        for region in [
            None,
            Some((4, 0)),
            Some((4, 15)),
            Some((8, 0)),
            Some((8, 63)),
        ] {
            for field in [0, 1] {
                let candidates = field_candidates(field, false, region).unwrap();
                let mut full = vec![0.; 1024];
                full[1023] = 1e30;
                full[(candidates[0] - OFFSETS[field]) as usize] = 50.;
                let legal: Vec<_> = candidates
                    .iter()
                    .map(|&id| full[(id - OFFSETS[field]) as usize])
                    .collect();
                for _ in 0..8 {
                    let chosen = renyi_uniform_index(&legal, 4., &mut rng).unwrap();
                    assert_eq!(candidates[chosen], candidates[0]);
                    assert_ne!(candidates[chosen] - OFFSETS[field], 1023);
                }
            }
        }
    }

    #[test]
    fn packing_preserves_whole_regions_and_budget() {
        let fixed = vec![BOS_RECORD; 87];
        let mut complete = Vec::new();
        for i in 0..16 {
            let mut records = vec![region_marker(4, i)];
            records.extend(vec![[0; 9]; i % 20]);
            records.push(END_RECORD);
            complete.push(RegionBlock {
                side: 4,
                index: i,
                records,
            });
        }
        let target = RegionBlock {
            side: 8,
            index: 4,
            records: vec![region_marker(8, 4)],
        };
        let packed = pack_context(&fixed, &complete, &target, 224).unwrap();
        assert!(packed.len() + target.cap() + 2 + 224 <= 1024);
        assert_eq!(&packed[..87], fixed.as_slice());
        assert!(packed.len() >= fixed.len());
    }

    fn projected_blocks(context: &[[u32; 9]], fixed_count: usize) -> Vec<RegionBlock> {
        let mut blocks = Vec::new();
        let mut cursor = fixed_count;
        while cursor < context.len() {
            let (side, index) = marker_identity(&context[cursor]).unwrap();
            let start = cursor;
            cursor += 1;
            while cursor < context.len()
                && context[cursor] != END_RECORD
                && context[cursor] != INTERRUPT_RECORD
            {
                validate_visible(&context[cursor]).unwrap();
                cursor += 1;
            }
            assert!(cursor < context.len(), "projection split a region");
            cursor += 1;
            blocks.push(RegionBlock { side, index, records: context[start..cursor].to_vec() });
        }
        blocks
    }

    #[test]
    fn overclock_keeps_global_then_all_coarse_then_nearest_fine_and_can_exceed_897() {
        let fixed = global_prefix();
        let mut history: Vec<_> = (0..16).rev().map(|i| complete_block(4, i, 20)).collect();
        history.extend((0..63).map(|i| complete_block(8, i, 10)));
        let target = RegionBlock { side: 8, index: 63, records: vec![region_marker(8, 63)] };
        let context = pack_context_for_mode(&fixed, &history, &target, 2, true).unwrap();
        assert_eq!(&context[..fixed.len()], fixed.as_slice());
        assert!(context.len() > 897);
        let selected = projected_blocks(&context, fixed.len());
        assert_eq!(selected[..16].iter().map(|b| b.index).collect::<Vec<_>>(), (0..16).rev().collect::<Vec<_>>());
        assert!(selected[..16].iter().all(|b| b.side == 4));
        assert!(selected[16..].iter().all(|b| b.side == 8));
        let center = target.center();
        let distance = |block: &RegionBlock| {
            let point = block.center();
            (point.0 - center.0).powi(2) + (point.1 - center.1).powi(2)
        };
        let fine = &selected[16..];
        assert!(fine.windows(2).all(|pair| distance(&pair[0]) <= distance(&pair[1])));
        assert!(fine.iter().any(|b| b.index == 62));
        assert!(!fine.iter().any(|b| b.index == 0));
        assert!(selected.iter().all(|block| history.iter().any(|original|
            original.side == block.side && original.index == block.index && original.records == block.records)));
        let remaining = MAX_CONTEXT - 2 - context.len();
        assert!((12..24).contains(&remaining));
        assert_eq!(context, pack_context_for_mode(&fixed, &history, &target, 2, true).unwrap());
    }

    #[test]
    fn overclock_long_text_and_partly_drawn_active_region_never_exceed_hard_limit() {
        let fixed = global_prefix();
        let mut history: Vec<_> = (0..16).map(|i| complete_block(4, i, 20)).collect();
        history.extend((0..63).map(|i| complete_block(8, i, 10)));
        for (side, text_length, existing) in [(8, 224, 7), (8, 2, 9), (4, 224, 17)] {
            let mut records = vec![region_marker(side, side * side - 1)];
            records.extend(vec![ordinary(); existing]);
            let target = RegionBlock { side, index: side * side - 1, records };
            let context = pack_overclock_context(&fixed, &history, &target, text_length).unwrap();
            let generation_footprint = target.records.len() + target.cap() - existing + 1;
            assert!(text_length + context.len() + generation_footprint <= MAX_CONTEXT);
            // All training-sized coarse history is higher priority than fine.
            assert!(projected_blocks(&context, fixed.len())[..16].iter().all(|b| b.side == 4));
        }
    }

    #[test]
    fn overclock_small_context_keeps_every_block_and_off_mode_is_unchanged() {
        let fixed = global_prefix();
        let history = vec![complete_block(4, 1, 2), complete_block(8, 7, 3), complete_block(8, 8, 1)];
        let target = RegionBlock { side: 8, index: 0, records: vec![region_marker(8, 0)] };
        let context = pack_overclock_context(&fixed, &history, &target, 224).unwrap();
        assert_eq!(context.len(), fixed.len() + history.iter().map(|b| b.records.len()).sum::<usize>());
        assert_eq!(pack_context_for_mode(&fixed, &history, &target, 224, false).unwrap(),
            pack_context(&fixed, &history, &target, 224).unwrap());
        assert!(pack_overclock_context(&fixed, &history, &target, 1024).is_err());
    }

    #[test]
    fn overclock_preserves_interrupted_control_chain_and_raw_history() {
        let fixed = global_prefix();
        let closed = complete_block(4, 1, 20);
        let interrupted = RegionBlock { side: 4, index: 2, records: vec![region_marker(4, 2), ordinary(), INTERRUPT_RECORD] };
        let target = RegionBlock { side: 4, index: 3, records: vec![region_marker(4, 3), ordinary()] };
        let mut full = fixed.clone();
        full.extend_from_slice(&closed.records);
        full.extend_from_slice(&interrupted.records);
        full.extend_from_slice(&target.records);
        let original = full.clone();
        let context = pack_overclock_context(&fixed, &[closed, interrupted], &target, 224).unwrap();
        let mut active_context = context;
        active_context.extend_from_slice(&target.records);
        let parsed = parse_history(&active_context).unwrap();
        assert_eq!(parsed.interrupted.len(), 1);
        assert_eq!(parsed.active.unwrap().index, 3);
        assert_eq!(full, original);
    }

    #[test]
    fn overclock_large_manual_global_prefix_retains_priority_and_source_unchanged() {
        let mut fixed = global_prefix();
        fixed.extend(vec![ordinary(); 1500]);
        let original = fixed.clone();
        let target = RegionBlock { side: 8, index: 0, records: vec![region_marker(8, 0)] };
        let context = pack_overclock_context(&fixed, &[], &target, 2).unwrap();
        assert!(context.len() > 897);
        assert!(context.len() + 2 + 12 <= MAX_CONTEXT);
        assert_eq!(context, fixed[..MAX_CONTEXT - 2 - 12]);
        assert_eq!(fixed, original);
        assert!(!context.iter().any(is_marker));
    }

    #[test]
    fn overclock_keeps_more_than_86_global_primitives_before_coarse_and_fine() {
        let mut fixed = global_prefix();
        fixed.extend(vec![ordinary(); 100]);
        let blocks = vec![complete_block(4, 1, 20), complete_block(8, 2, 10)];
        let target = RegionBlock { side: 8, index: 3, records: vec![region_marker(8, 3)] };
        let context = pack_overclock_context(&fixed, &blocks, &target, 224).unwrap();
        assert_eq!(&context[..fixed.len()], fixed.as_slice());
        let selected = projected_blocks(&context, fixed.len());
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].side, 4);
        assert_eq!(selected[1].side, 8);
    }

    #[test]
    fn overclock_oversized_global_prefix_excludes_lower_priority_regions() {
        let mut fixed = global_prefix();
        fixed.extend(vec![ordinary(); 1500]);
        let blocks = vec![complete_block(4, 1, 20), complete_block(8, 2, 10)];
        let target = RegionBlock { side: 8, index: 3, records: vec![region_marker(8, 3)] };
        let context = pack_overclock_context(&fixed, &blocks, &target, 224).unwrap();
        assert_eq!(context.len(), MAX_CONTEXT - 224 - 12);
        assert_eq!(context, fixed[..context.len()]);
        assert!(!context.iter().any(is_marker));
    }

    #[test]
    fn rolling_context_fills_threshold_without_truncating_regions() {
        let fixed = global_prefix();
        let mut complete: Vec<_> = (0..16).map(|i| complete_block(4, i, 20)).collect();
        complete.extend((0..63).map(|i| complete_block(8, i, 10)));
        let target = RegionBlock {
            side: 8,
            index: 63,
            records: vec![region_marker(8, 63)],
        };
        let context = pack_context(&fixed, &complete, &target, 224).unwrap();
        let remaining = 800 - context.len();
        assert!((12..24).contains(&remaining));
        // This is a distance-reordered tensor prefill window, not chronological
        // full raw history. Coarse and fine completed blocks may interleave;
        // the full-raw parser correctly rejects stage reversal and must not be
        // used to validate this input-only projection.
        let mut selected = Vec::new();
        let mut cursor = fixed.len();
        while cursor < context.len() {
            let (side, index) = marker_identity(&context[cursor]).unwrap();
            let start = cursor;
            cursor += 1;
            while cursor < context.len() && context[cursor] != END_RECORD {
                validate_visible(&context[cursor]).unwrap();
                cursor += 1;
            }
            assert!(
                cursor < context.len(),
                "context contains a truncated region block"
            );
            cursor += 1;
            let block = RegionBlock {
                side,
                index,
                records: context[start..cursor].to_vec(),
            };
            assert!(complete.iter().any(|original| original.side == side
                && original.index == index
                && original.records == block.records));
            selected.push(block);
        }
        assert!(selected.len() < complete.len());
        let mut last_distance = -1.;
        let c = target.center();
        for block in selected {
            let p = block.center();
            let d = (p.0 - c.0).powi(2) + (p.1 - c.1).powi(2);
            assert!(d >= last_distance);
            last_distance = d;
            assert!(block.records.last() == Some(&END_RECORD));
            assert_eq!(block.records.len(), block.cap() + 2);
        }
    }

    #[test]
    fn large_edited_canvas_stays_complete_while_model_input_is_bounded() {
        let mut fixed = global_prefix();
        for i in 0..1250 {
            let mut row = ordinary();
            row[0] = (i % 1024) as u32;
            row[1] = OFFSETS[1] + ((i * 17) % 1024) as u32;
            row[3] = OFFSETS[3] + ((i * 13) % 2048) as u32;
            fixed.push(row);
        }
        let before = fixed.clone();
        let target = RegionBlock {
            side: 8,
            index: 0,
            records: vec![region_marker(8, 0)],
        };
        let input = pack_context(&fixed, &[], &target, 224).unwrap();
        assert_eq!(fixed, before);
        assert_eq!(fixed.len(), 1337);
        assert_eq!(&input[..87], &fixed[..87]);
        assert!(input.len() + target.cap() + 2 + 224 <= MAX_CONTEXT);
        let projected = edited_context_blocks(&fixed[87..]).unwrap();
        assert_eq!(
            projected.iter().map(|b| b.records.len() - 2).sum::<usize>(),
            1250
        );
        let mut cursor = 87;
        let mut included = 0;
        while cursor < input.len() {
            assert!(is_marker(&input[cursor]));
            cursor += 1;
            let mut count = 0;
            while input[cursor] != END_RECORD {
                assert!(fixed[87..].contains(&input[cursor]));
                count += 1;
                cursor += 1;
            }
            assert!((1..=10).contains(&count));
            included += count;
            cursor += 1;
        }
        assert!(included > 0 && included < 1250);
        // The original raw history, including all visible records and order,
        // is still independently parseable; projected markers never entered it.
        let actual = parse_history(&fixed).unwrap();
        assert_eq!(actual.fixed, before);
        assert!(actual.complete.is_empty() && actual.active.is_none());
    }

    #[test]
    fn edited_projection_does_not_change_true_region_completion_or_full_order() {
        let mut fixed = global_prefix();
        fixed.extend(vec![ordinary(); 1000]);
        let completed = vec![complete_block(4, 2, 5)];
        let target = RegionBlock {
            side: 4,
            index: 6,
            records: vec![region_marker(4, 6)],
        };
        let mut raw = fixed.clone();
        raw.extend_from_slice(&completed[0].records);
        raw.push(region_marker(4, 6));
        raw.push(ordinary());
        let before = raw.clone();
        let input = pack_context(&fixed, &completed, &target, 224).unwrap();
        assert!(input.len() + 22 + 224 <= 1024);
        assert_eq!(raw, before);
        let restored = parse_history(&raw).unwrap();
        assert_eq!(restored.fixed, fixed);
        assert_eq!(restored.complete.len(), 1);
        assert_eq!(restored.complete[0].index, 2);
        assert_eq!(restored.active.as_ref().unwrap().index, 6);
        assert_eq!(next_stage(&restored), Some(4));
    }
}
