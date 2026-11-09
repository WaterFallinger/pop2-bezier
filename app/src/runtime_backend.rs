//! Application-facing backend boundary. UI/history contracts are shared; GPU
//! providers retain independent runtimes, implementations and failure state.
use crate::inference::{self, GenerationRequest, GeneratedPrimitive, GenerationSummary,
    LoadPreparationReport, LoadPreparationSettings, ParallelSummary};
use crate::{acceleration::{AccelerationSettings, BackendCapabilities}, region_state::DrawingRegion};
use anyhow::Result;
use std::sync::atomic::AtomicBool;

pub(super) enum Engine {
    Native(inference::NativeEngine),
    #[cfg(feature = "aten-hip")]
    Aten(inference::aten_engine::AtenEngine),
}

macro_rules! delegate {
    ($self:ident, $engine:ident, $call:expr) => {
        match $self {
            Engine::Native($engine) => $call,
            #[cfg(feature = "aten-hip")]
            Engine::Aten($engine) => $call,
        }
    };
}

impl Engine {
    pub fn load_configured(config: &inference::EngineConfig,
        settings: &LoadPreparationSettings, mut progress: impl FnMut(String)) -> Result<Self> {
        let cpu_requested=std::env::var("PAINTER_COMPUTE_PROVIDER").as_deref()==Ok("cpu");
        if cpu_requested {
            progress("兼容计算：当前使用CPU，生成速度会较慢；GPU不可用原因可在设置中的诊断日志查看".into());
            return inference::NativeEngine::load_configured(&cpu_config(config),settings,progress).map(Self::Native);
        }
        #[cfg(feature = "aten-hip")]
        let loaded=inference::aten_engine::AtenEngine::load_configured(config,settings,&mut progress).map(Self::Aten);
        #[cfg(not(feature = "aten-hip"))]
        let loaded=inference::NativeEngine::load_configured(config,settings,&mut progress).map(Self::Native);
        match loaded {
            Ok(engine)=>Ok(engine),
            Err(error) if std::env::var("PAINTER_ALLOW_CPU_FALLBACK").as_deref()==Ok("1")
                && config.model_path.is_file() && config.tokenizer_path.is_file()
                && can_fallback_before_generation(&format!("{error:#}")) => {
                crate::diagnostics::error("gpu_model_load",&error);
                crate::diagnostics::event("cpu_compatibility_fallback",serde_json::json!({
                    "stage":"model_load_before_user_execution","reason":format!("{error:#}"),"automatic":true,"visible_to_user":true}));
                progress(format!("CPU兼容原因：{error:#}"));
                progress("GPU基础路径暂不可用，已转为CPU兼容计算（速度较慢）；详细原因已写入诊断日志".into());
                inference::NativeEngine::load_configured(&cpu_config(config),settings,progress)
                    .map(Self::Native).map_err(|cpu|anyhow::anyhow!("GPU初始化不可用：{error:#}；CPU兼容加载也未完成：{cpu:#}"))
            },
            Err(error)=>Err(error),
        }
    }
    pub fn diagnostic_info(&self)->serde_json::Value { delegate!(self, engine, engine.diagnostic_info()) }
    pub(super) fn load_cpu_compatible(config:&inference::EngineConfig,settings:&LoadPreparationSettings,
        progress:impl FnMut(String))->Result<Self>{
        inference::NativeEngine::load_configured(&cpu_config(config),settings,progress).map(Self::Native)
    }
    pub fn model_description(&self) -> &str { delegate!(self, engine, &engine.model_description) }
    pub fn implementation_label(&self) -> &'static str {
        match self { Self::Native(_) => "原生 Rust", #[cfg(feature = "aten-hip")] Self::Aten(_) => "原生 C++ / Rust" }
    }
    pub fn acceleration_capabilities(&self) -> &BackendCapabilities {
        delegate!(self, engine, engine.acceleration_capabilities())
    }
    pub fn prepare_after_load(&mut self, settings: &LoadPreparationSettings,
        progress: impl FnMut(String)) -> Result<LoadPreparationReport> {
        delegate!(self, engine, engine.prepare_after_load(settings, progress))
    }
    pub fn set_gpu_sampling_override(&mut self, mode: Option<bool>) {
        delegate!(self, engine, engine.set_gpu_sampling_override(mode))
    }
    pub fn set_acceleration_preferences(&mut self, settings: AccelerationSettings) {
        delegate!(self, engine, engine.set_acceleration_preferences(settings))
    }
    pub fn set_diagnostic_cache_forecast(&mut self, forecast: Option<usize>) {
        delegate!(self, engine, engine.set_diagnostic_cache_forecast(forecast))
    }
    pub fn primitive_graph_status(&self) -> String { delegate!(self, engine, engine.primitive_graph_status()) }
    pub fn execution_error_message(&self, error: &anyhow::Error) -> String {
        delegate!(self, engine, engine.execution_error_message(error))
    }
    pub fn generate_with_regions(&mut self, request: &GenerationRequest, stop: &AtomicBool,
        on_primitive: impl FnMut(GeneratedPrimitive), on_status: impl FnMut(String),
        on_region: impl FnMut(DrawingRegion)) -> Result<GenerationSummary> {
        delegate!(self, engine, engine.generate_with_regions(request, stop, on_primitive, on_status, on_region))
    }
    pub fn generate_parallel(&mut self, request: &GenerationRequest, lanes: usize, stop: &AtomicBool,
        on_primitive: impl FnMut(usize, GeneratedPrimitive), on_status: impl FnMut(String),
        on_region: impl FnMut(usize, DrawingRegion), on_finished: impl FnMut(usize, GenerationSummary))
        -> Result<ParallelSummary> {
        delegate!(self, engine, engine.generate_parallel(request, lanes, stop, on_primitive,
            on_status, on_region, on_finished))
    }
}

fn cpu_config(config:&inference::EngineConfig)->inference::EngineConfig {
    inference::EngineConfig{model_path:config.model_path.clone(),tokenizer_path:config.tokenizer_path.clone(),use_cuda:false}
}

pub(super) fn can_fallback_before_generation(message:&str)->bool {
    let text=message.to_ascii_uppercase();
    if inference::uncertain_device_execution(message)
        ||["MODEL_NONFINITE","NATIVE WEIGHT METADATA"].iter().any(|part|text.contains(part)) {return false}
    ["CUDA_ERROR_","CUDA DRIVER","CUDA DEVICE","NO CUDA","HIP COUNT","NO HIP MODEL DEVICE","HIP MODEL DEVICE",
        "NOT IMPLEMENTED FOR","NOT SUPPORTED","NO KERNEL IMAGE","NO BINARY FOR GPU","OUT OF MEMORY",
        "无法加载ATEN原生库","缺少ATEN","原生库缺少接口","ATEN原生ABI","ROCBLAS_STATUS_ARCH_MISMATCH",
        "HIPERRORNOBINARYFORGPU","GPU基础","HIP DEVICE"].iter().any(|part|text.contains(part))
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn only_pre_execution_compatibility_failures_allow_cpu_fallback(){
        assert!(can_fallback_before_generation("CUDA_ERROR_NO_BINARY_FOR_GPU"));
        assert!(can_fallback_before_generation("no HIP model device"));
        assert!(!can_fallback_before_generation("EXECUTION_UNCERTAIN: CUDA_ERROR_ILLEGAL_ADDRESS"));
        assert!(!can_fallback_before_generation("CAPTURE_STATE_UNCERTAIN"));
        assert!(!can_fallback_before_generation("CAPTURE_STATE_UNCERTAIN: NOT SUPPORTED"));
        assert!(!can_fallback_before_generation("EXECUTION_UNCERTAIN: CUDA_ERROR_NO_BINARY_FOR_GPU"));
        assert!(!can_fallback_before_generation("CUDA_ERROR_ASSERT: NOT SUPPORTED"));
        assert!(!can_fallback_before_generation("model.pt file not found"));
        assert!(!can_fallback_before_generation("MODEL_NONFINITE"));
    }
}
