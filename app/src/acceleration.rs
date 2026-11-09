//! Pure capability and fallback policy. A display adapter is not an inference
//! backend: capabilities are supplied only after that provider actually loads.
//! The policy never changes devices, allocates tensors, retries a partial
//! primitive, or silently substitutes CPU execution for a missing GPU backend.

use serde::{Deserialize, Serialize};
use std::fmt;

#[path = "backend_detection.rs"]
mod backend_detection;
pub use backend_detection::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerationMode {
    #[default]
    Auto,
    Compatible,
    Optimized,
}
impl AccelerationMode {
    pub fn label(self) -> &'static str {
        match self { Self::Auto => "自动", Self::Compatible => "普通兼容", Self::Optimized => "优先加速" }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaturePreference {
    #[default]
    Auto,
    Disabled,
    Enabled,
}

/// SamplingBackend remains the one user-facing sampler control. Passing that
/// choice here avoids a second contradictory GPU-sampling checkbox.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamplingPreference { #[default] Auto, Cpu, Gpu }

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AccelerationSettings {
    pub mode: AccelerationMode,
    pub fusion: FeaturePreference,
    pub split_k: FeaturePreference,
    pub primitive_graph: FeaturePreference,
    pub kv_int8: FeaturePreference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuVendor { Nvidia, Amd, Intel, Other, Cpu }
impl GpuVendor {
    pub fn from_pci_vendor_id(id: u32) -> Self {
        match id { 0x10de => Self::Nvidia, 0x1002 => Self::Amd, 0x8086 => Self::Intel, _ => Self::Other }
    }
    pub fn label(self) -> &'static str {
        match self { Self::Nvidia => "NVIDIA", Self::Amd => "AMD", Self::Intel => "Intel", Self::Other => "其他显卡", Self::Cpu => "CPU" }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend { Cuda, Hip, AtenHip, Vulkan, Cpu }
impl ComputeBackend {
    pub fn label(self) -> &'static str {
        match self { Self::Cuda => "CUDA", Self::Hip => "HIP", Self::AtenHip => "HIP / ATen", Self::Vulkan => "Vulkan", Self::Cpu => "CPU" }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationFeature { GpuSampling, Fusion, SplitK, PrimitiveGraph }
impl OptimizationFeature {
    pub fn label(self) -> &'static str {
        match self {
            Self::GpuSampling => "GPU采样", Self::Fusion => "算子融合",
            Self::SplitK => "Split-K注意力", Self::PrimitiveGraph => "整图元Graph",
        }
    }
}
const FEATURES: [OptimizationFeature; 4] = [OptimizationFeature::GpuSampling, OptimizationFeature::Fusion, OptimizationFeature::SplitK, OptimizationFeature::PrimitiveGraph];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureSet {
    pub gpu_sampling: bool,
    pub fusion: bool,
    pub split_k: bool,
    pub primitive_graph: bool,
}
impl FeatureSet {
    pub fn has(self, feature: OptimizationFeature) -> bool {
        match feature { OptimizationFeature::GpuSampling => self.gpu_sampling, OptimizationFeature::Fusion => self.fusion, OptimizationFeature::SplitK => self.split_k, OptimizationFeature::PrimitiveGraph => self.primitive_graph }
    }
    fn set(&mut self, feature: OptimizationFeature, enabled: bool) {
        match feature { OptimizationFeature::GpuSampling => self.gpu_sampling = enabled, OptimizationFeature::Fusion => self.fusion = enabled, OptimizationFeature::SplitK => self.split_k = enabled, OptimizationFeature::PrimitiveGraph => self.primitive_graph = enabled }
    }
}

/// This describes the selected MODEL provider, not the Wgpu drawing adapter.
/// Never mark a feature available merely because its source was compiled.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub vendor: GpuVendor,
    pub backend: ComputeBackend,
    /// Stable provider+device identity, e.g. CUDA device UUID, for this load.
    pub device_key: String,
    pub ordinary_compute_available: bool,
    pub available: FeatureSet,
    /// Features with confirmed benefit for the current device/workload.
    pub auto_recommended: FeatureSet,
}
impl BackendCapabilities {
    pub fn baseline(vendor: GpuVendor, backend: ComputeBackend, device_key: impl Into<String>) -> Self {
        Self { vendor, backend, device_key: device_key.into(), ordinary_compute_available: true,
            available: FeatureSet::default(), auto_recommended: FeatureSet::default() }
    }
    fn validate(&self) -> Result<(), PolicyError> {
        let valid_pair = match self.backend {
            ComputeBackend::Cuda => self.vendor == GpuVendor::Nvidia,
            ComputeBackend::Hip | ComputeBackend::AtenHip => self.vendor == GpuVendor::Amd,
            ComputeBackend::Vulkan => self.vendor != GpuVendor::Cpu,
            ComputeBackend::Cpu => self.vendor == GpuVendor::Cpu,
        };
        if !valid_pair { return Err(PolicyError::InvalidProviderPair); }
        if self.device_key.is_empty() { return Err(PolicyError::MissingDeviceIdentity); }
        if !self.ordinary_compute_available { return Err(PolicyError::OrdinaryProviderUnavailable { vendor: self.vendor, backend: self.backend }); }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Unsupported,
    Preparation,
    AllocationBeforeExecution,
    /// Kernel execution failure, device loss, or any uncertain partial state.
    ExecutionOrDevice,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Caller must restore the last complete cache/RNG state before retrying.
    ReplanFromCompletePrimitive,
    StopPreservingCompletedCanvas,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureFailure {
    pub feature: OptimizationFeature,
    pub class: FailureClass,
    pub reason: String,
}

/// New model/device load => new session. Failures must not disable another GPU.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionFailures {
    device_key: String,
    failures: Vec<FeatureFailure>,
    fatal_reason: Option<String>,
}
impl SessionFailures {
    pub fn new(capabilities: &BackendCapabilities) -> Self {
        Self { device_key: capabilities.device_key.clone(), failures: Vec::new(), fatal_reason: None }
    }
    pub fn record(&mut self, feature: OptimizationFeature, class: FailureClass, reason: impl Into<String>) -> RecoveryAction {
        let reason = reason.into();
        if let Some(existing) = self.failures.iter_mut().find(|failure| failure.feature == feature) {
            *existing = FeatureFailure { feature, class, reason: reason.clone() };
        } else {
            self.failures.push(FeatureFailure { feature, class, reason: reason.clone() });
        }
        if class == FailureClass::ExecutionOrDevice {
            self.fatal_reason = Some(reason);
            RecoveryAction::StopPreservingCompletedCanvas
        } else {
            RecoveryAction::ReplanFromCompletePrimitive
        }
    }
    pub fn failures(&self) -> &[FeatureFailure] { &self.failures }
    fn reason(&self, feature: OptimizationFeature) -> Option<&str> {
        self.failures.iter().find(|failure| failure.feature == feature).map(|failure| failure.reason.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPath { OrdinaryGpu, OptimizedGpu, OrdinaryCpu }

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureDecision {
    pub feature: OptimizationFeature,
    pub enabled: bool,
    pub reason: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccelerationPlan {
    pub vendor: GpuVendor,
    pub backend: ComputeBackend,
    pub path: ExecutionPath,
    pub features: FeatureSet,
    pub decisions: Vec<FeatureDecision>,
}
impl AccelerationPlan {
    pub fn label(&self) -> String {
        let mode = match self.path {
            ExecutionPath::OrdinaryGpu => "普通兼容", ExecutionPath::OptimizedGpu => "加速执行", ExecutionPath::OrdinaryCpu => "CPU普通执行",
        };
        format!("{} · {} · {mode}", self.vendor.label(), self.backend.label())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyError {
    InvalidProviderPair,
    MissingDeviceIdentity,
    OrdinaryProviderUnavailable { vendor: GpuVendor, backend: ComputeBackend },
    FailureSessionMismatch,
    UnsafeDeviceState(String),
}
impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProviderPair => write!(f, "显卡厂商与模型执行后端不匹配"),
            Self::MissingDeviceIdentity => write!(f, "模型执行后端缺少设备标识"),
            Self::OrdinaryProviderUnavailable { vendor, backend } => write!(f, "{}的{}普通模型执行后端不可用；不会静默改用CPU", vendor.label(), backend.label()),
            Self::FailureSessionMismatch => write!(f, "加速失败记录属于其他设备，请为当前模型重新建立执行会话"),
            Self::UnsafeDeviceState(reason) => write!(f, "设备执行状态不确定，已保留完整画布并停止：{reason}"),
        }
    }
}
impl std::error::Error for PolicyError {}

fn preference(settings: &AccelerationSettings, feature: OptimizationFeature) -> FeaturePreference {
    match feature {
        OptimizationFeature::Fusion => settings.fusion,
        OptimizationFeature::SplitK => settings.split_k,
        OptimizationFeature::PrimitiveGraph => settings.primitive_graph,
        OptimizationFeature::GpuSampling => FeaturePreference::Auto,
    }
}

pub fn resolve(
    settings: &AccelerationSettings,
    capabilities: &BackendCapabilities,
    sampling: SamplingPreference,
    failures: &SessionFailures,
) -> Result<AccelerationPlan, PolicyError> {
    capabilities.validate()?;
    if failures.device_key != capabilities.device_key { return Err(PolicyError::FailureSessionMismatch); }
    if let Some(reason) = &failures.fatal_reason { return Err(PolicyError::UnsafeDeviceState(reason.clone())); }
    let cpu = capabilities.backend == ComputeBackend::Cpu;
    let mut plan = AccelerationPlan { vendor: capabilities.vendor, backend: capabilities.backend,
        path: if cpu { ExecutionPath::OrdinaryCpu } else { ExecutionPath::OrdinaryGpu },
        features: FeatureSet::default(), decisions: Vec::new() };

    for feature in [OptimizationFeature::Fusion, OptimizationFeature::SplitK, OptimizationFeature::PrimitiveGraph] {
        let selected = preference(settings, feature);
        let (enabled, reason) = if cpu {
            (false, "当前明确选择CPU执行，不启用GPU加速".into())
        } else if feature == OptimizationFeature::PrimitiveGraph && capabilities.backend == ComputeBackend::Hip {
            (false, "HIP Graph尚未实现，保留同设备普通逐步执行".into())
        } else if settings.mode == AccelerationMode::Compatible {
            (false, "普通兼容模式".into())
        } else if selected == FeaturePreference::Disabled {
            (false, "用户已关闭".into())
        } else if let Some(reason) = failures.reason(feature) {
            (false, format!("本次模型会话已停用：{reason}"))
        } else if !capabilities.available.has(feature) {
            (false, "当前模型后端尚不具备此能力".into())
        } else if selected == FeaturePreference::Enabled || settings.mode == AccelerationMode::Optimized || capabilities.auto_recommended.has(feature) {
            (true, "当前后端支持".into())
        } else {
            (false, "自动模式尚未确认当前设备／工作量的收益".into())
        };
        plan.features.set(feature, enabled);
        plan.decisions.push(FeatureDecision { feature, enabled, reason });
    }

    // The whole-primitive graph uses the static fused decoder. Respect an
    // explicit fusion-off choice rather than secretly switching that decoder
    // back on. Check this before sampling so Auto does not enable device
    // sampling solely for a graph that cannot actually execute.
    if plan.features.primitive_graph && !plan.features.fusion {
        let fusion_unavailable = settings.fusion == FeaturePreference::Disabled
            || !capabilities.available.fusion
            || failures.reason(OptimizationFeature::Fusion).is_some();
        if fusion_unavailable {
            plan.features.primitive_graph = false;
            let decision = plan.decisions.iter_mut().find(|item| item.feature == OptimizationFeature::PrimitiveGraph).unwrap();
            decision.enabled = false;
            decision.reason = "整图元Graph需要融合解码器；融合已关闭或不可用，保留普通逐步执行".into();
        }
    }

    let feature = OptimizationFeature::GpuSampling;
    let (gpu_sampling, reason) = if cpu || settings.mode == AccelerationMode::Compatible {
        (false, "普通执行使用CPU采样，模型设备不变".into())
    } else if sampling == SamplingPreference::Cpu {
        (false, "用户明确选择CPU采样".into())
    } else if let Some(reason) = failures.reason(feature) {
        (false, format!("本次模型会话已停用：{reason}"))
    } else if !capabilities.available.gpu_sampling {
        (false, "当前模型后端尚不具备GPU采样能力".into())
    } else if sampling == SamplingPreference::Gpu || settings.mode == AccelerationMode::Optimized || plan.features.primitive_graph || capabilities.auto_recommended.gpu_sampling {
        (true, "GPU采样可用".into())
    } else {
        (false, "自动模式保留普通采样路径".into())
    };
    plan.features.gpu_sampling = gpu_sampling;
    plan.decisions.push(FeatureDecision { feature, enabled: gpu_sampling, reason });
    if plan.features.primitive_graph && !gpu_sampling {
        plan.features.primitive_graph = false;
        let decision = plan.decisions.iter_mut().find(|item| item.feature == OptimizationFeature::PrimitiveGraph).unwrap();
        decision.enabled = false;
        decision.reason = "整图元Graph需要GPU内的采样与反馈；当前保留普通逐步执行".into();
    }
    if plan.features.primitive_graph && !plan.features.fusion {
        // Graph's dependencies are now both available, and fusion was not
        // explicitly disabled. Auto fusion follows this chosen execution path.
        plan.features.fusion = true;
        let decision = plan.decisions.iter_mut().find(|item| item.feature == OptimizationFeature::Fusion).unwrap();
        decision.enabled = true;
        decision.reason = "整图元Graph依赖融合解码器，已随该执行路径启用".into();
    }
    if !cpu && FEATURES.iter().any(|feature| plan.features.has(*feature)) {
        plan.path = ExecutionPath::OptimizedGpu;
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mature_aten_hip_graph_is_separate_from_unimplemented_candle_hip_graph() {
        let mut aten = BackendCapabilities::baseline(GpuVendor::Amd, ComputeBackend::AtenHip, "aten:actual-gpu");
        aten.available = FeatureSet { fusion: true, gpu_sampling: true, primitive_graph: true, split_k: false };
        aten.auto_recommended = aten.available;
        let selected = resolve(&AccelerationSettings::default(), &aten, SamplingPreference::Auto,
            &SessionFailures::new(&aten)).unwrap();
        assert!(selected.features.primitive_graph && selected.features.gpu_sampling);
        let mut legacy = aten.clone();
        legacy.backend = ComputeBackend::Hip;
        let ordinary = resolve(&AccelerationSettings::default(), &legacy, SamplingPreference::Auto,
            &SessionFailures::new(&legacy)).unwrap();
        assert!(!ordinary.features.primitive_graph);
        aten.vendor = GpuVendor::Nvidia;
        assert!(resolve(&AccelerationSettings::default(), &aten, SamplingPreference::Auto,
            &SessionFailures::new(&aten)).is_err());
    }
    fn nvidia() -> BackendCapabilities {
        let mut caps = BackendCapabilities::baseline(GpuVendor::Nvidia, ComputeBackend::Cuda, "cuda:uuid:test");
        caps.available = FeatureSet { gpu_sampling: true, fusion: true, split_k: true, primitive_graph: true };
        caps
    }
    fn plan(settings: &AccelerationSettings, caps: &BackendCapabilities, sampler: SamplingPreference) -> AccelerationPlan {
        resolve(settings, caps, sampler, &SessionFailures::new(caps)).unwrap()
    }
    #[test]
    fn old_settings_default_to_conservative_auto_and_round_trip() {
        assert_eq!(serde_json::from_str::<AccelerationSettings>("{}").unwrap(), AccelerationSettings::default());
        let settings = AccelerationSettings { mode: AccelerationMode::Optimized, fusion: FeaturePreference::Disabled, split_k: FeaturePreference::Enabled, primitive_graph: FeaturePreference::Auto, kv_int8: FeaturePreference::Enabled };
        assert_eq!(serde_json::from_str::<AccelerationSettings>(&serde_json::to_string(&settings).unwrap()).unwrap(), settings);
        let legacy: AccelerationSettings = serde_json::from_str(r#"{"mode":"optimized","fusion":"enabled","split_k":"disabled","primitive_graph":"auto"}"#).unwrap();
        assert_eq!(legacy.kv_int8, FeaturePreference::Auto);
        assert_eq!(AccelerationSettings::default().kv_int8, FeaturePreference::Auto);
    }
    #[test]
    fn auto_does_not_mistake_compiled_source_for_a_proven_speedup() {
        let caps = nvidia();
        let value = plan(&AccelerationSettings::default(), &caps, SamplingPreference::Auto);
        assert_eq!(value.features, FeatureSet::default());
        assert_eq!(value.path, ExecutionPath::OrdinaryGpu);
    }
    #[test]
    fn ordinary_compatibility_keeps_gpu_model_and_disables_all_optional_features() {
        let caps = nvidia();
        let value = plan(&AccelerationSettings { mode: AccelerationMode::Compatible, ..Default::default() }, &caps, SamplingPreference::Gpu);
        assert_eq!(value.backend, ComputeBackend::Cuda);
        assert_eq!(value.path, ExecutionPath::OrdinaryGpu);
        assert_eq!(value.features, FeatureSet::default());
        assert!(value.label().contains("NVIDIA · CUDA · 普通兼容"));
    }
    #[test]
    fn optimized_still_respects_explicit_disabled_features() {
        let caps = nvidia();
        let value = plan(&AccelerationSettings { mode: AccelerationMode::Optimized, split_k: FeaturePreference::Disabled, ..Default::default() }, &caps, SamplingPreference::Auto);
        assert!(value.features.fusion && value.features.primitive_graph && value.features.gpu_sampling);
        assert!(!value.features.split_k);
    }
    #[test]
    fn optimized_can_try_gpu_sampling_without_requiring_graph_support() {
        let mut caps = nvidia();
        caps.available = FeatureSet { gpu_sampling: true, ..Default::default() };
        let value = plan(&AccelerationSettings { mode: AccelerationMode::Optimized, ..Default::default() }, &caps, SamplingPreference::Auto);
        assert!(value.features.gpu_sampling);
        assert!(!value.features.primitive_graph);
    }
    #[test]
    fn graph_auto_recommendation_brings_required_device_sampling_with_it() {
        let mut caps = nvidia(); caps.auto_recommended.primitive_graph = true;
        let value = plan(&AccelerationSettings::default(), &caps, SamplingPreference::Auto);
        assert!(value.features.primitive_graph && value.features.gpu_sampling);
        assert!(value.features.fusion);
        assert!(!value.features.split_k);
    }
    #[test]
    fn explicit_fusion_off_prevents_graph_and_never_gets_overridden() {
        let mut caps = nvidia(); caps.auto_recommended.primitive_graph = true;
        let value = plan(&AccelerationSettings {
            fusion: FeaturePreference::Disabled,
            primitive_graph: FeaturePreference::Enabled,
            ..Default::default()
        }, &caps, SamplingPreference::Auto);
        assert!(!value.features.fusion && !value.features.primitive_graph && !value.features.gpu_sampling);
        assert!(value.decisions.iter().any(|item| item.feature == OptimizationFeature::PrimitiveGraph && item.reason.contains("需要融合")));
    }
    #[test]
    fn unavailable_or_failed_fusion_prevents_graph() {
        let mut caps = nvidia(); caps.auto_recommended.primitive_graph = true;
        caps.available.fusion = false;
        assert!(!plan(&AccelerationSettings::default(), &caps, SamplingPreference::Auto).features.primitive_graph);
        caps.available.fusion = true;
        let mut failed = SessionFailures::new(&caps);
        failed.record(OptimizationFeature::Fusion, FailureClass::Preparation, "fusion module unavailable");
        let value = resolve(&AccelerationSettings::default(), &caps, SamplingPreference::Auto, &failed).unwrap();
        assert!(!value.features.primitive_graph && !value.features.fusion && !value.features.gpu_sampling);
    }
    #[test]
    fn explicit_cpu_sampling_prevents_graph_without_changing_model_device() {
        let caps = nvidia();
        let value = plan(&AccelerationSettings { primitive_graph: FeaturePreference::Enabled, ..Default::default() }, &caps, SamplingPreference::Cpu);
        assert!(!value.features.primitive_graph && !value.features.gpu_sampling);
        assert!(!value.features.fusion);
        assert_eq!(value.backend, ComputeBackend::Cuda);
        assert!(value.decisions.iter().any(|item| item.feature == OptimizationFeature::PrimitiveGraph && item.reason.contains("需要GPU")));
    }
    #[test]
    fn unsupported_forced_feature_is_reported_and_ordinary_provider_remains() {
        let caps = BackendCapabilities::baseline(GpuVendor::Amd, ComputeBackend::Hip, "hip:uuid:rx7900");
        let value = plan(&AccelerationSettings { primitive_graph: FeaturePreference::Enabled, ..Default::default() }, &caps, SamplingPreference::Gpu);
        assert_eq!(value.vendor, GpuVendor::Amd);
        assert_eq!(value.backend, ComputeBackend::Hip);
        assert_eq!(value.path, ExecutionPath::OrdinaryGpu);
        assert_eq!(value.features, FeatureSet::default());
        assert!(!value.label().contains("CPU"));
    }
    #[test]
    fn amd_features_are_selected_from_amd_provider_capabilities_not_nvidia_defaults() {
        let mut caps = BackendCapabilities::baseline(GpuVendor::Amd, ComputeBackend::Hip, "hip:uuid:rx7900");
        caps.available.split_k = true; caps.auto_recommended.split_k = true;
        let value = plan(&AccelerationSettings::default(), &caps, SamplingPreference::Auto);
        assert!(value.features.split_k);
        assert!(!value.features.primitive_graph && !value.features.gpu_sampling && !value.features.fusion);
        assert_eq!(value.vendor, GpuVendor::Amd);
    }
    #[test]
    fn a_missing_amd_provider_never_silently_becomes_cpu_success() {
        let mut caps = BackendCapabilities::baseline(GpuVendor::Amd, ComputeBackend::Hip, "hip:missing");
        caps.ordinary_compute_available = false;
        assert!(matches!(resolve(&AccelerationSettings::default(), &caps, SamplingPreference::Auto, &SessionFailures::new(&caps)), Err(PolicyError::OrdinaryProviderUnavailable { vendor: GpuVendor::Amd, .. })));
    }
    #[test]
    fn preparation_failure_disables_graph_for_session_and_allows_replanning() {
        let mut caps = nvidia(); caps.auto_recommended.primitive_graph = true;
        let mut failed = SessionFailures::new(&caps);
        assert_eq!(failed.record(OptimizationFeature::PrimitiveGraph, FailureClass::Preparation, "capture unsupported"), RecoveryAction::ReplanFromCompletePrimitive);
        let value = resolve(&AccelerationSettings::default(), &caps, SamplingPreference::Auto, &failed).unwrap();
        assert!(!value.features.primitive_graph && !value.features.gpu_sampling);
        assert_eq!(value.path, ExecutionPath::OrdinaryGpu);
        assert!(value.decisions.iter().any(|item| item.reason.contains("capture unsupported")));
    }
    #[test]
    fn execution_failure_stops_instead_of_replaying_uncertain_partial_state() {
        let caps = nvidia(); let mut failed = SessionFailures::new(&caps);
        assert_eq!(failed.record(OptimizationFeature::PrimitiveGraph, FailureClass::ExecutionOrDevice, "device lost"), RecoveryAction::StopPreservingCompletedCanvas);
        assert!(matches!(resolve(&AccelerationSettings { mode: AccelerationMode::Compatible, ..Default::default() }, &caps, SamplingPreference::Auto, &failed), Err(PolicyError::UnsafeDeviceState(_))));
    }
    #[test]
    fn reload_or_device_change_requires_new_failure_session() {
        let caps = nvidia(); let failed = SessionFailures::new(&caps);
        let mut other = caps.clone(); other.device_key = "cuda:other".into();
        assert_eq!(resolve(&AccelerationSettings::default(), &other, SamplingPreference::Auto, &failed), Err(PolicyError::FailureSessionMismatch));
    }
    #[test]
    fn cpu_is_explicit_and_cannot_be_labeled_amd_acceleration() {
        let caps = BackendCapabilities::baseline(GpuVendor::Cpu, ComputeBackend::Cpu, "cpu");
        let value = plan(&AccelerationSettings { mode: AccelerationMode::Optimized, ..Default::default() }, &caps, SamplingPreference::Gpu);
        assert_eq!(value.path, ExecutionPath::OrdinaryCpu);
        assert_eq!(value.features, FeatureSet::default());
        assert!(value.label().starts_with("CPU · CPU"));
    }
    #[test]
    fn pci_vendor_identification_and_provider_pair_validation_are_unambiguous() {
        assert_eq!(GpuVendor::from_pci_vendor_id(0x10de), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_pci_vendor_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_pci_vendor_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_pci_vendor_id(0x1234), GpuVendor::Other);
        let caps = BackendCapabilities::baseline(GpuVendor::Amd, ComputeBackend::Cuda, "wrong");
        assert_eq!(resolve(&AccelerationSettings::default(), &caps, SamplingPreference::Auto, &SessionFailures::new(&caps)), Err(PolicyError::InvalidProviderPair));
    }
}
