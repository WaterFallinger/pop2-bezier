//! Evidence-based model-provider identification. These APIs perform no GPU
//! initialization, driver loading, filesystem mutation or device fallback.
//! The provider supplies observations from its actual compute context.

use super::{BackendCapabilities, ComputeBackend, FeatureSet, GpuVendor};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentitySource {
    ModelComputeRuntime,
    /// Kept explicit so accidental use of a renderer observation is rejected.
    RenderingAdapter,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeDeviceFingerprint {
    pub source: IdentitySource,
    pub vendor: GpuVendor,
    pub pci_vendor_id: Option<u32>,
    pub uuid: Option<String>,
    pub pci_bus_id: Option<String>,
    pub ordinal: Option<usize>,
    /// Exact compute target queried by the provider, e.g. gfx1100 or sm_120.
    pub architecture: String,
}

impl ComputeDeviceFingerprint {
    pub fn device_id(&self) -> Result<String, String> {
        if self.source != IdentitySource::ModelComputeRuntime {
            return Err("渲染适配器不能替代模型实际计算设备".into());
        }
        if let Some(pci) = self.pci_vendor_id {
            if GpuVendor::from_pci_vendor_id(pci) != self.vendor {
                return Err("计算设备PCI厂商与运行时厂商不一致".into());
            }
        }
        if self.architecture.trim().is_empty() {
            return Err("计算设备缺少实际架构标识".into());
        }
        if self.vendor == GpuVendor::Cpu {
            if self.pci_vendor_id.is_some() || self.uuid.is_some() || self.pci_bus_id.is_some() {
                return Err("CPU执行不能冒用GPU身份".into());
            }
            return Ok("cpu:local".into());
        }
        if let Some(uuid) = self.uuid.as_deref().filter(|value| !value.trim().is_empty()) {
            return Ok(format!("uuid:{}", uuid.trim().to_ascii_lowercase()));
        }
        if let Some(bus) = self.pci_bus_id.as_deref().filter(|value| !value.trim().is_empty()) {
            return Ok(format!("pci:{}", bus.trim().to_ascii_lowercase()));
        }
        self.ordinal.map(|ordinal| format!("ordinal:{ordinal}:{}", self.architecture))
            .ok_or_else(|| "计算设备缺少UUID、PCI位置或实际运行时ordinal".into())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeKernelMatch {
    pub driver_version: String,
    pub runtime_version: String,
    pub kernel_runtime_version: String,
    /// Compatibility must be checked by the provider, not inferred from a DLL name.
    pub driver_runtime_verified: bool,
    pub kernel_runtime_verified: bool,
}

impl RuntimeKernelMatch {
    fn verified(&self) -> bool {
        !self.driver_version.trim().is_empty() && !self.runtime_version.trim().is_empty()
            && !self.kernel_runtime_version.trim().is_empty()
            && self.driver_runtime_verified && self.kernel_runtime_verified
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionalCompilerFingerprint {
    pub kind: Option<OptionalCompilerKind>,
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    pub library_loaded: bool,
    pub required_symbols_verified: bool,
    pub compile_probe_passed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionalCompilerKind { Nvrtc, Hiprtc, ExternalHipcc }

impl OptionalCompilerFingerprint {
    pub fn usable(&self) -> bool {
        let verified = match self.kind {
            Some(OptionalCompilerKind::Nvrtc | OptionalCompilerKind::Hiprtc) =>
                self.library_loaded && self.required_symbols_verified,
            Some(OptionalCompilerKind::ExternalHipcc) => self.compile_probe_passed,
            None => false,
        };
        verified && self.version.as_deref().is_some_and(|version| !version.trim().is_empty())
            && self.path.as_ref().is_some_and(|path| path.is_absolute() && path.is_file())
    }

    pub fn external_hipcc(path: PathBuf, version: String, compile_probe_passed: bool) -> Self {
        Self { kind: Some(OptionalCompilerKind::ExternalHipcc), path: Some(path), version: Some(version),
            compile_probe_passed, ..Default::default() }
    }
}

/// A validation result is bound to an exact target/runtime/model layout. A
/// numerical check on one gfx target cannot authorize every AMD GPU.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelFeatureVerification {
    pub device_id: String,
    pub architecture: String,
    pub runtime_version: String,
    pub dtype: String,
    pub head_width: usize,
    pub numerical_checks_passed: bool,
    pub features: FeatureSet,
    pub auto_recommended: FeatureSet,
    /// Performance evidence is more restrictive than numerical compatibility.
    /// None means no device-specific Auto recommendation has been established.
    pub performance_device_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendRuntimeFingerprint {
    pub device: ComputeDeviceFingerprint,
    pub backend: ComputeBackend,
    pub runtime: RuntimeKernelMatch,
    pub dtype: String,
    pub head_width: usize,
    /// Kernel numerical compatibility may be model-independent. Performance
    /// recommendations require this complete actual model layout.
    #[serde(default)]
    pub model_layout: Option<ModelLayoutFingerprint>,
    /// Actual ordinary model math has run on this selected provider/device.
    pub ordinary_model_verified: bool,
    pub compiler: OptionalCompilerFingerprint,
    /// Features implemented in this build, never a claim based on DLL discovery.
    pub implemented: FeatureSet,
    pub verification: Option<KernelFeatureVerification>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLayoutFingerprint {
    pub layers: usize,
    pub width: usize,
    pub heads: usize,
    pub ffn_hidden: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityDetection {
    pub capabilities: BackendCapabilities,
    pub details: Vec<String>,
}

fn pair_valid(vendor: GpuVendor, backend: ComputeBackend) -> bool {
    match backend {
        ComputeBackend::Cuda => vendor == GpuVendor::Nvidia,
        ComputeBackend::Hip | ComputeBackend::AtenHip => vendor == GpuVendor::Amd,
        ComputeBackend::Vulkan => vendor != GpuVendor::Cpu,
        ComputeBackend::Cpu => vendor == GpuVendor::Cpu,
    }
}

fn intersect(left: FeatureSet, right: FeatureSet) -> FeatureSet {
    FeatureSet { gpu_sampling: left.gpu_sampling && right.gpu_sampling,
        fusion: left.fusion && right.fusion, split_k: left.split_k && right.split_k,
        primitive_graph: left.primitive_graph && right.primitive_graph }
}

pub fn capabilities_from_fingerprint(fingerprint: &BackendRuntimeFingerprint)
    -> Result<CapabilityDetection, String>
{
    validate_loaded_model_device(fingerprint)?;
    let identity = fingerprint.device.device_id()?;
    // A software/target change also needs a new optional-failure session.
    let device_key = serde_json::to_string(&(fingerprint.backend, &identity,
        &fingerprint.device.architecture, &fingerprint.runtime.driver_version,
        &fingerprint.runtime.runtime_version, &fingerprint.runtime.kernel_runtime_version,
        &fingerprint.dtype, fingerprint.head_width, &fingerprint.model_layout)).map_err(|error| error.to_string())?;
    let mut capabilities = BackendCapabilities::baseline(fingerprint.device.vendor,
        fingerprint.backend, device_key);
    let mut details = vec![format!("模型计算设备：{} / {} / {}",
        fingerprint.backend.label(), identity, fingerprint.device.architecture)];
    if fingerprint.backend == ComputeBackend::Cpu {
        details.push("明确选择CPU普通执行，不授予GPU加速能力".into());
        return Ok(CapabilityDetection { capabilities, details });
    }
    if !fingerprint.runtime.verified() {
        details.push("driver/runtime/kernel配套关系未核验；仅保留已验证的同设备普通模型路径".into());
        return Ok(CapabilityDetection { capabilities, details });
    }
    let compiler_matches_backend = matches!((fingerprint.backend, fingerprint.compiler.kind),
        (ComputeBackend::Cuda, Some(OptionalCompilerKind::Nvrtc))
            | (ComputeBackend::Hip, Some(OptionalCompilerKind::Hiprtc | OptionalCompilerKind::ExternalHipcc)));
    if !compiler_matches_backend || !fingerprint.compiler.usable() {
        details.push("可选编译器的文件、版本或执行实证不完整；跳过可选JIT".into());
        return Ok(CapabilityDetection { capabilities, details });
    }
    let matched = fingerprint.verification.as_ref().filter(|verification| {
        verification.numerical_checks_passed
            && verification.device_id == identity
            && verification.architecture == fingerprint.device.architecture
            && verification.runtime_version == fingerprint.runtime.runtime_version
            && verification.dtype == fingerprint.dtype
            && verification.head_width == fingerprint.head_width
    });
    let Some(verification) = matched else {
        details.push("没有精确匹配当前架构、runtime、dtype和头宽的数值验证记录；可选功能保持关闭".into());
        return Ok(CapabilityDetection { capabilities, details });
    };
    capabilities.available = intersect(fingerprint.implemented, verification.features);
    if fingerprint.backend == ComputeBackend::Hip {
        // HIP Graph has no implementation in this application yet, regardless
        // of driver symbols or a future/foreign validation record.
        capabilities.available.primitive_graph = false;
        details.push("HIP Graph尚未实现，保持关闭；不复用NVIDIA Graph路径".into());
    }
    if verification.performance_device_id.as_deref() == Some(identity.as_str()) {
        capabilities.auto_recommended = intersect(capabilities.available, verification.auto_recommended);
    } else {
        details.push("当前设备没有绑定的性能实证，Auto不套用其他显卡的加速收益结论".into());
    }
    details.push("可选能力来自已实现且精确匹配的实证记录，准备失败仍由本设备会话独立降级".into());
    Ok(CapabilityDetection { capabilities, details })
}

/// Additive validation for an already established provider policy. In
/// particular, callers can preserve the current CUDA availability/Auto policy
/// while still rejecting renderer identities and mismatched selected devices.
pub fn validate_loaded_model_device(fingerprint: &BackendRuntimeFingerprint) -> Result<(), String> {
    fingerprint.device.device_id()?;
    if !pair_valid(fingerprint.device.vendor, fingerprint.backend) {
        return Err("模型实际计算设备与选择的执行后端不匹配；不会切换到其他GPU或CPU".into());
    }
    if !fingerprint.ordinary_model_verified {
        return Err("实际模型普通计算尚未验证，不能仅凭运行库宣告后端可用".into());
    }
    if fingerprint.dtype.trim().is_empty() || fingerprint.head_width == 0 {
        return Err("模型缺少实际dtype或注意力头宽度".into());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDeviceSelection {
    pub backend: ComputeBackend,
    /// The provider-derived value returned by ComputeDeviceFingerprint::device_id.
    pub device_id: String,
}

/// Enumerated compute contexts, not rendering adapters, are the candidates.
/// Mixed/multiple devices require an explicit choice; ordinal zero of a
/// different provider must never be substituted when the requested one is absent.
pub fn select_model_device<'a>(candidates: &'a [BackendRuntimeFingerprint],
    selection: Option<&ModelDeviceSelection>) -> Result<&'a BackendRuntimeFingerprint, String>
{
    let mut matches = Vec::new();
    for candidate in candidates {
        if !candidate.ordinary_model_verified { continue; }
        let id = candidate.device.device_id()?;
        if !pair_valid(candidate.device.vendor, candidate.backend) {
            return Err("计算设备候选的厂商与执行后端不匹配".into());
        }
        if selection.is_none_or(|selected| selected.backend == candidate.backend && selected.device_id == id) {
            matches.push(candidate);
        }
    }
    if selection.is_none() && matches.iter().any(|candidate| candidate.backend != ComputeBackend::Cpu) {
        matches.retain(|candidate| candidate.backend != ComputeBackend::Cpu);
    }
    match matches.as_slice() {
        [candidate] => Ok(*candidate),
        [] => Err("找不到所选模型计算设备；不会改用渲染显卡、另一GPU或CPU".into()),
        _ => Err("存在多个实际计算设备，请明确选择后端与设备，不能根据渲染适配器猜测".into()),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendArtifactFlavor {
    pub id: String,
    pub backend: ComputeBackend,
    pub runtime_root: PathBuf,
}

/// Pure validation for future side-by-side builds. No DLL or installation file
/// is copied/replaced, and CUDA/HIP artifact directories cannot overlap.
pub fn validate_side_by_side_flavors(flavors: &[BackendArtifactFlavor]) -> Result<(), String> {
    let mut roots: Vec<(String, String)> = Vec::new();
    for flavor in flavors {
        if flavor.id.trim().is_empty() || !flavor.runtime_root.is_absolute()
            || flavor.runtime_root.components().any(|part| matches!(part, std::path::Component::ParentDir)) {
            return Err("后端产物需要独立名称和无上级跳转的绝对运行库目录".into());
        }
        // Windows paths are case-insensitive. A trailing slash must not hide an overlap.
        let root = flavor.runtime_root.to_string_lossy().replace('\\', "/")
            .trim_end_matches('/').to_ascii_lowercase();
        for (previous_id, previous_root) in &roots {
            if previous_id == &flavor.id || root == *previous_root
                || root.starts_with(&format!("{previous_root}/"))
                || previous_root.starts_with(&format!("{root}/")) {
                return Err("后端产物名称或目录重叠，不能用HIP运行库替换NVIDIA产物".into());
            }
        }
        roots.push((flavor.id.clone(), root));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceleration::{resolve, AccelerationMode, AccelerationSettings, ExecutionPath,
        FailureClass, FeaturePreference, OptimizationFeature, SamplingPreference, SessionFailures};

    fn all_features() -> FeatureSet {
        FeatureSet { fusion: true, split_k: true, gpu_sampling: true, primitive_graph: true }
    }

    fn fingerprint(vendor: GpuVendor, backend: ComputeBackend, architecture: &str) -> BackendRuntimeFingerprint {
        let device = ComputeDeviceFingerprint {
            source: IdentitySource::ModelComputeRuntime, vendor,
            pci_vendor_id: match vendor { GpuVendor::Nvidia => Some(0x10de),
                GpuVendor::Amd => Some(0x1002), GpuVendor::Intel => Some(0x8086), _ => None },
            uuid: (vendor != GpuVendor::Cpu).then(|| format!("test-{vendor:?}-device0")), pci_bus_id: None,
            ordinal: Some(0), architecture: architecture.into(),
        };
        let identity = device.device_id().unwrap();
        BackendRuntimeFingerprint {
            device, backend, dtype: "BF16".into(), head_width: 64, model_layout: None,
            ordinary_model_verified: true,
            runtime: RuntimeKernelMatch { driver_version: "driver-verified".into(),
                runtime_version: "runtime-7.2".into(), kernel_runtime_version: "sdk-7.1".into(),
                driver_runtime_verified: true, kernel_runtime_verified: true },
            // This is a pure evidence fixture. It does not load this executable
            // as a DLL or run any compiler/GPU in the test.
            compiler: OptionalCompilerFingerprint { kind: Some(if backend == ComputeBackend::Hip {
                    OptionalCompilerKind::Hiprtc } else { OptionalCompilerKind::Nvrtc }),
                path: Some(std::env::current_exe().unwrap()), version: Some("compiler-verified".into()),
                library_loaded: true, required_symbols_verified: true, compile_probe_passed: false },
            implemented: all_features(),
            verification: Some(KernelFeatureVerification { device_id: identity.clone(),
                architecture: architecture.into(), runtime_version: "runtime-7.2".into(),
                dtype: "BF16".into(), head_width: 64, numerical_checks_passed: true,
                features: all_features(), auto_recommended: FeatureSet {
                    fusion: true, primitive_graph: true, ..Default::default() },
                performance_device_id: Some(identity) }),
        }
    }

    #[test]
    fn validated_cuda_preserves_available_features_and_existing_auto_recommendations() {
        let fp = fingerprint(GpuVendor::Nvidia, ComputeBackend::Cuda, "sm_120");
        validate_loaded_model_device(&fp).unwrap();
        let caps = capabilities_from_fingerprint(&fp).unwrap().capabilities;
        assert_eq!(caps.available, all_features());
        assert!(caps.auto_recommended.fusion && caps.auto_recommended.primitive_graph);
        assert!(!caps.auto_recommended.split_k);
        let plan = resolve(&AccelerationSettings::default(), &caps, SamplingPreference::Auto,
            &SessionFailures::new(&caps)).unwrap();
        assert!(plan.features.fusion && plan.features.primitive_graph && plan.features.gpu_sampling);
        assert!(!plan.features.split_k);
    }

    #[test]
    fn hipcc_verified_kernels_do_not_require_hiprtc_and_never_enable_hip_graph() {
        let mut fp = fingerprint(GpuVendor::Amd, ComputeBackend::Hip, "gfx1100");
        fp.compiler = OptionalCompilerFingerprint::external_hipcc(std::env::current_exe().unwrap(),
            "hipcc-7.1".into(), true);
        assert!(!fp.compiler.library_loaded && !fp.compiler.required_symbols_verified);
        fp.implemented.split_k = false;
        fp.verification.as_mut().unwrap().performance_device_id = None;
        let caps = capabilities_from_fingerprint(&fp).unwrap().capabilities;
        assert!(caps.available.fusion && caps.available.gpu_sampling);
        assert!(!caps.available.split_k && !caps.available.primitive_graph);
        assert_eq!(caps.auto_recommended, FeatureSet::default());
        let automatic = resolve(&AccelerationSettings::default(), &caps, SamplingPreference::Auto,
            &SessionFailures::new(&caps)).unwrap();
        assert_eq!(automatic.path, ExecutionPath::OrdinaryGpu);
        let requested = resolve(&AccelerationSettings { fusion: FeaturePreference::Enabled,
            primitive_graph: FeaturePreference::Enabled, ..Default::default() }, &caps,
            SamplingPreference::Gpu, &SessionFailures::new(&caps)).unwrap();
        assert!(requested.features.fusion && requested.features.gpu_sampling);
        assert!(!requested.features.primitive_graph);
        // Even a provider bypassing this factory cannot accidentally authorize
        // the absent HIP Graph implementation through the generic policy.
        let mut mistaken = caps.clone();
        mistaken.available.primitive_graph = true;
        let guarded = resolve(&AccelerationSettings { mode: AccelerationMode::Optimized,
            ..Default::default() }, &mistaken, SamplingPreference::Gpu,
            &SessionFailures::new(&mistaken)).unwrap();
        assert!(!guarded.features.primitive_graph);
    }

    #[test]
    fn verification_is_bound_to_actual_device_target_runtime_dtype_and_layout() {
        for change in 0..8 {
            let mut fp = fingerprint(GpuVendor::Amd, ComputeBackend::Hip, "gfx1100");
            match change {
                0 => fp.device.architecture = "unknown-gfx".into(),
                1 => fp.runtime.driver_runtime_verified = false,
                2 => fp.runtime.kernel_runtime_verified = false,
                3 => fp.verification.as_mut().unwrap().runtime_version = "another-runtime".into(),
                4 => fp.head_width = 128,
                5 => fp.device.uuid = Some("another-same-gfx-device".into()),
                6 => fp.verification.as_mut().unwrap().numerical_checks_passed = false,
                _ => fp.dtype = "F32".into(),
            }
            let result = capabilities_from_fingerprint(&fp).unwrap();
            assert!(result.capabilities.ordinary_compute_available);
            assert_eq!(result.capabilities.backend, ComputeBackend::Hip);
            assert_eq!(result.capabilities.available, FeatureSet::default(), "case {change}");
            assert!(!result.details.is_empty());
        }
        // Different SDK/runtime minor strings are allowed after actual
        // compatibility and numerical validation; no string-equality shortcut.
        let compatible = fingerprint(GpuVendor::Amd, ComputeBackend::Hip, "gfx1100");
        assert_ne!(compatible.runtime.kernel_runtime_version, compatible.runtime.runtime_version);
        assert!(capabilities_from_fingerprint(&compatible).unwrap().capabilities.available.fusion);
    }

    #[test]
    fn compiler_library_presence_is_insufficient_without_api_version_or_external_probe() {
        for change in 0..6 {
            let mut fp = fingerprint(GpuVendor::Amd, ComputeBackend::Hip, "gfx1100");
            match change {
                0 => fp.compiler.required_symbols_verified = false,
                1 => fp.compiler.version = None,
                2 => fp.compiler.path = Some("relative/hiprtc.dll".into()),
                3 => fp.compiler.kind = None,
                4 => fp.compiler = OptionalCompilerFingerprint::external_hipcc(
                    std::env::current_exe().unwrap(), "hipcc-7.1".into(), false),
                _ => fp.compiler.kind = Some(OptionalCompilerKind::Nvrtc),
            }
            assert_eq!(capabilities_from_fingerprint(&fp).unwrap().capabilities.available,
                FeatureSet::default());
        }
    }

    #[test]
    fn sole_runnable_gpu_is_selected_but_two_vendor_models_require_explicit_choice() {
        let gpu = fingerprint(GpuVendor::Nvidia, ComputeBackend::Cuda, "sm_120");
        let mut integrated = fingerprint(GpuVendor::Intel, ComputeBackend::Vulkan, "integrated-target");
        integrated.ordinary_model_verified = false;
        let candidates = vec![integrated, gpu.clone()];
        assert_eq!(select_model_device(&candidates, None).unwrap().backend, ComputeBackend::Cuda);
        let amd = fingerprint(GpuVendor::Amd, ComputeBackend::Hip, "gfx1100");
        let mixed = vec![gpu, amd.clone()];
        assert!(select_model_device(&mixed, None).is_err());
        let selection = ModelDeviceSelection { backend: ComputeBackend::Hip,
            device_id: amd.device.device_id().unwrap() };
        assert_eq!(select_model_device(&mixed, Some(&selection)).unwrap(), &amd);
        let absent = ModelDeviceSelection { backend: ComputeBackend::Hip, device_id: "uuid:absent".into() };
        assert!(select_model_device(&mixed, Some(&absent)).is_err());
    }

    #[test]
    fn renderer_pci_mismatch_cpu_impersonation_and_unverified_model_are_rejected() {
        for change in 0..4 {
            let mut fp = fingerprint(GpuVendor::Nvidia, ComputeBackend::Cuda, "sm_120");
            match change {
                0 => fp.device.source = IdentitySource::RenderingAdapter,
                1 => fp.device.pci_vendor_id = Some(0x1002),
                2 => fp.backend = ComputeBackend::Cpu,
                _ => fp.ordinary_model_verified = false,
            }
            assert!(capabilities_from_fingerprint(&fp).is_err());
        }
        let mut cpu = fingerprint(GpuVendor::Cpu, ComputeBackend::Cpu, "cpu");
        cpu.device.uuid = None;
        cpu.device.ordinal = None;
        let caps = capabilities_from_fingerprint(&cpu).unwrap().capabilities;
        assert_eq!(caps.available, FeatureSet::default());
        let cuda = fingerprint(GpuVendor::Nvidia, ComputeBackend::Cuda, "sm_120");
        let mixed = vec![cpu, cuda];
        assert_eq!(select_model_device(&mixed, None).unwrap().backend, ComputeBackend::Cuda);
    }

    #[test]
    fn feature_failures_remain_local_to_the_same_provider_device_and_software() {
        let fp = fingerprint(GpuVendor::Amd, ComputeBackend::Hip, "gfx1100");
        let caps = capabilities_from_fingerprint(&fp).unwrap().capabilities;
        let mut failures = SessionFailures::new(&caps);
        failures.record(OptimizationFeature::Fusion, FailureClass::Preparation, "optional compile failed");
        let ordinary = resolve(&AccelerationSettings::default(), &caps, SamplingPreference::Auto,
            &failures).unwrap();
        assert!(!ordinary.features.fusion);
        assert_eq!(ordinary.backend, ComputeBackend::Hip);
        let mut another = fp.clone();
        another.runtime.runtime_version = "new-session-runtime".into();
        let other = capabilities_from_fingerprint(&another).unwrap().capabilities;
        assert!(resolve(&AccelerationSettings::default(), &other, SamplingPreference::Auto, &failures).is_err());
        failures.record(OptimizationFeature::Fusion, FailureClass::ExecutionOrDevice, "device state uncertain");
        assert!(resolve(&AccelerationSettings { mode: AccelerationMode::Compatible, ..Default::default() },
            &caps, SamplingPreference::Auto, &failures).is_err());
    }

    #[test]
    fn side_by_side_flavors_never_share_or_nest_runtime_directories() {
        let root = std::env::temp_dir().join("painter-flavor-policy");
        let cuda = BackendArtifactFlavor { id: "cuda".into(), backend: ComputeBackend::Cuda,
            runtime_root: root.join("cuda") };
        let hip = BackendArtifactFlavor { id: "hip".into(), backend: ComputeBackend::Hip,
            runtime_root: root.join("hip") };
        validate_side_by_side_flavors(&[cuda.clone(), hip.clone()]).unwrap();
        let overlap = BackendArtifactFlavor { runtime_root: cuda.runtime_root.join("hip"), ..hip.clone() };
        assert!(validate_side_by_side_flavors(&[cuda.clone(), overlap]).is_err());
        let relative = BackendArtifactFlavor { runtime_root: "runtime/hip".into(), ..hip };
        assert!(validate_side_by_side_flavors(&[cuda, relative]).is_err());
    }
}
