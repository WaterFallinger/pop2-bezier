//! Read-only, exact-workload Auto recommendations. This never runs a tuner,
//! touches the GPU, or turns an external speed result into numerical permission.
use crate::acceleration::{BackendRuntimeFingerprint, ComputeBackend, FeatureSet, ModelLayoutFingerprint};
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, HashMap}, path::PathBuf, sync::{Mutex, OnceLock}};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HipKernelSourceFingerprint {
    pub kernel_name: String,
    pub source_fingerprint: String,
    /// Binds compilation flags, compiler identity and environment as well.
    pub cache_fingerprint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HipWorkloadPerformance {
    pub physical_rows: usize,
    pub guidance: usize,
    pub features: FeatureSet,
    pub baseline_attribute_tps: f64,
    pub optimized_attribute_tps: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HipPerformanceProfile {
    pub schema_version: u32,
    pub application_version: String,
    pub device_id: String,
    pub arch: String,
    pub runtime_version: String,
    pub driver_version: String,
    pub kernel_sdk_version: String,
    pub dtype: String,
    pub head_width: usize,
    pub model_layout: ModelLayoutFingerprint,
    pub compiler_version: String,
    pub kernel_sources: Vec<HipKernelSourceFingerprint>,
    pub workloads: Vec<HipWorkloadPerformance>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HipWorkloadRecommendation {
    pub features: FeatureSet,
    pub details: Vec<String>,
}

type SourceRegistry = HashMap<String, Vec<HipKernelSourceFingerprint>>;
static SOURCES: OnceLock<Mutex<SourceRegistry>> = OnceLock::new();

fn source_map(sources: &[HipKernelSourceFingerprint]) -> Option<BTreeMap<&str, (&str, &str)>> {
    let mut result = BTreeMap::new();
    if sources.is_empty() { return None; }
    for source in sources {
        if source.kernel_name.is_empty() || source.source_fingerprint.is_empty()
            || source.cache_fingerprint.is_empty()
            || result.insert(source.kernel_name.as_str(),
                (source.source_fingerprint.as_str(), source.cache_fingerprint.as_str())).is_some() {
            return None;
        }
    }
    Some(result)
}

/// Called once by the provider with its actual successfully loaded modules.
/// Later lookups use only this host-side snapshot, never Device::new_rocm.
pub(super) fn register_kernel_sources(fp: &BackendRuntimeFingerprint,
    sources: Vec<HipKernelSourceFingerprint>) -> Result<(), String> {
    let key = serde_json::to_string(fp).map_err(|error| error.to_string())?;
    let mut registry = SOURCES.get_or_init(|| Mutex::new(HashMap::new()))
        .lock().map_err(|_| "HIP source registry is unavailable".to_owned())?;
    if source_map(&sources).is_some() { registry.insert(key, sources); }
    else { registry.remove(&key); }
    Ok(())
}

fn recommend(fp: &BackendRuntimeFingerprint, physical_rows: usize, guidance: usize,
    profile: &HipPerformanceProfile, actual_sources: &[HipKernelSourceFingerprint])
    -> Result<FeatureSet, &'static str>
{
    if fp.backend != ComputeBackend::Hip || !fp.ordinary_model_verified
        || physical_rows == 0 || !matches!(guidance, 1 | 2) || physical_rows % guidance != 0 {
        return Err("当前模型或工作量不是已验证的HIP配置");
    }
    let identity = fp.device.device_id().map_err(|_| "缺少实际计算设备身份")?;
    let layout = fp.model_layout.as_ref().ok_or("本次加载缺少实际完整模型布局，不能复用吞吐实测")?;
    if layout.layers == 0 || layout.width == 0 || layout.heads == 0 || layout.ffn_hidden == 0
        || profile.model_layout != *layout {
        return Err("性能配置的层数、宽度、头数或FFN布局与本次模型不一致");
    }
    if profile.schema_version != 1 || profile.application_version != env!("CARGO_PKG_VERSION")
        || profile.device_id != identity || profile.arch != fp.device.architecture
        || profile.runtime_version != fp.runtime.runtime_version
        || profile.driver_version != fp.runtime.driver_version
        || profile.kernel_sdk_version != fp.runtime.kernel_runtime_version
        || profile.dtype != fp.dtype || profile.head_width != fp.head_width
        || fp.compiler.version.as_deref() != Some(profile.compiler_version.as_str()) {
        return Err("性能配置与本次应用、设备、运行库、编译器或模型布局不匹配");
    }
    if !fp.runtime.driver_runtime_verified || !fp.runtime.kernel_runtime_verified || !fp.compiler.usable() {
        return Err("本次运行库或编译器没有有效实证");
    }
    let Some(verified) = fp.verification.as_ref().filter(|value| value.numerical_checks_passed
        && value.device_id == identity && value.architecture == fp.device.architecture
        && value.runtime_version == fp.runtime.runtime_version && value.dtype == fp.dtype
        && value.head_width == fp.head_width) else { return Err("本次加载未通过精确数值验证"); };
    let expected = source_map(&profile.kernel_sources).ok_or("性能配置缺少唯一kernel源码指纹")?;
    let actual = source_map(actual_sources).ok_or("本次加载没有kernel源码实证")?;
    if expected != actual { return Err("kernel源码、编译选项或缓存身份已变化，旧性能配置不再适用"); }
    let mut matching = profile.workloads.iter().filter(|workload|
        workload.physical_rows == physical_rows && workload.guidance == guidance);
    let workload = matching.next().ok_or("没有此物理批量与CFG分支数的实测结果")?;
    if matching.next().is_some() { return Err("同一工作量有重复的性能配置"); }
    if !workload.baseline_attribute_tps.is_finite() || !workload.optimized_attribute_tps.is_finite()
        || workload.baseline_attribute_tps <= 0.
        || workload.optimized_attribute_tps <= workload.baseline_attribute_tps * 1.05 {
        return Err("实测加速未严格超过5%，保持普通路径");
    }
    let measured = workload.features;
    if measured.primitive_graph {
        return Err("HIP Graph尚未实现，不能采用包含Graph的整体性能成绩");
    }
    if (measured.fusion && !(fp.implemented.fusion && verified.features.fusion))
        || (measured.split_k && !(fp.implemented.split_k && verified.features.split_k))
        || (measured.gpu_sampling && !(fp.implemented.gpu_sampling && verified.features.gpu_sampling)) {
        return Err("本次无法执行完整的已测功能组合，不能把未实测子组合授予Auto");
    }
    Ok(measured)
}

pub fn workload_recommendation_report(fp: &BackendRuntimeFingerprint,
    physical_rows: usize, guidance: usize) -> HipWorkloadRecommendation {
    let result = (|| -> Result<FeatureSet, String> {
        let path = PathBuf::from(std::env::var_os("PAINTER_HIP_PERFORMANCE_PROFILE")
            .ok_or("未设置HIP实测性能配置，Auto保留普通路径")?);
        if !path.is_absolute() { return Err("HIP性能配置必须使用绝对文件路径".into()); }
        let metadata = std::fs::metadata(&path).map_err(|_| "HIP性能配置文件不可读取")?;
        if !metadata.is_file() || metadata.len() > 1_048_576 { return Err("HIP性能配置文件类型或大小无效".into()); }
        let profile: HipPerformanceProfile = serde_json::from_slice(&std::fs::read(path)
            .map_err(|_| "HIP性能配置读取失败")?).map_err(|_| "HIP性能配置JSON无效")?;
        let key = serde_json::to_string(fp).map_err(|error| error.to_string())?;
        let registry = SOURCES.get().ok_or("本次没有已准备kernel源码快照")?
            .lock().map_err(|_| "HIP源码快照不可用")?;
        let sources = registry.get(&key).ok_or("性能配置没有匹配本次加载的kernel证据")?;
        recommend(fp, physical_rows, guidance, &profile, sources).map_err(str::to_owned)
    })();
    match result {
        Ok(features) => HipWorkloadRecommendation { features,
            details: vec!["Auto采用同设备、同源码、同工作量严格超过5%的实测推荐；Graph保持关闭".into()] },
        Err(reason) => HipWorkloadRecommendation { features: FeatureSet::default(), details: vec![reason] },
    }
}

pub fn workload_recommendations(fp: &BackendRuntimeFingerprint,
    physical_rows: usize, guidance: usize) -> FeatureSet {
    workload_recommendation_report(fp, physical_rows, guidance).features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceleration::{ComputeDeviceFingerprint, GpuVendor, IdentitySource,
        KernelFeatureVerification, OptionalCompilerFingerprint, RuntimeKernelMatch};

    fn fixture() -> (BackendRuntimeFingerprint, HipPerformanceProfile) {
        let device = ComputeDeviceFingerprint { source: IdentitySource::ModelComputeRuntime,
            vendor: GpuVendor::Amd, pci_vendor_id: Some(0x1002), uuid: Some("test-gpu".into()),
            pci_bus_id: None, ordinal: Some(0), architecture: "gfx1100".into() };
        let identity = device.device_id().unwrap();
        let features = FeatureSet { fusion: true, split_k: true, gpu_sampling: true, primitive_graph: false };
        let layout = ModelLayoutFingerprint { layers: 35, width: 1536, heads: 24, ffn_hidden: 4096 };
        let fp = BackendRuntimeFingerprint { device, backend: ComputeBackend::Hip, dtype: "BF16".into(),
            model_layout: Some(layout),
            head_width: 64, ordinary_model_verified: true,
            runtime: RuntimeKernelMatch { driver_version: "driver".into(), runtime_version: "7.2".into(),
                kernel_runtime_version: "7.1".into(), driver_runtime_verified: true, kernel_runtime_verified: true },
            compiler: OptionalCompilerFingerprint::external_hipcc(std::env::current_exe().unwrap(), "hipcc-7.1".into(), true),
            implemented: features, verification: Some(KernelFeatureVerification {
                device_id: identity.clone(), architecture: "gfx1100".into(), runtime_version: "7.2".into(),
                dtype: "BF16".into(), head_width: 64, numerical_checks_passed: true,
                features, auto_recommended: FeatureSet::default(), performance_device_id: None }) };
        let profile = HipPerformanceProfile { schema_version: 1, application_version: env!("CARGO_PKG_VERSION").into(),
            device_id: identity, arch: "gfx1100".into(), runtime_version: "7.2".into(), driver_version: "driver".into(),
            kernel_sdk_version: "7.1".into(), dtype: "BF16".into(), head_width: 64, compiler_version: "hipcc-7.1".into(),
            model_layout: layout,
            kernel_sources: vec![HipKernelSourceFingerprint { kernel_name: "attention".into(),
                source_fingerprint: "source-v2".into(), cache_fingerprint: "exact-cache-identity".into() }],
            workloads: vec![HipWorkloadPerformance { physical_rows: 8, guidance: 1, features,
                baseline_attribute_tps: 100., optimized_attribute_tps: 120. }] };
        (fp, profile)
    }

    #[test]
    fn exact_workload_recommends_only_the_complete_verified_measured_bundle() {
        let (fp, profile) = fixture();
        let features = recommend(&fp, 8, 1, &profile, &profile.kernel_sources).unwrap();
        assert!(features.fusion && features.split_k && features.gpu_sampling && !features.primitive_graph);
        for feature in 0..3 {
            for missing_implementation in [false, true] {
                let mut unavailable = fp.clone();
                let flags = if missing_implementation { &mut unavailable.implemented }
                    else { &mut unavailable.verification.as_mut().unwrap().features };
                match feature { 0 => flags.fusion = false, 1 => flags.split_k = false,
                    _ => flags.gpu_sampling = false }
                assert!(recommend(&unavailable, 8, 1, &profile, &profile.kernel_sources).is_err(),
                    "missing measured component must reject the whole bundle");
            }
        }
        let mut graph_claim = profile.clone();
        graph_claim.workloads[0].features.primitive_graph = true;
        assert!(recommend(&fp, 8, 1, &graph_claim, &profile.kernel_sources).is_err());
        assert!(recommend(&fp, 1, 1, &profile, &profile.kernel_sources).is_err());
        assert!(recommend(&fp, 8, 2, &profile, &profile.kernel_sources).is_err());
    }

    #[test]
    fn profile_cannot_survive_identity_code_version_or_numeric_mismatch() {
        let (fp, original) = fixture();
        for case in 0..15 {
            let mut profile = original.clone();
            match case {
                0 => profile.device_id.push_str("other"), 1 => profile.arch.push_str("other"),
                2 => profile.runtime_version.push_str("other"), 3 => profile.driver_version.push_str("other"),
                4 => profile.kernel_sdk_version.push_str("other"), 5 => profile.dtype = "F32".into(),
                6 => profile.head_width = 128, 7 => profile.compiler_version.push_str("other"),
                8 => profile.application_version.push_str("other"),
                9 => profile.kernel_sources[0].source_fingerprint.push_str("old"),
                10 => profile.kernel_sources[0].cache_fingerprint.push_str("old"),
                11 => profile.model_layout.layers = 12,
                12 => profile.model_layout.width = 512,
                13 => profile.model_layout.heads = 8,
                _ => profile.model_layout.ffn_hidden = 1536,
            }
            assert!(recommend(&fp, 8, 1, &profile, &original.kernel_sources).is_err(), "case {case}");
        }
        let mut missing_layout = fp.clone();
        missing_layout.model_layout = None;
        assert!(recommend(&missing_layout, 8, 1, &original, &original.kernel_sources).is_err());
        for field in 0..4 {
            let mut zero_layout = fp.clone();
            let layout = zero_layout.model_layout.as_mut().unwrap();
            match field { 0 => layout.layers = 0, 1 => layout.width = 0,
                2 => layout.heads = 0, _ => layout.ffn_hidden = 0 }
            let mut zero_profile = original.clone();
            zero_profile.model_layout = *layout;
            assert!(recommend(&zero_layout, 8, 1, &zero_profile, &original.kernel_sources).is_err());
        }
        let mut unverified = fp;
        unverified.verification.as_mut().unwrap().numerical_checks_passed = false;
        assert!(recommend(&unverified, 8, 1, &original, &original.kernel_sources).is_err());
    }

    #[test]
    fn threshold_is_strict_and_ambiguous_or_invalid_measurements_are_rejected() {
        let (fp, mut profile) = fixture();
        for optimized in [105., 100., 1., f64::INFINITY, f64::NAN] {
            profile.workloads[0].optimized_attribute_tps = optimized;
            assert!(recommend(&fp, 8, 1, &profile, &profile.kernel_sources).is_err());
        }
        profile.workloads[0].optimized_attribute_tps = 105.01;
        assert!(recommend(&fp, 8, 1, &profile, &profile.kernel_sources).is_ok());
        profile.workloads.push(profile.workloads[0].clone());
        assert!(recommend(&fp, 8, 1, &profile, &profile.kernel_sources).is_err());
        assert!(source_map(&[]).is_none());
        let duplicate = vec![profile.kernel_sources[0].clone(), profile.kernel_sources[0].clone()];
        assert!(source_map(&duplicate).is_none());
    }
}
