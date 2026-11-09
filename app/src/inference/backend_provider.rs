//! Explicit HIP provider factory. CUDA loading remains a separate build/path.
//! No renderer adapter or external benchmark report authorizes these features:
//! every load checks the selected GPU and actual model before returning it.

#[path = "hip_performance_profile.rs"]
mod hip_performance_profile;
pub use hip_performance_profile::{HipKernelSourceFingerprint, HipPerformanceProfile,
    HipWorkloadPerformance, HipWorkloadRecommendation, workload_recommendations,
    workload_recommendation_report};

#[cfg(feature = "rocm")]
#[path = "hip_sampling_reference.rs"]
mod sampler_reference;

#[cfg(feature = "rocm")]
mod rocm {
    use super::super::*;
    use super::sampler_reference;
    use super::hip_performance_profile::{register_kernel_sources, HipKernelSourceFingerprint};
    use crate::acceleration::{BackendCapabilities, BackendRuntimeFingerprint, ComputeBackend,
        ComputeDeviceFingerprint, FailureClass, FeatureSet, GpuVendor, IdentitySource,
        KernelFeatureVerification, ModelLayoutFingerprint, OptionalCompilerFingerprint, OptimizationFeature,
        RuntimeKernelMatch, SessionFailures, AccelerationMode, FeaturePreference,
        capabilities_from_fingerprint};
    use anyhow::ensure;
    use candle_core::{DType, Device, Tensor};
    use candle_core::rocm_backend::{hip_sys::hip_runtime as sys, strict};

    #[link(name = "amdhip64")]
    extern "C" {
        fn hipDeviceGetPCIBusId(output: *mut std::ffi::c_char, length: i32, device: i32) -> i32;
    }

    fn selected_pci_bus(ordinal: usize) -> Result<String> {
        let mut buffer = [0 as std::ffi::c_char; 64];
        ensure!(unsafe { hipDeviceGetPCIBusId(buffer.as_mut_ptr(), buffer.len() as i32,
            ordinal.try_into().context("HIP ordinal超出运行时范围")?) } == 0,
            "无法读取所选HIP计算设备的真实PCI身份");
        let bus = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }.to_str()?.trim();
        ensure!(!bus.is_empty() && bus.chars().all(|ch| ch.is_ascii_hexdigit() || ch == ':' || ch == '.'),
            "HIP运行时返回无效PCI位置");
        Ok(bus.to_owned())
    }

    fn check_ordinary_math(device: &Device) -> Result<()> {
        let a = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), device)?;
        let b = Tensor::from_vec(vec![5f32, 6., 7., 8.], (2, 2), device)?;
        for dtype in [DType::F32, DType::BF16] {
            let result = {
                let _guard = strict::DeviceOnlyGuard::enter();
                a.to_dtype(dtype)?.matmul(&b.to_dtype(dtype)?)?
            };
            ensure!(result.device().is_rocm(), "HIP ordinary math left the selected GPU");
            let values = result.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            ensure!(values == vec![19., 22., 43., 50.],
                "HIP ordinary {dtype:?} matrix numerical check failed: {values:?}");
        }
        Ok(())
    }

    fn compare_model(reference: &ForwardProbe, actual: &ForwardProbe) -> Result<(f64, f64)> {
        ensure!(reference.text_ids == actual.text_ids
            && reference.attributes.len() == actual.attributes.len()
            && !reference.attributes.is_empty(), "model probe layout mismatch");
        let mut maximum = 0f64;
        let mut error = 0f64;
        let mut energy = 0f64;
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        for (&expected, &value) in reference.attributes.iter().chain(&reference.termination)
            .zip(actual.attributes.iter().chain(&actual.termination)) {
            ensure!(expected.is_finite() && value.is_finite(), "model probe returned non-finite logits");
            let difference = value as f64 - expected as f64;
            maximum = maximum.max(difference.abs());
            error += difference * difference;
            energy += (expected as f64).powi(2);
            low = low.min(expected);
            high = high.max(expected);
        }
        ensure!(high - low > 1e-5, "model probe logits are unexpectedly constant");
        let relative_rms = (error / energy.max(1e-12)).sqrt();
        // Compare native BF16 GEMMs with two legitimate batching schedules.
        // This is a correctness guard, not a speed/quality benchmark.
        ensure!(maximum <= 0.15 && relative_rms <= 0.02,
            "OPTIONAL_NUMERICAL_MISMATCH: BF16 model max_error={maximum:.6}, relative_rms={relative_rms:.6}");
        Ok((maximum, relative_rms))
    }

    fn probe_prefix() -> Vec<InferencePrimitive> {
        vec![
            InferencePrimitive { tokens: [511, 1535, 2048, 4365, 6413, 6414, 6702, 6830, 6958] },
            InferencePrimitive { tokens: [512, 1536, 2048, 2518, 4566, 6415, 6702, 6830, 6958] },
        ]
    }

    fn check_sampler(device: &Device) -> Result<()> {
        let batch = 8usize;
        let choices = 128usize;
        let guidance = 2usize;
        for uniform_distribution in [true, false] {
            let mut raw = vec![0f32; batch * guidance * choices];
            if !uniform_distribution {
                for row in 0..batch {
                    for choice in 0..choices {
                        raw[(row * 2) * choices + choice] = match choice {
                            0 => 2., 1 => 1., 2 => 0., _ => -8.,
                        };
                    }
                }
            }
            let uniforms: Vec<_> = (0..batch).map(|index| (index as f32 + 0.25) / batch as f32).collect();
            let bounds: Vec<_> = (0..batch).map(|index| if index % 2 == 0 {
                [0u32, choices as u32] } else { [17u32, 23u32] }).collect();
            let expected = sampler_reference::reference(&raw, batch * guidance, choices,
                &uniforms, &bounds, 3.5, 2.5, guidance, 0).map_err(anyhow::Error::msg)?;
            let logits = Tensor::from_vec(raw, (batch * guidance, choices), device)?;
            let uniforms = Tensor::from_vec(uniforms, (batch, 1), device)?;
            let bounds = Tensor::from_vec(bounds.iter().flatten().copied().collect::<Vec<_>>(),
                (batch, 2), device)?;
            let (tokens, valid, support) = {
                let _guard = strict::DeviceOnlyGuard::enter();
                hip_sampling::sample(&logits, &uniforms, &bounds, 3.5, 2.5, guidance, 0)?
            };
            ensure!(tokens.device().is_rocm() && valid.device().is_rocm() && support.device().is_rocm(),
                "HIP sampling left the actual model GPU");
            let tokens = tokens.to_vec2::<u32>()?;
            let valid = valid.to_vec2::<u32>()?;
            let support = support.to_vec2::<u32>()?;
            for row in 0..batch {
                ensure!(tokens[row][0] == expected[row].token && valid[row][0] == expected[row].valid
                    && support[row][0] == expected[row].support,
                    "OPTIONAL_NUMERICAL_MISMATCH: HIP sampler row {row} disagrees with FP32 reference");
            }
        }
        Ok(())
    }

    fn record_optional_result(result: Result<()>, feature: OptimizationFeature,
        failures: &mut Vec<(OptimizationFeature, String)>, progress: &mut impl FnMut(String)) -> Result<bool> {
        match result {
            Ok(()) => Ok(true),
            Err(error) => {
                let reason = format!("{error:#}");
                if uncertain_device_execution(&reason) {
                    bail!("HIP设备执行状态不确定，停止加载并保留已有画布：{reason}");
                }
                progress(format!("HIP优化准备：{}未通过，保留同设备普通路径：{reason}", feature.label()));
                failures.push((feature, reason));
                Ok(false)
            }
        }
    }

    fn preparation_permissions(settings: &LoadPreparationSettings) -> (bool, bool, bool) {
        if settings.acceleration.mode == AccelerationMode::Compatible {
            return (false, false, false);
        }
        (settings.acceleration.fusion != FeaturePreference::Disabled,
            settings.acceleration.split_k != FeaturePreference::Disabled,
            settings.sampling_override != Some(false))
    }

    fn default_preparation_settings() -> LoadPreparationSettings {
        LoadPreparationSettings { lanes: 8, cfg_scale: 1., renyi_alpha: 4.,
            sampling_override: None, acceleration: Default::default() }
    }

    /// The ordinal belongs to the HIP model runtime, never the Wgpu adapter.
    pub fn load_rocm_at_with_preparation(config: &EngineConfig, ordinal: usize,
        settings: &LoadPreparationSettings,
        mut progress: impl FnMut(String)) -> Result<NativeEngine> {
        let (allow_fusion, allow_split, allow_sampling) = preparation_permissions(settings);
        progress(format!("初始化明确选择的HIP模型计算设备 {ordinal}"));
        let device = Device::new_rocm(ordinal).context("HIP模型设备不可用；不会切换另一GPU或CPU")?;
        let rocm = device.as_rocm_device()?;
        let pci_bus_id = selected_pci_bus(rocm.ordinal())?;
        let info = rocm.runtime_info();
        let architecture = info.arch.clone().context("HIP模型设备没有实际gfx标识")?;
        ensure!(info.kernels_match == Some(true), "普通HIP模型kernels与当前gfx未匹配，拒绝CPU替代");
        let (mut driver, mut runtime) = (0i32, 0i32);
        ensure!(unsafe { sys::hipDriverGetVersion(&mut driver) } == 0
            && unsafe { sys::hipRuntimeGetVersion(&mut runtime) } == 0,
            "HIP真实driver/runtime版本查询失败");
        progress(format!("HIP实际运行环境：{info}"));
        let before = strict::counters();
        check_ordinary_math(&device).context("实际HIP普通数学验证失败")?;
        let initial_caps = BackendCapabilities::baseline(GpuVendor::Amd, ComputeBackend::Hip,
            format!("hip:loading:ordinal:{ordinal}:{architecture}:runtime:{runtime}"));
        let mut engine = NativeEngine::load_on_device(config, device.clone(), DType::BF16,
            initial_caps, &mut progress)?;
        let prefix = probe_prefix();
        progress("HIP普通模型验证：正在检查实际完整权重的两种前向计算路径…".into());
        let ordinary = engine.forward_probe("金发蓝眼", &prefix)?;
        let sequential = engine.forward_probe_sequential("金发蓝眼", &prefix)?;
        let (error, rms) = compare_model(&ordinary, &sequential).context("HIP普通模型前向验证失败")?;
        progress(format!("HIP普通模型验证完成：max_error={error:.6}, relative_rms={rms:.6}"));

        let mut failures = Vec::new();
        let mut verified = FeatureSet::default();
        // Respect ordinary compatibility before even querying a toolchain path:
        // toolchain discovery itself may invoke hipconfig on some SDK layouts.
        let optional_requested = allow_sampling || (engine.model.architecture.head_width == 64
            && (allow_fusion || allow_split));
        let compiler_present = optional_requested
            && candle_core::rocm_backend::hip_runtime::toolchain::hipcc_path()
                .is_some_and(|path| path.is_absolute() && path.is_file());
        if allow_fusion && compiler_present && engine.model.architecture.head_width == 64 {
            progress("HIP优化准备：正在编译或复用融合注意力，并检查本次权重前向…".into());
            let fusion = prepare_optional(|| {
                engine.model.prepare_optional_kernels(true, false)?;
                let actual = engine.forward_probe_sequential("金发蓝眼", &prefix)?;
                compare_model(&sequential, &actual)?;
                Ok(())
            });
            verified.fusion = record_optional_result(fusion, OptimizationFeature::Fusion,
                &mut failures, &mut progress)?;
            engine.model.prepare_optional_kernels(false, false)?;
        }
        if allow_split && compiler_present && engine.model.architecture.head_width == 64 {
            progress("HIP优化准备：正在检查非零长KV的BF16/F32 Split-K分片与合并数值…".into());
            verified.split_k = record_optional_result(prepare_optional(||
                model::hip_attention::validate_split_k(&device)), OptimizationFeature::SplitK,
                &mut failures, &mut progress)?;
        }
        if allow_sampling && compiler_present {
            progress("HIP优化准备：正在编译或复用8序列GPU采样，并检查FP32参考结果…".into());
            verified.gpu_sampling = record_optional_result(prepare_optional(|| check_sampler(&device)),
                OptimizationFeature::GpuSampling, &mut failures, &mut progress)?;
        }
        if !optional_requested {
            progress("HIP普通兼容准备完成：根据当前设置未执行可选JIT或编译器探测".into());
        } else if !compiler_present {
            progress("HIP可选编译器不可用，保留已验证的同设备普通GPU模型路径".into());
        }
        let after = strict::counters();
        ensure!(before.blocked_downloads == after.blocked_downloads
            && before.blocked_dtype_widenings == after.blocked_dtype_widenings,
            "HIP模型验证期间尝试了隐藏CPU计算或隐式精度回退");
        let evidence = if !optional_requested { Vec::new() } else { match model::hip_kernel::preparation_evidence(rocm) {
            Ok(evidence) => evidence.into_iter().filter(|item| item.module_loaded
                && item.gfx == architecture && item.pci_bus_id.eq_ignore_ascii_case(&pci_bus_id)
                && item.runtime_version == runtime && item.driver_version == driver
                && item.kernel_sdk_version == info.build_hip_version).collect::<Vec<_>>(),
            Err(error) => {
                progress(format!("未取得当前设备的可选kernel加载实证，保持普通GPU能力：{error}"));
                Vec::new()
            }
        }};
        let compiler = evidence.first().map(|item| OptionalCompilerFingerprint::external_hipcc(
            item.compiler_path.clone(), item.compiler_version.clone(),
            item.module_loaded && (item.compiled_successfully_this_process || item.code_object_cache_hit)))
            .unwrap_or_default();
        let identity = ComputeDeviceFingerprint { source: IdentitySource::ModelComputeRuntime,
            vendor: GpuVendor::Amd, pci_vendor_id: None, uuid: None,
            pci_bus_id: Some(pci_bus_id),
            ordinal: Some(rocm.ordinal()), architecture: architecture.clone() };
        let device_id = identity.device_id().map_err(anyhow::Error::msg)?;
        let dtype = format!("{:?}", DType::BF16);
        let fingerprint = BackendRuntimeFingerprint { device: identity, backend: ComputeBackend::Hip,
            runtime: RuntimeKernelMatch { driver_version: driver.to_string(), runtime_version: runtime.to_string(),
                kernel_runtime_version: info.build_hip_version.into(), driver_runtime_verified: true,
                kernel_runtime_verified: true }, dtype: dtype.clone(),
            head_width: engine.model.architecture.head_width, ordinary_model_verified: true, compiler,
            model_layout: Some(ModelLayoutFingerprint { layers: engine.model.architecture.layers,
                width: engine.model.architecture.width, heads: engine.model.architecture.heads,
                ffn_hidden: engine.model.architecture.ffn_hidden }),
            implemented: FeatureSet { fusion: engine.model.architecture.head_width == 64,
                gpu_sampling: true, split_k: engine.model.architecture.head_width == 64,
                primitive_graph: false },
            verification: Some(KernelFeatureVerification { device_id, architecture,
                runtime_version: runtime.to_string(), dtype, head_width: engine.model.architecture.head_width,
                numerical_checks_passed: true, features: verified, auto_recommended: FeatureSet::default(),
                performance_device_id: None }) };
        let detection = capabilities_from_fingerprint(&fingerprint).map_err(anyhow::Error::msg)?;
        let sources = evidence.iter().map(|item| HipKernelSourceFingerprint {
            kernel_name: item.kernel_name.clone(), source_fingerprint: item.source_fingerprint.clone(),
            cache_fingerprint: item.cache_fingerprint.clone(),
        }).collect();
        if let Err(error) = register_kernel_sources(&fingerprint, sources) {
            progress(format!("HIP性能推荐源码快照不可用，Auto保持普通路径：{error}"));
        }
        engine.hip_fingerprint = Some(fingerprint.clone());
        engine.capabilities = detection.capabilities;
        engine.acceleration_failures = SessionFailures::new(&engine.capabilities);
        engine.provider_preparation_attempted = compiler_present && optional_requested;
        engine.provider_preparation_details = vec![
            format!("本次同设备F32/BF16普通矩阵与真实完整模型A/B通过：max_error={error:.6}, relative_rms={rms:.6}"),
            format!("本次可选数值验证：Fusion={}，长KV SplitK={}，8序列GPU采样={}",
                verified.fusion, verified.split_k, verified.gpu_sampling),
            format!("按设置跳过：Fusion={}，SplitK={}，GPU采样={}",
                !allow_fusion, !allow_split, !allow_sampling),
        ];
        for (feature, reason) in &failures {
            engine.provider_preparation_details.push(format!("{}准备未通过，保留同设备普通路径：{reason}", feature.label()));
        }
        for item in &evidence {
            engine.provider_preparation_details.push(format!("{}：本进程编译成功={}，加载缓存={}，源码={}",
                item.kernel_name, item.compiled_successfully_this_process, item.code_object_cache_hit,
                item.source_fingerprint));
        }
        engine.set_acceleration_preferences(settings.acceleration.clone());
        engine.set_gpu_sampling_override(settings.sampling_override);
        for (feature, reason) in failures {
            engine.acceleration_failures.record(feature, FailureClass::Preparation, reason);
        }
        for detail in detection.details { progress(detail); }
        progress(format!("HIP本次加载已验证能力：Fusion={}，SplitK={}，GPU采样={}；Graph与未实测Auto收益保持关闭",
            engine.capabilities.available.fusion, engine.capabilities.available.split_k,
            engine.capabilities.available.gpu_sampling));
        Ok(engine)
    }

    pub fn load_rocm_at(config: &EngineConfig, ordinal: usize,
        progress: impl FnMut(String)) -> Result<NativeEngine> {
        load_rocm_at_with_preparation(config, ordinal, &default_preparation_settings(), progress)
    }

    pub fn load_rocm_with_preparation(config: &EngineConfig, settings: &LoadPreparationSettings,
        progress: impl FnMut(String)) -> Result<NativeEngine> {
        let mut count = 0i32;
        ensure!(unsafe { sys::hipGetDeviceCount(&mut count) } == 0 && count > 0,
            "找不到可用的HIP计算设备；不会改用渲染适配器、CUDA或CPU");
        let selected = std::env::var("PAINTER_HIP_DEVICE");
        let ordinal = match selected {
            Ok(value) => {
                let ordinal: usize = value.trim().parse().context("PAINTER_HIP_DEVICE必须是HIP设备编号")?;
                ensure!(ordinal < count as usize, "PAINTER_HIP_DEVICE超出当前{count}张HIP显卡的范围");
                ordinal
            }
            Err(std::env::VarError::NotPresent) if count == 1 => 0,
            Err(std::env::VarError::NotPresent) => bail!(
                "检测到{count}张HIP显卡，请用PAINTER_HIP_DEVICE明确选择0至{}，不会静默使用0号卡", count - 1),
            Err(error) => bail!("无法读取PAINTER_HIP_DEVICE：{error}"),
        };
        load_rocm_at_with_preparation(config, ordinal, settings, progress)
    }

    pub fn load_rocm(config: &EngineConfig, progress: impl FnMut(String)) -> Result<NativeEngine> {
        load_rocm_with_preparation(config, &default_preparation_settings(), progress)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn compatible_mode_skips_both_optional_paths_even_when_forced_on() {
            let mut settings = default_preparation_settings();
            settings.acceleration.mode = AccelerationMode::Compatible;
            settings.acceleration.fusion = FeaturePreference::Enabled;
            settings.sampling_override = Some(true);
            assert_eq!(preparation_permissions(&settings), (false, false, false));
        }

        #[test]
        fn per_feature_disables_are_independent_before_toolchain_discovery() {
            let mut settings = default_preparation_settings();
            assert_eq!(preparation_permissions(&settings), (true, true, true));
            settings.acceleration.fusion = FeaturePreference::Disabled;
            assert_eq!(preparation_permissions(&settings), (false, true, true));
            settings.sampling_override = Some(false);
            assert_eq!(preparation_permissions(&settings), (false, true, false));
            settings.acceleration.split_k = FeaturePreference::Disabled;
            assert_eq!(preparation_permissions(&settings), (false, false, false));
            settings.acceleration.fusion = FeaturePreference::Enabled;
            assert_eq!(preparation_permissions(&settings), (true, false, false));
        }
    }
}

#[cfg(feature = "rocm")]
pub use rocm::{load_rocm, load_rocm_at, load_rocm_with_preparation, load_rocm_at_with_preparation};
