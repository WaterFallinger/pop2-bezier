//! Native ATen/HIP execution with the same canvas/region/history contracts.
//! All model mathematics and all nine attributes stay behind the C ABI.

use super::*;
use crate::acceleration::{
    AccelerationMode, AccelerationSettings, BackendCapabilities, ComputeBackend, FeaturePreference,
    FeatureSet, GpuVendor,
};
use std::rc::Rc;
#[path = "aten_ffi.rs"]
mod ffi;
#[path = "aten_profile.rs"]
mod performance_profile;

fn requested_kv(settings: &AccelerationSettings) -> (u32, &'static str) {
    let (value, label) = match settings.kv_int8 {
        FeaturePreference::Auto => (0, "auto"),
        FeaturePreference::Enabled => (1, "enabled"),
        FeaturePreference::Disabled => (2, "disabled"),
    };
    (
        if settings.mode == AccelerationMode::Compatible {
            2
        } else {
            value
        },
        label,
    )
}
fn configure_kv_info(
    model: &ffi::Model,
    settings: &AccelerationSettings,
) -> Result<serde_json::Value> {
    let (preference, label) = requested_kv(settings);
    let mut info = model.configure_kv(preference)?;
    info["kv_int8"]["requested"] = label.into();
    if settings.mode == AccelerationMode::Compatible {
        info["kv_int8"]["reason"] = "普通兼容模式关闭INT8 KV，使用同GPU BF16历史缓存".into();
    }
    Ok(info)
}

// Older ABI1 libraries can refresh Graph capabilities without reporting KV
// storage. Preserve only the missing extension report, never stale Graph data.
fn refreshed_runtime_info(
    mut refreshed: serde_json::Value,
    previous: &serde_json::Value,
) -> serde_json::Value {
    if refreshed.get("kv_int8").is_none() {
        if let Some(kv) = previous.get("kv_int8").filter(|value| value.is_object()) {
            refreshed["kv_int8"] = kv.clone();
        }
    }
    refreshed
}

#[derive(Clone)]
struct AtenGuidedCache {
    conditional: ffi::Cache,
    unconditional: Option<ffi::Cache>,
}

struct AtenLane {
    index: usize,
    full: Vec<[u32; 9]>,
    history: History,
    rng: StdRng,
    generated: usize,
    cache: Option<AtenGuidedCache>,
    window_records: Vec<[u32; 9]>,
    full_raw_cursor: usize,
    manual_review_regions: HashSet<(usize, usize)>,
    fine_aux_seen: bool,
    pending_stage_context: Vec<PendingStageRecord>,
    activated_pending: HashSet<usize>,
    summary: Option<GenerationSummary>,
    pause_target: Option<Stage>,
    redraw_scope: Option<RedrawScope>,
}
#[derive(Default, serde::Serialize)]
struct ContextWindowStats {
    append_boundary_records: u64,
    reorders: u64,
    prefilled_visual_tokens: u64,
    reused_prefix_tokens: u64,
}
impl AtenLane {
    fn is_global(&self) -> bool {
        global_incomplete(&self.history)
    }
    fn background(&self) -> bool {
        self.is_global() && self.history.fixed.len() == 1
    }
    fn region(&self) -> Option<(usize, usize)> {
        self.history.active.as_ref().map(|r| (r.side, r.index))
    }
    fn fine_aux_pending(&self,aux:&[RegionBlock])->bool{
        self.history.active.as_ref().is_some_and(|active|active.side==8)
            &&!self.fine_aux_seen&&!aux.is_empty()
    }
}

fn switch_half_region(lane: &mut AtenLane, enabled: bool) -> Result<bool> {
    let Some(active) = lane.history.active.as_ref() else {
        return Ok(false);
    };
    let Some(index) = half_region_switch_target(&lane.history, active, enabled, &mut lane.rng)
    else {
        return Ok(false);
    };
    let active = lane.history.active.take().context("半程切换缺少活动区域")?;
    let next = interrupt_and_select_region(&mut lane.history, &mut lane.full, active, index);
    lane.history.active = Some(next);
    // The old KV still represents the exact prefix before INTERRUPT+marker.
    // Keep its window and cursor so the next preparation appends only that
    // control tail, or reuses the common prefix if capacity requires packing.
    Ok(true)
}

fn region_end_allowed(
    count: usize,
    cap: usize,
    review: bool,
    fresh_model_end: Option<bool>,
) -> bool {
    (!review && count >= cap) || fresh_model_end == Some(true)
}

fn may_half_switch(review: bool, fresh_model_end: Option<bool>) -> bool {
    !review || fresh_model_end == Some(false)
}

fn review_window_needs_space(
    records: usize,
    text: usize,
    remaining: usize,
    overclock: bool,
) -> bool {
    records.checked_add(remaining).is_none_or(|visual| {
        (!overclock && visual > 897)
            || visual
                .checked_add(text)
                .is_none_or(|total| total > MAX_CONTEXT)
    })
}

/// At most two templates live during one preparation round: its first context
/// remains pinned while only the most recent divergent context is replaced.
/// This prevents lane0's empty-region hops from evicting every sibling's common
/// starting point. Values are cloned; GuidedCache provides explicit KV COW.
/// Text, CFG, model and RoPE policy are fixed by the enclosing request, and this
/// cache is discarded before leaving that request's preparation round.
struct PrefillReuseCache<T> {
    first: Option<(Vec<[u32; 9]>, T)>,
    recent: Option<(Vec<[u32; 9]>, T)>,
    actual_prefill_calls: usize,
    reuse_hits: usize,
}

impl<T> Default for PrefillReuseCache<T> {
    fn default() -> Self {
        Self {
            first: None,
            recent: None,
            actual_prefill_calls: 0,
            reuse_hits: 0,
        }
    }
}

impl<T: Clone> PrefillReuseCache<T> {
    fn get(&mut self, records: &[[u32; 9]]) -> Option<T> {
        let matching = self
            .first
            .as_ref()
            .filter(|(key, _)| key == records)
            .or_else(|| self.recent.as_ref().filter(|(key, _)| key == records));
        let value = matching.map(|(_, value)| value.clone());
        if value.is_some() {
            self.reuse_hits += 1;
        }
        value
    }

    fn remember(&mut self, records: &[[u32; 9]], value: T) {
        if self.first.is_none() {
            self.first = Some((records.to_vec(), value));
        } else if self.first.as_ref().is_some_and(|(key, _)| key == records) {
            // The common template is immutable for the whole round.
        } else {
            self.recent = Some((records.to_vec(), value));
        }
    }
}

pub struct AtenEngine {
    model: Rc<ffi::Model>,
    tokenizer: SentencePieceProcessor,
    pub model_description: String,
    capabilities: BackendCapabilities,
    settings: AccelerationSettings,
    sampling_override: Option<bool>,
    context_overclock: bool,
    fine_aux_blocks: Vec<RegionBlock>,
    graph_enabled: bool,
    gpu_sampling: bool,
    efficient_attention: bool,
    fault: std::cell::RefCell<Option<String>>,
    adaptive: std::cell::RefCell<performance_profile::AdaptiveStore>,
    runtime_info: serde_json::Value,
    optional_prepared: bool,
    request_scope: &'static str,
    request_alpha: f64,
    active_lanes: std::cell::Cell<usize>,
    last_frame_plan: std::cell::Cell<Option<(FeatureSet, usize, bool)>>,
    mode_notices: std::cell::RefCell<Vec<String>>,
    context_stats: std::cell::RefCell<ContextWindowStats>,
}

impl AtenEngine {
    pub fn load_configured(
        config: &EngineConfig,
        preparation: &LoadPreparationSettings,
        mut progress: impl FnMut(String),
    ) -> Result<Self> {
        if config.use_cuda {
            bail!("所选原生ATen构建使用AMD HIP，无法替换CUDA请求");
        }
        if !config.model_path.is_absolute() || !config.model_path.is_file() {
            bail!("模型权重路径必须为存在的绝对文件路径");
        }
        progress("加载原生 SentencePiece 分词器".into());
        let tokenizer = SentencePieceProcessor::open(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("无法加载分词器：{e}"))?;
        let ordinal = match std::env::var("PAINTER_HIP_DEVICE") {
            Ok(value) => value
                .parse::<u32>()
                .context("PAINTER_HIP_DEVICE必须是有效设备编号")?,
            Err(std::env::VarError::NotPresent) => u32::MAX,
            Err(error) => return Err(error.into()),
        };
        let efficient = preparation.acceleration.mode != AccelerationMode::Compatible
            && preparation.acceleration.fusion != FeaturePreference::Disabled;
        progress("ATen优化准备：读取原始权重并验证实际AMD设备与运行库，请稍候…".into());
        let model = ffi::Model::load(&config.model_path, ordinal, efficient)?;
        progress("正在准备历史KV存储与数值校验，请稍候…".into());
        let configured_info = configure_kv_info(&model, &preparation.acceleration)?;
        let info = &configured_info;
        for field in ["layers", "width", "heads", "ffn_hidden", "head_width"] {
            if info[field].as_u64().is_none_or(|v| v == 0) {
                bail!("ATen模型缺少实际架构字段 {field}");
            }
        }
        if info["width"].as_u64()
            != info["heads"]
                .as_u64()
                .zip(info["head_width"].as_u64())
                .map(|(a, b)| a * b)
        {
            bail!("ATen模型注意力头数与宽度不一致");
        }
        let metadata = config
            .model_path
            .parent()
            .and_then(|p| std::fs::read(p.join("READY.json")).ok())
            .and_then(|v| serde_json::from_slice::<serde_json::Value>(&v).ok());
        let epoch = metadata.as_ref().and_then(|v| v["epoch"].as_u64());
        let step = metadata.as_ref().and_then(|v| v["global_step"].as_u64());
        let model_description = format!(
            "{} · {}层 / {}宽 / {}头 / FFN{} · AMD HIP / 原生ATen C++ / BF16{}",
            epoch.map_or_else(|| "本地权重".into(), |e| format!("Epoch {e}")),
            info["layers"],
            info["width"],
            info["heads"],
            info["ffn_hidden"],
            step.map_or_else(String::new, |s| format!(" · step {s}"))
        );
        let key = serde_json::to_string(
            &serde_json::json!({"provider":"aten_hip","ordinal":info["device_ordinal"],
            "name":info["device_name"],"pci":info["pci_bus_id"],"arch":info["architecture"],"torch":info["torch_version"],
            "hip":info["hip_version"],"driver":info["hip_driver_version"],"abi":info["abi_version"],
            "build_id":info["build_id"],"runtime_id":info["runtime_id"],"dtype":info["dtype"],
            "layers":info["layers"],"width":info["width"],"heads":info["heads"],"head_width":info["head_width"],"ffn":info["ffn_hidden"]}),
        )?;
        let mut capabilities =
            BackendCapabilities::baseline(GpuVendor::Amd, ComputeBackend::AtenHip, key);
        capabilities.available = FeatureSet {
            fusion: info["efficient_attention_available"].as_bool() == Some(true),
            gpu_sampling: info["gpu_sampling_available"].as_bool() == Some(true),
            primitive_graph: info["hip_graph_available"].as_bool() == Some(true),
            split_k: false,
        };
        // The native library reports only capabilities it actually validates.
        // Performance recommendations can be narrower than compatibility.
        capabilities.auto_recommended = FeatureSet::default();
        progress(model_description.clone());
        let runtime_info = configured_info;
        let optional_prepared = runtime_info["optional_preparation_attempted"]
            .as_bool()
            .unwrap_or(efficient);
        Ok(Self {
            model,
            tokenizer,
            model_description,
            capabilities,
            settings: preparation.acceleration.clone(),
            sampling_override: preparation.sampling_override,
            context_overclock: false,
            fine_aux_blocks: Vec::new(),
            graph_enabled: false,
            gpu_sampling: false,
            efficient_attention: false,
            fault: std::cell::RefCell::new(None),
            adaptive: std::cell::RefCell::new(performance_profile::AdaptiveStore::new(
                &runtime_info,
            )),
            runtime_info,
            optional_prepared,
            request_scope: "global",
            request_alpha: preparation.renyi_alpha,
            active_lanes: std::cell::Cell::new(preparation.lanes),
            last_frame_plan: std::cell::Cell::new(None),
            mode_notices: std::cell::RefCell::new(Vec::new()),
            context_stats: std::cell::RefCell::new(ContextWindowStats::default()),
        })
    }
    pub fn acceleration_capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }
    pub fn diagnostic_info(&self) -> serde_json::Value {
        serde_json::json!({"provider":"aten_hip","runtime":self.runtime_info,
            "kv_int8":self.runtime_info["kv_int8"],
            "context_window":*self.context_stats.borrow(),"incremental_prefill_available":self.model.supports_extend(),
            "capabilities":self.capabilities,"selected":{"fusion":self.efficient_attention,
                "gpu_sampling":self.gpu_sampling,"primitive_graph":self.graph_enabled},
            "fault":self.fault.borrow().clone(),"backend_diagnostics":self.model.diagnostics().ok(),
            "auto_selection":self.adaptive.borrow().diagnostic_info()})
    }
    pub fn set_acceleration_preferences(&mut self, value: AccelerationSettings) {
        self.settings = value;
    }
    pub fn set_gpu_sampling_override(&mut self, value: Option<bool>) {
        self.sampling_override = value;
    }
    pub fn set_diagnostic_cache_forecast(&mut self, _: Option<usize>) {}
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        prompt_validation::validate_with_processor(&self.tokenizer, text)
            .into_token_ids()
            .map_err(anyhow::Error::msg)
    }
    pub fn prepare_after_load(
        &mut self,
        settings: &LoadPreparationSettings,
        mut progress: impl FnMut(String),
    ) -> Result<LoadPreparationReport> {
        self.settings = settings.acceleration.clone();
        self.sampling_override = settings.sampling_override;
        self.configure_generation_cache(
            settings.lanes,
            settings.cfg_scale,
            settings.renyi_alpha,
            0,
            usize::MAX,
            &mut progress,
        )?;
        let details = vec![
            "原生ATen/HIP使用已编译的计算库；本次没有把整个模型交给Python或Triton".into(),
            "加载时验证真实计算设备与模型；完整图元Graph按实际上下文、区域和批量捕获".into(),
            format!(
                "高效注意力={}，GPU采样={}，整图元Graph={}",
                self.efficient_attention, self.gpu_sampling, self.graph_enabled
            ),
        ];
        Ok(LoadPreparationReport {
            jit_attempted: false,
            ready: true,
            summary: "原生ATen/HIP模型与运行库已就绪".into(),
            details,
            graph_deferred: self.graph_enabled,
        })
    }
    pub fn primitive_graph_status(&self) -> String {
        match self.model.diagnostics() {
            Ok(v) => format!(
                "整图元Graph（ATen/HIP）：捕获{}次，重放{}次{}",
                v["graph_captures"].as_u64().unwrap_or(0),
                v["graph_replays"].as_u64().unwrap_or(0),
                v["last_graph_error"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map_or_else(String::new, |s| format!("；普通路径原因：{s}"))
            ),
            Err(_) => "整图元Graph（ATen/HIP）：诊断不可用".into(),
        }
    }
    pub fn execution_error_message(&self, error: &anyhow::Error) -> String {
        let message = format!("{error:#}");
        if message.contains("CAPTURE_STATE_UNCERTAIN") {
            *self.fault.borrow_mut() = Some(message.clone());
            if message.contains("需要重新启动应用") {
                return message;
            }
            return format!("CAPTURE_STATE_UNCERTAIN：捕获状态不确定，需要重新启动应用；已保留完整画布：{message}");
        }
        if message.contains("ATEN_EXECUTION_UNCERTAIN") || uncertain_device_execution(&message) {
            *self.fault.borrow_mut() = Some(message.clone());
            format!("设备执行状态不确定，请重新加载模型；已保留完整画布：{message}")
        } else {
            message
        }
    }
    fn configure_generation_cache(
        &mut self,
        lanes: usize,
        cfg: f64,
        alpha: f64,
        _prefix: usize,
        _new: usize,
        status: &mut impl FnMut(String),
    ) -> Result<()> {
        if let Some(error) = self.fault.borrow().as_ref() {
            if error.contains("CAPTURE_STATE_UNCERTAIN") {
                bail!("CAPTURE_STATE_UNCERTAIN：捕获状态不确定，需要重新启动应用；已保留完整画布");
            }
            bail!("请重新加载模型：{error}");
        }
        let guidance = if (cfg - 1.).abs() > 1e-12 { 2 } else { 1 };
        ffi::row_geometry(lanes, guidance)?;
        self.model.begin_request()?;
        if !cfg.is_finite() || cfg < 0. || !alpha.is_finite() || alpha <= 1. {
            bail!("ATen并行数、CFG或Rényi参数无效");
        }
        if self.settings.mode != AccelerationMode::Compatible && !self.optional_prepared {
            status("ATen优化准备：正在验证本次模型尚未准备的加速能力，请稍候…".into());
            let refreshed = self.model.prepare_optional()?;
            self.optional_prepared = true;
            self.capabilities.available = FeatureSet {
                fusion: refreshed["efficient_attention_available"].as_bool() == Some(true),
                gpu_sampling: refreshed["gpu_sampling_available"].as_bool() == Some(true),
                primitive_graph: refreshed["hip_graph_available"].as_bool() == Some(true),
                split_k: false,
            };
            self.runtime_info = refreshed;
        }
        self.runtime_info = configure_kv_info(&self.model, &self.settings)?;
        self.capabilities.available.primitive_graph =
            self.runtime_info["hip_graph_available"].as_bool() == Some(true);
        status(format!(
            "KV缓存：{}；{}",
            self.runtime_info["kv_int8"]["storage"]
                .as_str()
                .unwrap_or("bf16"),
            self.runtime_info["kv_int8"]["reason"]
                .as_str()
                .unwrap_or("未提供原因")
        ));
        let explicit_graph = self.settings.mode == AccelerationMode::Optimized
            || self.settings.primitive_graph == FeaturePreference::Enabled;
        if explicit_graph
            && self.runtime_info["kv_int8"]["enabled"].as_bool() != Some(true)
            && self.settings.mode != AccelerationMode::Compatible
            && self.settings.fusion != FeaturePreference::Disabled
            && self.sampling_override != Some(false)
            && self.capabilities.available.fusion
            && self.capabilities.available.gpu_sampling
            && !self.runtime_info["graph_preparation_attempted"]
                .as_bool()
                .unwrap_or(false)
        {
            status("ATen优化准备：正在按明确请求验证整图元Graph；自动模式不需要此步骤".into());
            let refreshed = self.model.prepare_graph()?;
            self.capabilities.available.primitive_graph =
                refreshed["hip_graph_available"].as_bool() == Some(true);
            self.runtime_info = refreshed_runtime_info(refreshed, &self.runtime_info);
        }
        self.request_alpha = alpha;
        self.active_lanes.set(lanes);
        self.mode_notices.borrow_mut().clear();
        if self.settings.mode == AccelerationMode::Auto {
            status(
                "自动硬件适配：读取本机缓存；未测过的配置会进行短小私有比较，不计入绘画速度".into(),
            );
        }
        let (selected, reason) =
            self.features_for_batch(lanes, guidance, cfg, alpha, self.request_scope)?;
        self.efficient_attention = selected.fusion;
        self.gpu_sampling = selected.gpu_sampling;
        self.graph_enabled = selected.primitive_graph;
        self.last_frame_plan
            .set(Some((selected, lanes, self.request_scope == "regional")));
        for notice in self.mode_notices.borrow_mut().drain(..) {
            status(notice);
        }
        status(reason);
        self.model.configure_prefill(self.efficient_attention)?;
        if self.settings.split_k == FeaturePreference::Enabled {
            status("ATen后端使用已验证的注意力实现，独立Split-K选项不适用于此执行器".into());
        }
        status(format!(
            "推理路径：AMD · HIP/ATen · {}",
            execution_label(selected)
        ));
        status(format!(
            "采样后端：{}",
            if self.gpu_sampling {
                "ATen GPU FP32 Rényi"
            } else {
                "原生C++ CPU f64 Rényi（模型仍在GPU）"
            }
        ));
        Ok(())
    }

    fn features_for_batch(
        &self,
        logical_rows: usize,
        guidance: usize,
        cfg: f64,
        alpha: f64,
        scope: &str,
    ) -> Result<(FeatureSet, String)> {
        let mut available = self.capabilities.available;
        let scalars = (cfg as f32).is_finite() && (alpha as f32).is_finite() && (alpha as f32) > 1.;
        available.gpu_sampling &= scalars;
        available.primitive_graph &= scalars;
        let candidate = FeatureSet {
            fusion: available.fusion && self.settings.fusion != FeaturePreference::Disabled,
            gpu_sampling: available.gpu_sampling && self.sampling_override != Some(false),
            primitive_graph: false,
            split_k: false,
        };
        let explicit_graph = self.settings.primitive_graph == FeaturePreference::Enabled;
        let (recommended, recommended_reason) = if self.settings.mode == AccelerationMode::Auto
            && !explicit_graph
        {
            let workload = performance_profile::AdaptiveWorkload {
                physical_rows: ffi::row_geometry(logical_rows, guidance)?.physical,
                guidance,
                scope: scope.into(),
                cfg_bits: cfg.to_bits(),
                alpha_bits: alpha.to_bits(),
                candidate,
                kv_storage: self.runtime_info["kv_int8"]["storage"]
                    .as_str()
                    .unwrap_or("bf16")
                    .into(),
            };
            let cached = self.adaptive.borrow_mut().lookup(&workload);
            let record = if let Some(record) = cached {
                record
            } else {
                self.mode_notices.borrow_mut().push(format!("私有硬件适配：正在比较{logical_rows}条分支的普通与已验证eager路径；不绘制作品、不使用用户随机数"));
                let probe = if candidate == FeatureSet::default() {
                    serde_json::json!({"status":"disabled","private_probe":true,"reason":"可选能力未通过数值验证、参数不适用或被关闭，未执行额外模型测量"})
                } else {
                    match self.model.probe_auto(
                        logical_rows,
                        guidance,
                        scope == "regional",
                        cfg,
                        alpha,
                        candidate.fusion,
                        candidate.gpu_sampling,
                        performance_profile::ADAPTIVE_PREFIX_RECORDS,
                    ) {
                        Ok(value) => value,
                        Err(error) => {
                            let message = format!("{error:#}");
                            if message.contains("CAPTURE_STATE_UNCERTAIN")
                                || message.contains("ATEN_EXECUTION_UNCERTAIN")
                                || uncertain_device_execution(&message)
                            {
                                *self.fault.borrow_mut() = Some(message);
                                return Err(error);
                            }
                            serde_json::json!({"status":"disabled","private_probe":true,"reason":format!("私有适配未成功，保留同GPU普通路径：{message}")})
                        }
                    }
                };
                let record = performance_profile::adaptive_record(workload, probe);
                if let Some(issue) = self.adaptive.borrow_mut().remember(record.clone()) {
                    self.mode_notices.borrow_mut().push(issue);
                }
                record
            };
            (record.selected, record.reason)
        } else {
            (
                FeatureSet::default(),
                "按明确选择使用已验证能力；未进行自动性能测量".into(),
            )
        };
        let selected = select_features(
            &self.settings,
            self.sampling_override,
            available,
            recommended,
        );
        let mut reason = match self.settings.mode {
            AccelerationMode::Compatible => "普通兼容模式使用GPU模型与CPU采样，关闭可选加速".into(),
            AccelerationMode::Optimized => "优先加速模式使用本次数值验证通过的可用能力".into(),
            AccelerationMode::Auto => recommended_reason,
        };
        let graph_requested = self.settings.primitive_graph != FeaturePreference::Disabled
            && (self.settings.primitive_graph == FeaturePreference::Enabled
                || self.settings.mode == AccelerationMode::Optimized
                || recommended.primitive_graph);
        if graph_requested
            && !selected.primitive_graph
            && self.settings.mode != AccelerationMode::Compatible
        {
            let dependency = if self.runtime_info["kv_int8"]["enabled"].as_bool() == Some(true) {
                "INT8 KV采用独立融合读写；当前关闭完整图元Graph，保留INT8缓存与GPU推理"
            } else if self.settings.fusion == FeaturePreference::Disabled {
                "完整图元Graph需要高效注意力；融合已关闭，本次使用普通GPU路径"
            } else if !available.fusion {
                "本次尚无可用的高效注意力，完整图元Graph保持关闭"
            } else if self.sampling_override == Some(false) {
                "完整图元Graph需要GPU采样反馈；当前明确选择CPU采样"
            } else if !available.gpu_sampling {
                "当前GPU采样未准备或参数不适用，完整图元Graph保持关闭"
            } else {
                "本次完整图元Graph未启用，使用已选择的普通GPU执行"
            };
            reason.push_str("；");
            reason.push_str(dependency);
        }
        Ok((selected, reason))
    }
    fn prefill(
        &self,
        records: &[[u32; 9]],
        text: &[u32],
        cfg: f64,
        scope: &str,
        stop: &AtomicBool,
        status: &mut impl FnMut(String),
    ) -> Result<Option<AtenGuidedCache>> {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        if records.is_empty()
            || text.len() < 2
            || records.len() + text.len() > MAX_CONTEXT
            || (!self.context_overclock && records.len() > 897)
        {
            bail!("当前节点超过模型的1024上下文；不会删除画布图元");
        }
        status(format!(
            "ATen准备上下文：{}个文字token＋{}个图元记录",
            text.len(),
            records.len()
        ));
        let guidance = if (cfg - 1.).abs() > 1e-12 { 2 } else { 1 };
        let (prefill_plan, _) = self.features_for_batch(
            self.active_lanes.get().max(1),
            guidance,
            cfg,
            self.request_alpha,
            scope,
        )?;
        self.model.configure_prefill(prefill_plan.fusion)?;
        let conditional = self.model.prefill(records, text, 0)?;
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let unconditional = if (cfg - 1.).abs() > 1e-12 {
            Some(self.model.prefill(records, &[2, 3], text.len() - 2)?)
        } else {
            None
        };
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        Ok(Some(AtenGuidedCache {
            conditional,
            unconditional,
        }))
    }
    fn parallel_aten_frame(
        &self,
        lanes: &mut [AtenLane],
        ready: &[bool],
        settings: &InferenceSettings,
        stop: &AtomicBool,
    ) -> Result<Option<(Vec<[u32; 9]>, usize, usize)>> {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let mut active = ffi::buffer(lanes.len())?;
        active.extend(
            lanes
                .iter()
                .filter(|lane| ready[lane.index])
                .map(|lane| lane.index),
        );
        let batch = active.len();
        let guidance = if (settings.cfg_scale - 1.).abs() > 1e-12 {
            2
        } else {
            1
        };
        let geometry = ffi::row_geometry(batch, guidance)?;
        let mut uniforms = ffi::filled_buffer(geometry.attributes, 0f32)?;
        let mut bounds = ffi::buffer(geometry.bounds)?;
        let mut inputs = ffi::buffer(geometry.physical)?;
        let mut rows = ffi::filled_buffer(lanes.len(), PARTIAL)?;
        for (row, &index) in active.iter().enumerate() {
            for field in 0..9 {
                uniforms[field * batch + row] = lanes[index].rng.random::<f32>();
            }
        }
        for field in 0..9 {
            for &index in &active {
                let lane = &lanes[index];
                let candidates = field_candidates(field, lane.background(), lane.region())?;
                let start = *candidates.first().context("ATen候选范围为空")?;
                if candidates
                    .iter()
                    .enumerate()
                    .any(|(i, &id)| id != start + i as u32)
                {
                    bail!("ATen候选范围必须连续");
                }
                bounds.push(start - OFFSETS[field]);
                bounds.push(candidates.len() as u32);
            }
        }
        for &index in &active {
            let cache = lanes[index].cache.as_ref().context("ATen并行缓存丢失")?;
            inputs.push(&cache.conditional);
            if let Some(u) = &cache.unconditional {
                inputs.push(u);
            }
        }
        let scope = if active.iter().any(|&index| lanes[index].region().is_some()) {
            "regional"
        } else {
            "global"
        };
        let (selected, _) = self.features_for_batch(
            batch,
            guidance,
            settings.cfg_scale,
            settings.renyi_alpha,
            scope,
        )?;
        let options = ffi::FrameOptions::new(
            batch,
            guidance,
            settings.cfg_scale,
            settings.renyi_alpha,
            selected.primitive_graph,
            selected.gpu_sampling,
            selected.fusion,
        )?;
        let result = self.model.frame(&inputs, &options, &bounds, &uniforms)?;
        drop(inputs);
        // Stop does not expose an internally completed but not yet submitted
        // frame. New generation rebuilds from the last visible tree node.
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let key = (selected, batch, scope == "regional");
        if self.last_frame_plan.get() != Some(key) {
            self.last_frame_plan.set(Some(key));
            let mut notices = self.mode_notices.borrow_mut();
            notices.push(format!(
                "推理路径：AMD · HIP/ATen · {} · {}条分支",
                execution_label(selected),
                batch
            ));
            notices.push(format!(
                "采样后端：{}",
                if selected.gpu_sampling {
                    "ATen GPU FP32 Rényi"
                } else {
                    "原生C++ CPU f64 Rényi（模型仍在GPU）"
                }
            ));
        }
        if result.records.len() != batch || result.caches.len() != geometry.physical {
            bail!("ATen完整图元返回行数不匹配");
        }
        validate_aten_records(&result.records, &bounds)?;
        let mut caches = result.caches.into_iter();
        for (row, &index) in active.iter().enumerate() {
            let conditional = caches.next().context("ATen条件缓存缺失")?;
            let unconditional = if guidance == 2 {
                Some(caches.next().context("ATen无条件缓存缺失")?)
            } else {
                None
            };
            lanes[index].cache = Some(AtenGuidedCache {
                conditional,
                unconditional,
            });
            rows[index] = result.records[row];
        }
        Ok(Some((rows, 9, geometry.physical)))
    }
    pub fn generate_with_regions(
        &mut self,
        request: &GenerationRequest,
        stop: &AtomicBool,
        mut primitive: impl FnMut(GeneratedPrimitive),
        status: impl FnMut(String),
        mut region: impl FnMut(DrawingRegion),
    ) -> Result<GenerationSummary> {
        let result = self.generate_parallel(
            request,
            1,
            stop,
            |_, p| primitive(p),
            status,
            |_, r| region(r),
            |_, _| {},
        )?;
        result
            .lane_summaries
            .into_iter()
            .next()
            .context("ATen单序列缺少结束状态")
    }

    pub fn generate_parallel(
        &mut self,
        request: &GenerationRequest,
        lanes: usize,
        stop: &AtomicBool,
        mut on_primitive: impl FnMut(usize, GeneratedPrimitive),
        mut on_status: impl FnMut(String),
        mut on_region: impl FnMut(usize, DrawingRegion),
        mut on_finished: impl FnMut(usize, GenerationSummary),
    ) -> Result<ParallelSummary> {
        ffi::row_geometry(
            lanes,
            if (request.settings.cfg_scale - 1.).abs() > 1e-12 {
                2
            } else {
                1
            },
        )?;
        let started = Instant::now();

        let effective_settings = request_settings(request);
        let settings = &effective_settings;
        self.context_overclock = settings.context_overclock;
        self.fine_aux_blocks = edited_context_blocks(&request.extra_fine_context)?;
        *self.context_stats.borrow_mut() = ContextWindowStats::default();
        if !settings.cfg_scale.is_finite() || settings.cfg_scale < 0. {
            bail!("CFG必须是有限非负数");
        }
        if !settings.renyi_alpha.is_finite() || settings.renyi_alpha <= 1. {
            bail!("Rényi α必须大于1且为有限数");
        }
        if settings.max_primitives == 0 {
            bail!("每序列本次生成图元上限必须大于零");
        }
        let review_regions = manual_reviews(request)?;
        for p in &request.primitives {
            validate_visible(&p.tokens)?;
        }
        let text = self.tokenize(&request.prompt)?;
        let full = if let Some(records) = &request.history_records {
            let actual: Vec<_> = records.iter().filter(|r| visible(r)).copied().collect();
            let expected: Vec<_> = request.primitives.iter().map(|p| p.tokens).collect();
            if actual != expected {
                bail!("并行初始模型上下文与画布不一致");
            }
            records.clone()
        } else {
            let mut records = vec![BOS_RECORD];
            records.extend(request.primitives.iter().map(|p| p.tokens));
            records
        };
        if full.first() != Some(&BOS_RECORD) {
            bail!("并行初始上下文缺少规范BOS");
        }
        self.request_scope = if (request.redraw_regions.is_empty() || request.redraw_regions.contains(&(1,0))) && global_incomplete(&parse_request_history(&full,request.stage_floor)?) {
            "global"
        } else {
            "regional"
        };
        self.configure_generation_cache(
            lanes,
            settings.cfg_scale,
            settings.renyi_alpha,
            full.len(),
            settings.max_primitives,
            &mut on_status,
        )?;
        let mut states = ffi::buffer(lanes)?;
        for lane in 0..lanes {
            let mut history = parse_request_history(&full,request.stage_floor)?;
            let redraw_scope = RedrawScope::initialize(request, &mut history)?;
            let pause_target = stage_pause_target(&history, settings.pause_at_stage_end);
            states.push(AtenLane {
                index: lane,
                full: full.clone(),
                history,
                rng: StdRng::seed_from_u64(parallel_seed(settings.seed, lane)),
                generated: 0,
                cache: None,
                window_records: Vec::new(),
                full_raw_cursor: 0,
                manual_review_regions: review_regions.clone(),
                fine_aux_seen: false,
                pending_stage_context: request.pending_stage_context.clone(),
                activated_pending: HashSet::new(),
                summary: None,
                pause_target,
                redraw_scope,
            });
            on_region(
                lane,
                DrawingRegion::from_history(&full).map_err(anyhow::Error::msg)?,
            );
        }
        on_status(format!("正在准备 {lanes} 条并行分支 · 画布显示第1分支"));
        if !request.redraw_regions.is_empty() { on_status("局部修改：各分支保留黄色区域已有图形并续画，先粗后细；红色区域须已清空".into()); }
        if settings.context_overclock {
            on_status(
                "上下文超频已开启：全局→粗网格→邻近细网格，保留当前完整区域的生成空间".into(),
            );
        }
        let mut batched_decode_calls = 0;
        let mut maximum_tensor_batch = 0;
        let mut actual_prefill_calls = 0;
        let mut prefill_reuse_hits = 0;
        let execution = (|| -> Result<()> {
            while states.iter().any(|lane| lane.summary.is_none()) {
                self.active_lanes.set(
                    states
                        .iter()
                        .filter(|lane| {
                            lane.summary.is_none() && lane.generated < settings.max_primitives
                        })
                        .count(),
                );
                let mut shared_prefill = PrefillReuseCache::default();
                for lane in states.iter_mut().filter(|lane| lane.summary.is_none()) {
                    if stop.load(Ordering::Relaxed) {
                        finish_lane(lane, true, false, started, &mut on_region, &mut on_finished)?;
                    } else if lane.generated >= settings.max_primitives {
                        finish_lane(
                            lane,
                            false,
                            false,
                            started,
                            &mut on_region,
                            &mut on_finished,
                        )?;
                    } else {
                        self.prepare_parallel_lane(
                            lane,
                            &text,
                            settings,
                            stop,
                            started,
                            &mut on_status,
                            &mut on_region,
                            &mut on_finished,
                            &mut shared_prefill,
                        )?;
                    }
                }
                // Forked layer buffers use copy-on-write. Release this temporary
                // pair before decoding, with no templates retained across rounds.
                actual_prefill_calls += shared_prefill.actual_prefill_calls;
                prefill_reuse_hits += shared_prefill.reuse_hits;
                drop(shared_prefill);
                if stop.load(Ordering::Relaxed) {
                    for lane in states.iter_mut().filter(|lane| lane.summary.is_none()) {
                        finish_lane(lane, true, false, started, &mut on_region, &mut on_finished)?;
                    }
                    break;
                }
                let ready: Vec<_> = states
                    .iter()
                    .map(|lane| lane.summary.is_none() && lane.cache.is_some())
                    .collect();
                if !ready.iter().any(|&v| v) {
                    continue;
                }
                let Some((rows, queries, physical)) =
                    self.parallel_aten_frame(&mut states, &ready, settings, stop)?
                else {
                    for lane in states.iter_mut().filter(|lane| lane.summary.is_none()) {
                        finish_lane(lane, true, false, started, &mut on_region, &mut on_finished)?;
                    }
                    break;
                };
                batched_decode_calls += queries;
                maximum_tensor_batch = maximum_tensor_batch.max(physical);
                for (lane, row) in states
                    .iter()
                    .zip(&rows)
                    .filter(|(lane, _)| ready[lane.index])
                {
                    validate_visible(row)
                        .with_context(|| format!("ATen序列{}完整图元非法", lane.index + 1))?;
                }
                // Commit every already-completed tensor row, even if a UI stop
                // arrives between these callbacks. No completed lane is lost.
                for lane in states.iter_mut().filter(|lane| ready[lane.index]) {
                    let row = rows[lane.index];
                    let global = lane.is_global();
                    lane.full.push(row);
                    lane.window_records.push(row);
                    lane.full_raw_cursor = lane.full.len();
                    if global {
                        lane.history.fixed.push(row);
                    } else {
                        let active = lane.history.active.as_mut().context("并行活动区域丢失")?;
                        active.records.push(row);
                        *lane
                            .history
                            .visit_counts
                            .entry((active.side, active.index))
                            .or_default() += 1;
                    }
                    lane.generated += 1;
                    if ready.iter().position(|&value| value) == Some(lane.index) {
                        for notice in self.mode_notices.borrow_mut().drain(..) {
                            on_status(notice);
                        }
                        on_status(self.primitive_graph_status());
                    }
                    on_primitive(
                        lane.index,
                        GeneratedPrimitive {
                            primitive: InferencePrimitive { tokens: row },
                            history_records: lane.full.clone(),
                        },
                    );
                    on_region(
                        lane.index,
                        DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?,
                    );
                    finish_stage_pause_if_ready(lane, started, &mut on_region, &mut on_finished)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = execution {
            // Preserve independently committed states before propagating an
            // explicit error (including OOM). Never retry with fewer lanes.
            for lane in states.iter_mut().filter(|lane| lane.summary.is_none()) {
                finish_lane(lane, true, false, started, &mut on_region, &mut on_finished)?;
            }
            return Err(error.context("原生并行批量推理失败；未静默降并行数或切CPU"));
        }
        let lane_summaries: Vec<_> = states
            .into_iter()
            .map(|lane| lane.summary.context("并行lane缺少结束状态"))
            .collect::<Result<_>>()?;
        Ok(ParallelSummary {
            lanes,
            generated: lane_summaries.iter().map(|s| s.generated).sum(),
            stopped: lane_summaries.iter().any(|s| s.stopped),
            paused_after_stage: common_paused_stage(&lane_summaries),
            elapsed_seconds: started.elapsed().as_secs_f64(),
            lane_summaries,
            batched_decode_calls,
            maximum_tensor_batch,
            prefill_strategy:
                "ATen immutable prefix snapshots; native full-primitive batch execution".into(),
            actual_prefill_calls,
            prefill_reuse_hits,
            prefill_statistics_tracked: true,
        })
    }

    fn prepare_parallel_lane(
        &self,
        lane: &mut AtenLane,
        text: &[u32],
        settings: &InferenceSettings,
        stop: &AtomicBool,
        started: Instant,
        on_status: &mut impl FnMut(String),
        on_region: &mut impl FnMut(usize, DrawingRegion),
        on_finished: &mut impl FnMut(usize, GenerationSummary),
        shared_prefill: &mut PrefillReuseCache<AtenGuidedCache>,
    ) -> Result<()> {
        loop {
            if finish_stage_pause_if_ready(lane, started, on_region, on_finished)? {
                return Ok(());
            }
            if stop.load(Ordering::Relaxed) {
                return finish_lane(lane, true, false, started, on_region, on_finished);
            }
            if lane.is_global() {
                if lane.cache.is_none() {
                    on_region(
                        lane.index,
                        DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?,
                    );
                    lane.cache = self.prefill_shared(
                        &lane.history.fixed,
                        text,
                        settings.cfg_scale,
                        "global",
                        stop,
                        &mut |s| on_status(format!("序列{}：{s}", lane.index + 1)),
                        shared_prefill,
                        None,
                    )?;
                    if lane.cache.is_none() {
                        return finish_lane(lane, true, false, started, on_region, on_finished);
                    }
                    lane.window_records = lane.history.fixed.clone();
                    lane.full_raw_cursor = lane.full.len();
                }
                return Ok(());
            }
            if lane.history.active.is_none() && activate_generation_pending(&mut lane.history,&mut lane.full,
                &lane.pending_stage_context,&mut lane.activated_pending,&mut lane.manual_review_regions,&mut lane.redraw_scope)? {
                on_region(lane.index,DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?);
                continue;
            }
            if lane.history.active.is_none() {
                let Some((side,index)) = generation_region_target(&lane.history, lane.redraw_scope.as_ref(), &mut lane.rng)? else {
                    return finish_lane(lane, false, true, started, on_region, on_finished);
                };
                lane.history.active = Some(begin_generation_region(&mut lane.history,&mut lane.full,side,index,
                    &lane.pending_stage_context,lane.redraw_scope.as_ref(),&mut lane.manual_review_regions)?);
                on_region(
                    lane.index,
                    DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?,
                );
            }
            let manual_pending = lane.history.active.as_ref().is_some_and(|active| {
                lane.manual_review_regions
                    .contains(&(active.side, active.index))
            });
            let fresh_model_end = if lane.full_raw_cursor == lane.full.len() {
                lane.cache
                    .as_ref()
                    .map(|cache| should_end(cache, settings.cfg_scale))
            } else {
                None
            };
            if may_half_switch(manual_pending, fresh_model_end)
                && switch_half_region(lane, settings.half_region_switch)?
            {
                on_status(format!(
                    "序列{}：已在半程中断当前区域，切换到另一未完成区域",
                    lane.index + 1
                ));
                on_region(
                    lane.index,
                    DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?,
                );
                continue;
            }
            let active = lane.history.active.as_ref().unwrap();
            let count = *lane
                .history
                .visit_counts
                .get(&(active.side, active.index))
                .unwrap_or(&0);
            let review = lane
                .manual_review_regions
                .contains(&(active.side, active.index));
            let remaining = remaining_region_records(&lane.history, active, review);
            let end = region_end_allowed(count, active.cap(), review, fresh_model_end);
            if end {
                lane.full.push(END_RECORD);
                let active = lane.history.active.take().unwrap();
                let key = (active.side, active.index);
                finish_visit(&mut lane.history, active)?;
                lane.manual_review_regions.remove(&key);
                on_region(
                    lane.index,
                    DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?,
                );
                continue;
            }
            if fresh_model_end.is_some() && guarded_scoped_review_full(settings,&lane.history,active,review,lane.redraw_scope.as_ref()) {
                on_status(format!("阶段检查：序列{}当前{}网格已达上限，请删除或调整图形后继续。",
                    lane.index+1,if active.side==4{"粗"}else{"细"}));
                return finish_lane(lane,true,false,started,on_region,on_finished);
            }
            let pending_window_full = review
                && lane.cache.is_some()
                && review_window_needs_space(
                    lane.window_records.len(),
                    text.len(),
                    remaining,
                    settings.context_overclock,
                );
            let inject_aux = lane.fine_aux_pending(&self.fine_aux_blocks);
            if lane.cache.is_none()
                || lane.full_raw_cursor != lane.full.len()
                || pending_window_full
                || inject_aux
            {
                let active = lane.history.active.as_ref().unwrap();
                let mut pool = Vec::new();
                for block in &lane.history.complete {
                    pool.extend(bounded_context_blocks(block));
                }
                for block in lane
                    .history
                    .interrupted
                    .iter()
                    .filter(|r| r.side == active.side)
                {
                    pool.extend(bounded_context_blocks(block));
                }
                if active.side==8 {pool.extend(self.fine_aux_blocks.iter().cloned());}
                let chronological = if settings.context_overclock || review || inject_aux {
                    fine_context_candidate(
                        lane.cache.as_ref().map(|_| lane.window_records.as_slice()),
                        &lane.full,
                        if lane.cache.is_some() {
                            lane.full_raw_cursor
                        } else {
                            0
                        },
                        active,
                        remaining,
                        text.len(),
                        settings.context_overclock,
                        &self.fine_aux_blocks,
                        inject_aux,
                    )?
                } else {
                    None
                };
                let context = if let Some(context) = chronological {
                    if lane.cache.is_some() {
                        self.context_stats.borrow_mut().append_boundary_records +=
                            (lane.full.len() - lane.full_raw_cursor) as u64;
                    }
                    context
                } else {
                    if settings.context_overclock || review || inject_aux {
                        self.context_stats.borrow_mut().reorders += 1;
                    }
                    let mut context = pack_context_for_mode_with_remaining(
                        &lane.history.fixed,
                        &pool,
                        active,
                        text.len(),
                        settings.context_overclock,
                        remaining,
                    )?;
                    context.extend_from_slice(&active.records);
                    context
                };
                on_status(format!(
                    "序列{}：上下文 {}+{}",
                    lane.index + 1,
                    text.len(),
                    context.len()
                ));
                on_region(
                    lane.index,
                    DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?,
                );
                lane.cache = self.prefill_shared(
                    &context,
                    text,
                    settings.cfg_scale,
                    "regional",
                    stop,
                    &mut |s| on_status(format!("序列{}：{s}", lane.index + 1)),
                    shared_prefill,
                    lane.cache
                        .as_ref()
                        .map(|cache| (cache, lane.window_records.as_slice())),
                )?;
                if lane.cache.is_none() {
                    return finish_lane(lane, true, false, started, on_region, on_finished);
                }
                lane.window_records = context;
                lane.full_raw_cursor = lane.full.len();
                if lane.history.active.as_ref().is_some_and(|active|active.side==8) {lane.fine_aux_seen=true;}
                // Evaluate the newly prefetched marker's termination head;
                // different lanes may close/select independently here.
                continue;
            }
            return Ok(());
        }
    }

    fn prefill_shared(
        &self,
        records: &[[u32; 9]],
        text: &[u32],
        cfg: f64,
        scope: &str,
        stop: &AtomicBool,
        on_status: &mut impl FnMut(String),
        shared: &mut PrefillReuseCache<AtenGuidedCache>,
        previous: Option<(&AtenGuidedCache, &[[u32; 9]])>,
    ) -> Result<Option<AtenGuidedCache>> {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        if let Some(cache) = shared.get(records) {
            on_status("复用本轮相同起点的模型上下文".into());
            self.context_stats.borrow_mut().reused_prefix_tokens += records.len() as u64;
            return Ok(Some(cache));
        }
        if let Some((cache, old)) = previous {
            if old == records {
                self.context_stats.borrow_mut().reused_prefix_tokens += records.len() as u64;
                return Ok(Some(cache.clone()));
            }
            let common = super::context_window::reusable_visual_prefix(old, records);
            if common > 0 && self.model.supports_extend() {
                let guidance = if (cfg - 1.).abs() > 1e-12 { 2 } else { 1 };
                let (plan, _) = self.features_for_batch(
                    self.active_lanes.get().max(1),
                    guidance,
                    cfg,
                    self.request_alpha,
                    scope,
                )?;
                self.model.configure_prefill(plan.fusion)?;
                let conditional = self
                    .model
                    .extend(&cache.conditional, text.len() + common, &records[common..])?
                    .context("ATen已声明增量接口但未返回条件缓存")?;
                if stop.load(Ordering::Relaxed) {
                    return Ok(None);
                }
                let unconditional = if guidance == 2 {
                    Some(
                        self.model
                            .extend(
                                cache
                                    .unconditional
                                    .as_ref()
                                    .context("ATen增量CFG缺少空文本缓存")?,
                                2 + common,
                                &records[common..],
                            )?
                            .context("ATen增量空文本缓存缺失")?,
                    )
                } else {
                    None
                };
                if stop.load(Ordering::Relaxed) {
                    return Ok(None);
                }
                let prepared = AtenGuidedCache {
                    conditional,
                    unconditional,
                };
                shared.actual_prefill_calls += 1;
                shared.remember(records, prepared.clone());
                let mut stats = self.context_stats.borrow_mut();
                stats.prefilled_visual_tokens += (records.len() - common) as u64;
                stats.reused_prefix_tokens += common as u64;
                on_status(format!(
                    "复用{common}个共同图元记录，仅计算{}个新增记录",
                    records.len() - common
                ));
                return Ok(Some(prepared));
            }
            if common > 0 && !self.model.supports_extend() {
                on_status("当前原生运行库缺少增量预填充接口，安全重建完整上下文".into());
            }
        }
        shared.actual_prefill_calls += 1;
        let prepared = self.prefill(records, text, cfg, scope, stop, on_status)?;
        if prepared.is_some() {
            self.context_stats.borrow_mut().prefilled_visual_tokens += records.len() as u64;
        }
        if let Some(cache) = &prepared {
            shared.remember(records, cache.clone());
        }
        Ok(prepared)
    }
}

fn execution_label(features: FeatureSet) -> &'static str {
    if features.primitive_graph {
        "完整图元Graph"
    } else if features.fusion || features.gpu_sampling {
        "高效GPU执行"
    } else {
        "普通GPU执行"
    }
}

fn select_features(
    settings: &AccelerationSettings,
    sampling: Option<bool>,
    available: FeatureSet,
    recommended: FeatureSet,
) -> FeatureSet {
    let choose = |auto: FeatureSet| {
        let compatible = settings.mode == AccelerationMode::Compatible;
        let optimized = settings.mode == AccelerationMode::Optimized;
        let enabled = |p: FeaturePreference, has: bool, reco: bool| {
            !compatible
                && has
                && p != FeaturePreference::Disabled
                && (optimized || p == FeaturePreference::Enabled || reco)
        };
        let base_fusion = enabled(settings.fusion, available.fusion, auto.fusion);
        let graph = enabled(
            settings.primitive_graph,
            available.primitive_graph,
            auto.primitive_graph,
        ) && available.fusion
            && settings.fusion != FeaturePreference::Disabled;
        let gpu = !compatible
            && available.gpu_sampling
            && sampling != Some(false)
            && (optimized || sampling == Some(true) || auto.gpu_sampling || graph);
        FeatureSet {
            fusion: base_fusion || (graph && gpu),
            gpu_sampling: gpu,
            primitive_graph: graph && gpu,
            split_k: false,
        }
    };
    let selected = choose(recommended);
    if settings.mode == AccelerationMode::Auto
        && recommended != FeatureSet::default()
        && selected != recommended
    {
        choose(FeatureSet::default())
    } else {
        selected
    }
}

fn should_end(cache: &AtenGuidedCache, cfg: f64) -> bool {
    let c = cache.conditional.terms;
    if let Some(u) = cache.unconditional.as_ref() {
        let u = u.terms;
        (u[1] as f64 + cfg * (c[1] - u[1]) as f64) > (u[0] as f64 + cfg * (c[0] - u[0]) as f64)
    } else {
        c[1] > c[0]
    }
}

fn validate_aten_records(records: &[[u32; 9]], bounds: &[u32]) -> Result<()> {
    let batch = records.len();
    if batch == 0 || bounds.len() != batch * 9 * 2 {
        bail!("ATen图元与区域候选范围数量不匹配");
    }
    for (row, record) in records.iter().enumerate() {
        validate_visible(record)?;
        for field in 0..9 {
            let at = (field * batch + row) * 2;
            let start = OFFSETS[field]
                .checked_add(bounds[at])
                .context("ATen候选起点溢出")?;
            let end = start
                .checked_add(bounds[at + 1])
                .context("ATen候选终点溢出")?;
            if bounds[at + 1] == 0 || !(start..end).contains(&record[field]) {
                bail!("ATen图元越过当前区域的合法候选范围");
            }
        }
    }
    Ok(())
}

fn finish_lane(
    lane: &mut AtenLane,
    stopped: bool,
    completed: bool,
    started: Instant,
    on_region: &mut impl FnMut(usize, DrawingRegion),
    on_finished: &mut impl FnMut(usize, GenerationSummary),
) -> Result<()> {
    finish_lane_with_stage(
        lane,
        stopped,
        completed,
        None,
        started,
        on_region,
        on_finished,
    )
}

fn finish_lane_with_stage(
    lane: &mut AtenLane,
    stopped: bool,
    completed: bool,
    paused_after_stage: Option<Stage>,
    started: Instant,
    on_region: &mut impl FnMut(usize, DrawingRegion),
    on_finished: &mut impl FnMut(usize, GenerationSummary),
) -> Result<()> {
    if lane.summary.is_some() {
        return Ok(());
    }
    on_region(
        lane.index,
        DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?,
    );
    let summary = GenerationSummary {
        generated: lane.generated,
        stopped,
        context_limit: false,
        completed,
        paused_after_stage,
        history_records: lane.full.clone(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    lane.cache = None;
    lane.summary = Some(summary.clone());
    on_finished(lane.index, summary);
    Ok(())
}

fn finish_stage_pause_if_ready(
    lane: &mut AtenLane,
    started: Instant,
    on_region: &mut impl FnMut(usize, DrawingRegion),
    on_finished: &mut impl FnMut(usize, GenerationSummary),
) -> Result<bool> {
    if lane.summary.is_some() {
        return Ok(true);
    }
    if lane.history.active.as_ref().is_some_and(|active| {
        !lane
            .manual_review_regions
            .contains(&(active.side, active.index))
            && capped_final_coarse_visit(&lane.history, active, lane.pause_target)
    }) {
        lane.full.push(END_RECORD);
        let active = lane
            .history
            .active
            .take()
            .context("阶段暂停缺少最后粗网格")?;
        finish_visit(&mut lane.history, active)?;
        lane.cache = None;
    }
    if let Some(stage) = reached_stage_pause(&lane.history, lane.pause_target) {
        finish_lane_with_stage(
            lane,
            false,
            false,
            Some(stage),
            started,
            on_region,
            on_finished,
        )?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn common_paused_stage(summaries: &[GenerationSummary]) -> Option<Stage> {
    let stage = summaries.first()?.paused_after_stage?;
    summaries
        .iter()
        .all(|summary| summary.paused_after_stage == Some(stage))
        .then_some(stage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_graph_refresh_preserves_missing_kv_report_without_stale_graph_data() {
        let previous = serde_json::json!({"hip_graph_available":false,"kv_int8":{
            "requested":"auto","available":false,"enabled":false,"storage":"bf16",
            "reason":"当前原生DLL没有INT8 KV扩展接口"}});
        let refreshed =
            serde_json::json!({"hip_graph_available":true,"graph_preparation_attempted":true});
        let merged = refreshed_runtime_info(refreshed, &previous);
        assert_eq!(merged["kv_int8"], previous["kv_int8"]);
        assert_eq!(merged["hip_graph_available"], true);
        assert_eq!(merged["graph_preparation_attempted"], true);

        let modern = serde_json::json!({"hip_graph_available":false,"kv_int8":{
            "requested":"enabled","available":true,"enabled":true,"storage":"int8_per_token_head",
            "reason":"validated"}});
        assert_eq!(refreshed_runtime_info(modern.clone(), &previous), modern);
    }

    #[test]
    fn int8_kv_preference_maps_abi_and_compatible_mode_without_changing_user_choice() {
        let mut settings = AccelerationSettings::default();
        assert_eq!(requested_kv(&settings), (0, "auto"));
        settings.kv_int8 = FeaturePreference::Enabled;
        assert_eq!(requested_kv(&settings), (1, "enabled"));
        settings.mode = AccelerationMode::Compatible;
        assert_eq!(requested_kv(&settings), (2, "enabled"));
        settings.kv_int8 = FeaturePreference::Disabled;
        assert_eq!(requested_kv(&settings), (2, "disabled"));
    }

    #[test]
    fn auto_keeps_the_measured_bundle_or_retains_only_explicit_user_choices() {
        let available = FeatureSet {
            fusion: true,
            gpu_sampling: true,
            primitive_graph: true,
            split_k: false,
        };
        let mut settings = AccelerationSettings::default();
        assert_eq!(
            select_features(&settings, None, available, available),
            available
        );
        settings.fusion = FeaturePreference::Disabled;
        assert_eq!(
            select_features(&settings, None, available, available),
            FeatureSet::default()
        );
        settings.primitive_graph = FeaturePreference::Enabled;
        let explicit = select_features(&settings, None, available, available);
        assert_eq!(explicit, FeatureSet::default());
        settings.fusion = FeaturePreference::Auto;
        let explicit = select_features(&settings, None, available, FeatureSet::default());
        assert!(explicit.fusion && explicit.gpu_sampling && explicit.primitive_graph);
        assert!(!select_features(&settings, Some(false), available, available).primitive_graph);
        settings.mode = AccelerationMode::Compatible;
        assert_eq!(
            select_features(&settings, Some(true), available, available),
            FeatureSet::default()
        );
    }

    fn ordinary() -> [u32; 9] {
        [512, 1536, 2048, 2518, 4566, 6415, 6702, 6830, 6958]
    }
    fn global_prefix() -> Vec<[u32; 9]> {
        let mut background = ordinary();
        background[5] = OFFSETS[5];
        let mut full = vec![BOS_RECORD, background];
        full.extend(std::iter::repeat_n(ordinary(), 85));
        full
    }
    fn lane(full: Vec<[u32; 9]>) -> AtenLane {
        let history = parse_history(&full).unwrap();
        let pause_target = stage_pause_target(&history, true);
        AtenLane {
            index: 0,
            full,
            history,
            pause_target,
            rng: StdRng::seed_from_u64(7),
            generated: 0,
            cache: None,
            window_records: Vec::new(),
            full_raw_cursor: 0,
            manual_review_regions: HashSet::new(),
            fine_aux_seen: false,
            pending_stage_context: Vec::new(),
            activated_pending: HashSet::new(),
            summary: None,
            redraw_scope: None,
        }
    }

    #[test]
    fn half_region_switch_preserves_amd_window_and_cumulative_visit_without_gpu() {
        for side in [4, 8] {
            let mut raw = global_prefix();
            raw.push(region_marker(side, 0));
            let half = if side == 4 { 10 } else { 5 };
            raw.extend(std::iter::repeat_n(ordinary(), half));
            let mut lane = lane(raw.clone());
            lane.window_records = raw.clone();
            lane.full_raw_cursor = raw.len();
            assert!(switch_half_region(&mut lane, true).unwrap());
            let active = lane.history.active.as_ref().unwrap();
            assert_eq!(active.side, side);
            assert_ne!(active.index, 0);
            assert_eq!(
                &lane.full[raw.len()..],
                &[INTERRUPT_RECORD, region_marker(side, active.index)]
            );
            assert_eq!(lane.window_records, raw);
            assert_eq!(lane.full_raw_cursor, raw.len());
            assert_eq!(lane.history.visit_counts[&(side, 0)], half);
            assert!(!lane.history.closed.contains(&(side, 0)));
            assert_eq!(lane.history.interrupted.len(), 1);
            assert_eq!(
                parse_history(&lane.full).unwrap().visit_counts[&(side, 0)],
                half
            );
            DrawingRegion::from_history(&lane.full).unwrap();
        }
    }

    #[test]
    fn disabled_or_last_unfinished_region_never_injects_an_interrupt() {
        let mut raw = global_prefix();
        raw.push(region_marker(8, 0));
        raw.extend(std::iter::repeat_n(ordinary(), 5));
        let mut lane = lane(raw.clone());
        assert!(!switch_half_region(&mut lane, false).unwrap());
        assert_eq!(lane.full, raw);
        lane.history.closed.extend((1..64).map(|index| (8, index)));
        assert!(!switch_half_region(&mut lane, true).unwrap());
        assert_eq!(lane.full, raw);
        assert!(lane.history.interrupted.is_empty());
    }

    #[test]
    fn fine_auxiliary_blocks_are_injected_once_without_entering_raw_history_or_quota(){
        let mut extra=ordinary();extra[0]=700;extra[1]=1100;
        let aux=edited_context_blocks(&[extra]).unwrap();assert_eq!(aux.len(),1);
        let mut coarse_raw=global_prefix();coarse_raw.push(region_marker(4,0));let coarse=lane(coarse_raw);
        assert!(!coarse.fine_aux_pending(&aux));
        let mut raw=global_prefix();raw.push(region_marker(8,0));let mut fine=lane(raw.clone());
        assert!(fine.fine_aux_pending(&aux));let active=fine.history.active.as_ref().unwrap();
        let window=fine_context_candidate(None,&fine.full,0,active,11,2,true,&aux,true).unwrap().unwrap();
        assert_eq!(window.iter().filter(|row|**row==extra).count(),1);assert_eq!(fine.full,raw);
        fine.window_records=window;fine.full_raw_cursor=fine.full.len();fine.fine_aux_seen=true;
        assert!(!fine.fine_aux_pending(&aux));
        fine.full.extend([END_RECORD,region_marker(8,1)]);fine.history=parse_history(&fine.full).unwrap();
        let next=fine_context_candidate(Some(&fine.window_records),&fine.full,fine.full_raw_cursor,
            fine.history.active.as_ref().unwrap(),11,2,true,&aux,fine.fine_aux_pending(&aux)).unwrap().unwrap();
        assert_eq!(next.iter().filter(|row|**row==extra).count(),1,"appending a boundary must not reinject readonly information");
        assert!(!fine.history.closed.contains(&(8,aux[0].index)));
        assert_eq!(fine.history.visit_counts.get(&(8,aux[0].index)).copied().unwrap_or(0),0);
        assert_eq!(&fine.full[..raw.len()],raw.as_slice());
    }

    #[test]
    fn fine_auxiliary_context_capacity_fallback_still_preserves_the_raw_canvas_contract(){
        let mut extra=ordinary();extra[0]=700;extra[1]=1100;let aux=edited_context_blocks(&[extra]).unwrap();
        let mut raw=global_prefix();raw.push(region_marker(8,0));let fine=lane(raw.clone());let active=fine.history.active.as_ref().unwrap();
        let mut old=global_prefix();old.extend(std::iter::repeat_n(ordinary(),789-old.len()-1));old.push(active.records[0]);
        assert!(fine_context_candidate(Some(&old),&fine.full,fine.full.len(),active,11,224,true,&aux,true).unwrap().is_none());
        let mut fallback=pack_context_for_mode_with_remaining(&fine.history.fixed,&aux,active,224,true,11).unwrap();fallback.extend_from_slice(&active.records);
        assert!(fallback.len()+224+11<=MAX_CONTEXT);assert!(fallback.contains(&extra));
        assert_eq!(fine.full,raw);assert!(fine.history.closed.is_empty());assert!(fine.history.visit_counts.values().all(|count|*count==0));
    }

    #[test]
    fn pending_stage_rows_activate_lazily_once_per_stage_without_rewinding_the_floor(){
        let mut state=lane(global_prefix());let initial=state.full.clone();
        state.pending_stage_context=vec![PendingStageRecord{side:4,index:5,tokens:ordinary()},PendingStageRecord{side:8,index:27,tokens:ordinary()}];
        assert!(activate_pending_stage(&mut state.history,&mut state.full,&state.pending_stage_context,&mut state.activated_pending,&mut state.manual_review_regions).unwrap());
        assert_eq!(&state.full[initial.len()..],&[region_marker(4,5),ordinary(),INTERRUPT_RECORD]);
        assert_eq!(state.manual_review_regions,HashSet::from([(4,5)]));let once=state.full.clone();
        assert!(!activate_pending_stage(&mut state.history,&mut state.full,&state.pending_stage_context,&mut state.activated_pending,&mut state.manual_review_regions).unwrap());assert_eq!(state.full,once);
        state.full.extend([region_marker(4,5),END_RECORD]);
        for index in (0..16).filter(|index|*index!=5){state.full.extend([region_marker(4,index),END_RECORD]);}
        state.history=parse_request_history(&state.full,Some(4)).unwrap();
        assert!(activate_pending_stage(&mut state.history,&mut state.full,&state.pending_stage_context,&mut state.activated_pending,&mut state.manual_review_regions).unwrap());
        assert_eq!(state.history.highest_side,Some(8));assert!(state.manual_review_regions.contains(&(8,27)));
        assert_eq!(state.activated_pending,HashSet::from([4,8]));
        assert!(!global_incomplete(&parse_request_history(&[BOS_RECORD],Some(8)).unwrap()));
        DrawingRegion::from_history(&state.full).unwrap();
    }

    #[test]
    fn stage_pause_finishes_before_pending_fine_rows_are_activated(){
        let mut raw=global_prefix();for index in 0..16{raw.extend([region_marker(4,index),END_RECORD]);}
        let mut state=lane(raw.clone());state.pause_target=Some(Stage::Coarse);
        state.pending_stage_context=vec![PendingStageRecord{side:8,index:27,tokens:ordinary()}];
        assert!(finish_stage_pause_if_ready(&mut state,Instant::now(),&mut|_,_|{},&mut|_,_|{}).unwrap());
        assert_eq!(state.full,raw);assert!(state.activated_pending.is_empty());
        assert_eq!(state.summary.as_ref().unwrap().paused_after_stage,Some(Stage::Coarse));
    }

    #[test]
    fn manual_review_requires_fresh_model_end_even_past_original_quota() {
        assert!(!may_half_switch(true, None));
        assert!(
            !may_half_switch(true, Some(true)),
            "fresh manual END has priority over half switching"
        );
        assert!(may_half_switch(true, Some(false)));
        assert!(
            may_half_switch(false, Some(true)),
            "ordinary half switching retains its original priority"
        );
        for cap in [20, 10] {
            for count in [cap, cap + 1, cap + 20] {
                assert!(
                    !region_end_allowed(count, cap, true, None),
                    "no query must not use the old count"
                );
                assert!(
                    !region_end_allowed(count, cap, true, Some(false)),
                    "negative end prediction must keep reviewing"
                );
                assert!(region_end_allowed(count, cap, true, Some(true)));
                assert!(
                    region_end_allowed(count, cap, false, None),
                    "ordinary quota behavior stays unchanged"
                );
            }
        }
    }

    #[test]
    fn manual_capacity_guard_waits_for_a_fresh_end_decision_and_keeps_raw_unclosed(){
        let mut settings=InferenceSettings{cfg_scale:1.,renyi_alpha:4.,seed:1,max_primitives:usize::MAX,
            context_overclock:true,pause_at_stage_end:true,half_region_switch:false};
        for side in [4,8]{
            let cap=if side==4{20}else{10};let mut raw=global_prefix();raw.push(region_marker(side,0));raw.extend(std::iter::repeat_n(ordinary(),cap));
            let state=lane(raw.clone());let active=state.history.active.as_ref().unwrap();
            assert!(guarded_review_full(&settings,&state.history,active,true));
            assert!(!region_end_allowed(cap,cap,true,None),"a count cannot replace the mandatory fresh prefill");
            assert!(!region_end_allowed(cap,cap,true,Some(false)),"negative model END must stop for editing, never append a fake END");
            assert!(region_end_allowed(cap,cap,true,Some(true)),"a real model END keeps normal close behavior");
            assert_eq!(state.full,raw);assert_ne!(state.full.last(),Some(&END_RECORD));
            settings.pause_at_stage_end=false;assert!(!guarded_review_full(&settings,&state.history,active,true));settings.pause_at_stage_end=true;
        }
    }

    #[test]
    fn manual_review_space_keeps_exact_fits_but_rebuilds_before_the_next_query_overflows() {
        assert!(!review_window_needs_space(998, 24, 2, true));
        assert!(review_window_needs_space(999, 24, 2, true));
        assert!(!review_window_needs_space(895, 20, 2, false));
        assert!(review_window_needs_space(896, 20, 2, false));
        assert!(review_window_needs_space(usize::MAX, 2, 2, true));
    }

    #[test]
    fn final_coarse_manual_review_cannot_pause_by_quota_before_a_new_query() {
        let mut full = global_prefix();
        for index in 0..15 {
            full.extend([region_marker(4, index), END_RECORD]);
        }
        full.push(region_marker(4, 15));
        full.extend(std::iter::repeat_n(ordinary(), 20));
        let mut lane = lane(full.clone());
        lane.manual_review_regions.insert((4, 15));
        assert!(!finish_stage_pause_if_ready(
            &mut lane,
            Instant::now(),
            &mut |_, _| {},
            &mut |_, _| {}
        )
        .unwrap());
        assert_eq!(lane.full, full);
        assert!(lane.history.active.is_some());
        assert!(lane.summary.is_none());
        lane.manual_review_regions.clear();
        assert!(finish_stage_pause_if_ready(
            &mut lane,
            Instant::now(),
            &mut |_, _| {},
            &mut |_, _| {}
        )
        .unwrap());
        assert_eq!(lane.full.last(), Some(&END_RECORD));
    }

    #[test]
    fn global_stage_boundary_retains_last_primitive_and_pauses_before_any_coarse_marker() {
        let mut full = global_prefix();
        let last = full.pop().unwrap();
        let mut lane = lane(full);
        let start = Instant::now();
        assert!(!finish_stage_pause_if_ready(
            &mut lane,
            start,
            &mut |_, _| {},
            &mut |_, _| panic!("early pause")
        )
        .unwrap());
        lane.full.push(last);
        lane.history.fixed.push(last);
        lane.generated = 1;
        assert!(
            finish_stage_pause_if_ready(&mut lane, start, &mut |_, _| {}, &mut |_, _| {}).unwrap()
        );
        let summary = lane.summary.unwrap();
        assert_eq!(summary.generated, 1);
        assert_eq!(summary.paused_after_stage, Some(Stage::Global));
        assert_eq!(summary.history_records.len(), 87);
        assert!(!summary.history_records.iter().any(is_marker));
    }

    #[test]
    fn final_coarse_cap_closes_the_visit_before_stage_pause() {
        let mut full = global_prefix();
        for index in 0..15 {
            full.push(region_marker(4, index));
            full.push(END_RECORD);
        }
        full.push(region_marker(4, 15));
        full.extend(std::iter::repeat_n(ordinary(), 19));
        let mut lane = lane(full);
        let row = ordinary();
        lane.full.push(row);
        lane.history.active.as_mut().unwrap().records.push(row);
        lane.history.visit_counts.insert((4, 15), 20);
        lane.generated = 1;
        assert!(finish_stage_pause_if_ready(
            &mut lane,
            Instant::now(),
            &mut |_, _| {},
            &mut |_, _| {}
        )
        .unwrap());
        let summary = lane.summary.unwrap();
        assert_eq!(summary.paused_after_stage, Some(Stage::Coarse));
        assert_eq!(summary.history_records.last(), Some(&END_RECORD));
        assert_eq!(
            parse_history(&summary.history_records)
                .unwrap()
                .closed
                .len(),
            16
        );
    }

    #[test]
    fn native_output_is_rejected_before_installing_caches_if_it_escapes_the_region() {
        let mut row = ordinary();
        row[0] = 800;
        row[1] = OFFSETS[1] + 100;
        let bounds = (0..9)
            .flat_map(|field| {
                let ids = field_candidates(field, false, Some((4, 3))).unwrap();
                [ids[0] - OFFSETS[field], ids.len() as u32]
            })
            .collect::<Vec<_>>();
        validate_aten_records(&[row], &bounds).unwrap();
        row[0] = 100;
        assert!(validate_aten_records(&[row], &bounds).is_err());
        row[0] = 800;
        row[5] = OFFSETS[5] + 99;
        assert!(validate_aten_records(&[row], &bounds).is_err());
        assert!(validate_aten_records(&[row], &bounds[..10]).is_err());
    }
}
