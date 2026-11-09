//! Native inference worker. UI and history never block on model execution.
use crate::document::{Primitive, SamplingBackend, Settings};
use crate::region_state::{DrawingRegion, Stage,PendingStageRecord};
#[path = "runtime_backend.rs"]
mod backend;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};

pub type RawHistory = Vec<[u32; 9]>;
pub type RawHistoryRecord=[u32;9];
pub enum Command {
    Load {
        model: PathBuf,
        tokenizer: PathBuf,
        settings: Settings,
    },
    Generate {
        prompt: String,
        settings: Settings,
        primitives: Vec<Primitive>,
        history: Option<RawHistory>,
        manual_review_regions: Vec<(usize, usize)>,
        extra_fine_context:Vec<RawHistoryRecord>,
        stage_floor:Option<usize>,pending_stage_context:Vec<PendingStageRecord>,
        redraw_regions:Vec<(usize,usize)>,
        /// Bounded work is diagnostic-only; ordinary UI generation has no cap.
        diagnostic_limit: Option<usize>,
    },
    Shutdown,
}
#[derive(Debug)]
pub enum Event {
    CompatibilityNotice(crate::compatibility_notice::CompatibilityNotice),
    Status(String),
    AccelerationPreparing {
        message: String,
    },
    AccelerationPrepared {
        report: crate::inference::LoadPreparationReport,
    },
    Region(DrawingRegion),
    ParallelRegion {
        lane: usize,
        region: DrawingRegion,
    },
    ParallelPrimitive {
        lane: usize,
        tokens: [u32; 9],
        history: RawHistory,
    },
    ParallelLaneFinished {
        lane: usize,
        summary: crate::inference::GenerationSummary,
    },
    ParallelFinished {
        lanes: usize,
        generated: usize,
        stopped: bool,
        paused_after_stage: Option<Stage>,
        elapsed_seconds: f64,
    },
    Loaded {
        backend: String,
        description: String,
    },
    Primitive {
        tokens: [u32; 9],
        history: RawHistory,
    },
    Finished {
        generated: usize,
        stopped: bool,
        paused_after_stage: Option<Stage>,
        context_limit: bool,
        completed: bool,
        history: RawHistory,
        elapsed_seconds: f64,
    },
    Failed(String),
}
pub struct Runtime {
    sender: mpsc::Sender<Command>,
    pub events: mpsc::Receiver<Event>,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl Runtime {
    pub fn start(repaint: eframe::egui::Context) -> Self {
        let (sender, commands) = mpsc::channel();
        let (events_tx, events) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let worker_builder = std::thread::Builder::new().name("native-inference".into());
        // HIP's native graph preparation traverses the complete nine-attribute
        // graph. Reserve sufficient host stack for that backend's deep graph;
        // keep the established CUDA worker configuration unchanged.
        #[cfg(feature = "aten-hip")]
        let worker_builder = worker_builder.stack_size(16 * 1024 * 1024);
        let worker = worker_builder
            .spawn(move || {
                let emit = |event: Event| {
                    observe_diagnostics(&event);
                    let _ = events_tx.send(event);
                    repaint.request_repaint();
                };
                let mut engine: Option<backend::Engine> = None;
                for command in commands {
                    match command {
                        Command::Shutdown => break,
                        Command::Load { model, tokenizer, settings } => {
                            engine = None;
                            crate::diagnostics::event("model_load_started",serde_json::json!({
                                "model_file":model.file_name().map(|v|v.to_string_lossy().into_owned()),
                                "model_bytes":std::fs::metadata(&model).ok().map(|m|m.len()),
                                "tokenizer_file":tokenizer.file_name().map(|v|v.to_string_lossy().into_owned()),
                                "requested_acceleration":settings.acceleration,"parallel_sequences":settings.parallel_sequences,
                                "selected_provider":std::env::var("PAINTER_COMPUTE_PROVIDER").ok(),
                                "selection_reason":std::env::var("PAINTER_SELECTION_REASON").ok()}));
                            let config = crate::inference::EngineConfig {
                                model_path: model,
                                tokenizer_path: tokenizer,
                                use_cuda: cfg!(feature = "cuda"),
                            };
                            let preparation = crate::inference::LoadPreparationSettings {
                                lanes: settings.parallel_sequences,
                                cfg_scale: settings.cfg as f64,
                                renyi_alpha: settings.renyi_alpha as f64,
                                sampling_override: match settings.sampling_backend {
                                    SamplingBackend::Auto => None,
                                    SamplingBackend::Gpu => Some(true),
                                    SamplingBackend::Cpu => Some(false),
                                },
                                acceleration: settings.acceleration,
                            };
                            let mut actual_cpu_fallback_reason:Option<String>=None;
                            match backend::Engine::load_configured(&config, &preparation, |s| {
                                if let Some(reason)=s.strip_prefix("CPU兼容原因：") {actual_cpu_fallback_reason=Some(reason.to_owned());}
                                if s.starts_with("HIP优化准备：") || s.starts_with("ATen优化准备：") {
                                    emit(Event::AccelerationPreparing { message: s.clone() });
                                }
                                emit(Event::Status(s))
                            }) {
                                Ok(mut loaded) => {
                                    crate::diagnostics::event("model_compute_context",loaded.diagnostic_info());
                                    let mut jit_started = false;
                                    let mut prepared = loaded.prepare_after_load(&preparation, |message| {
                                        jit_started = true;
                                        emit(Event::AccelerationPreparing { message });
                                    });
                                    if let Err(error)=&prepared {
                                        crate::diagnostics::error("acceleration_preparation",error);
                                        // Optional-feature failures are handled inside each
                                        // provider. An ordinary full-model baseline failure
                                        // must never be relabelled as a ready GPU model.
                                        if std::env::var("PAINTER_ALLOW_CPU_FALLBACK").as_deref()==Ok("1")
                                            && backend::can_fallback_before_generation(&format!("{error:#}")) {
                                            actual_cpu_fallback_reason=Some(format!("{error:#}"));
                                            let mut compatible=preparation.clone();
                                            compatible.acceleration.mode=crate::acceleration::AccelerationMode::Compatible;
                                            compatible.sampling_override=Some(false);
                                            crate::diagnostics::event("cpu_compatibility_fallback",serde_json::json!({"reason":format!("{error:#}"),"stage":"gpu_baseline_validation_before_user_generation","visible_to_user":true}));
                                            emit(Event::Status("GPU普通模型验证尚不可用，正在切换到明确的CPU兼容计算；原因已记录到日志".into()));
                                            match backend::Engine::load_cpu_compatible(&config,&compatible,|message|emit(Event::Status(message))){
                                                Ok(cpu)=>{loaded=cpu;prepared=loaded.prepare_after_load(&compatible,|message|emit(Event::AccelerationPreparing{message}));},
                                                Err(cpu_error)=>{crate::diagnostics::error("cpu_fallback_model_load",&cpu_error);emit(Event::Failed(format!("GPU普通路径验证失败，CPU兼容加载也未完成：{cpu_error:#}")));continue;}
                                            }
                                        }
                                    }
                                    let description=loaded.model_description().to_owned();
                                    let capabilities=loaded.acceleration_capabilities();
                                    let backend=format!("{} · {} · {}",capabilities.vendor.label(),capabilities.backend.label(),loaded.implementation_label());
                                    let mut actual=loaded.diagnostic_info();
                                    if let Some(reason)=&actual_cpu_fallback_reason {actual["cpu_compatibility_reason"]=serde_json::json!(reason);}
                                    crate::diagnostics::event("model_effective_configuration",actual.clone());
                                    let selection_reason=actual_cpu_fallback_reason.unwrap_or_else(||std::env::var("PAINTER_SELECTION_REASON").unwrap_or_default());
                                    let session_id=crate::diagnostics::session_id();
                                    let log_directory=crate::diagnostics::directory();
                                    let notice=crate::compatibility_notice::CompatibilityNotice::from_context(&crate::compatibility_notice::NoticeContext{
                                        actual:&actual,requested:&preparation.acceleration,sampling_override:preparation.sampling_override,
                                        preparation:prepared.as_ref().ok(),selection_reason:&selection_reason,
                                        session_id:&session_id,log_directory:log_directory.as_deref(),application_version:env!("CARGO_PKG_VERSION"),
                                    });
                                    // Preparation never silently reloads on another device
                                    // or discards weights after an optional compiler failure.
                                    engine = Some(loaded);
                                    match prepared {
                                        Ok(report) => {
                                            emit(Event::AccelerationPrepared { report });
                                            // This is the sole ready/unlock event and must
                                            // follow every preparation or honest skip report.
                                            for event in model_ready_events(notice,description,backend){emit(event);}
                                        }
                                        Err(error) => {
                                            let message = format!("模型权重已保留，但加速准备未完成：{error:#}");
                                            emit(Event::AccelerationPrepared {
                                                report: crate::inference::LoadPreparationReport {
                                                    jit_attempted: jit_started, ready: false,
                                                    summary: "优化准备未完成，请检查运行环境或重新加载模型".into(),
                                                    details: vec![message.clone()], graph_deferred: false,
                                                },
                                            });
                                            emit(Event::Failed(message));
                                        }
                                    }
                                }
                                Err(e) => emit(Event::Failed(format!("模型加载失败：{e:#}"))),
                            }
                        }
                        Command::Generate {
                            prompt,
                            settings,
                            primitives,
                            history,
                            manual_review_regions,
                            extra_fine_context,
                            stage_floor,pending_stage_context,redraw_regions,
                            diagnostic_limit,
                        } => {
                            crate::diagnostics::event("generation_started",serde_json::json!({
                                "prompt_chars":prompt.chars().count(),"visible_primitive_count":primitives.len(),
                                "history_record_count":history.as_ref().map(Vec::len),"cfg":settings.cfg,
                                "renyi_alpha":settings.renyi_alpha,"seed":settings.seed,
                                "auto_seed":settings.auto_seed,
                                "parallel_sequences":settings.parallel_sequences,"acceleration":settings.acceleration,
                                "context_overclock":settings.context_overclock,"pause_at_stage_end":settings.pause_at_stage_end,
                                "half_region_switch":settings.half_region_switch,
                                "manual_review_region_count":manual_review_regions.len(),
                                "deferred_fine_context_count":extra_fine_context.len(),
                                "stage_floor":stage_floor,"future_stage_context_count":pending_stage_context.len(),
                                "redraw_regions":redraw_regions,
                                "diagnostic_limit":diagnostic_limit}));
                            let Some(engine) = engine.as_mut() else {
                                emit(Event::Failed("请先加载模型".into()));
                                continue;
                            };
                            engine.set_gpu_sampling_override(match settings.sampling_backend {
                                SamplingBackend::Auto => None,
                                SamplingBackend::Gpu => Some(true),
                                SamplingBackend::Cpu => Some(false),
                            });
                            engine.set_acceleration_preferences(settings.acceleration.clone());
                            // Fixed-work diagnostics retain production's unlimited
                            // cache forecast; only their output count is bounded.
                            // Otherwise a short diagnostic could select a different
                            // KV format and misattribute that difference to the UI.
                            engine.set_diagnostic_cache_forecast(
                                diagnostic_limit.map(|_| usize::MAX),
                            );
                            let native: Result<Vec<_>, _> = primitives
                                .iter()
                                .map(|p| {
                                    p.to_tokens().map(|t| crate::inference::InferencePrimitive {
                                        tokens: t.map(u32::from),
                                    })
                                })
                                .collect();
                            let native = match native {
                                Ok(v) => v,
                                Err(e) => {
                                    emit(Event::Failed(e));
                                    continue;
                                }
                            };
                            let request = crate::inference::GenerationRequest {
                                prompt,
                                primitives: native,
                                history_records: history,
                                manual_review_regions,
                                extra_fine_context,
                                stage_floor,pending_stage_context,redraw_regions,
                                settings: crate::inference::InferenceSettings {
                                    cfg_scale: settings.cfg as f64,
                                    renyi_alpha: settings.renyi_alpha as f64,
                                    seed: settings.seed,
                                    max_primitives: generation_limit(diagnostic_limit),
                                    context_overclock: settings.context_overclock,
                                    pause_at_stage_end: settings.pause_at_stage_end,
                                    half_region_switch: settings.half_region_switch,
                                },
                            };
                            if settings.parallel_sequences > 1 {
                                match engine.generate_parallel(
                                    &request,
                                    settings.parallel_sequences,
                                    &worker_stop,
                                    |lane, p| {
                                        emit(Event::ParallelPrimitive {
                                            lane,
                                            tokens: p.primitive.tokens,
                                            history: p.history_records,
                                        })
                                    },
                                    |s| emit(Event::Status(s)),
                                    |lane, region| emit(Event::ParallelRegion { lane, region }),
                                    |lane, summary| {
                                        emit(Event::ParallelLaneFinished { lane, summary })
                                    },
                                ) {
                                    Ok(summary) => {
                                        if diagnostic_limit.is_some() { emit(Event::Status(engine.primitive_graph_status())); }
                                        emit(Event::ParallelFinished {
                                        lanes: summary.lanes,
                                        generated: summary.generated,
                                        stopped: summary.stopped,
                                        paused_after_stage: summary.paused_after_stage,
                                        elapsed_seconds: summary.elapsed_seconds,
                                        });
                                    },
                                    Err(error) => {
                                        emit(Event::Failed(format!("并行生成已停止：{}", engine.execution_error_message(&error))))
                                    }
                                }
                                let actual=engine.diagnostic_info();
                                crate::diagnostics::event("generation_context_window",serde_json::json!({
                                    "context_window":actual.get("context_window"),
                                    "kv_int8":actual.get("kv_int8"),
                                    "parallel_sequences":settings.parallel_sequences,
                                    "context_overclock":settings.context_overclock,
                                }));
                                continue;
                            }
                            match engine.generate_with_regions(
                                &request,
                                &worker_stop,
                                |p| {
                                    emit(Event::Primitive {
                                        tokens: p.primitive.tokens,
                                        history: p.history_records,
                                    })
                                },
                                |s| emit(Event::Status(s)),
                                |region| emit(Event::Region(region)),
                            ) {
                                Ok(summary) => {
                                    if diagnostic_limit.is_some() { emit(Event::Status(engine.primitive_graph_status())); }
                                    emit(Event::Finished {
                                    generated: summary.generated,
                                    stopped: summary.stopped,
                                    paused_after_stage: summary.paused_after_stage,
                                    context_limit: summary.context_limit,
                                    completed: summary.completed,
                                    history: summary.history_records,
                                    elapsed_seconds: summary.elapsed_seconds,
                                    });
                                },
                                Err(e) => emit(Event::Failed(format!("生成已停止：{}", engine.execution_error_message(&e)))),
                            }
                            let actual=engine.diagnostic_info();
                            crate::diagnostics::event("generation_context_window",serde_json::json!({
                                "context_window":actual.get("context_window"),
                                "kv_int8":actual.get("kv_int8"),
                                "parallel_sequences":1,"context_overclock":settings.context_overclock,
                            }));
                        }
                    }
                }
                crate::diagnostics::event("worker_shutdown",serde_json::json!({"completed":true}));
                crate::diagnostics::flush_summary();
            })
            .expect("Cannot create inference worker");
        Self {
            sender,
            events,
            stop,
            worker: Some(worker),
        }
    }
    pub fn load(&self, model: PathBuf, tokenizer: PathBuf) {
        self.load_with_settings(model, tokenizer, Settings::default());
    }
    pub fn load_with_settings(&self, model: PathBuf, tokenizer: PathBuf, settings: Settings) {
        let _ = self.sender.send(Command::Load { model, tokenizer, settings });
    }
    pub fn generate(
        &self,
        prompt: String,
        settings: Settings,
        primitives: Vec<Primitive>,
        history: Option<RawHistory>,
    ) {
        self.enqueue_generation(prompt, settings, primitives, history, Vec::new(),Vec::new(),None,Vec::new(),Vec::new(), None);
    }
    /// Explicit fixed-work entry point for tests and comparable benchmarks.
    /// This is never selected by the ordinary Generate/Continue UI action.
    pub fn generate_for_diagnostics(
        &self,
        prompt: String,
        settings: Settings,
        primitives: Vec<Primitive>,
        history: Option<RawHistory>,
        limit: usize,
    ) {
        self.enqueue_generation(prompt, settings, primitives, history, Vec::new(),Vec::new(),None,Vec::new(),Vec::new(), Some(limit));
    }
    pub fn generate_with_manual_reviews(&self, prompt:String, settings:Settings,
        primitives:Vec<Primitive>, history:Option<RawHistory>,
        manual_review_regions:Vec<(usize,usize)>, diagnostic_limit:Option<usize>) {
        self.enqueue_generation(prompt,settings,primitives,history,manual_review_regions,Vec::new(),None,Vec::new(),Vec::new(),diagnostic_limit);
    }
    pub fn generate_with_stage_context(&self,prompt:String,settings:Settings,primitives:Vec<Primitive>,
        history:Option<RawHistory>,manual_review_regions:Vec<(usize,usize)>,extra_fine_context:Vec<RawHistoryRecord>,
        stage_floor:Option<usize>,pending_stage_context:Vec<PendingStageRecord>,diagnostic_limit:Option<usize>){
        self.enqueue_generation(prompt,settings,primitives,history,manual_review_regions,extra_fine_context,stage_floor,pending_stage_context,Vec::new(),diagnostic_limit);
    }
    pub fn generate_redraw(&self,prompt:String,settings:Settings,primitives:Vec<Primitive>,history:RawHistory,
        stage_floor:Option<usize>,pending_stage_context:Vec<PendingStageRecord>,redraw_regions:Vec<(usize,usize)>,diagnostic_limit:Option<usize>){
        self.enqueue_generation(prompt,settings,primitives,Some(history),Vec::new(),Vec::new(),stage_floor,pending_stage_context,redraw_regions,diagnostic_limit);
    }
    fn enqueue_generation(
        &self,
        prompt: String,
        settings: Settings,
        primitives: Vec<Primitive>,
        history: Option<RawHistory>,
        manual_review_regions: Vec<(usize, usize)>,
        extra_fine_context:Vec<RawHistoryRecord>,
        stage_floor:Option<usize>,pending_stage_context:Vec<PendingStageRecord>,
        redraw_regions:Vec<(usize,usize)>,
        diagnostic_limit: Option<usize>,
    ) {
        self.stop.store(false, Ordering::Release);
        let _ = self.sender.send(Command::Generate {
            prompt,
            settings,
            primitives,
            history,
            manual_review_regions,
            extra_fine_context,
            stage_floor,pending_stage_context,redraw_regions,
            diagnostic_limit,
        });
    }
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
}

fn observe_diagnostics(event:&Event){
    use serde_json::json;
    match event {
        Event::CompatibilityNotice(notice)=>crate::diagnostics::event("compatibility_notice_ready",json!({"notice":notice})),
        Event::Status(message) if !message.starts_with("整图元Graph") &&
            ["推理路径","采样后端","Auto","优化","普通兼容","兼容计算","GPU基础路径"].iter().any(|p|message.starts_with(p)) =>
                crate::diagnostics::event("execution_decision",json!({"message":message})),
        Event::AccelerationPreparing{message}=>crate::diagnostics::event("acceleration_preparing",json!({"message":message})),
        Event::AccelerationPrepared{report}=>crate::diagnostics::event("acceleration_prepared",json!({"report":report})),
        Event::Loaded{backend,description}=>crate::diagnostics::event("model_ready",json!({"backend":backend,"description":description})),
        Event::Finished{generated,stopped,paused_after_stage,context_limit,completed,elapsed_seconds,..}=>
            crate::diagnostics::event("generation_finished",json!({"generated":generated,"stopped":stopped,"paused_after_stage":paused_after_stage,"context_limit":context_limit,"completed":completed,"elapsed_seconds":elapsed_seconds})),
        Event::ParallelLaneFinished{lane,summary}=>crate::diagnostics::event("generation_lane_finished",json!({
            "lane":lane,"generated":summary.generated,"stopped":summary.stopped,"completed":summary.completed,
            "paused_after_stage":summary.paused_after_stage})),
        Event::ParallelFinished{lanes,generated,stopped,paused_after_stage,elapsed_seconds}=>crate::diagnostics::event("generation_finished",json!({"lanes":lanes,"generated":generated,"stopped":stopped,"paused_after_stage":paused_after_stage,"elapsed_seconds":elapsed_seconds})),
        Event::Failed(message)=>{
            let safe=if message.contains("词表")||message.contains("提示词"){"提示词验证失败（原始内容已省略）"}else{message.as_str()};
            crate::diagnostics::error("inference_worker",&anyhow::anyhow!("{}",safe));
        },
        _=>{},
    }
}

fn model_ready_events(notice:Option<crate::compatibility_notice::CompatibilityNotice>,description:String,backend:String)->Vec<Event>{
    let mut events=Vec::with_capacity(2);
    // Queue the modal before readiness unlocks any native editor/generation
    // action. mpsc preserves this producer's order even across GUI frames.
    if let Some(notice)=notice{events.push(Event::CompatibilityNotice(notice));}
    events.push(Event::Loaded{description,backend});events
}

fn generation_limit(diagnostic_limit: Option<usize>) -> usize {
    diagnostic_limit.unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normal_generation_has_no_primitive_cap_and_diagnostics_remain_explicitly_bounded() {
        assert_eq!(generation_limit(None), usize::MAX);
        assert_eq!(generation_limit(Some(2)), 2);
        assert_eq!(generation_limit(Some(0)), 0); // rejected by the engine, never made unlimited.
    }
    #[test]
    fn compatibility_notice_is_always_queued_before_model_readiness_unlocks_generation(){
        let notice=crate::compatibility_notice::CompatibilityNotice{current_backend:"CPU".into(),current_scheme:"CPU兼容".into(),reason:"fixture".into(),disabled_features:vec![],ai_summary:"fixture".into(),log_directory:None};
        let events=model_ready_events(Some(notice),"fixture".into(),"CPU".into());
        assert!(matches!(events.first(),Some(Event::CompatibilityNotice(_))));
        assert!(matches!(events.get(1),Some(Event::Loaded{..})));
        assert_eq!(model_ready_events(None,"fixture".into(),"GPU".into()).len(),1);
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        self.stop();
        let _ = self.sender.send(Command::Shutdown);
        // Release CUDA resources on the owning worker before process teardown;
        // otherwise a closed GUI can remain alive while DLL teardown races it.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
