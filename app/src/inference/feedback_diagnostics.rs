//! Sampled feedback benchmark with a predetermined commit schedule.
//! This includes the sampler and host result processing, while deliberately
//! excluding region decisions, image rendering and UI/history-tree work.

use super::*;
use serde_json::{json, Value};

pub struct SamplerBenchmarkCase {
    pub gpu: bool,
    pub lanes: usize,
    /// Includes primitive BOS and one real background record.
    pub prefix_records: usize,
    pub frames: usize,
    pub repeats: usize,
    pub cfg: f64,
    pub alpha: f64,
    pub fp32_kv: bool,
    pub capture: bool,
}

const BENCHMARK_SEED: u64 = 20261011;
const BENCHMARK_PROMPT: &str =
    "金发蓝眼的白皙圆脸女性以明亮动画风呈现，张嘴露舌，粉色背景可见，取景包含头部和肩颈。";

fn feedback_cache_lengths(states: &[Lane]) -> Result<Vec<usize>> {
    let mut lengths = Vec::new();
    for lane in states {
        let cache = lane.cache.as_ref().context("benchmark lane has no cache")?;
        for physical in std::iter::once(&cache.conditional).chain(cache.unconditional.iter()) {
            if physical
                .layer_lengths()?
                .iter()
                .any(|&len| len != physical.length)
            {
                bail!("benchmark cache layer lengths differ");
            }
            lengths.push(physical.length);
        }
    }
    Ok(lengths)
}

fn feedback_candidates(field: usize, full_attribute_vocab: bool) -> Result<Vec<u32>> {
    if !full_attribute_vocab {
        return field_candidates(field, false, None);
    }
    if field >= 9 {
        bail!("diagnostic attribute index must be 0..8");
    }
    Ok((OFFSETS[field]..OFFSETS[field] + model::BINS[field]).collect())
}

/// The old Linux timing fixture sampled the complete field vocabulary,
/// including unused shape bins. Keep this relaxed field-range check local to
/// that diagnostic: production canvas/history validation remains unchanged.
fn validate_feedback_field_ranges(record: &[u32; 9]) -> Result<()> {
    for field in 0..9 {
        if !(OFFSETS[field]..OFFSETS[field] + model::BINS[field]).contains(&record[field]) {
            bail!(
                "diagnostic attribute {} is outside its field vocabulary",
                field + 1
            );
        }
    }
    Ok(())
}

impl NativeEngine {
    /// Load-time CUDA selection only: isolated caches and lane-local RNGs, no
    /// document/controller callbacks and no Graph capture or user-cache replay.
    #[cfg(feature="cuda")]
    pub(crate) fn private_cuda_auto_probe(&mut self,lanes:usize,cfg:f64,alpha:f64,
        allow_fusion:bool,allow_gpu:bool)->Result<Value> {
        use candle_core::{DType,Tensor};
        use crate::acceleration::FeatureSet;
        let before_gpu=self.gpu_sampling;
        let before_graph=self.primitive_graph_enabled;
        self.gpu_sampling=false;self.primitive_graph_enabled=false;
        self.model.prepare_optional_kernels(false,false)?;
        let fp32=self.model.current_fp32_kv();
        let result=(||->Result<Value>{
            let text=self.tokenize(BENCHMARK_PROMPT)?;
            let normal=[510,1500,2049,2500,4500,6415,6700,6800,7000];
            let mut background=normal;
            background[..5].copy_from_slice(&[0,1024,2048,2318,4366]);background[5]=OFFSETS[5];
            let mut prefix=vec![BOS_RECORD,background];prefix.resize(256,normal);
            let stop=AtomicBool::new(false);
            let prototype=self.prefill(&prefix,&text,cfg,&stop,&mut |_|{})
                .context("CUDA普通实际模型私有预填充验证失败")?
                .context("CUDA私有测速预填充未完成")?;
            let guidance=if (cfg-1.).abs()>1e-12{2}else{1};
            let physical=lanes*guidance;
            let required=(self.model.architecture.layers as u64)
                .saturating_mul(self.model.architecture.width as u64).saturating_mul(512)
                .saturating_mul((physical+guidance) as u64).saturating_mul(if fp32{8}else{4})
                .saturating_add(512u64<<20);
            let context=self.model.device.as_cuda_device()?.cuda_stream();
            if let Ok((free,_))=context.context().mem_get_info(){
                if (free as u64)<required {
                    return Ok(json!({"status":"skipped","reason":"可用显存不足以安全完成本批量私有对照，保留普通路径",
                        "required_free_bytes":required,"observed_free_bytes":free,"features":FeatureSet::default()}));
                }
            }
            let prediction=|p:&Prediction|->Result<model::DevicePrediction>{
                Ok(model::DevicePrediction{
                    attributes:Tensor::from_vec(p.attributes.clone(),(1,p.attributes.len()),&self.model.device)?,
                    termination:Some(Tensor::from_vec(p.termination.to_vec(),(1,2),&self.model.device)?),
                    attribute_offset:p.attribute_offset,
                })
            };
            let mut gpu_prototype=prototype.clone();
            if allow_gpu {
                let c=prediction(&prototype.conditional_pending)?;
                let u=prototype.unconditional_pending.as_ref().map(prediction).transpose()?;
                let mut pieces=vec![&c];if let Some(u)=&u{pieces.push(u);}
                let device_prediction=model::DevicePrediction::concat_rows(&pieces)?;
                let mut terms=vec![prototype.conditional_pending.termination];
                if let Some(u)=&prototype.unconditional_pending{terms.push(u.termination);}
                gpu_prototype.install_device(device_prediction,&terms)?;
            }
            let mut modes=vec![("ordinary",false,false)];
            if allow_fusion{modes.push(("fusion_cpu",true,false));}
            if allow_gpu{modes.push(("gpu_sampling",false,true));}
            if allow_fusion&&allow_gpu{modes.push(("fusion_gpu",true,true));}
            let mut times=vec![Vec::<f64>::new();modes.len()];
            let mut errors=vec![None::<String>;modes.len()];
            let settings=InferenceSettings{cfg_scale:cfg,renyi_alpha:alpha,seed:BENCHMARK_SEED,
                max_primitives:3,context_overclock:false,half_region_switch:false,pause_at_stage_end:false};
            for round in 0..3 {
                for offset in 0..modes.len(){
                    let index=(offset+round)%modes.len();if errors[index].is_some(){continue;}
                    let (_,fusion,gpu)=modes[index];
                    self.gpu_sampling=gpu;
                    let trial=(||->Result<f64>{
                        self.model.prepare_optional_kernels(fusion,false)?;
                        let source=if gpu{&gpu_prototype}else{&prototype};
                        let mut states=(0..lanes).map(|index|Ok(Lane{index,full:prefix.clone(),
                            history:parse_history(&prefix)?,rng:StdRng::seed_from_u64(parallel_seed(BENCHMARK_SEED,index)),
                            generated:0,cache:Some(source.clone()),raw_cursor:prefix.len(),summary:None,pause_target:None,manual_review_regions:HashSet::new(),fine_aux_seen:false,pending_stage_context:Vec::new(),activated_stages:HashSet::new(),redraw_scope:None})).collect::<Result<Vec<_>>>()?;
                        self.sampling_feedback_frame(&mut states,&settings,&stop,false)?;
                        self.model.device.synchronize()?;
                        let started=Instant::now();
                        for _ in 0..2{self.sampling_feedback_frame(&mut states,&settings,&stop,false)?;}
                        self.model.device.synchronize()?;
                        Ok(started.elapsed().as_secs_f64())
                    })();
                    match trial {
                        Ok(seconds) if seconds.is_finite()&&seconds>0.=>times[index].push(seconds),
                        Ok(_)=>errors[index]=Some("私有测速计时无效".into()),
                        Err(error)=>{
                            let message=format!("{error:#}");
                            if uncertain_device_execution(&message){self.poison_acceleration(message.clone());return Err(error);}
                            if index==0{return Err(error.context("CUDA普通实际模型前向未通过验证"));}
                            if known_safe_optional_gpu_failure(&message) {
                                errors[index]=Some(message);
                            }else{
                                let reason=format!("EXECUTION_UNCERTAIN: CUDA私有候选验证发生未分类异常，已停止GPU提交：{message}");
                                self.poison_acceleration(reason.clone());
                                bail!(reason);
                            }
                        }
                    }
                    // Private row owners have now dropped; flush asynchronous
                    // frees before measuring another candidate or memory plan.
                    self.model.device.synchronize()?;
                }
            }
            let median=|values:&[f64]|->Option<f64>{if values.len()!=3{return None;}let mut v=values.to_vec();v.sort_by(f64::total_cmp);Some(v[1])};
            let baseline=median(&times[0]).context("CUDA普通模型缺少完整私有对照")?;
            let mut best=0usize;let mut best_time=baseline;
            for i in 1..modes.len(){if let Some(value)=median(&times[i]){if value<best_time{best=i;best_time=value;}}}
            if best_time*1.05>=baseline{best=0;best_time=baseline;}
            let selected=FeatureSet{fusion:modes[best].1,gpu_sampling:modes[best].2,split_k:false,primitive_graph:false};
            let candidates=modes.iter().enumerate().map(|(i,(name,fusion,gpu))|json!({
                "name":name,"fusion":fusion,"gpu_sampling":gpu,"seconds":times[i],"error":errors[i],
                "median_attribute_tps":median(&times[i]).map(|s|2.*lanes as f64*9./s),
            })).collect::<Vec<_>>();
            Ok(json!({"status":"complete","scope":"private actual model, 256 synthetic legal prefix records, no document/RNG/Graph",
                "logical_lanes":lanes,"guidance":guidance,"cfg":cfg,"alpha":alpha,"fp32_kv":fp32,
                "prefix_records":prefix.len(),"text_tokens":text.len(),"warmup_frames":1,"timed_frames":2,"rounds":3,
                "baseline_attribute_tps":2.*lanes as f64*9./baseline,"selected_attribute_tps":2.*lanes as f64*9./best_time,
                "features":selected,"candidates":candidates,"no_image_quality_claim":true}))
        })();
        self.gpu_sampling=before_gpu;self.primitive_graph_enabled=before_graph;
        self.model.prepare_optional_kernels(false,false)?;
        if let Err(error)=&result{if uncertain_device_execution(&format!("{error:#}")){self.poison_acceleration(format!("{error:#}"));}}
        result
    }

    fn sampling_feedback_frame(
        &self,
        states: &mut [Lane],
        settings: &InferenceSettings,
        stop: &AtomicBool,
        full_attribute_vocab: bool,
    ) -> Result<(Vec<[u32; 9]>, usize)> {
        let ready = vec![true; states.len()];
        let (rows, queries) = if self.gpu_sampling {
            let (rows, queries, physical) = self
                .parallel_device_frame(states, &ready, settings, stop)?
                .context("benchmark GPU frame unexpectedly stopped")?;
            let guidance = if (settings.cfg_scale - 1.).abs() > 1e-12 {
                2
            } else {
                1
            };
            if physical != states.len() * guidance {
                bail!("benchmark GPU frame lost physical rows");
            }
            (rows, queries)
        } else if states.len() == 1 && !full_attribute_vocab {
            let lane = &mut states[0];
            let row = self
                .next_primitive(
                    lane.cache
                        .as_mut()
                        .context("missing single benchmark cache")?,
                    settings,
                    false,
                    None,
                    stop,
                    &mut lane.rng,
                )?
                .context("benchmark CPU frame unexpectedly stopped")?;
            (vec![row], 9)
        } else {
            let mut rows = vec![PARTIAL; states.len()];
            let mut queries = 0;
            for field in 0..9 {
                if field > 0 {
                    self.parallel_query(states, &rows, &ready, false, OutputScope::Field(field))?;
                    queries += 1;
                }
                // Every benchmark row is an ordinary global primitive, so
                // candidate count never depends on an earlier sampled token.
                let candidates = feedback_candidates(field, full_attribute_vocab)?;
                for lane in states.iter_mut() {
                    let cache = lane.cache.as_ref().context("missing CPU benchmark cache")?;
                    let logits: Vec<f64> = candidates
                        .iter()
                        .map(|&id| {
                            let c = cache.conditional_pending.attribute(id)?;
                            Ok(if let Some(unconditional) = &cache.unconditional_pending {
                                let u = unconditional.attribute(id)?;
                                u + settings.cfg_scale * (c - u)
                            } else {
                                c
                            })
                        })
                        .collect::<Result<_>>()?;
                    let index = renyi_uniform_index(&logits, settings.renyi_alpha, &mut lane.rng)?;
                    rows[lane.index][field] = candidates[index];
                }
            }
            for row in &rows {
                if full_attribute_vocab {
                    validate_feedback_field_ranges(row)?;
                } else {
                    validate_visible(row)?;
                }
            }
            self.parallel_query(states, &rows, &ready, true, OutputScope::NextPrimitive)?;
            (rows, queries + 1)
        };
        if rows.len() != states.len() || queries != 9 {
            bail!("ordinary sampled feedback must complete all lanes with nine queries");
        }
        for (lane, row) in states.iter_mut().zip(&rows) {
            if full_attribute_vocab {
                validate_feedback_field_ranges(row)?;
                if row[5] <= OFFSETS[5] + 2 {
                    validate_visible(row)?;
                }
            } else {
                validate_visible(row)?;
                if row[0] > 1022
                    || row[1] > OFFSETS[1] + 1022
                    || ![OFFSETS[5] + 1, OFFSETS[5] + 2].contains(&row[5])
                {
                    bail!("sampled feedback escaped its fixed global legal range");
                }
            }
            // Include the same small host record handling for both paths.
            // These records are not displayed as editor history nodes.
            lane.full.push(*row);
            lane.history.fixed.push(*row);
            lane.generated += 1;
        }
        Ok((rows, queries))
    }

    pub fn sampling_feedback_diagnostics(&mut self, case: &SamplerBenchmarkCase) -> Result<Value> {
        self.sampling_feedback_diagnostics_impl(case, None)
    }

    /// Isolated, bounded feedback timing using the historical Linux inputs.
    /// This does not enter production generation or create canvas/history nodes.
    /// Full-vocabulary matching has a separately keyed diagnostic workspace;
    /// the production sampler's ordinary-shape constraints remain unchanged.
    pub fn sampling_feedback_diagnostics_from_inputs(
        &mut self,
        case: &SamplerBenchmarkCase,
        prefix: &[[u32; 9]],
        text: &[u32],
    ) -> Result<Value> {
        self.sampling_feedback_diagnostics_impl(case, Some((prefix, text)))
    }

    fn sampling_feedback_diagnostics_impl(
        &mut self,
        case: &SamplerBenchmarkCase,
        supplied_inputs: Option<(&[[u32; 9]], &[u32])>,
    ) -> Result<Value> {
        if case.prefix_records < 2
            || case.prefix_records > 897
            || case.frames == 0
            || case.repeats == 0
            || !case.cfg.is_finite()
            || case.cfg < 0.
            || !case.alpha.is_finite()
            || case.alpha <= 1.
        {
            bail!("invalid sampled-feedback benchmark parameters");
        }
        let full_attribute_vocab = supplied_inputs.is_some();
        #[cfg(not(feature = "cuda"))]
        if case.capture {
            bail!("CUDA profiler capture requires the CUDA build");
        }
        let timed_frames = case
            .frames
            .checked_mul(case.repeats)
            .context("benchmark frame overflow")?;
        let growth = timed_frames
            .checked_add(2)
            .context("benchmark warm-up overflow")?;
        let (prefix, text) = if let Some((prefix, text)) = supplied_inputs {
            if prefix.len() != case.prefix_records {
                bail!(
                    "real prefix has {} records, expected {}",
                    prefix.len(),
                    case.prefix_records
                );
            }
            if text.is_empty()
                || text.len() > 224
                || text.iter().any(|&id| id >= 1500)
                || text.first() != Some(&2)
                || text.last() != Some(&3)
            {
                bail!("real text must contain valid 1500-vocabulary BOS/EOS IDs within 224 tokens");
            }
            // Markers/true_end are actual model context records, not visible
            // canvas primitives. Validate their full control grammar instead.
            parse_history(prefix)?;
            (prefix.to_vec(), text.to_vec())
        } else {
            let normal = [510, 1500, 2049, 2500, 4500, 6415, 6700, 6800, 7000];
            let mut background = normal;
            background[..5].copy_from_slice(&[0, 1024, 2048, 2318, 4366]);
            background[5] = OFFSETS[5];
            let mut prefix = vec![BOS_RECORD, background];
            prefix.extend(std::iter::repeat(normal).take(case.prefix_records - 2));
            for record in prefix.iter().skip(1) {
                validate_visible(record)?;
            }
            (prefix, self.tokenize(BENCHMARK_PROMPT)?)
        };
        if case
            .prefix_records
            .checked_add(text.len())
            .and_then(|n| n.checked_add(growth))
            .is_none_or(|n| n > MAX_CONTEXT)
        {
            bail!("sampled feedback would exceed the fixed 1024-position context");
        }
        let guidance = if (case.cfg - 1.).abs() > 1e-12 { 2 } else { 1 };
        let physical_rows = model::checked_batch_rows(case.lanes, guidance)?;
        let old_gpu = self.gpu_sampling;
        let old_full_vocab = self.diagnostic_full_attribute_vocab;
        let old_fp32 = self.model.set_diagnostic_fp32_kv(case.fp32_kv);
        let old_metadata = self.model.set_diagnostic_metadata_legacy(physical_rows < 4);
        self.gpu_sampling = case.gpu;
        self.diagnostic_full_attribute_vocab = full_attribute_vocab;
        let result = (|| -> Result<Value> {
            let stop = AtomicBool::new(false);
            self.model.device.synchronize()?;
            let prepare_start = Instant::now();
            let prototype = self
                .prefill(&prefix, &text, case.cfg, &stop, &mut |_| {})?
                .context("sampled-feedback prefill unexpectedly stopped")?;
            let initial_host_prediction = if full_attribute_vocab && self.gpu_sampling {
                Some(prototype.device_pending.as_ref().context("missing diagnostic GPU head")?
                    .to_host()?.remove(0))
            } else { None };
            let initial = initial_host_prediction.as_ref().unwrap_or(&prototype.conditional_pending);
            let initial_prefill_prediction = full_attribute_vocab.then(|| json!({
                "scope":"conditional field0 prediction before warm-up; same actual prefix/text as the historical reference",
                "attribute_offset":initial.attribute_offset,
                "attribute_logits":initial.attributes.clone(),
                "termination_logits":initial.termination,
                "kv_length":prototype.conditional.length,
            }));
            let mut states = Vec::new();
            states.try_reserve_exact(case.lanes).context("cannot allocate requested benchmark lane states")?;
            for index in 0..case.lanes {
                states.push(Lane {
                    index,
                    full: prefix.clone(),
                    history: parse_history(&prefix)?,
                    rng: StdRng::seed_from_u64(parallel_seed(BENCHMARK_SEED, index)),
                    generated: 0,
                    cache: Some(prototype.clone()),
                    raw_cursor: prefix.len(),
                    summary: None,
                    pause_target: None,
                    manual_review_regions: HashSet::new(),
                    fine_aux_seen:false,
                    pending_stage_context:Vec::new(),activated_stages:HashSet::new(),redraw_scope:None,
                });
            }
            drop(prototype);
            self.model.device.synchronize()?;
            let prefill_seconds = prepare_start.elapsed().as_secs_f64();
            let start_lengths = feedback_cache_lengths(&states)?;
            if start_lengths.len() != physical_rows {
                bail!("benchmark prefill has unexpected physical rows");
            }
            let settings = InferenceSettings {
                half_region_switch: false, pause_at_stage_end: false,
                cfg_scale: case.cfg,
                renyi_alpha: case.alpha,
                seed: BENCHMARK_SEED,
                max_primitives: growth,
                context_overclock: false,
            };
            let mut warmup_tokens = vec![Vec::new(); case.lanes];
            for _ in 0..2 {
                let (rows, _) = self.sampling_feedback_frame(
                    &mut states,
                    &settings,
                    &stop,
                    full_attribute_vocab,
                )?;
                for (lane, row) in rows.into_iter().enumerate() {
                    warmup_tokens[lane].push(row);
                }
            }
            self.model.device.synchronize()?;
            let timed_start_lengths = feedback_cache_lengths(&states)?;
            let mut tokens = vec![Vec::with_capacity(timed_frames); case.lanes];
            let mut samples = Vec::with_capacity(case.repeats);
            let mut timed_queries = 0usize;
            for repeat in 0..case.repeats {
                self.model.device.synchronize()?;
                #[cfg(feature = "cuda")]
                let profiler = if case.capture {
                    Some(candle_core::cuda_backend::cudarc::driver::Profiler::new()?)
                } else {
                    None
                };
                let started = Instant::now();
                for _ in 0..case.frames {
                    let (rows, queries) = self.sampling_feedback_frame(
                        &mut states,
                        &settings,
                        &stop,
                        full_attribute_vocab,
                    )?;
                    timed_queries += queries;
                    for (lane, row) in rows.into_iter().enumerate() {
                        tokens[lane].push(row);
                    }
                }
                self.model.device.synchronize()?;
                let seconds = started.elapsed().as_secs_f64();
                #[cfg(feature = "cuda")]
                drop(profiler);
                samples.push(json!({
                    "repeat":repeat,"seconds":seconds,"complete_primitives":case.frames*case.lanes,
                    "milliseconds_per_primitive_batch":seconds*1000./case.frames as f64,
                    "aggregate_complete_primitives_per_second":case.frames as f64*case.lanes as f64/seconds,
                    "queries":case.frames*9,
                }));
            }
            let end_lengths = feedback_cache_lengths(&states)?;
            if end_lengths
                .iter()
                .zip(&start_lengths)
                .any(|(end, start)| *end != *start + growth)
                || timed_queries != timed_frames * 9
                || states.iter().any(|lane| lane.generated != growth)
            {
                bail!("sampled-feedback history/query schedule changed unexpectedly");
            }
            let timed_seconds: f64 = samples
                .iter()
                .map(|sample| sample["seconds"].as_f64().unwrap())
                .sum();
            let native_shapes = tokens
                .iter()
                .flatten()
                .filter(|row| row[5] <= OFFSETS[5] + 2)
                .count();
            let ordinary_global_legal = tokens
                .iter()
                .flatten()
                .filter(|row| {
                    row[0] <= 1022
                        && row[1] <= OFFSETS[1] + 1022
                        && [OFFSETS[5] + 1, OFFSETS[5] + 2].contains(&row[5])
                })
                .count();
            let candidate_counts = (0..9)
                .map(|field| feedback_candidates(field, full_attribute_vocab).map(|ids| ids.len()))
                .collect::<Result<Vec<_>>>()?;
            let mut report = json!({
                "status":"complete",
                "scope":"Sampled CPU/GPU feedback + nine attribute queries + complete commit + host record handling. Global legal region and commit count predetermined; termination decisions, region selection, rendering and UI excluded. NOT complete-image TPS or model-only timing.",
                "sampler":self.sampler_name(),"device":self.model.architecture.device,
                "logical_lanes":case.lanes,"physical_rows":physical_rows,"cfg":case.cfg,"renyi_alpha":case.alpha,
                "prefix_records":case.prefix_records,"text_tokens":text.len(),"seed":BENCHMARK_SEED,
                "lane_seeds":(0..case.lanes).map(|lane|parallel_seed(BENCHMARK_SEED,lane)).collect::<Vec<_>>(),
                "fp32_main_kv":case.fp32_kv,"metadata_legacy":physical_rows<4,
                "guided_prefill_calls":1,"physical_prefill_rows":guidance,"shared_initial_prefix":true,
                "prefill_seconds":prefill_seconds,"warmup_frames":2,"frames_per_repeat":case.frames,
                "repeats":case.repeats,"timed_queries":timed_queries,"queries_per_complete_primitive_batch":9,
                "controlled_history_growth":true,"history_growth":growth,
                "kv_start_lengths":start_lengths,"kv_timed_start_lengths":timed_start_lengths,"kv_end_lengths":end_lengths,
                "region_decisions":0,"termination_decisions":0,"host_result_processing_included":true,
                "timed_seconds":timed_seconds,"timed_complete_primitives":timed_frames*case.lanes,
                "aggregate_complete_primitives_per_second":timed_frames as f64*case.lanes as f64/timed_seconds,
                "milliseconds_per_primitive_batch":1000.*timed_seconds/timed_frames as f64,
                "profile_instrumented":case.capture,"samples":samples,
                "warmup_tokens_by_lane":warmup_tokens,"tokens_by_lane":tokens,
                "seed_mapping_note":"Lane-local seeds; CPU f64/random-range and GPU f32/uniform-rank sampler versions need not produce identical tokens.",
                "synthetic_prefix_no_image_quality_claim":!full_attribute_vocab,
            });
            // Keep supplemental diagnostics outside the original report macro
            // so all production bins compile with the default recursion limit.
            report["gpu_sampling_active"] = json!(self.gpu_sampling);
            report["full_attribute_vocab"] = json!(full_attribute_vocab);
            report["candidate_counts_by_field"] = json!(candidate_counts);
            report["input_origin"] = json!(if full_attribute_vocab {
                "provided real control/primitive prefix and text IDs"
            } else {
                "legacy synthetic prefix and built-in prompt"
            });
            report["field_range_legal_generated_count"] = json!(timed_frames * case.lanes);
            report["native_shape_legal_generated_count"] = json!(native_shapes);
            report["non_native_shape_generated_count"] =
                json!(timed_frames * case.lanes - native_shapes);
            report["ordinary_global_legal_generated_count"] = json!(ordinary_global_legal);
            report["no_image_quality_claim"] = json!(true);
            report["reset_prefix_between_repeats"] = json!(false);
            report["historical_window_comparison_note"] = json!(if full_attribute_vocab {
                "Shared input prefix/text/full attribute vocabulary, but this native diagnostic preserves its original two warm-up frames and continuous KV growth across repeats. The old Linux benchmark reset its 512-record prefix before each measured 32-primitive window; these context-length schedules are not identical. CPU f64 sampling here is not the old GPU f32 sampler."
            } else {
                "Legacy synthetic diagnostic schedule retained."
            });
            if let Some(prediction) = initial_prefill_prediction {
                report["initial_prefill_field0"] = prediction;
            }
            Ok(report)
        })();
        self.gpu_sampling = old_gpu;
        self.diagnostic_full_attribute_vocab = old_full_vocab;
        self.model.set_diagnostic_fp32_kv(old_fp32);
        self.model.set_diagnostic_metadata_legacy(old_metadata);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_candidates_keep_full_fields_without_changing_ordinary_constraints() {
        assert_eq!(feedback_candidates(0, true).unwrap().len(), 1024);
        assert_eq!(feedback_candidates(0, false).unwrap().len(), 1023);
        assert_eq!(feedback_candidates(1, true).unwrap().len(), 1024);
        assert_eq!(feedback_candidates(5, true).unwrap().len(), 256);
        assert_eq!(
            feedback_candidates(5, false).unwrap(),
            vec![OFFSETS[5] + 1, OFFSETS[5] + 2]
        );
        assert!(feedback_candidates(9, true).is_err());
    }

    #[test]
    fn full_vocabulary_diagnostic_does_not_relax_canvas_shape_validation() {
        let ordinary = [510, 1500, 2049, 2500, 4500, 6415, 6700, 6800, 7000];
        validate_visible(&ordinary).unwrap();
        validate_feedback_field_ranges(&ordinary).unwrap();
        let mut unused_shape = ordinary;
        unused_shape[5] = OFFSETS[5] + 99;
        validate_feedback_field_ranges(&unused_shape).unwrap();
        assert!(validate_visible(&unused_shape).is_err());
        let mut outside_field = ordinary;
        outside_field[6] = OFFSETS[6] + model::BINS[6];
        assert!(validate_feedback_field_ranges(&outside_field).is_err());
    }
}
