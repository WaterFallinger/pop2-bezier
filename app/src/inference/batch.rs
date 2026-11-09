//! Independent creative lanes with genuinely batched attribute decoding.
//! Identical context preparation is shared; divergent regions remain independent.

use super::*;
#[path = "feedback_diagnostics.rs"]
mod feedback_diagnostics;
pub use feedback_diagnostics::SamplerBenchmarkCase;

pub fn parallel_seed(base: u64, lane: usize) -> u64 {
    base.wrapping_add(lane as u64)
}

#[derive(Clone, Debug)]
pub struct ParallelSummary {
    pub lanes: usize,
    pub generated: usize,
    pub stopped: bool,
    pub paused_after_stage: Option<Stage>,
    pub elapsed_seconds: f64,
    pub lane_summaries: Vec<GenerationSummary>,
    pub batched_decode_calls: usize,
    pub maximum_tensor_batch: usize,
    pub prefill_strategy: String,
    /// Guided-context prefill routine calls, not individual 64-record chunks.
    pub actual_prefill_calls: usize,
    pub prefill_reuse_hits: usize,
    /// The legacy single-lane path reports zeros without claiming measurement.
    pub prefill_statistics_tracked: bool,
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

struct Lane {
    index: usize,
    full: Vec<[u32; 9]>,
    history: History,
    rng: StdRng,
    generated: usize,
    cache: Option<GuidedCache>,
    /// Index into complete raw history already incorporated into visual KV.
    raw_cursor: usize,
    summary: Option<GenerationSummary>,
    pause_target: Option<Stage>,
    manual_review_regions: HashSet<(usize,usize)>,
    fine_aux_seen:bool,
    pending_stage_context:Vec<PendingStageRecord>,
    activated_stages:HashSet<usize>,
    redraw_scope: Option<RedrawScope>,
}

impl Lane {
    fn is_global(&self) -> bool {
        global_incomplete(&self.history)
    }
    fn background(&self) -> bool {
        self.is_global() && self.history.fixed.len() == 1
    }
    fn region(&self) -> Option<(usize, usize)> {
        self.history.active.as_ref().map(|r| (r.side, r.index))
    }
}

impl NativeEngine {
    /// Real-weight batched/single reference check. Four physical rows represent
    /// two logical conditional/unconditional pairs with unequal history lengths.
    /// This does not benchmark speed or claim final-image quality.
    pub fn parallel_reference_probe(&self, prompt: &str) -> Result<serde_json::Value> {
        let text = self.tokenize(prompt)?;
        let ordinary = [510, 1500, 2049, 2500, 4500, 6415, 6700, 6800, 7000];
        let mut caches = Vec::new();
        for row in 0..4 {
            let unconditional = row % 2 == 1;
            let mut cache = if unconditional {
                self.model.new_cache_offset(text.len() - 2)
            } else {
                self.model.new_cache()
            };
            self.model
                .prefill_text(if unconditional { &[2, 3] } else { &text }, &mut cache)?;
            self.model.forward_record(&BOS_RECORD, &mut cache, true)?;
            for _ in 0..if row < 2 { 1 } else { 3 } {
                self.model.forward_record(&ordinary, &mut cache, true)?;
            }
            caches.push(cache);
        }
        let lengths_before: Vec<_> = caches.iter().map(|c| c.length).collect();
        let positions_before: Vec<_> = caches.iter().map(|c| c.absolute_position()).collect();
        let mut reference = caches.clone();
        let mut partial = PARTIAL;
        partial[0] = 111;
        let partials = [partial; 4];
        let expected_partial: Vec<_> = reference
            .iter_mut()
            .map(|c| self.model.forward_record(&partial, c, false))
            .collect::<Result<_>>()?;
        let mut refs: Vec<_> = caches.iter_mut().collect();
        let actual_partial = self
            .model
            .forward_records_batched(&partials, &mut refs, false)?;
        drop(refs);
        let partial_error = prediction_error(&actual_partial, &expected_partial)?;
        let partial_did_not_commit = caches.iter().zip(&lengths_before).all(|(c, &n)| {
            c.length == n
                && c.layer_lengths()
                    .map(|v| v.into_iter().all(|x| x == n))
                    .unwrap_or(false)
        });
        if !partial_did_not_commit {
            bail!("批量partial查询错误地提交了KV");
        }
        let mut records = [ordinary; 4];
        records[2][0] = 620;
        records[3][0] = 620;
        let expected_committed: Vec<_> = reference
            .iter_mut()
            .zip(&records)
            .map(|(c, r)| self.model.forward_record(r, c, true))
            .collect::<Result<_>>()?;
        let mut refs: Vec<_> = caches.iter_mut().collect();
        let actual_committed = self
            .model
            .forward_records_batched(&records, &mut refs, true)?;
        drop(refs);
        let commit_error = prediction_error(&actual_committed, &expected_committed)?;
        let lengths_after: Vec<_> = caches.iter().map(|c| c.length).collect();
        for (cache, &before) in caches.iter().zip(&lengths_before) {
            if cache.length != before + 1
                || cache.layer_lengths()?.iter().any(|&n| n != cache.length)
            {
                bail!("批量commit把ragged padding写入了真实KV");
            }
        }
        let expected_next: Vec<_> = reference
            .iter_mut()
            .map(|c| self.model.forward_record(&partial, c, false))
            .collect::<Result<_>>()?;
        let mut refs: Vec<_> = caches.iter_mut().collect();
        let actual_next = self
            .model
            .forward_records_batched(&partials, &mut refs, false)?;
        drop(refs);
        let next_error = prediction_error(&actual_next, &expected_next)?;
        let mut projected_heads = Vec::new();
        for field in 0..9 {
            let mut refs: Vec<_> = caches.iter_mut().collect();
            let projected = self.model.forward_records_batched_for(
                &partials,
                &mut refs,
                false,
                OutputScope::Field(field),
            )?;
            let mut maximum = 0f64;
            let mut error = 0f64;
            let mut energy = 0f64;
            for (a, b) in projected.iter().zip(&actual_next) {
                for (index, &value) in a.attributes.iter().enumerate() {
                    let expected = b.attributes[a.attribute_offset + index] as f64;
                    let delta = value as f64 - expected;
                    maximum = maximum.max(delta.abs());
                    error += delta * delta;
                    energy += expected * expected;
                }
            }
            let rms = (error / energy.max(1e-24)).sqrt();
            if rms > 0.03 {
                bail!("projected field {field} relative RMS {rms} exceeds 0.03");
            }
            projected_heads.push(
                serde_json::json!({"field":field,"maximum_absolute":maximum,"relative_rms":rms}),
            );
        }
        let cfg_error = cfg_error(&actual_committed, &expected_committed, 2.)?;
        if partial_error.1 > 0.03
            || commit_error.1 > 0.03
            || next_error.1 > 0.03
            || cfg_error.1 > 0.05
        {
            bail!("真实批量与单序列参考差异过大：partial={partial_error:?} commit={commit_error:?} next={next_error:?} CFG={cfg_error:?}");
        }
        Ok(
            serde_json::json!({"status":"complete","scope":"real-checkpoint batched versus independent single-forward numerical parity",
            "physical_tensor_rows":4,"logical_cfg_pairs":2,"cfg_scale":2,"ragged_cache_lengths_before":lengths_before,
            "absolute_rope_positions_before":positions_before,"ragged_cache_lengths_after":lengths_after,
            "partial_no_commit":partial_did_not_commit,"persistent_kv_contains_no_padding":true,
            "partial_error":{"maximum_absolute":partial_error.0,"relative_rms":partial_error.1},
            "commit_error":{"maximum_absolute":commit_error.0,"relative_rms":commit_error.1},
            "next_partial_error":{"maximum_absolute":next_error.0,"relative_rms":next_error.1},
            "projected_heads_vs_full":projected_heads,
            "cfg2_composed_error":{"maximum_absolute":cfg_error.0,"relative_rms":cfg_error.1},
            "dtype_policy":"existing native model dtype; FP32 attention; no quantization added"}),
        )
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
        model::checked_batch_rows(lanes, if (request.settings.cfg_scale - 1.).abs() > 1e-12 { 2 } else { 1 })?;
        let started = Instant::now();
        if lanes == 1 {
            let summary = self.generate_with_regions(
                request,
                stop,
                |p| on_primitive(0, p),
                &mut on_status,
                |r| on_region(0, r),
            )?;
            on_finished(0, summary.clone());
            return Ok(ParallelSummary {
                lanes: 1,
                generated: summary.generated,
                stopped: summary.stopped,
                paused_after_stage: summary.paused_after_stage,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                lane_summaries: vec![summary],
                batched_decode_calls: 0,
                maximum_tensor_batch: 1,
                prefill_strategy: "single-lane legacy-compatible; prefill counters not tracked"
                    .into(),
                actual_prefill_calls: 0,
                prefill_reuse_hits: 0,
                prefill_statistics_tracked: false,
            });
        }
        let effective_settings = request_settings(request);
        let settings = &effective_settings;
        self.context_overclock = settings.context_overclock;
        self.fine_aux_blocks=edited_context_blocks(&request.extra_fine_context)?;
        if !settings.cfg_scale.is_finite() || settings.cfg_scale < 0. {
            bail!("CFG必须是有限非负数");
        }
        if !settings.renyi_alpha.is_finite() || settings.renyi_alpha <= 1. {
            bail!("Rényi α必须大于1且为有限数");
        }
        if settings.max_primitives == 0 {
            bail!("每序列本次生成图元上限必须大于零");
        }
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
        self.configure_generation_cache(
            lanes,
            settings.cfg_scale,
            settings.renyi_alpha,
            full.len(),
            settings.max_primitives,
            &mut on_status,
        )?;
        self.reset_context_window_stats();
        let mut states = Vec::new();
        states.try_reserve_exact(lanes).context("无法为请求的并行序列分配主机状态；未减少并行数")?;
        for lane in 0..lanes {
            let mut history = parse_request_history(&full,request.stage_floor)?;
            let redraw_scope = RedrawScope::initialize(request, &mut history)?;
            let pause_target = stage_pause_target(&history, settings.pause_at_stage_end);
            states.push(Lane {
                index: lane,
                full: full.clone(),
                history,
                rng: StdRng::seed_from_u64(parallel_seed(settings.seed, lane)),
                generated: 0,
                cache: None,
                raw_cursor: 0,
                summary: None,
                pause_target,
                manual_review_regions: manual_reviews(request)?,
                fine_aux_seen:false,
                pending_stage_context:request.pending_stage_context.clone(),activated_stages:HashSet::new(),redraw_scope,
            });
            on_region(
                lane,
                DrawingRegion::from_history(&full).map_err(anyhow::Error::msg)?,
            );
        }
        on_status(format!("正在准备 {lanes} 条并行分支 · 画布显示第1分支"));
        if !request.redraw_regions.is_empty() { on_status("局部修改：各分支保留黄色区域已有图形并续画，先粗后细；红色区域须已清空".into()); }
        if settings.context_overclock {
            on_status("上下文超频已开启：全局→粗网格→邻近细网格，保留当前完整区域的生成空间".into());
        }
        let mut batched_decode_calls = 0;
        let mut maximum_tensor_batch = 0;
        let mut actual_prefill_calls = 0;
        let mut prefill_reuse_hits = 0;
        let execution = (|| -> Result<()> {
            while states.iter().any(|lane| lane.summary.is_none()) {
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
                    .map(|lane| lane.summary.is_none() && lane.cache.is_some() && lane.raw_cursor == lane.full.len())
                    .collect();
                if !ready.iter().any(|&v| v) {
                    continue;
                }
                let mut rows = vec![PARTIAL; lanes];
                let mut interrupted = false;
                let mut device_committed = false;
                if self.gpu_sampling {
                    if let Some((generated, queries, physical)) =
                        self.parallel_device_frame(&mut states, &ready, settings, stop)?
                    {
                        rows = generated;
                        batched_decode_calls += queries;
                        maximum_tensor_batch = maximum_tensor_batch.max(physical);
                        device_committed = true;
                    } else {
                        interrupted = true;
                    }
                } else {
                    for field in 0..9 {
                        if stop.load(Ordering::Relaxed) {
                            interrupted = true;
                            break;
                        }
                        if field > 0 {
                            let query_ready: Vec<_> = states
                                .iter()
                                .map(|lane| ready[lane.index] && !(field == 5 && lane.background()))
                                .collect();
                            if query_ready.iter().any(|&v| v) {
                                let count = self.parallel_query(
                                    &mut states,
                                    &rows,
                                    &query_ready,
                                    false,
                                    OutputScope::Field(field),
                                )?;
                                batched_decode_calls += 1;
                                maximum_tensor_batch = maximum_tensor_batch.max(count);
                            }
                        }
                        for lane in states.iter_mut().filter(|lane| ready[lane.index]) {
                            let candidates =
                                field_candidates(field, lane.background(), lane.region())?;
                            if candidates.len() == 1 {
                                let index = renyi_uniform_index(
                                    &[0.],
                                    settings.renyi_alpha,
                                    &mut lane.rng,
                                )?;
                                rows[lane.index][field] = candidates[index];
                                continue;
                            }
                            let cache = lane.cache.as_ref().context("并行lane缓存丢失")?;
                            let logits: Vec<_> = candidates
                                .iter()
                                .map(|&id| {
                                    let c = cache.conditional_pending.attribute(id)?;
                                    Ok(if let Some(u) = &cache.unconditional_pending {
                                        let u = u.attribute(id)?;
                                        u + settings.cfg_scale * (c - u)
                                    } else {
                                        c
                                    })
                                })
                                .collect::<Result<_>>()?;
                            let index =
                                renyi_uniform_index(&logits, settings.renyi_alpha, &mut lane.rng)?;
                            rows[lane.index][field] = candidates[index];
                        }
                    }
                }
                if interrupted || (!device_committed && stop.load(Ordering::Relaxed)) {
                    // Scratch queries never committed: all current partial
                    // primitives are discarded, even if some attrs are ready.
                    for lane in states.iter_mut().filter(|lane| lane.summary.is_none()) {
                        finish_lane(lane, true, false, started, &mut on_region, &mut on_finished)?;
                    }
                    break;
                }
                for (lane, &row) in states
                    .iter()
                    .zip(&rows)
                    .filter(|(lane, _)| ready[lane.index])
                {
                    validate_visible(&row)
                        .with_context(|| format!("lane{}完整图元非法", lane.index))?;
                }
                if !device_committed {
                    let count = self.parallel_query(
                        &mut states,
                        &rows,
                        &ready,
                        true,
                        OutputScope::NextPrimitive,
                    )?;
                    batched_decode_calls += 1;
                    maximum_tensor_batch = maximum_tensor_batch.max(count);
                }
                // Commit every already-completed tensor row, even if a UI stop
                // arrives between these callbacks. No completed lane is lost.
                for lane in states.iter_mut().filter(|lane| ready[lane.index]) {
                    let row = rows[lane.index];
                    let global = lane.is_global();
                    lane.full.push(row);
                    lane.raw_cursor = lane.full.len();
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
                    self.flush_acceleration_notices(&mut on_status);
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
            prefill_strategy: "capacity-triggered rolling window; append controls while a full region fits; exact-prefix KV reuse; bounded COW start templates; B-row decode"
                .into(),
            actual_prefill_calls,
            prefill_reuse_hits,
            prefill_statistics_tracked: true,
        })
    }

    fn prepare_parallel_lane(
        &self,
        lane: &mut Lane,
        text: &[u32],
        settings: &InferenceSettings,
        stop: &AtomicBool,
        started: Instant,
        on_status: &mut impl FnMut(String),
        on_region: &mut impl FnMut(usize, DrawingRegion),
        on_finished: &mut impl FnMut(usize, GenerationSummary),
        shared_prefill: &mut PrefillReuseCache<GuidedCache>,
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
                        stop,
                        &mut |s| on_status(format!("序列{}：{s}", lane.index + 1)),
                        shared_prefill,
                    )?;
                    if lane.cache.is_none() {
                        return finish_lane(lane, true, false, started, on_region, on_finished);
                    }
                    lane.raw_cursor = lane.full.len();
                }
                return Ok(());
            }
            if lane.history.active.is_none() {
                if activate_generation_pending(&mut lane.history,&mut lane.full,&lane.pending_stage_context,
                    &mut lane.activated_stages,&mut lane.manual_review_regions,&mut lane.redraw_scope)?{
                    on_region(lane.index,DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?);
                    continue;
                }
                if !settings.context_overclock { lane.cache = None; lane.raw_cursor = 0; }
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
            let active = lane.history.active.as_ref().unwrap();
            let fresh=lane.cache.is_some() && lane.raw_cursor==lane.full.len();
            let review=lane.manual_review_regions.contains(&(active.side,active.index));
            let model_end=fresh && lane.cache.as_ref().is_some_and(|cache|should_end(cache,settings.cfg_scale));
            if let Some(index) = half_region_switch_target(&lane.history, active,
                settings.half_region_switch && (!review || (fresh && !model_end)), &mut lane.rng) {
                let active = lane.history.active.take().unwrap();
                let from = active.index;
                let side = active.side;
                lane.history.active = Some(interrupt_and_select_region(&mut lane.history, &mut lane.full, active, index));
                on_status(format!("序列{}：半程自动换块，{}区域 {} → {}", lane.index + 1, if side == 4 { "粗" } else { "细" }, from + 1, index + 1));
                on_region(lane.index, DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?);
                continue;
            }
            let active = lane.history.active.as_ref().unwrap();
            let count = *lane
                .history
                .visit_counts
                .get(&(active.side, active.index))
                .unwrap_or(&0);
            let review = lane.manual_review_regions.contains(&(active.side,active.index));
            let end = (!review && count >= active.cap())
                || (lane.raw_cursor == lane.full.len() && lane
                    .cache
                    .as_ref()
                    .map(|c| should_end(c, settings.cfg_scale))
                    .unwrap_or(false));
            if end {
                lane.full.push(END_RECORD);
                let active = lane.history.active.take().unwrap();
                lane.manual_review_regions.remove(&(active.side,active.index));
                finish_visit(&mut lane.history, active)?;
                if !settings.context_overclock { lane.cache = None; lane.raw_cursor = 0; }
                on_region(
                    lane.index,
                    DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?,
                );
                continue;
            }
            if fresh && guarded_scoped_review_full(settings,&lane.history,active,review,lane.redraw_scope.as_ref()){
                on_status(format!("阶段检查：第{}分支当前块已达到上下文上限。模型已重新读取但尚未预测结束；请先删除或调整图形后继续。",lane.index+1));
                return finish_lane(lane,true,false,started,on_region,on_finished);
            }
            if lane.cache.is_none() || lane.raw_cursor != lane.full.len()
                || (review && lane.cache.as_ref().is_some_and(|cache|{
                    let remaining=remaining_region_records(&lane.history,active,true);
                    cache.conditional.length+remaining>MAX_CONTEXT || (!settings.context_overclock && cache.visual_records.len()+remaining>897)
                })) {
                let active = lane.history.active.as_ref().unwrap();
                let context = self.region_context_window(lane.cache.as_ref(), &lane.full, lane.raw_cursor,
                    &lane.history, active, text.len(), settings.context_overclock, review,!lane.fine_aux_seen)?;
                on_status(format!("序列{}：上下文 {}+{}", lane.index + 1, text.len(), context.len()));
                on_region(
                    lane.index,
                    DrawingRegion::from_history(&lane.full).map_err(anyhow::Error::msg)?,
                );
                lane.cache = if let Some(previous) = lane.cache.take() {
                    self.retarget_visual_context(Some(previous), &context, text, settings.cfg_scale, stop,
                        &mut |s| on_status(format!("序列{}：{s}", lane.index + 1)))?
                } else {
                    self.prefill_shared(&context, text, settings.cfg_scale, stop,
                        &mut |s| on_status(format!("序列{}：{s}", lane.index + 1)), shared_prefill)?
                };
                if lane.cache.is_none() {
                    return finish_lane(lane, true, false, started, on_region, on_finished);
                }
                lane.raw_cursor = lane.full.len();
                if lane.history.active.as_ref().is_some_and(|active|active.side==8){lane.fine_aux_seen=true;}
                // Evaluate the newly prefetched marker's termination head;
                // different lanes may close/select independently here.
                continue;
            }
            return Ok(());
        }
    }

    fn parallel_query(
        &self,
        lanes: &mut [Lane],
        rows: &[[u32; 9]],
        ready: &[bool],
        commit: bool,
        scope: OutputScope,
    ) -> Result<usize> {
        let mut tensors = Vec::new();
        let mut caches = Vec::new();
        for lane in lanes.iter_mut().filter(|lane| ready[lane.index]) {
            let cache = lane.cache.as_mut().context("缺少并行KV")?;
            tensors.push(rows[lane.index]);
            caches.push(&mut cache.conditional);
            if let Some(unconditional) = cache.unconditional.as_mut() {
                tensors.push(rows[lane.index]);
                caches.push(unconditional);
            }
        }
        let tensor_batch = tensors.len();
        let predictions =
            self.model
                .forward_records_batched_for(&tensors, &mut caches, commit, scope)?;
        drop(caches);
        let mut predictions = predictions.into_iter();
        for lane in lanes.iter_mut().filter(|lane| ready[lane.index]) {
            let cache = lane.cache.as_mut().unwrap();
            cache.conditional_pending = predictions.next().context("条件分支批量输出缺失")?;
            cache.unconditional_pending = if cache.unconditional.is_some() {
                Some(predictions.next().context("无条件分支批量输出缺失")?)
            } else {
                None
            };
            if commit { cache.visual_records.push(rows[lane.index]); }
        }
        if predictions.next().is_some() {
            bail!("批量输出与lane映射不一致");
        }
        Ok(tensor_batch)
    }

    fn parallel_device_frame(
        &self,
        lanes: &mut [Lane],
        ready: &[bool],
        settings: &InferenceSettings,
        stop: &AtomicBool,
    ) -> Result<Option<(Vec<[u32; 9]>, usize, usize)>> {
        let active: Vec<_> = lanes
            .iter()
            .filter(|l| ready[l.index])
            .map(|l| l.index)
            .collect();
        let batch = active.len();
        let uniforms = {
            let mut rngs: Vec<_> = lanes
                .iter_mut()
                .filter(|l| ready[l.index])
                .map(|l| &mut l.rng)
                .collect();
            super::device::draw_uniforms(&mut rngs)
        };
        let mut specs = Vec::new();
        let mut heads = Vec::new();
        for (row, &index) in active.iter().enumerate() {
            let lane = &mut lanes[index];
            specs.push(super::device::DeviceLaneSpec {
                background: lane.background(),
                region: lane.region(),
            });
            heads.push(
                lane.cache
                    .as_ref()
                    .context("missing GPU lane cache")?
                    .device_pending
                    .as_ref()
                    .context("missing GPU head")?
                    .clone(),
            );
        }
        let pending: Vec<_> = heads.iter().collect();
        let mut caches = Vec::new();
        for lane in lanes.iter_mut().filter(|l| ready[l.index]) {
            let cache = lane.cache.as_mut().unwrap();
            caches.push(&mut cache.conditional);
            if let Some(u) = &mut cache.unconditional {
                caches.push(u);
            }
        }
        let physical = caches.len();
        let frame = self.device_frame(&specs, &pending, &mut caches, settings, uniforms, stop)?;
        drop(caches);
        let Some(frame) = frame else {
            return Ok(None);
        };
        let guidance = physical / batch;
        let mut rows = vec![PARTIAL; lanes.len()];
        for (row, &index) in active.iter().enumerate() {
            rows[index] = frame.records[row];
            lanes[index].cache.as_mut().unwrap().install_device(
                frame.next.narrow_rows(row * guidance, guidance)?,
                &frame.terms[row],
            )?;
            lanes[index].cache.as_mut().unwrap().visual_records.push(frame.records[row]);
        }
        Ok(Some((rows, frame.queries, physical)))
    }

    fn prefill_shared(
        &self,
        records: &[[u32; 9]],
        text: &[u32],
        cfg: f64,
        stop: &AtomicBool,
        on_status: &mut impl FnMut(String),
        shared: &mut PrefillReuseCache<GuidedCache>,
    ) -> Result<Option<GuidedCache>> {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        if let Some(cache) = shared.get(records) {
            if let Ok(mut stats) = self.context_window_stats.lock() {
                stats.reused_prefix_tokens = stats.reused_prefix_tokens.saturating_add(records.len() as u64);
            }
            on_status("复用本轮相同起点的模型上下文".into());
            return Ok(Some(cache));
        }
        shared.actual_prefill_calls += 1;
        let prepared = self.prefill(records, text, cfg, stop, on_status)?;
        if let Some(cache) = &prepared {
            shared.remember(records, cache.clone());
        }
        Ok(prepared)
    }
}

fn finish_lane(
    lane: &mut Lane,
    stopped: bool,
    completed: bool,
    started: Instant,
    on_region: &mut impl FnMut(usize, DrawingRegion),
    on_finished: &mut impl FnMut(usize, GenerationSummary),
) -> Result<()> {
    finish_lane_with_stage(lane, stopped, completed, None, started, on_region, on_finished)
}

fn finish_lane_with_stage(
    lane: &mut Lane,
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
    lane: &mut Lane,
    started: Instant,
    on_region: &mut impl FnMut(usize, DrawingRegion),
    on_finished: &mut impl FnMut(usize, GenerationSummary),
) -> Result<bool> {
    if lane.summary.is_some() {
        return Ok(true);
    }
    if lane.history.active.as_ref().is_some_and(|active| {
        !lane.manual_review_regions.contains(&(active.side,active.index)) && capped_final_coarse_visit(&lane.history, active, lane.pause_target)
    }) {
        lane.full.push(END_RECORD);
        let active = lane.history.active.take().context("阶段暂停缺少最后粗网格")?;
        finish_visit(&mut lane.history, active)?;
        lane.cache = None;
    }
    if let Some(stage) = reached_stage_pause(&lane.history, lane.pause_target) {
        finish_lane_with_stage(lane, false, false, Some(stage), started, on_region, on_finished)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn common_paused_stage(summaries: &[GenerationSummary]) -> Option<Stage> {
    let stage = summaries.first()?.paused_after_stage?;
    summaries.iter().all(|summary| summary.paused_after_stage == Some(stage)).then_some(stage)
}

fn prediction_error(actual: &[Prediction], expected: &[Prediction]) -> Result<(f64, f64)> {
    if actual.len() != expected.len() {
        bail!("批量输出数量不匹配");
    }
    let mut maximum: f64 = 0.;
    let mut error = 0.;
    let mut energy = 0.;
    for (a, b) in actual.iter().zip(expected) {
        for (&a, &b) in a
            .attributes
            .iter()
            .chain(&a.termination)
            .zip(b.attributes.iter().chain(&b.termination))
        {
            let d = a as f64 - b as f64;
            maximum = maximum.max(d.abs());
            error += d * d;
            energy += (b as f64).powi(2);
        }
    }
    Ok((maximum, (error / energy.max(1e-24)).sqrt()))
}

fn cfg_error(actual: &[Prediction], expected: &[Prediction], scale: f64) -> Result<(f64, f64)> {
    if actual.len() != expected.len() || actual.len() % 2 != 0 {
        bail!("CFG批量配对数量不匹配");
    }
    let mut maximum: f64 = 0.;
    let mut error = 0.;
    let mut energy = 0.;
    for (actual, expected) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
        for i in 0..actual[0].attributes.len() {
            let a = actual[1].attributes[i] as f64
                + scale * (actual[0].attributes[i] as f64 - actual[1].attributes[i] as f64);
            let b = expected[1].attributes[i] as f64
                + scale * (expected[0].attributes[i] as f64 - expected[1].attributes[i] as f64);
            let d = a - b;
            maximum = maximum.max(d.abs());
            error += d * d;
            energy += b * b;
        }
    }
    Ok((maximum, (error / energy.max(1e-24)).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pause_test_lane(index: usize, full: Vec<[u32; 9]>) -> Lane {
        let history = parse_history(&full).unwrap();
        let pause_target = stage_pause_target(&history, true);
        Lane { index, full, history, pause_target, generated: 0, manual_review_regions: HashSet::new(), fine_aux_seen:false,
                pending_stage_context:Vec::new(),activated_stages:HashSet::new(),redraw_scope:None,
                rng: StdRng::seed_from_u64(index as u64), cache: None, raw_cursor: 0, summary: None }
    }

    fn pause_test_global_prefix() -> Vec<[u32; 9]> {
        let mut full = vec![BOS_RECORD];
        let mut row = [512, 1536, 2048, 2518, 4566, 6415, 6702, 6830, 6958];
        row[5] = OFFSETS[5];
        full.push(row);
        row[5] = OFFSETS[5] + 1;
        full.extend(vec![row; 85]);
        full
    }

    #[test]
    fn stage_pause_finishes_each_lane_independently_without_stopping_its_siblings() {
        let mut initial = pause_test_global_prefix();
        let last = initial.pop().unwrap();
        let mut lanes: Vec<_> = (0..8).map(|index| pause_test_lane(index, initial.clone())).collect();
        let stop = AtomicBool::new(false);
        let started = Instant::now();
        let mut finished = Vec::new();
        for index in 0..8 {
            assert!(!finish_stage_pause_if_ready(&mut lanes[index], started, &mut |_, _| {},
                &mut |_, _| panic!("unfinished sibling must not be completed")).unwrap());
            lanes[index].full.push(last);
            lanes[index].history.fixed.push(last);
            lanes[index].generated = 1;
            assert!(finish_stage_pause_if_ready(&mut lanes[index], started,
                &mut |lane, region| { assert_eq!(lane, index); assert_eq!(region.completed_coarse, 0); },
                &mut |lane, summary| {
                    assert_eq!(summary.paused_after_stage, Some(Stage::Global));
                    assert!(!summary.stopped && !summary.completed && !summary.context_limit);
                    finished.push(lane);
                }).unwrap());
            assert!(!stop.load(Ordering::Relaxed));
            assert!(lanes[index + 1..].iter().all(|lane| lane.summary.is_none()));
        }
        assert_eq!(finished, (0..8).collect::<Vec<_>>());
        let mut summaries: Vec<_> = lanes.into_iter().map(|lane| lane.summary.unwrap()).collect();
        assert_eq!(common_paused_stage(&summaries), Some(Stage::Global));
        summaries[3].paused_after_stage = None;
        summaries[3].stopped = true;
        assert_eq!(common_paused_stage(&summaries), None);
    }

    #[test]
    fn final_capped_coarse_lane_keeps_real_end_and_can_resume_into_fine_stage() {
        let mut full = pause_test_global_prefix();
        for index in 0..15 { full.extend([region_marker(4, index), END_RECORD]); }
        full.push(region_marker(4, 15));
        let ordinary = [512, 1536, 2048, 2518, 4566, 6415, 6702, 6830, 6958];
        full.extend(vec![ordinary; 20]);
        let prefix = full.clone();
        let mut lane = pause_test_lane(0, full);
        lane.generated = 1;
        let mut finished = 0;
        assert!(finish_stage_pause_if_ready(&mut lane, Instant::now(),
            &mut |_, region| { assert_eq!(region.completed_coarse, 16); assert_eq!(region.stage, Stage::Fine); },
            &mut |_, summary| {
                assert_eq!(summary.paused_after_stage, Some(Stage::Coarse));
                assert!(!summary.stopped && !summary.completed);
                assert!(summary.history_records.starts_with(&prefix));
                assert_eq!(summary.history_records.last(), Some(&END_RECORD));
                finished += 1;
            }).unwrap());
        assert_eq!(finished, 1);
        let resumed = pause_test_lane(0, lane.full);
        assert_eq!(resumed.pause_target, None);
        assert_eq!(next_stage(&resumed.history), Some(8));
    }

    #[test]
    fn first_prefill_template_survives_lane_zero_empty_region_hops() {
        let common = vec![BOS_RECORD, [11; 9]];
        let region_a = vec![BOS_RECORD, [21; 9]];
        let region_b = vec![BOS_RECORD, [31; 9]];
        let mut cache = PrefillReuseCache::default();
        cache.remember(&common, "common");
        cache.remember(&region_a, "region_a");
        cache.remember(&region_b, "region_b");

        // A sibling comes back to the common starting context after lane0 has
        // already prepared two later empty regions. The old single-slot cache
        // evicted common here, repeating the expensive original prefill.
        assert_eq!(cache.get(&common), Some("common"));
        assert_eq!(cache.get(&region_b), Some("region_b"));
        assert_eq!(cache.get(&region_a), None);
        assert_eq!(cache.reuse_hits, 2);
        assert_eq!(cache.first.as_ref().unwrap().0, common);
        assert_eq!(cache.recent.as_ref().unwrap().0, region_b);
    }

    #[test]
    fn prefill_key_compares_complete_records_and_returned_value_is_a_clone() {
        let key = vec![BOS_RECORD, [17; 9], [23; 9]];
        let mut cache = PrefillReuseCache::default();
        cache.remember(&key, vec![4, 5]);
        for field in 0..9 {
            let mut altered = key.clone();
            altered[1][field] += 1;
            assert!(cache.get(&altered).is_none());
        }
        assert!(cache.get(&key[..2]).is_none());
        let mut result = cache.get(&key).unwrap();
        result.push(6);
        assert_eq!(cache.get(&key), Some(vec![4, 5]));
        assert_eq!(cache.reuse_hits, 2);
    }

    #[test]
    fn later_contexts_replace_only_recent_slot_and_each_round_starts_empty() {
        let common = vec![BOS_RECORD];
        let mut cache = PrefillReuseCache::default();
        cache.remember(&common, 100);
        for region in 1..16 {
            cache.remember(&[[region; 9]], region);
            assert_eq!(cache.get(&common), Some(100));
            assert_eq!(cache.get(&[[region; 9]]), Some(region));
            if region > 1 {
                assert!(cache.get(&[[region - 1; 9]]).is_none());
            }
        }
        cache.remember(&common, 999);
        assert_eq!(cache.get(&common), Some(100));
        drop(cache);
        let mut next_round = PrefillReuseCache::<u32>::default();
        assert!(next_round.get(&common).is_none());
        assert_eq!(next_round.reuse_hits, 0);
        assert_eq!(next_round.actual_prefill_calls, 0);
    }

    #[test]
    fn stable_parallel_seeds_keep_lane_zero_and_are_distinct_with_wraparound() {
        assert_eq!(parallel_seed(91, 0), 91);
        assert_eq!(parallel_seed(91, 15), 106);
        assert_eq!(parallel_seed(u64::MAX, 1), 0);
        let seeds: HashSet<_> = (0..16)
            .map(|lane| parallel_seed(u64::MAX - 7, lane))
            .collect();
        assert_eq!(seeds.len(), 16);
    }
}
