//! Sampling feedback stays on the device until a complete visible primitive.
use super::*;
use candle_core::{DType, Tensor};

pub(super) struct DeviceLaneSpec {
    pub background: bool,
    pub region: Option<(usize, usize)>,
}
pub(super) struct DeviceFrame {
    pub records: Vec<[u32; 9]>,
    pub next: model::DevicePrediction,
    pub terms: Vec<Vec<[f32; 2]>>,
    pub queries: usize,
}

pub(super) fn draw_uniforms(rngs: &mut [&mut StdRng]) -> Vec<f32> {
    let batch = rngs.len();
    let mut values = vec![0.; batch * 9];
    for (row, rng) in rngs.iter_mut().enumerate() {
        for field in 0..9 {
            values[field * batch + row] = rng.random::<f32>();
        }
    }
    values
}

fn empty_prediction(term: [f32; 2]) -> Prediction {
    Prediction {
        attributes: Vec::new(),
        termination: term,
        attribute_offset: 0,
    }
}

impl GuidedCache {
    pub(super) fn install_device(
        &mut self,
        prediction: model::DevicePrediction,
        terms: &[[f32; 2]],
    ) -> Result<()> {
        let count = if self.unconditional.is_some() { 2 } else { 1 };
        if prediction.rows()? != count || terms.len() != count {
            bail!("guided device row mapping differs");
        }
        self.conditional_pending = empty_prediction(terms[0]);
        self.unconditional_pending = if count == 2 {
            Some(empty_prediction(terms[1]))
        } else {
            None
        };
        self.device_pending = Some(prediction);
        Ok(())
    }
}

pub(super) fn read_device_terms(prediction: &model::DevicePrediction) -> Result<Vec<[f32; 2]>> {
    let term = prediction
        .termination
        .as_ref()
        .context("device prediction has no termination head")?;
    let valid = prediction.valid_rows()?.to_dtype(DType::F32)?;
    let packed = Tensor::cat(&[term, &valid], 1)?.to_vec2::<f32>()?;
    packed
        .into_iter()
        .map(|row| {
            if row[2] != 1. || row.iter().any(|x| !x.is_finite()) {
                bail!("模型输出出现非有限值，未提交当前图元");
            }
            Ok([row[0], row[1]])
        })
        .collect()
}

impl NativeEngine {
    /// Uniforms are field-major [9, logical_B]. Each lane owns its own RNG.
    pub(super) fn device_frame(
        &self,
        specs: &[DeviceLaneSpec],
        pending: &[&model::DevicePrediction],
        caches: &mut [&mut KvCache],
        settings: &InferenceSettings,
        uniforms: Vec<f32>,
        stop: &AtomicBool,
    ) -> Result<Option<DeviceFrame>> {
        let batch = specs.len();
        let guidance = if (settings.cfg_scale - 1.).abs() > 1e-12 {
            2
        } else {
            1
        };
        if batch == 0
            || pending.len() != batch
            || caches.len() != batch * guidance
            || uniforms.len() != batch * 9
        {
            bail!("device frame row counts differ");
        }
        let regional_auto = self.acceleration_settings.mode == crate::acceleration::AccelerationMode::Auto
            && self.acceleration_settings.primitive_graph == crate::acceleration::FeaturePreference::Auto
            && specs.iter().any(|spec| spec.region.is_some());
        if self.primitive_graph_enabled && regional_auto && !self.graph_stage_notified.swap(true,Ordering::Relaxed) {
            // Tiny regions cannot reliably amortize a new capture; retain the
            // fused ordinary path. Explicit Graph/Optimized remains available.
            *self.primitive_graph.lock().map_err(|_|anyhow::anyhow!("Graph会话锁不可用"))?=None;
            self.acceleration_notice("区域阶段使用融合逐步执行，避免短区域反复捕获Graph的开销");
        }
        if self.primitive_graph_enabled && !regional_auto {
            if let Some(frame)=self.try_primitive_graph_frame(specs,pending,caches,settings,&uniforms,stop)? {
                return Ok(Some(frame));
            }
            if stop.load(Ordering::Relaxed) { return Ok(None); }
        }
        self.model.unify_graph_fallback_caches(caches)?;
        let device = &self.model.device;
        #[cfg(feature = "rocm")]
        let hip_feedback_guard = device.is_rocm().then(candle_core::rocm_backend::strict::DeviceOnlyGuard::enter);
        let uniforms = Tensor::from_vec(uniforms, (9, batch), device)?;
        let partial: Vec<_> = (0..batch).flat_map(|_| PARTIAL).collect();
        let records = Tensor::from_vec(partial, (batch, 9), device)?;
        let key: Vec<_> = specs.iter().map(|s| (s.background, s.region)).collect();
        let workspace = self
            .device_workspace
            .lock()
            .map_err(|_| anyhow::anyhow!("sampling workspace poisoned"))?
            .get_for_diagnostic(&key, settings.renyi_alpha as f32, guidance, device,
                self.diagnostic_full_attribute_vocab)?;
        let row_map = &workspace.row_map;
        let field_indices = &workspace.field_indices;
        let candidate_ranges = &workspace.candidate_ranges;
        let plans = &workspace.plans;
        let mut prediction = model::DevicePrediction::concat_rows(pending)?;
        let mut valid = Tensor::ones((batch, 1), DType::U32, device)?;
        let mut queries = 0;
        for field in 0..9 {
            if stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let candidates = &candidate_ranges[field];
            let column = field_indices.narrow(0, field, 1)?.reshape((batch, 1))?;
            if let Some(forced) = &workspace.forced[field] {
                records.scatter_set(&column, forced, 1)?;
                continue;
            }
            if field > 0 {
                let physical = records.index_select(&row_map, 0)?;
                prediction = self.model.forward_device_records(
                    &physical,
                    caches,
                    false,
                    model::OutputScope::Field(field),
                )?;
                queries += 1;
            }
            if prediction.attribute_offset != model::OFFSETS[field] as usize {
                bail!("GPU projected head does not match attribute");
            }
            let u = uniforms.narrow(0, field, 1)?.reshape((batch, 1))?;
            let sampled = super::device_sampling::sample_guided_prepared(
                &prediction.attributes,
                &u,
                &plans[field],
                guidance,
                settings.cfg_scale as f32,
                self.fused_sampling,
            )?;
            valid = valid.broadcast_mul(&sampled.valid)?;
            records.scatter_set(&column, &sampled.tokens, 1)?;
        }
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let physical = records.index_select(&row_map, 0)?;
        let next = self.model.forward_device_records(
            &physical,
            caches,
            true,
            model::OutputScope::NextPrimitive,
        )?;
        queries += 1;
        // The cache is now internally committed. Any validation failure below
        // discards this failed cache and never creates a visible history node.
        let next_valid = next.valid_rows()?.reshape((batch, guidance))?;
        let mut pair_valid = next_valid.narrow(1, 0, 1)?;
        if guidance == 2 {
            pair_valid = pair_valid.broadcast_mul(&next_valid.narrow(1, 1, 1)?)?;
        }
        valid = valid.broadcast_mul(&pair_valid)?;
        let term = next
            .termination
            .as_ref()
            .context("next device termination missing")?
            .reshape((batch, guidance * 2))?;
        let packed = Tensor::cat(
            &[
                &records.to_dtype(DType::F32)?,
                &term,
                &valid.to_dtype(DType::F32)?,
            ],
            1,
        )?;
        if self.stop_after_device_enqueue {
            stop.store(true, Ordering::Relaxed);
        }
        #[cfg(feature = "rocm")]
        drop(hip_feedback_guard);
        let host = packed.to_vec2::<f32>()?;
        let mut output = Vec::with_capacity(batch);
        let mut terms = Vec::with_capacity(batch);
        for (index, row) in host.into_iter().enumerate() {
            if row[9 + guidance * 2] != 1. || row.iter().any(|v| !v.is_finite()) {
                bail!("GPU采样出现非有限输出，未提交当前图元");
            }
            let mut record = [0u32; 9];
            for field in 0..9 {
                let value = row[field];
                if value < 0. || value.fract() != 0. || value > model::BOS as f32 {
                    bail!("GPU采样返回非法token");
                }
                record[field] = value as u32;
                let (start, count) = candidate_ranges[field][index];
                if record[field] < start || record[field] >= start + count as u32 {
                    bail!("GPU采样结果越出合法绘制范围");
                }
            }
            if self.diagnostic_full_attribute_vocab {
                for field in 0..9 {
                    if !(OFFSETS[field]..OFFSETS[field] + model::BINS[field]).contains(&record[field]) {
                        bail!("diagnostic GPU sample escaped its field vocabulary");
                    }
                }
            } else {
                validate_visible(&record)?;
            }
            output.push(record);
            terms.push(
                (0..guidance)
                    .map(|g| [row[9 + g * 2], row[10 + g * 2]])
                    .collect(),
            );
        }
        // Stop may arrive while the GPU is still evaluating an enqueued frame.
        // No visible operation is committed until this host validation boundary.
        // The caller discards the internally advanced KV when None is returned.
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        Ok(Some(DeviceFrame {
            records: output,
            next,
            terms,
            queries,
        }))
    }

    pub(super) fn next_primitive_device(
        &self,
        cache: &mut GuidedCache,
        settings: &InferenceSettings,
        background: bool,
        region: Option<(usize, usize)>,
        stop: &AtomicBool,
        rng: &mut StdRng,
    ) -> Result<Option<[u32; 9]>> {
        let uniforms = draw_uniforms(&mut [rng]);
        let pending = cache
            .device_pending
            .as_ref()
            .context("GPU sampling cache has no device head")?
            .clone();
        let mut refs = vec![&mut cache.conditional];
        if let Some(u) = &mut cache.unconditional {
            refs.push(u);
        }
        let result = self.device_frame(
            &[DeviceLaneSpec { background, region }],
            &[&pending],
            &mut refs,
            settings,
            uniforms,
            stop,
        )?;
        drop(refs);
        if let Some(frame) = result {
            cache.install_device(frame.next, &frame.terms[0])?;
            Ok(Some(frame.records[0]))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uniforms_are_lane_local_even_when_a_sibling_stops() {
        let mut alone = StdRng::seed_from_u64(71);
        let mut a = StdRng::seed_from_u64(71);
        let mut sibling = StdRng::seed_from_u64(99);
        let single = draw_uniforms(&mut [&mut alone]);
        let pair = draw_uniforms(&mut [&mut a, &mut sibling]);
        assert_eq!(single, (0..9).map(|i| pair[i * 2]).collect::<Vec<_>>());
        assert_eq!(
            draw_uniforms(&mut [&mut alone]),
            draw_uniforms(&mut [&mut a])
        );
    }
}
