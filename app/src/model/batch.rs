//! One query per independent cache, with shared batched matrix operations.
//! Padding is an ephemeral attention input and is never committed to a lane.

use super::*;

impl PrimitiveModel {
    pub fn forward_records_batched(
        &self,
        records: &[[u32; 9]],
        caches: &mut [&mut KvCache],
        commit: bool,
    ) -> Result<Vec<Prediction>> {
        self.forward_records_batched_for(records, caches, commit, OutputScope::All)
    }

    pub fn forward_records_batched_for(
        &self,
        records: &[[u32; 9]],
        caches: &mut [&mut KvCache],
        commit: bool,
        scope: OutputScope,
    ) -> Result<Vec<Prediction>> {
        let batch = records.len();
        if batch == 0 || batch != caches.len() {
            bail!("批量图元与独立KV缓存数量不匹配");
        }
        let hidden =
            self.forward_batched_embedding(self.primitive_embeddings(records)?, caches, commit)?;
        self.predict_many(&hidden, scope)
    }

    pub fn forward_device_records(
        &self,
        records: &Tensor,
        caches: &mut [&mut KvCache],
        commit: bool,
        scope: OutputScope,
    ) -> Result<DevicePrediction> {
        let batch = records.dim(0)?;
        if batch == 0 || batch != caches.len() {
            bail!("device record rows and caches differ");
        }
        let embedding = self.primitive_embeddings_device(records)?;
        let hidden = if batch == 1 {
            self.forward_embedding(embedding, caches[0], commit)?
        } else {
            self.forward_batched_embedding(embedding, caches, commit)?
        };
        self.predict_device(&hidden, scope)
    }

    fn forward_batched_embedding(
        &self,
        mut x: Tensor,
        caches: &mut [&mut KvCache],
        commit: bool,
    ) -> Result<Tensor> {
        let _device_only = self.strict_math_guard();
        let batch = x.dim(0)?;
        let storage_dtype = caches[0].storage_dtype;
        if caches.iter().any(|c| c.storage_dtype != storage_dtype) {
            bail!("one batch must use the same KV storage dtype");
        }
        let heads = self.architecture.heads;
        let dh = self.architecture.head_width;
        let width = self.architecture.width;
        let mut positions = Vec::with_capacity(batch);
        let lengths: Vec<_> = caches.iter().map(|c| c.length + 1).collect();
        for cache in caches.iter() {
            let position = cache.length + cache.position_offset;
            if position >= MAX_CONTEXT {
                bail!("批量lane上下文已满；不截断或混用其他lane缓存");
            }
            if cache.layers.len() != self.blocks.len() {
                bail!("批量lane KV层数不匹配");
            }
            positions.push(position as u32);
        }
        let max_length = *lengths.iter().max().unwrap();
        let int8 = storage_dtype == DType::U8;
        if int8 && dh != 64 { bail!("INT8 historical KV requires 64 channels per head"); }
        let past_lens: Vec<_> = caches.iter().map(|cache| cache.length).collect();
        let int8_past = if int8 {
            Some(Tensor::from_vec(past_lens.iter().map(|&n| u32::try_from(n))
                .collect::<std::result::Result<Vec<_>, _>>()?, batch, &self.device)?)
        } else { None };
        let use_fused_attention = int8 || (self.optional_attention_for_rows(batch) && dh == 64);
        let (cos, sin, mask, write_plan, row_lengths) = if self.diagnostic_legacy_metadata {
            let positions = Tensor::from_vec(positions, batch, &self.device)?;
            let cos = self
                .rope_cos
                .index_select(&positions, 0)?
                .unsqueeze(1)?
                .contiguous()?;
            let sin = self
                .rope_sin
                .index_select(&positions, 0)?
                .unsqueeze(1)?
                .contiguous()?;
            let mask: Vec<f32> = lengths
                .iter()
                .flat_map(|&length| {
                    (0..max_length).map(move |i| if i < length { 0. } else { f32::NEG_INFINITY })
                })
                .collect();
            let mask = Tensor::from_vec(mask, (batch, 1, 1, max_length), &self.device)?;
            let row_lengths = if use_fused_attention && lengths.iter().any(|&length| length != max_length) {
                Some(Tensor::from_vec(lengths.iter().map(|&length| length as u32).collect::<Vec<_>>(), batch, &self.device)?)
            } else { None };
            (cos, sin, Some(mask), None, row_lengths)
        } else {
            let metadata = self
                .decode_metadata
                .lock()
                .map_err(|_| anyhow::anyhow!("decode metadata cache poisoned"))?
                .get(
                    &positions,
                    &lengths,
                    &self.rope_cos,
                    &self.rope_sin,
                    &self.device,
                    heads,
                )?;
            (
                metadata.cos,
                metadata.sin,
                metadata.mask,
                metadata.write_plan,
                metadata.row_lengths,
            )
        };
        for (layer_index, block) in self.blocks.iter().enumerate() {
            let normalized = candle_nn::ops::rms_norm(&x, &block.norm1, 1e-6)?;
            // These are real B-row GEMMs sharing one copy of each weight.
            let qkv = normalized.matmul(&block.qkv.t()?)?;
            let q = qkv
                .narrow(1, 0, width)?
                .reshape((batch, heads, 1, dh))?
                .contiguous()?;
            let k = qkv
                .narrow(1, width, width)?
                .reshape((batch, heads, 1, dh))?
                .contiguous()?;
            let v = qkv
                .narrow(1, width * 2, width)?
                .reshape((batch, heads, 1, dh))?
                .contiguous()?;
            let q = candle_nn::ops::rms_norm(&q, &block.q_norm, 1e-6)?;
            let k = candle_nn::ops::rms_norm(&k, &block.k_norm, 1e-6)?;
            let q = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
            let attended = if int8 {
                let mut layers = caches.iter_mut().map(|cache| &mut cache.layers[layer_index]).collect::<Vec<_>>();
                let history = LayerCache::query_int8_batch(&mut layers, &k, &v, &past_lens, commit, write_plan.as_ref())?;
                int8_kv_ops::attention(&q.to_dtype(DType::F32)?.reshape((batch * heads, 1, dh))?.contiguous()?,
                    &history.key, &history.value, &history.key_scale, &history.value_scale,
                    &k.to_dtype(DType::F32)?.reshape((batch * heads, 1, dh))?.contiguous()?,
                    &v.to_dtype(DType::F32)?.reshape((batch * heads, 1, dh))?.contiguous()?,
                    int8_past.as_ref().unwrap(), heads,
                    if self.optional_split_k && max_length >= 256 { 128 } else { 0 })?
                    .to_dtype(self.dtype)?.reshape((batch, width))?
            } else {
            let k = k.to_dtype(storage_dtype)?;
            let v = v.to_dtype(storage_dtype)?;
            let (keys, values) = if self.diagnostic_stack_kv {
                // Retained only for matched-workload A/B and numerical tests.
                let mut padded_keys = Vec::with_capacity(batch);
                let mut padded_values = Vec::with_capacity(batch);
                for (lane, cache) in caches.iter_mut().enumerate() {
                    let current_k = k.narrow(0, lane, 1)?.reshape((heads, 1, dh))?;
                    let current_v = v.narrow(0, lane, 1)?.reshape((heads, 1, dh))?;
                    let past_len = cache.length;
                    let (keys, values) = cache.layers[layer_index]
                        .query(&current_k, &current_v, past_len, commit)?;
                    let padding = max_length - lengths[lane];
                    if padding > 0 {
                        let zeros =
                            Tensor::zeros((heads, padding, dh), storage_dtype, &self.device)?;
                        padded_keys.push(Tensor::cat(&[&keys, &zeros], 1)?);
                        padded_values.push(Tensor::cat(&[&values, &zeros], 1)?);
                    } else {
                        padded_keys.push(keys);
                        padded_values.push(values);
                    }
                }
                (
                    Tensor::stack(&padded_keys, 0)?.reshape((batch * heads, max_length, dh))?,
                    Tensor::stack(&padded_values, 0)?.reshape((batch * heads, max_length, dh))?,
                )
            } else {
                let past_lens: Vec<_> = caches.iter().map(|c| c.length).collect();
                let mut layers: Vec<_> = caches
                    .iter_mut()
                    .map(|c| &mut c.layers[layer_index])
                    .collect();
                LayerCache::query_batch_with_plan(
                    &mut layers,
                    &k,
                    &v,
                    &past_lens,
                    commit,
                    write_plan.as_ref(),
                )?
            };
            let q = q.to_dtype(DType::F32)?.reshape((batch * heads, 1, dh))?;
            let attended = if use_fused_attention {
                super::fused_attention::decode_attention(
                    &q, &keys, &values, row_lengths.as_ref(), heads,
                    if self.optional_split_k && max_length >= 256 { 128 } else { 0 },
                )?.to_dtype(self.dtype)?.reshape((batch, width))?
            } else {
                let keys = keys.to_dtype(DType::F32)?;
                let values = values.to_dtype(DType::F32)?;
                let transposed_keys = keys.transpose(1, 2)?;
                let transposed_keys = if self.diagnostic_copy_transposed_keys { transposed_keys.contiguous()? } else { transposed_keys };
                let mut scores = (q.matmul(&transposed_keys)? / (dh as f64).sqrt())?
                    .reshape((batch, heads, 1, max_length))?;
                if let Some(mask) = &mask { scores = scores.broadcast_add(mask)?; }
                let probabilities = candle_nn::ops::softmax_last_dim(&scores)?.reshape((batch * heads, 1, max_length))?;
                probabilities.matmul(&values)?.to_dtype(self.dtype)?.reshape((batch, width))?
            };
            attended
            };
            x = (x + attended.matmul(&block.projection.t()?)?)?;
            let normalized = candle_nn::ops::rms_norm(&x, &block.norm2, 1e-6)?;
            let gate_up = normalized.matmul(&block.gate_up.t()?)?;
            let activated = if self.optional_fusion && self.device.is_cuda() && self.optional_attention_for_rows(batch) {
                super::graph_ops::swiglu(&gate_up)?
            } else {
                let gate = gate_up.narrow(1, 0, self.architecture.ffn_hidden)?;
                let up = gate_up.narrow(1, self.architecture.ffn_hidden, self.architecture.ffn_hidden)?;
                (gate.silu()? * up)?
            };
            x = (x + activated.matmul(&block.down.t()?)?)?;
        }
        if commit {
            for cache in caches.iter_mut() {
                cache.length += 1;
            }
        }
        let hidden = candle_nn::ops::rms_norm(&x, &self.final_norm, 1e-6)?;
        Ok(hidden)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn tensor(rows: usize, cols: usize, scale: f32) -> Tensor {
        let v: Vec<_> = (0..rows * cols)
            .map(|i| (((i * 37 + 11) % 101) as f32 - 50.) * scale)
            .collect();
        Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap()
    }
    pub(crate) fn small_model() -> PrimitiveModel {
        let device = Device::Cpu;
        let width = 8;
        let heads = 2;
        let dh = 4;
        let ffn = 12;
        let norm = || Tensor::ones(width, DType::F32, &device).unwrap();
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for p in 0..MAX_CONTEXT {
            for pair in 0..dh / 2 {
                let a = p as f32 * (1. / 10000f32.powf((pair * 2) as f32 / dh as f32));
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        PrimitiveModel {
            architecture: Architecture {
                width,
                layers: 2,
                heads,
                head_width: dh,
                ffn_hidden: ffn,
                device: "CPU test only".into(),
                dtype: "F32".into(),
            },
            device: device.clone(),
            dtype: DType::F32,
            embedding: tensor(7069, width, 0.004),
            text_embedding: tensor(1500, width, 0.003),
            blocks: (0..2)
                .map(|_| Block {
                    norm1: norm(),
                    qkv: tensor(width * 3, width, 0.002),
                    q_norm: Tensor::ones(dh, DType::F32, &device).unwrap(),
                    k_norm: Tensor::ones(dh, DType::F32, &device).unwrap(),
                    projection: tensor(width, width, 0.003),
                    norm2: norm(),
                    gate_up: tensor(ffn * 2, width, 0.002),
                    down: tensor(width, ffn, 0.003),
                })
                .collect(),
            final_norm: norm(),
            termination_weight: tensor(2, width, 0.003),
            termination_bias: Tensor::zeros(2, DType::F32, &device).unwrap(),
            rope_cos: Tensor::from_vec(cos, (MAX_CONTEXT, dh / 2), &device).unwrap(),
            rope_sin: Tensor::from_vec(sin, (MAX_CONTEXT, dh / 2), &device).unwrap(),
            diagnostic_copy_transposed_keys: false,
            diagnostic_stack_kv: false,
            diagnostic_legacy_metadata: false,
            fp32_kv_storage: false,
            int8_kv: Int8KvStatus::default(),
            optional_fusion: false,
            optional_split_k: false,
            #[cfg(feature="cuda")]
            cuda_auto_attention_rows:None,
            #[cfg(feature = "rocm")]
            hip_auto_attention_rows: None,
            decode_metadata: Mutex::new(DecodeMetadataCache::default()),
        }
    }
    fn error(a: &Prediction, b: &Prediction) -> f32 {
        a.attributes
            .iter()
            .chain(a.termination.iter())
            .zip(b.attributes.iter().chain(b.termination.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max)
    }
    #[test]
    fn projected_heads_match_full_head_and_reject_wrong_offsets() {
        let model = small_model();
        let hidden = tensor(2, model.architecture.width, 0.025);
        let all = model.predict_many(&hidden, OutputScope::All).unwrap();
        let next = model
            .predict_many(&hidden, OutputScope::NextPrimitive)
            .unwrap();
        for row in 0..2 {
            assert!(next[row]
                .attributes
                .iter()
                .zip(&all[row].attributes[..1024])
                .all(|(a, b)| (a - b).abs() < 1e-6));
            assert_eq!(next[row].termination, all[row].termination);
        }
        for field in 0..9 {
            let projected = model
                .predict_many(&hidden, OutputScope::Field(field))
                .unwrap();
            let start = OFFSETS[field] as usize;
            let end = start + BINS[field] as usize;
            for row in 0..2 {
                assert!(projected[row]
                    .attributes
                    .iter()
                    .zip(&all[row].attributes[start..end])
                    .all(|(a, b)| (a - b).abs() < 1e-6));
                assert!(
                    (projected[row].attribute(start as u32).unwrap()
                        - all[row].attributes[start] as f64)
                        .abs()
                        < 1e-6
                );
                assert!(projected[row].attribute(end as u32).is_err());
                if start > 0 {
                    assert!(projected[row].attribute(start as u32 - 1).is_err());
                }
                assert_eq!(projected[row].termination, [0., 0.]);
            }
        }
        assert!(model.predict_many(&hidden, OutputScope::Field(9)).is_err());
    }

    #[test]
    fn intermediate_prefill_skips_heads_but_keeps_identical_committed_context() {
        let model = small_model();
        let mut full = model.new_cache();
        let mut skipped = model.new_cache();
        let records = [
            BOS_RECORD,
            [510, 1500, 2049, 2500, 4500, 6415, 6700, 6800, 7000],
        ];
        model.prefill_records(&records, &mut full).unwrap();
        assert!(model
            .prefill_records_for(&records, &mut skipped, None)
            .unwrap()
            .is_none());
        assert_eq!(skipped.length, 2);
        let a = model.forward_record(&PARTIAL, &mut full, false).unwrap();
        let b = model.forward_record(&PARTIAL, &mut skipped, false).unwrap();
        assert!(error(&a, &b) < 1e-6);
    }
    #[test]
    fn true_batched_ragged_decode_matches_independent_reference_and_commit_lengths() {
        let model = small_model();
        let ordinary = [510, 1500, 2049, 2500, 4500, 6415, 6700, 6800, 7000];
        let mut caches = vec![
            model.new_cache(),
            model.new_cache(),
            model.new_cache_offset(3),
        ];
        for (lane, cache) in caches.iter_mut().enumerate() {
            model
                .prefill_text(if lane == 1 { &[2, 17, 19, 3] } else { &[2, 3] }, cache)
                .unwrap();
            model.forward_record(&BOS_RECORD, cache, true).unwrap();
            for _ in 0..lane {
                model.forward_record(&ordinary, cache, true).unwrap();
            }
        }
        let before_lengths: Vec<_> = caches.iter().map(|c| c.length).collect();
        let mut reference = caches.clone();
        let mut partial = PARTIAL;
        partial[0] = 123;
        let partials = [partial, partial, partial];
        let reference_partial: Vec<_> = reference
            .iter_mut()
            .map(|c| model.forward_record(&partial, c, false).unwrap())
            .collect();
        let mut refs: Vec<_> = caches.iter_mut().collect();
        let actual = model
            .forward_records_batched(&partials, &mut refs, false)
            .unwrap();
        drop(refs);
        for i in 0..3 {
            assert!(error(&actual[i], &reference_partial[i]) < 1e-4);
            assert_eq!(caches[i].length, before_lengths[i]);
        }
        let records = [ordinary, ordinary, ordinary];
        let expected: Vec<_> = reference
            .iter_mut()
            .map(|c| model.forward_record(&ordinary, c, true).unwrap())
            .collect();
        let mut refs: Vec<_> = caches.iter_mut().collect();
        let actual = model
            .forward_records_batched(&records, &mut refs, true)
            .unwrap();
        drop(refs);
        for i in 0..3 {
            assert!(error(&actual[i], &expected[i]) < 1e-4);
            assert_eq!(caches[i].length, before_lengths[i] + 1);
            for layer in &caches[i].layers {
                assert_eq!(
                    layer.key().unwrap().unwrap().dim(1).unwrap(),
                    caches[i].length
                );
            }
        }
        assert_ne!(caches[0].length, caches[1].length);
    }

    #[test]
    fn seventeen_logical_lanes_with_cfg_match_all_thirty_four_independent_rows() {
        let model = small_model();
        let logical_rows = 17;
        let physical_rows = checked_batch_rows(logical_rows, 2).unwrap();
        let mut caches = Vec::with_capacity(physical_rows);
        let mut partials = Vec::with_capacity(physical_rows);
        for row in 0..physical_rows {
            let mut cache = model.new_cache();
            model.prefill_text(if row % 2 == 0 { &[2, 17, 19, 3] } else { &[2, 3] }, &mut cache).unwrap();
            model.forward_record(&BOS_RECORD, &mut cache, true).unwrap();
            caches.push(cache);
            let mut partial = PARTIAL;
            partial[0] = row as u32;
            partials.push(partial);
        }
        let before: Vec<_> = caches.iter().map(|cache| cache.length).collect();
        let mut reference = caches.clone();
        let expected: Vec<_> = reference.iter_mut().zip(&partials)
            .map(|(cache, partial)| model.forward_record(partial, cache, false).unwrap()).collect();
        let mut references: Vec<_> = caches.iter_mut().collect();
        let actual = model.forward_records_batched(&partials, &mut references, false).unwrap();
        drop(references);
        assert_eq!(actual.len(), physical_rows, "no requested CFG row may be dropped");
        for row in 0..physical_rows {
            assert!(error(&actual[row], &expected[row]) < 1e-4, "row {row}");
            assert_eq!(caches[row].length, before[row]);
        }
    }

    #[test]
    fn unsupported_int8_request_keeps_the_same_cpu_model_and_floating_outputs() {
        let mut model = small_model();
        let mut reference = model.new_cache();
        model.forward_record(&BOS_RECORD, &mut reference, true).unwrap();
        let before = model.forward_record(&PARTIAL, &mut reference, false).unwrap();
        let status = model.prepare_int8_kv(true).unwrap();
        assert!(status.requested && !status.available && !status.enabled && !status.ready);
        assert!(!status.reason.is_empty());
        assert!(model.device.is_cpu());
        let mut fallback = model.new_cache();
        assert_eq!(fallback.storage_dtype, DType::F32);
        model.forward_record(&BOS_RECORD, &mut fallback, true).unwrap();
        let after = model.forward_record(&PARTIAL, &mut fallback, false).unwrap();
        assert_eq!(error(&before, &after), 0.);
    }
}
