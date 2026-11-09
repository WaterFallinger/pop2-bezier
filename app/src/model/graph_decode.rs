//! Fixed-capacity, GPU-position decoder for replayable primitive graphs.
//! The ordinary decoder remains the compatibility path. This path deliberately
//! does not derive RoPE, KV offsets or active masks from captured Rust values.

use super::graph_ops;
use super::*;

pub struct GraphDecodeState {
    pub physical_rows: usize,
    pub capacity: usize,
    pub positions: Tensor,
    pub lengths: Tensor,
    row_map: Tensor,
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    qk_rope_parameters: Vec<Tensor>,
    termination_bias: Tensor,
    storage_identities: Option<Vec<Vec<usize>>>,
    pinned_views: Vec<KvCache>,
    expected_lengths: Vec<usize>,
    expected_positions: Vec<usize>,
    adopted: bool,
    // Keep captured weight pointers/events alive until the owning graph has
    // been destroyed, even if NativeEngine's model field is dropped first.
    _weight_pins: Vec<Tensor>,
}

pub struct GraphFrameMetadata {
    write_indices: Tensor,
    pub valid_lengths: Tensor,
}

impl GraphDecodeState {
    pub fn refresh(&mut self, model: &PrimitiveModel, caches: &[&KvCache]) -> Result<()> {
        // Exported row views can have immutable snapshots elsewhere. Reusing
        // this arena for a different context would overwrite those prefixes,
        // or clobber a later source row during a physical-row permutation.
        // A changed identity/position therefore needs an independent arena.
        self.validate_caches(caches)?;
        self.set_positions(caches, model)
    }

    pub fn new(
        model: &PrimitiveModel,
        caches: &[&KvCache],
        row_map: &Tensor,
        capacity: usize,
    ) -> Result<Self> {
        let physical_rows = caches.len();
        let arch = &model.architecture;
        if model.int8_kv_enabled() || caches.iter().any(|cache| cache.int8_kv_enabled()) {
            bail!("F32 Graph is disabled for INT8 historical KV; use fused INT8 decode without expanding the history");
        }
        if !model.device.is_cuda()
            || physical_rows == 0
            || arch.head_width != 64
            || row_map.dims() != [physical_rows]
            || row_map.dtype() != DType::U32
            || !row_map.device().same_device(&model.device)
            || capacity == 0
            || capacity > MAX_CONTEXT
        {
            bail!("unsupported static graph decode geometry/device");
        }
        let mut expected_lengths = Vec::with_capacity(physical_rows);
        let mut expected_positions = Vec::with_capacity(physical_rows);
        for cache in caches {
            if cache.layers.len() != model.blocks.len()
                || cache.length >= capacity
                || cache.absolute_position() >= MAX_CONTEXT
                || cache.layers.iter().any(|layer| layer.len() != cache.length)
            {
                bail!("static graph input cache is inconsistent or full");
            }
            expected_lengths.push(cache.length);
            expected_positions.push(cache.absolute_position());
        }
        let lengths = Tensor::from_vec(
            expected_lengths
                .iter()
                .map(|&v| v as u32)
                .collect::<Vec<_>>(),
            physical_rows,
            &model.device,
        )?;
        let positions = Tensor::from_vec(
            expected_positions
                .iter()
                .map(|&v| v as u32)
                .collect::<Vec<_>>(),
            physical_rows,
            &model.device,
        )?;
        let mut keys = Vec::with_capacity(model.blocks.len());
        let mut values = Vec::with_capacity(model.blocks.len());
        let shape = (physical_rows * arch.heads, capacity, arch.head_width);
        for index in 0..model.blocks.len() {
            let key = Tensor::zeros(shape, DType::F32, &model.device)?;
            let value = Tensor::zeros(shape, DType::F32, &model.device)?;
            for (row, cache) in caches.iter().enumerate() {
                if cache.length > 0 {
                    let source_key = cache.layers[index]
                        .key()?
                        .context("graph source KV key absent")?
                        .to_dtype(DType::F32)?
                        .contiguous()?;
                    let source_value = cache.layers[index]
                        .value()?
                        .context("graph source KV value absent")?
                        .to_dtype(DType::F32)?
                        .contiguous()?;
                    key.narrow(0, row * arch.heads, arch.heads)?
                        .slice_set(&source_key, 1, 0)?;
                    value.narrow(0, row * arch.heads, arch.heads)?.slice_set(
                        &source_value,
                        1,
                        0,
                    )?;
                }
            }
            keys.push(key);
            values.push(value);
        }
        let mut qk_rope_parameters = Vec::with_capacity(model.blocks.len());
        let rope_cos = model.rope_cos.flatten_all()?;
        let rope_sin = model.rope_sin.flatten_all()?;
        for block in &model.blocks {
            qk_rope_parameters.push(
                Tensor::cat(&[&block.q_norm, &block.k_norm, &rope_cos, &rope_sin], 0)?
                    .contiguous()?,
            );
        }
        let termination_bias = model
            .termination_bias
            .unsqueeze(0)?
            .broadcast_as((physical_rows, 2))?
            .contiguous()?;
        let storage_identities = Some(
            caches
                .iter()
                .map(|cache| {
                    cache
                        .layers
                        .iter()
                        .map(|layer| {
                            layer
                                .graph_storage_identity()
                                .context("source graph KV allocation absent")
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?,
        );
        let pinned_views = caches.iter().map(|cache| (**cache).clone()).collect();
        let mut weight_pins = vec![
            model.embedding.clone(),
            model.final_norm.clone(),
            model.termination_weight.clone(),
            model.termination_bias.clone(),
            model.rope_cos.clone(),
            model.rope_sin.clone(),
        ];
        for block in &model.blocks {
            weight_pins.extend([
                block.norm1.clone(),
                block.qkv.clone(),
                block.q_norm.clone(),
                block.k_norm.clone(),
                block.projection.clone(),
                block.norm2.clone(),
                block.gate_up.clone(),
                block.down.clone(),
            ]);
        }
        Ok(Self {
            physical_rows,
            capacity,
            positions,
            lengths,
            row_map: row_map.clone(),
            keys,
            values,
            qk_rope_parameters,
            termination_bias,
            storage_identities,
            pinned_views,
            expected_lengths,
            expected_positions,
            adopted: false,
            _weight_pins: weight_pins,
        })
    }

    /// Set small, mutable graph inputs outside capture/replay. No scalar is
    /// copied from GPU to CPU; host cache counters are already authoritative.
    pub fn set_positions(&self, caches: &[&KvCache], model: &PrimitiveModel) -> Result<()> {
        self.validate_caches(caches)?;
        let lengths = Tensor::from_vec(
            caches
                .iter()
                .map(|cache| cache.length as u32)
                .collect::<Vec<_>>(),
            self.physical_rows,
            &model.device,
        )?;
        let positions = Tensor::from_vec(
            caches
                .iter()
                .map(|cache| cache.absolute_position() as u32)
                .collect::<Vec<_>>(),
            self.physical_rows,
            &model.device,
        )?;
        self.lengths.slice_set(&lengths, 0, 0)?;
        self.positions.slice_set(&positions, 0, 0)?;
        Ok(())
    }

    pub fn validate_caches(&self, caches: &[&KvCache]) -> Result<()> {
        if caches.len() != self.physical_rows {
            bail!("graph physical row count changed");
        }
        for (row, cache) in caches.iter().enumerate() {
            if cache.length != self.expected_lengths[row]
                || cache.absolute_position() != self.expected_positions[row]
                || cache.length >= self.capacity
                || cache.absolute_position() >= MAX_CONTEXT
            {
                bail!("graph cache identity/position changed or capacity exhausted");
            }
            if let Some(identities) = &self.storage_identities {
                if cache.layers.len() != identities[row].len()
                    || cache
                        .layers
                        .iter()
                        .zip(&identities[row])
                        .any(|(layer, &identity)| layer.graph_storage_identity() != Some(identity))
                {
                    bail!("graph KV backing was replaced, cloned for write or reordered");
                }
            }
        }
        Ok(())
    }

    /// Called while recording a graph. Everything returned depends on GPU
    /// input buffers; attention capacity and tensor ranks remain static.
    pub fn metadata(&self, model: &PrimitiveModel) -> Result<GraphFrameMetadata> {
        let heads = model.architecture.heads;
        let dim = model.architecture.head_width;
        Ok(GraphFrameMetadata {
            write_indices: graph_ops::indices(&self.lengths, heads, dim)?,
            valid_lengths: self.lengths.affine(1., 1.)?,
        })
    }

    pub fn forward(
        &self,
        model: &PrimitiveModel,
        logical_records: &Tensor,
        metadata: &GraphFrameMetadata,
        scope: OutputScope,
    ) -> Result<DevicePrediction> {
        let arch = &model.architecture;
        let physical = self.physical_rows;
        let heads = arch.heads;
        let dim = arch.head_width;
        let mut x = graph_ops::embedding(logical_records, &model.embedding, &self.row_map)?;
        for (index, block) in model.blocks.iter().enumerate() {
            let normalized = candle_nn::ops::rms_norm(&x, &block.norm1, 1e-6)?;
            let qkv = normalized.matmul(&block.qkv.t()?)?;
            let packed = graph_ops::qkv(
                &qkv,
                &self.qk_rope_parameters[index],
                &self.positions,
                heads,
                dim,
            )?;
            let q = packed
                .narrow(0, 0, 1)?
                .reshape((physical * heads, 1, dim))?
                .to_dtype(DType::F32)?;
            let k = packed
                .narrow(0, 1, 1)?
                .reshape((physical * heads, 1, dim))?
                .to_dtype(DType::F32)?;
            let v = packed
                .narrow(0, 2, 1)?
                .reshape((physical * heads, 1, dim))?
                .to_dtype(DType::F32)?;
            self.keys[index].scatter_set(&metadata.write_indices, &k, 1)?;
            self.values[index].scatter_set(&metadata.write_indices, &v, 1)?;
            let split_tokens = if model.optional_split_k && self.capacity >= 256 {
                128
            } else {
                0
            };
            let attended = super::fused_attention::decode_attention(
                &q,
                &self.keys[index],
                &self.values[index],
                Some(&metadata.valid_lengths),
                heads,
                split_tokens,
            )?
            .to_dtype(model.dtype)?
            .reshape((physical, arch.width))?;
            x = (x + attended.matmul(&block.projection.t()?)?)?;
            let normalized = candle_nn::ops::rms_norm(&x, &block.norm2, 1e-6)?;
            let gate_up = normalized.matmul(&block.gate_up.t()?)?;
            let mlp = graph_ops::swiglu(&gate_up)?.matmul(&block.down.t()?)?;
            x = (x + mlp)?;
        }
        let hidden = candle_nn::ops::rms_norm(&x, &model.final_norm, 1e-6)?;
        let (offset, count, termination) = scope.range()?;
        let attributes = hidden
            .matmul(&model.embedding.narrow(0, offset, count)?.t()?)?
            .to_dtype(DType::F32)?;
        let termination = if termination {
            Some(
                (hidden.matmul(&model.termination_weight.t()?)? + &self.termination_bias)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };
        Ok(DevicePrediction {
            attributes,
            termination,
            attribute_offset: offset,
        })
    }

    /// Attach the exact arena after a validated graph output. The graph only
    /// wrote the next scratch slot, so all older snapshots keep their prefix.
    pub fn commit(&mut self, model: &PrimitiveModel, caches: &mut [&mut KvCache]) -> Result<()> {
        let refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
        self.validate_caches(&refs)?;
        let first = !self.adopted;
        for (row, cache) in caches.iter_mut().enumerate() {
            let previous = cache.length;
            let next = previous + 1;
            if first {
                let mut layers = Vec::with_capacity(model.blocks.len());
                for index in 0..model.blocks.len() {
                    layers.push(LayerCache::adopt_graph_view(
                        self.keys[index].narrow(
                            0,
                            row * model.architecture.heads,
                            model.architecture.heads,
                        )?,
                        self.values[index].narrow(
                            0,
                            row * model.architecture.heads,
                            model.architecture.heads,
                        )?,
                        self.capacity,
                        next,
                    )?);
                }
                cache.layers = layers;
                cache.storage_dtype = DType::F32;
            } else {
                for layer in &mut cache.layers {
                    layer.advance_graph_length(previous, next)?;
                }
            }
            cache.length = next;
            self.expected_lengths[row] = next;
            self.expected_positions[row] += 1;
        }
        if first {
            self.storage_identities = Some(
                caches
                    .iter()
                    .map(|cache| {
                        cache
                            .layers
                            .iter()
                            .map(|layer| layer.graph_storage_identity().unwrap())
                            .collect()
                    })
                    .collect(),
            );
            self.pinned_views = caches.iter().map(|cache| (**cache).clone()).collect();
            self.adopted = true;
        }
        Ok(())
    }
}

pub fn graph_finite(input: &Tensor, guidance: usize) -> Result<Tensor> {
    graph_ops::finite(input, guidance)
}
pub fn graph_pack(
    records: &Tensor,
    terms: &Tensor,
    valid: &Tensor,
    guidance: usize,
) -> Result<Tensor> {
    graph_ops::pack(records, terms, valid, guidance)
}

impl PrimitiveModel {
    /// A ready lane can still hold Graph F32 KV while another lane has just
    /// rebuilt a compact BF16 region. Ordinary batched execution requires one
    /// storage dtype. Promote committed data into independent replacements;
    /// do not mutate any original lane until all allocations/copies succeed.
    pub fn unify_graph_fallback_caches(&self, caches: &mut [&mut KvCache]) -> Result<()> {
        unify_fallback_caches(caches)
    }
}

fn unify_fallback_caches(caches: &mut [&mut KvCache]) -> Result<()> {
    if caches.is_empty() {
        return Ok(());
    }
    let first = caches[0].storage_dtype;
    if caches.iter().all(|cache| cache.storage_dtype == first) {
        return Ok(());
    }
    if caches
        .iter()
        .any(|cache| !matches!(cache.storage_dtype, DType::BF16 | DType::F32))
    {
        bail!("graph fallback can only unify BF16/F32 committed KV");
    }
    let layers = caches[0].layers.len();
    for cache in caches.iter() {
        if cache.layers.len() != layers
            || cache.layers.iter().any(|layer| layer.len() != cache.length)
        {
            bail!("graph fallback cache metadata is inconsistent");
        }
        for layer in &cache.layers {
            if cache.length > 0 {
                let key = layer.key()?.context("fallback committed key absent")?;
                let value = layer.value()?.context("fallback committed value absent")?;
                if key.dtype() != cache.storage_dtype
                    || value.dtype() != cache.storage_dtype
                    || key.dims() != value.dims()
                {
                    bail!("graph fallback cache storage differs from its dtype metadata");
                }
            }
        }
    }
    let mut replacements = Vec::with_capacity(caches.len());
    for cache in caches.iter() {
        if cache.storage_dtype == DType::F32 {
            replacements.push(None);
            continue;
        }
        let mut promoted = Vec::with_capacity(layers);
        for layer in &cache.layers {
            let mut output = LayerCache::default();
            if cache.length > 0 {
                let key = layer
                    .key()?
                    .context("fallback key absent")?
                    .to_dtype(DType::F32)?
                    .contiguous()?;
                let value = layer
                    .value()?
                    .context("fallback value absent")?
                    .to_dtype(DType::F32)?
                    .contiguous()?;
                let _ = output.query(&key, &value, 0, true)?;
            }
            promoted.push(output);
        }
        replacements.push(Some(KvCache {
            layers: promoted,
            length: cache.length,
            position_offset: cache.position_offset,
            storage_dtype: DType::F32,
        }));
    }
    for (cache, replacement) in caches.iter_mut().zip(replacements) {
        if let Some(replacement) = replacement {
            **cache = replacement;
        }
    }
    Ok(())
}

#[cfg(test)]
mod fallback_tests {
    use super::*;
    fn cache(dtype: DType, value: f32, length: usize, offset: usize) -> KvCache {
        let device = Device::Cpu;
        let keys = Tensor::full(value, (2, length, 4), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let values = Tensor::full(value + 2., (2, length, 4), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let mut layer = LayerCache::default();
        let _ = layer.query(&keys, &values, 0, true).unwrap();
        KvCache {
            layers: vec![layer],
            length,
            position_offset: offset,
            storage_dtype: dtype,
        }
    }
    fn key(cache: &KvCache) -> Vec<f32> {
        cache.layers[0]
            .key()
            .unwrap()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }
    #[test]
    fn mixed_fallback_promotes_only_committed_data_and_keeps_snapshots_and_positions() {
        let mut compact = cache(DType::BF16, 1.5, 3, 7);
        let snapshot = compact.clone();
        let mut graph = cache(DType::F32, 4.25, 5, 2);
        let graph_id = graph.layers[0].graph_storage_identity();
        let snapshot_id = snapshot.layers[0].graph_storage_identity();
        unify_fallback_caches(&mut [&mut compact, &mut graph]).unwrap();
        assert_eq!(compact.storage_dtype, DType::F32);
        assert_eq!((compact.length, compact.position_offset), (3, 7));
        assert_eq!(key(&compact), key(&snapshot));
        assert_eq!(key(&graph), vec![4.25; 2 * 5 * 4]);
        assert_eq!(graph.layers[0].graph_storage_identity(), graph_id);
        assert_eq!(snapshot.storage_dtype, DType::BF16);
        assert_eq!(snapshot.layers[0].graph_storage_identity(), snapshot_id);
        assert_ne!(compact.layers[0].graph_storage_identity(), snapshot_id);
    }
    #[test]
    fn mixed_fallback_validation_fails_before_any_lane_is_replaced() {
        let mut compact = cache(DType::BF16, 1., 3, 0);
        let compact_id = compact.layers[0].graph_storage_identity();
        let mut malformed = cache(DType::F32, 2., 4, 0);
        malformed.length = 5;
        assert!(unify_fallback_caches(&mut [&mut compact, &mut malformed]).is_err());
        assert_eq!(compact.storage_dtype, DType::BF16);
        assert_eq!(compact.layers[0].graph_storage_identity(), compact_id);
        assert_eq!(key(&compact), vec![1.; 2 * 3 * 4]);
    }
}
