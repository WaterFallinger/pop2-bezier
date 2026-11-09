//! Persistent signed INT8 history with one FP32 scale per token/head.
//! The ordinary arena supplies its COW, geometric growth and ragged batch
//! ownership; packed values and scales are independent typed arenas.
use super::{cache::LayerCache, int8_kv_ops, write_plan::BatchWritePlan};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};

#[derive(Clone, Default)]
pub(super) struct Int8LayerCache {
    packed: LayerCache,
    scales: LayerCache,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn data(heads: usize, count: usize, bias: f32) -> Tensor {
        Tensor::from_vec((0..heads * count * 64).map(|i|
            (i as f32 * 0.173).sin() * 2. + bias).collect::<Vec<_>>(),
            (heads, count, 64), &Device::Cpu).unwrap()
    }
    fn bytes(tensor: &Tensor) -> Vec<u8> {
        tensor.contiguous().unwrap().flatten_all().unwrap().to_vec1::<u8>().unwrap()
    }
    fn floats(tensor: &Tensor) -> Vec<f32> {
        tensor.to_dtype(DType::F32).unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }
    fn prefix(read: &Int8Read, heads: usize, row: usize, count: usize) -> (Vec<u8>, Vec<f32>) {
        (bytes(&read.key.narrow(0, row * heads, heads).unwrap().narrow(1, 0, count).unwrap()),
         floats(&read.key_scale.narrow(0, row * heads, heads).unwrap().narrow(1, 0, count).unwrap()))
    }

    #[test]
    fn history_is_persistently_eight_bit_with_fp32_token_head_scales() {
        let mut cache = LayerCache::default();
        let k = data(2, 3, 0.);
        let v = data(2, 3, 1.);
        let read = cache.query_int8(&k, &v, 0, true).unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(read.key.dtype(), DType::U8);
        assert_eq!(read.value.dtype(), DType::U8);
        assert_eq!(read.key_scale.dtype(), DType::F32);
        assert_eq!(read.key.dims(), [2, 64, 64]);
        assert_eq!(read.key_scale.dims(), [2, 64]);
        assert_eq!(cache.reserved_bytes(), 2 * 64 * (64 * 2 + 4 * 2));
        assert_eq!(cache.int8_scale_bytes(), 2 * 64 * 4 * 2);
        // 136 bytes/head/token, compared with 256 for BF16 K+V.
        assert_eq!(cache.reserved_bytes() * 1000 / (2 * 64 * 64 * 4), 531);
        assert!(cache.key().is_err(), "ordinary reads must not silently expand INT8 history");
    }

    #[test]
    fn nine_partial_attributes_preserve_history_and_clone_commit_is_copy_on_write() {
        let mut cache = LayerCache::default();
        let initial = data(2, 3, 0.);
        let read = cache.query_int8(&initial, &initial, 0, true).unwrap();
        let expected = prefix(&read, 2, 0, 3);
        let mut branch = cache.clone();
        for field in 0..9 {
            let scratch = data(2, 1, field as f32 * 5.);
            let read = cache.query_int8(&scratch, &scratch, 3, false).unwrap();
            assert_eq!(prefix(&read, 2, 0, 3), expected);
            assert_eq!(cache.len(), 3);
        }
        let complete = data(2, 1, -5.);
        cache.query_int8(&complete, &complete, 3, true).unwrap();
        assert_eq!(cache.len(), 4);
        assert_eq!(branch.len(), 3);
        let read = branch.query_int8(&complete, &complete, 3, false).unwrap();
        assert_eq!(prefix(&read, 2, 0, 3), expected);
        assert!(cache.query_int8(&complete, &complete, 3, true).is_err());
    }

    #[test]
    fn ragged_batch_reuses_packed_arenas_without_losing_independent_history() {
        let heads = 2;
        let lengths = [1, 3, 2];
        let mut caches = vec![LayerCache::default(); lengths.len()];
        let mut expected = Vec::new();
        for (row, (&length, cache)) in lengths.iter().zip(&mut caches).enumerate() {
            let records = data(heads, length, row as f32);
            let read = cache.query_int8(&records, &records, 0, true).unwrap();
            expected.push(prefix(&read, heads, 0, length));
        }
        let mut saved = caches[0].clone();
        let current = Tensor::stack(&(0..3).map(|row| data(heads, 1, 10. + row as f32)).collect::<Vec<_>>(), 0).unwrap();
        let plan = BatchWritePlan::new(&lengths, heads, 64, &Device::Cpu).unwrap();
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        let read = LayerCache::query_int8_batch(&mut refs, &current, &current, &lengths, false, Some(&plan)).unwrap();
        drop(refs);
        assert_eq!(read.key.dims(), [3 * heads, 64, 64]);
        let arena = read.key.id();
        for row in 0..3 { assert_eq!(prefix(&read, heads, row, lengths[row]), expected[row]); }
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        let next = LayerCache::query_int8_batch(&mut refs, &current, &current, &lengths, true, Some(&plan)).unwrap();
        drop(refs);
        assert_eq!(next.key.id(), arena, "stable batch arena should be reused");
        for row in 0..3 {
            assert_eq!(caches[row].len(), lengths[row] + 1);
            assert_eq!(prefix(&next, heads, row, lengths[row]), expected[row]);
        }
        let scratch = data(heads, 1, -99.);
        let saved_read = saved.query_int8(&scratch, &scratch, lengths[0], false).unwrap();
        assert_eq!(prefix(&saved_read, heads, 0, lengths[0]), expected[0]);
    }

    #[test]
    fn geometric_growth_copies_quantized_values_and_scales_without_requantizing() {
        let mut cache = LayerCache::default();
        let initial = data(2, 63, 0.);
        let before = cache.query_int8(&initial, &initial, 0, true).unwrap();
        let expected = prefix(&before, 2, 0, 63);
        let tail = data(2, 2, 4.);
        let after = cache.query_int8(&tail, &tail, 63, true).unwrap();
        assert_eq!(cache.len(), 65);
        assert_eq!(after.key.dims(), [2, 128, 64]);
        assert_eq!(prefix(&after, 2, 0, 63), expected);
        let mut rebuilt = LayerCache::default();
        rebuilt.query_int8(&tail, &tail, 0, true).unwrap();
        assert_eq!(rebuilt.len(), 2, "a rebuilt context has its own local length");
        assert_eq!(cache.len(), 65);
    }

    #[test]
    fn prefill_uses_original_current_chunk_and_only_expands_history_temporarily() {
        let mut cache = LayerCache::default();
        let a = data(2, 3, 0.);
        let (keys, _) = cache.query_int8_prefill(&a, &a, 0, true).unwrap();
        assert_eq!(floats(&keys), floats(&a));
        let b = data(2, 2, 7.);
        let (keys, _) = cache.query_int8_prefill(&b, &b, 3, true).unwrap();
        assert_eq!(floats(&keys.narrow(1, 3, 2).unwrap()), floats(&b));
        let read = cache.query_int8(&b.narrow(1, 0, 1).unwrap(), &b.narrow(1, 0, 1).unwrap(), 5, false).unwrap();
        let history = int8_kv_ops::dequantize(&read.key.narrow(1, 0, 5).unwrap().contiguous().unwrap(),
            &read.key_scale.narrow(1, 0, 5).unwrap().contiguous().unwrap(), DType::F32).unwrap();
        assert_eq!(history.dims(), [2, 5, 64]);
        assert_eq!(cache.len(), 5);
        assert!(cache.query(&b, &b, 5, false).is_err());
    }

    #[test]
    fn rebuilt_row_reuses_both_packed_and_scale_slots_with_one_history_copy_each() {
        for changed in [0,1,2] {
            let heads=2;
            let mut caches=vec![Int8LayerCache::default();3];
            for (row,cache) in caches.iter_mut().enumerate(){let x=data(heads,3,row as f32);cache.query(&x,&x,0,true).unwrap();}
            let current=Tensor::stack(&(0..3).map(|row|data(heads,1,8.+row as f32)).collect::<Vec<_>>(),0).unwrap();
            let mut refs=caches.iter_mut().collect::<Vec<_>>();
            let first=Int8LayerCache::query_batch(&mut refs,&current,&current,&[3;3],false,None).unwrap();drop(refs);
            let expected=(0..3).map(|row|prefix(&first,heads,row,3)).collect::<Vec<_>>();
            let scale_arena_id=caches[0].scales.batch_arena().unwrap().0.id();
            let packed_count=caches[(changed+1)%3].packed.batch_copied_history_rows().unwrap();
            let scale_count=caches[(changed+1)%3].scales.batch_copied_history_rows().unwrap();
            caches[changed]=Int8LayerCache::default();
            let rebuilt=data(heads,2,20.);let rebuilt_read=caches[changed].query(&rebuilt,&rebuilt,0,true).unwrap();
            let rebuilt_expected=prefix(&rebuilt_read,heads,0,2);
            let lengths=(0..3).map(|row|if row==changed{2}else{3}).collect::<Vec<_>>();
            let mut refs=caches.iter_mut().collect::<Vec<_>>();
            let next=Int8LayerCache::query_batch(&mut refs,&current,&current,&lengths,true,None).unwrap();drop(refs);
            assert_eq!(first.key.id(),next.key.id());assert_eq!(first.value.id(),next.value.id());
            assert_eq!(scale_arena_id,caches[0].scales.batch_arena().unwrap().0.id());
            for row in 0..3 {assert_eq!(prefix(&next,heads,row,lengths[row]),if row==changed{rebuilt_expected.clone()}else{expected[row].clone()});}
            assert_eq!(caches[0].packed.batch_copied_history_rows().unwrap()-packed_count,1);
            assert_eq!(caches[0].scales.batch_copied_history_rows().unwrap()-scale_count,1);
        }
    }

    #[test]
    fn truncated_int8_outer_packed_and_scales_keep_reserve_and_snapshot_values() {
        let mut cache=LayerCache::default();let initial=data(2,5,0.);
        let old=cache.query_int8(&initial,&initial,0,true).unwrap();let saved=prefix(&old,2,0,5);
        let mut snapshot=cache.clone();let reserved=cache.reserved_bytes();
        assert!(cache.truncate_committed(6).is_err());assert_eq!(cache.len(),5);
        cache.truncate_committed(2).unwrap();assert_eq!(cache.len(),2);assert_eq!(cache.reserved_bytes(),reserved);
        let tail=data(2,1,7.);let next=cache.query_int8(&tail,&tail,2,true).unwrap();
        assert_eq!(cache.len(),3);assert_eq!(prefix(&next,2,0,2),prefix(&old,2,0,2));
        let snapshot_read=snapshot.query_int8(&tail,&tail,5,false).unwrap();
        assert_eq!(snapshot.len(),5);assert_eq!(prefix(&snapshot_read,2,0,5),saved);
        cache.truncate_committed(0).unwrap();assert_eq!(cache.len(),0);assert_eq!(cache.reserved_bytes(),reserved);
        cache.query_int8(&tail,&tail,0,true).unwrap();assert_eq!(cache.len(),1);
    }

    #[test]
    fn detached_int8_snapshot_blocks_slot_reuse_in_both_typed_arenas() {
        let mut caches=vec![Int8LayerCache::default();2];
        for(row,cache)in caches.iter_mut().enumerate(){let x=data(2,3,row as f32);cache.query(&x,&x,0,true).unwrap();}
        let current=Tensor::stack(&[data(2,1,8.),data(2,1,9.)],0).unwrap();
        let mut refs=caches.iter_mut().collect::<Vec<_>>();
        let old=Int8LayerCache::query_batch(&mut refs,&current,&current,&[3,3],false,None).unwrap();drop(refs);
        let old_scale_id=caches[0].scales.batch_arena().unwrap().0.id();
        let expected=prefix(&old,2,1,3);let mut snapshot=caches[1].clone();
        caches[1]=Int8LayerCache::default();let rebuilt=data(2,2,30.);caches[1].query(&rebuilt,&rebuilt,0,true).unwrap();
        let mut refs=caches.iter_mut().collect::<Vec<_>>();
        let next=Int8LayerCache::query_batch(&mut refs,&current,&current,&[3,2],true,None).unwrap();drop(refs);
        assert_ne!(old.key.id(),next.key.id());assert_ne!(old_scale_id,caches[0].scales.batch_arena().unwrap().0.id());
        let saved=snapshot.query(&data(2,1,-5.),&data(2,1,-5.),3,false).unwrap();
        assert_eq!(prefix(&saved,2,0,3),expected);
    }

    #[test]
    fn inconsistent_int8_lengths_reject_truncation_without_partial_metadata_changes() {
        let mut cache=Int8LayerCache::default();let initial=data(2,3,0.);
        cache.query(&initial,&initial,0,true).unwrap();cache.scales.truncate_committed(2).unwrap();
        assert!(cache.truncate_committed(0).is_err());
        assert_eq!(cache.packed.len(),3);assert_eq!(cache.scales.len(),2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an explicitly scheduled CUDA test slot; no model weights are loaded"]
    fn cuda_packed_batch_attention_preserves_scratch_history_and_cow() {
        let ordinal = std::env::var("PAINTER_TEST_CUDA_ORDINAL").ok()
            .map(|s| s.parse::<usize>().unwrap()).unwrap_or(0);
        let device = Device::new_cuda(ordinal).unwrap();
        int8_kv_ops::prepare(&device).unwrap();
        let heads = 2;
        let mut prototype = LayerCache::default();
        let initial = data(heads, 3, 0.).to_device(&device).unwrap().to_dtype(DType::BF16).unwrap();
        let first = prototype.query_int8(&initial, &initial, 0, true).unwrap();
        let expected = prefix(&first, heads, 0, 3);
        let mut caches = vec![prototype.clone(), prototype.clone()];
        let current = Tensor::stack(&[data(heads, 1, 0.37), data(heads, 1, -0.51)], 0).unwrap()
            .to_device(&device).unwrap().to_dtype(DType::BF16).unwrap();
        let q = Tensor::stack(&[data(heads, 1, 0.07), data(heads, 1, -0.19)], 0).unwrap()
            .reshape((4, 1, 64)).unwrap().to_device(&device).unwrap();
        let scratch = current.to_dtype(DType::F32).unwrap().reshape((4, 1, 64)).unwrap();
        let past = Tensor::from_vec(vec![3u32, 3], 2, &device).unwrap();
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        let read = LayerCache::query_int8_batch(&mut refs, &current, &current, &[3, 3], false, None).unwrap();
        drop(refs);
        assert_eq!(prefix(&read, heads, 0, 3), expected);
        assert_eq!(prefix(&read, heads, 1, 3), expected);
        let historical = int8_kv_ops::dequantize(&read.key, &read.key_scale, DType::BF16).unwrap()
            .to_dtype(DType::F32).unwrap();
        let qcpu = floats(&q);
        let hcpu = floats(&historical);
        let scpu = floats(&scratch);
        let capacity = read.key.dim(1).unwrap();
        let reference = (0..4).flat_map(|row| {
            let scores = (0..=3).map(|t| (0..64).map(|d|
                qcpu[row * 64 + d] * if t == 3 { scpu[row * 64 + d] } else { hcpu[(row * capacity + t) * 64 + d] })
                .sum::<f32>() * 0.125).collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probabilities = scores.iter().map(|&s| (s - maximum).exp()).collect::<Vec<_>>();
            let total = probabilities.iter().sum::<f32>();
            (0..64).map(|d| (0..=3).map(|t| probabilities[t]
                * if t == 3 { scpu[row * 64 + d] } else { hcpu[(row * capacity + t) * 64 + d] })
                .sum::<f32>() / total).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        for split in [32, 128] {
            let actual = int8_kv_ops::attention(&q, &read.key, &read.value, &read.key_scale, &read.value_scale,
                &scratch, &scratch, &past, heads, split).unwrap();
            assert!(floats(&actual).iter().zip(&reference).all(|(a, b)| a.is_finite() && (a - b).abs() < 1e-4));
        }
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        LayerCache::query_int8_batch(&mut refs, &current, &current, &[3, 3], true, None).unwrap();
        drop(refs);
        assert!(caches.iter().all(|cache| cache.len() == 4));
        assert_eq!(prototype.len(), 3);
        let scratch = current.narrow(0, 0, 1).unwrap().reshape((heads, 1, 64)).unwrap();
        let still_old = prototype.query_int8(&scratch, &scratch, 3, false).unwrap();
        assert_eq!(prefix(&still_old, heads, 0, 3), expected);
        device.synchronize().unwrap();
    }
}

pub(super) struct Int8Read {
    pub key: Tensor,
    pub value: Tensor,
    pub key_scale: Tensor,
    pub value_scale: Tensor,
}

impl Int8LayerCache {
    pub fn len(&self) -> usize { self.packed.len() }
    pub(super) fn validate_truncate_committed(&self, new_len: usize) -> Result<()> {
        if self.packed.len() != self.scales.len() { bail!("INT8 packed/scales committed lengths differ"); }
        self.packed.validate_truncate_committed(new_len)?;
        self.scales.validate_truncate_committed(new_len)?;
        Ok(())
    }
    pub(super) fn truncate_committed(&mut self, new_len: usize) -> Result<()> {
        self.validate_truncate_committed(new_len)?;
        self.packed.truncate_committed(new_len)?;
        self.scales.truncate_committed(new_len)?;
        Ok(())
    }
    pub fn reserved_bytes(&self) -> usize { self.packed.reserved_bytes() + self.scales.reserved_bytes() }
    pub fn scale_bytes(&self) -> usize { self.scales.reserved_bytes() }

    fn encoded(k: &Tensor, v: &Tensor, commit: bool) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (heads, tokens, width) = k.dims3()?;
        if heads == 0 || tokens == 0 || width != 64 || v.dims3()? != (heads, tokens, width)
            || !matches!(k.dtype(), DType::BF16 | DType::F32) || v.dtype() != k.dtype()
            || !k.device().same_device(v.device()) {
            bail!("INT8 history requires same-device BF16/F32[heads,tokens,64] K/V");
        }
        if commit {
            let (key, key_scale) = int8_kv_ops::quantize(&k.contiguous()?)?;
            let (value, value_scale) = int8_kv_ops::quantize(&v.contiguous()?)?;
            Ok((key, value, key_scale.unsqueeze(2)?, value_scale.unsqueeze(2)?))
        } else {
            // Partial attributes do not quantize current K/V. The attention
            // kernel reads their original floating scratch directly. This
            // zero-valued, uncommitted arena slot is never historical input.
            let placeholder = Tensor::zeros((heads, tokens, width), DType::U8, k.device())?;
            let scale = Tensor::ones((heads, tokens, 1), DType::F32, k.device())?;
            Ok((placeholder.clone(), placeholder, scale.clone(), scale))
        }
    }

    pub fn query(&mut self, k: &Tensor, v: &Tensor, past: usize, commit: bool) -> Result<Int8Read> {
        if self.packed.len() != past || self.scales.len() != past {
            bail!("INT8 packed/scales committed lengths differ from supplied history");
        }
        let (key, value, key_scale, value_scale) = Self::encoded(k, v, commit)?;
        self.packed.query(&key, &value, past, commit)?;
        self.scales.query(&key_scale, &value_scale, past, commit)?;
        let (key, value) = self.packed.arena().context("INT8 packed arena absent")?;
        let (key_scale, value_scale) = self.scales.arena().context("INT8 scale arena absent")?;
        Ok(Int8Read { key, value, key_scale: key_scale.squeeze(2)?, value_scale: value_scale.squeeze(2)? })
    }

    pub fn query_batch(
        caches: &mut [&mut Self], k: &Tensor, v: &Tensor, past: &[usize], commit: bool,
        plan: Option<&BatchWritePlan>,
    ) -> Result<Int8Read> {
        let (batch, heads, queries, width) = k.dims4()?;
        if batch == 0 || width != 64 || queries != 1 || caches.len() != batch || past.len() != batch
            || v.dims() != k.dims() {
            bail!("INT8 batch requires [B,H,1,64] scratch and matching row histories");
        }
        for (cache, &length) in caches.iter().zip(past) {
            if cache.packed.len() != length || cache.scales.len() != length {
                bail!("INT8 batch packed/scales history lengths disagree");
            }
        }
        let rows = batch.checked_mul(heads).context("INT8 physical head rows overflow")?;
        let (key, value, key_scale, value_scale) = Self::encoded(
            &k.reshape((rows, 1, 64))?.contiguous()?,
            &v.reshape((rows, 1, 64))?.contiguous()?, commit)?;
        {
            let mut packed = caches.iter_mut().map(|cache| &mut cache.packed).collect::<Vec<_>>();
            LayerCache::query_batch_with_plan(&mut packed,
                &key.reshape((batch, heads, 1, 64))?, &value.reshape((batch, heads, 1, 64))?, past, commit, plan)?;
        }
        {
            let mut scales = caches.iter_mut().map(|cache| &mut cache.scales).collect::<Vec<_>>();
            // A 64-channel scatter plan cannot index the single scale channel.
            // Its separate tiny U32 plan is cached by the shared scale arena.
            LayerCache::query_batch_with_plan(&mut scales,
                &key_scale.reshape((batch, heads, 1, 1))?, &value_scale.reshape((batch, heads, 1, 1))?, past, commit, None)?;
        }
        let (key, value) = caches[0].packed.batch_arena().context("INT8 batch packed arena absent")?;
        let (key_scale, value_scale) = caches[0].scales.batch_arena().context("INT8 batch scale arena absent")?;
        Ok(Int8Read { key, value, key_scale: key_scale.squeeze(2)?, value_scale: value_scale.squeeze(2)? })
    }

    /// Only prefill uses this transient floating prefix, matching the original
    /// server adapter. Single-token decode never calls this method.
    pub fn prefill_prefix(&self, dtype: DType) -> Result<Option<(Tensor, Tensor)>> {
        let Some(key) = self.packed.key()? else { return Ok(None); };
        let value = self.packed.value()?.context("INT8 value history absent")?;
        let key_scale = self.scales.key()?.context("INT8 key scales absent")?.squeeze(2)?;
        let value_scale = self.scales.value()?.context("INT8 value scales absent")?.squeeze(2)?;
        let key = int8_kv_ops::dequantize(&key.contiguous()?, &key_scale.contiguous()?, DType::BF16)?.to_dtype(dtype)?;
        let value = int8_kv_ops::dequantize(&value.contiguous()?, &value_scale.contiguous()?, DType::BF16)?.to_dtype(dtype)?;
        Ok(Some((key, value)))
    }
}
