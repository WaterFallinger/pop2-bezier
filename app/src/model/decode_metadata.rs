//! Read-only decode metadata reused across one primitive's partial queries.
//!
//! RoPE tables and returned metadata tensors must remain immutable. Tensor
//! clones share storage; callers must never use these as in-place destinations.

use super::{write_plan::BatchWritePlan, MAX_CONTEXT};
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, TensorId};

#[derive(Clone)]
pub(super) struct DecodeMetadata {
    pub cos: Tensor,
    pub sin: Tensor,
    /// Equal-length rows have no padded tail, so adding an all-zero mask is
    /// unnecessary. Ragged rows retain the previous right-padding mask.
    pub mask: Option<Tensor>,
    /// Actual committed-plus-scratch length of each physical row. Only ragged
    /// rows need this input; fused attention reads it on the device.
    pub row_lengths: Option<Tensor>,
    /// Shared across Transformer layers as well as all partial queries.
    pub write_plan: Option<BatchWritePlan>,
}

struct Entry {
    positions: Vec<u32>,
    lengths: Vec<usize>,
    heads: usize,
    rope_cos_id: TensorId,
    rope_sin_id: TensorId,
    metadata: DecodeMetadata,
}

#[derive(Default)]
pub(super) struct DecodeMetadataCache {
    entry: Option<Entry>,
}

impl DecodeMetadataCache {
    pub fn get(
        &mut self,
        positions: &[u32],
        lengths: &[usize],
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        device: &Device,
        heads: usize,
    ) -> Result<DecodeMetadata> {
        if positions.is_empty() || positions.len() != lengths.len() || heads == 0 {
            bail!("decode positions and lengths require the same nonempty batch");
        }
        let (context, half_width) = rope_cos.dims2()?;
        if context == 0 || half_width == 0 || rope_sin.dims2()? != (context, half_width) {
            bail!("decode RoPE tables require equal nonempty 2D shapes");
        }
        if rope_cos.dtype() != rope_sin.dtype()
            || !matches!(
                rope_cos.dtype(),
                DType::F16 | DType::BF16 | DType::F32 | DType::F64
            )
            || !rope_cos.device().same_device(device)
            || !rope_sin.device().same_device(device)
        {
            bail!("decode RoPE table precision or device mismatch");
        }
        for (&position, &length) in positions.iter().zip(lengths) {
            if position as usize >= context
                || length == 0
                || length > MAX_CONTEXT
                || length - 1 > position as usize
            {
                bail!("decode position or physical context length is out of range");
            }
        }
        if let Some(entry) = &self.entry {
            if entry.positions == positions
                && entry.lengths == lengths
                && entry.heads == heads
                && entry.rope_cos_id == rope_cos.id()
                && entry.rope_sin_id == rope_sin.id()
            {
                return Ok(entry.metadata.clone());
            }
        }

        let (cos, sin) = if positions.iter().all(|&p| p == positions[0]) {
            // Candle rope_i accepts 2D [query_length, half_width] tables shared
            // across the batch; no position upload or gather is necessary.
            let position = positions[0] as usize;
            (
                rope_cos.narrow(0, position, 1)?.contiguous()?,
                rope_sin.narrow(0, position, 1)?.contiguous()?,
            )
        } else {
            let indices = Tensor::from_slice(positions, positions.len(), device)?;
            (
                rope_cos
                    .index_select(&indices, 0)?
                    .unsqueeze(1)?
                    .contiguous()?,
                rope_sin
                    .index_select(&indices, 0)?
                    .unsqueeze(1)?
                    .contiguous()?,
            )
        };
        let max_length = *lengths.iter().max().unwrap();
        let (mask, write_plan, row_lengths) = if lengths.iter().all(|&length| length == max_length) {
            (None, None, None)
        } else {
            let values: Vec<f32> = lengths
                .iter()
                .flat_map(|&length| {
                    (0..max_length).map(move |i| if i < length { 0. } else { f32::NEG_INFINITY })
                })
                .collect();
            let mask = Tensor::from_vec(values, (lengths.len(), 1, 1, max_length), device)?;
            let past_lens: Vec<_> = lengths.iter().map(|&length| length - 1).collect();
            let width = half_width
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("decode head width overflow"))?;
            let plan = BatchWritePlan::new(&past_lens, heads, width, device)?;
            let actual_lengths: Vec<u32> = lengths.iter().map(|&length| length as u32).collect();
            let row_lengths = Tensor::from_vec(actual_lengths, lengths.len(), device)?;
            (Some(mask), Some(plan), Some(row_lengths))
        };
        let metadata = DecodeMetadata {
            cos,
            sin,
            mask,
            row_lengths,
            write_plan,
        };
        self.entry = Some(Entry {
            positions: positions.to_vec(),
            lengths: lengths.to_vec(),
            heads,
            rope_cos_id: rope_cos.id(),
            rope_sin_id: rope_sin.id(),
            metadata: metadata.clone(),
        });
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables(context: usize, half: usize) -> (Tensor, Tensor) {
        let angles: Vec<_> = (0..context)
            .flat_map(|p| (0..half).map(move |d| p as f32 * (d + 1) as f32 * 0.17))
            .collect();
        (
            Tensor::from_vec(
                angles.iter().map(|a| a.cos()).collect::<Vec<_>>(),
                (context, half),
                &Device::Cpu,
            )
            .unwrap(),
            Tensor::from_vec(
                angles.iter().map(|a| a.sin()).collect::<Vec<_>>(),
                (context, half),
                &Device::Cpu,
            )
            .unwrap(),
        )
    }
    fn flat(tensor: &Tensor) -> Vec<f32> {
        tensor
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    #[test]
    fn identical_keys_reuse_tensors_and_all_key_parts_invalidate() {
        let (cos, sin) = tables(32, 2);
        let mut cache = DecodeMetadataCache::default();
        let first = cache
            .get(&[7, 7], &[8, 3], &cos, &sin, &Device::Cpu, 2)
            .unwrap();
        let hit = cache
            .get(&[7, 7], &[8, 3], &cos, &sin, &Device::Cpu, 2)
            .unwrap();
        assert_eq!(first.cos.id(), hit.cos.id());
        assert_eq!(first.sin.id(), hit.sin.id());
        assert_eq!(first.row_lengths.as_ref().unwrap().id(), hit.row_lengths.as_ref().unwrap().id());
        assert_eq!(
            first.write_plan.as_ref().unwrap().indices().id(),
            hit.write_plan.as_ref().unwrap().indices().id()
        );
        assert_eq!(
            first.mask.as_ref().unwrap().id(),
            hit.mask.as_ref().unwrap().id()
        );
        let length_miss = cache
            .get(&[7, 7], &[8, 4], &cos, &sin, &Device::Cpu, 2)
            .unwrap();
        assert_ne!(
            first.mask.as_ref().unwrap().id(),
            length_miss.mask.as_ref().unwrap().id()
        );
        let position_miss = cache
            .get(&[8, 8], &[8, 4], &cos, &sin, &Device::Cpu, 2)
            .unwrap();
        assert_ne!(length_miss.cos.id(), position_miss.cos.id());
        let other_cos = cos.copy().unwrap();
        let table_miss = cache
            .get(&[8, 8], &[8, 4], &other_cos, &sin, &Device::Cpu, 2)
            .unwrap();
        assert_ne!(position_miss.cos.id(), table_miss.cos.id());
        let heads_miss = cache
            .get(&[8, 8], &[8, 4], &other_cos, &sin, &Device::Cpu, 3)
            .unwrap();
        assert_ne!(table_miss.cos.id(), heads_miss.cos.id());
        assert_ne!(
            table_miss.write_plan.as_ref().unwrap().indices().id(),
            heads_miss.write_plan.as_ref().unwrap().indices().id()
        );
        assert_eq!(
            heads_miss.write_plan.as_ref().unwrap().indices().dims(),
            &[6, 1, 4]
        );
    }

    #[test]
    fn ragged_mask_matches_previous_formula_and_equal_rows_need_none() {
        let (cos, sin) = tables(32, 2);
        let mut cache = DecodeMetadataCache::default();
        let lengths = [8, 3, 5, 8];
        let actual = cache
            .get(&[7, 7, 7, 7], &lengths, &cos, &sin, &Device::Cpu, 2)
            .unwrap();
        let expected: Vec<f32> = lengths
            .iter()
            .flat_map(|&n| (0..8).map(move |i| if i < n { 0. } else { f32::NEG_INFINITY }))
            .collect();
        assert_eq!(flat(actual.mask.as_ref().unwrap()), expected);
        assert_eq!(actual.mask.as_ref().unwrap().dims(), &[4, 1, 1, 8]);
        assert_eq!(actual.row_lengths.as_ref().unwrap().to_vec1::<u32>().unwrap(), vec![8, 3, 5, 8]);
        let equal = cache
            .get(&[7, 7, 7, 7], &[8, 8, 8, 8], &cos, &sin, &Device::Cpu, 2)
            .unwrap();
        assert!(equal.mask.is_none());
        assert!(equal.write_plan.is_none());
        assert!(equal.row_lengths.is_none());

        let scores = Tensor::from_vec(
            (0..4 * 2 * 8)
                .map(|i| (i % 11) as f32 - 5.)
                .collect::<Vec<_>>(),
            (4, 2, 1, 8),
            &Device::Cpu,
        )
        .unwrap();
        let zero = Tensor::zeros((4, 1, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let old = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&zero).unwrap()).unwrap();
        let new = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        assert_eq!(flat(&old), flat(&new));
    }

    #[test]
    fn shared_2d_rope_and_mixed_3d_rope_match_previous_gather() {
        let (cos, sin) = tables(32, 2);
        for (positions, lengths) in [
            (vec![7], vec![8]),
            (vec![7; 4], vec![8; 4]),
            // Conditional/unconditional rows have different physical lengths
            // but matching absolute RoPE positions.
            (vec![7; 4], vec![8, 3, 8, 3]),
            (vec![4, 7, 8, 12], vec![5, 8, 9, 13]),
        ] {
            let mut cache = DecodeMetadataCache::default();
            let metadata = cache
                .get(&positions, &lengths, &cos, &sin, &Device::Cpu, 2)
                .unwrap();
            let batch = positions.len();
            assert_eq!(
                metadata.cos.rank(),
                if positions.iter().all(|&p| p == positions[0]) {
                    2
                } else {
                    3
                }
            );
            let x = Tensor::from_vec(
                (0..batch * 2 * 4)
                    .map(|i| (i as f32 - 7.) * 0.125)
                    .collect::<Vec<_>>(),
                (batch, 2, 1, 4),
                &Device::Cpu,
            )
            .unwrap();
            let indices = Tensor::from_slice(&positions, batch, &Device::Cpu).unwrap();
            let old_cos = cos
                .index_select(&indices, 0)
                .unwrap()
                .unsqueeze(1)
                .unwrap()
                .contiguous()
                .unwrap();
            let old_sin = sin
                .index_select(&indices, 0)
                .unwrap()
                .unsqueeze(1)
                .unwrap()
                .contiguous()
                .unwrap();
            let old = candle_nn::rotary_emb::rope_i(&x, &old_cos, &old_sin).unwrap();
            let new = candle_nn::rotary_emb::rope_i(&x, &metadata.cos, &metadata.sin).unwrap();
            assert_eq!(flat(&old), flat(&new));
        }
    }

    #[test]
    fn invalid_inputs_do_not_replace_the_last_valid_entry() {
        let (cos, sin) = tables(32, 2);
        let mut cache = DecodeMetadataCache::default();
        let valid = cache
            .get(&[7, 7], &[8, 3], &cos, &sin, &Device::Cpu, 2)
            .unwrap();
        for (positions, lengths) in [
            (vec![], vec![]),
            (vec![7, 7], vec![8]),
            (vec![32], vec![8]),
            (vec![7], vec![0]),
            (vec![7], vec![MAX_CONTEXT + 1]),
            (vec![1], vec![3]),
        ] {
            assert!(cache
                .get(&positions, &lengths, &cos, &sin, &Device::Cpu, 2)
                .is_err());
        }
        let wrong_shape = Tensor::zeros((31, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(cache
            .get(&[7, 7], &[8, 3], &cos, &wrong_shape, &Device::Cpu, 2)
            .is_err());
        let wrong_type = sin.to_dtype(DType::F64).unwrap();
        assert!(cache
            .get(&[7, 7], &[8, 3], &cos, &wrong_type, &Device::Cpu, 2)
            .is_err());
        assert!(cache
            .get(&[7, 7], &[8, 3], &cos, &sin, &Device::Cpu, 0)
            .is_err());
        let hit = cache
            .get(&[7, 7], &[8, 3], &cos, &sin, &Device::Cpu, 2)
            .unwrap();
        assert_eq!(valid.cos.id(), hit.cos.id());
        assert_eq!(valid.mask.unwrap().id(), hit.mask.unwrap().id());
    }
}
