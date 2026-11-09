//! Read-only prepared sampler state shared by consecutive complete primitives.
//! Callers must not write into these tensors; records/uniforms are separate.

use super::device_sampling::{self, SamplingPlan};
use super::field_candidates;
use crate::model::{BINS, OFFSETS};
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};

type LaneSpec = (bool, Option<(usize, usize)>);

#[derive(Clone)]
pub(super) struct SamplingWorkspace {
    pub row_map: Tensor,
    pub field_indices: Tensor,
    pub plans: Vec<SamplingPlan>,
    /// Field-major, then logical-row: absolute token start and candidate count.
    pub candidate_ranges: Vec<Vec<(u32, usize)>>,
    pub forced: Vec<Option<Tensor>>,
}

struct Entry {
    specs: Vec<LaneSpec>,
    full_attribute_vocab: bool,
    alpha_bits: u32,
    guidance: usize,
    device: Device,
    workspace: SamplingWorkspace,
}

#[derive(Default)]
pub(super) struct SamplingWorkspaceCache {
    entry: Option<Entry>,
}

impl SamplingWorkspaceCache {
    pub fn get(
        &mut self,
        specs: &[LaneSpec],
        alpha: f32,
        guidance: usize,
        device: &Device,
    ) -> Result<SamplingWorkspace> {
        self.get_for_diagnostic(specs, alpha, guidance, device, false)
    }

    /// Only the fixed historical benchmark uses the full field vocabulary.
    /// It never relaxes the application's legal drawing candidates.
    pub fn get_for_diagnostic(
        &mut self,
        specs: &[LaneSpec],
        alpha: f32,
        guidance: usize,
        device: &Device,
        full_attribute_vocab: bool,
    ) -> Result<SamplingWorkspace> {
        if specs.is_empty() || !matches!(guidance, 1 | 2) {
            bail!("sampler workspace needs nonempty lanes and one/two guidance rows");
        }
        if !alpha.is_finite() || alpha <= 1. {
            bail!("sampler workspace alpha must be finite and greater than one");
        }
        if let Some(entry) = self.entry.as_ref().filter(|entry| {
            entry.specs == specs
                && entry.full_attribute_vocab == full_attribute_vocab
                && entry.alpha_bits == alpha.to_bits()
                && entry.guidance == guidance
                && entry.device.same_device(device)
        }) {
            return Ok(entry.workspace.clone());
        }

        let batch = specs.len();
        let physical = batch
            .checked_mul(guidance)
            .context("workspace row overflow")?;
        let field_cells = batch
            .checked_mul(9)
            .context("workspace field index overflow")?;
        u32::try_from(batch - 1).context("workspace logical row exceeds U32")?;
        // Resolve and validate all CPU ranges before introducing any GPU data.
        let mut candidate_ranges = Vec::with_capacity(9);
        for field in 0..9 {
            let rows = specs
                .iter()
                .map(|&(background, region)| {
                    let candidates = if full_attribute_vocab {
                        (OFFSETS[field]..OFFSETS[field] + BINS[field]).collect()
                    } else {
                        field_candidates(field, background, region)?
                    };
                    let start = *candidates
                        .first()
                        .context("empty sampler candidate range")?;
                    let end = OFFSETS[field]
                        .checked_add(BINS[field])
                        .context("attribute head overflow")?;
                    if start < OFFSETS[field] || candidates.last().copied().unwrap() >= end {
                        bail!("sampler candidate range is outside its attribute head");
                    }
                    if candidates
                        .iter()
                        .enumerate()
                        .any(|(i, &token)| start.checked_add(i as u32) != Some(token))
                    {
                        bail!("prepared sampler requires contiguous legal candidates");
                    }
                    Ok((start, candidates.len()))
                })
                .collect::<Result<Vec<_>>>()?;
            candidate_ranges.push(rows);
        }
        let row_map = (0..batch)
            .flat_map(|row| std::iter::repeat(row as u32).take(guidance))
            .collect::<Vec<_>>();
        let row_map = Tensor::from_vec(row_map, physical, device)?;
        let field_indices = (0..9)
            .flat_map(|field| std::iter::repeat(field as u32).take(batch))
            .collect::<Vec<_>>();
        debug_assert_eq!(field_indices.len(), field_cells);
        let field_indices = Tensor::from_vec(field_indices, (9, batch, 1), device)?;
        let mut plans = Vec::with_capacity(9);
        let mut forced = Vec::with_capacity(9);
        for field in 0..9 {
            let legal = candidate_ranges[field]
                .iter()
                .map(|&(start, count)| ((start - OFFSETS[field]) as usize, count))
                .collect::<Vec<_>>();
            plans.push(device_sampling::prepare(
                &legal,
                BINS[field] as usize,
                alpha,
                OFFSETS[field],
                device,
            )?);
            if candidate_ranges[field].iter().all(|&(_, count)| count == 1) {
                let tokens = candidate_ranges[field]
                    .iter()
                    .map(|&(start, _)| start)
                    .collect::<Vec<_>>();
                forced.push(Some(Tensor::from_vec(tokens, (batch, 1), device)?));
            } else {
                forced.push(None);
            }
        }
        let workspace = SamplingWorkspace {
            row_map,
            field_indices,
            plans,
            candidate_ranges,
            forced,
        };
        self.entry = Some(Entry {
            specs: specs.to_vec(),
            full_attribute_vocab,
            alpha_bits: alpha.to_bits(),
            guidance,
            device: device.clone(),
            workspace: workspace.clone(),
        });
        Ok(workspace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn historical_full_head_workspace_cannot_leak_into_normal_canvas_generation() -> Result<()> {
        let mut cache = SamplingWorkspaceCache::default();
        let specs = [(false, None)];
        let historical = cache.get_for_diagnostic(&specs, 4., 1, &Device::Cpu, true)?;
        assert_eq!(historical.candidate_ranges[5], vec![(OFFSETS[5], 256)]);
        let ordinary = cache.get(&specs, 4., 1, &Device::Cpu)?;
        assert_eq!(ordinary.candidate_ranges[5], vec![(OFFSETS[5] + 1, 2)]);
        assert_ne!(historical.row_map.id(), ordinary.row_map.id());
        Ok(())
    }

    #[test]
    fn repeated_specs_reuse_device_mapping_columns_and_forced_tokens() -> Result<()> {
        let mut cache = SamplingWorkspaceCache::default();
        let specs = [(true, None), (true, None)];
        let a = cache.get(&specs, 4., 2, &Device::Cpu)?;
        let b = cache.get(&specs, 4., 2, &Device::Cpu)?;
        assert_eq!(a.row_map.id(), b.row_map.id());
        assert_eq!(a.field_indices.id(), b.field_indices.id());
        assert_eq!(
            a.forced[5].as_ref().unwrap().id(),
            b.forced[5].as_ref().unwrap().id()
        );
        assert_eq!(a.row_map.to_vec1::<u32>()?, vec![0, 0, 1, 1]);
        assert_eq!(a.field_indices.dims(), &[9, 2, 1]);
        for field in 0..9 {
            let column = a.field_indices.narrow(0, field, 1)?.reshape((2, 1))?;
            assert_eq!(column.to_vec2::<u32>()?, vec![vec![field as u32]; 2]);
        }
        assert_eq!(
            a.forced[5].as_ref().unwrap().to_vec2::<u32>()?,
            vec![vec![OFFSETS[5]]; 2]
        );
        assert_eq!(a.forced.iter().filter(|v| v.is_some()).count(), 1);
        assert_eq!(a.plans.len(), 9);
        Ok(())
    }

    #[test]
    fn every_semantic_key_change_invalidates_the_workspace() -> Result<()> {
        let mut cache = SamplingWorkspaceCache::default();
        let specs = [(true, None), (false, Some((4, 3)))];
        let mut id = cache.get(&specs, 4., 2, &Device::Cpu)?.row_map.id();
        let changes = [
            (specs.to_vec(), 3.5, 2),
            (specs.to_vec(), 3.5, 1),
            (vec![(false, None), (false, Some((4, 3)))], 3.5, 1),
            (vec![(false, None), (false, Some((4, 4)))], 3.5, 1),
            (vec![(false, Some((4, 4))), (false, None)], 3.5, 1),
            (vec![(false, None)], 3.5, 1),
        ];
        for (next, alpha, guidance) in changes {
            let workspace = cache.get(&next, alpha, guidance, &Device::Cpu)?;
            assert_ne!(id, workspace.row_map.id());
            id = workspace.row_map.id();
        }
        Ok(())
    }

    #[test]
    fn region_ranges_and_mixed_background_masks_match_the_real_token_contract() -> Result<()> {
        let mut cache = SamplingWorkspaceCache::default();
        let workspace = cache.get(
            &[(false, Some((4, 3))), (false, Some((8, 9)))],
            4.,
            1,
            &Device::Cpu,
        )?;
        assert_eq!(workspace.candidate_ranges[0], vec![(768, 255), (128, 127)]);
        assert_eq!(
            workspace.candidate_ranges[1],
            vec![(OFFSETS[1], 255), (OFFSETS[1] + 128, 127)]
        );
        let sampled = device_sampling::sample_prepared(
            &Tensor::zeros((2, BINS[0] as usize), DType::F32, &Device::Cpu)?,
            &Tensor::from_vec(vec![0.9f32, 0.1], (2, 1), &Device::Cpu)?,
            &workspace.plans[0],
        )?;
        let tokens = sampled.tokens.to_vec2::<u32>()?;
        for (row, token) in tokens.iter().enumerate() {
            let (start, count) = workspace.candidate_ranges[0][row];
            assert!((start..start + count as u32).contains(&token[0]));
        }
        let mixed = cache.get(&[(true, None), (false, None)], 4., 2, &Device::Cpu)?;
        assert!(mixed.forced[5].is_none());
        let shapes = device_sampling::sample_prepared(
            &Tensor::zeros((2, BINS[5] as usize), DType::F32, &Device::Cpu)?,
            &Tensor::from_vec(vec![0.9f32, 0.9], (2, 1), &Device::Cpu)?,
            &mixed.plans[5],
        )?;
        let tokens = shapes.tokens.to_vec2::<u32>()?;
        assert_eq!(tokens[0][0], OFFSETS[5]);
        assert!((OFFSETS[5] + 1..=OFFSETS[5] + 2).contains(&tokens[1][0]));
        Ok(())
    }

    #[test]
    fn invalid_keys_do_not_evict_the_last_valid_workspace() -> Result<()> {
        let mut cache = SamplingWorkspaceCache::default();
        let specs = [(false, Some((8, 9)))];
        let first = cache.get(&specs, 4., 1, &Device::Cpu)?;
        assert!(cache.get(&[], 4., 1, &Device::Cpu).is_err());
        for guidance in [0, 3] {
            assert!(cache.get(&specs, 4., guidance, &Device::Cpu).is_err());
        }
        for alpha in [1., f32::NAN, f32::INFINITY] {
            assert!(cache.get(&specs, alpha, 1, &Device::Cpu).is_err());
        }
        assert!(cache
            .get(&[(false, Some((3, 0)))], 4., 1, &Device::Cpu)
            .is_err());
        assert!(cache
            .get(&[(false, Some((8, 64)))], 4., 1, &Device::Cpu)
            .is_err());
        assert_eq!(
            cache.get(&specs, 4., 1, &Device::Cpu)?.row_map.id(),
            first.row_map.id()
        );
        Ok(())
    }
}
