//! Device-resident FP32 Rényi support sampling. There is no host readback here.
//! Uniforms belong to independent caller-owned lane RNG streams. Equal-logit
//! sorting and FP32 reductions need not reproduce the legacy CPU sampler.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

pub(super) struct DeviceSample {
    pub tokens: Tensor,
    pub valid: Tensor,
    pub support: Tensor,
}

/// Immutable frame/attribute inputs. Prepare these before enqueuing partial
/// forwards so pageable HtoD uploads are never introduced between attributes.
#[derive(Clone)]
pub(super) struct SamplingPlan {
    batch: usize,
    choices: usize,
    alpha: f32,
    inverse_alpha_minus_one: f64,
    legal_mask: Tensor,
    legal_count: Tensor,
    log_legal_count: Tensor,
    placeholder: Tensor,
    zero_logits: Tensor,
    infinity_logits: Tensor,
    excluded_logits: Tensor,
    zero_rows: Tensor,
    one_rows: Tensor,
    infinity_rows: Tensor,
    one_index_rows: Tensor,
    token_base: Tensor,
    bounds: Tensor,
    token_offset: u32,
}

pub(super) fn prepare(
    legal: &[(usize, usize)],
    choices: usize,
    alpha: f32,
    token_offset: u32,
    device: &Device,
) -> Result<SamplingPlan> {
    let batch = legal.len();
    if batch == 0 || choices == 0 {
        bail!("device sampling requires a nonempty head and one legal range per row");
    }
    if !alpha.is_finite() || alpha <= 1. {
        bail!("device Rényi alpha must be finite and greater than one");
    }
    let last_choice = u32::try_from(choices - 1).context("head exceeds U32 token space")?;
    token_offset
        .checked_add(last_choice)
        .context("absolute token range overflow")?;
    let cells = batch
        .checked_mul(choices)
        .context("candidate mask size overflow")?;
    let mut mask = Vec::new();
    mask.try_reserve_exact(cells)
        .context("candidate mask allocation failed")?;
    let mut counts = Vec::with_capacity(batch);
    let mut placeholders = Vec::with_capacity(batch);
    let mut bounds = Vec::with_capacity(batch * 2);
    for &(start, count) in legal {
        let end = start.checked_add(count).context("legal range overflow")?;
        if count == 0 || end > choices {
            bail!("legal candidate range must be nonempty and within its head");
        }
        mask.extend((0..choices).map(|i| u8::from((start..end).contains(&i))));
        counts.push(count as f32);
        placeholders.push(token_offset + start as u32);
        bounds.push(u32::try_from(start)?);
        bounds.push(u32::try_from(count)?);
    }
    let legal_mask = Tensor::from_vec(mask, (batch, choices), device)?;
    let legal_count = Tensor::from_vec(counts, (batch, 1), device)?;
    let log_legal_count = legal_count.log()?;
    let placeholder = Tensor::from_vec(placeholders, (batch, 1), device)?;
    Ok(SamplingPlan {
        batch,
        choices,
        alpha,
        inverse_alpha_minus_one: 1. / (alpha - 1.) as f64,
        legal_mask,
        legal_count,
        log_legal_count,
        placeholder,
        zero_logits: Tensor::zeros((batch, choices), DType::F32, device)?,
        infinity_logits: Tensor::full(f32::INFINITY, (batch, choices), device)?,
        excluded_logits: Tensor::full(f32::NEG_INFINITY, (batch, choices), device)?,
        zero_rows: Tensor::zeros((batch, 1), DType::F32, device)?,
        one_rows: Tensor::ones((batch, 1), DType::F32, device)?,
        infinity_rows: Tensor::full(f32::INFINITY, (batch, 1), device)?,
        one_index_rows: Tensor::ones((batch, 1), DType::U32, device)?,
        token_base: Tensor::full(token_offset, (batch, 1), device)?,
        bounds: Tensor::from_vec(bounds, (batch, 2), device)?,
        token_offset,
    })
}

pub(super) fn sample_guided_prepared(
    raw: &Tensor,
    uniforms: &Tensor,
    plan: &SamplingPlan,
    guidance: usize,
    cfg: f32,
    fused: bool,
) -> Result<DeviceSample> {
    let fused_backend = raw.device().is_cuda();
    #[cfg(feature = "rocm")]
    let fused_backend = fused_backend || raw.device().is_rocm();
    if fused && fused_backend && plan.choices <= 2048 {
        let (tokens, valid, support) = super::fused_sampling::sample(
            raw,
            uniforms,
            &plan.bounds,
            plan.alpha,
            cfg,
            guidance,
            plan.token_offset,
        )?;
        return Ok(DeviceSample {
            tokens,
            valid,
            support,
        });
    }
    let guided = if guidance == 2 {
        let grouped = raw.reshape((plan.batch, 2, plan.choices))?;
        let c = grouped.narrow(1, 0, 1)?.squeeze(1)?;
        let u = grouped.narrow(1, 1, 1)?.squeeze(1)?;
        (&u + ((&c - &u)? * cfg as f64)?)?
    } else if guidance == 1 {
        raw.clone()
    } else {
        bail!("unsupported guidance row count")
    };
    sample_prepared(&guided, uniforms, plan)
}

/// Sample one absolute token per row. `legal` contains relative half-open
/// ranges represented as `(start, count)`. Invalid device values return a
/// deterministic legal placeholder with `valid=0`; callers must check that
/// flag before submitting a completed primitive. No device error is ignored.
pub(super) fn sample(
    logits: &Tensor,
    legal: &[(usize, usize)],
    uniforms: &Tensor,
    alpha: f32,
    token_offset: u32,
) -> Result<DeviceSample> {
    let (batch, choices) = logits.dims2()?;
    if legal.len() != batch {
        bail!("device sampling requires a nonempty head and one legal range per row");
    }
    let plan = prepare(legal, choices, alpha, token_offset, logits.device())?;
    sample_prepared(logits, uniforms, &plan)
}

/// Only device operations: no CPU-created constants, data uploads, host reads,
/// or explicit synchronization. `prepare` owns the metadata transfer boundary.
pub(super) fn sample_prepared(
    logits: &Tensor,
    uniforms: &Tensor,
    plan: &SamplingPlan,
) -> Result<DeviceSample> {
    let (batch, choices) = logits.dims2()?;
    if (batch, choices) != (plan.batch, plan.choices)
        || !logits.device().same_device(plan.legal_mask.device())
    {
        bail!("device sampling head shape/device differs from its prepared plan");
    }
    if logits.dtype() != DType::F32 || uniforms.dtype() != DType::F32 {
        bail!("device sampling logits and uniforms must be FP32");
    }
    if uniforms.dims2()? != (batch, 1) || !uniforms.device().same_device(logits.device()) {
        bail!("device sampling uniforms must be same-device [B,1]");
    }

    // Preserve a compact error flag while ensuring no NaN/Inf reaches sort,
    // float-to-index conversion, or a data-dependent gather.
    let finite_logits = logits.abs()?.lt(&plan.infinity_logits)?;
    let finite_rows = finite_logits
        .to_dtype(DType::F32)?
        .min_keepdim(1)?
        .eq(&plan.one_rows)?;
    let uniform_valid = uniforms
        .abs()?
        .lt(&plan.infinity_rows)?
        .mul(&uniforms.ge(&plan.zero_rows)?)?
        .mul(&uniforms.lt(&plan.one_rows)?)?;
    let safe_uniforms = uniform_valid.where_cond(uniforms, &plan.zero_rows)?;
    let safe_logits = finite_logits.where_cond(logits, &plan.zero_logits)?;
    let masked = plan
        .legal_mask
        .where_cond(&safe_logits, &plan.excluded_logits)?;
    let shifted = masked.broadcast_sub(&masked.max_keepdim(1)?)?;
    let log_z = shifted.exp()?.sum_keepdim(1)?.log()?;
    let log_za = shifted
        .affine(plan.alpha as f64, 0.)?
        .exp()?
        .sum_keepdim(1)?
        .log()?;

    // Equivalent to (logZa-alpha*logZ)/(1-alpha), without alpha*logZ
    // overflowing for a very large finite alpha. The shifted maximum is zero,
    // so each sum contains at least one exact exp(0), even at large alpha.
    let correction = (&log_z - &log_za)?.maximum(&plan.zero_rows)?;
    let entropy = (&log_z + correction.affine(plan.inverse_alpha_minus_one, 0.)?)?
        .maximum(&plan.zero_rows)?
        .minimum(&plan.log_legal_count)?;
    let support_f = entropy
        .exp()?
        .affine(1., 0.5)?
        .floor()?
        .maximum(&plan.one_rows)?
        .minimum(&plan.legal_count)?;
    let support_valid = support_f.abs()?.lt(&plan.infinity_rows)?;
    let safe_support = support_valid.where_cond(&support_f, &plan.one_rows)?;
    let valid = finite_rows.mul(&uniform_valid)?.mul(&support_valid)?;

    let last_rank = safe_support.affine(1., -1.)?;
    let rank = safe_uniforms
        .mul(&safe_support)?
        .floor()?
        .maximum(&plan.zero_rows)?
        .minimum(&last_rank)?
        .to_dtype(DType::U32)?;
    let order = masked.contiguous()?.arg_sort_last_dim(false)?;
    let relative_token = order.gather(&rank, 1)?;
    let token = relative_token.add(&plan.token_base)?;
    let tokens = valid.where_cond(&token, &plan.placeholder)?;
    let support = valid.where_cond(&safe_support.to_dtype(DType::U32)?, &plan.one_index_rows)?;
    Ok(DeviceSample {
        tokens,
        valid: valid.to_dtype(DType::U32)?,
        support,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn run(
        rows: Vec<Vec<f32>>,
        legal: &[(usize, usize)],
        uniforms: &[f32],
        alpha: f32,
        offset: u32,
    ) -> Result<DeviceSample> {
        let batch = rows.len();
        let choices = rows[0].len();
        let logits = Tensor::from_vec(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            (batch, choices),
            &Device::Cpu,
        )?;
        let u = Tensor::from_slice(uniforms, (batch, 1), &Device::Cpu)?;
        sample(&logits, legal, &u, alpha, offset)
    }

    #[test]
    fn known_distribution_selects_the_requested_uniform_rank() -> Result<()> {
        let row = vec![0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()];
        let result = run(
            vec![row; 4],
            &[(0, 3); 4],
            &[0., 0.49, 0.5, 0.99999994],
            2.,
            100,
        )?;
        assert_eq!(result.support.to_vec2::<u32>()?, vec![vec![2]; 4]);
        assert_eq!(
            result.tokens.to_vec2::<u32>()?,
            vec![vec![100], vec![100], vec![101], vec![101]]
        );
        assert_eq!(result.valid.to_vec2::<u32>()?, vec![vec![1]; 4]);
        Ok(())
    }

    #[test]
    fn prepared_constants_are_reused_and_results_follow_each_uniform() -> Result<()> {
        let plan = prepare(&[(0, 3)], 3, 2., 100, &Device::Cpu)?;
        let identities = (
            plan.legal_mask.id(),
            plan.token_base.id(),
            plan.log_legal_count.id(),
        );
        let logits = Tensor::from_vec(
            vec![0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()],
            (1, 3),
            &Device::Cpu,
        )?;
        let first = sample_prepared(&logits, &Tensor::full(0f32, (1, 1), &Device::Cpu)?, &plan)?;
        let second = sample_prepared(&logits, &Tensor::full(0.9f32, (1, 1), &Device::Cpu)?, &plan)?;
        assert_eq!(first.tokens.to_vec2::<u32>()?, vec![vec![100]]);
        assert_eq!(second.tokens.to_vec2::<u32>()?, vec![vec![101]]);
        assert_eq!(
            identities,
            (
                plan.legal_mask.id(),
                plan.token_base.id(),
                plan.log_legal_count.id()
            )
        );
        let cloned = plan.clone();
        assert_eq!(cloned.legal_mask.id(), plan.legal_mask.id());
        let wrong = Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?;
        assert!(sample_prepared(
            &wrong,
            &Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
            &plan
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn uniform_rank_grid_covers_all_legal_options_without_illegal_high_logits() -> Result<()> {
        let uniforms = (0..32).map(|i| (i as f32 + 0.5) / 32.).collect::<Vec<_>>();
        let result = run(
            vec![vec![100., 0., 0., 0., 0., 200.]; 32],
            &[(1, 4); 32],
            &uniforms,
            4.,
            20,
        )?;
        let tokens = result.tokens.to_vec2::<u32>()?;
        let mut counts = [0usize; 4];
        for row in tokens {
            assert!((21..25).contains(&row[0]));
            counts[(row[0] - 21) as usize] += 1;
        }
        assert_eq!(counts, [8, 8, 8, 8]);
        assert!(result
            .support
            .to_vec2::<u32>()?
            .iter()
            .all(|r| r == &vec![4]));
        Ok(())
    }

    #[test]
    fn distinct_rows_and_singleton_ranges_never_mix() -> Result<()> {
        let result = run(
            vec![vec![0., 30., 10.], vec![40., 0., 10.], vec![-10., 0., 50.]],
            &[(0, 3), (0, 3), (2, 1)],
            &[0.9, 0.7, 0.3],
            9.,
            200,
        )?;
        assert_eq!(
            result.tokens.to_vec2::<u32>()?,
            vec![vec![201], vec![200], vec![202]]
        );
        assert_eq!(result.support.to_vec2::<u32>()?, vec![vec![1]; 3]);
        Ok(())
    }

    #[test]
    fn nonfinite_logits_and_bad_uniforms_return_valid_zero_legal_placeholders() -> Result<()> {
        let rows = vec![
            vec![f32::NAN, 2., 1.],
            vec![f32::INFINITY, 2., 1.],
            vec![f32::NEG_INFINITY, 2., 1.],
            vec![1., 2., 3.],
            vec![1., 2., 3.],
            vec![1., 2., 3.],
            vec![1., 2., 3.],
        ];
        let result = run(
            rows,
            &[(1, 2); 7],
            &[0., 0.5, 0.999, f32::NAN, 1., -0.1, f32::INFINITY],
            4.,
            90,
        )?;
        assert_eq!(result.valid.to_vec2::<u32>()?, vec![vec![0]; 7]);
        assert_eq!(result.tokens.to_vec2::<u32>()?, vec![vec![91]; 7]);
        assert_eq!(result.support.to_vec2::<u32>()?, vec![vec![1]; 7]);
        Ok(())
    }

    #[test]
    fn high_alpha_near_uniform_uses_log_domain_without_mass_underflow() -> Result<()> {
        let n = 128;
        let row = (0..n)
            .map(|i| -1.5 + 1.5 * i as f32 / (n - 1) as f32)
            .collect::<Vec<_>>();
        let weights = row.iter().map(|&z| (z as f64).exp()).collect::<Vec<_>>();
        let normalizer = weights.iter().sum::<f64>();
        let mass = weights
            .iter()
            .map(|w| (w / normalizer).powf(32.))
            .sum::<f64>();
        let reference = (mass.ln() / (1. - 32.)).exp().round() as i64;
        let result = run(vec![row], &[(0, n)], &[0.25], 32., 0)?;
        let support = result.support.to_vec2::<u32>()?[0][0];
        assert!((support as i64 - reference).abs() <= 1);
        assert!(
            support < 110,
            "naive underflow/clamp would incorrectly select the full 128"
        );
        assert_eq!(
            result.tokens.to_vec2::<u32>()?[0][0],
            (n - 1) as u32 - ((support as f32 * 0.25).floor() as u32)
        );
        let uniform = run(vec![vec![0.; 2048]], &[(0, 2048)], &[0.99999994], 32., 0)?;
        assert_eq!(uniform.support.to_vec2::<u32>()?, vec![vec![2048]]);
        assert_eq!(uniform.valid.to_vec2::<u32>()?, vec![vec![1]]);
        let huge = run(vec![vec![0., -0.01, -0.02]], &[(0, 3)], &[0.5], f32::MAX, 0)?;
        assert_eq!(huge.valid.to_vec2::<u32>()?, vec![vec![1]]);
        Ok(())
    }

    #[test]
    fn malformed_cpu_metadata_is_rejected() -> Result<()> {
        let logits = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        let u = Tensor::zeros((2, 1), DType::F32, &Device::Cpu)?;
        assert!(sample(&logits, &[(0, 3)], &u, 4., 0).is_err());
        assert!(sample(&logits, &[(0, 0), (0, 3)], &u, 4., 0).is_err());
        assert!(sample(&logits, &[(usize::MAX, 2), (0, 3)], &u, 4., 0).is_err());
        assert!(sample(&logits, &[(1, 3), (0, 3)], &u, 4., 0).is_err());
        for alpha in [1., f32::NAN, f32::INFINITY] {
            assert!(sample(&logits, &[(0, 3); 2], &u, alpha, 0).is_err());
        }
        assert!(sample(&logits, &[(0, 3); 2], &u, 4., u32::MAX).is_err());
        let wrong_u = Tensor::zeros((2,), DType::F32, &Device::Cpu)?;
        assert!(sample(&logits, &[(0, 3); 2], &wrong_u, 4., 0).is_err());
        Ok(())
    }
}
