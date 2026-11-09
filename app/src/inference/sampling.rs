//! Backend-independent Rényi effective-support sampling.
//!
//! Preserve the legacy sampler's arithmetic, tie ordering, and RNG stream while
//! finding only the randomly requested rank instead of sorting every option.

use anyhow::{bail, Result};
use rand::{rngs::StdRng, Rng};

pub(super) fn renyi_uniform_index(logits: &[f64], alpha: f64, rng: &mut StdRng) -> Result<usize> {
    if !alpha.is_finite() || alpha <= 1. {
        bail!("Rényi α 必须是大于 1 的有限值");
    }
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        bail!("采样候选为空或包含非有限值");
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_z = logits.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln();
    let log_za = logits
        .iter()
        .map(|v| ((v - maximum) * alpha).exp())
        .sum::<f64>()
        .ln();
    let entropy = ((log_za - alpha * log_z) / (1. - alpha)).clamp(0., (logits.len() as f64).ln());
    let support = (entropy.exp() + 0.5).floor().clamp(1., logits.len() as f64) as usize;

    // The old sampler sorted first, then drew this exact same random rank.
    // No RNG is used by selection, so drawing before selection preserves the
    // complete random stream, including the support==1 case.
    let rank = rng.random_range(0..support);
    let mut order: Vec<_> = (0..logits.len()).collect();
    let (_, selected, _) = order.select_nth_unstable_by(rank, |&a, &b| {
        logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
    });
    Ok(*selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, SeedableRng};

    /// Independent copy of the pre-optimization full-sort implementation.
    fn full_sort_reference(logits: &[f64], alpha: f64, rng: &mut StdRng) -> Result<usize> {
        if !alpha.is_finite() || alpha <= 1. {
            bail!("Rényi α 必须是大于 1 的有限值");
        }
        if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
            bail!("采样候选为空或包含非有限值");
        }
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_z = logits.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln();
        let log_za = logits
            .iter()
            .map(|v| ((v - maximum) * alpha).exp())
            .sum::<f64>()
            .ln();
        let entropy =
            ((log_za - alpha * log_z) / (1. - alpha)).clamp(0., (logits.len() as f64).ln());
        let support = (entropy.exp() + 0.5).floor().clamp(1., logits.len() as f64) as usize;
        let mut order: Vec<_> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        Ok(order[rng.random_range(0..support)])
    }

    fn compare_stream(logits: &[f64], alpha: f64, seed: u64) {
        let mut actual_rng = StdRng::seed_from_u64(seed);
        let mut reference_rng = actual_rng.clone();
        for draw in 0..24 {
            let expected = full_sort_reference(logits, alpha, &mut reference_rng).unwrap();
            let actual = renyi_uniform_index(logits, alpha, &mut actual_rng).unwrap();
            assert_eq!(
                actual,
                expected,
                "different token: n={}, alpha={alpha}, seed={seed}, draw={draw}",
                logits.len()
            );
        }
        // Checking selected options alone can miss an accidental extra draw
        // when the support is one or probabilities tie.
        for word in 0..8 {
            assert_eq!(
                actual_rng.next_u64(),
                reference_rng.next_u64(),
                "different RNG state: n={}, alpha={alpha}, seed={seed}, word={word}",
                logits.len()
            );
        }
    }

    #[test]
    fn rank_selection_matches_legacy_tokens_and_rng_across_real_head_sizes() {
        for n in [1, 2, 128, 270, 1023, 2048] {
            let tied: Vec<_> = (0..n)
                .map(|i| ((i * 37 + 11) % 19) as f64 * 0.125)
                .collect();
            let smooth: Vec<_> = (0..n)
                .map(|i| (((i * 7919 + 104729) % 1009) as f64 - 504.) / 89.)
                .collect();
            let uniform = vec![3.25; n];
            let mut concentrated = vec![-1000.; n];
            concentrated[n / 2] = 1000.;
            for alpha in [1.0001, 2., 3.5, 4., 7., 32.] {
                for seed in [0, 7, 20261011, u64::MAX] {
                    for logits in [&tied, &smooth, &uniform, &concentrated] {
                        compare_stream(logits, alpha, seed);
                    }
                }
            }
        }
    }

    #[test]
    fn signed_zero_and_duplicate_scores_keep_the_legacy_total_order() {
        for logits in [
            vec![0., -0., 0., -0., 0., -0., 0.],
            vec![2., 2., -0., 0., 1., 1., -2., -2.],
            vec![-1e300, 1e300, 1e300, -1e300],
        ] {
            for alpha in [1.0001, 3.5, 4., 7., 32.] {
                for seed in 0..16 {
                    compare_stream(&logits, alpha, seed);
                }
            }
        }
    }

    #[test]
    fn rejects_the_same_invalid_inputs_without_consuming_randomness() {
        let invalid = [
            (vec![], 4.),
            (vec![f64::NAN], 4.),
            (vec![0., f64::INFINITY], 4.),
            (vec![f64::NEG_INFINITY, 0.], 4.),
            (vec![0., 1.], 1.),
            (vec![0., 1.], 0.),
            (vec![0., 1.], -2.),
            (vec![0., 1.], f64::NAN),
            (vec![0., 1.], f64::INFINITY),
        ];
        for (logits, alpha) in invalid {
            let mut actual_rng = StdRng::seed_from_u64(123);
            let mut reference_rng = actual_rng.clone();
            let before = actual_rng.clone().next_u64();
            let actual = renyi_uniform_index(&logits, alpha, &mut actual_rng).unwrap_err();
            let expected = full_sort_reference(&logits, alpha, &mut reference_rng).unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
            assert_eq!(actual_rng.next_u64(), before);
            assert_eq!(reference_rng.next_u64(), before);
        }
    }
}
