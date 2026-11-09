//! Explicit host-only validation oracle. Not imported by the production sampler.
//! Reproduces the HIP kernel's FP32 tree reduction and deterministic rank law;
//! libm rounding can still differ near an effective-support rounding boundary.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceSample {
    pub token: u32,
    pub valid: u32,
    pub support: u32,
}

pub fn reference(
    raw: &[f32],
    physical_rows: usize,
    choices: usize,
    uniforms: &[f32],
    bounds: &[[u32; 2]],
    alpha: f32,
    cfg: f32,
    guidance: usize,
    offset: u32,
) -> Result<Vec<ReferenceSample>, String> {
    if !matches!(guidance, 1 | 2)
        || physical_rows == 0
        || physical_rows % guidance != 0
        || !(1..=2048).contains(&choices)
        || !alpha.is_finite()
        || alpha <= 1.
        || !cfg.is_finite()
        || cfg < 0.
        || (guidance == 1 && cfg != 1.)
        || offset.checked_add((choices - 1) as u32).is_none()
    {
        return Err("invalid reference metadata".into());
    }
    let batch = physical_rows / guidance;
    if raw.len()
        != physical_rows
            .checked_mul(choices)
            .ok_or("reference shape overflow")?
        || uniforms.len() != batch
        || bounds.len() != batch
    {
        return Err("reference input lengths differ".into());
    }
    let mut output = Vec::with_capacity(batch);
    for row in 0..batch {
        let [start, count] = bounds[row];
        let fallback = offset + if start < choices as u32 { start } else { 0 };
        let bad = ReferenceSample {
            token: fallback,
            valid: 0,
            support: 1,
        };
        let uniform = uniforms[row];
        if !uniform.is_finite()
            || !(0. ..1.).contains(&uniform)
            || count == 0
            || start >= choices as u32
            || count > choices as u32 - start
        {
            output.push(bad);
            continue;
        }
        let conditional = row * guidance * choices;
        let mut logits = Vec::with_capacity(choices);
        let mut finite = true;
        for col in 0..choices {
            let c = raw[conditional + col];
            let value = if guidance == 2 {
                let u = raw[conditional + choices + col];
                finite &= u.is_finite();
                u + cfg * (c - u)
            } else {
                c
            };
            finite &= c.is_finite() && value.is_finite();
            logits.push(value);
        }
        if !finite {
            output.push(bad);
            continue;
        }
        let begin = start as usize;
        let end = begin + count as usize;
        let maximum = logits[begin..end]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sums = [0f32; 256];
        let mut sums_alpha = [0f32; 256];
        for thread in 0..256 {
            for compact in (thread..count as usize).step_by(256) {
                let shifted = logits[begin + compact] - maximum;
                sums[thread] += shifted.exp();
                sums_alpha[thread] += (alpha * shifted).exp();
            }
        }
        let mut step = 128;
        while step > 0 {
            for thread in 0..step {
                sums[thread] += sums[thread + step];
                sums_alpha[thread] += sums_alpha[thread + step];
            }
            step >>= 1;
        }
        let log_z = sums[0].ln();
        let log_za = sums_alpha[0].ln();
        let entropy = log_z + (log_z - log_za).max(0.) / (alpha - 1.);
        if !log_z.is_finite() || !log_za.is_finite() || !entropy.is_finite() {
            output.push(bad);
            continue;
        }
        let entropy = entropy.clamp(0., (count as f32).ln());
        let support = ((entropy.exp() + 0.5) as u32).clamp(1, count);
        let rank = ((uniform * support as f32) as u32).min(support - 1) as usize;
        let mut order = (begin..end).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            if logits[a] > logits[b] {
                std::cmp::Ordering::Less
            } else if logits[a] < logits[b] {
                std::cmp::Ordering::Greater
            } else {
                a.cmp(&b)
            }
        });
        output.push(ReferenceSample {
            token: offset + order[rank] as u32,
            valid: 1,
            support,
        });
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uniform_support_and_index_ties_use_fixed_uniform_rank() {
        for count in [1usize, 2, 127, 128, 255, 256, 270, 1024, 2048] {
            for alpha in [1.25, 3.5, 4., 7., 32.] {
                let raw = vec![0.; count * 4];
                let u = [0., 0.25, 0.75, 0.99999994];
                let bounds = vec![[0, count as u32]; 4];
                let result = reference(&raw, 4, count, &u, &bounds, alpha, 1., 1, 100).unwrap();
                for row in 0..4 {
                    assert_eq!(
                        result[row],
                        ReferenceSample {
                            token: 100 + ((u[row] * count as f32) as u32).min(count as u32 - 1),
                            valid: 1,
                            support: count as u32
                        }
                    );
                }
            }
        }
    }
    #[test]
    fn guided_rows_and_legal_ranges_do_not_mix_candidates() {
        // c0/u0/c1/u1, with independently restricted legal ranges.
        let raw = [
            0., 1., 2., 3., 1., 1., 1., 1., 9., 8., 7., 6., 1., 1., 1., 1.,
        ];
        let result = reference(&raw, 4, 4, &[0., 0.], &[[1, 2], [2, 2]], 4., 2.5, 2, 2048).unwrap();
        assert_eq!(result[0].token, 2050);
        assert_eq!(result[1].token, 2050);
        assert_eq!(result[0].valid, 1);
        assert_eq!(result[1].valid, 1);
    }
    #[test]
    fn all_raw_logits_are_checked_even_outside_legal_candidates() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let result = reference(&[1., 2., value], 1, 3, &[0.], &[[0, 2]], 4., 1., 1, 0).unwrap();
            assert_eq!(
                result[0],
                ReferenceSample {
                    token: 0,
                    valid: 0,
                    support: 1
                }
            );
        }
        for u in [1., -0.1, f32::NAN, f32::INFINITY] {
            assert_eq!(
                reference(&[1., 2.], 1, 2, &[u], &[[0, 2]], 4., 1., 1, 0).unwrap()[0].valid,
                0
            );
        }
        assert_eq!(
            reference(&[f32::MAX, -f32::MAX], 2, 1, &[0.], &[[0, 1]], 4., 2., 2, 0).unwrap()[0]
                .valid,
            0
        );
        assert_eq!(
            reference(&[0., 0.], 1, 2, &[0.], &[[2, 1]], 4., 1., 1, 0).unwrap()[0],
            ReferenceSample {
                token: 0,
                valid: 0,
                support: 1
            }
        );
    }
}
