//! GPU-resident output heads. Host materialization is deliberately explicit.
//! Row views and concatenated tensors are read-only prediction values.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};

#[derive(Clone)]
pub struct DevicePrediction {
    /// [physical_rows, projected_attribute_count], always F32.
    pub attributes: Tensor,
    /// Raw conditional/unconditional termination logits, not a guided decision.
    pub termination: Option<Tensor>,
    pub attribute_offset: usize,
}

impl DevicePrediction {
    fn validate(&self) -> Result<(usize, usize)> {
        let (rows, columns) = self.attributes.dims2()?;
        if rows == 0 || columns == 0 || self.attributes.dtype() != DType::F32 {
            bail!("device prediction requires a nonempty F32 attribute matrix");
        }
        let end = self
            .attribute_offset
            .checked_add(columns)
            .context("projected attribute range overflow")?;
        if end > super::ATTRIBUTE_VOCAB {
            bail!("projected attributes extend outside the model attribute vocabulary");
        }
        if let Some(termination) = &self.termination {
            if termination.dims2()? != (rows, 2)
                || termination.dtype() != DType::F32
                || !termination.device().same_device(self.attributes.device())
            {
                bail!("termination head must be F32 [physical_rows,2] on the attribute device");
            }
        }
        Ok((rows, columns))
    }

    pub fn rows(&self) -> Result<usize> {
        Ok(self.validate()?.0)
    }

    pub fn narrow_rows(&self, start: usize, len: usize) -> Result<Self> {
        let (rows, _) = self.validate()?;
        if len == 0 || start.checked_add(len).is_none_or(|end| end > rows) {
            bail!("device prediction row range is empty or out of bounds");
        }
        Ok(Self {
            attributes: self.attributes.narrow(0, start, len)?,
            termination: self
                .termination
                .as_ref()
                .map(|tensor| tensor.narrow(0, start, len))
                .transpose()?,
            attribute_offset: self.attribute_offset,
        })
    }

    pub fn concat_rows(parts: &[&Self]) -> Result<Self> {
        let first = *parts
            .first()
            .context("cannot concatenate zero predictions")?;
        let (_, columns) = first.validate()?;
        let mut rows = 0usize;
        for part in parts {
            let (count, width) = part.validate()?;
            if width != columns
                || part.attribute_offset != first.attribute_offset
                || part.termination.is_some() != first.termination.is_some()
                || !part
                    .attributes
                    .device()
                    .same_device(first.attributes.device())
            {
                bail!("concatenated prediction rows have different heads or devices");
            }
            rows = rows
                .checked_add(count)
                .context("prediction row count overflow")?;
        }
        if parts.len() == 1 {
            return Ok((*first).clone());
        }
        let attributes: Vec<_> = parts.iter().map(|part| &part.attributes).collect();
        let termination = if first.termination.is_some() {
            let heads: Vec<_> = parts
                .iter()
                .map(|part| part.termination.as_ref().unwrap())
                .collect();
            Some(Tensor::cat(&heads, 0)?)
        } else {
            None
        };
        let joined = Self {
            attributes: Tensor::cat(&attributes, 0)?,
            termination,
            attribute_offset: first.attribute_offset,
        };
        if joined.rows()? != rows {
            bail!("concatenated prediction row count changed");
        }
        Ok(joined)
    }

    /// One host transfer for the projected attribute head and optional raw
    /// termination head. Intended for reference/probe callers, not GPU sampling.
    pub fn to_host(&self) -> Result<Vec<super::Prediction>> {
        let (_, columns) = self.validate()?;
        let packed = if let Some(termination) = &self.termination {
            Tensor::cat(&[&self.attributes, termination], 1)?
        } else {
            self.attributes.clone()
        };
        packed
            .to_vec2::<f32>()?
            .into_iter()
            .map(|mut row| {
                if row.iter().any(|value| !value.is_finite()) {
                    bail!("模型输出出现非有限值，已停止且不提交当前图元");
                }
                let termination = if self.termination.is_some() {
                    [row[columns], row[columns + 1]]
                } else {
                    [0., 0.]
                };
                row.truncate(columns);
                Ok(super::Prediction {
                    attributes: row,
                    termination,
                    attribute_offset: self.attribute_offset,
                })
            })
            .collect()
    }

    /// A GPU U32 flag per physical row. Neither this operation nor finite_rows
    /// reads a scalar or probability distribution back to the CPU.
    pub fn valid_rows(&self) -> Result<Tensor> {
        self.validate()?;
        let attributes = finite_rows(&self.attributes)?;
        if let Some(termination) = &self.termination {
            Ok((&attributes * &finite_rows(termination)?)?)
        } else {
            Ok(attributes)
        }
    }
}

/// Tests a nonempty floating-point matrix before legality masking. Infinity
/// cannot be used here as an allowed masked value: this checks raw model/CFG
/// outputs. U32 reduction avoids the overflow of summing >255 U8 flags.
pub fn finite_rows(values: &Tensor) -> Result<Tensor> {
    let (rows, columns) = values.dims2()?;
    if rows == 0 || columns == 0 || columns > u32::MAX as usize {
        bail!("finite-row check requires a nonempty representable matrix");
    }
    if !matches!(
        values.dtype(),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64
    ) {
        bail!("finite-row check requires floating-point values");
    }
    Ok(values
        .abs()?
        .lt(f64::INFINITY)?
        .to_dtype(DType::U32)?
        .sum_keepdim(1)?
        .eq(columns as u32)?
        .to_dtype(DType::U32)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn prediction(termination: bool) -> DevicePrediction {
        DevicePrediction {
            attributes: Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (3, 2), &Device::Cpu)
                .unwrap(),
            termination: termination.then(|| {
                Tensor::from_vec(vec![7f32, 8., 9., 10., 11., 12.], (3, 2), &Device::Cpu).unwrap()
            }),
            attribute_offset: 1024,
        }
    }

    #[test]
    fn finite_flags_detect_nan_and_both_infinities_without_u8_overflow() {
        let values = Tensor::from_vec(
            vec![
                0f32,
                -0.,
                f32::MAX,
                f32::NAN,
                0.,
                1.,
                f32::INFINITY,
                2.,
                3.,
                f32::NEG_INFINITY,
                -1.,
                0.,
                f32::MIN_POSITIVE,
                f32::MIN,
                42.,
            ],
            (5, 3),
            &Device::Cpu,
        )
        .unwrap();
        let flags = finite_rows(&values).unwrap();
        assert_eq!(flags.dtype(), DType::U32);
        assert_eq!(flags.dims(), &[5, 1]);
        assert_eq!(
            flags.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            vec![1, 0, 0, 0, 1]
        );
        let mut wide = vec![1f32; 600];
        wide[599] = f32::NAN;
        assert_eq!(
            finite_rows(&Tensor::from_vec(wide, (2, 300), &Device::Cpu).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            vec![1, 0]
        );
        let f64_values =
            Tensor::from_vec(vec![f64::MAX, f64::INFINITY], (2, 1), &Device::Cpu).unwrap();
        assert_eq!(
            finite_rows(&f64_values)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            vec![1, 0]
        );
    }

    #[test]
    fn row_views_and_concatenation_preserve_heads_order_offsets_and_host_values() {
        let prediction = prediction(true);
        let last = prediction.narrow_rows(2, 1).unwrap();
        let first_two = prediction.narrow_rows(0, 2).unwrap();
        let joined = DevicePrediction::concat_rows(&[&last, &first_two]).unwrap();
        assert_eq!(joined.rows().unwrap(), 3);
        assert_eq!(joined.attribute_offset, 1024);
        let host = joined.to_host().unwrap();
        assert_eq!(host[0].attributes, vec![5., 6.]);
        assert_eq!(host[0].termination, [11., 12.]);
        assert_eq!(host[0].attribute(1024).unwrap(), 5.);
        assert_eq!(host[1].attributes, vec![1., 2.]);
        assert_eq!(host[2].attributes, vec![3., 4.]);
        assert_eq!(
            joined
                .valid_rows()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            vec![1, 1, 1]
        );
        let one = DevicePrediction::concat_rows(&[&last]).unwrap();
        assert_eq!(one.attributes.id(), last.attributes.id());
    }

    #[test]
    fn absent_term_and_nonfinite_term_are_handled_explicitly() {
        let no_term = prediction(false);
        let host = no_term.to_host().unwrap();
        assert_eq!(host[0].termination, [0., 0.]);
        assert!(no_term.narrow_rows(1, 1).unwrap().termination.is_none());
        let joined = DevicePrediction::concat_rows(&[&no_term, &no_term]).unwrap();
        assert!(joined.termination.is_none());
        assert_eq!(joined.rows().unwrap(), 6);
        let mut bad = prediction(true);
        bad.termination = Some(
            Tensor::from_vec(
                vec![0f32, 1., f32::NAN, 1., 0., f32::INFINITY],
                (3, 2),
                &Device::Cpu,
            )
            .unwrap(),
        );
        assert_eq!(
            bad.valid_rows()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            vec![1, 0, 0]
        );
        assert!(bad.to_host().is_err());
    }

    #[test]
    fn incompatible_geometry_offsets_and_head_presence_fail_before_concatenation() {
        let prediction = prediction(true);
        assert!(prediction.narrow_rows(0, 0).is_err());
        assert!(prediction.narrow_rows(2, 2).is_err());
        assert!(prediction.narrow_rows(usize::MAX, 2).is_err());
        assert!(DevicePrediction::concat_rows(&[]).is_err());
        let mut wrong = prediction.clone();
        wrong.attribute_offset += 1;
        assert!(DevicePrediction::concat_rows(&[&prediction, &wrong]).is_err());
        wrong.attribute_offset = usize::MAX;
        assert!(wrong.rows().is_err());
        wrong.attribute_offset = super::super::ATTRIBUTE_VOCAB;
        assert!(wrong.to_host().is_err());
        wrong = prediction.clone();
        wrong.termination = None;
        assert!(DevicePrediction::concat_rows(&[&prediction, &wrong]).is_err());
        wrong = prediction.clone();
        wrong.termination = Some(Tensor::zeros((3, 1), DType::F32, &Device::Cpu).unwrap());
        assert!(wrong.valid_rows().is_err());
        wrong = prediction.clone();
        wrong.attributes = wrong.attributes.to_dtype(DType::F64).unwrap();
        assert!(wrong.rows().is_err());
        assert!(finite_rows(&Tensor::zeros((1, 2), DType::U32, &Device::Cpu).unwrap()).is_err());
    }
}
