//! Immutable, backend-neutral scatter geometry shared across transformer layers.
//!
//! The indices describe writes, not values or a KV snapshot. Callers must not
//! mutate the tensor returned by `indices`; Candle tensors have interior mutable
//! storage even behind a shared reference. Clone shares the complete plan.

use super::MAX_CONTEXT;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;

#[derive(Debug)]
struct WritePlanInner {
    offsets: Vec<usize>,
    heads: usize,
    width: usize,
    device: Device,
    indices: Tensor,
}

#[derive(Clone, Debug)]
pub(super) struct BatchWritePlan {
    inner: Arc<WritePlanInner>,
}

/// Returns physical-head rows and scalar index count without unchecked products.
fn checked_shape(past_lens: &[usize], heads: usize, width: usize) -> Result<(usize, usize)> {
    if past_lens.is_empty() || heads == 0 || width == 0 {
        bail!("write plan requires nonempty rows, heads and width");
    }
    if past_lens.iter().any(|&offset| offset >= MAX_CONTEXT) {
        bail!("write plan offsets must leave a slot inside the maximum context");
    }
    let rows = past_lens
        .len()
        .checked_mul(heads)
        .context("write plan row count overflow")?;
    let elements = rows
        .checked_mul(width)
        .context("write plan index count overflow")?;
    elements
        .checked_mul(std::mem::size_of::<u32>())
        .context("write plan index byte count overflow")?;
    Ok((rows, elements))
}

impl BatchWritePlan {
    pub fn new(past_lens: &[usize], heads: usize, width: usize, device: &Device) -> Result<Self> {
        let (rows, elements) = checked_shape(past_lens, heads, width)?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(elements)
            .context("write plan index allocation failed")?;
        let elements_per_row = elements / past_lens.len();
        for &offset in past_lens {
            indices.extend(std::iter::repeat(offset as u32).take(elements_per_row));
        }
        let indices = Tensor::from_vec(indices, (rows, 1, width), device)?;
        Ok(Self {
            inner: Arc::new(WritePlanInner {
                offsets: past_lens.to_vec(),
                heads,
                width,
                device: device.clone(),
                indices,
            }),
        })
    }

    /// Checks only CPU geometry and tensor metadata; never reads device values.
    pub fn validate(
        &self,
        past_lens: &[usize],
        heads: usize,
        width: usize,
        device: &Device,
    ) -> Result<()> {
        let (rows, _) = checked_shape(past_lens, heads, width)?;
        let inner = &self.inner;
        if inner.offsets.as_slice() != past_lens || inner.heads != heads || inner.width != width {
            bail!("write plan key does not match the current row offsets and head shape");
        }
        if !inner.device.same_device(device) || !inner.indices.device().same_device(device) {
            bail!("write plan device differs from the current KV device");
        }
        if inner.indices.dtype() != DType::U32
            || inner.indices.dims() != [rows, 1, width]
            || !inner.indices.is_contiguous()
        {
            bail!("write plan indices must be contiguous U32 [B*H,1,D]");
        }
        Ok(())
    }

    /// Read-only geometry. The caller must not use in-place tensor operations.
    pub fn indices(&self) -> &Tensor {
        &self.inner.indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_has_exact_row_head_width_order_and_clone_shares_storage() {
        let plan = BatchWritePlan::new(&[1, 7, 1023], 2, 3, &Device::Cpu).unwrap();
        plan.validate(&[1, 7, 1023], 2, 3, &Device::Cpu).unwrap();
        assert_eq!(plan.indices().dims(), [6, 1, 3]);
        assert_eq!(plan.indices().dtype(), DType::U32);
        let expected = [vec![1; 6], vec![7; 6], vec![1023; 6]].concat();
        assert_eq!(
            plan.indices()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            expected
        );
        let cloned = plan.clone();
        assert_eq!(plan.indices().id(), cloned.indices().id());
        assert!(Arc::ptr_eq(&plan.inner, &cloned.inner));
    }

    #[test]
    fn invalid_or_changed_geometry_is_rejected_without_readback() {
        assert!(BatchWritePlan::new(&[], 1, 1, &Device::Cpu).is_err());
        assert!(BatchWritePlan::new(&[0], 0, 1, &Device::Cpu).is_err());
        assert!(BatchWritePlan::new(&[0], 1, 0, &Device::Cpu).is_err());
        assert!(BatchWritePlan::new(&[MAX_CONTEXT], 1, 1, &Device::Cpu).is_err());
        assert!(BatchWritePlan::new(&[0, 1], usize::MAX, 2, &Device::Cpu).is_err());
        assert!(BatchWritePlan::new(&[0], 1, usize::MAX, &Device::Cpu).is_err());
        let plan = BatchWritePlan::new(&[1, 3], 2, 3, &Device::Cpu).unwrap();
        for offsets in [&[3, 1][..], &[1, 4][..], &[1][..]] {
            assert!(plan.validate(offsets, 2, 3, &Device::Cpu).is_err());
        }
        assert!(plan.validate(&[1, 3], 1, 3, &Device::Cpu).is_err());
        assert!(plan.validate(&[1, 3], 2, 4, &Device::Cpu).is_err());
    }

    #[test]
    fn malformed_internal_tensor_metadata_is_rejected() {
        let make = |indices: Tensor| BatchWritePlan {
            inner: Arc::new(WritePlanInner {
                offsets: vec![1, 3],
                heads: 2,
                width: 3,
                device: Device::Cpu,
                indices,
            }),
        };
        let wrong_dtype = make(Tensor::zeros((4, 1, 3), DType::F32, &Device::Cpu).unwrap());
        assert!(wrong_dtype.validate(&[1, 3], 2, 3, &Device::Cpu).is_err());
        let wrong_shape = make(Tensor::zeros((2, 1, 3), DType::U32, &Device::Cpu).unwrap());
        assert!(wrong_shape.validate(&[1, 3], 2, 3, &Device::Cpu).is_err());
        let strided = Tensor::zeros((4, 1, 6), DType::U32, &Device::Cpu)
            .unwrap()
            .narrow(2, 0, 3)
            .unwrap();
        assert!(!strided.is_contiguous());
        assert!(make(strided).validate(&[1, 3], 2, 3, &Device::Cpu).is_err());
    }
}
