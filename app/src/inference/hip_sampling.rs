//! HIP-only FP32 logit CFG + Renyi effective-support uniform sampler.
//! Inputs and outputs remain on the selected ROCm device. Host validation only
//! reads tensor metadata; cpu_fwd deliberately refuses mathematical fallback.

use anyhow::{bail, Context, Result};
use candle_core::{CpuStorage, CustomOp3, DType, Layout, Shape, Tensor};

#[derive(Clone, Copy)]
struct HipSampler {
    batch: usize,
    choices: usize,
    alpha: f32,
    cfg: f32,
    guidance: usize,
    offset: u32,
}

fn validate(
    raw: &Tensor,
    uniforms: &Tensor,
    bounds: &Tensor,
    alpha: f32,
    cfg: f32,
    guidance: usize,
    offset: u32,
) -> Result<HipSampler> {
    let (physical, choices) = raw.dims2()?;
    if !matches!(guidance, 1 | 2)
        || physical == 0
        || physical % guidance != 0
        || !(1..=2048).contains(&choices)
    {
        bail!("HIP sampler needs 1/2 guidance rows and a nonempty head of at most 2048 choices");
    }
    let batch = physical / guidance;
    if batch > i32::MAX as usize || uniforms.dims2()? != (batch, 1) || bounds.dims2()? != (batch, 2)
    {
        bail!("HIP sampler logical row, uniform or bounds shape mismatch");
    }
    if raw.dtype() != DType::F32 || uniforms.dtype() != DType::F32 || bounds.dtype() != DType::U32 {
        bail!("HIP sampler expects FP32 logits/uniforms and U32 bounds");
    }
    if !raw.device().same_device(uniforms.device()) || !raw.device().same_device(bounds.device()) {
        bail!("HIP sampler inputs must share the same device");
    }
    if !raw.is_contiguous() || !uniforms.is_contiguous() || !bounds.is_contiguous() {
        bail!("HIP sampler inputs must be contiguous; nonzero view offsets are supported");
    }
    if !alpha.is_finite()
        || alpha <= 1.
        || !cfg.is_finite()
        || cfg < 0.
        || (guidance == 1 && cfg != 1.)
    {
        bail!("invalid HIP sampler alpha/CFG/guidance combination");
    }
    offset
        .checked_add((choices - 1) as u32)
        .context("HIP sampler token range overflow")?;
    batch
        .checked_mul(3)
        .context("HIP sampler output overflow")?;
    Ok(HipSampler {
        batch,
        choices,
        alpha,
        cfg,
        guidance,
        offset,
    })
}

/// `raw` is [logical_B*guidance, choices], with interleaved (conditional,
/// unconditional) rows for guidance=2. `uniforms` is [logical_B,1] and `bounds`
/// is [logical_B,2] containing relative (start,count) for each legal range.
/// Returns absolute tokens, valid flags and supports, each U32 [logical_B,1].
/// Invalid device values produce valid=0 and a deterministic placeholder;
/// callers must reject the incomplete primitive at its final readback.
pub(super) fn sample(
    raw: &Tensor,
    uniforms: &Tensor,
    bounds: &Tensor,
    alpha: f32,
    cfg: f32,
    guidance: usize,
    offset: u32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let op = validate(raw, uniforms, bounds, alpha, cfg, guidance, offset)?;
    #[cfg(feature = "rocm")]
    let supported = raw.device().is_rocm();
    #[cfg(not(feature = "rocm"))]
    let supported = false;
    if !supported {
        bail!("HIP fused Renyi sampler requires ROCm device storage; no CPU/CUDA fallback");
    }
    let packed = raw.apply_op3_no_bwd(uniforms, bounds, &op)?;
    Ok((
        packed.narrow(0, 0, 1)?.reshape((op.batch, 1))?,
        packed.narrow(0, 1, 1)?.reshape((op.batch, 1))?,
        packed.narrow(0, 2, 1)?.reshape((op.batch, 1))?,
    ))
}

impl CustomOp3 for HipSampler {
    fn name(&self) -> &'static str {
        "primitive-painter-hip-fp32-renyi-v2"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("HIP fused Renyi sampler has no CPU mathematical fallback")
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        raw: &candle_core::RocmStorage,
        raw_layout: &Layout,
        uniforms: &candle_core::RocmStorage,
        uniform_layout: &Layout,
        bounds: &candle_core::RocmStorage,
        bounds_layout: &Layout,
    ) -> candle_core::Result<(candle_core::RocmStorage, Shape)> {
        use candle_core::backend::BackendDevice;
        use std::ffi::c_void;
        let device = raw.rocm_device();
        device.bind_to_thread()?;
        let module = crate::model::hip_kernel::module(
            device,
            "primitive_painter_hip_renyi_fp32_v2",
            include_str!("renyi_sample.hip"),
        )?;
        let (raw_start, raw_end) = raw_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous HIP raw logits".into()))?;
        let (uniform_start, uniform_end) = uniform_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous HIP uniforms".into()))?;
        let (bounds_start, bounds_end) = bounds_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous HIP bounds".into()))?;
        if raw_end.checked_mul(4).is_none_or(|n| n > raw.byte_len())
            || uniform_end
                .checked_mul(4)
                .is_none_or(|n| n > uniforms.byte_len())
            || bounds_end
                .checked_mul(4)
                .is_none_or(|n| n > bounds.byte_len())
        {
            candle_core::bail!("HIP sampler layout extends outside allocation");
        }
        let shape: Shape = (3usize, self.batch).into();
        // The kernel initializes all three cells on every valid/invalid path.
        let mut output = unsafe { device.alloc_uninit(&shape, DType::U32) }?;
        let mut raw_pointer = unsafe { raw.device_ptr().add(raw_start * 4) } as *const f32;
        let mut uniform_pointer =
            unsafe { uniforms.device_ptr().add(uniform_start * 4) } as *const f32;
        let mut bounds_pointer = unsafe { bounds.device_ptr().add(bounds_start * 4) } as *const u32;
        let mut output_pointer = output.device_ptr_mut() as *mut u32;
        let mut batch = self.batch as u32;
        let mut choices = self.choices as u32;
        let mut guidance = self.guidance as u32;
        let mut alpha = self.alpha;
        let mut cfg = self.cfg;
        let mut offset = self.offset;
        let mut padded = self.choices.next_power_of_two() as u32;
        let mut args: [*mut c_void; 11] = [
            (&mut raw_pointer as *mut *const f32).cast(),
            (&mut uniform_pointer as *mut *const f32).cast(),
            (&mut bounds_pointer as *mut *const u32).cast(),
            (&mut output_pointer as *mut *mut u32).cast(),
            (&mut batch as *mut u32).cast(),
            (&mut choices as *mut u32).cast(),
            (&mut guidance as *mut u32).cast(),
            (&mut alpha as *mut f32).cast(),
            (&mut cfg as *mut f32).cast(),
            (&mut offset as *mut u32).cast(),
            (&mut padded as *mut u32).cast(),
        ];
        crate::model::hip_kernel::launch(
            device,
            &module,
            "pp_hip_renyi_sample",
            (batch, 1, 1),
            (256, 1, 1),
            padded * 8 + 256 * 16,
            &mut args,
        )?;
        Ok((output, shape))
    }
}

#[cfg(test)]
#[path = "hip_sampling_reference.rs"]
mod reference;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    fn inputs(rows: usize, choices: usize) -> (Tensor, Tensor, Tensor) {
        (
            Tensor::zeros((rows, choices), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros((rows, 1), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros((rows, 2), DType::U32, &Device::Cpu).unwrap(),
        )
    }
    #[test]
    fn validates_metadata_and_refuses_host_mathematics() -> Result<()> {
        let (raw, u, b) = inputs(2, 128);
        assert_eq!(validate(&raw, &u, &b, 3.5, 1., 1, 100)?.batch, 2);
        assert!(sample(&raw, &u, &b, 4., 1., 1, 0).is_err());
        for alpha in [1., 0., f32::NAN, f32::INFINITY] {
            assert!(validate(&raw, &u, &b, alpha, 1., 1, 0).is_err());
        }
        for cfg in [-1., f32::NAN, f32::INFINITY] {
            assert!(validate(&raw, &u, &b, 4., cfg, 1, 0).is_err());
        }
        assert!(validate(&raw, &u, &b, 4., 2., 1, 0).is_err());
        assert!(validate(&raw, &u, &b, 4., 1., 3, 0).is_err());
        assert!(validate(&raw, &u, &b, 4., 1., 1, u32::MAX).is_err());
        Ok(())
    }
    #[test]
    fn supports_cfg_pairs_offsets_and_rejects_wrong_shapes() -> Result<()> {
        let raw = Tensor::zeros((8, 270), DType::F32, &Device::Cpu)?.narrow(0, 2, 4)?;
        let u = Tensor::zeros((6, 1), DType::F32, &Device::Cpu)?.narrow(0, 3, 2)?;
        let b = Tensor::zeros((6, 2), DType::U32, &Device::Cpu)?.narrow(0, 2, 2)?;
        assert!(u.layout().start_offset() > 0);
        assert_eq!(validate(&raw, &u, &b, 3.5, 2.5, 2, 2048)?.batch, 2);
        let strided = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?.narrow(1, 0, 1)?;
        assert!(validate(&raw, &strided, &b, 4., 2., 2, 0).is_err());
        let too_wide = Tensor::zeros((4, 2049), DType::F32, &Device::Cpu)?;
        assert!(validate(&too_wide, &u, &b, 4., 2., 2, 0).is_err());
        let wrong_type = raw.to_dtype(DType::BF16)?;
        assert!(validate(&wrong_type, &u, &b, 4., 2., 2, 0).is_err());
        Ok(())
    }
}
