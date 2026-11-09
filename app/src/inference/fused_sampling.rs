//! NVIDIA sampler: one block per logical row. The tensor sampler remains the
//! portable reference; this module never claims support for another backend.

use anyhow::{bail, Context, Result};
use candle_core::{CpuStorage, CustomOp3, DType, Layout, Shape, Tensor};

#[derive(Clone, Copy)]
struct FusedSampler {
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
) -> Result<FusedSampler> {
    let (physical, choices) = raw.dims2()?;
    if !matches!(guidance, 1 | 2)
        || physical == 0
        || physical % guidance != 0
        || !(1..=2048).contains(&choices)
    {
        bail!("fused sampler needs 1/2 guidance rows and a nonempty head of at most 2048 choices");
    }
    let batch = physical / guidance;
    if batch > i32::MAX as usize || uniforms.dims2()? != (batch, 1) || bounds.dims2()? != (batch, 2)
    {
        bail!("fused sampler logical row, uniform or bounds shape mismatch");
    }
    if raw.dtype() != DType::F32 || uniforms.dtype() != DType::F32 || bounds.dtype() != DType::U32 {
        bail!("fused sampler expects FP32 logits/uniforms and U32 bounds");
    }
    if !raw.device().same_device(uniforms.device()) || !raw.device().same_device(bounds.device()) {
        bail!("fused sampler inputs must share the same device");
    }
    if !raw.is_contiguous() || !uniforms.is_contiguous() || !bounds.is_contiguous() {
        bail!("fused sampler inputs must be contiguous; nonzero view offsets are supported");
    }
    if !alpha.is_finite()
        || alpha <= 1.
        || !cfg.is_finite()
        || cfg < 0.
        || (guidance == 1 && cfg != 1.)
    {
        bail!("invalid fused sampler alpha/CFG/guidance combination");
    }
    offset
        .checked_add((choices - 1) as u32)
        .context("fused token range overflow")?;
    batch
        .checked_mul(3)
        .context("fused output shape overflow")?;
    Ok(FusedSampler {
        batch,
        choices,
        alpha,
        cfg,
        guidance,
        offset,
    })
}

pub(super) fn sample(
    raw: &Tensor,
    uniforms: &Tensor,
    bounds: &Tensor,
    alpha: f32,
    cfg: f32,
    guidance: usize,
    offset: u32,
) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "rocm")]
    if raw.device().is_rocm() {
        return super::hip_sampling::sample(raw, uniforms, bounds, alpha, cfg, guidance, offset);
    }
    let op = validate(raw, uniforms, bounds, alpha, cfg, guidance, offset)?;
    if !raw.device().is_cuda() {
        bail!("fused Rényi sampler is NVIDIA CUDA-only; select the tensor reference on other backends");
    }
    let packed = raw.apply_op3_no_bwd(uniforms, bounds, &op)?;
    Ok((
        packed.narrow(0, 0, 1)?.reshape((op.batch, 1))?,
        packed.narrow(0, 1, 1)?.reshape((op.batch, 1))?,
        packed.narrow(0, 2, 1)?.reshape((op.batch, 1))?,
    ))
}

impl CustomOp3 for FusedSampler {
    fn name(&self) -> &'static str {
        "primitive-painter-fused-renyi-v1"
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
        candle_core::bail!(
            "fused Rényi sampler has no CPU implementation; use the tensor reference"
        )
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        raw: &candle_core::CudaStorage,
        raw_layout: &Layout,
        uniforms: &candle_core::CudaStorage,
        uniform_layout: &Layout,
        bounds: &candle_core::CudaStorage,
        bounds_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::{cudarc, WrapErr};
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let device = &raw.device;
        let (module, ptx) = compiled_ptx(device)?;
        let function = device.get_or_load_custom_func("pp_renyi_sample", &module, ptx.as_str())?;
        let (ra, rb) = raw_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous fused raw storage".into()))?;
        let (ua, ub) = uniform_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous fused uniform storage".into()))?;
        let (ba, bb) = bounds_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous fused bounds storage".into()))?;
        let raw = raw.as_cuda_slice::<f32>()?.slice(ra..rb);
        let uniforms = uniforms.as_cuda_slice::<f32>()?.slice(ua..ub);
        let bounds = bounds.as_cuda_slice::<u32>()?.slice(ba..bb);
        let mut output = unsafe { device.alloc::<u32>(3 * self.batch) }?;
        let batch = self.batch as u32;
        let choices = self.choices as u32;
        let guidance = self.guidance as u32;
        let padded = self.choices.next_power_of_two() as u32;
        let threads = 256u32;
        let config = LaunchConfig {
            grid_dim: (batch, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: (padded * 8 + threads * 16),
        };
        let mut builder = function.builder();
        builder
            .arg(&raw)
            .arg(&uniforms)
            .arg(&bounds)
            .arg(&mut output)
            .arg(&batch)
            .arg(&choices)
            .arg(&guidance)
            .arg(&self.alpha)
            .arg(&self.cfg)
            .arg(&self.offset)
            .arg(&padded);
        unsafe { builder.launch(config) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device.clone()),
            (3usize, self.batch).into(),
        ))
    }
}

#[cfg(feature = "cuda")]
fn compiled_ptx(
    device: &candle_core::CudaDevice,
) -> candle_core::Result<(String, std::sync::Arc<String>)> {
    use candle_core::cuda_backend::cudarc;
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex, OnceLock},
    };
    #[derive(Default)]
    struct Cache {
        device_arch: HashMap<candle_core::cuda_backend::DeviceId, String>,
        ptx: HashMap<String, Arc<String>>,
    }
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(Cache::default()))
        .lock()
        .map_err(|_| candle_core::Error::Msg("fused sampler PTX cache poisoned".into()))?;
    let arch = if let Some(arch) = cache.device_arch.get(&device.id()) {
        arch.clone()
    } else {
        let arch = crate::model::cuda_compat::optional_jit_target(device)?;
        cache.device_arch.insert(device.id(), arch.clone());
        arch
    };
    if let Some(ptx) = cache.ptx.get(&arch) {
        return Ok((format!("primitive_painter_renyi_v1_{arch}"), ptx.clone()));
    }
    let options = cudarc::nvrtc::CompileOptions {
        options: vec![format!("--gpu-architecture={arch}"), "--std=c++14".into()],
        fmad: Some(false),
        prec_div: Some(true),
        name: Some("primitive_painter_renyi_sample.cu".into()),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(include_str!("renyi_sample.cu"), options)
        .map_err(candle_core::Error::wrap)?;
    let ptx = Arc::new(ptx.to_src());
    cache.ptx.insert(arch.clone(), ptx.clone());
    Ok((format!("primitive_painter_renyi_v1_{arch}"), ptx))
}

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
    fn validates_geometry_and_never_uses_cpu_as_fused_backend() -> Result<()> {
        let (raw, u, b) = inputs(2, 128);
        assert_eq!(validate(&raw, &u, &b, 4., 1., 1, 100)?.batch, 2);
        assert!(sample(&raw, &u, &b, 4., 1., 1, 100).is_err());
        assert!(validate(&raw, &u, &b, 1., 1., 1, 0).is_err());
        assert!(validate(&raw, &u, &b, f32::NAN, 1., 1, 0).is_err());
        assert!(validate(&raw, &u, &b, 4., f32::INFINITY, 1, 0).is_err());
        assert!(validate(&raw, &u, &b, 4., 2., 1, 0).is_err());
        assert!(validate(&raw, &u, &b, 4., 1., 3, 0).is_err());
        assert!(validate(&raw, &u, &b, 4., 1., 1, u32::MAX).is_err());
        let too_wide = Tensor::zeros((2, 2049), DType::F32, &Device::Cpu)?;
        assert!(validate(&too_wide, &u, &b, 4., 1., 1, 0).is_err());
        Ok(())
    }
    #[test]
    fn nonzero_contiguous_offsets_and_cfg_pairs_are_supported() -> Result<()> {
        let raw = Tensor::zeros((8, 270), DType::F32, &Device::Cpu)?.narrow(0, 2, 4)?;
        let u = Tensor::zeros((6, 1), DType::F32, &Device::Cpu)?.narrow(0, 3, 2)?;
        let b = Tensor::zeros((6, 2), DType::U32, &Device::Cpu)?.narrow(0, 2, 2)?;
        assert!(u.layout().start_offset() > 0);
        let op = validate(&raw, &u, &b, 3.5, 2.5, 2, 2048)?;
        assert_eq!((op.batch, op.choices), (2, 270));
        let strided = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?.narrow(1, 0, 1)?;
        assert!(validate(&raw, &strided, &b, 4., 2., 2, 0).is_err());
        Ok(())
    }
}
