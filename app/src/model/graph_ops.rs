//! Graph-safe device operations: no CPU-derived stride/shape tensors and no
//! process-global Candle parameter cache. Ordinary execution remains separate.
use anyhow::{bail, Context, Result};
use candle_core::{CpuStorage, CustomOp3, DType, Layout, Shape, Tensor};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Embedding,
    Qkv,
    Swiglu,
    Indices,
    Mask,
    Finite,
    Pack,
}
#[derive(Clone, Copy, Debug)]
struct Operation {
    kind: Kind,
    rows: usize,
    width: usize,
    heads: usize,
    capacity: usize,
    guidance: usize,
}

impl Operation {
    // Kernels have dynamic batch storage, but their row/element metadata is
    // U32. Check the actual launch/index span rather than capping batch size.
    fn checked_elements(&self) -> Result<usize> {
        if self.rows == 0 {
            bail!("graph operation needs at least one row");
        }
        for n in [self.rows, self.width, self.heads, self.capacity, self.guidance] {
            u32::try_from(n).context("graph metadata exceeds U32 indexing")?;
        }
        let product = |dimensions: &[usize]| -> Result<usize> {
            dimensions.iter().try_fold(1usize, |n, &dimension| {
                if dimension == 0 { bail!("graph operation has an empty dimension"); }
                n.checked_mul(dimension).context("graph operation element count overflow")
            })
        };
        let count = match self.kind {
            Kind::Embedding => product(&[self.rows, self.width])?,
            Kind::Qkv => {
                u32::try_from(product(&[self.rows, self.heads])?).context("graph head grid exceeds U32")?;
                u32::try_from(product(&[3, self.heads, self.width])?).context("graph QKV row exceeds U32")?;
                product(&[3, self.rows, self.heads, self.width])?
            }
            Kind::Swiglu => product(&[self.rows, self.width])?,
            Kind::Indices => product(&[self.rows, self.heads, self.width])?,
            Kind::Mask => product(&[self.rows, self.heads, self.capacity])?,
            Kind::Finite => {
                if !matches!(self.guidance, 1 | 2) { bail!("graph CFG geometry differs"); }
                u32::try_from(product(&[self.guidance, self.width])?).context("graph finite scan exceeds U32")?;
                self.rows
            }
            Kind::Pack => {
                if !matches!(self.guidance, 1 | 2) { bail!("graph CFG geometry differs"); }
                product(&[self.rows, 10 + self.guidance * 2])?
            }
        };
        if !matches!(self.kind, Kind::Qkv) {
            u32::try_from(count).context("graph flattened launch exceeds U32 indexing")?;
        }
        // Typed output may be BF16 or F32; both byte sizes must be representable.
        count.checked_mul(std::mem::size_of::<f32>()).context("graph allocation byte count overflow")?;
        Ok(count)
    }
}

fn same_device(inputs: &[&Tensor]) -> Result<()> {
    let first = inputs.first().context("graph operation has no inputs")?;
    if !first.device().is_cuda()
        || inputs
            .iter()
            .any(|t| !t.device().same_device(first.device()) || !t.is_contiguous())
    {
        bail!("graph operations require contiguous CUDA inputs on one device");
    }
    Ok(())
}
pub(super) fn embedding(records: &Tensor, table: &Tensor, row_map: &Tensor) -> Result<Tensor> {
    same_device(&[records, table, row_map])?;
    let (logical_rows, fields) = records.dims2()?;
    let (vocab, width) = table.dims2()?;
    let rows = row_map.elem_count();
    if fields != 9
        || vocab != super::BOS as usize + 1
        || records.dtype() != DType::U32
        || row_map.dtype() != DType::U32
    {
        bail!("graph embedding geometry differs");
    }
    u32::try_from(logical_rows.checked_mul(9).context("graph attribute index overflow")?)
        .context("graph attribute index exceeds U32")?;
    Ok(records.apply_op3_no_bwd(
        table,
        row_map,
        &Operation {
            kind: Kind::Embedding,
            rows,
            width,
            heads: 0,
            capacity: 0,
            guidance: 0,
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_graph_kernels_accept_batches_above_sixteen() {
        for rows in [17, 32, 64, 96, 1000] {
            for (kind, width, heads, capacity, guidance, expected) in [
                (Kind::Embedding, 1536, 0, 0, 0, rows * 1536),
                (Kind::Qkv, 64, 24, 1024, 0, 3 * rows * 24 * 64),
                (Kind::Swiglu, 4096, 0, 0, 0, rows * 4096),
                (Kind::Indices, 64, 24, 0, 0, rows * 24 * 64),
                (Kind::Mask, 0, 24, 1024, 0, rows * 24 * 1024),
                (Kind::Finite, 2048, 0, 0, 2, rows),
                (Kind::Pack, 0, 0, 0, 2, rows * 14),
            ] {
                let op = Operation { kind, rows, width, heads, capacity, guidance };
                assert_eq!(op.checked_elements().unwrap(), expected, "{op:?}");
            }
        }
    }

    #[test]
    fn graph_kernel_geometry_rejects_truncation_and_overflow_before_allocation() {
        let base = Operation { kind: Kind::Mask, rows: 1, width: 0, heads: 24, capacity: 1024, guidance: 0 };
        assert!(Operation { rows: 0, ..base }.checked_elements().is_err());
        assert!(Operation { rows: usize::MAX, ..base }.checked_elements().is_err());
        assert!(Operation { rows: u32::MAX as usize / 1024, ..base }.checked_elements().is_err());
        assert!(Operation { kind: Kind::Qkv, rows: 2, heads: u32::MAX as usize, width: 64, ..base }
            .checked_elements().is_err());
        assert!(Operation { kind: Kind::Pack, rows: u32::MAX as usize, guidance: 2, ..base }
            .checked_elements().is_err());
    }
}
pub(super) fn qkv(
    input: &Tensor,
    parameters: &Tensor,
    positions: &Tensor,
    heads: usize,
    dim: usize,
) -> Result<Tensor> {
    same_device(&[input, parameters, positions])?;
    let (rows, columns) = input.dims2()?;
    if dim != 64
        || Some(columns) != heads.checked_mul(dim).and_then(|n| n.checked_mul(3))
        || positions.dims() != [rows]
        || positions.dtype() != DType::U32
        || input.dtype() != parameters.dtype()
        || parameters.elem_count() != 2 * dim + super::MAX_CONTEXT * dim
    {
        bail!("graph QKV/RoPE geometry differs");
    }
    Ok(input.apply_op3_no_bwd(
        parameters,
        positions,
        &Operation {
            kind: Kind::Qkv,
            rows,
            width: dim,
            heads,
            capacity: super::MAX_CONTEXT,
            guidance: 0,
        },
    )?)
}
pub(super) fn swiglu(input: &Tensor) -> Result<Tensor> {
    same_device(&[input])?;
    let (rows, columns) = input.dims2()?;
    if columns % 2 != 0 {
        bail!("graph FFN width must be even");
    }
    Ok(input.apply_op3_no_bwd(
        input,
        input,
        &Operation {
            kind: Kind::Swiglu,
            rows,
            width: columns / 2,
            heads: 0,
            capacity: 0,
            guidance: 0,
        },
    )?)
}
pub(super) fn indices(lengths: &Tensor, heads: usize, dim: usize) -> Result<Tensor> {
    same_device(&[lengths])?;
    let rows = lengths.dims1()?;
    if lengths.dtype() != DType::U32 || heads == 0 || dim == 0 {
        bail!("graph write indices differ");
    }
    Ok(lengths.apply_op3_no_bwd(
        lengths,
        lengths,
        &Operation {
            kind: Kind::Indices,
            rows,
            width: dim,
            heads,
            capacity: 0,
            guidance: 0,
        },
    )?)
}
pub(super) fn mask(lengths: &Tensor, heads: usize, capacity: usize) -> Result<Tensor> {
    same_device(&[lengths])?;
    let rows = lengths.dims1()?;
    if lengths.dtype() != DType::U32 || !(1..=super::MAX_CONTEXT).contains(&capacity) {
        bail!("graph mask geometry differs");
    }
    Ok(lengths.apply_op3_no_bwd(
        lengths,
        lengths,
        &Operation {
            kind: Kind::Mask,
            rows,
            width: 0,
            heads,
            capacity,
            guidance: 0,
        },
    )?)
}
pub(super) fn finite(input: &Tensor, guidance: usize) -> Result<Tensor> {
    same_device(&[input])?;
    let (physical, width) = input.dims2()?;
    if !matches!(guidance, 1 | 2) || physical % guidance != 0 || input.dtype() != DType::F32 {
        bail!("graph finite check geometry differs");
    }
    Ok(input.apply_op3_no_bwd(
        input,
        input,
        &Operation {
            kind: Kind::Finite,
            rows: physical / guidance,
            width,
            heads: 0,
            capacity: 0,
            guidance,
        },
    )?)
}
pub(super) fn pack(
    records: &Tensor,
    terms: &Tensor,
    valid: &Tensor,
    guidance: usize,
) -> Result<Tensor> {
    same_device(&[records, terms, valid])?;
    let (rows, fields) = records.dims2()?;
    if fields != 9
        || !matches!(guidance, 1 | 2)
        || records.dtype() != DType::U32
        || terms.dtype() != DType::F32
        || valid.dtype() != DType::U32
        || rows.checked_mul(guidance).is_none_or(|physical| terms.dims() != [physical, 2])
        || valid.dims() != [rows, 1]
    {
        bail!("graph packed output geometry differs");
    }
    Ok(records.apply_op3_no_bwd(
        terms,
        valid,
        &Operation {
            kind: Kind::Pack,
            rows,
            width: 0,
            heads: 0,
            capacity: 0,
            guidance,
        },
    )?)
}

impl CustomOp3 for Operation {
    fn name(&self) -> &'static str {
        "primitive-painter-static-graph-op"
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
            "static graph kernels require NVIDIA CUDA; use ordinary compatibility execution"
        )
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        la: &Layout,
        b: &candle_core::CudaStorage,
        lb: &Layout,
        c: &candle_core::CudaStorage,
        lc: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::{cudarc, CudaStorageSlice as S, WrapErr};
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        self.checked_elements().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let device = &a.device;
        let (module, ptx) = compiled_ptx(device)?;
        let r = self.rows as u32;
        let w = self.width as u32;
        let h = self.heads as u32;
        let cap = self.capacity as u32;
        let g = self.guidance as u32;
        let offsets = |layout: &Layout| {
            layout
                .contiguous_offsets()
                .ok_or_else(|| candle_core::Error::Msg("graph input is not contiguous".into()))
        };
        let (aa, az) = offsets(la)?;
        let (ba, bz) = offsets(lb)?;
        let (ca, cz) = offsets(lc)?;
        let fn_for = |name: &str| device.get_or_load_custom_func(name, &module, ptx.as_str());
        let typed_output = |name: &str,
                            elements: usize,
                            shape: Shape,
                            first_is_table: bool|
         -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
            macro_rules! run_typed {
                ($input:expr,$variant:ident,$kind:expr) => {{
                    let function = fn_for(name)?;
                    let source = $input.slice(if first_is_table { ba..bz } else { aa..az });
                    // Infer BF16's type from its input slice, avoiding a second half crate.
                    fn allocate_like<T: cudarc::driver::DeviceRepr>(
                        dev: &candle_core::CudaDevice,
                        _like: &cudarc::driver::CudaView<'_, T>,
                        len: usize,
                    ) -> candle_core::Result<cudarc::driver::CudaSlice<T>> {
                        unsafe { dev.alloc::<T>(len) }
                    }
                    let mut out = allocate_like(device, &source, elements)?;
                    let kind = $kind as u32;
                    let config = if matches!(self.kind, Kind::Qkv) {
                        LaunchConfig {
                            grid_dim: (r * h, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        }
                    } else {
                        LaunchConfig::for_num_elems(elements as u32)
                    };
                    let mut builder = function.builder();
                    match self.kind {
                        Kind::Embedding => {
                            let ids = a.as_cuda_slice::<u32>()?.slice(aa..az);
                            let map = c.as_cuda_slice::<u32>()?.slice(ca..cz);
                            builder
                                .arg(&ids)
                                .arg(&source)
                                .arg(&map)
                                .arg(&mut out)
                                .arg(&r)
                                .arg(&w)
                                .arg(&kind);
                            unsafe { builder.launch(config) }.w()?;
                        }
                        Kind::Qkv => {
                            let params = match &b.slice {
                                S::$variant(params) => params.slice(ba..bz),
                                _ => candle_core::bail!("graph QKV parameter dtype differs"),
                            };
                            let positions = c.as_cuda_slice::<u32>()?.slice(ca..cz);
                            builder
                                .arg(&source)
                                .arg(&params)
                                .arg(&positions)
                                .arg(&mut out)
                                .arg(&r)
                                .arg(&h)
                                .arg(&w)
                                .arg(&cap)
                                .arg(&kind);
                            unsafe { builder.launch(config) }.w()?;
                        }
                        Kind::Swiglu => {
                            builder
                                .arg(&source)
                                .arg(&mut out)
                                .arg(&r)
                                .arg(&w)
                                .arg(&kind);
                            unsafe { builder.launch(config) }.w()?;
                        }
                        _ => unreachable!(),
                    }
                    Ok((
                        candle_core::CudaStorage {
                            slice: S::$variant(out),
                            device: device.clone(),
                        },
                        shape,
                    ))
                }};
            }
            let input = if first_is_table { &b.slice } else { &a.slice };
            match input {
                S::BF16(input) => run_typed!(input, BF16, 1),
                S::F32(input) => run_typed!(input, F32, 0),
                _ => candle_core::bail!("static graph requires BF16 or F32 model tensors"),
            }
        };
        match self.kind {
            Kind::Embedding => typed_output(
                "pp_graph_embedding",
                self.rows * self.width,
                (self.rows, self.width).into(),
                true,
            ),
            Kind::Qkv => typed_output(
                "pp_graph_qkv",
                3 * self.rows * self.heads * self.width,
                (3, self.rows * self.heads, 1, self.width).into(),
                false,
            ),
            Kind::Swiglu => typed_output(
                "pp_graph_swiglu",
                self.rows * self.width,
                (self.rows, self.width).into(),
                false,
            ),
            Kind::Indices => {
                let function = fn_for("pp_graph_indices")?;
                let source = a.as_cuda_slice::<u32>()?.slice(aa..az);
                let count = self.rows * self.heads * self.width;
                let mut out = unsafe { device.alloc::<u32>(count) }?;
                let mut builder = function.builder();
                builder.arg(&source).arg(&mut out).arg(&r).arg(&h).arg(&w);
                unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(out, device.clone()),
                    (self.rows * self.heads, 1, self.width).into(),
                ))
            }
            Kind::Mask => {
                let function = fn_for("pp_graph_mask")?;
                let source = a.as_cuda_slice::<u32>()?.slice(aa..az);
                let count = self.rows * self.heads * self.capacity;
                let mut out = unsafe { device.alloc::<f32>(count) }?;
                let mut builder = function.builder();
                builder.arg(&source).arg(&mut out).arg(&r).arg(&h).arg(&cap);
                unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(out, device.clone()),
                    (self.rows * self.heads, 1, self.capacity).into(),
                ))
            }
            Kind::Finite => {
                let function = fn_for("pp_graph_finite")?;
                let source = a.as_cuda_slice::<f32>()?.slice(aa..az);
                let mut out = unsafe { device.alloc::<u32>(self.rows) }?;
                let mut builder = function.builder();
                builder.arg(&source).arg(&mut out).arg(&r).arg(&g).arg(&w);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (r, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(out, device.clone()),
                    (self.rows, 1).into(),
                ))
            }
            Kind::Pack => {
                let function = fn_for("pp_graph_pack")?;
                let records = a.as_cuda_slice::<u32>()?.slice(aa..az);
                let terms = b.as_cuda_slice::<f32>()?.slice(ba..bz);
                let valid = c.as_cuda_slice::<u32>()?.slice(ca..cz);
                let stride = 10 + self.guidance * 2;
                let count = self.rows * stride;
                let mut out = unsafe { device.alloc::<f32>(count) }?;
                let mut builder = function.builder();
                builder
                    .arg(&records)
                    .arg(&terms)
                    .arg(&valid)
                    .arg(&mut out)
                    .arg(&r)
                    .arg(&g);
                unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(out, device.clone()),
                    (self.rows, stride).into(),
                ))
            }
        }
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
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<String>>>> = OnceLock::new();
    let arch = super::cuda_compat::optional_jit_target(device)?;
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| candle_core::Error::Msg("graph PTX cache poisoned".into()))?;
    if let Some(ptx) = cache.get(&arch) {
        return Ok((format!("primitive_painter_graph_ops_{arch}"), ptx.clone()));
    }
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        include_str!("graph_ops.cu"),
        cudarc::nvrtc::CompileOptions {
            options: vec![format!("--gpu-architecture={arch}"), "--std=c++14".into()],
            fmad: Some(false),
            prec_div: Some(true),
            name: Some("primitive_painter_graph_ops.cu".into()),
            ..Default::default()
        },
    )
    .map_err(candle_core::Error::wrap)?;
    let ptx = Arc::new(ptx.to_src());
    cache.insert(arch.clone(), ptx.clone());
    Ok((format!("primitive_painter_graph_ops_{arch}"), ptx))
}
