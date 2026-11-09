//! Optional device-specific fused decode / Split-K, with the ordinary tensor path kept.
use anyhow::{bail, Result};
use candle_core::{CpuStorage, CustomOp3, DType, Layout, Shape, Tensor};

#[derive(Clone)]
struct DecodeAttention { heads: usize, rows: usize, length: usize, split: usize, lengths: Option<Tensor> }

pub(super) fn decode_attention(q: &Tensor, k: &Tensor, v: &Tensor, lengths: Option<&Tensor>, heads: usize, split_tokens: usize) -> Result<Tensor> {
    let (rows, query, width) = q.dims3()?;
    let (key_rows, length, key_width) = k.dims3()?;
    if heads == 0 || rows == 0 || rows > u32::MAX as usize || rows%heads != 0 || query != 1 || width != 64
        || key_rows != rows || key_width != 64 || length == 0 || length > 1024 || v.dims() != k.dims()
        || q.dtype()!=DType::F32 || !matches!(k.dtype(),DType::BF16|DType::F32) || k.dtype()!=v.dtype()
        || !q.device().same_device(k.device()) || !q.device().same_device(v.device()) {
        bail!("fused decode attention requires F32[batch*heads,1,64] and same-device BF16/F32 K/V");
    }
    for tensor in [q,k,v] {
        if tensor.stride()[2] != 1 || tensor.stride()[1] != 64 {
            bail!("fused attention requires dense width and token strides");
        }
    }
    if let Some(lengths) = lengths {
        if lengths.dims1()? != rows/heads || lengths.dtype()!=DType::U32 || !lengths.is_contiguous() || !q.device().same_device(lengths.device()) {
            bail!("fused attention row lengths require same-device U32[batch]");
        }
    }
    #[cfg(feature="rocm")]
    if q.device().is_rocm() { return super::hip_attention::decode_attention(q,k,v,lengths,heads,split_tokens); }
    if !q.device().is_cuda() { bail!("NVIDIA fused attention needs CUDA; use ordinary attention on this backend"); }
    let split = if split_tokens==0 { length } else { split_tokens.clamp(32,1024) };
    let op = DecodeAttention { heads,rows,length,split,lengths:lengths.cloned() };
    Ok(q.apply_op3_no_bwd(k,v,&op)?)
}

impl CustomOp3 for DecodeAttention {
    fn name(&self)->&'static str { "primitive-painter-decode-attention" }
    fn cpu_fwd(&self,_:&CpuStorage,_:&Layout,_:&CpuStorage,_:&Layout,_:&CpuStorage,_:&Layout)->candle_core::Result<(CpuStorage,Shape)> {
        candle_core::bail!("select ordinary attention for CPU")
    }
    #[cfg(feature="cuda")]
    fn cuda_fwd(&self,q:&candle_core::CudaStorage,ql:&Layout,k:&candle_core::CudaStorage,kl:&Layout,v:&candle_core::CudaStorage,vl:&Layout)->candle_core::Result<(candle_core::CudaStorage,Shape)> {
        use candle_core::{backend::BackendStorage,Storage,cuda_backend::{cudarc,WrapErr}};
        use cudarc::driver::{LaunchConfig,PushKernelArg};
        let device=&q.device;
        let (module,ptx)=compiled_ptx(device)?;
        let name=match k.dtype(){DType::F32=>"pp_attention_f32",DType::BF16=>"pp_attention_bf16",_=>candle_core::bail!("unsupported attention KV dtype")};
        let function=device.get_or_load_custom_func(name,&module,ptx.as_str())?;
        let splits=self.length.div_ceil(self.split);
        let mut output=unsafe{device.alloc::<f32>(self.rows*64)}?;
        let mut partial=if splits>1 {Some(unsafe{device.alloc::<f32>(self.rows*splits*66)}?)}else{None};
        let q=q.as_cuda_slice::<f32>()?.slice(ql.start_offset()..);
        let lens=self.lengths.as_ref().map(Tensor::storage_and_layout);
        let lens_view=match &lens {
            Some((storage,layout))=>match &**storage {
                Storage::Cuda(storage)=>Some(storage.as_cuda_slice::<u32>()?.slice(layout.start_offset()..)),
                _=>candle_core::bail!("fused attention lengths not on CUDA"),
            },None=>None,
        };
        let heads=self.heads as u32; let n=self.length as u32; let split=self.split as u32; let ns=splits as u32;
        let dynamic=u32::from(lens_view.is_some());let null=0u64;
        let qs=ql.stride()[0] as u64;let ks=kl.stride()[0] as u64;let vs=vl.stride()[0] as u64;
        let config=LaunchConfig{grid_dim:(self.rows as u32,ns,1),block_dim:(128,1,1),shared_mem_bytes:((self.split+128)*4)as u32};
        macro_rules! launch {
            ($key:expr,$value:expr)=>{{
                let kk=$key.slice(kl.start_offset()..);
                let vv=$value.slice(vl.start_offset()..);
                let mut builder=function.builder();builder.arg(&q).arg(&kk).arg(&vv);
                if let Some(lens)=&lens_view {builder.arg(lens);}else{builder.arg(&null);}
                if let Some(partial)=partial.as_mut() {builder.arg(partial);}else{builder.arg(&mut output);}
                builder.arg(&heads).arg(&n).arg(&split).arg(&ns).arg(&dynamic).arg(&qs).arg(&ks).arg(&vs);
                unsafe{builder.launch(config)}.w()?;
            }};
        }
        use candle_core::cuda_backend::CudaStorageSlice as S;
        match (&k.slice,&v.slice){(S::F32(k),S::F32(v))=>launch!(k,v),(S::BF16(k),S::BF16(v))=>launch!(k,v),_=>candle_core::bail!("attention K/V storage differs")}
        if splits>1 {
            let merge=device.get_or_load_custom_func("pp_attention_merge",&module,ptx.as_str())?;
            let mut builder=merge.builder();builder.arg(partial.as_ref().unwrap()).arg(&mut output).arg(&ns);
            unsafe{builder.launch(LaunchConfig{grid_dim:(self.rows as u32,1,1),block_dim:(64,1,1),shared_mem_bytes:0})}.w()?;
        }
        Ok((candle_core::CudaStorage::wrap_cuda_slice(output,device.clone()),(self.rows,1,64).into()))
    }
}

#[cfg(feature="cuda")]
fn compiled_ptx(device:&candle_core::CudaDevice)->candle_core::Result<(String,std::sync::Arc<String>)> {
    use candle_core::cuda_backend::cudarc;
    use std::{collections::HashMap,sync::{Arc,Mutex,OnceLock}};
    static CACHE:OnceLock<Mutex<HashMap<String,Arc<String>>>>=OnceLock::new();
    let arch=super::cuda_compat::optional_jit_target(device)?;
    let mut cache=CACHE.get_or_init(||Mutex::new(HashMap::new())).lock().map_err(|_|candle_core::Error::Msg("attention PTX cache poisoned".into()))?;
    let ptx=if let Some(ptx)=cache.get(&arch){ptx.clone()}else{
        let options=cudarc::nvrtc::CompileOptions{options:vec![format!("--gpu-architecture={arch}"),"--std=c++14".into()],fmad:Some(false),prec_div:Some(true),name:Some("primitive_painter_attention.cu".into()),..Default::default()};
        let ptx=Arc::new(cudarc::nvrtc::compile_ptx_with_opts(include_str!("decode_attention.cu"),options).map_err(candle_core::Error::wrap)?.to_src());
        cache.insert(arch.clone(),ptx.clone());ptx
    };
    Ok((format!("primitive_painter_decode_attention_{arch}"),ptx))
}
