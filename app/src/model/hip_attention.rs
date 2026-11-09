//! Optional HIP decode attention, separate from the unchanged CUDA operator.
use anyhow::{bail,Result};
use candle_core::{CpuStorage,CustomOp3,DType,Layout,Shape,Tensor};

const SOURCE:&str=include_str!("decode_attention.hip");

pub(crate) fn prepare(device:&candle_core::Device)->Result<()> {
    let module=super::hip_kernel::module(device.as_rocm_device()?,"pp_decode_attention",SOURCE)?;
    for name in ["pp_attention_f32","pp_attention_bf16","pp_attention_merge"]{
        module.get_function(name).map_err(|error|anyhow::anyhow!("HIP_PREPARATION: {error}"))?;
    }
    Ok(())
}

/// Nonzero private scratch validation for granting the optional Split-K
/// capability. No user KV, history or RNG is involved. Completed numerical
/// mismatch disables this feature; failed post-launch readback is uncertain.
pub(crate) fn validate_split_k(device:&candle_core::Device)->Result<()> {
    prepare(device)?;
    let values=|count:usize,salt:usize,scale:f32|(0..count).map(|index|(((index*37+salt)%107)as f32-53.)*scale).collect::<Vec<_>>();
    let q=Tensor::from_vec(values(64,17,0.013),(1,1,64),device)?;
    for dtype in [DType::BF16,DType::F32]{
        let k=Tensor::from_vec(values(257*64,23,0.011),(1,257,64),device)?.to_dtype(dtype)?;
        let v=Tensor::from_vec(values(257*64,31,0.017),(1,257,64),device)?.to_dtype(dtype)?;
        let reference_tensor={
            let _guard=candle_core::rocm_backend::strict::DeviceOnlyGuard::enter();
            let scores=(q.matmul(&k.to_dtype(DType::F32)?.transpose(1,2)?)?/8.)?;
            candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v.to_dtype(DType::F32)?)?
        };
        let reference=reference_tensor.flatten_all()?.to_vec1::<f32>()
            .map_err(|error|anyhow::anyhow!("HIP_EXECUTION_UNCERTAIN: explicit F32 reference readback: {error}"))?;
        let ordinary=decode_attention(&q,&k,&v,None,1,0)?.flatten_all()?.to_vec1::<f32>()
            .map_err(|error|anyhow::anyhow!("HIP_EXECUTION_UNCERTAIN: Split-K reference readback: {error}"))?;
        let split=decode_attention(&q,&k,&v,None,1,128)?.flatten_all()?.to_vec1::<f32>()
            .map_err(|error|anyhow::anyhow!("HIP_EXECUTION_UNCERTAIN: Split-K validation readback: {error}"))?;
        let pair_error=ordinary.iter().zip(&split).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
        let ordinary_error=reference.iter().zip(&ordinary).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
        let split_error=reference.iter().zip(&split).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
        if !reference.iter().chain(&ordinary).chain(&split).all(|value|value.is_finite())
            ||pair_error>1e-4||ordinary_error>1e-4||split_error>1e-4{
            bail!("HIP_PREPARATION: {dtype:?} Split-K private numerical check failed, pair={pair_error}, ordinary_vs_reference={ordinary_error}, split_vs_reference={split_error}");
        }
    }
    Ok(())
}

pub(super) fn decode_attention(q:&Tensor,k:&Tensor,v:&Tensor,lengths:Option<&Tensor>,heads:usize,split_tokens:usize)->Result<Tensor>{
    let(rows,query,width)=q.dims3()?;
    let(key_rows,length,key_width)=k.dims3()?;
    if heads==0||rows==0||rows>u32::MAX as usize||rows%heads!=0||query!=1||width!=64
        ||key_rows!=rows||key_width!=64||length==0||length>1024||v.dims()!=k.dims()
        ||q.dtype()!=DType::F32||!matches!(k.dtype(),DType::BF16|DType::F32)||k.dtype()!=v.dtype()
        ||!q.device().is_rocm()||!q.device().same_device(k.device())||!q.device().same_device(v.device()){
        bail!("HIP decode requires same-device F32[batch*heads,1,64] Q and BF16/F32[batch*heads,length,64] K/V");
    }
    for tensor in [q,k,v]{if tensor.stride()[2]!=1||tensor.stride()[1]!=64{bail!("HIP attention requires dense width/token strides");}}
    if let Some(lengths)=lengths{if lengths.dims1()?!=rows/heads||lengths.dtype()!=DType::U32||!lengths.is_contiguous()||!q.device().same_device(lengths.device()){bail!("HIP attention lengths require same-device U32[batch]");}}
    let split=if split_tokens==0{length}else{split_tokens.clamp(32,1024)};
    Ok(q.apply_op3_no_bwd(k,v,&Decode{heads,rows,length,split,lengths:lengths.cloned()})?)
}

struct Decode{heads:usize,rows:usize,length:usize,split:usize,lengths:Option<Tensor>}
fn checked_span(storage:&candle_core::RocmStorage,layout:&Layout,rows:usize,length:usize,element_bytes:usize)->candle_core::Result<()> {
    let last=rows.checked_sub(1).and_then(|row|row.checked_mul(layout.stride()[0]))
        .and_then(|n|n.checked_add((length-1)*64+64))
        .and_then(|n|n.checked_add(layout.start_offset())).and_then(|n|n.checked_mul(element_bytes));
    if last.is_none_or(|last|last>storage.byte_len()){candle_core::bail!("HIP attention view exceeds underlying storage");}
    Ok(())
}
impl CustomOp3 for Decode{
    fn name(&self)->&'static str{"primitive-painter-hip-decode-attention"}
    fn cpu_fwd(&self,_:&CpuStorage,_:&Layout,_:&CpuStorage,_:&Layout,_:&CpuStorage,_:&Layout)->candle_core::Result<(CpuStorage,Shape)>{candle_core::bail!("HIP attention never falls back to CPU")}
    fn rocm_fwd(&self,q:&candle_core::RocmStorage,ql:&Layout,k:&candle_core::RocmStorage,kl:&Layout,v:&candle_core::RocmStorage,vl:&Layout)->candle_core::Result<(candle_core::RocmStorage,Shape)>{
        use candle_core::{backend::{BackendDevice,BackendStorage},Storage};
        use std::ffi::c_void;
        let device=q.rocm_device();
        let module=super::hip_kernel::module(device,"pp_decode_attention",SOURCE)?;
        let element_bytes=if k.dtype()==DType::BF16{2}else{4};
        checked_span(q,ql,self.rows,1,4)?;checked_span(k,kl,self.rows,self.length,element_bytes)?;checked_span(v,vl,self.rows,self.length,element_bytes)?;
        let output_shape:Shape=(self.rows,1,64).into();
        let mut output=unsafe{device.alloc_uninit(&output_shape,DType::F32)?};
        let splits=self.length.div_ceil(self.split);
        // Resolve every required symbol before enqueuing the first partition.
        // A missing merge symbol must never look like a pre-launch failure
        // after the partition kernel has already modified device scratch.
        let kernel_name=if k.dtype()==DType::BF16{"pp_attention_bf16"}else{"pp_attention_f32"};
        module.get_function(kernel_name).map_err(|error|candle_core::Error::Msg(format!("HIP_PREPARATION: {error}")))?;
        if splits>1{module.get_function("pp_attention_merge").map_err(|error|candle_core::Error::Msg(format!("HIP_PREPARATION: {error}")))?;}
        let mut partial=if splits>1{Some(unsafe{device.alloc_uninit(&(self.rows*splits*66).into(),DType::F32)?})}else{None};
        let length_storage=self.lengths.as_ref().map(Tensor::storage_and_layout);
        let mut lp:*const c_void=match &length_storage{
            Some((storage,layout))=>match &**storage{Storage::Rocm(storage)=>{
                let end=(layout.start_offset()+self.rows/self.heads)*4;
                if end>storage.byte_len(){candle_core::bail!("HIP length view exceeds storage");}
                (unsafe{storage.device_ptr().add(layout.start_offset()*4)}) as *const c_void
            },_=>candle_core::bail!("HIP attention lengths changed device")},None=>std::ptr::null(),
        };
        let mut qp=unsafe{q.device_ptr().add(ql.start_offset()*4)} as *const c_void;
        let mut kp=unsafe{k.device_ptr().add(kl.start_offset()*element_bytes)} as *const c_void;
        let mut vp=unsafe{v.device_ptr().add(vl.start_offset()*element_bytes)} as *const c_void;
        let mut op=partial.as_mut().map_or_else(||output.device_ptr_mut(),|part|part.device_ptr_mut()) as *mut c_void;
        let(mut heads,mut n,mut split,mut ns,mut dynamic)=(self.heads as u32,self.length as u32,self.split as u32,splits as u32,u32::from(self.lengths.is_some()));
        let(mut qs,mut ks,mut vs)=(ql.stride()[0] as u64,kl.stride()[0] as u64,vl.stride()[0] as u64);
        let mut args=[&mut qp as *mut _ as *mut c_void,&mut kp as *mut _ as *mut c_void,&mut vp as *mut _ as *mut c_void,&mut lp as *mut _ as *mut c_void,&mut op as *mut _ as *mut c_void,&mut heads as *mut _ as *mut c_void,&mut n as *mut _ as *mut c_void,&mut split as *mut _ as *mut c_void,&mut ns as *mut _ as *mut c_void,&mut dynamic as *mut _ as *mut c_void,&mut qs as *mut _ as *mut c_void,&mut ks as *mut _ as *mut c_void,&mut vs as *mut _ as *mut c_void];
        super::hip_kernel::launch(device,&module,kernel_name,(self.rows as u32,ns,1),(256,1,1),((self.split+256)*4)as u32,&mut args)?;
        if let Some(partial)=&partial{
            let mut input=partial.device_ptr() as *const c_void;
            let mut output_ptr=output.device_ptr_mut() as *mut c_void;
            let mut args=[&mut input as *mut _ as *mut c_void,&mut output_ptr as *mut _ as *mut c_void,&mut ns as *mut _ as *mut c_void];
            super::hip_kernel::launch(device,&module,"pp_attention_merge",(self.rows as u32,1,1),(64,1,1),0,&mut args)?;
        }
        Ok((output,output_shape))
    }
}
