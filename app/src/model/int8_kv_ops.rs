//! INT8 KV storage operations; decoding reads packed history directly.
use anyhow::{bail, ensure, Result};
use candle_core::{DType, Device, Tensor};

fn geometry(t: &Tensor) -> Result<(usize, usize)> {
    let (h,n,d)=t.dims3()?;
    ensure!(h>0&&n>0&&d==64&&t.is_contiguous(),"INT8 KV requires dense [heads,tokens,64]");
    ensure!(h.checked_mul(n).and_then(|x|x.checked_mul(64)).is_some(),"INT8 KV dimensions overflow");
    Ok((h,n))
}
pub(super) fn quantize(input:&Tensor)->Result<(Tensor,Tensor)>{
    let (h,n)=geometry(input)?;let x=input.to_dtype(DType::F32)?;
    if input.device().is_cpu(){
        let values=x.flatten_all()?.to_vec1::<f32>()?;let mut packed=Vec::with_capacity(values.len());let mut scales=Vec::with_capacity(h*n);
        for row in values.chunks_exact(64){
            ensure!(row.iter().all(|v|v.is_finite()),"non-finite INT8 KV input");
            let s=(row.iter().map(|x|x.abs()).fold(0f32,f32::max)/127.).max(1e-12);scales.push(s);
            packed.extend(row.iter().map(|v|((v/s).round_ties_even().clamp(-127.,127.)as i8)as u8));
        }
        return Ok((Tensor::from_vec(packed,(h,n,64),input.device())?,Tensor::from_vec(scales,(h,n),input.device())?));
    }
    #[cfg(feature="cuda")]
    if input.device().is_cuda(){
        let scale=x.apply_op1_no_bwd(&cuda::Scale{rows:h*n})?;
        let packed=x.apply_op2_no_bwd(&scale,&cuda::Pack{n:h*n*64,unpack:false})?;
        return Ok((packed,scale));
    }
    bail!("KV INT8 is unavailable on this native provider")
}
pub(super) fn dequantize(packed:&Tensor,scale:&Tensor,dtype:DType)->Result<Tensor>{
    let (h,n)=geometry(packed)?;
    ensure!(packed.dtype()==DType::U8&&scale.dtype()==DType::F32&&scale.dims()==[h,n]&&scale.is_contiguous()&&packed.device().same_device(scale.device()),"INT8 KV scales differ");
    if packed.device().is_cpu(){
        let p=packed.flatten_all()?.to_vec1::<u8>()?;let s=scale.flatten_all()?.to_vec1::<f32>()?;
        return Ok(Tensor::from_vec(p.iter().enumerate().map(|(i,&x)|(x as i8)as f32*s[i/64]).collect::<Vec<_>>(),(h,n,64),packed.device())?.to_dtype(dtype)?);
    }
    #[cfg(feature="cuda")]
    if packed.device().is_cuda(){return Ok(packed.apply_op2_no_bwd(scale,&cuda::Pack{n:h*n*64,unpack:true})?.to_dtype(dtype)?);}
    bail!("KV INT8 dequantization is unavailable on this provider")
}
#[allow(clippy::too_many_arguments)]
pub(super) fn attention(q:&Tensor,k:&Tensor,v:&Tensor,ks:&Tensor,vs:&Tensor,sk:&Tensor,sv:&Tensor,past:&Tensor,heads:usize,split_tokens:usize)->Result<Tensor>{
    let (rows,capacity)=geometry(k)?;
    ensure!(heads>0&&rows%heads==0&&capacity<=1024&&v.dims()==k.dims()&&v.is_contiguous()&&k.dtype()==DType::U8&&v.dtype()==DType::U8,"INT8 attention geometry differs");
    for t in [q,sk,sv]{ensure!(t.dims()==[rows,1,64]&&t.dtype()==DType::F32&&t.is_contiguous(),"INT8 scratch must be F32[head_rows,1,64]");}
    for t in [ks,vs]{ensure!(t.dims()==[rows,capacity]&&t.dtype()==DType::F32&&t.is_contiguous(),"INT8 scales must be F32[head_rows,capacity]");}
    ensure!(past.dims()==[rows/heads]&&past.dtype()==DType::U32&&past.is_contiguous(),"INT8 attention lengths differ");
    for t in [q,v,ks,vs,sk,sv,past]{ensure!(k.device().same_device(t.device()),"INT8 attention device mismatch");}
    #[cfg(feature="cuda")]
    if q.device().is_cuda(){
        ensure!(u32::try_from(rows).is_ok(),"INT8 attention row grid overflow");
        return Ok(q.apply_op3_no_bwd(k,v,&cuda::Attention{ks:ks.clone(),vs:vs.clone(),sk:sk.clone(),sv:sv.clone(),past:past.clone(),rows,capacity,heads,split:if split_tokens==0{128}else{split_tokens.clamp(32,1024)}})?);
    }
    bail!("KV INT8 fused attention is only supported by the CUDA native provider")
}
/// Private nonzero-data validation, independent of the user's cache and RNG.
pub(super) fn prepare(device:&Device)->Result<()>{
    ensure!(device.is_cuda(),"KV INT8 needs CUDA in the native provider");
    let data=(0..384).map(|i|((i as f32*0.17).sin())*2.).collect::<Vec<_>>();
    let x=Tensor::from_vec(data.clone(),(2,3,64),device)?;
    let (k,s)=quantize(&x)?;let restored=dequantize(&k,&s,DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    ensure!(restored.iter().zip(&data).all(|(a,b)|a.is_finite()&&(a-b).abs()<0.009),"KV INT8 quantization self-check failed");
    let qdata=(0..128).map(|i|(i as f32*0.07).cos()).collect::<Vec<_>>();
    let scratch=(0..128).map(|i|(i as f32*0.09).sin()).collect::<Vec<_>>();
    let q=Tensor::from_vec(qdata.clone(),(2,1,64),device)?;let sk=Tensor::from_vec(scratch.clone(),(2,1,64),device)?;
    let past=Tensor::from_vec(vec![2u32,1],2,device)?;
    let actual=attention(&q,&k,&k,&s,&s,&sk,&sk,&past,1,32)?.flatten_all()?.to_vec1::<f32>()?;
    let rounded=dequantize(&k,&s,DType::BF16)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    for row in 0..2{let len=if row==0{2}else{1};let mut scores=Vec::new();
        for t in 0..=len{scores.push((0..64).map(|d|qdata[row*64+d]*if t==len{scratch[row*64+d]}else{rounded[(row*3+t)*64+d]}).sum::<f32>()*0.125);}
        let max=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);let p=scores.iter().map(|v|(v-max).exp()).collect::<Vec<_>>();let z=p.iter().sum::<f32>();
        for d in 0..64{let expected=(0..=len).map(|t|p[t]*if t==len{scratch[row*64+d]}else{rounded[(row*3+t)*64+d]}).sum::<f32>()/z;
            ensure!(actual[row*64+d].is_finite()&&(actual[row*64+d]-expected).abs()<2e-5,"KV INT8 attention self-check failed");}
    }
    // Cover nonzero view offsets, multiple heads per sequence, an empty
    // committed history, unequal sequence lengths, and split reduction.
    let (rows,capacity,heads)=(4usize,67usize,2usize);
    let history=(0..(rows+1)*capacity*64).map(|i|(i as f32*0.013).sin()*1.7).collect::<Vec<_>>();
    let values=(0..history.len()).map(|i|(i as f32*0.019).cos()*0.8).collect::<Vec<_>>();
    let history=Tensor::from_vec(history,(rows+1,capacity,64),device)?.narrow(0,1,rows)?;
    let values=Tensor::from_vec(values,(rows+1,capacity,64),device)?.narrow(0,1,rows)?;
    let (k,ks)=quantize(&history)?;let (v,vs)=quantize(&values)?;
    // Narrow the packed/scaled views too, independently of quantization.
    let k=Tensor::cat(&[&k.narrow(0,0,1)?,&k],0)?.narrow(0,1,rows)?;
    let v=Tensor::cat(&[&v.narrow(0,0,1)?,&v],0)?.narrow(0,1,rows)?;
    let ks=Tensor::cat(&[&ks.narrow(0,0,1)?,&ks],0)?.narrow(0,1,rows)?;
    let vs=Tensor::cat(&[&vs.narrow(0,0,1)?,&vs],0)?.narrow(0,1,rows)?;
    let vector=|factor:f32,phase:f32|->Result<Tensor>{Ok(Tensor::from_vec((0..(rows+1)*64).map(|i|(i as f32*factor+phase).sin()).collect::<Vec<_>>(),(rows+1,1,64),device)?.narrow(0,1,rows)?)};
    let q=vector(0.027,0.4)?;let sk=vector(0.031,1.1)?;let sv=vector(0.047,-0.8)?;
    let past=Tensor::from_vec(vec![999u32,0,66],3,device)?.narrow(0,1,2)?;
    let qcpu=q.flatten_all()?.to_vec1::<f32>()?;let skcpu=sk.flatten_all()?.to_vec1::<f32>()?;let svcpu=sv.flatten_all()?.to_vec1::<f32>()?;
    let kc=dequantize(&k,&ks,DType::BF16)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vc=dequantize(&v,&vs,DType::BF16)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut reference=Vec::with_capacity(rows*64);
    for row in 0..rows{let len=if row/heads==0{0}else{66};
        let scores=(0..=len).map(|t|(0..64).map(|d|qcpu[row*64+d]*if t==len{skcpu[row*64+d]}else{kc[(row*capacity+t)*64+d]}).sum::<f32>()*0.125).collect::<Vec<_>>();
        let max=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);let p=scores.iter().map(|x|(x-max).exp()).collect::<Vec<_>>();let z=p.iter().sum::<f32>();
        reference.extend((0..64).map(|d|(0..=len).map(|t|p[t]*if t==len{svcpu[row*64+d]}else{vc[(row*capacity+t)*64+d]}).sum::<f32>()/z));
    }
    for split in [32,128]{let actual=attention(&q,&k,&v,&ks,&vs,&sk,&sv,&past,heads,split)?.flatten_all()?.to_vec1::<f32>()?;
        ensure!(actual.iter().zip(&reference).all(|(a,b)|a.is_finite()&&(a-b).abs()<2e-5),"KV INT8 offset/split attention self-check failed");}
    Ok(())
}

#[cfg(feature="cuda")]
mod cuda {
    use super::*;
    use candle_core::{CpuStorage,CustomOp1,CustomOp2,CustomOp3,Layout,Shape,Storage,cuda_backend::{cudarc,WrapErr}};
    use cudarc::driver::{LaunchConfig,PushKernelArg};
    use std::{collections::HashMap,sync::{Arc,Mutex,OnceLock}};
    fn ptx(device:&candle_core::CudaDevice)->candle_core::Result<(String,Arc<String>)>{
        static CACHE:OnceLock<Mutex<HashMap<String,Arc<String>>>>=OnceLock::new();
        let arch=crate::model::cuda_compat::optional_jit_target(device)?;
        let mut cache=CACHE.get_or_init(||Mutex::new(HashMap::new())).lock().map_err(|_|candle_core::Error::Msg("INT8 PTX cache poisoned".into()))?;
        let code=if let Some(code)=cache.get(&arch){code.clone()}else{
            let options=cudarc::nvrtc::CompileOptions{options:vec![format!("--gpu-architecture={arch}"),"--std=c++14".into()],fmad:Some(false),prec_div:Some(true),..Default::default()};
            let code=Arc::new(cudarc::nvrtc::compile_ptx_with_opts(include_str!("int8_kv.cu"),options).map_err(candle_core::Error::wrap)?.to_src());cache.insert(arch.clone(),code.clone());code};
        Ok((format!("pp_int8_kv_{arch}"),code))
    }
    pub(super) struct Scale{pub rows:usize}
    impl CustomOp1 for Scale{
        fn name(&self)->&'static str{"kv-int8-scale"}
        fn cpu_fwd(&self,_:&CpuStorage,_:&Layout)->candle_core::Result<(CpuStorage,Shape)>{candle_core::bail!("CUDA operation")}
        fn cuda_fwd(&self,x:&candle_core::CudaStorage,l:&Layout)->candle_core::Result<(candle_core::CudaStorage,Shape)>{
            let d=&x.device;let (module,ptx)=ptx(d)?;let fun=d.get_or_load_custom_func("pp_i8_scale",&module,ptx.as_str())?;
            let input=x.as_cuda_slice::<f32>()?.slice(l.start_offset()..);let mut output=unsafe{d.alloc::<f32>(self.rows)}?;
            let n=self.rows as u64;let mut b=fun.builder();b.arg(&input).arg(&mut output).arg(&n);
            unsafe{b.launch(LaunchConfig{grid_dim:(u32::try_from(self.rows).map_err(candle_core::Error::wrap)?,1,1),block_dim:(64,1,1),shared_mem_bytes:0})}.w()?;
            Ok((candle_core::CudaStorage::wrap_cuda_slice(output,d.clone()),l.shape().dims()[..2].into()))
        }
    }
    pub(super) struct Pack{pub n:usize,pub unpack:bool}
    impl CustomOp2 for Pack{
        fn name(&self)->&'static str{"kv-int8-pack"}
        fn cpu_fwd(&self,_:&CpuStorage,_:&Layout,_:&CpuStorage,_:&Layout)->candle_core::Result<(CpuStorage,Shape)>{candle_core::bail!("CUDA operation")}
        fn cuda_fwd(&self,x:&candle_core::CudaStorage,xl:&Layout,s:&candle_core::CudaStorage,sl:&Layout)->candle_core::Result<(candle_core::CudaStorage,Shape)>{
            let d=&x.device;let (module,ptx)=ptx(d)?;let fun=d.get_or_load_custom_func(if self.unpack{"pp_i8_unpack"}else{"pp_i8_pack"},&module,ptx.as_str())?;
            let scale=s.as_cuda_slice::<f32>()?.slice(sl.start_offset()..);let n=self.n as u64;let config=LaunchConfig{grid_dim:(u32::try_from(self.n.div_ceil(256)).map_err(candle_core::Error::wrap)?,1,1),block_dim:(256,1,1),shared_mem_bytes:0};
            if self.unpack{let input=x.as_cuda_slice::<u8>()?.slice(xl.start_offset()..);let mut output=unsafe{d.alloc::<f32>(self.n)}?;let mut b=fun.builder();b.arg(&input).arg(&scale).arg(&mut output).arg(&n);unsafe{b.launch(config)}.w()?;Ok((candle_core::CudaStorage::wrap_cuda_slice(output,d.clone()),xl.shape().clone()))}
            else{let input=x.as_cuda_slice::<f32>()?.slice(xl.start_offset()..);let mut output=unsafe{d.alloc::<u8>(self.n)}?;let mut b=fun.builder();b.arg(&input).arg(&scale).arg(&mut output).arg(&n);unsafe{b.launch(config)}.w()?;Ok((candle_core::CudaStorage::wrap_cuda_slice(output,d.clone()),xl.shape().clone()))}
        }
    }
    pub(super) struct Attention{pub ks:Tensor,pub vs:Tensor,pub sk:Tensor,pub sv:Tensor,pub past:Tensor,pub rows:usize,pub capacity:usize,pub heads:usize,pub split:usize}
    impl CustomOp3 for Attention{
        fn name(&self)->&'static str{"kv-int8-attention"}
        fn cpu_fwd(&self,_:&CpuStorage,_:&Layout,_:&CpuStorage,_:&Layout,_:&CpuStorage,_:&Layout)->candle_core::Result<(CpuStorage,Shape)>{candle_core::bail!("CUDA operation")}
        fn cuda_fwd(&self,q:&candle_core::CudaStorage,ql:&Layout,k:&candle_core::CudaStorage,kl:&Layout,v:&candle_core::CudaStorage,vl:&Layout)->candle_core::Result<(candle_core::CudaStorage,Shape)>{
            let d=&q.device;let (module,ptx)=ptx(d)?;let fun=d.get_or_load_custom_func("pp_i8_attention",&module,ptx.as_str())?;
            let qq=q.as_cuda_slice::<f32>()?.slice(ql.start_offset()..);let kk=k.as_cuda_slice::<u8>()?.slice(kl.start_offset()..);let vv=v.as_cuda_slice::<u8>()?.slice(vl.start_offset()..);
            let (ks,ksl)=self.ks.storage_and_layout();let (vs,vsl)=self.vs.storage_and_layout();let (sk,skl)=self.sk.storage_and_layout();let (sv,svl)=self.sv.storage_and_layout();let (lens,ll)=self.past.storage_and_layout();
            macro_rules! view{($storage:ident,$layout:ident,$ty:ty)=>{match &*$storage{Storage::Cuda(s)=>s.as_cuda_slice::<$ty>()?.slice($layout.start_offset()..),_=>candle_core::bail!("INT8 metadata not CUDA")}};}
            let ks=view!(ks,ksl,f32);let vs=view!(vs,vsl,f32);let sk=view!(sk,skl,f32);let sv=view!(sv,svl,f32);let lens=view!(lens,ll,u32);
            let splits=(self.capacity+1).div_ceil(self.split);let mut partial=unsafe{d.alloc::<f32>(self.rows*splits*66)}?;let mut output=unsafe{d.alloc::<f32>(self.rows*64)}?;
            let heads=self.heads as u32;let capacity=self.capacity as u32;let split=self.split as u32;let ns=splits as u32;
            let mut b=fun.builder();b.arg(&qq).arg(&kk).arg(&vv).arg(&ks).arg(&vs).arg(&sk).arg(&sv).arg(&lens).arg(&mut partial).arg(&heads).arg(&capacity).arg(&split).arg(&ns);
            unsafe{b.launch(LaunchConfig{grid_dim:(self.rows as u32,ns,1),block_dim:(128,1,1),shared_mem_bytes:((self.split+128)*4)as u32})}.w()?;
            let merge=d.get_or_load_custom_func("pp_i8_merge",&module,ptx.as_str())?;let mut b=merge.builder();b.arg(&partial).arg(&mut output).arg(&ns);
            unsafe{b.launch(LaunchConfig{grid_dim:(self.rows as u32,1,1),block_dim:(64,1,1),shared_mem_bytes:0})}.w()?;
            Ok((candle_core::CudaStorage::wrap_cuda_slice(output,d.clone()),(self.rows,1,64).into()))
        }
    }
}
