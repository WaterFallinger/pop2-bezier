//! Development-only GPU acceptance, including the exact production cache code.
use anyhow::{ensure, Result};
use candle_core::{Device, DType, Tensor};
use serde_json::{json, Value};

mod cache_model {
    use crate::model::LayerCache;
    use super::*;
    const HEADS: usize = 2;

    fn data(device: &Device, length: usize, bias: f32) -> Result<Tensor> {
        Ok(Tensor::from_vec((0..HEADS*length*64).map(|i|bias+(i as f32*0.173).sin()).collect::<Vec<_>>(),
            (HEADS,length,64),device)?.to_dtype(DType::BF16)?)
    }
    fn initialize(device:&Device,int8:bool,length:usize,bias:f32)->Result<LayerCache>{
        let mut cache=LayerCache::default();let tensor=data(device,length,bias)?;
        if int8{cache.query_int8(&tensor,&tensor,0,true)?;}else{cache.query(&tensor,&tensor,0,true)?;}
        Ok(cache)
    }
    fn tensor_bytes(tensor:&Tensor)->Result<Vec<u8>>{
        let flat=tensor.contiguous()?.flatten_all()?;
        Ok(if tensor.dtype()==DType::U8{flat.to_vec1::<u8>()?}else{
            flat.to_dtype(DType::F32)?.to_vec1::<f32>()?.into_iter().flat_map(f32::to_le_bytes).collect()
        })
    }
    fn read(cache:&mut LayerCache,device:&Device,int8:bool)->Result<Vec<u8>>{
        if int8{
            let scratch=data(device,1,0.)?;let length=cache.len();
            let r=cache.query_int8(&scratch,&scratch,length,false)?;
            let mut result=tensor_bytes(&r.key.narrow(1,0,length)?)?;
            result.extend(tensor_bytes(&r.value.narrow(1,0,length)?)?);
            result.extend(tensor_bytes(&r.key_scale.narrow(1,0,length)?)?);
            result.extend(tensor_bytes(&r.value_scale.narrow(1,0,length)?)?);Ok(result)
        }else{
            let mut result=tensor_bytes(&cache.key()?.unwrap())?;result.extend(tensor_bytes(&cache.value()?.unwrap())?);Ok(result)
        }
    }
    fn batch(caches:&mut [LayerCache],device:&Device,int8:bool,past:&[usize])->Result<candle_core::TensorId>{
        let scratch=Tensor::stack(&(0..caches.len()).map(|row|data(device,1,20.+row as f32)).collect::<Result<Vec<_>>>()?,0)?;
        let mut refs=caches.iter_mut().collect::<Vec<_>>();
        if int8{return Ok(LayerCache::query_int8_batch(&mut refs,&scratch,&scratch,past,false,None)?.key.id());}
        LayerCache::query_batch(&mut refs,&scratch,&scratch,past,false)?;drop(refs);
        Ok(caches[0].batch_arena().unwrap().0.id())
    }
    pub fn run(device:&Device)->Result<Value>{
        let mut modes=Vec::new();
        for int8 in [false,true]{
            let mut caches=(0..3).map(|row|initialize(device,int8,[2,3,1][row],row as f32)).collect::<Result<Vec<_>>>()?;
            let past=[2,3,1];let arena=batch(&mut caches,device,int8,&past)?;
            let row0=read(&mut caches[0],device,int8)?;
            let row2=read(&mut caches[2],device,int8)?;
            caches[1]=initialize(device,int8,3,7.)?;
            let same=batch(&mut caches,device,int8,&past)?;
            ensure!(same==arena,"changed row unnecessarily replaced the entire batch arena");
            ensure!(read(&mut caches[0],device,int8)?==row0&&read(&mut caches[2],device,int8)?==row2,"unchanged batch rows were corrupted");
            let row1=read(&mut caches[1],device,int8)?;
            let mut snapshot=caches[1].clone();
            caches[1]=initialize(device,int8,3,9.)?;
            let detached=batch(&mut caches,device,int8,&past)?;
            ensure!(detached!=same,"a retained old slot snapshot must prevent in-place slot reuse");
            ensure!(read(&mut snapshot,device,int8)?==row1,"old slot snapshot changed during replacement");
            let old_prefix=read(&mut caches[0],device,int8)?;
            let mut old_snapshot=caches[0].clone();
            let bytes=caches[0].reserved_bytes();
            caches[0].validate_truncate_committed(1)?;caches[0].truncate_committed(1)?;
            let replacement=data(device,1,11.)?;
            if int8{caches[0].query_int8(&replacement,&replacement,1,true)?;}else{caches[0].query(&replacement,&replacement,1,true)?;}
            ensure!(read(&mut old_snapshot,device,int8)?==old_prefix,"truncate then suffix rewrite corrupted the retained branch");
            ensure!(caches[0].len()==2&&old_snapshot.len()==2,"truncate/append logical length mismatch");
            caches[0].truncate_committed(0)?;
            ensure!(caches[0].reserved_bytes()==bytes,"truncate zero discarded reserved capacity");
            modes.push(json!({"int8":int8,"same_arena_after_single_row_rebuild":true,
                "unchanged_rows_preserved":true,"retained_slot_snapshot_forces_detach":true,
                "truncate_append_preserves_snapshot":true,"truncate_zero_keeps_capacity":true}));
        }
        device.synchronize()?;
        Ok(json!({"status":"passed","scope":"actual CUDA cache storage, no model weights, tiny buffers","modes":modes}))
    }
}

pub fn run(device:&Device)->Result<Value>{cache_model::run(device)}
