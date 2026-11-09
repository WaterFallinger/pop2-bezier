#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
//! Small production-AtenEngine scheduler acceptance; no UI and no throughput test.
#[path="../../src/acceleration.rs"] mod acceleration;
#[path="../../src/amd_capability.rs"] mod amd_capability;
#[path="../../src/aten_weight_index.rs"] mod aten_weight_index;
#[path="../../src/diagnostics.rs"] mod diagnostics;
#[path="../../src/inference.rs"] mod inference;
#[path="../../src/model/mod.rs"] mod model;
#[path="../../src/region_state.rs"] mod region_state;

use anyhow::{Context,Result,ensure};
use std::{path::PathBuf,sync::atomic::AtomicBool};
use inference::{EngineConfig,LoadPreparationSettings,GenerationRequest,InferenceSettings,InferencePrimitive};
use acceleration::{AccelerationMode,AccelerationSettings,FeaturePreference};

fn ordinary()->[u32;9]{[512,1536,2048,2518,4566,6415,6702,6830,6958]}
fn fixture(dense:bool)->Vec<[u32;9]>{
    let mut background=ordinary();background[..5].copy_from_slice(&[0,1024,2048,2318,4366]);background[5]=model::OFFSETS[5];
    let mut raw=vec![model::BOS_RECORD,background];raw.extend(std::iter::repeat_n(ordinary(),85));
    for(side,count)in[(4,16),(8,61)]{for index in 0..count{
        let marker=model::region_marker(side,index);let mut shape=ordinary();shape[..2].copy_from_slice(&marker[..2]);
        raw.push(marker);raw.extend(std::iter::repeat_n(shape,if dense{if side==4{20}else{10}}else{1}));raw.push(model::END_RECORD);
    }}
    let marker=model::region_marker(8,61);let mut shape=ordinary();shape[..2].copy_from_slice(&marker[..2]);
    raw.push(marker);raw.extend(std::iter::repeat_n(shape,9));raw
}
fn main()->Result<()>{
    let args:Vec<_>=std::env::args().collect();
    let config=EngineConfig{model_path:PathBuf::from(args.get(1).context("checkpoint argument missing")?),
        tokenizer_path:PathBuf::from(args.get(2).context("tokenizer argument missing")?),use_cuda:false};
    let output=PathBuf::from(args.get(3).context("output argument missing")?);
    let lanes=args.get(4).map(|value|value.parse()).transpose()?.unwrap_or(1);
    let dense=args.get(5).is_some_and(|value|value=="dense");
    let acceleration=AccelerationSettings{mode:AccelerationMode::Optimized,primitive_graph:FeaturePreference::Disabled,
        kv_int8:FeaturePreference::Enabled,..Default::default()};
    let preparation=LoadPreparationSettings{lanes,cfg_scale:2.,renyi_alpha:4.,sampling_override:Some(true),acceleration};
    let mut engine=inference::aten_engine::AtenEngine::load_configured(&config,&preparation,|_|{})?;
    engine.prepare_after_load(&preparation,|_|{})?;
    let raw=fixture(dense);region_state::DrawingRegion::from_history(&raw).map_err(anyhow::Error::msg)?;
    let request=GenerationRequest{prompt:"金发蓝眼的白皙圆脸女性以明亮动画风呈现，张嘴露舌，粉色背景可见，取景包含头部和肩颈。".into(),
        settings:InferenceSettings{cfg_scale:2.,renyi_alpha:4.,seed:20261104,max_primitives:if dense{12}else{3},context_overclock:true,half_region_switch:false,pause_at_stage_end:false},
        primitives:raw.iter().copied().filter(|record|model::validate_visible(record).is_ok()).map(|tokens|InferencePrimitive{tokens}).collect(),
        manual_review_regions: Vec::new(), extra_fine_context:Vec::new(),stage_floor:None,pending_stage_context:Vec::new(), history_records:Some(raw.clone())};
    let mut delivered=0;
    let summary=engine.generate_parallel(&request,lanes,&AtomicBool::new(false),|_,primitive|{
        assert!(primitive.history_records.starts_with(&raw));delivered+=1;
    },|_|{},|_,_|{},|_,_|{})?;
    for lane in &summary.lane_summaries{ensure!(lane.history_records.starts_with(&raw),"chronological history changed");}
    let actual=engine.diagnostic_info();let stats=&actual["context_window"];
    if dense{ensure!(stats["reorders"].as_u64().unwrap_or(0)>=1,"overfull history did not reproject");}
    else{ensure!(stats["reorders"].as_u64()==Some(0),"fitting fine boundaries reordered");}
    if !dense{ensure!(stats["append_boundary_records"].as_u64().unwrap_or(0)>0,"no boundary was appended");}
    ensure!(stats["reused_prefix_tokens"].as_u64().unwrap_or(0)>0,"no common prefix reused");
    let backend=actual["backend_diagnostics"].as_object().context("missing native backend diagnostics")?;
    ensure!(backend["full_prefill_calls"].as_u64()==Some(2),"fine boundaries triggered full conditional/unconditional prefill");
    let result=serde_json::json!({"status":"passed","production_aten_scheduler":true,"dense_overflow":dense,"lanes":lanes,"generated":delivered,
        "raw_prefix_records":raw.len(),"chronology_preserved":true,"context_window":stats,"actual":actual});
    std::fs::write(output,serde_json::to_vec_pretty(&result)?)?;println!("{}",result);Ok(())
}
