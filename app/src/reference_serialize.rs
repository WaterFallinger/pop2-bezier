//! Native single-image adaptation of the project's original training serializer.
//! Original candidate search and stage scheduler are kept in the vendored WGSL.
//! The accepted canvas is normalized and measured with the editor's renderer.
mod gpu;
mod raster;
mod pause;
use crate::{document::{Primitive,PrimitiveStageAssignment,ShapeKind,StageTier},history_context::RawRecord,region_state::{self,Stage}};
use image::RgbImage;
use std::{sync::atomic::{AtomicBool,Ordering},time::{Duration,Instant}};
use gpu::{Engine,GpuShape,ImageState};
use raster::Raster;

#[derive(Clone,Debug)]
pub struct SerializeProgress{
    pub primitives:Vec<Primitive>,pub records:Vec<RawRecord>,pub rmse:f64,pub shape_count:usize,
    pub stage_label:String,pub preview:RgbImage,
}
#[derive(Clone,Debug)]
pub struct SerializeResult{
    pub primitives:Vec<Primitive>,pub records:Vec<RawRecord>,pub rmse:f64,pub shape_count:usize,
    pub stage_label:String,pub preview:RgbImage,pub target_rmse:f64,pub stop_reason:String,pub threshold_reached:bool,
}

const MAX_HISTORY:usize=895;
const MAX_ATTEMPTS:u32=2000;
const SOURCE:&str=include_str!("reference_serialize/original.wgsl");
/// Training uses 12.75. Reference preparation may be equally strict or looser.
pub const MIN_STOP_RMSE:f64=12.75;
pub const MAX_STOP_RMSE:f64=255.0;

pub fn validate_stop_rmse(rmse:f64)->Result<(),String>{
    if !rmse.is_finite()||!(MIN_STOP_RMSE..=MAX_STOP_RMSE).contains(&rmse){return Err("截止RMSE须为12.75至255之间的有限数，不能比训练集标准更严格".into());}
    Ok(())
}

fn shader(target_rmse:f64)->String{
    // Parameter values are the actual source main.rs values, not its older README.
    let similarity=1.-target_rmse as f32/255.;
    format!("const CANVAS_WIDTH:u32=512u;\nconst CANVAS_HEIGHT:u32=512u;\nconst SHAPE_ALPHA:f32=0.5;\nconst PENALTY_FACTOR:f32=1.8;\nconst MAX_SEARCH_ATTEMPTS:u32=2000u;\nconst MAX_HISTORY_RECORDS:u32=895u;\nconst GRID_SIDE:u32=4u;\nconst GRID_COUNT:u32=16u;\nconst MAX_ASPECT_RATIO:f32=8.0;\nconst REGION_SIMILARITY_THRESHOLD:f32={similarity:.9};\nconst GRID_MIN_SUCCESSFUL_SHAPES:u32=0u;\nconst GRID_MAX_SUCCESSFUL_SHAPES:u32=20u;\nconst INITIAL_CANDIDATE_COUNT:u32=50u;\nconst SURVIVOR_COUNT:u32=5u;\nconst MUTATION_STAGNATION_LIMIT:u32=100u;\nconst MAX_MUTATIONS_PER_SURVIVOR:u32=10000u;\nconst POSITION_MUTATION_RANGE:f32=56.0;\nconst WIDTH_MUTATION_RANGE:f32=56.0;\nconst HEIGHT_MUTATION_RANGE:f32=56.0;\nconst ANGLE_MUTATION_RANGE_RADIANS:f32=0.27925268;\nconst MIN_HALF_SIZE:f32=1.0;\nconst MAX_HALF_SIZE:f32=256.0;\nconst POSITION_MUTATION_WEIGHT:u32=1u;\nconst WIDTH_MUTATION_WEIGHT:u32=1u;\nconst HEIGHT_MUTATION_WEIGHT:u32=1u;\nconst ANGLE_MUTATION_WEIGHT:u32=1u;\nconst SHAPE_TYPE_MUTATION_WEIGHT:u32=1u;\nconst TOTAL_MUTATION_WEIGHT:u32=5u;\n")+SOURCE
}
fn check(target:&RgbImage,rmse:f64)->Result<(),String>{
    if target.dimensions()!=(512,512){return Err("参考序列化需要512×512的已裁切图像".into());}
    validate_stop_rmse(rmse)
}
fn cancelled(cancel:&AtomicBool)->Result<(),String>{if cancel.load(Ordering::Relaxed){Err("已取消参考序列化".into())}else{Ok(())}}
fn initial(target:&RgbImage)->Primitive{
    let mut sums=[0u64;3];for pixel in target.pixels(){for c in 0..3{sums[c]+=u64::from(pixel[c]);}}
    let mut bg=Primitive::background(sums.map(|v|(v/(512*512))as u8)).normalized().expect("byte colors and finite native background");
    bg.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});bg
}
fn stage(state:&ImageState,raster:&Raster,target:&RgbImage)->String{match state.stage{
    1=>format!("粗网格绘制阶段 · 第{}格 · 本格RMSE {:.3}",state.active_grid+1,raster.region_rmse(target,4,state.active_grid as usize)),
    2=>{let index=state.active_grid.saturating_sub(16)as usize;
        format!("细网格绘制阶段 · 第{}格 · 本格RMSE {:.3}",index+1,raster.region_rmse(target,8,index))},
    _=>"全图绘制阶段".into()}}
fn phase_goal_reached(phase:u32,raster:&Raster,target:&RgbImage,threshold:f64)->bool{
    match phase{
        0=>raster.rmse()<=threshold,
        1=>all_cells_reached(raster,target,4,threshold),
        2=>all_cells_reached(raster,target,8,threshold),
        _=>false,
    }
}
fn all_cells_reached(raster:&Raster,target:&RgbImage,side:usize,threshold:f64)->bool{
    (0..side*side).all(|index|raster.region_rmse(target,side,index)<=threshold)
}
fn global_cutoff_reached(actual_stage:u32,raster:&Raster,threshold:f64)->bool{
    actual_stage==0&&raster.rmse()<=threshold
}
fn coarse_stage_goal_reached(state:&ImageState,actual_stage:u32,raster:&Raster,target:&RgbImage,threshold:f64)->bool{
    actual_stage==1&&state.closed[..16].iter().all(|closed|*closed!=0)
        &&all_cells_reached(raster,target,4,threshold)
}
fn actual_context_state(next:&ImageState,row:&GpuShape)->ImageState{
    let mut actual=*next;
    actual.stage=row.pad1;
    actual.active_grid=match row.pad1{1=>row.pad2,2=>16+row.pad2,_=>0};
    actual
}
fn progress(primitives:&[Primitive],records:&[RawRecord],raster:&Raster,state:&ImageState,target:&RgbImage)->Result<SerializeProgress,String>{
    Ok(SerializeProgress{primitives:primitives.to_vec(),records:pause::resume_records(records)?,rmse:raster.rmse(),shape_count:primitives.len().saturating_sub(1),
        stage_label:stage(state,raster,target),preview:raster.preview()})
}
fn finish(primitives:Vec<Primitive>,records:Vec<RawRecord>,raster:Raster,state:&ImageState,target:&RgbImage,target_rmse:f64,reason:&str)->Result<SerializeResult,String>{
    let rmse=raster.rmse();let reached=phase_goal_reached(state.stage,&raster,target,target_rmse);
    Ok(SerializeResult{shape_count:primitives.len().saturating_sub(1),stage_label:stage(state,&raster,target),
        primitives,records:pause::resume_records(&records)?,rmse,preview:raster.preview(),target_rmse,stop_reason:reason.into(),threshold_reached:reached})
}
fn primitive(row:&GpuShape,order:usize)->Result<Primitive,String>{
    let shape=match row.shape_type{0=>ShapeKind::Rectangle,1=>ShapeKind::Ellipse,_=>return Err("序列化返回了未知的普通图形".into())};
    let (tier,cell)=match row.pad1{0=>(StageTier::Global,0),1 if row.pad2<16=>(StageTier::Coarse,row.pad2 as usize),
        2 if row.pad2<64=>(StageTier::Fine,row.pad2 as usize),_=>return Err("序列化返回的图形阶段或区域无效".into())};
    let rgb=[row.r,row.g,row.b].map(|v|u8::try_from(v).map_err(|_|"序列化返回颜色越界".to_owned()));
    let mut p=Primitive::new(shape,row.cx,row.cy,row.hw*2.,row.hh*2.,[rgb[0].clone()?,rgb[1].clone()?,rgb[2].clone()?]);
    p.angle_deg=row.theta.to_degrees();p.stage_assignment=Some(PrimitiveStageAssignment{stage:tier,cell,queue_order:order as u64});p.normalized()
}
fn marker(row:&GpuShape)->Result<RawRecord,String>{
    let stage=match row.pad1{1=>Stage::Coarse,2=>Stage::Fine,_=>return Err("序列化控制标记缺少有效阶段".into())};
    if row.pad2 as usize>=stage.side()*stage.side(){return Err("序列化控制标记区域编号无效".into());}
    match row.shape_type{2=>region_state::marker_record(stage,row.pad2 as usize),3=>Ok(region_state::END_RECORD),
        4=>Ok(region_state::INTERRUPT_RECORD),_=>Err("序列化返回未知控制标记".into())}
}
fn set_actual_error(state:&mut ImageState,raster:&Raster,target:&RgbImage,target_rmse:f64){
    state.current_sse=raster.sse as f32;
    if state.stage==1||state.stage==2{
        let side=if state.active_grid<16{4}else{8};let index=if side==4{state.active_grid}else{state.active_grid-16}as usize;
        state.region_sse=raster.region_sse(target,side,index);
        if matches!(state.phase,2|3)&&state.is_done==0{
            let quota=if side==4{20}else{10};
            let local=raster.region_rmse(target,side,index);
            state.phase=if state.grid_shape_count>=quota||local<=target_rmse{3}else{2};
        }
    }
}

pub fn serialize(target:&RgbImage,stop_rmse:f64,cancel:&AtomicBool,on_progress:impl FnMut(SerializeProgress))->Result<SerializeResult,String>{
    serialize_inner(target,stop_rmse,cancel,on_progress,None,None,&mut None)
}
fn serialize_inner(target:&RgbImage,stop_rmse:f64,cancel:&AtomicBool,mut on_progress:impl FnMut(SerializeProgress),max_shapes:Option<usize>,max_steps:Option<usize>,adapter_name:&mut Option<String>)->Result<SerializeResult,String>{
    check(target,stop_rmse)?;cancelled(cancel)?;
    let bg=initial(target);let mut raster=Raster::new(bg.color,target);
    let mut records=vec![region_state::BOS_RECORD,bg.to_tokens()?.map(u32::from)];let mut primitives=vec![bg];
    let mut state=gpu::empty_state(raster.sse);
    // A kernel can already schedule the next stage while returning the final
    // row of the previous one. Only emitted rows determine the cutoff scope.
    let mut actual_state=state;
    // Stable image-derived random stream makes changing the cutoff comparable.
    state.rng=target.as_raw().iter().fold(2_166_136_261u32,|h,b|(h^u32::from(*b)).wrapping_mul(16_777_619)).max(1);
    on_progress(progress(&primitives,&records,&raster,&actual_state,target)?);
    if global_cutoff_reached(actual_state.stage,&raster,stop_rmse){return finish(primitives,records,raster,&actual_state,target,stop_rmse,"全图绘制阶段已达到截止RMSE");}
    let packed_target=target.pixels().map(|p|(u32::from(p[0])<<16)|(u32::from(p[1])<<8)|u32::from(p[2])).collect::<Vec<_>>();
    let pixels=raster.packed();let engine=pollster::block_on(Engine::new(&shader(stop_rmse),&packed_target,pixels[0],&state))?;
    *adapter_name=Some(engine.name.clone());
    crate::diagnostics::event("reference_serializer_ready",serde_json::json!({"adapter":engine.name,"backend":"DX12/Vulkan original search","target_rmse":stop_rmse,"canvas":512}));
    cancelled(cancel)?;let mut reported=Instant::now();
    for _ in 0..max_steps.unwrap_or(MAX_HISTORY+MAX_ATTEMPTS as usize+8){
        cancelled(cancel)?;let before=state;let mut next=engine.step()?;
        if next.stage>2||next.phase>4||next.is_done>1||!next.current_sse.is_finite()
            ||next.stage==1&&next.active_grid>=16||next.stage==2&&!(16..80).contains(&next.active_grid){
            return Err("序列化GPU返回了无效的阶段或误差状态，结果没有导入画布".into());
        }
        if next.history_count<before.history_count||next.history_count>before.history_count+1||next.history_count as usize>MAX_HISTORY{
            return Err("序列化GPU返回的历史计数异常，结果没有导入画布".into());
        }
        if next.history_count>before.history_count{
            let row:GpuShape=engine.read(&engine.history,u64::from(before.history_count)*std::mem::size_of::<GpuShape>()as u64)?;
            if row.shape_type<=1{
                let p=primitive(&row,primitives.len())?;let mut candidate=raster.clone();candidate.apply(&p,target);
                if candidate.sse+0.000001<raster.sse{
                    raster=candidate;records.push(p.to_tokens()?.map(u32::from));primitives.push(p);
                    set_actual_error(&mut next,&raster,target,stop_rmse);
                    actual_state=actual_context_state(&next,&row);
                }else{
                    // A floating search improvement may disappear after the
                    // actual model lattice/AA. Retain only the search RNG and
                    // attempt count; do not pretend a non-improving shape exists.
                    let attempts=next.search_attempt_count;let rng=next.rng;next=before;next.search_attempt_count=attempts;next.rng=rng;
                    set_actual_error(&mut next,&raster,target,stop_rmse);
                }
                engine.update(&next,Some(&raster.packed()));
            }else{records.push(marker(&row)?);set_actual_error(&mut next,&raster,target,stop_rmse);engine.update(&next,None);
                actual_state=actual_context_state(&next,&row);}
            region_state::DrawingRegion::from_history(&records)?;
        }
        state=next;
        if reported.elapsed()>=Duration::from_millis(200){on_progress(progress(&primitives,&records,&raster,&actual_state,target)?);reported=Instant::now();}
        if global_cutoff_reached(actual_state.stage,&raster,stop_rmse){return finish(primitives,records,raster,&actual_state,target,stop_rmse,"全图绘制阶段已达到截止RMSE");}
        if coarse_stage_goal_reached(&state,actual_state.stage,&raster,target,stop_rmse){
            return finish(primitives,records,raster,&actual_state,target,stop_rmse,"全部粗网格均已达到截止RMSE，停止于粗网格阶段");
        }
        if max_shapes.is_some_and(|limit|primitives.len().saturating_sub(1)>=limit){return finish(primitives,records,raster,&actual_state,target,stop_rmse,"测试步数上限");}
        if state.is_done!=0{
            let reached=phase_goal_reached(actual_state.stage,&raster,target,stop_rmse);
            let reason=match state.termination_reason{
                1 if reached=>"全部细网格均已达到截止RMSE",
                1=>"细网格绘制已完成，部分细网格尚未达到截止RMSE",
                2=>"已达到896条参考记录上限，保留当前绘制前缀",
                3=>"候选搜索已达到安全上限，保留当前绘制前缀",
                _=>return Err("序列化GPU返回未知停止状态".into()),
            };
            return finish(primitives,records,raster,&actual_state,target,stop_rmse,reason);
        }
    }
    if max_steps.is_some(){finish(primitives,records,raster,&actual_state,target,stop_rmse,"内部验收步数上限")}
    else{Err("图像序列化超过逻辑步数上限，结果没有导入画布".into())}
}

/// Seed only the scheduler state so the existing GPU kernel's grid stop rules
/// can be tested without drawing 85 unrelated global primitives first.
fn acceptance_grid_thresholds()->Result<Vec<String>,String>{
    use image::Rgb;
    let target=RgbImage::from_fn(512,512,|x,y|if x<128&&y<128{Rgb([40;3])}else{Rgb([0;3])});
    let raster=Raster::new([0;3],&target);let packed=raster.packed();
    let packed_target=target.pixels().map(|p|(u32::from(p[0])<<16)|(u32::from(p[1])<<8)|u32::from(p[2])).collect::<Vec<_>>();
    let threshold=20.;
    let engine=pollster::block_on(Engine::new(&shader(threshold),&packed_target,0,&gpu::empty_state(raster.sse)))?;
    let mut checks=Vec::new();
    for side in [4usize,8]{
        let encoded=if side==4{0}else{16};
        let mut seed=gpu::empty_state(raster.sse);seed.stage=if side==4{1}else{2};
        seed.active_grid=encoded;seed.phase=2;seed.first_phase_shape_count=85;
        seed.region_sse=raster.region_sse(&target,side,0);
        if raster.rmse()>threshold||raster.region_rmse(&target,side,0)<=threshold{return Err("阶段阈值GPU验收的局部误差构造无效".into());}
        engine.update(&seed,Some(&packed));let outcome=engine.step()?;
        if outcome.closed[encoded as usize]!=0{return Err("整图误差低于阈值时，GPU仍须保留未达标的当前网格".into());}
        if outcome.history_count>0{
            let row:GpuShape=engine.read(&engine.history,0)?;
            if row.shape_type==3{return Err("GPU错误地使用整图RMSE结束了未达标网格".into());}
        }
        checks.push(format!("{side}x{side}_bad_local_cell_does_not_end_from_low_global_rmse"));
        let mut good_pixels=packed.clone();let width=512/side;
        for y in 0..width{for x in 0..width{good_pixels[y*512+x]=(40<<16)|(40<<8)|40;}}
        let mut good=seed;good.region_sse=0.;good.grid_shape_count=1;good.counts[encoded as usize]=1;
        good.current_sse=good_pixels.iter().zip(target.pixels()).map(|(&p,t)|{
            let p=[(p>>16)&255,(p>>8)&255,p&255];(0..3).map(|c|(p[c]as f64-t[c]as f64).powi(2)).sum::<f64>()
        }).sum::<f64>()as f32;
        engine.update(&good,Some(&good_pixels));let outcome=engine.step()?;
        let row:GpuShape=engine.read(&engine.history,0)?;
        if outcome.closed[encoded as usize]!=1||row.shape_type!=3{return Err("当前网格达到局部阈值后未结束本次编码访问".into());}
        let ordinary=Primitive::new(ShapeKind::Rectangle,20.,20.,8.,8.,[40;3]).to_tokens()?.map(u32::from);
        let regional_stage=if side==4{Stage::Coarse}else{Stage::Fine};
        let logical=vec![region_state::BOS_RECORD,region_state::marker_record(regional_stage,0)?,ordinary,marker(&row)?];
        let resumable=pause::resume_records(&logical)?;
        if resumable.last()!=Some(&region_state::INTERRUPT_RECORD)
            ||resumable.contains(&region_state::END_RECORD){return Err("RMSE达标的参考网格没有保留为手动暂停状态".into());}
        checks.push(format!("{side}x{side}_local_goal_emits_manual_pause"));
        let mut capped=seed;capped.grid_shape_count=if side==4{20}else{10};capped.counts[encoded as usize]=capped.grid_shape_count;
        engine.update(&capped,Some(&packed));let outcome=engine.step()?;
        let row:GpuShape=engine.read(&engine.history,0)?;
        if outcome.closed[encoded as usize]!=1||row.shape_type!=3{return Err("网格图元上限的原有终止条件失效".into());}
        let mut logical=vec![region_state::BOS_RECORD,region_state::marker_record(regional_stage,0)?];
        logical.extend(std::iter::repeat_n(ordinary,capped.grid_shape_count as usize));logical.push(marker(&row)?);
        if pause::resume_records(&logical)?.last()!=Some(&region_state::END_RECORD){return Err("图元配额正常结束的参考网格不应转换成暂停".into());}
        checks.push(format!("{side}x{side}_original_quota_still_emits_end"));
    }
    Ok(checks)
}

/// Explicit developer acceptance only; does not create/open any user document.
pub fn run_acceptance(report:&std::path::Path)->Result<(),String>{
    use image::Rgb;
    let started=Instant::now();
    let result=(||{
        let target=RgbImage::from_fn(512,512,|x,y|if x<256{Rgb([220,(40+y/16)as u8,80])}else{Rgb([40,80,(160+y/8)as u8])});
        let baseline=Raster::new(initial(&target).color,&target).rmse();
        let mut adapter=None;let result=serialize_inner(&target,MIN_STOP_RMSE,&AtomicBool::new(false),|_|{},Some(3),Some(32),&mut adapter)?;
        if result.shape_count==0||result.rmse>=baseline{return Err("原GPU算法没有产生能改善实际RMSE的图元".into());}
        if result.records.len()>12{return Err("GPU验收参考记录超出本次小样本预算".into());}
        let native=crate::canvas::render_reference_preview(&result.primitives);
        let equal=native.pixels.iter().zip(result.preview.pixels()).all(|(p,q)|[p.r(),p.g(),p.b()]==q.0);
        if !equal{return Err("参考预览与实际导入画布的抗锯齿渲染不一致".into());}
        let exact=result.preview.pixels().zip(target.pixels()).map(|(p,t)|(0..3).map(|c|(p[c]as f64-t[c]as f64).powi(2)).sum::<f64>()).sum::<f64>();
        let exact=(exact/(512.*512.*3.)).sqrt();
        if (exact-result.rmse).abs()>0.00001{return Err("序列化RMSE没有依据实际规范化输出计算".into());}
        let constant=RgbImage::from_pixel(512,512,Rgb([40,80,120]));
        let constant=serialize(&constant,MIN_STOP_RMSE,&AtomicBool::new(false),|_|{})?;
        if !constant.threshold_reached||constant.shape_count!=0{return Err("纯色背景的提前达标路径异常".into());}
        if serialize(&target,MIN_STOP_RMSE,&AtomicBool::new(true),|_|{}).is_ok(){return Err("预先取消未阻止序列化".into());}
        let phase_checks=acceptance_grid_thresholds()?;
        Ok(serde_json::json!({"status":"passed","scope":"original WGSL GPU search; no trained model loaded or canvas import",
            "adapter":adapter,"initial_rmse":baseline,"final_rmse":result.rmse,"actual_native_rmse":exact,"shape_count":result.shape_count,
            "record_count":result.records.len(),"preview_matches_native_antialias":equal,"constant_background_early_stop":true,
            "precancel_blocks_job":true,"threshold_reached":result.threshold_reached,"stop_reason":result.stop_reason,
            "phase_scoped_gpu_checks":phase_checks,
            "elapsed_seconds":started.elapsed().as_secs_f64()}))
    })();
    let value=match &result{Ok(value)=>value.clone(),Err(error)=>serde_json::json!({"status":"failed","error":error,"elapsed_seconds":started.elapsed().as_secs_f64()})};
    if let Some(parent)=report.parent(){std::fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
    std::fs::write(report,serde_json::to_vec_pretty(&value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    result.map(|_|())
}

#[cfg(test)]mod tests{
    use super::*;use image::Rgb;
    #[test]fn original_gpu_struct_layout_matches_wgsl_storage(){
        assert_eq!(std::mem::size_of::<GpuShape>(),48);assert_eq!(std::mem::size_of::<ImageState>(),1024);
    }
    #[test]fn constant_target_reaches_cutoff_with_only_a_real_native_background(){
        let image=RgbImage::from_pixel(512,512,Rgb([40,80,120]));let mut events=0;
        let result=serialize(&image,MIN_STOP_RMSE,&AtomicBool::new(false),|_|events+=1).unwrap();
        assert_eq!(events,1);assert!(result.threshold_reached);assert_eq!(result.rmse,0.);
        assert_eq!(result.primitives.len(),1);assert_eq!(result.shape_count,0);assert_eq!(result.records.len(),2);
        assert_eq!(result.preview,image);assert!(!result.records.contains(&region_state::END_RECORD));
    }
    #[test]fn invalid_inputs_and_precancel_never_initialize_gpu_or_emit_results(){
        let image=RgbImage::from_pixel(512,512,Rgb([40,80,120]));
        for rmse in [f64::NAN,-1.,0.,5.,12.749999,256.,f64::INFINITY]{assert!(serialize(&image,rmse,&AtomicBool::new(false),|_|panic!()).is_err());}
        for rmse in [12.75,20.,255.]{assert!(validate_stop_rmse(rmse).is_ok());}
        assert!(serialize(&RgbImage::new(10,10),MIN_STOP_RMSE,&AtomicBool::new(false),|_|panic!()).is_err());
        assert!(serialize(&image,MIN_STOP_RMSE,&AtomicBool::new(true),|_|panic!()).unwrap_err().contains("取消"));
    }
    #[test]fn grid_phase_never_stops_from_a_good_global_average_with_a_bad_local_cell(){
        for side in [4usize,8]{
            let width=512/side;
            let target=RgbImage::from_fn(512,512,|x,y|if x<width as u32&&y<width as u32{Rgb([40;3])}else{Rgb([0;3])});
            let raster=Raster::new([0;3],&target);let phase=if side==4{1}else{2};
            assert!(raster.rmse()<20.);assert_eq!(raster.region_rmse(&target,side,0),40.);
            assert!(global_cutoff_reached(0,&raster,20.));assert!(!global_cutoff_reached(phase,&raster,20.));
            assert!(!phase_goal_reached(phase,&raster,&target,20.));
            let mut state=gpu::empty_state(raster.sse);state.stage=phase;state.active_grid=if side==4{0}else{16};state.phase=3;
            set_actual_error(&mut state,&raster,&target,20.);assert_eq!(state.phase,2,"a good global average cannot close a locally unfinished cell");
            state.grid_shape_count=if side==4{20}else{10};set_actual_error(&mut state,&raster,&target,20.);assert_eq!(state.phase,3,"the original per-cell quota still ends a region");
            assert!(!phase_goal_reached(phase,&raster,&target,20.),"quota completion must not falsely report the user's quality goal");
        }
    }
    #[test]fn coarse_boundary_checks_every_current_cell_and_stops_before_fine_begin(){
        let target=RgbImage::from_pixel(512,512,Rgb([0;3]));let mut raster=Raster::new([0;3],&target);
        let mut state=gpu::empty_state(0.);state.stage=2;state.active_grid=16;state.phase=1;
        state.closed[..16].fill(1);
        let row=GpuShape{shape_type:3,pad1:1,pad2:15,..bytemuck::Zeroable::zeroed()};
        let actual=actual_context_state(&state,&row);
        assert_eq!(actual.stage,1);assert_eq!(actual.active_grid,15);
        assert!(coarse_stage_goal_reached(&state,actual.stage,&raster,&target,20.));
        let rows=vec![region_state::BOS_RECORD,region_state::marker_record(Stage::Coarse,15).unwrap(),region_state::END_RECORD];
        let result=finish(Vec::new(),rows.clone(),raster.clone(),&actual,&target,20.,"粗网格达标").unwrap();
        assert!(result.threshold_reached);assert!(result.stage_label.starts_with("粗网格"));
        assert_eq!(result.records,vec![region_state::BOS_RECORD],"an empty RMSE visit is unvisited rather than a fake BEGIN + pause pair");
        assert!(!result.records.iter().any(|row|region_state::marker_identity(row).is_ok_and(|(stage,_)|stage==Stage::Fine)));
        // A later cross-border drawing can damage a cell that already ended.
        // The boundary check must inspect the final pixels, not old close-time errors.
        raster.apply(&Primitive::new(ShapeKind::Rectangle,64.,64.,128.,128.,[80;3]),&target);
        assert!(raster.rmse()<20.);assert!(raster.region_rmse(&target,4,0)>20.);
        assert!(!coarse_stage_goal_reached(&state,actual.stage,&raster,&target,20.));
        state.closed[15]=0;let clean=Raster::new([0;3],&target);
        assert!(!coarse_stage_goal_reached(&state,1,&clean,&target,20.),"coarse cells must legally close before the whole coarse stage ends");
    }
    #[test]fn the_last_global_primitive_keeps_global_cutoff_even_when_gpu_has_scheduled_coarse(){
        let target=RgbImage::from_pixel(512,512,Rgb([0;3]));let raster=Raster::new([0;3],&target);
        let mut scheduled=gpu::empty_state(0.);scheduled.stage=1;scheduled.active_grid=7;scheduled.phase=1;
        let row=GpuShape{shape_type:0,pad1:0,pad2:0,..bytemuck::Zeroable::zeroed()};
        let actual=actual_context_state(&scheduled,&row);
        assert_eq!(actual.stage,0);assert!(global_cutoff_reached(actual.stage,&raster,MIN_STOP_RMSE));
        let result=finish(Vec::new(),vec![region_state::BOS_RECORD],raster,&actual,&target,MIN_STOP_RMSE,"全图达标").unwrap();
        assert!(result.threshold_reached);assert_eq!(result.stage_label,"全图绘制阶段");
    }
    #[test]fn finished_reference_keeps_fine_geometry_while_the_model_prefix_resumes_paused_coarse(){
        let mut bg=Primitive::background([0;3]);bg.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});
        let mut coarse=Primitive::new(ShapeKind::Rectangle,64.,64.,128.,128.,[80;3]);
        coarse.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Coarse,cell:0,queue_order:1});
        let mut fine=Primitive::new(ShapeKind::Ellipse,32.,32.,50.,50.,[160;3]);
        fine.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Fine,cell:0,queue_order:2});
        let all=vec![bg,coarse,fine];let pixels=crate::canvas::render_reference_preview(&all).pixels.into_iter()
            .flat_map(|p|[p.r(),p.g(),p.b()]).collect::<Vec<_>>();
        let target=RgbImage::from_raw(512,512,pixels).unwrap();let mut raster=Raster::new(all[0].color,&target);
        for p in &all[1..]{raster.apply(p,&target);}
        let raw=vec![region_state::BOS_RECORD,all[0].to_tokens().unwrap().map(u32::from),
            region_state::marker_record(Stage::Coarse,0).unwrap(),all[1].to_tokens().unwrap().map(u32::from),region_state::END_RECORD,
            region_state::marker_record(Stage::Fine,0).unwrap(),all[2].to_tokens().unwrap().map(u32::from),region_state::END_RECORD];
        let mut state=gpu::empty_state(raster.sse);state.stage=2;state.active_grid=16;
        let result=finish(all.clone(),raw,raster,&state,&target,MIN_STOP_RMSE,"参考拟合已达标").unwrap();
        assert_eq!(result.primitives,all);assert_eq!(result.preview,target);assert!(result.threshold_reached);
        assert_eq!(result.records.len(),5);assert_eq!(result.records.last(),Some(&region_state::INTERRUPT_RECORD));
        assert!(!result.records.iter().any(|row|region_state::marker_identity(row).is_ok_and(|(s,_)|s==Stage::Fine)));
        let parsed=region_state::DrawingRegion::from_history(&result.records).unwrap();
        assert!(!parsed.completed);assert!(parsed.progress.coarse_completed.is_empty());
    }
    #[test]fn fine_success_and_local_metrics_use_exact_pixels_not_f32_or_whole_image_average(){
        let mut target=RgbImage::from_pixel(512,512,Rgb([0;3]));
        for y in 0..64{for x in 0..64{target.put_pixel(x,y,Rgb([40;3]));}}
        target.put_pixel(0,0,Rgb([41,40,40]));let raster=Raster::new([0;3],&target);
        let local=raster.region_rmse(&target,8,0);
        assert!(local>40.);assert!(raster.rmse()<20.);
        assert!(!phase_goal_reached(2,&raster,&target,local-1e-8));
        assert!(phase_goal_reached(2,&raster,&target,local));
        assert!(!global_cutoff_reached(2,&raster,local));
    }
    #[test]fn shape_conversion_uses_the_actual_model_lattice_and_keeps_owned_region(){
        let row=GpuShape{cx:39.9,cy:149.9,hw:10.11,hh:20.12,theta:120f32.to_radians(),shape_type:0,r:101,g:151,b:201,delta_err:-1.,pad1:1,pad2:4};
        let p=primitive(&row,8).unwrap();assert_eq!(p.x,39.5);assert_eq!(p.y,149.5);assert_eq!(p.angle_deg,30.);
        assert_eq!(p.width,40.);assert_eq!(p.height,20.);assert_eq!(p.color,[100,150,200]);assert_eq!(p.stage_assignment.unwrap().cell,4);
        assert_eq!(p.normalized().unwrap(),p);
    }
    #[test]#[ignore="requires a local graphics adapter; original GPU search smoke"]
    fn original_gpu_search_reduces_actual_quantized_rmse_without_importing(){
        let image=RgbImage::from_fn(512,512,|x,_|if x<256{Rgb([220,60,80])}else{Rgb([40,80,200])});
        let initial=Raster::new(initial(&image).color,&image).rmse();
        let result=serialize_inner(&image,MIN_STOP_RMSE,&AtomicBool::new(false),|_|{},Some(1),Some(32),&mut None).unwrap();
        assert_eq!(result.shape_count,1);assert!(result.rmse<initial);assert!(!result.threshold_reached);
        assert_eq!(result.records.len(),3);assert_eq!(result.stop_reason,"测试步数上限");
        assert!(result.primitives.iter().all(|p|p.normalized().unwrap()==*p));
    }
}
