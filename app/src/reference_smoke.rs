//! Isolated native controller acceptance; worker messages are synthetic, while
//! document adoption, stage/ID reconciliation, persistence and UI are real.
use super::*;
use crate::{document::{PrimitiveStageAssignment,StageTier,ShapeKind},region_state};

fn require(checks:&mut Vec<String>,name:&str,condition:bool)->Result<(),String>{
    if !condition{return Err(format!("参考导入验收失败：{name}"));}checks.push(name.into());Ok(())
}
fn fixture(rmse:f64)->SerializeResult{
    let mut bg=Primitive::background([240,244,250]);bg.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});
    let mut shape=Primitive::new(ShapeKind::Ellipse,256.,256.,200.,240.,[80,140,200]);
    shape.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:1});
    let primitives=vec![bg,shape];let mut records=vec![region_state::BOS_RECORD];
    records.extend(primitives.iter().map(|p|p.to_tokens().unwrap().map(u32::from)));
    let pixels=crate::canvas::render_reference_preview(&primitives).pixels.into_iter().flat_map(|c|[c.r(),c.g(),c.b()]).collect::<Vec<_>>();
    SerializeResult{primitives,records,rmse:6.,shape_count:1,stage_label:"全图绘制阶段".into(),
        preview:image::RgbImage::from_raw(512,512,pixels).unwrap(),target_rmse:rmse,
        stop_reason:"原生控制器合成验收".into(),threshold_reached:6.<=rmse}
}
fn mixed_reference_fixture()->SerializeResult{
    let mut primitives=Vec::new();let mut records=vec![region_state::BOS_RECORD];
    let mut bg=Primitive::background([238,242,248]);
    bg.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});
    records.push(bg.to_tokens().unwrap().map(u32::from));primitives.push(bg);
    for n in 0..85{
        let mut p=Primitive::new(ShapeKind::Ellipse,32.+(n%10)as f32*40.,32.+(n/10)as f32*40.,8.,10.,[120,160,200]);
        p.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:n+1});
        records.push(p.to_tokens().unwrap().map(u32::from));primitives.push(p);
    }
    for (tier,cell,count)in[(StageTier::Coarse,0,20),(StageTier::Coarse,1,1),(StageTier::Fine,0,10),(StageTier::Fine,1,1)]{
        let side=tier.side();let step=512./side as f32;
        if side==4{records.push(region_state::marker_record(region_state::Stage::Coarse,cell).unwrap());}
        for _ in 0..count{
            let mut p=Primitive::new(ShapeKind::Rectangle,(cell%side)as f32*step+step/2.,(cell/side)as f32*step+step/2.,8.,8.,[80,140,180]);
            p.stage_assignment=Some(PrimitiveStageAssignment{stage:tier,cell,queue_order:primitives.len()as u64});
            if side==4{records.push(p.to_tokens().unwrap().map(u32::from));}primitives.push(p);
        }
        if side==4{records.push(if count==20{region_state::END_RECORD}else{region_state::INTERRUPT_RECORD});}
    }
    let image=crate::canvas::render_reference_preview(&primitives);
    let pixels=image.pixels.into_iter().flat_map(|c|[c.r(),c.g(),c.b()]).collect();
    SerializeResult{shape_count:primitives.len()-1,primitives,records,rmse:18.,stage_label:"细阶段参考已保留，模型从暂停的粗阶段继续".into(),
        preview:image::RgbImage::from_raw(512,512,pixels).unwrap(),target_rmse:20.,
        stop_reason:"混合暂停/配额的原生合成验收".into(),threshold_reached:false}
}
fn attach(app:&mut PainterApp,requested:f64,busy:bool)->SyncSender<Event>{
    let result=fixture(requested);let target=Arc::new(result.preview.clone());
    let source=Arc::new(SourceImage{rgba:image::DynamicImage::ImageRgb8((*target).clone()).into_rgba8()});
    let view=ReferenceView{stage:ReferenceStage::Prepare,source:Some(source_view(&source,"参考流程验收.png")),
        target:Some(rgb_view(&target,"截取的参考图像")),busy,status:"准备序列化（合成验收）".into(),stop_rmse:requested,..Default::default()};
    let (tx,rx)=mpsc::sync_channel(3);let session=ReferenceSession{origin:Origin::capture(app),view: view.clone(),source:Some(source),
        target:Some(target),result:None,receiver:Some(rx),cancel:Arc::new(AtomicBool::new(false)),job:revision(),resume_auto_load:false};
    app.auto_load_pending=false;app.editor.set_reference_view(None);app.editor.set_reference_view(Some(view));app.reference_session=Some(session);tx
}
fn finish(app:&mut PainterApp,tx:&SyncSender<Event>,requested:f64)->Result<(),String>{
    let job=app.reference_session.as_ref().unwrap().job;
    tx.send(Event::Finished(job,Ok(fixture(requested)))).map_err(|e|e.to_string())?;app.poll_reference_session();Ok(())
}
impl PainterApp{
    pub(crate) fn drive_reference_smoke(&mut self){
        let args=std::env::args().collect::<Vec<_>>();
        let Some(path)=args.iter().position(|a|a=="--reference-smoke-output").and_then(|i|args.get(i+1)).map(PathBuf::from)else{return;};
        self.auto_load_pending=false;if self.frames<2||self.reference_smoke_done{return;}
        self.reference_smoke_done=true;let mut checks=Vec::new();let result=self.run_reference_smoke(&mut checks);
        let report=serde_json::json!({"status":if result.is_ok(){"passed"}else{"failed"},"error":result.err(),"checks":checks,
            "nodes":self.document.nodes.len(),"current":self.document.current,"model_loaded":self.loaded,"auto_load_pending":self.auto_load_pending,
            "scope":"native UI/controller, synthetic worker events, real import and save/load; no serializer GPU work and no model load"});
        if let Some(parent)=path.parent(){let _=std::fs::create_dir_all(parent);}
        let _=std::fs::write(path,serde_json::to_vec_pretty(&report).unwrap());
    }
    fn run_reference_smoke(&mut self,checks:&mut Vec<String>)->Result<(),String>{
        require(checks,"startup_no_model_or_implicit_archive",self.save_session.home&&self.save_session.entries.is_empty()&&!self.loaded)?;
        let library=self.save_session.catalog.as_ref().map_err(Clone::clone)?.clone();
        self.poll_library_outcome(save_session::perform(save_session::LibraryWork::Create{library:library.clone(),name:"参考图像导入验收".into()}));
        self.auto_load_pending=false;require(checks,"explicit_blank_archive_accepts_reference",self.reference_can_enter())?;
        let invalid=crate::reference_serialize::MIN_STOP_RMSE-0.001;
        let invalid_tx=attach(self,invalid,false);let job=self.reference_session.as_ref().unwrap().job;
        let before=serde_json::to_vec(&self.document.project_ref()).unwrap();
        self.reference_action(ReferenceAction::Serialize{stop_rmse:invalid});
        require(checks,"rmse_below_training_minimum_cannot_start_a_programmatic_serialization_job",self.reference_session.as_ref().is_some_and(|session|
            session.job==job&&!session.view.busy&&session.result.is_none()&&session.view.error.as_ref().is_some_and(|error|error.contains("12.75")))
            &&serde_json::to_vec(&self.document.project_ref()).unwrap()==before)?;
        finish(self,&invalid_tx,invalid)?;self.reference_action(ReferenceAction::Import);
        require(checks,"below_training_minimum_result_cannot_bypass_manual_import_validation",self.document.primitives.is_empty()
            &&self.reference_session.as_ref().is_some_and(|session|session.view.error.as_ref().is_some_and(|error|error.contains("12.75")))
            &&serde_json::to_vec(&self.document.project_ref()).unwrap()==before)?;
        self.cancel_reference_session();
        let tx=attach(self,12.75,true);let job=self.reference_session.as_ref().unwrap().job;let result=fixture(12.75);
        let before=serde_json::to_vec(&self.document.project_ref()).unwrap();
        tx.send(Event::Progress(job,SerializeProgress{primitives:result.primitives.clone(),records:result.records.clone(),rmse:result.rmse,
            shape_count:result.shape_count,stage_label:result.stage_label.clone(),preview:result.preview.clone()})).map_err(|e|e.to_string())?;
        self.poll_reference_session();
        require(checks,"progress_preview_never_imports_or_creates_history",serde_json::to_vec(&self.document.project_ref()).unwrap()==before
            &&self.reference_session.as_ref().unwrap().view.result.is_some()&&!self.reference_session.as_ref().unwrap().view.can_import)?;
        finish(self,&tx,12.75)?;
        require(checks,"completed_result_stays_outside_document_until_manual_import",self.document.primitives.is_empty()&&self.document.nodes.len()==1
            &&self.reference_session.as_ref().unwrap().view.can_import)?;
        self.reference_action(ReferenceAction::Import);
        require(checks,"manual_import_creates_one_image_edit_with_native_ids_and_attributes",self.reference_session.is_none()&&self.document.nodes.len()==2
            &&self.document.primitives.len()==2&&self.document.primitives.iter().all(|p|p.id!=0)&&self.save_session.dirty)?;
        require(checks,"reference_import_enters_modify_mode_and_global_prefix_can_continue",self.is_redraw()
            &&self.can_continue_reference_global()&&self.document.settings.pause_at_stage_end==false)?;
        require(checks,"nonempty_canvas_cannot_start_another_import",!self.reference_can_enter())?;
        let first_import=self.document.current;let imported=self.document.primitives.clone();
        self.document.jump(0)?;self.stage_map_node=None;
        let tx=attach(self,12.75,true);finish(self,&tx,12.75)?;
        self.document.commit_control_canvas("另一个空白状态",Vec::new())?;let changed=self.document.current;let count=self.document.nodes.len();
        self.reference_action(ReferenceAction::Import);
        require(checks,"changed_empty_origin_rejects_a_stale_completed_result",self.document.current==changed&&self.document.nodes.len()==count
            &&self.document.primitives.is_empty()&&self.reference_session.as_ref().unwrap().view.error.is_some())?;
        self.cancel_reference_session();
        let tx=attach(self,12.75,true);let job=self.reference_session.as_ref().unwrap().job;
        self.reference_action(ReferenceAction::CancelSerialization);finish(self,&tx,12.75)?;
        require(checks,"cancelled_job_cannot_import_a_late_success",self.document.primitives.is_empty()&&self.reference_session.as_ref().unwrap().result.is_none()
            &&!self.reference_session.as_ref().unwrap().view.busy&&!self.reference_session.as_ref().unwrap().view.can_import)?;
        self.cancel_reference_session();
        let tx=attach(self,12.75,true);let new_job=self.reference_session.as_ref().unwrap().job;
        tx.send(Event::Finished(job,Ok(fixture(12.75)))).map_err(|e|e.to_string())?;self.poll_reference_session();
        require(checks,"older_job_result_is_ignored_after_retry",new_job!=job&&self.reference_session.as_ref().unwrap().result.is_none())?;
        finish(self,&tx,20.)?;let count=self.document.nodes.len();self.reference_action(ReferenceAction::Import);
        require(checks,"changed_rmse_requires_a_matching_new_result_before_import",self.document.primitives.is_empty()&&self.document.nodes.len()==count
            &&self.reference_session.as_ref().unwrap().view.error.is_some())?;
        let session=self.reference_session.as_mut().unwrap();session.view.stop_rmse=20.;session.view.error=None;
        self.editor.set_reference_view(None);self.editor.set_reference_view(Some(session.view.clone()));
        self.reference_action(ReferenceAction::Import);
        require(checks,"retry_result_can_be_manually_accepted_without_auto_import",self.reference_session.is_none()&&self.document.primitives.len()==2
            &&self.document.materialize(first_import)?==imported)?;
        let entry=self.save_session.active.clone().unwrap();let id=entry.id.clone();
        let outcome=save_session::perform(save_session::LibraryWork::Save{library:library.clone(),entry,document:std::mem::take(&mut self.document),histories:std::mem::take(&mut self.histories)});
        self.poll_library_outcome(outcome);if let Some(error)=&self.save_session.error{return Err(error.clone());}
        let (loaded,raw)=project_file::load_project(&library.project_path(&id)?)?;
        require(checks,"reference_tree_and_raw_context_save_load_without_weights",loaded.primitives==self.document.primitives&&loaded.nodes.len()==self.document.nodes.len()
            &&raw.context_for(&loaded,loaded.current)?==self.histories.context_for(&self.document,self.document.current)?)?;
        self.document.jump(0)?;self.stage_map_node=None;self.redraw_view_key=None;
        let tx=attach(self,20.,true);let job=self.reference_session.as_ref().unwrap().job;
        tx.send(Event::Finished(job,Ok(mixed_reference_fixture()))).map_err(|e|e.to_string())?;self.poll_reference_session();
        self.reference_action(ReferenceAction::Import);
        let imported=self.document.primitives.clone();let imported_node=self.document.current;
        let selection=self.save_session.active.as_ref().unwrap().workspace.selection.clone();
        require(checks,"mixed_reference_defaults_to_only_unfinished_regions_in_modify_mode",self.is_redraw()
            &&!selection.selected_coarse().contains(&0)&&selection.selected_coarse().len()==15
            &&!selection.selected_fine().contains(&0)&&selection.selected_fine().len()==63)?;
        require(checks,"future_fine_data_is_retained_without_forcing_red_deletion",imported.len()==118&&selection.locked_fine().is_empty()
            &&self.histories.stage_floor(&self.document,imported_node)?==Some(4))?;
        let plan=crate::redraw_context::prepare_redraw(&self.document,&self.histories,imported_node,&selection)?;
        require(checks,"initial_model_context_has_only_global_and_coarse_data",plan.records.iter().all(|r|
            !region_state::marker_identity(r).is_ok_and(|(stage,_)|stage==region_state::Stage::Fine))
            &&plan.primitive_ids.iter().all(|id|self.document.primitives.iter().find(|p|p.id==*id).unwrap().stage_assignment.unwrap().stage!=StageTier::Fine))?;
        let origin=self.document.commit_control_canvas("参考逐阶段续画验收",plan.canvas)?;
        self.histories.set_control_overlay_with_ids(&self.document,origin,&plan.records,&plan.primitive_ids)?;
        self.histories.begin_redraw_origin(&self.document,origin,&plan.regions,&plan.records,&plan.primitive_ids)?;
        let mut parallel=ParallelHistory::new(&self.document,origin,String::new(),document::Settings{parallel_sequences:1,..self.document.settings.clone()})?;
        parallel.initialize_explicit_model_input(&self.document,&self.histories,&plan.records,&plan.primitive_ids,plan.pending_stage_ids,plan.regions.clone())?;
        let mut model_raw=plan.records;
        for &(side,index)in plan.regions.iter().filter(|(side,_)|*side==4){
            model_raw.push(region_state::marker_record(region_state::Stage::Coarse,index)?);
            model_raw.extend(plan.pending_stage_context.iter().filter(|p|(p.side,p.index)==(side,index)).map(|p|p.tokens));
            model_raw.push(region_state::END_RECORD);
        }
        let coarse_end=model_raw.len();
        let mut conditioning=std::collections::BTreeMap::<usize,Vec<crate::history_context::RawRecord>>::new();
        for p in &plan.pending_stage_context{if p.side==8&&!plan.regions.contains(&(p.side,p.index)){conditioning.entry(p.index).or_default().push(p.tokens);}}
        for (index,rows)in conditioning{model_raw.push(region_state::marker_record(region_state::Stage::Fine,index)?);model_raw.extend(rows);model_raw.push(region_state::END_RECORD);}
        for &(side,index)in plan.regions.iter().filter(|(side,_)|*side==8){
            model_raw.push(region_state::marker_record(region_state::Stage::Fine,index)?);
            model_raw.extend(plan.pending_stage_context.iter().filter(|p|(p.side,p.index)==(side,index)).map(|p|p.tokens));
            model_raw.push(region_state::END_RECORD);
        }
        require(checks,"fine_conditioning_follows_all_real_coarse_end_decisions",model_raw.iter().position(|r|
            region_state::marker_identity(r).is_ok_and(|(stage,_)|stage==region_state::Stage::Fine)).is_some_and(|position|position>=coarse_end))?;
        parallel.finish_lane(&self.document,&mut self.histories,0,&model_raw)?;
        require(checks,"model_can_resume_all_paused_regions_without_altering_reference_geometry",parallel.all_redraw_regions_complete()
            &&self.document.primitives==imported&&self.document.materialize(imported_node)?==imported)?;
        self.parallel=Some(parallel);self.stage_map_node=None;self.redraw_view_key=None;
        // The real UI observes the newly reached stage before the user saves.
        // Include that workspace reconciliation in this synthetic save too.
        self.refresh_redraw_ui();
        let entry=self.save_session.active.clone().unwrap();
        let outcome=save_session::perform(save_session::LibraryWork::Save{library:library.clone(),entry,document:std::mem::take(&mut self.document),histories:std::mem::take(&mut self.histories)});
        self.poll_library_outcome(outcome);if let Some(error)=&self.save_session.error{return Err(error.clone());}
        let (loaded,history)=project_file::load_project(&library.project_path(&id)?)?;
        require(checks,"modify_mode_and_deferred_reference_resume_survive_save_load",!self.save_session.dirty&&loaded.primitives==imported
            &&history.context_for(&loaded,loaded.current)?==Some(model_raw)
            &&library.list()?.entries[0].workspace.mode==crate::workspace::WorkspaceMode::Redraw)?;
        let result=fixture(20.);let target=Arc::new(result.preview.clone());let source=SourceImage{rgba:image::DynamicImage::ImageRgb8((*target).clone()).into_rgba8()};
        self.editor.set_reference_view(Some(ReferenceView{stage:ReferenceStage::Prepare,source:Some(source_view(&source,"参考流程验收.png")),
            target:Some(rgb_view(&target,"截取的参考图像")),result:Some(ReferenceResultView{preview:rgb_view(&result.preview,"序列化结果"),actual_rmse:result.rmse,
                shape_count:1,requested_rmse:20.,stop_reason:"已完成手动导入（合成验收）".into()}),status:"参考导入控制器验收通过".into(),stop_rmse:20.,can_import:false,..Default::default()}));
        self.auto_load_pending=false;self.error=None;self.status="参考导入验收通过 · 合成事件，没有加载模型".into();Ok(())
    }
}
