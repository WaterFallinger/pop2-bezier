//! Opt-in synthetic installed-application acceptance. No model is loaded.
use super::*;

fn require(report:&mut serde_json::Value,name:&str,condition:bool)->Result<(),String>{
    report["checks"].as_array_mut().unwrap().push(serde_json::json!({"name":name,"passed":condition}));
    if condition{Ok(())}else{Err(format!("合成阶段检查失败：{name}"))}
}
fn phase(app:&PainterApp)->Result<Option<usize>,String>{app.histories.stage_floor(&app.document,app.document.current)}
fn map(app:&PainterApp)->Result<BTreeMap<u64,region_state::Stage>,String>{app.histories.stage_map(&app.document,app.document.current)}
fn has_fine_marker(raw:&[[u32;9]])->bool{raw.iter().any(|row|region_state::marker_identity(row).is_ok_and(|(stage,_)|stage==region_state::Stage::Fine))}
fn assert_no_empty_interrupt(report:&mut serde_json::Value,name:&str,raw:&[[u32;9]])->Result<(),String>{
    require(report,name,!raw.windows(2).any(|pair|region_state::is_marker(&pair[0])&&pair[1]==region_state::INTERRUPT_RECORD))
}
fn record_step(app:&PainterApp,report:&mut serde_json::Value,name:&str)->Result<(),String>{
    let raw=app.histories.effective_history(&app.document,app.document.current)?.records;
    assert_no_empty_interrupt(report,&format!("{name}_raw_has_no_empty_marker_interrupt"),&raw)?;
    report["steps"].as_array_mut().unwrap().push(serde_json::json!({"step":name,"nodes":app.document.nodes.len(),
        "current":app.document.current,"canvas_primitives":app.document.primitives.len(),"phase_floor":phase(app)?}));Ok(())
}
fn rect(x:f32,y:f32,color:[u8;3])->Primitive{Primitive::new(document::ShapeKind::Rectangle,x,y,12.,10.,color)}

impl PainterApp {
    pub(super) fn drive_editor_stage_smoke(&mut self){
        let Some(path)=self.editor_stage_smoke.clone()else{return;};
        // This flag is an independent synthetic app. Disable autoload already
        // on its first frame, even though the acceptance runs from frame two.
        self.auto_load_pending=false;
        if self.frames<2||self.editor_stage_smoke_done{return;}
        self.editor_stage_smoke_done=true;
        let mut report=serde_json::json!({"status":"running","scope":"synthetic Main UiActions; no model inference",
            "checks":[],"steps":[]});
        let result=self.run_editor_stage_smoke(&path,&mut report);
        match result{Ok(())=>{report["status"]="passed".into();self.status="阶段编辑合成检查已通过".into();},
            Err(error)=>{report["status"]="failed".into();report["error"]=error.clone().into();self.status=error;}}
        report["final_nodes"]=self.document.nodes.len().into();report["final_current"]=self.document.current.into();
        report["final_phase_floor"]=serde_json::json!(phase(self).ok().flatten());
        report["model_loaded"]=self.loaded.into();report["auto_load_pending"]=self.auto_load_pending.into();
        if let Some(parent)=path.parent().filter(|p|!p.as_os_str().is_empty()){let _=std::fs::create_dir_all(parent);}
        match serde_json::to_vec_pretty(&report){Ok(bytes)=>{if let Err(error)=std::fs::write(&path,bytes){eprintln!("stage smoke report write failed: {error}");}},
            Err(error)=>eprintln!("stage smoke report encoding failed: {error}")}
    }

    fn run_editor_stage_smoke(&mut self,path:&std::path::Path,report:&mut serde_json::Value)->Result<(),String>{
        require(report,"no_model_loaded_or_loading",!self.loaded&&!self.loading&&!self.compiling&&!self.running)?;
        self.document=Document::new();self.histories=RawHistoryStore::new();self.parallel=None;self.pending_canvas_edit=None;
        self.cancel_global_edit();self.error=None;self.stage_map_node=None;
        self.document.settings.pause_at_stage_end=true;self.document.settings.auto_reorder_before_generation=false;
        self.document.settings.auto_seed=false;
        let mut canvas=vec![Primitive::background([255;3])];
        canvas.extend((0..85).map(|_|rect(256.,256.,[80,100,120])));
        self.action(ui::UiAction::CommitCanvas{label:"Smoke：白色背景与85个全图图元".into(),primitives:canvas});
        require(report,"base_has_background_plus_85_global",self.document.primitives.len()==86
            &&map(self)?.values().filter(|stage|**stage==region_state::Stage::Global).count()==86)?;
        let mut recolored=self.document.primitives.clone();recolored[1].color=[100,140,180];
        self.action(ui::UiAction::CommitCanvas{label:"Smoke：全图满额仅修改颜色".into(),primitives:recolored});
        let effective=self.histories.effective_history(&self.document,self.document.current)?;
        require(report,"A_recolor_does_not_advance_phase",phase(self)?.is_none()&&!effective.records.iter().any(region_state::is_marker))?;
        record_step(self,report,"A")?;

        let before=self.document.primitives.clone();let nodes=self.document.nodes.len();
        let mut proposed=before.clone();proposed.push(rect(64.,64.,[20,60,100]));
        self.action(ui::UiAction::CommitCanvas{label:"Smoke：待确认全局溢出".into(),primitives:proposed.clone()});
        require(report,"B_overflow_is_pending_and_not_committed",self.pending_global_edit.is_some()&&self.document.nodes.len()==nodes&&self.document.primitives==before)?;
        self.action(ui::UiAction::CancelGlobalTransition);
        require(report,"B_cancel_does_not_add_a_node_or_primitive",self.pending_global_edit.is_none()&&self.document.nodes.len()==nodes&&self.document.primitives==before)?;
        self.action(ui::UiAction::CommitCanvas{label:"Smoke：再次请求全局溢出".into(),primitives:proposed});
        self.action(ui::UiAction::ConfirmGlobalTransition);
        let stages=map(self)?;let first_coarse=*stages.iter().find(|(_,stage)|**stage==region_state::Stage::Coarse).ok_or("确认后缺少粗网格图元")?.0;
        require(report,"B_confirm_adds_one_coarse_and_enters_coarse",self.pending_global_edit.is_none()&&self.document.primitives.len()==87
            &&stages.values().filter(|stage|**stage==region_state::Stage::Coarse).count()==1&&phase(self)?==Some(4))?;
        let before_region=self.drawing_region.clone();
        let before_controls=self.histories.effective_history(&self.document,self.document.current)?.records.into_iter()
            .filter(|row|model::validate_visible(row).is_err()).collect::<Vec<_>>();
        let global_id=self.document.primitives.iter().find(|p|p.shape!=ShapeKind::Background&&stages[&p.id]==region_state::Stage::Global)
            .ok_or("缺少历史全局图元")?.id;
        let mut recolored=self.document.primitives.clone();recolored.iter_mut().find(|p|p.id==global_id).unwrap().color=[120,160,200];
        self.action(ui::UiAction::CommitCanvas{label:"Smoke：粗网格期间只修改全局图形颜色".into(),primitives:recolored});
        let after=self.histories.effective_history(&self.document,self.document.current)?;
        let after_controls=after.records.iter().copied().filter(|row|model::validate_visible(row).is_err()).collect::<Vec<_>>();
        let new_global=self.document.primitives.iter().find(|p|p.id==global_id).unwrap().to_tokens()?.map(u32::from);
        require(report,"B_global_recolor_in_coarse_updates_color_without_reopening_or_lighting_coarse_cell",self.drawing_region==before_region
            &&before_controls==after_controls&&map(self)?[&global_id]==region_state::Stage::Global&&after.records.contains(&new_global))?;
        record_step(self,report,"B")?;

        let mut proposed=self.document.primitives.clone();proposed.extend((0..20).map(|n|rect(32.+(n%5)as f32*10.,32.+(n/5)as f32*10.,[40+n as u8*2,80+n as u8*2,120])));
        self.action(ui::UiAction::CommitCanvas{label:"Smoke：同粗网格输入21个图元".into(),primitives:proposed});
        let stages=map(self)?;let coarse=stages.iter().filter(|(_,stage)|**stage==region_state::Stage::Coarse).map(|(id,_)|*id).collect::<std::collections::HashSet<_>>();
        let effective=self.histories.effective_history(&self.document,self.document.current)?;
        let admitted=effective.primitive_ids.iter().filter(|id|coarse.contains(*id)).count();
        let deferred=*coarse.iter().find(|id|!effective.primitive_ids.contains(*id)).ok_or("缺少待Fine参考的粗网格溢出图元")?;
        require(report,"C_all_21_remain_on_canvas_but_only_20_are_admitted",coarse.len()==21&&admitted==20&&self.histories.deferred_fine_context(&self.document,self.document.current)?.len()==1)?;
        let proposed=self.document.primitives.iter().filter(|p|p.id!=first_coarse).cloned().collect();
        self.action(ui::UiAction::CommitCanvas{label:"Smoke：删除已准入的旧粗网格图元".into(),primitives:proposed});
        let effective=self.histories.effective_history(&self.document,self.document.current)?;
        require(report,"C_old_deferred_id_is_promoted_after_deletion",self.histories.deferred_fine_context(&self.document,self.document.current)?.is_empty()
            &&effective.primitive_ids.contains(&deferred)&&phase(self)?==Some(4))?;
        record_step(self,report,"C")?;

        let parent=self.document.current;let parent_raw=self.histories.effective_history(&self.document,parent)?.records;
        self.action(ui::UiAction::AssignStage{ids:vec![deferred],stage:region_state::Stage::Fine});
        let assigned=self.document.primitives.iter().find(|p|p.id==deferred).ok_or("阶段调整丢失图元")?.to_tokens()?.map(u32::from);
        let future=self.histories.pending_stage_context(&self.document,self.document.current)?;
        let effective=self.histories.effective_history(&self.document,self.document.current)?;
        require(report,"D_assigning_fine_does_not_advance_model_phase",phase(self)?==Some(4)&&map(self)?.get(&deferred)==Some(&region_state::Stage::Fine)
            &&future.iter().any(|entry|entry.side==8&&entry.tokens==assigned)&&!has_fine_marker(&effective.records))?;
        require(report,"D_parent_raw_snapshot_is_immutable",self.histories.effective_history(&self.document,parent)?.records==parent_raw)?;
        record_step(self,report,"D")?;

        let mut proposed=self.document.primitives.clone();
        for n in 0..10{let mut p=rect(460.+(n%5)as f32*2.,460.+(n/5)as f32*2.,[140+n as u8*2,40,80]);
            p.stage_assignment=Some(document::PrimitiveStageAssignment{stage:document::StageTier::Fine,cell:63,queue_order:10000+n});proposed.push(p);}
        let nodes=self.document.nodes.len();self.action(ui::UiAction::CommitCanvas{label:"Smoke：另一个Fine格输入10个未来图元".into(),primitives:proposed});
        require(report,"E_ten_future_fine_primitives_are_accepted",self.document.nodes.len()==nodes+1&&phase(self)?==Some(4)
            &&self.document.primitives.iter().filter(|p|p.stage_assignment.is_some_and(|a|a.stage==document::StageTier::Fine&&a.cell==63)).count()==10)?;
        let before=self.document.primitives.clone();let nodes=self.document.nodes.len();let mut proposed=before.clone();let mut p=rect(470.,464.,[180,40,80]);
        p.stage_assignment=Some(document::PrimitiveStageAssignment{stage:document::StageTier::Fine,cell:63,queue_order:20000});proposed.push(p);
        self.action(ui::UiAction::CommitCanvas{label:"Smoke：拒绝Fine格第11个输入".into(),primitives:proposed});
        require(report,"E_eleventh_fine_primitive_is_rejected_without_mutation",self.document.nodes.len()==nodes&&self.document.primitives==before)?;
        record_step(self,report,"E")?;

        let saved_phase=phase(self)?;let saved_current=self.document.current;let saved_canvas=self.document.primitives.clone();let saved_map=map(self)?;
        let saved_effective=self.histories.effective_history(&self.document,saved_current)?;let saved_deferred=self.histories.deferred_fine_context(&self.document,saved_current)?;
        let saved_pending=self.histories.pending_stage_context(&self.document,saved_current)?.into_iter().map(|p|(p.side,p.index,p.tokens)).collect::<Vec<_>>();
        assert_no_empty_interrupt(report,"raw_has_no_empty_marker_interrupt_pairs",&saved_effective.records)?;
        if let Some(parent)=path.parent().filter(|p|!p.as_os_str().is_empty()){std::fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
        let project_path=path.with_extension("pproj");project_file::save_project(&project_path,&self.document,&self.histories)?;
        let(loaded,history)=project_file::load_project(&project_path)?;let loaded_effective=history.effective_history(&loaded,loaded.current)?;
        let loaded_pending=history.pending_stage_context(&loaded,loaded.current)?.into_iter().map(|p|(p.side,p.index,p.tokens)).collect::<Vec<_>>();
        require(report,"F_project_restores_phase_ids_admission_stage_ownership_and_pending",loaded.current==saved_current&&loaded.primitives==saved_canvas
            &&history.stage_floor(&loaded,loaded.current)?==saved_phase&&history.stage_map(&loaded,loaded.current)?==saved_map
            &&loaded_effective.primitive_ids==saved_effective.primitive_ids&&loaded_effective.records==saved_effective.records
            &&history.deferred_fine_context(&loaded,loaded.current)?==saved_deferred&&loaded_pending==saved_pending)?;
        self.document=loaded;self.histories=history;self.stage_map_node=None;self.editor.invalidate_history_after_deletion();self.refresh_drawing_region();
        let head=self.document.current;let nodes=self.document.nodes.len();
        let parent=self.document.nodes[head].parent.ok_or("合成场景缺少可回看的父节点")?;
        let parent_canvas=self.document.materialize(parent)?;
        self.action(ui::UiAction::BrowseHistory(parent as u64));
        require(report,"G_browse_parent_restores_canvas_without_editing_tree",self.document.current==parent
            &&self.document.primitives==parent_canvas&&self.document.nodes.len()==nodes)?;
        self.action(ui::UiAction::BrowseHistory(head as u64));
        require(report,"G_browse_forward_restores_canvas_without_creating_branch",self.document.current==head
            &&self.document.primitives==saved_canvas&&self.document.nodes.len()==nodes)?;
        let selected=*map(self)?.iter().find(|(_,stage)|**stage==region_state::Stage::Coarse).ok_or("合成结果缺少可截图的粗网格图元")?.0;
        self.editor.canvas.selection=vec![selected];record_step(self,report,"F")?;
        report["selected_coarse_primitive"]=selected.into();Ok(())
    }
}
