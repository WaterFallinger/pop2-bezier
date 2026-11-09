//! Isolated native Main/controller acceptance. It never loads a model.
use super::*;
use crate::{document::{PrimitiveStageAssignment,StageTier},region_state::{self,Stage}};

fn require(checks:&mut Vec<String>,name:&str,passed:bool)->Result<(),String>{
    if !passed{return Err(format!("局部重绘验收失败：{name}"));}checks.push(name.into());Ok(())
}
fn shape(stage:StageTier,cell:usize,order:u64,color:[u8;3])->Primitive{
    let side=stage.side();let x=if side==1{240.}else{(cell%side)as f32*512./side as f32+32.};
    let y=if side==1{240.}else{(cell/side)as f32*512./side as f32+32.};
    let mut p=Primitive::new(ShapeKind::Ellipse,x,y,32.,28.,color);
    p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:order});p
}
fn fixture(app:&mut PainterApp)->Result<(),String>{
    let mut bg=Primitive::background([238,242,250]);bg.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});
    let canvas=vec![bg,shape(StageTier::Global,0,1,[240,168,96]),shape(StageTier::Coarse,0,2,[80,140,210]),
        shape(StageTier::Coarse,5,3,[210,100,80]),shape(StageTier::Fine,18,4,[80,170,110]),
        shape(StageTier::Fine,27,5,[150,100,200]),shape(StageTier::Fine,63,6,[210,160,60])];
    let node=app.document.commit_canvas("已完成的三个绘制阶段",OperationSource::User,canvas)?;
    let mut rows=vec![region_state::BOS_RECORD];let mut ids=Vec::new();
    for p in &app.document.primitives[..2]{rows.push(p.to_tokens()?.map(u32::from));ids.push(p.id);}
    for (stage,index,position) in [(Stage::Coarse,0,2),(Stage::Coarse,5,3),(Stage::Fine,18,4),(Stage::Fine,27,5),(Stage::Fine,63,6)]{
        rows.extend([region_state::marker_record(stage,index)?,app.document.primitives[position].to_tokens()?.map(u32::from),region_state::END_RECORD]);
        ids.push(app.document.primitives[position].id);
    }
    app.histories.set_control_overlay_with_ids(&app.document,node,&rows,&ids)?;app.refresh_drawing_region();Ok(())
}

impl PainterApp{
    pub(super) fn drive_redraw_smoke(&mut self){
        let args=std::env::args().collect::<Vec<_>>();
        let Some(path)=args.iter().position(|a|a=="--redraw-smoke-output").and_then(|i|args.get(i+1)).map(PathBuf::from)else{return;};
        self.auto_load_pending=false;
        if self.frames<2||self.redraw_smoke_done{return;}
        self.redraw_smoke_done=true;
        let mut checks=Vec::new();let result=self.run_redraw_smoke(&mut checks);
        let report=serde_json::json!({"status":if result.is_ok(){"passed"}else{"failed"},"error":result.err(),"checks":checks,
            "model_loaded":self.loaded,"auto_load_pending":self.auto_load_pending,"nodes":self.document.nodes.len(),
            "current":self.document.current,"mode":"redraw","scope":"native Main actions and real delta/ID callbacks; synthetic model tokens, no model load"});
        if let Some(parent)=path.parent(){let _=std::fs::create_dir_all(parent);}
        let _=std::fs::write(path,serde_json::to_vec_pretty(&report).unwrap());
    }
    fn run_redraw_smoke(&mut self,checks:&mut Vec<String>)->Result<(),String>{
        require(checks,"startup_no_implicit_save",self.save_session.home&&self.save_session.entries.is_empty())?;
        let library=self.save_session.catalog.as_ref().map_err(Clone::clone)?.clone();
        self.poll_library_outcome(save_session::perform(save_session::LibraryWork::Create{library:library.clone(),name:"局部重绘验收".into()}));
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:true});
        require(checks,"same_archive_switches_to_redraw_while_canvas_stays_blank",self.is_redraw()&&self.document.primitives.is_empty()&&!self.loaded)?;
        fixture(self)?;self.stage_map_node=None;self.redraw_view_key=None;
        let original_node=self.document.current;let original=self.document.primitives.clone();
        self.action(ui::UiAction::ToggleRedrawRegion{stage:Stage::Fine,index:18});
        self.action(ui::UiAction::ToggleRedrawRegion{stage:Stage::Coarse,index:5});
        let selection=self.save_session.active.as_ref().unwrap().workspace.selection.clone();
        require(checks,"coarse_selection_locks_only_occupied_fine_children",selection.selected_coarse()==[5]&&selection.locked_fine()==[18,27])?;
        let before_nodes=self.document.nodes.len();
        self.start_redraw_generation(String::new(),self.document.settings.clone());
        require(checks,"occupied_generate_rejected_without_deleting_or_creating_node",self.document.nodes.len()==before_nodes&&self.document.primitives==original&&!self.running)?;
        self.action(ui::UiAction::DeleteLockedRedrawRegions);
        let red_cleared=self.save_session.active.as_ref().unwrap().workspace.selection.clone();
        let map=self.histories.stage_map(&self.document,self.document.current)?;
        require(checks,"explicit_red_clear_keeps_yellow_coarse_and_clears_only_red_children",
            self.document.primitives.iter().any(|p|p==&original[3])
                &&!self.document.primitives.iter().any(|p|p.id==original[4].id||p.id==original[5].id)
                &&red_cleared.validate_empty(&self.document.primitives,&map).is_ok()
                &&self.document.materialize(original_node)?==original)?;
        self.action(ui::UiAction::DeleteRedrawRegions);
        let selection=self.save_session.active.as_ref().unwrap().workspace.selection.clone();
        require(checks,"manual_delete_removes_selected_stages_only",self.document.primitives.iter().map(|p|p.id).collect::<Vec<_>>()
            ==[original[0].id,original[1].id,original[2].id,original[6].id])?;
        require(checks,"red_locks_stay_after_manual_delete",selection.locked_fine()==[18,27])?;
        let latched=selection.clone();
        self.action(ui::UiAction::ToggleRedrawRegion{stage:Stage::Coarse,index:5});
        let selection=self.save_session.active.as_ref().unwrap().workspace.selection.clone();
        require(checks,"deselect_parent_restores_previous_yellow_fine",selection.locked_fine().is_empty()&&selection.selected_fine()==[18])?;
        // Resume the latched workspace snapshot to verify that its red locks
        // survive saving as well as the preceding delete operation.
        self.save_session.active.as_mut().unwrap().workspace.selection=latched;self.redraw_view_key=None;
        let selection=self.save_session.active.as_ref().unwrap().workspace.selection.clone();
        let plan=redraw_context::prepare_redraw(&self.document,&self.histories,self.document.current,&selection)?;
        require(checks,"redraw_can_revisit_coarse_after_fine_without_altering_old_branch",plan.stage_floor==Some(4)
            &&self.histories.stage_floor(&self.document,self.document.current)?==Some(8)&&self.document.materialize(original_node)?==original)?;
        let outside=self.document.primitives.clone();
        let origin=self.document.commit_control_canvas("开始局部重绘验收",plan.canvas)?;
        self.histories.set_control_overlay_with_ids(&self.document,origin,&plan.records,&plan.primitive_ids)?;
        let settings=document::Settings{parallel_sequences:2,..self.document.settings.clone()};
        let mut parallel=ParallelHistory::new(&self.document,origin,String::new(),settings)?;
        parallel.initialize_explicit_model_input(&self.document,&self.histories,&plan.records,&plan.primitive_ids,plan.pending_stage_ids,plan.regions.clone())?;
        let mut full=plan.records.clone();
        let tokens=shape(StageTier::Coarse,5,99,[100,180,230]).to_tokens()?.map(u32::from);
        full.extend([region_state::marker_record(Stage::Coarse,5)?,tokens]);
        parallel.submit(&mut self.document,&mut self.histories,0,tokens,&full)?;
        let new=self.document.primitives.iter().find(|p|!outside.iter().any(|old|old.id==p.id)).ok_or("缺少新的粗网格图元")?;
        require(checks,"redrawn_coarse_inserted_below_existing_fine",self.document.primitives.iter().position(|p|p.id==new.id)
            <self.document.primitives.iter().position(|p|p.id==outside.last().unwrap().id))?;
        let count=self.document.nodes.len();let mut illegal=plan.records.clone();
        illegal.extend([region_state::marker_record(Stage::Coarse,6)?,tokens]);
        require(checks,"out_of_scope_callback_cannot_mutate_canvas",parallel.submit(&mut self.document,&mut self.histories,1,tokens,&illegal).is_err()&&self.document.nodes.len()==count)?;
        parallel.fail_lane(1,"synthetic forbidden callback rejected")?;
        full.push(region_state::END_RECORD);
        for pending in &plan.pending_stage_context{full.extend([region_state::marker_record(Stage::Fine,pending.index)?,pending.tokens,region_state::END_RECORD]);}
        for cell in [18,27]{let tokens=shape(StageTier::Fine,cell,100+cell as u64,[140,200,120]).to_tokens()?.map(u32::from);
            full.extend([region_state::marker_record(Stage::Fine,cell)?,tokens]);parallel.submit(&mut self.document,&mut self.histories,0,tokens,&full)?;full.push(region_state::END_RECORD);}
        parallel.finish_lane(&self.document,&mut self.histories,0,&full)?;
        require(checks,"failed_parallel_branch_not_reported_as_full_scope_complete",!parallel.all_redraw_regions_complete())?;
        require(checks,"unselected_global_coarse_and_fine_primitives_unchanged",outside.iter().all(|old|self.document.primitives.iter().find(|p|p.id==old.id)==Some(old)))?;
        let stages=self.histories.stage_map(&self.document,self.document.current)?;
        let counts=redraw::stage_canvases(&self.document.primitives,&stages)?;
        require(checks,"only_requested_regions_received_new_geometry",counts.coarse_counts[5]==1&&counts.fine_counts[18]==1&&counts.fine_counts[27]==1
            &&self.document.primitives.len()==outside.len()+3)?;
        // A populated yellow fine region is a reread/continuation request,
        // while the earlier populated red-lock check still refuses deletion.
        self.save_session.active.as_mut().unwrap().workspace.selection=redraw::RedrawSelection::default();
        self.redraw_view_key=None;self.action(ui::UiAction::ToggleRedrawRegion{stage:Stage::Fine,index:63});
        let before_yellow=self.document.primitives.clone();let yellow_parent=self.document.current;
        let selection=self.save_session.active.as_ref().unwrap().workspace.selection.clone();
        let yellow=redraw_context::prepare_redraw(&self.document,&self.histories,yellow_parent,&selection)?;
        require(checks,"populated_yellow_target_is_retained_without_manual_empty",yellow.canvas==before_yellow
            &&yellow.pending_stage_ids.iter().any(|&(s,i,_)|(s,i)==(8,63)))?;
        let origin=self.document.commit_control_canvas("黄色区域续画验收",yellow.canvas)?;
        self.histories.set_control_overlay_with_ids(&self.document,origin,&yellow.records,&yellow.primitive_ids)?;
        self.histories.begin_redraw_origin(&self.document,origin,&yellow.regions,&yellow.records,&yellow.primitive_ids)?;
        let mut continued=ParallelHistory::new(&self.document,origin,String::new(),document::Settings{parallel_sequences:2,..self.document.settings.clone()})?;
        continued.initialize_explicit_model_input(&self.document,&self.histories,&yellow.records,&yellow.primitive_ids,yellow.pending_stage_ids,yellow.regions.clone())?;
        let mut reread=yellow.records.clone();let mut groups=std::collections::BTreeMap::<usize,Vec<[u32;9]>>::new();
        for pending in &yellow.pending_stage_context{if pending.index!=63{groups.entry(pending.index).or_default().push(pending.tokens);}}
        for (index,rows)in groups{reread.push(region_state::marker_record(Stage::Fine,index)?);reread.extend(rows);reread.push(region_state::END_RECORD);}
        reread.push(region_state::marker_record(Stage::Fine,63)?);
        let tokens=shape(StageTier::Fine,63,999,[160,100,200]).to_tokens()?.map(u32::from);
        let mut missing=reread.clone();missing.push(tokens);let before_nodes=self.document.nodes.len();
        require(checks,"callback_cannot_skip_retained_yellow_input",continued.submit(&mut self.document,&mut self.histories,0,tokens,&missing).is_err()
            &&self.document.nodes.len()==before_nodes&&self.document.primitives==before_yellow)?;
        for pending in &yellow.pending_stage_context{if pending.index==63{reread.push(pending.tokens);}}
        require(checks,"selected_yellow_input_has_no_old_end",reread.last()!=Some(&region_state::END_RECORD)
            &&!self.histories.completion_evidence(&self.document,origin)?.regions.iter().any(|p|(p.side,p.index)==(8,63)))?;
        let mut generated=reread.clone();generated.push(tokens);
        continued.submit(&mut self.document,&mut self.histories,0,tokens,&generated)?;
        require(checks,"parallel_reread_ids_preserve_every_existing_primitive",before_yellow.iter().all(|old|self.document.primitives.iter().find(|p|p.id==old.id)==Some(old))
            &&self.document.primitives.len()==before_yellow.len()+1)?;
        generated.push(region_state::END_RECORD);continued.finish_lane(&self.document,&mut self.histories,0,&generated)?;
        reread.push(region_state::END_RECORD);continued.finish_lane(&self.document,&mut self.histories,1,&reread)?;
        require(checks,"fresh_model_end_completes_yellow_visit_and_old_branch_is_unchanged",continued.all_redraw_regions_complete()
            &&self.document.materialize(yellow_parent)?==before_yellow
            &&self.histories.completion_evidence(&self.document,self.document.current)?.regions.iter().any(|p|(p.side,p.index)==(8,63)))?;
        parallel=continued;
        // Revisit Global on the original completed canvas. No dependency is
        // deleted by selecting it or by a rejected Generate request.
        self.document.jump(original_node)?;self.stage_map_node=None;self.redraw_view_key=None;
        self.save_session.active.as_mut().unwrap().workspace.selection=redraw::RedrawSelection::default();
        self.action(ui::UiAction::ToggleRedrawRegion{stage:Stage::Global,index:0});
        let global_selection=self.save_session.active.as_ref().unwrap().workspace.selection.clone();
        require(checks,"global_selection_locks_every_occupied_coarse_and_fine_region",global_selection.global_selected()
            &&global_selection.locked_coarse()==[0,5]&&global_selection.locked_fine()==[18,27,63]
            &&self.document.primitives==original)?;
        let nodes=self.document.nodes.len();self.start_redraw_generation(String::new(),self.document.settings.clone());
        require(checks,"global_generate_refuses_remaining_red_geometry_without_side_effects",!self.running
            &&self.document.nodes.len()==nodes&&self.document.primitives==original)?;
        self.action(ui::UiAction::DeleteLockedRedrawRegions);
        let retained=self.document.primitives.clone();
        require(checks,"manual_global_dependency_clear_preserves_background_and_global_geometry",retained==original[..2]
            &&self.save_session.active.as_ref().unwrap().workspace.selection.clone().locked_coarse()==[0,5]&&self.save_session.active.as_ref().unwrap().workspace.selection.clone().locked_fine()==[18,27,63])?;
        let plan=redraw_context::prepare_redraw(&self.document,&self.histories,self.document.current,&self.save_session.active.as_ref().unwrap().workspace.selection.clone())?;
        let origin=self.document.commit_control_canvas("准备局部重绘",plan.canvas)?;
        self.histories.set_control_overlay_with_ids(&self.document,origin,&plan.records,&plan.primitive_ids)?;
        self.histories.begin_redraw_origin(&self.document,origin,&plan.regions,&plan.records,&plan.primitive_ids)?;
        require(checks,"global_fresh_branch_revokes_old_end_markers_and_phase_completion",self.histories.stage_floor(&self.document,origin)?.unwrap_or(1)==1
            &&self.histories.completion_evidence(&self.document,origin)?.regions.is_empty()
            &&plan.records.iter().all(|r|!region_state::is_marker(r)&&r!=&region_state::END_RECORD))?;
        let mut global_parallel=ParallelHistory::new(&self.document,origin,String::new(),document::Settings{parallel_sequences:1,..self.document.settings.clone()})?;
        global_parallel.initialize_explicit_model_input(&self.document,&self.histories,&plan.records,&plan.primitive_ids,plan.pending_stage_ids,plan.regions)?;
        let mut global_raw=plan.records;let row=shape(StageTier::Global,0,2000,[160,120,220]).to_tokens()?.map(u32::from);
        global_raw.push(row);global_parallel.submit(&mut self.document,&mut self.histories,0,row,&global_raw)?;
        global_parallel.finish_lane(&self.document,&mut self.histories,0,&global_raw)?;
        require(checks,"global_model_callback_extends_retained_prefix_and_preserves_old_branch",self.document.primitives.len()==retained.len()+1
            &&retained.iter().all(|old|self.document.primitives.iter().find(|p|p.id==old.id)==Some(old))
            &&self.document.materialize(original_node)?==original&&!global_parallel.all_redraw_regions_complete())?;
        parallel=global_parallel;
        self.parallel=Some(parallel);self.running=false;self.stopping=false;self.stage_map_node=None;self.redraw_view_key=None;
        self.refresh_redraw_ui();
        let entry=self.save_session.active.clone().unwrap();
        let outcome=save_session::perform(save_session::LibraryWork::Save{library:library.clone(),entry,document:std::mem::take(&mut self.document),histories:std::mem::take(&mut self.histories)});
        self.poll_library_outcome(outcome);
        if let Some(error)=&self.save_session.error{return Err(error.clone());}
        let entry=library.list()?.entries.into_iter().next().ok_or("持久化后没有修改存档")?;
        let (loaded,store)=project_file::load_project(&library.project_path(&entry.id)?)?;
        require(checks,"saved_tree_and_redraw_mode_selection_roundtrip",loaded.primitives==self.document.primitives&&loaded.nodes.len()==self.document.nodes.len()
            &&entry.workspace==self.save_session.active.as_ref().unwrap().workspace
            &&store.stage_map(&loaded,loaded.current)?==self.histories.stage_map(&self.document,self.document.current)?)?;
        self.refresh_drawing_region();self.refresh_redraw_ui();self.error=None;
        self.status="全图与网格修改续画验收通过 · 合成图元，没有加载模型".into();Ok(())
    }
}
