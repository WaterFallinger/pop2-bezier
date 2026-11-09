//! Native controller acceptance with synthetic callbacks and no model load.
use super::*;
use crate::{document::{PrimitiveStageAssignment,StageTier},region_state::{self,Stage}};
fn check(checks:&mut Vec<String>,name:&str,ok:bool)->Result<(),String>{if !ok{return Err(format!("模式互转验收失败：{name}"));}checks.push(name.into());Ok(())}
fn cell_shape(stage:StageTier,cell:usize,order:u64)->Primitive{
    let side=stage.side();let mut p=Primitive::new(ShapeKind::Rectangle,(cell%side)as f32*512./side as f32+24.,
        (cell/side)as f32*512./side as f32+24.,12.,10.,[80,140,200]);
    p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:order});p
}
fn same_visible(a:&[Primitive],b:&[Primitive])->bool{a.len()==b.len()&&a.iter().zip(b).all(|(a,b)|{
    let mut b=b.clone();b.stage_assignment=a.stage_assignment;&b==a
})}
impl PainterApp{
    pub(super) fn drive_mode_switch_smoke(&mut self){
        let args=std::env::args().collect::<Vec<_>>();
        let Some(path)=args.iter().position(|a|a=="--mode-switch-smoke-output").and_then(|i|args.get(i+1)).map(PathBuf::from)else{return;};
        self.auto_load_pending=false;if self.frames<2||self.mode_switch_smoke_done{return;}
        self.mode_switch_smoke_done=true;let mut checks=Vec::new();let result=self.run_mode_switch_smoke(&mut checks);
        let report=serde_json::json!({"status":if result.is_ok(){"passed"}else{"failed"},"error":result.err(),"checks":checks,
            "model_loaded":self.loaded,"auto_load_pending":self.auto_load_pending,"same_save_id":self.save_session.active.as_ref().map(|e|e.id.clone()),
            "mode":if self.is_redraw(){"modify"}else{"standard"},"nodes":self.document.nodes.len(),"current":self.document.current,
            "scope":"native controller and tree/provenance/project persistence; synthetic callbacks"});
        if let Some(parent)=path.parent(){let _=std::fs::create_dir_all(parent);}
        let _=std::fs::write(path,serde_json::to_vec_pretty(&report).unwrap());
    }
    fn run_mode_switch_smoke(&mut self,checks:&mut Vec<String>)->Result<(),String>{
        let library=self.save_session.catalog.as_ref().map_err(Clone::clone)?.clone();
        self.poll_library_outcome(save_session::perform(save_session::LibraryWork::Create{library:library.clone(),name:"统一存档互转验收".into()}));
        let id=self.save_session.active.as_ref().ok_or("没有创建统一存档")?.id.clone();
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:true});
        check(checks,"standard_to_modify_blank_unrestricted_same_save",self.is_redraw()&&self.document.primitives.is_empty()&&self.save_session.active.as_ref().unwrap().id==id)?;
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:false});
        check(checks,"blank_modify_to_standard_works_same_save",!self.is_redraw()&&self.save_session.active.as_ref().unwrap().id==id)?;
        let mut canvas=vec![Primitive::background([238,242,248])];
        canvas.extend((0..85).map(|i|Primitive::new(ShapeKind::Ellipse,(i%10)as f32*40.+30.,(i/10)as f32*40.+30.,12.,10.,[160,180,210])));
        self.action(ui::UiAction::CommitCanvas{label:"完整全局阶段".into(),primitives:canvas});
        let settings=document::Settings{parallel_sequences:1,..self.document.settings.clone()};
        let mut raw=self.histories.effective_history(&self.document,self.document.current)?.records;
        let mut parallel=ParallelHistory::new(&self.document,self.document.current,String::new(),settings.clone())?;
        parallel.initialize_model_input(&self.document,&self.histories)?;
        for cell in 0..16{raw.extend([region_state::marker_record(Stage::Coarse,cell)?,region_state::END_RECORD]);}
        parallel.finish_lane(&self.document,&mut self.histories,0,&raw)?;
        check(checks,"genuine_zero_shape_model_ends_recorded",self.histories.completion_evidence(&self.document,self.document.current)?.regions.iter().filter(|p|p.side==4&&p.primitive_ids.is_empty()).count()==16)?;
        let mut next=ParallelHistory::new(&self.document,self.document.current,String::new(),settings)?;
        next.initialize_model_input(&self.document,&self.histories)?;
        let token=cell_shape(StageTier::Fine,0,500).to_tokens()?.map(u32::from);
        raw.extend([region_state::marker_record(Stage::Fine,0)?,token]);
        next.submit(&mut self.document,&mut self.histories,0,token,&raw)?;raw.push(region_state::END_RECORD);
        next.finish_lane(&self.document,&mut self.histories,0,&raw)?;
        let fine_id=self.document.primitives.last().unwrap().id;
        let original=self.document.primitives.clone();let original_node=self.document.current;
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:true});
        check(checks,"switch_preserves_all_geometry",self.is_redraw()&&same_visible(&self.document.primitives,&original))?;
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:false});
        check(checks,"fine_above_genuinely_ended_empty_coarse_is_legal",!self.is_redraw()&&same_visible(&self.document.primitives,&original))?;
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:true});
        let mut candidate=self.document.primitives.clone();candidate.push(cell_shape(StageTier::Coarse,0,600));
        self.action(ui::UiAction::CommitCanvas{label:"人为加入粗格图形".into(),primitives:candidate});
        let coarse_id=self.document.primitives.last().unwrap().id;
        let candidate=self.document.primitives.iter().filter(|p|p.id!=coarse_id).cloned().collect();
        self.action(ui::UiAction::CommitCanvas{label:"删除粗格图形但保留细格".into(),primitives:candidate});
        let before_nodes=serde_json::to_vec(&self.document.nodes).unwrap();let before=self.document.primitives.clone();let current=self.document.current;
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:false});
        check(checks,"missing_coarse_with_fine_blocks_without_mutation",self.is_redraw()&&self.document.current==current&&self.document.primitives==before
            &&serde_json::to_vec(&self.document.nodes).unwrap()==before_nodes)?;
        let proof=self.histories.completion_evidence(&self.document,self.document.current)?;
        let errors=mode_transition::prepare_paint_resume(&self.document,&self.histories,self.document.current,&proof).unwrap_err();
        check(checks,"report_identifies_coarse_row_and_column",errors.issues.iter().any(|i|i.kind==mode_transition::TransitionIssueKind::CoarseIncompleteBeforeFine&&i.row_col()==Some((1,1))))?;
        check(checks,"old_empty_end_does_not_revive_after_membership_changed",!proof.regions.iter().any(|p|p.side==4&&p.index==0))?;
        self.editor.dismiss_mode_issues();
        let mut candidate=self.document.primitives.clone();candidate.extend((0..20).map(|i|cell_shape(StageTier::Coarse,0,700+i)));
        self.action(ui::UiAction::CommitCanvas{label:"手动补齐粗格配额".into(),primitives:candidate});
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:false});
        check(checks,"repair_allows_conversion_without_new_save",!self.is_redraw()&&self.save_session.active.as_ref().unwrap().id==id)?;
        check(checks,"older_branch_unmodified",same_visible(&self.document.materialize(original_node)?,&original))?;
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:true});
        self.action(ui::UiAction::CommitCanvas{label:"移除全部细阶段".into(),primitives:self.document.primitives.iter().filter(|p|p.id!=fine_id).cloned().collect()});
        let coarse_ids=self.document.primitives.iter().filter(|p|p.stage_assignment.is_some_and(|a|a.stage==StageTier::Coarse)).map(|p|p.id).collect::<Vec<_>>();
        let candidate=self.document.primitives.iter().filter(|p|!coarse_ids.iter().skip(3).any(|id|*id==p.id)).cloned().collect();
        self.action(ui::UiAction::CommitCanvas{label:"粗格保留三个图元".into(),primitives:candidate});
        self.action(ui::UiAction::SwitchWorkspaceMode{redraw:false});
        check(checks,"earlier_valid_frontier_resumes_coarse",!self.is_redraw()&&self.histories.stage_floor(&self.document,self.document.current)?==Some(4))?;
        let entry=self.save_session.active.clone().unwrap();
        let outcome=save_session::perform(save_session::LibraryWork::Save{library:library.clone(),entry,document:std::mem::take(&mut self.document),histories:std::mem::take(&mut self.histories)});
        self.poll_library_outcome(outcome);
        if let Some(error)=&self.save_session.error{return Err(error.clone());}
        let (loaded,history)=project_file::load_project(&library.project_path(&id)?)?;
        check(checks,"same_tree_and_resume_boundary_save_load",loaded.nodes.len()==self.document.nodes.len()&&loaded.primitives==self.document.primitives
            &&history.stage_floor(&loaded,loaded.current)?==Some(4)&&library.list()?.entries.len()==1)?;
        self.status="同一存档双向切换验收通过（未加载模型）".into();self.error=None;self.auto_load_pending=false;self.stage_map_node=None;
        self.editor.show_canvas();self.refresh_drawing_region();self.refresh_redraw_ui();Ok(())
    }
}
