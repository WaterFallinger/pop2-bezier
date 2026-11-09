//! Isolated native acceptance: idle edits never write or start a blocking job.
use super::*;
use crate::manual_save::LeaveAction;
use std::time::{Duration,Instant};

impl PainterApp{
    fn manual_save_check(&mut self,name:&str,ok:bool)->Result<(),String>{
        if !ok{return Err(format!("手动保存验收失败：{name}"));}self.manual_save_smoke_checks.push(name.into());Ok(())
    }
    fn manual_smoke_recolor(&mut self,color:[u8;3]){
        let mut canvas=self.document.primitives.clone();canvas.last_mut().unwrap().color=color;
        self.action(ui::UiAction::CommitCanvas{label:"验收改色".into(),primitives:canvas});
    }
    pub(super) fn drive_manual_save_smoke(&mut self){
        let args=std::env::args().collect::<Vec<_>>();
        let Some(path)=args.iter().position(|a|a=="--manual-save-smoke-output").and_then(|i|args.get(i+1)).map(PathBuf::from)else{return;};
        self.auto_load_pending=false;if self.frames<2||self.manual_save_smoke_stage>=9{return;}
        self.file_repaint.request_repaint_after(Duration::from_millis(60));
        let result=self.manual_save_smoke_step();
        if result.is_err()||self.manual_save_smoke_stage>=9{
            let report=serde_json::json!({"status":if result.is_ok(){"passed"}else{"failed"},"error":result.err(),
                "checks":self.manual_save_smoke_checks,"model_loaded":self.loaded,"auto_load_pending":self.auto_load_pending,
                "automatic_project_writes_observed":0,"explicit_successful_project_saves":2,
                "rasterize_canvas":self.editor.canvas.rasterize_canvas,
                "scope":"native controller, real file workers, synthetic canvas; no model load"});
            self.manual_save_smoke_stage=9;
            if let Some(parent)=path.parent(){let _=std::fs::create_dir_all(parent);}
            let _=std::fs::write(path,serde_json::to_vec_pretty(&report).unwrap());
        }
    }
    fn manual_save_smoke_step(&mut self)->Result<(),String>{
        if self.file_busy_message.is_some(){return Ok(());}
        let library=self.save_session.catalog.as_ref().map_err(Clone::clone)?.clone();
        match self.manual_save_smoke_stage{
            0=>{
                self.poll_library_outcome(save_session::perform(save_session::LibraryWork::Create{library:library.clone(),name:"手动保存验收".into()}));
                self.manual_save_smoke_id=self.save_session.active.as_ref().unwrap().id.clone();
                self.manual_save_smoke_disk=std::fs::read(library.project_path(&self.manual_save_smoke_id)?).map_err(|e|e.to_string())?;
                self.action(ui::UiAction::CommitCanvas{label:"画布编辑".into(),primitives:vec![Primitive::background([240,242,248]),
                    Primitive::new(ShapeKind::Ellipse,256.,256.,220.,260.,[100,160,220])]});
                self.manual_save_smoke_started=Some(Instant::now());self.manual_save_smoke_stage=1;
            },
            1=>{
                if self.manual_save_smoke_started.unwrap().elapsed()<Duration::from_millis(2100){return Ok(());}
                let bytes=std::fs::read(library.project_path(&self.manual_save_smoke_id)?).map_err(|e|e.to_string())?;
                self.manual_save_check("idle_edit_remains_live_dirty_without_worker_or_disk_write",self.document.primitives.len()==2&&self.save_session.dirty&&self.file_task.is_none()&&bytes==self.manual_save_smoke_disk)?;
                self.request_leave(LeaveAction::Home);self.resolve_unsaved_choice(ui::UnsavedChoice::Cancel);
                self.manual_save_check("cancel_home_keeps_document_and_does_not_save",!self.save_session.home&&self.save_session.dirty&&self.file_task.is_none()&&self.save_session.pending_leave.is_none())?;
                let mut canvas=self.document.primitives.clone();canvas.last_mut().unwrap().color=[60,120,180];
                self.pending_canvas_edit=Some((self.document.current,ui::UiAction::CommitCanvas{label:"排队的用户改色".into(),primitives:canvas}));
                self.running=true;self.stopping=false;
                self.request_leave(LeaveAction::Close);self.resolve_unsaved_choice(ui::UnsavedChoice::Cancel);
                self.manual_save_check("cancel_close_neither_stops_model_nor_erases_queued_edit",self.running&&!self.stopping&&self.pending_canvas_edit.is_some()&&!self.save_session.approved_close)?;
                self.running=false;self.finish_pending_edit();
                let path=library.project_path(&self.manual_save_smoke_id)?;self.action(ui::UiAction::SaveProject(path));self.manual_save_smoke_stage=2;
            },
            2=>{
                self.manual_save_check("explicit_save_persists_latest_edit_and_clears_unsaved",!self.save_session.dirty&&self.document.primitives.last().unwrap().color==[60,120,180])?;
                self.manual_save_smoke_disk=std::fs::read(library.project_path(&self.manual_save_smoke_id)?).map_err(|e|e.to_string())?;
                self.manual_smoke_recolor([110,140,180]);
                self.request_leave(LeaveAction::Home);self.resolve_unsaved_choice(ui::UnsavedChoice::Discard);
                let bytes=std::fs::read(library.project_path(&self.manual_save_smoke_id)?).map_err(|e|e.to_string())?;
                self.manual_save_check("discard_home_keeps_saved_file_unchanged",self.save_session.home&&self.save_session.active.is_none()&&bytes==self.manual_save_smoke_disk)?;
                self.library_action(save_library_ui::Action::Open{id:self.manual_save_smoke_id.clone()});self.manual_save_smoke_stage=3;
            },
            3=>{
                self.manual_save_check("reopen_restores_last_explicit_save",!self.save_session.home&&!self.save_session.dirty&&self.document.primitives.last().unwrap().color==[60,120,180])?;
                self.action(ui::UiAction::SetPrompt("尚未保存的提示词".into()));
                self.request_leave(LeaveAction::Import(library.root().join("missing.pproj")));self.resolve_unsaved_choice(ui::UnsavedChoice::Discard);self.manual_save_smoke_stage=4;
            },
            4=>{
                self.manual_save_check("failed_discard_import_retains_dirty_tree_and_lease",self.error.is_some()&&self.save_session.dirty&&self.document.prompt=="尚未保存的提示词"
                    &&self.save_session.active.as_ref().unwrap().id==self.manual_save_smoke_id&&library.claim(&self.manual_save_smoke_id).is_err())?;
                self.error=None;self.save_session.error=None;
                self.save_session.active.as_mut().unwrap().display_name="\n".into();
                self.request_leave(LeaveAction::Close);self.resolve_unsaved_choice(ui::UnsavedChoice::Save);self.manual_save_smoke_stage=5;
            },
            5=>{
                self.manual_save_check("failed_save_reopens_choice_and_keeps_live_unsaved_data",self.error.is_some()&&self.save_session.dirty&&!self.save_session.approved_close
                    &&self.save_session.pending_leave.as_ref().is_some_and(|p|p.choice.is_none())&&self.document.prompt=="尚未保存的提示词"&&self.document.primitives.len()==2)?;
                for _ in 0..3{self.drive_pending_leave();}
                self.manual_save_check("failed_save_does_not_retry_without_a_new_user_choice",self.file_task.is_none()&&self.file_busy_message.is_none()&&self.save_session.dirty)?;
                self.resolve_unsaved_choice(ui::UnsavedChoice::Cancel);
                self.manual_save_check("cancel_after_failed_save_keeps_edits_and_clears_leave",self.save_session.pending_leave.is_none()&&self.document.prompt=="尚未保存的提示词"&&self.save_session.dirty)?;
                self.request_leave(LeaveAction::Home);self.resolve_unsaved_choice(ui::UnsavedChoice::Discard);
                let bytes=std::fs::read(library.project_path(&self.manual_save_smoke_id)?).map_err(|e|e.to_string())?;
                self.manual_save_check("discard_after_failed_save_leaves_without_saving_or_changing_existing_file",self.save_session.home&&self.file_task.is_none()&&bytes==self.manual_save_smoke_disk)?;
                self.poll_library_outcome(save_session::perform(save_session::LibraryWork::Open{library:library.clone(),id:self.manual_save_smoke_id.clone()}));
                self.action(ui::UiAction::SetPrompt("尚未保存的提示词".into()));self.error=None;self.save_session.error=None;
                let mut canvas=self.document.primitives.clone();canvas.last_mut().unwrap().color=[220,180,80];
                self.pending_canvas_edit=Some((self.document.current,ui::UiAction::CommitCanvas{label:"离开前排队的编辑".into(),primitives:canvas}));
                self.running=true;self.stopping=false;
                self.request_leave(LeaveAction::Home);self.resolve_unsaved_choice(ui::UnsavedChoice::Save);
                self.manual_save_check("approved_save_waits_for_terminal_and_queued_canvas_edit",self.stopping&&self.file_task.is_none()&&self.pending_canvas_edit.is_some())?;
                self.running=false;self.stopping=false;self.finish_pending_edit();
                self.manual_save_check("queued_edit_applied_before_save_owns_document",self.document.primitives.last().unwrap().color==[220,180,80]&&self.pending_canvas_edit.is_none())?;
                self.drive_pending_leave();self.manual_save_smoke_stage=6;
            },
            6=>{
                let (saved,_)=project_file::load_project(&library.project_path(&self.manual_save_smoke_id)?)?;
                self.manual_save_check("save_and_leave_includes_queued_edit_and_prompt",self.save_session.home&&saved.primitives.last().unwrap().color==[220,180,80]&&saved.prompt=="尚未保存的提示词")?;
                self.library_action(save_library_ui::Action::Open{id:self.manual_save_smoke_id.clone()});self.manual_save_smoke_stage=7;
            },
            7=>{
                self.manual_save_check("saved_project_reopens_clean_without_repeated_save",!self.save_session.home&&!self.save_session.dirty&&self.file_task.is_none()
                    &&self.document.primitives.last().unwrap().color==[220,180,80])?;
                if std::env::args().any(|a|a=="--manual-save-smoke-close-failure"){
                    self.manual_save_smoke_disk=std::fs::read(library.project_path(&self.manual_save_smoke_id)?).map_err(|e|e.to_string())?;
                    self.manual_smoke_recolor([200,100,80]);
                    self.save_session.active.as_mut().unwrap().display_name="\n".into();
                    self.request_leave(LeaveAction::Close);self.resolve_unsaved_choice(ui::UnsavedChoice::Save);self.manual_save_smoke_stage=8;
                }else{self.manual_save_smoke_stage=9;self.status="手动保存验收通过 · 编辑与空闲不会自动保存".into();}
            },
            8=>{
                self.manual_save_check("failed_close_save_restores_tree_and_offers_fresh_choice",self.error.is_some()&&self.save_session.dirty
                    &&self.save_session.pending_leave.as_ref().is_some_and(|p|p.action==LeaveAction::Close&&p.choice.is_none())
                    &&self.document.primitives.last().unwrap().color==[200,100,80]&&!self.save_session.approved_close)?;
                let bytes=std::fs::read(library.project_path(&self.manual_save_smoke_id)?).map_err(|e|e.to_string())?;
                self.resolve_unsaved_choice(ui::UnsavedChoice::Discard);
                self.manual_save_check("discard_after_failed_close_save_approves_natural_close_without_disk_write",self.save_session.approved_close
                    &&self.save_session.pending_leave.is_none()&&self.file_task.is_none()&&bytes==self.manual_save_smoke_disk)?;
                self.manual_save_smoke_stage=9;
            },_=>{},
        }Ok(())
    }
}
