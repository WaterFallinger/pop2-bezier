//! Saving is explicit. Ordinary edits, idle time and model callbacks never
//! transfer the document to a file worker or display a blocking save window.
use super::*;

#[derive(Clone,Debug,PartialEq,Eq)]
pub(super) enum LeaveAction{Home,Close,Import(PathBuf)}
impl LeaveAction{fn label(&self)->&'static str{match self{Self::Home=>"返回存档页",Self::Close=>"关闭窗口",Self::Import(_)=>"导入其他状态树"}}}
pub(super) struct PendingLeave{pub action:LeaveAction,pub choice:Option<ui::UnsavedChoice>,prompted:bool}

impl PainterApp{
    pub(super) fn request_leave(&mut self,action:LeaveAction){
        if self.save_session.active.is_none(){self.execute_leave(action);return;}
        match &mut self.save_session.pending_leave{
            Some(pending)=>{
                if pending.action!=action{pending.action=action;pending.prompted=false;}
            },
            None=>self.save_session.pending_leave=Some(PendingLeave{action,choice:None,prompted:false}),
        }
        self.drive_pending_leave();
    }
    pub(super) fn resolve_unsaved_choice(&mut self,choice:ui::UnsavedChoice){
        self.editor.clear_unsaved_changes();
        if choice==ui::UnsavedChoice::Cancel{
            self.save_session.pending_leave=None;self.save_session.approved_close=false;
            // In particular, do not stop generation or erase a queued edit.
            return;
        }
        let Some(pending)=&mut self.save_session.pending_leave else{return;};pending.choice=Some(choice);
        self.cancel_global_edit();
        if self.running{self.runtime.stop();self.stopping=true;self.status="正在完成停止，请稍候…".into();}
        self.drive_pending_leave();
    }
    pub(super) fn drive_pending_leave(&mut self){
        if self.file_busy_message.is_some(){return;}
        let Some(pending)=&mut self.save_session.pending_leave else{return;};
        if pending.choice.is_none(){
            if self.save_session.dirty||self.running||self.stopping||self.pending_canvas_edit.is_some(){
                if !pending.prompted{self.editor.show_unsaved_changes(pending.action.label());pending.prompted=true;}
                return;
            }
            let action=self.save_session.pending_leave.take().unwrap().action;self.execute_leave(action);return;
        }
        // Terminal callbacks and any user edit queued before that stop must
        // enter the tree before an explicit save takes document ownership.
        if self.running||self.stopping||self.pending_canvas_edit.is_some(){return;}
        match pending.choice.unwrap(){
            ui::UnsavedChoice::Save=>self.begin_library_save(),
            ui::UnsavedChoice::Discard=>{
                let action=self.save_session.pending_leave.take().unwrap().action;self.execute_leave(action);
            },ui::UnsavedChoice::Cancel=>unreachable!(),
        }
    }
    pub(super) fn finish_saved_leave(&mut self){
        if self.save_session.pending_leave.as_ref().is_some_and(|p|p.choice==Some(ui::UnsavedChoice::Save)){
            let action=self.save_session.pending_leave.take().unwrap().action;self.execute_leave(action);
        }
    }
    pub(super) fn recover_failed_save_leave(&mut self,action:LeaveAction,error:&str){
        // A failed explicit save is not a new approval to retry. Keep the
        // original leave destination, but require a fresh user decision and
        // always offer the non-saving escape route.
        self.save_session.pending_leave=Some(PendingLeave{action:action.clone(),choice:None,prompted:true});
        self.save_session.pending_save=false;self.save_session.approved_close=false;
        self.editor.show_failed_save_leave(action.label(),error);
    }
    fn execute_leave(&mut self,action:LeaveAction){
        self.editor.clear_unsaved_changes();
        match action{
            LeaveAction::Home=>{self.editor.dismiss_mode_issues();self.leave_saved_document();},
            LeaveAction::Close=>{self.save_session.approved_close=true;self.file_repaint.send_viewport_cmd(egui::ViewportCommand::Close);},
            // The old document and its lease stay alive until this isolated
            // import succeeds, even after the user approved discarding it.
            LeaveAction::Import(path)=>self.begin_library_import(path),
        }
    }
    pub(super) fn guard_library_close(&mut self,ctx:&egui::Context){
        if !self.save_session.enabled||self.save_session.active.is_none()||self.save_session.approved_close{return;}
        if ctx.input(|i|i.viewport().close_requested()){
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);self.request_leave(LeaveAction::Close);
        }
    }
}
