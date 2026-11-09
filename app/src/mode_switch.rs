//! A save is one tree. Mode changes create branch-local continuation boundaries,
//! without changing geometry or replacing the save identity.
use super::*;
use crate::workspace::WorkspaceMode;

impl PainterApp{
    pub(super) fn switch_workspace_mode(&mut self,redraw:bool){
        if self.running||self.loading||self.compiling||self.stopping||self.file_busy_message.is_some()
            ||self.save_session.active.is_none(){return;}
        if redraw==self.is_redraw(){return;}
        self.cancel_global_edit();
        if !redraw{self.resume_standard_workflow();return;}
        let result=(||{
            let current=self.document.current;
            let map=self.histories.stage_map(&self.document,current)?;
            let view=history_context::stage_edit::view(&self.document.primitives,&map,false)?;
            let effective=self.histories.effective_history(&self.document,current)?;
            let mut canvas=self.document.primitives.clone();
            for p in &mut canvas{p.stage_assignment=Some(view.assignments[&p.id]);}
            let node=self.document.commit_control_canvas("切换为修改模式",canvas)?;
            self.histories.set_control_overlay_with_ids(&self.document,node,&effective.records,&effective.primitive_ids)?;
            self.histories.begin_redraw_origin(&self.document,node,&[],&effective.records,&effective.primitive_ids)?;
            Ok::<_,String>(())
        })();
        // Standard -> modification has no semantic compatibility gate. Even
        // an unprovable old completion remains editable; it grants no END proof.
        self.save_session.active.as_mut().unwrap().workspace.mode=WorkspaceMode::Redraw;
        self.parallel=None;self.pending_canvas_edit=None;self.stage_map_node=None;self.redraw_view_key=None;
        self.editor.dismiss_mode_issues();self.editor.set_workflow_mode(true);self.editor.invalidate_history_after_deletion();
        self.refresh_drawing_region();self.refresh_redraw_ui();self.save_session.mark_dirty();
        self.status="已切换为修改模式 · 当前存档和画布保持不变".into();
        if let Err(error)=result{self.editor.notify(format!("已进入修改模式；部分旧区域完成记录暂无法确认：{error}"));}
        diagnostics::event("workspace_mode_switched",serde_json::json!({"mode":"redraw","same_save":true,"node":self.document.current}));
    }

    pub(super) fn resume_standard_workflow(&mut self)->bool{
        let evidence=match self.histories.completion_evidence(&self.document,self.document.current){
            Ok(value)=>value,Err(error)=>{self.editor.show_mode_issues(vec![format!("无法确认当前区域的真实结束记录：{error}")]);return false;},
        };
        let plan=match mode_transition::prepare_paint_resume(&self.document,&self.histories,self.document.current,&evidence){
            Ok(plan)=>plan,
            Err(report)=>{
                let issues=report.issues.into_iter().map(|issue|issue.message).collect::<Vec<_>>();
                diagnostics::event("workspace_mode_switch_rejected",serde_json::json!({"issues":issues,"node":self.document.current}));
                self.editor.show_mode_issues(issues);return false;
            },
        };
        let result=(||{
            let node=self.document.commit_control_canvas("切换为标准模式",plan.canvas)?;
            self.histories.restart_normal_history(&self.document,node,&plan.records,&plan.primitive_ids,plan.stage_floor.unwrap_or(1),&plan.manual_review_regions)?;
            Ok::<_,String>(())
        })();
        if let Err(error)=result{self.editor.show_mode_issues(vec![format!("无法安全恢复标准绘制上下文，当前画布已保留：{error}")]);return false;}
        if let Some(entry)=&mut self.save_session.active{entry.workspace.mode=WorkspaceMode::Paint;}
        self.parallel=None;self.pending_canvas_edit=None;self.stage_map_node=None;self.redraw_view_key=None;
        self.editor.dismiss_mode_issues();self.editor.set_workflow_mode(false);self.editor.invalidate_history_after_deletion();
        self.refresh_drawing_region();self.refresh_redraw_ui();self.save_session.mark_dirty();self.error=None;
        self.status="检查通过，已切换为标准模式 · 继续使用当前存档".into();
        diagnostics::event("workspace_mode_switched",serde_json::json!({"mode":"paint","same_save":true,"node":self.document.current}));
        true
    }
}
