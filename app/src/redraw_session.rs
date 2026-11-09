use super::*;
use crate::{region_state::Stage,workspace::WorkspaceMode};

impl PainterApp{
    pub(super) fn is_redraw(&self)->bool{self.save_session.active.as_ref().is_some_and(|e|e.workspace.mode==WorkspaceMode::Redraw)}
    fn redraw_selection(&self)->redraw::RedrawSelection{self.save_session.active.as_ref().map(|e|e.workspace.selection.clone()).unwrap_or_default()}
    fn set_redraw_selection(&mut self,selection:redraw::RedrawSelection){
        if let Some(entry)=&mut self.save_session.active{entry.workspace.selection=selection;}
        self.redraw_view_key=None;self.save_session.mark_workspace_dirty();
    }
    pub(super) fn toggle_redraw_region(&mut self,stage:Stage,index:usize){
        if !self.is_redraw()||self.running||self.loading||self.compiling||self.stopping{return;}
        let result=(||{
            let map=self.histories.stage_map(&self.document,self.document.current)?;let mut selection=self.redraw_selection();
            let floor=self.histories.stage_floor(&self.document,self.document.current)?.unwrap_or(1);
            selection.toggle_at_floor(stage,index,&self.document.primitives,&map,floor)?;Ok::<_,String>(selection)
        })();
        match result{Ok(selection)=>{self.set_redraw_selection(selection);self.error=None;},Err(error)=>self.editor.notify(error)}
    }
    pub(super) fn delete_redraw_regions(&mut self){
        if !self.is_redraw()||self.running||self.loading||self.compiling||self.stopping{return;}
        let result=(||{
            let map=self.histories.stage_map(&self.document,self.document.current)?;
            let mut selection=self.redraw_selection();
            selection.observe_at_floor(&self.document.primitives,&map,self.histories.stage_floor(&self.document,self.document.current)?.unwrap_or(1))?;
            let canvas=selection.delete_canvas(&self.document.primitives,&map)?;Ok::<_,String>((selection,canvas))
        })();
        match result{
            Ok((selection,canvas))=>{
                // Selection remains latched: red fine children are still part
                // of the required redraw after their old contents are removed.
                self.set_redraw_selection(selection);
                let count=self.document.primitives.len().saturating_sub(canvas.len());
                if count==0{self.editor.notify("所选区域已为空");return;}
                self.commit_stage_edit(format!("手动清空所选区域 · 删除{count}个图元"),canvas,None,false);
                self.redraw_view_key=None;
            },Err(error)=>self.editor.notify(error),
        }
    }
    pub(super) fn delete_locked_redraw_regions(&mut self){
        if !self.is_redraw()||self.running||self.loading||self.compiling||self.stopping{return;}
        let result=(||{
            let map=self.histories.stage_map(&self.document,self.document.current)?;
            let mut selection=self.redraw_selection();
            selection.observe_at_floor(&self.document.primitives,&map,self.histories.stage_floor(&self.document,self.document.current)?.unwrap_or(1))?;
            let canvas=selection.delete_locked_canvas(&self.document.primitives,&map)?;Ok::<_,String>((selection,canvas))
        })();
        match result{Ok((selection,canvas))=>{
            self.set_redraw_selection(selection);let count=self.document.primitives.len().saturating_sub(canvas.len());
            if count==0{self.editor.notify("红色锁定区域已为空");return;}
            self.commit_stage_edit(format!("手动清空红色区域 · 删除{count}个图元"),canvas,None,false);self.redraw_view_key=None;
        },Err(error)=>self.editor.notify(error)}
    }
    pub(super) fn refresh_redraw_ui(&mut self){
        if !self.is_redraw(){self.editor.set_redraw_model(None);self.redraw_view_key=None;return;}
        let mut selection=self.redraw_selection();
        if self.redraw_view_key.as_ref()==Some(&(self.document.current,selection.clone())){return;}
        let result=(||{
            let map=self.histories.stage_map(&self.document,self.document.current)?;
            let prior=selection.clone();
            selection.observe_at_floor(&self.document.primitives,&map,self.histories.stage_floor(&self.document,self.document.current)?.unwrap_or(1))?;
            if selection!=prior{self.set_redraw_selection(selection.clone());}
            let stages=redraw::stage_canvases(&self.document.primitives,&map)?;
            let can_global=self.can_continue_reference_global();
            let reason=if can_global{None}else{selection.validate_empty(&self.document.primitives,&map).err()};
            Ok::<_,String>(ui::redraw_ui::RedrawUiModel{
                global:stages.global,coarse:stages.coarse,fine:stages.fine,
                coarse_counts:stages.coarse_counts,fine_counts:stages.fine_counts,
                selected_global:selection.global_selected(),locked_coarse:selection.locked_coarse(),
                selected_coarse:selection.selected_coarse(),selected_fine:selection.selected_fine(),locked_fine:selection.locked_fine(),
                can_generate:reason.is_none(),can_continue_global:can_global||selection.global_selected(),reason,
            })
        })();
        match result{
            Ok(view)=>{self.editor.set_redraw_model(Some(view));self.redraw_view_key=Some((self.document.current,selection));},
            Err(error)=>{self.error=Some(error);self.editor.set_redraw_model(None);},
        }
    }
    pub(super) fn start_redraw_generation(&mut self,prompt:String,settings:document::Settings){
        if self.can_continue_reference_global(){self.start_reference_global_generation(prompt,settings);return;}
        let result=(||{
            // Validate before changing any geometry, seed, context, or history.
            let plan=redraw_context::prepare_redraw(&self.document,&self.histories,self.document.current,&self.redraw_selection())?;
            settings.validate()?;
            inference::prompt_validation::PromptValidator::open(&self.tokenizer_path)
                .and_then(|validator|validator.validate(&prompt).into_token_ids())?;
            let settings=settings_for_generation(settings)?;
            Ok::<_,String>((plan,settings))
        })();
        let (plan,settings)=match result{Ok(data)=>data,Err(error)=>{self.editor.notify(error);return;}};
        let previous_prompt=self.document.prompt.clone();let previous_settings=self.document.settings.clone();
        self.document.prompt=prompt;self.document.settings=settings;
        let node=match self.document.commit_control_canvas("准备局部重绘",plan.canvas){
            Ok(node)=>node,Err(error)=>{self.document.prompt=previous_prompt;self.document.settings=previous_settings;self.editor.notify(error);return;}
        };
        if let Err(error)=self.histories.set_control_overlay_with_ids(&self.document,node,&plan.records,&plan.primitive_ids){
            let _=self.document.rollback_last_canvas(node);self.document.prompt=previous_prompt;self.document.settings=previous_settings;
            self.editor.notify(error);return;
        }
        if let Err(error)=self.histories.begin_redraw_origin(&self.document,node,&plan.regions,&plan.records,&plan.primitive_ids){self.editor.notify(error);return;}
        let result=(||{
            let mut parallel=ParallelHistory::new(&self.document,node,self.document.prompt.clone(),self.document.settings.clone())?;
            parallel.initialize_explicit_model_input(&self.document,&self.histories,&plan.records,&plan.primitive_ids,plan.pending_stage_ids,plan.regions.clone())?;
            let by_id=self.document.primitives.iter().map(|p|(p.id,p)).collect::<std::collections::HashMap<_,_>>();
            let primitives=plan.primitive_ids.iter().map(|id|by_id.get(id).map(|p|(*p).clone()).ok_or("重绘上下文引用缺失图元".to_owned())).collect::<Result<Vec<_>,_>>()?;
            Ok::<_,String>((parallel,primitives))
        })();
        let (parallel,primitives)=match result{Ok(data)=>data,Err(error)=>{self.editor.notify(error);return;}};
        self.parallel=Some(parallel);self.running=true;self.stopping=false;self.error=None;
        self.stage_map_node=None;self.redraw_view_key=None;self.save_session.mark_dirty();
        diagnostics::event("redraw_generation_requested",serde_json::json!({"regions":plan.regions,"global_and_unselected_primitives_preserved":true,
            "automatic_deletion":false,"automatic_reorder_skipped":self.document.settings.auto_reorder_before_generation}));
        self.runtime.generate_redraw(self.document.prompt.clone(),self.document.settings.clone(),primitives,plan.records,
            plan.stage_floor,plan.pending_stage_context,plan.regions,None);
        self.status="正在续画黄色区域并重绘已清空的红色区域 · 已有图元保留".into();
    }
}
