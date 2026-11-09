//! Local-redraw selections are separate from primitive selection and ownership.
//! A parent's occupied fine children stay latched after deletion so they are
//! still part of the redraw request. Removing the parent only removes its lock.
use crate::{document::{Primitive,ShapeKind},history_context::stage_edit,region_state::Stage};
use serde::{Deserialize,Serialize};
use std::collections::{BTreeMap,BTreeSet};

#[derive(Clone,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(default,deny_unknown_fields)]
pub struct RedrawSelection{
    global_selected:bool,
    manual_coarse:BTreeSet<usize>,
    manual_fine:BTreeSet<usize>,
    locked_children:BTreeMap<usize,BTreeSet<usize>>,
    global_locked_coarse:BTreeSet<usize>,
    global_locked_fine:BTreeSet<usize>,
}

#[derive(Clone,Debug)]
pub struct StageCanvases{
    pub global:Vec<Primitive>,pub coarse:Vec<Primitive>,pub fine:Vec<Primitive>,
    pub coarse_counts:[usize;16],pub fine_counts:[usize;64],
}
impl Default for StageCanvases{fn default()->Self{Self{global:Vec::new(),coarse:Vec::new(),fine:Vec::new(),
    coarse_counts:[0;16],fine_counts:[0;64]}}}

pub fn fine_children(coarse:usize)->Result<[usize;4],String>{
    if coarse>=16{return Err("粗网格编号无效".into());}
    let first=(coarse/4)*16+(coarse%4)*2;Ok([first,first+1,first+8,first+9])
}

pub fn stage_canvases(canvas:&[Primitive],legacy:&BTreeMap<u64,Stage>)->Result<StageCanvases,String>{
    let view=stage_edit::view(canvas,legacy,false)?;let mut result=StageCanvases::default();
    for primitive in canvas{
        let a=view.assignments.get(&primitive.id).ok_or("图元缺少阶段归属")?;
        match stage_edit::stage(a.stage){
            Stage::Global=>result.global.push(primitive.clone()),
            Stage::Coarse=>{result.coarse_counts[a.cell]+=1;result.coarse.push(primitive.clone());},
            Stage::Fine=>{result.fine_counts[a.cell]+=1;result.fine.push(primitive.clone());},
            Stage::Complete=>return Err("图元归属不能是绘制完成状态".into()),
        }
    }
    Ok(result)
}

impl RedrawSelection{
    /// Fresh selection for an explicitly imported reference. Deferred fine
    /// geometry is selected for continuation, not latched as a dependency.
    pub fn from_reference_regions(regions:&[(usize,usize)])->Result<Self,String>{
        let mut result=Self::default();
        for &(side,index)in regions{match side{
            1 if index==0=>{result.global_selected=true;},
            4 if index<16=>{result.manual_coarse.insert(index);},
            8 if index<64=>{result.manual_fine.insert(index);},
            _=>return Err("参考续画区域编号无效".into()),
        }}
        result.validate()?;Ok(result)
    }
    pub fn validate(&self)->Result<(),String>{
        if self.manual_coarse.len()>16||self.manual_coarse.iter().any(|i|*i>=16)
            ||self.manual_fine.len()>64||self.manual_fine.iter().any(|i|*i>=64)
            ||self.locked_children.len()>16
            ||self.global_locked_coarse.len()>16||self.global_locked_coarse.iter().any(|i|*i>=16)
            ||self.global_locked_fine.len()>64||self.global_locked_fine.iter().any(|i|*i>=64){return Err("局部重绘选择超出网格范围".into());}
        if !self.global_selected&&(!self.global_locked_coarse.is_empty()||!self.global_locked_fine.is_empty()){
            return Err("粗细网格的全图锁定缺少被选择的全图阶段".into());
        }
        for (&parent,children) in &self.locked_children{
            if !self.manual_coarse.contains(&parent){return Err("细网格锁定缺少被选择的父粗网格".into());}
            let legal=fine_children(parent)?;
            if children.len()>4||children.iter().any(|i|!legal.contains(i)){return Err("细网格锁定不属于对应粗网格".into());}
        }
        Ok(())
    }
    pub fn global_selected(&self)->bool{self.global_selected}
    pub fn selected_coarse(&self)->Vec<usize>{self.manual_coarse.iter().copied().collect()}
    /// Manual yellow selections. A simultaneously locked entry is rendered red.
    pub fn selected_fine(&self)->Vec<usize>{self.manual_fine.iter().copied().collect()}
    pub fn locked_coarse(&self)->Vec<usize>{self.global_locked_coarse.iter().copied().collect()}
    pub fn locked_fine(&self)->Vec<usize>{self.locked_children.values().flat_map(|v|v.iter().copied())
        .chain(self.global_locked_fine.iter().copied()).collect::<BTreeSet<_>>().into_iter().collect()}
    pub fn regions(&self)->Vec<(usize,usize)>{
        let mut regions=if self.global_selected{vec![(1,0)]}else{Vec::new()};
        regions.extend(self.manual_coarse.iter().copied().chain(self.locked_coarse()).collect::<BTreeSet<_>>().into_iter().map(|i|(4,i)));
        regions.extend(self.manual_fine.iter().copied().chain(self.locked_fine()).collect::<BTreeSet<_>>().into_iter().map(|i|(8,i)));regions
    }
    pub fn toggle(&mut self,stage:Stage,index:usize,canvas:&[Primitive],legacy:&BTreeMap<u64,Stage>)->Result<bool,String>{
        self.toggle_at_floor(stage,index,canvas,legacy,8)
    }
    pub fn toggle_at_floor(&mut self,stage:Stage,index:usize,canvas:&[Primitive],legacy:&BTreeMap<u64,Stage>,floor:usize)->Result<bool,String>{
        self.validate()?;
        match stage{
            Stage::Global if index==0=>{
                if self.global_selected{
                    self.global_selected=false;self.global_locked_coarse.clear();self.global_locked_fine.clear();
                }else{self.global_selected=true;self.observe_at_floor(canvas,legacy,floor)?;}
            },
            Stage::Coarse if index<16=>{
                if self.global_locked_coarse.contains(&index){return Err("该粗网格由已选全图阶段锁定，请先取消全图阶段的选择".into());}
                if self.manual_coarse.remove(&index){self.locked_children.remove(&index);}
                else{self.manual_coarse.insert(index);self.observe_at_floor(canvas,legacy,floor)?;}
            },
            Stage::Fine if index<64=>{
                if self.global_locked_fine.contains(&index){return Err("该细网格由已选全图阶段锁定，请先取消全图阶段的选择".into());}
                if self.locked_fine().contains(&index){return Err("该细网格由已选粗网格锁定，请先取消对应粗网格的选择".into());}
                if !self.manual_fine.remove(&index){self.manual_fine.insert(index);}
            },
            _=>return Err("局部重绘网格编号无效".into()),
        }
        Ok(true)
    }
    /// Observe newly imported/drawn contents, without forgetting already latched
    /// locks when those contents were manually removed or the history is browsed.
    pub fn observe(&mut self,canvas:&[Primitive],legacy:&BTreeMap<u64,Stage>)->Result<(),String>{
        self.observe_at_floor(canvas,legacy,8)
    }
    /// Future-owned fine reference geometry has not yet entered the model's
    /// drawing history. It must not become a destructive red dependency while
    /// the model is still finishing coarse cells. Existing latched locks stay.
    pub fn observe_at_floor(&mut self,canvas:&[Primitive],legacy:&BTreeMap<u64,Stage>,floor:usize)->Result<(),String>{
        self.validate()?;let stages=stage_canvases(canvas,legacy)?;
        // Explicitly revisiting Global invalidates all finer conditioning,
        // including future-owned reference data not yet admitted to the model.
        if self.global_selected{
            self.global_locked_coarse.extend(stages.coarse_counts.iter().enumerate().filter_map(|(i,n)|(*n>0).then_some(i)));
            self.global_locked_fine.extend(stages.fine_counts.iter().enumerate().filter_map(|(i,n)|(*n>0).then_some(i)));
        }
        if floor<8{return Ok(());}
        for parent in &self.manual_coarse{
            let children=self.locked_children.entry(*parent).or_default();
            for child in fine_children(*parent)?{if stages.fine_counts[child]>0{children.insert(child);}}
        }
        Ok(())
    }
    /// Yellow manual selections are continuation requests. Every forced red
    /// descendant must be cleared before its selected ancestor can be redrawn.
    pub fn validate_empty(&self,canvas:&[Primitive],legacy:&BTreeMap<u64,Stage>)->Result<(),String>{
        self.validate()?;let regions=self.regions();
        if regions.is_empty(){return Err("请先选择要续画的全图、粗网格或细网格区域".into());}
        let stages=stage_canvases(canvas,legacy)?;let mut count=0;let mut coarse_cells=0;let mut fine_cells=0;
        for index in self.locked_coarse(){let n=stages.coarse_counts[index];count+=n;if n>0{coarse_cells+=1;}}
        for index in self.locked_fine(){let n=stages.fine_counts[index];count+=n;if n>0{fine_cells+=1;}}
        if count>0{return Err(format!("红色锁定的{coarse_cells}个粗网格、{fine_cells}个细网格中仍有{count}个图形。请先手动清空所有红色区域，再开始绘制；黄色区域保留已有图形并续画，程序不会自动删除。"));}
        Ok(())
    }
    pub fn delete_canvas(&self,canvas:&[Primitive],legacy:&BTreeMap<u64,Stage>)->Result<Vec<Primitive>,String>{
        self.validate()?;let regions=self.regions().into_iter().collect::<BTreeSet<_>>();
        if regions.is_empty(){return Err("请先选择要清空的全图、粗网格或细网格区域".into());}
        let view=stage_edit::view(canvas,legacy,false)?;
        Ok(canvas.iter().filter(|p|{
            let a=view.assignments[&p.id];p.shape==ShapeKind::Background||!regions.contains(&(a.stage.side(),a.cell))
        }).cloned().collect())
    }
    pub fn delete_locked_canvas(&self,canvas:&[Primitive],legacy:&BTreeMap<u64,Stage>)->Result<Vec<Primitive>,String>{
        self.validate()?;let locked_coarse=self.locked_coarse().into_iter().collect::<BTreeSet<_>>();
        let locked_fine=self.locked_fine().into_iter().collect::<BTreeSet<_>>();
        let view=stage_edit::view(canvas,legacy,false)?;
        Ok(canvas.iter().filter(|p|{
            let a=view.assignments[&p.id];p.shape==ShapeKind::Background
                ||!((a.stage.side()==4&&locked_coarse.contains(&a.cell))||(a.stage.side()==8&&locked_fine.contains(&a.cell)))
        }).cloned().collect())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::document::{Document,OperationSource,PrimitiveStageAssignment,StageTier};
    fn primitive(stage:StageTier,cell:usize,x:f32)->Primitive{
        let mut p=Primitive::new(ShapeKind::Rectangle,x,30.,12.,10.,[40,80,120]);
        p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:cell as u64});p
    }
    fn fixture()->Document{
        let mut doc=Document::new();doc.commit_canvas("mixed",OperationSource::User,vec![Primitive::background([240;3]),
            primitive(StageTier::Global,0,20.),primitive(StageTier::Coarse,5,20.),primitive(StageTier::Coarse,4,20.),
            primitive(StageTier::Fine,18,20.),primitive(StageTier::Fine,27,20.),primitive(StageTier::Fine,19,20.),primitive(StageTier::Fine,63,20.)]).unwrap();doc
    }
    #[test]
    fn coarse_children_lock_and_survive_deletion_then_restore_manual_yellow(){
        let doc=fixture();let legacy=BTreeMap::new();let mut selection=RedrawSelection::default();
        selection.toggle(Stage::Fine,18,&doc.primitives,&legacy).unwrap();
        selection.toggle(Stage::Coarse,5,&doc.primitives,&legacy).unwrap();
        assert_eq!(fine_children(5).unwrap(),[18,19,26,27]);assert_eq!(selection.locked_fine(),[18,19,27]);
        assert!(selection.toggle(Stage::Fine,18,&doc.primitives,&legacy).is_err());
        assert!(selection.validate_empty(&doc.primitives,&legacy).is_err());
        let emptied=selection.delete_canvas(&doc.primitives,&legacy).unwrap();selection.observe(&emptied,&legacy).unwrap();
        assert_eq!(selection.locked_fine(),[18,19,27]);selection.validate_empty(&emptied,&legacy).unwrap();
        selection.toggle(Stage::Coarse,5,&emptied,&legacy).unwrap();assert!(selection.locked_fine().is_empty());
        assert_eq!(selection.selected_fine(),[18]);assert_eq!(selection.regions(),[(8,18)]);
    }
    #[test]
    fn region_deletion_uses_ownership_not_center_or_overlapping_stage(){
        let doc=fixture();let legacy=BTreeMap::new();let mut selection=RedrawSelection::default();
        selection.toggle(Stage::Coarse,5,&doc.primitives,&legacy).unwrap();
        let kept=selection.delete_canvas(&doc.primitives,&legacy).unwrap();
        assert_eq!(kept.len(),4);assert_eq!(kept.iter().map(|p|p.id).collect::<Vec<_>>(),vec![doc.primitives[0].id,doc.primitives[1].id,doc.primitives[3].id,doc.primitives[7].id]);
        // Every shape deliberately has the same center. Its explicit ownership,
        // not its visual location or uppermost painting tier, decides removal.
        assert_eq!(kept[2].stage_assignment.unwrap().cell,4);
    }
    #[test]
    fn red_only_clear_keeps_selected_yellow_coarse_and_unrelated_fine(){
        let doc=fixture();let legacy=BTreeMap::new();let mut selection=RedrawSelection::default();
        selection.toggle(Stage::Coarse,5,&doc.primitives,&legacy).unwrap();
        let kept=selection.delete_locked_canvas(&doc.primitives,&legacy).unwrap();
        assert_eq!(kept.iter().map(|p|p.id).collect::<Vec<_>>(),vec![doc.primitives[0].id,doc.primitives[1].id,
            doc.primitives[2].id,doc.primitives[3].id,doc.primitives[7].id]);
        selection.validate_empty(&kept,&legacy).unwrap();
    }
    #[test]
    fn yellow_manual_cells_may_keep_geometry_and_invalid_persistence_is_rejected(){
        let doc=fixture();let legacy=BTreeMap::new();let mut selection=RedrawSelection::default();
        assert!(selection.validate_empty(&doc.primitives,&legacy).is_err());
        selection.toggle(Stage::Fine,62,&doc.primitives,&legacy).unwrap();selection.validate_empty(&doc.primitives,&legacy).unwrap();
        selection.toggle(Stage::Fine,63,&doc.primitives,&legacy).unwrap();selection.validate_empty(&doc.primitives,&legacy).unwrap();
        selection.toggle(Stage::Fine,63,&doc.primitives,&legacy).unwrap();selection.validate_empty(&doc.primitives,&legacy).unwrap();
        let persisted=serde_json::to_vec(&selection).unwrap();let restored:RedrawSelection=serde_json::from_slice(&persisted).unwrap();
        assert_eq!(restored,selection);restored.validate().unwrap();
        let invalid:RedrawSelection=serde_json::from_value(serde_json::json!({"manual_coarse":[16]})).unwrap();assert!(invalid.validate().is_err());
        let invalid:RedrawSelection=serde_json::from_value(serde_json::json!({"manual_coarse":[5],"locked_children":{"5":[63]}})).unwrap();assert!(invalid.validate().is_err());
        assert!(selection.toggle(Stage::Global,1,&doc.primitives,&legacy).is_err());
    }
    #[test]
    fn later_fine_edits_are_latched_only_under_selected_parents_and_toggle_failures_do_not_unlock(){
        let mut doc=Document::new();let legacy=BTreeMap::new();let mut selection=RedrawSelection::default();
        selection.toggle(Stage::Coarse,0,&[],&legacy).unwrap();assert!(selection.locked_fine().is_empty());
        doc.commit_canvas("fine edits",OperationSource::User,vec![primitive(StageTier::Fine,9,20.),primitive(StageTier::Fine,63,20.)]).unwrap();
        selection.observe(&doc.primitives,&legacy).unwrap();assert_eq!(selection.locked_fine(),[9]);
        assert!(selection.toggle(Stage::Fine,9,&doc.primitives,&legacy).is_err());assert_eq!(selection.locked_fine(),[9]);
        selection.toggle(Stage::Fine,63,&doc.primitives,&legacy).unwrap();assert_eq!(selection.regions(),[(4,0),(8,9),(8,63)]);
        selection.toggle(Stage::Coarse,0,&doc.primitives,&legacy).unwrap();assert_eq!(selection.regions(),[(8,63)]);
    }
    #[test]
    fn deferred_fine_reference_does_not_lock_until_it_is_actually_admitted(){
        let doc=fixture();let legacy=BTreeMap::new();let mut selection=RedrawSelection::default();
        selection.toggle_at_floor(Stage::Coarse,5,&doc.primitives,&legacy,4).unwrap();
        assert!(selection.locked_fine().is_empty());
        selection.validate_empty(&doc.primitives,&legacy).unwrap();
        assert_eq!(selection.delete_locked_canvas(&doc.primitives,&legacy).unwrap(),doc.primitives);
        selection.observe_at_floor(&doc.primitives,&legacy,8).unwrap();
        assert_eq!(selection.locked_fine(),[18,19,27]);
        selection.observe_at_floor(&[],&legacy,4).unwrap();
        assert_eq!(selection.locked_fine(),[18,19,27],"browsing back must not erase an existing dependency lock");
    }
    #[test]
    fn global_selection_forces_all_occupied_descendants_even_before_fine_admission(){
        let doc=fixture();let legacy=BTreeMap::new();let mut selection=RedrawSelection::default();
        selection.toggle_at_floor(Stage::Global,0,&doc.primitives,&legacy,4).unwrap();
        assert!(selection.global_selected());assert_eq!(selection.locked_coarse(),[4,5]);
        assert_eq!(selection.locked_fine(),[18,19,27,63]);
        assert_eq!(selection.regions(),[(1,0),(4,4),(4,5),(8,18),(8,19),(8,27),(8,63)]);
        assert!(selection.validate_empty(&doc.primitives,&legacy).unwrap_err().contains("2个粗网格、4个细网格"));
        assert!(selection.toggle(Stage::Coarse,4,&doc.primitives,&legacy).is_err());
        assert!(selection.toggle(Stage::Fine,63,&doc.primitives,&legacy).is_err());
        let kept=selection.delete_locked_canvas(&doc.primitives,&legacy).unwrap();
        assert_eq!(kept,doc.primitives[..2],"red-only removal preserves the background and Global geometry");
        selection.observe_at_floor(&kept,&legacy,1).unwrap();selection.validate_empty(&kept,&legacy).unwrap();
        assert_eq!(selection.locked_coarse(),[4,5]);assert_eq!(selection.locked_fine(),[18,19,27,63]);
        assert_eq!(selection.delete_canvas(&doc.primitives,&legacy).unwrap(),doc.primitives[..1],"all-selected removal keeps the background");
    }
    #[test]
    fn cancelling_global_only_removes_its_locks_and_restores_prior_manual_selections(){
        let doc=fixture();let legacy=BTreeMap::new();let mut selection=RedrawSelection::default();
        selection.toggle(Stage::Fine,18,&doc.primitives,&legacy).unwrap();
        selection.toggle(Stage::Fine,63,&doc.primitives,&legacy).unwrap();
        selection.toggle(Stage::Coarse,5,&doc.primitives,&legacy).unwrap();let before=selection.clone();
        selection.toggle_at_floor(Stage::Global,0,&doc.primitives,&legacy,4).unwrap();
        assert_eq!(selection.selected_coarse(),[5]);assert_eq!(selection.selected_fine(),[18,63]);
        assert_eq!(selection.locked_coarse(),[4,5]);assert_eq!(selection.locked_fine(),[18,19,27,63]);
        let kept=selection.delete_locked_canvas(&doc.primitives,&legacy).unwrap();selection.observe_at_floor(&kept,&legacy,4).unwrap();
        selection.toggle(Stage::Global,0,&kept,&legacy).unwrap();assert_eq!(selection,before);
        assert_eq!(selection.locked_fine(),[18,19,27]);assert!(selection.locked_coarse().is_empty());
    }
    #[test]
    fn global_dependency_latches_follow_later_edits_and_round_trip_without_requiring_contents(){
        let doc=fixture();let legacy=BTreeMap::new();let mut selection=RedrawSelection::default();
        selection.toggle_at_floor(Stage::Global,0,&[],&legacy,1).unwrap();assert_eq!(selection.regions(),[(1,0)]);
        selection.observe_at_floor(&doc.primitives,&legacy,1).unwrap();
        let kept=selection.delete_locked_canvas(&doc.primitives,&legacy).unwrap();selection.observe_at_floor(&kept,&legacy,1).unwrap();
        let stored=serde_json::to_vec(&selection).unwrap();let restored:RedrawSelection=serde_json::from_slice(&stored).unwrap();
        assert_eq!(selection,restored);restored.validate_empty(&kept,&legacy).unwrap();
        assert_eq!(restored.locked_coarse(),[4,5]);assert_eq!(restored.locked_fine(),[18,19,27,63]);
    }
    #[test]
    fn old_selection_json_defaults_to_no_global_selection_and_invalid_global_locks_are_rejected(){
        let old=r#"{"manual_coarse":[5],"manual_fine":[18],"locked_children":{"5":[18,19,27]}}"#;
        let selection:RedrawSelection=serde_json::from_str(old).unwrap();selection.validate().unwrap();
        assert!(!selection.global_selected());assert!(selection.locked_coarse().is_empty());assert_eq!(selection.locked_fine(),[18,19,27]);
        for invalid in [serde_json::json!({"global_locked_coarse":[0]}),serde_json::json!({"global_locked_fine":[0]}),
            serde_json::json!({"global_selected":true,"global_locked_coarse":[16]}),
            serde_json::json!({"global_selected":true,"global_locked_fine":[64]})]{
            let selection:RedrawSelection=serde_json::from_value(invalid).unwrap();assert!(selection.validate().is_err());
        }
        assert!(serde_json::from_str::<RedrawSelection>(r#"{"global_selected":true,"unknown_lock":[0]}"#).is_err());
    }
}
