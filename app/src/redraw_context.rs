//! Build a fresh, ID-mapped redraw prefix without rewriting the source branch.
//! Fine history is queued separately: a legal autoregressive stream cannot put
//! a coarse marker after fine markers, even when a user revisits a coarse cell.
use crate::{document::{Document,NodeId,Primitive},history_context::{RawHistoryStore,RawRecord,stage_edit},
    redraw::RedrawSelection,region_state::{self,PendingStageRecord,Stage}};
use std::collections::{BTreeMap,HashMap};

#[derive(Clone,Debug)]
pub struct RedrawPlan{
    /// Explicitly freeze legacy ownership before omitting future fine rows.
    /// Metadata does not change any of the nine model-visible attributes.
    pub canvas:Vec<Primitive>,
    pub records:Vec<RawRecord>,
    pub primitive_ids:Vec<u64>,
    pub pending_stage_context:Vec<PendingStageRecord>,
    pub pending_stage_ids:Vec<(usize,usize,u64)>,
    pub stage_floor:Option<usize>,
    pub regions:Vec<(usize,usize)>,
}

pub fn prepare_redraw(doc:&Document,store:&RawHistoryStore,node:NodeId,selection:&RedrawSelection)->Result<RedrawPlan,String>{
    let mut canvas=doc.materialize(node)?;let legacy=store.stage_map(doc,node)?;
    // This is deliberately a validation-only operation. In particular, it
    // never turns Generate into an implicit delete operation.
    selection.validate_empty(&canvas,&legacy)?;
    let view=stage_edit::view(&canvas,&legacy,false)?;
    for primitive in &mut canvas{primitive.stage_assignment=Some(*view.assignments.get(&primitive.id).ok_or("局部重绘缺少图元阶段归属")?);}
    if selection.global_selected(){
        // A Global visit has no region-start marker. Re-read its retained
        // ordinary prefix and start a fresh branch, after explicit deletion
        // of every later-stage dependency.
        if canvas.iter().any(|p|view.assignments[&p.id].stage.side()!=1){
            return Err("全图续画前须手动清空所有红色粗网格和细网格区域".into());
        }
        let backgrounds=canvas.iter().filter(|p|p.shape==crate::document::ShapeKind::Background).count();
        if !canvas.is_empty()&&(backgrounds!=1||canvas[0].shape!=crate::document::ShapeKind::Background){
            return Err("全图续画需要唯一且位于首位的背景图元".into());
        }
        if canvas.len().saturating_sub(backgrounds)>crate::region_state::GLOBAL_PROGRESS_CAP{
            return Err("全图阶段已超过85个普通图形，请先删减全图图形再继续绘制".into());
        }
        let mut records=vec![region_state::BOS_RECORD];
        for primitive in &canvas{records.push(primitive.to_tokens()?.map(u32::from));}
        let primitive_ids=canvas.iter().map(|p|p.id).collect::<Vec<_>>();
        store.validate_context_with_model_ids(doc,node,&records,&primitive_ids)?;
        return Ok(RedrawPlan{canvas,records,primitive_ids,pending_stage_context:Vec::new(),pending_stage_ids:Vec::new(),
            stage_floor:Some(1),regions:selection.regions()});
    }
    let regions=selection.regions();let wants_fine=regions.iter().any(|(side,_)|*side==8);
    let stage_floor=Some(if regions.iter().any(|(side,_)|*side==4){4}else{8});
    let by_id=canvas.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();
    let mut records=vec![region_state::BOS_RECORD];let mut primitive_ids=Vec::new();
    let mut coarse:BTreeMap<usize,Vec<u64>>=BTreeMap::new();
    let mut pending_stage_context=Vec::new();let mut pending_stage_ids=Vec::new();
    // StageView groups by stage/cell while preserving actual canvas stacking
    // order within each cell. IDs and token rows always use that same order.
    for primitive in &canvas{
        let id=&primitive.id;
        let assignment=view.assignments[id];let primitive=by_id[id];
        let target=regions.contains(&(assignment.stage.side(),assignment.cell));
        // Selected yellow contents are reread at the start of that exact visit.
        // Their previous permanent END is deliberately absent from this prefix.
        if target{
            pending_stage_context.push(PendingStageRecord{side:assignment.stage.side(),index:assignment.cell,tokens:primitive.to_tokens()?.map(u32::from)});
            pending_stage_ids.push((assignment.stage.side(),assignment.cell,*id));continue;
        }
        if !view.admitted_ids.contains(id){continue;}
        match stage_edit::stage(assignment.stage){
            Stage::Global=>{records.push(primitive.to_tokens()?.map(u32::from));primitive_ids.push(*id);},
            Stage::Coarse=>{coarse.entry(assignment.cell).or_default().push(*id);},
            Stage::Fine if wants_fine=>{
                pending_stage_context.push(PendingStageRecord{side:8,index:assignment.cell,tokens:primitive.to_tokens()?.map(u32::from)});
                pending_stage_ids.push((8,assignment.cell,*id));
            },
            Stage::Fine=>{},
            Stage::Complete=>return Err("局部重绘存在无效图元阶段".into()),
        }
    }
    for (cell,ids) in coarse{
        records.push(region_state::marker_record(Stage::Coarse,cell)?);
        for id in ids{records.push(by_id[&id].to_tokens()?.map(u32::from));primitive_ids.push(id);}
        records.push(region_state::END_RECORD);
    }
    // Only unselected coarse blocks are closed conditioning. Selected contents
    // remain queued until their fresh visit, without copying a previous END.
    store.validate_context_with_model_ids(doc,node,&records,&primitive_ids)?;
    if records.iter().any(|row|row==&region_state::INTERRUPT_RECORD
        ||region_state::is_marker(row)&&region_state::marker_identity(row).is_ok_and(|(stage,_)|stage==Stage::Fine)){
        return Err("局部重绘的初始上下文不能含细网格或悬挂的人为中断记录".into());
    }
    Ok(RedrawPlan{canvas,records,primitive_ids,pending_stage_context,pending_stage_ids,stage_floor,regions})
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::document::{OperationSource,ShapeKind,PrimitiveStageAssignment,StageTier};
    fn primitive(stage:StageTier,cell:usize,x:f32,color:[u8;3])->Primitive{
        let y=match stage{StageTier::Global=>20.,StageTier::Coarse=>(cell/4)as f32*128.+32.,StageTier::Fine=>(cell/8)as f32*64.+32.};
        let mut p=Primitive::new(ShapeKind::Rectangle,x,y,12.,10.,color);
        p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:cell as u64});p
    }
    fn fixture()->(Document,RawHistoryStore){
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let node=doc.commit_canvas("old fine branch",OperationSource::User,vec![
            Primitive::background([240;3]),primitive(StageTier::Global,0,20.,[40,60,80]),
            primitive(StageTier::Coarse,0,20.,[80,100,120]),primitive(StageTier::Coarse,1,140.,[120,100,80]),
            primitive(StageTier::Fine,0,20.,[40,80,120]),primitive(StageTier::Fine,3,210.,[120,80,40]),
            primitive(StageTier::Fine,63,480.,[80,40,120]),
        ]).unwrap();
        let mut raw=vec![region_state::BOS_RECORD];let mut ids=Vec::new();
        raw.extend(doc.primitives[..2].iter().map(|p|p.to_tokens().unwrap().map(u32::from)));ids.extend(doc.primitives[..2].iter().map(|p|p.id));
        for (stage,index,position) in [(Stage::Coarse,0,2),(Stage::Coarse,1,3),(Stage::Fine,0,4),(Stage::Fine,3,5),(Stage::Fine,63,6)]{
            raw.extend([region_state::marker_record(stage,index).unwrap(),doc.primitives[position].to_tokens().unwrap().map(u32::from),region_state::END_RECORD]);
            ids.push(doc.primitives[position].id);
        }
        store.set_control_overlay_with_ids(&doc,node,&raw,&ids).unwrap();(doc,store)
    }
    fn delete_selected(doc:&mut Document,store:&mut RawHistoryStore,selection:&RedrawSelection){
        let parent=doc.current;let legacy=store.stage_map(doc,parent).unwrap();
        let canvas=selection.delete_canvas(&doc.primitives,&legacy).unwrap();
        let plan=store.prepare_edit(doc,parent,canvas,None,false).unwrap();
        let node=doc.commit_canvas("manually clear selected",OperationSource::User,plan.canvas.clone()).unwrap();
        store.apply_prepared_edit(doc,parent,node,plan).unwrap();
    }
    #[test]
    fn global_selection_requires_explicit_descendant_clear_and_reuses_only_global_prefix(){
        let(mut doc,mut store)=fixture();let original=doc.primitives.clone();let original_node=doc.current;
        let mut selection=RedrawSelection::default();let legacy=store.stage_map(&doc,doc.current).unwrap();
        selection.toggle(Stage::Global,0,&doc.primitives,&legacy).unwrap();
        assert!(prepare_redraw(&doc,&store,doc.current,&selection).is_err());
        let canvas=selection.delete_locked_canvas(&doc.primitives,&legacy).unwrap();
        let edit=store.prepare_edit(&doc,doc.current,canvas,None,false).unwrap();let parent=doc.current;
        let node=doc.commit_canvas("manually clear all Global dependencies",OperationSource::User,edit.canvas.clone()).unwrap();
        store.apply_prepared_edit(&doc,parent,node,edit).unwrap();
        let plan=prepare_redraw(&doc,&store,doc.current,&selection).unwrap();
        assert_eq!(plan.canvas.iter().map(|p|(p.id,p.to_tokens().unwrap())).collect::<Vec<_>>(),
            original[..2].iter().map(|p|(p.id,p.to_tokens().unwrap())).collect::<Vec<_>>());
        assert_eq!(plan.stage_floor,Some(1));
        assert_eq!(plan.primitive_ids,original[..2].iter().map(|p|p.id).collect::<Vec<_>>());
        assert_eq!(plan.regions,[(1,0),(4,0),(4,1),(8,0),(8,3),(8,63)]);
        assert!(plan.pending_stage_context.is_empty()&&plan.pending_stage_ids.is_empty());
        assert!(plan.records.iter().all(|r|!region_state::is_marker(r)&&r!=&region_state::END_RECORD&&r!=&region_state::INTERRUPT_RECORD));
        let origin=doc.commit_control_canvas("准备局部重绘",plan.canvas).unwrap();
        store.set_control_overlay_with_ids(&doc,origin,&plan.records,&plan.primitive_ids).unwrap();
        store.begin_redraw_origin(&doc,origin,&plan.regions,&plan.records,&plan.primitive_ids).unwrap();
        assert_eq!(store.stage_floor(&doc,origin).unwrap().unwrap_or(1),1);
        assert!(store.completion_evidence(&doc,origin).unwrap().regions.is_empty());
        assert_eq!(doc.materialize(original_node).unwrap(),original);
    }
    #[test]
    fn coarse_redraw_from_fine_parent_has_closed_coarse_prefix_and_id_matched_fine_queue(){
        let(mut doc,mut store)=fixture();let original_node=doc.current;
        let original_raw=store.context_for(&doc,original_node).unwrap();let original_canvas=doc.primitives.clone();
        let legacy=store.stage_map(&doc,doc.current).unwrap();let mut selection=RedrawSelection::default();
        selection.toggle(Stage::Coarse,0,&doc.primitives,&legacy).unwrap();
        assert!(prepare_redraw(&doc,&store,doc.current,&selection).is_err());
        delete_selected(&mut doc,&mut store,&selection);
        let plan=prepare_redraw(&doc,&store,doc.current,&selection).unwrap();
        assert_eq!(plan.regions,[(4,0),(8,0)]);assert_eq!(plan.stage_floor,Some(4));
        assert_eq!(store.stage_floor(&doc,doc.current).unwrap(),Some(8),"persistent generation floor never rewinds");
        assert_eq!(plan.records.iter().filter_map(|row|region_state::marker_identity(row).ok()).collect::<Vec<_>>(),[(Stage::Coarse,1)]);
        assert_eq!(plan.pending_stage_ids.iter().map(|&(s,i,_)|(s,i)).collect::<Vec<_>>(),[(8,3),(8,63)]);
        assert_eq!(plan.pending_stage_context.iter().map(|p|(p.side,p.index)).collect::<Vec<_>>(),[(8,3),(8,63)]);
        for (row,&(_,_,id)) in plan.pending_stage_context.iter().zip(&plan.pending_stage_ids){
            assert_eq!(row.tokens,plan.canvas.iter().find(|p|p.id==id).unwrap().to_tokens().unwrap().map(u32::from));
        }
        assert_eq!(store.context_for(&doc,original_node).unwrap(),original_raw);assert_eq!(doc.materialize(original_node).unwrap(),original_canvas);
        assert!(plan.canvas.iter().all(|p|p.stage_assignment.is_some()));
    }
    #[test]
    fn fine_only_scope_keeps_fine_rows_pending_and_empty_blank_scope_does_not_invent_background(){
        let(mut doc,mut store)=fixture();let legacy=store.stage_map(&doc,doc.current).unwrap();let mut selection=RedrawSelection::default();
        selection.toggle(Stage::Fine,3,&doc.primitives,&legacy).unwrap();delete_selected(&mut doc,&mut store,&selection);
        let plan=prepare_redraw(&doc,&store,doc.current,&selection).unwrap();assert_eq!(plan.stage_floor,Some(8));
        assert_eq!(plan.regions,[(8,3)]);assert!(plan.pending_stage_ids.iter().all(|&(side,_,_)|side==8));
        assert!(plan.records.iter().filter_map(|r|region_state::marker_identity(r).ok()).all(|(stage,_)|stage==Stage::Coarse));
        let doc=Document::new();let store=RawHistoryStore::new();let mut selection=RedrawSelection::default();
        selection.toggle(Stage::Coarse,15,&[],&BTreeMap::new()).unwrap();
        let plan=prepare_redraw(&doc,&store,0,&selection).unwrap();assert_eq!(plan.records,[region_state::BOS_RECORD]);
        assert!(plan.primitive_ids.is_empty()&&plan.pending_stage_ids.is_empty()&&plan.canvas.is_empty());
    }
    #[test]
    fn populated_yellow_region_is_retained_and_queued_without_its_previous_end(){
        let(doc,store)=fixture();let old=store.context_for(&doc,doc.current).unwrap();
        let mut selection=RedrawSelection::default();let map=store.stage_map(&doc,doc.current).unwrap();
        selection.toggle(Stage::Fine,3,&doc.primitives,&map).unwrap();
        let plan=prepare_redraw(&doc,&store,doc.current,&selection).unwrap();
        assert_eq!(plan.canvas.len(),doc.primitives.len());
        assert!(plan.canvas.iter().zip(&doc.primitives).all(|(a,b)|a.id==b.id&&a.to_tokens().unwrap()==b.to_tokens().unwrap()));
        assert!(plan.records.iter().all(|r|!region_state::marker_identity(r).is_ok_and(|(s,i)|s==Stage::Fine&&i==3)));
        assert!(plan.pending_stage_ids.contains(&(8,3,doc.primitives[5].id)));
        assert_eq!(plan.pending_stage_context.iter().find(|p|(p.side,p.index)==(8,3)).unwrap().tokens,
            doc.primitives[5].to_tokens().unwrap().map(u32::from));
        assert_eq!(store.context_for(&doc,doc.current).unwrap(),old);
    }
}
