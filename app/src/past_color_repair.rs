//! Read-only recovery of controls for provably historical RGB-only edits.
use super::*;

pub(super) fn historical_color_origin(store:&RawHistoryStore,doc:&Document,node:NodeId)->Result<Option<NodeId>,String>{
    let start=doc.nodes.get(node).ok_or("历史改色状态不存在")?;
    if node==doc.current&&doc.settings.pause_at_stage_end!=start.settings.pause_at_stage_end{return Ok(None);}
    let mut cursor=node;
    for _ in 0..doc.nodes.len(){
        // Public overlays have no creation timestamp. Never graft a possible
        // later sibling's real controls into an existing descendant.
        if store.control_overlays.contains_key(&cursor){return Ok(None);}
        let state=doc.nodes.get(cursor).ok_or("历史改色父链无效")?;
        if state.id!=cursor{return Err("历史改色状态编号无效".into());}
        if state.source!=OperationSource::User||state.patches.is_empty()
            ||state.patches.iter().any(|patch|!matches!(patch,Patch::Replace{..})){
            return Ok((cursor!=node).then_some(cursor));
        }
        let Some(parent)=state.parent else{return Ok((cursor!=node).then_some(cursor));};
        let previous=doc.nodes.get(parent).filter(|p|p.id==parent&&parent<cursor).ok_or("历史改色父节点无效")?;
        if state.settings.pause_at_stage_end!=previous.settings.pause_at_stage_end{
            return Ok((cursor!=node).then_some(cursor));
        }
        let floor=store.stage_floor(doc,cursor)?.unwrap_or(1);
        if floor<=1||store.stage_floor(doc,parent)?.unwrap_or(1)!=floor{
            return Ok((cursor!=node).then_some(cursor));
        }
        let old=doc.materialize(parent)?;let new=doc.materialize(cursor)?;
        if old.len()!=new.len()||old.iter().zip(&new).any(|(a,b)|a.id!=b.id){return Ok((cursor!=node).then_some(cursor));}
        let old_by_id=old.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();
        let old_stages=store.stage_map(doc,parent)?;let new_stages=store.stage_map(doc,cursor)?;
        let mut changed=false;let mut valid=true;let mut replaced=HashSet::new();
        for patch in &state.patches{
            let Patch::Replace{primitive}=patch else{unreachable!()};
            replaced.insert(primitive.id);
            let Some(before)=old_by_id.get(&primitive.id).copied()else{valid=false;break;};
            let Some(stage)=old_stages.get(&primitive.id).copied()else{valid=false;break;};
            let side=match stage{Stage::Global=>1,Stage::Coarse=>4,Stage::Fine=>8,Stage::Complete=>{valid=false;break;}};
            if side>=floor||new_stages.get(&primitive.id)!=Some(&stage)||!only_color_or_legacy_metadata(before,primitive,stage){valid=false;break;}
            changed|=before!=primitive;
        }
        if !valid||!changed{return Ok((cursor!=node).then_some(cursor));}
        // Checkpoints must agree with the same RGB-only structural contract.
        if old.iter().zip(&new).any(|(before,after)|{
            before!=after&&!replaced.contains(&before.id)
                ||old_stages.get(&before.id).is_none_or(|stage|!only_color_or_legacy_metadata(before,after,*stage))
        }){
            return Ok((cursor!=node).then_some(cursor));
        }
        cursor=parent;
    }
    Err("历史改色父链存在循环或超出状态数".into())
}

fn only_color_or_legacy_metadata(before:&Primitive,after:&Primitive,stage:Stage)->bool{
    let mut structural=after.clone();structural.color=before.color;structural.stage_assignment=before.stage_assignment;
    if &structural!=before{return false;}
    if before.stage_assignment==after.stage_assignment{return true;}
    match(before.stage_assignment,after.stage_assignment){
        (None,Some(value))=>stage_edit::stage(value.stage)==stage&&value.queue_order==before.id
            &&stage_edit::cell_for(value.stage,before).is_ok_and(|cell|cell==value.cell),
        _=>false,
    }
}

#[cfg(test)]
mod tests{
    use super::*;use crate::document::PrimitiveStageAssignment;
    fn fixture()->(Document,RawHistoryStore,NodeId,u64,u64,u64){
        let mut doc=Document::new();doc.settings.pause_at_stage_end=true;let mut store=RawHistoryStore::new();let mut raw=vec![BOS_RECORD];
        let shapes=[Primitive::background([240;3]),Primitive::new(ShapeKind::Rectangle,80.,80.,12.,10.,[60,80,100]),
            Primitive::new(ShapeKind::Rectangle,160.,160.,12.,10.,[80,100,120]),Primitive::new(ShapeKind::Rectangle,350.,350.,12.,10.,[100,120,140])];
        for(index,p)in shapes.into_iter().enumerate(){let node=doc.commit_generated(p).unwrap();
            if index==2{raw.push(crate::region_state::marker_record(Stage::Coarse,5).unwrap());}
            if index==3{raw.push(crate::region_state::marker_record(Stage::Fine,45).unwrap());}
            raw.push(doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from));store.attach(&doc,node,&raw).unwrap();
            if index>=2{raw.push(crate::region_state::END_RECORD);store.finish(&doc,node,&raw).unwrap();}
        }
        let origin=doc.current;let ids=doc.primitives.iter().map(|p|p.id).collect::<Vec<_>>();
        (doc,store,origin,ids[1],ids[2],ids[3])
    }
    /// Construct the old bug explicitly, bypassing the repaired prepare path:
    /// an older-stage Replace wrongly reopened the current fine visit.
    fn legacy_edit(doc:&mut Document,store:&mut RawHistoryStore,id:u64,color:[u8;3],annotate:bool,queue_delta:u64,move_x:f32)->NodeId{
        let parent=doc.current;let mut raw=store.context_for(doc,parent).unwrap().unwrap();let ids=store.model_ids_for_node(doc,parent).unwrap();
        let stage=store.stage_map(doc,parent).unwrap()[&id];let mut next=doc.primitives.clone();let p=next.iter_mut().find(|p|p.id==id).unwrap();p.color=color;p.x+=move_x;
        if annotate{let tier=stage_edit::tier(stage).unwrap();p.stage_assignment=Some(PrimitiveStageAssignment{stage:tier,
            cell:stage_edit::cell_for(tier,p).unwrap(),queue_order:id+queue_delta});}
        let node=doc.commit_canvas("legacy RGB edit",OperationSource::User,next).unwrap();let by_id=doc.primitives.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();
        let mut cursor=0;for row in &mut raw{if is_visible(row){*row=by_id[&ids[cursor]].to_tokens().unwrap().map(u32::from);cursor+=1;}}
        if raw.last()==Some(&crate::region_state::END_RECORD){*raw.last_mut().unwrap()=crate::region_state::INTERRUPT_RECORD;}
        store.entries.insert(node,HistoryDelta{base:HistoryBase::Empty,keep:0,tail:raw});
        store.put_model_ids(doc,node,HistoryBase::Empty,&ids,8).unwrap();store.manual_reviews.insert(node,vec![(8,45)]);node
    }
    #[test]
    fn global_and_coarse_past_color_chain_finds_real_origin_without_rewriting_old_payload(){
        let(mut doc,mut store,origin,global,coarse,_)=fixture();let original=store.context_for(&doc,origin).unwrap();
        legacy_edit(&mut doc,&mut store,global,[120,140,160],true,0,0.);
        let node=legacy_edit(&mut doc,&mut store,coarse,[140,160,180],true,0,0.);
        let bad=store.context_for(&doc,node).unwrap();let pending=store.manual_reviews[&node].clone();let canvas=doc.primitives.clone();
        assert_eq!(historical_color_origin(&store,&doc,node).unwrap(),Some(origin));
        assert_eq!(store.context_for(&doc,node).unwrap(),bad);assert_eq!(store.context_for(&doc,origin).unwrap(),original);
        assert_eq!(store.manual_reviews[&node],pending);assert_eq!(doc.primitives,canvas);
        assert_eq!(bad.as_ref().unwrap().last(),Some(&crate::region_state::INTERRUPT_RECORD));
        let original=original.unwrap();let controls=|raw:&[RawRecord]|raw.iter().filter(|row|!is_visible(row)).copied().collect::<Vec<_>>();
        let fixed=store.effective_history(&doc,node).unwrap();assert_eq!(controls(&fixed.records),controls(&original));
        let by_id=doc.primitives.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();
        for(row,id)in fixed.records.iter().filter(|row|is_visible(row)).zip(&fixed.primitive_ids){assert_eq!(*row,by_id[id].to_tokens().unwrap().map(u32::from));}
        assert_eq!(store.manual_review_regions(&doc,node).unwrap(),store.manual_review_regions(&doc,origin).unwrap());
        assert_eq!(store.context_for(&doc,node).unwrap(),bad);
        // A legacy edit could also have lit up an unrelated empty current cell.
        let marker=crate::region_state::marker_record(Stage::Fine,9).unwrap();store.entries.get_mut(&node).unwrap().tail.push(marker);
        store.manual_reviews.insert(node,vec![(8,9)]);let old_payload=store.context_for(&doc,node).unwrap();
        let fixed=store.effective_history(&doc,node).unwrap();assert!(!fixed.records.contains(&marker));assert_eq!(controls(&fixed.records),controls(&original));
        assert_eq!(crate::region_state::DrawingRegion::from_history(&fixed.records).unwrap().progress,
            crate::region_state::DrawingRegion::from_history(&original).unwrap().progress);
        assert!(store.manual_review_regions(&doc,node).unwrap().is_empty());assert_eq!(store.context_for(&doc,node).unwrap(),old_payload);
    }
    #[test]
    fn model_overlay_guard_current_stage_and_geometry_are_conservative_barriers(){
        for barrier in 0..7{
            let(mut doc,mut store,origin,global,_,fine)=fixture();let target=if barrier==5{fine}else{global};
            let node=legacy_edit(&mut doc,&mut store,target,[120,140,160],true,0,if barrier==6{2.}else{0.});
            match barrier{
                0=>{doc.commit_generated(Primitive::new(ShapeKind::Rectangle,360.,360.,10.,10.,[160;3])).unwrap();},
                1=>{let raw=store.context_for(&doc,node).unwrap().unwrap();store.control_overlays.insert(node,raw);},
                2=>{let raw=store.context_for(&doc,origin).unwrap().unwrap();store.control_overlays.insert(origin,raw);},
                3=>doc.settings.pause_at_stage_end=false,
                4=>{doc.settings.pause_at_stage_end=false;doc.nodes[node].settings.pause_at_stage_end=false;},_=>{},
            }
            assert_eq!(historical_color_origin(&store,&doc,doc.current).unwrap(),None,"barrier {barrier}");
        }
    }
    #[test]
    fn legacy_metadata_preserves_queue_and_long_color_chains_are_walked_iteratively(){
        let(mut doc,mut store,origin,global,_,_)=fixture();
        let node=legacy_edit(&mut doc,&mut store,global,[120,140,160],true,100,0.);
        assert_eq!(historical_color_origin(&store,&doc,node).unwrap(),None,"a changed admission queue is not just RGB");
        let(mut doc,mut store,origin,global,_,_)=fixture();
        for n in 0..70{legacy_edit(&mut doc,&mut store,global,[(n*2)as u8,120,140],true,0,0.);}
        assert_eq!(historical_color_origin(&store,&doc,doc.current).unwrap(),Some(origin));
    }
    #[test]
    fn repaired_input_keeps_fake_reviews_excluded_when_zero_output_overlay_precedes_late_sibling(){
        let(mut doc,mut store,_,global,_,_)=fixture();
        let origin=legacy_edit(&mut doc,&mut store,global,[120,140,160],true,0,0.);
        let fake_marker=crate::region_state::marker_record(Stage::Fine,9).unwrap();
        store.entries.get_mut(&origin).unwrap().tail.push(fake_marker);store.manual_reviews.insert(origin,vec![(8,9)]);
        let fixed=store.effective_history(&doc,origin).unwrap();assert!(!fixed.records.contains(&fake_marker));
        let mut settings=doc.settings.clone();settings.parallel_sequences=2;
        let mut parallel=crate::parallel_history::ParallelHistory::new(&doc,origin,doc.prompt.clone(),settings).unwrap();
        parallel.initialize_model_input(&doc,&store).unwrap();
        parallel.finish_lane(&doc,&mut store,0,&fixed.records).unwrap();
        assert_eq!(historical_color_origin(&store,&doc,origin).unwrap(),None,"real overlay is now a repair barrier");
        assert!(store.manual_review_regions(&doc,origin).unwrap().is_empty());
        assert_eq!(store.creation_review_snapshot(origin),vec![(8,9)],"the old creation snapshot stays immutable");
        let fresh=Primitive::new(ShapeKind::Rectangle,480.,480.,8.,8.,[180,80,120]).to_tokens().unwrap().map(u32::from);
        let real_marker=crate::region_state::marker_record(Stage::Fine,63).unwrap();
        let mut incoming=fixed.records.clone();incoming.extend([real_marker,fresh]);
        let child=parallel.submit(&mut doc,&mut store,1,fresh,&incoming).unwrap();
        assert!(!store.manual_review_regions(&doc,child).unwrap().contains(&(8,9)));
        assert!(!store.creation_review_snapshot(child).contains(&(8,9)));
        let child_raw=store.context_for(&doc,child).unwrap().unwrap();assert!(child_raw.contains(&real_marker));
        assert_eq!(crate::region_state::DrawingRegion::from_history(&child_raw).unwrap().index,Some(63));
        assert_eq!(store.creation_review_snapshot(origin),vec![(8,9)]);
    }
}
