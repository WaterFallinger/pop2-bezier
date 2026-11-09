use super::*;
use crate::parallel_history::ParallelHistory;
use crate::document::Settings;

fn shape(x:f32)->Primitive{Primitive::new(ShapeKind::Rectangle,x,31.5,10.,12.,[80,120,160])}
fn owned(stage:StageTier,index:usize,ticket:u64)->Primitive{let mut p=shape(31.5);p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell:index,queue_order:ticket});p}
fn commit_plan(doc:&mut Document,store:&mut RawHistoryStore,plan:StageEditPlan)->NodeId{
    let parent=doc.current;let node=doc.commit_canvas("stage edit",OperationSource::User,plan.canvas.clone()).unwrap();
    store.apply_prepared_edit(doc,parent,node,plan).unwrap();node
}

fn past_edit_fixture(floor:usize)->(Document,RawHistoryStore,NodeId){
    let mut doc=Document::new();doc.settings.pause_at_stage_end=true;let mut store=RawHistoryStore::new();
    let mut global=shape(200.);global.y=200.;global.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:1});
    let mut coarse=shape(31.5);coarse.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Coarse,cell:0,queue_order:2});
    let mut canvas=vec![Primitive::background([240;3]),global,coarse];
    if floor==8{let mut fine=shape(480.);fine.y=480.;fine.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Fine,cell:63,queue_order:3});canvas.push(fine);}
    let parent=doc.commit_canvas("synthetic entered stage",OperationSource::Model,canvas).unwrap();
    let rows=doc.primitives.iter().map(|p|p.to_tokens().unwrap().map(u32::from)).collect::<Vec<_>>();
    let mut raw=vec![BOS_RECORD,rows[0],rows[1],crate::region_state::marker_record(Stage::Coarse,0).unwrap(),rows[2]];
    if floor==8{raw.extend([crate::region_state::END_RECORD,crate::region_state::marker_record(Stage::Fine,63).unwrap(),rows[3]]);}
    store.finish(&doc,parent,&raw).unwrap();(doc,store,parent)
}

#[test]
fn recoloring_a_global_shape_in_coarse_changes_input_color_without_any_coarse_control_or_progress_change(){
    let(mut doc,mut store,parent)=past_edit_fixture(4);let before=store.effective_history(&doc,parent).unwrap();
    let before_region=DrawingRegion::from_history(&before.records).unwrap();let id=doc.primitives[1].id;
    let mut candidate=doc.primitives.clone();candidate[1].color=[20,60,100];let row=candidate[1].to_tokens().unwrap().map(u32::from);
    let plan=store.prepare_edit(&doc,parent,doc.preview_canvas(candidate).unwrap(),None,false).unwrap();
    assert!(plan.manual_review_regions.is_empty());assert!(plan.raw.contains(&row));
    let controls=|rows:&[RawRecord]|rows.iter().filter(|row|!is_visible(row)).copied().collect::<Vec<_>>();
    assert_eq!(controls(&plan.raw),controls(&before.records));
    let after_region=DrawingRegion::from_history(&plan.raw).unwrap();assert_eq!(after_region,before_region);
    let child=commit_plan(&mut doc,&mut store,plan);
    assert_eq!(store.stage_map(&doc,child).unwrap()[&id],Stage::Global);assert_eq!(store.stage_floor(&doc,child).unwrap(),Some(4));
}

#[test]
fn changing_historical_geometry_or_deleting_it_never_reopens_a_current_stage_cell(){
    for floor in [4,8]{for index in if floor==4{vec![1]}else{vec![1,2]}{
        let(mut doc,store,parent)=past_edit_fixture(floor);
        for remove in [false,true]{let mut candidate=doc.primitives.clone();if remove{candidate.remove(index);}else{candidate[index].x=220.;candidate[index].y=220.;candidate[index].color=[20,60,100];}
            let plan=store.prepare_edit(&doc,parent,doc.preview_canvas(candidate).unwrap(),None,false).unwrap();
            assert!(plan.manual_review_regions.is_empty(),"historical edit floor={floor} index={index}");
            let before=DrawingRegion::from_history(&store.effective_history(&doc,parent).unwrap().records).unwrap();let after=DrawingRegion::from_history(&plan.raw).unwrap();
            if floor==4{assert_eq!(after.progress.coarse,before.progress.coarse);assert_eq!(after.progress.coarse_shape_counts,before.progress.coarse_shape_counts);}
            else{assert_eq!(after.progress.fine,before.progress.fine);assert_eq!(after.progress.fine_shape_counts,before.progress.fine_shape_counts);}
            assert_eq!(after.index,before.index);
        }
    }}
}

#[test]
fn current_stage_edits_and_same_cell_number_migration_still_request_the_real_current_region(){
    let(mut doc,store,parent)=past_edit_fixture(8);let fine=doc.primitives[3].id;
    let mut candidate=doc.primitives.clone();candidate[3].color=[20,60,100];
    let plan=store.prepare_edit(&doc,parent,doc.preview_canvas(candidate).unwrap(),None,false).unwrap();
    assert_eq!(plan.manual_review_regions,vec![(8,63)]);
    let plan=store.prepare_edit(&doc,parent,doc.primitives.clone(),Some(StageAssignmentRequest{ids:vec![fine],stage:Stage::Global}),false).unwrap();
    assert_eq!(plan.manual_review_regions,vec![(8,63)]);assert!(!plan.manual_review_regions.contains(&(8,0)));
}

#[test]
fn admission_changes_refresh_only_the_current_coarse_cell_not_the_global_shape_center(){
    let(mut doc,store,parent)=past_edit_fixture(4);let mut store=store;
    let mut canvas=doc.primitives.clone();canvas.extend((3..=22).map(|ticket|owned(StageTier::Coarse,0,ticket)));
    let plan=store.prepare_edit(&doc,parent,doc.preview_canvas(canvas).unwrap(),None,false).unwrap();let crowded=commit_plan(&mut doc,&mut store,plan);
    store.set_manual_reviews(crowded,Vec::new());
    // A mode change admits the previously deferred row even though that row's
    // geometry is unchanged. It is a real input change in cell0, not cell5.
    doc.settings.pause_at_stage_end=false;let mut candidate=doc.primitives.clone();candidate[1].color=[20,60,100];
    let plan=store.prepare_edit(&doc,crowded,doc.preview_canvas(candidate).unwrap(),None,false).unwrap();
    assert_eq!(plan.manual_review_regions,vec![(4,0)]);
}

#[test]
fn old_empty_coarse_interrupt_after_global_edit_does_not_advance_actual_phase(){
    let mut doc=Document::new();let mut store=RawHistoryStore::new();
    let parent=doc.commit_canvas("global",OperationSource::User,(0..85).map(|_|shape(31.5)).collect()).unwrap();
    let raw=canvas_records(&doc,parent).unwrap();store.finish(&doc,parent,&raw).unwrap();
    let mut edited=doc.primitives.clone();edited[0].color=[22,44,66];let legacy=doc.commit_canvas("old editor",OperationSource::User,edited).unwrap();
    let mut buggy=canvas_records(&doc,legacy).unwrap();buggy.extend([crate::region_state::marker_record(Stage::Coarse,0).unwrap(),crate::region_state::INTERRUPT_RECORD]);
    store.entries.insert(legacy,HistoryDelta{base:HistoryBase::Node(parent),keep:0,tail:buggy});
    assert_eq!(store.stage_floor(&doc,legacy).unwrap(),None);
    let effective=store.effective_history(&doc,legacy).unwrap();assert_eq!(effective.primitive_ids.len(),85);
    assert!(!effective.records.iter().any(crate::region_state::is_marker));
    let mut changed=doc.primitives.clone();changed[1].x=400.;let changed=doc.preview_canvas(changed).unwrap();
    let plan=store.prepare_edit(&doc,legacy,changed,None,false).unwrap();assert_eq!(plan.phase_floor,1);
    assert!(!plan.raw.iter().any(crate::region_state::is_marker));
    let node=commit_plan(&mut doc,&mut store,plan);assert_eq!(store.editing_stage(&doc,node).unwrap(),Stage::Global);
}

#[test]
fn assigning_future_stage_keeps_actual_phase_and_later_model_callbacks_use_explicit_queued_ids(){
    let mut doc=Document::new();let mut store=RawHistoryStore::new();
    let candidate=doc.preview_canvas(vec![shape(31.5)]).unwrap();let plan=store.prepare_edit(&doc,0,candidate,None,false).unwrap();
    let first=commit_plan(&mut doc,&mut store,plan);let user_id=doc.primitives[0].id;
    let plan=store.prepare_edit(&doc,first,doc.primitives.clone(),Some(StageAssignmentRequest{ids:vec![user_id],stage:Stage::Fine}),false).unwrap();
    let future=commit_plan(&mut doc,&mut store,plan);assert_eq!(store.stage_floor(&doc,future).unwrap(),None);
    assert!(store.effective_history(&doc,future).unwrap().primitive_ids.is_empty());
    let queued=store.pending_stage_context(&doc,future).unwrap();assert_eq!((queued.len(),queued[0].side,queued[0].index),(1,8,0));
    let settings=Settings{parallel_sequences:2,..Default::default()};
    let mut branches=ParallelHistory::new(&doc,future,"future".into(),settings).unwrap();branches.initialize_model_input(&doc,&store).unwrap();
    let marker=crate::region_state::marker_record(Stage::Fine,0).unwrap();let mut raw=vec![BOS_RECORD,marker,queued[0].tokens,crate::region_state::INTERRUPT_RECORD,marker];
    let next=shape(40.).to_tokens().unwrap().map(u32::from);raw.push(next);
    let child=branches.submit(&mut doc,&mut store,0,next,&raw).unwrap();
    let ids=store.model_ids_for_node(&doc,child).unwrap();assert_eq!(ids[0],user_id);assert_eq!(ids.len(),2);
    assert_eq!(store.stage_floor(&doc,child).unwrap(),Some(8));assert_eq!(doc.materialize(child).unwrap().len(),2);
    assert_eq!(store.manual_review_regions(&doc,child).unwrap(),vec![(8,0)]);
    let mut ended=raw.clone();ended.push(crate::region_state::END_RECORD);branches.finish_lane(&doc,&mut store,0,&ended).unwrap();
    let mut other=vec![BOS_RECORD,marker,queued[0].tokens,crate::region_state::INTERRUPT_RECORD,marker];
    let token=shape(50.).to_tokens().unwrap().map(u32::from);other.push(token);
    let sibling=branches.submit(&mut doc,&mut store,1,token,&other).unwrap();
    assert_eq!(store.manual_review_regions(&doc,sibling).unwrap(),vec![(8,0)],"late sibling retains its own activated-input review after lane0 END");
    assert_eq!(store.stage_floor(&doc,future).unwrap(),None,"future origin snapshot is immutable");
}

#[test]
fn draft_guard_is_frozen_for_all_lanes_and_deferred_geometry_remains_on_canvas(){
    let mut doc=Document::new();let mut store=RawHistoryStore::new();
    let origin=doc.commit_canvas("twenty one",OperationSource::User,(1..=21).map(|ticket|owned(StageTier::Coarse,0,ticket)).collect()).unwrap();
    let mut raw=vec![BOS_RECORD,crate::region_state::marker_record(Stage::Coarse,0).unwrap()];
    raw.extend(doc.primitives.iter().map(|p|p.to_tokens().unwrap().map(u32::from)));store.finish(&doc,origin,&raw).unwrap();
    let deferred=doc.primitives[20].id;assert!(!doc.nodes[origin].settings.pause_at_stage_end);doc.settings.pause_at_stage_end=true;
    let effective=store.effective_history(&doc,origin).unwrap();assert_eq!(effective.primitive_ids.len(),20);assert!(!effective.primitive_ids.contains(&deferred));
    assert_eq!(store.deferred_fine_context(&doc,origin).unwrap().len(),1);
    let mut settings=doc.settings.clone();settings.parallel_sequences=2;let mut branches=ParallelHistory::new(&doc,origin,"guard".into(),settings).unwrap();
    branches.initialize_model_input(&doc,&store).unwrap();
    for lane in 0..2{let mut callback=effective.records.clone();let token=shape(40.+lane as f32*10.).to_tokens().unwrap().map(u32::from);callback.push(token);
        let child=branches.submit(&mut doc,&mut store,lane,token,&callback).unwrap();
        assert_eq!(doc.materialize(child).unwrap().len(),22);assert!(!store.model_ids_for_node(&doc,child).unwrap().contains(&deferred));}
    // Historical origin retains its saved guard=false preference once lane0
    // follows a child; only the frozen request uses the draft guard=true.
    assert!(store.deferred_fine_context(&doc,origin).unwrap().is_empty());
}

#[test]
fn mapped_delete_compacts_all_id_phase_and_overlay_state_without_changing_survivors(){
    let mut doc=Document::new();let mut store=RawHistoryStore::new();
    let plan=store.prepare_edit(&doc,0,doc.preview_canvas(vec![shape(31.5)]).unwrap(),None,false).unwrap();let parent=commit_plan(&mut doc,&mut store,plan);
    let mut candidate=doc.primitives.clone();candidate.push(shape(50.));let plan=store.prepare_edit(&doc,parent,doc.preview_canvas(candidate).unwrap(),None,false).unwrap();
    let removed=commit_plan(&mut doc,&mut store,plan);doc.jump(parent).unwrap();
    let mut candidate=doc.primitives.clone();candidate.push(shape(60.));let plan=store.prepare_edit(&doc,parent,doc.preview_canvas(candidate).unwrap(),None,false).unwrap();
    let keep=commit_plan(&mut doc,&mut store,plan);let canvas=doc.primitives.clone();let ids=store.model_ids_for_node(&doc,keep).unwrap();
    let mut raw=store.context_for(&doc,keep).unwrap().unwrap();raw.push(crate::region_state::marker_record(Stage::Fine,0).unwrap());
    store.set_control_overlay_with_ids(&doc,keep,&raw,&ids).unwrap();
    let plan=doc.plan_delete_subtrees(&[removed]).unwrap();let result=store.apply_delete_subtrees(&mut doc,plan).unwrap();let new=result.old_to_new[keep].unwrap();
    assert_eq!(doc.materialize(new).unwrap(),canvas);assert_eq!(store.model_ids_for_node(&doc,new).unwrap(),ids);
    assert_eq!(store.context_for(&doc,new).unwrap(),Some(raw));assert_eq!(store.stage_floor(&doc,new).unwrap(),Some(8));
}
