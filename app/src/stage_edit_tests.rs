use super::*;
use crate::document::{Document,OperationSource};

fn shape(x:f32,y:f32)->Primitive {Primitive::new(ShapeKind::Rectangle,x,y,10.,12.,[80,120,160])}
fn owned(stage:StageTier,cell:usize,order:u64)->Primitive {
    let mut p=shape(31.5,31.5);p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:order});p
}
fn document(stage:StageTier,count:usize,background:bool)->Document {
    let mut doc=Document::new();let mut canvas=Vec::new();
    if background{canvas.push(Primitive::background([254;3]));}
    canvas.extend((0..count).map(|index|owned(stage,0,index as u64+1)));
    doc.commit_canvas("fixture",OperationSource::User,canvas).unwrap();doc
}
fn ids(canvas:&[Primitive])->Vec<u64>{canvas.iter().map(|p|p.id).collect()}

#[test]
fn global_85_ordinary_with_or_without_background_requires_confirmation_before_new_draw_commits(){
    for background in [false,true] {
        let mut doc=document(StageTier::Global,85,background);let original=doc.primitives.clone();let original_node=doc.current;
        let mut next=original.clone();next.push(shape(31.5,31.5));let proposed=doc.preview_canvas(next).unwrap();let new_id=proposed.last().unwrap().id;
        let blocked=prepare_canvas(&original,proposed,&BTreeMap::new(),Stage::Global,None,true,false).unwrap_err();
        let candidate=match blocked{StageEditBlock::GlobalOverflow{candidate_canvas,..}=>candidate_canvas,other=>panic!("expected explicit confirmation, got {other:?}")};
        // Cancelling/dismissing that proposal has no document mutation.
        assert_eq!(doc.current,original_node);assert_eq!(doc.primitives,original);
        let confirmed=prepare_canvas(&original,candidate,&BTreeMap::new(),Stage::Global,None,true,true).unwrap();
        let admitted=view(&confirmed,&BTreeMap::new(),true).unwrap();
        assert_eq!(confirmed.len(),original.len()+1);assert_eq!(admitted.stage_by_id[&new_id],Stage::Coarse);
        assert_eq!(confirmed.iter().filter(|p|p.shape!=ShapeKind::Background&&admitted.stage_by_id[&p.id]==Stage::Global).count(),85);
        doc.commit_canvas("confirmed",OperationSource::User,confirmed.clone()).unwrap();
        assert_eq!(doc.primitives,confirmed);assert_ne!(doc.current,original_node);
    }
}

#[test]
fn editing_old_global_color_or_geometry_never_adopts_the_later_creation_phase(){
    let doc=document(StageTier::Global,85,true);let original=doc.primitives.clone();
    let mut proposed=original.clone();proposed[1].color=[10,30,50];proposed[2].x=300.;
    let proposed=doc.preview_canvas(proposed).unwrap();
    let result=prepare_canvas(&original,proposed,&BTreeMap::new(),Stage::Coarse,None,true,false).unwrap();
    let after=view(&result,&BTreeMap::new(),true).unwrap();
    assert_eq!(ids(&result),ids(&original));assert!(after.stage_by_id.values().all(|stage|*stage==Stage::Global));
    assert_eq!(doc.primitives,original);
}

#[test]
fn coarse_overflow_remains_visible_and_deleting_oldest_admitted_promotes_fifo_without_layer_changes(){
    let mut doc=document(StageTier::Coarse,21,false);let first=doc.primitives[0].id;let last=doc.primitives[20].id;
    // Arrival tickets, rather than stacking position, determine admission.
    let mut stacking=doc.primitives.clone();stacking.swap(0,20);doc.commit_canvas("stacking",OperationSource::User,stacking).unwrap();
    let before=doc.primitives.clone();let staged=prepare_canvas(&before,doc.preview_canvas(before.clone()).unwrap(),&BTreeMap::new(),Stage::Fine,None,true,false).unwrap();
    let active=view(&staged,&BTreeMap::new(),true).unwrap();
    assert_eq!(staged.len(),21);assert_eq!(active.deferred_ids,vec![last]);assert_eq!(active.admitted_ids.len(),20);
    let removed=before.iter().filter(|p|p.id!=first).cloned().collect();
    let promoted=prepare_canvas(&before,doc.preview_canvas(removed).unwrap(),&BTreeMap::new(),Stage::Fine,None,true,false).unwrap();
    let next=view(&promoted,&BTreeMap::new(),true).unwrap();
    assert!(next.deferred_ids.is_empty());assert!(next.admitted_ids.contains(&last));
    assert_eq!(ids(&promoted),ids(&before).into_iter().filter(|id|*id!=first).collect::<Vec<_>>());
    assert_eq!(promoted.iter().find(|p|p.id==last).unwrap().stage_assignment.unwrap().queue_order,21);
    assert_eq!(doc.primitives,before);
}

#[test]
fn fine_eleventh_shape_is_rejected_without_touching_the_canvas(){
    let doc=document(StageTier::Fine,10,false);let original=doc.primitives.clone();let node=doc.current;
    let mut proposed=original.clone();proposed.push(shape(31.5,31.5));
    let result=prepare_canvas(&original,doc.preview_canvas(proposed).unwrap(),&BTreeMap::new(),Stage::Fine,None,true,false);
    assert!(matches!(result,Err(StageEditBlock::Rejected(_))));assert_eq!(doc.primitives,original);assert_eq!(doc.current,node);
}

#[test]
fn changing_tier_or_cell_issues_later_tickets_but_keeps_stacking_order(){
    let mut doc=Document::new();let mut coarse=owned(StageTier::Coarse,0,2);coarse.x=35.;
    let mut fine=owned(StageTier::Fine,4,3);fine.x=300.;
    doc.commit_canvas("mixed",OperationSource::User,vec![owned(StageTier::Global,0,1),coarse,fine]).unwrap();
    let original=doc.primitives.clone();let selected=StageAssignmentRequest{ids:vec![original[0].id],stage:Stage::Coarse};
    let moved=prepare_canvas(&original,doc.preview_canvas(original.clone()).unwrap(),&BTreeMap::new(),Stage::Fine,Some(&selected),true,false).unwrap();
    let a=moved[0].stage_assignment.unwrap();assert_eq!((a.stage,a.cell),(StageTier::Coarse,0));assert!(a.queue_order>3);assert_eq!(ids(&moved),ids(&original));
    let mut next=moved.clone();next[0].x=270.;let changed=prepare_canvas(&moved,doc.preview_canvas(next).unwrap(),&BTreeMap::new(),Stage::Fine,None,true,false).unwrap();
    let b=changed[0].stage_assignment.unwrap();assert_eq!((b.stage,b.cell),(StageTier::Coarse,2));assert!(b.queue_order>a.queue_order);assert_eq!(ids(&changed),ids(&original));
    let selected=StageAssignmentRequest{ids:vec![changed[2].id],stage:Stage::Global};
    let earlier=prepare_canvas(&changed,doc.preview_canvas(changed.clone()).unwrap(),&BTreeMap::new(),Stage::Fine,Some(&selected),true,false).unwrap();
    assert_eq!(earlier[2].stage_assignment.unwrap().stage,StageTier::Global);assert!(earlier[2].stage_assignment.unwrap().queue_order>b.queue_order);
    assert_eq!(ids(&earlier),ids(&original));
}

#[test]
fn legacy_none_keeps_inferred_stage_cell_and_ticket_until_the_object_is_edited(){
    let mut doc=Document::new();doc.commit_canvas("legacy",OperationSource::User,vec![shape(20.,20.),shape(160.,160.)]).unwrap();
    let original=doc.primitives.clone();let legacy=BTreeMap::from([(original[0].id,Stage::Global),(original[1].id,Stage::Coarse)]);
    let unchanged=prepare_canvas(&original,doc.preview_canvas(original.clone()).unwrap(),&legacy,Stage::Fine,None,true,false).unwrap();
    assert!(unchanged.iter().all(|p|p.stage_assignment.is_none()));
    let view=view(&unchanged,&legacy,true).unwrap();assert_eq!(view.assignments[&original[1].id].cell,5);assert_eq!(view.assignments[&original[1].id].queue_order,original[1].id);
    let mut changed=original.clone();changed[1].color=[22,44,66];
    let changed=prepare_canvas(&original,doc.preview_canvas(changed).unwrap(),&legacy,Stage::Fine,None,true,false).unwrap();
    assert!(changed[0].stage_assignment.is_none());assert_eq!(changed[1].stage_assignment,Some(PrimitiveStageAssignment{stage:StageTier::Coarse,cell:5,queue_order:original[1].id}));
    assert_eq!(ids(&changed),ids(&original));
}
