//! Pure validation and a complete ID-mapped ordinary-generation prefix.
//! Completion evidence is supplied by the history store; redraw conditioning
//! END records must never be turned into evidence by inspecting raw tokens here.
use crate::{document::{Document,NodeId,Primitive,ShapeKind,StageTier},history_context::{RawHistoryStore,RawRecord,stage_edit},
    region_state::{self,Stage}};
use serde::{Deserialize,Serialize};
use std::collections::{BTreeMap,BTreeSet,HashMap,HashSet};

#[derive(Clone,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionCompletionProof{pub side:usize,pub index:usize,pub primitive_ids:Vec<u64>}

#[derive(Clone,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionEvidence{pub regions:Vec<RegionCompletionProof>}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum TransitionIssueKind{InvalidHistory,InvalidEvidence,InvalidOwnership,GlobalIncomplete,GlobalOverflow,MissingBackground,
    CoarseOverflow,FineOverflow,CoarseIncompleteBeforeFine,GeometryOutsideOwnedCell}

#[derive(Clone,Debug,PartialEq,Eq)]
pub struct TransitionIssue{
    pub kind:TransitionIssueKind,pub stage:Stage,pub index:Option<usize>,pub count:Option<usize>,pub message:String,
}
impl TransitionIssue{
    pub fn row_col(&self)->Option<(usize,usize)>{let side=self.stage.side();
        self.index.filter(|i|matches!(side,4|8)&&*i<side*side).map(|i|(i/side+1,i%side+1))}
    pub fn message(&self)->&str{&self.message}
}

#[derive(Clone,Debug,Default,PartialEq,Eq)]
pub struct TransitionReport{pub issues:Vec<TransitionIssue>}
impl TransitionReport{pub fn message(&self)->String{self.issues.iter().map(|i|i.message.as_str()).collect::<Vec<_>>().join("\n")}}
impl std::fmt::Display for TransitionReport{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.write_str(&self.message())}}
impl std::error::Error for TransitionReport{}

#[derive(Clone,Debug)]
pub struct PaintResumePlan{
    pub canvas:Vec<Primitive>,pub records:Vec<RawRecord>,pub primitive_ids:Vec<u64>,
    pub stage_floor:Option<usize>,pub manual_review_regions:Vec<(usize,usize)>,
}

impl CompletionEvidence{
    pub fn validate(&self)->Result<(),String>{
        if self.regions.len()>80{return Err("区域完成证据超过16个粗网格与64个细网格的范围".into());}
        let mut keys=HashSet::new();
        for proof in &self.regions{
            if !matches!(proof.side,4|8)||proof.index>=proof.side*proof.side||!keys.insert((proof.side,proof.index)){
                return Err("区域完成证据含无效或重复的网格编号".into());
            }
            if proof.primitive_ids.len()>100_000{return Err("区域完成证据超过单画布对象上限".into());}
            let mut ids=HashSet::new();
            if proof.primitive_ids.iter().any(|id|*id==0||!ids.insert(*id)){return Err("区域完成证据含零或重复图元编号".into());}
        }
        Ok(())
    }
}

fn issue(report:&mut TransitionReport,kind:TransitionIssueKind,stage:Stage,index:Option<usize>,count:Option<usize>,message:impl Into<String>){
    report.issues.push(TransitionIssue{kind,stage,index,count,message:message.into()});
}
fn failure(kind:TransitionIssueKind,message:impl Into<String>)->TransitionReport{
    let mut report=TransitionReport::default();issue(&mut report,kind,Stage::Global,None,None,message);report
}
fn cell_name(stage:Stage,index:usize)->String{
    let side=stage.side();let name=if stage==Stage::Coarse{"粗网格"}else{"细网格"};format!("{name}第{}行第{}列",index/side+1,index%side+1)
}
fn same_membership(a:&[u64],b:&[u64])->bool{
    if a.len()!=b.len(){return false;}
    let mut a=a.to_vec();let mut b=b.to_vec();a.sort_unstable();b.sort_unstable();a==b
}

/// Every current figure is retained. This does not repair geometry, invent a
/// model END, delete rows, or change a document's phase floor. The caller first
/// commits the successful plan as an explicit, branch-local mode transition.
pub fn prepare_paint_resume(doc:&Document,store:&RawHistoryStore,node:NodeId,evidence:&CompletionEvidence)
    ->Result<PaintResumePlan,TransitionReport>{
    let mut canvas=doc.materialize(node).map_err(|error|failure(TransitionIssueKind::InvalidHistory,error))?;
    let legacy=store.stage_map(doc,node).map_err(|error|failure(TransitionIssueKind::InvalidHistory,error))?;
    let view=stage_edit::view(&canvas,&legacy,false).map_err(|error|failure(TransitionIssueKind::InvalidOwnership,error))?;
    let mut report=TransitionReport::default();
    let evidence_valid=match evidence.validate(){Ok(())=>true,Err(error)=>{issue(&mut report,TransitionIssueKind::InvalidEvidence,Stage::Global,None,None,error);false}};
    let mut groups:BTreeMap<(usize,usize),Vec<u64>>=BTreeMap::new();let mut globals=Vec::new();let mut backgrounds=0usize;
    let mut misplaced:BTreeMap<(usize,usize),usize>=BTreeMap::new();
    let mut seen=HashSet::new();
    for primitive in &mut canvas{
        if primitive.id==0||!seen.insert(primitive.id){issue(&mut report,TransitionIssueKind::InvalidHistory,Stage::Global,None,None,"画布含零或重复图元编号，请重新导入有效状态树。");}
        let assignment=view.assignments[&primitive.id];
        if primitive.shape==ShapeKind::Background{
            backgrounds+=1;
            if primitive.stage_assignment.is_some_and(|a|a.stage!=StageTier::Global||a.cell!=0){
                issue(&mut report,TransitionIssueKind::InvalidOwnership,Stage::Global,None,None,"背景图元必须属于全局阶段，请修正背景归属后再切换。");
            }
        }
        match assignment.stage{
            StageTier::Global=>globals.push(primitive.id),
            tier=>{
                let stage=stage_edit::stage(tier);groups.entry((stage.side(),assignment.cell)).or_default().push(primitive.id);
                if stage_edit::cell_for(tier,primitive).ok()!=Some(assignment.cell){*misplaced.entry((stage.side(),assignment.cell)).or_default()+=1;}
            },
        }
        // Freezing legacy metadata cannot change the rendered canvas or any of
        // the model's nine attributes, but makes future ID projections stable.
        primitive.stage_assignment=Some(assignment);
    }
    for ((side,index),count) in misplaced{
        let stage=if side==4{Stage::Coarse}else{Stage::Fine};
        issue(&mut report,TransitionIssueKind::GeometryOutsideOwnedCell,stage,Some(index),Some(count),
            format!("{}有{count}个图形的中心位于其他网格。请把图形移回本格，或修正其阶段／区域归属后再切换。",cell_name(stage,index)));
    }
    let global_count=globals.len().saturating_sub(backgrounds);
    if backgrounds>1{issue(&mut report,TransitionIssueKind::InvalidOwnership,Stage::Global,None,Some(backgrounds),"画布只能有一个全局背景图元，请先删除重复背景。");}
    if global_count>85{issue(&mut report,TransitionIssueKind::GlobalOverflow,Stage::Global,None,Some(global_count),
        format!("全局阶段有{global_count}个普通图形，超过85个的上限。请在修改模式调整全局图形后再切换。"));}
    for (&(side,index),ids) in &groups{
        let stage=if side==4{Stage::Coarse}else{Stage::Fine};let cap=stage.cap();
        if ids.len()>cap{issue(&mut report,if side==4{TransitionIssueKind::CoarseOverflow}else{TransitionIssueKind::FineOverflow},stage,Some(index),Some(ids.len()),
            format!("{}有{}个图形，超过{cap}个的上限。请先移出或删除超额图形。",cell_name(stage,index),ids.len()));}
    }
    let mut closed=BTreeSet::new();
    if evidence_valid{for proof in &evidence.regions{
        let current=groups.get(&(proof.side,proof.index)).map(Vec::as_slice).unwrap_or(&[]);
        // Edits to color, shape geometry within the same cell, or stacking order
        // do not reopen an old phase. Adding/removing/migrating IDs does.
        if same_membership(&proof.primitive_ids,current){closed.insert((proof.side,proof.index));}
    }}
    // At quota, the ordinary scheduler emits END itself before another shape.
    // This is the sole completion rule that does not need a model-END proof.
    for (&(side,index),ids) in &groups{let cap=if side==4{20}else{10};if ids.len()==cap{closed.insert((side,index));}}
    let has_fine_geometry=groups.keys().any(|(side,_)|*side==8);
    let coarse_complete=(0..16).all(|index|closed.contains(&(4,index)));
    let rollback_empty_fine=!has_fine_geometry&&!coarse_complete&&closed.iter().any(|(side,_)|*side==8);
    // Deleting all fine geometry can legitimately return the working frontier
    // to unfinished coarse cells. A genuine zero-shape Fine END remains a
    // historical fact, but must not force an earlier phase to look complete or
    // put Fine markers ahead of the resumed coarse work.
    if !has_fine_geometry&&!coarse_complete{closed.retain(|(side,_)|*side!=8);}
    let has_fine=has_fine_geometry||closed.iter().any(|(side,_)|*side==8);
    let has_coarse=has_fine||rollback_empty_fine||groups.keys().any(|(side,_)|*side==4)||closed.iter().any(|(side,_)|*side==4);
    if has_coarse{
        if global_count<85{issue(&mut report,TransitionIssueKind::GlobalIncomplete,Stage::Global,None,Some(global_count),
            format!("已有粗／细网格内容，但全局阶段只有{global_count}个普通图形，尚未达到85个。请先在修改模式补齐全局阶段，或移除后续阶段内容。"));}
        if backgrounds==0{issue(&mut report,TransitionIssueKind::MissingBackground,Stage::Global,None,None,
            "已有粗／细网格内容，但缺少全局背景。请先在修改模式设置背景颜色后再切换。" );}
    }
    if has_fine_geometry{for index in 0..16{if !closed.contains(&(4,index)){
        let count=groups.get(&(4,index)).map_or(0,Vec::len);
        let why=if count==0{"当前为空，且没有与当前内容对应的真实结束记录"}else{"尚未达到20个图形，也没有与当前内容对应的真实结束记录"};
        issue(&mut report,TransitionIssueKind::CoarseIncompleteBeforeFine,Stage::Coarse,Some(index),Some(count),
            format!("{}尚未完成（{why}），但画布已有细网格内容。请先完成该粗网格，或移除细网格阶段内容；程序不会代填结束记录。",cell_name(Stage::Coarse,index)));
    }}}
    if !report.issues.is_empty(){return Err(report);}
    let by_id=canvas.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();let mut records=vec![region_state::BOS_RECORD];let mut primitive_ids=Vec::new();
    for id in globals{records.push(by_id[&id].to_tokens().map_err(|error|failure(TransitionIssueKind::InvalidHistory,error))?.map(u32::from));primitive_ids.push(id);}
    let mut manual_review_regions=Vec::new();
    for side in [4usize,8]{
        let stage=if side==4{Stage::Coarse}else{Stage::Fine};
        let mut keys=(0..side*side).filter(|index|closed.contains(&(side,*index))||groups.contains_key(&(side,*index))).collect::<Vec<_>>();
        // Ended cells are fixed conditioning. Every unfinished known cell is
        // retained as a suspended visit, with the final visit left active.
        keys.sort_by_key(|index|(!closed.contains(&(side,*index)),*index));
        for (position,&index) in keys.iter().enumerate(){
            records.push(region_state::marker_record(stage,index).map_err(|error|failure(TransitionIssueKind::InvalidHistory,error))?);
            if let Some(ids)=groups.get(&(side,index)){for id in ids{
                records.push(by_id[id].to_tokens().map_err(|error|failure(TransitionIssueKind::InvalidHistory,error))?.map(u32::from));primitive_ids.push(*id);
            }}
            if closed.contains(&(side,index)){records.push(region_state::END_RECORD);}
            else{
                manual_review_regions.push((side,index));
                if position+1<keys.len(){records.push(region_state::INTERRUPT_RECORD);}
            }
        }
    }
    store.validate_context_with_model_ids(doc,node,&records,&primitive_ids).map_err(|error|failure(TransitionIssueKind::InvalidHistory,error))?;
    if primitive_ids.len()!=canvas.len(){return Err(failure(TransitionIssueKind::InvalidHistory,"恢复上下文没有包含完整画布，拒绝切换。"));}
    Ok(PaintResumePlan{canvas,records,primitive_ids,stage_floor:if has_fine{Some(8)}else if has_coarse{Some(4)}else{None},manual_review_regions})
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::document::{OperationSource,PrimitiveStageAssignment};
    fn shape(stage:StageTier,cell:usize)->Primitive{
        let side=stage.side();let (x,y)=if side==1{(256.,256.)}else{let step=512./side as f32;((cell%side)as f32*step+step/2.,(cell/side)as f32*step+step/2.)};
        let mut primitive=Primitive::new(ShapeKind::Rectangle,x,y,10.,8.,[60,100,140]);
        primitive.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:1});primitive
    }
    fn base(globals:usize)->Vec<Primitive>{let mut canvas=vec![Primitive::background([240;3])];canvas.extend((0..globals).map(|_|shape(StageTier::Global,0)));canvas}
    fn document(canvas:Vec<Primitive>)->(Document,RawHistoryStore){
        let mut doc=Document::new();if !canvas.is_empty(){doc.commit_canvas("fixture",OperationSource::User,canvas).unwrap();}
        (doc,RawHistoryStore::new())
    }
    fn empty_coarse_proofs()->CompletionEvidence{CompletionEvidence{regions:(0..16).map(|index|RegionCompletionProof{side:4,index,primitive_ids:Vec::new()}).collect()}}
    fn plan(doc:&Document,store:&RawHistoryStore,evidence:&CompletionEvidence)->Result<PaintResumePlan,TransitionReport>{prepare_paint_resume(doc,store,doc.current,evidence)}
    #[test]
    fn empty_and_partial_global_frontiers_are_legal_without_inventing_content(){
        let (doc,store)=document(Vec::new());let result=plan(&doc,&store,&CompletionEvidence::default()).unwrap();
        assert_eq!(result.records,[region_state::BOS_RECORD]);assert_eq!(result.stage_floor,None);assert!(result.canvas.is_empty());
        let (doc,store)=document(base(17));let result=plan(&doc,&store,&CompletionEvidence::default()).unwrap();
        assert_eq!(result.stage_floor,None);assert_eq!(result.primitive_ids.len(),18);assert!(result.manual_review_regions.is_empty());
        assert!(!result.records.contains(&region_state::END_RECORD));
    }
    #[test]
    fn partial_coarse_frontier_keeps_every_known_shape_and_only_suspends_unfinished_cells(){
        let mut canvas=base(85);canvas.extend((0..7).map(|_|shape(StageTier::Coarse,5)));canvas.extend((0..3).map(|_|shape(StageTier::Coarse,9)));
        let (doc,store)=document(canvas);let before=serde_json::to_vec(&doc.nodes).unwrap();
        let result=plan(&doc,&store,&CompletionEvidence::default()).unwrap();assert_eq!(result.stage_floor,Some(4));
        assert_eq!(result.manual_review_regions,[(4,5),(4,9)]);assert_eq!(result.primitive_ids.len(),doc.primitives.len());
        assert_eq!(result.records.iter().filter(|row|**row==region_state::INTERRUPT_RECORD).count(),1);
        assert!(!result.records.contains(&region_state::END_RECORD));
        let region=region_state::DrawingRegion::from_history(&result.records).unwrap();assert_eq!(region.stage,Stage::Coarse);assert_eq!(region.index,Some(9));
        assert_eq!(result.canvas.iter().map(|p|p.id).collect::<Vec<_>>(),doc.primitives.iter().map(|p|p.id).collect::<Vec<_>>());
        for (old,new) in doc.primitives.iter().zip(&result.canvas){assert_eq!(old.to_tokens().unwrap(),new.to_tokens().unwrap());}
        assert_eq!(serde_json::to_vec(&doc.nodes).unwrap(),before);
    }
    #[test]
    fn fine_before_any_unfinished_coarse_reports_all_sixteen_cells_with_one_based_coordinates(){
        let mut canvas=base(85);canvas.push(shape(StageTier::Fine,0));let(doc,store)=document(canvas);
        let error=plan(&doc,&store,&CompletionEvidence::default()).unwrap_err();
        let issues=error.issues.iter().filter(|i|i.kind==TransitionIssueKind::CoarseIncompleteBeforeFine).collect::<Vec<_>>();
        assert_eq!(issues.len(),16);assert_eq!(issues[0].row_col(),Some((1,1)));assert_eq!(issues[15].row_col(),Some((4,4)));
        assert!(error.message().contains("程序不会代填结束记录"));
    }
    #[test]
    fn genuine_zero_shape_end_is_valid_but_missing_or_stale_old_membership_is_not(){
        let mut canvas=base(85);canvas.extend((0..2).map(|_|shape(StageTier::Fine,0)));let(doc,store)=document(canvas);
        let mut evidence=empty_coarse_proofs();let result=plan(&doc,&store,&evidence).unwrap();
        assert_eq!(result.stage_floor,Some(8));assert_eq!(result.manual_review_regions,[(8,0)]);
        assert_eq!(result.records.iter().filter(|row|**row==region_state::END_RECORD).count(),16);
        evidence.regions[0].primitive_ids=vec![999_999];
        let error=plan(&doc,&store,&evidence).unwrap_err();assert_eq!(error.issues.len(),1);assert_eq!(error.issues[0].index,Some(0));
        evidence.regions[0].primitive_ids.clear();evidence.regions.remove(15);
        let error=plan(&doc,&store,&evidence).unwrap_err();assert_eq!(error.issues.len(),1);assert_eq!(error.issues[0].index,Some(15));
    }
    #[test]
    fn empty_fine_completion_does_not_block_a_legal_coarse_frontier_or_emit_premature_fine_markers(){
        let mut canvas=base(85);canvas.extend((0..3).map(|_|shape(StageTier::Coarse,5)));
        let(doc,store)=document(canvas);
        let evidence=CompletionEvidence{regions:vec![RegionCompletionProof{side:8,index:7,primitive_ids:Vec::new()}]};
        let result=plan(&doc,&store,&evidence).unwrap();assert_eq!(result.stage_floor,Some(4));
        assert_eq!(result.manual_review_regions,[(4,5)]);
        assert!(result.records.iter().filter_map(|row|region_state::marker_identity(row).ok()).all(|(stage,_)|stage==Stage::Coarse));
        assert!(!result.records.contains(&region_state::END_RECORD));
        let region=region_state::DrawingRegion::from_history(&result.records).unwrap();assert_eq!(region.stage,Stage::Coarse);assert_eq!(region.index,Some(5));
        // The same empty Fine proof is meaningful once every coarse cell is
        // actually complete. It is retained as a real zero-shape Fine visit.
        let(doc,store)=document(base(85));let mut evidence=empty_coarse_proofs();
        evidence.regions.push(RegionCompletionProof{side:8,index:7,primitive_ids:Vec::new()});
        let result=plan(&doc,&store,&evidence).unwrap();assert_eq!(result.stage_floor,Some(8));
        assert_eq!(result.records.iter().filter(|row|**row==region_state::END_RECORD).count(),17);
        assert!(result.records.iter().any(|row|region_state::marker_identity(row).ok()==Some((Stage::Fine,7))));
        // An entirely emptied coarse phase is also a coarse frontier, rather
        // than a synthetic Fine visit caused by the old empty completion.
        let result=plan(&doc,&store,&CompletionEvidence{regions:vec![RegionCompletionProof{side:8,index:7,primitive_ids:Vec::new()}]}).unwrap();
        assert_eq!(result.stage_floor,Some(4));assert_eq!(result.records.len(),doc.primitives.len()+1);
    }
    #[test]
    fn historical_color_geometry_and_stacking_edits_do_not_invalidate_same_cell_membership(){
        let mut canvas=base(85);canvas.extend([shape(StageTier::Coarse,0),shape(StageTier::Coarse,0),shape(StageTier::Fine,0)]);
        let(mut doc,store)=document(canvas);let ids=doc.primitives[86..88].iter().map(|p|p.id).collect::<Vec<_>>();let mut evidence=empty_coarse_proofs();evidence.regions[0].primitive_ids=ids;
        let mut changed=doc.primitives.clone();changed[86].color=[120,80,40];changed[86].width=18.;changed.swap(86,87);
        doc.commit_canvas("old coarse edit",OperationSource::User,changed).unwrap();
        assert!(plan(&doc,&store,&evidence).is_ok());
        let mut changed=doc.primitives.clone();changed.remove(86);doc.commit_canvas("remove old coarse",OperationSource::User,changed).unwrap();
        assert!(plan(&doc,&store,&evidence).unwrap_err().issues.iter().any(|i|i.kind==TransitionIssueKind::CoarseIncompleteBeforeFine&&i.index==Some(0)));
    }
    #[test]
    fn full_quota_is_a_real_scheduler_completion_and_multiple_partial_fine_visits_remain_legal(){
        let mut canvas=base(85);for cell in 0..16{canvas.extend((0..20).map(|_|shape(StageTier::Coarse,cell)));}
        canvas.extend((0..2).map(|_|shape(StageTier::Fine,7)));canvas.extend((0..3).map(|_|shape(StageTier::Fine,20)));
        let(doc,store)=document(canvas);let result=plan(&doc,&store,&CompletionEvidence::default()).unwrap();
        assert_eq!(result.stage_floor,Some(8));assert_eq!(result.manual_review_regions,[(8,7),(8,20)]);
        assert_eq!(result.records.iter().filter(|row|**row==region_state::END_RECORD).count(),16);
        let region=region_state::DrawingRegion::from_history(&result.records).unwrap();assert_eq!(region.index,Some(20));assert_eq!(region.generated_in_region,3);
        assert_eq!(result.primitive_ids.len(),doc.primitives.len());
    }
    #[test]
    fn conversion_reports_caps_global_prerequisites_and_wrong_cell_geometry_together(){
        let mut canvas=base(3);canvas.extend((0..21).map(|_|shape(StageTier::Coarse,5)));canvas.extend((0..11).map(|_|shape(StageTier::Fine,18)));
        canvas.last_mut().unwrap().x=480.;let(doc,store)=document(canvas);
        let error=plan(&doc,&store,&CompletionEvidence::default()).unwrap_err();
        for kind in [TransitionIssueKind::GlobalIncomplete,TransitionIssueKind::CoarseOverflow,TransitionIssueKind::FineOverflow,TransitionIssueKind::GeometryOutsideOwnedCell]{
            assert!(error.issues.iter().any(|issue|issue.kind==kind),"missing issue {kind:?}");
        }
        let (doc,store)=document(base(86));assert!(plan(&doc,&store,&CompletionEvidence::default()).unwrap_err().issues.iter().any(|i|i.kind==TransitionIssueKind::GlobalOverflow));
        let(mut canvas)=base(85);canvas.remove(0);canvas.push(shape(StageTier::Coarse,0));let(doc,store)=document(canvas);
        assert!(plan(&doc,&store,&CompletionEvidence::default()).unwrap_err().issues.iter().any(|i|i.kind==TransitionIssueKind::MissingBackground));
    }
    #[test]
    fn malformed_evidence_is_rejected_without_trusting_context_end_tokens(){
        let proof=RegionCompletionProof{side:4,index:0,primitive_ids:Vec::new()};
        assert!(CompletionEvidence{regions:vec![proof.clone(),proof]}.validate().is_err());
        assert!(CompletionEvidence{regions:vec![RegionCompletionProof{side:8,index:64,primitive_ids:Vec::new()}]}.validate().is_err());
        assert!(CompletionEvidence{regions:vec![RegionCompletionProof{side:4,index:0,primitive_ids:vec![1,1]}]}.validate().is_err());
        let mut canvas=base(85);canvas.push(shape(StageTier::Fine,0));let(doc,mut store)=document(canvas);
        let ids=doc.primitives.iter().map(|p|p.id).collect::<Vec<_>>();let mut raw=vec![region_state::BOS_RECORD];
        raw.extend(doc.primitives[..86].iter().map(|p|p.to_tokens().unwrap().map(u32::from)));
        for index in 0..16{raw.extend([region_state::marker_record(Stage::Coarse,index).unwrap(),region_state::END_RECORD]);}
        raw.extend([region_state::marker_record(Stage::Fine,0).unwrap(),doc.primitives[86].to_tokens().unwrap().map(u32::from)]);
        store.set_control_overlay_with_ids(&doc,doc.current,&raw,&ids).unwrap();
        // Merely seeing sixteen END carriers in raw is not authoritative. A
        // redraw prefix can contain exactly those synthetic conditioning rows.
        assert!(plan(&doc,&store,&CompletionEvidence::default()).is_err());
    }
}
