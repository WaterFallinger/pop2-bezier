//! Transactional adoption of an explicitly accepted serialized reference.
//! The approximation is prepared elsewhere; this module never edits old nodes.
use crate::{document::{Document,NodeId,OperationSource,Primitive,ShapeKind,StageTier},history_context::{RawHistoryStore,RawRecord},
    mode_transition::{CompletionEvidence,RegionCompletionProof},region_state};
use std::collections::{BTreeMap,HashSet,VecDeque};

pub struct ImportOutcome{pub node:NodeId,pub primitives:usize,pub reset_boundary:bool,pub resume_regions:Vec<(usize,usize)>}

struct PreparedReference{canvas:Vec<Primitive>,ids:Vec<u64>,quota_evidence:CompletionEvidence,frontier:usize,resume_regions:Vec<(usize,usize)>}

fn prepared(doc:&Document,origin:NodeId,source:&[Primitive],raw:&[RawRecord])->Result<PreparedReference,String>{
    if doc.current!=origin||!doc.primitives.is_empty()||!doc.materialize(origin)?.is_empty(){return Err("导入参考只能用于原来的空白画布；当前画布已经变化，未导入任何内容".into());}
    if source.is_empty()||source.len()>crate::document::project_data::MAX_CANVAS{return Err("参考序列没有有效图元，或超过单画布上限".into());}
    if source.first().is_none_or(|p|p.shape!=ShapeKind::Background)||source.iter().filter(|p|p.shape==ShapeKind::Background).count()!=1{
        return Err("参考序列必须以唯一的不透明背景开始".into());
    }
    if raw.first()!=Some(&region_state::BOS_RECORD)||raw.len()>crate::document::project_data::MAX_RECORDS{return Err("参考序列缺少规范起始标记或超过记录上限".into());}
    region_state::DrawingRegion::from_history(raw)?;
    let mut candidate=source.to_vec();for p in &mut candidate{p.id=0;}
    let candidate=doc.preview_canvas(candidate)?;
    let mut groups=BTreeMap::<(usize,usize),VecDeque<&Primitive>>::new();
    for p in &candidate{
        if p.stage_assignment.is_none(){return Err("参考图元缺少绘制阶段归属".into());}
        let assignment=p.stage_assignment.unwrap();
        if crate::history_context::stage_edit::cell_for(assignment.stage,p)?!=assignment.cell{return Err("参考图元量化后的中心不属于其序列区域".into());}
        groups.entry((assignment.stage.side(),assignment.cell)).or_default().push_back(p);
    }
    let mut quota_evidence=CompletionEvidence::default();
    for (&(side,index),rows) in &groups{
        if side==1{continue;}
        let quota=if side==4{20}else{10};
        if rows.len()>quota{return Err("参考图元超过了对应区域的原有图元数上限".into());}
        if rows.len()==quota{
            let mut primitive_ids=rows.iter().map(|p|p.id).collect::<Vec<_>>();primitive_ids.sort_unstable();
            quota_evidence.regions.push(RegionCompletionProof{side,index,primitive_ids});
        }
    }
    let mut active=(1usize,0usize);let mut floor=1;let mut ids=Vec::new();let mut closed=HashSet::new();
    for row in raw{
        if region_state::is_marker(row){let (stage,index)=region_state::marker_identity(row)?;
            active=(stage.side(),index);floor=floor.max(stage.side());closed.remove(&active);
        }else if row==&region_state::END_RECORD{
            let count=candidate.iter().filter(|p|p.stage_assignment.is_some_and(|a|(a.stage.side(),a.cell)==active)).count();
            let quota=if active.0==4{20}else{10};
            if count<quota{return Err("参考的RMSE达标区域必须保留为手动暂停，不能伪装成永久结束；请重新序列化".into());}
            closed.insert(active);
        }else if region_state::validate_visible(row).is_ok(){
            let p=groups.get_mut(&active).and_then(VecDeque::pop_front).ok_or("参考序列引用了当前区域中不存在或重复的图元")?;
            if p.to_tokens()?.map(u32::from)!=*row{return Err("参考序列与预览图元的九属性顺序不一致".into());}
            ids.push(p.id);
        }
    }
    let has_fine=candidate.iter().any(|p|p.stage_assignment.is_some_and(|a|a.stage==StageTier::Fine));
    if floor==1&&(has_fine||candidate.iter().filter(|p|p.shape!=ShapeKind::Background&&p.stage_assignment.is_some_and(|a|a.stage==StageTier::Global)).count()>=85){floor=4;}
    for (&(side,_),remaining) in &groups{
        if remaining.is_empty(){continue;}
        if side<=floor||floor!=4||side!=8{return Err("参考序列遗漏了当前已启用阶段的图元".into());}
    }
    if groups.iter().any(|(&(side,_),rows)|side==8&&!rows.is_empty())&&closed.iter().filter(|(side,_)|*side==4).count()==16{
        return Err("粗网格已经全部结束时，不应丢弃已生成的细网格上下文".into());
    }
    let completed=quota_evidence.regions.iter().map(|p|(p.side,p.index)).collect::<HashSet<_>>();
    let mut resume_regions=Vec::new();
    if floor==4{
        resume_regions.extend((0..16).filter(|index|!completed.contains(&(4,*index))).map(|index|(4,index)));
        if has_fine||resume_regions.is_empty(){resume_regions.extend((0..64).filter(|index|!completed.contains(&(8,*index))).map(|index|(8,index)));}
    }else if floor==8{
        resume_regions.extend((0..64).filter(|index|!completed.contains(&(8,*index))).map(|index|(8,index)));
    }
    Ok(PreparedReference{canvas:candidate,ids,quota_evidence,frontier:floor,resume_regions})
}

/// A reference serializer's real END records are trusted external conditioning,
/// rather than fabricated model decisions. Only those records establish ends.
pub fn import_reference(doc:&mut Document,history:&mut RawHistoryStore,origin:NodeId,source:&[Primitive],raw:&[RawRecord])->Result<ImportOutcome,String>{
    let PreparedReference{canvas,ids,quota_evidence,frontier,resume_regions}=prepared(doc,origin,source,raw)?;
    // Preflight the complete payload in a tiny isolated document before any
    // operation can touch the user's large state tree.
    let mut probe=Document::new();probe.settings=doc.settings.clone();probe.prompt=doc.prompt.clone();
    let test=probe.commit_canvas("参考预检",OperationSource::User,canvas.clone())?;
    let mut test_history=RawHistoryStore::new();test_history.set_control_overlay_with_ids(&probe,test,raw,&ids)?;
    test_history.set_reference_frontier(&probe,test,frontier)?;
    test_history.set_completion_baseline(&probe,test,&quota_evidence)?;
    test_history.record_completed_regions(&probe,test,raw,&ids,0,&[],true)?;
    let limits=crate::document::project_data::Limits::default();let mut budget=crate::document::project_data::Budget::new(limits);
    let mut work=crate::document::project_data::Work::new(limits.work);
    let projections=probe.validate_project(&mut budget,&mut work)?;
    test_history.validate_project(&probe,&projections,&mut budget,&mut work)?;
    let old_prompt=doc.prompt.clone();let old_settings=doc.settings.clone();
    let mut first_new=None;
    let result=(||{
        let floor=history.stage_floor(doc,origin)?.unwrap_or(1);
        let restart=if floor!=1||history.requires_normal_restart(doc,origin)?||!history.completion_evidence(doc,origin)?.regions.is_empty(){
            let node=doc.commit_control_canvas("空白画布参考导入起点",Vec::new())?;first_new=Some(node);
            history.restart_empty_reference_origin(doc,node)?;true
        }else{false};
        let node=doc.commit_canvas("导入参考图像序列",OperationSource::User,canvas)?;
        if first_new.is_none(){first_new=Some(node);}
        history.set_control_overlay_with_ids(doc,node,raw,&ids)?;
        history.set_reference_frontier(doc,node,frontier)?;
        history.set_completion_baseline(doc,node,&quota_evidence)?;
        history.record_completed_regions(doc,node,raw,&ids,0,&[],true)?;
        Ok(ImportOutcome{node,primitives:doc.primitives.len(),reset_boundary:restart,resume_regions})
    })();
    if result.is_err(){
        let rollback_error=first_new.and_then(|first|doc.plan_delete_subtrees(&[first]).and_then(|plan|history.apply_delete_subtrees(doc,plan)).err());
        // A parent history node contains its committed prompt/settings. Its
        // current uncommitted editor drafts are independent and must survive.
        doc.prompt=old_prompt;doc.settings=old_settings;
        if let Some(e)=rollback_error{return Err(format!("参考未能导入，回滚未完成，请保留当前窗口：{e}"));}
    }
    result
}

#[cfg(test)]mod tests{
    use super::*;use crate::document::{PrimitiveStageAssignment,StageTier};
    fn source()->(Vec<Primitive>,Vec<RawRecord>){
        let mut bg=Primitive::background([240;3]);bg.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});
        let mut all=vec![bg];let mut raw=vec![region_state::BOS_RECORD,all[0].to_tokens().unwrap().map(u32::from),region_state::marker_record(region_state::Stage::Coarse,0).unwrap()];
        for index in 0..20{let mut p=Primitive::new(ShapeKind::Rectangle,20.+index as f32,30.,8.,10.,[40,80,120]);
            p.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Coarse,cell:0,queue_order:index+1});raw.push(p.to_tokens().unwrap().map(u32::from));all.push(p);}
        raw.push(region_state::END_RECORD);(all,raw)
    }
    #[test]fn explicit_import_preserves_old_tree_drafts_normalized_geometry_and_real_encoder_end(){
        let mut doc=Document::new();let mut history=RawHistoryStore::new();doc.prompt="保持草稿".into();let old=doc.materialize(0).unwrap();
        let (source,raw)=source();let result=import_reference(&mut doc,&mut history,0,&source,&raw).unwrap();
        assert_eq!(result.node,1);assert!(!result.reset_boundary);assert_eq!(doc.nodes.len(),2);assert_eq!(doc.prompt,"保持草稿");
        assert_eq!(doc.materialize(0).unwrap(),old);assert_eq!(doc.nodes[0].children,[result.node]);assert_eq!(history.context_for(&doc,doc.current).unwrap(),Some(raw));
        assert!(history.completion_evidence(&doc,doc.current).unwrap().regions.iter().any(|p|(p.side,p.index)==(4,0)));
        assert!(doc.primitives.iter().all(|p|p.id!=0));
    }
    #[test]fn rejects_nonempty_stale_or_bad_payload_without_creating_a_node(){
        let (source,raw)=source();let mut doc=Document::new();let mut history=RawHistoryStore::new();
        doc.prompt="未提交的提示词草稿".into();doc.settings.cfg=2.5;
        let before=serde_json::to_vec(&doc.project_ref()).unwrap();
        assert!(import_reference(&mut doc,&mut history,1,&source,&raw).is_err());assert_eq!(doc.nodes.len(),1);
        let mut bad=raw.clone();bad.pop();bad.pop();assert!(import_reference(&mut doc,&mut history,0,&source,&bad).is_err());assert_eq!(doc.nodes.len(),1);
        assert_eq!(serde_json::to_vec(&doc.project_ref()).unwrap(),before);
        doc.commit_generated(Primitive::background([20;3])).unwrap();let node=doc.current;let before=doc.primitives.clone();
        assert!(import_reference(&mut doc,&mut history,node,&source,&raw).is_err());assert_eq!(doc.primitives,before);assert_eq!(doc.nodes.len(),2);
    }
    #[test]fn empty_late_stage_resets_new_branch_and_preserves_old_phase(){
        let mut doc=Document::new();let mut history=RawHistoryStore::new();
        let model=doc.commit_generated(Primitive::background([20;3])).unwrap();let model_ids=vec![doc.primitives[0].id];
        let model_raw=vec![region_state::BOS_RECORD,doc.primitives[0].to_tokens().unwrap().map(u32::from),region_state::marker_record(region_state::Stage::Fine,0).unwrap(),region_state::END_RECORD];
        history.attach_with_model_ids(&doc,model,&model_raw,&model_ids).unwrap();
        history.record_completed_regions(&doc,model,&model_raw,&model_ids,0,&[],false).unwrap();
        let old=doc.commit_canvas("空的细网格历史",OperationSource::User,Vec::new()).unwrap();
        let raw=vec![region_state::BOS_RECORD,region_state::marker_record(region_state::Stage::Fine,0).unwrap(),region_state::END_RECORD];
        history.set_control_overlay_with_ids(&doc,old,&raw,&[]).unwrap();assert_eq!(history.stage_floor(&doc,old).unwrap(),Some(8));
        assert!(!history.completion_evidence(&doc,old).unwrap().regions.is_empty());
        let mut bg=Primitive::background([240;3]);bg.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});
        let raw=vec![region_state::BOS_RECORD,bg.to_tokens().unwrap().map(u32::from)];
        let result=import_reference(&mut doc,&mut history,old,&[bg],&raw).unwrap();assert!(result.reset_boundary);
        assert_eq!(history.stage_floor(&doc,doc.current).unwrap().unwrap_or(1),1);
        assert_eq!(history.editing_stage(&doc,doc.current).unwrap(),region_state::Stage::Global);
        assert_eq!(history.stage_floor(&doc,old).unwrap(),Some(8));
        assert!(history.completion_evidence(&doc,doc.current).unwrap().regions.is_empty());
        assert!(!history.completion_evidence(&doc,old).unwrap().regions.is_empty());
        let path=std::env::temp_dir().join(format!("painter-reference-import-{}.pproj",std::process::id()));
        crate::project_file::save_project(&path,&doc,&history).unwrap();let (loaded,raw)=crate::project_file::load_project(&path).unwrap();
        assert_eq!(loaded.primitives,doc.primitives);assert_eq!(raw.stage_floor(&loaded,old).unwrap(),Some(8));
        assert_eq!(raw.editing_stage(&loaded,loaded.current).unwrap(),region_state::Stage::Global);
        assert!(raw.completion_evidence(&loaded,loaded.current).unwrap().regions.is_empty());std::fs::remove_file(path).unwrap();
    }
    fn owned(stage:StageTier,cell:usize,order:u64)->Primitive{
        let side=stage.side();let mut p=Primitive::new(ShapeKind::Rectangle,(cell%side)as f32*512./side as f32+20.,
            (cell/side)as f32*512./side as f32+20.,8.,10.,[60,100,140]);
        p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:order});p
    }
    #[test]fn paused_coarse_import_preserves_future_fine_and_does_not_grant_rmse_completion(){
        let (mut source,mut raw)=source();
        let coarse=owned(StageTier::Coarse,1,30);source.push(coarse.clone());
        raw.extend([region_state::marker_record(region_state::Stage::Coarse,1).unwrap(),coarse.to_tokens().unwrap().map(u32::from),region_state::INTERRUPT_RECORD]);
        for index in 0..10{source.push(owned(StageTier::Fine,0,40+index));}
        source.push(owned(StageTier::Fine,1,60));
        let mut doc=Document::new();let mut history=RawHistoryStore::new();let result=import_reference(&mut doc,&mut history,0,&source,&raw).unwrap();
        assert_eq!(doc.primitives.len(),source.len());
        assert!(doc.primitives.iter().zip(&source).all(|(a,b)|a.to_tokens().unwrap()==b.to_tokens().unwrap()));
        assert_eq!(history.stage_floor(&doc,doc.current).unwrap(),Some(4));
        let input=history.effective_history(&doc,doc.current).unwrap();assert_eq!(input.records,raw);
        assert_eq!(input.primitive_ids.len(),22);
        assert!(!input.records.iter().any(|r|region_state::marker_identity(r).is_ok_and(|(stage,_)|stage==region_state::Stage::Fine)));
        let proof=history.completion_evidence(&doc,doc.current).unwrap();
        assert!(proof.regions.iter().any(|p|(p.side,p.index)==(4,0)));
        assert!(proof.regions.iter().any(|p|(p.side,p.index)==(8,0)));
        assert!(!proof.regions.iter().any(|p|(p.side,p.index)==(4,1)||(p.side,p.index)==(8,1)));
        assert!(!result.resume_regions.contains(&(4,0)));assert!(!result.resume_regions.contains(&(8,0)));
        assert!(result.resume_regions.contains(&(4,1))&&result.resume_regions.contains(&(8,1)));
        let selection=crate::redraw::RedrawSelection::from_reference_regions(&result.resume_regions).unwrap();
        assert!(selection.locked_fine().is_empty());selection.validate_empty(&doc.primitives,&history.stage_map(&doc,doc.current).unwrap()).unwrap();
        assert_eq!(history.pending_stage_context(&doc,doc.current).unwrap().len(),11);
        let path=std::env::temp_dir().join(format!("painter-paused-reference-{}.pproj",std::process::id()));
        crate::project_file::save_project(&path,&doc,&history).unwrap();let(loaded,stored)=crate::project_file::load_project(&path).unwrap();
        assert_eq!(loaded.primitives,doc.primitives);assert_eq!(stored.effective_history(&loaded,loaded.current).unwrap().records,raw);
        assert_eq!(stored.stage_floor(&loaded,loaded.current).unwrap(),Some(4));assert_eq!(stored.pending_stage_ids(&loaded,loaded.current).unwrap().len(),11);
        std::fs::remove_file(path).unwrap();
    }
    #[test]fn global_implicit_pause_and_empty_coarse_frontier_never_invent_an_interrupt(){
        let mut doc=Document::new();let mut history=RawHistoryStore::new();
        let mut bg=Primitive::background([240;3]);bg.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});
        let raw=vec![region_state::BOS_RECORD,bg.to_tokens().unwrap().map(u32::from)];
        let imported=import_reference(&mut doc,&mut history,0,&[bg.clone()],&raw).unwrap();
        assert!(imported.resume_regions.is_empty());assert_eq!(history.stage_floor(&doc,doc.current).unwrap(),None);
        assert!(!history.context_for(&doc,doc.current).unwrap().unwrap().contains(&region_state::INTERRUPT_RECORD));
        let mut source=vec![bg];for n in 0..85{source.push(owned(StageTier::Global,0,n+1));}source.push(owned(StageTier::Fine,1,100));
        let mut raw=vec![region_state::BOS_RECORD];raw.extend(source[..86].iter().map(|p|p.to_tokens().unwrap().map(u32::from)));
        let mut doc=Document::new();let mut history=RawHistoryStore::new();let imported=import_reference(&mut doc,&mut history,0,&source,&raw).unwrap();
        assert_eq!(history.stage_floor(&doc,doc.current).unwrap(),Some(4));assert_eq!(history.effective_history(&doc,doc.current).unwrap().records,raw);
        assert_eq!(imported.resume_regions.iter().filter(|(side,_)|*side==4).count(),16);
        assert_eq!(imported.resume_regions.iter().filter(|(side,_)|*side==8).count(),64);
    }
    #[test]fn a_fully_quota_completed_coarse_reference_defaults_to_new_fine_work(){
        let (mut source,mut raw)=source();
        for cell in 1..16{
            raw.push(region_state::marker_record(region_state::Stage::Coarse,cell).unwrap());
            for n in 0..20{let p=owned(StageTier::Coarse,cell,(cell*20+n+1)as u64);raw.push(p.to_tokens().unwrap().map(u32::from));source.push(p);}
            raw.push(region_state::END_RECORD);
        }
        let mut doc=Document::new();let mut history=RawHistoryStore::new();
        let imported=import_reference(&mut doc,&mut history,0,&source,&raw).unwrap();
        assert_eq!(history.stage_floor(&doc,doc.current).unwrap(),Some(4));
        assert_eq!(imported.resume_regions,(0..64).map(|index|(8,index)).collect::<Vec<_>>());
        assert_eq!(history.completion_evidence(&doc,doc.current).unwrap().regions.len(),16);
        assert_eq!(history.context_for(&doc,doc.current).unwrap(),Some(raw));
    }
}
