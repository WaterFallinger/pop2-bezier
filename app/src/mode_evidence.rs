//! Typed completion provenance and explicit ordinary-mode restart boundaries.
//! Conditioning END tokens are syntax, never proof of a model decision.
use super::*;
use crate::mode_transition::{CompletionEvidence,RegionCompletionProof};

#[derive(Clone,Debug,Default)]
pub(super) struct ModeEvidenceNode {
    pub baseline:Option<CompletionEvidence>,
    pub events:CompletionEvidence,
    pub overlay:Option<CompletionEvidence>,
    pub redraw_scope:Option<Vec<(usize,usize)>>,
    pub normal_restart:bool,
}
#[derive(Serialize)]
pub(super) struct ModeEvidenceRef<'a>{
    node:NodeId,baseline:Option<&'a CompletionEvidence>,events:&'a CompletionEvidence,
    overlay:Option<&'a CompletionEvidence>,redraw_scope:Option<&'a [(usize,usize)]>,normal_restart:bool,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]
pub(super) struct ModeEvidenceData{
    node:NodeId,baseline:Option<EvidenceData>,events:EvidenceData,overlay:Option<EvidenceData>,
    redraw_scope:Option<project_data::LimitedVec<(usize,usize),81>>,normal_restart:bool,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct EvidenceData{
    regions:project_data::LimitedVec<ProofData,80>,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct ProofData{
    side:usize,index:usize,
    primitive_ids:project_data::LimitedVec<project_data::Counted<u64,{project_data::RECORDS}>,{project_data::MAX_CANVAS}>,
}
impl EvidenceData{fn into_value(self)->CompletionEvidence{CompletionEvidence{regions:self.regions.0.into_iter().map(|p|RegionCompletionProof{
    side:p.side,index:p.index,primitive_ids:p.primitive_ids.0.into_iter().map(|id|id.0).collect()}).collect()}}}

fn key(proof:&RegionCompletionProof)->(usize,usize){(proof.side,proof.index)}
fn checked_keys(keys:&[(usize,usize)])->Result<(),String>{
    let mut seen=HashSet::new();
    if keys.len()>80||keys.iter().any(|&(side,index)|!matches!(side,4|8)||index>=side*side||!seen.insert((side,index))){
        return Err("区域完成证据的范围无效".into());}Ok(())
}
fn checked_scope(keys:&[(usize,usize)])->Result<(),String>{
    let mut seen=HashSet::new();
    if keys.len()>81||keys.iter().any(|&(side,index)|!matches!(side,1|4|8)||index>=side*side||!seen.insert((side,index))){
        return Err("局部重绘范围无效".into());}Ok(())
}
fn validate_global_redraw_prefix(store:&RawHistoryStore,doc:&Document,node:NodeId,raw:&[RawRecord],ids:&[u64])->Result<(),String>{
    let canvas=doc.materialize(node)?;let map=store.stage_map(doc,node)?;
    let view=stage_edit::view(&canvas,&map,false)?;
    if canvas.iter().any(|p|view.assignments[&p.id].stage!=StageTier::Global)
        ||canvas.iter().filter(|p|p.shape!=ShapeKind::Background).count()>85 {
        return Err("全图续画前必须清空全部粗网格和细网格，且全图图形不能超过85个".into());
    }
    if !canvas.is_empty()&&(canvas[0].shape!=ShapeKind::Background
        ||canvas.iter().skip(1).any(|p|p.shape==ShapeKind::Background)) {
        return Err("全图续画必须保留唯一的原有背景".into());
    }
    if ids!=canvas.iter().map(|p|p.id).collect::<Vec<_>>()||raw.len()!=canvas.len()+1||raw.first()!=Some(&BOS_RECORD){
        return Err("全图续画前缀必须完整匹配当前全图画布".into());
    }
    for (p,row)in canvas.iter().zip(&raw[1..]){if p.to_tokens()?.map(u32::from)!=*row{
        return Err("全图续画前缀属性与当前画布不一致".into());
    }}
    Ok(())
}
fn memberships(store:&RawHistoryStore,doc:&Document,node:NodeId)->Result<BTreeMap<(usize,usize),Vec<u64>>,String>{
    let canvas=doc.materialize(node)?;let stages=store.stage_map(doc,node)?;let view=stage_edit::view(&canvas,&stages,false)?;
    let mut groups=BTreeMap::<(usize,usize),Vec<u64>>::new();
    for p in canvas{let a=view.assignments[&p.id];if a.stage!=StageTier::Global{groups.entry((a.stage.side(),a.cell)).or_default().push(p.id);}}
    for ids in groups.values_mut(){ids.sort_unstable();}Ok(groups)
}
fn filtered(value:CompletionEvidence,members:&BTreeMap<(usize,usize),Vec<u64>>)->CompletionEvidence{
    CompletionEvidence{regions:value.regions.into_iter().filter(|proof|members.get(&key(proof)).map(Vec::as_slice).unwrap_or(&[])==proof.primitive_ids.as_slice()).collect()}
}
fn replace(map:&mut BTreeMap<(usize,usize),RegionCompletionProof>,value:&CompletionEvidence){for proof in &value.regions{map.insert(key(proof),proof.clone());}}
fn unchanged_boundary(store:&RawHistoryStore,doc:&Document,node:NodeId)->Result<NodeId,String>{
    let state=doc.nodes.get(node).ok_or("模式切换状态不存在")?;
    let parent=state.parent.ok_or("模式切换必须创建新的控制状态")?;
    if state.source!=OperationSource::User{return Err("模式切换只能发生在新的用户控制状态".into());}
    let old=doc.materialize(parent)?;let new=doc.materialize(node)?;
    if old.len()!=new.len()||old.iter().zip(&new).any(|(a,b)|{let mut b=b.clone();b.stage_assignment=a.stage_assignment;&b!=a}){
        return Err("模式切换不能增加、删除、重排或修改已有图形".into());
    }
    if store.stage_map(doc,parent)?!=store.stage_map(doc,node)?{return Err("模式切换不能改变已有图元的阶段归属".into());}
    Ok(parent)
}

impl RawHistoryStore {
    /// Freeze this before any branch can publish an exact-origin overlay.
    pub fn completion_evidence(&self,doc:&Document,node:NodeId)->Result<CompletionEvidence,String>{
        if node>=doc.nodes.len(){return Err("区域完成证据状态不存在".into());}
        let mut chain=Vec::new();let mut cursor=Some(node);
        while let Some(id)=cursor{
            chain.push(id);
            if self.mode_evidence.get(&id).is_some_and(|e|e.baseline.is_some()){break;}
            cursor=doc.nodes[id].parent;
        }
        let mut result=BTreeMap::new();
        let mut legacy_redraw=chain.last().copied().map(|base|self.requires_normal_restart(doc,base)).transpose()?.unwrap_or(false);
        for id in chain.into_iter().rev(){
            let state=&doc.nodes[id];let typed=self.mode_evidence.get(&id);
            if state.source==OperationSource::User&&state.title=="准备局部重绘"{legacy_redraw=true;}
            if let Some(data)=typed{
                if let Some(baseline)=&data.baseline{result.clear();replace(&mut result,baseline);}
                if data.redraw_scope.is_some(){legacy_redraw=true;}
                if data.normal_restart{legacy_redraw=false;}
                replace(&mut result,&data.events);
            }else if !legacy_redraw&&state.source==OperationSource::Model
                &&self.entries.get(&id).is_some_and(|delta|delta.tail.iter().any(|row|row==&crate::region_state::END_RECORD)){
                // END-free model steps need no materialization or raw replay.
                let raw=self.resolve(doc,HistoryBase::Node(id))?;
                if let Some(parent)=state.parent{
                    let creation=if self.entries.contains_key(&parent){self.resolve(doc,HistoryBase::Node(parent))?}else{canvas_records(doc,parent)?};
                    let public=self.control_overlays.get(&parent);
                    let start=std::iter::once(creation.as_slice()).chain(public.map(Vec::as_slice))
                        .filter(|prefix|raw.starts_with(prefix)).map(<[RawRecord]>::len).max();
                    if let Some(start)=start{replace(&mut result,&self.proofs_from_tail(doc,id,&raw,start,&[])?);}
                }
            }
            if id==node{
                if let Some(overlay)=typed.and_then(|e|e.overlay.as_ref()){replace(&mut result,overlay);}
                else if !legacy_redraw{
                    if let Some(raw)=self.control_overlays.get(&id){let creation=if self.entries.contains_key(&id){self.resolve(doc,HistoryBase::Node(id))?}else{canvas_records(doc,id)?};
                        if raw.starts_with(&creation){replace(&mut result,&self.proofs_from_tail(doc,id,raw,creation.len(),&[])?);}
                    }
                }
            }
        }
        let members=memberships(self,doc,node)?;
        Ok(filtered(CompletionEvidence{regions:result.into_values().collect()},&members))
    }

    /// Only for a newly attached first child. Later origin overlays must never
    /// leak into its immutable creation-time completion evidence.
    pub fn set_completion_baseline(&mut self,doc:&Document,node:NodeId,evidence:&CompletionEvidence)->Result<(),String>{
        if node>=doc.nodes.len(){return Err("完成证据快照状态不存在".into());}
        if self.mode_evidence.get(&node).is_some_and(|e|e.baseline.is_some()||!e.events.regions.is_empty()||e.overlay.is_some()){
            return Err("不能覆盖已经发布的完成证据快照".into());
        }
        let value=filtered(evidence.clone(),&memberships(self,doc,node)?);
        self.mode_evidence.entry(node).or_default().baseline=Some(value);Ok(())
    }

    pub fn begin_redraw_origin(&mut self,doc:&Document,node:NodeId,scope:&[(usize,usize)],raw:&[RawRecord],ids:&[u64])->Result<(),String>{
        checked_scope(scope)?;
        let parent=unchanged_boundary(self,doc,node)?;
        self.validate_context_with_model_ids(doc,node,raw,ids)?;
        if self.mode_evidence.contains_key(&node){return Err("局部重绘来源已经登记".into());}
        let global=scope.contains(&(1,0));
        if global {validate_global_redraw_prefix(self,doc,node,raw,ids)?;}
        let mut baseline=if global{CompletionEvidence::default()}else{filtered(self.completion_evidence(doc,parent)?,&memberships(self,doc,node)?)};
        // Reopening a selected region revokes its old completion even when
        // all its existing primitives are kept byte-for-byte. Only a new END
        // appended by this request can prove that region complete again.
        baseline.regions.retain(|proof|!scope.contains(&key(proof)));
        if global {
            // Reset only this fresh immutable branch. The source branch and
            // its later-stage END/provenance remain untouched.
            let base=if self.entries.contains_key(&parent){HistoryBase::Node(parent)}else{HistoryBase::Canvas(parent)};
            let previous=self.resolve(doc,base)?;let keep=common_prefix(&previous,raw);
            let previous_ids=self.resolve_model_ids(doc,base)?;let id_keep=previous_ids.iter().zip(ids).take_while(|(a,b)|a==b).count();
            self.entries.insert(node,HistoryDelta{base,keep,tail:raw[keep..].to_vec()});
            self.model_ids.insert(node,ModelIdDelta{base,keep:id_keep,tail:ids[id_keep..].to_vec(),phase_floor:1});
            self.control_overlays.remove(&node);self.overlay_model_ids.remove(&node);self.overlay_phase_floors.remove(&node);self.overlay_manual_reviews.remove(&node);
            self.set_manual_reviews(node,Vec::new());
        }
        self.mode_evidence.insert(node,ModeEvidenceNode{baseline:Some(baseline),redraw_scope:Some(scope.to_vec()),..Default::default()});Ok(())
    }

    pub fn requires_normal_restart(&self,doc:&Document,node:NodeId)->Result<bool,String>{
        if node>=doc.nodes.len(){return Err("模式来源状态不存在".into());}
        let mut cursor=Some(node);
        while let Some(id)=cursor{
            if let Some(entry)=self.mode_evidence.get(&id){
                if entry.normal_restart{return Ok(false);}
                if entry.redraw_scope.is_some(){return Ok(true);}
            }
            let state=&doc.nodes[id];
            // A legacy title only removes trust; it never proves completion.
            if state.source==OperationSource::User&&state.title=="准备局部重绘"{return Ok(true);}
            cursor=state.parent;
        }Ok(false)
    }

    pub fn record_completed_regions(&mut self,doc:&Document,node:NodeId,raw:&[RawRecord],ids:&[u64],append_from:usize,scope:&[(usize,usize)],overlay:bool)->Result<(),String>{
        checked_scope(scope)?;
        if append_from>raw.len(){return Err("完成证据的模型追加游标无效".into());}
        if !raw[append_from..].iter().any(|row|row==&crate::region_state::END_RECORD){return Ok(());}
        self.validate_context_with_model_ids(doc,node,raw,ids)?;
        let proofs=self.proofs_from_tail(doc,node,raw,append_from,scope)?;
        if proofs.regions.is_empty(){return Ok(());}
        let data=self.mode_evidence.entry(node).or_default();
        let events=if overlay{data.overlay.get_or_insert_with(Default::default)}else{&mut data.events};
        let mut merged=events.regions.iter().cloned().map(|p|(key(&p),p)).collect::<BTreeMap<_,_>>();replace(&mut merged,&proofs);
        events.regions=merged.into_values().collect();Ok(())
    }

    fn proofs_from_tail(&self,doc:&Document,node:NodeId,raw:&[RawRecord],start:usize,scope:&[(usize,usize)])->Result<CompletionEvidence,String>{
        let mut active=None;let mut ended=HashSet::new();
        for(index,row)in raw.iter().enumerate(){
            if crate::region_state::is_marker(row){let(s,i)=crate::region_state::marker_identity(row)?;let k=(s.side(),i);active=Some(k);ended.remove(&k);}
            else if row==&crate::region_state::END_RECORD{
                let k=active.take().ok_or("完成证据出现区域外END")?;
                if index>=start&&(scope.is_empty()||scope.contains(&k)){ended.insert(k);}
            }else if row==&crate::region_state::INTERRUPT_RECORD{active=None;}
        }
        let members=memberships(self,doc,node)?;
        let mut regions=ended.into_iter().map(|(side,index)|RegionCompletionProof{side,index,primitive_ids:members.get(&(side,index)).cloned().unwrap_or_default()}).collect::<Vec<_>>();
        regions.sort_by_key(key);Ok(CompletionEvidence{regions})
    }

    pub fn restart_normal_history(&mut self,doc:&Document,node:NodeId,raw:&[RawRecord],ids:&[u64],floor:usize,reviews:&[(usize,usize)])->Result<(),String>{
        if !matches!(floor,1|4|8){return Err("普通作画恢复阶段无效".into());}checked_keys(reviews)?;
        let parent=unchanged_boundary(self,doc,node)?;
        if self.mode_evidence.contains_key(&node){return Err("普通作画恢复边界已经登记".into());}
        self.validate_context_with_model_ids(doc,node,raw,ids)?;
        if raw_phase(raw,false)?>floor{return Err("普通作画恢复上下文超出其阶段".into());}
        let baseline=filtered(self.completion_evidence(doc,parent)?,&memberships(self,doc,node)?);
        let base=if self.entries.contains_key(&parent){HistoryBase::Node(parent)}else{HistoryBase::Canvas(parent)};
        let previous=self.resolve(doc,base)?;let keep=common_prefix(&previous,raw);
        let previous_ids=self.resolve_model_ids(doc,base)?;let id_keep=previous_ids.iter().zip(ids).take_while(|(a,b)|a==b).count();
        self.entries.insert(node,HistoryDelta{base,keep,tail:raw[keep..].to_vec()});
        self.model_ids.insert(node,ModelIdDelta{base,keep:id_keep,tail:ids[id_keep..].to_vec(),phase_floor:floor});
        self.control_overlays.remove(&node);self.overlay_model_ids.remove(&node);self.overlay_phase_floors.remove(&node);self.overlay_manual_reviews.remove(&node);
        self.set_manual_reviews(node,reviews.to_vec());
        self.mode_evidence.insert(node,ModeEvidenceNode{baseline:Some(baseline),normal_restart:true,..Default::default()});Ok(())
    }

    /// A newly accepted reference on a blank canvas starts an independent
    /// drawing frontier. Old zero-shape region completions are not its proof.
    pub fn restart_empty_reference_origin(&mut self,doc:&Document,node:NodeId)->Result<(),String>{
        if !doc.materialize(node)?.is_empty(){return Err("参考导入起点必须为空白画布".into());}
        self.restart_normal_history(doc,node,&[crate::region_state::BOS_RECORD],&[],1,&[])?;
        let data=self.mode_evidence.get_mut(&node).ok_or("参考起点缺少重置边界")?;
        data.baseline=Some(CompletionEvidence::default());data.events=CompletionEvidence::default();data.overlay=None;
        Ok(())
    }

    pub(super) fn freeze_user_completion(&mut self,doc:&Document,parent:NodeId,node:NodeId)->Result<(),String>{
        let evidence=filtered(self.completion_evidence(doc,parent)?,&memberships(self,doc,node)?);
        self.mode_evidence.entry(node).or_default().baseline=Some(evidence);Ok(())
    }
    pub(super) fn mode_refs(&self)->Vec<ModeEvidenceRef<'_>>{
        let mut entries=self.mode_evidence.iter().map(|(&node,value)|ModeEvidenceRef{node,baseline:value.baseline.as_ref(),events:&value.events,
            overlay:value.overlay.as_ref(),redraw_scope:value.redraw_scope.as_deref(),normal_restart:value.normal_restart}).collect::<Vec<_>>();
        entries.sort_by_key(|entry|entry.node);entries
    }
    pub(super) fn load_mode_data(&mut self,entries:Vec<ModeEvidenceData>)->Result<(),String>{
        for entry in entries{let node=entry.node;let value=ModeEvidenceNode{baseline:entry.baseline.map(EvidenceData::into_value),events:entry.events.into_value(),
            overlay:entry.overlay.map(EvidenceData::into_value),redraw_scope:entry.redraw_scope.map(|keys|keys.0),normal_restart:entry.normal_restart};
            if self.mode_evidence.insert(node,value).is_some(){return Err("项目含重复模式来源状态".into());}}
        Ok(())
    }
    pub(super) fn validate_mode_data(&self,doc:&Document,budget:&mut project_data::Budget,work:&mut project_data::Work)->Result<(),String>{
        budget.charge(project_data::RAW_ENTRIES,self.mode_evidence.len())?;
        for(&node,value)in &self.mode_evidence{
            if node>=doc.nodes.len(){return Err("模式来源引用不存在状态".into());}work.charge(1)?;
            if value.normal_restart||value.redraw_scope.is_some(){unchanged_boundary(self,doc,node)?;if value.baseline.is_none(){return Err("模式切换缺少完成证据快照".into());}}
            if value.normal_restart&&(value.redraw_scope.is_some()||!self.model_ids.contains_key(&node)){return Err("普通作画恢复边界元数据无效".into());}
            if let Some(scope)=&value.redraw_scope{
                checked_scope(scope)?;
                if scope.contains(&(1,0)) {
                    let raw=self.resolve(doc,HistoryBase::Node(node))?;let ids=self.resolve_model_ids(doc,HistoryBase::Node(node))?;
                    validate_global_redraw_prefix(self,doc,node,&raw,&ids)?;
                    if !value.baseline.as_ref().is_some_and(|baseline|baseline.regions.is_empty()){
                        return Err("全图续画边界不能继承旧网格完成证据".into());
                    }
                }
            }
            let members=memberships(self,doc,node)?;
            for evidence in value.baseline.iter().chain(std::iter::once(&value.events)).chain(value.overlay.iter()){
                if evidence.regions.len()>80{return Err("完成证据区域过多".into());}let mut keys=HashSet::new();
                for proof in &evidence.regions{
                    checked_keys(&[key(proof)])?;if !keys.insert(key(proof)){return Err("完成证据区域重复".into());}
                    budget.charge(project_data::RECORDS,proof.primitive_ids.len())?;work.charge(proof.primitive_ids.len().saturating_add(1))?;
                    if proof.primitive_ids.iter().any(|id|*id==0)||proof.primitive_ids.windows(2).any(|pair|pair[0]>=pair[1])
                        ||members.get(&key(proof)).map(Vec::as_slice).unwrap_or(&[])!=proof.primitive_ids.as_slice(){return Err("完成证据与当前区域图元成员不一致".into());}
                }
            }
        }Ok(())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn shape(stage:StageTier,cell:usize,color:[u8;3])->Primitive{
        let side=stage.side();let step=512./side as f32;
        let mut p=Primitive::new(ShapeKind::Rectangle,(cell%side)as f32*step+24.,(cell/side)as f32*step+24.,12.,10.,color);
        p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:cell as u64});p
    }
    fn row(p:&Primitive)->RawRecord{p.to_tokens().unwrap().map(u32::from)}
    fn coarse_fixture()->(Document,RawHistoryStore,Vec<RawRecord>,Vec<u64>){
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let bg=doc.commit_generated(Primitive::background([240;3])).unwrap();let mut raw=vec![BOS_RECORD,row(&doc.primitives[0])];let mut ids=vec![doc.primitives[0].id];
        store.attach_with_model_ids(&doc,bg,&raw,&ids).unwrap();
        let node=doc.commit_generated(shape(StageTier::Coarse,0,[80,100,120])).unwrap();raw.extend([crate::region_state::marker_record(Stage::Coarse,0).unwrap(),row(doc.primitives.last().unwrap())]);ids.push(doc.primitives.last().unwrap().id);
        store.attach_with_model_ids(&doc,node,&raw,&ids).unwrap();(doc,store,raw,ids)
    }
    fn edit(doc:&mut Document,store:&mut RawHistoryStore,canvas:Vec<Primitive>)->NodeId{
        let parent=doc.current;let canvas=doc.preview_canvas(canvas).unwrap();let plan=store.prepare_edit(doc,parent,canvas,None,false).unwrap();
        let node=doc.commit_canvas("manual edit",OperationSource::User,plan.canvas.clone()).unwrap();store.apply_prepared_edit(doc,parent,node,plan).unwrap();node
    }
    #[test]
    fn typed_readonly_ends_are_not_proof_zero_model_end_is_and_late_child_keeps_frozen_snapshot(){
        let(mut doc,mut store,mut raw,ids)=coarse_fixture();let old=doc.current;
        let origin=doc.commit_control_canvas("准备局部重绘",doc.primitives.clone()).unwrap();raw.push(crate::region_state::END_RECORD);
        store.set_control_overlay_with_ids(&doc,origin,&raw,&ids).unwrap();store.begin_redraw_origin(&doc,origin,&[(4,1),(4,2)],&raw,&ids).unwrap();
        assert!(store.completion_evidence(&doc,origin).unwrap().regions.is_empty(),"readonly coarse END cannot prove a model decision");
        let frozen=store.completion_evidence(&doc,origin).unwrap();let append_from=raw.len();
        raw.extend([crate::region_state::marker_record(Stage::Coarse,1).unwrap(),crate::region_state::END_RECORD]);
        store.set_control_overlay_with_ids(&doc,origin,&raw,&ids).unwrap();store.record_completed_regions(&doc,origin,&raw,&ids,append_from,&[(4,1),(4,2)],true).unwrap();
        assert_eq!(store.completion_evidence(&doc,origin).unwrap().regions,vec![RegionCompletionProof{side:4,index:1,primitive_ids:Vec::new()}]);
        let child=doc.commit_generated(shape(StageTier::Coarse,2,[100,120,140])).unwrap();
        raw.extend([crate::region_state::marker_record(Stage::Coarse,2).unwrap(),row(doc.primitives.last().unwrap())]);let mut child_ids=ids.clone();child_ids.push(doc.primitives.last().unwrap().id);
        store.attach_with_model_ids(&doc,child,&raw,&child_ids).unwrap();store.set_completion_baseline(&doc,child,&frozen).unwrap();
        assert!(store.completion_evidence(&doc,child).unwrap().regions.is_empty(),"late origin overlay must not become child creation evidence");
        assert!(store.requires_normal_restart(&doc,child).unwrap());assert!(!store.requires_normal_restart(&doc,old).unwrap());
        let path=std::env::temp_dir().join(format!("painter-mode-overlay-{}.pproj",std::process::id()));
        crate::project_file::save_project(&path,&doc,&store).unwrap();let(loaded,history)=crate::project_file::load_project(&path).unwrap();
        assert_eq!(history.completion_evidence(&loaded,origin).unwrap().regions,vec![RegionCompletionProof{side:4,index:1,primitive_ids:Vec::new()}]);
        assert!(history.completion_evidence(&loaded,child).unwrap().regions.is_empty());std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn reopened_yellow_keeps_geometry_but_old_completion_cannot_return_without_a_fresh_end(){
        let(mut doc,mut store,mut raw,ids)=coarse_fixture();let old=doc.current;let original_canvas=doc.primitives.clone();
        let start=raw.len();raw.extend([crate::region_state::END_RECORD,
            crate::region_state::marker_record(Stage::Coarse,1).unwrap(),crate::region_state::END_RECORD]);
        store.set_control_overlay_with_ids(&doc,old,&raw,&ids).unwrap();
        store.record_completed_regions(&doc,old,&raw,&ids,start,&[],true).unwrap();
        let old_proof=store.completion_evidence(&doc,old).unwrap();assert_eq!(old_proof.regions.len(),2);
        let original_raw=store.context_for(&doc,old).unwrap();
        let origin=doc.commit_control_canvas("准备局部重绘",doc.primitives.clone()).unwrap();
        let reopened=vec![BOS_RECORD,row(&doc.primitives[0]),
            crate::region_state::marker_record(Stage::Coarse,1).unwrap(),crate::region_state::END_RECORD,
            crate::region_state::marker_record(Stage::Coarse,0).unwrap(),row(&doc.primitives[1])];
        store.set_control_overlay_with_ids(&doc,origin,&reopened,&ids).unwrap();
        store.begin_redraw_origin(&doc,origin,&[(4,0)],&reopened,&ids).unwrap();
        let frozen=store.completion_evidence(&doc,origin).unwrap();
        assert_eq!(frozen.regions,vec![RegionCompletionProof{side:4,index:1,primitive_ids:Vec::new()}]);
        assert_eq!(doc.primitives,original_canvas,"reopening must not delete or replace user geometry");
        assert_eq!(store.completion_evidence(&doc,old).unwrap(),old_proof,"the older branch stays complete");
        let path=std::env::temp_dir().join(format!("painter-reopened-yellow-proof-{}-{origin}.pproj",std::process::id()));
        crate::project_file::save_project(&path,&doc,&store).unwrap();let(loaded,history)=crate::project_file::load_project(&path).unwrap();
        assert_eq!(history.completion_evidence(&loaded,origin).unwrap(),frozen,"save/load must not resurrect removed completion");
        let child=doc.commit_generated(shape(StageTier::Coarse,0,[100,120,140])).unwrap();
        let mut child_raw=reopened.clone();child_raw.push(row(doc.primitives.last().unwrap()));
        let mut child_ids=ids.clone();child_ids.push(doc.primitives.last().unwrap().id);
        store.attach_with_model_ids(&doc,child,&child_raw,&child_ids).unwrap();store.set_completion_baseline(&doc,child,&frozen).unwrap();
        let mut origin_end=reopened.clone();origin_end.push(crate::region_state::END_RECORD);
        store.set_control_overlay_with_ids(&doc,origin,&origin_end,&ids).unwrap();
        store.record_completed_regions(&doc,origin,&origin_end,&ids,reopened.len(),&[(4,0)],true).unwrap();
        assert!(store.completion_evidence(&doc,origin).unwrap().regions.iter().any(|p|key(p)==(4,0)));
        assert_eq!(store.completion_evidence(&doc,child).unwrap(),frozen,"an origin's later END must not leak into an existing child");
        let append_from=child_raw.len();child_raw.push(crate::region_state::END_RECORD);
        store.set_control_overlay_with_ids(&doc,child,&child_raw,&child_ids).unwrap();
        store.record_completed_regions(&doc,child,&child_raw,&child_ids,append_from,&[(4,0)],true).unwrap();
        let completed=store.completion_evidence(&doc,child).unwrap();
        assert_eq!(completed.regions.iter().find(|p|key(p)==(4,0)).unwrap().primitive_ids,child_ids[1..]);
        assert_eq!(doc.materialize(old).unwrap(),original_canvas);assert_eq!(store.context_for(&doc,old).unwrap(),original_raw);
        crate::project_file::save_project(&path,&doc,&store).unwrap();let(loaded,history)=crate::project_file::load_project(&path).unwrap();
        assert_eq!(history.completion_evidence(&loaded,child).unwrap(),completed);std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn proof_ignores_color_and_within_cell_geometry_but_delete_and_membership_changes_invalidate(){
        let(mut doc,mut store,mut raw,ids)=coarse_fixture();let node=doc.current;let start=raw.len();raw.push(crate::region_state::END_RECORD);
        store.set_control_overlay_with_ids(&doc,node,&raw,&ids).unwrap();store.record_completed_regions(&doc,node,&raw,&ids,start,&[],true).unwrap();
        let proof=store.completion_evidence(&doc,node).unwrap();assert_eq!(proof.regions[0].primitive_ids,vec![ids[1]]);
        let mut canvas=doc.primitives.clone();canvas[1].color=[20,40,60];canvas[1].x+=3.;let same=edit(&mut doc,&mut store,canvas);
        assert_eq!(store.completion_evidence(&doc,same).unwrap(),proof);
        let mut canvas=doc.primitives.clone();canvas.remove(1);let removed=edit(&mut doc,&mut store,canvas);
        assert!(store.completion_evidence(&doc,removed).unwrap().regions.is_empty());
        doc.jump(same).unwrap();let mut canvas=doc.primitives.clone();canvas.push(shape(StageTier::Coarse,0,[80;3]));let added=edit(&mut doc,&mut store,canvas);
        assert!(store.completion_evidence(&doc,added).unwrap().regions.is_empty());
        doc.jump(same).unwrap();let mut canvas=doc.primitives.clone();canvas[1].x=150.;let migrated=edit(&mut doc,&mut store,canvas);
        assert!(store.completion_evidence(&doc,migrated).unwrap().regions.is_empty());
    }
    #[test]
    fn normal_restart_rewinds_only_new_branch_and_survives_project_roundtrip_and_delete_remap(){
        let(mut doc,mut store,mut raw,mut ids)=coarse_fixture();raw.push(crate::region_state::END_RECORD);
        let fine=doc.commit_generated(shape(StageTier::Fine,0,[90;3])).unwrap();raw.extend([crate::region_state::marker_record(Stage::Fine,0).unwrap(),row(doc.primitives.last().unwrap())]);ids.push(doc.primitives.last().unwrap().id);
        store.attach_with_model_ids(&doc,fine,&raw,&ids).unwrap();let original=store.context_for(&doc,fine).unwrap();
        let mut sibling=doc.primitives.clone();sibling[2].color=[100;3];let unrelated=edit(&mut doc,&mut store,sibling);doc.jump(fine).unwrap();
        let mut canvas=doc.primitives.clone();canvas.pop();let cleared=edit(&mut doc,&mut store,canvas);assert_eq!(store.stage_floor(&doc,cleared).unwrap(),Some(8));
        let restart=doc.commit_control_canvas("恢复普通作画",doc.primitives.clone()).unwrap();
        let restart_raw=vec![BOS_RECORD,row(&doc.primitives[0]),crate::region_state::marker_record(Stage::Coarse,0).unwrap(),row(&doc.primitives[1])];
        let restart_ids=doc.primitives.iter().map(|p|p.id).collect::<Vec<_>>();
        store.restart_normal_history(&doc,restart,&restart_raw,&restart_ids,4,&[(4,0)]).unwrap();
        assert_eq!(store.editing_stage(&doc,restart).unwrap(),Stage::Coarse);assert_eq!(store.stage_floor(&doc,fine).unwrap(),Some(8));
        assert_eq!(store.context_for(&doc,fine).unwrap(),original);assert!(!store.requires_normal_restart(&doc,restart).unwrap());
        let path=std::env::temp_dir().join(format!("painter-mode-proof-{}-{}.pproj",std::process::id(),restart));
        crate::project_file::save_project(&path,&doc,&store).unwrap();let(mut loaded,mut history)=crate::project_file::load_project(&path).unwrap();
        assert_eq!(history.stage_floor(&loaded,restart).unwrap(),Some(4));
        let deletion=loaded.plan_delete_subtrees(&[unrelated]).unwrap();let changed=history.apply_delete_subtrees(&mut loaded,deletion).unwrap();let remapped=changed.old_to_new[restart].unwrap();
        assert_eq!(history.editing_stage(&loaded,remapped).unwrap(),Stage::Coarse);
        crate::project_file::save_project(&path,&loaded,&history).unwrap();
        history.mode_evidence.get_mut(&remapped).unwrap().normal_restart=false;
        assert!(crate::project_file::save_project(&path,&loaded,&history).is_err(),"ordinary nodes must still reject phase regression");
        std::fs::remove_file(path).unwrap();
        let bad=doc.commit_canvas("not a control boundary",OperationSource::User,vec![Primitive::background([20;3])]).unwrap();
        assert!(store.restart_normal_history(&doc,bad,&[BOS_RECORD,row(&doc.primitives[0])],&[doc.primitives[0].id],1,&[]).is_err());
    }
    #[test]
    fn global_redraw_restarts_only_the_new_clean_branch_and_revokes_even_empty_old_region_proofs(){
        let(mut doc,mut store,mut raw,mut ids)=coarse_fixture();
        raw.extend([crate::region_state::END_RECORD,crate::region_state::marker_record(Stage::Coarse,1).unwrap(),crate::region_state::END_RECORD]);
        let fine=doc.commit_generated(shape(StageTier::Fine,0,[100;3])).unwrap();
        raw.extend([crate::region_state::marker_record(Stage::Fine,0).unwrap(),row(doc.primitives.last().unwrap())]);ids.push(doc.primitives.last().unwrap().id);
        store.attach_with_model_ids(&doc,fine,&raw,&ids).unwrap();let start=raw.len();raw.push(crate::region_state::END_RECORD);
        store.set_control_overlay_with_ids(&doc,fine,&raw,&ids).unwrap();store.record_completed_regions(&doc,fine,&raw,&ids,start,&[],true).unwrap();
        let old=store.context_for(&doc,fine).unwrap();let old_canvas=doc.primitives.clone();
        let global=vec![doc.primitives[0].clone()];edit(&mut doc,&mut store,global);
        let origin=doc.commit_control_canvas("准备全图续画",doc.primitives.clone()).unwrap();
        let prefix=vec![BOS_RECORD,row(&doc.primitives[0])];let prefix_ids=vec![doc.primitives[0].id];
        store.set_control_overlay_with_ids(&doc,origin,&prefix,&prefix_ids).unwrap();
        assert_eq!(store.stage_floor(&doc,origin).unwrap(),Some(8));
        let scope=std::iter::once((1,0)).chain((0..16).map(|i|(4,i))).chain((0..64).map(|i|(8,i))).collect::<Vec<_>>();
        store.begin_redraw_origin(&doc,origin,&scope,&prefix,&prefix_ids).unwrap();
        assert_eq!(store.stage_floor(&doc,origin).unwrap(),None);assert_eq!(store.context_for(&doc,origin).unwrap(),Some(prefix));
        assert!(store.completion_evidence(&doc,origin).unwrap().regions.is_empty());assert!(store.requires_normal_restart(&doc,origin).unwrap());
        assert_eq!(store.context_for(&doc,fine).unwrap(),old);assert_eq!(doc.materialize(fine).unwrap(),old_canvas);
        let path=std::env::temp_dir().join(format!("painter-global-redraw-origin-{}-{origin}.pproj",std::process::id()));
        crate::project_file::save_project(&path,&doc,&store).unwrap();let(loaded,history)=crate::project_file::load_project(&path).unwrap();
        assert_eq!(history.stage_floor(&loaded,origin).unwrap(),None);assert!(history.completion_evidence(&loaded,origin).unwrap().regions.is_empty());
        assert_eq!(history.mode_evidence[&origin].redraw_scope.as_ref().unwrap().len(),81);
        store.mode_evidence.get_mut(&origin).unwrap().redraw_scope=None;
        assert!(crate::project_file::save_project(&path,&doc,&store).is_err(),"ordinary nodes must not inherit the Global restart exception");
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn global_redraw_validation_does_not_relax_regional_completion_keys_or_accept_uncleared_geometry(){
        assert!(checked_scope(&[(1,0)]).is_ok());assert!(checked_keys(&[(1,0)]).is_err());
        for keys in [vec![(1,1)],vec![(1,0),(1,0)],vec![(2,0)]]{assert!(checked_scope(&keys).is_err());}
        let(mut doc,mut store,_raw,_ids)=coarse_fixture();let origin=doc.commit_control_canvas("invalid global",doc.primitives.clone()).unwrap();
        let raw=vec![BOS_RECORD,row(&doc.primitives[0])];let ids=vec![doc.primitives[0].id];
        assert!(store.begin_redraw_origin(&doc,origin,&[(1,0)],&raw,&ids).is_err());
        assert!(!store.mode_evidence.contains_key(&origin));
    }
}
