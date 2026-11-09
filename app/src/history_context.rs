//! Model-only region/control history attached to visible creative states.
//!
//! Deltas refer either to a prior model state or lazily to the visible canvas
//! at the parent state. Manual edits reconcile visible records by stable ID and
//! retain valid control history, rather than blindly inheriting stale tokens.
//! Ordinary nodes store only deltas. Rare control-only results use a separate
//! exact-node overlay, without changing bases already shared by older branches.

use crate::document::{Document, HistoryDeletion, HistoryDeletionPlan, NodeId, OperationSource, Patch, Primitive, PrimitiveStageAssignment, ShapeKind,StageTier};
use crate::region_state::{DrawingRegion, Stage};
use std::collections::{HashMap, HashSet};
use crate::document::project_data;
use serde::{Deserialize, Serialize};
#[path="stage_edit.rs"]
pub(crate) mod stage_edit;
#[path="stage_project_validation.rs"]
mod stage_project_validation;
#[path="past_color_repair.rs"]
mod past_color_repair;
#[path="mode_evidence.rs"]
mod mode_evidence;
#[cfg(test)]
#[path="stage_history_tests.rs"]
mod stage_history_tests;
pub use stage_edit::{StageAssignmentRequest,StageEditBlock,StageEditPlan};
use std::collections::BTreeMap;

pub type RawRecord = [u32; 9];
const PAD: u32 = 7054;
const REGION_MARKER: u32 = 7057;
const TRUE_END: u32 = 7066;
const INTERRUPT: u32 = 7067;
const BOS: u32 = 7068;
const BOS_RECORD: RawRecord = [BOS, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag="kind",content="node",rename_all="snake_case",deny_unknown_fields)]
enum HistoryBase {
    Node(NodeId),
    Canvas(NodeId),
    Empty,
}

#[derive(Clone, Debug)]
struct HistoryDelta {
    base: HistoryBase,
    keep: usize,
    tail: Vec<RawRecord>,
}

#[derive(Clone,Debug)]struct ModelIdDelta{base:HistoryBase,keep:usize,tail:Vec<u64>,phase_floor:usize}
#[derive(Clone,Debug)]pub struct EffectiveHistory{pub records:Vec<RawRecord>,pub primitive_ids:Vec<u64>}

#[derive(Default)]
pub struct RawHistoryStore {
    entries: HashMap<NodeId, HistoryDelta>,
    // A control-only request has no new visible/tree node of its own. Keep
    // its public context separately so immutable bases of older branches do
    // not acquire another lane's end/region-selection records.
    control_overlays: HashMap<NodeId, Vec<RawRecord>>,
    /// Immutable creation-time snapshots. Effective pending regions are filtered
    /// by the queried node's own raw ENDs; parent overlays never clear a late
    /// sibling's inherited creation snapshot. Missing always means empty.
    manual_reviews: HashMap<NodeId, Vec<(usize,usize)>>,
    model_ids: HashMap<NodeId,ModelIdDelta>,
    overlay_model_ids: HashMap<NodeId,Vec<u64>>,
    overlay_phase_floors:HashMap<NodeId,usize>,
    overlay_manual_reviews:HashMap<NodeId,Vec<(usize,usize)>>,
    mode_evidence:HashMap<NodeId,mode_evidence::ModeEvidenceNode>,
}

#[derive(Serialize)]pub(crate)struct RawHistoryProjectRef<'a>{entries:Vec<RawEntryRef<'a>>,control_overlays:Vec<OverlayRef<'a>>,manual_reviews:Vec<ManualReviewRef<'a>>,model_ids:Vec<ModelIdRef<'a>>,mode_evidence:Vec<mode_evidence::ModeEvidenceRef<'a>>}
#[derive(Serialize)]struct ModelIdRef<'a>{node:NodeId,base:HistoryBase,keep:usize,tail:&'a[u64],phase_floor:usize,
    #[serde(skip_serializing_if="Option::is_none")]overlay:Option<&'a[u64]>,
    #[serde(skip_serializing_if="Option::is_none")]overlay_phase_floor:Option<usize>,
    #[serde(skip_serializing_if="Option::is_none")]overlay_manual_reviews:Option<&'a[(usize,usize)]>}
#[derive(Serialize)]struct RawEntryRef<'a>{node:NodeId,base:HistoryBase,keep:usize,tail:&'a[RawRecord]}
#[derive(Serialize)]struct OverlayRef<'a>{node:NodeId,records:&'a[RawRecord]}
#[derive(Serialize)]struct ManualReviewRef<'a>{node:NodeId,regions:&'a[(usize,usize)]}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]pub(crate)struct RawHistoryProjectData{
    entries:project_data::LimitedVec<project_data::Counted<RawEntryData,{project_data::RAW_ENTRIES}>,{project_data::MAX_NODES}>,
    control_overlays:project_data::LimitedVec<project_data::Counted<OverlayData,{project_data::OVERLAYS}>,{project_data::MAX_NODES}>,
    #[serde(default)]manual_reviews:Option<project_data::LimitedVec<project_data::Counted<ManualReviewData,{project_data::MANUAL_REVIEWS}>,{project_data::MAX_NODES}>>,
    #[serde(default)]model_ids:Option<project_data::LimitedVec<project_data::Counted<ModelIdData,{project_data::MODEL_IDS}>,{project_data::MAX_NODES}>>,
    #[serde(default)]mode_evidence:Option<project_data::LimitedVec<project_data::Counted<mode_evidence::ModeEvidenceData,{project_data::RAW_ENTRIES}>,{project_data::MAX_NODES}>>,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct ModelIdData{
    node:NodeId,base:HistoryBase,keep:usize,phase_floor:usize,
    tail:project_data::LimitedVec<project_data::Counted<u64,{project_data::RECORDS}>,{project_data::MAX_CANVAS}>,
    #[serde(default)]overlay:Option<project_data::LimitedVec<project_data::Counted<u64,{project_data::RECORDS}>,{project_data::MAX_CANVAS}>>,
    #[serde(default)]overlay_phase_floor:Option<usize>,
    #[serde(default)]overlay_manual_reviews:Option<project_data::LimitedVec<project_data::Counted<(usize,usize),{project_data::RECORDS}>,80>>,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct RawEntryData{
    node:NodeId,base:HistoryBase,keep:usize,
    tail:project_data::LimitedVec<project_data::Counted<RawRecord,{project_data::RECORDS}>,{project_data::MAX_RECORDS}>,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct OverlayData{
    node:NodeId,records:project_data::LimitedVec<project_data::Counted<RawRecord,{project_data::RECORDS}>,{project_data::MAX_RECORDS}>,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct ManualReviewData{
    node:NodeId,regions:project_data::LimitedVec<project_data::Counted<(usize,usize),{project_data::RECORDS}>,80>,
}
impl RawHistoryProjectData{pub(crate)fn has_manual_reviews(&self)->bool{self.manual_reviews.is_some()}
    pub(crate)fn has_model_ids(&self)->bool{self.model_ids.is_some()}
    pub(crate)fn has_mode_evidence(&self)->bool{self.mode_evidence.is_some()}}

impl RawHistoryStore {
    fn resolve_model_ids(&self,doc:&Document,mut base:HistoryBase)->Result<Vec<u64>,String>{
        let mut path=Vec::new();let mut seen=HashSet::new();
        let mut ids=loop{match base{
            HistoryBase::Node(node)=>{if !seen.insert(node){return Err("模型图元ID顺序循环引用".into());}
                if let Some(delta)=self.model_ids.get(&node){path.push(delta);base=delta.base;}
                else{break doc.materialize(node)?.into_iter().map(|p|p.id).collect::<Vec<_>>();}},
            HistoryBase::Canvas(node)=>break doc.materialize(node)?.into_iter().map(|p|p.id).collect::<Vec<_>>(),
            HistoryBase::Empty=>break Vec::new(),
        }};
        for delta in path.into_iter().rev(){if delta.keep>ids.len(){return Err("模型ID顺序增量超出基底".into());}
            ids.truncate(delta.keep);ids.extend_from_slice(&delta.tail);}
        Ok(ids)
    }
    pub fn model_ids_for_node(&self,doc:&Document,node:NodeId)->Result<Vec<u64>,String>{
        if node>=doc.nodes.len(){return Err("模型输入状态不存在".into());}
        if let Some(ids)=self.overlay_model_ids.get(&node){return Ok(ids.clone());}
        self.resolve_model_ids(doc,HistoryBase::Node(node))
    }
    fn put_model_ids(&mut self,doc:&Document,node:NodeId,base:HistoryBase,ids:&[u64],phase_floor:usize)->Result<(),String>{
        let previous=self.resolve_model_ids(doc,base)?;let keep=previous.iter().zip(ids).take_while(|(a,b)|a==b).count();
        self.model_ids.insert(node,ModelIdDelta{base,keep,tail:ids[keep..].to_vec(),phase_floor});Ok(())
    }
    fn verify_node_raw(&self,doc:&Document,node:NodeId,raw:&[RawRecord],public:bool)->Result<(),String>{
        if self.model_ids.contains_key(&node)||self.overlay_model_ids.contains_key(&node){
            let ids=if public{self.model_ids_for_node(doc,node)?}else{self.resolve_model_ids(doc,HistoryBase::Node(node))?};
            verify_projection_ids(&doc.materialize(node)?,raw,&ids)
        }else{verify_visible(doc,node,raw)}
    }
    pub fn stage_map(&self,doc:&Document,node:NodeId)->Result<BTreeMap<u64,Stage>,String>{
        let canvas=doc.materialize(node)?;let mut legacy=BTreeMap::new();
        if let Some(raw)=self.context_for(doc,node)?{
            let ids=self.model_ids_for_node(doc,node)?;let mut cursor=0;let mut active=Stage::Global;
            for row in &raw{if crate::region_state::is_marker(row){active=crate::region_state::marker_identity(row)?.0;}
                else if is_visible(row){let id=*ids.get(cursor).ok_or("阶段映射缺少模型图元ID")?;legacy.insert(id,active);cursor+=1;}}
            if cursor!=ids.len(){return Err("阶段映射与模型图元ID数量不一致".into());}
        }
        for p in &canvas{let phase=if p.shape==ShapeKind::Background{Stage::Global}else{p.stage_assignment.map(|a|stage_edit::stage(a.stage))
            .or_else(||legacy.get(&p.id).copied()).unwrap_or(Stage::Global)};legacy.insert(p.id,phase);}
        Ok(legacy)
    }
    pub fn editing_stage(&self,doc:&Document,node:NodeId)->Result<Stage,String>{
        Ok(match self.stage_floor(doc,node)?{Some(4)=>Stage::Coarse,Some(8)=>Stage::Fine,_=>Stage::Global})
    }
    fn immutable_phase_floor(&self,doc:&Document,node:NodeId)->Result<usize,String>{
        if node>=doc.nodes.len(){return Err("阶段状态不存在".into());}
        let mut cursor=Some(node);let mut floor=1;
        while let Some(id)=cursor{
            if let Some(delta)=self.model_ids.get(&id){floor=floor.max(delta.phase_floor);break;}
            if self.entries.contains_key(&id){let raw=self.resolve(doc,HistoryBase::Node(id))?;
                let explicit_model=doc.nodes[id].source==OperationSource::Model
                    ||self.entries.get(&id).is_some_and(|d|matches!(d.base,HistoryBase::Canvas(base)if base==id));
                floor=floor.max(raw_phase(&raw,!explicit_model)?);
            }
            cursor=doc.nodes[id].parent;
        }
        if !matches!(floor,1|4|8){return Err("阶段游标无效".into());}Ok(floor)
    }
    pub fn stage_floor(&self,doc:&Document,node:NodeId)->Result<Option<usize>,String>{
        let mut floor=self.immutable_phase_floor(doc,node)?;
        if let Some(&overlay)=self.overlay_phase_floors.get(&node){floor=floor.max(overlay);}
        else if let Some(raw)=self.control_overlays.get(&node){floor=floor.max(raw_phase(raw,false)?);}
        Ok((floor>1).then_some(floor))
    }
    pub fn pending_stage_context(&self,doc:&Document,node:NodeId)->Result<Vec<crate::region_state::PendingStageRecord>,String>{
        let canvas=doc.materialize(node)?;let legacy=self.stage_map(doc,node)?;
        let floor=self.stage_floor(doc,node)?.unwrap_or(1);let guard=if node==doc.current{doc.settings.pause_at_stage_end}else{doc.nodes[node].settings.pause_at_stage_end};
        let view=stage_edit::view(&canvas,&legacy,guard)?;let by_id=canvas.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();
        view.admitted_ids.iter().filter(|id|view.assignments[id].stage.side()>floor).map(|id|{let a=view.assignments[id];
            Ok(crate::region_state::PendingStageRecord{side:a.stage.side(),index:a.cell,tokens:by_id[id].to_tokens()?.map(u32::from)})}).collect()
    }
    pub fn pending_stage_ids(&self,doc:&Document,node:NodeId)->Result<Vec<(usize,usize,u64)>,String>{
        let canvas=doc.materialize(node)?;let legacy=self.stage_map(doc,node)?;let floor=self.stage_floor(doc,node)?.unwrap_or(1);
        let guard=if node==doc.current{doc.settings.pause_at_stage_end}else{doc.nodes[node].settings.pause_at_stage_end};
        let view=stage_edit::view(&canvas,&legacy,guard)?;
        Ok(view.admitted_ids.iter().filter_map(|id|{let a=view.assignments[id];(a.stage.side()>floor).then_some((a.stage.side(),a.cell,*id))}).collect())
    }
    pub fn effective_history(&self,doc:&Document,node:NodeId)->Result<EffectiveHistory,String>{
        let canvas=doc.materialize(node)?;let legacy=self.stage_map(doc,node)?;
        let guard=if node==doc.current{doc.settings.pause_at_stage_end}else{doc.nodes[node].settings.pause_at_stage_end};
        let view=stage_edit::view(&canvas,&legacy,guard)?;let floor=self.stage_floor(doc,node)?.unwrap_or(1);
        let raw=self.context_for(doc,node)?.unwrap_or_else(||vec![BOS_RECORD]);
        let ids=self.model_ids_for_node(doc,node)?;
        let clean=clean_empty_interrupts(&raw);
        let eligible=view.admitted_ids.iter().copied().filter(|id|view.assignments[id].stage.side()<=floor).collect::<HashSet<_>>();
        if let Some(origin)=past_color_repair::historical_color_origin(self,doc,node)?{
            let mut original=self.effective_history(doc,origin)?;
            if self.stage_floor(doc,origin)?==self.stage_floor(doc,node)?
                &&original.primitive_ids.len()==eligible.len()&&original.primitive_ids.iter().all(|id|eligible.contains(id)){
                let by_id=canvas.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();let mut index=0;
                for row in &mut original.records{if is_visible(row){
                    let id=*original.primitive_ids.get(index).ok_or("历史改色修复缺少图元ID")?;
                    *row=by_id.get(&id).ok_or("历史改色修复缺少图元")?.to_tokens()?.map(u32::from);index+=1;
                }}
                verify_projection_ids(&canvas,&original.records,&original.primitive_ids)?;
                return Ok(original);
            }
        }
        if raw_phase(&clean,false)?<=floor&&ids.len()==eligible.len()&&ids.iter().all(|id|eligible.contains(id))&&projection_assignments_match(&clean,&ids,&view){
            verify_projection_ids(&canvas,&clean,&ids)?;return Ok(EffectiveHistory{records:clean,primitive_ids:ids});
        }
        let pending=self.manual_review_regions(doc,node)?;
        canonical_history(&canvas,&view,floor,&raw,&pending)
    }
    pub fn prepare_edit(&self,doc:&Document,parent:NodeId,candidate:Vec<Primitive>,assignment:Option<StageAssignmentRequest>,global_confirmed:bool)
        ->Result<StageEditPlan,StageEditBlock>{
        let old=doc.materialize(parent)?;let legacy=self.stage_map(doc,parent)?;let guard=doc.settings.pause_at_stage_end;
        let prior_floor=self.stage_floor(doc,parent)?.unwrap_or(1);
        let canvas=stage_edit::prepare_canvas(&old,candidate,&legacy,self.editing_stage(doc,parent)?,assignment.as_ref(),guard,global_confirmed)?;
        let floor=if global_confirmed{prior_floor.max(4)}else{prior_floor};
        let view=stage_edit::view(&canvas,&legacy,guard)?;
        let old_view=stage_edit::view(&old,&legacy,doc.nodes[parent].settings.pause_at_stage_end)?;
        let old_by_id=old.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();let new_by_id=canvas.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();
        let mut pending=self.manual_review_regions(doc,parent)?;
        if matches!(floor,4|8){
            let old_current=|id:&u64|old_view.assignments.get(id).is_some_and(|a|a.stage.side()==floor)
                &&!old_view.deferred_ids.contains(id);
            let new_current=|id:&u64|view.assignments.get(id).is_some_and(|a|a.stage.side()==floor)
                &&!view.deferred_ids.contains(id);
            for p in &canvas{
                if p.shape!=ShapeKind::Background&&new_current(&p.id)
                    &&(!old_current(&p.id)||old_by_id.get(&p.id).is_none_or(|old|*old!=p)){
                    pending.push((floor,view.assignments[&p.id].cell));
                }
            }
            for p in &old{
                if p.shape!=ShapeKind::Background&&old_current(&p.id)
                    &&(!new_current(&p.id)||view.assignments.get(&p.id).is_some_and(|a|a.cell!=old_view.assignments[&p.id].cell)){
                    pending.push((floor,old_view.assignments[&p.id].cell));
                }
            }
        }
        pending.sort_unstable();pending.dedup();pending.retain(|(side,_)|*side==floor);
        let raw=self.effective_history(doc,parent)?.records;let history=canonical_history(&canvas,&view,floor,&raw,&pending)?;
        let old_deferred=old_view.deferred_ids.iter().copied().collect::<HashSet<_>>();let deferred=view.deferred_ids.iter().copied().collect::<HashSet<_>>();
        let new_deferred=deferred.difference(&old_deferred).count();let promoted=old_deferred.iter().filter(|id|new_by_id.contains_key(id)&&!deferred.contains(id)).count();
        let mut notices=Vec::new();if new_deferred>0{notices.push(format!("{new_deferred}个超额粗网格图元保留在画布，当前不进入粗网格模型上下文，将在细阶段作为输入使用"));}
        if promoted>0{notices.push(format!("已将{promoted}个排队图元自动纳入粗网格有效上下文"));}
        Ok(StageEditPlan{parent,canvas,raw:history.records,effective_ids:history.primitive_ids,deferred_ids:view.deferred_ids,notices,
            manual_review_regions:pending,phase_floor:floor})
    }
    pub fn apply_prepared_edit(&mut self,doc:&Document,parent:NodeId,node:NodeId,plan:StageEditPlan)->Result<DrawingRegion,String>{
        if plan.parent!=parent||doc.nodes.get(node).is_none_or(|n|n.parent!=Some(parent)||n.source!=OperationSource::User)
            ||doc.materialize(node)?!=plan.canvas{return Err("阶段编辑计划与新状态不一致".into());}
        verify_projection_ids(&plan.canvas,&plan.raw,&plan.effective_ids)?;let region=DrawingRegion::from_history(&plan.raw)?;
        let base=if self.entries.contains_key(&parent){HistoryBase::Node(parent)}else{HistoryBase::Canvas(parent)};
        let previous=self.resolve(doc,base)?;let keep=common_prefix(&previous,&plan.raw);
        let previous_ids=self.resolve_model_ids(doc,base)?;let id_keep=previous_ids.iter().zip(&plan.effective_ids).take_while(|(a,b)|a==b).count();
        let raw_delta=HistoryDelta{base,keep,tail:plan.raw[keep..].to_vec()};
        let id_delta=ModelIdDelta{base,keep:id_keep,tail:plan.effective_ids[id_keep..].to_vec(),phase_floor:plan.phase_floor};
        self.entries.insert(node,raw_delta);self.model_ids.insert(node,id_delta);self.set_manual_reviews(node,plan.manual_review_regions);
        self.freeze_user_completion(doc,parent,node)?;Ok(region)
    }
    pub fn deferred_fine_context(&self,doc:&Document,node:NodeId)->Result<Vec<RawRecord>,String>{
        let canvas=doc.materialize(node)?;let legacy=self.stage_map(doc,node)?;
        let guard=if node==doc.current{doc.settings.pause_at_stage_end}else{doc.nodes[node].settings.pause_at_stage_end};
        let view=stage_edit::view(&canvas,&legacy,guard)?;let deferred=view.deferred_ids.into_iter().collect::<HashSet<_>>();
        canvas.iter().filter(|p|deferred.contains(&p.id)).map(|p|p.to_tokens().map(|t|t.map(u32::from))).collect()
    }
    pub fn generated_assignment(&self,doc:&Document,parent:NodeId,raw:&[RawRecord])->Result<PrimitiveStageAssignment,String>{
        let canvas=doc.materialize(parent)?;let legacy=self.stage_map(doc,parent)?;let view=stage_edit::view(&canvas,&legacy,false)?;
        let order=view.assignments.values().map(|a|a.queue_order).max().unwrap_or(0).checked_add(1).ok_or("阶段队列编号已用尽")?;
        let mut active=(StageTier::Global,0);let mut last=active;
        for row in raw{if crate::region_state::is_marker(row){let(stage,index)=crate::region_state::marker_identity(row)?;active=(stage_edit::tier(stage)?,index);}
            else if is_visible(row){last=active;}}
        Ok(PrimitiveStageAssignment{stage:last.0,cell:last.1,queue_order:order})
    }
    pub fn validate_context_with_model_ids(&self,doc:&Document,node:NodeId,raw:&[RawRecord],ids:&[u64])->Result<DrawingRegion,String>{
        verify_projection_ids(&doc.materialize(node)?,raw,ids)?;
        DrawingRegion::from_history(raw)
    }
    pub fn validate_generated_with_ids(&self,doc:&Document,parent:NodeId,primitive:&Primitive,raw:&[RawRecord],known_ids:&[u64])
        ->Result<DrawingRegion,String>{
        let canvas=doc.materialize(parent)?;let mut candidate=primitive.normalized()?;
        if candidate.shape==ShapeKind::Background&&canvas.iter().any(|p|p.shape==ShapeKind::Background){return Err("模型不能生成重复背景".into());}
        let placeholder=canvas.iter().map(|p|p.id).max().unwrap_or(0).checked_add(1).ok_or("模型预览对象编号已用尽")?;
        candidate.id=placeholder;let mut full=canvas;full.push(candidate);let mut ids=known_ids.to_vec();ids.push(placeholder);
        verify_projection_ids(&full,raw,&ids)?;DrawingRegion::from_history(raw)
    }
    pub fn attach_with_model_ids(&mut self,doc:&Document,node:NodeId,raw:&[RawRecord],ids:&[u64])->Result<(),String>{
        let state=doc.nodes.get(node).ok_or("模型状态不存在")?;
        if state.source!=OperationSource::Model||self.entries.contains_key(&node)||self.control_overlays.contains_key(&node){return Err("模型状态上下文重复或来源无效".into());}
        let canvas=doc.materialize(node)?;verify_projection_ids(&canvas,raw,ids)?;
        let base=match state.parent{Some(parent)if self.entries.contains_key(&parent)=>HistoryBase::Node(parent),Some(parent)=>HistoryBase::Canvas(parent),None=>HistoryBase::Empty};
        let previous=self.resolve(doc,base)?;let keep=common_prefix(&previous,raw);
        let previous_ids=self.resolve_model_ids(doc,base)?;let id_keep=previous_ids.iter().zip(ids).take_while(|(a,b)|a==b).count();
        let parent_floor=state.parent.map(|parent|self.immutable_phase_floor(doc,parent)).transpose()?.unwrap_or(1);
        let floor=parent_floor.max(raw_phase(raw,false)?);
        let mut pending=state.parent.map(|parent|self.manual_reviews.get(&parent).cloned().unwrap_or_default()).unwrap_or_default();
        let parent_canvas_ids=state.parent.map(|parent|doc.materialize(parent)).transpose()?.unwrap_or_default().into_iter().map(|p|p.id).collect::<HashSet<_>>();
        for p in &canvas{if ids.contains(&p.id)&&parent_canvas_ids.contains(&p.id){if let Some(a)=p.stage_assignment{
            if a.stage.side()>parent_floor{pending.push((a.stage.side(),a.cell));}}}}
        let ended=permanent_end_regions(raw)?;pending.retain(|key|!ended.contains(key));
        self.entries.insert(node,HistoryDelta{base,keep,tail:raw[keep..].to_vec()});
        self.model_ids.insert(node,ModelIdDelta{base,keep:id_keep,tail:ids[id_keep..].to_vec(),phase_floor:floor});
        self.set_manual_reviews(node,pending);Ok(())
    }
    pub fn set_control_overlay_with_ids(&mut self,doc:&Document,node:NodeId,raw:&[RawRecord],ids:&[u64])->Result<(),String>{
        verify_projection_ids(&doc.materialize(node)?,raw,ids)?;
        if self.context_for(doc,node)?.as_deref()==Some(raw)&&self.model_ids_for_node(doc,node)?==ids{return Ok(());}
        let floor=self.immutable_phase_floor(doc,node)?.max(raw_phase(raw,false)?);
        let snapshot_ids=self.resolve_model_ids(doc,HistoryBase::Node(node))?;
        let snapshot_floor=self.immutable_phase_floor(doc,node)?;
        let mut pending=self.manual_reviews.get(&node).cloned().unwrap_or_default();
        for p in doc.materialize(node)?{if ids.contains(&p.id){if let Some(a)=p.stage_assignment{if a.stage.side()>snapshot_floor{pending.push((a.stage.side(),a.cell));}}}}
        let ended=permanent_end_regions(raw)?;pending.retain(|key|!ended.contains(key));pending.sort_unstable();pending.dedup();
        if !self.model_ids.contains_key(&node){self.model_ids.insert(node,ModelIdDelta{base:HistoryBase::Canvas(node),keep:snapshot_ids.len(),tail:Vec::new(),phase_floor:snapshot_floor});}
        self.control_overlays.insert(node,raw.to_vec());
        if ids==snapshot_ids{self.overlay_model_ids.remove(&node);}else{self.overlay_model_ids.insert(node,ids.to_vec());}
        self.overlay_phase_floors.insert(node,floor);self.overlay_manual_reviews.insert(node,pending);Ok(())
    }
    /// A fresh reference can contain future fine geometry while its model
    /// frontier remains coarse. Empty paused visits need no invented tokens.
    pub(crate) fn set_reference_frontier(&mut self,doc:&Document,node:NodeId,floor:usize)->Result<(),String>{
        if !matches!(floor,1|4|8)||doc.nodes.get(node).is_none_or(|n|n.source!=OperationSource::User||!n.children.is_empty()){
            return Err("参考绘制起点或阶段无效".into());
        }
        let raw=self.context_for(doc,node)?.ok_or("参考阶段缺少已校验的上下文")?;
        if raw_phase(&raw,false)?>floor||self.immutable_phase_floor(doc,node)?>floor{return Err("参考绘制起点尚未正确重置".into());}
        let state=self.model_ids.get_mut(&node).ok_or("参考阶段缺少图元映射")?;
        state.phase_floor=floor;self.overlay_phase_floors.insert(node,floor);Ok(())
    }
    pub(crate)fn project_ref(&self)->RawHistoryProjectRef<'_>{
        let mut entries=self.entries.iter().map(|(&node,d)|RawEntryRef{node,base:d.base,keep:d.keep,tail:&d.tail}).collect::<Vec<_>>();
        entries.sort_by_key(|d|d.node);
        let mut control_overlays=self.control_overlays.iter().map(|(&node,records)|OverlayRef{node,records}).collect::<Vec<_>>();
        control_overlays.sort_by_key(|o|o.node);
        let mut manual_reviews=self.manual_reviews.iter().map(|(&node,regions)|ManualReviewRef{node,regions}).collect::<Vec<_>>();
        manual_reviews.sort_by_key(|r|r.node);
        let mut model_ids=self.model_ids.iter().map(|(&node,d)|ModelIdRef{node,base:d.base,keep:d.keep,tail:&d.tail,phase_floor:d.phase_floor,
            overlay:self.overlay_model_ids.get(&node).map(|ids|ids.as_slice()),overlay_phase_floor:self.overlay_phase_floors.get(&node).copied(),
            overlay_manual_reviews:self.overlay_manual_reviews.get(&node).map(|reviews|reviews.as_slice())}).collect::<Vec<_>>();model_ids.sort_by_key(|d|d.node);
        RawHistoryProjectRef{entries,control_overlays,manual_reviews,model_ids,mode_evidence:self.mode_refs()}
    }
    pub(crate)fn from_project_data(data:RawHistoryProjectData)->Result<Self,String>{
        let mut result=Self::new();for entry in data.entries.0{let entry=entry.0;
            if result.entries.insert(entry.node,HistoryDelta{base:entry.base,keep:entry.keep,
                tail:entry.tail.0.into_iter().map(|r|r.0).collect()}).is_some(){return Err("项目含重复模型历史节点".into());}}
        for overlay in data.control_overlays.0{let overlay=overlay.0;
            if result.control_overlays.insert(overlay.node,overlay.records.0.into_iter().map(|r|r.0).collect()).is_some(){return Err("项目含重复控制历史节点".into());}}
        if let Some(reviews)=data.manual_reviews{for review in reviews.0{let review=review.0;
            if result.manual_reviews.insert(review.node,review.regions.0.into_iter().map(|r|r.0).collect()).is_some(){return Err("项目含重复用户待复查状态".into());}}}
        if let Some(ids)=data.model_ids{for d in ids.0{let d=d.0;
            if result.model_ids.insert(d.node,ModelIdDelta{base:d.base,keep:d.keep,tail:d.tail.0.into_iter().map(|id|id.0).collect(),phase_floor:d.phase_floor}).is_some(){return Err("项目含重复模型ID顺序".into());}
            if let Some(overlay)=d.overlay{result.overlay_model_ids.insert(d.node,overlay.0.into_iter().map(|id|id.0).collect());}
            if let Some(floor)=d.overlay_phase_floor{result.overlay_phase_floors.insert(d.node,floor);}
            if let Some(reviews)=d.overlay_manual_reviews{result.overlay_manual_reviews.insert(d.node,reviews.0.into_iter().map(|r|r.0).collect());}
        }}
        if let Some(entries)=data.mode_evidence{result.load_mode_data(entries.0.into_iter().map(|entry|entry.0).collect())?;}
        Ok(result)
    }
    pub(crate)fn validate_project(&self,doc:&Document,projections:&[project_data::Projection],
        budget:&mut project_data::Budget,work:&mut project_data::Work)->Result<(),String>{
        budget.charge(project_data::RAW_ENTRIES,self.entries.len())?;budget.charge(project_data::OVERLAYS,self.control_overlays.len())?;
        if self.entries.keys().chain(self.control_overlays.keys()).chain(self.manual_reviews.keys()).any(|&node|node>=doc.nodes.len()){
            return Err("项目模型历史引用不存在的节点".into());}
        let mut validated=(0..doc.nodes.len()).map(|_|None).collect::<Vec<Option<ProjectRawNode>>>();
        let mut memo=HashMap::new();
        for node in 0..doc.nodes.len(){let Some(delta)=self.entries.get(&node)else{continue};
            budget.charge(project_data::RECORDS,delta.tail.len())?;
            match delta.base{HistoryBase::Node(base)if base>=node=>return Err("项目模型历史循环或引用未来节点".into()),
                HistoryBase::Canvas(base)if base>node=>return Err("项目模型历史画布基底引用未来节点".into()),_=>{}}
            let mut state=project_prefix(delta.base,delta.keep,&validated,doc,projections,&mut memo,work)?;
            let mut states=Vec::new();states.try_reserve_exact(delta.tail.len()).map_err(|_|"内存不足，无法验证项目历史".to_owned())?;
            for row in &delta.tail{work.charge(1)?;state.push(row)?;states.push(state.clone());}
            if state.raw_len==0||(!self.model_ids.contains_key(&node)&&!state.projection.same(&projections[node])){return Err("项目模型历史可见投影与状态画布不一致".into());}
            validated[node]=Some(ProjectRawNode{base:delta.base,keep:delta.keep,len:state.raw_len,states,end:state});
        }
        let mut effective=HashMap::new();
        for(&node,records)in &self.control_overlays{budget.charge(project_data::RECORDS,records.len())?;
            let mut state=ProjectRawState::default();for row in records{work.charge(1)?;state.push(row)?;}
            if state.raw_len==0||(!self.model_ids.contains_key(&node)&&!state.projection.same(&projections[node])){return Err("项目控制历史可见投影与画布不一致".into());}effective.insert(node,state);}
        budget.charge(project_data::MANUAL_REVIEWS,self.manual_reviews.len())?;
        for(&node,regions)in &self.manual_reviews{budget.charge(project_data::RECORDS,regions.len())?;
            if regions.is_empty()||regions.len()>80{return Err("项目用户复查区域列表无效".into());}
            let mut unique=HashSet::new();let state=effective.get(&node).or_else(||validated[node].as_ref().map(|r|&r.end))
                .ok_or("用户复查区域缺少模型上下文")?;
            for &(side,index)in regions{
                if !matches!(side,4|8)||index>=side*side||!unique.insert((side,index))||!state.saw_region{
                    return Err("项目用户复查区域无效".into());}
                if !state.has_end(side,index)&&state.saw_fine&&side!=8{return Err("项目未完成用户复查不能退回粗网格".into());}
            }
        }
        stage_project_validation::validate_mapped_projects(self,doc,projections,&validated,&effective,budget,work)?;
        self.validate_mode_data(doc,budget,work)
    }
    pub fn new() -> Self {
        Self::default()
    }

    pub fn manual_review_regions(&self,doc:&Document,node:NodeId)->Result<Vec<(usize,usize)>,String>{
        if node>=doc.nodes.len(){return Err("用户复查状态不存在".into());}
        if let Some(origin)=past_color_repair::historical_color_origin(self,doc,node)?{
            return self.manual_review_regions(doc,origin);
        }
        if !self.manual_reviews.contains_key(&node)&&!self.overlay_manual_reviews.contains_key(&node){return Ok(Vec::new());}
        let raw=match self.context_for(doc,node)?{Some(raw)=>raw,None=>canvas_records(doc,node)?};
        let mut pending=self.overlay_manual_reviews.get(&node).or_else(||self.manual_reviews.get(&node)).cloned().unwrap_or_default();
        let ended=permanent_end_regions(&raw)?;pending.retain(|key|!ended.contains(key));Ok(pending)
    }
    fn set_manual_reviews(&mut self,node:NodeId,mut regions:Vec<(usize,usize)>){
        regions.sort_unstable();regions.dedup();if regions.is_empty(){self.manual_reviews.remove(&node);}else{self.manual_reviews.insert(node,regions);}
    }
    pub(crate) fn creation_review_snapshot(&self,node:NodeId)->Vec<(usize,usize)>{
        self.manual_reviews.get(&node).cloned().unwrap_or_default()
    }
    pub(crate) fn exclude_frozen_review_keys(&mut self,node:NodeId,keys:&[(usize,usize)],overlay:bool){
        if keys.is_empty(){return;}
        let map=if overlay{&mut self.overlay_manual_reviews}else{&mut self.manual_reviews};
        if let Some(reviews)=map.get_mut(&node){reviews.retain(|key|!keys.contains(key));}
        // An empty overlay deliberately overrides the immutable creation list.
        if !overlay&&map.get(&node).is_some_and(Vec::is_empty){map.remove(&node);}
    }
    fn reviewed_after_model(&self,node:NodeId,raw:&[RawRecord])->Result<Vec<(usize,usize)>,String>{
        let mut pending=self.manual_reviews.get(&node).cloned().unwrap_or_default();
        if pending.is_empty(){return Ok(pending);}
        let ended=permanent_end_regions(raw)?;pending.retain(|region|!ended.contains(region));Ok(pending)
    }

    /// Prepare only remapped metadata and destination capacities. Immutable raw
    /// tails/overlays and document checkpoints are moved, never duplicated.
    fn prepare_delete_subtrees(&self, doc: &Document, plan: &HistoryDeletionPlan) -> Result<Self, String> {
        plan.validate_source(doc)?;
        let result = plan.result();
        if result.root_reset { return Ok(Self::new()); }
        let mapping = &result.old_to_new;
        if self.entries.keys().chain(self.control_overlays.keys()).chain(self.manual_reviews.keys())
            .chain(self.model_ids.keys()).chain(self.overlay_model_ids.keys()).chain(self.overlay_phase_floors.keys())
            .chain(self.mode_evidence.keys())
            .chain(self.overlay_manual_reviews.keys()).any(|&node| node >= mapping.len()) {
            return Err("模型上下文引用不存在的状态节点，未删除任何历史".into());
        }
        let mut lengths = vec![None; mapping.len()];
        // Deltas created by attach/reconcile always refer to an older node or
        // their own canvas. Resolve lengths once, without replaying N raw lists.
        for node in 0..mapping.len() {
            if mapping[node].is_none() { continue; }
            let Some(delta) = self.entries.get(&node) else { continue; };
            let base_len = match delta.base {
                HistoryBase::Node(base) => {
                    if base >= node || mapping.get(base).is_none_or(Option::is_none) {
                        return Err("存活模型上下文的前置状态无效或将被删除，原历史已保留".into());
                    }
                    lengths[base].ok_or("模型上下文缺少有效的前置增量")?
                }
                HistoryBase::Canvas(base) => {
                    if base > node || mapping.get(base).is_none_or(Option::is_none) {
                        return Err("存活模型上下文的画布基底无效或将被删除，原历史已保留".into());
                    }
                    doc.nodes[base].primitive_count.checked_add(1).ok_or("模型上下文长度溢出")?
                }
                HistoryBase::Empty => 0,
            };
            if delta.keep > base_len { return Err("模型上下文增量超出前置长度，原历史已保留".into()); }
            lengths[node] = Some(delta.keep.checked_add(delta.tail.len()).ok_or("模型上下文长度溢出")?);
        }
        // Control-only overlays are rare and public. Validate their full
        // projection once; immutable delta payloads retain their exact bases.
        for (&node, records) in &self.control_overlays {
            if mapping[node].is_some() {
                self.verify_node_raw(doc,node,records,true)?;
                DrawingRegion::from_history(records)?;
            }
        }
        let mut prepared = Self::new();
        prepared.entries.try_reserve(self.entries.keys().filter(|&&node| mapping[node].is_some()).count())
            .map_err(|_|"内存不足，无法更新模型上下文；原历史已保留".to_owned())?;
        prepared.control_overlays.try_reserve(self.control_overlays.keys().filter(|&&node| mapping[node].is_some()).count())
            .map_err(|_|"内存不足，无法更新控制记录；原历史已保留".to_owned())?;
        prepared.manual_reviews.try_reserve(self.manual_reviews.keys().filter(|&&node|mapping[node].is_some()).count())
            .map_err(|_|"内存不足，无法更新用户复查状态；原历史已保留".to_owned())?;
        // ID-order deltas obey the same ancestor closure as raw records. They
        // remain sparse and reference immutable bases after compaction.
        let mut id_lengths=vec![None;mapping.len()];
        for node in 0..mapping.len(){if mapping[node].is_none(){continue;}let Some(delta)=self.model_ids.get(&node)else{continue};
            let base_len=match delta.base{
                HistoryBase::Node(base)=>{if base>=node||mapping.get(base).is_none_or(Option::is_none){return Err("模型ID顺序基底将被删除或无效".into());}
                    id_lengths[base].unwrap_or(doc.nodes[base].primitive_count)},
                HistoryBase::Canvas(base)=>{if base>node||mapping.get(base).is_none_or(Option::is_none){return Err("模型ID画布基底将被删除或无效".into());}doc.nodes[base].primitive_count},
                HistoryBase::Empty=>0};
            if delta.keep>base_len{return Err("模型ID顺序增量超出基底".into());}
            id_lengths[node]=Some(delta.keep.checked_add(delta.tail.len()).ok_or("模型ID长度溢出")?);
        }
        for(count,destination)in [
            (self.model_ids.keys().filter(|&&node|mapping[node].is_some()).count(),0usize),
            (self.overlay_model_ids.keys().filter(|&&node|mapping[node].is_some()).count(),1),
            (self.overlay_phase_floors.keys().filter(|&&node|mapping[node].is_some()).count(),2),
            (self.overlay_manual_reviews.keys().filter(|&&node|mapping[node].is_some()).count(),3)]{
            let reserve=match destination{0=>prepared.model_ids.try_reserve(count),1=>prepared.overlay_model_ids.try_reserve(count),
                2=>prepared.overlay_phase_floors.try_reserve(count),_=>prepared.overlay_manual_reviews.try_reserve(count)};
            reserve.map_err(|_|"内存不足，无法更新阶段归属；原历史已保留".to_owned())?;
        }
        prepared.mode_evidence.try_reserve(self.mode_evidence.keys().filter(|&&node|mapping[node].is_some()).count())
            .map_err(|_|"内存不足，无法更新模式完成证据；原历史已保留".to_owned())?;
        Ok(prepared)
    }

    /// Atomic document + model-history deletion. All fallible validation and
    /// allocations precede document mutation; the final history move cannot fail.
    pub fn apply_delete_subtrees(&mut self, doc: &mut Document, plan: HistoryDeletionPlan) -> Result<HistoryDeletion, String> {
        let mut prepared = self.prepare_delete_subtrees(doc, &plan)?;
        let result = doc.apply_delete_subtrees_plan(plan)?;
        if !result.root_reset {
            for (old_id, mut delta) in std::mem::take(&mut self.entries) {
                let Some(node) = result.old_to_new[old_id] else { continue; };
                delta.base = match delta.base {
                    HistoryBase::Node(base) => HistoryBase::Node(result.old_to_new[base].expect("raw node base was validated")),
                    HistoryBase::Canvas(base) => HistoryBase::Canvas(result.old_to_new[base].expect("raw canvas base was validated")),
                    HistoryBase::Empty => HistoryBase::Empty,
                };
                prepared.entries.insert(node, delta);
            }
            for (old_id, records) in std::mem::take(&mut self.control_overlays) {
                if let Some(node) = result.old_to_new[old_id] { prepared.control_overlays.insert(node, records); }
            }
            for(old_id,regions)in std::mem::take(&mut self.manual_reviews){if let Some(node)=result.old_to_new[old_id]{prepared.manual_reviews.insert(node,regions);}}
            for(old_id,mut delta)in std::mem::take(&mut self.model_ids){let Some(node)=result.old_to_new[old_id]else{continue};
                delta.base=match delta.base{HistoryBase::Node(base)=>HistoryBase::Node(result.old_to_new[base].expect("ID base validated")),
                    HistoryBase::Canvas(base)=>HistoryBase::Canvas(result.old_to_new[base].expect("ID canvas validated")),HistoryBase::Empty=>HistoryBase::Empty};
                prepared.model_ids.insert(node,delta);}
            for(old_id,ids)in std::mem::take(&mut self.overlay_model_ids){if let Some(node)=result.old_to_new[old_id]{prepared.overlay_model_ids.insert(node,ids);}}
            for(old_id,floor)in std::mem::take(&mut self.overlay_phase_floors){if let Some(node)=result.old_to_new[old_id]{prepared.overlay_phase_floors.insert(node,floor);}}
            for(old_id,regions)in std::mem::take(&mut self.overlay_manual_reviews){if let Some(node)=result.old_to_new[old_id]{prepared.overlay_manual_reviews.insert(node,regions);}}
            for(old_id,value)in std::mem::take(&mut self.mode_evidence){if let Some(node)=result.old_to_new[old_id]{prepared.mode_evidence.insert(node,value);}}
        }
        *self = prepared;
        Ok(result)
    }

    /// States without explicitly attached/reconciled context return None.
    /// This never walks upward looking for a possibly stale ancestor cache.
    pub fn context_for(
        &self,
        doc: &Document,
        node: NodeId,
    ) -> Result<Option<Vec<RawRecord>>, String> {
        if node >= doc.nodes.len() {
            return Err("状态节点不存在".into());
        }
        if let Some(records) = self.control_overlays.get(&node) {
            self.verify_node_raw(doc,node,records,true)?;
            return Ok(Some(records.clone()));
        }
        if !self.entries.contains_key(&node) {
            return Ok(None);
        }
        let records = self.resolve(doc, HistoryBase::Node(node))?;
        self.verify_node_raw(doc,node,&records,false)?;
        Ok(Some(records))
    }

    /// Persist a no-visible-output request without modifying a shared delta
    /// base. Every accepted update only appends to this node's public context.
    pub fn set_control_overlay(
        &mut self,
        doc: &Document,
        node: NodeId,
        raw: &[RawRecord],
    ) -> Result<(), String> {
        verify_visible(doc, node, raw)?;
        DrawingRegion::from_history(raw)?;
        let previous = self.context_for(doc, node)?.unwrap_or(canvas_records(doc, node)?);
        if !raw.starts_with(&previous) {
            return Err("控制记录不是当前上下文的追加内容，保留原上下文".into());
        }
        self.control_overlays.insert(node, raw.to_vec());
        Ok(())
    }

    /// Sort contributions only inside the original stage/cell's visible slots.
    /// The history controls determine provenance, rather than the shape's
    /// current center: a manual move must not silently transfer its history.
    /// Re-entered visits of the same cell share one bucket, while every raw
    /// marker/interrupt/end and the interleaving of other cells stays fixed.
    pub fn contribution_ordered_canvas(
        &self,
        doc: &Document,
        node: NodeId,
    ) -> Result<Vec<Primitive>, String> {
        let canvas = doc.materialize(node)?;
        if self.model_ids.contains_key(&node)||canvas.iter().any(|p|p.stage_assignment.is_some()){
            let legacy=self.stage_map(doc,node)?;let view=stage_edit::view(&canvas,&legacy,false)?;
            let background=canvas.iter().find(|p|p.shape==ShapeKind::Background).map(|p|p.color).unwrap_or([255;3]);
            let scores=canvas.iter().map(|p|crate::document::primitive_contribution(p,background)).collect::<Result<Vec<_>,_>>()?;
            let mut groups=BTreeMap::<(StageTier,usize),Vec<usize>>::new();
            for(index,p)in canvas.iter().enumerate(){let a=view.assignments[&p.id];groups.entry((a.stage,a.cell)).or_default().push(index);}
            let mut ordered=canvas.clone();for positions in groups.values(){let mut ranked=positions.clone();
                ranked.sort_by(|&a,&b|{(canvas[b].shape==ShapeKind::Background).cmp(&(canvas[a].shape==ShapeKind::Background))
                    .then_with(||scores[b].total_cmp(&scores[a]))});
                for(&destination,&source)in positions.iter().zip(&ranked){ordered[destination]=canvas[source].clone();}}
            return Ok(ordered);
        }
        let raw = match self.context_for(doc, node)? {
            Some(raw) => raw,
            None => canvas_records(doc, node)?,
        };
        DrawingRegion::from_history(&raw)?;
        let background = canvas.iter().find(|primitive| primitive.shape == ShapeKind::Background)
            .map(|primitive| primitive.color).unwrap_or([255; 3]);
        let scores = canvas.iter()
            .map(|primitive| crate::document::primitive_contribution(primitive, background))
            .collect::<Result<Vec<_>, _>>()?;
        let mut groups: HashMap<(Stage, usize), Vec<usize>> = HashMap::new();
        let mut current_region = Some((Stage::Global, 0));
        let mut visible_index = 0;
        for row in &raw {
            if row[6..9] == [REGION_MARKER; 3] {
                current_region = Some(crate::region_state::marker_identity(row)?);
            } else if row[0] == TRUE_END || row[0] == INTERRUPT {
                current_region = None;
            } else if is_visible(row) {
                let region = current_region.ok_or("无法重排序：图元缺少原始绘制区域")?;
                if visible_index >= canvas.len() {
                    return Err("无法重排序：区域历史与画布图元数量不一致".into());
                }
                groups.entry(region).or_default().push(visible_index);
                visible_index += 1;
            }
        }
        if visible_index != canvas.len() {
            return Err("无法重排序：区域历史遗漏画布图元".into());
        }
        let mut ordered = canvas.clone();
        for positions in groups.values() {
            let mut ranked = positions.clone();
            ranked.sort_by(|&left, &right| {
                let left_background = canvas[left].shape == ShapeKind::Background;
                let right_background = canvas[right].shape == ShapeKind::Background;
                right_background.cmp(&left_background)
                    .then_with(|| scores[right].total_cmp(&scores[left]))
            });
            for (&destination, &source) in positions.iter().zip(&ranked) {
                ordered[destination] = canvas[source].clone();
            }
        }
        Ok(ordered)
    }

    /// Attach an already-committed, complete model primitive's context.
    /// The visible projection must agree exactly with the immutable node.
    pub fn attach(
        &mut self,
        doc: &Document,
        node: NodeId,
        rawfull: &[RawRecord],
    ) -> Result<(), String> {
        let state = doc.nodes.get(node).ok_or("状态节点不存在")?;
        if state.source != OperationSource::Model {
            return Err("只能为模型完成的状态附加生成上下文".into());
        }
        if self.entries.contains_key(&node) || self.control_overlays.contains_key(&node) {
            return Err("节点已有关联上下文，请使用 finish 追加控制记录".into());
        }
        verify_visible(doc, node, rawfull)?;
        DrawingRegion::from_history(rawfull)?;
        let base = match state.parent {
            Some(parent) if self.entries.contains_key(&parent) => HistoryBase::Node(parent),
            Some(parent) => HistoryBase::Canvas(parent),
            None => HistoryBase::Empty,
        };
        let previous = self.resolve(doc, base.clone())?;
        let keep = common_prefix(&previous, rawfull);
        let pending=if let Some(parent)=state.parent{self.reviewed_after_model(parent,rawfull)?}else{Vec::new()};
        self.entries.insert(
            node,
            HistoryDelta {
                base,
                keep,
                tail: rawfull[keep..].to_vec(),
            },
        );
        self.set_manual_reviews(node,pending);
        Ok(())
    }

    /// A region may end without producing another visible primitive. Keep that
    /// trailing end/control record on the same state, retaining the original
    /// base so this cannot create a self-reference. Existing child prefixes
    /// remain valid because finishing is append-only.
    pub fn finish(
        &mut self,
        doc: &Document,
        current: NodeId,
        rawfull: &[RawRecord],
    ) -> Result<(), String> {
        if current >= doc.nodes.len() {
            return Err("状态节点不存在".into());
        }
        verify_visible(doc, current, rawfull)?;
        DrawingRegion::from_history(rawfull)?;
        if self.control_overlays.contains_key(&current) {
            // Once a node has an overlay, finishing it must append to that
            // public context rather than regress to or rewrite its old base.
            return self.set_control_overlay(doc, current, rawfull);
        }
        let Some(original) = self.entries.get(&current).cloned() else {
            // A fresh request may select a region but stop before producing a
            // visible primitive. Preserve its validated control state on the
            // existing node without inventing an image/history-tree change.
            DrawingRegion::from_history(rawfull)?;
            let base = HistoryBase::Canvas(current);
            let base_records = self.resolve(doc, base.clone())?;
            let keep = common_prefix(&base_records, rawfull);
            self.entries.insert(
                current,
                HistoryDelta {
                    base,
                    keep,
                    tail: rawfull[keep..].to_vec(),
                },
            );
            return Ok(());
        };
        let old = self.resolve(doc, HistoryBase::Node(current))?;
        if !rawfull.starts_with(&old) {
            return Err("生成结束记录不是当前上下文的追加内容，保留原上下文".into());
        }
        let base_records = self.resolve(doc, original.base.clone())?;
        let keep = common_prefix(&base_records, rawfull);
        self.entries.insert(
            current,
            HistoryDelta {
                base: original.base,
                keep,
                tail: rawfull[keep..].to_vec(),
            },
        );
        Ok(())
    }

    /// Reconcile a single already-committed user operation. Spatial changes
    /// select the same-stage cell containing their center; color/layer-only
    /// edits retain the current active region. Existing visible primitives
    /// occur exactly once and retain the new canvas's stacking order.
    pub fn reconcile_manual(
        &mut self,
        doc: &Document,
        parent: NodeId,
        node: NodeId,
    ) -> Result<DrawingRegion, String> {
        if parent == node {
            let raw = self
                .context_for(doc, node)?
                .unwrap_or(canvas_records(doc, node)?);
            return DrawingRegion::from_history(&raw);
        }
        let state = doc.nodes.get(node).ok_or("人工状态节点不存在")?;
        if state.parent != Some(parent) || state.source != OperationSource::User {
            return Err("人工上下文更新必须对应刚提交的单个人工状态".into());
        }
        if self.entries.contains_key(&node) || self.control_overlays.contains_key(&node) {
            return Err("该人工状态已有关联上下文".into());
        }
        let old = doc.materialize(parent)?;
        let new = doc.materialize(node)?;
        if self.model_ids.contains_key(&parent)||new.iter().any(|p|p.stage_assignment.is_some()){
            let mut plan=self.prepare_edit(doc,parent,new.clone(),None,false).map_err(|e|e.to_string())?;
            // This legacy entry point accepts an already committed canvas.
            // New UI edits prepare before committing; old callers can retain
            // their optional legacy metadata while still getting mapped raw.
            plan.canvas=new;
            return self.apply_prepared_edit(doc,parent,node,plan);
        }
        let old_by_id: HashMap<u64, &Primitive> = old.iter().map(|p| (p.id, p)).collect();
        let new_by_id: HashMap<u64, &Primitive> = new.iter().map(|p| (p.id, p)).collect();
        let parent_raw = self
            .context_for(doc, parent)?
            .unwrap_or(canvas_records(doc, parent)?);
        DrawingRegion::from_history(&parent_raw)?;
        let review_stage=self.editing_stage(doc,parent)?;
        let old_stages=self.stage_map(doc,parent)?;
        let current_owned=|p:&Primitive|old_stages.get(&p.id).copied().unwrap_or(review_stage)==review_stage;
        let mut touched=Vec::new();
        let mut touch=|p:&Primitive|{
            if p.shape!=ShapeKind::Background&&current_owned(p){if let Some(index)=DrawingRegion::cell_for_center(review_stage,p.x,p.y){
                let key=(review_stage.side(),index);if !touched.contains(&key){touched.push(key);}
            }}
        };
        for patch in &state.patches{match patch{
            Patch::Add{primitive,..}|Patch::Replace{primitive}=>{if let Some(old)=old_by_id.get(&primitive.id){touch(old);}touch(primitive);},
            Patch::Remove{id}=>{if let Some(old)=old_by_id.get(id){touch(old);}},
            Patch::Move{..}=>{},
        }}
        let mut events = Vec::new();
        for patch in &state.patches {
            let p = match patch {
                Patch::Add { primitive, .. } | Patch::Replace { primitive } => primitive,
                _ => continue,
            };
            if p.shape == ShapeKind::Background {
                continue;
            }
            if !current_owned(p){continue;}
            let spatial = old_by_id
                .get(&p.id)
                .map(|old| {
                    old.x != p.x
                        || old.y != p.y
                        || old.width != p.width
                        || old.height != p.height
                        || old.angle_deg != p.angle_deg
                        || old.shape != p.shape
                })
                .unwrap_or(true);
            if spatial {
                events.push(p.id);
            }
        }
        let simple_append = new.len() >= old.len() && new[..old.len()] == old;
        let mut raw;
        if simple_append {
            // The common draw/paste path preserves the exact prior raw prefix.
            // A newly drawn out-of-cell primitive follows interrupt+begin.
            raw = parent_raw.clone();
            for p in &new[old.len()..] {
                if p.shape != ShapeKind::Background {
                    select_for_center(&mut raw, review_stage, p.x, p.y)?;
                }
                raw.push(p.to_tokens()?.map(u32::from));
            }
        } else {
            // Map surviving old slots to the reordered retained IDs. Controls
            // remain in their original sequence while visible rows are edited
            // in place. Reordered/modified old primitives are never appended a
            // second time just to signal a new active editing cell.
            let retained: Vec<&Primitive> = new
                .iter()
                .filter(|p| old_by_id.contains_key(&p.id))
                .collect();
            let mut reordered = Vec::with_capacity(parent_raw.len());
            let mut old_index = 0;
            let mut retained_index = 0;
            for row in &parent_raw {
                if is_visible(row) {
                    let old_id = old.get(old_index).ok_or("父状态可见记录数量不一致")?.id;
                    old_index += 1;
                    if new_by_id.contains_key(&old_id) {
                        let replacement = retained
                            .get(retained_index)
                            .ok_or("图层重排记录数量不一致")?;
                        reordered.push(replacement.to_tokens()?.map(u32::from));
                        retained_index += 1;
                    }
                } else {
                    reordered.push(*row);
                }
            }
            if old_index != old.len() || retained_index != retained.len() {
                return Err("人工记录重建遗漏图元".into());
            }
            // Interior insertions preserve the intended stacking position;
            // their target-cell controls are applied at the final edit point.
            // Ordinary tail insertions remain after their region switch.
            let mut rebuilt =
                Vec::with_capacity(reordered.len() + new.len().saturating_sub(retained.len()));
            let mut desired = 0;
            let mut existing = 0;
            for row in reordered {
                if row == BOS_RECORD {
                    rebuilt.push(row);
                    if let Some(bg) = new.first().filter(|p| {
                        p.shape == ShapeKind::Background && !old_by_id.contains_key(&p.id)
                    }) {
                        rebuilt.push(bg.to_tokens()?.map(u32::from));
                        desired += 1;
                    }
                } else if is_visible(&row) {
                    let id = retained[existing].id;
                    while desired < new.len() && new[desired].id != id {
                        if old_by_id.contains_key(&new[desired].id) {
                            return Err("人工图层重排无法对应保留记录".into());
                        }
                        rebuilt.push(new[desired].to_tokens()?.map(u32::from));
                        desired += 1;
                    }
                    rebuilt.push(row);
                    desired += 1;
                    existing += 1;
                } else {
                    rebuilt.push(row);
                }
            }
            raw = rebuilt;
            let trailing: HashSet<u64> = new[desired..].iter().map(|p| p.id).collect();
            let mut appended = HashSet::new();
            for id in events {
                let p = new_by_id[&id];
                select_for_center(&mut raw, review_stage, p.x, p.y)?;
                if trailing.contains(&id) {
                    raw.push(p.to_tokens()?.map(u32::from));
                    appended.insert(id);
                }
            }
            for p in &new[desired..] {
                if appended.contains(&p.id) {
                    continue;
                }
                if p.shape != ShapeKind::Background {
                    select_for_center(&mut raw, review_stage, p.x, p.y)?;
                }
                raw.push(p.to_tokens()?.map(u32::from));
            }
        }
        if !touched.is_empty(){reopen_manual_regions(&mut raw,&touched)?;}
        let mut pending=self.reviewed_after_model(parent,&raw)?;pending.extend(touched);
        verify_visible(doc, node, &raw)?;
        let region = DrawingRegion::from_history(&raw)?;
        let base = if self.entries.contains_key(&parent) {
            HistoryBase::Node(parent)
        } else {
            HistoryBase::Canvas(parent)
        };
        let previous = self.resolve(doc, base.clone())?;
        let keep = common_prefix(&previous, &raw);
        self.entries.insert(
            node,
            HistoryDelta {
                base,
                keep,
                tail: raw[keep..].to_vec(),
            },
        );
        self.set_manual_reviews(node,pending);
        Ok(region)
    }

    pub fn stored_record_count(&self) -> usize {
        self.entries.values().map(|d| d.tail.len()).sum::<usize>()
            + self.control_overlays.values().map(Vec::len).sum::<usize>()
    }

    /// Validate a lane callback before creating its visible state. This keeps
    /// malformed/stale/duplicate outputs from leaving a phantom tree node.
    pub fn validate_generated_at(
        &self,
        doc: &Document,
        parent: NodeId,
        primitive: &Primitive,
        rawfull: &[RawRecord],
    ) -> Result<DrawingRegion, String> {
        let mut canvas = doc.materialize(parent)?;
        let primitive = primitive.normalized()?;
        if primitive.shape == ShapeKind::Background {
            if canvas.iter().any(|p| p.shape == ShapeKind::Background) {
                return Err("分支已有背景，回调不能再添加背景".into());
            }
            canvas.insert(0, primitive);
        } else {
            canvas.push(primitive);
        }
        verify_projection(&canvas, rawfull)?;
        // Also verify the stored base before its future append can mutate doc.
        self.context_for(doc, parent)?;
        DrawingRegion::from_history(rawfull)
    }

    pub fn validate_context_for_node(
        &self,
        doc: &Document,
        node: NodeId,
        rawfull: &[RawRecord],
    ) -> Result<DrawingRegion, String> {
        verify_visible(doc, node, rawfull)?;
        DrawingRegion::from_history(rawfull)
    }

    fn resolve(&self, doc: &Document, mut cursor: HistoryBase) -> Result<Vec<RawRecord>, String> {
        // Iterative reconstruction avoids stack growth and repeated full-list
        // allocations on long primitive-by-primitive generation chains.
        let mut path = Vec::new();
        let mut seen = HashSet::new();
        let mut records = loop {
            match cursor {
                HistoryBase::Node(node) => {
                    // Resolve only the immutable delta graph. Public overlays
                    // belong to their exact node, never to existing children.
                    if !seen.insert(node) {
                        return Err("生成上下文存在循环引用".into());
                    }
                    let delta = self.entries.get(&node).ok_or("生成上下文缺少前置状态")?;
                    path.push(delta);
                    cursor = delta.base.clone();
                }
                HistoryBase::Canvas(node) => break canvas_records(doc, node)?,
                HistoryBase::Empty => break Vec::new(),
            }
        };
        for delta in path.into_iter().rev() {
            if delta.keep > records.len() {
                return Err("生成上下文增量超出前置记录长度".into());
            }
            records.truncate(delta.keep);
            records.extend_from_slice(&delta.tail);
        }
        Ok(records)
    }
}

#[derive(Clone,Default)]struct ProjectRawState{
    raw_len:usize,projection:project_data::Projection,active:Option<(Stage,usize)>,suspended:Option<Stage>,saw_region:bool,saw_fine:bool,
    ended_coarse:u16,ended_fine:u64,
    // Match raw_phase without resolving and replaying every ancestor at each
    // saved node. An immediately interrupted empty marker contributes no phase.
    phase_explicit:usize,phase_evidence:usize,phase_before_marker:usize,last_marker:bool,
}
impl ProjectRawState{
    fn phase(&self,evidence_only:bool)->usize{if evidence_only{self.phase_evidence.max(1)}else{self.phase_explicit.max(1)}}
    fn has_end(&self,side:usize,index:usize)->bool{
        match side{4=>self.ended_coarse&(1u16<<index)!=0,8=>self.ended_fine&(1u64<<index)!=0,_=>true}
    }
    fn push(&mut self,row:&RawRecord)->Result<(),String>{
        let previous_phase=self.active.map_or(1,|(stage,_)|stage.side());
        if self.raw_len==0{if *row!=BOS_RECORD{return Err("项目模型历史缺少规范BOS".into());}}
        else if row[0]==BOS{return Err("项目模型历史含重复BOS".into());}
        else if crate::region_state::is_marker(row){let key=crate::region_state::marker_identity(row)?;
            if self.active.is_some(){return Err("项目区域开始记录嵌套".into());}
            if self.suspended.is_some_and(|stage|stage!=key.0){return Err("项目中断记录后阶段不一致".into());}
            if self.saw_fine&&key.0==Stage::Coarse{return Err("项目细网格之后不能回到粗网格".into());}
            self.suspended=None;self.saw_region=true;self.saw_fine|=key.0==Stage::Fine;self.active=Some(key);
        }else if row[0]==TRUE_END||row[0]==INTERRUPT{
            let expected=if row[0]==TRUE_END{crate::region_state::END_RECORD}else{crate::region_state::INTERRUPT_RECORD};
            if *row!=expected{return Err("项目结束控制记录不规范".into());}
            let key=self.active.take().ok_or("项目没有活动区域却出现结束控制")?;
            if row[0]==INTERRUPT{self.suspended=Some(key.0);}
            else if key.0==Stage::Coarse{self.ended_coarse|=1u16<<key.1;}else{self.ended_fine|=1u64<<key.1;}
        }else if row.iter().all(|&v|v==PAD){if self.suspended.is_some(){return Err("项目中断后必须立即开始同阶段区域".into());}}
        else{
            if self.suspended.is_some()||self.saw_region&&self.active.is_none(){return Err("项目图元不在有效绘制区域序列中".into());}
            crate::region_state::validate_visible(row)?;self.projection.push(row)?;
        }
        if crate::region_state::is_marker(row){
            self.phase_before_marker=self.phase_explicit;
            self.phase_explicit=self.phase_explicit.max(self.active.map_or(1,|(stage,_)|stage.side()));
            self.last_marker=true;
        }else{
            if row==&crate::region_state::INTERRUPT_RECORD&&self.last_marker{self.phase_explicit=self.phase_before_marker;}
            if is_visible(row)||row==&crate::region_state::END_RECORD{self.phase_evidence=self.phase_evidence.max(previous_phase);}
            self.last_marker=false;
        }
        self.raw_len=self.raw_len.checked_add(1).ok_or("项目模型历史长度溢出")?;Ok(())
    }
}
struct ProjectRawNode{base:HistoryBase,keep:usize,len:usize,states:Vec<ProjectRawState>,end:ProjectRawState}
#[derive(Clone,Copy,PartialEq,Eq,Hash)]struct ProjectPrefixKey{node:NodeId,index:usize,canvas:bool}
fn project_prefix(mut base:HistoryBase,index:usize,validated:&[Option<ProjectRawNode>],doc:&Document,
    projections:&[project_data::Projection],memo:&mut HashMap<ProjectPrefixKey,ProjectRawState>,work:&mut project_data::Work)
    ->Result<ProjectRawState,String>{
    let mut path=Vec::new();let state=loop{work.charge(1)?;match base{
        HistoryBase::Empty=>{if index!=0{return Err("项目历史截断超出空基底".into());}break ProjectRawState::default()},
        HistoryBase::Node(node)=>{
            let data=validated.get(node).and_then(Option::as_ref).ok_or("项目缺少前置模型历史")?;
            if index>data.len{return Err("项目模型历史keep超出基底长度".into());}
            if index==data.len{break data.end.clone();}
            let key=ProjectPrefixKey{node,index,canvas:false};if let Some(state)=memo.get(&key){break state.clone();}
            if index>data.keep{break data.states[index-data.keep-1].clone();}
            path.push(key);base=data.base;
        },
        HistoryBase::Canvas(node)=>{
            let projection=projections.get(node).ok_or("项目画布历史基底不存在")?;
            if index>projection.count+1{return Err("项目模型历史keep超出画布基底".into());}
            if index==0{break ProjectRawState::default();}
            if index==projection.count+1{break ProjectRawState{raw_len:index,projection:projection.clone(),..Default::default()};}
            let key=ProjectPrefixKey{node,index,canvas:true};if let Some(state)=memo.get(&key){break state.clone();}
            work.charge(projection.count.saturating_mul(65).saturating_add(index))?;
            let canvas=doc.materialize(node)?;let mut state=ProjectRawState::default();state.push(&BOS_RECORD)?;
            for primitive in canvas.iter().take(index-1){state.push(&primitive.to_tokens()?.map(u32::from))?;}
            path.push(key);break state;
        },
    }};
    if memo.len().checked_add(path.len()).is_none_or(|n|n>500_000){return Err("项目历史前缀依赖超过安全预算".into());}
    memo.try_reserve(path.len()).map_err(|_|"内存不足，无法验证项目前缀依赖".to_owned())?;
    for key in path{memo.insert(key,state.clone());}Ok(state)
}

fn common_prefix(a: &[RawRecord], b: &[RawRecord]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

fn is_visible(row: &RawRecord) -> bool {
    ![BOS, TRUE_END, INTERRUPT, PAD].contains(&row[0])
        && !row[6..].iter().all(|&v| v == REGION_MARKER)
}

/// One center decision per user operation, including a half-pixel gap. We do
/// not move/clip the user's primitive to satisfy the model's candidate mask.
fn select_for_center(raw: &mut Vec<RawRecord>, stage:Stage, x: f32, y: f32) -> Result<(), String> {
    let state = DrawingRegion::from_history(raw)?;
    if stage == Stage::Global {
        return Ok(());
    }
    let index = DrawingRegion::cell_for_center(stage, x, y).ok_or("人工图元的目标区域无法定位")?;
    if state.index.is_some() && state.stage==stage && state.contains_center(x, y) {
        return Ok(());
    }
    if state.index.is_some() {
        if state.stage!=stage{return Err("活动块与实际绘制阶段不一致，保留原上下文".into());}
        raw.push([INTERRUPT, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD]);
    }
    raw.push(crate::region_state::marker_record(stage, index)?);
    Ok(())
}

fn permanent_end_regions(raw:&[RawRecord])->Result<HashSet<(usize,usize)>,String>{
    let mut active=None;let mut ended=HashSet::new();
    for row in raw{
        if crate::region_state::is_marker(row){let(stage,index)=crate::region_state::marker_identity(row)?;active=Some((stage.side(),index));}
        else if row==&crate::region_state::END_RECORD{ended.insert(active.take().ok_or("没有活动用户区域却出现永久结束")?);}
        else if row==&crate::region_state::INTERRUPT_RECORD{active=None;}
    }Ok(ended)
}

/// A user edit invalidates every older permanent end of the touched current-
/// stage cells. Interrupts preserve all former visit contents without claiming
/// completion; the next explicit begin is still required by the token grammar.
fn reopen_manual_regions(raw:&mut Vec<RawRecord>,touched:&[(usize,usize)])->Result<(),String>{
    let Some(&(side,preferred))=touched.last()else{return Ok(())};
    if !matches!(side,4|8)||touched.iter().any(|&(s,index)|s!=side||index>=side*side){return Err("用户复查区域阶段无效".into());}
    let targets:HashSet<_>=touched.iter().copied().collect();let mut active=None;
    for row in raw.iter_mut(){
        if crate::region_state::is_marker(row){let(stage,index)=crate::region_state::marker_identity(row)?;active=Some((stage.side(),index));}
        else if row==&crate::region_state::END_RECORD{
            let key=active.take().ok_or("用户区域的结束控制无效")?;
            if targets.contains(&key){*row=crate::region_state::INTERRUPT_RECORD;}
        }else if row==&crate::region_state::INTERRUPT_RECORD{active=None;}
    }
    let state=DrawingRegion::from_history(raw)?;
    if state.side==side&&state.index==Some(preferred){return Ok(());}
    if state.index.is_some(){raw.push(crate::region_state::INTERRUPT_RECORD);}
    let stage=if side==4{Stage::Coarse}else{Stage::Fine};
    raw.push(crate::region_state::marker_record(stage,preferred)?);
    DrawingRegion::from_history(raw)?;Ok(())
}

fn canvas_records(doc: &Document, node: NodeId) -> Result<Vec<RawRecord>, String> {
    let canvas = doc.materialize(node)?;
    let mut records = Vec::with_capacity(canvas.len() + 1);
    records.push(BOS_RECORD);
    for primitive in canvas {
        records.push(primitive.to_tokens()?.map(u32::from));
    }
    Ok(records)
}

fn verify_visible(doc: &Document, node: NodeId, records: &[RawRecord]) -> Result<(), String> {
    verify_projection(&doc.materialize(node)?, records)
}

fn verify_projection_ids(canvas:&[Primitive],records:&[RawRecord],ids:&[u64])->Result<(),String>{
    if records.first()!=Some(&BOS_RECORD){return Err("模型上下文缺少规范BOS".into());}
    let by_id=canvas.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();let mut seen=HashSet::new();let mut cursor=0;
    for row in records{if is_visible(row){let id=*ids.get(cursor).ok_or("模型输入缺少显式图元ID")?;
        if !seen.insert(id){return Err("模型输入重复引用同一个图元ID".into());}
        let primitive=by_id.get(&id).ok_or("模型输入ID不在当前完整画布中")?;
        if primitive.to_tokens()?.map(u32::from)!=*row{return Err("模型输入ID对应的九属性与画布不一致".into());}cursor+=1;
    }}
    if cursor!=ids.len(){return Err("模型输入显式ID与可见记录数量不一致".into());}
    DrawingRegion::from_history(records)?;Ok(())
}
fn clean_empty_interrupts(records:&[RawRecord])->Vec<RawRecord>{
    let mut clean=Vec::with_capacity(records.len());
    for row in records{if row==&crate::region_state::INTERRUPT_RECORD&&clean.last().is_some_and(crate::region_state::is_marker){clean.pop();}
        else{clean.push(*row);}}clean
}
fn raw_phase(records:&[RawRecord],evidence_only:bool)->Result<usize,String>{
    let clean=clean_empty_interrupts(records);let mut current=1;let mut maximum=1;
    for row in &clean{if crate::region_state::is_marker(row){current=crate::region_state::marker_identity(row)?.0.side();if !evidence_only{maximum=maximum.max(current);}}
        else if is_visible(row)||row==&crate::region_state::END_RECORD{maximum=maximum.max(current);}}
    Ok(maximum)
}
fn projection_assignments_match(raw:&[RawRecord],ids:&[u64],view:&stage_edit::StageView)->bool{
    let mut current=(1,0);let mut cursor=0;
    for row in raw{if crate::region_state::is_marker(row){let Ok((stage,index))=crate::region_state::marker_identity(row)else{return false};current=(stage.side(),index);}
        else if is_visible(row){let Some(id)=ids.get(cursor)else{return false};let Some(a)=view.assignments.get(id)else{return false};
            if (a.stage.side(),a.cell)!=current{return false;}cursor+=1;}}
    cursor==ids.len()
}
fn canonical_history(canvas:&[Primitive],view:&stage_edit::StageView,floor:usize,old_raw:&[RawRecord],pending:&[(usize,usize)])
    ->Result<EffectiveHistory,String>{
    if !matches!(floor,1|4|8){return Err("模型阶段游标无效".into());}
    let by_id=canvas.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();let mut records=vec![BOS_RECORD];let mut ids=Vec::new();
    let mut groups:BTreeMap<(usize,usize),Vec<u64>>=BTreeMap::new();
    for id in &view.admitted_ids{let a=view.assignments[id];if a.stage.side()>floor{continue;}
        if a.stage==StageTier::Global{records.push(by_id[id].to_tokens()?.map(u32::from));ids.push(*id);}
        else{groups.entry((a.stage.side(),a.cell)).or_default().push(*id);}}
    let mut closed=HashSet::new();let mut active=None;
    for row in clean_empty_interrupts(old_raw){if crate::region_state::is_marker(&row){let(stage,index)=crate::region_state::marker_identity(&row)?;
            active=Some((stage.side(),index));closed.remove(&(stage.side(),index));}
        else if row==crate::region_state::END_RECORD{if let Some(key)=active.take(){closed.insert(key);}}
        else if row==crate::region_state::INTERRUPT_RECORD{active=None;}}
    for &key in &closed{if key.0<=floor{groups.entry(key).or_default();}}
    for &key in pending{if key.0==floor{groups.entry(key).or_default();closed.remove(&key);}}
    if let Some(key)=active{if key.0==floor{groups.entry(key).or_default();}}
    let preferred=pending.iter().rev().find(|(side,_)|*side==floor).copied()
        .or_else(||active.filter(|key|key.0==floor))
        .or_else(||groups.keys().rev().copied().find(|key|key.0==floor&&!closed.contains(key)));
    let mut ordered=groups.keys().copied().filter(|key|Some(*key)!=preferred).collect::<Vec<_>>();
    if let Some(key)=preferred{ordered.push(key);}
    // Stage ordering is canonical. An active highest-stage cell is last; all
    // true zero-shape END visits remain represented, unlike empty interrupts.
    ordered.sort_by_key(|key|(key.0,Some(*key)==preferred));
    for key in ordered{let stage=if key.0==4{Stage::Coarse}else{Stage::Fine};records.push(crate::region_state::marker_record(stage,key.1)?);
        for id in &groups[&key]{records.push(by_id[id].to_tokens()?.map(u32::from));ids.push(*id);}
        if Some(key)==preferred{continue;}
        if key.0<floor||closed.contains(&key){records.push(crate::region_state::END_RECORD);}else{records.push(crate::region_state::INTERRUPT_RECORD);}
    }
    let records=clean_empty_interrupts(&records);verify_projection_ids(canvas,&records,&ids)?;
    Ok(EffectiveHistory{records,primitive_ids:ids})
}

fn verify_projection(canvas: &[Primitive], records: &[RawRecord]) -> Result<(), String> {
    if records.first() != Some(&BOS_RECORD) {
        return Err("生成上下文必须以唯一图元 BOS 开始".into());
    }
    let mut visible = Vec::new();
    for (index, row) in records.iter().enumerate() {
        if index == 0 {
            continue;
        }
        if row[0] == BOS {
            return Err("生成上下文包含重复 BOS".into());
        }
        if [TRUE_END, INTERRUPT, PAD].contains(&row[0]) {
            if row[1..].iter().any(|&v| v != PAD) {
                return Err("控制记录格式不完整".into());
            }
            continue;
        }
        let marker_components = row[6..].iter().filter(|&&v| v == REGION_MARKER).count();
        if marker_components != 0 {
            if marker_components != 3 || row[5] != 6417 {
                return Err("区域 marker 格式不完整".into());
            }
            continue;
        }
        let mut tokens = [0u16; 9];
        for i in 0..9 {
            tokens[i] = row[i].try_into().map_err(|_| "图元 token 超出原生词表")?;
        }
        Primitive::from_tokens(0, tokens)?;
        visible.push(*row);
    }
    let expected = canvas
        .iter()
        .map(|p| p.to_tokens().map(|r| r.map(u32::from)))
        .collect::<Result<Vec<_>, _>>()?;
    if visible != expected {
        return Err("生成上下文的可见图元与状态画布不一致，拒绝附加过期记录".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{OperationSource, ShapeKind};

    fn p(x: f32) -> Primitive {
        Primitive::new(ShapeKind::Ellipse, x, 50.0, 20.0, 10.0, [100, 150, 200])
    }
    fn marker() -> RawRecord {
        crate::region_state::marker_record(Stage::Coarse, 0).unwrap()
    }
    fn end() -> RawRecord {
        [TRUE_END, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD]
    }

    #[test]
    fn subtree_transaction_remaps_surviving_raw_deltas_and_exact_control_overlays() {
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let bg=doc.commit_generated(Primitive::background([20,30,40])).unwrap();
        let bgraw=canvas_records(&doc,bg).unwrap();store.attach(&doc,bg,&bgraw).unwrap();
        let origin=doc.commit_generated(p(80.)).unwrap();
        let mut original=bgraw;original.push(marker());original.push(doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from));
        store.attach(&doc,origin,&original).unwrap();
        let deleted=doc.commit_generated(p(90.)).unwrap();let mut deleted_raw=original.clone();
        deleted_raw.push(doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from));store.attach(&doc,deleted,&deleted_raw).unwrap();
        let hidden=doc.commit_generated(p(100.)).unwrap();deleted_raw.push(doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from));
        store.attach(&doc,hidden,&deleted_raw).unwrap();
        doc.jump(origin).unwrap();let keep=doc.commit_generated(p(110.)).unwrap();let mut keep_raw=original.clone();
        keep_raw.push(doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from));store.attach(&doc,keep,&keep_raw).unwrap();
        let keep_child=doc.commit_generated(p(120.)).unwrap();let mut child_raw=keep_raw.clone();
        child_raw.push(doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from));store.attach(&doc,keep_child,&child_raw).unwrap();
        let mut origin_overlay=original.clone();origin_overlay.extend([end(),crate::region_state::marker_record(Stage::Coarse,1).unwrap()]);
        store.set_control_overlay(&doc,origin,&origin_overlay).unwrap();
        let mut child_overlay=child_raw.clone();child_overlay.push(end());store.set_control_overlay(&doc,keep_child,&child_overlay).unwrap();
        let surviving=[bg,origin,keep,keep_child].into_iter().map(|node|
            (node,doc.materialize(node).unwrap(),store.context_for(&doc,node).unwrap(),store.resolve(&doc,HistoryBase::Node(node)).unwrap())).collect::<Vec<_>>();
        let count_before=store.stored_record_count();doc.prompt="保留的输入草稿".into();doc.settings.cfg=3.;
        let plan=doc.plan_delete_subtrees(&[deleted,hidden,deleted]).unwrap();
        let result=store.apply_delete_subtrees(&mut doc,plan).unwrap();
        assert_eq!(doc.nodes.len(),5);assert!(!result.current_was_deleted);
        assert_eq!(doc.prompt,"保留的输入草稿");assert_eq!(doc.settings.cfg,3.);
        for(old,canvas,public,immutable)in surviving{
            let node=result.old_to_new[old].unwrap();assert_eq!(doc.materialize(node).unwrap(),canvas);
            assert_eq!(store.context_for(&doc,node).unwrap(),public);
            assert_eq!(store.resolve(&doc,HistoryBase::Node(node)).unwrap(),immutable);
        }
        assert!(store.stored_record_count()<count_before);
        let new_keep=result.old_to_new[keep].unwrap();let new_child=result.old_to_new[keep_child].unwrap();
        assert!(matches!(store.entries[&new_child].base,HistoryBase::Node(base) if base==new_keep));
        let new_origin=result.old_to_new[origin].unwrap();doc.jump(new_origin).unwrap();
        let future=doc.commit_generated(p(160.)).unwrap();let mut future_raw=origin_overlay;
        future_raw.push(doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from));
        store.attach(&doc,future,&future_raw).unwrap();assert_eq!(store.context_for(&doc,future).unwrap(),Some(future_raw));
        assert_eq!(store.context_for(&doc,new_child).unwrap(),Some(child_overlay));
    }

    #[test]
    fn canvas_based_contexts_are_remapped_and_current_deletion_falls_back_atomically() {
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let deleted=doc.commit_generated(p(20.)).unwrap();doc.jump(0).unwrap();
        let manual=doc.commit_canvas("manual base",OperationSource::User,vec![Primitive::background([40,80,120])]).unwrap();
        let model=doc.commit_generated(p(100.)).unwrap();let raw=canvas_records(&doc,model).unwrap();store.attach(&doc,model,&raw).unwrap();
        assert!(matches!(store.entries[&model].base,HistoryBase::Canvas(base) if base==manual));
        let plan=doc.plan_delete_subtrees(&[deleted]).unwrap();
        let result=store.apply_delete_subtrees(&mut doc,plan).unwrap();
        let new_manual=result.old_to_new[manual].unwrap();let new_model=result.old_to_new[model].unwrap();
        assert_eq!(new_manual,1);assert!(matches!(store.entries[&new_model].base,HistoryBase::Canvas(base) if base==new_manual));
        assert_eq!(store.context_for(&doc,new_model).unwrap(),Some(raw));
        doc.jump(new_model).unwrap();let plan=doc.plan_delete_subtrees(&[new_model]).unwrap();
        let result=store.apply_delete_subtrees(&mut doc,plan).unwrap();
        assert!(result.current_was_deleted);assert_eq!(doc.current,new_manual);
        assert_eq!(doc.prompt,doc.current_node().prompt);
        assert!(store.entries.is_empty());assert!(store.context_for(&doc,new_manual).unwrap().is_none());
    }

    #[test]
    fn bad_surviving_history_or_stale_plan_never_partially_deletes_document() {
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let bg=doc.commit_generated(Primitive::background([1,2,3])).unwrap();let raw=canvas_records(&doc,bg).unwrap();store.attach(&doc,bg,&raw).unwrap();
        let keep=doc.commit_generated(p(20.)).unwrap();let keep_raw=canvas_records(&doc,keep).unwrap();store.attach(&doc,keep,&keep_raw).unwrap();
        doc.jump(bg).unwrap();let deleted=doc.commit_generated(p(30.)).unwrap();let deleted_raw=canvas_records(&doc,deleted).unwrap();store.attach(&doc,deleted,&deleted_raw).unwrap();
        let valid_base=store.entries[&keep].base;
        for invalid_base in [HistoryBase::Node(deleted),HistoryBase::Node(keep),HistoryBase::Canvas(usize::MAX)]{
            store.entries.get_mut(&keep).unwrap().base=invalid_base;
            let before=serde_json::to_value(&doc.nodes).unwrap();let count=store.stored_record_count();
            let plan=doc.plan_delete_subtrees(&[deleted]).unwrap();assert!(store.apply_delete_subtrees(&mut doc,plan).is_err());
            assert_eq!(serde_json::to_value(&doc.nodes).unwrap(),before);assert_eq!(store.stored_record_count(),count);
            assert_eq!(store.entries[&keep].base,invalid_base);
        }
        store.entries.get_mut(&keep).unwrap().base=valid_base;let valid_keep=store.entries[&keep].keep;
        store.entries.get_mut(&keep).unwrap().keep=usize::MAX;
        let before=serde_json::to_value(&doc.nodes).unwrap();let plan=doc.plan_delete_subtrees(&[deleted]).unwrap();
        assert!(store.apply_delete_subtrees(&mut doc,plan).is_err());assert_eq!(serde_json::to_value(&doc.nodes).unwrap(),before);
        store.entries.get_mut(&keep).unwrap().keep=valid_keep;
        let plan=doc.plan_delete_subtrees(&[deleted]).unwrap();doc.nodes[keep].title.push_str(" changed");
        let before=serde_json::to_value(&doc.nodes).unwrap();assert!(store.apply_delete_subtrees(&mut doc,plan).is_err());
        assert_eq!(serde_json::to_value(&doc.nodes).unwrap(),before);assert_eq!(store.context_for(&doc,keep).unwrap(),Some(keep_raw));
    }

    #[test]
    fn root_deletion_clears_all_model_history_and_new_generation_is_attachable() {
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let bg=doc.commit_generated(Primitive::background([40,50,60])).unwrap();let raw=canvas_records(&doc,bg).unwrap();store.attach(&doc,bg,&raw).unwrap();
        let mut overlay=raw;overlay.push(marker());store.set_control_overlay(&doc,bg,&overlay).unwrap();
        let max_id=doc.primitives[0].id;
        let plan=doc.plan_delete_subtrees(&[0]).unwrap();let result=store.apply_delete_subtrees(&mut doc,plan).unwrap();
        assert!(result.root_reset);assert!(store.entries.is_empty()&&store.control_overlays.is_empty());
        assert_eq!(store.stored_record_count(),0);assert!(store.context_for(&doc,0).unwrap().is_none());
        let next=doc.commit_generated(Primitive::background([100,120,140])).unwrap();assert!(doc.primitives[0].id>max_id);
        let next_raw=canvas_records(&doc,next).unwrap();store.attach(&doc,next,&next_raw).unwrap();
        assert_eq!(store.context_for(&doc,next).unwrap(),Some(next_raw));
    }

    #[test]
    fn region_control_records_are_retained_as_small_deltas() {
        let mut doc = Document::new();
        let mut store = RawHistoryStore::new();
        let bg = doc
            .commit_generated(Primitive::background([20, 30, 40]))
            .unwrap();
        let mut raw = canvas_records(&doc, bg).unwrap();
        store.attach(&doc, bg, &raw).unwrap();
        let node = doc.commit_generated(p(80.0)).unwrap();
        raw.push(marker());
        raw.push(
            doc.primitives
                .last()
                .unwrap()
                .to_tokens()
                .unwrap()
                .map(u32::from),
        );
        store.attach(&doc, node, &raw).unwrap();
        assert_eq!(store.context_for(&doc, node).unwrap(), Some(raw.clone()));
        // BOS is reconstructed from the parent canvas; no full prefix copies.
        assert_eq!(store.stored_record_count(), 3);
        raw.push(end());
        store.finish(&doc, node, &raw).unwrap();
        assert_eq!(store.context_for(&doc, node).unwrap(), Some(raw));
        assert_eq!(store.stored_record_count(), 4);
    }

    #[test]
    fn manual_branch_has_no_stale_context_and_next_generation_uses_edited_canvas() {
        let mut doc = Document::new();
        let mut store = RawHistoryStore::new();
        let first = doc.commit_generated(p(80.0)).unwrap();
        let raw_first = canvas_records(&doc, first).unwrap();
        store.attach(&doc, first, &raw_first).unwrap();
        let mut edited = doc.primitives.clone();
        edited[0].color = [40, 60, 80];
        let manual = doc
            .commit_canvas("修改颜色", OperationSource::User, edited)
            .unwrap();
        assert!(store.context_for(&doc, manual).unwrap().is_none());
        let mut new_raw = canvas_records(&doc, manual).unwrap();
        // A validated fresh request can attach control state without adding a
        // visible node. It cannot silently inherit the stale parent colors.
        assert!(store.finish(&doc, manual, &raw_first).is_err());
        store.finish(&doc, manual, &new_raw).unwrap();
        assert_eq!(
            store.context_for(&doc, manual).unwrap(),
            Some(new_raw.clone())
        );
        let generated = doc.commit_generated(p(120.0)).unwrap();
        new_raw.push(
            doc.primitives
                .last()
                .unwrap()
                .to_tokens()
                .unwrap()
                .map(u32::from),
        );
        store.attach(&doc, generated, &new_raw).unwrap();
        assert_eq!(store.context_for(&doc, generated).unwrap(), Some(new_raw));
        assert_eq!(store.context_for(&doc, first).unwrap(), Some(raw_first));
    }

    #[test]
    fn finish_keeps_original_base_and_does_not_change_existing_child_prefix() {
        let mut doc = Document::new();
        let mut store = RawHistoryStore::new();
        let first = doc.commit_generated(p(80.0)).unwrap();
        let mut raw_first = vec![
            BOS_RECORD,
            marker(),
            doc.primitives[0].to_tokens().unwrap().map(u32::from),
        ];
        store.attach(&doc, first, &raw_first).unwrap();
        let second = doc.commit_generated(p(120.0)).unwrap();
        let mut raw_second = raw_first.clone();
        raw_second.push(doc.primitives[1].to_tokens().unwrap().map(u32::from));
        store.attach(&doc, second, &raw_second).unwrap();
        raw_first.push(end());
        store.finish(&doc, first, &raw_first).unwrap();
        assert_eq!(store.context_for(&doc, first).unwrap(), Some(raw_first));
        assert_eq!(store.context_for(&doc, second).unwrap(), Some(raw_second));
        // Cannot replace a prior model record or introduce a self-reference.
        let old = store.context_for(&doc, second).unwrap().unwrap();
        assert!(store.finish(&doc, second, &old[..1]).is_err());
        assert_eq!(store.context_for(&doc, second).unwrap(), Some(old));
    }

    #[test]
    fn mismatched_callback_is_rejected_without_poisoning_other_branch() {
        let mut doc = Document::new();
        let mut store = RawHistoryStore::new();
        let first = doc.commit_generated(p(80.0)).unwrap();
        let raw_first = canvas_records(&doc, first).unwrap();
        store.attach(&doc, first, &raw_first).unwrap();
        doc.jump(0).unwrap();
        let sibling = doc.commit_generated(p(200.0)).unwrap();
        assert!(store.attach(&doc, sibling, &raw_first).is_err());
        assert!(store.context_for(&doc, sibling).unwrap().is_none());
        let raw_sibling = canvas_records(&doc, sibling).unwrap();
        store.attach(&doc, sibling, &raw_sibling).unwrap();
        assert_eq!(store.context_for(&doc, first).unwrap(), Some(raw_first));
        assert_eq!(store.context_for(&doc, sibling).unwrap(), Some(raw_sibling));
    }

    fn at(x: f32, y: f32) -> Primitive {
        let mut value = p(x);
        value.y = y;
        value
    }

    fn regional_fixture(stage: Stage, index: usize) -> (Document, RawHistoryStore, NodeId) {
        let mut doc = Document::new();
        let mut canvas = vec![Primitive::background([20, 40, 60])];
        canvas.extend((0..85).map(|i| at(20.0 + i as f32, 30.0)));
        let side = stage.side();
        let cell = 512.0 / side as f32;
        canvas.push(at(
            (index % side) as f32 * cell + 32.0,
            (index / side) as f32 * cell + 32.0,
        ));
        let node = doc
            .commit_canvas("人工区域fixture", OperationSource::User, canvas)
            .unwrap();
        let mut raw = vec![BOS_RECORD];
        raw.extend(
            doc.primitives[..86]
                .iter()
                .map(|p| p.to_tokens().unwrap().map(u32::from)),
        );
        raw.push(crate::region_state::marker_record(stage, index).unwrap());
        raw.push(doc.primitives[86].to_tokens().unwrap().map(u32::from));
        let mut store = RawHistoryStore::new();
        store.finish(&doc, node, &raw).unwrap();
        (doc, store, node)
    }

    #[test]
    fn manual_color_revokes_old_permanent_end_and_only_new_model_end_consumes_review(){
        let(mut doc,mut store,parent)=regional_fixture(Stage::Fine,0);
        let mut original=store.context_for(&doc,parent).unwrap().unwrap();original.push(end());store.finish(&doc,parent,&original).unwrap();
        let old_canvas=doc.primitives.clone();let mut edited=old_canvas.clone();edited.last_mut().unwrap().color=[80,120,160];
        let manual=doc.commit_canvas("change closed fine color",OperationSource::User,edited).unwrap();
        let region=store.reconcile_manual(&doc,parent,manual).unwrap();let raw=store.context_for(&doc,manual).unwrap().unwrap();
        assert_eq!(region.stage,Stage::Fine);assert_eq!(region.index,Some(0));
        assert_eq!(store.manual_review_regions(&doc,manual).unwrap(),[(8,0)]);
        assert!(!permanent_end_regions(&raw).unwrap().contains(&(8,0)));
        assert!(raw.contains(&crate::region_state::INTERRUPT_RECORD));
        assert_eq!(raw.iter().filter(|r|is_visible(r)).count(),doc.primitives.len());
        assert_eq!(store.context_for(&doc,parent).unwrap(),Some(original));
        let mut suspended=raw.clone();suspended.extend([crate::region_state::INTERRUPT_RECORD,
            crate::region_state::marker_record(Stage::Fine,1).unwrap()]);
        store.finish(&doc,manual,&suspended).unwrap();assert_eq!(store.manual_review_regions(&doc,manual).unwrap(),[(8,0)]);
        let mut reviewed=suspended;reviewed.extend([crate::region_state::INTERRUPT_RECORD,
            crate::region_state::marker_record(Stage::Fine,0).unwrap(),end()]);
        store.finish(&doc,manual,&reviewed).unwrap();assert!(store.manual_review_regions(&doc,manual).unwrap().is_empty());
    }

    #[test]
    fn overfilled_manual_block_is_retained_and_pending_snapshots_are_independent(){
        let(mut doc,mut store,parent)=regional_fixture(Stage::Coarse,0);
        let mut edited=doc.primitives.clone();for _ in 0..25{edited.push(at(60.,60.));}
        let manual=doc.commit_canvas("manual beyond coarse cap",OperationSource::User,edited).unwrap();
        store.reconcile_manual(&doc,parent,manual).unwrap();let raw=store.context_for(&doc,manual).unwrap().unwrap();
        assert_eq!(DrawingRegion::from_history(&raw).unwrap().generated_in_region,26);
        assert_eq!(store.manual_review_regions(&doc,manual).unwrap(),[(4,0)]);
        let child=doc.commit_generated(at(70.,70.)).unwrap();let mut child_raw=raw.clone();child_raw.push(row_for_test(&doc));
        store.attach(&doc,child,&child_raw).unwrap();assert_eq!(store.manual_review_regions(&doc,child).unwrap(),[(4,0)]);
        let mut ended=raw;ended.push(end());store.finish(&doc,manual,&ended).unwrap();
        assert!(store.manual_review_regions(&doc,manual).unwrap().is_empty());
        assert_eq!(store.manual_review_regions(&doc,child).unwrap(),[(4,0)]);
        assert_eq!(store.context_for(&doc,child).unwrap(),Some(child_raw));
    }

    #[test]
    fn parent_zero_output_end_before_late_sibling_does_not_clear_its_manual_review(){
        for overlay in [false,true]{
            let(mut doc,mut store,parent)=regional_fixture(Stage::Fine,0);
            let mut edited=doc.primitives.clone();edited.last_mut().unwrap().color=[80,120,160];
            let manual=doc.commit_canvas("parallel origin pending",OperationSource::User,edited).unwrap();
            store.reconcile_manual(&doc,parent,manual).unwrap();let private=store.context_for(&doc,manual).unwrap().unwrap();
            let mut zero_output=private.clone();zero_output.push(end());
            if overlay{store.set_control_overlay(&doc,manual,&zero_output).unwrap();}else{store.finish(&doc,manual,&zero_output).unwrap();}
            assert!(store.manual_review_regions(&doc,manual).unwrap().is_empty());
            assert_eq!(store.manual_reviews[&manual],[(8,0)],"creation snapshot must remain immutable");
            let late=doc.commit_generated(at(30.,30.)).unwrap();let mut late_raw=private.clone();late_raw.push(row_for_test(&doc));
            store.attach(&doc,late,&late_raw).unwrap();
            assert_eq!(store.manual_review_regions(&doc,late).unwrap(),[(8,0)]);
            assert_eq!(store.context_for(&doc,late).unwrap(),Some(late_raw));
            assert_eq!(store.context_for(&doc,manual).unwrap(),Some(zero_output));
            doc.jump(manual).unwrap();let ended_child=doc.commit_generated(at(35.,35.)).unwrap();
            let mut ended_raw=private;ended_raw.extend([row_for_test(&doc),end()]);store.attach(&doc,ended_child,&ended_raw).unwrap();
            assert!(store.manual_review_regions(&doc,ended_child).unwrap().is_empty());
            assert_eq!(store.manual_review_regions(&doc,late).unwrap(),[(8,0)]);
        }
    }

    fn row_for_test(doc:&Document)->RawRecord{doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from)}

    #[test]
    fn multiple_touched_cells_revoke_all_their_ends_and_delete_remaps_pending(){
        let(mut doc,mut store,parent)=regional_fixture(Stage::Fine,0);
        let obsolete=doc.commit_generated_at(0,Primitive::background([2,4,6]),String::new(),doc.settings.clone()).unwrap();
        let obsolete_raw=canvas_records(&doc,obsolete).unwrap();store.attach(&doc,obsolete,&obsolete_raw).unwrap();
        let mut completed=store.context_for(&doc,parent).unwrap().unwrap();completed.extend([end(),
            crate::region_state::marker_record(Stage::Fine,1).unwrap()]);
        let second=doc.commit_generated(at(90.,30.)).unwrap();completed.push(row_for_test(&doc));completed.push(end());store.attach(&doc,second,&completed).unwrap();
        let mut canvas=doc.primitives.clone();canvas[86].color=[60,100,140];canvas[87].color=[80,120,160];
        let manual=doc.commit_canvas("change both completed cells",OperationSource::User,canvas).unwrap();store.reconcile_manual(&doc,second,manual).unwrap();
        assert_eq!(store.manual_review_regions(&doc,manual).unwrap(),[(8,0),(8,1)]);
        let raw=store.context_for(&doc,manual).unwrap().unwrap();assert!(!permanent_end_regions(&raw).unwrap().contains(&(8,0)));
        assert!(!permanent_end_regions(&raw).unwrap().contains(&(8,1)));
        let plan=doc.plan_delete_subtrees(&[obsolete]).unwrap();
        let deleted=store.apply_delete_subtrees(&mut doc,plan).unwrap();let mapped=deleted.old_to_new[manual].unwrap();
        assert!(mapped<manual);
        assert_eq!(store.manual_review_regions(&doc,mapped).unwrap(),[(8,0),(8,1)]);
    }

    #[test]
    fn global_manual_inputs_are_read_as_prefix_without_forced_region_marker(){
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let first=doc.commit_canvas("manual global",OperationSource::User,vec![Primitive::background([20,40,60]),at(40.,40.)]).unwrap();
        store.reconcile_manual(&doc,0,first).unwrap();assert!(store.manual_review_regions(&doc,first).unwrap().is_empty());
        let raw=store.context_for(&doc,first).unwrap().unwrap();assert!(raw.iter().all(|r|!crate::region_state::is_marker(r)));
        assert_eq!(raw.iter().filter(|r|is_visible(r)).count(),2);
    }

    #[test]
    fn manual_append_outside_cell_is_exact_interrupt_begin_primitive_suffix() {
        let (mut doc, mut store, parent) = regional_fixture(Stage::Coarse, 0);
        let old = store.context_for(&doc, parent).unwrap().unwrap();
        let mut canvas = doc.primitives.clone();
        canvas.push(at(191.5, 191.5));
        let node = doc
            .commit_canvas("区域外新增", OperationSource::User, canvas)
            .unwrap();
        let region = store.reconcile_manual(&doc, parent, node).unwrap();
        let raw = store.context_for(&doc, node).unwrap().unwrap();
        assert_eq!(&raw[..old.len()], &old);
        assert_eq!(
            raw[old.len()],
            [INTERRUPT, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD]
        );
        assert_eq!(
            raw[old.len() + 1],
            crate::region_state::marker_record(Stage::Coarse, 5).unwrap()
        );
        assert_eq!(
            raw[old.len() + 2],
            doc.primitives
                .last()
                .unwrap()
                .to_tokens()
                .unwrap()
                .map(u32::from)
        );
        assert_eq!(raw.len(), old.len() + 3);
        assert_eq!(region.index, Some(5));
        assert_eq!(region.completed_coarse, 0);
        assert_eq!(
            raw.iter().filter(|r| is_visible(r)).count(),
            doc.primitives.len()
        );
    }

    #[test]
    fn same_cell_uses_center_not_contour_and_manual_over_cap_is_preserved() {
        let (mut doc, mut store, parent) = regional_fixture(Stage::Coarse, 0);
        let old = store.context_for(&doc, parent).unwrap().unwrap();
        let mut canvas = doc.primitives.clone();
        for _ in 0..20 {
            let mut shape = at(100.0, 100.0);
            shape.width = 500.0;
            shape.height = 500.0;
            canvas.push(shape);
        }
        let node = doc
            .commit_canvas("区域内多选添加", OperationSource::User, canvas)
            .unwrap();
        let region = store.reconcile_manual(&doc, parent, node).unwrap();
        let raw = store.context_for(&doc, node).unwrap().unwrap();
        assert_eq!(&raw[..old.len()], &old);
        assert_eq!(raw.len(), old.len() + 20);
        assert_eq!(region.index, Some(0));
        assert_eq!(region.generated_in_region, 21);
        assert!(!raw.iter().any(|r| r[0] == INTERRUPT));
    }

    #[test]
    fn moving_old_primitive_updates_it_once_and_only_appends_target_controls() {
        let (mut doc, mut store, parent) = regional_fixture(Stage::Coarse, 0);
        let old_raw = store.context_for(&doc, parent).unwrap().unwrap();
        let mut canvas = doc.primitives.clone();
        let moved = canvas.last_mut().unwrap();
        moved.x = 191.5;
        moved.y = 191.5;
        let id = moved.id;
        let node = doc
            .commit_canvas("移动旧图元", OperationSource::User, canvas)
            .unwrap();
        let region = store.reconcile_manual(&doc, parent, node).unwrap();
        let raw = store.context_for(&doc, node).unwrap().unwrap();
        assert_eq!(raw.len(), old_raw.len() + 2);
        assert_eq!(raw.iter().filter(|r| is_visible(r)).count(), 87);
        assert_eq!(
            raw[old_raw.len() - 1],
            doc.primitives
                .iter()
                .find(|p| p.id == id)
                .unwrap()
                .to_tokens()
                .unwrap()
                .map(u32::from)
        );
        assert_eq!(region.index, Some(5));
        assert_eq!(region.generated_in_region, 0);
    }

    #[test]
    fn color_delete_and_layer_reorder_preserve_controls_and_unique_visible_projection() {
        let (mut doc, mut store, parent) = regional_fixture(Stage::Coarse, 0);
        let old_raw = store.context_for(&doc, parent).unwrap().unwrap();
        let mut canvas = doc.primitives.clone();
        canvas.last_mut().unwrap().color = [40, 80, 120];
        let layer = canvas.remove(10);
        canvas.push(layer);
        canvas.remove(20);
        let node = doc
            .commit_canvas("改色删除排序", OperationSource::User, canvas)
            .unwrap();
        let region = store.reconcile_manual(&doc, parent, node).unwrap();
        let raw = store.context_for(&doc, node).unwrap().unwrap();
        assert_eq!(
            raw.iter()
                .filter(|r| !is_visible(r))
                .copied()
                .collect::<Vec<_>>(),
            old_raw
                .iter()
                .filter(|r| !is_visible(r))
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            raw.iter().filter(|r| is_visible(r)).count(),
            doc.primitives.len()
        );
        assert_eq!(region.index, Some(0));
    }

    #[test]
    fn multi_cell_append_and_half_pixel_gap_make_finite_single_pass_switches() {
        let (mut doc, mut store, parent) = regional_fixture(Stage::Coarse, 0);
        let mut canvas = doc.primitives.clone();
        canvas.extend([at(191.5, 191.5), at(319.5, 191.5)]);
        let node = doc
            .commit_canvas("跨两格新增", OperationSource::User, canvas)
            .unwrap();
        assert_eq!(
            store.reconcile_manual(&doc, parent, node).unwrap().index,
            Some(6)
        );
        let raw = store.context_for(&doc, node).unwrap().unwrap();
        assert_eq!(raw.iter().filter(|r| r[0] == INTERRUPT).count(), 2);
        doc.jump(parent).unwrap();
        let mut canvas = doc.primitives.clone();
        canvas.push(at(127.5, 50.0));
        let gap_node = doc
            .commit_canvas("半像素边界", OperationSource::User, canvas)
            .unwrap();
        assert_eq!(
            store
                .reconcile_manual(&doc, parent, gap_node)
                .unwrap()
                .index,
            Some(0)
        );
        let gap_raw = store.context_for(&doc, gap_node).unwrap().unwrap();
        assert_eq!(gap_raw.iter().filter(|r| r[0] == INTERRUPT).count(), 1);
        assert_eq!(doc.primitives.last().unwrap().x, 127.5);
    }

    #[test]
    fn fine_stage_stays_fine_and_explicit_edit_can_reopen_completed_cell() {
        let (mut doc, mut store, parent) = regional_fixture(Stage::Fine, 0);
        let mut canvas = doc.primitives.clone();
        canvas.push(at(191.5, 191.5));
        let fine_node = doc
            .commit_canvas("细区越界", OperationSource::User, canvas)
            .unwrap();
        let fine = store.reconcile_manual(&doc, parent, fine_node).unwrap();
        assert_eq!(fine.stage, Stage::Fine);
        assert_eq!(fine.index, Some(18));

        let (mut doc, mut store, parent) = regional_fixture(Stage::Coarse, 0);
        let mut raw = store.context_for(&doc, parent).unwrap().unwrap();
        raw.extend([
            end(),
            crate::region_state::marker_record(Stage::Coarse, 1).unwrap(),
        ]);
        let second = doc.commit_generated(at(191.5, 64.0)).unwrap();
        raw.push(
            doc.primitives
                .last()
                .unwrap()
                .to_tokens()
                .unwrap()
                .map(u32::from),
        );
        store.attach(&doc, second, &raw).unwrap();
        assert_eq!(
            DrawingRegion::from_history(&raw).unwrap().completed_coarse,
            1
        );
        let mut canvas = doc.primitives.clone();
        canvas.push(at(64.0, 64.0));
        let reopened = doc
            .commit_canvas("重开已完成区域", OperationSource::User, canvas)
            .unwrap();
        let region = store.reconcile_manual(&doc, second, reopened).unwrap();
        assert_eq!(region.index, Some(0));
        assert_eq!(region.completed_coarse, 0);
        assert_eq!(region.generated_in_region, 2);
        assert_eq!(store.manual_review_regions(&doc,reopened).unwrap(),[(4,0)]);
        assert!(!permanent_end_regions(&store.context_for(&doc,reopened).unwrap().unwrap()).unwrap().contains(&(4,0)));
    }

    #[test]
    fn fresh_zero_visible_region_start_is_saved_without_creating_image_node() {
        let mut doc = Document::new();
        let mut canvas = vec![Primitive::background([20, 40, 60])];
        canvas.extend((0..85).map(|i| at(i as f32, 30.0)));
        let node = doc
            .commit_canvas("全局画布", OperationSource::User, canvas)
            .unwrap();
        let mut raw = canvas_records(&doc, node).unwrap();
        raw.push(crate::region_state::marker_record(Stage::Coarse, 3).unwrap());
        let mut store = RawHistoryStore::new();
        let node_count = doc.nodes.len();
        store.finish(&doc, node, &raw).unwrap();
        assert_eq!(doc.nodes.len(), node_count);
        let restored = store.context_for(&doc, node).unwrap().unwrap();
        assert_eq!(restored, raw);
        let state = DrawingRegion::from_history(&restored).unwrap();
        assert_eq!(state.index, Some(3));
        assert_eq!(state.generated_in_region, 0);
    }

    #[test]
    fn control_overlay_preserves_existing_children_and_is_inherited_by_new_operations() {
        let (mut doc, mut store, origin) = regional_fixture(Stage::Coarse, 0);
        let original = store.context_for(&doc, origin).unwrap().unwrap();
        let original_canvas = doc.materialize(origin).unwrap();
        let original_entry = store.entries[&origin].clone();
        let mut children = Vec::new();
        for x in [70.0, 90.0] {
            doc.jump(origin).unwrap();
            let child = doc.commit_generated(at(x, 50.0)).unwrap();
            let mut raw = original.clone();
            raw.push(doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from));
            store.attach(&doc, child, &raw).unwrap();
            children.push((child, raw));
        }
        let nodes_before = doc.nodes.len();
        let selected_before = doc.current;
        let mut overlay = original.clone();
        overlay.push(end());
        for index in 1..16 {
            overlay.push(crate::region_state::marker_record(Stage::Coarse, index).unwrap());
            overlay.push(end());
        }
        store.set_control_overlay(&doc, origin, &overlay).unwrap();
        assert_eq!(store.context_for(&doc, origin).unwrap(), Some(overlay.clone()));
        assert_eq!(store.resolve(&doc, HistoryBase::Node(origin)).unwrap(), original);
        assert_eq!(store.entries[&origin].keep, original_entry.keep);
        assert_eq!(store.entries[&origin].tail, original_entry.tail);
        assert_eq!(doc.nodes.len(), nodes_before);
        assert_eq!(doc.current, selected_before);
        assert_eq!(doc.materialize(origin).unwrap(), original_canvas);
        for (child, raw) in &children {
            assert_eq!(store.resolve(&doc, HistoryBase::Node(*child)).unwrap(), *raw);
            assert_eq!(store.context_for(&doc, *child).unwrap(), Some(raw.clone()));
        }
        let region = DrawingRegion::from_history(&overlay).unwrap();
        assert_eq!(region.stage, Stage::Fine);
        assert!(region.awaiting_region);
        assert_eq!(region.completed_coarse, 16);

        // A historical Global color edit inherits public completion controls
        // without inventing a Fine visit or duplicating visible records.
        doc.jump(origin).unwrap();
        let mut canvas = doc.primitives.clone();
        canvas[1].color = [20, 40, 60];
        let colored = doc.commit_canvas("overlay后改色", OperationSource::User, canvas).unwrap();
        let colored_region=store.reconcile_manual(&doc, origin, colored).unwrap();
        assert_eq!(colored_region.stage,Stage::Fine);assert_eq!(colored_region.index,None);
        assert!(store.manual_review_regions(&doc,colored).unwrap().is_empty());
        let colored_raw = store.context_for(&doc, colored).unwrap().unwrap();
        let controls = |raw: &[RawRecord]| raw.iter().filter(|row| !is_visible(row)).copied().collect::<Vec<_>>();
        assert_eq!(controls(&colored_raw),controls(&overlay));

        let ordered = store.contribution_ordered_canvas(&doc, colored).unwrap();
        let sorted = doc.commit_canvas("overlay后贡献排序", OperationSource::User, ordered).unwrap();
        assert_ne!(sorted, colored);
        assert_eq!(store.reconcile_manual(&doc, colored, sorted).unwrap(), colored_region);
        assert_eq!(controls(&store.context_for(&doc, sorted).unwrap().unwrap()), controls(&colored_raw));

        // New model output is compressed against the immutable old entry;
        // its delta therefore carries the overlay suffix explicitly.
        doc.jump(origin).unwrap();
        let new_head = doc.commit_generated(at(160.0, 32.0)).unwrap();
        let mut generated_raw = overlay.clone();
        generated_raw.push(crate::region_state::marker_record(Stage::Fine, 2).unwrap());
        generated_raw.push(doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from));
        store.attach(&doc, new_head, &generated_raw).unwrap();
        assert_eq!(store.context_for(&doc, new_head).unwrap(), Some(generated_raw));
        assert_eq!(store.context_for(&doc, origin).unwrap(), Some(overlay));
        assert_eq!(store.resolve(&doc, HistoryBase::Node(origin)).unwrap(), original);
        for (child, raw) in children {
            assert_eq!(store.context_for(&doc, child).unwrap(), Some(raw));
        }
        assert_eq!(doc.materialize(origin).unwrap(), original_canvas);
    }

    #[test]
    fn control_overlay_without_an_entry_uses_the_canvas_baseline_and_survives_finish() {
        let mut doc = Document::new();
        let mut store = RawHistoryStore::new();
        let original_canvas = doc.primitives.clone();
        let child = doc.commit_generated(p(80.0)).unwrap();
        let child_raw = canvas_records(&doc, child).unwrap();
        store.attach(&doc, child, &child_raw).unwrap();
        let nodes_before = doc.nodes.len();
        let mut overlay = vec![BOS_RECORD, marker()];
        store.set_control_overlay(&doc, 0, &overlay).unwrap();
        assert!(!store.entries.contains_key(&0));
        assert_eq!(store.context_for(&doc, 0).unwrap(), Some(overlay.clone()));
        assert_eq!(store.context_for(&doc, child).unwrap(), Some(child_raw.clone()));
        overlay.push(end());
        store.finish(&doc, 0, &overlay).unwrap();
        assert!(!store.entries.contains_key(&0));
        assert_eq!(store.context_for(&doc, 0).unwrap(), Some(overlay));
        assert_eq!(store.context_for(&doc, child).unwrap(), Some(child_raw));
        assert_eq!(doc.nodes.len(), nodes_before);
        assert_eq!(doc.materialize(0).unwrap(), original_canvas);
        assert_eq!(store.resolve(&doc, HistoryBase::Canvas(0)).unwrap(), vec![BOS_RECORD]);
    }

    #[test]
    fn invalid_or_regressed_control_overlays_do_not_replace_public_or_immutable_contexts() {
        let (doc, mut store, origin) = regional_fixture(Stage::Coarse, 0);
        let original = store.context_for(&doc, origin).unwrap().unwrap();
        let mut overlay = original.clone();
        overlay.push(end());
        store.set_control_overlay(&doc, origin, &overlay).unwrap();
        let count_before = store.stored_record_count();
        let mut changed_visible = overlay.clone();
        changed_visible[1][6] += 1;
        let mut invalid_end = overlay.clone();
        invalid_end.push(end());
        let mut changed_controls = overlay.clone();
        let marker_index = original.iter().position(|row| row[6..] == [REGION_MARKER; 3]).unwrap();
        changed_controls[marker_index] = crate::region_state::marker_record(Stage::Coarse, 1).unwrap();
        let mut added_visible = overlay.clone();
        added_visible.push(p(80.0).to_tokens().unwrap().map(u32::from));
        for invalid in [original.clone(), changed_visible, changed_controls, invalid_end, added_visible] {
            assert!(store.set_control_overlay(&doc, origin, &invalid).is_err());
            assert!(store.finish(&doc, origin, &invalid).is_err());
            assert_eq!(store.context_for(&doc, origin).unwrap(), Some(overlay.clone()));
            assert_eq!(store.resolve(&doc, HistoryBase::Node(origin)).unwrap(), original);
            assert_eq!(store.stored_record_count(), count_before);
        }
        let mut extended = overlay.clone();
        extended.push(crate::region_state::marker_record(Stage::Coarse, 1).unwrap());
        store.finish(&doc, origin, &extended).unwrap();
        assert_eq!(store.context_for(&doc, origin).unwrap(), Some(extended.clone()));
        assert_eq!(store.resolve(&doc, HistoryBase::Node(origin)).unwrap(), original);
        assert!(store.set_control_overlay(&doc, origin, &overlay).is_err());
        assert_eq!(store.context_for(&doc, origin).unwrap(), Some(extended));
        assert!(store.set_control_overlay(&doc, doc.nodes.len(), &[]).is_err());
    }

    #[test]
    fn tree_jump_restores_each_manual_branches_region_without_changing_other_branch() {
        let (mut doc, mut store, parent) = regional_fixture(Stage::Coarse, 0);
        let mut canvas = doc.primitives.clone();
        canvas.push(at(191.5, 191.5));
        let a = doc
            .commit_canvas("分支A", OperationSource::User, canvas)
            .unwrap();
        store.reconcile_manual(&doc, parent, a).unwrap();
        let raw_a = store.context_for(&doc, a).unwrap().unwrap();
        doc.jump(parent).unwrap();
        let mut canvas = doc.primitives.clone();
        canvas.push(at(319.5, 191.5));
        let b = doc
            .commit_canvas("分支B", OperationSource::User, canvas)
            .unwrap();
        store.reconcile_manual(&doc, parent, b).unwrap();
        for (id, index) in [(a, 5), (parent, 0), (b, 6), (a, 5)] {
            doc.jump(id).unwrap();
            let raw = store.context_for(&doc, doc.current).unwrap().unwrap();
            assert_eq!(
                DrawingRegion::from_history(&raw).unwrap().index,
                Some(index)
            );
        }
        assert_eq!(store.context_for(&doc, a).unwrap(), Some(raw_a));
    }

    fn contribution_fixture(complete: bool) -> (Document, RawHistoryStore, NodeId, Vec<RawRecord>) {
        let mut doc = Document::new();
        let rectangle = |x, y, level| Primitive::new(ShapeKind::Rectangle, x, y, 10.0, 10.0, [level; 3]);
        let canvas = vec![
            Primitive::background([0, 0, 0]),
            rectangle(32.0, 32.0, 20), rectangle(32.0, 32.0, 80),
            rectangle(64.0, 64.0, 10), rectangle(64.0, 64.0, 100),
            rectangle(192.0, 64.0, 20), rectangle(192.0, 64.0, 200),
            rectangle(32.0, 32.0, 30), rectangle(32.0, 32.0, 220),
            rectangle(352.0, 32.0, 40), rectangle(352.0, 32.0, 252),
        ];
        let parent = doc.commit_canvas("跨阶段贡献排序测试", OperationSource::User, canvas).unwrap();
        let rows = doc.primitives.iter().map(|p| p.to_tokens().unwrap().map(u32::from)).collect::<Vec<_>>();
        let mut raw = vec![BOS_RECORD, rows[0], rows[1], rows[2]];
        for index in 0..16 {
            raw.push(crate::region_state::marker_record(Stage::Coarse, index).unwrap());
            if index == 0 { raw.extend_from_slice(&rows[3..5]); }
            if index == 1 { raw.extend_from_slice(&rows[5..7]); }
            raw.push(end());
        }
        if complete {
            for index in 0..64 {
                raw.push(crate::region_state::marker_record(Stage::Fine, index).unwrap());
                if index == 0 { raw.extend_from_slice(&rows[7..9]); }
                if index == 5 { raw.extend_from_slice(&rows[9..11]); }
                raw.push(end());
            }
        } else {
            raw.push(crate::region_state::marker_record(Stage::Fine, 0).unwrap());
            raw.extend_from_slice(&rows[7..9]);
            raw.push(end());
            raw.push(crate::region_state::marker_record(Stage::Fine, 5).unwrap());
            raw.extend_from_slice(&rows[9..11]);
        }
        let mut store = RawHistoryStore::new();
        store.finish(&doc, parent, &raw).unwrap();
        (doc, store, parent, raw)
    }

    #[test]
    fn contribution_reorder_preserves_raw_controls_region_progress_and_old_branch() {
        let (mut doc, mut store, parent, original_raw) = contribution_fixture(false);
        let original_canvas = doc.primitives.clone();
        let original_region = DrawingRegion::from_history(&original_raw).unwrap();
        let ordered = store.contribution_ordered_canvas(&doc, parent).unwrap();
        assert_eq!(ordered.iter().map(|p| p.id).collect::<Vec<_>>(), vec![1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]);
        // The strongest fine-cell shape remains in fine5, and cannot jump in
        // front of a weaker global or coarse shape from another region.
        assert_eq!(ordered.last().unwrap().id, 10);
        let node = doc.commit_canvas("按贡献重排序", OperationSource::User, ordered).unwrap();
        assert_ne!(node, parent);
        assert_eq!(doc.nodes[node].parent, Some(parent));
        assert!(doc.nodes[node].patches.iter().all(|patch| matches!(patch, Patch::Move { .. })));
        let region = store.reconcile_manual(&doc, parent, node).unwrap();
        let raw = store.context_for(&doc, node).unwrap().unwrap();
        assert_eq!(region, original_region);
        assert_eq!(raw.len(), original_raw.len());
        assert_eq!(raw.iter().filter(|row| !is_visible(row)).collect::<Vec<_>>(),
            original_raw.iter().filter(|row| !is_visible(row)).collect::<Vec<_>>());
        assert_eq!(raw.iter().filter(|row| is_visible(row)).copied().collect::<Vec<_>>(),
            doc.primitives.iter().map(|p| p.to_tokens().unwrap().map(u32::from)).collect::<Vec<_>>());
        assert_eq!(doc.primitives.iter().map(|p| p.id).collect::<HashSet<_>>().len(), original_canvas.len());
        assert_eq!(store.context_for(&doc, parent).unwrap(), Some(original_raw));
        assert_eq!(doc.materialize(parent).unwrap(), original_canvas);
        doc.jump(parent).unwrap();
        assert_eq!(doc.primitives, original_canvas);
        doc.jump(node).unwrap();
        assert_eq!(DrawingRegion::from_history(&store.context_for(&doc, node).unwrap().unwrap()).unwrap(), original_region);
    }

    #[test]
    fn contribution_reorder_finished_canvas_stays_complete_and_repeat_is_noop() {
        let (mut doc, mut store, parent, original_raw) = contribution_fixture(true);
        let original_region = DrawingRegion::from_history(&original_raw).unwrap();
        assert!(original_region.completed);
        let ordered = store.contribution_ordered_canvas(&doc, parent).unwrap();
        let node = doc.commit_canvas("已完成画布重排序", OperationSource::User, ordered).unwrap();
        let region = store.reconcile_manual(&doc, parent, node).unwrap();
        assert_eq!(region, original_region);
        assert_eq!(region.completed_coarse, 16);
        assert_eq!(region.completed_fine, 64);
        assert_eq!(region.generation_block_reason(), Some(crate::region_state::DRAWING_COMPLETE_REASON));
        let sorted_raw = store.context_for(&doc, node).unwrap().unwrap();
        let node_count = doc.nodes.len();
        let stored_records = store.stored_record_count();
        let ordered = store.contribution_ordered_canvas(&doc, node).unwrap();
        let same = doc.commit_canvas("重复重排序", OperationSource::User, ordered).unwrap();
        assert_eq!(same, node);
        assert_eq!(store.reconcile_manual(&doc, node, same).unwrap(), original_region);
        assert_eq!(doc.nodes.len(), node_count);
        assert_eq!(store.stored_record_count(), stored_records);
        assert_eq!(store.context_for(&doc, node).unwrap(), Some(sorted_raw));
        assert_eq!(store.context_for(&doc, parent).unwrap(), Some(original_raw));
    }

    #[test]
    fn same_cell_reentries_sort_together_without_crossing_the_other_cell_slots() {
        let mut doc = Document::new();
        let shape = |x, level| Primitive::new(ShapeKind::Rectangle, x, 32.0, 10.0, 10.0, [level; 3]);
        let parent = doc.commit_canvas("中断重入fixture", OperationSource::User, vec![
            Primitive::background([0; 3]), shape(32.0, 20), shape(192.0, 252), shape(32.0, 100),
        ]).unwrap();
        let rows = doc.primitives.iter().map(|p| p.to_tokens().unwrap().map(u32::from)).collect::<Vec<_>>();
        let raw = vec![
            BOS_RECORD, rows[0],
            crate::region_state::marker_record(Stage::Coarse, 0).unwrap(), rows[1],
            [INTERRUPT, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD],
            crate::region_state::marker_record(Stage::Coarse, 1).unwrap(), rows[2], end(),
            crate::region_state::marker_record(Stage::Coarse, 0).unwrap(), rows[3],
        ];
        let mut store = RawHistoryStore::new();
        store.finish(&doc, parent, &raw).unwrap();
        let before = DrawingRegion::from_history(&raw).unwrap();
        let ordered = store.contribution_ordered_canvas(&doc, parent).unwrap();
        assert_eq!(ordered.iter().map(|p| p.id).collect::<Vec<_>>(), vec![1, 4, 3, 2]);
        let node = doc.commit_canvas("原始区域内排序", OperationSource::User, ordered).unwrap();
        let after = store.reconcile_manual(&doc, parent, node).unwrap();
        assert_eq!(after, before);
        let reordered = store.context_for(&doc, node).unwrap().unwrap();
        assert_eq!(reordered.iter().filter(|r| !is_visible(r)).collect::<Vec<_>>(),
            raw.iter().filter(|r| !is_visible(r)).collect::<Vec<_>>());
    }

    #[test]
    fn absent_control_history_uses_global_bucket_and_exact_shape_area_contribution() {
        let mut doc = Document::new();
        let parent = doc.commit_canvas("面积贡献fixture", OperationSource::User, vec![
            Primitive::background([0; 3]),
            Primitive::new(ShapeKind::Ellipse, 32.0, 32.0, 20.0, 10.0, [100, 40, 20]),
            Primitive::new(ShapeKind::Rectangle, 32.0, 32.0, 20.0, 10.0, [100, 40, 20]),
        ]).unwrap();
        let store = RawHistoryStore::new();
        let ordered = store.contribution_ordered_canvas(&doc, parent).unwrap();
        assert_eq!(ordered.iter().map(|p| p.id).collect::<Vec<_>>(), vec![1, 3, 2]);
        let rectangle = crate::document::primitive_contribution(&ordered[1], [0; 3]).unwrap();
        let ellipse = crate::document::primitive_contribution(&ordered[2], [0; 3]).unwrap();
        assert_eq!(rectangle, (10000.0 + 1600.0 + 400.0) * 200.0);
        assert!((ellipse - rectangle * std::f64::consts::PI / 4.0).abs() < 0.000001);
        assert_eq!(doc.current, parent);
        assert!(store.context_for(&doc, parent).unwrap().is_none());
    }
}
