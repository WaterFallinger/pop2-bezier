//! Pure stage ownership and guarded capacity planning. The canvas is never
//! truncated; coarse overflow is a derived, reversible admission decision.
use crate::document::{NodeId,Primitive,PrimitiveStageAssignment,ShapeKind,StageTier};
use crate::region_state::{DrawingRegion,Stage};
use std::collections::{BTreeMap,HashMap,HashSet};

#[derive(Clone,Debug)]
pub struct StageAssignmentRequest{pub ids:Vec<u64>,pub stage:Stage}

#[derive(Debug)]
pub enum StageEditBlock{
    GlobalOverflow{candidate_canvas:Vec<Primitive>,message:String},
    Rejected(String),
}
impl StageEditBlock{
    pub fn message(&self)->&str{match self{Self::GlobalOverflow{message,..}|Self::Rejected(message)=>message}}
    pub fn candidate_canvas(self)->Option<Vec<Primitive>>{match self{Self::GlobalOverflow{candidate_canvas,..}=>Some(candidate_canvas),_=>None}}
}
impl std::fmt::Display for StageEditBlock{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.write_str(self.message())}}
impl std::error::Error for StageEditBlock{}
impl From<String> for StageEditBlock{fn from(value:String)->Self{Self::Rejected(value)}}
impl From<&str> for StageEditBlock{fn from(value:&str)->Self{Self::Rejected(value.into())}}

#[derive(Clone,Debug)]
pub struct StageEditPlan{
    pub parent:NodeId,pub canvas:Vec<Primitive>,pub raw:Vec<[u32;9]>,pub effective_ids:Vec<u64>,
    pub deferred_ids:Vec<u64>,pub notices:Vec<String>,pub manual_review_regions:Vec<(usize,usize)>,
    pub phase_floor:usize,
}

#[derive(Clone,Debug,Default)]
pub struct StageView{
    pub stage_by_id:BTreeMap<u64,Stage>,
    pub assignments:BTreeMap<u64,PrimitiveStageAssignment>,
    pub admitted_ids:Vec<u64>,pub deferred_ids:Vec<u64>,
}
pub fn tier(stage:Stage)->Result<StageTier,String>{match stage{Stage::Global=>Ok(StageTier::Global),Stage::Coarse=>Ok(StageTier::Coarse),
    Stage::Fine=>Ok(StageTier::Fine),Stage::Complete=>Err("完成状态不是图元归属阶段".into())}}
pub fn stage(tier:StageTier)->Stage{match tier{StageTier::Global=>Stage::Global,StageTier::Coarse=>Stage::Coarse,StageTier::Fine=>Stage::Fine}}
pub fn cell_for(stage:StageTier,p:&Primitive)->Result<usize,String>{
    if stage==StageTier::Global{return Ok(0);}
    DrawingRegion::cell_for_center(self::stage(stage),p.x,p.y).ok_or("无法确定图元所属网格".into())
}
fn assignment(p:&Primitive,legacy:&BTreeMap<u64,Stage>)->Result<PrimitiveStageAssignment,String>{
    if p.shape==ShapeKind::Background{return Ok(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:p.stage_assignment.map_or(p.id,|a|a.queue_order)});}
    if let Some(a)=p.stage_assignment{a.validate()?;return Ok(a);}
    let stage=tier(legacy.get(&p.id).copied().unwrap_or(Stage::Global))?;
    Ok(PrimitiveStageAssignment{stage,cell:cell_for(stage,p)?,queue_order:p.id})
}
pub fn view(canvas:&[Primitive],legacy:&BTreeMap<u64,Stage>,guard:bool)->Result<StageView,String>{
    let mut result=StageView::default();let mut coarse:BTreeMap<usize,Vec<(u64,u64)>>=BTreeMap::new();
    for p in canvas{let a=assignment(p,legacy)?;result.stage_by_id.insert(p.id,stage(a.stage));result.assignments.insert(p.id,a);
        if a.stage==StageTier::Coarse&&p.shape!=ShapeKind::Background{coarse.entry(a.cell).or_default().push((a.queue_order,p.id));}}
    let mut deferred=HashSet::new();if guard{for values in coarse.values_mut(){values.sort_unstable();deferred.extend(values.iter().skip(20).map(|v|v.1));}}
    let mut admitted=canvas.iter().filter(|p|!deferred.contains(&p.id)).collect::<Vec<_>>();
    // Stage/cell grouping is independent of canvas stacking; keep stacking order
    // stable inside each group and carry IDs explicitly beside model records.
    admitted.sort_by_key(|p|{let a=result.assignments[&p.id];(a.stage,a.cell)});
    result.admitted_ids=admitted.into_iter().map(|p|p.id).collect();
    result.deferred_ids=canvas.iter().filter(|p|deferred.contains(&p.id)).map(|p|p.id).collect();Ok(result)
}

pub fn prepare_canvas(existing:&[Primitive],mut proposed:Vec<Primitive>,legacy:&BTreeMap<u64,Stage>,new_stage:Stage,
    requested:Option<&StageAssignmentRequest>,guard:bool,global_confirmed:bool)->Result<Vec<Primitive>,StageEditBlock>{
    let old=existing.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();
    let selected=requested.map(|r|r.ids.iter().copied().collect::<HashSet<_>>()).unwrap_or_default();
    if selected.iter().any(|id|!old.contains_key(id)){return Err(StageEditBlock::Rejected("阶段归属选择含不存在的图元".into()));}
    let requested_stage=requested.map(|r|tier(r.stage)).transpose()?;let new_stage=tier(new_stage)?;
    let before=view(existing,legacy,guard)?;
    let mut ticket=before.assignments.values().map(|a|a.queue_order).max().unwrap_or(0);
    let mut ids=HashSet::new();
    for p in &mut proposed{
        if p.id==0||!ids.insert(p.id){return Err(StageEditBlock::Rejected("阶段编辑需要已预览的唯一稳定图元编号".into()));}
        let previous=old.get(&p.id).map(|old|assignment(old,legacy)).transpose()?;
        let desired=if selected.contains(&p.id){requested_stage.unwrap()}else if let Some(a)=p.stage_assignment{a.stage}
            else if let Some(a)=previous{a.stage}else{new_stage};
        if p.shape==ShapeKind::Background&&selected.contains(&p.id)&&desired!=StageTier::Global{return Err(StageEditBlock::Rejected("背景图元只能属于全图阶段".into()));}
        let desired=if p.shape==ShapeKind::Background{StageTier::Global}else{desired};let cell=cell_for(desired,p)?;
        let changed=previous.is_none_or(|a|a.stage!=desired||a.cell!=cell);
        let order=if changed{ticket=ticket.checked_add(1).ok_or_else(||StageEditBlock::Rejected("阶段队列编号已用尽".into()))?;ticket}
            else{previous.unwrap().queue_order};
        // Preserve legacy None on unchanged objects; only changed/new ownership
        // becomes a delta. A color/geometry edit may keep its old explicit tier.
        let content_changed=old.get(&p.id).is_none_or(|old|*old!=&*p);
        if changed||p.stage_assignment.is_some()||content_changed||selected.contains(&p.id){
            p.stage_assignment=Some(PrimitiveStageAssignment{stage:desired,cell,queue_order:order});}
    }
    if guard{
        let global_count=|canvas:&[Primitive],map:&StageView|canvas.iter().filter(|p|p.shape!=ShapeKind::Background&&map.stage_by_id[&p.id]==Stage::Global).count();
        let after=view(&proposed,legacy,guard)?;let prior_global=global_count(existing,&before);let next_global=global_count(&proposed,&after);
        if next_global>85&&next_global>prior_global{
            if requested_stage==Some(StageTier::Global){return Err(StageEditBlock::Rejected("全图阶段最多85个普通图元，请先移出或删除已有全图图元".into()));}
            let new_global=proposed.iter().filter(|p|!old.contains_key(&p.id)&&p.shape!=ShapeKind::Background&&after.stage_by_id[&p.id]==Stage::Global)
                .map(|p|p.id).collect::<HashSet<_>>();
            if new_global.is_empty(){return Err(StageEditBlock::Rejected("全图阶段已满，不能再把已有图元归入全图".into()));}
            if !global_confirmed{return Err(StageEditBlock::GlobalOverflow{candidate_canvas:proposed,
                message:"全图阶段已达到85个普通图元；进入粗网格后才会提交本次新增".into()});}
            for p in &mut proposed{if new_global.contains(&p.id){let cell=cell_for(StageTier::Coarse,p)?;ticket=ticket.checked_add(1).ok_or("阶段队列编号已用尽")?;
                p.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Coarse,cell,queue_order:ticket});}}
        }
        let fine_counts=|canvas:&[Primitive]|->Result<BTreeMap<usize,usize>,String>{let mut counts=BTreeMap::new();for p in canvas{
            let a=assignment(p,legacy)?;if a.stage==StageTier::Fine{*counts.entry(a.cell).or_default()+=1;}}Ok(counts)};
        let old_fine=fine_counts(existing)?;for(cell,count)in fine_counts(&proposed)?{
            if count>10&&count>old_fine.get(&cell).copied().unwrap_or(0){return Err(StageEditBlock::Rejected("上下文超支，不能添加，请先删除旧图形".into()));}}
    }
    Ok(proposed)
}

#[cfg(test)]
#[path="stage_edit_tests.rs"]
mod tests;
