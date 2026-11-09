//! The optimizer's closed cells are not necessarily complete model visits.
//! Quality cutoffs suspend a visit; reaching its trained quota completes it.
//! Model input must resume the earliest suspended stage before later geometry.
use crate::{history_context::RawRecord,region_state::{self,Stage}};
use std::collections::{HashMap,HashSet};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
enum Ending{Pause,Quota,Open}
struct Visit{key:(Stage,usize),rows:Vec<RawRecord>,ending:Ending,visible:usize}

pub(super) fn resume_records(raw:&[RawRecord])->Result<Vec<RawRecord>,String>{
    region_state::DrawingRegion::from_history(raw)?;
    let mut global=Vec::new();let mut visits=Vec::<Visit>::new();let mut active:Option<Visit>=None;
    let mut counts=HashMap::<(Stage,usize),usize>::new();
    let mut closed=HashSet::new();let mut unfinished=HashSet::new();
    for row in raw{
        if region_state::is_marker(row){
            let key=region_state::marker_identity(row)?;
            if closed.remove(&key){counts.insert(key,0);}
            active=Some(Visit{key,rows:vec![*row],ending:Ending::Open,visible:0});
        }else if row==&region_state::END_RECORD||row==&region_state::INTERRUPT_RECORD{
            let mut visit=active.take().ok_or("参考暂停转换缺少对应区域开始标记")?;
            let quota=if visit.key.0==Stage::Coarse{20}else{10};
            let count=counts.get(&visit.key).copied().unwrap_or(0);
            visit.ending=if row==&region_state::END_RECORD&&count>=quota{Ending::Quota}else{Ending::Pause};
            if visit.ending==Ending::Quota{closed.insert(visit.key);unfinished.remove(&visit.key);}
            else{unfinished.insert(visit.key);}
            visits.push(visit);
        }else if let Some(visit)=active.as_mut(){
            visit.rows.push(*row);
            if region_state::validate_visible(row).is_ok(){
                visit.visible+=1;*counts.entry(visit.key).or_default()+=1;
            }
        }else{global.push(*row);}
    }
    if let Some(visit)=active{
        unfinished.insert(visit.key);visits.push(visit);
    }
    // Fine geometry remains in SerializeResult.primitives. When coarse visits
    // are suspended it is queued by the importer until their real completion.
    let resume_side=unfinished.iter().map(|(stage,_)|stage.side()).min().unwrap_or(8);
    let mut result=global;
    for visit in visits{
        if visit.key.0.side()>resume_side{continue;}
        // There is no semantic content in an empty BEGIN + manual pause pair.
        // Omission leaves this cell naturally unvisited and available to draw.
        if visit.ending==Ending::Pause&&visit.visible==0{continue;}
        result.extend(visit.rows);
        match visit.ending{
            Ending::Pause=>result.push(region_state::INTERRUPT_RECORD),
            Ending::Quota=>result.push(region_state::END_RECORD),
            Ending::Open=>{}, // Budget/search exits retain a legal open prefix.
        }
    }
    region_state::DrawingRegion::from_history(&result)?;
    Ok(result)
}

#[cfg(test)]mod tests{
    use super::*;use crate::document::{Primitive,ShapeKind};
    fn row()->RawRecord{Primitive::new(ShapeKind::Ellipse,20.,20.,8.,10.,[40,80,120]).to_tokens().unwrap().map(u32::from)}
    fn visit(raw:&mut Vec<RawRecord>,stage:Stage,index:usize,n:usize,end:RawRecord){
        raw.push(region_state::marker_record(stage,index).unwrap());raw.extend(std::iter::repeat_n(row(),n));raw.push(end);
    }
    #[test]fn quality_cutoffs_pause_each_nonempty_cell_and_do_not_claim_completion(){
        for stage in [Stage::Coarse,Stage::Fine]{
            let mut raw=vec![region_state::BOS_RECORD];visit(&mut raw,stage,0,2,region_state::END_RECORD);visit(&mut raw,stage,1,3,region_state::END_RECORD);
            let result=resume_records(&raw).unwrap();assert_eq!(result.last(),Some(&region_state::INTERRUPT_RECORD));
            assert!(!result.contains(&region_state::END_RECORD));
            let state=region_state::DrawingRegion::from_history(&result).unwrap();assert!(!state.completed);
            assert!(state.progress.coarse_completed.is_empty()&&state.progress.fine_completed.is_empty());
        }
    }
    #[test]fn reaching_the_real_shape_quota_keeps_end_instead_of_pause(){
        for (stage,quota)in[(Stage::Coarse,20),(Stage::Fine,10)]{
            let mut raw=vec![region_state::BOS_RECORD];visit(&mut raw,stage,0,quota,region_state::END_RECORD);
            assert_eq!(resume_records(&raw).unwrap(),raw);
        }
    }
    #[test]fn coarse_pause_defers_fine_rows_but_keeps_same_stage_quota_ends(){
        let mut raw=vec![region_state::BOS_RECORD];visit(&mut raw,Stage::Coarse,0,20,region_state::END_RECORD);
        visit(&mut raw,Stage::Coarse,1,2,region_state::END_RECORD);
        visit(&mut raw,Stage::Fine,0,3,region_state::END_RECORD);
        let result=resume_records(&raw).unwrap();assert_eq!(result.last(),Some(&region_state::INTERRUPT_RECORD));
        assert_eq!(result.iter().filter(|r|region_state::validate_visible(r).is_ok()).count(),22);
        assert_eq!(result.iter().filter_map(|r|region_state::marker_identity(r).ok()).collect::<Vec<_>>(),[(Stage::Coarse,0),(Stage::Coarse,1)]);
        let state=region_state::DrawingRegion::from_history(&result).unwrap();assert_eq!(state.progress.coarse_completed,[0]);
        assert!(state.progress.fine.is_empty());
    }
    #[test]fn empty_quality_cells_are_unvisited_without_fake_begin_pause_pairs(){
        let mut raw=vec![region_state::BOS_RECORD];visit(&mut raw,Stage::Coarse,0,0,region_state::END_RECORD);
        visit(&mut raw,Stage::Coarse,1,2,region_state::END_RECORD);visit(&mut raw,Stage::Fine,0,3,region_state::END_RECORD);
        let result=resume_records(&raw).unwrap();assert_eq!(result.len(),5);
        assert_eq!(result[1],region_state::marker_record(Stage::Coarse,1).unwrap());
        assert_eq!(result.last(),Some(&region_state::INTERRUPT_RECORD));
        assert!(!result.windows(2).any(|p|region_state::is_marker(&p[0])&&p[1]==region_state::INTERRUPT_RECORD));
    }
    #[test]fn global_and_budget_open_prefixes_never_get_an_illegal_manual_pause(){
        let raw=vec![region_state::BOS_RECORD,row()];assert_eq!(resume_records(&raw).unwrap(),raw);
        let mut raw=vec![region_state::BOS_RECORD,region_state::marker_record(Stage::Fine,0).unwrap(),row()];
        assert_eq!(resume_records(&raw).unwrap(),raw);
        raw.push(region_state::END_RECORD);assert_eq!(resume_records(&raw).unwrap().last(),Some(&region_state::INTERRUPT_RECORD));
    }
    #[test]fn a_prior_interrupt_accumulates_until_a_real_quota_end(){
        let mut raw=vec![region_state::BOS_RECORD];visit(&mut raw,Stage::Coarse,0,10,region_state::INTERRUPT_RECORD);
        visit(&mut raw,Stage::Coarse,0,10,region_state::END_RECORD);
        visit(&mut raw,Stage::Fine,0,1,region_state::END_RECORD);
        let result=resume_records(&raw).unwrap();assert!(result.iter().any(|r|region_state::marker_identity(r).is_ok_and(|(s,_)|s==Stage::Fine)));
        assert_eq!(result.last(),Some(&region_state::INTERRUPT_RECORD));
    }
}
