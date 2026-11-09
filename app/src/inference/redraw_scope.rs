//! Request-local region scheduling; never changes stored global preferences.
use super::*;

#[derive(Clone, Debug)]
pub(super) struct RedrawScope { targets: Vec<(usize, usize)>, fine_context_activated: bool }

pub(super) fn request_settings(request: &GenerationRequest) -> InferenceSettings {
    let mut settings = request.settings.clone();
    if !request.redraw_regions.is_empty() {
        settings.pause_at_stage_end = false;
        settings.half_region_switch = false;
    }
    settings
}

impl RedrawScope {
    pub(super) fn initialize(request: &GenerationRequest, history: &mut History) -> Result<Option<Self>> {
        if request.redraw_regions.is_empty() { return Ok(None); }
        let mut targets = request.redraw_regions.clone();
        if targets.len()>81 || targets.iter().any(|&(side,index)| ![1,4,8].contains(&side) || index >= side*side) {
            bail!("局部重绘区域编号无效");
        }
        targets.sort_unstable(); targets.dedup();
        let global=targets.contains(&(1,0));
        if history.active.is_some() || history.pending_interrupt_side.is_some() {
            bail!("局部重绘初始上下文必须由完整区域组成，不能包含活动块或悬空中断");
        }
        if global {
            // Global is the ordinary BOS/background prefix, never a synthetic
            // regional marker or END. All later-stage contents must be cleared.
            if history.highest_side.is_some() || !history.complete.is_empty() || !history.interrupted.is_empty()
                || global_ordinary_count(history)>85 || history.fixed.len()>87 {
                bail!("全图续画必须使用已清空粗细阶段的合法全图前缀");
            }
            if history.fixed.len()>1 && (history.fixed[1][5]!=OFFSETS[5]
                ||history.fixed.iter().skip(2).any(|row|row[5]==OFFSETS[5])) {
                bail!("全图续画必须保留唯一的原有背景");
            }
        }
        if targets.iter().any(|target| history.complete.iter().chain(&history.interrupted)
            .any(|block| (block.side,block.index)==*target && block.records.iter().any(visible))) {
            bail!("局部重绘目标的旧历史应作为待重新读取的上下文提供，不能保留旧的结束标记");
        }
        if targets.iter().any(|(side,_)|*side==4) && history.highest_side==Some(8) {
            bail!("局部粗网格重绘必须使用粗网格前缀；既有细网格应作为延后上下文提供");
        }
        if !request.manual_review_regions.is_empty() {
            bail!("局部重绘不接受普通模式的人工复查区域，请重新构建局部上下文");
        }
        for pending in &request.pending_stage_context {
            if ![4,8].contains(&pending.side) || pending.index>=pending.side*pending.side
                || pending.side==4&&!targets.contains(&(pending.side,pending.index)) { bail!("局部重绘延后上下文的区域无效"); }
            validate_visible(&pending.tokens)?;
            if pending.tokens[5]==OFFSETS[5] {
                bail!("局部重绘延后上下文不能包含背景");
            }
        }
        for target in &targets { history.closed.remove(target); history.visit_counts.remove(target); }
        // A sparse/empty global prefix must never enter global generation.
        if !global { history.highest_side=Some(targets[0].0); }
        Ok(Some(Self { targets, fine_context_activated:false }))
    }
    fn next(&self, history: &History) -> Option<(usize, usize)> {
        if self.targets.contains(&(1,0)) && global_incomplete(history) { return None; }
        self.targets.iter().copied().find(|target| target.0!=1 && !history.closed.contains(target))
    }
}

/// A selected region gets its existing yellow contents exactly once, when its
/// fresh visit starts. No prior END is replayed as completion of that visit.
pub(super) fn begin_generation_region(history:&mut History,full:&mut Vec<[u32;9]>,side:usize,index:usize,
    pending:&[PendingStageRecord],scope:Option<&RedrawScope>,reviews:&mut HashSet<(usize,usize)>)->Result<RegionBlock>{
    let marker=region_marker(side,index);let mut records=vec![marker];
    if let Some(scope)=scope{
        if !scope.targets.contains(&(side,index)){bail!("局部重绘尝试进入未选择区域");}
        records.extend(pending.iter().filter(|p|(p.side,p.index)==(side,index)).map(|p|p.tokens));
        // Even an empty selected visit must obtain a real model END. Hitting
        // its trained quota alone may not invent a permanent completion.
        reviews.insert((side,index));
    }
    full.extend_from_slice(&records);history.pending_interrupt_side=None;history.highest_side=Some(side);
    *history.visit_counts.entry((side,index)).or_default()+=records.len()-1;
    Ok(RegionBlock{side,index,records})
}

pub(super) fn guarded_scoped_review_full(settings:&InferenceSettings,history:&History,active:&RegionBlock,
    review:bool,scope:Option<&RedrawScope>)->bool{
    guarded_review_full(settings,history,active,review)
        ||scope.is_some()&&review&&history.visit_counts.get(&(active.side,active.index)).copied().unwrap_or(0)>=active.cap()
}

pub(super) fn generation_region_target(history: &History, scope: Option<&RedrawScope>, rng: &mut StdRng) -> Result<Option<(usize,usize)>> {
    if let Some(scope)=scope { return Ok(scope.next(history)); }
    let Some(side)=next_stage(history) else { return Ok(None); };
    let available=(0..side*side).filter(|index|!history.closed.contains(&(side,*index))).collect::<Vec<_>>();
    if available.is_empty() { bail!("同阶段中断后没有可恢复的未完成区域"); }
    Ok(Some((side,available[rng.random_range(0..available.len())])))
}

pub(super) fn activate_generation_pending(history:&mut History,full:&mut Vec<[u32;9]>,pending:&[PendingStageRecord],
    activated:&mut HashSet<usize>,reviews:&mut HashSet<(usize,usize)>,scope:&mut Option<RedrawScope>)->Result<bool> {
    let Some(scope)=scope.as_mut() else { return activate_pending_stage(history,full,pending,activated,reviews); };
    if history.active.is_some() || scope.fine_context_activated || scope.next(history).is_none_or(|(side,_)|side!=8) { return Ok(false); }
    let mut groups=std::collections::BTreeMap::<usize,Vec<[u32;9]>>::new();
    for entry in pending {if entry.side==8&&!scope.targets.contains(&(entry.side,entry.index)){
        groups.entry(entry.index).or_default().push(entry.tokens);
    }}
    scope.fine_context_activated=true;
    if groups.is_empty() { return Ok(false); }
    let previous_closed=history.closed.clone();
    let mut next=full.clone();
    for (index,records) in groups { next.push(region_marker(8,index));next.extend(records);next.push(END_RECORD); }
    let mut parsed=parse_request_history(&next,Some(8))?;
    // Old empty target markers are conditioning only, not completion of this request.
    for target in &scope.targets { if !previous_closed.contains(target) { parsed.closed.remove(target); } }
    *history=parsed;*full=next;activated.insert(8);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ordinary()->[u32;9] { [510,1500,2049,2500,4500,6415,6700,6800,7000] }
    fn request(targets:Vec<(usize,usize)>,raw:Vec<[u32;9]>)->GenerationRequest {
        GenerationRequest { prompt:String::new(), settings:InferenceSettings { cfg_scale:1.,renyi_alpha:4.,seed:1,max_primitives:100,
            context_overclock:true,pause_at_stage_end:true,half_region_switch:true },primitives:Vec::new(),history_records:Some(raw),
            manual_review_regions:Vec::new(),extra_fine_context:Vec::new(),stage_floor:Some(4),pending_stage_context:Vec::new(),redraw_regions:targets }
    }
    #[test]
    fn redraw_only_visits_selected_coarse_then_fine_and_ignores_normal_completion_counts() {
        let request=request(vec![(8,63),(4,7),(4,2),(4,7)],vec![BOS_RECORD]);
        let mut history=parse_request_history(request.history_records.as_ref().unwrap(),Some(4)).unwrap();
        let scope=RedrawScope::initialize(&request,&mut history).unwrap();let mut rng=StdRng::seed_from_u64(100);
        assert!(!global_incomplete(&history));
        for target in [(4,2),(4,7),(8,63)] {
            assert_eq!(generation_region_target(&history,scope.as_ref(),&mut rng).unwrap(),Some(target));
            history.closed.insert(target);
        }
        assert_eq!(generation_region_target(&history,scope.as_ref(),&mut rng).unwrap(),None);
        let settings=request_settings(&request);assert!(!settings.pause_at_stage_end&&!settings.half_region_switch);
        assert!(request.settings.pause_at_stage_end&&request.settings.half_region_switch);
    }
    #[test]
    fn fine_context_is_injected_once_only_after_selected_coarse_and_never_becomes_review_or_target() {
        let mut request=request(vec![(4,3),(8,20)],vec![BOS_RECORD]);
        request.pending_stage_context=vec![PendingStageRecord{side:8,index:9,tokens:ordinary()}];
        let mut full=request.history_records.clone().unwrap();let mut history=parse_request_history(&full,Some(4)).unwrap();
        let mut scope=RedrawScope::initialize(&request,&mut history).unwrap();let mut activated=HashSet::new();let mut reviews=HashSet::new();
        assert!(!activate_generation_pending(&mut history,&mut full,&request.pending_stage_context,&mut activated,&mut reviews,&mut scope).unwrap());
        full.extend([region_marker(4,3),END_RECORD]);history=parse_request_history(&full,Some(4)).unwrap();
        assert!(activate_generation_pending(&mut history,&mut full,&request.pending_stage_context,&mut activated,&mut reviews,&mut scope).unwrap());
        assert_eq!(&full[full.len()-3..],&[region_marker(8,9),ordinary(),END_RECORD]);assert!(reviews.is_empty());
        assert_eq!(scope.as_ref().unwrap().next(&history),Some((8,20)));let once=full.clone();
        assert!(!activate_generation_pending(&mut history,&mut full,&request.pending_stage_context,&mut activated,&mut reviews,&mut scope).unwrap());assert_eq!(full,once);
    }
    #[test]
    fn redraw_rejects_occupied_bad_or_reversed_history_and_coarse_only_keeps_fine_deferred() {
        for raw in [vec![BOS_RECORD,region_marker(4,3),ordinary(),END_RECORD],vec![BOS_RECORD,region_marker(4,3)],vec![BOS_RECORD,region_marker(8,9),END_RECORD]] {
            let request=request(vec![(4,3)],raw.clone());let mut history=parse_request_history(&raw,Some(4)).unwrap();
            assert!(RedrawScope::initialize(&request,&mut history).is_err());
        }
        let mut request=request(vec![(4,3)],vec![BOS_RECORD]);request.pending_stage_context=vec![PendingStageRecord{side:8,index:9,tokens:ordinary()}];
        let mut full=request.history_records.clone().unwrap();let mut history=parse_request_history(&full,Some(4)).unwrap();let mut scope=RedrawScope::initialize(&request,&mut history).unwrap();
        history.closed.insert((4,3));assert!(!activate_generation_pending(&mut history,&mut full,&request.pending_stage_context,&mut HashSet::new(),&mut HashSet::new(),&mut scope).unwrap());
        assert_eq!(full,vec![BOS_RECORD]);request.redraw_regions=vec![(8,64)];assert!(RedrawScope::initialize(&request,&mut history).is_err());
    }
    #[test]
    fn selected_yellow_context_is_re_read_at_its_visit_without_replaying_an_end(){
        let mut request=request(vec![(4,3),(8,20)],vec![BOS_RECORD]);
        request.pending_stage_context=vec![PendingStageRecord{side:4,index:3,tokens:ordinary()},
            PendingStageRecord{side:8,index:20,tokens:ordinary()},PendingStageRecord{side:8,index:9,tokens:ordinary()}];
        let mut full=vec![BOS_RECORD];let mut history=parse_request_history(&full,Some(4)).unwrap();
        let mut scope=RedrawScope::initialize(&request,&mut history).unwrap();let mut reviews=HashSet::new();
        let active=begin_generation_region(&mut history,&mut full,4,3,&request.pending_stage_context,scope.as_ref(),&mut reviews).unwrap();
        assert_eq!(full,vec![BOS_RECORD,region_marker(4,3),ordinary()]);assert_eq!(history.visit_counts[&(4,3)],1);
        assert!(reviews.contains(&(4,3)));full.push(END_RECORD);finish_visit(&mut history,active).unwrap();
        assert!(activate_generation_pending(&mut history,&mut full,&request.pending_stage_context,&mut HashSet::new(),&mut reviews,&mut scope).unwrap());
        assert_eq!(&full[full.len()-3..],&[region_marker(8,9),ordinary(),END_RECORD]);
        assert!(!full.contains(&region_marker(8,20)));
        let active=begin_generation_region(&mut history,&mut full,8,20,&request.pending_stage_context,scope.as_ref(),&mut reviews).unwrap();
        assert_eq!(&full[full.len()-2..],&[region_marker(8,20),ordinary()]);assert_eq!(active.records.len(),2);
        assert!(!history.closed.contains(&(8,20)));
    }
    #[test]
    fn global_scope_uses_the_real_prefix_then_only_selected_regional_targets(){
        let mut background=ordinary();background[5]=OFFSETS[5];
        let mut full=vec![BOS_RECORD,background];full.extend(std::iter::repeat(ordinary()).take(84));
        let mut request=request(vec![(8,63),(1,0),(4,2)],full.clone());request.stage_floor=None;
        let mut history=parse_request_history(&full,None).unwrap();let scope=RedrawScope::initialize(&request,&mut history).unwrap();
        assert!(global_incomplete(&history));assert_eq!(history.highest_side,None);
        assert_eq!(scope.as_ref().unwrap().next(&history),None,"a Global visit has no regional marker");
        full.push(ordinary());history.fixed.push(ordinary());assert!(!global_incomplete(&history));
        let mut rng=StdRng::seed_from_u64(1);
        assert_eq!(generation_region_target(&history,scope.as_ref(),&mut rng).unwrap(),Some((4,2)));
        history.closed.insert((4,2));assert_eq!(generation_region_target(&history,scope.as_ref(),&mut rng).unwrap(),Some((8,63)));
        history.closed.insert((8,63));assert_eq!(generation_region_target(&history,scope.as_ref(),&mut rng).unwrap(),None);
        assert!(full.iter().all(|row|!is_marker(row)&&row!=&END_RECORD&&row!=&INTERRUPT_RECORD));
    }
    #[test]
    fn global_scope_allows_first_background_but_rejects_old_regional_history_and_bad_global_prefixes(){
        let mut request=request(vec![(1,0)],vec![BOS_RECORD]);request.stage_floor=None;
        let mut history=parse_request_history(&[BOS_RECORD],None).unwrap();
        assert!(RedrawScope::initialize(&request,&mut history).unwrap().is_some());assert!(global_incomplete(&history));
        let mut background=ordinary();background[5]=OFFSETS[5];
        for full in [vec![BOS_RECORD,ordinary()],vec![BOS_RECORD,background,background],
            vec![BOS_RECORD,background,region_marker(4,0),END_RECORD],
            std::iter::once(BOS_RECORD).chain(std::iter::once(background)).chain(std::iter::repeat(ordinary()).take(86)).collect()]{
            let mut history=parse_request_history(&full,None).unwrap();assert!(RedrawScope::initialize(&request,&mut history).is_err());
        }
    }
}
