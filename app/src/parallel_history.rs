//! Independent generation branches sharing an immutable origin state.
//! Output order may interleave freely. Lane zero follows its current head only
//! while the user remains on it; browsing another state never stops generation.

use crate::document::{parallel_seed, Document, NodeId, Primitive, Settings};
use crate::history_context::{RawHistoryStore, RawRecord};
use crate::region_state::DrawingRegion;
use std::collections::HashSet;

#[derive(Clone)]struct ModelLaneInput{ids:Vec<u64>,last_raw:Vec<RawRecord>,activated:HashSet<usize>}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaneStatus {
    Running,
    Finished,
    Failed(String),
}

pub struct ParallelHistory {
    pub origin: NodeId,
    pub heads: Vec<NodeId>,
    pub seeds: Vec<u64>,
    pub generated: Vec<usize>,
    pub statuses: Vec<LaneStatus>,
    prompt: String,
    settings: Settings,
    regions: Vec<Option<DrawingRegion>>,
    follow_live: bool,
    // A lane that emits no visible primitive has no independent tree branch.
    // Only lane zero publishes a separate display-control overlay. Immutable
    // origin deltas and existing descendants are never rewritten by a lane.
    zero_output_contexts: Vec<Option<Vec<RawRecord>>>,
    model_inputs:Vec<Option<ModelLaneInput>>,
    origin_excluded_reviews:Vec<(usize,usize)>,
    redraw_scope:Vec<(usize,usize)>,
    origin_completion:Option<crate::mode_transition::CompletionEvidence>,
    queued_stage_ids:Vec<(usize,usize,u64)>,
    queued_stage_tokens:Vec<(usize,usize,u64,RawRecord)>,
}

impl ParallelHistory {
    pub fn new(
        doc: &Document,
        origin: NodeId,
        prompt: String,
        settings: Settings,
    ) -> Result<Self, String> {
        settings.validate()?;
        doc.materialize(origin)?;
        let count = settings.parallel_sequences;
        fn filled<T:Clone>(count:usize,value:T)->Result<Vec<T>,String>{
            let mut result=Vec::new();result.try_reserve_exact(count).map_err(|_|"并行分支过多，系统内存不足或分支数量无法表示".to_owned())?;
            result.resize(count,value);Ok(result)
        }
        let mut seeds=Vec::new();seeds.try_reserve_exact(count).map_err(|_|"并行分支过多，系统内存不足或分支数量无法表示".to_owned())?;
        seeds.extend((0..count).map(|lane|parallel_seed(settings.seed,lane)));
        Ok(Self {
            origin,
            heads: filled(count,origin)?,
            seeds,
            generated: filled(count,0)?,
            statuses: filled(count,LaneStatus::Running)?,
            prompt,
            settings,
            regions: filled(count,None)?,
            follow_live: true,
            zero_output_contexts: filled(count,None)?,
            model_inputs:filled(count,None)?,queued_stage_ids:Vec::new(),queued_stage_tokens:Vec::new(),origin_excluded_reviews:Vec::new(),redraw_scope:Vec::new(),origin_completion:None,
        })
    }

    /// Freeze conditioning once for all lanes. In particular, a displayed lane
    /// moving doc.current or publishing a zero-output overlay cannot change a
    /// sibling's admission policy, queued IDs or phase cursor.
    pub fn initialize_model_input(&mut self,doc:&Document,store:&RawHistoryStore)->Result<(),String>{
        self.origin_completion=Some(store.completion_evidence(doc,self.origin)?);
        let operative=store.manual_review_regions(doc,self.origin)?;
        self.origin_excluded_reviews=store.creation_review_snapshot(self.origin).into_iter().filter(|key|!operative.contains(key)).collect();
        let input=store.effective_history(doc,self.origin)?;let floor=store.stage_floor(doc,self.origin)?.unwrap_or(1);
        self.queued_stage_ids=store.pending_stage_ids(doc,self.origin)?;
        let activated=[4usize,8].into_iter().filter(|side|*side<=floor).collect::<HashSet<_>>();
        for entry in &mut self.model_inputs{*entry=Some(ModelLaneInput{ids:input.primitive_ids.clone(),last_raw:input.records.clone(),activated:activated.clone()});}
        Ok(())
    }
    /// A redraw request intentionally projects a subset of an older completed
    /// canvas. Freeze that request directly rather than rebuilding its normal
    /// monotonic phase history, and activate queued fine IDs only on real markers.
    pub fn initialize_explicit_model_input(&mut self,doc:&Document,store:&RawHistoryStore,
        records:&[RawRecord],ids:&[u64],queued_stage_ids:Vec<(usize,usize,u64)>,redraw_scope:Vec<(usize,usize)>)->Result<(),String>{
        store.validate_context_with_model_ids(doc,self.origin,records,ids)?;
        self.origin_completion=Some(store.completion_evidence(doc,self.origin)?);
        let mut highest=1;
        for row in records{if crate::region_state::is_marker(row){highest=highest.max(crate::region_state::marker_identity(row)?.0.side());}}
        let activated=[4usize,8].into_iter().filter(|side|*side<=highest).collect::<HashSet<_>>();
        if redraw_scope.is_empty()||redraw_scope.len()>81||redraw_scope.iter().any(|(side,index)|![1,4,8].contains(side)||*index>=side*side){return Err("局部重绘的分支区域范围无效".into());}
        let canvas=doc.materialize(self.origin)?;let by_id=canvas.iter().map(|p|(p.id,p)).collect::<std::collections::HashMap<_,_>>();
        self.queued_stage_tokens=queued_stage_ids.iter().map(|&(s,i,id)|{
            let row=by_id.get(&id).ok_or("局部续画待重新读取的图元不存在")?.to_tokens()?.map(u32::from);Ok((s,i,id,row))
        }).collect::<Result<Vec<_>,String>>()?;
        self.redraw_scope=redraw_scope;self.queued_stage_ids=queued_stage_ids;
        self.origin_excluded_reviews=store.creation_review_snapshot(self.origin);
        for entry in &mut self.model_inputs{*entry=Some(ModelLaneInput{ids:ids.to_vec(),last_raw:records.to_vec(),activated:activated.clone()});}
        Ok(())
    }
    fn next_model_input(&self,doc:&Document,store:&RawHistoryStore,lane:usize,raw:&[RawRecord],new_primitive:bool)->Result<ModelLaneInput,String>{
        let mut state=if let Some(state)=&self.model_inputs[lane]{state.clone()}else{
            // Legacy callers may not initialize a request. Their first callback
            // must not inherit a different zero-output lane's public overlay.
            ModelLaneInput{ids:store.model_ids_for_node(doc,self.heads[lane])?,last_raw:Vec::new(),activated:HashSet::new()}
        };
        if !state.last_raw.is_empty()&&!raw.starts_with(&state.last_raw){return Err("模型回调不再追加于该分支冻结的输入上下文".into());}
        if self.redraw_scope.is_empty(){
            let mut maximum=1;for row in raw{if crate::region_state::is_marker(row){maximum=maximum.max(crate::region_state::marker_identity(row)?.0.side());}}
            for side in [4usize,8]{if maximum>=side&&state.activated.insert(side){
                state.ids.extend(self.queued_stage_ids.iter().filter(|(s,_,_)|*s==side).map(|(_,_,id)|*id));
            }}
        }else{
            // Scoped visits activate retained yellow contents per exact block,
            // not all queued IDs as soon as one stage marker is encountered.
            let used=state.ids.iter().copied().collect::<HashSet<_>>();
            let mut queued=std::collections::BTreeMap::<(usize,usize),std::collections::VecDeque<(u64,RawRecord)>>::new();
            for &(side,index,id,row)in &self.queued_stage_tokens{if !used.contains(&id){
                queued.entry((side,index)).or_default().push_back((id,row));
            }}
            let mut key=None;let mut appended=Vec::new();let mut entered=HashSet::new();
            for (offset,row)in raw.iter().enumerate(){
                if crate::region_state::is_marker(row){let(stage,index)=crate::region_state::marker_identity(row)?;key=Some((stage.side(),index));
                    if offset>=state.last_raw.len(){entered.insert((stage.side(),index));}}
                else if offset>=state.last_raw.len()&&crate::region_state::validate_visible(row).is_ok(){appended.push((key,*row));}
            }
            let retained=appended.len().checked_sub(usize::from(new_primitive)).ok_or("局部续画回调缺少新图元")?;
            for &(key,row)in appended.iter().take(retained){
                let key=key.ok_or("局部续画不能重新注入全局图元")?;
                let (id,expected)=queued.get_mut(&key).and_then(|q|q.pop_front()).ok_or("局部续画回调出现未排队的已有图元")?;
                if row!=expected{return Err("局部续画重新读取的图元顺序或属性与原画布不一致".into());}
                state.ids.push(id);
            }
            if entered.iter().any(|key|queued.get(key).is_some_and(|rows|!rows.is_empty())){
                return Err("局部续画开始区域前尚未完整重新读取该区域的已有图元".into());
            }
        }
        let count=raw.iter().filter(|row|!crate::region_state::is_marker(row)
            &&![crate::region_state::BOS,crate::region_state::PAD,crate::region_state::TRUE_END,crate::region_state::INTERRUPT].contains(&row[0])).count();
        if count!=state.ids.len()+usize::from(new_primitive){return Err("模型输入ID顺序与已激活未来图元数量不一致".into());}
        state.last_raw=raw.to_vec();Ok(state)
    }

    pub fn lane_count(&self) -> usize {
        self.heads.len()
    }
    pub fn all_terminal(&self) -> bool {
        self.statuses.iter().all(|s| *s != LaneStatus::Running)
    }
    pub fn all_redraw_regions_complete(&self)->bool{
        !self.redraw_scope.is_empty()&&self.statuses.iter().enumerate().all(|(lane,status)|{
            *status==LaneStatus::Finished&&self.regions[lane].as_ref().is_some_and(|region|{
                self.redraw_scope.iter().all(|(side,index)|if *side==1{
                    region.progress.global_complete
                }else if *side==4{
                    region.progress.coarse_completed.contains(index)
                }else{region.progress.fine_completed.contains(index)})
            })
        })
    }
    pub fn set_follow_live(&mut self, follow: bool) {
        self.follow_live = follow;
    }
    pub fn is_following(&self, doc: &Document) -> bool {
        self.follow_live && self.heads.first().copied() == Some(doc.current)
    }
    pub fn region_for_lane(&self, lane: usize) -> Option<&DrawingRegion> {
        self.regions.get(lane).and_then(Option::as_ref)
    }

    pub fn set_region(&mut self, lane: usize, region: DrawingRegion) -> Result<(), String> {
        self.require_running(lane)?;
        self.regions[lane] = Some(region);
        Ok(())
    }

    pub fn context_for_lane(
        &self,
        doc: &Document,
        store: &RawHistoryStore,
        lane: usize,
    ) -> Result<Option<Vec<RawRecord>>, String> {
        self.require_lane(lane)?;
        if let Some(raw) = &self.zero_output_contexts[lane] {
            return Ok(Some(raw.clone()));
        }
        store.context_for(doc, self.heads[lane])
    }

    /// Prefer the displayed lane's control-only result, never a different
    /// historical state. A shared zero-output origin deterministically uses
    /// lane zero before another lane; unrelated browsing uses the store.
    pub fn displayed_context_for(
        &self,
        doc: &Document,
        store: &RawHistoryStore,
    ) -> Result<Option<Vec<RawRecord>>, String> {
        if let Some(lane) = self.heads.iter().position(|&head| head == doc.current) {
            return self.context_for_lane(doc, store, lane);
        }
        store.context_for(doc, doc.current)
    }

    pub fn submit(
        &mut self,
        doc: &mut Document,
        store: &mut RawHistoryStore,
        lane: usize,
        tokens: RawRecord,
        raw: &[RawRecord],
    ) -> Result<NodeId, String> {
        self.require_running(lane)?;
        let mut native = [0u16; 9];
        for i in 0..9 {
            native[i] = tokens[i]
                .try_into()
                .map_err(|_| "完整图元 token 超出原生词表")?;
        }
        let mut primitive = Primitive::from_tokens(0, native)?;
        let parent = self.heads[lane];
        let append_from=self.model_inputs[lane].as_ref().map_or(raw.len(),|input|input.last_raw.len());
        let mut input=self.next_model_input(doc,store,lane,raw,true)?;
        primitive.stage_assignment=Some(store.generated_assignment(doc,parent,raw)?);
        if !self.redraw_scope.is_empty(){
            let assignment=primitive.stage_assignment.unwrap();
            let new_global_background=primitive.shape==crate::document::ShapeKind::Background
                &&self.redraw_scope.contains(&(1,0))&&doc.materialize(parent)?.is_empty();
            if (primitive.shape==crate::document::ShapeKind::Background&&!new_global_background)
                ||!self.redraw_scope.contains(&(assignment.stage.side(),assignment.cell)){
                return Err("模型尝试在局部重绘所选范围之外输出，已保留原画布".into());
            }
        }
        let follow = lane == 0 && self.follow_live && doc.current == parent;
        // Validation precedes any document mutation, including ID allocation.
        let region = store.validate_generated_with_ids(doc, parent, &primitive, raw,&input.ids)?;
        let mut settings = self.settings.clone();
        settings.seed = self.seeds[lane];
        let node = doc.commit_generated_at_ordered(parent, primitive, self.prompt.clone(), settings,!self.redraw_scope.is_empty())?;
        let generated_id=match doc.nodes[node].patches.as_slice(){[crate::document::Patch::Add{primitive,..}]=>primitive.id,
            _=>{doc.rollback_last_generated(node)?;return Err("新模型状态不是单图元增量".into());}};
        input.ids.push(generated_id);
        if let Err(error) = store.attach_with_model_ids(doc, node, raw,&input.ids) {
            if let Err(rollback) = doc.rollback_last_generated(node) {
                return Err(format!("{error}；回滚模型状态失败：{rollback}"));
            }
            return Err(error);
        }
        if parent==self.origin {
            store.exclude_frozen_review_keys(node,&self.origin_excluded_reviews,false);
            if let Some(evidence)=&self.origin_completion{store.set_completion_baseline(doc,node,evidence)?;}
        }
        store.record_completed_regions(doc,node,raw,&input.ids,append_from,&self.redraw_scope,false)?;
        if self.lane_count() > 1 {
            doc.nodes[node].title = format!(
                "并行 {}/{} · {}", lane + 1, self.lane_count(), doc.nodes[node].title
            );
        }
        self.heads[lane] = node;
        self.model_inputs[lane]=Some(input);
        self.generated[lane] += 1;
        self.regions[lane] = Some(region);
        self.zero_output_contexts[lane] = None;
        if follow {
            doc.jump(node)?;
        }
        Ok(node)
    }

    pub fn finish_lane(
        &mut self,
        doc: &Document,
        store: &mut RawHistoryStore,
        lane: usize,
        raw: &[RawRecord],
    ) -> Result<(), String> {
        self.require_running(lane)?;
        let head = self.heads[lane];
        let append_from=self.model_inputs[lane].as_ref().map_or(raw.len(),|input|input.last_raw.len());
        let input=self.next_model_input(doc,store,lane,raw,false)?;
        let region = store.validate_context_with_model_ids(doc, head, raw,&input.ids)?;
        if self.generated[lane] == 0 {
            if lane == 0 {
                store.set_control_overlay_with_ids(doc, head, raw,&input.ids)?;
                if head==self.origin{store.exclude_frozen_review_keys(head,&self.origin_excluded_reviews,true);}
            }
            self.zero_output_contexts[lane] = Some(raw.to_vec());
        } else {
            store.set_control_overlay_with_ids(doc, head, raw,&input.ids)?;
        }
        if self.generated[lane]>0||lane==0{store.record_completed_regions(doc,head,raw,&input.ids,append_from,&self.redraw_scope,true)?;}
        self.regions[lane] = Some(region);
        self.model_inputs[lane]=Some(input);
        self.statuses[lane] = LaneStatus::Finished;
        Ok(())
    }

    pub fn fail_lane(&mut self, lane: usize, message: impl Into<String>) -> Result<(), String> {
        self.require_running(lane)?;
        self.statuses[lane] = LaneStatus::Failed(message.into());
        Ok(())
    }

    fn require_lane(&self, lane: usize) -> Result<(), String> {
        if lane >= self.lane_count() {
            Err(format!("并行序列 {} 不存在", lane + 1))
        } else {
            Ok(())
        }
    }
    fn require_running(&self, lane: usize) -> Result<(), String> {
        self.require_lane(lane)?;
        if self.statuses[lane] != LaneStatus::Running {
            return Err(format!("并行序列 {} 已结束，拒绝迟到或重复输出", lane + 1));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn more_than_sixteen_branches_preserve_all_heads_and_unique_seeds(){
        let document=Document::new();
        for count in [17,32,1000]{
            let settings=Settings{parallel_sequences:count,..Default::default()};
            let branches=ParallelHistory::new(&document,document.current,String::new(),settings).unwrap();
            assert_eq!(branches.heads.len(),count);assert_eq!(branches.seeds.len(),count);
            assert_eq!(branches.generated.len(),count);assert_eq!(branches.statuses.len(),count);
            assert_eq!(branches.seeds.iter().copied().collect::<std::collections::HashSet<_>>().len(),count);
            assert_eq!(document.nodes.len(),1);
        }
        assert!(ParallelHistory::new(&document,document.current,String::new(),Settings{parallel_sequences:usize::MAX,..Default::default()}).is_err());
    }
    use crate::document::{OperationSource, ShapeKind};
    use crate::region_state::{marker_record, Stage, BOS_RECORD, END_RECORD};

    fn rect(x: f32) -> Primitive {
        Primitive::new(ShapeKind::Rectangle, x, 50., 20., 10., [100, 150, 200])
    }
    fn records(canvas: &[Primitive]) -> Vec<RawRecord> {
        let mut r = vec![BOS_RECORD];
        r.extend(canvas.iter().map(|p| p.to_tokens().unwrap().map(u32::from)));
        r
    }
    fn callback_raw(base: &[RawRecord], p: &Primitive) -> Vec<RawRecord> {
        let mut raw = base.to_vec();
        raw.push(p.to_tokens().unwrap().map(u32::from));
        raw
    }

    #[test]
    fn interleaved_lanes_use_explicit_parents_and_only_lane_zero_changes_selection() {
        let mut doc = Document::new();
        let mut store = RawHistoryStore::new();
        doc.prompt = "未被其他lane覆盖的当前参数".into();
        doc.settings.seed = 700;
        let original_settings = doc.settings.clone();
        let settings = Settings {
            parallel_sequences: 3,
            seed: 100,
            ..Default::default()
        };
        let mut session = ParallelHistory::new(&doc, 0, "同一提示词".into(), settings).unwrap();
        let a = Primitive::background([20, 40, 60]);
        let raw1 = callback_raw(&[BOS_RECORD], &a);
        let lane1 = session
            .submit(
                &mut doc,
                &mut store,
                1,
                a.to_tokens().unwrap().map(u32::from),
                &raw1,
            )
            .unwrap();
        assert_eq!(doc.current, 0);
        assert!(doc.primitives.is_empty());
        assert_eq!(doc.prompt, "未被其他lane覆盖的当前参数");
        assert_eq!(doc.settings, original_settings);
        let b = Primitive::background([100, 120, 140]);
        let raw0 = callback_raw(&[BOS_RECORD], &b);
        let lane0 = session
            .submit(
                &mut doc,
                &mut store,
                0,
                b.to_tokens().unwrap().map(u32::from),
                &raw0,
            )
            .unwrap();
        assert_eq!(doc.current, lane0);
        assert_eq!(doc.settings.seed, 100);
        let follow_canvas = doc.primitives.clone();
        let follow_prompt = doc.prompt.clone();
        let follow_settings = doc.settings.clone();
        let raw2 = callback_raw(&[BOS_RECORD], &Primitive::background([200, 80, 20]));
        let lane2 = session
            .submit(&mut doc, &mut store, 2, raw2[1], &raw2)
            .unwrap();
        assert_eq!(doc.current, lane0);
        assert_eq!(doc.primitives, follow_canvas);
        assert_eq!(doc.prompt, follow_prompt);
        assert_eq!(doc.settings, follow_settings);
        let c = rect(90.0);
        let next1 = callback_raw(&raw1, &c);
        let lane1_second = session
            .submit(
                &mut doc,
                &mut store,
                1,
                c.to_tokens().unwrap().map(u32::from),
                &next1,
            )
            .unwrap();
        assert_eq!(doc.nodes[lane1_second].parent, Some(lane1));
        assert_eq!(doc.nodes[lane0].parent, Some(0));
        assert_eq!(doc.nodes[lane2].parent, Some(0));
        assert_eq!(doc.nodes[0].children.len(), 3);
        assert_eq!(doc.current, lane0);
        assert_eq!(session.generated, vec![1, 2, 1]);
        assert_eq!(doc.nodes[lane1_second].settings.seed, 101);
        assert!(doc.nodes[lane2].title.starts_with("并行 3/3"));
        let mut ids = std::collections::HashSet::new();
        for node in [lane0, lane1, lane2, lane1_second] {
            if let crate::document::Patch::Add { primitive, .. } = &doc.nodes[node].patches[0] {
                assert!(ids.insert(primitive.id));
            }
        }
    }

    #[test]
    fn per_lane_history_and_regions_are_isolated_when_finishing_in_any_order() {
        let mut doc = Document::new();
        doc.commit_canvas(
            "共同底图",
            OperationSource::User,
            vec![Primitive::background([20, 40, 60])],
        )
        .unwrap();
        let origin = doc.current;
        let mut store = RawHistoryStore::new();
        let base = records(&doc.primitives);
        store.finish(&doc, origin, &base).unwrap();
        let mut session = ParallelHistory::new(
            &doc,
            origin,
            "同图不同区域".into(),
            Settings {
                parallel_sequences: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let mut a = base.clone();
        a.push(marker_record(Stage::Coarse, 0).unwrap());
        a.push(rect(60.).to_tokens().unwrap().map(u32::from));
        let a_node = session
            .submit(&mut doc, &mut store, 0, *a.last().unwrap(), &a)
            .unwrap();
        let mut b = base.clone();
        b.push(marker_record(Stage::Coarse, 1).unwrap());
        b.push(rect(190.).to_tokens().unwrap().map(u32::from));
        let b_node = session
            .submit(&mut doc, &mut store, 1, *b.last().unwrap(), &b)
            .unwrap();
        assert_eq!(session.region_for_lane(0).unwrap().index, Some(0));
        assert_eq!(session.region_for_lane(1).unwrap().index, Some(1));
        let mut finished_b = b.clone();
        finished_b.push(END_RECORD);
        session
            .finish_lane(&doc, &mut store, 1, &finished_b)
            .unwrap();
        assert_eq!(doc.current, a_node);
        assert_eq!(store.context_for(&doc, a_node).unwrap(), Some(a.clone()));
        assert_eq!(store.context_for(&doc, b_node).unwrap(), Some(finished_b));
        assert_eq!(store.context_for(&doc, origin).unwrap(), Some(base));
        session.finish_lane(&doc, &mut store, 0, &a).unwrap();
        assert!(session.all_terminal());
    }

    #[test]
    fn zero_output_lanes_keep_private_contexts_and_publish_only_lane_zero_overlay() {
        let mut doc = Document::new();
        let origin = 0;
        let mut store = RawHistoryStore::new();
        store.finish(&doc, origin, &[BOS_RECORD]).unwrap();
        let mut session = ParallelHistory::new(
            &doc,
            origin,
            "区域选择后立即停止".into(),
            Settings {
                parallel_sequences: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let a = vec![BOS_RECORD, marker_record(Stage::Coarse, 0).unwrap()];
        let b = vec![BOS_RECORD, marker_record(Stage::Coarse, 7).unwrap()];
        session.finish_lane(&doc, &mut store, 1, &b).unwrap();
        assert_eq!(store.context_for(&doc, origin).unwrap(), Some(vec![BOS_RECORD]));
        assert_eq!(session.displayed_context_for(&doc, &store).unwrap(), Some(vec![BOS_RECORD]),
            "a non-displayed lane must not supply shared-origin view context");
        session.finish_lane(&doc, &mut store, 0, &a).unwrap();
        assert_eq!(doc.nodes.len(), 1);
        assert_eq!(doc.current, origin);
        assert!(doc.primitives.is_empty());
        assert_eq!(
            store.context_for(&doc, origin).unwrap(),
            Some(a.clone())
        );
        assert_eq!(session.displayed_context_for(&doc, &store).unwrap(), Some(a.clone()));
        assert_eq!(session.context_for_lane(&doc, &store, 0).unwrap(), Some(a));
        assert_eq!(session.context_for_lane(&doc, &store, 1).unwrap(), Some(b));
        assert_eq!(session.heads, vec![origin, origin]);
        assert_eq!(session.generated, vec![0, 0]);
    }

    #[test]
    fn browsing_and_explicit_freeze_do_not_interrupt_generation_or_steal_the_canvas() {
        let mut doc = Document::new();
        let mut store = RawHistoryStore::new();
        let mut session = ParallelHistory::new(&doc, 0, "继续生成".into(),
            Settings { parallel_sequences: 2, ..Default::default() }).unwrap();
        assert!(session.is_following(&doc));
        let background = Primitive::background([20, 40, 60]);
        let raw = callback_raw(&[BOS_RECORD], &background);
        let first = session.submit(&mut doc, &mut store, 0, raw[1], &raw).unwrap();
        assert_eq!(doc.current, first);

        // Freeze even while the selected node still equals the live head. A
        // pointer gesture can now survive subsequent model callbacks.
        session.set_follow_live(false);
        assert!(!session.is_following(&doc));
        let snapshot = doc.primitives.clone();
        let second_raw = callback_raw(&raw, &rect(90.));
        let second = session.submit(&mut doc, &mut store, 0, *second_raw.last().unwrap(), &second_raw).unwrap();
        assert_eq!(doc.nodes[second].parent, Some(first));
        assert_eq!(doc.current, first);
        assert_eq!(doc.primitives, snapshot);
        assert_eq!(session.statuses[0], LaneStatus::Running);

        // Browsing elsewhere also defeats following even if enabled.
        doc.jump(0).unwrap();
        session.set_follow_live(true);
        assert!(!session.is_following(&doc));
        let third_raw = callback_raw(&second_raw, &rect(120.));
        let third = session.submit(&mut doc, &mut store, 0, *third_raw.last().unwrap(), &third_raw).unwrap();
        assert_eq!(doc.nodes[third].parent, Some(second));
        assert_eq!(doc.current, 0);
        assert!(doc.primitives.is_empty());

        // The user explicitly returns to the live head and enables follow.
        doc.jump(third).unwrap();
        session.set_follow_live(true);
        assert!(session.is_following(&doc));
        let fourth_raw = callback_raw(&third_raw, &rect(150.));
        let fourth = session.submit(&mut doc, &mut store, 0, *fourth_raw.last().unwrap(), &fourth_raw).unwrap();
        assert_eq!(doc.current, fourth);
        assert_eq!(session.generated[0], 4);
    }

    #[test]
    fn single_lane_follows_normally_without_parallel_title_and_preserves_its_prompt() {
        let mut doc = Document::new();
        let mut store = RawHistoryStore::new();
        let settings = Settings { parallel_sequences: 1, seed: 123, ..Default::default() };
        let mut session = ParallelHistory::new(&doc, 0, "单路作品".into(), settings).unwrap();
        let background = Primitive::background([20, 40, 60]);
        let first_raw = callback_raw(&[BOS_RECORD], &background);
        let first = session.submit(&mut doc, &mut store, 0, first_raw[1], &first_raw).unwrap();
        let final_raw = callback_raw(&first_raw, &rect(90.));
        let second = session.submit(&mut doc, &mut store, 0, *final_raw.last().unwrap(), &final_raw).unwrap();
        assert_eq!(doc.current, second);
        assert_eq!(doc.nodes[second].parent, Some(first));
        assert_eq!(doc.prompt, "单路作品");
        assert_eq!(doc.settings.seed, 123);
        assert_eq!(doc.settings.parallel_sequences, 1);
        assert!(!doc.nodes[first].title.contains("并行"));
        assert!(!doc.nodes[second].title.contains("并行"));
        session.finish_lane(&doc, &mut store, 0, &final_raw).unwrap();
        assert_eq!(session.displayed_context_for(&doc, &store).unwrap(), Some(final_raw));
    }

    #[test]
    fn zero_output_overlay_survives_new_requests_and_manual_edits_without_rewriting_children() {
        let mut doc = Document::new();
        doc.commit_canvas("共同背景", OperationSource::User,
            vec![Primitive::background([20, 40, 60])]).unwrap();
        let origin = doc.current;
        let mut base = records(&doc.primitives);
        for index in 0..15 { base.extend([marker_record(Stage::Coarse, index).unwrap(), END_RECORD]); }
        base.push(marker_record(Stage::Coarse, 15).unwrap());
        let mut store = RawHistoryStore::new();
        store.finish(&doc, origin, &base).unwrap();

        let mut previous_shape = rect(450.);
        previous_shape.y = 450.;
        let old_raw = callback_raw(&base, &previous_shape);
        let old_child = doc.commit_generated_at(origin, previous_shape, "旧分支".into(),
            Settings::default()).unwrap();
        store.attach(&doc, old_child, &old_raw).unwrap();

        let mut session = ParallelHistory::new(&doc, origin, "零输出阶段收尾".into(),
            Settings { parallel_sequences: 2, ..Default::default() }).unwrap();
        let mut ended = base.clone();
        ended.push(END_RECORD);
        session.finish_lane(&doc, &mut store, 0, &ended).unwrap();
        assert_eq!(store.context_for(&doc, origin).unwrap(), Some(ended.clone()));
        assert_eq!(store.context_for(&doc, old_child).unwrap(), Some(old_raw.clone()),
            "an existing child must continue resolving against immutable origin entries");

        let mut other = base.clone();
        other.extend([crate::region_state::INTERRUPT_RECORD,
            marker_record(Stage::Coarse, 14).unwrap(), END_RECORD]);
        session.finish_lane(&doc, &mut store, 1, &other).unwrap();
        assert_eq!(session.context_for_lane(&doc, &store, 1).unwrap(), Some(other));
        assert_eq!(store.context_for(&doc, origin).unwrap(), Some(ended.clone()));

        // Replacing the old generation session must not lose its lane0 tail.
        let mut next_session = ParallelHistory::new(&doc, origin, "继续细网格".into(),
            Settings { parallel_sequences: 1, ..Default::default() }).unwrap();
        let resumed = next_session.context_for_lane(&doc, &store, 0).unwrap().unwrap();
        assert_eq!(resumed, ended);
        assert_eq!(DrawingRegion::from_history(&resumed).unwrap().stage, Stage::Fine);
        let mut fine = resumed.clone();
        fine.push(marker_record(Stage::Fine, 0).unwrap());
        fine.push(rect(30.).to_tokens().unwrap().map(u32::from));
        let fine_node = next_session.submit(&mut doc, &mut store, 0, *fine.last().unwrap(), &fine).unwrap();
        assert_eq!(store.context_for(&doc, fine_node).unwrap(), Some(fine));
        assert_eq!(store.context_for(&doc, old_child).unwrap(), Some(old_raw));

        doc.jump(origin).unwrap();
        let mut edited = doc.primitives.clone();
        edited[0].color = [100, 120, 140];
        let manual = doc.commit_canvas("编辑背景", OperationSource::User, edited).unwrap();
        let region = store.reconcile_manual(&doc, origin, manual).unwrap();
        assert_eq!(region.stage, Stage::Fine, "manual edit must inherit the completed coarse boundary");
        assert_eq!(region.completed_coarse, 16);
        let manual_raw = store.context_for(&doc, manual).unwrap().unwrap();
        assert_eq!(session.displayed_context_for(&doc, &store).unwrap(), Some(manual_raw),
            "browsing an unrelated node must not use a lane's origin overlay");
    }

    #[test]
    fn invalid_duplicate_and_terminal_callbacks_never_create_phantom_nodes() {
        let mut doc = Document::new();
        let mut store = RawHistoryStore::new();
        let mut session = ParallelHistory::new(
            &doc,
            0,
            "验证回调事务".into(),
            Settings {
                parallel_sequences: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let p = Primitive::background([20, 40, 60]);
        let raw = callback_raw(&[BOS_RECORD], &p);
        assert!(session
            .submit(
                &mut doc,
                &mut store,
                0,
                p.to_tokens().unwrap().map(u32::from),
                &[BOS_RECORD]
            )
            .is_err());
        assert_eq!(doc.nodes.len(), 1);
        assert_eq!(session.generated[0], 0);
        let head = session
            .submit(
                &mut doc,
                &mut store,
                0,
                p.to_tokens().unwrap().map(u32::from),
                &raw,
            )
            .unwrap();
        let count = doc.nodes.len();
        assert!(session
            .submit(
                &mut doc,
                &mut store,
                0,
                p.to_tokens().unwrap().map(u32::from),
                &raw
            )
            .is_err());
        assert_eq!(doc.nodes.len(), count);
        assert_eq!(session.heads[0], head);
        session.finish_lane(&doc, &mut store, 0, &raw).unwrap();
        assert!(session.finish_lane(&doc, &mut store, 0, &raw).is_err());
        assert!(session.fail_lane(0, "重复终结").is_err());
        session.fail_lane(1, "该lane独立失败").unwrap();
        assert!(session
            .finish_lane(&doc, &mut store, 1, &[BOS_RECORD])
            .is_err());
        assert!(session
            .submit(
                &mut doc,
                &mut store,
                1,
                p.to_tokens().unwrap().map(u32::from),
                &raw
            )
            .is_err());
        assert!(session.all_terminal());
        assert_eq!(doc.nodes.len(), count);
    }
    #[test]
    fn global_redraw_parallel_lanes_preserve_the_prefix_and_finish_without_a_synthetic_global_end(){
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let mut canvas=vec![Primitive::background([240;3])];
        canvas.extend((0..84).map(|i|rect(20.+i as f32*4.)));
        doc.commit_canvas("global prefix",OperationSource::User,canvas).unwrap();let original=doc.current;let before=doc.primitives.clone();
        let origin=doc.commit_control_canvas("global redraw",doc.primitives.clone()).unwrap();let mut raw=records(&doc.primitives);let ids=doc.primitives.iter().map(|p|p.id).collect::<Vec<_>>();
        store.set_control_overlay_with_ids(&doc,origin,&raw,&ids).unwrap();store.begin_redraw_origin(&doc,origin,&[(1,0)],&raw,&ids).unwrap();
        let mut session=ParallelHistory::new(&doc,origin,String::new(),Settings{parallel_sequences:2,..Default::default()}).unwrap();
        session.initialize_explicit_model_input(&doc,&store,&raw,&ids,Vec::new(),vec![(1,0)]).unwrap();
        for lane in [1,0]{
            let p=rect(300.+lane as f32*20.);let row=p.to_tokens().unwrap().map(u32::from);raw=records(&before);raw.push(row);
            session.submit(&mut doc,&mut store,lane,row,&raw).unwrap();session.finish_lane(&doc,&mut store,lane,&raw).unwrap();
        }
        assert!(session.all_redraw_regions_complete());assert_ne!(session.heads[0],session.heads[1]);
        assert_eq!(doc.materialize(original).unwrap(),before);assert_eq!(doc.materialize(origin).unwrap(),before);
        for lane in 0..2{let history=store.context_for(&doc,session.heads[lane]).unwrap().unwrap();assert_eq!(history.len(),87);
            assert!(history.iter().all(|row|!crate::region_state::is_marker(row)&&row!=&crate::region_state::END_RECORD));}
    }
    #[test]
    fn global_redraw_empty_origin_accepts_one_background_per_independent_lane(){
        let mut doc=Document::new();let mut store=RawHistoryStore::new();let origin=doc.commit_control_canvas("empty global redraw",Vec::new()).unwrap();
        store.set_control_overlay_with_ids(&doc,origin,&[BOS_RECORD],&[]).unwrap();store.begin_redraw_origin(&doc,origin,&[(1,0)],&[BOS_RECORD],&[]).unwrap();
        let mut session=ParallelHistory::new(&doc,origin,String::new(),Settings{parallel_sequences:2,..Default::default()}).unwrap();
        session.initialize_explicit_model_input(&doc,&store,&[BOS_RECORD],&[],Vec::new(),vec![(1,0)]).unwrap();
        for lane in 0..2{let bg=Primitive::background([200+lane as u8*10;3]);let row=bg.to_tokens().unwrap().map(u32::from);let raw=vec![BOS_RECORD,row];
            let node=session.submit(&mut doc,&mut store,lane,row,&raw).unwrap();assert_eq!(doc.materialize(node).unwrap().len(),1);
            let mut duplicate=raw.clone();duplicate.push(row);assert!(session.submit(&mut doc,&mut store,lane,row,&duplicate).is_err());
            session.finish_lane(&doc,&mut store,lane,&raw).unwrap();}
        assert!(!session.all_redraw_regions_complete());assert!(doc.materialize(origin).unwrap().is_empty());
    }
}
