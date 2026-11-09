//! Schema-3 mapped views are checked against stable IDs, never a full-canvas
//! projection which would incorrectly include deferred/future primitives.
use super::*;

// A bounded recent-prefix cache removes repeated walks through a generated
// branch's whole ancestry. Bound both nodes and IDs, not just node count: a
// hostile file must not turn N small deltas into N full canvas copies.
#[derive(Default)]struct IdPrefixCache{
    nodes:HashMap<NodeId,Vec<u64>>,order:std::collections::VecDeque<NodeId>,ids:usize,
}
impl IdPrefixCache{
    const MAX_NODES:usize=256;
    const MAX_IDS:usize=262_144;
    fn insert(&mut self,node:NodeId,ids:Vec<u64>){
        if ids.len()>Self::MAX_IDS{return;}
        while self.nodes.len()>=Self::MAX_NODES||self.ids.saturating_add(ids.len())>Self::MAX_IDS{
            let Some(old)=self.order.pop_front()else{break};
            if let Some(old)=self.nodes.remove(&old){self.ids-=old.len();}
        }
        self.ids+=ids.len();self.nodes.insert(node,ids);self.order.push_back(node);
    }
}

struct ProjectIdNode{
    base:HistoryBase,keep:usize,len:usize,states:Vec<project_data::IdProjection>,end:project_data::IdProjection,
}
fn id_prefix(mut base:HistoryBase,index:usize,validated:&[Option<ProjectIdNode>],doc:&Document,
    projections:&[project_data::Projection],memo:&mut HashMap<ProjectPrefixKey,project_data::IdProjection>,work:&mut project_data::Work)
    ->Result<project_data::IdProjection,String>{
    let mut path=Vec::new();let state=loop{work.charge(1)?;match base{
        HistoryBase::Empty=>{if index!=0{return Err("模型ID映射keep超出空基底".into());}break project_data::IdProjection::default()},
        HistoryBase::Node(node)=>{
            if let Some(data)=validated.get(node).and_then(Option::as_ref){
                if index>data.len{return Err("模型ID映射keep超出基底".into());}
                if index==data.len{break data.end.clone();}
                if index>data.keep{break data.states[index-data.keep-1].clone();}
                let key=ProjectPrefixKey{node,index,canvas:false};if let Some(state)=memo.get(&key){break state.clone();}
                path.push(key);base=data.base;
            }else{base=HistoryBase::Canvas(node);}
        },
        HistoryBase::Canvas(node)=>{
            let projection=projections.get(node).ok_or("模型ID画布基底不存在")?;
            if index>projection.count{return Err("模型ID映射keep超出画布基底".into());}
            if index==0{break project_data::IdProjection::default();}
            if index==projection.count{break projection.ordered_ids().clone();}
            let key=ProjectPrefixKey{node,index,canvas:true};if let Some(state)=memo.get(&key){break state.clone();}
            work.charge(projection.count.saturating_mul(65).saturating_add(index))?;
            let canvas=doc.materialize(node)?;let mut state=project_data::IdProjection::default();
            for primitive in canvas.iter().take(index){state.push(primitive.id)?;}
            path.push(key);break state;
        },
    }};
    if memo.len().checked_add(path.len()).is_none_or(|n|n>500_000){return Err("模型ID前缀依赖超过安全预算".into());}
    memo.try_reserve(path.len()).map_err(|_|"内存不足，无法验证对象编号前缀".to_owned())?;
    for key in path{memo.insert(key,state.clone());}Ok(state)
}

fn checked_canvas(doc:&Document,node:NodeId,work:&mut project_data::Work)->Result<Vec<Primitive>,String>{
    let state=doc.nodes.get(node).ok_or("模型ID映射引用不存在的状态")?;
    work.charge(state.primitive_count.saturating_add(1))?;
    doc.materialize(node)
}

fn resolve_ids(
    store:&RawHistoryStore,doc:&Document,mut cursor:HistoryBase,
    current:NodeId,canvas:&[Primitive],cache:&IdPrefixCache,validated:&[Option<ProjectIdNode>],
    projections:&[project_data::Projection],work:&mut project_data::Work,
)->Result<Vec<u64>,String>{
    let mut path=Vec::new();
    let mut ids=loop{
        work.charge(1)?;
        match cursor{
            HistoryBase::Node(node)=>{
                if node>=doc.nodes.len(){return Err("模型ID基底状态不存在".into());}
                if let Some(ids)=cache.nodes.get(&node){
                    work.charge(ids.len())?;
                    let mut copy=Vec::new();copy.try_reserve_exact(ids.len()).map_err(|_|"内存不足，无法复用已验证模型ID前缀".to_owned())?;
                    copy.extend_from_slice(ids);break copy;
                }
                if validated.get(node).and_then(Option::as_ref).is_some_and(|ids|
                    projections.get(node).is_some_and(|p|p.ordered_ids().same(&ids.end))){
                    if node==current{break canvas.iter().map(|p|p.id).collect::<Vec<_>>();}
                    break checked_canvas(doc,node,work)?.into_iter().map(|p|p.id).collect::<Vec<_>>();
                }
                if let Some(delta)=store.model_ids.get(&node){
                    // This ordering is checked for every mapping before any
                    // resolution, so neither cycles nor future chains run.
                    path.push(delta);cursor=delta.base;
                }else{
                    if node==current{break canvas.iter().map(|p|p.id).collect::<Vec<_>>();}
                    break checked_canvas(doc,node,work)?.into_iter().map(|p|p.id).collect::<Vec<_>>();
                }
            },
            HistoryBase::Canvas(node)=>{
                if node==current{break canvas.iter().map(|p|p.id).collect::<Vec<_>>();}
                break checked_canvas(doc,node,work)?.into_iter().map(|p|p.id).collect::<Vec<_>>();
            },
            HistoryBase::Empty=>break Vec::new(),
        }
    };
    for delta in path.into_iter().rev(){
        if delta.keep>ids.len(){return Err("模型ID映射keep超出基底".into());}
        let length=delta.keep.checked_add(delta.tail.len()).ok_or("模型ID映射长度溢出")?;
        if length>project_data::MAX_CANVAS{return Err("模型ID映射超过单画布对象上限".into());}
        work.charge(delta.tail.len().saturating_add(1))?;
        ids.truncate(delta.keep);ids.try_reserve_exact(delta.tail.len()).map_err(|_|"内存不足，无法验证模型ID增量".to_owned())?;
        ids.extend_from_slice(&delta.tail);
    }
    Ok(ids)
}

fn mapped_projection(canvas:&[Primitive],ids:&[u64],work:&mut project_data::Work)->Result<project_data::Projection,String>{
    if ids.len()>canvas.len(){return Err("模型ID映射数量超过当前画布".into());}
    work.charge(ids.len().saturating_add(canvas.len()))?;
    let by_id=canvas.iter().map(|p|(p.id,p)).collect::<HashMap<_,_>>();
    let mut seen=HashSet::new();seen.try_reserve(ids.len()).map_err(|_|"内存不足，无法验证模型ID唯一性".to_owned())?;
    let mut projection=project_data::Projection::default();
    for id in ids{
        if *id==0||!seen.insert(*id){return Err("模型ID映射包含零或重复对象编号".into());}
        let primitive=by_id.get(id).ok_or("模型ID映射引用当前画布中不存在的图元")?;
        projection.push(&primitive.to_tokens()?.map(u32::from))?;
    }
    Ok(projection)
}

fn checked_raw_phase(states:&[Option<ProjectRawNode>],node:NodeId,evidence_only:bool,work:&mut project_data::Work)->Result<usize,String>{
    work.charge(1)?;
    let state=states.get(node).and_then(Option::as_ref).ok_or("阶段验证缺少已验证模型历史")?;
    Ok(state.end.phase(evidence_only))
}

pub(super) fn validate_mapped_projects(
    store:&RawHistoryStore,doc:&Document,projections:&[project_data::Projection],
    raw_project_states:&[Option<ProjectRawNode>],overlays:&HashMap<NodeId,ProjectRawState>,
    budget:&mut project_data::Budget,work:&mut project_data::Work,
)->Result<(),String>{
    budget.charge(project_data::MODEL_IDS,store.model_ids.len())?;
    // Validate literal metadata before max()/inheritance could hide a bad
    // value, and before following any user-controlled mapping chain.
    for(&node,delta)in &store.model_ids{
        if node>=doc.nodes.len()||!matches!(delta.phase_floor,1|4|8){return Err("模型ID映射状态或阶段游标无效".into());}
        match delta.base{
            HistoryBase::Node(base)if base>=node=>return Err("模型ID映射循环或引用未来状态".into()),
            HistoryBase::Canvas(base)if base>node=>return Err("模型ID映射画布基底引用未来状态".into()),
            HistoryBase::Empty if delta.keep!=0=>return Err("空模型ID基底不能保留前缀".into()),_=>{},
        }
        budget.charge(project_data::RECORDS,delta.tail.len())?;
    }
    let mut nodes=store.model_ids.keys().copied().collect::<Vec<_>>();nodes.sort_unstable();
    let mut id_states=(0..doc.nodes.len()).map(|_|None).collect::<Vec<Option<ProjectIdNode>>>();
    let mut id_memo=HashMap::new();
    for &node in &nodes{
        let delta=&store.model_ids[&node];
        let mut state=id_prefix(delta.base,delta.keep,&id_states,doc,projections,&mut id_memo,work)?;
        let len=delta.keep.checked_add(delta.tail.len()).ok_or("模型ID映射长度溢出")?;
        if len>project_data::MAX_CANVAS{return Err("模型ID映射超过单画布对象上限".into());}
        let mut states=Vec::new();states.try_reserve_exact(delta.tail.len()).map_err(|_|"内存不足，无法验证对象编号增量".to_owned())?;
        for &id in &delta.tail{work.charge(1)?;state.push(id)?;states.push(state.clone());}
        id_states[node]=Some(ProjectIdNode{base:delta.base,keep:delta.keep,len,states,end:state});
    }
    for &node in store.overlay_model_ids.keys().chain(store.overlay_phase_floors.keys()).chain(store.overlay_manual_reviews.keys()){
        if node>=doc.nodes.len()||!store.model_ids.contains_key(&node)||!store.control_overlays.contains_key(&node){
            return Err("模型ID覆盖层缺少对应状态、ID快照或控制历史".into());
        }
    }
    for(&node,&floor)in &store.overlay_phase_floors{
        if !matches!(floor,1|4|8)||floor<store.model_ids[&node].phase_floor{return Err("模型覆盖层阶段游标无效或倒退".into());}
    }
    if !store.model_ids.is_empty(){
        // Ownership may move backwards; the actual generation phase cannot.
        // Inherit from the document parent, not an arbitrary ID-delta base,
        // and ignore parent overlays which belong only to that exact node.
        let mut immutable_floors=vec![1usize;doc.nodes.len()];
        for (node,state) in doc.nodes.iter().enumerate(){
            work.charge(1)?;let inherited=state.parent.map_or(1,|parent|immutable_floors[parent]);
            let mut own=1;
            if store.entries.contains_key(&node){let explicit=store.model_ids.contains_key(&node)
                ||state.source==OperationSource::Model||store.entries.get(&node).is_some_and(|d|matches!(d.base,HistoryBase::Canvas(base)if base==node));
                own=checked_raw_phase(raw_project_states,node,!explicit,work)?;
            }
            let restart=store.mode_evidence.get(&node).is_some_and(|entry|entry.normal_restart
                ||entry.redraw_scope.as_ref().is_some_and(|scope|scope.contains(&(1,0))));
            let required=if restart{own}else{inherited.max(own)};
            immutable_floors[node]=if let Some(delta)=store.model_ids.get(&node){
                if delta.phase_floor<required{return Err("模型阶段游标低于父状态或实际模型历史".into());}delta.phase_floor
            }else{required};
        }
        for(&node,records)in &store.control_overlays{if let Some(delta)=store.model_ids.get(&node){
            work.charge(1)?;let _=records;
            let phase=overlays.get(&node).ok_or("阶段验证缺少已验证控制历史")?.phase(false);
            let required=delta.phase_floor.max(phase);
            if store.overlay_phase_floors.get(&node).copied().unwrap_or(delta.phase_floor)<required{
                return Err("模型覆盖层阶段游标低于实际控制历史".into());
            }
        }}
    }
    for(&node,regions)in &store.overlay_manual_reviews{
        budget.charge(project_data::RECORDS,regions.len())?;
        if regions.len()>80{return Err("模型覆盖层用户复查区域过多".into());}
        let mut unique=HashSet::new();
        for &(side,index)in regions{if !matches!(side,4|8)||index>=side*side||!unique.insert((side,index)){
            return Err(format!("状态{node}的模型覆盖层用户复查区域无效"));}}
    }
    let mut cache=IdPrefixCache::default();
    for node in nodes{
        let canvas_projection=projections.get(node).ok_or("模型ID映射缺少画布摘要")?;
        let signature=&id_states[node].as_ref().ok_or("缺少已验证对象编号摘要")?.end;
        let mut canvas=None;
        let projection=if canvas_projection.ordered_ids().same(signature){
            // Every ID, its order, and every canvas patch were already checked.
            // Reuse its token digest rather than expanding an entire 1000-token
            // canvas once per generated token on each parallel branch.
            work.charge(1)?;canvas_projection.clone()
        }else{
            let full=checked_canvas(doc,node,work)?;
            let ids=resolve_ids(store,doc,HistoryBase::Node(node),node,&full,&cache,&id_states,projections,work)?;
            let projection=mapped_projection(&full,&ids,work)?;canvas=Some(full);cache.insert(node,ids);projection
        };
        if let Some(raw)=raw_project_states.get(node).and_then(Option::as_ref){
            if !projection.same(&raw.end.projection){return Err("模型ID顺序的九属性投影与模型历史不一致".into());}
        }
        if let Some(raw)=overlays.get(&node){
            let overlay_projection=if let Some(ids)=store.overlay_model_ids.get(&node){
                budget.charge(project_data::RECORDS,ids.len())?;work.charge(ids.len())?;
                let mut signature=project_data::IdProjection::default();for &id in ids{signature.push(id)?;}
                if canvas_projection.ordered_ids().same(&signature){canvas_projection.clone()}
                else{if canvas.is_none(){canvas=Some(checked_canvas(doc,node,work)?);}
                    mapped_projection(canvas.as_ref().unwrap(),ids,work)?}
            }else{projection};
            if !overlay_projection.same(&raw.projection){return Err("模型ID覆盖层顺序与控制历史不一致".into());}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests{
    use super::*;
    fn fixture()->(Document,RawHistoryStore,Vec<Option<ProjectRawNode>>,HashMap<NodeId,ProjectRawState>,NodeId){
        let mut doc=Document::new();let p=Primitive::new(ShapeKind::Rectangle,30.,30.,20.,20.,[80,120,160]);
        let mut q=p.clone();q.color=[160,120,80];
        let node=doc.commit_canvas("fixture",OperationSource::User,vec![p,q]).unwrap();
        let raw=vec![BOS_RECORD,doc.primitives[0].to_tokens().unwrap().map(u32::from),doc.primitives[1].to_tokens().unwrap().map(u32::from)];
        let mut state=ProjectRawState::default();for row in &raw{state.push(row).unwrap();}
        let mut store=RawHistoryStore::new();store.entries.insert(node,HistoryDelta{base:HistoryBase::Empty,keep:0,tail:raw});
        store.model_ids.insert(node,ModelIdDelta{base:HistoryBase::Empty,keep:0,tail:doc.primitives.iter().map(|p|p.id).collect(),phase_floor:1});
        let mut states=(0..doc.nodes.len()).map(|_|None).collect::<Vec<_>>();
        states[node]=Some(ProjectRawNode{base:HistoryBase::Empty,keep:0,len:state.raw_len,states:Vec::new(),end:state});
        (doc,store,states,HashMap::new(),node)
    }
    fn canvas_projections(doc:&Document)->Vec<project_data::Projection>{
        doc.validate_project(&mut project_data::Budget::new(project_data::Limits::default()),
            &mut project_data::Work::new(project_data::MAX_WORK)).unwrap()
    }
    fn check(doc:&Document,store:&RawHistoryStore,states:&[Option<ProjectRawNode>],overlays:&HashMap<NodeId,ProjectRawState>)->Result<(),String>{
        validate_mapped_projects(store,doc,&canvas_projections(doc),states,overlays,&mut project_data::Budget::new(project_data::Limits::default()),&mut project_data::Work::new(project_data::MAX_WORK))
    }
    #[test]
    fn mapped_ids_reject_unknown_duplicate_and_wrong_token_order(){
        let(doc,mut store,states,overlays,node)=fixture();assert!(check(&doc,&store,&states,&overlays).is_ok());
        let original=store.model_ids[&node].tail.clone();
        for ids in [vec![original[0],u64::MAX],vec![original[0],original[0]],vec![original[1],original[0]]]{
            store.model_ids.get_mut(&node).unwrap().tail=ids;assert!(check(&doc,&store,&states,&overlays).is_err());
        }
    }
    #[test]
    fn phase_literal_and_overlay_binding_are_checked_before_inheritance(){
        let(doc,mut store,states,mut overlays,node)=fixture();
        for floor in [0,2,16]{store.model_ids.get_mut(&node).unwrap().phase_floor=floor;assert!(check(&doc,&store,&states,&overlays).is_err());}
        store.model_ids.get_mut(&node).unwrap().phase_floor=1;
        store.overlay_phase_floors.insert(node,4);assert!(check(&doc,&store,&states,&overlays).is_err());
        let raw=store.entries[&node].tail.clone();store.control_overlays.insert(node,raw);overlays.insert(node,states[node].as_ref().unwrap().end.clone());
        assert!(check(&doc,&store,&states,&overlays).is_ok());
        store.overlay_phase_floors.insert(node,0);assert!(check(&doc,&store,&states,&overlays).is_err());
    }
    #[test]
    fn budget_is_charged_for_materialization_mapping_resolution_and_projection(){
        let(doc,store,states,overlays,_)=fixture();
        assert!(validate_mapped_projects(&store,&doc,&canvas_projections(&doc),&states,&overlays,&mut project_data::Budget::new(project_data::Limits::default()),&mut project_data::Work::new(1)).is_err());
    }
    #[test]
    fn phase_cannot_retreat_below_immutable_parent_or_real_raw_phase(){
        let(mut doc,mut store,mut states,overlays,parent)=fixture();
        store.model_ids.get_mut(&parent).unwrap().phase_floor=8;
        let mut canvas=doc.primitives.clone();canvas[0].color=[30,60,90];
        let child=doc.commit_canvas("child",OperationSource::User,canvas).unwrap();
        let mut raw=vec![BOS_RECORD];raw.extend(doc.primitives.iter().map(|p|p.to_tokens().unwrap().map(u32::from)));
        let mut state=ProjectRawState::default();for row in &raw{state.push(row).unwrap();}
        store.entries.insert(child,HistoryDelta{base:HistoryBase::Empty,keep:0,tail:raw});
        store.model_ids.insert(child,ModelIdDelta{base:HistoryBase::Empty,keep:0,tail:doc.primitives.iter().map(|p|p.id).collect(),phase_floor:1});
        states.push(Some(ProjectRawNode{base:HistoryBase::Empty,keep:0,len:state.raw_len,states:Vec::new(),end:state}));
        assert!(check(&doc,&store,&states,&overlays).is_err());
        store.model_ids.get_mut(&child).unwrap().phase_floor=8;assert!(check(&doc,&store,&states,&overlays).is_ok());

        let(doc,mut store,mut states,overlays,node)=fixture();
        let mut raw=vec![BOS_RECORD,crate::region_state::marker_record(Stage::Fine,0).unwrap()];
        raw.extend(doc.primitives.iter().map(|p|p.to_tokens().unwrap().map(u32::from)));
        let mut state=ProjectRawState::default();for row in &raw{state.push(row).unwrap();}
        store.entries.get_mut(&node).unwrap().tail=raw;
        states[node]=Some(ProjectRawNode{base:HistoryBase::Empty,keep:0,len:state.raw_len,states:Vec::new(),end:state});
        assert!(check(&doc,&store,&states,&overlays).is_err());
        store.model_ids.get_mut(&node).unwrap().phase_floor=8;assert!(check(&doc,&store,&states,&overlays).is_ok());
    }
    #[test]
    fn overlay_phase_covers_real_markers_but_ignores_empty_interrupted_pair(){
        let(doc,mut store,states,mut overlays,node)=fixture();
        let mut raw=store.entries[&node].tail.clone();
        raw.extend([crate::region_state::marker_record(Stage::Fine,0).unwrap(),crate::region_state::INTERRUPT_RECORD]);
        let mut state=ProjectRawState::default();for row in &raw{state.push(row).unwrap();}
        store.control_overlays.insert(node,raw);overlays.insert(node,state);
        store.overlay_phase_floors.insert(node,1);assert!(check(&doc,&store,&states,&overlays).is_ok());
        let mut raw=vec![BOS_RECORD,crate::region_state::marker_record(Stage::Fine,0).unwrap()];
        raw.extend(doc.primitives.iter().map(|p|p.to_tokens().unwrap().map(u32::from)));
        let mut state=ProjectRawState::default();for row in &raw{state.push(row).unwrap();}
        store.control_overlays.insert(node,raw);overlays.insert(node,state);
        store.overlay_phase_floors.insert(node,4);assert!(check(&doc,&store,&states,&overlays).is_err());
        store.overlay_phase_floors.insert(node,8);assert!(check(&doc,&store,&states,&overlays).is_ok());
    }

    #[test]
    fn incremental_phase_matches_full_replay_for_every_valid_prefix(){
        let visible=Primitive::new(ShapeKind::Rectangle,30.,30.,20.,20.,[80,120,160]).to_tokens().unwrap().map(u32::from);
        let histories=[
            vec![BOS_RECORD,visible,
                crate::region_state::marker_record(Stage::Coarse,0).unwrap(),crate::region_state::INTERRUPT_RECORD,
                crate::region_state::marker_record(Stage::Coarse,1).unwrap(),visible,crate::region_state::END_RECORD,
                crate::region_state::marker_record(Stage::Fine,0).unwrap(),crate::region_state::INTERRUPT_RECORD,
                crate::region_state::marker_record(Stage::Fine,1).unwrap(),crate::region_state::END_RECORD],
            vec![BOS_RECORD,crate::region_state::marker_record(Stage::Coarse,0).unwrap(),[PAD;9],crate::region_state::INTERRUPT_RECORD,
                crate::region_state::marker_record(Stage::Coarse,1).unwrap(),crate::region_state::END_RECORD],
        ];
        for raw in histories{let mut state=ProjectRawState::default();
            for (index,row)in raw.iter().enumerate(){state.push(row).unwrap();
                for evidence_only in [false,true]{assert_eq!(state.phase(evidence_only),raw_phase(&raw[..=index],evidence_only).unwrap(),
                    "prefix {} evidence_only={evidence_only}",index+1);}
            }
        }
    }

    #[test]
    fn six_thousand_edit_nodes_validate_under_original_budget_without_ancestor_replay(){
        // This is a legal modification-mode pattern: editing the same canvas
        // keeps IDs stable while raw/token deltas inherit a long shared prefix.
        // The old validator replayed all 6000 ancestors for each node and
        // rejected it despite there being only 17 objects on each canvas.
        let mut doc=Document::new();
        let mut canvas=vec![Primitive::background([240;3])];
        for index in 0..16{canvas.push(Primitive::new(ShapeKind::Rectangle,16.+index as f32*20.,40.,12.,14.,[30,70,110]));}
        let first=doc.commit_canvas("修改模式起点",OperationSource::User,canvas).unwrap();
        let mut store=RawHistoryStore::new();let mut raw=vec![BOS_RECORD];
        raw.extend(doc.primitives.iter().map(|p|p.to_tokens().unwrap().map(u32::from)));
        let ids=doc.primitives.iter().map(|p|p.id).collect::<Vec<_>>();
        store.entries.insert(first,HistoryDelta{base:HistoryBase::Empty,keep:0,tail:raw.clone()});
        store.model_ids.insert(first,ModelIdDelta{base:HistoryBase::Empty,keep:0,tail:ids.clone(),phase_floor:1});
        for index in 0..6000{
            let parent=doc.current;let mut canvas=doc.primitives.clone();
            canvas.last_mut().unwrap().color=[20+(index%2)as u8*100,60,90];
            let node=doc.commit_canvas("调整图形颜色",OperationSource::User,canvas).unwrap();
            assert_ne!(node,parent);
            *raw.last_mut().unwrap()=doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from);
            store.entries.insert(node,HistoryDelta{base:HistoryBase::Node(parent),keep:raw.len()-1,tail:vec![*raw.last().unwrap()]});
            store.model_ids.insert(node,ModelIdDelta{base:HistoryBase::Node(parent),keep:ids.len(),tail:Vec::new(),phase_floor:1});
        }
        let limits=project_data::Limits::default();let mut budget=project_data::Budget::new(limits);
        let mut work=project_data::Work::new(limits.work);
        let projections=doc.validate_project(&mut budget,&mut work).unwrap();
        store.validate_project(&doc,&projections,&mut budget,&mut work).unwrap();
        // Optimizing the shared prefixes must not weaken leaf-ID validation.
        store.model_ids.get_mut(&doc.current).unwrap().tail.push(u64::MAX);
        let mut budget=project_data::Budget::new(limits);let mut work=project_data::Work::new(limits.work);
        assert!(store.validate_project(&doc,&projections,&mut budget,&mut work).is_err());
    }

    #[test]
    fn id_prefix_cache_has_both_node_and_memory_bounds(){
        let mut cache=IdPrefixCache::default();
        for node in 0..2000{cache.insert(node,vec![node as u64;3000]);
            assert!(cache.nodes.len()<=IdPrefixCache::MAX_NODES);
            assert!(cache.ids<=IdPrefixCache::MAX_IDS);
            assert_eq!(cache.ids,cache.nodes.values().map(Vec::len).sum::<usize>());
        }
        for node in 2000..3000{cache.insert(node,Vec::new());assert!(cache.nodes.len()<=IdPrefixCache::MAX_NODES);}
    }

    #[test]
    fn sixteen_parallel_thousand_primitive_branches_fit_original_integrity_budget(){
        let mut doc=Document::new();let mut store=RawHistoryStore::new();
        let origin=doc.commit_generated(Primitive::background([240;3])).unwrap();
        let initial=vec![BOS_RECORD,doc.primitives[0].to_tokens().unwrap().map(u32::from)];
        store.entries.insert(origin,HistoryDelta{base:HistoryBase::Empty,keep:0,tail:initial.clone()});
        store.model_ids.insert(origin,ModelIdDelta{base:HistoryBase::Empty,keep:0,tail:vec![doc.primitives[0].id],phase_floor:1});
        for lane in 0..16{doc.jump(origin).unwrap();let mut raw=initial.clone();let mut previous=(1,0);
            for index in 0..1000{
                let parent=doc.current;let keep=raw.len();
                let (side,cell)=if index<85{(1,0)}else if index<405{(4,(index-85)/20)}else{(8,(index-405)/10)};
                if (side,cell)!=previous{
                    if previous.0!=1{raw.push(crate::region_state::END_RECORD);}
                    raw.push(crate::region_state::marker_record(if side==4{Stage::Coarse}else{Stage::Fine},cell).unwrap());
                    previous=(side,cell);
                }
                let step=512./side as f32;
                let mut p=Primitive::new(ShapeKind::Rectangle,(cell%side)as f32*step+step/2.,(cell/side)as f32*step+step/2.,8.,8.,[30+lane*2,70,110]);
                p.stage_assignment=Some(crate::document::PrimitiveStageAssignment{
                    stage:match side{4=>crate::document::StageTier::Coarse,8=>crate::document::StageTier::Fine,_=>crate::document::StageTier::Global},
                    cell,queue_order:index as u64});
                let node=doc.commit_generated(p).unwrap();let p=doc.primitives.last().unwrap();raw.push(p.to_tokens().unwrap().map(u32::from));
                store.entries.insert(node,HistoryDelta{base:HistoryBase::Node(parent),keep,tail:raw[keep..].to_vec()});
                store.model_ids.insert(node,ModelIdDelta{base:HistoryBase::Node(parent),keep:doc.primitives.len()-1,tail:vec![p.id],phase_floor:side});
            }
        }
        assert_eq!(doc.nodes.len(),16002);
        let limits=project_data::Limits::default();let mut budget=project_data::Budget::new(limits);let mut work=project_data::Work::new(limits.work);
        let projections=doc.validate_project(&mut budget,&mut work).unwrap();
        store.validate_project(&doc,&projections,&mut budget,&mut work).unwrap();
        // Same IDs do not excuse a mismatching token row in the fast path.
        store.entries.get_mut(&doc.current).unwrap().tail.last_mut().unwrap()[6]=1;
        let mut budget=project_data::Budget::new(limits);let mut work=project_data::Work::new(limits.work);
        assert!(store.validate_project(&doc,&projections,&mut budget,&mut work).is_err());
    }
}
