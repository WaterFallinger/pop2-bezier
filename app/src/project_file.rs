//! Self-contained versioned project files; weights and device paths are excluded.
use crate::{document::{Document,DocumentProjectData,DocumentProjectRef,project_data::{self,Limits}},
    history_context::{RawHistoryStore,RawHistoryProjectData,RawHistoryProjectRef}};
use flate2::{bufread::GzDecoder,write::GzEncoder,Compression};
use serde::{Deserialize,Serialize};
use std::{fs::File,io::{self,BufReader,Read,Seek,SeekFrom,Write},path::Path};

const MAGIC:&[u8;8]=b"POPPRJ\0\0";
const SCHEMA:u32=4;
const HEADER:u64=12;
#[derive(Serialize)]struct ProjectRef<'a>{schema:u32,document:DocumentProjectRef<'a>,history:RawHistoryProjectRef<'a>}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct ProjectData{schema:u32,document:DocumentProjectData,history:RawHistoryProjectData}

fn validate(doc:&Document,history:&RawHistoryStore,limits:Limits)->Result<(),String>{
    let mut budget=project_data::Budget::new(limits);let mut work=project_data::Work::new(limits.work);
    let projections=doc.validate_project(&mut budget,&mut work)
        .map_err(|e|format!("存档完整性检查／状态树：{e}"))?;
    history.validate_project(doc,&projections,&mut budget,&mut work)
        .map_err(|e|format!("存档完整性检查／模型历史与对象映射：{e}"))
}

pub(crate) fn save_project(path:&Path,doc:&Document,history:&RawHistoryStore)->Result<(),String>{
    save_with_limits(path,doc,history,Limits::default())
}
fn save_with_limits(path:&Path,doc:&Document,history:&RawHistoryStore,limits:Limits)->Result<(),String>{
    validate(doc,history,limits)?;
    crate::atomic_file::write_atomic(path,|file|{
        file.write_all(MAGIC).and_then(|_|file.write_all(&SCHEMA.to_le_bytes())).map_err(|e|format!("无法写入项目头：{e}"))?;
        let compressed=LimitedWrite{inner:file,remaining:limits.compressed};
        let mut encoder=GzEncoder::new(compressed,Compression::fast());
        let json=LimitedWrite{inner:&mut encoder,remaining:limits.json};
        serde_json::to_writer(json,&ProjectRef{schema:SCHEMA,document:doc.project_ref(),history:history.project_ref()})
            .map_err(|e|format!("无法完整保存项目：{e}"))?;
        encoder.finish().map_err(|e|format!("项目压缩未完成：{e}"))?;
        Ok(())
    })
}

pub(crate) fn load_project(path:&Path)->Result<(Document,RawHistoryStore),String>{load_with_limits(path,Limits::default())}
fn load_with_limits(path:&Path,limits:Limits)->Result<(Document,RawHistoryStore),String>{
    let mut file=File::open(path).map_err(|e|format!("无法打开项目：{e}"))?;
    let length=file.metadata().map_err(|e|format!("无法读取项目大小：{e}"))?.len();
    if length>limits.compressed.saturating_add(HEADER){return Err("项目压缩文件超过安全上限".into());}
    let mut magic=[0u8;8];let mut version=[0u8;4];
    file.read_exact(&mut magic).and_then(|_|file.read_exact(&mut version)).map_err(|e|format!("项目文件头不完整：{e}"))?;
    if magic!=*MAGIC{return Err("这不是Primitive Painter项目文件".into());}
    let version=u32::from_le_bytes(version);
    if !matches!(version,1|2|3|4){return Err("不支持此项目文件版本".into());}
    // Our format has no optional gzip name/comment fields. Reject those before
    // the gzip library could allocate an unbounded header string.
    let mut gzip_header=[0u8;10];file.read_exact(&mut gzip_header).map_err(|e|format!("项目压缩头不完整：{e}"))?;
    if gzip_header[..4]!=[0x1f,0x8b,8,0]{return Err("项目压缩头不符合schema".into());}
    file.seek(SeekFrom::Start(HEADER)).map_err(|e|format!("无法读取项目内容：{e}"))?;
    let compressed=LimitedRead{inner:file,remaining:limits.compressed};
    let mut decoder=GzDecoder::new(BufReader::new(compressed));
    let checked=CheckedJsonRead{inner:&mut decoder,remaining:limits.json,in_string:false,escaped:false,string_bytes:0};
    let mut checked=BufReader::new(checked);
    let _budget=project_data::ReadGuard::begin(limits)?;
    let mut deserializer=serde_json::Deserializer::from_reader(&mut checked);
    let data=ProjectData::deserialize(&mut deserializer).map_err(|e|format!("项目内容无效：{e}"))?;
    deserializer.end().map_err(|e|format!("项目末尾或压缩校验无效：{e}"))?;
    drop(deserializer);drop(checked);
    let mut source=decoder.into_inner();let mut extra=[0u8;1];
    if source.read(&mut extra).map_err(|e|format!("无法校验项目末尾：{e}"))?!=0{return Err("项目含额外压缩成员或尾部数据".into());}
    if data.schema!=version{return Err("项目内容schema与文件头不一致".into());}
    if version>=2&&!data.history.has_manual_reviews(){return Err("项目缺少用户区域复查状态".into());}
    if version==1&&data.history.has_manual_reviews(){return Err("schema1项目不能包含schema2用户复查字段".into());}
    if version>=3&&!data.history.has_model_ids(){return Err("项目缺少schema3模型输入对象映射".into());}
    if version<3&&data.history.has_model_ids(){return Err("旧schema项目不能包含schema3模型输入对象映射".into());}
    if version==4&&!data.history.has_mode_evidence(){return Err("项目缺少schema4模式来源与完成证据".into());}
    if version<4&&data.history.has_mode_evidence(){return Err("旧schema项目不能包含schema4模式来源与完成证据".into());}
    let doc=Document::from_project_data(data.document)?;
    if version<3&&doc.has_stage_assignments(){return Err("旧schema项目不能包含schema3图元阶段归属".into());}
    let history=RawHistoryStore::from_project_data(data.history)?;
    validate(&doc,&history,limits)?;
    Ok((doc,history))
}

struct LimitedWrite<W>{inner:W,remaining:u64}
impl<W:Write>Write for LimitedWrite<W>{
    fn write(&mut self,bytes:&[u8])->io::Result<usize>{
        if bytes.len()as u64>self.remaining{return Err(io::Error::other("项目写入大小超过安全上限"));}
        let count=self.inner.write(bytes)?;self.remaining-=count as u64;Ok(count)
    }
    fn flush(&mut self)->io::Result<()>{self.inner.flush()}
}
struct LimitedRead<R>{inner:R,remaining:u64}
impl<R:Read>Read for LimitedRead<R>{
    fn read(&mut self,bytes:&mut[u8])->io::Result<usize>{
        if bytes.is_empty(){return Ok(0);}
        if self.remaining==0{let mut probe=[0u8;1];return if self.inner.read(&mut probe)?==0{Ok(0)}else{Err(io::Error::other("项目读取大小超过安全上限"))};}
        let max=bytes.len().min(self.remaining as usize);let count=self.inner.read(&mut bytes[..max])?;self.remaining-=count as u64;Ok(count)
    }
}
struct CheckedJsonRead<R>{inner:R,remaining:u64,in_string:bool,escaped:bool,string_bytes:usize}
impl<R:Read>Read for CheckedJsonRead<R>{
    fn read(&mut self,bytes:&mut[u8])->io::Result<usize>{
        if bytes.is_empty(){return Ok(0);}
        if self.remaining==0{let mut probe=[0u8;1];return if self.inner.read(&mut probe)?==0{Ok(0)}else{Err(io::Error::other("项目解压大小超过安全上限"))};}
        let max=bytes.len().min(self.remaining as usize);let count=self.inner.read(&mut bytes[..max])?;self.remaining-=count as u64;
        for &byte in &bytes[..count]{
            if self.in_string{self.string_bytes+=1;
                if self.string_bytes>project_data::MAX_TEXT*6+2{return Err(io::Error::other("项目JSON字符串超过安全上限"));}
                if self.escaped{self.escaped=false;}else if byte==b'\\'{self.escaped=true;}else if byte==b'"'{self.in_string=false;}
            }else if byte==b'"'{self.in_string=true;self.string_bytes=0;}
        }Ok(count)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::{document::{Primitive,ShapeKind},region_state::{self,Stage}};
    use serde_json::{json,Value};
    use std::{fs, path::PathBuf,sync::atomic::{AtomicU64,Ordering}};
    struct TestDir(PathBuf);
    impl TestDir{fn new()->Self{static NEXT:AtomicU64=AtomicU64::new(1);
        let path=std::env::temp_dir().join(format!("primitive-painter-project-test-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&path).unwrap();Self(path)}fn file(&self,name:&str)->PathBuf{self.0.join(name)}}
    impl Drop for TestDir{fn drop(&mut self){if let Ok(entries)=fs::read_dir(&self.0){for entry in entries.flatten(){
        if entry.file_type().is_ok_and(|t|t.is_file()){let _=fs::remove_file(entry.path());}else{let _=fs::remove_dir(entry.path());}
    }}let _=fs::remove_dir(&self.0);}}
    fn shape(x:f32)->Primitive{Primitive::new(ShapeKind::Ellipse,x,50.,20.,10.,[100,150,200])}
    fn row(doc:&Document)->[u32;9]{doc.primitives.last().unwrap().to_tokens().unwrap().map(u32::from)}
    fn fixture()->(Document,RawHistoryStore){
        let mut doc=Document::new();let mut history=RawHistoryStore::new();
        let bg=doc.commit_generated(Primitive::background([20,40,60])).unwrap();
        let mut raw=vec![region_state::BOS_RECORD,row(&doc)];history.attach(&doc,bg,&raw).unwrap();
        let origin=doc.commit_generated(shape(70.)).unwrap();raw.extend([region_state::marker_record(Stage::Coarse,0).unwrap(),row(&doc)]);
        history.attach(&doc,origin,&raw).unwrap();
        let a=doc.commit_generated(shape(90.)).unwrap();let mut ar=raw.clone();ar.push(row(&doc));history.attach(&doc,a,&ar).unwrap();
        let a2=doc.commit_generated(shape(110.)).unwrap();ar.push(row(&doc));history.attach(&doc,a2,&ar).unwrap();
        let mut overlay=ar;overlay.push(region_state::END_RECORD);history.set_control_overlay(&doc,a2,&overlay).unwrap();
        doc.jump(origin).unwrap();let b=doc.commit_generated(shape(80.)).unwrap();let mut br=raw.clone();br.push(row(&doc));history.attach(&doc,b,&br).unwrap();
        raw.extend([region_state::END_RECORD,region_state::marker_record(Stage::Coarse,1).unwrap()]);
        history.set_control_overlay(&doc,origin,&raw).unwrap();
        doc.prompt="未生成的中文草稿，保留标点；？！".into();doc.settings.cfg=2.5;doc.settings.renyi_alpha=3.5;
        doc.settings.parallel_sequences=32;doc.settings.pause_at_stage_end=true;doc.settings.auto_reorder_before_generation=true;
        (doc,history)
    }
    fn payload(doc:&Document,history:&RawHistoryStore)->Value{
        serde_json::to_value(ProjectRef{schema:SCHEMA,document:doc.project_ref(),history:history.project_ref()}).unwrap()
    }
    fn mapped_fixture()->(Document,RawHistoryStore){
        let mut doc=Document::new();let mut history=RawHistoryStore::new();let parent=doc.current;
        let candidate=doc.preview_canvas(vec![shape(40.)]).unwrap();
        let plan=history.prepare_edit(&doc,parent,candidate,None,false).unwrap();
        let node=doc.commit_canvas("mapped fixture",crate::document::OperationSource::User,plan.canvas.clone()).unwrap();
        history.apply_prepared_edit(&doc,parent,node,plan).unwrap();(doc,history)
    }
    fn write_json(path:&Path,bytes:&[u8],version:u32){
        let mut file=File::create(path).unwrap();file.write_all(MAGIC).unwrap();file.write_all(&version.to_le_bytes()).unwrap();
        let mut encoder=GzEncoder::new(file,Compression::fast());encoder.write_all(bytes).unwrap();encoder.finish().unwrap();
    }
    fn write_value(path:&Path,value:&Value){write_json(path,&serde_json::to_vec(value).unwrap(),SCHEMA)}

    #[test]
    fn project_roundtrip_preserves_entire_fork_tree_drafts_raw_overlays_and_resume_delete(){
        let dir=TestDir::new();let path=dir.file("分支项目.pproj");let(doc,history)=fixture();
        let expected=payload(&doc,&history);save_project(&path,&doc,&history).unwrap();
        let(mut loaded,mut raw)=load_project(&path).unwrap();assert_eq!(payload(&loaded,&raw),expected);
        for node in 0..doc.nodes.len(){assert_eq!(loaded.materialize(node).unwrap(),doc.materialize(node).unwrap());
            assert_eq!(raw.context_for(&loaded,node).unwrap(),history.context_for(&doc,node).unwrap());}
        assert_eq!(loaded.prompt,doc.prompt);assert_eq!(loaded.settings,doc.settings);
        let old_current=loaded.current;let mut context=raw.context_for(&loaded,old_current).unwrap().unwrap();
        let generated=loaded.commit_generated(shape(100.)).unwrap();context.push(row(&loaded));raw.attach(&loaded,generated,&context).unwrap();
        assert!(loaded.primitives.last().unwrap().id>doc.primitives.iter().map(|p|p.id).max().unwrap());
        let plan=loaded.plan_delete_subtrees(&[3]).unwrap();let deleted=raw.apply_delete_subtrees(&mut loaded,plan).unwrap();
        let current=deleted.old_to_new[generated].unwrap();assert_eq!(raw.context_for(&loaded,current).unwrap(),Some(context));
        let second=dir.file("再次保存.pproj");save_project(&second,&loaded,&raw).unwrap();let(reloaded,rraw)=load_project(&second).unwrap();
        assert_eq!(payload(&reloaded,&rraw),payload(&loaded,&raw));
    }

    #[test]
    fn project_rejects_unknown_schema_truncation_extra_members_and_unknown_fields(){
        let dir=TestDir::new();let(doc,history)=fixture();let value=payload(&doc,&history);let path=dir.file("bad.pproj");
        write_json(&path,&serde_json::to_vec(&value).unwrap(),99);assert!(load_project(&path).is_err());
        write_value(&path,&value);let valid=fs::read(&path).unwrap();fs::write(&path,&valid[..valid.len()-5]).unwrap();assert!(load_project(&path).is_err());
        let mut trailing=valid.clone();trailing.extend(b"trailing");fs::write(&path,trailing).unwrap();assert!(load_project(&path).is_err());
        let mut second=valid.clone();second.extend_from_slice(&valid[HEADER as usize..]);fs::write(&path,second).unwrap();assert!(load_project(&path).is_err());
        let mut wrong=value.clone();wrong["schema"]=json!(99);write_value(&path,&wrong);assert!(load_project(&path).is_err());
        let mut wrong=value.clone();wrong["absolute_model_path"]=json!("C:/not-in-project/model.pt");write_value(&path,&wrong);assert!(load_project(&path).is_err());
        let mut wrong=value.clone();wrong["document"]["settings"]["unknown"]=json!(true);write_value(&path,&wrong);assert!(load_project(&path).is_err());
        let text=serde_json::to_string(&value).unwrap();let duplicate=text.replacen('{',"{\"schema\":1,",1);
        write_json(&path,duplicate.as_bytes(),SCHEMA);assert!(load_project(&path).is_err());
    }

    #[test]
    fn project_strict_validation_rejects_cycles_patches_lattice_ids_checkpoints_and_raw(){
        let dir=TestDir::new();let(doc,history)=fixture();let original=payload(&doc,&history);let path=dir.file("invalid.pproj");
        let mut cases=Vec::new();
        let mut v=original.clone();v["document"]["nodes"][1]["parent"]=json!(1);cases.push(v);
        let mut v=original.clone();v["document"]["nodes"][0]["children"].as_array_mut().unwrap().push(json!(0));cases.push(v);
        let mut v=original.clone();v["document"]["nodes"][2]["patches"][0]["Add"]["index"]=json!(999);cases.push(v);
        let mut v=original.clone();v["document"]["nodes"][1]["patches"][0]["Add"]["primitive"]["x"]=json!(0.25);cases.push(v);
        let mut v=original.clone();v["document"]["nodes"][1]["patches"][0]["Add"]["primitive"]["id"]=json!(0);cases.push(v);
        let mut v=original.clone();v["document"]["nodes"][2]["patches"][0]["Add"]["primitive"]["id"]=json!(1);cases.push(v);
        let mut v=original.clone();let background=v["document"]["nodes"][1]["patches"][0]["Add"]["primitive"].clone();
        v["document"]["checkpoints"][0]["canvas"].as_array_mut().unwrap().push(background);cases.push(v);
        let mut v=original.clone();v["document"]["next_primitive_id"]=json!(1);cases.push(v);
        let mut v=original.clone();v["document"]["next_primitive_id"]=json!(u64::MAX);cases.push(v);
        let mut v=original.clone();v["document"]["primitives"][0]["color"]=json!([0,0,0]);cases.push(v);
        let mut v=original.clone();v["history"]["entries"][1]["base"]=json!({"kind":"node","node":2});cases.push(v);
        let mut v=original.clone();v["history"]["entries"][1]["keep"]=json!(usize::MAX);cases.push(v);
        let mut v=original.clone();v["history"]["entries"][1]["tail"][1][0]=json!(400);cases.push(v);
        let mut v=original.clone();v["history"]["control_overlays"][0]["node"]=json!(999);cases.push(v);
        let mut v=original.clone();v["document"]["checkpoints"].as_array_mut().unwrap().push(original["document"]["checkpoints"][0].clone());cases.push(v);
        for(index,wrong)in cases.into_iter().enumerate(){write_value(&path,&wrong);assert!(load_project(&path).is_err(),"invalid case {index} accepted");}
        assert_eq!(payload(&doc,&history),original,"bad imports never mutate the active objects");
    }

    #[test]
    fn project_streaming_resource_limits_apply_to_both_save_and_load(){
        let dir=TestDir::new();let path=dir.file("limits.pproj");let(doc,history)=fixture();save_project(&path,&doc,&history).unwrap();
        let old=fs::read(&path).unwrap();
        for(kind,max)in [(project_data::NODES,2),(project_data::PATCHES,1),(project_data::RECORDS,1),
            (project_data::CURRENT_PRIMITIVES,1),(project_data::EDGES,1)]{
            let mut limits=Limits::default();limits.counts[kind as usize]=max;
            assert!(load_with_limits(&path,limits).is_err());assert!(save_with_limits(&path,&doc,&history,limits).is_err());
            assert_eq!(fs::read(&path).unwrap(),old);
        }
        let mut limits=Limits::default();limits.compressed=16;assert!(load_with_limits(&path,limits).is_err());
        assert!(save_with_limits(&path,&doc,&history,limits).is_err());assert_eq!(fs::read(&path).unwrap(),old);
        let mut limits=Limits::default();limits.json=32;assert!(load_with_limits(&path,limits).is_err());
        assert!(save_with_limits(&path,&doc,&history,limits).is_err());assert_eq!(fs::read(&path).unwrap(),old);
        assert!(!fs::read_dir(&dir.0).unwrap().flatten().any(|e|e.file_name().to_string_lossy().ends_with(".tmp")));
    }

    #[test]
    fn project_save_rejects_invalid_source_and_failed_overwrite_keeps_previous_file(){
        let dir=TestDir::new();let path=dir.file("existing.pproj");let(mut doc,history)=fixture();save_project(&path,&doc,&history).unwrap();
        let old=fs::read(&path).unwrap();let mut bad=payload(&doc,&history);
        bad["history"]["entries"][1]["keep"]=json!(usize::MAX);
        let bad_history=RawHistoryStore::from_project_data(serde_json::from_value(bad["history"].clone()).unwrap()).unwrap();
        assert!(save_project(&path,&doc,&bad_history).is_err());assert_eq!(fs::read(&path).unwrap(),old);
        #[cfg(windows)]{
            use std::os::windows::fs::OpenOptionsExt;
            let held=fs::OpenOptions::new().read(true).share_mode(0x1|0x2).open(&path).unwrap();
            doc.prompt.push_str(" changed");assert!(save_project(&path,&doc,&history).is_err());drop(held);
            assert_eq!(fs::read(&path).unwrap(),old);
        }
        doc.prompt.push_str(" valid overwrite");save_project(&path,&doc,&history).unwrap();
        assert_eq!(load_project(&path).unwrap().0.prompt,doc.prompt);
    }

    #[test]
    fn project_preserves_manual_overfilled_regions_interrupts_and_same_cell_reopen(){
        let dir=TestDir::new();
        for(stage,count,moved_x)in[(Stage::Coarse,25,160.),(Stage::Fine,15,100.)]{
            let mut doc=Document::new();let mut history=RawHistoryStore::new();
            let bg=doc.commit_generated(Primitive::background([20,40,60])).unwrap();
            let base=vec![region_state::BOS_RECORD,row(&doc)];history.attach(&doc,bg,&base).unwrap();
            let mut selected=base;selected.push(region_state::marker_record(stage,0).unwrap());history.set_control_overlay(&doc,bg,&selected).unwrap();
            let mut canvas=doc.primitives.clone();for _ in 0..count{canvas.push(Primitive::new(ShapeKind::Ellipse,20.,20.,8.,8.,[80,120,160]));}
            let overfilled=doc.commit_canvas("manual over training cap",crate::document::OperationSource::User,canvas).unwrap();
            history.reconcile_manual(&doc,bg,overfilled).unwrap();
            let mut canvas=doc.primitives.clone();canvas[1].x=moved_x;
            let moved=doc.commit_canvas("manual region switch",crate::document::OperationSource::User,canvas).unwrap();
            history.reconcile_manual(&doc,overfilled,moved).unwrap();
            let mut closed=history.context_for(&doc,moved).unwrap().unwrap();closed.push(region_state::END_RECORD);history.finish(&doc,moved,&closed).unwrap();
            let mut canvas=doc.primitives.clone();canvas.push(Primitive::new(ShapeKind::Rectangle,moved_x,20.,8.,8.,[100,140,180]));
            let reopened=doc.commit_canvas("manual same cell reopen",crate::document::OperationSource::User,canvas).unwrap();
            history.reconcile_manual(&doc,moved,reopened).unwrap();
            let expected=payload(&doc,&history);let path=dir.file(if stage==Stage::Fine{"fine.pproj"}else{"coarse.pproj"});
            save_project(&path,&doc,&history).unwrap();let(loaded,raw)=load_project(&path).unwrap();
            assert_eq!(payload(&loaded,&raw),expected);
            let records=raw.context_for(&loaded,loaded.current).unwrap().unwrap();
            assert!(records.contains(&region_state::INTERRUPT_RECORD));
            let actual=region_state::DrawingRegion::from_history(&records).unwrap();assert_eq!(actual.stage,stage);assert_eq!(actual.index,Some(1));
        }
    }

    #[test]
    fn project_preserves_more_than_86_global_shapes_and_sparse_checkpoints(){
        let dir=TestDir::new();let mut doc=Document::new();let mut history=RawHistoryStore::new();
        let mut canvas=vec![Primitive::background([10,20,30])];canvas.extend((0..140).map(|n|shape(20.+n as f32)));
        let edited=doc.commit_canvas("large manual global",crate::document::OperationSource::User,canvas).unwrap();
        history.reconcile_manual(&doc,0,edited).unwrap();
        let path=dir.file("manual.pproj");save_project(&path,&doc,&history).unwrap();
        let(loaded,raw)=load_project(&path).unwrap();assert_eq!(payload(&loaded,&raw),payload(&doc,&history));
        let mut deep=Document::new();for n in 0..65{deep.commit_generated(shape(n as f32+10.)).unwrap();}
        let path=dir.file("checkpoints.pproj");let empty=RawHistoryStore::new();save_project(&path,&deep,&empty).unwrap();
        let(loaded,raw)=load_project(&path).unwrap();assert_eq!(payload(&loaded,&raw),payload(&deep,&empty));
        assert_eq!(loaded.materialize(64).unwrap(),deep.materialize(64).unwrap());
    }

    #[test]
    fn project_schema2_preserves_pending_manual_reviews_through_roundtrip_and_delete(){
        let dir=TestDir::new();let(mut doc,mut history)=fixture();let parent=doc.current;
        let mut canvas=doc.primitives.clone();canvas.last_mut().unwrap().color=[60,100,140];
        let manual=doc.commit_canvas("manual current coarse review",crate::document::OperationSource::User,canvas).unwrap();
        history.reconcile_manual(&doc,parent,manual).unwrap();assert_eq!(history.manual_review_regions(&doc,manual).unwrap(),[(4,0)]);
        let path=dir.file("pending.pproj");save_project(&path,&doc,&history).unwrap();
        let(mut loaded,mut raw)=load_project(&path).unwrap();assert_eq!(payload(&loaded,&raw),payload(&doc,&history));
        assert_eq!(raw.manual_review_regions(&loaded,loaded.current).unwrap(),[(4,0)]);
        let plan=loaded.plan_delete_subtrees(&[3]).unwrap();let deletion=raw.apply_delete_subtrees(&mut loaded,plan).unwrap();
        let mapped=deletion.old_to_new[manual].unwrap();assert_eq!(raw.manual_review_regions(&loaded,mapped).unwrap(),[(4,0)]);
        save_project(&path,&loaded,&raw).unwrap();let(after,store)=load_project(&path).unwrap();
        assert_eq!(store.manual_review_regions(&after,after.current).unwrap(),[(4,0)]);
    }

    #[test]
    fn project_loads_schema1_without_inventing_pending_reviews_and_requires_schema2_field(){
        let dir=TestDir::new();let(doc,history)=fixture();let original=payload(&doc,&history);let path=dir.file("legacy.pproj");
        let mut legacy=original.clone();legacy["schema"]=json!(1);legacy["history"].as_object_mut().unwrap().remove("manual_reviews");
        legacy["history"].as_object_mut().unwrap().remove("model_ids");
        legacy["history"].as_object_mut().unwrap().remove("mode_evidence");
        write_json(&path,&serde_json::to_vec(&legacy).unwrap(),1);let(loaded,raw)=load_project(&path).unwrap();
        assert_eq!(loaded.current,doc.current);assert_eq!(loaded.primitives,doc.primitives);assert_eq!(loaded.settings,doc.settings);
        for node in 0..loaded.nodes.len(){assert_eq!(loaded.materialize(node).unwrap(),doc.materialize(node).unwrap());
            assert_eq!(raw.context_for(&loaded,node).unwrap(),history.context_for(&doc,node).unwrap());
            assert!(raw.manual_review_regions(&loaded,node).unwrap().is_empty());}
        let mut malformed=original;malformed["history"].as_object_mut().unwrap().remove("manual_reviews");
        write_value(&path,&malformed);assert!(load_project(&path).is_err());
    }

    #[test]
    fn schema3_requires_model_ids_and_schema2_remains_readable_without_them(){
        let dir=TestDir::new();let(doc,history)=fixture();let original=payload(&doc,&history);let path=dir.file("versions.pproj");
        let mut legacy=original.clone();legacy["schema"]=json!(2);legacy["history"].as_object_mut().unwrap().remove("model_ids");
        legacy["history"].as_object_mut().unwrap().remove("mode_evidence");
        write_json(&path,&serde_json::to_vec(&legacy).unwrap(),2);let(loaded,raw)=load_project(&path).unwrap();
        assert_eq!(loaded.current,doc.current);assert_eq!(loaded.primitives,doc.primitives);assert_eq!(loaded.settings,doc.settings);
        for node in 0..doc.nodes.len(){assert_eq!(raw.context_for(&loaded,node).unwrap(),history.context_for(&doc,node).unwrap());}
        let mut missing=original.clone();missing["history"].as_object_mut().unwrap().remove("model_ids");
        write_value(&path,&missing);assert!(load_project(&path).is_err());
        write_json(&path,&serde_json::to_vec(&original).unwrap(),2);assert!(load_project(&path).is_err(),"header/schema mismatch remains invalid");
        let mut old_with_sidecar=original;old_with_sidecar["schema"]=json!(2);
        write_json(&path,&serde_json::to_vec(&old_with_sidecar).unwrap(),2);assert!(load_project(&path).is_err());
    }

    #[test]
    fn schema4_preserves_legacy_schema3_without_inventing_typed_mode_sources(){
        let dir=TestDir::new();let(doc,history)=fixture();let mut legacy=payload(&doc,&history);
        legacy["schema"]=json!(3);legacy["history"].as_object_mut().unwrap().remove("mode_evidence");
        let path=dir.file("schema3.pproj");write_json(&path,&serde_json::to_vec(&legacy).unwrap(),3);
        let(loaded,history)=load_project(&path).unwrap();assert_eq!(loaded.primitives,doc.primitives);
        assert_eq!(payload(&loaded,&history)["history"]["mode_evidence"],json!([]));
        save_project(&path,&loaded,&history).unwrap();assert_eq!(&fs::read(&path).unwrap()[8..12],&4u32.to_le_bytes());
    }

    #[test]
    fn stage_assignment_metadata_roundtrips_in_schema3_and_rejects_invalid_cells_or_tickets(){
        use crate::document::{OperationSource,PrimitiveStageAssignment,StageTier};
        let dir=TestDir::new();let path=dir.file("stages.pproj");let mut doc=Document::new();
        let mut p=shape(40.);p.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Global,cell:0,queue_order:0});
        doc.commit_canvas("stage assignment",OperationSource::User,vec![p]).unwrap();let history=RawHistoryStore::new();
        save_project(&path,&doc,&history).unwrap();let(loaded,raw)=load_project(&path).unwrap();
        assert_eq!(payload(&loaded,&raw),payload(&doc,&history));
        let good=payload(&doc,&history);
        for assignment in [json!({"stage":"coarse","cell":16,"queue_order":1}),json!({"stage":"fine","cell":64,"queue_order":1}),
            json!({"stage":"global","cell":1,"queue_order":1}),json!({"stage":"coarse","cell":0,"queue_order":-1}),
            json!({"stage":"coarse","cell":0,"queue_order":1.5}),json!({"stage":"coarse","cell":0,"queue_order":1,"deferred":true})]{
            let mut bad=good.clone();bad["document"]["primitives"][0]["stage_assignment"]=assignment;
            write_value(&path,&bad);assert!(load_project(&path).is_err());
        }
        let mut old=good;old["schema"]=json!(2);old["history"].as_object_mut().unwrap().remove("model_ids");
        write_json(&path,&serde_json::to_vec(&old).unwrap(),2);assert!(load_project(&path).is_err(),"schema2 must not silently accept schema3 ownership");
    }

    #[test]
    fn schema3_model_ids_reject_invalid_phase_and_unknown_primitive_references(){
        let dir=TestDir::new();let(doc,history)=mapped_fixture();let good=payload(&doc,&history);let path=dir.file("ids.pproj");
        assert!(!good["history"]["model_ids"].as_array().unwrap().is_empty());
        for phase in [json!(0),json!(2),json!(16),json!(-1),json!(4.5),json!("4"),Value::Null]{
            let mut bad=good.clone();bad["history"]["model_ids"][0]["phase_floor"]=phase;
            write_value(&path,&bad);assert!(load_project(&path).is_err());
        }
        let mut missing=good.clone();missing["history"]["model_ids"][0].as_object_mut().unwrap().remove("phase_floor");
        write_value(&path,&missing);assert!(load_project(&path).is_err());
        let mut unknown=good;unknown["history"]["model_ids"][0]["tail"]=json!([u64::MAX]);
        write_value(&path,&unknown);assert!(load_project(&path).is_err());
    }

    #[test]
    fn project_keeps_consumed_parent_snapshot_and_pending_late_sibling_independent(){
        let dir=TestDir::new();let(mut doc,mut history)=fixture();let parent=doc.current;
        let mut canvas=doc.primitives.clone();canvas.last_mut().unwrap().color=[60,100,140];
        let manual=doc.commit_canvas("pending parallel origin",crate::document::OperationSource::User,canvas).unwrap();
        history.reconcile_manual(&doc,parent,manual).unwrap();let private=history.context_for(&doc,manual).unwrap().unwrap();
        let mut ended=private.clone();ended.push(region_state::END_RECORD);history.set_control_overlay(&doc,manual,&ended).unwrap();
        assert!(history.manual_review_regions(&doc,manual).unwrap().is_empty());
        let late=doc.commit_generated(shape(100.)).unwrap();let mut late_raw=private;late_raw.push(row(&doc));history.attach(&doc,late,&late_raw).unwrap();
        let path=dir.file("parallel-pending.pproj");save_project(&path,&doc,&history).unwrap();let(mut loaded,mut raw)=load_project(&path).unwrap();
        assert!(raw.manual_review_regions(&loaded,manual).unwrap().is_empty());
        assert_eq!(raw.manual_review_regions(&loaded,late).unwrap(),[(4,0)]);
        let plan=loaded.plan_delete_subtrees(&[3]).unwrap();let removed=raw.apply_delete_subtrees(&mut loaded,plan).unwrap();
        assert!(raw.manual_review_regions(&loaded,removed.old_to_new[manual].unwrap()).unwrap().is_empty());
        assert_eq!(raw.manual_review_regions(&loaded,removed.old_to_new[late].unwrap()).unwrap(),[(4,0)]);
    }

    #[test]
    fn project_seed_modes_and_recorded_numeric_seeds_roundtrip_without_randomizing(){
        let dir=TestDir::new();let(mut doc,history)=fixture();let parent=doc.current;
        doc.settings.seed=0x1234_5678_9abc_def0;doc.settings.auto_seed=true;
        let automatic=doc.commit_generated(shape(110.)).unwrap();
        doc.jump(parent).unwrap();doc.settings.seed=17;doc.settings.auto_seed=false;
        let manual=doc.commit_generated(shape(120.)).unwrap();
        // The staged choice is independently persisted from both committed nodes.
        doc.settings.seed=987654321;doc.settings.auto_seed=true;
        let expected=payload(&doc,&history);let path=dir.file("seed-modes.pproj");save_project(&path,&doc,&history).unwrap();
        let(mut loaded,raw)=load_project(&path).unwrap();assert_eq!(payload(&loaded,&raw),expected);
        assert_eq!(loaded.settings.seed,987654321);assert!(loaded.settings.auto_seed);
        loaded.jump(automatic).unwrap();assert_eq!(loaded.settings.seed,0x1234_5678_9abc_def0);assert!(loaded.settings.auto_seed);
        loaded.jump(manual).unwrap();assert_eq!(loaded.settings.seed,17);assert!(!loaded.settings.auto_seed);
        let mut legacy=expected;legacy["document"]["settings"].as_object_mut().unwrap().remove("auto_seed");
        for node in legacy["document"]["nodes"].as_array_mut().unwrap(){node["settings"].as_object_mut().unwrap().remove("auto_seed");}
        write_value(&path,&legacy);let(old,_)=load_project(&path).unwrap();
        assert!(old.settings.auto_seed&&old.nodes.iter().all(|node|node.settings.auto_seed));
        assert_eq!(old.settings.seed,987654321);assert_eq!(old.nodes[automatic].settings.seed,0x1234_5678_9abc_def0);
        assert_eq!(old.nodes[manual].settings.seed,17);
    }
}
