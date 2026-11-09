#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
mod canvas;
mod reference_image;
mod reference_serialize;
mod reference_import;
mod reference_session;
mod reference_continue;
mod save_library;
mod redraw;
mod redraw_context;
mod redraw_session;
mod redraw_smoke;
mod workspace;
mod mode_transition;
mod mode_switch;
mod mode_switch_smoke;
mod save_library_ui;
mod save_session;
mod manual_save;
mod manual_save_smoke;
mod library_smoke;
mod auto_seed;
mod document;
mod history_context;
mod inference;
mod model;
mod parallel_history;
mod atomic_file;
mod project_file;
mod image_export;
mod region_state;
mod resources;
mod model_download;
mod runtime;
mod thumbnails;
mod tree_layout;
mod tree_view;
mod ui;
mod ui_perf;
mod stage_smoke;
mod diagnostics;
mod compatibility_notice;
mod compatibility_dialog;
mod compatibility_startup;
mod acceleration;
mod amd_capability;
#[cfg(feature = "aten-hip")]
mod aten_weight_index;

use document::{Document, OperationSource, Primitive, ShapeKind};
use eframe::egui;
use history_context::RawHistoryStore;
use parallel_history::ParallelHistory;
use region_state::DrawingRegion;
use runtime::{Event, Runtime};
use std::path::PathBuf;
use std::collections::BTreeMap;

struct PendingGlobalEdit { origin:usize,label:String,canvas:Vec<Primitive> }

fn settings_for_generation(mut settings:document::Settings)->Result<document::Settings,String>{
    settings.validate()?;
    if settings.auto_seed {settings.seed=auto_seed::reserve_seed(settings.parallel_sequences)?;}
    Ok(settings)
}

// Parallelism is a session/workload preference, not model conditioning.
// Selecting an older state restores its prompt/sampler/seed without silently
// disabling the user's currently enabled parallel generation mode.
fn select_history_node(document: &mut Document, node: usize) -> Result<(), String> {
    let parallel = document.settings.parallel_sequences;
    document.jump(node)?;
    document.settings.parallel_sequences = parallel;
    Ok(())
}

/// A rate control may freeze the live head before it has advanced a step.
/// Preserve staged inputs in that case instead of reloading the same node.
fn browse_history_node(document:&mut Document,node:usize)->Result<(),String>{
    if node>=document.nodes.len(){return Err("要翻阅的状态不存在".into());}
    if node!=document.current{select_history_node(document,node)?;}
    Ok(())
}
fn browse_history_session(document:&mut Document,parallel:&mut Option<ParallelHistory>,node:usize)->Result<(),String>{
    browse_history_node(document,node)?;
    if let Some(parallel)=parallel{parallel.set_follow_live(false);}
    Ok(())
}

fn delete_history_subtrees(
    document: &mut Document,
    histories: &mut RawHistoryStore,
    selected: &[u64],
) -> Result<document::HistoryDeletion, String> {
    let roots = selected.iter().map(|&id| usize::try_from(id)
        .map_err(|_| "状态节点编号超出支持范围".to_owned())).collect::<Result<Vec<_>, _>>()?;
    let parallel = document.settings.parallel_sequences;
    let plan = document.plan_delete_subtrees(&roots)?;
    let result = histories.apply_delete_subtrees(document, plan)?;
    if result.current_was_deleted {
        document.settings.parallel_sequences = parallel;
    }
    Ok(result)
}

enum FileWork {
    Library(save_session::LibraryWork),
    Save { path:PathBuf, document:Document, histories:RawHistoryStore },
    Import { path:PathBuf },
    Png { path:PathBuf, primitives:Vec<Primitive>, resolution:u32 },
    Svg { path:PathBuf, primitives:Vec<Primitive> },
}
enum FileOutcome {
    Library(save_session::LibraryOutcome),
    Saved { path:PathBuf, document:Document, histories:RawHistoryStore, result:Result<(),String> },
    Imported { path:PathBuf, result:Result<(Document,RawHistoryStore),String> },
    Exported { path:PathBuf, kind:&'static str, result:Result<(),String> },
}
struct FileTask {
    receiver:std::sync::mpsc::Receiver<FileOutcome>,
    _worker:std::thread::JoinHandle<()>,
}
fn file_attempt<T>(action:impl FnOnce()->Result<T,String>)->Result<T,String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action))
        .unwrap_or_else(|_|Err("文件处理未完成；请查看日志并重试".into()))
}
fn perform_file_work(work:FileWork)->FileOutcome {
    match work {
        FileWork::Library(work)=>FileOutcome::Library(save_session::perform(work)),
        FileWork::Save {path,document,histories}=>{
            let result=file_attempt(||project_file::save_project(&path,&document,&histories));
            // Even failed serialization returns ownership of the full tree.
            FileOutcome::Saved{path,document,histories,result}
        },
        FileWork::Import{path}=>{
            let result=file_attempt(||project_file::load_project(&path));
            FileOutcome::Imported{path,result}
        },
        FileWork::Png{path,primitives,resolution}=>{
            let result=file_attempt(||image_export::export_png(&path,&primitives,resolution));
            FileOutcome::Exported{path,kind:"PNG",result}
        },
        FileWork::Svg{path,primitives}=>{
            let result=file_attempt(||image_export::export_svg(&path,&primitives));
            FileOutcome::Exported{path,kind:"SVG",result}
        },
    }
}
fn spawn_file_task(repaint:egui::Context)->Result<(std::sync::mpsc::Sender<FileWork>,FileTask),String>{
    let (sender,work)=std::sync::mpsc::channel();
    let (finished,receiver)=std::sync::mpsc::channel();
    let worker=std::thread::Builder::new().name("project-file-io".into()).spawn(move||{
        if let Ok(work)=work.recv(){
            let outcome=perform_file_work(work);
            repaint.request_repaint();
            let _=finished.send(outcome);
        }
    }).map_err(|e|format!("无法启动文件处理任务：{e}"))?;
    Ok((sender,FileTask{receiver,_worker:worker}))
}

fn should_auto_load_model(
    preview_or_self_test: bool,
    configuration_valid: bool,
    model_exists: bool,
    tokenizer_exists: bool,
) -> bool {
    !preview_or_self_test && configuration_valid && model_exists && tokenizer_exists
}

fn observe_acceleration_preparation(
    event: &Event,
    compiling: &mut bool,
    report: &mut Option<inference::LoadPreparationReport>,
) {
    match event {
        Event::AccelerationPreparing { .. } => *compiling = true,
        Event::AccelerationPrepared { report: prepared } => *report = Some(prepared.clone()),
        Event::Loaded { .. } | Event::Failed(_) => *compiling = false,
        _ => {},
    }
}

fn generation_is_ready(running: bool, loading: bool, compiling: bool, loaded: bool) -> bool {
    !running && !loading && !compiling && loaded
}

fn reorder_current_canvas(
    document: &mut Document,
    histories: &mut RawHistoryStore,
) -> Result<(bool, DrawingRegion), String> {
    let parent = document.current;
    // Validate the selected branch before creating any new editable state.
    let previous = match histories.context_for(document, parent)? {
        Some(records) => records,
        None => {
            let mut records = vec![model::BOS_RECORD];
            for primitive in &document.primitives {
                records.push(primitive.to_tokens()?.map(u32::from));
            }
            records
        }
    };
    DrawingRegion::from_history(&previous)?;
    let sorted = histories.contribution_ordered_canvas(document, parent)?;
    let node = document.commit_canvas("区域内按图形贡献重排序", OperationSource::User, sorted)?;
    let region = histories.reconcile_manual(document, parent, node)?;
    Ok((node != parent, region))
}

#[cfg(test)]
mod session_tests {
    use super::*;

    #[test]
    fn browsing_current_state_does_not_discard_staged_prompt_or_settings(){
        let mut doc=Document::new();doc.prompt="正在输入的提示词".into();doc.settings.cfg=2.5;
        doc.settings.seed=918273;let before=serde_json::to_vec(&doc.nodes).unwrap();
        browse_history_node(&mut doc,0).unwrap();
        assert_eq!(doc.prompt,"正在输入的提示词");assert_eq!(doc.settings.cfg,2.5);assert_eq!(doc.settings.seed,918273);
        assert_eq!(serde_json::to_vec(&doc.nodes).unwrap(),before);
        assert!(browse_history_node(&mut doc,999).is_err());assert_eq!(doc.current,0);
    }

    #[test]
    fn browsing_the_live_head_keeps_new_model_states_off_the_view_without_stopping_the_branch(){
        let mut doc=Document::new();let mut histories=RawHistoryStore::new();
        let settings=document::Settings{parallel_sequences:1,..Default::default()};
        let mut parallel=Some(ParallelHistory::new(&doc,0,"".into(),settings).unwrap());
        parallel.as_mut().unwrap().initialize_model_input(&doc,&histories).unwrap();
        let bg=Primitive::background([100,120,140]).to_tokens().unwrap().map(u32::from);
        let first=parallel.as_mut().unwrap().submit(&mut doc,&mut histories,0,bg,&[model::BOS_RECORD,bg]).unwrap();
        assert!(parallel.as_ref().unwrap().is_following(&doc));
        doc.prompt="未提交的输入".into();browse_history_session(&mut doc,&mut parallel,first).unwrap();
        assert_eq!(doc.prompt,"未提交的输入");assert!(!parallel.as_ref().unwrap().is_following(&doc));
        let shape=Primitive::new(ShapeKind::Rectangle,30.,30.,10.,10.,[80,100,120]).to_tokens().unwrap().map(u32::from);
        let next=parallel.as_mut().unwrap().submit(&mut doc,&mut histories,0,shape,&[model::BOS_RECORD,bg,shape]).unwrap();
        assert_ne!(next,first);assert_eq!(doc.current,first);assert_eq!(doc.primitives.len(),1);
        assert_eq!(parallel.as_ref().unwrap().heads[0],next);
        assert_eq!(parallel.as_ref().unwrap().statuses[0],parallel_history::LaneStatus::Running);
    }

    #[test]
    fn automatic_seed_is_frozen_once_per_request_and_parallel_ranges_do_not_repeat(){
        let staged=document::Settings{parallel_sequences:8,seed:42,..Default::default()};
        let first=settings_for_generation(staged.clone()).unwrap();
        let second=settings_for_generation(staged.clone()).unwrap();
        assert!(first.auto_seed&&second.auto_seed);
        assert!(second.seed>=first.seed+8);assert_eq!(staged.seed,42);
        let mut doc=Document::new();doc.settings=first.clone();
        let mut tree=ParallelHistory::new(&doc,0,"".into(),first.clone()).unwrap();
        let mut store=RawHistoryStore::new();
        let tokens=Primitive::background([10,20,30]).to_tokens().unwrap().map(u32::from);
        tree.submit(&mut doc,&mut store,0,tokens,&[model::BOS_RECORD,tokens]).unwrap();
        assert_eq!(doc.current_node().settings.seed,first.seed);
        assert_eq!(tree.seeds,(0..8).map(|lane|document::parallel_seed(first.seed,lane)).collect::<Vec<_>>());
    }

    #[test]
    fn manual_seed_mode_preserves_exact_requested_value_and_settings(){
        for seed in [0,42,u64::MAX] {
            let manual=document::Settings{seed,auto_seed:false,..Default::default()};
            assert_eq!(settings_for_generation(manual.clone()).unwrap(),manual);
        }
    }

    fn file_test_directory()->PathBuf{
        use std::sync::atomic::{AtomicU64,Ordering};
        static NEXT:AtomicU64=AtomicU64::new(1);
        let dir=std::env::temp_dir().join(format!("painter-file-worker-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::create_dir_all(&dir).unwrap();dir
    }

    #[test]
    fn failed_file_save_returns_the_complete_original_tree_and_draft(){
        let dir=file_test_directory();let mut doc=Document::new();
        doc.commit_canvas("现有画布",OperationSource::User,vec![Primitive::background([20,40,60])]).unwrap();
        doc.prompt="未提交到节点的提示词".into();doc.settings.cfg=2.5;
        let before=serde_json::to_vec(&doc.nodes).unwrap();let current=doc.current;
        match perform_file_work(FileWork::Save{path:dir.join("missing-parent").join("project.pproj"),document:doc,histories:RawHistoryStore::new()}){
            FileOutcome::Saved{document,histories:_,result,..}=>{
                assert!(result.is_err());assert_eq!(serde_json::to_vec(&document.nodes).unwrap(),before);
                assert_eq!(document.current,current);assert_eq!(document.prompt,"未提交到节点的提示词");
                assert_eq!(document.settings.cfg,2.5);assert_eq!(document.primitives.len(),1);
            },_=>panic!("save must always return its full document payload"),
        }
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn file_worker_roundtrip_and_exports_use_canvas_without_mutating_original_state(){
        let dir=file_test_directory();let mut document=Document::new();
        document.commit_canvas("颜色背景",OperationSource::User,vec![Primitive::background([20,40,60])]).unwrap();
        let current=document.current;let path=dir.join("完整状态树.pproj");
        let document=match perform_file_work(FileWork::Save{path:path.clone(),document,histories:RawHistoryStore::new()}){
            FileOutcome::Saved{document,result,..}=>{result.unwrap();document},_=>unreachable!(),
        };
        let restored=match perform_file_work(FileWork::Import{path:path.clone()}){
            FileOutcome::Imported{result,..}=>result.unwrap().0,_=>unreachable!(),
        };
        assert_eq!(document.current,current);assert_eq!(restored.primitives,document.primitives);
        let png=dir.join("画布.png");let svg=dir.join("画布.svg");
        for work in [FileWork::Png{path:png.clone(),primitives:restored.primitives.clone(),resolution:37},
            FileWork::Svg{path:svg.clone(),primitives:restored.primitives.clone()}]{
            match perform_file_work(work){FileOutcome::Exported{result,..}=>result.unwrap(),_=>unreachable!()}
        }
        let image=image::open(&png).unwrap().to_rgb8();assert_eq!(image.dimensions(),(37,37));
        assert_eq!(image.get_pixel(18,18).0,[20,40,60]);
        assert!(std::fs::read_to_string(&svg).unwrap().contains("viewBox=\"0 0 512 512\""));
        assert_eq!(document.current,current);assert_eq!(document.primitives,restored.primitives);
        for file in [path,png,svg]{std::fs::remove_file(file).unwrap();}std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn file_task_panics_are_errors_and_never_suppress_the_return_path(){
        assert!(file_attempt::<()>(||panic!("simulated file writer failure")).is_err());
    }

    #[test]
    fn subtree_deletion_after_last_completed_rows_removes_new_descendants_and_preserves_sibling() {
        let mut doc=Document::new();
        doc.settings.parallel_sequences=2;
        let mut store=RawHistoryStore::new();
        let mut parallel=ParallelHistory::new(&doc,0,String::new(),doc.settings.clone()).unwrap();
        let background=Primitive::background([20,40,60]).to_tokens().unwrap().map(u32::from);
        let initial=vec![model::BOS_RECORD,background];
        let selected=parallel.submit(&mut doc,&mut store,0,background,&initial).unwrap();
        parallel.submit(&mut doc,&mut store,1,background,&initial).unwrap();
        // A stop request is not the event barrier: final complete rows must
        // still be committed before computing the descendant deletion plan.
        parallel.set_follow_live(false);
        let shape=Primitive::new(ShapeKind::Rectangle,100.,100.,20.,20.,[80,100,120])
            .to_tokens().unwrap().map(u32::from);
        let mut tail=initial.clone();tail.push(shape);
        let late=parallel.submit(&mut doc,&mut store,0,shape,&tail).unwrap();
        let sibling=parallel.submit(&mut doc,&mut store,1,shape,&tail).unwrap();
        let before=store.context_for(&doc,sibling).unwrap();
        drop(parallel);
        let deleted=delete_history_subtrees(&mut doc,&mut store,&[selected as u64]).unwrap();
        assert!(deleted.removed_nodes.contains(&late));
        let kept=deleted.old_to_new[sibling].unwrap();
        assert_eq!(store.context_for(&doc,kept).unwrap(),before);
        assert_eq!(doc.current,0);
        assert_eq!(doc.nodes.len(),3);
        assert_eq!(doc.settings.parallel_sequences,2);
        assert_eq!(doc.materialize(kept).unwrap().len(),2);
    }

    #[test]
    fn deleting_viewed_history_restores_ancestor_but_keeps_parallel_preference() {
        let mut doc=Document::new();let mut histories=RawHistoryStore::new();
        doc.commit_canvas("背景",OperationSource::User,vec![Primitive::background([20;3])]).unwrap();
        let deleted=doc.current;doc.settings.parallel_sequences=32;
        delete_history_subtrees(&mut doc,&mut histories,&[deleted as u64]).unwrap();
        assert_eq!(doc.current,0);assert!(doc.primitives.is_empty());
        assert_eq!(doc.settings.parallel_sequences,32);
        let before=doc.nodes.len();
        assert!(delete_history_subtrees(&mut doc,&mut histories,&[u64::MAX]).is_err());
        assert_eq!(doc.nodes.len(),before);
    }

    #[test]
    fn startup_load_requires_valid_files_and_never_runs_in_preview_or_test_modes() {
        assert!(should_auto_load_model(false, true, true, true));
        assert!(!should_auto_load_model(true, true, true, true));
        assert!(!should_auto_load_model(false, false, true, true));
        assert!(!should_auto_load_model(false, true, false, true));
        assert!(!should_auto_load_model(false, true, true, false));
    }

    #[test]
    fn preparation_events_keep_generation_locked_until_loaded_and_failures_close_waiting() {
        let mut compiling = false;
        let mut report = None;
        let prepared = inference::LoadPreparationReport {
            jit_attempted: true, ready: true, summary: "可用加速准备完成".into(),
            details: vec!["采样与注意力已准备".into()], graph_deferred: true,
        };
        observe_acceleration_preparation(&Event::AccelerationPreparing {
            message: "正在编译采样算子".into(),
        }, &mut compiling, &mut report);
        assert!(compiling);
        assert!(!generation_is_ready(false, true, compiling, false));
        observe_acceleration_preparation(&Event::AccelerationPrepared { report: prepared.clone() },
            &mut compiling, &mut report);
        assert!(compiling, "Prepared must not dismiss the waiting window");
        assert!(report.as_ref().unwrap().graph_deferred);
        assert!(!generation_is_ready(false, false, compiling, true), "compiling itself blocks generation");
        observe_acceleration_preparation(&Event::Loaded { backend: "mock GPU".into(), description: "mock model".into() },
            &mut compiling, &mut report);
        assert!(!compiling);
        assert!(generation_is_ready(false, false, compiling, true));
        observe_acceleration_preparation(&Event::AccelerationPreparing { message: "再次加载后重新准备".into() },
            &mut compiling, &mut report);
        observe_acceleration_preparation(&Event::Failed("准备失败".into()), &mut compiling, &mut report);
        assert!(!compiling, "failure must close the waiting window");
        assert!(!generation_is_ready(false, false, compiling, false));
    }

    #[test]
    fn cpu_or_unsupported_skip_report_never_claims_active_compilation() {
        let mut compiling = false;
        let mut report = None;
        observe_acceleration_preparation(&Event::AccelerationPrepared {
            report: inference::LoadPreparationReport { jit_attempted: false, ready: true,
                summary: "CPU兼容路径，无需编译GPU算子".into(), details: vec![], graph_deferred: false },
        }, &mut compiling, &mut report);
        assert!(!compiling);
        assert!(!report.as_ref().unwrap().jit_attempted);
        assert!(!generation_is_ready(false, true, compiling, false));
        observe_acceleration_preparation(&Event::Loaded { backend: "CPU".into(), description: "mock model".into() },
            &mut compiling, &mut report);
        assert!(generation_is_ready(false, false, compiling, true));
    }

    #[test]
    fn selecting_old_node_restores_conditions_but_keeps_enabled_parallel_count() {
        let mut doc = Document::new();
        doc.prompt = "旧提示词".into();
        doc.settings.seed = 70;
        doc.settings.cfg = 2.0;
        doc.settings.parallel_sequences = 2;
        let node = doc
            .commit_canvas(
                "旧状态",
                OperationSource::User,
                vec![Primitive::background([20, 40, 60])],
            )
            .unwrap();
        doc.settings.parallel_sequences = 4;
        doc.prompt = "新输入".into();
        select_history_node(&mut doc, 0).unwrap();
        assert_eq!(doc.settings.parallel_sequences, 4);
        assert!(doc.prompt.is_empty());
        assert_eq!(doc.settings.seed, document::Settings::default().seed);
        select_history_node(&mut doc, node).unwrap();
        assert_eq!(doc.settings.parallel_sequences, 4);
        assert_eq!(doc.prompt, "旧提示词");
        assert_eq!(doc.settings.seed, 70);
        assert_eq!(doc.settings.cfg, 2.0);
        assert_eq!(doc.nodes[node].settings.parallel_sequences, 2);
    }

    #[test]
    fn automatic_ordering_creates_one_shared_parallel_origin_and_keeps_old_branch() {
        let mut document = Document::new();
        document.commit_canvas("未排序输入", OperationSource::User, vec![
            Primitive::background([254, 254, 254]),
            Primitive::new(ShapeKind::Rectangle, 100.0, 100.0, 80.0, 80.0, [240; 3]),
            Primitive::new(ShapeKind::Rectangle, 100.0, 100.0, 80.0, 80.0, [0; 3]),
        ]).unwrap();
        let parent = document.current;
        let original = document.primitives.clone();
        let mut histories = RawHistoryStore::new();
        document.prompt = "金发蓝眼".into();
        document.settings.auto_reorder_before_generation = true;
        document.settings.parallel_sequences = 4;
        let (changed, _) = reorder_current_canvas(&mut document, &mut histories).unwrap();
        assert!(changed);
        assert_eq!(document.current_node().parent, Some(parent));
        assert_eq!(document.primitives.iter().map(|p| p.id).collect::<Vec<_>>(), vec![1, 3, 2]);
        let parallel = ParallelHistory::new(&document, document.current, document.prompt.clone(), document.settings.clone()).unwrap();
        assert_eq!(parallel.heads, vec![document.current; 4]);
        assert_eq!(document.materialize(parent).unwrap(), original);
        let count = document.nodes.len();
        assert!(!reorder_current_canvas(&mut document, &mut histories).unwrap().0);
        assert_eq!(document.nodes.len(), count);
        document.jump(parent).unwrap();
        assert_eq!(document.primitives, original);
        assert!(!document.settings.auto_reorder_before_generation);
    }
}

struct PainterApp {
    redraw_view_key:Option<(usize,redraw::RedrawSelection)>,
    redraw_smoke_done:bool,
    mode_switch_smoke_done:bool,
    manual_save_smoke_stage:u8,manual_save_smoke_started:Option<std::time::Instant>,
    manual_save_smoke_checks:Vec<String>,manual_save_smoke_disk:Vec<u8>,manual_save_smoke_id:String,
    save_session:save_session::SaveSession,
    library_smoke_done:bool,
    library_smoke_stage:u8,
    library_smoke_checks:Vec<String>,
    file_task:Option<FileTask>,
    reference_session:Option<reference_session::ReferenceSession>,
    reference_smoke_done:bool,
    pending_project_save:Option<PathBuf>,
    pending_project_import:Option<(PathBuf,Document,RawHistoryStore)>,
    file_busy_message:Option<String>,
    file_repaint:egui::Context,
    project_path:Option<PathBuf>,
    close_after_file:bool,
    document: Document,
    editor: ui::EditorUi,
    runtime: Runtime,
    running: bool,
    stopping: bool,
    loading: bool,
    compiling: bool,
    acceleration_preparation_report: Option<inference::LoadPreparationReport>,
    compatibility_startup: compatibility_startup::CompatibilityStartup,
    loaded: bool,
    auto_load_pending: bool,
    pending_canvas_edit: Option<(usize, ui::UiAction)>,
    pending_global_edit:Option<PendingGlobalEdit>,
    stage_by_id:BTreeMap<u64,region_state::Stage>,
    stage_map_node:Option<usize>,
    model_status: String,
    execution_status: String,
    graph_execution_status: String,
    model_path: String,
    tokenizer_path: String,
    status: String,
    error: Option<String>,
    histories: RawHistoryStore,
    drawing_region: DrawingRegion,
    parallel: Option<ParallelHistory>,
    capture: Option<PathBuf>,
    frames: usize,
    self_test: Option<PathBuf>,
    self_test_stage: u8,
    capture_requested: bool,
    self_test_lanes: usize,
    self_test_history: bool,
    self_test_base_primitives: usize,
    self_test_base_nodes: usize,
    self_test_origin: usize,
    self_test_count: usize,
    self_test_browse_edit: bool,
    self_test_browse_observed: bool,
    self_test_edit_requested: bool,
    ui_performance: Option<ui_perf::UiPerformance>,
    keep_open: bool,
    startup_report: Option<PathBuf>,
    startup_report_written: bool,
    automatic_load_attempts: usize,
    editor_stage_smoke:Option<PathBuf>,
    editor_stage_smoke_done:bool,
}

impl PainterApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        sample: bool,
        capture: Option<PathBuf>,
        fixture: Option<PathBuf>,
        history_page: bool,
        history_compact: bool,
        history_fit: bool,
        self_test: Option<PathBuf>,
        self_test_lanes: usize,
        keep_open: bool,
    ) -> Self {
        if let Some(render)=cc.wgpu_render_state.as_ref(){
            let info=render.adapter.get_info();
            diagnostics::event("ui_renderer.selected",serde_json::json!({
                "name":info.name,"backend":format!("{:?}",info.backend),
                "device_type":format!("{:?}",info.device_type),
                "software":info.device_type==wgpu::DeviceType::Cpu,
                "scope":"UI rendering only; does not select or replace model compute backend",
                "windows_dx12_fallback_compiled":true
            }));
        }
        let resources = resources::find_resources();
        let model_path = resources.model.display().to_string();
        let tokenizer_path = resources.tokenizer.display().to_string();
        let library_enabled=std::env::args().any(|a|a=="--library-preview"||a=="--library-smoke-output"||a=="--redraw-smoke-output"||a=="--mode-switch-smoke-output"||a=="--manual-save-smoke-output"||a=="--reference-smoke-output")
            || !(sample||capture.is_some()||fixture.is_some()||self_test.is_some()||history_page||history_compact||history_fit
                ||std::env::args().any(|a|matches!(a.as_str(),"--editor-stage-smoke"|"--startup-report"|"--settings"|"--prompt-templates")));
        let auto_load_pending = !library_enabled && should_auto_load_model(
            sample
                || capture.is_some()
                || fixture.is_some()
                || self_test.is_some()
                || history_page
                || std::env::args().any(|arg|arg=="--editor-stage-smoke")
                || history_compact
                || history_fit,
            resources.configuration_error.is_none(),
            std::path::Path::new(&model_path).is_file(),
            std::path::Path::new(&tokenizer_path).is_file(),
        );
        let mut document = Document::new();
        let compatibility = std::env::args().any(|arg|arg=="--compatibility");
        if compatibility {
            document.settings.acceleration.mode=acceleration::AccelerationMode::Compatible;
            document.settings.sampling_backend=document::SamplingBackend::Cpu;
            document.nodes[0].settings=document.settings.clone();
        }
        if sample {
            document.prompt = "示例画布 · 手工图元，用于编辑和历史分支验收".into();
            let _ = document.commit_canvas(
                "添加示例背景",
                OperationSource::User,
                vec![Primitive::background([238, 242, 248])],
            );
            let mut p = document.primitives.clone();
            p.push(Primitive::new(
                ShapeKind::Ellipse,
                256.0,
                244.0,
                270.0,
                330.0,
                [252, 206, 164],
            ));
            let _ = document.commit_canvas("添加椭圆", OperationSource::User, p);
            let mut p = document.primitives.clone();
            let mut rectangle = Primitive::new(
                ShapeKind::Rectangle,
                180.0,
                150.0,
                128.0,
                260.0,
                [244, 192, 68],
            );
            rectangle.angle_deg = 28.0;
            p.push(rectangle);
            let _ = document.commit_canvas("添加旋转矩形", OperationSource::User, p);
            let fork = document.current;
            let mut p = document.primitives.clone();
            p.push(Primitive::new(
                ShapeKind::Ellipse,
                213.0,
                238.0,
                52.0,
                34.0,
                [54, 118, 214],
            ));
            p.push(Primitive::new(
                ShapeKind::Ellipse,
                304.0,
                238.0,
                52.0,
                34.0,
                [54, 118, 214],
            ));
            let _ = document.commit_canvas("添加两枚蓝色椭圆", OperationSource::User, p);
            let _ = document.jump(fork);
            let mut p = document.primitives.clone();
            p[2].color = [214, 132, 100];
            let _ = document.commit_canvas("另一个分支：修改矩形颜色", OperationSource::User, p);
            let _ = document.jump(fork + 1);
        }
        let mut fixture_error = resources.configuration_error.clone();
        let mut initial_history: Option<Vec<[u32; 9]>> = None;
        if let Some(path) = fixture {
            let result = (|| -> Result<(), String> {
                let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
                let report: serde_json::Value =
                    serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                document = Document::new();
                document.prompt = report["prompt"].as_str().unwrap_or("").to_string();
                if let Some(operations) = report["operations"].as_array() {
                    for operation in operations {
                        let parent =
                            operation["parent"].as_u64().ok_or("预览操作缺少父节点")? as usize;
                        document.jump(parent)?;
                        if let Some(prompt) = operation["prompt"].as_str() {
                            document.prompt = prompt.to_string();
                        }
                        if !operation["settings"].is_null() {
                            document.settings =
                                serde_json::from_value(operation["settings"].clone())
                                    .map_err(|e| e.to_string())?;
                        }
                        let primitives: Vec<Primitive> =
                            serde_json::from_value(operation["primitives"].clone())
                                .map_err(|e| e.to_string())?;
                        let node = document.commit_canvas(
                            operation["title"].as_str().unwrap_or("编辑预览图元"),
                            OperationSource::User,
                            primitives,
                        )?;
                        if let Some(expected) = operation["id"].as_u64() {
                            if node as u64 != expected {
                                return Err("预览操作节点编号不一致".into());
                            }
                        }
                    }
                    if let Some(current) = report["current_node"].as_u64() {
                        document.jump(current as usize)?;
                    }
                } else {
                    let rows: Vec<[u16; 9]> = serde_json::from_value(report["primitives"].clone())
                        .map_err(|e| e.to_string())?;
                    for tokens in rows {
                        document.commit_generated(Primitive::from_tokens(0, tokens)?)?;
                    }
                }
                if report["history_records"].is_array() {
                    initial_history = Some(
                        serde_json::from_value(report["history_records"].clone())
                            .map_err(|e| e.to_string())?,
                    );
                }
                Ok(())
            })();
            if let Err(e) = result {
                fixture_error = Some(format!("无法加载验收图元：{e}"));
            }
        }
        let mut editor = ui::EditorUi::new(&cc.egui_ctx);
        editor.canvas.rasterize_canvas=std::env::args().any(|arg|arg=="--rasterized-canvas");
        if std::env::args().any(|arg| arg == "--select-first-shape") {
            if let Some(primitive) = document.primitives.iter().find(|p| p.shape != ShapeKind::Background) {
                editor.canvas.selection = vec![primitive.id];
            }
        }
        if std::env::args().any(|arg|arg=="--settings") { editor.show_settings(); }
        if std::env::args().any(|arg|arg=="--prompt-templates") { editor.show_prompt_templates(); }
        if history_page {
            editor.show_history();
        }
        if history_page || history_compact || history_fit {
            editor.configure_history(history_compact, history_fit);
        }
        if std::env::args().any(|arg| arg == "--history-emphasis") {
            editor.configure_history_mode(tree_layout::TreeDisplayMode::EmphasizeBranches, history_fit);
        }
        let mut raw = vec![model::BOS_RECORD];
        for p in &document.primitives {
            if let Ok(tokens) = p.to_tokens() {
                raw.push(tokens.map(u32::from));
            }
        }
        let mut histories = RawHistoryStore::new();
        if let Some(history) = initial_history {
            match histories.finish(&document, document.current, &history) {
                Ok(()) => raw = history,
                Err(error) => fixture_error = Some(format!("预览区域历史无效：{error}")),
            }
        }
        let drawing_region = DrawingRegion::from_history(&raw).unwrap_or_default();
        if compatibility {
            document.settings.acceleration.mode=acceleration::AccelerationMode::Compatible;
            document.settings.sampling_backend=document::SamplingBackend::Cpu;
        }
        let compilation_preview = capture.is_some()
            && std::env::args().any(|arg| arg == "--compilation-preview");
        Self {
            redraw_view_key:None,redraw_smoke_done:false,mode_switch_smoke_done:false,manual_save_smoke_stage:0,manual_save_smoke_started:None,manual_save_smoke_checks:Vec::new(),manual_save_smoke_disk:Vec::new(),manual_save_smoke_id:String::new(),
            save_session:save_session::SaveSession::new(library_enabled),
            library_smoke_done:false,
            library_smoke_stage:0,library_smoke_checks:Vec::new(),
            file_task:None,
            reference_session:None,
            reference_smoke_done:false,
            pending_project_save:None,
            pending_project_import:None,
            file_busy_message:None,
            file_repaint:cc.egui_ctx.clone(),
            project_path:None,
            close_after_file:false,
            document,
            editor,
            runtime: Runtime::start(cc.egui_ctx.clone()),
            running: false,
            stopping: false,
            loading: compilation_preview,
            compiling: compilation_preview,
            acceleration_preparation_report: None,
            compatibility_startup: compatibility_startup::CompatibilityStartup::default(),
            loaded: false,
            auto_load_pending,
            pending_canvas_edit: None,
            model_status: if compilation_preview {
                "正在为当前显卡编译融合算子和采样模块（等待界面预览）".into()
            } else { "模型未加载".into() },
            execution_status: String::new(),
            graph_execution_status: String::new(),
            model_path,
            tokenizer_path,
            status: if compilation_preview {
                "正在为当前显卡准备融合算子和采样模块（界面预览）".into()
            } else {
                resources.discovery_notice.clone().unwrap_or_else(||"就绪 · 自动加载本地权重后即可生成".into())
            },
            error: fixture_error,
            histories,
            drawing_region,
            parallel: None,
            pending_global_edit:None,
            stage_by_id:BTreeMap::new(),stage_map_node:None,
            capture,
            frames: 0,
            self_test,
            self_test_stage: 0,
            capture_requested: false,
            self_test_lanes,
            self_test_history: history_page || history_compact || history_fit,
            self_test_base_primitives: 0,
            self_test_base_nodes: 1,
            self_test_origin: 0,
            self_test_count: 2,
            self_test_browse_edit: std::env::args().any(|arg| arg == "--ui-smoke-browse-edit"),
            self_test_browse_observed: false,
            self_test_edit_requested: false,
            ui_performance: None,
            keep_open,
            startup_report: {
                let args=std::env::args().collect::<Vec<_>>();
                args.iter().position(|arg|arg=="--startup-report").and_then(|index|args.get(index+1)).map(PathBuf::from)
            },
            startup_report_written: false,
            automatic_load_attempts: 0,
            editor_stage_smoke:{let args=std::env::args().collect::<Vec<_>>();args.iter().position(|arg|arg=="--editor-stage-smoke").and_then(|i|args.get(i+1)).map(PathBuf::from)},
            editor_stage_smoke_done:false,
        }
    }

    fn finish_pending_edit(&mut self) {
        if self.running { return; }
        if let Some((path,document,histories))=self.pending_project_import.take(){
            self.apply_imported_project(path,document,histories);
            return;
        }
        if self.save_session.pending_save{self.begin_library_save();return;}
        if let Some(path)=self.pending_project_save.take(){
            self.begin_project_save(path);
            return;
        }
        if let Some((node, action)) = self.pending_canvas_edit.take() {
            if matches!(&action, ui::UiAction::DeleteHistory(_)) {
                self.action(action);
                return;
            }
            match browse_history_node(&mut self.document, node) {
                Ok(()) => {
                    self.refresh_drawing_region();
                    self.action(action);
                }
                Err(error) => self.error = Some(error),
            }
        }
    }

    fn cancel_global_edit(&mut self){
        self.pending_global_edit=None;
        self.editor.cancel_global_transition();
    }

    fn commit_stage_edit(&mut self,label:String,primitives:Vec<Primitive>,assignment:Option<history_context::StageAssignmentRequest>,confirmed:bool){
        self.cancel_global_edit();
        let parent=self.document.current;
        let candidate=match self.document.preview_canvas(primitives){
            Ok(canvas)=>canvas,Err(error)=>{self.editor.notify(error);return;}
        };
        let plan=match self.histories.prepare_edit(&self.document,parent,candidate,assignment,confirmed){
            Ok(plan)=>plan,
            Err(history_context::StageEditBlock::GlobalOverflow{candidate_canvas,..})=>{
                self.pending_global_edit=Some(PendingGlobalEdit{origin:parent,label,canvas:candidate_canvas});
                self.editor.request_global_transition("全局绘制已达到上下文上限。确认后将把这次新增图形放入粗网格并进入下一阶段；之后修改全局数据需要更谨慎。关闭提示则不添加。");
                return;
            },
            Err(history_context::StageEditBlock::Rejected(error))=>{self.editor.notify(error);return;}
        };
        let old_prompt=self.document.prompt.clone();let old_settings=self.document.settings.clone();
        let notices=plan.notices.clone();
        let next=plan.canvas.clone();
        match self.document.commit_canvas(label.clone(),OperationSource::User,next){
            Ok(node)if node!=parent=>{
                match self.histories.apply_prepared_edit(&self.document,parent,node,plan){
                    Ok(region)=>{self.drawing_region=region;self.stage_map_node=None;self.error=None;self.status=label;
                        self.refresh_drawing_region();
                        for notice in notices{self.editor.notify(notice);}
                    },
                    Err(error)=>{
                        let rollback=self.document.rollback_last_canvas(node);
                        self.document.prompt=old_prompt;self.document.settings=old_settings;
                        self.editor.notify(if let Err(rollback)=rollback{format!("{error}；无法安全回滚：{rollback}")}else{error});
                        self.stage_map_node=None;self.refresh_drawing_region();
                    }
                }
            },
            Ok(_)=>{},Err(error)=>self.editor.notify(error),
        }
    }

    fn notify_stage_boundary(&mut self,stage:Option<region_state::Stage>){
        let message=match stage{
            Some(region_state::Stage::Global)=>"全局绘制阶段已结束，模型已中断。请检查画布，确认后重新点击生成，进入粗网格阶段。",
            Some(region_state::Stage::Coarse)=>"粗网格绘制阶段已结束，模型已中断。请检查画布，确认后重新点击生成，进入细网格阶段。",
            _=>return,
        };
        self.editor.notify(message);
    }

    fn generation_follows_view(&self) -> bool {
        self.parallel.as_ref().is_some_and(|parallel| parallel.is_following(&self.document))
    }

    fn set_file_busy(&mut self,message:Option<String>){
        self.editor.set_file_busy(message.is_some());
        self.file_busy_message=message;
    }

    fn start_file_job(&mut self,message:&str,make:impl FnOnce(&mut Self)->FileWork){
        let (sender,task)=match spawn_file_task(self.file_repaint.clone()){
            Ok(task)=>task,
            Err(error)=>{self.close_after_file=false;self.set_file_busy(None);self.save_session.pending_leave=None;self.save_session.pending_save=false;self.save_session.approved_close=false;self.editor.clear_unsaved_changes();self.error=Some(error);return;},
        };
        self.set_file_busy(Some(message.into()));
        self.file_task=Some(task);
        // Ownership leaves the app only after a worker was started. If the
        // input channel fails, recover the complete save payload immediately.
        if let Err(failed)=sender.send(make(self)){
            match failed.0 {
                FileWork::Save{document,histories,..}|FileWork::Library(save_session::LibraryWork::Save{document,histories,..})=>{
                    self.document=document;self.histories=histories;
                },_=>{},
            }
            self.file_task=None;self.set_file_busy(None);
            self.close_after_file=false;
            self.save_session.pending_leave=None;self.save_session.pending_save=false;self.save_session.approved_close=false;self.editor.clear_unsaved_changes();
            self.error=Some("文件任务未能启动，当前状态树已保留".into());
        }
    }

    fn begin_project_save(&mut self,path:PathBuf){
        self.start_file_job("正在保存状态树…",move|app|FileWork::Save{
            path,document:std::mem::take(&mut app.document),histories:std::mem::take(&mut app.histories),
        });
    }

    fn apply_imported_project(&mut self,path:PathBuf,document:Document,histories:RawHistoryStore){
        self.cancel_global_edit();self.stage_map_node=None;
        self.document=document;self.histories=histories;
        self.parallel=None;self.pending_canvas_edit=None;
        self.editor.invalidate_history_after_deletion();
        self.editor.set_project_path(Some(path.clone()));
        self.editor.show_history();self.project_path=Some(path);
        self.editor.locate_history_current();
        self.refresh_drawing_region();
        if self.self_test.is_some(){self.self_test_stage=3;}
        self.self_test_origin=self.document.current;
        self.self_test_base_nodes=self.document.nodes.len();
        self.self_test_base_primitives=self.document.primitives.len();
        self.status=format!("已导入 {} 个状态 · 当前状态 #{}",self.document.nodes.len(),self.document.current);
        self.error=None;self.set_file_busy(None);
        diagnostics::event("project_imported",serde_json::json!({"nodes":self.document.nodes.len(),"current":self.document.current}));
    }

    fn poll_file_task(&mut self){
        let message=match self.file_task.as_ref().map(|task|task.receiver.try_recv()){
            Some(Ok(message))=>message,
            Some(Err(std::sync::mpsc::TryRecvError::Disconnected))=>{
                self.file_task=None;self.set_file_busy(None);
                self.close_after_file=false;
                self.error=Some("文件处理线程提前结束，未发布新文件".into());return;
            },
            _=>return,
        };
        self.file_task=None;
        match message{
            FileOutcome::Library(outcome)=>self.poll_library_outcome(outcome),
            FileOutcome::Saved{path,document,histories,result}=>{
                self.document=document;self.histories=histories;
                self.set_file_busy(None);
                match result{
                    Ok(())=>{
                        if !self.save_session.enabled||self.save_session.active.is_none(){
                            self.editor.set_project_path(Some(path.clone()));self.project_path=Some(path.clone());
                        }
                        self.status=format!("状态树已保存：{}",path.file_name().unwrap_or_default().to_string_lossy());
                        self.error=None;
                        diagnostics::event("project_saved",serde_json::json!({"nodes":self.document.nodes.len(),"current":self.document.current}));
                    },
                    Err(error)=>{self.close_after_file=false;self.error=Some(format!("保存失败，当前状态树已保留：{error}"));},
                }
            },
            FileOutcome::Imported{path,result}=>match result{
                Ok((document,histories))=>{
                    if self.running{
                        if let Some(parallel)=&mut self.parallel{parallel.set_follow_live(false);}
                        self.pending_project_import=Some((path,document,histories));
                        self.runtime.stop();self.stopping=true;
                        self.set_file_busy(Some("项目已验证，正在停止生成后导入…".into()));
                    }else{self.apply_imported_project(path,document,histories);}
                },
                Err(error)=>{self.close_after_file=false;self.set_file_busy(None);self.error=Some(format!("导入失败，原状态树未改变：{error}"));},
            },
            FileOutcome::Exported{path,kind,result}=>{
                self.set_file_busy(None);
                match result{
                    Ok(())=>{
                        self.status=format!("{kind} 已保存：{}",path.file_name().unwrap_or_default().to_string_lossy());
                        self.error=None;diagnostics::event("canvas_exported",serde_json::json!({"format":kind}));
                    },
                    Err(error)=>{self.close_after_file=false;self.error=Some(format!("{kind} 导出失败：{error}"));},
                }
            },
        }
    }

    fn accept_generation_primitive(&mut self, lane: usize, tokens: [u32; 9], history: &runtime::RawHistory) {
        let viewed=self.document.current;
        if let Some(parallel) = &mut self.parallel {
            if let Err(error) = parallel.submit(&mut self.document, &mut self.histories, lane, tokens, history) {
                self.error = Some(error);
                self.runtime.stop();
                self.stopping = true;
            }
        }
        if self.document.current!=viewed{self.refresh_drawing_region();}
    }

    fn accept_generation_region(&mut self, lane: usize, region: DrawingRegion) {
        let follows = lane == 0 && self.generation_follows_view();
        if let Some(parallel) = &mut self.parallel {
            let _ = parallel.set_region(lane, region.clone());
        }
        if follows { self.drawing_region = region; }
    }

    fn drive_startup_model_load(&mut self) {
        // Render the first window before starting the existing background worker.
        // Consume this request even after a failed load; retries remain explicit.
        if (self.save_session.enabled&&self.save_session.home) || !self.auto_load_pending || self.frames < 2 || self.file_busy_message.is_some() {
            return;
        }
        self.auto_load_pending = false;
        if !self.loaded && !self.loading && !self.compiling && !self.running {
            self.automatic_load_attempts += 1;
            self.action(ui::UiAction::LoadModel {
                checkpoint_path: self.model_path.clone(),
                tokenizer_path: self.tokenizer_path.clone(),
            });
        }
    }

    fn report_startup(&mut self, error: Option<&str>) {
        if self.startup_report_written { return; }
        let Some(path)=&self.startup_report else { return; };
        let report=serde_json::json!({"status":if self.loaded {"complete"}else{"failed"},
            "model_loaded":self.loaded,"automatic_load_attempts":self.automatic_load_attempts,
            "parallel_sequences":self.document.settings.parallel_sequences,
            "compiling":self.compiling,"acceleration_preparation":self.acceleration_preparation_report,
            "pause_at_stage_end":self.document.settings.pause_at_stage_end,
            "model_path":self.model_path,"tokenizer_path":self.tokenizer_path,"model":self.model_status,"error":error});
        if let Some(parent)=path.parent() { let _=std::fs::create_dir_all(parent); }
        if let Ok(bytes)=serde_json::to_vec_pretty(&report) {
            if std::fs::write(path,bytes).is_ok() { self.startup_report_written=true; }
        }
    }

    fn refresh_drawing_region(&mut self) {
        let raw = match if self.is_redraw(){
            self.histories.context_for(&self.document,self.document.current).and_then(|raw|match raw{
                Some(records)=>Ok(history_context::EffectiveHistory{records,primitive_ids:Vec::new()}),
                None=>self.histories.effective_history(&self.document,self.document.current),
            })
        }else{self.histories.effective_history(&self.document,self.document.current)} {
            Ok(history) => history.records,
            Err(error) => {
                self.error = Some(error);
                return;
            }
        };
        match DrawingRegion::from_history(&raw) {
            Ok(mut region) => {
                if !self.is_redraw(){if let Ok(floor)=self.histories.stage_floor(&self.document,self.document.current){
                    let stage=match floor{Some(8)=>region_state::Stage::Fine,Some(4)=>region_state::Stage::Coarse,_=>region_state::Stage::Global};
                    // A completed stage's geometry stays editable until a real
                    // generation request enters the next stage. Ownership alone
                    // does not move the model's progress cursor.
                    if floor.is_none(){
                        region.stage=stage;region.side=1;region.index=None;region.awaiting_region=false;region.completed=false;
                        region.generated_in_region=raw.iter().filter(|row|model::validate_visible(row).is_ok()).count();
                    }else if region.stage!=stage && region.stage!=region_state::Stage::Complete {
                        let progress=region.progress;let coarse=region.completed_coarse;let fine=region.completed_fine;
                        region=DrawingRegion::waiting(stage,coarse,fine);region.progress=progress;
                    }
                }
                }
                self.drawing_region=region;
            },
            Err(error) => self.error = Some(error),
        }
    }

    fn drain_events(&mut self) {
        while let Ok(event) = self.runtime.events.try_recv() {
            let event_started = std::time::Instant::now();
            if matches!(&event,Event::ParallelPrimitive{..}|Event::Primitive{..}|Event::ParallelLaneFinished{..}
                |Event::ParallelFinished{..}|Event::Finished{..}){self.save_session.mark_dirty();}
            if let Some(perf) = &mut self.ui_performance { perf.observe(&event); }
            observe_acceleration_preparation(&event, &mut self.compiling,
                &mut self.acceleration_preparation_report);
            match event {
                Event::CompatibilityNotice(notice) => self.compatibility_startup.present(notice),
                Event::AccelerationPreparing { message } => {
                    self.loading = true;
                    self.loaded = false;
                    self.model_status = message.clone();
                    self.status = message;
                }
                Event::AccelerationPrepared { report } => {
                    self.status = report.summary.clone();
                    if self.compiling { self.model_status = report.summary; }
                }
                Event::ParallelRegion { lane, region } => {
                    self.accept_generation_region(lane, region);
                }
                Event::ParallelPrimitive {
                    lane,
                    tokens,
                    history,
                } => {
                    self.accept_generation_primitive(lane, tokens, &history);
                }
                Event::ParallelLaneFinished { lane, summary } => {
                    if let Some(parallel) = &mut self.parallel {
                        if let Err(error) = parallel.finish_lane(
                            &self.document,
                            &mut self.histories,
                            lane,
                            &summary.history_records,
                        ) {
                            self.error = Some(error);
                            self.runtime.stop();
                            self.stopping = true;
                        }
                    }
                }
                Event::ParallelFinished {
                    lanes,
                    generated,
                    stopped,
                    paused_after_stage,
                    elapsed_seconds: _,
                } => {
                    self.running = false;
                    self.stopping = false;
                    self.refresh_drawing_region();
                    self.status = format!(
                        "{} · {} 条分支共新增 {} 个完整图元 · 当前查看状态 #{}",
                        if let Some(stage) = paused_after_stage {
                            match stage {
                                region_state::Stage::Global => "全局阶段结束，已暂停；点击生成进入粗网格阶段",
                                region_state::Stage::Coarse => "粗网格阶段结束，已暂停；点击生成进入细网格阶段",
                                _ => "阶段结束，已暂停；点击生成继续",
                            }
                        } else if stopped { "并行生成已停止" } else { "并行生成结束" },
                        lanes,
                        generated,
                        self.document.current
                    );
                    if self.is_redraw(){self.status=if paused_after_stage==Some(region_state::Stage::Global){"全图阶段已完成；请选择粗网格区域继续"}else if stopped{"局部重绘已停止，完整图元已保留"}else if self.parallel.as_ref().is_some_and(|p|p.all_redraw_regions_complete()){"所选区域重绘完成"}else{"局部重绘已中断，完整图元已保留"}.into();}
                    self.notify_stage_boundary(paused_after_stage);
                    self.finish_pending_edit();
                }
                Event::Region(region) => self.accept_generation_region(0, region),
                Event::Status(message) => {
                    if let Some(notice)=message.strip_prefix("阶段检查："){
                        self.editor.notify(notice);self.runtime.stop();self.stopping=self.running;
                        self.status="正在中断模型，请检查当前块".into();continue;
                    }
                    if self.loading {
                        self.model_status = message.clone();
                    }
                    if message.starts_with("推理路径：") {
                        self.execution_status = message.clone();
                        self.graph_execution_status.clear();
                    } else if message.starts_with("整图元Graph") || message.starts_with("整图元CUDA Graph") || message.starts_with("区域阶段使用") {
                        self.graph_execution_status = message.split('：').next().unwrap_or(&message).to_owned();
                    }
                    self.status = message.lines().next().unwrap_or(&message).chars().take(180).collect();
                }
                Event::Loaded {
                    backend,
                    description,
                } => {
                    self.loaded = true;
                    self.loading = false;
                    self.model_status = format!("{description} · {backend}");
                    self.execution_status.clear();
                    self.graph_execution_status.clear();
                    self.status = "真实模型已加载，可以开始生成".into();
                    if let Err(error) = resources::save_model_paths(
                        std::path::Path::new(&self.model_path),
                        std::path::Path::new(&self.tokenizer_path),
                    ) {
                        self.status = format!("模型已加载，但路径未能保存：{error}");
                    }
                    self.report_startup(None);
                }
                Event::Primitive { tokens, history } => {
                    self.accept_generation_primitive(0, tokens, &history);
                }
                Event::Finished {
                    generated,
                    stopped,
                    paused_after_stage,
                    context_limit,
                    completed,
                    history,
                    elapsed_seconds: _,
                } => {
                    self.running = false;
                    self.stopping = false;
                    if let Some(parallel) = &mut self.parallel {
                        if let Err(error) = parallel.finish_lane(&self.document, &mut self.histories, 0, &history) {
                            self.error = Some(error);
                        }
                    }
                    self.status = format!(
                        "{} · 新增 {generated} 个完整图元{}",
                        if let Some(stage) = paused_after_stage {
                            match stage {
                                region_state::Stage::Global => "全局阶段结束，已暂停；点击生成进入粗网格阶段",
                                region_state::Stage::Coarse => "粗网格阶段结束，已暂停；点击生成进入细网格阶段",
                                _ => "阶段结束，已暂停；点击生成继续",
                            }
                        } else if stopped {
                            "已停止"
                        } else if completed {
                            "整幅生成完成"
                        } else {
                            "本次生成结束"
                        },
                        if context_limit {
                            " · 已到上下文边界"
                        } else {
                            ""
                        }
                    );
                    self.refresh_drawing_region();
                    if self.is_redraw(){self.status=if paused_after_stage==Some(region_state::Stage::Global){"全图阶段已完成；请选择粗网格区域继续"}else if stopped{"局部重绘已停止，完整图元已保留"}else if completed{"所选区域重绘完成"}else{"局部重绘已中断，完整图元已保留"}.into();}
                    self.notify_stage_boundary(paused_after_stage);
                    self.finish_pending_edit();
                }
                Event::Failed(message) => {
                    self.running = false;
                    self.stopping = false;
                    if self.loading {
                        self.loaded = false;
                        self.loading = false;
                        self.model_status = "模型加载失败".into();
                    }
                    if message.contains("CAPTURE_STATE_UNCERTAIN") {
                        self.loaded = false;
                        self.model_status = "捕获状态不确定，请重新启动应用；已保留完整画布".into();
                    } else if message.contains("重新加载模型") || message.contains("CUDA_ERROR_ILLEGAL_ADDRESS") || message.contains("CUDA_ERROR_LAUNCH_FAILED") {
                        self.loaded = false;
                        self.model_status = "执行后端已停止，请重新加载模型；已保留完整画布".into();
                    }
                    self.report_startup(Some(&message));
                    self.error = Some(message);
                    if let Some(parallel) = &mut self.parallel {
                        for lane in 0..parallel.heads.len() {
                            let _ = parallel.fail_lane(lane, "推理任务中止".to_string());
                        }
                    }
                    self.refresh_drawing_region();
                    let failure = self.error.clone();
                    self.finish_pending_edit();
                    if failure.is_some() { self.error = failure; }
                }
            }
            if let Some(perf) = &mut self.ui_performance { perf.event_cpu(event_started.elapsed().as_secs_f64()); }
        }
    }

    fn action(&mut self, action: ui::UiAction) {
        if self.file_busy_message.is_some() && !matches!(&action,ui::UiAction::Stop){return;}
        if self.save_session.enabled&&self.save_session.home&&!self.save_session.settings
            &&!matches!(&action,ui::UiAction::OpenLogs|ui::UiAction::ExportDiagnostics|ui::UiAction::Stop|ui::UiAction::LoadModel{..}){return;}
        if matches!(&action,ui::UiAction::CommitCanvas{..}|ui::UiAction::PasteCanvas{..}|ui::UiAction::AssignStage{..}
            |ui::UiAction::Jump(_)|ui::UiAction::BrowseHistory(_)|ui::UiAction::Generate{..}|ui::UiAction::DeleteHistory(_)
            |ui::UiAction::ReorderByContribution|ui::UiAction::SetSettings(_)|ui::UiAction::SetPrompt(_)|ui::UiAction::ConfirmGlobalTransition){self.save_session.mark_dirty();self.redraw_view_key=None;}
        if matches!(&action,ui::UiAction::CommitCanvas{..}|ui::UiAction::PasteCanvas{..}|ui::UiAction::AssignStage{..}
            |ui::UiAction::Jump(_)|ui::UiAction::BrowseHistory(_)|ui::UiAction::Generate{..}|ui::UiAction::DeleteHistory(_)
            |ui::UiAction::ReorderByContribution|ui::UiAction::SetSettings(_)
            |ui::UiAction::SetPrompt(_)|ui::UiAction::ExportPng{..}|ui::UiAction::ExportSvg(_)
            |ui::UiAction::LoadModel{..}|ui::UiAction::Stop|ui::UiAction::SwitchWorkspaceMode{..}
            |ui::UiAction::SaveProject(_)|ui::UiAction::ImportProject(_)) {self.cancel_global_edit();}
        if self.running && matches!(&action,
            ui::UiAction::CommitCanvas { .. } | ui::UiAction::PasteCanvas { .. }
                | ui::UiAction::ReorderByContribution | ui::UiAction::DeleteHistory(_)|ui::UiAction::AssignStage{..})
        {
            if self.pending_canvas_edit.is_none() {
                if let Some(parallel) = &mut self.parallel { parallel.set_follow_live(false); }
                let deleting = matches!(&action, ui::UiAction::DeleteHistory(_));
                self.pending_canvas_edit = Some((self.document.current, action));
                self.runtime.stop();
                self.stopping = true;
                self.status = if deleting {
                    "正在停止生成，随后删除所选状态及全部后代…"
                } else { "正在停止生成，随后应用你的画布编辑…" }.into();
            }
            return;
        }
        match action {
            ui::UiAction::OpenReference=>self.open_reference_import(),
            ui::UiAction::Reference(action)=>self.reference_action(action),
            ui::UiAction::ResolveUnsaved{choice}=>self.resolve_unsaved_choice(choice),
            ui::UiAction::SwitchWorkspaceMode{redraw}=>{self.switch_workspace_mode(redraw);},
            ui::UiAction::ToggleRedrawRegion{stage,index}=>self.toggle_redraw_region(stage,index),
            ui::UiAction::DeleteRedrawRegions=>self.delete_redraw_regions(),
            ui::UiAction::DeleteLockedRedrawRegions=>self.delete_locked_redraw_regions(),
            ui::UiAction::ReturnToLibrary=>{if self.save_session.enabled{self.return_to_library();}},
            ui::UiAction::CancelGlobalTransition=>self.cancel_global_edit(),
            ui::UiAction::ConfirmGlobalTransition if !self.running&&!self.loading&&!self.compiling&&!self.stopping=>{
                if let Some(edit)=self.pending_global_edit.take(){
                    self.editor.cancel_global_transition();
                    if edit.origin==self.document.current{self.commit_stage_edit(edit.label,edit.canvas,None,true);}
                }
            },
            ui::UiAction::AssignStage{ids,stage} if !self.running&&!self.loading&&!self.compiling&&!self.stopping=>{
                self.commit_stage_edit(format!("将所选图形归入{}",stage.label()),self.document.primitives.clone(),
                    Some(history_context::StageAssignmentRequest{ids,stage}),false);
            },
            ui::UiAction::SaveProject(path) if !self.loading&&!self.compiling&&!self.stopping=>{
                if self.save_session.enabled&&self.project_path.as_ref()==Some(&path){self.request_library_save();return;}
                if self.running{
                    if let Some(parallel)=&mut self.parallel{parallel.set_follow_live(false);}
                    self.pending_project_save=Some(path);self.runtime.stop();self.stopping=true;
                    self.set_file_busy(Some("正在停止生成并保存完整状态树…".into()));
                }else{self.begin_project_save(path);}
            },
            ui::UiAction::ImportProject(path) if !self.loading&&!self.compiling&&!self.stopping=>{
                if self.save_session.enabled{
                    self.request_leave(manual_save::LeaveAction::Import(path));return;
                }
                // Validate in isolation first. A corrupt/cancelled import never
                // replaces the old tree or stops an otherwise healthy worker.
                self.start_file_job("正在读取并验证状态树…",move|_|FileWork::Import{path});
            },
            ui::UiAction::ExportPng{path,resolution} if !self.loading&&!self.compiling=>{
                self.start_file_job("正在导出 PNG…",move|app|FileWork::Png{path,resolution,primitives:app.document.primitives.clone()});
            },
            ui::UiAction::ExportSvg(path) if !self.loading&&!self.compiling=>{
                self.start_file_job("正在导出 SVG…",move|app|FileWork::Svg{path,primitives:app.document.primitives.clone()});
            },
            ui::UiAction::DeleteHistory(ids)
                if !self.running && !self.stopping && !self.loading && !self.compiling => {
                match delete_history_subtrees(&mut self.document, &mut self.histories, &ids) {
                    Ok(result) if !result.removed_nodes.is_empty() => {
                        // The worker has delivered its terminal event before
                        // IDs are compacted; no late completed row is lost.
                        self.parallel = None;
                        self.stage_map_node=None;
                        self.pending_canvas_edit = None;
                        self.editor.invalidate_history_after_deletion();
                        self.refresh_drawing_region();
                        // Diagnostics holding old node IDs cannot continue
                        // across a manual destructive history operation.
                        if self.self_test.is_some() { self.self_test_stage = 3; }
                        self.self_test_origin = self.document.current;
                        self.self_test_base_nodes = self.document.nodes.len();
                        self.self_test_base_primitives = self.document.primitives.len();
                        self.error = None;
                        self.status = if result.root_reset {
                            "已清空状态树，回到初始画布".into()
                        } else {
                            format!("已删除 {} 个状态（包含全部后代） · 当前状态 #{}",
                                result.removed_nodes.len(), self.document.current)
                        };
                        diagnostics::event("history_subtrees_deleted", serde_json::json!({
                            "selected_roots":ids.len(),"removed_nodes":result.removed_nodes.len(),
                            "remaining_nodes":self.document.nodes.len(),"current":self.document.current,
                            "current_was_deleted":result.current_was_deleted,"root_reset":result.root_reset,
                        }));
                    },
                    Ok(_) => {},
                    Err(error) => self.error = Some(error),
                }
            },
            ui::UiAction::OpenLogs => {
                match diagnostics::open_directory(){
                    Ok(())=>self.status="已打开日志目录".into(),
                    Err(error)=>{diagnostics::error("diagnostics.open",&error);self.error=Some(error.to_string());}
                }
            }
            ui::UiAction::ExportDiagnostics => {
                match diagnostics::export_diagnostics(None){
                    Ok(path)=>{
                        diagnostics::event("diagnostics.exported",serde_json::json!({"archive_name":path.file_name().map(|n|n.to_string_lossy())}));
                        self.status=format!("诊断已导出：{}",path.display());
                        let _=diagnostics::open_directory();
                    }
                    Err(error)=>{diagnostics::error("diagnostics.export",&error);self.error=Some(error.to_string());}
                }
            }
            ui::UiAction::FreezeGenerationView => {
                if let Some(parallel) = &mut self.parallel { parallel.set_follow_live(false); }
                self.refresh_drawing_region();
            }
            ui::UiAction::SetPrompt(prompt) if !self.running && !self.loading => {
                self.document.prompt = prompt
            }
            ui::UiAction::SetSettings(settings) if !self.running && !self.loading => {
                self.document.settings = settings
            }
            ui::UiAction::ReorderByContribution if !self.running && !self.loading => {
                match reorder_current_canvas(&mut self.document, &mut self.histories) {
                    Ok((changed, region)) => {
                        self.drawing_region = region;
                        self.error = None;
                        self.status = if changed { "已在各自区域内按贡献重排序，可在状态树返回原顺序" } else { "各区域内图形已按贡献排序，无需调整" }.into();
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            ui::UiAction::CommitCanvas { label, primitives } if !self.running && !self.loading => {
                self.commit_stage_edit(label,primitives,None,false);
            }
            ui::UiAction::PasteCanvas { label, primitives } if !self.running && !self.loading => {
                let parent = self.document.current;
                let previous_ids: std::collections::HashSet<_> =
                    self.document.primitives.iter().map(|primitive| primitive.id).collect();
                self.action(ui::UiAction::CommitCanvas { label, primitives });
                if self.document.current != parent {
                    self.editor.canvas.selection = self.document.primitives.iter()
                        .filter(|primitive| primitive.shape != ShapeKind::Background
                            && !previous_ids.contains(&primitive.id))
                        .map(|primitive| primitive.id).collect();
                }
            }
            ui::UiAction::BrowseHistory(node) if !self.loading&&!self.compiling&&!self.stopping => {
                match usize::try_from(node).map_err(|_|"状态编号超出本机范围".to_string())
                    .and_then(|node|browse_history_session(&mut self.document,&mut self.parallel,node)) {
                    Ok(())=>{
                        self.editor.reset_generation_view_freeze();
                        self.refresh_drawing_region();
                        self.status=if self.running{format!("正在翻阅状态 #{node}；模型继续在原分支生成")}
                            else{format!("正在翻阅状态 #{node}")};
                    },Err(error)=>self.error=Some(error),
                }
            },
            ui::UiAction::Jump(node) => {
                if self.loading {
                    return;
                }
                match select_history_node(&mut self.document, node as usize) {
                    Ok(()) => {
                        self.editor.reset_generation_view_freeze();
                        if let Some(parallel) = &mut self.parallel {
                            parallel.set_follow_live(parallel.heads.first() == Some(&(node as usize)));
                        }
                        self.refresh_drawing_region();
                        self.status = if self.running {
                            format!("正在查看状态 #{node}；模型继续在原分支生成")
                        } else { format!("已切换到状态 #{node}；下一次操作将从这里继续") };
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            ui::UiAction::Stop if self.running => {
                self.runtime.stop();
                self.stopping = true;
                self.status = "正在停止 · 尚未完成的图元不会进入画布".into();
            }
            ui::UiAction::Generate { prompt, settings }
                if generation_is_ready(self.running, self.loading, self.compiling, self.loaded)
                    && !self.compatibility_startup.is_open() =>
            {
                if self.is_redraw(){self.start_redraw_generation(prompt,settings);return;}
                match self.histories.requires_normal_restart(&self.document,self.document.current){
                    Ok(true)=>{if !self.resume_standard_workflow(){return;}},
                    Ok(false)=>{},Err(error)=>{self.editor.show_mode_issues(vec![error]);return;},
                }
                if let Some(reason) = self.drawing_region.generation_block_reason() {
                    self.error = Some(reason.into());
                    return;
                }
                if let Err(e) = settings.validate() {
                    self.error = Some(e);
                    return;
                }
                let validation = inference::prompt_validation::PromptValidator::open(&self.tokenizer_path)
                    .and_then(|validator| validator.validate(&prompt).into_token_ids());
                if let Err(error) = validation {
                    self.error = Some(error);
                    return;
                }
                let settings=match settings_for_generation(settings){
                    Ok(settings)=>settings,Err(error)=>{self.error=Some(error);return;}
                };
                self.document.prompt = prompt;
                self.document.settings = settings;
                if self.document.settings.auto_reorder_before_generation {
                    match reorder_current_canvas(&mut self.document, &mut self.histories) {
                        Ok((_, region)) => self.drawing_region = region,
                        Err(error) => {
                            self.error = Some(error);
                            return;
                        }
                    }
                }
                let effective=match self.histories.effective_history(&self.document,self.document.current){
                    Ok(value)=>value,
                    Err(e) => {
                        self.error = Some(e);
                        return;
                    }
                };
                let by_id=self.document.primitives.iter().map(|p|(p.id,p)).collect::<std::collections::HashMap<_,_>>();
                let context_primitives=match effective.primitive_ids.iter().map(|id|by_id.get(id).map(|p|(*p).clone()).ok_or("有效上下文引用了缺失图形".to_string())).collect::<Result<Vec<_>,_>>(){
                    Ok(values)=>values,Err(error)=>{self.error=Some(error);return;}
                };
                let extra_fine_context=match self.histories.deferred_fine_context(&self.document,self.document.current){
                    Ok(values)=>values,Err(error)=>{self.error=Some(error);return;}
                };
                let history=Some(effective.records);
                let stage_floor=match self.histories.stage_floor(&self.document,self.document.current){Ok(value)=>value,Err(error)=>{self.error=Some(error);return;}};
                let pending_stage_context=match self.histories.pending_stage_context(&self.document,self.document.current){Ok(value)=>value,Err(error)=>{self.error=Some(error);return;}};
                let manual_review_regions=match self.histories.manual_review_regions(&self.document,self.document.current) {
                    Ok(regions)=>regions,
                    Err(error)=>{self.error=Some(error);return;}
                };
                self.parallel = match ParallelHistory::new(
                        &self.document,
                        self.document.current,
                        self.document.prompt.clone(),
                        self.document.settings.clone(),
                    ) {
                        Ok(parallel) => Some(parallel),
                        Err(error) => {
                            self.error = Some(error);
                            return;
                        }
                };
                if let Some(parallel)=&mut self.parallel{
                    if let Err(error)=parallel.initialize_model_input(&self.document,&self.histories){self.error=Some(error);return;}
                }
                self.error = None;
                self.running = true;
                self.stopping = false;
                let diagnostic_limit=(self.self_test.is_some() && self.self_test_stage == 2).then_some(self.self_test_count);
                self.runtime.generate_with_stage_context(self.document.prompt.clone(), self.document.settings.clone(),
                    context_primitives,history,manual_review_regions,extra_fine_context,stage_floor,pending_stage_context,diagnostic_limit);
                self.status = if self.document.settings.parallel_sequences > 1 {
                    format!(
                        "正在从当前节点准备 {} 条分支 · 画布显示第 1 分支",
                        self.document.settings.parallel_sequences
                    )
                } else {
                    "正在准备当前节点的模型上下文…".into()
                };
            }
            ui::UiAction::LoadModel {
                checkpoint_path,
                tokenizer_path,
            } if !self.running && !self.loading && !self.compiling => {
                if !std::path::Path::new(&checkpoint_path).is_file() {
                    self.error = Some("权重文件不存在，请在设置中检查路径".into());
                    return;
                }
                if !std::path::Path::new(&tokenizer_path).is_file() {
                    self.error = Some("分词器文件不存在，请指定 caption_spm_1500.model".into());
                    return;
                }
                self.model_path = checkpoint_path;
                self.tokenizer_path = tokenizer_path;
                self.error = None;
                self.loaded = false;
                self.loading = true;
                self.compiling = false;
                self.acceleration_preparation_report = None;
                self.model_status = "正在加载原生推理模型…".into();
                self.status = "正在加载本地权重，请稍候…".into();
                self.runtime.load_with_settings(
                    PathBuf::from(&self.model_path),
                    PathBuf::from(&self.tokenizer_path),
                    self.document.settings.clone(),
                );
            }
            _ => {}
        }
    }

    /// Runs the actual UI action -> worker -> document path in an isolated QA
    /// window; this does not inject input into any other Windows application.
    fn drive_self_test(&mut self, ctx: &egui::Context) {
        if self.self_test.is_none() || self.frames < 2 || self.self_test_stage == 3 {
            return;
        }
        match self.self_test_stage {
            0 => {
                self.self_test_stage = 1;
                self.action(ui::UiAction::LoadModel {
                    checkpoint_path: self.model_path.clone(),
                    tokenizer_path: self.tokenizer_path.clone(),
                });
            }
            1 if self.loaded && !self.loading && !self.compiling => {
                self.self_test_stage = 2;
                let mut settings = self.document.settings.clone();
                settings.parallel_sequences = self.self_test_lanes;
                let cli: Vec<_> = std::env::args().collect();
                if let Some(index)=cli.iter().position(|arg|arg=="--ui-smoke-acceleration") {
                    settings.acceleration.mode=match cli.get(index+1).map(String::as_str) {
                        Some("optimized")=>acceleration::AccelerationMode::Optimized,
                        Some("compatible")=>acceleration::AccelerationMode::Compatible,
                        _=>acceleration::AccelerationMode::Auto,
                    };
                }
                self.self_test_count = cli.iter().position(|arg| arg == "--ui-smoke-count")
                    .and_then(|index| cli.get(index+1)).and_then(|value| value.parse::<usize>().ok()).filter(|&n| n>0).unwrap_or(2);
                if cli.iter().any(|arg| arg == "--ui-smoke-perf") {
                    self.ui_performance = Some(ui_perf::UiPerformance::new(self.self_test_lanes));
                }
                if cli.iter().any(|arg| arg == "--ui-smoke-overclock") {
                    settings.context_overclock = true;
                }
                settings.auto_reorder_before_generation = cli.iter().any(|arg| arg == "--ui-smoke-auto-reorder");
                if let Some(index) = cli.iter().position(|x| x == "--ui-smoke-sampler") {
                    settings.sampling_backend = match cli.get(index + 1).map(String::as_str) {
                        Some("gpu") => document::SamplingBackend::Gpu,
                        Some("cpu") => document::SamplingBackend::Cpu,
                        _ => document::SamplingBackend::Auto,
                    };
                }
                self.self_test_base_primitives = self.document.primitives.len();
                self.action(ui::UiAction::Generate { prompt: "金发蓝眼的白皙圆脸女性以明亮动画风呈现，张嘴露舌，粉色背景可见，取景包含头部和肩颈。".into(), settings });
                self.self_test_base_nodes = self.document.nodes.len();
                self.self_test_origin = self.document.current;
                if self.self_test_browse_edit {
                    self.action(ui::UiAction::Jump(0));
                }
            }
            _ => {}
        }
        if self.self_test_browse_edit && self.self_test_stage == 2
            && !self.self_test_edit_requested && self.running
            && self.parallel.as_ref().is_some_and(|parallel| parallel.generated.iter().sum::<usize>() > 0)
        {
            self.self_test_browse_observed = self.document.current == 0 && !self.stopping
                && self.self_test_origin != 0;
            self.self_test_edit_requested = true;
            self.action(ui::UiAction::CommitCanvas {
                label: "验收：浏览根节点后手动设置背景".into(),
                primitives: vec![Primitive::background([100, 120, 130])],
            });
        }
        let done = self.self_test_stage == 2 && !self.running && !self.loading && !self.compiling;
        if done || self.error.is_some() {
            let parallel_ok = if self.self_test_lanes > 1 {
                self.parallel
                    .as_ref()
                    .map(|p| {
                        p.heads.len() == self.self_test_lanes
                            && self.document.current == p.heads[0]
                            && p.generated.iter().all(|&count| count == self.self_test_count)
                    })
                    .unwrap_or(false)
            } else {
                true
            };
            let ordinary_passed = done
                && self.error.is_none()
                && self.document.primitives.len() == self.self_test_base_primitives + self.self_test_count
                && self.document.nodes.len() == self.self_test_base_nodes + self.self_test_count * self.self_test_lanes
                && parallel_ok;
            let edit_applied = self.document.current_node().parent == Some(0)
                && self.document.primitives.len() == 1
                && self.document.primitives[0].shape == ShapeKind::Background
                && self.document.primitives[0].color == [100, 120, 130]
                && self.pending_canvas_edit.is_none();
            let passed = if self.self_test_browse_edit {
                done && self.error.is_none() && self.self_test_browse_observed
                    && self.self_test_edit_requested && edit_applied
                    && self.parallel.as_ref().is_some_and(|parallel| {
                        parallel.heads.iter().all(|head| *head != self.document.current)
                            && parallel.generated.iter().sum::<usize>() > 0
                    })
            } else { ordinary_passed };
            let report = serde_json::json!({ "status":if passed {"complete"} else {"failed"},
                "scope":"actual_native_window_ui_actions_worker_document",
                "model_loaded":self.loaded,"model":self.model_status,"error":self.error,
                "compiling":self.compiling,"acceleration_preparation":self.acceleration_preparation_report,
                "primitives":self.document.primitives,"node_count":self.document.nodes.len(),
                "current_node":self.document.current,"inference_thread_and_ui_messages_exercised":true,
                "startup_compatibility_notice":self.compatibility_startup.snapshot(),
                "parallel_sequences": self.self_test_lanes,"only_first_lane_shown":parallel_ok,
                "context_overclock": self.document.settings.context_overclock,
                "pause_at_stage_end": self.document.settings.pause_at_stage_end,
                "half_region_switch": self.document.settings.half_region_switch,
                "auto_seed":self.document.settings.auto_seed,
                "actual_seed":self.document.settings.seed,
                "pending_manual_reviews":self.histories.manual_review_regions(&self.document,self.document.current).ok(),
                "browse_edit_scenario": self.self_test_browse_edit,
                "generation_continued_while_browsing": self.self_test_browse_observed,
                "edit_requested_stop": self.self_test_edit_requested,
                "edit_applied_to_browsed_root": edit_applied,
                "auto_reorder_before_generation": self.document.settings.auto_reorder_before_generation,
                "generation_origin_node": self.self_test_origin,
                "generation_origin_ids": self.document.materialize(self.self_test_origin).ok().map(|canvas| canvas.into_iter().map(|p| p.id).collect::<Vec<_>>()),
                "branch_heads":self.parallel.as_ref().map(|p| &p.heads),
                "branch_seeds":self.parallel.as_ref().map(|p| &p.seeds),
                "branch_generated_counts":self.parallel.as_ref().map(|p| &p.generated) });
            let mut report = report;
            if let Some(perf) = &self.ui_performance { report["performance"] = perf.report(); }
            if let Some(path) = &self.self_test {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Ok(bytes) = serde_json::to_vec_pretty(&report) {
                    let _ = std::fs::write(path, bytes);
                }
            }
            self.self_test_stage = 3;
            if self.self_test_history {
                self.editor.configure_history(false, true);
            }
            if self.capture.is_none() && !self.keep_open {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn maybe_pending_compatibility_notice(&mut self){
        if std::env::args().any(|a|a=="--library-smoke-output"||a=="--redraw-smoke-output"||a=="--mode-switch-smoke-output"||a=="--manual-save-smoke-output"||a=="--reference-smoke-output")||(self.save_session.enabled&&self.save_session.home&&!self.save_session.settings)||self.compatibility_startup.was_presented()||self.loaded||self.loading||self.compiling||self.auto_load_pending||self.self_test.is_some()||self.editor_stage_smoke.is_some(){return}
        let provider=std::env::var("PAINTER_COMPUTE_PROVIDER").unwrap_or_else(|_|{
            if cfg!(feature="aten-hip"){"amd"}else if cfg!(feature="cuda"){"nvidia"}else{"cpu"}.into()
        });
        let compatible=self.document.settings.acceleration.mode==acceleration::AccelerationMode::Compatible;
        if provider!="cpu"&&!compatible{return}
        let backend=match provider.as_str(){"amd"=>"aten_hip","nvidia"=>"cuda",_=>"cpu"};
        let actual=serde_json::json!({"provider":if backend=="cuda"{"native_candle_cuda"}else{backend},
            "model_loaded":false,"capabilities":{"backend":backend},"active_features":acceleration::FeatureSet::default()});
        let mut reason=std::env::var("PAINTER_SELECTION_REASON").unwrap_or_default();
        reason.push_str("；当前尚未加载模型，可以关闭提示后在设置中选择或加载权重。");
        let session=diagnostics::session_id();let directory=diagnostics::directory();
        if let Some(notice)=compatibility_notice::CompatibilityNotice::from_context(&compatibility_notice::NoticeContext{
            actual:&actual,requested:&self.document.settings.acceleration,sampling_override:None,preparation:None,
            selection_reason:&reason,session_id:&session,log_directory:directory.as_deref(),application_version:env!("CARGO_PKG_VERSION"),
        }){self.compatibility_startup.present(notice);}
    }
}

impl eframe::App for PainterApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frames += 1;
        self.drain_events();
        self.drive_editor_stage_smoke();
        self.drive_library_smoke();
        self.drive_redraw_smoke();
        self.drive_mode_switch_smoke();
        self.drive_manual_save_smoke();
        self.drive_reference_smoke();
        if self.save_session.active.is_none()&&self.file_busy_message.is_some()&&ctx.input(|i|i.viewport().close_requested()){
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.close_after_file=true;
            if self.running {self.runtime.stop();self.stopping=true;}
        }
        self.poll_file_task();
        self.poll_reference_session();
        if self.file_busy_message.is_none() {
            if let Some(action)=self.editor.poll_model_download(!self.running&&!self.loading&&!self.compiling&&!self.stopping) {
                self.auto_load_pending=false;
                self.action(action);
            }
        }
        if self.close_after_file&&self.file_busy_message.is_none(){
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        self.drive_pending_leave();
        self.editor.set_file_busy(self.file_busy_message.is_some());
        self.drive_startup_model_load();
        self.maybe_pending_compatibility_notice();
        if !self.compatibility_startup.is_open()&&self.file_busy_message.is_none(){self.drive_self_test(ctx);}
        if self.startup_report.is_some() && !self.startup_report_written && !self.auto_load_pending && !self.loading && !self.loaded {
            let error=self.error.clone();
            if error.is_some() { self.report_startup(error.as_deref()); }
        }
        if self.startup_report_written && std::env::args().any(|arg|arg=="--quit-after-startup") {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if self.compatibility_startup.is_open()&&!self.loading&&!self.compiling{
            egui::CentralPanel::default().show(ctx,|_|{});
            self.compatibility_startup.show(ctx,self.loaded,self.running,self.document.nodes.len());
        }else if self.save_session.enabled&&self.save_session.home&&!self.save_session.settings{
            self.render_library(ctx);
        }else{
        if self.file_busy_message.is_none(){self.refresh_redraw_ui();}
        let view_started = std::time::Instant::now();
        let nodes: Vec<_> = self
            .document
            .nodes
            .iter()
            .map(|n| ui::NodeView {
                id: n.id as u64,
                parent: n.parent.map(|v| v as u64),
                title: n.title.clone(),
                source: match n.source {
                    OperationSource::Initial => "初始",
                    OperationSource::User => "用户",
                    OperationSource::Model => "模型",
                }
                .into(),
                primitive_count: n.primitive_count,
            })
            .collect();
        let execution_description = format!("{}\n{}\n{}",self.model_status,self.execution_status,self.graph_execution_status);
        if self.stage_map_node!=Some(self.document.current){
            match self.histories.stage_map(&self.document,self.document.current){
                Ok(map)=>self.stage_by_id=map,
                Err(error)=>{self.stage_by_id.clear();self.error=Some(error);}
            }
            self.stage_map_node=Some(self.document.current);
        }
        let model = ui::UiModel {
            document: &self.document,
            stage_by_id:Some(&self.stage_by_id),
            drawing_region: &self.drawing_region,
            primitives: &self.document.primitives,
            prompt: &self.document.prompt,
            settings: &self.document.settings,
            current_node: self.document.current as u64,
            nodes: &nodes,
            running: self.running,
            loading: self.loading,
            compiling: self.compiling,
            stopping: self.stopping,
            model_loaded: self.loaded,
            model_status: &execution_description,
            model_path: &self.model_path,
            tokenizer_path: &self.tokenizer_path,
            error: self.error.as_deref(),
            status: &self.status,
        };
        let node_view_seconds = view_started.elapsed().as_secs_f64();
        let ui_started = std::time::Instant::now();
        self.editor.set_save_name(self.save_session.active.as_ref().map(|e|e.display_name.clone()));
        self.editor.set_workflow_mode(self.is_redraw());
        self.editor.set_unsaved(self.save_session.dirty);
        let actions = if self.save_session.enabled&&self.save_session.home&&self.save_session.settings{
            let (actions,back)=self.editor.render_library_settings(ctx,&model);
            if back{self.save_session.settings=false;self.save_session.error=None;}
            actions
        }else{self.editor.render(ctx, &model)};
        if let Some(perf) = &mut self.ui_performance {
            if self.self_test_stage == 2 { perf.frame_cpu(node_view_seconds, ui_started.elapsed().as_secs_f64()); }
        }
        if actions.iter().any(|a|matches!(a,ui::UiAction::ReturnToLibrary|ui::UiAction::SwitchWorkspaceMode{..})){
            // A toolbar click and the final text edit may arrive together.
            // Commit this frame's drafts before transferring ownership to I/O.
            for draft in actions.iter().filter(|a|matches!(a,ui::UiAction::SetPrompt(_)|ui::UiAction::SetSettings(_))).cloned().collect::<Vec<_>>(){self.action(draft);}
        }
        let mut file_actions=Vec::new();
        let mut structural=false;
        for action in actions {
            if matches!(&action,ui::UiAction::SaveProject(_)|ui::UiAction::ImportProject(_)
                |ui::UiAction::ExportPng{..}|ui::UiAction::ExportSvg(_)){
                file_actions.push(action);continue;
            }
            structural = matches!(&action, ui::UiAction::DeleteHistory(_)|ui::UiAction::ReturnToLibrary|ui::UiAction::SwitchWorkspaceMode{..});
            self.action(action);
            // Remaining UI actions were authored against the old node IDs.
            if structural { break; }
        }
        if !structural{for action in file_actions{self.action(action);}}
        }
        self.editor.set_unsaved(self.save_session.dirty);
        self.guard_library_close(ctx);
        self.drive_pending_leave();
        if self.capture.is_some()
            && self.frames >= 5
            && (!std::env::args().any(|a|a=="--library-smoke-output")||self.library_smoke_done)
            && (!std::env::args().any(|a|a=="--redraw-smoke-output")||(self.redraw_smoke_done&&self.frames>=240))
            && (!std::env::args().any(|a|a=="--mode-switch-smoke-output")||self.mode_switch_smoke_done)
            && (!std::env::args().any(|a|a=="--manual-save-smoke-output")||self.manual_save_smoke_stage>=9)
            && (!std::env::args().any(|a|a=="--reference-smoke-output")||self.reference_smoke_done)
            && !self.capture_requested
            && (self.self_test.is_none() || self.self_test_stage == 3 || self.compatibility_startup.capture_requested())
        {
            self.capture_requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
        }
        let screenshots: Vec<_> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|e| {
                    if let egui::Event::Screenshot { image, .. } = e {
                        Some(image.clone())
                    } else {
                        None
                    }
                })
                .collect()
        });
        for shot in screenshots {
            if let Some(path) = self.capture.take() {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let bytes: Vec<u8> = shot.pixels.iter().flat_map(|c| c.to_array()).collect();
                if let Err(e) = image::save_buffer(
                    &path,
                    &bytes,
                    shot.size[0] as u32,
                    shot.size[1] as u32,
                    image::ColorType::Rgba8,
                ) {
                    eprintln!("Screenshot failed: {e}");
                }
                if !self.keep_open {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
        if self.running
            || self.loading
            || self.compiling
            || self.auto_load_pending
            || self.file_busy_message.is_some()
            || self.compatibility_startup.is_open()
            || self.capture.is_some()
            || (self.self_test.is_some() && self.self_test_stage != 3)
        {
            ctx.request_repaint_after(std::time::Duration::from_millis(40));
        }
    }
}

fn main() -> eframe::Result<()> {
    let _diagnostics=diagnostics::init("frontend");
    let args: Vec<_> = std::env::args().collect();
    if let Some(index)=args.iter().position(|arg|arg=="--reference-serializer-self-test"){
        let code=match args.get(index+1){
            Some(path)=>match reference_serialize::run_acceptance(std::path::Path::new(path)){
                Ok(())=>0,Err(error)=>{eprintln!("{error}");1},
            },None=>{eprintln!("--reference-serializer-self-test needs a report path");2},
        };
        std::process::exit(code);
    }
    let sample = args.iter().any(|a| a == "--sample");
    let capture = args
        .iter()
        .position(|a| a == "--capture")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let fixture = args
        .iter()
        .position(|a| a == "--fixture")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let history_page = args.iter().any(|a| a == "--history" || a == "--history-emphasis");
    let history_compact = args.iter().any(|a| a == "--history-compact");
    let history_fit = args.iter().any(|a| a == "--history-fit");
    let self_test = args
        .iter()
        .position(|a| a == "--ui-smoke-output")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let self_test_lanes = args
        .iter()
        .position(|a| a == "--ui-smoke-lanes")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let keep_open = args.iter().any(|arg| arg == "--keep-open");
    let compact_preview = args.iter().any(|arg| arg == "--compact-preview");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(if compact_preview { [1100.0, 720.0] } else { [1440.0, 920.0] })
            .with_min_inner_size([1100.0, 720.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    let title = format!(
        "Primitive Painter {} · 原生矢量绘画{}",
        env!("CARGO_PKG_VERSION"),
        if args.iter().any(|arg|arg=="--compatibility") {" · 普通兼容"}else{""}
    );
    let result=eframe::run_native(
        &title,
        options,
        Box::new(move |cc| {
            Box::new(PainterApp::new(
                cc,
                sample,
                capture,
                fixture,
                history_page,
                history_compact,
                history_fit,
                self_test,
                self_test_lanes,
                keep_open,
            ))
        }),
    );
    match &result {
        Ok(())=>diagnostics::event("frontend.closed",serde_json::json!({})),
        Err(error)=>diagnostics::event("frontend.failed",serde_json::json!({"error_chain":[error.to_string()]})),
    }
    diagnostics::flush_summary();
    result
}
