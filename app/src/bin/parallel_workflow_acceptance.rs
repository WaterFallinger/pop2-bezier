#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
#[path="../mode_transition.rs"] mod mode_transition;
// Real CUDA, real-checkpoint acceptance for parallel creative history flows.
// Short generation only: 3x2 primitives, then 3x1 from a selected old branch.

#[path = "../acceleration.rs"]
mod acceleration;
#[path = "../document.rs"]
mod document;
#[path = "../history_context.rs"]
mod history_context;
#[path = "../inference.rs"]
mod inference;
#[path = "../model/mod.rs"]
mod model;
#[path = "../parallel_history.rs"]
mod parallel_history;
#[path = "../region_state.rs"]
mod region_state;

use anyhow::{bail, ensure, Context, Result};
use document::{Document, NodeId, Primitive, Settings};
use history_context::RawHistoryStore;
use inference::{
    EngineConfig, GenerationRequest, InferencePrimitive, InferenceSettings, NativeEngine,
};
use parallel_history::{LaneStatus, ParallelHistory};
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    collections::HashSet,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

const PROMPT: &str =
    "金发蓝眼的白皙圆脸女性以明亮动画风呈现，张嘴露舌，粉色背景可见，取景包含头部和肩颈。";
const SEED: u64 = 20261011;
const LANES: usize = 3;

struct Args {
    checkpoint: PathBuf,
    tokenizer: PathBuf,
    output: PathBuf,
    check_stop: bool,
    gpu_sampler: bool,
}
fn args() -> Result<Option<Args>> {
    let mut result = Args {
        checkpoint: "models/model.pt".into(),
        tokenizer: "assets/caption_spm_1500.model".into(),
        output: "artifacts/parallel_workflow_acceptance.json".into(),
        check_stop: false,
        gpu_sampler: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--checkpoint" => {
                result.checkpoint = args.next().context("--checkpoint 缺少路径")?.into()
            }
            "--tokenizer" => result.tokenizer = args.next().context("--tokenizer 缺少路径")?.into(),
            "--output" => result.output = args.next().context("--output 缺少路径")?.into(),
            "--check-stop" => result.check_stop = true,
            "--gpu-sampling" => result.gpu_sampler = true,
            "--help" | "-h" => {
                println!("parallel-workflow-acceptance --checkpoint MODEL.pt --tokenizer TOKENIZER.model --output REPORT.json [--check-stop]\nOne CUDA model load, 3x2 generation, branch selection and 3x1 continuation; optional coordinated stop after one complete round.");
                return Ok(None);
            }
            _ => bail!("未知参数：{arg}"),
        }
    }
    Ok(Some(result))
}

fn tokens(canvas: &[Primitive]) -> Result<Vec<[u32; 9]>> {
    canvas
        .iter()
        .map(|p| {
            p.to_tokens()
                .map(|r| r.map(u32::from))
                .map_err(anyhow::Error::msg)
        })
        .collect()
}

fn run_round(
    engine: &mut NativeEngine,
    doc: Document,
    store: RawHistoryStore,
    origin: NodeId,
    settings: Settings,
    diagnostic_limit: usize,
    stop_after_round: bool,
    report: &mut Value,
) -> Result<(Document, RawHistoryStore, ParallelHistory)> {
    let started = Instant::now();
    let lanes = settings.parallel_sequences;
    let manager = ParallelHistory::new(&doc, origin, PROMPT.into(), settings.clone())
        .map_err(anyhow::Error::msg)?;
    for lane in 0..lanes {
        ensure!(
            manager.seeds[lane] == inference::parallel_seed(settings.seed, lane),
            "核心与推理种子派生不一致"
        );
    }
    let request = GenerationRequest { redraw_regions:Vec::new(),
        prompt: PROMPT.into(),
        settings: InferenceSettings { half_region_switch: false, pause_at_stage_end: false,
            cfg_scale: settings.cfg as f64,
            renyi_alpha: settings.renyi_alpha as f64,
            seed: settings.seed,
            max_primitives: diagnostic_limit,
            context_overclock: settings.context_overclock,
        },
        primitives: tokens(&doc.materialize(origin).map_err(anyhow::Error::msg)?)?
            .into_iter()
            .map(|tokens| InferencePrimitive { tokens })
            .collect(),
        manual_review_regions: Vec::new(), extra_fine_context:Vec::new(),stage_floor:None,pending_stage_context:Vec::new(), history_records: store
            .context_for(&doc, origin)
            .map_err(anyhow::Error::msg)?,
    };
    let state = RefCell::new((doc, store, manager));
    let stop = AtomicBool::new(false);
    let error: RefCell<Option<String>> = RefCell::new(None);
    let callbacks = RefCell::new(Vec::new());
    let regions = RefCell::new(Vec::new());
    let summaries = RefCell::new(Vec::new());
    let messages = RefCell::new(Vec::new());
    let first_round = RefCell::new(HashSet::new());
    let result=engine.generate_parallel(&request,lanes,&stop,|lane,event| {
        if error.borrow().is_some(){ return; }
        let submitted=(||->Result<()> {
            let mut shared=state.borrow_mut();
            let (doc,store,manager)=&mut *shared;
            let before_current=doc.current;
            let before_canvas=doc.primitives.clone();
            let before_prompt=doc.prompt.clone();
            let before_settings=doc.settings.clone();
            let parent=manager.heads.get(lane).copied().context("回调lane越界")?;
            let node=manager.submit(doc,store,lane,event.primitive.tokens,&event.history_records).map_err(anyhow::Error::msg)?;
            ensure!(doc.nodes[node].parent==Some(parent),"完整输出没有挂到自己的lane head");
            if lane==0 { ensure!(doc.current==node,"主画布没有跟随lane0"); }
            else {
                ensure!(doc.current==before_current && doc.primitives==before_canvas,"非lane0输出改变了主画布");
                ensure!(doc.prompt==before_prompt && doc.settings==before_settings,"非lane0输出污染了主提示词/参数");
            }
            ensure!(doc.nodes[node].settings.seed==manager.seeds[lane],"节点没有记录自身lane的种子");
            ensure!(store.context_for(doc,node).map_err(anyhow::Error::msg)?==Some(event.history_records.clone()),"lane原始控制历史没有正确恢复");
            callbacks.borrow_mut().push(json!({"lane":lane,"node":node,"parent":parent,
                "seed":manager.seeds[lane],"tokens":event.primitive.tokens,
                "raw_records":event.history_records.len(),"current_after_callback":doc.current,
                "non_primary_view_unchanged":lane==0 || doc.current==before_current,
                "seconds":started.elapsed().as_secs_f64()}));
            eprintln!("[parallel workflow] lane {} -> node {node}",lane+1);
            Ok(())
        })();
        if let Err(failure)=submitted { *error.borrow_mut()=Some(format!("{failure:#}"));stop.store(true,Ordering::Relaxed);return; }
        if stop_after_round {
            first_round.borrow_mut().insert(lane);
            if first_round.borrow().len()==lanes { stop.store(true,Ordering::Relaxed); }
        }
    },|message| { eprintln!("[parallel workflow] {message}");messages.borrow_mut().push(message); },|lane,region| {
        if error.borrow().is_some(){return;}
        let mut shared=state.borrow_mut();
        if let Err(failure)=shared.2.set_region(lane,region.clone()) {
            *error.borrow_mut()=Some(failure);stop.store(true,Ordering::Relaxed);
        } else { regions.borrow_mut().push(json!({"lane":lane,"region":region})); }
    },|lane,summary| {
        if error.borrow().is_some(){return;}
        let finished=(||->Result<()> {
            let mut shared=state.borrow_mut();
            let (doc,store,manager)=&mut *shared;
            ensure!(manager.generated.get(lane)==Some(&summary.generated),"lane结束计数与已提交节点数不一致");
            let current=doc.current;let canvas=doc.primitives.clone();let prompt=doc.prompt.clone();let settings=doc.settings.clone();
            manager.finish_lane(doc,store,lane,&summary.history_records).map_err(anyhow::Error::msg)?;
            ensure!(doc.current==current && doc.primitives==canvas && doc.prompt==prompt && doc.settings==settings,"lane结束改变了主画布/参数");
            summaries.borrow_mut().push(json!({"lane":lane,"generated":summary.generated,"stopped":summary.stopped,
                "completed":summary.completed,"context_limit":summary.context_limit,
                "elapsed_seconds":summary.elapsed_seconds,"history_records":summary.history_records.len(),
                "head":manager.heads[lane]}));
            Ok(())
        })();
        if let Err(failure)=finished { *error.borrow_mut()=Some(format!("{failure:#}"));stop.store(true,Ordering::Relaxed); }
    });
    let (doc, store, manager) = state.into_inner();
    let batch = match &result {
        Ok(summary) => {
            json!({"lanes":summary.lanes,"generated":summary.generated,"stopped":summary.stopped,
            "elapsed_seconds":summary.elapsed_seconds,"batched_decode_calls":summary.batched_decode_calls,
            "maximum_tensor_batch":summary.maximum_tensor_batch,"prefill_strategy":summary.prefill_strategy})
        }
        Err(failure) => json!({"error":format!("{failure:#}")}),
    };
    *report = json!({"status":"running","origin":origin,"settings":settings,"stop_after_first_round":stop_after_round,
        "callbacks":callbacks.into_inner(),"regions":regions.into_inner(),"lane_summaries":summaries.into_inner(),
        "messages":messages.into_inner(),"heads":manager.heads,"seeds":manager.seeds,"generated":manager.generated,
        "selected_node":doc.current,"node_count":doc.nodes.len(),"engine":batch,"seconds":started.elapsed().as_secs_f64()});
    if let Some(failure) = error.into_inner() {
        report["status"] = json!("failed");
        report["callback_error"] = json!(failure);
        bail!("并行历史回调失败：{failure}");
    }
    let summary = result.context("真实CUDA并行生成失败")?;
    ensure!(
        summary.maximum_tensor_batch >= lanes,
        "没有观测到真实多行模型张量批量"
    );
    ensure!(summary.batched_decode_calls > 0, "没有执行原生批量decode");
    ensure!(
        manager.statuses.iter().all(|s| *s == LaneStatus::Finished),
        "部分lane缺少独立结束事件"
    );
    ensure!(
        manager.generated.iter().all(|&n| n
            == if stop_after_round {
                1
            } else {
                diagnostic_limit
            }),
        "各lane完整图元数不符合本轮设置"
    );
    ensure!(doc.current == manager.heads[0], "本轮结束时主画布不在lane0");
    for lane in 0..lanes {
        let raw = store
            .context_for(&doc, manager.heads[lane])
            .map_err(anyhow::Error::msg)?
            .context("输出lane缺少历史")?;
        store
            .validate_context_for_node(&doc, manager.heads[lane], &raw)
            .map_err(anyhow::Error::msg)?;
    }
    if stop_after_round {
        ensure!(summary.stopped, "协调停止请求未被模型报告");
    }
    report["status"] = json!("passed");
    Ok((doc, store, manager))
}

fn run(args: &Args, report: &mut Value) -> Result<()> {
    ensure!(
        args.checkpoint.is_file(),
        "真实权重文件不存在：{}",
        args.checkpoint.display()
    );
    ensure!(
        args.tokenizer.is_file(),
        "原生分词器不存在：{}",
        args.tokenizer.display()
    );
    report["checkpoint_bytes"] = json!(std::fs::metadata(&args.checkpoint)?.len());
    let load_started = Instant::now();
    let mut engine = NativeEngine::load(
        &EngineConfig {
            model_path: args.checkpoint.clone(),
            tokenizer_path: args.tokenizer.clone(),
            use_cuda: true,
        },
        |s| eprintln!("[load] {s}"),
    )?;
    engine.set_gpu_sampling_override(if args.gpu_sampler { Some(true) } else { None });
    report["model"] = json!(engine.model_description);
    report["load_seconds"] = json!(load_started.elapsed().as_secs_f64());
    let mut doc = Document::new();
    doc.prompt = PROMPT.into();
    let settings = Settings {
        parallel_sequences: LANES,
        cfg: 1.,
        renyi_alpha: 4.,
        seed: SEED,
        ..Default::default()
    };
    doc.settings = settings.clone();
    report["rounds"].as_array_mut().unwrap().push(Value::Null);
    let (mut doc, store, first) = run_round(
        &mut engine,
        doc,
        RawHistoryStore::new(),
        0,
        settings,
        2,
        false,
        report["rounds"].as_array_mut().unwrap().last_mut().unwrap(),
    )?;
    ensure!(doc.nodes.len() == 7, "首轮3x2输出没有恰好创建6个模型节点");
    ensure!(
        doc.nodes[0].children.len() == 3,
        "首轮不是从同一origin创建三个直接兄弟分支"
    );
    ensure!(
        first.seeds.iter().copied().collect::<HashSet<_>>().len() == LANES
            && first.seeds[0] == SEED,
        "首轮lane种子不独立或lane0未保留base seed"
    );
    for &head in &first.heads {
        let middle = doc.nodes[head].parent.context("分支缺少首节点")?;
        ensure!(
            doc.nodes[middle].parent == Some(0),
            "首轮分支长度不等于两个完整图元"
        );
    }
    let first_heads = first.heads.clone();
    let original_canvases = first_heads
        .iter()
        .map(|&h| doc.materialize(h))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(anyhow::Error::msg)?;
    let original_histories = first_heads
        .iter()
        .map(|&h| store.context_for(&doc, h))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(anyhow::Error::msg)?;
    let selected_origin = first_heads[2];
    doc.jump(selected_origin).map_err(anyhow::Error::msg)?;
    let second_settings = Settings {
        parallel_sequences: LANES,
        cfg: 1.,
        renyi_alpha: 4.,
        seed: SEED + 1000,
        ..Default::default()
    };
    doc.settings = second_settings.clone();
    report["rounds"].as_array_mut().unwrap().push(Value::Null);
    let (mut doc, store, second) = run_round(
        &mut engine,
        doc,
        store,
        selected_origin,
        second_settings,
        1,
        false,
        report["rounds"].as_array_mut().unwrap().last_mut().unwrap(),
    )?;
    ensure!(
        doc.nodes.len() == 10,
        "第二轮3x1没有恰好追加三个完整图元节点"
    );
    ensure!(
        doc.nodes[selected_origin].children.len() == 3,
        "选中lane2旧head后没有从其创建三个兄弟分支"
    );
    for &head in &second.heads {
        ensure!(
            doc.nodes[head].parent == Some(selected_origin),
            "第二轮节点没有挂到选中的lane2 head"
        );
    }
    for lane in 0..LANES {
        ensure!(
            doc.materialize(first_heads[lane])
                .map_err(anyhow::Error::msg)?
                == original_canvases[lane],
            "第二次生成修改了首轮分支画布"
        );
        ensure!(
            store
                .context_for(&doc, first_heads[lane])
                .map_err(anyhow::Error::msg)?
                == original_histories[lane],
            "第二次生成污染了首轮区域历史"
        );
    }
    let mut final_history = second;
    let mut final_store = store;
    if args.check_stop {
        let origin = final_history.heads[0];
        doc.jump(origin).map_err(anyhow::Error::msg)?;
        let settings = Settings {
            parallel_sequences: LANES,
            cfg: 1.,
            renyi_alpha: 4.,
            seed: SEED + 2000,
            ..Default::default()
        };
        doc.settings = settings.clone();
        report["rounds"].as_array_mut().unwrap().push(Value::Null);
        let (stopped_doc, stopped_store, stopped_history) = run_round(
            &mut engine,
            doc,
            final_store,
            origin,
            settings,
            4,
            true,
            report["rounds"].as_array_mut().unwrap().last_mut().unwrap(),
        )?;
        doc = stopped_doc;
        final_store = stopped_store;
        final_history = stopped_history;
    }
    report["checks"] = json!({"real_cuda_model_loaded_once":true,"true_batched_decode_observed":true,
        "three_sibling_branches_with_two_nodes":true,"only_lane_zero_follows_canvas":true,
        "prompt_settings_seed_isolation":true,"all_lane_histories_match_own_canvas":true,
        "selected_lane_two_can_be_origin_of_next_parallel_run":true,"three_new_children_on_selected_origin":true,
        "prior_branches_and_region_histories_preserved":true,"optional_stop_checked":args.check_stop});
    report["first_heads"] = json!(first_heads);
    report["second_origin"] = json!(selected_origin);
    report["final_heads"] = json!(final_history.heads);
    report["current_head"] = json!(doc.current);
    report["history_nodes"] = serde_json::to_value(&doc.nodes)?;
    report["node_count"] = json!(doc.nodes.len());
    report["primitives"] = json!(tokens(&doc.primitives)?);
    report["history_records"] = json!(final_store
        .context_for(&doc, doc.current)
        .map_err(anyhow::Error::msg)?);
    Ok(())
}

fn main() -> Result<()> {
    let Some(args) = args()? else {
        return Ok(());
    };
    let started = Instant::now();
    let mut report = json!({"status":"running","scope":"real_cuda_parallel_history_workflow","native_rust":true,
        "checkpoint":args.checkpoint,"tokenizer":args.tokenizer,"prompt":PROMPT,"cfg":1.,"renyi_alpha":4.,
        "base_seed":SEED,"parallel_sequences":LANES,"rounds":[],"note":"Short real-weight functional acceptance, not a quality/throughput benchmark."});
    let result = run(&args, &mut report);
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    match &result {
        Ok(()) => report["status"] = json!("complete"),
        Err(e) => {
            report["status"] = json!("failed");
            report["error"] = json!(format!("{e:#}"));
        }
    }
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", args.output.display());
    result
}

#[path = "../diagnostics.rs"] mod diagnostics;
