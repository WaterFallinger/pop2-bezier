#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
#[path="../mode_transition.rs"] mod mode_transition;
// Real-weight native CUDA acceptance of generation, stop, edits, and forks.
// No GUI, Python process, network service, or substitute generator is involved.

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
#[path = "../region_state.rs"]
mod region_state;

use anyhow::{bail, ensure, Context, Result};
use document::{Document, OperationSource, Patch, Primitive, ShapeKind};
use history_context::RawHistoryStore;
use inference::{
    EngineConfig, GenerationRequest, InferencePrimitive, InferenceSettings, NativeEngine,
};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

const PROMPT: &str =
    "金发蓝眼的白皙圆脸女性以明亮动画风呈现，张嘴露舌，粉色背景可见，取景包含头部和肩颈。";
const BASE_SEED: u64 = 20261011;

struct Arguments {
    checkpoint: PathBuf,
    tokenizer: PathBuf,
    output: PathBuf,
}

fn parse_arguments() -> Result<Option<Arguments>> {
    let mut args = std::env::args().skip(1);
    let mut result = Arguments {
        checkpoint: "models/model.pt".into(),
        tokenizer: "assets/caption_spm_1500.model".into(),
        output: "artifacts/native_e2e_acceptance.json".into(),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--checkpoint" => {
                result.checkpoint = args.next().context("--checkpoint 缺少路径")?.into()
            }
            "--tokenizer" => result.tokenizer = args.next().context("--tokenizer 缺少路径")?.into(),
            "--output" => result.output = args.next().context("--output 缺少路径")?.into(),
            "--help" | "-h" => {
                println!("acceptance --checkpoint MODEL.pt --tokenizer TOKENIZER.model --output REPORT.json\nReal CUDA inference, one model load; stop, resume, manual edit and branch acceptance.");
                return Ok(None);
            }
            _ => bail!("未知参数：{arg}"),
        }
    }
    Ok(Some(result))
}

fn visible_tokens(doc: &Document) -> Result<Vec<[u32; 9]>> {
    doc.primitives
        .iter()
        .map(|p| {
            p.to_tokens()
                .map(|r| r.map(u32::from))
                .map_err(anyhow::Error::msg)
        })
        .collect()
}

fn decode_primitive(tokens: [u32; 9]) -> Result<Primitive> {
    let mut native = [0u16; 9];
    for i in 0..9 {
        native[i] = tokens[i]
            .try_into()
            .context("完整图元 token 超出 u16 原生词表")?;
    }
    Primitive::from_tokens(0, native).map_err(anyhow::Error::msg)
}

/// The callback performs the same complete-operation commit used by the app.
/// It cannot return Result to the engine, so a callback error requests stop and
/// is propagated before any successful stage is reported.
fn run_stage(
    engine: &mut NativeEngine,
    doc: &mut Document,
    store: &mut RawHistoryStore,
    maximum: usize,
    stop_after_first: bool,
    label: &str,
) -> Result<Value> {
    let started = Instant::now();
    let parent = doc.current;
    let old_nodes = doc.nodes.len();
    let old_primitives = doc.primitives.len();
    let history = store.context_for(doc, parent).map_err(anyhow::Error::msg)?;
    let initial_history_records = history.as_ref().map(Vec::len);
    let request = GenerationRequest { redraw_regions:Vec::new(),
        prompt: doc.prompt.clone(),
        settings: InferenceSettings { half_region_switch: false, pause_at_stage_end: false,
            cfg_scale: doc.settings.cfg as f64,
            renyi_alpha: doc.settings.renyi_alpha as f64,
            seed: doc.settings.seed,
            max_primitives: maximum,
            context_overclock: doc.settings.context_overclock,
        },
        primitives: visible_tokens(doc)?
            .into_iter()
            .map(|tokens| InferencePrimitive { tokens })
            .collect(),
        manual_review_regions: Vec::new(), extra_fine_context:Vec::new(),stage_floor:None,pending_stage_context:Vec::new(), history_records: history,
    };
    let stop = AtomicBool::new(false);
    let mut callback_error = None;
    let mut callbacks = Vec::new();
    let mut status_messages = Vec::new();
    let summary = engine
        .generate(
            &request,
            &stop,
            |event| {
                if callback_error.is_some() {
                    return;
                }
                let result = (|| -> Result<()> {
                    let primitive = decode_primitive(event.primitive.tokens)?;
                    let before = doc.current;
                    let node = doc
                        .commit_generated(primitive)
                        .map_err(anyhow::Error::msg)?;
                    store
                        .attach(doc, node, &event.history_records)
                        .map_err(anyhow::Error::msg)?;
                    callbacks.push(json!({
                        "node":node, "parent":before, "tokens":event.primitive.tokens,
                        "raw_history_records":event.history_records.len(),
                        "canvas_primitives":doc.primitives.len(),
                        "seconds_since_stage_start":started.elapsed().as_secs_f64(),
                        "seed":doc.nodes[node].settings.seed,
                    }));
                    eprintln!("[{label}] 完整图元 {} -> 节点 {node}", callbacks.len());
                    Ok(())
                })();
                if let Err(error) = result {
                    callback_error = Some(error);
                    stop.store(true, Ordering::Relaxed);
                } else if stop_after_first {
                    stop.store(true, Ordering::Relaxed);
                }
            },
            |message| {
                eprintln!("[{label}] {message}");
                status_messages.push(message);
            },
        )
        .with_context(|| format!("{label}：原生模型生成失败"))?;
    if let Some(error) = callback_error {
        return Err(error.context(format!("{label}：完成图元提交失败")));
    }
    store
        .finish(doc, doc.current, &summary.history_records)
        .map_err(anyhow::Error::msg)?;
    let restored = store
        .context_for(doc, doc.current)
        .map_err(anyhow::Error::msg)?
        .context("生成完成的节点没有可恢复上下文")?;
    ensure!(
        restored == summary.history_records,
        "{label}：结束上下文与增量恢复结果不一致"
    );
    ensure!(
        callbacks.len() == summary.generated,
        "{label}：回调数量与模型计数不一致"
    );
    ensure!(
        doc.nodes.len() - old_nodes == callbacks.len(),
        "{label}：每完整图元没有恰好创建一个状态节点"
    );
    ensure!(
        doc.primitives.len() - old_primitives == callbacks.len(),
        "{label}：画布新增图元数量错误"
    );
    let mut expected_parent = parent;
    for node in &doc.nodes[old_nodes..] {
        ensure!(
            node.source == OperationSource::Model && node.parent == Some(expected_parent),
            "{label}：生成节点没有形成独立的顺序链"
        );
        ensure!(
            node.patches.len() == 1 && matches!(node.patches[0], Patch::Add { .. }),
            "{label}：模型节点不是单一图元增量"
        );
        expected_parent = node.id;
    }
    Ok(json!({
        "name":label, "status":"passed", "requested_new_primitives":maximum,
        "stop_after_first_complete_primitive":stop_after_first,
        "start_node":parent, "end_node":doc.current,
        "existing_primitives":old_primitives, "generated":summary.generated,
        "stopped":summary.stopped, "completed_all_regions":summary.completed,
        "context_limit":summary.context_limit, "input_had_saved_context":initial_history_records.is_some(),
        "initial_history_records":initial_history_records, "final_history_records":summary.history_records.len(),
        "one_node_per_complete_primitive":true, "history_delta_roundtrip_matches":true,
        "generation_seconds":summary.elapsed_seconds, "stage_seconds":started.elapsed().as_secs_f64(),
        "callbacks":callbacks, "status_messages":status_messages,
        "canvas_tokens_after":visible_tokens(doc)?,
    }))
}

fn run_acceptance(args: &Arguments, report: &mut Value) -> Result<()> {
    ensure!(
        args.checkpoint.is_file(),
        "权重文件不存在：{}",
        args.checkpoint.display()
    );
    ensure!(
        args.tokenizer.is_file(),
        "Tokenizer 不存在：{}",
        args.tokenizer.display()
    );
    report["checkpoint_bytes"] = json!(std::fs::metadata(&args.checkpoint)?.len());
    let metadata_path = args.checkpoint.parent().map(|p| p.join("READY.json"));
    report["checkpoint_metadata"] = metadata_path
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .unwrap_or(Value::Null);
    let load_started = Instant::now();
    let mut engine = NativeEngine::load(
        &EngineConfig {
            model_path: args.checkpoint.clone(),
            tokenizer_path: args.tokenizer.clone(),
            use_cuda: true,
        },
        |message| eprintln!("[load] {message}"),
    )?;
    report["model"] = json!(engine.model_description);
    report["model_load_seconds"] = json!(load_started.elapsed().as_secs_f64());
    report["text_ids"] = json!(engine.tokenize(PROMPT)?);

    let mut doc = Document::new();
    doc.prompt = PROMPT.into();
    doc.settings.cfg = 1.0;
    doc.settings.renyi_alpha = 4.0;
    doc.settings.seed = BASE_SEED;
    let mut store = RawHistoryStore::new();

    let stage1 = run_stage(
        &mut engine,
        &mut doc,
        &mut store,
        3,
        true,
        "stop_after_background",
    )?;
    report["stages"]
        .as_array_mut()
        .unwrap()
        .push(stage1.clone());
    ensure!(
        stage1["generated"] == 1 && stage1["stopped"] == true,
        "首阶段未在第一个完整图元后停止"
    );
    ensure!(
        doc.primitives.len() == 1 && doc.primitives[0].shape == ShapeKind::Background,
        "首阶段停止后的画布没有恰好一个背景图元"
    );
    let background_node = doc.current;
    let background_settings = doc.settings.clone();

    let stage2 = run_stage(
        &mut engine,
        &mut doc,
        &mut store,
        2,
        false,
        "resume_two_primitives",
    )?;
    report["stages"]
        .as_array_mut()
        .unwrap()
        .push(stage2.clone());
    ensure!(
        stage2["generated"] == 2 && stage2["stopped"] == false,
        "第二阶段未完成两个续画图元"
    );
    ensure!(
        doc.primitives.len() == 3
            && doc
                .primitives
                .iter()
                .filter(|p| p.shape == ShapeKind::Background)
                .count()
                == 1,
        "第二阶段续画修改了背景数量或图元数量"
    );
    let original_head = doc.current;
    let original_canvas = doc.primitives.clone();
    let original_history = store
        .context_for(&doc, original_head)
        .map_err(anyhow::Error::msg)?
        .unwrap();

    // The same document branch operation used by the native UI.
    let edit_started = Instant::now();
    doc.jump(background_node).map_err(anyhow::Error::msg)?;
    ensure!(
        doc.settings == background_settings && doc.prompt == PROMPT,
        "跳转背景节点没有恢复完整生成设置"
    );
    let mut edited = doc.primitives.clone();
    let edited_background = [20, 70, 120]; // Exact native even-valued RGB lattice.
    edited[0].color = edited_background;
    doc.settings.seed = BASE_SEED + 12345;
    let manual_node = doc
        .commit_canvas("人工改背景色并从旧状态分叉", OperationSource::User, edited)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        doc.nodes[manual_node].parent == Some(background_node),
        "人工编辑没有从选中的旧节点分叉"
    );
    ensure!(
        doc.nodes[background_node].children.len() == 2,
        "旧节点的两个分支没有同时保留"
    );
    ensure!(
        store
            .context_for(&doc, manual_node)
            .map_err(anyhow::Error::msg)?
            .is_none(),
        "人工改图后仍复用了旧模型上下文"
    );
    report["stages"].as_array_mut().unwrap().push(json!({
        "name":"jump_and_manual_background_edit", "status":"passed",
        "jumped_to":background_node, "created_user_node":manual_node,
        "previous_generation_settings_restored":true, "raw_model_context_invalidated":true,
        "new_background_rgb":edited_background, "new_seed":doc.settings.seed,
        "stage_seconds":edit_started.elapsed().as_secs_f64(),
    }));

    let stage3 = run_stage(
        &mut engine,
        &mut doc,
        &mut store,
        1,
        false,
        "generate_on_edited_branch",
    )?;
    report["stages"]
        .as_array_mut()
        .unwrap()
        .push(stage3.clone());
    ensure!(
        stage3["generated"] == 1 && stage3["input_had_saved_context"] == false,
        "第三阶段没有从编辑画布重新构建上下文并生成一个图元"
    );
    ensure!(
        doc.primitives.len() == 2
            && doc.primitives[0].shape == ShapeKind::Background
            && doc.primitives[0].color == edited_background,
        "第三阶段没有保留人工修改的背景"
    );
    ensure!(
        doc.nodes[doc.current].settings.seed == BASE_SEED + 12345,
        "新分支生成节点未记录新随机种子"
    );
    let new_head = doc.current;
    ensure!(
        doc.nodes[new_head].parent == Some(manual_node),
        "新生成节点的父节点不是人工编辑状态"
    );
    ensure!(
        doc.materialize(original_head).map_err(anyhow::Error::msg)? == original_canvas,
        "原生成分支的画布被新分支覆盖"
    );
    ensure!(
        store
            .context_for(&doc, original_head)
            .map_err(anyhow::Error::msg)?
            == Some(original_history),
        "原分支区域控制历史被污染"
    );
    ensure!(doc.nodes.len() == 6, "验收操作产生了多余或遗漏的历史状态");
    ensure!(
        doc.nodes
            .iter()
            .filter(|n| n.source == OperationSource::Model)
            .count()
            == 4,
        "四个完整模型图元没有恰好对应四个模型节点"
    );

    // Exercise both final states once more, including restoration of metadata.
    doc.jump(original_head).map_err(anyhow::Error::msg)?;
    ensure!(
        doc.primitives == original_canvas && doc.settings.seed == BASE_SEED,
        "跳回原分支不能恢复原画布与种子"
    );
    doc.jump(new_head).map_err(anyhow::Error::msg)?;
    ensure!(
        doc.settings.seed == BASE_SEED + 12345 && doc.primitives[0].color == edited_background,
        "跳回新分支不能恢复编辑后的条件"
    );

    report["checks"] = json!({
        "true_checkpoint_used":true, "cuda_enabled":true, "model_loaded_once":true,
        "stop_kept_only_complete_background":true, "resume_used_saved_region_history":true,
        "manual_edit_invalidated_model_history":true, "manual_background_retained":true,
        "original_branch_unchanged":true, "both_branches_retained":true,
        "one_model_node_per_complete_primitive":true, "prompt_settings_seed_restore":true,
        "raw_history_matches_visible_state":true,
    });
    report["original_branch_head"] = json!(original_head);
    report["current_head"] = json!(new_head);
    report["background_node"] = json!(background_node);
    report["manual_edit_node"] = json!(manual_node);
    report["history_nodes"] = serde_json::to_value(&doc.nodes)?;
    report["node_count"] = json!(doc.nodes.len());
    report["stored_canvas_patch_count"] = json!(doc.stored_patch_count());
    report["stored_raw_context_record_count"] = json!(store.stored_record_count());
    report["primitives"] = json!(visible_tokens(&doc)?);
    report["original_branch_primitives"] = json!(original_canvas
        .iter()
        .map(|p| p.to_tokens().map(|row| row.map(u32::from)))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(anyhow::Error::msg)?);
    report["history_records"] = json!(store
        .context_for(&doc, new_head)
        .map_err(anyhow::Error::msg)?);
    Ok(())
}

fn main() -> Result<()> {
    let Some(args) = parse_arguments()? else {
        return Ok(());
    };
    let started = Instant::now();
    let mut report = json!({
        "status":"running", "scope":"native_cuda_real_weight_editor_end_to_end",
        "native_rust":true, "checkpoint":args.checkpoint, "tokenizer":args.tokenizer,
        "prompt":PROMPT, "cfg":1.0, "renyi_alpha":4.0, "seed":BASE_SEED,
        "stages":[],
        "note":"A short real inference/editor acceptance, not a full-image quality or throughput benchmark.",
    });
    let result = run_acceptance(&args, &mut report);
    report["total_seconds"] = json!(started.elapsed().as_secs_f64());
    if let Err(error) = &result {
        report["status"] = json!("failed");
        report["error"] = json!(format!("{error:#}"));
    } else {
        report["status"] = json!("complete");
    }
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", args.output.display());
    result
}

#[path = "../diagnostics.rs"] mod diagnostics;
