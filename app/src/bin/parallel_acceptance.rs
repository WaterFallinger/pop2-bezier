#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
// Real-weight native tensor-batch reference and whole-batch stop acceptance.
#[path = "../acceleration.rs"]
mod acceleration;
#[path = "../document.rs"]
mod document;
#[path = "../inference.rs"]
mod inference;
#[path = "../model/mod.rs"]
mod model;
#[path = "../region_state.rs"]
mod region_state;

use anyhow::{ensure, Context, Result};
use inference::{parallel_seed, EngineConfig, GenerationRequest, InferenceSettings, NativeEngine};
use serde_json::json;
use std::{
    collections::HashSet,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

const PROMPT: &str =
    "金发蓝眼的白皙圆脸女性以明亮动画风呈现，张嘴露舌，粉色背景可见，取景包含头部和肩颈。";

fn main() -> Result<()> {
    let mut checkpoint = PathBuf::from("models/model.pt");
    let mut tokenizer = PathBuf::from("assets/caption_spm_1500.model");
    let mut output = PathBuf::from("artifacts/native_parallel_acceptance.json");
    let mut gpu_sampler = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--checkpoint" => checkpoint = args.next().context("--checkpoint 缺少路径")?.into(),
            "--tokenizer" => tokenizer = args.next().context("--tokenizer 缺少路径")?.into(),
            "--output" => output = args.next().context("--output 缺少路径")?.into(),
            "--gpu-sampling" => gpu_sampler = true,
            "--help" | "-h" => {
                println!("parallel-acceptance --checkpoint MODEL.pt --tokenizer TOKENIZER.model --output REPORT.json\nOne CUDA model load; real ragged tensor batch parity, CFG2B, and three-lane complete-round stopping.");
                return Ok(());
            }
            _ => anyhow::bail!("未知参数：{arg}"),
        }
    }
    let started = Instant::now();
    for lane in 0..16 {
        ensure!(
            parallel_seed(20261011, lane) == document::parallel_seed(20261011, lane),
            "UI与引擎lane seed不一致"
        );
    }
    let mut engine = NativeEngine::load(
        &EngineConfig {
            model_path: checkpoint.clone(),
            tokenizer_path: tokenizer,
            use_cuda: true,
        },
        |s| eprintln!("{s}"),
    )?;
    engine.set_gpu_sampling_override(if gpu_sampler { Some(true) } else { None });
    let parity = engine.parallel_reference_probe(PROMPT)?;
    let request = GenerationRequest { redraw_regions:Vec::new(),
        prompt: PROMPT.into(),
        settings: InferenceSettings { half_region_switch: false, pause_at_stage_end: false,
            cfg_scale: 2.,
            renyi_alpha: 4.,
            seed: 20261011,
            max_primitives: 4,
            context_overclock: false,
        },
        primitives: Vec::new(),
        manual_review_regions: Vec::new(), extra_fine_context:Vec::new(),stage_floor:None,pending_stage_context:Vec::new(), history_records: None,
    };
    let stop = AtomicBool::new(false);
    let mut outputs = vec![Vec::new(); 3];
    let mut regions = Vec::new();
    let mut finished = Vec::new();
    let result = engine.generate_parallel(
        &request,
        3,
        &stop,
        |lane, p| {
            outputs[lane].push(p.primitive.tokens);
            if lane == 0 {
                stop.store(true, Ordering::Relaxed);
            }
        },
        |s| eprintln!("{s}"),
        |lane, region| regions.push(json!({"lane":lane,"region":region})),
        |lane, summary| finished.push((lane, summary)),
    )?;
    ensure!(
        result.maximum_tensor_batch == 6,
        "CFG2三条逻辑序列未真实合成6行tensor batch"
    );
    ensure!(
        result.batched_decode_calls == 8,
        "首背景应跳过唯一shape候选，保留7个partial和1个commit；停止仍须保留整批"
    );
    ensure!(
        result.generated == 3 && result.lane_summaries.len() == 3,
        "全批停止丢失已经完成的lane"
    );
    ensure!(
        outputs.iter().all(|v| v.len() == 1),
        "停止轮边界不是每lane一个完整图元"
    );
    ensure!(
        result.stopped
            && result
                .lane_summaries
                .iter()
                .all(|s| s.stopped && s.generated == 1),
        "停止状态不一致"
    );
    ensure!(
        finished
            .iter()
            .map(|(lane, _)| *lane)
            .collect::<HashSet<_>>()
            .len()
            == 3,
        "lane结束事件重复或遗漏"
    );
    for row in outputs.iter().flatten() {
        model::validate_visible(row)?;
        ensure!(row[5] == model::OFFSETS[5], "首图元不是背景");
    }
    let lane_summaries:Vec<_>=result.lane_summaries.iter().enumerate().map(|(lane,s)|json!({"lane":lane,"seed":parallel_seed(20261011,lane),
        "generated":s.generated,"stopped":s.stopped,"completed":s.completed,"history_records":s.history_records})).collect();
    let report = json!({"status":"complete","native_rust":true,"model_loads":1,"model":engine.model_description,"checkpoint":checkpoint,
        "tensor_reference_parity":parity,"stop_acceptance":{"logical_lanes":3,"cfg":2,"maximum_tensor_batch":result.maximum_tensor_batch,
            "batched_decode_calls":result.batched_decode_calls,"all_completed_rows_retained_when_stop_arrives_after_lane0_callback":true,
            "prefill_strategy":result.prefill_strategy,"generated_primitives_by_lane":outputs,"lane_summaries":lane_summaries,"region_events":regions},
        "scope":"real native batched decode correctness and stop control, not throughput or final-image quality",
        "seconds":started.elapsed().as_secs_f64()});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"status":"complete","output":output,"model_loads":1,"maximum_tensor_batch":6,"batched_decode_calls":result.batched_decode_calls,"generated":3})
        )?
    );
    Ok(())
}

#[path = "../diagnostics.rs"] mod diagnostics;
