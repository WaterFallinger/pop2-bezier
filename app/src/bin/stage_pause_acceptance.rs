#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
// One model load; real single/eight-lane inference at global/coarse boundaries.
// Synthetic protocol fixtures need at most two new visible primitives per case.
#[path = "../acceleration.rs"] mod acceleration;
#[path = "../inference.rs"] mod inference;
#[path = "../model/mod.rs"] mod model;
#[path = "../region_state.rs"] mod region_state;

use anyhow::{ensure, Context, Result};
use inference::{EngineConfig, GenerationRequest, GenerationSummary, InferencePrimitive,
    InferenceSettings, NativeEngine};
use region_state::{DrawingRegion, Stage};
use serde_json::{json, Value};
use std::{path::PathBuf, sync::atomic::{AtomicBool, Ordering}, time::Instant};

struct Args { checkpoint: PathBuf, tokenizer: PathBuf, output: PathBuf, lanes: usize, cuda: bool }

fn main() -> Result<()> {
    let mut args = Args {
        checkpoint: "models/model.pt".into(),
        tokenizer: "assets/caption_spm_1500.model".into(),
        output: "artifacts/stage_pause/real_inference_acceptance.json".into(),
        lanes: 8, cuda: true,
    };
    let mut cli = std::env::args().skip(1);
    while let Some(arg) = cli.next() {
        match arg.as_str() {
            "--checkpoint" => args.checkpoint = cli.next().context("missing checkpoint")?.into(),
            "--tokenizer" => args.tokenizer = cli.next().context("missing tokenizer")?.into(),
            "--output" => args.output = cli.next().context("missing output")?.into(),
            "--lanes" => args.lanes = cli.next().context("missing lanes")?.parse()?,
            "--cpu" => args.cuda = false,
            "--help" | "-h" => {
                println!("stage-pause-acceptance [--checkpoint MODEL.pt] [--tokenizer TOKENIZER.model] [--output REPORT.json] [--lanes 8] [--cpu]");
                return Ok(());
            }
            _ => anyhow::bail!("unknown argument: {arg}"),
        }
    }
    ensure!((2..=16).contains(&args.lanes), "parallel acceptance requires 2..16 lanes");
    let result = run(&args);
    let report = match &result {
        Ok(value) => value.clone(),
        Err(error) => json!({"status":"failed", "error":format!("{error:#}")}),
    };
    if let Some(parent) = args.output.parent() { std::fs::create_dir_all(parent)?; }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    result.map(|_| ())
}

fn ordinary(side: usize, index: usize, variant: usize) -> [u32; 9] {
    let mut row = model::region_marker(side, index);
    row[3] = model::OFFSETS[3] + 64;
    row[4] = model::OFFSETS[4] + 64;
    row[5] = model::OFFSETS[5] + 1 + (variant % 2) as u32;
    row[6] = model::OFFSETS[6] + 40;
    row[7] = model::OFFSETS[7] + 50;
    row[8] = model::OFFSETS[8] + 60;
    row
}

fn global_fixture() -> Vec<[u32; 9]> {
    let mut full = vec![model::BOS_RECORD];
    let mut background = ordinary(4, 0, 0);
    background[5] = model::OFFSETS[5];
    full.push(background);
    // Background + 84 ordinary primitives: one real prediction remains.
    full.extend((0..84).map(|index| ordinary(4, index % 16, index)));
    full
}

fn coarse_fixture() -> Vec<[u32; 9]> {
    let mut full = global_fixture();
    full.push(ordinary(4, 0, 84));
    for index in 0..15 {
        // Legitimate early TRUE_END cells must count toward the 16-cell stage.
        full.extend([model::region_marker(4, index), model::END_RECORD]);
    }
    full.push(model::region_marker(4, 15));
    full.extend((0..19).map(|index| ordinary(4, 15, index)));
    full
}

fn visible(row: &&[u32; 9]) -> bool {
    model::validate_visible(row).is_ok()
}

fn request(full: &[[u32; 9]], pause: bool, maximum: usize) -> GenerationRequest {
    GenerationRequest { redraw_regions:Vec::new(),
        prompt: "金发蓝眼".into(),
        settings: InferenceSettings { half_region_switch:false, cfg_scale: 1., renyi_alpha: 4., seed: 20261031,
            max_primitives: maximum, context_overclock: true, pause_at_stage_end: pause },
        primitives: full.iter().filter(visible)
            .map(|tokens| InferencePrimitive { tokens: *tokens }).collect(),
        manual_review_regions: Vec::new(), extra_fine_context:Vec::new(),stage_floor:None,pending_stage_context:Vec::new(), history_records: Some(full.to_vec()),
    }
}

fn infer(engine: &mut NativeEngine, request: &GenerationRequest, lanes: usize, label: &str)
    -> Result<(Vec<GenerationSummary>, Value)>
{
    let stop = AtomicBool::new(false);
    let mut callbacks = vec![0usize; lanes];
    let mut finish_callbacks = vec![0usize; lanes];
    let mut last_regions = vec![None; lanes];
    let initial = request.history_records.as_ref().unwrap();
    let mut invalid_primitive = false;
    let mut outputs = Vec::new();
    let mut tensor_batch = 1;
    if lanes == 1 {
        let summary = engine.generate_with_regions(request, &stop,
            |event| {
                callbacks[0] += 1;
                invalid_primitive |= model::validate_visible(&event.primitive.tokens).is_err()
                    || !event.history_records.starts_with(initial);
            },
            |status| eprintln!("{label}: {status}"),
            |region| last_regions[0] = Some(region),
        )?;
        finish_callbacks[0] = 1;
        outputs.push(summary);
    } else {
        let parallel = engine.generate_parallel(request, lanes, &stop,
            |lane, event| {
                callbacks[lane] += 1;
                invalid_primitive |= model::validate_visible(&event.primitive.tokens).is_err()
                    || !event.history_records.starts_with(initial);
            },
            |status| eprintln!("{label}: {status}"),
            |lane, region| last_regions[lane] = Some(region),
            |lane, _| finish_callbacks[lane] += 1,
        )?;
        let common_pause = parallel.lane_summaries.first().and_then(|summary| summary.paused_after_stage)
            .filter(|stage| parallel.lane_summaries.iter()
                .all(|summary| summary.paused_after_stage == Some(*stage)));
        ensure!(parallel.paused_after_stage == common_pause && !parallel.stopped,
            "{label}: parallel stage-pause summary is inconsistent with its lanes");
        tensor_batch = parallel.maximum_tensor_batch;
        outputs = parallel.lane_summaries;
    }
    ensure!(!invalid_primitive, "{label}: invalid primitive callback or changed history prefix");
    ensure!(!stop.load(Ordering::Relaxed), "stage pause must not set the shared stop flag");
    for (lane, summary) in outputs.iter().enumerate() {
        ensure!(!summary.stopped && !summary.context_limit, "{label} lane{lane}: wrong finish reason");
        ensure!(summary.generated == callbacks[lane], "{label}: partial or missing primitive");
        ensure!(finish_callbacks[lane] == 1, "{label}: lane completion must be exactly once");
        ensure!(summary.history_records.starts_with(initial), "{label}: original history changed");
        let actual = DrawingRegion::from_history(&summary.history_records).map_err(anyhow::Error::msg)?;
        ensure!(last_regions[lane].as_ref() == Some(&actual), "{label}: missing final region event");
    }
    let rows: Vec<_> = outputs.iter().enumerate().map(|(lane, summary)| json!({
        "lane":lane, "generated":summary.generated, "stopped":summary.stopped,
        "completed":summary.completed, "paused_after_stage":summary.paused_after_stage,
        "history_records":summary.history_records, "region":last_regions[lane],
    })).collect();
    Ok((outputs, json!({"case":label,"lanes":lanes,"maximum_tensor_batch":tensor_batch,"outputs":rows})))
}

fn check_boundary(summary: &GenerationSummary, stage: Stage) -> Result<()> {
    ensure!(summary.paused_after_stage == Some(stage) && !summary.completed,
        "expected pause after {stage:?}, got {:?}", summary.paused_after_stage);
    let region = DrawingRegion::from_history(&summary.history_records).map_err(anyhow::Error::msg)?;
    match stage {
        Stage::Global => {
            ensure!(summary.generated == 1 && summary.history_records.len() == 87,
                "global pause must retain exactly background + 85 ordinary primitives");
            ensure!(!summary.history_records.iter().any(model::is_marker), "coarse marker crossed global pause");
            ensure!(region.stage == Stage::Coarse && region.awaiting_region, "incorrect global boundary region");
        }
        Stage::Coarse => {
            ensure!(summary.generated <= 1 && region.completed_coarse == 16, "coarse stage did not finish");
            ensure!(summary.history_records.last() == Some(&model::END_RECORD), "last coarse TRUE_END missing");
            ensure!(region.stage == Stage::Fine && region.awaiting_region, "fine marker crossed coarse pause");
        }
        _ => anyhow::bail!("unexpected boundary stage"),
    }
    Ok(())
}

fn run(args: &Args) -> Result<Value> {
    let started = Instant::now();
    let mut engine = NativeEngine::load(&EngineConfig {
        model_path: args.checkpoint.clone(), tokenizer_path: args.tokenizer.clone(), use_cuda: args.cuda,
    }, |status| eprintln!("{status}"))?;
    let mut reports = Vec::new();
    for lanes in [1, args.lanes] {
        for (stage, full) in [(Stage::Global, global_fixture()), (Stage::Coarse, coarse_fixture())] {
            let label = format!("{stage:?}_pause_{lanes}lanes");
            // The last visible primitive may coincide with this request cap.
            let (paused, report) = infer(&mut engine, &request(&full, true, 1), lanes, &label)?;
            for summary in &paused { check_boundary(summary, stage)?; }
            if lanes > 1 && stage == Stage::Global {
                ensure!(report["maximum_tensor_batch"].as_u64().unwrap_or(0) >= lanes as u64,
                    "global case must exercise true tensor batching");
            }
            reports.push(report);

            let continued = request(&paused[0].history_records, true, 1);
            let (resumed, report) = infer(&mut engine, &continued, lanes,
                &format!("{stage:?}_resume_{lanes}lanes"))?;
            for summary in &resumed {
                ensure!(summary.paused_after_stage != Some(stage), "continuation paused at the same boundary");
                ensure!(summary.history_records.len() > paused[0].history_records.len(), "continuation made no progress");
                let expected_side = if stage == Stage::Global { 4 } else { 8 };
                ensure!(summary.history_records[paused[0].history_records.len()..]
                    .iter().any(|row| (0..expected_side * expected_side)
                        .any(|index| *row == model::region_marker(expected_side, index))),
                    "continuation never entered the next stage");
            }
            reports.push(report);

            let (off, report) = infer(&mut engine, &request(&full, false, 2), lanes,
                &format!("{stage:?}_pause_off_{lanes}lanes"))?;
            for summary in &off {
                ensure!(summary.paused_after_stage.is_none(), "disabled pause unexpectedly stopped");
                let next_side = if stage == Stage::Global { 4 } else { 8 };
                ensure!(summary.history_records.iter().any(|row| (0..next_side * next_side)
                    .any(|index| *row == model::region_marker(next_side, index))),
                    "disabled pause did not cross the stage boundary");
            }
            reports.push(report);
        }
    }
    Ok(json!({"status":"passed","model_loads":1,"parallel_lanes":args.lanes,
        "elapsed_seconds":started.elapsed().as_secs_f64(),"synthetic_protocol_fixture":true,
        "file_hash_checks":false,"cases":reports}))
}

#[path = "../diagnostics.rs"] mod diagnostics;
