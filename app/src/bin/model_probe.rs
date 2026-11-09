#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
// Native-only inference acceptance CLI; deliberately independent of the GUI.
#[path = "../acceleration.rs"]
mod acceleration;
#[path = "../inference.rs"]
mod inference;
#[path = "../model/mod.rs"]
mod model;
#[path = "../region_state.rs"]
mod region_state;

use anyhow::{bail, Context, Result};
use inference::{
    EngineConfig, GenerationRequest, InferencePrimitive, InferenceSettings, NativeEngine,
};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

const PROMPT: &str =
    "金发蓝眼的白皙圆脸女性以明亮动画风呈现，张嘴露舌，粉色背景可见，取景包含头部和肩颈。";
const EXPECTED_IDS: &[u32] = &[
    2, 262, 398, 356, 264, 302, 452, 291, 350, 304, 954, 427, 673, 347, 260, 434, 585, 894, 260,
    281, 286, 274, 260, 1340, 965, 874, 819, 376, 267, 448, 618, 261, 3,
];

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut checkpoint = PathBuf::from("models/model.pt");
    let mut tokenizer = PathBuf::from("assets/caption_spm_1500.model");
    let mut prompt = PROMPT.to_owned();
    let mut output = None;
    let mut cuda = false;
    let mut count = 2;
    let mut cfg = 1.;
    let mut alpha = 4.;
    let mut seed = 20261011;
    let mut tokenizer_only = false;
    let mut forward_only = false;
    let mut check_prefill = false;
    let mut stop_after_first = false;
    let mut stop_during_device = false;
    let mut gpu_sampler = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--checkpoint" => checkpoint = args.next().context("--checkpoint 缺少路径")?.into(),
            "--tokenizer" => tokenizer = args.next().context("--tokenizer 缺少路径")?.into(),
            "--prompt" => prompt = args.next().context("--prompt 缺少文字")?,
            "--output" => output = Some(PathBuf::from(args.next().context("--output 缺少路径")?)),
            "--count" => count = args.next().context("--count 缺少数字")?.parse()?,
            "--cfg" => cfg = args.next().context("--cfg 缺少数字")?.parse()?,
            "--alpha" => alpha = args.next().context("--alpha 缺少数字")?.parse()?,
            "--seed" => seed = args.next().context("--seed 缺少数字")?.parse()?,
            "--cuda" => cuda = true,
            "--tokenizer-only" => tokenizer_only = true,
            "--forward-only" => forward_only = true,
            "--check-prefill" => check_prefill = true,
            "--stop-after-first" => stop_after_first = true,
            "--stop-during-device-frame" => stop_during_device = true,
            "--gpu-sampling" => gpu_sampler = true,
            "--help" => {
                println!("model-probe --checkpoint MODEL.pt --tokenizer TOKENIZER.model [--cuda] [--count 2] [--cfg 1] [--alpha 4] [--seed 20261011] [--output report.json] [--forward-only|--tokenizer-only|--stop-after-first|--check-prefill]");
                return Ok(());
            }
            _ => bail!("未知参数：{arg}"),
        }
    }
    let start = Instant::now();
    let result = if tokenizer_only {
        let sp = sentencepiece_rs::SentencePieceProcessor::open(&tokenizer)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut ids = vec![2];
        ids.extend(
            sp.encode_to_ids(&prompt)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .into_iter()
                .map(|v| v as u32),
        );
        ids.push(3);
        let known_match = prompt != PROMPT || ids == EXPECTED_IDS;
        if !known_match {
            bail!("原生分词结果与已记录 SentencePiece 输出不一致：{ids:?}");
        }
        json!({"status":"complete", "native_rust":true, "scope":"tokenizer_only", "text_ids":ids,
            "matches_recorded_sentencepiece":known_match, "seconds":start.elapsed().as_secs_f64()})
    } else {
        let config = EngineConfig {
            model_path: checkpoint.clone(),
            tokenizer_path: tokenizer,
            use_cuda: cuda,
        };
        let mut engine = NativeEngine::load(&config, |s| eprintln!("{s}"))?;
        if check_prefill {
            let prefix: Vec<_> = (0..8)
                .map(|i| InferencePrimitive {
                    tokens: [
                        256 + i * 32,
                        1280 + i * 32,
                        2048,
                        2518,
                        4566,
                        6415,
                        6702,
                        6830,
                        6958,
                    ],
                })
                .collect();
            let chunked = engine.forward_probe(&prompt, &prefix)?;
            let sequential = engine.forward_probe_sequential(&prompt, &prefix)?;
            let max_abs = chunked
                .attributes
                .iter()
                .zip(&sequential.attributes)
                .map(|(a, b)| (a - b).abs() as f64)
                .fold(0., f64::max);
            let error_energy = chunked
                .attributes
                .iter()
                .zip(&sequential.attributes)
                .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
                .sum::<f64>();
            let signal_energy = sequential
                .attributes
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>();
            let relative_rms = (error_energy / signal_energy.max(1e-24)).sqrt();
            if relative_rms > 0.03 {
                bail!("批量预填充与逐位置原生前向差异过大：relative_rms={relative_rms}, max_abs={max_abs}");
            }
            json!({"status":"complete","native_rust":true,"scope":"real_weights_chunked_vs_sequential_prefill",
                "model":engine.model_description,"text_ids":chunked.text_ids,"max_absolute_error":max_abs,
                "relative_rms_error":relative_rms,"relative_rms_threshold":0.03,
                "prefix_primitives":prefix.len(),
                "chunked_termination":chunked.termination,"sequential_termination":sequential.termination,
                "note":"Native implementation self-consistency; not a comparison with Python/Triton",
                "seconds":start.elapsed().as_secs_f64()})
        } else if forward_only {
            let prediction = engine.forward_probe(&prompt, &[])?;
            json!({"status":"complete", "native_rust":true, "scope":"real_checkpoint_forward",
                "model":engine.model_description, "checkpoint":checkpoint, "prompt":prompt,
                "text_ids":prediction.text_ids, "attribute_logits":prediction.attributes,
                "termination_logits":prediction.termination, "seconds":start.elapsed().as_secs_f64()})
        } else {
            let stop = AtomicBool::new(false);
            let request = GenerationRequest { redraw_regions:Vec::new(),
                prompt: prompt.clone(),
                settings: InferenceSettings { half_region_switch: false, pause_at_stage_end: false,
                    cfg_scale: cfg,
                    renyi_alpha: alpha,
                    seed,
                    max_primitives: count,
                    context_overclock: false,
                },
                primitives: Vec::new(),
                manual_review_regions: Vec::new(), extra_fine_context:Vec::new(),stage_floor:None,pending_stage_context:Vec::new(), history_records: None,
            };
            if gpu_sampler {
                engine.set_gpu_sampling_override(Some(true));
            }
            if stop_during_device {
                engine.set_gpu_sampling_override(Some(true));
                engine.diagnostic_stop_after_device_enqueue(true);
            }
            let mut records = Vec::new();
            let generation = engine.generate(
                &request,
                &stop,
                |event| {
                    eprintln!(
                        "完整图元 {}：{:?}",
                        records.len() + 1,
                        event.primitive.tokens
                    );
                    records.push(event.primitive.tokens);
                    if stop_after_first {
                        stop.store(true, Ordering::Relaxed);
                    }
                },
                |s| eprintln!("{s}"),
            )?;
            if records.len() != generation.generated {
                bail!("完整图元回调数量不一致");
            }
            if stop_during_device && (generation.generated != 0 || !generation.stopped) {
                bail!("enqueued GPU frame leaked a visible primitive after stop");
            }
            for row in &records {
                model::validate_visible(row)?;
            }
            if let Some(background) = records.first() {
                if background[5] != model::OFFSETS[5] {
                    bail!("首背景图元类型错误");
                }
            }
            json!({"status":"complete", "native_rust":true, "scope":"real_checkpoint_generation",
                "model":engine.model_description,"checkpoint":checkpoint,"prompt":prompt,
                "cfg":cfg,"renyi_alpha":alpha,"seed":seed,"generated":generation.generated,
                "stopped":generation.stopped,"completed_all_regions":generation.completed,
                "context_limit":generation.context_limit,"generation_seconds":generation.elapsed_seconds,
                "history_records":generation.history_records,"primitives":records,"seconds":start.elapsed().as_secs_f64()})
        }
    };
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_vec_pretty(&result)?)?;
        println!("{}", path.display());
    } else {
        println!("{}", serde_json::to_string_pretty(&result)?);
    }
    Ok(())
}

#[path = "../diagnostics.rs"] mod diagnostics;
