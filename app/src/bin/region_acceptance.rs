#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
#[path="../mode_transition.rs"] mod mode_transition;
// One-load real CUDA acceptance for structured region events and a manual
// cross-region edit. The input prefix is explicitly human-authored test data,
// never presented as a trained model's previous output or an image-quality test.

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

use anyhow::{ensure, Context, Result};
use document::{Document, OperationSource, Primitive, ShapeKind};
use history_context::RawHistoryStore;
use inference::{
    EngineConfig, GenerationRequest, InferencePrimitive, InferenceSettings, NativeEngine,
};
use region_state::{marker_record, DrawingRegion, Stage, BOS_RECORD, INTERRUPT_RECORD};
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

fn tokens(p: &Primitive) -> Result<[u32; 9]> {
    Ok(p.to_tokens().map_err(anyhow::Error::msg)?.map(u32::from))
}

fn native_canvas(doc: &Document) -> Result<Vec<InferencePrimitive>> {
    doc.primitives
        .iter()
        .map(|p| Ok(InferencePrimitive { tokens: tokens(p)? }))
        .collect()
}

fn synthetic_global_prefix() -> Vec<Primitive> {
    let mut canvas = vec![Primitive::background([238, 218, 230])];
    for i in 0..85 {
        let mut p = Primitive::new(
            if i % 2 == 0 {
                ShapeKind::Ellipse
            } else {
                ShapeKind::Rectangle
            },
            30. + (i % 10) as f32 * 48.,
            30. + (i / 10) as f32 * 50.,
            38.,
            42.,
            [
                80 + (i % 30) as u8 * 4,
                90 + (i % 25) as u8 * 4,
                120 + (i % 20) as u8 * 4,
            ],
        );
        p.angle_deg = (i % 6) as f32 * 12.;
        canvas.push(p);
    }
    canvas
}

fn main() -> Result<()> {
    let mut checkpoint = PathBuf::from("models/model.pt");
    let mut tokenizer = PathBuf::from("assets/caption_spm_1500.model");
    let mut output = PathBuf::from("artifacts/native_region_acceptance.json");
    let mut gpu_sampler = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--checkpoint" => checkpoint = args.next().context("--checkpoint 缺少路径")?.into(),
            "--tokenizer" => tokenizer = args.next().context("--tokenizer 缺少路径")?.into(),
            "--output" => output = args.next().context("--output 缺少路径")?.into(),
            "--gpu-sampling" => gpu_sampler = true,
            "--help" | "-h" => {
                println!("region-acceptance --checkpoint MODEL.pt --tokenizer TOKENIZER.model --output REPORT.json\nOne real CUDA model load; human-authored prefix, manual region interrupt, structured events, one generated primitive.");
                return Ok(());
            }
            _ => anyhow::bail!("未知参数：{arg}"),
        }
    }
    let start = Instant::now();
    let mut doc = Document::new();
    doc.prompt = "手工图元前缀的区域续画验收".into();
    doc.settings.cfg = 1.;
    doc.settings.renyi_alpha = 4.;
    doc.settings.seed = 2026103005;
    let mut fixture = synthetic_global_prefix();
    fixture.push(Primitive::new(
        ShapeKind::Ellipse,
        64.,
        64.,
        36.,
        30.,
        [58, 112, 208],
    ));
    let fixture_node = doc
        .commit_canvas(
            "人工验收前缀：86个全局图元及粗区域0内1个图元",
            OperationSource::User,
            fixture,
        )
        .map_err(anyhow::Error::msg)?;
    ensure!(doc.primitives.len() == 87, "人工 fixture 图元数量错误");
    let mut fixture_raw = vec![BOS_RECORD];
    for p in &doc.primitives[..86] {
        fixture_raw.push(tokens(p)?);
    }
    fixture_raw.push(marker_record(Stage::Coarse, 0).map_err(anyhow::Error::msg)?);
    fixture_raw.push(tokens(&doc.primitives[86])?);
    let initial = DrawingRegion::from_history(&fixture_raw).map_err(anyhow::Error::msg)?;
    ensure!(
        initial.stage == Stage::Coarse
            && initial.index == Some(0)
            && initial.generated_in_region == 1,
        "人工 fixture 活动区域错误"
    );
    let mut store = RawHistoryStore::new();
    store
        .finish(&doc, fixture_node, &fixture_raw)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        store
            .context_for(&doc, fixture_node)
            .map_err(anyhow::Error::msg)?
            .as_ref()
            == Some(&fixture_raw),
        "fixture sidecar 附存失败"
    );

    let mut edited = doc.primitives.clone();
    edited.push(Primitive::new(
        ShapeKind::Rectangle,
        191.5,
        191.5,
        58.,
        42.,
        [244, 98, 66],
    ));
    let manual_node = doc
        .commit_canvas(
            "人工跨区：在粗区域5中心添加矩形",
            OperationSource::User,
            edited,
        )
        .map_err(anyhow::Error::msg)?;
    let manual_region = store
        .reconcile_manual(&doc, fixture_node, manual_node)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        manual_region.stage == Stage::Coarse && manual_region.index == Some(5),
        "手工跨区没有激活 coarse5"
    );
    let manual_raw = store
        .context_for(&doc, manual_node)
        .map_err(anyhow::Error::msg)?
        .context("手工跨区缺少真实模型上下文")?;
    let expected_tail = [
        INTERRUPT_RECORD,
        marker_record(Stage::Coarse, 5).map_err(anyhow::Error::msg)?,
        tokens(doc.primitives.last().unwrap())?,
    ];
    ensure!(
        manual_raw.starts_with(&fixture_raw),
        "手工跨区改写了原始前缀"
    );
    ensure!(
        &manual_raw[fixture_raw.len()..] == expected_tail.as_slice(),
        "缺少 interrupt -> same-stage marker -> user primitive 顺序"
    );
    ensure!(
        manual_region.generated_in_region == 1 && manual_region.completed_coarse == 0,
        "中断被误记为完成区域"
    );
    let before_model = doc.primitives.len();
    let mut engine = NativeEngine::load(
        &EngineConfig {
            model_path: checkpoint.clone(),
            tokenizer_path: tokenizer,
            use_cuda: true,
        },
        |s| eprintln!("{s}"),
    )?;
    engine.set_gpu_sampling_override(if gpu_sampler { Some(true) } else { None });
    let request = GenerationRequest { redraw_regions:Vec::new(),
        prompt: doc.prompt.clone(),
        settings: InferenceSettings { half_region_switch: false, pause_at_stage_end: false,
            cfg_scale: 1.,
            renyi_alpha: 4.,
            seed: doc.settings.seed,
            max_primitives: 1,
            context_overclock: false,
        },
        primitives: native_canvas(&doc)?,
        manual_review_regions: Vec::new(), extra_fine_context:Vec::new(),stage_floor:None,pending_stage_context:Vec::new(), history_records: Some(manual_raw.clone()),
    };
    let events = RefCell::new(Vec::<DrawingRegion>::new());
    let stop = AtomicBool::new(false);
    let mut callbacks = Vec::<Value>::new();
    let mut callback_error: Option<String> = None;
    let generation=engine.generate_with_regions(&request,&stop,|event|{
        if callback_error.is_some(){return;}
        let result=(||->Result<()>{
            let region=events.borrow().last().cloned().context("模型图元前没有结构化区域事件")?;
            let mut t=[0u16;9];
            for i in 0..9{t[i]=event.primitive.tokens[i].try_into().context("模型图元token超出u16")?;}
            let p=Primitive::from_tokens(0,t).map_err(anyhow::Error::msg)?;
            ensure!(region.contains_center(p.x,p.y),"生成中心不在此前区域事件的合法坐标内");
            let node=doc.commit_generated(p.clone()).map_err(anyhow::Error::msg)?;
            store.attach(&doc,node,&event.history_records).map_err(anyhow::Error::msg)?;
            callbacks.push(json!({"node":node,"tokens":event.primitive.tokens,"center":[p.x,p.y],"region_before_commit":region}));
            Ok(())
        })();
        if let Err(e)=result{callback_error=Some(format!("{e:#}"));stop.store(true,Ordering::Relaxed);}
    },|s|eprintln!("{s}"),|region|{eprintln!("REGION {:?} {:?} count{}",region.stage,region.index,region.generated_in_region);events.borrow_mut().push(region);})?;
    if let Some(error) = callback_error {
        anyhow::bail!("{error}");
    }
    ensure!(
        generation.generated == 1 && callbacks.len() == 1,
        "真实模型未产生且仅产生一个完整图元"
    );
    ensure!(
        doc.primitives.len() == before_model + 1,
        "画布没有完整保留人工图元"
    );
    let events = events.into_inner();
    ensure!(
        events.first().map(|r| (r.stage, r.index)) == Some((Stage::Coarse, Some(5))),
        "第一个结构化事件不是 coarse5"
    );
    store
        .finish(&doc, doc.current, &generation.history_records)
        .map_err(anyhow::Error::msg)?;
    let generated_node = doc.current;
    let generated_raw = store
        .context_for(&doc, generated_node)
        .map_err(anyhow::Error::msg)?
        .context("模型节点缺少上下文")?;
    ensure!(
        generated_raw == generation.history_records,
        "模型最终 sidecar 与summary不一致"
    );

    // A second manual fork checks that per-node region state does not leak
    // across branches. It does not request a second model load or generation.
    doc.jump(fixture_node).map_err(anyhow::Error::msg)?;
    let mut alternative = doc.primitives.clone();
    alternative.push(Primitive::new(
        ShapeKind::Ellipse,
        319.5,
        191.5,
        44.,
        38.,
        [72, 194, 112],
    ));
    let alternative_node = doc
        .commit_canvas(
            "人工另一分支：在粗区域6添加椭圆",
            OperationSource::User,
            alternative,
        )
        .map_err(anyhow::Error::msg)?;
    let alternative_region = store
        .reconcile_manual(&doc, fixture_node, alternative_node)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        alternative_region.index == Some(6),
        "另一分支没有独立 coarse6 状态"
    );
    ensure!(
        doc.nodes[fixture_node].children == vec![manual_node, alternative_node],
        "人工分叉没有保留两条路线"
    );
    doc.jump(generated_node).map_err(anyhow::Error::msg)?;
    let restored = store
        .context_for(&doc, generated_node)
        .map_err(anyhow::Error::msg)?
        .context("返回模型分支时上下文丢失")?;
    ensure!(restored == generated_raw, "另一分支污染了原分支控制序列");
    let current_region = DrawingRegion::from_history(&restored).map_err(anyhow::Error::msg)?;
    let report = json!({"status":"complete","native_rust":true,"real_cuda_model_loads":1,"model":engine.model_description,"checkpoint":checkpoint,
        "fixture":{"human_authored":true,"model_generated_prefix":false,"global_visible_primitives":86,"regional_visible_primitives":1,
            "initial_active_region":initial,"quality_scope":"synthetic-prefix control/API acceptance, not image quality or a real training prefix"},
        "manual_cross_region":{"parent":fixture_node,"node":manual_node,"raw_prefix_preserved":true,"appended_records":expected_tail,
            "expected_control_order":["interrupt_end","coarse marker5","user primitive"],"region":manual_region},
        "region_events":events,"generated_primitives":callbacks,"generated_count":generation.generated,"generation_seconds":generation.elapsed_seconds,
        "final_history_records":generation.history_records,"branch":{"fork":fixture_node,"children":doc.nodes[fixture_node].children,
            "original_model_leaf":generated_node,"alternative_manual_leaf":alternative_node,"alternative_region":alternative_region,"original_raw_unchanged":true},
        "current_node":doc.current,"current_region":current_region,"all_manual_primitives_retained":true,"seconds":start.elapsed().as_secs_f64()});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"status":"complete","output":output,"generated_count":generation.generated,"real_cuda_model_loads":1,
        "initial_generation_region":"coarse5","current_region":current_region,"manual_prefix_preserved":true})
        )?
    );
    Ok(())
}

#[path = "../diagnostics.rs"] mod diagnostics;
