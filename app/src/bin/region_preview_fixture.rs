#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
#[path="../mode_transition.rs"] mod mode_transition;
// Hand-authored UI fixtures. No model weights, GPU, or inference are used.
#[path = "../acceleration.rs"]
mod acceleration;
#[path = "../document.rs"]
mod document;
#[path = "../history_context.rs"]
mod history_context;
#[path = "../region_state.rs"]
mod region_state;

use anyhow::{ensure, Context, Result};
use document::{Document, OperationSource, Primitive, ShapeKind};
use history_context::RawHistoryStore;
use region_state::{marker_record, DrawingRegion, RegionRecord, Stage, BOS_RECORD, END_RECORD};
use serde_json::json;
use std::path::PathBuf;

const PROMPT: &str = "手工图元界面验收：以下头像由程序手工摆放原生图元，用于检查全图、粗网格和细网格的区域显示；不是模型生成结果。";

fn manual_portrait() -> Vec<Primitive> {
    let shape = |kind, x, y, width, height, angle, color| {
        let mut primitive = Primitive::new(kind, x, y, width, height, color);
        primitive.angle_deg = angle;
        primitive
    };
    use ShapeKind::{Ellipse, Rectangle};
    vec![
        Primitive::background([234, 240, 250]),
        shape(Ellipse, 256.0, 270.0, 310.0, 390.0, 0.0, [236, 180, 32]),
        shape(Ellipse, 256.0, 280.0, 224.0, 286.0, 0.0, [252, 206, 164]),
        shape(Ellipse, 256.0, 280.0, 220.0, 282.0, 0.0, [252, 206, 164]),
        shape(Rectangle, 256.0, 458.0, 302.0, 108.0, 0.0, [44, 88, 184]),
        shape(Rectangle, 256.0, 458.0, 298.0, 104.0, 0.0, [44, 88, 184]),
        shape(Rectangle, 250.0, 158.0, 230.0, 76.0, 10.0, [246, 194, 32]),
        shape(Rectangle, 174.0, 213.0, 68.0, 154.0, 18.0, [246, 194, 32]),
        shape(Ellipse, 214.0, 278.0, 36.0, 26.0, 0.0, [28, 94, 216]),
        shape(Ellipse, 296.0, 278.0, 36.0, 26.0, 0.0, [28, 94, 216]),
        shape(Ellipse, 214.0, 279.0, 15.0, 20.0, 0.0, [8, 28, 80]),
        shape(Ellipse, 296.0, 279.0, 15.0, 20.0, 0.0, [8, 28, 80]),
        shape(Ellipse, 260.0, 350.0, 62.0, 32.0, 0.0, [174, 34, 62]),
        shape(Ellipse, 264.0, 359.0, 26.0, 14.0, 0.0, [242, 104, 142]),
    ]
}

fn base_document() -> Result<Document> {
    let mut document = Document::new();
    document.prompt = PROMPT.into();
    document
        .commit_canvas(
            "手工图元界面验收：导入头像",
            OperationSource::User,
            manual_portrait(),
        )
        .map_err(anyhow::Error::msg)?;
    ensure!(
        document.current == 1 && document.nodes.len() == 2,
        "Manual fixture must create one user state"
    );
    ensure!(
        document.nodes[1].source == OperationSource::User,
        "Preview is not model generated"
    );
    Ok(document)
}

fn raw_history(document: &Document, stage: Stage) -> Result<Vec<RegionRecord>> {
    let mut records = vec![BOS_RECORD];
    for primitive in &document.primitives {
        records.push(
            primitive
                .to_tokens()
                .map_err(anyhow::Error::msg)?
                .map(u32::from),
        );
    }
    match stage {
        Stage::Global => {}
        Stage::Coarse => records.push(marker_record(Stage::Coarse, 5).map_err(anyhow::Error::msg)?),
        Stage::Fine => {
            // Canonical closed empty coarse visits are legal. Visible portrait
            // records remain in global history exactly once and in z order.
            for index in 0..16 {
                records.push(marker_record(Stage::Coarse, index).map_err(anyhow::Error::msg)?);
                records.push(END_RECORD);
            }
            records.push(marker_record(Stage::Fine, 45).map_err(anyhow::Error::msg)?);
        }
        Stage::Complete => anyhow::bail!("This fixture only previews three active stages"),
    }
    Ok(records)
}

fn write_fixture(output: &PathBuf, document: &Document, stage: Stage) -> Result<serde_json::Value> {
    let raw = raw_history(document, stage)?;
    let region = DrawingRegion::from_history(&raw).map_err(anyhow::Error::msg)?;
    ensure!(
        region.stage() == stage,
        "History stage does not match requested fixture"
    );
    let expected_index = match stage {
        Stage::Coarse => Some(5),
        Stage::Fine => Some(45),
        _ => None,
    };
    ensure!(
        region.index == expected_index && !region.awaiting_region && !region.completed,
        "Fixture region is not active"
    );
    ensure!(
        stage != Stage::Fine || region.completed_coarse == 16,
        "Fine fixture must close all 16 coarse visits"
    );
    let visible: Vec<[u16; 9]> = document
        .primitives
        .iter()
        .map(|primitive| primitive.to_tokens().map_err(anyhow::Error::msg))
        .collect::<Result<_>>()?;
    let projection: Vec<[u16; 9]> = raw
        .iter()
        .filter(|record| region_state::validate_visible(record).is_ok())
        .map(|record| record.map(|token| token as u16))
        .collect();
    ensure!(
        projection == visible,
        "Raw visible projection differs from the preview canvas"
    );
    let mut store = RawHistoryStore::new();
    store
        .finish(document, document.current, &raw)
        .map_err(anyhow::Error::msg)?;
    let mounted = store
        .context_for(document, document.current)
        .map_err(anyhow::Error::msg)?
        .context("Fixture history mount is absent")?;
    ensure!(
        mounted == raw,
        "Production history store did not round-trip raw records"
    );
    let restored = DrawingRegion::from_history(&mounted).map_err(anyhow::Error::msg)?;
    ensure!(
        restored == region,
        "Mounted stage differs from serialized stage"
    );
    let report = json!({
        "schema_version":1,
        "purpose":"manual_region_indicator_ui_acceptance",
        "model_generated":false,
        "prompt":PROMPT,
        "primitives":visible,
        "history_records":raw,
        "operations":[{
            "id":1,"parent":0,"title":"手工图元界面验收：导入头像",
            "prompt":PROMPT,"settings":document.settings,"primitives":document.primitives
        }],
        "current_node":1,
        "drawing_region":region,
        "expected":{
            "stage":stage,"index":expected_index,"visible_primitive_count":visible.len(),
            "visual_bounds":restored.bounds(),"legal_center_bounds":restored.center_bounds(),
            "raw_visible_projection_matches":true,"production_history_mount_verified":true,
            "completed_coarse_regions":restored.completed_coarse,
            "model_generated":false
        }
    });
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    Ok(
        json!({"path":output,"stage":stage,"index":expected_index,"visible_primitives":visible.len(),"raw_records":report["history_records"].as_array().unwrap().len(),"model_generated":false}),
    )
}

fn main() -> Result<()> {
    let mut output_dir = PathBuf::from("artifacts");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-dir" => output_dir = args.next().context("--output-dir 缺少路径")?.into(),
            "--help" => {
                println!("region-preview-fixture [--output-dir artifacts] — manual native UI fixtures; no model or GPU");
                return Ok(());
            }
            _ => anyhow::bail!("未知参数：{arg}"),
        }
    }
    let document = base_document()?;
    let mut outputs = Vec::new();
    for (name, stage) in [
        ("global", Stage::Global),
        ("coarse", Stage::Coarse),
        ("fine", Stage::Fine),
    ] {
        outputs.push(write_fixture(
            &output_dir.join(format!("region_{name}_preview.json")),
            &document,
            stage,
        )?);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"status":"complete","outputs":outputs,"model_generated":false})
        )?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture_raw_records_preserve_manual_canvas_and_exact_stage() {
        let document = base_document().unwrap();
        for stage in [Stage::Global, Stage::Coarse, Stage::Fine] {
            let raw = raw_history(&document, stage).unwrap();
            let region = DrawingRegion::from_history(&raw).unwrap();
            assert_eq!(region.stage(), stage);
            let projection: Vec<_> = raw
                .iter()
                .filter(|record| region_state::validate_visible(record).is_ok())
                .copied()
                .collect();
            let expected: Vec<_> = document
                .primitives
                .iter()
                .map(|primitive| primitive.to_tokens().unwrap().map(u32::from))
                .collect();
            assert_eq!(projection, expected);
            let mut store = RawHistoryStore::new();
            store.finish(&document, document.current, &raw).unwrap();
            assert_eq!(
                store
                    .context_for(&document, document.current)
                    .unwrap()
                    .unwrap(),
                raw
            );
        }
    }
}
