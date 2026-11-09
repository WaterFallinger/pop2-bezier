#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
// Native editor-only fixture. None of these primitives are model generated.
// Every operation is committed through the production delta-backed Document.

#[path = "../acceleration.rs"]
mod acceleration;
#[path = "../document.rs"]
mod document;

use anyhow::{ensure, Context, Result};
use document::{Document, NodeId, OperationSource, Primitive, Settings, ShapeKind};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeSet, path::PathBuf};

const PROMPT: &str =
    "编辑器状态树演示：手工图元和编辑操作，用于检查分叉、跳转和过滤；不是模型生成结果。";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Operation {
    id: NodeId,
    parent: NodeId,
    title: String,
    prompt: String,
    settings: Settings,
    primitives: Vec<Primitive>,
}

fn edit(
    doc: &mut Document,
    operations: &mut Vec<Operation>,
    parent: NodeId,
    title: impl Into<String>,
    change: impl FnOnce(&mut Vec<Primitive>),
) -> Result<NodeId> {
    doc.jump(parent).map_err(anyhow::Error::msg)?;
    doc.prompt = PROMPT.into();
    let mut next = doc.primitives.clone();
    change(&mut next);
    let title = title.into();
    let expected_id = doc.nodes.len();
    doc.settings.seed = 202610300000 + expected_id as u64;
    let id = doc
        .commit_canvas(title.clone(), OperationSource::User, next)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        id == expected_id,
        "Fixture operation did not produce one new state: {title}"
    );
    ensure!(
        doc.nodes[id].parent == Some(parent),
        "Fixture parent mismatch"
    );
    ensure!(
        !doc.nodes[id].patches.is_empty(),
        "Fixture contains a no-op"
    );
    operations.push(Operation {
        id,
        parent,
        title,
        prompt: doc.prompt.clone(),
        settings: doc.settings.clone(),
        primitives: doc.primitives.clone(),
    });
    Ok(id)
}

fn background(canvas: &mut [Primitive], color: [u8; 3]) {
    canvas
        .iter_mut()
        .find(|p| p.shape == ShapeKind::Background)
        .unwrap()
        .color = color;
}

fn hair(canvas: &mut [Primitive], color: [u8; 3]) {
    // The first rectangle is the original, stable-ID hair cap.
    canvas
        .iter_mut()
        .find(|p| p.shape == ShapeKind::Rectangle)
        .unwrap()
        .color = color;
}

fn eyes(canvas: &mut [Primitive], color: [u8; 3]) {
    for p in canvas.iter_mut().filter(|p| p.id == 4 || p.id == 5) {
        p.color = color;
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut output = PathBuf::from("artifacts/editor_tree_fixture.json");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => output = args.next().context("--output requires a path")?.into(),
            "--help" | "-h" => {
                println!("tree-preview-fixture [--output artifacts/editor_tree_fixture.json]\nCreates and replays a native hand-edited document tree; never loads a model or GPU.");
                return Ok(());
            }
            _ => anyhow::bail!("Unknown argument: {arg}"),
        }
    }
    let mut doc = Document::new();
    let mut operations = Vec::new();

    let bg = edit(&mut doc, &mut operations, 0, "添加粉色背景", |v| {
        v.push(Primitive::background([246, 220, 232]));
    })?;
    let face = edit(&mut doc, &mut operations, bg, "添加面部椭圆", |v| {
        v.push(Primitive::new(
            ShapeKind::Ellipse,
            256.,
            270.,
            200.,
            250.,
            [248, 208, 168],
        ));
    })?;
    let hair_base = edit(&mut doc, &mut operations, face, "添加金色头发", |v| {
        v.push(Primitive::new(
            ShapeKind::Rectangle,
            256.,
            174.,
            224.,
            128.,
            [238, 178, 44],
        ));
    })?;
    let eye_base = edit(
        &mut doc,
        &mut operations,
        hair_base,
        "同时添加两只蓝眼睛",
        |v| {
            v.push(Primitive::new(
                ShapeKind::Ellipse,
                218.,
                254.,
                38.,
                28.,
                [44, 118, 228],
            ));
            v.push(Primitive::new(
                ShapeKind::Ellipse,
                294.,
                254.,
                38.,
                28.,
                [44, 118, 228],
            ));
        },
    )?;
    let common = edit(
        &mut doc,
        &mut operations,
        eye_base,
        "添加微笑嘴型",
        |v| {
            v.push(Primitive::new(
                ShapeKind::Ellipse,
                256.,
                320.,
                54.,
                28.,
                [208, 70, 92],
            ));
        },
    )?;

    // A 30-operation branch deliberately stresses long single-child runs.
    let mut long_chain_ids = Vec::new();
    let mut parent = common;
    for step in 0..30 {
        let title = match step % 6 {
            0 => format!("主线 {:02}：面部向右移动", step + 1),
            1 => format!("主线 {:02}：调整金色头发", step + 1),
            2 => format!("主线 {:02}：调整嘴巴高度", step + 1),
            3 => format!("主线 {:02}：旋转当前图元", step + 1),
            4 => format!("主线 {:02}：添加衣服色块", step + 1),
            _ => format!("主线 {:02}：调整背景色", step + 1),
        };
        parent = edit(&mut doc, &mut operations, parent, title, |v| {
            match step % 6 {
                0 => {
                    for p in v.iter_mut().filter(|p| p.shape != ShapeKind::Background) {
                        p.x += 6.;
                    }
                }
                1 => hair(
                    v,
                    [
                        238 - (step as u8) * 2,
                        164 + (step as u8) * 2,
                        32 + (step as u8) * 2,
                    ],
                ),
                2 => {
                    let mouth = v.iter_mut().find(|p| p.id == 6).unwrap();
                    mouth.height = if mouth.height < 40. { 50. } else { 24. };
                }
                3 => {
                    let p = v
                        .iter_mut()
                        .rev()
                        .find(|p| p.shape != ShapeKind::Background)
                        .unwrap();
                    p.angle_deg += 12.;
                }
                4 => v.push(Primitive::new(
                    ShapeKind::Rectangle,
                    256. + step as f32,
                    414.,
                    190.,
                    70.,
                    [66 + (step as u8) * 2, 102, 196],
                )),
                _ => background(v, [242, 204 + (step as u8), 226 - (step as u8)]),
            }
        })?;
        long_chain_ids.push(parent);
    }

    // Fork one: a second path from the common face.
    let cool_a = edit(
        &mut doc,
        &mut operations,
        common,
        "分支B：更换蓝色背景",
        |v| background(v, [178, 214, 246]),
    )?;
    let cool_b = edit(
        &mut doc,
        &mut operations,
        cool_a,
        "分支B：改成紫色头发",
        |v| hair(v, [146, 86, 202]),
    )?;
    let cool_leaf = edit(
        &mut doc,
        &mut operations,
        cool_b,
        "分支B：修改眼睛颜色",
        |v| eyes(v, [54, 196, 120]),
    )?;

    // Fork two is nested inside the first long-chain path.
    let nested_parent = long_chain_ids[8];
    let nested_a = edit(
        &mut doc,
        &mut operations,
        nested_parent,
        "分支A1：背景改为暖黄色",
        |v| background(v, [248, 232, 164]),
    )?;
    let nested_b = edit(
        &mut doc,
        &mut operations,
        nested_a,
        "分支A1：头发改为珊瑚色",
        |v| hair(v, [230, 110, 94]),
    )?;
    let nested_c = edit(
        &mut doc,
        &mut operations,
        nested_b,
        "分支A1：扩大左眼",
        |v| {
            let p = v.iter_mut().find(|p| p.id == 4).unwrap();
            p.width = 54.;
            p.height = 40.;
        },
    )?;
    let nested_leaf = edit(
        &mut doc,
        &mut operations,
        nested_c,
        "分支A1：添加深色衣领",
        |v| {
            v.push(Primitive::new(
                ShapeKind::Rectangle,
                268.,
                392.,
                122.,
                40.,
                [70, 80, 126],
            ));
        },
    )?;

    // Fork three is nested again, producing a clearly separate subtree.
    let deep_a = edit(
        &mut doc,
        &mut operations,
        nested_b,
        "分支A1a：改变嘴型",
        |v| {
            let p = v.iter_mut().find(|p| p.id == 6).unwrap();
            p.width = 76.;
            p.height = 46.;
            p.color = [208, 54, 88];
        },
    )?;
    let deep_b = edit(
        &mut doc,
        &mut operations,
        deep_a,
        "分支A1a：两眼改为深蓝",
        |v| eyes(v, [30, 68, 170]),
    )?;
    let deep_leaf = edit(
        &mut doc,
        &mut operations,
        deep_b,
        "分支A1a：添加橙色发饰",
        |v| {
            let mut p = Primitive::new(ShapeKind::Rectangle, 346., 156., 52., 26., [246, 126, 44]);
            p.angle_deg = 24.;
            v.push(p);
        },
    )?;
    let deep_other = edit(
        &mut doc,
        &mut operations,
        deep_a,
        "分支A1b：添加绿色发饰",
        |v| {
            let mut p = Primitive::new(ShapeKind::Ellipse, 180., 160., 44., 36., [70, 190, 126]);
            p.angle_deg = 18.;
            v.push(p);
        },
    )?;

    // One more root-level creative route makes the root itself a fork.
    let root_alt = edit(
        &mut doc,
        &mut operations,
        0,
        "新路线：添加薄荷色背景",
        |v| {
            v.push(Primitive::background([196, 238, 218]));
        },
    )?;
    let root_alt_two = edit(
        &mut doc,
        &mut operations,
        root_alt,
        "新路线：添加橙色矩形",
        |v| {
            let mut p =
                Primitive::new(ShapeKind::Rectangle, 220., 242., 206., 176., [236, 142, 56]);
            p.angle_deg = 18.;
            v.push(p);
        },
    )?;
    let root_leaf = edit(
        &mut doc,
        &mut operations,
        root_alt_two,
        "新路线：添加蓝色椭圆",
        |v| {
            v.push(Primitive::new(
                ShapeKind::Ellipse,
                326.,
                306.,
                156.,
                124.,
                [64, 136, 218],
            ));
        },
    )?;

    // Pick a state on a single-child path so implementations can explicitly
    // decide whether the current node remains visible in branch-only mode.
    let current_node = long_chain_ids[20];
    doc.jump(current_node).map_err(anyhow::Error::msg)?;
    let roots: Vec<_> = doc
        .nodes
        .iter()
        .filter(|n| n.parent.is_none())
        .map(|n| n.id)
        .collect();
    let leaves: Vec<_> = doc
        .nodes
        .iter()
        .filter(|n| n.children.is_empty())
        .map(|n| n.id)
        .collect();
    let forks: Vec<_> = doc
        .nodes
        .iter()
        .filter(|n| n.children.len() > 1)
        .map(|n| n.id)
        .collect();
    let mut branch_only = BTreeSet::new();
    branch_only.extend(roots.iter().copied());
    branch_only.extend(leaves.iter().copied());
    branch_only.extend(forks.iter().copied());
    for &id in &forks {
        branch_only.extend(doc.nodes[id].children.iter().copied());
    }
    let branch_only_ids: Vec<_> = branch_only.iter().copied().collect();
    let mut branch_with_current = branch_only.clone();
    branch_with_current.insert(current_node);
    let direct_children: Vec<_> = forks
        .iter()
        .map(|&id| json!({"id":id,"children":doc.nodes[id].children}))
        .collect();
    let max_depth = doc.nodes.iter().map(|n| n.depth).max().unwrap_or(0);

    // Replay uses the exact public operation schema expected by the UI loader.
    let mut replay = Document::new();
    for operation in &operations {
        replay.jump(operation.parent).map_err(anyhow::Error::msg)?;
        replay.prompt = operation.prompt.clone();
        replay.settings = operation.settings.clone();
        let id = replay
            .commit_canvas(
                operation.title.clone(),
                OperationSource::User,
                operation.primitives.clone(),
            )
            .map_err(anyhow::Error::msg)?;
        ensure!(
            id == operation.id,
            "Replay node ID mismatch: {id} vs {}",
            operation.id
        );
        ensure!(
            replay.nodes[id].parent == Some(operation.parent),
            "Replay parent mismatch"
        );
        ensure!(
            replay.primitives == operation.primitives,
            "Replay canvas mismatch at {id}"
        );
    }
    replay.jump(current_node).map_err(anyhow::Error::msg)?;
    ensure!(
        replay.primitives == doc.primitives,
        "Replay final selected canvas mismatch"
    );
    ensure!(
        forks.len() >= 4 && leaves.len() >= 5,
        "Fixture needs multiple genuine forks and leaves"
    );
    ensure!(max_depth >= 30, "Fixture needs a long chain");
    let result = json!({
        "schema_version":1,
        "purpose":"editor_demo_not_model_generation",
        "description":"All shapes are hand-authored native primitives committed through the real delta-backed Document. No model, GPU, or network is used.",
        "prompt":PROMPT,
        "operations":operations,
        "current_node":current_node,
        "history_nodes":doc.nodes,
        "expected":{
            "total_nodes":doc.nodes.len(),"max_depth":max_depth,
            "root_ids":roots,"leaf_ids":leaves,"fork_ids":forks,
            "direct_children_of_forks":direct_children,
            "branch_only_definition":"roots union leaves union forks union immediate children of forks",
            "branch_only_ids":branch_only_ids,
            "branch_only_with_current_ids":branch_with_current.into_iter().collect::<Vec<_>>(),
            "long_chain_ids":long_chain_ids,
            "labeled_anchors":{"common_face":common,"cool_leaf":cool_leaf,"nested_parent":nested_parent,
                "nested_leaf":nested_leaf,"deep_leaf":deep_leaf,"deep_other_leaf":deep_other,"root_alternative_leaf":root_leaf},
            "selected_node_is_single_child_intermediate":doc.nodes[current_node].children.len()==1,
            "all_operations_create_exactly_one_nonempty_user_node":true,
            "serialized_operations_replayed_with_identical_ids_and_canvases":true,
            "model_generated":false
        }
    });
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&result)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"output":output,"nodes":result["expected"]["total_nodes"],
        "forks":result["expected"]["fork_ids"],"leaves":result["expected"]["leaf_ids"],
        "branch_only_ids":result["expected"]["branch_only_ids"],"current_node":current_node,
        "replay_verified":true,"model_generated":false})
        )?
    );
    Ok(())
}
