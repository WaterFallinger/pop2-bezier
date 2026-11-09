#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
// Headless, no-model acceptance for a real saved tree. Input is read-only;
// all output goes to a new, explicitly chosen file and report.
#![allow(dead_code, unused_imports)]
#[path = "../acceleration.rs"] mod acceleration;
#[path = "../atomic_file.rs"] mod atomic_file;
#[path = "../document.rs"] mod document;
#[path = "../history_context.rs"] mod history_context;
#[path = "../mode_transition.rs"] mod mode_transition;
#[path = "../project_file.rs"] mod project_file;
#[path = "../region_state.rs"] mod region_state;

use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}, time::Instant};

fn parse() -> Result<Option<(PathBuf, PathBuf, PathBuf)>> {
    let mut args = std::env::args().skip(1);
    let (mut input, mut copy, mut report) = (None, None, None);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = Some(PathBuf::from(args.next().context("--input requires a file")?)),
            "--copy" => copy = Some(PathBuf::from(args.next().context("--copy requires a new file")?)),
            "--report" => report = Some(PathBuf::from(args.next().context("--report requires a new file")?)),
            "--help" | "-h" => {
                println!("project-roundtrip --input ORIGINAL.pproj --copy NEW.pproj --report NEW.json\nLoads, saves to a new copy, reloads and compares the complete serialized document and history. No GPU/model load. Original input is never written. Workspace mode is reported from sibling metadata when present; it is not part of a standalone .pproj file.");
                return Ok(None);
            },
            _ => bail!("Unknown argument: {arg}"),
        }
    }
    Ok(Some((input.context("--input required")?, copy.context("--copy required")?, report.context("--report required")?)))
}

fn absolute_new(path: &Path) -> Result<PathBuf> {
    ensure!(!path.exists(), "Output already exists; refusing to replace {}", path.display());
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent).context("Output parent must already exist")?;
    Ok(parent.join(path.file_name().context("Output must have a filename")?))
}

fn run(input: &Path, copy: &Path) -> Result<Value> {
    let started = Instant::now();
    let before = fs::read(input).context("Reading original bytes")?;
    let load_started = Instant::now();
    let (doc, history) = project_file::load_project(input).map_err(anyhow::Error::msg).context("Original project load failed")?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.;
    let expected_document = serde_json::to_vec(&doc.project_ref())?;
    let expected_history = serde_json::to_vec(&history.project_ref())?;
    let metadata = input.parent().map(|p| p.join("metadata.json")).filter(|p|p.exists())
        .map(fs::read).transpose()?.map(|bytes|serde_json::from_slice::<Value>(&bytes)).transpose()?;
    let mode = metadata.as_ref().and_then(|m|m.pointer("/workspace/mode")).cloned();
    let save_started = Instant::now();
    project_file::save_project(copy, &doc, &history).map_err(anyhow::Error::msg).context("Copy save failed")?;
    let save_ms = save_started.elapsed().as_secs_f64() * 1000.;
    let reload_started = Instant::now();
    let (loaded, raw) = project_file::load_project(copy).map_err(anyhow::Error::msg).context("Copy reload failed")?;
    let reload_ms = reload_started.elapsed().as_secs_f64() * 1000.;
    ensure!(serde_json::to_vec(&loaded.project_ref())? == expected_document, "Entire serialized document changed during roundtrip");
    ensure!(serde_json::to_vec(&raw.project_ref())? == expected_history, "Entire serialized model history/evidence changed during roundtrip");
    ensure!(fs::read(input)? == before, "Original changed during read-only acceptance");
    Ok(json!({
        "status":"passed", "input":input, "copy":copy,
        "nodes":doc.nodes.len(), "current_node":doc.current, "visible_primitives":doc.primitives.len(),
        "patches":doc.stored_patch_count(), "input_bytes":before.len(), "copy_bytes":fs::metadata(copy)?.len(),
        "load_ms":load_ms, "save_ms":save_ms, "reload_ms":reload_ms,
        "total_ms":started.elapsed().as_secs_f64()*1000.,
        "entire_document_equal":true, "entire_history_and_mode_evidence_equal":true,
        "original_bytes_unchanged":true, "gpu_model_loads":0,
        "source_library_workspace_mode":mode,
        "workspace_note":"Standalone project files do not store library workspace mode. This tool reports, and does not modify, source metadata."
    }))
}

fn main() -> Result<()> {
    let Some((input, copy, report)) = parse()? else {return Ok(());};
    let input = fs::canonicalize(input).context("Input project must exist")?;
    let copy = absolute_new(&copy)?;
    let report = absolute_new(&report)?;
    ensure!(input != copy && input != report && copy != report, "Input, copy and report paths must be distinct");
    let result = run(&input, &copy);
    let value = match &result {
        Ok(value) => value.clone(),
        Err(error) => json!({"status":"failed", "input":input, "copy":copy, "error":format!("{error:#}"), "gpu_model_loads":0}),
    };
    let bytes = serde_json::to_vec_pretty(&value)?;
    // create_new prevents an invocation from accidentally replacing evidence.
    use std::io::Write;
    fs::OpenOptions::new().write(true).create_new(true).open(&report)?.write_all(&bytes)?;
    println!("{}", String::from_utf8_lossy(&bytes));
    result.map(|_|())
}
