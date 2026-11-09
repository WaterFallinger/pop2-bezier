#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
#[path = "../aten_weight_index.rs"]
mod aten_weight_index;
use anyhow::{Context, Result};
use std::path::PathBuf;
fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model = PathBuf::from(args.next().context("需要原始 model.pt 路径")?);
    let destination = PathBuf::from(args.next().context("需要新的 index.ppateni 路径")?);
    if args.next().is_some() { anyhow::bail!("仅接受权重与目录两个路径"); }
    let started = std::time::Instant::now();
    let count = aten_weight_index::write_weight_index(&model, &destination)?;
    println!("{}", serde_json::json!({"status":"complete","tensor_count":count,
        "source_bytes":std::fs::metadata(&model)?.len(),"index_bytes":std::fs::metadata(&destination)?.len(),
        "index_path":destination,"seconds":started.elapsed().as_secs_f64(),
        "weights_copied":false,"gpu_used":false,"python_used":false,"file_hashes":false}));
    Ok(())
}
