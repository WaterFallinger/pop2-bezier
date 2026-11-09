#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
// Installs only pinned native DLLs/GPU data; never imports or invokes Python.
#[path = "../aten_runtime_install.rs"]
mod install;
use anyhow::{bail, Context, Result};
use std::path::PathBuf;

fn run() -> Result<()> {
    let (mut manifest, mut runtime, mut cache, mut site) = (None, None, None, None);
    let (mut verify, mut dry, mut keep) = (false, false, false);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--manifest" => {
                manifest = Some(PathBuf::from(args.next().context("缺少运行库清单路径")?))
            }
            "--runtime-dir" => {
                runtime = Some(PathBuf::from(args.next().context("缺少运行库目录")?))
            }
            "--cache-dir" => cache = Some(PathBuf::from(args.next().context("缺少缓存目录")?)),
            "--source-site" => {
                site = Some(PathBuf::from(args.next().context("缺少离线原生文件来源")?))
            }
            "--verify-only" => verify = true,
            "--dry-run" => dry = true,
            "--keep-downloads" => keep = true,
            "--help" => {
                println!("aten-runtime-installer --manifest FILE --runtime-dir DIR [--cache-dir DIR | --source-site DIR] [--verify-only | --dry-run] [--keep-downloads]");
                return Ok(());
            }
            _ => bail!("未知参数：{arg}"),
        }
    }
    let text = std::fs::read_to_string(manifest.context("需要 --manifest")?)?;
    let parsed = install::parse_manifest(&text)?;
    if dry {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"status":"dry_run","runtime_id":parsed.runtime_id,"files":parsed.total_native_files,"native_bytes":parsed.total_native_bytes,"download_bytes":parsed.total_download_bytes,"python_required":false})
            )?
        );
        return Ok(());
    }
    let result = install::install(
        &text,
        &runtime.context("需要 --runtime-dir")?,
        cache.as_deref(),
        site.as_deref(),
        verify,
        keep,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("AMD原生运行组件准备失败：{error:#}");
        std::process::exit(1)
    }
}
