#![cfg(not(test))]
// Standalone real-model tools are executed explicitly. Shared module unit tests
// run in the application frontend, where their full crate context is available.
// Small installer bootstrap. It links no application/CUDA/tensor modules.
#[path="../installer_runtime/mod.rs"]mod installer_runtime;
#[path="../platform_check.rs"]mod platform_check;
#[path="../resources.rs"]mod resources;

use anyhow::{bail,Context,Result};
use serde_json::json;
use std::path::PathBuf;
const MANIFEST:&str=include_str!("../../installer/runtime-manifest.json");

#[derive(Default)]
struct Args{runtime:Option<PathBuf>,cache:Option<PathBuf>,hardware:bool,human:bool,verify:bool,
    running:bool,app_dir:Option<PathBuf>,set_model:Option<PathBuf>,tokenizer:Option<PathBuf>,config_dir:Option<PathBuf>,
    package:Option<String>,force_download:bool}
fn arguments()->Result<Option<Args>>{
    let mut result=Args::default();let mut args=std::env::args().skip(1);
    while let Some(arg)=args.next(){match arg.as_str(){
        "--runtime-dir"=>result.runtime=Some(args.next().context("--runtime-dir 缺少目录")?.into()),
        "--cache-dir"=>result.cache=Some(args.next().context("--cache-dir 缺少目录")?.into()),
        "--prefer-installed"=>{},
        "--verify-only"=>result.verify=true,
        "--hardware-check"=>result.hardware=true,
        "--human"=>result.human=true,
        "--check-running"=>result.running=true,
        "--app-dir"=>result.app_dir=Some(args.next().context("--app-dir 缺少目录")?.into()),
        "--set-model"=>result.set_model=Some(args.next().context("--set-model 缺少权重路径")?.into()),
        "--tokenizer"=>result.tokenizer=Some(args.next().context("--tokenizer 缺少路径")?.into()),
        "--config-dir"=>result.config_dir=Some(args.next().context("--config-dir 缺少目录")?.into()),
        "--package"=>result.package=Some(args.next().context("--package 缺少包 ID")?),
        "--force-download"=>result.force_download=true,
        "--help"|"-h"=>{
            println!("runtime-installer --runtime-dir DIR --cache-dir DIR [--prefer-installed] [--package ID --force-download]\n  --runtime-dir DIR --verify-only\n  --hardware-check [--human]\n  --check-running --app-dir DIR\n  --set-model MODEL --tokenizer TOKENIZER --config-dir DIR\nExit: 0 success; 1 error; 2 unsupported hardware; 10 exact installed application is running.");
            return Ok(None);
        }
        _=>bail!("未知参数：{arg}"),
    }}
    if [result.hardware,result.running,result.set_model.is_some(),result.verify].iter().filter(|&&v|v).count()>1{bail!("安装器辅助模式不能混用");}
    Ok(Some(result))
}
fn run(args:Args)->Result<i32>{
    if args.hardware{
        return match platform_check::probe_nvidia(){
            Ok(report)=>{let exit=if report.compatible{0}else{2};if args.human{println!("{}\n{}",report.device_name,report.reason);}else{println!("{}",serde_json::to_string_pretty(&report)?);}Ok(exit)},
            Err(error)=>{if args.human{println!("{error}");}else{println!("{}",serde_json::to_string_pretty(&json!({"compatible":false,"reason":error}))?);}Ok(2)},
        };
    }
    if args.running{
        let app=args.app_dir.context("--check-running 必须指定 --app-dir")?;
        let processes=installer_runtime::running::check_running(&app)?;let running=!processes.is_empty();
        println!("{}",serde_json::to_string_pretty(&json!({"app_dir":app,"running":running,"processes":processes}))?);
        return Ok(if running{10}else{0});
    }
    if let Some(model)=args.set_model{
        let tokenizer=args.tokenizer.context("--set-model 必须指定 --tokenizer")?;
        let config=args.config_dir.context("--set-model 必须指定 --config-dir")?;
        let saved=resources::save_model_paths_at(&config,&model,&tokenizer).map_err(anyhow::Error::msg)?;
        println!("{}",serde_json::to_string_pretty(&json!({"status":"saved","config":saved}))?);return Ok(0);
    }
    let runtime=args.runtime.context("必须指定 --runtime-dir，或选择一个辅助模式")?;
    let mut manifest=installer_runtime::parse_manifest(MANIFEST)?;
    if let Some(id)=args.package{
        manifest.packages.retain(|p|p.id==id);
        if manifest.packages.is_empty(){bail!("内置清单不存在运行库包：{id}");}
    }
    let result=if args.verify{
        installer_runtime::verify_runtime(&manifest,&runtime)?
    }else{
        let cache=args.cache.context("安装运行库必须显式指定 --cache-dir")?;
        installer_runtime::install_with_options(&manifest,&runtime,&cache,true,args.force_download)?
    };
    println!("{}",serde_json::to_string_pretty(&json!({"status":"complete","runtime":runtime,"verified_files":result.files,"cuda_release":result.cuda_release}))?);
    Ok(0)
}
fn main(){
    let result=arguments().and_then(|args|match args{Some(args)=>run(args),None=>Ok(0)});
    let exit=match result{Ok(exit)=>exit,Err(error)=>{eprintln!("安装器辅助程序失败：{error:#}");1}};
    std::process::exit(exit);
}
