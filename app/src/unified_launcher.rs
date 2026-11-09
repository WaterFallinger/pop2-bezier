//! One public entry point, isolated native providers, and an explicit CPU fallback.
use crate::{diagnostics, hardware_inventory as inventory, platform_check};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{ffi::OsString, path::{Path, PathBuf}, process::Command};

#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize,Deserialize)]
#[serde(rename_all="snake_case")]
pub enum Provider { Nvidia, Amd, Cpu }
impl Provider { fn id(self)->&'static str { match self{Self::Nvidia=>"nvidia",Self::Amd=>"amd",Self::Cpu=>"cpu"} } }

#[derive(Clone,Debug,Serialize)]
pub struct Selection {
    pub provider:Provider,
    pub cuda_flavor:String,
    pub reason:String,
    pub cpu_fallback:bool,
}
#[derive(Default)]
struct Args { preferred:Option<Provider>, prepare:bool, diagnose:bool, report:Option<PathBuf>,
    no_download:bool, no_cpu_fallback:bool, forwarded:Vec<OsString> }

fn arguments()->Result<Args>{
    let mut result=Args::default();let mut args=std::env::args_os().skip(1);
    while let Some(arg)=args.next(){match arg.to_str(){
        Some("--backend")=>{let v=args.next().context("--backend requires auto/nvidia/amd/cpu")?;result.preferred=match v.to_str(){
            Some("auto")=>None,Some("nvidia"|"cuda")=>Some(Provider::Nvidia),Some("amd"|"hip")=>Some(Provider::Amd),Some("cpu")=>Some(Provider::Cpu),_=>bail!("Unknown compute backend")};},
        Some("--cuda-device")|Some("--hip-device")=>{
            let key=if arg=="--cuda-device"{"PAINTER_CUDA_DEVICE"}else{"PAINTER_HIP_DEVICE"};
            let value=args.next().context("Device ordinal is missing")?;
            let ordinal=value.to_str().context("Device ordinal is invalid text")?.parse::<u32>().context("Device ordinal must be non-negative")?;
            std::env::set_var(key,ordinal.to_string());
        },
        Some("--prepare-runtime")=>result.prepare=true,
        Some("--diagnose")=>result.diagnose=true,
        Some("--report")=>result.report=Some(args.next().context("--report requires a destination")?.into()),
        Some("--no-runtime-download")=>result.no_download=true,
        Some("--no-cpu-fallback")=>result.no_cpu_fallback=true,
        _=>result.forwarded.push(arg),
    }}
    Ok(result)
}

pub fn select(preferred:Option<Provider>,adapters:&[inventory::Adapter],nvidia:Option<&platform_check::HardwareReport>)->Selection{
    if preferred==Some(Provider::Cpu){return Selection{provider:Provider::Cpu,cuda_flavor:String::new(),reason:"用户明确选择CPU兼容计算".into(),cpu_fallback:false}}
    let nv=nvidia.filter(|r|r.compatible);
    // The distributed 1B model needs more than its 2GB weights alone. Avoid
    // downloading a large HIP runtime just for a small display-only iGPU;
    // explicit --backend amd still permits its real capability test.
    let amd=adapters.iter().any(|a|a.vendor_id==0x1002&&!a.software&&a.dedicated_bytes>=2*1024*1024*1024);
    let selected=match preferred {
        Some(Provider::Nvidia)=>nv.map(|_|Provider::Nvidia),
        Some(Provider::Amd)=>Some(Provider::Amd),
        _=>match (nv.is_some(),amd){
            (true,true)=>Some(if inventory::dedicated_for(adapters,0x1002)>nv.and_then(|r|r.total_memory_bytes).unwrap_or_else(||inventory::dedicated_for(adapters,0x10de)){Provider::Amd}else{Provider::Nvidia}),
            (true,false)=>Some(Provider::Nvidia),(false,true)=>Some(Provider::Amd),_=>None,
        }
    };
    let provider=selected.unwrap_or(Provider::Cpu);
    let flavor=if nv.is_some_and(|r|(r.compute_major,r.compute_minor)==(12,0)){"optimized"}else{"portable"};
    let reason=match provider{
        Provider::Nvidia=>format!("CUDA基础架构与驱动检查通过；选择{flavor}执行组件，模型加载后再验证可选加速"),
        Provider::Amd=>"检测到AMD适配器；交由HIP实际设备枚举与模型数值检查确认计算能力".into(),
        Provider::Cpu=>format!("没有满足当前GPU基础要求的候选设备，使用CPU兼容计算（速度较慢）。Auto不会仅为低于2GiB专用显存的AMD显示适配器预装大型HIP组件；可以显式指定AMD做能力验证。NVIDIA检查：{}",nvidia.map(|r|r.reason.as_str()).unwrap_or("驱动查询不可用")),
    };
    Selection{provider,cuda_flavor:if provider==Provider::Nvidia{flavor.into()}else{String::new()},reason,cpu_fallback:provider==Provider::Cpu}
}

fn application(root:&Path,selection:&Selection)->PathBuf{
    let directory=root.join("runtime").join(format!("app-{}",env!("CARGO_PKG_VERSION")));
    let file=match selection.provider{
        Provider::Nvidia=>if selection.cuda_flavor=="optimized"{"primitive-painter-cuda-optimized.exe"}else{"primitive-painter-cuda-portable.exe"},
        Provider::Amd=>"primitive-painter-aten-hip.exe",Provider::Cpu=>"primitive-painter-cpu.exe",
    };
    directory.join(file)
}

fn cuda_directory(root:&Path)->PathBuf {
    let directory=root.join("runtime/cuda");
    if directory.join("cublas64_13.dll").is_file(){directory}else{root.join("runtime")}
}
fn runtime_ready(root:&Path,provider:Provider)->Result<()> {
    let required:Vec<PathBuf>=match provider{
        Provider::Nvidia=>["cublas64_13.dll","cublasLt64_13.dll","curand64_10.dll"].iter().map(|name|cuda_directory(root).join(name)).collect(),
        Provider::Amd=>["torch/torch_cpu.dll","torch/torch_hip.dll","torch/c10.dll","torch/c10_hip.dll","core/amdhip64_7.dll","libraries/rocblas.dll"].iter().map(|name|root.join("runtime/aten").join(name)).collect(),
        Provider::Cpu=>Vec::new(),
    };
    for path in required{if !path.is_file(){bail!("缺少所选计算后端的基础运行组件：{}",path.display())}}
    if provider==Provider::Amd&&!root.join("runtime").join(format!("app-{}",env!("CARGO_PKG_VERSION"))).join("painter_aten_hip.dll").is_file(){bail!("缺少AMD原生推理组件")}
    Ok(())
}

fn prepare_runtime(root:&Path,provider:Provider)->Result<()> {
    if provider==Provider::Cpu{return Ok(())}
    let mut command=match provider{
        Provider::Nvidia=>{
            let mut c=Command::new(root.join("runtime/runtime-installer.exe"));
            c.args(["--runtime-dir"]).arg(cuda_directory(root)).arg("--cache-dir").arg(root.join("cache/cuda-downloads")).arg("--prefer-installed");c
        },
        Provider::Amd=>{
            let mut c=Command::new(root.join("runtime/aten-runtime-installer.exe"));
            c.arg("--manifest").arg(root.join("native-runtime-manifest.json"))
                .arg("--runtime-dir").arg(root.join("runtime/aten"))
                .arg("--cache-dir").arg(root.join("cache/aten-downloads"));c
        },Provider::Cpu=>unreachable!(),
    };
    #[cfg(windows)]{use std::os::windows::process::CommandExt;command.creation_flags(0x08000000);}
    diagnostics::event("runtime_prepare_started",json!({"provider":provider,"source":"pinned_official_native_components"}));
    let output=command.output().context("无法启动运行组件准备工具")?;
    diagnostics::event("runtime_prepare_finished",json!({"provider":provider,"exit_code":output.status.code(),
        "stdout":String::from_utf8_lossy(&output.stdout),"stderr":String::from_utf8_lossy(&output.stderr)}));
    if !output.status.success(){bail!("GPU运行组件准备失败，详细原因已保存到本次日志")}
    runtime_ready(root,provider)
}

fn selected_environment(root:&Path,selection:&Selection)->Result<Vec<(String,OsString)>> {
    let mut result=diagnostics::child_environment();
    if is_test_installation(root){
        let profile=root.join("qa-profile");
        std::fs::create_dir_all(&profile)?;
        result.push(("LOCALAPPDATA".into(),profile.clone().into_os_string()));
        result.push(("APPDATA".into(),profile.join("Roaming").into_os_string()));
    }
    result.push(("PAINTER_COMPUTE_PROVIDER".into(),selection.provider.id().into()));
    result.push(("PAINTER_SELECTION_REASON".into(),selection.reason.clone().into()));
    result.push(("PAINTER_INSTALL_ROOT".into(),root.as_os_str().to_owned()));
    result.push(("PAINTER_ALLOW_CPU_FALLBACK".into(),"1".into()));
    let mut dirs=Vec::new();
    match selection.provider{
        Provider::Nvidia=>dirs.push(cuda_directory(root)),
        Provider::Amd=>{
            dirs.extend([root.join("runtime/aten/torch"),root.join("runtime/aten/core"),root.join("runtime/aten/libraries")]);
            result.push(("PAINTER_ATEN_RUNTIME_DIRS".into(),std::env::join_paths(&dirs)?));
            result.push(("PAINTER_ATEN_RUNTIME_ID".into(),"aten-hip-torch2.9.0-rocm7.1.1-win-amd64".into()));
            result.push(("PAINTER_ATEN_LIBRARY".into(),application(root,selection).with_file_name("painter_aten_hip.dll").into_os_string()));
            result.push(("PAINTER_ATEN_CACHE_DIR".into(),root.join("cache/aten").into_os_string()));
            result.push(("ATEN_ROCM_ALLOW_EXPERIMENTAL_AOTRITON".into(),"1".into()));
            result.push(("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL".into(),"1".into()));
        },Provider::Cpu=>{},
    }
    dirs.extend([application(root,selection).parent().unwrap().to_owned(),root.to_owned()]);
    dirs.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()));
    result.push(("PATH".into(),std::env::join_paths(dirs)?));
    Ok(result)
}

fn is_test_installation(root:&Path)->bool {
    let path=root.join(".primitive-painter-install");
    if !std::fs::metadata(&path).is_ok_and(|m|m.is_file()&&m.len()<1024){return false}
    std::fs::read_to_string(path).ok().is_some_and(|text|{
        let mut lines=text.lines();lines.next()==Some("PrimitivePainter.Installation.v1")&&lines.next()==Some("1")
    })
}

fn write_report(path:Option<&Path>,value:&Value)->Result<()> {
    if let Some(path)=path{
        if !path.is_absolute(){bail!("诊断报告目标必须是绝对路径")}
        if let Some(parent)=path.parent(){std::fs::create_dir_all(parent)?}
        std::fs::write(path,serde_json::to_vec_pretty(value)?)?;
    }
    Ok(())
}

pub fn run()->Result<()> {
    let log=diagnostics::init("launcher");
    let args=arguments()?;
    let executable=std::env::current_exe()?;
    let root=executable.parent().context("无法定位应用目录")?;
    let adapters=match inventory::enumerate(){Ok(a)=>a,Err(error)=>{log.event("inventory_unavailable",json!({"reason":error}));Vec::new()}};
    log.event("display_inventory",json!({"adapters":adapters,"compute_support_verified":false}));
    let nvidia=match platform_check::probe_nvidia_for_profile(platform_check::CudaKernelProfile::PortableSm80){
        Ok(report)=>Some(report),Err(reason)=>{log.event("cuda_driver_probe_unavailable",json!({"reason":reason}));None},
    };
    log.event("cuda_admission",json!({"report":nvidia}));
    let mut selection=select(args.preferred,&adapters,nvidia.as_ref());
    log.event("provider_candidate",json!({"selection":selection,"policy":"explicit preference; otherwise supported discrete candidates ranked by dedicated VRAM, NVIDIA on tie; actual model provider validates separately"}));
    if args.diagnose{
        let report=json!({"version":env!("CARGO_PKG_VERSION"),"adapters":adapters,"cuda":nvidia,"selection":selection,"model_compute_tested":false,"log_directory":log.directory()});
        write_report(args.report.as_deref(),&report)?;println!("{}",serde_json::to_string_pretty(&report)?);log.flush_summary();return Ok(())
    }
    if selection.provider==Provider::Cpu&&selection.cpu_fallback&&args.no_cpu_fallback{bail!("没有可用GPU候选，已按要求停止自动回退：{}",selection.reason)}
    if selection.provider!=Provider::Cpu{
        let mut candidates=vec![selection.clone()];
        if args.preferred.is_none(){
            let alternate=match selection.provider{Provider::Nvidia if adapters.iter().any(|a|a.vendor_id==0x1002&&!a.software&&a.dedicated_bytes>=2*1024*1024*1024)=>Some(Provider::Amd),Provider::Amd if nvidia.as_ref().is_some_and(|r|r.compatible)=>Some(Provider::Nvidia),_=>None};
            if let Some(provider)=alternate{candidates.push(select(Some(provider),&adapters,nvidia.as_ref()));}
        }
        let mut chosen=None;let mut last_error=None;
        for mut candidate in candidates{
            if args.prepare&&!args.no_download&&candidate.provider==selection.provider{
                if let Err(error)=prepare_runtime(root,candidate.provider){
                    log.error("runtime_prepare",&error);
                    // Basic GPU math may still be available when only an
                    // optional compiler component could not be downloaded.
                }
            }
            if candidate.provider==Provider::Nvidia&&candidate.cuda_flavor=="optimized"&&!application(root,&candidate).is_file(){
                candidate.cuda_flavor="portable".into();candidate.reason.push_str("；独立优化组件缺失，保留同GPU并使用通用PTX组件");
                log.event("cuda_worker_fallback",json!({"reason":"Architecture-optimized worker absent; retain same GPU using portable PTX"}));
            }
            let valid=(||->Result<()>{
                if !application(root,&candidate).is_file(){bail!("缺少所选后端的应用组件")}
                runtime_ready(root,candidate.provider)
            })();
            match valid{
                Ok(())=>{if candidate.provider!=selection.provider{log.event("alternate_gpu_selected",json!({"from":selection.provider,"to":candidate.provider}));}chosen=Some(candidate);break},
                Err(error)=>{log.error("provider_preflight",&error);last_error=Some(error)},
            }
        }
        selection=if let Some(candidate)=chosen{candidate}else{cpu_fallback(selection,&last_error.unwrap_or_else(||anyhow::anyhow!("No usable provider")),args.no_cpu_fallback)?};
    }
    let application=application(root,&selection);
    if !application.is_file(){bail!("缺少兼容应用组件 {}，请重新运行安装程序；诊断目录：{:?}",application.display(),log.directory())}
    let cuda_ordinal=if selection.provider==Provider::Nvidia{nvidia.as_ref().map(|r|r.cuda_ordinal)}else{None};
    log.event("provider_selected",json!({"selection":selection,"application":application,"actual_cuda_ordinal":cuda_ordinal}));
    if args.prepare{
        let report=json!({"status":"ready","selection":selection,"log_directory":log.directory()});
        write_report(args.report.as_deref(),&report)?;println!("{}",serde_json::to_string_pretty(&report)?);log.flush_summary();return Ok(())
    }
    let mut command=Command::new(&application);
    command.current_dir(root).envs(selected_environment(root,&selection)?).args(args.forwarded);
    if let Some(ordinal)=cuda_ordinal{command.env("PAINTER_CUDA_DEVICE",ordinal.to_string());}
    if args.no_cpu_fallback{command.env("PAINTER_ALLOW_CPU_FALLBACK","0");}
    let child=command.spawn().context("无法启动所选原生应用组件")?;
    log.event("frontend_started",json!({"pid":child.id(),"selection":selection}));
    log.flush_summary();Ok(())
}

fn cpu_fallback(previous:Selection,error:&anyhow::Error,strict:bool)->Result<Selection>{
    if strict{bail!("所选GPU后端不可用，已按要求停止自动回退：{error:#}")}
    Ok(Selection{provider:Provider::Cpu,cuda_flavor:String::new(),cpu_fallback:true,
        reason:format!("{}组件暂不可用，已启用CPU兼容计算（速度较慢）：{error:#}。可以重新运行安装程序准备GPU组件，并查看设置中的诊断日志。",previous.provider.id())})
}

#[cfg(test)]mod tests{
    use super::*;
    fn adapter(vendor:u32,bytes:u64)->inventory::Adapter{inventory::Adapter{name:"fixture".into(),vendor_id:vendor,device_id:1,dedicated_bytes:bytes,shared_bytes:0,luid:"fixture".into(),software:false}}
    #[test]fn no_hardware_keeps_a_visible_cpu_compatible_candidate(){let s=select(None,&[],None);assert_eq!(s.provider,Provider::Cpu);assert!(s.cpu_fallback);assert!(s.reason.contains("CPU"));}
    #[test]fn amd_inventory_does_not_claim_math_has_already_run(){let s=select(None,&[adapter(0x1002,24<<30)],None);assert_eq!(s.provider,Provider::Amd);assert!(s.reason.contains("数值检查"));}
    #[test]fn explicit_cpu_never_gets_overridden_by_gpu_inventory(){let s=select(Some(Provider::Cpu),&[adapter(0x1002,24<<30)],None);assert_eq!(s.provider,Provider::Cpu);assert!(!s.cpu_fallback);}
    #[test]fn a_small_display_igpu_does_not_trigger_large_automatic_hip_downloads(){let adapters=[adapter(0x1002,512<<20)];assert_eq!(select(None,&adapters,None).provider,Provider::Cpu);assert_eq!(select(Some(Provider::Amd),&adapters,None).provider,Provider::Amd);}
    #[test]fn only_same_version_payload_is_selected(){let s=select(Some(Provider::Cpu),&[],None);assert!(application(Path::new("C:/owned"),&s).ends_with(format!("runtime/app-{}/primitive-painter-cpu.exe",env!("CARGO_PKG_VERSION"))));}
    #[test]fn strict_debugging_can_refuse_automatic_cpu_fallback(){let s=select(None,&[adapter(0x1002,1)],None);assert!(cpu_fallback(s,&anyhow::anyhow!("fixture"),true).is_err());}
    #[test]fn test_installations_are_identified_only_by_their_explicit_owned_scope(){
        let root=std::env::temp_dir().join(format!("painter-scope-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&root).unwrap();assert!(!is_test_installation(&root));
        std::fs::write(root.join(".primitive-painter-install"),"PrimitivePainter.Installation.v1\r\n1\r\n").unwrap();assert!(is_test_installation(&root));
        std::fs::write(root.join(".primitive-painter-install"),"PrimitivePainter.Installation.v1\r\n0\r\n").unwrap();assert!(!is_test_installation(&root));
        std::fs::remove_file(root.join(".primitive-painter-install")).unwrap();std::fs::remove_dir(root).unwrap();
    }
}
