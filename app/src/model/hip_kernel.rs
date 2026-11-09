//! Optional HIP code-object preparation and same-device dispatch.
//! Only a compiled ROCm provider may use this module; DLL presence is not a capability.

use candle_core::rocm_backend::{hip_runtime::{module::HipModule,toolchain},hip_sys::hip_runtime as sys,RocmDevice};
use std::{collections::HashMap,ffi::c_void,path::PathBuf,process::Command,sync::{Arc,Mutex,OnceLock}};
use std::sync::atomic::{AtomicU64,Ordering};

static ATTENTION_PARTITIONS:AtomicU64=AtomicU64::new(0);
static ATTENTION_MERGES:AtomicU64=AtomicU64::new(0);
static RENYI_SAMPLES:AtomicU64=AtomicU64::new(0);
static OTHER_LAUNCHES:AtomicU64=AtomicU64::new(0);
static UNCERTAIN_FAILURES:AtomicU64=AtomicU64::new(0);
#[derive(Clone,Copy,Debug,serde::Serialize)]
pub(crate) struct KernelLaunchCounts{
    pub attention_partitions:u64,pub attention_merges:u64,pub renyi_samples:u64,pub other:u64,pub uncertain_failures:u64,
}
/// Process-wide successful enqueue counters. Read a delta around a serialized
/// diagnostic; this does not replace completion/finite checks or time the GPU.
pub(crate) fn launch_counts()->KernelLaunchCounts{
    KernelLaunchCounts{attention_partitions:ATTENTION_PARTITIONS.load(Ordering::Relaxed),attention_merges:ATTENTION_MERGES.load(Ordering::Relaxed),renyi_samples:RENYI_SAMPLES.load(Ordering::Relaxed),other:OTHER_LAUNCHES.load(Ordering::Relaxed),uncertain_failures:UNCERTAIN_FAILURES.load(Ordering::Relaxed)}
}

#[derive(Clone,Debug,serde::Serialize)]
pub(crate) struct KernelPreparationEvidence {
    pub kernel_name:String,
    pub compiler_path:PathBuf,
    pub compiler_version:String,
    pub runtime_version:i32,
    pub driver_version:i32,
    pub gfx:String,
    pub pci_bus_id:String,
    pub device_name:String,
    pub kernel_sdk_version:String,
    pub compile_flags:Vec<String>,
    pub code_object:PathBuf,
    pub source_fingerprint:String,
    pub cache_fingerprint:String,
    pub compiled_this_process:bool,
    pub compiled_successfully_this_process:bool,
    pub code_object_cache_hit:bool,
    pub module_loaded:bool,
}
struct LoadedModule{module:Arc<HipModule>,evidence:KernelPreparationEvidence}
type ModuleCache=HashMap<(usize,String,usize,usize),LoadedModule>;
static MODULES:OnceLock<Mutex<ModuleCache>>=OnceLock::new();
const FLAGS:&[&str]=&["--genco","-O3","-std=c++17","-ffp-contract=off"];

#[link(name="amdhip64")]
extern "C" { fn hipDeviceGetPCIBusId(output:*mut std::ffi::c_char,length:i32,device:i32)->i32; }

fn compiler_command(path:&std::path::Path)->Command {
    let mut command=Command::new(path);
    #[cfg(windows)] {use std::os::windows::process::CommandExt;command.creation_flags(0x0800_0000);}
    command
}

fn cache_identity(device:&RocmDevice,compiler:&std::path::Path)->candle_core::Result<(String,KernelPreparationEvidence)> {
    let(mut runtime,mut driver)=(0i32,0i32);
    let mut pci=[0i8;64];
    let status=unsafe{(sys::hipRuntimeGetVersion(&mut runtime),sys::hipDriverGetVersion(&mut driver),hipDeviceGetPCIBusId(pci.as_mut_ptr(),pci.len() as i32,device.ordinal() as i32))};
    if status!=(0,0,0){return Err(preparation(format!("cannot establish HIP runtime/driver/PCI identity: {status:?}")));}
    let pci=unsafe{std::ffi::CStr::from_ptr(pci.as_ptr())}.to_string_lossy().into_owned();
    let version=compiler_command(compiler).arg("--version").output().map_err(preparation)?;
    if !version.status.success(){return Err(preparation("existing HIP compiler version query failed"));}
    let metadata=std::fs::metadata(compiler).map_err(preparation)?;
    let changed=metadata.modified().ok().and_then(|time|time.duration_since(std::time::UNIX_EPOCH).ok()).map(|time|time.as_nanos());
    let device_name=device.name().map_err(preparation)?;
    let compiler_version=format!("{}{}",String::from_utf8_lossy(&version.stdout),String::from_utf8_lossy(&version.stderr));
    let sdk=candle_core::rocm_backend::kernels::BUILD_HIP_VERSION;
    let mut identity=format!("pci={pci}\nordinal={}\nname={device_name}\nbytes={}\nruntime={runtime}\ndriver={driver}\ncompiler={}\ncompiler_len={}\ncompiler_modified={changed:?}\ncompiler_version={compiler_version}\nflags={FLAGS:?}\nframework_build={sdk}\n",device.ordinal(),device.total_memory().map_err(preparation)?,compiler.display(),metadata.len());
    for name in ["HIP_PATH","ROCM_PATH","HIPCC_COMPILE_FLAGS_APPEND","HIPCC_LINK_FLAGS_APPEND","HIP_PLATFORM","HIP_COMPILER","HIP_DEVICE_LIB_PATH"] {
        identity.push_str(&format!("{name}={:?}\n",std::env::var_os(name)));
    }
    let gfx=device.arch().ok_or_else(||preparation("HIP GPU architecture unavailable"))?.name.clone();
    let evidence=KernelPreparationEvidence{kernel_name:String::new(),compiler_path:compiler.to_owned(),compiler_version,runtime_version:runtime,driver_version:driver,gfx,pci_bus_id:pci,device_name,kernel_sdk_version:sdk.into(),compile_flags:FLAGS.iter().map(|flag|flag.to_string()).collect(),code_object:PathBuf::new(),source_fingerprint:String::new(),cache_fingerprint:String::new(),compiled_this_process:false,compiled_successfully_this_process:false,code_object_cache_hit:false,module_loaded:false};
    Ok((identity,evidence))
}

fn preparation(message:impl std::fmt::Display)->candle_core::Error {
    candle_core::Error::Msg(format!("HIP_PREPARATION: {message}"))
}
fn source_key(source:&str)->u64 {
    source.as_bytes().iter().fold(0xcbf29ce484222325u64,|value,byte|(value^(*byte as u64)).wrapping_mul(0x100000001b3))
}
fn cache_root()->candle_core::Result<PathBuf> {
    let root=if let Some(path)=std::env::var_os("PAINTER_HIP_CACHE_DIR") {
        let path=PathBuf::from(path);
        if !path.is_absolute(){return Err(preparation("PAINTER_HIP_CACHE_DIR must be absolute"));}
        path
    }else if let Some(path)=std::env::var_os("LOCALAPPDATA") {
        PathBuf::from(path).join("PrimitivePainter").join("Cache").join("hip")
    }else{std::env::temp_dir().join("primitive-painter-hip")};
    std::fs::create_dir_all(&root).map_err(preparation)?;
    Ok(root)
}

/// Preparation only: compilation/module loading failures may disable the optional
/// feature before any user-cache kernel is enqueued. Never substitutes CPU math.
pub(crate) fn module(device:&RocmDevice,stem:&'static str,source:&'static str)->candle_core::Result<Arc<HipModule>> {
    if stem.is_empty()||!stem.bytes().all(|byte|byte.is_ascii_alphanumeric()||byte==b'_'||byte==b'-') {
        return Err(preparation("invalid kernel cache stem"));
    }
    let arch=device.arch().ok_or_else(||preparation("HIP did not report a GPU architecture"))?.name.clone();
    if !arch.starts_with("gfx")||!arch.bytes().all(|byte|byte.is_ascii_alphanumeric()) {
        return Err(preparation("unsupported HIP architecture identifier"));
    }
    let mut cache=MODULES.get_or_init(||Mutex::new(HashMap::new())).lock().map_err(|_|preparation("HIP module cache poisoned"))?;
    // Static source identity avoids re-hashing or invoking the compiler on a
    // hot decode. Device/driver/compiler fingerprints govern the disk cache;
    // an active HIP device context does not change runtime within this process.
    let cache_key=(device.ordinal(),arch.clone(),source.as_ptr() as usize,source.len());
    if let Some(loaded)=cache.get(&cache_key){return Ok(loaded.module.clone());}
    device.set_current().map_err(preparation)?;
    let compiler=toolchain::hipcc_path().ok_or_else(||preparation("no existing HIP compiler found; optional native kernels are unavailable"))?;
    let(identity,mut evidence)=cache_identity(device,&compiler)?;
    let key=source_key(&format!("{identity}\narch={arch}\nsource={source}"));
    let root=cache_root()?;
    let basename=format!("{stem}_{arch}_{key:016x}");
    let code=root.join(format!("{basename}.co"));
    evidence.kernel_name=stem.into();evidence.code_object=code.clone();evidence.source_fingerprint=format!("{:016x}",source_key(source));evidence.cache_fingerprint=format!("{key:016x}");
    evidence.compile_flags.push(format!("--offload-arch={arch}"));
    evidence.code_object_cache_hit=code.is_file();
    if !code.is_file() {
        let input=root.join(format!("{basename}.hip"));
        std::fs::write(&input,source).map_err(preparation)?;
        let temporary=root.join(format!("{basename}_{}.tmp.co",std::process::id()));
        let mut command=compiler_command(&compiler);
        command.args(FLAGS)
            .arg(format!("--offload-arch={arch}")).arg("-o").arg(&temporary).arg(&input);
        let result=command.output().map_err(preparation)?;
        if !result.status.success() {
            let stderr=String::from_utf8_lossy(&result.stderr);
            let detail=stderr.chars().rev().take(6000).collect::<String>().chars().rev().collect::<String>();
            return Err(preparation(format!("native HIP kernel compilation failed ({}): {detail}",result.status)));
        }
        if !temporary.is_file(){return Err(preparation("HIP compiler did not produce a code object"));}
        evidence.compiled_this_process=true;evidence.compiled_successfully_this_process=true;
        if let Err(error)=std::fs::rename(&temporary,&code) {
            // Another process may have finished the same immutable source. Only
            // our own compiler temporary is eligible for removal here.
            if code.is_file(){let _=std::fs::remove_file(&temporary);}else{return Err(preparation(error));}
        }
        std::fs::write(root.join(format!("{basename}.meta.txt")),&identity).map_err(preparation)?;
    }
    let module=Arc::new(HipModule::load(&code).map_err(preparation)?);
    evidence.module_loaded=true;
    cache.insert(cache_key,LoadedModule{module:module.clone(),evidence});
    Ok(module)
}

/// Successful preparation records only, not a claim that a numerical kernel
/// probe ran. Callers must separately validate execution/precision before
/// advertising a feature. A disk-cache hit is not called a new compilation.
pub(crate) fn preparation_evidence(device:&RocmDevice)->candle_core::Result<Vec<KernelPreparationEvidence>> {
    let cache=MODULES.get_or_init(||Mutex::new(HashMap::new())).lock().map_err(|_|preparation("HIP module cache poisoned"))?;
    let mut evidence=cache.iter().filter(|((ordinal,_,_,_),_)|*ordinal==device.ordinal()).map(|(_,loaded)|loaded.evidence.clone()).collect::<Vec<_>>();
    if evidence.is_empty(){return Err(preparation("no optional kernel module has been successfully prepared for this device"));}
    evidence.sort_by(|a,b|a.kernel_name.cmp(&b.kernel_name));
    Ok(evidence)
}

/// After this function attempts a launch, any returned error is an execution
/// uncertainty, never permission to replay the current primitive on a fallback.
pub(crate) fn launch(device:&RocmDevice,module:&HipModule,name:&str,grid:(u32,u32,u32),block:(u32,u32,u32),shared:u32,args:&mut[*mut c_void])->candle_core::Result<()> {
    device.set_current().map_err(|error|candle_core::Error::Msg(format!("HIP_EXECUTION_UNCERTAIN: selecting kernel device: {error}")))?;
    let function=module.get_function(name).map_err(preparation)?;
    if let Err(error)=unsafe{HipModule::launch(function,grid,block,shared,args)} {
        UNCERTAIN_FAILURES.fetch_add(1,Ordering::Relaxed);
        return Err(candle_core::Error::Msg(format!("HIP_EXECUTION_UNCERTAIN: {name}: {error}")));
    }
    let counter=match name{"pp_attention_f32"|"pp_attention_bf16"=>&ATTENTION_PARTITIONS,"pp_attention_merge"=>&ATTENTION_MERGES,"pp_renyi_sample"|"pp_hip_renyi_sample"=>&RENYI_SAMPLES,_=>&OTHER_LAUNCHES};
    counter.fetch_add(1,Ordering::Relaxed);
    Ok(())
}
