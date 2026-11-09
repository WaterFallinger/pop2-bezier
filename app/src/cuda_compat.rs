//! Build/runtime contracts for the CUDA workers. A lower ordinary PTX target
//! is distinct from optional kernels compiled for the actual selected device.
use serde::{Deserialize, Serialize};

pub const MINIMUM_DRIVER_API: i32 = 13010;
pub const PORTABLE_MINIMUM_CC: (i32, i32) = (8, 0);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaKernelProfile {
    PortableSm80,
    BlackwellSm120a,
}
impl CudaKernelProfile {
    pub fn id(self) -> &'static str {
        match self { Self::PortableSm80 => "portable", Self::BlackwellSm120a => "optimized" }
    }
    pub fn embedded_target(self) -> &'static str {
        match self { Self::PortableSm80 => "sm_80", Self::BlackwellSm120a => "sm_120a" }
    }
    pub fn accepts(self, major: i32, minor: i32) -> bool {
        if minor < 0 || minor > 9 { return false; }
        match self {
            Self::PortableSm80 => (major, minor) >= PORTABLE_MINIMUM_CC,
            Self::BlackwellSm120a => (major, minor) == (12, 0),
        }
    }
}

pub fn compiled_kernel_profile() -> CudaKernelProfile {
    match option_env!("PAINTER_CUDA_PROFILE") {
        Some("portable") => CudaKernelProfile::PortableSm80,
        _ => CudaKernelProfile::BlackwellSm120a,
    }
}

pub fn parse_cuda_ordinal(value: Option<&str>) -> Result<usize, String> {
    match value {
        None => Ok(0),
        Some(value) => {
            let ordinal = value.trim().parse::<usize>()
                .map_err(|_| "PAINTER_CUDA_DEVICE必须是非负CUDA设备编号".to_owned())?;
            if ordinal > i32::MAX as usize {
                return Err("PAINTER_CUDA_DEVICE超出CUDA设备编号范围".into());
            }
            Ok(ordinal)
        }
    }
}
pub fn selected_cuda_ordinal() -> Result<usize, String> {
    Ok(explicit_cuda_ordinal()?.unwrap_or(0))
}
pub fn explicit_cuda_ordinal() -> Result<Option<usize>,String> {
    match std::env::var("PAINTER_CUDA_DEVICE") {
        Ok(value) => parse_cuda_ordinal(Some(&value)).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err("PAINTER_CUDA_DEVICE不是有效文本".into()),
    }
}

/// Generic virtual targets retain PTX forward compatibility. No architecture-
/// accelerated `a`/`f` target is inferred from a marketing name.
pub fn choose_jit_compute(actual: (i32,i32), supported: &[i32]) -> Result<i32, String> {
    if actual.0 < 8 || actual.1 < 0 || actual.1 > 9 {
        return Err("当前CUDA模型路径要求计算能力8.0或更新的原生BF16设备".into());
    }
    let actual = actual.0.checked_mul(10).and_then(|major| major.checked_add(actual.1))
        .ok_or_else(|| "CUDA计算能力数值无效".to_owned())?;
    supported.iter().copied().filter(|&cc| cc >= 80 && cc <= actual).max()
        .ok_or_else(|| "当前可选CUDA编译器没有适合该设备的通用PTX目标；保留普通GPU路径".to_owned())
}

#[cfg(feature="cuda")]
static NVRTC_TARGETS:std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<candle_core::cuda_backend::DeviceId,String>>>=std::sync::OnceLock::new();

#[cfg(feature="cuda")]
pub(crate) fn cached_optional_jit_target(device:&candle_core::CudaDevice)->Option<String> {
    NVRTC_TARGETS.get()?.lock().ok()?.get(&device.id()).cloned()
}

#[cfg(feature="cuda")]
pub(crate) fn optional_jit_target(device: &candle_core::CudaDevice) -> candle_core::Result<String> {
    use candle_core::cuda_backend::cudarc;
    use std::{collections::HashMap,sync::Mutex};
    let mut cache=NVRTC_TARGETS.get_or_init(||Mutex::new(HashMap::new())).lock()
        .map_err(|_|candle_core::Error::Msg("CUDA target policy cache unavailable".into()))?;
    if let Some(target)=cache.get(&device.id()){return Ok(target.clone());}
    let actual = device.cuda_stream().context().compute_capability()
        .map_err(candle_core::Error::wrap)?;
    let mut count = 0i32;
    let status = unsafe { cudarc::nvrtc::sys::nvrtcGetNumSupportedArchs(&mut count) };
    if status != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS || !(1..=256).contains(&count) {
        candle_core::bail!("无法读取当前可选NVRTC支持的计算目标；保留普通GPU路径");
    }
    let mut supported = vec![0i32; count as usize];
    let status = unsafe { cudarc::nvrtc::sys::nvrtcGetSupportedArchs(supported.as_mut_ptr()) };
    if status != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        candle_core::bail!("无法读取当前可选NVRTC的计算目标列表；保留普通GPU路径");
    }
    let target = choose_jit_compute(actual,&supported).map_err(candle_core::Error::Msg)?;
    let target=format!("compute_{target}");
    cache.insert(device.id(),target.clone());
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn portable_ptx_accepts_ampere_and_newer_without_weakening_accelerated_target() {
        for cc in [(8,0),(8,6),(8,9),(9,0),(10,0),(12,0),(12,1),(13,0)] {
            assert!(CudaKernelProfile::PortableSm80.accepts(cc.0,cc.1));
        }
        assert!(!CudaKernelProfile::PortableSm80.accepts(7,5));
        assert!(CudaKernelProfile::BlackwellSm120a.accepts(12,0));
        assert!(!CudaKernelProfile::BlackwellSm120a.accepts(12,1));
        assert!(!CudaKernelProfile::BlackwellSm120a.accepts(13,0));
    }
    #[test]
    fn cuda_ordinal_is_explicit_and_never_a_renderer_index() {
        assert_eq!(parse_cuda_ordinal(None).unwrap(),0);
        assert_eq!(parse_cuda_ordinal(Some(" 1 ")).unwrap(),1);
        for value in ["", "-1", "gpu0", "2147483648"] { assert!(parse_cuda_ordinal(Some(value)).is_err()); }
    }
    #[test]
    fn optional_jit_uses_real_supported_virtual_targets_and_can_forward_jit() {
        let targets=[75,80,86,89,90,100,120];
        assert_eq!(choose_jit_compute((12,0),&targets).unwrap(),120);
        assert_eq!(choose_jit_compute((8,6),&targets).unwrap(),86);
        assert_eq!(choose_jit_compute((13,0),&targets).unwrap(),120);
        assert!(choose_jit_compute((7,5),&targets).is_err());
        assert!(choose_jit_compute((8,0),&[90,120]).is_err());
    }
}
