//! Read-only CUDA device admission for explicitly identified worker builds.
//! No CUDA import library is linked. The system driver is loaded explicitly;
//! cuInit and read-only device queries do not create a CUDA context or run work.

use serde::Serialize;
#[path = "cuda_compat.rs"]
mod cuda_compat;
pub use cuda_compat::{CudaKernelProfile, selected_cuda_ordinal};

pub const REQUIRED_DRIVER_API: i32 = cuda_compat::MINIMUM_DRIVER_API;

#[derive(Clone, Debug, Serialize)]
pub struct HardwareReport {
    pub driver_api: i32,
    pub device_name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub compatible: bool,
    pub reason: String,
    pub cuda_ordinal: usize,
    pub kernel_profile: CudaKernelProfile,
    pub embedded_ptx_target: String,
    pub minimum_driver_api: i32,
    pub bf16_supported: bool,
    pub total_memory_bytes: Option<u64>,
    pub candidates: Vec<CudaDeviceCandidate>,
    pub selection_reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct CudaDeviceCandidate {
    pub cuda_ordinal:usize,
    pub device_name:Option<String>,
    pub compute_major:Option<i32>,
    pub compute_minor:Option<i32>,
    pub total_memory_bytes:Option<u64>,
    pub compatible:bool,
    pub reason:String,
}

pub fn select_cuda_candidate(candidates:&[CudaDeviceCandidate],explicit:Option<usize>)->Result<usize,String> {
    if let Some(ordinal)=explicit {
        return candidates.iter().position(|c|c.cuda_ordinal==ordinal)
            .ok_or_else(||format!("指定CUDA设备{ordinal}不存在；不会替换为其他CUDA设备"));
    }
    let rank=|c:&CudaDeviceCandidate|(c.total_memory_bytes.unwrap_or(0),c.compute_major.unwrap_or(-1),c.compute_minor.unwrap_or(-1),std::cmp::Reverse(c.cuda_ordinal));
    candidates.iter().enumerate().filter(|(_,c)|c.compatible).max_by_key(|(_,c)|rank(c)).map(|(i,_)|i)
        .or_else(||candidates.iter().enumerate().filter(|(_,c)|c.compute_major.is_some()&&c.compute_minor.is_some()).max_by_key(|(_,c)|rank(c)).map(|(i,_)|i))
        .ok_or_else(||"CUDA设备属性均未能读取；保留CPU兼容候选".into())
}

/// Pure decision, deliberately not a scan for some other supported adapter.
/// `compatible` means the selected CUDA device meets this build's criteria;
/// it is not a VRAM-capacity, model-quality, or full inference test.
pub fn evaluate_compatibility(
    driver_api: i32,
    device_name: &str,
    major: i32,
    minor: i32,
) -> HardwareReport {
    evaluate_compatibility_for_profile(driver_api,device_name,major,minor,
        cuda_compat::compiled_kernel_profile())
}

pub fn evaluate_compatibility_for_profile(driver_api:i32,device_name:&str,major:i32,minor:i32,
    profile:CudaKernelProfile)->HardwareReport {
    let architecture_ok = profile.accepts(major,minor);
    let driver_ok = driver_api >= REQUIRED_DRIVER_API;
    let mut problems = Vec::new();
    if !architecture_ok {
        problems.push(match profile {
            CudaKernelProfile::PortableSm80=>format!("所选CUDA设备“{device_name}”的计算能力为{major}.{minor}；普通GPU后端要求8.0或更新及原生BF16，可选择CPU兼容后端"),
            CudaKernelProfile::BlackwellSm120a=>format!("所选CUDA设备“{device_name}”的计算能力为{major}.{minor}；此专用后端需要12.0（sm_120a），请使用通用CUDA或CPU兼容后端"),
        });
    }
    if !driver_ok {
        let version = if driver_api > 0 {
            format!("{}.{}", driver_api / 1000, (driver_api % 1000) / 10)
        } else {
            "未知".into()
        };
        problems.push(format!("当前驱动报告的CUDA API为{version}；此安装包需要支持CUDA13.1／PTX9.1的NVIDIA驱动，请通过NVIDIA官方渠道更新驱动"));
    }
    HardwareReport {
        driver_api,
        device_name: device_name.into(),
        compute_major: major,
        compute_minor: minor,
        compatible: architecture_ok && driver_ok,
        reason: if problems.is_empty() {
            "所选CUDA设备和驱动满足该后端最低条件；实际模型计算与显存仍需加载验证".into()
        } else {
            problems.join("；")
        },
        cuda_ordinal: 0,
        kernel_profile: profile,
        embedded_ptx_target:profile.embedded_target().into(),
        minimum_driver_api:REQUIRED_DRIVER_API,
        bf16_supported:CudaKernelProfile::PortableSm80.accepts(major,minor),
        total_memory_bytes:None,
        candidates:Vec::new(),
        selection_reason:"纯硬件条件评估，尚未枚举实际CUDA设备".into(),
    }
}

#[cfg(all(windows, target_pointer_width = "64"))]
pub fn probe_nvidia() -> Result<HardwareReport, String> {
    probe_nvidia_for_profile(cuda_compat::compiled_kernel_profile())
}

#[cfg(all(windows, target_pointer_width = "64"))]
pub fn probe_nvidia_for_profile(profile:CudaKernelProfile)->Result<HardwareReport,String> {
    windows_driver::probe(profile)
}

#[cfg(not(all(windows, target_pointer_width = "64")))]
pub fn probe_nvidia() -> Result<HardwareReport, String> {
    Err("此安装包的显卡检查仅支持64位Windows".into())
}
#[cfg(not(all(windows, target_pointer_width = "64")))]
pub fn probe_nvidia_for_profile(_:CudaKernelProfile)->Result<HardwareReport,String> {
    Err("CUDA后端检查仅支持64位Windows".into())
}

#[cfg(all(windows, target_pointer_width = "64"))]
mod windows_driver {
    use super::*;
    use std::ffi::{c_char, c_void, CStr};

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(filename: *const u16) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn GetSystemDirectoryW(buffer: *mut u16, size: u32) -> u32;
    }

    struct DriverLibrary(*mut c_void);
    impl Drop for DriverLibrary {
        fn drop(&mut self) {
            unsafe {
                FreeLibrary(self.0);
            }
        }
    }

    type CuInit = unsafe extern "system" fn(u32) -> i32;
    type CuDriverGetVersion = unsafe extern "system" fn(*mut i32) -> i32;
    type CuDeviceGetCount = unsafe extern "system" fn(*mut i32) -> i32;
    type CuDeviceGet = unsafe extern "system" fn(*mut i32, i32) -> i32;
    type CuDeviceGetName = unsafe extern "system" fn(*mut c_char, i32, i32) -> i32;
    type CuDeviceGetAttribute = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;
    type CuDeviceTotalMem = unsafe extern "system" fn(*mut usize,i32)->i32;

    fn system_driver_path() -> Result<Vec<u16>, String> {
        let mut path = vec![0u16; 512];
        let mut length = unsafe { GetSystemDirectoryW(path.as_mut_ptr(), path.len() as u32) };
        if length == 0 {
            return Err(format!(
                "无法定位Windows系统目录：{}",
                std::io::Error::last_os_error()
            ));
        }
        if length as usize >= path.len() {
            path.resize(length as usize + 1, 0);
            length = unsafe { GetSystemDirectoryW(path.as_mut_ptr(), path.len() as u32) };
            if length == 0 || length as usize >= path.len() {
                return Err("无法读取Windows系统目录".into());
            }
        }
        path.truncate(length as usize);
        if path.last() != Some(&(b'\\' as u16)) {
            path.push(b'\\' as u16);
        }
        path.extend("nvcuda.dll".encode_utf16());
        path.push(0);
        Ok(path)
    }

    fn symbol(library: &DriverLibrary, name: &'static [u8]) -> Result<*mut c_void, String> {
        let symbol = unsafe { GetProcAddress(library.0, name.as_ptr().cast()) };
        if symbol.is_null() {
            let function = String::from_utf8_lossy(&name[..name.len() - 1]);
            return Err(format!(
                "NVIDIA驱动缺少{function}接口，请安装完整的官方NVIDIA驱动"
            ));
        }
        Ok(symbol)
    }

    fn check(status: i32, operation: &str) -> Result<(), String> {
        if status == 0 {
            return Ok(());
        }
        if status == 100 {
            return Err("未检测到可用的NVIDIA CUDA显卡；可使用CPU兼容后端".into());
        }
        Err(format!("NVIDIA驱动的{operation}检查失败（CUDA错误{status}）；请检查显卡驱动是否完整并支持CUDA13.1。安装器不会自动更改驱动"))
    }

    pub(super) fn probe(profile:CudaKernelProfile) -> Result<HardwareReport, String> {
        let path = system_driver_path()?;
        // An absolute system path prevents an application-directory DLL from
        // impersonating the user's GPU driver. No DLL search paths are changed.
        let handle = unsafe { LoadLibraryW(path.as_ptr()) };
        if handle.is_null() {
            return Err(format!("无法加载系统NVIDIA驱动（nvcuda.dll）：{}。请从NVIDIA官方渠道安装兼容驱动；无需安装CUDA开发工具包",std::io::Error::last_os_error()));
        }
        let library = DriverLibrary(handle);
        let init: CuInit = unsafe { std::mem::transmute(symbol(&library, b"cuInit\0")?) };
        let get_version: CuDriverGetVersion =
            unsafe { std::mem::transmute(symbol(&library, b"cuDriverGetVersion\0")?) };
        let get_count: CuDeviceGetCount =
            unsafe { std::mem::transmute(symbol(&library, b"cuDeviceGetCount\0")?) };
        let get_device: CuDeviceGet =
            unsafe { std::mem::transmute(symbol(&library, b"cuDeviceGet\0")?) };
        let get_name: CuDeviceGetName =
            unsafe { std::mem::transmute(symbol(&library, b"cuDeviceGetName\0")?) };
        let get_attribute: CuDeviceGetAttribute =
            unsafe { std::mem::transmute(symbol(&library, b"cuDeviceGetAttribute\0")?) };
        let total_memory:CuDeviceTotalMem=unsafe{std::mem::transmute(symbol(&library,b"cuDeviceTotalMem_v2\0")?)};
        // This initializes only the driver API. No cuCtxCreate, primary-context
        // retain, CUDA tensor allocation, PTX loading, or kernel launch is called.
        check(unsafe { init(0) }, "初始化")?;
        let mut version = 0;
        check(unsafe { get_version(&mut version) }, "版本")?;
        let mut count = 0;
        check(unsafe { get_count(&mut count) }, "设备数量")?;
        if !(1..=1024).contains(&count) {return Err("未检测到合理数量的NVIDIA CUDA设备".into());}
        let explicit=cuda_compat::explicit_cuda_ordinal()?;
        let mut candidates=Vec::new();
        for ordinal in 0..count as usize {
            let observed=(||->Result<CudaDeviceCandidate,String>{
                let mut device=0;check(unsafe{get_device(&mut device,ordinal as i32)},"枚举CUDA设备")?;
                let mut name=[0 as c_char;256];check(unsafe{get_name(name.as_mut_ptr(),name.len() as i32,device)},"设备名称")?;
                name[255]=0;let name=unsafe{CStr::from_ptr(name.as_ptr())}.to_string_lossy().into_owned();
                let(mut major,mut minor)=(0,0);
                check(unsafe{get_attribute(&mut major,75,device)},"计算能力major")?;
                check(unsafe{get_attribute(&mut minor,76,device)},"计算能力minor")?;
                let mut bytes=0usize;let memory=if unsafe{total_memory(&mut bytes,device)}==0{Some(bytes as u64)}else{None};
                let report=evaluate_compatibility_for_profile(version,&name,major,minor,profile);
                Ok(CudaDeviceCandidate{cuda_ordinal:ordinal,device_name:Some(name),compute_major:Some(major),compute_minor:Some(minor),
                    total_memory_bytes:memory,compatible:report.compatible,reason:report.reason})
            })();
            candidates.push(observed.unwrap_or_else(|error|CudaDeviceCandidate{cuda_ordinal:ordinal,device_name:None,
                compute_major:None,compute_minor:None,total_memory_bytes:None,compatible:false,reason:error}));
        }
        let selected=select_cuda_candidate(&candidates,explicit)?;
        let candidate=&candidates[selected];
        let name=candidate.device_name.as_deref().unwrap_or("CUDA设备名称不可读取");
        let major=candidate.compute_major.ok_or_else(||candidate.reason.clone())?;
        let minor=candidate.compute_minor.ok_or_else(||candidate.reason.clone())?;
        let mut report=evaluate_compatibility_for_profile(version,&name,major,minor,profile);
        report.cuda_ordinal=candidate.cuda_ordinal;
        report.total_memory_bytes=candidate.total_memory_bytes;
        report.selection_reason=if explicit.is_some(){format!("尊重明确指定的CUDA设备{}，不替换设备",candidate.cuda_ordinal)}
            else if candidate.compatible{format!("自动枚举CUDA设备后选择满足后端条件且总显存较大的设备{}；总显存不代表当前空闲显存",candidate.cuda_ordinal)}
            else{"没有CUDA设备满足该后端条件，记录可读取设备后交由CPU兼容选择处理".into()};
        report.candidates=candidates;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepts_only_this_build_architecture_with_sufficient_driver_api() {
        let report = evaluate_compatibility(13010, "device0 test fixture", 12, 0);
        assert!(report.compatible);
        assert_eq!(report.driver_api, 13010);
        assert!(evaluate_compatibility(14000, "device0 test fixture", 12, 0).compatible);
    }
    #[test]
    fn ptx91_driver_requirement_is_not_cuda13_major_only() {
        let report = evaluate_compatibility(13000, "device0 test fixture", 12, 0);
        assert!(!report.compatible);
        assert!(report.reason.contains("13.1"));
        assert!(!evaluate_compatibility(0, "device0 test fixture", 12, 0).compatible);
    }
    #[test]
    fn architecture_specific_target_does_not_accept_newer_or_older_cc() {
        for (major, minor) in [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 1), (13, 0)] {
            let report = evaluate_compatibility_for_profile(13010, "CUDA device0", major, minor,CudaKernelProfile::BlackwellSm120a);
            assert!(
                !report.compatible,
                "unexpected acceptance for {major}.{minor}"
            );
            assert_eq!((report.compute_major, report.compute_minor), (major, minor));
            assert!(report.reason.contains("所选CUDA设备"));
        }
    }
    #[test]
    fn combined_incompatibilities_explain_both_without_changing_device() {
        let report = evaluate_compatibility_for_profile(12040, "unsupported selected CUDA device0", 8, 6,CudaKernelProfile::BlackwellSm120a);
        assert!(!report.compatible);
        assert!(report.reason.contains("12.0"));
        assert!(report.reason.contains("13.1"));
        assert_eq!(report.device_name, "unsupported selected CUDA device0");
    }
    #[test]
    fn portable_worker_has_a_real_lower_ptx_contract_and_separate_driver_gate() {
        for (major,minor) in [(8,0),(8,6),(8,9),(9,0),(10,0),(12,0),(12,1)] {
            let r=evaluate_compatibility_for_profile(13010,"CUDA selected",major,minor,CudaKernelProfile::PortableSm80);
            assert!(r.compatible && r.bf16_supported);
            assert_eq!(r.embedded_ptx_target,"sm_80");
        }
        assert!(!evaluate_compatibility_for_profile(13010,"Turing",7,5,CudaKernelProfile::PortableSm80).compatible);
        assert!(!evaluate_compatibility_for_profile(12040,"Ampere old driver",8,6,CudaKernelProfile::PortableSm80).compatible);
    }
    #[test]
    fn automatic_cuda_selection_checks_all_compute_devices_and_explicit_choice_wins() {
        let device=|ordinal,compatible,memory|CudaDeviceCandidate{cuda_ordinal:ordinal,
            device_name:Some(format!("CUDA{ordinal}")),compute_major:Some(if compatible{8}else{7}),
            compute_minor:Some(if compatible{6}else{5}),total_memory_bytes:Some(memory),compatible,reason:String::new()};
        let mixed=vec![device(0,false,24<<30),device(1,true,12<<30)];
        assert_eq!(select_cuda_candidate(&mixed,None).unwrap(),1);
        assert_eq!(select_cuda_candidate(&mixed,Some(0)).unwrap(),0);
        assert!(select_cuda_candidate(&mixed,Some(2)).is_err());
        let modern=vec![device(0,true,8<<30),device(1,true,24<<30)];
        assert_eq!(select_cuda_candidate(&modern,None).unwrap(),1);
        assert_eq!(select_cuda_candidate(&[device(0,true,8<<30),device(1,true,8<<30)],None).unwrap(),0);
    }
}
