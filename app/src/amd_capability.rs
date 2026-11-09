//! AMD library capability inspection, not an implemented model backend.
//!
//! This module never calls HIP device initialization, allocates GPU memory, or
//! runs a kernel. A future AMD provider must separately validate its numerical
//! results and supported device before advertising model execution.

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LibraryProbe {
    pub path: Option<PathBuf>,
    pub loaded: bool,
    pub missing_symbols: Vec<String>,
    pub load_error: Option<String>,
}

impl LibraryProbe {
    pub fn has_required_symbols(&self) -> bool {
        self.loaded && self.missing_symbols.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AmdCapabilityReport {
    pub hip_runtime: LibraryProbe,
    pub rocblas: LibraryProbe,
    pub hiprtc: LibraryProbe,
    pub graph_missing_symbols: Vec<String>,
    /// Library discovery cannot establish coherent SDK versions.
    pub version_pairing_verified: bool,
    /// Deliberately false until the actual model implementation is connected.
    pub model_backend_implemented: bool,
    pub hardware_validated: bool,
    pub reason: String,
}

impl AmdCapabilityReport {
    /// Version must come from the selected provider's actual compiler query.
    /// Merely discovering HIPRTC or its DLL name supplies no version evidence.
    /// An ExternalHipcc build uses OptionalCompilerFingerprint::external_hipcc
    /// instead and does not require HIPRTC to be present.
    pub fn hiprtc_compiler_fingerprint(&self, detected_version: Option<String>)
        -> crate::acceleration::OptionalCompilerFingerprint
    {
        crate::acceleration::OptionalCompilerFingerprint {
            kind: Some(crate::acceleration::OptionalCompilerKind::Hiprtc),
            path: self.hiprtc.path.clone(), version: detected_version,
            library_loaded: self.hiprtc.loaded,
            required_symbols_verified: self.hiprtc.has_required_symbols(),
            compile_probe_passed: false,
        }
    }

    pub fn model_execution_available(&self) -> bool {
        self.model_backend_implemented
            && self.hardware_validated
            && self.version_pairing_verified
            && self.hip_runtime.has_required_symbols()
            && self.rocblas.has_required_symbols()
    }

    pub fn graph_library_symbols_available(&self) -> bool {
        self.hip_runtime.has_required_symbols() && self.graph_missing_symbols.is_empty()
    }
}

const HIP_REQUIRED: &[&str] = &[
    "hipGetDeviceCount", "hipSetDevice", "hipMalloc", "hipFree", "hipMemcpy",
    "hipStreamCreate", "hipStreamDestroy", "hipStreamSynchronize", "hipModuleLoadData",
    "hipModuleGetFunction", "hipModuleLaunchKernel",
];
const ROCBLAS_REQUIRED: &[&str] = &[
    "rocblas_create_handle", "rocblas_destroy_handle", "rocblas_set_stream",
    "rocblas_gemm_ex", "rocblas_gemm_strided_batched_ex",
];
const HIPRTC_REQUIRED: &[&str] = &[
    "hiprtcCreateProgram", "hiprtcCompileProgram", "hiprtcGetCodeSize",
    "hiprtcGetCode", "hiprtcDestroyProgram",
];
const GRAPH_REQUIRED: &[&str] = &[
    "hipStreamBeginCapture", "hipStreamEndCapture", "hipGraphInstantiate",
    "hipGraphLaunch", "hipGraphExecDestroy", "hipGraphDestroy",
];

fn unavailable_report(
    hip_runtime: LibraryProbe,
    rocblas: LibraryProbe,
    hiprtc: LibraryProbe,
    graph_missing_symbols: Vec<String>,
) -> AmdCapabilityReport {
    AmdCapabilityReport {
        hip_runtime,
        rocblas,
        hiprtc,
        graph_missing_symbols,
        version_pairing_verified: false,
        model_backend_implemented: false,
        hardware_validated: false,
        reason: "此处只核对AMD运行库；模型是否可用须以实际所选HIP设备的普通计算和数值验证为准，不能仅凭DLL发现授予能力。".into(),
    }
}

/// Inspect only an explicitly selected, absolute application runtime folder
/// plus Windows System32. No PATH/current-directory DLL search is performed.
/// Call this only when AMD is selected; normal NVIDIA startup need not load
/// any AMD DLL. `None` restricts inspection to the system driver location.
#[cfg(windows)]
pub fn inspect_windows_libraries(runtime_dir: Option<&Path>) -> AmdCapabilityReport {
    let mut roots = Vec::new();
    if let Some(root) = runtime_dir.filter(|root| root.is_absolute()) {
        roots.push(root.to_path_buf());
    }
    if let Some(system) = windows::system_directory() {
        if !roots.contains(&system) { roots.push(system); }
    }
    let hip_names = ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];
    let (hip_runtime, graph_missing) = windows::inspect(&roots, &hip_names, HIP_REQUIRED, GRAPH_REQUIRED);
    let (rocblas, _) = windows::inspect(&roots, &["rocblas.dll", "librocblas.dll"], ROCBLAS_REQUIRED, &[]);
    let (hiprtc, _) = windows::inspect(&roots, &["hiprtc.dll", "hiprtc0700.dll", "hiprtc0701.dll", "hiprtc0702.dll", "hiprtc0600.dll"], HIPRTC_REQUIRED, &[]);
    unavailable_report(hip_runtime, rocblas, hiprtc, graph_missing)
}

#[cfg(not(windows))]
pub fn inspect_windows_libraries(_runtime_dir: Option<&Path>) -> AmdCapabilityReport {
    let mut report = unavailable_report(LibraryProbe::default(), LibraryProbe::default(), LibraryProbe::default(), GRAPH_REQUIRED.iter().map(|s| (*s).into()).collect());
    report.reason = "此探测入口仅检查Windows AMD运行库，不执行模型。".into();
    report
}

#[cfg(windows)]
mod windows {
    use super::*;
    use std::{ffi::{c_char, c_void, CString}, os::windows::ffi::OsStrExt};

    #[link(name = "kernel32")]
    extern "system" {
        fn GetSystemDirectoryW(buffer: *mut u16, size: u32) -> u32;
        fn LoadLibraryExW(path: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
        fn GetLastError() -> u32;
    }

    struct Module(*mut c_void);
    impl Drop for Module {
        fn drop(&mut self) { if !self.0.is_null() { unsafe { FreeLibrary(self.0); } } }
    }

    pub(super) fn system_directory() -> Option<PathBuf> {
        let mut wide = vec![0u16; 32_768];
        let length = unsafe { GetSystemDirectoryW(wide.as_mut_ptr(), wide.len() as u32) } as usize;
        if length == 0 || length >= wide.len() { return None; }
        Some(PathBuf::from(String::from_utf16_lossy(&wide[..length])))
    }

    fn missing(module: &Module, names: &[&str]) -> Vec<String> {
        names.iter().filter(|name| {
            let name = CString::new(**name).expect("constant API symbol");
            unsafe { GetProcAddress(module.0, name.as_ptr()).is_null() }
        }).map(|name| (*name).to_owned()).collect()
    }

    pub(super) fn inspect(
        roots: &[PathBuf], names: &[&str], required: &[&str], optional: &[&str],
    ) -> (LibraryProbe, Vec<String>) {
        let mut first_failure = None;
        for root in roots {
            for name in names {
                let candidate = root.join(name);
                if !candidate.is_file() { continue; }
                let path = match candidate.canonicalize() {
                    Ok(path) => path,
                    Err(error) => {
                        first_failure.get_or_insert(LibraryProbe { path: Some(candidate), load_error: Some(error.to_string()), ..Default::default() });
                        continue;
                    }
                };
                let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
                // Search the selected DLL's own directory and trusted Windows
                // defaults for dependencies, never the process working folder.
                let raw = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), 0x0000_0100 | 0x0000_1000) };
                if raw.is_null() {
                    let error = unsafe { GetLastError() };
                    first_failure.get_or_insert(LibraryProbe { path: Some(path), load_error: Some(format!("Windows loader error {error}")), ..Default::default() });
                    continue;
                }
                let module = Module(raw);
                let missing_symbols = missing(&module, required);
                let optional_missing = missing(&module, optional);
                let probe = LibraryProbe { path: Some(path), loaded: true, missing_symbols, load_error: None };
                if probe.has_required_symbols() { return (probe, optional_missing); }
                first_failure.get_or_insert(probe);
            }
        }
        (first_failure.unwrap_or_default(), optional.iter().map(|s| (*s).to_owned()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn present() -> LibraryProbe {
        LibraryProbe { path: Some(PathBuf::from("C:/test/runtime.dll")), loaded: true, ..Default::default() }
    }

    #[test]
    fn library_symbols_alone_never_advertise_model_execution() {
        let report = unavailable_report(present(), present(), present(), Vec::new());
        assert!(report.graph_library_symbols_available());
        assert!(!report.model_execution_available());
        assert!(!report.model_backend_implemented);
        assert!(!report.hardware_validated);
        assert!(!report.version_pairing_verified);
        assert!(!report.hiprtc_compiler_fingerprint(None).usable(),
            "DLL presence and symbols do not establish a real compiler version or implementation");
    }

    #[test]
    fn graph_symbols_are_not_reported_when_runtime_is_absent_or_incomplete() {
        let missing = unavailable_report(LibraryProbe::default(), present(), present(), Vec::new());
        assert!(!missing.graph_library_symbols_available());
        let incomplete = unavailable_report(present(), present(), present(), vec!["hipGraphLaunch".into()]);
        assert!(!incomplete.graph_library_symbols_available());
    }

    #[test]
    fn ordinary_model_backend_still_requires_matching_runtime_and_blas() {
        let mut report = unavailable_report(present(), present(), LibraryProbe::default(), Vec::new());
        report.model_backend_implemented = true;
        report.hardware_validated = true;
        report.version_pairing_verified = true;
        assert!(report.model_execution_available()); // Offline kernels need no HIPRTC.
        report.rocblas.loaded = false;
        assert!(!report.model_execution_available());
    }
}
