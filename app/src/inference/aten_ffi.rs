//! Worker-bound C ABI bridge to the native ATen/HIP provider.
//! The frontend does not link or initialize another HIP/CUDA tensor runtime.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::{
    cell::Cell,
    ffi::{c_char, c_void, CStr, CString},
    path::{Path, PathBuf},
    ptr::NonNull,
    rc::Rc,
};

const ABI_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug)]
pub(super) struct RowGeometry {
    pub logical: u32,
    pub physical: usize,
    pub attributes: usize,
    pub bounds: usize,
    pub terms: usize,
}

/// Only the native ABI and address-space representation constrain row counts.
/// Available host/GPU memory is checked by fallible allocation, never by a
/// product-specific parallel-sequence cap.
pub(super) fn row_geometry(rows: usize, guidance: usize) -> Result<RowGeometry> {
    if rows == 0 || !matches!(guidance, 1 | 2) {
        bail!("ATen并行数必须至少为1，CFG物理分支数必须为1或2");
    }
    let logical = u32::try_from(rows).context("并行数超出当前原生接口可表示的整数范围")?;
    let multiply = |count: usize, factor: usize| {
        count
            .checked_mul(factor)
            .ok_or_else(|| anyhow::anyhow!("ATen批量维度超出地址空间可表示范围"))
    };
    let physical = multiply(rows, guidance)?;
    let attributes = multiply(rows, 9)?;
    let bounds = multiply(attributes, 2)?;
    let terms = multiply(physical, 2)?;
    for (count, bytes) in [
        (bounds, 4),
        (attributes, 4),
        (terms, 4),
        (physical, std::mem::size_of::<usize>()),
        (rows, std::mem::size_of::<[u32; 9]>()),
    ] {
        if multiply(count, bytes)? > isize::MAX as usize {
            bail!("ATen批量缓冲区超出本进程地址空间");
        }
    }
    Ok(RowGeometry {
        logical,
        physical,
        attributes,
        bounds,
        terms,
    })
}

pub(super) fn buffer<T>(capacity: usize) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .context("系统内存不足，无法准备本次并行缓冲区；未减少用户设置的并行数")?;
    Ok(output)
}
pub(super) fn filled_buffer<T: Clone>(length: usize, value: T) -> Result<Vec<T>> {
    let mut output = buffer(length)?;
    output.resize(length, value);
    Ok(output)
}

#[repr(C)]
struct LoadOptions {
    struct_size: u32,
    device_ordinal: u32,
    prefer_efficient_attention: u32,
    reserved: u32,
    index_utf8: *const c_char,
}

#[repr(C)]
pub(super) struct FrameOptions {
    pub struct_size: u32,
    pub logical_rows: u32,
    pub guidance: u32,
    pub graph_allowed: u32,
    pub sampling_gpu: u32,
    pub efficient_attention: u32,
    pub cfg: f64,
    pub alpha: f64,
}

#[repr(C)]
struct AutoProbeOptions {
    struct_size: u32,
    logical_rows: u32,
    guidance: u32,
    regional: u32,
    candidate_efficient: u32,
    candidate_gpu_sampling: u32,
    prefix_records: u32,
    reserved: u32,
    cfg: f64,
    alpha: f64,
}
impl FrameOptions {
    pub fn new(
        rows: usize,
        guidance: usize,
        cfg: f64,
        alpha: f64,
        graph: bool,
        gpu_sampling: bool,
        efficient: bool,
    ) -> Result<Self> {
        let geometry = row_geometry(rows, guidance)?;
        if !matches!(guidance, 1 | 2)
            || !cfg.is_finite()
            || cfg < 0.
            || !alpha.is_finite()
            || alpha <= 1.
        {
            bail!("ATen生成参数无效");
        }
        Ok(Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            logical_rows: geometry.logical,
            guidance: u32::try_from(guidance)?,
            graph_allowed: u32::from(graph && gpu_sampling),
            sampling_gpu: u32::from(gpu_sampling),
            efficient_attention: u32::from(efficient),
            cfg,
            alpha,
        })
    }
}

type AbiVersion = unsafe extern "C" fn() -> u32;
type LastError = unsafe extern "C" fn() -> *const c_char;
type Load = unsafe extern "C" fn(*const c_char, *const LoadOptions, *mut *mut c_void) -> i32;
type ModelInfo = unsafe extern "C" fn(*const c_void) -> *const c_char;
type Release = unsafe extern "C" fn(*mut c_void);
type Configure = unsafe extern "C" fn(*mut c_void, u32) -> i32;
type Prepare = unsafe extern "C" fn(*mut c_void) -> i32;
type ProbeAuto = unsafe extern "C" fn(*mut c_void, *const AutoProbeOptions) -> i32;
type Prefill = unsafe extern "C" fn(
    *mut c_void,
    *const u32,
    usize,
    *const u32,
    usize,
    *const u8,
    usize,
    *mut *mut c_void,
    *mut f32,
) -> i32;
type CacheInfo = unsafe extern "C" fn(*const c_void, *mut usize, *mut usize) -> i32;
type Extend = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    usize,
    *const u32,
    usize,
    *mut *mut c_void,
    *mut f32,
) -> i32;
type Frame = unsafe extern "C" fn(
    *mut c_void,
    *const *const c_void,
    *const FrameOptions,
    *const u32,
    *const f32,
    *mut u32,
    *mut f32,
    *mut u32,
    *mut *mut c_void,
) -> i32;

struct Api {
    // Deliberately kept loaded for process lifetime: frameworks own global
    // dispatcher/thread-pool state. Model and cache GPU allocations are still
    // explicitly released; unloading framework DLLs between models is avoided.
    _module: NonNull<c_void>,
    _directory_cookies: Vec<NonNull<c_void>>,
    last_error: LastError,
    load: Load,
    info: ModelInfo,
    model_release: Release,
    configure: Configure,
    configure_kv: Option<Configure>,
    begin_request: Option<Prepare>,
    prepare: Prepare,
    prepare_graph: Prepare,
    probe_auto: ProbeAuto,
    probe_info: ModelInfo,
    prefill: Prefill,
    extend: Option<Extend>,
    cache_info: CacheInfo,
    cache_release: Release,
    frame: Frame,
    diagnostics: ModelInfo,
}

#[cfg(windows)]
impl Api {
    fn open(path: &Path) -> Result<Rc<Self>> {
        use std::os::windows::ffi::OsStrExt;
        #[link(name = "kernel32")]
        extern "system" {
            fn LoadLibraryExW(path: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
            fn GetLastError() -> u32;
            fn AddDllDirectory(path: *const u16) -> *mut c_void;
        }
        if !path.is_absolute() || !path.is_file() {
            bail!("ATen原生库必须是存在的绝对文件路径：{}", path.display());
        }
        let mut directory_cookies = Vec::new();
        let flags = if let Some(value) = std::env::var_os("PAINTER_ATEN_RUNTIME_DIRS") {
            let directories = std::env::split_paths(&value).collect::<Vec<_>>();
            if directories.is_empty() {
                bail!("ATen运行库搜索目录不能为空");
            }
            for directory in directories {
                if !directory.is_absolute() || !directory.is_dir() {
                    bail!(
                        "ATen运行库搜索目录必须是存在的绝对目录：{}",
                        directory.display()
                    );
                }
                let directory: Vec<u16> =
                    directory.as_os_str().encode_wide().chain(Some(0)).collect();
                directory_cookies.push(
                    NonNull::new(unsafe { AddDllDirectory(directory.as_ptr()) }).with_context(
                        || {
                            format!(
                                "无法设置本进程ATen运行库目录（系统错误 {}）",
                                unsafe { GetLastError() }
                            )
                        },
                    )?,
                );
            }
            // Framework package directories precede System32; a different
            // driver-provided HIP runtime must not silently replace the pair.
            0x100 | 0x400 | 0x800
        } else {
            // Explicit developer harnesses may supply their own process PATH.
            0x8
        };
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let module =
            NonNull::new(unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), flags) })
                .with_context(|| {
                    format!(
                        "无法加载ATen原生库 {}（系统错误 {}），请检查该后端的运行库",
                        path.display(),
                        unsafe { GetLastError() }
                    )
                })?;
        unsafe fn symbol<T: Copy>(module: NonNull<c_void>, name: &'static [u8]) -> Result<T> {
            let ptr = GetProcAddress(module.as_ptr(), name.as_ptr().cast());
            if ptr.is_null() {
                bail!(
                    "ATen原生库缺少接口 {}",
                    String::from_utf8_lossy(&name[..name.len() - 1])
                );
            }
            if std::mem::size_of::<T>() != std::mem::size_of::<*mut c_void>() {
                bail!("原生接口指针尺寸不匹配");
            }
            Ok(std::mem::transmute_copy(&ptr))
        }
        let version: AbiVersion = unsafe { symbol(module, b"pp_aten_abi_version\0")? };
        if unsafe { version() } != ABI_VERSION {
            bail!("ATen后端与应用的接口版本不匹配");
        }
        unsafe {
            Ok(Rc::new(Self {
                _module: module,
                _directory_cookies: directory_cookies,
                last_error: symbol(module, b"pp_aten_last_error\0")?,
                load: symbol(module, b"pp_aten_load\0")?,
                info: symbol(module, b"pp_aten_model_info\0")?,
                model_release: symbol(module, b"pp_aten_model_release\0")?,
                configure: symbol(module, b"pp_aten_configure\0")?,
                configure_kv: symbol(module, b"pp_aten_configure_kv\0").ok(),
                begin_request: symbol(module, b"pp_aten_begin_request\0").ok(),
                prepare: symbol(module, b"pp_aten_prepare\0")?,
                prepare_graph: symbol(module, b"pp_aten_prepare_graph\0")?,
                probe_auto: symbol(module, b"pp_aten_probe_auto\0")?,
                probe_info: symbol(module, b"pp_aten_probe_info\0")?,
                prefill: symbol(module, b"pp_aten_prefill\0")?,
                extend: symbol(module, b"pp_aten_extend\0").ok(),
                cache_info: symbol(module, b"pp_aten_cache_info\0")?,
                cache_release: symbol(module, b"pp_aten_cache_release\0")?,
                frame: symbol(module, b"pp_aten_frame\0")?,
                diagnostics: symbol(module, b"pp_aten_diagnostics\0")?,
            }))
        }
    }
}
#[cfg(not(windows))]
impl Api {
    fn open(_: &Path) -> Result<Rc<Self>> {
        bail!("本ATen/HIP后端构建只支持Windows");
    }
}

fn copy_string(ptr: *const c_char) -> Result<String> {
    if ptr.is_null() {
        bail!("ATen后端返回空字符串");
    }
    Ok(unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .context("ATen后端返回非UTF8数据")?
        .to_owned())
}
fn validate_identity_unchanged(original: &Value, refreshed: &Value) -> Result<()> {
    for key in [
        "provider",
        "vendor",
        "device_ordinal",
        "device_count",
        "architecture",
        "pci_bus_id",
        "hip_version",
        "hip_driver_version",
        "torch_version",
        "dtype",
        "layers",
        "width",
        "heads",
        "ffn_hidden",
        "head_width",
        "abi_version",
        "build_id",
        "runtime_id",
    ] {
        if refreshed.get(key) != original.get(key) {
            bail!("ATen准备期间实际设备、库或模型身份发生变化：{key}");
        }
    }
    Ok(())
}

pub(super) struct Model {
    api: Rc<Api>,
    raw: NonNull<c_void>,
    pub info: Value,
    poisoned: Cell<bool>,
    restart_required: Cell<bool>,
}
pub(super) struct CacheInner {
    model: Rc<Model>,
    raw: NonNull<c_void>,
}
#[derive(Clone)]
pub(super) struct Cache {
    inner: Rc<CacheInner>,
    pub length: usize,
    pub absolute_position: usize,
    pub terms: [f32; 2],
}
pub(super) struct FrameResult {
    pub records: Vec<[u32; 9]>,
    pub caches: Vec<Cache>,
}

impl Model {
    pub fn library_path() -> Result<PathBuf> {
        if let Some(path) = std::env::var_os("PAINTER_ATEN_LIBRARY") {
            let path = PathBuf::from(path);
            if !path.is_absolute() {
                bail!("PAINTER_ATEN_LIBRARY必须为绝对路径");
            }
            return Ok(path);
        }
        let executable = std::env::current_exe()?;
        Ok(executable
            .parent()
            .context("无法定位原生应用目录")?
            .join("painter_aten_hip.dll"))
    }
    pub fn load(path: &Path, ordinal: u32, efficient: bool) -> Result<Rc<Self>> {
        #[cfg(windows)]
        let _source_read_lock = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(1)
                .open(path)
                .context("权重正在写入或不可读取，请等待保存完成后重新加载")?
        };
        let api = Api::open(&Self::library_path()?)?;
        let utf8 = path.to_str().context("权重路径不能转换为UTF8")?;
        let path = CString::new(utf8).context("权重路径含无效空字符")?;
        let options = LoadOptions {
            struct_size: std::mem::size_of::<LoadOptions>() as u32,
            device_ordinal: ordinal,
            prefer_efficient_attention: u32::from(efficient),
            reserved: 0,
            index_utf8: std::ptr::null(),
        };
        let index = crate::aten_weight_index::ensure_weight_index(Path::new(utf8))?;
        let index = CString::new(index.to_str().context("ATen索引路径不能转换为UTF8")?)?;
        let options = LoadOptions {
            index_utf8: index.as_ptr(),
            ..options
        };
        let mut output = std::ptr::null_mut();
        let status = unsafe { (api.load)(path.as_ptr(), &options, &mut output) };
        if status != 0 {
            let error = copy_string(unsafe { (api.last_error)() })
                .unwrap_or_else(|_| "未知ATen加载错误".into());
            if !output.is_null() {
                unsafe { (api.model_release)(output) }
            }
            if error.contains("CAPTURE_STATE_UNCERTAIN") {
                bail!("CAPTURE_STATE_UNCERTAIN：捕获状态不确定，需要重新启动应用；已保留完整画布：{error}");
            }
            bail!("ATen模型加载失败：{error}");
        }
        let raw = NonNull::new(output).context("ATen加载未返回模型")?;
        let info_result = copy_string(unsafe { (api.info)(raw.as_ptr()) })
            .and_then(|s| serde_json::from_str(&s).map_err(Into::into));
        let info: Value = match info_result {
            Ok(info) => info,
            Err(error) => {
                unsafe { (api.model_release)(raw.as_ptr()) };
                return Err(error);
            }
        };
        if info["provider"] != "aten_hip"
            || info["vendor"] != "amd"
            || info["dtype"] != "BF16"
            || (ordinal != u32::MAX && info["device_ordinal"].as_u64() != Some(ordinal as u64))
            || (ordinal == u32::MAX && !valid_auto_selection(&info))
            || !info["architecture"]
                .as_str()
                .is_some_and(|s| s.starts_with("gfx"))
        {
            unsafe { (api.model_release)(raw.as_ptr()) };
            bail!("ATen实际计算设备、精度或后端与所选AMD配置不一致");
        }
        Ok(Rc::new(Self {
            api,
            raw,
            info,
            poisoned: Cell::new(false),
            restart_required: Cell::new(false),
        }))
    }
    fn check(&self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let text = copy_string(unsafe { (self.api.last_error)() })
            .unwrap_or_else(|_| "ATen后端未返回详细错误".into());
        if status == 3 {
            self.poisoned.set(true);
            if text.contains("CAPTURE_STATE_UNCERTAIN") {
                self.restart_required.set(true);
                bail!("CAPTURE_STATE_UNCERTAIN：捕获状态不确定，需要重新启动应用；已保留完整画布：{text}");
            }
            bail!("ATEN_EXECUTION_UNCERTAIN：请重新加载模型；已保留完整画布：{text}");
        }
        bail!("ATen执行失败（{status}）：{text}")
    }
    fn ready(&self) -> Result<()> {
        if self.restart_required.get() {
            bail!("CAPTURE_STATE_UNCERTAIN：捕获状态不确定，需要重新启动应用；已保留完整画布");
        }
        if self.poisoned.get() {
            bail!("ATen设备执行状态不确定，请重新加载模型");
        }
        Ok(())
    }
    pub fn prefill(
        self: &Rc<Self>,
        records: &[[u32; 9]],
        text: &[u32],
        offset: usize,
    ) -> Result<Cache> {
        self.ready()?;
        let mut raw = std::ptr::null_mut();
        let mut terms = [0f32; 2];
        let code = unsafe {
            (self.api.prefill)(
                self.raw.as_ptr(),
                records.as_ptr().cast(),
                records.len(),
                text.as_ptr(),
                text.len(),
                std::ptr::null(),
                offset,
                &mut raw,
                terms.as_mut_ptr(),
            )
        };
        if code != 0 {
            let error = self
                .check(code)
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("ATen预填返回失败但未提供错误"));
            if !raw.is_null() {
                unsafe { (self.api.cache_release)(raw) }
            };
            return Err(error);
        }
        self.wrap_cache(raw, terms)
    }
    fn wrap_cache(self: &Rc<Self>, raw: *mut c_void, terms: [f32; 2]) -> Result<Cache> {
        let raw = NonNull::new(raw).context("ATen未返回有效缓存")?;
        let inner = Rc::new(CacheInner {
            model: self.clone(),
            raw,
        });
        let (mut length, mut absolute_position) = (0, 0);
        self.check(unsafe {
            (self.api.cache_info)(raw.as_ptr(), &mut length, &mut absolute_position)
        })?;
        if terms.iter().any(|v| !v.is_finite()) || length > 1024 || absolute_position > 1024 {
            bail!("ATen缓存位置或终止输出非法");
        }
        Ok(Cache {
            inner,
            length,
            absolute_position,
            terms,
        })
    }
    pub fn supports_extend(&self) -> bool {
        self.api.extend.is_some()
    }
    pub fn begin_request(&self) -> Result<()> {
        self.ready()?;
        if let Some(begin) = self.api.begin_request {
            self.check(unsafe { begin(self.raw.as_ptr()) })?;
        }
        Ok(())
    }
    pub fn extend(
        self: &Rc<Self>,
        source: &Cache,
        retained: usize,
        records: &[[u32; 9]],
    ) -> Result<Option<Cache>> {
        self.ready()?;
        let Some(extend) = self.api.extend else {
            return Ok(None);
        };
        if !Rc::ptr_eq(&source.inner.model, self)
            || retained > source.length
            || records.is_empty()
            || retained
                .checked_add(records.len())
                .is_none_or(|n| n >= 1024)
        {
            bail!("ATen增量上下文的模型、保留长度或后缀无效");
        }
        let mut raw = std::ptr::null_mut();
        let mut terms = [0f32; 2];
        let code = unsafe {
            extend(
                self.raw.as_ptr(),
                source.inner.raw.as_ptr(),
                retained,
                records.as_ptr().cast(),
                records.len(),
                &mut raw,
                terms.as_mut_ptr(),
            )
        };
        if code != 0 {
            let error = self.check(code).unwrap_err();
            if !raw.is_null() {
                unsafe { (self.api.cache_release)(raw) }
            }
            return Err(error);
        }
        let cache = self.wrap_cache(raw, terms)?;
        let offset = source
            .absolute_position
            .checked_sub(source.length)
            .context("ATen原上下文位置偏移无效")?;
        if cache.length != retained + records.len()
            || cache.absolute_position != cache.length + offset
        {
            bail!("ATen增量上下文长度或RoPE偏移不匹配");
        }
        Ok(Some(cache))
    }
    pub fn frame(
        self: &Rc<Self>,
        caches: &[&Cache],
        options: &FrameOptions,
        bounds: &[u32],
        uniforms: &[f32],
    ) -> Result<FrameResult> {
        self.ready()?;
        let logical = options.logical_rows as usize;
        let geometry = row_geometry(logical, options.guidance as usize)?;
        let physical = geometry.physical;
        if caches.len() != physical
            || bounds.len() != geometry.bounds
            || uniforms.len() != geometry.attributes
            || caches
                .iter()
                .any(|cache| !Rc::ptr_eq(&cache.inner.model, self))
        {
            bail!("ATen完整图元的缓存、边界、随机数或模型映射不一致");
        }
        let mut input = buffer(physical)?;
        input.extend(
            caches
                .iter()
                .map(|cache| cache.inner.raw.as_ptr() as *const c_void),
        );
        let mut records = filled_buffer(logical, [0u32; 9])?;
        let mut terms = filled_buffer(geometry.terms, 0f32)?;
        let mut valid = filled_buffer(logical, 0u32)?;
        let mut output = filled_buffer(physical, std::ptr::null_mut())?;
        let mut wrapped = buffer(physical)?;
        let code = unsafe {
            (self.api.frame)(
                self.raw.as_ptr(),
                input.as_ptr(),
                options,
                bounds.as_ptr(),
                uniforms.as_ptr(),
                records.as_mut_ptr().cast(),
                terms.as_mut_ptr(),
                valid.as_mut_ptr(),
                output.as_mut_ptr(),
            )
        };
        if code != 0 {
            let error = self
                .check(code)
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("ATen完整图元返回失败但未提供错误"));
            for ptr in output {
                if !ptr.is_null() {
                    unsafe { (self.api.cache_release)(ptr) }
                }
            }
            return Err(error);
        }
        let result = (|| -> Result<()> {
            if valid.iter().any(|&v| v != 1) || terms.iter().any(|v| !v.is_finite()) {
                bail!("ATen图元采样或终止输出出现非有限值");
            }
            for index in 0..physical {
                let ptr = std::mem::replace(&mut output[index], std::ptr::null_mut());
                let cache = self.wrap_cache(ptr, [terms[index * 2], terms[index * 2 + 1]])?;
                if cache.length != caches[index].length + 1
                    || cache.absolute_position != caches[index].absolute_position + 1
                {
                    bail!("ATen完整图元没有按预期提交恰好一个缓存位置");
                }
                wrapped.push(cache);
            }
            Ok(())
        })();
        for ptr in output {
            if !ptr.is_null() {
                unsafe { (self.api.cache_release)(ptr) }
            }
        }
        result?;
        Ok(FrameResult {
            records,
            caches: wrapped,
        })
    }
    pub fn diagnostics(&self) -> Result<Value> {
        serde_json::from_str(&copy_string(unsafe {
            (self.api.diagnostics)(self.raw.as_ptr())
        })?)
        .map_err(Into::into)
    }

    pub fn configure_prefill(&self, efficient: bool) -> Result<()> {
        self.ready()?;
        self.check(unsafe { (self.api.configure)(self.raw.as_ptr(), u32::from(efficient)) })
    }
    pub fn configure_kv(&self, preference: u32) -> Result<Value> {
        self.ready()?;
        if preference > 2 {
            bail!("INT8 KV原生偏好值无效");
        }
        if let Some(configure) = self.api.configure_kv {
            self.check(unsafe { configure(self.raw.as_ptr(), preference) })?;
            let refreshed: Value =
                serde_json::from_str(&copy_string(unsafe { (self.api.info)(self.raw.as_ptr()) })?)?;
            validate_identity_unchanged(&self.info, &refreshed)?;
            return Ok(refreshed);
        }
        let mut info = self.info.clone();
        info["kv_int8"] = serde_json::json!({"requested":match preference {0=>"auto",1=>"enabled",_=>"disabled"},
            "available":false,"enabled":false,"storage":"bf16",
            "reason":"当前原生DLL没有INT8 KV扩展接口；保留同GPU BF16历史缓存"});
        Ok(info)
    }

    pub fn prepare_optional(&self) -> Result<Value> {
        self.ready()?;
        self.check(unsafe { (self.api.prepare)(self.raw.as_ptr()) })?;
        let refreshed: Value =
            serde_json::from_str(&copy_string(unsafe { (self.api.info)(self.raw.as_ptr()) })?)?;
        if let Err(error) = validate_identity_unchanged(&self.info, &refreshed) {
            self.poisoned.set(true);
            bail!("ATEN_EXECUTION_UNCERTAIN：请重新加载模型：{error}");
        }
        Ok(refreshed)
    }
    pub fn prepare_graph(&self) -> Result<Value> {
        self.ready()?;
        self.check(unsafe { (self.api.prepare_graph)(self.raw.as_ptr()) })?;
        let refreshed: Value =
            serde_json::from_str(&copy_string(unsafe { (self.api.info)(self.raw.as_ptr()) })?)?;
        if let Err(error) = validate_identity_unchanged(&self.info, &refreshed) {
            self.poisoned.set(true);
            bail!("ATEN_EXECUTION_UNCERTAIN：请重新加载模型：{error}");
        }
        Ok(refreshed)
    }
    pub fn probe_auto(
        &self,
        logical_rows: usize,
        guidance: usize,
        regional: bool,
        cfg: f64,
        alpha: f64,
        efficient: bool,
        gpu_sampling: bool,
        prefix_records: usize,
    ) -> Result<Value> {
        self.ready()?;
        let geometry = row_geometry(logical_rows, guidance)?;
        let options = AutoProbeOptions {
            struct_size: std::mem::size_of::<AutoProbeOptions>() as u32,
            logical_rows: geometry.logical,
            guidance: u32::try_from(guidance)?,
            regional: u32::from(regional),
            candidate_efficient: u32::from(efficient),
            candidate_gpu_sampling: u32::from(gpu_sampling),
            prefix_records: u32::try_from(prefix_records)?,
            reserved: 0,
            cfg: if guidance == 1 { 1. } else { cfg },
            alpha,
        };
        self.check(unsafe { (self.api.probe_auto)(self.raw.as_ptr(), &options) })?;
        serde_json::from_str(&copy_string(unsafe {
            (self.api.probe_info)(self.raw.as_ptr())
        })?)
        .context("ATen私有适配返回的诊断JSON无效")
    }
}

fn valid_auto_selection(info: &Value) -> bool {
    let Some((count, ordinal)) = info["device_count"]
        .as_u64()
        .zip(info["device_ordinal"].as_u64())
    else {
        return false;
    };
    count > 0
        && ordinal < count
        && info["auto_device_selection"].as_bool() == Some(true)
        && info["device_candidates"]
            .as_array()
            .is_some_and(|candidates| {
                candidates.iter().any(|candidate| {
                    candidate["ordinal"].as_u64() == Some(ordinal)
                        && candidate["properties_status"].as_i64() == Some(0)
                        && candidate["amd_gfx"].as_bool() == Some(true)
                })
            })
}
impl Drop for CacheInner {
    fn drop(&mut self) {
        unsafe { (self.model.api.cache_release)(self.raw.as_ptr()) }
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        unsafe { (self.api.model_release)(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ffi_layout_and_frame_constraints_follow_the_versioned_c_header() {
        assert_eq!(std::mem::size_of::<LoadOptions>(), 24);
        assert_eq!(std::mem::size_of::<FrameOptions>(), 40);
        assert_eq!(std::mem::size_of::<AutoProbeOptions>(), 48);
        assert_eq!(std::mem::align_of::<FrameOptions>(), 8);
        let options = FrameOptions::new(8, 2, 2.5, 3.5, true, false, true).unwrap();
        assert_eq!(options.graph_allowed, 0);
        assert!(FrameOptions::new(0, 1, 1., 4., true, true, true).is_err());
        assert!(FrameOptions::new(8, 3, 1., 4., true, true, true).is_err());
    }
    #[test]
    fn parallel_rows_above_sixteen_preserve_logical_and_cfg_geometry() {
        for rows in [17, 32, 96, 240, 1024] {
            for guidance in [1, 2] {
                let geometry = row_geometry(rows, guidance).unwrap();
                assert_eq!(geometry.logical as usize, rows);
                assert_eq!(geometry.physical, rows * guidance);
                assert_eq!(geometry.attributes, rows * 9);
                assert_eq!(geometry.bounds, rows * 18);
                assert_eq!(geometry.terms, rows * guidance * 2);
                let options = FrameOptions::new(rows, guidance, 1., 4., false, true, true).unwrap();
                assert_eq!(options.logical_rows as usize, rows);
            }
        }
        assert!(row_geometry(usize::MAX, 2).is_err());
        if usize::BITS > 32 {
            assert!(row_geometry(u32::MAX as usize + 1, 1).is_err());
        }
        assert!(buffer::<u64>(usize::MAX).is_err());
    }
    #[test]
    fn auto_device_identity_accepts_actual_discrete_hip_selection_with_an_apu_present() {
        let info = serde_json::json!({"device_count":2,"device_ordinal":1,"auto_device_selection":true,
            "device_candidates":[{"ordinal":0,"properties_status":0,"amd_gfx":true,"integrated":true},
                {"ordinal":1,"properties_status":0,"amd_gfx":true,"integrated":false}]});
        assert!(valid_auto_selection(&info));
        for value in [
            serde_json::json!({"device_ordinal":5}),
            serde_json::json!({"auto_device_selection":false}),
            serde_json::json!({"device_candidates":[]}),
        ] {
            let mut changed = info.clone();
            for (key, value) in value.as_object().unwrap() {
                changed[key] = value.clone();
            }
            assert!(!valid_auto_selection(&changed));
        }
    }
    #[test]
    fn lazy_preparation_may_refresh_capabilities_but_never_switch_identity() {
        let original = serde_json::json!({"provider":"aten_hip","vendor":"amd","architecture":"gfx1100",
            "device_ordinal":0,"runtime_id":"runtime-v1","hip_version":"701","dtype":"BF16","width":1536,
            "efficient_attention_available":false,"loaded_libraries":{"hip_runtime":"E:/old/place/amdhip64_7.dll"}});
        let mut refreshed = original.clone();
        refreshed["efficient_attention_available"] = true.into();
        refreshed["loaded_libraries"] =
            serde_json::json!({"hip_runtime":"E:/validated/new/place/amdhip64_7.dll"});
        validate_identity_unchanged(&original, &refreshed).unwrap();
        for key in [
            "provider",
            "architecture",
            "device_ordinal",
            "runtime_id",
            "hip_version",
            "dtype",
            "width",
        ] {
            let mut different = refreshed.clone();
            different[key] = "changed".into();
            assert!(validate_identity_unchanged(&original, &different).is_err());
        }
    }
}
