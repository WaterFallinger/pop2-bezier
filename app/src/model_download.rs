//! Pinned public model download. No Python, credentials, or GPU work is used.
use eframe::egui;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};

pub const REPO: &str = "https://huggingface.co/doubixz/primitive-operation-painter-2-1b";
const REVISION: &str = "ca575269a461c90ecf630dcf70c4beaaf319b3a1";
#[derive(Clone, Copy)]
struct Asset {
    name: &'static str,
    bytes: u64,
    sha: &'static str,
}
const MODEL: Asset = Asset {
    name: "model.pt",
    bytes: 2_008_472_552,
    sha: "143f447aaecf9be75c6d14f3a3170002e94e49ff70a758f66d568d27ff65e9a0",
};
const TOKENIZER: Asset = Asset {
    name: "caption_spm_1500.model",
    bytes: 261326,
    sha: "8dc18fe4e13f2058807ed658bfed93461c296da9a7818f0224452b73a797497e",
};
fn default_directory() -> PathBuf {
    crate::resources::user_config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("models/pop2-1b-epoch60")
}
#[derive(Clone, Debug)]
pub struct Downloaded {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}
#[derive(Clone, Debug)]
enum Event {
    Progress(String, u64, u64),
    Complete(Downloaded),
    Failed(String),
}
struct Job {
    receiver: mpsc::Receiver<Event>,
    cancel: Arc<AtomicBool>,
}
impl Drop for Job {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
#[derive(Default)]
pub struct DownloadUi {
    directory: String,
    job: Option<Job>,
    progress: Option<(String, u64, u64)>,
    error: Option<String>,
    ready: Option<Downloaded>,
    stopping: bool,
}
impl DownloadUi {
    pub fn poll(&mut self, idle: bool) -> Option<Downloaded> {
        if self.directory.is_empty() {
            self.directory = default_directory().display().to_string();
        }
        if let Some(job) = &self.job {
            let mut events = Vec::new();
            let mut disconnected = false;
            loop {
                match job.receiver.try_recv() {
                    Ok(event) => events.push(event),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
            for event in events {
                match event {
                    Event::Progress(text, n, total) => self.progress = Some((text, n, total)),
                    Event::Complete(paths) => {
                        self.ready = Some(paths);
                        self.job = None;
                        self.stopping = false;
                    }
                    Event::Failed(error) => {
                        self.error = Some(error);
                        self.job = None;
                        self.stopping = false;
                    }
                }
            }
            if self.job.is_some() && disconnected {
                self.job = None;
                self.error = Some("下载任务意外结束，断点已保留，请重试。".into());
                self.stopping = false;
            }
        }
        if idle {
            self.ready.take()
        } else {
            None
        }
    }
    pub fn show(&mut self, ui: &mut egui::Ui, idle: bool) {
        if self.directory.is_empty() {
            self.directory = default_directory().display().to_string();
        }
        ui.label(egui::RichText::new("在线下载 · POP 2 / 1B BF16").strong());
        ui.label("约 2.01 GB · Epoch 60 · 不需要登录，下载完成后自动加载");
        ui.add_enabled(
            self.job.is_none(),
            egui::TextEdit::singleline(&mut self.directory)
                .desired_width(f32::INFINITY)
                .hint_text("权重保存文件夹"),
        );
        if let Some((text, n, total)) = &self.progress {
            ui.label(text);
            ui.add(
                egui::ProgressBar::new(*n as f32 / (*total).max(1) as f32).text(format!(
                    "{:.1} / {:.1} MB",
                    *n as f64 / 1e6,
                    *total as f64 / 1e6
                )),
            );
        }
        if self.job.is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(150));
            if ui
                .add_enabled(
                    !self.stopping,
                    egui::Button::new(if self.stopping {
                        "正在暂停…"
                    } else {
                        "暂停下载（保留断点）"
                    }),
                )
                .clicked()
            {
                if let Some(job) = &self.job {
                    job.cancel.store(true, Ordering::Relaxed);
                }
                self.stopping = true;
            }
        } else if self.ready.is_none()
            && ui
                .add_enabled(idle, egui::Button::new("下载 / 继续下载并加载"))
                .on_disabled_hover_text("请先等待模型停止生成或加载完成")
                .clicked()
        {
            self.error = None;
            self.progress = None;
            self.stopping = false;
            let directory = PathBuf::from(self.directory.trim());
            if !directory.is_absolute() {
                self.error =
                    Some("请填写完整的绝对文件夹路径，例如 D:\\PrimitivePainterModels。".into());
            } else {
                self.job = Some(start(directory, ui.ctx().clone()));
            }
        }
        ui.hyperlink_to("查看模型介绍与手动下载", REPO);
        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(171, 66, 53), error);
        }
        if self.ready.is_some() && !idle {
            ui.label("下载校验完成，等待当前生成或加载任务结束后自动配置。");
        }
    }
}
fn start(directory: PathBuf, ctx: egui::Context) -> Job {
    let (tx, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    std::thread::spawn(move || {
        let progress = |text: String, n, total| {
            let _ = tx.send(Event::Progress(text, n, total));
            ctx.request_repaint();
        };
        crate::diagnostics::event(
            "model_download_started",
            serde_json::json!({"repository":REPO,"revision":REVISION,"directory":directory}),
        );
        let result = download_package(&directory, &stop, &progress);
        match result {
            Ok(paths) => {
                crate::diagnostics::event(
                    "model_download_complete",
                    serde_json::json!({"revision":REVISION,"model":paths.model}),
                );
                let _ = tx.send(Event::Complete(paths));
            }
            Err(error) => {
                crate::diagnostics::event(
                    "model_download_failed",
                    serde_json::json!({"error":error}),
                );
                let _ = tx.send(Event::Failed(error));
            }
        }
        ctx.request_repaint();
    });
    Job { receiver, cancel }
}
fn check_cancel(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        Err("下载已暂停，断点已保留；点击继续即可恢复。".into())
    } else {
        Ok(())
    }
}
fn validate_file(
    path: &Path,
    asset: Asset,
    cancel: &AtomicBool,
    progress: &impl Fn(String, u64, u64),
) -> Result<bool, String> {
    if !path.is_file() {
        return Ok(false);
    }
    let mut f = File::open(path).map_err(|e| format!("无法读取下载文件：{e}"))?;
    if f.metadata().map_err(|e| e.to_string())?.len() != asset.bytes {
        return Ok(false);
    }
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    let mut n = 0;
    let mut time = Instant::now();
    loop {
        check_cancel(cancel)?;
        let read = f
            .read(&mut buffer)
            .map_err(|e| format!("校验读取失败：{e}"))?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
        n += read as u64;
        if time.elapsed() > Duration::from_millis(150) {
            progress(format!("校验 {}", asset.name), n, asset.bytes);
            time = Instant::now();
        }
    }
    Ok(format!("{:x}", hash.finalize()) == asset.sha)
}
struct Response {
    status: u32,
    range: Option<String>,
    body: Box<dyn Read>,
}
fn resume_offset(status: u32, range: Option<&str>, offset: u64, total: u64) -> Result<u64, String> {
    if status == 200 {
        return Ok(0);
    }
    if status == 206 && range == Some(format!("bytes {offset}-{}/{total}", total - 1).as_str()) {
        return Ok(offset);
    }
    Err(format!(
        "服务器响应不符合下载断点（HTTP {status}），断点已保留，请稍后重试。"
    ))
}
fn download_one(
    asset: Asset,
    directory: &Path,
    cancel: &AtomicBool,
    progress: &impl Fn(String, u64, u64),
    mut open: impl FnMut(u64) -> Result<Response, String>,
) -> Result<PathBuf, String> {
    let destination = directory.join(asset.name);
    let part = directory.join(format!("{}.part", asset.name));
    if validate_file(&destination, asset, cancel, progress)? {
        progress(
            format!("{} 已存在且校验通过", asset.name),
            asset.bytes,
            asset.bytes,
        );
        return Ok(destination);
    }
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&part)
        .map_err(|e| format!("无法写入下载目录：{e}"))?;
    let mut offset = file.metadata().map_err(|e| e.to_string())?.len();
    if offset > asset.bytes {
        file.set_len(0).map_err(|e| e.to_string())?;
        offset = 0;
    }
    if offset < asset.bytes {
        check_cancel(cancel)?;
        progress(format!("连接下载 {}", asset.name), offset, asset.bytes);
        let mut response = open(offset)?;
        offset = resume_offset(
            response.status,
            response.range.as_deref(),
            offset,
            asset.bytes,
        )?;
        file.set_len(offset).map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| e.to_string())?;
        let mut buffer = vec![0; 256 * 1024];
        let mut time = Instant::now();
        loop {
            check_cancel(cancel)?;
            let n = response
                .body
                .read(&mut buffer)
                .map_err(|e| format!("网络读取失败，已保留断点，可继续下载：{e}"))?;
            if n == 0 {
                break;
            }
            if offset + n as u64 > asset.bytes {
                return Err("下载数据超过预期大小，未启用该文件。".into());
            }
            file.write_all(&buffer[..n])
                .map_err(|e| format!("磁盘写入失败，断点已保留，请检查剩余空间：{e}"))?;
            offset += n as u64;
            if time.elapsed() > Duration::from_millis(150) {
                progress(format!("下载 {}", asset.name), offset, asset.bytes);
                time = Instant::now();
            }
        }
        file.sync_all().map_err(|e| e.to_string())?;
    }
    drop(file);
    if !validate_file(&part, asset, cancel, progress)? {
        // Only reset our owned partial file. Never remove a user's existing model.
        if fs::metadata(&part)
            .map(|m| m.len() == asset.bytes)
            .unwrap_or(false)
        {
            OpenOptions::new()
                .write(true)
                .open(&part)
                .and_then(|f| f.set_len(0))
                .map_err(|e| e.to_string())?;
        }
        return Err(format!(
            "{} 长度或 SHA-256 校验未通过，未启用该文件；请重试下载。",
            asset.name
        ));
    }
    check_cancel(cancel)?;
    crate::resources::replace_file(&part, &destination)
        .map_err(|e| format!("无法完成下载文件，请确认文件没有被其他程序占用：{e}"))?;
    progress(format!("{} 校验通过", asset.name), asset.bytes, asset.bytes);
    Ok(destination)
}
fn download_package(
    directory: &Path,
    cancel: &AtomicBool,
    progress: &impl Fn(String, u64, u64),
) -> Result<Downloaded, String> {
    fs::create_dir_all(directory).map_err(|e| format!("无法创建模型文件夹：{e}"))?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(directory.join(".download.lock"))
        .map_err(|e| e.to_string())?;
    lock.try_lock()
        .map_err(|_| "另一个窗口正在向此目录下载，请等待它完成或选择不同目录。".to_owned())?;
    let get = |asset| {
        download_one(asset, directory, cancel, progress, |offset| {
            network::open(asset, offset)
        })
    };
    let model = get(MODEL)?;
    let tokenizer = get(TOKENIZER)?;
    Ok(Downloaded { model, tokenizer })
}

#[cfg(windows)]
mod network {
    use super::*;
    use std::{ffi::c_void, ptr};
    type H = *mut c_void;
    #[link(name = "winhttp")]
    unsafe extern "system" {
        fn WinHttpOpen(
            agent: *const u16,
            access: u32,
            proxy: *const u16,
            bypass: *const u16,
            flags: u32,
        ) -> H;
        fn WinHttpSetTimeouts(h: H, resolve: i32, connect: i32, send: i32, receive: i32) -> i32;
        fn WinHttpConnect(h: H, host: *const u16, port: u16, reserved: u32) -> H;
        fn WinHttpOpenRequest(
            h: H,
            verb: *const u16,
            path: *const u16,
            version: *const u16,
            referer: *const u16,
            accept: *const *const u16,
            flags: u32,
        ) -> H;
        fn WinHttpSetOption(h: H, option: u32, value: *const c_void, len: u32) -> i32;
        fn WinHttpSendRequest(
            h: H,
            headers: *const u16,
            len: u32,
            data: *const c_void,
            data_len: u32,
            total: u32,
            context: usize,
        ) -> i32;
        fn WinHttpReceiveResponse(h: H, reserved: *const c_void) -> i32;
        fn WinHttpQueryHeaders(
            h: H,
            level: u32,
            name: *const u16,
            buffer: *mut c_void,
            len: *mut u32,
            index: *mut u32,
        ) -> i32;
        fn WinHttpReadData(h: H, buffer: *mut c_void, size: u32, read: *mut u32) -> i32;
        fn WinHttpCloseHandle(h: H) -> i32;
    }
    struct Handle(H);
    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe {
                WinHttpCloseHandle(self.0);
            }
        }
    }
    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(Some(0)).collect()
    }
    fn handle(h: H) -> Result<Handle, String> {
        if h.is_null() {
            Err(format!(
                "Windows HTTPS 请求失败：{}",
                std::io::Error::last_os_error()
            ))
        } else {
            Ok(Handle(h))
        }
    }
    fn ok(n: i32) -> Result<(), String> {
        if n == 0 {
            Err(format!(
                "HTTPS 下载失败，请检查网络或代理后重试（断点已保留）：{}",
                std::io::Error::last_os_error()
            ))
        } else {
            Ok(())
        }
    }
    struct Stream {
        request: Handle,
        _connection: Handle,
        _session: Handle,
    }
    impl Read for Stream {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let mut n = 0;
            let r = unsafe {
                WinHttpReadData(
                    self.request.0,
                    buffer.as_mut_ptr().cast(),
                    buffer.len().min(u32::MAX as usize) as u32,
                    &mut n,
                )
            };
            if r == 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(n as usize)
            }
        }
    }
    pub(super) fn open(asset: Asset, offset: u64) -> Result<Response, String> {
        unsafe {
            let session = handle(WinHttpOpen(
                wide("PrimitivePainter/0.9.28").as_ptr(),
                4,
                ptr::null(),
                ptr::null(),
                0,
            ))?;
            ok(WinHttpSetTimeouts(session.0, 15000, 15000, 15000, 15000))?;
            let connection = handle(WinHttpConnect(
                session.0,
                wide("huggingface.co").as_ptr(),
                443,
                0,
            ))?;
            let path = format!(
                "/doubixz/primitive-operation-painter-2-1b/resolve/{REVISION}/{}",
                asset.name
            );
            let request = handle(WinHttpOpenRequest(
                connection.0,
                wide("GET").as_ptr(),
                wide(&path).as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0x00800000,
            ))?;
            // Permit CDN redirects, but never HTTPS -> HTTP. Default certificate verification remains enabled.
            let policy: u32 = 1;
            ok(WinHttpSetOption(
                request.0,
                88,
                (&policy as *const u32).cast(),
                4,
            ))?;
            let headers = wide(&format!(
                "Accept-Encoding: identity\r\nRange: bytes={offset}-\r\n"
            ));
            ok(WinHttpSendRequest(
                request.0,
                headers.as_ptr(),
                (headers.len() - 1) as u32,
                ptr::null(),
                0,
                0,
                0,
            ))?;
            ok(WinHttpReceiveResponse(request.0, ptr::null()))?;
            let mut status = 0u32;
            let mut len = 4;
            ok(WinHttpQueryHeaders(
                request.0,
                19 | 0x20000000,
                ptr::null(),
                (&mut status as *mut u32).cast(),
                &mut len,
                ptr::null_mut(),
            ))?;
            let mut buffer = [0u16; 256];
            let mut len = 512;
            let name = wide("Content-Range");
            let range = if WinHttpQueryHeaders(
                request.0,
                65535,
                name.as_ptr(),
                buffer.as_mut_ptr().cast(),
                &mut len,
                ptr::null_mut(),
            ) != 0
            {
                Some(
                    String::from_utf16_lossy(&buffer[..(len as usize / 2)])
                        .trim_end_matches('\0')
                        .to_owned(),
                )
            } else {
                None
            };
            Ok(Response {
                status,
                range,
                body: Box::new(Stream {
                    request,
                    _connection: connection,
                    _session: session,
                }),
            })
        }
    }
}
#[cfg(not(windows))]
mod network {
    use super::*;
    pub(super) fn open(_: Asset, _: u64) -> Result<Response, String> {
        Err("当前在线下载功能仅支持 Windows。".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    const BYTES: &[u8] = b"a native model download fixture";
    fn asset() -> Asset {
        Asset {
            name: "model.pt",
            bytes: BYTES.len() as u64,
            sha: "aa",
        }
    }
    fn fixture() -> (PathBuf, Asset) {
        let dir = std::env::temp_dir().join(format!(
            "painter-download-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let sha = Box::leak(format!("{:x}", Sha256::digest(BYTES)).into_boxed_str());
        (dir, Asset { sha, ..asset() })
    }
    fn response(offset: u64) -> Response {
        Response {
            status: 206,
            range: Some(format!(
                "bytes {offset}-{}/{}",
                BYTES.len() - 1,
                BYTES.len()
            )),
            body: Box::new(Cursor::new(BYTES[offset as usize..].to_vec())),
        }
    }
    #[test]
    fn interrupted_transfer_resumes_and_only_verified_file_becomes_visible() {
        let (dir, a) = fixture();
        let stop = AtomicBool::new(false);
        let result = download_one(a, &dir, &stop, &|_, _, _| {}, |_| {
            Ok(Response {
                status: 200,
                range: None,
                body: Box::new(Cursor::new(BYTES[..9].to_vec())),
            })
        });
        assert!(result.is_err());
        assert!(!dir.join(a.name).exists());
        assert_eq!(fs::metadata(dir.join("model.pt.part")).unwrap().len(), 9);
        let p = download_one(a, &dir, &stop, &|_, _, _| {}, |offset| {
            assert_eq!(offset, 9);
            Ok(response(offset))
        })
        .unwrap();
        assert_eq!(fs::read(&p).unwrap(), BYTES);
        assert!(!dir.join("model.pt.part").exists());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn range_ignored_safely_restarts_and_bad_range_keeps_partial() {
        let (dir, a) = fixture();
        fs::write(dir.join("model.pt.part"), &BYTES[..6]).unwrap();
        let stop = AtomicBool::new(false);
        assert!(download_one(a, &dir, &stop, &|_, _, _| {}, |_| Ok(response(0))).is_err());
        assert_eq!(fs::read(dir.join("model.pt.part")).unwrap(), &BYTES[..6]);
        let p = download_one(a, &dir, &stop, &|_, _, _| {}, |_| {
            Ok(Response {
                status: 200,
                range: None,
                body: Box::new(Cursor::new(BYTES)),
            })
        })
        .unwrap();
        assert_eq!(fs::read(p).unwrap(), BYTES);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn corruption_does_not_replace_existing_model_and_retry_recovers() {
        let (dir, a) = fixture();
        fs::write(dir.join(a.name), b"existing user model").unwrap();
        let stop = AtomicBool::new(false);
        assert!(
            download_one(a, &dir, &stop, &|_, _, _| {}, |_| Ok(Response {
                status: 200,
                range: None,
                body: Box::new(Cursor::new(vec![0; BYTES.len()]))
            }))
            .is_err()
        );
        assert_eq!(fs::read(dir.join(a.name)).unwrap(), b"existing user model");
        assert_eq!(fs::metadata(dir.join("model.pt.part")).unwrap().len(), 0);
        download_one(a, &dir, &stop, &|_, _, _| {}, |n| Ok(response(n))).unwrap();
        assert_eq!(fs::read(dir.join(a.name)).unwrap(), BYTES);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn pause_retains_partial_and_valid_existing_file_avoids_network() {
        let (dir, a) = fixture();
        let stop = AtomicBool::new(true);
        fs::write(dir.join("model.pt.part"), &BYTES[..4]).unwrap();
        assert!(download_one(a, &dir, &stop, &|_, _, _| {}, |_| panic!("cancelled")).is_err());
        assert_eq!(fs::read(dir.join("model.pt.part")).unwrap(), &BYTES[..4]);
        stop.store(false, Ordering::Relaxed);
        fs::write(dir.join(a.name), BYTES).unwrap();
        download_one(a, &dir, &stop, &|_, _, _| {}, |_| panic!("already valid")).unwrap();
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn response_validation_and_overlong_body_never_publish() {
        assert!(resume_offset(403, None, 0, 20).is_err());
        assert!(resume_offset(206, Some("bytes 4-19/21"), 4, 20).is_err());
        let (dir, a) = fixture();
        assert!(
            download_one(a, &dir, &AtomicBool::new(false), &|_, _, _| {}, |_| Ok(
                Response {
                    status: 200,
                    range: None,
                    body: Box::new(Cursor::new(vec![1; 100]))
                }
            ))
            .is_err()
        );
        assert!(!dir.join(a.name).exists());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn ui_waits_for_idle_and_disconnect_does_not_drop_completion() {
        let (tx, rx) = mpsc::channel();
        let mut ui = DownloadUi::default();
        let stop = Arc::new(AtomicBool::new(false));
        ui.job = Some(Job {
            receiver: rx,
            cancel: stop,
        });
        tx.send(Event::Complete(Downloaded {
            model: "model.pt".into(),
            tokenizer: "tokenizer.model".into(),
        }))
        .unwrap();
        drop(tx);
        assert!(ui.poll(false).is_none());
        assert!(ui.ready.is_some());
        assert!(ui.error.is_none());
        assert!(ui.poll(true).is_some());
        assert!(ui.poll(true).is_none());
        let (tx, rx) = mpsc::channel();
        ui.job = Some(Job {
            receiver: rx,
            cancel: Arc::new(AtomicBool::new(false)),
        });
        drop(tx);
        ui.poll(true);
        assert!(ui.error.as_ref().unwrap().contains("意外结束"));
    }
    #[test]
    fn ui_download_completion_does_not_modify_canvas_or_generate() {
        let ctx = egui::Context::default();
        let mut ui = DownloadUi::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |egui| {
                ui.show(egui, false);
            });
        });
        assert!(ui.job.is_none());
        assert!(ui.ready.is_none());
    }
    #[test]
    #[ignore = "Uses official public HTTPS; explicitly run during download release acceptance"]
    fn official_https_tokenizer_and_model_resume_range() {
        let response = network::open(TOKENIZER, 0).unwrap();
        assert_eq!(response.status, 206);
        assert_eq!(
            resume_offset(
                response.status,
                response.range.as_deref(),
                0,
                TOKENIZER.bytes
            )
            .unwrap(),
            0
        );
        let mut bytes = Vec::new();
        response
            .body
            .take(TOKENIZER.bytes + 1)
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(bytes.len() as u64, TOKENIZER.bytes);
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), TOKENIZER.sha);
        let offset = MODEL.bytes - 64;
        let response = network::open(MODEL, offset).unwrap();
        assert_eq!(
            resume_offset(
                response.status,
                response.range.as_deref(),
                offset,
                MODEL.bytes
            )
            .unwrap(),
            offset
        );
        let mut bytes = Vec::new();
        response.body.take(65).read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 64);
    }
}
