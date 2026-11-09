//! Local session diagnostics shared by the launcher and every provider flavor.
//! Logging failure never changes model/device selection or aborts inference.
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

static GLOBAL: OnceLock<Arc<SessionLog>> = OnceLock::new();
const MAX_RECENT: usize = 256;
const SESSION_ENV: &str = "PAINTER_DIAGNOSTIC_SESSION";
const DIRECTORY_ENV: &str = "PAINTER_DIAGNOSTIC_DIR";

#[derive(Clone)]
struct Privacy {
    replacements: Vec<(String, String)>,
}
impl Privacy {
    fn current() -> Self {
        let mut replacements = Vec::new();
        if let Some(path) = std::env::var_os("USERPROFILE").filter(|v| !v.is_empty()) {
            let path = path.to_string_lossy().into_owned();
            replacements.push((path.clone(), "%USERPROFILE%".into()));
            replacements.push((path.replace('\\', "/"), "%USERPROFILE%".into()));
        }
        Self { replacements }
    }
    fn text(&self, input: &str) -> String {
        if input.contains("PRIVATE KEY-----") {
            return "[private key omitted]".into();
        }
        if input.contains("提示词中有模型文字词表不支持") || input.contains("unsupported prompt")
        {
            return "提示词验证失败（原始内容已省略）".into();
        }
        let mut text = input.chars().take(8192).collect::<String>();
        for (from, to) in &self.replacements {
            text = replace_ascii_case(&text, from, to);
        }
        text = redact_urls(&text);
        for marker in [
            "bearer ",
            "password=",
            "passwd=",
            "api_key=",
            "access_token=",
            "refresh_token=",
            "authorization:",
        ] {
            text = redact_after(&text, marker);
        }
        text
    }
    fn value(&self, input: &Value, depth: usize) -> Value {
        if depth > 10 {
            return Value::String("[nested data omitted]".into());
        }
        match input {
            Value::String(value) => Value::String(self.text(value)),
            Value::Array(values) => Value::Array(
                values
                    .iter()
                    .take(64)
                    .map(|v| self.value(v, depth + 1))
                    .chain((values.len() > 64).then(|| {
                        Value::String(format!("[{} additional items omitted]", values.len() - 64))
                    }))
                    .collect(),
            ),
            Value::Object(values) => {
                let mut output = serde_json::Map::new();
                for (key, value) in values.iter().take(128) {
                    if sensitive_key(key, value) {
                        output.insert(key.clone(), Value::String("[omitted]".into()));
                    } else {
                        output.insert(key.clone(), self.value(value, depth + 1));
                    }
                }
                Value::Object(output)
            }
            value => value.clone(),
        }
    }
}
fn sensitive_key(key: &str, value: &Value) -> bool {
    let key = key
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    if (key.ends_with("count")
        || key.ends_with("length")
        || key.ends_with("bytes")
        || key.ends_with("tokens")
        || key.ends_with("primitives")
        || key.ends_with("records"))
        && value.is_number()
    {
        return false;
    }
    [
        "prompt",
        "prompttext",
        "caption",
        "captiontext",
        "inputtext",
        "rawtext",
        "inputids",
        "textids",
        "tokenids",
        "tokens",
        "records",
        "primitives",
        "password",
        "passwd",
        "secret",
        "authorization",
        "apikey",
        "accesstoken",
        "refreshtoken",
        "privatekey",
        "hftoken",
        "credentials",
        "cookie",
        "weights",
        "weightdata",
        "rawweights",
        "statedict",
        "modelstatedict",
        "tensorvalues",
    ]
    .iter()
    .any(|name| key == *name || key.ends_with(name))
}
fn find_ascii_case(text: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    text.as_bytes()
        .windows(needle.len())
        .position(|v| v.eq_ignore_ascii_case(needle.as_bytes()))
}
fn replace_ascii_case(input: &str, from: &str, to: &str) -> String {
    let mut remaining = input;
    let mut out = String::new();
    while let Some(at) = find_ascii_case(remaining, from) {
        out.push_str(&remaining[..at]);
        out.push_str(to);
        remaining = &remaining[at + from.len()..];
    }
    out.push_str(remaining);
    out
}
fn redact_after(input: &str, marker: &str) -> String {
    let mut remaining = input;
    let mut out = String::new();
    while let Some(at) = find_ascii_case(remaining, marker) {
        let start = at + marker.len();
        out.push_str(&remaining[..start]);
        out.push_str("[redacted]");
        let value = &remaining[start..];
        let end = value
            .char_indices()
            .find(|(_, c)| c.is_whitespace() || matches!(c, '&' | ';' | '"' | '\'' | '\r' | '\n'))
            .map_or(value.len(), |(at, _)| at);
        remaining = &value[end..];
    }
    out.push_str(remaining);
    out
}
fn redact_urls(input: &str) -> String {
    let mut output = String::new();
    let mut remaining = input;
    loop {
        let at = [remaining.find("https://"), remaining.find("http://")]
            .into_iter()
            .flatten()
            .min();
        let Some(at) = at else {
            output.push_str(remaining);
            break;
        };
        output.push_str(&remaining[..at]);
        let url = &remaining[at..];
        let end = url
            .char_indices()
            .find(|(_, c)| c.is_whitespace() || matches!(c, '"' | '\'' | '>' | ')'))
            .map_or(url.len(), |(at, _)| at);
        let raw = &url[..end];
        let scheme = raw.find("://").unwrap() + 3;
        let path_end = raw.find(['?', '#']).unwrap_or(raw.len());
        let mut safe = raw[..path_end].to_string();
        if let Some(at) = safe[scheme..].find('@') {
            safe.replace_range(scheme..scheme + at + 1, "[credentials]@");
        }
        output.push_str(&safe);
        if path_end < raw.len() {
            output.push_str("?[redacted]");
        }
        remaining = &url[end..];
    }
    output
}
fn safe_label(value: &str) -> String {
    if !value.is_empty()
        && value.len() <= 100
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
    {
        value.into()
    } else {
        "event".into()
    }
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}
fn utc(milliseconds: u64) -> String {
    let seconds = (milliseconds / 1000) as i64;
    let days = seconds / 86400;
    let daytime = seconds % 86400;
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        daytime / 3600,
        daytime / 60 % 60,
        daytime % 60,
        milliseconds % 1000
    )
}
fn build_flavor() -> &'static str {
    if cfg!(feature = "aten-hip") {
        "aten-hip"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "rocm") {
        "candle-hip"
    } else {
        "cpu-compatible"
    }
}

struct State {
    file: Option<File>,
    directory: Option<PathBuf>,
    attempted: usize,
    sequence: u64,
    recent: VecDeque<Value>,
    warning: Option<String>,
    previous_directories: Vec<PathBuf>,
}
pub struct SessionLog {
    id: String,
    role: String,
    pid: u32,
    started: Instant,
    context: Value,
    privacy: Privacy,
    candidates: Vec<PathBuf>,
    state: Mutex<State>,
}
impl SessionLog {
    fn create(role: &str, id: String, candidates: Vec<PathBuf>, privacy: Privacy) -> Self {
        let role = safe_label(role);
        let pid = std::process::id();
        let context = json!({"schema_version":1,"session_id":id,"role":role,"pid":pid,
            "application_version":env!("CARGO_PKG_VERSION"),"build_flavor":build_flavor(),
            "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"started_utc":utc(now_ms()),
            "privacy":"Prompts, raw token/primitive arrays, weight contents and credentials are omitted by default. User-profile path is masked."});
        let logger = Self {
            id,
            role,
            pid,
            started: Instant::now(),
            context,
            privacy,
            candidates,
            state: Mutex::new(State {
                file: None,
                directory: None,
                attempted: 0,
                sequence: 0,
                recent: VecDeque::new(),
                warning: None,
                previous_directories: Vec::new(),
            }),
        };
        logger.event("session.started", json!({}));
        logger
    }
    fn files_stem(&self) -> String {
        format!("{}-{}", self.role, self.pid)
    }
    fn ensure_writer(&self, state: &mut State) {
        while state.file.is_none() && state.attempted < self.candidates.len() {
            let directory = &self.candidates[state.attempted];
            state.attempted += 1;
            let result = (|| -> std::io::Result<File> {
                fs::create_dir_all(directory)?;
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(directory.join(format!("events-{}.jsonl", self.files_stem())))
            })();
            match result {
                Ok(file) => {
                    state.directory = Some(directory.clone());
                    state.file = Some(file);
                    let metadata = directory.join(format!("session-{}.json", self.files_stem()));
                    if let Ok(bytes) =
                        serde_json::to_vec_pretty(&self.privacy.value(&self.context, 0))
                    {
                        let _ = fs::write(metadata, bytes);
                    }
                }
                Err(error) => {
                    state.warning = Some(format!(
                        "日志目录不可写，已尝试备用位置：{}",
                        self.privacy.text(&error.to_string())
                    ))
                }
            }
        }
    }
    pub fn event(&self, kind: &str, fields: Value) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.sequence += 1;
        let row = json!({"schema_version":1,"utc":utc(now_ms()),"session_id":self.id,"role":self.role,"pid":self.pid,
            "sequence":state.sequence,"application_version":env!("CARGO_PKG_VERSION"),"build_flavor":build_flavor(),
            "elapsed_ms":self.started.elapsed().as_millis().min(u64::MAX as u128)as u64,
            "event":safe_label(kind),"fields":self.privacy.value(&fields,0)});
        if state.recent.len() == MAX_RECENT {
            state.recent.pop_front();
        }
        state.recent.push_back(row.clone());
        let Ok(mut bytes) = serde_json::to_vec(&row) else {
            return;
        };
        bytes.push(b'\n');
        loop {
            self.ensure_writer(&mut state);
            let Some(file) = state.file.as_mut() else {
                break;
            };
            match file.write_all(&bytes).and_then(|_| file.flush()) {
                Ok(()) => break,
                Err(error) => {
                    state.warning = Some(format!(
                        "日志写入失败，模型执行保持可用：{}",
                        self.privacy.text(&error.to_string())
                    ));
                    state.file = None;
                    if let Some(directory) = state.directory.take() {
                        state.previous_directories.push(directory);
                    }
                }
            }
        }
        self.write_summary(&mut state);
    }
    pub fn error(&self, stage: &str, error: &anyhow::Error) {
        let chain = error
            .chain()
            .take(16)
            .map(|e| self.privacy.text(&e.to_string()))
            .collect::<Vec<_>>();
        self.event(
            "error",
            json!({"stage":safe_label(stage),"error_chain":chain}),
        );
    }
    fn write_summary(&self, state: &mut State) {
        let Some(directory) = state.directory.as_ref() else {
            return;
        };
        let mut out=format!("Primitive Painter 诊断摘要\n会话：{}\n进程：{} / {}\n应用版本：{}\n构建：{}\n事件数：{}\n默认不记录提示词、原始图元数组、权重内容或凭据。\n\n",
            self.id,self.role,self.pid,env!("CARGO_PKG_VERSION"),build_flavor(),state.sequence);
        if let Some(warning) = &state.warning {
            out.push_str(warning);
            out.push_str("\n\n");
        }
        if state.sequence > state.recent.len() as u64 {
            out.push_str("以下保留最近事件；完整事件见同目录JSONL文件。\n\n");
        }
        for row in &state.recent {
            out.push_str(&format!(
                "{}  {}\n",
                row["utc"].as_str().unwrap_or(""),
                row["event"].as_str().unwrap_or("")
            ));
            if let Ok(fields) = serde_json::to_string_pretty(&row["fields"]) {
                out.push_str(&fields);
                out.push_str("\n\n");
            }
        }
        let _ = fs::write(
            directory.join(format!("summary-{}.txt", self.files_stem())),
            out,
        );
    }
    pub fn directory(&self) -> Option<PathBuf> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .directory
            .clone()
    }
    pub fn session_id(&self) -> String {
        self.id.clone()
    }
    pub fn warning(&self) -> Option<String> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .warning
            .clone()
    }
    pub fn child_environment(&self) -> Vec<(String, OsString)> {
        let mut out = vec![(SESSION_ENV.into(), OsString::from(&self.id))];
        if let Some(directory) = self.directory() {
            out.push((DIRECTORY_ENV.into(), directory.into_os_string()));
        }
        out
    }
    pub fn flush_summary(&self) -> Option<PathBuf> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(file) = state.file.as_mut() {
            let _ = file.flush();
        }
        self.write_summary(&mut state);
        state
            .directory
            .as_ref()
            .map(|p| p.join(format!("summary-{}.txt", self.files_stem())))
    }
    pub fn snapshot(&self) -> Value {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        json!({"session_id":self.id,"role":self.role,"pid":self.pid,"events":state.sequence,
            "directory":state.directory.as_ref().map(|p|self.privacy.text(&p.to_string_lossy())),"warning":state.warning})
    }
    pub fn export_diagnostics(&self, destination: Option<&Path>) -> Result<PathBuf> {
        self.flush_summary();
        let (directory, previous, recent) = {
            let s = self.state.lock().unwrap_or_else(|p| p.into_inner());
            (
                s.directory.clone(),
                s.previous_directories.clone(),
                s.recent.clone(),
            )
        };
        let destination = if let Some(path) = destination {
            path.to_path_buf()
        } else {
            directory
                .as_ref()
                .context("没有可写日志目录，请选择可写导出路径")?
                .join(format!("diagnostics-{}-{}.zip", self.id, now_ms()))
        };
        if !destination.is_absolute() {
            bail!("诊断导出路径必须为绝对路径");
        }
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)
            .context("无法创建诊断包，文件可能已存在")?;
        let mut archive = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        let mut roots = previous;
        if let Some(directory) = directory {
            roots.push(directory);
        }
        let mut names = std::collections::HashSet::new();
        for (root_index, root) in roots.iter().enumerate() {
            let Ok(entries) = fs::read_dir(root) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                let approved = (name.starts_with("events-") && name.ends_with(".jsonl"))
                    || (name.starts_with("summary-") && name.ends_with(".txt"))
                    || (name.starts_with("session-") && name.ends_with(".json"));
                if !approved || !entry.file_type()?.is_file() {
                    continue;
                }
                let name = if root_index == 0 {
                    name
                } else {
                    format!("location{root_index}/{name}")
                };
                if !names.insert(name.clone()) {
                    continue;
                }
                archive.start_file(name, options)?;
                std::io::copy(&mut File::open(path)?, &mut archive)?;
            }
        }
        archive.start_file("export-info.json", options)?;
        archive.write_all(&serde_json::to_vec_pretty(&json!({"schema_version":1,"session_id":self.id,"exported_utc":utc(now_ms()),
            "privacy":"No prompts or model/weight contents included by the diagnostic logger","recent_in_memory_events":recent}))?)?;
        archive.finish()?.sync_all()?;
        Ok(destination)
    }
    pub fn open_directory(&self) -> Result<()> {
        open_folder(&self.directory().context("日志目录当前不可写")?)
    }
}

pub fn init(role: &str) -> Arc<SessionLog> {
    GLOBAL
        .get_or_init(|| {
            let id = std::env::var(SESSION_ENV)
                .ok()
                .filter(|s| {
                    !s.is_empty()
                        && s.len() < 100
                        && s.bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
                })
                .unwrap_or_else(|| format!("{}-{}", now_ms(), std::process::id()));
            let mut candidates = Vec::new();
            if let Some(path) = std::env::var_os(DIRECTORY_ENV) {
                let path = PathBuf::from(path);
                if path.is_absolute() {
                    candidates.push(path);
                }
            }
            if let Some(path) = std::env::var_os("LOCALAPPDATA") {
                candidates.push(PathBuf::from(path).join("PrimitivePainter/logs").join(&id));
            }
            candidates.push(std::env::temp_dir().join("PrimitivePainter/logs").join(&id));
            candidates.dedup();
            Arc::new(SessionLog::create(role, id, candidates, Privacy::current()))
        })
        .clone()
}
pub fn event(kind: &str, fields: Value) {
    init("application").event(kind, fields)
}
pub fn error(stage: &str, error: &anyhow::Error) {
    init("application").error(stage, error)
}
pub fn directory() -> Option<PathBuf> {
    GLOBAL.get().and_then(|s| s.directory())
}
pub fn session_id() -> String {
    GLOBAL.get().map(|s| s.session_id()).unwrap_or_default()
}
pub fn child_environment() -> Vec<(String, OsString)> {
    init("application").child_environment()
}
pub fn flush_summary() -> Option<PathBuf> {
    GLOBAL.get().and_then(|s| s.flush_summary())
}
pub fn snapshot() -> Value {
    GLOBAL
        .get()
        .map(|s| s.snapshot())
        .unwrap_or_else(|| json!({"initialized":false}))
}
pub fn export_diagnostics(destination: Option<&Path>) -> Result<PathBuf> {
    init("application").export_diagnostics(destination)
}
pub fn open_directory() -> Result<()> {
    init("application").open_directory()
}
#[cfg(windows)]
fn open_folder(path: &Path) -> Result<()> {
    use std::{ffi::c_void, os::windows::ffi::OsStrExt};
    #[link(name = "shell32")]
    extern "system" {
        fn ShellExecuteW(
            window: *mut c_void,
            verb: *const u16,
            file: *const u16,
            args: *const u16,
            dir: *const u16,
            show: i32,
        ) -> isize;
    }
    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let verb: Vec<u16> = "open\0".encode_utf16().collect();
    if unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            verb.as_ptr(),
            path.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            1,
        )
    } <= 32
    {
        bail!("无法打开日志目录");
    }
    Ok(())
}
#[cfg(not(windows))]
fn open_folder(_: &Path) -> Result<()> {
    bail!("此构建仅在Windows提供原生目录入口")
}

#[cfg(test)]
mod tests {
    use super::*;
    static NEXT_TEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    fn temporary() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "painter-diag-test-{}-{}-{}",
            std::process::id(),
            now_ms(),
            NEXT_TEST.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }
    fn cleanup(path: PathBuf) {
        let p = fs::canonicalize(&path).unwrap();
        assert!(p.starts_with(fs::canonicalize(std::env::temp_dir()).unwrap()));
        assert!(path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("painter-diag-test-"));
        fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn private_inputs_and_credentials_do_not_reach_serialized_logs() {
        let privacy = Privacy {
            replacements: vec![("C:\\Users\\Alice".into(), "%USERPROFILE%".into())],
        };
        let value=privacy.value(&json!({"prompt":"private sentence","prompt_text":"private input","input_ids":[2,10,3],"HF_TOKEN":"private authentication","text_ids":[2,10,3],"model_state_dict":{"raw":[1,2]},"password":"secret","initial_records":["private drawing"],"initial_primitives":8,
            "prompt_tokens":33,"device_name":"7900XTX","path":"C:\\Users\\Alice\\models\\model.pt",
            "error":"GET https://name:password@host/path?token=secret#fragment Bearer abcdef"}),0).to_string();
        assert!(
            !value.contains("private sentence")
                && !value.contains("private input")
                && !value.contains("private authentication")
                && !value.contains("private drawing")
                && !value.contains("Alice")
                && !value.contains("abcdef")
                && !value.contains("token=secret")
        );
        assert!(
            value.contains("33") && value.contains("7900XTX") && value.contains("%USERPROFILE%")
        );
    }
    #[test]
    fn unwritable_preferred_directory_falls_back_without_failing_model_work() {
        let root = temporary();
        let blocked = root.join("blocked");
        fs::write(&blocked, b"file").unwrap();
        let fallback = root.join("fallback");
        let log = SessionLog::create(
            "test",
            "safe-session".into(),
            vec![blocked, fallback.clone()],
            Privacy {
                replacements: vec![],
            },
        );
        log.event(
            "backend.selected",
            json!({"backend":"cpu","reason":"explicit compatibility"}),
        );
        assert_eq!(log.directory(), Some(fallback));
        assert!(log.warning().is_some());
        assert!(log.flush_summary().unwrap().is_file());
        drop(log);
        cleanup(root);
    }
    #[test]
    fn summary_and_export_include_only_diagnostic_files() {
        let root = temporary();
        let directory = root.join("session");
        let log = SessionLog::create(
            "test",
            "export-session".into(),
            vec![directory.clone()],
            Privacy {
                replacements: vec![],
            },
        );
        log.event(
            "backend.capabilities",
            json!({"device":"7900XTX","disabled_reason":"unverified"}),
        );
        log.error("load", &anyhow::anyhow!("missing optional runtime"));
        fs::write(directory.join("model.pt"), b"not exported").unwrap();
        fs::write(directory.join("user-prompt.txt"), b"not exported").unwrap();
        let export = log
            .export_diagnostics(Some(&root.join("diagnostics.zip")))
            .unwrap();
        let archive = zip::ZipArchive::new(File::open(export).unwrap()).unwrap();
        let names = archive.file_names().collect::<Vec<_>>();
        assert!(names.iter().any(|s| s.ends_with(".jsonl")) && names.contains(&"export-info.json"));
        assert!(!names.contains(&"model.pt") && !names.contains(&"user-prompt.txt"));
        drop(archive);
        drop(log);
        cleanup(root);
    }
    #[test]
    fn unavailable_storage_keeps_events_exportable_without_interrupting_work() {
        let root = temporary();
        let blocked = root.join("blocked");
        fs::write(&blocked, b"file").unwrap();
        let log = SessionLog::create(
            "frontend",
            "memory-session".into(),
            vec![blocked],
            Privacy {
                replacements: vec![],
            },
        );
        log.event("generation.finished", json!({"complete_primitives":8}));
        assert!(log.directory().is_none());
        assert!(log.warning().is_some());
        let file = log
            .export_diagnostics(Some(&root.join("memory.zip")))
            .unwrap();
        let mut archive = zip::ZipArchive::new(File::open(file).unwrap()).unwrap();
        let info: Value =
            serde_json::from_reader(archive.by_name("export-info.json").unwrap()).unwrap();
        assert_eq!(
            info["recent_in_memory_events"][1]["fields"]["complete_primitives"],
            8
        );
        drop(archive);
        drop(log);
        cleanup(root);
    }
    #[test]
    fn multiple_process_roles_share_session_and_export_without_file_collisions() {
        let root = temporary();
        let directory = root.join("shared-session");
        let launcher = SessionLog::create(
            "launcher",
            "shared-session".into(),
            vec![directory.clone()],
            Privacy {
                replacements: vec![],
            },
        );
        let frontend = SessionLog::create(
            "frontend",
            "shared-session".into(),
            vec![directory],
            Privacy {
                replacements: vec![],
            },
        );
        launcher.event("provider.selected", json!({"backend":"aten-hip"}));
        frontend.event("model.loaded", json!({"device":"7900XTX"}));
        let inherited = launcher.child_environment();
        assert!(inherited
            .iter()
            .any(|(key, value)| key == SESSION_ENV && value == "shared-session"));
        let file = frontend
            .export_diagnostics(Some(&root.join("shared.zip")))
            .unwrap();
        let archive = zip::ZipArchive::new(File::open(file).unwrap()).unwrap();
        assert_eq!(
            archive
                .file_names()
                .filter(|name| name.starts_with("events-"))
                .count(),
            2
        );
        drop(archive);
        drop(frontend);
        drop(launcher);
        cleanup(root);
    }
    #[test]
    fn utc_and_schema_are_stable() {
        assert_eq!(utc(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(utc(1_793_577_600_000), "2026-11-02T00:00:00.000Z");
    }
}
