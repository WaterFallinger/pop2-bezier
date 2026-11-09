//! Native-only ATen runtime installation from pinned AMD wheel entries.
//! Wheels are ZIP containers; no Python interpreter or package scripts run.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    process::Command,
};

const MARKER: &str = ".primitive-painter-aten-runtime";
const OFFICIAL_BASE: &str = "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub runtime_id: String,
    pub components: Vec<RuntimeComponent>,
    pub total_native_files: u64,
    pub total_native_bytes: u64,
    pub total_download_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeComponent {
    pub id: String,
    pub version: String,
    pub url: String,
    pub download_bytes: u64,
    pub file_count: usize,
    pub native_bytes: u64,
    pub files: Vec<RuntimeFile>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeFile {
    pub archive_entry: String,
    pub bundle_path: String,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Receipt {
    schema_version: u32,
    runtime_id: String,
    manifest_sha256: String,
    files: u64,
    bytes: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct InstallResult {
    pub status: String,
    pub runtime_id: String,
    pub runtime_directory: PathBuf,
    pub files: u64,
    pub bytes: u64,
    pub source: String,
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}
fn safe_relative(name: &str) -> Result<PathBuf> {
    if name.is_empty() || name.contains(['\\', ':', '\0']) || name.starts_with('/') {
        bail!("运行库清单含不安全路径");
    }
    for piece in name.split('/') {
        if piece.is_empty() || piece == "." || piece == ".." || piece.ends_with([' ', '.']) {
            bail!("运行库路径含歧义或父目录");
        }
        let stem = piece.split('.').next().unwrap_or("").to_ascii_uppercase();
        if ["CON", "PRN", "AUX", "NUL"].contains(&stem.as_str())
            || (stem.len() == 4
                && (stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.as_bytes()[3].is_ascii_digit())
        {
            bail!("运行库路径含Windows设备名");
        }
    }
    Ok(PathBuf::from(name))
}
pub fn parse_manifest(text: &str) -> Result<Manifest> {
    let manifest: Manifest = serde_json::from_str(text).context("原生运行库清单格式不正确")?;
    if manifest.schema_version != 1
        || manifest.components.is_empty()
        || manifest.components.len() > 3
        || manifest.runtime_id.is_empty()
        || !manifest
            .runtime_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    {
        bail!("不支持的原生运行库清单");
    }
    let mut ids = HashSet::new();
    let mut outputs = HashSet::new();
    let mut files = 0u64;
    let mut bytes = 0u64;
    let mut download = 0u64;
    for component in &manifest.components {
        let prefix = match component.id.as_str() {
            "torch" => "torch/lib/",
            "core" => "_rocm_sdk_core/bin/",
            "libraries" => "_rocm_sdk_libraries_custom/bin/",
            _ => bail!("未知原生运行库组件"),
        };
        if !ids.insert(component.id.clone())
            || component.files.is_empty()
            || component.files.len() != component.file_count
            || !component.url.starts_with(OFFICIAL_BASE)
            || !component.url.ends_with(".whl")
            || component.url.contains(['?', '#', '@', '\\'])
            || component.download_bytes == 0
        {
            bail!("运行库组件来源或文件列表无效");
        }
        let mut entries = HashSet::new();
        let mut component_bytes = 0u64;
        for file in &component.files {
            safe_relative(&file.archive_entry)?;
            safe_relative(&file.bundle_path)?;
            let relative = file
                .archive_entry
                .strip_prefix(prefix)
                .context("运行库源条目不属于所选组件")?;
            if file.bundle_path != format!("{}/{relative}", component.id) {
                bail!("运行库目标不符合组件布局");
            }
            let extension = Path::new(relative)
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !["dll", "aks2", "dat", "hsaco", "co", "txt"].contains(&extension.as_str())
                || Path::new(relative)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .eq_ignore_ascii_case("torch_python.dll")
            {
                bail!("运行库清单包含Python、编译器或未批准文件");
            }
            if !valid_hash(&file.sha256)
                || file.bytes == 0
                || !entries.insert(file.archive_entry.to_ascii_lowercase())
                || !outputs.insert(file.bundle_path.to_ascii_lowercase())
            {
                bail!("运行库校验值或路径重复");
            }
            files += 1;
            bytes = bytes.checked_add(file.bytes).context("运行库长度溢出")?;
            component_bytes = component_bytes
                .checked_add(file.bytes)
                .context("组件长度溢出")?;
        }
        if component_bytes != component.native_bytes {
            bail!("组件展开长度与清单不一致");
        }
        download = download
            .checked_add(component.download_bytes)
            .context("下载长度溢出")?;
    }
    if files != manifest.total_native_files
        || bytes != manifest.total_native_bytes
        || download != manifest.total_download_bytes
    {
        bail!("运行库总量与组件清单不一致");
    }
    Ok(manifest)
}

fn reject_reparse(path: &Path) -> Result<()> {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            if metadata.file_attributes() & 0x400 != 0 {
                bail!("运行库目录不能穿过链接或重解析点：{}", path.display());
            }
        }
        if metadata.file_type().is_symlink() {
            bail!("运行库目录不能穿过符号链接：{}", path.display());
        }
    }
    Ok(())
}
fn safe_target(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = safe_relative(relative)?;
    let mut current = root.to_path_buf();
    reject_reparse(&current)?;
    for part in relative.components() {
        if let Component::Normal(part) = part {
            current.push(part);
            reject_reparse(&current)?;
        } else {
            bail!("运行库路径不能是绝对路径或父目录");
        }
    }
    Ok(current)
}
fn ensure_root(root: &Path, runtime_id: &str) -> Result<()> {
    if !root.is_absolute() {
        bail!("运行库安装目录必须为绝对路径");
    }
    for ancestor in root.ancestors() {
        reject_reparse(ancestor)?;
    }
    if root.exists() {
        let marker = root.join(MARKER);
        if marker.exists() {
            if fs::read_to_string(marker)?.trim() != runtime_id {
                bail!("运行库目录属于其他版本，停止覆盖");
            }
        } else if fs::read_dir(root)?.next().is_some() {
            bail!("运行库目标目录不是空目录，停止覆盖其他文件");
        }
    }
    fs::create_dir_all(root)?;
    let marker = root.join(MARKER);
    if !marker.exists() {
        fs::write(marker, format!("{runtime_id}\n"))?;
    }
    Ok(())
}
fn verify_file(path: &Path, file: &RuntimeFile) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("缺少运行库文件 {}", path.display()))?;
    if !metadata.is_file() || metadata.len() != file.bytes {
        bail!("运行库文件长度不匹配：{}", path.display());
    }
    let mut input = fs::File::open(path)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hash = Sha256::new();
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    if format!("{:x}", hash.finalize()) != file.sha256.to_ascii_lowercase() {
        bail!("运行库文件SHA256不匹配：{}", path.display());
    }
    Ok(())
}
fn copy_checked(input: &mut impl Read, target: &Path, file: &RuntimeFile) -> Result<()> {
    fs::create_dir_all(target.parent().context("运行库目标无父目录")?)?;
    let mut output = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(target)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut copied = 0u64;
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        copied = copied.checked_add(n as u64).context("文件长度溢出")?;
        if copied > file.bytes {
            bail!("运行库条目超过固定大小");
        }
        hash.update(&buffer[..n]);
        output.write_all(&buffer[..n])?;
    }
    output.flush()?;
    output.sync_all()?;
    if copied != file.bytes || format!("{:x}", hash.finalize()) != file.sha256.to_ascii_lowercase()
    {
        bail!(
            "下载或复制的运行库文件未通过固定SHA256校验：{}",
            file.bundle_path
        );
    }
    Ok(())
}
#[cfg(windows)]
fn replace_file(source: &Path, target: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn MoveFileExW(a: *const u16, b: *const u16, flags: u32) -> i32;
    }
    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let target: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
    if unsafe { MoveFileExW(source.as_ptr(), target.as_ptr(), 0x1 | 0x8) } == 0 {
        bail!(
            "无法发布运行库文件，请先关闭使用该安装目录的应用：{}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}
#[cfg(not(windows))]
fn replace_file(source: &Path, target: &Path) -> Result<()> {
    fs::rename(source, target)?;
    Ok(())
}
struct Stage {
    root: PathBuf,
    files: Vec<PathBuf>,
}
impl Stage {
    fn new(root: &Path, id: &str) -> Result<Self> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = root.join(format!(".stage-{id}-{}-{nonce}", std::process::id()));
        fs::create_dir(&root)?;
        Ok(Self {
            root,
            files: Vec::new(),
        })
    }
    fn target(&mut self, file: &RuntimeFile) -> Result<PathBuf> {
        let path = safe_target(&self.root, &file.bundle_path)?;
        self.files.push(path.clone());
        Ok(path)
    }
}
impl Drop for Stage {
    fn drop(&mut self) {
        let mut dirs = HashSet::new();
        for path in &self.files {
            if !path.starts_with(&self.root) {
                continue;
            }
            let _ = fs::remove_file(path);
            let mut parent = path.parent();
            while let Some(dir) = parent {
                if !dir.starts_with(&self.root) {
                    break;
                }
                dirs.insert(dir.to_path_buf());
                parent = dir.parent();
            }
        }
        dirs.insert(self.root.clone());
        let mut dirs = dirs.into_iter().collect::<Vec<_>>();
        dirs.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
        for directory in dirs {
            let _ = fs::remove_dir(directory);
        }
    }
}
fn stage_offline(
    source_site: &Path,
    component: &RuntimeComponent,
    stage: &mut Stage,
) -> Result<()> {
    if !source_site.is_absolute() || !source_site.is_dir() {
        bail!("离线组件来源必须是存在的绝对目录");
    }
    for file in &component.files {
        let source = safe_target(source_site, &file.archive_entry)?;
        let mut input = fs::File::open(&source)
            .with_context(|| format!("离线来源缺少 {}", file.archive_entry))?;
        let target = stage.target(file)?;
        copy_checked(&mut input, &target, file)?;
    }
    Ok(())
}
fn stage_archive(archive: &Path, component: &RuntimeComponent, stage: &mut Stage) -> Result<()> {
    let mut archive =
        zip::ZipArchive::new(fs::File::open(archive)?).context("无法读取官方运行库wheel")?;
    let mut names = HashSet::new();
    for index in 0..archive.len() {
        let name = archive.by_index(index)?.name().to_ascii_lowercase();
        if !names.insert(name) {
            bail!("运行库ZIP出现重复条目");
        }
    }
    for file in &component.files {
        let mut entry = archive
            .by_name(&file.archive_entry)
            .with_context(|| format!("组件缺少固定条目 {}", file.archive_entry))?;
        if entry.is_dir()
            || entry.size() != file.bytes
            || entry.unix_mode().is_some_and(|m| m & 0o170000 == 0o120000)
        {
            bail!("运行库条目类型或长度不匹配");
        }
        let target = stage.target(file)?;
        copy_checked(&mut entry, &target, file)?;
    }
    Ok(())
}
fn download(component: &RuntimeComponent, cache: &Path, runtime_id: &str) -> Result<PathBuf> {
    ensure_root(cache, &format!("{runtime_id}.downloads"))?;
    let target = cache.join(format!("{}-{}.whl.part", runtime_id, component.id));
    reject_reparse(&target)?;
    if target.is_file() && fs::metadata(&target)?.len() == component.download_bytes {
        return Ok(target);
    }
    if target.is_file() && fs::metadata(&target)?.len() > component.download_bytes {
        fs::remove_file(&target)?;
    }
    #[cfg(windows)]
    let curl = std::env::var_os("WINDIR")
        .map(PathBuf::from)
        .context("无法定位Windows目录")?
        .join("System32/curl.exe");
    #[cfg(not(windows))]
    let curl = PathBuf::from("curl");
    let mut command = Command::new(curl);
    command
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-redirs",
            "3",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--tlsv1.2",
            "--retry",
            "2",
            "--continue-at",
            "-",
            "--output",
        ])
        .arg(&target)
        .arg(&component.url);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    let code = command.status().context("无法启动Windows原生HTTPS下载器")?;
    if !code.success() {
        bail!("下载{}失败，已保留断点文件；可重试安装", component.id);
    }
    if fs::metadata(&target)?.len() != component.download_bytes {
        bail!("官方组件下载长度与固定清单不符");
    }
    Ok(target)
}
pub fn install(
    manifest_text: &str,
    runtime: &Path,
    cache: Option<&Path>,
    source_site: Option<&Path>,
    verify_only: bool,
    keep_downloads: bool,
) -> Result<InstallResult> {
    let manifest = parse_manifest(manifest_text)?;
    let digest = hash_bytes(&serde_json::to_vec(&manifest)?);
    if verify_only {
        if !runtime.is_absolute() {
            bail!("运行库目录必须是绝对路径");
        }
        for component in &manifest.components {
            for file in &component.files {
                verify_file(&safe_target(runtime, &file.bundle_path)?, file)?;
            }
        }
        return Ok(InstallResult {
            status: "verified".into(),
            runtime_id: manifest.runtime_id,
            runtime_directory: runtime.to_path_buf(),
            files: manifest.total_native_files,
            bytes: manifest.total_native_bytes,
            source: "existing_native_runtime".into(),
        });
    }
    ensure_root(runtime, &manifest.runtime_id)?;
    let receipt_path = runtime.join("native-runtime-receipt.json");
    if let Ok(bytes) = fs::read(&receipt_path) {
        if let Ok(receipt) = serde_json::from_slice::<Receipt>(&bytes) {
            if receipt.manifest_sha256 == digest
                && receipt.runtime_id == manifest.runtime_id
                && manifest.components.iter().flat_map(|c| &c.files).all(|f| {
                    safe_target(runtime, &f.bundle_path)
                        .and_then(|p| Ok(fs::metadata(p)?.len() == f.bytes))
                        .unwrap_or(false)
                })
            {
                return Ok(InstallResult {
                    status: "already_installed".into(),
                    runtime_id: manifest.runtime_id,
                    runtime_directory: runtime.to_path_buf(),
                    files: manifest.total_native_files,
                    bytes: manifest.total_native_bytes,
                    source: "verified_install_receipt".into(),
                });
            }
        }
    }
    for (index, component) in manifest.components.iter().enumerate() {
        println!(
            "正在准备AMD原生运行组件 {}/{}（{}）…",
            index + 1,
            manifest.components.len(),
            component.id
        );
        let mut stage = Stage::new(runtime, &component.id)?;
        let archive = if let Some(site) = source_site {
            stage_offline(site, component, &mut stage)?;
            None
        } else {
            let cache = cache.context("联网安装必须提供绝对缓存目录")?;
            let path = download(component, cache, &manifest.runtime_id)?;
            if let Err(error)=stage_archive(&path,component,&mut stage){
                if !keep_downloads{let _=fs::remove_file(&path);}
                return Err(error.context("官方组件未通过固定校验，未发布该组件；再次安装会重新获取失效缓存"));
            }
            Some(path)
        };
        // Publish only after every selected file in this component passed its
        // independent pinned SHA256. No package installation script is run.
        for file in &component.files {
            let source = safe_target(&stage.root, &file.bundle_path)?;
            let target = safe_target(runtime, &file.bundle_path)?;
            fs::create_dir_all(target.parent().context("运行库目标无父目录")?)?;
            replace_file(&source, &target)?;
        }
        drop(stage);
        if let Some(path) = archive {
            if !keep_downloads {
                fs::remove_file(path)?;
            }
        }
    }
    let receipt = Receipt {
        schema_version: 1,
        runtime_id: manifest.runtime_id.clone(),
        manifest_sha256: digest,
        files: manifest.total_native_files,
        bytes: manifest.total_native_bytes,
    };
    let temporary = runtime.join(format!(".receipt-{}.tmp", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(&receipt)?)?;
    replace_file(&temporary, &receipt_path)?;
    Ok(InstallResult {
        status: "installed".into(),
        runtime_id: manifest.runtime_id,
        runtime_directory: runtime.to_path_buf(),
        files: manifest.total_native_files,
        bytes: manifest.total_native_bytes,
        source: if source_site.is_some() {
            "existing_official_native_files"
        } else {
            "official_https_wheels"
        }
        .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> String {
        let bytes = b"native fixture";
        serde_json::json!({"schema_version":1,"runtime_id":"fixture-v1","total_native_files":1,"total_native_bytes":bytes.len(),"total_download_bytes":123,
        "components":[{"id":"core","version":"test","url":format!("{OFFICIAL_BASE}fixture.whl"),"download_bytes":123,"file_count":1,"native_bytes":bytes.len(),
            "files":[{"archive_entry":"_rocm_sdk_core/bin/example.dll","bundle_path":"core/example.dll","bytes":bytes.len(),"sha256":hash_bytes(bytes)}]}]}).to_string()
    }
    fn temporary(tag: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "aten-runtime-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }
    fn cleanup(path:PathBuf){
        let resolved=fs::canonicalize(&path).unwrap();let temporary=fs::canonicalize(std::env::temp_dir()).unwrap();
        assert!(resolved.starts_with(&temporary));
        assert!(path.file_name().unwrap().to_string_lossy().starts_with("aten-runtime-"));
        fs::remove_dir_all(resolved).unwrap();
    }
    #[test]
    fn rejects_escape_python_and_unpinned_sources() {
        for name in [
            "../x.dll",
            "core/../x.dll",
            "core/NUL.dll",
            "C:/x.dll",
            "core/x.dll ",
            "core\\x.dll",
        ] {
            assert!(safe_relative(name).is_err());
        }
        let mut v: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
        v["components"][0]["url"] = "https://example.com/fixture.whl".into();
        assert!(parse_manifest(&v.to_string()).is_err());
        v = serde_json::from_str(&fixture()).unwrap();
        v["components"][0]["files"][0]["bundle_path"] = "../example.dll".into();
        assert!(parse_manifest(&v.to_string()).is_err());
    }
    #[test]
    fn offline_install_is_verified_and_preserves_unrelated_files() {
        let root = temporary("offline");
        let source = root.join("source");
        fs::create_dir_all(source.join("_rocm_sdk_core/bin")).unwrap();
        fs::write(
            source.join("_rocm_sdk_core/bin/example.dll"),
            b"native fixture",
        )
        .unwrap();
        let target = root.join("runtime");
        assert_eq!(
            install(&fixture(), &target, None, Some(&source), false, false)
                .unwrap()
                .status,
            "installed"
        );
        fs::write(target.join("user-note.txt"), b"keep").unwrap();
        assert_eq!(
            install(&fixture(), &target, None, None, false, false)
                .unwrap()
                .status,
            "already_installed"
        );
        install(&fixture(), &target, None, None, true, false).unwrap();
        assert_eq!(fs::read(target.join("user-note.txt")).unwrap(), b"keep");
        fs::write(target.join("core/example.dll"), b"broken fixture").unwrap();
        assert!(install(&fixture(), &target, None, None, true, false).is_err());
        cleanup(root);
    }
    #[test]
    fn zip_extracts_only_exact_pinned_entries() {
        let root = temporary("zip");
        let archive = root.join("fixture.whl");
        let mut zip = zip::ZipWriter::new(fs::File::create(&archive).unwrap());
        zip.start_file(
            "_rocm_sdk_core/bin/example.dll",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        zip.write_all(b"native fixture").unwrap();
        zip.start_file(
            "../../unselected.py",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        zip.write_all(b"not executed").unwrap();
        zip.finish().unwrap();
        let manifest = parse_manifest(&fixture()).unwrap();
        let mut stage = Stage::new(&root, "zip").unwrap();
        stage_archive(&archive, &manifest.components[0], &mut stage).unwrap();
        assert_eq!(stage.files.len(), 1);
        assert!(!root.join("unselected.py").exists());
        drop(stage);
        cleanup(root);
    }
}
