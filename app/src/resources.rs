//! One-shot local resource discovery and persistent per-user model paths.
use serde::{Deserialize, Serialize};
use std::{
    ffi::OsStr,
    io::Write,
    path::{Path, PathBuf},
};
const CONFIG_FILE: &str = "model-config.json";
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPaths {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
    pub first_run_required: bool,
    pub config_path: Option<PathBuf>,
    pub configuration_error: Option<String>,
    /// Nonblocking explanation when a stale preference was recovered locally.
    pub discovery_notice: Option<String>,
}
#[derive(Serialize, Deserialize)]
struct ModelConfig {
    #[serde(default = "schema_version")]
    schema_version: u32,
    model_path: String,
    tokenizer_path: String,
}
fn schema_version() -> u32 {
    1
}
pub fn user_config_dir() -> Option<PathBuf> {
    if let Some(base) = std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(base).join("PrimitivePainter"));
    }
    std::env::var_os("USERPROFILE")
        .filter(|v| !v.is_empty())
        .map(|p| PathBuf::from(p).join("AppData/Local/PrimitivePainter"))
}
pub fn find_resources() -> ModelPaths {
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    let user = user_config_dir();
    let (models, tokenizers) = user_default_locations(std::env::var_os("USERPROFILE").as_deref());
    apply_overrides(
        find_resources_at_with_defaults(&exe, user.as_deref(), &models, &tokenizers),
        std::env::var_os("PAINTER_MODEL").as_deref(),
        std::env::var_os("PAINTER_TOKENIZER").as_deref(),
    )
}
/// `user_dir` is the PrimitivePainter config directory, not LocalAppData itself.
/// This deterministic seam does not read environment or create directories.
pub fn find_resources_at(executable_dir: &Path, user_dir: Option<&Path>) -> ModelPaths {
    find_resources_at_with_defaults(executable_dir, user_dir, &[], &[])
}

/// Bounded, deterministic seam: callers provide additional agreed locations;
/// the resolver never walks folders recursively or reads a weight's contents.
pub fn find_resources_at_with_defaults(
    executable_dir: &Path,
    user_dir: Option<&Path>,
    additional_models: &[PathBuf],
    additional_tokenizers: &[PathBuf],
) -> ModelPaths {
    let root = installation_root(executable_dir);
    let defaults = || ModelPaths {
        model: root.join("models/model.pt"),
        tokenizer: root.join("assets/caption_spm_1500.model"),
        first_run_required: true,
        config_path: None,
        configuration_error: None,
        discovery_notice: None,
    };
    let mut tokenizer_defaults = vec![root.join("assets/caption_spm_1500.model")];
    if let Some(user)=user_dir {tokenizer_defaults.push(user.join("models/pop2-1b-epoch60/caption_spm_1500.model"));}
    tokenizer_defaults.extend(additional_tokenizers.iter().cloned());
    let mut candidates = Vec::new();
    if let Some(user) = user_dir {
        candidates.push(user.join(CONFIG_FILE));
    }
    candidates.push(executable_dir.join(CONFIG_FILE));
    if root != executable_dir {
        candidates.push(root.join(CONFIG_FILE));
    }
    let mut notices = Vec::new();
    let mut attempted_models = Vec::new();
    let mut first_invalid_pair = None;
    for path in candidates {
        match read_config(&path) {
            Ok(Some((model, tokenizer))) => {
                if !attempted_models.contains(&model) {
                    attempted_models.push(model.clone());
                }
                if model.is_file() && tokenizer.is_file() {
                    return discovered(model, tokenizer, Some(path), &notices);
                }
                first_invalid_pair.get_or_insert((model.clone(), tokenizer.clone(), path.clone()));
                if model.is_file() {
                    if let Some(replacement) = tokenizer_defaults
                        .iter()
                        .find(|candidate| candidate.is_file())
                    {
                        notices.push(format!(
                            "配置中的分词器未找到，已使用本地默认分词器：{}",
                            replacement.display()
                        ));
                        return discovered(model, replacement.clone(), Some(path), &notices);
                    }
                }
                notices.push(format!(
                    "已保存的模型或分词器路径不可用：{}",
                    path.display()
                ));
            }
            Ok(None) => {}
            Err(error) => notices.push(error),
        }
    }
    let mut model_defaults = vec![root.join("models/model.pt"), root.join("model.pt")];
    if let Some(user) = user_dir {
        model_defaults.push(user.join("models/pop2-1b-epoch60/model.pt"));
        model_defaults.push(user.join("models/model.pt"));
        model_defaults.push(user.join("model.pt"));
    }
    model_defaults.extend(additional_models.iter().cloned());
    let mut first_existing_default_model = None;
    for model in model_defaults {
        if !attempted_models.contains(&model) {
            attempted_models.push(model.clone());
        }
        if model.is_file() {
            if let Some(tokenizer) = tokenizer_defaults.iter().find(|path| path.is_file()) {
                return discovered(model, tokenizer.clone(), None, &notices);
            }
            first_existing_default_model.get_or_insert(model);
        }
    }
    let mut result = defaults();
    if let Some((model, tokenizer, path)) = first_invalid_pair {
        result.model = model;
        result.tokenizer = tokenizer;
        result.config_path = Some(path);
    }
    if !result.model.is_file() {
        if let Some(model) = first_existing_default_model {
            result.model = model;
            result.config_path = None;
        }
    }
    if !result.tokenizer.is_file() {
        if let Some(tokenizer) = tokenizer_defaults.iter().find(|path| path.is_file()) {
            result.tokenizer = tokenizer.clone();
        }
    }
    let locations = attempted_models
        .iter()
        .take(4)
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join("；");
    let extra = if attempted_models.len() > 4 {
        format!("，共检查{}个约定位置", attempted_models.len())
    } else {
        String::new()
    };
    result.configuration_error=Some(format!("启动时未找到可用的模型权重和分词器。请在“设置”中下载官方权重或选择本地文件。已检查：{locations}{extra}。"));
    if !notices.is_empty() {
        result.discovery_notice = Some(notices.join("\n"));
    }
    result
}

fn discovered(
    model: PathBuf,
    tokenizer: PathBuf,
    config_path: Option<PathBuf>,
    notices: &[String],
) -> ModelPaths {
    ModelPaths {
        model,
        tokenizer,
        first_run_required: false,
        config_path,
        configuration_error: None,
        discovery_notice: (!notices.is_empty())
            .then(|| format!("{}\n已自动找到可用的本地权重与分词器。", notices.join("\n"))),
    }
}

fn user_default_locations(_profile: Option<&OsStr>) -> (Vec<PathBuf>, Vec<PathBuf>) {
    // Public builds use the installed/user cache and the saved path preference.
    // No developer workstation directories are searched.
    (Vec::new(), Vec::new())
}
/// Call after a successful model load; never edits a portable/install config.
pub fn save_model_paths(model: &Path, tokenizer: &Path) -> Result<PathBuf, String> {
    let dir = user_config_dir().ok_or("无法定位当前用户的 LocalAppData 配置目录")?;
    save_model_paths_at(&dir, model, tokenizer)
}
pub fn save_model_paths_at(
    user_dir: &Path,
    model: &Path,
    tokenizer: &Path,
) -> Result<PathBuf, String> {
    let model = absolute_file(model, "权重")?;
    let tokenizer = absolute_file(tokenizer, "分词器")?;
    let config = ModelConfig {
        schema_version: 1,
        model_path: model.to_str().ok_or("权重路径不是有效 Unicode")?.into(),
        tokenizer_path: tokenizer
            .to_str()
            .ok_or("分词器路径不是有效 Unicode")?
            .into(),
    };
    let mut bytes =
        serde_json::to_vec_pretty(&config).map_err(|e| format!("无法编码模型配置：{e}"))?;
    bytes.push(b'\n');
    std::fs::create_dir_all(user_dir)
        .map_err(|e| format!("无法创建用户配置目录 {}：{e}", user_dir.display()))?;
    let destination = user_dir.join(CONFIG_FILE);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary = user_dir.join(format!(".model-config-{}-{nonce}.tmp", std::process::id()));
    let result = (|| -> Result<(), String> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| format!("无法写临时模型配置：{e}"))?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| format!("保存模型配置失败：{e}"))?;
        drop(file);
        replace_file(&temporary, &destination).map_err(|e| format!("无法更新用户模型配置：{e}"))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result?;
    Ok(destination)
}
fn installation_root(exe: &Path) -> PathBuf {
    if exe.file_name().and_then(OsStr::to_str).is_some_and(|name| name.starts_with("aten-app-") || name.starts_with("app-"))
        && exe.parent().and_then(Path::file_name).and_then(OsStr::to_str)
            .is_some_and(|name| name.eq_ignore_ascii_case("runtime")) {
        return exe.parent().and_then(Path::parent).unwrap_or(exe).to_path_buf();
    }
    let nested = exe
        .file_name()
        .and_then(OsStr::to_str)
        .map(|n| n.eq_ignore_ascii_case("runtime") || n.eq_ignore_ascii_case("bin"))
        .unwrap_or(false);
    if nested {
        exe.parent().unwrap_or(exe).to_path_buf()
    } else {
        exe.to_path_buf()
    }
}
fn read_config(path: &Path) -> Result<Option<(PathBuf, PathBuf)>, String> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("无法读取配置 {}：{e}", path.display())),
    };
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let config: ModelConfig = serde_json::from_slice(bytes)
        .map_err(|e| format!("模型配置 {} 格式不正确：{e}", path.display()))?;
    if config.schema_version != 1 {
        return Err(format!("模型配置 {} 的版本暂不支持", path.display()));
    }
    if config.model_path.trim().is_empty() || config.tokenizer_path.trim().is_empty() {
        return Err(format!("模型配置 {} 缺少有效路径", path.display()));
    }
    let root = path.parent().unwrap_or(Path::new("."));
    Ok(Some((
        resolve(root, &config.model_path),
        resolve(root, &config.tokenizer_path),
    )))
}
fn resolve(root: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}
fn absolute_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    let path = std::path::absolute(path).map_err(|e| format!("无法定位{label}文件：{e}"))?;
    if !path.is_file() {
        return Err(format!("{label}文件不存在：{}", path.display()));
    }
    Ok(path)
}
fn apply_overrides(
    mut paths: ModelPaths,
    model: Option<&OsStr>,
    tokenizer: Option<&OsStr>,
) -> ModelPaths {
    let model = model.filter(|p| !p.is_empty());
    let tokenizer = tokenizer.filter(|p| !p.is_empty());
    if let Some(value) = model {
        paths.model = PathBuf::from(value);
    }
    if let Some(value) = tokenizer {
        paths.tokenizer = PathBuf::from(value);
    }
    if model.is_some() || tokenizer.is_some() {
        paths.config_path = None;
        paths.discovery_notice = None;
        paths.configuration_error = if paths.model.is_file() && paths.tokenizer.is_file() {
            None
        } else {
            Some(format!(
                "指定的模型权重或分词器不存在：{}；{}。请在“设置”中检查。",
                paths.model.display(),
                paths.tokenizer.display()
            ))
        };
    }
    paths.first_run_required =
        paths.configuration_error.is_some() || !paths.model.is_file() || !paths.tokenizer.is_file();
    paths
}
#[cfg(windows)]
pub(crate) fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32;
    }
    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    // Atomic replace; the old user preference remains in place on failure.
    if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), 0x1 | 0x8) } == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}
#[cfg(not(windows))]
pub(crate) fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sandbox(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "primitive-painter-resources-{}-{name}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }
    fn file(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"resource-path-test").unwrap();
    }
    fn config(dir: &Path, model: &str, tokenizer: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let value = serde_json::json!({"model_path":model,"tokenizer_path":tokenizer});
        std::fs::write(dir.join(CONFIG_FILE), serde_json::to_vec(&value).unwrap()).unwrap();
    }
    #[test]
    fn installed_defaults_use_models_and_assets_under_installation() {
        let root = sandbox("defaults");
        let runtime = root.join("runtime");
        std::fs::create_dir_all(&runtime).unwrap();
        let found = find_resources_at(&runtime, None);
        assert_eq!(found.model, root.join("models/model.pt"));
        assert_eq!(found.tokenizer, root.join("assets/caption_spm_1500.model"));
        assert!(found.first_run_required && found.config_path.is_none());
        assert!(found
            .configuration_error
            .as_deref()
            .unwrap()
            .contains("未找到"));
        file(&found.model);
        file(&found.tokenizer);
        assert!(!find_resources_at(&runtime, None).first_run_required);
    }
    #[test]
    fn versioned_aten_application_recovers_existing_root_weight_configuration() {
        let root = sandbox("aten-versioned");
        let application = root.join("runtime/aten-app-0.8.0");
        std::fs::create_dir_all(&application).unwrap();
        file(&root.join("existing/model.pt"));
        file(&root.join("assets/caption_spm_1500.model"));
        config(&root, "existing/model.pt", "assets/caption_spm_1500.model");
        let found = find_resources_at(&application, None);
        assert_eq!(found.model, root.join("existing/model.pt"));
        assert_eq!(found.tokenizer, root.join("assets/caption_spm_1500.model"));
        assert!(!found.first_run_required && found.configuration_error.is_none());
    }
    #[test]
    fn valid_saved_pair_wins_over_installed_defaults() {
        let root = sandbox("priority");
        let installed = root.join("app");
        let user = root.join("user");
        config(&installed, "portable.pt", "portable.model");
        file(&installed.join("portable.pt"));
        file(&installed.join("portable.model"));
        config(&user, "chosen/checkpoint.pt", "chosen/tokenizer.model");
        file(&user.join("chosen/checkpoint.pt"));
        file(&user.join("chosen/tokenizer.model"));
        let found = find_resources_at(&installed, Some(&user));
        assert_eq!(found.model, user.join("chosen/checkpoint.pt"));
        assert_eq!(found.tokenizer, user.join("chosen/tokenizer.model"));
        assert_eq!(found.config_path, Some(user.join(CONFIG_FILE)));
        assert!(!found.first_run_required);
    }
    #[test]
    fn portable_legacy_configs_resolve_relative_to_their_own_directory() {
        let root = sandbox("portable");
        let runtime = root.join("runtime");
        std::fs::create_dir_all(&runtime).unwrap();
        config(&root, "models/legacy.pt", "assets/legacy.model");
        file(&root.join("models/legacy.pt"));
        file(&root.join("assets/legacy.model"));
        let found = find_resources_at(&runtime, None);
        assert_eq!(found.model, root.join("models/legacy.pt"));
        assert!(!found.first_run_required);
        config(&runtime, "local.pt", "local.model");
        file(&runtime.join("local.pt"));
        file(&runtime.join("local.model"));
        assert_eq!(
            find_resources_at(&runtime, None).model,
            runtime.join("local.pt")
        );
    }
    #[test]
    fn preferences_persist_absolute_paths_and_replace_previous_selection_atomically() {
        let root = sandbox("save");
        let user = root.join("prefs");
        let exe = root.join("application");
        let first = root.join("first.pt");
        let second = root.join("second.pt");
        let tok = root.join("tokenizer.model");
        file(&first);
        file(&second);
        file(&tok);
        assert_eq!(
            save_model_paths_at(&user, &first, &tok).unwrap(),
            user.join(CONFIG_FILE)
        );
        let found = find_resources_at(&exe, Some(&user));
        assert_eq!(found.model, std::path::absolute(&first).unwrap());
        assert!(!found.first_run_required);
        save_model_paths_at(&user, &second, &tok).unwrap();
        assert_eq!(
            find_resources_at(&exe, Some(&user)).model,
            std::path::absolute(&second).unwrap()
        );
        assert_eq!(std::fs::read_dir(&user).unwrap().count(), 1);
    }
    #[test]
    fn invalid_selection_never_creates_or_overwrites_user_preferences() {
        let root = sandbox("invalid");
        let user = root.join("prefs");
        let tok = root.join("tokenizer.model");
        file(&tok);
        assert!(save_model_paths_at(&user, &root.join("missing.pt"), &tok).is_err());
        assert!(!user.exists());
        let good = root.join("good.pt");
        file(&good);
        save_model_paths_at(&user, &good, &tok).unwrap();
        let prior = std::fs::read(user.join(CONFIG_FILE)).unwrap();
        assert!(save_model_paths_at(&user, &root.join("missing.pt"), &tok).is_err());
        assert_eq!(std::fs::read(user.join(CONFIG_FILE)).unwrap(), prior);
    }
    #[test]
    fn corrupt_saved_configuration_can_recover_to_a_valid_installed_pair_with_notice() {
        let root = sandbox("corrupt");
        let user = root.join("prefs");
        std::fs::create_dir_all(&user).unwrap();
        std::fs::write(user.join(CONFIG_FILE), b"{ broken json").unwrap();
        config(&root, "portable.pt", "portable.model");
        file(&root.join("portable.pt"));
        file(&root.join("portable.model"));
        let found = find_resources_at(&root, Some(&user));
        assert_eq!(found.model, root.join("portable.pt"));
        assert!(!found.first_run_required && found.configuration_error.is_none());
        assert!(found
            .discovery_notice
            .as_deref()
            .unwrap()
            .contains("格式不正确"));
    }
    #[test]
    fn explicit_environment_paths_override_discovery_without_mutating_global_environment() {
        let root = sandbox("env");
        let model = root.join("explicit.pt");
        let tok = root.join("explicit.model");
        file(&model);
        file(&tok);
        let found = apply_overrides(
            find_resources_at(&root, None),
            Some(model.as_os_str()),
            Some(tok.as_os_str()),
        );
        assert_eq!(found.model, model);
        assert_eq!(found.tokenizer, tok);
        assert!(!found.first_run_required);
    }

    #[test]
    fn missing_saved_weight_recovers_default_model_without_prompting_for_a_file() {
        let root = sandbox("missing-preference");
        let user = root.join("prefs");
        config(&user, "disconnected/model.pt", "gone.model");
        file(&root.join("models/model.pt"));
        file(&root.join("assets/caption_spm_1500.model"));
        let found = find_resources_at(&root, Some(&user));
        assert_eq!(found.model, root.join("models/model.pt"));
        assert!(!found.first_run_required && found.configuration_error.is_none());
        assert!(found.discovery_notice.is_some());
        assert_eq!(std::fs::read_dir(&user).unwrap().count(), 1);
    }

    #[test]
    fn valid_saved_model_uses_bundled_tokenizer_if_its_saved_tokenizer_is_gone() {
        let root = sandbox("missing-tokenizer");
        let user = root.join("prefs");
        config(&user, "chosen.pt", "gone.model");
        file(&user.join("chosen.pt"));
        file(&root.join("assets/caption_spm_1500.model"));
        let found = find_resources_at(&root, Some(&user));
        assert_eq!(found.model, user.join("chosen.pt"));
        assert_eq!(found.tokenizer, root.join("assets/caption_spm_1500.model"));
        assert!(!found.first_run_required && found.discovery_notice.is_some());
    }

    #[test]
    fn public_build_does_not_search_developer_workstation_locations() {
        let (models,tokens)=user_default_locations(Some(OsStr::new("C:/Users/ExampleUser")));
        assert!(models.is_empty() && tokens.is_empty());
    }


    #[test]
    fn partial_explicit_environment_revalidates_pair_and_missing_explicit_path_never_falls_back() {
        let root = sandbox("partial-env");
        let user = root.join("prefs");
        config(&user, "gone.pt", "gone.model");
        let model = root.join("explicit.pt");
        let token = root.join("assets/caption_spm_1500.model");
        file(&model);
        file(&token);
        let discovered = find_resources_at(&root, Some(&user));
        assert!(discovered.configuration_error.is_some());
        let overridden = apply_overrides(discovered, Some(model.as_os_str()), None);
        assert_eq!(overridden.model, model);
        assert_eq!(overridden.tokenizer, token);
        assert!(!overridden.first_run_required && overridden.configuration_error.is_none());
        let missing = root.join("explicit-missing.pt");
        let blocked = apply_overrides(
            find_resources_at(&root, Some(&user)),
            Some(missing.as_os_str()),
            None,
        );
        assert_eq!(blocked.model, missing);
        assert!(blocked.first_run_required && blocked.configuration_error.is_some());
    }

    #[test]
    fn corrupt_configuration_and_no_resources_gives_an_error_instead_of_a_false_success() {
        let root = sandbox("all-missing");
        let user = root.join("prefs");
        std::fs::create_dir_all(&user).unwrap();
        std::fs::write(user.join(CONFIG_FILE), b"{ broken").unwrap();
        let found = find_resources_at(&root, Some(&user));
        assert!(
            found.first_run_required
                && found
                    .configuration_error
                    .as_deref()
                    .unwrap()
                    .contains("设置")
        );
        assert!(found
            .discovery_notice
            .as_deref()
            .unwrap()
            .contains("格式不正确"));
        assert!(!root.join("models").exists());
    }
    #[test]
    fn verified_download_directory_is_discovered_without_changing_existing_preference(){
        let root=sandbox("download-default");let user=root.join("user");let dir=user.join("models/pop2-1b-epoch60");
        file(&dir.join("model.pt"));file(&dir.join("caption_spm_1500.model"));
        let paths=find_resources_at(&root,Some(&user));assert_eq!(paths.model,dir.join("model.pt"));assert!(!paths.first_run_required);
        file(&user.join("chosen.pt"));config(&user,"chosen.pt","models/pop2-1b-epoch60/caption_spm_1500.model");
        let paths=find_resources_at(&root,Some(&user));assert_eq!(paths.model,user.join("chosen.pt"));
        std::fs::remove_dir_all(root).unwrap();
    }

}
