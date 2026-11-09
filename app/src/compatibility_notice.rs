//! Startup compatibility explanation assembled from the actual compute context.
//! This module does not change execution, read model data, or open a window.
use crate::acceleration::{AccelerationMode, AccelerationSettings, FeaturePreference};
use crate::inference::LoadPreparationReport;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub struct NoticeContext<'a> {
    pub actual: &'a Value,
    pub requested: &'a AccelerationSettings,
    pub sampling_override: Option<bool>,
    pub preparation: Option<&'a LoadPreparationReport>,
    pub selection_reason: &'a str,
    pub session_id: &'a str,
    pub log_directory: Option<&'a Path>,
    pub application_version: &'a str,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisabledFeature {
    pub name: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatibilityNotice {
    pub current_backend: String,
    pub current_scheme: String,
    pub reason: String,
    pub disabled_features: Vec<DisabledFeature>,
    pub ai_summary: String,
    pub log_directory: Option<PathBuf>,
}

const FEATURES: [(&str, &str); 4] = [
    ("fusion", "融合算子／高效注意力"),
    ("gpu_sampling", "GPU采样"),
    ("primitive_graph", "完整图元Graph"),
    ("split_k", "Split-K注意力"),
];

impl CompatibilityNotice {
    pub fn from_context(context: &NoticeContext<'_>) -> Option<Self> {
        let actual = context.actual;
        let runtime = actual.get("runtime").filter(|v| v.is_object()).unwrap_or(actual);
        let provider = string(actual.get("provider")).unwrap_or_default();
        let backend = string(actual.pointer("/capabilities/backend")).unwrap_or_default();
        let cpu = provider == "native_candle_cpu" || provider == "cpu" || backend == "cpu";
        let amd = provider == "aten_hip" || backend == "aten_hip" || backend == "hip";
        let active = actual.get("active_features").filter(|v| v.is_object())
            .or_else(|| actual.get("selected").filter(|v| v.is_object()));
        let enabled = |key: &str| active.and_then(|v| v.get(key)).and_then(Value::as_bool).unwrap_or(false);
        let ordinary = active.is_some() && !FEATURES.iter().any(|(key, _)| enabled(key));
        let explicit = context.requested.mode == AccelerationMode::Compatible;
        let fault = string(actual.get("fault")).filter(|v| !v.is_empty());
        let prep_failed = context.preparation.is_some_and(|p| !p.ready);
        let failures = current_failures(context, runtime);
        let kv_failure = kv_int8_failure(context);
        let selection_fallback = is_selection_fallback(context.selection_reason);
        if !(cpu || ordinary || explicit || prep_failed || fault.is_some() || selection_fallback || !failures.is_empty() || kv_failure.is_some()) {
            return None;
        }

        let current_backend = if cpu { "CPU" } else if amd { "AMD / HIP / ATen" }
            else if provider.contains("cuda") || backend == "cuda" { "NVIDIA / CUDA" }
            else { "后端尚未确认" }.to_string();
        let pending = actual.get("model_loaded").and_then(Value::as_bool) == Some(false) || prep_failed || fault.is_some();
        let current_scheme = if pending {
            format!("{current_backend}（模型尚未就绪）")
        } else if cpu {
            "CPU兼容计算（生成速度较慢）".into()
        } else if ordinary {
            "普通GPU兼容计算 + CPU采样".into()
        } else if enabled("gpu_sampling") {
            "GPU计算 + GPU采样（部分可选加速未启用）".into()
        } else {
            "GPU计算 + CPU采样（部分可选加速未启用）".into()
        };
        let reason = if let Some(fault) = fault {
            format!("模型执行已停止：{}", safe_text(fault))
        } else if prep_failed {
            context.preparation.map(|p| safe_text(&p.summary)).unwrap_or_else(|| "模型准备尚未完成".into())
        } else if cpu {
            if context.selection_reason.is_empty() { "当前实际模型使用CPU兼容计算".into() }
            else { safe_text(context.selection_reason) }
        } else if explicit && (FEATURES.iter().any(|(key,_)|enabled(key)) || kv_int8_status(actual).and_then(|v|v.get("enabled")).and_then(Value::as_bool)==Some(true)) {
            "已请求普通兼容模式，但后端仍报告有加速功能启用；请依据下方实际状态检查设置是否已应用".into()
        } else if explicit {
            "已选择普通兼容模式，关闭可选加速；基础模型计算仍在GPU上执行".into()
        } else if selection_fallback {
            safe_text(context.selection_reason)
        } else if let Some(failure) = &kv_failure {
            format!("KV INT8未启用，继续使用原浮点缓存：{failure}")
        } else if let Some(failure) = failures.first() {
            format!("部分可选能力未启用：{failure}")
        } else {
            "当前采用普通GPU路径；未启用的可选能力及原因见下方".into()
        };

        let mut disabled_features = Vec::new();
        if cpu {
            disabled_features.push(DisabledFeature {
                name: "GPU模型计算".into(),
                reason: if context.selection_reason.is_empty() { "当前选择CPU兼容后端".into() } else { safe_text(context.selection_reason) },
            });
        }
        for (key, label) in FEATURES {
            if enabled(key) { continue; }
            disabled_features.push(DisabledFeature {
                name: if key == "fusion" && amd { "高效注意力".into() } else { label.into() },
                reason: feature_reason(context, runtime, key, cpu, amd, &failures),
            });
        }
        if let Some(kv) = kv_int8_status(actual).filter(|v|v.get("enabled").and_then(Value::as_bool)==Some(false)) {
            let reason = string(kv.get("reason")).filter(|v|!v.trim().is_empty()).map(safe_text)
                .unwrap_or_else(|| if cpu { "CPU兼容后端使用原浮点缓存".into() }
                    else if explicit { "普通兼容模式使用原浮点缓存".into() }
                    else if context.requested.kv_int8 == FeaturePreference::Disabled { "用户已关闭KV INT8，使用原浮点缓存".into() }
                    else { "后端报告未启用KV INT8，但未提供原因；请查看实际计算上下文".into() });
            disabled_features.push(DisabledFeature { name:"KV INT8".into(), reason });
        }
        if selection_fallback && !cpu && provider.contains("cuda")
            && string(actual.get("compiled_cuda_profile")).is_some_and(|v| v.contains("portable")) {
            disabled_features.insert(0, DisabledFeature {
                name: "设备专用CUDA执行组件".into(), reason: safe_text(context.selection_reason),
            });
        }

        let mut notice = Self {
            current_backend, current_scheme, reason, disabled_features,
            ai_summary: String::new(), log_directory: context.log_directory.map(Path::to_path_buf),
        };
        notice.ai_summary = notice.summary(context, runtime, &enabled);
        Some(notice)
    }

    fn summary(&self, context: &NoticeContext<'_>, runtime: &Value, enabled: &impl Fn(&str) -> bool) -> String {
        let a = context.actual;
        let device = string(a.get("device_name")).or_else(|| string(runtime.get("device_name")))
            .or_else(|| string(a.pointer("/architecture/device"))).unwrap_or("未提供");
        let architecture = printable(a.get("compute_capability")).or_else(|| printable(runtime.get("architecture")))
            .unwrap_or_else(|| "未提供".into());
        let ordinal = printable(a.get("actual_cuda_ordinal")).or_else(|| printable(runtime.get("device_ordinal")))
            .unwrap_or_else(|| "不适用或未提供".into());
        let driver = printable(a.get("driver_api_version")).or_else(|| printable(runtime.get("hip_driver_version")))
            .unwrap_or_else(|| "未提供".into());
        let runtime_version = printable(a.get("runtime_api_version"))
            .or_else(|| printable(runtime.get("runtime_id")))
            .unwrap_or_else(|| printable(a.get("compiled_cuda_toolkit")).map(|v| format!("API未提供；CUDA编译工具链为{v}"))
                .unwrap_or_else(|| "未提供，不推断".into()));
        let pci = printable(a.get("pci_bus_id")).or_else(|| printable(runtime.get("pci_bus_id")))
            .unwrap_or_else(|| "不适用或未提供".into());
        let target = printable(a.get("embedded_ptx_target")).or_else(|| printable(runtime.get("build_id")))
            .unwrap_or_else(|| "未提供".into());
        let dtype = printable(a.pointer("/architecture/dtype")).or_else(|| printable(runtime.get("dtype")))
            .unwrap_or_else(|| "未提供".into());
        let on = FEATURES.iter().filter(|(key, _)| enabled(key)).map(|(_, name)| *name).collect::<Vec<_>>();
        let mut text = format!(
            "Primitive Painter 兼容诊断摘要\n应用版本：{}\n当前后端：{}\n当前方案：{}\n设备：{}\n计算设备编号：{}\nPCI位置：{}\n架构／计算能力：{}\n驱动API：{}\n运行组件：{}\n内核目标／构建：{}\n计算精度：{}\n兼容原因：{}\n已启用的可选加速：{}\n",
            safe_text(context.application_version), self.current_backend, self.current_scheme, safe_text(device),
            ordinal, pci, architecture, driver, runtime_version, target, dtype, self.reason,
            if on.is_empty() { "无".into() } else { on.join("、") },
        );
        text.push_str(&kv_int8_summary(context));
        text.push_str("未启用的功能：\n");
        for row in &self.disabled_features { text.push_str(&format!("- {}：{}\n", row.name, row.reason)); }
        text.push_str(&format!("日志会话：{}\n日志目录：{}\n",
            safe_text(context.session_id),
            context.log_directory.map(|p| safe_text(&p.to_string_lossy())).unwrap_or_else(|| "目录不可写；诊断保留在内存".into())));
        text.push_str("排查入口：先查看 model_compute_context 和 model_effective_configuration，再查看 provider_preflight、acceleration_prepared、cpu_compatibility_fallback 与错误链。请区分主动兼容选择、性能策略与真正的准备失败。\n此摘要未包含原始提示词、图元或权重数据；详细日志可在设置中导出。");
        text
    }
}

fn kv_int8_status(actual: &Value) -> Option<&Value> {
    actual.get("kv_int8").filter(|v|v.is_object())
        .or_else(||actual.pointer("/runtime/kv_int8").filter(|v|v.is_object()))
}

fn kv_int8_failure(context: &NoticeContext<'_>) -> Option<String> {
    // An explicitly disabled user setting never creates a warning on its own.
    if context.requested.kv_int8 == FeaturePreference::Disabled { return None; }
    let status = kv_int8_status(context.actual)?;
    if status.get("enabled").and_then(Value::as_bool) != Some(false) { return None; }
    if string(status.get("requested")) == Some("disabled") { return None; }
    string(status.get("reason")).filter(|v|!v.trim().is_empty()).map(safe_text)
}

fn kv_int8_summary(context: &NoticeContext<'_>) -> String {
    let preference = match context.requested.kv_int8 {
        FeaturePreference::Auto => "auto", FeaturePreference::Enabled => "enabled", FeaturePreference::Disabled => "disabled",
    };
    let Some(status) = kv_int8_status(context.actual) else {
        return format!("KV INT8：用户设置={preference}；后端未提供实际状态，不推断是否启用或存储精度\n");
    };
    let requested = string(status.get("requested")).map(safe_text).unwrap_or_else(||"未提供".into());
    let bool_text = |key| match status.get(key).and_then(Value::as_bool) { Some(true)=>"true",Some(false)=>"false",None=>"未提供" };
    let storage = string(status.get("storage")).filter(|v|!v.is_empty()).map(safe_text).unwrap_or_else(||"未提供，不推断".into());
    let reason = string(status.get("reason")).filter(|v|!v.trim().is_empty()).map(safe_text).unwrap_or_else(||"未提供".into());
    let mut text = format!("KV INT8实际状态：requested={requested}，available={}，enabled={}，storage={storage}；用户设置={preference}\nKV INT8原因：{reason}\n",bool_text("available"),bool_text("enabled"));
    if status.get("enabled").and_then(Value::as_bool)==Some(true)
        && (context.requested.mode==AccelerationMode::Compatible || context.requested.kv_int8==FeaturePreference::Disabled || status.get("available").and_then(Value::as_bool)==Some(false)) {
        text.push_str("KV INT8状态冲突：后端仍报告已启用，请检查设置应用与后端状态；以上保留实际报告，不按设置覆盖。\n");
    }
    text
}

fn string(value: Option<&Value>) -> Option<&str> { value.and_then(Value::as_str) }
fn printable(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Null | Value::Object(_) => None,
        Value::String(v) if v.is_empty() => None,
        Value::String(v) => Some(safe_text(v)),
        Value::Number(v) => Some(v.to_string()),
        Value::Array(v) if v.len() <= 4 && v.iter().all(Value::is_number) => Some(v.iter().map(Value::to_string).collect::<Vec<_>>().join(".")),
        _ => None,
    }
}

fn is_selection_fallback(reason: &str) -> bool {
    let lower = reason.to_ascii_lowercase();
    ["组件暂不可用", "组件缺失", "独立优化组件缺失", "已回退", "已转为", "已启用CPU兼容"].iter().any(|v| reason.contains(v))
        || lower.contains("fallback")
}
fn looks_failed(reason: &str) -> bool {
    let lower = reason.to_ascii_lowercase();
    if lower.contains("deferred until explicit") || reason.contains("自动模式不需要") { return false; }
    ["失败", "未通过", "不可用", "不适用", "资源不足", "显存不足"].iter().any(|v| reason.contains(v))
        || ["unsupported", "not supported", "unavailable", "failed", "mismatch", "out of memory", "error:"].iter().any(|v| lower.contains(v))
}
fn current_failures(context: &NoticeContext<'_>, runtime: &Value) -> Vec<String> {
    let mut values = Vec::new();
    let active=context.actual.get("active_features").or_else(||context.actual.get("selected"));
    let enabled=|key:&str|active.and_then(|v|v.get(key)).and_then(Value::as_bool)==Some(true);
    if let Some(reasons) = runtime.get("disabled_reasons").and_then(Value::as_object) {
        values.extend(reasons.iter().filter(|(key,_)|!enabled(if key.as_str()=="efficient_attention"{"fusion"}else if key.as_str()=="hip_graph"{"primitive_graph"}else{key.as_str()}))
            .filter_map(|(_,v)|v.as_str()).filter(|v| looks_failed(v)).map(safe_text));
    }
    if let Some(p) = context.preparation {
        values.extend(p.details.iter().filter(|v| looks_failed(v)).map(|v| safe_text(v)));
    }
    if let Some(candidates) = context.actual.pointer("/cuda_private_auto_probe/candidates").and_then(Value::as_array) {
        values.extend(candidates.iter().filter(|v|["fusion","gpu_sampling"].iter().any(|key|v.get(*key).and_then(Value::as_bool)==Some(true)&&!enabled(key)))
            .filter_map(|v| v.get("error").and_then(Value::as_str)).filter(|v| !v.is_empty()).map(safe_text));
    }
    values.sort(); values.dedup(); values.truncate(8); values
}
fn feature_reason(context: &NoticeContext<'_>, runtime: &Value, key: &str, cpu: bool, amd: bool, failures: &[String]) -> String {
    if cpu { return "当前模型使用CPU兼容后端，GPU加速不适用".into(); }
    if context.requested.mode == AccelerationMode::Compatible { return "普通兼容模式主动关闭此项".into(); }
    let preference = match key { "fusion" => Some(context.requested.fusion), "primitive_graph" => Some(context.requested.primitive_graph), "split_k" => Some(context.requested.split_k), _ => None };
    if preference == Some(FeaturePreference::Disabled) { return "用户已关闭此项".into(); }
    if key == "gpu_sampling" && context.sampling_override == Some(false) { return "用户选择CPU采样；模型计算后端保持不变".into(); }
    let native_key = if key == "fusion" { "efficient_attention" } else if key == "primitive_graph" { "hip_graph" } else { key };
    if let Some(reason) = runtime.pointer(&format!("/disabled_reasons/{native_key}")).and_then(Value::as_str).filter(|v| !v.is_empty()) { return safe_text(reason); }
    let needles: &[&str] = match key { "fusion" => &["融合", "注意力", "attention", "fusion", "rms", "rope"], "gpu_sampling" => &["采样", "sampl"], "primitive_graph" => &["graph", "捕获"], _ => &["split-k", "splitk"] };
    if let Some(reason) = failures.iter().find(|reason| { let lower=reason.to_ascii_lowercase(); needles.iter().any(|n| lower.contains(n)) }) { return reason.clone(); }
    if context.actual.pointer(&format!("/capabilities/available/{key}")).and_then(Value::as_bool) == Some(false) {
        return if amd && key == "split_k" { "该原生后端不使用独立Split-K开关".into() } else { "当前设备、模型或后端未提供已验证的此项能力".into() };
    }
    match key {
        "primitive_graph" => "当前自动执行策略未选用Graph；按实际生命周期与收益选择".into(),
        "split_k" => "自动模式未确认独立Split-K有收益，保持关闭".into(),
        "gpu_sampling" => "当前选择CPU采样，或自动短测没有确认GPU采样组合的收益".into(),
        _ => "自动验证或短测未确认适用且有收益的融合组合".into(),
    }
}

// Only whitelisted metadata is copied. Reasons may contain local runtime
// errors, so remove credential/URL payloads and never echo input validation.
fn safe_text(input: &str) -> String {
    let lower=input.to_ascii_lowercase();
    if input.contains("PRIVATE KEY-----") { return "[私钥内容已省略]".into(); }
    if input.contains("提示词") || lower.contains("prompt=") || lower.contains("prompt:") { return "输入相关细节已省略；请查看本机诊断日志".into(); }
    let mut text=input.chars().take(2048).map(|c|if c.is_control(){' '}else{c}).collect::<String>();
    for marker in ["password=","password:","passwd=","api_key=","access_token=","refresh_token=","authorization:","bearer "] {
        if let Some(at)=text.to_ascii_lowercase().find(marker){text.truncate(at);text.push_str("[凭据内容已省略]");}
    }
    let mut out=String::new();let mut tail=text.as_str();
    loop {
        let found=[tail.find("https://"),tail.find("http://")].into_iter().flatten().min();
        let Some(at)=found else {out.push_str(tail);break};
        out.push_str(&tail[..at]);let url=&tail[at..];
        let end=url.char_indices().find(|(_,c)|c.is_whitespace()||matches!(c,'"'|'\''|'<'|'>'|'，'|'。')).map(|(i,_)|i).unwrap_or(url.len());
        let url=&url[..end];let no_query=url.split(['?','#']).next().unwrap_or(url);
        let scheme=no_query.find("://").unwrap()+3;out.push_str(&no_query[..scheme]);
        let after=&no_query[scheme..];out.push_str(after.rsplit_once('@').map_or(after,|(_,rest)|rest));
        tail=&tail[at+end..];
    }
    if let Ok(profile)=std::env::var("USERPROFILE") {if !profile.is_empty(){out=replace_ascii_case(&out,&profile,"%USERPROFILE%");out=replace_ascii_case(&out,&profile.replace('\\',"/"),"%USERPROFILE%");}}
    out.trim().to_string()
}
fn replace_ascii_case(input:&str,from:&str,to:&str)->String {
    if from.is_empty(){return input.into()}
    let mut tail=input;let mut out=String::new();
    while let Some(at)=tail.as_bytes().windows(from.len()).position(|v|v.eq_ignore_ascii_case(from.as_bytes())){
        out.push_str(&tail[..at]);out.push_str(to);tail=&tail[at+from.len()..];
    }
    out.push_str(tail);out
}

#[cfg(test)]mod tests {
    use super::*;use serde_json::json;
    fn settings()->AccelerationSettings{AccelerationSettings::default()}
    fn context<'a>(actual:&'a Value,settings:&'a AccelerationSettings)->NoticeContext<'a>{NoticeContext{actual,requested:settings,sampling_override:None,preparation:None,selection_reason:"",session_id:"session-1",log_directory:Some(Path::new("C:/logs/session-1")),application_version:"0.9.3"}}
    fn healthy_amd()->Value{json!({"provider":"aten_hip","runtime":{"device_name":"7900XTX","architecture":"gfx1100","hip_driver_version":"70152802","runtime_id":"native-rocm","disabled_reasons":{"hip_graph":"deferred until explicit Graph request; Auto uses measured eager paths"}},"capabilities":{"backend":"aten_hip","available":{"fusion":true,"gpu_sampling":true,"primitive_graph":false,"split_k":false}},"selected":{"fusion":true,"gpu_sampling":true,"primitive_graph":false}})}
    #[test]fn normal_auto_graph_and_splitk_omissions_do_not_claim_compatibility_failure(){let a=healthy_amd();assert!(CompatibilityNotice::from_context(&context(&a,&settings())).is_none());}
    #[test]fn actual_cpu_fallback_is_not_hidden_by_auto_settings(){let a=json!({"provider":"native_candle_cpu","architecture":{"device":"CPU","dtype":"F32"},"active_features":{"fusion":false,"gpu_sampling":false,"primitive_graph":false,"split_k":false}});let s=settings();let mut c=context(&a,&s);c.selection_reason="GPU组件暂不可用，已启用CPU兼容";let n=CompatibilityNotice::from_context(&c).unwrap();assert_eq!(n.current_backend,"CPU");assert!(n.disabled_features.iter().any(|r|r.name=="GPU模型计算"));assert!(n.ai_summary.contains("F32"));assert!(n.ai_summary.contains("GPU组件暂不可用"));}
    #[test]fn explicit_gpu_compatibility_lists_all_disabled_features_without_claiming_cpu_model(){let mut a=healthy_amd();a["selected"]=json!({"fusion":false,"gpu_sampling":false,"primitive_graph":false});let mut s=settings();s.mode=AccelerationMode::Compatible;let n=CompatibilityNotice::from_context(&context(&a,&s)).unwrap();assert_eq!(n.disabled_features.len(),4);assert!(n.current_backend.contains("AMD"));assert!(n.reason.contains("基础模型计算仍在GPU"));}
    #[test]fn failed_preparation_of_one_feature_is_visible_even_if_others_are_fast(){let mut a=healthy_amd();a["runtime"]["disabled_reasons"]["gpu_sampling"]=json!("GPU sampling preparation failed: unsupported kernel");a["selected"]["gpu_sampling"]=json!(false);let s=settings();let n=CompatibilityNotice::from_context(&context(&a,&s)).unwrap();assert!(n.disabled_features.iter().any(|r|r.name=="GPU采样"&&r.reason.contains("unsupported kernel")));}
    #[test]fn healthy_partial_performance_choice_is_not_mislabelled_as_a_failure(){let a=json!({"provider":"native_candle_cuda","capabilities":{"backend":"cuda","available":{"fusion":true,"gpu_sampling":true,"primitive_graph":true,"split_k":true}},"active_features":{"fusion":true,"gpu_sampling":false,"primitive_graph":false,"split_k":false},"cuda_private_auto_probe":{"status":"complete","candidates":[{"name":"fusion_cpu","error":null}]}});assert!(CompatibilityNotice::from_context(&context(&a,&settings())).is_none());}
    #[test]fn all_ordinary_auto_reports_the_current_choice(){let a=json!({"provider":"native_candle_cuda","active_features":{"fusion":false,"gpu_sampling":false,"primitive_graph":false,"split_k":false}});let n=CompatibilityNotice::from_context(&context(&a,&settings())).unwrap();assert!(n.reason.contains("普通GPU路径"));assert_eq!(n.disabled_features.len(),4);}
    #[test]fn compact_summary_never_embeds_raw_diagnostic_payloads_or_credentials(){let a=json!({"provider":"native_candle_cpu","architecture":{"device":"CPU","dtype":"F32"},"prompt":"SECRET_CAPTION","tokens":[123456789],"weights":"SECRET_WEIGHTS","active_features":{"fusion":false}});let s=settings();let mut c=context(&a,&s);c.selection_reason="GPU组件暂不可用 https://user:SECRET_PASS@example.test/path?token=SECRET_TOKEN password=SECRET_PASSWORD";let n=CompatibilityNotice::from_context(&c).unwrap();for secret in ["SECRET_CAPTION","123456789","SECRET_WEIGHTS","SECRET_PASS","SECRET_TOKEN","SECRET_PASSWORD"]{assert!(!n.ai_summary.contains(secret),"leaked {secret}");}assert!(n.ai_summary.contains("example.test/path"));}
    #[test]fn pending_or_faulted_models_are_not_described_as_ready(){let a=json!({"provider":"cpu","model_loaded":false});let n=CompatibilityNotice::from_context(&context(&a,&settings())).unwrap();assert!(n.current_scheme.contains("尚未就绪"));}
    #[test]fn failed_alternative_does_not_hide_a_healthy_selected_combination(){let mut a=healthy_amd();a["cuda_private_auto_probe"]=json!({"candidates":[{"fusion":true,"gpu_sampling":false,"error":"unsupported alternative"}]});assert!(CompatibilityNotice::from_context(&context(&a,&settings())).is_none());}
    #[test]fn requested_mode_never_overwrites_reported_actual_feature_states(){let a=healthy_amd();let mut s=settings();s.mode=AccelerationMode::Compatible;let n=CompatibilityNotice::from_context(&context(&a,&s)).unwrap();assert!(n.current_scheme.contains("GPU采样"));assert!(n.reason.contains("仍报告"));assert!(!n.disabled_features.iter().any(|r|r.name=="GPU采样"));}
    #[test]fn profile_redaction_is_case_insensitive(){assert_eq!(replace_ascii_case("C:/USERS/Alice/logs","c:/users/alice","%USERPROFILE%"),"%USERPROFILE%/logs");}
    #[test]fn actual_gpu_context_is_authoritative_over_a_stale_cpu_routing_hint(){let a=healthy_amd();let s=settings();let mut c=context(&a,&s);c.selection_reason="上一阶段已回退CPU；现在已恢复";let n=CompatibilityNotice::from_context(&c).unwrap();assert!(n.current_backend.contains("AMD"));assert!(!n.disabled_features.iter().any(|r|r.name=="GPU模型计算"));}
    fn kv_status(requested:&str,available:bool,enabled:bool,reason:&str,storage:&str)->Value{json!({"requested":requested,"available":available,"enabled":enabled,"reason":reason,"storage":storage})}
    #[test]fn healthy_auto_int8_does_not_open_a_compatibility_warning(){let mut a=healthy_amd();a["kv_int8"]=kv_status("auto",true,true,"per-token/head INT8 cache active","int8_per_token_head");assert!(CompatibilityNotice::from_context(&context(&a,&settings())).is_none());}
    #[test]fn auto_int8_failure_is_visible_while_other_gpu_features_keep_working(){let mut a=healthy_amd();a["kv_int8"]=kv_status("auto",false,false,"device does not provide the INT8 cache kernel","bf16");let n=CompatibilityNotice::from_context(&context(&a,&settings())).unwrap();assert!(n.reason.contains("KV INT8未启用"));assert!(n.current_scheme.contains("GPU采样"));assert!(n.disabled_features.iter().any(|r|r.name=="KV INT8"&&r.reason.contains("INT8 cache kernel")));assert!(n.ai_summary.contains("available=false，enabled=false，storage=bf16"));}
    #[test]fn explicit_int8_request_failure_is_reported_from_nested_runtime(){let mut a=healthy_amd();a["runtime"]["kv_int8"]=kv_status("enabled",false,false,"cache format not supported","bf16");let mut s=settings();s.kv_int8=FeaturePreference::Enabled;let n=CompatibilityNotice::from_context(&context(&a,&s)).unwrap();assert!(n.ai_summary.contains("requested=enabled"));assert!(n.disabled_features.iter().any(|r|r.name=="KV INT8"&&r.reason=="cache format not supported"));}
    #[test]fn disabled_int8_is_not_a_warning_even_with_a_stale_backend_request(){let mut a=healthy_amd();a["kv_int8"]=kv_status("auto",false,false,"kernel unavailable","bf16");let mut s=settings();s.kv_int8=FeaturePreference::Disabled;assert!(CompatibilityNotice::from_context(&context(&a,&s)).is_none());a["kv_int8"]["requested"]=json!("disabled");assert!(CompatibilityNotice::from_context(&context(&a,&settings())).is_none());}
    #[test]fn no_reason_or_no_actual_enabled_state_cannot_invent_an_int8_failure(){for value in [kv_status("auto",false,false,"   ","bf16"),json!({"requested":"auto","available":false,"reason":"unsupported"})]{let mut a=healthy_amd();a["kv_int8"]=value;assert!(CompatibilityNotice::from_context(&context(&a,&settings())).is_none());}}
    #[test]fn top_level_actual_int8_state_takes_precedence_over_stale_runtime(){let mut a=healthy_amd();a["kv_int8"]=kv_status("auto",true,true,"active","int8_per_token_head");a["runtime"]["kv_int8"]=kv_status("auto",false,false,"old failure","bf16");assert!(CompatibilityNotice::from_context(&context(&a,&settings())).is_none());}
    #[test]fn cpu_and_explicit_compatible_reasons_preserve_actual_float_storage(){for(cpu,mode,storage,reason)in[(true,AccelerationMode::Auto,"fp32","CPU backend uses FP32 cache"),(false,AccelerationMode::Compatible,"bf16","compatible mode preserves floating point cache")]{let mut a=healthy_amd();if cpu{a["provider"]=json!("native_candle_cpu");}a["kv_int8"]=kv_status("auto",false,false,reason,storage);let mut s=settings();s.mode=mode;let n=CompatibilityNotice::from_context(&context(&a,&s)).unwrap();assert!(n.disabled_features.iter().any(|r|r.name=="KV INT8"&&r.reason==reason));assert!(n.ai_summary.contains(&format!("storage={storage}")));}}
    #[test]fn actual_enabled_int8_is_not_overwritten_by_compatible_settings(){let mut a=healthy_amd();a["kv_int8"]=kv_status("auto",true,true,"active","int8_per_token_head");let mut s=settings();s.mode=AccelerationMode::Compatible;s.kv_int8=FeaturePreference::Disabled;let n=CompatibilityNotice::from_context(&context(&a,&s)).unwrap();assert!(!n.disabled_features.iter().any(|r|r.name=="KV INT8"));assert!(n.ai_summary.contains("enabled=true"));assert!(n.ai_summary.contains("状态冲突"));}
    #[test]fn missing_int8_report_is_unknown_and_does_not_claim_enabled_or_disabled(){let mut s=settings();s.mode=AccelerationMode::Compatible;let n=CompatibilityNotice::from_context(&context(&healthy_amd(),&s)).unwrap();assert!(n.ai_summary.contains("后端未提供实际状态，不推断"));assert!(!n.disabled_features.iter().any(|r|r.name=="KV INT8"));}
    #[test]fn int8_reason_uses_the_same_credential_redaction_as_other_diagnostics(){let mut a=healthy_amd();a["kv_int8"]=kv_status("auto",false,false,"kernel unavailable password=SECRET_KV_CREDENTIAL","bf16");let n=CompatibilityNotice::from_context(&context(&a,&settings())).unwrap();assert!(!n.ai_summary.contains("SECRET_KV_CREDENTIAL"));assert!(n.ai_summary.contains("凭据内容已省略"));}
}
