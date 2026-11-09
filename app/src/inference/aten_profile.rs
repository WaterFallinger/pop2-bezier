//! Measurements belong to an actual device/software/model/workload combination.
//! No fixed-PC admission list is used; absent entries require a private probe.
use crate::acceleration::FeatureSet;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

pub(super) const ADAPTIVE_PROBE_VERSION: u32 = 1;
pub(super) const ADAPTIVE_PREFIX_RECORDS: usize = 128;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct AdaptiveWorkload {
    pub physical_rows: usize,
    pub guidance: usize,
    pub scope: String,
    pub cfg_bits: u64,
    pub alpha_bits: u64,
    pub candidate: FeatureSet,
    #[serde(default = "bf16_storage")]
    pub kv_storage: String,
}
fn bf16_storage() -> String {
    "bf16".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct AdaptiveRecord {
    pub key: String,
    pub workload: AdaptiveWorkload,
    pub selected: FeatureSet,
    pub status: String,
    pub reason: String,
    pub probe: Value,
    pub observed_unix_millis: u64,
    pub persistent: bool,
}

#[derive(Serialize, Deserialize)]
struct AdaptiveFile {
    schema_version: u32,
    identity: Value,
    records: Vec<AdaptiveRecord>,
}

pub(super) struct AdaptiveStore {
    identity: Value,
    path: Option<PathBuf>,
    records: BTreeMap<String, AdaptiveRecord>,
    issues: Vec<String>,
    hits: usize,
    probes: usize,
}

fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn adaptive_identity(info: &Value) -> Value {
    let keys = [
        "provider",
        "abi_version",
        "build_id",
        "runtime_id",
        "pci_bus_id",
        "device_name",
        "device_ordinal",
        "architecture",
        "hip_version",
        "hip_driver_version",
        "torch_version",
        "dtype",
        "layers",
        "width",
        "heads",
        "ffn_hidden",
        "head_width",
    ];
    let mut identity = serde_json::Map::new();
    identity.insert(
        "private_probe_version".into(),
        ADAPTIVE_PROBE_VERSION.into(),
    );
    for key in keys {
        identity.insert(key.into(), info.get(key).cloned().unwrap_or(Value::Null));
    }
    // A development runtime without a verified package id is still eligible
    // for private measurement. Its actual loaded files disambiguate caching.
    if info["runtime_id"].as_str().is_none_or(str::is_empty) {
        identity.insert(
            "development_libraries".into(),
            info["loaded_libraries"].clone(),
        );
    }
    Value::Object(identity)
}

impl AdaptiveStore {
    pub fn new(info: &Value) -> Self {
        let identity = adaptive_identity(info);
        let root = std::env::var_os("PAINTER_ATEN_AUTO_CACHE_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("LOCALAPPDATA").map(|p| {
                    PathBuf::from(p)
                        .join("PrimitivePainter")
                        .join("Cache")
                        .join("aten-auto-v1")
                })
            });
        let mut result = Self {
            identity,
            path: None,
            records: BTreeMap::new(),
            issues: Vec::new(),
            hits: 0,
            probes: 0,
        };
        if let Some(root) = root {
            if !root.is_absolute() {
                result
                    .issues
                    .push("自动适配缓存目录不是绝对路径，本轮仅使用内存缓存".into());
                return result;
            }
            let bytes = serde_json::to_vec(&result.identity).unwrap_or_default();
            result.path = Some(root.join(format!("{}.json", digest(&bytes))));
            result.read();
        }
        result
    }
    fn read(&mut self) {
        let Some(path) = &self.path else { return };
        let metadata = match std::fs::metadata(path) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(e) => {
                self.issues.push(format!("自动适配缓存不可读取：{e}"));
                return;
            }
        };
        let read = (|| -> anyhow::Result<AdaptiveFile> {
            anyhow::ensure!(
                metadata.is_file() && metadata.len() <= 1_048_576,
                "缓存类型或大小无效"
            );
            let file: AdaptiveFile = serde_json::from_slice(&std::fs::read(path)?)?;
            anyhow::ensure!(
                file.schema_version == 1
                    && file.identity == self.identity
                    && file.records.len() <= 128,
                "缓存身份或版本不匹配"
            );
            Ok(file)
        })();
        match read {
            Ok(file) => {
                for record in file.records {
                    if record.persistent
                        && record.key == self.key(&record.workload)
                        && valid_cached_record(&record)
                    {
                        self.records.insert(record.key.clone(), record);
                    }
                }
            }
            Err(error) => self
                .issues
                .push(format!("自动适配缓存已忽略，将重新私有实测：{error:#}")),
        }
    }
    pub fn key(&self, workload: &AdaptiveWorkload) -> String {
        digest(
            &serde_json::to_vec(&serde_json::json!({"identity":self.identity,"workload":workload}))
                .unwrap_or_default(),
        )
    }
    pub fn lookup(&mut self, workload: &AdaptiveWorkload) -> Option<AdaptiveRecord> {
        let result = self.records.get(&self.key(workload)).cloned();
        if result.is_some() {
            self.hits += 1;
        }
        result
    }
    pub fn remember(&mut self, mut record: AdaptiveRecord) -> Option<String> {
        record.key = self.key(&record.workload);
        self.probes += 1;
        self.records.insert(record.key.clone(), record);
        if self.records.len() > 128 {
            let oldest = self
                .records
                .iter()
                .min_by_key(|(_, r)| r.observed_unix_millis)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest {
                self.records.remove(&key);
            }
        }
        let write = (|| -> anyhow::Result<()> {
            let Some(path) = &self.path else {
                return Ok(());
            };
            let parent = path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("缓存目录无效"))?;
            std::fs::create_dir_all(parent)?;
            let file = AdaptiveFile {
                schema_version: 1,
                identity: self.identity.clone(),
                records: self
                    .records
                    .values()
                    .filter(|r| r.persistent)
                    .cloned()
                    .collect(),
            };
            let bytes = serde_json::to_vec_pretty(&file)?;
            anyhow::ensure!(bytes.len() <= 1_048_576, "自动适配缓存超过大小限制");
            let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
            std::fs::write(&temporary, bytes)?;
            if let Err(error) = std::fs::rename(&temporary, path) {
                let _ = std::fs::remove_file(&temporary);
                return Err(error.into());
            }
            Ok(())
        })();
        match write {
            Ok(()) => None,
            Err(error) => {
                let reason = format!("自动适配缓存写入失败，本轮选择仍有效：{error:#}");
                self.issues.push(reason.clone());
                Some(reason)
            }
        }
    }
    pub fn diagnostic_info(&self) -> Value {
        serde_json::json!({"source":"adaptive_private_probe","identity":self.identity,"cache_path":self.path,
            "cache_hits":self.hits,"private_probes":self.probes,"cache_issues":self.issues,
            "measurements":self.records.values().collect::<Vec<_>>()})
    }
}

fn valid_cached_record(record: &AdaptiveRecord) -> bool {
    let w = &record.workload;
    let cfg = f64::from_bits(w.cfg_bits);
    let alpha = f64::from_bits(w.alpha_bits);
    w.physical_rows > 0
        && matches!(w.guidance, 1 | 2)
        && w.physical_rows % w.guidance == 0
        && u32::try_from(w.physical_rows / w.guidance).is_ok()
        && w.physical_rows
            .checked_mul(18 * std::mem::size_of::<u32>())
            .is_some_and(|n| n <= isize::MAX as usize)
        && cfg.is_finite()
        && cfg >= 0.
        && alpha.is_finite()
        && alpha > 1.
        && matches!(w.scope.as_str(), "global" | "regional")
        && matches!(w.kv_storage.as_str(), "bf16" | "int8_per_token_head")
        && !w.candidate.primitive_graph
        && !w.candidate.split_k
        && (!record.selected.fusion || w.candidate.fusion)
        && (!record.selected.gpu_sampling || w.candidate.gpu_sampling)
        && !record.selected.primitive_graph
        && !record.selected.split_k
        && (record.status == "measured" || record.status == "disabled")
        && record.status == record.probe["status"].as_str().unwrap_or("")
        && record.selected == adaptive_record(w.clone(), record.probe.clone()).selected
}

pub(super) fn adaptive_record(workload: AdaptiveWorkload, probe: Value) -> AdaptiveRecord {
    let status = probe["status"].as_str().unwrap_or("disabled").to_owned();
    let mut selected = FeatureSet::default();
    let mut reason = probe["reason"]
        .as_str()
        .unwrap_or("私有适配未取得有效测量，保留普通路径")
        .to_owned();
    if status == "measured" {
        let parse = |key: &str| {
            probe[key]
                .as_array()
                .filter(|v| v.len() == 2)
                .and_then(|v| v.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())
                .filter(|v| v.iter().all(|x| x.is_finite() && *x > 0.))
        };
        if let (Some(base), Some(candidate)) =
            (parse("baseline_seconds"), parse("candidate_seconds"))
        {
            let baseline = (base[0] + base[1]) * 0.5;
            let optimized = (candidate[0] + candidate[1]) * 0.5;
            let gain = baseline / optimized - 1.;
            let separated = candidate.iter().copied().fold(0., f64::max)
                < base.iter().copied().fold(f64::INFINITY, f64::min) * 0.98;
            if gain > 0.10 && separated {
                selected = workload.candidate;
                reason=format!("本机实际模型私有ABBA短测选择eager加速组合：平均延迟降低约{:.1}%，两轮区间分离；这不是作品生成速度统计",100.*(1.-optimized/baseline));
            } else {
                reason = format!(
                    "私有短测加速收益约{:.1}%且不足以越过10%及噪声门槛，Auto保留普通路径",
                    gain * 100.
                );
            }
        } else {
            reason = "私有适配返回的测量无效，保留普通路径".into();
        }
    }
    let persistent = status == "measured" || status == "disabled";
    AdaptiveRecord {
        key: String::new(),
        workload,
        selected,
        status,
        reason,
        probe,
        persistent,
        observed_unix_millis: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u64::MAX as u128) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn workload() -> AdaptiveWorkload {
        AdaptiveWorkload {
            physical_rows: 8,
            guidance: 1,
            scope: "global".into(),
            cfg_bits: 1f64.to_bits(),
            alpha_bits: 4f64.to_bits(),
            candidate: FeatureSet {
                fusion: true,
                gpu_sampling: true,
                ..Default::default()
            },
            kv_storage: bf16_storage(),
        }
    }
    fn measured(base: [f64; 2], candidate: [f64; 2]) -> Value {
        serde_json::json!({"status":"measured","method":"private_eager_abba_v1","private_probe":true,
            "baseline_seconds":base,"candidate_seconds":candidate})
    }
    fn store(info: &Value) -> AdaptiveStore {
        AdaptiveStore {
            identity: adaptive_identity(info),
            path: None,
            records: BTreeMap::new(),
            issues: vec![],
            hits: 0,
            probes: 0,
        }
    }
    #[test]
    fn clear_two_pair_gain_selects_eager_without_graph() {
        let row = adaptive_record(workload(), measured([0.8, 0.82], [0.4, 0.42]));
        assert!(row.selected.fusion && row.selected.gpu_sampling && !row.selected.primitive_graph);
        assert!(row.persistent && valid_cached_record(&row));
    }
    #[test]
    fn private_measurements_above_sixteen_rows_remain_cacheable() {
        for logical in [17, 96, 240, 1024] {
            for guidance in [1, 2] {
                let mut w = workload();
                w.physical_rows = logical * guidance;
                w.guidance = guidance;
                let row = adaptive_record(w, measured([1., 1.], [0.4, 0.4]));
                assert!(row.persistent && valid_cached_record(&row));
            }
        }
    }
    #[test]
    fn small_or_noisy_or_invalid_results_keep_same_gpu_ordinary() {
        for probe in [
            measured([1., 1.], [0.95, 0.95]),
            measured([0.4, 1.], [0.3, 0.5]),
            measured([1., 1.], [0., 0.3]),
            serde_json::json!({"status":"measured","baseline_seconds":[1.,1.]}),
            serde_json::json!({"status":"disabled","reason":"unsupported candidate"}),
        ] {
            assert_eq!(
                adaptive_record(workload(), probe).selected,
                FeatureSet::default()
            );
        }
    }
    #[test]
    fn cache_is_bound_to_actual_software_model_and_workload_not_a_fixed_pc_list() {
        let info = serde_json::json!({"provider":"aten_hip","runtime_id":"verified-package","build_id":"abc",
            "device_name":"a-new-amd-gpu","architecture":"gfx-future","hip_version":"new","width":1536});
        let mut cache = store(&info);
        let first = workload();
        assert!(cache.lookup(&first).is_none());
        cache.remember(adaptive_record(
            first.clone(),
            measured([1., 1.], [0.4, 0.4]),
        ));
        assert!(cache.lookup(&first).is_some());
        for changed in [
            AdaptiveWorkload {
                physical_rows: 7,
                ..first.clone()
            },
            AdaptiveWorkload {
                guidance: 2,
                ..first.clone()
            },
            AdaptiveWorkload {
                scope: "regional".into(),
                ..first.clone()
            },
            AdaptiveWorkload {
                alpha_bits: 7f64.to_bits(),
                ..first.clone()
            },
            AdaptiveWorkload {
                candidate: FeatureSet::default(),
                ..first.clone()
            },
            AdaptiveWorkload {
                kv_storage: "int8_per_token_head".into(),
                ..first.clone()
            },
        ] {
            assert!(cache.lookup(&changed).is_none());
        }
        for key in [
            "runtime_id",
            "build_id",
            "device_name",
            "architecture",
            "hip_version",
            "width",
        ] {
            let mut changed = info.clone();
            changed[key] = "different".into();
            assert_ne!(cache.key(&first), store(&changed).key(&first));
        }
    }
    #[test]
    fn development_runtime_paths_are_part_of_identity_only_without_verified_runtime_id() {
        let info = serde_json::json!({"runtime_id":"verified-package","loaded_libraries":{"hip":"E:/old"}});
        let mut moved = info.clone();
        moved["loaded_libraries"] = serde_json::json!({"hip":"E:/new"});
        assert_eq!(adaptive_identity(&info), adaptive_identity(&moved));
        let mut dev = info;
        dev["runtime_id"] = "".into();
        moved["runtime_id"] = "".into();
        assert_ne!(adaptive_identity(&dev), adaptive_identity(&moved));
    }
    #[test]
    fn persisted_choice_cannot_grant_unmeasured_graph_or_unsupported_features() {
        let mut row = adaptive_record(workload(), measured([1., 1.], [0.4, 0.4]));
        row.selected.primitive_graph = true;
        assert!(!valid_cached_record(&row));
        row.selected.primitive_graph = false;
        row.probe = measured([1., 1.], [1.1, 1.1]);
        assert!(!valid_cached_record(&row));
        row.selected = FeatureSet::default();
        row.workload.physical_rows = usize::MAX;
        assert!(!valid_cached_record(&row));
    }
    #[test]
    fn low_memory_deferral_stays_in_memory_without_persisting_a_permanent_slow_path() {
        let row = adaptive_record(
            workload(),
            serde_json::json!({"status":"deferred","reason":"private memory headroom"}),
        );
        assert!(!row.persistent);
        assert_eq!(row.selected, FeatureSet::default());
    }
    fn unique_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "painter-auto-{label}-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
    #[test]
    fn measured_cache_round_trip_ignores_changed_identity_and_rechecks_timing() {
        let path = unique_path("roundtrip");
        let info = serde_json::json!({"build_id":"private-build"});
        let mut original = store(&info);
        original.path = Some(path.clone());
        assert!(original
            .remember(adaptive_record(workload(), measured([1., 1.], [0.4, 0.4])))
            .is_none());
        let mut loaded = store(&info);
        loaded.path = Some(path.clone());
        loaded.read();
        assert!(loaded.lookup(&workload()).unwrap().selected.fusion);
        let mut different = store(&serde_json::json!({"build_id":"updated-build"}));
        different.path = Some(path.clone());
        different.read();
        assert!(different.lookup(&workload()).is_none());
        let mut file: AdaptiveFile =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        file.records[0].probe["candidate_seconds"] = serde_json::json!([1.2, 1.2]);
        std::fs::write(&path, serde_json::to_vec(&file).unwrap()).unwrap();
        let mut invalid = store(&info);
        invalid.path = Some(path.clone());
        invalid.read();
        assert!(invalid.lookup(&workload()).is_none());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn failed_cache_write_does_not_discard_the_in_memory_decision() {
        let blocker = unique_path("blocked");
        std::fs::write(&blocker, b"occupied").unwrap();
        let mut cache = store(&serde_json::json!({}));
        cache.path = Some(blocker.join("cache.json"));
        assert!(cache
            .remember(adaptive_record(workload(), measured([1., 1.], [0.4, 0.4])))
            .is_some());
        assert!(cache.lookup(&workload()).unwrap().selected.fusion);
        std::fs::remove_file(blocker).unwrap();
    }
}
