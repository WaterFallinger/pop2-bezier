//! Versioned three-stage export for large packed-CSV persistence.
use super::*;
use bytemuck::Zeroable;
use rand::{seq::SliceRandom, SeedableRng};
use serde_json::{json, Value};

pub const FORMAT: &str = "geometrize_512_three_stage_interrupt_v2_bezier";

pub const STORAGE_LAYOUT: &str = "large_csv_v1";

pub const HEADER: &str =
    "image_name,cx,cy,w,h,shape_type,theta,r,g,b,is_region_marker,stage,region_id,e0,e1,e2,e3\n";
pub struct RenderedImage {
    pub name: String,
    pub csv: String,
    pub summary: Value,
}

pub fn base_seed() -> u64 {
    std::env::var("SHAPE_RENDERER_SEED")
        .map(|s| s.parse().expect("invalid seed"))
        .unwrap_or(20260916)
}
pub fn mode() -> String {
    std::env::var("SHAPE_RENDERER_MODE").unwrap_or("mixed".into())
}
pub fn image_seed(path: &Path, version: usize) -> u64 {
    let mut hash = 14695981039346656037u64 ^ base_seed();
    for b in path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .bytes()
        .chain((version as u64).to_le_bytes())
    {
        hash = (hash ^ b as u64).wrapping_mul(1099511628211);
    }
    hash
}
pub fn make_state(initial_sse: f32, seed: u64, global_count: u32) -> ImageState {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mixed_draw = rng.gen::<f64>();
    let perturbed = match mode().as_str() {
        "normal" => false,
        "perturbed" => true,
        "mixed" => mixed_draw < 0.70,
        _ => panic!("invalid mode"),
    };
    let mut state = ImageState::zeroed();
    state.initial_sse = initial_sse;
    state.current_sse = initial_sse;
    state.phase_two_start_count = global_count;
    state.perturbed = u32::from(perturbed);
    state.rng = rng.gen::<u32>().max(1);
    // Pre-sample the two independent sets; each is used only upon entering its stage.
    if perturbed {
        for (start, count, max_selected) in [(0, 16, 3), (16, 64, 12)] {
            let n = rng.gen_range(0..=max_selected);
            let mut ids: Vec<usize> = (start..start + count).collect();
            ids.shuffle(&mut rng);
            for &id in &ids[..n] {
                state.perturb[id] = 1;
            }
        }
    }
    state
}
pub fn export_image(
    path: &Path,
    version: usize,
    bg: (u8, u8, u8),
    history: &[GpuShape],
    state: &ImageState,
) -> RenderedImage {
    let name = path.file_stem().unwrap().to_string_lossy().to_string();
    let mut csv = format!(
        "{name},0,0,0,0,-1,0,{},{},{},0,global,-1,0,0,0,0\n",
        bg.0, bg.1, bg.2
    );
    let mut shapes = [0usize; 3];
    let mut begins = [0usize; 3];
    let mut ends = [0usize; 3];
    let mut interrupts = [0usize; 3];
    for s in history {
        let stage = ["global", "coarse", "fine"][s.pad1 as usize];
        let id = if s.pad1 == 0 { -1 } else { s.pad2 as i32 };
        csv.push_str(&format!(
            "{name},{},{},{},{},{},{},{},{},{},{},{stage},{id},{},{},{},{}\n",
            s.cx,
            s.cy,
            s.hw * 2.0,
            s.hh * 2.0,
            s.shape_type,
            s.theta,
            s.r,
            s.g,
            s.b,
            u32::from(s.shape_type >= 2),
            s.e0,
            s.e1,
            s.e2,
            s.e3
        ));
        match s.shape_type {
            0 | 1 => shapes[s.pad1 as usize] += 1,
            2 => begins[s.pad1 as usize] += 1,
            3 => ends[s.pad1 as usize] += 1,
            4 => interrupts[s.pad1 as usize] += 1,
            _ => unreachable!(),
        }
    }
    let reason = match state.termination_reason {
        1 => "all_stages_complete",
        2 => "record_limit",
        3 => "search_exhausted",
        _ => "error",
    };
    let summary = json!({"format":FORMAT,"image_name":name,"source":path,"version":version+1,
        "canvas_size":512,"seed":image_seed(path,version),"base_seed":base_seed(),
        "flow":if state.perturbed==1 {"perturbed"}else{"normal"},
        "perturbed_coarse":(0..16).filter(|&i|state.perturb[i]==1).collect::<Vec<_>>(),
        "perturbed_fine":(16..80).filter(|&i|state.perturb[i]==1).map(|i|i-16).collect::<Vec<_>>(),
        "stage_shapes":shapes,"stage_begins":begins,"stage_true_ends":ends,"stage_interrupt_ends":interrupts,
        "total_records":history.len()+1,"termination_reason":reason,"search_attempts":state.search_attempt_count,
        "region_shape_counts":state.counts.to_vec(),"closed_regions":state.closed.to_vec(),
        "valid_for_training":matches!(state.termination_reason,1|2)});
    RenderedImage { name, csv, summary }
}

pub struct OutputLock {
    _file: fs::File,
}
impl OutputLock {
    pub fn acquire(root: &Path) -> Result<Self, String> {
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(".run.lock"))
            .map_err(|e| e.to_string())?;
        file.try_lock()
            .map_err(|e| format!("output already locked: {e}"))?;
        Ok(Self { _file: file })
    }
}
fn durable_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()
}
pub fn validate_manifest(
    root: &Path,
    constants: &ShaderConstants,
    versions: usize,
    part_size_bytes: u64,
    resume_seed_dir: Option<&Path>,
) -> Result<(), String> {
    if !["normal", "perturbed", "mixed"].contains(&mode().as_str()) {
        return Err("mode must be normal, perturbed or mixed".into());
    }

    let resume_seed_dir = resume_seed_dir
        .map(|path| path.to_string_lossy().to_string());

    let value = json!({
        "format": FORMAT,
        "storage_layout": STORAGE_LAYOUT,
        "csv_header": HEADER.trim_end_matches('\n'),
        "part_size_bytes": part_size_bytes,
        "shader_constants": format!("{constants:?}"),
        "mode": mode(),
        "seed": base_seed(),
        "versions": versions,
        "max_total_records": constants.max_history_records + 1,
        "resume_seed_dir": resume_seed_dir,
    });

    let path = root.join("manifest.json");

    if path.exists() {
        let old: Value =
            serde_json::from_slice(
                &fs::read(&path).map_err(|e| e.to_string())?
            )
            .map_err(|e| e.to_string())?;

        if old != value {
            return Err(
                "output manifest differs; choose a new output directory"
                    .into()
            );
        }
    } else {
        if fs::read_dir(root)
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .any(|p| p.file_name() != ".run.lock")
        {
            return Err(
                "refusing unversioned nonempty output".into()
            );
        }

        let tmp = root.join(".manifest.tmp");

        durable_write(
            &tmp,
            serde_json::to_string_pretty(&value)
                .unwrap()
                .as_bytes(),
        )
        .map_err(|e| e.to_string())?;

        fs::rename(tmp, path)
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn production_defaults_compile_to_expected_shader() {
        let c = default_shader_constants();
        let source = build_shader_source(&c).unwrap();
        assert_eq!(c.max_history_records + 1, 896);
        assert_eq!(c.max_half_size, 256.0);
        assert!(source.contains("const CANVAS_WIDTH: u32 = 512u;"));
        assert!(source.contains("const GRID_MAX_SUCCESSFUL_SHAPES: u32 = 20u;"));
        assert!(source.contains("10u, id >= 16u"));
    }
    #[test]
    fn state_layout_and_sampling() {
        assert_eq!(std::mem::size_of::<ImageState>(), 1024);
        let mut perturbed = 0;
        for seed in 1..10000 {
            let s = make_state(1.0, seed, 85);
            perturbed += s.perturbed;
            assert!(s.perturb[..16].iter().sum::<u32>() <= 3);
            assert!(s.perturb[16..].iter().sum::<u32>() <= 12);
            if s.perturbed == 0 {
                assert_eq!(s.perturb.iter().sum::<u32>(), 0);
            }
        }
        assert!((6500..7500).contains(&perturbed));
    }
    #[test]
    fn output_lock_and_manifest_are_stable() {
        let root = std::env::temp_dir().join(format!(
            "renderer-v3-test-{}",
            std::process::id()
        ));

        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }

        fs::create_dir_all(&root).unwrap();

        let lock = OutputLock::acquire(&root).unwrap();

        assert!(OutputLock::acquire(&root).is_err());

        let constants = default_shader_constants();

        validate_manifest(
            &root,
            &constants,
            100,
            1024 * 1024 * 1024,
            None,
        )
        .unwrap();

        let manifest: Value = serde_json::from_slice(
            &fs::read(root.join("manifest.json")).unwrap(),
        )
        .unwrap();

        assert_eq!(
            manifest["storage_layout"],
            STORAGE_LAYOUT
        );

        assert_eq!(
            manifest["part_size_bytes"].as_u64(),
            Some(1024 * 1024 * 1024)
        );

        drop(lock);

        fs::remove_dir_all(root).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn input_scan_includes_file_symlinks_without_following_directory_cycles() {
        let root = std::env::temp_dir().join(format!("renderer-v3-links-{}", std::process::id()));
        fs::create_dir_all(root.join("input")).unwrap();
        fs::write(root.join("source.jpg"), b"scan-only fixture").unwrap();
        std::os::unix::fs::symlink(root.join("source.jpg"), root.join("input/linked.jpg")).unwrap();
        std::os::unix::fs::symlink(root.join("input"), root.join("input/cycle")).unwrap();
        assert_eq!(
            collect_image_paths(&root.join("input")).unwrap(),
            vec![root.join("input/linked.jpg")]
        );
        fs::remove_dir_all(root).unwrap();
    }
}
