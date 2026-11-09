//! Shared semantic drawing stage and legal center bounds for UI and inference.
//! Only true_end completes a visit; interrupt_end suspends it. An explicit
//! marker can reopen a closed cell for manual editing without erasing history.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub type RegionRecord = [u32; 9];
pub const PAD: u32 = 7054;
pub const REGION_MARKER: u32 = 7057;
pub const TRUE_END: u32 = 7066;
pub const INTERRUPT: u32 = 7067;
pub const BOS: u32 = 7068;
pub const BOS_RECORD: RegionRecord = [BOS, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
pub const INTERRUPT_RECORD: RegionRecord = [INTERRUPT, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
pub const END_RECORD: RegionRecord = [TRUE_END, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
pub const DRAWING_COMPLETE_REASON: &str = "所有区域已绘制完毕，无法继续生成";
pub const GLOBAL_PROGRESS_CAP: usize = 85;

/// Existing user-owned geometry waiting for its drawing stage. These rows are
/// activated as real prefix data, independently of read-only coarse overflow.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct PendingStageRecord {pub side:usize,pub index:usize,pub tokens:RegionRecord}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stage {
    Global,
    Coarse,
    Fine,
    Complete,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Global => "全局绘制",
            Self::Coarse => "粗区域绘制",
            Self::Fine => "细区域绘制",
            Self::Complete => "绘制完成",
        }
    }
    pub fn side(self) -> usize {
        match self {
            Self::Global => 1,
            Self::Coarse => 4,
            Self::Fine => 8,
            Self::Complete => 0,
        }
    }
    pub fn cap(self) -> usize {
        match self {
            Self::Global => 86,
            Self::Coarse => 20,
            Self::Fine => 10,
            Self::Complete => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrawingRegion {
    pub stage: Stage,
    pub side: usize,
    pub index: Option<usize>,
    pub generated_in_region: usize,
    pub completed_coarse: usize,
    pub completed_fine: usize,
    pub awaiting_region: bool,
    pub completed: bool,
    #[serde(default)]
    pub progress: RegionProgress,
}

/// Attained spatial progress, independent of the currently open visit. A
/// manually reopened cell retains its earlier work; completion counters do not
/// encode spatial identity and must never be used to guess colored cells.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RegionProgress {
    pub global: bool,
    /// Regions that have actually appeared in raw marker history. Merely
    /// opening a region is visible even before its first ordinary shape.
    pub coarse: Vec<usize>,
    pub fine: Vec<usize>,
    pub global_shape_count: usize,
    pub global_complete: bool,
    /// Indexed by cell, independent of the currently active region/visit.
    pub coarse_shape_counts: Vec<usize>,
    pub fine_shape_counts: Vec<usize>,
    /// Historical true_end events persist when a cell is manually reopened.
    pub coarse_completed: Vec<usize>,
    pub fine_completed: Vec<usize>,
}

impl RegionProgress {
    pub fn stage_at_fine_cell(&self, index: usize) -> Option<Stage> {
        self.fine_cell_progress(index).map(|(stage, _)| stage)
    }

    /// Highest attained stage and ordinary-shape fraction for a spatial cell.
    /// Opening a coarse/fine marker yields fraction zero, while a true_end
    /// yields one even when the model ended that region before its shape cap.
    pub fn fine_cell_progress(&self, index: usize) -> Option<(Stage, f32)> {
        if index >= 64 {
            return None;
        }
        let coarse_index = (index / 8 / 2) * 4 + (index % 8 / 2);
        if self.fine.contains(&index) {
            Some((
                Stage::Fine,
                completion_fraction(
                    self.fine_shape_counts.get(index).copied().unwrap_or(0),
                    Stage::Fine.cap(),
                    self.fine_completed.contains(&index),
                ),
            ))
        } else if self.coarse.contains(&coarse_index) {
            Some((
                Stage::Coarse,
                completion_fraction(
                    self.coarse_shape_counts
                        .get(coarse_index)
                        .copied()
                        .unwrap_or(0),
                    Stage::Coarse.cap(),
                    self.coarse_completed.contains(&coarse_index),
                ),
            ))
        } else if self.global || self.global_shape_count > 0 || self.global_complete {
            Some((
                Stage::Global,
                completion_fraction(
                    self.global_shape_count,
                    GLOBAL_PROGRESS_CAP,
                    self.global_complete,
                ),
            ))
        } else {
            None
        }
    }
}

fn completion_fraction(count: usize, cap: usize, completed: bool) -> f32 {
    if completed {
        1.0
    } else {
        count.min(cap) as f32 / cap as f32
    }
}

impl Default for DrawingRegion {
    fn default() -> Self {
        Self::global(0)
    }
}

impl DrawingRegion {
    pub fn global(generated: usize) -> Self {
        // The generation controller counts the initial background as one of
        // its 86 records. Progress counts only the 85 ordinary shapes.
        let shape_count = generated.saturating_sub(1);
        Self {
            stage: Stage::Global,
            side: 1,
            index: None,
            generated_in_region: generated,
            completed_coarse: 0,
            completed_fine: 0,
            awaiting_region: false,
            completed: false,
            progress: RegionProgress {
                global: shape_count > 0,
                global_shape_count: shape_count,
                global_complete: shape_count >= GLOBAL_PROGRESS_CAP,
                ..Default::default()
            },
        }
    }
    pub fn active(
        stage: Stage,
        index: usize,
        generated: usize,
        coarse: usize,
        fine: usize,
    ) -> Self {
        let side = stage.side();
        let valid = matches!(stage, Stage::Coarse | Stage::Fine) && index < side * side;
        Self {
            stage,
            side,
            index: valid.then_some(index),
            generated_in_region: generated,
            completed_coarse: coarse,
            completed_fine: fine,
            awaiting_region: !valid,
            completed: stage == Stage::Complete,
            progress: RegionProgress::default(),
        }
    }
    pub fn waiting(stage: Stage, coarse: usize, fine: usize) -> Self {
        Self {
            stage,
            side: stage.side(),
            index: None,
            generated_in_region: 0,
            completed_coarse: coarse,
            completed_fine: fine,
            awaiting_region: stage != Stage::Complete,
            completed: stage == Stage::Complete,
            progress: RegionProgress::default(),
        }
    }
    pub fn complete() -> Self {
        let mut region = Self::waiting(Stage::Complete, 16, 64);
        region.progress = RegionProgress {
            global: true,
            coarse: (0..16).collect(),
            fine: (0..64).collect(),
            global_shape_count: GLOBAL_PROGRESS_CAP,
            global_complete: true,
            coarse_shape_counts: vec![Stage::Coarse.cap(); 16],
            fine_shape_counts: vec![Stage::Fine.cap(); 64],
            coarse_completed: (0..16).collect(),
            fine_completed: (0..64).collect(),
        };
        region
    }
    pub fn stage(&self) -> Stage {
        self.stage
    }
    pub fn generation_block_reason(&self) -> Option<&'static str> {
        (self.completed || self.stage == Stage::Complete).then_some(DRAWING_COMPLETE_REASON)
    }
    pub fn grid_side(&self) -> usize {
        self.side
    }
    pub fn label(&self) -> String {
        match self.index {
            Some(index) => format!(
                "{} · 区域 {}/{}",
                self.stage.label(),
                index + 1,
                self.side * self.side
            ),
            None if self.awaiting_region => format!("{} · 等待选择区域", self.stage.label()),
            None => self.stage.label().into(),
        }
    }
    /// Visual cell rectangle in the editor's 512-unit canvas.
    pub fn bounds(&self) -> Option<[f32; 4]> {
        if self.stage == Stage::Global {
            return Some([0.0, 0.0, 512.0, 512.0]);
        }
        let index = self.index?;
        let cell = 512.0 / self.side as f32;
        let x = (index % self.side) as f32 * cell;
        let y = (index / self.side) as f32 * cell;
        Some([x, y, x + cell, y + cell])
    }
    /// Exact renderer_center_bounds_v1 inclusive center bounds. The legal
    /// last center is end-1, not the half-pixel at end-0.5.
    pub fn center_bounds(&self) -> Option<[f32; 4]> {
        self.bounds()
            .map(|[x0, y0, x1, y1]| [x0, y0, x1 - 1.0, y1 - 1.0])
    }
    pub fn contains_center(&self, x: f32, y: f32) -> bool {
        if self.stage == Stage::Global {
            return x.is_finite() && y.is_finite();
        }
        self.center_bounds()
            .map(|[x0, y0, x1, y1]| x >= x0 && x <= x1 && y >= y0 && y <= y1)
            .unwrap_or(false)
    }
    /// Cell identification is single-pass even in half-pixel gaps between the
    /// renderer's inclusive legal center intervals. It never alters the edit.
    pub fn cell_for_center(stage: Stage, x: f32, y: f32) -> Option<usize> {
        let side = stage.side();
        if !matches!(stage, Stage::Coarse | Stage::Fine) || !x.is_finite() || !y.is_finite() {
            return None;
        }
        let cell = 512.0 / side as f32;
        let column = (x.clamp(0.0, 511.5) / cell).floor() as usize;
        let row = (y.clamp(0.0, 511.5) / cell).floor() as usize;
        Some(row.min(side - 1) * side + column.min(side - 1))
    }

    pub fn from_history(records: &[RegionRecord]) -> Result<Self, String> {
        if records.is_empty() {
            return Ok(Self::global(0));
        }
        if records[0] != BOS_RECORD {
            return Err("区域历史缺少规范图元 BOS".into());
        }
        let mut global = 0;
        let mut global_shapes = 0usize;
        let mut active: Option<(Stage, usize)> = None;
        let mut suspended_stage = None;
        let mut counts: HashMap<(Stage, usize), usize> = HashMap::new();
        let mut completed: HashSet<(Stage, usize)> = HashSet::new();
        let mut attained: HashSet<(Stage, usize)> = HashSet::new();
        let mut attained_counts: HashMap<(Stage, usize), usize> = HashMap::new();
        let mut attained_completed: HashSet<(Stage, usize)> = HashSet::new();
        let mut saw_region = false;
        let mut saw_fine = false;
        for row in records.iter().skip(1) {
            if is_marker(row) {
                let key = marker_identity(row)?;
                if active.is_some() {
                    return Err("区域开始记录嵌套，缺少结束或中断记录".into());
                }
                if let Some(stage) = suspended_stage {
                    if stage != key.0 {
                        return Err("interrupt_end 后必须立即跟同阶段 marker_begin".into());
                    }
                }
                suspended_stage = None;
                if saw_fine && key.0 == Stage::Coarse {
                    return Err("细区域阶段不能退回粗区域".into());
                }
                saw_fine |= key.0 == Stage::Fine;
                saw_region = true;
                if completed.remove(&key) {
                    counts.insert(key, 0);
                }
                counts.entry(key).or_default();
                attained.insert(key);
                active = Some(key);
            } else if row[0] == TRUE_END {
                if *row != END_RECORD {
                    return Err("true_end 载体不规范".into());
                }
                let key = active.take().ok_or("没有活动区域却出现 true_end")?;
                completed.insert(key);
                attained.insert(key);
                attained_completed.insert(key);
            } else if row[0] == INTERRUPT {
                if *row != INTERRUPT_RECORD {
                    return Err("interrupt_end 载体不规范".into());
                }
                let key = active.take().ok_or("没有活动区域却出现 interrupt_end")?;
                suspended_stage = Some(key.0);
            } else if row.iter().all(|&v| v == PAD) {
                if suspended_stage.is_some() {
                    return Err("interrupt_end 后必须立即跟同阶段 marker_begin".into());
                }
            } else {
                if suspended_stage.is_some() {
                    return Err("interrupt_end 后必须立即跟同阶段 marker_begin".into());
                }
                validate_visible(row)?;
                if let Some(key) = active {
                    *counts.entry(key).or_default() += 1;
                    attained.insert(key);
                    if row[5] != 6414 {
                        let count = attained_counts.entry(key).or_default();
                        *count = count.saturating_add(1);
                    }
                } else if saw_region {
                    return Err("区域之外出现普通图元".into());
                } else {
                    global += 1;
                    if row[5] != 6414 {
                        global_shapes = global_shapes.saturating_add(1);
                    }
                }
            }
        }
        let coarse = completed
            .iter()
            .filter(|(stage, _)| *stage == Stage::Coarse)
            .count();
        let fine = completed
            .iter()
            .filter(|(stage, _)| *stage == Stage::Fine)
            .count();
        let mut progress = RegionProgress {
            global: global_shapes > 0,
            global_shape_count: global_shapes,
            global_complete: global_shapes >= GLOBAL_PROGRESS_CAP,
            ..Default::default()
        };
        for &(stage, index) in &attained {
            match stage {
                Stage::Coarse => progress.coarse.push(index),
                Stage::Fine => progress.fine.push(index),
                _ => {}
            }
        }
        progress.coarse.sort_unstable();
        progress.fine.sort_unstable();
        for (&(stage, index), &count) in &attained_counts {
            let stage_counts = match stage {
                Stage::Coarse => &mut progress.coarse_shape_counts,
                Stage::Fine => &mut progress.fine_shape_counts,
                _ => continue,
            };
            stage_counts.resize(stage.side() * stage.side(), 0);
            stage_counts[index] = count;
        }
        for &(stage, index) in &attained_completed {
            match stage {
                Stage::Coarse => progress.coarse_completed.push(index),
                Stage::Fine => progress.fine_completed.push(index),
                _ => {}
            }
        }
        progress.coarse_completed.sort_unstable();
        progress.fine_completed.sort_unstable();
        let mut result = if let Some((stage, index)) = active {
            Self::active(stage, index, counts[&(stage, index)], coarse, fine)
        } else if let Some(stage) = suspended_stage {
            Self::waiting(stage, coarse, fine)
        } else if !saw_region && global_shapes < GLOBAL_PROGRESS_CAP {
            Self::global(global)
        } else if saw_fine || coarse == 16 {
            if fine == 64 {
                Self::waiting(Stage::Complete, coarse, fine)
            } else {
                Self::waiting(Stage::Fine, coarse, fine)
            }
        } else {
            Self::waiting(Stage::Coarse, coarse, fine)
        };
        result.progress = progress;
        Ok(result)
    }
}

pub fn is_marker(row: &RegionRecord) -> bool {
    row[6..].iter().all(|&v| v == REGION_MARKER)
}

pub fn marker_record(stage: Stage, index: usize) -> Result<RegionRecord, String> {
    let side = stage.side();
    if !matches!(stage, Stage::Coarse | Stage::Fine) || index >= side * side {
        return Err("区域 marker 的阶段或序号非法".into());
    }
    let cell = 512 / side;
    let x = ((index % side) * cell * 2 + cell - 1) as u32;
    let y = ((index / side) * cell * 2 + cell - 1) as u32;
    Ok([
        x,
        1024 + y,
        2048,
        2318 + (cell * 4) as u32,
        4366 + (cell * 4) as u32,
        6417,
        REGION_MARKER,
        REGION_MARKER,
        REGION_MARKER,
    ])
}

pub fn marker_identity(row: &RegionRecord) -> Result<(Stage, usize), String> {
    if !is_marker(row) || row[5] != 6417 {
        return Err("区域 marker 格式不规范".into());
    }
    let width = row[3].checked_sub(2318).ok_or("区域 marker 宽度非法")?;
    let stage = match width {
        512 => Stage::Coarse,
        256 => Stage::Fine,
        _ => return Err("区域 marker 宽度不是粗/细网格".into()),
    };
    let side = stage.side();
    let y = row[1].checked_sub(1024).ok_or("区域 marker y 非法")?;
    let cell_bins = 1024 / side as u32;
    let column = row[0] / cell_bins;
    let line = y / cell_bins;
    if column >= side as u32 || line >= side as u32 {
        return Err("区域 marker 越界".into());
    }
    let index = line as usize * side + column as usize;
    if marker_record(stage, index)? != *row {
        return Err("区域 marker 未对齐规范网格".into());
    }
    Ok((stage, index))
}

pub fn validate_visible(row: &RegionRecord) -> Result<(), String> {
    let offsets = [0, 1024, 2048, 2318, 4366, 6414, 6670, 6798, 6926];
    let bins = [1024, 1024, 270, 2048, 2048, 256, 128, 128, 128];
    for i in 0..9 {
        if row[i] < offsets[i] || row[i] >= offsets[i] + bins[i] {
            return Err(format!("可见图元属性 {i} 越界"));
        }
    }
    if row[5] > 6416 {
        return Err("非原生可见图元类型".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn primitive(x: u32, y: u32) -> RegionRecord {
        [x, 1024 + y, 2048, 2358, 4406, 6415, 6670, 6798, 6926]
    }
    #[test]
    fn legal_centers_are_inclusive_and_ignore_shape_extent() {
        let region = DrawingRegion::active(Stage::Coarse, 0, 0, 0, 0);
        assert_eq!(region.bounds(), Some([0., 0., 128., 128.]));
        assert_eq!(region.center_bounds(), Some([0., 0., 127., 127.]));
        assert!(region.contains_center(127., 127.));
        assert!(!region.contains_center(127.5, 10.));
        assert_eq!(
            DrawingRegion::cell_for_center(Stage::Coarse, 127.5, 10.),
            Some(0)
        );
        assert_eq!(
            DrawingRegion::cell_for_center(Stage::Coarse, 128., 10.),
            Some(1)
        );
    }
    #[test]
    fn all_markers_roundtrip_and_unselected_region_has_no_bounds() {
        for stage in [Stage::Coarse, Stage::Fine] {
            for index in 0..stage.side().pow(2) {
                assert_eq!(
                    marker_identity(&marker_record(stage, index).unwrap()).unwrap(),
                    (stage, index)
                );
            }
        }
        assert!(DrawingRegion::waiting(Stage::Fine, 16, 2)
            .bounds()
            .is_none());
    }
    #[test]
    fn interrupts_suspend_without_completion_and_reopen_resets_only_closed_visit() {
        let mut history = vec![
            BOS_RECORD,
            marker_record(Stage::Coarse, 0).unwrap(),
            primitive(10, 10),
            INTERRUPT_RECORD,
            marker_record(Stage::Coarse, 1).unwrap(),
            primitive(300, 10),
        ];
        let region = DrawingRegion::from_history(&history).unwrap();
        assert_eq!(region.index, Some(1));
        assert_eq!(region.completed_coarse, 0);
        history.extend([
            INTERRUPT_RECORD,
            marker_record(Stage::Coarse, 0).unwrap(),
            primitive(12, 12),
        ]);
        assert_eq!(
            DrawingRegion::from_history(&history)
                .unwrap()
                .generated_in_region,
            2
        );
        history.push(END_RECORD);
        assert_eq!(
            DrawingRegion::from_history(&history)
                .unwrap()
                .completed_coarse,
            1
        );
        history.push(marker_record(Stage::Coarse, 0).unwrap());
        let reopened = DrawingRegion::from_history(&history).unwrap();
        assert_eq!(reopened.completed_coarse, 0);
        assert_eq!(reopened.generated_in_region, 0);
    }
    #[test]
    fn interrupt_requires_immediate_same_stage_marker_and_fine_never_reverses() {
        let start = vec![
            BOS_RECORD,
            marker_record(Stage::Coarse, 0).unwrap(),
            INTERRUPT_RECORD,
        ];
        assert!(
            DrawingRegion::from_history(&[start.clone(), vec![primitive(1, 1)]].concat()).is_err()
        );
        assert!(DrawingRegion::from_history(
            &[start, vec![marker_record(Stage::Fine, 0).unwrap()]].concat()
        )
        .is_err());
        let bad = vec![
            BOS_RECORD,
            marker_record(Stage::Fine, 0).unwrap(),
            END_RECORD,
            marker_record(Stage::Coarse, 0).unwrap(),
        ];
        assert!(DrawingRegion::from_history(&bad).is_err());
    }

    #[test]
    fn progress_tracks_sparse_cells_and_inherits_the_furthest_stage() {
        let history = vec![
            BOS_RECORD,
            primitive(400, 400),
            marker_record(Stage::Coarse, 5).unwrap(),
            primitive(300, 300),
            END_RECORD,
            marker_record(Stage::Coarse, 12).unwrap(),
            END_RECORD,
            marker_record(Stage::Fine, 18).unwrap(),
            primitive(300, 300),
            END_RECORD,
            marker_record(Stage::Fine, 63).unwrap(),
        ];
        let region = DrawingRegion::from_history(&history).unwrap();
        assert_eq!(region.progress.coarse, vec![5, 12]);
        assert_eq!(region.progress.fine, vec![18, 63]);
        assert_eq!(region.progress.stage_at_fine_cell(18), Some(Stage::Fine));
        for cell in [19, 26, 27, 48, 49, 56, 57] {
            assert_eq!(
                region.progress.stage_at_fine_cell(cell),
                Some(Stage::Coarse)
            );
        }
        assert_eq!(
            region.progress.fine_cell_progress(63),
            Some((Stage::Fine, 0.0))
        );
        assert_eq!(region.progress.stage_at_fine_cell(0), Some(Stage::Global));
        assert_eq!(region.progress.stage_at_fine_cell(64), None);
        assert_eq!(region.index, Some(63));
    }

    #[test]
    fn markers_show_zero_shape_entry_without_fabricating_completion() {
        let history = vec![
            BOS_RECORD,
            marker_record(Stage::Fine, 30).unwrap(),
            INTERRUPT_RECORD,
            marker_record(Stage::Fine, 60).unwrap(),
        ];
        let region = DrawingRegion::from_history(&history).unwrap();
        assert_eq!(region.progress.fine, vec![30, 60]);
        assert!(region.progress.fine_completed.is_empty());
        assert_eq!(
            region.progress.fine_cell_progress(30),
            Some((Stage::Fine, 0.0))
        );
        assert_eq!(
            region.progress.fine_cell_progress(60),
            Some((Stage::Fine, 0.0))
        );
        assert!((0..64)
            .filter(|index| ![30, 60].contains(index))
            .all(|index| region.progress.stage_at_fine_cell(index).is_none()));
        let mut finished_empty = history;
        finished_empty.push(END_RECORD);
        let finished = DrawingRegion::from_history(&finished_empty).unwrap();
        assert_eq!(finished.progress.fine, vec![30, 60]);
        assert_eq!(finished.progress.fine_completed, vec![60]);
        assert_eq!(
            finished.progress.fine_cell_progress(60),
            Some((Stage::Fine, 1.0))
        );
    }

    #[test]
    fn reopening_and_interrupting_preserves_attained_work() {
        let history = vec![
            BOS_RECORD,
            marker_record(Stage::Coarse, 5).unwrap(),
            primitive(300, 300),
            END_RECORD,
            marker_record(Stage::Coarse, 5).unwrap(),
            INTERRUPT_RECORD,
            marker_record(Stage::Coarse, 9).unwrap(),
        ];
        let region = DrawingRegion::from_history(&history).unwrap();
        assert_eq!(region.completed_coarse, 0);
        assert_eq!(region.progress.coarse, vec![5, 9]);
        assert_eq!(region.index, Some(9));
        assert_eq!(region.progress.stage_at_fine_cell(18), Some(Stage::Coarse));
        assert_eq!(
            region.progress.fine_cell_progress(18),
            Some((Stage::Coarse, 1.0))
        );
        assert_eq!(
            region.progress.fine_cell_progress(34),
            Some((Stage::Coarse, 0.0))
        );
    }

    #[test]
    fn global_progress_excludes_background_and_reaches_half_then_full_at_85_shapes() {
        let mut background = primitive(511, 511);
        background[5] = 6414;
        let mut history = vec![BOS_RECORD, background];
        let empty = DrawingRegion::from_history(&history).unwrap();
        assert_eq!(empty.generated_in_region, 1);
        assert_eq!(empty.progress.global_shape_count, 0);
        assert!(!empty.progress.global_complete);
        assert_eq!(empty.progress.fine_cell_progress(0), None);

        history.extend(std::iter::repeat(primitive(20, 20)).take(43));
        let half = DrawingRegion::from_history(&history).unwrap();
        assert_eq!(half.generated_in_region, 44);
        assert_eq!(half.progress.global_shape_count, 43);
        assert_eq!(
            half.progress.fine_cell_progress(0),
            Some((Stage::Global, 43.0 / 85.0))
        );
        assert!(!half.progress.global_complete);

        history.extend(std::iter::repeat(primitive(20, 20)).take(42));
        let full = DrawingRegion::from_history(&history).unwrap();
        assert_eq!(full.progress.global_shape_count, 85);
        assert!(full.progress.global_complete);
        assert_eq!(
            full.progress.fine_cell_progress(63),
            Some((Stage::Global, 1.0))
        );
        assert_eq!(DrawingRegion::global(44).progress.global_shape_count, 43);
    }

    #[test]
    fn shape_counts_are_independent_per_cell_and_highest_stage_takes_precedence() {
        let mut history = vec![BOS_RECORD, marker_record(Stage::Coarse, 0).unwrap()];
        history.extend(std::iter::repeat(primitive(500, 500)).take(10));
        history.extend([INTERRUPT_RECORD, marker_record(Stage::Coarse, 1).unwrap()]);
        history.extend(std::iter::repeat(primitive(0, 0)).take(20));
        history.extend([END_RECORD, marker_record(Stage::Fine, 9).unwrap()]);
        history.extend(std::iter::repeat(primitive(0, 0)).take(2));
        history.extend([INTERRUPT_RECORD, marker_record(Stage::Fine, 63).unwrap()]);
        let region = DrawingRegion::from_history(&history).unwrap();
        // The deliberately unrelated shape centers must not change the marker
        // identities used to count or color these four spatial regions.
        assert_eq!(region.progress.coarse_shape_counts[0], 10);
        assert_eq!(region.progress.coarse_shape_counts[1], 20);
        assert_eq!(region.progress.fine_shape_counts[9], 2);
        assert_eq!(
            region.progress.fine_cell_progress(0),
            Some((Stage::Coarse, 0.5))
        );
        assert_eq!(
            region.progress.fine_cell_progress(2),
            Some((Stage::Coarse, 1.0))
        );
        assert_eq!(
            region.progress.fine_cell_progress(9),
            Some((Stage::Fine, 0.2))
        );
        assert_eq!(
            region.progress.fine_cell_progress(63),
            Some((Stage::Fine, 0.0))
        );
        assert_eq!(region.progress.fine_cell_progress(62), None);
        assert!(region.progress.fine_completed.is_empty());
    }

    #[test]
    fn early_true_end_is_full_but_interrupt_and_reopened_visits_never_erase_attained_progress() {
        for stage in [Stage::Coarse, Stage::Fine] {
            let cell = if stage == Stage::Coarse { 18 } else { 5 };
            let mut history = vec![
                BOS_RECORD,
                marker_record(stage, 5).unwrap(),
                primitive(10, 10),
            ];
            let partial = DrawingRegion::from_history(&history).unwrap();
            assert_eq!(
                partial.progress.fine_cell_progress(cell),
                Some((stage, 1.0 / stage.cap() as f32))
            );
            history.push(INTERRUPT_RECORD);
            let interrupted = DrawingRegion::from_history(&history).unwrap();
            assert_eq!(
                interrupted.progress.fine_cell_progress(cell),
                partial.progress.fine_cell_progress(cell)
            );
            assert!(interrupted.progress.coarse_completed.is_empty());
            assert!(interrupted.progress.fine_completed.is_empty());
            history.extend([marker_record(stage, 5).unwrap(), END_RECORD]);
            let completed = DrawingRegion::from_history(&history).unwrap();
            assert_eq!(
                completed.progress.fine_cell_progress(cell),
                Some((stage, 1.0))
            );
            history.push(marker_record(stage, 5).unwrap());
            let reopened = DrawingRegion::from_history(&history).unwrap();
            assert_eq!(reopened.generated_in_region, 0);
            assert_eq!(
                reopened.progress.fine_cell_progress(cell),
                Some((stage, 1.0))
            );
            history.extend([
                primitive(20, 20),
                INTERRUPT_RECORD,
                marker_record(stage, 9).unwrap(),
            ]);
            let later = DrawingRegion::from_history(&history).unwrap();
            assert_eq!(later.progress.fine_cell_progress(cell), Some((stage, 1.0)));
            let counts = if stage == Stage::Coarse {
                &later.progress.coarse_shape_counts
            } else {
                &later.progress.fine_shape_counts
            };
            assert_eq!(counts[5], 2);
        }
    }

    #[test]
    fn legacy_progress_deserializes_without_guessing_counts_or_completion() {
        let legacy: RegionProgress =
            serde_json::from_str(r#"{"global":true,"coarse":[5],"fine":[18]}"#).unwrap();
        assert_eq!(legacy.coarse, vec![5]);
        assert_eq!(legacy.fine, vec![18]);
        assert_eq!(legacy.global_shape_count, 0);
        assert!(!legacy.global_complete);
        assert!(legacy.coarse_shape_counts.is_empty() && legacy.fine_shape_counts.is_empty());
        assert!(legacy.coarse_completed.is_empty() && legacy.fine_completed.is_empty());
        assert_eq!(legacy.fine_cell_progress(18), Some((Stage::Fine, 0.0)));
        assert_eq!(legacy.fine_cell_progress(19), Some((Stage::Coarse, 0.0)));
        assert_eq!(legacy.fine_cell_progress(0), Some((Stage::Global, 0.0)));
    }

    #[test]
    fn progress_serializes_and_old_region_snapshots_remain_readable() {
        let history = vec![
            BOS_RECORD,
            primitive(30, 30),
            marker_record(Stage::Fine, 47).unwrap(),
            END_RECORD,
        ];
        let region = DrawingRegion::from_history(&history).unwrap();
        let mut serialized = serde_json::to_value(&region).unwrap();
        assert_eq!(
            serde_json::from_value::<DrawingRegion>(serialized.clone()).unwrap(),
            region
        );
        serialized.as_object_mut().unwrap().remove("progress");
        let old: DrawingRegion = serde_json::from_value(serialized).unwrap();
        assert_eq!(old.stage, region.stage);
        assert_eq!(old.progress, RegionProgress::default());
    }
}
