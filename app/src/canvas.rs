use crate::document::{Primitive, ShapeKind};
use crate::region_state::{DrawingRegion, Stage};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use std::collections::HashSet;
#[path="canvas_raster.rs"]
mod raster;
pub(crate) fn render_reference_preview(primitives:&[Primitive])->egui::ColorImage{raster::render_reference_preview(primitives)}

const BLUE: Color32 = Color32::from_rgb(58, 110, 242);
const PRIMITIVE_CLIPBOARD_PREFIX: &str = "PrimitivePainter.Primitives.v1\n";
const MAX_CLIPBOARD_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct RegionIndicatorGeometry {
    pub panel: Rect,
    pub board: Rect,
    pub legal_centers: Option<Rect>,
}

/// A separate, read-only view of the controller's actual drawing region.
/// This function deliberately owns no CanvasState and emits no CanvasEdit.
pub fn show_region_indicator(
    ui: &mut egui::Ui,
    primitives: &[Primitive],
    region: &DrawingRegion,
) -> RegionIndicatorGeometry {
    let (response, painter) =
        ui.allocate_painter(ui.available_size().max(Vec2::splat(80.0)), Sense::hover());
    let panel = response.rect;
    painter.rect_filled(panel, 0.0, Color32::from_rgb(247, 249, 252));
    painter.vline(
        panel.left(),
        panel.y_range(),
        Stroke::new(1.0, Color32::from_rgb(222, 232, 225)),
    );
    painter.text(
        panel.left_top() + Vec2::new(15.0, 22.0),
        egui::Align2::LEFT_CENTER,
        "绘制进度",
        egui::FontId::proportional(13.0),
        Color32::from_rgb(34, 65, 49),
    );
    painter.text(
        panel.right_top() + Vec2::new(-15.0, 22.0),
        egui::Align2::RIGHT_CENTER,
        "只读",
        egui::FontId::proportional(10.0),
        Color32::from_rgb(111, 133, 120),
    );

    let size = (panel.width() - 28.0).min(panel.height() - 172.0).max(24.0);
    let center_y = (panel.center().y - 10.0).max(panel.top() + 54.0 + size * 0.5);
    let board = Rect::from_center_size(Pos2::new(panel.center().x, center_y), Vec2::splat(size));
    let scale = size / 512.0;
    let art = painter.with_clip_rect(board.intersect(panel));
    art.rect_filled(board, 0.0, Color32::WHITE);
    for primitive in primitives {
        paint_primitive(&art, primitive, board, scale);
    }
    // Always show the same spatial cells: a coarse cell covers 2x2 fine cells.
    // Fill denotes attained work; the worksite icon alone denotes the cursor.
    for index in 0..64 {
        if let Some(color) = region.progress.fine_cell_progress(index)
            .and_then(|(stage, fraction)| progress_overlay(stage, fraction)) {
            let cell_size = size / 8.0;
            let cell = Rect::from_min_size(
                board.min + Vec2::new((index % 8) as f32, (index / 8) as f32) * cell_size,
                Vec2::splat(cell_size),
            );
            art.rect_filled(cell, 0.0, color);
        }
    }
    let side = region.grid_side();
    {
        let side = 8;
        let grid = Color32::from_rgba_premultiplied(22, 33, 28, 66);
        for line in 1..side {
            let offset = line as f32 / side as f32 * size;
            art.vline(
                board.left() + offset,
                board.y_range(),
                Stroke::new(if line % 2 == 0 { 1.1 } else { 0.5 }, grid),
            );
            art.hline(
                board.x_range(),
                board.top() + offset,
                Stroke::new(if line % 2 == 0 { 1.1 } else { 0.5 }, grid),
            );
        }
    }
    let to_screen = |bounds: [f32; 4]| {
        Rect::from_min_max(
            board.min + Vec2::new(bounds[0], bounds[1]) * scale,
            board.min + Vec2::new(bounds[2], bounds[3]) * scale,
        )
    };
    if let Some(bounds) = region.bounds() {
        let active = to_screen(bounds);
        paint_worksite_icon(&art, active.center(), (active.width() * 0.68).clamp(14.0, 36.0));
    }
    let legal_centers = region.center_bounds().map(to_screen);
    painter.rect_stroke(
        board,
        0.0,
        Stroke::new(1.0, Color32::from_rgb(196, 211, 202)),
    );
    let text_top = board.bottom() + 21.0;
    let (label, color) = if region.awaiting_region {
        (
            "等待选择绘制区域".to_owned(),
            Color32::from_rgb(175, 127, 46),
        )
    } else if region.completed || region.stage() == Stage::Complete {
        (
            "没有活动绘制区域".to_owned(),
            Color32::from_rgb(111, 133, 120),
        )
    } else if side > 1 {
        match region.index {
            Some(index) => (
                format!("第 {} 行 · 第 {} 列", index / side + 1, index % side + 1),
                Color32::from_rgb(62, 76, 101),
            ),
            None => (
                "等待选择绘制区域".to_owned(),
                Color32::from_rgb(175, 127, 46),
            ),
        }
    } else {
        ("全图均可绘制".to_owned(), Color32::from_rgb(62, 76, 101))
    };
    painter.text(
        Pos2::new(panel.center().x, text_top),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(12.0),
        color,
    );
    if legal_centers.is_some() {
        painter.text(
            Pos2::new(panel.center().x, text_top + 21.0),
            egui::Align2::CENTER_CENTER,
            "施工标记：当前绘制区域",
            egui::FontId::proportional(10.0),
            Color32::from_rgb(111, 133, 120),
        );
    }
    let legend = [(None, "未绘制"), (Some(Stage::Global), "全局"), (Some(Stage::Coarse), "粗网格"), (Some(Stage::Fine), "细网格")];
    let step = ((panel.width() - 20.0) / 4.0).max(20.0);
    let legend_width = step * 4.0;
    for (index, (stage, text)) in legend.iter().enumerate() {
        let left = panel.center().x - legend_width * 0.5 + index as f32 * step;
        let swatch = Rect::from_min_size(Pos2::new(left, text_top + 40.0), Vec2::splat(8.0));
        painter.rect_filled(swatch, 1.0, stage.and_then(progress_color).unwrap_or(Color32::WHITE));
        painter.rect_stroke(swatch, 1.0, Stroke::new(0.7, Color32::from_rgb(169, 178, 191)));
        painter.text(Pos2::new(left + 12.0, swatch.center().y), egui::Align2::LEFT_CENTER,
            *text, egui::FontId::proportional(10.0), Color32::from_rgb(105, 117, 137));
    }
    RegionIndicatorGeometry {
        panel,
        board,
        legal_centers,
    }
}

fn progress_color(stage: Stage) -> Option<Color32> {
    match stage {
        Stage::Global => Some(Color32::from_rgb(48, 171, 104)),
        Stage::Coarse => Some(Color32::from_rgb(231, 182, 42)),
        Stage::Fine => Some(Color32::from_rgb(220, 77, 83)),
        Stage::Complete => None,
    }
}

fn progress_overlay(stage: Stage, fraction: f32) -> Option<Color32> {
    let color = progress_color(stage)?;
    let fraction = if fraction.is_finite() { fraction.clamp(0.0, 1.0) } else { 0.0 };
    let alpha = match stage {
        Stage::Global => (255.0 * fraction).round() as u8,
        Stage::Coarse | Stage::Fine => (40.0 + 215.0 * fraction).round() as u8,
        Stage::Complete => return None,
    };
    (alpha > 0).then(|| Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha))
}

/// Vector construction barricade: remains legible without emoji/font support.
fn paint_worksite_icon(painter: &egui::Painter, center: Pos2, size: f32) {
    let bar = Rect::from_center_size(center - Vec2::new(0.0, size * 0.08), Vec2::new(size, size * 0.36));
    let halo = bar.expand(size * 0.08);
    painter.rect_filled(halo, 2.0, Color32::from_white_alpha(235));
    let ink = Color32::from_rgb(51, 59, 73);
    for dx in [-0.33, 0.33] {
        let top = center + Vec2::new(size * dx, size * 0.08);
        let bottom = top + Vec2::new(size * dx * 0.25, size * 0.34);
        painter.line_segment([top, bottom], Stroke::new((size * 0.09).max(1.3), ink));
    }
    painter.rect_filled(bar, 1.0, Color32::from_rgb(255, 199, 50));
    let stripes = painter.with_clip_rect(bar.intersect(painter.clip_rect()));
    for stripe in -2..=3 {
        let x = center.x + stripe as f32 * size * 0.26;
        stripes.line_segment([Pos2::new(x - size * 0.1, bar.bottom()), Pos2::new(x + size * 0.1, bar.top())],
            Stroke::new((size * 0.12).max(1.4), ink));
    }
    painter.rect_stroke(bar, 1.0, Stroke::new(0.8, ink));
}

pub fn drawing_stage_label(region: &DrawingRegion) -> String {
    match region.stage() {
        Stage::Global => "当前阶段：全图".into(),
        Stage::Coarse => "当前阶段：粗网格 · 4 × 4".into(),
        Stage::Fine => "当前阶段：细网格 · 8 × 8".into(),
        Stage::Complete => crate::region_state::DRAWING_COMPLETE_REASON.into(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    Select,
    Rectangle,
    Ellipse,
}

pub struct CanvasEdit {
    pub label: String,
    pub primitives: Vec<Primitive>,
}

#[derive(Clone)]
enum GestureKind {
    Move,
    Resize(usize),
    Rotate,
    Marquee,
    Draw(ShapeKind),
}

struct Gesture {
    kind: GestureKind,
    start: Pos2,
    original: Vec<Primitive>,
    bounds: Rect,
    moved: bool,
}

pub struct CanvasState {
    pub tool: Tool,
    pub selection: Vec<u64>,
    pub new_color: [u8; 3],
    pub zoom: f32,
    /// Display preference only. The editable vector model stays unchanged.
    pub rasterize_canvas: bool,
    raster: raster::RasterCanvas,
    hidden_ids: HashSet<u64>,
    pan: Vec2,
    node: Option<u64>,
    gesture: Option<Gesture>,
    preview: Option<Vec<Primitive>>,
    marquee: Option<Rect>,
    clipboard: Vec<Primitive>,
}

impl Default for CanvasState {
    fn default() -> Self {
        Self {
            tool: Tool::Select,
            selection: Vec::new(),
            new_color: [89, 122, 198],
            zoom: 0.5,
            rasterize_canvas: false,
            raster: raster::RasterCanvas::default(),
            hidden_ids: HashSet::new(),
            pan: Vec2::ZERO,
            node: None,
            gesture: None,
            preview: None,
            marquee: None,
            clipboard: Vec::new(),
        }
    }
}

impl CanvasState {
    /// A display/interaction filter, never a destructive canvas conversion.
    pub fn set_hidden_ids(&mut self,ids:&HashSet<u64>){
        if &self.hidden_ids!=ids{
            self.hidden_ids.clone_from(ids);
            self.selection.retain(|id|!self.hidden_ids.contains(id));
            self.gesture=None;self.preview=None;self.marquee=None;
        }
    }
    pub fn is_hidden(&self,id:u64)->bool{self.hidden_ids.contains(&id)}

    pub fn invalidate_history_after_deletion(&mut self) {
        // Compaction can reuse the same numeric node ID for another state.
        // Keep the user's view, tool, colors and clipboard, but discard edits
        // and object selections that belonged to the old graph.
        self.node = None;
        self.gesture = None;
        self.preview = None;
        self.marquee = None;
        self.selection.clear();
    }

    pub fn fit(&mut self) {
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
    }
    pub fn clear_selection(&mut self) {
        self.selection.clear();
    }

    pub fn sync_node(&mut self, node: u64, primitives: &[Primitive]) {
        if self.node != Some(node) {
            self.node = Some(node);
            self.gesture = None;
            self.preview = None;
            self.marquee = None;
            self.selection
                .retain(|id| primitives.iter().any(|p| p.id == *id));
        }
    }

    pub fn selected<'a>(&self, primitives: &'a [Primitive]) -> Vec<&'a Primitive> {
        primitives
            .iter()
            .filter(|p| self.selection.contains(&p.id)&&!self.hidden_ids.contains(&p.id))
            .collect()
    }

    pub fn delete(&mut self, primitives: &[Primitive]) -> Option<CanvasEdit> {
        if self.selected(primitives).is_empty() {
            return None;
        }
        let n = self.selection.len();
        let next = primitives
            .iter()
            .filter(|p| !self.selection.contains(&p.id)||self.hidden_ids.contains(&p.id))
            .cloned()
            .collect();
        self.selection.clear();
        Some(CanvasEdit {
            label: format!("删除 {n} 个图元"),
            primitives: next,
        })
    }

    pub fn copy(&mut self, primitives: &[Primitive]) {
        self.clipboard = self
            .selected(primitives)
            .into_iter()
            .filter(|p| p.shape != ShapeKind::Background)
            .cloned()
            .collect();
    }

    /// Text suitable for the Windows clipboard, shared by application windows.
    /// Returning None leaves the caller's system clipboard untouched.
    pub fn copy_payload(&mut self, primitives: &[Primitive]) -> Option<String> {
        self.copy(primitives);
        if self.clipboard.is_empty() || !self.clipboard.iter().all(valid_clipboard_primitive) {
            return None;
        }
        let json = serde_json::to_string(&self.clipboard).ok()?;
        let payload = format!("{PRIMITIVE_CLIPBOARD_PREFIX}{json}");
        (payload.len() <= MAX_CLIPBOARD_BYTES).then_some(payload)
    }

    /// Accept only our versioned native-shape format. Unrelated clipboard text
    /// and malformed payloads must neither paste nor replace a previous copy.
    pub fn paste_payload(&mut self, primitives: &[Primitive], payload: &str) -> Option<CanvasEdit> {
        if payload.len() > MAX_CLIPBOARD_BYTES {
            return None;
        }
        let json = payload.strip_prefix(PRIMITIVE_CLIPBOARD_PREFIX)?;
        let decoded: Vec<Primitive> = serde_json::from_str(json).ok()?;
        if decoded.is_empty() || !decoded.iter().all(valid_clipboard_primitive) {
            return None;
        }
        self.clipboard = decoded;
        self.paste(primitives)
    }

    pub fn paste(&mut self, primitives: &[Primitive]) -> Option<CanvasEdit> {
        if self.clipboard.is_empty() {
            return None;
        }
        let mut next = primitives.to_vec();
        for p in &self.clipboard {
            let mut p = p.clone();
            p.id = 0;
            if p.shape != ShapeKind::Background {
                p.x = (p.x + 16.0).clamp(0.0, 511.5);
                p.y = (p.y + 16.0).clamp(0.0, 511.5);
            }
            next.push(p);
        }
        Some(CanvasEdit {
            label: format!("粘贴 {} 个图元", self.clipboard.len()),
            primitives: next,
        })
    }

    pub fn duplicate(&mut self, primitives: &[Primitive]) -> Option<CanvasEdit> {
        self.copy(primitives);
        self.paste(primitives).map(|mut edit| {
            edit.label = format!("复制 {} 个图元", self.clipboard.len());
            edit
        })
    }

    pub fn reorder(
        &self,
        primitives: &[Primitive],
        forward: bool,
        extreme: bool,
    ) -> Option<CanvasEdit> {
        if self.selection.is_empty() {
            return None;
        }
        let mut next = primitives.to_vec();
        if extreme {
            let (chosen, others): (Vec<_>, Vec<_>) = next
                .into_iter()
                .partition(|p| self.selection.contains(&p.id));
            next = if forward {
                others.into_iter().chain(chosen).collect()
            } else {
                chosen.into_iter().chain(others).collect()
            };
        } else if forward {
            for i in (0..next.len().saturating_sub(1)).rev() {
                if self.selection.contains(&next[i].id) && !self.selection.contains(&next[i + 1].id)
                {
                    next.swap(i, i + 1);
                }
            }
        } else {
            for i in 1..next.len() {
                if self.selection.contains(&next[i].id) && !self.selection.contains(&next[i - 1].id)
                {
                    next.swap(i, i - 1);
                }
            }
        }
        let label = match (forward, extreme) {
            (true, true) => "置于顶层",
            (false, true) => "置于底层",
            (true, false) => "上移一层",
            _ => "下移一层",
        };
        Some(CanvasEdit {
            label: label.into(),
            primitives: next,
        })
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        primitives: &[Primitive],
        editable: bool,
        node: u64,
    ) -> Option<CanvasEdit> {
        self.sync_node(node, primitives);
        self.selection.retain(|id|!self.hidden_ids.contains(id));
        if !editable {
            self.gesture = None;
            self.preview = None;
            self.marquee = None;
        }
        let available = ui.available_size().max(Vec2::splat(100.0));
        let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
        let area = response.rect;
        let interaction_area=area.intersect(ui.clip_rect());
        let painter=painter.with_clip_rect(interaction_area);
        painter.rect_filled(area, 0.0, Color32::from_rgb(235, 239, 245));
        let fit_scale = ((area.width() - 88.0).min(area.height() - 88.0) / 512.0).max(0.08);
        let previous_scale = fit_scale * self.zoom;
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                let before = area.center() + self.pan;
                self.zoom = (self.zoom * (scroll * 0.0025).exp()).clamp(0.2, 8.0);
                if let Some(pos) = response.hover_pos() {
                    let factor = fit_scale * self.zoom / previous_scale;
                    self.pan += (pos - before) * (1.0 - factor);
                }
            }
        }
        let scale = fit_scale * self.zoom;
        let board = Rect::from_center_size(area.center() + self.pan, Vec2::splat(512.0 * scale));
        let to_model =
            |p: Pos2| Pos2::new((p.x - board.left()) / scale, (p.y - board.top()) / scale);
        let to_screen = |p: Pos2| Pos2::new(board.left() + p.x * scale, board.top() + p.y * scale);
        let pointer = ui.input(|i| i.pointer.interact_pos());
        let pan_down = ui.input(|i| {
            i.pointer.middle_down() || (i.key_down(egui::Key::Space) && i.pointer.primary_down())
        });
        if response.hovered() && pan_down {
            self.pan += ui.input(|i| i.pointer.delta());
        }
        let shift = ui.input(|i| i.modifiers.shift);
        let hittable:std::borrow::Cow<'_,[Primitive]>=if self.hidden_ids.is_empty(){std::borrow::Cow::Borrowed(primitives)}
            else{std::borrow::Cow::Owned(primitives.iter().filter(|p|!self.hidden_ids.contains(&p.id)).cloned().collect())};
        let mut edit = None;
        if editable && !pan_down {
            if response.clicked_by(egui::PointerButton::Primary) {
                if let Some(pos) = response
                    .interact_pointer_pos()
                    .filter(|p| interaction_area.contains(*p))
                {
                    let control=self.tool==Tool::Select
                        &&selected_transform_control(&self.selected(primitives),pos,scale,to_screen).is_some();
                    if control {
                        // A tap or sub-threshold movement on a handle must not
                        // clear selection or cycle through overlapping shapes.
                    } else if board.contains(pos) && self.tool == Tool::Select {
                        let point=to_model(pos);
                        let hit = if shift {hit_test(&hittable,point)}else{click_through_hit(&hittable,point,&self.selection)};
                        match hit {
                            Some(id) if shift => {
                                if self.selection.contains(&id) {
                                    self.selection.retain(|s| *s != id);
                                } else {
                                    self.selection.push(id);
                                }
                            }
                            Some(id) => self.selection = vec![id],
                            None if !shift => self.selection.clear(),
                            _ => {}
                        }
                    } else if !board.contains(pos)&&!shift {
                        self.selection.clear();
                    }
                } else if !shift {
                    self.selection.clear();
                }
            }
            if response.drag_started_by(egui::PointerButton::Primary) {
                if let Some(origin) = ui
                    .input(|i| i.pointer.press_origin())
                    .filter(|p| interaction_area.contains(*p))
                {
                    let start = to_model(origin);
                    let selected = self.selected(primitives);
                    let mode = if self.tool != Tool::Select {
                        board.contains(origin).then_some(GestureKind::Draw(if self.tool == Tool::Ellipse {
                            ShapeKind::Ellipse
                        } else {
                            ShapeKind::Rectangle
                        }))
                    } else if let Some(control)=selected_transform_control(&selected,origin,scale,to_screen) {
                        Some(control)
                    } else if !board.contains(origin) {
                        None
                    } else if let Some(id) = drag_selected_hit(&hittable, start, &self.selection) {
                        if !self.selection.contains(&id) {
                            if !shift {
                                self.selection.clear();
                            }
                            self.selection.push(id);
                        }
                        if primitives
                            .iter()
                            .find(|p| p.id == id)
                            .map(|p| p.shape != ShapeKind::Background)
                            .unwrap_or(false)
                        {
                            Some(GestureKind::Move)
                        } else {
                            Some(GestureKind::Marquee)
                        }
                    } else {
                        if !shift {
                            self.selection.clear();
                        }
                        Some(GestureKind::Marquee)
                    };
                    if let Some(kind) = mode {
                        let selected = self.selected(primitives);
                        let bounds =
                            selection_bounds(&selected).unwrap_or(Rect::from_min_max(start, start));
                        self.gesture = Some(Gesture {
                            kind,
                            start,
                            original: primitives.to_vec(),
                            bounds,
                            moved: false,
                        });
                    }
                }
            }
            if let (Some(gesture), Some(pointer)) = (self.gesture.as_mut(), pointer) {
                let now = to_model(pointer);
                gesture.moved |= now.distance(gesture.start) > 0.5;
                let mut next = gesture.original.clone();
                match gesture.kind {
                    GestureKind::Move => {
                        let delta = now - gesture.start;
                        for p in &mut next {
                            if self.selection.contains(&p.id) && p.shape != ShapeKind::Background {
                                p.x = (p.x + delta.x).clamp(0.0, 511.5);
                                p.y = (p.y + delta.y).clamp(0.0, 511.5);
                            }
                        }
                        self.preview = Some(next);
                    }
                    GestureKind::Resize(corner) => {
                        let transform_shapes: Vec<&Primitive> = gesture
                            .original
                            .iter()
                            .filter(|p| {
                                self.selection.contains(&p.id) && p.shape != ShapeKind::Background
                            })
                            .collect();
                        if transform_shapes.len() == 1 {
                            let original = transform_shapes[0];
                            let anchor = primitive_corners(original)[(corner + 2) % 4];
                            let delta = now - anchor;
                            let (sin, cos) = original.angle_deg.to_radians().sin_cos();
                            if let Some(p) = next.iter_mut().find(|p| p.id == original.id) {
                                p.x = ((anchor.x + now.x) / 2.0).clamp(0.0, 511.5);
                                p.y = ((anchor.y + now.y) / 2.0).clamp(0.0, 511.5);
                                p.width = (delta.x * cos + delta.y * sin).abs().clamp(0.25, 511.75);
                                p.height =
                                    (-delta.x * sin + delta.y * cos).abs().clamp(0.25, 511.75);
                            }
                        } else {
                            // A uniform group scale preserves rotated native rectangles and ellipses.
                            let corners = corner_points(gesture.bounds);
                            let anchor = corners[(corner + 2) % 4];
                            let original = corners[corner] - anchor;
                            let factor = ((now - anchor).dot(original)
                                / original.length_sq().max(0.001))
                            .max(0.005);
                            let new_bounds = Rect::from_two_pos(anchor, anchor + original * factor);
                            scale_selection(&mut next, &self.selection, gesture.bounds, new_bounds);
                        }
                        self.preview = Some(next);
                    }
                    GestureKind::Rotate => {
                        let center = gesture.bounds.center();
                        let start = gesture.start - center;
                        let end = now - center;
                        let mut degrees =
                            (end.y.atan2(end.x) - start.y.atan2(start.x)).to_degrees();
                        if shift {
                            degrees = (degrees / 15.0).round() * 15.0;
                        }
                        rotate_selection(&mut next, &self.selection, center, degrees);
                        self.preview = Some(next);
                    }
                    GestureKind::Marquee => {
                        self.marquee = Some(Rect::from_two_pos(gesture.start, now));
                    }
                    GestureKind::Draw(shape) => {
                        let bounds = Rect::from_two_pos(gesture.start, now);
                        next.push(Primitive {
                            id: 0,
                            x: bounds.center().x.clamp(0.0, 511.5),
                            y: bounds.center().y.clamp(0.0, 511.5),
                            width: bounds.width().max(0.25),
                            height: bounds.height().max(0.25),
                            angle_deg: 0.0,
                            shape,
                            color: self.new_color,
                            stage_assignment:None,
                        });
                        self.preview = Some(next);
                    }
                }
            }
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                if let Some(gesture) = self.gesture.take() {
                    match gesture.kind {
                        GestureKind::Marquee => {
                            if let Some(rect) = self.marquee.take() {
                                let ids = primitives
                                    .iter()
                                    .filter(|p| {
                                        p.shape != ShapeKind::Background&&!self.hidden_ids.contains(&p.id)
                                            && rect.intersects(primitive_bounds(p))
                                    })
                                    .map(|p| p.id);
                                let mut chosen: HashSet<u64> = if shift {
                                    self.selection.iter().copied().collect()
                                } else {
                                    HashSet::new()
                                };
                                chosen.extend(ids);
                                self.selection = primitives
                                    .iter()
                                    .filter(|p| chosen.contains(&p.id)&&!self.hidden_ids.contains(&p.id))
                                    .map(|p| p.id)
                                    .collect();
                            }
                        }
                        kind if gesture.moved => {
                            if let Some(next) = self.preview.take() {
                                let label = match kind {
                                    GestureKind::Move => {
                                        format!("移动 {} 个图元", self.selection.len())
                                    }
                                    GestureKind::Resize(_) => {
                                        format!("缩放 {} 个图元", self.selection.len())
                                    }
                                    GestureKind::Rotate => {
                                        format!("旋转 {} 个图元", self.selection.len())
                                    }
                                    GestureKind::Draw(ShapeKind::Ellipse) => "添加椭圆".into(),
                                    _ => "添加矩形".into(),
                                };
                                edit = Some(CanvasEdit {
                                    label,
                                    primitives: next,
                                });
                            }
                            if matches!(kind, GestureKind::Draw(_)) {
                                self.tool = Tool::Select;
                                self.selection.clear();
                            }
                        }
                        _ => {
                            self.preview = None;
                        }
                    }
                }
            }
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.gesture = None;
                self.preview = None;
                self.marquee = None;
            }
        }

        painter.rect_filled(
            board.expand(5.0).translate(Vec2::new(0.0, 6.0)),
            2.0,
            Color32::from_black_alpha(12),
        );
        let art = painter.with_clip_rect(board.intersect(area));
        art.rect_filled(board, 0.0, Color32::WHITE);
        let visible = self.preview.as_deref().unwrap_or(primitives);
        let artwork:std::borrow::Cow<'_,[Primitive]>=if self.hidden_ids.is_empty(){std::borrow::Cow::Borrowed(visible)}
            else{std::borrow::Cow::Owned(visible.iter().filter(|p|!self.hidden_ids.contains(&p.id)).cloned().collect())};
        if self.rasterize_canvas {
            self.raster.paint(ui.ctx(), &art, board, &artwork);
        } else {
            for p in artwork.iter() {
                paint_primitive(&art, p, board, scale);
            }
        }
        painter.rect_stroke(
            board,
            0.0,
            Stroke::new(1.0, Color32::from_rgb(210, 218, 229)),
        );
        if editable {
            let selected: Vec<&Primitive> = visible
                .iter()
                .filter(|p| self.selection.contains(&p.id)&&!self.hidden_ids.contains(&p.id))
                .collect();
            for p in &selected {
                if p.shape == ShapeKind::Background {
                    continue;
                }
                let points: Vec<Pos2> = primitive_corners(p).into_iter().map(to_screen).collect();
                painter.add(egui::Shape::closed_line(
                    points,
                    Stroke::new(1.0, Color32::from_rgb(111, 151, 250)),
                ));
            }
            if let Some(bounds) = selection_bounds(&selected) {
                let screen = Rect::from_min_max(to_screen(bounds.min), to_screen(bounds.max));
                let transform_shapes: Vec<&Primitive> = selected
                    .iter()
                    .copied()
                    .filter(|p| p.shape != ShapeKind::Background)
                    .collect();
                if transform_shapes.len() != 1 {
                    painter.rect_stroke(screen, 0.0, Stroke::new(1.2, BLUE));
                }
                if selected.iter().any(|p| p.shape != ShapeKind::Background) {
                    let handles = if transform_shapes.len() == 1 {
                        primitive_corners(transform_shapes[0]).map(to_screen)
                    } else {
                        corner_points(screen)
                    };
                    for point in handles {
                        painter.rect_filled(
                            Rect::from_center_size(point, Vec2::splat(7.0)),
                            1.0,
                            Color32::WHITE,
                        );
                        painter.rect_stroke(
                            Rect::from_center_size(point, Vec2::splat(7.0)),
                            1.0,
                            Stroke::new(1.2, BLUE),
                        );
                    }
                    let (top, handle) = if transform_shapes.len() == 1 {
                        let p = transform_shapes[0];
                        let corners = primitive_corners(p);
                        (
                            to_screen(Pos2::new(
                                (corners[0].x + corners[1].x) / 2.0,
                                (corners[0].y + corners[1].y) / 2.0,
                            )),
                            single_rotation_handle(p, scale, to_screen),
                        )
                    } else {
                        let top = Pos2::new(screen.center().x, screen.top());
                        (top, top - Vec2::new(0.0, 25.0))
                    };
                    painter.line_segment([top, handle], Stroke::new(1.0, BLUE));
                    painter.circle_filled(handle, 4.5, Color32::WHITE);
                    painter.circle_stroke(handle, 4.5, Stroke::new(1.2, BLUE));
                }
            }
            if let Some(rect) = self.marquee {
                let screen = Rect::from_min_max(to_screen(rect.min), to_screen(rect.max));
                painter.rect_filled(
                    screen,
                    0.0,
                    Color32::from_rgba_unmultiplied(58, 110, 242, 25),
                );
                painter.rect_stroke(screen, 0.0, Stroke::new(1.0, BLUE));
            }
        }
        if response.hovered() && !pan_down {
            ui.ctx().set_cursor_icon(if self.tool == Tool::Select {
                egui::CursorIcon::Default
            } else {
                egui::CursorIcon::Crosshair
            });
        }
        edit
    }
}

fn valid_clipboard_primitive(primitive: &Primitive) -> bool {
    matches!(primitive.shape, ShapeKind::Rectangle | ShapeKind::Ellipse)
        && [primitive.x, primitive.y, primitive.angle_deg, primitive.width, primitive.height]
            .iter()
            .all(|value| value.is_finite())
        && (0.0..=511.5).contains(&primitive.x)
        && (0.0..=511.5).contains(&primitive.y)
        && (0.0..90.0).contains(&primitive.angle_deg)
        && (0.0..=511.75).contains(&primitive.width)
        && (0.0..=511.75).contains(&primitive.height)
}

pub fn primitive_corners(p: &Primitive) -> [Pos2; 4] {
    let center = Pos2::new(p.x, p.y);
    let radians = p.angle_deg.to_radians();
    let (sin, cos) = radians.sin_cos();
    let local = [
        Vec2::new(-p.width / 2.0, -p.height / 2.0),
        Vec2::new(p.width / 2.0, -p.height / 2.0),
        Vec2::new(p.width / 2.0, p.height / 2.0),
        Vec2::new(-p.width / 2.0, p.height / 2.0),
    ];
    local.map(|v| center + Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos))
}

pub fn primitive_bounds(p: &Primitive) -> Rect {
    if p.shape == ShapeKind::Background {
        return Rect::from_min_max(Pos2::ZERO, Pos2::new(512.0, 512.0));
    }
    let corners = primitive_corners(p);
    let mut bounds = Rect::NOTHING;
    for point in corners {
        bounds.extend_with(point);
    }
    bounds
}

pub fn selection_bounds(primitives: &[&Primitive]) -> Option<Rect> {
    let mut bounds = Rect::NOTHING;
    let has_shape = primitives.iter().any(|p| p.shape != ShapeKind::Background);
    for p in primitives {
        if !has_shape || p.shape != ShapeKind::Background {
            bounds = bounds.union(primitive_bounds(p));
        }
    }
    (!primitives.is_empty()).then_some(bounds)
}

pub fn hit_test(primitives: &[Primitive], point: Pos2) -> Option<u64> {
    primitives
        .iter()
        .rev()
        .find(|p| {
            if p.shape == ShapeKind::Background {
                return point.x >= 0.0 && point.x <= 512.0 && point.y >= 0.0 && point.y <= 512.0;
            }
            let delta = point - Pos2::new(p.x, p.y);
            let (sin, cos) = p.angle_deg.to_radians().sin_cos();
            let x = delta.x * cos + delta.y * sin;
            let y = -delta.x * sin + delta.y * cos;
            let hw = (p.width / 2.0).max(0.125);
            let hh = (p.height / 2.0).max(0.125);
            match p.shape {
                ShapeKind::Ellipse => (x / hw).powi(2) + (y / hh).powi(2) <= 1.0,
                _ => x.abs() <= hw && y.abs() <= hh,
            }
        })
        .map(|p| p.id)
}

/// Repeated plain clicks descend from the uppermost selected shape beneath
/// the pointer. At the bottom, start again at the top of the same hit stack.
fn click_through_hit(primitives:&[Primitive],point:Pos2,selection:&[u64])->Option<u64>{
    let selected=primitives.iter().rposition(|p|selection.contains(&p.id)
        &&hit_test(std::slice::from_ref(p),point)==Some(p.id));
    selected.and_then(|index|hit_test(&primitives[..index],point))
        .or_else(||hit_test(primitives,point))
}

/// After selecting through an overlap, dragging must move that selected shape
/// rather than silently replacing it with the occluding topmost shape.
fn drag_selected_hit(primitives:&[Primitive],point:Pos2,selection:&[u64])->Option<u64>{
    primitives.iter().rev().find(|p|p.shape!=ShapeKind::Background&&selection.contains(&p.id)
        &&hit_test(std::slice::from_ref(*p),point)==Some(p.id)).map(|p|p.id)
        .or_else(||hit_test(primitives,point))
}

fn paint_primitive(painter: &egui::Painter, p: &Primitive, board: Rect, scale: f32) {
    let color = Color32::from_rgb(p.color[0], p.color[1], p.color[2]);
    if p.shape == ShapeKind::Background {
        painter.rect_filled(board, 0.0, color);
        return;
    }
    let color = native_half_alpha_color(p.color);
    let convert = |q: Pos2| Pos2::new(board.left() + q.x * scale, board.top() + q.y * scale);
    let points = if p.shape == ShapeKind::Ellipse {
        let (sin, cos) = p.angle_deg.to_radians().sin_cos();
        (0..96)
            .map(|i| {
                let theta = i as f32 / 96.0 * std::f32::consts::TAU;
                let x = p.width / 2.0 * theta.cos();
                let y = p.height / 2.0 * theta.sin();
                convert(Pos2::new(p.x + x * cos - y * sin, p.y + x * sin + y * cos))
            })
            .collect()
    } else {
        primitive_corners(p).into_iter().map(convert).collect()
    };
    painter.add(egui::Shape::convex_polygon(points, color, Stroke::NONE));
}

fn corner_points(rect: Rect) -> [Pos2; 4] {
    [
        rect.left_top(),
        rect.right_top(),
        rect.right_bottom(),
        rect.left_bottom(),
    ]
}

/// The native model composites half-opacity primitives in encoded sRGB bytes.
/// egui-wgpu 0.27's preferred Unorm framebuffer blends these premultiplied
/// bytes directly. `from_rgba_unmultiplied` instead linearizes before its
/// premultiplication, which would wash out this native renderer's colors.
/// Model colors are even-valued, so each channel's half is exact.
fn native_half_alpha_color(rgb: [u8; 3]) -> Color32 {
    Color32::from_rgba_premultiplied(rgb[0] / 2, rgb[1] / 2, rgb[2] / 2, 128)
}

fn single_rotation_handle(p: &Primitive, scale: f32, to_screen: impl Fn(Pos2) -> Pos2) -> Pos2 {
    let (sin, cos) = p.angle_deg.to_radians().sin_cos();
    let distance = p.height / 2.0 + 25.0 / scale;
    to_screen(Pos2::new(p.x + sin * distance, p.y - cos * distance))
}

/// Transform controls live in the editing viewport, not in the clipped image.
/// These screen-space positions match the selected-shape overlay exactly.
fn selected_transform_control(selected:&[&Primitive],point:Pos2,scale:f32,to_screen:impl Fn(Pos2)->Pos2)->Option<GestureKind>{
    let shapes=selected.iter().copied().filter(|p|p.shape!=ShapeKind::Background).collect::<Vec<_>>();
    if shapes.is_empty(){return None;}
    let bounds=selection_bounds(selected)?;
    let (corners,rotation)=if shapes.len()==1{
        (primitive_corners(shapes[0]).map(&to_screen),single_rotation_handle(shapes[0],scale,&to_screen))
    }else{
        (corner_points(bounds).map(&to_screen),to_screen(Pos2::new(bounds.center().x,bounds.top()))-Vec2::new(0.,25.))
    };
    if point.distance(rotation)<=10.{return Some(GestureKind::Rotate);}
    corners.iter().position(|corner|point.distance(*corner)<=10.).map(GestureKind::Resize)
}

fn scale_selection(primitives: &mut [Primitive], selection: &[u64], original: Rect, target: Rect) {
    let sx = target.width().max(0.25) / original.width().max(0.25);
    let sy = target.height().max(0.25) / original.height().max(0.25);
    for p in primitives {
        if selection.contains(&p.id) && p.shape != ShapeKind::Background {
            p.x = (target.left() + (p.x - original.left()) * sx).clamp(0.0, 511.5);
            p.y = (target.top() + (p.y - original.top()) * sy).clamp(0.0, 511.5);
            p.width = (p.width * sx).clamp(0.25, 512.0);
            p.height = (p.height * sy).clamp(0.25, 512.0);
        }
    }
}

fn rotate_selection(primitives: &mut [Primitive], selection: &[u64], center: Pos2, degrees: f32) {
    let (sin, cos) = degrees.to_radians().sin_cos();
    for p in primitives {
        if selection.contains(&p.id) && p.shape != ShapeKind::Background {
            let delta = Pos2::new(p.x, p.y) - center;
            p.x = (center.x + delta.x * cos - delta.y * sin).clamp(0.0, 511.5);
            p.y = (center.y + delta.x * sin + delta.y * cos).clamp(0.0, 511.5);
            p.angle_deg += degrees;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deleting_history_invalidates_old_node_edits_but_keeps_view_and_clipboard() {
        let mut canvas=CanvasState::default();
        let all=vec![p(1,ShapeKind::Rectangle)];
        canvas.selection=vec![1];canvas.copy(&all);
        canvas.node=Some(17);canvas.zoom=0.8;canvas.pan=Vec2::new(23.,-11.);
        canvas.new_color=[10,20,30];canvas.tool=Tool::Ellipse;
        canvas.preview=Some(all.clone());canvas.marquee=Some(Rect::EVERYTHING);
        canvas.gesture=Some(Gesture{kind:GestureKind::Move,start:Pos2::ZERO,
            original:all.clone(),bounds:Rect::EVERYTHING,moved:true});
        canvas.invalidate_history_after_deletion();
        assert!(canvas.node.is_none()&&canvas.gesture.is_none()&&canvas.preview.is_none()&&canvas.marquee.is_none());
        assert!(canvas.selection.is_empty());assert_eq!(canvas.clipboard,all);
        assert_eq!(canvas.zoom,0.8);assert_eq!(canvas.pan,Vec2::new(23.,-11.));
        assert_eq!(canvas.new_color,[10,20,30]);assert_eq!(canvas.tool,Tool::Ellipse);
    }

    fn painted_primary_board(ctx: &egui::Context, canvas: &mut CanvasState) -> Rect {
        let output = ctx.run(egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::splat(800.0))),
            ..Default::default()
        }, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| { canvas.show(ui, &[], true, 0); });
        });
        output.shapes.iter().filter_map(|clipped| match &clipped.shape {
            egui::Shape::Rect(shape) if shape.fill == Color32::WHITE => Some(shape.rect),
            _ => None,
        }).max_by(|left, right| left.width().total_cmp(&right.width())).expect("primary canvas must paint its white board")
    }

    #[test]
    fn initial_canvas_is_half_fit_size_and_drag_still_uses_model_coordinates() {
        let ctx = egui::Context::default();
        let mut canvas = CanvasState::default();
        let initial = painted_primary_board(&ctx, &mut canvas);
        assert_eq!(canvas.zoom, 0.5);
        canvas.fit();
        let fitted = painted_primary_board(&ctx, &mut canvas);
        assert!((fitted.width() - initial.width()*2.0).abs() < 0.01);
        assert!((fitted.height() - initial.height()*2.0).abs() < 0.01);
        assert!(fitted.center().distance(initial.center()) < 0.01);
        assert_eq!(canvas.zoom, 1.0);

        let mut canvas = CanvasState::default();
        let all = vec![p(1, ShapeKind::Rectangle)];
        frame(&ctx, &mut canvas, &all, true, vec![]);
        let start = initial.center();
        let end = start + Vec2::new(32.0,16.0)*(initial.width()/512.0);
        frame(&ctx, &mut canvas, &all, true, vec![egui::Event::PointerMoved(start),press(start,true)]);
        frame(&ctx, &mut canvas, &all, true, vec![egui::Event::PointerMoved(end)]);
        let edit = frame(&ctx, &mut canvas, &all, true, vec![press(end,false)]).expect("drag must commit");
        assert!((edit.primitives[0].x-288.0).abs() < 0.02);
        assert!((edit.primitives[0].y-272.0).abs() < 0.02);
        assert_eq!(crate::document::CANVAS_SIZE, 512.0);
    }
    fn p(id: u64, shape: ShapeKind) -> Primitive {
        Primitive {
            id,
            x: 256.0,
            y: 256.0,
            width: 100.0,
            height: 40.0,
            angle_deg: 45.0,
            shape,
            color: [255, 0, 0],
            stage_assignment:None,
        }
    }

    fn indicator_frame(
        region: &DrawingRegion,
        primitives: &[Primitive],
    ) -> RegionIndicatorGeometry {
        let ctx = egui::Context::default();
        let mut geometry = None;
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 600.0))),
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                geometry = Some(show_region_indicator(ui, primitives, region));
            });
        });
        geometry.unwrap()
    }

    #[test]
    fn region_indicator_uses_controller_bounds_not_selected_geometry() {
        let region = DrawingRegion::active(Stage::Coarse, 6, 0, 0, 0);
        let mut shape = p(1, ShapeKind::Ellipse);
        shape.x = 16.0;
        shape.y = 16.0;
        let geometry = indicator_frame(&region, &[shape]);
        let bounds = region.center_bounds().unwrap();
        let expected_min =
            geometry.board.min + Vec2::new(bounds[0], bounds[1]) * (geometry.board.width() / 512.0);
        let expected_max =
            geometry.board.min + Vec2::new(bounds[2], bounds[3]) * (geometry.board.width() / 512.0);
        let legal = geometry.legal_centers.unwrap();
        assert!(legal.min.distance(expected_min) < 0.001);
        assert!(legal.max.distance(expected_max) < 0.001);
        assert!(geometry.board.contains_rect(legal));
    }

    #[test]
    fn pending_and_finished_regions_have_no_guessed_active_area() {
        let waiting = DrawingRegion::waiting(Stage::Fine, 16, 4);
        assert!(indicator_frame(&waiting, &[]).legal_centers.is_none());
        assert!(indicator_frame(&DrawingRegion::complete(), &[])
            .legal_centers
            .is_none());
        assert_eq!(drawing_stage_label(&waiting), "当前阶段：细网格 · 8 × 8");
        assert_eq!(drawing_stage_label(&DrawingRegion::complete()), "所有区域已绘制完毕，无法继续生成");
    }

    #[test]
    fn progress_colors_are_stage_colors_and_active_cell_does_not_set_them() {
        let active = DrawingRegion::active(Stage::Fine, 45, 0, 16, 20);
        assert!(active.progress.stage_at_fine_cell(45).is_none());
        assert_eq!(progress_color(Stage::Global), Some(Color32::from_rgb(48, 171, 104)));
        assert_eq!(progress_color(Stage::Coarse), Some(Color32::from_rgb(231, 182, 42)));
        assert_eq!(progress_color(Stage::Fine), Some(Color32::from_rgb(220, 77, 83)));
        assert!(progress_color(Stage::Complete).is_none());
        assert!(indicator_frame(&active, &[]).legal_centers.is_some());
    }

    #[test]
    fn progress_overlay_deepens_by_shape_count_and_completion_is_opaque() {
        assert!(progress_overlay(Stage::Global, 0.0).is_none());
        assert_eq!(progress_overlay(Stage::Global, 43.0 / 85.0).unwrap().a(), 129);
        assert_eq!(progress_overlay(Stage::Global, 1.0).unwrap().a(), 255);
        for stage in [Stage::Coarse, Stage::Fine] {
            assert_eq!(progress_overlay(stage, 0.0).unwrap().a(), 40);
            assert_eq!(progress_overlay(stage, 0.5).unwrap().a(), 148);
            assert_eq!(progress_overlay(stage, 1.0).unwrap().a(), 255);
            assert_eq!(progress_overlay(stage, 10.0).unwrap().a(), 255);
        }
        assert!(progress_overlay(Stage::Complete, 1.0).is_none());

        let history = vec![crate::region_state::BOS_RECORD,
            crate::region_state::marker_record(Stage::Fine, 9).unwrap(),
            crate::region_state::END_RECORD];
        let early_end = DrawingRegion::from_history(&history).unwrap();
        let (stage, fraction) = early_end.progress.fine_cell_progress(9).unwrap();
        assert_eq!(progress_overlay(stage, fraction).unwrap().a(), 255);
        assert!(early_end.progress.fine_cell_progress(8).is_none());
    }

    #[test]
    fn native_half_alpha_fixture_preserves_srgb_source_over() {
        let foreground = [252, 206, 164];
        let background = [238, 242, 248];
        let premultiplied = native_half_alpha_color(foreground).to_array();
        assert_eq!(premultiplied, [126, 103, 82, 128]);
        // Unorm's alpha=128/255 differs from exact 0.5 by less than one byte.
        // The fully covered pixel must remain near the native renderer's
        // [245,224,206], rather than the incorrect near-white screenshot.
        let composite: [u8; 3] = std::array::from_fn(|channel| {
            (premultiplied[channel] as f32
                + background[channel] as f32 * (1.0 - premultiplied[3] as f32 / 255.0))
                .round() as u8
        });
        let expected = [245u8, 224, 206];
        assert!(composite
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) <= 1));
    }
    #[test]
    fn rotated_ellipse_hit_test_respects_local_axes() {
        let shape = p(1, ShapeKind::Ellipse);
        assert_eq!(hit_test(&[shape.clone()], Pos2::new(280.0, 280.0)), Some(1));
        assert_eq!(hit_test(&[shape], Pos2::new(280.0, 232.0)), None);
    }
    #[test]
    fn topmost_primitive_wins_hit_test() {
        assert_eq!(
            hit_test(
                &[p(1, ShapeKind::Rectangle), p(2, ShapeKind::Ellipse)],
                Pos2::new(256.0, 256.0)
            ),
            Some(2)
        );
    }
    #[test]
    fn plain_click_descends_selected_hit_stack_and_wraps_without_affecting_drag_target(){
        let point=Pos2::new(256.,256.);let mut background=Primitive::background([255;3]);background.id=10;
        let mut outside=p(4,ShapeKind::Rectangle);outside.x=40.;outside.y=40.;
        let all=vec![background,p(1,ShapeKind::Rectangle),p(2,ShapeKind::Ellipse),p(3,ShapeKind::Rectangle),outside];
        assert_eq!(click_through_hit(&all,point,&[]),Some(3));
        assert_eq!(click_through_hit(&all,point,&[3]),Some(2));
        assert_eq!(click_through_hit(&all,point,&[2]),Some(1));
        assert_eq!(click_through_hit(&all,point,&[1]),Some(10));
        assert_eq!(click_through_hit(&all,point,&[10]),Some(3));
        assert_eq!(click_through_hit(&all,point,&[4,3,2]),Some(2));
        assert_eq!(drag_selected_hit(&all,point,&[2]),Some(2));
        assert_eq!(drag_selected_hit(&all,point,&[10]),Some(3));
        assert_eq!(click_through_hit(&all[1..],Pos2::new(511.,0.),&[1]),None);
    }
    #[test]
    fn group_rotation_changes_centers_without_mutating_unselected() {
        let mut a = p(1, ShapeKind::Rectangle);
        a.x = 206.0;
        a.y = 256.0;
        let b = p(2, ShapeKind::Rectangle);
        let original = b.clone();
        let mut all = vec![a, b];
        rotate_selection(&mut all, &[1], Pos2::new(256.0, 256.0), 90.0);
        assert!((all[0].x - 256.0).abs() < 0.001);
        assert!((all[0].y - 206.0).abs() < 0.001);
        assert_eq!(all[1], original);
    }
    #[test]
    fn multi_selection_layer_move_keeps_internal_order() {
        let mut canvas = CanvasState::default();
        canvas.selection = vec![1, 2];
        let all = vec![
            p(1, ShapeKind::Rectangle),
            p(2, ShapeKind::Rectangle),
            p(3, ShapeKind::Rectangle),
        ];
        let edit = canvas.reorder(&all, true, false).unwrap();
        assert_eq!(
            edit.primitives.iter().map(|p| p.id).collect::<Vec<_>>(),
            vec![3, 1, 2]
        );
    }

    #[test]
    fn copying_mixed_selection_does_not_duplicate_unique_background() {
        let mut background = Primitive::background([20, 30, 40]);
        background.id = 1;
        let all = vec![background, p(2, ShapeKind::Rectangle)];
        let mut canvas = CanvasState::default();
        canvas.selection = vec![1, 2];
        let edit = canvas.duplicate(&all).unwrap();
        assert_eq!(
            edit.primitives
                .iter()
                .filter(|p| p.shape == ShapeKind::Background)
                .count(),
            1
        );
        assert_eq!(edit.primitives.len(), 3);
    }

    #[test]
    fn versioned_clipboard_payload_pastes_in_another_canvas_with_new_ids_and_offset() {
        let mut background = Primitive::background([20, 30, 40]);
        background.id = 1;
        let rectangle = p(2, ShapeKind::Rectangle);
        let mut ellipse = p(3, ShapeKind::Ellipse);
        ellipse.x = 510.0;
        ellipse.y = 507.0;
        let source = vec![background.clone(), rectangle.clone(), ellipse.clone()];
        let mut sender = CanvasState::default();
        sender.selection = vec![1, 2, 3];
        let payload = sender.copy_payload(&source).expect("ordinary shapes must be copied");
        assert!(payload.starts_with(PRIMITIVE_CLIPBOARD_PREFIX));
        let decoded: Vec<Primitive> = serde_json::from_str(payload.strip_prefix(PRIMITIVE_CLIPBOARD_PREFIX).unwrap()).unwrap();
        assert_eq!(decoded, vec![rectangle, ellipse]);

        let mut receiver = CanvasState::default();
        assert!(receiver.paste(&[background.clone()]).is_none());
        let edit = receiver.paste_payload(&[background.clone()], &payload).expect("another canvas must accept the application clipboard format");
        assert_eq!(edit.label, "粘贴 2 个图元");
        assert_eq!(edit.primitives.len(), 3);
        assert_eq!(edit.primitives[0], background);
        assert_eq!(edit.primitives[1].shape, ShapeKind::Rectangle);
        assert_eq!(edit.primitives[1].id, 0);
        assert_eq!(edit.primitives[1].x, 272.0);
        assert_eq!(edit.primitives[1].y, 272.0);
        assert_eq!(edit.primitives[2].shape, ShapeKind::Ellipse);
        assert_eq!(edit.primitives[2].id, 0);
        assert_eq!(edit.primitives[2].x, 511.5);
        assert_eq!(edit.primitives[2].y, 511.5);
        assert_eq!(receiver.paste(&[]).unwrap().primitives.len(), 2);
    }

    #[test]
    fn empty_or_background_only_selection_does_not_offer_system_clipboard_text() {
        let mut canvas = CanvasState::default();
        let mut background = Primitive::background([10, 20, 30]);
        background.id = 1;
        let all = vec![background, p(2, ShapeKind::Rectangle)];
        assert!(canvas.copy_payload(&all).is_none());
        canvas.selection = vec![2];
        assert!(canvas.copy_payload(&all).is_some());
        canvas.selection = vec![1];
        assert!(canvas.copy_payload(&all).is_none());
        assert!(canvas.paste(&all).is_none());
        canvas.selection = vec![999];
        assert!(canvas.copy_payload(&all).is_none());
    }

    #[test]
    fn invalid_external_clipboard_preserves_previous_internal_copy() {
        let valid = p(7, ShapeKind::Ellipse);
        let mut canvas = CanvasState::default();
        canvas.selection = vec![7];
        canvas.copy(&[valid.clone()]);
        let encode = |items: &[Primitive]| format!("{PRIMITIVE_CLIPBOARD_PREFIX}{}", serde_json::to_string(items).unwrap());
        let mut bad = vec![
            String::new(),
            "普通文本 / a file path is not a shape".into(),
            encode(&[valid.clone()]).replacen(".v1\n", ".v2\n", 1),
            format!("{PRIMITIVE_CLIPBOARD_PREFIX}[]"),
            format!("{PRIMITIVE_CLIPBOARD_PREFIX}{{}}"),
            format!("{PRIMITIVE_CLIPBOARD_PREFIX}["),
            encode(&[Primitive::background([1, 2, 3])]),
            encode(&[valid.clone(), Primitive::background([1, 2, 3])]),
            encode(&[valid.clone()]).replace("Ellipse", "UnknownShape"),
            encode(&[valid.clone()]).replace("256.0", "NaN"),
            encode(&[valid.clone()]).replace("256.0", "1e999"),
        ];
        for (field, value) in [
            ("x", -0.5), ("x", 512.0), ("y", -1.0), ("y", 512.0),
            ("width", -0.25), ("width", 512.0), ("height", -1.0), ("height", 512.0),
            ("angle_deg", -1.0), ("angle_deg", 90.0),
        ] {
            let mut json = serde_json::to_value(&valid).unwrap();
            json[field] = serde_json::json!(value);
            bad.push(format!("{PRIMITIVE_CLIPBOARD_PREFIX}[{json}]"));
        }
        for payload in bad {
            assert!(canvas.paste_payload(&[], &payload).is_none(), "must reject {payload}");
            assert_eq!(canvas.clipboard, vec![valid.clone()]);
        }
        let previous = canvas.paste(&[]).expect("unrelated clipboard text must not discard the prior internal copy");
        assert_eq!(previous.primitives.len(), 1);
        assert_eq!(previous.primitives[0].shape, ShapeKind::Ellipse);
        assert_eq!(previous.primitives[0].id, 0);
    }

    #[test]
    fn group_transform_bounds_exclude_unique_background() {
        let background = Primitive::background([20, 30, 40]);
        let shape = p(2, ShapeKind::Rectangle);
        let bounds = selection_bounds(&[&background, &shape]).unwrap();
        assert!(bounds.width() < 200.0);
        assert_eq!(bounds, primitive_bounds(&shape));
    }

    fn frame(
        ctx: &egui::Context,
        canvas: &mut CanvasState,
        primitives: &[Primitive],
        editable: bool,
        events: Vec<egui::Event>,
    ) -> Option<CanvasEdit> {
        let mut result = None;
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::splat(800.0))),
            events,
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                result = canvas.show(ui, primitives, editable, 0);
            });
        });
        result
    }

    fn press(pos: Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        }
    }

    #[test]
    fn a_mouse_drag_commits_once_on_release() {
        let ctx = egui::Context::default();
        let mut canvas = CanvasState::default();
        let all = vec![p(1, ShapeKind::Rectangle)];
        frame(&ctx, &mut canvas, &all, true, vec![]);
        let start = Pos2::new(400.0, 400.0);
        assert!(frame(
            &ctx,
            &mut canvas,
            &all,
            true,
            vec![egui::Event::PointerMoved(start), press(start, true)]
        )
        .is_none());
        let end = Pos2::new(430.0, 410.0);
        assert!(frame(
            &ctx,
            &mut canvas,
            &all,
            true,
            vec![egui::Event::PointerMoved(end)]
        )
        .is_none());
        let edit = frame(&ctx, &mut canvas, &all, true, vec![press(end, false)])
            .expect("release must commit completed drag");
        assert_eq!(edit.primitives.len(), 1);
        assert!(edit.primitives[0].x > all[0].x);
        assert!(frame(&ctx, &mut canvas, &edit.primitives, true, vec![]).is_none());
    }

    #[test]
    fn raster_display_preserves_native_drag_and_outside_handle_results(){
        for outside in [false,true]{
            let run=|rasterize_canvas|{
                let ctx=egui::Context::default();let mut canvas=CanvasState::default();
                canvas.rasterize_canvas=rasterize_canvas;
                let board=painted_primary_board(&ctx,&mut canvas);let scale=board.width()/512.;
                let mut shape=p(1,ShapeKind::Rectangle);
                if outside{shape.x=5.;shape.y=5.;shape.width=100.;shape.height=80.;shape.angle_deg=30.;}
                let all=vec![shape];canvas.selection=vec![1];
                let start=if outside{board.min+primitive_corners(&all[0])[0].to_vec2()*scale}else{board.center()};
                let end=if outside{Pos2::new(0.,100.)}else{start+Vec2::new(30.,10.)};
                frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(start),press(start,true)]);
                assert!(frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(end)]).is_none());
                let edit=frame(&ctx,&mut canvas,&all,true,vec![press(end,false)]).expect("display mode must not prevent editing");
                assert_eq!(canvas.selection,vec![1]);
                assert_eq!(all.len(),1,"the borrowed source cannot be flattened or rewritten");
                edit.primitives
            };
            assert_eq!(run(false),run(true),"native coordinates and pointer capture must be identical, outside={outside}");
        }
    }

    #[test]
    fn hidden_stage_shapes_never_intercept_hits_and_survive_visible_edits(){
        for raster in [false,true]{
            let ctx=egui::Context::default();let mut canvas=CanvasState::default();canvas.rasterize_canvas=raster;
            let board=painted_primary_board(&ctx,&mut canvas);let start=board.center();
            let all=vec![p(1,ShapeKind::Rectangle),p(2,ShapeKind::Ellipse)];
            canvas.set_hidden_ids(&HashSet::from([2]));
            frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(start),press(start,true)]);
            frame(&ctx,&mut canvas,&all,true,vec![press(start,false)]);
            assert_eq!(canvas.selection,[1],"the hidden top shape must not intercept the lower visible shape");
            frame(&ctx,&mut canvas,&all,true,vec![press(start,true)]);
            let end=start+Vec2::new(30.,10.);frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(end)]);
            let edit=frame(&ctx,&mut canvas,&all,true,vec![press(end,false)]).unwrap();
            assert_eq!(edit.primitives.len(),2);assert_eq!(edit.primitives[1],all[1]);
            assert!(edit.primitives[0].x>all[0].x);
            let deleted=canvas.delete(&edit.primitives).unwrap();assert_eq!(deleted.primitives,vec![all[1].clone()]);
            canvas.selection=vec![2];assert!(canvas.selected(&all).is_empty());assert!(canvas.delete(&all).is_none());
        }
    }

    #[test]
    fn hiding_all_artwork_clears_selection_and_uploads_a_white_512_surface(){
        let ctx=egui::Context::default();let mut canvas=CanvasState::default();canvas.rasterize_canvas=true;
        let mut bg=Primitive::background([20,40,60]);bg.id=1;let all=vec![bg,p(2,ShapeKind::Rectangle)];
        canvas.selection=vec![2];canvas.set_hidden_ids(&HashSet::from([1,2]));assert!(canvas.selection.is_empty());
        let output=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::splat(800.))),..Default::default()},|ctx|{
            egui::CentralPanel::default().show(ctx,|ui|{assert!(canvas.show(ui,&all,true,0).is_none());});
        });
        let (_,delta)=output.textures_delta.set.iter().find(|(_,d)|d.image.size()==[512,512]).unwrap();
        let egui::ImageData::Color(image)=&delta.image else{panic!("expected color raster")};
        assert!(image.pixels.iter().all(|&c|c==Color32::WHITE));
        canvas.set_hidden_ids(&HashSet::new());assert!(!canvas.is_hidden(2));assert_eq!(all.len(),2);
    }

    #[test]
    fn outside_board_corner_can_be_clicked_and_resized_and_capture_continues_outside_viewport(){
        let ctx=egui::Context::default();let mut canvas=CanvasState::default();let board=painted_primary_board(&ctx,&mut canvas);
        let mut shape=p(1,ShapeKind::Rectangle);shape.x=5.;shape.y=5.;shape.width=100.;shape.height=80.;shape.angle_deg=30.;
        let all=vec![shape];canvas.selection=vec![1];let scale=board.width()/512.;
        let corner=primitive_corners(&all[0])[0];let start=board.min+corner.to_vec2()*scale;
        assert!(!board.contains(start));assert!(Rect::from_min_max(Pos2::new(8.,8.),Pos2::new(792.,792.)).contains(start));
        frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(start),press(start,true)]);
        assert!(frame(&ctx,&mut canvas,&all,true,vec![press(start,false)]).is_none());assert_eq!(canvas.selection,vec![1]);
        frame(&ctx,&mut canvas,&all,true,vec![press(start,true)]);
        let end=Pos2::new(0.,100.);frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(end)]);
        let edit=frame(&ctx,&mut canvas,&all,true,vec![press(end,false)]).expect("outside corner must keep pointer capture");
        assert!(edit.primitives[0].width>all[0].width);assert_eq!(canvas.selection,vec![1]);
    }

    #[test]
    fn outside_board_rotation_handle_rotates_selected_shape_and_preserves_background(){
        let ctx=egui::Context::default();let mut canvas=CanvasState::default();let board=painted_primary_board(&ctx,&mut canvas);
        let mut bg=Primitive::background([40,60,80]);bg.id=10;
        let mut shape=p(1,ShapeKind::Ellipse);shape.x=5.;shape.y=5.;shape.width=100.;shape.height=80.;shape.angle_deg=30.;
        let all=vec![bg,shape];canvas.selection=vec![10,1];let scale=board.width()/512.;
        let convert=|p:Pos2|board.min+p.to_vec2()*scale;
        let start=single_rotation_handle(&all[1],scale,convert);assert!(!board.contains(start));
        frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(start),press(start,true)]);
        assert!(frame(&ctx,&mut canvas,&all,true,vec![press(start,false)]).is_none());assert_eq!(canvas.selection,vec![10,1]);
        frame(&ctx,&mut canvas,&all,true,vec![press(start,true)]);
        let center=convert(Pos2::new(all[1].x,all[1].y));let delta=start-center;let end=center+Vec2::new(-delta.y,delta.x);
        frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(end)]);
        let edit=frame(&ctx,&mut canvas,&all,true,vec![press(end,false)]).expect("outside rotation control must rotate");
        assert_eq!(edit.primitives[0],all[0]);assert!((edit.primitives[1].angle_deg-all[1].angle_deg).abs()>20.);
    }

    #[test]
    fn outside_board_group_control_resizes_only_ordinary_shapes(){
        let ctx=egui::Context::default();let mut canvas=CanvasState::default();let board=painted_primary_board(&ctx,&mut canvas);
        let mut bg=Primitive::background([40,60,80]);bg.id=10;
        let mut first=p(1,ShapeKind::Rectangle);first.x=5.;first.y=5.;first.angle_deg=0.;
        let mut second=p(2,ShapeKind::Ellipse);second.x=75.;second.y=30.;second.angle_deg=0.;
        let all=vec![bg,first,second];canvas.selection=vec![10,1,2];let scale=board.width()/512.;
        let bounds=selection_bounds(&[&all[0],&all[1],&all[2]]).unwrap();let start=board.min+bounds.min.to_vec2()*scale;
        assert!(!board.contains(start));let end=start-Vec2::new(20.,15.);
        frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(start),press(start,true)]);
        frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(end)]);
        let edit=frame(&ctx,&mut canvas,&all,true,vec![press(end,false)]).expect("outside group corner must resize");
        assert_eq!(edit.primitives[0],all[0]);assert!(edit.primitives[1].width>all[1].width);assert!(edit.primitives[2].width>all[2].width);
    }

    #[test]
    fn gray_editing_area_does_not_become_an_outside_drawing_or_selection_surface(){
        for tool in [Tool::Select,Tool::Rectangle,Tool::Ellipse]{
            let ctx=egui::Context::default();let mut canvas=CanvasState::default();let board=painted_primary_board(&ctx,&mut canvas);
            let mut shape=p(1,ShapeKind::Rectangle);shape.x=5.;shape.y=5.;shape.angle_deg=0.;let all=vec![shape];
            canvas.selection=vec![1];canvas.tool=tool;
            let start=if tool==Tool::Select{board.min-Vec2::splat(80.)}else{board.min+primitive_corners(&all[0])[0].to_vec2()*(board.width()/512.)};
            assert!(!board.contains(start));let end=start-Vec2::new(18.,16.);
            frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(start),press(start,true)]);
            frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(end)]);
            assert!(frame(&ctx,&mut canvas,&all,true,vec![press(end,false)]).is_none());assert!(canvas.gesture.is_none());
        }
    }

    #[test]
    fn overlapping_shapes_cycle_on_real_clicks_and_drag_keeps_the_lower_selection(){
        let ctx=egui::Context::default();let mut canvas=CanvasState::default();
        let all=vec![p(1,ShapeKind::Rectangle),p(2,ShapeKind::Ellipse),p(3,ShapeKind::Rectangle)];
        frame(&ctx,&mut canvas,&all,true,vec![]);let point=Pos2::new(400.,400.);
        for expected in [3,2,1,3,2]{
            assert!(frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(point),press(point,true)]).is_none());
            assert!(frame(&ctx,&mut canvas,&all,true,vec![press(point,false)]).is_none());
            assert_eq!(canvas.selection,vec![expected]);
        }
        frame(&ctx,&mut canvas,&all,true,vec![press(point,true)]);
        let end=point+Vec2::new(30.,12.);frame(&ctx,&mut canvas,&all,true,vec![egui::Event::PointerMoved(end)]);
        let edit=frame(&ctx,&mut canvas,&all,true,vec![press(end,false)]).expect("selected lower shape must remain draggable");
        assert_eq!(canvas.selection,vec![2]);
        assert_eq!(edit.primitives[0],all[0]);assert_eq!(edit.primitives[2],all[2]);
        assert!(edit.primitives[1].x>all[1].x);
    }

    #[test]
    fn read_only_mode_cancels_uncommitted_drag() {
        let ctx = egui::Context::default();
        let mut canvas = CanvasState::default();
        let all = vec![p(1, ShapeKind::Rectangle)];
        frame(&ctx, &mut canvas, &all, true, vec![]);
        let start = Pos2::new(400.0, 400.0);
        frame(
            &ctx,
            &mut canvas,
            &all,
            true,
            vec![egui::Event::PointerMoved(start), press(start, true)],
        );
        frame(
            &ctx,
            &mut canvas,
            &all,
            true,
            vec![egui::Event::PointerMoved(Pos2::new(430.0, 410.0))],
        );
        assert!(frame(
            &ctx,
            &mut canvas,
            &all,
            false,
            vec![press(Pos2::new(430.0, 410.0), false)]
        )
        .is_none());
        assert!(canvas.gesture.is_none());
        assert!(canvas.preview.is_none());
    }
}
