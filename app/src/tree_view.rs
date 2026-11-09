//! Native, unbounded world-space navigation for the creative state tree.
use crate::thumbnails::ThumbnailCache;
use crate::tree_layout::{build_layout, LayoutOptions, TreeDisplayMode, TreeLayout, TreeNode};
use crate::ui::{NodeView, UiModel};
use eframe::egui::{
    self, Align, Color32, FontId, Layout, Pos2, Rect, RichText, Sense, Stroke, Vec2,
};
use std::collections::{HashMap, HashSet};

const CARD: Vec2 = Vec2::new(248.0, 128.0);
const BLUE: Color32 = Color32::from_rgb(58, 110, 242);
const INK: Color32 = Color32::from_rgb(34, 43, 62);
const MUTED: Color32 = Color32::from_rgb(111, 122, 143);
const LINE: Color32 = Color32::from_rgb(198, 209, 227);

fn safe_device_delta(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1_000_000.0, 1_000_000.0)
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug)]
struct Camera {
    center: Pos2,
    zoom: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            center: Pos2::ZERO,
            zoom: 1.0,
        }
    }
}
impl Camera {
    fn to_screen(self, point: Pos2, area: Rect) -> Pos2 {
        area.center() + (point - self.center) * self.zoom
    }
    fn to_world(self, point: Pos2, area: Rect) -> Pos2 {
        self.center + (point - area.center()) / self.zoom
    }
    fn viewport(self, area: Rect) -> Rect {
        Rect::from_center_size(self.center, area.size() / self.zoom)
    }
    fn pan(&mut self, delta: Vec2) {
        // Bound a single device event, not the camera's world position. This
        // prevents corrupt/overflowing scroll events without imposing edges.
        let delta = Vec2::new(safe_device_delta(delta.x), safe_device_delta(delta.y));
        let next = self.center - delta / self.zoom;
        if next.x.is_finite() && next.y.is_finite() {
            self.center = next;
        }
    }
    fn zoom_at(&mut self, anchor: Pos2, area: Rect, target: f32) {
        let world = self.to_world(anchor, area);
        if target.is_nan() {
            return;
        }
        self.zoom = target.clamp(0.000_001, 3.0);
        self.center = world - (anchor - area.center()) / self.zoom;
    }
}

#[derive(Clone, Copy)]
struct MiniMap {
    rect: Rect,
    world: Rect,
    scale: f32,
}
impl MiniMap {
    fn new(rect: Rect, world: Rect) -> Self {
        let world = if world.is_positive() {
            world.expand(30.0)
        } else {
            Rect::from_center_size(Pos2::ZERO, CARD)
        };
        let scale = ((rect.width() - 20.0) / world.width())
            .min((rect.height() - 20.0) / world.height())
            .max(0.000_000_001);
        Self { rect, world, scale }
    }
    fn to_screen(self, point: Pos2) -> Pos2 {
        self.rect.center() + (point - self.world.center()) * self.scale
    }
    fn to_world(self, point: Pos2) -> Pos2 {
        self.world.center() + (point - self.rect.center()) / self.scale
    }
    fn viewport(self, world: Rect) -> Rect {
        Rect::from_min_max(self.to_screen(world.min), self.to_screen(world.max))
    }
}

struct PanDrag {
    pointer: Pos2,
    center: Pos2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeAction {
    Jump(u64),
    DeleteSubtrees(Vec<u64>),
}

struct Marquee {
    start: Pos2,
    end: Pos2,
    additive: HashSet<u64>,
}

pub struct TreeViewState {
    camera: Camera,
    initialized: bool,
    mode: TreeDisplayMode,
    layout_mode: TreeDisplayMode,
    input: Vec<TreeNode>,
    layout: TreeLayout,
    all_positions: HashMap<u64, Pos2>,
    previous_current: Option<u64>,
    pan_drag: Option<PanDrag>,
    selection: HashSet<u64>,
    marquee: Option<Marquee>,
    suppress_primary_click: bool,
    recover_view_after_deletion: bool,
    mini_drag_offset: Option<Vec2>,
    request_fit: bool,
    request_locate: bool,
    last_area: Rect,
    last_minimap: Rect,
    last_mode_strip: Rect,
    mode_button_rects: [Rect; 3],
}

impl Default for TreeViewState {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            initialized: false,
            mode: TreeDisplayMode::All,
            layout_mode: TreeDisplayMode::All,
            input: Vec::new(),
            layout: TreeLayout::default(),
            all_positions: HashMap::new(),
            previous_current: None,
            pan_drag: None,
            selection: HashSet::new(),
            marquee: None,
            suppress_primary_click: false,
            recover_view_after_deletion: false,
            mini_drag_offset: None,
            request_fit: false,
            request_locate: false,
            last_area: Rect::NOTHING,
            last_minimap: Rect::NOTHING,
            last_mode_strip: Rect::NOTHING,
            mode_button_rects: [Rect::NOTHING; 3],
        }
    }
}

impl TreeViewState {
    pub fn request_locate_current(&mut self) { self.request_locate = true; }

    pub fn invalidate_after_deletion(&mut self) {
        self.selection.clear();
        self.marquee = None;
        self.pan_drag = None;
        self.mini_drag_offset = None;
        self.suppress_primary_click = false;
        self.input.clear();
        self.layout = TreeLayout::default();
        self.all_positions.clear();
        self.previous_current = None;
        self.last_minimap = Rect::NOTHING;
        self.request_fit = false;
        self.request_locate = false;
        // Keep the user's camera unless its whole viewport is now outside
        // the surviving tree. Only that case needs a one-time recovery.
        self.recover_view_after_deletion = true;
    }

    pub fn configure(&mut self, compact: bool, fit: bool) {
        self.configure_mode(if compact { TreeDisplayMode::BranchesOnly } else { TreeDisplayMode::All }, fit);
    }

    pub fn configure_mode(&mut self, mode: TreeDisplayMode, fit: bool) {
        self.mode = mode;
        self.request_fit = fit;
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        model: &UiModel<'_>,
        thumbnails: &mut ThumbnailCache,
    ) -> Option<TreeAction> {
        let mut target_zoom = None;
        egui::Frame::none()
            .fill(Color32::from_rgb(247, 249, 252))
            .inner_margin(egui::Margin::symmetric(18.0, 10.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("创作状态树").strong().color(INK));
                    ui.label(
                        RichText::new(format!("{} 个状态", model.nodes.len()))
                            .size(11.0)
                            .color(MUTED),
                    );
                    if !self.selection.is_empty() {
                        ui.label(RichText::new(format!("已选 {} 个",self.selection.len())).size(11.0).color(BLUE));
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.small_button("定位当前").clicked() {
                            self.request_locate = true;
                        }
                        if ui.small_button("全部适应").clicked() {
                            self.request_fit = true;
                        }
                        if ui.small_button("＋").clicked() {
                            target_zoom = Some(self.camera.zoom * 1.2);
                        }
                        ui.label(
                            RichText::new(format!("{:.0}%", self.camera.zoom * 100.0))
                                .size(11.0)
                                .color(MUTED),
                        );
                        if ui.small_button("－").clicked() {
                            target_zoom = Some(self.camera.zoom / 1.2);
                        }
                    });
                });
                ui.label(RichText::new("空白拖动框选 · Del 删除选中分支 · 中键 / 右键 / 空格拖动平移 · 滚轮缩放")
                    .size(10.0).color(MUTED));
            });
        let (response, painter) =
            ui.allocate_painter(ui.available_size().max(Vec2::splat(120.0)), Sense::hover());
        let full_area = response.rect;
        let strip_height = 52.0_f32.min(full_area.height() * 0.3);
        let area = Rect::from_min_max(full_area.min,
            full_area.right_bottom() - Vec2::new(0.0, strip_height));
        let strip = Rect::from_min_max(area.left_bottom(), full_area.right_bottom());
        self.last_area = area;
        self.last_mode_strip = strip;
        self.mode_strip(ui, strip);
        let input: Vec<TreeNode> = model
            .nodes
            .iter()
            .map(|node| TreeNode {
                id: node.id,
                parent: node.parent,
            })
            .collect();
        let changed = input != self.input
            || self.layout_mode != self.mode
            || self.layout.nodes.is_empty();
        if changed {
            let options = LayoutOptions {
                compact: false,
                display_mode: self.mode,
                node_width: CARD.x,
                node_height: CARD.y,
                horizontal_gap: 54.0,
                vertical_gap: 58.0,
            };
            let layout = build_layout(&input, options);
            let positions = complete_positions(&input, &layout);
            if self.initialized {
                // A compact leaf becomes an interpolated hidden step when it
                // gains a child. Anchoring to that moving interpolation on
                // every append made the entire tree drift down. Growth must
                // anchor a card that is displayed in both layouts instead.
                let anchor = if self.layout_mode != self.mode {
                    self.all_positions.contains_key(&model.current_node)
                        .then_some(model.current_node)
                        .or(self.previous_current)
                } else {
                    stable_layout_anchor(
                        model.current_node,
                        self.previous_current,
                        &input,
                        &self.layout,
                        &layout,
                    )
                };
                if let Some((old, new)) = anchor.and_then(|id| {
                    self.all_positions
                        .get(&id)
                        .copied()
                        .zip(positions.get(&id).copied())
                }) {
                    let correction = new - old;
                    self.camera.center += correction;
                    if let Some(drag) = &mut self.pan_drag {
                        // The drag origin uses the same world coordinates as
                        // the camera; otherwise the next pointer frame would
                        // undo this correction and make the view jump.
                        drag.center += correction;
                    }
                    if let Some(marquee) = &mut self.marquee {
                        marquee.start += correction;
                        marquee.end += correction;
                    }
                }
            }
            self.layout = layout;
            self.all_positions = positions;
            self.input = input;
            self.layout_mode = self.mode;
        }
        self.previous_current = Some(model.current_node);
        painter.rect_filled(area, 0.0, Color32::from_rgb(239, 243, 249));
        if !self.initialized && !self.layout.nodes.is_empty() {
            let root = self
                .layout
                .roots
                .first()
                .and_then(|id| self.all_positions.get(id))
                .copied()
                .unwrap_or(Pos2::ZERO);
            self.camera.center = Pos2::new(root.x, root.y + area.height() * 0.5 - 94.0);
            self.initialized = true;
        }
        let world_bounds = Rect::from_min_max(
            Pos2::new(self.layout.bounds.min_x, self.layout.bounds.min_y),
            Pos2::new(self.layout.bounds.max_x, self.layout.bounds.max_y),
        );
        if self.request_fit && !self.layout.nodes.is_empty() {
            self.camera.center = world_bounds.center();
            self.camera.zoom = ((area.width() - 72.0).max(40.0) / world_bounds.width().max(CARD.x))
                .min((area.height() - 72.0).max(40.0) / world_bounds.height().max(CARD.y))
                .clamp(0.000_001, 1.25);
            self.request_fit = false;
            self.pan_drag = None;
        }
        let visible_ids: HashSet<u64> = self.layout.nodes.iter().map(|node| node.id).collect();
        self.selection.retain(|id| self.all_positions.contains_key(id));
        let locate_id = nearest_visible(model.current_node, &self.input, &visible_ids);
        if self.recover_view_after_deletion {
            if !self.camera.viewport(area).intersects(world_bounds) { self.request_locate = true; }
            self.recover_view_after_deletion = false;
        }
        if self.request_locate {
            if let Some(point) = locate_id
                .and_then(|id| self.all_positions.get(&id))
                .copied()
            {
                self.camera.center = point;
                self.camera.zoom = self.camera.zoom.max(0.7);
            }
            self.request_locate = false;
            self.pan_drag = None;
        }
        if let Some(zoom) = target_zoom {
            self.camera.zoom_at(area.center(), area, zoom);
        }
        let mini_rect = Rect::from_min_size(
            Pos2::new(
                (area.right() - 220.0).max(area.left() + 8.0),
                area.top() + 12.0,
            ),
            Vec2::new(208.0_f32.min(area.width() - 16.0), 142.0),
        );
        self.last_minimap = mini_rect;
        let map = MiniMap::new(mini_rect.shrink2(Vec2::new(0.0, 8.0)), world_bounds);
        let snapshot = ui.input(|i| InputSnapshot {
            pointer: i.pointer.interact_pos(),
            press_origin: i.pointer.press_origin(),
            primary_pressed: i.pointer.button_pressed(egui::PointerButton::Primary),
            middle_pressed: i.pointer.button_pressed(egui::PointerButton::Middle),
            secondary_pressed: i.pointer.button_pressed(egui::PointerButton::Secondary),
            primary_down: i.pointer.primary_down(),
            middle_down: i.pointer.middle_down(),
            secondary_down: i.pointer.secondary_down(),
            space: i.key_down(egui::Key::Space),
            shift: i.modifiers.shift,
            released: i.pointer.any_released(),
            delta: i.smooth_scroll_delta,
            pinch_zoom: i.zoom_delta(),
        });
        if changed && self.mini_drag_offset.is_some() {
            if let Some(pointer) = snapshot.pointer {
                // The minimap changes scale as the tree grows. Rebase its
                // drag offset so a stationary pointer does not pan the camera.
                self.mini_drag_offset = Some(
                    self.camera.center - map.to_world(pointer.clamp(mini_rect.min, mini_rect.max)),
                );
            }
        }
        if ui.is_enabled() {
            self.navigate(snapshot, area, mini_rect, map);
        } else {
            self.pan_drag = None;
            self.marquee = None;
            self.mini_drag_offset = None;
        }

        let camera = self.camera;
        let clip = painter.with_clip_rect(area);
        let viewport = camera.viewport(area).expand(CARD.x);
        let positions: HashMap<u64, Pos2> = self
            .layout
            .nodes
            .iter()
            .map(|node| (node.id, Pos2::new(node.x, node.y)))
            .collect();
        let highlighted = locate_id.filter(|_| !visible_ids.contains(&model.current_node));
        for edge in &self.layout.edges {
            if let Some((a, b)) = positions.get(&edge.parent).zip(positions.get(&edge.child)) {
                let start_world = *a + Vec2::new(0.0, CARD.y * 0.5);
                let end_world = *b - Vec2::new(0.0, CARD.y * 0.5);
                if !viewport.intersects(Rect::from_two_pos(start_world, end_world)) {
                    continue;
                }
                let start = camera.to_screen(start_world, area);
                let end = camera.to_screen(end_world, area);
                let middle = (start.y + end.y) * 0.5;
                clip.add(egui::epaint::CubicBezierShape::from_points_stroke(
                    [
                        start,
                        Pos2::new(start.x, middle),
                        Pos2::new(end.x, middle),
                        end,
                    ],
                    false,
                    Color32::TRANSPARENT,
                    Stroke::new((1.8 * camera.zoom).clamp(0.7, 2.0), LINE),
                ));
                if edge.hidden_steps > 0 && camera.zoom >= 0.45 {
                    let center = Pos2::new((start.x + end.x) * 0.5, middle);
                    let label = format!("{} 步", edge.hidden_steps);
                    let galley = clip.layout_no_wrap(
                        label,
                        FontId::proportional((10.0 * camera.zoom).max(8.0)),
                        MUTED,
                    );
                    let bounds =
                        Rect::from_center_size(center, galley.size() + Vec2::new(12.0, 5.0));
                    clip.rect_filled(bounds, 5.0, Color32::from_rgb(239, 243, 249));
                    clip.galley(bounds.center() - galley.size() * 0.5, galley, MUTED);
                }
            }
        }
        let node_info: HashMap<u64, &NodeView> =
            model.nodes.iter().map(|node| (node.id, node)).collect();
        let mut clicked = None;
        for node in &self.layout.nodes {
            let center = Pos2::new(node.x, node.y);
            if !viewport.contains(center) {
                continue;
            }
            let rect = Rect::from_center_size(camera.to_screen(center, area), CARD * camera.zoom);
            if !rect.intersects(area) {
                continue;
            }
            let Some(info) = node_info.get(&node.id).copied() else {
                continue;
            };
            let pointer_in_minimap = snapshot
                .pointer
                .map(|p| mini_rect.contains(p))
                .unwrap_or(false);
            let response = ui.interact(
                rect.intersect(area),
                ui.id().with(("tree-node", node.id)),
                if model.loading {
                    Sense::hover()
                } else {
                    Sense::click()
                },
            );
            paint_card(
                &clip,
                rect,
                info,
                node.id == model.current_node,
                highlighted == Some(node.id),
                response.hovered() && !pointer_in_minimap,
                camera.zoom,
                if camera.zoom >= 0.32 {
                    thumbnails.texture(ui.ctx(), model.document, node.id)
                } else {
                    None
                },
            );
            if self.selection.contains(&node.id) {
                clip.rect_stroke(rect.expand((2.0 * camera.zoom).max(1.0)),
                    (8.0 * camera.zoom).min(8.0), Stroke::new(2.2, BLUE));
            }
            if response.clicked()
                && !pointer_in_minimap
                && !model.loading
                && !self.suppress_primary_click
            {
                if !snapshot.shift { self.selection.clear(); }
                self.selection.insert(node.id);
                if node.id != model.current_node { clicked = Some(TreeAction::Jump(node.id)); }
            }
            if !pointer_in_minimap {
                response.on_hover_text(if model.loading {
                    "模型加载中，暂不能切换状态".to_owned()
                } else if model.running {
                    format!("{}\n查看此状态；生成继续进行", info.title)
                } else {
                    format!("{}\n点击回到此状态", info.title)
                });
            }
        }
        if let Some(marquee) = &self.marquee {
            let bounds = Rect::from_two_pos(camera.to_screen(marquee.start, area),
                camera.to_screen(marquee.end, area));
            clip.rect_filled(bounds, 1.0, Color32::from_rgba_unmultiplied(58,110,242,24));
            clip.rect_stroke(bounds, 1.0, Stroke::new(1.2,BLUE));
        }
        if self.mode != TreeDisplayMode::All && !visible_ids.contains(&model.current_node) {
            let text = format!("当前状态 #{} 位于隐藏的连续步骤中", model.current_node);
            clip.text(
                area.left_bottom() + Vec2::new(16.0, -18.0),
                egui::Align2::LEFT_BOTTOM,
                text,
                FontId::proportional(11.0),
                MUTED,
            );
        }
        self.paint_minimap(ui, &clip, model, map, mini_rect);
        if ui.is_enabled() && !model.loading && !model.compiling && !model.stopping
            && !text_input_has_focus(ui.ctx()) && !self.selection.is_empty()
            && ui.input(|i|i.key_pressed(egui::Key::Delete)) {
            let mut selected = self.selection.iter().copied().collect::<Vec<_>>();
            selected.sort_unstable();
            return Some(TreeAction::DeleteSubtrees(selected));
        }
        clicked
    }

    fn mode_strip(&mut self, ui: &mut egui::Ui, rect: Rect) {
        ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(247, 249, 252));
        ui.painter().line_segment([rect.left_top(), rect.right_top()],
            Stroke::new(1.0, Color32::from_rgb(224, 230, 240)));
        let width = ((rect.width() - 52.0) / 3.0).clamp(90.0, 144.0);
        let group_width = width * 3.0 + 16.0;
        let bounds = Rect::from_center_size(rect.center(), Vec2::new(group_width, 32.0));
        ui.allocate_ui_at_rect(bounds, |ui| {
            ui.horizontal(|ui| {
                for (index, (mode, label, hint)) in [
                    (TreeDisplayMode::All, "显示全部节点", "显示每个创作状态。"),
                    (TreeDisplayMode::EmphasizeBranches, "着重显示岔点", "保留起点、分叉点、终点，并在连续步骤中每50步显示一个预览。"),
                    (TreeDisplayMode::BranchesOnly, "只显示岔点", "保留起点、终点、分叉点与每次分叉后的首个状态。"),
                ].into_iter().enumerate() {
                    let selected = self.mode == mode;
                    let response = ui.add_sized([width, 32.0],
                        egui::Button::new(RichText::new(label).size(12.0)
                            .color(if selected { Color32::WHITE } else { MUTED }))
                            .fill(if selected { BLUE } else { Color32::from_rgb(234, 239, 247) })
                            .stroke(Stroke::NONE).rounding(6.0));
                    self.mode_button_rects[index] = response.rect;
                    if response.clicked() { self.mode = mode; }
                    response.on_hover_text(hint);
                }
            });
        });
    }

    fn navigate(&mut self, input: InputSnapshot, area: Rect, mini_rect: Rect, map: MiniMap) {
        if input.primary_pressed { self.suppress_primary_click = false; }
        if let Some(pointer) = input.pointer {
            if input.primary_pressed && mini_rect.contains(pointer) {
                let viewport = map.viewport(self.camera.viewport(area));
                self.mini_drag_offset = Some(if viewport.contains(pointer) {
                    self.camera.center - map.to_world(pointer)
                } else {
                    Vec2::ZERO
                });
                self.pan_drag = None;
                self.marquee = None;
            }
            if let Some(offset) = self.mini_drag_offset {
                if input.primary_down || input.primary_pressed {
                    self.camera.center =
                        map.to_world(pointer.clamp(mini_rect.min, mini_rect.max)) + offset;
                }
                if input.released || !input.primary_down {
                    self.mini_drag_offset = None;
                }
                return;
            }
            // An existing box selection finishes even when the pointer leaves
            // the tree. Its end is clamped to the visible drawing viewport.
            if let Some(marquee) = &mut self.marquee {
                marquee.end = self.camera.to_world(pointer.clamp(area.min,area.max),area);
                let bounds = Rect::from_two_pos(marquee.start,marquee.end);
                self.selection.clone_from(&marquee.additive);
                for node in &self.layout.nodes {
                    let card = Rect::from_center_size(Pos2::new(node.x,node.y),CARD);
                    if card.intersects(bounds) && card.intersects(self.camera.viewport(area)) {
                        self.selection.insert(node.id);
                    }
                }
                if input.released || !input.primary_down { self.marquee = None; }
                return;
            }
            if !area.contains(pointer) {
                if input.released {
                    self.pan_drag = None;
                }
                return;
            }
            if mini_rect.contains(pointer) {
                // The minimap controls the main viewport. Wheel zoom keeps
                // that viewport centered rather than jumping to a distant map point.
                self.wheel_zoom(input, area.center(), area);
                return;
            }
            let in_card = self.layout.nodes.iter().any(|node| {
                Rect::from_center_size(
                    self.camera.to_screen(Pos2::new(node.x, node.y), area),
                    CARD * self.camera.zoom,
                )
                .contains(pointer)
            });
            if input.middle_pressed || input.secondary_pressed || (input.primary_pressed && input.space) {
                self.pan_drag = Some(PanDrag {
                    pointer,
                    center: self.camera.center,
                });
                self.suppress_primary_click = true;
            } else if input.primary_pressed && !in_card {
                let additive = if input.shift { self.selection.clone() } else { HashSet::new() };
                self.selection.clone_from(&additive);
                let start = self.camera.to_world(pointer,area);
                self.marquee = Some(Marquee { start, end:start, additive });
                self.suppress_primary_click = true;
            }
            if let Some(drag) = &self.pan_drag {
                if input.primary_down || input.middle_down || input.secondary_down {
                    self.camera.center = drag.center - (pointer - drag.pointer) / self.camera.zoom;
                }
                if input.released || (!input.primary_down && !input.middle_down && !input.secondary_down) {
                    self.pan_drag = None;
                }
            }
            self.wheel_zoom(input, pointer, area);
        } else if input.released {
            self.pan_drag = None;
            self.mini_drag_offset = None;
            self.marquee = None;
        }
    }

    fn wheel_zoom(&mut self, input: InputSnapshot, anchor: Pos2, area: Rect) {
        let pinch = input.pinch_zoom;
        let factor = if pinch.is_finite() && pinch > 0.0 && (pinch-1.0).abs()>0.001 {
            pinch
        } else {
            (safe_device_delta(input.delta.y).clamp(-240.0,240.0)*0.004).exp()
        };
        if (factor-1.0).abs() <= f32::EPSILON { return; }
        self.camera.zoom_at(anchor, area, self.camera.zoom*factor);
        if let Some(drag) = &mut self.pan_drag {
            drag.pointer = input.pointer.unwrap_or(anchor);
            drag.center = self.camera.center;
        }
    }

    fn paint_minimap(
        &self,
        ui: &mut egui::Ui,
        painter: &egui::Painter,
        model: &UiModel<'_>,
        map: MiniMap,
        rect: Rect,
    ) {
        painter.rect_filled(
            rect.translate(Vec2::new(0.0, 3.0)),
            8.0,
            Color32::from_black_alpha(12),
        );
        painter.rect_filled(rect, 8.0, Color32::from_rgb(252, 253, 255));
        painter.rect_stroke(
            rect,
            8.0,
            Stroke::new(1.0, Color32::from_rgb(214, 223, 238)),
        );
        let painter = painter.with_clip_rect(rect.shrink(5.0));
        for node in &self.input {
            if let Some((a, b)) = node
                .parent
                .and_then(|parent| self.all_positions.get(&parent))
                .zip(self.all_positions.get(&node.id))
            {
                painter.line_segment(
                    [map.to_screen(*a), map.to_screen(*b)],
                    Stroke::new(1.0, Color32::from_rgb(155, 174, 202)),
                );
            }
        }
        for root in &self.layout.roots {
            if let Some(p) = self.all_positions.get(root) {
                painter.circle_filled(map.to_screen(*p), 2.5, MUTED);
            }
        }
        let view = map.viewport(self.camera.viewport(self.last_area));
        painter.rect_filled(view, 1.0, Color32::from_rgba_unmultiplied(58, 110, 242, 18));
        painter.rect_stroke(view, 1.0, Stroke::new(1.2, BLUE));
        if let Some(point) = self.all_positions.get(&model.current_node) {
            painter.circle_filled(map.to_screen(*point), 3.3, BLUE);
        }
        ui.interact(rect, ui.id().with("tree-minimap"), Sense::click_and_drag())
            .on_hover_text("完整结构总览：点击或拖动以导航，滚轮缩放当前视图。连续隐藏步骤沿连接线分布。");
    }
}

fn text_input_has_focus(ctx: &egui::Context) -> bool {
    ctx.memory(|memory|memory.focused())
        .is_some_and(|id|egui::TextEdit::load_state(ctx,id).is_some())
}

#[derive(Clone, Copy, Default)]
struct InputSnapshot {
    pointer: Option<Pos2>,
    press_origin: Option<Pos2>,
    primary_pressed: bool,
    middle_pressed: bool,
    secondary_pressed: bool,
    primary_down: bool,
    middle_down: bool,
    secondary_down: bool,
    space: bool,
    shift: bool,
    released: bool,
    delta: Vec2,
    pinch_zoom: f32,
}

fn complete_positions(input: &[TreeNode], layout: &TreeLayout) -> HashMap<u64, Pos2> {
    let mut positions: HashMap<u64, Pos2> = layout
        .nodes
        .iter()
        .map(|node| (node.id, Pos2::new(node.x, node.y)))
        .collect();
    let parents: HashMap<u64, Option<u64>> =
        input.iter().map(|node| (node.id, node.parent)).collect();
    for edge in &layout.edges {
        if edge.hidden_steps == 0 {
            continue;
        }
        let Some((start, end)) = positions
            .get(&edge.parent)
            .copied()
            .zip(positions.get(&edge.child).copied())
        else {
            continue;
        };
        let mut chain = vec![edge.child];
        let mut cursor = edge.child;
        for _ in 0..input.len() {
            let Some(parent) = parents.get(&cursor).copied().flatten() else {
                break;
            };
            chain.push(parent);
            cursor = parent;
            if parent == edge.parent {
                break;
            }
        }
        if cursor != edge.parent {
            continue;
        }
        chain.reverse();
        let steps = (chain.len() - 1).max(1) as f32;
        for (index, id) in chain.into_iter().enumerate() {
            positions.insert(id, start.lerp(end, index as f32 / steps));
        }
    }
    positions
}

fn stable_layout_anchor(
    current: u64,
    previous_current: Option<u64>,
    input: &[TreeNode],
    previous: &TreeLayout,
    next: &TreeLayout,
) -> Option<u64> {
    let old_visible: HashSet<u64> = previous.nodes.iter().map(|node| node.id).collect();
    let common: HashSet<u64> = next.nodes.iter().map(|node| node.id)
        .filter(|id| old_visible.contains(id)).collect();
    for origin in [Some(current), previous_current].into_iter().flatten() {
        if let Some(anchor) = nearest_visible(origin, input, &common) {
            return Some(anchor);
        }
    }
    previous.roots.iter().copied().find(|id| common.contains(id))
        .or_else(|| previous.nodes.iter().map(|node| node.id).find(|id| common.contains(id)))
}

fn nearest_visible(current: u64, input: &[TreeNode], visible: &HashSet<u64>) -> Option<u64> {
    let parents: HashMap<u64, Option<u64>> =
        input.iter().map(|node| (node.id, node.parent)).collect();
    let mut cursor = current;
    for _ in 0..=input.len() {
        if visible.contains(&cursor) {
            return Some(cursor);
        }
        cursor = parents.get(&cursor).copied().flatten()?;
    }
    None
}

fn paint_card(
    painter: &egui::Painter,
    rect: Rect,
    node: &NodeView,
    active: bool,
    ancestor: bool,
    hovered: bool,
    zoom: f32,
    texture: Option<egui::TextureId>,
) {
    let source_color =
        if node.source.contains("模型") || node.source.to_ascii_lowercase().contains("model") {
            Color32::from_rgb(126, 91, 212)
        } else {
            BLUE
        };
    let fill = if active {
        Color32::from_rgb(234, 240, 255)
    } else if hovered {
        Color32::from_rgb(247, 250, 255)
    } else {
        Color32::WHITE
    };
    painter.rect_filled(
        rect.translate(Vec2::new(0.0, 3.0 * zoom)),
        (8.0 * zoom).min(8.0),
        Color32::from_black_alpha(9),
    );
    painter.rect_filled(rect, 8.0 * zoom, fill);
    painter.rect_stroke(
        rect,
        8.0 * zoom,
        Stroke::new(
            if active || ancestor { 1.6 } else { 0.9 },
            if active || ancestor {
                BLUE
            } else {
                Color32::from_rgb(220, 228, 240)
            },
        ),
    );
    if zoom < 0.32 {
        painter.rect_filled(
            Rect::from_min_size(rect.min, Vec2::new((3.0 * zoom).max(1.0), rect.height())),
            0.0,
            source_color,
        );
        return;
    }
    painter.circle_filled(
        rect.min + Vec2::new(13.0, 15.0) * zoom,
        3.0 * zoom,
        source_color,
    );
    painter.text(
        rect.min + Vec2::new(23.0, 15.0) * zoom,
        egui::Align2::LEFT_CENTER,
        format!("#{}  {}", node.id, node.source),
        FontId::proportional(10.0 * zoom),
        MUTED,
    );
    let thumb = Rect::from_min_size(
        rect.min + Vec2::new(10.0, 33.0) * zoom,
        Vec2::splat(84.0 * zoom),
    );
    painter.rect_filled(thumb, 3.0 * zoom, Color32::WHITE);
    if let Some(texture) = texture {
        painter.image(
            texture,
            thumb,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }
    painter.rect_stroke(
        thumb,
        2.0 * zoom,
        Stroke::new(0.7, Color32::from_rgb(225, 231, 240)),
    );
    let text_origin = rect.min + Vec2::new(106.0, 38.0) * zoom;
    let galley = painter.layout(
        node.title.clone(),
        FontId::proportional(12.0 * zoom),
        INK,
        128.0 * zoom,
    );
    let clip = painter.with_clip_rect(Rect::from_min_max(
        text_origin,
        rect.max - Vec2::splat(9.0 * zoom),
    ));
    clip.galley(text_origin, galley, INK);
    painter.text(
        rect.min + Vec2::new(106.0, 104.0) * zoom,
        egui::Align2::LEFT_CENTER,
        format!("{} 个图元", node.primitive_count),
        FontId::proportional(10.0 * zoom),
        MUTED,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Document, OperationSource, Primitive, ShapeKind};

    fn fixture() -> Document {
        let mut document = Document::new();
        for i in 0..5 {
            let mut next = document.primitives.clone();
            next.push(Primitive::new(
                ShapeKind::Rectangle,
                90.0 + i as f32 * 35.0,
                150.0,
                45.0,
                40.0,
                [80, 120, 200],
            ));
            document
                .commit_canvas(format!("添加矩形 {i}"), OperationSource::User, next)
                .unwrap();
        }
        document
    }

    fn input_frame(
        ctx: &egui::Context,
        state: &mut TreeViewState,
        document: &Document,
        thumbnails: &mut ThumbnailCache,
        events: Vec<egui::Event>,
    ) -> Option<TreeAction> {
        input_frame_sized(ctx, state, document, thumbnails, Vec2::new(900.0, 700.0), events)
    }

    fn input_frame_sized(
        ctx: &egui::Context,
        state: &mut TreeViewState,
        document: &Document,
        thumbnails: &mut ThumbnailCache,
        size: Vec2,
        events: Vec<egui::Event>,
    ) -> Option<TreeAction> {
        input_frame_with_status(ctx, state, document, thumbnails, size, events, (false, false, false))
    }

    fn input_frame_with_status(
        ctx: &egui::Context,
        state: &mut TreeViewState,
        document: &Document,
        thumbnails: &mut ThumbnailCache,
        size: Vec2,
        events: Vec<egui::Event>,
        (running, loading, stopping): (bool, bool, bool),
    ) -> Option<TreeAction> {
        let nodes: Vec<NodeView> = document
            .nodes
            .iter()
            .map(|node| NodeView {
                id: node.id as u64,
                parent: node.parent.map(|id| id as u64),
                title: node.title.clone(),
                source: "用户".into(),
                primitive_count: node.primitive_count,
            })
            .collect();
        let drawing_region = crate::region_state::DrawingRegion::default();
        let model = UiModel {
            document,
            stage_by_id:None,
            drawing_region: &drawing_region,
            primitives: &document.primitives,
            prompt: &document.prompt,
            settings: &document.settings,
            current_node: document.current as u64,
            nodes: &nodes,
            running,
            loading,
            compiling: false,
            stopping,
            model_loaded: false,
            model_status: "未加载",
            model_path: "",
            tokenizer_path: "",
            error: None,
            status: "就绪",
        };
        let mut action = None;
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)),
            events,
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                action = state.show(ui, &model, thumbnails);
            });
        });
        action
    }

    fn pointer_button(pos: Pos2, pressed: bool) -> egui::Event {
        pointer_kind(pos,egui::PointerButton::Primary,pressed)
    }

    fn pointer_kind(pos: Pos2, button:egui::PointerButton, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers: egui::Modifiers::default(),
        }
    }

    fn append_step(document: &mut Document) {
        let mut next = document.primitives.clone();
        next.push(Primitive::new(
            ShapeKind::Ellipse, 180.0, 200.0, 30.0, 25.0, [110, 140, 200],
        ));
        document.commit_canvas("生成一个图元", OperationSource::Model, next).unwrap();
    }

    fn delete_key() -> egui::Event {
        egui::Event::Key { key:egui::Key::Delete,physical_key:None,pressed:true,repeat:false,modifiers:Default::default() }
    }

    #[test]
    fn blank_drag_selects_cards_without_pan_or_stopping_and_delete_requests_subtrees() {
        let ctx=egui::Context::default();let mut tree=TreeViewState::default();
        let document=fixture();let mut thumbnails=ThumbnailCache::new();
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);
        let before=tree.camera.center;
        let root=tree.camera.to_screen(tree.all_positions[&0],tree.last_area);
        let one=tree.camera.to_screen(tree.all_positions[&1],tree.last_area);
        let start=root-Vec2::new(CARD.x/2.+12.,CARD.y/2.+8.);
        let end=one+Vec2::new(CARD.x/2.+12.,CARD.y/2.+8.);
        for events in [vec![egui::Event::PointerMoved(start),pointer_button(start,true)],
            vec![egui::Event::PointerMoved(end)],vec![pointer_button(end,false)]] {
            assert!(input_frame_with_status(&ctx,&mut tree,&document,&mut thumbnails,Vec2::new(900.,700.),events,(true,false,false)).is_none());
        }
        assert_eq!(tree.selection,HashSet::from([0,1]));
        assert!(tree.camera.center.distance(before)<0.01);assert!(tree.pan_drag.is_none());
        assert_eq!(input_frame_with_status(&ctx,&mut tree,&document,&mut thumbnails,Vec2::new(900.,700.),vec![delete_key()],(true,false,false)),
            Some(TreeAction::DeleteSubtrees(vec![0,1])));
        assert_eq!(document.nodes.len(),6,"tree selection/deletion requests never mutate the document");
    }

    #[test]
    fn compact_marquee_selects_visible_cards_only_and_keeps_hidden_descendants_for_document() {
        let ctx=egui::Context::default();let mut tree=TreeViewState::default();tree.configure(true,false);
        let document=fixture();let mut thumbnails=ThumbnailCache::new();
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);
        let root=tree.camera.to_screen(tree.all_positions[&0],tree.last_area);
        let leaf=tree.camera.to_screen(tree.all_positions[&5],tree.last_area);
        let start=root-Vec2::new(CARD.x/2.+10.,CARD.y/2.+8.);
        let end=leaf+Vec2::new(CARD.x/2.+10.,CARD.y/2.+8.);
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![egui::Event::PointerMoved(start),pointer_button(start,true)]);
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![egui::Event::PointerMoved(end)]);
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![pointer_button(end,false)]);
        assert_eq!(tree.selection,HashSet::from([0,5]));
        assert_eq!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![delete_key()]),Some(TreeAction::DeleteSubtrees(vec![0,5])));
    }

    #[test]
    fn a_selected_compact_leaf_remains_selected_when_live_generation_hides_it() {
        let ctx=egui::Context::default();let mut tree=TreeViewState::default();tree.configure(true,false);
        let mut document=fixture();let mut thumbnails=ThumbnailCache::new();
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);
        let point=tree.camera.to_screen(tree.all_positions[&5],tree.last_area);
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![egui::Event::PointerMoved(point),pointer_button(point,true)]);
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![pointer_button(point,false)]);
        assert_eq!(tree.selection,HashSet::from([5]));
        append_step(&mut document);input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);
        assert!(tree.layout.node(5).is_none());assert_eq!(tree.selection,HashSet::from([5]));
        assert_eq!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![delete_key()]),Some(TreeAction::DeleteSubtrees(vec![5])));
    }

    #[test]
    fn delete_is_disabled_while_loading_or_stopping_and_empty_selection_is_a_noop() {
        for status in [(false,true,false),(true,false,true),(false,false,true)] {
            let ctx=egui::Context::default();let mut tree=TreeViewState::default();let document=fixture();let mut thumbnails=ThumbnailCache::new();
            tree.selection.insert(1);
            assert!(input_frame_with_status(&ctx,&mut tree,&document,&mut thumbnails,Vec2::new(900.,700.),vec![delete_key()],status).is_none());
            assert_eq!(tree.selection,HashSet::from([1]));
        }
        let ctx=egui::Context::default();let mut tree=TreeViewState::default();let document=fixture();let mut thumbnails=ThumbnailCache::new();
        assert!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![delete_key()]).is_none());
    }

    #[test]
    fn clicking_a_focusable_card_then_delete_is_not_mistaken_for_text_editing() {
        let ctx=egui::Context::default();let mut tree=TreeViewState::default();let document=fixture();let mut thumbnails=ThumbnailCache::new();
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);
        let point=tree.camera.to_screen(tree.all_positions[&1],tree.last_area);
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![egui::Event::PointerMoved(point),pointer_button(point,true)]);
        assert_eq!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![pointer_button(point,false)]),Some(TreeAction::Jump(1)));
        assert_eq!(tree.selection,HashSet::from([1]));
        assert!(!text_input_has_focus(&ctx));
        assert_eq!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![delete_key()]),Some(TreeAction::DeleteSubtrees(vec![1])));
    }

    #[test]
    fn text_focus_suppresses_delete_without_clearing_the_tree_selection() {
        let ctx=egui::Context::default();let mut tree=TreeViewState::default();let document=fixture();let mut thumbnails=ThumbnailCache::new();
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);tree.selection.insert(1);
        let id=egui::Id::new("history-delete-test-text");
        egui::TextEdit::store_state(&ctx,id,Default::default());ctx.memory_mut(|m|m.request_focus(id));
        assert!(text_input_has_focus(&ctx));
        assert!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![delete_key()]).is_none());
        assert_eq!(tree.selection,HashSet::from([1]));
    }

    #[test]
    fn right_and_space_left_pan_over_a_card_without_selecting_or_jumping() {
        for space in [false,true] {
            let ctx=egui::Context::default();let mut tree=TreeViewState::default();let document=fixture();let mut thumbnails=ThumbnailCache::new();
            input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);
            let start=tree.camera.to_screen(tree.all_positions[&1],tree.last_area);let end=start+Vec2::new(45.,30.);let before=tree.camera.center;
            let button=if space{egui::PointerButton::Primary}else{egui::PointerButton::Secondary};
            let mut press=vec![egui::Event::PointerMoved(start)];
            if space {press.push(egui::Event::Key{key:egui::Key::Space,physical_key:None,pressed:true,repeat:false,modifiers:Default::default()});}
            press.push(pointer_kind(start,button,true));
            assert!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,press).is_none());
            assert!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![egui::Event::PointerMoved(end)]).is_none());
            assert!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![pointer_kind(end,button,false)]).is_none());
            assert!(tree.selection.is_empty());assert!((tree.camera.center-before-Vec2::new(-45.,-30.)).length()<0.01);
        }
    }

    #[test]
    fn deletion_invalidation_drops_numeric_id_caches_and_selection_without_moving_camera() {
        let ctx=egui::Context::default();let mut tree=TreeViewState::default();let document=fixture();let mut thumbnails=ThumbnailCache::new();
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);tree.selection.insert(1);
        let before=tree.camera;tree.invalidate_after_deletion();
        assert!(tree.selection.is_empty()&&tree.input.is_empty()&&tree.layout.nodes.is_empty()&&tree.all_positions.is_empty());
        assert!(tree.previous_current.is_none()&&tree.pan_drag.is_none()&&tree.marquee.is_none()&&tree.mini_drag_offset.is_none());
        assert_eq!(tree.camera.zoom,before.zoom);assert!(tree.camera.center.distance(before.center)<0.01);
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);
        assert!(tree.camera.center.distance(before.center)<0.01,"unchanged surviving bounds must not trigger a camera jump");
    }

    #[test]
    fn real_history_click_during_generation_returns_only_the_requested_node() {
        for status in [(true, false, false), (true, false, true), (false, false, true)] {
            let ctx = egui::Context::default();
            let mut tree = TreeViewState::default();
            let document = fixture();
            let current = document.current;
            let node_count = document.nodes.len();
            let mut thumbnails = ThumbnailCache::new();
            let size = Vec2::new(900.0, 700.0);
            input_frame_with_status(&ctx, &mut tree, &document, &mut thumbnails, size, vec![], status);
            let point = tree.camera.to_screen(tree.all_positions[&1], tree.last_area);
            assert!(tree.last_area.contains(point) && !tree.last_minimap.contains(point));
            assert!(input_frame_with_status(&ctx, &mut tree, &document, &mut thumbnails, size,
                vec![egui::Event::PointerMoved(point), pointer_button(point, true)], status).is_none());
            assert_eq!(input_frame_with_status(&ctx, &mut tree, &document, &mut thumbnails, size,
                vec![pointer_button(point, false)], status), Some(TreeAction::Jump(1)));
            // The view only requests a jump. It cannot stop the worker or edit
            // the running generation's document/history itself.
            assert_eq!(document.current, current);
            assert_eq!(document.nodes.len(), node_count);
        }
    }

    #[test]
    fn model_loading_still_blocks_real_history_clicks_even_if_generation_is_marked_running() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        let document = fixture();
        let mut thumbnails = ThumbnailCache::new();
        let size = Vec2::new(900.0, 700.0);
        let status = (true, true, false);
        input_frame_with_status(&ctx, &mut tree, &document, &mut thumbnails, size, vec![], status);
        let point = tree.camera.to_screen(tree.all_positions[&1], tree.last_area);
        for pressed in [true, false] {
            assert!(input_frame_with_status(&ctx, &mut tree, &document, &mut thumbnails, size,
                vec![egui::Event::PointerMoved(point), pointer_button(point, pressed)], status).is_none());
        }
    }

    fn click_mode(
        ctx: &egui::Context, tree: &mut TreeViewState, document: &Document,
        thumbnails: &mut ThumbnailCache, index: usize,
    ) {
        let point = tree.mode_button_rects[index].center();
        for pressed in [true, false] {
            assert!(input_frame(ctx, tree, document, thumbnails, vec![
                egui::Event::PointerMoved(point), pointer_button(point, pressed),
            ]).is_none(), "display buttons cannot jump to a document node");
        }
    }

    #[test]
    fn footer_buttons_select_one_mode_and_rebuild_the_same_frame_without_editing_history() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        let mut document = fixture();
        let old_leaf = document.current;
        document.jump(2).unwrap();
        append_step(&mut document);
        document.jump(old_leaf).unwrap();
        let current = document.current;
        let count = document.nodes.len();
        let primitives = document.primitives.clone();
        let mut thumbnails = ThumbnailCache::new();
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
        for (index, expected) in [
            (1, TreeDisplayMode::EmphasizeBranches),
            (2, TreeDisplayMode::BranchesOnly),
            (0, TreeDisplayMode::All),
        ] {
            click_mode(&ctx, &mut tree, &document, &mut thumbnails, index);
            assert_eq!(tree.mode, expected);
            assert_eq!(tree.layout_mode, expected, "layout must update in the button release frame");
            assert_eq!([TreeDisplayMode::All, TreeDisplayMode::EmphasizeBranches, TreeDisplayMode::BranchesOnly]
                .iter().filter(|&&mode| tree.mode == mode).count(), 1);
            let visible: Vec<_> = tree.layout.nodes.iter().map(|node| node.id).collect();
            let camera = tree.camera.center;
            click_mode(&ctx, &mut tree, &document, &mut thumbnails, index);
            assert_eq!(tree.mode, expected, "clicking the active mode cannot turn it off");
            assert_eq!(tree.layout.nodes.iter().map(|node| node.id).collect::<Vec<_>>(), visible);
            assert!(tree.camera.center.distance(camera) < 0.01);
            assert_eq!(document.current, current);
            assert_eq!(document.nodes.len(), count);
            assert_eq!(document.primitives, primitives);
        }
    }

    #[test]
    fn emphasized_mode_shows_only_forks_and_fifty_step_previews_on_a_single_chain() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        tree.configure_mode(TreeDisplayMode::EmphasizeBranches, false);
        let mut document = fixture();
        let mut thumbnails = ThumbnailCache::new();
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
        assert_eq!(tree.layout.nodes.iter().map(|node| node.id).collect::<Vec<_>>(), vec![0, 5]);
        while document.current < 121 { append_step(&mut document); }
        document.jump(73).unwrap();
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
        assert_eq!(tree.layout.nodes.iter().map(|node| node.id).collect::<Vec<_>>(), vec![0, 50, 100, 121]);
        assert_eq!(tree.all_positions.len(), document.nodes.len(), "minimap must still retain every original step");
        let visible = tree.layout.nodes.iter().map(|node| node.id).collect();
        assert_eq!(nearest_visible(73, &tree.input, &visible), Some(50));
        tree.request_locate = true;
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
        assert!(tree.camera.center.distance(tree.all_positions[&50]) < 0.01);
        assert_eq!(document.current, 73, "locating a hidden step must not switch history");
    }

    #[test]
    fn footer_controls_are_reserved_outside_tree_and_minimap_on_a_small_window() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        let document = fixture();
        let mut thumbnails = ThumbnailCache::new();
        let size = Vec2::new(1100.0, 720.0);
        input_frame_sized(&ctx, &mut tree, &document, &mut thumbnails, size, vec![]);
        assert!(tree.last_area.bottom() <= tree.last_mode_strip.top());
        assert!(tree.last_minimap.bottom() < tree.last_mode_strip.top());
        assert!(tree.last_mode_strip.bottom() <= size.y);
        for button in tree.mode_button_rects {
            assert!(tree.last_mode_strip.contains_rect(button));
            assert!(!tree.last_area.intersects(button));
            assert!(!tree.last_minimap.intersects(button));
        }
        assert!(tree.mode_button_rects[0].right() < tree.mode_button_rects[1].left());
        assert!(tree.mode_button_rects[1].right() < tree.mode_button_rects[2].left());
        let before = tree.camera;
        let point = tree.mode_button_rects[1].center();
        input_frame_sized(&ctx, &mut tree, &document, &mut thumbnails, size,
            vec![egui::Event::PointerMoved(point), egui::Event::Scroll(Vec2::new(0.0, 90.0))]);
        assert_eq!(tree.camera.zoom, before.zoom, "scrolling over display buttons must not zoom the tree");
        assert!(tree.camera.center.distance(before.center) < 0.01);
    }

    #[test]
    fn sequential_generation_never_drifts_existing_cards_in_any_display_mode() {
        for mode in [TreeDisplayMode::All, TreeDisplayMode::EmphasizeBranches, TreeDisplayMode::BranchesOnly] {
            let ctx = egui::Context::default();
            let mut tree = TreeViewState::default();
            tree.configure_mode(mode, false);
            let mut document = fixture();
            let mut thumbnails = ThumbnailCache::new();
            input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
            let center = tree.camera.center;
            let root_screen = tree.camera.to_screen(tree.all_positions[&0], tree.last_area);
            let count = if mode == TreeDisplayMode::EmphasizeBranches { 110 } else { 40 };
            for _ in 0..count {
                append_step(&mut document);
                input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
                assert!(tree.camera.center.distance(center) < 0.01,
                    "mode={mode:?}, current={}, camera={:?}, initial={center:?}", document.current, tree.camera.center);
                let current_root = tree.camera.to_screen(tree.all_positions[&0], tree.last_area);
                assert!(current_root.distance(root_screen) < 0.01,
                    "existing root drifted while a new leaf was added");
            }
        }
    }

    #[test]
    fn branch_relayout_during_pan_rebases_drag_origin_without_next_frame_jump() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        let mut document = fixture();
        let mut thumbnails = ThumbnailCache::new();
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
        let pointer = tree.last_area.left_top() + Vec2::new(25.0, 260.0);
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![
            egui::Event::PointerMoved(pointer), pointer_kind(pointer,egui::PointerButton::Middle,true),
        ]);
        assert!(tree.pan_drag.is_some());
        let previous = tree.camera.center;
        let selected = document.current;
        document.jump(0).unwrap();
        append_step(&mut document);
        document.jump(selected).unwrap();
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![egui::Event::PointerMoved(pointer)]);
        let rebalanced = tree.camera.center;
        assert!(rebalanced.distance(previous) > 1.0, "fixture must actually rebalance a branch");
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![egui::Event::PointerMoved(pointer)]);
        assert!(tree.camera.center.distance(rebalanced) < 0.01);
    }

    #[test]
    fn minimap_drag_keeps_camera_still_when_tree_bounds_grow_under_stationary_pointer() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        let mut document = fixture();
        let mut thumbnails = ThumbnailCache::new();
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
        let pointer = tree.last_minimap.right_bottom() - Vec2::splat(18.0);
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![
            egui::Event::PointerMoved(pointer), pointer_button(pointer, true),
        ]);
        assert!(tree.mini_drag_offset.is_some());
        let before = tree.camera.center;
        append_step(&mut document);
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![egui::Event::PointerMoved(pointer)]);
        assert!(tree.camera.center.distance(before) < 0.01);
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![egui::Event::PointerMoved(pointer)]);
        assert!(tree.camera.center.distance(before) < 0.01);
    }

    #[test]
    fn camera_pan_is_unbounded_in_all_four_directions() {
        let mut camera = Camera::default();
        camera.pan(Vec2::new(1000.0, 900.0));
        assert_eq!(camera.center, Pos2::new(-1000.0, -900.0));
        camera.pan(Vec2::new(-2500.0, -2100.0));
        assert_eq!(camera.center, Pos2::new(1500.0, 1200.0));
    }
    #[test]
    fn zoom_retains_world_point_under_pointer() {
        let area = Rect::from_min_size(Pos2::new(30.0, 50.0), Vec2::new(1000.0, 700.0));
        let mut camera = Camera {
            center: Pos2::new(-100.0, 800.0),
            zoom: 0.8,
        };
        let anchor = Pos2::new(800.0, 200.0);
        let before = camera.to_world(anchor, area);
        camera.zoom_at(anchor, area, 1.7);
        assert!(camera.to_world(anchor, area).distance(before) < 0.001);
    }

    #[test]
    fn extreme_scroll_and_repeated_zoom_keep_viewport_finite() {
        let area = Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 700.0));
        let mut camera = Camera::default();
        for _ in 0..500 {
            camera.zoom_at(area.center(), area, camera.zoom * 0.01);
        }
        camera.pan(Vec2::new(f32::MAX, -f32::MAX));
        camera.pan(Vec2::new(f32::INFINITY, f32::NAN));
        let viewport = camera.viewport(area);
        assert!(viewport.min.x.is_finite() && viewport.min.y.is_finite());
        assert!(viewport.max.x.is_finite() && viewport.max.y.is_finite());
        assert!(viewport.width().is_finite() && viewport.height().is_finite());
        assert!(camera.zoom >= 0.000_001);
        let mini = MiniMap::new(
            Rect::from_min_size(Pos2::ZERO, Vec2::new(208.0, 142.0)),
            Rect::from_min_size(Pos2::ZERO, Vec2::new(2000.0, 190_000_000.0)),
        );
        let mapped = mini.viewport(viewport);
        assert!(mapped.min.x.is_finite() && mapped.max.y.is_finite());
    }
    #[test]
    fn minimap_mapping_round_trips_and_navigation_does_not_pan_main_view() {
        let rect = Rect::from_min_size(Pos2::new(580.0, 10.0), Vec2::new(200.0, 140.0));
        let bounds = Rect::from_min_max(Pos2::new(-800.0, 0.0), Pos2::new(800.0, 1800.0));
        let map = MiniMap::new(rect, bounds);
        let point = Pos2::new(400.0, 1500.0);
        assert!(map.to_world(map.to_screen(point)).distance(point) < 0.001);
        let mut tree = TreeViewState::default();
        tree.navigate(
            InputSnapshot {
                pointer: Some(map.to_screen(point)),
                primary_pressed: true,
                primary_down: true,
                pinch_zoom: 1.0,
                ..Default::default()
            },
            Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0)),
            rect,
            map,
        );
        assert!(tree.camera.center.distance(point) < 0.001);
        assert!(tree.pan_drag.is_none());
        assert!(tree.mini_drag_offset.is_some());
    }
    #[test]
    fn compact_map_retains_all_hidden_steps_without_making_current_visible() {
        let input: Vec<TreeNode> = (0..8)
            .map(|id| TreeNode {
                id,
                parent: id.checked_sub(1),
            })
            .collect();
        let layout = build_layout(
            &input,
            LayoutOptions {
                compact: true,
                ..Default::default()
            },
        );
        assert!(layout.node(4).is_none());
        let positions = complete_positions(&input, &layout);
        assert_eq!(positions.len(), input.len());
        assert!(positions[&4].y > positions[&0].y && positions[&4].y < positions[&7].y);
        let visible = layout.nodes.iter().map(|node| node.id).collect();
        assert_eq!(nearest_visible(4, &input, &visible), Some(0));
    }

    #[test]
    fn middle_pointer_drag_pans_but_ordinary_wheel_zooms_at_pointer() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        let document = fixture();
        let mut thumbnails = ThumbnailCache::new();
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
        let start = tree.last_area.left_top() + Vec2::new(25.0, 260.0);
        let before = tree.camera.center;
        input_frame(
            &ctx,
            &mut tree,
            &document,
            &mut thumbnails,
            vec![
                egui::Event::PointerMoved(start),
                pointer_kind(start,egui::PointerButton::Middle,true),
            ],
        );
        let end = start + Vec2::new(80.0, 50.0);
        input_frame(
            &ctx,
            &mut tree,
            &document,
            &mut thumbnails,
            vec![egui::Event::PointerMoved(end)],
        );
        assert!((tree.camera.center - before - Vec2::new(-80.0, -50.0)).length() < 0.01);
        input_frame(
            &ctx,
            &mut tree,
            &document,
            &mut thumbnails,
            vec![pointer_kind(end,egui::PointerButton::Middle,false)],
        );
        let before_zoom = tree.camera.zoom;
        let before_anchor = tree.camera.to_world(end, tree.last_area);
        input_frame(
            &ctx,
            &mut tree,
            &document,
            &mut thumbnails,
            vec![egui::Event::Scroll(Vec2::new(-35.0, -60.0))],
        );
        assert!(tree.camera.zoom < before_zoom);
        assert!(tree.camera.to_world(end, tree.last_area).distance(before_anchor) < 0.01);
    }

    #[test]
    fn wheel_over_minimap_zooms_main_view_without_navigation_jump() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        let document = fixture();
        let mut thumbnails = ThumbnailCache::new();
        input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![]);
        let pointer=tree.last_minimap.center();
        let before=tree.camera.center;
        let zoom=tree.camera.zoom;
        assert!(input_frame(&ctx,&mut tree,&document,&mut thumbnails,vec![egui::Event::PointerMoved(pointer),egui::Event::Scroll(Vec2::new(0.0,60.0))]).is_none());
        assert!(tree.camera.zoom>zoom);
        assert!(tree.camera.center.distance(before)<0.01);
        assert!(tree.pan_drag.is_none());
    }

    #[test]
    fn raw_minimap_drag_is_navigation_only_without_jump_or_main_pan() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        let document = fixture();
        let mut thumbnails = ThumbnailCache::new();
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
        let mini = tree.last_minimap;
        let click = mini.right_bottom() - Vec2::splat(18.0);
        let before = tree.camera.center;
        assert!(input_frame(
            &ctx,
            &mut tree,
            &document,
            &mut thumbnails,
            vec![
                egui::Event::PointerMoved(click),
                pointer_button(click, true)
            ]
        )
        .is_none());
        assert!(tree.camera.center.distance(before) > 1.0);
        assert!(tree.pan_drag.is_none());
        let moved = click - Vec2::new(38.0, 35.0);
        assert!(input_frame(
            &ctx,
            &mut tree,
            &document,
            &mut thumbnails,
            vec![egui::Event::PointerMoved(moved)]
        )
        .is_none());
        assert!(tree.pan_drag.is_none());
        assert!(input_frame(
            &ctx,
            &mut tree,
            &document,
            &mut thumbnails,
            vec![pointer_button(moved, false)]
        )
        .is_none());
    }

    #[test]
    fn raw_zoom_event_keeps_pointer_anchor() {
        let ctx = egui::Context::default();
        let mut tree = TreeViewState::default();
        let document = fixture();
        let mut thumbnails = ThumbnailCache::new();
        input_frame(&ctx, &mut tree, &document, &mut thumbnails, vec![]);
        let anchor = tree.last_area.left_top() + Vec2::new(100.0, 220.0);
        let before = tree.camera.to_world(anchor, tree.last_area);
        input_frame(
            &ctx,
            &mut tree,
            &document,
            &mut thumbnails,
            vec![egui::Event::PointerMoved(anchor), egui::Event::Zoom(1.5)],
        );
        assert!((tree.camera.zoom - 1.5).abs() < 0.01);
        assert!(
            tree.camera
                .to_world(anchor, tree.last_area)
                .distance(before)
                < 0.01
        );
    }
}
