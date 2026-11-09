use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};

const SV_HEIGHT: f32 = 108.0;
const HUE_HEIGHT: f32 = 16.0;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct HsvResponse {
    pub changed: bool,
    pub finished: bool,
}

#[derive(Clone, Copy)]
enum DragTarget {
    Hue,
    SaturationValue,
}

pub(super) struct HsvPicker {
    hue: f32,
    saturation: f32,
    value: f32,
    last_rgb: Option<[u8; 3]>,
    active: Option<DragTarget>,
    #[cfg(test)]
    pub(super) hue_rect: Rect,
    #[cfg(test)]
    pub(super) sv_rect: Rect,
}

impl Default for HsvPicker {
    fn default() -> Self {
        Self {
            hue: 0.0,
            saturation: 0.0,
            value: 1.0,
            last_rgb: None,
            active: None,
            #[cfg(test)]
            hue_rect: Rect::NOTHING,
            #[cfg(test)]
            sv_rect: Rect::NOTHING,
        }
    }
}

impl HsvPicker {
    pub(super) fn cancel_interaction(&mut self) {
        self.active = None;
    }

    /// Wrap each picker in its own `ui.push_id` when showing several together.
    pub(super) fn show(&mut self, ui: &mut egui::Ui, rgb: &mut [u8; 3], width: f32) -> HsvResponse {
        self.sync_external_color(*rgb);
        let width = if width.is_finite() {
            width.max(1.0)
        } else {
            160.0
        }
        .min(ui.available_width().max(1.0));
        let (hue_rect, hue_response) =
            ui.allocate_exact_size(Vec2::new(width, HUE_HEIGHT), Sense::click_and_drag());
        ui.add_space(4.0);
        let (sv_rect, sv_response) =
            ui.allocate_exact_size(Vec2::new(width, SV_HEIGHT), Sense::click_and_drag());
        #[cfg(test)]
        {
            self.hue_rect = hue_rect;
            self.sv_rect = sv_rect;
        }

        let mut result = HsvResponse::default();
        if ui.is_enabled() {
            let (pressed, down, released, pointer) = ui.input(|input| {
                (
                    input.pointer.button_pressed(egui::PointerButton::Primary),
                    input.pointer.button_down(egui::PointerButton::Primary),
                    input.pointer.button_released(egui::PointerButton::Primary),
                    input.pointer.interact_pos(),
                )
            });
            // Capture from the initial press, without egui's drag-distance delay.
            // `clicked` also covers a press and release delivered in one frame.
            if self.active.is_none() {
                if (pressed && hue_response.is_pointer_button_down_on()) || hue_response.clicked() {
                    self.active = Some(DragTarget::Hue);
                } else if (pressed && sv_response.is_pointer_button_down_on())
                    || sv_response.clicked()
                {
                    self.active = Some(DragTarget::SaturationValue);
                }
            }
            if let Some(target) = self.active {
                if let Some(pointer) = pointer {
                    match target {
                        DragTarget::Hue => {
                            self.hue =
                                ((pointer.x - hue_rect.left()) / hue_rect.width()).clamp(0.0, 1.0);
                        }
                        DragTarget::SaturationValue => {
                            self.saturation =
                                ((pointer.x - sv_rect.left()) / sv_rect.width()).clamp(0.0, 1.0);
                            self.value = (1.0 - (pointer.y - sv_rect.top()) / sv_rect.height())
                                .clamp(0.0, 1.0);
                        }
                    }
                    let selected = hsv_to_rgb(self.hue, self.saturation, self.value);
                    result.changed = selected != *rgb;
                    *rgb = selected;
                    self.last_rgb = Some(selected);
                }
                if released || !down {
                    // Report the completed gesture even if its final RGB is unchanged.
                    result.finished = true;
                    self.active = None;
                }
            }
        } else {
            self.active = None;
        }

        paint_hue(ui.painter(), hue_rect, self.hue);
        paint_sv(ui.painter(), sv_rect, self.hue, self.saturation, self.value);
        result
    }

    fn sync_external_color(&mut self, rgb: [u8; 3]) {
        if self.last_rgb == Some(rgb) {
            return;
        }
        let (hue, saturation, value) = rgb_to_hsv(rgb);
        if let Some(hue) = hue {
            self.hue = hue;
        }
        // At black both hue and saturation are indeterminate. Keep the user's
        // selection so increasing brightness restores the selected color.
        if value > 0.0 {
            self.saturation = saturation;
        }
        self.value = value;
        self.last_rgb = Some(rgb);
    }
}

fn paint_hue(painter: &egui::Painter, rect: Rect, selected: f32) {
    let mut mesh = egui::Mesh::default();
    for step in 0..=6 {
        let fraction = step as f32 / 6.0;
        let x = egui::lerp(rect.x_range(), fraction);
        let color = color32(hsv_to_rgb(fraction, 1.0, 1.0));
        mesh.colored_vertex(Pos2::new(x, rect.top()), color);
        mesh.colored_vertex(Pos2::new(x, rect.bottom()), color);
        if step > 0 {
            let index = (step * 2) as u32;
            mesh.add_triangle(index - 2, index - 1, index);
            mesh.add_triangle(index, index - 1, index + 1);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
    painter.rect_stroke(rect, 2.0, Stroke::new(0.75, Color32::from_gray(125)));
    let marker = Rect::from_center_size(
        Pos2::new(egui::lerp(rect.x_range(), selected), rect.center().y),
        Vec2::new(5.0, rect.height() + 3.0),
    );
    painter.rect_stroke(marker, 1.0, Stroke::new(3.0, Color32::from_gray(35)));
    painter.rect_stroke(marker, 1.0, Stroke::new(1.4, Color32::WHITE));
}

fn paint_sv(painter: &egui::Painter, rect: Rect, hue: f32, saturation: f32, value: f32) {
    // Subdivision preserves the bilinear HSV gradient; a single two-triangle
    // quad would create a visibly incorrect diagonal across the square.
    const STEPS: usize = 16;
    let mut mesh = egui::Mesh::default();
    for row in 0..=STEPS {
        let y = row as f32 / STEPS as f32;
        for column in 0..=STEPS {
            let x = column as f32 / STEPS as f32;
            mesh.colored_vertex(
                Pos2::new(egui::lerp(rect.x_range(), x), egui::lerp(rect.y_range(), y)),
                color32(hsv_to_rgb(hue, x, 1.0 - y)),
            );
        }
    }
    for row in 0..STEPS {
        for column in 0..STEPS {
            let top_left = (row * (STEPS + 1) + column) as u32;
            let bottom_left = top_left + STEPS as u32 + 1;
            mesh.add_triangle(top_left, top_left + 1, bottom_left);
            mesh.add_triangle(top_left + 1, bottom_left + 1, bottom_left);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
    painter.rect_stroke(rect, 2.0, Stroke::new(0.75, Color32::from_gray(125)));
    let center = Pos2::new(
        egui::lerp(rect.x_range(), saturation),
        egui::lerp(rect.y_range(), 1.0 - value),
    );
    painter.circle_stroke(center, 4.5, Stroke::new(3.0, Color32::from_gray(35)));
    painter.circle_stroke(center, 4.5, Stroke::new(1.4, Color32::WHITE));
}

fn color32(rgb: [u8; 3]) -> Color32 {
    Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

// Work directly on sRGB channels, matching conventional desktop HSV pickers.
fn rgb_to_hsv(rgb: [u8; 3]) -> (Option<f32>, f32, f32) {
    let [r, g, b] = rgb.map(|channel| channel as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let range = max - min;
    let hue = if range <= f32::EPSILON {
        None
    } else {
        let sector = if max == r {
            ((g - b) / range).rem_euclid(6.0)
        } else if max == g {
            (b - r) / range + 2.0
        } else {
            (r - g) / range + 4.0
        };
        Some(sector / 6.0)
    };
    (hue, if max > 0.0 { range / max } else { 0.0 }, max)
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [u8; 3] {
    let chroma = value.clamp(0.0, 1.0) * saturation.clamp(0.0, 1.0);
    let sector = hue.rem_euclid(1.0) * 6.0;
    let secondary = chroma * (1.0 - ((sector % 2.0) - 1.0).abs());
    let (r, g, b) = match sector as u8 {
        0 => (chroma, secondary, 0.0),
        1 => (secondary, chroma, 0.0),
        2 => (0.0, chroma, secondary),
        3 => (0.0, secondary, chroma),
        4 => (secondary, 0.0, chroma),
        _ => (chroma, 0.0, secondary),
    };
    let white = value.clamp(0.0, 1.0) - chroma;
    [r, g, b].map(|channel| ((channel + white) * 255.0).round().clamp(0.0, 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(
        ctx: &egui::Context,
        picker: &mut HsvPicker,
        rgb: &mut [u8; 3],
        enabled: bool,
        events: Vec<egui::Event>,
    ) -> HsvResponse {
        let mut response = HsvResponse::default();
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(400.0, 400.0))),
                events,
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_enabled_ui(enabled, |ui| {
                        ui.push_id("test_picker", |ui| {
                            response = picker.show(ui, rgb, 160.0);
                        });
                    });
                });
            },
        );
        response
    }

    fn pointer_button(pos: Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        }
    }

    #[test]
    fn srgb_hsv_roundtrips_primary_and_intermediate_colors() {
        for r in [0, 1, 32, 64, 128, 200, 255] {
            for g in [0, 1, 32, 64, 128, 200, 255] {
                for b in [0, 1, 32, 64, 128, 200, 255] {
                    let rgb = [r, g, b];
                    let (h, s, v) = rgb_to_hsv(rgb);
                    assert_eq!(hsv_to_rgb(h.unwrap_or(0.0), s, v), rgb);
                }
            }
        }
        assert_eq!(hsv_to_rgb(0.0, 0.5, 128.0 / 255.0), [128, 64, 64]);
        assert_eq!(hsv_to_rgb(1.0, 1.0, 1.0), [255, 0, 0]);
    }

    #[test]
    fn external_achromatic_colors_preserve_selected_hue() {
        let mut picker = HsvPicker::default();
        picker.sync_external_color([0, 0, 255]);
        let blue = picker.hue;
        for rgb in [[255, 255, 255], [96, 96, 96], [0, 0, 0]] {
            picker.sync_external_color(rgb);
            assert_eq!(picker.hue, blue);
        }
        picker.sync_external_color([255, 0, 0]);
        picker.sync_external_color([0, 0, 0]);
        assert_eq!(picker.saturation, 1.0);
    }

    #[test]
    fn real_pointer_drag_sets_hue_and_sv_and_finishes_only_on_release() {
        let ctx = egui::Context::default();
        let mut picker = HsvPicker::default();
        let mut rgb = [255, 0, 0];
        frame(&ctx, &mut picker, &mut rgb, true, vec![]);
        let hue_point = Pos2::new(
            picker.hue_rect.left() + picker.hue_rect.width() / 3.0,
            picker.hue_rect.center().y,
        );
        let press = frame(
            &ctx,
            &mut picker,
            &mut rgb,
            true,
            vec![
                egui::Event::PointerMoved(hue_point),
                pointer_button(hue_point, true),
            ],
        );
        assert!(press.changed);
        assert!(!press.finished);
        assert_eq!(rgb, [0, 255, 0]);
        let release = frame(
            &ctx,
            &mut picker,
            &mut rgb,
            true,
            vec![pointer_button(hue_point, false)],
        );
        assert!(release.finished);
        assert!(!release.changed);
        assert!(!frame(&ctx, &mut picker, &mut rgb, true, vec![]).finished);

        let sv_point = Pos2::new(
            picker.sv_rect.left() + picker.sv_rect.width() * 0.75,
            picker.sv_rect.center().y,
        );
        let press = frame(
            &ctx,
            &mut picker,
            &mut rgb,
            true,
            vec![
                egui::Event::PointerMoved(sv_point),
                pointer_button(sv_point, true),
            ],
        );
        assert!(press.changed);
        assert_eq!(rgb, [32, 128, 32]);
        let outside = picker.sv_rect.right_top() + Vec2::new(100.0, -50.0);
        let dragged = frame(
            &ctx,
            &mut picker,
            &mut rgb,
            true,
            vec![egui::Event::PointerMoved(outside)],
        );
        assert!(dragged.changed);
        assert!(!dragged.finished);
        assert_eq!(rgb, [0, 255, 0]);
        assert!(
            frame(
                &ctx,
                &mut picker,
                &mut rgb,
                true,
                vec![pointer_button(outside, false)]
            )
            .finished
        );
    }

    #[test]
    fn white_hue_change_survives_idle_and_same_color_click_still_finishes() {
        let ctx = egui::Context::default();
        let mut picker = HsvPicker::default();
        let mut rgb = [255, 255, 255];
        frame(&ctx, &mut picker, &mut rgb, true, vec![]);
        let point = Pos2::new(
            picker.hue_rect.left() + picker.hue_rect.width() * 2.0 / 3.0,
            picker.hue_rect.center().y,
        );
        let press = frame(
            &ctx,
            &mut picker,
            &mut rgb,
            true,
            vec![
                egui::Event::PointerMoved(point),
                pointer_button(point, true),
            ],
        );
        assert!(!press.changed);
        assert!(!press.finished);
        assert!(
            frame(
                &ctx,
                &mut picker,
                &mut rgb,
                true,
                vec![pointer_button(point, false)]
            )
            .finished
        );
        frame(&ctx, &mut picker, &mut rgb, true, vec![]);
        assert!((picker.hue - 2.0 / 3.0).abs() < 0.001);
        let sv = picker.sv_rect.right_top() + Vec2::new(-1.0, 1.0);
        assert!(
            frame(
                &ctx,
                &mut picker,
                &mut rgb,
                true,
                vec![egui::Event::PointerMoved(sv), pointer_button(sv, true)]
            )
            .changed
        );
        assert!(rgb[2] > 240 && rgb[0] < 5 && rgb[1] < 5);
    }

    #[test]
    fn disabled_picker_does_not_change_color_or_finish_a_gesture() {
        let ctx = egui::Context::default();
        let mut picker = HsvPicker::default();
        let mut rgb = [200, 32, 32];
        frame(&ctx, &mut picker, &mut rgb, false, vec![]);
        let point = picker.sv_rect.center();
        for pressed in [true, false] {
            let response = frame(
                &ctx,
                &mut picker,
                &mut rgb,
                false,
                vec![
                    egui::Event::PointerMoved(point),
                    pointer_button(point, pressed),
                ],
            );
            assert!(!response.changed && !response.finished);
            assert_eq!(rgb, [200, 32, 32]);
        }
    }
}
