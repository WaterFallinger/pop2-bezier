//! Small, lazy canvas previews for immutable history states.
//! No model execution is involved; snapshots are reconstructed from document deltas.
use crate::document::{Document, Primitive, ShapeKind};
use eframe::egui::{self, TextureHandle, TextureId};
use std::collections::HashMap;

pub const PREVIEW_SIZE: usize = 96;
const CACHE_TARGET: usize = 256;

struct Preview {
    texture: TextureHandle,
    last_frame: u64,
}

#[derive(Default)]
pub struct ThumbnailCache {
    previews: HashMap<u64, Preview>,
    document_identity: usize,
    previous_nodes: usize,
}

impl ThumbnailCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn texture(&mut self, ctx: &egui::Context, doc: &Document, node: u64) -> Option<TextureId> {
        let identity = doc as *const Document as usize;
        if identity != self.document_identity || doc.nodes.len() < self.previous_nodes {
            self.previews.clear();
            self.document_identity = identity;
        }
        self.previous_nodes = doc.nodes.len();
        let frame = ctx.frame_nr();
        if let Some(preview) = self.previews.get_mut(&node) {
            preview.last_frame = frame;
            return Some(preview.texture.id());
        }
        let primitives = doc.materialize(node.try_into().ok()?).ok()?;
        let image = render_preview(&primitives, PREVIEW_SIZE);
        // Never free textures referenced earlier in the same frame. At tiny
        // zoom levels, a frame may legitimately need more than the soft limit.
        while self.previews.len() >= CACHE_TARGET {
            let oldest = self
                .previews
                .iter()
                .filter(|(_, p)| p.last_frame < frame)
                .min_by_key(|(_, p)| p.last_frame)
                .map(|(&id, _)| id);
            if let Some(id) = oldest {
                self.previews.remove(&id);
            } else {
                break;
            }
        }
        let texture = ctx.load_texture(
            format!("history-preview-{node}"),
            image,
            egui::TextureOptions::LINEAR,
        );
        let id = texture.id();
        self.previews.insert(
            node,
            Preview {
                texture,
                last_frame: frame,
            },
        );
        Some(id)
    }
}

/// Render fixed native source-over in RGB numerical space. Low resolution is
/// deliberate: this is a navigation preview, not an export or model input.
pub fn render_preview(primitives: &[Primitive], size: usize) -> egui::ColorImage {
    assert!(size > 0);
    let mut pixels = vec![[255.0f32; 3]; size * size];
    let step = 512.0 / size as f32;
    for p in primitives {
        let color = p.color.map(f32::from);
        if p.shape == ShapeKind::Background {
            pixels.fill(color);
            continue;
        }
        if p.width <= 0.0 || p.height <= 0.0 {
            continue;
        }
        let (sin, cos) = p.angle_deg.to_radians().sin_cos();
        let hw = p.width * 0.5;
        let hh = p.height * 0.5;
        let (ex, ey) = match p.shape {
            ShapeKind::Ellipse => ((hw * cos).hypot(hh * sin), (hw * sin).hypot(hh * cos)),
            _ => (
                hw * cos.abs() + hh * sin.abs(),
                hw * sin.abs() + hh * cos.abs(),
            ),
        };
        let x0 = ((p.x - ex) / step).floor().clamp(0.0, size as f32) as usize;
        let x1 = ((p.x + ex) / step).ceil().clamp(0.0, size as f32) as usize;
        let y0 = ((p.y - ey) / step).floor().clamp(0.0, size as f32) as usize;
        let y1 = ((p.y + ey) / step).ceil().clamp(0.0, size as f32) as usize;
        for y in y0..y1 {
            for x in x0..x1 {
                let dx = (x as f32 + 0.5) * step - p.x;
                let dy = (y as f32 + 0.5) * step - p.y;
                let lx = dx * cos + dy * sin;
                let ly = -dx * sin + dy * cos;
                let inside = match p.shape {
                    ShapeKind::Ellipse => (lx / hw).powi(2) + (ly / hh).powi(2) <= 1.0,
                    _ => lx.abs() <= hw && ly.abs() <= hh,
                };
                if inside {
                    let dest = &mut pixels[y * size + x];
                    for c in 0..3 {
                        dest[c] = (dest[c] + color[c]) * 0.5;
                    }
                }
            }
        }
    }
    egui::ColorImage {
        size: [size, size],
        pixels: pixels
            .into_iter()
            .map(|p| {
                egui::Color32::from_rgb(p[0].round() as u8, p[1].round() as u8, p[2].round() as u8)
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::OperationSource;

    #[test]
    fn preview_uses_opaque_background_and_native_half_alpha_layer_order() {
        let bg = Primitive::background([20, 40, 100]);
        let fg = Primitive::new(
            ShapeKind::Rectangle,
            256.0,
            256.0,
            256.0,
            256.0,
            [220, 80, 0],
        );
        let image = render_preview(&[bg.clone(), fg.clone()], 16);
        assert_eq!(image[(0, 0)], egui::Color32::from_rgb(20, 40, 100));
        assert_eq!(image[(8, 8)], egui::Color32::from_rgb(120, 60, 50));
        let reversed = render_preview(&[fg, bg], 16);
        assert_eq!(reversed[(8, 8)], egui::Color32::from_rgb(20, 40, 100));
    }

    #[test]
    fn rotated_ellipse_preview_has_geometry_not_its_bounding_box() {
        let mut ellipse = Primitive::new(ShapeKind::Ellipse, 256.0, 256.0, 256.0, 64.0, [0, 0, 0]);
        ellipse.angle_deg = 45.0;
        let image = render_preview(&[ellipse], 32);
        assert_ne!(image[(20, 20)], egui::Color32::WHITE);
        assert_eq!(image[(20, 12)], egui::Color32::WHITE);
    }

    #[test]
    fn previews_reconstruct_each_branch_without_using_current_canvas() {
        let mut doc = Document::new();
        let blue = doc
            .commit_canvas(
                "blue",
                OperationSource::User,
                vec![Primitive::background([0, 0, 200])],
            )
            .unwrap();
        doc.jump(0).unwrap();
        let red = doc
            .commit_canvas(
                "red",
                OperationSource::User,
                vec![Primitive::background([200, 0, 0])],
            )
            .unwrap();
        let blue_preview = render_preview(&doc.materialize(blue).unwrap(), 8);
        let red_preview = render_preview(&doc.materialize(red).unwrap(), 8);
        assert_ne!(blue_preview.pixels, red_preview.pixels);
        assert_eq!(blue_preview[(3, 3)], egui::Color32::from_rgb(0, 0, 200));
        assert_eq!(doc.current, red);
    }
}
