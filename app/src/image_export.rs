//! CPU exports of the current native canvas; no UI overlays or model execution.
use crate::document::{Primitive, ShapeKind};
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use std::{
    io::{BufWriter, Write},
    path::Path,
};

pub const MAX_PNG_SIZE: u32 = 8192;
const CANVAS: f64 = 512.0;
const STRIP_ROWS: usize = 64;
const COVERAGE_SAMPLES: usize = 4;

struct Geometry {
    shape: ShapeKind,
    color: [f32; 3],
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    angle: f64,
    sin: f64,
    cos: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

fn geometry(primitives: &[Primitive], size: usize) -> Result<Vec<Geometry>, String> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(primitives.len())
        .map_err(|_| "系统内存不足，无法准备导出图元".to_owned())?;
    let scale = size as f64 / CANVAS;
    for (index, p) in primitives.iter().enumerate() {
        if ![p.x, p.y, p.width, p.height, p.angle_deg]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(format!(
                "第 {} 个图元含非有限坐标、尺寸或角度，无法导出",
                index + 1
            ));
        }
        if p.shape != ShapeKind::Background && (p.width < 0.0 || p.height < 0.0) {
            return Err(format!("第 {} 个图元的尺寸不能为负数", index + 1));
        }
        if p.shape != ShapeKind::Background && (p.width == 0.0 || p.height == 0.0) {
            continue;
        }
        let (x, y, width, height) = (p.x as f64, p.y as f64, p.width as f64, p.height as f64);
        let angle = (p.angle_deg as f64).rem_euclid(360.0);
        let (sin, cos) = angle.to_radians().sin_cos();
        let (hx, hy) = (width * 0.5, height * 0.5);
        let (ex, ey) = match p.shape {
            ShapeKind::Ellipse => ((hx * cos).hypot(hy * sin), (hx * sin).hypot(hy * cos)),
            _ => (
                hx * cos.abs() + hy * sin.abs(),
                hx * sin.abs() + hy * cos.abs(),
            ),
        };
        let low = |value: f64| (value * scale).floor().clamp(0.0, size as f64) as usize;
        let high = |value: f64| (value * scale).ceil().clamp(0.0, size as f64) as usize;
        result.push(Geometry {
            shape: p.shape,
            color: p.color.map(f32::from),
            x,
            y,
            width,
            height,
            angle,
            sin,
            cos,
            x0: low(x - ex),
            x1: high(x + ex),
            y0: low(y - ey),
            y1: high(y + ey),
        });
    }
    Ok(result)
}

fn inside(g: &Geometry, x: f64, y: f64) -> bool {
    match g.shape {
        ShapeKind::Ellipse => (x / (g.width * 0.5)).powi(2) + (y / (g.height * 0.5)).powi(2) <= 1.0,
        _ => x.abs() <= g.width * 0.5 && y.abs() <= g.height * 0.5,
    }
}

fn coverage(g: &Geometry, x: f64, y: f64, radius: f64, offsets: &[(f64, f64); 16]) -> f32 {
    let (hx, hy) = (g.width * 0.5, g.height * 0.5);
    let (ax, ay) = (x.abs(), y.abs());
    let (near_x, near_y) = ((ax - radius).max(0.0), (ay - radius).max(0.0));
    let (full, empty) = match g.shape {
        ShapeKind::Ellipse => (
            ((ax + radius) / hx).powi(2) + ((ay + radius) / hy).powi(2) <= 1.0,
            (near_x / hx).powi(2) + (near_y / hy).powi(2) > 1.0,
        ),
        _ => (
            ax + radius <= hx && ay + radius <= hy,
            near_x > hx || near_y > hy,
        ),
    };
    if full {
        return 1.0;
    }
    if empty {
        return 0.0;
    }
    offsets
        .iter()
        .filter(|&&(dx, dy)| inside(g, x + dx, y + dy))
        .count() as f32
        / 16.0
}

fn rasterize(shapes: &[Geometry], size: usize) -> Result<Vec<u8>, String> {
    let length = size
        .checked_mul(size)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "PNG像素数量超出可表示范围".to_owned())?;
    let strip_length = size
        .checked_mul(size.min(STRIP_ROWS))
        .ok_or_else(|| "PNG工作区长度溢出".to_owned())?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(length)
        .map_err(|_| "系统内存不足，无法分配目标分辨率的PNG画布".to_owned())?;
    pixels.resize(length, 255u8);
    // Keep only a bounded floating-point strip. Each output pixel is rounded
    // once after all layers, matching the native CPU preview's color math.
    let mut strip = Vec::new();
    strip
        .try_reserve_exact(strip_length)
        .map_err(|_| "系统内存不足，无法分配PNG工作区".to_owned())?;
    strip.resize(strip_length, [255f32; 3]);
    let step = CANVAS / size as f64;
    for start in (0..size).step_by(STRIP_ROWS) {
        let end = (start + STRIP_ROWS).min(size);
        let working = &mut strip[..(end - start) * size];
        working.fill([255.0; 3]);
        for g in shapes {
            if g.shape == ShapeKind::Background {
                working.fill(g.color);
                continue;
            }
            let (y0, y1) = (g.y0.max(start), g.y1.min(end));
            if g.x0 >= g.x1 || y0 >= y1 {
                continue;
            }
            let offsets: [(f64, f64); 16] = std::array::from_fn(|index| {
                let dx = ((index % COVERAGE_SAMPLES) as f64 + 0.5) / COVERAGE_SAMPLES as f64 - 0.5;
                let dy = ((index / COVERAGE_SAMPLES) as f64 + 0.5) / COVERAGE_SAMPLES as f64 - 0.5;
                (
                    step * (dx * g.cos + dy * g.sin),
                    step * (-dx * g.sin + dy * g.cos),
                )
            });
            let radius = step * 0.5 * (g.cos.abs() + g.sin.abs());
            for py in y0..y1 {
                let dx = (g.x0 as f64 + 0.5) * step - g.x;
                let dy = (py as f64 + 0.5) * step - g.y;
                let mut x = dx * g.cos + dy * g.sin;
                let mut y = -dx * g.sin + dy * g.cos;
                for px in g.x0..g.x1 {
                    let alpha = coverage(g, x, y, radius, &offsets) * 0.5;
                    if alpha > 0.0 {
                        let destination = &mut working[(py - start) * size + px];
                        for channel in 0..3 {
                            destination[channel] =
                                destination[channel] * (1.0 - alpha) + g.color[channel] * alpha;
                        }
                    }
                    x += step * g.cos;
                    y -= step * g.sin;
                }
            }
        }
        for (index, color) in working.iter().enumerate() {
            let target = (start * size + index) * 3;
            for channel in 0..3 {
                pixels[target + channel] = color[channel].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(pixels)
}

/// Export the current canvas directly at the requested resolution. The image
/// is opaque, uses a white base without a background primitive, and has no UI.
pub fn export_png(path: &Path, primitives: &[Primitive], size: u32) -> Result<(), String> {
    if !(1..=MAX_PNG_SIZE).contains(&size) {
        return Err(format!("PNG分辨率必须为1至{MAX_PNG_SIZE}像素"));
    }
    let shapes = geometry(primitives, size as usize)?;
    let pixels = rasterize(&shapes, size as usize)?;
    crate::atomic_file::write_atomic(path, |file| {
        let mut writer = BufWriter::new(file);
        PngEncoder::new(&mut writer)
            .write_image(&pixels, size, size, ColorType::Rgb8)
            .map_err(|error| format!("PNG编码失败：{error}"))?;
        writer
            .flush()
            .map_err(|error| format!("PNG写入失败：{error}"))
    })
}

/// Export editable native rectangles and ellipses; no embedded raster preview.
pub fn export_svg(path: &Path, primitives: &[Primitive]) -> Result<(), String> {
    let shapes = geometry(primitives, 512)?;
    crate::atomic_file::write_atomic(path, |file| {
        let mut writer = BufWriter::new(file);
        let write = (|| -> std::io::Result<()> {
            writeln!(writer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
            writeln!(writer, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"512\" height=\"512\" viewBox=\"0 0 512 512\" overflow=\"hidden\">")?;
            writeln!(writer, "<defs><clipPath id=\"canvas-clip\"><rect width=\"512\" height=\"512\"/></clipPath></defs>")?;
            writeln!(
                writer,
                "<g clip-path=\"url(#canvas-clip)\" stroke=\"none\">"
            )?;
            writeln!(
                writer,
                "<rect x=\"0\" y=\"0\" width=\"512\" height=\"512\" fill=\"#ffffff\"/>"
            )?;
            for g in &shapes {
                let [r, green, b] = g.color.map(|value| value as u8);
                let color = format!("#{r:02x}{green:02x}{b:02x}");
                if g.shape == ShapeKind::Background {
                    writeln!(
                        writer,
                        "<rect x=\"0\" y=\"0\" width=\"512\" height=\"512\" fill=\"{color}\"/>"
                    )?;
                    continue;
                }
                let transform = format!("rotate({} {} {})", g.angle, g.x, g.y);
                match g.shape {
                    ShapeKind::Ellipse => writeln!(writer, "<ellipse cx=\"{}\" cy=\"{}\" rx=\"{}\" ry=\"{}\" transform=\"{transform}\" fill=\"{color}\" fill-opacity=\"0.5\"/>", g.x, g.y, g.width * 0.5, g.height * 0.5)?,
                    _ => writeln!(writer, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" transform=\"{transform}\" fill=\"{color}\" fill-opacity=\"0.5\"/>", g.x - g.width * 0.5, g.y - g.height * 0.5, g.width, g.height)?,
                }
            }
            writeln!(writer, "</g>\n</svg>")?;
            writer.flush()
        })();
        write.map_err(|error| format!("SVG写入失败：{error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "painter-export-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn file(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            if let Ok(entries) = fs::read_dir(&self.0) {
                for entry in entries.flatten() {
                    if entry.file_type().is_ok_and(|kind| kind.is_file()) {
                        let _ = fs::remove_file(entry.path());
                    }
                }
            }
            let _ = fs::remove_dir(&self.0);
        }
    }
    fn rectangle(color: [u8; 3]) -> Primitive {
        Primitive::new(ShapeKind::Rectangle, 256., 256., 256., 256., color)
    }

    #[test]
    fn png_decodes_at_requested_size_with_white_base_and_native_background_order() {
        let dir = TestDirectory::new();
        let path = dir.file("canvas.png");
        export_png(&path, &[], 37).unwrap();
        let image = image::open(&path).unwrap().to_rgba8();
        assert_eq!(image.dimensions(), (37, 37));
        assert!(image.pixels().all(|p| p.0 == [255; 4]));
        let primitives = vec![
            Primitive::background([20, 40, 100]),
            rectangle([220, 80, 0]),
        ];
        export_png(&path, &primitives, 64).unwrap();
        let image = image::open(&path).unwrap().to_rgb8();
        assert_eq!(image.get_pixel(0, 0).0, [20, 40, 100]);
        assert_eq!(image.get_pixel(32, 32).0, [120, 60, 50]);
        export_png(
            &path,
            &[
                rectangle([220, 80, 0]),
                Primitive::background([20, 40, 100]),
            ],
            64,
        )
        .unwrap();
        assert!(image::open(&path)
            .unwrap()
            .to_rgb8()
            .pixels()
            .all(|p| p.0 == [20, 40, 100]));
    }
    #[test]
    fn overlapping_layers_round_only_once() {
        let dir = TestDirectory::new();
        let path = dir.file("layers.png");
        export_png(
            &path,
            &[
                Primitive::background([20; 3]),
                rectangle([221; 3]),
                rectangle([0; 3]),
            ],
            64,
        )
        .unwrap();
        assert_eq!(
            image::open(path).unwrap().to_rgb8().get_pixel(32, 32).0,
            [60; 3]
        );
    }
    #[test]
    fn rotated_native_geometry_is_clipped_without_strokes() {
        let dir = TestDirectory::new();
        let path = dir.file("rotated.png");
        let mut r = Primitive::new(ShapeKind::Rectangle, 256., 256., 256., 64., [0; 3]);
        r.angle_deg = 90.;
        export_png(&path, &[r], 512).unwrap();
        let image = image::open(&path).unwrap().to_rgb8();
        assert_eq!(image.get_pixel(256, 170).0, [128; 3]);
        assert_eq!(image.get_pixel(350, 256).0, [255; 3]);
        let mut e = Primitive::new(ShapeKind::Ellipse, 256., 256., 256., 64., [0; 3]);
        e.angle_deg = 45.;
        export_png(&path, &[e], 512).unwrap();
        let image = image::open(&path).unwrap().to_rgb8();
        assert_eq!(image.get_pixel(320, 320).0, [128; 3]);
        assert_eq!(image.get_pixel(320, 192).0, [255; 3]);
        let off = Primitive::new(ShapeKind::Rectangle, -10., 256., 40., 200., [0; 3]);
        export_png(&path, &[off], 512).unwrap();
        let image = image::open(&path).unwrap().to_rgb8();
        assert_eq!(image.get_pixel(2, 256).0, [128; 3]);
        assert_eq!(image.get_pixel(20, 256).0, [255; 3]);
    }
    #[test]
    fn invalid_geometry_and_resolution_keep_existing_file_unchanged() {
        let dir = TestDirectory::new();
        let path = dir.file("existing.png");
        fs::write(&path, b"original bytes").unwrap();
        for size in [0, 8193, u32::MAX] {
            assert!(export_png(&path, &[], size).is_err());
        }
        let mut invalid = rectangle([0; 3]);
        invalid.x = f32::NAN;
        assert!(export_png(&path, &[invalid.clone()], 16).is_err());
        assert!(export_svg(&path, &[invalid]).is_err());
        let mut negative = rectangle([0; 3]);
        negative.width = -1.;
        assert!(export_png(&path, &[negative], 16).is_err());
        assert_eq!(fs::read(path).unwrap(), b"original bytes");
    }
    #[test]
    fn svg_keeps_editable_rotated_shapes_and_exact_fixed_opacity() {
        let dir = TestDirectory::new();
        let path = dir.file("canvas.svg");
        let mut r = rectangle([30, 60, 90]);
        r.angle_deg = 30.;
        let mut e = Primitive::new(ShapeKind::Ellipse, 150., 160., 80., 60., [210, 160, 110]);
        e.angle_deg = 45.;
        let zero = Primitive::new(ShapeKind::Rectangle, 0., 0., 0., 30., [0; 3]);
        export_svg(&path, &[Primitive::background([20, 40, 100]), r, e, zero]).unwrap();
        let text = fs::read_to_string(path).unwrap();
        assert!(text.contains("viewBox=\"0 0 512 512\""));
        assert!(text.contains("clip-path=\"url(#canvas-clip)\""));
        assert_eq!(text.matches("fill-opacity=\"0.5\"").count(), 2);
        assert!(text.contains("rotate(30 256 256)"));
        assert!(text.contains("<ellipse cx=\"150\" cy=\"160\" rx=\"40\" ry=\"30\""));
        assert!(!text.contains("<image"));
        assert!(!text.contains("stroke-width"));
        assert!(text.ends_with("</svg>\n"));
    }
    #[test]
    fn failed_atomic_commit_preserves_destination_and_cleans_temporary_files() {
        let dir = TestDirectory::new();
        let target = dir.file("not-a-file");
        fs::create_dir(&target).unwrap();
        let marker = target.join("keep.txt");
        fs::write(&marker, b"keep").unwrap();
        assert!(export_png(&target, &[], 8).is_err());
        assert!(export_svg(&target, &[]).is_err());
        assert_eq!(fs::read(&marker).unwrap(), b"keep");
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
        fs::remove_file(marker).unwrap();
        fs::remove_dir(target).unwrap();
    }
    #[test]
    fn review_fixture_exports_png_and_svg_from_the_same_vector_canvas() {
        let dir = TestDirectory::new();
        let mut primitives = vec![
            Primitive::background([238, 244, 252]),
            Primitive::new(ShapeKind::Ellipse, 256., 270., 290., 330., [236, 164, 134]),
        ];
        let mut hair = Primitive::new(ShapeKind::Rectangle, 256., 132., 302., 112., [242, 186, 36]);
        hair.angle_deg = -12.;
        primitives.push(hair);
        for x in [199., 313.] {
            primitives.push(Primitive::new(
                ShapeKind::Ellipse,
                x,
                258.,
                54.,
                37.,
                [10, 72, 174],
            ));
            primitives.push(Primitive::new(
                ShapeKind::Ellipse,
                x + 3.,
                254.,
                13.,
                13.,
                [255; 3],
            ));
        }
        primitives.push(Primitive::new(
            ShapeKind::Ellipse,
            256.,
            348.,
            95.,
            43.,
            [146, 38, 54],
        ));
        let png = dir.file("sample.png");
        let svg = dir.file("sample.svg");
        export_png(&png, &primitives, 1024).unwrap();
        export_svg(&svg, &primitives).unwrap();
        assert_eq!(image::open(&png).unwrap().width(), 1024);
        if let Some(root) = std::env::var_os("PAINTER_EXPORT_SAMPLE_DIR") {
            let root = PathBuf::from(root);
            fs::create_dir_all(&root).unwrap();
            fs::copy(png, root.join("canvas_export_sample_1024.png")).unwrap();
            fs::copy(svg, root.join("canvas_export_sample.svg")).unwrap();
        }
    }
}
