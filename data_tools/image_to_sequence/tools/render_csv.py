#!/usr/bin/env python3
"""Replay current POP2 packed CSV without source images or model weights.

SDF coverage, alpha=0.5, and per-operation integer RGB truncation match the
training GPU generator. This is a visual replay, not the strict training validator.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

FIELDS = {"image_name", "cx", "cy", "w", "h", "shape_type", "theta", "r", "g", "b", "is_region_marker", "stage", "region_id"}

def read_groups(version_dir: Path, limit: int = 16, image_name: str | None = None) -> OrderedDict:
    parts = sorted(version_dir.glob("data_part_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not parts:
        raise FileNotFoundError(f"No data_part_N.csv in {version_dir}")
    groups = OrderedDict()
    done = False
    for part in parts:
        with part.open("r", encoding="utf-8-sig", newline="") as f:
            rows = csv.DictReader(f)
            if not FIELDS.issubset(rows.fieldnames or []):
                raise ValueError(f"{part.name}: current 13-field POP2 header required")
            for index, row in enumerate(rows, start=2):
                name = row["image_name"]
                if image_name and name != image_name:
                    continue
                if name not in groups and len(groups) >= limit:
                    done = True
                    break
                if not name or name in {".", ".."} or any(c in name for c in "/\\:\0"):
                    raise ValueError("Unsafe or empty image_name")
                group = groups.setdefault(name, {"background": None, "shapes": [], "marker_count": 0, "records": 0})
                context = f"{part.name}:{index} ({name})"
                values = {k: float(row[k]) for k in ("cx", "cy", "w", "h", "theta", "r", "g", "b")}
                for edge_key in ("e0", "e1", "e2", "e3"):
                    raw = row[edge_key] if edge_key in row else 0.0
                    values[edge_key] = float(raw) if raw == raw else 0.0
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError(f"{context}: nonfinite numeric field")
                kind = int(row["shape_type"])
                marker = int(row["is_region_marker"])
                region = int(row["region_id"])
                stage = row["stage"]
                if stage not in {"global", "coarse", "fine"}:
                    raise ValueError(f"{context}: unknown stage")
                maximum = {"global": 1, "coarse": 16, "fine": 64}[stage]
                if not (region == -1 if stage == "global" else 0 <= region < maximum):
                    raise ValueError(f"{context}: invalid region_id")
                rgb = tuple(int(values[k]) for k in ("r", "g", "b"))
                if any(not 0 <= c <= 255 for c in rgb):
                    raise ValueError(f"{context}: RGB outside 0..255")
                if kind == -1:
                    if group["records"] or marker:
                        raise ValueError(f"{context}: background must be unique and first")
                    group["background"] = rgb
                else:
                    if group["background"] is None:
                        raise ValueError(f"{context}: missing first background row")
                    if kind in (2, 3, 4):
                        if marker != 1:
                            raise ValueError(f"{context}: marker flag mismatch")
                        group["marker_count"] += 1
                    elif kind in (0, 1):
                        if marker or values["w"] <= 0 or values["h"] <= 0:
                            raise ValueError(f"{context}: invalid ordinary shape")
                        group["shapes"].append({**values, "shape_type": kind, "rgb": rgb})
                    else:
                        raise ValueError(f"{context}: unsupported shape type {kind}")
                group["records"] += 1
        if done:
            break
    if not groups:
        raise ValueError("No selected images found")
    return groups

def wgsl_smoothstep_coverage(distance: np.ndarray) -> np.ndarray:
    # smoothstep(-0.5, 0.5, d): normalized t is simply clamp(d + 0.5).
    t = np.clip(distance + np.float32(0.5), 0.0, 1.0).astype(np.float32)
    return np.float32(1.0) - t * t * (np.float32(3.0) - np.float32(2.0) * t)


def apply_shape(
    canvas: np.ndarray, shape: dict[str, Any], alpha: np.float32
) -> None:
    height, width, _ = canvas.shape
    cx = np.float32(shape["cx"])
    cy = np.float32(shape["cy"])
    hw = np.float32(shape["w"]) / np.float32(2.0)
    hh = np.float32(shape["h"]) / np.float32(2.0)
    theta = np.float32(shape["theta"])
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))

    e0 = np.float32(shape.get("e0", 0.0))
    e1 = np.float32(shape.get("e1", 0.0))
    e2 = np.float32(shape.get("e2", 0.0))
    e3 = np.float32(shape.get("e3", 0.0))

    if shape["shape_type"] == 0:
        extent_x = np.float32(abs(cos_t) * hw + abs(sin_t) * hh)
        extent_y = np.float32(abs(sin_t) * hw + abs(cos_t) * hh)
        extent_x += np.float32(0.5 * max(abs(e0), abs(e2)) * hw)
        extent_y += np.float32(0.5 * max(abs(e1), abs(e3)) * hh)
    else:
        hw_cos = np.float32(hw * cos_t)
        hh_sin = np.float32(hh * sin_t)
        hw_sin = np.float32(hw * sin_t)
        hh_cos = np.float32(hh * cos_t)
        extent_x = np.float32(np.sqrt(hw_cos * hw_cos + hh_sin * hh_sin))
        extent_y = np.float32(np.sqrt(hw_sin * hw_sin + hh_cos * hh_cos))

    min_x = max(0, math.floor(float(cx - extent_x - np.float32(1.0))))
    max_x = min(width - 1, math.ceil(float(cx + extent_x + np.float32(1.0))))
    min_y = max(0, math.floor(float(cy - extent_y - np.float32(1.0))))
    max_y = min(height - 1, math.ceil(float(cy + extent_y + np.float32(1.0))))
    if min_x > max_x or min_y > max_y:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float32)
    ys = np.arange(min_y, max_y + 1, dtype=np.float32)
    dx = xs[None, :] - cx
    dy = ys[:, None] - cy
    local_x = dx * cos_t + dy * sin_t
    local_y = -dx * sin_t + dy * cos_t

    if shape["shape_type"] == 0:
        if abs(e0) + abs(e1) + abs(e2) + abs(e3) < np.float32(1e-6):
            qx = np.abs(local_x) - hw
            qy = np.abs(local_y) - hh
            outside = np.sqrt(
                np.maximum(qx, np.float32(0.0)) ** 2
                + np.maximum(qy, np.float32(0.0)) ** 2
            )
            distance = outside + np.minimum(np.maximum(qx, qy), np.float32(0.0))
        else:
            # Bezier-bent rims collapse to parabolas; closed-form signed distance.
            qy2 = np.clip((local_y / hh) ** 2, 0.0, np.float32(1.0))
            qx2 = np.clip((local_x / hw) ** 2, 0.0, np.float32(1.0))
            d_right = local_x - hw * (1.0 + np.float32(0.5) * e0 * (1.0 - qy2))
            d_left = -local_x - hw * (1.0 + np.float32(0.5) * e2 * (1.0 - qy2))
            d_top = local_y - hh * (1.0 + np.float32(0.5) * e1 * (1.0 - qx2))
            d_bottom = -local_y - hh * (1.0 + np.float32(0.5) * e3 * (1.0 - qx2))
            distance = np.maximum(
                np.maximum(d_right, d_left), np.maximum(d_top, d_bottom)
            )
    else:
        inv_hw2 = np.float32(1.0) / max(
            np.float32(hw * hw), np.float32(0.0001)
        )
        inv_hh2 = np.float32(1.0) / max(
            np.float32(hh * hh), np.float32(0.0001)
        )
        f = local_x * local_x * inv_hw2 + local_y * local_y * inv_hh2 - 1.0
        gx = np.float32(2.0) * local_x * inv_hw2
        gy = np.float32(2.0) * local_y * inv_hh2
        gradient = np.sqrt(gx * gx + gy * gy)
        distance = f.copy()
        np.divide(
            f,
            gradient,
            out=distance,
            where=gradient > np.float32(1e-5),
        )

    coverage = wgsl_smoothstep_coverage(distance.astype(np.float32))
    mask = coverage > np.float32(0.0)
    if not np.any(mask):
        return

    region = canvas[min_y : max_y + 1, min_x : max_x + 1]
    effective_alpha = alpha * coverage
    shape_rgb = np.asarray(shape["rgb"], dtype=np.float32)
    mixed = (
        shape_rgb[None, None, :] * effective_alpha[:, :, None]
        + region.astype(np.float32)
        * (np.float32(1.0) - effective_alpha[:, :, None])
    )
    if np.any(mixed < 0.0) or np.any(mixed > 255.0001):
        raise ValueError("Reconstructed RGB escaped the WGSL u32-compatible range")

    # WGSL's u32(f32) conversion truncates non-negative fractional values.
    converted = mixed.astype(np.uint32).astype(np.uint8)
    region[mask] = converted[mask]


def render(group: dict, max_primitives: int | None = None) -> np.ndarray:
    canvas = np.empty((512, 512, 3), dtype=np.uint8)
    canvas[:] = group["background"]
    shapes = group["shapes"] if max_primitives is None else group["shapes"][:max_primitives]
    for shape in shapes:
        apply_shape(canvas, shape, np.float32(0.5))
    return canvas

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--limit", type=int, default=16, help="At most this many image groups; defaults to a small preview")
    parser.add_argument("--image-name", default=None)
    parser.add_argument("--max-primitives", type=int, default=None)
    args = parser.parse_args()
    if args.limit <= 0 or args.max_primitives is not None and args.max_primitives < 0:
        parser.error("--limit must be positive and --max-primitives non-negative")
    groups = read_groups(args.csv_root / args.version, args.limit, args.image_name)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, group in groups.items():
        destination = args.output / f"{name}.png"
        Image.fromarray(render(group, args.max_primitives)).save(destination)
        summary.append({"image_name": name, "records": group["records"], "ordinary_shapes": len(group["shapes"]), "markers": group["marker_count"], "png": destination.name})
    (args.output / "replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Replayed {len(summary)} images into {args.output}")

if __name__ == "__main__":
    main()
