"""Compare edge-off vs edge-on on the same images/seed."""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

ROOT = Path(r"D:\DS_workspaceNew\DS_Studynew\ex\work\ab3")
SRC = Path(r"D:\DS_workspaceNew\DS_Studynew\ex\work\real-input")
IMAGES = ["anime_upper", "anime_mid"]

rows = []
for mode in ("edge-off", "edge-on"):
    df = pd.read_csv(ROOT / mode / "v1" / "data_part_1.csv")
    shapes = df[(df.shape_type >= 0) & (df.shape_type <= 1)]
    bent_mask = (shapes.e0.abs() > 1e-6) | (shapes.e1.abs() > 1e-6) | \
                (shapes.e2.abs() > 1e-6) | (shapes.e3.abs() > 1e-6)
    for name in IMAGES:
        group = shapes[shapes.image_name == name]
        gbent = bent_mask[shapes.image_name == name].sum()
        orig = np.array(Image.open(SRC / f"{name}.png").convert("RGB"), dtype=np.float32)
        rep = np.array(Image.open(ROOT / f"preview-{mode}" / f"{name}.png").convert("RGB"), dtype=np.float32)
        diff = orig - rep
        rows.append({
            "mode": mode, "image": name, "shapes": len(group),
            "bent": int(gbent), "bent_pct": 100.0 * gbent / max(1, len(group)),
            "rmse": float(np.sqrt((diff * diff).mean())),
            "mae": float(np.abs(diff).mean()),
        })

table = pd.DataFrame(rows)
print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print()
for name in IMAGES:
    off = table[(table["mode"] == "edge-off") & (table["image"] == name)].iloc[0]
    on = table[(table["mode"] == "edge-on") & (table["image"] == name)].iloc[0]
    print(f"{name}: shapes {int(off.shapes)} -> {int(on.shapes)} "
          f"({100*(on.shapes-off.shapes)/off.shapes:+.1f}%), "
          f"RMSE {off.rmse:.3f} -> {on.rmse:.3f} ({100*(on.rmse-off.rmse)/off.rmse:+.2f}%)")
