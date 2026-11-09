"""Build a labeled side-by-side A/B sheet so the two variants cannot be mistaken."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"D:\DS_workspaceNew\DS_Studynew\ex\work\ab3")
PAIRS = {
    "anime_upper": ("15.381", "14.168"),
    "anime_mid": ("18.003", "16.336"),
}

try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 17)
    SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
except OSError:
    FONT = SMALL = ImageFont.load_default()


def build(name):
    off = Image.open(ROOT / "preview-edge-off" / f"{name}.png").convert("RGB")
    on = Image.open(ROOT / "preview-edge-on" / f"{name}.png").convert("RGB")
    w, h = off.size
    pad, bar = 12, 46
    sheet = Image.new("RGB", (w * 2 + pad * 3, h + bar + pad * 2), (250, 250, 250))
    sheet.paste(off, (pad, bar + pad))
    sheet.paste(on, (w + pad * 2, bar + pad))
    draw = ImageDraw.Draw(sheet)
    rmse_off, rmse_on = PAIRS[name]
    draw.text((pad + 4, 10), f"edge OFF (straight rims)", font=FONT, fill=(150, 40, 40))
    draw.text((pad + 4, 26), f"RMSE {rmse_off}", font=SMALL, fill=(90, 90, 90))
    draw.text((w + pad * 2 + 4, 10), "edge ON (bezier rims)", font=FONT, fill=(30, 110, 50))
    draw.text((w + pad * 2 + 4, 26), f"RMSE {rmse_on}   (-8%)", font=SMALL, fill=(90, 90, 90))
    out = ROOT / f"A-B_{name}.png"
    sheet.save(out)
    print("wrote", out)


for key in PAIRS:
    build(key)
