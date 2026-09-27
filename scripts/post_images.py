"""Raster images for the post: the PatchCore heat-map figure and the cover background.

Heat is drawn in magenta: most VisA products (cashew, macaroni, candles, fryum) are yellow or
orange, and an amber overlay disappears on them.
  uv run python scripts/post_images.py <site public/blog/<slug> dir> [--maps results/demo/visa-maps]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from jev_inspection import visa

HEAT = np.array([255, 43, 214], np.float32)  # magenta
GT = (45, 212, 191)  # ground-truth outline, teal

ap = argparse.ArgumentParser()
ap.add_argument("public", type=Path)
ap.add_argument("--maps", type=Path, default=Path("results/maps-k16-s0-256"),
                help="PatchCore k=16 seed 0 maps at 256 px, saved by the main run (--maps-for 16 0)")
ap.add_argument("--cover", type=Path, default=Path("output/cover-bg.png"))
a = ap.parse_args()
root = visa.DEFAULT_ROOT
IMG = "cashew/Data/Images/Anomaly/086.JPG"
MASK = "cashew/Data/Masks/Anomaly/086.png"


def overlay(img: Image.Image, m: np.ndarray, start=0.55, span=0.35, alpha=0.85) -> Image.Image:
    """Blend magenta where the map is in the top part of this image's range."""
    hv = (m - m.min()) / max(1e-6, m.max() - m.min())
    hv = np.asarray(Image.fromarray((hv * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
    al = (np.clip((hv - start) / span, 0, 1) * alpha)[..., None]
    return Image.fromarray((np.asarray(img).astype(np.float32) * (1 - al) + HEAT * al).astype(np.uint8))


m = np.load(a.maps / "cashew_k16_s0" / (IMG.replace("/", "__") + ".npy")).astype(np.float32)
im = Image.open(root / IMG).convert("RGB")

# Figure: the photo, and the same photo with the heat and the ground-truth outline.
side = 560
h = int(side * im.height / im.width)
photo = im.resize((side, h), Image.LANCZOS)
heat = overlay(photo, m)
gt = Image.open(root / MASK).convert("L").resize((side, h), Image.NEAREST)
edge = gt.point(lambda v: 255 if v > 0 else 0).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
heat.paste(Image.new("RGB", (side, h), GT), (0, 0), edge)
fig = Image.new("RGB", (side * 2 + 16, h), (19, 23, 34))
fig.paste(photo, (0, 0))
fig.paste(heat, (side + 16, 0))
fig.save(a.public / "cashew-scratch-heatmap.webp", quality=82)

# Cover background: the heat-mapped nut on the right, the eight good cashews of the k = 8 draw below.
W, H = 1200, 675
bg = Image.new("RGB", (W, H), (15, 18, 24))
s = min(im.size) * 0.78
cx, cy = im.width * 0.5, im.height * 0.52
box = (int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))
sq = 290  # right-hand column only: the title runs to about x = 860
full = overlay(im, m)
nut = full.crop(box).resize((sq, sq), Image.LANCZOS)
rmask = Image.new("L", (sq, sq), 0)
ImageDraw.Draw(rmask).rounded_rectangle((0, 0, sq - 1, sq - 1), 18, fill=255)
x0, y0 = W - sq - 40, 70
bg.paste(nut, (x0, y0), rmask)
draw = json.loads(visa.FROZEN.read_text())["patchcore_subsets"]["cashew"]["0"]["8"]
t, gap = 32, 5
rx, ry = x0 + (sq - (8 * t + 7 * gap)) // 2, y0 + sq + 16
for i, p in enumerate(draw):
    g = Image.open(root / p).convert("RGB")
    ss = min(g.size) * 0.8
    g = g.crop((int(g.width / 2 - ss / 2), int(g.height / 2 - ss / 2),
                int(g.width / 2 + ss / 2), int(g.height / 2 + ss / 2))).resize((t, t), Image.LANCZOS)
    mk = Image.new("L", (t, t), 0)
    ImageDraw.Draw(mk).rounded_rectangle((0, 0, t - 1, t - 1), 6, fill=255)
    bg.paste(g, (rx + i * (t + gap), ry), mk)
a.cover.parent.mkdir(parents=True, exist_ok=True)
bg.save(a.cover)
print("wrote", a.public / "cashew-scratch-heatmap.webp", "and", a.cover)
