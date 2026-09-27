"""The VisA video: test parts on an indexing belt, judged by Gemma 4 zero-shot (day 0) and by
PatchCore with 16 good parts at 512 px (day 1), next to the ground truth.

Every verdict comes from the saved scores; thresholds are the 5% escape / 5% reject cuts fitted
on all test images of the product. Nothing is re-run here.
  uv run python scripts/render_visa_video.py out.mp4 --maps results/demo/visa-maps
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jev_inspection import score, visa

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--maps", type=Path, required=True)
ap.add_argument("--selection", type=Path, default=Path("scripts/visa_selection.txt"))
ap.add_argument("--results", type=Path, default=Path("results"))
ap.add_argument("--fonts", type=Path,
                default=Path.home() / "condados-inspection/node_modules/@fontsource/inter/files")
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()

W, H, FPS = 1280, 720, a.fps
BG, PANEL, GRID, TXT, MUTED = (15, 18, 24), (26, 32, 48), (44, 49, 63), (232, 236, 243), (151, 160, 181)
COL = {"accept": (45, 212, 191), "person": (151, 160, 181), "reject": (248, 113, 113)}
WORD = {"accept": "ACCEPT", "person": "SEND TO A PERSON", "reject": "REJECT"}
F = lambda w, s: ImageFont.truetype(str(a.fonts / f"inter-latin-{w}-normal.woff"), s)
f_title, f_big, f_mid, f_small, f_tiny = F(700, 40), F(700, 26), F(600, 19), F(400, 17), F(400, 14)

items = {i.image: i for i in visa.load() if i.split == "test"}
sel = [p.strip() for p in a.selection.read_text().split() if p.strip()]
vlm, _ = score.load_vlm([a.results / "jev_visa.jsonl", a.results / "base_visa.jsonl"])
pc = score.load_patchcore(a.results / "patchcore_wide_resnet50_2_r512.jsonl")["16"][0]


def cuts(obj, s):
    its = [i for i in items.values() if i.obj == obj]
    good = [s[i.image] for i in its if not i.is_anomaly]
    bad = [s[i.image] for i in its if i.is_anomaly]
    lo, hi = np.quantile(bad, 0.05, method="lower"), np.quantile(good, 0.95, method="higher")
    return (lo, hi) if lo <= hi else ((lo + hi) / 2,) * 2


def decide(v, t):
    return "accept" if v < t[0] else ("reject" if v > t[1] else "person")


# One tile per selected image, with its PatchCore map blended in as a second layer.
TW, TH = 300, 226
tiles = []
for p in sel:
    it = items[p]
    img = Image.open(visa.DEFAULT_ROOT / p).convert("RGB")
    img.thumbnail((TW, TH), Image.LANCZOS)
    tile = Image.new("RGB", (TW, TH), (8, 10, 14))
    ox, oy = (TW - img.width) // 2, (TH - img.height) // 2
    tile.paste(img, (ox, oy))
    m = np.load(a.maps / (p.replace("/", "__") + ".npy")).astype(np.float32)
    # Colour on the product's own threshold scale, so a good part stays mostly uncoloured.
    lo_m, hi_m = cuts(it.obj, pc)
    hv = np.clip((m - 0.95 * hi_m) / (0.2 * hi_m), 0, 1)
    hv = np.asarray(Image.fromarray((hv * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
    base = np.asarray(img).astype(np.float32)
    heat_rgb = np.array([255, 43, 214], np.float32)  # magenta: amber vanishes on yellow products
    al = (np.clip(hv / 0.5, 0, 1) * 0.85)[..., None]
    heat = Image.fromarray((base * (1 - al) + heat_rgb * al).astype(np.uint8))
    htile = tile.copy()
    htile.paste(heat, (ox, oy))
    g = vlm["B0"][p]
    tiles.append({
        "path": p, "obj": it.obj, "tile": tile, "heat": htile,
        "truth": "good" if not it.is_anomaly else "defect: " + ", ".join(it.types[:2]),
        "gemma": decide(g, cuts(it.obj, vlm["B0"])), "p_def": 1 / (1 + math.exp(-g)),
        "gcut": [1 / (1 + math.exp(-c)) for c in cuts(it.obj, vlm["B0"])],
        "pc": decide(pc[p], cuts(it.obj, pc)), "pc_score": pc[p], "pcut": list(cuts(it.obj, pc)),
        "is_bad": it.is_anomaly,
    })

MOVE, DWELL, SPACING, BELT_Y = 0.55, 1.35, 340, 330
INTRO, OUTRO = 3.0, 4.5
SEG = MOVE + DWELL
T_TOTAL = INTRO + SEG * len(tiles) + OUTRO
ease = lambda x: 0.5 - 0.5 * math.cos(math.pi * min(1, max(0, x)))


def pill(d, x, y, text, col, font, anchor="mm", pad=(14, 7)):
    b = d.textbbox((x, y), text, font=font, anchor=anchor)
    d.rounded_rectangle((b[0] - pad[0], b[1] - pad[1], b[2] + pad[0], b[3] + pad[1]), 10, fill=col)
    d.text((x, y), text, font=font, fill=(10, 12, 16), anchor=anchor)


def footer(d):
    d.text((24, H - 22), "Images: VisA (Zou et al., ECCV 2022), CC BY 4.0. Verdicts from saved scores; "
           "cuts at 5% escapes / 5% rejects per product.", font=f_tiny, fill=MUTED, anchor="ls")
    d.text((W - 24, H - 22), "condados.ai", font=f_tiny, fill=MUTED, anchor="rs")
    d.text((24, H - 42), "Cuts per product: accept below the first, reject above the second, a person in between. "
           "Magenta: PatchCore patches near or above the reject cut.", font=f_tiny, fill=MUTED, anchor="ls")


def belt_frame(t):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    k = int(t // SEG)
    phase = (t - k * SEG) / MOVE
    shift = (k + ease(phase)) * SPACING  # belt position, in pixels travelled
    # Belt and its moving slats.
    d.rectangle((0, BELT_Y - 20, W, BELT_Y + TH + 20), fill=PANEL)
    for x in range(-60, W + 60, 40):
        xx = x - (shift % 40)
        d.line((xx, BELT_Y - 20, xx + 10, BELT_Y + TH + 20), fill=(34, 40, 58), width=2)
    cx = W // 2
    d.rounded_rectangle((cx - TW // 2 - 14, BELT_Y - 34, cx + TW // 2 + 14, BELT_Y + TH + 34), 14,
                        outline=(99, 102, 241), width=3)
    d.text((cx, BELT_Y - 48), "INSPECTION", font=f_tiny, fill=(165, 180, 252), anchor="ms")
    dwelling = phase >= 1
    for j, tl in enumerate(tiles):
        x = cx + j * SPACING - shift + SPACING
        if x < -TW or x > W + TW:
            continue
        inspected = j < k or (j == k and dwelling)
        src = tl["heat"] if (j == k and dwelling and (t - k * SEG - MOVE) > 0.35) or j < k else tl["tile"]
        im.paste(src, (int(x - TW // 2), BELT_Y))
        if j < k:  # already judged: a small tag stays on the part as it leaves
            pill(d, x, BELT_Y + TH + 2, WORD[tl["pc"]].split()[0] if tl["pc"] != "person" else "PERSON",
                 COL[tl["pc"]], f_tiny, pad=(8, 4))
    if 0 <= k < len(tiles) and dwelling:
        tl = tiles[k]
        dt = t - k * SEG - MOVE
        # Day 0 card (left) and day 1 card (right), then the truth.
        for side, (hdr, sub, verdict, extra) in enumerate((
                ("DAY 0 · Gemma 4 12B, zero-shot", "no good parts seen", tl["gemma"],
                 f"P(defective) {tl['p_def']:.2f}   cuts {tl['gcut'][0]:.2f} / {tl['gcut'][1]:.2f}"),
                ("DAY 1 · PatchCore (512 px)", "16 good parts seen", tl["pc"],
                 f"distance {tl['pc_score']:.1f}   cuts {tl['pcut'][0]:.1f} / {tl['pcut'][1]:.1f}"))):
            x0 = 40 if side == 0 else W - 40 - 330
            if dt > 0.1 + 0.25 * side:
                d.rounded_rectangle((x0, 60, x0 + 330, 228), 14, fill=PANEL, outline=GRID, width=2)
                d.text((x0 + 18, 92), hdr, font=f_mid, fill=TXT, anchor="ls")
                d.text((x0 + 18, 116), sub, font=f_small, fill=MUTED, anchor="ls")
                pill(d, x0 + 165, 160, WORD[verdict], COL[verdict], f_big)
                d.text((x0 + 18, 212), extra, font=f_small, fill=MUTED, anchor="ls")
        if dt > 0.75:
            col = COL["reject"] if tl["is_bad"] else COL["accept"]
            d.text((cx, BELT_Y + TH + 70), f"Truth: {tl['truth']}", font=f_mid, fill=col, anchor="ms")
            d.text((cx, BELT_Y + TH + 96), tl["obj"], font=f_small, fill=MUTED, anchor="ms")
    footer(d)
    return im


def card(lines, t, dur):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    alpha = min(1, t / 0.4, (dur - t) / 0.4)
    y = H // 2 - 30 * len(lines)
    for font, text, col in lines:
        c = tuple(int(BG[i] + (col[i] - BG[i]) * alpha) for i in range(3))
        d.text((W // 2, y), text, font=font, fill=c, anchor="ms")
        y += font.size + 22
    footer(d)
    return im


intro = [(f_title, "How many good parts does inspection need?", TXT),
         (f_mid, "VisA test parts on a belt. Two inspectors look at each one:", MUTED),
         (f_mid, "day 0: Gemma 4 zero-shot, no good parts   ·   day 1: PatchCore, 16 good parts", MUTED)]
outro = [(f_title, "On all 2,162 VisA test images", TXT),
         (f_mid, "Gemma 4 zero-shot: 82.9 AUROC   ·   Jev-Omni: 81.1", MUTED),
         (f_mid, "PatchCore, 16 good parts at 512 px: 92.4", COL["accept"]),
         (f_mid, "PatchCore passes Jev-Omni with 1 good part at 512 px, 8 at 256 px", TXT)]

a.out.parent.mkdir(parents=True, exist_ok=True)
ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                       "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)],
                      stdin=subprocess.PIPE)
n = int(T_TOTAL * FPS)
for f in range(n):
    t = f / FPS
    if t < INTRO:
        im = card(intro, t, INTRO)
    elif t < INTRO + SEG * len(tiles):
        im = belt_frame(t - INTRO)
    else:
        im = card(outro, t - INTRO - SEG * len(tiles), OUTRO)
    ff.stdin.write(im.tobytes())
ff.stdin.close()
ff.wait()
json.dump([{k: v for k, v in tl.items() if k not in ("tile", "heat")} for tl in tiles],
          open(a.out.with_suffix(".json"), "w"), indent=1)
print(f"{n} frames, {T_TOTAL:.1f} s -> {a.out}")
