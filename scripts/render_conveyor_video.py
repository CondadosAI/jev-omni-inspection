"""The day-0 conveyor video: every tangerine is detected, tracked, cropped once and judged by
Gemma 4 12B zero-shot; boxes turn to the verdict once the fruit has been inspected.

There are no labels for this clip. The two cuts on P(defective) are fixed by hand and say so on
screen: on day 0 there is nothing to calibrate them with.
  uv run python scripts/render_conveyor_video.py clip.mp4 results/demo out.mp4
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("demo", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--lo", type=float, default=0.4, help="accept below this P(defective)")
ap.add_argument("--hi", type=float, default=0.6, help="reject above this P(defective)")
ap.add_argument("--slow", type=int, default=2, help="each source frame is shown this many times")
ap.add_argument("--fonts", type=Path,
                default=Path.home() / "condados-inspection/node_modules/@fontsource/inter/files")
a = ap.parse_args()

tr = json.loads((a.demo / "tracks.json").read_text())
vd = json.loads((a.demo / "verdicts.json").read_text())["verdicts"]
W, H = 1920, 1080
PANEL_W = 420
BG, PANEL, GRID, TXT, MUTED = (15, 18, 24), (26, 32, 48), (44, 49, 63), (232, 236, 243), (151, 160, 181)
COL = {"accept": (45, 212, 191), "person": (200, 205, 215), "reject": (248, 113, 113), "tracking": (120, 128, 150)}
WORD = {"accept": "OK", "person": "PERSON", "reject": "REJECT"}
F = lambda w, s: ImageFont.truetype(str(a.fonts / f"inter-latin-{w}-normal.woff"), s)
f_title, f_mid, f_small, f_tiny = F(700, 44), F(600, 28), F(400, 24), F(400, 20)


def verdict(tid):
    p = vd[str(tid)]["p_defective"]
    return "accept" if p < a.lo else ("reject" if p > a.hi else "person")


crops = {t: Image.open(a.demo / "crops" / f"{t}.png").convert("RGB") for t in vd}
first_inspected = {int(t): v["crop_frame"] for t, v in tr["tracks"].items()}

cap = cv2.VideoCapture(a.clip)
cap.set(cv2.CAP_PROP_POS_MSEC, tr["t0"] * 1000)
fps = tr["fps"]
a.out.parent.mkdir(parents=True, exist_ok=True)
ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-r", str(round(fps)), "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "slow",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], stdin=subprocess.PIPE)


def card(lines, n):
    for f in range(n):
        alpha = min(1, f / 12, (n - f) / 12)
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = H // 2 - 40 * len(lines)
        for font, text, col in lines:
            d.text((W // 2, y), text, font=font, fill=tuple(int(BG[i] + (col[i] - BG[i]) * alpha) for i in range(3)),
                   anchor="ms")
            y += font.size + 30
        foot(d)
        ff.stdin.write(im.tobytes())


def foot(d):
    d.text((28, H - 26), f"Illustration, not a measurement: this clip has no labels. Cuts at P(defective) "
           f"{a.lo} / {a.hi} set by hand. Footage: Thiago Zanutim Lucas, Pexels.", font=f_tiny, fill=MUTED, anchor="ls")
    d.text((W - 28, H - 26), "condados.ai", font=f_tiny, fill=MUTED, anchor="rs")


card([(f_title, "Day 0 on a real line: no good parts collected yet", TXT),
      (f_mid, "detect each fruit  ·  track it  ·  crop it once  ·  ask Gemma 4 zero-shot", MUTED),
      (f_mid, "\"Is everything in the photo good, or is at least one part defective?\"", MUTED)], int(3.2 * fps))

inspected_order = []
for fr in tr["frames"]:
    ok, bgr = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    vid_w = W - PANEL_W
    scale = vid_w / rgb.shape[1]
    small = cv2.resize(rgb, (vid_w, int(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    im = Image.new("RGB", (W, H), BG)
    oy = (H - small.shape[0]) // 2 - 30
    im.paste(Image.fromarray(small), (0, oy))
    d = ImageDraw.Draw(im)
    for b in fr["boxes"]:
        tid = b["id"]
        if str(tid) not in vd:
            continue
        x0, y0, x1, y1 = [v * scale for v in b["box"]]
        y0, y1 = y0 + oy, y1 + oy
        done = fr["frame"] >= first_inspected[tid]
        if done and tid not in inspected_order:
            inspected_order.append(tid)
        state = verdict(tid) if done else "tracking"
        d.rounded_rectangle((x0, y0, x1, y1), 10, outline=COL[state], width=4 if done else 2)
        if done:
            lab = f"{WORD[state]} {vd[str(tid)]['p_defective']:.2f}"
            tb = d.textbbox((x0 + 6, y0 - 6), lab, font=f_tiny, anchor="ls")
            d.rectangle((tb[0] - 5, tb[1] - 4, tb[2] + 5, tb[3] + 4), fill=COL[state])
            d.text((x0 + 6, y0 - 6), lab, font=f_tiny, fill=(10, 12, 16), anchor="ls")
    # Side panel: the crops Gemma 4 was shown, newest first.
    px = W - PANEL_W
    d.rectangle((px, 0, W, H), fill=PANEL)
    d.text((px + 24, 56), "What the model sees", font=f_mid, fill=TXT, anchor="ls")
    d.text((px + 24, 88), "one crop per fruit, P(defective)", font=f_tiny, fill=MUTED, anchor="ls")
    y = 116
    for tid in list(reversed(inspected_order))[:5]:
        c = crops[str(tid)].copy()
        c.thumbnail((150, 150))
        im.paste(c, (px + 24, y))
        st = verdict(tid)
        d.rounded_rectangle((px + 24, y, px + 24 + c.width, y + c.height), 8, outline=COL[st], width=3)
        d.text((px + 196, y + 50), WORD[st], font=f_mid, fill=COL[st], anchor="ls")
        d.text((px + 196, y + 86), f"P = {vd[str(tid)]['p_defective']:.2f}", font=f_small, fill=TXT, anchor="ls")
        y += 170
    d.text((28, 52), "Gemma 4 12B, zero-shot  ·  0 good parts seen", font=f_mid, fill=TXT, anchor="ls")
    foot(d)
    for _ in range(a.slow):
        ff.stdin.write(im.tobytes())

counts = {s: sum(verdict(t) == s for t in vd) for s in ("accept", "person", "reject")}
card([(f_title, f"{counts['reject']} of {len(vd)} fruit flagged, with no example of a good one", TXT),
      (f_mid, "Without labels you cannot tell whether the fruit or the cut is wrong.", MUTED),
      (f_mid, "Day 1: collect a few good parts, set the cut per product, then compare with PatchCore.", MUTED),
      (f_mid, "On VisA, PatchCore passes this zero-shot readout with 1 to 8 good parts.", TXT)],
     int(5 * fps))
ff.stdin.close()
ff.wait()
print(counts, "->", a.out)
