"""Day-0 line demo (vertical): every item is tracked, judged by Gemma 4 zero-shot before it reaches
the counting line, and counted as OK / to a person / rejected when it crosses.

The clip has no labels. The score is Gemma 4's log-odds of "defective" with a written product
spec; the two cuts on it are set by eye on this clip and say so on screen.
  uv run python scripts/render_counting_video.py clip.mp4 demo_dir out.mp4 --verdicts verdicts_spec.json
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("demo", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--verdicts", default="verdicts_spec.json")
ap.add_argument("--lo", type=float, default=6.0, help="accept below this log-odds")
ap.add_argument("--hi", type=float, default=10.0, help="reject above this log-odds")
ap.add_argument("--slow", type=int, default=2, help="each processed frame is shown this many times")
ap.add_argument("--credit", default="Footage: Comercial GB, Pexels 32953325.")
ap.add_argument("--fonts", type=Path,
                default=Path.home() / "condados-inspection/node_modules/@fontsource/inter/files")
a = ap.parse_args()

tr = json.loads((a.demo / "tracks.json").read_text())
vj = json.loads((a.demo / a.verdicts).read_text())
vd = vj["verdicts"]
W, H = tr["size"]
FPS_OUT = round(tr["fps"])
LINE_Y = tr["line_y"]
BG, PANEL, TXT, MUTED = (15, 18, 24), (15, 18, 24, 215), (232, 236, 243), (170, 178, 196)
COL = {"accept": (45, 212, 191), "person": (226, 232, 240), "reject": (248, 113, 113), "tracking": (140, 148, 170)}
WORD = {"accept": "OK", "person": "PERSON", "reject": "REJECT"}
F = lambda w, s: ImageFont.truetype(str(a.fonts / f"inter-latin-{w}-normal.woff"), s)
f_title, f_big, f_mid, f_small, f_tiny = F(700, 46), F(700, 84), F(600, 32), F(400, 28), F(400, 22)


def verdict(tid):
    z = vd[str(tid)]["logodds"]
    return "accept" if z < a.lo else ("reject" if z > a.hi else "person")


tracks = {int(k): v for k, v in tr["tracks"].items()}
crops = {int(k): Image.open(a.demo / "crops" / f"{k}.png").convert("RGB") for k in vd}

ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-r", str(FPS_OUT), "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "slow",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], stdin=subprocess.PIPE)


def foot(d, y=H - 30):
    d.text((W // 2, y - 30), "Illustration: this clip has no labels. Score = Gemma 4 log-odds of \"defective\"",
           font=f_tiny, fill=MUTED, anchor="ms")
    d.text((W // 2, y), f"with the written spec; cuts at {a.lo:g} / {a.hi:g} set by eye. {a.credit}",
           font=f_tiny, fill=MUTED, anchor="ms")


def card(lines, seconds):
    n = int(seconds * FPS_OUT)
    for f in range(n):
        alpha = min(1, f / 10, (n - f) / 10)
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = H // 2 - 70 * len(lines) // 2
        for font, text, col in lines:
            d.text((W // 2, y), text, font=font, fill=tuple(int(BG[i] + (col[i] - BG[i]) * alpha) for i in range(3)),
                   anchor="ms")
            y += font.size + 34
        foot(d)
        ff.stdin.write(im.tobytes())


card([(f_title, "Day 0 on a lime line", TXT),
      (f_mid, "no good examples collected yet", MUTED),
      (f_mid, "detect · track · crop · ask Gemma 4 · count", MUTED),
      (f_small, "\"Export grade requires a green skin;", MUTED),
      (f_small, "yellowing, spots or damage count as defects.\"", MUTED)], 3.5)

cap = cv2.VideoCapture(a.clip)
cap.set(cv2.CAP_PROP_POS_MSEC, tr["t0"] * 1000)
count = {"accept": 0, "person": 0, "reject": 0}
judged, flash, src = [], 0, 0
frames = tr["frames"]
fi = 0
while fi < len(frames):
    ok, bgr = cap.read()
    if not ok:
        break
    src += 1
    if (src - 1) % tr["stride"]:
        continue
    fr = frames[fi]
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for b in fr["boxes"]:
        tid = b["id"]
        if str(tid) not in vd:
            continue
        info = tracks[tid]
        done = fr["frame"] >= info["crop_frame"]
        if done and tid not in judged:
            judged.append(tid)
        if info["crossed"] == fr["frame"]:
            count[verdict(tid)] += 1
            flash = 8
        state = verdict(tid) if done else "tracking"
        x0, y0, x1, y1 = b["box"]
        d.rounded_rectangle((x0, y0, x1, y1), 8, outline=COL[state] + (255,), width=5 if done else 2)
        if done:
            tb = d.textbbox((x0, y0 - 8), WORD[state], font=f_tiny, anchor="ls")
            d.rectangle((tb[0] - 5, tb[1] - 4, tb[2] + 5, tb[3] + 4), fill=COL[state] + (255,))
            d.text((x0, y0 - 8), WORD[state], font=f_tiny, fill=(10, 12, 16, 255), anchor="ls")
    # Counting line.
    lc = (255, 255, 255, 255) if flash == 0 else (99, 102, 241, 255)
    for x in range(0, W, 40):
        d.line((x, LINE_Y, x + 24, LINE_Y), fill=lc, width=6)
    d.text((W - 20, LINE_Y - 14), "counting line", font=f_tiny, fill=lc, anchor="rs")
    flash = max(0, flash - 1)
    # Top panel: title and the three counters.
    d.rectangle((0, 0, W, 330), fill=PANEL)
    d.text((40, 70), "DAY 0 · Gemma 4 12B, zero-shot", font=f_mid, fill=TXT, anchor="ls")
    d.text((40, 108), "0 good examples · a one-sentence spec", font=f_small, fill=MUTED, anchor="ls")
    for j, s in enumerate(("accept", "person", "reject")):
        cx = 180 + j * 360
        d.text((cx, 250), str(count[s]), font=f_big, fill=COL[s] + (255,), anchor="ms")
        d.text((cx, 296), WORD[s] if s != "accept" else "OK", font=f_small, fill=MUTED, anchor="ms")
    # Bottom panel: the crops the model was shown, newest first.
    d.rectangle((0, H - 400, W, H), fill=PANEL)
    d.text((40, H - 350), "What the model sees: one crop per lime, before the line", font=f_small, fill=MUTED,
           anchor="ls")
    im = Image.alpha_composite(im, over).convert("RGB")
    d2 = ImageDraw.Draw(im)
    for j, tid in enumerate(list(reversed(judged))[:5]):
        c = crops[tid].copy()
        c.thumbnail((170, 170))
        x = 40 + j * 205
        im.paste(c, (x, H - 320))
        st = verdict(tid)
        d2.rounded_rectangle((x - 3, H - 323, x + c.width + 3, H - 320 + c.height + 3), 8, outline=COL[st], width=4)
        d2.text((x, H - 120), f"{WORD[st]}", font=f_small, fill=COL[st], anchor="ls")
        d2.text((x, H - 90), f"score {vd[str(tid)]['logodds']:.1f}", font=f_tiny, fill=MUTED, anchor="ls")
    foot(d2)
    for _ in range(a.slow):
        ff.stdin.write(im.tobytes())
    fi += 1

total = sum(count.values())
card([(f_title, f"{total} limes crossed the line", TXT),
      (f_mid, f"{count['accept']} OK  ·  {count['person']} to a person  ·  {count['reject']} rejected", MUTED),
      (f_small, "Day 0 gives a ranking you can read by eye,", TXT),
      (f_small, "not a threshold: the cuts need labelled parts.", TXT),
      (f_small, "On VisA, PatchCore passes this readout", MUTED),
      (f_small, "with 4 to 16 good parts.", MUTED)], 5)
ff.stdin.close()
ff.wait()
print(count, "->", a.out)
