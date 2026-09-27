"""LinkedIn montage (1080 x 1350, 4:5): the System One idea, the two architectures, the VisA belt,
the headline charts, the day-0 lime line, and the decision. Captions are burned in because feed
video plays muted; every frame carries the condados.ai mark.

  uv run python scripts/linkedin_montage.py --assets <dir with fig-*.png> \
      --belt visa-belt.mp4 --limes limes-day0.mp4 out.mp4

The figures are the post's own SVGs rendered to PNG (post_figures.py, post_diagrams.py); the two
clips are the renders from render_visa_video.py and render_counting_video.py.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--assets", type=Path, required=True)
ap.add_argument("--belt", required=True)
ap.add_argument("--limes", required=True)
ap.add_argument("--fonts", type=Path,
                default=Path.home() / "condados-inspection/node_modules/@fontsource/inter/files")
a = ap.parse_args()

W, H, FPS = 1080, 1350, 30
BG, TXT, MUTED = (15, 18, 24), (232, 236, 243), (160, 168, 186)
GEM, PC, OK = (167, 139, 250), (56, 189, 248), (45, 212, 191)
F = lambda w, s: ImageFont.truetype(str(a.fonts / f"inter-latin-{w}-normal.woff"), s)
f_hero, f_label, f_cap, f_small, f_wm = F(700, 64), F(700, 40), F(400, 32), F(400, 26), F(600, 30)
f_code = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 31)
JEV = (245, 158, 11)
TOP, BOTTOM = 250, H - 90  # content area


def wrap(d, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def base(label=None, caption=None, label_col=TXT):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    if label:
        d.text((60, 90), label, font=f_label, fill=label_col, anchor="ls")
    if caption:
        y = 140
        for line in wrap(d, caption, f_cap, W - 120):
            d.text((60, y), line, font=f_cap, fill=MUTED, anchor="ls")
            y += 42
    return im, d


def mark(im):
    """The condados.ai mark, bottom-right, white at ~55% with a soft shadow."""
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    x, y = W - 40, H - 40
    d.text((x + 2, y + 2), "condados.ai", font=f_wm, fill=(0, 0, 0, 110), anchor="rs")
    d.text((x, y), "condados.ai", font=f_wm, fill=(255, 255, 255, 140), anchor="rs")
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")


def fade(im, t, dur, edge=0.3):
    """Fade in and out over `edge` seconds at the scene boundaries."""
    k = min(1.0, t / edge, (dur - t) / edge)
    if k >= 1:
        return im
    return Image.blend(Image.new("RGB", im.size, BG), im, max(0.0, k))


def place(im, pic, top=TOP, bottom=BOTTOM):
    """Fit `pic` into the content area, centred; returns its box."""
    box_w, box_h = W - 80, bottom - top
    s = min(box_w / pic.width, box_h / pic.height)
    pic = pic.resize((int(pic.width * s), int(pic.height * s)), Image.LANCZOS)
    x, y = (W - pic.width) // 2, top + (box_h - pic.height) // 2
    im.paste(pic, (x, y))
    return x, y, pic


def pan(im, pic, t, dur, height, top=TOP):
    """Show `pic` large enough to read on a phone and slide across it left to right, the order a
    diagram or a chart is read in. Holds the first and last half second."""
    s = height / pic.height
    p = pic.resize((int(pic.width * s), height), Image.LANCZOS)
    y = top + (BOTTOM - top - height) // 2
    if p.width <= W - 40:
        im.paste(p, ((W - p.width) // 2, y))
        return
    travel = p.width - (W - 40)
    k = min(1.0, max(0.0, (t - 0.5) / max(0.1, dur - 1.0)))
    k = 0.5 - 0.5 * np.cos(np.pi * k)  # ease in and out
    x0 = int(travel * k)
    im.paste(p.crop((x0, 0, x0 + W - 40, height)), (20, y))


def png(name):
    return Image.open(a.assets / f"{name}.png").convert("RGB")


class Clip:
    def __init__(self, path, start):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.start = start

    def frame(self, t):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, (self.start + t) * 1000)
        ok, bgr = self.cap.read()
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if ok else None


arch_vlm, arch_pc, kcurve, cover, pipe_line = (png(n) for n in (
    "fig-arch-vlm", "fig-arch-patchcore", "fig-k-curve", "fig-coverage", "fig-pipeline-line"))
belt, limes = Clip(a.belt, 3.0), Clip(a.limes, 3.5)


def s_title(t, dur):
    im, d = base()
    y = 430
    for line in ("A “System One” AI model", "on a production line"):
        d.text((W // 2, y), line, font=f_hero, fill=TXT, anchor="ms")
        y += 84
    for line, col in (("The idea behind TypeSafe's Jev: decide in one pass, no examples.", MUTED),
                      ("How many photos of good parts does a", MUTED),
                      ("dedicated detector need to beat it?", MUTED),
                      ("", MUTED), ("2,162 parts · 12 products · the VisA benchmark", GEM)):
        d.text((W // 2, y + 30), line, font=f_cap, fill=col, anchor="ms")
        y += 46
    return im


def s_two_readouts(t, dur):
    im, d = base("Two System One readouts", "Same photo, same question, one pass each. What does "
                 "the System One training add?", GEM)
    cards = [(JEV, "Jev-Omni", ["Gemma 4 12B", "+ fine-tuned weights", "+ a trained decision head", "",
                                "trained for System One", "open, not affiliated", "with TypeSafe"]),
             (GEM, "Gemma 4, one pass", ["the same base model,", "untouched", "", "read the probability",
                                        "of the answer \u201c1\u201d or \u201c2\u201d", "", "no training"])]
    for i, (col, title, lines) in enumerate(cards):
        if t < 0.4 + 0.8 * i:
            continue
        x0 = 60 + i * 500
        d.rounded_rectangle((x0, 300, x0 + 460, 900), 24, fill=(26, 32, 48), outline=col, width=4)
        d.text((x0 + 36, 380), title, font=f_label, fill=col, anchor="ls")
        y = 460
        for line in lines:
            d.text((x0 + 36, y), line, font=f_cap, fill=TXT, anchor="ls")
            y += 52
    if t > 2.6:
        for j, line in enumerate(wrap(d, "On VisA photos the untouched Gemma 4 scored higher, 82.9 against 81.1. "
                                         "Jev-Omni's card reports decision, audio and video benchmarks, none on "
                                         "images or inspection. The lime demo uses Gemma 4.", f_cap, W - 120)):
            d.text((60, 990 + 44 * j), line, font=f_cap, fill=MUTED, anchor="ls")
    return im


def s_code(t, dur):
    im, d = base("In Python", "Simplified; the full, tested code is in the post and the repo.", TXT)
    blocks = [
        (JEV, "# Jev-Omni: its own predict()", [
            "result = classifier.predict(",
            "    state=\"Inspection photo of a cashew nut.\",",
            "    question=\"Good, or is a part defective?\",",
            "    options=[\"all good\", \"defective\"],",
            "    media=\"cashew.jpg\", modality=\"image\")",
            "# {'all good': 0.39, 'defective': 0.61}"]),
        (GEM, "# Gemma 4: one pass, read two probabilities", [
            "inputs = processor.apply_chat_template(",
            "    [photo, question + \"1. all good 2. defective\"])",
            "logits = model(**inputs).logits[0, -1]",
            "p = softmax(logits[[id(\"1\"), id(\"2\")]])",
            "# {'all good': 0.029, 'defective': 0.971}"]),
    ]
    y = 290
    for i, (col, head, lines) in enumerate(blocks):
        if t < 0.3 + 2.5 * i:
            continue
        d.rounded_rectangle((30, y - 44, W - 30, y + 50 * len(lines) + 30), 16, fill=(10, 12, 18))
        d.text((50, y), head, font=f_code, fill=col, anchor="ls")
        for j, line in enumerate(lines):
            c = OK if line.startswith("#") else TXT
            d.text((50, y + 50 * (j + 1)), line, font=f_code, fill=c, anchor="ls")
        y += 50 * len(lines) + 150
    return im


def s_diagram(pic, label, caption, col, height=820):
    def f(t, dur):
        im, d = base(label, caption, col)
        pan(im, pic, t, dur, height)
        return im
    return f


def s_system_one(t, dur):
    im, d = base("What is a \u201cSystem One\u201d model?", None, GEM)
    y = 250
    blocks = [
        ("Chat models answer by writing, word by word.", TXT),
        ("A System One model answers a question with a decision: one probability per option, "
         "in a single pass, with no text.", TXT),
        ("The name borrows Kahneman's System 1, the fast, automatic kind of thinking.", MUTED),
        ("TypeSafe launched the idea with its model Jev on 15 Sep 2026, in early access.", MUTED),
        ("Jev-Omni is an open model built the same way on Gemma 4, for images too, and not "
         "affiliated with TypeSafe. That is the one tested here.", GEM),
    ]
    shown = int(t / 1.1) + 1  # one more line every 1.1 s
    for text, col in blocks[:shown]:
        for line in wrap(d, text, f_cap, W - 120):
            d.text((60, y), line, font=f_cap, fill=col, anchor="ls")
            y += 44
        y += 26
    return im


def s_clip(clip, label, caption, col, top=TOP):
    def f(t, dur):
        im, d = base(label, caption, col)
        fr = clip.frame(t)
        if fr is not None:
            place(im, fr, top)
        return im
    return f


def s_close(t, dur):
    im, d = base()
    y = 360
    lines = [(f_label, "Use a System One model on day one.", TXT),
             (f_label, "Photograph good parts from shift one.", TXT),
             (f_label, "Set its threshold on labelled parts", TXT),
             (f_label, "before it decides alone.", TXT), (f_cap, "", TXT),
             (f_cap, "The base model it was built on, read the same", MUTED),
             (f_cap, "one-pass way, scored higher: 82.9 against 81.1.", MUTED), (f_cap, "", TXT),
             (f_small, "Full write-up, code and every number:", MUTED),
             (f_small, "condados.ai/blog/patchcore-vs-zero-shot-vlm-visa", OK)]
    for font, text, col in lines:
        d.text((W // 2, y), text, font=font, fill=col, anchor="ms")
        y += font.size + 22
    return im


scenes = [
    (3.5, s_title),
    (7.0, s_system_one),
    (6.0, s_two_readouts),
    (8.0, s_code),
    (8.0, s_diagram(arch_vlm, "Day 0 · a System One readout",
                    "Both readouts run the photo and the question through Gemma 4 once and return one probability "
                    "per answer. No examples needed.", GEM)),
    (7.0, s_diagram(arch_pc, "Day 1 · PatchCore", "Learns what good parts look like from a few photos, then flags "
                    "any patch that looks like none of them.", PC)),
    (14.0, s_clip(belt, "Same parts, both inspectors", "Real VisA test parts. Verdicts read from the saved scores; "
                  "magenta marks PatchCore's unusual patches.", TXT)),
    (7.0, s_diagram(kcurve, "How many good photos until it wins?", "1 to 8 against Jev-Omni, 4 to 16 against "
                    "Gemma 4, depending on photo size. Macro AUROC: 50 is a coin flip.", TXT, height=900)),
    (6.0, s_diagram(cover, "What it means on the line", "Share of parts decided without a person, with at most 5% "
                    "of defects let through on a line with 1% defective.", OK, height=620)),
    (5.0, s_diagram(pipe_line, "Day 0 on a real lime line", "Detect, track, crop, ask the model with a one-sentence "
                    "spec, decide, count.", GEM, height=760)),
    (12.0, s_clip(limes, "No labels, one sentence of spec", "Gemma 4 ranks the limes by how far they are from "
                  "export grade. The cuts were set by eye: an illustration, not a measurement.", GEM)),
    (6.0, s_close),
]

a.out.parent.mkdir(parents=True, exist_ok=True)
ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "slow",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], stdin=subprocess.PIPE)
total = 0
for dur, fn in scenes:
    for i in range(int(dur * FPS)):
        t = i / FPS
        ff.stdin.write(mark(fade(fn(t, dur), t, dur)).tobytes())
    total += dur
ff.stdin.close()
ff.wait()
print(f"{total:.1f} s -> {a.out}")
