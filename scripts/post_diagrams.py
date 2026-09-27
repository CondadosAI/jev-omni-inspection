"""The two pipeline diagrams in the post (static, no data): the VisA belt video and the
day-0 lime line.
  python scripts/post_diagrams.py <out dir>
"""
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
BG, PANEL, GRID, TXT, MUTED = "#131722", "#1a2030", "#2c313f", "#e8ecf3", "#97a0b5"
GEM, PC, OK = "#a78bfa", "#38bdf8", "#2dd4bf"
FONT = "ui-sans-serif,system-ui,sans-serif"


def box(x, y, w, h, lines, col=GRID, num=None):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{PANEL}" stroke="{col}" stroke-width="1.6"/>']
    ty = y + 22
    if num is not None:
        s.append(f'<circle cx="{x + 16}" cy="{y + 17}" r="10" fill="{col}"/><text x="{x + 16}" y="{y + 21}" '
                 f'font-size="11" font-weight="700" fill="{BG}" text-anchor="middle">{num}</text>')
    for i, (t, size, c, wgt) in enumerate(lines):
        xx = x + (32 if num is not None and i == 0 else 12)
        s.append(f'<text x="{xx}" y="{ty}" font-size="{size}" fill="{c}" font-weight="{wgt}">{t}</text>')
        ty += size + 5
    return "".join(s)


def arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#ah)"/>'


DEFS = (f'<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L8,4 L0,8 z" fill="{MUTED}"/></marker></defs>')


def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{label}" '
            f'style="width:100%;max-width:{w}px;height:auto;font-family:{FONT}">'
            f'<rect width="{w}" height="{h}" rx="10" fill="{BG}"/>{DEFS}{body}</svg>')


def sub(t):
    return (t, 11, MUTED, 400)


def title(t):
    return (t, 12.5, TXT, 700)


# VisA belt video: one photo, two inspectors, the same kind of decision, then the truth.
W, H = 720, 330
b = [box(12, 124, 118, 82, [title("test photo"), sub("one VisA part,"), sub("never seen"), sub("before")], col=MUTED, num=1),
     f'<text x="150" y="30" font-size="12" fill="{GEM}" font-weight="700">Day 0 · Gemma 4 zero-shot, no good photos</text>',
     f'<text x="150" y="190" font-size="12" fill="{PC}" font-weight="700">Day 1 · PatchCore, 16 good photos of this product</text>']
rows = ((GEM, 42, [("photo + question", "good or defective?", "no examples given"),
                   ("a score", "log-odds: above 0", "leans defective"),
                   ("two cuts", "accept · person ·", "reject")]),
        (PC, 202, [("compare patches", "with 16 good photos", "of the same product"),
                   ("a score", "distance of its most", "unusual patch"),
                   ("two cuts", "accept · person · reject,", "and the magenta map")]))
xs, ws = [150, 324, 472], [160, 134, 152]
for col, y0, row in rows:
    for i, ((t, s1, s2), x, w) in enumerate(zip(row, xs, ws)):
        b.append(box(x, y0, w, 74, [title(t), sub(s1), sub(s2)], col=col, num=i + 2))
    for (x, w), nx in zip(zip(xs, ws), xs[1:]):
        b.append(arrow(x + w, y0 + 37, nx - 4, y0 + 37))
b += [arrow(130, 150, 146, 90), arrow(130, 180, 146, 236),
      box(636, 118, 76, 94, [title("truth"), sub("VisA label,"), sub("shown"), sub("last")], col=OK, num=5),
      arrow(624, 79, 656, 115), arrow(624, 239, 656, 215),
      f'<text x="12" y="{H - 16}" font-size="11" fill="{MUTED}">Cuts are set per product: let at most 5% of defective '
      f'parts through, reject at most 5% of good ones.</text>']
(out / "fig-pipeline-belt.svg").write_text(svg(W, H, "".join(b), (
    "Pipeline of the VisA belt video: one test photo goes to Gemma 4 with a question and to PatchCore with 16 good "
    "photos; each gives a score, two cuts turn it into accept, person or reject, and the VisA label is shown last.")))

# Day-0 lime line: detect, track, crop before the line, ask, decide, count.
W, H = 720, 320
steps = [("detect", ["RF-DETR draws a box", "round each lime"], PC),
         ("track", ["ByteTrack gives each", "lime one ID over time"], PC),
         ("crop, before the line", ["largest clear view,", "neighbours greyed out"], PC),
         ("ask", ["Gemma 4: the question", "plus a one-sentence spec"], GEM),
         ("decide", ["score below 6: OK", "above 10: reject", "in between: a person"], GEM),
         ("count", ["when it crosses the", "line, add 1 to its", "verdict's counter"], OK)]
pos = [(16, 40), (260, 40), (504, 40), (504, 186), (260, 186), (16, 186)]
b = [box(x, y, 200, 90, [title(t)] + [sub(s) for s in subs], col=col, num=i + 1)
     for i, ((t, subs, col), (x, y)) in enumerate(zip(steps, pos))]
b += [arrow(216, 85, 256, 85), arrow(460, 85, 500, 85), arrow(604, 130, 604, 182), arrow(504, 231, 464, 231),
      arrow(260, 231, 220, 231),
      f'<text x="16" y="28" font-size="12" fill="{MUTED}">every video frame</text>',
      f'<text x="16" y="{H - 16}" font-size="11" fill="{MUTED}">Steps 1 and 2 run on every frame; steps 3 to 5 once per '
      f'lime, before it reaches the line; step 6 when it crosses.</text>']
(out / "fig-pipeline-line.svg").write_text(svg(W, H, "".join(b), (
    "Pipeline of the lime-line video: detect each lime, track it, crop and isolate it before the counting line, ask "
    "Gemma 4 with a one-sentence spec, turn the score into OK, person or reject, and count it when it crosses the line.")))
print("wrote", out / "fig-pipeline-belt.svg", out / "fig-pipeline-line.svg")
