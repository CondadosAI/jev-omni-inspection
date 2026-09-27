"""Inline SVG figures for the CondadosAI post, drawn from summary.json only.

Usage: uv run python scripts/post_figures.py <results dir> <out dir>
Palette and type follow the site's dark lab style.
"""
import json
import math
import sys
from pathlib import Path

res, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
S = json.loads((res / "summary.json").read_text())
S5 = json.loads((res / "summary_r512.json").read_text())  # PatchCore at 512 x 512, k = 1, 4, 16

BG, PANEL, GRID, TXT, MUTED = "#131722", "#1a2030", "#2c313f", "#e8ecf3", "#97a0b5"
PC, JEV, GEM, PC5 = "#2dd4bf", "#f59e0b", "#a78bfa", "#38bdf8"
FONT = "ui-sans-serif,system-ui,sans-serif"


def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{label}" '
            f'style="width:100%;max-width:{w}px;height:auto;font-family:{FONT}">'
            f'<rect width="{w}" height="{h}" rx="10" fill="{BG}"/>{body}</svg>')


def text(x, y, s, size=12, fill=TXT, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


# --- Figure: macro AUROC against the number of good parts ----------------------------------
def k_curve():
    W, H, L, R, T, B = 720, 400, 62, 150, 30, 58
    ks = ["1", "2", "4", "8", "16", "64", "all"]
    x0 = L + 26  # room left of k = 1 for the reference-pair markers
    xs = {k: x0 + i * (W - x0 - R - 20) / (len(ks) - 1) for i, k in enumerate(ks)}
    lo, hi = 76, 94
    y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    m = S["macro_auroc"]
    b = []
    for v in range(lo, hi + 1, 2):
        b.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>')
        b.append(text(L - 8, y(v) + 4, v, 11, MUTED, "end"))
    for k in ks:
        n = "all (449–904)" if k == "all" else k
        b.append(text(xs[k], H - B + 20, n, 11, MUTED, "middle"))
    b.append(text((L + W - R) / 2, H - 12, "good parts in PatchCore's memory bank (k)", 12, MUTED, "middle"))
    b.append(text(16, T + (H - T - B) / 2, "macro image AUROC",
                  12, MUTED, "middle").replace("<text", f'<text transform="rotate(-90 16 {T + (H - T - B) / 2:.1f})"'))
    # VLM bands: zero good parts, drawn across the whole axis.
    for key, col, lab in (("A0", JEV, "Jev-Omni"), ("B0", GEM, "Gemma 4")):
        p, (c0, c1) = 100 * m[key]["point"], [100 * v for v in m[key]["ci95"]]
        b.append(f'<rect x="{L}" y="{y(c1):.1f}" width="{W - R - L}" height="{y(c0) - y(c1):.1f}" '
                 f'fill="{col}" opacity="0.13"/>')
        b.append(f'<line x1="{L}" x2="{W - R}" y1="{y(p):.1f}" y2="{y(p):.1f}" stroke="{col}" '
                 f'stroke-width="1.6" stroke-dasharray="6 4"/>')
        b.append(text(W - R + 8, y(p) + (4 if key == "B0" else 10), f"{lab} {p:.1f}", 11, col))
    # PatchCore curve with interval bars.
    pts = [(xs[k], 100 * m[f"C{k}"]["point"], [100 * v for v in m[f"C{k}"]["ci95"]]) for k in ks]
    b.append('<polyline fill="none" stroke="%s" stroke-width="2.2" points="%s"/>'
             % (PC, " ".join(f"{x:.1f},{y(v):.1f}" for x, v, _ in pts)))
    for x, v, (c0, c1) in pts:
        b.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(c0):.1f}" y2="{y(c1):.1f}" stroke="{PC}" stroke-width="1.4"/>')
        b.append(f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="4" fill="{PC}"/>')
    b.append(text(xs["all"] + 10, y(pts[-1][1]) + 4, "256 px", 11, PC))
    m5 = S5["macro_auroc"]
    p5 = [(xs[k], 100 * m5[f"C{k}"]["point"], [100 * v for v in m5[f"C{k}"]["ci95"]]) for k in ("1", "4", "16")]
    b.append('<polyline fill="none" stroke="%s" stroke-width="2.2" stroke-dasharray="7 4" points="%s"/>'
             % (PC5, " ".join(f"{x:.1f},{y(v):.1f}" for x, v, _ in p5)))
    for x, v, (c0, c1) in p5:
        b.append(f'<line x1="{x + 5:.1f}" x2="{x + 5:.1f}" y1="{y(c0):.1f}" y2="{y(c1):.1f}" stroke="{PC5}" stroke-width="1.4"/>')
        b.append(f'<rect x="{x + 1:.1f}" y="{y(v) - 4:.1f}" width="8" height="8" fill="{PC5}"/>')
    b.append(text(p5[-1][0] + 14, y(p5[-1][1]) + 4, "512 px", 11, PC5))
    b.append(text(L + 4, T + 12, "PatchCore input:  ● 256 px (anomalib default)   ■ 512 px", 11, MUTED))
    # VLMs shown one good part sit at k = 1.
    for key, col, dx in (("A1", JEV, -9), ("B1", GEM, 9)):
        p, (c0, c1) = 100 * m[key]["point"], [100 * v for v in m[key]["ci95"]]
        x = xs["1"] + dx
        b.append(f'<line x1="{x}" x2="{x}" y1="{y(c0):.1f}" y2="{y(c1):.1f}" stroke="{col}"/>')
        b.append(f'<rect x="{x - 4}" y="{y(p) - 4:.1f}" width="8" height="8" fill="{col}" transform="rotate(45 {x} {y(p):.1f})"/>')
    ks_a0 = S["k_star"]["A0"]["k_star"]
    b.append(f'<line x1="{xs[ks_a0]:.1f}" x2="{xs[ks_a0]:.1f}" y1="{T}" y2="{H - B}" stroke="{TXT}" '
             f'stroke-dasharray="2 3" opacity="0.6"/>')
    b.append(text(xs[ks_a0] + 6, y(78.4), f"k* = {ks_a0} at 256 px", 11, TXT))
    b.append(text(xs["1"] + 16, y(p5[0][1]) - 14, "k* = 1 at 512 px", 11, PC5))
    b.append(text(xs["2"] - 30, H - B - 8, "◆ VLM shown one known-good reference, placed at k = 1", 10.5, MUTED))
    b.append(text(W - R + 8, y(pts[0][1]) + 30, "bands: VLM with", 10, MUTED))
    b.append(text(W - R + 8, y(pts[0][1]) + 43, "0 good parts, 95% CI", 10, MUTED))
    return svg(W, H, "".join(b), "Macro image AUROC of PatchCore against the number of good parts at two input "
               "sizes, with the two VLMs as flat bands. PatchCore passes Jev-Omni at eight good parts at 256 px "
               "and at one good part at 512 px.")


# --- Figure: per-category AUROC, VLM against PatchCore ------------------------------------
def per_category():
    pc = S["per_category_auroc"]
    order = ["capsules", "macaroni1", "macaroni2", "candle", "cashew", "chewinggum", "fryum", "pipe_fryum",
             "pcb1", "pcb2", "pcb3", "pcb4"]
    groups = {"capsules": "several parts per photo", "cashew": "one part per photo", "pcb1": "circuit boards"}
    W, L, R, T, row = 720, 130, 30, 44, 24
    H = T + row * len(order) + 3 * 22 + 40
    lo, hi = 50, 100
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    b = []
    for v in range(lo, hi + 1, 10):
        b.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{T - 6}" y2="{H - 34}" stroke="{GRID}"/>')
        b.append(text(x(v), H - 18, v, 11, MUTED, "middle"))
    legend = [("Jev-Omni, 0 good parts", JEV, "A0"), ("PatchCore 16, 256 px", "hollow", "C16"),
              ("PatchCore 16, 512 px", PC5, "C16r")]
    for i, (lab, col, _) in enumerate(legend):
        mk = (f'fill="{BG}" stroke="{PC}" stroke-width="2"' if col == "hollow" else f'fill="{col}"')
        b.append(f'<circle cx="{L + i * 200 + 6}" cy="18" r="5" {mk}/>' + text(L + i * 200 + 16, 22, lab, 11, MUTED))
    yy = T
    for c in order:
        if c in groups:
            yy += 22
            b.append(text(12, yy - 6, groups[c], 10.5, MUTED, "start", 600))
        vals = {"A0": 100 * pc["A0"][c], "C16": 100 * pc["C16"][c], "C16r": 100 * S5["per_category_auroc"]["C16"][c]}
        b.append(text(L - 10, yy + 12, c, 12, TXT, "end"))
        b.append(f'<line x1="{x(min(vals.values())):.1f}" x2="{x(max(vals.values())):.1f}" y1="{yy + 8}" y2="{yy + 8}" '
                 f'stroke="{GRID}" stroke-width="3"/>')
        for lab, col, k in legend:
            mk = (f'fill="{BG}" stroke="{PC}" stroke-width="2"' if col == "hollow" else f'fill="{col}"')
            b.append(f'<circle cx="{x(vals[k]):.1f}" cy="{yy + 8}" r="5" {mk}/>')
        yy += row
    return svg(W, H, "".join(b), "Image AUROC per VisA product for Jev-Omni with no good parts and PatchCore "
               "with 16 good parts at 256 and at 512 pixels.")


# --- Figure: missing parts, one image against a reference pair ----------------------------
def missing_parts():
    bt = S["per_defect_type_auroc"]
    keys = ["pcb1/missing", "pcb2/missing", "pcb3/missing", "cashew/small scratches"]
    W, H, L, T, B = 720, 250, 250, 40, 40
    lo, hi = 20, 100
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - 40)
    b = [f'<line x1="{x(50):.1f}" x2="{x(50):.1f}" y1="{T - 10}" y2="{H - B + 4}" stroke="{MUTED}" '
         f'stroke-dasharray="3 3"/>', text(x(50) + 4, H - B + 2, "coin flip", 10.5, MUTED)]
    for v in range(lo, hi + 1, 20):
        b.append(text(x(v), H - 16, v, 11, MUTED, "middle"))
    rowh = (H - T - B) / len(keys)
    for i, k in enumerate(keys):
        yy = T + i * rowh + rowh / 2
        a0, a1 = 100 * bt["A0"][k]["auroc"], 100 * bt["A1"][k]["auroc"]
        n = bt["A0"][k]["n"]
        b.append(text(L - 10, yy + 4, f"{k.replace('/', ' · ')} (n={n})", 11.5, TXT, "end"))
        b.append(f'<line x1="{x(a0):.1f}" x2="{x(a1):.1f}" y1="{yy}" y2="{yy}" stroke="{JEV}" stroke-width="2" '
                 f'marker-end="url(#arr)"/>')
        b.append(f'<circle cx="{x(a0):.1f}" cy="{yy}" r="4.5" fill="{BG}" stroke="{JEV}" stroke-width="2"/>')
        b.append(text(x(a0) - 8, yy - 8, f"{a0:.0f}", 10.5, MUTED, "end"))
        b.append(text(x(a1) + 8, yy - 8, f"{a1:.0f}", 10.5, JEV))
    defs = (f'<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
            f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{JEV}"/></marker></defs>')
    b.insert(0, defs)
    b.append(text(L, 18, "Jev-Omni AUROC: ○ photo alone → with one known-good reference", 11.5, MUTED))
    return svg(W, H, "".join(b), "Jev-Omni AUROC on missing-component and small-scratch defects, alone and with "
               "a known-good reference image.")


# --- Figure: triage coverage ---------------------------------------------------------------
def coverage():
    t = S["triage"]
    rows = [("Jev-Omni, 0 good parts", "A0", JEV), ("Gemma 4 zero-shot, 0", "B0", GEM),
            ("PatchCore, 1 good part", "C1", PC), ("PatchCore, 16", "C16", PC), ("PatchCore, all", "Call", PC),
            ("PatchCore, 16 at 512 px", "C16r", PC5)]
    W, L, T, row = 720, 200, 34, 34
    H = T + row * len(rows) + 30
    x = lambda v: L + v * (W - L - 70)
    b = [text(L, 20, "share of a 1%-defective line decided without a human, escapes ≤ 5%", 11.5, MUTED)]
    for i, (lab, k, col) in enumerate(rows):
        yy = T + i * row
        v = (S5["triage"]["C16"] if k == "C16r" else t[k])["per_category"]["coverage_at_prevalence_0.01"]
        b.append(text(L - 10, yy + 16, lab, 12, TXT, "end"))
        b.append(f'<rect x="{L}" y="{yy + 4}" width="{x(v) - L:.1f}" height="18" rx="3" fill="{col}" opacity="0.85"/>')
        b.append(text(x(v) + 8, yy + 17, f"{100 * v:.0f}%", 12, TXT))
    return svg(W, H, "".join(b), "Share of parts auto-decided at 5% escapes and 1% prevalence, per system.")


for name, fn in (("k-curve", k_curve), ("per-category", per_category), ("missing-parts", missing_parts),
                 ("coverage", coverage)):
    (out / f"fig-{name}.svg").write_text(fn())
    print("wrote", name)
