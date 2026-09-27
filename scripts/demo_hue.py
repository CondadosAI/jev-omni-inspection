"""Day-0 lime demo: with no labels, compare each lime's skin colour with Gemma 4's score.

Hue is the median HSV hue (degrees) of the crop pixels outside the grey mask with saturation
above 0.35; on these limes lower means yellower. Spearman's rho between hue and the log-odds of
"defective", for the frozen question and for the one-sentence spec.
  uv run --with matplotlib python scripts/demo_hue.py results/demo-limes
Writes results/demo-limes/hue_analysis.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy.stats import spearmanr

d = Path(sys.argv[1])
g = json.loads((d / "verdicts_generic.json").read_text())["verdicts"]
s = json.loads((d / "verdicts_spec.json").read_text())["verdicts"]
hue = {}
for k in g:
    px = np.asarray(Image.open(d / "crops" / f"{k}.png").convert("RGB")).reshape(-1, 3).astype(float)
    px = px[~(px == 128).all(1)] / 255  # drop the grey mask outside the ellipse
    hsv = rgb_to_hsv(px)
    hue[k] = float(np.median(hsv[hsv[:, 1] > 0.35, 0] * 360))
ks = sorted(g, key=int)
h = [hue[k] for k in ks]
out = {"what": "median skin hue (degrees, HSV, pixels with saturation > 0.35 inside the crop ellipse) vs "
               "Gemma 4 log-odds of 'defective'; lower hue = yellower",
       "n_crops": len(ks)}
for name, v in (("generic", g), ("spec", s)):
    rho, p = spearmanr(h, [v[k]["logodds"] for k in ks])
    out[f"spearman_{name}"] = {"rho": float(rho), "p": float(p)}
    out[f"flagged_p_gt_0.6_{name}"] = int(sum(v[k]["p_defective"] > 0.6 for k in ks))
out["hue"] = hue
(d / "hue_analysis.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "hue"})
