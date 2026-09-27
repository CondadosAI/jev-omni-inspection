"""Day-0 conveyor demo, step 2: ask Gemma 4 12B zero-shot about each tracked fruit.

The same frozen question and options as the VisA run, both option orders averaged; only the
object phrase changes. There are no labels for this clip, so nothing here is a measurement.
  uv run python scripts/demo_classify.py out/ --phrase "a tangerine"
Writes out/verdicts.json: {track_id: {logodds, p_defective, order0, order1}}.
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image

from jev_inspection import visa, vlm

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--phrase", default="a tangerine")
ap.add_argument("--model", default="base", choices=["base", "jev"])
ap.add_argument("--spec", default="", help="a product spec appended to the state, e.g. what counts as a defect")
ap.add_argument("--name", default="verdicts", help="output file stem")
a = ap.parse_args()

clf, prompt_fn = vlm.load(a.model, "cuda")
state = f"Production-line inspection photo of {a.phrase}." + (f" {a.spec}" if a.spec else "")
res = {}
for crop in sorted((a.out / "crops").glob("*.png"), key=lambda p: int(p.stem)):
    img = Image.open(crop).convert("RGB")
    lo = []
    for order in visa.ORDERS:
        opts = [visa.GOOD, visa.DEFECTIVE] if order == 0 else [visa.DEFECTIVE, visa.GOOD]
        inputs = vlm.make_inputs(clf.processor, [img], prompt_fn(state, visa.QUESTION, opts), "cuda")
        z = vlm.base_logprobs(clf, inputs, 2) if a.model == "base" else vlm.jev_logits(clf, inputs, 2)
        d = opts.index(visa.DEFECTIVE)
        lo.append((z[d] - z[1 - d]).item())
    m = sum(lo) / 2
    res[crop.stem] = {"logodds": m, "p_defective": 1 / (1 + math.exp(-m)), "order0": lo[0], "order1": lo[1]}
    print(crop.stem, round(m, 3), round(res[crop.stem]["p_defective"], 3))
json.dump({"model": a.model, "state": state, "question": visa.QUESTION, "verdicts": res},
          open(a.out / f"{a.name}.json", "w"), indent=1)
