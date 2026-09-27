"""Numbers and data files the CondadosAI post uses, derived from saved run artifacts only.

Usage: uv run python scripts/post_data.py <results dir> <out dir>
Writes post_numbers.json (every number the prose quotes that is not already in summary.json)
and triage-cashew.json (the scores the TriageLab widget plays with).
"""
import json
import sys
from pathlib import Path

import numpy as np

from jev_inspection import score, visa

res, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
vlm, _ = score.load_vlm([res / "jev_visa.jsonl", res / "base_visa.jsonl"])
pc = score.load_patchcore(res / "patchcore_wide_resnet50_2.jsonl")

numbers = {}

# Worked example 1: one real cashew image through Jev-Omni, both option orders.
rows = [json.loads(line) for line in (res / "jev_visa.jsonl").open()]
ex = {}
for name in ("cashew/Data/Images/Anomaly/000.JPG", "cashew/Data/Images/Normal/023.JPG"):
    r = {x["order"]: x for x in rows if x["image"] == name and x["config"] == "single"}
    ex[name] = {"order0_scores": r[0]["scores"], "order1_scores": r[1]["scores"],
                "order0_logodds": r[0]["logodds_defective"], "order1_logodds": r[1]["logodds_defective"],
                "mean_logodds": (r[0]["logodds_defective"] + r[1]["logodds_defective"]) / 2}
numbers["vlm_score_example"] = ex

# Worked example 2 and 4 on cashew: AUROC as a pair count, and cross-fitted triage.
tab = score.Table(visa.DEFAULT_ROOT, ["cashew"])
systems = {"A0": vlm["A0"].reindex(tab.images).to_numpy(),
           "C1": pc["1"][0].reindex(tab.images).to_numpy(),
           "C16": pc["16"][0].reindex(tab.images).to_numpy()}
n_idx, a_idx = tab.idx["cashew"]
numbers["cashew"] = {"n_good": int(len(n_idx)), "n_defective": int(len(a_idx)), "systems": {}}
for name, s in systems.items():
    sa, sn = s[a_idx], s[n_idx]
    wins = float((sa[:, None] > sn[None, :]).sum() + 0.5 * (sa[:, None] == sn[None, :]).sum())
    t = score.triage(tab, s, True)
    numbers["cashew"]["systems"][name] = {
        "pairs": int(len(sa) * len(sn)), "pairs_won": wins, "auroc": wins / (len(sa) * len(sn)),
        "triage": t,
        "per_10000_at_1pct": {
            "defective_parts": 100,
            "escaped": round(100 * t["escape"], 1),
            "to_human": round(100 * (1 - t["auto_given_defective"]) + 9900 * (1 - t["auto_given_good"]), 0),
            "good_rejected": round(9900 * t["overkill"], 0),
        }}

# The TriageLab data: cashew scores for three systems (PatchCore at seed 0).
lab = {"category": "cashew", "systems": []}
for key, label, note in (("A0", "Jev-Omni, no good parts", "log-odds of 'defective'"),
                         ("C1", "PatchCore, 1 good part", "nearest-patch distance"),
                         ("C16", "PatchCore, 16 good parts", "nearest-patch distance")):
    s = systems[key]
    lab["systems"].append({"key": key, "label": label, "score": note,
                           "good": [round(float(x), 4) for x in s[n_idx]],
                           "defective": [round(float(x), 4) for x in s[a_idx]]})
(out / "triage-cashew.json").write_text(json.dumps(lab, separators=(",", ":")))
(out / "post_numbers.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers["cashew"], indent=1)[:2500])
print(json.dumps(numbers["vlm_score_example"], indent=1))
