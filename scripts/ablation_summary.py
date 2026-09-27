"""Resolution ablation: the scorer's k* and macro AUROC with PatchCore at 512 x 512 (k = 1, 4, 16).

Usage: uv run python scripts/ablation_summary.py <results dir>
Writes <results dir>/summary_r512.json.
"""
import json
import sys
from pathlib import Path

from jev_inspection import score, visa

res = Path(sys.argv[1])
s = score.summarise(visa.DEFAULT_ROOT, [res / "jev_visa.jsonl", res / "base_visa.jsonl"],
                    res / "patchcore_wide_resnet50_2_r512.jsonl")
keep = {k: s[k] for k in ("macro_auroc", "k_star", "per_group_auroc")}
keep["triage"] = {k: v for k, v in s["triage"].items() if k.startswith("C")}
keep["per_category_auroc"] = {k: v for k, v in s["per_category_auroc"].items() if k.startswith("C")}
(res / "summary_r512.json").write_text(json.dumps(keep, indent=1))
for k in ("C1", "C4", "C16"):
    m = keep["macro_auroc"][k]
    print(k, round(100 * m["point"], 1), [round(100 * x, 1) for x in m["ci95"]])
for v in ("A0", "B0"):
    print(v, "k* =", keep["k_star"][v]["k_star"],
          [(r["k"], round(100 * r["diff"], 1), [round(100 * x, 1) for x in r["ci95"]]) for r in keep["k_star"][v]["curve"]])
