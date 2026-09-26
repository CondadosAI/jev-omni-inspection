#!/usr/bin/env bash
# One pod session (L40S, driver >= 580): VisA fetch + hash check, PatchCore sanity anchor
# (stops the run if it is far off), then the VLM run, the PatchCore k-sweep and the probe.
# Copy the repo to /workspace/jev-omni-inspection first; run from there under nohup.
set -uo pipefail
cd /workspace/jev-omni-inspection
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1 VISA_ROOT=/workspace/visa
mkdir -p results
exec > >(tee -a results/run.log) 2>&1
echo "== $(date -u) setup"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v uv >/dev/null || pip install --break-system-packages -q uv
uv sync -q --frozen --extra vlm --extra patchcore --python 3.12 || { echo '!! uv sync failed'; exit 1; }
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" || { echo '!! CUDA not usable'; exit 1; }
uv pip freeze > results/pip-freeze.txt

if [ ! -f "$VISA_ROOT/split_csv/1cls.csv" ]; then
  echo "== $(date -u) VisA"
  mkdir -p "$VISA_ROOT" && cd "$VISA_ROOT"
  curl -sS -o VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
  echo "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362  VisA_20220922.tar" | sha256sum -c - \
    || { echo '!! VisA hash mismatch'; exit 1; }
  tar xf VisA_20220922.tar && rm VisA_20220922.tar
  cd /workspace/jev-omni-inspection
fi

echo "== $(date -u) VLM smoke, the cheap gate (2+2 cashew images; the rows are reused by the full run)"
uv run jev-inspection vlm --model jev --category cashew --limit 2 --out results || { echo '!! jev smoke failed'; exit 1; }

echo "== $(date -u) sanity anchor: PatchCore WRN-101, k=all, seed 0 (EfficientAD reports 94.3 mean image AUROC)"
uv run jev-inspection patchcore --backbone wide_resnet101_2 --k all --seed 0 --out results
uv run python - <<'EOF'
import json, collections, sys
from sklearn.metrics import roc_auc_score
g = collections.defaultdict(lambda: ([], []))
for r in map(json.loads, open("results/patchcore_wide_resnet101_2.jsonl")):
    g[r["obj"]][0].append(r["label"] == "anomaly"); g[r["obj"]][1].append(r["score"])
per = {o: roc_auc_score(*v) for o, v in g.items()}
macro = 100 * sum(per.values()) / len(per)
json.dump({"macro": macro, "per_category": per}, open("results/anchor.json", "w"), indent=1)
print(f"anchor macro image AUROC {macro:.1f} (reference 94.3)")
sys.exit(0 if macro >= 90 else 1)
EOF
[ $? -eq 0 ] || { echo '!! anchor far below 94.3: stop and investigate before the sweep'; exit 1; }

for m in jev base; do
  echo "== $(date -u) VLM $m"
  uv run jev-inspection vlm --model $m --out results || echo "!! $m failed"
done
echo "== $(date -u) contamination probe"
uv run jev-inspection probe --out results || echo "!! probe failed"
echo "== $(date -u) PatchCore sweep"
uv run jev-inspection patchcore --out results --maps-for 16 0 || echo "!! sweep failed"
echo "== $(date -u) score"
uv run jev-inspection score --results results > results/score_stdout.json
echo "== $(date -u) DONE"
