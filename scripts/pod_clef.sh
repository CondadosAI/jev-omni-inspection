#!/usr/bin/env bash
# The Clef arm on one pod session: VisA fetch + hash check, a 2+2 cashew smoke gate, the full test
# set in both configs, then the scorer against the committed Jev-Omni, Gemma 4 and PatchCore rows.
# Usage: bash scripts/pod_clef.sh clef-flash   (L40S 48 GB is enough: 18.8 GB of bf16 weights)
#        bash scripts/pod_clef.sh clef         (needs an 80 GB card: 54.7 GB of bf16 weights)
# Copy the repo to /workspace/jev-omni-inspection first; run from there under nohup. To score both
# releases together, copy the first run's results-clef/ into the second pod before starting it.
set -uo pipefail
MODEL=${1:?clef-flash or clef}
cd /workspace/jev-omni-inspection
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1 VISA_ROOT=/workspace/visa
OUT=results-clef
mkdir -p $OUT
exec > >(tee -a $OUT/run-$MODEL.log) 2>&1
echo "== $(date -u) setup $MODEL"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v uv >/dev/null || pip install --break-system-packages -q uv
uv sync -q --frozen --extra vlm --extra patchcore --python 3.12 || { echo '!! uv sync failed'; exit 1; }
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" || { echo '!! CUDA not usable'; exit 1; }
uv pip freeze > $OUT/pip-freeze.txt

if [ ! -f "$VISA_ROOT/split_csv/1cls.csv" ]; then
  echo "== $(date -u) VisA"
  mkdir -p "$VISA_ROOT" && cd "$VISA_ROOT"
  curl -sS -o VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
  echo "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362  VisA_20220922.tar" | sha256sum -c - \
    || { echo '!! VisA hash mismatch'; exit 1; }
  tar xf VisA_20220922.tar && rm VisA_20220922.tar
  cd /workspace/jev-omni-inspection
fi

echo "== $(date -u) smoke: 2+2 cashew images (the equivalence check against systemone() runs first)"
uv run jev-inspection vlm --model $MODEL --category cashew --limit 2 --out $OUT || { echo '!! smoke failed'; exit 1; }
uv run python - "$OUT/${MODEL}_visa.jsonl" <<'EOF' || { echo '!! smoke rows look wrong'; exit 1; }
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
for r in rows:
    print(r["config"], r["label"], round(r["logodds_defective"], 3), r["option_ids"], r["n_tokens"], r["ms_total"])
assert len(rows) == 8 and all(r["option_ids"] == ["all good", "defective"] for r in rows)
assert len({round(r["logodds_defective"], 6) for r in rows}) > 1, "constant output"
EOF

echo "== $(date -u) full run"
uv run jev-inspection vlm --model $MODEL --out $OUT || { echo "!! $MODEL failed"; exit 1; }

echo "== $(date -u) score against the committed rows"
for f in jev_visa.jsonl base_visa.jsonl patchcore_wide_resnet50_2.jsonl; do cp -n results/$f $OUT/; done
uv run jev-inspection score --results $OUT > $OUT/score_stdout.json
echo "== $(date -u) DONE"
