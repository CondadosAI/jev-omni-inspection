#!/usr/bin/env bash
# PatchCore resolution ablation: 512x512 input, k = 1, 4, 16, seeds 0-2 (fast fits only).
set -uo pipefail
cd /workspace/jev-omni-inspection
export HF_HOME=/workspace/hf VISA_ROOT=/workspace/visa
mkdir -p results
exec > >(tee -a results/ablation.log) 2>&1
echo "== $(date -u) setup"; nvidia-smi --query-gpu=name,driver_version --format=csv
command -v uv >/dev/null || pip install --break-system-packages -q uv
uv sync -q --frozen --extra patchcore --python 3.12 || { echo '!! uv sync failed'; exit 1; }
if [ ! -f "$VISA_ROOT/split_csv/1cls.csv" ]; then
  mkdir -p "$VISA_ROOT" && cd "$VISA_ROOT"
  curl -sS -o VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
  echo "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362  VisA_20220922.tar" | sha256sum -c - || { echo '!! hash'; exit 1; }
  tar xf VisA_20220922.tar && rm VisA_20220922.tar
  cd /workspace/jev-omni-inspection
fi
echo "== $(date -u) ablation"
uv run jev-inspection patchcore --image-size 512 --k 1 --k 4 --k 16 --out results || echo "!! ablation failed"
echo "== $(date -u) DONE"
