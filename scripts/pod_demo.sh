#!/usr/bin/env bash
# Day-0 conveyor demo on one GPU pod: fetch the clip, detect + track (RF-DETR, own env),
# then Gemma 4 zero-shot on each tracked fruit (the measured stack).
set -uo pipefail
cd /workspace/jev-omni-inspection
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p results/demo
exec > >(tee -a results/demo/demo.log) 2>&1
echo "== $(date -u) setup"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
command -v uv >/dev/null || pip install --break-system-packages -q uv
uv sync -q --frozen --extra vlm --python 3.12 || { echo '!! uv sync failed'; exit 1; }
CLIP=/workspace/tangerines-1080p.mp4
curl -sL "$CLIP_URL" -o $CLIP
echo "$CLIP_SHA256  $CLIP" | sha256sum -c - || { echo '!! clip hash'; exit 1; }
echo "== $(date -u) detect"
uv run --no-project --python 3.12 --with rfdetr==1.11.0 --with supervision python scripts/demo_detect.py \
  $CLIP results/demo --t0 4.5 --t1 10.75 || { echo '!! detect failed'; exit 1; }
echo "== $(date -u) classify"
uv run python scripts/demo_classify.py results/demo --phrase "a tangerine" || { echo '!! classify failed'; exit 1; }
echo "== $(date -u) DONE"
