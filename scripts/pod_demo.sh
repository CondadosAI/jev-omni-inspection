#!/usr/bin/env bash
# Day-0 lime-line demo on one GPU pod: fetch the clip, detect + track + count (RF-DETR in its own
# environment), then Gemma 4 zero-shot on each tracked lime, with the frozen question and with a
# one-sentence spec; finally the PatchCore 512 px maps for the parts in the VisA belt video.
#   CLIP_URL=<Pexels 32953325, 1080x1920 file> bash scripts/pod_demo.sh
# Afterwards, locally:
#   uv run --with matplotlib python scripts/demo_hue.py results/demo-limes
#   uv run python scripts/render_counting_video.py <clip> results/demo-limes limes-day0.mp4
#   uv run python scripts/render_visa_video.py visa-belt.mp4 --maps results/demo/visa-maps
set -uo pipefail
cd /workspace/jev-omni-inspection
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p results/demo-limes
exec > >(tee -a results/demo-limes/demo.log) 2>&1
echo "== $(date -u) setup"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
command -v uv >/dev/null || pip install --break-system-packages -q uv
uv sync -q --frozen --extra vlm --extra patchcore --python 3.12 || { echo '!! uv sync failed'; exit 1; }
CLIP=/workspace/limes-1080x1920.mp4
curl -sL "$CLIP_URL" -o $CLIP
echo "2bec11f3377cd8cab94c04ad19c9d5cf604db6669ab66220af5849711ca21d72  $CLIP" | sha256sum -c - \
  || { echo '!! clip hash'; exit 1; }

echo "== $(date -u) detect, track, count"
uv run --no-project --python 3.12 --with rfdetr==1.11.0 --with supervision --with opencv-python-headless \
  python scripts/demo_detect.py $CLIP results/demo-limes --stride 2 --classes "sports ball,orange,apple" \
  --threshold 0.05 --line 0.62 || { echo '!! detect failed'; exit 1; }

echo "== $(date -u) classify"
uv run python scripts/demo_classify.py results/demo-limes --phrase "a lime" --name verdicts_generic \
  || { echo '!! classify failed'; exit 1; }
uv run python scripts/demo_classify.py results/demo-limes --phrase "a Tahiti lime graded for export" \
  --spec "Export grade requires a green skin; yellowing, spots or damage count as defects." \
  --name verdicts_spec || { echo '!! classify failed'; exit 1; }

echo "== $(date -u) VisA maps for the belt video"
export VISA_ROOT=/workspace/visa
if [ ! -f "$VISA_ROOT/split_csv/1cls.csv" ]; then
  mkdir -p $VISA_ROOT && curl -sS https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar -o /workspace/visa.tar
  echo "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362  /workspace/visa.tar" | sha256sum -c - \
    || { echo '!! visa hash'; exit 1; }
  tar xf /workspace/visa.tar -C $VISA_ROOT && rm /workspace/visa.tar
fi
uv run python scripts/demo_maps.py results/demo/visa-maps $(cat scripts/visa_selection.txt) || { echo '!! maps failed'; exit 1; }
echo "== $(date -u) DONE"
