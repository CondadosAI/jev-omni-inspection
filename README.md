# jev-omni-inspection

How many good parts does PatchCore need to beat a vision-language model that has seen none?
Measured on VisA: Jev-Omni and the untouched Gemma 4 12B, zero-shot, against PatchCore fitted on
k = 1, 2, 4, 8, 16, 64 and all good parts per product.

Write-up: [condados.ai/blog/patchcore-vs-zero-shot-vlm-visa](https://condados.ai/blog/patchcore-vs-zero-shot-vlm-visa).
The pre-registration, including the frozen prompt and every departure from it, is in
[`DESIGN.md`](DESIGN.md).

## Results

| System | Good parts seen | Macro image AUROC (95% CI) |
|:--|--:|:--|
| Jev-Omni, test photo only | 0 | 81.1 (79.4–82.7) |
| Gemma 4 12B zero-shot, test photo only | 0 | 82.9 (81.3–84.5) |
| PatchCore 256 px, k = 1 / 8 / 16 | 1–16 | 80.8 / 84.2 / 85.7 |
| PatchCore 256 px, all | 449–904 | 90.2 (89.0–91.4) |
| PatchCore 512 px, k = 1 / 4 / 16 | 1–16 | 84.5 / 89.7 / 92.4 |
| RSI-Jev v4.0-VL, test photo only (added after the pre-registered design) | 0 | 86.6 (85.1–88.1) |

k\* (the first k whose paired-bootstrap interval on the difference is above zero) is 8 against
Jev-Omni and 16 against Gemma 4 at anomalib's default 256 px (pre-registered), and 1 and 4 at
512 px (an ablation added after the run). `results/summary.json` holds the 256 px run,
`results/summary_r512.json` the ablation.

The RSI-Jev v4.0-VL row was added after the pre-registered design, by the RSI-Jev authors, and is
not part of the pre-registered comparison. It uses the same frozen prompt, split, option orders
and statistics; per-image scores are in `results/rsijev_visa.jsonl`, the summary in
`results/summary_rsijev.json`. Caveats: it reads about 1,000 input tokens per photo, against about
350 for Jev-Omni and Gemma 4; VisA is not among its fine-tuning sources, but its base model's
(Qwen3.5-2B) pretraining data cannot be checked; and it is below both VLMs on chewinggum and
pipe_fryum.

## Layout

```
src/jev_inspection/
  visa.py        official 1cls split, the frozen prompt, the frozen draws
  vlm.py         Jev-Omni head and Gemma 4 digit readout -> float64 log-odds, both option orders
  rsijev.py      RSI-Jev over its HTTP server (Jev's request plus images), added after the run
  patchcore.py   anomalib's PatchcoreModel on k good parts
  score.py       macro AUROC, paired bootstrap, k*, cross-fitted triage, Holm
  cli.py         jev-inspection {vlm, patchcore, probe, score, step0}
scripts/
  pod_run.sh       the full run on one GPU pod
  pod_ablation.sh  PatchCore at 512 x 512
  post_data.py     numbers and lab data used by the post
  post_figures.py  the post's SVG figures
  ablation_summary.py      the 512 px ablation -> results/summary_r512.json
  post_images.py           the heat-map figure and the cover background
  render_visa_video.py     VisA parts on a belt, verdicts from the saved scores
  demo_maps.py             PatchCore 512 px maps for the parts in that video
  pod_demo.sh / demo_detect.py / demo_classify.py / demo_hue.py / render_counting_video.py
                   day-0 demo on a lime line: detect, track, count, ask Gemma 4 (unlabelled clip)
examples/
  system_one_readout.py    the same photo and question through Jev-Omni's predict() and through
                           Gemma 4 read in one pass (the two snippets in the post)
data/              step-0 counts and the frozen reference / subset draws
results/           per-image scores and summaries cited by the post
```

## Reproduce

```bash
uv sync --extra vlm --extra patchcore
export VISA_ROOT=/path/to/visa     # default: ~/.condados-jev/visa
bash scripts/pod_run.sh            # on a CUDA GPU with >= 48 GB (an L40S took 4 h 52 min);
                                   # the script expects the repo at /workspace/jev-omni-inspection
uv run jev-inspection score --results results
uv run pytest                      # the end-to-end test needs VisA under VISA_ROOT
```

RSI-Jev v4.0-VL (added after the run; it is served over HTTP, so no GPU stack here):

```bash
pip install "rsi-jev[vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev"
rsi-jev serve shgao/rsi-jev-v4.0-vl-qwen3.5-2b --port 8000
uv run jev-inspection vlm --model rsijev --server http://localhost:8000
uv run jev-inspection score --results results --rsijev    # -> results/summary_rsijev.json
```

The RSI-Jev repository has the same run as one script, `scripts/vision_benches/visa.py`
([Shanghua-Gao/RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev/tree/main/scripts/vision_benches)),
which imports this repository's prompt and statistics.

The VisA archive is downloaded from the official URL and checked against
sha256 `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`.

## Licence

The code in this repository is Apache-2.0 (see `LICENSE`). **That licence covers the code and the
result files only.** The scripts download third-party data and weights at run time; none of them is
redistributed here, and each keeps its own terms:

| Asset | Revision | Terms |
|---|---|---|
| [VisA](https://github.com/amazon-science/spot-diff) (Zou et al., ECCV 2022) | archive `VisA_20220922.tar`, repo `2a692ab5` | CC BY 4.0 (`LICENSE-DATASET` in the repo); the repo's code is Apache-2.0 |
| [akhilaaa3/Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | `5addda86` | Apache-2.0 on the model card |
| [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | `707f0a3b` | Apache-2.0 on the model card, which also links the [Gemma 4 license page](https://ai.google.dev/gemma/docs/gemma_4_license) |
| [timm/wide_resnet50_2.racm_in1k](https://huggingface.co/timm/wide_resnet50_2.racm_in1k) | `30f73ace` | Apache-2.0 on the model card; trained on ImageNet-1k, whose images have their own terms |
| [timm/wide_resnet101_2.tv_in1k](https://huggingface.co/timm/wide_resnet101_2.tv_in1k) (sanity anchor only) | `bc795a74` | BSD-3-Clause on the model card; torchvision's ImageNet-1k weights |
| [shgao/rsi-jev-v4.0-vl-qwen3.5-2b](https://huggingface.co/shgao/rsi-jev-v4.0-vl-qwen3.5-2b) (post-hoc row only) | v4.0-VL | Apache-2.0 on the model card |
| [anomalib](https://github.com/open-edge-platform/anomalib) | 2.6.2 (PyPI) | Apache-2.0 |
| [RF-DETR](https://github.com/roboflow/rf-detr) base, COCO weights (day-0 demo only) | rfdetr 1.11.0 | Apache-2.0 |
| [Lime sorting on conveyor belt in factory](https://www.pexels.com/video/lime-sorting-on-conveyor-belt-in-factory-32953325/), Comercial GB (day-0 demo only) | Pexels 32953325, 1080 × 1920, sha256 `2bec11f3…1d72` | [Pexels licence](https://www.pexels.com/license/); downloaded, not redistributed |

`output/cover-bg.png` (the post's cover background) contains VisA images, under CC BY 4.0 with
attribution to Zou et al. (2022).

Jev-Omni states that it is not affiliated with TypeSafe AI or its Jev model. Neither is this
repository.
