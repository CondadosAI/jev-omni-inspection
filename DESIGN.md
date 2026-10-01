# How many good parts does PatchCore need to beat a VLM that has seen none?

Status: PLAN v2, 25 Sep 2026; step 0 done, prompt frozen. Post 3 of the Jev-Omni series (1: benchmark vs Gemma 4 zero-shot; 2: Jev-Omni on OpenVINO). Rewritten after an adversarial review of v1 (findings
folded in below). Kept as written at each step; later sections record what changed and why.

## The question

A zero-shot VLM needs no good parts to start inspecting. PatchCore needs a memory bank of
them. Sweep PatchCore's number of good parts **k = 1, 2, 4, 8, 16, 64, all** and find
the smallest k at which it beats Jev-Omni (and the same backbone zero-shot) on VisA.

This is a finding in either direction and it is the question a line engineer asks
("can I start inspecting on day one, and when does collecting normals pay off?"). v1's
"Jev-Omni vs PatchCore at full data" was a strawman: PatchCore's published VisA image
AUROC is ~94 and the result would have been foregone.

Secondary question, same data: at a fixed defect-escape rate, what fraction of the line
can each system decide without a human (triage)?

## Data (verified 25 Sep 2026 unless marked)

**VisA**, Zou et al., "SPot-the-Difference Self-Supervised Pre-training for Anomaly
Detection and Segmentation", ECCV 2022 (arXiv 2207.14315, title/authors/venue opened).
**CC BY 4.0** (README: "The data is released under the CC BY 4.0 license"; repo code
Apache-2.0). Download:
`https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`.

- Official `split_csv/1cls.csv`: **test = 2,162 images, 962 normal + 1,200 anomalous,
  exactly 100 anomalies per category, 50–101 normals per category**. Train = 450–905
  normals per category. anomalib's VisA datamodule applies this same split.
- Test prevalence is 55% defective. A real line runs ~0.1–2%. Every prevalence-dependent
  number is either defined without prevalence (FNR, FPR) or reweighted to a stated line
  prevalence (1%). Never reported raw.
- Groups from the README, used for the per-group breakdown: PCB (pcb1–4), multiple
  instances (capsules, candles, macaroni1–2), single instance (cashew, chewinggum,
  fryum, pipe_fryum). Capsules and macaroni2 "largely differ in locations and poses".
- **Step 0:** download; count per category; dump the defect vocabulary from
  `image_anno.csv` / `id2class.py` (at most 9 classes incl. "Other", verified), and how
  many images carry more than one type. Only needed for the appendix Q2.

## Systems

| # | System | Good parts seen | Notes |
|---|---|---|---|
| A0 | Jev-Omni head, single image | 0 | akhilaaa3/Jev-Omni @ 5addda86 |
| A1 | Jev-Omni head, reference + test | 1 | out of the head's training distribution; stated |
| B0 | Gemma 4 12B IT zero-shot, digit-logit readout, single image | 0 | google/gemma-4-12B-it @ 707f0a3b |
| B1 | same, reference + test | 1 | |
| C(k) | PatchCore (anomalib, pinned), k good parts | k | k ∈ {1,2,4,8,16,64,all}, 3 sampling seeds each |

A1/B1 place the VLMs on the same x-axis as PatchCore at k = 1, which makes the chart
honest: a VLM shown one good part against PatchCore shown one good part.

PatchCore settings, pinned at scaffold: anomalib version, WideResNet-50 backbone, input
resolution, coreset ratio (1.0 below some k, where subsampling is meaningless), 3 coreset
/ sampling seeds. One extra run at k = all with **WideResNet-101** is the sanity anchor:
EfficientAD (Batzner et al., arXiv 2303.14535, Tables 2/9/10) reports **PatchCore 94.3
mean VisA image AUROC with WRN-101** (verified by the reviewer in the paper; I re-open it
before citing). If our WRN-101 run is far off, stop and find out why before the sweep.

## The VLM prompt (frozen before the run)

- Q1 only in the main text: options "the part is good" / "the part is defective".
- **Both option orders**, averaged per image (a 2-option readout has positional bias;
  costs ~2k extra forwards per system/config).
- State: category name + one neutral sentence on what a good part looks like, identical
  for A and B. A1/B1: "The first image shows a known-good part. The second image shows
  the part under inspection." Reference = one fixed train normal per category (seeded).
- **Score = log-odds, not probability.** Log the raw head logits (A) and digit
  log-probs (B) in float64; AUROC and thresholds use log P(defective) − log P(good).
  BLINK showed 57/733 wrong answers at ≥ 0.99 confidence, so probabilities saturate and
  tie.

## Metrics (pre-registered)

**Primary endpoint (the only confirmatory test):** macro image AUROC over the 12
categories, per system. The headline number is **k\***, the smallest k at which C(k)'s
macro AUROC exceeds A0's, with k\* defined on the lower bound of a paired, category-
stratified bootstrap CI (2,000 resamples, resampling test images and, for C, the k-sample
seed). Same k\* reported against B0, A1, B1.

Descriptive (no significance claims beyond Holm-corrected ones):
- Per-category and per-group AUROC, Holm-corrected across the 12 categories.
- A0 vs B0 macro AUROC difference with bootstrap CI (does the head help on inspection at
  all? On BLINK it did not).
- **Triage:** two thresholds, auto-accept / human / auto-reject. **One global threshold
  pair per system** (per-category thresholds are noise with 100 anomalies). Fit by
  5-fold cross-fitting over the test set so all 1,200 anomalies are scored.
  - Escape = FNR among anomalies (fraction of defective parts auto-accepted).
  - Operating point: **escape ≤ 5%** (primary for triage). Escape ≤ 1% is descriptive
    only, with a Clopper-Pearson interval: at 1,200 anomalies it rests on ~12 events.
  - Coverage at line prevalence π = 1%: π · P(auto | defective) + (1 − π) · P(auto | good).
- ECE after a per-system temperature fit inside the same cross-fitting folds. Appendix.
- Contamination probe: ask B0 whether it recognises the images as VisA / names the
  dataset, on 50 images. Reported either way.

## Latency

A/B median per image on the L40S (bf16, batch 1, warmup discarded), **including image
preprocessing**. PatchCore timed on the same L40S (so the comparison is same-hardware)
and on the laptop CPU (where it would actually run). Both rows labelled with hardware.

## Compute and cost

- VLMs: 2,162 images × 2 configs × 2 option orders × 2 models ≈ 17k forwards; A1/B1 have
  twice the image tokens. Plus Q2 appendix if kept. Estimate 1–2 h of L40S, **< $2**.
- PatchCore sweep: 12 categories × 7 k × 3 seeds = 252 fits, each a feature pass over
  ≤ 905 images plus inference over the test set. Minutes to an hour; on the pod
  alongside the VLM run.

## Code: what the existing bench does not do

The image-benchmark harness from the first Jev-Omni post (`eval_img.py` in CondadosAI/jev-omni-eval) is the starting point, not a drop-in:
- Resume keys on `id` and writes `{model}_{bench}.jsonl`: put config + option order in
  the id, or later configs are silently skipped.
- Replace `check_jev_equivalence` (hard-wired to BLINK Counting) with a VisA image;
  the two-image path has no official-`predict()` equivalent to check against, say so.
- Add: VisA loader, single/ref configs, option-order swap, float64 logit logging,
  preprocessing inside the timer.
- New scorer (not a reuse of `score_img.py`): AUROC, bootstrap, k\*, cross-fit triage,
  temperature, Holm, merge with PatchCore outputs.

## Cut from v1 (appendix or later)

- Q2 defect type. If kept in the appendix: correct = matches any of the image's types,
  "Other" dropped from options, "Other"-only images excluded.
- Fixed-crop config (meaningless for capsules/macaroni2). If resolution needs its own
  ablation later: 2×2 tiling scored by max log-odds, not a crop.
- PatchCore on OpenVINO / iGPU (a separate edge post if the numbers justify it; the
  `openvino` + `IntelSoftwareInnovator` tags only if that ships).
- Laptop Jev-Omni (no measured GGUF/OpenVINO port with the head exists here).

## The post

- Slug proposal `patchcore-vs-zero-shot-vlm-visa` (permanent, confirm before publish).
  seoTitle: "Zero-shot VLM vs PatchCore on VisA" (≤ 49). Title and TL;DR name VisA;
  it is a benchmark, not a line (rule 6).
- Headline chart: macro AUROC vs k (log x-axis) for C(k) with seed bands; A0/B0 as flat
  lines, A1/B1 as points at k = 1; k\* marked.
- Second chart: coverage at escape ≤ 5%, π = 1%, per system.
- **Worked example** on one real category (cashew): threshold from the fold, escapes
  out of 100 with its interval, coverage reweighted to 1% prevalence, what it means on a
  10,000-part shift.
- **Interactive lab:** `ConfusionMatrixLab` has one threshold; triage needs two. Either
  extend it with a reject band + coverage readout at a prevalence slider, or add a
  `TriageLab` in the repo's lab-kit style. Seed it with real saved log-odds for cashew.
- PatchCore heatmaps in the demo; the VLM gives no localization, shown honestly.
- Reproducibility block per CLAUDE.md §9 plus: PatchCore resolution, coreset ratio,
  seeds, k-sampling seeds, bootstrap count, option-order averaging.
- References: VisA paper, PatchCore (Roth et al., CVPR 2022, verify arXiv ID before
  citing), EfficientAD, anomalib docs (versioned), Jev-Omni card, Gemma 4 card, plus a
  book for AUROC/thresholds (locator checked against a real TOC).

## Demo (LinkedIn + cover)

Simulated conveyor of VisA test parts: A0 decision + confidence + auto/human, PatchCore
(k = 16 or k\*) beside it with its heatmap. Rendered offline from saved predictions;
overlay says "VisA benchmark images, decisions precomputed, latency measured on an L40S".
Close on the k-curve. Lossy WebP q50 + static 1200×675 `ogImage`.

## Licences for the public repo README

To verify at scaffold, from the actual pages (not this list): VisA CC BY 4.0
(verified; attribution, downloaded never redistributed); Gemma 4 12B IT Apache-2.0
(verified on the HF card 25 Sep 2026); Jev-Omni Apache-2.0 per its card (re-check, and
note its training data's terms if stated); anomalib; torchvision WRN-50/101 ImageNet
weights. Repo `CondadosAI/jev-omni-inspection`, Apache-2.0.

## Limitations to carry into the post

- VisA public since 2022: possible pretraining contamination (probe result reported).
- One prompt wording; two option orders; one reference image per category.
- A1/B1 feed the head an input format it was not trained on.
- 100 anomalies per category: per-category claims are weak; escape ≤ 1% is descriptive.
- Low-resolution VLM input (~280 soft tokens/image) vs PatchCore's pinned resolution:
  part of what is measured, stated as such, not hidden.
- Simulated line: no motion blur, lighting drift or throughput limit.

## Order of work

0. Download VisA; counts + vocabulary; freeze prompt, reference images, seeds. (laptop)
1. Eval code changes above; dry run on 20 images locally for shape only.
2. One pod session: WRN-101 sanity anchor first (stop if off), then VLM run + PatchCore
   sweep, JSONL + scores out. (< $2)
3. Score: k\*, AUROC table, triage, contamination. Headline decided from the numbers.
4. TriageLab, figures, demo render, draft, voice sweep, audit, PR.

Series order as planned: post 1 (DecisionBench + BLINK measurements) and post 2 (OpenVINO port) ship first; this post links both.

## Step 0 results (25 Sep 2026)

- Archive `VisA_20220922.tar`, 1,929,840,640 bytes, sha256
  `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`. Its `split_csv/` is
  byte-identical to `amazon-science/spot-diff@2a692ab`. FiftyOne's zoo has no VisA (only
  third-party HF mirrors), so the official S3 URL + this hash is the reproducible source.
- 1cls counts confirmed: train 8,659 normals (450–905/category); test 962 normal +
  1,200 anomalous, 100 anomalies in every category.
- Image sizes 1274–1562 px wide, one size per category.
- Multi-type anomalies: capsules 70/100, pcb4 56, chewinggum 43, the rest 7–22. "Other"
  appears on 3 images in total. Vocabulary per category in `visa_summary.json`.
- Frozen draws in `frozen_draws.json` (reference image per category from
  `Random("ref-<obj>")`; nested PatchCore subsets, reference excluded, seeds 0–2).
- Artifacts: `data/visa_summary.json`, `data/frozen_draws.json`, script `src/jev_inspection/step0.py`
  (move both into the companion repo at scaffold).

## Prompt (FROZEN 25 Sep 2026, approved by Luis)

The text below is what goes into `jev_omni._prompt(state, question, options)` (the same
call as the BLINK run), which wraps it in its own template and the chat template. It is
not the full prompt string.

Single image:
- state: `Production-line inspection photo of {phrase}.`
- question: `Is everything in the photo good, or is at least one part defective?`
- options: `["all good", "defective"]` and the reversed order, averaged per image.

Reference + test (two images, reference first):
- state: `Production-line inspection photos of {phrase}. The first image is a known-good
  example. The second image is the one under inspection.`
- question and options as above, asked of the second image.

`{phrase}` per category, written from the frozen reference images (no counts, because
the multi-instance categories are not guaranteed a fixed count, and no model names on
the boards):

| category | phrase |
|---|---|
| candle | candles |
| capsules | green gel capsules |
| cashew | a cashew nut |
| chewinggum | a piece of chewing gum |
| fryum | a wheel-shaped fried snack |
| macaroni1, macaroni2 | macaroni pieces |
| pcb1–pcb4 | a printed circuit board |
| pipe_fryum | a pipe-shaped fried snack |

Wording rationale: the v1 idea of a hand-written "what a good part looks like" sentence
per category was dropped; it is a per-category degree of freedom the reader cannot
audit. The question covers both the single-instance and the multi-instance groups.

`k = all` for PatchCore means every train normal **except** the reference image, the same
pool the k ≤ 64 draws come from (449–904 images per category). Draws verified
reproducible: a second run of `visa_step0.py` produced a byte-identical
`frozen_draws.json`.

## Step 1: code, and where it departs from the text above (25 Sep 2026)

This repository (commit 0585117; published as
`CondadosAI/jev-omni-inspection`). `jev-inspection {vlm,patchcore,probe,score,step0}`,
`scripts/pod_run.sh` for the single pod session. 5 unit tests pass (frozen prompt text, nested
draws, AUROC == sklearn with ties, Holm, full summary on synthetic scores with a known k*).

Departures, decided before any real score exists:
- **Triage thresholds are per category** for every system, cross-fit 5-fold within the
  category. The "one global pair" above does not work for PatchCore: its scores are
  distances with a different scale per product, and a line fits one threshold per product
  anyway. A global pair is still reported for the VLMs, whose log-odds share one scale.
  Auto-reject threshold: at most 5% of training normals rejected (the plan had not fixed it).
- **PatchCore coreset:** full memory bank for k ≤ 64, anomalib's 0.1 for k = all.
- **k = all runs one seed.** Its training set is fixed; the seed would only move the coreset
  start, and these are the expensive fits. The bootstrap then has no seed to draw at k = all.
- PatchCore is driven through anomalib's `PatchcoreModel` directly (not the Engine), with
  anomalib's default pre-processing (256×256, ImageNet normalisation, no centre crop).
- Jev-Omni log-odds come from the head evaluated in float32 outside autocast; the run
  checks our path against the official `predict()` first (abort above 1e-3) and logs the
  fp32-vs-bf16 head difference.
- Sanity anchor gate in the pod script: WRN-101 at k = all must reach ≥ 90 macro AUROC
  (reference 94.3) or the run stops before the VLMs and the sweep.

Smoke on the laptop (feature extraction only): PatchCore WRN-50 on cashew, seed 0,
AUROC 0.848 at k = 1 (bank 1,024 patches) and 0.940 at k = 16 (bank 16,384); 14–18 ms per
image on an RTX 3060 Laptop. Not a result, a check that the pipeline runs end to end.

Cost revision: the k = all coreset selections dominate the PatchCore side; expect 3–4 h of
L40S in total (~$3–4 at $1.09/h), not < $2.

Local VLM dry run (processor only, no weights; repo commit 9990571): imports and the
two-image path work. Gemma 4's processor gives **280 patches of 48×48 px per image**
(`pixel_values` (n, 280, 6912)), roughly 960×670 px of effective input; single-image prompts
are ~341–353 tokens, reference + test ~622–646. So the VLM sees *more* pixels than PatchCore
at anomalib's 256×256. The resolution limitation above is rewritten accordingly: it is not
"the VLM cannot see small defects" by construction.

Still owed: PatchCore latency on the laptop CPU (the row a line would actually run; batch-1
L40S latency is now logged at k = 16, seed 0). At publish, un-ignore the cited files in
`results/` (summary, anchor, latency, per-model JSONL) per the allowlist rule.

## Step 2 results (pod run 26 Sep 2026, 10:36–15:28 UTC, L40S, ~$5.35)

Artifacts: `results/` (summary.json, all JSONL, anchor,
latency, probe, pip-freeze, run.log, anomaly maps k=16 seed 0). Scorer: B = 2,000.

Macro image AUROC (95% bootstrap CI):

| system | good parts seen | AUROC |
|---|---|---|
| A0 Jev-Omni, single | 0 | 81.1 [79.4, 82.7] |
| A1 Jev-Omni, ref + test | 1 | 81.7 [79.9, 83.3] |
| B0 Gemma 4 zero-shot, single | 0 | 82.9 [81.3, 84.5] |
| B1 Gemma 4 zero-shot, ref + test | 1 | 83.5 [81.8, 85.1] |
| C PatchCore WRN-50, k = 1 / 2 / 4 / 8 | 1–8 | 80.8 / 82.8 / 83.6 / 84.2 |
| C k = 16 / 64 / all | 16–904 | 85.7 / 87.5 / 90.2 (all: one seed) |

- **k\* (primary endpoint): 8 against Jev-Omni (A0), 16 against A1, B0 and B1.** At k = 1
  PatchCore and the VLMs are level (A0: −0.3 [−3.4, 2.6]).
- **The Jev-Omni head is worse than its own backbone read zero-shot:** A0 − B0 macro
  −1.8, CI [−2.8, −0.8] (descriptive per pre-registration); Holm-significant in candle,
  cashew, macaroni2. Same direction as BLINK.
- Positional bias (log-odds, order 0 − order 1): A0 0.41, A1 0.92, B0 0.12, B1 0.17.
- Triage at escape ≤ 5% (per-category thresholds), coverage at 1% prevalence: VLMs
  0.36–0.39; PatchCore 0.44 (k=1) → 0.54 (k=16) → 0.66 (all). Global VLM thresholds:
  0.18–0.23.
- Contamination probe: 0/50 answers name VisA (top answers Food-101 13, Roboflow Universe
  9, MVTec AD 3).
- Sanity anchor WRN-101 k=all: 90.5 vs 94.3 published; weak on capsules 0.68, macaroni2
  0.65. Our PatchCore is on the weak side of published, which biases k\* toward the VLM.
- Latency (L40S, batch 1, JPEG decode + preprocessing inside the timer): PatchCore k=16
  median 48.5 ms (10 warmup discarded); VLM median `ms_total` per forward: Jev-Omni 154 ms
  single / 357 ms ref, Gemma 204 / 409 ms (no warmup discarded). The averaged-order score
  costs two forwards per image.
- **Correction to the plan's assumption:** VLM log-odds do *not* share one scale across
  products. A global threshold pair covers 0.18–0.23 of the line vs 0.36–0.39 with
  per-category thresholds. Reported as a finding.
- EfficientAD (arXiv 2303.14535, HTML read 26 Sep): they ran PatchCore themselves and
  "disable the cropping of the center 76.6% of input images"; resolution and coreset
  settings for the 94.3 are not stated in the text read. The reason for our 90.5 is
  therefore unverified. Not computed yet: ECE appendix, laptop-CPU PatchCore latency.

## Resolution ablation (27 Sep 2026, RunPod L40, ~$0.30)

PatchCore WRN-50 at 512×512 (everything else unchanged), k = 1, 4, 16 × 3 seeds;
`results/summary_r512.json`. Macro AUROC 84.5 / 89.7 / 92.4 (256 px: 80.8 / 83.6 / 85.7).
**k\* drops to 1 against Jev-Omni** (+3.4, [1.0, 5.8]) **and to 4 against Gemma 4**
(k=1: +1.7, [−0.8, 3.9]; k=4: +6.8, [4.7, 9.0]). The multi-instance advantage of the VLMs
at 256 px disappears: capsules PatchCore k=16 90.1 vs Jev-Omni 82.3; macaroni1 90.8 vs 86.4.
So the pre-registered k\* = 8 is a property of anomalib's 256 px default, and the
"VLM wins on randomly posed products" reading was a resolution artifact.
EFFICIENTAD (Appendix B.5): their 94.3 used WRN-101, 224×224, coreset 1 %, no crop,
official patchcore-inspection code; our WRN-101 anchor at 256 / 10 % gave 90.5, so the
anchor gap is not resolution and remains unexplained (implementation is the candidate).

## Post-hoc addition: RSI-Jev v4.0-VL (1 Oct 2026)

Not pre-registered; contributed by the RSI-Jev authors after the run. RSI-Jev v4.0-VL
(Qwen3.5-2B base, `shgao/rsi-jev-v4.0-vl-qwen3.5-2b`) on config "single" only, both option
orders, the frozen prompt sent as a two-option choice to its `/v1/systemone` server
(`src/jev_inspection/rsijev.py`). The score is the log-odds of "defective" from the served,
calibrated probabilities; photos are scaled to fit 1,536 px. `jev-inspection score --rsijev`
adds it as R0 and writes `results/summary_rsijev.json`; `results/summary.json` is unchanged.
Macro AUROC 86.6 (85.1–88.1). It uses about 1,000 input tokens per photo against about 350
for A0/B0, VisA is not among its fine-tuning sources but its base model's pretraining data
cannot be checked, and it is below both A0 and B0 on chewinggum and pipe_fryum.
