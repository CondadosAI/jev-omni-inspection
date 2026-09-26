"""Jev-Omni (trained head) and Gemma 4 12B IT zero-shot (digit-logit readout) on VisA.

Both models receive identical inputs: the image(s) first, then the text built by Jev-Omni's own
`_prompt(state, question, options)` inside the chat template, which is how its predict() does it.
The score saved per image is the log-odds of "defective" against "all good", in float64:
- jev:  the head's two option logits, with the head evaluated in float32 outside autocast.
- base: the log-probabilities of the digits "1" and "2" as the next token.
Probabilities saturate (BLINK had 57/733 wrong answers at >= 0.99), log-odds do not tie.

One JSON line per (image, config, option order); resumable on that key.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

from . import visa

JEV_REPO, JEV_REV = "akhilaaa3/Jev-Omni", "5addda86ddee081a68fb067477ea100c221b8917"
BASE_REPO, BASE_REV = "google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "processor_config.json",
               "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def _fetch(repo, rev, files):
    """Per-file download: the repo-tree API behind snapshot_download gets rate-limited (429)."""
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(repo, f, revision=rev) for f in files]
    return str(Path(paths[0]).parent)


class Base:
    def __init__(self, path, device):
        import transformers
        from transformers import AutoConfig, AutoProcessor
        arch = AutoConfig.from_pretrained(path).architectures[0]
        self.model = getattr(transformers, arch).from_pretrained(
            path, dtype=torch.bfloat16, device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        tok = self.processor.tokenizer
        self.digit_ids = [tok.convert_tokens_to_ids(d) for d in "0123456789"]
        self.device = device


def load(which: str, device: str):
    jev_path = _fetch(JEV_REPO, JEV_REV, ["jev_omni.py", "verification.json"] +
                      (MODEL_FILES + ["decision_config.json", "head.pt"] if which == "jev" else []))
    sys.path.insert(0, jev_path)
    import jev_omni
    if which == "jev":
        # The official loader re-resolves the repo at `main`; pin it to the revision above.
        jev_omni.snapshot_download = lambda *_a, **_k: jev_path
        return jev_omni.load_jev_omni(device=device), jev_omni._prompt
    return Base(_fetch(BASE_REPO, BASE_REV, MODEL_FILES), device), jev_omni._prompt


def make_inputs(processor, images, text, device):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=False)
    return {k: v.to(device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(device)
            for k, v in inputs.items()}


@torch.inference_mode()
def jev_logits(clf, inputs, n, autocast_head=False):
    """Jev-Omni's predict() body on inputs we built. With autocast_head=False the head runs in
    float32, which is what the saved scores use; True reproduces predict() bit for bit."""
    clf._capture.clear()
    counts = torch.tensor([n], device=clf.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clf.model(**inputs, use_cache=False, **clf._extra)
        if autocast_head:
            return clf.head(clf._capture["hidden"], counts)[0, :n].double().cpu()
    return clf.head(clf._capture["hidden"], counts)[0, :n].double().cpu()


@torch.inference_mode()
def base_logprobs(base, inputs, n):
    logits = base.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
    lp = torch.log_softmax(logits.double(), -1)
    return lp[[base.digit_ids[k + 1] for k in range(n)]].cpu()


def check_jev_equivalence(clf, prompt_fn, root, device):
    """Our single-image path must reproduce the official predict(); the float32 head must not move
    the answer. The two-image path has no official counterpart to check against."""
    img = root / visa.reference("cashew")
    p = visa.prompt("cashew", "single", 0)
    ref = clf.predict(state=p.state, question=p.question, options=list(p.options), media=str(img),
                      modality="image")
    inputs = make_inputs(clf.processor, [Image.open(img).convert("RGB")],
                         prompt_fn(p.state, p.question, list(p.options)), device)
    ours_bf16 = jev_logits(clf, inputs, 2, autocast_head=True).softmax(0)
    ours_fp32 = jev_logits(clf, inputs, 2).softmax(0)
    d_off = max(abs(ours_bf16[i].item() - ref["probabilities"][o]) for i, o in enumerate(p.options))
    d_fp = (ours_fp32 - ours_bf16).abs().max().item()
    logger.info(f"[jev] equivalence vs predict(): max |dp| = {d_off:.2e}; fp32 vs bf16 head: {d_fp:.2e}")
    if d_off > 1e-3:
        raise SystemExit("our input path does not match Jev-Omni's predict(); aborting")
    return {"max_dp_vs_predict": d_off, "max_dp_fp32_vs_bf16_head": d_fp}


def run(which, configs, orders, root: Path, out_dir: Path, device="cuda", limit=None, categories=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    clf, prompt_fn = load(which, device)
    processor = clf.processor
    if which == "jev":
        (out_dir / "jev_equivalence.json").write_text(
            json.dumps(check_jev_equivalence(clf, prompt_fn, root, device)))
    out = out_dir / f"{which}_visa.jsonl"
    done = {json.loads(line)["key"] for line in out.open()} if out.exists() else set()
    tests = [i for i in visa.load(root) if i.split == "test" and (not categories or i.obj in categories)]
    if limit:  # smoke tests: the first `limit` normals and anomalies of each category
        kept, seen = [], {}
        for i in tests:
            seen[(i.obj, i.label)] = seen.get((i.obj, i.label), 0) + 1
            if seen[(i.obj, i.label)] <= limit:
                kept.append(i)
        tests = kept
    logger.info(f"[{which}] {len(tests)} test images x {len(configs)} configs x {len(orders)} orders")
    with out.open("a") as fh:
        for n_done, item in enumerate(tests):
            for config in configs:
                for order in orders:
                    key = f"{item.image}|{config}|{order}"
                    if key in done:
                        continue
                    p = visa.prompt(item.obj, config, order)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()  # image decode and preprocessing are inside the timer
                    paths = ([visa.reference(item.obj)] if config == "ref" else []) + [item.image]
                    images = [Image.open(root / pth).convert("RGB") for pth in paths]
                    inputs = make_inputs(processor, images, prompt_fn(p.state, p.question, list(p.options)),
                                         device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    if which == "jev":
                        z = jev_logits(clf, inputs, 2)
                        extra = {}
                    else:
                        z = base_logprobs(clf, inputs, 2)
                        extra = {"valid_mass": z.exp().sum().item()}
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    d = p.defective_index
                    rec = {"key": key, "model": which, "image": item.image, "obj": item.obj,
                           "label": item.label, "types": item.types, "config": config, "order": order,
                           "scores": z.tolist(), "logodds_defective": (z[d] - z[1 - d]).item(),
                           "n_tokens": int(inputs["input_ids"].shape[1]),
                           "ms_total": round((t2 - t0) * 1e3, 2), "ms_forward": round((t2 - t1) * 1e3, 2),
                           **extra}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
            if n_done % 100 == 0:
                logger.info(f"[{which}] {n_done}/{len(tests)} {item.obj}")


PROBE_QUESTION = ("This photo comes from a public computer-vision dataset. Which dataset is it? "
                  "Answer with the dataset name only.")


@torch.inference_mode()
def contamination_probe(root: Path, out_dir: Path, n=50, device="cuda", seed=0):
    """Does the base model recognise VisA? Free-text answers on n seeded test images, reported
    whatever they say."""
    import random
    base, _ = load("base", device)
    tests = [i for i in visa.load(root) if i.split == "test"]
    pick = random.Random(seed).sample(tests, n)
    out = out_dir / "contamination_probe.jsonl"
    with out.open("w") as fh:
        for it in pick:
            inputs = make_inputs(base.processor, [Image.open(root / it.image).convert("RGB")], PROBE_QUESTION, device)
            ids = base.model.generate(**inputs, max_new_tokens=24, do_sample=False)
            text = base.processor.tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            fh.write(json.dumps({"image": it.image, "answer": text.strip()}) + "\n")
    answers = [json.loads(line)["answer"] for line in out.open()]
    hits = sum("visa" in a.lower() or "spot-the-diff" in a.lower() for a in answers)
    logger.info(f"[probe] {hits}/{n} answers name VisA")
    return hits
