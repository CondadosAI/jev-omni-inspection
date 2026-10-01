"""RSI-Jev v4.0-VL through its HTTP server. Added after the pre-registered run (see DESIGN.md).

RSI-Jev serves Jev's `POST /v1/systemone` request, plus an `images` list of data URLs. It is a
separate model (Qwen3.5-2B base), so it cannot go through the in-process path in vlm.py; this
module sends the same frozen prompt over HTTP instead:

    rsi-jev serve shgao/rsi-jev-v4.0-vl-qwen3.5-2b --port 8000
    jev-inspection vlm --model rsijev --server http://localhost:8000

The state and question are the frozen ones from visa.prompt(); the two options go in as a
choice question in the listed order, so both option orders are asked as for the other VLMs.
The score saved per image is the log-odds of "defective" against "all good" from the served
(calibrated) probabilities. Photos are scaled to fit 1,536 px and sent as PNG.

One JSON line per (image, config, option order); resumable on that key. Single-image config only.
"""
from __future__ import annotations

import base64
import io
import json
import math
import urllib.request
from pathlib import Path

from loguru import logger
from PIL import Image

from . import visa

MAX_SIDE = 1536
LETTERS = "AB"


def data_url(path: Path) -> str:
    im = Image.open(path)
    im.draft("RGB", (MAX_SIDE, MAX_SIDE))
    im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ask(server: str, model: str, state: str, question: str, options: list[str], image_url: str):
    body = {"model": model, "state": state, "images": [image_url],
            "questions": {"inspection": {"type": "choice", "instructions": question,
                                         "criteria": dict(zip(LETTERS, options))}}}
    req = urllib.request.Request(server.rstrip("/") + "/v1/systemone", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    probs = resp["answers"]["inspection"]["probabilities"]
    return [probs[k] for k in LETTERS[:len(options)]], resp.get("usage", {}).get("input_tokens")


def run(server, model, root: Path, out_dir: Path, limit=None, categories=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "rsijev_visa.jsonl"
    done = {json.loads(line)["key"] for line in out.open()} if out.exists() else set()
    tests = [i for i in visa.load(root) if i.split == "test" and (not categories or i.obj in categories)]
    if limit:
        kept, seen = [], {}
        for i in tests:
            seen[(i.obj, i.label)] = seen.get((i.obj, i.label), 0) + 1
            if seen[(i.obj, i.label)] <= limit:
                kept.append(i)
        tests = kept
    logger.info(f"[rsijev] {len(tests)} test images x 2 orders via {server}")
    with out.open("a") as fh:
        for n_done, item in enumerate(tests):
            url = None
            for order in visa.ORDERS:
                key = f"{item.image}|single|{order}"
                if key in done:
                    continue
                url = url or data_url(root / item.image)
                p = visa.prompt(item.obj, "single", order)
                probs, n_tokens = ask(server, model, p.state, p.question, list(p.options), url)
                lp = [math.log(max(x, 1e-12)) for x in probs]
                d = p.defective_index
                rec = {"key": key, "model": "rsijev", "image": item.image, "obj": item.obj,
                       "label": item.label, "types": item.types, "config": "single", "order": order,
                       "scores": lp, "logodds_defective": lp[d] - lp[1 - d], "n_tokens": n_tokens}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            if n_done % 100 == 0:
                logger.info(f"[rsijev] {n_done}/{len(tests)} {item.obj}")
