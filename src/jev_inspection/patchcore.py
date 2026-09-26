"""PatchCore (anomalib's implementation) fitted on k good parts, scored on the VisA test set.

anomalib's PatchcoreModel is driven directly rather than through its Engine: the Engine wants a
datamodule per training subset, and the sweep needs 7 k values x 3 seeds x 12 categories.
Pre-processing matches anomalib's PatchCore default (resize to 256x256, ImageNet normalisation,
no centre crop).

Coreset: the full memory bank for k <= 64 (subsampling a bank built from a handful of images
only throws away the little PatchCore has), anomalib's default ratio 0.1 for k = all.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision.transforms import v2 as T

from . import visa

KS = (1, 2, 4, 8, 16, 64, "all")
IMAGE_SIZE = (256, 256)
TRANSFORM = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Resize(IMAGE_SIZE, antialias=True),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def coreset_ratio(k) -> float:
    return 0.1 if k == "all" else 1.0


def _batches(root: Path, paths: list[str], bs: int, device):
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        yield chunk, torch.stack([TRANSFORM(Image.open(root / p).convert("RGB")) for p in chunk]).to(device)


def fit(root: Path, train: list[str], k, seed: int, backbone: str, device: str, bs: int = 16):
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    torch.manual_seed(seed)  # KCenterGreedy starts from a random index
    np.random.seed(seed)
    model = PatchcoreModel(layers=("layer2", "layer3"), backbone=backbone, pre_trained=True,
                           num_neighbors=9).to(device)
    model.train()
    model.feature_extractor.eval()  # frozen backbone: keep BatchNorm in inference mode
    for _, x in _batches(root, train, bs, device):
        model(x)
    model.subsample_embedding(coreset_ratio(k))
    return model.eval()


@torch.inference_mode()
def score(model, root: Path, tests: list[visa.Item], device: str, bs: int = 16, maps_dir: Path | None = None):
    out = []
    for paths, x in _batches(root, [i.image for i in tests], bs, device):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3 / len(paths)
        for j, p in enumerate(paths):
            out.append({"image": p, "score": pred.pred_score[j].item(), "ms_forward_per_image": round(ms, 3)})
            if maps_dir is not None:
                m = pred.anomaly_map[j, 0].float().cpu().numpy()
                dst = maps_dir / (p.replace("/", "__") + ".npy")
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.save(dst, m.astype(np.float16))
    return out


def sweep(root: Path, out_dir: Path, categories, ks, seeds, backbone="wide_resnet50_2", device="cuda",
          maps_for: tuple | None = None):
    """One JSON line per (category, k, seed, test image); resumable per (category, k, seed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"patchcore_{backbone}.jsonl"
    done = set()
    if out.exists():
        done = {(r["obj"], str(r["k"]), r["seed"]) for r in map(json.loads, out.open())}
    items = visa.load(root)
    with out.open("a") as fh:
        for obj in categories:
            tests = [i for i in items if i.obj == obj and i.split == "test"]
            for k in ks:
                for seed in seeds:
                    # k=all has no draw to vary, only the coreset's random start: one seed, and
                    # these are the expensive fits (coreset selection over ~0.5-0.9M patches).
                    if k == "all" and seed != seeds[0]:
                        continue
                    if (obj, str(k), seed) in done:
                        continue
                    train = visa.patchcore_train(obj, k, seed, items)
                    t0 = time.perf_counter()
                    model = fit(root, train, k, seed, backbone, device)
                    fit_s = time.perf_counter() - t0
                    maps = out_dir / "maps" / f"{obj}_k{k}_s{seed}" if maps_for == (str(k), seed) else None
                    rows = score(model, root, tests, device, maps_dir=maps)
                    bank = int(model.memory_bank.shape[0])
                    by_img = {i.image: i for i in tests}
                    for r in rows:
                        it = by_img[r["image"]]
                        fh.write(json.dumps({"obj": obj, "k": k, "seed": seed, "backbone": backbone,
                                             "n_train": len(train), "coreset_ratio": coreset_ratio(k),
                                             "memory_bank": bank, "fit_s": round(fit_s, 2), "label": it.label,
                                             "types": it.types, **r}) + "\n")
                    fh.flush()
                    logger.info(f"[patchcore/{backbone}] {obj} k={k} seed={seed} n_train={len(train)} "
                                f"bank={bank} fit={fit_s:.1f}s")
                    del model
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
