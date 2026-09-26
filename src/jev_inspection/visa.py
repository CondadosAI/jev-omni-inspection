"""VisA on the official 1cls split, and the frozen prompt that every VLM run uses.

The prompt text below was frozen on 25 Sep 2026 before any VLM saw a VisA image. Changing
it invalidates every result produced with it.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(os.environ.get("VISA_ROOT", Path.home() / ".condados-jev/visa"))
FROZEN = REPO / "data/frozen_draws.json"

ARCHIVE_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
ARCHIVE_SHA256 = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"

CATEGORIES = ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
              "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")

# The README's own grouping, used for the per-group breakdown.
GROUPS = {
    "pcb": ("pcb1", "pcb2", "pcb3", "pcb4"),
    "multiple_instances": ("candle", "capsules", "macaroni1", "macaroni2"),
    "single_instance": ("cashew", "chewinggum", "fryum", "pipe_fryum"),
}

# --- frozen prompt -------------------------------------------------------------------------
PHRASES = {
    "candle": "candles",
    "capsules": "green gel capsules",
    "cashew": "a cashew nut",
    "chewinggum": "a piece of chewing gum",
    "fryum": "a wheel-shaped fried snack",
    "macaroni1": "macaroni pieces",
    "macaroni2": "macaroni pieces",
    "pcb1": "a printed circuit board",
    "pcb2": "a printed circuit board",
    "pcb3": "a printed circuit board",
    "pcb4": "a printed circuit board",
    "pipe_fryum": "a pipe-shaped fried snack",
}
QUESTION = "Is everything in the photo good, or is at least one part defective?"
GOOD, DEFECTIVE = "all good", "defective"
CONFIGS = ("single", "ref")  # single image, or a known-good reference followed by the test image
ORDERS = (0, 1)  # 0: [good, defective], 1: [defective, good]
# -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    state: str
    question: str
    options: tuple[str, str]
    defective_index: int


def prompt(obj: str, config: str, order: int) -> Prompt:
    phrase = PHRASES[obj]
    if config == "single":
        state = f"Production-line inspection photo of {phrase}."
    elif config == "ref":
        state = (f"Production-line inspection photos of {phrase}. The first image is a known-good "
                 "example. The second image is the one under inspection.")
    else:
        raise ValueError(config)
    options = (GOOD, DEFECTIVE) if order == 0 else (DEFECTIVE, GOOD)
    return Prompt(state, QUESTION, options, options.index(DEFECTIVE))


@dataclass
class Item:
    obj: str
    split: str  # train | test
    label: str  # normal | anomaly
    image: str  # relative to the VisA root
    mask: str
    types: list[str] = field(default_factory=list)

    @property
    def is_anomaly(self) -> bool:
        return self.label == "anomaly"


def load(root: Path = DEFAULT_ROOT) -> list[Item]:
    """Every image of the official 1cls split, with its defect types from image_anno.csv."""
    types = {}
    for obj in CATEGORIES:
        for a in csv.DictReader(open(root / obj / "image_anno.csv")):
            lab = a["label"].strip()
            types[a["image"]] = [] if lab == "normal" else [t.strip() for t in lab.split(",") if t.strip()]
    items = [Item(r["object"], r["split"], r["label"], r["image"], r["mask"], types.get(r["image"], []))
             for r in csv.DictReader(open(root / "split_csv/1cls.csv"))]
    missing = [i.image for i in items if i.is_anomaly and not i.types]
    if missing:
        raise ValueError(f"{len(missing)} anomalies without a defect type, e.g. {missing[0]}")
    return items


def frozen() -> dict:
    return json.loads(FROZEN.read_text())


def reference(obj: str) -> str:
    return frozen()["reference"][obj]


def patchcore_train(obj: str, k: int | str, seed: int, items: list[Item]) -> list[str]:
    """The k good parts PatchCore sees. `all` is the same pool the draws come from: every
    train normal except the VLM's reference image."""
    fr = frozen()
    if k == "all":
        ref = fr["reference"][obj]
        return sorted(i.image for i in items if i.obj == obj and i.split == "train" and i.image != ref)
    return fr["patchcore_subsets"][obj][str(seed)][str(k)]
