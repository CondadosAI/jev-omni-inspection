"""Step 0 for the VisA inspection post: counts, defect vocabulary, image sizes, and the
frozen seeds (reference image per category, PatchCore k-subsets).

Usage: python visa_step0.py <VisA root> <spot-diff checkout> <out dir>
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(sys.argv[2]) / "utils"))
from id2class import id2class_map  # noqa: E402

root, spotdiff, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

KS = [1, 2, 4, 8, 16, 64]
SEEDS = [0, 1, 2]

rows = list(csv.DictReader(open(spotdiff / "split_csv/1cls.csv")))
counts = defaultdict(Counter)
train_normals = defaultdict(list)
for r in rows:
    counts[r["object"]][f'{r["split"]}_{r["label"]}'] += 1
    if r["split"] == "train":
        train_normals[r["object"]].append(r["image"])

summary = {}
for obj in sorted(counts):
    anno = list(csv.DictReader(open(root / obj / "image_anno.csv")))
    labels = [a["label"].strip() for a in anno if a["label"].strip() and a["label"].strip() != "normal"]
    per_type = Counter()
    multi = 0
    for lab in labels:
        types = [t.strip() for t in lab.split(",") if t.strip()]
        multi += len(types) > 1
        per_type.update(types)
    sizes = Counter()
    for a in anno[:: max(1, len(anno) // 20)]:
        with Image.open(root / a["image"]) as im:
            sizes[f"{im.width}x{im.height}"] += 1
    summary[obj] = {
        "split_1cls": dict(counts[obj]),
        "anno_columns": list(anno[0].keys()),
        "anomalous_in_anno": len(labels),
        "multi_type_images": multi,
        "type_counts": dict(per_type.most_common()),
        "id2class": [v for k, v in id2class_map[obj].items() if k != "0"],
        "sampled_sizes": dict(sizes),
    }

# Frozen draws. Sorted input so the draw depends only on the seed, not on CSV order.
frozen = {"ks": KS, "seeds": SEEDS, "reference": {}, "patchcore_subsets": {}}
for obj, imgs in sorted(train_normals.items()):
    imgs = sorted(imgs)
    frozen["reference"][obj] = random.Random(f"ref-{obj}").choice(imgs)
    # Nested subsets: the k=4 draw contains the k=2 draw, so the k-curve for one seed
    # adds good parts rather than swapping them. The reference image is excluded so
    # PatchCore at k=1 and the VLM at k=1 never share the same good part by accident.
    pool = [i for i in imgs if i != frozen["reference"][obj]]
    frozen["patchcore_subsets"][obj] = {}
    for s in SEEDS:
        draw = random.Random(f"pc-{obj}-{s}").sample(pool, max(KS))
        frozen["patchcore_subsets"][obj][str(s)] = {str(k): draw[:k] for k in KS}

json.dump(summary, open(out / "visa_summary.json", "w"), indent=1)
json.dump(frozen, open(out / "frozen_draws.json", "w"), indent=1)

tot = Counter()
print(f"{'object':12} {'train_n':>7} {'test_n':>6} {'test_a':>6} {'multi':>5}  sizes")
for obj, s in summary.items():
    c = s["split_1cls"]
    tot.update(c)
    print(f"{obj:12} {c.get('train_normal',0):7} {c.get('test_normal',0):6} {c.get('test_anomaly',0):6} "
          f"{s['multi_type_images']:5}  {s['sampled_sizes']}")
print("total", dict(tot))
