"""PatchCore anomaly maps at 512 x 512 (k = 16, seed 0) for the images shown in the VisA video.
  uv run python scripts/demo_maps.py results/demo/visa-maps <image> [<image> ...]
"""
import sys
from pathlib import Path

from jev_inspection import patchcore, visa

out, images = Path(sys.argv[1]), sys.argv[2:]
patchcore.set_image_size(512)
items = visa.load()
for obj in sorted({p.split("/")[0] for p in images}):
    model = patchcore.fit(visa.DEFAULT_ROOT, visa.patchcore_train(obj, 16, 0, items), 16, 0, "wide_resnet50_2", "cuda")
    tests = [i for i in items if i.image in images and i.obj == obj]
    for r in patchcore.score(model, visa.DEFAULT_ROOT, tests, "cuda", maps_dir=out):
        print(r["image"], round(r["score"], 2))
