"""Day-0 conveyor demo, step 1: detect and track every fruit, keep one crop per track.

Runs in its own environment so RF-DETR's pins never touch the measured stack:
  uv run --no-project --with rfdetr==1.11.0 --with supervision --with opencv-python-headless python scripts/demo_detect.py \
      clip.mp4 out/ --t0 4.5 --t1 10.75
Writes out/tracks.json (per-frame boxes with track ids) and out/crops/<track>.png.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.assets.coco_classes import COCO_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out", type=Path)
ap.add_argument("--t0", type=float, default=0.0)
ap.add_argument("--t1", type=float, default=1e9)
ap.add_argument("--classes", default="orange")
ap.add_argument("--threshold", type=float, default=0.4)
ap.add_argument("--min-frames", type=int, default=6, help="tracks shorter than this are dropped")
ap.add_argument("--max-area", type=float, default=0.06, help="drop boxes larger than this share of the frame")
ap.add_argument("--min-side", type=int, default=90, help="a crop smaller than this (px) is not judged")
a = ap.parse_args()
(a.out / "crops").mkdir(parents=True, exist_ok=True)

def isolate(rgb, x0, y0, x1, y1, margin=0.06):
    """The fruit alone: crop to its box, then grey out everything outside the inscribed ellipse,
    so a neighbour's blemish cannot answer a question about this fruit."""
    m = margin * max(x1 - x0, y1 - y0)
    H, W = rgb.shape[:2]
    X0, Y0, X1, Y1 = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))
    crop = rgb[Y0:Y1, X0:X1].copy()
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    cx, cy = (x0 + x1) / 2 - X0, (y0 + y1) / 2 - Y0
    rx, ry = (x1 - x0) / 2 * 1.04, (y1 - y0) / 2 * 1.04
    outside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 > 1
    crop[outside] = 128
    return crop


wanted = {i for i, n in COCO_CLASSES.items() if n in a.classes.split(",")}
model = RFDETRBase()
tracker = sv.ByteTrack()
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_MSEC, a.t0 * 1000)
frames, best = [], {}
i = 0
while True:
    ok, bgr = cap.read()
    t = a.t0 + i / fps
    if not ok or t > a.t1:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    det = model.predict(Image.fromarray(rgb), threshold=a.threshold)
    det = det[np.isin(det.class_id, list(wanted))]
    # One fruit, not a pile: a box covering a large share of the frame is a group detection.
    det = det[(det.box_area / (W * H)) < a.max_area]
    det = tracker.update_with_detections(det)
    boxes = []
    for (x0, y0, x1, y1), tid, conf in zip(det.xyxy, det.tracker_id, det.confidence):
        boxes.append({"id": int(tid), "box": [float(x0), float(y0), float(x1), float(y1)], "conf": float(conf)})
        # The crop a person would judge: the largest view of the fruit that is not cut by the frame edge.
        inside = x0 > 4 and y0 > 4 and x1 < W - 4 and y1 < H - 4
        area = (x1 - x0) * (y1 - y0)
        if inside and min(x1 - x0, y1 - y0) >= a.min_side and area > best.get(int(tid), (0, None))[0]:
            best[int(tid)] = (area, isolate(rgb, x0, y0, x1, y1), i)
    frames.append({"frame": i, "t": round(t, 3), "boxes": boxes})
    i += 1

counts = {}
for f in frames:
    for b in f["boxes"]:
        counts[b["id"]] = counts.get(b["id"], 0) + 1
keep = {tid for tid, n in counts.items() if n >= a.min_frames and tid in best}
for tid in keep:
    Image.fromarray(best[tid][1]).save(a.out / "crops" / f"{tid}.png")
for f in frames:
    f["boxes"] = [b for b in f["boxes"] if b["id"] in keep]
json.dump({"video": a.video, "fps": fps, "t0": a.t0, "size": [W, H], "frames": frames,
           "tracks": {str(t): {"frames": counts[t], "crop_frame": best[t][2]} for t in sorted(keep)}},
          open(a.out / "tracks.json", "w"))
print(f"{len(frames)} frames, {len(keep)} tracks kept of {len(counts)}")
