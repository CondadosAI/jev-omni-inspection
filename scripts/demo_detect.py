"""Day-0 conveyor demo, step 1: detect, track and count every item crossing a line, and keep one
isolated crop per item taken before it reaches the line (so a verdict exists when it crosses).

Runs in its own environment so RF-DETR's pins never touch the measured stack:
  uv run --no-project --with rfdetr==1.11.0 --with supervision --with opencv-python-headless \
      python scripts/demo_detect.py clip.mp4 out/ --classes "sports ball,orange,apple" --line 0.7
Writes out/tracks.json (per-frame boxes, per-track crossing frame) and out/crops/<track>.png.
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
ap.add_argument("--stride", type=int, default=1, help="use every n-th source frame")
ap.add_argument("--classes", default="orange", help="COCO names, comma-separated; used class-agnostically")
ap.add_argument("--threshold", type=float, default=0.15)
ap.add_argument("--line", type=float, default=0.7, help="counting line, as a fraction of frame height")
ap.add_argument("--min-frames", type=int, default=6, help="tracks shorter than this are dropped")
ap.add_argument("--max-area", type=float, default=0.06, help="drop boxes larger than this share of the frame")
ap.add_argument("--min-side", type=int, default=60, help="a crop smaller than this (px) is not judged")
a = ap.parse_args()
(a.out / "crops").mkdir(parents=True, exist_ok=True)


def isolate(rgb, x0, y0, x1, y1, margin=0.06):
    """The item alone: crop to its box, then grey out everything outside the inscribed ellipse,
    so a neighbour's blemish cannot answer a question about this item."""
    m = margin * max(x1 - x0, y1 - y0)
    H, W = rgb.shape[:2]
    X0, Y0, X1, Y1 = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))
    crop = rgb[Y0:Y1, X0:X1].copy()
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    cx, cy = (x0 + x1) / 2 - X0, (y0 + y1) / 2 - Y0
    rx, ry = (x1 - x0) / 2 * 1.04, (y1 - y0) / 2 * 1.04
    crop[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 > 1] = 128
    return crop


wanted = {i for i, n in COCO_CLASSES.items() if n in a.classes.split(",")}
model = RFDETRBase()
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) / a.stride
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# A COCO detector scores off-class items low (limes come out as "sports ball" at 0.05-0.45), and
# supervision's ByteTrack multiplies IoU by that score when matching, so tracks die after one frame.
# Scores are stretched by SCORE_GAIN before tracking (the ranking is unchanged); a track starts at
# track_activation_threshold + 0.1 inside ByteTrack, so that offset is subtracted back.
SCORE_GAIN = 5.0
tracker = sv.ByteTrack(track_activation_threshold=min(1.0, a.threshold * SCORE_GAIN) - 0.1,
                       lost_track_buffer=int(fps / 2), frame_rate=round(fps))
line_y = int(a.line * H)
line = sv.LineZone(start=sv.Point(0, line_y), end=sv.Point(W, line_y))
cap.set(cv2.CAP_PROP_POS_MSEC, a.t0 * 1000)
frames, best, crossed, src = [], {}, {}, 0
i = 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    src += 1
    if (src - 1) % a.stride:
        continue
    t = a.t0 + i / fps
    if t > a.t1:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    det = model.predict(Image.fromarray(rgb), threshold=a.threshold)
    det = det[np.isin(det.class_id, list(wanted))]
    det = det[(det.box_area / (W * H)) < a.max_area]  # one item, not a pile
    det = det.with_nms(threshold=0.5, class_agnostic=True)  # one item scored as two COCO classes
    det.confidence = np.clip(det.confidence * SCORE_GAIN, 0, 1)
    det = tracker.update_with_detections(det)
    c_in, c_out = line.trigger(det)
    boxes = []
    for j, ((x0, y0, x1, y1), tid, conf) in enumerate(zip(det.xyxy, det.tracker_id, det.confidence)):
        tid = int(tid)
        if (c_in[j] or c_out[j]) and tid not in crossed:
            crossed[tid] = i
        boxes.append({"id": tid, "box": [float(x0), float(y0), float(x1), float(y1)], "conf": float(conf)})
        # The crop the model judges: the largest view before the line, not cut by the frame edge.
        inside = x0 > 4 and x1 < W - 4 and y0 > 4 and y1 < H - 4
        before = tid not in crossed or crossed[tid] == i
        area = (x1 - x0) * (y1 - y0)
        if inside and before and min(x1 - x0, y1 - y0) >= a.min_side and area > best.get(tid, (0, None, 0))[0]:
            best[tid] = (area, isolate(rgb, x0, y0, x1, y1), i)
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
json.dump({"video": Path(a.video).name, "fps": fps, "stride": a.stride, "t0": a.t0, "size": [W, H],
           "line_y": line_y, "classes": a.classes, "threshold": a.threshold, "frames": frames,
           "tracks": {str(t): {"frames": counts[t], "crop_frame": best[t][2], "crossed": crossed.get(t)}
                      for t in sorted(keep)}},
          open(a.out / "tracks.json", "w"))
n_cross = sum(1 for t in keep if t in crossed)
print(f"{len(frames)} frames, {len(keep)} tracks kept of {len(counts)}, {n_cross} crossed the line")
