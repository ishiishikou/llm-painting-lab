#!/usr/bin/env python3
"""v11: reorder the complete proven stroke set without selecting or rewriting strokes.

The source painting is the complete v6 artifact-cleanup stroke document.  Every
source stroke must appear exactly once in the output with identical JSON content.
The only permitted operation is changing sequence.

A current MediaPipe Face Landmarker model runs on the *reference image* on CPU.
Its landmarks are used only to semantically label already-existing strokes (eye,
nose, mouth, face) so the scheduler can prefer a human-like coarse-to-fine order.
Landmarks never move, resize, recolor, replace, add or delete a stroke.

Because alpha compositing and canvas-dependent brushes are order-sensitive, v11
preserves the original relative order of strokes whose conservative paint bounds
can interact. Mixer/smudge strokes are global barriers. This leaves independent
work free to move while keeping the finished raster identical. CI verifies exact
pixel equality and exact stroke-multiset equality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageChops, ImageDraw

import reference_to_brush_process as base


PHASE_DEFAULT = ["composition", "silhouette", "light_shadow", "face_structure", "detail", "finish"]
SEMANTIC_DEFAULT = ["background", "subject", "headwrap", "garment", "face", "nose", "mouth", "eye", "pearl", "accent"]
CANVAS_DEPENDENT = {"mixerBrush", "smudgeBrush"}

# MediaPipe Face Landmarker canonical indices. These are used only to make
# semantic neighborhoods for ordering; no stroke geometry is derived from them.
EYE_A = [33, 133, 159, 145, 160, 144]
EYE_B = [362, 263, 386, 374, 387, 373]
NOSE = [1, 2, 4, 5, 168]
MOUTH = [13, 14, 61, 291, 78, 308]


def canonical_stroke(s):
    return json.dumps(s, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stroke_digest(s):
    return hashlib.sha256(canonical_stroke(s).encode("utf-8")).hexdigest()


def multiset_digest(strokes):
    counts = Counter(stroke_digest(s) for s in strokes)
    payload = "\n".join(f"{k}:{counts[k]}" for k in sorted(counts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), counts


def stroke_center(s):
    x1 = float(s.get("x1", s.get("x", 0.0)))
    y1 = float(s.get("y1", s.get("y", 0.0)))
    x2 = float(s.get("x2", x1))
    y2 = float(s.get("y2", y1))
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def stroke_width(s):
    vals = []
    for k in ("width", "widthStart", "widthEnd", "size", "radius", "bristleWidth"):
        v = s.get(k)
        if isinstance(v, (int, float)):
            vals.append(abs(float(v)))
    return max(vals) if vals else 4.0


def conservative_bbox(s, canvas_w, canvas_h):
    x1 = float(s.get("x1", s.get("x", 0.0)))
    y1 = float(s.get("y1", s.get("y", 0.0)))
    x2 = float(s.get("x2", x1))
    y2 = float(s.get("y2", y1))
    w = stroke_width(s)
    brush = s.get("brush", "")
    extra = 10.0 + w * 1.35
    if brush == "dryBrush":
        extra += 10.0
    elif brush == "glaze":
        extra += 6.0
    return (
        max(0.0, min(x1, x2) - extra),
        max(0.0, min(y1, y2) - extra),
        min(float(canvas_w - 1), max(x1, x2) + extra),
        min(float(canvas_h - 1), max(y1, y2) + extra),
    )


def mean_point(points, indices):
    chosen = [points[i] for i in indices if i < len(points)]
    if not chosen:
        return None
    return (
        sum(float(p.x) for p in chosen) / len(chosen),
        sum(float(p.y) for p in chosen) / len(chosen),
    )


def detect_landmarks(reference_path, model_path, width, height):
    ref = base.crop_resize(Image.open(reference_path), width, height).convert("RGB")
    arr = np.ascontiguousarray(np.asarray(ref, dtype=np.uint8))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
    )
    with FaceLandmarker.create_from_options(opts) as detector:
        result = detector.detect(image)
    if not result.face_landmarks:
        raise SystemExit("MediaPipe Face Landmarker did not detect a face")
    pts = result.face_landmarks[0]
    xs = [float(p.x) for p in pts]
    ys = [float(p.y) for p in pts]
    info = {
        "engine": "MediaPipe Face Landmarker",
        "delegate": "CPU",
        "landmark_count": len(pts),
        "face_bbox_normalized": [min(xs), min(ys), max(xs), max(ys)],
        "eye_a": mean_point(pts, EYE_A),
        "eye_b": mean_point(pts, EYE_B),
        "nose": mean_point(pts, NOSE),
        "mouth": mean_point(pts, MOUTH),
    }
    return ref, info


def normalized_center(s, width, height):
    x, y = stroke_center(s)
    return x / max(1.0, width), y / max(1.0, height)


def elliptical_near(p, q, rx, ry):
    if q is None:
        return False
    return ((p[0] - q[0]) / rx) ** 2 + ((p[1] - q[1]) / ry) ** 2 <= 1.0


def semantic_label(s, landmarks, width, height):
    role = str(s.get("role", "")).lower()
    brush = str(s.get("brush", ""))
    if "pearl" in role:
        return "pearl"
    if "background" in role or role == "bg":
        return "background"
    if "headwrap" in role or "turban" in role or "cloth" in role:
        return "headwrap"
    if "garment" in role or "collar" in role or "shoulder" in role:
        return "garment"

    p = normalized_center(s, width, height)
    fb = landmarks.get("face_bbox_normalized")
    in_face = False
    if fb:
        x0, y0, x1, y1 = fb
        padx = (x1 - x0) * 0.12
        pady = (y1 - y0) * 0.12
        in_face = x0 - padx <= p[0] <= x1 + padx and y0 - pady <= p[1] <= y1 + pady

    if in_face or "face" in role or role in {"eye-line", "mouth-line"}:
        # Neighbourhood sizes are intentionally loose. They only decide ordering.
        if elliptical_near(p, landmarks.get("eye_a"), 0.075, 0.055) or elliptical_near(p, landmarks.get("eye_b"), 0.075, 0.055):
            return "eye"
        if elliptical_near(p, landmarks.get("nose"), 0.065, 0.09):
            return "nose"
        if elliptical_near(p, landmarks.get("mouth"), 0.085, 0.055):
            return "mouth"
        return "face"

    if brush in {"glaze"} or "accent" in role or "highlight" in role:
        return "accent"
    if role:
        return "subject"
    return "subject"


def work_rank(s):
    w = stroke_width(s)
    if w >= 16:
        return 0
    if w >= 7:
        return 1
    return 2


def cells_for_bbox(box, cell):
    x0, y0, x1, y1 = box
    gx0, gy0 = int(math.floor(x0 / cell)), int(math.floor(y0 / cell))
    gx1, gy1 = int(math.floor(x1 / cell)), int(math.floor(y1 / cell))
    for gy in range(gy0, gy1 + 1):
        for gx in range(gx0, gx1 + 1):
            yield gx, gy


def reorder_slice(strokes, global_start, priorities, semantic_labels, width, height, cell_size):
    n = len(strokes)
    if n <= 1:
        return list(strokes), list(range(global_start, global_start + n))

    edges = [[] for _ in range(n)]
    indeg = [0] * n
    last_for_cell = {}
    boxes = [conservative_bbox(s, width, height) for s in strokes]
    for i, box in enumerate(boxes):
        preds = set()
        for c in cells_for_bbox(box, cell_size):
            prev = last_for_cell.get(c)
            if prev is not None and prev != i:
                preds.add(prev)
            last_for_cell[c] = i
        for p in preds:
            edges[p].append(i)
            indeg[i] += 1

    phase_rank = priorities["phase_rank"]
    semantic_rank = priorities["semantic_rank"]
    heap = []
    for i, d in enumerate(indeg):
        if d == 0:
            s = strokes[i]
            key = (
                phase_rank.get(s.get("phase"), 999),
                work_rank(s),
                semantic_rank.get(semantic_labels[global_start + i], 999),
                global_start + i,
            )
            heapq.heappush(heap, (key, i))

    order = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(i)
        for j in edges[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                s = strokes[j]
                key = (
                    phase_rank.get(s.get("phase"), 999),
                    work_rank(s),
                    semantic_rank.get(semantic_labels[global_start + j], 999),
                    global_start + j,
                )
                heapq.heappush(heap, (key, j))
    if len(order) != n:
        raise SystemExit("dependency graph unexpectedly contains a cycle")
    return [strokes[i] for i in order], [global_start + i for i in order]


def reorder_with_barriers(strokes, labels, config, width, height):
    phase_order = config.get("human_order", {}).get("phase_priority", PHASE_DEFAULT)
    semantic_order = config.get("human_order", {}).get("semantic_priority", SEMANTIC_DEFAULT)
    priorities = {
        "phase_rank": {v: i for i, v in enumerate(phase_order)},
        "semantic_rank": {v: i for i, v in enumerate(semantic_order)},
    }
    out = []
    source_indices = []
    cell = int(config.get("dependency_cell_size", 8))
    start = 0
    for i, s in enumerate(strokes):
        if s.get("brush") not in CANVAS_DEPENDENT:
            continue
        part, idxs = reorder_slice(strokes[start:i], start, priorities, labels, width, height, cell)
        out.extend(part)
        source_indices.extend(idxs)
        # Canvas-dependent brush is a hard global barrier and keeps its exact slot
        # relative to every stroke before/after it.
        out.append(s)
        source_indices.append(i)
        start = i + 1
    part, idxs = reorder_slice(strokes[start:], start, priorities, labels, width, height, cell)
    out.extend(part)
    source_indices.extend(idxs)
    return out, source_indices


def save_landmark_overlay(ref, landmarks, path):
    im = ref.copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    fb = landmarks.get("face_bbox_normalized")
    if fb:
        d.rectangle((fb[0] * w, fb[1] * h, fb[2] * w, fb[3] * h), outline="white", width=2)
    for label, key in (("E1", "eye_a"), ("E2", "eye_b"), ("N", "nose"), ("M", "mouth")):
        p = landmarks.get(key)
        if not p:
            continue
        x, y = p[0] * w, p[1] * h
        r = 6
        d.ellipse((x-r, y-r, x+r, y+r), outline="white", width=2)
        d.text((x+r+2, y-r-2), label, fill="white")
    im.save(path)


def make_checkpoints(data, out_dir, fractions):
    strokes = data["strokes"]
    canvas = data["canvas"]
    width, height = int(canvas["width"]), int(canvas["height"])
    bg = canvas["background"]
    cp = out_dir / "checkpoints"
    cp.mkdir(parents=True, exist_ok=True)
    images = []
    for i, frac in enumerate(fractions, 1):
        count = max(1, min(len(strokes), int(round(len(strokes) * frac))))
        im = base.render(strokes[:count], width, height, bg)
        p = cp / f"{i:02d}-{int(frac*100):03d}pct.png"
        im.save(p)
        images.append((f"{int(frac*100)}% / {count}", im))
    tw = 216
    th = round(height * tw / width)
    cols = 4
    rows = math.ceil(len(images) / cols)
    board = Image.new("RGB", (tw * cols, (th + 28) * rows), (238, 238, 238))
    d = ImageDraw.Draw(board)
    for i, (label, im) in enumerate(images):
        x = (i % cols) * tw
        y = (i // cols) * (th + 28)
        d.text((x + 6, y + 6), label, fill="black")
        board.paste(im.resize((tw, th)), (x, y + 28))
    board.save(out_dir / "order-checkpoints.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_strokes")
    ap.add_argument("reference")
    ap.add_argument("model")
    ap.add_argument("config")
    ap.add_argument("--out-dir", default="artifact/v11")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(args.source_strokes).read_text(encoding="utf-8"))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    strokes = source["strokes"]
    canvas = source["canvas"]
    width, height = int(canvas["width"]), int(canvas["height"])
    bg = canvas["background"]

    ref, landmarks = detect_landmarks(args.reference, args.model, width, height)
    labels = [semantic_label(s, landmarks, width, height) for s in strokes]
    reordered, source_indices = reorder_with_barriers(strokes, labels, config, width, height)

    # Hard invariant: no selection, rejection, mutation, addition or deletion.
    source_digest, source_counts = multiset_digest(strokes)
    ordered_digest, ordered_counts = multiset_digest(reordered)
    if len(reordered) != len(strokes):
        raise SystemExit(f"stroke count changed: {len(strokes)} -> {len(reordered)}")
    if source_digest != ordered_digest or source_counts != ordered_counts:
        raise SystemExit("stroke multiset changed; v11 permits ordering only")

    original_final = base.render(strokes, width, height, bg)
    reordered_final = base.render(reordered, width, height, bg)
    diff = ImageChops.difference(original_final.convert("RGB"), reordered_final.convert("RGB"))
    bbox = diff.getbbox()
    extrema = diff.getextrema()
    max_channel_diff = max(v[1] for v in extrema)
    if bbox is not None or max_channel_diff != 0:
        # Fail closed. If a conservative bound was insufficient, we do not accept a
        # visually changed final painting just to obtain a nicer replay order.
        original_final.save(out / "original-final.png")
        reordered_final.save(out / "reordered-final-FAILED.png")
        diff.save(out / "final-diff-FAILED.png")
        raise SystemExit(f"final raster changed; bbox={bbox}, max_channel_diff={max_channel_diff}")

    result = copy.deepcopy(source)
    result["strokes"] = reordered
    result.setdefault("metadata", {}).update({
        "slug": "order-only-v11",
        "title": "完成画を変えずに全ストロークの順序だけを人間寄りに再配置",
        "source_mode": "complete-v6-stroke-set-order-only",
        "selection_allowed": False,
        "all_source_strokes_preserved": True,
        "geometry_changed": False,
        "final_raster_identical": True,
        "landmark_engine": "MediaPipe Face Landmarker / CPU",
    })
    (out / "strokes.reordered.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    original_final.save(out / "original-final.png")
    reordered_final.save(out / "final.png")
    ref.save(out / "reference.png")
    save_landmark_overlay(ref, landmarks, out / "landmark-overlay.png")
    (out / "landmarks.json").write_text(json.dumps(landmarks, ensure_ascii=False, indent=2), encoding="utf-8")

    moved = sum(1 for new_i, old_i in enumerate(source_indices) if new_i != old_i)
    displacement = [abs(new_i - old_i) for new_i, old_i in enumerate(source_indices)]
    label_counts = Counter(labels)
    first_positions = {}
    for pos, old_i in enumerate(source_indices):
        label = labels[old_i]
        first_positions.setdefault(label, pos)
    report = {
        "source_stroke_count": len(strokes),
        "output_stroke_count": len(reordered),
        "source_multiset_digest": source_digest,
        "output_multiset_digest": ordered_digest,
        "stroke_multiset_identical": True,
        "final_raster_identical": True,
        "max_final_pixel_difference": max_channel_diff,
        "moved_stroke_count": moved,
        "moved_stroke_ratio": moved / max(1, len(strokes)),
        "mean_absolute_order_displacement": sum(displacement) / max(1, len(displacement)),
        "max_absolute_order_displacement": max(displacement) if displacement else 0,
        "semantic_counts": dict(label_counts),
        "first_semantic_positions": first_positions,
        "canvas_dependent_barriers": sum(1 for s in strokes if s.get("brush") in CANVAS_DEPENDENT),
        "landmark_engine": landmarks["engine"],
        "landmark_count": landmarks["landmark_count"],
    }
    (out / "equivalence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_checkpoints(result, out, [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0])
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
