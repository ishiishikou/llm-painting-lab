#!/usr/bin/env python3
"""v10: place proven v7 stroke templates at automatically detected landmarks.

The reviewer chooses only a semantic target and intent.  Exact target coordinates
come from deterministic computer-vision landmark detection on the reference image.
Stroke geometry comes from coherent v7 paint-load passages.  Candidate generation
never asks the reviewer to author x/y, angle, width or control points.

For the first experiment this tool supports the two visible eyes.  It deliberately
starts narrow: establish whether semantic-target selection + proven stroke geometry
works before expanding to nose, mouth, pearl, headwrap and garment.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import reference_to_brush_process as base


def stroke_center(s):
    return (
        (float(s.get("x1", 0)) + float(s.get("x2", s.get("x1", 0)))) / 2.0,
        (float(s.get("y1", 0)) + float(s.get("y2", s.get("y1", 0)))) / 2.0,
    )


def bundle_centroid(strokes):
    cs = [stroke_center(s) for s in strokes]
    return (
        sum(x for x, _ in cs) / max(1, len(cs)),
        sum(y for _, y in cs) / max(1, len(cs)),
    )


def color_family_ok(s, family):
    if not family:
        return True
    color = s.get("color")
    if not color:
        return False
    try:
        r, g, b = base.parse_hex(color)
    except Exception:
        return False
    l = base.lum((r, g, b))
    if family == "eye_dark":
        warm_neutral = r >= b * 0.82 and g >= b * 0.72 and r >= g * 0.72
        return warm_neutral and 18 <= l <= 112
    if family == "skin_mid":
        return base.skin((r, g, b)) and l <= 205
    if family == "skin_light":
        return base.skin((r, g, b)) and l >= 170
    return True


def persisted_v9_ids(v9_state):
    ids = set()
    for passage in v9_state.get("accepted_passages", []):
        ids.update(int(x) for x in passage.get("source_stroke_ids", []))
    return ids


def reconstruct_v9_current(source, v9_state):
    base_phases = set(v9_state.get("frozen_base_phases", ["silhouette", "light_shadow"]))
    current = [copy.deepcopy(s) for s in source["strokes"] if s.get("phase") in base_phases]
    by_id = {int(s.get("id", 0)): s for s in source["strokes"]}
    for sid in sorted(persisted_v9_ids(v9_state)):
        if sid not in by_id:
            raise SystemExit(f"v9 accepted source stroke missing: {sid}")
        current.append(copy.deepcopy(by_id[sid]))
    return current


def dark_centroid(gray, box):
    x, y, w, h = map(int, box)
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return x + w / 2.0, y + h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    core = (
        (xx > w * 0.12) & (xx < w * 0.88) &
        (yy > h * 0.15) & (yy < h * 0.85)
    )
    vals = roi[core]
    if vals.size < 20:
        return x + w / 2.0, y + h / 2.0
    threshold = float(np.quantile(vals, 0.08))
    mask = core & (roi <= threshold)
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return x + w / 2.0, y + h / 2.0
    weights = np.maximum(1.0, threshold + 1.0 - roi[mask].astype(float))
    return x + float(np.average(xs, weights=weights)), y + float(np.average(ys, weights=weights))


def choose_eye_pair(eyes, face_box):
    fx, fy, fw, fh = map(float, face_box)
    usable = []
    for e in eyes:
        x, y, w, h = map(float, e)
        cy = y + h / 2.0
        if cy <= fy + fh * 0.68:
            usable.append((x, y, w, h))
    if len(usable) < 2:
        usable = [tuple(map(float, e)) for e in eyes]
    if len(usable) < 2:
        raise SystemExit(f"eye detector returned fewer than two usable eyes: {len(usable)}")

    best = None
    best_score = -1e18
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            a, b = usable[i], usable[j]
            ac = (a[0] + a[2] / 2, a[1] + a[3] / 2)
            bc = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            sep = abs(ac[0] - bc[0])
            ydiff = abs(ac[1] - bc[1])
            area = a[2] * a[3] + b[2] * b[3]
            if sep < fw * 0.16:
                continue
            score = sep * 2.0 - ydiff * 1.2 + math.sqrt(max(1.0, area))
            if score > best_score:
                best_score, best = score, (a, b)
    if best is None:
        best = sorted(usable, key=lambda e: e[2] * e[3], reverse=True)[:2]
    return sorted(best, key=lambda e: e[0] + e[2] / 2.0)


def detect_landmarks(reference_path, width, height):
    ref = base.crop_resize(Image.open(reference_path), width, height)
    rgb = np.asarray(ref.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if face_cascade.empty() or eye_cascade.empty():
        raise SystemExit("OpenCV Haar cascades are unavailable")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(120, 120))
    if len(faces) == 0:
        raise SystemExit("automatic face landmark detection failed")
    face = max(faces, key=lambda q: int(q[2]) * int(q[3]))
    fx, fy, fw, fh = map(int, face)
    roi = gray[fy:fy + fh, fx:fx + fw]
    eyes_rel = eye_cascade.detectMultiScale(roi, scaleFactor=1.03, minNeighbors=3, minSize=(20, 20))
    eyes_abs = [(fx + int(x), fy + int(y), int(w), int(h)) for x, y, w, h in eyes_rel]
    pair = choose_eye_pair(eyes_abs, face)

    refined = []
    for e in pair:
        px, py = dark_centroid(gray, e)
        refined.append({
            "detector_box": [round(float(v), 2) for v in e],
            "anchor": [round(px, 3), round(py, 3)],
        })

    left, right = refined
    dx = right["anchor"][0] - left["anchor"][0]
    dy = right["anchor"][1] - left["anchor"][1]
    return ref, {
        "face": [fx, fy, fw, fh],
        "viewer_left_eye": left,
        "viewer_right_eye": right,
        "eye_line_angle_deg": round(math.degrees(math.atan2(dy, dx)), 3),
        "method": "opencv-haar-eye-box + dark-centroid refinement",
    }


def eligible_template_stroke(s, intent):
    if s.get("phase") not in set(intent.get("source_phases", ["detail"])):
        return False
    if s.get("role") not in set(intent.get("source_roles", ["face"])):
        return False
    allowed = set(intent.get("allow_brushes", ["line", "variableBrush"]))
    if s.get("brush") not in allowed:
        return False
    return color_family_ok(s, intent.get("color_family"))


def make_templates(source, intent):
    by_load = defaultdict(list)
    for s in source["strokes"]:
        if not eligible_template_stroke(s, intent):
            continue
        lid = s.get("paintLoadId")
        if lid is None:
            continue
        by_load[str(lid)].append(s)

    minn = int(intent.get("min_strokes_per_template", 3))
    maxn = int(intent.get("max_strokes_per_template", 7))
    templates = []
    for lid, ss in by_load.items():
        ss = sorted(ss, key=lambda q: int(q.get("id", 0)))
        if not (minn <= len(ss) <= maxn):
            continue
        xs, ys = zip(*(stroke_center(s) for s in ss))
        if max(xs) - min(xs) > float(intent.get("max_template_span", 36)):
            continue
        if max(ys) - min(ys) > float(intent.get("max_template_span", 36)):
            continue
        templates.append((f"paint-load-{lid}", ss))
    templates.sort(key=lambda q: int(q[0].split("-")[-1]))
    return templates


def shape_signature(strokes):
    cx, cy = bundle_centroid(strokes)
    pts = []
    for s in strokes:
        x1, y1 = float(s.get("x1", 0)), float(s.get("y1", 0))
        x2, y2 = float(s.get("x2", x1)), float(s.get("y2", y1))
        pts.extend([(x1 - cx, y1 - cy), (x2 - cx, y2 - cy)])
    if not pts:
        return (0, 0, 0)
    radius = max(math.hypot(x, y) for x, y in pts)
    lengths = [math.hypot(float(s.get("x2", 0)) - float(s.get("x1", 0)), float(s.get("y2", 0)) - float(s.get("y1", 0))) for s in strokes]
    angles = [math.atan2(float(s.get("y2", 0)) - float(s.get("y1", 0)), float(s.get("x2", 0)) - float(s.get("x1", 0))) for s in strokes]
    sx = sum(math.cos(2 * a) for a in angles)
    sy = sum(math.sin(2 * a) for a in angles)
    axis = math.degrees(math.atan2(sy, sx) / 2.0) if angles else 0.0
    return (round(radius, 2), round(sum(lengths) / max(1, len(lengths)), 2), round(axis, 1))


def template_distance(a, b):
    sa, sb = shape_signature(a), shape_signature(b)
    return abs(sa[0] - sb[0]) + abs(sa[1] - sb[1]) * 1.5 + abs(sa[2] - sb[2]) * 0.12


def choose_templates(templates, wanted, excluded_ids):
    pool = [(tid, ss) for tid, ss in templates if tid not in excluded_ids]
    if len(pool) < wanted:
        wanted = len(pool)
    if wanted == 0:
        raise SystemExit("no eligible stroke templates remain")
    chosen = [pool[0]]
    remain = pool[1:]
    while len(chosen) < wanted and remain:
        best = max(remain, key=lambda item: min(template_distance(item[1], q[1]) for q in chosen))
        chosen.append(best)
        remain.remove(best)
    return chosen


def translate_bundle(strokes, anchor, template_id, target, label):
    cx, cy = bundle_centroid(strokes)
    dx, dy = float(anchor[0]) - cx, float(anchor[1]) - cy
    out = []
    max_err = 0.0
    for src in strokes:
        s = copy.deepcopy(src)
        old_dx = float(s.get("x2", s.get("x1", 0))) - float(s.get("x1", 0))
        old_dy = float(s.get("y2", s.get("y1", 0))) - float(s.get("y1", 0))
        for key in ("x1", "x2"):
            if key in s:
                s[key] = round(float(s[key]) + dx, 4)
        for key in ("y1", "y2"):
            if key in s:
                s[key] = round(float(s[key]) + dy, 4)
        s.pop("clipBox", None)
        new_dx = float(s.get("x2", s.get("x1", 0))) - float(s.get("x1", 0))
        new_dy = float(s.get("y2", s.get("y1", 0))) - float(s.get("y1", 0))
        max_err = max(max_err, abs(new_dx - old_dx), abs(new_dy - old_dy))
        s["v10TemplateId"] = template_id
        s["v10TemplateSourceStrokeId"] = src.get("id")
        s["v10Target"] = target
        s["v10CandidateLabel"] = label
        out.append(s)
    return out, (dx, dy), max_err


def accepted_template_strokes(source, state, landmarks):
    by_id = {int(s.get("id", 0)): s for s in source["strokes"]}
    out = []
    for entry in state.get("accepted_templates", []):
        target = entry["target"]
        if target not in landmarks:
            raise SystemExit(f"accepted target landmark unavailable: {target}")
        ss = []
        for sid in entry.get("source_stroke_ids", []):
            src = by_id.get(int(sid))
            if src is None:
                raise SystemExit(f"accepted template source stroke missing: {sid}")
            ss.append(src)
        transformed, _, _ = translate_bundle(ss, landmarks[target]["anchor"], entry["template_id"], target, "accepted")
        out.extend(transformed)
    return out


def make_review_board(reference, current, candidates, out_path):
    images = [("REFERENCE", reference), ("CURRENT", current)] + [(f"CANDIDATE {k}", v) for k, v in candidates]
    cols = 3
    rows = math.ceil(len(images) / cols)
    scale = 0.34 if rows <= 2 else 0.27
    tw, th = round(current.width * scale), round(current.height * scale)
    cell_h = th + 34
    board = Image.new("RGB", (tw * cols, cell_h * rows), (238, 238, 238))
    draw = ImageDraw.Draw(board)
    for i, (label, im) in enumerate(images):
        x, y = (i % cols) * tw, (i // cols) * cell_h
        board.paste(im.resize((tw, th)), (x, y + 34))
        draw.text((x + 8, y + 9), label, fill="black")
    board.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_strokes")
    ap.add_argument("v9_state")
    ap.add_argument("v10_state")
    ap.add_argument("reference")
    ap.add_argument("--out-dir", default="artifact-v10")
    args = ap.parse_args()

    source = json.loads(Path(args.source_strokes).read_text(encoding="utf-8"))
    v9_state = json.loads(Path(args.v9_state).read_text(encoding="utf-8"))
    state = json.loads(Path(args.v10_state).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas = source["canvas"]
    width, height = int(canvas["width"]), int(canvas["height"])
    background = canvas["background"]
    reference, landmarks = detect_landmarks(args.reference, width, height)
    (out_dir / "landmarks.json").write_text(json.dumps(landmarks, ensure_ascii=False, indent=2), encoding="utf-8")

    current_strokes = reconstruct_v9_current(source, v9_state)
    current_strokes += accepted_template_strokes(source, state, landmarks)
    current = base.render(current_strokes, width, height, background)
    current.save(out_dir / "current.png")

    intent = state["intent_card"]
    target = intent.get("target")
    if target not in {"viewer_left_eye", "viewer_right_eye"}:
        raise SystemExit(f"unsupported or missing semantic target: {target}")
    anchor = landmarks[target]["anchor"]

    templates = make_templates(source, intent)
    accepted_ids = {x.get("template_id") for x in state.get("accepted_templates", []) if x.get("target") == target}
    rejected_ids = set(state.get("rejected_template_ids", []))
    excluded = accepted_ids | rejected_ids
    chosen = choose_templates(templates, int(intent.get("candidate_count", 6)), excluded)

    manifest = {
        "review_round": int(state.get("review_round", 1)),
        "intent_card": intent,
        "candidate_mode": "landmark_template",
        "target": target,
        "detected_landmark": landmarks[target],
        "coordinates_authored_by_reviewer": False,
        "reference_used_for_landmark_detection": True,
        "reference_pixels_used_for_template_ranking": False,
        "base_stroke_count": len(current_strokes),
        "candidate_count": len(chosen),
        "candidates": [],
    }
    candidate_images = []
    metrics = {"current": base.metrics(reference, current), "candidates": {}}

    for i, (template_id, ss) in enumerate(chosen):
        label = chr(ord("A") + i)
        transformed, delta, geometry_error = translate_bundle(ss, anchor, template_id, target, label)
        rendered = base.render(current_strokes + transformed, width, height, background)
        rendered.save(out_dir / f"candidate-{label}.png")
        candidate_images.append((label, rendered))
        metrics["candidates"][label] = base.metrics(reference, rendered)
        ccx, ccy = bundle_centroid(transformed)
        manifest["candidates"].append({
            "label": label,
            "template_id": template_id,
            "source_stroke_ids": [s.get("id") for s in ss],
            "stroke_count": len(ss),
            "brushes": dict(Counter(s.get("brush") for s in ss)),
            "colors": dict(Counter(s.get("color") for s in ss)),
            "source_shape_signature": list(shape_signature(ss)),
            "generated_translation": [round(delta[0], 3), round(delta[1], 3)],
            "candidate_centroid": [round(ccx, 3), round(ccy, 3)],
            "anchor_distance": round(math.hypot(ccx - anchor[0], ccy - anchor[1]), 6),
            "max_geometry_error": round(geometry_error, 8),
        })

    (out_dir / "candidate-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "evaluation-only.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    make_review_board(reference, current, candidate_images, out_dir / "review-board.png")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
