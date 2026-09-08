#!/usr/bin/env python3
"""v9: select reusable v7 stroke bundles from semantic intent cards.

The reviewer never specifies stroke coordinates. This tool receives a frozen v7
stroke document, a semantic intent card, and accepted/rejected bundle ids from
earlier visual reviews. It proposes three small candidate bundles and renders
them. It never opens or analyses the reference image. Coordinates, colors,
widths and brush geometry are copied from already-generated v7 strokes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import reference_to_brush_process as base


def center(stroke):
    return ((float(stroke.get("x1", 0)) + float(stroke.get("x2", stroke.get("x1", 0)))) / 2,
            (float(stroke.get("y1", 0)) + float(stroke.get("y2", stroke.get("y1", 0)))) / 2)


def width_of(stroke):
    return float(max(stroke.get("width", 0), stroke.get("widthStart", 0), stroke.get("widthEnd", 0)))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def face_relative(c, face):
    x0, y0, x1, y1 = map(float, face)
    return ((c[0] - x0) / max(1.0, x1 - x0), (c[1] - y0) / max(1.0, y1 - y0))


def focus_score(c, face, focus):
    rx, ry = face_relative(c, face)
    if focus in (None, "", "whole_face"):
        return 1.0 if -0.08 <= rx <= 1.08 and -0.08 <= ry <= 1.08 else 0.15
    if focus == "left_half":
        return 1.0 if rx <= 0.55 and -0.05 <= ry <= 1.05 else 0.1
    if focus == "right_half":
        return 1.0 if rx >= 0.45 and -0.05 <= ry <= 1.05 else 0.1
    bands = {
        "upper_face": (0.00, 0.48),
        "eye_band": (0.26, 0.52),
        "mid_face": (0.38, 0.72),
        "lower_face": (0.58, 1.04),
    }
    if focus in bands:
        lo, hi = bands[focus]
        return 1.0 if lo <= ry <= hi and -0.08 <= rx <= 1.08 else 0.1
    return 0.6


def family_of(strokes):
    brushes = Counter(s.get("brush") for s in strokes)
    phases = Counter(s.get("phase") for s in strokes)
    widths = [width_of(s) for s in strokes]
    mean_w = sum(widths) / max(1, len(widths))
    if brushes["mixerBrush"] + brushes["smudgeBrush"] >= max(1, len(strokes) // 3):
        return "blend"
    if phases["face_structure"] >= max(1, len(strokes) // 2) and brushes["variableBrush"] >= max(1, len(strokes) // 2):
        return "structure"
    if brushes["line"] >= max(1, len(strokes) // 3) or mean_w <= 2.6:
        return "edge"
    if mean_w >= 9.0 and brushes["line"] == 0:
        return "plane"
    if phases["detail"] + phases["finish"] >= max(1, len(strokes) // 2) and mean_w <= 6.5:
        return "accent"
    return "structure"


def bundle_centroid(strokes):
    cs = [center(s) for s in strokes]
    return (sum(x for x, _ in cs) / len(cs), sum(y for _, y in cs) / len(cs))


def make_bundles(source, max_micro=6):
    strokes = source["strokes"]
    bundles = {}
    by_load = defaultdict(list)
    for s in strokes:
        if s.get("paintLoadId"):
            by_load[str(s["paintLoadId"])].append(s)
    for lid, ss in by_load.items():
        if 1 <= len(ss) <= 9:
            bundles[f"load-{lid}"] = ss

    raw = [s for s in strokes
           if s.get("phase") == "face_structure"
           and not s.get("paintLoadId")
           and s.get("role") == "face-plane"
           and s.get("brush") in {"variableBrush", "flatBrush"}]
    unused = set(range(len(raw)))
    serial = 1
    while unused:
        seed_i = min(unused, key=lambda i: int(raw[i].get("id", i)))
        unused.remove(seed_i)
        seed_c = center(raw[seed_i])
        near = sorted(unused, key=lambda i: dist(seed_c, center(raw[i])))
        group = [seed_i]
        for i in near:
            if len(group) >= max_micro:
                break
            if dist(seed_c, center(raw[i])) <= 44:
                group.append(i)
        for i in group[1:]:
            unused.discard(i)
        bundles[f"micro-{serial:04d}"] = [raw[i] for i in group]
        serial += 1
    return bundles


def candidate_score(bundle_id, ss, intent, face, accepted_centroids):
    phase_wanted = set(intent.get("source_phases", ["face_structure"]))
    region = intent.get("region", "face")
    families = set(intent.get("preferred_families", ["structure", "plane"]))
    focus = intent.get("focus", "whole_face")
    max_strokes = int(intent.get("max_strokes_per_candidate", 7))
    avoid = set(intent.get("avoid_brushes", []))
    phases = Counter(s.get("phase") for s in ss)
    roles = Counter(s.get("role", "") for s in ss)
    brushes = Counter(s.get("brush", "") for s in ss)
    fam = family_of(ss)
    c = bundle_centroid(ss)
    score = 0.0
    score += 8.0 if any(p in phase_wanted for p in phases) else -8.0
    if region == "face":
        if roles["face"] + roles["face-plane"] > 0:
            score += 7.0
        rx, ry = face_relative(c, face)
        score += 4.0 if -0.10 <= rx <= 1.10 and -0.10 <= ry <= 1.10 else -5.0
    score += 6.0 if fam in families else -2.0
    score += 5.0 * focus_score(c, face, focus)
    score += 2.5 if len(ss) <= max_strokes else -(len(ss) - max_strokes) * 2.0
    if any(brushes[b] for b in avoid):
        score -= 8.0
    if accepted_centroids:
        nearest = min(dist(c, a) for a in accepted_centroids)
        score += min(3.0, nearest / 80.0)
    score += (sum(ord(ch) for ch in bundle_id) % 97) / 1000.0
    return score


def choose_three(scored):
    if not scored:
        return []
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    chosen = [scored[0]]
    pool = scored[1:80]
    while len(chosen) < 3 and pool:
        best, best_val = None, -1e9
        for item in pool:
            score, _, _, c = item
            sep = min(dist(c, q[3]) for q in chosen)
            value = score + min(5.5, sep / 60.0)
            if value > best_val:
                best_val, best = value, item
        chosen.append(best)
        pool.remove(best)
    return chosen


def clone_bundle(ss, bundle_id, label):
    out = []
    for source_s in ss:
        s = copy.deepcopy(source_s)
        s["v9BundleId"] = bundle_id
        s["v9CandidateLabel"] = label
        s["v9SourceStrokeId"] = source_s.get("id")
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description="既存v7ストロークを意図カードから候補化するv9")
    ap.add_argument("source_strokes")
    ap.add_argument("state")
    ap.add_argument("--out-dir", default="artifact-v9")
    args = ap.parse_args()

    source = json.loads(Path(args.source_strokes).read_text(encoding="utf-8"))
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    face = source["metadata"]["face_bbox"]
    canvas = source["canvas"]
    background = canvas["background"]
    width, height = int(canvas["width"]), int(canvas["height"])
    bundles = make_bundles(source)
    accepted_ids = list(state.get("accepted_bundle_ids", []))
    rejected_ids = set(state.get("rejected_bundle_ids", []))
    intent = state["intent_card"]
    max_strokes = int(intent.get("max_strokes_per_candidate", 7))

    base_phases = set(state.get("frozen_base_phases", ["silhouette", "light_shadow"]))
    base_strokes = [copy.deepcopy(s) for s in source["strokes"] if s.get("phase") in base_phases]
    accepted_strokes, accepted_centroids, missing = [], [], []
    for bid in accepted_ids:
        ss = bundles.get(bid)
        if not ss:
            missing.append(bid)
            continue
        accepted_strokes.extend(clone_bundle(ss, bid, "accepted"))
        accepted_centroids.append(bundle_centroid(ss))
    if missing:
        raise SystemExit(f"accepted bundle ids missing from source: {missing}")

    current_strokes = base_strokes + accepted_strokes
    base.render(current_strokes, width, height, background).save(out_dir / "current.png")

    scored = []
    for bid, ss in bundles.items():
        if bid in accepted_ids or bid in rejected_ids or len(ss) > max_strokes or len(ss) < 3:
            continue
        score = candidate_score(bid, ss, intent, face, accepted_centroids)
        if score >= 4.0:
            scored.append((score, bid, ss, bundle_centroid(ss)))
    chosen = choose_three(scored)
    if len(chosen) < 3:
        raise SystemExit(f"not enough candidate bundles: {len(chosen)}")

    manifest = {
        "review_round": int(state.get("review_round", 1)),
        "intent_card": intent,
        "frozen_base_phases": sorted(base_phases),
        "base_stroke_count": len(base_strokes),
        "accepted_bundle_ids": accepted_ids,
        "accepted_stroke_count": len(accepted_strokes),
        "candidate_count": len(chosen),
        "candidates": [],
        "reference_used_for_candidate_selection": False,
        "coordinates_authored_by_reviewer": False,
        "source": "exact v7 stroke geometry"
    }
    for label, (score, bid, ss, c) in zip(["A", "B", "C"], chosen):
        cand_strokes = clone_bundle(ss, bid, label)
        base.render(current_strokes + cand_strokes, width, height, background).save(out_dir / f"candidate-{label}.png")
        manifest["candidates"].append({
            "label": label,
            "bundle_id": bid,
            "score_without_reference": round(score, 4),
            "stroke_count": len(ss),
            "family": family_of(ss),
            "brushes": dict(Counter(s.get("brush") for s in ss)),
            "phases": dict(Counter(s.get("phase") for s in ss)),
            "roles": dict(Counter(s.get("role") for s in ss)),
            "centroid_normalized": [round(c[0] / width, 4), round(c[1] / height, 4)],
            "source_stroke_ids": [s.get("id") for s in ss]
        })

    trace = {
        "metadata": {
            "mode": "semantic-intent-selects-existing-v7-bundles",
            "reference_access": False,
            "reviewer_coordinates": False,
            "base_stroke_count": len(base_strokes),
            "accepted_stroke_count": len(accepted_strokes)
        },
        "accepted_strokes": accepted_strokes
    }
    (out_dir / "accepted-strokes.json").write_text(json.dumps(trace, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out_dir / "candidate-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
