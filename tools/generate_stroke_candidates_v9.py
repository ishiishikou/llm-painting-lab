#!/usr/bin/env python3
"""v9: choose reusable v7 stroke passages from semantic intent cards.

The reviewer never authors stroke coordinates.  This tool receives a frozen v7
stroke document, a semantic intent card, and accepted/rejected bundle ids from
earlier visual reviews.  It proposes small coherent passages built only from
already-generated v7 strokes and renders them for visual selection.

No reference image is opened or analysed here.  Coordinates, colors, widths,
opacity, brush geometry and stroke order are inherited from v7.
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
    return (
        (float(stroke.get("x1", 0)) + float(stroke.get("x2", stroke.get("x1", 0)))) / 2,
        (float(stroke.get("y1", 0)) + float(stroke.get("y2", stroke.get("y1", 0)))) / 2,
    )


def width_of(stroke):
    return float(
        max(stroke.get("width", 0), stroke.get("widthStart", 0), stroke.get("widthEnd", 0))
    )


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def face_relative(c, face):
    x0, y0, x1, y1 = map(float, face)
    return (
        (c[0] - x0) / max(1.0, x1 - x0),
        (c[1] - y0) / max(1.0, y1 - y0),
    )


def focus_score(c, face, focus):
    """Soft semantic prior only; exact placement still comes from v7 strokes."""
    rx, ry = face_relative(c, face)
    if focus in (None, "", "whole_face"):
        return 1.0 if -0.18 <= rx <= 1.02 and 0.05 <= ry <= 1.02 else 0.25
    if focus == "left_half":
        return 1.0 if rx <= 0.55 and -0.05 <= ry <= 1.05 else 0.1
    if focus == "right_half":
        return 1.0 if rx >= 0.45 and -0.05 <= ry <= 1.05 else 0.1
    bands = {
        "upper_face": (0.10, 0.48),
        "eye_band": (0.26, 0.52),
        "mid_face": (0.38, 0.72),
        "lower_face": (0.58, 1.04),
    }
    if focus in bands:
        lo, hi = bands[focus]
        return 1.0 if lo <= ry <= hi and -0.18 <= rx <= 1.02 else 0.08
    return 0.6


def stroke_family(stroke):
    brush = stroke.get("brush")
    phase = stroke.get("phase")
    w = width_of(stroke)
    if brush in {"mixerBrush", "smudgeBrush"}:
        return "blend"
    if brush == "line" or w <= 2.8:
        return "edge"
    if brush in {"dryBrush", "flatBrush"} or w >= 9.0:
        return "plane"
    if phase in {"detail", "finish"} and w <= 6.5:
        return "accent"
    return "structure"


def family_of(strokes):
    families = Counter(stroke_family(s) for s in strokes)
    if not families:
        return "structure"
    if families["structure"] + families["accent"] >= max(1, len(strokes) // 2):
        return "structure"
    return families.most_common(1)[0][0]


def bundle_centroid(strokes):
    cs = [center(s) for s in strokes]
    return (
        sum(x for x, _ in cs) / len(cs),
        sum(y for _, y in cs) / len(cs),
    )


def make_micro_bundles(source, max_micro=6):
    strokes = source["strokes"]
    bundles = {}
    by_load = defaultdict(list)
    for s in strokes:
        if s.get("paintLoadId"):
            by_load[str(s["paintLoadId"])].append(s)
    for lid, ss in by_load.items():
        if 1 <= len(ss) <= 9:
            bundles[f"load-{lid}"] = ss

    raw = [
        s
        for s in strokes
        if s.get("phase") == "face_structure"
        and not s.get("paintLoadId")
        and s.get("role") == "face-plane"
        and s.get("brush") in {"variableBrush", "flatBrush"}
    ]
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


def passage_eligible(stroke):
    return (
        stroke.get("phase") in {"face_structure", "detail", "finish"}
        and stroke.get("role") in {"face", "face-plane"}
        and stroke.get("brush") not in {"mixerBrush", "smudgeBrush", "glaze"}
    )


def passage_distance(seed, other):
    """Prefer local geometry and nearby source order without inventing geometry."""
    spatial = dist(center(seed), center(other))
    sid = abs(int(seed.get("id", 0)) - int(other.get("id", 0)))
    phase_penalty = 0 if seed.get("phase") == other.get("phase") else 26
    family_penalty = 0 if stroke_family(seed) == stroke_family(other) else 18
    return spatial + min(85, sid * 0.018) + phase_penalty + family_penalty


def make_passage_bundles(source, min_strokes=12, max_strokes=24):
    """Build coherent 12-30 stroke passages from exact v7 geometry.

    Passages are centered on existing strokes.  Their members are chosen by
    spatial/source-order proximity; nothing is moved, recolored or redrawn.
    """
    eligible = [s for s in source["strokes"] if passage_eligible(s)]
    if not eligible:
        return {}

    cell = 54.0
    seeded = {}
    for s in eligible:
        x, y = center(s)
        key = (s.get("phase"), stroke_family(s), int(x // cell), int(y // cell))
        old = seeded.get(key)
        if old is None or int(s.get("id", 0)) < int(old.get("id", 0)):
            seeded[key] = s

    bundles = {}
    signatures = []
    serial = 1
    target = max(min_strokes, min(max_strokes, int(round((min_strokes + max_strokes) / 2))))

    for _, seed in sorted(seeded.items(), key=lambda kv: int(kv[1].get("id", 0))):
        ranked = sorted(eligible, key=lambda s: passage_distance(seed, s))
        local = []
        seed_c = center(seed)
        for s in ranked:
            radius = 92 if stroke_family(seed) in {"structure", "edge", "accent"} else 78
            if dist(seed_c, center(s)) > radius:
                continue
            if stroke_family(s) == "blend":
                continue
            local.append(s)
            if len(local) >= target:
                break

        if len(local) < min_strokes:
            continue

        local = sorted(local[:max_strokes], key=lambda s: int(s.get("id", 0)))
        ids = frozenset(int(s.get("id", 0)) for s in local)
        duplicate = False
        for old in signatures:
            inter = len(ids & old)
            union = len(ids | old)
            if union and inter / union >= 0.72:
                duplicate = True
                break
        if duplicate:
            continue
        signatures.append(ids)
        bundles[f"passage-{serial:04d}"] = local
        serial += 1
    return bundles


def make_bundles(source, intent):
    mode = intent.get("candidate_mode", "micro")
    if mode == "passage":
        return make_passage_bundles(
            source,
            int(intent.get("min_strokes_per_candidate", 12)),
            int(intent.get("max_strokes_per_candidate", 24)),
        )
    return make_micro_bundles(source, int(intent.get("max_strokes_per_candidate", 7)))


def candidate_score(bundle_id, ss, intent, face, accepted_centroids):
    phase_wanted = set(intent.get("source_phases", ["face_structure"]))
    region = intent.get("region", "face")
    families = set(intent.get("preferred_families", ["structure", "plane"]))
    focus = intent.get("focus", "whole_face")
    max_strokes = int(intent.get("max_strokes_per_candidate", 7))
    min_strokes = int(intent.get("min_strokes_per_candidate", 1))
    avoid = set(intent.get("avoid_brushes", []))
    phases = Counter(s.get("phase") for s in ss)
    roles = Counter(s.get("role", "") for s in ss)
    brushes = Counter(s.get("brush", "") for s in ss)
    fam = family_of(ss)
    c = bundle_centroid(ss)

    score = 0.0
    wanted_fraction = sum(phases[p] for p in phase_wanted) / max(1, len(ss))
    score += 10.0 * wanted_fraction - 3.0 * (1.0 - wanted_fraction)

    if region == "face":
        face_fraction = (roles["face"] + roles["face-plane"]) / max(1, len(ss))
        score += 8.0 * face_fraction
        rx, ry = face_relative(c, face)
        score += 2.0 if -0.25 <= rx <= 1.08 and -0.02 <= ry <= 1.08 else -3.0

    score += 6.0 if fam in families else -2.5
    score += 5.0 * focus_score(c, face, focus)

    if min_strokes <= len(ss) <= max_strokes:
        score += 3.0
    else:
        score -= 2.0 * abs(len(ss) - max(min_strokes, min(max_strokes, len(ss))))

    if any(brushes[b] for b in avoid):
        score -= 10.0

    if accepted_centroids:
        nearest = min(dist(c, a) for a in accepted_centroids)
        score += min(3.5, nearest / 75.0)

    score += (sum(ord(ch) for ch in bundle_id) % 97) / 1000.0
    return score


def choose_candidates(scored, count=3):
    if not scored:
        return []
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    chosen = [scored[0]]
    pool = scored[1:160]
    while len(chosen) < count and pool:
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
    intent = state["intent_card"]
    bundles = make_bundles(source, intent)
    accepted_ids = list(state.get("accepted_bundle_ids", []))
    rejected_ids = set(state.get("rejected_bundle_ids", []))
    max_strokes = int(intent.get("max_strokes_per_candidate", 7))
    min_strokes = int(intent.get("min_strokes_per_candidate", 1))
    candidate_count = int(intent.get("candidate_count", 3))

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
        if bid in accepted_ids or bid in rejected_ids:
            continue
        if len(ss) > max_strokes or len(ss) < min_strokes:
            continue
        score = candidate_score(bid, ss, intent, face, accepted_centroids)
        if score >= 4.0:
            scored.append((score, bid, ss, bundle_centroid(ss)))

    chosen = choose_candidates(scored, candidate_count)
    if len(chosen) < candidate_count:
        raise SystemExit(f"not enough candidate bundles: {len(chosen)} / {candidate_count}")

    labels = [chr(ord("A") + i) for i in range(candidate_count)]
    manifest = {
        "review_round": int(state.get("review_round", 1)),
        "intent_card": intent,
        "candidate_mode": intent.get("candidate_mode", "micro"),
        "frozen_base_phases": sorted(base_phases),
        "base_stroke_count": len(base_strokes),
        "catalogue_bundle_count": len(bundles),
        "accepted_bundle_ids": accepted_ids,
        "accepted_stroke_count": len(accepted_strokes),
        "candidate_count": len(chosen),
        "candidates": [],
        "reference_used_for_candidate_selection": False,
        "coordinates_authored_by_reviewer": False,
        "source": "exact v7 stroke geometry",
    }

    for label, (score, bid, ss, c) in zip(labels, chosen):
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
            "source_stroke_ids": [s.get("id") for s in ss],
        })

    trace = {
        "metadata": {
            "mode": "semantic-intent-selects-existing-v7-passages",
            "reference_access": False,
            "reviewer_coordinates": False,
            "base_stroke_count": len(base_strokes),
            "accepted_stroke_count": len(accepted_strokes),
        },
        "accepted_strokes": accepted_strokes,
    }
    (out_dir / "accepted-strokes.json").write_text(json.dumps(trace, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out_dir / "candidate-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
