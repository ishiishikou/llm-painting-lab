#!/usr/bin/env python3
"""v9 semantic passage selector.

The reviewer decides only semantic intent (region/focus/goal). Exact coordinates,
colors, widths and brush geometry are inherited from already-generated v7 strokes.
This tool groups those proven strokes into larger local passages so each visual
review has a meaningful effect without asking the reviewer to author coordinates.

Accepted passages are persisted by exact v7 source-stroke IDs. Candidate passages
are built only from not-yet-accepted source strokes, so a candidate preview never
repaints already accepted marks and then disappears after acceptance.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

import reference_to_brush_process as base


def center(s):
    return (
        (float(s.get("x1", 0)) + float(s.get("x2", s.get("x1", 0)))) / 2,
        (float(s.get("y1", 0)) + float(s.get("y2", s.get("y1", 0)))) / 2,
    )


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def face_relative(c, face):
    x0, y0, x1, y1 = map(float, face)
    return (
        (c[0] - x0) / max(1.0, x1 - x0),
        (c[1] - y0) / max(1.0, y1 - y0),
    )


def focus_ok(c, face, focus):
    rx, ry = face_relative(c, face)
    if not (-0.20 <= rx <= 1.05):
        return False
    bands = {
        "whole_face": (-0.05, 1.05),
        "upper_face": (0.08, 0.50),
        "eye_band": (0.24, 0.53),
        "mid_face": (0.36, 0.73),
        "lower_face": (0.57, 1.06),
    }
    lo, hi = bands.get(focus, bands["whole_face"])
    return lo <= ry <= hi


def color_family_ok(s, intent):
    family = intent.get("color_family")
    if not family:
        return True
    color = s.get("color")
    if not color:
        return False
    try:
        p = base.parse_hex(color)
    except Exception:
        return False
    r, g, b = p
    if family == "skin":
        return base.skin(p)
    if family == "skin_mid":
        return base.skin(p) and base.lum(p) <= 205
    if family == "skin_or_shadow":
        if base.skin(p):
            return True
        return r >= g >= b and 35 <= r <= 155 and max(p) - min(p) >= 8
    if family == "face_shadow":
        l = base.lum(p)
        warm_or_neutral = r >= b * 0.95 and g >= b * 0.82 and r >= g * 0.88
        return warm_or_neutral and 35 <= l <= 165 and max(p) - min(p) >= 6
    return True


def eligible(s, intent, face):
    if s.get("phase") not in set(intent.get("source_phases", ["face_structure"])):
        return False
    if s.get("role") not in {"face", "face-plane"}:
        return False
    brush = s.get("brush")
    allowed = intent.get("allow_brushes")
    if allowed and brush not in set(allowed):
        return False
    if brush in set(intent.get("avoid_brushes", [])):
        return False
    if s.get("role") in {"face-center", "eye-line", "mouth-line"}:
        return False
    if not color_family_ok(s, intent):
        return False
    return focus_ok(center(s), face, intent.get("focus", "whole_face"))


def nearest_bundle(seed, pool, count):
    ranked = sorted(
        pool,
        key=lambda s: (
            dist(center(seed), center(s)),
            abs(int(seed.get("id", 0)) - int(s.get("id", 0))) * 0.025,
            int(s.get("id", 0)),
        ),
    )
    return sorted(ranked[:count], key=lambda s: int(s.get("id", 0)))


def bundle_id(focus, seed, count):
    return f"semantic-{focus}-{int(seed.get('id', 0))}-{count}"


def parse_bundle_id(bid):
    parts = bid.split("-")
    if len(parts) < 4 or parts[0] != "semantic":
        return None
    try:
        return "-".join(parts[1:-2]), int(parts[-2]), int(parts[-1])
    except ValueError:
        return None


def jaccard(a, b):
    sa = {int(x.get("id", 0)) for x in a}
    sb = {int(x.get("id", 0)) for x in b}
    return len(sa & sb) / max(1, len(sa | sb))


def persisted_accepted_ids(state):
    ids = set()
    for passage in state.get("accepted_passages", []):
        for sid in passage.get("source_stroke_ids", []):
            ids.add(int(sid))
    return ids


def build_candidates(source, state):
    intent = state["intent_card"]
    face = source["metadata"]["face_bbox"]
    already_accepted = persisted_accepted_ids(state)
    pool = [
        s for s in source["strokes"]
        if eligible(s, intent, face) and int(s.get("id", 0)) not in already_accepted
    ]
    pool.sort(key=lambda s: int(s.get("id", 0)))
    if not pool:
        raise SystemExit("no eligible semantic strokes after excluding accepted passages")

    minn = int(intent.get("min_strokes_per_candidate", 60))
    maxn = int(intent.get("max_strokes_per_candidate", 90))
    count = max(minn, min(maxn, int(round((minn + maxn) / 2))))
    wanted = int(intent.get("candidate_count", 6))
    rejected = set(state.get("rejected_bundle_ids", []))
    accepted_names = set(state.get("accepted_bundle_ids", []))

    seeds = []
    first = min(pool, key=lambda s: int(s.get("id", 0)))
    seeds.append(first)
    target_seed_count = max(24, wanted * 8)
    while len(seeds) < min(target_seed_count, len(pool)):
        best = None
        best_sep = -1.0
        for s in pool:
            if s in seeds:
                continue
            c = center(s)
            sep = min(dist(c, center(q)) for q in seeds)
            if sep > best_sep:
                best_sep, best = sep, s
        if best is None:
            break
        seeds.append(best)

    candidates = []
    for seed in seeds:
        ss = nearest_bundle(seed, pool, count)
        if len(ss) < minn:
            continue
        bid = bundle_id(intent.get("focus", "whole_face"), seed, len(ss))
        if bid in rejected or bid in accepted_names:
            continue
        if any(jaccard(ss, old[1]) >= 0.58 for old in candidates):
            continue
        candidates.append((bid, ss, center(seed)))
        if len(candidates) >= wanted * 3:
            break

    if len(candidates) < wanted:
        raise SystemExit(f"not enough semantic candidates: {len(candidates)} / {wanted}")

    chosen = [candidates[0]]
    remain = candidates[1:]
    while len(chosen) < wanted and remain:
        best = max(
            remain,
            key=lambda item: min(dist(item[2], q[2]) for q in chosen),
        )
        chosen.append(best)
        remain.remove(best)
    return pool, chosen


def reconstruct_accepted(source, state, current_pool):
    source_by_id = {int(s.get("id", 0)): s for s in source["strokes"]}
    persisted = persisted_accepted_ids(state)
    if persisted:
        missing = sorted(sid for sid in persisted if sid not in source_by_id)
        if missing:
            raise SystemExit(f"persisted accepted source ids missing: {missing[:12]}")
        return [copy.deepcopy(source_by_id[sid]) for sid in sorted(persisted)]

    accepted = []
    missing = []
    by_id = {int(s.get("id", 0)): s for s in current_pool}
    for bid in state.get("accepted_bundle_ids", []):
        parsed = parse_bundle_id(bid)
        if not parsed:
            missing.append(bid)
            continue
        _focus, seed_id, count = parsed
        seed = by_id.get(seed_id)
        if seed is None:
            missing.append(bid)
            continue
        accepted.extend(nearest_bundle(seed, current_pool, count))
    if missing:
        raise SystemExit(f"legacy accepted semantic bundle ids missing: {missing}")
    uniq = {int(s.get("id", 0)): s for s in accepted}
    return [copy.deepcopy(uniq[k]) for k in sorted(uniq)]


def clone(ss, bid, label):
    out = []
    for src in ss:
        s = copy.deepcopy(src)
        s["v9BundleId"] = bid
        s["v9CandidateLabel"] = label
        s["v9SourceStrokeId"] = src.get("id")
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_strokes")
    ap.add_argument("state")
    ap.add_argument("--out-dir", default="artifact-v9")
    args = ap.parse_args()

    source = json.loads(Path(args.source_strokes).read_text(encoding="utf-8"))
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intent = state["intent_card"]
    canvas = source["canvas"]
    width, height = int(canvas["width"]), int(canvas["height"])
    background = canvas["background"]
    base_phases = set(state.get("frozen_base_phases", ["silhouette", "light_shadow"]))
    base_strokes = [copy.deepcopy(s) for s in source["strokes"] if s.get("phase") in base_phases]

    pool, chosen = build_candidates(source, state)
    accepted_strokes = reconstruct_accepted(source, state, pool)
    accepted_source_ids = {int(s.get("id", 0)) for s in accepted_strokes}
    current_strokes = base_strokes + accepted_strokes
    base.render(current_strokes, width, height, background).save(out_dir / "current.png")

    labels = [chr(ord("A") + i) for i in range(len(chosen))]
    manifest = {
        "review_round": int(state.get("review_round", 1)),
        "intent_card": intent,
        "candidate_mode": "semantic_passage",
        "frozen_base_phases": sorted(base_phases),
        "base_stroke_count": len(base_strokes),
        "accepted_bundle_ids": list(state.get("accepted_bundle_ids", [])),
        "accepted_stroke_count": len(accepted_strokes),
        "candidate_count": len(chosen),
        "candidates": [],
        "reference_used_for_candidate_selection": False,
        "coordinates_authored_by_reviewer": False,
        "source": "exact v7 stroke geometry grouped by semantic focus",
    }

    for label, (bid, ss, seed_c) in zip(labels, chosen):
        cand_ids = {int(s.get("id", 0)) for s in ss}
        overlap = len(cand_ids & accepted_source_ids)
        if overlap:
            raise SystemExit(f"candidate {bid} overlaps accepted work by {overlap} strokes")
        cand = clone(ss, bid, label)
        base.render(current_strokes + cand, width, height, background).save(out_dir / f"candidate-{label}.png")
        manifest["candidates"].append({
            "label": label,
            "bundle_id": bid,
            "stroke_count": len(ss),
            "family": "semantic_passage",
            "brushes": dict(Counter(s.get("brush") for s in ss)),
            "phases": dict(Counter(s.get("phase") for s in ss)),
            "roles": dict(Counter(s.get("role") for s in ss)),
            "centroid_normalized": [round(seed_c[0] / width, 4), round(seed_c[1] / height, 4)],
            "overlap_with_accepted": overlap,
            "source_stroke_ids": [s.get("id") for s in ss],
        })

    trace = {
        "metadata": {
            "mode": "semantic-intent-selects-exact-v7-passages",
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
