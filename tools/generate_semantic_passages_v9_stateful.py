#!/usr/bin/env python3
"""Stateful wrapper for semantic v9 passages.

Accepted passages are frozen by exact v7 source-stroke ids.  The reviewer only
changes semantic intent; no stroke coordinates are authored here.

This wrapper also adds a few conservative colour families used by later visual
review rounds.  They only filter already-existing v7 strokes; they do not create,
move or recolour any stroke.
"""
from __future__ import annotations

import copy

import generate_semantic_passages_v9 as semantic


def reconstruct_accepted(source, state, pool):
    explicit = state.get("accepted_passages", [])
    if not explicit:
        return semantic._original_reconstruct_accepted(source, state, pool)

    by_id = {int(s.get("id", 0)): s for s in source["strokes"]}
    uniq = {}
    missing = []
    for passage in explicit:
        bid = passage.get("bundle_id", "accepted")
        for sid in passage.get("source_stroke_ids", []):
            s = by_id.get(int(sid))
            if s is None:
                missing.append((bid, sid))
            else:
                uniq[int(sid)] = s
    if missing:
        raise SystemExit(f"accepted source strokes missing: {missing[:8]}")
    return [copy.deepcopy(uniq[k]) for k in sorted(uniq)]


def extended_color_family_ok(s, intent):
    family = intent.get("color_family")
    if family not in {"eye_dark", "face_mid", "skin_light"}:
        return semantic._original_color_family_ok(s, intent)

    color = s.get("color")
    if not color:
        return False
    try:
        p = semantic.base.parse_hex(color)
    except Exception:
        return False
    r, g, b = p
    l = semantic.base.lum(p)

    if family == "eye_dark":
        # Dark brown/neutral facial accents.  The warm-neutral guard rejects the
        # blue headwrap even when an old face classification was imperfect.
        warm_neutral = r >= b * 0.82 and g >= b * 0.72 and r >= g * 0.72
        return warm_neutral and 18 <= l <= 112
    if family == "face_mid":
        warm_neutral = r >= b * 0.90 and g >= b * 0.78 and r >= g * 0.82
        return warm_neutral and 105 <= l <= 205
    if family == "skin_light":
        return semantic.base.skin(p) and l >= 170
    return False


semantic._original_reconstruct_accepted = semantic.reconstruct_accepted
semantic.reconstruct_accepted = reconstruct_accepted
semantic._original_color_family_ok = semantic.color_family_ok
semantic.color_family_ok = extended_color_family_ok

if __name__ == "__main__":
    semantic.main()
