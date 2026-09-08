#!/usr/bin/env python3
"""Stateful wrapper for semantic v9 passages.

Accepted passages can be frozen by exact v7 source stroke ids. This lets the
reviewer move from eyes to nose/mouth/other regions without losing earlier work.
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


semantic._original_reconstruct_accepted = semantic.reconstruct_accepted
semantic.reconstruct_accepted = reconstruct_accepted

if __name__ == "__main__":
    semantic.main()
