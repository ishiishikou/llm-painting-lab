#!/usr/bin/env python3
"""Render the exact reference and landmark overlay used by v10 CI.

This is diagnostic only: it does not influence candidate generation or ranking.
It makes the detector result auditable before any stroke template is accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

import reference_to_brush_process as base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("current")
    ap.add_argument("landmarks")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    current = Image.open(args.current).convert("RGB")
    ref = base.crop_resize(Image.open(args.reference), current.width, current.height)
    ref.save(out / "reference.png")

    data = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    overlay = ref.copy()
    draw = ImageDraw.Draw(overlay)

    face = data.get("face")
    if face and len(face) == 4:
        x, y, w, h = face
        draw.rectangle((x, y, x + w, y + h), outline=(255, 255, 255), width=2)
        draw.text((x + 4, y + 4), "FACE", fill=(255, 255, 255))

    for label, key in (("L", "viewer_left_eye"), ("R", "viewer_right_eye")):
        item = data.get(key, {})
        box = item.get("detector_box")
        if box and len(box) == 4:
            x, y, w, h = box
            draw.rectangle((x, y, x + w, y + h), outline=(255, 255, 255), width=2)
        anchor = item.get("anchor")
        if anchor and len(anchor) == 2:
            x, y = anchor
            r = 7
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255), width=2)
            draw.line((x - 10, y, x + 10, y), fill=(255, 255, 255), width=2)
            draw.line((x, y - 10, x, y + 10), fill=(255, 255, 255), width=2)
            draw.text((x + 9, y - 12), label, fill=(255, 255, 255))

    overlay.save(out / "landmark-overlay.png")


if __name__ == "__main__":
    main()
