#!/usr/bin/env python3
"""Quality refinement for observe-act v7.

Keep the human-like observe/decide/paint/reobserve loop, but avoid destroying an
already-good block-in with an overly coarse finite palette. Paint is still loaded
once and reused for several strokes; a load is now mixed from a finite regional
palette color plus the observed target at the start of the run, similar to mixing a
new pile on a physical palette rather than sampling a fresh RGB for every mark.
"""
from collections import Counter
import math

import reference_to_brush_process as base
import generate_observe_act_v7 as v7
import generate_observe_act_v7_refined as refined


_original_nearest = v7.nearest_palette


def build_palette(im, bg, sub, face, colors_per_region=20):
    p = im.load(); w, h = im.size
    counts = {r: Counter() for r in v7.REGIONS}
    # Finer but still finite palette: it remains a small set of paint piles rather
    # than direct per-stroke image sampling.
    for y in range(4, h, 5):
        for x in range(4, w, 5):
            q = p[x, y]
            r = base.region(x, y, q, bg, sub, face)
            if r in counts:
                c = tuple(int(max(0, min(255, round(ch / 10) * 10))) for ch in q)
                counts[r][c] += 1
    palettes = {}
    for r in v7.REGIONS:
        common = [c for c, _ in counts[r].most_common(colors_per_region)]
        # Preserve value extremes because highlights/shadows are visually important
        # but may be too rare to appear in a simple frequency-only palette.
        allc = list(counts[r].keys())
        if allc:
            darkest = min(allc, key=base.lum)
            lightest = max(allc, key=base.lum)
            common.extend([darkest, lightest])
        seen = []
        for c in common:
            if c not in seen: seen.append(c)
        palettes[r] = seen[:colors_per_region + 2] or [(128, 128, 128)]
    return palettes


def mixed_load(q, palette):
    # Pick one existing paint pile, then mix it toward the observed target ONCE for
    # this paint load. Subsequent strokes share the result until the next load.
    p = _original_nearest(q, palette)
    t = .52
    return tuple(int(round(p[i] * (1 - t) + q[i] * t)) for i in range(3))


v7.build_palette = build_palette
v7.nearest_palette = mixed_load


def rgb_error(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)) / 3)


def quality_mass_pass(strokes, current, ref, bg, sub, face, palettes, rng,
                      step, length, width, blur, opacity):
    """Add mid-size masses only where they improve a visibly wrong area.

    The first refined v7 painted every grid position with quantized paint, which
    degraded a good v6 block-in. Here each broad mark must justify itself from the
    current canvas error before being applied.
    """
    src = ref.filter(base.ImageFilter.GaussianBlur(blur)) if blur else ref
    p = src.load(); raw = ref.load(); w, h = ref.size
    groups = {r: [] for r in v7.REGIONS}
    off = step // 2
    for y in range(off, h, step):
        for x in range(off, w, step):
            q = p[x, y]
            r = base.region(x, y, q, bg, sub, face)
            if r not in groups: continue
            cur = current.getpixel((x, y))
            err = rgb_error(q, cur)
            # Keep already-good passages intact. Face gets a slightly lower threshold
            # because its large forms matter more to likeness.
            threshold = 9.0 if r == 'face' else 12.0
            if err < threshold: continue
            la, m = base.gradient(raw, x, y, w, h)
            angle = base.direction_field(x, y, r, sub, face, la, m, 'light_shadow', rng)
            groups[r].append((err, x, y, q, angle, m))

    for region in ('face', 'headwrap', 'garment', 'subject'):
        vals = groups[region]
        vals.sort(key=lambda t: (-t[0], int(t[2] // (step * 3)), int(t[1] // (step * 3))))
        load = None; load_left = 0; load_id = 0
        for err, x, y, q, angle, m in vals:
            if load_left <= 0:
                load = mixed_load(q, palettes[region]); load_left = rng.randint(7, 13); load_id += 1
            # Softer than the previous mass pass so it models rather than repaints.
            alpha = .36 if step >= 28 else .43
            start = len(strokes)
            base.add_flat(strokes, 'light_shadow', x, y,
                          length * (.82 if m > 24 else 1),
                          width * (.84 if m > 28 else 1),
                          base.hexcolor(load), alpha, angle,
                          rng.randrange(1, 2**31 - 1), region, None)
            s = strokes[-1]
            s['paintRun'] = 'light-shadow-mass-quality'
            s['paintLoadId'] = f'qmass-{step}-{region}-{load_id}'
            s['intent'] = 'value'; s['observationRegion'] = region
            s['loadedColor'] = base.hexcolor(load)
            base.apply_stroke(current, s)
            load_left -= 1


refined.palette_mass_pass = quality_mass_pass


if __name__ == '__main__':
    v7.main()
