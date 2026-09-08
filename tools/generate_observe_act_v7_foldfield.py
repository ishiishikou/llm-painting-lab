#!/usr/bin/env python3
"""Final v7 tuning: replace the garment's vertical capsule rows with coherent fold families."""
import math
from collections import defaultdict

from PIL import ImageFilter

import reference_to_brush_process as base
import generate_observe_act_v7_final as final


def fold_direction(x, y, region, sub, face, local_angle, local_strength, phase, rng):
    a = base.direction_field(x, y, region, sub, face, local_angle, local_strength, phase, rng)
    sx0, sy0, sx1, sy1 = sub
    sw = max(1.0, sx1 - sx0)
    sh = max(1.0, sy1 - sy0)
    u = max(0.0, min(1.0, (x - sx0) / sw))
    v = max(0.0, min(1.0, (y - sy0) / sh))

    if region == 'garment':
        # A garment is not one vertical field.  Treat it as several broad fold families.
        # The sine field changes continuously across x, so neighbouring strokes share a fold,
        # while separate folds lean in visibly different directions.
        fold = .62 * math.sin(u * math.pi * 3.15 + v * math.pi * .52)
        fold += .13 * math.sin(v * math.pi * 2.35)
        if v < .67:
            fold += .16 * (u - .5) * 2
        flow = base.axis(math.pi / 2 + fold)
        a = base.blend_axis(a, flow, .90 if phase == 'silhouette' else .78)
        return base.axis(a + rng.uniform(-.075, .075))

    if region == 'headwrap':
        flow = base.axis(a + .20 * math.sin(u * math.pi * 2.5 + v * math.pi * .8))
        return base.blend_axis(a, flow, .48)
    if region == 'face':
        return base.axis(a + .055 * math.sin(u * math.pi * 3.2 + v * math.pi * 2.0))
    return base.axis(a + .14 * math.sin(u * math.pi * 2.1 + v * math.pi * 1.4))


def fold_broad_pass(strokes, im, phase, bg, sub, face, rng, step, length, width,
                    blur, opacity, under=False,
                    order=('headwrap', 'face', 'garment', 'subject')):
    src = im.filter(ImageFilter.GaussianBlur(blur)) if blur else im
    p = src.load()
    raw = im.load()
    w, h = im.size
    groups = defaultdict(list)

    for gy in range(step // 2, h, step):
        for gx in range(step // 2, w, step):
            x = int(max(2, min(w - 3, gx + rng.uniform(-.34, .34) * step)))
            y = int(max(2, min(h - 3, gy + rng.uniform(-.34, .34) * step)))
            q = p[x, y]
            region = base.region(x, y, q, bg, sub, face)
            if region == 'background':
                continue
            local_angle, strength = base.gradient(raw, x, y, w, h)
            angle = fold_direction(x, y, region, sub, face, local_angle, strength, phase, rng)
            groups[region].append((x, y, q, angle, strength))

    sx0, sy0, sx1, sy1 = sub
    for region in order:
        vals = groups.get(region, [])
        rng.shuffle(vals)
        # Work by large passages rather than scanline order.
        vals.sort(key=lambda t: (int((t[1] - sy0) // max(1, step * 2.6)),
                                 int((t[0] - sx0) // max(1, step * 3.0))))
        for idx, (x, y, q, angle, strength) in enumerate(vals):
            # Fewer, broader garment strokes remove the 'picket fence' of rounded bars.
            if region == 'garment' and idx % 3 == 2:
                continue
            edge_scale = .82 if strength > 28 else (1.08 if strength < 8 else 1.0)
            region_len = 1.48 if region == 'garment' else 1.0
            region_wid = 1.38 if region == 'garment' else 1.0
            ln = length * edge_scale * region_len * rng.uniform(.76, 1.27)
            wd = width * region_wid * (.84 if strength > 30 else 1.0) * rng.uniform(.80, 1.22)
            op = max(.20, min(.90, opacity + rng.uniform(-.055, .045)))
            color = base.underpaint_color(q) if under else base.hexcolor(q)
            seed = rng.randrange(1, 2**31 - 1)
            clip = None if region == 'face' else sub

            if under:
                base.add_dry(strokes, phase, x, y, ln, wd, color, op, angle, seed,
                             'underpaint-' + region, 7 if region == 'garment' else 6, clip)
                continue

            # Blend while the passage is wet.  Mixer/smudge are local brush actions, not filters.
            period = 5 if region == 'garment' else (6 if phase == 'light_shadow' else 9)
            if idx % period == period - 1:
                base.add_mixer(strokes, phase, x, y, ln * .76, wd * .70, color,
                               op * .64, angle, 'mix-' + region, clip, .12, .52, .30)
            elif phase == 'light_shadow' and idx % 11 == 8 and region in {'face', 'headwrap', 'garment'}:
                base.add_smudge(strokes, phase, x, y, ln * .58, wd * .58,
                                op * .52, angle, 'smudge-' + region, clip, .20, .07)
            else:
                base.add_flat(strokes, phase, x, y, ln, wd, color, op, angle, seed,
                              region, clip)


final.organic_broad_pass = fold_broad_pass

if __name__ == '__main__':
    final.main()
