#!/usr/bin/env python3
"""v7 production candidate: organic block-in + local observe/act cycles.

The previous hybrid reached good pixel quality, but the early block-in still exposed the
sampling lattice as repeated capsule strokes, especially on the garment.  This variant
keeps the proven observe/act loop and changes only the broad painting stage: positions,
lengths, widths and directions vary inside coarse work clusters, and mixer/smudge strokes
are interleaved while the large masses are established.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageFilter

import reference_to_brush_process as base
import generate_observe_act_v7_hybrid as hybrid


def organic_direction(x, y, region, sub, face, local_angle, local_strength, phase, rng):
    """Blend the v5 form field with slow region-scale flow instead of a single angle."""
    a = base.direction_field(x, y, region, sub, face, local_angle, local_strength, phase, rng)
    sx0, sy0, sx1, sy1 = sub
    sw = max(1.0, sx1 - sx0)
    sh = max(1.0, sy1 - sy0)
    nx = (x - (sx0 + sx1) / 2) / sw
    ny = (y - sy0) / sh

    if region == 'garment':
        # Gravity dominates, but shoulder/fold flow bends gradually across the body.
        flow = math.pi / 2 + 0.48 * nx + 0.18 * math.sin(ny * math.pi * 2.2)
        a = base.blend_axis(a, flow, .58 if phase == 'silhouette' else .42)
    elif region == 'headwrap':
        # Wrap around the skull with a slow wave so adjacent marks are related but not parallel.
        flow = a + 0.16 * math.sin(nx * math.pi * 2.4 + ny * math.pi)
        a = base.blend_axis(a, flow, .45)
    elif region == 'face':
        # Preserve form-following face strokes and only break mechanical repetition slightly.
        a = base.axis(a + 0.07 * math.sin(nx * math.pi * 3 + ny * math.pi * 2))
    else:
        a = base.axis(a + 0.12 * math.sin(nx * math.pi * 2 + ny * math.pi * 1.5))

    jitter = .11 if phase == 'silhouette' else .075
    return base.axis(a + rng.uniform(-jitter, jitter))


def organic_broad_pass(strokes, im, phase, bg, sub, face, rng, step, length, width,
                       blur, opacity, under=False,
                       order=('headwrap', 'face', 'garment', 'subject')):
    """Paint large masses in coarse work clusters without exposing a regular grid."""
    src = im.filter(ImageFilter.GaussianBlur(blur)) if blur else im
    p = src.load()
    raw = im.load()
    w, h = im.size
    groups = defaultdict(list)

    for gy in range(step // 2, h, step):
        for gx in range(step // 2, w, step):
            # Jitter each coarse sample; the lattice must not become visible as a brush pattern.
            x = int(max(2, min(w - 3, gx + rng.uniform(-.31, .31) * step)))
            y = int(max(2, min(h - 3, gy + rng.uniform(-.31, .31) * step)))
            q = p[x, y]
            region = base.region(x, y, q, bg, sub, face)
            if region == 'background':
                continue
            local_angle, strength = base.gradient(raw, x, y, w, h)
            angle = organic_direction(x, y, region, sub, face, local_angle, strength, phase, rng)
            groups[region].append((x, y, q, angle, strength, rng.random()))

    sx0, sy0, sx1, sy1 = sub
    for region in order:
        vals = groups.get(region, [])
        rng.shuffle(vals)
        # Work one loose patch at a time; retain random ordering inside each coarse patch.
        vals.sort(key=lambda t: (int((t[1] - sy0) // max(1, step * 2.4)),
                                 int((t[0] - sx0) // max(1, step * 2.8))))
        for idx, (x, y, q, angle, strength, _) in enumerate(vals):
            edge_scale = .82 if strength > 28 else (1.08 if strength < 8 else 1.0)
            ln = length * edge_scale * rng.uniform(.72, 1.24)
            wd = width * (.84 if strength > 30 else 1.0) * rng.uniform(.76, 1.20)
            op = max(.20, min(.90, opacity + rng.uniform(-.055, .045)))
            color = base.underpaint_color(q) if under else base.hexcolor(q)
            seed = rng.randrange(1, 2**31 - 1)
            # v6 rule: do not create a hard rectangular face cutout.
            clip = None if region == 'face' else sub

            if under:
                base.add_dry(strokes, phase, x, y, ln, wd, color, op, angle, seed,
                             'underpaint-' + region, 6, clip)
                continue

            # Wet transitions are part of block-in, not a global blur after the fact.
            period = 6 if phase == 'light_shadow' else 9
            if idx % period == period - 1:
                base.add_mixer(strokes, phase, x, y, ln * .72, wd * .72, color,
                               op * .66, angle, 'mix-' + region, clip,
                               .12, .50, .30)
            elif phase == 'light_shadow' and idx % 13 == 11 and region in {'face', 'headwrap', 'garment'}:
                base.add_smudge(strokes, phase, x, y, ln * .56, wd * .60,
                                op * .54, angle, 'smudge-' + region, clip, .20, .07)
            else:
                base.add_flat(strokes, phase, x, y, ln, wd, color, op, angle, seed,
                              region, clip)


def main():
    ap = argparse.ArgumentParser(description='有機的な大面積ブロックイン後に観察→局所作業を反復するv7')
    ap.add_argument('input')
    ap.add_argument('--output', default='strokes.generated.json')
    ap.add_argument('--preview', default='preview.png')
    ap.add_argument('--metrics', default='metrics.json')
    ap.add_argument('--observation-log', default='observation-log.json')
    ap.add_argument('--checkpoint-dir')
    ap.add_argument('--width', type=int, default=864)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=20260908)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # Reset module-level visit counters so repeated invocation is deterministic.
    hybrid.VISITS = {p: Counter() for p in ('light_shadow', 'face_structure', 'detail', 'finish')}

    ref = base.crop_resize(Image.open(a.input), a.width, a.height)
    bg_rgb = base.border_bg(ref)
    bg = base.hexcolor(bg_rgb)
    sub = base.infer_subject(ref, bg_rgb)
    face = base.infer_face(ref, sub)
    palettes = hybrid.build_palette(ref, bg_rgb, sub, face)
    strokes = []
    checkpoints = []
    decisions = []
    cdir = Path(a.checkpoint_dir) if a.checkpoint_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)

    def checkpoint(stage, name, current):
        m = base.metrics(ref, current)
        checkpoints.append({'stage': stage, 'stroke_count': len(strokes), **m})
        if cdir:
            current.save(cdir / name)

    base.composition(strokes, sub, face)
    current = base.render(strokes, a.width, a.height, bg)
    checkpoint('composition', '01-composition.png', current)

    organic_broad_pass(strokes, ref, 'silhouette', bg_rgb, sub, face, rng,
                       52, 108, 35, 22, .57, True,
                       ('garment', 'headwrap', 'face', 'subject'))
    organic_broad_pass(strokes, ref, 'silhouette', bg_rgb, sub, face, rng,
                       38, 84, 28, 15, .70, False,
                       ('headwrap', 'face', 'garment', 'subject'))
    current = base.render(strokes, a.width, a.height, bg)
    checkpoint('silhouette', '02-silhouette.png', current)

    organic_broad_pass(strokes, ref, 'light_shadow', bg_rgb, sub, face, rng,
                       24, 54, 18, 9, .68, False,
                       ('face', 'headwrap', 'garment', 'subject'))
    organic_broad_pass(strokes, ref, 'light_shadow', bg_rgb, sub, face, rng,
                       16, 35, 10.5, 4, .74, False,
                       ('face', 'headwrap', 'garment', 'subject'))
    current = base.render(strokes, a.width, a.height, bg)
    cycle = 1
    loads = 0
    cycle, loads = hybrid.loop(strokes, current, ref, bg_rgb, sub, face, palettes,
                               'light_shadow', 8, 46, 7, 68, rng, decisions, cycle, loads)
    checkpoint('light_shadow', '03-light-shadow.png', current)

    start = len(strokes)
    base.face_structure(strokes, ref, sub, face, rng)
    hybrid.apply_new(current, strokes, start)
    cycle, loads = hybrid.loop(strokes, current, ref, bg_rgb, sub, face, palettes,
                               'face_structure', 10, 44, 6, 58, rng, decisions, cycle, loads)
    checkpoint('face_structure', '04-face-structure.png', current)

    cycle, loads = hybrid.loop(strokes, current, ref, bg_rgb, sub, face, palettes,
                               'detail', 130, 76, 3, 42, rng, decisions, cycle, loads)
    checkpoint('detail', '05-detail.png', current)

    start = len(strokes)
    base.glaze_pass(strokes, ref, bg_rgb, sub, face, rng)
    hybrid.apply_new(current, strokes, start)
    cycle, loads = hybrid.loop(strokes, current, ref, bg_rgb, sub, face, palettes,
                               'finish', 90, 64, 2, 34, rng, decisions, cycle, loads)
    checkpoint('finish', '06-finish.png', current)
    current.save(a.preview)

    for i, s in enumerate(strokes, 1):
        s['id'] = i

    direction = base.direction_stats(strokes)
    artifact = hybrid.artifact_checks(strokes, face)
    used_loads = [s.get('paintLoadId') for s in strokes if s.get('paintLoadId')]
    brush_counts = Counter(s['brush'] for s in strokes)
    metadata = {
        'slug': 'observe-act-v7-final',
        'title': '有機的なブロックイン後に観察して局所作業を反復する描画工程',
        'seed': a.seed,
        'source_mode': 'reference-guided-observe-act-v7',
        'stroke_count': len(strokes),
        'subject_bbox': [round(v, 2) for v in sub],
        'face_bbox': [round(v, 2) for v in face],
        'phase_order': [p[0] for p in base.PHASES],
        'background_policy': 'toned-ground-no-progress',
        'tool_policy': 'docs/painting-tool-rules.md',
        'evaluation_policy': 'docs/painting-evaluation-rules.md',
        'direction_policy': 'region-form-following-v5',
        'block_in_policy': 'jittered-clustered-form-flow-with-wet-transitions-v7',
        'artifact_policy': 'no-hard-rectangular-face-mask-v6',
        'observe_act_policy': 'broad-block-in-then-observe-local-run-reobserve-v7',
        'palette_policy': 'persistent-local-mixed-load-v7',
        'palette': {r: [base.hexcolor(c) for c in palettes[r]] for r in hybrid.REGIONS},
        'decision_cycles': len(decisions),
        'paint_load_count': len(set(used_loads)),
        'direction_stats': direction,
        'artifact_checks': artifact,
        'quality_metrics': checkpoints[-1],
    }
    data = {
        'metadata': metadata,
        'canvas': {'width': a.width, 'height': a.height, 'background': bg},
        'phases': [{'id': i, 'label': label} for i, label in base.PHASES],
        'strokes': strokes,
    }
    Path(a.output).write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    Path(a.observation_log).write_text(
        json.dumps({'decisions': decisions}, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(a.metrics).write_text(json.dumps({
        'checkpoints': checkpoints,
        'direction_stats': direction,
        'artifact_checks': artifact,
        'decision_summary': {
            'cycles': len(decisions),
            'regions': dict(Counter(d['region'] for d in decisions)),
            'intents': dict(Counter(d['intent'] for d in decisions)),
            'paint_loads': len(set(used_loads)),
        },
        'brush_counts': dict(brush_counts),
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'strokes': len(strokes),
        'phase_counts': {p: sum(1 for s in strokes if s['phase'] == p) for p, _ in base.PHASES},
        'brush_counts': dict(brush_counts),
        'cycles': len(decisions),
        'paint_loads': len(set(used_loads)),
        'artifact_checks': artifact,
        'quality': checkpoints[-1],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
