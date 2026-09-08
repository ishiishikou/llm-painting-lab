#!/usr/bin/env python3
"""v8: render only the explicit stroke actions written in a visual-review plan.

This generator intentionally has no reference-image input.  It does not sample source
pixels, rank residuals, infer a face box, or decide what to paint next.  A reviewer must
look at the reference and the current preview, then append a small, semantically meaningful
action to the JSON plan.  The renderer executes that decision deterministically.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import reference_to_brush_process as base

PHASES = [
    ('composition', '構図'),
    ('silhouette', '大きな形'),
    ('light_shadow', '明暗の面'),
    ('face_structure', '顔構造'),
    ('detail', '細部'),
    ('finish', '仕上げ'),
]


def norm_xy(p, width, height):
    return float(p[0]) * width, float(p[1]) * height


def bbox_px(b, width, height):
    return (float(b[0]) * width, float(b[1]) * height,
            float(b[2]) * width, float(b[3]) * height)


def point_in_poly(x, y, pts):
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        cross = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if cross:
            inside = not inside
        j = i
    return inside


def shape_sampler(shape, width, height, rng):
    st = shape.get('type', 'ellipse')
    if st == 'ellipse':
        x0, y0, x1, y1 = bbox_px(shape['bbox'], width, height)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2

        def sample():
            # Uniform area sampling avoids a rectangular lattice or perimeter ring.
            r = math.sqrt(rng.random())
            t = rng.random() * math.tau
            return cx + math.cos(t) * rx * r, cy + math.sin(t) * ry * r

        return sample, (x0, y0, x1, y1)

    if st == 'polygon':
        pts = [norm_xy(p, width, height) for p in shape['points']]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bounds = (min(xs), min(ys), max(xs), max(ys))

        def sample():
            x0, y0, x1, y1 = bounds
            for _ in range(400):
                x = rng.uniform(x0, x1)
                y = rng.uniform(y0, y1)
                if point_in_poly(x, y, pts):
                    return x, y
            # Degenerate polygons still produce a deterministic fallback.
            return sum(xs) / len(xs), sum(ys) / len(ys)

        return sample, bounds

    if st == 'box':
        x0, y0, x1, y1 = bbox_px(shape['bbox'], width, height)

        def sample():
            return rng.uniform(x0, x1), rng.uniform(y0, y1)

        return sample, (x0, y0, x1, y1)

    raise ValueError(f'unsupported shape type: {st}')


def flow_angle(action, x, y, bounds, rng):
    flow = action.get('flow', 'fixed')
    base_deg = float(action.get('angle_deg', 0.0))
    jitter = math.radians(float(action.get('angle_jitter_deg', 0.0)))
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx = max(1.0, (x1 - x0) / 2)
    ry = max(1.0, (y1 - y0) / 2)

    if flow == 'ellipse_tangent':
        # Tangent to the ellipse through the sampled point.  This is geometry from the
        # reviewer-specified shape, not a field extracted from the reference image.
        nx = (x - cx) / (rx * rx)
        ny = (y - cy) / (ry * ry)
        angle = math.atan2(ny, nx) + math.pi / 2
    elif flow == 'vertical_folds':
        u = max(0.0, min(1.0, (x - x0) / max(1.0, x1 - x0)))
        v = max(0.0, min(1.0, (y - y0) / max(1.0, y1 - y0)))
        bend = math.radians(float(action.get('bend_deg', 24.0)))
        freq = float(action.get('frequency', 2.4))
        angle = math.pi / 2 + bend * math.sin((u * freq + v * .35) * math.pi)
    elif flow == 'horizontal_wrap':
        u = max(0.0, min(1.0, (x - x0) / max(1.0, x1 - x0)))
        v = max(0.0, min(1.0, (y - y0) / max(1.0, y1 - y0)))
        bend = math.radians(float(action.get('bend_deg', 16.0)))
        angle = math.radians(base_deg) + bend * ((v - .5) * 1.4 + .35 * math.sin(u * math.pi))
    else:
        angle = math.radians(base_deg)

    return base.axis(angle + rng.uniform(-jitter, jitter))


def add_cluster(strokes, action, width, height, rng):
    sample, bounds = shape_sampler(action['shape'], width, height, rng)
    brush = action['brush']
    count = int(action.get('count', 1))
    color = action.get('color', '#808080')
    opacity = float(action.get('opacity', .65))
    base_len = float(action.get('length', 40.0))
    base_width = float(action.get('width', 14.0))
    length_jitter = float(action.get('length_jitter', .18))
    width_jitter = float(action.get('width_jitter', .14))
    phase = action.get('phase', 'silhouette')
    role = action.get('role', action['id'])

    for n in range(count):
        x, y = sample()
        angle = flow_angle(action, x, y, bounds, rng)
        ln = max(1.0, base_len * (1 + rng.uniform(-length_jitter, length_jitter)))
        wd = max(.5, base_width * (1 + rng.uniform(-width_jitter, width_jitter)))
        seed = rng.randrange(1, 2**31 - 1)
        start = len(strokes)

        if brush == 'flatBrush':
            base.add_flat(strokes, phase, x, y, ln, wd, color, opacity, angle, seed, role, None)
        elif brush == 'dryBrush':
            base.add_dry(strokes, phase, x, y, ln, wd, color, opacity, angle, seed,
                         role, int(action.get('strands', 7)), None)
        elif brush == 'variableBrush':
            base.add_variable(strokes, phase, x, y, ln, wd,
                              max(.55, wd * float(action.get('taper', .36))),
                              color, opacity, angle, role, None)
        elif brush == 'line':
            base.add_line(strokes, phase, x, y, ln, wd, color, opacity, angle, role)
        elif brush == 'mixerBrush':
            base.add_mixer(strokes, phase, x, y, ln, wd, color, opacity, angle, role, None,
                           float(action.get('pickup', .14)),
                           float(action.get('mix', .50)),
                           float(action.get('deposit', .30)))
        elif brush == 'smudgeBrush':
            base.add_smudge(strokes, phase, x, y, ln, wd, opacity, angle, role, None,
                            float(action.get('strength', .22)),
                            float(action.get('pickup', .08)))
        elif brush == 'glaze':
            base.add_glaze(strokes, phase, x, y, ln, wd, color, opacity, angle, role, None)
        else:
            raise ValueError(f'unsupported brush: {brush}')

        for s in strokes[start:]:
            s['reviewActionId'] = action['id']
            s['reviewRound'] = int(action.get('review_round', 0))
            s['decision'] = action.get('decision', '')
            s['intent'] = action.get('intent', '')
            s['paintLoadId'] = action.get('paint_load_id', action['id'])
            s['actionStrokeIndex'] = n + 1


def add_segments(strokes, action, width, height):
    phase = action.get('phase', 'face_structure')
    brush = action.get('brush', 'variableBrush')
    color = action.get('color', '#888888')
    opacity = float(action.get('opacity', .65))
    role = action.get('role', action['id'])
    for n, seg in enumerate(action['segments'], 1):
        x1, y1 = norm_xy(seg['from'], width, height)
        x2, y2 = norm_xy(seg['to'], width, height)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        wd = float(seg.get('width', action.get('width', 2.0)))
        start = len(strokes)
        if brush == 'line':
            base.add_line(strokes, phase, x, y, length, wd, color, opacity, angle, role)
        else:
            taper = float(seg.get('taper', action.get('taper', .35)))
            base.add_variable(strokes, phase, x, y, length, wd, max(.5, wd * taper),
                              color, opacity, angle, role, None)
        for s in strokes[start:]:
            s['reviewActionId'] = action['id']
            s['reviewRound'] = int(action.get('review_round', 0))
            s['decision'] = action.get('decision', '')
            s['intent'] = action.get('intent', '')
            s['paintLoadId'] = action.get('paint_load_id', action['id'])
            s['actionStrokeIndex'] = n


def main():
    ap = argparse.ArgumentParser(description='AIの視覚レビュー指示だけを実行するv8描画器')
    ap.add_argument('plan')
    ap.add_argument('--output', default='strokes.generated.json')
    ap.add_argument('--preview', default='preview.png')
    ap.add_argument('--width', type=int)
    ap.add_argument('--height', type=int)
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    width = args.width or int(plan.get('canvas', {}).get('width', 864))
    height = args.height or int(plan.get('canvas', {}).get('height', 1024))
    background = plan.get('canvas', {}).get('background', '#21180d')
    seed = int(plan.get('seed', 20260908))
    strokes = []
    enabled_actions = []

    for idx, action in enumerate(plan.get('actions', []), 1):
        if not action.get('enabled', True):
            continue
        local_rng = random.Random(seed ^ (idx * 0x9E3779B1) ^ int(action.get('seed', idx)))
        if action.get('kind', 'cluster') == 'segments':
            add_segments(strokes, action, width, height)
        else:
            add_cluster(strokes, action, width, height, local_rng)
        enabled_actions.append(action['id'])

    for i, s in enumerate(strokes, 1):
        s['id'] = i

    image = base.render(strokes, width, height, background)
    image.save(args.preview)
    brushes = Counter(s['brush'] for s in strokes)
    phases = Counter(s['phase'] for s in strokes)
    metadata = {
        'slug': 'vision-guided-v8',
        'title': '途中画像をAIが実際に見て指示したストロークだけで描くv8',
        'source_mode': 'review-plan-only-no-reference-access',
        'reference_access_in_generator': False,
        'stroke_count': len(strokes),
        'review_round': int(plan.get('review_round', 0)),
        'enabled_actions': enabled_actions,
        'phase_counts': dict(phases),
        'brush_counts': dict(brushes),
        'plan_file': str(args.plan),
    }
    data = {
        'metadata': metadata,
        'canvas': {'width': width, 'height': height, 'background': background},
        'phases': [{'id': p, 'label': label} for p, label in PHASES],
        'strokes': strokes,
    }
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')),
                                 encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == '__main__':
    main()
