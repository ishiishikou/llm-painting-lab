#!/usr/bin/env python3
"""v7: paint in short observe -> decide -> act loops instead of one huge residual pass.

The top-level six stages stay familiar, but the implementation changes from
"rank thousands of pixels, then paint them all" to repeated short work sessions:
observe the current canvas, choose the most important region/problem, load one
palette color, paint a small connected run, then look again.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageFilter

import reference_to_brush_process as base
import generate_artifact_cleanup_v6 as cleanup  # installs v5 direction + v6 organic face handling

REGIONS = ('face', 'headwrap', 'garment', 'subject')
REGION_WEIGHT = {'face': 1.45, 'headwrap': 1.12, 'garment': .88, 'subject': .72}


def bucket_color(q, step=24):
    return tuple(int(max(0, min(255, round(v / step) * step))) for v in q)


def build_palette(im, bg, sub, face, colors_per_region=8):
    p = im.load(); w, h = im.size
    counts = {r: Counter() for r in REGIONS}
    for y in range(4, h, 7):
        for x in range(4, w, 7):
            q = p[x, y]
            r = base.region(x, y, q, bg, sub, face)
            if r in counts:
                counts[r][bucket_color(q)] += 1
    palettes = {}
    for r in REGIONS:
        vals = [c for c, _ in counts[r].most_common(colors_per_region)]
        palettes[r] = vals or [(128, 128, 128)]
    return palettes


def nearest_palette(q, palette):
    return min(palette, key=lambda c: base.dist(q, c))


def coarse_observe(ref, current, bg, sub, face, step=10):
    rp, cp = ref.load(), current.load(); w, h = ref.size
    acc = {r: {'n': 0, 'rgb': 0.0, 'value': 0.0, 'edge': 0.0} for r in REGIONS}
    for y in range(2, h - 2, step):
        for x in range(2, w - 2, step):
            q = rp[x, y]
            r = base.region(x, y, q, bg, sub, face)
            if r not in acc:
                continue
            c = cp[x, y]
            a, ma = base.gradient(rp, x, y, w, h)
            b, mb = base.gradient(cp, x, y, w, h)
            d = math.sqrt(sum((q[i] - c[i]) ** 2 for i in range(3)) / 3)
            v = abs(base.lum(q) - base.lum(c))
            e = abs(ma - mb)
            z = acc[r]; z['n'] += 1; z['rgb'] += d; z['value'] += v; z['edge'] += e
    out = {}
    for r, z in acc.items():
        n = max(1, z['n'])
        rgb = z['rgb'] / n; value = z['value'] / n; edge = z['edge'] / n
        chroma = max(0.0, rgb - value * .58)
        out[r] = {
            'count': z['n'],
            'rgb_mae': round(rgb, 3),
            'value_mae': round(value, 3),
            'edge_mae': round(edge, 3),
            'chroma_mae': round(chroma, 3),
            'score': round(rgb + edge * .60, 3),
        }
    return out


def choose_problem(obs, phase, recent_regions):
    choices = []
    for r in REGIONS:
        z = obs[r]
        fatigue = .74 if len(recent_regions) >= 2 and recent_regions[-1] == recent_regions[-2] == r else 1.0
        choices.append((z['score'] * REGION_WEIGHT[r] * fatigue, r))
    _, region = max(choices)
    z = obs[region]
    if phase == 'light_shadow':
        intent = 'value' if z['value_mae'] >= z['edge_mae'] * .72 else 'edge'
    elif phase == 'face_structure':
        intent = 'structure' if region == 'face' else ('edge' if z['edge_mae'] > z['chroma_mae'] else 'color')
    elif phase == 'detail':
        intent = 'structure' if region == 'face' and z['edge_mae'] > 5 else ('edge' if z['edge_mae'] > z['chroma_mae'] else 'color')
    else:
        intent = 'edge' if z['edge_mae'] > z['chroma_mae'] * .75 else 'color'
    return region, intent


def candidate_cluster(ref, current, bg, sub, face, region, intent, rng, count, stride, radius):
    rp, cp = ref.load(), current.load(); w, h = ref.size
    candidates = []
    offx, offy = rng.randrange(stride), rng.randrange(stride)
    for y in range(2 + offy, h - 2, stride):
        for x in range(2 + offx, w - 2, stride):
            q = rp[x, y]
            if base.region(x, y, q, bg, sub, face) != region:
                continue
            c = cp[x, y]
            a, m = base.gradient(rp, x, y, w, h)
            _, cm = base.gradient(cp, x, y, w, h)
            rgb = math.sqrt(sum((q[i] - c[i]) ** 2 for i in range(3)) / 3)
            value = abs(base.lum(q) - base.lum(c))
            edge = abs(m - cm)
            if intent == 'value': score = value * 1.2 + rgb * .35
            elif intent in {'edge', 'structure'}: score = edge * 1.15 + rgb * .42
            else: score = rgb
            if score > 3.0:
                candidates.append((score, x, y, q, a, m))
    if not candidates:
        return []
    candidates.sort(reverse=True)
    seed = candidates[0]
    sx, sy = seed[1], seed[2]
    local = [v for v in candidates if (v[1]-sx)**2 + (v[2]-sy)**2 <= radius*radius]
    local.sort(reverse=True)
    return local[:count]


def mark_new(strokes, start, cycle, run_id, load_id, intent, region, color):
    for s in strokes[start:]:
        s['decisionCycle'] = cycle
        s['paintRun'] = run_id
        s['paintLoadId'] = load_id
        s['intent'] = intent
        s['observationRegion'] = region
        s['loadedColor'] = color


def apply_new(current, strokes, start):
    for s in strokes[start:]:
        base.apply_stroke(current, s)


def paint_run(strokes, current, ref, bg, sub, face, palettes, phase, region, intent,
              candidates, rng, cycle, run_id, load_counter):
    if not candidates:
        return 0, load_counter
    # One load is intentionally reused for several strokes. This is closer to a real
    # palette/brush action than sampling a fresh RGB value for every independent mark.
    made = 0
    i = 0
    while i < len(candidates):
        block = candidates[i:i + rng.randint(7, 13)]
        q0 = block[0][3]
        load = nearest_palette(q0, palettes[region])
        load_hex = base.hexcolor(load)
        load_counter += 1
        start = len(strokes)
        for score, x, y, q, local_angle, m in block:
            angle = base.direction_field(x, y, region, sub, face, local_angle, m, phase, rng)
            if phase == 'light_shadow':
                if made % 5 == 4:
                    base.add_mixer(strokes, phase, x, y, 20 if region == 'face' else 28,
                                   8 if region == 'face' else 11, load_hex, .48, angle,
                                   f'mix-{region}', None, .14, .52, .34)
                else:
                    base.add_flat(strokes, phase, x, y, 28 if region == 'face' else 38,
                                  10 if region == 'face' else 14, load_hex, .52, angle,
                                  rng.randrange(1, 2**31-1), region, None)
            elif phase == 'face_structure':
                if region == 'face' and made % 6 != 5:
                    base.add_variable(strokes, phase, x, y, 8.0, 4.2, 1.4, load_hex, .72,
                                      angle, 'face-plane', None)
                else:
                    base.add_mixer(strokes, phase, x, y, 14, 7, load_hex, .42, angle,
                                   f'mix-{region}', None, .12, .50, .30)
            elif phase == 'detail':
                if made % 5:
                    base.add_variable(strokes, phase, x, y, 6.2 if region == 'face' else 7.2,
                                      2.6, .75, load_hex, .82, angle, region, None)
                else:
                    base.add_line(strokes, phase, x, y, 4.6 if region == 'face' else 5.4,
                                  1.35 if region == 'face' else 1.65, load_hex, .86, angle, region)
            else:
                if intent == 'color' and made % 5 == 4:
                    base.add_mixer(strokes, phase, x, y, 12, 6.5, load_hex, .34, angle,
                                   f'mix-{region}', None, .10, .46, .26)
                elif made % 4:
                    base.add_variable(strokes, phase, x, y, 4.4 if region == 'face' else 5.2,
                                      1.9, .55, load_hex, .76, angle, region, None)
                else:
                    base.add_line(strokes, phase, x, y, 3.2 if region == 'face' else 3.8,
                                  1.05 if region == 'face' else 1.25, load_hex, .82, angle, region)
            made += 1
        mark_new(strokes, start, cycle, run_id, load_counter, intent, region, load_hex)
        apply_new(current, strokes, start)
        i += len(block)
    return made, load_counter


def run_observe_act(strokes, current, ref, bg, sub, face, palettes, phase, cycles,
                    batch, stride, radius, rng, log, start_cycle, load_counter):
    recent = []
    for j in range(cycles):
        cycle = start_cycle + j
        before = coarse_observe(ref, current, bg, sub, face)
        region, intent = choose_problem(before, phase, recent)
        candidates = candidate_cluster(ref, current, bg, sub, face, region, intent, rng,
                                       batch, stride, radius)
        run_id = f'{phase}-{cycle:03d}-{region}'
        made, load_counter = paint_run(strokes, current, ref, bg, sub, face, palettes,
                                       phase, region, intent, candidates, rng, cycle,
                                       run_id, load_counter)
        after = coarse_observe(ref, current, bg, sub, face)
        log.append({
            'cycle': cycle, 'phase': phase, 'region': region, 'intent': intent,
            'stroke_count': made,
            'before_score': before[region]['score'],
            'after_score': after[region]['score'],
            'global_before': round(sum(before[r]['score'] for r in REGIONS), 3),
            'global_after': round(sum(after[r]['score'] for r in REGIONS), 3),
        })
        recent.append(region); recent = recent[-3:]
    return start_cycle + cycles, load_counter


def artifact_checks(strokes, face_box):
    fb = [round(v, 2) for v in face_box]
    face_strokes = [s for s in strokes if cleanup.face_role(s.get('role', ''))]
    return {
        'face_role_strokes': len(face_strokes),
        'face_role_hard_clip_count': sum(1 for s in face_strokes if s.get('clipBox')),
        'exact_face_bbox_clip_count': sum(1 for s in strokes if s.get('clipBox') == fb),
    }


def main():
    ap = argparse.ArgumentParser(description='観察→判断→数筆→再観察で進む油彩風v7')
    ap.add_argument('input'); ap.add_argument('--output', default='strokes.generated.json')
    ap.add_argument('--preview', default='preview.png'); ap.add_argument('--metrics', default='metrics.json')
    ap.add_argument('--observation-log', default='observation-log.json'); ap.add_argument('--checkpoint-dir')
    ap.add_argument('--width', type=int, default=864); ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=20260908)
    a = ap.parse_args(); rng = random.Random(a.seed)

    ref = base.crop_resize(Image.open(a.input), a.width, a.height)
    bg_rgb = base.border_bg(ref); bg = base.hexcolor(bg_rgb)
    sub = base.infer_subject(ref, bg_rgb); face = base.infer_face(ref, sub)
    palettes = build_palette(ref, bg_rgb, sub, face)
    strokes = []; checkpoints = []; decisions = []; cdir = Path(a.checkpoint_dir) if a.checkpoint_dir else None
    if cdir: cdir.mkdir(parents=True, exist_ok=True)

    def save_cp(stage, name, current):
        m = base.metrics(ref, current)
        checkpoints.append({'stage': stage, 'stroke_count': len(strokes), **m})
        if cdir: current.save(cdir / name)

    base.composition(strokes, sub, face)
    current = base.render(strokes, a.width, a.height, bg)
    save_cp('composition', '01-composition.png', current)

    # Big-shape stage remains deliberately broad. It establishes the painting before
    # iterative diagnosis begins.
    base.broad_pass(strokes, ref, 'silhouette', bg_rgb, sub, face, rng, 52, 105, 34, 22, .58, True,
                    ('garment','headwrap','face','subject'))
    base.broad_pass(strokes, ref, 'silhouette', bg_rgb, sub, face, rng, 38, 82, 27, 15, .72, False,
                    ('headwrap','face','garment','subject'))
    current = base.render(strokes, a.width, a.height, bg)
    save_cp('silhouette', '02-silhouette.png', current)

    cycle = 1; load_counter = 0
    cycle, load_counter = run_observe_act(strokes, current, ref, bg_rgb, sub, face, palettes,
                                          'light_shadow', 12, 42, 10, 105, rng, decisions,
                                          cycle, load_counter)
    # A small local blend follows observation instead of a single global blend pass.
    save_cp('light_shadow', '03-light-shadow.png', current)

    # Establish a few facial construction lines, then alternate observation and local work.
    base.face_structure(strokes, ref, sub, face, rng)
    current = base.render(strokes, a.width, a.height, bg)
    cycle, load_counter = run_observe_act(strokes, current, ref, bg_rgb, sub, face, palettes,
                                          'face_structure', 10, 36, 8, 82, rng, decisions,
                                          cycle, load_counter)
    save_cp('face_structure', '04-face-structure.png', current)

    cycle, load_counter = run_observe_act(strokes, current, ref, bg_rgb, sub, face, palettes,
                                          'detail', 34, 54, 5, 72, rng, decisions,
                                          cycle, load_counter)
    save_cp('detail', '05-detail.png', current)

    # Glaze is kept sparse and late, then the painter looks again between short finishing runs.
    start = len(strokes); base.glaze_pass(strokes, ref, bg_rgb, sub, face, rng); apply_new(current, strokes, start)
    cycle, load_counter = run_observe_act(strokes, current, ref, bg_rgb, sub, face, palettes,
                                          'finish', 20, 42, 4, 60, rng, decisions,
                                          cycle, load_counter)
    save_cp('finish', '06-finish.png', current); current.save(a.preview)

    for i, s in enumerate(strokes, 1): s['id'] = i
    dstats = base.direction_stats(strokes)
    art = artifact_checks(strokes, face)
    phase_counts = {p: sum(1 for s in strokes if s['phase'] == p) for p, _ in base.PHASES}
    brush_counts = Counter(s['brush'] for s in strokes)
    loaded = [s.get('paintLoadId') for s in strokes if s.get('paintLoadId')]
    metadata = {
        'slug': 'observe-act-v7', 'title': '観察して数筆ずつ進める描画工程', 'seed': a.seed,
        'source_mode': 'reference-guided-observe-act-v7', 'stroke_count': len(strokes),
        'subject_bbox': [round(v,2) for v in sub], 'face_bbox': [round(v,2) for v in face],
        'phase_order': [p[0] for p in base.PHASES], 'background_policy': 'toned-ground-no-progress',
        'tool_policy': 'docs/painting-tool-rules.md', 'evaluation_policy': 'docs/painting-evaluation-rules.md',
        'direction_policy': 'region-form-following-v5', 'artifact_policy': 'no-hard-rectangular-face-mask-v6',
        'observe_act_policy': 'observe-decide-local-run-reobserve-v7',
        'palette_policy': 'persistent-load-quantized-regional-palette-v7',
        'palette': {r: [base.hexcolor(c) for c in palettes[r]] for r in REGIONS},
        'decision_cycles': len(decisions), 'paint_load_count': len(set(loaded)),
        'direction_stats': dstats, 'artifact_checks': art,
        'quality_metrics': checkpoints[-1],
    }
    data = {'metadata': metadata, 'canvas': {'width': a.width, 'height': a.height, 'background': bg},
            'phases': [{'id': i, 'label': l} for i,l in base.PHASES], 'strokes': strokes}
    Path(a.output).write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    Path(a.observation_log).write_text(json.dumps({'decisions': decisions}, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(a.metrics).write_text(json.dumps({'checkpoints': checkpoints, 'direction_stats': dstats,
                                          'artifact_checks': art, 'decision_summary': {
                                              'cycles': len(decisions),
                                              'regions': dict(Counter(d['region'] for d in decisions)),
                                              'intents': dict(Counter(d['intent'] for d in decisions)),
                                              'paint_loads': len(set(loaded)),
                                          }}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'strokes': len(strokes), 'phase_counts': phase_counts,
                      'brush_counts': dict(brush_counts), 'cycles': len(decisions),
                      'paint_loads': len(set(loaded)), 'artifact_checks': art,
                      'quality': checkpoints[-1]}, ensure_ascii=False))

if __name__ == '__main__':
    main()
