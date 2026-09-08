#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, v))


def parse_hex(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def midpoint(stroke):
    if all(key in stroke for key in ('x1', 'y1', 'x2', 'y2')):
        return ((stroke['x1'] + stroke['x2']) / 2, (stroke['y1'] + stroke['y2']) / 2)
    return (stroke.get('x', 0), stroke.get('y', 0))


def classify_region(stroke, width, height):
    x, y = midpoint(stroke)
    r, g, b = parse_hex(stroke.get('color', '#000000'))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    nx = x / max(1, width)
    ny = y / max(1, height)

    warm = r > g * 1.02 and g > b * 1.04 and r > 55
    blue = b > r * 1.08 and b > g * 1.02 and b > 55
    yellow = r > 95 and g > 70 and r > b * 1.45 and g > b * 1.25
    pale = lum > 150 and abs(r - g) < 55 and abs(g - b) < 70

    if 0.46 <= nx <= 0.72 and 0.42 <= ny <= 0.72 and pale:
        return 'accent'
    if 0.30 <= nx <= 0.73 and 0.25 <= ny <= 0.66 and warm:
        return 'face'
    if 0.20 <= nx <= 0.88 and ny <= 0.48 and (blue or yellow):
        return 'headwrap'
    if ny >= 0.58 and 0.16 <= nx <= 0.92:
        return 'garment'
    if lum < 50:
        return 'background'
    return 'other'


REGION_ANCHORS = {
    'background': (0.12, 0.18),
    'garment': (0.48, 0.80),
    'headwrap': (0.50, 0.20),
    'face': (0.52, 0.46),
    'accent': (0.61, 0.60),
    'other': (0.50, 0.50),
}

PHASE_REGION_ORDER = {
    'coarse': ['background', 'garment', 'headwrap', 'face', 'other', 'accent'],
    'medium': ['garment', 'headwrap', 'face', 'other', 'background', 'accent'],
    'refine': ['face', 'headwrap', 'garment', 'accent', 'other', 'background'],
    'detail': ['face', 'accent', 'headwrap', 'garment', 'other', 'background'],
    'repair1': ['face', 'accent', 'headwrap', 'garment', 'other', 'background'],
    'repair2': ['face', 'accent', 'headwrap', 'garment', 'other', 'background'],
}

PHASE_REMAP = {
    'coarse': ('block_in', '大きな形を置く'),
    'medium': ('values', '明暗の面を作る'),
    'refine': ('structure', '形と顔の構造を整える'),
    'detail': ('features', '目・鼻・口・布・真珠を描き込む'),
    'repair1': ('finish1', '重要部を見直す'),
    'repair2': ('finish2', '最終仕上げ'),
}


def tile_key(stroke, tile_size):
    x, y = midpoint(stroke)
    return (int(x // tile_size), int(y // tile_size))


def tile_center(tile, tile_size):
    tx, ty = tile
    return ((tx + 0.5) * tile_size, (ty + 0.5) * tile_size)


def nearest_tile_order(tiles, start_xy, tile_size):
    remaining = set(tiles)
    ordered = []
    cx, cy = start_xy
    while remaining:
        best = min(
            remaining,
            key=lambda tile: (tile_center(tile, tile_size)[0] - cx) ** 2 +
                             (tile_center(tile, tile_size)[1] - cy) ** 2,
        )
        ordered.append(best)
        remaining.remove(best)
        cx, cy = tile_center(best, tile_size)
    return ordered


def localized_order(strokes, width, height, region, phase, rng, tile_size=96):
    by_tile = defaultdict(list)
    for stroke in strokes:
        by_tile[tile_key(stroke, tile_size)].append(stroke)

    ax, ay = REGION_ANCHORS.get(region, (0.5, 0.5))
    tile_order = nearest_tile_order(by_tile.keys(), (ax * width, ay * height), tile_size)

    result = []
    for tile in tile_order:
        bucket = by_tile[tile]
        rng.shuffle(bucket)
        # 太いストロークから先に置き、同じ局所領域の中でも「大きな筆→小さな筆」にする。
        bucket.sort(key=lambda s: -(s.get('width', 1) * max(1.0, math.hypot(
            s.get('x2', s.get('x', 0)) - s.get('x1', s.get('x', 0)),
            s.get('y2', s.get('y', 0)) - s.get('y1', s.get('y', 0)),
        ))))
        result.extend(bucket)
    return result


def main():
    parser = argparse.ArgumentParser(description='ストローク文書を人間らしい局所描画順へ並べ替える')
    parser.add_argument('input')
    parser.add_argument('--output', default='strokes.painterly.json')
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--tile-size', type=int, default=96)
    args = parser.parse_args()

    document = json.loads(Path(args.input).read_text(encoding='utf-8'))
    width = document['canvas']['width']
    height = document['canvas']['height']
    rng = random.Random(args.seed)

    original_phases = []
    by_phase = defaultdict(list)
    for stroke in document.get('strokes', []):
        phase = stroke.get('phase', 'detail')
        if phase not in original_phases:
            original_phases.append(phase)
        by_phase[phase].append(stroke)

    ordered_strokes = []
    new_phases = []

    for phase in original_phases:
        source = by_phase[phase]
        phase_id, phase_label = PHASE_REMAP.get(phase, (phase, phase))
        new_phases.append({'id': phase_id, 'label': phase_label})

        by_region = defaultdict(list)
        for stroke in source:
            by_region[classify_region(stroke, width, height)].append(stroke)

        region_order = PHASE_REGION_ORDER.get(
            phase,
            ['face', 'headwrap', 'garment', 'accent', 'other', 'background'],
        )
        for region in region_order:
            bucket = by_region.get(region, [])
            if not bucket:
                continue
            localized = localized_order(
                bucket, width, height, region, phase, rng, tile_size=args.tile_size
            )
            for stroke in localized:
                copied = dict(stroke)
                copied['phase'] = phase_id
                copied['region'] = region
                ordered_strokes.append(copied)

    for index, stroke in enumerate(ordered_strokes, 1):
        stroke['id'] = index

    output = dict(document)
    output['phases'] = new_phases
    output['strokes'] = ordered_strokes
    output['metadata'] = dict(document.get('metadata', {}))
    output['metadata'].update({
        'slug': 'reference-guided-painterly',
        'title': '参照画像ガイド描画（人間らしい描画順）',
        'ordering': 'painterly-localized',
        'stroke_count': len(ordered_strokes),
    })

    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, separators=(',', ':')),
        encoding='utf-8',
    )
    print(json.dumps({
        'input_strokes': len(document.get('strokes', [])),
        'output_strokes': len(ordered_strokes),
        'output': args.output,
        'tile_size': args.tile_size,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
