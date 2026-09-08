#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


def clamp(value, low=0, high=255):
    return max(low, min(high, value))


def rgb_tuple(rgb):
    return tuple(int(clamp(v)) for v in rgb)


def hexcolor(rgb):
    return '#%02x%02x%02x' % rgb_tuple(rgb)


def parse_hex(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def crop_resize(image, width, height):
    image = image.convert('RGB')
    source_ratio = image.width / image.height
    target_ratio = width / height
    if source_ratio > target_ratio:
        new_width = int(image.height * target_ratio)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    else:
        new_height = int(image.width / target_ratio)
        top = (image.height - new_height) // 2
        image = image.crop((0, top, image.width, top + new_height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def luminance(pixel):
    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]


def gradient_angle(pixels, x, y, width, height):
    x0, x1 = max(0, x - 1), min(width - 1, x + 1)
    y0, y1 = max(0, y - 1), min(height - 1, y + 1)
    gx = luminance(pixels[x1, y]) - luminance(pixels[x0, y])
    gy = luminance(pixels[x, y1]) - luminance(pixels[x, y0])
    magnitude = math.hypot(gx, gy)
    # 輪郭を横切らず、輪郭の接線方向へブラシを流す。
    angle = math.atan2(gy, gx) + math.pi / 2 if magnitude > 1.5 else 0.0
    return angle, magnitude


def add_line(strokes, phase, x, y, length, width, color, opacity, angle):
    dx = math.cos(angle) * length / 2
    dy = math.sin(angle) * length / 2
    strokes.append({
        'phase': phase,
        'brush': 'line',
        'x1': round(x - dx, 2),
        'y1': round(y - dy, 2),
        'x2': round(x + dx, 2),
        'y2': round(y + dy, 2),
        'width': round(width, 2),
        'color': color,
        'opacity': round(opacity, 3),
        'lineCap': 'round',
    })


def render_preview(document, output):
    """Canvas の source-over + globalAlpha に近い見た目を Pillow で再現する。"""
    width = document['canvas']['width']
    height = document['canvas']['height']
    image = Image.new('RGB', (width, height), document['canvas']['background'])
    draw = ImageDraw.Draw(image, 'RGBA')
    for stroke in document['strokes']:
        if stroke['brush'] != 'line':
            continue
        rgb = parse_hex(stroke['color'])
        alpha = int(clamp(round((stroke.get('opacity', 1.0)) * 255)))
        fill = (*rgb, alpha)
        line_width = max(1, int(round(stroke['width'])))
        xy = (stroke['x1'], stroke['y1'], stroke['x2'], stroke['y2'])
        draw.line(xy, fill=fill, width=line_width)
        # Canvas の round lineCap に近づける。
        if line_width >= 3:
            radius = line_width / 2
            for x, y in ((stroke['x1'], stroke['y1']), (stroke['x2'], stroke['y2'])):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    image.save(output, quality=95)


def add_grid_pass(strokes, pixels, width, height, rng, *, step, length,
                  brush_width, opacity, phase, jitter):
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            xx = int(clamp(x + rng.uniform(-jitter, jitter), 0, width - 1))
            yy = int(clamp(y + rng.uniform(-jitter, jitter), 0, height - 1))
            angle, magnitude = gradient_angle(pixels, xx, yy, width, height)
            red, green, blue = pixels[xx, yy]
            # 大きな筆ではわずかに色を揺らし、細い筆ほど参照色へ忠実にする。
            color_jitter = max(0.35, step / 5.0)
            delta = rng.uniform(-color_jitter, color_jitter)
            color = hexcolor((red + delta, green + delta * 0.65, blue + delta * 0.4))
            local_length = length * (0.78 if magnitude > 26 else 1.0)
            add_line(strokes, phase, xx, yy, local_length, brush_width, color, opacity, angle)


def collect_detail_samples(pixels, width, height, rng, count):
    samples = []
    # 高コントラスト部を優先する。目、唇、鼻、顔輪郭、ターバン境界の密度が自然に上がる。
    attempts = max(count * 4, count)
    for _ in range(attempts):
        x = rng.randrange(2, width - 2)
        y = rng.randrange(2, height - 2)
        angle, magnitude = gradient_angle(pixels, x, y, width, height)
        probability = min(0.96, 0.10 + magnitude / 62)
        if rng.random() < probability:
            samples.append((x, y, angle, magnitude))
        if len(samples) >= count:
            break
    while len(samples) < count:
        x = rng.randrange(2, width - 2)
        y = rng.randrange(2, height - 2)
        angle, magnitude = gradient_angle(pixels, x, y, width, height)
        samples.append((x, y, angle, magnitude))
    return samples


def main():
    parser = argparse.ArgumentParser(description='参照画像を順序付きブラシストロークへ変換する')
    parser.add_argument('input')
    parser.add_argument('--output', default='strokes.generated.json')
    parser.add_argument('--preview', default='preview.png')
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--detail', type=int, default=50000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    image = crop_resize(Image.open(args.input), args.width, args.height)
    pixels = image.load()
    width, height = image.size
    strokes = []

    # 低周波 → 中周波 → 高周波の順で積み上げる。
    # 約12万ストロークを目安にしつつ、最終層ほど参照画像へ忠実にする。
    passes = [
        dict(step=14, length=18.0, brush_width=15.0, opacity=0.97,
             phase='coarse', jitter=1.8),
        dict(step=6, length=8.5, brush_width=6.6, opacity=0.90,
             phase='medium', jitter=0.75),
        dict(step=4, length=5.3, brush_width=4.4, opacity=0.91,
             phase='refine', jitter=0.35),
    ]
    for spec in passes:
        add_grid_pass(strokes, pixels, width, height, rng, **spec)

    for x, y, angle, magnitude in collect_detail_samples(
        pixels, width, height, rng, args.detail
    ):
        red, green, blue = pixels[x, y]
        is_edge = magnitude > 18
        add_line(
            strokes,
            'detail',
            x,
            y,
            3.2 if is_edge else 4.2,
            1.65 if is_edge else 2.15,
            hexcolor((red, green, blue)),
            0.95 if is_edge else 0.78,
            angle,
        )

    phases = [
        {'id': 'coarse', 'label': '下塗り'},
        {'id': 'medium', 'label': '中間描画'},
        {'id': 'refine', 'label': '形状精密化'},
        {'id': 'detail', 'label': '細部'},
    ]
    for index, stroke in enumerate(strokes, 1):
        stroke['id'] = index

    document = {
        'metadata': {
            'slug': 'reference-guided',
            'title': '参照画像ガイド描画',
            'seed': args.seed,
            'source_mode': 'reference-guided',
            'stroke_count': len(strokes),
        },
        'canvas': {
            'width': width,
            'height': height,
            'background': hexcolor(pixels[0, 0]),
        },
        'phases': phases,
        'strokes': strokes,
    }

    Path(args.output).write_text(json.dumps(document, separators=(',', ':')), encoding='utf-8')
    render_preview(document, args.preview)
    print(json.dumps({
        'strokes': len(strokes),
        'output': args.output,
        'preview': args.preview,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
