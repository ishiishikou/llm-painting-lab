#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


def clamp(value, low=0, high=255):
    return max(low, min(high, value))


def hexcolor(rgb):
    return '#%02x%02x%02x' % tuple(int(clamp(v)) for v in rgb)


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
    angle = math.atan2(gy, gx) + math.pi / 2 if magnitude > 2 else 0.0
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
    width = document['canvas']['width']
    height = document['canvas']['height']
    image = Image.new('RGB', (width, height), document['canvas']['background'])
    draw = ImageDraw.Draw(image)
    for stroke in document['strokes']:
        if stroke['brush'] != 'line':
            continue
        draw.line(
            (stroke['x1'], stroke['y1'], stroke['x2'], stroke['y2']),
            fill=stroke['color'],
            width=max(1, int(round(stroke['width']))),
        )
    image.save(output, quality=95)


def main():
    parser = argparse.ArgumentParser(description='参照画像を順序付きブラシストロークへ変換する')
    parser.add_argument('input')
    parser.add_argument('--output', default='strokes.generated.json')
    parser.add_argument('--preview', default='preview.png')
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--detail', type=int, default=18000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    image = crop_resize(Image.open(args.input), args.width, args.height)
    pixels = image.load()
    width, height = image.size
    strokes = []

    # 粗い下塗りと中間描画。輪郭の接線方向にストロークを向ける。
    for step, length, brush_width, opacity, phase, jitter in [
        (14, 18, 15, 0.96, 'coarse', 2.0),
        (7, 10, 7.5, 0.84, 'medium', 1.3),
    ]:
        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                xx = int(clamp(x + rng.uniform(-jitter, jitter), 0, width - 1))
                yy = int(clamp(y + rng.uniform(-jitter, jitter), 0, height - 1))
                angle, _ = gradient_angle(pixels, xx, yy, width, height)
                red, green, blue = pixels[xx, yy]
                delta = rng.uniform(-2.5, 2.5)
                color = hexcolor((red + delta, green + delta * 0.7, blue + delta * 0.4))
                add_line(strokes, phase, xx, yy, length, brush_width, color, opacity, angle)

    # 細部は勾配の大きい領域を優先し、目・口・輪郭などへストローク密度を寄せる。
    detail_samples = []
    for _ in range(args.detail * 3):
        x = rng.randrange(2, width - 2)
        y = rng.randrange(2, height - 2)
        angle, magnitude = gradient_angle(pixels, x, y, width, height)
        if rng.random() < min(0.9, 0.16 + magnitude / 85):
            detail_samples.append((x, y, angle, magnitude))
        if len(detail_samples) >= args.detail:
            break

    while len(detail_samples) < args.detail:
        x = rng.randrange(2, width - 2)
        y = rng.randrange(2, height - 2)
        angle, magnitude = gradient_angle(pixels, x, y, width, height)
        detail_samples.append((x, y, angle, magnitude))

    for x, y, angle, magnitude in detail_samples:
        red, green, blue = pixels[x, y]
        add_line(
            strokes,
            'detail',
            x,
            y,
            4.2 if magnitude > 20 else 5.5,
            2.2 if magnitude > 20 else 2.8,
            hexcolor((red, green, blue)),
            0.88 if magnitude > 20 else 0.72,
            angle,
        )

    phases = [
        {'id': 'coarse', 'label': '下塗り'},
        {'id': 'medium', 'label': '中間描画'},
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
