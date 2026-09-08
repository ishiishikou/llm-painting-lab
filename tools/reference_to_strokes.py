#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat


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


def render_strokes(strokes, width, height, background):
    """Canvas の source-over + globalAlpha に近い見た目を Pillow で再現する。"""
    image = Image.new('RGB', (width, height), background)
    draw = ImageDraw.Draw(image, 'RGBA')
    for stroke in strokes:
        if stroke.get('brush') != 'line':
            continue
        rgb = parse_hex(stroke['color'])
        alpha = int(clamp(round(stroke.get('opacity', 1.0) * 255)))
        fill = (*rgb, alpha)
        line_width = max(1, int(round(stroke['width'])))
        xy = (stroke['x1'], stroke['y1'], stroke['x2'], stroke['y2'])
        draw.line(xy, fill=fill, width=line_width)
        # Canvas の round lineCap に近づける。
        if line_width >= 3:
            radius = line_width / 2
            for x, y in ((stroke['x1'], stroke['y1']), (stroke['x2'], stroke['y2'])):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return image


def image_metrics(reference, rendered):
    diff = ImageChops.difference(reference, rendered)
    stat = ImageStat.Stat(diff)
    mae = sum(stat.mean) / len(stat.mean)
    rms = math.sqrt(sum(value * value for value in stat.rms) / len(stat.rms))
    psnr = 99.0 if rms == 0 else 20 * math.log10(255.0 / rms)

    ref_edges = reference.convert('L').filter(ImageFilter.FIND_EDGES)
    out_edges = rendered.convert('L').filter(ImageFilter.FIND_EDGES)
    edge_diff = ImageChops.difference(ref_edges, out_edges)
    edge_mae = ImageStat.Stat(edge_diff).mean[0]

    return {
        'mae': round(mae, 4),
        'rmse': round(rms, 4),
        'psnr': round(psnr, 4),
        'edge_mae': round(edge_mae, 4),
    }


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


def collect_residual_samples(reference, rendered, count, *, stride=2, offset=0):
    """現在の描画と参照画像の残差が大きい場所を優先して返す。"""
    ref_pixels = reference.load()
    out_pixels = rendered.load()
    width, height = reference.size
    candidates = []
    start = offset % stride

    for y in range(max(2, start), height - 2, stride):
        for x in range(max(2, start), width - 2, stride):
            ref = ref_pixels[x, y]
            out = out_pixels[x, y]
            dr = ref[0] - out[0]
            dg = ref[1] - out[1]
            db = ref[2] - out[2]
            color_error = math.sqrt((dr * dr + dg * dg + db * db) / 3)
            angle, magnitude = gradient_angle(ref_pixels, x, y, width, height)
            # 形状に効く輪郭部を少し優先するが、色の残差を主目的にする。
            score = color_error * (1.0 + min(magnitude, 80) / 220.0)
            if score > 2.0:
                candidates.append((score, x, y, angle, magnitude))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:count]


def add_residual_repair(strokes, reference, rendered, count, *, phase, stride, offset):
    ref_pixels = reference.load()
    width, height = reference.size
    samples = collect_residual_samples(
        reference, rendered, count, stride=stride, offset=offset
    )
    for _, x, y, angle, magnitude in samples:
        red, green, blue = ref_pixels[x, y]
        is_edge = magnitude > 18
        add_line(
            strokes,
            phase,
            x,
            y,
            1.9 if is_edge else 2.7,
            1.05 if is_edge else 1.45,
            hexcolor((red, green, blue)),
            0.99 if is_edge else 0.94,
            angle,
        )
    return len(samples)


def main():
    parser = argparse.ArgumentParser(description='参照画像を順序付きブラシストロークへ変換する')
    parser.add_argument('input')
    parser.add_argument('--output', default='strokes.generated.json')
    parser.add_argument('--preview', default='preview.png')
    parser.add_argument('--metrics', default='metrics.json')
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--detail', type=int, default=28000)
    parser.add_argument('--repair', type=int, default=20000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    image = crop_resize(Image.open(args.input), args.width, args.height)
    pixels = image.load()
    width, height = image.size
    background = hexcolor(pixels[0, 0])
    strokes = []
    metrics_history = []

    # 低周波 → 中周波 → 高周波の順で積み上げる。
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

    # ここからが実際のフィードバックループ。
    # 一度描いた結果を評価し、誤差の大きい場所だけを追加ストロークで修正する。
    rendered = render_strokes(strokes, width, height, background)
    metrics_history.append({
        'stage': 'base',
        'stroke_count': len(strokes),
        **image_metrics(image, rendered),
    })

    repair_first = int(args.repair * 0.6)
    repair_second = max(0, args.repair - repair_first)

    added = add_residual_repair(
        strokes, image, rendered, repair_first,
        phase='repair1', stride=2, offset=0
    )
    rendered = render_strokes(strokes, width, height, background)
    metrics_history.append({
        'stage': 'repair1',
        'added_strokes': added,
        'stroke_count': len(strokes),
        **image_metrics(image, rendered),
    })

    added = add_residual_repair(
        strokes, image, rendered, repair_second,
        phase='repair2', stride=2, offset=1
    )
    rendered = render_strokes(strokes, width, height, background)
    metrics_history.append({
        'stage': 'repair2',
        'added_strokes': added,
        'stroke_count': len(strokes),
        **image_metrics(image, rendered),
    })

    phases = [
        {'id': 'coarse', 'label': '下塗り'},
        {'id': 'medium', 'label': '中間描画'},
        {'id': 'refine', 'label': '形状精密化'},
        {'id': 'detail', 'label': '細部'},
        {'id': 'repair1', 'label': '残差修正 1'},
        {'id': 'repair2', 'label': '残差修正 2'},
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
            'quality_metrics': metrics_history[-1],
        },
        'canvas': {
            'width': width,
            'height': height,
            'background': background,
        },
        'phases': phases,
        'strokes': strokes,
    }

    Path(args.output).write_text(json.dumps(document, separators=(',', ':')), encoding='utf-8')
    rendered.save(args.preview, quality=95)
    Path(args.metrics).write_text(
        json.dumps({'history': metrics_history}, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(json.dumps({
        'strokes': len(strokes),
        'output': args.output,
        'preview': args.preview,
        'metrics': args.metrics,
        'quality': metrics_history[-1],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
