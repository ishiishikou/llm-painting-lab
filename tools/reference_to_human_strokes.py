#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import defaultdict
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


def color_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def gradient_angle(pixels, x, y, width, height):
    x0, x1 = max(0, x - 1), min(width - 1, x + 1)
    y0, y1 = max(0, y - 1), min(height - 1, y + 1)
    gx = luminance(pixels[x1, y]) - luminance(pixels[x0, y])
    gy = luminance(pixels[x, y1]) - luminance(pixels[x, y0])
    magnitude = math.hypot(gx, gy)
    angle = math.atan2(gy, gx) + math.pi / 2 if magnitude > 1.5 else 0.0
    return angle, magnitude


def add_line(strokes, phase, x, y, length, width, color, opacity, angle=0.0, **extra):
    dx = math.cos(angle) * length / 2
    dy = math.sin(angle) * length / 2
    stroke = {
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
    }
    stroke.update(extra)
    strokes.append(stroke)


def add_ellipse(strokes, phase, x, y, rx, ry, color, opacity, rotation=0.0, **extra):
    stroke = {
        'phase': phase,
        'brush': 'ellipse',
        'x': round(x, 2),
        'y': round(y, 2),
        'rx': round(rx, 2),
        'ry': round(ry, 2),
        'rotation': round(rotation, 4),
        'color': color,
        'opacity': round(opacity, 3),
    }
    stroke.update(extra)
    strokes.append(stroke)


def render_strokes(strokes, width, height, background):
    image = Image.new('RGB', (width, height), background)
    draw = ImageDraw.Draw(image, 'RGBA')
    for stroke in strokes:
        rgb = parse_hex(stroke.get('color', '#ffffff'))
        alpha = int(clamp(round(stroke.get('opacity', 1.0) * 255)))
        fill = (*rgb, alpha)
        brush = stroke.get('brush', 'line')
        if brush == 'line':
            line_width = max(1, int(round(stroke.get('width', 1))))
            xy = (stroke['x1'], stroke['y1'], stroke['x2'], stroke['y2'])
            draw.line(xy, fill=fill, width=line_width)
            if line_width >= 3:
                radius = line_width / 2
                for x, y in ((stroke['x1'], stroke['y1']), (stroke['x2'], stroke['y2'])):
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        elif brush == 'ellipse':
            x, y = stroke['x'], stroke['y']
            rx, ry = stroke.get('rx', 10), stroke.get('ry', 10)
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=fill)
    return image


def image_metrics(reference, rendered):
    diff = ImageChops.difference(reference, rendered)
    stat = ImageStat.Stat(diff)
    mae = sum(stat.mean) / len(stat.mean)
    rms = math.sqrt(sum(value * value for value in stat.rms) / len(stat.rms))
    psnr = 99.0 if rms == 0 else 20 * math.log10(255.0 / rms)
    ref_edges = reference.convert('L').filter(ImageFilter.FIND_EDGES)
    out_edges = rendered.convert('L').filter(ImageFilter.FIND_EDGES)
    edge_mae = ImageStat.Stat(ImageChops.difference(ref_edges, out_edges)).mean[0]
    return {
        'mae': round(mae, 4),
        'rmse': round(rms, 4),
        'psnr': round(psnr, 4),
        'edge_mae': round(edge_mae, 4),
    }


def border_background(image):
    width, height = image.size
    pixels = image.load()
    points = []
    stride = max(4, min(width, height) // 100)
    for x in range(0, width, stride):
        points += [pixels[x, 0], pixels[x, height - 1]]
    for y in range(0, height, stride):
        points += [pixels[0, y], pixels[width - 1, y]]
    channels = [sorted(pixel[i] for pixel in points) for i in range(3)]
    return tuple(channel[len(channel) // 2] for channel in channels)


def infer_subject_bbox(image, background):
    width, height = image.size
    pixels = image.load()
    xs, ys = [], []
    step = 4
    background_luminance = luminance(background)
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            pixel = pixels[x, y]
            if color_distance(pixel, background) > 48 or luminance(pixel) > background_luminance + 26:
                xs.append(x)
                ys.append(y)
    if not xs:
        return (width * 0.2, height * 0.1, width * 0.85, height * 0.95)
    xs.sort()
    ys.sort()
    low = int(len(xs) * 0.02)
    high = int(len(xs) * 0.98) - 1
    return (xs[low], ys[low], xs[high], ys[high])


def skin_like(pixel):
    red, green, blue = pixel
    maximum, minimum = max(pixel), min(pixel)
    return (
        red > 95 and green > 65 and blue > 45 and
        red > green * 1.02 and green > blue * 0.93 and
        (maximum - minimum) > 12
    )


def infer_face_bbox(image, subject_bbox):
    width, height = image.size
    pixels = image.load()
    sx0, sy0, sx1, sy1 = subject_bbox
    xs, ys = [], []
    x0, x1 = max(0, int(sx0)), min(width, int(sx1))
    y0 = max(0, int(sy0))
    y1 = min(height, int(sy0 + (sy1 - sy0) * 0.68))
    for y in range(y0, y1, 3):
        for x in range(x0, x1, 3):
            if skin_like(pixels[x, y]):
                xs.append(x)
                ys.append(y)
    if len(xs) < 100:
        return (
            sx0 + (sx1 - sx0) * 0.18,
            sy0 + (sy1 - sy0) * 0.20,
            sx0 + (sx1 - sx0) * 0.67,
            sy0 + (sy1 - sy0) * 0.64,
        )
    xs.sort()
    ys.sort()

    def quantile(values, fraction):
        return values[min(len(values) - 1, max(0, int(len(values) * fraction)))]

    return (quantile(xs, 0.08), quantile(ys, 0.08), quantile(xs, 0.92), quantile(ys, 0.92))


def region_class(pixel, background):
    red, green, blue = pixel
    if color_distance(pixel, background) < 45 and luminance(pixel) < 75:
        return 'background'
    if blue > red * 1.12 and blue > green * 1.05:
        return 'blue'
    if red > 105 and green > 80 and blue < green * 0.82:
        return 'ochre'
    if skin_like(pixel):
        return 'skin'
    if luminance(pixel) < 72:
        return 'dark'
    return 'neutral'


def nearest_order(points, start=None):
    if not points:
        return []
    remaining = points[:]
    if start is None:
        current = remaining.pop(0)
    else:
        index = min(
            range(len(remaining)),
            key=lambda i: (remaining[i][0] - start[0]) ** 2 + (remaining[i][1] - start[1]) ** 2,
        )
        current = remaining.pop(index)
    result = [current]
    while remaining:
        limit = min(64, len(remaining))
        index = min(
            range(limit),
            key=lambda i: (remaining[i][0] - current[0]) ** 2 + (remaining[i][1] - current[1]) ** 2,
        )
        current = remaining.pop(index)
        result.append(current)
    return result


def tile_order(points, tile=96, face_bbox=None):
    groups = defaultdict(list)
    for point in points:
        groups[(int(point[0] // tile), int(point[1] // tile))].append(point)

    def tile_score(item):
        (tile_x, tile_y), values = item
        score = sum(value[-1] if isinstance(value[-1], (int, float)) else 0 for value in values) / max(1, len(values))
        center_x = (tile_x + 0.5) * tile
        center_y = (tile_y + 0.5) * tile
        if face_bbox:
            fx0, fy0, fx1, fy1 = face_bbox
            if fx0 <= center_x <= fx1 and fy0 <= center_y <= fy1:
                score += 1000
        return score

    ordered = []
    for _, values in sorted(groups.items(), key=tile_score, reverse=True):
        ordered.extend(nearest_order(values))
    return ordered


def phase_composition(strokes, subject_bbox, face_bbox):
    sx0, sy0, sx1, sy1 = subject_bbox
    fx0, fy0, fx1, fy1 = face_bbox
    guide = '#b8aa8f'
    center_x = (sx0 + sx1) / 2
    center_y = (sy0 + sy1) / 2
    add_line(strokes, 'composition', center_x, center_y, (sy1 - sy0) * 0.88, 2.0, guide, 0.34, math.pi / 2, role='subject-axis')
    add_line(strokes, 'composition', center_x, sy1 - (sy1 - sy0) * 0.18, (sx1 - sx0) * 0.82, 2.0, guide, 0.30, 0, role='shoulder-axis')
    face_center_x = (fx0 + fx1) / 2
    face_center_y = (fy0 + fy1) / 2
    add_line(strokes, 'composition', face_center_x, face_center_y, (fy1 - fy0) * 0.95, 1.7, '#d8b49a', 0.34, math.pi / 2, role='face-centerline')
    add_line(strokes, 'composition', face_center_x, fy0 + (fy1 - fy0) * 0.42, (fx1 - fx0) * 0.96, 1.5, '#d8b49a', 0.30, 0, role='eye-line')
    add_line(strokes, 'composition', face_center_x, fy0 + (fy1 - fy0) * 0.72, (fx1 - fx0) * 0.70, 1.4, '#d8b49a', 0.26, 0, role='mouth-line')
    points = [
        (sx0 + (sx1 - sx0) * 0.15, sy0 + (sy1 - sy0) * 0.18),
        (sx0 + (sx1 - sx0) * 0.45, sy0),
        (sx0 + (sx1 - sx0) * 0.78, sy0 + (sy1 - sy0) * 0.15),
        (sx1, sy0 + (sy1 - sy0) * 0.55),
        (sx0 + (sx1 - sx0) * 0.88, sy1),
        (sx0 + (sx1 - sx0) * 0.28, sy1),
        (sx0, sy0 + (sy1 - sy0) * 0.63),
    ]
    for first, second in zip(points, points[1:] + points[:1]):
        mid_x = (first[0] + second[0]) / 2
        mid_y = (first[1] + second[1]) / 2
        add_line(
            strokes, 'composition', mid_x, mid_y,
            math.hypot(second[0] - first[0], second[1] - first[1]),
            1.8, guide, 0.28,
            math.atan2(second[1] - first[1], second[0] - first[0]),
            role='envelope',
        )


def phase_silhouette(strokes, image, background, subject_bbox, rng):
    blurred = image.filter(ImageFilter.GaussianBlur(16))
    pixels = blurred.load()
    width, height = image.size
    sx0, sy0, sx1, sy1 = subject_bbox
    points = defaultdict(list)
    step = 24
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            pixel = pixels[x, y]
            category = region_class(pixel, background)
            if category == 'background':
                continue
            if not (sx0 - step <= x <= sx1 + step and sy0 - step <= y <= sy1 + step) and color_distance(pixel, background) < 70:
                continue
            points[category].append((x, y, pixel))
    last = None
    for category in ['dark', 'neutral', 'ochre', 'blue', 'skin']:
        bucket = points.get(category, [])
        rng.shuffle(bucket)
        ordered = nearest_order(bucket, start=last)
        for x, y, pixel in ordered:
            add_ellipse(strokes, 'silhouette', x, y, 15.5, 13.5, hexcolor(pixel), 0.90, role=category)
            last = (x, y)


def add_brush_pass(strokes, image, phase, background, rng, *, step, width, length, blur_radius):
    source = image.filter(ImageFilter.GaussianBlur(blur_radius)) if blur_radius else image
    pixels = source.load()
    raw_pixels = image.load()
    canvas_width, canvas_height = image.size
    points = defaultdict(list)
    for y in range(step // 2, canvas_height, step):
        for x in range(step // 2, canvas_width, step):
            pixel = pixels[x, y]
            category = region_class(pixel, background)
            angle, magnitude = gradient_angle(raw_pixels, x, y, canvas_width, canvas_height)
            if category == 'background' and ((x // step + y // step) % 4):
                continue
            points[category].append((x, y, pixel, angle, magnitude))
    last = None
    for category in ['background', 'dark', 'neutral', 'ochre', 'blue', 'skin']:
        bucket = points.get(category, [])
        rng.shuffle(bucket)
        ordered = nearest_order(bucket, start=last)
        for x, y, pixel, angle, magnitude in ordered:
            local_length = length * (0.72 if magnitude > 25 else 1.0)
            local_width = width * (0.82 if magnitude > 30 else 1.0)
            add_line(
                strokes, phase, x, y, local_length, local_width,
                hexcolor(pixel), 0.86 if step > 7 else 0.90, angle,
                role=category,
            )
            last = (x, y)


def phase_face_structure(strokes, image, face_bbox, rng):
    pixels = image.load()
    width, height = image.size
    fx0, fy0, fx1, fy1 = map(int, face_bbox)
    fx0, fy0 = max(2, fx0), max(2, fy0)
    fx1, fy1 = min(width - 2, fx1), min(height - 2, fy1)
    center_x = (fx0 + fx1) / 2
    tone = hexcolor(pixels[int(center_x), int((fy0 + fy1) / 2)])
    add_line(strokes, 'face_structure', center_x, (fy0 + fy1) / 2, (fy1 - fy0) * 0.88, 2.2, tone, 0.42, math.pi / 2, role='centerline')
    add_line(strokes, 'face_structure', center_x, fy0 + (fy1 - fy0) * 0.42, (fx1 - fx0) * 0.90, 2.0, tone, 0.38, 0, role='eye-axis')
    add_line(strokes, 'face_structure', center_x, fy0 + (fy1 - fy0) * 0.72, (fx1 - fx0) * 0.62, 1.8, tone, 0.34, 0, role='mouth-axis')
    points = []
    step = 5
    for y in range(fy0, fy1, step):
        for x in range(fx0, fx1, step):
            angle, magnitude = gradient_angle(pixels, x, y, width, height)
            points.append((x, y, pixels[x, y], angle, magnitude))
    rng.shuffle(points)
    for x, y, pixel, angle, magnitude in tile_order(points, tile=54, face_bbox=face_bbox):
        add_line(
            strokes, 'face_structure', x, y,
            6.4 if magnitude < 18 else 4.5,
            3.0 if magnitude < 18 else 2.2,
            hexcolor(pixel), 0.91, angle, role='face-plane',
        )


def collect_detail_points(image, count, rng, face_bbox):
    pixels = image.load()
    width, height = image.size
    candidates = []
    for y in range(2, height - 2, 3):
        for x in range(2, width - 2, 3):
            angle, magnitude = gradient_angle(pixels, x, y, width, height)
            if magnitude < 5:
                continue
            fx0, fy0, fx1, fy1 = face_bbox
            bonus = 38 if fx0 <= x <= fx1 and fy0 <= y <= fy1 else 0
            score = magnitude + bonus + rng.random() * 4
            candidates.append((x, y, pixels[x, y], angle, score))
    candidates.sort(key=lambda point: point[-1], reverse=True)
    return tile_order(candidates[:count], tile=72, face_bbox=face_bbox)


def phase_detail(strokes, image, count, rng, face_bbox):
    for x, y, pixel, angle, _ in collect_detail_points(image, count, rng, face_bbox):
        add_line(strokes, 'detail', x, y, 3.4, 1.7, hexcolor(pixel), 0.94, angle, role='detail')


def collect_residual(reference, rendered, count, face_bbox):
    reference_pixels = reference.load()
    output_pixels = rendered.load()
    width, height = reference.size
    points = []
    for y in range(2, height - 2, 3):
        for x in range(2, width - 2, 3):
            ref = reference_pixels[x, y]
            out = output_pixels[x, y]
            error = math.sqrt(sum((ref[i] - out[i]) ** 2 for i in range(3)) / 3)
            angle, magnitude = gradient_angle(reference_pixels, x, y, width, height)
            fx0, fy0, fx1, fy1 = face_bbox
            if fx0 <= x <= fx1 and fy0 <= y <= fy1:
                error *= 1.15
            score = error * (1 + min(magnitude, 80) / 260)
            if score > 4:
                points.append((x, y, ref, angle, score))
    points.sort(key=lambda point: point[-1], reverse=True)
    return tile_order(points[:count], tile=72, face_bbox=face_bbox)


def phase_finish(strokes, image, background, count, face_bbox):
    rendered = render_strokes(strokes, *image.size, background)
    for x, y, pixel, angle, _ in collect_residual(image, rendered, count, face_bbox):
        add_line(strokes, 'finish', x, y, 2.3, 1.2, hexcolor(pixel), 0.97, angle, role='correction')
    return render_strokes(strokes, *image.size, background)


def main():
    parser = argparse.ArgumentParser(description='参照画像から、人が絵を描く6工程の順でストロークを生成する')
    parser.add_argument('input')
    parser.add_argument('--output', default='strokes.generated.json')
    parser.add_argument('--preview', default='preview.png')
    parser.add_argument('--metrics', default='metrics.json')
    parser.add_argument('--width', type=int, default=864)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--detail', type=int, default=45000)
    parser.add_argument('--finish', type=int, default=30000)
    parser.add_argument('--checkpoint-dir', default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    image = crop_resize(Image.open(args.input), args.width, args.height)
    background_rgb = border_background(image)
    subject_bbox = infer_subject_bbox(image, background_rgb)
    face_bbox = infer_face_bbox(image, subject_bbox)
    background = hexcolor(background_rgb)
    strokes = []
    checkpoints = []
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(name):
        if checkpoint_dir:
            render_strokes(strokes, args.width, args.height, background).save(checkpoint_dir / f'{name}.png')

    phase_composition(strokes, subject_bbox, face_bbox)
    checkpoints.append({'stage': 'composition', 'stroke_count': len(strokes)})
    save_checkpoint('01-composition')

    phase_silhouette(strokes, image, background_rgb, subject_bbox, rng)
    checkpoints.append({'stage': 'silhouette', 'stroke_count': len(strokes)})
    save_checkpoint('02-silhouette')

    add_brush_pass(strokes, image, 'light_shadow', background_rgb, rng, step=12, width=10.5, length=16.0, blur_radius=7)
    add_brush_pass(strokes, image, 'light_shadow', background_rgb, rng, step=6, width=5.2, length=8.0, blur_radius=2.5)
    checkpoints.append({'stage': 'light_shadow', 'stroke_count': len(strokes)})
    save_checkpoint('03-light-shadow')

    phase_face_structure(strokes, image, face_bbox, rng)
    checkpoints.append({'stage': 'face_structure', 'stroke_count': len(strokes)})
    save_checkpoint('04-face-structure')

    phase_detail(strokes, image, args.detail, rng, face_bbox)
    checkpoints.append({'stage': 'detail', 'stroke_count': len(strokes)})
    save_checkpoint('05-detail')

    rendered = phase_finish(strokes, image, background, args.finish, face_bbox)
    checkpoints.append({'stage': 'finish', 'stroke_count': len(strokes), **image_metrics(image, rendered)})
    if checkpoint_dir:
        rendered.save(checkpoint_dir / '06-finish.png')

    phases = [
        {'id': 'composition', 'label': '1. 構図'},
        {'id': 'silhouette', 'label': '2. シルエット'},
        {'id': 'light_shadow', 'label': '3. 明暗の面'},
        {'id': 'face_structure', 'label': '4. 顔構造'},
        {'id': 'detail', 'label': '5. 細部'},
        {'id': 'finish', 'label': '6. 仕上げ'},
    ]
    for index, stroke in enumerate(strokes, 1):
        stroke['id'] = index

    document = {
        'metadata': {
            'slug': 'human-painting-order',
            'title': '人間の描画順を模した参照ガイド描画',
            'seed': args.seed,
            'source_mode': 'reference-guided-human-order',
            'stroke_count': len(strokes),
            'subject_bbox': [round(value, 2) for value in subject_bbox],
            'face_bbox': [round(value, 2) for value in face_bbox],
            'quality_metrics': checkpoints[-1],
            'phase_order': [phase['id'] for phase in phases],
        },
        'canvas': {
            'width': args.width,
            'height': args.height,
            'background': background,
        },
        'phases': phases,
        'strokes': strokes,
    }
    Path(args.output).write_text(json.dumps(document, separators=(',', ':')), encoding='utf-8')
    rendered.save(args.preview)
    Path(args.metrics).write_text(
        json.dumps({'checkpoints': checkpoints}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps({
        'strokes': len(strokes),
        'subject_bbox': subject_bbox,
        'face_bbox': face_bbox,
        'quality': checkpoints[-1],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
