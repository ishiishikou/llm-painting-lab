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


def add_line(strokes, phase, x, y, length, width, color, opacity,
             angle=0.0, line_cap='round', **extra):
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
        'lineCap': line_cap,
    }
    stroke.update(extra)
    strokes.append(stroke)


def render_strokes(strokes, width, height, background):
    image = Image.new('RGB', (width, height), background)
    draw = ImageDraw.Draw(image, 'RGBA')
    for stroke in strokes:
        if stroke.get('brush') != 'line':
            continue
        rgb = parse_hex(stroke.get('color', '#ffffff'))
        alpha = int(clamp(round(stroke.get('opacity', 1.0) * 255)))
        fill = (*rgb, alpha)
        line_width = max(1, int(round(stroke.get('width', 1))))
        xy = (stroke['x1'], stroke['y1'], stroke['x2'], stroke['y2'])
        draw.line(xy, fill=fill, width=line_width)
        if stroke.get('lineCap', 'round') == 'round' and line_width >= 3:
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
    background_luminance = luminance(background)
    for y in range(2, height, 4):
        for x in range(2, width, 4):
            pixel = pixels[x, y]
            if color_distance(pixel, background) > 45 or luminance(pixel) > background_luminance + 24:
                xs.append(x)
                ys.append(y)
    if not xs:
        return (width * .18, height * .08, width * .88, height * .96)
    xs.sort()
    ys.sort()
    q0, q1 = int(len(xs) * .015), int(len(xs) * .985) - 1
    return (xs[q0], ys[q0], xs[q1], ys[q1])


def skin_like(pixel):
    r, g, b = pixel
    return (
        r > 88 and g > 58 and b > 38 and
        r > g * 1.01 and g > b * .90 and
        max(pixel) - min(pixel) > 10
    )


def infer_face_bbox(image, subject_bbox):
    width, height = image.size
    pixels = image.load()
    sx0, sy0, sx1, sy1 = subject_bbox
    xs, ys = [], []
    y1 = min(height, int(sy0 + (sy1 - sy0) * .67))
    for y in range(max(0, int(sy0)), y1, 3):
        for x in range(max(0, int(sx0)), min(width, int(sx1)), 3):
            if skin_like(pixels[x, y]):
                xs.append(x)
                ys.append(y)
    if len(xs) < 100:
        return (
            sx0 + (sx1 - sx0) * .18,
            sy0 + (sy1 - sy0) * .22,
            sx0 + (sx1 - sx0) * .67,
            sy0 + (sy1 - sy0) * .62,
        )
    xs.sort(); ys.sort()
    def q(values, f):
        return values[min(len(values)-1, max(0, int(len(values)*f)))]
    return (q(xs,.08), q(ys,.08), q(xs,.92), q(ys,.92))


def inside(box, x, y, pad=0):
    x0, y0, x1, y1 = box
    return x0-pad <= x <= x1+pad and y0-pad <= y <= y1+pad


def subject_pixel(pixel, background):
    return color_distance(pixel, background) > 42 or luminance(pixel) > luminance(background) + 22


def semantic_region(x, y, pixel, background, subject_bbox, face_bbox):
    sx0, sy0, sx1, sy1 = subject_bbox
    fx0, fy0, fx1, fy1 = face_bbox
    sh = sy1-sy0
    if not subject_pixel(pixel, background):
        return 'background'
    if inside(face_bbox, x, y, pad=max(10, (fx1-fx0)*.08)):
        return 'face'
    if y < sy0 + sh * .48:
        return 'headwrap'
    if y > fy1 - (fy1-fy0)*.05:
        return 'garment'
    return 'subject'


def ordered_local(points, start=None, lookahead=72):
    if not points:
        return []
    remaining = points[:]
    if start is None:
        current = remaining.pop(0)
    else:
        idx = min(range(len(remaining)), key=lambda i:
                  (remaining[i][0]-start[0])**2 + (remaining[i][1]-start[1])**2)
        current = remaining.pop(idx)
    result = [current]
    while remaining:
        limit = min(lookahead, len(remaining))
        idx = min(range(limit), key=lambda i:
                  (remaining[i][0]-current[0])**2 + (remaining[i][1]-current[1])**2)
        current = remaining.pop(idx)
        result.append(current)
    return result


def phase_composition(strokes, subject_bbox, face_bbox):
    sx0, sy0, sx1, sy1 = subject_bbox
    fx0, fy0, fx1, fy1 = face_bbox
    guide = '#a88f72'
    fc_x = (fx0 + fx1) / 2
    fc_y = (fy0 + fy1) / 2
    fw, fh = fx1-fx0, fy1-fy0
    sw, sh = sx1-sx0, sy1-sy0
    add_line(strokes, 'composition', fc_x, fc_y, fh*.82, 1.7, guide, .30, math.pi/2,
             role='face-center')
    add_line(strokes, 'composition', fc_x, fy0+fh*.42, fw*.88, 1.5, guide, .28, 0,
             role='eye-line')
    add_line(strokes, 'composition', fc_x, fy0+fh*.72, fw*.58, 1.3, guide, .24, 0,
             role='mouth-line')
    add_line(strokes, 'composition',
             sx0+sw*.52, sy0+sh*.78, sw*.72, 2.0, guide, .34, -0.06,
             role='shoulder-gesture')
    marks = [
        (sx0+sw*.34, sy0+sh*.10, sw*.28, .55),
        (sx0+sw*.57, sy0+sh*.11, sw*.30, -.15),
        (sx0+sw*.72, sy0+sh*.28, sh*.25, 1.25),
        (sx0+sw*.23, sy0+sh*.31, sh*.23, 1.88),
        (sx0+sw*.67, sy0+sh*.60, sh*.25, 1.43),
        (sx0+sw*.43, sy0+sh*.73, sw*.35, .02),
    ]
    for x, y, length, angle in marks:
        add_line(strokes, 'composition', x, y, length, 1.8, guide, .22, angle,
                 role='contour-note')


def broad_subject_pass(strokes, image, phase, background, subject_bbox, face_bbox, rng,
                       *, step, length, brush_width, blur_radius, opacity,
                       region_order=('garment','headwrap','face','subject')):
    source = image.filter(ImageFilter.GaussianBlur(blur_radius)) if blur_radius else image
    pixels = source.load()
    raw = image.load()
    width, height = image.size
    groups = defaultdict(list)
    for y in range(step//2, height, step):
        for x in range(step//2, width, step):
            pixel = pixels[x, y]
            region = semantic_region(x, y, pixel, background, subject_bbox, face_bbox)
            if region == 'background':
                continue
            angle, magnitude = gradient_angle(raw, x, y, width, height)
            if magnitude < 4:
                angle = {
                    'headwrap': -0.12,
                    'face': math.pi/2 * .72,
                    'garment': math.pi/2 * .86,
                    'subject': math.pi/2 * .7,
                }.get(region, 0)
            groups[region].append((x, y, pixel, angle, magnitude))
    last = None
    for region in region_order:
        bucket = groups.get(region, [])
        rng.shuffle(bucket)
        ordered = ordered_local(bucket, start=last)
        for x, y, pixel, angle, magnitude in ordered:
            local_len = length * (.76 if magnitude > 24 else 1.0)
            local_w = brush_width * (.78 if magnitude > 28 else 1.0)
            add_line(strokes, phase, x, y, local_len, local_w, hexcolor(pixel), opacity,
                     angle, line_cap='butt', role=region)
            last = (x, y)


def phase_silhouette(strokes, image, background, subject_bbox, face_bbox, rng):
    broad_subject_pass(
        strokes, image, 'silhouette', background, subject_bbox, face_bbox, rng,
        step=34, length=64, brush_width=31, blur_radius=18, opacity=.88,
        region_order=('garment','headwrap','face','subject')
    )
    broad_subject_pass(
        strokes, image, 'silhouette', background, subject_bbox, face_bbox, rng,
        step=26, length=48, brush_width=23, blur_radius=12, opacity=.76,
        region_order=('headwrap','face','garment','subject')
    )


def phase_light_shadow(strokes, image, background, subject_bbox, face_bbox, rng):
    broad_subject_pass(
        strokes, image, 'light_shadow', background, subject_bbox, face_bbox, rng,
        step=18, length=38, brush_width=16, blur_radius=8, opacity=.84
    )
    broad_subject_pass(
        strokes, image, 'light_shadow', background, subject_bbox, face_bbox, rng,
        step=11, length=24, brush_width=10, blur_radius=4, opacity=.87,
        region_order=('face','headwrap','garment','subject')
    )


def phase_face_structure(strokes, image, face_bbox, rng):
    pixels = image.load()
    width, height = image.size
    fx0, fy0, fx1, fy1 = map(int, face_bbox)
    fx0, fy0 = max(2, fx0), max(2, fy0)
    fx1, fy1 = min(width-2, fx1), min(height-2, fy1)
    fw, fh = fx1-fx0, fy1-fy0
    cx = (fx0+fx1)/2
    tone = hexcolor(pixels[int(cx), int((fy0+fy1)/2)])
    add_line(strokes, 'face_structure', cx, (fy0+fy1)/2, fh*.74, 1.5, tone, .34, math.pi/2,
             role='face-center')
    add_line(strokes, 'face_structure', cx, fy0+fh*.42, fw*.78, 1.4, tone, .30, 0,
             role='eye-line')
    add_line(strokes, 'face_structure', cx, fy0+fh*.70, fw*.50, 1.3, tone, .28, 0,
             role='mouth-line')
    points = []
    for y in range(fy0, fy1, 7):
        for x in range(fx0, fx1, 7):
            angle, magnitude = gradient_angle(pixels, x, y, width, height)
            points.append((x, y, pixels[x,y], angle, magnitude))
    rng.shuffle(points)
    groups = defaultdict(list)
    tile = max(42, int(fw*.23))
    for p in points:
        groups[(int((p[0]-fx0)//tile), int((p[1]-fy0)//tile))].append(p)
    for _, bucket in sorted(groups.items(), key=lambda kv: -sum(p[4] for p in kv[1])/max(1,len(kv[1]))):
        for x, y, pixel, angle, magnitude in ordered_local(bucket):
            add_line(
                strokes, 'face_structure', x, y,
                10.0 if magnitude < 14 else 6.5,
                4.2 if magnitude < 14 else 2.8,
                hexcolor(pixel), .90, angle, role='face-plane'
            )


def collect_detail_points(image, background, subject_bbox, face_bbox, count, rng):
    pixels = image.load()
    width, height = image.size
    candidates = []
    fx0, fy0, fx1, fy1 = face_bbox
    for y in range(2, height-2, 3):
        for x in range(2, width-2, 3):
            pixel = pixels[x,y]
            region = semantic_region(x, y, pixel, background, subject_bbox, face_bbox)
            if region == 'background':
                continue
            angle, magnitude = gradient_angle(pixels, x, y, width, height)
            if magnitude < 4:
                continue
            bonus = 34 if fx0 <= x <= fx1 and fy0 <= y <= fy1 else 0
            score = magnitude + bonus + rng.random()*4
            candidates.append((score, x, y, pixel, angle, region))
    candidates.sort(reverse=True, key=lambda t:t[0])
    selected = candidates[:count]
    groups = defaultdict(list)
    tile = 76
    for item in selected:
        _, x, y, *_ = item
        groups[(x//tile, y//tile)].append(item)
    ordered = []
    def group_score(item):
        _, vals = item
        score = sum(v[0] for v in vals)/max(1,len(vals))
        if any(v[-1]=='face' for v in vals):
            score += 250
        return score
    for _, vals in sorted(groups.items(), key=group_score, reverse=True):
        points = [(v[1],v[2],v) for v in vals]
        for _,_,v in ordered_local(points):
            ordered.append(v)
    return ordered


def phase_detail(strokes, image, background, subject_bbox, face_bbox, count, rng):
    for score, x, y, pixel, angle, region in collect_detail_points(
        image, background, subject_bbox, face_bbox, count, rng
    ):
        add_line(
            strokes, 'detail', x, y,
            4.2 if region=='face' else 5.0,
            1.8 if region=='face' else 2.1,
            hexcolor(pixel), .94, angle, role=region
        )


def collect_residual(reference, rendered, background, subject_bbox, face_bbox, count):
    rp = reference.load()
    op = rendered.load()
    width, height = reference.size
    fx0, fy0, fx1, fy1 = face_bbox
    points = []
    for y in range(2, height-2, 3):
        for x in range(2, width-2, 3):
            ref = rp[x,y]
            region = semantic_region(x, y, ref, background, subject_bbox, face_bbox)
            if region == 'background':
                continue
            out = op[x,y]
            err = math.sqrt(sum((ref[i]-out[i])**2 for i in range(3))/3)
            angle, magnitude = gradient_angle(rp, x, y, width, height)
            if fx0 <= x <= fx1 and fy0 <= y <= fy1:
                err *= 1.20
            score = err * (1 + min(magnitude,80)/240)
            if score > 3:
                points.append((score,x,y,ref,angle,region))
    points.sort(reverse=True, key=lambda t:t[0])
    selected = points[:count]
    groups = defaultdict(list)
    tile=68
    for p in selected:
        groups[(p[1]//tile,p[2]//tile)].append(p)
    ordered=[]
    for _, vals in sorted(groups.items(), key=lambda kv:max(v[0] for v in kv[1]), reverse=True):
        pts=[(v[1],v[2],v) for v in vals]
        for _,_,v in ordered_local(pts):
            ordered.append(v)
    return ordered


def phase_finish(strokes, image, background_hex, background_rgb, subject_bbox, face_bbox, count):
    rendered = render_strokes(strokes, *image.size, background_hex)
    for score, x, y, pixel, angle, region in collect_residual(
        image, rendered, background_rgb, subject_bbox, face_bbox, count
    ):
        add_line(
            strokes, 'finish', x, y,
            2.8 if region=='face' else 3.4,
            1.2 if region=='face' else 1.45,
            hexcolor(pixel), .98, angle, role=region
        )
    return render_strokes(strokes, *image.size, background_hex)


def main():
    parser = argparse.ArgumentParser(
        description='参照画像から、画家のブロックインに近い6工程でストロークを生成する'
    )
    parser.add_argument('input')
    parser.add_argument('--output', default='strokes.generated.json')
    parser.add_argument('--preview', default='preview.png')
    parser.add_argument('--metrics', default='metrics.json')
    parser.add_argument('--width', type=int, default=864)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--detail', type=int, default=26000)
    parser.add_argument('--finish', type=int, default=14000)
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

    def checkpoint(stage, filename):
        rendered = render_strokes(strokes, args.width, args.height, background)
        if checkpoint_dir:
            rendered.save(checkpoint_dir/filename)
        checkpoints.append({'stage':stage,'stroke_count':len(strokes), **image_metrics(image,rendered)})
        return rendered

    phase_composition(strokes, subject_bbox, face_bbox)
    checkpoint('composition','01-composition.png')

    phase_silhouette(strokes, image, background_rgb, subject_bbox, face_bbox, rng)
    checkpoint('silhouette','02-silhouette.png')

    phase_light_shadow(strokes, image, background_rgb, subject_bbox, face_bbox, rng)
    checkpoint('light_shadow','03-light-shadow.png')

    phase_face_structure(strokes, image, face_bbox, rng)
    checkpoint('face_structure','04-face-structure.png')

    phase_detail(strokes, image, background_rgb, subject_bbox, face_bbox, args.detail, rng)
    checkpoint('detail','05-detail.png')

    rendered = phase_finish(
        strokes, image, background, background_rgb, subject_bbox, face_bbox, args.finish
    )
    if checkpoint_dir:
        rendered.save(checkpoint_dir/'06-finish.png')
    checkpoints.append({'stage':'finish','stroke_count':len(strokes), **image_metrics(image,rendered)})

    phases = [
        {'id':'composition','label':'1. 構図'},
        {'id':'silhouette','label':'2. 大きな形'},
        {'id':'light_shadow','label':'3. 明暗の面'},
        {'id':'face_structure','label':'4. 顔構造'},
        {'id':'detail','label':'5. 細部'},
        {'id':'finish','label':'6. 仕上げ'},
    ]
    for i, stroke in enumerate(strokes,1):
        stroke['id']=i

    document = {
        'metadata':{
            'slug':'painterly-process-v2',
            'title':'画家のブロックインに近い描画順',
            'seed':args.seed,
            'source_mode':'reference-guided-painterly-v2',
            'stroke_count':len(strokes),
            'subject_bbox':[round(v,2) for v in subject_bbox],
            'face_bbox':[round(v,2) for v in face_bbox],
            'quality_metrics':checkpoints[-1],
            'phase_order':[p['id'] for p in phases],
            'background_policy':'toned-ground-no-progress',
        },
        'canvas':{'width':args.width,'height':args.height,'background':background},
        'phases':phases,
        'strokes':strokes,
    }
    Path(args.output).write_text(json.dumps(document,separators=(',',':')),encoding='utf-8')
    rendered.save(args.preview)
    Path(args.metrics).write_text(json.dumps({'checkpoints':checkpoints},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({
        'strokes':len(strokes),
        'subject_bbox':subject_bbox,
        'face_bbox':face_bbox,
        'quality':checkpoints[-1],
        'phase_counts':{
            phase:sum(1 for s in strokes if s['phase']==phase)
            for phase in [p['id'] for p in phases]
        },
    },ensure_ascii=False))


if __name__ == '__main__':
    main()
