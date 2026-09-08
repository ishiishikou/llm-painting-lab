#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont


def parse_hex(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def clamp(value, low=0, high=255):
    return max(low, min(high, int(round(value))))


def find_font(size, bold=False):
    candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansJP-Bold.ttf' if bold else '/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf',
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def apply_line(draw, stroke):
    rgb = parse_hex(stroke.get('color', '#ffffff'))
    alpha = clamp(stroke.get('opacity', 1.0) * 255)
    fill = (*rgb, alpha)
    width = max(1, int(round(stroke.get('width', 1))))
    x1, y1 = stroke.get('x1', 0), stroke.get('y1', 0)
    x2, y2 = stroke.get('x2', x1), stroke.get('y2', y1)
    draw.line((x1, y1, x2, y2), fill=fill, width=width)
    if width >= 3:
        radius = width / 2
        draw.ellipse((x1-radius, y1-radius, x1+radius, y1+radius), fill=fill)
        draw.ellipse((x2-radius, y2-radius, x2+radius, y2+radius), fill=fill)


def main():
    parser = argparse.ArgumentParser(description='ストローク文書から進捗表示付きMP4を生成する')
    parser.add_argument('input')
    parser.add_argument('--output', default='replay.mp4')
    parser.add_argument('--seconds', type=float, default=75.0)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--width', type=int, default=640)
    args = parser.parse_args()

    document = json.loads(Path(args.input).read_text(encoding='utf-8'))
    strokes = document.get('strokes', [])
    total = len(strokes)
    canvas_width = document['canvas']['width']
    canvas_height = document['canvas']['height']
    background = parse_hex(document['canvas'].get('background', '#111318'))
    phases = {p['id']: p.get('label', p['id']) for p in document.get('phases', [])}

    if total == 0:
        raise SystemExit('ストロークがありません')

    font_title = find_font(25, bold=True)
    font_main = find_font(23)
    font_small = find_font(19)

    header_height = 108
    output_width = args.width
    raw_height = round((canvas_height + header_height) * output_width / canvas_width)
    output_height = raw_height if raw_height % 2 == 0 else raw_height + 1

    frame_count = max(2, round(args.seconds * args.fps))
    targets = []
    for index in range(frame_count):
        target = max(1, round(total * index / (frame_count - 1)))
        if not targets or target != targets[-1]:
            targets.append(target)
    targets[-1] = total

    image = Image.new('RGBA', (canvas_width, canvas_height), (*background, 255))
    draw = ImageDraw.Draw(image, 'RGBA')
    current = 0
    frames = []
    previous_phase = None

    for target in targets:
        while current < target:
            stroke = strokes[current]
            if stroke.get('brush') == 'line':
                apply_line(draw, stroke)
            current += 1

        phase_id = strokes[current - 1].get('phase')
        phase_label = phases.get(phase_id, phase_id or '—')
        region = strokes[current - 1].get('region')
        region_label = {
            'background': '背景',
            'garment': '衣服',
            'headwrap': 'ターバン',
            'face': '顔',
            'accent': '真珠・ハイライト',
            'other': 'その他',
        }.get(region, region or '—')

        frame = Image.new('RGB', (canvas_width, canvas_height + header_height), 'white')
        frame.paste(image.convert('RGB'), (0, header_height))
        frame_draw = ImageDraw.Draw(frame)
        progress = current / total * 100

        frame_draw.text((18, 9), 'LLM Painting Lab　描画リプレイ', fill='black', font=font_title)
        frame_draw.text(
            (18, 43),
            f'{current:,} / {total:,} ストローク　{progress:5.1f}%　工程: {phase_label}',
            fill='black',
            font=font_main,
        )
        frame_draw.text(
            (18, 76),
            f'描画中の領域: {region_label}',
            fill=(70, 70, 70),
            font=font_small,
        )

        resized = frame.resize((output_width, output_height), Image.Resampling.BILINEAR)
        frames.append(resized)

        if previous_phase is not None and phase_id != previous_phase:
            frames.extend([resized.copy() for _ in range(max(1, args.fps // 2))])
        previous_phase = phase_id

    frames.extend([frames[-1].copy() for _ in range(args.fps * 2)])

    iio.imwrite(
        args.output,
        frames,
        fps=args.fps,
        codec='libx264',
        macro_block_size=1,
    )
    print(json.dumps({
        'output': args.output,
        'frames': len(frames),
        'seconds': round(len(frames) / args.fps, 2),
        'strokes': total,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
