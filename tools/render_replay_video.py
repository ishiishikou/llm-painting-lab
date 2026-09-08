#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_hex(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def clamp(value, low=0, high=255):
    return max(low, min(high, int(round(value))))


def find_font(bold=False):
    candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def apply_stroke(draw, stroke):
    rgb = parse_hex(stroke.get('color', '#ffffff'))
    alpha = clamp(stroke.get('opacity', 1.0) * 255)
    fill = (*rgb, alpha)
    brush = stroke.get('brush', 'line')
    if brush == 'line':
        width = max(1, int(round(stroke.get('width', 1))))
        x1, y1 = stroke['x1'], stroke['y1']
        x2, y2 = stroke['x2'], stroke['y2']
        draw.line((x1, y1, x2, y2), fill=fill, width=width)
        if width >= 3:
            radius = width / 2
            draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=fill)
            draw.ellipse((x2 - radius, y2 - radius, x2 + radius, y2 + radius), fill=fill)
    elif brush == 'ellipse':
        x, y = stroke['x'], stroke['y']
        rx, ry = stroke.get('rx', 10), stroke.get('ry', 10)
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=fill)


def main():
    parser = argparse.ArgumentParser(description='6段階のストローク文書から、日本語進捗付きMP4を生成する')
    parser.add_argument('input')
    parser.add_argument('--output', default='replay.mp4')
    parser.add_argument('--seconds', type=float, default=82.0)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=640)
    args = parser.parse_args()

    document = json.loads(Path(args.input).read_text(encoding='utf-8'))
    strokes = document.get('strokes', [])
    total = len(strokes)
    if total == 0:
        raise SystemExit('ストロークがありません')

    canvas_width = document['canvas']['width']
    canvas_height = document['canvas']['height']
    background = parse_hex(document['canvas'].get('background', '#111318'))
    labels = {phase['id']: phase.get('label', phase['id']) for phase in document.get('phases', [])}

    boundaries = []
    previous = None
    for index, stroke in enumerate(strokes, 1):
        phase = stroke.get('phase')
        if phase != previous:
            boundaries.append((index, phase))
            previous = phase

    hold_frames = max(1, args.fps // 2)
    opening_frames = args.fps
    ending_frames = args.fps * 2
    main_frames = max(
        2,
        int(args.seconds * args.fps) - len(boundaries) * hold_frames - opening_frames - ending_frames,
    )
    targets = []
    for index in range(main_frames):
        target = max(1, round(total * index / (main_frames - 1)))
        if not targets or target != targets[-1]:
            targets.append(target)
    targets[-1] = total

    header_height = 116
    output_width = args.width
    output_height = math.ceil((canvas_height + header_height) * output_width / canvas_width)
    if output_height % 2:
        output_height += 1

    regular_path = find_font(False)
    bold_path = find_font(True)
    font_main = ImageFont.truetype(regular_path, 22) if regular_path else ImageFont.load_default()
    font_small = ImageFont.truetype(regular_path, 18) if regular_path else ImageFont.load_default()
    font_title = ImageFont.truetype(bold_path, 24) if bold_path else font_main

    painting = Image.new('RGB', (canvas_width, canvas_height), background)
    painting_draw = ImageDraw.Draw(painting, 'RGBA')
    current = 0
    previous_phase = None

    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec='libx264',
        macro_block_size=1,
        quality=7,
    )

    def append_frame(index, note=''):
        page = Image.new('RGB', (canvas_width, canvas_height + header_height), 'white')
        page.paste(painting, (0, header_height))
        page_draw = ImageDraw.Draw(page)
        if index:
            phase_id = strokes[index - 1].get('phase')
            phase_label = labels.get(phase_id, phase_id or '—')
        else:
            phase_label = '開始前'
        progress = index / total * 100
        page_draw.text((18, 10), 'LLM Painting Lab　人間の描画順リプレイ', fill='black', font=font_title)
        page_draw.text(
            (18, 43),
            f'{index:,} / {total:,} ストローク　{progress:5.1f}%　工程: {phase_label}',
            fill='black',
            font=font_main,
        )
        page_draw.text(
            (18, 77),
            note or '構図 → シルエット → 明暗の面 → 顔構造 → 細部 → 仕上げ',
            fill=(65, 65, 65),
            font=font_small,
        )
        resized = page.resize((output_width, output_height), Image.Resampling.BILINEAR)
        writer.append_data(np.asarray(resized))

    for _ in range(opening_frames):
        append_frame(0, '6段階の描画順をそのまま再生します')

    for target in targets:
        while current < target:
            apply_stroke(painting_draw, strokes[current])
            current += 1

        current_phase = strokes[current - 1].get('phase')
        transitioned = current_phase != previous_phase
        append_frame(current, '工程切替' if transitioned else '')
        if transitioned:
            for _ in range(hold_frames - 1):
                append_frame(current, '工程切替')
            previous_phase = current_phase

    for _ in range(ending_frames):
        append_frame(total, '完成')

    writer.close()
    print(json.dumps({
        'output': args.output,
        'seconds': args.seconds,
        'fps': args.fps,
        'strokes': total,
        'phase_order': [phase for _, phase in boundaries],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
