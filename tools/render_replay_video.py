#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PHASE_WEIGHTS = {
    'composition': 0.10,
    'silhouette': 0.16,
    'light_shadow': 0.25,
    'face_structure': 0.15,
    'detail': 0.23,
    'finish': 0.11,
}


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


def stroke_midpoint(stroke):
    return (
        (stroke.get('x1', stroke.get('x', 0)) + stroke.get('x2', stroke.get('x', 0))) / 2,
        (stroke.get('y1', stroke.get('y', 0)) + stroke.get('y2', stroke.get('y', 0))) / 2,
    )


def should_draw(stroke, subject_bbox):
    if stroke.get('phase') == 'composition' or not subject_bbox:
        return True
    x, y = stroke_midpoint(stroke)
    x0, y0, x1, y1 = subject_bbox
    pad = max(18, min(x1 - x0, y1 - y0) * 0.05)
    return x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad


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
        if stroke.get('lineCap', 'round') == 'round' and width >= 3:
            radius = width / 2
            draw.ellipse((x1-radius, y1-radius, x1+radius, y1+radius), fill=fill)
            draw.ellipse((x2-radius, y2-radius, x2+radius, y2+radius), fill=fill)


def phase_slices(strokes, phase_order):
    indices = defaultdict(list)
    for idx, stroke in enumerate(strokes):
        indices[stroke.get('phase')].append(idx)
    result = []
    for phase in phase_order:
        values = indices.get(phase, [])
        if values:
            result.append((phase, values[0], values[-1] + 1))
    return result


def main():
    parser = argparse.ArgumentParser(description='工程ごとの時間配分で、日本語進捗付きMP4を生成する')
    parser.add_argument('input')
    parser.add_argument('--output', default='replay.mp4')
    parser.add_argument('--seconds', type=float, default=82.0)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=640)
    args = parser.parse_args()

    document = json.loads(Path(args.input).read_text(encoding='utf-8'))
    strokes = document.get('strokes', [])
    if not strokes:
        raise SystemExit('ストロークがありません')

    phase_order = [p['id'] for p in document.get('phases', [])]
    labels = {p['id']: p.get('label', p['id']) for p in document.get('phases', [])}
    slices = phase_slices(strokes, phase_order)
    total = len(strokes)
    subject_bbox = document.get('metadata', {}).get('subject_bbox')

    canvas_width = document['canvas']['width']
    canvas_height = document['canvas']['height']
    background = parse_hex(document['canvas'].get('background', '#111318'))

    header_height = 126
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

    writer = imageio.get_writer(
        args.output, fps=args.fps, codec='libx264',
        macro_block_size=1, quality=7
    )

    total_frames = max(1, int(args.seconds * args.fps))
    opening_frames = args.fps * 2
    ending_frames = args.fps * 2
    usable = max(1, total_frames - opening_frames - ending_frames)

    raw_weights = [PHASE_WEIGHTS.get(phase, 1/len(slices)) for phase,_,_ in slices]
    weight_sum = sum(raw_weights) or 1
    phase_frames = [max(1, round(usable * w / weight_sum)) for w in raw_weights]
    phase_frames[-1] += usable - sum(phase_frames)

    phase_number = {phase: i+1 for i,(phase,_,_) in enumerate(slices)}

    def append_frame(index, phase_id=None, phase_progress=0.0, note=''):
        page = Image.new('RGB', (canvas_width, canvas_height + header_height), 'white')
        page.paste(painting, (0, header_height))
        d = ImageDraw.Draw(page)
        if phase_id is None:
            phase_label = '開始前'
            phase_no = 0
        else:
            phase_label = labels.get(phase_id, phase_id)
            phase_no = phase_number.get(phase_id, 0)
        d.text((18, 9), 'LLM Painting Lab　画家に近い描画工程リプレイ', fill='black', font=font_title)
        if phase_id:
            d.text(
                (18, 43),
                f'工程 {phase_no} / {len(slices)}　{phase_label}　工程内 {phase_progress:5.1f}%',
                fill='black', font=font_main
            )
        else:
            d.text((18,43),'地塗り済みキャンバスから開始',fill='black',font=font_main)
        d.text(
            (18, 78),
            note or '背景の細かな更新は省き、人物の描画工程を中心に再生します',
            fill=(65,65,65), font=font_small
        )
        resized = page.resize((output_width, output_height), Image.Resampling.BILINEAR)
        writer.append_data(np.asarray(resized))

    for _ in range(opening_frames):
        append_frame(0, note='背景は地塗りとして最初から置き、人物の描写を追います')

    current = 0
    for (phase_id, start, end), frames_for_phase in zip(slices, phase_frames):
        while current < start:
            if should_draw(strokes[current], subject_bbox):
                apply_stroke(painting_draw, strokes[current])
            current += 1
        count = end - start
        for frame_index in range(frames_for_phase):
            target = start + max(1, round(count * (frame_index + 1) / frames_for_phase))
            target = min(target, end)
            while current < target:
                if should_draw(strokes[current], subject_bbox):
                    apply_stroke(painting_draw, strokes[current])
                current += 1
            progress = (current - start) / max(1, count) * 100
            append_frame(current, phase_id, progress, '工程切替' if frame_index == 0 else '')

    while current < total:
        if should_draw(strokes[current], subject_bbox):
            apply_stroke(painting_draw, strokes[current])
        current += 1

    for _ in range(ending_frames):
        append_frame(total, slices[-1][0], 100.0, '完成')

    writer.close()
    print(json.dumps({
        'output': args.output,
        'seconds': args.seconds,
        'fps': args.fps,
        'strokes': total,
        'subject_bbox': subject_bbox,
        'phase_frames': {
            phase: frames for (phase,_,_),frames in zip(slices,phase_frames)
        },
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
