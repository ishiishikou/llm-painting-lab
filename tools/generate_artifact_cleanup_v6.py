#!/usr/bin/env python3
"""v6: remove axis-aligned rectangular face artifacts from the v5 painting process.

The v5 generator used face_bbox as clipBox for thousands of face strokes. That made
an implementation detail visible as a rectangular frame on the finished face. v6
keeps the semantic face box for observation only, but never uses it as a hard paint
boundary. Face classification is also changed from a rectangle to a soft organic
region so stroke density does not terminate on four axis-aligned sides.
"""
import json
import math
import sys
from pathlib import Path

import reference_to_brush_process as base
import generate_direction_field_v5  # noqa: F401: installs the v5 direction field into base


def face_geometry(face):
    fx0, fy0, fx1, fy1 = face
    fw = max(1.0, fx1 - fx0)
    fh = max(1.0, fy1 - fy0)
    cx = (fx0 + fx1) / 2 + fw * 0.03
    cy = (fy0 + fy1) / 2 + fh * 0.02
    return cx, cy, fw * 0.58, fh * 0.59


def organic_region(x, y, p, bg, sub, face):
    sx0, sy0, sx1, sy1 = sub
    fx0, fy0, fx1, fy1 = face
    sh = sy1 - sy0
    if not base.subject_pixel(p, bg):
        return 'background'

    # face_bbox is an observation aid, not a rectangular mask. Use a rounded,
    # slightly asymmetric support region plus the actual skin-color observation.
    fw = max(1.0, fx1 - fx0)
    fh = max(1.0, fy1 - fy0)
    cx, cy, rx, ry = face_geometry(face)
    nx = (x - cx) / rx
    ny = (y - cy) / ry
    ellipse = nx * nx + ny * ny
    near_face = fx0 - fw * 0.12 <= x <= fx1 + fw * 0.12 and fy0 - fh * 0.10 <= y <= fy1 + fh * 0.12
    if near_face and (base.skin(p) or ellipse <= 1.08):
        return 'face'

    if y < sy0 + sh * .48:
        return 'headwrap'
    if y > fy1 - (fy1 - fy0) * .05:
        return 'garment'
    return 'subject'


base.region = organic_region


def face_role(role):
    role = role or ''
    return (
        role == 'face'
        or role.startswith('face-')
        or role.startswith('underpaint-face')
        or role.startswith('mix-face')
        or role.startswith('smudge-face')
        or role.startswith('glaze-face')
        or role in {'eye-line', 'mouth-line'}
    )


def fit_face_broad_length(x, y, length, angle, face):
    """Keep broad face strokes inside a rounded support region without hard clipping.

    This prevents the new failure mode where removing clipBox exposes rows of round
    flat-brush caps outside the cheek/forehead. The stroke is shortened before it is
    painted, so no artificial edge is cut into the rendered pixels.
    """
    cx, cy, rx, ry = face_geometry(face)
    # Slightly enlarge the support region so adjacent strokes overlap naturally.
    rx *= 1.06
    ry *= 1.06
    px, py = x - cx, y - cy
    dx, dy = math.cos(angle), math.sin(angle)
    a = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)
    b = 2 * ((px * dx) / (rx * rx) + (py * dy) / (ry * ry))
    c = (px * px) / (rx * rx) + (py * py) / (ry * ry) - 1
    disc = b * b - 4 * a * c
    if disc <= 0 or a <= 0:
        return max(8.0, length * .58)
    root = math.sqrt(disc)
    t0 = (-b - root) / (2 * a)
    t1 = (-b + root) / (2 * a)
    if not (t0 < 0 < t1):
        return max(8.0, length * .62)
    half = min(-t0, t1) * .90
    return max(8.0, min(length, half * 2))


def no_face_box_clip(fn):
    def wrapped(*args, **kwargs):
        # role / clip positions follow the stable add_* signatures in
        # reference_to_brush_process.py. Only the hard clip is removed.
        name = fn.__name__
        role_index = {
            'add_variable': 10,
            'add_flat': 10,
            'add_dry': 10,
            'add_mixer': 9,
            'add_smudge': 8,
            'add_glaze': 9,
        }.get(name)
        clip_index = {
            'add_variable': 11,
            'add_flat': 11,
            'add_dry': 12,
            'add_mixer': 10,
            'add_smudge': 9,
            'add_glaze': 10,
        }.get(name)

        role = kwargs.get('role')
        if role is None and role_index is not None and len(args) > role_index:
            role = args[role_index]
        if face_role(role):
            args = list(args)
            original_clip = kwargs.get('clip')
            if original_clip is None and clip_index is not None and len(args) > clip_index:
                original_clip = args[clip_index]

            # Broad brushes should approach the organic face edge rather than leave
            # pill-shaped caps protruding into the background/headwrap.
            if name in {'add_flat', 'add_dry'} and original_clip and len(args) > 8:
                args[4] = fit_face_broad_length(args[2], args[3], args[4], args[8], original_clip)

            if 'clip' in kwargs:
                kwargs['clip'] = None
            elif clip_index is not None and len(args) > clip_index:
                args[clip_index] = None
            args = tuple(args)
        return fn(*args, **kwargs)
    return wrapped


for _name in ('add_variable', 'add_flat', 'add_dry', 'add_mixer', 'add_smudge', 'add_glaze'):
    setattr(base, _name, no_face_box_clip(getattr(base, _name)))


def arg_value(flag, default):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def stamp_v6_metadata():
    output = Path(arg_value('--output', 'strokes.generated.json'))
    metrics_path = Path(arg_value('--metrics', 'metrics.json'))
    data = json.loads(output.read_text(encoding='utf-8'))
    face_box = [round(v, 2) for v in data['metadata']['face_bbox']]
    face_strokes = [s for s in data['strokes'] if face_role(s.get('role', ''))]
    hard_face_clips = [s for s in face_strokes if s.get('clipBox')]
    exact_face_box_clips = [s for s in data['strokes'] if s.get('clipBox') == face_box]

    data['metadata'].update({
        'slug': 'artifact-cleanup-v6',
        'title': '矩形マスク痕を除去した部位別方向場の描画工程',
        'source_mode': 'reference-guided-artifact-cleanup-v6',
        'artifact_policy': 'no-hard-rectangular-face-mask-v6',
        'face_region_policy': 'organic-observation-region-no-hard-clip',
        'artifact_checks': {
            'face_role_strokes': len(face_strokes),
            'face_role_hard_clip_count': len(hard_face_clips),
            'exact_face_bbox_clip_count': len(exact_face_box_clips),
        },
    })
    output.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        metrics['artifact_checks'] = data['metadata']['artifact_checks']
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(data['metadata']['artifact_checks'], ensure_ascii=False))


if __name__ == '__main__':
    base.main()
    stamp_v6_metadata()
