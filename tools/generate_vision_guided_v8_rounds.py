#!/usr/bin/env python3
"""Render v8 base plan plus reviewer-authored round files, without reference access."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import generate_vision_guided_v8 as v8
import reference_to_brush_process as base


def load_plan(path: Path):
    plan = json.loads(path.read_text(encoding='utf-8'))
    actions = list(plan.get('actions', []))
    review_round = int(plan.get('review_round', 0))
    used = []
    disabled = set()
    round_dir = path.parent / 'v8-rounds'
    if round_dir.exists():
        for rf in sorted(round_dir.glob('round-*.json')):
            data = json.loads(rf.read_text(encoding='utf-8'))
            disabled.update(data.get('disable_actions', []))
            actions.extend(data.get('actions', []))
            review_round = max(review_round, int(data.get('review_round', 0)))
            used.append(str(rf))
    for action in actions:
        if action.get('id') in disabled:
            action['enabled'] = False
    plan['actions'] = actions
    plan['review_round'] = review_round
    plan['round_files'] = used
    plan['disabled_actions'] = sorted(disabled)
    return plan


def main():
    ap = argparse.ArgumentParser(description='v8の視覚レビューラウンドを累積して描画する')
    ap.add_argument('plan')
    ap.add_argument('--output', default='strokes.generated.json')
    ap.add_argument('--preview', default='preview.png')
    args = ap.parse_args()

    plan_path = Path(args.plan)
    plan = load_plan(plan_path)
    width = int(plan.get('canvas', {}).get('width', 864))
    height = int(plan.get('canvas', {}).get('height', 1024))
    background = plan.get('canvas', {}).get('background', '#21180d')
    seed = int(plan.get('seed', 20260908))
    strokes = []
    enabled_actions = []

    for idx, action in enumerate(plan.get('actions', []), 1):
        if not action.get('enabled', True):
            continue
        local_rng = random.Random(seed ^ (idx * 0x9E3779B1) ^ int(action.get('seed', idx)))
        if action.get('kind', 'cluster') == 'segments':
            v8.add_segments(strokes, action, width, height)
        else:
            v8.add_cluster(strokes, action, width, height, local_rng)
        enabled_actions.append(action['id'])

    for i, s in enumerate(strokes, 1):
        s['id'] = i

    image = base.render(strokes, width, height, background)
    image.save(args.preview)
    brushes = Counter(s['brush'] for s in strokes)
    phases = Counter(s['phase'] for s in strokes)
    metadata = {
        'slug': 'vision-guided-v8',
        'title': '途中画像をAIが実際に見て指示したストロークだけで描くv8',
        'source_mode': 'review-plan-only-no-reference-access',
        'reference_access_in_generator': False,
        'stroke_count': len(strokes),
        'review_round': int(plan.get('review_round', 0)),
        'round_files': plan.get('round_files', []),
        'disabled_actions': plan.get('disabled_actions', []),
        'enabled_actions': enabled_actions,
        'phase_counts': dict(phases),
        'brush_counts': dict(brushes),
        'plan_file': str(args.plan),
    }
    data = {
        'metadata': metadata,
        'canvas': {'width': width, 'height': height, 'background': background},
        'phases': [{'id': p, 'label': label} for p, label in v8.PHASES],
        'strokes': strokes,
    }
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')),
                                 encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == '__main__':
    main()
