"""Phase 63B Step B5: Select character exemplars and normalize."""

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from voynich.core._paths import results_dir


TARGET_EXEMPLARS = 20
MIN_EXEMPLARS = 3


@dataclass
class CharTypeStats:
    eva_char: str = ''
    total_instances: int = 0
    after_filter: int = 0
    n_exemplars: int = 0
    median_width: float = 0.0
    median_height: float = 0.0


@dataclass
class ExemplarResult:
    n_char_types: int = 0
    n_char_types_with_exemplars: int = 0
    n_total_exemplars: int = 0
    n_total_instances: int = 0
    per_char: List[CharTypeStats] = field(default_factory=list)
    output_dir: str = ''
    elapsed: float = 0.0


def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, set):
        return sorted(obj)
    return obj


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


def _score_crop(img_array, median_width, median_height):
    """Score a character crop for quality. Higher = better.

    Considers:
    - Width relative to median (penalize outliers)
    - Ink density (too little or too much is bad)
    - Centering of ink mass
    """
    h, w = img_array.shape[:2]
    if h == 0 or w == 0:
        return -1.0

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array.astype(float)

    ink = gray < 180  # ink pixels
    ink_frac = ink.sum() / ink.size

    # Reject extremes
    if ink_frac < 0.02 or ink_frac > 0.65:
        return -1.0

    # Width score (penalize outliers)
    if median_width > 0:
        width_ratio = w / median_width
        if width_ratio < 0.3 or width_ratio > 3.0:
            return -1.0
        width_score = 1.0 - abs(1.0 - width_ratio) * 0.5
    else:
        width_score = 0.5

    # Ink density score (prefer moderate density)
    density_score = 1.0 - abs(0.15 - ink_frac) * 3.0

    # Centering score
    if ink.sum() > 0:
        rows, cols = np.where(ink)
        center_y = rows.mean() / h
        center_x = cols.mean() / w
        centering = 1.0 - (abs(center_y - 0.5) + abs(center_x - 0.5))
    else:
        centering = 0.0

    return width_score * 0.3 + density_score * 0.4 + centering * 0.3


def run_p63b_exemplars():
    """Select best character exemplars from segmented crops."""
    t0 = time.time()
    rd = str(results_dir())

    from voynich.visual.normalize import normalize_image

    seg_dir = os.path.join(rd, 'p63b_segments')
    if not os.path.exists(seg_dir):
        print("ERROR: Segments not found. Run ms-segment first.")
        return

    output_dir = os.path.join(rd, 'p63b_exemplars')
    os.makedirs(output_dir, exist_ok=True)

    # Collect all character crops grouped by EVA char type
    char_instances = defaultdict(list)

    for folio_id in os.listdir(seg_dir):
        folio_dir = os.path.join(seg_dir, folio_id)
        meta_path = os.path.join(folio_dir, 'char_metadata.json')
        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            char_meta = json.load(f)

        chars_dir = os.path.join(folio_dir, 'chars')
        for entry in char_meta:
            eva_char = entry['eva_char']
            img_path = os.path.join(chars_dir, entry['filename'])
            if os.path.exists(img_path):
                char_instances[eva_char].append({
                    'path': img_path,
                    'width': entry['width'],
                    'height': entry['height'],
                    'folio': entry['folio'],
                })

    print(f"Phase 63B B5: Selecting exemplars...")
    print(f"  Character types found: {len(char_instances)}")
    print(f"  Total instances: {sum(len(v) for v in char_instances.values())}")

    per_char_stats = []
    total_exemplars = 0
    types_with_exemplars = 0

    for eva_char in sorted(char_instances.keys()):
        instances = char_instances[eva_char]
        n_total = len(instances)

        # Compute median dimensions
        widths = [inst['width'] for inst in instances]
        heights = [inst['height'] for inst in instances]
        median_w = float(np.median(widths)) if widths else 0
        median_h = float(np.median(heights)) if heights else 0

        # Score each instance
        scored = []
        for inst in instances:
            try:
                img = np.array(Image.open(inst['path']))
                score = _score_crop(img, median_w, median_h)
                if score > 0:
                    scored.append((score, inst))
            except Exception:
                continue

        # Select top exemplars
        scored.sort(key=lambda x: x[0], reverse=True)
        n_select = min(TARGET_EXEMPLARS, len(scored))

        if n_select < MIN_EXEMPLARS:
            per_char_stats.append(CharTypeStats(
                eva_char=eva_char, total_instances=n_total,
                after_filter=len(scored), n_exemplars=0,
                median_width=median_w, median_height=median_h,
            ))
            continue

        # Save exemplars
        safe_name = eva_char.replace('/', '_')
        char_dir = os.path.join(output_dir, safe_name)
        os.makedirs(char_dir, exist_ok=True)

        exemplar_paths = []
        for rank, (score, inst) in enumerate(scored[:n_select]):
            img = Image.open(inst['path'])
            normalized = normalize_image(img)
            out_path = os.path.join(char_dir, f'exemplar_{rank:03d}.png')
            normalized.save(out_path)
            exemplar_paths.append(out_path)

        total_exemplars += n_select
        types_with_exemplars += 1

        per_char_stats.append(CharTypeStats(
            eva_char=eva_char, total_instances=n_total,
            after_filter=len(scored), n_exemplars=n_select,
            median_width=median_w, median_height=median_h,
        ))

    result = ExemplarResult(
        n_char_types=len(char_instances),
        n_char_types_with_exemplars=types_with_exemplars,
        n_total_exemplars=total_exemplars,
        n_total_instances=sum(len(v) for v in char_instances.values()),
        per_char=per_char_stats,
        output_dir=output_dir,
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63b_exemplars.json', asdict(result))

    print(f"\n  Types with exemplars: {types_with_exemplars}/{len(char_instances)}")
    print(f"  Total exemplars selected: {total_exemplars}")

    # Show top 10 by instance count
    top = sorted(per_char_stats, key=lambda x: x.total_instances, reverse=True)[:10]
    print(f"\n  Top character types:")
    for s in top:
        print(f"    '{s.eva_char}': {s.total_instances} instances -> {s.n_exemplars} exemplars")

    print(f"\n  Elapsed: {result.elapsed:.1f}s")
