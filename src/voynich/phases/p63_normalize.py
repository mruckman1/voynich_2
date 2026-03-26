"""Phase 63 Step A2: Normalize EVA renders + Costamagna crops to 224x224."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from voynich.core._paths import data_dir, results_dir


@dataclass
class NormalizeResult:
    n_costamagna: int = 0
    n_eva: int = 0
    n_costa_failed: int = 0
    n_eva_failed: int = 0
    output_dir: str = ''
    image_size: int = 224
    elapsed: float = 0.0


def _convert(obj: Any) -> Any:
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


def _safe_load(path: str) -> Any:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def run_p63_normalize():
    """Normalize both EVA renders and Costamagna crops."""
    t0 = time.time()
    rd = str(results_dir())

    from voynich.visual.normalize import (
        normalize_costamagna_crops,
        normalize_eva_renders,
    )

    # Load Costamagna data
    crops_dir = str(data_dir('GL.S.III.MISC.12/costamagna_crops'))
    costa_meta_path = os.path.join(crops_dir, 'metadata.json')
    syl_table_path = str(data_dir('GL.S.III.MISC.12/extraction/syllabary_table.json'))

    costa_meta = _safe_load(costa_meta_path)
    syllabary = _safe_load(syl_table_path)

    if not costa_meta:
        print("ERROR: Costamagna metadata.json not found")
        return
    if not syllabary:
        print("ERROR: syllabary_table.json not found")
        return

    # Load EVA render metadata from A1
    eva_meta_path = os.path.join(rd, 'p63_renders', 'metadata.json')
    eva_meta = _safe_load(eva_meta_path)
    if not eva_meta:
        print("ERROR: EVA render metadata not found. Run vis-render first.")
        return

    output_dir = os.path.join(rd, 'p63_normalized')
    costa_out = os.path.join(output_dir, 'costamagna')
    eva_out = os.path.join(output_dir, 'eva')

    print(f"Phase 63 A2: Normalizing images...")
    print(f"  Costamagna crops: {len(costa_meta)} signs")
    print(f"  EVA renders: {len(eva_meta)} characters")

    # Normalize Costamagna
    print("\n  Normalizing Costamagna crops...")
    costa_result = normalize_costamagna_crops(
        crops_dir, costa_meta, syllabary, costa_out)

    # Normalize EVA
    print("  Normalizing EVA renders...")
    eva_result = normalize_eva_renders(
        os.path.join(rd, 'p63_renders'), eva_meta, eva_out)

    # Save enriched metadata
    with open(os.path.join(output_dir, 'costamagna_metadata.json'), 'w') as f:
        json.dump(_convert(costa_result), f, indent=2)
    with open(os.path.join(output_dir, 'eva_metadata.json'), 'w') as f:
        json.dump(_convert(eva_result), f, indent=2)

    result = NormalizeResult(
        n_costamagna=len(costa_result),
        n_eva=len(eva_result),
        n_costa_failed=len(costa_meta) - len(costa_result),
        n_eva_failed=len(eva_meta) - len(eva_result),
        output_dir=output_dir,
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63_normalize.json', asdict(result))

    print(f"\n  Costamagna normalized: {result.n_costamagna} (failed: {result.n_costa_failed})")
    print(f"  EVA normalized: {result.n_eva} (failed: {result.n_eva_failed})")
    print(f"  Elapsed: {result.elapsed:.1f}s")
