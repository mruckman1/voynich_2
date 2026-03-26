"""Phase 63 Step A1: Render EVA glyphs from font as 224x224 PNGs."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from voynich.core._paths import data_dir, results_dir


@dataclass
class RenderResult:
    n_rendered: int = 0
    n_failed: int = 0
    rendered_chars: List[str] = field(default_factory=list)
    failed_chars: List[str] = field(default_factory=list)
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


def run_p63_render():
    """Render all EVA characters from the font file."""
    t0 = time.time()
    rd = str(results_dir())

    from voynich.visual.render_eva import EVA_RENDER_MAP, render_all_eva

    font_path = str(data_dir('Voynich EVA Hand A.ttf'))
    output_dir = os.path.join(rd, 'p63_renders')

    if not os.path.exists(font_path):
        print(f"ERROR: Font file not found at {font_path}")
        return

    print(f"Phase 63 A1: Rendering {len(EVA_RENDER_MAP)} EVA characters...")
    print(f"  Font: {font_path}")
    print(f"  Output: {output_dir}")

    metadata = render_all_eva(font_path, output_dir)

    rendered = [m['eva_name'] for m in metadata]
    failed = [c for c in EVA_RENDER_MAP if c not in rendered]

    result = RenderResult(
        n_rendered=len(rendered),
        n_failed=len(failed),
        rendered_chars=rendered,
        failed_chars=failed,
        output_dir=output_dir,
        elapsed=time.time() - t0,
    )

    # Save metadata for downstream steps
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(_convert(metadata), f, indent=2)

    _save_json(rd, 'p63_render.json', asdict(result))

    print(f"\n  Rendered: {result.n_rendered}")
    print(f"  Failed:   {result.n_failed}")
    if failed:
        print(f"  Failed chars: {failed}")
    print(f"  Elapsed: {result.elapsed:.1f}s")
