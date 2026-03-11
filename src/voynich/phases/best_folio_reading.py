"""
Step 40.12 – Best Non-f57v Folio Reading
==========================================
Rank all folio readings by quality and produce the best non-f57v reading.

Dependency chain:
    folio_reconstruction.json  (Step 40.10)
    f57v_reading.json          (Step 40.11)
        → best_folio_reading.json  (this step)
"""

import json
import os
import time
from typing import Any, Dict, List

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_best_folio_reading() -> None:
    """Step 40.12: Best Non-f57v Folio Reading."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.12: Best Non-f57v Folio Reading")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    recon = _safe_load(os.path.join(rd, 'folio_reconstruction.json'))
    f57v_data = _safe_load(os.path.join(rd, 'f57v_reading.json'))

    folio_readings = recon.get('folio_readings', [])
    print(f"    Folio readings: {len(folio_readings)}")

    # ── 2. Score and rank folios ──
    print("\n  2. Ranking folios by quality …")
    ranked = []
    for r in folio_readings:
        coverage = r.get('coverage', 0.0)
        coherence = r.get('coherence', 0.0)
        recipe = r.get('recipe_patterns', 0)
        # Recipe completeness: normalize by token count
        recipe_norm = min(recipe / max(r.get('n_tokens', 1), 1) * 10, 1.0)
        quality = coverage * 0.4 + coherence * 0.4 + recipe_norm * 0.2
        ranked.append({
            'folio': r.get('folio', ''),
            'n_tokens': r.get('n_tokens', 0),
            'coverage': round(coverage, 4),
            'coherence': round(coherence, 4),
            'recipe_patterns': recipe,
            'quality_score': round(quality, 4),
            'max_consecutive': r.get('max_consecutive_glossed', 0),
            'best_reading': r.get('best_reading', '')[:300],
        })

    ranked.sort(key=lambda x: x['quality_score'], reverse=True)

    # Exclude f57v for best non-f57v
    non_f57v = [r for r in ranked if r['folio'] != 'f57v']

    for r in ranked[:5]:
        print(f"    {r['folio']}: quality={r['quality_score']:.3f} "
              f"(cov={r['coverage']:.2%}, coh={r['coherence']:.2%}, "
              f"recipes={r['recipe_patterns']})")

    # ── 3. Best non-f57v reading ──
    print("\n  3. Best non-f57v folio:")
    best_non_f57v = non_f57v[0] if non_f57v else None
    if best_non_f57v:
        print(f"    Folio: {best_non_f57v['folio']}")
        print(f"    Quality: {best_non_f57v['quality_score']:.3f}")
        reading_preview = best_non_f57v['best_reading'][:200]
        print(f"    Reading: {reading_preview}")
    else:
        print("    No non-f57v readings available")

    # ── 4. Aggregate statistics ──
    print("\n  4. Aggregate statistics:")
    if ranked:
        mean_coverage = sum(r['coverage'] for r in ranked) / len(ranked)
        mean_coherence = sum(r['coherence'] for r in ranked) / len(ranked)
        n_with_recipes = sum(1 for r in ranked if r['recipe_patterns'] > 0)
        print(f"    Mean coverage: {mean_coverage:.2%}")
        print(f"    Mean coherence: {mean_coherence:.2%}")
        print(f"    Folios with recipe patterns: {n_with_recipes}/{len(ranked)}")
    else:
        mean_coverage = 0.0
        mean_coherence = 0.0
        n_with_recipes = 0

    # ── 5. f57v comparison ──
    f57v_cov = f57v_data.get('coverage_pct', 0.0)
    f57v_coh = f57v_data.get('coherence_score', 0.0)
    print(f"\n  5. f57v comparison:")
    print(f"    f57v coverage: {f57v_cov:.2%}, coherence: {f57v_coh:.2%}")
    if best_non_f57v:
        print(f"    Best non-f57v coverage: {best_non_f57v['coverage']:.2%}, "
              f"coherence: {best_non_f57v.get('coherence', 0):.2%}")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'ranked_folios': ranked,
        'best_folio': ranked[0]['folio'] if ranked else '',
        'best_quality_score': ranked[0]['quality_score'] if ranked else 0.0,
        'best_non_f57v_folio': best_non_f57v['folio'] if best_non_f57v else '',
        'best_non_f57v_quality': best_non_f57v['quality_score'] if best_non_f57v else 0.0,
        'best_non_f57v_reading': best_non_f57v['best_reading'] if best_non_f57v else '',
        'aggregate_coverage': round(mean_coverage, 4),
        'aggregate_coherence': round(mean_coherence, 4),
        'n_folios_with_recipes': n_with_recipes,
        'f57v_coverage': round(f57v_cov, 4),
        'f57v_coherence': round(f57v_coh, 4),
        'phase40_reading_verdict': (
            'READABLE' if mean_coverage > 0.3 and mean_coherence > 0.1
            else 'PARTIALLY_READABLE' if mean_coverage > 0.15
            else 'NOT_READABLE'
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'best_folio_reading.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
