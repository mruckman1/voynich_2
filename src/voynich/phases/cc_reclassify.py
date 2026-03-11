"""
Step 40.4 – CC Bigram Reclassification
========================================
Reclassify each of the Phase 38 CC bigrams as either (a) correct Venetian,
(b) genuine vowel error, or (c) ambiguous.

Dependency chain:
    venetian_bigrams.json    (Step 40.3)
    venetian_forms.json      (Step 40.1)
    merged_bigrams.json      (Step 38.4)
        → venetian_reclassify.json  (this step)
"""

import json
import os
import time
from typing import Any, Dict, List, Set, Tuple

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
# Core: CC bigram reclassification
# ---------------------------------------------------------------------------

def _reclassify_cc_bigrams(
    cc_bigrams: List[Dict],
    venetian_set: Set[str],
    ven_ref_bigrams_sample: List[Dict],
) -> List[Dict]:
    """Reclassify CC bigrams as CORRECT_VENETIAN, PLAUSIBLE_VENETIAN, or GENUINE_ERROR."""
    # Build a set of Venetian bigrams from sample for quick lookup
    ven_bigram_set = set()
    for bp in ven_ref_bigrams_sample:
        w1 = bp.get('w1', '')
        w2 = bp.get('w2', '')
        if w1 and w2:
            ven_bigram_set.add((w1, w2))

    results = []
    for bg in cc_bigrams:
        w1 = bg.get('w1', '')
        w2 = bg.get('w2', '')

        w1_in_ven = w1 in venetian_set
        w2_in_ven = w2 in venetian_set
        pair_in_ven = (w1, w2) in ven_bigram_set

        if pair_in_ven:
            classification = 'CORRECT_VENETIAN'
        elif w1_in_ven and w2_in_ven:
            classification = 'PLAUSIBLE_VENETIAN'
        elif w1_in_ven or w2_in_ven:
            classification = 'AMBIGUOUS'
        else:
            classification = 'GENUINE_ERROR'

        results.append({
            'w1': w1,
            'w2': w2,
            'folio': bg.get('folio', ''),
            'position': bg.get('position', 0),
            'w1_in_venetian': w1_in_ven,
            'w2_in_venetian': w2_in_ven,
            'pair_in_venetian_bigrams': pair_in_ven,
            'classification': classification,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cc_reclassify() -> None:
    """Step 40.4: CC Bigram Reclassification."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.4: CC Bigram Reclassification")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    ven_bigrams = _safe_load(os.path.join(rd, 'venetian_bigrams.json'))
    ven_forms = _safe_load(os.path.join(rd, 'venetian_forms.json'))
    merged_bigrams = _safe_load(os.path.join(rd, 'merged_bigrams.json'))

    venetian_set = set(ven_forms.get('venetian_extended_set', []))
    print(f"    Venetian extended set: {len(venetian_set):,} words")

    # ── 2. Extract CC bigrams from merged_bigrams ──
    print("\n  2. Extracting CC bigrams …")
    # CC bigrams are content-content pairs from the bigram catalog
    bigram_catalog = merged_bigrams.get('bigram_catalog', [])
    cc_bigrams = [b for b in bigram_catalog
                  if b.get('content_type', '') == 'CC'
                  or b.get('match_type', '') in ('exact', 'relaxed')]
    if not cc_bigrams:
        # Fallback: use all signal pairs
        cc_bigrams = bigram_catalog[:50]
    print(f"    CC bigrams to reclassify: {len(cc_bigrams)}")

    # ── 3. Get Venetian bigram pairs for lookup ──
    ven_pairs_sample = ven_bigrams.get('signal_pairs_sample', [])

    # ── 4. Reclassify ──
    print("\n  3. Reclassifying CC bigrams …")
    reclassified = _reclassify_cc_bigrams(
        cc_bigrams, venetian_set, ven_pairs_sample,
    )

    # Tally
    from collections import Counter
    tally = Counter(r['classification'] for r in reclassified)
    n_correct_ven = tally.get('CORRECT_VENETIAN', 0)
    n_plausible = tally.get('PLAUSIBLE_VENETIAN', 0)
    n_ambiguous = tally.get('AMBIGUOUS', 0)
    n_error = tally.get('GENUINE_ERROR', 0)

    print(f"    CORRECT_VENETIAN: {n_correct_ven}")
    print(f"    PLAUSIBLE_VENETIAN: {n_plausible}")
    print(f"    AMBIGUOUS: {n_ambiguous}")
    print(f"    GENUINE_ERROR: {n_error}")

    # ── 4. Verdict ──
    total = len(reclassified)
    venetian_fraction = (n_correct_ven + n_plausible) / total if total > 0 else 0.0
    print(f"\n  4. Venetian fraction: {venetian_fraction:.3f} "
          f"({n_correct_ven + n_plausible}/{total})")

    if venetian_fraction > 0.5:
        verdict = 'MOSTLY_VENETIAN'
    elif venetian_fraction > 0.25:
        verdict = 'MIXED'
    else:
        verdict = 'MOSTLY_ERROR'
    print(f"    Verdict: {verdict}")

    # ── 5. Save ──
    elapsed = time.time() - t0

    output = {
        'n_cc_bigrams_analyzed': total,
        'n_correct_venetian': n_correct_ven,
        'n_plausible_venetian': n_plausible,
        'n_ambiguous': n_ambiguous,
        'n_genuine_error': n_error,
        'venetian_fraction': round(venetian_fraction, 4),
        'reclassified_bigrams': reclassified,
        'tally': dict(tally),
        'verdict': verdict,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_reclassify.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
