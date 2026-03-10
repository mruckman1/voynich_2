"""
Step 39.15 -- Signal Isolation at Calibrated Dictionary
=========================================================
Run signal isolation at the calibrated dictionary.  Standard 4-class
pipeline.

Dependency chain:
    amplified_dict.json        (Step 39.14)
    targeted_vowel_fix.json    (Step 39.3)  or  combined_refine.json (P15)
    null_corpus.json           (Phase 17)
    modifier_integrate.json    (Phase 16)
    decode_10k.json            (Step 36.1)
        -> amplified_signal.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
)


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
# Signal classification
# ---------------------------------------------------------------------------

def _classify_4class(
    real_hits: List[bool],
    null_hits_list: List[List[bool]],
) -> List[str]:
    """4-class token classification."""
    n_null = len(null_hits_list)
    classifications = []
    for i in range(len(real_hits)):
        n_null_hits = sum(
            null_hits_list[j][i]
            for j in range(n_null)
            if i < len(null_hits_list[j])
        )
        real_hit = real_hits[i]
        if real_hit and n_null_hits <= 1:
            classifications.append('SIGNAL')
        elif real_hit and n_null_hits >= 3:
            classifications.append('SHARED_HIT')
        elif not real_hit and n_null_hits >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')
    return classifications


def _per_word_sigma(
    decoded_lower: List[str],
    real_hits: List[bool],
    null_hits_list: List[List[bool]],
    calibrated_dict: Set[str],
    min_freq: int = 5,
) -> List[Dict]:
    """Compute per-word sigma scores."""
    n_null = len(null_hits_list)
    n_tokens = len(decoded_lower)

    word_counts = Counter(
        w for w, h in zip(decoded_lower, real_hits) if h
    )

    null_word_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * n_null)
    for ni, nh in enumerate(null_hits_list):
        for i in range(min(len(nh), n_tokens)):
            if nh[i]:
                w = decoded_lower[i]
                null_word_counts[w][ni] += 1

    results = []
    for word, real_count in word_counts.items():
        if real_count < min_freq:
            continue

        null_counts = null_word_counts.get(word, [0] * n_null)
        null_mean = sum(null_counts) / n_null if n_null > 0 else 0.0
        null_var = (sum((c - null_mean) ** 2 for c in null_counts) / n_null
                    if n_null > 0 else 0.0)
        null_std = null_var ** 0.5
        sigma = ((real_count - null_mean) / null_std if null_std > 0
                 else (10.0 if real_count > null_mean else 0.0))
        selectivity = real_count / null_mean if null_mean > 0 else 10.0

        results.append({
            'word': word,
            'real_count': real_count,
            'null_mean': round(null_mean, 2),
            'sigma': round(sigma, 2),
            'selectivity': round(selectivity, 2),
            'is_genuine_signal': sigma > 2.0 and real_count >= min_freq,
        })

    results.sort(key=lambda x: x['sigma'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_amplified_signal() -> None:
    """Step 39.15: Signal Isolation at Calibrated Dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.15: Signal Isolation at Calibrated Dictionary")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # -- 1. Load inputs --
    print("\n  1. Loading inputs ...")

    amp_dict_data = _safe_load(os.path.join(rd, 'amplified_dict.json'))
    calibrated_words: Set[str] = set(amp_dict_data.get('calibrated_words', []))

    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    signal_38 = _safe_load(os.path.join(rd, 'merged_signal.json'))

    # Best assignment
    vowel_fix = _safe_load(os.path.join(rd, 'targeted_vowel_fix.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))

    if vowel_fix.get('corrected_assignment'):
        assignment = vowel_fix['corrected_assignment']
        assignment_source = 'targeted_vowel_fix'
    else:
        assignment = refine_data.get('best_assignment', {})
        assignment_source = 'combined_refine'

    modifier_chars: Set[str] = set(mod_data.get('modifier_chars', []))

    # Decoded tokens
    token_decoded = decode_data.get('token_decoded', [])
    token_folios = decode_data.get('token_folios', [])
    decoded_lower = [w.lower() for w in token_decoded]
    n_tokens = len(decoded_lower)

    print(f"     Calibrated dict: {len(calibrated_words)} words")
    print(f"     Tokens: {n_tokens}")
    print(f"     Assignment source: {assignment_source}")

    # -- 2. Match real corpus against calibrated dict --
    print("\n  2. Matching real corpus ...")

    real_hits = [w in calibrated_words for w in decoded_lower]
    dict_hit_rate = sum(real_hits) / max(n_tokens, 1)

    print(f"     Dict hit rate: {dict_hit_rate:.4f}")

    # -- 3. Match null corpora --
    print("\n  3. Matching null corpora ...")

    null_decoded_lists = null_data.get('null_decoded', [])
    null_hits_list: List[List[bool]] = []
    null_rates: List[float] = []

    if isinstance(null_decoded_lists, list):
        for null_decoded in null_decoded_lists[:5]:
            if isinstance(null_decoded, list):
                nd_lower = [w.lower() for w in null_decoded]
                nh = [w in calibrated_words for w in nd_lower]
                null_hits_list.append(nh)
                null_rates.append(sum(nh) / max(len(nh), 1))

    null_hit_rate = sum(null_rates) / len(null_rates) if null_rates else 0.0

    print(f"     Null hit rate: {null_hit_rate:.4f} ({len(null_rates)} null corpora)")

    # -- 4. 4-class signal classification --
    print("\n  4. Signal classification ...")

    classifications = _classify_4class(real_hits, null_hits_list)

    n_signal = sum(1 for c in classifications if c == 'SIGNAL')
    n_shared_hit = sum(1 for c in classifications if c == 'SHARED_HIT')
    n_shared_miss = sum(1 for c in classifications if c == 'SHARED_MISS')
    n_anti = sum(1 for c in classifications if c == 'ANTI_SIGNAL')
    signal_rate = n_signal / max(n_tokens, 1)

    print(f"     SIGNAL: {n_signal} ({signal_rate:.4f})")
    print(f"     SHARED_HIT: {n_shared_hit}")
    print(f"     SHARED_MISS: {n_shared_miss}")
    print(f"     ANTI_SIGNAL: {n_anti}")

    selectivity = dict_hit_rate / max(null_hit_rate, 0.001)
    print(f"     Selectivity: {selectivity:.2f}x")

    # -- 5. Per-word sigma scoring --
    print("\n  5. Per-word sigma scoring ...")

    word_signals = _per_word_sigma(
        decoded_lower, real_hits, null_hits_list, calibrated_words)

    n_genuine = sum(1 for ws in word_signals if ws.get('is_genuine_signal'))

    print(f"     Genuine signal words (sigma>2, freq>=5): {n_genuine}")
    if word_signals:
        print("     Top signal words:")
        for ws in word_signals[:15]:
            print(f"       {ws['word']:<12s} sigma={ws['sigma']:>8.2f}  "
                  f"count={ws['real_count']:>5d}")

    # -- 6. Compare vs Phase 38 merged baseline --
    print("\n  6. Comparison vs Phase 38 merged baseline ...")

    p38_signal_rate = signal_38.get('signal_rate', 0.0)
    p38_n_genuine = signal_38.get('n_genuine_signal_words', 0)

    delta_signal_rate = signal_rate - p38_signal_rate
    delta_n_genuine = n_genuine - p38_n_genuine

    print(f"     Phase 38 signal rate: {p38_signal_rate:.4f}")
    print(f"     Amplified signal rate: {signal_rate:.4f} "
          f"(delta={delta_signal_rate:+.4f})")
    print(f"     Phase 38 genuine words: {p38_n_genuine}")
    print(f"     Amplified genuine words: {n_genuine} "
          f"(delta={delta_n_genuine:+d})")

    # -- 7. Save --
    elapsed = time.time() - t0

    output = {
        'n_tokens': n_tokens,
        'dict_hit_rate': round(dict_hit_rate, 4),
        'null_hit_rate': round(null_hit_rate, 4),
        'n_signal': n_signal,
        'signal_rate': round(signal_rate, 4),
        'selectivity': round(selectivity, 2),
        'n_genuine_signal_words': n_genuine,
        'word_signals': word_signals,
        'token_classifications': classifications,
        'token_decoded': decoded_lower,
        'token_folios': token_folios,
        'delta_vs_phase38': {
            'signal_rate_delta': round(delta_signal_rate, 4),
            'n_genuine_delta': delta_n_genuine,
            'phase38_signal_rate': round(p38_signal_rate, 4),
            'phase38_n_genuine': p38_n_genuine,
        },
        'assignment_source': assignment_source,
        'verdict': (
            f"Amplified signal rate: {signal_rate:.4f} "
            f"(Phase 38: {p38_signal_rate:.4f}, "
            f"delta={delta_signal_rate:+.4f}). "
            f"{n_genuine} genuine signal words. "
            f"Selectivity: {selectivity:.2f}x."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'amplified_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved -> {out_path} ({elapsed:.1f}s)")
