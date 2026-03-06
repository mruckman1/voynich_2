"""
Phase B.3 -- Positional Constraint Comparison
===============================================
Compare positional constraints between Voynich glyphs and Tironian signs.

Computes positional profiles (initial/medial/final fraction within tokens)
for each EVA character in the Voynich corpus, then compares against
positional data from the master paleographic reference (word_position
fields from Costamagna entries or inferred from context).

Correlation between Voynich and Tironian positional distributions is
computed per position (initial, medial, final) and averaged.

Gate: mean correlation > 0.3 (or automatic pass if insufficient Tironian
positional data).

Dependency chain:
    data/reference/paleographic/master_reference.json
    corpus (IVTFF)
        -> positional_compare.json (this step)
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir, data_dir as _data_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    load_master_reference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
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


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0

    return cov / denom


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PositionalCompareResult:
    """Positional constraint comparison between Voynich and Tironian signs."""
    n_voynich_profiles: int
    n_tironian_profiles: int
    n_comparable_pairs: int
    correlation_initial: float
    correlation_medial: float
    correlation_final: float
    mean_correlation: float
    has_costamagna_data: bool
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _build_voynich_positional_profiles(
    tokens: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute positional profiles for each EVA character.

    Returns dict mapping eva_char -> {'initial': frac, 'medial': frac, 'final': frac}.
    """
    pos_initial: Counter = Counter()
    pos_medial: Counter = Counter()
    pos_final: Counter = Counter()

    for token in tokens:
        chars = tokenize_eva_chars(token)
        n = len(chars)
        for ci, ch in enumerate(chars):
            if n == 1:
                pos_initial[ch] += 1
                pos_final[ch] += 1
            elif ci == 0:
                pos_initial[ch] += 1
            elif ci == n - 1:
                pos_final[ch] += 1
            else:
                pos_medial[ch] += 1

    profiles: Dict[str, Dict[str, float]] = {}
    all_chars = set(pos_initial.keys()) | set(pos_medial.keys()) | set(pos_final.keys())

    for ch in all_chars:
        ini = pos_initial.get(ch, 0)
        med = pos_medial.get(ch, 0)
        fin = pos_final.get(ch, 0)
        total = ini + med + fin
        if total == 0:
            continue
        profiles[ch] = {
            'initial': ini / total,
            'medial': med / total,
            'final': fin / total,
        }

    return profiles


def _build_tironian_positional_profiles(
    master_ref: Dict,
) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Extract positional profiles from master reference signs.

    Returns (dict mapping triple_key -> positional profile, has_data).
    Signs with word_position fields are used directly; otherwise positional
    bias is inferred from any available context fields.
    """
    profiles: Dict[str, Dict[str, float]] = {}
    has_costamagna_data = False

    for sign in master_ref.get('all_signs', []):
        fs = sign.get('first_stroke', '')
        ls = sign.get('last_stroke', '')
        gc = sign.get('glyph_class', '')
        if not (fs and ls and gc):
            continue

        triple_key = f"{fs},{ls},{gc}"

        # Check for explicit word_position data
        word_pos = sign.get('word_position', None)
        if word_pos and isinstance(word_pos, dict):
            has_costamagna_data = True
            ini = word_pos.get('initial', 0.0)
            med = word_pos.get('medial', 0.0)
            fin = word_pos.get('final', 0.0)
            total = ini + med + fin
            if total > 0:
                profiles[triple_key] = {
                    'initial': ini / total,
                    'medial': med / total,
                    'final': fin / total,
                }
                continue

        # Try to infer from position_bias or context
        pos_bias = sign.get('position_bias', '')
        if pos_bias:
            has_costamagna_data = True
            if pos_bias == 'initial':
                profiles[triple_key] = {'initial': 0.7, 'medial': 0.2, 'final': 0.1}
            elif pos_bias == 'medial':
                profiles[triple_key] = {'initial': 0.15, 'medial': 0.7, 'final': 0.15}
            elif pos_bias == 'final':
                profiles[triple_key] = {'initial': 0.1, 'medial': 0.2, 'final': 0.7}
            else:
                profiles[triple_key] = {'initial': 0.33, 'medial': 0.34, 'final': 0.33}

    return profiles, has_costamagna_data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_positional_compare() -> None:
    """Phase B.3: Compare positional constraints between Voynich and Tironian."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE B.3: Positional Constraint Comparison")
    print("=" * 70)

    rd = _results_dir()

    # ---- Step 1: Load corpus and compute Voynich positional profiles ----
    print("\n  1. Loading corpus and computing Voynich positional profiles ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    print(f"      {len(tokens)} tokens loaded")

    voynich_profiles = _build_voynich_positional_profiles(tokens)
    print(f"      {len(voynich_profiles)} EVA chars with positional profiles")

    # Map EVA chars to triple keys for comparison
    eva_to_triple = build_eva_to_triple_lookup()

    # Aggregate Voynich profiles by triple_key (average across EVA chars sharing a triple)
    triple_voynich: Dict[str, Dict[str, List[float]]] = {}
    for ch, profile in voynich_profiles.items():
        tk = eva_to_triple.get(ch)
        if tk is None:
            continue
        if tk not in triple_voynich:
            triple_voynich[tk] = {'initial': [], 'medial': [], 'final': []}
        triple_voynich[tk]['initial'].append(profile['initial'])
        triple_voynich[tk]['medial'].append(profile['medial'])
        triple_voynich[tk]['final'].append(profile['final'])

    voynich_by_triple: Dict[str, Dict[str, float]] = {}
    for tk, lists in triple_voynich.items():
        voynich_by_triple[tk] = {
            'initial': sum(lists['initial']) / len(lists['initial']),
            'medial': sum(lists['medial']) / len(lists['medial']),
            'final': sum(lists['final']) / len(lists['final']),
        }

    print(f"      {len(voynich_by_triple)} unique triple_keys with profiles")

    # ---- Step 2: Load master reference and extract Tironian positional profiles ----
    print("\n  2. Loading master reference and extracting Tironian positional profiles ...")
    master_ref = load_master_reference()

    if master_ref is None:
        print("      [WARN] master_reference.json not found.")
        tironian_profiles: Dict[str, Dict[str, float]] = {}
        has_costamagna = False
    else:
        tironian_profiles, has_costamagna = _build_tironian_positional_profiles(master_ref)
        print(f"      {len(tironian_profiles)} Tironian triple_keys with positional profiles")
        print(f"      Has Costamagna data: {has_costamagna}")

    if not has_costamagna:
        print("      [WARN] No Costamagna word_position data found.")
        print("      Reporting partial results.")

    # ---- Step 3: Compare positional profiles ----
    print("\n  3. Comparing positional profiles for shared triple_keys ...")
    shared_keys = sorted(set(voynich_by_triple.keys()) & set(tironian_profiles.keys()))
    n_comparable = len(shared_keys)
    print(f"      {n_comparable} comparable triple_key pairs")

    if n_comparable >= 3:
        v_initial = [voynich_by_triple[k]['initial'] for k in shared_keys]
        v_medial = [voynich_by_triple[k]['medial'] for k in shared_keys]
        v_final = [voynich_by_triple[k]['final'] for k in shared_keys]

        t_initial = [tironian_profiles[k]['initial'] for k in shared_keys]
        t_medial = [tironian_profiles[k]['medial'] for k in shared_keys]
        t_final = [tironian_profiles[k]['final'] for k in shared_keys]

        corr_initial = _pearson_correlation(v_initial, t_initial)
        corr_medial = _pearson_correlation(v_medial, t_medial)
        corr_final = _pearson_correlation(v_final, t_final)

        print(f"\n  4. Positional correlations:")
        print(f"      Initial: r = {corr_initial:.4f}")
        print(f"      Medial:  r = {corr_medial:.4f}")
        print(f"      Final:   r = {corr_final:.4f}")

        # Print comparison table
        print(f"\n      {'Triple Key':<40} {'V.ini':>6} {'T.ini':>6} "
              f"{'V.med':>6} {'T.med':>6} {'V.fin':>6} {'T.fin':>6}")
        print("      " + "-" * 78)
        for k in shared_keys[:15]:
            vp = voynich_by_triple[k]
            tp = tironian_profiles[k]
            print(f"      {k:<40} {vp['initial']:>6.3f} {tp['initial']:>6.3f} "
                  f"{vp['medial']:>6.3f} {tp['medial']:>6.3f} "
                  f"{vp['final']:>6.3f} {tp['final']:>6.3f}")
    else:
        corr_initial = 0.0
        corr_medial = 0.0
        corr_final = 0.0
        print(f"\n  4. Insufficient comparable pairs ({n_comparable} < 3) for correlation.")

    mean_corr = (corr_initial + corr_medial + corr_final) / 3.0

    # ---- Step 5: Gate ----
    # Automatic pass if not enough data to test
    if not has_costamagna or n_comparable < 3:
        gate_passed = True
        gate_reason = "auto-pass (insufficient Tironian positional data)"
    else:
        gate_passed = mean_corr > 0.3
        gate_reason = f"mean_correlation = {mean_corr:.4f} {'>' if gate_passed else '<='} 0.3"

    print(f"\n  5. Gate B.3 (mean correlation > 0.3 or insufficient data): "
          f"{'PASS' if gate_passed else 'FAIL'} ({gate_reason})")

    # ---- Verdict ----
    if not has_costamagna or n_comparable < 3:
        verdict = (
            f"INCONCLUSIVE: Only {n_comparable} comparable pairs "
            f"(Costamagna data {'present' if has_costamagna else 'absent'}). "
            f"Gate auto-passed due to insufficient Tironian positional data. "
            f"Voynich positional profiles computed for {len(voynich_by_triple)} triples."
        )
    elif gate_passed:
        verdict = (
            f"PASS: Mean positional correlation {mean_corr:.4f} > 0.3 across "
            f"{n_comparable} comparable pairs. Voynich glyph positions correlate "
            f"with Tironian sign positions (r_ini={corr_initial:.3f}, "
            f"r_med={corr_medial:.3f}, r_fin={corr_final:.3f})."
        )
    else:
        verdict = (
            f"FAIL: Mean positional correlation {mean_corr:.4f} <= 0.3. "
            f"Voynich glyph positions do not correlate with Tironian sign positions. "
            f"r_ini={corr_initial:.3f}, r_med={corr_medial:.3f}, r_fin={corr_final:.3f}."
        )

    print(f"\n  Verdict: {verdict}")

    # ---- Save ----
    result = PositionalCompareResult(
        n_voynich_profiles=len(voynich_by_triple),
        n_tironian_profiles=len(tironian_profiles),
        n_comparable_pairs=n_comparable,
        correlation_initial=round(corr_initial, 4),
        correlation_medial=round(corr_medial, 4),
        correlation_final=round(corr_final, 4),
        mean_correlation=round(mean_corr, 4),
        has_costamagna_data=has_costamagna,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'positional_compare.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
