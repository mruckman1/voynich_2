"""
Phase 62, Investigation 10: Hand-by-Hand CVC Analysis
======================================================
Split CVC-decoded corpus by scribe (1-5).  Per-hand: signal words,
coda distribution, dice/dise ratio, Latin endings.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/phase62_hand_cvc.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    _infer_scribe,
    tokenize_eva_chars,
)
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_corpus_cvc_v2,
    LATIN_ENDINGS,
)
from voynich.phases.coda_markers import get_coda
from voynich.phases.suffix_calibration import SIGNAL_WORDS_SET


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latin_ending_rate(decoded_tokens: List[str]) -> float:
    n_matched = 0
    n_total = 0
    for w in decoded_tokens:
        if not w or w == '?' or len(w) < 3:
            continue
        n_total += 1
        suffix2 = '-' + w[-2:]
        suffix3 = '-' + w[-3:] if len(w) >= 4 else ''
        if suffix2 in LATIN_ENDINGS or suffix3 in LATIN_ENDINGS:
            n_matched += 1
    return n_matched / n_total if n_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HandCVCResult:
    phase: str = "62"
    step: str = "62.10"
    experiment: str = "hand_cvc"
    per_hand: List[Dict] = field(default_factory=list)
    coda_chi2: float = 0.0
    coda_p: float = 1.0
    dice_dise_by_hand: Dict[str, Dict] = field(default_factory=dict)
    n_hands_with_signal: int = 0
    hand_exclusive_signals: Dict[str, List[str]] = field(default_factory=dict)
    # Gates
    g1_coda_differs: bool = False      # coda chi² p < 0.05
    g2_dice_dise: bool = False         # dice/dise ratio differs >1.5x between hands
    g3_exclusive: bool = False         # >= 3 hand-exclusive signal words
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_hand_cvc():
    """Phase 62.10: Hand-by-hand CVC analysis."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 10: Hand-by-Hand CVC")
    print("=" * 70)

    # Load
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)

    # Split by hand
    hand_data = {}  # hand -> {'eva_tokens': [], 'decoded': []}
    for folio_id, page in corpus.pages.items():
        hand = _infer_scribe(folio_id)
        tokens = page.all_tokens
        dec = decode_corpus_cvc_v2(tokens, assignment, eva_to_triple, coda_table)
        if hand not in hand_data:
            hand_data[hand] = {'eva_tokens': [], 'decoded': []}
        hand_data[hand]['eva_tokens'].extend(tokens)
        hand_data[hand]['decoded'].extend(dec)

    per_hand = []
    hand_signal_sets = {}
    hand_coda_dists = {}

    for hand in sorted(hand_data.keys()):
        data = hand_data[hand]
        decoded = [d for d in data['decoded'] if d and d != '?']
        eva_tokens = data['eva_tokens']

        # Signal words
        signal_counter = Counter(d for d in decoded if d in SIGNAL_WORDS_SET)
        signals = set(signal_counter.keys())
        hand_signal_sets[hand] = signals

        # Coda distribution
        coda_counts = Counter()
        for eva_tok in eva_tokens:
            chars = tokenize_eva_chars(eva_tok)
            classified = classify_token_chars_v2(chars, coda_table)
            for role, char in classified:
                if role == 'CODA_MARKER':
                    coda = get_coda(char, coda_table)
                    if coda:
                        coda_counts[coda] += 1
        hand_coda_dists[hand] = coda_counts

        # dice/dise counts
        dice_count = sum(1 for d in decoded if d.startswith('dice'))
        dise_count = sum(1 for d in decoded if d.startswith('dise'))

        # Latin endings
        latin_end = _latin_ending_rate(decoded)

        per_hand.append({
            'hand': hand,
            'n_tokens': len(decoded),
            'n_signal_words': len(signals),
            'top_signals': signal_counter.most_common(10),
            'coda_distribution': dict(coda_counts.most_common()),
            'dice_count': dice_count,
            'dise_count': dise_count,
            'dice_dise_ratio': round(dice_count / dise_count, 2) if dise_count > 0 else float('inf') if dice_count > 0 else 0.0,
            'latin_ending_rate': round(latin_end, 4),
            'mean_decoded_length': round(float(np.mean([len(d) for d in decoded])), 2) if decoded else 0.0,
        })

        print(f"  Hand {hand}: {len(decoded)} tokens, {len(signals)} signal words, "
              f"dice={dice_count} dise={dise_count} latin_end={latin_end:.1%}")

    # Cross-hand chi² on coda distribution
    hands_sorted = sorted(hand_coda_dists.keys())
    all_codas = sorted(set().union(*[set(c.keys()) for c in hand_coda_dists.values()]))
    if len(hands_sorted) >= 2 and len(all_codas) >= 2:
        contingency = np.array([
            [hand_coda_dists[h].get(c, 0) for c in all_codas]
            for h in hands_sorted
        ])
        chi2, coda_p = sp_stats.chi2_contingency(contingency)[:2]
    else:
        chi2, coda_p = 0.0, 1.0

    # dice/dise ratio comparison
    dd_by_hand = {}
    ratios = []
    for ph in per_hand:
        dd_by_hand[str(ph['hand'])] = {
            'dice': ph['dice_count'],
            'dise': ph['dise_count'],
            'ratio': ph['dice_dise_ratio'],
        }
        if ph['dice_count'] > 0 or ph['dise_count'] > 0:
            ratios.append(ph['dice_dise_ratio'])

    # Check if ratios differ by > 1.5x
    dd_differs = False
    if len(ratios) >= 2:
        finite_ratios = [r for r in ratios if r != float('inf')]
        if len(finite_ratios) >= 2:
            dd_differs = max(finite_ratios) > min(finite_ratios) * 1.5

    # Hand-exclusive signal words
    all_signals_union = set().union(*hand_signal_sets.values())
    exclusive = {}
    total_exclusive = 0
    for hand, sigs in hand_signal_sets.items():
        others = set().union(*[s for h, s in hand_signal_sets.items() if h != hand])
        exc = sigs - others
        if exc:
            exclusive[str(hand)] = sorted(exc)
            total_exclusive += len(exc)

    n_hands_with_signal = sum(1 for s in hand_signal_sets.values() if len(s) > 0)

    # Gates
    g1 = coda_p < 0.05
    g2 = dd_differs
    g3 = total_exclusive >= 3
    gates_passed = sum([g1, g2, g3])

    if gates_passed >= 2:
        verdict = "HANDS_DIVERGENT"
    elif gates_passed == 1:
        verdict = "MINOR_HAND_DIFFERENCES"
    else:
        verdict = "HANDS_SIMILAR"

    result = HandCVCResult(
        per_hand=per_hand,
        coda_chi2=round(float(chi2), 2),
        coda_p=round(float(coda_p), 6),
        dice_dise_by_hand=dd_by_hand,
        n_hands_with_signal=n_hands_with_signal,
        hand_exclusive_signals=exclusive,
        g1_coda_differs=g1,
        g2_dice_dise=g2,
        g3_exclusive=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    print(f"\n  Coda chi²={chi2:.1f}, p={coda_p:.4f}")
    print(f"  dice/dise differs > 1.5×: {dd_differs}")
    print(f"  Hand-exclusive signals: {exclusive}")
    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_hand_cvc.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
