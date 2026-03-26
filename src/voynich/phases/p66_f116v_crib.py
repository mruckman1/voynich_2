"""
Phase 66, Track 3: f116v Crib Exploitation
============================================
CVC-decodes the Voynichese tokens on f116v ('oror', 'sheey'), finds
closest pharmaceutical words, and compares edit distances against null
controls (random EVA tokens decoded through the same table).

Dependency chain:
    results/combined_refine.json      (Phase 15)
    data/corpus/                      (f116v transcription)
        -> results/p66_f116v_crib.json
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
)
from voynich.phases.p66_validation import _edit_distance
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class F116vCribResult:
    phase: str = "66"
    step: str = "66.3"
    experiment: str = "f116v_crib"
    # Real tokens
    real_tokens: List[Dict] = field(default_factory=list)
    real_mean_min_ed: float = 0.0
    # Null controls
    n_null_tokens: int = 0
    null_mean_min_ed: float = 0.0
    null_std_min_ed: float = 0.0
    # Comparison
    z_score: float = 0.0
    p_value: float = 1.0
    # Contextual info
    latin_context: str = ""
    german_context: str = ""
    # Gates
    f1_close_match: bool = False   # >= 1 token within ED 2
    f2_real_beats_null: bool = False  # real mean ED < null mean ED
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# f116v known tokens
# ---------------------------------------------------------------------------

# Voynichese tokens embedded in Latin/German text on f116v
# Source: marginal_cribs.py, ZL3b-n.txt annotation
F116V_TOKENS = ['oror', 'sheey']

# Known surrounding context from f116v
F116V_LATIN = "valsch vbren"
F116V_GERMAN = "so nim gaf mich o"


def _find_closest_words(
    decoded: str,
    word_sets: List[Tuple[str, Set[str]]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Find closest words by edit distance across multiple word sets."""
    candidates = []
    for source_name, word_set in word_sets:
        for word in word_set:
            if abs(len(word) - len(decoded)) > 3:
                continue
            ed = _edit_distance(decoded, word)
            if ed <= 4:
                candidates.append({
                    'word': word,
                    'edit_distance': ed,
                    'source': source_name,
                })
    candidates.sort(key=lambda x: x['edit_distance'])
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_f116v_crib() -> None:
    """Phase 66, Track 3: f116v Crib Exploitation."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 70)
    print("Phase 66, Track 3: f116v Crib Exploitation")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load dependencies
    # ------------------------------------------------------------------
    print("\n[1] Loading dependencies...")
    corpus = load_corpus(verbose=False)
    cr = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = cr.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    if not assignment:
        print("  ERROR: No assignment table found")
        result = F116vCribResult(
            verdict="ERROR — no assignment table",
            runtime_seconds=round(time.time() - t0, 2),
        )
        _save_json(rd, 'p66_f116v_crib.json', asdict(result))
        return

    # Build word sets for matching
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref.get_combined_tokens('latin'))
    expanded_set, _ = build_expanded_word_set(base_words)
    signal_set = set(SIGNAL_WORDS_51.keys())

    # Pharma vocab (from signal words with pharmaceutical types)
    pharma_set = {w for w, info in SIGNAL_WORDS_51.items()
                  if info.get('type') in ('pharm', 'botanical', 'quality')}

    word_sets = [
        ('signal', signal_set),
        ('pharma', pharma_set),
        ('expanded', expanded_set),
    ]

    print(f"  Assignment: {len(assignment)} triples")
    print(f"  Expanded dict: {len(expanded_set)} words")

    # ------------------------------------------------------------------
    # CVC decode f116v tokens
    # ------------------------------------------------------------------
    print(f"\n[2] CVC decoding f116v tokens: {F116V_TOKENS}")
    real_results = []
    real_min_eds = []

    for token in F116V_TOKENS:
        result = decode_token_cvc_v2(
            token, assignment, eva_to_triple, coda_table)
        decoded_cv = result.decoded_cv
        decoded_cvc = result.decoded_cvc

        closest = _find_closest_words(decoded_cvc, word_sets, top_n=5)
        min_ed = closest[0]['edit_distance'] if closest else 99

        in_expanded = decoded_cvc.lower() in expanded_set
        in_signal = decoded_cvc.lower() in signal_set

        token_result = {
            'eva_token': token,
            'decoded_cv': decoded_cv,
            'decoded_cvc': decoded_cvc,
            'closest_words': closest,
            'min_edit_distance': min_ed,
            'in_expanded_exact': in_expanded,
            'in_signal_exact': in_signal,
        }
        real_results.append(token_result)
        real_min_eds.append(min_ed)

        print(f"  {token} → CV: {decoded_cv}, CVC: {decoded_cvc}")
        if closest:
            print(f"    Closest: {closest[0]['word']} "
                  f"(ED={closest[0]['edit_distance']}, "
                  f"source={closest[0]['source']})")

    # ------------------------------------------------------------------
    # Null control: random EVA tokens
    # ------------------------------------------------------------------
    print(f"\n[3] Generating null controls (100 random tokens)...")
    all_tokens = []
    for page in corpus.pages.values():
        all_tokens.extend(page.all_tokens)

    # Sample tokens with similar length distribution
    target_lengths = [len(t) for t in F116V_TOKENS]
    rng = random.Random(42)

    # Filter to tokens of similar length
    candidate_tokens = [t for t in all_tokens
                        if any(abs(len(t) - tl) <= 2
                               for tl in target_lengths)]
    if len(candidate_tokens) < 100:
        candidate_tokens = all_tokens  # fallback

    null_tokens = rng.sample(candidate_tokens,
                              min(100, len(candidate_tokens)))

    null_min_eds = []
    for token in null_tokens:
        result = decode_token_cvc_v2(
            token, assignment, eva_to_triple, coda_table)
        decoded_cvc = result.decoded_cvc
        if not decoded_cvc or '?' in decoded_cvc:
            continue
        closest = _find_closest_words(decoded_cvc, word_sets, top_n=1)
        min_ed = closest[0]['edit_distance'] if closest else 99
        null_min_eds.append(min_ed)

    print(f"  Null tokens decoded: {len(null_min_eds)}")

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print("\n[4] Comparing real vs null...")
    real_mean_ed = float(np.mean(real_min_eds)) if real_min_eds else 99.0
    null_mean_ed = float(np.mean(null_min_eds)) if null_min_eds else 99.0
    null_std_ed = float(np.std(null_min_eds)) if null_min_eds else 1.0

    z = (null_mean_ed - real_mean_ed) / null_std_ed if null_std_ed > 0 else 0.0

    # One-sided p-value (how often null is as close as real)
    n_null_as_close = sum(1 for ned in null_min_eds if ned <= real_mean_ed)
    p_value = n_null_as_close / len(null_min_eds) if null_min_eds else 1.0

    print(f"  Real mean min ED: {real_mean_ed:.2f}")
    print(f"  Null mean min ED: {null_mean_ed:.2f} ± {null_std_ed:.2f}")
    print(f"  Z-score: {z:.2f}")
    print(f"  P-value: {p_value:.4f}")

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    f1 = any(ed <= 2 for ed in real_min_eds)
    f2 = real_mean_ed < null_mean_ed

    gates = [f1, f2]
    gates_passed = sum(gates)

    if gates_passed == 2:
        verdict = "CRIB_SUPPORTED"
    elif gates_passed == 1:
        verdict = "WEAK_SUPPORT"
    else:
        verdict = "NO_SIGNAL"

    print(f"\n  F1 close match (ED≤2): {'PASS' if f1 else 'FAIL'}")
    print(f"  F2 real < null: {'PASS' if f2 else 'FAIL'}")
    print(f"  Gates: {gates_passed}/2 → {verdict}")
    print(f"\n  NOTE: Only 2 real tokens — statistical power is very low.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    result = F116vCribResult(
        real_tokens=real_results,
        real_mean_min_ed=round(real_mean_ed, 2),
        n_null_tokens=len(null_min_eds),
        null_mean_min_ed=round(null_mean_ed, 2),
        null_std_min_ed=round(null_std_ed, 2),
        z_score=round(z, 2),
        p_value=round(p_value, 4),
        latin_context=F116V_LATIN,
        german_context=F116V_GERMAN,
        f1_close_match=f1,
        f2_real_beats_null=f2,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 1,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    _save_json(rd, 'p66_f116v_crib.json', asdict(result))
    print(f"\n  Saved to results/p66_f116v_crib.json")
    print(f"  Runtime: {result.runtime_seconds}s")
