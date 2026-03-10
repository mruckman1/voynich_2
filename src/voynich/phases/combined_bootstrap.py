"""
Step 35.6 – Combined Ventris Bootstrap
========================================
Re-run Phase 30's iterative bootstrap under the combined spatial+10K
model.  The higher SIGNAL rate should let more candidates pass Check 2
(signal position >= 50%), breaking the Phase 30 stall.

Dependency chain:
    combined_decode.json       (Step 35.2)
    combined_signal.json       (Step 35.3)
    combined_context.json      (Step 35.5)
    combined_refine.json       (Phase 15 assignment)
    crib_extraction.json       (Phase 28.1)
    bootstrap_loop.json        (Phase 30 baseline)
        → combined_bootstrap.json (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, tokenize_eva_chars
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.dict_calibration import _build_dict_variants


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


def _load_json(rd: str, filename: str) -> Optional[Dict]:
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bootstrap checks (adapted from compound_bootstrap.py)
# ---------------------------------------------------------------------------

def _check_triple_consistency(
    word: str,
    decoded: List[str],
    evas: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 1: Triple assignments implied by this word are consistent."""
    positions = [i for i, w in enumerate(decoded) if w == word]
    if not positions:
        return False, "word not found"

    implied: Dict[str, Set[str]] = {}
    for pos in positions[:20]:
        token = evas[pos]
        chars = tokenize_eva_chars(token)
        for ch in chars:
            triple = eva_to_triple.get(ch)
            if triple and triple in assignment:
                implied.setdefault(triple, set()).add(assignment[triple])

    for triple, syllables in implied.items():
        if len(syllables) > 1:
            return False, f"conflict at {triple}: {syllables}"

    return True, "consistent"


def _check_signal_position(
    word: str,
    decoded: List[str],
    classifications: List[str],
    threshold: float = 0.50,
) -> Tuple[bool, float]:
    """Check 2: What fraction of this word's occurrences are SIGNAL?"""
    positions = [i for i, w in enumerate(decoded) if w == word]
    if not positions:
        return False, 0.0
    n_signal = sum(1 for i in positions if classifications[i] == 'SIGNAL')
    rate = n_signal / len(positions)
    return rate >= threshold, rate


def _check_context_reciprocity(
    word: str,
    decoded: List[str],
    folios: List[str],
    signal_words: List[str],
    min_reciprocal: int = 1,
) -> Tuple[bool, int]:
    """Check 3: Does this word appear in the context of confirmed signal words?"""
    signal_set = set(signal_words)
    n_tokens = len(decoded)
    reciprocal_sws = set()

    for i in range(n_tokens):
        if decoded[i] == word:
            for j in range(max(0, i - 2), min(n_tokens, i + 3)):
                if j != i and decoded[j] in signal_set and folios[j] == folios[i]:
                    reciprocal_sws.add(decoded[j])

    return len(reciprocal_sws) >= min_reciprocal, len(reciprocal_sws)


def _check_typological(
    word: str,
    assignment: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 4: Is the assignment typologically consistent?"""
    for triple, syllable in assignment.items():
        if len(syllable) < 1:
            continue
        first = syllable[0]
        if first not in 'abcdefghijklmnopqrstuvwxyz':
            return False, f"invalid char in {syllable}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_combined_bootstrap() -> None:
    """Step 35.6: Combined spatial+10K Ventris bootstrap."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 35.6: Combined Ventris Bootstrap")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")
    cd = _load_json(rd, 'combined_decode.json')
    cs = _load_json(rd, 'combined_signal.json')
    cc = _load_json(rd, 'combined_context.json')
    p30 = _load_json(rd, 'bootstrap_loop.json')

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    decoded = cd['token_decoded']
    classifications = cs['token_classifications']
    evas = cd['token_evas']
    folios = cd['token_folios']
    n_tokens = cd['n_tokens']

    # Existing signal words
    signal_words = [ws['word'] for ws in cs['word_signals']
                    if ws.get('is_genuine', False)]
    confirmed_words = list(signal_words)

    # Candidates from context analysis
    context_candidates = [c['word'] for c in cc.get('new_crib_candidates', [])]

    # Phase 30 baseline
    p30_n_accepted = 0
    if p30:
        p30_n_accepted = p30.get('n_total_accepted', 0)

    # Build 10K word set for dict hit check
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    variants = _build_dict_variants(base_words, ref_corpus, [10000])
    dict_10k = variants[0][1]

    print(f"     {len(signal_words)} signal words, {len(context_candidates)} context candidates")

    # ── 2. Build candidate pool ──
    print("\n  2. Building candidate pool ...")
    word_counts = Counter(decoded)
    candidate_pool = set()

    for c in context_candidates:
        if c not in set(signal_words):
            candidate_pool.add(c)

    # Also add frequent dict-hit words not yet confirmed
    for word, count in word_counts.most_common(100):
        if word in dict_10k and word not in set(signal_words) and count >= 10:
            candidate_pool.add(word)

    candidates = sorted(candidate_pool)
    print(f"     {len(candidates)} unique candidates to test")

    # ── 3. Bootstrap iterations ──
    print("\n  3. Running bootstrap iterations ...")
    max_iterations = 5
    iteration_results = []
    all_accepted = []

    for iteration in range(max_iterations):
        print(f"\n     --- Iteration {iteration + 1} ---")
        accepted_this_round = []
        deferred = []
        rejected = []

        for word in candidates:
            if word in set(confirmed_words):
                continue

            # Check 1: Triple consistency
            c1_pass, c1_reason = _check_triple_consistency(
                word, decoded, evas, assignment, eva_to_triple,
            )
            if not c1_pass:
                rejected.append({'word': word, 'reason': f'C1: {c1_reason}'})
                continue

            # Check 2: Signal position
            c2_pass, c2_rate = _check_signal_position(
                word, decoded, classifications,
            )
            if not c2_pass:
                deferred.append({
                    'word': word, 'reason': f'C2: signal_rate={c2_rate:.2f}',
                    'signal_rate': round(c2_rate, 4),
                })
                continue

            # Check 3: Context reciprocity
            c3_pass, c3_count = _check_context_reciprocity(
                word, decoded, folios, confirmed_words,
            )
            if not c3_pass:
                deferred.append({
                    'word': word, 'reason': f'C3: reciprocal_count={c3_count}',
                })
                continue

            # Check 4: Typological
            c4_pass, c4_reason = _check_typological(word, assignment)
            if not c4_pass:
                rejected.append({'word': word, 'reason': f'C4: {c4_reason}'})
                continue

            # All 4 checks passed
            accepted_this_round.append({
                'word': word,
                'signal_rate': round(c2_rate, 4),
                'context_count': c3_count,
                'count': word_counts.get(word, 0),
            })
            confirmed_words.append(word)

        all_accepted.extend(accepted_this_round)

        iteration_results.append({
            'iteration': iteration + 1,
            'n_accepted': len(accepted_this_round),
            'n_deferred': len(deferred),
            'n_rejected': len(rejected),
            'accepted': accepted_this_round,
            'deferred_sample': deferred[:10],
        })

        print(f"     Accepted: {len(accepted_this_round)}, "
              f"Deferred: {len(deferred)}, Rejected: {len(rejected)}")

        for a in accepted_this_round:
            print(f"       + {a['word']:12s} sig_rate={a['signal_rate']:.2f} "
                  f"ctx={a['context_count']} count={a['count']}")

        if len(accepted_this_round) == 0:
            print(f"     Converged at iteration {iteration + 1} (no new words)")
            break

    # ── 4. Summary ──
    print("\n  4. Bootstrap summary ...")
    n_total_accepted = len(all_accepted)
    n_iterations = len(iteration_results)

    if n_total_accepted > p30_n_accepted:
        cascade_shape = "improved"
    elif n_total_accepted == p30_n_accepted:
        cascade_shape = "unchanged"
    else:
        cascade_shape = "degraded"

    print(f"     Total accepted: {n_total_accepted} (Phase 30: {p30_n_accepted})")
    print(f"     Iterations: {n_iterations}")
    print(f"     Cascade shape: {cascade_shape}")
    print(f"     Confirmed vocabulary: {len(confirmed_words)}")

    # ── 5. Save ──
    print("\n  5. Saving combined_bootstrap.json ...")
    output = {
        'n_iterations': n_iterations,
        'n_total_accepted': n_total_accepted,
        'accepted_words': [a['word'] for a in all_accepted],
        'confirmed_vocabulary': confirmed_words,
        'confirmed_vocabulary_size': len(confirmed_words),
        'iteration_results': iteration_results,
        'cascade_shape': cascade_shape,
        'phase30_n_accepted': p30_n_accepted,
        'delta_accepted': n_total_accepted - p30_n_accepted,
        'gate_passed': n_total_accepted > 0,
        'verdict': (f"{n_total_accepted} words accepted in {n_iterations} iterations, "
                    f"cascade={cascade_shape}"),
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(rd, 'combined_bootstrap.json'), 'w') as f:
        json.dump(_convert(output), f, indent=2)

    print(f"\n  Step 35.6 completed in {time.time() - t0:.1f}s")
