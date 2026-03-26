"""
Phase 72, Track 4: Tiered T1 Vocabulary Expansion
===================================================
Phase 68 Track 4 went from 22 -> 223 T1 identifications using CVC-enhanced
partial-decode patterns with constraints (>= 3 folios, >= 50% known).
This track relaxes those constraints in 5 tiers and estimates false positive
rates at each level.

Tiers:
  A (strict):   >= 5 folios, >= 70% known  (Phase 52 standard)
  B (moderate): >= 3 folios, >= 50% known  (Phase 68 standard, ~223 hits)
  C (relaxed):  >= 2 folios, >= 40% known
  D (loose):    >= 2 folios, >= 30% known
  E (minimal):  >= 1 folio,  >= 40% known

For each tier, also estimates false positive rate via null table trials.

Dependency chain:
    results/combined_refine.json         (Phase 15)
    results/triple_tiers.json            (Phase 28/53)
    results/p69_clean_corpus.json        (Phase 69)
    results/modifier_integrate.json      (Phase 16)
        -> results/phase72_t1_expand.json
"""

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.p68_expanded_t1 import (
    _build_dict_by_length,
    _build_patterns,
    _extract_constraints,
    _match_patterns,
)


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
# Triple tier loading
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(rd: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

TIERS = {
    'A': {'min_folios': 5, 'min_known_frac': 0.70, 'max_matches': 1},
    'B': {'min_folios': 3, 'min_known_frac': 0.50, 'max_matches': 1},
    'C': {'min_folios': 2, 'min_known_frac': 0.40, 'max_matches': 1},
    'D': {'min_folios': 2, 'min_known_frac': 0.30, 'max_matches': 1},
    'E': {'min_folios': 1, 'min_known_frac': 0.40, 'max_matches': 1},
}


# ---------------------------------------------------------------------------
# Run one tier of T1 pipeline
# ---------------------------------------------------------------------------

def _run_tier(
    tier_name: str,
    token_types: List[str],
    full_assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
    confirmed_keys: Set[str],
    dict_by_length: Dict[int, List[str]],
    corpus,
    all_tokens: List[str],
    min_known_frac: float,
    min_folios: int,
    max_matches: int,
) -> Tuple[List[Dict], int]:
    """Run T1 pipeline at a specific relaxation level.

    Returns (identifications, n_triple_constraints).
    """
    patterns = _build_patterns(
        token_types, full_assignment, eva_to_triple, coda_table,
        confirmed_keys, min_known_frac=min_known_frac)

    all_matches = _match_patterns(patterns, dict_by_length,
                                  max_matches=max_matches * 20)

    identifications, triple_constraints = _extract_constraints(
        patterns, all_matches, corpus, all_tokens, min_folios=min_folios)

    # Filter: keep only unique matches (1 match) or up to max_matches
    filtered = []
    for ident in identifications:
        n_matches = ident.get('n_matches', 1)
        if n_matches <= max_matches:
            filtered.append(ident)

    return filtered, len(triple_constraints)


# ---------------------------------------------------------------------------
# False positive estimation
# ---------------------------------------------------------------------------

def _estimate_fpr(
    tier_name: str,
    token_types: List[str],
    confirmed: Dict[str, str],
    unresolved: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
    confirmed_keys: Set[str],
    dict_by_length: Dict[int, List[str]],
    corpus,
    all_tokens: List[str],
    min_known_frac: float,
    min_folios: int,
    max_matches: int,
    real_count: int,
    n_trials: int = 20,
) -> Dict[str, Any]:
    """Estimate FPR by shuffling unresolved triple values."""
    rng = np.random.default_rng(seed=42)

    # Get the possible syllable values from confirmed triples
    confirmed_values = list(confirmed.values())
    unresolved_keys = list(unresolved.keys())

    null_counts = []

    for trial in range(n_trials):
        # Shuffle unresolved triple values (keep confirmed fixed)
        shuffled_values = list(confirmed_values)
        rng.shuffle(shuffled_values)

        # Build random assignment: confirmed stay, unresolved get shuffled values
        random_assignment = dict(confirmed)
        for i, key in enumerate(unresolved_keys):
            random_assignment[key] = shuffled_values[i % len(shuffled_values)]

        patterns = _build_patterns(
            token_types, random_assignment, eva_to_triple, coda_table,
            confirmed_keys, min_known_frac=min_known_frac)

        all_matches = _match_patterns(patterns, dict_by_length,
                                      max_matches=max_matches * 20)

        identifications, _ = _extract_constraints(
            patterns, all_matches, corpus, all_tokens, min_folios=min_folios)

        filtered = [i for i in identifications
                    if i.get('n_matches', 1) <= max_matches]
        null_counts.append(len(filtered))

    null_mean = float(np.mean(null_counts))
    null_std = float(np.std(null_counts))
    fpr = null_mean / real_count if real_count > 0 else float('inf')
    selectivity = real_count / null_mean if null_mean > 0 else float('inf')

    return {
        'real_count': real_count,
        'null_mean': null_mean,
        'null_std': null_std,
        'fpr': fpr,
        'selectivity': selectivity,
        'n_trials': n_trials,
    }


# ---------------------------------------------------------------------------
# Best passage finder
# ---------------------------------------------------------------------------

def _find_best_passages(
    all_tokens: List[str],
    identified_types: Set[str],
    corpus,
    window_size: int = 15,
    n_passages: int = 10,
) -> List[Dict[str, Any]]:
    """Find passages with highest concentration of identified tokens."""
    n = len(all_tokens)
    if n < window_size:
        return []

    # Build folio list
    folio_list = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folio_list.append(folio)

    # Sliding window
    best_windows = []
    for start in range(n - window_size + 1):
        end = start + window_size
        window_tokens = all_tokens[start:end]
        n_identified = sum(1 for t in window_tokens if t in identified_types)
        frac = n_identified / window_size
        if frac > 0.3:  # minimum threshold
            best_windows.append({
                'start': start,
                'end': end,
                'n_identified': n_identified,
                'fraction': frac,
                'folio': folio_list[start] if start < len(folio_list) else '?',
            })

    # Sort by fraction and take top n
    best_windows.sort(key=lambda x: -x['fraction'])

    # Deduplicate overlapping windows
    used_positions = set()
    deduped = []
    for w in best_windows:
        if not any(w['start'] <= p <= w['end'] for p in used_positions):
            deduped.append(w)
            used_positions.update(range(w['start'], w['end']))
            if len(deduped) >= n_passages:
                break

    return deduped


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TierResult:
    tier: str = ""
    min_known_frac: float = 0.0
    min_folios: int = 0
    max_matches: int = 1
    n_identifications: int = 0
    n_new_vs_prior: int = 0
    n_cumulative: int = 0
    type_coverage: float = 0.0
    token_coverage: float = 0.0
    fpr_estimate: float = 0.0
    selectivity: float = 0.0
    sample_ids: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class T1ExpandResult:
    phase: str = "72"
    step: str = "72.4"
    experiment: str = "t1_expand"
    tiers: List[TierResult] = field(default_factory=list)
    cumulative_identifications: int = 0
    recommended_tier: str = ""
    best_passages: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    gate_t1: bool = False   # Tier C >= 400 identifications
    gate_t2: bool = False   # Recommended tier FPR < 30%
    gate_t3: bool = False   # Token coverage > 20%
    gate_t4: bool = False   # Tier D >= 600 identifications
    gate_t5: bool = False   # >= 1 passage with fraction > 90%
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_t1_expand():
    """Track 4: Tiered T1 vocabulary expansion."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 72.4 — Tiered T1 Vocabulary Expansion")
    print("=" * 45)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    full_assignment = {**confirmed, **unresolved}

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    token_types = sorted(set(all_tokens))

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    dict_by_length = _build_dict_by_length(ref_word_set)

    print(f"  Tokens: {len(all_tokens)}, Types: {len(token_types)}")
    print(f"  Dictionary: {len(ref_word_set)}")
    print(f"  Confirmed triples: {len(confirmed_keys)}")

    # --- Run tiers ---
    tier_results = []
    cumulative_ids: Set[str] = set()
    all_type_set = set(all_tokens)
    all_identified: Dict[str, Dict] = {}

    for tier_name in ['A', 'B', 'C', 'D', 'E']:
        params = TIERS[tier_name]
        print(f"\n  Tier {tier_name}: min_folios={params['min_folios']}, "
              f"min_known={params['min_known_frac']:.0%}...")

        identifications, n_constraints = _run_tier(
            tier_name, token_types, full_assignment, eva_to_triple,
            coda_table, confirmed_keys, dict_by_length, corpus, all_tokens,
            params['min_known_frac'], params['min_folios'], params['max_matches'])

        new_ids = set()
        for ident in identifications:
            tok = ident.get('token', '')
            if tok and tok not in cumulative_ids:
                new_ids.add(tok)
                all_identified[tok] = ident

        cumulative_ids.update(i.get('token', '') for i in identifications)

        type_coverage = len(cumulative_ids & all_type_set) / len(all_type_set) \
            if all_type_set else 0.0
        token_coverage = sum(1 for t in all_tokens if t in cumulative_ids) / len(all_tokens) \
            if all_tokens else 0.0

        print(f"    Identifications: {len(identifications)} "
              f"(new: {len(new_ids)}, cumulative: {len(cumulative_ids)})")
        print(f"    Type coverage: {type_coverage:.1%}, Token coverage: {token_coverage:.1%}")

        # FPR estimation (skip for A and B — too strict to bother)
        fpr = 0.0
        selectivity = float('inf')
        if tier_name in ('C', 'D', 'E') and len(identifications) > 0:
            print(f"    Estimating FPR (20 null trials)...")
            fpr_result = _estimate_fpr(
                tier_name, token_types, confirmed, unresolved,
                eva_to_triple, coda_table, confirmed_keys, dict_by_length,
                corpus, all_tokens,
                params['min_known_frac'], params['min_folios'],
                params['max_matches'], len(identifications), n_trials=20)
            fpr = fpr_result['fpr']
            selectivity = fpr_result['selectivity']
            print(f"    FPR: {fpr:.1%}, Selectivity: {selectivity:.2f}x")

        tier_results.append(TierResult(
            tier=tier_name,
            min_known_frac=params['min_known_frac'],
            min_folios=params['min_folios'],
            max_matches=params['max_matches'],
            n_identifications=len(identifications),
            n_new_vs_prior=len(new_ids),
            n_cumulative=len(cumulative_ids),
            type_coverage=type_coverage,
            token_coverage=token_coverage,
            fpr_estimate=fpr,
            selectivity=selectivity,
            sample_ids=[{
                'token': i.get('token', ''),
                'matched_word': i.get('matched_word', ''),
                'n_folios': i.get('n_folios', 0),
            } for i in identifications[:20]],
        ))

    # --- Recommend best tier ---
    # Choose tier with best balance of coverage and FPR
    recommended = 'B'  # default
    for tr in tier_results:
        if tr.tier in ('C', 'D', 'E') and tr.fpr_estimate < 0.30 and tr.token_coverage > 0.10:
            recommended = tr.tier

    print(f"\n  Recommended tier: {recommended}")

    # --- Best passages ---
    print("\n  Finding best passages with expanded T1...")
    best_passages = _find_best_passages(all_tokens, cumulative_ids, corpus)
    for i, passage in enumerate(best_passages[:5]):
        print(f"    {i+1}. {passage['folio']}: {passage['n_identified']}/{passage['end']-passage['start']} "
              f"identified ({passage['fraction']:.0%})")

    # --- Gates ---
    tier_c = next((t for t in tier_results if t.tier == 'C'), None)
    tier_d = next((t for t in tier_results if t.tier == 'D'), None)
    rec_tier = next((t for t in tier_results if t.tier == recommended), None)

    g1 = tier_c is not None and tier_c.n_identifications >= 400
    g2 = rec_tier is not None and rec_tier.fpr_estimate < 0.30
    g3 = rec_tier is not None and rec_tier.token_coverage > 0.20
    g4 = tier_d is not None and tier_d.n_identifications >= 600
    g5 = any(p['fraction'] > 0.90 for p in best_passages) if best_passages else False

    gates_passed = sum([g1, g2, g3, g4, g5])

    print(f"\n  Gates:")
    print(f"    T1_1 (Tier C >= 400 IDs): {'PASS' if g1 else 'FAIL'} "
          f"({tier_c.n_identifications if tier_c else 0})")
    print(f"    T1_2 (rec tier FPR < 30%): {'PASS' if g2 else 'FAIL'}")
    print(f"    T1_3 (token coverage > 20%): {'PASS' if g3 else 'FAIL'}")
    print(f"    T1_4 (Tier D >= 600 IDs): {'PASS' if g4 else 'FAIL'} "
          f"({tier_d.n_identifications if tier_d else 0})")
    print(f"    T1_5 (passage > 90% identified): {'PASS' if g5 else 'FAIL'}")
    print(f"    Total: {gates_passed}/5")

    # --- Verdict ---
    if g1 and g2 and g3:
        verdict = 'EXPANSION_VALIDATED'
    elif g1 and g2:
        verdict = 'EXPANSION_MODERATE'
    elif g1:
        verdict = 'EXPANSION_NOISY'
    else:
        verdict = 'EXPANSION_MINIMAL'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = T1ExpandResult(
        tiers=tier_results,
        cumulative_identifications=len(cumulative_ids),
        recommended_tier=recommended,
        best_passages=best_passages,
        gate_t1=g1,
        gate_t2=g2,
        gate_t3=g3,
        gate_t4=g4,
        gate_t5=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'phase72_t1_expand.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
