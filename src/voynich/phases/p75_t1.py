"""
Phase 75, Track 3: Corrected T1 Identification (3-Coda Model)
=============================================================
Re-run Phase 73 Track 3's T1 pipeline with the 3-coda decode model
(connector→null AND descender→null). Tokens containing connectors or
descenders now decode to shorter strings, changing wildcard patterns
and potentially enabling new matches.

Compare stability: how many of Phase 73's T1 IDs survive the correction?

Dependency chain:
    results/p75_redecode.json          (Step 0)
    results/combined_refine.json       (Phase 15)
    results/triple_tiers.json          (Phase 28/53)
    results/p73_t1.json                (Phase 73, for stability comparison)
        -> results/p75_t1.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import build_coda_table_v2
from voynich.phases.p68_expanded_t1 import (
    _build_dict_by_length,
    _build_patterns,
    _extract_constraints,
    _aggregate_constraints,
    _match_patterns,
)
from voynich.phases.p69_clean_validation import _get_confirmed_and_unresolved
from voynich.phases.p75_redecode import _build_3coda_table


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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorrectedT1Result:
    phase: str = "75"
    step: str = "75.3"
    experiment: str = "t1_3coda"
    # Pipeline stats
    n_token_types: int = 0
    n_patterns_built: int = 0
    n_unique_matches: int = 0
    n_identifications: int = 0
    identifications: List[Dict[str, Any]] = field(default_factory=list)
    triple_candidates: Dict[str, str] = field(default_factory=dict)
    n_triples_constrained: int = 0
    mean_consistency: float = 0.0
    # Stability comparison (against Phase 73)
    old_n_identifications: int = 0
    n_stable: int = 0
    n_lost: int = 0
    n_gained: int = 0
    stability_fraction: float = 0.0
    changed_words: List[Dict[str, str]] = field(default_factory=list)
    lost_sample: List[Dict[str, str]] = field(default_factory=list)
    gained_sample: List[Dict[str, str]] = field(default_factory=list)
    # Gates
    gate_t1: bool = False  # >= 180 stable (80% of old)
    gate_t2: bool = False  # >= 20 new IDs
    gate_t3: bool = False  # Total >= 220
    gates_passed: int = 0
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_t1_3coda() -> CorrectedT1Result:
    """Track 3: Re-run T1 pipeline with 3-coda decode (connector→null, descender→null)."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 75.3 — Corrected T1 Identification (3-Coda Model)")
    print("=" * 58)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    full_assignment = {**confirmed, **unresolved}

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table_3 = _build_3coda_table()  # THE KEY CHANGE: both connector and descender → null

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    dict_by_length = _build_dict_by_length(ref_word_set)

    print(f"  Confirmed: {len(confirmed)}, Unresolved: {len(unresolved)}")
    print(f"  Dictionary: {len(ref_word_set)}")

    # --- Load Phase 73 T1 IDs for comparison ---
    old_t1_data = _safe_load(os.path.join(rd, 'p73_t1.json'))
    old_ids = old_t1_data.get('identifications', [])
    old_n = len(old_ids)
    old_map = {i['token']: i['matched_word'] for i in old_ids if 'token' in i}
    print(f"  Old T1 identifications (Phase 73): {old_n}")

    # --- Step 1: Build patterns with 3-coda table ---
    token_types = sorted(set(all_tokens))
    n_token_types = len(token_types)
    print(f"\n  Token types: {n_token_types}")

    print("  Building wildcard patterns (3-coda model)...")
    patterns = _build_patterns(
        token_types, full_assignment, eva_to_triple, coda_table_3,
        confirmed_keys, min_known_frac=0.50)
    n_patterns = len(patterns)
    print(f"  Patterns built: {n_patterns}")

    # --- Step 2: Match ---
    print("  Matching against dictionary...")
    all_matches = _match_patterns(patterns, dict_by_length, max_matches=20)
    n_unique = sum(1 for m in all_matches if len(m) == 1)
    print(f"  Unique matches: {n_unique}")

    # --- Step 3: Extract ---
    print("  Extracting identifications...")
    identifications, triple_constraints = _extract_constraints(
        patterns, all_matches, corpus, all_tokens, min_folios=3)
    n_identifications = len(identifications)
    print(f"  Identifications: {n_identifications}")

    # --- Step 4: Aggregate constraints ---
    print("  Aggregating constraints...")
    triple_candidates, triple_details, mean_consistency = _aggregate_constraints(
        triple_constraints, patterns, full_assignment, eva_to_triple,
        coda_table_3, confirmed_keys)

    # --- Stability comparison against Phase 73 ---
    new_map = {i['token']: i['matched_word'] for i in identifications if 'token' in i}

    stable = {t: w for t, w in new_map.items() if t in old_map and old_map[t] == w}
    changed = [{'token': t, 'old': old_map[t], 'new': w}
               for t, w in new_map.items() if t in old_map and old_map[t] != w]
    lost = [{'token': t, 'word': w} for t, w in old_map.items() if t not in new_map]
    gained = [{'token': t, 'word': w} for t, w in new_map.items() if t not in old_map]

    n_stable = len(stable)
    n_lost = len(lost)
    n_gained = len(gained)
    stability = n_stable / old_n if old_n > 0 else 0.0

    print(f"\n  Stability (vs Phase 73):")
    print(f"    Stable:  {n_stable} ({100*stability:.1f}%)")
    print(f"    Changed: {len(changed)}")
    print(f"    Lost:    {n_lost}")
    print(f"    Gained:  {n_gained}")
    print(f"    Total:   {n_identifications} (was {old_n})")

    # --- Gates ---
    gate_t1 = n_stable >= 180
    gate_t2 = n_gained >= 20
    gate_t3 = n_identifications >= 220
    gates_passed = sum([gate_t1, gate_t2, gate_t3])

    if gates_passed >= 2:
        verdict = 'T1_STABLE'
    elif gates_passed >= 1:
        verdict = 'T1_PARTIAL'
    else:
        verdict = 'T1_DEGRADED'

    result = CorrectedT1Result(
        n_token_types=n_token_types,
        n_patterns_built=n_patterns,
        n_unique_matches=n_unique,
        n_identifications=n_identifications,
        identifications=identifications,
        triple_candidates=triple_candidates,
        n_triples_constrained=len(triple_candidates),
        mean_consistency=round(mean_consistency, 4),
        old_n_identifications=old_n,
        n_stable=n_stable,
        n_lost=n_lost,
        n_gained=n_gained,
        stability_fraction=round(stability, 4),
        changed_words=changed[:30],
        lost_sample=lost[:30],
        gained_sample=gained[:30],
        gate_t1=gate_t1,
        gate_t2=gate_t2,
        gate_t3=gate_t3,
        gates_passed=gates_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p75_t1.json', asdict(result))
    print(f"\n  Verdict: {verdict} ({gates_passed}/3)")
    print(f"  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
