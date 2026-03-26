"""
Phase 67, Track 2: Frequency-Matched Triple Resolution
========================================================
For each unresolved triple, restrict the candidate syllable domain to
Costamagna CV syllables whose frequency rank in Latin reference text
is within ±30 % of the triple's frequency rank in the Voynich corpus.

Validated by LOO on the 12 confirmed triples: does rank-matching
retain the correct syllable?

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
    data/reference/latin/             (Latin reference corpora)
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
        -> results/p67_frequency.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
)
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import syllabify_latin
from voynich.phases.costamagna_csp import _load_costamagna_inventory


# ---------------------------------------------------------------------------
# JSON helpers (standard pattern)
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
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13).

    Only the 12 truly CONFIRMED triples are treated as confirmed.
    LANDSCAPE_CONFIRMED and GENUINELY_AMBIGUOUS are unresolved.
    """
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

    # If no tiering data at all, fall back to treating everything as confirmed
    # (Phase 67 would have nothing to resolve)
    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TripleDomain:
    triple_key: str
    triple_rank: int
    triple_freq: float
    candidates: List[str]
    n_candidates: int
    reduction_frac: float   # fraction of full inventory retained


@dataclass
class FrequencyResult:
    phase: str = "67"
    step: str = "67.2"
    experiment: str = "frequency_matching"
    # Triple frequencies
    n_triples: int = 0
    n_confirmed: int = 0
    n_unresolved: int = 0
    # Syllable frequencies
    n_latin_syllables: int = 0
    n_unique_syllables: int = 0
    n_costamagna_cv: int = 0
    # Rank tolerance
    rank_tolerance: float = 0.30
    # Per-triple constrained domains
    domains: List[TripleDomain] = field(default_factory=list)
    mean_domain_size: float = 0.0
    mean_reduction: float = 0.0
    # LOO validation on confirmed triples
    loo_total: int = 0
    loo_correct: int = 0
    loo_recall: float = 0.0
    loo_details: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_loo_recall: bool = False       # F1: >= 80%
    g2_mean_reduction: bool = False   # F2: > 50%
    g3_narrow_triples: bool = False   # F3: >= 5 with domain < 20
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _compute_triple_frequencies(
    all_tokens: List[str],
    eva_to_triple: Dict[str, str],
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, int]]:
    """Count triple occurrences across the corpus.

    Returns (counts, freqs, ranks) where ranks are 1-indexed (1 = most common).
    """
    counts: Counter = Counter()
    for token in all_tokens:
        triples = token_to_triples(token, eva_to_triple)
        counts.update(triples)

    total = sum(counts.values())
    freqs = {k: v / total for k, v in counts.items()} if total > 0 else {}

    ranked = sorted(freqs.keys(), key=lambda k: -freqs[k])
    ranks = {k: i + 1 for i, k in enumerate(ranked)}

    return dict(counts), freqs, ranks


def _compute_syllable_frequencies(
    ref_corpus,
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, int]]:
    """Syllabify Latin reference text and count syllable occurrences.

    Returns (counts, freqs, ranks).
    """
    counts: Counter = Counter()

    for lang in ['latin']:
        tokens = ref_corpus.get_combined_tokens(lang)
        for word in tokens:
            syls = syllabify_latin(word.lower())
            for syl in syls:
                counts[syl.lower()] += 1

    total = sum(counts.values())
    freqs = {k: v / total for k, v in counts.items()} if total > 0 else {}

    ranked = sorted(freqs.keys(), key=lambda k: -freqs[k])
    ranks = {k: i + 1 for i, k in enumerate(ranked)}

    return dict(counts), freqs, ranks


def _rank_match(
    triple_key: str,
    triple_ranks: Dict[str, int],
    syllable_ranks: Dict[str, int],
    costamagna_cv: Set[str],
    confirmed_values: Set[str],
    n_triples: int,
    n_syllables: int,
    tolerance: float = 0.30,
) -> List[str]:
    """Find Costamagna CV syllables within ±tolerance of the triple's rank fraction."""
    if triple_key not in triple_ranks:
        return sorted(costamagna_cv)

    triple_rank_frac = triple_ranks[triple_key] / n_triples

    min_rank_frac = max(0, triple_rank_frac - tolerance)
    max_rank_frac = min(1, triple_rank_frac + tolerance)

    min_rank = max(1, int(min_rank_frac * n_syllables))
    max_rank = min(n_syllables, int(max_rank_frac * n_syllables) + 1)

    candidates = []
    for syl, rank in syllable_ranks.items():
        if min_rank <= rank <= max_rank and syl in costamagna_cv:
            candidates.append(syl)

    return sorted(candidates)


def _loo_validation(
    confirmed: Dict[str, str],
    triple_ranks: Dict[str, int],
    syllable_ranks: Dict[str, int],
    costamagna_cv: Set[str],
    n_triples: int,
    n_syllables: int,
    tolerance: float = 0.30,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """LOO cross-validation: for each confirmed triple, does rank matching
    include the known syllable in the candidate domain?

    Returns (n_correct, n_total, details_list).
    """
    n_correct = 0
    details = []

    for triple_key, known_syllable in confirmed.items():
        # Exclude this triple's syllable from confirmed values for fairness
        other_values = {v for k, v in confirmed.items() if k != triple_key}

        candidates = _rank_match(
            triple_key, triple_ranks, syllable_ranks,
            costamagna_cv, other_values, n_triples, n_syllables, tolerance,
        )

        hit = known_syllable in candidates
        if hit:
            n_correct += 1

        details.append({
            'triple_key': triple_key,
            'known_syllable': known_syllable,
            'triple_rank': triple_ranks.get(triple_key, -1),
            'n_candidates': len(candidates),
            'hit': hit,
            'candidates_sample': candidates[:10],
        })

    return n_correct, len(confirmed), details


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_freq_match():
    """Track 2: Frequency-matched triple resolution."""
    t0 = time.time()
    rd = str(_results_dir())
    tolerance = 0.30

    print("Phase 67.2 — Frequency-Matched Triple Resolution")
    print("=" * 55)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # --- Triple frequencies ---
    triple_counts, triple_freqs, triple_ranks = _compute_triple_frequencies(
        all_tokens, eva_to_triple)
    n_triples = len(triple_ranks)
    print(f"  Unique triples in corpus: {n_triples}")

    # --- Syllable frequencies ---
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    syl_counts, syl_freqs, syl_ranks = _compute_syllable_frequencies(ref_corpus)
    n_syllables = len(syl_ranks)
    n_total_syls = sum(syl_counts.values())
    print(f"  Latin syllable types: {n_syllables}")
    print(f"  Latin syllable tokens: {n_total_syls}")

    # --- Costamagna CV inventory ---
    cv_set, cvc_set, all_set = _load_costamagna_inventory()
    costamagna_cv = cv_set | cvc_set  # use full inventory
    n_costamagna = len(costamagna_cv)
    print(f"  Costamagna inventory: {n_costamagna}")

    # --- LOO validation on confirmed ---
    print("\n  LOO validation on confirmed triples...")
    confirmed_values = set(confirmed.values())
    loo_correct, loo_total, loo_details = _loo_validation(
        confirmed, triple_ranks, syl_ranks, costamagna_cv,
        n_triples, n_syllables, tolerance,
    )
    loo_recall = loo_correct / loo_total if loo_total > 0 else 0.0
    print(f"  LOO: {loo_correct}/{loo_total} = {loo_recall:.1%}")

    # --- Build constrained domains for unresolved triples ---
    print("\n  Building constrained domains for unresolved triples...")
    domains = []
    for triple_key in sorted(unresolved.keys()):
        candidates = _rank_match(
            triple_key, triple_ranks, syl_ranks,
            costamagna_cv, confirmed_values, n_triples, n_syllables, tolerance,
        )
        n_cand = len(candidates)
        reduction = n_cand / n_costamagna if n_costamagna > 0 else 1.0

        domains.append(TripleDomain(
            triple_key=triple_key,
            triple_rank=triple_ranks.get(triple_key, -1),
            triple_freq=round(triple_freqs.get(triple_key, 0.0), 6),
            candidates=candidates,
            n_candidates=n_cand,
            reduction_frac=round(reduction, 4),
        ))

        current_val = unresolved[triple_key]
        marker = " *" if current_val in candidates else " (T_P15 value excluded)"
        print(f"    {triple_key}: rank {triple_ranks.get(triple_key, '?')}, "
              f"{n_cand} candidates ({reduction:.0%} of inventory){marker}")

    mean_domain = (sum(d.n_candidates for d in domains) / len(domains)
                   if domains else 0)
    mean_reduction = (sum(d.reduction_frac for d in domains) / len(domains)
                      if domains else 1.0)
    n_narrow = sum(1 for d in domains if d.n_candidates < 20)

    # --- Gates ---
    g1 = loo_recall >= 0.80
    g2 = mean_reduction < 0.50   # retained < 50% means > 50% reduction
    g3 = n_narrow >= 5
    gates_passed = sum([g1, g2, g3])

    result = FrequencyResult(
        n_triples=n_triples,
        n_confirmed=len(confirmed),
        n_unresolved=len(unresolved),
        n_latin_syllables=n_total_syls,
        n_unique_syllables=n_syllables,
        n_costamagna_cv=n_costamagna,
        rank_tolerance=tolerance,
        domains=domains,
        mean_domain_size=round(mean_domain, 1),
        mean_reduction=round(mean_reduction, 4),
        loo_total=loo_total,
        loo_correct=loo_correct,
        loo_recall=round(loo_recall, 4),
        loo_details=loo_details,
        g1_loo_recall=g1,
        g2_mean_reduction=g2,
        g3_narrow_triples=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p67_frequency.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  LOO recall:        {loo_recall:.1%} ({'PASS' if g1 else 'FAIL'} >= 80%)")
    print(f"  Mean domain size:  {mean_domain:.1f} / {n_costamagna}")
    print(f"  Mean reduction:    {mean_reduction:.1%} ({'PASS' if g2 else 'FAIL'} < 50%)")
    print(f"  Narrow (< 20):     {n_narrow} ({'PASS' if g3 else 'FAIL'} >= 5)")
    print(f"  Gates: {gates_passed}/3")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
