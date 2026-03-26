"""
Phase 68, Track 2: Within-Token Co-Occurrence Constraints
==========================================================
Count how often confirmed and unresolved triples co-occur within the
same EVA token.  Cross-reference with Latin syllable pair frequencies
to score candidate syllable values for unresolved triples.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
        -> results/p68_within_token.json
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import syllabify_latin
from voynich.phases.coda_markers import CodaTable
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
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
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13)."""
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

    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Common CV syllables
# ---------------------------------------------------------------------------

_COMMON_CV = sorted({
    'ba', 'be', 'bi', 'bo', 'bu',
    'ca', 'ce', 'ci', 'co', 'cu',
    'da', 'de', 'di', 'do', 'du',
    'fa', 'fe', 'fi', 'fo', 'fu',
    'la', 'le', 'li', 'lo', 'lu',
    'ma', 'me', 'mi', 'mo', 'mu',
    'na', 'ne', 'ni', 'no', 'nu',
    'pa', 'pe', 'pi', 'po', 'pu',
    'ra', 're', 'ri', 'ro', 'ru',
    'sa', 'se', 'si', 'so', 'su',
    'ta', 'te', 'ti', 'to', 'tu',
    'va', 've', 'vi', 'vo', 'vu',
})


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class WithinTokenResult:
    phase: str = "68"
    step: str = "68.2"
    experiment: str = "within_token_cooccurrence"
    n_corpus_tokens: int = 0
    n_pairs_counted: int = 0
    n_unresolved_with_data: int = 0
    n_unresolved_scored: int = 0
    # Per-triple scoring
    triple_candidates: Dict[str, str] = field(default_factory=dict)
    triple_details: List[Dict[str, Any]] = field(default_factory=list)
    # LOO validation
    loo_correct: int = 0
    loo_total: int = 0
    loo_accuracy: float = 0.0
    # Gates
    g1_coverage: bool = False       # WT1: >= 10 of 13 triples have data
    g2_clear_winners: bool = False  # WT2: >= 3 triples with clear winner (2x ratio)
    g3_loo: bool = False            # WT3: LOO > 50%
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _build_cooccurrence_matrix(
    all_tokens: List[str],
    eva_to_triple: Dict[str, str],
    confirmed_keys: Set[str],
    unresolved_keys: Set[str],
    coda_table: CodaTable,
) -> Dict[Tuple[str, str, str], int]:
    """Count (confirmed_triple, unresolved_triple, BEFORE|AFTER) within tokens.

    BEFORE means the confirmed triple appears before the unresolved triple
    in the token; AFTER means the confirmed triple appears after.

    Returns dict mapping (confirmed_key, unresolved_key, direction) -> count.
    """
    matrix: Counter = Counter()

    for token in all_tokens:
        eva_chars = tokenize_eva_chars(token)
        if not eva_chars:
            continue

        classified = classify_token_chars_v2(eva_chars, coda_table)

        # Collect syllabic triple keys in order
        syllabic_triples: List[Tuple[str, bool]] = []  # (triple_key, is_confirmed)
        for role, char in classified:
            if role == 'SYLLABIC':
                triple_key = eva_to_triple.get(char, '')
                if not triple_key:
                    continue
                is_conf = triple_key in confirmed_keys
                is_unres = triple_key in unresolved_keys
                if is_conf or is_unres:
                    syllabic_triples.append((triple_key, is_conf))

        # Record all (confirmed, unresolved) pairs
        for i, (tk_i, is_conf_i) in enumerate(syllabic_triples):
            for j, (tk_j, is_conf_j) in enumerate(syllabic_triples):
                if i == j:
                    continue
                # We want pairs where one is confirmed and the other unresolved
                if is_conf_i and not is_conf_j:
                    direction = 'BEFORE' if i < j else 'AFTER'
                    matrix[(tk_i, tk_j, direction)] += 1

    return dict(matrix)


def _build_latin_syllable_bigrams(
    ref_corpus: Any,
) -> Dict[Tuple[str, str], int]:
    """Syllabify all Latin words and count adjacent syllable pairs.

    Returns dict mapping (syl_a, syl_b) -> count.
    """
    bigrams: Counter = Counter()

    latin_tokens = ref_corpus.get_combined_tokens('latin')
    for word in latin_tokens:
        word_lower = word.lower().strip()
        if len(word_lower) < 2:
            continue
        syllables = syllabify_latin(word_lower)
        if len(syllables) < 2:
            continue
        for k in range(len(syllables) - 1):
            pair = (syllables[k].lower(), syllables[k + 1].lower())
            bigrams[pair] += 1

    return dict(bigrams)


def _score_candidates(
    cooc_matrix: Dict[Tuple[str, str, str], int],
    confirmed_assignment: Dict[str, str],
    latin_bigrams: Dict[Tuple[str, str], int],
) -> Dict[str, Dict[str, float]]:
    """Score candidate syllable values for each unresolved triple.

    For each unresolved triple, collect all confirmed co-occurrences.
    For each candidate syllable, compute:
        score = sum over co-occurrences: count * log(latin_pair_freq + 1)

    The direction determines pair order:
        BEFORE: (confirmed_syllable, candidate)
        AFTER:  (candidate, confirmed_syllable)

    Returns dict mapping unresolved_key -> {candidate_syllable: score}.
    """
    # Collect all unresolved keys that appear in the matrix
    unresolved_cooc: Dict[str, List[Tuple[str, str, int]]] = {}
    for (conf_key, unres_key, direction), count in cooc_matrix.items():
        if unres_key not in unresolved_cooc:
            unresolved_cooc[unres_key] = []
        unresolved_cooc[unres_key].append((conf_key, direction, count))

    # Build candidate set
    candidate_syllables = sorted(
        set(confirmed_assignment.values()) | set(_COMMON_CV)
    )

    scores: Dict[str, Dict[str, float]] = {}

    for unres_key, coocs in unresolved_cooc.items():
        candidate_scores: Dict[str, float] = {}

        for candidate in candidate_syllables:
            score = 0.0
            for conf_key, direction, count in coocs:
                conf_syl = confirmed_assignment.get(conf_key, '')
                if not conf_syl:
                    continue
                if direction == 'BEFORE':
                    pair = (conf_syl, candidate)
                else:
                    pair = (candidate, conf_syl)
                freq = latin_bigrams.get(pair, 0)
                score += count * math.log(freq + 1)
            candidate_scores[candidate] = round(score, 4)

        scores[unres_key] = candidate_scores

    return scores


def _loo_validation(
    cooc_matrix: Dict[Tuple[str, str, str], int],
    confirmed: Dict[str, str],
    latin_bigrams: Dict[Tuple[str, str], int],
) -> Tuple[int, int]:
    """Leave-one-out validation on confirmed triples.

    For each confirmed triple, pretend it is unresolved: hide it from the
    confirmed set, score candidates using co-occurrences with the remaining
    confirmed triples, and check if the top candidate matches the true value.

    Returns (n_correct, n_total).
    """
    confirmed_keys = set(confirmed.keys())
    n_correct = 0
    n_total = 0

    for hold_key in sorted(confirmed.keys()):
        true_value = confirmed[hold_key]

        # Build a reduced confirmed set without the held-out triple
        reduced_confirmed = {k: v for k, v in confirmed.items() if k != hold_key}
        reduced_keys = set(reduced_confirmed.keys())

        # Collect co-occurrences where hold_key appears as "unresolved"
        # and the other triple is in the reduced confirmed set
        hold_coocs: List[Tuple[str, str, int]] = []
        for (conf_key, unres_key, direction), count in cooc_matrix.items():
            if unres_key == hold_key and conf_key in reduced_keys:
                hold_coocs.append((conf_key, direction, count))

        if not hold_coocs:
            continue

        n_total += 1

        # Score candidates
        candidate_syllables = sorted(
            set(reduced_confirmed.values()) | set(_COMMON_CV)
        )
        best_syl = ''
        best_score = -1.0

        for candidate in candidate_syllables:
            score = 0.0
            for conf_key, direction, count in hold_coocs:
                conf_syl = reduced_confirmed.get(conf_key, '')
                if not conf_syl:
                    continue
                if direction == 'BEFORE':
                    pair = (conf_syl, candidate)
                else:
                    pair = (candidate, conf_syl)
                freq = latin_bigrams.get(pair, 0)
                score += count * math.log(freq + 1)

            if score > best_score:
                best_score = score
                best_syl = candidate

        if best_syl == true_value:
            n_correct += 1

    return n_correct, n_total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_within_token():
    """Track 2: Within-token co-occurrence constraints."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 68.2 — Within-Token Co-Occurrence Constraints")
    print("=" * 50)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    unresolved_keys = set(unresolved.keys())
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    print(f"  Corpus tokens: {len(all_tokens)}")

    # --- Load Latin reference ---
    print("\n  [68.2] Loading Latin reference corpus...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)

    # --- Step 1: Build co-occurrence matrix ---
    print("  [68.2] Building within-token co-occurrence matrix...")
    cooc_matrix = _build_cooccurrence_matrix(
        all_tokens, eva_to_triple, confirmed_keys, unresolved_keys, coda_table)
    n_pairs = sum(cooc_matrix.values())
    print(f"  [68.2] Co-occurrence pairs counted: {n_pairs}")

    # --- Step 2: Build Latin syllable bigrams ---
    print("  [68.2] Building Latin syllable bigrams...")
    latin_bigrams = _build_latin_syllable_bigrams(ref_corpus)
    print(f"  [68.2] Unique syllable bigrams: {len(latin_bigrams)}")

    # --- Step 3: Score candidates ---
    print("  [68.2] Scoring candidate syllables for unresolved triples...")
    all_scores = _score_candidates(cooc_matrix, confirmed, latin_bigrams)
    n_unresolved_with_data = len(all_scores)
    print(f"  [68.2] Unresolved triples with co-occurrence data: {n_unresolved_with_data}")

    # Pick best candidate per unresolved triple
    triple_candidates: Dict[str, str] = {}
    triple_details: List[Dict[str, Any]] = []
    n_clear_winners = 0

    for unres_key in sorted(unresolved.keys()):
        current_value = unresolved[unres_key]
        cand_scores = all_scores.get(unres_key)

        if not cand_scores:
            triple_details.append({
                'triple_key': unres_key,
                'current_value': current_value,
                'proposed_value': current_value,
                'n_cooccurrences': 0,
                'top_candidates': [],
                'clear_winner': False,
                'changed': False,
            })
            continue

        # Rank candidates
        ranked = sorted(cand_scores.items(), key=lambda x: -x[1])
        top_syl, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Clear winner: top score >= 2x second score (and top_score > 0)
        clear = (top_score > 0 and
                 (second_score <= 0 or top_score >= 2.0 * second_score))
        if clear:
            n_clear_winners += 1

        triple_candidates[unres_key] = top_syl

        # Count total co-occurrences for this triple
        n_cooc = sum(
            count for (ck, uk, d), count in cooc_matrix.items()
            if uk == unres_key
        )

        triple_details.append({
            'triple_key': unres_key,
            'current_value': current_value,
            'proposed_value': top_syl,
            'n_cooccurrences': n_cooc,
            'top_candidates': [
                {'syllable': s, 'score': sc} for s, sc in ranked[:5]
            ],
            'clear_winner': clear,
            'changed': top_syl != current_value,
        })

        status = '*' if top_syl != current_value else ' '
        winner_tag = ' [CLEAR]' if clear else ''
        print(f"    {status} {unres_key}: {current_value} -> {top_syl} "
              f"(score={top_score:.2f}, coocs={n_cooc}){winner_tag}")

    n_scored = len(triple_candidates)

    # --- Step 4: LOO validation ---
    print("\n  [68.2] Running leave-one-out validation on confirmed triples...")
    loo_correct, loo_total = _loo_validation(cooc_matrix, confirmed, latin_bigrams)
    loo_accuracy = loo_correct / loo_total if loo_total > 0 else 0.0
    print(f"  [68.2] LOO: {loo_correct}/{loo_total} correct ({loo_accuracy:.1%})")

    # --- Gates ---
    n_unresolved_total = len(unresolved)
    g1 = n_unresolved_with_data >= min(10, n_unresolved_total)
    g2 = n_clear_winners >= 3
    g3 = loo_accuracy > 0.50
    gates_passed = sum([g1, g2, g3])

    result = WithinTokenResult(
        n_corpus_tokens=len(all_tokens),
        n_pairs_counted=n_pairs,
        n_unresolved_with_data=n_unresolved_with_data,
        n_unresolved_scored=n_scored,
        triple_candidates=triple_candidates,
        triple_details=triple_details,
        loo_correct=loo_correct,
        loo_total=loo_total,
        loo_accuracy=round(loo_accuracy, 4),
        g1_coverage=g1,
        g2_clear_winners=g2,
        g3_loo=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p68_within_token.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Co-occurrence pairs:  {n_pairs}")
    print(f"  Unresolved w/ data:   {n_unresolved_with_data}/{n_unresolved_total} "
          f"({'PASS' if g1 else 'FAIL'} >= {min(10, n_unresolved_total)})")
    print(f"  Clear winners:        {n_clear_winners} "
          f"({'PASS' if g2 else 'FAIL'} >= 3)")
    print(f"  LOO accuracy:         {loo_correct}/{loo_total} ({loo_accuracy:.1%}) "
          f"({'PASS' if g3 else 'FAIL'} > 50%)")
    print(f"  Candidates proposed:  {n_scored}")
    print(f"  Gates: {gates_passed}/3")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
