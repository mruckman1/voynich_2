"""
Phase 78: CVC T1 Permutation Validation
=========================================
Validate that Phase 75's 316 CVC T1 identifications are table-specific
by running 1,000 random CV assignment tables through the same T1 pipeline.

The test is identical to Phase 75's T1 pipeline except the CV assignment
values are randomised.  CVC coda assignments (hook→n, sigmoid→s,
vertical→t, connector→null, descender→null) are FIXED across all trials.
Only the 25 CV triple→syllable mappings are randomised.

Dependency chain:
    results/combined_refine.json       (Phase 15 best_assignment)
    results/triple_tiers.json          (Phase 28/53 confirmed triples)
    results/p75_t1.json                (Phase 75 baseline: 316 IDs)
        -> results/phase78_cvc_t1_perm.json
"""

import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.coda_markers import CodaTable, get_coda
from voynich.phases.corrected_coda import classify_token_chars_v2
from voynich.phases.p68_expanded_t1 import _build_dict_by_length
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
# Pattern template — assignment-independent structural skeleton
# ---------------------------------------------------------------------------

# Position types in a template
POS_CODA = 'CODA'        # fixed coda character
POS_HIGH = 'HIGH'        # from confirmed triple — filled per assignment
POS_LOW  = 'LOW'         # unresolved triple — always wildcard [a-z]


@dataclass
class TemplatePosition:
    """One character position in a pattern template."""
    kind: str                     # POS_CODA, POS_HIGH, POS_LOW
    fixed_char: str = ''          # for CODA: the coda character
    triple_key: str = ''          # for HIGH/LOW: which triple
    offset_in_syllable: int = 0   # for HIGH: 0-based offset within syllable


@dataclass
class PatternTemplate:
    """Precomputed structural skeleton for one token type."""
    token: str
    positions: List[TemplatePosition] = field(default_factory=list)
    target_len: int = 0
    n_known: int = 0      # HIGH + CODA count
    n_total: int = 0
    known_frac: float = 0.0
    has_high: bool = False  # True if any HIGH positions exist


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CVCT1PermResult:
    phase: str = "78"
    step: str = "78.1"
    experiment: str = "cvc_t1_permutation"
    n_trials: int = 0
    real_n_ids: int = 0
    real_n_distinct: int = 0
    # Null distribution
    null_mean: float = 0.0
    null_std: float = 0.0
    null_median: float = 0.0
    null_max: int = 0
    null_min: int = 0
    # Significance
    p_value: float = 0.0
    z_score: float = 0.0
    # Per-word specificity
    n_real_words: int = 0
    n_unique_to_real: int = 0
    mean_word_specificity: float = 0.0
    top_specific_words: List[Dict[str, Any]] = field(default_factory=list)
    top_common_words: List[Dict[str, Any]] = field(default_factory=list)
    # Distribution summary
    percentiles: Dict[str, float] = field(default_factory=dict)
    # Comparison with CV result
    cv_comparison: Dict[str, Any] = field(default_factory=dict)
    # Gates
    gate_pvalue: bool = False
    gate_zscore: bool = False
    gate_specificity: bool = False
    gates_passed: int = 0
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Template building (once)
# ---------------------------------------------------------------------------

def _build_pattern_templates(
    token_types: List[str],
    real_assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    confirmed_keys: Set[str],
    min_known_frac: float = 0.50,
) -> List[PatternTemplate]:
    """Precompute structural pattern templates.

    Uses the real assignment only to determine syllable LENGTHS at each
    triple (needed to know how many character positions a triple produces).
    The actual syllable characters are NOT baked in — they are filled
    per trial from the random assignment.

    NOTE: We need the real assignment to determine how many decoded
    characters each triple produces (syllable length, typically 2 for CV).
    For random assignments with different syllable lengths, the template
    structure would change.  However, all syllables in the Latin CV pool
    are either 1 char (pure vowel) or 2 chars (CV).  To keep the null
    test fair, we fix the template structure from the real assignment
    and require random syllables to have the same length per triple.
    Since the CV pool contains both 1-char and 2-char syllables, we
    record the expected length per triple from the real assignment.
    """
    templates: List[PatternTemplate] = []

    for token in token_types:
        eva_chars = tokenize_eva_chars(token)
        if not eva_chars:
            continue

        classified = classify_token_chars_v2(eva_chars, coda_table)
        positions: List[TemplatePosition] = []
        has_high = False

        for role, char in classified:
            if role == 'SYLLABIC':
                triple_key = eva_to_triple.get(char, '')
                syllable = real_assignment.get(triple_key, '?') if triple_key else '?'
                is_confirmed = triple_key in confirmed_keys

                for offset, c in enumerate(syllable):
                    if is_confirmed:
                        positions.append(TemplatePosition(
                            kind=POS_HIGH,
                            triple_key=triple_key,
                            offset_in_syllable=offset,
                        ))
                        has_high = True
                    else:
                        positions.append(TemplatePosition(kind=POS_LOW))

            elif role == 'CODA_MARKER':
                coda = get_coda(char, coda_table)
                if coda:
                    positions.append(TemplatePosition(
                        kind=POS_CODA,
                        fixed_char=coda,
                    ))

        if not positions:
            continue

        n_known = sum(1 for p in positions if p.kind in (POS_HIGH, POS_CODA))
        n_total = len(positions)
        known_frac = n_known / n_total

        if known_frac < min_known_frac:
            continue

        templates.append(PatternTemplate(
            token=token,
            positions=positions,
            target_len=n_total,
            n_known=n_known,
            n_total=n_total,
            known_frac=known_frac,
            has_high=has_high,
        ))

    return templates


# ---------------------------------------------------------------------------
# Template filling + matching
# ---------------------------------------------------------------------------

def _build_match_mask(
    template: PatternTemplate,
    assignment: Dict[str, str],
) -> Optional[List[Tuple[int, str]]]:
    """Build a match mask: list of (position, required_char) for non-wildcard positions.

    Returns None if any HIGH position would produce an out-of-range
    offset (assignment syllable shorter than expected).
    Wildcard (LOW) positions are omitted — they match any lowercase letter.
    """
    mask: List[Tuple[int, str]] = []
    for i, pos in enumerate(template.positions):
        if pos.kind == POS_CODA:
            mask.append((i, pos.fixed_char))
        elif pos.kind == POS_HIGH:
            syllable = assignment.get(pos.triple_key, '')
            if pos.offset_in_syllable >= len(syllable):
                return None
            mask.append((i, syllable[pos.offset_in_syllable]))
        # LOW positions: wildcard, skip
    return mask


def _match_words_fast(
    mask: List[Tuple[int, str]],
    target_len: int,
    words: List[str],
    max_matches: int = 20,
) -> List[str]:
    """Match words using direct character comparison (no regex).

    Equivalent to regex matching with ^literal[a-z]literal...$ but
    ~10x faster because it avoids regex compilation overhead.
    """
    matches: List[str] = []
    for word in words:
        if len(word) != target_len:
            continue
        ok = True
        for pos_idx, required_char in mask:
            if word[pos_idx] != required_char:
                ok = False
                break
        if ok:
            matches.append(word)
            if len(matches) >= max_matches:
                break
    return matches


def _count_identifications_fast(
    templates: List[PatternTemplate],
    assignment: Dict[str, str],
    dict_by_length: Dict[int, List[str]],
    token_n_folios: Dict[str, int],
    static_cache: Dict[str, List[str]],
    min_folios: int = 3,
    max_matches: int = 20,
) -> Tuple[int, Set[str]]:
    """Run the T1 pipeline for one assignment table.

    Returns (n_identifications, set_of_identified_words).

    static_cache: pre-matched results for templates with no HIGH positions
    (their regex is identical regardless of assignment).
    """
    n_ids = 0
    identified_words: Set[str] = set()

    for tmpl in templates:
        # Use cached matches for static templates
        if not tmpl.has_high:
            matches = static_cache.get(tmpl.token, [])
        else:
            mask = _build_match_mask(tmpl, assignment)
            if mask is None:
                continue

            words = dict_by_length.get(tmpl.target_len, [])
            if not words:
                continue

            matches = _match_words_fast(mask, tmpl.target_len, words,
                                        max_matches)

        # Check: exactly 1 match + sufficient folios
        if len(matches) == 1:
            if token_n_folios.get(tmpl.token, 0) >= min_folios:
                n_ids += 1
                identified_words.add(matches[0])

    return n_ids, identified_words


def _build_static_cache(
    templates: List[PatternTemplate],
    real_assignment: Dict[str, str],
    dict_by_length: Dict[int, List[str]],
    max_matches: int = 20,
) -> Dict[str, List[str]]:
    """Pre-match templates with no HIGH positions (static across trials)."""
    cache: Dict[str, List[str]] = {}

    for tmpl in templates:
        if tmpl.has_high:
            continue

        mask = _build_match_mask(tmpl, real_assignment)
        if mask is None:
            cache[tmpl.token] = []
            continue

        words = dict_by_length.get(tmpl.target_len, [])
        if not words:
            cache[tmpl.token] = []
            continue

        cache[tmpl.token] = _match_words_fast(mask, tmpl.target_len, words,
                                               max_matches)

    return cache


def _build_token_folio_counts(corpus, all_tokens: List[str]) -> Dict[str, int]:
    """Precompute number of distinct folios per token type."""
    token_folios: Dict[str, Set[str]] = {}
    for page_id, page in corpus.pages.items():
        for token in page.all_tokens:
            tok_str = token if isinstance(token, str) else str(token)
            if tok_str not in token_folios:
                token_folios[tok_str] = set()
            token_folios[tok_str].add(page_id)
    return {tok: len(folios) for tok, folios in token_folios.items()}


# ---------------------------------------------------------------------------
# Random assignment generation
# ---------------------------------------------------------------------------

def _generate_random_assignment(
    triple_keys: List[str],
    cv_pool: List[str],
    rng: np.random.Generator,
    syllable_lengths: Dict[str, int],
) -> Optional[Dict[str, str]]:
    """Generate one random CV assignment table.

    Each of the 25 triples gets a syllable sampled uniformly with
    replacement from the CV pool.  Only syllables matching the expected
    length for that triple are eligible (to preserve template structure).

    Returns None if no valid syllable exists for some triple (shouldn't
    happen with the Latin CV pool).
    """
    # Pre-group pool by length
    pool_by_len: Dict[int, List[str]] = {}
    for s in cv_pool:
        sl = len(s)
        if sl not in pool_by_len:
            pool_by_len[sl] = []
        pool_by_len[sl].append(s)

    assignment: Dict[str, str] = {}
    for tk in triple_keys:
        expected_len = syllable_lengths.get(tk, 2)
        candidates = pool_by_len.get(expected_len, [])
        if not candidates:
            return None
        assignment[tk] = candidates[rng.integers(len(candidates))]

    return assignment


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_cvc_t1_perm(n_trials: int = 1000) -> CVCT1PermResult:
    """Phase 78: CVC T1 Permutation Validation."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 78 — CVC T1 Permutation Validation")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Step 1: Load shared data
    # ------------------------------------------------------------------
    print("\n  Step 1: Loading data...")

    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    real_assignment = {**confirmed, **unresolved}
    triple_keys = sorted(real_assignment.keys())

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = _build_3coda_table()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    token_types = sorted(set(all_tokens))

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    dict_by_length = _build_dict_by_length(ref_word_set)

    cv_pool = build_cv_syllable_table('latin')

    # Load Phase 75 baseline for comparison
    p75_data = _safe_load(os.path.join(rd, 'p75_t1.json'))
    p75_ids = p75_data.get('identifications', [])
    p75_n_ids = len(p75_ids)
    p75_words = set(i['matched_word'] for i in p75_ids if 'matched_word' in i)

    print(f"    Triples: {len(triple_keys)} ({len(confirmed_keys)} confirmed)")
    print(f"    Token types: {len(token_types)}")
    print(f"    Dictionary: {len(ref_word_set)}")
    print(f"    CV pool: {len(cv_pool)} syllables")
    print(f"    Phase 75 baseline: {p75_n_ids} identifications, "
          f"{len(p75_words)} distinct words")

    # Record syllable lengths from real assignment (for template consistency)
    syllable_lengths = {tk: len(syl) for tk, syl in real_assignment.items()}

    # ------------------------------------------------------------------
    # Step 2: Precompute pattern templates
    # ------------------------------------------------------------------
    print("\n  Step 2: Building pattern templates...")

    templates = _build_pattern_templates(
        token_types, real_assignment, eva_to_triple, coda_table,
        confirmed_keys, min_known_frac=0.50,
    )

    n_static = sum(1 for t in templates if not t.has_high)
    n_dynamic = sum(1 for t in templates if t.has_high)
    print(f"    Templates: {len(templates)} "
          f"({n_static} static, {n_dynamic} dynamic)")

    # Precompute folio counts and static cache
    token_n_folios = _build_token_folio_counts(corpus, all_tokens)
    static_cache = _build_static_cache(
        templates, real_assignment, dict_by_length)

    # ------------------------------------------------------------------
    # Step 3: Sanity check — reproduce Phase 75 result
    # ------------------------------------------------------------------
    print("\n  Step 3: Sanity check (real assignment)...")

    real_n_ids, real_words = _count_identifications_fast(
        templates, real_assignment, dict_by_length,
        token_n_folios, static_cache,
    )

    print(f"    Template pipeline: {real_n_ids} IDs, "
          f"{len(real_words)} distinct words")
    print(f"    Phase 75 result:   {p75_n_ids} IDs, "
          f"{len(p75_words)} distinct words")

    if real_n_ids != p75_n_ids:
        print(f"    WARNING: Template count ({real_n_ids}) != "
              f"Phase 75 count ({p75_n_ids})")
        print(f"    Difference: {real_n_ids - p75_n_ids}")
        print(f"    Using template count as baseline for permutation test.")

    # Use the template-derived count as the true baseline
    baseline_n_ids = real_n_ids
    baseline_words = real_words

    # ------------------------------------------------------------------
    # Step 4: Run null trials
    # ------------------------------------------------------------------
    print(f"\n  Step 4: Running {n_trials} null trials...")

    rng = np.random.default_rng(seed=42)
    null_counts: List[int] = []
    null_distinct: List[int] = []
    word_trial_counts: Counter = Counter()

    for trial in range(n_trials):
        rand_assignment = _generate_random_assignment(
            triple_keys, cv_pool, rng, syllable_lengths,
        )
        if rand_assignment is None:
            null_counts.append(0)
            null_distinct.append(0)
            continue

        n_ids, words = _count_identifications_fast(
            templates, rand_assignment, dict_by_length,
            token_n_folios, static_cache,
        )
        null_counts.append(n_ids)
        null_distinct.append(len(words))

        for w in words:
            word_trial_counts[w] += 1

        if (trial + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    Trial {trial + 1}/{n_trials}: "
                  f"{n_ids} IDs, {len(words)} distinct "
                  f"[{elapsed:.0f}s elapsed]",
                  flush=True)

    # ------------------------------------------------------------------
    # Step 5: Compute statistics
    # ------------------------------------------------------------------
    print("\n  Step 5: Computing statistics...")

    null_arr = np.array(null_counts, dtype=float)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))
    null_median = float(np.median(null_arr))
    null_max = int(np.max(null_arr))
    null_min = int(np.min(null_arr))

    # Conservative p-value
    n_ge = int(np.sum(null_arr >= baseline_n_ids))
    p_value = (n_ge + 1) / (n_trials + 1)

    # Z-score
    if null_std > 0:
        z_score = (baseline_n_ids - null_mean) / null_std
    else:
        z_score = float('inf') if baseline_n_ids > null_mean else 0.0

    # Per-word specificity
    n_unique_to_real = 0
    word_specs: List[Tuple[str, float]] = []
    for w in baseline_words:
        trial_count = word_trial_counts.get(w, 0)
        frac_trials = trial_count / n_trials
        specificity = 1.0 - frac_trials
        word_specs.append((w, specificity))
        if trial_count == 0:
            n_unique_to_real += 1

    word_specs.sort(key=lambda x: -x[1])
    mean_specificity = (
        np.mean([s for _, s in word_specs]) if word_specs else 0.0
    )

    top_specific = [
        {'word': w, 'specificity': round(s, 4),
         'null_trials': word_trial_counts.get(w, 0)}
        for w, s in word_specs[:20]
    ]
    top_common = [
        {'word': w, 'specificity': round(s, 4),
         'null_trials': word_trial_counts.get(w, 0)}
        for w, s in word_specs[-20:]
    ]

    # Percentiles
    percentiles = {
        '25': float(np.percentile(null_arr, 25)),
        '50': float(np.percentile(null_arr, 50)),
        '75': float(np.percentile(null_arr, 75)),
        '90': float(np.percentile(null_arr, 90)),
        '95': float(np.percentile(null_arr, 95)),
        '99': float(np.percentile(null_arr, 99)),
    }

    # Fraction producing zero
    frac_zero = float(np.mean(null_arr == 0))

    # ------------------------------------------------------------------
    # Step 6: Verdict
    # ------------------------------------------------------------------
    gate_pvalue = p_value < 0.001
    gate_zscore = z_score > 3.0
    gate_specificity = n_unique_to_real > 50
    gates_passed = sum([gate_pvalue, gate_zscore, gate_specificity])

    if gates_passed == 3:
        verdict = 'CVC_T1_VALIDATED'
    elif gates_passed >= 1:
        verdict = 'CVC_T1_SIGNIFICANT'
    else:
        verdict = 'CVC_T1_NOT_SIGNIFICANT'

    # Print summary
    print(f"\n  Results:")
    print(f"    Real table: {baseline_n_ids} IDs "
          f"({len(baseline_words)} distinct)")
    print(f"    Null distribution: mean={null_mean:.1f} ± {null_std:.1f}, "
          f"median={null_median:.0f}, max={null_max}")
    print(f"    p-value: {p_value:.6f}")
    print(f"    z-score: {z_score:.2f}")
    print(f"    Unique to real: {n_unique_to_real}/{len(baseline_words)} words")
    print(f"    Mean word specificity: {mean_specificity:.4f}")
    print(f"    Frac null=0: {frac_zero:.3f}")
    print(f"\n  Gates:")
    print(f"    G1 p < 0.001:          {'PASS' if gate_pvalue else 'FAIL'} "
          f"(p={p_value:.6f})")
    print(f"    G2 z > 3.0:            {'PASS' if gate_zscore else 'FAIL'} "
          f"(z={z_score:.2f})")
    print(f"    G3 unique_words > 50:  {'PASS' if gate_specificity else 'FAIL'} "
          f"(n={n_unique_to_real})")
    print(f"\n  Verdict: {verdict} ({gates_passed}/3)")

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    result = CVCT1PermResult(
        n_trials=n_trials,
        real_n_ids=baseline_n_ids,
        real_n_distinct=len(baseline_words),
        null_mean=round(null_mean, 2),
        null_std=round(null_std, 2),
        null_median=round(null_median, 1),
        null_max=null_max,
        null_min=null_min,
        p_value=round(p_value, 6),
        z_score=round(z_score, 2),
        n_real_words=len(baseline_words),
        n_unique_to_real=n_unique_to_real,
        mean_word_specificity=round(float(mean_specificity), 4),
        top_specific_words=top_specific,
        top_common_words=top_common,
        percentiles={k: round(v, 1) for k, v in percentiles.items()},
        cv_comparison={
            'cv_real_ids': 22,
            'cv_p': 0.009,
            'cvc_real_ids': baseline_n_ids,
            'cvc_p': round(p_value, 6),
            'cvc_null_mean': round(null_mean, 2),
            'cvc_null_std': round(null_std, 2),
            'cvc_frac_zero': round(frac_zero, 3),
        },
        gate_pvalue=gate_pvalue,
        gate_zscore=gate_zscore,
        gate_specificity=gate_specificity,
        gates_passed=gates_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'phase78_cvc_t1_perm.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
