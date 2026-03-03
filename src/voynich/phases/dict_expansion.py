"""
Phase 15.1 – Dictionary Expansion
==================================
Catalog near-misses from Phase 14 decoded output, build an expanded
medieval Latin dictionary with spelling variants and pharmaceutical
terminology, re-score the decoded corpus, and validate that selectivity
holds.
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    build_expanded_word_set,
    generate_medieval_variants,
    load_reference_corpus,
    MEDIEVAL_SPELLING_RULES,
    PHARMACEUTICAL_VOCABULARY,
)
from voynich.phases.csp_solver import _convert, decode_corpus


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NearMissEntry:
    decoded_token: str
    closest_word: str
    edit_distance: int
    edit_category: str


@dataclass
class DictExpansionResult:
    # 1a: Near-miss catalog
    n_near_misses: int
    near_miss_categories: Dict[str, int]
    near_miss_samples: List[Dict]

    # 1b: Expanded dictionary
    original_dict_size: int
    expanded_dict_size: int
    n_from_variants: int
    n_from_pharma: int
    n_from_inflections: int

    # 1c: Re-scoring
    dict_hit_original: float
    dict_hit_expanded: float
    new_hits: List[str]
    new_hits_by_category: Dict[str, int]

    # 1d: Selectivity validation
    random_dict_hit_original: float
    random_dict_hit_expanded: float
    selectivity_original: float
    selectivity_expanded: float
    expanded_selectivity_ratio: float

    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _categorize_edit(decoded: str, closest: str) -> str:
    """Classify the edit operation between decoded and closest dict word."""
    if len(decoded) != len(closest):
        if len(decoded) > len(closest):
            return 'insertion'
        return 'deletion'
    # Same length — find the substitution
    for dc, cc in zip(decoded, closest):
        if dc != cc:
            vowels = {'a', 'e', 'i', 'o', 'u'}
            if dc in vowels and cc in vowels:
                return 'vowel_interchange'
            voiced = {'b', 'd', 'g'}
            voiceless = {'p', 't', 'c'}
            if (dc in voiced and cc in voiceless) or (dc in voiceless and cc in voiced):
                return 'voicing'
            if dc == 'h' or cc == 'h':
                return 'h_variation'
            return 'consonant_substitution'
    return 'identical'


def _catalog_near_misses(
    decoded_tokens: List[str],
    ref_word_set: set,
    max_distance: int = 2,
) -> List[NearMissEntry]:
    """Find decoded tokens that are edit distance 1-2 from a dictionary word."""
    # Index dictionary by length for efficient lookup
    dict_by_len: Dict[int, List[str]] = {}
    for w in ref_word_set:
        dict_by_len.setdefault(len(w), []).append(w)

    misses: List[NearMissEntry] = []
    seen = set()

    for token in decoded_tokens:
        if token in ref_word_set or token in seen or len(token) < 2:
            continue
        seen.add(token)

        best_dist = max_distance + 1
        best_word = ''

        # Only compare with words of similar length
        for length in range(max(2, len(token) - max_distance),
                           len(token) + max_distance + 1):
            for dict_word in dict_by_len.get(length, []):
                d = _edit_distance(token, dict_word)
                if d < best_dist:
                    best_dist = d
                    best_word = dict_word
                if d == 1:
                    break
            if best_dist == 1:
                break

        if 1 <= best_dist <= max_distance:
            category = _categorize_edit(token, best_word)
            misses.append(NearMissEntry(
                decoded_token=token,
                closest_word=best_word,
                edit_distance=best_dist,
                edit_category=category,
            ))

    return misses


def _compute_random_baseline(
    variables_keys: List[str],
    all_syls: List[str],
    voynich_tokens: List[str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    n_trials: int = 50,
    max_tokens: int = 500,
    seed: int = 42,
) -> float:
    """Compute mean dict_hit rate for random assignments."""
    rng = random.Random(seed)
    random_hits: List[float] = []

    for _ in range(n_trials):
        rand_map = {k: rng.choice(all_syls) for k in variables_keys}
        decoded = decode_corpus(voynich_tokens, rand_map, eva_to_triple, max_tokens)
        hits = sum(1 for w in decoded if w in ref_word_set)
        random_hits.append(hits / len(decoded) if decoded else 0.0)

    return sum(random_hits) / len(random_hits) if random_hits else 0.001


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dict_expansion() -> None:
    """Step 15.1: Dictionary expansion and re-scoring."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 15.1: Medieval Latin Dictionary Expansion")
    print("=" * 70)

    rd = _results_dir()

    # Load Phase 14 results
    fd_path = os.path.join(rd, 'feature_decode.json')
    if not os.path.exists(fd_path):
        print("  [SKIP] feature_decode.json not found — run feature-decode first")
        return

    with open(fd_path) as f:
        fd_data = json.load(f)

    best_assignment = fd_data.get('best_assignment', {})
    if not best_assignment:
        print("  [SKIP] No best assignment in feature_decode.json")
        return

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("  [SKIP] No Language A tokens found")
        return

    eva_to_triple = build_eva_to_triple_lookup()

    # Load reference corpus and build original ref_word_set
    ref_corpus = load_reference_corpus(verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    original_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)

    print(f"  Original dictionary size: {len(original_word_set):,} words")

    # ─── 1a: Catalog near-misses ───
    print("\n  1a: Cataloging near-misses ...")
    decoded = decode_corpus(tokens, best_assignment, eva_to_triple, max_tokens=5000)

    near_misses = _catalog_near_misses(decoded, original_word_set, max_distance=2)
    near_miss_categories: Dict[str, int] = Counter(
        nm.edit_category for nm in near_misses
    )

    print(f"      Near-misses found: {len(near_misses)}")
    for cat, count in sorted(near_miss_categories.items(), key=lambda x: -x[1]):
        print(f"        {cat}: {count}")

    # ─── 1b: Build expanded dictionary ───
    print("\n  1b: Building expanded dictionary ...")
    expanded_word_set, provenance = build_expanded_word_set(original_word_set)

    # Count sources
    n_from_variants = sum(1 for v in provenance.values() if v.startswith('variant:'))
    n_from_pharma = sum(1 for v in provenance.values()
                        if v.startswith('pharma:') and not v.startswith('pharma_variant:'))
    n_from_inflections = sum(1 for v in provenance.values() if v.startswith('inflection:'))
    n_from_pharma_variants = sum(1 for v in provenance.values()
                                  if v.startswith('pharma_variant:'))

    print(f"      Expanded dictionary size: {len(expanded_word_set):,} words")
    print(f"        From spelling variants: {n_from_variants}")
    print(f"        From pharmaceutical vocab: {n_from_pharma}")
    print(f"        From inflected forms: {n_from_inflections}")
    print(f"        From pharma variants: {n_from_pharma_variants}")

    # ─── 1c: Re-score with expanded dictionary ───
    print("\n  1c: Re-scoring decoded corpus ...")
    # Original scoring
    original_hits_list = [w for w in decoded if w in original_word_set]
    dict_hit_original = len(original_hits_list) / len(decoded) if decoded else 0.0

    # Expanded scoring
    expanded_hits_list = [w for w in decoded if w in expanded_word_set]
    dict_hit_expanded = len(expanded_hits_list) / len(decoded) if decoded else 0.0

    # Find new hits (in expanded but not original)
    new_hit_set = set(expanded_hits_list) - set(original_hits_list)
    new_hits = sorted(new_hit_set)

    # Categorize new hits by provenance
    new_hits_by_category: Dict[str, int] = Counter()
    for w in new_hits:
        source = provenance.get(w, 'base')
        cat = source.split(':')[0]
        new_hits_by_category[cat] += 1

    print(f"      Original dict_hit: {dict_hit_original:.1%}")
    print(f"      Expanded dict_hit: {dict_hit_expanded:.1%}")
    print(f"      New hits: {len(new_hits)}")
    if new_hits[:20]:
        print(f"      Sample new hits: {new_hits[:20]}")

    # ─── 1d: Selectivity validation ───
    print("\n  1d: Validating selectivity ...")
    variables_keys = list(best_assignment.keys())
    all_syls = build_cv_syllable_table('latin')

    random_hit_original = _compute_random_baseline(
        variables_keys, all_syls, tokens, eva_to_triple,
        original_word_set, n_trials=50, seed=42,
    )
    random_hit_expanded = _compute_random_baseline(
        variables_keys, all_syls, tokens, eva_to_triple,
        expanded_word_set, n_trials=50, seed=42,
    )

    selectivity_original = dict_hit_original / max(random_hit_original, 0.001)
    selectivity_expanded = dict_hit_expanded / max(random_hit_expanded, 0.001)

    # How much does expansion help real vs random?
    if selectivity_original > 0:
        expanded_selectivity_ratio = selectivity_expanded / selectivity_original
    else:
        expanded_selectivity_ratio = 0.0

    print(f"      Random baseline (original): {random_hit_original:.3%}")
    print(f"      Random baseline (expanded): {random_hit_expanded:.3%}")
    print(f"      Selectivity (original): {selectivity_original:.2f}x")
    print(f"      Selectivity (expanded): {selectivity_expanded:.2f}x")
    print(f"      Selectivity ratio: {expanded_selectivity_ratio:.2f}")

    # ─── Gate ───
    gate_passed = expanded_selectivity_ratio >= 0.9 and dict_hit_expanded > dict_hit_original

    if gate_passed:
        verdict = (
            f"Dictionary expansion successful: {dict_hit_expanded:.1%} dict_hit "
            f"(+{dict_hit_expanded - dict_hit_original:.1%}), "
            f"selectivity ratio {expanded_selectivity_ratio:.2f} (>=0.9). "
            f"{len(new_hits)} new vocabulary items."
        )
    else:
        if expanded_selectivity_ratio < 0.9:
            verdict = (
                f"Expansion too permissive: selectivity ratio "
                f"{expanded_selectivity_ratio:.2f} < 0.9. "
                "Random assignments benefit more than real decoding."
            )
        else:
            verdict = (
                f"No improvement: expanded dict_hit {dict_hit_expanded:.1%} "
                f"vs original {dict_hit_original:.1%}."
            )

    elapsed = time.time() - t0

    result = DictExpansionResult(
        n_near_misses=len(near_misses),
        near_miss_categories=dict(near_miss_categories),
        near_miss_samples=[asdict(nm) for nm in near_misses[:50]],
        original_dict_size=len(original_word_set),
        expanded_dict_size=len(expanded_word_set),
        n_from_variants=n_from_variants,
        n_from_pharma=n_from_pharma,
        n_from_inflections=n_from_inflections + n_from_pharma_variants,
        dict_hit_original=round(dict_hit_original, 4),
        dict_hit_expanded=round(dict_hit_expanded, 4),
        new_hits=new_hits,
        new_hits_by_category=dict(new_hits_by_category),
        random_dict_hit_original=round(random_hit_original, 4),
        random_dict_hit_expanded=round(random_hit_expanded, 4),
        selectivity_original=round(selectivity_original, 2),
        selectivity_expanded=round(selectivity_expanded, 2),
        expanded_selectivity_ratio=round(expanded_selectivity_ratio, 2),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    # Save
    out_path = os.path.join(rd, 'dict_expansion.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=_convert)

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")
    print(f"\n  → {out_path}")
