"""
Phase 66, Track 9: Collocational Analysis
==========================================
Decoded token co-occurrence networks compared against CI ingredient
co-occurrence. Null: shuffled token order destroys collocations.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    data/reference/latin/circa_instans.txt
        -> results/p66_collocations.json
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
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
# Edit distance
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


# ---------------------------------------------------------------------------
# CI loader
# ---------------------------------------------------------------------------

def _load_ci_entries(ci_path: str) -> List[Set[str]]:
    """Load Circa Instans text and split into entries (paragraphs).

    Returns a list of sets, each set containing the lowercase words in
    one CI entry.
    """
    if not os.path.exists(ci_path):
        return []
    with open(ci_path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    # Split on blank lines (paragraph boundaries)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    entries = []
    for para in paragraphs:
        words = set()
        for token in para.lower().split():
            # Strip punctuation
            cleaned = ''.join(c for c in token if c.isalpha())
            if len(cleaned) >= 2:
                words.add(cleaned)
        if words:
            entries.append(words)
    return entries


def _ci_pair_has_match(word_a: str, word_b: str, ci_entries: List[Set[str]]) -> bool:
    """Check if any CI entry contains both words (exact or ED <= 2)."""
    for entry_words in ci_entries:
        found_a = False
        found_b = False
        for ew in entry_words:
            if not found_a:
                if word_a == ew or _edit_distance(word_a, ew) <= 2:
                    found_a = True
            if not found_b:
                if word_b == ew or _edit_distance(word_b, ew) <= 2:
                    found_b = True
            if found_a and found_b:
                return True
    return False


# ---------------------------------------------------------------------------
# Co-occurrence computation
# ---------------------------------------------------------------------------

def _compute_cooccurrences(
    page_decoded: List[List[str]], window: int = 5
) -> Tuple[Counter, Counter, int]:
    """Compute pairwise co-occurrence counts within a sliding window.

    Returns (pair_counts, unigram_counts, total_windows).
    """
    pair_counts: Counter = Counter()
    unigram_counts: Counter = Counter()
    total_windows = 0

    for tokens in page_decoded:
        valid = [t for t in tokens if t and t != '?']
        for i in range(len(valid)):
            unigram_counts[valid[i]] += 1
            for j in range(i + 1, min(i + window, len(valid))):
                pair = tuple(sorted([valid[i], valid[j]]))
                pair_counts[pair] += 1
            total_windows += 1

    return pair_counts, unigram_counts, total_windows


def _significant_collocations(
    pair_counts: Counter, unigram_counts: Counter, total_windows: int,
    t_threshold: float = 2.0
) -> List[Tuple[Tuple[str, str], float, int]]:
    """Extract collocations with t-score > threshold.

    t = (O - E) / sqrt(O) where E = (f1 * f2) / N
    Returns list of ((word_a, word_b), t_score, count).
    """
    N = total_windows if total_windows > 0 else 1
    results = []
    for (w_a, w_b), obs in pair_counts.items():
        if obs < 2:
            continue
        f1 = unigram_counts.get(w_a, 0)
        f2 = unigram_counts.get(w_b, 0)
        expected = (f1 * f2) / N
        if obs > 0:
            t_score = (obs - expected) / (obs ** 0.5)
            if t_score > t_threshold:
                results.append(((w_a, w_b), round(t_score, 3), obs))
    results.sort(key=lambda x: -x[1])
    return results


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CollocationResult:
    phase: str = "66"
    step: str = "66.9"
    experiment: str = "collocational_analysis"
    n_voynich_types: int = 0
    n_collocations: int = 0
    n_significant: int = 0
    n_ci_matching: int = 0
    null_mean_ci_match: float = 0.0
    null_std_ci_match: float = 0.0
    selectivity: float = 0.0
    top_collocations: List[Dict] = field(default_factory=list)
    ci_matches: List[Dict] = field(default_factory=list)
    c1_significant: bool = False        # >= 50 significant collocations
    c2_ci_match: bool = False           # >= 5 match CI
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_collocations():
    """Phase 66.9: Collocational analysis."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 66, Track 9: Collocational Analysis")
    print("=" * 70)

    # Load dependencies
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    if not assignment:
        print("  WARNING: combined_refine.json not found or empty; using empty assignment")
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)

    # Decode all tokens, grouped by page
    page_decoded: List[List[str]] = []
    all_decoded_flat: List[str] = []
    for folio_id, page in corpus.pages.items():
        tokens = page.all_tokens
        decoded = decode_corpus_cvc_v2(tokens, assignment, eva_to_triple, coda_table)
        page_decoded.append(decoded)
        all_decoded_flat.extend(decoded)

    valid_flat = [d for d in all_decoded_flat if d and d != '?']
    n_types = len(set(valid_flat))
    print(f"  Decoded tokens: {len(valid_flat)}, types: {n_types}")

    # Compute real co-occurrences
    pair_counts, unigram_counts, total_windows = _compute_cooccurrences(page_decoded)
    n_all_pairs = len(pair_counts)
    sig_collocs = _significant_collocations(pair_counts, unigram_counts, total_windows)
    n_significant = len(sig_collocs)
    print(f"  Total co-occurrence pairs: {n_all_pairs}")
    print(f"  Significant collocations (t>2.0): {n_significant}")

    # Load CI entries
    ci_path = os.path.join(str(_data_dir()), 'reference', 'latin', 'circa_instans.txt')
    ci_entries = _load_ci_entries(ci_path)
    print(f"  CI entries loaded: {len(ci_entries)}")

    # Check significant collocations against CI
    ci_match_list = []
    if ci_entries and sig_collocs:
        # Only check top 200 significant collocations for performance
        check_limit = min(200, len(sig_collocs))
        print(f"  Checking top {check_limit} collocations against CI...")
        for (w_a, w_b), t_score, count in sig_collocs[:check_limit]:
            if _ci_pair_has_match(w_a, w_b, ci_entries):
                ci_match_list.append({
                    'pair': [w_a, w_b],
                    't_score': t_score,
                    'count': count,
                })
    n_ci_matching = len(ci_match_list)
    print(f"  CI-matching collocations: {n_ci_matching}")

    # Null test: 100 shuffled orderings
    print("  Running null test (100 shuffles)...")
    rng = random.Random(42)
    null_ci_counts = []
    n_null_trials = 100
    for trial in range(n_null_trials):
        # Shuffle tokens within each page
        shuffled_pages = []
        for page_tokens in page_decoded:
            valid_page = [t for t in page_tokens if t and t != '?']
            shuffled = valid_page.copy()
            rng.shuffle(shuffled)
            shuffled_pages.append(shuffled)

        null_pairs, null_uni, null_tw = _compute_cooccurrences(shuffled_pages)
        null_sig = _significant_collocations(null_pairs, null_uni, null_tw)

        # Count CI matches in null
        null_ci = 0
        if ci_entries and null_sig:
            check_n = min(200, len(null_sig))
            for (w_a, w_b), t_score, count in null_sig[:check_n]:
                if _ci_pair_has_match(w_a, w_b, ci_entries):
                    null_ci += 1
        null_ci_counts.append(null_ci)

        if (trial + 1) % 25 == 0:
            print(f"    Trial {trial + 1}/{n_null_trials}: null CI matches = {null_ci}")

    null_mean = float(np.mean(null_ci_counts)) if null_ci_counts else 0.0
    null_std = float(np.std(null_ci_counts)) if null_ci_counts else 0.0
    selectivity = n_ci_matching / null_mean if null_mean > 0 else (
        float('inf') if n_ci_matching > 0 else 0.0
    )

    print(f"  Null CI match mean: {null_mean:.2f} +/- {null_std:.2f}")
    print(f"  Selectivity: {selectivity:.2f}x")

    # Top collocations for output
    top_collocs = [
        {'pair': list(pair), 't_score': t, 'count': c}
        for (pair, t, c) in sig_collocs[:50]
    ]

    # Gates
    c1 = n_significant >= 50
    c2 = n_ci_matching >= 5
    gates_passed = sum([c1, c2])

    if gates_passed == 2:
        verdict = "COLLOCATIONS_CONFIRMED"
    elif gates_passed == 1:
        verdict = "COLLOCATIONS_MARGINAL"
    else:
        verdict = "COLLOCATIONS_NO_SIGNAL"

    result = CollocationResult(
        n_voynich_types=n_types,
        n_collocations=n_all_pairs,
        n_significant=n_significant,
        n_ci_matching=n_ci_matching,
        null_mean_ci_match=round(null_mean, 2),
        null_std_ci_match=round(null_std, 2),
        selectivity=round(selectivity, 2),
        top_collocations=top_collocs,
        ci_matches=ci_match_list,
        c1_significant=c1,
        c2_ci_match=c2,
        gates_passed=gates_passed,
        gate_passed=gates_passed == 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  {'Metric':<30s} {'Value':>10s}")
    print(f"  {'Voynich types':<30s} {n_types:10d}")
    print(f"  {'Total co-occurrence pairs':<30s} {n_all_pairs:10d}")
    print(f"  {'Significant (t>2.0)':<30s} {n_significant:10d}")
    print(f"  {'CI-matching':<30s} {n_ci_matching:10d}")
    print(f"  {'Null mean CI matches':<30s} {null_mean:10.2f}")
    print(f"  {'Selectivity':<30s} {selectivity:10.2f}")
    print(f"\n  Gates: C1={'PASS' if c1 else 'FAIL'} C2={'PASS' if c2 else 'FAIL'} "
          f"({gates_passed}/2)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'p66_collocations.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
