"""
Phase 66, Track 10: N-gram Frequency Ranking
==============================================
Multi-token frequency rankings compared against syllabified CI word
frequencies via Spearman correlation.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    data/reference/latin/circa_instans.txt
        -> results/p66_ngram_freq.json
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)
from voynich.core.stats import syllabify_latin
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
# Spearman fallback
# ---------------------------------------------------------------------------

def _spearman_rho(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Spearman rank correlation.

    Tries scipy first; falls back to manual computation.
    """
    if len(x) < 3 or len(x) != len(y):
        return 0.0, 1.0
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except ImportError:
        pass
    # Manual rank computation
    n = len(x)

    def _rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[order[j]] == vals[order[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1.0 - (6.0 * d_sq) / (n * (n * n - 1))
    # Approximate p-value using t-distribution approximation
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t_stat = rho * ((n - 2) / (1 - rho * rho)) ** 0.5
        # Very rough two-tailed p from normal approximation for large n
        z = abs(t_stat)
        p = max(2.0 * np.exp(-0.5 * z * z) / (z * (2 * np.pi) ** 0.5), 1e-300) if z > 0 else 1.0
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# CI loader
# ---------------------------------------------------------------------------

def _load_ci_words(ci_path: str) -> List[str]:
    """Load Circa Instans text and return all words (lowercase, alpha-only)."""
    if not os.path.exists(ci_path):
        return []
    with open(ci_path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    words = []
    for token in text.lower().split():
        cleaned = re.sub(r'[^a-z]', '', token)
        if len(cleaned) >= 2:
            words.append(cleaned)
    return words


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class NgramFreqResult:
    phase: str = "66"
    step: str = "66.10"
    experiment: str = "ngram_frequency"
    n_voynich_types: int = 0
    n_ci_types: int = 0
    n_matched_exact: int = 0
    n_matched_ed2: int = 0
    spearman_rho: float = 0.0
    spearman_p: float = 1.0
    top_matched: List[Dict] = field(default_factory=list)
    n1_rho: bool = False                # rho > 0.3
    n2_matched: bool = False            # >= 10 matched pairs at ED <= 2
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_ngram_freq():
    """Phase 66.10: N-gram frequency ranking."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 66, Track 10: N-gram Frequency Ranking")
    print("=" * 70)

    # Load dependencies
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    if not assignment:
        print("  WARNING: combined_refine.json not found or empty; using empty assignment")
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)

    # Decode all Voynich tokens
    all_tokens = corpus.get_tokens()
    decoded = decode_corpus_cvc_v2(all_tokens, assignment, eva_to_triple, coda_table)
    v_valid = [d for d in decoded if d and d != '?']

    # Build frequency-ranked list of decoded word types
    v_counter = Counter(v_valid)
    v_types_ranked = v_counter.most_common()  # (word, count), sorted by freq desc
    v_rank_map = {w: rank + 1 for rank, (w, _) in enumerate(v_types_ranked)}
    n_v_types = len(v_types_ranked)
    print(f"  Voynich decoded types: {n_v_types}")
    print(f"  Voynich decoded tokens: {len(v_valid)}")

    # Load CI text
    ci_path = os.path.join(str(_data_dir()), 'reference', 'latin', 'circa_instans.txt')
    ci_words = _load_ci_words(ci_path)
    print(f"  CI words loaded: {len(ci_words)}")

    if not ci_words:
        print("  WARNING: No CI words found; results will be empty")

    # Syllabify CI words and build frequency-ranked list
    ci_counter = Counter()
    for w in ci_words:
        syls = syllabify_latin(w)
        if syls:
            # Concatenate syllables to get the "decoded form equivalent"
            decoded_form = ''.join(syls)
            ci_counter[decoded_form] += 1
        else:
            # If syllabification fails, use the raw word
            ci_counter[w] += 1

    ci_types_ranked = ci_counter.most_common()
    ci_rank_map = {w: rank + 1 for rank, (w, _) in enumerate(ci_types_ranked)}
    n_ci_types = len(ci_types_ranked)
    print(f"  CI types (syllabified): {n_ci_types}")

    # Match Voynich types to CI types by edit distance
    ci_type_set = set(ci_rank_map.keys())
    matched_pairs_exact = []
    matched_pairs_ed2 = []

    # For efficiency, only check Voynich types with rank <= 500
    # and CI types with rank <= 500 for ED matching
    v_check = [w for w, _ in v_types_ranked[:500]]
    ci_check = [w for w, _ in ci_types_ranked[:500]]

    print(f"  Matching top {len(v_check)} Voynich types against top {len(ci_check)} CI types...")

    for v_word in v_check:
        # Check exact match first
        if v_word in ci_type_set:
            matched_pairs_exact.append({
                'voynich_word': v_word,
                'ci_word': v_word,
                'edit_distance': 0,
                'v_rank': v_rank_map[v_word],
                'ci_rank': ci_rank_map[v_word],
                'v_count': v_counter[v_word],
                'ci_count': ci_counter[v_word],
            })
            continue

        # Check ED <= 2 against CI top types
        best_ed = 999
        best_ci = None
        for ci_word in ci_check:
            # Quick length filter
            if abs(len(v_word) - len(ci_word)) > 2:
                continue
            ed = _edit_distance(v_word, ci_word)
            if ed < best_ed:
                best_ed = ed
                best_ci = ci_word
            if ed == 0:
                break
        if best_ci is not None and best_ed <= 2:
            matched_pairs_ed2.append({
                'voynich_word': v_word,
                'ci_word': best_ci,
                'edit_distance': best_ed,
                'v_rank': v_rank_map[v_word],
                'ci_rank': ci_rank_map[best_ci],
                'v_count': v_counter[v_word],
                'ci_count': ci_counter[best_ci],
            })

    # Merge exact into ed2 list (exact are also ED <= 2)
    all_matched = matched_pairs_exact + matched_pairs_ed2
    n_exact = len(matched_pairs_exact)
    n_ed2 = len(all_matched)
    print(f"  Exact matches: {n_exact}")
    print(f"  ED<=2 matches: {n_ed2}")

    # Compute Spearman rank correlation on matched pairs
    if all_matched:
        v_ranks = [m['v_rank'] for m in all_matched]
        ci_ranks = [m['ci_rank'] for m in all_matched]
        rho, p_val = _spearman_rho(v_ranks, ci_ranks)
    else:
        rho, p_val = 0.0, 1.0

    print(f"  Spearman rho: {rho:.4f}, p: {p_val:.6f}")

    # Sort matched by Voynich rank for output
    all_matched.sort(key=lambda m: m['v_rank'])
    top_matched = all_matched[:50]

    if top_matched:
        print(f"\n  Top matched pairs:")
        print(f"  {'V_word':<15s} {'CI_word':<15s} {'ED':>3s} {'V_rank':>7s} {'CI_rank':>7s}")
        for m in top_matched[:15]:
            print(f"  {m['voynich_word']:<15s} {m['ci_word']:<15s} "
                  f"{m['edit_distance']:3d} {m['v_rank']:7d} {m['ci_rank']:7d}")

    # Gates
    n1 = rho > 0.3
    n2 = n_ed2 >= 10
    gates_passed = sum([n1, n2])

    if gates_passed == 2:
        verdict = "FREQUENCY_CORRELATED"
    elif gates_passed == 1:
        verdict = "FREQUENCY_MARGINAL"
    else:
        verdict = "FREQUENCY_NO_SIGNAL"

    result = NgramFreqResult(
        n_voynich_types=n_v_types,
        n_ci_types=n_ci_types,
        n_matched_exact=n_exact,
        n_matched_ed2=n_ed2,
        spearman_rho=round(rho, 4),
        spearman_p=round(p_val, 6) if p_val > 1e-15 else p_val,
        top_matched=top_matched,
        n1_rho=n1,
        n2_matched=n2,
        gates_passed=gates_passed,
        gate_passed=gates_passed == 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  {'Metric':<30s} {'Value':>10s}")
    print(f"  {'Voynich types':<30s} {n_v_types:10d}")
    print(f"  {'CI types':<30s} {n_ci_types:10d}")
    print(f"  {'Exact matches':<30s} {n_exact:10d}")
    print(f"  {'ED<=2 matches':<30s} {n_ed2:10d}")
    print(f"  {'Spearman rho':<30s} {rho:10.4f}")
    print(f"  {'Spearman p':<30s} {p_val:10.6f}")
    print(f"\n  Gates: N1={'PASS' if n1 else 'FAIL'} N2={'PASS' if n2 else 'FAIL'} "
          f"({gates_passed}/2)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'p66_ngram_freq.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
