"""
Phase 16.3 – Syllable Distribution Matching (Approach A)
========================================================
Finds the modifier set that makes the Voynich token-length distribution
match the Latin word-length (syllable-count) distribution.

Latin medical words peak at 2–3 syllables (mean ≈ 2.5). The current
feature model gives ~3–5 triples per token (mean ≈ 3.3). Removing
modifier characters should bring the corrected distribution into alignment.

Dependency chain:
    modifier_standalone.json  (candidate list from Approach B)
    corpus (IVTFF)
    reference corpus (Latin)
        → modifier_distribution.json (this step)
"""

import itertools
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import syllabify_latin


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DistributionMatch:
    """Result for one candidate modifier set."""
    modifier_chars: List[str]
    n_modifiers: int
    corrected_mean_syllables: float
    corrected_std_syllables: float
    corrected_distribution: Dict[str, float]  # str keys for JSON
    ks_statistic: float
    chi_squared: float
    distribution_score: float  # lower is better


@dataclass
class DistributionResult:
    latin_syllable_distribution: Dict[str, float]
    latin_mean_syllables: float
    voynich_raw_mean_triples: float
    voynich_raw_distribution: Dict[str, float]
    n_candidate_modifiers: int
    n_subsets_tested: int
    top_k_matches: List[Dict]
    best_match: Dict
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_latin_syllable_distribution(
    ref_tokens: List[str],
) -> Tuple[Dict[int, float], float]:
    """Compute syllable-count distribution from Latin reference words.

    Returns (distribution, mean_syllables).
    """
    counts: Counter = Counter()
    total = 0
    total_syl = 0

    for word in ref_tokens:
        word_clean = word.lower().strip()
        if len(word_clean) < 2:
            continue
        syls = syllabify_latin(word_clean)
        n_syl = max(len(syls), 1)
        counts[n_syl] += 1
        total += 1
        total_syl += n_syl

    if total == 0:
        return {}, 0.0

    dist = {k: v / total for k, v in sorted(counts.items())}
    mean = total_syl / total
    return dist, mean


def compute_voynich_syllable_distribution(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> Tuple[Dict[int, float], float, float]:
    """Compute syllable-count distribution after stripping modifier chars.

    Returns (distribution, mean, std).
    """
    counts: Counter = Counter()
    all_n: List[int] = []

    for token in tokens:
        chars = tokenize_eva_chars(token)
        # Count non-modifier chars that have a triple mapping
        n_syllabic = sum(
            1 for ch in chars
            if ch not in modifier_chars and ch in eva_to_triple
        )
        n_syllabic = max(n_syllabic, 1)  # at least 1 syllable per token
        counts[n_syllabic] += 1
        all_n.append(n_syllabic)

    total = len(all_n)
    if total == 0:
        return {}, 0.0, 0.0

    dist = {k: v / total for k, v in sorted(counts.items())}
    mean = sum(all_n) / total
    std = (sum((x - mean) ** 2 for x in all_n) / total) ** 0.5
    return dist, mean, std


def _ks_statistic(dist_a: Dict[int, float], dist_b: Dict[int, float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic between two distributions."""
    all_keys = sorted(set(dist_a.keys()) | set(dist_b.keys()))
    cdf_a = 0.0
    cdf_b = 0.0
    max_diff = 0.0
    for k in all_keys:
        cdf_a += dist_a.get(k, 0.0)
        cdf_b += dist_b.get(k, 0.0)
        diff = abs(cdf_a - cdf_b)
        if diff > max_diff:
            max_diff = diff
    return max_diff


def _chi_squared(
    observed: Dict[int, float],
    expected: Dict[int, float],
) -> float:
    """Chi-squared distance between two normalised distributions."""
    all_keys = sorted(set(observed.keys()) | set(expected.keys()))
    chi2 = 0.0
    for k in all_keys:
        o = observed.get(k, 0.0)
        e = expected.get(k, 0.0)
        if e > 0:
            chi2 += (o - e) ** 2 / e
    return chi2


def search_modifier_subsets(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    candidates: List[str],
    latin_dist: Dict[int, float],
    latin_mean: float,
) -> List[DistributionMatch]:
    """Search over subsets of modifier candidates for best distribution match.

    Strategy:
    1. If <= 12 candidates: exhaustive enumeration (2^12 = 4096)
    2. If > 12: greedy forward selection + random sampling
    """
    results: List[DistributionMatch] = []

    if len(candidates) <= 12:
        # Exhaustive search
        for size in range(1, len(candidates) + 1):
            for subset in itertools.combinations(candidates, size):
                modifier_set = set(subset)
                dist, mean, std = compute_voynich_syllable_distribution(
                    tokens, eva_to_triple, modifier_set,
                )
                ks = _ks_statistic(dist, latin_dist)
                chi2 = _chi_squared(dist, latin_dist)

                # Combined score: KS + distance from Latin mean
                mean_penalty = abs(mean - latin_mean) / latin_mean
                score = ks + 0.5 * mean_penalty

                results.append(DistributionMatch(
                    modifier_chars=sorted(subset),
                    n_modifiers=len(subset),
                    corrected_mean_syllables=round(mean, 3),
                    corrected_std_syllables=round(std, 3),
                    corrected_distribution={str(k): round(v, 4) for k, v in dist.items()},
                    ks_statistic=round(ks, 4),
                    chi_squared=round(chi2, 4),
                    distribution_score=round(score, 4),
                ))
    else:
        # Greedy forward selection
        current_set: Set[str] = set()
        remaining = list(candidates)

        for _ in range(min(len(candidates), 10)):
            best_cand = None
            best_score = float('inf')
            best_match = None

            for cand in remaining:
                test_set = current_set | {cand}
                dist, mean, std = compute_voynich_syllable_distribution(
                    tokens, eva_to_triple, test_set,
                )
                ks = _ks_statistic(dist, latin_dist)
                mean_penalty = abs(mean - latin_mean) / latin_mean
                score = ks + 0.5 * mean_penalty

                if score < best_score:
                    best_score = score
                    best_cand = cand
                    chi2 = _chi_squared(dist, latin_dist)
                    best_match = DistributionMatch(
                        modifier_chars=sorted(test_set),
                        n_modifiers=len(test_set),
                        corrected_mean_syllables=round(mean, 3),
                        corrected_std_syllables=round(std, 3),
                        corrected_distribution={str(k): round(v, 4) for k, v in dist.items()},
                        ks_statistic=round(ks, 4),
                        chi_squared=round(chi2, 4),
                        distribution_score=round(best_score, 4),
                    )

            if best_cand and best_match:
                current_set.add(best_cand)
                remaining.remove(best_cand)
                results.append(best_match)

    # Sort by score (lower is better)
    results.sort(key=lambda m: m.distribution_score)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_modifier_distribution() -> None:
    """Step 16.3: Syllable distribution matching (Approach A)."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 16.3: Syllable Distribution Matching (Approach A)")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load modifier candidates from Approach B ───
    standalone_path = os.path.join(rd, 'modifier_standalone.json')
    if not os.path.exists(standalone_path):
        print("  [SKIP] modifier_standalone.json not found — run mod-standalone first")
        return

    with open(standalone_path) as f:
        standalone_data = json.load(f)

    candidates = standalone_data.get('modifier_candidates', [])
    print(f"\n  1. Loaded {len(candidates)} modifier candidates from Approach B:")
    print(f"      {candidates}")

    # ─── Load corpus ───
    print("\n  2. Loading Voynich corpus …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"      {len(tokens)} tokens")

    # ─── Compute raw Voynich distribution ───
    print("\n  3. Computing raw Voynich triples-per-token distribution …")
    raw_dist, raw_mean, raw_std = compute_voynich_syllable_distribution(
        tokens, eva_to_triple, set(),  # no modifiers
    )
    print(f"      Mean triples/token: {raw_mean:.2f} (std: {raw_std:.2f})")

    # ─── Compute Latin syllable distribution ───
    print("\n  4. Computing Latin syllable distribution …")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        ref_tokens = ref_corpus.get_combined_tokens('latin')
    except (FileNotFoundError, KeyError):
        # Fallback: use pharmaceutical vocabulary
        from voynich.core.reference import PHARMACEUTICAL_VOCABULARY
        ref_tokens = []
        for words in PHARMACEUTICAL_VOCABULARY.values():
            ref_tokens.extend(words)
        print("      (Using pharmaceutical vocabulary as fallback)")

    latin_dist, latin_mean = compute_latin_syllable_distribution(ref_tokens)
    print(f"      Latin mean syllables/word: {latin_mean:.2f}")
    print(f"      Distribution: ", end="")
    for k in sorted(latin_dist.keys()):
        print(f"{k}syl={latin_dist[k]:.1%}  ", end="")
    print()

    # ─── Search modifier subsets ───
    print(f"\n  5. Searching {len(candidates)} candidates "
          f"(up to 2^{len(candidates)} = {2**len(candidates)} subsets) …")
    matches = search_modifier_subsets(
        tokens, eva_to_triple, candidates, latin_dist, latin_mean,
    )
    n_tested = len(matches)
    print(f"      Tested {n_tested} subsets")

    # ─── Results ───
    top_k = 10
    top_matches = matches[:top_k]

    if top_matches:
        best = top_matches[0]
        print(f"\n  6. Best modifier set: {best.modifier_chars}")
        print(f"      Corrected mean: {best.corrected_mean_syllables:.2f} "
              f"(target: {latin_mean:.2f})")
        print(f"      KS statistic: {best.ks_statistic:.4f}")
        print(f"      Chi-squared: {best.chi_squared:.4f}")
        print(f"      Score: {best.distribution_score:.4f}")

        print(f"\n  7. Top {min(top_k, len(top_matches))} matches:")
        print(f"      {'Set':<40} {'Mean':>6} {'KS':>7} {'Chi2':>7} {'Score':>7}")
        print("      " + "-" * 70)
        for m in top_matches:
            chars_str = ','.join(m.modifier_chars)
            if len(chars_str) > 38:
                chars_str = chars_str[:35] + '...'
            print(f"      {chars_str:<40} {m.corrected_mean_syllables:>6.2f} "
                  f"{m.ks_statistic:>7.4f} {m.chi_squared:>7.4f} "
                  f"{m.distribution_score:>7.4f}")
    else:
        best = None
        print("\n  6. No modifier subsets tested")

    # ─── Gate ───
    gate_passed = (
        best is not None
        and best.ks_statistic < 0.15
        and 2.0 <= best.corrected_mean_syllables <= 3.0
    )
    if best:
        verdict = (
            f"PASS: Best set {best.modifier_chars} gives mean "
            f"{best.corrected_mean_syllables:.2f} syl/token, KS={best.ks_statistic:.4f}."
            if gate_passed
            else f"FAIL: Best KS={best.ks_statistic:.4f}, "
            f"mean={best.corrected_mean_syllables:.2f} "
            f"(target: 2.0–3.0, KS < 0.15)."
        )
    else:
        verdict = "FAIL: No modifier subsets could be tested."

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ─── Save ───
    result = DistributionResult(
        latin_syllable_distribution={str(k): round(v, 4) for k, v in latin_dist.items()},
        latin_mean_syllables=round(latin_mean, 3),
        voynich_raw_mean_triples=round(raw_mean, 3),
        voynich_raw_distribution={str(k): round(v, 4) for k, v in raw_dist.items()},
        n_candidate_modifiers=len(candidates),
        n_subsets_tested=n_tested,
        top_k_matches=[_convert(asdict(m)) for m in top_matches],
        best_match=_convert(asdict(best)) if best else {},
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'modifier_distribution.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
