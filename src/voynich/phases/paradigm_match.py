"""
Phase 5.2: Paradigm-to-Language Matching
=========================================
Match the observed Voynich paradigm shapes against the morphological
systems of candidate Romance languages (Latin, Occitan).

Sub-analyses:
  5.2a — Build morphological reference profiles
  5.2b — Match Voynich paradigm shapes to language profiles
  5.2c — Affix alignment (map Voynich affixes to language endings)

Output:
  results/paradigm_match.json
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus
from voynich.core.stats import (
    jensen_shannon_divergence, rank_correlation,
    chi_squared_goodness, selectivity_ratio, bootstrap_ci,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    load_reference_corpus, compute_suffix_inventory,
    get_paradigm_shape_profile, ReferenceCorpus,
    LATIN_DECLENSION_SUFFIXES, OCCITAN_DECLENSION_SUFFIXES,
    LATIN_PARADIGM_PROFILES, OCCITAN_PARADIGM_PROFILES,
)
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes, decompose_corpus_morphemes,
    MorphemeDecomposition, KNOWN_SUFFIXES,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LanguageMorphProfile:
    """Morphological reference profile for one language."""
    language: str
    n_paradigm_types: int
    paradigm_sizes: List[int]           # Expected sizes per type
    suffix_inventory_size: int
    suffix_distribution: Dict[str, int]
    expected_shape_distribution: Dict[str, float]  # Binned paradigm size dist


@dataclass
class AffixAlignment:
    """Mapping of one Voynich affix to a candidate language ending."""
    voynich_affix: str
    candidate_ending: str
    frequency_rank_voynich: int
    frequency_rank_candidate: int
    rank_distance: int
    co_occurrence_score: float


@dataclass
class ParadigmMatchResult:
    """Full Phase 5.2 output."""
    languages_tested: List[str]
    # Per-language match scores
    jsd_scores: Dict[str, float]
    rank_correlations: Dict[str, float]
    chi2_scores: Dict[str, List[float]]   # [chi2, p_value]
    combined_scores: Dict[str, float]
    # Best match
    best_language: str
    second_language: str
    best_jsd: float
    second_jsd: float
    jsd_ratio: float
    # Affix alignment
    affix_alignments: List[Dict]
    alignment_consistency: float
    # Null test
    null_jsd_mean: float
    null_jsd_std: float
    real_vs_null_z: float
    # Gates
    jsd_separation_met: bool
    consistency_met: bool
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# 5.2a: Build morphological reference profiles
# ---------------------------------------------------------------------------

def build_morph_profile(
    language: str,
    ref_corpus: Optional[ReferenceCorpus] = None,
) -> LanguageMorphProfile:
    """
    Build morphological reference profile from language data.

    Uses embedded paradigm tables + empirical suffix frequencies.
    """
    inventory = compute_suffix_inventory(language, corpus=ref_corpus)
    profiles = get_paradigm_shape_profile(language)

    paradigm_sizes = [p['mean_forms'] for p in profiles.values()]

    # Build expected paradigm-size distribution as probabilities
    # Weight: nouns 40%, verbs 30%, adjectives 20%, invariable 10%
    weights = {
        'noun_declension': 0.40,
        'adj_declension': 0.20,
        'verb_conjugation': 0.30,
        'invariable': 0.10,
    }

    # Create a histogram over paradigm sizes 1-10+
    bins = list(range(1, 11))  # 1..10 (10 = "10+")
    expected_dist: Dict[str, float] = {str(b): 0.0 for b in bins}

    for ptype, prof in profiles.items():
        w = weights.get(ptype, 0.1)
        mean_f = prof['mean_forms']
        std_f = prof['std_forms']
        # Gaussian contribution to each bin
        for b in bins:
            val = b if b < 10 else mean_f  # Bin 10 = "10+"
            contrib = w * math.exp(-0.5 * ((val - mean_f) / max(std_f, 0.5)) ** 2)
            expected_dist[str(b)] += contrib

    # Normalize
    total = sum(expected_dist.values())
    if total > 0:
        expected_dist = {k: v / total for k, v in expected_dist.items()}

    return LanguageMorphProfile(
        language=language,
        n_paradigm_types=len(profiles),
        paradigm_sizes=paradigm_sizes,
        suffix_inventory_size=len(inventory.get('suffix_types', [])),
        suffix_distribution=inventory.get('suffix_distribution', {}),
        expected_shape_distribution=expected_dist,
    )


# ---------------------------------------------------------------------------
# 5.2b: Match paradigm shapes
# ---------------------------------------------------------------------------

def compute_paradigm_size_distribution(
    paradigm_data: Dict,
) -> np.ndarray:
    """
    Convert Voynich paradigm size distribution to a probability array.

    Bins: 1, 2, 3, ..., 9, 10+ (10 bins).
    """
    raw_dist = paradigm_data.get('paradigm_size_distribution', {})
    bins = np.zeros(10)
    for size_str, count in raw_dist.items():
        size = int(size_str)
        idx = min(size, 10) - 1  # bin 0=size1, bin 9=size10+
        bins[idx] += count

    total = bins.sum()
    if total > 0:
        bins = bins / total
    return bins


def match_paradigm_shapes(
    voynich_dist: np.ndarray,
    language_profiles: Dict[str, LanguageMorphProfile],
) -> Dict[str, Dict[str, float]]:
    """
    Match Voynich paradigm shape distribution against each language.

    Returns {language: {jsd, rho, rho_p, chi2, chi2_p, combined}}.
    """
    results: Dict[str, Dict[str, float]] = {}

    for lang, profile in language_profiles.items():
        # Build expected distribution array
        expected = np.array([
            profile.expected_shape_distribution.get(str(i + 1), 0.0)
            for i in range(10)
        ])
        # Ensure both are valid probability distributions
        if expected.sum() > 0:
            expected = expected / expected.sum()

        # JSD
        jsd = jensen_shannon_divergence(voynich_dist, expected)

        # Spearman rank correlation
        rho, rho_p = rank_correlation(voynich_dist, expected)

        # Chi-squared (scale to counts for meaningful test)
        n_paradigms = max(1, int(voynich_dist.sum() * 1000))  # Approximate N
        observed_counts = voynich_dist * n_paradigms
        expected_counts = expected * n_paradigms
        chi2, chi2_p = chi_squared_goodness(observed_counts, expected_counts)

        # Combined score (higher = better match)
        jsd_score = max(0, 1.0 - jsd)
        rho_score = (1.0 + rho) / 2.0
        chi2_score = 1.0 if chi2_p > 0.05 else 0.0
        combined = jsd_score * 0.4 + rho_score * 0.3 + chi2_score * 0.3

        results[lang] = {
            'jsd': round(jsd, 6),
            'rho': round(rho, 4),
            'rho_p': round(rho_p, 6),
            'chi2': round(chi2, 4),
            'chi2_p': round(chi2_p, 6),
            'combined': round(combined, 4),
        }

    return results


# ---------------------------------------------------------------------------
# 5.2c: Affix alignment
# ---------------------------------------------------------------------------

def align_affixes(
    voynich_suffix_dist: Dict[str, int],
    language: str,
) -> Tuple[List[AffixAlignment], float]:
    """
    Align Voynich suffixes to candidate language endings by frequency rank.

    Returns (alignments, consistency_ratio).
    Consistency = fraction of alignments where rank distance <= 3.
    """
    if language == 'latin':
        suffix_table = LATIN_DECLENSION_SUFFIXES
    elif language == 'occitan':
        suffix_table = OCCITAN_DECLENSION_SUFFIXES
    else:
        return [], 0.0

    # Flatten language suffix inventory with approximate frequencies
    lang_suffix_freq: Counter = Counter()
    for paradigm_type, suffixes in suffix_table.items():
        for sfx in suffixes:
            if sfx:
                lang_suffix_freq[sfx] += 1

    # Rank Voynich suffixes
    voynich_ranked = sorted(voynich_suffix_dist.items(), key=lambda x: -x[1])
    # Rank language suffixes
    lang_ranked = sorted(lang_suffix_freq.items(), key=lambda x: -x[1])

    # Create rank lookup
    lang_rank_map = {sfx: rank for rank, (sfx, _) in enumerate(lang_ranked)}

    # Align: for each Voynich suffix, find best-matching language suffix
    alignments: List[AffixAlignment] = []
    used_lang: set = set()

    for v_rank, (v_sfx, v_count) in enumerate(voynich_ranked):
        best_match = None
        best_dist = float('inf')

        for l_sfx, l_rank in lang_rank_map.items():
            if l_sfx in used_lang:
                continue
            dist = abs(v_rank - l_rank)
            if dist < best_dist:
                best_dist = dist
                best_match = l_sfx

        if best_match is not None:
            used_lang.add(best_match)
            l_rank = lang_rank_map[best_match]
            # Score: inverse of rank distance, normalized
            max_rank = max(len(voynich_ranked), len(lang_ranked))
            score = max(0.0, 1.0 - best_dist / max(max_rank, 1))

            alignments.append(AffixAlignment(
                voynich_affix=v_sfx,
                candidate_ending=best_match,
                frequency_rank_voynich=v_rank,
                frequency_rank_candidate=l_rank,
                rank_distance=int(best_dist),
                co_occurrence_score=round(score, 4),
            ))

    # Consistency: fraction with rank distance <= 3
    if alignments:
        n_consistent = sum(1 for a in alignments if a.rank_distance <= 3)
        consistency = n_consistent / len(alignments)
    else:
        consistency = 0.0

    return alignments, consistency


# ---------------------------------------------------------------------------
# Null test
# ---------------------------------------------------------------------------

def null_test_paradigm_match(
    tokens: List[str],
    best_jsd: float,
    best_language: str,
    ref_corpus: Optional[ReferenceCorpus] = None,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test: shuffle token-internal character order, re-decompose,
    rebuild paradigms, re-match against best language.

    Returns (null_mean_jsd, null_std_jsd, z_score).
    """
    rng = random.Random(seed)
    profile = build_morph_profile(best_language, ref_corpus)
    null_jsds: List[float] = []

    for trial in range(n_trials):
        # Shuffle characters within each token
        shuffled_tokens = []
        for t in tokens:
            chars = list(t)
            rng.shuffle(chars)
            shuffled_tokens.append(''.join(chars))

        # Decompose
        shuffled_decomps = [decompose_token_morphemes(t) for t in shuffled_tokens]

        # Build paradigm size distribution
        stem_groups: Dict[str, set] = defaultdict(set)
        for d in shuffled_decomps:
            if d.stem:
                stem_groups[d.stem].add(d.token)

        size_dist = Counter(len(forms) for forms in stem_groups.values())

        # Convert to array
        bins = np.zeros(10)
        for size, count in size_dist.items():
            idx = min(size, 10) - 1
            bins[idx] += count
        total = bins.sum()
        if total > 0:
            bins = bins / total

        # Match against best language
        expected = np.array([
            profile.expected_shape_distribution.get(str(i + 1), 0.0)
            for i in range(10)
        ])
        if expected.sum() > 0:
            expected = expected / expected.sum()

        jsd = jensen_shannon_divergence(bins, expected)
        null_jsds.append(jsd)

    null_arr = np.array(null_jsds)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))

    # Lower JSD is better, so z is inverted
    z = (null_mean - best_jsd) / null_std if null_std > 0 else 0.0

    return null_mean, null_std, z


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def _check_gate(
    name: str, value: float, threshold: float, direction: str = 'greater',
) -> Tuple[bool, str]:
    """Check a single gate condition."""
    if direction == 'greater':
        passed = value > threshold
        op = '>'
    else:
        passed = value < threshold
        op = '<'
    status = 'PASSED' if passed else 'FAILED'
    return passed, f"  Gate [{name}]: {value:.4f} {op} {threshold} -> {status}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_paradigm_match(paradigm_data: Dict = None) -> Dict:
    """
    Run Phase 5.2: Paradigm-to-Language Matching.

    1. Load Phase 5.1 results
    2. Build morphological profiles for Latin and Occitan
    3. Match Voynich paradigm shapes to language profiles
    4. Align affixes to best-matching language
    5. Null test
    6. Gate: JSD separation + alignment consistency
    """
    print("=" * 70)
    print("Phase 5.2: Paradigm-to-Language Matching")
    print("=" * 70)

    # 1. Load Phase 5.1 results
    if paradigm_data is None:
        result_path = os.path.join(_results_dir(), 'paradigm_discovery.json')
        if not os.path.exists(result_path):
            print("  ERROR: Phase 5.1 results not found. Run 'voynich paradigms' first.")
            return {}
        with open(result_path) as f:
            paradigm_data = json.load(f)

    # Check 5.1 gate
    if not paradigm_data.get('gate_passed', False):
        print("  WARNING: Phase 5.1 gate failed. Proceeding with caution.")

    print(f"\n  Phase 5.1 input: {paradigm_data['n_stems']} stems, "
          f"mean paradigm size {paradigm_data['mean_paradigm_size']}")

    # 2. Build morphological profiles
    print("\n  5.2a: Building morphological reference profiles")
    try:
        ref_corpus = load_reference_corpus(verbose=False)
    except FileNotFoundError:
        ref_corpus = None
        print("    WARNING: No reference corpus found, using embedded profiles only")

    languages = ['latin', 'occitan']
    profiles: Dict[str, LanguageMorphProfile] = {}
    for lang in languages:
        profiles[lang] = build_morph_profile(lang, ref_corpus)
        p = profiles[lang]
        print(f"    {lang}: {p.n_paradigm_types} paradigm types, "
              f"{p.suffix_inventory_size} suffix types")

    # 3. Match paradigm shapes
    print("\n  5.2b: Matching paradigm shapes to language profiles")
    voynich_dist = compute_paradigm_size_distribution(paradigm_data)
    match_scores = match_paradigm_shapes(voynich_dist, profiles)

    jsd_scores: Dict[str, float] = {}
    rho_scores: Dict[str, float] = {}
    chi2_scores: Dict[str, List[float]] = {}
    combined_scores: Dict[str, float] = {}

    for lang, scores in match_scores.items():
        jsd_scores[lang] = scores['jsd']
        rho_scores[lang] = scores['rho']
        chi2_scores[lang] = [scores['chi2'], scores['chi2_p']]
        combined_scores[lang] = scores['combined']
        print(f"    {lang}: JSD={scores['jsd']:.4f}, rho={scores['rho']:.4f}, "
              f"chi2_p={scores['chi2_p']:.4f}, combined={scores['combined']:.4f}")

    # Best and second-best
    sorted_langs = sorted(jsd_scores, key=lambda l: jsd_scores[l])
    best_lang = sorted_langs[0]
    second_lang = sorted_langs[1] if len(sorted_langs) > 1 else sorted_langs[0]
    best_jsd = jsd_scores[best_lang]
    second_jsd = jsd_scores[second_lang]
    jsd_ratio = best_jsd / second_jsd if second_jsd > 0 else 0.0

    print(f"\n    Best match: {best_lang} (JSD={best_jsd:.4f})")
    print(f"    Second: {second_lang} (JSD={second_jsd:.4f})")
    print(f"    JSD ratio (best/second): {jsd_ratio:.4f}")

    # 4. Affix alignment
    print(f"\n  5.2c: Affix alignment against {best_lang}")
    # Get Voynich suffix distribution from Phase 5.1
    suffix_dist: Dict[str, int] = {}
    for p_dict in paradigm_data.get('top_20_paradigms', []):
        for sfx in p_dict.get('suffixes', []):
            suffix_dist[sfx] = suffix_dist.get(sfx, 0) + 1

    # If not enough from top-20, get from full corpus
    if len(suffix_dist) < 5:
        corpus = load_corpus(verbose=False)
        tokens = corpus.get_tokens(language='A', paragraph_only=True)
        if not tokens:
            tokens = corpus.get_tokens(paragraph_only=True)
        decomps = [decompose_token_morphemes(t) for t in set(tokens)]
        suffix_dist = Counter(d.suffix for d in decomps if d.suffix)

    alignments, consistency = align_affixes(suffix_dist, best_lang)
    print(f"    Aligned {len(alignments)} Voynich suffixes to {best_lang} endings")
    print(f"    Alignment consistency: {consistency:.2%}")
    for a in alignments[:8]:
        print(f"      '{a.voynich_affix}' -> '{a.candidate_ending}' "
              f"(rank dist={a.rank_distance}, score={a.co_occurrence_score:.2f})")

    # 5. Null test
    print("\n  Null test: paradigm match vs shuffled text")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        tokens = corpus.get_tokens(paragraph_only=True)
    null_mean, null_std, null_z = null_test_paradigm_match(
        tokens, best_jsd, best_lang, ref_corpus, n_trials=100,
    )
    print(f"    Real JSD: {best_jsd:.4f}")
    print(f"    Null JSD mean: {null_mean:.4f} +/- {null_std:.4f}")
    print(f"    z-score: {null_z:.2f}")

    # 6. Gates
    jsd_sep_ok, jsd_msg = _check_gate(
        'jsd_separation', jsd_ratio, 0.80, 'less',
    )
    cons_ok, cons_msg = _check_gate(
        'alignment_consistency', consistency, 0.50, 'greater',
    )
    gate_passed = jsd_sep_ok and cons_ok

    print(f"\n{jsd_msg}")
    print(f"{cons_msg}")
    verdict = 'language_identified' if gate_passed else 'gate_failed'
    if not jsd_sep_ok and consistency > 0.5:
        verdict = 'romance_family_only'
    print(f"  Verdict: {verdict}")
    if gate_passed:
        print(f"  Best language: {best_lang}")

    # Build result
    result = ParadigmMatchResult(
        languages_tested=languages,
        jsd_scores=jsd_scores,
        rank_correlations=rho_scores,
        chi2_scores=chi2_scores,
        combined_scores=combined_scores,
        best_language=best_lang,
        second_language=second_lang,
        best_jsd=round(best_jsd, 6),
        second_jsd=round(second_jsd, 6),
        jsd_ratio=round(jsd_ratio, 4),
        affix_alignments=[asdict(a) for a in alignments],
        alignment_consistency=round(consistency, 4),
        null_jsd_mean=round(null_mean, 6),
        null_jsd_std=round(null_std, 6),
        real_vs_null_z=round(null_z, 2),
        jsd_separation_met=jsd_sep_ok,
        consistency_met=cons_ok,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    # Save
    out_path = os.path.join(_results_dir(), 'paradigm_match.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return asdict(result)
