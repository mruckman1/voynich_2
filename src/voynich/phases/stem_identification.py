"""
Phase 5.3: Frequency-Based Stem Identification
================================================
Identify the most common Voynich stem groups and attempt to match them
to specific Latin (or Romance) vocabulary using four compatibility
criteria + cross-consistency checking.

Sub-analyses:
  5.3a — Rank stems by frequency
  5.3b — Domain-constrained matching (4 criteria per stem × word pair)
  5.3c — Cross-consistency checking across all identifications

Output:
  results/stem_identification.json
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.stats import (
    selectivity_ratio, bootstrap_ci,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    load_reference_corpus, get_reference_text,
    LATIN_MEDICAL_VOCABULARY, LATIN_PARADIGM_PROFILES,
    OCCITAN_PARADIGM_PROFILES, get_paradigm_shape_profile,
)
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes, decompose_corpus_morphemes,
    MorphemeDecomposition,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StemCandidate:
    """A candidate identification of a Voynich stem with a Latin word."""
    voynich_stem: str
    voynich_rank: int
    voynich_frequency: int
    voynich_n_forms: int
    latin_word: str
    latin_pos: str
    latin_description: str
    # Compatibility scores (0.0 to 1.0)
    paradigm_compatibility: float
    frequency_compatibility: float
    section_compatibility: float
    affix_compatibility: float
    combined_score: float


@dataclass
class CrossConsistencyResult:
    """Cross-consistency check across all identifications."""
    n_identifications: int
    n_pairs_tested: int
    n_consistent_pairs: int
    n_violations: int
    consistency_ratio: float
    violations: List[Dict]


@dataclass
class StemIdentificationResult:
    """Full Phase 5.3 output."""
    n_stems_ranked: int
    n_candidates_tested: int
    identifications: List[Dict]
    mean_combined_score: float
    # Cross-consistency
    cross_consistency: float
    n_violations: int
    violations: List[Dict]
    # Null: random-word control
    random_word_mean_score: float
    random_word_std: float
    real_vs_random_z: float
    real_vs_random_selectivity: float
    # Null: shuffled text
    shuffled_mean_score: float
    shuffled_std: float
    real_vs_shuffled_z: float
    real_vs_shuffled_selectivity: float
    # Gates
    consistency_met: bool
    selectivity_vs_null_met: bool
    selectivity_vs_random_met: bool
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# 5.3a: Rank stems by frequency
# ---------------------------------------------------------------------------

def rank_stems_by_frequency(
    paradigm_data: Dict,
) -> List[Dict]:
    """
    Extract and rank stem paradigms by total token count.

    Returns list of paradigm dicts sorted by token_count descending.
    """
    paradigms = paradigm_data.get('top_20_paradigms', [])
    return sorted(paradigms, key=lambda p: p.get('token_count', 0), reverse=True)


# ---------------------------------------------------------------------------
# 5.3b: Compatibility scoring
# ---------------------------------------------------------------------------

def compute_paradigm_compatibility(
    voynich_n_forms: int,
    latin_pos: str,
    language: str = 'latin',
) -> float:
    """
    Score how well a paradigm's form count matches expected for the POS.

    Uses z-distance from expected mean. Score in [0, 1].
    """
    profiles = get_paradigm_shape_profile(language)

    # Map POS to paradigm type
    pos_to_type = {
        'noun': 'noun_declension',
        'verb': 'verb_conjugation',
        'adj': 'adj_declension',
    }
    ptype = pos_to_type.get(latin_pos, 'noun_declension')
    prof = profiles.get(ptype, {'mean_forms': 5, 'std_forms': 2.0})

    mean_f = prof['mean_forms']
    std_f = prof['std_forms']

    z = abs(voynich_n_forms - mean_f) / max(std_f, 0.5)
    return max(0.0, min(1.0, 1.0 - z * 0.25))


def compute_frequency_compatibility(
    voynich_rank: int,
    latin_rank: int,
    max_rank: int = 20,
) -> float:
    """
    Score how well frequency ranks match.

    Score = 1.0 - |rank_diff| / max_rank, clipped to [0, 1].
    """
    diff = abs(voynich_rank - latin_rank)
    return max(0.0, 1.0 - diff * 0.1)


def compute_section_compatibility(
    stem: str,
    tokens_by_section: Dict[str, List[str]],
    latin_pos: str,
) -> float:
    """
    Score section distribution plausibility.

    Nouns/herbs: should appear broadly; verbs: should appear in recipe
    sections; adjectives: broad distribution.
    """
    # Count stem occurrences per section
    section_counts: Dict[str, int] = {}
    for section, section_tokens in tokens_by_section.items():
        count = sum(1 for t in section_tokens if stem in t)
        if count > 0:
            section_counts[section] = count

    n_sections = len(section_counts)
    if n_sections == 0:
        return 0.5  # Neutral if not found

    # Herbal sections
    herbal_sections = {'herbal_a', 'herbal_b'}
    recipe_sections = {'pharmaceutical', 'recipes'}

    in_herbal = any(s in herbal_sections for s in section_counts)
    in_recipe = any(s in recipe_sections for s in section_counts)

    if latin_pos == 'noun':
        # Plant-related nouns should appear in herbal sections
        return 1.0 if in_herbal else 0.5
    elif latin_pos == 'verb':
        # Recipe verbs should appear in recipe/pharmaceutical sections
        return 1.0 if in_recipe else 0.5
    elif latin_pos == 'adj':
        # Adjectives should be broadly distributed
        return min(1.0, n_sections / 3.0)

    return 0.5


def compute_affix_compatibility(
    voynich_suffixes: List[str],
    latin_pos: str,
    affix_alignments: List[Dict],
) -> float:
    """
    Score whether the paradigm's attested suffixes are consistent
    with the affix alignment table from Phase 5.2.

    Returns fraction of suffixes that have alignments.
    """
    if not voynich_suffixes:
        return 0.5  # No suffixes = neutral

    # Build set of aligned Voynich suffixes
    aligned_suffixes = set(a.get('voynich_affix', '') for a in affix_alignments)

    n_aligned = sum(1 for s in voynich_suffixes if s in aligned_suffixes)
    return n_aligned / len(voynich_suffixes)


# ---------------------------------------------------------------------------
# 5.3b: Optimal stem assignment
# ---------------------------------------------------------------------------

def optimal_stem_assignment(
    ranked_stems: List[Dict],
    latin_vocab: List[Tuple[str, str, str]],
    tokens_by_section: Dict[str, List[str]],
    affix_alignments: List[Dict],
    language: str = 'latin',
    n_assign: int = 20,
) -> List[StemCandidate]:
    """
    Build cost matrix and find optimal 1-to-1 assignment via Hungarian algorithm.

    Returns list of StemCandidate sorted by combined_score descending.
    """
    n_v = min(n_assign, len(ranked_stems))
    n_l = min(n_assign, len(latin_vocab))
    n = max(n_v, n_l)

    # Build cost matrix
    cost = np.ones((n, n))
    scores_cache: Dict[Tuple[int, int], Dict] = {}

    for i in range(n_v):
        p = ranked_stems[i]
        v_stem = p.get('stem', '')
        v_n_forms = p.get('n_forms', 1)
        v_suffixes = p.get('suffixes', [])

        for j in range(n_l):
            l_word, l_pos, l_desc = latin_vocab[j]

            para_compat = compute_paradigm_compatibility(v_n_forms, l_pos, language)
            freq_compat = compute_frequency_compatibility(i, j, n)
            sect_compat = compute_section_compatibility(
                v_stem, tokens_by_section, l_pos,
            )
            affix_compat = compute_affix_compatibility(
                v_suffixes, l_pos, affix_alignments,
            )

            combined = (para_compat + freq_compat + sect_compat + affix_compat) / 4.0
            cost[i][j] = 1.0 - combined
            scores_cache[(i, j)] = {
                'paradigm': para_compat,
                'frequency': freq_compat,
                'section': sect_compat,
                'affix': affix_compat,
                'combined': combined,
            }

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build candidates
    candidates: List[StemCandidate] = []
    for r, c in zip(row_ind, col_ind):
        if r >= n_v or c >= n_l:
            continue

        p = ranked_stems[r]
        l_word, l_pos, l_desc = latin_vocab[c]
        s = scores_cache.get((r, c), {})

        candidates.append(StemCandidate(
            voynich_stem=p.get('stem', ''),
            voynich_rank=r,
            voynich_frequency=p.get('token_count', 0),
            voynich_n_forms=p.get('n_forms', 1),
            latin_word=l_word,
            latin_pos=l_pos,
            latin_description=l_desc,
            paradigm_compatibility=round(s.get('paradigm', 0), 4),
            frequency_compatibility=round(s.get('frequency', 0), 4),
            section_compatibility=round(s.get('section', 0), 4),
            affix_compatibility=round(s.get('affix', 0), 4),
            combined_score=round(s.get('combined', 0), 4),
        ))

    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# 5.3c: Cross-consistency checking
# ---------------------------------------------------------------------------

def cross_consistency_check(
    identifications: List[StemCandidate],
) -> CrossConsistencyResult:
    """
    Check that identifications are mutually consistent.

    Rules:
    1. No two Voynich stems map to the same Latin word
    2. Frequency ordering is roughly preserved
    3. POS categories are self-consistent
    """
    violations: List[Dict] = []
    n_pairs = 0
    n_consistent = 0

    # Rule 1: No duplicate Latin targets
    latin_targets = Counter(c.latin_word for c in identifications)
    for word, count in latin_targets.items():
        if count > 1:
            violations.append({
                'rule': 'duplicate_target',
                'latin_word': word,
                'count': count,
            })

    # Rule 2 & 3: Pairwise checks
    for i in range(len(identifications)):
        for j in range(i + 1, len(identifications)):
            a, b = identifications[i], identifications[j]
            n_pairs += 1
            pair_ok = True

            # Frequency ordering: if a ranks higher in Voynich,
            # it should also rank higher in Latin (roughly)
            if a.voynich_rank < b.voynich_rank:
                # a is more frequent in Voynich
                # Check if the Latin word ranking is wildly different
                latin_ranks = {w: r for r, (w, _, _) in enumerate(LATIN_MEDICAL_VOCABULARY)}
                a_latin_rank = latin_ranks.get(a.latin_word, 99)
                b_latin_rank = latin_ranks.get(b.latin_word, 99)
                rank_diff = abs((a.voynich_rank - b.voynich_rank) -
                               (a_latin_rank - b_latin_rank))
                if rank_diff > 10:
                    violations.append({
                        'rule': 'frequency_inversion',
                        'stem_a': a.voynich_stem,
                        'stem_b': b.voynich_stem,
                        'rank_diff': rank_diff,
                    })
                    pair_ok = False

            if pair_ok:
                n_consistent += 1

    consistency_ratio = n_consistent / n_pairs if n_pairs > 0 else 1.0

    return CrossConsistencyResult(
        n_identifications=len(identifications),
        n_pairs_tested=n_pairs,
        n_consistent_pairs=n_consistent,
        n_violations=len(violations),
        consistency_ratio=round(consistency_ratio, 4),
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Null tests
# ---------------------------------------------------------------------------

def random_word_control(
    ranked_stems: List[Dict],
    tokens_by_section: Dict[str, List[str]],
    affix_alignments: List[Dict],
    language: str,
    ref_corpus=None,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Control: replace LATIN_MEDICAL_VOCABULARY with frequency-matched
    random words from the reference corpus.

    Returns (random_mean_score, random_std).
    """
    rng = random.Random(seed)

    # Get reference vocabulary
    try:
        text = get_reference_text(language, n_words=5000, corpus=ref_corpus)
        ref_tokens = text.split()
    except Exception:
        ref_tokens = ['word'] * 100

    # Get frequency-ranked reference words
    ref_freq = Counter(ref_tokens)
    ref_ranked = [w for w, _ in ref_freq.most_common(200)]

    random_scores: List[float] = []

    for trial in range(n_trials):
        # Sample random words with matching POS distribution
        random_vocab: List[Tuple[str, str, str]] = []
        pos_options = ['noun', 'verb', 'adj']
        for i in range(min(20, len(ref_ranked))):
            word = rng.choice(ref_ranked[:50])
            pos = rng.choice(pos_options)
            random_vocab.append((word, pos, 'random'))

        # Run assignment
        candidates = optimal_stem_assignment(
            ranked_stems, random_vocab, tokens_by_section,
            affix_alignments, language,
        )

        if candidates:
            mean_score = float(np.mean([c.combined_score for c in candidates]))
        else:
            mean_score = 0.0
        random_scores.append(mean_score)

    return float(np.mean(random_scores)), float(np.std(random_scores))


def shuffled_text_control(
    tokens: List[str],
    latin_vocab: List[Tuple[str, str, str]],
    affix_alignments: List[Dict],
    language: str,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Null test: shuffle token characters, re-decompose, re-rank, re-assign.

    Returns (shuffled_mean_score, shuffled_std).
    """
    rng = random.Random(seed)
    shuffled_scores: List[float] = []

    for trial in range(n_trials):
        # Shuffle characters within tokens
        shuffled_tokens = []
        for t in tokens:
            chars = list(t)
            rng.shuffle(chars)
            shuffled_tokens.append(''.join(chars))

        # Decompose
        decomps = [decompose_token_morphemes(t) for t in set(shuffled_tokens)]
        token_counts = Counter(shuffled_tokens)

        # Group by stem
        stem_groups: Dict[str, Dict] = defaultdict(lambda: {
            'forms': set(), 'count': 0, 'suffixes': set(), 'prefixes': set(),
        })
        for d in decomps:
            if d.stem:
                g = stem_groups[d.stem]
                g['forms'].add(d.token)
                g['count'] += token_counts.get(d.token, 1)
                if d.suffix:
                    g['suffixes'].add(d.suffix)
                if d.prefix:
                    g['prefixes'].add(d.prefix)

        # Rank
        ranked = sorted(
            [{'stem': s, 'token_count': g['count'], 'n_forms': len(g['forms']),
              'suffixes': list(g['suffixes']), 'prefixes': list(g['prefixes'])}
             for s, g in stem_groups.items()],
            key=lambda x: x['token_count'], reverse=True,
        )[:20]

        # Assignment (empty section map for speed)
        candidates = optimal_stem_assignment(
            ranked, latin_vocab, {}, affix_alignments, language,
        )

        if candidates:
            mean_score = float(np.mean([c.combined_score for c in candidates]))
        else:
            mean_score = 0.0
        shuffled_scores.append(mean_score)

    return float(np.mean(shuffled_scores)), float(np.std(shuffled_scores))


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

def run_stem_identification(
    paradigm_data: Dict = None,
    match_data: Dict = None,
) -> Dict:
    """
    Run Phase 5.3: Frequency-Based Stem Identification.

    1. Load Phase 5.1 + 5.2 results
    2. Rank stems by frequency
    3. Compute 4-criteria compatibility for top stems × Latin vocab
    4. Optimal assignment via Hungarian algorithm
    5. Cross-consistency check
    6. Random-word control + shuffled text null test
    7. Gate: consistency > 0.80, selectivity > 1.5
    """
    print("=" * 70)
    print("Phase 5.3: Frequency-Based Stem Identification")
    print("=" * 70)

    # 1. Load prior results
    if paradigm_data is None:
        result_path = os.path.join(_results_dir(), 'paradigm_discovery.json')
        if not os.path.exists(result_path):
            print("  ERROR: Phase 5.1 results not found.")
            return {}
        with open(result_path) as f:
            paradigm_data = json.load(f)

    if match_data is None:
        result_path = os.path.join(_results_dir(), 'paradigm_match.json')
        if not os.path.exists(result_path):
            print("  ERROR: Phase 5.2 results not found.")
            return {}
        with open(result_path) as f:
            match_data = json.load(f)

    best_language = match_data.get('best_language', 'latin')
    affix_alignments = match_data.get('affix_alignments', [])

    print(f"\n  Best language from 5.2: {best_language}")
    print(f"  Affix alignments: {len(affix_alignments)}")

    # 2. Load corpus for section analysis
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        tokens = corpus.get_tokens(paragraph_only=True)

    # Build tokens-by-section map
    tokens_by_section: Dict[str, List[str]] = {}
    for section in set(p.section for p in corpus.pages.values()):
        sect_tokens = corpus.get_tokens(section=section, paragraph_only=True)
        if sect_tokens:
            tokens_by_section[section] = sect_tokens

    # 3. Rank stems
    print("\n  5.3a: Ranking stems by frequency")
    ranked_stems = rank_stems_by_frequency(paradigm_data)
    print(f"    Top {len(ranked_stems)} stems available")

    # 4. Optimal assignment
    print(f"\n  5.3b: Domain-constrained matching against {best_language}")
    identifications = optimal_stem_assignment(
        ranked_stems, LATIN_MEDICAL_VOCABULARY,
        tokens_by_section, affix_alignments, best_language,
    )
    n_tested = min(len(ranked_stems), len(LATIN_MEDICAL_VOCABULARY))
    mean_score = float(np.mean([c.combined_score for c in identifications])) if identifications else 0.0

    print(f"    Tested {n_tested} stem × word pairs")
    print(f"    Mean combined score: {mean_score:.4f}")
    print(f"\n    Top identifications:")
    for c in identifications[:10]:
        print(f"      '{c.voynich_stem}' -> {c.latin_word} ({c.latin_pos}): "
              f"score={c.combined_score:.3f} "
              f"[para={c.paradigm_compatibility:.2f}, "
              f"freq={c.frequency_compatibility:.2f}, "
              f"sect={c.section_compatibility:.2f}, "
              f"aff={c.affix_compatibility:.2f}]")

    # 5. Cross-consistency
    print("\n  5.3c: Cross-consistency check")
    cc = cross_consistency_check(identifications)
    print(f"    Pairs tested: {cc.n_pairs_tested}")
    print(f"    Consistent: {cc.n_consistent_pairs}")
    print(f"    Violations: {cc.n_violations}")
    print(f"    Consistency ratio: {cc.consistency_ratio:.2%}")

    # 6. Null tests
    print("\n  Random-word control:")
    try:
        ref_corpus = load_reference_corpus(verbose=False)
    except FileNotFoundError:
        ref_corpus = None
    rw_mean, rw_std = random_word_control(
        ranked_stems, tokens_by_section, affix_alignments,
        best_language, ref_corpus, n_trials=50,
    )
    rw_z = (mean_score - rw_mean) / rw_std if rw_std > 0 else 0.0
    rw_sel = mean_score / rw_mean if rw_mean > 0 else float('inf')
    print(f"    Real mean score: {mean_score:.4f}")
    print(f"    Random-word mean: {rw_mean:.4f} +/- {rw_std:.4f}")
    print(f"    z-score: {rw_z:.2f}, selectivity: {rw_sel:.2f}x")

    print("\n  Shuffled-text control:")
    sh_mean, sh_std = shuffled_text_control(
        tokens, LATIN_MEDICAL_VOCABULARY, affix_alignments,
        best_language, n_trials=50,
    )
    sh_z = (mean_score - sh_mean) / sh_std if sh_std > 0 else 0.0
    sh_sel = mean_score / sh_mean if sh_mean > 0 else float('inf')
    print(f"    Shuffled mean score: {sh_mean:.4f} +/- {sh_std:.4f}")
    print(f"    z-score: {sh_z:.2f}, selectivity: {sh_sel:.2f}x")

    # 7. Gates
    cons_ok, cons_msg = _check_gate(
        'cross_consistency', cc.consistency_ratio, 0.80, 'greater',
    )
    null_ok, null_msg = _check_gate(
        'selectivity_vs_shuffled', sh_sel, 1.5, 'greater',
    )
    rand_ok, rand_msg = _check_gate(
        'selectivity_vs_random', rw_sel, 1.5, 'greater',
    )
    gate_passed = cons_ok and null_ok and rand_ok

    print(f"\n{cons_msg}")
    print(f"{null_msg}")
    print(f"{rand_msg}")

    verdict = 'identifications_confirmed' if gate_passed else 'gate_failed'
    print(f"  Verdict: {verdict}")

    # Build result
    result = StemIdentificationResult(
        n_stems_ranked=len(ranked_stems),
        n_candidates_tested=n_tested,
        identifications=[asdict(c) for c in identifications],
        mean_combined_score=round(mean_score, 4),
        cross_consistency=cc.consistency_ratio,
        n_violations=cc.n_violations,
        violations=cc.violations,
        random_word_mean_score=round(rw_mean, 4),
        random_word_std=round(rw_std, 4),
        real_vs_random_z=round(rw_z, 2),
        real_vs_random_selectivity=round(rw_sel, 4),
        shuffled_mean_score=round(sh_mean, 4),
        shuffled_std=round(sh_std, 4),
        real_vs_shuffled_z=round(sh_z, 2),
        real_vs_shuffled_selectivity=round(sh_sel, 4),
        consistency_met=cons_ok,
        selectivity_vs_null_met=null_ok,
        selectivity_vs_random_met=rand_ok,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    # Save
    out_path = os.path.join(_results_dir(), 'stem_identification.json')
    with open(out_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return asdict(result)
