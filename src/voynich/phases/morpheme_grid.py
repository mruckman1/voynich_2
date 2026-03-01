"""
Phase 4.5 Priority B: Morpheme Grid Reinterpretation
=====================================================
Test whether the syllabary grid axes map to morphological structure
(stem + affix) rather than phonetic structure (consonant + vowel).

Rationale:
  Findings #3/#4: Voynich tokens decompose into low-entropy affixes
  (grammatical wrappers) and high-entropy stems (semantic content).
  The Phase 4 abugida test found nucleus predicts onset (R = 0.61)
  rather than vice versa. If grid axes map to morpheme roles, this
  reverse R is expected: stems constrain which affixes attach.

Sub-analyses:
  B.1 — Morpheme decomposition (prefix + stem + suffix)
  B.2 — Map morpheme components to grid axes (contingency table test)
  B.3 — Cross-validate with entropy decomposition
  B.4 — Reinterpret reverse R with morphological labels
  B.5 — Entropy stripping test (full tokens vs stems only)

Output:
  results/morpheme_grid.json
"""

import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2_contingency

from voynich.core.corpus import (
    load_corpus, VoynichCorpus,
    tokenize_eva_chars,
)
from voynich.core.stats import (
    first_order_entropy, conditional_entropy,
    word_unigram_entropy,
    jensen_shannon_divergence,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.analysis.strokes import (
    decompose_glyph, Stroke, SyllabaryGrid,
)
from voynich.phases.grid_validate import build_grid_from_tokens
from voynich.phases.abugida_test import (
    decompose_tokens_onset_nucleus, compute_onset_nucleus_entropy,
)


# ---------------------------------------------------------------------------
# Morpheme inventory (from prior project Phases 6/7)
# ---------------------------------------------------------------------------

# Known prefixes, sorted longest-first for greedy matching.
# These are initial-biased EVA sequences identified through distributional
# analysis (entropy drops at these positions).
KNOWN_PREFIXES = sorted([
    'qot', 'qok', 'qo',   # q-initial ligatures
    'ch', 'sh',             # aspirate onsets
    'd', 's', 'y', 'o',    # single-char prefixes
], key=len, reverse=True)

# Known suffixes, sorted longest-first for greedy matching.
# These are final-biased EVA sequences (terminal descenders, flourishes, nasals).
KNOWN_SUFFIXES = sorted([
    'aiiin', 'aiin',        # long nasal chains
    'iiin', 'iin',          # medium nasal chains
    'dy', 'ey',             # gallows + descender
    'ol', 'al',             # loop + lateral
    'in', 'an', 'am',       # nasal endings
    'y', 'n', 'm',          # single-char suffixes
], key=len, reverse=True)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MorphemeDecomposition:
    """Decomposition of one token into prefix + stem + suffix."""
    token: str
    prefix: str
    stem: str
    suffix: str
    prefix_glyphs: List[str]
    stem_glyphs: List[str]
    suffix_glyphs: List[str]


@dataclass
class MorphemeStats:
    """Corpus-level morpheme decomposition statistics."""
    n_tokens: int
    n_with_prefix: int
    n_with_suffix: int
    n_with_both: int
    n_stem_only: int
    pct_with_prefix: float
    pct_with_suffix: float
    pct_with_both: float
    mean_stem_length: float
    n_stem_types: int
    n_prefix_types: int
    n_suffix_types: int
    prefix_distribution: Dict[str, int]
    suffix_distribution: Dict[str, int]


@dataclass
class ContingencyResult:
    """Chi-squared test on morpheme role × grid axis association."""
    # Per-axis chi-squared: rows = {affix, stem}, cols = stroke types
    onset_chi2: float
    onset_p_value: float
    nucleus_chi2: float
    nucleus_p_value: float
    # Entropy of stroke distributions per role per axis
    h_affix_onset: float
    h_affix_nucleus: float
    h_stem_onset: float
    h_stem_nucleus: float
    # JSD between affix and stem distributions per axis
    jsd_onset: float
    jsd_nucleus: float
    # Which axis differentiates roles more?
    differentiating_axis: str
    morpheme_hypothesis_supported: bool


@dataclass
class EntropyStrippingResult:
    """Comparison of full-token vs stem-only entropy."""
    h1_full: float
    h1_stems: float
    h2_full: float
    h2_stems: float
    word_h1_full: float
    word_h1_stems: float
    h2_increased: bool
    closest_latin_level: str  # 'character', 'syllable', or 'word'


@dataclass
class MorphemeGridResult:
    """Full Priority B output."""
    stats: MorphemeStats
    contingency: ContingencyResult
    # Axis identification
    onset_axis_role: str       # 'affix' or 'stem'
    nucleus_axis_role: str     # 'affix' or 'stem'
    onset_axis_entropy: float
    nucleus_axis_entropy: float
    # Reinterpreted R values
    r_affix_given_stem: float
    r_stem_given_affix: float
    r_pattern_natural: bool
    # Entropy stripping
    stripping: EntropyStrippingResult
    # Null test (per axis)
    null_onset: Tuple[float, float, float]   # mean, std, z
    null_nucleus: Tuple[float, float, float]
    verdict: str


# ---------------------------------------------------------------------------
# B.1: Morpheme decomposition
# ---------------------------------------------------------------------------

def decompose_token_morphemes(token: str) -> MorphemeDecomposition:
    """
    Decompose an EVA token into prefix + stem + suffix.

    Rules:
    1. Tokenize into EVA chars via tokenize_eva_chars()
    2. Greedy match prefix from the start (longest-first from KNOWN_PREFIXES)
    3. Greedy match suffix from the end (longest-first from KNOWN_SUFFIXES)
    4. Everything in between is the stem
    5. If prefix+suffix would consume the entire token with no stem,
       treat the whole thing as stem (no decomposition)
    """
    eva_chars = tokenize_eva_chars(token)

    if not eva_chars:
        return MorphemeDecomposition(
            token=token, prefix='', stem=token, suffix='',
            prefix_glyphs=[], stem_glyphs=[], suffix_glyphs=[],
        )

    # Try to match prefix (single EVA char that matches a known prefix)
    prefix_glyphs = []
    prefix_str = ''
    remaining = list(eva_chars)

    for pfx in KNOWN_PREFIXES:
        # Check if the beginning of remaining chars forms this prefix
        # A prefix can be a single EVA char (like 'qo') or multi-char sequence
        candidate = ''.join(remaining[:len(pfx)])
        if candidate == pfx:
            # Check if this is a single EVA token from the ligature table
            # or a sequence of EVA chars that spells the prefix
            pfx_chars = tokenize_eva_chars(pfx)
            n_consumed = len(pfx_chars)
            if n_consumed <= len(remaining):
                # Verify the consumed chars actually match
                consumed = remaining[:n_consumed]
                if ''.join(consumed) == pfx:
                    prefix_glyphs = consumed
                    prefix_str = pfx
                    remaining = remaining[n_consumed:]
                    break

    # Try to match suffix from the end
    suffix_glyphs = []
    suffix_str = ''
    for sfx in KNOWN_SUFFIXES:
        sfx_chars = tokenize_eva_chars(sfx)
        n_consumed = len(sfx_chars)
        if n_consumed <= len(remaining):
            candidate = remaining[-n_consumed:]
            if ''.join(candidate) == sfx:
                suffix_glyphs = candidate
                suffix_str = sfx
                remaining = remaining[:-n_consumed]
                break

    # Remaining is the stem
    stem_glyphs = remaining
    stem_str = ''.join(remaining)

    # If no stem remains, treat the whole token as stem
    if not stem_str and (prefix_str or suffix_str):
        return MorphemeDecomposition(
            token=token, prefix='', stem=token, suffix='',
            prefix_glyphs=[], stem_glyphs=list(eva_chars), suffix_glyphs=[],
        )

    return MorphemeDecomposition(
        token=token,
        prefix=prefix_str,
        stem=stem_str,
        suffix=suffix_str,
        prefix_glyphs=prefix_glyphs,
        stem_glyphs=stem_glyphs,
        suffix_glyphs=suffix_glyphs,
    )


def decompose_corpus_morphemes(
    tokens: List[str],
) -> Tuple[List[MorphemeDecomposition], MorphemeStats]:
    """Decompose all tokens and compute corpus-level statistics."""
    decompositions = [decompose_token_morphemes(t) for t in tokens]

    n_with_prefix = sum(1 for d in decompositions if d.prefix)
    n_with_suffix = sum(1 for d in decompositions if d.suffix)
    n_with_both = sum(1 for d in decompositions if d.prefix and d.suffix)
    n_stem_only = sum(1 for d in decompositions if not d.prefix and not d.suffix)

    stem_lengths = [len(d.stem_glyphs) for d in decompositions]
    mean_stem_len = float(np.mean(stem_lengths)) if stem_lengths else 0.0

    prefix_counter = Counter(d.prefix for d in decompositions if d.prefix)
    suffix_counter = Counter(d.suffix for d in decompositions if d.suffix)
    stem_types = set(d.stem for d in decompositions)

    stats = MorphemeStats(
        n_tokens=len(tokens),
        n_with_prefix=n_with_prefix,
        n_with_suffix=n_with_suffix,
        n_with_both=n_with_both,
        n_stem_only=n_stem_only,
        pct_with_prefix=round(n_with_prefix / len(tokens) * 100, 1) if tokens else 0,
        pct_with_suffix=round(n_with_suffix / len(tokens) * 100, 1) if tokens else 0,
        pct_with_both=round(n_with_both / len(tokens) * 100, 1) if tokens else 0,
        mean_stem_length=round(mean_stem_len, 2),
        n_stem_types=len(stem_types),
        n_prefix_types=len(prefix_counter),
        n_suffix_types=len(suffix_counter),
        prefix_distribution=dict(prefix_counter.most_common()),
        suffix_distribution=dict(suffix_counter.most_common()),
    )
    return decompositions, stats


# ---------------------------------------------------------------------------
# B.2: Map morpheme components to grid axes
# ---------------------------------------------------------------------------

def map_morphemes_to_grid_axes(
    decompositions: List[MorphemeDecomposition],
) -> ContingencyResult:
    """
    Test whether affix and stem glyphs use different stroke distributions
    on the onset and nucleus axes.

    For each glyph in each morpheme role, collect the stroke it contributes
    to the onset axis (first stroke) and nucleus axis (last stroke). Then
    build a 2×K contingency table (affix vs stem × K stroke types) for
    each axis and run chi-squared. If one axis shows significantly different
    distributions between affix and stem roles, that axis carries the
    morphological distinction.
    """
    # Collect stroke values per role per axis
    affix_onset_strokes = []
    affix_nucleus_strokes = []
    stem_onset_strokes = []
    stem_nucleus_strokes = []

    for d in decompositions:
        for glyph in d.prefix_glyphs + d.suffix_glyphs:
            strokes = decompose_glyph(glyph)
            if strokes:
                affix_onset_strokes.append(strokes[0].name)
                affix_nucleus_strokes.append(strokes[-1].name)

        for glyph in d.stem_glyphs:
            strokes = decompose_glyph(glyph)
            if strokes:
                stem_onset_strokes.append(strokes[0].name)
                stem_nucleus_strokes.append(strokes[-1].name)

    def _stroke_entropy(items):
        if not items:
            return 0.0
        counts = Counter(items)
        total = len(items)
        return -sum((c / total) * math.log2(c / total)
                     for c in counts.values() if c > 0)

    h_affix_onset = _stroke_entropy(affix_onset_strokes)
    h_affix_nucleus = _stroke_entropy(affix_nucleus_strokes)
    h_stem_onset = _stroke_entropy(stem_onset_strokes)
    h_stem_nucleus = _stroke_entropy(stem_nucleus_strokes)

    # Build 2×K contingency table per axis and run chi-squared
    def _axis_chi2(role_a_strokes, role_b_strokes):
        """Chi-squared test: do two roles use different stroke distributions?"""
        all_types = sorted(set(role_a_strokes) | set(role_b_strokes))
        if len(all_types) < 2:
            return 0.0, 1.0
        count_a = Counter(role_a_strokes)
        count_b = Counter(role_b_strokes)
        table = np.array([
            [count_a.get(t, 0) for t in all_types],
            [count_b.get(t, 0) for t in all_types],
        ], dtype=float)
        # Add pseudocount to avoid zero cells
        table = table + 0.5
        try:
            chi2, p, _, _ = chi2_contingency(table)
            return chi2, p
        except ValueError:
            return 0.0, 1.0

    onset_chi2, onset_p = _axis_chi2(affix_onset_strokes, stem_onset_strokes)
    nucleus_chi2, nucleus_p = _axis_chi2(affix_nucleus_strokes, stem_nucleus_strokes)

    # JSD between affix and stem distributions per axis
    def _jsd(strokes_a, strokes_b):
        all_types = sorted(set(strokes_a) | set(strokes_b))
        if not all_types:
            return 0.0
        count_a = Counter(strokes_a)
        count_b = Counter(strokes_b)
        total_a = len(strokes_a) or 1
        total_b = len(strokes_b) or 1
        p = np.array([count_a.get(t, 0) / total_a for t in all_types])
        q = np.array([count_b.get(t, 0) / total_b for t in all_types])
        return jensen_shannon_divergence(p, q)

    jsd_onset = _jsd(affix_onset_strokes, stem_onset_strokes)
    jsd_nucleus = _jsd(affix_nucleus_strokes, stem_nucleus_strokes)

    # Which axis differentiates roles more?
    if jsd_onset > jsd_nucleus:
        diff_axis = 'onset'
    elif jsd_nucleus > jsd_onset:
        diff_axis = 'nucleus'
    else:
        diff_axis = 'neither'

    # Morpheme hypothesis supported if at least one axis shows significant
    # association (p < 0.01) between morpheme role and stroke distribution
    supported = onset_p < 0.01 or nucleus_p < 0.01

    return ContingencyResult(
        onset_chi2=round(onset_chi2, 4),
        onset_p_value=round(onset_p, 6),
        nucleus_chi2=round(nucleus_chi2, 4),
        nucleus_p_value=round(nucleus_p, 6),
        h_affix_onset=round(h_affix_onset, 4),
        h_affix_nucleus=round(h_affix_nucleus, 4),
        h_stem_onset=round(h_stem_onset, 4),
        h_stem_nucleus=round(h_stem_nucleus, 4),
        jsd_onset=round(jsd_onset, 6),
        jsd_nucleus=round(jsd_nucleus, 6),
        differentiating_axis=diff_axis,
        morpheme_hypothesis_supported=supported,
    )


# ---------------------------------------------------------------------------
# B.3: Cross-validate with entropy decomposition
# ---------------------------------------------------------------------------

def cross_validate_entropy(
    decompositions: List[MorphemeDecomposition],
    tokens: List[str],
) -> Dict:
    """
    Compare morpheme-axis identification with onset/nucleus entropy asymmetry.

    From the abugida test:
    - H(onset) = 1.9545 bits
    - H(nucleus) = 3.0012 bits
    Nucleus has higher entropy.

    If affix → onset axis and stem → nucleus axis:
    - Affix axis (onset) has LOWER entropy — correct for grammatical wrappers
    - Stem axis (nucleus) has HIGHER entropy — correct for semantic content
    This would be the linguistically natural assignment.
    """
    pairs = decompose_tokens_onset_nucleus(tokens)
    ent = compute_onset_nucleus_entropy(pairs)

    # Collect per-role entropy on each axis
    affix_onset_strokes = []
    affix_nucleus_strokes = []
    stem_onset_strokes = []
    stem_nucleus_strokes = []

    for d in decompositions:
        for glyph in d.prefix_glyphs + d.suffix_glyphs:
            strokes = decompose_glyph(glyph)
            if strokes:
                affix_onset_strokes.append(strokes[0].name)
                affix_nucleus_strokes.append(strokes[-1].name)
        for glyph in d.stem_glyphs:
            strokes = decompose_glyph(glyph)
            if strokes:
                stem_onset_strokes.append(strokes[0].name)
                stem_nucleus_strokes.append(strokes[-1].name)

    def _entropy(items):
        if not items:
            return 0.0
        counts = Counter(items)
        total = len(items)
        return -sum((c / total) * math.log2(c / total)
                     for c in counts.values() if c > 0)

    h_affix_onset = _entropy(affix_onset_strokes)
    h_affix_nucleus = _entropy(affix_nucleus_strokes)
    h_stem_onset = _entropy(stem_onset_strokes)
    h_stem_nucleus = _entropy(stem_nucleus_strokes)

    # Determine axis assignment
    # If onset has lower entropy overall → onset = affix axis
    if ent.h_onset < ent.h_nucleus:
        onset_role = 'affix'
        nucleus_role = 'stem'
    else:
        onset_role = 'stem'
        nucleus_role = 'affix'

    # Consistency check: does the per-role entropy match?
    # Affix strokes should have lower entropy than stem strokes on their axis
    if onset_role == 'affix':
        consistent = (h_affix_onset < h_stem_onset)
    else:
        consistent = (h_stem_onset < h_affix_onset)

    return {
        'h_onset_overall': round(ent.h_onset, 4),
        'h_nucleus_overall': round(ent.h_nucleus, 4),
        'h_affix_onset': round(h_affix_onset, 4),
        'h_affix_nucleus': round(h_affix_nucleus, 4),
        'h_stem_onset': round(h_stem_onset, 4),
        'h_stem_nucleus': round(h_stem_nucleus, 4),
        'onset_axis_role': onset_role,
        'nucleus_axis_role': nucleus_role,
        'entropy_pattern_consistent': consistent,
    }


# ---------------------------------------------------------------------------
# B.4: Reinterpret reverse R
# ---------------------------------------------------------------------------

def reinterpret_r_values(
    tokens: List[str],
    onset_role: str,
    nucleus_role: str,
) -> Dict:
    """
    Relabel onset/nucleus as affix/stem and check if R values become
    linguistically natural.

    From abugida test: R(nucleus|onset) = 0.3942, reverse_R(onset|nucleus) = 0.6054
    Meaning: knowing the nucleus predicts the onset strongly (0.61).

    If onset = affix and nucleus = stem:
      R(affix|stem) = reverse_R = 0.6054  → stems constrain affixes
      R(stem|affix) = R = 0.3942          → affixes partially constrain stems
      This is linguistically NATURAL (verb stems take verb endings)

    If onset = stem and nucleus = affix:
      R(stem|affix) = reverse_R = 0.6054  → affixes constrain stems (unusual)
      R(affix|stem) = R = 0.3942
      This is linguistically UNUSUAL
    """
    pairs = decompose_tokens_onset_nucleus(tokens)
    ent = compute_onset_nucleus_entropy(pairs)

    if onset_role == 'affix':
        r_affix_given_stem = ent.reverse_r   # R(onset|nucleus) = reverse_r
        r_stem_given_affix = ent.reduction_r  # R(nucleus|onset) = r
    else:
        r_affix_given_stem = ent.reduction_r
        r_stem_given_affix = ent.reverse_r

    # Linguistically natural: stem constrains affix more than vice versa
    # R(affix|stem) should be > R(stem|affix)
    natural = r_affix_given_stem > r_stem_given_affix

    return {
        'onset_role': onset_role,
        'nucleus_role': nucleus_role,
        'original_r': ent.reduction_r,
        'original_reverse_r': ent.reverse_r,
        'r_affix_given_stem': round(r_affix_given_stem, 4),
        'r_stem_given_affix': round(r_stem_given_affix, 4),
        'pattern_natural': natural,
        'interpretation': (
            f"Stems constrain affixes at R={r_affix_given_stem:.2f} — "
            f"{'linguistically natural' if natural else 'linguistically unusual'}"
        ),
    }


# ---------------------------------------------------------------------------
# B.5: Entropy stripping test
# ---------------------------------------------------------------------------

def entropy_stripping_test(
    tokens: List[str],
    decompositions: List[MorphemeDecomposition],
) -> EntropyStrippingResult:
    """
    Compare entropy of full tokens vs stripped stems.
    If affixes suppress entropy, H₂(stems) > H₂(full_tokens).
    """
    full_text = ' '.join(tokens)
    h1_full = first_order_entropy(full_text)
    h2_full = conditional_entropy(full_text, order=2)
    word_h1_full = word_unigram_entropy(tokens)

    # Stems only
    stems = [d.stem for d in decompositions if d.stem]
    stem_text = ' '.join(stems)
    h1_stems = first_order_entropy(stem_text) if stems else 0.0
    h2_stems = conditional_entropy(stem_text, order=2) if stems else 0.0
    word_h1_stems = word_unigram_entropy(stems) if stems else 0.0

    h2_increased = h2_stems > h2_full

    # Compare stem H₂ to Latin reference levels
    # Latin character-level H₂: ~2.3 bits
    # Latin syllable-level H₂: ~3.5 bits
    # Latin word-level H₂: ~1.4 bits
    latin_levels = {
        'character': 2.3,
        'syllable': 3.5,
        'word': 1.4,
    }
    closest = min(latin_levels, key=lambda k: abs(latin_levels[k] - h2_stems))

    return EntropyStrippingResult(
        h1_full=round(h1_full, 4),
        h1_stems=round(h1_stems, 4),
        h2_full=round(h2_full, 4),
        h2_stems=round(h2_stems, 4),
        word_h1_full=round(word_h1_full, 4),
        word_h1_stems=round(word_h1_stems, 4),
        h2_increased=h2_increased,
        closest_latin_level=closest,
    )


# ---------------------------------------------------------------------------
# Null testing
# ---------------------------------------------------------------------------

def null_test_morpheme_structure(
    decompositions: List[MorphemeDecomposition],
    real_jsd_onset: float,
    real_jsd_nucleus: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Null test: randomly reassign glyph-role labels within each token.

    For each token, collect all EVA glyphs and randomly split them into
    'affix' and 'stem' pools (preserving the original affix/stem counts).
    This breaks the positional structure (prefix=initial, suffix=final)
    while preserving corpus-level glyph counts.

    If the real morpheme-axis JSD is higher than null, the positional
    assignment carries real information about grid axis mapping.

    Returns dict with 'onset' and 'nucleus' keys, each (mean, std, z).
    """
    rng = random.Random(seed)

    null_jsd_onset = []
    null_jsd_nucleus = []

    for _ in range(n_trials):
        # Create shuffled decompositions: randomly partition each token's
        # glyphs into affix and stem roles
        shuffled = []
        for d in decompositions:
            all_glyphs = d.prefix_glyphs + d.stem_glyphs + d.suffix_glyphs
            n_affix = len(d.prefix_glyphs) + len(d.suffix_glyphs)

            if n_affix == 0 or n_affix >= len(all_glyphs):
                # No reassignment possible
                shuffled.append(d)
                continue

            shuffled_glyphs = list(all_glyphs)
            rng.shuffle(shuffled_glyphs)
            affix_g = shuffled_glyphs[:n_affix]
            stem_g = shuffled_glyphs[n_affix:]

            shuffled.append(MorphemeDecomposition(
                token=d.token,
                prefix=''.join(affix_g[:len(d.prefix_glyphs)]),
                stem=''.join(stem_g),
                suffix=''.join(affix_g[len(d.prefix_glyphs):]),
                prefix_glyphs=affix_g[:len(d.prefix_glyphs)],
                stem_glyphs=stem_g,
                suffix_glyphs=affix_g[len(d.prefix_glyphs):],
            ))

        result = map_morphemes_to_grid_axes(shuffled)
        null_jsd_onset.append(result.jsd_onset)
        null_jsd_nucleus.append(result.jsd_nucleus)

    def _z(real, null_arr):
        arr = np.array(null_arr)
        m = float(np.mean(arr))
        s = float(np.std(arr))
        return round(m, 6), round(s, 6), round((real - m) / s if s > 0 else 0.0, 2)

    return {
        'onset': _z(real_jsd_onset, null_jsd_onset),
        'nucleus': _z(real_jsd_nucleus, null_jsd_nucleus),
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_results(result: MorphemeGridResult) -> None:
    """Print formatted Priority B results."""
    s = result.stats
    c = result.contingency
    st = result.stripping

    print("\nMorpheme Grid Reinterpretation")
    print("=" * 65)

    print("\nMorpheme Decomposition:")
    print(f"  Tokens with prefix:  {s.pct_with_prefix:.1f}%")
    print(f"  Tokens with suffix:  {s.pct_with_suffix:.1f}%")
    print(f"  Tokens with both:    {s.pct_with_both:.1f}%")
    print(f"  Stem-only tokens:    {s.n_stem_only} ({s.n_stem_only / s.n_tokens * 100:.1f}%)")
    print(f"  Mean stem length:    {s.mean_stem_length:.1f} EVA characters")
    print(f"  Stem types:          {s.n_stem_types}")
    print(f"  Prefix types:        {s.n_prefix_types}")
    print(f"  Suffix types:        {s.n_suffix_types}")

    print(f"\n  Top prefixes: ", end='')
    for pfx, count in list(s.prefix_distribution.items())[:8]:
        print(f"{pfx}({count})", end='  ')
    print()

    print(f"  Top suffixes: ", end='')
    for sfx, count in list(s.suffix_distribution.items())[:8]:
        print(f"{sfx}({count})", end='  ')
    print()

    print(f"\nGrid Axis Mapping (stroke distribution per role):")
    print(f"  {'':>15s} {'H(onset)':>10s} {'H(nucleus)':>12s}")
    print(f"  {'Affix strokes':>15s} {c.h_affix_onset:>10.4f} {c.h_affix_nucleus:>12.4f}")
    print(f"  {'Stem strokes':>15s} {c.h_stem_onset:>10.4f} {c.h_stem_nucleus:>12.4f}")
    print(f"\n  Onset axis:   χ² = {c.onset_chi2:.2f}, p = {c.onset_p_value:.6f}")
    print(f"  Nucleus axis: χ² = {c.nucleus_chi2:.2f}, p = {c.nucleus_p_value:.6f}")
    print(f"  JSD(affix, stem) on onset:   {c.jsd_onset:.6f}")
    print(f"  JSD(affix, stem) on nucleus: {c.jsd_nucleus:.6f}")
    print(f"  Differentiating axis: {c.differentiating_axis.upper()}")
    print(f"  Morpheme hypothesis: "
          f"{'SUPPORTED' if c.morpheme_hypothesis_supported else 'NOT SUPPORTED'}")

    print(f"\nAxis Identification:")
    print(f"  Onset axis = {result.onset_axis_role.upper()} "
          f"(entropy {result.onset_axis_entropy:.4f} bits)")
    print(f"  Nucleus axis = {result.nucleus_axis_role.upper()} "
          f"(entropy {result.nucleus_axis_entropy:.4f} bits)")

    print(f"\n  Reinterpreted R values:")
    print(f"    R(affix | stem) = {result.r_affix_given_stem:.4f} "
          f"[expected: 0.4–0.7 for natural language]")
    print(f"    R(stem | affix) = {result.r_stem_given_affix:.4f} "
          f"[expected: 0.2–0.4]")
    print(f"    Pattern: {'LINGUISTICALLY NATURAL' if result.r_pattern_natural else 'UNUSUAL'}")

    print(f"\nEntropy Stripping:")
    print(f"  H₂(full tokens):  {st.h2_full:.4f} bits")
    print(f"  H₂(stems only):   {st.h2_stems:.4f} bits")
    delta = st.h2_stems - st.h2_full
    print(f"  Δ = {delta:+.4f} bits ({'INCREASE' if st.h2_increased else 'DECREASE'})")
    print(f"  Word H₁(full):    {st.word_h1_full:.4f} bits")
    print(f"  Word H₁(stems):   {st.word_h1_stems:.4f} bits")
    print(f"  H₂(stems) closest to Latin {st.closest_latin_level} level")

    print(f"\nNull Test (JSD with shuffled role assignments):")
    no = result.null_onset
    nn = result.null_nucleus
    print(f"  Onset axis:   null JSD = {no[0]:.6f} ± {no[1]:.6f}, z = {no[2]:.2f}")
    print(f"  Nucleus axis: null JSD = {nn[0]:.6f} ± {nn[1]:.6f}, z = {nn[2]:.2f}")

    print(f"\nVERDICT: {result.verdict.upper()}")
    if result.verdict == 'morphological':
        print("  -> Grid axes capture morphological structure (stem + affix)")
        print("  -> The reverse R is explained by stems constraining affixes")
    elif result.verdict == 'phonetic':
        print("  -> Grid axes capture phonetic structure (consonant + vowel)")
        print("  -> The reverse R remains anomalous")
    else:
        print("  -> Evidence is mixed; both interpretations partially supported")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_morpheme_grid() -> Dict:
    """Run Priority B: morpheme grid reinterpretation."""
    print("=" * 70)
    print("PHASE 4.5 PRIORITY B: MORPHEME GRID REINTERPRETATION")
    print("=" * 70)

    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)

    # B.1: Morpheme decomposition
    print("\nDecomposing tokens into morphemes...")
    decompositions, stats = decompose_corpus_morphemes(tokens)

    # B.2: Map to grid axes
    print("Mapping morpheme components to grid axes...")
    contingency = map_morphemes_to_grid_axes(decompositions)

    # B.3: Cross-validate with entropy
    print("Cross-validating with entropy decomposition...")
    entropy_xval = cross_validate_entropy(decompositions, tokens)

    # B.4: Reinterpret R
    print("Reinterpreting R values with morphological labels...")
    r_interp = reinterpret_r_values(
        tokens,
        entropy_xval['onset_axis_role'],
        entropy_xval['nucleus_axis_role'],
    )

    # B.5: Entropy stripping
    print("Running entropy stripping test...")
    stripping = entropy_stripping_test(tokens, decompositions)

    # Null test
    print("Running null test (100 trials)...")
    null_results = null_test_morpheme_structure(
        decompositions, contingency.jsd_onset, contingency.jsd_nucleus,
    )

    # Verdict
    morpheme_supported = contingency.morpheme_hypothesis_supported
    entropy_confirms = stripping.h2_increased
    r_natural = r_interp['pattern_natural']

    if morpheme_supported and entropy_confirms and r_natural:
        verdict = 'morphological'
    elif not morpheme_supported:
        verdict = 'phonetic'
    else:
        verdict = 'inconclusive'

    result = MorphemeGridResult(
        stats=stats,
        contingency=contingency,
        onset_axis_role=entropy_xval['onset_axis_role'],
        nucleus_axis_role=entropy_xval['nucleus_axis_role'],
        onset_axis_entropy=entropy_xval['h_onset_overall'],
        nucleus_axis_entropy=entropy_xval['h_nucleus_overall'],
        r_affix_given_stem=r_interp['r_affix_given_stem'],
        r_stem_given_affix=r_interp['r_stem_given_affix'],
        r_pattern_natural=r_natural,
        stripping=stripping,
        null_onset=null_results['onset'],
        null_nucleus=null_results['nucleus'],
        verdict=verdict,
    )

    _print_results(result)

    # Save
    rd = _results_dir()
    out_data = {
        'stats': asdict(stats),
        'contingency': asdict(contingency),
        'axis_identification': {
            'onset_axis_role': result.onset_axis_role,
            'nucleus_axis_role': result.nucleus_axis_role,
            'onset_axis_entropy': result.onset_axis_entropy,
            'nucleus_axis_entropy': result.nucleus_axis_entropy,
        },
        'r_reinterpretation': r_interp,
        'entropy_stripping': asdict(stripping),
        'entropy_crossvalidation': entropy_xval,
        'null_test': {
            'onset': {
                'null_mean': result.null_onset[0],
                'null_std': result.null_onset[1],
                'z_score': result.null_onset[2],
            },
            'nucleus': {
                'null_mean': result.null_nucleus[0],
                'null_std': result.null_nucleus[1],
                'z_score': result.null_nucleus[2],
            },
        },
        'verdict': verdict,
    }

    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    out_data = _convert(out_data)

    out_path = os.path.join(rd, 'morpheme_grid.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out_data
