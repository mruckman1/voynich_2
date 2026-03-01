"""
Stroke-Level Syllabary Analysis (Approach 1)
=============================================
Decompose EVA characters into constituent stroke elements, analyze
positional constraints, build a Ventris-style syllabary grid, and validate.

Phases:
  1.1 — EVA stroke decomposition (stroke primitives + decomposition table)
  1.2 — Positional analysis (P(stroke|position), MI, chi-squared)
  1.3 — Ventris grid construction (onset x nucleus grid)
  1.4 — Token-level syllabic segmentation
  1.5 — Discriminant validation (real vs null)
"""

import math
import os
import json
import random
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from corpus import (
    VoynichCorpus, load_corpus, tokenize_eva_chars,
    EVA_GLYPHS, EVA_LIGATURES_SORTED,
)


# ---------------------------------------------------------------------------
# Phase 1.1: Stroke Primitives & Decomposition Table
# ---------------------------------------------------------------------------

class Stroke(Enum):
    """
    Atomic stroke primitives observed in Voynich glyphs.

    These are the irreducible visual elements from which all EVA characters
    are composed. The inventory is derived from paleographic analysis of
    the manuscript's writing system.
    """
    LOOP = 'loop'                # Closed circular stroke (o-shape)
    OPEN_CURVE = 'open_curve'    # Open c-shaped curve
    VERTICAL = 'vertical'        # Straight vertical stroke (minim)
    HOOK = 'hook'                # Terminal hook/flourish (rightward curl)
    DESCENDER = 'descender'      # Stroke descending below baseline
    ASCENDER = 'ascender'        # Stroke ascending above x-height (gallows)
    CROSSBAR = 'crossbar'        # Horizontal stroke crossing a vertical
    SIGMOID = 'sigmoid'          # S-shaped curve
    PLUME = 'plume'              # Decorative extension on gallows glyphs
    CONNECTOR = 'connector'      # Linking/bridge element between strokes
    TAIL = 'tail'                # Short trailing element


@dataclass
class GlyphDecomposition:
    """Decomposition of one EVA character into ordered strokes."""
    eva: str
    strokes: Tuple[Stroke, ...]
    glyph_class: str = ''  # bench/gallows/minim/suffix
    notes: str = ''


# The core decomposition table
# Each EVA character -> ordered sequence of 1-3 stroke primitives
EVA_STROKE_TABLE: Dict[str, GlyphDecomposition] = {
    # --- Bench class (loop-based) ---
    'o': GlyphDecomposition('o', (Stroke.LOOP,),
                            'bench', 'simple closed loop'),
    'a': GlyphDecomposition('a', (Stroke.LOOP, Stroke.TAIL),
                            'bench', 'loop with trailing connector'),
    'e': GlyphDecomposition('e', (Stroke.LOOP, Stroke.LOOP),
                            'bench', 'double bench element'),
    'c': GlyphDecomposition('c', (Stroke.OPEN_CURVE,),
                            'bench', 'open c-curve'),
    'h': GlyphDecomposition('h', (Stroke.OPEN_CURVE, Stroke.VERTICAL, Stroke.CONNECTOR),
                            'bench', 'c-curve + vertical + connector'),
    'l': GlyphDecomposition('l', (Stroke.LOOP, Stroke.VERTICAL),
                            'bench', 'loop with minim extension'),
    'r': GlyphDecomposition('r', (Stroke.LOOP, Stroke.SIGMOID),
                            'bench', 'loop with flourish'),
    'x': GlyphDecomposition('x', (Stroke.CROSSBAR,),
                            'bench', 'rare cross-stroke'),
    'v': GlyphDecomposition('v', (Stroke.OPEN_CURVE, Stroke.HOOK),
                            'rare', 'curve with hook'),
    'w': GlyphDecomposition('w', (Stroke.OPEN_CURVE, Stroke.OPEN_CURVE),
                            'rare', 'double curve'),
    'z': GlyphDecomposition('z', (Stroke.SIGMOID, Stroke.HOOK),
                            'rare', 'sigmoid with hook'),

    # --- Gallows class (ascending strokes) ---
    'k': GlyphDecomposition('k', (Stroke.ASCENDER, Stroke.ASCENDER),
                            'gallows', 'double-post gallows'),
    't': GlyphDecomposition('t', (Stroke.ASCENDER, Stroke.ASCENDER, Stroke.CROSSBAR),
                            'gallows', 'gallows with crossbar'),
    'p': GlyphDecomposition('p', (Stroke.ASCENDER, Stroke.ASCENDER, Stroke.PLUME),
                            'gallows', 'gallows with plume'),
    'f': GlyphDecomposition('f', (Stroke.ASCENDER, Stroke.ASCENDER, Stroke.PLUME, Stroke.CROSSBAR),
                            'gallows', 'gallows with plume and crossbar'),
    'd': GlyphDecomposition('d', (Stroke.ASCENDER, Stroke.VERTICAL),
                            'gallows', 'tall prefix gallows'),
    's': GlyphDecomposition('s', (Stroke.SIGMOID,),
                            'gallows', 'initial sigmoid stroke'),
    'q': GlyphDecomposition('q', (Stroke.ASCENDER, Stroke.DESCENDER),
                            'gallows', 'gallows with descender'),

    # --- Minim class ---
    'i': GlyphDecomposition('i', (Stroke.VERTICAL,),
                            'minim', 'single minim stroke'),
    'g': GlyphDecomposition('g', (Stroke.VERTICAL, Stroke.ASCENDER),
                            'minim', 'rare minim with ascender'),

    # --- Suffix class ---
    'n': GlyphDecomposition('n', (Stroke.VERTICAL, Stroke.HOOK),
                            'suffix', 'minim with terminal hook'),
    'm': GlyphDecomposition('m', (Stroke.VERTICAL, Stroke.CONNECTOR, Stroke.VERTICAL),
                            'suffix', 'double minim with connector'),
    'y': GlyphDecomposition('y', (Stroke.VERTICAL, Stroke.DESCENDER),
                            'suffix', 'minim with descender'),
}

# Multi-character EVA ligatures decomposed at the stroke level
EVA_LIGATURE_STROKES: Dict[str, GlyphDecomposition] = {
    'sh': GlyphDecomposition('sh', (Stroke.SIGMOID, Stroke.OPEN_CURVE, Stroke.VERTICAL, Stroke.CONNECTOR),
                             'ligature', 's + h combined'),
    'ch': GlyphDecomposition('ch', (Stroke.OPEN_CURVE, Stroke.OPEN_CURVE, Stroke.VERTICAL, Stroke.CONNECTOR),
                             'ligature', 'c + h combined'),
    'cth': GlyphDecomposition('cth', (Stroke.OPEN_CURVE, Stroke.ASCENDER, Stroke.ASCENDER, Stroke.CROSSBAR,
                                       Stroke.OPEN_CURVE, Stroke.VERTICAL, Stroke.CONNECTOR),
                              'ligature', 'c + t + h combined'),
    'ckh': GlyphDecomposition('ckh', (Stroke.OPEN_CURVE, Stroke.ASCENDER, Stroke.ASCENDER,
                                       Stroke.OPEN_CURVE, Stroke.VERTICAL, Stroke.CONNECTOR),
                              'ligature', 'c + k + h combined'),
    'cph': GlyphDecomposition('cph', (Stroke.OPEN_CURVE, Stroke.ASCENDER, Stroke.ASCENDER, Stroke.PLUME,
                                       Stroke.OPEN_CURVE, Stroke.VERTICAL, Stroke.CONNECTOR),
                              'ligature', 'c + p + h combined'),
    'cfh': GlyphDecomposition('cfh', (Stroke.OPEN_CURVE, Stroke.ASCENDER, Stroke.ASCENDER, Stroke.PLUME,
                                       Stroke.CROSSBAR, Stroke.OPEN_CURVE, Stroke.VERTICAL, Stroke.CONNECTOR),
                              'ligature', 'c + f + h combined'),
    # Bench + minim combinations
    'ol': GlyphDecomposition('ol', (Stroke.LOOP, Stroke.LOOP, Stroke.VERTICAL),
                             'ligature', 'o + l combined'),
    'or': GlyphDecomposition('or', (Stroke.LOOP, Stroke.LOOP, Stroke.SIGMOID),
                             'ligature', 'o + r combined'),
    'al': GlyphDecomposition('al', (Stroke.LOOP, Stroke.TAIL, Stroke.LOOP, Stroke.VERTICAL),
                             'ligature', 'a + l combined'),
    'ar': GlyphDecomposition('ar', (Stroke.LOOP, Stroke.TAIL, Stroke.LOOP, Stroke.SIGMOID),
                             'ligature', 'a + r combined'),
    # Nasal sequences
    'iin': GlyphDecomposition('iin', (Stroke.VERTICAL, Stroke.VERTICAL, Stroke.VERTICAL, Stroke.HOOK),
                              'ligature', 'i + i + n combined'),
    'iiin': GlyphDecomposition('iiin', (Stroke.VERTICAL, Stroke.VERTICAL, Stroke.VERTICAL, Stroke.VERTICAL, Stroke.HOOK),
                               'ligature', 'i + i + i + n combined'),
    'aiin': GlyphDecomposition('aiin', (Stroke.LOOP, Stroke.TAIL, Stroke.VERTICAL, Stroke.VERTICAL, Stroke.VERTICAL, Stroke.HOOK),
                               'ligature', 'a + i + i + n combined'),
    'aiiin': GlyphDecomposition('aiiin', (Stroke.LOOP, Stroke.TAIL, Stroke.VERTICAL, Stroke.VERTICAL,
                                          Stroke.VERTICAL, Stroke.VERTICAL, Stroke.HOOK),
                                'ligature', 'a + i + i + i + n combined'),
    # q-initial sequences
    'qo': GlyphDecomposition('qo', (Stroke.ASCENDER, Stroke.DESCENDER, Stroke.LOOP),
                              'ligature', 'q + o combined'),
    'qok': GlyphDecomposition('qok', (Stroke.ASCENDER, Stroke.DESCENDER, Stroke.LOOP, Stroke.ASCENDER, Stroke.ASCENDER),
                               'ligature', 'q + o + k combined'),
    'qot': GlyphDecomposition('qot', (Stroke.ASCENDER, Stroke.DESCENDER, Stroke.LOOP,
                                       Stroke.ASCENDER, Stroke.ASCENDER, Stroke.CROSSBAR),
                               'ligature', 'q + o + t combined'),
    # Suffix combinations
    'dy': GlyphDecomposition('dy', (Stroke.ASCENDER, Stroke.VERTICAL, Stroke.VERTICAL, Stroke.DESCENDER),
                              'ligature', 'd + y combined'),
    'ey': GlyphDecomposition('ey', (Stroke.LOOP, Stroke.LOOP, Stroke.VERTICAL, Stroke.DESCENDER),
                              'ligature', 'e + y combined'),
}

# Combined lookup: ligatures first (longest match), then single chars
_ALL_DECOMPOSITIONS: Dict[str, GlyphDecomposition] = {}
_ALL_DECOMPOSITIONS.update(EVA_LIGATURE_STROKES)
_ALL_DECOMPOSITIONS.update(EVA_STROKE_TABLE)


def decompose_glyph(eva_char: str) -> Tuple[Stroke, ...]:
    """Get the stroke sequence for a single EVA character or ligature."""
    if eva_char in _ALL_DECOMPOSITIONS:
        return _ALL_DECOMPOSITIONS[eva_char].strokes
    return (Stroke.CONNECTOR,)  # unknown character


def decompose_token(token: str) -> List[Tuple[str, Tuple[Stroke, ...]]]:
    """
    Decompose an EVA token into a sequence of (eva_char, stroke_tuple) pairs.
    Uses longest-match-first to handle ligatures.

    Example: 'shody' -> [('sh', (SIGMOID, OPEN_CURVE, VERTICAL, CONNECTOR)),
                          ('o', (LOOP,)),
                          ('d', (ASCENDER, VERTICAL)),
                          ('y', (VERTICAL, DESCENDER))]
    """
    eva_chars = tokenize_eva_chars(token)
    result = []
    for ec in eva_chars:
        strokes = decompose_glyph(ec)
        result.append((ec, strokes))
    return result


def get_stroke_sequence(token: str) -> List[Stroke]:
    """Get the flat stroke sequence for a token."""
    parts = decompose_token(token)
    strokes = []
    for _, s in parts:
        strokes.extend(s)
    return strokes


def get_glyph_stroke_pairs(token: str) -> List[Tuple[str, Stroke]]:
    """Get (glyph_label, stroke) pairs for positional analysis.
    Each glyph contributes its strokes tagged with the glyph name."""
    parts = decompose_token(token)
    pairs = []
    for glyph, strokes in parts:
        for s in strokes:
            pairs.append((glyph, s))
    return pairs


# ---------------------------------------------------------------------------
# Phase 1.2: Positional Analysis of Stroke Elements
# ---------------------------------------------------------------------------

def stroke_positional_analysis(tokens: List[str]) -> Dict:
    """
    For each stroke type, compute P(stroke | position=initial/medial/final).
    Also compute MI(stroke_identity, position).

    A glyph's strokes inherit its word position:
    - First glyph in token -> 'initial'
    - Last glyph in token -> 'final'
    - All other glyphs -> 'medial'
    Singleton tokens -> 'singleton'

    Returns dict with:
      'stroke_positions': {stroke: {initial, medial, final, singleton, total}}
      'stroke_position_probs': {stroke: {initial, medial, final}}
      'mi_stroke_position': float
      'position_entropy_per_stroke': {stroke: H(position)}
    """
    # Count strokes at each glyph position
    stroke_pos = defaultdict(lambda: {'initial': 0, 'medial': 0, 'final': 0, 'singleton': 0})

    for token in tokens:
        glyphs = tokenize_eva_chars(token)
        if len(glyphs) == 0:
            continue

        for idx, glyph in enumerate(glyphs):
            strokes = decompose_glyph(glyph)

            if len(glyphs) == 1:
                pos = 'singleton'
            elif idx == 0:
                pos = 'initial'
            elif idx == len(glyphs) - 1:
                pos = 'final'
            else:
                pos = 'medial'

            for s in strokes:
                stroke_pos[s][pos] += 1

    # Compute probabilities and MI
    stroke_positions = {}
    stroke_probs = {}
    positions = ['initial', 'medial', 'final']

    # Grand totals for MI
    grand_total = 0
    pos_totals = defaultdict(int)

    for s, counts in stroke_pos.items():
        total = sum(counts.values())
        stroke_positions[s] = {**counts, 'total': total}
        grand_total += total

        pos_only = {p: counts[p] for p in positions}
        pos_sum = sum(pos_only.values())
        if pos_sum > 0:
            stroke_probs[s] = {p: pos_only[p] / pos_sum for p in positions}
        else:
            stroke_probs[s] = {p: 0.0 for p in positions}

        for p in positions:
            pos_totals[p] += counts[p]

    # MI(stroke, position) = sum p(s,p) * log2(p(s,p) / (p(s) * p(p)))
    mi = 0.0
    if grand_total > 0:
        for s, counts in stroke_pos.items():
            for p in positions:
                joint = counts[p]
                if joint == 0:
                    continue
                p_sp = joint / grand_total
                p_s = sum(counts[pp] for pp in positions) / grand_total
                p_p = pos_totals[p] / grand_total
                if p_s > 0 and p_p > 0:
                    mi += p_sp * math.log2(p_sp / (p_s * p_p))

    # Per-stroke positional entropy
    pos_entropy_per_stroke = {}
    for s, probs in stroke_probs.items():
        h = 0.0
        for p_val in probs.values():
            if p_val > 0:
                h -= p_val * math.log2(p_val)
        pos_entropy_per_stroke[s] = round(h, 4)

    return {
        'stroke_positions': {s.value: v for s, v in stroke_positions.items()},
        'stroke_position_probs': {s.value: v for s, v in stroke_probs.items()},
        'mi_stroke_position': round(mi, 6),
        'position_entropy_per_stroke': {s.value: v for s, v in pos_entropy_per_stroke.items()},
    }


def stroke_bigram_matrix(tokens: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Build stroke-level bigram transition matrix (within tokens).
    Returns (matrix, stroke_labels).
    """
    # Collect all within-token stroke bigrams
    stroke_types = sorted(set(s.value for s in Stroke))
    s_to_idx = {s: i for i, s in enumerate(stroke_types)}
    n = len(stroke_types)
    counts = np.zeros((n, n), dtype=float)

    for token in tokens:
        seq = get_stroke_sequence(token)
        for i in range(len(seq) - 1):
            s1 = seq[i].value
            s2 = seq[i + 1].value
            counts[s_to_idx[s1]][s_to_idx[s2]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = counts / row_sums

    return matrix, stroke_types


def stroke_position_chi_squared(
    analysis: Dict,
    null_model: str = 'random'
) -> Tuple[float, int]:
    """
    Chi-squared test of stroke positional distribution vs null model.

    Null models:
        'random'     — strokes are uniformly distributed across positions
        'alphabetic' — strokes follow English-like mild positional preference
        'syllabic'   — strokes follow strong onset/nucleus/coda separation

    Returns (chi2_statistic, degrees_of_freedom).
    """
    positions = ['initial', 'medial', 'final']
    stroke_data = analysis['stroke_positions']

    chi2 = 0.0
    df = 0

    for stroke_name, counts in stroke_data.items():
        total = sum(counts.get(p, 0) for p in positions)
        if total < 5:
            continue

        if null_model == 'random':
            # Uniform distribution
            expected = {p: total / 3.0 for p in positions}
        elif null_model == 'alphabetic':
            # Mild preference: 40% initial, 30% medial, 30% final
            expected = {'initial': total * 0.4, 'medial': total * 0.3, 'final': total * 0.3}
        elif null_model == 'syllabic':
            # Strong preference varies by stroke type (we use actual distribution shape)
            # For comparison: strong prefix strokes should be 70% initial
            # Use the overall position distribution as expected
            pos_total = sum(counts.get(p, 0) for p in positions
                           for counts in stroke_data.values())
            if pos_total == 0:
                continue
            expected = {}
            for p in positions:
                p_total = sum(d.get(p, 0) for d in stroke_data.values())
                expected[p] = total * (p_total / pos_total)
        else:
            raise ValueError(f"Unknown null model: {null_model}")

        for p in positions:
            observed = counts.get(p, 0)
            exp = expected[p]
            if exp > 0:
                chi2 += (observed - exp) ** 2 / exp
                df += 1

    df = max(0, df - len(stroke_data))  # subtract constraints
    return chi2, df


# ---------------------------------------------------------------------------
# Phase 1.3: Ventris Grid Construction
# ---------------------------------------------------------------------------

@dataclass
class SyllabaryGrid:
    """A consonant x vowel syllabary grid (Ventris-style)."""
    row_labels: List[str]       # Putative onset (initial stroke pattern) classes
    col_labels: List[str]       # Putative nucleus (final stroke pattern) classes
    cells: Dict[str, List[str]] = field(default_factory=dict)  # 'row,col' -> EVA chars
    occupancy: float = 0.0     # Fraction of cells filled
    n_filled: int = 0
    n_total: int = 0


def build_ventris_grid(tokens: List[str]) -> SyllabaryGrid:
    """
    Build a consonant x vowel grid from shared initial/final stroke patterns.

    Algorithm:
    1. For each unique glyph, extract its initial stroke(s) and final stroke(s)
    2. Cluster initial strokes -> putative onset classes (rows)
    3. Cluster final strokes -> putative nucleus classes (columns)
    4. Place each glyph in its grid cell
    5. Compute occupancy
    """
    # Collect all unique EVA glyphs from the corpus
    glyph_counter = Counter()
    for token in tokens:
        for g in tokenize_eva_chars(token):
            glyph_counter[g] += 1

    # For each glyph, determine initial and final stroke categories
    onset_map = {}   # glyph -> onset stroke(s)
    nucleus_map = {} # glyph -> final stroke(s)

    for glyph in glyph_counter:
        strokes = decompose_glyph(glyph)
        if not strokes:
            continue
        # Onset = first stroke, nucleus = last stroke
        onset_map[glyph] = strokes[0].value
        nucleus_map[glyph] = strokes[-1].value

    # Unique onset and nucleus categories
    onset_types = sorted(set(onset_map.values()))
    nucleus_types = sorted(set(nucleus_map.values()))

    # Build grid
    cells = {}
    for glyph in glyph_counter:
        if glyph not in onset_map or glyph not in nucleus_map:
            continue
        key = f"{onset_map[glyph]},{nucleus_map[glyph]}"
        if key not in cells:
            cells[key] = []
        cells[key].append(glyph)

    n_total = len(onset_types) * len(nucleus_types)
    n_filled = len(cells)
    occupancy = n_filled / n_total if n_total > 0 else 0.0

    return SyllabaryGrid(
        row_labels=onset_types,
        col_labels=nucleus_types,
        cells=cells,
        occupancy=occupancy,
        n_filled=n_filled,
        n_total=n_total,
    )


def compare_grid_to_reference(grid: SyllabaryGrid, reference: str = 'linear_b') -> Dict:
    """
    Compare grid occupancy pattern against known syllabary structures.

    Reference syllabaries:
        'linear_b' — ~12 onsets x ~5 nuclei, ~60% occupancy, 87 signs
        'hiragana' — ~10 onsets x ~5 nuclei, ~92% occupancy, 46 signs
        'cypriot'  — ~12 onsets x ~5 nuclei, ~55% occupancy, 56 signs
    """
    references = {
        'linear_b': {
            'n_onsets': 12, 'n_nuclei': 5, 'n_signs': 87,
            'occupancy': 0.60, 'description': 'Mycenaean Greek syllabary',
        },
        'hiragana': {
            'n_onsets': 10, 'n_nuclei': 5, 'n_signs': 46,
            'occupancy': 0.92, 'description': 'Japanese syllabary',
        },
        'cypriot': {
            'n_onsets': 12, 'n_nuclei': 5, 'n_signs': 56,
            'occupancy': 0.55, 'description': 'Ancient Cypriot syllabary',
        },
    }

    if reference not in references:
        raise ValueError(f"Unknown reference: {reference}. "
                        f"Available: {list(references.keys())}")

    ref = references[reference]
    n_unique_glyphs = sum(len(v) for v in grid.cells.values())

    return {
        'voynich': {
            'n_onsets': len(grid.row_labels),
            'n_nuclei': len(grid.col_labels),
            'n_signs': n_unique_glyphs,
            'occupancy': round(grid.occupancy, 4),
        },
        'reference': ref,
        'comparison': {
            'onset_ratio': len(grid.row_labels) / ref['n_onsets'],
            'nuclei_ratio': len(grid.col_labels) / ref['n_nuclei'],
            'sign_ratio': n_unique_glyphs / ref['n_signs'],
            'occupancy_diff': grid.occupancy - ref['occupancy'],
        },
    }


# ---------------------------------------------------------------------------
# Phase 1.4: Token-Level Syllabic Segmentation
# ---------------------------------------------------------------------------

def segment_token_as_syllables(
    token: str,
    grid: SyllabaryGrid,
) -> List[str]:
    """
    Re-segment a Voynich token as a sequence of syllable units from the grid.
    Each EVA glyph maps to a grid cell (onset, nucleus) = one syllable unit.
    """
    glyphs = tokenize_eva_chars(token)
    syllables = []
    for g in glyphs:
        strokes = decompose_glyph(g)
        if strokes:
            onset = strokes[0].value
            nucleus = strokes[-1].value
            key = f"{onset},{nucleus}"
            syllables.append(key)
    return syllables


def syllable_sequence_stats(
    tokens: List[str],
    grid: SyllabaryGrid,
) -> Dict:
    """
    Compute statistics over syllable sequences.
    """
    all_syllables = []
    syllable_counts_per_token = []

    for token in tokens:
        syls = segment_token_as_syllables(token, grid)
        all_syllables.extend(syls)
        syllable_counts_per_token.append(len(syls))

    if not all_syllables:
        return {'error': 'no syllables extracted'}

    syl_counter = Counter(all_syllables)
    n_types = len(syl_counter)
    n_tokens_syl = len(all_syllables)

    # Syllable-level entropy
    h1 = 0.0
    for count in syl_counter.values():
        p = count / n_tokens_syl
        if p > 0:
            h1 -= p * math.log2(p)

    # Syllable bigram conditional entropy
    bigrams = Counter()
    unigrams = Counter()
    for token in tokens:
        syls = segment_token_as_syllables(token, grid)
        for i in range(len(syls) - 1):
            bigrams[(syls[i], syls[i + 1])] += 1
            unigrams[syls[i]] += 1

    h2 = 0.0
    total_bi = sum(bigrams.values())
    total_uni = sum(unigrams.values())
    if total_bi > 0 and total_uni > 0:
        h_joint = -sum((c / total_bi) * math.log2(c / total_bi)
                       for c in bigrams.values() if c > 0)
        h_ctx = -sum((c / total_uni) * math.log2(c / total_uni)
                     for c in unigrams.values() if c > 0)
        h2 = h_joint - h_ctx

    return {
        'n_syllable_types': n_types,
        'n_syllable_tokens': n_tokens_syl,
        'syllable_ttr': n_types / n_tokens_syl if n_tokens_syl > 0 else 0,
        'syllable_h1': round(h1, 4),
        'syllable_h2': round(h2, 4),
        'mean_syllables_per_token': float(np.mean(syllable_counts_per_token)),
        'std_syllables_per_token': float(np.std(syllable_counts_per_token)),
        'top_20_syllables': syl_counter.most_common(20),
    }


# ---------------------------------------------------------------------------
# Phase 1.5: Discriminant Validation
# ---------------------------------------------------------------------------

def syllabary_discriminant_test(
    real_tokens: List[str],
    grid: SyllabaryGrid,
    n_shuffled: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Test whether the syllabary structure is statistically significant.

    Generates shuffled texts (character-level shuffle within tokens,
    preserving token lengths) and compares syllable statistics.

    If the grid captures genuine structure, real text should show lower
    syllable-level H2 (more predictable transitions) than shuffled text.
    """
    rng = random.Random(seed)

    # Real stats
    real_stats = syllable_sequence_stats(real_tokens, grid)

    # Shuffled stats
    shuffled_h1s = []
    shuffled_h2s = []

    for trial in range(n_shuffled):
        rng_trial = random.Random(seed + trial)
        shuffled = []
        for t in real_tokens:
            chars = list(t)
            rng_trial.shuffle(chars)
            shuffled.append(''.join(chars))

        s_stats = syllable_sequence_stats(shuffled, grid)
        shuffled_h1s.append(s_stats.get('syllable_h1', 0))
        shuffled_h2s.append(s_stats.get('syllable_h2', 0))

    real_h1 = real_stats.get('syllable_h1', 0)
    real_h2 = real_stats.get('syllable_h2', 0)
    mean_shuffled_h1 = float(np.mean(shuffled_h1s))
    mean_shuffled_h2 = float(np.mean(shuffled_h2s))

    # z-scores
    std_h1 = float(np.std(shuffled_h1s)) if np.std(shuffled_h1s) > 0 else 1.0
    std_h2 = float(np.std(shuffled_h2s)) if np.std(shuffled_h2s) > 0 else 1.0
    z_h1 = (real_h1 - mean_shuffled_h1) / std_h1
    z_h2 = (real_h2 - mean_shuffled_h2) / std_h2

    discriminates = abs(z_h2) > 2.0  # significant at ~95% level

    return {
        'real': real_stats,
        'shuffled_mean_h1': round(mean_shuffled_h1, 4),
        'shuffled_std_h1': round(std_h1, 4),
        'shuffled_mean_h2': round(mean_shuffled_h2, 4),
        'shuffled_std_h2': round(std_h2, 4),
        'z_score_h1': round(z_h1, 4),
        'z_score_h2': round(z_h2, 4),
        'discriminates': discriminates,
        'n_trials': n_shuffled,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_stroke_analysis():
    """Run the full stroke-level syllabary analysis pipeline."""
    print("=" * 70)
    print("APPROACH 1: STROKE-LEVEL SYLLABARY ANALYSIS")
    print("=" * 70)

    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)

    # --- Phase 1.1: Stroke Decomposition ---
    print("\n--- Phase 1.1: EVA Stroke Decomposition ---")
    print(f"\n  Stroke primitives: {len(Stroke)} types")
    for s in Stroke:
        print(f"    {s.value}")

    print(f"\n  EVA character decompositions: {len(EVA_STROKE_TABLE)} entries")
    print(f"  EVA ligature decompositions: {len(EVA_LIGATURE_STROKES)} entries")

    # Show sample decompositions
    print(f"\n  Sample token decompositions:")
    for token in tokens[:10]:
        parts = decompose_token(token)
        stroke_str = ' | '.join(
            f"{g}:[{','.join(s.value for s in strokes)}]"
            for g, strokes in parts
        )
        print(f"    {token:15s} -> {stroke_str}")

    # Stroke frequency in corpus
    stroke_counter = Counter()
    for token in tokens:
        for s in get_stroke_sequence(token):
            stroke_counter[s.value] += 1
    print(f"\n  Stroke frequency distribution ({sum(stroke_counter.values())} total):")
    for s, count in stroke_counter.most_common():
        pct = 100.0 * count / sum(stroke_counter.values())
        bar = '#' * int(pct)
        print(f"    {s:<14s} {count:6d} ({pct:5.1f}%) {bar}")

    # --- Phase 1.2: Positional Analysis ---
    print("\n--- Phase 1.2: Positional Analysis of Stroke Elements ---")
    pos_analysis = stroke_positional_analysis(tokens)

    print(f"\n  MI(stroke, position) = {pos_analysis['mi_stroke_position']:.6f}")
    print(f"  (Higher MI = stronger positional constraints = more syllabic)")

    print(f"\n  Stroke positional probabilities:")
    print(f"  {'Stroke':<14s} {'Initial':>8s} {'Medial':>8s} {'Final':>8s} {'H(pos)':>8s}")
    print(f"  {'-'*46}")
    for stroke_name in sorted(pos_analysis['stroke_position_probs'].keys()):
        probs = pos_analysis['stroke_position_probs'][stroke_name]
        h = pos_analysis['position_entropy_per_stroke'].get(stroke_name, 0)
        print(f"  {stroke_name:<14s} {probs['initial']:>8.3f} {probs['medial']:>8.3f} "
              f"{probs['final']:>8.3f} {h:>8.4f}")

    # Chi-squared tests
    for null in ['random', 'alphabetic']:
        chi2, df = stroke_position_chi_squared(pos_analysis, null_model=null)
        print(f"\n  Chi-squared vs {null}: {chi2:.2f} (df={df})")

    # Stroke bigram matrix
    bmat, labels = stroke_bigram_matrix(tokens)
    print(f"\n  Stroke bigram matrix: {len(labels)}x{len(labels)}")

    # Find strongest stroke transitions
    transitions = []
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if bmat[i, j] > 0.15:
                transitions.append((l1, l2, bmat[i, j]))
    transitions.sort(key=lambda x: -x[2])
    print(f"  Top stroke transitions (P > 0.15):")
    for s1, s2, p in transitions[:15]:
        print(f"    {s1} -> {s2}: {p:.3f}")

    # --- Phase 1.3: Ventris Grid ---
    print("\n--- Phase 1.3: Ventris Grid Construction ---")
    grid = build_ventris_grid(tokens)

    print(f"\n  Grid dimensions: {len(grid.row_labels)} onsets x {len(grid.col_labels)} nuclei")
    print(f"  Cells filled: {grid.n_filled} / {grid.n_total} ({grid.occupancy:.1%})")

    print(f"\n  Onset (row) categories: {grid.row_labels}")
    print(f"  Nucleus (col) categories: {grid.col_labels}")

    # Grid display
    print(f"\n  Grid contents:")
    header = 'Onset\\Nucleus'
    print(f"  {header:<14s}", end='')
    for col in grid.col_labels:
        print(f" {col:<12s}", end='')
    print()
    for row in grid.row_labels:
        print(f"  {row:<14s}", end='')
        for col in grid.col_labels:
            key = f"{row},{col}"
            glyphs = grid.cells.get(key, [])
            cell = ','.join(glyphs[:3]) if glyphs else '-'
            print(f" {cell:<12s}", end='')
        print()

    # Compare to references
    for ref in ['linear_b', 'hiragana', 'cypriot']:
        comp = compare_grid_to_reference(grid, ref)
        ref_data = comp['reference']
        diff = comp['comparison']
        print(f"\n  vs {ref} ({ref_data['description']}):")
        print(f"    Onsets: {len(grid.row_labels)} vs {ref_data['n_onsets']} "
              f"(ratio {diff['onset_ratio']:.2f})")
        print(f"    Nuclei: {len(grid.col_labels)} vs {ref_data['n_nuclei']} "
              f"(ratio {diff['nuclei_ratio']:.2f})")
        print(f"    Occupancy: {grid.occupancy:.2f} vs {ref_data['occupancy']:.2f} "
              f"(diff {diff['occupancy_diff']:+.2f})")

    # --- Phase 1.4: Token Segmentation ---
    print("\n--- Phase 1.4: Token-Level Syllabic Segmentation ---")
    syl_stats = syllable_sequence_stats(tokens, grid)

    print(f"\n  Syllable types: {syl_stats['n_syllable_types']}")
    print(f"  Syllable tokens: {syl_stats['n_syllable_tokens']}")
    print(f"  Syllable TTR: {syl_stats['syllable_ttr']:.4f}")
    print(f"  Syllable H1: {syl_stats['syllable_h1']:.4f}")
    print(f"  Syllable H2: {syl_stats['syllable_h2']:.4f}")
    print(f"  Mean syllables/token: {syl_stats['mean_syllables_per_token']:.2f}")

    print(f"\n  Top 20 syllable types:")
    for syl, count in syl_stats.get('top_20_syllables', []):
        print(f"    {syl}: {count}")

    # Sample segmentations
    print(f"\n  Sample token segmentations:")
    for token in tokens[:10]:
        syls = segment_token_as_syllables(token, grid)
        print(f"    {token:15s} -> {' . '.join(syls)}")

    # --- Phase 1.5: Discriminant Validation ---
    print("\n--- Phase 1.5: Discriminant Validation ---")
    disc = syllabary_discriminant_test(tokens, grid, n_shuffled=100)

    print(f"\n  Real syllable H1: {disc['real'].get('syllable_h1', 0):.4f}")
    print(f"  Real syllable H2: {disc['real'].get('syllable_h2', 0):.4f}")
    print(f"  Shuffled mean H1: {disc['shuffled_mean_h1']:.4f} "
          f"(+/- {disc['shuffled_std_h1']:.4f})")
    print(f"  Shuffled mean H2: {disc['shuffled_mean_h2']:.4f} "
          f"(+/- {disc['shuffled_std_h2']:.4f})")
    print(f"  Z-score H1: {disc['z_score_h1']:.4f}")
    print(f"  Z-score H2: {disc['z_score_h2']:.4f}")
    print(f"  Discriminates (|z_H2| > 2.0): {'YES' if disc['discriminates'] else 'NO'}")

    # --- Save results ---
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save positional analysis
    with open(os.path.join(results_dir, 'stroke_positional.json'), 'w') as f:
        json.dump(pos_analysis, f, indent=2)

    # Save grid
    grid_data = {
        'row_labels': grid.row_labels,
        'col_labels': grid.col_labels,
        'cells': grid.cells,
        'occupancy': grid.occupancy,
        'n_filled': grid.n_filled,
        'n_total': grid.n_total,
    }
    with open(os.path.join(results_dir, 'ventris_grid.json'), 'w') as f:
        json.dump(grid_data, f, indent=2)

    # Save syllable stats
    syl_stats_serializable = {k: v for k, v in syl_stats.items()
                              if k != 'top_20_syllables'}
    syl_stats_serializable['top_20_syllables'] = [
        {'syllable': s, 'count': c} for s, c in syl_stats.get('top_20_syllables', [])
    ]
    with open(os.path.join(results_dir, 'syllable_stats.json'), 'w') as f:
        json.dump(syl_stats_serializable, f, indent=2, default=str)

    # Save discriminant results
    disc_serializable = {k: v for k, v in disc.items() if k != 'real'}
    disc_serializable['real_h1'] = disc['real'].get('syllable_h1', 0)
    disc_serializable['real_h2'] = disc['real'].get('syllable_h2', 0)
    with open(os.path.join(results_dir, 'stroke_discriminant.json'), 'w') as f:
        json.dump(disc_serializable, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")
    print("=" * 70)

    return {
        'positional_analysis': pos_analysis,
        'grid': grid,
        'syllable_stats': syl_stats,
        'discriminant': disc,
    }


if __name__ == '__main__':
    run_stroke_analysis()
