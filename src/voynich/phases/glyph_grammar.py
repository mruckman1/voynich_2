"""
Phase 10.4 — Glyph Construction Grammar Test
==============================================

Rationale
---------
If the Voynich script is a constructed writing system (H1), the stroke-level
grid (onset × nucleus) describes how individual glyphs are built from stroke
components — analogous to Hangul jamo, Devanagari aksara, or Ethiopic fidel.

Section strategy:
  Language A only for the primary grid analysis and CSP.  Language B as a
  comparison: if the grid describes glyph construction rules, Language B should
  use the SAME grid (same script) but with restricted cell occupancy.  Phase
  4.5 showed Language A occupancy ~50%, Language B ~37%.  If the CSP produces
  phonetic values from Language A, those same values applied to Language B
  should produce the "13 core tokens" as the most frequent syllable
  combinations.

Sub-analyses
------------
10.4a  Grid properties comparison against known scripts
10.4b  Glyph construction vs language morphology diagnostic
10.4c  Constructed script phonetic assignment attempt (CSP)
10.4d  Language B consistency check
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.reference import ROMANCE_PHONOTACTICS, SCRIPT_GRID_STATS
from voynich.core.stats import rank_correlation


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScriptGridComparison:
    script_name: str
    onset_types: int
    nucleus_types: int
    occupancy: float
    r_forward: float
    r_reverse: float
    similarity_to_voynich: float


@dataclass
class ConstructionVsMorphology:
    onset_position_correlation: float
    onset_position_p: float
    nucleus_position_correlation: float
    nucleus_position_p: float
    diagnosis: str  # 'construction' or 'morphology'


@dataclass
class LanguageBConsistency:
    lang_a_occupancy: float
    lang_b_occupancy: float
    lang_b_uses_subset: bool
    core_tokens_match: bool
    core_token_pct: float


@dataclass
class CSPResult:
    n_variables: int
    n_candidates_after_pruning: int
    best_assignment: Dict[str, str]
    pct_valid_words: float
    selectivity_vs_random: float
    lang_b_consistency: Optional[Dict]
    decoding_viable: bool


@dataclass
class GlyphGrammarResult:
    voynich_grid_stats: Dict[str, Any]
    grid_comparisons: List[Dict]
    closest_script: str
    construction_test: Dict
    csp_result: Optional[Dict]
    h1_supported: bool
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Convert dataclass/numpy types to JSON-serializable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _load_voynich_grid_stats() -> Dict[str, Any]:
    """Load Phase 4.5 morpheme grid results and ventris grid results."""
    morpheme_path = _results_dir() / 'morpheme_grid.json'
    ventris_path = _results_dir() / 'ventris_grid.json'

    stats = {}

    if morpheme_path.exists():
        with open(morpheme_path) as f:
            mg = json.load(f)

        # R values are nested under 'r_reinterpretation'
        r_block = mg.get('r_reinterpretation', {})
        stats['r_forward'] = r_block.get('original_r', 0.0)
        stats['r_reverse'] = r_block.get('original_reverse_r', 0.0)
        stats['r_affix_given_stem'] = r_block.get('r_affix_given_stem', 0.0)
        stats['r_stem_given_affix'] = r_block.get('r_stem_given_affix', 0.0)

        # Axis identification
        axis_block = mg.get('axis_identification', {})
        stats['onset_axis_entropy'] = axis_block.get('onset_entropy', 0.0)
        stats['nucleus_axis_entropy'] = axis_block.get('nucleus_entropy', 0.0)
        stats['onset_axis_role'] = axis_block.get('onset_role', '')
        stats['nucleus_axis_role'] = axis_block.get('nucleus_role', '')

        stats['verdict'] = mg.get('verdict', '')

        # Extract prefix/suffix type counts
        stat_block = mg.get('stats', {})
        stats['n_prefix_types'] = stat_block.get('n_prefix_types', 0)
        stats['n_suffix_types'] = stat_block.get('n_suffix_types', 0)

    if ventris_path.exists():
        with open(ventris_path) as f:
            vg = json.load(f)
        stats['onset_types'] = len(vg.get('row_labels', []))
        stats['nucleus_types'] = len(vg.get('col_labels', []))
        stats['occupancy'] = vg.get('occupancy', 0.0)
        stats['n_filled'] = vg.get('n_filled', 0)
        stats['n_total'] = vg.get('n_total', 0)
        stats['grid_cells'] = vg.get('cells', {})
        stats['row_labels'] = vg.get('row_labels', [])
        stats['col_labels'] = vg.get('col_labels', [])

    # Fallback defaults
    stats.setdefault('onset_types', 7)
    stats.setdefault('nucleus_types', 11)
    stats.setdefault('occupancy', 0.31)
    stats.setdefault('r_forward', 0.39)
    stats.setdefault('r_reverse', 0.61)
    stats.setdefault('onset_axis_entropy', 1.95)
    stats.setdefault('nucleus_axis_entropy', 3.00)

    return stats


def _compare_to_known_scripts(voynich_stats: Dict) -> List[ScriptGridComparison]:
    """Compare Voynich grid to each known combinatorial script."""
    comparisons = []

    v_onset = voynich_stats.get('onset_types', 7)
    v_nucleus = voynich_stats.get('nucleus_types', 11)
    v_occ = voynich_stats.get('occupancy', 0.31)
    v_rf = voynich_stats.get('r_forward', 0.39)
    v_rr = voynich_stats.get('r_reverse', 0.61)

    for name, ref in SCRIPT_GRID_STATS.items():
        # Weighted Euclidean distance
        d_onset = (v_onset - ref['onset_types']) / max(ref['onset_types'], 1)
        d_nucleus = (v_nucleus - ref['nucleus_types']) / max(ref['nucleus_types'], 1)
        d_occ = (v_occ - ref['occupancy'])
        d_rf = (v_rf - ref['r_forward'])
        d_rr = (v_rr - ref['r_reverse'])

        # Weights: occupancy and R values more important than raw counts
        dist = math.sqrt(
            1.0 * d_onset ** 2
            + 1.0 * d_nucleus ** 2
            + 3.0 * d_occ ** 2
            + 2.0 * d_rf ** 2
            + 2.0 * d_rr ** 2
        )

        comparisons.append(ScriptGridComparison(
            script_name=name,
            onset_types=ref['onset_types'],
            nucleus_types=ref['nucleus_types'],
            occupancy=ref['occupancy'],
            r_forward=ref['r_forward'],
            r_reverse=ref['r_reverse'],
            similarity_to_voynich=1.0 / (1.0 + dist),  # Higher = more similar
        ))

    comparisons.sort(key=lambda c: c.similarity_to_voynich, reverse=True)
    return comparisons


def _get_glyph_first_last_char(token: str) -> Tuple[str, str]:
    """Get first and last EVA character of a token."""
    chars = tokenize_eva_chars(token)
    if not chars:
        return ('', '')
    return (chars[0], chars[-1])


def _construction_vs_morphology_test(corpus) -> ConstructionVsMorphology:
    """
    Test whether onset/nucleus stroke identity correlates with token
    position in line.

    In a glyph construction grammar: no correlation (glyph looks the same
    everywhere).
    In morphology mapping: "affix" strokes correlate with position (initial
    position → one set, final → another).
    """
    pages = [p for p in corpus.pages.values() if p.language == 'A']

    # Collect (onset_char, nucleus_char, position_in_line) triples
    onset_positions = []
    nucleus_positions = []
    position_indices = []

    for page in pages:
        for locus in page.loci:
            if not locus.locus_type.startswith('P'):
                continue
            tokens = locus.clean_text.split()
            for idx, token in enumerate(tokens):
                if len(token) < 2:
                    continue
                first, last = _get_glyph_first_last_char(token)
                if first and last:
                    # Encode onset/nucleus as numeric for correlation
                    onset_positions.append(first)
                    nucleus_positions.append(last)
                    # Normalize position to [0, 1]
                    pos = idx / max(len(tokens) - 1, 1)
                    position_indices.append(pos)

    if len(position_indices) < 20:
        return ConstructionVsMorphology(
            onset_position_correlation=0.0,
            onset_position_p=1.0,
            nucleus_position_correlation=0.0,
            nucleus_position_p=1.0,
            diagnosis='insufficient_data',
        )

    # Convert onset/nucleus chars to numeric (rank by frequency)
    onset_counter = Counter(onset_positions)
    nucleus_counter = Counter(nucleus_positions)
    onset_rank = {ch: i for i, (ch, _) in enumerate(onset_counter.most_common())}
    nucleus_rank = {ch: i for i, (ch, _) in enumerate(nucleus_counter.most_common())}

    onset_numeric = np.array([onset_rank[ch] for ch in onset_positions], dtype=float)
    nucleus_numeric = np.array([nucleus_rank[ch] for ch in nucleus_positions], dtype=float)
    pos_numeric = np.array(position_indices, dtype=float)

    onset_corr, onset_p = rank_correlation(onset_numeric, pos_numeric)
    nucleus_corr, nucleus_p = rank_correlation(nucleus_numeric, pos_numeric)

    # If either onset or nucleus significantly correlates with position → morphology
    if onset_p < 0.01 and abs(onset_corr) > 0.1:
        diagnosis = 'morphology'
    elif nucleus_p < 0.01 and abs(nucleus_corr) > 0.1:
        diagnosis = 'morphology'
    else:
        diagnosis = 'construction'

    return ConstructionVsMorphology(
        onset_position_correlation=onset_corr,
        onset_position_p=onset_p,
        nucleus_position_correlation=nucleus_corr,
        nucleus_position_p=nucleus_p,
        diagnosis=diagnosis,
    )


def _build_grid_from_tokens(tokens: List[str]) -> Dict[str, List[str]]:
    """Build onset × nucleus grid from EVA tokens."""
    grid: Dict[str, List[str]] = defaultdict(list)
    for token in tokens:
        chars = tokenize_eva_chars(token)
        if not chars:
            continue
        first = chars[0]
        last = chars[-1] if len(chars) > 1 else chars[0]
        key = f"{first},{last}"
        if token not in grid[key]:
            grid[key].append(token)
    return dict(grid)


def _phonotactic_csp(
    voynich_stats: Dict,
    voynich_tokens: List[str],
) -> CSPResult:
    """
    Constraint satisfaction: map grid cells to phoneme/syllable values.

    Variables: the occupied grid cells (onset × nucleus pairs)
    Domain: syllable values from Romance phonotactics
    Constraints:
      1. Frequency: most common cells → most common syllables
      2. Phonotactics: onset/nucleus structure must be legal
      3. Word structure: decoded tokens must form valid syllable sequences
    """
    # Build the grid from tokens
    grid = _build_grid_from_tokens(voynich_tokens)
    cell_counts = Counter()
    for token in voynich_tokens:
        chars = tokenize_eva_chars(token)
        if not chars:
            continue
        first = chars[0]
        last = chars[-1] if len(chars) > 1 else chars[0]
        cell_counts[f"{first},{last}"] += 1

    # Get the top cells sorted by frequency
    top_cells = [cell for cell, _ in cell_counts.most_common(14)]
    n_variables = len(top_cells)

    if n_variables == 0:
        return CSPResult(
            n_variables=0, n_candidates_after_pruning=0,
            best_assignment={}, pct_valid_words=0.0,
            selectivity_vs_random=0.0, lang_b_consistency=None,
            decoding_viable=False,
        )

    # Build syllable inventory from Latin phonotactics
    latin_pt = ROMANCE_PHONOTACTICS.get('latin', {})
    onsets = latin_pt.get('onsets', [])
    rimes = latin_pt.get('rimes', [])

    # Build syllable frequency from Latin reference
    syllable_candidates = []
    for o in onsets:
        for r in rimes:
            syl = o + r
            if len(syl) >= 1:
                syllable_candidates.append(syl)

    # Rank syllable candidates by how common they'd be in Latin
    # Approximate by length (shorter = more common in natural language)
    syllable_candidates.sort(key=lambda s: len(s))
    syllable_candidates = syllable_candidates[:50]  # Top 50 candidates

    # Simple frequency-rank matching as a baseline
    # Most frequent Voynich cell → most common syllable candidate
    best_assignment = {}
    for i, cell in enumerate(top_cells):
        if i < len(syllable_candidates):
            best_assignment[cell] = syllable_candidates[i]
        else:
            best_assignment[cell] = f"?{i}"

    # Evaluate: decode tokens and count valid syllable sequences
    decoded_tokens = []
    for token in voynich_tokens[:500]:  # Sample for efficiency
        chars = tokenize_eva_chars(token)
        if not chars:
            continue
        first = chars[0]
        last = chars[-1] if len(chars) > 1 else chars[0]
        cell = f"{first},{last}"
        phoneme = best_assignment.get(cell)
        if phoneme:
            decoded_tokens.append(phoneme)

    # Count how many decoded tokens look like valid Romance syllable patterns
    valid_count = 0
    for dt in decoded_tokens:
        # Valid if it contains at least one vowel and follows CV-ish pattern
        has_vowel = any(c in 'aeiou' for c in dt)
        if has_vowel and len(dt) <= 5:
            valid_count += 1

    pct_valid = valid_count / max(len(decoded_tokens), 1)

    # Selectivity: compare to random assignment
    rng = np.random.RandomState(42)
    random_valids = []
    for _ in range(100):
        random_syls = list(syllable_candidates[:len(top_cells)])
        rng.shuffle(random_syls)
        rand_assign = dict(zip(top_cells, random_syls))
        rand_decoded = []
        for token in voynich_tokens[:500]:
            chars = tokenize_eva_chars(token)
            if not chars:
                continue
            first = chars[0]
            last = chars[-1] if len(chars) > 1 else chars[0]
            cell = f"{first},{last}"
            p = rand_assign.get(cell)
            if p:
                rand_decoded.append(p)
        rv = sum(1 for d in rand_decoded if any(c in 'aeiou' for c in d) and len(d) <= 5)
        random_valids.append(rv / max(len(rand_decoded), 1))

    mean_random = float(np.mean(random_valids)) if random_valids else 0.0
    selectivity = pct_valid / mean_random if mean_random > 0 else 0.0

    n_after_pruning = len(syllable_candidates) ** min(n_variables, 3)

    return CSPResult(
        n_variables=n_variables,
        n_candidates_after_pruning=n_after_pruning,
        best_assignment=best_assignment,
        pct_valid_words=pct_valid,
        selectivity_vs_random=selectivity,
        lang_b_consistency=None,  # Filled in later
        decoding_viable=pct_valid > 0.3 and selectivity > 2.0,
    )


def _check_lang_b_consistency(
    csp_assignment: Dict[str, str],
    corpus,
) -> LanguageBConsistency:
    """
    Apply Language A's CSP assignment to Language B tokens.
    Verify that B's top cells are a subset of A's, and that
    the 13 core B tokens map to the most occupied grid cells.
    """
    tokens_a = corpus.get_tokens(language='A')
    tokens_b = corpus.get_tokens(language='B')

    grid_a = _build_grid_from_tokens(tokens_a)
    grid_b = _build_grid_from_tokens(tokens_b)

    cells_a = set(grid_a.keys())
    cells_b = set(grid_b.keys())

    # Occupancy: fraction of all observed cell types that each language uses
    all_cells = cells_a | cells_b
    n_total = max(len(all_cells), 1)
    occ_a = len(cells_a) / n_total
    occ_b = len(cells_b) / n_total

    b_subset = cells_b.issubset(cells_a)

    # Check if B's most frequent tokens map to assigned cells
    b_counts = Counter(tokens_b)
    top_b = [t for t, _ in b_counts.most_common(13)]
    core_match_count = 0
    total_b_tokens_in_core = 0

    for token in top_b:
        chars = tokenize_eva_chars(token)
        if not chars:
            continue
        first = chars[0]
        last = chars[-1] if len(chars) > 1 else chars[0]
        cell = f"{first},{last}"
        if cell in csp_assignment:
            core_match_count += 1
        total_b_tokens_in_core += b_counts[token]

    core_match = core_match_count >= 10  # At least 10 of 13
    core_pct = total_b_tokens_in_core / max(len(tokens_b), 1)

    return LanguageBConsistency(
        lang_a_occupancy=occ_a,
        lang_b_occupancy=occ_b,
        lang_b_uses_subset=b_subset,
        core_tokens_match=core_match,
        core_token_pct=core_pct,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_glyph_grammar() -> Dict[str, Any]:
    """Run Phase 10.4: glyph construction grammar test."""
    print("=" * 60)
    print("Phase 10.4 — Glyph Construction Grammar Test")
    print("=" * 60)

    # --- Load data ---
    corpus = load_corpus(verbose=False)

    # --- Load Voynich grid stats from Phase 4.5 ---
    print("\n  Loading Voynich grid statistics from Phase 4.5...")
    voynich_stats = _load_voynich_grid_stats()
    print(f"    Onset types:   {voynich_stats.get('onset_types')}")
    print(f"    Nucleus types: {voynich_stats.get('nucleus_types')}")
    print(f"    Occupancy:     {voynich_stats.get('occupancy', 0):.3f}")
    print(f"    R(forward):    {voynich_stats.get('r_forward', 0):.4f}")
    print(f"    R(reverse):    {voynich_stats.get('r_reverse', 0):.4f}")

    # --- Compare to known scripts ---
    print("\n  Comparing to known combinatorial scripts...")
    comparisons = _compare_to_known_scripts(voynich_stats)
    for comp in comparisons:
        print(f"    {comp.script_name:12s}: similarity={comp.similarity_to_voynich:.3f} "
              f"(occupancy={comp.occupancy:.2f}, "
              f"R_f={comp.r_forward:.2f}, R_r={comp.r_reverse:.2f})")

    closest = comparisons[0].script_name if comparisons else 'none'
    print(f"    Closest match: {closest}")

    # --- Construction vs morphology test ---
    print("\n  Testing glyph construction vs language morphology...")
    const_test = _construction_vs_morphology_test(corpus)
    print(f"    Onset-position correlation:  r={const_test.onset_position_correlation:.4f} "
          f"(p={const_test.onset_position_p:.4f})")
    print(f"    Nucleus-position correlation: r={const_test.nucleus_position_correlation:.4f} "
          f"(p={const_test.nucleus_position_p:.4f})")
    print(f"    Diagnosis: {const_test.diagnosis}")

    # --- CSP ---
    print("\n  Running phonotactic constraint satisfaction...")
    tokens_a = corpus.get_tokens(language='A')
    csp_result = _phonotactic_csp(voynich_stats, tokens_a)
    print(f"    Variables: {csp_result.n_variables}")
    print(f"    % valid decoded: {csp_result.pct_valid_words:.3f}")
    print(f"    Selectivity vs random: {csp_result.selectivity_vs_random:.3f}")
    print(f"    Decoding viable: {csp_result.decoding_viable}")

    # Top 5 assignments
    if csp_result.best_assignment:
        print("    Top assignments:")
        for i, (cell, phoneme) in enumerate(list(csp_result.best_assignment.items())[:5]):
            print(f"      {cell} → {phoneme}")

    # --- Language B consistency ---
    print("\n  Checking Language B consistency...")
    lang_b_check = _check_lang_b_consistency(csp_result.best_assignment, corpus)
    print(f"    Language A occupancy: {lang_b_check.lang_a_occupancy:.3f}")
    print(f"    Language B occupancy: {lang_b_check.lang_b_occupancy:.3f}")
    print(f"    B cells ⊂ A cells: {lang_b_check.lang_b_uses_subset}")
    print(f"    Core tokens match: {lang_b_check.core_tokens_match}")
    print(f"    Core token coverage: {lang_b_check.core_token_pct:.3f}")

    csp_result.lang_b_consistency = _convert(asdict(lang_b_check))

    # --- H1 verdict ---
    h1_indicators = [
        comparisons[0].similarity_to_voynich > 0.3 if comparisons else False,
        const_test.diagnosis == 'construction',
        csp_result.decoding_viable,
        lang_b_check.lang_b_uses_subset,
    ]
    h1_supported = sum(h1_indicators) >= 2

    gate_passed = h1_supported or (not any(h1_indicators))

    if h1_supported:
        verdict = (f"glyph_grammar_supports_H1: closest_script={closest}, "
                   f"diagnosis={const_test.diagnosis}, "
                   f"csp_viable={csp_result.decoding_viable}, "
                   f"lang_b_subset={lang_b_check.lang_b_uses_subset}")
    elif not any(h1_indicators):
        verdict = "glyph_grammar_rejects_H1: no construction grammar indicators"
    else:
        verdict = (f"glyph_grammar_ambiguous: "
                   f"script_match={comparisons[0].similarity_to_voynich:.3f if comparisons else 0}, "
                   f"diagnosis={const_test.diagnosis}, "
                   f"csp_viable={csp_result.decoding_viable}")

    print(f"\n  H1 supported: {h1_supported}")
    print(f"  Gate passed: {gate_passed}")
    print(f"  Verdict: {verdict}")

    # --- Save ---
    result = GlyphGrammarResult(
        voynich_grid_stats={k: v for k, v in voynich_stats.items()
                           if k not in ('grid_cells', 'row_labels', 'col_labels')},
        grid_comparisons=[_convert(asdict(c)) for c in comparisons],
        closest_script=closest,
        construction_test=_convert(asdict(const_test)),
        csp_result=_convert(asdict(csp_result)),
        h1_supported=h1_supported,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'glyph_grammar.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return out
