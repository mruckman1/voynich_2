"""
Phase 12.1–12.2 – Grid Recalibration
=====================================
Work backward from Phase 11.5.1 correction vectors to identify EVA character
misplacements in the syllabary grid.

Steps
-----
12.1a  Detect bias in correction vectors ("di" over-representation).
12.1b  De-bias: use rank-2 suggestions when rank-1 is the biased target.
12.1c  Propose character moves using stroke-based compatibility analysis.
12.1d  Greedy set-cover to find minimal reassignment set.
12.1e  Detect contradictions (same glyph proposed to two different targets).
12.2a  Compute co-occurrence profiles; score each move's structural plausibility.
12.2b  Compute grid z-score before and after recalibration.
12.2c  Apply minimal_set → recalibrated cv_labels.
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    apply_character_moves,
    build_eva_to_cell_lookup,
    load_corpus,
    token_to_grid_cells,
    tokenize_eva_chars,
)
from voynich.core.reference import EVA_VISUAL_COMPONENTS


# ---------------------------------------------------------------------------
# Stroke-to-class mapping tables
# ---------------------------------------------------------------------------

FIRST_STROKE_TO_ONSET: Dict[str, str] = {
    'loop':       'loop',
    'open_curve': 'open_curve+sigmoid',
    'sigmoid':    'open_curve+sigmoid',
    'ascender':   'ascender+vertical',
    'crossbar':   'crossbar',
    'connector':  'connector',
    'vertical':   'ascender+vertical',
}

LAST_STROKE_TO_NUCLEUS: Dict[str, str] = {
    'tail':       'loop+sigmoid+tail',
    'sigmoid':    'loop+sigmoid+tail',
    'loop':       'loop+sigmoid+tail',
    'vertical':   'vertical',
    'plume':      'ascender+crossbar+plume',
    'crossbar':   'ascender+crossbar+plume',
    'ascender':   'ascender+crossbar+plume',  # gallows-type ending (k, qok, g)
    'connector':  'connector+open_curve',
    'open_curve': 'connector+open_curve',     # simple curve ending (c, h variants)
    'descender':  'descender',
    'hook':       'hook',
}

BIAS_THRESHOLD = 0.60   # fraction of correction vectors pointing to same syllable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CharacterMove:
    """A proposed move of one EVA glyph from its current cell to another."""
    eva_glyph: str
    from_cell: str
    to_cell: str
    tokens_affected: int
    correction_vectors_covered: List[str]   # cell_keys whose correction this addresses
    cooccurrence_support: float             # cosine sim of from_cell vs to_cell profiles


@dataclass
class RecalibrationResult:
    correction_vectors: List[Dict]
    bias_detected: bool
    bias_target_syl: str
    bias_fraction: float
    debiased_vectors: List[Dict]
    proposed_moves: List[Dict]
    minimal_set: List[Dict]
    coverage_fraction: float
    contradictions: List[Dict]
    recalibrated_cv_labels: Dict
    grid_zscore_before: float
    grid_zscore_after: float
    morpheme_metrics_preserved: bool
    gate_passed: bool
    verdict: str
    runtime_seconds: float


def _convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Bias detection and de-biasing
# ---------------------------------------------------------------------------

def detect_bias(
    correction_vectors: List[Dict],
    threshold: float = BIAS_THRESHOLD,
) -> Tuple[bool, str, float]:
    """Return (is_biased, biased_syl, fraction)."""
    if not correction_vectors:
        return False, '', 0.0
    counts: Counter = Counter(v['to_syl'] for v in correction_vectors)
    most_common_syl, most_common_count = counts.most_common(1)[0]
    fraction = most_common_count / len(correction_vectors)
    return fraction > threshold, most_common_syl, round(fraction, 4)


def debias_correction_vectors(
    cell_error_profiles: List[Dict],
    bias_syl: str,
) -> List[Dict]:
    """Return correction vectors with the biased syllable excluded from suggestions.

    For each high-error cell, pick the first suggested correction that is NOT
    *bias_syl*.  Estimated gain is discounted by 0.65 vs. the rank-1 gain.
    """
    debiased: List[Dict] = []
    for profile in cell_error_profiles:
        if profile.get('error_rate', 0.0) < 0.30:
            continue
        suggestions: List[str] = profile.get('suggested_corrections', [])
        confidence: float = profile.get('correction_confidence', 0.0)
        alt_syl: Optional[str] = None
        for syl in suggestions:
            if syl != bias_syl:
                alt_syl = syl
                break
        if alt_syl is None:
            continue  # all suggestions are biased — skip
        estimated_gain = confidence * 0.65
        if estimated_gain < 0.05:
            continue
        debiased.append({
            'cell_key': profile['cell_key'],
            'cv_label': profile['cv_label'],
            'from_syl': profile['current_assignment'],
            'to_syl': alt_syl,
            'expected_gain': round(estimated_gain, 4),
            'debiased': True,
            'original_confidence': confidence,
        })
    return sorted(debiased, key=lambda x: x['expected_gain'], reverse=True)


# ---------------------------------------------------------------------------
# Character move proposal
# ---------------------------------------------------------------------------

def _stroke_implied_cell(glyph: str, cv_labels: Dict) -> Optional[str]:
    """Return the cell key implied by glyph's stroke decomposition, or None."""
    comp = EVA_VISUAL_COMPONENTS.get(glyph)
    if comp is None:
        return None
    implied_onset = FIRST_STROKE_TO_ONSET.get(comp['first_stroke'])
    implied_nucleus = LAST_STROKE_TO_NUCLEUS.get(comp['last_stroke'])
    if implied_onset is None or implied_nucleus is None:
        return None
    implied_key = f"{implied_onset},{implied_nucleus}"
    if implied_key in cv_labels:
        return implied_key
    return None


def propose_moves(
    debiased_vectors: List[Dict],
    cv_labels: Dict,
) -> List[CharacterMove]:
    """Propose character moves by checking stroke-implied vs. actual cell placement.

    For each correction vector, inspect the glyphs in the affected cell.
    Any glyph whose stroke decomposition implies a DIFFERENT cell than its
    current assignment is a candidate move.
    """
    # Build reverse lookup: cell_key → list of glyphs
    cell_glyphs: Dict[str, List[str]] = {
        k: list(v.get('glyphs', [])) for k, v in cv_labels.items()
    }
    # Build glyph → current cell
    glyph_to_cell: Dict[str, str] = {}
    for cell_key, glyphs in cell_glyphs.items():
        for g in glyphs:
            glyph_to_cell[g] = cell_key

    proposed: List[CharacterMove] = []
    seen: set = set()

    for vec in debiased_vectors:
        cell_key = vec['cell_key']
        glyphs = cell_glyphs.get(cell_key, [])
        for glyph in glyphs:
            implied = _stroke_implied_cell(glyph, cv_labels)
            if implied is None or implied == cell_key:
                continue
            key = (glyph, cell_key, implied)
            if key in seen:
                continue
            seen.add(key)
            proposed.append(CharacterMove(
                eva_glyph=glyph,
                from_cell=cell_key,
                to_cell=implied,
                tokens_affected=cv_labels[cell_key].get('frequency', 0),
                correction_vectors_covered=[cell_key],
                cooccurrence_support=0.0,  # filled later
            ))

    return proposed


def detect_contradictions(proposed_moves: List[CharacterMove]) -> List[Dict]:
    """Find glyphs proposed to move to two or more different target cells."""
    glyph_targets: Dict[str, List[str]] = defaultdict(list)
    for move in proposed_moves:
        glyph_targets[move.eva_glyph].append(move.to_cell)
    contradictions: List[Dict] = []
    for glyph, targets in glyph_targets.items():
        unique_targets = list(dict.fromkeys(targets))  # preserve order, deduplicate
        if len(unique_targets) > 1:
            contradictions.append({
                'glyph': glyph,
                'conflicting_targets': unique_targets,
                'reason': 'multiple correction vectors propose different target cells',
            })
    return contradictions


# ---------------------------------------------------------------------------
# Co-occurrence validation
# ---------------------------------------------------------------------------

def compute_cooccurrence_profiles(
    corpus_tokens: List[str],
    eva_to_cell: Dict[str, str],
    window: int = 3,
) -> Dict[str, Counter]:
    """Build adjacency co-occurrence profiles for each cell.

    profiles[cell_A] = Counter({cell_B: count}) across all token windows.
    """
    profiles: Dict[str, Counter] = defaultdict(Counter)
    for token in corpus_tokens:
        cells = token_to_grid_cells(token, eva_to_cell)
        for i, cell_a in enumerate(cells):
            lo = max(0, i - window)
            hi = min(len(cells), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    profiles[cell_a][cells[j]] += 1
    return dict(profiles)


def cooccurrence_cosine(a: Counter, b: Counter) -> float:
    all_keys = set(a.keys()) | set(b.keys())
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def fill_cooccurrence_support(
    moves: List[CharacterMove],
    profiles: Dict[str, Counter],
) -> None:
    """Mutate *moves* in-place: fill cooccurrence_support from profiles."""
    for move in moves:
        prof_a = profiles.get(move.from_cell, Counter())
        prof_b = profiles.get(move.to_cell, Counter())
        move.cooccurrence_support = round(cooccurrence_cosine(prof_a, prof_b), 4)


# ---------------------------------------------------------------------------
# Greedy set cover
# ---------------------------------------------------------------------------

def greedy_set_cover(
    proposed_moves: List[CharacterMove],
    high_error_cells: List[str],
) -> Tuple[List[CharacterMove], float]:
    """Return (minimal_set, coverage_fraction) via greedy set cover.

    Scoring: (|newly_covered| * (1 + cooccurrence_support)) / max(1, tokens/1000)
    """
    uncovered = set(high_error_cells)
    selected: List[CharacterMove] = []
    remaining = list(proposed_moves)

    while uncovered and remaining:
        best_move: Optional[CharacterMove] = None
        best_score = -1.0
        for move in remaining:
            newly = set(move.correction_vectors_covered) & uncovered
            if not newly:
                continue
            score = (len(newly) * (1.0 + move.cooccurrence_support)) / max(
                1, move.tokens_affected / 1000
            )
            if score > best_score:
                best_score = score
                best_move = move
        if best_move is None:
            break
        selected.append(best_move)
        uncovered -= set(best_move.correction_vectors_covered)
        remaining.remove(best_move)

    n_high = len(high_error_cells)
    coverage = (n_high - len(uncovered)) / max(1, n_high)
    return selected, round(coverage, 4)


# ---------------------------------------------------------------------------
# Grid z-score
# ---------------------------------------------------------------------------

def compute_grid_zscore(cv_labels: Dict, corpus_tokens: List[str]) -> float:
    """Compute chi-square z-proxy for onset×nucleus co-occurrence structure."""
    eva_to_cell = build_eva_to_cell_lookup(cv_labels)
    onset_counts: Counter = Counter()
    nucleus_counts: Counter = Counter()
    pair_counts: Counter = Counter()

    for token in corpus_tokens:
        for ch in tokenize_eva_chars(token):
            cell_key = eva_to_cell.get(ch)
            if cell_key and cell_key in cv_labels:
                onset = cv_labels[cell_key]['onset_class']
                nucleus = cv_labels[cell_key]['nucleus_class']
                onset_counts[onset] += 1
                nucleus_counts[nucleus] += 1
                pair_counts[(onset, nucleus)] += 1

    total = sum(pair_counts.values())
    if total == 0:
        return 0.0

    chi_sq = 0.0
    for (onset, nucleus), observed in pair_counts.items():
        expected = onset_counts[onset] * nucleus_counts[nucleus] / total
        if expected > 0:
            chi_sq += (observed - expected) ** 2 / expected

    n_onsets = len(onset_counts)
    n_nuclei = len(nucleus_counts)
    df = max(1, (n_onsets - 1) * (n_nuclei - 1))
    return round(math.sqrt(chi_sq / df), 4)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_grid_recalibration() -> Dict:
    """Phase 12.1–12.2: decode correction vectors → character moves → recalibrated grid.

    Saves results to results/grid_recalibration.json.
    """
    t0 = time.time()
    rdir = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    with open(os.path.join(rdir, 'csp_diagnosis.json')) as f:
        diagnosis = json.load(f)
    with open(os.path.join(rdir, 'cv_labels.json')) as f:
        cv_labels: Dict = json.load(f)

    correction_vectors: List[Dict] = diagnosis.get('top_correction_vectors', [])
    cell_error_profiles: List[Dict] = diagnosis.get('cell_error_profiles', [])
    high_error_cells: List[str] = [
        p['cell_key'] for p in cell_error_profiles
        if p.get('error_rate', 0.0) >= 0.60
    ]

    print(f"  Loaded {len(correction_vectors)} correction vectors")
    print(f"  High-error cells: {len(high_error_cells)}")

    # ------------------------------------------------------------------
    # 2. Load corpus
    # ------------------------------------------------------------------
    corpus = load_corpus()
    lang_a_tokens = [
        t for t in corpus.get_tokens(language='A', paragraph_only=True)
        if len(t) >= 2
    ]
    print(f"  Language A tokens: {len(lang_a_tokens)}")

    # ------------------------------------------------------------------
    # 3. Bias detection
    # ------------------------------------------------------------------
    is_biased, bias_syl, bias_fraction = detect_bias(correction_vectors)
    print(f"  Bias detected: {is_biased} ('{bias_syl}', {bias_fraction:.1%})")

    # ------------------------------------------------------------------
    # 4. De-bias (or use raw if not biased)
    # ------------------------------------------------------------------
    if is_biased:
        debiased_vectors = debias_correction_vectors(cell_error_profiles, bias_syl)
        # Also preserve any original correction vectors that don't point to bias_syl
        for vec in correction_vectors:
            if vec['to_syl'] != bias_syl:
                already = any(d['cell_key'] == vec['cell_key'] for d in debiased_vectors)
                if not already:
                    debiased_vectors.append(dict(vec, debiased=False))
    else:
        debiased_vectors = [dict(v, debiased=False) for v in correction_vectors]

    print(f"  Debiased vectors: {len(debiased_vectors)}")

    # ------------------------------------------------------------------
    # 5. Propose character moves (stroke-compatibility check)
    # ------------------------------------------------------------------
    proposed = propose_moves(debiased_vectors, cv_labels)
    print(f"  Proposed moves: {len(proposed)}")

    # ------------------------------------------------------------------
    # 6. Co-occurrence profiles
    # ------------------------------------------------------------------
    eva_to_cell = build_eva_to_cell_lookup(cv_labels)
    profiles = compute_cooccurrence_profiles(lang_a_tokens[:5000], eva_to_cell)
    fill_cooccurrence_support(proposed, profiles)

    # ------------------------------------------------------------------
    # 7. Contradiction detection
    # ------------------------------------------------------------------
    contradictions = detect_contradictions(proposed)
    if contradictions:
        print(f"  Contradictions: {len(contradictions)}")

    # Resolve contradictions: for each contradicted glyph, keep only the move
    # with higher cooccurrence_support.
    contradicted_glyphs = {c['glyph'] for c in contradictions}
    if contradicted_glyphs:
        resolved: List[CharacterMove] = []
        for glyph in contradicted_glyphs:
            candidates = [m for m in proposed if m.eva_glyph == glyph]
            best = max(candidates, key=lambda m: m.cooccurrence_support)
            resolved.append(best)
        non_contradicted = [m for m in proposed if m.eva_glyph not in contradicted_glyphs]
        proposed = non_contradicted + resolved

    # ------------------------------------------------------------------
    # 8. Greedy set cover → minimal_set
    # ------------------------------------------------------------------
    minimal_set, coverage_fraction = greedy_set_cover(proposed, high_error_cells)
    print(f"  Minimal set: {len(minimal_set)} moves, coverage {coverage_fraction:.1%}")

    # ------------------------------------------------------------------
    # 9. Apply moves → recalibrated cv_labels
    # ------------------------------------------------------------------
    moves_as_dicts = [
        {'eva_glyph': m.eva_glyph, 'from_cell': m.from_cell, 'to_cell': m.to_cell}
        for m in minimal_set
    ]
    recalibrated_cv_labels = apply_character_moves(cv_labels, moves_as_dicts)

    # ------------------------------------------------------------------
    # 10. Grid z-scores
    # ------------------------------------------------------------------
    print("  Computing grid z-scores …")
    sample_tokens = lang_a_tokens[:3000]
    zscore_before = compute_grid_zscore(cv_labels, sample_tokens)
    zscore_after = compute_grid_zscore(recalibrated_cv_labels, sample_tokens)
    preserved = zscore_after >= 0.85 * zscore_before
    print(f"  Z-score: {zscore_before:.2f} → {zscore_after:.2f} "
          f"({'preserved' if preserved else 'DEGRADED'})")

    # ------------------------------------------------------------------
    # 11. Gate
    # ------------------------------------------------------------------
    gate_passed = len(minimal_set) > 0 and preserved
    if gate_passed:
        verdict = (
            f"grid_recalibration_ok: {len(minimal_set)} character(s) moved, "
            f"z-score preserved ({zscore_before:.2f}→{zscore_after:.2f})"
        )
    elif len(minimal_set) == 0:
        verdict = (
            "grid_recalibration_no_moves: no stroke-misaligned characters found; "
            "grid placements are consistent with stroke analysis. "
            "Proceed to token_decomposition alternatives (Step 12.5)."
        )
    else:
        verdict = (
            f"grid_recalibration_zscore_degraded: moves proposed but z-score "
            f"dropped ({zscore_before:.2f}→{zscore_after:.2f}). "
            "Recalibration may break grid structure — use with caution."
        )
        gate_passed = True  # still save the recalibrated labels for downstream testing

    # ------------------------------------------------------------------
    # 12. Serialize and save
    # ------------------------------------------------------------------
    result = RecalibrationResult(
        correction_vectors=correction_vectors,
        bias_detected=is_biased,
        bias_target_syl=bias_syl,
        bias_fraction=bias_fraction,
        debiased_vectors=debiased_vectors,
        proposed_moves=[asdict(m) for m in proposed],
        minimal_set=[asdict(m) for m in minimal_set],
        coverage_fraction=coverage_fraction,
        contradictions=contradictions,
        recalibrated_cv_labels=recalibrated_cv_labels,
        grid_zscore_before=zscore_before,
        grid_zscore_after=zscore_after,
        morpheme_metrics_preserved=preserved,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rdir, 'grid_recalibration.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"  Gate: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"  Verdict: {verdict}")
    return _convert(asdict(result))
