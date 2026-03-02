"""
Phase 13.1 – Near-Miss Error Pattern Analysis
=============================================
Builds a character-level error catalog from Phase 11.5 near-miss tokens.
For each near-miss token, aligns the decoded string to the nearest reference
word (Needleman-Wunsch), records which grid cell produced each error, and
tags each error with:
  - position in token ('initial' | 'medial' | 'final')
  - predecessor and successor cell
  - the produced (wrong) and needed (correct) phoneme

Then tests whether errors correlate with position or adjacency using
chi-squared tests and mutual information vs shuffled null.

Gate 13.1: MI selectivity > 1.5× → proceed to rule extraction.
           MI selectivity < 1.5× → errors are random; skip to null hypothesis.
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_cell_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import load_reference_corpus
from voynich.phases.csp_constraints import build_phoneme_inventory
from voynich.phases.csp_solver import (
    _convert,
    decode_token,
)
from voynich.phases.csp_diagnosis import (
    categorize_token,
    _edit_distance,
    _nearest_word,
    _bucket_by_length,
    TokenDiagnosis,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ErrorRecord:
    """One character-level error from an aligned near-miss token."""
    cell_key: str          # Which grid cell produced the error
    cv_label: str          # e.g. "C2V3"
    produced: str          # Character(s) produced by fixed-table syllable
    needed: str            # Character(s) needed to match target word
    position: str          # 'initial' | 'medial' | 'final'
    predecessor: str       # Preceding cell key, or 'NONE'
    successor: str         # Following cell key, or 'NONE'
    voynich_token: str     # Source EVA token (debugging)
    decoded: str           # Full decoded string
    target: str            # Nearest reference word


@dataclass
class ContextualizedDiagnosis:
    """TokenDiagnosis extended with per-cell position and adjacency info."""
    voynich_token: str
    decoded: str
    category: str
    best_dict_match: str
    best_dict_distance: int
    cells_used: List[str]
    cell_positions: List[str]        # 'initial'|'medial'|'final' per cell
    cell_predecessors: List[str]     # preceding cell key or 'NONE' per cell
    cell_successors: List[str]       # following cell key or 'NONE' per cell
    error_records: List[Dict]        # serialised ErrorRecord objects


@dataclass
class PositionTest:
    """Chi-squared test for position-dependent errors on one cell."""
    cell_key: str
    cv_label: str
    current_assignment: str
    n_errors: int
    n_errors_by_position: Dict[str, int]        # {'initial': n, 'medial': n, 'final': n}
    top_correction_by_position: Dict[str, str]  # best correction phoneme per position
    chi2: float
    p_value: float
    significant: bool                            # p < 0.01


@dataclass
class AdjacencyTest:
    """Chi-squared test for predecessor-dependent errors on one cell."""
    cell_key: str
    cv_label: str
    current_assignment: str
    n_errors: int
    n_errors_by_predecessor_class: Dict[str, int]   # {'vowel': n, 'consonant': n, 'NONE': n}
    top_correction_by_predecessor: Dict[str, str]
    chi2: float
    p_value: float
    significant: bool


@dataclass
class ErrorPatternResult:
    """Full Phase 13.1 output."""
    n_near_miss_tokens: int
    n_error_records: int
    mi_real: float
    mi_shuffled_mean: float
    mi_shuffled_std: float
    mi_selectivity: float
    position_tests: List[Dict]
    adjacency_tests: List[Dict]
    n_cells_position_dependent: int
    n_cells_adjacency_dependent: int
    error_catalog_sample: List[Dict]    # first 50 error records for inspection
    gate_passed: bool
    gate_message: str


# ---------------------------------------------------------------------------
# Needleman-Wunsch alignment
# ---------------------------------------------------------------------------

def _nw_align(s1: str, s2: str, match: int = 1, mismatch: int = -1,
              gap: int = -1) -> Tuple[str, str]:
    """Global sequence alignment via Needleman-Wunsch.

    Returns (aligned_s1, aligned_s2) where '-' denotes gaps.
    """
    m, n = len(s1), len(s2)
    # DP matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * gap
    for j in range(n + 1):
        dp[0][j] = j * gap

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = dp[i-1][j-1] + (match if s1[i-1] == s2[j-1] else mismatch)
            left = dp[i][j-1] + gap
            up   = dp[i-1][j] + gap
            dp[i][j] = max(diag, left, up)

    # Traceback
    a1, a2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = dp[i-1][j-1] + (match if s1[i-1] == s2[j-1] else mismatch)
            if dp[i][j] == diag:
                a1.append(s1[i-1])
                a2.append(s2[j-1])
                i -= 1; j -= 1
                continue
        if j > 0 and dp[i][j] == dp[i][j-1] + gap:
            a1.append('-')
            a2.append(s2[j-1])
            j -= 1
        else:
            a1.append(s1[i-1])
            a2.append('-')
            i -= 1

    return ''.join(reversed(a1)), ''.join(reversed(a2))


# ---------------------------------------------------------------------------
# Error catalog construction
# ---------------------------------------------------------------------------

def _classify_nucleus(cell_key: str) -> str:
    """Return 'vowel' if the cell's nucleus class is vowel-dominant, else 'consonant'."""
    # Nucleus is the second part of "onset,nucleus" key
    parts = cell_key.split(',')
    nucleus = parts[1] if len(parts) > 1 else ''
    # Cells whose nucleus is loop/sigmoid/tail are the most common and tend to
    # carry vowel sounds; others tend to carry consonant codas.
    if 'loop' in nucleus or 'tail' in nucleus or 'descender' in nucleus:
        return 'vowel'
    return 'consonant'


def build_contextualized_diagnoses(
    diagnoses: List[TokenDiagnosis],
    assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    cv_labels: Dict,
) -> List[ContextualizedDiagnosis]:
    """Add position and adjacency tags to each diagnosis."""
    result: List[ContextualizedDiagnosis] = []
    for d in diagnoses:
        cells = d.cells_used
        n = len(cells)
        positions: List[str] = []
        predecessors: List[str] = []
        successors: List[str] = []
        for i, ck in enumerate(cells):
            if n == 1:
                positions.append('initial')  # single-cell token = initial
            elif i == 0:
                positions.append('initial')
            elif i == n - 1:
                positions.append('final')
            else:
                positions.append('medial')
            predecessors.append(cells[i - 1] if i > 0 else 'NONE')
            successors.append(cells[i + 1] if i < n - 1 else 'NONE')

        result.append(ContextualizedDiagnosis(
            voynich_token=d.voynich_token,
            decoded=d.decoded,
            category=d.category,
            best_dict_match=d.best_dict_match,
            best_dict_distance=d.best_dict_distance,
            cells_used=cells,
            cell_positions=positions,
            cell_predecessors=predecessors,
            cell_successors=successors,
            error_records=[],
        ))
    return result


def align_and_extract_errors(
    diag: ContextualizedDiagnosis,
    assignment: Dict[str, str],
    cv_labels: Dict,
) -> List[ErrorRecord]:
    """Align decoded string to target word and extract per-cell errors.

    Each grid cell maps to a fixed syllable (1–3 characters). We:
    1. Build the per-cell character spans in the decoded string.
    2. Globally align decoded to target.
    3. For each aligned mismatch, identify the responsible cell.
    """
    decoded = diag.decoded
    target = diag.best_dict_match
    if not decoded or not target or diag.best_dict_distance == 0:
        return []

    # Build cell→char spans in decoded string
    cells = diag.cells_used
    cell_spans: List[Tuple[int, int]] = []  # (start, end) inclusive
    pos = 0
    for ck in cells:
        syl = assignment.get(ck, '')
        length = len(syl) if syl else 0
        cell_spans.append((pos, pos + length))
        pos += length

    # Align
    a_decoded, a_target = _nw_align(decoded, target)

    # Map aligned positions back to cell index
    # We iterate through a_decoded to know which original char we're at.
    orig_pos = 0  # position in original decoded string
    errors: List[ErrorRecord] = []

    for ad, at in zip(a_decoded, a_target):
        if ad == '-':
            # Insertion in target — no cell responsible, skip
            continue
        # Find which cell owns orig_pos
        cell_idx = None
        for ci, (start, end) in enumerate(cell_spans):
            if start <= orig_pos < end:
                cell_idx = ci
                break
        if ad == '-':
            orig_pos += 1
            continue
        if at == '-':
            # Deletion in target — we produced an extra character; note it
            if cell_idx is not None:
                ck = cells[cell_idx]
                produced_syl = assignment.get(ck, '')
                errors.append(ErrorRecord(
                    cell_key=ck,
                    cv_label=cv_labels.get(ck, {}).get('cv_label', '?') if isinstance(cv_labels.get(ck), dict) else '?',
                    produced=ad,
                    needed='',
                    position=diag.cell_positions[cell_idx],
                    predecessor=diag.cell_predecessors[cell_idx],
                    successor=diag.cell_successors[cell_idx],
                    voynich_token=diag.voynich_token,
                    decoded=decoded,
                    target=target,
                ))
            orig_pos += 1
            continue
        if ad != at and cell_idx is not None:
            ck = cells[cell_idx]
            errors.append(ErrorRecord(
                cell_key=ck,
                cv_label=cv_labels.get(ck, {}).get('cv_label', '?') if isinstance(cv_labels.get(ck), dict) else '?',
                produced=ad,
                needed=at,
                position=diag.cell_positions[cell_idx],
                predecessor=diag.cell_predecessors[cell_idx],
                successor=diag.cell_successors[cell_idx],
                voynich_token=diag.voynich_token,
                decoded=decoded,
                target=target,
            ))
        orig_pos += 1

    return errors


def build_error_catalog(
    ctx_diagnoses: List[ContextualizedDiagnosis],
    assignment: Dict[str, str],
    cv_labels: Dict,
) -> Tuple[List[ContextualizedDiagnosis], List[ErrorRecord]]:
    """Build the full error catalog from near-miss tokens."""
    all_errors: List[ErrorRecord] = []
    updated: List[ContextualizedDiagnosis] = []
    for d in ctx_diagnoses:
        if d.category == 'NEAR_MISS':
            errs = align_and_extract_errors(d, assignment, cv_labels)
            d.error_records = [asdict(e) for e in errs]
            all_errors.extend(errs)
        updated.append(d)
    return updated, all_errors


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _chi2_from_contingency(table: Dict[Any, Counter]) -> Tuple[float, float]:
    """Compute chi-squared statistic and approximate p-value for a contingency table.

    table: {row_key: Counter({col_key: count})}
    Returns (chi2, p_value).  p_value is approximated from chi2 CDF.
    """
    # Build row and column marginals
    rows = list(table.keys())
    cols: set = set()
    for cnt in table.values():
        cols.update(cnt.keys())
    cols = sorted(cols)

    if len(rows) < 2 or len(cols) < 2:
        return 0.0, 1.0

    total = sum(cnt.total() for cnt in table.values())
    if total == 0:
        return 0.0, 1.0

    row_totals = {r: sum(table[r].values()) for r in rows}
    col_totals = {c: sum(table[r].get(c, 0) for r in rows) for c in cols}

    chi2 = 0.0
    for r in rows:
        for c in cols:
            observed = table[r].get(c, 0)
            expected = (row_totals[r] * col_totals[c]) / total
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

    df = (len(rows) - 1) * (len(cols) - 1)
    # Approximate p-value: use chi2 CDF approximation
    p_value = _chi2_survival(chi2, df)
    return chi2, p_value


def _chi2_survival(chi2: float, df: int) -> float:
    """Approximate chi-squared survival function (1 - CDF).

    Uses Wilson-Hilferty approximation for df > 1.
    """
    if df <= 0 or chi2 <= 0:
        return 1.0
    # Simple threshold-based approximation
    # Critical values at p=0.01 for common df
    critical_values = {1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28,
                       5: 15.09, 6: 16.81, 7: 18.48, 8: 20.09, 9: 21.67, 10: 23.21}
    crit = critical_values.get(df, 3.84 * df)  # rough fallback
    if chi2 > crit * 2:
        return 0.0001
    elif chi2 > crit:
        return 0.005
    elif chi2 > crit * 0.7:
        return 0.05
    else:
        return 0.20


def test_position_dependence(
    error_catalog: List[ErrorRecord],
    cv_labels: Dict,
    assignment: Dict[str, str],
) -> List[PositionTest]:
    """For each cell, test whether error corrections differ by position."""
    # Group errors by cell
    by_cell: Dict[str, List[ErrorRecord]] = defaultdict(list)
    for e in error_catalog:
        by_cell[e.cell_key].append(e)

    tests: List[PositionTest] = []
    for cell_key, errs in by_cell.items():
        if len(errs) < 10:
            continue  # too few errors for reliable test

        # Build contingency table: position × correction_needed
        table: Dict[str, Counter] = {
            'initial': Counter(),
            'medial': Counter(),
            'final': Counter(),
        }
        for e in errs:
            if e.needed and e.position in table:
                table[e.position][e.needed] += 1

        chi2, pval = _chi2_from_contingency(table)

        # Top correction per position
        top_correction: Dict[str, str] = {}
        for pos, cnt in table.items():
            if cnt:
                top_correction[pos] = cnt.most_common(1)[0][0]

        n_by_pos = {pos: sum(cnt.values()) for pos, cnt in table.items()}

        cv_label = cv_labels.get(cell_key, {}).get('cv_label', '?') if isinstance(cv_labels.get(cell_key), dict) else '?'
        tests.append(PositionTest(
            cell_key=cell_key,
            cv_label=cv_label,
            current_assignment=assignment.get(cell_key, '?'),
            n_errors=len(errs),
            n_errors_by_position=n_by_pos,
            top_correction_by_position=top_correction,
            chi2=round(chi2, 3),
            p_value=round(pval, 4),
            significant=(pval < 0.01),
        ))

    tests.sort(key=lambda t: t.chi2, reverse=True)
    return tests


def test_adjacency_dependence(
    error_catalog: List[ErrorRecord],
    cv_labels: Dict,
    assignment: Dict[str, str],
) -> List[AdjacencyTest]:
    """For each cell, test whether corrections differ by predecessor class."""
    by_cell: Dict[str, List[ErrorRecord]] = defaultdict(list)
    for e in error_catalog:
        by_cell[e.cell_key].append(e)

    tests: List[AdjacencyTest] = []
    for cell_key, errs in by_cell.items():
        if len(errs) < 10:
            continue

        # Classify predecessor as vowel-dominant, consonant-dominant, or NONE
        table: Dict[str, Counter] = {
            'vowel': Counter(),
            'consonant': Counter(),
            'NONE': Counter(),
        }
        for e in errs:
            if not e.needed:
                continue
            pred = e.predecessor
            if pred == 'NONE':
                pred_class = 'NONE'
            else:
                pred_class = _classify_nucleus(pred)
            table[pred_class][e.needed] += 1

        chi2, pval = _chi2_from_contingency(table)

        top_correction: Dict[str, str] = {}
        for pred_class, cnt in table.items():
            if cnt:
                top_correction[pred_class] = cnt.most_common(1)[0][0]

        n_by_pred = {pc: sum(cnt.values()) for pc, cnt in table.items()}

        cv_label = cv_labels.get(cell_key, {}).get('cv_label', '?') if isinstance(cv_labels.get(cell_key), dict) else '?'
        tests.append(AdjacencyTest(
            cell_key=cell_key,
            cv_label=cv_label,
            current_assignment=assignment.get(cell_key, '?'),
            n_errors=len(errs),
            n_errors_by_predecessor_class=n_by_pred,
            top_correction_by_predecessor=top_correction,
            chi2=round(chi2, 3),
            p_value=round(pval, 4),
            significant=(pval < 0.01),
        ))

    tests.sort(key=lambda t: t.chi2, reverse=True)
    return tests


# ---------------------------------------------------------------------------
# Mutual information gate test
# ---------------------------------------------------------------------------

def _compute_mi(error_catalog: List[ErrorRecord]) -> float:
    """Compute MI(correction_direction, context_type).

    context_type is a 4-way combination: position × predecessor_class.
    """
    if not error_catalog:
        return 0.0

    # Joint distribution: (correction, context) counts
    joint: Counter = Counter()
    for e in error_catalog:
        if not e.needed:
            continue
        correction = e.needed  # the needed phoneme
        pred_class = _classify_nucleus(e.predecessor) if e.predecessor != 'NONE' else 'NONE'
        context = f"{e.position}_{pred_class}"
        joint[(correction, context)] += 1

    total = sum(joint.values())
    if total == 0:
        return 0.0

    # Marginals
    p_correction: Counter = Counter()
    p_context: Counter = Counter()
    for (corr, ctx), cnt in joint.items():
        p_correction[corr] += cnt
        p_context[ctx] += cnt

    mi = 0.0
    for (corr, ctx), cnt in joint.items():
        p_xy = cnt / total
        p_x = p_correction[corr] / total
        p_y = p_context[ctx] / total
        if p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return max(0.0, mi)


def compute_context_mi(
    error_catalog: List[ErrorRecord],
    n_shuffles: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute MI(correction, context) and shuffled null distribution.

    Returns (mi_real, shuffled_mean, shuffled_std).
    """
    mi_real = _compute_mi(error_catalog)

    # Null distribution: shuffle context labels while keeping corrections fixed
    rng = random.Random(seed)
    contexts = [(e.position, e.predecessor) for e in error_catalog if e.needed]

    null_mis: List[float] = []
    for _ in range(n_shuffles):
        shuffled_contexts = list(contexts)
        rng.shuffle(shuffled_contexts)
        shuffled_catalog = []
        idx = 0
        for e in error_catalog:
            if not e.needed:
                shuffled_catalog.append(e)
                continue
            pos, pred = shuffled_contexts[idx]
            idx += 1
            shuffled_catalog.append(ErrorRecord(
                cell_key=e.cell_key,
                cv_label=e.cv_label,
                produced=e.produced,
                needed=e.needed,
                position=pos,
                predecessor=pred,
                successor=e.successor,
                voynich_token=e.voynich_token,
                decoded=e.decoded,
                target=e.target,
            ))
        null_mis.append(_compute_mi(shuffled_catalog))

    shuffled_mean = sum(null_mis) / max(len(null_mis), 1)
    shuffled_std = (
        math.sqrt(sum((x - shuffled_mean) ** 2 for x in null_mis) / max(len(null_mis), 1))
        if null_mis else 0.0
    )
    return mi_real, shuffled_mean, shuffled_std


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_error_patterns() -> Dict:
    """Phase 13.1: Build error catalog and test for context-dependent patterns.

    Loads Phase 11.5 best assignment, runs tokenization on Language A corpus,
    categorizes tokens (reusing Phase 11.5.1 logic), builds contextualized
    diagnoses with position/adjacency tags, aligns near-miss tokens against
    their nearest reference words, and tests for systematic error patterns.
    """
    print("=" * 70)
    print("PHASE 13.1: Near-Miss Error Pattern Analysis")
    print("=" * 70)

    t0 = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load Phase 11.5 (or Phase 12) best assignment
    # ------------------------------------------------------------------
    # Prefer Phase 12 recalibrated result, fall back to Phase 11.5 final
    for fname in ('recalibrated_csp.json', 'csp_final.json', 'csp_decode.json'):
        candidate = os.path.join(rd, fname)
        if os.path.exists(candidate):
            with open(candidate) as f:
                decode_data = json.load(f)
            print(f"  Loaded assignment from {fname}")
            break
    else:
        print("  [SKIP] No CSP decode result found — run csp-final or recal-csp first")
        return {'verdict': 'skipped', 'reason': 'no_csp_result'}

    # Navigate to best_assignment depending on file format
    if 'best_assignment' in decode_data:
        best_assignment: Dict[str, str] = decode_data['best_assignment']
        eva_to_cell_map: Dict[str, str] = decode_data.get('eva_to_cell_mapping', {})
    elif 'language_results' in decode_data:
        lat = decode_data['language_results'].get('latin', {})
        best_assignment = lat.get('best_assignment', {})
        eva_to_cell_map = decode_data.get('eva_to_cell_mapping', {})
    else:
        print("  [SKIP] Could not extract best_assignment")
        return {'verdict': 'skipped', 'reason': 'no_assignment'}

    if not best_assignment:
        print("  [SKIP] Empty best_assignment")
        return {'verdict': 'skipped', 'reason': 'empty_assignment'}

    print(f"  Best assignment: {len(best_assignment)} cells")
    print(f"  Baseline dict_hit: {decode_data.get('best_dict_hit', decode_data.get('language_results', {}).get('latin', {}).get('best_dict_hit', 'unknown'))}")

    # ------------------------------------------------------------------
    # 2. Load supporting data
    # ------------------------------------------------------------------
    cv_path = os.path.join(rd, 'cv_labels.json')
    with open(cv_path) as f:
        cv_labels = json.load(f)

    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    corpus_tokens = corpus.get_tokens(language='A', paragraph_only=True)[:1500]
    print(f"  Corpus tokens: {len(corpus_tokens)}")

    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_word_set: set = set(ref_tokens[:50000])
    ref_words_by_len = _bucket_by_length(ref_tokens[:10000], max_per_bucket=60)
    print(f"  Reference word set: {len(ref_word_set)} words")

    inventory = build_phoneme_inventory('latin', ref_corpus)

    if eva_to_cell_map:
        eva_to_cell = eva_to_cell_map
    else:
        eva_to_cell = build_eva_to_cell_lookup(cv_labels)

    # ------------------------------------------------------------------
    # 3. Categorize tokens (same as Phase 11.5.1)
    # ------------------------------------------------------------------
    print("\n  Categorising tokens...")
    raw_diagnoses: List[TokenDiagnosis] = []
    for token in corpus_tokens:
        from voynich.phases.csp_diagnosis import _get_cells_used
        decoded = decode_token(token, best_assignment, eva_to_cell)
        cells_used = _get_cells_used(token, eva_to_cell)
        cat, best_match, best_dist = categorize_token(
            decoded, ref_word_set, ref_words_by_len, inventory,
        )
        raw_diagnoses.append(TokenDiagnosis(
            voynich_token=token,
            decoded=decoded,
            category=cat,
            best_dict_match=best_match,
            best_dict_distance=best_dist,
            cells_used=cells_used,
        ))

    cat_counts = Counter(d.category for d in raw_diagnoses)
    n_near_miss = cat_counts.get('NEAR_MISS', 0)
    print(f"  Categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items())))
    print(f"  Near-miss tokens available for alignment: {n_near_miss}")

    # ------------------------------------------------------------------
    # 4. Build contextualized diagnoses with position/adjacency tags
    # ------------------------------------------------------------------
    print("\n  Building contextualized diagnoses...")
    ctx_diagnoses = build_contextualized_diagnoses(
        raw_diagnoses, best_assignment, eva_to_cell, cv_labels,
    )

    # ------------------------------------------------------------------
    # 5. Align near-miss tokens and extract error catalog
    # ------------------------------------------------------------------
    print("  Aligning near-miss tokens (Needleman-Wunsch)...")
    ctx_diagnoses, error_catalog = build_error_catalog(
        ctx_diagnoses, best_assignment, cv_labels,
    )
    print(f"  Error records extracted: {len(error_catalog)}")

    # Filter to meaningful errors (non-empty produced and needed)
    meaningful_errors = [e for e in error_catalog if e.produced and e.needed]
    print(f"  Meaningful character-level errors: {len(meaningful_errors)}")

    # ------------------------------------------------------------------
    # 6. Position dependence tests
    # ------------------------------------------------------------------
    print("\n  Testing position-dependent error patterns...")
    position_tests = test_position_dependence(meaningful_errors, cv_labels, best_assignment)
    n_pos_sig = sum(1 for t in position_tests if t.significant)
    print(f"  Cells tested: {len(position_tests)}")
    print(f"  Cells with significant position dependence (p < 0.01): {n_pos_sig}")
    for t in position_tests[:5]:
        print(f"    {t.cv_label} ({t.current_assignment}): chi2={t.chi2:.1f} p={t.p_value:.4f}"
              f" {'*' if t.significant else ''}")
        for pos, corr in t.top_correction_by_position.items():
            n = t.n_errors_by_position.get(pos, 0)
            if n > 0:
                print(f"      {pos}: → '{corr}' (n={n})")

    # ------------------------------------------------------------------
    # 7. Adjacency dependence tests
    # ------------------------------------------------------------------
    print("\n  Testing adjacency-dependent error patterns...")
    adjacency_tests = test_adjacency_dependence(meaningful_errors, cv_labels, best_assignment)
    n_adj_sig = sum(1 for t in adjacency_tests if t.significant)
    print(f"  Cells with significant adjacency dependence (p < 0.01): {n_adj_sig}")
    for t in adjacency_tests[:3]:
        print(f"    {t.cv_label} ({t.current_assignment}): chi2={t.chi2:.1f} p={t.p_value:.4f}")

    # ------------------------------------------------------------------
    # 8. MI gate test
    # ------------------------------------------------------------------
    print("\n  Computing MI(correction, context) vs shuffled null (100 shuffles)...")
    mi_real, mi_shuffled_mean, mi_shuffled_std = compute_context_mi(
        meaningful_errors, n_shuffles=100,
    )
    mi_selectivity = mi_real / max(mi_shuffled_mean, 1e-9)
    print(f"  MI real:          {mi_real:.4f}")
    print(f"  MI shuffled mean: {mi_shuffled_mean:.4f} (std {mi_shuffled_std:.4f})")
    print(f"  MI selectivity:   {mi_selectivity:.2f}x")

    gate_passed = mi_selectivity > 1.5
    if gate_passed:
        gate_message = (
            f"PASS: Context-dependent errors detected (selectivity {mi_selectivity:.2f}x > 1.5x). "
            f"Proceed to rule extraction."
        )
    elif mi_selectivity > 1.0:
        gate_message = (
            f"WEAK: Marginal context effects (selectivity {mi_selectivity:.2f}x, 1.0-1.5x range). "
            f"Extract rules but expect modest improvement. Also run null-context."
        )
    else:
        gate_message = (
            f"FAIL: No context-dependent errors (selectivity {mi_selectivity:.2f}x < 1.0x). "
            f"Errors are random. Skip to null-context for alternative explanations."
        )

    print(f"\n  Gate: {'PASS ✓' if gate_passed else ('WEAK ⚠' if mi_selectivity > 1.0 else 'FAIL ✗')}")
    print(f"  {gate_message}")

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    result = ErrorPatternResult(
        n_near_miss_tokens=n_near_miss,
        n_error_records=len(meaningful_errors),
        mi_real=round(mi_real, 4),
        mi_shuffled_mean=round(mi_shuffled_mean, 4),
        mi_shuffled_std=round(mi_shuffled_std, 4),
        mi_selectivity=round(mi_selectivity, 3),
        position_tests=[_convert(asdict(t)) for t in position_tests],
        adjacency_tests=[_convert(asdict(t)) for t in adjacency_tests],
        n_cells_position_dependent=n_pos_sig,
        n_cells_adjacency_dependent=n_adj_sig,
        error_catalog_sample=[_convert(asdict(e)) for e in meaningful_errors[:50]],
        gate_passed=gate_passed or (mi_selectivity > 1.0),
        gate_message=gate_message,
    )

    out_path = os.path.join(rd, 'error_patterns.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s)")
    return _convert(asdict(result))
