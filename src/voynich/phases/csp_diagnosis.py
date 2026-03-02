"""
Phase 11.5.1 – CSP failure diagnosis
======================================
Categorises every decoded token from the best Phase 11 assignment into:
  HIT        – exact match in the reference word set
  NEAR_MISS  – edit distance ≤ 2 to a reference word
  SHORT      – decoded string too short (< 2 chars)
  LONG       – decoded string too long (> 12 chars)
  ILLEGAL    – fails phonotactic legality check
  GIBBERISH  – none of the above

Also builds per-cell error profiles and correction vectors to prescribe
which cells and which relaxation strategies to apply in Phase 11.5.2-3.
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_cell_lookup,
    load_corpus,
    token_to_grid_cells,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    PHONEME_INVENTORIES,
    build_cv_syllable_table,
    load_reference_corpus,
)
from voynich.phases.csp_constraints import (
    PhonemeInventory,
    build_phoneme_inventory,
    check_phonotactic_legality,
)
from voynich.phases.csp_solver import (
    _convert,
    decode_token,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TokenDiagnosis:
    """Diagnosis for a single decoded token."""
    voynich_token: str
    decoded: str
    category: str          # HIT | NEAR_MISS | SHORT | LONG | ILLEGAL | GIBBERISH
    best_dict_match: str
    best_dict_distance: int
    cells_used: List[str]


@dataclass
class CellErrorProfile:
    """Error profile for a single grid cell."""
    cell_key: str
    cv_label: str
    current_assignment: str
    error_categories: Dict[str, int]   # category -> token count
    total_tokens: int
    error_rate: float                  # fraction of tokens NOT HIT
    dominant_error: str
    suggested_corrections: List[str]
    correction_confidence: float


@dataclass
class DiagnosisResult:
    """Full Phase 11.5.1 output."""
    n_tokens_analyzed: int
    category_counts: Dict[str, int]
    category_fractions: Dict[str, float]
    cell_error_profiles: List[Dict]
    length_mismatch_analysis: Dict
    top_correction_vectors: List[Dict]   # [{cell_key, from_syl, to_syl, expected_gain}]
    high_error_cells: List[str]          # cells with error_rate > 0.60
    diagnosis_verdict: str
    gate_passed: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _nearest_word(decoded: str, ref_words_by_len: Dict[int, List[str]], window: int = 2) -> Tuple[str, int]:
    """Find the nearest word within *window* length difference.

    Uses early-exit when distance 0 or 1 is found, and skips words whose
    length difference alone exceeds the current best distance.
    """
    target_len = len(decoded)
    best_word = ''
    best_dist = 3  # only care about dist <= 2

    for length in range(max(1, target_len - window), target_len + window + 1):
        len_diff = abs(length - target_len)
        if len_diff >= best_dist:
            continue  # length gap alone exceeds current best
        for word in ref_words_by_len.get(length, []):
            d = _edit_distance(decoded, word)
            if d < best_dist:
                best_dist = d
                best_word = word
            if d <= 1:
                return best_word, d  # good enough, stop early

    return best_word, best_dist


def _bucket_by_length(words: List[str], max_per_bucket: int = 60) -> Dict[int, List[str]]:
    """Bucket reference words by length for fast nearest-word lookup."""
    buckets: Dict[int, List[str]] = {}
    for w in words:
        l = len(w)
        if l not in buckets:
            buckets[l] = []
        if len(buckets[l]) < max_per_bucket:
            buckets[l].append(w)
    return buckets


# ---------------------------------------------------------------------------
# Token categorisation
# ---------------------------------------------------------------------------

def categorize_token(
    decoded: str,
    ref_word_set: set,
    ref_words_by_len: Dict[int, List[str]],
    inventory: PhonemeInventory,
) -> Tuple[str, str, int]:
    """Return (category, best_match, best_distance) for a decoded token."""
    if not decoded or decoded == '?':
        return 'SHORT', '', 999

    if decoded in ref_word_set:
        return 'HIT', decoded, 0

    if len(decoded) < 2:
        return 'SHORT', '', 999

    if len(decoded) > 12:
        return 'LONG', '', 999

    best_word, best_dist = _nearest_word(decoded, ref_words_by_len)
    if best_dist <= 2:
        return 'NEAR_MISS', best_word, best_dist

    if not check_phonotactic_legality(decoded, inventory):
        return 'ILLEGAL', best_word, best_dist

    return 'GIBBERISH', best_word, best_dist


# ---------------------------------------------------------------------------
# Per-cell error profiling
# ---------------------------------------------------------------------------

def _get_cells_used(token: str, eva_to_cell: Dict[str, str]) -> List[str]:
    """Return the ordered list of cell keys used to decode this token."""
    chars = tokenize_eva_chars(token)
    cells = []
    for ch in chars:
        cell = eva_to_cell.get(ch)
        if cell:
            cells.append(cell)
    return cells


def build_cell_error_profiles(
    diagnoses: List[TokenDiagnosis],
    cv_labels: Dict,
    assignment: Dict[str, str],
) -> List[CellErrorProfile]:
    """Build a per-cell error profile from token diagnoses."""
    cell_cats: Dict[str, Counter] = {}
    cell_totals: Dict[str, int] = {}

    for diag in diagnoses:
        for cell_key in diag.cells_used:
            if cell_key not in cell_cats:
                cell_cats[cell_key] = Counter()
                cell_totals[cell_key] = 0
            cell_cats[cell_key][diag.category] += 1
            cell_totals[cell_key] += 1

    profiles: List[CellErrorProfile] = []
    for cell_key, counts in cell_cats.items():
        total = cell_totals[cell_key]
        hits = counts.get('HIT', 0)
        error_rate = (total - hits) / max(total, 1)

        dominant = max(counts, key=counts.get)
        current_syl = assignment.get(cell_key, '?')
        cv_label = cv_labels.get(cell_key, {}).get('cv_label', '?')

        profiles.append(CellErrorProfile(
            cell_key=cell_key,
            cv_label=cv_label,
            current_assignment=current_syl,
            error_categories=dict(counts),
            total_tokens=total,
            error_rate=error_rate,
            dominant_error=dominant,
            suggested_corrections=[],  # filled in compute_correction_vectors
            correction_confidence=0.0,
        ))

    # Sort by error rate descending
    profiles.sort(key=lambda p: p.error_rate, reverse=True)
    return profiles


def compute_correction_vectors(
    cell_profiles: List[CellErrorProfile],
    diagnoses: List[TokenDiagnosis],
    assignment: Dict[str, str],
    eva_to_cell: Dict[str, str],
    cv_labels: Dict,
    ref_word_set: set,
    ref_words_by_len: Dict[int, List[str]],
    inventory: PhonemeInventory,
    n_candidates: int = 5,
    sample_size: int = 200,
) -> Tuple[List[CellErrorProfile], List[Dict]]:
    """Compute correction vectors for high-error cells.

    For cells dominated by NEAR_MISS errors, test candidate replacement
    syllables against a sample of tokens using that cell.

    Returns (updated_profiles, correction_vectors).
    """
    # Build token index: cell_key -> list of (token, decoded, category)
    cell_token_index: Dict[str, List[Tuple[str, str, str]]] = {}
    for diag in diagnoses:
        for ck in diag.cells_used:
            if ck not in cell_token_index:
                cell_token_index[ck] = []
            cell_token_index[ck].append((diag.voynich_token, diag.decoded, diag.category))

    cv_syllables = inventory.cv_syllables
    correction_vectors: List[Dict] = []

    for profile in cell_profiles:
        if profile.error_rate < 0.30:
            continue  # low-error cells don't need correction

        cell_key = profile.cell_key
        tokens_with_cell = cell_token_index.get(cell_key, [])
        if not tokens_with_cell:
            continue

        # Sample near-miss tokens for this cell
        near_miss_tokens = [
            (tok, dec) for tok, dec, cat in tokens_with_cell
            if cat in ('NEAR_MISS', 'GIBBERISH')
        ][:sample_size]

        if not near_miss_tokens:
            continue

        current_syl = assignment.get(cell_key, '')
        best_gain = 0.0
        best_candidates: List[Tuple[str, float]] = []

        # Test up to n_candidates replacement syllables (frequency-ranked)
        candidates = [s for s in cv_syllables if s != current_syl][:15]
        for cand_syl in candidates:
            # Create modified assignment
            test_asgn = dict(assignment)
            test_asgn[cell_key] = cand_syl

            # Re-decode and score
            n_improved = 0
            for token, _old_dec in near_miss_tokens:
                new_decoded = decode_token(token, test_asgn, eva_to_cell)
                if new_decoded in ref_word_set:
                    n_improved += 1
                else:
                    _, new_dist = _nearest_word(new_decoded, ref_words_by_len)
                    _, old_dist = _nearest_word(_old_dec, ref_words_by_len)
                    if new_dist < old_dist:
                        n_improved += 0.5

            gain = n_improved / max(len(near_miss_tokens), 1)
            if gain > 0:
                best_candidates.append((cand_syl, gain))

        best_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [c for c, _ in best_candidates[:n_candidates]]
        top_gain = best_candidates[0][1] if best_candidates else 0.0

        profile.suggested_corrections = top_candidates
        profile.correction_confidence = top_gain

        if top_candidates and top_gain > best_gain:
            best_gain = top_gain
            correction_vectors.append({
                'cell_key': cell_key,
                'cv_label': profile.cv_label,
                'from_syl': current_syl,
                'to_syl': top_candidates[0] if top_candidates else '',
                'expected_gain': round(top_gain, 4),
                'dominant_error': profile.dominant_error,
            })

    correction_vectors.sort(key=lambda x: x['expected_gain'], reverse=True)
    return cell_profiles, correction_vectors


# ---------------------------------------------------------------------------
# Length mismatch analysis
# ---------------------------------------------------------------------------

def compute_length_mismatch(
    diagnoses: List[TokenDiagnosis],
) -> Dict:
    """Analyse decoded string lengths vs expected word lengths."""
    lengths: Dict[str, List[int]] = {cat: [] for cat in
                                     ('HIT', 'NEAR_MISS', 'SHORT', 'LONG', 'ILLEGAL', 'GIBBERISH')}

    for diag in diagnoses:
        lengths[diag.category].append(len(diag.decoded))

    result: Dict[str, Any] = {}
    for cat, lens in lengths.items():
        if not lens:
            result[cat] = {'n': 0, 'mean': 0.0, 'min': 0, 'max': 0}
        else:
            result[cat] = {
                'n': len(lens),
                'mean': round(sum(lens) / len(lens), 2),
                'min': min(lens),
                'max': max(lens),
            }

    # Overall length ratio: decoded / expected average Latin word (~6 chars)
    all_lens = [len(d.decoded) for d in diagnoses if d.decoded]
    expected_avg = 6.0
    result['overall_mean_length'] = round(sum(all_lens) / max(len(all_lens), 1), 2)
    result['expected_avg_latin_length'] = expected_avg
    result['mean_length_ratio'] = round(
        result['overall_mean_length'] / expected_avg, 3
    )

    if result['mean_length_ratio'] < 0.7:
        result['interpretation'] = 'SYSTEMATICALLY_SHORT: model underproduces syllables'
    elif result['mean_length_ratio'] > 1.4:
        result['interpretation'] = 'SYSTEMATICALLY_LONG: model overproduces syllables'
    else:
        result['interpretation'] = 'MIXED: variable length mismatch'

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_csp_diagnosis() -> Dict:
    """Phase 11.5.1: Categorise decoded tokens and build per-cell error profiles.

    Loads the best Phase 11 assignment from csp_decode.json, decodes up
    to 3000 Language A tokens, categorises each, and saves results to
    results/csp_diagnosis.json.
    """
    print("=" * 70)
    print("PHASE 11.5.1: CSP Failure Diagnosis")
    print("=" * 70)

    t0 = time.time()
    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load Phase 11 results
    # ------------------------------------------------------------------
    decode_path = os.path.join(rd, 'csp_decode.json')
    if not os.path.exists(decode_path):
        print("  [SKIP] csp_decode.json not found — run csp-decode first")
        return {'verdict': 'skipped', 'reason': 'no_csp_decode'}

    with open(decode_path) as f:
        decode_data = json.load(f)

    best_assignment: Dict[str, str] = decode_data.get('best_assignment', {})
    eva_to_cell_map: Dict[str, str] = decode_data.get('eva_to_cell_mapping', {})

    if not best_assignment:
        print("  [SKIP] No best_assignment in csp_decode.json")
        return {'verdict': 'skipped', 'reason': 'no_assignment'}

    print(f"  Loaded best assignment ({len(best_assignment)} cells)")
    print(f"  Phase 11 dict_hit_rate: {decode_data.get('best_dict_hit', 0):.4f}")

    # ------------------------------------------------------------------
    # 2. Load supporting data
    # ------------------------------------------------------------------
    cv_path = os.path.join(rd, 'cv_labels.json')
    with open(cv_path) as f:
        cv_labels = json.load(f)

    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    corpus_tokens = corpus.get_tokens(language='A', paragraph_only=True)
    corpus_tokens = corpus_tokens[:1500]
    print(f"  Corpus tokens to analyse: {len(corpus_tokens)}")

    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_word_set: set = set(ref_tokens[:50000])
    print(f"  Reference word set size: {len(ref_word_set)}")

    ref_words_by_len = _bucket_by_length(ref_tokens[:10000], max_per_bucket=60)

    # Build phoneme inventory (strict CV, Level 0)
    inventory = build_phoneme_inventory('latin', ref_corpus)

    # Build eva_to_cell (use loaded mapping if available, else rebuild)
    if eva_to_cell_map:
        eva_to_cell = eva_to_cell_map
    else:
        from voynich.core.corpus import build_eva_to_cell_lookup
        eva_to_cell = build_eva_to_cell_lookup(cv_labels)

    # ------------------------------------------------------------------
    # 3. Decode and categorise all tokens
    # ------------------------------------------------------------------
    print("\n  Categorising tokens...")
    categories = ['HIT', 'NEAR_MISS', 'SHORT', 'LONG', 'ILLEGAL', 'GIBBERISH']
    category_counts: Counter = Counter()
    diagnoses: List[TokenDiagnosis] = []

    for token in corpus_tokens:
        decoded = decode_token(token, best_assignment, eva_to_cell)
        cells_used = _get_cells_used(token, eva_to_cell)
        cat, best_match, best_dist = categorize_token(
            decoded, ref_word_set, ref_words_by_len, inventory,
        )
        category_counts[cat] += 1
        diagnoses.append(TokenDiagnosis(
            voynich_token=token,
            decoded=decoded,
            category=cat,
            best_dict_match=best_match,
            best_dict_distance=best_dist,
            cells_used=cells_used,
        ))

    n_total = len(diagnoses)
    category_fractions = {
        cat: round(category_counts.get(cat, 0) / max(n_total, 1), 4)
        for cat in categories
    }

    print(f"\n  Token category distribution (n={n_total}):")
    for cat in categories:
        cnt = category_counts.get(cat, 0)
        frac = category_fractions[cat]
        bar = '#' * int(frac * 40)
        print(f"    {cat:12s}: {cnt:5d} ({frac:.1%}) {bar}")

    # ------------------------------------------------------------------
    # 4. Build per-cell error profiles
    # ------------------------------------------------------------------
    print("\n  Building cell error profiles...")
    cell_profiles = build_cell_error_profiles(diagnoses, cv_labels, best_assignment)

    print(f"\n  Per-cell error rates (sorted by error rate):")
    for p in cell_profiles[:14]:
        print(f"    {p.cv_label:6s} ({p.current_assignment:5s})  "
              f"error={p.error_rate:.1%}  dominant={p.dominant_error}")

    # ------------------------------------------------------------------
    # 5. Compute correction vectors
    # ------------------------------------------------------------------
    print("\n  Computing correction vectors...")
    cell_profiles, correction_vectors = compute_correction_vectors(
        cell_profiles, diagnoses, best_assignment, eva_to_cell,
        cv_labels, ref_word_set, ref_words_by_len, inventory,
    )

    if correction_vectors:
        print(f"\n  Top correction vectors:")
        for cv_entry in correction_vectors[:5]:
            print(f"    {cv_entry['cv_label']} ({cv_entry['from_syl']}) → "
                  f"{cv_entry['to_syl']}  gain={cv_entry['expected_gain']:.3f}")

    # ------------------------------------------------------------------
    # 6. Length mismatch analysis
    # ------------------------------------------------------------------
    length_analysis = compute_length_mismatch(diagnoses)
    print(f"\n  Length mismatch: mean_decoded={length_analysis['overall_mean_length']:.1f} "
          f"chars, ratio={length_analysis['mean_length_ratio']:.2f}x")
    print(f"  Interpretation: {length_analysis['interpretation']}")

    # ------------------------------------------------------------------
    # 7. Identify high-error cells
    # ------------------------------------------------------------------
    high_error_cells = [p.cell_key for p in cell_profiles if p.error_rate > 0.60]
    print(f"\n  High-error cells (error > 60%): {len(high_error_cells)}")
    for p in cell_profiles:
        if p.error_rate > 0.60:
            print(f"    {p.cv_label} ({p.current_assignment}): {p.error_rate:.1%}")

    # ------------------------------------------------------------------
    # 8. Gate check
    # ------------------------------------------------------------------
    hit_frac = category_fractions.get('HIT', 0.0)
    near_frac = category_fractions.get('NEAR_MISS', 0.0)
    signal_frac = hit_frac + near_frac
    gate_passed = signal_frac >= 0.15

    print(f"\n  Signal fraction (HIT + NEAR_MISS): {signal_frac:.1%}")
    print(f"  Gate threshold: 15.0%")
    print(f"  Gate: {'PASS' if gate_passed else 'FAIL'}")

    # Prescriptive verdict
    short_frac = category_fractions.get('SHORT', 0.0)
    near_frac2 = category_fractions.get('NEAR_MISS', 0.0)
    illegal_frac = category_fractions.get('ILLEGAL', 0.0)
    gibberish_frac = category_fractions.get('GIBBERISH', 0.0)

    if near_frac2 > 0.30:
        verdict = 'NEAR_MISS_DOMINANT: correction vectors directly prescribe fixes'
    elif short_frac > 0.30:
        verdict = 'SHORT_DOMINANT: inherent vowel and CVC relaxation are priority'
    elif illegal_frac > 0.20:
        verdict = 'ILLEGAL_DOMINANT: phonotactic constraints too loose'
    elif gibberish_frac > 0.50:
        verdict = 'GIBBERISH_DOMINANT: most cells wrong, fix high-error cells first'
    else:
        verdict = 'MIXED_ERRORS: apply graduated relaxation sweep'

    print(f"\n  Diagnostic verdict: {verdict}")

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    result = DiagnosisResult(
        n_tokens_analyzed=n_total,
        category_counts=dict(category_counts),
        category_fractions=category_fractions,
        cell_error_profiles=[_convert(asdict(p)) for p in cell_profiles],
        length_mismatch_analysis=length_analysis,
        top_correction_vectors=correction_vectors[:10],
        high_error_cells=high_error_cells,
        diagnosis_verdict=verdict,
        gate_passed=gate_passed,
    )

    out_path = os.path.join(rd, 'csp_diagnosis.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Saved to {out_path} ({elapsed:.1f}s)")
    print(f"\n  Gate: {'PASS ✓' if gate_passed else 'FAIL ✗'}")

    return _convert(asdict(result))
