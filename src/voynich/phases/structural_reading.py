"""
Step 43.9 -- Structural Reading
=================================
Produce a structural reading of the manuscript -- not a translation, but a
map of what kind of content appears where, derived from signal word positions.

Dependency chain:
    results/positional_profiles.json     (Step 43.7)
    results/cooccurrence_structure.json  (Step 43.8)
    results/signal_positions.json        (Step 43.6)
    results/signal_10k.json              (Phase 36.2)
        -> structural_reading.json        (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEDROCK_WORDS = ['bene', 'codi', 'sero', 'sene', 'de', 'raro', 'dine', 'cola']

# Structural role categories (used for folio classification)
PREPARATION_VERBS = {'cola'}          # words implying active preparation
CONNECTIVES = {'de'}                  # grammatical connectors
QUALITY_WORDS = {'bene', 'raro'}      # descriptive / quality terms


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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _folio_to_section(folio: str) -> str:
    """Heuristic section assignment from folio ID."""
    f = folio.lower().replace('f', '').rstrip('rv')
    try:
        num = int(''.join(c for c in f if c.isdigit()))
    except ValueError:
        return 'unknown'
    if num <= 56:
        return 'herbal_a'
    elif num <= 67:
        return 'pharmaceutical'
    elif num <= 73:
        return 'zodiac'
    elif num <= 84:
        return 'biological'
    elif num <= 86:
        return 'cosmological'
    elif num <= 102:
        return 'herbal_b'
    elif num <= 116:
        return 'stars'
    else:
        return 'unknown'


def _folio_sort_key(folio: str) -> int:
    """Extract a numeric sort key from a folio ID for ordering."""
    f = folio.lower().replace('f', '')
    # Strip trailing r/v and sub-page markers
    digits = ''.join(c for c in f if c.isdigit())
    try:
        base = int(digits) * 10
    except ValueError:
        return 9999
    # r=0, v=1 for ordering recto before verso
    if 'v' in f:
        base += 1
    return base


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation from two equal-length lists."""
    n = len(x)
    if n < 3:
        return 0.0
    # Convert to ranks
    def _rankdata(vals):
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-based
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rankdata(x)
    ry = _rankdata(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))
    return rho


def _spearman_p_value(rho: float, n: int) -> float:
    """Approximate two-tailed p-value for Spearman rho via t-distribution."""
    if n < 3 or abs(rho) >= 1.0:
        return 0.0 if abs(rho) >= 1.0 and n >= 3 else 1.0
    t_stat = rho * math.sqrt((n - 2) / (1.0 - rho ** 2))
    # Approximate via normal for df > 30, else use crude approximation
    df = n - 2
    # Two-tailed p-value using normal approximation of t
    z = abs(t_stat)
    if z > 6.0:
        return 0.0
    p = math.erfc(z / math.sqrt(2.0))  # two-tailed
    return round(p, 6)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class StructuralReadingResult:
    # Per-folio annotations
    folio_readings: List[Dict]  # [{folio, section, cluster_id, n_signal, n_tokens, structural_type, signal_skeleton}, ...]
    n_folios_annotated: int
    # Structural type counts
    type_counts: Dict[str, int]  # RECIPE_COLLECTION, DESCRIPTION, etc.
    # Recipe estimates
    estimated_recipe_count: int
    recipe_folios: List[str]
    mean_recipes_per_folio: float
    # Organization tests
    organization_tests: Dict[str, Dict]  # {body_part: {corr, p}, alphabetical: {corr, p}, seasonal: {corr, p}}
    best_organization: str
    # Signal skeleton patterns
    n_recurring_patterns: int
    top_patterns: List[Dict]
    # Best folios for structural reading
    best_folio: str
    best_folio_signal_rate: float
    best_folio_skeleton: List[str]
    # Structural coherence
    structural_coherence: float  # 0-1 score
    # Approach 4 verdict
    approach4_verdict: str  # STRUCTURAL_SIGNAL, WEAK_SIGNAL, NO_SIGNAL
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_structural_reading() -> None:
    """Step 43.9: Structural Reading."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.9: Structural Reading")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print("\n  1. Loading inputs ...")

    pos_prof = _safe_load(os.path.join(rd, 'positional_profiles.json'))
    cooc = _safe_load(os.path.join(rd, 'cooccurrence_structure.json'))
    sig_pos = _safe_load(os.path.join(rd, 'signal_positions.json'))
    sig_10k = _safe_load(os.path.join(rd, 'signal_10k.json'))

    if not sig_10k:
        print("  [SKIP] signal_10k.json not found")
        return

    token_decoded = sig_10k.get('token_decoded', [])
    token_folios = sig_10k.get('token_folios', [])
    n_tokens = len(token_decoded)

    print(f"     signal_10k.json: {n_tokens} tokens")
    print(f"     positional_profiles.json: {'loaded' if pos_prof else 'not found (optional)'}")
    print(f"     cooccurrence_structure.json: {'loaded' if cooc else 'not found (optional)'}")
    print(f"     signal_positions.json: {'loaded' if sig_pos else 'not found (optional)'}")

    # Role classification from positional_profiles
    role_classification = pos_prof.get('role_classification', {})
    if role_classification:
        print(f"     Role classifications: {len(role_classification)} words")

    # Cluster assignments from cooccurrence_structure
    cluster_assignments = cooc.get('cluster_assignments', {})
    recurring_patterns_raw = cooc.get('recurring_patterns', [])
    section_signal_rates = cooc.get('section_signal_rates', {})
    transition_matrix = cooc.get('transition_matrix', [])

    # Folio heat map from signal_positions
    folio_heat_map = sig_pos.get('folio_heat_map', {})
    per_word_summary = sig_pos.get('per_word_summary', {})

    bedrock_set = set(BEDROCK_WORDS)

    # ------------------------------------------------------------------
    # 2. Build per-folio token and signal data
    # ------------------------------------------------------------------
    print("\n  2. Building per-folio signal data ...")

    # Organize tokens by folio in order
    folio_tokens: Dict[str, List[str]] = defaultdict(list)
    folio_sections: Dict[str, str] = {}
    for i in range(n_tokens):
        fol = token_folios[i]
        folio_tokens[fol].append(token_decoded[i])
        if fol not in folio_sections:
            folio_sections[fol] = _folio_to_section(fol)

    all_folios = sorted(folio_tokens.keys(), key=_folio_sort_key)
    print(f"     {len(all_folios)} folios with tokens")

    # ------------------------------------------------------------------
    # 3. Per-folio structural annotation
    # ------------------------------------------------------------------
    print("\n  3. Annotating folios with signal skeletons ...")

    folio_readings: List[Dict] = []
    type_counts: Dict[str, int] = Counter()
    recipe_folios: List[str] = []
    total_recipe_estimate = 0

    # Track skeleton patterns for recurrence analysis
    skeleton_pattern_counter: Counter = Counter()

    best_folio = ''
    best_folio_signal_rate = 0.0
    best_folio_skeleton: List[str] = []

    for fol in all_folios:
        tokens = folio_tokens[fol]
        n_fol_tokens = len(tokens)
        section = folio_sections.get(fol, 'unknown')
        cluster_id = cluster_assignments.get(fol, -1)

        # Build signal skeleton: list of (word, gap_since_last)
        skeleton: List[Dict] = []
        last_signal_pos = -1
        n_signal = 0
        n_prep_verbs = 0
        n_connectives = 0
        gap_lengths: List[int] = []

        for idx, tok in enumerate(tokens):
            if tok in bedrock_set:
                gap = idx - last_signal_pos - 1 if last_signal_pos >= 0 else idx
                skeleton.append({
                    'word': tok,
                    'position': idx,
                    'gap_before': gap,
                })
                if gap > 0 and last_signal_pos >= 0:
                    gap_lengths.append(gap)
                last_signal_pos = idx
                n_signal += 1
                if tok in PREPARATION_VERBS:
                    n_prep_verbs += 1
                if tok in CONNECTIVES:
                    n_connectives += 1

        signal_rate = n_signal / max(n_fol_tokens, 1)

        # Classify structural type
        if n_signal < 3:
            structural_type = 'SPARSE'
        elif n_prep_verbs >= 2:
            structural_type = 'RECIPE_COLLECTION'
        elif n_connectives >= 3 and n_prep_verbs <= 1:
            structural_type = 'DESCRIPTION'
        elif len(gap_lengths) >= 3:
            gap_arr = np.array(gap_lengths, dtype=float)
            gap_mean = gap_arr.mean()
            gap_std = gap_arr.std()
            cv = gap_std / gap_mean if gap_mean > 0 else 999.0
            if cv < 0.5:
                structural_type = 'FORMULAIC'
            else:
                structural_type = 'UNKNOWN'
        else:
            structural_type = 'UNKNOWN'

        type_counts[structural_type] += 1

        # Recipe counting
        if structural_type == 'RECIPE_COLLECTION':
            recipe_folios.append(fol)
            # Estimate recipes per folio: each 'cola' roughly marks a recipe
            est_recipes = max(n_prep_verbs, 1)
            total_recipe_estimate += est_recipes

        # Build compact skeleton string for pattern analysis
        skeleton_words = tuple(s['word'] for s in skeleton)
        if len(skeleton_words) >= 2:
            # Track bigram sub-patterns
            for si in range(len(skeleton_words) - 1):
                bigram = (skeleton_words[si], skeleton_words[si + 1])
                skeleton_pattern_counter[bigram] += 1

        # Track best folio
        if signal_rate > best_folio_signal_rate and n_fol_tokens >= 10:
            best_folio_signal_rate = signal_rate
            best_folio = fol
            best_folio_skeleton = [s['word'] for s in skeleton]

        # Build skeleton string for output
        skeleton_str: List[str] = []
        for s in skeleton:
            if s['gap_before'] > 0:
                skeleton_str.append(f"_gap_{s['gap_before']}")
            skeleton_str.append(s['word'])

        folio_readings.append({
            'folio': fol,
            'section': section,
            'cluster_id': cluster_id,
            'n_signal': n_signal,
            'n_tokens': n_fol_tokens,
            'signal_rate': round(signal_rate, 4),
            'structural_type': structural_type,
            'signal_skeleton': skeleton_str,
            'n_prep_verbs': n_prep_verbs,
            'n_connectives': n_connectives,
        })

    n_folios_annotated = len(folio_readings)

    print(f"     Annotated {n_folios_annotated} folios")
    print(f"     Type distribution:")
    for stype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"       {stype:20s}: {cnt}")
    print(f"     Best folio: {best_folio} (signal_rate={best_folio_signal_rate:.4f})")

    # ------------------------------------------------------------------
    # 4. Recipe count estimate
    # ------------------------------------------------------------------
    print("\n  4. Recipe count estimate ...")

    n_recipe_folios = len(recipe_folios)
    mean_recipes_per_folio = (
        total_recipe_estimate / n_recipe_folios
        if n_recipe_folios > 0 else 0.0
    )

    print(f"     Recipe folios: {n_recipe_folios}")
    print(f"     Estimated total recipes: {total_recipe_estimate}")
    print(f"     Mean recipes per recipe-folio: {mean_recipes_per_folio:.2f}")
    if recipe_folios:
        print(f"     First 10 recipe folios: {recipe_folios[:10]}")

    # ------------------------------------------------------------------
    # 5. Section organization tests
    # ------------------------------------------------------------------
    print("\n  5. Testing section organization hypotheses ...")

    # For each bedrock word, compute Spearman correlation between
    # folio order and signal word rate
    folio_order = {fol: i for i, fol in enumerate(all_folios)}

    organization_tests: Dict[str, Dict] = {}

    # --- Test A: body-part hypothesis ---
    # Do certain signal words concentrate in specific folio ranges?
    # Compute per-word Spearman(folio_number, word_rate) across folios
    body_part_correlations: Dict[str, float] = {}
    body_part_p_values: Dict[str, float] = {}
    for word in BEDROCK_WORDS:
        folio_nums: List[float] = []
        word_rates: List[float] = []
        for fol in all_folios:
            tokens = folio_tokens[fol]
            n_t = len(tokens)
            if n_t < 5:
                continue
            cnt = sum(1 for t in tokens if t == word)
            folio_nums.append(float(folio_order[fol]))
            word_rates.append(cnt / n_t)
        rho = _spearman_rho(folio_nums, word_rates)
        p = _spearman_p_value(rho, len(folio_nums))
        body_part_correlations[word] = round(rho, 4)
        body_part_p_values[word] = p

    # Overall body-part signal: max absolute correlation
    max_abs_corr_word = max(body_part_correlations, key=lambda w: abs(body_part_correlations[w]))
    max_abs_corr = abs(body_part_correlations[max_abs_corr_word])

    organization_tests['body_part'] = {
        'per_word_correlations': body_part_correlations,
        'per_word_p_values': body_part_p_values,
        'strongest_word': max_abs_corr_word,
        'strongest_abs_corr': round(max_abs_corr, 4),
        'significant': max_abs_corr > 0.2 and any(
            p < 0.05 for p in body_part_p_values.values()
        ),
    }

    print(f"     Body-part hypothesis:")
    for w in BEDROCK_WORDS:
        r = body_part_correlations[w]
        p = body_part_p_values[w]
        sig = '*' if abs(r) > 0.2 and p < 0.05 else ''
        print(f"       {w:8s}: rho={r:+.4f}  p={p:.4f} {sig}")
    print(f"       Strongest: {max_abs_corr_word} (|rho|={max_abs_corr:.4f})")

    # --- Test B: alphabetical hypothesis ---
    # Do the first decoded tokens on each folio follow alphabetical order?
    # Use the first signal word on each folio and check if they are ordered
    first_signal_per_folio: List[str] = []
    first_signal_folio_idx: List[float] = []
    for fol in all_folios:
        tokens = folio_tokens[fol]
        for tok in tokens:
            if tok in bedrock_set:
                first_signal_per_folio.append(tok)
                first_signal_folio_idx.append(float(folio_order[fol]))
                break

    # Map first-letters to numeric ranks for correlation
    if first_signal_per_folio:
        first_letters = [w[0] if w else 'z' for w in first_signal_per_folio]
        letter_vals = [float(ord(c)) for c in first_letters]
        alpha_rho = _spearman_rho(first_signal_folio_idx, letter_vals)
        alpha_p = _spearman_p_value(alpha_rho, len(first_signal_folio_idx))
    else:
        alpha_rho = 0.0
        alpha_p = 1.0

    organization_tests['alphabetical'] = {
        'corr': round(alpha_rho, 4),
        'p': round(alpha_p, 6),
        'n_folios_tested': len(first_signal_per_folio),
        'significant': abs(alpha_rho) > 0.2 and alpha_p < 0.05,
    }
    print(f"     Alphabetical hypothesis: rho={alpha_rho:.4f}, p={alpha_p:.4f}, "
          f"n={len(first_signal_per_folio)}")

    # --- Test C: seasonal / section-boundary hypothesis ---
    # Does the signal word density change across sections (not just folios)?
    # Compute Spearman(section_order, total_signal_rate)
    section_order_map = {
        'herbal_a': 0, 'pharmaceutical': 1, 'zodiac': 2,
        'biological': 3, 'cosmological': 4, 'herbal_b': 5,
        'stars': 6, 'unknown': 7,
    }

    section_signal_totals: Dict[str, int] = Counter()
    section_token_totals: Dict[str, int] = Counter()
    for fol in all_folios:
        sec = folio_sections.get(fol, 'unknown')
        n_t = len(folio_tokens[fol])
        section_token_totals[sec] += n_t
        sig_count = sum(1 for t in folio_tokens[fol] if t in bedrock_set)
        section_signal_totals[sec] += sig_count

    sections_with_data = [s for s in section_order_map
                          if section_token_totals.get(s, 0) > 0]
    section_orders: List[float] = [float(section_order_map[s]) for s in sections_with_data]
    section_rates: List[float] = [
        section_signal_totals[s] / max(section_token_totals[s], 1)
        for s in sections_with_data
    ]

    if len(sections_with_data) >= 3:
        seasonal_rho = _spearman_rho(section_orders, section_rates)
        seasonal_p = _spearman_p_value(seasonal_rho, len(sections_with_data))
    else:
        seasonal_rho = 0.0
        seasonal_p = 1.0

    organization_tests['seasonal'] = {
        'corr': round(seasonal_rho, 4),
        'p': round(seasonal_p, 6),
        'section_rates': {
            s: round(section_signal_totals[s] / max(section_token_totals[s], 1), 6)
            for s in sections_with_data
        },
        'significant': abs(seasonal_rho) > 0.2 and seasonal_p < 0.05,
    }
    print(f"     Seasonal hypothesis: rho={seasonal_rho:.4f}, p={seasonal_p:.4f}")
    for s in sections_with_data:
        rate = section_signal_totals[s] / max(section_token_totals[s], 1)
        print(f"       {s:15s}: {rate*100:.2f}% signal rate "
              f"({section_signal_totals[s]}/{section_token_totals[s]})")

    # Determine best organization
    best_organization = 'none'
    best_org_strength = 0.0
    for hyp_name, hyp_data in organization_tests.items():
        if hyp_name == 'body_part':
            strength = hyp_data.get('strongest_abs_corr', 0.0)
        else:
            strength = abs(hyp_data.get('corr', 0.0))
        if strength > best_org_strength and hyp_data.get('significant', False):
            best_org_strength = strength
            best_organization = hyp_name
    if best_organization == 'none':
        # Pick highest even if not significant
        for hyp_name, hyp_data in organization_tests.items():
            if hyp_name == 'body_part':
                strength = hyp_data.get('strongest_abs_corr', 0.0)
            else:
                strength = abs(hyp_data.get('corr', 0.0))
            if strength > best_org_strength:
                best_org_strength = strength
                best_organization = hyp_name + '_weak'

    print(f"     Best organization: {best_organization} "
          f"(strength={best_org_strength:.4f})")

    # ------------------------------------------------------------------
    # 6. Signal skeleton pattern analysis
    # ------------------------------------------------------------------
    print("\n  6. Recurring skeleton patterns ...")

    # Top bigram patterns from skeleton
    top_patterns: List[Dict] = []
    for (w1, w2), cnt in skeleton_pattern_counter.most_common(20):
        if cnt >= 3:
            top_patterns.append({
                'pattern': [w1, w2],
                'count': cnt,
            })

    n_recurring_patterns = len(top_patterns)
    print(f"     {n_recurring_patterns} recurring skeleton patterns (count >= 3)")
    for tp in top_patterns[:10]:
        print(f"       {tp['pattern'][0]:8s} -> {tp['pattern'][1]:8s}: {tp['count']}")

    # ------------------------------------------------------------------
    # 7. Structural coherence score
    # ------------------------------------------------------------------
    print("\n  7. Computing structural coherence score ...")

    # Coherence is a 0-1 composite:
    #  - typed_fraction: fraction of folios with a non-UNKNOWN/non-SPARSE type
    #  - pattern_regularity: fraction of skeleton bigrams that are recurring (count >= 3)
    #  - organization_signal: best absolute correlation from section 5
    #  - type_concentration: 1 - entropy(type distribution) / log2(n_types)

    n_typed = sum(
        cnt for stype, cnt in type_counts.items()
        if stype not in ('UNKNOWN', 'SPARSE')
    )
    typed_fraction = n_typed / max(n_folios_annotated, 1)

    total_bigrams = sum(skeleton_pattern_counter.values())
    recurring_bigrams = sum(
        cnt for cnt in skeleton_pattern_counter.values()
        if cnt >= 3
    )
    pattern_regularity = recurring_bigrams / max(total_bigrams, 1)

    organization_signal = best_org_strength

    # Type concentration via entropy
    n_types = len(type_counts)
    type_total = sum(type_counts.values())
    if n_types > 1 and type_total > 0:
        type_entropy = 0.0
        for cnt in type_counts.values():
            if cnt > 0:
                p = cnt / type_total
                type_entropy -= p * math.log2(p)
        max_entropy = math.log2(n_types)
        type_concentration = 1.0 - (type_entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        type_concentration = 0.0

    # Weighted composite
    structural_coherence = (
        0.30 * typed_fraction +
        0.25 * pattern_regularity +
        0.25 * min(organization_signal * 2.0, 1.0) +  # scale so 0.5 -> 1.0
        0.20 * type_concentration
    )
    structural_coherence = round(min(max(structural_coherence, 0.0), 1.0), 4)

    print(f"     typed_fraction:     {typed_fraction:.4f}")
    print(f"     pattern_regularity: {pattern_regularity:.4f}")
    print(f"     organization_signal:{organization_signal:.4f}")
    print(f"     type_concentration: {type_concentration:.4f}")
    print(f"     structural_coherence: {structural_coherence:.4f}")

    # ------------------------------------------------------------------
    # 8. Approach 4 verdict
    # ------------------------------------------------------------------
    print("\n  8. Determining verdict ...")

    # STRUCTURAL_SIGNAL: coherence >= 0.4 and at least one significant org test
    # WEAK_SIGNAL: coherence >= 0.2 or at least 5 recipe folios
    # NO_SIGNAL: otherwise

    any_significant = any(
        hyp.get('significant', False)
        for hyp in organization_tests.values()
    )

    if structural_coherence >= 0.4 and any_significant:
        approach4_verdict = 'STRUCTURAL_SIGNAL'
    elif structural_coherence >= 0.2 or n_recipe_folios >= 5:
        approach4_verdict = 'WEAK_SIGNAL'
    else:
        approach4_verdict = 'NO_SIGNAL'

    print(f"     Verdict: {approach4_verdict}")
    print(f"     (coherence={structural_coherence:.4f}, "
          f"any_significant={any_significant}, "
          f"n_recipe_folios={n_recipe_folios})")

    # ------------------------------------------------------------------
    # 9. Structural narrative
    # ------------------------------------------------------------------
    print("\n  9. Structural narrative summary ...")

    narrative_parts: List[str] = []

    # Manuscript scope
    narrative_parts.append(
        f"The manuscript comprises {n_folios_annotated} folios with "
        f"{n_tokens} decoded tokens."
    )

    # Type breakdown
    if type_counts.get('RECIPE_COLLECTION', 0) > 0:
        narrative_parts.append(
            f"{type_counts['RECIPE_COLLECTION']} folios are classified as "
            f"RECIPE_COLLECTION (containing preparation verb 'cola'), "
            f"with an estimated {total_recipe_estimate} recipes total."
        )
    if type_counts.get('DESCRIPTION', 0) > 0:
        narrative_parts.append(
            f"{type_counts['DESCRIPTION']} folios are DESCRIPTION-type, "
            f"dominated by the connective 'de' with few preparation verbs."
        )
    if type_counts.get('FORMULAIC', 0) > 0:
        narrative_parts.append(
            f"{type_counts['FORMULAIC']} folios show FORMULAIC structure "
            f"with regularly-spaced signal words."
        )
    sparse_count = type_counts.get('SPARSE', 0) + type_counts.get('UNKNOWN', 0)
    if sparse_count > 0:
        narrative_parts.append(
            f"{sparse_count} folios are SPARSE or UNKNOWN (too few signal "
            f"words or no clear pattern)."
        )

    # Organization
    if any_significant:
        sig_tests = [
            name for name, hyp in organization_tests.items()
            if hyp.get('significant', False)
        ]
        narrative_parts.append(
            f"Significant organizational signal detected in: "
            f"{', '.join(sig_tests)}."
        )
    else:
        narrative_parts.append(
            "No significant organizational signal detected across hypotheses."
        )

    # Best folio
    if best_folio:
        narrative_parts.append(
            f"The folio with highest signal density is {best_folio} "
            f"({best_folio_signal_rate*100:.1f}% signal rate), "
            f"which may serve as the best candidate for close reading."
        )

    narrative = ' '.join(narrative_parts)
    print(f"     {narrative}")

    # ------------------------------------------------------------------
    # 10. Summary and save
    # ------------------------------------------------------------------
    runtime = round(time.time() - t0, 2)

    print("\n  " + "=" * 66)
    print(f"  Folios annotated: {n_folios_annotated}")
    print(f"  Type counts: {dict(type_counts)}")
    print(f"  Recipe folios: {n_recipe_folios}, estimated recipes: {total_recipe_estimate}")
    print(f"  Best organization: {best_organization}")
    print(f"  Recurring patterns: {n_recurring_patterns}")
    print(f"  Best folio: {best_folio} ({best_folio_signal_rate*100:.1f}%)")
    print(f"  Structural coherence: {structural_coherence}")
    print(f"  Verdict: {approach4_verdict}")
    print(f"  Runtime: {runtime}s")

    result = StructuralReadingResult(
        folio_readings=folio_readings,
        n_folios_annotated=n_folios_annotated,
        type_counts=dict(type_counts),
        estimated_recipe_count=total_recipe_estimate,
        recipe_folios=recipe_folios,
        mean_recipes_per_folio=round(mean_recipes_per_folio, 4),
        organization_tests=organization_tests,
        best_organization=best_organization,
        n_recurring_patterns=n_recurring_patterns,
        top_patterns=top_patterns,
        best_folio=best_folio,
        best_folio_signal_rate=round(best_folio_signal_rate, 4),
        best_folio_skeleton=best_folio_skeleton,
        structural_coherence=structural_coherence,
        approach4_verdict=approach4_verdict,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, 'structural_reading.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
