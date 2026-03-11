"""
Step 43.7 -- Structural Role Classification
=============================================
Classify each signal word's structural role based on its positional profile.

Dependency chain:
    results/signal_positions.json     (Step 43.6)
    results/signal_10k.json           (Phase 36.2: decoded tokens)
        -> positional_profiles.json    (this step)
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PositionalProfileResult:
    per_word_profiles: Dict[str, Dict]       # word -> {features + classification}
    role_classification: Dict[str, str]      # word -> structural_role
    n_function_words: int
    n_content_words: int
    n_formulaic: int
    n_unknown: int
    feature_summary: Dict[str, Dict[str, float]]  # word -> feature values
    context_analysis: Dict[str, Dict]        # word -> {top_left_contexts, top_right_contexts}
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Structural role templates
# ---------------------------------------------------------------------------

ROLE_TEMPLATES = {
    'CONNECTIVE': {
        'description': 'Uniform distribution, no positional preference, high frequency',
        'category': 'function',
    },
    'RECIPE_MARKER': {
        'description': 'Clusters at paragraph starts, regular intervals, high line-initial rate',
        'category': 'formulaic',
    },
    'PREPARATION_VERB': {
        'description': 'Appears after ingredient-like words, section-specific',
        'category': 'content',
    },
    'QUALITY': {
        'description': 'Clusters in description sections, appears after prepositions',
        'category': 'content',
    },
    'INGREDIENT': {
        'description': 'Clusters on specific folios, rare on others, high section specificity',
        'category': 'content',
    },
    'SECTION_MARKER': {
        'description': 'Appears at folio starts, first lines',
        'category': 'formulaic',
    },
    'DOSAGE': {
        'description': 'Appears at recipe ends, near quantities',
        'category': 'content',
    },
    'STRUCTURAL_UNKNOWN': {
        'description': 'None of the above match',
        'category': 'unknown',
    },
}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_section_entropy(section_counts: Dict[str, int]) -> float:
    """Shannon entropy of the section distribution for a word."""
    total = sum(section_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in section_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def _compute_uniformity_cv(
    word: str,
    folio_heat_map: Dict[str, Dict[str, int]],
) -> float:
    """Coefficient of variation of per-folio frequency.

    Low = uniform (CONNECTIVE), high = concentrated (INGREDIENT).
    Only counts folios where the word appears at least once.
    We use ALL folios (including zeros) for true CV.
    """
    counts = []
    for folio, word_counts in folio_heat_map.items():
        counts.append(word_counts.get(word, 0))
    if not counts:
        return 0.0
    arr = np.array(counts, dtype=float)
    mean_val = arr.mean()
    if mean_val == 0.0:
        return 0.0
    std_val = arr.std()
    return float(std_val / mean_val)


def _compute_inter_occurrence_regularity(
    positions: List[int],
) -> float:
    """Regularity = std / mean of gaps between consecutive occurrences.

    Low = regular spacing (RECIPE_MARKER), high = irregular.
    """
    if len(positions) < 2:
        return 999.0  # Not enough data
    sorted_pos = sorted(positions)
    gaps = [sorted_pos[i + 1] - sorted_pos[i] for i in range(len(sorted_pos) - 1)]
    if not gaps:
        return 999.0
    arr = np.array(gaps, dtype=float)
    mean_gap = arr.mean()
    if mean_gap == 0.0:
        return 999.0
    return float(arr.std() / mean_gap)


def _compute_word_features(
    word: str,
    per_word_summary: Dict[str, Dict],
    folio_heat_map: Dict[str, Dict[str, int]],
    token_decoded: List[str],
    token_folios: List[str],
    all_signal_words: List[str],
    frequency_rank: int,
) -> Dict[str, float]:
    """Compute all positional features for a signal word."""
    summary = per_word_summary.get(word, {})
    count = summary.get('count', 0)
    sections = summary.get('sections', {})
    mean_pos = summary.get('mean_relative_pos', 0.5)

    # uniformity_cv
    uniformity_cv = _compute_uniformity_cv(word, folio_heat_map)

    # section_entropy
    section_entropy = _compute_section_entropy(sections)

    # position_bias
    position_bias = mean_pos

    # line_initial_rate: fraction of occurrences at token_index 0 in folio
    # We approximate this by finding all positions of the word in token_decoded
    # and checking if it's the first token of its folio
    n_line_initial = 0
    n_total = 0
    word_global_positions = []  # For inter-occurrence regularity

    prev_folio = None
    folio_first_index = 0
    for i, (dec, fol) in enumerate(zip(token_decoded, token_folios)):
        if fol != prev_folio:
            folio_first_index = i
            prev_folio = fol
        if dec == word:
            n_total += 1
            word_global_positions.append(i)
            if i == folio_first_index:
                n_line_initial += 1

    line_initial_rate = n_line_initial / n_total if n_total > 0 else 0.0

    # inter_occurrence_regularity
    inter_occurrence_regularity = _compute_inter_occurrence_regularity(
        word_global_positions,
    )

    return {
        'uniformity_cv': round(uniformity_cv, 4),
        'line_initial_rate': round(line_initial_rate, 4),
        'section_entropy': round(section_entropy, 4),
        'inter_occurrence_regularity': round(inter_occurrence_regularity, 4),
        'position_bias': round(position_bias, 4),
        'frequency_rank': frequency_rank,
        'count': count,
    }


# ---------------------------------------------------------------------------
# Context analysis
# ---------------------------------------------------------------------------

def _compute_context(
    word: str,
    token_decoded: List[str],
    token_folios: List[str],
    top_n: int = 5,
) -> Dict:
    """Extract left and right context words for a signal word."""
    left_context: Counter = Counter()
    right_context: Counter = Counter()
    n_tokens = len(token_decoded)

    for i in range(n_tokens):
        if token_decoded[i] != word:
            continue
        # Left context (same folio)
        if i > 0 and token_folios[i] == token_folios[i - 1]:
            left_context[token_decoded[i - 1]] += 1
        # Right context (same folio)
        if i < n_tokens - 1 and token_folios[i] == token_folios[i + 1]:
            right_context[token_decoded[i + 1]] += 1

    left_diversity = len(left_context)
    right_diversity = len(right_context)

    return {
        'top_left_contexts': [
            {'word': w, 'count': c}
            for w, c in left_context.most_common(top_n)
        ],
        'top_right_contexts': [
            {'word': w, 'count': c}
            for w, c in right_context.most_common(top_n)
        ],
        'left_context_diversity': left_diversity,
        'right_context_diversity': right_diversity,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _score_template(
    features: Dict[str, float],
    context: Dict,
) -> Dict[str, float]:
    """Score each structural role template for a word.

    Returns a dict of role -> confidence score (0.0 to 1.0).
    Higher = better match.
    """
    scores: Dict[str, float] = {}

    ucv = features['uniformity_cv']
    lir = features['line_initial_rate']
    se = features['section_entropy']
    ior = features['inter_occurrence_regularity']
    pb = features['position_bias']
    fk = features['frequency_rank']
    rcd = context.get('right_context_diversity', 0)

    # CONNECTIVE: uniform, high entropy, high frequency
    connective_score = 0.0
    if se > 1.5:
        connective_score += 0.35
    elif se > 1.2:
        connective_score += 0.15
    if ucv < 1.0:
        connective_score += 0.30
    elif ucv < 1.5:
        connective_score += 0.10
    if fk <= 3:
        connective_score += 0.35
    elif fk <= 5:
        connective_score += 0.15
    scores['CONNECTIVE'] = round(connective_score, 4)

    # RECIPE_MARKER: line-initial, regular spacing
    recipe_score = 0.0
    if lir > 0.15:
        recipe_score += 0.50
    elif lir > 0.08:
        recipe_score += 0.20
    if ior < 0.5:
        recipe_score += 0.50
    elif ior < 0.8:
        recipe_score += 0.20
    scores['RECIPE_MARKER'] = round(recipe_score, 4)

    # PREPARATION_VERB: low section entropy, limited right context
    prep_score = 0.0
    if se < 1.5:
        prep_score += 0.40
    elif se < 2.0:
        prep_score += 0.15
    if rcd < 15:
        prep_score += 0.30
    if 0.3 <= pb <= 0.7:
        prep_score += 0.30
    scores['PREPARATION_VERB'] = round(prep_score, 4)

    # QUALITY: mid-range position, moderate entropy
    quality_score = 0.0
    if 0.3 <= pb <= 0.7:
        quality_score += 0.40
    if se > 1.2:
        quality_score += 0.30
    elif se > 0.8:
        quality_score += 0.15
    if ucv < 1.5:
        quality_score += 0.30
    scores['QUALITY'] = round(quality_score, 4)

    # INGREDIENT: concentrated (high CV), low section entropy
    ingredient_score = 0.0
    if ucv > 1.5:
        ingredient_score += 0.50
    elif ucv > 1.0:
        ingredient_score += 0.20
    if se < 1.2:
        ingredient_score += 0.50
    elif se < 1.5:
        ingredient_score += 0.25
    scores['INGREDIENT'] = round(ingredient_score, 4)

    # SECTION_MARKER: very high line-initial rate, position bias toward start
    section_score = 0.0
    if lir > 0.20:
        section_score += 0.50
    elif lir > 0.10:
        section_score += 0.20
    if pb < 0.35:
        section_score += 0.50
    elif pb < 0.45:
        section_score += 0.20
    scores['SECTION_MARKER'] = round(section_score, 4)

    # DOSAGE: end-biased position, low section entropy
    dosage_score = 0.0
    if pb > 0.55:
        dosage_score += 0.50
    elif pb > 0.45:
        dosage_score += 0.15
    if se < 1.5:
        dosage_score += 0.50
    elif se < 2.0:
        dosage_score += 0.20
    scores['DOSAGE'] = round(dosage_score, 4)

    return scores


def _classify_word(
    word: str,
    features: Dict[str, float],
    context: Dict,
) -> Dict:
    """Classify a word into its best-matching structural role."""
    template_scores = _score_template(features, context)

    # Sort by score descending
    ranked = sorted(template_scores.items(), key=lambda x: -x[1])
    best_role, best_score = ranked[0]
    second_role, second_score = ranked[1] if len(ranked) > 1 else ('', 0.0)

    # If best score is too low, assign STRUCTURAL_UNKNOWN
    if best_score < 0.35:
        best_role = 'STRUCTURAL_UNKNOWN'
        best_score = 0.0

    # Confidence: how much better is best vs second
    confidence = best_score - second_score if best_role != 'STRUCTURAL_UNKNOWN' else 0.0

    category = ROLE_TEMPLATES.get(best_role, {}).get('category', 'unknown')

    return {
        'role': best_role,
        'score': round(best_score, 4),
        'confidence': round(confidence, 4),
        'category': category,
        'template_scores': template_scores,
        'runner_up': second_role,
        'runner_up_score': round(second_score, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_positional_profiles() -> None:
    """Step 43.7: Structural Role Classification."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.7: Structural Role Classification")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

    sp_path = os.path.join(rd, 'signal_positions.json')
    sp_data = _safe_load(sp_path)
    if not sp_data:
        print(f"  [SKIP] signal_positions.json not found at {sp_path}")
        return

    signal_words = sp_data.get('signal_words', [])
    per_word_summary = sp_data.get('per_word_summary', {})
    folio_heat_map = sp_data.get('folio_heat_map', {})
    per_section_summary = sp_data.get('per_section_summary', {})

    print(f"     Signal words: {signal_words}")
    print(f"     Per-word summary: {len(per_word_summary)} entries")
    print(f"     Folio heat map: {len(folio_heat_map)} folios")

    # Load signal_10k for decoded token arrays
    s10k_path = os.path.join(rd, 'signal_10k.json')
    s10k_data = _safe_load(s10k_path)
    if not s10k_data:
        print(f"  [SKIP] signal_10k.json not found at {s10k_path}")
        return

    token_decoded = s10k_data.get('token_decoded', [])
    token_folios = s10k_data.get('token_folios', [])
    token_classifications = s10k_data.get('token_classifications', [])

    print(f"     Decoded tokens: {len(token_decoded)}")
    print(f"     Token folios: {len(token_folios)}")

    # ── 2. Rank words by frequency ──
    print("\n  2. Ranking words by frequency ...")

    word_counts = {
        w: per_word_summary.get(w, {}).get('count', 0)
        for w in signal_words
    }
    ranked_words = sorted(word_counts.items(), key=lambda x: -x[1])
    frequency_ranks: Dict[str, int] = {}
    for rank, (word, count) in enumerate(ranked_words, start=1):
        frequency_ranks[word] = rank
        print(f"     Rank {rank}: {word:8s} ({count} occurrences)")

    # ── 3. Compute features for each word ──
    print("\n  3. Computing positional features ...")

    feature_summary: Dict[str, Dict[str, float]] = {}

    for word in signal_words:
        features = _compute_word_features(
            word=word,
            per_word_summary=per_word_summary,
            folio_heat_map=folio_heat_map,
            token_decoded=token_decoded,
            token_folios=token_folios,
            all_signal_words=signal_words,
            frequency_rank=frequency_ranks.get(word, 8),
        )
        feature_summary[word] = features

        print(f"     {word:8s}: CV={features['uniformity_cv']:.3f}  "
              f"LIR={features['line_initial_rate']:.3f}  "
              f"SE={features['section_entropy']:.3f}  "
              f"IOR={features['inter_occurrence_regularity']:.3f}  "
              f"PB={features['position_bias']:.3f}  "
              f"Rank={features['frequency_rank']}")

    # ── 4. Context analysis ──
    print("\n  4. Analyzing left/right contexts ...")

    context_analysis: Dict[str, Dict] = {}
    for word in signal_words:
        ctx = _compute_context(
            word=word,
            token_decoded=token_decoded,
            token_folios=token_folios,
            top_n=5,
        )
        context_analysis[word] = ctx

        left_str = ', '.join(
            f"{c['word']}({c['count']})"
            for c in ctx['top_left_contexts'][:3]
        )
        right_str = ', '.join(
            f"{c['word']}({c['count']})"
            for c in ctx['top_right_contexts'][:3]
        )
        print(f"     {word:8s}: L=[{left_str}]  R=[{right_str}]  "
              f"Ldiv={ctx['left_context_diversity']}  "
              f"Rdiv={ctx['right_context_diversity']}")

    # ── 5. Classify each word ──
    print("\n  5. Classifying structural roles ...")

    role_classification: Dict[str, str] = {}
    per_word_profiles: Dict[str, Dict] = {}

    for word in signal_words:
        features = feature_summary[word]
        ctx = context_analysis[word]
        classification = _classify_word(word, features, ctx)

        role_classification[word] = classification['role']

        per_word_profiles[word] = {
            'features': features,
            'context': ctx,
            'classification': classification,
        }

        print(f"     {word:8s} -> {classification['role']:20s} "
              f"(score={classification['score']:.3f}, "
              f"conf={classification['confidence']:.3f}, "
              f"cat={classification['category']}, "
              f"runner_up={classification['runner_up']})")

    # ── 6. Tally category counts ──
    print("\n  6. Category summary ...")

    n_function_words = 0
    n_content_words = 0
    n_formulaic = 0
    n_unknown = 0

    for word in signal_words:
        cat = per_word_profiles[word]['classification']['category']
        if cat == 'function':
            n_function_words += 1
        elif cat == 'content':
            n_content_words += 1
        elif cat == 'formulaic':
            n_formulaic += 1
        else:
            n_unknown += 1

    print(f"     Function words: {n_function_words}")
    print(f"     Content words:  {n_content_words}")
    print(f"     Formulaic:      {n_formulaic}")
    print(f"     Unknown:        {n_unknown}")

    # ── 7. Expected vs actual comparison ──
    print("\n  7. Expected vs actual classification ...")

    expected_roles = {
        'de': 'CONNECTIVE',
        'cola': 'PREPARATION_VERB',
        'bene': 'QUALITY',
        'sene': 'INGREDIENT',     # or CONNECTIVE ("without")
        'sero': 'DOSAGE',         # "serum" / "late/evening"
        'codi': 'STRUCTURAL_UNKNOWN',
        'raro': 'QUALITY',
        'dine': 'STRUCTURAL_UNKNOWN',
    }

    n_match = 0
    for word in signal_words:
        actual = role_classification.get(word, 'STRUCTURAL_UNKNOWN')
        expected = expected_roles.get(word, 'STRUCTURAL_UNKNOWN')
        match = actual == expected
        if match:
            n_match += 1
        tag = 'MATCH' if match else 'DIFFER'
        print(f"     {word:8s}: expected={expected:20s}  "
              f"actual={actual:20s}  [{tag}]")

    print(f"\n     Agreement: {n_match}/{len(signal_words)} "
          f"({100.0 * n_match / len(signal_words) if signal_words else 0:.1f}%)")

    # ── 8. Summary ──
    elapsed = round(time.time() - t0, 2)

    print("\n  " + "=" * 66)
    print(f"  Signal words classified: {len(signal_words)}")
    print(f"  Function: {n_function_words}  Content: {n_content_words}  "
          f"Formulaic: {n_formulaic}  Unknown: {n_unknown}")
    print(f"  Expected agreement: {n_match}/{len(signal_words)}")
    print(f"  Runtime: {elapsed}s")

    # ── 9. Save ──
    result = PositionalProfileResult(
        per_word_profiles=per_word_profiles,
        role_classification=role_classification,
        n_function_words=n_function_words,
        n_content_words=n_content_words,
        n_formulaic=n_formulaic,
        n_unknown=n_unknown,
        feature_summary=feature_summary,
        context_analysis=context_analysis,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'positional_profiles.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
