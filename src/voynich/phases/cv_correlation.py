"""
Step 37.2 – Within-Class Selectivity Correlation
==================================================
Test whether signal words sharing a consonant onset show correlated signal
patterns — the key prediction of the consonant-correct-vowel-wrong hypothesis.

Dependency chain:
    consonant_grouping.json    (Step 37.1)
    signal_10k.json            (Step 36.2)
    decode_10k.json            (Step 36.1)
        → cv_correlation.json  (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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


def _pearson(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    sy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cv_correlation() -> None:
    """Step 37.2: Within-Class Selectivity Correlation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.2: Within-Class Selectivity Correlation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    cg_data = _safe_load(os.path.join(rd, 'consonant_grouping.json'))
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))

    consonant_groups = cg_data.get('consonant_groups', [])
    token_decoded = signal_data.get('token_decoded', [])
    token_folios = signal_data.get('token_folios', [])
    token_classifications = signal_data.get('token_classifications', [])
    word_signals = signal_data.get('word_signals', [])
    genuine_words = set(w['word'] for w in word_signals if w.get('is_genuine_signal'))

    print(f"     {len(consonant_groups)} consonant groups")
    print(f"     {len(token_decoded)} tokens")
    print(f"     {len(genuine_words)} genuine signal words")

    # ── 2. Build per-folio frequency vectors ──
    print("  2. Building folio-level frequency vectors …")
    all_folios = sorted(set(token_folios))
    folio_idx = {f: i for i, f in enumerate(all_folios)}
    n_folios = len(all_folios)

    # For each genuine signal word, build a folio-frequency vector
    word_folio_vec: Dict[str, List[float]] = {}
    for word in genuine_words:
        vec = [0.0] * n_folios
        for i, (w, fol) in enumerate(zip(token_decoded, token_folios)):
            if w == word:
                vec[folio_idx[fol]] += 1.0
        word_folio_vec[word] = vec

    # ── 3. Compute within-group correlations ──
    print("  3. Computing within-group correlations …")
    within_corrs = []
    between_corrs = []
    group_results = []

    for cg in consonant_groups:
        words = [w for w in cg['words'] if w in word_folio_vec]
        if len(words) < 2:
            group_results.append({
                'onset': cg['onset'],
                'n_words': len(words),
                'mean_within_corr': None,
                'pairs_tested': 0,
            })
            continue

        # Within-group: all pairs within this group
        pair_corrs = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                r = _pearson(word_folio_vec[words[i]], word_folio_vec[words[j]])
                pair_corrs.append(r)
                within_corrs.append(r)

        mean_within = sum(pair_corrs) / len(pair_corrs) if pair_corrs else 0.0
        group_results.append({
            'onset': cg['onset'],
            'n_words': len(words),
            'words': words,
            'mean_within_corr': round(mean_within, 4),
            'pairs_tested': len(pair_corrs),
            'pair_correlations': [round(r, 4) for r in pair_corrs],
        })

        print(f"     {cg['onset']:<10s} {len(words)} words, "
              f"mean within-corr={mean_within:.4f} ({len(pair_corrs)} pairs)")

    # ── 4. Compute between-group correlations ──
    print("  4. Computing between-group correlations …")
    all_group_words = []
    for cg in consonant_groups:
        group_words = [w for w in cg['words'] if w in word_folio_vec]
        if group_words:
            all_group_words.append((cg['onset'], group_words))

    for i in range(len(all_group_words)):
        for j in range(i + 1, len(all_group_words)):
            _, words_i = all_group_words[i]
            _, words_j = all_group_words[j]
            for wi in words_i:
                for wj in words_j:
                    r = _pearson(word_folio_vec[wi], word_folio_vec[wj])
                    between_corrs.append(r)

    mean_within = sum(within_corrs) / len(within_corrs) if within_corrs else 0.0
    mean_between = sum(between_corrs) / len(between_corrs) if between_corrs else 0.0

    print(f"     Mean within-group:  {mean_within:.4f} ({len(within_corrs)} pairs)")
    print(f"     Mean between-group: {mean_between:.4f} ({len(between_corrs)} pairs)")

    within_exceeds_between = mean_within > mean_between

    # ── 5. Selectivity variance test ──
    print("  5. Selectivity variance test (within vs between class) …")
    # Within-class selectivity variance: variance of selectivity within each group
    within_vars = []
    between_sels = []
    group_means = []
    for cg in consonant_groups:
        sels = cg.get('selectivities', [])
        if len(sels) >= 2:
            m = sum(sels) / len(sels)
            v = sum((s - m) ** 2 for s in sels) / len(sels)
            within_vars.append(v)
            group_means.append(m)
        elif len(sels) == 1:
            group_means.append(sels[0])

    # Between-class variance: variance of the group means
    if len(group_means) >= 2:
        overall_mean = sum(group_means) / len(group_means)
        between_var = sum((m - overall_mean) ** 2 for m in group_means) / len(group_means)
    else:
        between_var = 0.0

    mean_within_var = sum(within_vars) / len(within_vars) if within_vars else 0.0

    # F-statistic: between / within
    f_stat = between_var / mean_within_var if mean_within_var > 0 else float('inf')

    print(f"     Mean within-class variance: {mean_within_var:.4f}")
    print(f"     Between-class variance:     {between_var:.4f}")
    print(f"     F-statistic:                {f_stat:.4f}")

    # Under C5×V4 hypothesis, within-class variance should be LOW
    # (all words in a class have ~same selectivity because consonant is correct)
    within_lower = mean_within_var < between_var

    # ── 6. Theoretical selectivity comparison ──
    print("  6. Theoretical selectivity comparison …")
    overall_mean_sel = cg_data.get('overall_mean_selectivity', 0.0)
    n_cons_classes = cg_data.get('n_consonant_only_classes', 0)

    # Under C×V model with correct consonants:
    # selectivity ≈ C (each correct consonant matches 1/C of dictionary,
    # so real hits are C× more than random)
    predicted_sel_c5 = 5.0
    predicted_sel_actual = float(n_cons_classes) if n_cons_classes > 0 else 0.0

    print(f"     Observed mean selectivity: {overall_mean_sel:.2f}×")
    print(f"     C5×V4 prediction:          {predicted_sel_c5:.1f}×")
    print(f"     C{n_cons_classes} prediction:            {predicted_sel_actual:.1f}×")

    sel_matches_c5 = abs(overall_mean_sel - predicted_sel_c5) < 1.5
    sel_matches_actual = abs(overall_mean_sel - predicted_sel_actual) < 1.5

    # ── 7. Hypothesis verdict ──
    print("  7. Hypothesis assessment …")
    evidence_points = 0
    if within_exceeds_between:
        evidence_points += 1
        print("     ✓ Within-group correlation > between-group")
    else:
        print("     ✗ Within-group correlation ≤ between-group")

    if within_lower:
        evidence_points += 1
        print("     ✓ Within-class selectivity variance < between-class")
    else:
        print("     ✗ Within-class selectivity variance ≥ between-class")

    if sel_matches_c5:
        evidence_points += 1
        print(f"     ✓ Mean selectivity ≈ C5 prediction ({overall_mean_sel:.2f} ≈ 5.0)")
    else:
        print(f"     ✗ Mean selectivity ≠ C5 prediction ({overall_mean_sel:.2f} ≠ 5.0)")

    hypothesis_confirmed = evidence_points >= 2

    # ── 8. Save ──
    elapsed = time.time() - t0
    output = {
        'group_results': group_results,
        'n_within_pairs': len(within_corrs),
        'n_between_pairs': len(between_corrs),
        'mean_within_correlation': round(mean_within, 4),
        'mean_between_correlation': round(mean_between, 4),
        'within_exceeds_between': within_exceeds_between,
        'mean_within_class_variance': round(mean_within_var, 4),
        'between_class_variance': round(between_var, 4),
        'f_statistic': round(f_stat, 4),
        'within_variance_lower': within_lower,
        'observed_mean_selectivity': round(overall_mean_sel, 3),
        'c5v4_predicted_selectivity': predicted_sel_c5,
        'actual_n_class_predicted': round(predicted_sel_actual, 1),
        'selectivity_matches_c5': sel_matches_c5,
        'selectivity_matches_actual': sel_matches_actual,
        'evidence_points': evidence_points,
        'hypothesis_confirmed': hypothesis_confirmed,
        'verdict': (
            f"Consonant-correct hypothesis: "
            f"{'CONFIRMED' if hypothesis_confirmed else 'NOT CONFIRMED'} "
            f"({evidence_points}/3 evidence points). "
            f"Within corr={mean_within:.4f} vs between={mean_between:.4f}. "
            f"Mean selectivity={overall_mean_sel:.2f}× (C5 pred=5.0×)."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'cv_correlation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
