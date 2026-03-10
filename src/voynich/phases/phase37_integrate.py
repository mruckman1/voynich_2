"""
Step 37.15 – Phase 37 Integration
====================================
Synthesize all five investigations into a unified assessment.

Dependency chain:
    consonant_grouping.json    (Step 37.1)
    cv_correlation.json        (Step 37.2)
    vowel_confusion.json       (Step 37.3)
    pair_concat.json           (Step 37.4)
    concat_signal.json         (Step 37.5)
    concat_bigrams.json        (Step 37.6)
    joint_target.json          (Step 37.7)
    joint_swap.json            (Step 37.8)
    joint_validate.json        (Step 37.9)
    f57v_eva_analysis.json     (Step 37.10)
    f57v_structure.json        (Step 37.11)
    italian_corpus.json        (Step 37.12)
    italian_10k.json           (Step 37.13)
    italian_signal.json        (Step 37.14)
    signal_10k.json            (Step 36.2)
    bigrams_10k.json           (Step 36.3)
        → phase37_integrate.json   (this step)
"""

import json
import os
import time
from typing import Any, Dict, List

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase37_integrate() -> None:
    """Step 37.15: Phase 37 Integration."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.15: Phase 37 Integration")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all investigation results ──
    print("\n  1. Loading all investigation results …")
    cg = _safe_load(os.path.join(rd, 'consonant_grouping.json'))
    cv = _safe_load(os.path.join(rd, 'cv_correlation.json'))
    vc = _safe_load(os.path.join(rd, 'vowel_confusion.json'))
    pc = _safe_load(os.path.join(rd, 'pair_concat.json'))
    cs = _safe_load(os.path.join(rd, 'concat_signal.json'))
    cb = _safe_load(os.path.join(rd, 'concat_bigrams.json'))
    jt = _safe_load(os.path.join(rd, 'joint_target.json'))
    js = _safe_load(os.path.join(rd, 'joint_swap.json'))
    jv = _safe_load(os.path.join(rd, 'joint_validate.json'))
    fe = _safe_load(os.path.join(rd, 'f57v_eva_analysis.json'))
    fs = _safe_load(os.path.join(rd, 'f57v_structure.json'))
    ic = _safe_load(os.path.join(rd, 'italian_corpus.json'))
    i10 = _safe_load(os.path.join(rd, 'italian_10k.json'))
    isg = _safe_load(os.path.join(rd, 'italian_signal.json'))
    sig = _safe_load(os.path.join(rd, 'signal_10k.json'))
    big = _safe_load(os.path.join(rd, 'bigrams_10k.json'))

    # ── 2. Investigation 1 verdict: Consonant-Vowel Decomposition ──
    print("  2. Investigation 1: Consonant-Vowel Decomposition …")
    inv1_hypothesis = cv.get('hypothesis_confirmed', False)
    inv1_n_classes = cg.get('n_consonant_only_classes', 0)
    inv1_mean_sel = cg.get('overall_mean_selectivity', 0.0)
    inv1_cc_bigrams = vc.get('n_content_content_bigrams', 0)
    inv1_delta_content = vc.get('delta_content', 0)
    inv1_generalizes = vc.get('generalizes', False)

    if inv1_hypothesis and inv1_cc_bigrams > 0 and inv1_generalizes:
        inv1_verdict = 'CONFIRMED_AND_CORRECTABLE'
    elif inv1_hypothesis:
        inv1_verdict = 'CONFIRMED_NOT_CORRECTABLE'
    else:
        inv1_verdict = 'NOT_CONFIRMED'

    print(f"     Verdict: {inv1_verdict}")
    print(f"     {inv1_n_classes} consonant classes, mean sel={inv1_mean_sel:.2f}×")
    print(f"     CC bigrams from vowel correction: {inv1_cc_bigrams}")

    # ── 3. Investigation 2 verdict: Signal Pair Concatenation ──
    print("  3. Investigation 2: Signal Pair Concatenation …")
    inv2_significant = pc.get('significant', False)
    inv2_match_rate = pc.get('pair_match_rate', 0.0)
    inv2_content = pc.get('n_content_matches', 0)
    inv2_z_improved = cb.get('z_improved', False)
    inv2_merged_z = cb.get('merged_bigram_z', 0.0)
    inv2_cc = cb.get('n_content_content', 0)

    if inv2_significant and inv2_z_improved:
        inv2_verdict = 'EVA_TOKENS_ARE_SYLLABLES'
    elif inv2_significant:
        inv2_verdict = 'SIGNIFICANT_NOT_IMPROVED'
    else:
        inv2_verdict = 'EVA_TOKENS_ARE_WORDS'

    print(f"     Verdict: {inv2_verdict}")
    print(f"     Match rate: {inv2_match_rate:.3%}, content={inv2_content}")
    print(f"     Merged z: {inv2_merged_z:.2f}, CC={inv2_cc}")

    # ── 4. Investigation 3 verdict: Multi-Triple Joint Swap ──
    print("  4. Investigation 3: Multi-Triple Joint Swap …")
    inv3_n_swaps = jv.get('n_swaps', js.get('n_swaps', 0))
    inv3_cc = jv.get('corrected_cc_bigrams', 0)
    inv3_z = jv.get('corrected_bigram_z', 0.0)
    inv3_generalizes = jv.get('generalizes', False)
    inv3_vowel_only = jv.get('vowel_only_changes', 0)
    inv3_consonant = jv.get('consonant_changes', 0)

    if inv3_cc > 0 and inv3_generalizes and inv3_z >= 10:
        inv3_verdict = 'CONTENT_WORDS_FOUND'
    elif inv3_cc > 0:
        inv3_verdict = 'CONTENT_FOUND_NOT_GENERALIZED'
    else:
        inv3_verdict = 'NO_CONTENT_WORDS'

    print(f"     Verdict: {inv3_verdict}")
    print(f"     {inv3_n_swaps} swaps, CC={inv3_cc}, z={inv3_z:.2f}")
    print(f"     Vowel-only: {inv3_vowel_only}, consonant: {inv3_consonant}")

    # ── 5. Investigation 4 verdict: f57v ──
    print("  5. Investigation 4: f57v Deep Examination …")
    inv4_compression = fe.get('compression_ratio', 0.0)
    inv4_diversity = fe.get('diversity_verdict', '')
    inv4_content_type = fs.get('content_type', '')
    inv4_recipe_match = fs.get('matches_recipe_template', False)

    if inv4_diversity == 'TABLE_COLLAPSE':
        inv4_verdict = 'TABLE_COLLAPSE'
    elif inv4_diversity == 'GENUINE_REPETITION' and inv4_recipe_match:
        inv4_verdict = 'PHARMACEUTICAL_RECIPES'
    elif inv4_diversity == 'GENUINE_REPETITION':
        inv4_verdict = 'GENUINE_REPETITION'
    else:
        inv4_verdict = 'MODERATE'

    print(f"     Verdict: {inv4_verdict}")
    print(f"     Compression: {inv4_compression:.3f}, content: {inv4_content_type}")

    # ── 6. Investigation 5 verdict: Italian ──
    print("  6. Investigation 5: Northern Italian 10K …")
    inv5_italian_sel = i10.get('italian_selectivity', 0.0)
    inv5_latin_sel = i10.get('latin_selectivity', 0.0)
    inv5_prefers = i10.get('italian_prefers', False)
    inv5_macaronic = isg.get('is_macaronic', False)
    inv5_merged_z = isg.get('merged_bigram_z', 0.0)
    inv5_n_italian_only = isg.get('n_italian_only_signals', 0)

    if inv5_prefers:
        inv5_verdict = 'ITALIAN_PREFERRED'
    elif inv5_macaronic:
        inv5_verdict = 'MACARONIC'
    else:
        inv5_verdict = 'LATIN_PREFERRED'

    print(f"     Verdict: {inv5_verdict}")
    print(f"     Italian sel={inv5_italian_sel:.3f}× vs Latin sel={inv5_latin_sel:.3f}×")
    print(f"     {inv5_n_italian_only} Italian-only signals, merged z={inv5_merged_z:.2f}")

    # ── 7. Cross-investigation interactions ──
    print("  7. Cross-investigation interactions …")
    interactions = []

    if inv1_verdict.startswith('CONFIRMED') and inv2_verdict == 'EVA_TOKENS_ARE_SYLLABLES':
        interactions.append(
            "Vowel correction + concatenation: correct vowels first, "
            "then concatenate corrected signal pairs for better content words"
        )

    if inv3_verdict.startswith('CONTENT') and inv5_verdict in ('ITALIAN_PREFERRED', 'MACARONIC'):
        interactions.append(
            "Joint swap + Italian: content words from swap may be Italian, "
            "not Latin. Re-evaluate against Italian dictionary."
        )

    if inv4_verdict == 'PHARMACEUTICAL_RECIPES' and inv5_verdict in ('ITALIAN_PREFERRED', 'MACARONIC'):
        interactions.append(
            "f57v recipes + Italian: f57v encodes Italian pharmaceutical "
            "recipes, closest analogue is Anonimo Veneziano"
        )

    for interaction in interactions:
        print(f"     • {interaction}")

    if not interactions:
        print("     No significant cross-investigation interactions")

    # ── 8. Best overall configuration ──
    print("  8. Best overall configuration …")
    # Determine which investigation produced the best improvement
    baseline_z = big.get('bigram_z', 0.0)
    baseline_signal = sig.get('signal_rate', 0.0)
    baseline_cc = 0  # Phase 36 had 0 content-content

    best_config = 'BASELINE'
    best_z = baseline_z
    best_cc = 0

    if inv3_z >= 10 and inv3_cc > best_cc:
        best_config = 'JOINT_SWAP'
        best_z = inv3_z
        best_cc = inv3_cc

    if inv2_merged_z > best_z:
        best_config = 'CONCATENATION'
        best_z = inv2_merged_z
        best_cc = inv2_cc

    if inv1_cc_bigrams > best_cc and inv1_generalizes:
        best_config = 'VOWEL_CORRECTION'
        best_cc = inv1_cc_bigrams

    print(f"     Best config: {best_config}")
    print(f"     Best z: {best_z:.2f}")
    print(f"     Best CC bigrams: {best_cc}")

    # ── 9. Final progression table ──
    print("\n  9. Final progression table:")
    print("     " + "-" * 70)
    print(f"     {'Phase':<10s} {'Dict':<8s} {'SIGNAL':<10s} "
          f"{'Bigram z':<10s} {'CC bigrams':<12s} {'Advance'}")
    print("     " + "-" * 70)
    print(f"     {'29':<10s} {'131K':<8s} {'16.5%':<10s} "
          f"{'6.14':<10s} {'0':<12s} {'Bigram discovery'}")
    print(f"     {'36':<10s} {'10K':<8s} {baseline_signal:<10.4f} "
          f"{baseline_z:<10.2f} {'0':<12s} {'10K pipeline'}")
    print(f"     {'37':<10s} {'—':<8s} {'—':<10s} "
          f"{best_z:<10.2f} {best_cc:<12d} {'Multi-vector'}")
    print("     " + "-" * 70)

    # ── 10. Save ──
    elapsed = time.time() - t0

    output = {
        'investigation_verdicts': {
            'inv1_consonant_vowel': inv1_verdict,
            'inv2_concatenation': inv2_verdict,
            'inv3_joint_swap': inv3_verdict,
            'inv4_f57v': inv4_verdict,
            'inv5_italian': inv5_verdict,
        },
        'investigation_details': {
            'inv1': {
                'hypothesis': inv1_hypothesis,
                'n_classes': inv1_n_classes,
                'mean_selectivity': inv1_mean_sel,
                'cc_bigrams': inv1_cc_bigrams,
                'generalizes': inv1_generalizes,
            },
            'inv2': {
                'significant': inv2_significant,
                'match_rate': inv2_match_rate,
                'content_matches': inv2_content,
                'merged_z': inv2_merged_z,
                'cc': inv2_cc,
            },
            'inv3': {
                'n_swaps': inv3_n_swaps,
                'cc_bigrams': inv3_cc,
                'bigram_z': inv3_z,
                'generalizes': inv3_generalizes,
                'vowel_only': inv3_vowel_only,
                'consonant_changed': inv3_consonant,
            },
            'inv4': {
                'compression': inv4_compression,
                'diversity': inv4_diversity,
                'content_type': inv4_content_type,
                'recipe_match': inv4_recipe_match,
            },
            'inv5': {
                'italian_selectivity': inv5_italian_sel,
                'latin_selectivity': inv5_latin_sel,
                'prefers_italian': inv5_prefers,
                'macaronic': inv5_macaronic,
                'italian_only_signals': inv5_n_italian_only,
            },
        },
        'cross_interactions': interactions,
        'best_config': best_config,
        'best_bigram_z': round(best_z, 2),
        'best_cc_bigrams': best_cc,
        'progression': {
            'phase29': {'dict': '131K', 'signal': 0.165, 'bigram_z': 6.14, 'cc': 0},
            'phase36': {'dict': '10K', 'signal': round(baseline_signal, 4),
                        'bigram_z': round(baseline_z, 2), 'cc': 0},
            'phase37': {'dict': '—', 'signal': None,
                        'bigram_z': round(best_z, 2), 'cc': best_cc},
        },
        'verdict': (
            f"Phase 37: {best_config}. "
            f"Inv1={inv1_verdict}, Inv2={inv2_verdict}, "
            f"Inv3={inv3_verdict}, Inv4={inv4_verdict}, Inv5={inv5_verdict}. "
            f"Best z={best_z:.2f}, CC={best_cc}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'phase37_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
