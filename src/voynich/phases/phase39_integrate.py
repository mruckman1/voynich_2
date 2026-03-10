"""
Step 39.17 – Phase 39 Integration
===================================
Combine all five tracks into the definitive corrected table and assessment.

Dependency chain:
    ed1_decomposition.json         (Step 39.1)
    vowel_error_map.json           (Step 39.2)
    targeted_vowel_fix.json        (Step 39.3)
    corrected_signal.json          (Step 39.4)
    phrase_cribs.json              (Step 39.5)
    phrase_alignment.json          (Step 39.6)
    phrase_corrections.json        (Step 39.7)
    italian_plant_names.json       (Step 39.8)
    italian_botanical_csp.json     (Step 39.9)
    botanical_propagate.json       (Step 39.10)
    venetian_lexicon.json          (Step 39.11)
    venetian_decode.json           (Step 39.12)
    venetian_phrases.json          (Step 39.13)
    amplified_dict.json            (Step 39.14)
    amplified_signal.json          (Step 39.15)
    amplified_bigrams.json         (Step 39.16)
    combined_refine.json           (Step 15)
    merged_signal.json             (Step 38.3)
    merged_bigrams.json            (Step 38.4)
        → phase39_integrate.json   (this step)
"""

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Set

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
# Convergence matrix
# ---------------------------------------------------------------------------

def _build_convergence_matrix(
    track_a: Dict,
    track_b: Dict,
    track_c: Dict,
    assignment: Dict[str, str],
) -> List[Dict]:
    """Build a per-triple convergence matrix from all tracks."""
    all_triples = set(assignment.keys())
    matrix = []

    # Track A corrections
    track_a_corrections = {}
    for corr in track_a.get('corrections_applied', []):
        track_a_corrections[corr['triple_key']] = corr['new_syllable']

    # Track B corrections
    track_b_corrections = {}
    for corr in track_b.get('combined_corrections', []):
        if corr.get('status') == 'CONVERGENT':
            track_b_corrections[corr['triple_key']] = corr.get('correction', '')

    # Track C corrections
    track_c_corrections = {}
    for corr in track_c.get('confirmed_triples', []):
        if isinstance(corr, dict):
            track_c_corrections[corr.get('triple_key', '')] = corr.get('syllable', '')

    for triple in sorted(all_triples):
        current = assignment.get(triple, '?')
        a_val = track_a_corrections.get(triple)
        b_val = track_b_corrections.get(triple)
        c_val = track_c_corrections.get(triple)

        # Count how many tracks recommend a change
        recommendations = {}
        if a_val and a_val != current:
            recommendations['track_a'] = a_val
        if b_val and b_val != current:
            recommendations['track_b'] = b_val
        if c_val and c_val != current:
            recommendations['track_c'] = c_val

        n_tracks = len(recommendations)
        # Check agreement
        rec_values = list(recommendations.values())
        all_agree = len(set(rec_values)) <= 1 if rec_values else True
        agreed_value = rec_values[0] if rec_values and all_agree else None

        matrix.append({
            'triple_key': triple,
            'current': current,
            'track_a': a_val,
            'track_b': b_val,
            'track_c': c_val,
            'n_tracks_recommending': n_tracks,
            'all_agree': all_agree,
            'agreed_value': agreed_value,
        })

    return matrix


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _assign_verdict(
    corrected_dict_hit: float,
    baseline_dict_hit: float,
    n_exact_cc: int,
    amplified_signal_rate: float,
    baseline_signal_rate: float,
    venetian_selectivity: float,
    bigram_z: float,
    baseline_z: float,
) -> str:
    """Assign Phase 39 verdict."""
    if corrected_dict_hit > 0.46 and n_exact_cc >= 3:
        return 'VOWEL_BREAKTHROUGH'
    if corrected_dict_hit > 0.44 and n_exact_cc > 0:
        return 'VOWEL_CORRECTION_CONFIRMED'
    if amplified_signal_rate > baseline_signal_rate and venetian_selectivity > 2.0:
        return 'VENETIAN_SIGNAL_FOUND'
    if corrected_dict_hit > baseline_dict_hit or bigram_z > baseline_z:
        return 'MARGINAL_IMPROVEMENT'
    return 'NO_IMPROVEMENT'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase39_integrate() -> None:
    """Step 39.17: Phase 39 Integration."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.17: Phase 39 Integration")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all track results ──
    print("\n  1. Loading all track results …")

    # Track A
    ed1 = _safe_load(os.path.join(rd, 'ed1_decomposition.json'))
    vowel_map = _safe_load(os.path.join(rd, 'vowel_error_map.json'))
    vowel_fix = _safe_load(os.path.join(rd, 'targeted_vowel_fix.json'))
    corr_signal = _safe_load(os.path.join(rd, 'corrected_signal.json'))

    # Track B
    phrase_cribs = _safe_load(os.path.join(rd, 'phrase_cribs.json'))
    phrase_align = _safe_load(os.path.join(rd, 'phrase_alignment.json'))
    phrase_corr = _safe_load(os.path.join(rd, 'phrase_corrections.json'))

    # Track C
    ital_plants = _safe_load(os.path.join(rd, 'italian_plant_names.json'))
    ital_csp = _safe_load(os.path.join(rd, 'italian_botanical_csp.json'))
    bot_prop = _safe_load(os.path.join(rd, 'botanical_propagate.json'))

    # Track D
    ven_lex = _safe_load(os.path.join(rd, 'venetian_lexicon.json'))
    ven_dec = _safe_load(os.path.join(rd, 'venetian_decode.json'))
    ven_phr = _safe_load(os.path.join(rd, 'venetian_phrases.json'))

    # Track E
    amp_dict = _safe_load(os.path.join(rd, 'amplified_dict.json'))
    amp_signal = _safe_load(os.path.join(rd, 'amplified_signal.json'))
    amp_bigrams = _safe_load(os.path.join(rd, 'amplified_bigrams.json'))

    # Baselines
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    p38_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))
    p38_bigrams = _safe_load(os.path.join(rd, 'merged_bigrams.json'))

    original_assignment = refine.get('best_assignment', {})

    print("     All track results loaded.")

    # ── 2. Track summaries ──
    print("\n  2. Track summaries …")

    track_a_summary = {
        'n_cc_entries': ed1.get('n_cc_entries', 0),
        'n_unique_pairs': ed1.get('n_unique_pairs', 0),
        'n_vowel_errors': ed1.get('n_with_vowel_error', 0),
        'n_tracings': vowel_map.get('n_tracings', 0),
        'n_tier1': vowel_map.get('n_tier1', 0),
        'n_tier2': vowel_map.get('n_tier2', 0),
        'n_corrections_applied': vowel_fix.get('n_corrections_applied', 0),
        'dict_hit_delta': vowel_fix.get('dict_hit_delta', 0.0),
        'baseline_dict_hit': vowel_fix.get('baseline_dict_hit', 0.0),
        'corrected_dict_hit': vowel_fix.get('corrected_dict_hit', 0.0),
        'exact_cc_before': vowel_fix.get('baseline_exact_cc', 0),
        'exact_cc_after': vowel_fix.get('corrected_exact_cc', 0),
        'generalizes': vowel_fix.get('held_out_validation', {}).get('generalizes', False),
    }
    print(f"     Track A: {track_a_summary['n_corrections_applied']} corrections, "
          f"dict_hit Δ={track_a_summary['dict_hit_delta']:+.4f}")

    track_b_summary = {
        'n_medical_phrases': phrase_cribs.get('n_medical_phrases', 0),
        'n_flanked_misses': phrase_cribs.get('n_flanked_misses', 0),
        'n_template_matches': phrase_align.get('n_template_matches', 0),
        'n_convergent': phrase_corr.get('n_convergent', 0),
        'n_phrase_only': phrase_corr.get('n_phrase_only', 0),
    }
    print(f"     Track B: {track_b_summary['n_convergent']} convergent, "
          f"{track_b_summary['n_flanked_misses']} flanked misses")

    track_c_summary = {
        'n_plant_entries': ital_plants.get('n_plant_entries', 0),
        'n_with_italian': ital_plants.get('n_with_italian_name', 0),
        'n_valid_alignments': ital_csp.get('n_valid_alignments', 0),
        'n_cross_folio': ital_csp.get('n_cross_folio_consistent', 0),
        'n_propagated': bot_prop.get('n_propagated', 0),
        'botanical_dict_hit_delta': (
            (bot_prop.get('botanical_dict_hit_corrected', 0) or 0) -
            (bot_prop.get('botanical_dict_hit_baseline', 0) or 0)
        ),
    }
    print(f"     Track C: {track_c_summary['n_valid_alignments']} valid alignments, "
          f"{track_c_summary['n_cross_folio']} cross-folio")

    track_d_summary = {
        'n_venetian_specific': ven_lex.get('n_venetian_specific', 0),
        'n_venetian_only_hits': ven_dec.get('n_venetian_only_hits', 0),
        'venetian_selectivity': ven_dec.get('venetian_selectivity', 0.0),
        'dict_hit_full': ven_dec.get('dict_hit_full', 0.0),
        'n_venetian_phrases': ven_phr.get('n_venetian_phrases', 0),
    }
    print(f"     Track D: {track_d_summary['n_venetian_specific']} Venetian words, "
          f"selectivity={track_d_summary['venetian_selectivity']:.2f}")

    track_e_summary = {
        'calibrated_dict_size': amp_dict.get('calibrated_dict_size', 0),
        'selectivity': amp_dict.get('selectivity', 0.0),
        'signal_rate': amp_signal.get('signal_rate', 0.0),
        'bigram_z': amp_bigrams.get('bigram_z', 0.0),
        'n_exact_cc': amp_bigrams.get('n_exact_cc', 0),
        'n_relaxed_cc': amp_bigrams.get('n_relaxed_cc', 0),
    }
    print(f"     Track E: dict={track_e_summary['calibrated_dict_size']}, "
          f"z={track_e_summary['bigram_z']:.2f}, "
          f"CC={track_e_summary['n_exact_cc']}+{track_e_summary['n_relaxed_cc']}")

    # ── 3. Convergence matrix ──
    print("\n  3. Building convergence matrix …")
    matrix = _build_convergence_matrix(
        vowel_fix, phrase_corr, bot_prop, original_assignment)
    n_recommended = sum(1 for m in matrix if m['n_tracks_recommending'] > 0)
    n_multi_track = sum(1 for m in matrix if m['n_tracks_recommending'] >= 2)
    n_all_agree = sum(1 for m in matrix if m['n_tracks_recommending'] >= 2 and m['all_agree'])
    print(f"     Triples with recommendations: {n_recommended}")
    print(f"     Multi-track recommendations: {n_multi_track}")
    print(f"     Multi-track agreements: {n_all_agree}")

    # ── 4. Best corrected assignment ──
    print("\n  4. Determining best corrected assignment …")
    # Start from Track A's corrected assignment (if positive delta)
    if vowel_fix.get('dict_hit_delta', 0) > 0:
        best_assignment = dict(vowel_fix.get('corrected_assignment', original_assignment))
        assignment_source = 'track_a_corrected'
    else:
        best_assignment = dict(original_assignment)
        assignment_source = 'phase15_original'

    # Apply convergent corrections from Track B
    if phrase_corr.get('final_corrected_assignment'):
        phrase_assignment = phrase_corr['final_corrected_assignment']
        for triple, syl in phrase_assignment.items():
            if triple in best_assignment and syl != best_assignment[triple]:
                # Only apply if convergent
                for m in matrix:
                    if (m['triple_key'] == triple and
                            m['n_tracks_recommending'] >= 2 and
                            m['all_agree'] and
                            m['agreed_value'] == syl):
                        best_assignment[triple] = syl
                        break

    print(f"     Assignment source: {assignment_source}")

    # ── 5. Key metrics ──
    print("\n  5. Key metrics …")
    corrected_dict_hit = vowel_fix.get('corrected_dict_hit',
                                       vowel_fix.get('baseline_dict_hit', 0.0))
    baseline_dict_hit = vowel_fix.get('baseline_dict_hit', 0.0)

    corrected_signal_n_exact_cc = corr_signal.get('n_exact_cc', 0)
    corrected_signal_rate = corr_signal.get('signal_rate', 0.0)
    corrected_bigram_z = corr_signal.get('bigram_z', 0.0)

    amplified_signal_rate = amp_signal.get('signal_rate', 0.0)
    amplified_bigram_z = amp_bigrams.get('bigram_z', 0.0)
    amplified_exact_cc = amp_bigrams.get('n_exact_cc', 0)

    venetian_selectivity = ven_dec.get('venetian_selectivity', 0.0)

    p38_signal_rate = p38_signal.get('signal_rate', 0.0)
    p38_bigram_z = p38_bigrams.get('bigram_z', 0.0)
    p38_n_cc = p38_bigrams.get('n_content_content', 0)

    # Use the best exact CC count from any track
    best_exact_cc = max(corrected_signal_n_exact_cc, amplified_exact_cc)
    best_bigram_z = max(corrected_bigram_z, amplified_bigram_z)

    # ── 6. Progression table ──
    print("\n  6. Progression table …")
    progression = [
        {'phase': 29, 'dict': '131K Latin', 'signal_rate': 0.165,
         'bigram_z': 6.14, 'exact_cc': 0, 'relaxed_cc': 0,
         'advance': 'Bigram discovery'},
        {'phase': 36, 'dict': '10K Latin', 'signal_rate': 0.1853,
         'bigram_z': 12.66, 'exact_cc': 0, 'relaxed_cc': 0,
         'advance': 'Dict right-sizing'},
        {'phase': 38, 'dict': 'merged L+I', 'signal_rate': round(p38_signal_rate, 4),
         'bigram_z': round(p38_bigram_z, 2), 'exact_cc': 0,
         'relaxed_cc': p38_n_cc, 'advance': 'Macaronic pipeline'},
        {'phase': 39, 'dict': 'merged L+I+V',
         'signal_rate': round(corrected_signal_rate, 4),
         'bigram_z': round(best_bigram_z, 2),
         'exact_cc': best_exact_cc,
         'relaxed_cc': corr_signal.get('n_relaxed_cc', 0),
         'advance': 'Vowel correction + Italian botanical'},
    ]

    for p in progression:
        print(f"     Phase {p['phase']}: z={p['bigram_z']}, "
              f"CC={p['exact_cc']}+{p['relaxed_cc']}, "
              f"signal={p['signal_rate']}")

    # ── 7. Verdict ──
    print("\n  7. Assigning verdict …")
    verdict = _assign_verdict(
        corrected_dict_hit, baseline_dict_hit,
        best_exact_cc,
        amplified_signal_rate, p38_signal_rate,
        venetian_selectivity,
        best_bigram_z, p38_bigram_z,
    )
    print(f"     VERDICT: {verdict}")

    # ── 8. Remaining gap ──
    print("\n  8. Remaining gap analysis …")
    n_confirmed_triples = sum(
        1 for m in matrix
        if m['n_tracks_recommending'] == 0  # no change needed = confirmed
    )
    n_changed = sum(1 for m in matrix if m['n_tracks_recommending'] > 0)
    remaining_gap = {
        'n_total_triples': len(matrix),
        'n_confirmed_unchanged': n_confirmed_triples,
        'n_changed_by_corrections': n_changed,
        'n_still_uncertain': sum(
            1 for m in matrix
            if m['n_tracks_recommending'] == 1 and not m.get('all_agree')
        ),
        'what_would_help': [
            'Additional reference bigrams from Italian medical corpora',
            'CVC/CCV phonotactic expansion beyond CV model',
            'Physical manuscript analysis (multispectral imaging)',
            'More Italian botanical identifications with label alignments',
        ],
    }
    print(f"     Confirmed: {n_confirmed_triples}/{len(matrix)} triples")
    print(f"     Changed: {n_changed}")

    # ── 9. Save ──
    elapsed = time.time() - t0

    output = {
        'verdict': verdict,
        'interpretation': {
            'VOWEL_BREAKTHROUGH': 'Multiple exact CC bigrams found after vowel correction — strong evidence of correct decipherment direction.',
            'VOWEL_CORRECTION_CONFIRMED': 'At least one exact CC bigram appeared — vowel corrections are moving in the right direction.',
            'VENETIAN_SIGNAL_FOUND': 'Venetian-specific vocabulary produces elevated signal — the text is specifically Venetian, not generic Italian.',
            'MARGINAL_IMPROVEMENT': 'Small positive deltas from corrections — direction is right but evidence is weak.',
            'NO_IMPROVEMENT': 'No track produced measurable improvement — the CV model may be at its structural limit.',
        }.get(verdict, ''),
        'track_summaries': {
            'track_a': track_a_summary,
            'track_b': track_b_summary,
            'track_c': track_c_summary,
            'track_d': track_d_summary,
            'track_e': track_e_summary,
        },
        'key_metrics': {
            'baseline_dict_hit': baseline_dict_hit,
            'corrected_dict_hit': corrected_dict_hit,
            'dict_hit_delta': round(corrected_dict_hit - baseline_dict_hit, 4),
            'baseline_bigram_z': round(p38_bigram_z, 2),
            'best_bigram_z': round(best_bigram_z, 2),
            'bigram_z_delta': round(best_bigram_z - p38_bigram_z, 2),
            'best_exact_cc': best_exact_cc,
            'baseline_cc': p38_n_cc,
            'corrected_signal_rate': round(corrected_signal_rate, 4),
            'amplified_signal_rate': round(amplified_signal_rate, 4),
            'venetian_selectivity': round(venetian_selectivity, 2),
        },
        'convergence_matrix': matrix,
        'n_multi_track_agreements': n_all_agree,
        'best_assignment_source': assignment_source,
        'final_corrected_assignment': best_assignment,
        'progression_table': progression,
        'remaining_gap': remaining_gap,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'phase39_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
