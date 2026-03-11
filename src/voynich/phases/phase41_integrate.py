"""
Step 41.16 – Phase 41 Integration
===================================
Combine all four tracks into the definitive Phase 41 assessment.

Dependency chain:
    venetian_confirmed.json           (Track A — Step 41.4)
    venetian_validated.json           (Track A — Step 41.2)
    venetian_signal_proper.json       (Track A — Step 41.3)
    complete_lexicon.json             (Track B — Step 41.8)
    f57v_complete_reading.json        (Track C — Step 41.12)
    botanical_predictions_v2.json     (Track D — Step 41.15)
        → phase41_integrate.json  (this step)
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
# Validation tests
# ---------------------------------------------------------------------------

def _run_validations(
    track_a: Dict,
    track_b: Dict,
    track_c: Dict,
    track_d: Dict,
) -> List[Dict]:
    """Run the Phase 41 validation battery."""
    validations = []

    # V1: Venetian selectivity (corrected) >= 1.5×
    ven_valid = track_a.get('venetian_validated', {})
    selectivity = ven_valid.get('venetian_selectivity', 0.0)
    v1 = selectivity >= 1.5
    validations.append({
        'id': 'V1',
        'test': 'Venetian selectivity (corrected) >= 1.5×',
        'value': round(selectivity, 4),
        'threshold': 1.5,
        'passed': v1,
    })

    # V2: Corrected bigram z >= 3.0
    z_total = ven_valid.get('z_total', 0.0)
    v2 = z_total >= 3.0
    validations.append({
        'id': 'V2',
        'test': 'Corrected bigram z >= 3.0',
        'value': round(z_total, 4),
        'threshold': 3.0,
        'passed': v2,
    })

    # V3: Lexicon glossed count >= 50/73
    lexicon = track_b.get('complete_lexicon', {})
    n_glossed = lexicon.get('n_glossed', 0)
    n_total_words = lexicon.get('n_total', 73)
    v3 = n_glossed >= 50
    validations.append({
        'id': 'V3',
        'test': 'Lexicon glossed >= 50/73',
        'value': n_glossed,
        'threshold': 50,
        'passed': v3,
    })

    # V4: f57v coverage >= 55%
    f57v = track_c.get('f57v_complete_reading', {})
    cov_data = f57v.get('coverage', {})
    coverage = cov_data.get('glossed_pct', 0.0) if isinstance(cov_data, dict) else 0.0
    v4 = coverage >= 0.55
    validations.append({
        'id': 'V4',
        'test': 'f57v coverage >= 55%',
        'value': round(coverage, 4),
        'threshold': 0.55,
        'passed': v4,
    })

    # V5: Formula pattern detected >= 1
    formula = track_c.get('formula_segmentation', {})
    n_formulas = formula.get('n_formula_zones', 0)
    v5 = n_formulas >= 1
    validations.append({
        'id': 'V5',
        'test': 'Formula pattern detected >= 1',
        'value': n_formulas,
        'threshold': 1,
        'passed': v5,
    })

    # V6: Botanical soft match >= 1
    bot = track_d.get('botanical_predictions_v2', {})
    n_matches = bot.get('n_with_matches', 0)
    v6 = n_matches >= 1
    validations.append({
        'id': 'V6',
        'test': 'Botanical soft match >= 1',
        'value': n_matches,
        'threshold': 1,
        'passed': v6,
    })

    # V7: Full-corpus dict_hit no regression (>= 43%)
    null_ven = track_a.get('null_venetian_decode', {})
    real_hit = null_ven.get('real_venetian_hit_rate', 0.0)
    v7 = real_hit >= 0.30  # Venetian hit rate, not merged
    validations.append({
        'id': 'V7',
        'test': 'Venetian dict-hit >= 30% (no regression)',
        'value': round(real_hit, 4),
        'threshold': 0.30,
        'passed': v7,
    })

    return validations


def _assign_verdict(validations: List[Dict]) -> str:
    """Assign overall Phase 41 verdict."""
    v_map = {v['id']: v['passed'] for v in validations}

    if v_map.get('V1') and v_map.get('V2'):
        return 'VENETIAN_VALIDATED'
    elif v_map.get('V1') or v_map.get('V2'):
        return 'VENETIAN_PARTIAL'
    else:
        return 'VENETIAN_REFUTED'


def _build_progression_table() -> List[Dict]:
    """Build the definitive progression table."""
    return [
        {'phase': 14, 'advance': 'Stroke-triple model', 'metric': '19.4% dict-hit'},
        {'phase': 19, 'advance': 'Tachygraphic ID', 'metric': 'cosine 0.820'},
        {'phase': 29, 'advance': 'Sequential structure', 'metric': 'z=6.14'},
        {'phase': 36, 'advance': 'Dict right-sizing', 'metric': 'z=12.66'},
        {'phase': 37, 'advance': 'Macaronic discovery', 'metric': 'Italian 5.45×'},
        {'phase': 38, 'advance': 'Content-content bigrams', 'metric': 'z=14.37'},
        {'phase': 39, 'advance': 'Venetian confirmation', 'metric': 'sel 4.58×'},
        {'phase': 40, 'advance': 'Folio reading (unvalidated)', 'metric': 'z=319 (bug)'},
        {'phase': 41, 'advance': 'Validated Venetian', 'metric': 'TBD'},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase41_integrate() -> None:
    """Step 41.16: Phase 41 Integration — combine all tracks."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.16: Phase 41 Integration")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all track results ──
    print("\n  1. Loading track results …")

    # Track A
    null_ven = _safe_load(os.path.join(rd, 'null_venetian_decode.json'))
    ven_valid = _safe_load(os.path.join(rd, 'venetian_validated.json'))
    ven_sig = _safe_load(os.path.join(rd, 'venetian_signal_proper.json'))
    ven_conf = _safe_load(os.path.join(rd, 'venetian_confirmed.json'))

    # Track B
    ungloss = _safe_load(os.path.join(rd, 'unglossed_analysis.json'))
    ven_dict = _safe_load(os.path.join(rd, 'venetian_dictionary_search.json'))
    context_dis = _safe_load(os.path.join(rd, 'context_disambiguation.json'))
    lexicon = _safe_load(os.path.join(rd, 'complete_lexicon.json'))

    # Track C
    formula = _safe_load(os.path.join(rd, 'formula_segmentation.json'))
    inter_form = _safe_load(os.path.join(rd, 'inter_formula_tokens.json'))
    ingred = _safe_load(os.path.join(rd, 'ingredient_search.json'))
    f57v = _safe_load(os.path.join(rd, 'f57v_complete_reading.json'))

    # Track D
    bot_fix = _safe_load(os.path.join(rd, 'botanical_data_fix.json'))
    drosera = _safe_load(os.path.join(rd, 'drosera_propagation.json'))
    bot_pred = _safe_load(os.path.join(rd, 'botanical_predictions_v2.json'))

    track_a = {
        'null_venetian_decode': null_ven,
        'venetian_validated': ven_valid,
        'venetian_signal_proper': ven_sig,
        'venetian_confirmed': ven_conf,
    }
    track_b = {
        'unglossed_analysis': ungloss,
        'venetian_dictionary_search': ven_dict,
        'context_disambiguation': context_dis,
        'complete_lexicon': lexicon,
    }
    track_c = {
        'formula_segmentation': formula,
        'inter_formula_tokens': inter_form,
        'ingredient_search': ingred,
        'f57v_complete_reading': f57v,
    }
    track_d = {
        'botanical_data_fix': bot_fix,
        'drosera_propagation': drosera,
        'botanical_predictions_v2': bot_pred,
    }

    loaded = sum(1 for d in [null_ven, ven_valid, ven_sig, ven_conf,
                              ungloss, ven_dict, context_dis, lexicon,
                              formula, inter_form, ingred, f57v,
                              bot_fix, drosera, bot_pred] if d)
    print(f"    Loaded {loaded}/15 upstream results")

    # ── 2. Track A summary ──
    print("\n  2. Track A: Venetian Null Validation")
    selectivity = ven_valid.get('venetian_selectivity', 0.0)
    z_total = ven_valid.get('z_total', 0.0)
    z_exact = ven_valid.get('z_exact', 0.0)
    original_z = ven_valid.get('original_z', 319.76)
    n_confirmed_vocab = ven_conf.get('n_confirmed', 0)

    print(f"    Original (buggy) z: {original_z:.2f}")
    print(f"    Validated z (total): {z_total:.2f}")
    print(f"    Validated z (exact): {z_exact:.2f}")
    print(f"    Venetian selectivity: {selectivity:.2f}×")
    print(f"    Confirmed vocabulary: {n_confirmed_vocab} words")

    track_a_verdict = ven_valid.get('verdict', 'UNKNOWN')
    print(f"    Track A verdict: {track_a_verdict}")

    track_a_summary = {
        'original_z': round(original_z, 4),
        'validated_z_total': round(z_total, 4),
        'validated_z_exact': round(z_exact, 4),
        'selectivity': round(selectivity, 4),
        'n_confirmed_vocab': n_confirmed_vocab,
        'verdict': track_a_verdict,
    }

    # ── 3. Track B summary ──
    print("\n  3. Track B: Lexicon Completion")
    n_glossed = lexicon.get('n_glossed', 0)
    n_total = lexicon.get('n_total', 73)
    n_new = ven_dict.get('n_new_glosses', 0)
    print(f"    Glossed: {n_glossed}/{n_total}")
    print(f"    New glosses from dict search: {n_new}")

    track_b_summary = {
        'n_glossed': n_glossed,
        'n_total': n_total,
        'n_new_glosses': n_new,
        'gloss_rate': round(n_glossed / n_total, 4) if n_total > 0 else 0.0,
    }

    # ── 4. Track C summary ──
    print("\n  4. Track C: f57v Inter-Formula Content")
    n_formula_zones = formula.get('n_formula_zones', 0)
    n_content_zones = formula.get('n_content_zones', 0)
    f57v_cov = f57v.get('coverage', {})
    coverage = f57v_cov.get('glossed_pct', 0.0) if isinstance(f57v_cov, dict) else 0.0
    best_passage = f57v.get('best_passage', {})
    best_passage_len = best_passage.get('length', 0) if isinstance(best_passage, dict) else 0
    n_ingredients = ingred.get('total_exact_matches', 0)
    print(f"    Formula zones: {n_formula_zones}")
    print(f"    Content zones: {n_content_zones}")
    print(f"    Coverage: {coverage:.4f}")
    print(f"    Best passage length: {best_passage_len}")
    print(f"    Ingredients found: {n_ingredients}")

    track_c_summary = {
        'n_formula_zones': n_formula_zones,
        'n_content_zones': n_content_zones,
        'coverage': round(coverage, 4),
        'best_passage_length': best_passage_len,
        'n_ingredients': n_ingredients,
    }

    # ── 5. Track D summary ──
    print("\n  5. Track D: Botanical Pipeline Fix")
    n_plants = bot_fix.get('n_with_italian_names', 0)
    n_predictions = drosera.get('n_plants_analyzed', 0)
    n_matches = bot_pred.get('n_with_matches', 0)
    n_wrong = bot_pred.get('n_cross_folio_conflicting', 0)
    print(f"    Plants identified: {n_plants}")
    print(f"    Predictions generated: {n_predictions}")
    print(f"    Matches on correct folio: {n_matches}")
    print(f"    Matches on wrong folios: {n_wrong}")

    track_d_summary = {
        'n_plants': n_plants,
        'n_predictions': n_predictions,
        'n_matches_correct': n_matches,
        'n_matches_wrong': n_wrong,
    }

    # ── 6. Validations ──
    print("\n  6. Running validation battery …")
    validations = _run_validations(track_a, track_b, track_c, track_d)
    n_passed = sum(1 for v in validations if v['passed'])

    for v in validations:
        status = 'PASS' if v['passed'] else 'FAIL'
        print(f"    {v['id']}: {status}  {v['test']}  "
              f"(value={v['value']}, threshold={v['threshold']})")

    print(f"\n    Validations passed: {n_passed}/{len(validations)}")

    # ── 7. Verdict ──
    verdict = _assign_verdict(validations)
    print(f"\n  7. VERDICT: {verdict}")

    if verdict == 'VENETIAN_VALIDATED':
        print("    The Venetian identification is confirmed with proper null "
              "validation.")
    elif verdict == 'VENETIAN_PARTIAL':
        print("    Partial Venetian signal — one criterion met but not both.")
    else:
        print("    The Venetian hypothesis is not supported after proper "
              "null correction.")

    # ── 8. Progression table ──
    print("\n  8. Progression table:")
    progression = _build_progression_table()
    # Update Phase 41 entry with actual metric
    if progression:
        progression[-1]['metric'] = f"z={z_total:.2f} (validated)"

    for row in progression:
        print(f"    Phase {row['phase']:2d}: {row['advance']:30s}  "
              f"{row['metric']}")

    # ── 9. Save ──
    elapsed = time.time() - t0

    output = {
        'track_a': track_a_summary,
        'track_b': track_b_summary,
        'track_c': track_c_summary,
        'track_d': track_d_summary,
        'validations': validations,
        'n_validations_passed': n_passed,
        'n_validations_total': len(validations),
        'verdict': verdict,
        'progression': progression,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'phase41_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
