"""
Step 43.15 – Phase 43 Integration
====================================
Combine verdicts from all three Phase 43 approaches into a unified
assessment.

Dependency chain:
    results/inversion_validate.json     (Step 43.5: Approach 1 verdict)
    results/structural_reading.json     (Step 43.9: Approach 4 verdict)
    results/hmm_signal.json             (Step 43.14: Approach 5 verdict)
    results/symmetric_recompute.json    (Phase 42.2: baseline z)
    results/ground_truth.json           (Phase 42.5: ground truth)
        → phase43_integrate.json        (this step)
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# JSON helpers
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase43IntegrateResult:
    # Per-approach summaries
    approach1_verdict: str
    approach1_summary: Dict
    approach4_verdict: str
    approach4_summary: Dict
    approach5_verdict: str
    approach5_summary: Dict
    # Cross-approach
    n_approaches_positive: int
    concordance: str
    # Progression
    progression_table: List[Dict]
    # Overall
    phase43_verdict: str
    phase43_rationale: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase43_integrate() -> None:
    """Step 43.15: integrate all Phase 43 approach verdicts."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.15: Phase 43 Integration")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load approach results ──
    print("\n  1. Loading approach results …")
    inv_val = _safe_load(os.path.join(rd, 'inversion_validate.json'))
    struct_read = _safe_load(os.path.join(rd, 'structural_reading.json'))
    hmm_sig = _safe_load(os.path.join(rd, 'hmm_signal.json'))
    sym = _safe_load(os.path.join(rd, 'symmetric_recompute.json'))
    gt = _safe_load(os.path.join(rd, 'ground_truth.json'))

    baseline_z = sym.get('best_surviving_z_exact', 3.80)

    # ── 2. Approach 1: Re-Encoding Inversion ──
    print("\n  2. Approach 1: Re-Encoding Inversion …")
    a1_verdict = inv_val.get('approach1_verdict', 'UNKNOWN')
    a1_summary = {
        'dict_hit': inv_val.get('real_dict_hit', 0.0),
        'phase15_dict_hit': inv_val.get('phase15_dict_hit', 0.0),
        'delta_dict_hit': inv_val.get('delta_dict_hit', 0.0),
        'bigram_z': inv_val.get('bigram_z_score', 0.0),
        'signal_rate': inv_val.get('signal_rate', 0.0),
        'n_passed': inv_val.get('n_passed', 0),
        'n_total': inv_val.get('n_total', 0),
        'n_bedrock_preserved': inv_val.get('n_bedrock_preserved', 0),
    }
    print(f"     Verdict: {a1_verdict}")
    print(f"     Dict-hit: {a1_summary['dict_hit']:.1%} (Δ={a1_summary['delta_dict_hit']:+.1%})")
    print(f"     Bigram z: {a1_summary['bigram_z']:.2f}")
    print(f"     Validations: {a1_summary['n_passed']}/{a1_summary['n_total']}")

    # ── 3. Approach 4: Signal Word Structural Probing ──
    print("\n  3. Approach 4: Structural Probing …")
    a4_verdict = struct_read.get('approach4_verdict', 'UNKNOWN')
    a4_summary = {
        'structural_coherence': struct_read.get('structural_coherence', 0.0),
        'n_folios_annotated': struct_read.get('n_folios_annotated', 0),
        'estimated_recipes': struct_read.get('estimated_recipe_count', 0),
        'best_organization': struct_read.get('best_organization', 'unknown'),
        'best_folio': struct_read.get('best_folio', ''),
        'best_folio_signal_rate': struct_read.get('best_folio_signal_rate', 0.0),
        'n_recurring_patterns': struct_read.get('n_recurring_patterns', 0),
        'type_counts': struct_read.get('type_counts', {}),
    }
    print(f"     Verdict: {a4_verdict}")
    print(f"     Coherence: {a4_summary['structural_coherence']:.4f}")
    print(f"     Folios annotated: {a4_summary['n_folios_annotated']}")
    print(f"     Best organization: {a4_summary['best_organization']}")

    # ── 4. Approach 5: HMM Decoding ──
    print("\n  4. Approach 5: HMM Decoding …")
    a5_verdict = hmm_sig.get('approach5_verdict', 'UNKNOWN')
    a5_summary = {
        'signal_rate': hmm_sig.get('signal_rate', 0.0),
        'bigram_z': hmm_sig.get('bigram_z_score', 0.0),
        'n_passed': hmm_sig.get('n_passed', 0),
        'n_total': hmm_sig.get('n_total', 0),
        'overlap_with_10k': hmm_sig.get('overlap_with_10k_signal', 0.0),
        'held_out_ratio': hmm_sig.get('held_out_ratio', 0.0),
        'n_bedrock_preserved': hmm_sig.get('n_bedrock_preserved', 0),
    }
    print(f"     Verdict: {a5_verdict}")
    print(f"     Signal rate: {a5_summary['signal_rate']:.1%}")
    print(f"     Bigram z: {a5_summary['bigram_z']:.2f}")
    print(f"     Validations: {a5_summary['n_passed']}/{a5_summary['n_total']}")

    # ── 5. Cross-approach analysis ──
    print("\n  5. Cross-approach analysis …")
    positive_verdicts = {
        'IMPROVEMENT', 'STRUCTURAL_SIGNAL', 'WEAK_SIGNAL',
    }
    approach_verdicts = {
        'approach1': a1_verdict,
        'approach4': a4_verdict,
        'approach5': a5_verdict,
    }
    n_positive = sum(
        1 for v in approach_verdicts.values()
        if v in positive_verdicts
    )

    # Concordance: do the approaches agree?
    if n_positive >= 3:
        concordance = 'FULL_AGREEMENT'
    elif n_positive == 2:
        concordance = 'MAJORITY_POSITIVE'
    elif n_positive == 1:
        concordance = 'MINORITY_POSITIVE'
    else:
        concordance = 'NO_POSITIVE'

    print(f"     Positive approaches: {n_positive}/3")
    print(f"     Concordance: {concordance}")
    for name, v in approach_verdicts.items():
        marker = "+" if v in positive_verdicts else "-"
        print(f"       [{marker}] {name}: {v}")

    # ── 6. Progression table ──
    print("\n  6. Building progression table …")
    progression = [
        {'phase': 'Phase 11', 'dict_hit': 0.111, 'selectivity': 1.92,
         'description': 'CV phonotactic model'},
        {'phase': 'Phase 14', 'dict_hit': 0.194, 'selectivity': 3.00,
         'description': 'Sub-cell feature model'},
        {'phase': 'Phase 15', 'dict_hit': 0.354, 'selectivity': 2.55,
         'description': 'Feature model refinement + dict expansion'},
        {'phase': 'Phase 16', 'dict_hit': 0.436, 'selectivity': 3.38,
         'description': 'Modifier detection (full corpus)'},
        {'phase': 'Phase 42', 'baseline_z': baseline_z,
         'description': 'Symmetric bigram z baseline'},
    ]

    # Add Phase 43 approaches
    if a1_summary.get('dict_hit', 0) > 0:
        progression.append({
            'phase': 'Phase 43 A1',
            'dict_hit': a1_summary['dict_hit'],
            'bigram_z': a1_summary['bigram_z'],
            'verdict': a1_verdict,
            'description': 'Re-encoding inversion',
        })
    progression.append({
        'phase': 'Phase 43 A4',
        'coherence': a4_summary['structural_coherence'],
        'verdict': a4_verdict,
        'description': 'Signal word structural probing',
    })
    if a5_summary.get('bigram_z', 0) != 0:
        progression.append({
            'phase': 'Phase 43 A5',
            'bigram_z': a5_summary['bigram_z'],
            'signal_rate': a5_summary['signal_rate'],
            'verdict': a5_verdict,
            'description': 'Context-dependent HMM decoding',
        })

    for row in progression:
        desc = row.get('description', '')
        if 'dict_hit' in row:
            print(f"     {row['phase']:16s} dict_hit={row['dict_hit']:.1%}  {desc}")
        elif 'baseline_z' in row:
            print(f"     {row['phase']:16s} z={row['baseline_z']:.2f}  {desc}")
        else:
            v = row.get('verdict', '')
            print(f"     {row['phase']:16s} {v:12s}  {desc}")

    # ── 7. Overall verdict ──
    print("\n  7. Overall Phase 43 verdict …")

    # Check if any approach produced an actual improvement over baseline
    a1_improved = a1_verdict == 'IMPROVEMENT'
    a5_improved = a5_verdict == 'IMPROVEMENT'
    a4_has_signal = a4_verdict in ('STRUCTURAL_SIGNAL', 'WEAK_SIGNAL')

    if a1_improved and a5_improved:
        verdict = 'BREAKTHROUGH'
        rationale = (
            'Both re-encoding inversion and HMM decoding independently '
            'improve upon the Phase 42 baseline. Multiple orthogonal '
            'approaches converge on enhanced signal.'
        )
    elif a1_improved or a5_improved:
        improved = 'Approach 1' if a1_improved else 'Approach 5'
        verdict = 'IMPROVEMENT'
        rationale = (
            f'{improved} improves upon the Phase 42 baseline. '
            f'Structural probing {"confirms" if a4_has_signal else "does not confirm"} '
            f'content organization.'
        )
    elif n_positive >= 2:
        verdict = 'LATERAL_WITH_INSIGHT'
        rationale = (
            f'{n_positive}/3 approaches show positive signal but no clear '
            f'improvement over Phase 42 baseline (z={baseline_z:.2f}). '
            f'Structural probing provides manuscript organization map.'
        )
    elif n_positive == 1:
        verdict = 'LATERAL'
        rationale = (
            'One approach shows positive signal but the others do not '
            'improve upon the baseline. The Phase 16 table remains the '
            'best decoding.'
        )
    else:
        verdict = 'NO_IMPROVEMENT'
        rationale = (
            'None of the three orthogonal approaches produced an improvement '
            'over the Phase 42 baseline. The Phase 16 table at 43.6% dict-hit '
            'remains the local optimum.'
        )

    print(f"     Verdict: {verdict}")
    print(f"     Rationale: {rationale}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    result = Phase43IntegrateResult(
        approach1_verdict=a1_verdict,
        approach1_summary=a1_summary,
        approach4_verdict=a4_verdict,
        approach4_summary=a4_summary,
        approach5_verdict=a5_verdict,
        approach5_summary=a5_summary,
        n_approaches_positive=n_positive,
        concordance=concordance,
        progression_table=progression,
        phase43_verdict=verdict,
        phase43_rationale=rationale,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'phase43_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path} ({elapsed:.1f}s)")
