"""
Phase 28.9 – Phase 28 Verdict
================================
Final integration and verdict for the Ventris-style crib propagation
and signal isolation analysis.

Dependency chain:
    crib_extraction.json      (Step 28.1)
    crib_consistency.json     (Step 28.2)
    family_propagation.json   (Step 28.3)
    signal_isolation.json     (Step 28.4)
    crib_localization.json    (Step 28.5)
    ventris_table.json        (Step 28.6)
    ventris_decode.json       (Step 28.7)
    ventris_readability.json  (Step 28.8)
        → phase28_verdict.json  (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Phase28VerdictResult:
    # Crib summary
    n_tier1_cribs: int
    n_tier2_cribs: int
    n_tier3_cribs: int
    n_cross_source_independent: int
    # Consistency
    family_consistency_rate: float
    cross_source_consistency_rate: float
    null_z_score: float
    # Table
    n_tier1_triples: int
    n_tier2_triples: int
    n_tier3_triples: int
    n_corrections_applied: int
    # Signal
    n_genuine_signals: int
    n_signal_tokens: int
    signal_token_rate: float
    mean_signal_selectivity: float
    # Localization
    domain_accuracy: float
    best_passage_folio: str
    # Decode
    final_dict_hit: float
    phase16_baseline: float
    improvement_vs_phase16: float
    # Validation
    n_validations_passed: int
    n_validations_total: int
    # Verdict
    verdict: str
    verdict_description: str
    key_findings: List[str]
    next_steps: List[str]
    progression: Dict
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase28_verdict() -> None:
    """Step 28.9: Final Phase 28 verdict."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 28.9: Phase 28 Verdict")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all upstream results ──
    print("\n  1. Loading upstream results …")

    def _load(name):
        path = os.path.join(rd, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        print(f"     [WARN] {name} not found")
        return {}

    crib = _load('crib_extraction.json')
    consist = _load('crib_consistency.json')
    prop = _load('family_propagation.json')
    signal = _load('signal_isolation.json')
    local = _load('crib_localization.json')
    vtab = _load('ventris_table.json')
    decode = _load('ventris_decode.json')
    readability = _load('ventris_readability.json')

    # ── 2. Extract metrics ──
    n_tier1_cribs = crib.get('n_tier1', 0)
    n_tier2_cribs = crib.get('n_tier2', 0)
    n_tier3_cribs = crib.get('n_tier3', 0)
    n_cross_source = consist.get('n_cross_source_tests', 0)

    family_rate = consist.get('family_consistency_rate', 0.0)
    cross_rate = consist.get('cross_source_rate', 0.0)
    null_z = consist.get('null_z_score', 0.0)

    n_t1_triples = vtab.get('n_tier1', 0)
    n_t2_triples = vtab.get('n_tier2', 0)
    n_t3_triples = vtab.get('n_tier3', 0)
    n_corrections = vtab.get('n_changed_vs_phase16', 0)

    n_genuine = signal.get('n_genuine_signals', 0)
    n_signal_tokens = signal.get('n_signal_tokens', 0)
    signal_rate = signal.get('signal_token_rate', 0.0)
    mean_sel = signal.get('mean_selectivity', 0.0)

    domain_acc = local.get('domain_accuracy', 0.0)
    best_folio = local.get('best_passage_folio', decode.get('best_passage_folio', ''))

    final_hit = decode.get('corpus_dict_hit', 0.0)
    phase16 = decode.get('phase16_baseline', 0.0)
    improvement = decode.get('improvement_vs_phase16', 0.0)

    n_val_passed = readability.get('n_passed', 0)
    n_val_total = readability.get('n_total', 0)

    # ── 3. Determine verdict ──
    print("\n  2. Determining verdict …")

    if n_corrections >= 1 and improvement > 0 and n_val_passed >= 6:
        verdict = 'TABLE_CORRECTED'
        verdict_desc = (
            f"Ventris analysis corrected {n_corrections} triple(s) and "
            f"improved dict_hit by {improvement:+.4f}. "
            f"{n_val_passed}/{n_val_total} validations passed."
        )
    elif n_genuine >= 10 and domain_acc >= 0.50:
        verdict = 'SIGNAL_CONFIRMED'
        verdict_desc = (
            f"{n_genuine} crib words carry genuine signal (σ>2.0). "
            f"{n_signal_tokens} SIGNAL tokens ({signal_rate:.1%}). "
            f"Domain accuracy {domain_acc:.0%}. "
            f"Confidence-tiered table: {n_t1_triples}+{n_t2_triples}+{n_t3_triples}."
        )
    elif n_t1_triples + n_t2_triples >= 10 and n_val_passed >= 6:
        verdict = 'TABLE_TIERED'
        verdict_desc = (
            f"Table unchanged but confidence-tiered: "
            f"{n_t1_triples} Tier-1, {n_t2_triples} Tier-2, "
            f"{n_t3_triples} Tier-3. "
            f"{n_val_passed}/{n_val_total} validations passed."
        )
    elif n_genuine < 5 and family_rate < 0.80:
        verdict = 'CRIBS_COLLISIONS'
        verdict_desc = (
            f"Only {n_genuine} genuine signal words. "
            f"Family consistency {family_rate:.0%}. "
            f"Confirmed words are likely dictionary collisions."
        )
    else:
        verdict = 'TABLE_TIERED'
        verdict_desc = (
            f"Confidence tiers assigned: {n_t1_triples}+{n_t2_triples}+"
            f"{n_t3_triples}. {n_genuine} genuine signals. "
            f"{n_val_passed}/{n_val_total} validations."
        )

    # ── 4. Key findings ──
    key_findings = []
    key_findings.append(
        f"Cross-source crib pool: {n_cross_source} testable triples "
        f"(de, bene only). All 18 Phase 14 hits are trivially consistent "
        f"(same table)."
    )
    key_findings.append(
        f"Family typological consistency: {family_rate:.0%} "
        f"({consist.get('n_family_consistent', 0)}/{consist.get('n_triples_total', 0)} triples). "
        f"Null z-score: {null_z:.1f}."
    )
    key_findings.append(
        f"Signal isolation: {n_genuine}/{signal.get('n_words_tested', 0)} crib words "
        f"carry genuine signal (σ>2.0). {n_signal_tokens} SIGNAL tokens "
        f"({signal_rate:.1%} of corpus)."
    )
    if n_corrections > 0:
        key_findings.append(
            f"Table corrected at {n_corrections} triple(s) via family propagation."
        )
    else:
        key_findings.append(
            "No corrections applied — Phase 15 assignment unchanged."
        )
    key_findings.append(
        f"Final dict_hit: {final_hit:.4f} (Δ={improvement:+.4f} vs Phase 16)."
    )

    # ── 5. Next steps ──
    next_steps = []
    if n_genuine >= 5:
        next_steps.append(
            "Use the SIGNAL token set to weight future table optimization — "
            "focus on tokens that discriminate real from null."
        )
    if n_corrections == 0:
        next_steps.append(
            "The Phase 15/16 table is already near-optimal within the "
            "CV syllabary model. Improvements require expanding beyond CV "
            "(CVC, CCV) or changing the segmentation model."
        )
    next_steps.append(
        "The confidence tiers can prioritize which triples to investigate "
        "with external evidence (paleographic, illustration-based)."
    )

    # ── 6. Progression ──
    progression = {
        'phase11': 0.111,
        'phase14': 0.194,
        'phase15': 0.354,
        'phase16': round(phase16, 4),
        'phase28': round(final_hit, 4),
    }

    # ── 7. Print verdict ──
    print(f"\n  Verdict: {verdict}")
    print(f"  {verdict_desc}")
    print("\n  Key findings:")
    for i, f in enumerate(key_findings, 1):
        print(f"    {i}. {f}")
    print("\n  Next steps:")
    for i, s in enumerate(next_steps, 1):
        print(f"    {i}. {s}")
    print(f"\n  Progression: {progression}")

    # ── 8. Save ──
    result = Phase28VerdictResult(
        n_tier1_cribs=n_tier1_cribs,
        n_tier2_cribs=n_tier2_cribs,
        n_tier3_cribs=n_tier3_cribs,
        n_cross_source_independent=n_cross_source,
        family_consistency_rate=round(family_rate, 4),
        cross_source_consistency_rate=round(cross_rate, 4),
        null_z_score=round(null_z, 2),
        n_tier1_triples=n_t1_triples,
        n_tier2_triples=n_t2_triples,
        n_tier3_triples=n_t3_triples,
        n_corrections_applied=n_corrections,
        n_genuine_signals=n_genuine,
        n_signal_tokens=n_signal_tokens,
        signal_token_rate=round(signal_rate, 4),
        mean_signal_selectivity=round(mean_sel, 2),
        domain_accuracy=round(domain_acc, 4),
        best_passage_folio=best_folio,
        final_dict_hit=round(final_hit, 4),
        phase16_baseline=round(phase16, 4),
        improvement_vs_phase16=round(improvement, 4),
        n_validations_passed=n_val_passed,
        n_validations_total=n_val_total,
        verdict=verdict,
        verdict_description=verdict_desc,
        key_findings=key_findings,
        next_steps=next_steps,
        progression=progression,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase28_verdict.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
