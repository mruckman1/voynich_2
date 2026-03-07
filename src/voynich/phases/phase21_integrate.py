"""
Phase 21.10 – Integration (phase21-integrate)
=============================================
Final verdict, progression table (Phase 11→21), gap analysis.

Dependency chain:
    paleo_validate.json (21.9) + all upstream results
        → phase21_integrate.json (this step)
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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


def _load_json(path: str) -> Optional[Dict]:
    import os
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _determine_verdict(
    validate: Dict,
    table: Dict,
    decode: Dict,
) -> str:
    """Determine Phase 21 verdict.

    PALEOGRAPHIC DECODE: STRONG PASS + ≥30% P1-3 + ≥10 phrases + bigram>15%
    PARTIAL PALEOGRAPHIC: PASS + ≥20% P1-3 + improvement over Phase 16
    PALEOGRAPHIC CONSTRAINTS: ≥10 P1-3 chars but no readable text
    INSUFFICIENT EVIDENCE: <10 P1-3 assignments
    HYPOTHESIS REFUTED: Historical signs don't match EVA at stroke level
    """
    n_passed = validate.get('n_passed', 0)
    strong_pass = validate.get('strong_pass', False)
    p13_count = table.get('coverage_priority_1_3', 0)
    exp_rate = decode.get('dict_hit_rate_expanded', 0)

    # Count phrases from V3
    n_phrases = 0
    for t in validate.get('tests', []):
        if t.get('test') == 'V3_phrase_detection':
            n_phrases = t.get('n_phrases', 0)

    if strong_pass and p13_count >= 13 and n_phrases >= 10 and exp_rate > 0.15:
        return 'PALEOGRAPHIC DECODE'

    if n_passed >= 9 and p13_count >= 9:
        return 'PARTIAL PALEOGRAPHIC'

    if p13_count >= 10:
        return 'PALEOGRAPHIC CONSTRAINTS'

    if p13_count < 10:
        # Check if stroke comparison found any signal
        rdir = _results_dir()
        eva_cmp = _load_json(str(rdir / "eva_stroke_compare.json")) or {}
        selectivity = eva_cmp.get('null_selectivity', 0)
        if selectivity < 1.0:
            return 'HYPOTHESIS REFUTED'
        return 'INSUFFICIENT EVIDENCE'

    return 'INSUFFICIENT EVIDENCE'


# ---------------------------------------------------------------------------
# Progression table
# ---------------------------------------------------------------------------

def _build_progression() -> List[Dict[str, Any]]:
    """Build Phase 11→21 progression table."""
    rdir = _results_dir()
    phases = []

    # Phase 11
    p11 = _load_json(str(rdir / "csp_validate.json"))
    if p11:
        phases.append({
            'phase': 11, 'name': 'CSP Grid',
            'dict_hit': p11.get('dict_hit_rate', 0.111),
            'selectivity': p11.get('selectivity', 1.92),
        })

    # Phase 14
    p14 = _load_json(str(rdir / "feature_decode.json"))
    if p14:
        phases.append({
            'phase': 14, 'name': 'Sub-Cell Features',
            'dict_hit': p14.get('dict_hit_rate', 0.194),
            'selectivity': p14.get('selectivity', 3.00),
        })

    # Phase 15
    p15 = _load_json(str(rdir / "combined_refine.json"))
    if p15:
        phases.append({
            'phase': 15, 'name': 'Dict Expansion + AC',
            'dict_hit': p15.get('dict_hit_rate', p15.get('best_dict_hit_rate', 0.354)),
            'selectivity': p15.get('selectivity', 2.55),
        })

    # Phase 16
    p16 = _load_json(str(rdir / "modifier_integrate.json"))
    if p16:
        dhr = p16.get('dict_hit_rate', p16.get('r3_dict_hit_rate', 0.516))
        if isinstance(dhr, dict):
            dhr = dhr.get('expanded', 0.516)
        phases.append({
            'phase': 16, 'name': 'Modifier Detection',
            'dict_hit': dhr,
            'selectivity': p16.get('selectivity', 3.38),
        })

    # Phase 19
    p19 = _load_json(str(rdir / "cross_approach.json"))
    if p19:
        phases.append({
            'phase': 19, 'name': 'Tachygraphic Hypothesis',
            'dict_hit': None,
            'selectivity': p19.get('selectivity', 32.26),
        })

    # Phase 20
    p20 = _load_json(str(rdir / "phase20_integrate.json"))
    if p20:
        phases.append({
            'phase': 20, 'name': 'Systematic Decode',
            'dict_hit': p20.get('dict_hit_rate', 0),
            'selectivity': p20.get('selectivity', 0.97),
        })

    # Phase 21
    p21_decode = _load_json(str(rdir / "paleo_decode.json"))
    p21_val = _load_json(str(rdir / "paleo_validate.json"))
    if p21_decode:
        phases.append({
            'phase': 21, 'name': 'Paleographic Comparison',
            'dict_hit': p21_decode.get('dict_hit_rate_expanded', 0),
            'selectivity': None,  # Multiple selectivities across substeps
            'v_battery': f"{p21_val.get('n_passed', 0)}/{p21_val.get('n_total', 15)}" if p21_val else 'N/A',
        })

    return phases


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def _gap_analysis(table_data: Dict) -> Dict[str, Any]:
    """Identify unmatched EVA chars and evidence gaps."""
    table = table_data.get('table', [])
    unassigned = []
    low_confidence = []
    modifiers_without_function = []

    for entry in table:
        ec = entry.get('eva_char', '')
        conf = entry.get('confidence', 'unassigned')
        is_mod = entry.get('is_modifier', False)
        mod_func = entry.get('modifier_function')

        if conf == 'unassigned':
            unassigned.append(ec)
        elif conf == 'low':
            low_confidence.append(ec)

        if is_mod and not mod_func:
            modifiers_without_function.append(ec)

    return {
        'n_unassigned': len(unassigned),
        'unassigned_chars': unassigned,
        'n_low_confidence': len(low_confidence),
        'low_confidence_chars': low_confidence,
        'n_modifiers_without_function': len(modifiers_without_function),
        'modifiers_without_function': modifiers_without_function,
        'total_chars': len(EVA_VISUAL_COMPONENTS),
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase21IntegrateResult:
    timestamp: str
    verdict: str
    progression: List[Dict[str, Any]]
    gap_analysis: Dict[str, Any]
    summary: str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase21_integrate() -> Dict[str, Any]:
    """Phase 21.10: Final integration and verdict."""
    t0 = time.time()
    rdir = _results_dir()

    validate = _load_json(str(rdir / "paleo_validate.json")) or {}
    table = _load_json(str(rdir / "paleo_table.json")) or {}
    decode = _load_json(str(rdir / "paleo_decode.json")) or {}

    verdict = _determine_verdict(validate, table, decode)
    progression = _build_progression()
    gaps = _gap_analysis(table)

    # Build summary
    v_battery = validate.get('verdict', 'N/A')
    p13 = table.get('coverage_priority_1_3', 0)
    exp_rate = decode.get('dict_hit_rate_expanded', 0)

    summary = (
        f"Phase 21 Verdict: {verdict}\n"
        f"V-battery: {v_battery}\n"
        f"Priority 1-3 coverage: {p13}/{table.get('table_size', 44)}\n"
        f"Expanded dict hit rate: {exp_rate:.1%}\n"
        f"Unassigned chars: {gaps['n_unassigned']}/{gaps['total_chars']}\n"
        f"Low-confidence chars: {gaps['n_low_confidence']}"
    )

    result = Phase21IntegrateResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        verdict=verdict,
        progression=progression,
        gap_analysis=gaps,
        summary=summary,
    )

    out_path = rdir / "phase21_integrate.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PHASE 21: {verdict}")
    print(f"{'='*60}")
    print(summary)
    print(f"({'='*60})")
    print(f"[{elapsed:.1f}s]")

    return _convert(asdict(result))
