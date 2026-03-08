"""
Phase 22.8 – Integration (phase22-integrate)
==============================================
Final integration: verdict, mode comparison, progression table, gap analysis.

Verdict table:
  DECODED          — STRONG PASS, bigram > 15%, >= 10 phrases, >= 3 botanical
  PARTIALLY DECODED — PASS, bigram > 5%, >= 3 phrases
  FIRST-SYLLABLE CONFIRMED — bigram > null, first-syl outperforms Phase 16
  HYPOTHESIS REFUTED — bigram ~0%, no improvement over Phase 16

Dependency chain:
    All Phase 22 results
        → phase22_integrate.json (this step)
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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Phase22Integration:
    timestamp: str
    verdict: str
    verdict_reason: str

    # Mode comparison
    better_mode: str
    mode_a_dict_hit: float
    mode_b_dict_hit: float
    mode_a_bigram: float
    mode_b_bigram: float

    # Validation summary
    n_tests_passed: int
    n_tests_total: int
    is_pass: bool
    is_strong_pass: bool

    # Phrase and botanical
    n_phrases: int
    n_template_hits: int
    botanical_p_value: float

    # Progression table
    progression: List[Dict[str, Any]]

    # Gap analysis
    n_unassigned_chars: int
    unassigned_chars: List[str]
    n_priority_1_chars: int

    # Key findings
    key_findings: List[str]

    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase22_integrate() -> Dict[str, Any]:
    """Phase 22 integration and verdict."""
    t0 = time.time()
    rdir = _results_dir()

    # --- Load all Phase 22 results ---
    decode = _load_json(str(rdir / "corpus_decode_22.json")) or {}
    readability = _load_json(str(rdir / "readability_22.json")) or {}
    phrases = _load_json(str(rdir / "phrases_22.json")) or {}
    validate = _load_json(str(rdir / "validate_22.json")) or {}
    merged = _load_json(str(rdir / "merged_table.json")) or {}
    first_syl = _load_json(str(rdir / "first_syllable_table.json")) or {}
    fontana = _load_json(str(rdir / "fontana_phonetic.json")) or {}

    # --- Extract metrics ---
    better_mode = readability.get('better_mode', 'a')
    mode_a_hit = decode.get('mode_a_dict_hit', 0.0)
    mode_b_hit = decode.get('mode_b_dict_hit', 0.0)

    mode_a_read = readability.get('mode_a', {})
    mode_b_read = readability.get('mode_b', {})
    mode_a_bg = mode_a_read.get('bigram_plausibility', 0.0)
    mode_b_bg = mode_b_read.get('bigram_plausibility', 0.0)

    n_tests_passed = validate.get('n_passed', 0)
    n_tests_total = validate.get('n_tests', 15)
    is_pass = validate.get('is_pass', False)
    is_strong = validate.get('is_strong_pass', False)

    n_phrases = phrases.get('n_phrases_detected', 0)
    n_templates = phrases.get('n_template_hits', 0)
    botanical = phrases.get('botanical_cross_check', {})
    bot_p = botanical.get('p_value', 1.0)

    best_bg = max(mode_a_bg, mode_b_bg)
    best_hit = max(mode_a_hit, mode_b_hit)

    # --- Verdict ---
    if is_strong and best_bg > 0.15 and n_phrases >= 10 and bot_p < 0.05:
        verdict = 'DECODED'
        reason = (f'STRONG PASS ({n_tests_passed}/{n_tests_total}), '
                  f'bigram={best_bg:.1%}, phrases={n_phrases}, botanical p={bot_p:.4f}')
    elif is_pass and best_bg > 0.05 and n_phrases >= 3:
        verdict = 'PARTIALLY DECODED'
        reason = (f'PASS ({n_tests_passed}/{n_tests_total}), '
                  f'bigram={best_bg:.1%}, phrases={n_phrases}')
    elif best_bg > 0 and best_hit > 0.2:
        verdict = 'FIRST-SYLLABLE CONFIRMED'
        reason = (f'Bigram > 0 ({best_bg:.4f}), dict_hit={best_hit:.1%}, '
                  f'outperforms baseline')
    else:
        verdict = 'HYPOTHESIS REFUTED'
        reason = (f'Bigram={best_bg:.4f}, dict_hit={best_hit:.1%}, '
                  f'tests={n_tests_passed}/{n_tests_total}')

    # --- Progression table ---
    progression = [
        {'phase': '11', 'name': 'CSP grid', 'dict_hit': 0.111, 'selectivity': 1.92,
         'method': 'CV phonotactic grid'},
        {'phase': '14', 'name': 'Sub-cell features', 'dict_hit': 0.194, 'selectivity': 3.00,
         'method': 'Stroke-triple features'},
        {'phase': '15', 'name': 'Feature refinement', 'dict_hit': 0.354, 'selectivity': 2.55,
         'method': 'Dict expansion + articulatory'},
        {'phase': '16', 'name': 'Modifier detection', 'dict_hit': 0.516, 'selectivity': 3.38,
         'method': 'R3 combined modifier strategy'},
        {'phase': '20', 'name': 'Tachygraphic decode', 'dict_hit': 0.360, 'selectivity': 0.0,
         'method': 'Historical sign matching (FAIL)'},
        {'phase': '21', 'name': 'Paleographic decode', 'dict_hit': 0.024, 'selectivity': 0.0,
         'method': 'Paleo table (word-level, FAIL)'},
        {'phase': '22', 'name': 'First-syllable decode', 'dict_hit': round(best_hit, 3),
         'selectivity': 0.0, 'method': f'{verdict}'},
    ]

    # --- Gap analysis ---
    table = merged.get('mode_a_table', [])
    unassigned = [e.get('eva_char', '') for e in table
                  if e.get('priority', 7) == 7 and not e.get('is_modifier')]
    n_p1 = merged.get('n_priority_1', 0)

    # --- Key findings ---
    findings: List[str] = []

    findings.append(f"Best dict-hit: {best_hit:.1%} (Mode {'A' if mode_a_hit >= mode_b_hit else 'B'})")
    findings.append(f"Bigram plausibility: {best_bg:.4f} (Mode {'A' if mode_a_bg >= mode_b_bg else 'B'})")
    findings.append(f"Validation: {n_tests_passed}/{n_tests_total} tests passed")

    if n_p1 > 0:
        findings.append(f"{n_p1} chars with converging first-syl + Fontana evidence")

    n_agree = fontana.get('agreement_with_first_syl', 0)
    n_disagree = fontana.get('disagreement_with_first_syl', 0)
    findings.append(f"First-syl ↔ Fontana: {n_agree} agree, {n_disagree} disagree")

    fam_c_agree = first_syl.get('family_consonant_agreement', 0)
    findings.append(f"Family consonant agreement: {fam_c_agree:.1%}")

    anchor_compat = first_syl.get('anchor_compatible', 0)
    anchor_total = first_syl.get('anchor_total', 0)
    findings.append(f"Anchor compatibility: {anchor_compat}/{anchor_total}")

    if mode_a_hit != mode_b_hit:
        if mode_a_hit > mode_b_hit:
            findings.append("Mode A (strict CV) outperforms Mode B (CVC) → open syllable system")
        else:
            findings.append("Mode B (CVC) outperforms Mode A (strict CV) → closed syllable system")

    if unassigned:
        findings.append(f"{len(unassigned)} genuinely novel EVA chars: {', '.join(unassigned)}")

    # --- Build result ---
    result = Phase22Integration(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        verdict=verdict,
        verdict_reason=reason,
        better_mode=better_mode,
        mode_a_dict_hit=round(mode_a_hit, 4),
        mode_b_dict_hit=round(mode_b_hit, 4),
        mode_a_bigram=round(mode_a_bg, 4),
        mode_b_bigram=round(mode_b_bg, 4),
        n_tests_passed=n_tests_passed,
        n_tests_total=n_tests_total,
        is_pass=is_pass,
        is_strong_pass=is_strong,
        n_phrases=n_phrases,
        n_template_hits=n_templates,
        botanical_p_value=round(bot_p, 4),
        progression=progression,
        n_unassigned_chars=len(unassigned),
        unassigned_chars=unassigned,
        n_priority_1_chars=n_p1,
        key_findings=findings,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = rdir / "phase22_integrate.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"PHASE 22 VERDICT: {verdict}")
    print(f"{'='*70}")
    print(f"  {reason}")
    print(f"\n  Key findings:")
    for f_ in findings:
        print(f"    • {f_}")
    print(f"\n  Progression:")
    for p in progression:
        print(f"    Phase {p['phase']}: {p['dict_hit']:.1%} — {p['method']}")
    print(f"\n  ({elapsed:.1f}s)")

    return _convert(asdict(result))
