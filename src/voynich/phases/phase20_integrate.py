"""
Phase 20.8 – Phase 20 Integration and Verdict
==============================================
Compile all Phase 20 results into a final assessment with verdict,
tachygraphic table, and progression tracking.

Dependency chain:
    tachy_anchors.json + tachy_families.json + tachy_grid_solve.json
    + tachy_decode.json + tachy_readability.json + tachy_phrases.json
    + tachy_validate.json
        → phase20_integrate.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

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
class Phase20IntegrateResult:
    # Tachygraphic table
    tachygraphic_table: Dict[str, str]
    n_chars_mapped: int
    n_modifier_chars: int
    n_unmapped: int
    # Performance
    dict_hit: float
    expanded_dict_hit: float
    cross_entropy: float
    null_selectivity: float
    # Validation
    n_tests_passed: int
    n_tests_total: int
    strong_pass: bool
    # Phrases
    n_phrases: int
    phrase_selectivity: float
    n_botanical_matches: int
    # Readability
    bigram_plausibility: float
    n_domains_with_hits: int
    # Progression
    progression: Dict[str, Dict]
    # Verdict
    outcome: str            # DECODED, PARTIALLY_DECODED, STRUCTURALLY_CONFIRMED, FAILED
    verdict: str
    next_action: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_json(rd: str, fname: str) -> Dict:
    path = os.path.join(rd, fname)
    if not os.path.exists(path):
        print(f"    [WARN] {fname} not found")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase20_integrate() -> None:
    """Step 20.8: Final integration and verdict."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 20.8: Integration and Verdict")
    print("=" * 70)

    rd = _results_dir()

    # Load all results
    print("\n  1. Loading all Phase 20 results …")
    anchors = _load_json(rd, 'tachy_anchors.json')
    families = _load_json(rd, 'tachy_families.json')
    grid = _load_json(rd, 'tachy_grid_solve.json')
    decode = _load_json(rd, 'tachy_decode.json')
    readability = _load_json(rd, 'tachy_readability.json')
    phrases = _load_json(rd, 'tachy_phrases.json')
    validate = _load_json(rd, 'tachy_validate.json')

    # Extract key metrics
    best_assignment = grid.get('best_assignment', {})
    n_chars_mapped = len(best_assignment)
    n_modifier = len(_load_json(rd, 'modifier_integrate.json').get('modifier_chars', []))
    n_unmapped = 44 - n_chars_mapped - n_modifier

    dict_hit = decode.get('dict_hit_rate', 0.0)
    expanded_dict_hit = decode.get('expanded_dict_hit_rate', 0.0)
    cross_entropy = grid.get('best_cross_entropy', 99.0)
    null_selectivity = grid.get('null_selectivity', 0.0)

    n_tests_passed = validate.get('n_passed', 0)
    n_tests_total = validate.get('n_total', 12)
    strong_pass = validate.get('strong_pass', False)

    n_phrases = phrases.get('n_phrases_detected', 0)
    phrase_selectivity = phrases.get('phrase_selectivity', 0.0)
    n_botanical = phrases.get('n_botanical_matches', 0)

    bigram_plausibility = readability.get('bigram_plausibility', 0.0)
    n_domains = readability.get('n_domains_with_hits', 0)

    # ─── 2. Print tachygraphic table ───
    print("\n  2. Final tachygraphic table:")
    tachy_table_display = grid.get('tachygraphic_table', {})
    for ch in sorted(tachy_table_display.keys()):
        info = tachy_table_display[ch]
        syl = info.get('syllable', '?')
        gc = info.get('glyph_class', '?')
        anchored = '*' if info.get('is_anchored') else ' '
        print(f"      {ch:8s} → {syl:4s}  {anchored} [{gc}]")

    # ─── 3. Progression table ───
    print("\n  3. Progression:")
    progression = {
        'phase11': {'dict_hit': 0.111, 'selectivity': 1.92, 'note': 'CV phonotactic model'},
        'phase13': {'dict_hit': 0.1143, 'selectivity': 1.86, 'note': 'Context rules'},
        'phase14': {'dict_hit': 0.194, 'selectivity': 3.00, 'note': 'Sub-cell features'},
        'phase15': {'dict_hit': 0.354, 'selectivity': 2.55, 'note': 'Dict expansion'},
        'phase16': {'dict_hit': 0.516, 'selectivity': 3.38, 'note': 'Modifier detection'},
        'phase20': {
            'dict_hit': expanded_dict_hit,
            'selectivity': null_selectivity,
            'note': 'Tachygraphic table',
            'n_phrases': n_phrases,
            'bigram_plausibility': bigram_plausibility,
            'n_botanical': n_botanical,
        },
    }

    for phase, info in progression.items():
        dh = info['dict_hit']
        sel = info.get('selectivity', 0)
        note = info.get('note', '')
        marker = '  ←' if phase == 'phase20' else ''
        print(f"      {phase:10s}: dict_hit={dh:.1%}  "
              f"selectivity={sel:.2f}×  {note}{marker}")

    # ─── 4. Determine outcome ───
    print("\n  4. Determining outcome …")

    gate_pass = validate.get('gate_passed', False)

    if strong_pass and n_phrases >= 20 and n_botanical >= 3 and bigram_plausibility > 0.15:
        outcome = 'DECODED'
        next_action = (
            "Publish. Produce full decoded text with confidence annotations. "
            "Cross-check against all known botanical identifications. "
            "Submit for peer review."
        )
    elif gate_pass and n_phrases >= 5 and bigram_plausibility > 0.08:
        outcome = 'PARTIALLY_DECODED'
        next_action = (
            "Identify highest-error characters from section domain coherence. "
            "Use confirmed phrases to bootstrap new character-level anchors. "
            "Re-run Steps 20.1-20.3 with expanded anchor set."
        )
    elif gate_pass:
        outcome = 'STRUCTURALLY_CONFIRMED'
        next_action = (
            "Tachygraphic mechanism confirmed but table imprecise. "
            "Return to Step 20.3 with relaxed constraints (6 vowel variants "
            "or 6 consonant classes). Test alternative family sub-segmentations."
        )
    else:
        outcome = 'FAILED'
        next_action = (
            "Tachygraphic hypothesis not supported by decoding evidence. "
            "Statistical matches in Phases 19.2/19.5/19.6 may be coincidental. "
            "Redirect to alternative hypotheses."
        )

    verdict = (
        f"{outcome}: V-battery {n_tests_passed}/{n_tests_total}, "
        f"dict_hit={expanded_dict_hit:.1%}, "
        f"phrases={n_phrases}, "
        f"bigram={bigram_plausibility:.3f}, "
        f"botanical={n_botanical}."
    )

    print(f"\n      Outcome: {outcome}")
    print(f"      Verdict: {verdict}")
    print(f"      Next: {next_action}")

    # ─── 5. Save ───
    result = Phase20IntegrateResult(
        tachygraphic_table=best_assignment,
        n_chars_mapped=n_chars_mapped,
        n_modifier_chars=n_modifier,
        n_unmapped=n_unmapped,
        dict_hit=dict_hit,
        expanded_dict_hit=expanded_dict_hit,
        cross_entropy=cross_entropy,
        null_selectivity=null_selectivity,
        n_tests_passed=n_tests_passed,
        n_tests_total=n_tests_total,
        strong_pass=strong_pass,
        n_phrases=n_phrases,
        phrase_selectivity=phrase_selectivity,
        n_botanical_matches=n_botanical,
        bigram_plausibility=bigram_plausibility,
        n_domains_with_hits=n_domains,
        progression=progression,
        outcome=outcome,
        verdict=verdict,
        next_action=next_action,
        runtime_seconds=time.time() - t0,
    )

    out_path = os.path.join(rd, 'phase20_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
