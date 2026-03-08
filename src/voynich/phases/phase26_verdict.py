"""
Step 26.8 – Phase 26 Verdict
==============================
Final integration and verdict for the Zodiac Known-Plaintext Attack.

Dependency chain:
    phase26_validate.json (Step 26.7)
    zodiac_decode.json (Step 26.6)
    zodiac_table.json (Step 26.5)
    month_crib.json (Step 26.2)
        → phase26_verdict.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Phase26VerdictResult:
    timestamp: str
    # Summary metrics
    corpus_dict_hit: float
    corpus_selectivity: float
    zodiac_dict_hit: float
    phase16_dict_hit: float
    n_tier1_assignments: int
    n_tier2_assignments: int
    best_crib_language: str
    month_crib_selectivity: float
    n_consistent_assignments: int
    n_csp_solutions: int
    n_planet_matches: int
    n_body_matches: int
    n_validations_passed: int
    n_validations_total: int
    # Verdict
    verdict: str
    verdict_description: str
    next_steps: List[str]
    # Progression
    progression: Dict
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase26_verdict() -> None:
    t0 = time.time()
    print("=" * 70)
    print("STEP 26.8: Phase 26 Verdict — Zodiac Known-Plaintext Attack")
    print("=" * 70)

    rd = _results_dir()

    # Load all Phase 26 results
    validate_data = _load_json(os.path.join(rd, 'phase26_validate.json'))
    decode_data = _load_json(os.path.join(rd, 'zodiac_decode.json'))
    table_data = _load_json(os.path.join(rd, 'zodiac_table.json'))
    month_data = _load_json(os.path.join(rd, 'month_crib.json'))
    astro_data = _load_json(os.path.join(rd, 'astro_crib.json'))
    label_data = _load_json(os.path.join(rd, 'label_decode.json'))

    # Gather metrics
    corpus_hit = decode_data.get('corpus_dict_hit', 0) if decode_data else 0
    corpus_sel = decode_data.get('selectivity', 0) if decode_data else 0
    zodiac_hit = decode_data.get('zodiac_dict_hit', 0) if decode_data else 0
    p16_hit = decode_data.get('phase16_dict_hit', 0) if decode_data else 0
    n_t1 = table_data.get('n_tier1', 0) if table_data else 0
    n_t2 = table_data.get('n_tier2', 0) if table_data else 0
    best_lang = month_data.get('best_language', '') if month_data else ''
    month_sel = month_data.get('selectivity_ratio', 0) if month_data else 0
    n_consistent = month_data.get('n_consistent', 0) if month_data else 0
    n_csp = month_data.get('n_csp_solutions', 0) if month_data else 0
    n_exact = month_data.get('n_forward_exact', 0) if month_data else 0
    n_close = month_data.get('n_forward_close', 0) if month_data else 0
    n_planet = astro_data.get('n_planet_matches', 0) if astro_data else 0
    n_body = astro_data.get('n_body_correct', 0) if astro_data else 0
    n_val_passed = validate_data.get('n_passed', 0) if validate_data else 0
    n_val_total = validate_data.get('n_total', 12) if validate_data else 12

    n_month_total = n_exact + n_close

    # -------------------------------------------------------------------
    # Verdict decision tree
    # -------------------------------------------------------------------
    print(f"\n  Summary Metrics:")
    print(f"      Corpus dict_hit:     {corpus_hit:.1%} (Phase16: {p16_hit:.1%})")
    print(f"      Corpus selectivity:  {corpus_sel:.2f}×")
    print(f"      Zodiac dict_hit:     {zodiac_hit:.1%}")
    print(f"      Tier 1 assignments:  {n_t1}")
    print(f"      Tier 2 assignments:  {n_t2}")
    print(f"      Best crib language:  {best_lang}")
    print(f"      Month selectivity:   {month_sel:.2f}×")
    print(f"      Consistent assigns:  {n_consistent}")
    print(f"      CSP solutions:       {n_csp}")
    print(f"      Month matches:       {n_month_total} (exact={n_exact}, close={n_close})")
    print(f"      Planet matches:      {n_planet}")
    print(f"      Body part matches:   {n_body}")
    print(f"      Validations:         {n_val_passed}/{n_val_total}")

    # Decision tree
    if (n_month_total >= 6 and n_planet >= 3 and
            n_t1 >= 10 and zodiac_hit >= 0.7):
        verdict = 'ZODIAC_DECODED'
        description = (
            f"Zodiac pages substantially decoded via known-plaintext attack. "
            f"{n_month_total} month names matched, {n_planet} planet confirmations, "
            f"{n_t1} definite character assignments. "
            f"Zodiac folios produce {zodiac_hit:.0%} dict_hit."
        )
        next_steps = [
            "Apply zodiac-derived table to full corpus decode",
            "Extract readable zodiac passages for linguistic analysis",
            "Publish zodiac decoding results",
        ]
    elif (n_month_total >= 3 and n_t1 + n_t2 >= 5 and
            corpus_hit >= p16_hit - 0.01):
        verdict = 'PARTIAL_ZODIAC'
        description = (
            f"Partial zodiac decode: {n_month_total} month name matches, "
            f"{n_t1 + n_t2} crib-derived assignments. "
            f"Table {'improves' if corpus_hit > p16_hit else 'preserves'} "
            f"corpus dict_hit ({corpus_hit:.1%} vs Phase16 {p16_hit:.1%})."
        )
        next_steps = [
            "Use zodiac assignments as new anchors for Phase 24-style error correction",
            "Focus additional crib analysis on folios with highest match rates",
            "Cross-validate with boustrophedon reading direction",
        ]
    elif (n_month_total >= 1 or n_csp >= 2) and n_consistent >= 2:
        verdict = 'CRIBS_FOUND'
        description = (
            f"Month name cribs detected: {n_month_total} matches, "
            f"{n_csp} CSP solutions, {n_consistent} consistent assignments. "
            f"Insufficient for readable text but provides grounded anchors."
        )
        next_steps = [
            "Use consistent assignments as hard constraints in future CSP runs",
            "Investigate why matched labels don't propagate to full text",
            "Check if zodiac text uses abbreviated forms",
        ]
    elif month_sel > 1.5 and best_lang:
        # Check if one language clearly dominates
        lang_scores = month_data.get('language_scores', []) if month_data else []
        if lang_scores:
            sorted_langs = sorted(lang_scores, key=lambda x: x.get('mean_agreement', 0), reverse=True)
            if len(sorted_langs) >= 2:
                top = sorted_langs[0].get('mean_agreement', 0)
                second = sorted_langs[1].get('mean_agreement', 0)
                if top > second * 1.3:
                    verdict = 'LANGUAGE_IDENTIFIED'
                    description = (
                        f"Month name cribs prefer {best_lang} "
                        f"(score {top:.3f} vs next {second:.3f}). "
                        f"Selectivity {month_sel:.2f}×. "
                        f"Narrows source language but decode improvement minimal."
                    )
                    next_steps = [
                        f"Re-run all analyses with {best_lang} as primary language",
                        f"Build {best_lang}-specific dictionary and bigram model",
                        "Check if zodiac labels use abbreviated month names",
                    ]
                else:
                    verdict = 'NO_SIGNAL'
                    description = (
                        f"No clear language separation. Month selectivity {month_sel:.2f}× "
                        f"but no consistent language preference."
                    )
                    next_steps = [
                        "Investigate whether zodiac text is astrological medicine rather than astrology",
                        "Check for abbreviated or unusual month name forms",
                    ]
            else:
                verdict = 'NO_SIGNAL'
                description = "Insufficient language comparison data."
                next_steps = ["Add more language variants."]
        else:
            verdict = 'NO_SIGNAL'
            description = "No month crib language data available."
            next_steps = []
    else:
        verdict = 'NO_SIGNAL'
        description = (
            f"No statistically significant signal from zodiac known-plaintext attack. "
            f"Month matches: {n_month_total}, selectivity: {month_sel:.2f}×, "
            f"consistent assignments: {n_consistent}."
        )
        next_steps = [
            "Zodiac text may not contain standard astrological descriptions",
            "Consider astrological medicine or calendrical content",
            "Check if labels encode sign names rather than month names",
            "The cipher may be resistant to known-plaintext attack even with 12 cribs",
        ]

    # -------------------------------------------------------------------
    # Progression
    # -------------------------------------------------------------------
    progression = {
        'phase11': {'dict_hit': 0.111, 'selectivity': 1.92},
        'phase14': {'dict_hit': 0.194, 'selectivity': 3.00},
        'phase15': {'dict_hit': 0.354, 'selectivity': 2.55},
        'phase16': {'dict_hit': round(p16_hit, 4), 'selectivity': 3.38},
        'phase26': {
            'dict_hit': round(corpus_hit, 4),
            'selectivity': round(corpus_sel, 2),
            'zodiac_dict_hit': round(zodiac_hit, 4),
            'n_tier1': n_t1,
            'n_tier2': n_t2,
            'best_language': best_lang,
            'verdict': verdict,
        },
        'trend': ('improvement' if corpus_hit > p16_hit + 0.005
                  else 'plateau' if corpus_hit >= p16_hit - 0.005
                  else 'regression'),
    }

    print(f"\n  {'=' * 60}")
    print(f"  VERDICT: {verdict}")
    print(f"  {'=' * 60}")
    print(f"  {description}")
    print(f"\n  Next steps:")
    for i, step in enumerate(next_steps, 1):
        print(f"    {i}. {step}")

    print(f"\n  Progression:")
    for phase, data in progression.items():
        if phase == 'trend':
            print(f"    Trend: {data}")
        elif isinstance(data, dict):
            hit = data.get('dict_hit', 0)
            sel = data.get('selectivity', 0)
            print(f"    {phase:10s}: dict_hit={hit:.1%}, selectivity={sel:.2f}×")

    result = Phase26VerdictResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        corpus_dict_hit=round(corpus_hit, 4),
        corpus_selectivity=round(corpus_sel, 2),
        zodiac_dict_hit=round(zodiac_hit, 4),
        phase16_dict_hit=round(p16_hit, 4),
        n_tier1_assignments=n_t1,
        n_tier2_assignments=n_t2,
        best_crib_language=best_lang,
        month_crib_selectivity=round(month_sel, 4),
        n_consistent_assignments=n_consistent,
        n_csp_solutions=n_csp,
        n_planet_matches=n_planet,
        n_body_matches=n_body,
        n_validations_passed=n_val_passed,
        n_validations_total=n_val_total,
        verdict=verdict,
        verdict_description=description,
        next_steps=next_steps,
        progression=progression,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase26_verdict.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  → {out_path}")
