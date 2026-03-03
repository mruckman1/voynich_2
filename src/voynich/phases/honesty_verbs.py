"""
Phase 17.0.3 – Positional Verb Decode Test
===========================================
Decodes the 15 verb candidate stems from Phase 9 and compares them to
Latin pharmaceutical imperatives.  Tests whether the phonetic table
produces Latin-imperative-like strings for position-0 concentrated tokens.

NOTE: Phase 9 verb identification FAILED its own gate (selectivity 0.92×).
Success here would be surprising independent confirmation; failure is expected.

Dependency chain:
    verb_identification.json  (Phase 9 – 15 verb candidates)
    modifier_integrate.json   (Phase 16 modifiers)
    combined_refine.json      (Phase 15 best_assignment)
        → honesty_verbs.json  (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
)
from voynich.core.reference import LATIN_IMPERATIVE_RANKED
from voynich.phases.csp_solver import decode_token


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


def _reconstruct_modifier_rules(data: Dict) -> Tuple[Set[str], Dict[str, str]]:
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VerbDecodeResult:
    voynich_stem: str
    frequency_rank: int
    decoded_stem: str
    decoded_stripped: str
    # vs assigned imperative (from Phase 9)
    assigned_imperative: str
    ed_to_assigned: int
    # vs best-matching imperative overall
    best_imperative: str
    ed_to_best: int
    is_match_assigned: bool
    is_match_best: bool


@dataclass
class HonestyVerbResult:
    phase9_gate_passed: bool
    phase9_selectivity: float

    n_verbs: int
    n_imperatives: int
    verb_results: List[Dict]

    # Match counts (against assigned imperative)
    n_exact_match: int
    n_ed1_match: int
    n_ed2_match: int

    # Match counts (against best-matching imperative)
    n_best_exact: int
    n_best_ed1: int
    n_best_ed2: int

    # Rank correlation
    rho: float
    rho_p_value: float

    # Imperative syllable coverage
    imperative_coverage: Dict[str, List[str]]

    gate_passed: bool
    gate_note: str
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_honesty_verbs() -> None:
    """Step 17.0.3: Positional verb decode test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 17.0.3: Positional Verb Decode Test")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load Phase 9 verb identification ───
    print("\n  1. Loading Phase 9 verb identification …")
    verb_path = os.path.join(rd, 'verb_identification.json')
    if not os.path.exists(verb_path):
        print("  [SKIP] verb_identification.json not found")
        return
    with open(verb_path) as f:
        verb_data = json.load(f)

    p9_gate = verb_data.get('gate_passed', False)
    p9_selectivity = verb_data.get('assignment_selectivity', 0.0)
    voynich_stems = verb_data.get('voynich_stems', [])
    p9_assignments = verb_data.get('assignments', [])

    print(f"      Phase 9 gate: {'PASS' if p9_gate else 'FAIL'}")
    print(f"      Phase 9 selectivity: {p9_selectivity:.2f}×")
    print(f"      {len(voynich_stems)} verb candidate stems")

    # Build stem → assigned imperative mapping
    stem_to_imperative: Dict[str, str] = {}
    for a in p9_assignments:
        stem_to_imperative[a['voynich_stem']] = a['latin_verb']

    # Build stem → frequency rank
    stem_profiles = verb_data.get('voynich_verb_profiles', [])
    stem_to_rank: Dict[str, int] = {}
    for p in stem_profiles:
        stem_to_rank[p['stem']] = p['frequency_rank']

    # ─── Load Phase 16 modifiers ───
    print("\n  2. Loading Phase 16 results …")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # ─── Load Phase 15 assignment ───
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    eva_to_triple = build_eva_to_triple_lookup()
    imperatives = list(LATIN_IMPERATIVE_RANKED.keys())

    # ─── Decode each verb stem ───
    print(f"\n  3. Decoding {len(voynich_stems)} verb stems …")
    results: List[VerbDecodeResult] = []

    print(f"      {'Rank':>4} {'Stem':<12} {'Decoded':<15} {'Stripped':<15} "
          f"{'Assigned':<12} {'ED':>3} {'Best':<12} {'ED':>3}")
    print("      " + "-" * 80)

    for stem in voynich_stems:
        freq_rank = stem_to_rank.get(stem, 99)
        assigned = stem_to_imperative.get(stem, '')

        # Decode with modifier rules
        decoded = decode_token_modifier_aware(
            stem, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        # Decode with modifier stripping
        stripped = decode_token_modifier_aware(
            stem, assignment, eva_to_triple, modifier_chars,
        )

        # Compare to assigned imperative
        ed_assigned = _edit_distance(decoded.lower(), assigned) if assigned else 99

        # Find best-matching imperative
        best_imp = ''
        best_ed = 999
        for imp in imperatives:
            ed = _edit_distance(decoded.lower(), imp)
            if ed < best_ed:
                best_ed = ed
                best_imp = imp
        # Also check stripped version
        for imp in imperatives:
            ed = _edit_distance(stripped.lower(), imp)
            if ed < best_ed:
                best_ed = ed
                best_imp = imp

        vdr = VerbDecodeResult(
            voynich_stem=stem,
            frequency_rank=freq_rank,
            decoded_stem=decoded,
            decoded_stripped=stripped,
            assigned_imperative=assigned,
            ed_to_assigned=ed_assigned if ed_assigned < 99 else -1,
            best_imperative=best_imp,
            ed_to_best=best_ed if best_ed < 999 else -1,
            is_match_assigned=ed_assigned <= 1,
            is_match_best=best_ed <= 1,
        )
        results.append(vdr)

        a_marker = '*' if ed_assigned <= 1 else ' '
        b_marker = '*' if best_ed <= 1 else ' '
        print(f"    {a_marker} {freq_rank:>4} {stem:<12} {decoded:<15} {stripped:<15} "
              f"{assigned:<12} {ed_assigned:>3} {best_imp:<12} {best_ed:>3} {b_marker}")

    # ─── Count matches ───
    n_exact_assigned = sum(1 for r in results if r.ed_to_assigned == 0)
    n_ed1_assigned = sum(1 for r in results if 0 <= r.ed_to_assigned <= 1)
    n_ed2_assigned = sum(1 for r in results if 0 <= r.ed_to_assigned <= 2)

    n_best_exact = sum(1 for r in results if r.ed_to_best == 0)
    n_best_ed1 = sum(1 for r in results if 0 <= r.ed_to_best <= 1)
    n_best_ed2 = sum(1 for r in results if 0 <= r.ed_to_best <= 2)

    print(f"\n  4. Match summary (vs assigned imperative):")
    print(f"      Exact: {n_exact_assigned}/15")
    print(f"      ED≤1:  {n_ed1_assigned}/15")
    print(f"      ED≤2:  {n_ed2_assigned}/15")
    print(f"\n      Match summary (vs best imperative):")
    print(f"      Exact: {n_best_exact}/15")
    print(f"      ED≤1:  {n_best_ed1}/15")
    print(f"      ED≤2:  {n_best_ed2}/15")

    # ─── Rank correlation ───
    print("\n  5. Rank correlation (frequency rank vs match quality) …")
    ranks = []
    match_scores = []
    for r in results:
        ranks.append(r.frequency_rank)
        if r.ed_to_best == 0:
            match_scores.append(1.0)
        elif r.ed_to_best == 1:
            match_scores.append(0.5)
        elif r.ed_to_best == 2:
            match_scores.append(0.25)
        else:
            match_scores.append(0.0)

    rho = 0.0
    rho_p = 1.0
    if len(ranks) >= 3:
        try:
            from voynich.core.stats import rank_correlation
            rho, rho_p = rank_correlation(ranks, match_scores)
        except (ImportError, ValueError):
            pass
    print(f"      rho = {rho:.3f}, p = {rho_p:.4f}")

    # ─── Imperative syllable coverage ───
    print("\n  6. Checking imperative syllable coverage in assignment …")
    assigned_syllables = set(assignment.values())
    coverage: Dict[str, List[str]] = {}
    for imp in imperatives:
        # Check which 2-char syllables from the imperative are in the assignment values
        present = []
        for i in range(0, len(imp) - 1, 2):
            syl = imp[i:i + 2]
            if syl in assigned_syllables:
                present.append(syl)
        coverage[imp] = present
        marker = '*' if present else ' '
        print(f"      {marker} {imp:<12} syllables present: {present}")

    # ─── Gate ───
    gate_passed = n_best_ed1 >= 5 and abs(rho) > 0.3
    gate_note = "Phase 9 verb identification FAILED its own gate (selectivity 0.92×)"

    print(f"\n  7. Gate: n_ed1_match >= 5 AND |rho| > 0.3")
    print(f"      n_best_ed1 = {n_best_ed1}")
    print(f"      rho = {rho:.3f}")
    print(f"      NOTE: {gate_note}")
    print(f"      {'PASS' if gate_passed else 'FAIL'}")

    # ─── Verdict ───
    if gate_passed:
        verdict = (
            f"PASS: {n_best_ed1}/15 verb candidates decode within ED≤1 of "
            f"a Latin imperative (rho={rho:.3f}). Independent confirmation."
        )
    elif n_best_ed1 >= 2 or n_best_ed2 >= 5:
        verdict = (
            f"MARGINAL: {n_best_ed1} at ED≤1, {n_best_ed2} at ED≤2. "
            f"Some near-misses but below threshold (rho={rho:.3f}). "
            f"Note: Phase 9 verb ID itself failed."
        )
    else:
        verdict = (
            f"FAIL: {n_best_ed1}/15 at ED≤1 (rho={rho:.3f}). "
            f"Positionally-identified verbs don't decode to Latin imperatives. "
            f"Expected given Phase 9 verb ID failure."
        )

    print(f"\n  Verdict: {verdict}")

    # ─── Save ───
    result = HonestyVerbResult(
        phase9_gate_passed=p9_gate,
        phase9_selectivity=round(p9_selectivity, 4),
        n_verbs=len(voynich_stems),
        n_imperatives=len(imperatives),
        verb_results=[_convert(asdict(r)) for r in results],
        n_exact_match=n_exact_assigned,
        n_ed1_match=n_ed1_assigned,
        n_ed2_match=n_ed2_assigned,
        n_best_exact=n_best_exact,
        n_best_ed1=n_best_ed1,
        n_best_ed2=n_best_ed2,
        rho=round(rho, 4),
        rho_p_value=round(rho_p, 6),
        imperative_coverage=coverage,
        gate_passed=gate_passed,
        gate_note=gate_note,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'honesty_verbs.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
