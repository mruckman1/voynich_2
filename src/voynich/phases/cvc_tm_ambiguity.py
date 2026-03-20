"""
Phase 59, Investigation 3: The t/m Coda Ambiguity
===================================================
The vertical stroke group (al, ol, am, i, m, g) maps to either "t" or "m".
Phase 57 found no corpus-level difference (27.5% vs 27.2%), but per-token
analysis on affected tokens may resolve which coda is correct, or reveal
that different vertical-group characters encode different codas.

Dependency chain:
    results/coda_table.json           (Phase 57.1)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/cvc_tm_ambiguity.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import (
    build_coda_table,
    decode_token_cvc,
)


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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Vertical-group modifiers
# ---------------------------------------------------------------------------

VERTICAL_MODIFIERS = {'al', 'ol', 'am', 'i', 'm', 'g'}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PerModifierTmStats:
    """Per-modifier-character t vs m statistics."""
    modifier_char: str
    t_wins: int = 0
    m_wins: int = 0
    both: int = 0
    neither: int = 0
    total: int = 0
    t_fraction: float = 0.0
    m_fraction: float = 0.0


@dataclass
class CvcTmResult:
    """Full Investigation 3 output."""
    phase: str = "59"
    investigation: str = "3"
    experiment: str = "cvc_tm_ambiguity"
    n_affected_tokens: int = 0
    t_wins: int = 0
    m_wins: int = 0
    both: int = 0
    neither: int = 0
    t_fraction: float = 0.0
    m_fraction: float = 0.0
    verdict: str = ''   # 't', 'm', or 'ambiguous'
    per_modifier: List[PerModifierTmStats] = field(default_factory=list)
    chi2_p: Optional[float] = None   # sub-group independence test
    sample_tokens: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_enough_data: bool = False      # ≥ 100 affected tokens
    g2_clear_winner: bool = False     # t or m wins by ≥ 1.5×
    g3_subgroup_split: Optional[bool] = None  # chi² p < 0.05
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def find_affected_tokens(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_t,
    coda_m,
    modifier_set: Set[str],
) -> List[Dict[str, Any]]:
    """Find tokens where changing vertical→t vs vertical→m changes the decode."""
    affected = []

    for idx, token in enumerate(all_tokens):
        chars = tokenize_eva_chars(token)
        # Check if any vertical-group modifier is present (not first char)
        has_vertical = False
        vert_char = None
        for ci, char in enumerate(chars):
            if ci == 0:
                continue
            if char in VERTICAL_MODIFIERS and char in modifier_set:
                has_vertical = True
                vert_char = char
                break

        if not has_vertical:
            continue

        result_t = decode_token_cvc(token, assignment, eva_to_triple, coda_t)
        result_m = decode_token_cvc(token, assignment, eva_to_triple, coda_m)

        if result_t.decoded_cvc != result_m.decoded_cvc:
            affected.append({
                'token_idx': idx,
                'eva_token': token,
                'modifier_char': vert_char,
                'decoded_t': result_t.decoded_cvc,
                'decoded_m': result_m.decoded_cvc,
            })

    return affected


def compare_tm_per_token(
    affected_tokens: List[Dict[str, Any]],
    ref_word_set: Set[str],
) -> Tuple[int, int, int, int, List[Dict[str, Any]]]:
    """For each affected token, check which coda produces a dict hit."""
    t_wins = m_wins = both = neither = 0
    per_token: List[Dict[str, Any]] = []

    for tok in affected_tokens:
        t_hit = tok['decoded_t'].lower() in ref_word_set
        m_hit = tok['decoded_m'].lower() in ref_word_set

        if t_hit and not m_hit:
            winner = 't'
            t_wins += 1
        elif m_hit and not t_hit:
            winner = 'm'
            m_wins += 1
        elif t_hit and m_hit:
            winner = 'both'
            both += 1
        else:
            winner = 'neither'
            neither += 1

        per_token.append({
            **tok,
            't_hit': t_hit,
            'm_hit': m_hit,
            'winner': winner,
        })

    return t_wins, m_wins, both, neither, per_token


def per_modifier_analysis(per_token_results: List[Dict[str, Any]]) -> List[PerModifierTmStats]:
    """Per-modifier-character t vs m tally."""
    by_mod: Dict[str, Dict[str, int]] = {}
    for tok in per_token_results:
        char = tok['modifier_char']
        if char not in by_mod:
            by_mod[char] = {'t': 0, 'm': 0, 'both': 0, 'neither': 0}
        by_mod[char][tok['winner']] += 1

    results = []
    for char, counts in sorted(by_mod.items()):
        total = sum(counts.values())
        results.append(PerModifierTmStats(
            modifier_char=char,
            t_wins=counts['t'],
            m_wins=counts['m'],
            both=counts['both'],
            neither=counts['neither'],
            total=total,
            t_fraction=counts['t'] / total if total > 0 else 0,
            m_fraction=counts['m'] / total if total > 0 else 0,
        ))
    return results


def chi2_independence(per_modifier: List[PerModifierTmStats]) -> Optional[float]:
    """Chi-squared test: is t/m ratio independent of which modifier char?"""
    # Only test chars with enough data
    testable = [pm for pm in per_modifier if pm.t_wins + pm.m_wins >= 5]
    if len(testable) < 2:
        return None

    from scipy.stats import chi2_contingency
    table = [[pm.t_wins, pm.m_wins] for pm in testable]
    try:
        chi2, p, _, _ = chi2_contingency(table)
        return round(p, 6)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_tm():
    """Investigation 3: Resolve the t/m coda ambiguity."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 3: t/m Coda Ambiguity")
    print("=" * 70)

    rd = str(_results_dir())

    # Load data
    eva_to_triple = build_eva_to_triple_lookup()
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_set: Set[str] = set()
    for cls in mod_data.get('classifications', []):
        if cls['final_classification'] in ('modifier', 'ambiguous'):
            modifier_set.add(cls['eva_char'])

    # Build reference dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    coda_t = build_coda_table('primary')     # vertical → t
    coda_m = build_coda_table('alternate')   # vertical → m

    # Find affected tokens
    print("\n  Finding tokens affected by t/m ambiguity ...")
    affected = find_affected_tokens(
        all_tokens, assignment, eva_to_triple, coda_t, coda_m, modifier_set)
    print(f"  Affected tokens: {len(affected)}")

    if not affected:
        print("  No affected tokens found. Saving empty result.")
        result = CvcTmResult(verdict='no_data', runtime_seconds=round(time.time() - t0, 2))
        _save_json(rd, 'cvc_tm_ambiguity.json', result)
        return

    # Per-token comparison
    print("  Comparing t vs m per token ...")
    t_wins, m_wins, both, neither, per_token = compare_tm_per_token(affected, ref_word_set)

    total = len(affected)
    t_frac = t_wins / total
    m_frac = m_wins / total
    verdict = 't' if t_wins > m_wins * 1.5 else 'm' if m_wins > t_wins * 1.5 else 'ambiguous'

    print(f"  t wins:   {t_wins} ({t_frac:.1%})")
    print(f"  m wins:   {m_wins} ({m_frac:.1%})")
    print(f"  Both:     {both}")
    print(f"  Neither:  {neither}")
    print(f"  Verdict:  {verdict}")

    # Per-modifier analysis
    print("\n  Per-modifier character analysis:")
    per_mod = per_modifier_analysis(per_token)
    print(f"  {'Char':<8} {'t':>6} {'m':>6} {'Both':>6} {'Neither':>8} {'Total':>6}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for pm in per_mod:
        print(f"  {pm.modifier_char:<8} {pm.t_wins:>6} {pm.m_wins:>6} "
              f"{pm.both:>6} {pm.neither:>8} {pm.total:>6}")

    # Chi-squared independence test
    chi2_p = chi2_independence(per_mod)
    if chi2_p is not None:
        print(f"\n  Chi² independence test: p = {chi2_p:.6f}")
        print(f"  Sub-groups {'DO' if chi2_p < 0.05 else 'do NOT'} encode different codas")

    # Sample tokens
    samples = []
    for tok in per_token[:20]:
        samples.append({
            'eva': tok['eva_token'],
            'mod': tok['modifier_char'],
            't': tok['decoded_t'],
            'm': tok['decoded_m'],
            'winner': tok['winner'],
        })

    print(f"\n  Sample affected tokens:")
    for s in samples[:10]:
        print(f"    {s['eva']:14s} mod={s['mod']:4s} "
              f"t={s['t']:12s} m={s['m']:12s} → {s['winner']}")

    # Gates
    g1 = len(affected) >= 100
    g2 = (t_wins > m_wins * 1.5) or (m_wins > t_wins * 1.5)
    g3 = chi2_p < 0.05 if chi2_p is not None else None
    gates_passed = sum(filter(None, [g1, g2, g3 if g3 is not None else False]))

    print(f"\n  Validation Gates:")
    print(f"    G1 ≥ 100 affected tokens:  {'PASS' if g1 else 'FAIL'} ({len(affected)})")
    print(f"    G2 winner by ≥ 1.5×:       {'PASS' if g2 else 'FAIL'}")
    if chi2_p is not None:
        print(f"    G3 sub-group chi² p<0.05:  {'PASS' if g3 else 'FAIL'} (p={chi2_p:.4f})")
    else:
        print(f"    G3 sub-group chi²:         N/A (insufficient data)")
    print(f"    Gates passed: {gates_passed}/3")

    result = CvcTmResult(
        n_affected_tokens=len(affected),
        t_wins=t_wins,
        m_wins=m_wins,
        both=both,
        neither=neither,
        t_fraction=round(t_frac, 4),
        m_fraction=round(m_frac, 4),
        verdict=verdict,
        per_modifier=per_mod,
        chi2_p=chi2_p,
        sample_tokens=samples,
        g1_enough_data=g1,
        g2_clear_winner=g2,
        g3_subgroup_split=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_tm_ambiguity.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 3 completed in {time.time() - t0:.1f}s")
