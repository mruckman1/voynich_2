"""
Phase 59, Investigation 7: Connector Group Investigation
=========================================================
EVA characters b, h, ckh, u are mapped to coda "l" with no Costamagna
justification.  Costamagna documents 5 codas (m, n, r, s, t) — no "l".
This module tests 7 hypotheses for the connector group: l, m, n, r, s, t,
and null (connectors are NOT codas — they're syllabic).

Dependency chain:
    results/coda_table.json           (Phase 57.1)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/cvc_connectors.json
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
    CodaTable,
    STROKE_TO_CODA_PRIMARY,
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
# Constants
# ---------------------------------------------------------------------------

CONNECTOR_MODIFIERS = {'b', 'h', 'ckh', 'u'}
CODA_CANDIDATES = ['l', 'm', 'n', 'r', 's', 't', None]  # None = syllabic (no coda)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CodaCandidateResult:
    """Results for one coda candidate."""
    coda: Optional[str]
    dict_hit_rate: float
    n_hits: int
    n_tokens: int
    sample_decoded: List[str] = field(default_factory=list)


@dataclass
class PerCharResult:
    """Per-connector-character results."""
    connector_char: str
    best_coda: Optional[str]
    best_rate: float
    per_coda: Dict[str, float] = field(default_factory=dict)
    total: int = 0


@dataclass
class CvcConnectorResult:
    """Full Investigation 7 output."""
    phase: str = "59"
    investigation: str = "7"
    experiment: str = "cvc_connectors"
    n_affected_tokens: int = 0
    per_coda_results: List[CodaCandidateResult] = field(default_factory=list)
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    best_coda: Optional[str] = None
    best_rate: float = 0.0
    null_hypothesis_rate: float = 0.0   # connector as syllabic
    per_char_results: List[PerCharResult] = field(default_factory=list)
    # Gates
    g1_enough_data: bool = False      # ≥ 50 affected tokens
    g2_clear_winner: bool = False     # best ≥ 1.5× second-best
    g3_null_tested: bool = True       # always True (we test it)
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _build_custom_coda_table(connector_coda: Optional[str]) -> CodaTable:
    """Build a coda table with a custom connector coda value."""
    table = build_coda_table('primary')

    if connector_coda is None:
        # Remove connector from the coda mapping entirely
        table.stroke_to_coda = {k: v for k, v in table.stroke_to_coda.items()
                                if k != 'connector'}
        # Reclassify connector chars as non-modifier
        for char in CONNECTOR_MODIFIERS:
            if char in table.modifier_confidence:
                del table.modifier_confidence[char]
            if char in table.eva_modifiers:
                del table.eva_modifiers[char]
    else:
        table.stroke_to_coda['connector'] = connector_coda

    return table


def find_connector_tokens(
    all_tokens: List[str],
    modifier_set: Set[str],
) -> List[Tuple[int, str, str]]:
    """Find tokens containing connector-group modifiers (not first char).

    Returns list of (token_idx, token, connector_char).
    """
    affected = []
    for idx, token in enumerate(all_tokens):
        chars = tokenize_eva_chars(token)
        for ci, char in enumerate(chars):
            if ci == 0:
                continue
            if char in CONNECTOR_MODIFIERS and char in modifier_set:
                affected.append((idx, token, char))
                break
    return affected


def test_coda_candidates(
    affected: List[Tuple[int, str, str]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: Set[str],
) -> List[CodaCandidateResult]:
    """Test each coda candidate on all affected tokens."""
    results = []

    for coda in CODA_CANDIDATES:
        custom_table = _build_custom_coda_table(coda)
        hits = 0
        decoded_list = []

        for _, token, _ in affected:
            result = decode_token_cvc(token, assignment, eva_to_triple, custom_table)
            decoded = result.decoded_cvc
            decoded_list.append(decoded)
            if decoded.lower() in ref_word_set:
                hits += 1

        n = len(affected)
        results.append(CodaCandidateResult(
            coda=coda,
            dict_hit_rate=round(hits / n, 4) if n > 0 else 0.0,
            n_hits=hits,
            n_tokens=n,
            sample_decoded=decoded_list[:10],
        ))

    return results


def per_char_analysis(
    affected: List[Tuple[int, str, str]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: Set[str],
) -> List[PerCharResult]:
    """Per-connector-character analysis: test all candidates independently."""
    results = []

    for target_char in sorted(CONNECTOR_MODIFIERS):
        char_tokens = [(idx, tok, ch) for idx, tok, ch in affected if ch == target_char]
        if not char_tokens:
            continue

        per_coda: Dict[str, float] = {}
        best_coda = None
        best_rate = 0.0

        for coda in CODA_CANDIDATES:
            custom_table = _build_custom_coda_table(coda)
            hits = sum(
                1 for _, tok, _ in char_tokens
                if decode_token_cvc(tok, assignment, eva_to_triple, custom_table)
                .decoded_cvc.lower() in ref_word_set
            )
            rate = hits / len(char_tokens) if char_tokens else 0.0
            per_coda[str(coda)] = round(rate, 4)
            if rate > best_rate:
                best_rate = rate
                best_coda = coda

        results.append(PerCharResult(
            connector_char=target_char,
            best_coda=best_coda,
            best_rate=round(best_rate, 4),
            per_coda=per_coda,
            total=len(char_tokens),
        ))

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_connector():
    """Investigation 7: Test connector group coda candidates."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 7: Connector Group Coda Investigation")
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

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Find affected tokens
    print("\n  Finding tokens with connector modifiers ...")
    affected = find_connector_tokens(all_tokens, modifier_set)
    print(f"  Affected tokens: {len(affected)}")

    if not affected:
        print("  No affected tokens found.")
        result = CvcConnectorResult(runtime_seconds=round(time.time() - t0, 2))
        _save_json(rd, 'cvc_connectors.json', result)
        return

    # Test all 7 coda candidates
    print("\n  Testing 7 coda candidates ...")
    per_coda = test_coda_candidates(affected, assignment, eva_to_triple, ref_word_set)

    # Rank by dict_hit_rate
    ranked = sorted(per_coda, key=lambda x: -x.dict_hit_rate)
    ranking = [{'coda': str(r.coda), 'rate': r.dict_hit_rate} for r in ranked]

    best = ranked[0]
    second_best = ranked[1] if len(ranked) > 1 else None
    null_result = next((r for r in per_coda if r.coda is None), None)

    print(f"\n  {'Coda':<8} {'DictHit':>8} {'Hits':>6} {'Sample decoded'}")
    print(f"  {'-'*8} {'-'*8} {'-'*6} {'-'*30}")
    for r in ranked:
        coda_str = str(r.coda) if r.coda else 'null'
        sample = ', '.join(r.sample_decoded[:3])
        print(f"  {coda_str:<8} {r.dict_hit_rate:>7.1%} {r.n_hits:>6} {sample}")

    # Per-character analysis
    print("\n  Per-connector-character analysis ...")
    per_char = per_char_analysis(affected, assignment, eva_to_triple, ref_word_set)

    print(f"\n  {'Char':<8} {'Best':>6} {'Rate':>8} {'Total':>6}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*6}")
    for pc in per_char:
        coda_str = str(pc.best_coda) if pc.best_coda else 'null'
        print(f"  {pc.connector_char:<8} {coda_str:>6} {pc.best_rate:>7.1%} {pc.total:>6}")

    # Gates
    g1 = len(affected) >= 50
    g2 = (best.dict_hit_rate >= second_best.dict_hit_rate * 1.5
          if second_best and second_best.dict_hit_rate > 0 else False)
    g3 = True  # null hypothesis always tested
    gates_passed = sum([g1, g2, g3])

    print(f"\n  Validation Gates:")
    print(f"    G1 ≥ 50 affected tokens:     {'PASS' if g1 else 'FAIL'} ({len(affected)})")
    print(f"    G2 best ≥ 1.5× second-best:  {'PASS' if g2 else 'FAIL'} "
          f"({best.dict_hit_rate:.1%} vs "
          f"{second_best.dict_hit_rate:.1%})" if second_best else "N/A")
    print(f"    G3 null hypothesis tested:    PASS")
    print(f"    Gates passed: {gates_passed}/3")

    result = CvcConnectorResult(
        n_affected_tokens=len(affected),
        per_coda_results=per_coda,
        ranking=ranking,
        best_coda=best.coda,
        best_rate=best.dict_hit_rate,
        null_hypothesis_rate=null_result.dict_hit_rate if null_result else 0.0,
        per_char_results=per_char,
        g1_enough_data=g1,
        g2_clear_winner=g2,
        g3_null_tested=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_connectors.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 7 completed in {time.time() - t0:.1f}s")
