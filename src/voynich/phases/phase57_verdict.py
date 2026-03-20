"""
Phase 57: CVC Coda Decode — Token Diagnostics + Verdict
========================================================
Evaluates 7 validation gates, produces top-20 token diagnostics,
and orchestrates the full Phase 57 pipeline.

Dependency chain:
    results/coda_table.json          (Step 57.1)
    results/cvc_coda_signal.json     (Step 57.4)
    results/cvc_compare.json         (Step 57.5)
        -> results/phase57_verdict.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TokenDiagnostic:
    eva_token: str
    frequency: int
    old_cv_decode: str
    new_cvc_decode: str
    cv_length: int
    cvc_length: int
    cvc_in_dict: bool
    cv_in_dict: bool
    char_roles: List[str]


@dataclass
class Phase57Result:
    phase: str = "57"
    experiment: str = "cvc_coda_decode"
    token_diagnostics: List[TokenDiagnostic] = field(default_factory=list)
    gates: Dict[str, bool] = field(default_factory=dict)
    gate_details: Dict[str, str] = field(default_factory=dict)
    n_passed: int = 0
    n_total: int = 7
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Token diagnostics (Step 57.8)
# ---------------------------------------------------------------------------

def _token_diagnostics(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
    ref_word_set: set,
    n_top: int = 20,
) -> List[TokenDiagnostic]:
    """Generate diagnostic comparison for the top-N most frequent tokens."""
    freq = Counter(all_tokens)
    top_tokens = [tok for tok, _ in freq.most_common(n_top)]

    diagnostics = []
    for tok in top_tokens:
        r = decode_token_cvc(tok, assignment, eva_to_triple, coda_table)
        diagnostics.append(TokenDiagnostic(
            eva_token=tok,
            frequency=freq[tok],
            old_cv_decode=r.decoded_cv,
            new_cvc_decode=r.decoded_cvc,
            cv_length=len(r.decoded_cv),
            cvc_length=len(r.decoded_cvc),
            cvc_in_dict=r.decoded_cvc.lower() in ref_word_set,
            cv_in_dict=r.decoded_cv.lower() in ref_word_set,
            char_roles=r.char_roles,
        ))

    return diagnostics


# ---------------------------------------------------------------------------
# Validation gates (Step 57.7)
# ---------------------------------------------------------------------------

def _evaluate_gates(compare_data: Dict) -> Dict[str, bool]:
    """Evaluate 7 validation gates from cvc_compare.json data."""
    strategies = {s['name']: s for s in compare_data.get('strategies', [])}

    # Get the CVC primary strategy and the baseline
    cvc = strategies.get('cvc_primary', {})
    cv_strip = strategies.get('cv_strip', {})
    r3 = strategies.get('r3_combined', {})

    # Use cv_strip as baseline for comparison
    baseline = cv_strip if cv_strip else r3

    gates = {}
    details = {}

    # G1: CVC dict_hit >= 43.6% (0.436)
    cvc_dh = cvc.get('dict_hit', 0)
    gates['G1_no_regression'] = cvc_dh >= 0.436
    details['G1_no_regression'] = f"CVC dict_hit={cvc_dh:.4f} (threshold=0.436)"

    # G2: CVC selectivity >= 1.5x
    cvc_sel = cvc.get('mean_selectivity', 0)
    gates['G2_selectivity'] = cvc_sel >= 1.5
    details['G2_selectivity'] = f"CVC selectivity={cvc_sel:.2f}x (threshold=1.5x)"

    # G3: CVC Costamagna attestation >= 50%
    cvc_val = cvc.get('cvc_validation', {})
    attest_rate = cvc_val.get('attestation_rate_type', 0) if cvc_val else 0
    gates['G3_costamagna_attest'] = attest_rate >= 0.50
    details['G3_costamagna_attest'] = f"CVC attestation={attest_rate:.1%} (threshold=50%)"

    # G4: CVC signal count >= 56
    cvc_sig = cvc.get('n_signal_words', 0)
    gates['G4_signal_count'] = cvc_sig >= 56
    details['G4_signal_count'] = f"CVC signal words={cvc_sig} (threshold=56)"

    # G5: CVC bigram z >= 2.0
    cvc_z = cvc.get('bigram_z', 0)
    gates['G5_bigram_z'] = cvc_z >= 2.0
    details['G5_bigram_z'] = f"CVC bigram z={cvc_z:.2f} (threshold=2.0)"

    # G6: CVC mean word length > CV mean word length
    cvc_len = cvc.get('mean_word_length', 0)
    cv_len = baseline.get('mean_word_length', 0)
    gates['G6_word_length'] = cvc_len > cv_len
    details['G6_word_length'] = f"CVC len={cvc_len:.2f} vs CV len={cv_len:.2f}"

    # G7: CVC net signal > CV net signal
    cvc_net = cvc.get('net_signal', 0)
    cv_net = baseline.get('net_signal', 0)
    gates['G7_net_signal'] = cvc_net > cv_net
    details['G7_net_signal'] = f"CVC net={cvc_net} vs CV net={cv_net}"

    return gates, details


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_cvc_tokens():
    """Step 57.8: Token-level diagnostic comparison."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 57, Step 8: Token Diagnostics")
    print("=" * 70)

    rd = str(_results_dir())
    eva_to_triple = build_eva_to_triple_lookup()

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    coda_table = build_coda_table('primary')

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    diagnostics = _token_diagnostics(
        all_tokens, assignment, eva_to_triple, coda_table, ref_word_set)

    print(f"\n  {'Token':<14} {'Freq':>5} {'CV decode':<14} {'CVC decode':<14} "
          f"{'CV?':>4} {'CVC?':>4} {'Roles'}")
    print(f"  {'-'*14} {'-'*5} {'-'*14} {'-'*14} {'-'*4} {'-'*4} {'-'*30}")
    for d in diagnostics:
        cv_mark = 'Y' if d.cv_in_dict else '-'
        cvc_mark = 'Y' if d.cvc_in_dict else '-'
        roles = ','.join(r[0] for r in d.char_roles)
        print(f"  {d.eva_token:<14} {d.frequency:>5} {d.old_cv_decode:<14} "
              f"{d.new_cvc_decode:<14} {cv_mark:>4} {cvc_mark:>4} {roles}")

    print(f"\n  Step 57.8 completed in {time.time() - t0:.1f}s")


def run_phase57_verdict():
    """Phase 57 verdict: evaluate gates and produce final assessment."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 57: CVC Coda Decode Verdict")
    print("=" * 70)

    rd = str(_results_dir())

    # Load comparison results
    compare_data = _safe_load(os.path.join(rd, 'cvc_compare.json'))
    if not compare_data:
        print("  [SKIP] cvc_compare.json not found - run cvc-compare first")
        return

    # Evaluate gates
    gates, details = _evaluate_gates(compare_data)
    n_passed = sum(gates.values())

    if n_passed >= 5:
        verdict = f"PASS ({n_passed}/7) - CVC coda decode improves on baseline"
    elif n_passed >= 3:
        verdict = f"PARTIAL ({n_passed}/7) - CVC shows promise, needs refinement"
    else:
        verdict = f"FAIL ({n_passed}/7) - Coda mapping hypothesis not supported"

    # Print gate results
    print(f"\n  Validation Gates:")
    for gate_id, passed in gates.items():
        status = 'PASS' if passed else 'FAIL'
        detail = details.get(gate_id, '')
        print(f"    [{status}] {gate_id}: {detail}")

    print(f"\n  Score: {n_passed}/7")
    print(f"  Verdict: {verdict}")

    # Token diagnostics
    print("\n  Running token diagnostics ...")
    eva_to_triple = build_eva_to_triple_lookup()
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    coda_table = build_coda_table('primary')

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    diagnostics = _token_diagnostics(
        all_tokens, assignment, eva_to_triple, coda_table, ref_word_set)

    print(f"\n  Top-10 token comparisons:")
    for d in diagnostics[:10]:
        cv_mark = 'dict' if d.cv_in_dict else '----'
        cvc_mark = 'dict' if d.cvc_in_dict else '----'
        print(f"    {d.eva_token:<12} CV={d.old_cv_decode:<10}[{cv_mark}]  "
              f"CVC={d.new_cvc_decode:<10}[{cvc_mark}]")

    result = Phase57Result(
        token_diagnostics=diagnostics,
        gates=gates,
        gate_details=details,
        n_passed=n_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )
    path = _save_json(rd, 'phase57_verdict.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Phase 57 verdict completed in {time.time() - t0:.1f}s")


def run_phase57():
    """Run full Phase 57 pipeline."""
    from voynich.phases.coda_markers import run_coda_table
    from voynich.phases.cvc_coda_signal import run_cvc_coda_signal, run_cvc_compare

    print("=" * 70)
    print("PHASE 57: CVC Coda Decode (Full Pipeline)")
    print("=" * 70)
    t0 = time.time()

    print("\n--- Step 57.1: Coda Marker Table ---")
    run_coda_table()

    print("\n\n--- Step 57.4: CVC Signal Isolation ---")
    run_cvc_coda_signal()

    print("\n\n--- Step 57.5: Strategy Comparison ---")
    run_cvc_compare()

    print("\n\n--- Step 57.7-8: Verdict + Token Diagnostics ---")
    run_phase57_verdict()

    print(f"\n{'='*70}")
    print(f"Phase 57 completed in {time.time() - t0:.1f}s")
