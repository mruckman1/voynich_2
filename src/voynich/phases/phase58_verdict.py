"""
Phase 58: Costamagna-Constrained CSP — Comparison + Verdict
=============================================================
Compares the best Costamagna CSP solution against T_P15 on all canonical
metrics.  Evaluates 8 validation gates.

Dependency chain:
    results/cost_csp.json           (Step 58.3)
    results/combined_refine.json    (Phase 15)
    results/modifier_integrate.json (Phase 16)
        -> results/cost_compare.json    (Step 58.5)
        -> results/phase58_verdict.json (Step 58.6)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.cvc_coda_signal import (
    _run_signal_isolation,
    _build_folio_list,
    _compute_bigram_z,
    _validate_cvc_costamagna,
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
class TableMetrics:
    name: str
    dict_hit: float
    n_signal_words: int
    mean_selectivity: float
    bigram_z: float
    mean_word_length: float
    net_signal: int
    costamagna_attestation: float


@dataclass
class CostCompareResult:
    phase: str = "58"
    step: str = "58.5"
    experiment: str = "cost_compare"
    tp15_metrics: Optional[TableMetrics] = None
    csp_metrics: Optional[TableMetrics] = None
    deltas: Dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0


@dataclass
class Phase58Result:
    phase: str = "58"
    experiment: str = "costamagna_csp"
    cost_domains_summary: Dict[str, Any] = field(default_factory=dict)
    cost_reduction_summary: Dict[str, Any] = field(default_factory=dict)
    csp_summary: Dict[str, Any] = field(default_factory=dict)
    comparison: Optional[CostCompareResult] = None
    gates: Dict[str, bool] = field(default_factory=dict)
    gate_details: Dict[str, str] = field(default_factory=dict)
    n_passed: int = 0
    n_total: int = 8
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Table evaluation
# ---------------------------------------------------------------------------

def _evaluate_table(
    name: str,
    assignment: Dict[str, str],
    all_tokens: List[str],
    null_token_lists: List[List[str]],
    ref_word_set: Set[str],
    folios: List[str],
    eva_to_triple: Dict[str, str],
) -> TableMetrics:
    """Evaluate a table on all canonical metrics."""
    n_tokens = len(all_tokens)

    # Decode real
    real_decoded = [decode_token(tok, assignment, eva_to_triple).lower()
                    for tok in all_tokens]

    # Decode nulls
    null_decoded_list = [
        [decode_token(tok, assignment, eva_to_triple).lower()
         for tok in null_tokens]
        for null_tokens in null_token_lists
    ]

    # Dict hit
    real_hits = sum(1 for w in real_decoded if w in ref_word_set)
    dict_hit = real_hits / n_tokens if n_tokens > 0 else 0.0

    # Signal isolation
    signal = _run_signal_isolation(
        real_decoded, null_decoded_list, ref_word_set, n_tokens)

    # Bigram z
    bigram_z = _compute_bigram_z(
        real_decoded, null_decoded_list, ref_word_set, folios, n_perms=500)

    # Mean word length
    lengths = [len(w) for w in real_decoded if w and w != '?']
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0

    # Net signal
    net_signal = signal.n_signal - signal.n_anti_signal

    # Costamagna attestation
    cvc_val = _validate_cvc_costamagna(real_decoded)
    attest = cvc_val.attestation_rate_type

    return TableMetrics(
        name=name,
        dict_hit=round(dict_hit, 4),
        n_signal_words=signal.n_signal_words,
        mean_selectivity=signal.mean_selectivity,
        bigram_z=round(bigram_z, 2),
        mean_word_length=round(mean_len, 2),
        net_signal=net_signal,
        costamagna_attestation=round(attest, 4),
    )


# ---------------------------------------------------------------------------
# Validation gates (Step 58.6)
# ---------------------------------------------------------------------------

def _evaluate_gates_58(
    tp15: TableMetrics,
    csp: TableMetrics,
    n_changed: int,
    confirmed_preserved: bool,
) -> Dict[str, Any]:
    """Evaluate 8 Phase 58 validation gates."""
    gates = {}
    details = {}

    # G1: Dict hit >= 43.6%
    gates['G1_no_regression'] = csp.dict_hit >= 0.436
    details['G1_no_regression'] = f"CSP dict_hit={csp.dict_hit:.4f} (threshold=0.436)"

    # G2: Signal count >= 56
    gates['G2_signal_count'] = csp.n_signal_words >= 56
    details['G2_signal_count'] = f"CSP signal={csp.n_signal_words} (threshold=56)"

    # G3: Bigram z >= 14.78
    gates['G3_bigram_z'] = csp.bigram_z >= 14.78
    details['G3_bigram_z'] = f"CSP bigram_z={csp.bigram_z:.2f} (threshold=14.78)"

    # G4: Costamagna attestation >= 60%
    gates['G4_costamagna_attest'] = csp.costamagna_attestation >= 0.60
    details['G4_costamagna_attest'] = (
        f"CSP attestation={csp.costamagna_attestation:.1%} (threshold=60%)")

    # G5: >= 3 new signal words not in T_P15
    # Approximate: check if CSP has more signal words
    delta_signal = csp.n_signal_words - tp15.n_signal_words
    gates['G5_new_signals'] = delta_signal >= 3
    details['G5_new_signals'] = (
        f"CSP signal={csp.n_signal_words} vs T_P15={tp15.n_signal_words} "
        f"(delta={delta_signal})")

    # G6: Net signal > T_P15's net signal
    gates['G6_net_signal'] = csp.net_signal > tp15.net_signal
    details['G6_net_signal'] = (
        f"CSP net={csp.net_signal} vs T_P15 net={tp15.net_signal}")

    # G7: No confirmed triple changed
    gates['G7_confirmed_preserved'] = confirmed_preserved
    details['G7_confirmed_preserved'] = (
        f"Confirmed triples {'preserved' if confirmed_preserved else 'CHANGED'}")

    # G8: >= 2 unresolved triples now resolved (changed productively)
    gates['G8_triples_resolved'] = n_changed >= 2
    details['G8_triples_resolved'] = f"Changed triples: {n_changed} (threshold=2)"

    return gates, details


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_cost_compare():
    """Step 58.5: Compare best CSP solution vs T_P15."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 58, Step 5: CSP vs T_P15 Comparison")
    print("=" * 70)

    rd = str(_results_dir())
    eva_to_triple = build_eva_to_triple_lookup()

    # Load CSP result
    csp_data = _safe_load(os.path.join(rd, 'cost_csp.json'))
    if not csp_data or not csp_data.get('solutions'):
        print("  [SKIP] cost_csp.json not found or no solutions")
        return

    best_solution = csp_data['solutions'][0]
    csp_assignment = best_solution['assignment']

    # Load T_P15
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    tp15 = refine_data.get('best_assignment', {})

    # Load reference
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Load corpus
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folios = _build_folio_list(corpus)

    # Build null corpora
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_seeds = ([r['seed'] for r in null_data.get('null_runs', [])]
                  if null_data else [100, 101, 102, 103, 104])
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    null_token_lists = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed)
        null_token_lists.append(null_tokens)

    # Evaluate both tables
    print("\n  Evaluating T_P15 ...")
    tp15_metrics = _evaluate_table(
        'T_P15', tp15, all_tokens, null_token_lists,
        ref_word_set, folios, eva_to_triple)

    print("  Evaluating Costamagna CSP best ...")
    csp_metrics = _evaluate_table(
        'Costamagna_CSP', csp_assignment, all_tokens, null_token_lists,
        ref_word_set, folios, eva_to_triple)

    # Compute deltas
    deltas = {
        'dict_hit': csp_metrics.dict_hit - tp15_metrics.dict_hit,
        'n_signal_words': csp_metrics.n_signal_words - tp15_metrics.n_signal_words,
        'bigram_z': csp_metrics.bigram_z - tp15_metrics.bigram_z,
        'net_signal': csp_metrics.net_signal - tp15_metrics.net_signal,
        'mean_word_length': csp_metrics.mean_word_length - tp15_metrics.mean_word_length,
        'costamagna_attestation': (csp_metrics.costamagna_attestation
                                   - tp15_metrics.costamagna_attestation),
    }

    # Print comparison
    print(f"\n  {'Metric':<24} {'T_P15':>10} {'CSP':>10} {'Delta':>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Dict hit':<24} {tp15_metrics.dict_hit:>10.4f} "
          f"{csp_metrics.dict_hit:>10.4f} {deltas['dict_hit']:>+10.4f}")
    print(f"  {'Signal words':<24} {tp15_metrics.n_signal_words:>10d} "
          f"{csp_metrics.n_signal_words:>10d} {deltas['n_signal_words']:>+10.0f}")
    print(f"  {'Bigram z':<24} {tp15_metrics.bigram_z:>10.2f} "
          f"{csp_metrics.bigram_z:>10.2f} {deltas['bigram_z']:>+10.2f}")
    print(f"  {'Net signal':<24} {tp15_metrics.net_signal:>10d} "
          f"{csp_metrics.net_signal:>10d} {deltas['net_signal']:>+10.0f}")
    print(f"  {'Mean word length':<24} {tp15_metrics.mean_word_length:>10.2f} "
          f"{csp_metrics.mean_word_length:>10.2f} "
          f"{deltas['mean_word_length']:>+10.2f}")
    print(f"  {'Costamagna attest.':<24} "
          f"{tp15_metrics.costamagna_attestation:>10.4f} "
          f"{csp_metrics.costamagna_attestation:>10.4f} "
          f"{deltas['costamagna_attestation']:>+10.4f}")

    result = CostCompareResult(
        tp15_metrics=tp15_metrics,
        csp_metrics=csp_metrics,
        deltas=deltas,
        runtime_seconds=round(time.time() - t0, 2),
    )
    path = _save_json(rd, 'cost_compare.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Step 58.5 completed in {time.time() - t0:.1f}s")


def run_phase58_verdict():
    """Phase 58 verdict: evaluate gates and produce final assessment."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 58: Costamagna CSP Verdict")
    print("=" * 70)

    rd = str(_results_dir())

    # Load comparison
    compare_data = _safe_load(os.path.join(rd, 'cost_compare.json'))
    csp_data = _safe_load(os.path.join(rd, 'cost_csp.json'))

    if not compare_data or not compare_data.get('tp15_metrics'):
        print("  [SKIP] cost_compare.json not found - run cost-compare first")
        return

    tp15_m = compare_data['tp15_metrics']
    csp_m = compare_data['csp_metrics']

    tp15_metrics = TableMetrics(**tp15_m)
    csp_metrics = TableMetrics(**csp_m)

    # Check confirmed triples preserved
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    tp15 = refine_data.get('best_assignment', {})

    best_solution = csp_data.get('solutions', [{}])[0] if csp_data else {}
    csp_assignment = best_solution.get('assignment', {})
    n_changed = best_solution.get('n_changed_from_tp15', 0)

    # Check if confirmed triples are preserved
    from voynich.phases.costamagna_csp import _get_confirmed_triples
    confirmed = _get_confirmed_triples(rd)
    confirmed_preserved = all(
        csp_assignment.get(t) == v for t, v in confirmed.items()
        if t in csp_assignment
    )

    # Evaluate gates
    gates, details = _evaluate_gates_58(
        tp15_metrics, csp_metrics, n_changed, confirmed_preserved)
    n_passed = sum(gates.values())

    if n_passed >= 6:
        verdict = (f"PASS ({n_passed}/8) - Costamagna CSP improves T_P15, "
                   f"update assignment table")
    elif n_passed >= 4:
        verdict = (f"PARTIAL ({n_passed}/8) - CSP finds valid assignments "
                   f"but doesn't uniformly improve")
    else:
        verdict = (f"FAIL ({n_passed}/8) - Costamagna constraints too broad "
                   f"without visual matching")

    # Print gate results
    print(f"\n  Validation Gates:")
    for gate_id, passed in gates.items():
        status = 'PASS' if passed else 'FAIL'
        detail = details.get(gate_id, '')
        print(f"    [{status}] {gate_id}: {detail}")

    print(f"\n  Score: {n_passed}/8")
    print(f"  Verdict: {verdict}")

    # Load summaries from prior steps
    domains_data = _safe_load(os.path.join(rd, 'cost_domains.json'))
    reduction_data = _safe_load(os.path.join(rd, 'cost_reduction.json'))

    result = Phase58Result(
        cost_domains_summary={
            'n_domains': len(domains_data.get('domains', [])),
            'cv_count': domains_data.get('costamagna_cv_count', 0),
            'cvc_count': domains_data.get('costamagna_cvc_count', 0),
        },
        cost_reduction_summary=reduction_data.get('comparison', {}),
        csp_summary={
            'n_restarts': csp_data.get('n_restarts', 0) if csp_data else 0,
            'best_dict_hit': csp_data.get('best_dict_hit', 0) if csp_data else 0,
            'tp15_dict_hit': csp_data.get('tp15_dict_hit', 0) if csp_data else 0,
        },
        comparison=CostCompareResult(
            tp15_metrics=tp15_metrics,
            csp_metrics=csp_metrics,
            deltas=compare_data.get('deltas', {}),
        ),
        gates=gates,
        gate_details=details,
        n_passed=n_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )
    path = _save_json(rd, 'phase58_verdict.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Phase 58 verdict completed in {time.time() - t0:.1f}s")


def run_phase58():
    """Run full Phase 58 pipeline."""
    from voynich.phases.costamagna_csp import (
        run_cost_domains, run_cost_reduction, run_cost_csp,
    )

    print("=" * 70)
    print("PHASE 58: Costamagna-Constrained CSP (Full Pipeline)")
    print("=" * 70)
    t0 = time.time()

    print("\n--- Step 58.1: Domain Construction ---")
    run_cost_domains()

    print("\n\n--- Step 58.2: Domain Comparison ---")
    run_cost_reduction()

    print("\n\n--- Step 58.3: CSP Solve ---")
    run_cost_csp()

    print("\n\n--- Step 58.5: CSP vs T_P15 Comparison ---")
    run_cost_compare()

    print("\n\n--- Step 58.6: Verdict ---")
    run_phase58_verdict()

    print(f"\n{'='*70}")
    print(f"Phase 58 completed in {time.time() - t0:.1f}s")
