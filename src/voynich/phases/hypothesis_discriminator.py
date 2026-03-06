"""
Phase 18.6 – Hypothesis Discriminator
=======================================

Aggregates the five independent diagnostic tests (Phase 18.1–18.5) into
a final weighted verdict on which of the three hypotheses best explains
the Voynich manuscript:

  H1  Procedural Hoax (Cardan Grille / table-lookup)
  H2  Verbose State-Machine Cipher
  H3  Taxonomic / Philosophical Language

Each test has a designed discriminative strength per hypothesis.  The
discriminator combines them with weighted aggregation, then builds a
human-readable chain of evidence.

Dependency chain:
    burstiness_test.json   (Phase 18.1)
    stride_entropy.json    (Phase 18.2)
    trie_topology.json     (Phase 18.3)
    hmm_pos_induction.json (Phase 18.4)
    lz_complexity.json     (Phase 18.5)
        -> hypothesis_discriminator.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# JSON serialiser
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HypothesisDiscriminatorResult:
    tests_loaded: List[str]
    tests_missing: List[str]
    test_verdicts: Dict[str, str]
    test_support_scores: Dict[str, Dict[str, float]]
    h1_weighted_evidence: float
    h2_weighted_evidence: float
    h3_weighted_evidence: float
    winning_hypothesis: str          # H1 / H2 / H3 / INDETERMINATE
    confidence: float
    reasoning: List[str]
    burstiness_mean_cv: Optional[float]
    stride_floor_collapse: Optional[bool]
    trie_colless_index: Optional[float]
    hmm_transition_entropy: Optional[float]
    lz_asymptotic_ratio: Optional[float]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Per-test discriminative weights (higher = more relevant)
# ---------------------------------------------------------------------------

# { test_name: { 'H1': weight, 'H2': weight, 'H3': weight } }
TEST_WEIGHTS = {
    'burstiness_test': {'H1': 1.5, 'H2': 1.0, 'H3': 0.8},
    'stride_entropy':  {'H1': 0.8, 'H2': 2.0, 'H3': 0.5},
    'trie_topology':   {'H1': 0.8, 'H2': 0.5, 'H3': 2.0},
    'hmm_pos_induction': {'H1': 1.2, 'H2': 1.0, 'H3': 1.0},
    'lz_complexity':   {'H1': 1.0, 'H2': 1.2, 'H3': 1.5},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_test_result(rd: str, filename: str) -> Optional[Dict]:
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _extract_support(result: Dict) -> Dict[str, float]:
    """Pull out hypothesis_support dict, default to uniform if missing."""
    hs = result.get('hypothesis_support', {})
    return {
        'H1': float(hs.get('H1', 0.333)),
        'H2': float(hs.get('H2', 0.333)),
        'H3': float(hs.get('H3', 0.333)),
    }


def _weighted_aggregate(
    supports: Dict[str, Dict[str, float]],
    weights: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Weighted combination of per-test hypothesis support scores."""
    agg = {'H1': 0.0, 'H2': 0.0, 'H3': 0.0}
    total_w = {'H1': 0.0, 'H2': 0.0, 'H3': 0.0}

    for test_name, sup in supports.items():
        w = weights.get(test_name, {'H1': 1.0, 'H2': 1.0, 'H3': 1.0})
        for h in ('H1', 'H2', 'H3'):
            agg[h] += sup[h] * w[h]
            total_w[h] += w[h]

    for h in ('H1', 'H2', 'H3'):
        if total_w[h] > 0:
            agg[h] /= total_w[h]

    return {k: round(v, 4) for k, v in agg.items()}


def _build_reasoning(
    test_results: Dict[str, Optional[Dict]],
) -> List[str]:
    """Build a human-readable chain of evidence from individual test results."""
    reasoning: List[str] = []

    # Test 1 — Burstiness
    r = test_results.get('burstiness_test')
    if r:
        cv = r.get('voynich_mean_cv', 0)
        if cv < 1.2:
            reasoning.append(
                f"Burstiness: mean CV = {cv:.3f} (near-Poisson) — consistent with "
                "procedural generation (H1).")
        else:
            reasoning.append(
                f"Burstiness: mean CV = {cv:.3f} (bursty) — topical clustering detected, "
                "inconsistent with H1.")

    # Test 2 — Stride Entropy
    r = test_results.get('stride_entropy')
    if r:
        collapse = r.get('floor_collapse_found', False)
        if collapse:
            strides = r.get('floor_collapse_strides', [])
            reasoning.append(
                f"Stride entropy: floor collapse at stride(s) {strides} — strong H2 "
                "(verbose cipher) evidence.")
        else:
            reasoning.append(
                "Stride entropy: no floor collapse at any stride — H2 not supported "
                "by decimation.")

    # Test 3 — Trie Topology
    r = test_results.get('trie_topology')
    if r:
        colless = r.get('trie_colless_index', 0)
        if colless < 0.25:
            reasoning.append(
                f"Trie topology: Colless = {colless:.4f} (very balanced) — consistent "
                "with engineered taxonomic vocabulary (H3).")
        else:
            reasoning.append(
                f"Trie topology: Colless = {colless:.4f} (imbalanced) — organic "
                "vocabulary structure, inconsistent with H3.")

    # Test 4 — HMM
    r = test_results.get('hmm_pos_induction')
    if r:
        ent = r.get('transition_entropy_mean', 0)
        if ent < 1.2:
            reasoning.append(
                f"HMM transitions: entropy = {ent:.3f} (rigid) — consistent with "
                "deterministic table generation (H1).")
        else:
            reasoning.append(
                f"HMM transitions: entropy = {ent:.3f} (complex) — grammar-like "
                "structure, inconsistent with H1.")

    # Test 5 — LZ Complexity
    r = test_results.get('lz_complexity')
    if r:
        vs_cardan = r.get('voynich_vs_cardan_ratio')
        vs_latin = r.get('voynich_vs_latin_ratio')
        if vs_cardan is not None and abs(vs_cardan - 1.0) < 0.15:
            reasoning.append(
                f"LZ complexity: Voynich/Cardan ratio = {vs_cardan:.3f} — compression "
                "profile matches hoax null (H1).")
        elif vs_latin is not None and abs(vs_latin - 1.0) < 0.2:
            reasoning.append(
                f"LZ complexity: Voynich/Latin ratio = {vs_latin:.3f} — compression "
                "matches natural language (H2).")
        else:
            reasoning.append(
                f"LZ complexity: Voynich/Cardan = {vs_cardan}, Voynich/Latin = {vs_latin} — "
                "no clear match.")

    return reasoning


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_hypothesis_discriminator() -> None:
    """Phase 18.6: final hypothesis discrimination."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 18.6: Hypothesis Discriminator")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all test results ──────────────────────────────────────
    test_files = {
        'burstiness_test': 'burstiness_test.json',
        'stride_entropy': 'stride_entropy.json',
        'trie_topology': 'trie_topology.json',
        'hmm_pos_induction': 'hmm_pos_induction.json',
        'lz_complexity': 'lz_complexity.json',
    }

    test_results: Dict[str, Optional[Dict]] = {}
    loaded: List[str] = []
    missing: List[str] = []

    print("\n  1. Loading test results …")
    for name, fname in test_files.items():
        r = _load_test_result(rd, fname)
        test_results[name] = r
        if r is not None:
            loaded.append(name)
            print(f"     ✓ {fname}")
        else:
            missing.append(name)
            print(f"     ✗ {fname} — MISSING")

    if not loaded:
        print("\n  [SKIP] No test results found — run individual tests first.")
        return

    # ── 2. Extract per-test support ───────────────────────────────────
    print(f"\n  2. Aggregating {len(loaded)} test(s) …")
    supports: Dict[str, Dict[str, float]] = {}
    verdicts: Dict[str, str] = {}
    for name in loaded:
        supports[name] = _extract_support(test_results[name])
        verdicts[name] = test_results[name].get('verdict', '')

    agg = _weighted_aggregate(supports, TEST_WEIGHTS)
    print(f"     Weighted evidence:  H1={agg['H1']:.3f}  H2={agg['H2']:.3f}  H3={agg['H3']:.3f}")

    # ── 3. Determine winner ───────────────────────────────────────────
    sorted_h = sorted(agg.items(), key=lambda kv: -kv[1])
    top_h, top_score = sorted_h[0]
    runner_h, runner_score = sorted_h[1]

    confidence = (top_score - runner_score) / max(top_score, 0.01)
    confidence = round(min(max(confidence, 0.0), 1.0), 4)

    if confidence < 0.15:
        winner = 'INDETERMINATE'
    else:
        winner = top_h

    print(f"     Winner: {winner}  (confidence = {confidence:.3f})")

    # ── 4. Build reasoning chain ──────────────────────────────────────
    reasoning = _build_reasoning(test_results)
    reasoning.append(f"Aggregate: H1={agg['H1']:.3f}, H2={agg['H2']:.3f}, H3={agg['H3']:.3f}.")
    reasoning.append(f"Conclusion: {winner} (confidence {confidence:.2f}).")

    print("\n  3. Evidence chain:")
    for r in reasoning:
        print(f"     • {r}")

    # ── Extract key metrics for top-level fields ──────────────────────
    burst_cv = test_results.get('burstiness_test', {}).get('voynich_mean_cv') if test_results.get('burstiness_test') else None
    stride_collapse = test_results.get('stride_entropy', {}).get('floor_collapse_found') if test_results.get('stride_entropy') else None
    colless = test_results.get('trie_topology', {}).get('trie_colless_index') if test_results.get('trie_topology') else None
    hmm_ent = test_results.get('hmm_pos_induction', {}).get('transition_entropy_mean') if test_results.get('hmm_pos_induction') else None
    lz_ratio = test_results.get('lz_complexity', {}).get('voynich_asymptotic_zlib') if test_results.get('lz_complexity') else None

    # ── Verdict ───────────────────────────────────────────────────────
    hyp_labels = {
        'H1': 'Procedural Hoax',
        'H2': 'Verbose State-Machine Cipher',
        'H3': 'Taxonomic / Philosophical Language',
        'INDETERMINATE': 'Indeterminate',
    }
    if winner == 'INDETERMINATE':
        verdict = (f"INDETERMINATE: top two hypotheses ({sorted_h[0][0]} vs "
                   f"{sorted_h[1][0]}) are too close to discriminate "
                   f"(confidence = {confidence:.2f}).")
    else:
        verdict = (f"{winner} — {hyp_labels[winner]}: weighted evidence = "
                   f"{agg[winner]:.3f}, confidence = {confidence:.2f}. "
                   f"{len(loaded)}/5 tests completed.")

    print(f"\n  FINAL VERDICT: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────
    result = HypothesisDiscriminatorResult(
        tests_loaded=loaded,
        tests_missing=missing,
        test_verdicts=verdicts,
        test_support_scores=supports,
        h1_weighted_evidence=agg['H1'],
        h2_weighted_evidence=agg['H2'],
        h3_weighted_evidence=agg['H3'],
        winning_hypothesis=winner,
        confidence=confidence,
        reasoning=reasoning,
        burstiness_mean_cv=burst_cv,
        stride_floor_collapse=stride_collapse,
        trie_colless_index=colless,
        hmm_transition_entropy=hmm_ent,
        lz_asymptotic_ratio=lz_ratio,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'hypothesis_discriminator.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
