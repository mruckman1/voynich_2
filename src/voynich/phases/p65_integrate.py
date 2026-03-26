"""
Phase 65, Step 6: Integration — Method Comparison + Consensus + Verdict
========================================================================
Compare four word boundary discovery methods, compute consensus
boundaries, and determine overall verdict.

Dependency chain:
    results/p65_harris.json          (Step 65.2)
    results/p65_bayesian.json        (Step 65.3)
    results/p65_lm_segment.json      (Step 65.4)
    results/p65_recipe_segment.json  (Step 65.5)
    results/p65_decoded_stream.json  (Step 65.1)
        -> results/p65_integrate.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus


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
class Phase65IntegrateResult:
    phase: str = "65"
    step: str = "65.6"
    experiment: str = "word_boundary_integration"
    # Per-method summary
    method_summary: List[Dict] = field(default_factory=list)
    best_method: str = ""
    best_selectivity: float = 0.0
    # Consensus (section-level)
    consensus_n_boundaries: int = 0
    consensus_n_words: int = 0
    consensus_mean_word_length: float = 0.0
    consensus_dict_hit: float = 0.0
    consensus_null_dict_hit: float = 0.0
    consensus_selectivity: float = 0.0
    consensus_top_words: List[Dict] = field(default_factory=list)
    consensus_sample: str = ""
    # Agreement
    agreement_matrix: List[List[float]] = field(default_factory=list)
    mean_pairwise_agreement: float = 0.0
    # EVA baseline comparison
    eva_baseline_dict_hit: float = 0.0
    eva_baseline_selectivity: float = 0.0
    # Gates
    g1_methods_passing: bool = False
    g2_consensus_selectivity: bool = False
    g3_word_length: bool = False
    g4_agreement: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segment(stream: str, boundaries: List[int]) -> List[str]:
    if not boundaries:
        return [stream] if stream else []
    words = []
    prev = 0
    for b in sorted(set(boundaries)):
        if b > prev and b <= len(stream):
            words.append(stream[prev:b])
            prev = b
    if prev < len(stream):
        words.append(stream[prev:])
    return words


def _pairwise_agreement(
    bounds_a: Set[int], bounds_b: Set[int], tolerance: int = 1,
) -> float:
    """Fraction of boundaries in A that have a match in B within tolerance."""
    if not bounds_a or not bounds_b:
        return 0.0
    matched = 0
    for a in bounds_a:
        for off in range(-tolerance, tolerance + 1):
            if (a + off) in bounds_b:
                matched += 1
                break
    # Symmetric: average of both directions
    matched_b = 0
    for b in bounds_b:
        for off in range(-tolerance, tolerance + 1):
            if (b + off) in bounds_a:
                matched_b += 1
                break
    agreement_a = matched / len(bounds_a)
    agreement_b = matched_b / len(bounds_b)
    return (agreement_a + agreement_b) / 2.0


def _build_10k_dict() -> set:
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    freq = Counter(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                   if len(w) >= 2 and w.isalpha())
    return set(w for w, _ in freq.most_common(10000))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase65_verdict():
    """Phase 65.6: Integration and verdict."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 65, Step 6: Integration — Method Comparison + Verdict")
    print("=" * 70)

    # Load all method results
    harris = _safe_load(os.path.join(rd, 'p65_harris.json'))
    bayesian = _safe_load(os.path.join(rd, 'p65_bayesian.json'))
    lm = _safe_load(os.path.join(rd, 'p65_lm_segment.json'))
    recipes = _safe_load(os.path.join(rd, 'p65_recipe_segment.json'))
    stream_data = _safe_load(os.path.join(rd, 'p65_decoded_stream.json'))

    if not stream_data:
        print("  ERROR: Missing stream data.")
        return None

    section_texts = stream_data.get('section_stream_texts', {})
    full_text = stream_data.get('full_stream', {}).get('text', '')
    dictionary = _build_10k_dict()

    # Summarize each method
    print("\n  Method Summary:")
    print(f"  {'Method':<15} {'Dict Hit':>10} {'Selectivity':>12} {'Mean WL':>8} {'Gates':>6} {'Verdict':<20}")
    print(f"  {'-'*75}")

    method_names = ['harris', 'bayesian', 'lm', 'recipes']
    method_data = [harris, bayesian, lm, recipes]
    method_summary: List[Dict] = []

    for name, data in zip(method_names, method_data):
        if not data:
            method_summary.append({'name': name, 'available': False})
            print(f"  {name:<15} {'N/A':>10}")
            continue

        dict_hit = data.get('dict_hit_rate', data.get('mean_coverage', 0))
        sel = data.get('selectivity', 0)
        mean_wl = data.get('mean_word_length', data.get('mean_word_length', 0))
        gp = data.get('gates_passed', 0)
        verdict = data.get('verdict', 'N/A')

        method_summary.append({
            'name': name,
            'available': True,
            'dict_hit_rate': dict_hit,
            'selectivity': sel,
            'mean_word_length': mean_wl,
            'gates_passed': gp,
            'verdict': verdict,
            'latin_f1': data.get('latin_f1', data.get('latin_f1', 0)),
        })

        print(f"  {name:<15} {dict_hit:>10.4f} {sel:>12.3f}x {mean_wl:>8.2f} {gp:>6} {verdict:<20}")

    # Count methods passing
    methods_passing = sum(1 for m in method_summary
                         if m.get('available') and m.get('gates_passed', 0) >= 3)

    # Best method by selectivity
    available_methods = [m for m in method_summary if m.get('available')]
    if available_methods:
        best = max(available_methods, key=lambda m: m.get('selectivity', 0))
        best_method = best['name']
        best_selectivity = best.get('selectivity', 0)
    else:
        best_method = 'none'
        best_selectivity = 0.0

    # Consensus boundaries (section-level)
    print("\n  Computing consensus boundaries...")

    # Collect section-level boundaries from each method
    harris_sec_bounds = harris.get('section_boundaries', {}) if harris else {}
    bayesian_sec_bounds = bayesian.get('section_boundaries', {}) if bayesian else {}
    lm_sec_bounds = lm.get('section_boundaries', {}) if lm else {}

    consensus_all_words: List[str] = []
    consensus_total_boundaries = 0

    for section_key in section_texts:
        section_text = section_texts[section_key]
        if len(section_text) < 20:
            continue

        # Get boundaries from each method for this section
        h_bounds = set(harris_sec_bounds.get(section_key, []))
        b_bounds = set(bayesian_sec_bounds.get(section_key, []))
        l_bounds = set(lm_sec_bounds.get(section_key, []))

        # Consensus: positions where >= 2 of 3 methods agree (±1 tolerance)
        all_positions = h_bounds | b_bounds | l_bounds
        consensus_positions: Set[int] = set()

        for pos in all_positions:
            votes = 0
            for method_bounds in [h_bounds, b_bounds, l_bounds]:
                for off in range(-1, 2):
                    if (pos + off) in method_bounds:
                        votes += 1
                        break
            if votes >= 2:
                consensus_positions.add(pos)

        consensus_boundaries = sorted(consensus_positions)
        consensus_total_boundaries += len(consensus_boundaries)
        words = _segment(section_text, consensus_boundaries)
        consensus_all_words.extend(words)

    # Evaluate consensus
    if consensus_all_words:
        dict_hits = sum(1 for w in consensus_all_words if w in dictionary)
        consensus_dict_hit = dict_hits / len(consensus_all_words)
        word_lengths = [len(w) for w in consensus_all_words]
        consensus_mean_wl = float(np.mean(word_lengths))
    else:
        consensus_dict_hit = 0.0
        consensus_mean_wl = 0.0

    # Null for consensus
    rng = np.random.default_rng(42)
    null_rates = []
    for _ in range(50):
        rb = sorted(rng.choice(len(full_text), size=max(1, consensus_total_boundaries),
                               replace=False).tolist())
        rw = _segment(full_text, rb)
        null_rates.append(sum(1 for w in rw if w in dictionary) / len(rw) if rw else 0.0)
    null_mean = float(np.mean(null_rates))
    consensus_sel = consensus_dict_hit / null_mean if null_mean > 0 else float('inf')

    print(f"  Consensus: {len(consensus_all_words)} words, "
          f"dict hit {consensus_dict_hit:.4f}, sel {consensus_sel:.3f}x, "
          f"mean length {consensus_mean_wl:.1f}")

    # Pairwise agreement between methods
    print("\n  Pairwise agreement:")
    all_method_bounds = []
    method_labels = []
    for name, bounds_dict in [('harris', harris_sec_bounds),
                               ('bayesian', bayesian_sec_bounds),
                               ('lm', lm_sec_bounds)]:
        combined = set()
        for b_list in bounds_dict.values():
            combined.update(b_list)
        all_method_bounds.append(combined)
        method_labels.append(name)

    agreement_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                agreement_matrix[i, j] = _pairwise_agreement(
                    all_method_bounds[i], all_method_bounds[j])

    upper_tri = []
    for i in range(3):
        for j in range(i + 1, 3):
            agreement_matrix[i, j] = round(agreement_matrix[i, j], 4)
            agreement_matrix[j, i] = agreement_matrix[i, j]
            upper_tri.append(agreement_matrix[i, j])
            print(f"    {method_labels[i]} vs {method_labels[j]}: "
                  f"{agreement_matrix[i, j]:.3f}")
    mean_agreement = float(np.mean(upper_tri)) if upper_tri else 0.0
    print(f"  Mean pairwise agreement: {mean_agreement:.3f}")

    # EVA baseline: use original token boundaries as word boundaries
    full_boundaries = stream_data.get('full_stream', {}).get('token_boundaries', [])
    if full_boundaries:
        eva_words = _segment(full_text, full_boundaries)
        eva_dict_hits = sum(1 for w in eva_words if w in dictionary)
        eva_dict_rate = eva_dict_hits / len(eva_words) if eva_words else 0.0
        eva_null_rates = []
        for _ in range(50):
            rb = sorted(rng.choice(len(full_text), size=max(1, len(full_boundaries)),
                                   replace=False).tolist())
            rw = _segment(full_text, rb)
            eva_null_rates.append(sum(1 for w in rw if w in dictionary) / len(rw) if rw else 0.0)
        eva_null = float(np.mean(eva_null_rates))
        eva_sel = eva_dict_rate / eva_null if eva_null > 0 else float('inf')
        print(f"\n  EVA baseline: dict hit {eva_dict_rate:.4f}, sel {eva_sel:.3f}x")
    else:
        eva_dict_rate = 0.0
        eva_sel = 0.0

    # Top consensus words
    consensus_counter = Counter(consensus_all_words)
    consensus_top = [{'word': w, 'count': c, 'in_dict': w in dictionary}
                     for w, c in consensus_counter.most_common(30)]

    # Gates
    g1 = methods_passing >= 2
    g2 = consensus_sel > 1.5
    g3 = 3.5 <= consensus_mean_wl <= 7.0
    g4 = mean_agreement > 0.3
    gates_passed = sum([g1, g2, g3, g4])

    if consensus_sel > 2.0 and consensus_dict_hit > 0.15 and methods_passing >= 2:
        verdict = "WORD_BOUNDARIES_FOUND"
    elif consensus_sel > 1.5 or methods_passing >= 1:
        verdict = "PARTIAL_SEGMENTATION"
    else:
        verdict = "SEGMENTATION_FAILED"

    print(f"\n  Gates:")
    print(f"    G1 (>=2 methods pass):   {'PASS' if g1 else 'FAIL'} ({methods_passing})")
    print(f"    G2 (consensus sel>1.5):  {'PASS' if g2 else 'FAIL'} ({consensus_sel:.3f}x)")
    print(f"    G3 (word length 3.5-7):  {'PASS' if g3 else 'FAIL'} ({consensus_mean_wl:.1f})")
    print(f"    G4 (agreement>0.3):      {'PASS' if g4 else 'FAIL'} ({mean_agreement:.3f})")
    print(f"\n  VERDICT: {verdict} ({gates_passed}/4)")

    result = Phase65IntegrateResult(
        method_summary=method_summary,
        best_method=best_method,
        best_selectivity=round(best_selectivity, 3),
        consensus_n_boundaries=consensus_total_boundaries,
        consensus_n_words=len(consensus_all_words),
        consensus_mean_word_length=round(consensus_mean_wl, 2),
        consensus_dict_hit=round(consensus_dict_hit, 4),
        consensus_null_dict_hit=round(null_mean, 4),
        consensus_selectivity=round(consensus_sel, 3),
        consensus_top_words=consensus_top,
        consensus_sample=' '.join(consensus_all_words[:100])[:500],
        agreement_matrix=agreement_matrix.tolist(),
        mean_pairwise_agreement=round(mean_agreement, 4),
        eva_baseline_dict_hit=round(eva_dict_rate, 4),
        eva_baseline_selectivity=round(eva_sel, 3),
        g1_methods_passing=g1,
        g2_consensus_selectivity=g2,
        g3_word_length=g3,
        g4_agreement=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    _save_json(rd, 'p65_integrate.json', asdict(result))
    print(f"\n  Sample consensus: {result.consensus_sample[:200]}...")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result


def run_phase65():
    """Run full Phase 65 pipeline."""
    print("\n" + "=" * 70)
    print("PHASE 65: Word Boundary Discovery in the Decoded Stream")
    print("=" * 70)

    from voynich.phases.p65_decoded_stream import run_build_stream
    from voynich.phases.p65_harris import run_harris_segment
    from voynich.phases.p65_bayesian import run_bayesian_segment
    from voynich.phases.p65_lm_segment import run_lm_segment
    from voynich.phases.p65_recipe_segment import run_recipe_segment

    print("\n--- Step 65.1: Build Decoded Streams ---")
    run_build_stream()

    print("\n--- Step 65.2: Harris MI Boundaries ---")
    run_harris_segment()

    print("\n--- Step 65.3: Bayesian Word Segmentation ---")
    run_bayesian_segment()

    print("\n--- Step 65.4: LM Perplexity Minimization ---")
    run_lm_segment()

    print("\n--- Step 65.5: Recipe Template Segmentation ---")
    run_recipe_segment()

    print("\n--- Step 65.6: Integration + Verdict ---")
    result = run_phase65_verdict()

    print("\n" + "=" * 70)
    print("PHASE 65 COMPLETE")
    if result:
        print(f"  Verdict: {result.verdict}")
        print(f"  Best method: {result.best_method} ({result.best_selectivity:.3f}x)")
        print(f"  Consensus: dict hit {result.consensus_dict_hit:.4f}, "
              f"sel {result.consensus_selectivity:.3f}x")
    print("=" * 70)
    return result
