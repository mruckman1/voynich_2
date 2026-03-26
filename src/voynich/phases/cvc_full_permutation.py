"""
Phase 61, Track B: Full CV Permutation Test Under CVC Decode
==============================================================
Generates 1,000 random CV assignment tables, applies each with the FIXED
corrected coda rules (connector→r, i→syllabic), and measures signal word
count + coherence.  Produces the definitive CVC p-values directly
comparable to the paper's CV headline (Section 6.2): p=0.001 (count)
and p=0.011 (coherence).

Key distinction from existing tests:
  - reviewer_permutation.py: permutes CV table, CV decode
  - cvc_permutation.py (Phase 59): permutes coda table, CVC decode
  - THIS: permutes CV table, fixes corrected codas, CVC decode

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    results/null_corpus.json          (Phase 17)
        -> results/phase61_cvc_full_permutation.json
"""

import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_corpus_cvc_v2,
)
from voynich.phases.coda_markers import get_coda
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _load_shared_data,
)
from voynich.phases.cvc_permutation import (
    FUNCTION_KIT,
    PHARMA_REGISTER,
    VERB_PARADIGM_WORDS,
    _fast_signal_words,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
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
# CVC coherence criteria (recalibrated from Phase 60 Track B)
# ---------------------------------------------------------------------------

def _score_cv_coherence(signal_words: List[Dict[str, Any]]) -> bool:
    """Original paper Section 6.2 coherence: verb + function + pharma."""
    words = set(w['word'].lower() for w in signal_words)
    has_verb = len(words & VERB_PARADIGM_WORDS) >= 2
    has_function = len(words & FUNCTION_KIT) >= 3
    has_pharma = len(words & PHARMA_REGISTER) >= 1
    return has_verb and has_function and has_pharma


def _expanded_metrics(signal_words: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute expanded CVC coherence metrics for a signal word set."""
    words = set(w['word'].lower() for w in signal_words)

    # Content words (non-function, length >= 3)
    content = set(w for w in words if w not in FUNCTION_KIT and len(w) >= 3)

    # Latin ending diversity
    endings = set()
    latin_endings = {'en', 'in', 'an', 'on', 'un', 'er', 'ar', 'or',
                     'es', 'is', 'us', 'um', 'am'}
    for w in words:
        if len(w) >= 3:
            end2 = w[-2:]
            if end2 in latin_endings:
                endings.add(end2)

    # Pharma terms
    pharma = words & PHARMA_REGISTER

    return {
        'n_signal': len(signal_words),
        'n_content': len(content),
        'n_endings': len(endings),
        'n_pharma': len(pharma),
        'mean_word_len': (sum(len(w['word']) for w in signal_words)
                          / len(signal_words) if signal_words else 0.0),
    }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CvcFullPermResult:
    phase: str = "61"
    step: str = "61.2"
    experiment: str = "cvc_full_permutation"
    n_trials: int = 0
    # Real table results
    real_n_signal: int = 0
    real_mean_selectivity: float = 0.0
    real_cv_coherence: bool = False
    real_metrics: Dict[str, float] = field(default_factory=dict)
    # Random distribution
    random_mean_signal: float = 0.0
    random_std_signal: float = 0.0
    random_mean_selectivity: float = 0.0
    # P-values
    p_count: float = 0.0
    p_selectivity: float = 0.0
    p_cv_coherence: float = 0.0
    n_random_cv_coherence: int = 0
    # CVC coherence (calibrated thresholds)
    cvc_thresholds: Dict[str, float] = field(default_factory=dict)
    p_cvc_coherence: float = 0.0
    n_random_cvc_coherence: int = 0
    # Comparison to paper's CV results
    cv_p_count: float = 0.001
    cv_p_coherence: float = 0.011
    cvc_improved_count: bool = False
    cvc_improved_coherence: bool = False
    # Gates
    g1_count: bool = False          # p(count) < 0.01
    g2_cvc_coherence: bool = False  # p(CVC coherence) < 0.05
    g3_beats_cv: bool = False       # CVC p <= CV p = 0.011
    g4_selectivity: bool = False    # p(selectivity) < 0.05
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Fast precomputed CVC decode
# ---------------------------------------------------------------------------

def _precompute_token_blueprints(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    coda_table,
) -> List[List[Tuple[Optional[str], Optional[str]]]]:
    """Precompute (triple_key_or_None, coda_char_or_None) for each char in each token.

    This is the FIXED part of CVC decode — independent of the CV assignment.
    For each EVA char:
      - SYLLABIC: (triple_key, None) — needs assignment lookup at decode time
      - CODA_MARKER: (None, coda_consonant) — fixed string to append
    """
    blueprints = []
    for token in tokens:
        eva_chars = tokenize_eva_chars(token)
        if not eva_chars:
            blueprints.append([])
            continue

        classified = classify_token_chars_v2(eva_chars, coda_table)
        bp = []
        for role, char in classified:
            if role == 'SYLLABIC':
                triple = eva_to_triple.get(char)
                bp.append((triple, None))
            elif role == 'CODA_MARKER':
                coda = get_coda(char, coda_table)
                bp.append((None, coda))
        blueprints.append(bp)

    return blueprints


def _fast_decode_from_blueprint(
    blueprints: List[List[Tuple[Optional[str], Optional[str]]]],
    assignment: Dict[str, str],
) -> List[str]:
    """Decode tokens using precomputed blueprints — just dict lookups + string concat.

    ~10x faster than decode_corpus_cvc_v2 because tokenize_eva_chars() and
    classify_token_chars_v2() are already done.
    """
    decoded = []
    for bp in blueprints:
        if not bp:
            decoded.append('?')
            continue

        parts: List[str] = []
        for triple_key, coda in bp:
            if triple_key is not None:
                # SYLLABIC: look up assignment
                syl = assignment.get(triple_key, '?')
                parts.append(syl)
            elif coda is not None:
                # CODA_MARKER: append to last syllable
                if parts:
                    parts[-1] = parts[-1] + coda

        decoded.append(''.join(parts))

    return decoded


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cvc_full_perm():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("Phase 61, Track B: Full CV Permutation Test Under CVC Decode", flush=True)
    print("=" * 70, flush=True)

    rd = str(_results_dir())
    N_TRIALS = 1000

    # Load data
    print("\n  Loading shared data ...", flush=True)
    data = _load_shared_data()

    all_tokens = data['all_tokens']
    assignment = data['assignment']
    eva_to_triple = data['eva_to_triple']
    ref_word_set = data['ref_word_set']
    null_token_lists = data['null_token_lists']
    n_tokens = len(all_tokens)

    # Extract T_P15's triple names and syllable inventory
    triple_names = sorted(assignment.keys())
    inventory = sorted(set(assignment.values()))
    inventory_arr = np.array(inventory)
    print(f"  Triples: {len(triple_names)}, Syllable inventory: {len(inventory)}", flush=True)
    print(f"  Inventory: {', '.join(inventory)}", flush=True)

    # Build corrected coda table (FIXED across all trials)
    coda_table = build_coda_table_v2()
    print(f"  Coda table: {coda_table.stroke_to_coda}", flush=True)

    # -----------------------------------------------------------------------
    # Precompute blueprints (the BIG optimization)
    # -----------------------------------------------------------------------
    print("\n  Precomputing token blueprints (one-time cost) ...", flush=True)
    bp_t0 = time.time()
    real_blueprints = _precompute_token_blueprints(all_tokens, eva_to_triple, coda_table)
    null_blueprints = [
        _precompute_token_blueprints(nt, eva_to_triple, coda_table)
        for nt in null_token_lists
    ]
    bp_elapsed = time.time() - bp_t0
    print(f"  Blueprints precomputed in {bp_elapsed:.1f}s "
          f"(real: {len(real_blueprints)}, null: {len(null_blueprints)}x{len(null_blueprints[0]) if null_blueprints else 0})",
          flush=True)

    # -----------------------------------------------------------------------
    # Evaluate REAL table
    # -----------------------------------------------------------------------
    print("\n  1. Evaluating real T_P15 table with corrected CVC ...", flush=True)

    real_decoded = _fast_decode_from_blueprint(real_blueprints, assignment)
    null_decoded_list = [
        _fast_decode_from_blueprint(nbp, assignment)
        for nbp in null_blueprints
    ]
    null_counters = [Counter(nd) for nd in null_decoded_list]

    real_signal = _fast_signal_words(real_decoded, null_counters, ref_word_set)
    real_n_signal = len(real_signal)
    finite_sels = [w['selectivity'] for w in real_signal if w['selectivity'] < 900]
    real_mean_sel = sum(finite_sels) / len(finite_sels) if finite_sels else 0.0
    real_cv_coherence = _score_cv_coherence(real_signal)
    real_metrics = _expanded_metrics(real_signal)

    print(f"  Real signal words: {real_n_signal}", flush=True)
    print(f"  Real mean selectivity: {real_mean_sel:.2f}×", flush=True)
    print(f"  Real CV coherence: {real_cv_coherence}", flush=True)
    print(f"  Real metrics: {real_metrics}", flush=True)

    # -----------------------------------------------------------------------
    # Run 1000 random CV tables (FAST path: blueprint-based decode)
    # -----------------------------------------------------------------------
    print(f"\n  2. Running {N_TRIALS} random CV tables with fixed corrected codas ...", flush=True)

    random_signal_counts: List[int] = []
    random_mean_sels: List[float] = []
    random_cv_passes: List[bool] = []
    random_expanded: List[Dict[str, float]] = []

    progress_step = max(1, N_TRIALS // 10)
    for trial in range(N_TRIALS):
        if trial % progress_step == 0:
            elapsed = time.time() - t0
            rate = trial / elapsed if elapsed > 0 and trial > 0 else 0
            eta = (N_TRIALS - trial) / rate if rate > 0 else 0
            print(f"    Trial {trial}/{N_TRIALS} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

        # Random CV assignment
        rng = np.random.default_rng(seed=trial + 10000)  # offset to avoid collision
        random_syls = rng.choice(inventory_arr, size=len(triple_names), replace=True)
        random_assignment = dict(zip(triple_names, random_syls.tolist()))

        # FAST decode via precomputed blueprints
        trial_decoded = _fast_decode_from_blueprint(real_blueprints, random_assignment)
        trial_null_decoded = [
            _fast_decode_from_blueprint(nbp, random_assignment)
            for nbp in null_blueprints
        ]
        trial_null_counters = [Counter(nd) for nd in trial_null_decoded]

        # Signal isolation
        trial_signal = _fast_signal_words(trial_decoded, trial_null_counters, ref_word_set)
        trial_n_signal = len(trial_signal)
        trial_finite = [w['selectivity'] for w in trial_signal if w['selectivity'] < 900]
        trial_mean_sel = sum(trial_finite) / len(trial_finite) if trial_finite else 0.0

        random_signal_counts.append(trial_n_signal)
        random_mean_sels.append(trial_mean_sel)
        random_cv_passes.append(_score_cv_coherence(trial_signal))
        random_expanded.append(_expanded_metrics(trial_signal))

    elapsed = time.time() - t0
    print(f"  {N_TRIALS} trials completed in {elapsed:.0f}s", flush=True)

    # -----------------------------------------------------------------------
    # Compute p-values
    # -----------------------------------------------------------------------
    print("\n  3. Computing p-values ...")

    # Count test
    p_count = sum(1 for c in random_signal_counts if c >= real_n_signal) / N_TRIALS
    random_mean_n = np.mean(random_signal_counts)
    random_std_n = np.std(random_signal_counts)

    # Selectivity test
    p_sel = sum(1 for s in random_mean_sels if s >= real_mean_sel) / N_TRIALS
    random_mean_sel_val = np.mean(random_mean_sels)

    # CV coherence (original criteria)
    n_cv_pass = sum(1 for p in random_cv_passes if p)
    p_cv_coherence = n_cv_pass / N_TRIALS

    print(f"  p(count): {p_count:.4f} (real={real_n_signal}, "
          f"random mean={random_mean_n:.1f} ± {random_std_n:.1f})")
    print(f"  p(selectivity): {p_sel:.4f} (real={real_mean_sel:.2f}×, "
          f"random mean={random_mean_sel_val:.2f}×)")
    print(f"  p(CV coherence): {p_cv_coherence:.4f} "
          f"({n_cv_pass}/{N_TRIALS} pass)")

    # -----------------------------------------------------------------------
    # CVC coherence (calibrated thresholds at 92nd percentile)
    # -----------------------------------------------------------------------
    print("\n  4. Calibrating CVC coherence thresholds ...")

    # Collect expanded metric distributions
    metric_arrays = {
        'n_signal': [m['n_signal'] for m in random_expanded],
        'n_content': [m['n_content'] for m in random_expanded],
        'n_endings': [m['n_endings'] for m in random_expanded],
        'n_pharma': [m['n_pharma'] for m in random_expanded],
    }

    thresholds = {}
    for name, vals in metric_arrays.items():
        thr = float(np.percentile(vals, 92))
        pass_rate = sum(1 for v in vals if v >= thr) / len(vals)
        real_passes = real_metrics.get(name, 0) >= thr
        thresholds[name] = {
            'threshold': round(thr, 1),
            'pass_rate': round(pass_rate, 3),
            'real_value': real_metrics.get(name, 0),
            'real_passes': real_passes,
        }
        print(f"  {name}: threshold={thr:.1f}, "
              f"random pass={pass_rate:.1%}, "
              f"real={real_metrics.get(name, 0):.1f} "
              f"({'PASS' if real_passes else 'FAIL'})")

    # Joint CVC coherence: all 4 metrics must pass
    def _passes_cvc(metrics: Dict[str, float]) -> bool:
        for name in thresholds:
            if metrics.get(name, 0) < thresholds[name]['threshold']:
                return False
        return True

    n_cvc_pass = sum(1 for m in random_expanded if _passes_cvc(m))
    p_cvc_coherence = n_cvc_pass / N_TRIALS

    real_cvc_passes = _passes_cvc(real_metrics)
    print(f"\n  Joint CVC coherence: {n_cvc_pass}/{N_TRIALS} random pass = "
          f"p={p_cvc_coherence:.4f}")
    print(f"  Real passes CVC coherence: {real_cvc_passes}")

    # -----------------------------------------------------------------------
    # Comparison to paper
    # -----------------------------------------------------------------------
    print("\n  5. Comparison to paper's CV results ...")
    print(f"  CV p(count):     {0.001} → CVC p(count):     {p_count:.4f}")
    print(f"  CV p(coherence): {0.011} → CVC p(coherence): {p_cvc_coherence:.4f}")
    cvc_improved_count = p_count <= 0.001
    cvc_improved_coherence = p_cvc_coherence <= 0.011
    print(f"  Count improved:     {cvc_improved_count}")
    print(f"  Coherence improved: {cvc_improved_coherence}")

    # Gates
    g1 = p_count < 0.01
    g2 = p_cvc_coherence < 0.05
    g3 = p_cvc_coherence <= 0.011
    g4 = p_sel < 0.05
    gates = sum([g1, g2, g3, g4])

    result = CvcFullPermResult(
        n_trials=N_TRIALS,
        real_n_signal=real_n_signal,
        real_mean_selectivity=round(real_mean_sel, 2),
        real_cv_coherence=real_cv_coherence,
        real_metrics={k: round(v, 2) for k, v in real_metrics.items()},
        random_mean_signal=round(float(random_mean_n), 2),
        random_std_signal=round(float(random_std_n), 2),
        random_mean_selectivity=round(float(random_mean_sel_val), 2),
        p_count=round(p_count, 4),
        p_selectivity=round(p_sel, 4),
        p_cv_coherence=round(p_cv_coherence, 4),
        n_random_cv_coherence=n_cv_pass,
        cvc_thresholds={k: _convert(v) for k, v in thresholds.items()},
        p_cvc_coherence=round(p_cvc_coherence, 4),
        n_random_cvc_coherence=n_cvc_pass,
        cvc_improved_count=cvc_improved_count,
        cvc_improved_coherence=cvc_improved_coherence,
        g1_count=g1,
        g2_cvc_coherence=g2,
        g3_beats_cv=g3,
        g4_selectivity=g4,
        gates_passed=gates,
        gate_passed=gates >= 3,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'phase61_cvc_full_permutation.json', result)

    # Summary
    print("\n" + "=" * 70)
    print("  TRACK B SUMMARY: Full CV Permutation Under CVC Decode")
    print("=" * 70)
    print(f"  Trials:               {N_TRIALS}")
    print(f"  Real signal words:    {real_n_signal}")
    print(f"  Random mean signal:   {random_mean_n:.1f} ± {random_std_n:.1f}")
    print(f"  p(count):             {p_count:.4f}")
    print(f"  p(selectivity):       {p_sel:.4f}")
    print(f"  p(CV coherence):      {p_cv_coherence:.4f}")
    print(f"  p(CVC coherence):     {p_cvc_coherence:.4f}")
    print(f"  Paper CV p(count):    0.001")
    print(f"  Paper CV p(coher):    0.011")
    print(f"\n  Gates: {gates}/4 passed")
    print(f"    G1 (p_count<0.01):    {g1}")
    print(f"    G2 (p_cvc_coh<0.05):  {g2}")
    print(f"    G3 (beats CV 0.011):  {g3}")
    print(f"    G4 (p_sel<0.05):      {g4}")
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
