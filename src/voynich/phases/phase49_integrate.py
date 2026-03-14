"""
Phase 49 Integration: Novel Computational Approaches
======================================================
Combine results from five tracks (A–E) and produce final verdict.

Track A: External LM Lattice Decode
Track B: Fourier/Spectral Periodicity Analysis
Track C: Optimal Transport Language Identification
Track D: Spectral Graph Matching
Track E: RL Feasibility Assessment + Integration

Dependency chain:
    lm_decode.json              (49A.5)
    spectral_crossfolio.json    (49B.4)
    ot_langid.json              (49C.4)
    graph_verdict.json          (49D.4)
        -> phase49_combine.json     (49E.1)
        -> rl_assess.json           (49E.2)
        -> phase49_validate.json    (49E.3)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _load_json(rd: str, filename: str) -> Optional[Dict]:
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    test_id: str
    description: str
    metric: str
    threshold: str
    value: str
    passed: bool


@dataclass
class Phase49CombineResult:
    track_a_dict_hit_10k: float
    track_a_dict_hit_131k: float
    track_a_delta: float
    track_a_cc_bigrams: int
    track_b_n_periodic: int
    track_b_formula_boundaries: int
    track_c_top_language: str
    track_c_discriminates: bool
    track_d_top_language: str
    track_d_discriminates: bool
    langid_agreement: bool
    langid_consensus: str
    runtime_seconds: float


@dataclass
class RLAssessResult:
    lm_reward_variance: float
    lm_reward_dynamic_range: float
    reward_sharpness: str
    rl_feasible: bool
    rl_recommendation: str
    rl_dict_hit: Optional[float]
    rl_delta: Optional[float]
    runtime_seconds: float


@dataclass
class Phase49ValidateResult:
    validations: List[Dict]
    n_passed: int
    n_total: int
    gate_passed: bool
    phase49_verdict: str
    phase49_rationale: str
    best_dict_hit: float
    best_source: str
    progression_table: List[Dict]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 49E.1: Combine Track A-D
# ---------------------------------------------------------------------------

def run_phase49_combine() -> None:
    """Step 49E.1: Combine findings from Tracks A-D."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 49E.1: Combine Track A-D Findings")
    print("=" * 70)

    rd = _results_dir()

    # Track A
    lm_decode = _load_json(rd, 'lm_decode.json')
    a_dict_10k = lm_decode.get('dict_hit_rate_10k', 0.0) if lm_decode else 0.0
    a_dict_131k = lm_decode.get('dict_hit_rate_131k', 0.0) if lm_decode else 0.0
    a_delta = lm_decode.get('delta_vs_phase16', 0.0) if lm_decode else 0.0
    a_cc = lm_decode.get('cc_bigrams', 0) if lm_decode else 0

    print(f"\n  Track A (LM Decode):")
    print(f"    Dict-hit 10K: {a_dict_10k:.2%}")
    print(f"    Dict-hit 131K: {a_dict_131k:.2%}")
    print(f"    Delta vs Phase 16: {a_delta:+.4f}")
    print(f"    CC bigrams: {a_cc}")

    # Track B
    spectral = _load_json(rd, 'spectral_crossfolio.json')
    fft_data = _load_json(rd, 'spectral_fft.json')
    stft_data = _load_json(rd, 'spectral_stft.json')
    b_periodic = fft_data.get('n_periodic_folios', 0) if fft_data else 0
    b_boundaries = stft_data.get('n_formula_boundaries', 0) if stft_data else 0

    print(f"\n  Track B (Spectral):")
    print(f"    Periodic folios: {b_periodic}")
    print(f"    Formula boundaries: {b_boundaries}")

    # Track C
    ot_langid = _load_json(rd, 'ot_langid.json')
    c_top = ot_langid.get('top_language', 'unknown') if ot_langid else 'unknown'
    c_disc = ot_langid.get('discriminates_top2', False) if ot_langid else False

    print(f"\n  Track C (Optimal Transport):")
    print(f"    Top language: {c_top}")
    print(f"    Discriminates top-2: {c_disc}")

    # Track D
    graph_v = _load_json(rd, 'graph_verdict.json')
    d_top = graph_v.get('top_language', 'unknown') if graph_v else 'unknown'
    d_disc = graph_v.get('discriminates_top2', False) if graph_v else False

    print(f"\n  Track D (Graph Matching):")
    print(f"    Top language: {d_top}")
    print(f"    Discriminates top-2: {d_disc}")

    # Agreement
    agreement = c_top == d_top
    consensus = c_top if agreement else 'ambiguous'

    print(f"\n  Language ID Agreement: {agreement}")
    print(f"  Consensus language: {consensus}")

    result = Phase49CombineResult(
        track_a_dict_hit_10k=a_dict_10k,
        track_a_dict_hit_131k=a_dict_131k,
        track_a_delta=a_delta,
        track_a_cc_bigrams=a_cc,
        track_b_n_periodic=b_periodic,
        track_b_formula_boundaries=b_boundaries,
        track_c_top_language=c_top,
        track_c_discriminates=c_disc,
        track_d_top_language=d_top,
        track_d_discriminates=d_disc,
        langid_agreement=agreement,
        langid_consensus=consensus,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'phase49_combine.json', asdict(result))
    print(f"\n  Saved -> {out}")


# ---------------------------------------------------------------------------
# Step 49E.2: RL Feasibility Assessment
# ---------------------------------------------------------------------------

def run_rl_assess() -> None:
    """Step 49E.2: Assess whether WFST LM provides sharp enough reward for RL."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 49E.2: RL Feasibility Assessment")
    print("=" * 70)

    rd = _results_dir()

    # Load lattice and LM
    lattice_data = _load_json(rd, 'lm_lattice.json')
    lm_char_path = os.path.join(rd, 'lm_char5.pkl')
    lm_word_path = os.path.join(rd, 'lm_word3.pkl')

    if lattice_data is None:
        print("  [SKIP] No lattice data — Track A not run")
        result = RLAssessResult(
            lm_reward_variance=0.0,
            lm_reward_dynamic_range=0.0,
            reward_sharpness='UNKNOWN',
            rl_feasible=False,
            rl_recommendation='DEFER',
            rl_dict_hit=None,
            rl_delta=None,
            runtime_seconds=round(time.time() - t0, 2),
        )
        _save_json(rd, 'rl_assess.json', asdict(result))
        return

    lattice = lattice_data.get('lattice', {})

    # Load char LM
    char_lm = None
    if os.path.exists(lm_char_path):
        with open(lm_char_path, 'rb') as f:
            char_lm = pickle.load(f)

    if char_lm is None:
        print("  [SKIP] No char LM available")
        result = RLAssessResult(
            lm_reward_variance=0.0,
            lm_reward_dynamic_range=0.0,
            reward_sharpness='UNKNOWN',
            rl_feasible=False,
            rl_recommendation='DEFER',
            rl_dict_hit=None,
            rl_delta=None,
            runtime_seconds=round(time.time() - t0, 2),
        )
        _save_json(rd, 'rl_assess.json', asdict(result))
        return

    # Compute LM score variance across lattice alternatives
    from voynich.core.stats import cross_entropy_lm

    print("\n  Computing LM score variance across lattice alternatives...")

    score_ranges = []
    n_sampled = 0

    for idx_str, entries in lattice.items():
        if n_sampled >= 1000:
            break
        if len(entries) < 2:
            continue

        scores = []
        for word, prior in entries:
            if not word or len(word) < 2:
                continue
            text = '_' + word + '_'
            ce = cross_entropy_lm(text, char_lm, per_char=True)
            scores.append(-ce)  # negative CE = higher is better

        if len(scores) >= 2:
            score_ranges.append(max(scores) - min(scores))
            n_sampled += 1

    if score_ranges:
        variance = float(np.var(score_ranges))
        dynamic_range = float(np.mean(score_ranges))
    else:
        variance = 0.0
        dynamic_range = 0.0

    # Assess sharpness
    if dynamic_range > 2.0:
        sharpness = 'SHARP'
        feasible = True
        recommendation = 'IMPLEMENT'
    elif dynamic_range > 1.0:
        sharpness = 'MODERATE'
        feasible = True
        recommendation = 'IMPLEMENT'
    else:
        sharpness = 'FLAT'
        feasible = False
        recommendation = 'DEFER'

    print(f"\n  Sampled {n_sampled} tokens with alternatives")
    print(f"  Mean dynamic range: {dynamic_range:.4f}")
    print(f"  Variance: {variance:.4f}")
    print(f"  Sharpness: {sharpness}")
    print(f"  RL feasible: {feasible}")
    print(f"  Recommendation: {recommendation}")

    # If feasible, run simple REINFORCE
    rl_dict_hit = None
    rl_delta = None

    if feasible and n_sampled >= 100:
        print("\n  Running simple REINFORCE policy gradient (50 episodes)...")

        # Simple policy: softmax over lattice alternatives, reward = -char_CE
        # Train on first 1000 tokens with alternatives
        sb = _load_json(rd, 'signal_bigrams.json')
        token_decoded = sb.get('token_decoded', []) if sb else []

        # Build reference word set for evaluation
        from voynich.core.reference import load_reference_corpus
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        ref_words = set()
        for lang, texts in ref_corpus.texts.items():
            for t in texts:
                for w in t.tokens:
                    ref_words.add(w.lower())
        ref_10k = set(list(sorted(ref_words))[:10000])

        # Collect tokens with alternatives
        alt_indices = []
        for idx_str, entries in lattice.items():
            if len(entries) >= 2:
                alt_indices.append((int(idx_str), entries))
            if len(alt_indices) >= 500:
                break

        # REINFORCE: learn per-alternative preferences
        n_episodes = 50
        lr = 0.01
        best_words: Dict[int, str] = {}

        for idx, entries in alt_indices:
            n_alts = len(entries)
            logits = np.zeros(n_alts)

            for _ep in range(n_episodes):
                # Sample from softmax policy
                probs = np.exp(logits - np.max(logits))
                probs = probs / (probs.sum() + 1e-20)
                chosen = np.random.choice(n_alts, p=probs)

                word = entries[chosen][0]
                text = '_' + word + '_'
                reward = -cross_entropy_lm(text, char_lm, per_char=True)

                # REINFORCE update
                grad = -probs.copy()
                grad[chosen] += 1.0
                logits += lr * reward * grad

            # Pick best
            final_probs = np.exp(logits - np.max(logits))
            final_probs = final_probs / (final_probs.sum() + 1e-20)
            best_idx = int(np.argmax(final_probs))
            best_words[idx] = entries[best_idx][0]

        # Evaluate: count dict hits for RL-selected words
        rl_hits = sum(1 for w in best_words.values() if w.lower() in ref_10k)
        baseline_hits = sum(
            1 for idx, entries in alt_indices
            if entries[0][0].lower() in ref_10k
        )

        rl_dict_hit = rl_hits / max(len(alt_indices), 1)
        baseline_rate = baseline_hits / max(len(alt_indices), 1)
        rl_delta = rl_dict_hit - baseline_rate

        print(f"  RL dict-hit on alternatives: {rl_dict_hit:.4f}")
        print(f"  Baseline dict-hit: {baseline_rate:.4f}")
        print(f"  RL delta: {rl_delta:+.4f}")

    result = RLAssessResult(
        lm_reward_variance=round(variance, 6),
        lm_reward_dynamic_range=round(dynamic_range, 4),
        reward_sharpness=sharpness,
        rl_feasible=feasible,
        rl_recommendation=recommendation,
        rl_dict_hit=round(rl_dict_hit, 4) if rl_dict_hit is not None else None,
        rl_delta=round(rl_delta, 4) if rl_delta is not None else None,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'rl_assess.json', asdict(result))
    print(f"\n  Saved -> {out}")


# ---------------------------------------------------------------------------
# Step 49E.3: Validation Battery + Verdict
# ---------------------------------------------------------------------------

def run_phase49_validate() -> None:
    """Step 49E.3: Validation battery and final Phase 49 verdict."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 49E.3: Phase 49 Validation + Verdict")
    print("=" * 70)

    rd = _results_dir()

    # Load all results
    combine = _load_json(rd, 'phase49_combine.json')
    rl = _load_json(rd, 'rl_assess.json')
    lm_decode = _load_json(rd, 'lm_decode.json')
    lm_calibrate = _load_json(rd, 'lm_calibrate.json')
    spectral_fft = _load_json(rd, 'spectral_fft.json')
    ot_langid = _load_json(rd, 'ot_langid.json')
    graph_verdict = _load_json(rd, 'graph_verdict.json')

    if combine is None:
        print("  [ERROR] phase49_combine.json not found — run phase49-combine first")
        return

    # Extract metrics
    a_dict_10k = combine.get('track_a_dict_hit_10k', 0.0)
    a_dict_131k = combine.get('track_a_dict_hit_131k', 0.0)
    a_delta = combine.get('track_a_delta', 0.0)
    a_cc = combine.get('track_a_cc_bigrams', 0)
    b_periodic = combine.get('track_b_n_periodic', 0)
    c_top = combine.get('track_c_top_language', 'unknown')
    c_disc = combine.get('track_c_discriminates', False)
    d_top = combine.get('track_d_top_language', 'unknown')
    d_disc = combine.get('track_d_discriminates', False)
    agreement = combine.get('langid_agreement', False)

    # LM perplexity comparison
    lm_build = _load_json(rd, 'lm_build.json')
    baseline_perp = lm_build.get('combined_char_perplexity', 999.0) if lm_build else 999.0

    # Compute decoded corpus perplexity (from lm_viterbi, which has char CE)
    lm_viterbi = _load_json(rd, 'lm_viterbi.json')
    decode_ce = lm_viterbi.get('mean_char_ce', 999.0) if lm_viterbi else 999.0

    # ── Validation Battery ──
    print("\n  Validation Battery:")
    validations = []

    # V1: dict_hit_131k >= 43.6% (no regression from Phase 16)
    v1_pass = a_dict_131k >= 0.436
    validations.append(asdict(ValidationResult(
        test_id='V1', description='No regression: dict_hit_131k >= 43.6%',
        metric='dict_hit_131k', threshold='≥0.436',
        value=f'{a_dict_131k:.4f}', passed=v1_pass,
    )))

    # V2: dict_hit with 10K dict > 30%
    v2_pass = a_dict_10k > 0.30
    validations.append(asdict(ValidationResult(
        test_id='V2', description='Track A target: dict_hit_10k > 30%',
        metric='dict_hit_10k', threshold='>0.30',
        value=f'{a_dict_10k:.4f}', passed=v2_pass,
    )))

    # V3: CC bigrams >= 5
    v3_pass = a_cc >= 5
    validations.append(asdict(ValidationResult(
        test_id='V3', description='Track A target: CC bigrams >= 5',
        metric='cc_bigrams', threshold='≥5',
        value=str(a_cc), passed=v3_pass,
    )))

    # V4: >= 10 periodic folios
    v4_pass = b_periodic >= 10
    validations.append(asdict(ValidationResult(
        test_id='V4', description='Track B target: >= 10 periodic folios',
        metric='n_periodic_folios', threshold='≥10',
        value=str(b_periodic), passed=v4_pass,
    )))

    # V5: OT discriminates top-2 languages
    v5_pass = c_disc
    validations.append(asdict(ValidationResult(
        test_id='V5', description='Track C: OT discriminates top-2 languages',
        metric='discriminates_top2', threshold='True',
        value=str(c_disc), passed=v5_pass,
    )))

    # V6: Graph matching ranks Latin or Italian #1
    v6_pass = d_top in ('latin', 'italian')
    validations.append(asdict(ValidationResult(
        test_id='V6', description='Track D: Latin or Italian ranked #1',
        metric='top_language', threshold='latin or italian',
        value=d_top, passed=v6_pass,
    )))

    # V7: Tracks C and D agree on top language
    v7_pass = agreement
    validations.append(asdict(ValidationResult(
        test_id='V7', description='Tracks C & D agree on top language',
        metric='agreement', threshold='True',
        value=f'{c_top} vs {d_top}', passed=v7_pass,
    )))

    # V8: LM decode perplexity < baseline
    v8_pass = decode_ce < baseline_perp
    validations.append(asdict(ValidationResult(
        test_id='V8', description='LM decode perplexity < baseline',
        metric='char_cross_entropy', threshold=f'<{baseline_perp:.2f}',
        value=f'{decode_ce:.2f}', passed=v8_pass,
    )))

    n_pass = sum(1 for v in validations if v['passed'])
    n_total = len(validations)

    for v in validations:
        marker = '✓' if v['passed'] else '✗'
        print(f"    {marker} {v['test_id']}: {v['description']} — "
              f"{v['value']} {'PASS' if v['passed'] else 'FAIL'}")

    print(f"\n  Result: {n_pass}/{n_total}")

    # Gate
    gate_passed = n_pass >= 4
    print(f"  Gate (≥4/8): {'PASS' if gate_passed else 'FAIL'}")

    # Verdict
    if a_delta > 0.05 and a_cc >= 5:
        verdict = 'IMPROVEMENT'
        rationale = (f'LM decode improved dict-hit by {a_delta:+.2%} with '
                     f'{a_cc} CC bigrams')
    elif a_delta > 0.0 or b_periodic >= 10 or agreement:
        verdict = 'MARGINAL'
        parts = []
        if a_delta > 0.0:
            parts.append(f'dict-hit {a_delta:+.2%}')
        if b_periodic >= 10:
            parts.append(f'{b_periodic} periodic folios')
        if agreement:
            parts.append(f'language ID: {c_top}')
        rationale = 'Marginal progress: ' + ', '.join(parts)
    else:
        verdict = 'NO_IMPROVEMENT'
        rationale = 'No metric exceeded target thresholds'

    print(f"\n  Verdict: {verdict}")
    print(f"  Rationale: {rationale}")

    # Determine best dict-hit source
    best_dict_hit = max(a_dict_10k, 0.436)  # at least Phase 16 baseline
    best_source = 'Track A (LM decode)' if a_dict_10k > 0.30 else 'Phase 16 (baseline)'

    # Progression table
    progression = [
        {'phase': '16', 'dict_hit': '43.6%', 'notes': 'Full corpus baseline (T_P15)'},
        {'phase': '47', 'dict_hit': '43.6%', 'notes': 'Z-score audit, no change'},
        {'phase': '48', 'dict_hit': '43.6%', 'notes': 'CRIB_SUGGESTIVE, no change'},
        {'phase': '49', 'dict_hit': f'{a_dict_131k:.1%}',
         'notes': f'{verdict} — LM dict_10k={a_dict_10k:.1%}, '
                  f'periodic={b_periodic}, lang={c_top}'},
    ]

    print("\n  Progression:")
    for p in progression:
        print(f"    Phase {p['phase']}: {p['dict_hit']} — {p['notes']}")

    result = Phase49ValidateResult(
        validations=validations,
        n_passed=n_pass,
        n_total=n_total,
        gate_passed=gate_passed,
        phase49_verdict=verdict,
        phase49_rationale=rationale,
        best_dict_hit=best_dict_hit,
        best_source=best_source,
        progression_table=progression,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'phase49_validate.json', asdict(result))
    print(f"\n  Saved -> {out}")


# ---------------------------------------------------------------------------
# Track E runner
# ---------------------------------------------------------------------------

def run_track_e_49() -> None:
    """Run all Track E steps sequentially."""
    run_phase49_combine()
    run_rl_assess()
    run_phase49_validate()


# ---------------------------------------------------------------------------
# Full Phase 49 runner
# ---------------------------------------------------------------------------

def run_phase49() -> None:
    """Run full Phase 49 pipeline: all five tracks + integration."""
    from voynich.phases.lm_lattice_decode import run_track_a_49
    from voynich.phases.spectral_periodicity import run_track_b_49
    from voynich.phases.optimal_transport import run_track_c_49
    from voynich.phases.spectral_graph_match import run_track_d_49

    print("\n" + "█" * 70)
    print("  PHASE 49: Novel Computational Approaches")
    print("█" * 70)

    run_track_a_49()
    run_track_b_49()
    run_track_c_49()
    run_track_d_49()
    run_track_e_49()

    print("\n" + "█" * 70)
    print("  PHASE 49 COMPLETE")
    print("█" * 70)
