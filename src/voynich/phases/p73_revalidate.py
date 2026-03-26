"""
Phase 73, Track 1: Re-Validate Clean Core (Mandatory Gate)
=============================================================
Re-run Phase 69 Track 0's three permutation tests under the corrected
decode model (connector→null):

  Test 0A: CV permutation — shuffle confirmed triple → syllable mappings
  Test 0B: Coda permutation — shuffle coda stroke → consonant mappings
           (connector always null, only hook/descender/sigmoid/vertical shuffled)
  Test 0C: Signal coherence — combined linguistic coherence test

Phase 69 results: 0A p=0.092, 0B p=0.318, 0C p=0.006.
Prediction: removing spurious -r from connectors improves 0A and 0B.

Dependency chain:
    results/p73_redecode.json            (Step 0)
    results/p69_clean_corpus.json        (Phase 69, for comparison)
    results/p69_clean_validation.json    (Phase 69, for comparison)
    results/combined_refine.json         (Phase 15)
    results/triple_tiers.json            (Phase 28/53)
    results/modifier_integrate.json      (Phase 16)
        -> results/p73_revalidate.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
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
    _reconstruct_modifier_rules,
)
from voynich.phases.p69_clean_corpus import _classify_token_confidence
from voynich.phases.p69_clean_validation import (
    _get_confirmed_and_unresolved,
    _precompute_clean_blueprints,
    _fast_decode_from_blueprint,
    _fast_decode_from_blueprint_custom_codas,
    _score_coherence_criteria,
)
from voynich.phases.p72_connector import _build_coda_table_with_connector


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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RevalidateResult:
    phase: str = "73"
    step: str = "73.1"
    experiment: str = "revalidate_null_connector"
    n_clean_tokens: int = 0
    n_trials: int = 0
    # Test 0A: CV permutation
    test_0a_real_dict_hit: float = 0.0
    test_0a_null_mean: float = 0.0
    test_0a_null_std: float = 0.0
    test_0a_z: float = 0.0
    test_0a_p: float = 1.0
    test_0a_real_signal: int = 0
    test_0a_null_mean_signal: float = 0.0
    gate_0a: bool = False
    # Test 0B: Coda permutation (connector always null)
    test_0b_real_dict_hit: float = 0.0
    test_0b_null_mean: float = 0.0
    test_0b_null_std: float = 0.0
    test_0b_z: float = 0.0
    test_0b_p: float = 1.0
    gate_0b: bool = False
    # Test 0C: Coherence
    test_0c_real_coherence: bool = False
    test_0c_real_n_signal: int = 0
    test_0c_real_n_content: int = 0
    test_0c_real_n_pharma: int = 0
    test_0c_n_random_all_pass: int = 0
    test_0c_p: float = 1.0
    gate_0c: bool = False
    # Comparison with Phase 69
    old_test_0a_p: float = 0.0
    old_test_0b_p: float = 0.0
    old_test_0c_p: float = 0.0
    # Overall
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "FAILED"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_revalidate() -> RevalidateResult:
    """Track 1: Re-validate clean subset with corrected decode (connector→null)."""
    t0 = time.time()
    rd = str(_results_dir())
    N_TRIALS = 1000

    print("Phase 73.1 — Re-Validate Clean Core (Connector→Null)")
    print("=" * 55)

    # --- Load old Phase 69 results for comparison ---
    old_validation = _safe_load(os.path.join(rd, 'p69_clean_validation.json'))
    old_0a_p = old_validation.get('test_0a_p', 0.092)
    old_0b_p = old_validation.get('test_0b_p', 0.318)
    old_0c_p = old_validation.get('test_0c_p', 0.006)

    # --- Load assignment data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = sorted(confirmed.keys())
    full_assignment = {**confirmed, **unresolved}
    inventory = sorted(set(confirmed.values()))
    inventory_arr = np.array(inventory)
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Syllable inventory: {len(inventory)} ({', '.join(inventory)})")

    # --- Load corpus and build dictionaries ---
    eva_to_triple = build_eva_to_triple_lookup()
    corrected_coda = _build_coda_table_with_connector('')  # THE KEY CHANGE

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"  Dictionary size: {len(ref_word_set)}")

    # --- Rebuild clean partition under corrected model ---
    confirmed_key_set = set(confirmed.keys())
    clean_indices = []
    for i, token in enumerate(all_tokens):
        n_conf, n_coda, n_unres = _classify_token_confidence(
            token, eva_to_triple, confirmed_key_set, corrected_coda)
        if n_unres == 0 and (n_conf + n_coda) > 0:
            clean_indices.append(i)

    n_clean = len(clean_indices)
    clean_tokens = [all_tokens[i] for i in clean_indices]
    print(f"  Clean tokens (corrected model): {n_clean}")

    # --- Build null corpora ---
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_token_lists = []
    for seed in range(5):
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths,
            n_tokens=n_clean, seed=seed + 42)
        null_token_lists.append(null_tokens)

    # --- Precompute blueprints (with corrected coda table) ---
    print("\n  Precomputing clean token blueprints (corrected model)...")
    clean_blueprints = _precompute_clean_blueprints(
        clean_tokens, eva_to_triple, corrected_coda)

    null_blueprints = [
        _precompute_clean_blueprints(nt, eva_to_triple, corrected_coda)
        for nt in null_token_lists
    ]

    # ===================================================================
    # Test 0A: CV Permutation on Clean Subset
    # ===================================================================
    print(f"\n  Test 0A: CV Permutation ({N_TRIALS} trials)...")

    real_decoded = _fast_decode_from_blueprint(clean_blueprints, full_assignment)
    real_dict_hits = sum(1 for d in real_decoded if d and '?' not in d and d in ref_word_set)
    real_dict_hit = real_dict_hits / n_clean if n_clean else 0.0

    real_null_decoded = [_fast_decode_from_blueprint(nbp, full_assignment)
                        for nbp in null_blueprints]
    real_null_counters = [Counter(nd) for nd in real_null_decoded]
    real_signal = _fast_signal_words(real_decoded, real_null_counters, ref_word_set)
    real_n_signal = len(real_signal)

    print(f"    Real dict hit: {real_dict_hit:.3f} ({real_dict_hits}/{n_clean})")
    print(f"    Real signal words: {real_n_signal}")

    random_dict_hits_0a: List[float] = []
    random_signals_0a: List[int] = []
    progress_step = max(1, N_TRIALS // 10)

    for trial in range(N_TRIALS):
        if trial % progress_step == 0:
            print(f"    Trial {trial}/{N_TRIALS}", flush=True)

        rng = np.random.default_rng(seed=trial + 10000)
        random_syls = rng.choice(inventory_arr, size=len(confirmed_keys), replace=True)
        random_confirmed = dict(zip(confirmed_keys, random_syls.tolist()))
        random_assignment = {**random_confirmed, **unresolved}

        trial_decoded = _fast_decode_from_blueprint(clean_blueprints, random_assignment)
        trial_hits = sum(1 for d in trial_decoded
                        if d and '?' not in d and d in ref_word_set)
        trial_dict_hit = trial_hits / n_clean if n_clean else 0.0
        random_dict_hits_0a.append(trial_dict_hit)

        trial_null_decoded = [_fast_decode_from_blueprint(nbp, random_assignment)
                             for nbp in null_blueprints]
        trial_null_counters = [Counter(nd) for nd in trial_null_decoded]
        trial_signal = _fast_signal_words(trial_decoded, trial_null_counters, ref_word_set)
        random_signals_0a.append(len(trial_signal))

    null_mean_0a = float(np.mean(random_dict_hits_0a))
    null_std_0a = float(np.std(random_dict_hits_0a))
    z_0a = (real_dict_hit - null_mean_0a) / null_std_0a if null_std_0a > 0 else 0.0
    p_0a = sum(1 for r in random_dict_hits_0a if r >= real_dict_hit) / N_TRIALS
    null_mean_signal_0a = float(np.mean(random_signals_0a))
    gate_0a = p_0a < 0.05

    print(f"    p(dict_hit): {p_0a:.4f} (z={z_0a:.2f}, "
          f"real={real_dict_hit:.3f}, null={null_mean_0a:.3f}±{null_std_0a:.3f})")
    print(f"    Signal: real={real_n_signal}, null mean={null_mean_signal_0a:.1f}")
    print(f"    Gate 0A: {'PASS' if gate_0a else 'FAIL'} (was p={old_0a_p})")

    # ===================================================================
    # Test 0B: Coda Permutation — connector ALWAYS null
    # ===================================================================
    print(f"\n  Test 0B: Coda Permutation ({N_TRIALS} trials)...")
    print("    Note: connector always null; only hook/descender/sigmoid/vertical shuffled")

    CODA_CONSONANTS = ['l', 'm', 'n', 'r', 's', 't']
    STROKE_GROUPS = ['hook', 'descender', 'sigmoid', 'vertical']  # NOT connector

    random_dict_hits_0b: List[float] = []

    for trial in range(N_TRIALS):
        if trial % progress_step == 0:
            print(f"    Trial {trial}/{N_TRIALS}", flush=True)

        rng = np.random.default_rng(seed=trial + 20000)

        # Build random coda table — connector always null
        random_coda_table = _build_coda_table_with_connector('')
        for group in STROKE_GROUPS:
            random_coda_table.stroke_to_coda[group] = rng.choice(CODA_CONSONANTS)

        trial_decoded = _fast_decode_from_blueprint_custom_codas(
            clean_blueprints, full_assignment, clean_tokens,
            eva_to_triple, random_coda_table)

        trial_hits = sum(1 for d in trial_decoded
                        if d and '?' not in d and d in ref_word_set)
        trial_dict_hit = trial_hits / n_clean if n_clean else 0.0
        random_dict_hits_0b.append(trial_dict_hit)

    null_mean_0b = float(np.mean(random_dict_hits_0b))
    null_std_0b = float(np.std(random_dict_hits_0b))
    z_0b = (real_dict_hit - null_mean_0b) / null_std_0b if null_std_0b > 0 else 0.0
    p_0b = sum(1 for r in random_dict_hits_0b if r >= real_dict_hit) / N_TRIALS
    gate_0b = p_0b < 0.05

    print(f"    p(dict_hit): {p_0b:.4f} (z={z_0b:.2f}, "
          f"real={real_dict_hit:.3f}, null={null_mean_0b:.3f}±{null_std_0b:.3f})")
    print(f"    Gate 0B: {'PASS' if gate_0b else 'FAIL'} (was p={old_0b_p})")

    # ===================================================================
    # Test 0C: Coherence on Clean Subset
    # ===================================================================
    print(f"\n  Test 0C: Coherence Test ({N_TRIALS} trials)...")

    real_coherence = _score_coherence_criteria(real_signal)
    print(f"    Real coherence: {real_coherence}")

    n_random_all_pass = 0
    for trial in range(N_TRIALS):
        if trial % progress_step == 0:
            print(f"    Trial {trial}/{N_TRIALS}", flush=True)

        rng = np.random.default_rng(seed=trial + 30000)
        random_syls = rng.choice(inventory_arr, size=len(confirmed_keys), replace=True)
        random_confirmed = dict(zip(confirmed_keys, random_syls.tolist()))
        random_assignment = {**random_confirmed, **unresolved}

        trial_decoded = _fast_decode_from_blueprint(clean_blueprints, random_assignment)
        trial_null_decoded = [_fast_decode_from_blueprint(nbp, random_assignment)
                             for nbp in null_blueprints]
        trial_null_counters = [Counter(nd) for nd in trial_null_decoded]
        trial_signal = _fast_signal_words(trial_decoded, trial_null_counters, ref_word_set)

        trial_coherence = _score_coherence_criteria(trial_signal)
        if trial_coherence['all_pass']:
            n_random_all_pass += 1

    p_0c = n_random_all_pass / N_TRIALS
    gate_0c = p_0c < 0.05

    print(f"    Random tables with full coherence: {n_random_all_pass}/{N_TRIALS}")
    print(f"    p(coherence): {p_0c:.4f}")
    print(f"    Gate 0C: {'PASS' if gate_0c else 'FAIL'} (was p={old_0c_p})")

    # ===================================================================
    # Overall verdict
    # ===================================================================
    gates_passed = sum([gate_0a, gate_0b, gate_0c])

    if gates_passed == 3:
        verdict = 'VALIDATED'
    elif gates_passed >= 1:
        verdict = 'PARTIAL'
    else:
        verdict = 'FAILED'

    result = RevalidateResult(
        n_clean_tokens=n_clean,
        n_trials=N_TRIALS,
        # Test 0A
        test_0a_real_dict_hit=round(real_dict_hit, 4),
        test_0a_null_mean=round(null_mean_0a, 4),
        test_0a_null_std=round(null_std_0a, 4),
        test_0a_z=round(z_0a, 2),
        test_0a_p=round(p_0a, 4),
        test_0a_real_signal=real_n_signal,
        test_0a_null_mean_signal=round(null_mean_signal_0a, 1),
        gate_0a=gate_0a,
        # Test 0B
        test_0b_real_dict_hit=round(real_dict_hit, 4),
        test_0b_null_mean=round(null_mean_0b, 4),
        test_0b_null_std=round(null_std_0b, 4),
        test_0b_z=round(z_0b, 2),
        test_0b_p=round(p_0b, 4),
        gate_0b=gate_0b,
        # Test 0C
        test_0c_real_coherence=real_coherence['all_pass'],
        test_0c_real_n_signal=real_n_signal,
        test_0c_real_n_content=real_coherence['n_content'],
        test_0c_real_n_pharma=real_coherence['n_pharma'],
        test_0c_n_random_all_pass=n_random_all_pass,
        test_0c_p=round(p_0c, 4),
        gate_0c=gate_0c,
        # Comparison
        old_test_0a_p=old_0a_p,
        old_test_0b_p=old_0b_p,
        old_test_0c_p=old_0c_p,
        # Overall
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p73_revalidate.json', asdict(result))

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Clean tokens: {n_clean}")
    print(f"  0A (CV perm):  p={p_0a:.4f} {'PASS' if gate_0a else 'FAIL'} "
          f"(was p={old_0a_p})")
    print(f"  0B (Coda perm): p={p_0b:.4f} {'PASS' if gate_0b else 'FAIL'} "
          f"(was p={old_0b_p})")
    print(f"  0C (Coherence): p={p_0c:.4f} {'PASS' if gate_0c else 'FAIL'} "
          f"(was p={old_0c_p})")
    print(f"  Gates: {gates_passed}/3 → {verdict}")
    print(f"  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
