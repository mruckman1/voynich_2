"""
Phase 69, Track 0: Clean Subset Validation (Mandatory Gate)
=============================================================
Three permutation tests on the clean subset (22,823 tokens that use
only confirmed triples) to validate that the 12 confirmed assignments
produce significantly better results than random:

  Test 0A: CV permutation — shuffle confirmed triple → syllable mappings
  Test 0B: Coda permutation — shuffle coda stroke → consonant mappings
  Test 0C: Signal coherence — combined linguistic coherence test

ALL THREE must pass for VALIDATED; 1-2 = PARTIAL; 0 = FAILED.
If FAILED, Tracks 1-3 cannot proceed.

Dependency chain:
    results/p69_clean_corpus.json        (Step 0)
    results/combined_refine.json         (Phase 15)
    results/triple_tiers.json            (Phase 28/53)
    results/modifier_integrate.json      (Phase 16)
    results/null_corpus.json             (Phase 17)
        -> results/p69_clean_validation.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_corpus_cvc_v2,
    LATIN_ENDINGS,
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
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13)."""
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CleanValidationResult:
    phase: str = "69"
    step: str = "69.1"
    experiment: str = "clean_validation"
    n_clean_tokens: int = 0
    n_trials: int = 0
    # Test 0A: CV permutation on clean subset
    test_0a_real_dict_hit: float = 0.0
    test_0a_null_mean: float = 0.0
    test_0a_null_std: float = 0.0
    test_0a_z: float = 0.0
    test_0a_p: float = 1.0
    test_0a_real_signal: int = 0
    test_0a_null_mean_signal: float = 0.0
    gate_0a: bool = False
    # Test 0B: Coda permutation on clean subset
    test_0b_real_dict_hit: float = 0.0
    test_0b_null_mean: float = 0.0
    test_0b_null_std: float = 0.0
    test_0b_z: float = 0.0
    test_0b_p: float = 1.0
    gate_0b: bool = False
    # Test 0C: Coherence on clean subset
    test_0c_real_coherence: bool = False
    test_0c_real_n_signal: int = 0
    test_0c_real_n_content: int = 0
    test_0c_real_n_pharma: int = 0
    test_0c_n_random_all_pass: int = 0
    test_0c_p: float = 1.0
    gate_0c: bool = False
    # Overall
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "FAILED"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Blueprint-based fast decode (adapted from cvc_full_permutation.py)
# ---------------------------------------------------------------------------

def _precompute_clean_blueprints(
    clean_tokens: List[str],
    eva_to_triple: Dict[str, str],
    coda_table,
) -> List[List[Tuple[Optional[str], Optional[str]]]]:
    """Precompute (triple_key_or_None, coda_char_or_None) per char per token.

    Same as cvc_full_permutation._precompute_token_blueprints but applied
    to clean tokens only.
    """
    blueprints = []
    for token in clean_tokens:
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
    """Decode tokens using precomputed blueprints."""
    decoded = []
    for bp in blueprints:
        if not bp:
            decoded.append('?')
            continue

        parts: List[str] = []
        for triple_key, coda in bp:
            if triple_key is not None:
                syl = assignment.get(triple_key, '?')
                parts.append(syl)
            elif coda is not None:
                if parts:
                    parts[-1] = parts[-1] + coda
        decoded.append(''.join(parts))

    return decoded


def _fast_decode_from_blueprint_custom_codas(
    blueprints: List[List[Tuple[Optional[str], Optional[str]]]],
    assignment: Dict[str, str],
    clean_tokens: List[str],
    eva_to_triple: Dict[str, str],
    custom_coda_table,
) -> List[str]:
    """Decode with custom coda rules — must re-classify since coda assignments change."""
    decoded = []
    for token in clean_tokens:
        eva_chars = tokenize_eva_chars(token)
        if not eva_chars:
            decoded.append('?')
            continue

        classified = classify_token_chars_v2(eva_chars, custom_coda_table)
        parts: List[str] = []
        for role, char in classified:
            if role == 'SYLLABIC':
                triple = eva_to_triple.get(char)
                syl = assignment.get(triple, '?') if triple else '?'
                parts.append(syl)
            elif role == 'CODA_MARKER':
                coda = get_coda(char, custom_coda_table)
                if coda and parts:
                    parts[-1] = parts[-1] + coda
        decoded.append(''.join(parts))

    return decoded


# ---------------------------------------------------------------------------
# Coherence scoring (adapted from cvc_full_permutation._score_cv_coherence)
# ---------------------------------------------------------------------------

def _score_coherence_criteria(signal_words: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score 4 coherence criteria on signal words."""
    words = set(w['word'] for w in signal_words)

    n_verb = len(words & set(VERB_PARADIGM_WORDS))
    n_function = len(words & set(FUNCTION_KIT))
    n_pharma = len(words & set(PHARMA_REGISTER))
    n_content = len([w for w in signal_words
                     if w.get('real_count', 0) >= 3 and
                     w['word'] not in FUNCTION_KIT])

    has_verb = n_verb >= 2
    has_function = n_function >= 2
    has_pharma = n_pharma >= 1
    has_content = n_content >= 5

    all_pass = has_verb and has_function and has_pharma and has_content

    return {
        'n_verb': n_verb,
        'n_function': n_function,
        'n_pharma': n_pharma,
        'n_content': n_content,
        'has_verb': has_verb,
        'has_function': has_function,
        'has_pharma': has_pharma,
        'has_content': has_content,
        'all_pass': all_pass,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_validate_clean():
    """Track 0: Mandatory clean subset validation (3 permutation tests)."""
    t0 = time.time()
    rd = str(_results_dir())
    N_TRIALS = 1000

    print("Phase 69.1 — Clean Subset Validation (Mandatory Gate)")
    print("=" * 55)

    # --- Load clean corpus ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    if not clean_data:
        print("  ERROR: p69_clean_corpus.json not found. Run build-clean first.")
        return

    clean_indices = clean_data.get('clean_indices', [])
    n_clean = len(clean_indices)
    print(f"  Clean tokens: {n_clean}")

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
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"  Dictionary size: {len(ref_word_set)}")

    # Extract clean tokens
    clean_tokens = [all_tokens[i] for i in clean_indices]

    # Build null corpora for signal isolation (5 null corpora decoded with each table)
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_token_lists = []
    for seed in range(5):
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths,
            n_tokens=n_clean, seed=seed + 42)
        null_token_lists.append(null_tokens)

    # --- Precompute blueprints ---
    print("\n  Precomputing clean token blueprints...")
    clean_blueprints = _precompute_clean_blueprints(
        clean_tokens, eva_to_triple, coda_table)

    null_blueprints = [
        _precompute_clean_blueprints(nt, eva_to_triple, coda_table)
        for nt in null_token_lists
    ]

    # ===================================================================
    # Test 0A: CV Permutation on Clean Subset
    # ===================================================================
    print(f"\n  Test 0A: CV Permutation ({N_TRIALS} trials)...")

    # Real decode
    real_decoded = _fast_decode_from_blueprint(clean_blueprints, full_assignment)
    real_dict_hits = sum(1 for d in real_decoded if d and '?' not in d and d in ref_word_set)
    real_dict_hit = real_dict_hits / n_clean if n_clean else 0.0

    # Real signal words
    real_null_decoded = [_fast_decode_from_blueprint(nbp, full_assignment)
                        for nbp in null_blueprints]
    real_null_counters = [Counter(nd) for nd in real_null_decoded]
    real_signal = _fast_signal_words(real_decoded, real_null_counters, ref_word_set)
    real_n_signal = len(real_signal)

    print(f"    Real dict hit: {real_dict_hit:.3f} ({real_dict_hits}/{n_clean})")
    print(f"    Real signal words: {real_n_signal}")

    # Random trials
    random_dict_hits_0a: List[float] = []
    random_signals_0a: List[int] = []

    progress_step = max(1, N_TRIALS // 10)
    for trial in range(N_TRIALS):
        if trial % progress_step == 0:
            print(f"    Trial {trial}/{N_TRIALS}", flush=True)

        rng = np.random.default_rng(seed=trial + 10000)

        # Random CV assignment for confirmed triples only
        random_syls = rng.choice(inventory_arr, size=len(confirmed_keys), replace=True)
        random_confirmed = dict(zip(confirmed_keys, random_syls.tolist()))
        random_assignment = {**random_confirmed, **unresolved}

        trial_decoded = _fast_decode_from_blueprint(clean_blueprints, random_assignment)
        trial_hits = sum(1 for d in trial_decoded
                        if d and '?' not in d and d in ref_word_set)
        trial_dict_hit = trial_hits / n_clean if n_clean else 0.0
        random_dict_hits_0a.append(trial_dict_hit)

        # Signal words for this trial
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
    print(f"    Gate 0A: {'PASS' if gate_0a else 'FAIL'}")

    # ===================================================================
    # Test 0B: Coda Permutation on Clean Subset
    # ===================================================================
    print(f"\n  Test 0B: Coda Permutation ({N_TRIALS} trials)...")

    CODA_CONSONANTS = ['l', 'm', 'n', 'r', 's', 't']
    STROKE_GROUPS = ['hook', 'descender', 'sigmoid', 'vertical', 'connector']

    # Real dict hit already computed above

    random_dict_hits_0b: List[float] = []

    for trial in range(N_TRIALS):
        if trial % progress_step == 0:
            print(f"    Trial {trial}/{N_TRIALS}", flush=True)

        rng = np.random.default_rng(seed=trial + 20000)

        # Build random coda table
        random_coda_table = build_coda_table_v2()  # start from real
        for group in STROKE_GROUPS:
            random_coda_table.stroke_to_coda[group] = rng.choice(CODA_CONSONANTS)

        # Must re-decode with custom coda table (can't use blueprints since
        # coda assignments change)
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
    print(f"    Gate 0B: {'PASS' if gate_0b else 'FAIL'}")

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

        # Random CV table
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
    print(f"    Gate 0C: {'PASS' if gate_0c else 'FAIL'}")

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

    result = CleanValidationResult(
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
        # Overall
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p69_clean_validation.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Test 0A (CV perm):    p={p_0a:.4f} ({'PASS' if gate_0a else 'FAIL'})")
    print(f"  Test 0B (Coda perm):  p={p_0b:.4f} ({'PASS' if gate_0b else 'FAIL'})")
    print(f"  Test 0C (Coherence):  p={p_0c:.4f} ({'PASS' if gate_0c else 'FAIL'})")
    print(f"  Gates: {gates_passed}/3")
    print(f"  Verdict: {verdict}")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
