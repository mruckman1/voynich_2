"""
Reviewer Analysis 1: Random Syllabary Permutation Test
======================================================

Tests whether ANY random CV syllable assignment produces ~5.5x
selectivity, or whether 5.5x is specific to T_P15.

Phase 50A tested shuffling syllables AMONG the 25 existing triples
(answer: barely matters, 1.10x). This test is fundamentally different:
it asks whether the CHOICE of syllables matters — whether "di,se,ne,co..."
is special compared to "pa,ku,vo,ri..." drawn from the same inventory.

Two inventory options:
  Option A (conservative): 20 distinct syllables from T_P15
  Option B (expansive): all 2-char words in merged dict + all CV combos
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)


def _convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Fast R3-style decode (inlined for performance in hot loop)
# ---------------------------------------------------------------------------

def _fast_decode_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 strategy: try alteration, then strip, then raw."""
    decoded = []
    for token in tokens:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


def _compute_signal_stats(
    real_decoded: List[str],
    null_decoded_list: List[List[str]],
    dict_words: set,
    sigma_threshold: float = 2.0,
) -> Dict:
    """Compute per-word signal statistics (sigma, selectivity)."""
    real_hits = Counter(
        w for w in real_decoded if w in dict_words
    )
    null_hits_list = [
        Counter(w for w in nd if w in dict_words)
        for nd in null_decoded_list
    ]

    # All words that hit in real or any null
    all_words = set(real_hits.keys())
    for nh in null_hits_list:
        all_words.update(nh.keys())

    signal_words = []
    selectivities = []

    for word in all_words:
        real_count = real_hits.get(word, 0)
        null_counts = [nh.get(word, 0) for nh in null_hits_list]
        null_mean = float(np.mean(null_counts))
        null_std = float(np.std(null_counts, ddof=0))

        if null_std > 0:
            sigma = (real_count - null_mean) / null_std
        elif real_count > null_mean:
            sigma = float("inf")
        else:
            sigma = 0.0

        selectivity = real_count / null_mean if null_mean > 0 else (
            float("inf") if real_count > 0 else 0.0
        )

        if sigma > sigma_threshold and real_count > 0:
            signal_words.append({
                "word": word,
                "real_count": real_count,
                "null_mean": null_mean,
                "sigma": sigma,
                "selectivity": selectivity,
            })
            if selectivity != float("inf"):
                selectivities.append(selectivity)

    n_signal = len(signal_words)
    mean_sel = float(np.mean(selectivities)) if selectivities else 0.0
    sel_cv = (
        float(np.std(selectivities) / np.mean(selectivities))
        if len(selectivities) > 2 and np.mean(selectivities) > 0
        else float("inf")
    )

    n_dict_hits = sum(1 for w in real_decoded if w in dict_words)
    dict_hit_rate = n_dict_hits / len(real_decoded) if real_decoded else 0.0

    return {
        "n_signal_words": n_signal,
        "mean_selectivity": mean_sel,
        "selectivity_cv": sel_cv,
        "n_dict_hits": n_dict_hits,
        "dict_hit_rate": dict_hit_rate,
        "top_10_words": sorted(
            signal_words, key=lambda x: x["sigma"], reverse=True
        )[:10],
    }


def _run_option(
    option_label: str,
    inventory: List[str],
    triple_names: List[str],
    real_tokens: List[str],
    null_corpora: List[List[str]],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    dict_merged: set,
    real_assignment: Dict[str, str],
    n_trials: int,
) -> Dict:
    """Run the permutation test for one inventory option."""
    print(f"\n  Option {option_label}: inventory size = {len(inventory)}, "
          f"{n_trials} trials", flush=True)

    # First compute real table stats
    print("    Computing real table stats...", flush=True)
    real_decoded = _fast_decode_r3(
        real_tokens, real_assignment, eva_to_triple,
        modifier_chars, modifier_rules, dict_merged,
    )
    null_decoded_list = [
        _fast_decode_r3(
            nc, real_assignment, eva_to_triple,
            modifier_chars, modifier_rules, dict_merged,
        )
        for nc in null_corpora
    ]
    real_stats = _compute_signal_stats(
        real_decoded, null_decoded_list, dict_merged,
    )
    print(f"    Real table: {real_stats['n_signal_words']} signal words, "
          f"mean sel {real_stats['mean_selectivity']:.2f}, "
          f"CV {real_stats['selectivity_cv']:.3f}")

    # Run null trials
    trial_summaries = []
    inventory_arr = np.array(inventory)

    t_start = time.time()
    for trial in range(n_trials):
        if trial % 100 == 0 and trial > 0:
            elapsed = time.time() - t_start
            rate = trial / elapsed
            eta = (n_trials - trial) / rate
            print(f"    Trial {trial}/{n_trials} "
                  f"({rate:.0f} trials/s, ETA {eta:.0f}s)", flush=True)

        rng = np.random.default_rng(seed=trial)
        random_syls = rng.choice(inventory_arr, size=len(triple_names),
                                 replace=True)
        random_table = dict(zip(triple_names, random_syls.tolist()))

        # Decode real corpus
        trial_real_decoded = _fast_decode_r3(
            real_tokens, random_table, eva_to_triple,
            modifier_chars, modifier_rules, dict_merged,
        )
        # Decode null corpora
        trial_null_decoded = [
            _fast_decode_r3(
                nc, random_table, eva_to_triple,
                modifier_chars, modifier_rules, dict_merged,
            )
            for nc in null_corpora
        ]
        trial_stats = _compute_signal_stats(
            trial_real_decoded, trial_null_decoded, dict_merged,
        )
        trial_summaries.append(trial_stats)

    elapsed = time.time() - t_start
    print(f"    {n_trials} trials completed in {elapsed:.1f}s "
          f"({n_trials / elapsed:.0f} trials/s)")

    # Aggregate null distribution
    null_n_signals = [t["n_signal_words"] for t in trial_summaries]
    null_mean_sels = [t["mean_selectivity"] for t in trial_summaries]
    null_sel_cvs = [
        t["selectivity_cv"] for t in trial_summaries
        if t["selectivity_cv"] != float("inf")
    ]
    null_dict_hits = [t["dict_hit_rate"] for t in trial_summaries]

    # Z-scores
    def _z(real_val, null_vals):
        arr = np.array(null_vals)
        if len(arr) == 0 or np.std(arr) == 0:
            return 0.0
        return float((real_val - np.mean(arr)) / np.std(arr))

    z_n = _z(real_stats["n_signal_words"], null_n_signals)
    z_sel = _z(real_stats["mean_selectivity"], null_mean_sels)
    z_cv = _z(
        real_stats["selectivity_cv"],
        null_sel_cvs,
    ) if null_sel_cvs else 0.0

    # P-values (fraction of null trials meeting or exceeding real)
    p_n = float(np.mean([1 if n >= real_stats["n_signal_words"] else 0
                         for n in null_n_signals]))
    p_sel = float(np.mean([1 if s >= real_stats["mean_selectivity"] else 0
                           for s in null_mean_sels]))

    return {
        "option": option_label,
        "inventory_size": len(inventory),
        "n_trials": n_trials,
        "real_table": {
            "n_signal_words": real_stats["n_signal_words"],
            "mean_selectivity": real_stats["mean_selectivity"],
            "selectivity_cv": real_stats["selectivity_cv"],
            "dict_hit_rate": real_stats["dict_hit_rate"],
            "top_10_words": real_stats["top_10_words"],
        },
        "null_distribution": {
            "n_signal_words": {
                "mean": float(np.mean(null_n_signals)),
                "std": float(np.std(null_n_signals)),
                "min": int(np.min(null_n_signals)),
                "max": int(np.max(null_n_signals)),
                "p5": int(np.percentile(null_n_signals, 5)),
                "p95": int(np.percentile(null_n_signals, 95)),
            },
            "mean_selectivity": {
                "mean": float(np.mean(null_mean_sels)),
                "std": float(np.std(null_mean_sels)),
                "min": float(np.min(null_mean_sels)),
                "max": float(np.max(null_mean_sels)),
                "p5": float(np.percentile(null_mean_sels, 5)),
                "p95": float(np.percentile(null_mean_sels, 95)),
            },
            "selectivity_cv": {
                "mean": float(np.mean(null_sel_cvs)) if null_sel_cvs else None,
                "std": float(np.std(null_sel_cvs)) if null_sel_cvs else None,
            },
            "dict_hit_rate": {
                "mean": float(np.mean(null_dict_hits)),
                "std": float(np.std(null_dict_hits)),
            },
        },
        "z_scores": {
            "n_signal_words": z_n,
            "mean_selectivity": z_sel,
            "selectivity_clustering": z_cv,
        },
        "p_values": {
            "n_signal_ge_real": p_n,
            "mean_sel_ge_real": p_sel,
        },
    }


# ---------------------------------------------------------------------------
# Linguistic coherence scoring
# ---------------------------------------------------------------------------

# Italian verb paradigms: sets of conjugated forms from the same verb.
# A trial "has a paradigm" if it produces ≥3 forms from any one verb.
VERB_PARADIGMS = {
    "dire": {"di", "dice", "dico", "dici", "dicu", "diga", "diri", "dise"},
    "dare": {"da", "dedi", "dido", "dere"},
    "fare": {"fa", "fe"},
    "essere": {"se", "si"},
    "cola_tere": {"cola", "tere", "raso"},  # pharmaceutical verbs
}

# Complete function-word inventory: articles, prepositions, pronouns, aux.
# A trial "has function kit" if it produces items from ≥4 of 5 categories.
FUNCTION_CATEGORIES = {
    "articles": {"la", "li", "le"},
    "prepositions": {"di", "de", "co", "su", "cu", "du"},
    "pronouns": {"te", "ti", "tu", "se", "si", "ci", "ne", "me"},
    "auxiliaries": {"ha", "fa"},
    "conjunctions": {"ni", "ne", "se"},
}

# Pharmaceutical register: terms from the Circa Instans tradition.
# A trial "has pharma" if it produces ≥3 of these.
PHARMA_TERMS = {
    "cola", "tere", "raso", "bene", "sene", "sero", "nera", "sera",
    "sede", "tela", "rati", "dine", "raro", "dico", "sere", "dira",
}


def _score_coherence(signal_word_set: set) -> Dict:
    """Score how linguistically coherent a set of signal words is."""
    # 1. Verb paradigm check: ≥3 forms from any one verb
    best_verb = ""
    best_verb_count = 0
    for verb, forms in VERB_PARADIGMS.items():
        n = len(signal_word_set & forms)
        if n > best_verb_count:
            best_verb_count = n
            best_verb = verb
    has_paradigm = best_verb_count >= 3

    # 2. Function-word inventory: items from ≥4 of 5 categories
    cats_hit = 0
    for cat_name, cat_words in FUNCTION_CATEGORIES.items():
        if signal_word_set & cat_words:
            cats_hit += 1
    has_function_kit = cats_hit >= 4

    # 3. Pharmaceutical register: ≥3 terms
    pharma_count = len(signal_word_set & PHARMA_TERMS)
    has_pharma = pharma_count >= 3

    # Composite: how many of 3 coherence tests pass
    n_coherent = sum([has_paradigm, has_function_kit, has_pharma])

    return {
        "has_paradigm": has_paradigm,
        "best_verb": best_verb,
        "best_verb_count": best_verb_count,
        "has_function_kit": has_function_kit,
        "function_categories_hit": cats_hit,
        "has_pharma": has_pharma,
        "pharma_count": pharma_count,
        "n_coherent": n_coherent,
    }


def run_reviewer_coherence() -> None:
    """Follow-up to permutation test: check linguistic coherence of signal words.

    Re-runs Option A trials (same seeds, same infrastructure) but records
    per-trial signal word sets and scores them for verb paradigms, function-word
    completeness, and pharmaceutical register. Compares against T_P15's
    coherence score.
    """
    t0 = time.time()
    rd = _results_dir()

    print("=" * 70)
    print("REVIEWER ANALYSIS 1b: Signal Word Coherence Check")
    print("=" * 70)

    # ── Load infrastructure (same as run_reviewer_permutation) ──
    print("\n  Loading infrastructure...", flush=True)
    eva_to_triple = build_eva_to_triple_lookup()

    refine_path = os.path.join(rd, "combined_refine.json")
    with open(refine_path) as f:
        assignment = json.load(f)["best_assignment"]
    triple_names = sorted(assignment.keys())

    mod_path = os.path.join(rd, "modifier_integrate.json")
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Merged dictionary (Latin 10K + Italian 10K = ~19K words).
    merged_path = os.path.join(rd, "merged_dict.json")
    with open(merged_path) as f:
        merged_data = json.load(f)
    dict_merged = set(merged_data["merged_words"])

    corpus = load_corpus(verbose=False)
    real_tokens = corpus.get_tokens()

    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, "null_corpus.json")
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r["seed"] for r in null_data.get("null_runs", [])]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        real_tokens
    )
    null_corpora = [
        _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths,
            len(real_tokens), seed,
        )
        for seed in null_seeds
    ]

    inventory = sorted(set(assignment.values()))
    print(f"  Merged dict: {len(dict_merged)}, inventory: {len(inventory)}, "
          f"tokens: {len(real_tokens)}", flush=True)

    # ── Score the real table ──
    print("  Scoring real table...", flush=True)
    real_decoded = _fast_decode_r3(
        real_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, dict_merged,
    )
    null_decoded_list = [
        _fast_decode_r3(
            nc, assignment, eva_to_triple,
            modifier_chars, modifier_rules, dict_merged,
        )
        for nc in null_corpora
    ]
    real_stats = _compute_signal_stats(real_decoded, null_decoded_list, dict_merged)
    real_signal_set = set(
        w["word"] for w in real_stats["top_10_words"]
    )
    # Get ALL signal words, not just top 10
    real_hits = Counter(w for w in real_decoded if w in dict_merged)
    null_hits_list = [
        Counter(w for w in nd if w in dict_merged) for nd in null_decoded_list
    ]
    real_signal_set = set()
    for word in real_hits:
        null_counts = [nh.get(word, 0) for nh in null_hits_list]
        null_mean = float(np.mean(null_counts))
        null_std = float(np.std(null_counts, ddof=0))
        sigma = ((real_hits[word] - null_mean) / null_std) if null_std > 0 else (
            float("inf") if real_hits[word] > null_mean else 0.0
        )
        if sigma > 2.0:
            real_signal_set.add(word)

    real_coherence = _score_coherence(real_signal_set)
    print(f"  Real table: {len(real_signal_set)} signal words, "
          f"coherence {real_coherence['n_coherent']}/3 "
          f"(paradigm={real_coherence['has_paradigm']}, "
          f"function_kit={real_coherence['has_function_kit']}, "
          f"pharma={real_coherence['has_pharma']})", flush=True)

    # ── Run 1000 trials ──
    N_TRIALS = 1000
    inventory_arr = np.array(inventory)
    trial_coherences = []
    n_with_paradigm = 0
    n_with_function_kit = 0
    n_with_pharma = 0
    n_with_all_3 = 0
    max_coherence_seen = 0

    t_start = time.time()
    for trial in range(N_TRIALS):
        if trial % 100 == 0 and trial > 0:
            elapsed = time.time() - t_start
            rate = trial / elapsed
            eta = (N_TRIALS - trial) / rate
            print(f"    Trial {trial}/{N_TRIALS} "
                  f"({rate:.0f}/s, ETA {eta:.0f}s) "
                  f"paradigm={n_with_paradigm}, "
                  f"func={n_with_function_kit}, "
                  f"pharma={n_with_pharma}", flush=True)

        rng = np.random.default_rng(seed=trial)
        random_syls = rng.choice(inventory_arr, size=len(triple_names),
                                 replace=True)
        random_table = dict(zip(triple_names, random_syls.tolist()))

        # Decode real + null with this random table
        trial_real = _fast_decode_r3(
            real_tokens, random_table, eva_to_triple,
            modifier_chars, modifier_rules, dict_merged,
        )
        trial_nulls = [
            _fast_decode_r3(
                nc, random_table, eva_to_triple,
                modifier_chars, modifier_rules, dict_merged,
            )
            for nc in null_corpora
        ]

        # Identify signal words for this trial
        t_real_hits = Counter(w for w in trial_real if w in dict_merged)
        t_null_hits = [
            Counter(w for w in nd if w in dict_merged) for nd in trial_nulls
        ]
        trial_signal_set: set = set()
        for word in t_real_hits:
            nc_counts = [nh.get(word, 0) for nh in t_null_hits]
            nm = float(np.mean(nc_counts))
            ns = float(np.std(nc_counts, ddof=0))
            sig = ((t_real_hits[word] - nm) / ns) if ns > 0 else (
                float("inf") if t_real_hits[word] > nm else 0.0
            )
            if sig > 2.0:
                trial_signal_set.add(word)

        coh = _score_coherence(trial_signal_set)
        trial_coherences.append(coh)
        if coh["has_paradigm"]:
            n_with_paradigm += 1
        if coh["has_function_kit"]:
            n_with_function_kit += 1
        if coh["has_pharma"]:
            n_with_pharma += 1
        if coh["n_coherent"] == 3:
            n_with_all_3 += 1
        if coh["n_coherent"] > max_coherence_seen:
            max_coherence_seen = coh["n_coherent"]

    elapsed = time.time() - t_start
    print(f"    {N_TRIALS} trials in {elapsed:.1f}s", flush=True)

    # ── Aggregate ──
    coherence_scores = [c["n_coherent"] for c in trial_coherences]
    p_paradigm = n_with_paradigm / N_TRIALS
    p_function = n_with_function_kit / N_TRIALS
    p_pharma = n_with_pharma / N_TRIALS
    p_all_3 = n_with_all_3 / N_TRIALS
    p_ge_real = sum(
        1 for s in coherence_scores if s >= real_coherence["n_coherent"]
    ) / N_TRIALS

    if real_coherence["n_coherent"] > max_coherence_seen:
        verdict = "COHERENCE_UNIQUE"
    elif p_ge_real < 0.01:
        verdict = "COHERENCE_RARE"
    elif p_ge_real < 0.05:
        verdict = "COHERENCE_UNCOMMON"
    else:
        verdict = "COHERENCE_COMMON"

    result = {
        "test": "signal_word_coherence",
        "n_trials": N_TRIALS,
        "real_table": {
            "n_signal_words": len(real_signal_set),
            "signal_words": sorted(real_signal_set),
            "coherence": real_coherence,
        },
        "null_distribution": {
            "n_with_paradigm": n_with_paradigm,
            "p_paradigm": p_paradigm,
            "n_with_function_kit": n_with_function_kit,
            "p_function_kit": p_function,
            "n_with_pharma": n_with_pharma,
            "p_pharma": p_pharma,
            "n_with_all_3": n_with_all_3,
            "p_all_3": p_all_3,
            "max_coherence": max_coherence_seen,
            "mean_coherence": float(np.mean(coherence_scores)),
            "coherence_distribution": {
                str(i): sum(1 for s in coherence_scores if s == i)
                for i in range(4)
            },
        },
        "p_ge_real": p_ge_real,
        "verdict": verdict,
        "runtime_seconds": time.time() - t0,
    }

    # ── Print ──
    print("\n" + "=" * 70)
    print("COHERENCE SUMMARY")
    print("=" * 70)
    print(f"\n  Real table ({len(real_signal_set)} signal words):")
    print(f"    Verb paradigm: {real_coherence['has_paradigm']} "
          f"(best: {real_coherence['best_verb']} with "
          f"{real_coherence['best_verb_count']} forms)")
    print(f"    Function kit:  {real_coherence['has_function_kit']} "
          f"({real_coherence['function_categories_hit']}/5 categories)")
    print(f"    Pharma terms:  {real_coherence['has_pharma']} "
          f"({real_coherence['pharma_count']} terms)")
    print(f"    Coherence:     {real_coherence['n_coherent']}/3")
    print(f"\n  1000 random trials:")
    print(f"    With verb paradigm (≥3 forms):   {n_with_paradigm} "
          f"({p_paradigm:.1%})")
    print(f"    With function kit (≥4/5 cats):   {n_with_function_kit} "
          f"({p_function:.1%})")
    print(f"    With pharma register (≥3 terms): {n_with_pharma} "
          f"({p_pharma:.1%})")
    print(f"    With ALL THREE:                  {n_with_all_3} "
          f"({p_all_3:.1%})")
    print(f"    Max coherence:                   {max_coherence_seen}/3")
    print(f"    Coherence distribution:          "
          f"0={sum(1 for s in coherence_scores if s==0)}, "
          f"1={sum(1 for s in coherence_scores if s==1)}, "
          f"2={sum(1 for s in coherence_scores if s==2)}, "
          f"3={sum(1 for s in coherence_scores if s==3)}")
    print(f"\n  P(random ≥ real coherence): {p_ge_real:.4f}")
    print(f"  Verdict: {verdict}")

    out_path = os.path.join(rd, "reviewer_coherence.json")
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved to {out_path}")
    print(f"  Runtime: {time.time() - t0:.1f}s")


def run_reviewer_permutation() -> None:
    t0 = time.time()
    rd = _results_dir()

    print("=" * 70)
    print("REVIEWER ANALYSIS 1: Random Syllabary Permutation Test")
    print("=" * 70)

    # ── Load infrastructure ──
    print("\n  Loading infrastructure...", flush=True)
    eva_to_triple = build_eva_to_triple_lookup()

    # Assignment table
    refine_path = os.path.join(rd, "combined_refine.json")
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data["best_assignment"]
    triple_names = sorted(assignment.keys())

    # Modifiers
    mod_path = os.path.join(rd, "modifier_integrate.json")
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Merged dictionary (Latin 10K + Italian 10K = ~19K words).
    # This is the dictionary that produces the 70 signal words for T_P15.
    # Loaded from pre-built results/merged_dict.json (Phase 38).
    merged_path = os.path.join(rd, "merged_dict.json")
    with open(merged_path) as f:
        merged_data = json.load(f)
    dict_merged = set(merged_data["merged_words"])
    print(f"  Merged dictionary: {len(dict_merged)} words")

    # Real corpus
    corpus = load_corpus(verbose=False)
    real_tokens = corpus.get_tokens()
    print(f"  Real corpus: {len(real_tokens)} tokens")

    # Generate null corpora
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, "null_corpus.json")
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r["seed"] for r in null_data.get("null_runs", [])]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        real_tokens
    )
    null_corpora = []
    for seed in null_seeds:
        nc = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths,
            len(real_tokens), seed,
        )
        null_corpora.append(nc)
    print(f"  Generated {len(null_corpora)} null corpora "
          f"(seeds {null_seeds})")

    # ── Build inventories ──
    # Option A: same 20 syllables from T_P15
    inventory_a = sorted(set(assignment.values()))
    print(f"  Option A inventory: {len(inventory_a)} syllables from T_P15")

    # Option B: all 2-char dict words + full CV grid
    inventory_b_set = set(
        w for w in dict_merged if len(w) == 2 and w.isalpha()
    )
    for c in "bcdgklmnprst":
        for v in "aeiou":
            inventory_b_set.add(c + v)
    inventory_b = sorted(inventory_b_set)
    print(f"  Option B inventory: {len(inventory_b)} syllables "
          f"(2-char dict + CV grid)")

    # ── Run both options ──
    N_TRIALS = 1000

    result_a = _run_option(
        "A", inventory_a, triple_names,
        real_tokens, null_corpora,
        eva_to_triple, modifier_chars, modifier_rules,
        dict_merged, assignment, N_TRIALS,
    )

    result_b = _run_option(
        "B", inventory_b, triple_names,
        real_tokens, null_corpora,
        eva_to_triple, modifier_chars, modifier_rules,
        dict_merged, assignment, N_TRIALS,
    )

    # ── Verdict ──
    # Use Option A as the primary (tighter) null test
    p_n_a = result_a["p_values"]["n_signal_ge_real"]
    p_sel_a = result_a["p_values"]["mean_sel_ge_real"]

    if p_n_a < 0.01 and p_sel_a < 0.01:
        verdict = "SIGNAL_GENUINE"
        interp = (
            "Random syllabary tables almost never produce this many signal "
            "words ({}) at this mean selectivity ({:.2f}x). The T_P15 table "
            "is special: p(n_signal) = {:.4f}, p(mean_sel) = {:.4f}.".format(
                result_a["real_table"]["n_signal_words"],
                result_a["real_table"]["mean_selectivity"],
                p_n_a, p_sel_a,
            )
        )
    elif p_n_a > 0.05 or p_sel_a > 0.05:
        verdict = "SIGNAL_ARTIFACT"
        interp = (
            "Random syllabary tables routinely match or exceed T_P15's "
            "signal word count and selectivity. "
            "p(n_signal) = {:.4f}, p(mean_sel) = {:.4f}. "
            "The ~5.5x selectivity is a property of the Voynich token "
            "distribution interacting with any CV syllabary, not a "
            "property of the specific T_P15 assignment.".format(
                p_n_a, p_sel_a,
            )
        )
    else:
        verdict = "SIGNAL_MARGINAL"
        interp = (
            "T_P15 is better than most random tables but not overwhelmingly "
            "so. p(n_signal) = {:.4f}, p(mean_sel) = {:.4f}. "
            "The signal is partially table-specific and partially structural."
            .format(p_n_a, p_sel_a)
        )

    result = {
        "test": "random_syllabary_permutation",
        "n_trials": N_TRIALS,
        "option_a": result_a,
        "option_b": result_b,
        "verdict": verdict,
        "interpretation": interp,
        "runtime_seconds": time.time() - t0,
    }

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for opt_label, opt_result in [("A", result_a), ("B", result_b)]:
        rt = opt_result["real_table"]
        nd = opt_result["null_distribution"]
        zs = opt_result["z_scores"]
        pv = opt_result["p_values"]
        print(f"\n  Option {opt_label} (inventory {opt_result['inventory_size']}):")
        print(f"    Real table: {rt['n_signal_words']} signal words, "
              f"mean sel {rt['mean_selectivity']:.2f}x, "
              f"CV {rt['selectivity_cv']:.3f}")
        print(f"    Null mean:  {nd['n_signal_words']['mean']:.1f} signal words "
              f"(std {nd['n_signal_words']['std']:.1f}), "
              f"mean sel {nd['mean_selectivity']['mean']:.2f}x "
              f"(std {nd['mean_selectivity']['std']:.2f})")
        print(f"    Z-scores: n_signal={zs['n_signal_words']:.2f}, "
              f"mean_sel={zs['mean_selectivity']:.2f}, "
              f"clustering={zs['selectivity_clustering']:.2f}")
        print(f"    P-values: n_signal={pv['n_signal_ge_real']:.4f}, "
              f"mean_sel={pv['mean_sel_ge_real']:.4f}")

    print(f"\n  Verdict: {verdict}")
    print(f"  {interp}")

    # ── Save ──
    out_path = os.path.join(rd, "reviewer_permutation.json")
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved to {out_path}")
    print(f"\n  Total runtime: {time.time() - t0:.1f}s")
