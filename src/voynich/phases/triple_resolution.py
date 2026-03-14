"""
Phase 53 Track B: Validate and Apply Triple Corrections
========================================================
Load accepted corrections from Track A, compute pre/post baselines,
run null test to validate that paradigm consensus exceeds random,
and produce final verdict.

Dependency chain:
    paradigm_constraints.json  (Track A)
    combined_refine.json       (Phase 15)
    bootstrap_loop.json        (Phase 30)
    modifier_integrate.json    (Phase 16)
    signal_bigrams.json        (Phase 29)
    triple_tiers.json          (Phase 44)
        -> triple_resolution.json (this step)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.concatenation_bridge import (
    _build_partial_decode,
    _build_pharma_dict,
    _extract_implied_assignments,
    _search_dict,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_SET
from voynich.phases.word_validation import _extract_stem, _find_paradigms


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
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, set):
        return sorted(_convert(item) for item in obj)
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


def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    return hits / len(decoded)


def _decode_tokens_simple(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> List[str]:
    """Decode tokens using modifier-aware strip (no modifier rules)."""
    decoded = []
    for token in tokens:
        d = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        decoded.append(d.lower())
    return decoded


# ---------------------------------------------------------------------------
# Signal word verification
# ---------------------------------------------------------------------------

def _verify_signal_words(
    assignment: Dict[str, str],
    original_assignment: Dict[str, str],
    token_evas: List[str],
    token_decoded: List[str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> Tuple[int, int, List[str]]:
    """Verify signal words are preserved after correction.

    Compares decode output of original vs corrected assignment for all
    EVA types that produce signal words. Returns how many signal words
    are NOT broken by the correction.

    Returns (n_preserved, n_total_checked, broken_words).
    """
    # Find unique EVA types that produce signal words
    signal_eva_types: Dict[str, str] = {}  # eva_type -> signal word
    for eva, dec in zip(token_evas, token_decoded):
        if dec in SIGNAL_WORDS_SET:
            if eva not in signal_eva_types:
                signal_eva_types[eva] = dec

    n_total = len(signal_eva_types)
    broken = []
    n_preserved = 0

    for eva_type, expected_signal in signal_eva_types.items():
        original_decode = decode_token_modifier_aware(
            eva_type, original_assignment, eva_to_triple, modifier_chars,
        )
        corrected_decode = decode_token_modifier_aware(
            eva_type, assignment, eva_to_triple, modifier_chars,
        )

        if original_decode == corrected_decode:
            n_preserved += 1
        else:
            broken.append(
                f"{eva_type}: original={original_decode}, "
                f"corrected={corrected_decode} (signal={expected_signal})"
            )

    return n_preserved, n_total, broken


# ---------------------------------------------------------------------------
# Null test: paradigm consensus on shuffled tables
# ---------------------------------------------------------------------------

def _null_paradigm_consensus(
    token_evas: List[str],
    token_decoded: List[str],
    token_folios: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    confirmed_triples: Set[str],
    pharma_dict: Set[str],
    rng: random.Random,
    sample_rate: float = 0.5,
) -> float:
    """Run bridge search with shuffled assignment, build paradigms,
    extract constraints, and return the max consensus rate.

    Returns the maximum consensus across any triple (0.0 if no constraints).
    """
    n_tokens = len(token_evas)

    # Shuffle assignment values
    keys = list(assignment.keys())
    values = list(assignment.values())
    rng.shuffle(values)
    null_assignment = dict(zip(keys, values))

    # Find signal positions, sample a fraction
    signal_positions = [i for i in range(n_tokens)
                        if token_decoded[i] in SIGNAL_WORDS_SET]
    n_sample = max(1, int(len(signal_positions) * sample_rate))
    sampled = set(rng.sample(signal_positions, n_sample))

    seen: Set[int] = set()
    pair_data: Dict[Tuple[str, str], Dict] = defaultdict(
        lambda: {'folios': set(), 'implied': {}}
    )

    for sig_idx in sampled:
        for offset in (-1, 1):
            nbr_idx = sig_idx + offset
            if nbr_idx < 0 or nbr_idx >= n_tokens:
                continue
            if token_decoded[nbr_idx] in SIGNAL_WORDS_SET:
                continue
            if nbr_idx in seen:
                continue
            seen.add(nbr_idx)

            dark_eva = token_evas[nbr_idx]
            pattern, details = _build_partial_decode(
                dark_eva, null_assignment, eva_to_triple,
                modifier_chars, confirmed_triples,
            )

            n_conf = sum(1 for _, _, _, c in details if c)
            n_free = sum(1 for _, _, _, c in details if not c)
            if n_conf < 1 or n_free < 1 or n_free > 3:
                continue

            matches = _search_dict(pattern, pharma_dict)
            for mword in matches:
                key = (dark_eva, mword)
                pair_data[key]['folios'].add(token_folios[nbr_idx])
                implied = _extract_implied_assignments(
                    pattern, mword, details, eva_to_triple,
                )
                if implied:
                    for tk, tv in implied.items():
                        pair_data[key]['implied'][tk] = tv

    # Build catalog-like entries for paradigm detection
    catalog_entries = []
    for (eva_type, latin_word), info in pair_data.items():
        if len(info['folios']) >= 2:
            catalog_entries.append({
                'eva_type': eva_type,
                'latin_word': latin_word,
                'tier': 'T2',
                'confidence': 0.5,
                'implied_assignments': info['implied'],
            })

    if not catalog_entries:
        return 0.0

    # Find paradigms
    paradigms = _find_paradigms(catalog_entries)

    # Extract constraints
    triple_values: Dict[str, List[str]] = defaultdict(list)
    for entry in catalog_entries:
        implied = entry.get('implied_assignments', {})
        for tk, tv in implied.items():
            # Check if this entry belongs to any paradigm
            stem = _extract_stem(entry['latin_word'])
            for p in paradigms:
                if (p['stem'] == stem and
                    entry['eva_type'] in p['eva_types']):
                    triple_values[tk].append(tv)
                    break

    if not triple_values:
        return 0.0

    # Compute max consensus
    max_consensus = 0.0
    for tk, values in triple_values.items():
        if len(values) < 3:
            continue
        counts = Counter(values)
        top_count = counts.most_common(1)[0][1]
        consensus = top_count / len(values)
        max_consensus = max(max_consensus, consensus)

    return max_consensus


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_triple_resolution() -> None:
    """Phase 53 Track B: Validate and apply triple corrections."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 53 TRACK B: Triple Resolution and Validation")
    print("=" * 70)

    rd = _results_dir()

    # ── Load data ─────────────────────────────────────────────────────
    print("\n  B.1  Loading data...")

    constraints_data = _safe_load(os.path.join(rd, 'paradigm_constraints.json'))
    if not constraints_data:
        print("  *** paradigm_constraints.json not found — run Track A first ***")
        return

    accepted_corrections = constraints_data.get('accepted_corrections', [])
    per_triple = constraints_data.get('per_triple_summary', {})

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        assignment = json.load(f)['best_assignment']

    with open(os.path.join(rd, 'bootstrap_loop.json')) as f:
        boot_data = json.load(f)
    confirmed_triples = set(boot_data.get('confirmed_triples', []))

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars = set(mod_data.get('modifier_chars', []))

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        bigram_data = json.load(f)
    token_evas = bigram_data['token_evas']
    token_decoded = bigram_data['token_decoded']
    token_folios = bigram_data['token_folios']

    eva_to_triple = build_eva_to_triple_lookup()

    # Build reference word set
    print("       Building reference dictionary...")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for text in ref_corpus.get_texts('latin')
            for w in [t.lower() for t in text.tokens]
            if len(w) >= 2 and w.isalpha()
        )
    except Exception:
        base_words = set()
    expanded_dict, _ = build_expanded_word_set(base_words)
    ref_10k = base_words  # base reference ~10K
    ref_131k = base_words | expanded_dict

    print(f"       {len(accepted_corrections)} accepted corrections from Track A")
    print(f"       {len(per_triple)} triples with constraints")

    # ── Pre-correction baseline ───────────────────────────────────────
    print("\n  B.2  Pre-correction baseline...")

    decoded_pre = _decode_tokens_simple(
        token_evas, assignment, eva_to_triple, modifier_chars,
    )
    dict_hit_pre_10k = _compute_dict_hit(decoded_pre, ref_10k)
    dict_hit_pre_131k = _compute_dict_hit(decoded_pre, ref_131k)

    # Signal word baseline: compare against itself (should be 100%)
    n_signal_pre, n_total_signal, _ = _verify_signal_words(
        assignment, assignment, token_evas, token_decoded,
        eva_to_triple, modifier_chars,
    )

    print(f"       Dict-hit (10K): {dict_hit_pre_10k:.4f}")
    print(f"       Dict-hit (131K): {dict_hit_pre_131k:.4f}")
    print(f"       Signal EVA types: {n_total_signal} "
          f"(preserved: {n_signal_pre}/{n_total_signal})")

    baseline = {
        'dict_hit_10k': round(dict_hit_pre_10k, 6),
        'dict_hit_131k': round(dict_hit_pre_131k, 6),
        'signal_word_count': n_signal_pre,
    }

    # ── Apply corrections ─────────────────────────────────────────────
    corrected_assignment = dict(assignment)
    applied_corrections = []

    if accepted_corrections:
        print(f"\n  B.3  Applying {len(accepted_corrections)} corrections...")

        for corr in accepted_corrections:
            corrected_assignment[corr['triple']] = corr['new_value']
            applied_corrections.append(corr)
            print(f"         {corr['triple']}: {corr['old_value']} -> "
                  f"{corr['new_value']}")

        # Verify signal words after correction (compare vs original)
        n_signal_post, _, broken = _verify_signal_words(
            corrected_assignment, assignment, token_evas, token_decoded,
            eva_to_triple, modifier_chars,
        )

        if n_signal_post < n_total_signal:
            print(f"\n  *** Signal word regression! {n_signal_post}/{n_total_signal}")
            print(f"      Broken: {broken}")
            # Revert all corrections
            corrected_assignment = dict(assignment)
            applied_corrections = []
            print("      Reverted ALL corrections")
    else:
        print("\n  B.3  No corrections to apply — skipping")

    # ── Post-correction metrics ───────────────────────────────────────
    decoded_post = _decode_tokens_simple(
        token_evas, corrected_assignment, eva_to_triple, modifier_chars,
    )
    dict_hit_post_10k = _compute_dict_hit(decoded_post, ref_10k)
    dict_hit_post_131k = _compute_dict_hit(decoded_post, ref_131k)

    n_signal_final, _, _ = _verify_signal_words(
        corrected_assignment, assignment, token_evas, token_decoded,
        eva_to_triple, modifier_chars,
    )

    delta_10k = dict_hit_post_10k - dict_hit_pre_10k
    delta_131k = dict_hit_post_131k - dict_hit_pre_131k

    print(f"\n  B.4  Post-correction metrics...")
    print(f"       Dict-hit (10K):  {dict_hit_post_10k:.4f} "
          f"(delta={delta_10k:+.4f})")
    print(f"       Dict-hit (131K): {dict_hit_post_131k:.4f} "
          f"(delta={delta_131k:+.4f})")
    print(f"       Signal words: {n_signal_final}/{n_total_signal}")

    post_correction = {
        'dict_hit_10k': round(dict_hit_post_10k, 6),
        'dict_hit_131k': round(dict_hit_post_131k, 6),
        'signal_word_count': n_signal_final,
        'delta_dict_hit_10k': round(delta_10k, 6),
        'delta_dict_hit_131k': round(delta_131k, 6),
    }

    # ── Null test ─────────────────────────────────────────────────────
    print("\n  B.5  Null test (20 shuffled assignments)...")

    # Compute real consensus rate
    real_max_consensus = 0.0
    for summary in per_triple.values():
        cons = summary.get('consensus', 0.0)
        n_obs = summary.get('n_unique_constraints', 0)
        if n_obs >= 3 and cons > real_max_consensus:
            real_max_consensus = cons

    rng = random.Random(53)
    pharma_dict = _build_pharma_dict()
    null_consensuses: List[float] = []

    for trial in range(20):
        nc = _null_paradigm_consensus(
            token_evas, token_decoded, token_folios,
            assignment, eva_to_triple, modifier_chars,
            confirmed_triples, pharma_dict, rng,
            sample_rate=0.3,
        )
        null_consensuses.append(nc)
        if (trial + 1) % 5 == 0:
            print(f"       Trial {trial + 1}/20 done")

    null_mean = sum(null_consensuses) / len(null_consensuses) if null_consensuses else 0.0
    null_std = (math.sqrt(sum((x - null_mean) ** 2 for x in null_consensuses)
                          / len(null_consensuses))
                if null_consensuses else 1.0)
    null_z = ((real_max_consensus - null_mean) / null_std
              if null_std > 0 else
              (float('inf') if real_max_consensus > null_mean else 0.0))
    null_selectivity = (real_max_consensus / null_mean
                        if null_mean > 0 else float('inf'))

    print(f"       Real max consensus: {real_max_consensus:.4f}")
    print(f"       Null mean: {null_mean:.4f} (std={null_std:.4f})")
    print(f"       Z-score: {null_z:.2f}")
    print(f"       Selectivity: {null_selectivity:.2f}x")

    null_test = {
        'real_consensus_rate': round(real_max_consensus, 4),
        'null_mean_consensus': round(null_mean, 4),
        'null_std': round(null_std, 4),
        'z_score': round(null_z, 2),
        'selectivity': round(null_selectivity, 4),
        'n_iterations': 20,
    }

    # ── Verdict ───────────────────────────────────────────────────────
    if len(applied_corrections) > 0 and null_z > 2.0 and delta_10k > 0:
        verdict = 'CORRECTIONS_VALID'
    elif len(applied_corrections) > 0 and null_z > 2.0:
        verdict = 'CORRECTIONS_MARGINAL'
    else:
        verdict = 'NO_CORRECTIONS'

    print(f"\n  VERDICT: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = {
        'baseline': baseline,
        'corrections_applied': len(applied_corrections),
        'corrections_detail': applied_corrections,
        'corrected_assignment': corrected_assignment,
        'post_correction': post_correction,
        'null_test': null_test,
        'verdict': verdict,
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'triple_resolution.json', result)
    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {runtime:.1f}s")
