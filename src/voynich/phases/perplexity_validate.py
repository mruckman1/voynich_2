"""
Phase 33.7 -- Perplexity Cross-Validation
============================================
Cross-validates the perplexity-optimal table (Step 33.6) against the
SIGNAL-optimal table (Step 33.3) and Phase 15's original table.  Builds
a consensus table where both methods agree.

Evaluates all four tables (Phase 15, SIGNAL, perplexity, consensus) on a
held-out validation split (even-numbered folios) using dict_hit, signal
classification, signal rate, and bigram z-score.

Dependency chain:
    combined_refine.json       (Phase 15 best_assignment)
    signal_guided_swap.json    (Step 33.3 — SIGNAL-corrected table)
    perplexity_search.json     (Step 33.6 — perplexity-optimised table)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 null seeds)
        → perplexity_validate.json  (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.csp_solver import decode_token
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)


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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _folio_number(folio: str) -> int:
    """Extract numeric part from folio name (e.g. 'f1r' -> 1, 'f70v2' -> 70)."""
    return int(''.join(c for c in folio if c.isdigit()))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TableMetrics:
    name: str
    dict_hit: float
    signal_rate: float
    n_signal: int
    n_anti_signal: int
    bigram_z: float
    n_signal_pairs: int
    n_bigram_hits: int


@dataclass
class TripleAgreement:
    triple_key: str
    phase15_syllable: str
    signal_syllable: str
    ppl_syllable: str
    consensus_syllable: str
    agreement: str  # 'BOTH_AGREE', 'SIGNAL_ONLY', 'PPL_ONLY', 'CONFLICT', 'UNCHANGED'


@dataclass
class PerplexityValidateResult:
    # Three-table comparison
    phase15_metrics: Dict
    signal_metrics: Dict
    ppl_metrics: Dict
    consensus_metrics: Dict
    # Agreement
    n_both_agree: int
    n_signal_only: int
    n_ppl_only: int
    n_conflict: int
    n_unchanged: int
    triple_agreements: List[Dict]
    # Consensus table
    consensus_assignment: Dict[str, str]
    n_consensus_changes: int
    # Best table
    best_table: str  # which table has highest bigram_z
    best_bigram_z: float
    # Verdict
    verdict: str  # 'CONSENSUS_IMPROVED', 'SINGLE_APPROACH_BEST', 'NO_IMPROVEMENT'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def _classify_tokens(
    real_hits: List[bool],
    null_hits_lists: List[List[bool]],
    n_tokens: int,
) -> Tuple[int, int, int, List[str]]:
    """Classify every token and return (n_signal, n_anti_signal, n_shared_hit, classifications)."""
    n_signal = 0
    n_anti_signal = 0
    n_shared_hit = 0
    classifications: List[str] = []

    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_lists if nh[idx])

        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
            n_signal += 1
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
            n_shared_hit += 1
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
            n_anti_signal += 1
        else:
            classifications.append('SHARED_MISS')

    return n_signal, n_anti_signal, n_shared_hit, classifications


# ---------------------------------------------------------------------------
# Bigram z-score computation
# ---------------------------------------------------------------------------

def _compute_bigram_z(
    classifications: List[str],
    decoded: List[str],
    folios: List[str],
    ref_bigrams: Set[Tuple[str, str]],
    n_perms: int = 1000,
    seed: int = 42,
) -> Tuple[int, int, float]:
    """Compute bigram z-score for SIGNAL-SIGNAL pairs.

    Returns (n_signal_pairs, n_bigram_hits, z_score).
    """
    n_tokens = len(classifications)
    n_signal = sum(1 for c in classifications if c == 'SIGNAL')

    # Count actual SIGNAL-SIGNAL bigram hits
    n_pairs = 0
    n_hits = 0
    for i in range(n_tokens - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and folios[i] == folios[i + 1]):
            n_pairs += 1
            if (decoded[i], decoded[i + 1]) in ref_bigrams:
                n_hits += 1

    if n_pairs == 0:
        return 0, 0, 0.0

    hit_rate = n_hits / n_pairs

    # Null permutation test
    rng = random.Random(seed)
    indices = list(range(n_tokens))
    null_rates: List[float] = []

    for _ in range(n_perms):
        fake_signal = set(rng.sample(indices, min(n_signal, n_tokens)))
        fp = 0
        fh = 0
        for i in range(n_tokens - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and folios[i] == folios[i + 1]):
                fp += 1
                if (decoded[i], decoded[i + 1]) in ref_bigrams:
                    fh += 1
        null_rates.append(fh / fp if fp > 0 else 0.0)

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = (sum((r - null_mean) ** 2 for r in null_rates)
                / len(null_rates) if null_rates else 0.0)
    null_std = null_var ** 0.5

    if null_std > 0:
        z_score = (hit_rate - null_mean) / null_std
    else:
        z_score = float('inf') if hit_rate > null_mean else 0.0

    return n_pairs, n_hits, z_score


# ---------------------------------------------------------------------------
# Folio arrays
# ---------------------------------------------------------------------------

def _build_folio_arrays(corpus) -> Tuple[List[str], List[str]]:
    """Build parallel (token, folio) arrays from the corpus."""
    all_tokens: List[str] = []
    token_folios: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
    return all_tokens, token_folios


# ---------------------------------------------------------------------------
# Evaluate a table on a subset of tokens
# ---------------------------------------------------------------------------

def _evaluate_table(
    name: str,
    assignment: Dict[str, str],
    all_tokens: List[str],
    token_folios: List[str],
    subset_indices: Set[int],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    ref_bigrams: Set[Tuple[str, str]],
    null_seeds: List[int],
    bigram_probs: Any,
    initial_probs: Any,
    token_lengths: Any,
) -> TableMetrics:
    """Decode and evaluate a table on a subset of tokens (held-out validation).

    Returns a TableMetrics summarising dict_hit, signal classification, and
    bigram z-score for the subset.
    """
    n_total = len(all_tokens)
    subset_list = sorted(subset_indices)
    n_subset = len(subset_list)

    # Decode real tokens (full corpus needed for null comparison)
    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]

    # Dict-hit on validation subset only
    subset_hits = sum(1 for i in subset_list if real_hits[i])
    dict_hit = subset_hits / n_subset if n_subset > 0 else 0.0

    # Decode null corpora (full corpus to keep index alignment)
    null_hits_lists: List[List[bool]] = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_total, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_hits_lists.append([w in ref_word_set for w in null_decoded])

    # Classify full corpus (needed for correct null_hit_count at each index)
    _, _, _, classifications = _classify_tokens(
        real_hits, null_hits_lists, n_total,
    )

    # Subset classification counts
    n_signal = sum(1 for i in subset_list if classifications[i] == 'SIGNAL')
    n_anti = sum(1 for i in subset_list if classifications[i] == 'ANTI_SIGNAL')
    signal_rate = n_signal / n_subset if n_subset > 0 else 0.0

    # Bigram z on the subset: build subset-restricted classification array
    sub_cls: List[str] = []
    sub_decoded: List[str] = []
    sub_folios: List[str] = []
    for i in subset_list:
        sub_cls.append(classifications[i])
        sub_decoded.append(real_decoded[i])
        sub_folios.append(token_folios[i])

    n_signal_pairs, n_bigram_hits, bigram_z = _compute_bigram_z(
        sub_cls, sub_decoded, sub_folios, ref_bigrams,
        n_perms=1000, seed=42,
    )

    return TableMetrics(
        name=name,
        dict_hit=round(dict_hit, 6),
        signal_rate=round(signal_rate, 6),
        n_signal=n_signal,
        n_anti_signal=n_anti,
        bigram_z=round(bigram_z, 2) if bigram_z != float('inf') else 999.0,
        n_signal_pairs=n_signal_pairs,
        n_bigram_hits=n_bigram_hits,
    )


# ---------------------------------------------------------------------------
# Agreement analysis
# ---------------------------------------------------------------------------

def _analyse_agreement(
    phase15: Dict[str, str],
    signal_tab: Dict[str, str],
    ppl_tab: Dict[str, str],
) -> Tuple[List[TripleAgreement], Dict[str, str]]:
    """Compare three tables triple-by-triple and build a consensus table.

    Returns (agreements_list, consensus_assignment).
    """
    all_keys = sorted(set(phase15.keys()) | set(signal_tab.keys()) | set(ppl_tab.keys()))

    agreements: List[TripleAgreement] = []
    consensus: Dict[str, str] = {}

    for tk in all_keys:
        p15 = phase15.get(tk, '')
        sig = signal_tab.get(tk, '')
        ppl = ppl_tab.get(tk, '')

        sig_changed = sig != p15 and sig != ''
        ppl_changed = ppl != p15 and ppl != ''

        if not sig_changed and not ppl_changed:
            # Neither method changed this triple
            agreement = 'UNCHANGED'
            consensus[tk] = p15
        elif sig_changed and ppl_changed and sig == ppl:
            # Both changed to the same syllable — strongest agreement
            agreement = 'BOTH_AGREE'
            consensus[tk] = sig
        elif sig_changed and ppl_changed and sig != ppl:
            # Both changed but to different syllables — conflict
            agreement = 'CONFLICT'
            consensus[tk] = p15  # keep Phase 15 original
        elif sig_changed and not ppl_changed:
            agreement = 'SIGNAL_ONLY'
            consensus[tk] = sig  # adopt if it improved; refined below
        elif ppl_changed and not sig_changed:
            agreement = 'PPL_ONLY'
            consensus[tk] = ppl  # adopt if it improved; refined below
        else:
            agreement = 'UNCHANGED'
            consensus[tk] = p15

        agreements.append(TripleAgreement(
            triple_key=tk,
            phase15_syllable=p15,
            signal_syllable=sig,
            ppl_syllable=ppl,
            consensus_syllable=consensus[tk],
            agreement=agreement,
        ))

    return agreements, consensus


def _refine_consensus(
    agreements: List[TripleAgreement],
    consensus: Dict[str, str],
    phase15: Dict[str, str],
    signal_metrics: TableMetrics,
    ppl_metrics: TableMetrics,
    phase15_metrics: TableMetrics,
) -> Dict[str, str]:
    """Refine single-method changes: adopt only if that method improved
    both dict_hit and bigram_z over Phase 15.

    For SIGNAL_ONLY: adopt if signal_metrics improved both metrics.
    For PPL_ONLY: adopt if ppl_metrics improved both metrics.
    Otherwise revert to Phase 15.
    """
    signal_improved_both = (
        signal_metrics.dict_hit >= phase15_metrics.dict_hit
        and signal_metrics.bigram_z >= phase15_metrics.bigram_z
    )
    ppl_improved_both = (
        ppl_metrics.dict_hit >= phase15_metrics.dict_hit
        and ppl_metrics.bigram_z >= phase15_metrics.bigram_z
    )

    refined = dict(consensus)
    for ag in agreements:
        if ag.agreement == 'SIGNAL_ONLY' and not signal_improved_both:
            refined[ag.triple_key] = phase15.get(ag.triple_key, ag.phase15_syllable)
            ag.consensus_syllable = refined[ag.triple_key]
        elif ag.agreement == 'PPL_ONLY' and not ppl_improved_both:
            refined[ag.triple_key] = phase15.get(ag.triple_key, ag.phase15_syllable)
            ag.consensus_syllable = refined[ag.triple_key]

    return refined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_perplexity_validate() -> None:
    """Step 33.7: Perplexity cross-validation — consensus table from SIGNAL
    and perplexity methods."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 33.7: Perplexity Cross-Validation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load three tables ──
    print("\n  1. Loading tables ...")

    # Phase 15 original
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    phase15_assignment = dict(refine_data.get('best_assignment', {}))
    print(f"     Phase 15: {len(phase15_assignment)} triples")

    # SIGNAL-corrected (Step 33.3)
    signal_fallback = False
    swap_path = os.path.join(rd, 'signal_guided_swap.json')
    if os.path.exists(swap_path):
        with open(swap_path) as f:
            swap_data = json.load(f)
        signal_assignment = dict(swap_data.get('new_assignment', {}))
        if not signal_assignment:
            signal_assignment = dict(phase15_assignment)
            signal_fallback = True
    else:
        signal_assignment = dict(phase15_assignment)
        signal_fallback = True
    if signal_fallback:
        print("     SIGNAL table: falling back to Phase 15 (signal_guided_swap.json missing/empty)")
    else:
        n_sig_diff = sum(1 for k in phase15_assignment
                         if signal_assignment.get(k) != phase15_assignment.get(k))
        print(f"     SIGNAL table: {len(signal_assignment)} triples, "
              f"{n_sig_diff} differ from Phase 15")

    # Perplexity-optimised (Step 33.6)
    ppl_fallback = False
    ppl_path = os.path.join(rd, 'perplexity_search.json')
    if os.path.exists(ppl_path):
        with open(ppl_path) as f:
            ppl_data = json.load(f)
        ppl_assignment = dict(ppl_data.get('best_assignment', {}))
        if not ppl_assignment:
            ppl_assignment = dict(phase15_assignment)
            ppl_fallback = True
    else:
        ppl_assignment = dict(phase15_assignment)
        ppl_fallback = True
    if ppl_fallback:
        print("     Perplexity table: falling back to Phase 15 (perplexity_search.json missing/empty)")
    else:
        n_ppl_diff = sum(1 for k in phase15_assignment
                         if ppl_assignment.get(k) != phase15_assignment.get(k))
        print(f"     Perplexity table: {len(ppl_assignment)} triples, "
              f"{n_ppl_diff} differ from Phase 15")

    # ── 2. Load modifier rules, null seeds, reference word set ──
    print("\n  2. Loading supporting data ...")

    # Phase 16 modifiers
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Null corpus seeds
    null_path = os.path.join(rd, 'null_corpus.json')
    null_seeds = [100, 101, 102, 103, 104]
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    print(f"     Modifiers: {len(modifier_chars)} chars")
    print(f"     Null seeds: {null_seeds}")

    # Reference word set
    print("\n  3. Building reference word set ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # Reference bigrams
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]
    ref_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens) - 1):
        ref_bigrams.add((ref_tokens[i], ref_tokens[i + 1]))
    print(f"     {len(ref_bigrams)} reference bigrams")

    # ── 3. Load corpus and split into held-out validation (even folios) ──
    print("\n  4. Loading corpus and splitting by folio parity ...")
    corpus = load_corpus(verbose=False)
    all_tokens, token_folios = _build_folio_arrays(corpus)
    n_tokens = len(all_tokens)

    # Build even/odd folio split
    folio_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, folio in enumerate(token_folios):
        folio_to_indices[folio].append(idx)

    validation_indices: Set[int] = set()  # even folio numbers
    train_indices: Set[int] = set()       # odd folio numbers
    for folio, idxs in folio_to_indices.items():
        fnum = _folio_number(folio)
        if fnum % 2 == 0:
            validation_indices.update(idxs)
        else:
            train_indices.update(idxs)

    n_val = len(validation_indices)
    n_train = len(train_indices)
    print(f"     Total tokens: {n_tokens}")
    print(f"     Train (odd folios): {n_train} tokens")
    print(f"     Validation (even folios): {n_val} tokens")

    # ── 4. Build bigram model for null corpus generation ──
    print("\n  5. Building EVA bigram model for null corpora ...")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    # ── 5. Evaluate all three tables on the validation split ──
    print("\n  6. Evaluating Phase 15 table on validation split ...")
    phase15_metrics = _evaluate_table(
        'phase15', phase15_assignment,
        all_tokens, token_folios, validation_indices,
        eva_to_triple, modifier_chars, modifier_rules,
        ref_word_set, ref_bigrams, null_seeds,
        bigram_probs, initial_probs, token_lengths,
    )
    print(f"     dict_hit={phase15_metrics.dict_hit:.4f}  "
          f"signal_rate={phase15_metrics.signal_rate:.4f}  "
          f"bigram_z={phase15_metrics.bigram_z:.2f}  "
          f"n_signal={phase15_metrics.n_signal}  "
          f"n_anti={phase15_metrics.n_anti_signal}")

    print("\n  7. Evaluating SIGNAL table on validation split ...")
    signal_metrics = _evaluate_table(
        'signal', signal_assignment,
        all_tokens, token_folios, validation_indices,
        eva_to_triple, modifier_chars, modifier_rules,
        ref_word_set, ref_bigrams, null_seeds,
        bigram_probs, initial_probs, token_lengths,
    )
    print(f"     dict_hit={signal_metrics.dict_hit:.4f}  "
          f"signal_rate={signal_metrics.signal_rate:.4f}  "
          f"bigram_z={signal_metrics.bigram_z:.2f}  "
          f"n_signal={signal_metrics.n_signal}  "
          f"n_anti={signal_metrics.n_anti_signal}")

    print("\n  8. Evaluating perplexity table on validation split ...")
    ppl_metrics = _evaluate_table(
        'perplexity', ppl_assignment,
        all_tokens, token_folios, validation_indices,
        eva_to_triple, modifier_chars, modifier_rules,
        ref_word_set, ref_bigrams, null_seeds,
        bigram_probs, initial_probs, token_lengths,
    )
    print(f"     dict_hit={ppl_metrics.dict_hit:.4f}  "
          f"signal_rate={ppl_metrics.signal_rate:.4f}  "
          f"bigram_z={ppl_metrics.bigram_z:.2f}  "
          f"n_signal={ppl_metrics.n_signal}  "
          f"n_anti={ppl_metrics.n_anti_signal}")

    # ── 6. Agreement analysis ──
    print("\n  9. Agreement analysis ...")
    agreements, consensus_assignment = _analyse_agreement(
        phase15_assignment, signal_assignment, ppl_assignment,
    )

    # Refine: revert single-method changes if that method did not improve
    consensus_assignment = _refine_consensus(
        agreements, consensus_assignment,
        phase15_assignment,
        signal_metrics, ppl_metrics, phase15_metrics,
    )

    # Count agreement categories
    n_both_agree = sum(1 for a in agreements if a.agreement == 'BOTH_AGREE')
    n_signal_only = sum(1 for a in agreements if a.agreement == 'SIGNAL_ONLY')
    n_ppl_only = sum(1 for a in agreements if a.agreement == 'PPL_ONLY')
    n_conflict = sum(1 for a in agreements if a.agreement == 'CONFLICT')
    n_unchanged = sum(1 for a in agreements if a.agreement == 'UNCHANGED')
    n_consensus_changes = sum(
        1 for tk in phase15_assignment
        if consensus_assignment.get(tk) != phase15_assignment.get(tk)
    )

    print(f"     BOTH_AGREE:  {n_both_agree}")
    print(f"     SIGNAL_ONLY: {n_signal_only}")
    print(f"     PPL_ONLY:    {n_ppl_only}")
    print(f"     CONFLICT:    {n_conflict}")
    print(f"     UNCHANGED:   {n_unchanged}")
    print(f"     Consensus changes from Phase 15: {n_consensus_changes}")

    if n_both_agree > 0:
        print("\n     BOTH_AGREE triples:")
        for a in agreements:
            if a.agreement == 'BOTH_AGREE':
                print(f"       {a.triple_key}: {a.phase15_syllable} -> "
                      f"{a.consensus_syllable}")

    if n_conflict > 0:
        print("\n     CONFLICT triples (kept Phase 15):")
        for a in agreements:
            if a.agreement == 'CONFLICT':
                print(f"       {a.triple_key}: Phase15={a.phase15_syllable}  "
                      f"SIGNAL={a.signal_syllable}  PPL={a.ppl_syllable}")

    # ── 7. Evaluate consensus table on validation split ──
    print("\n  10. Evaluating consensus table on validation split ...")
    consensus_metrics = _evaluate_table(
        'consensus', consensus_assignment,
        all_tokens, token_folios, validation_indices,
        eva_to_triple, modifier_chars, modifier_rules,
        ref_word_set, ref_bigrams, null_seeds,
        bigram_probs, initial_probs, token_lengths,
    )
    print(f"     dict_hit={consensus_metrics.dict_hit:.4f}  "
          f"signal_rate={consensus_metrics.signal_rate:.4f}  "
          f"bigram_z={consensus_metrics.bigram_z:.2f}  "
          f"n_signal={consensus_metrics.n_signal}  "
          f"n_anti={consensus_metrics.n_anti_signal}")

    # ── 8. Summary comparison ──
    print("\n  11. Summary comparison:")
    all_metrics = [phase15_metrics, signal_metrics, ppl_metrics, consensus_metrics]
    print(f"     {'Table':12s}  {'dict_hit':>8s}  {'signal':>8s}  {'bigram_z':>8s}  "
          f"{'n_sig':>5s}  {'n_anti':>6s}  {'pairs':>5s}  {'hits':>4s}")
    print("     " + "-" * 66)
    for m in all_metrics:
        print(f"     {m.name:12s}  {m.dict_hit:8.4f}  {m.signal_rate:8.4f}  "
              f"{m.bigram_z:8.2f}  {m.n_signal:5d}  {m.n_anti_signal:6d}  "
              f"{m.n_signal_pairs:5d}  {m.n_bigram_hits:4d}")

    # ── 9. Best table determination ──
    best_metric = max(all_metrics, key=lambda m: m.bigram_z)
    best_table = best_metric.name
    best_bigram_z = best_metric.bigram_z

    print(f"\n     Best table by bigram_z: {best_table} ({best_bigram_z:.2f})")

    # ── 10. Verdict ──
    consensus_better_than_all = (
        consensus_metrics.bigram_z > phase15_metrics.bigram_z
        and consensus_metrics.bigram_z > signal_metrics.bigram_z
        and consensus_metrics.bigram_z > ppl_metrics.bigram_z
    )
    consensus_better_than_p15 = (
        consensus_metrics.bigram_z > phase15_metrics.bigram_z
        and consensus_metrics.dict_hit >= phase15_metrics.dict_hit
    )

    any_improved = (
        signal_metrics.bigram_z > phase15_metrics.bigram_z
        or ppl_metrics.bigram_z > phase15_metrics.bigram_z
    )

    if consensus_better_than_all and n_consensus_changes > 0:
        verdict = (
            f"CONSENSUS_IMPROVED: consensus table ({n_consensus_changes} changes) "
            f"achieves best bigram_z={consensus_metrics.bigram_z:.2f} "
            f"(dict_hit={consensus_metrics.dict_hit:.4f}), "
            f"surpassing Phase 15 ({phase15_metrics.bigram_z:.2f}), "
            f"SIGNAL ({signal_metrics.bigram_z:.2f}), "
            f"and perplexity ({ppl_metrics.bigram_z:.2f})."
        )
    elif any_improved:
        verdict = (
            f"SINGLE_APPROACH_BEST: {best_table} table is best with "
            f"bigram_z={best_bigram_z:.2f} "
            f"(dict_hit={best_metric.dict_hit:.4f}). "
            f"Consensus ({consensus_metrics.bigram_z:.2f}) did not surpass it. "
            f"Phase 15 bigram_z={phase15_metrics.bigram_z:.2f}."
        )
    else:
        verdict = (
            f"NO_IMPROVEMENT: Phase 15 remains best or tied "
            f"(bigram_z={phase15_metrics.bigram_z:.2f}, "
            f"dict_hit={phase15_metrics.dict_hit:.4f}). "
            f"SIGNAL={signal_metrics.bigram_z:.2f}, "
            f"perplexity={ppl_metrics.bigram_z:.2f}, "
            f"consensus={consensus_metrics.bigram_z:.2f}."
        )

    print(f"\n  Verdict: {verdict}")

    # ── 11. Save ──
    result = PerplexityValidateResult(
        phase15_metrics=_convert(asdict(phase15_metrics)),
        signal_metrics=_convert(asdict(signal_metrics)),
        ppl_metrics=_convert(asdict(ppl_metrics)),
        consensus_metrics=_convert(asdict(consensus_metrics)),
        n_both_agree=n_both_agree,
        n_signal_only=n_signal_only,
        n_ppl_only=n_ppl_only,
        n_conflict=n_conflict,
        n_unchanged=n_unchanged,
        triple_agreements=[_convert(asdict(a)) for a in agreements],
        consensus_assignment=consensus_assignment,
        n_consensus_changes=n_consensus_changes,
        best_table=best_table,
        best_bigram_z=best_bigram_z,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'perplexity_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
