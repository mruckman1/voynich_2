"""
Step 33.16 – Phase 33 Integration
====================================
Combine all six Phase 33 approaches into a unified assessment.
Build a cross-approach agreement matrix, consensus table, and run final
validation.

The six approaches:
  1+2. Anti-Signal Diagnosis + Signal-Guided Swap  (Steps 33.1–33.4)
  3.   Latin Perplexity Optimization                (Steps 33.5–33.7)
  4.   Suffix-Constrained Root Search               (Steps 33.8–33.9)
  5.   Long Botanical Crib Attack                   (Steps 33.10–33.12)
  6.   Token-Pair Distributional Isomorphism         (Steps 33.13–33.15)

Dependency chain:
    signal_corrected_decode.json   (Step 33.4  — Approach 1+2)
    signal_guided_swap.json        (Step 33.3  — swap details)
    perplexity_validate.json       (Step 33.7  — Approach 3 consensus)
    perplexity_search.json         (Step 33.6  — ppl best_assignment)
    suffix_constrained_search.json (Step 33.9  — Approach 4)
    long_crib_propagate.json       (Step 33.12 — Approach 5, preferred)
    long_crib_csp.json             (Step 33.11 — Approach 5, fallback)
    distributional_validate.json   (Step 33.15 — Approach 6)
    combined_refine.json           (Phase 15   — original baseline)
    modifier_integrate.json        (Phase 16   — modifiers)
    null_corpus.json               (Phase 17   — null seeds)
    signal_bigrams.json            (Phase 29.1 — baseline bigram z)
        -> phase33_integrate.json  (this step)
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


def _load_json(rd: str, filename: str) -> Optional[Dict]:
    """Load a JSON result file, returning None if not found."""
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ApproachSummary:
    approach: str
    ran: bool
    n_changes: int
    dict_hit: float
    signal_rate: float
    bigram_z: float
    key_finding: str


@dataclass
class TripleConsensus:
    triple_key: str
    phase15_syllable: str
    signal_syllable: str
    ppl_syllable: str
    suffix_syllable: str
    crib_syllable: str
    distrib_syllable: str
    n_agree_on_change: int
    consensus_syllable: str
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW', 'UNCHANGED'


@dataclass
class Phase33IntegrateResult:
    # Per-approach
    approach_summaries: List[Dict]
    # Agreement matrix
    triple_consensus: List[Dict]
    n_high_confidence: int
    n_medium_confidence: int
    n_unchanged: int
    # Consensus table
    consensus_assignment: Dict[str, str]
    n_consensus_changes: int
    # Final validation
    consensus_dict_hit: float
    consensus_signal_rate: float
    consensus_bigram_z: float
    baseline_bigram_z: float
    delta_bigram_z: float
    # Progression
    progression: List[Dict]
    # Gap analysis
    unresolved_triples: List[str]
    n_unresolved: int
    # Verdict
    verdict: str  # 'CONSENSUS_IMPROVED', 'SINGLE_APPROACH_BEST', 'TABLE_CONFIRMED'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Extract per-approach assignments from result files
# ---------------------------------------------------------------------------

def _extract_signal_assignment(rd: str) -> Optional[Dict[str, str]]:
    """Extract corrected assignment from signal_guided_swap.json."""
    data = _load_json(rd, 'signal_guided_swap.json')
    if data is None:
        return None
    assignment = data.get('new_assignment')
    if isinstance(assignment, dict) and len(assignment) > 0:
        return assignment
    return None


def _extract_ppl_assignment(rd: str) -> Optional[Dict[str, str]]:
    """Extract perplexity-optimised assignment from perplexity_search.json."""
    data = _load_json(rd, 'perplexity_search.json')
    if data is None:
        return None
    assignment = data.get('best_assignment')
    if isinstance(assignment, dict) and len(assignment) > 0:
        return assignment
    return None


def _extract_suffix_assignment(rd: str) -> Optional[Dict[str, str]]:
    """Extract assignment from suffix_constrained_search.json."""
    data = _load_json(rd, 'suffix_constrained_search.json')
    if data is None:
        return None
    assignment = data.get('best_assignment')
    if isinstance(assignment, dict) and len(assignment) > 0:
        return assignment
    return None


def _extract_crib_assignment(rd: str) -> Optional[Dict[str, str]]:
    """Extract new triple assignments from long-crib results.

    Prefers long_crib_propagate.json; falls back to long_crib_csp.json.
    Returns a PARTIAL dict (only the newly confirmed triples), not a full
    25-triple assignment.
    """
    # Try propagate first
    data = _load_json(rd, 'long_crib_propagate.json')
    if data is not None:
        assignment = data.get('propagated_assignment')
        if isinstance(assignment, dict) and len(assignment) > 0:
            return assignment
        # Fall through to new_confirmed_triples
        new = data.get('new_confirmed_triples')
        if isinstance(new, dict) and len(new) > 0:
            return new

    # Fallback: long_crib_csp.json
    data = _load_json(rd, 'long_crib_csp.json')
    if data is None:
        return None
    new = data.get('new_confirmed_triples')
    if isinstance(new, dict) and len(new) > 0:
        return new
    consistent = data.get('consistent_new_assignments')
    if isinstance(consistent, dict) and len(consistent) > 0:
        return consistent
    return None


def _extract_distrib_proposals(rd: str) -> Optional[Dict[str, str]]:
    """Extract reverse proposals from distributional_validate.json."""
    data = _load_json(rd, 'distributional_validate.json')
    if data is None:
        return None
    proposals = data.get('reverse_proposals')
    if isinstance(proposals, dict) and len(proposals) > 0:
        return proposals
    # Also check hybrid_assignment
    hybrid = data.get('hybrid_assignment')
    if isinstance(hybrid, dict) and len(hybrid) > 0:
        return hybrid
    return None


# ---------------------------------------------------------------------------
# Build approach summaries
# ---------------------------------------------------------------------------

def _build_approach_summaries(rd: str) -> List[ApproachSummary]:
    """Build a summary for each of the 6 approaches."""
    summaries: List[ApproachSummary] = []

    # Approach 1+2: Signal-Guided Swap + Corrected Decode
    sig_decode = _load_json(rd, 'signal_corrected_decode.json')
    sig_swap = _load_json(rd, 'signal_guided_swap.json')
    if sig_decode is not None:
        summaries.append(ApproachSummary(
            approach='1+2: Signal-Guided Swap',
            ran=True,
            n_changes=sig_decode.get('n_swaps_applied', 0),
            dict_hit=sig_decode.get('dict_hit', 0.0),
            signal_rate=sig_decode.get('signal_rate', 0.0),
            bigram_z=sig_decode.get('bigram_z_score', 0.0),
            key_finding=sig_decode.get('verdict', ''),
        ))
    elif sig_swap is not None:
        summaries.append(ApproachSummary(
            approach='1+2: Signal-Guided Swap',
            ran=True,
            n_changes=sig_swap.get('n_swaps_accepted', 0),
            dict_hit=sig_swap.get('new_dict_hit', 0.0),
            signal_rate=0.0,
            bigram_z=0.0,
            key_finding=sig_swap.get('verdict', ''),
        ))
    else:
        summaries.append(ApproachSummary(
            approach='1+2: Signal-Guided Swap',
            ran=False,
            n_changes=0, dict_hit=0.0, signal_rate=0.0, bigram_z=0.0,
            key_finding='NOT_RUN',
        ))

    # Approach 3: Perplexity Optimization
    ppl_val = _load_json(rd, 'perplexity_validate.json')
    ppl_search = _load_json(rd, 'perplexity_search.json')
    if ppl_val is not None:
        cons_metrics = ppl_val.get('consensus_metrics', {})
        summaries.append(ApproachSummary(
            approach='3: Perplexity Optimization',
            ran=True,
            n_changes=ppl_val.get('n_consensus_changes', 0),
            dict_hit=cons_metrics.get('dict_hit', 0.0),
            signal_rate=cons_metrics.get('signal_rate', 0.0),
            bigram_z=cons_metrics.get('bigram_z', 0.0),
            key_finding=ppl_val.get('verdict', ''),
        ))
    elif ppl_search is not None:
        summaries.append(ApproachSummary(
            approach='3: Perplexity Optimization',
            ran=True,
            n_changes=ppl_search.get('n_changes', 0),
            dict_hit=ppl_search.get('optimized_dict_hit', 0.0),
            signal_rate=0.0,
            bigram_z=0.0,
            key_finding=ppl_search.get('verdict', ''),
        ))
    else:
        summaries.append(ApproachSummary(
            approach='3: Perplexity Optimization',
            ran=False,
            n_changes=0, dict_hit=0.0, signal_rate=0.0, bigram_z=0.0,
            key_finding='NOT_RUN',
        ))

    # Approach 4: Suffix-Constrained Search
    suffix_data = _load_json(rd, 'suffix_constrained_search.json')
    if suffix_data is not None:
        summaries.append(ApproachSummary(
            approach='4: Suffix-Constrained Search',
            ran=True,
            n_changes=suffix_data.get('n_changes', 0),
            dict_hit=suffix_data.get('dict_hit', 0.0),
            signal_rate=suffix_data.get('signal_rate', 0.0),
            bigram_z=suffix_data.get('bigram_z', 0.0),
            key_finding=suffix_data.get('verdict', ''),
        ))
    else:
        summaries.append(ApproachSummary(
            approach='4: Suffix-Constrained Search',
            ran=False,
            n_changes=0, dict_hit=0.0, signal_rate=0.0, bigram_z=0.0,
            key_finding='NOT_RUN',
        ))

    # Approach 5: Long Botanical Crib
    crib_prop = _load_json(rd, 'long_crib_propagate.json')
    crib_csp = _load_json(rd, 'long_crib_csp.json')
    crib_data = crib_prop if crib_prop is not None else crib_csp
    if crib_data is not None:
        n_new = len(crib_data.get('new_confirmed_triples', {})
                     if isinstance(crib_data.get('new_confirmed_triples'), dict)
                     else [])
        summaries.append(ApproachSummary(
            approach='5: Long Botanical Crib',
            ran=True,
            n_changes=n_new,
            dict_hit=crib_data.get('dict_hit', 0.0),
            signal_rate=0.0,
            bigram_z=crib_data.get('bigram_z', 0.0),
            key_finding=crib_data.get('verdict', ''),
        ))
    else:
        summaries.append(ApproachSummary(
            approach='5: Long Botanical Crib',
            ran=False,
            n_changes=0, dict_hit=0.0, signal_rate=0.0, bigram_z=0.0,
            key_finding='NOT_RUN',
        ))

    # Approach 6: Distributional Isomorphism
    distrib_val = _load_json(rd, 'distributional_validate.json')
    if distrib_val is not None:
        summaries.append(ApproachSummary(
            approach='6: Distributional Isomorphism',
            ran=True,
            n_changes=len(distrib_val.get('reverse_proposals', {})),
            dict_hit=distrib_val.get('dict_hit', 0.0),
            signal_rate=distrib_val.get('signal_rate', 0.0),
            bigram_z=distrib_val.get('bigram_z', 0.0),
            key_finding=distrib_val.get('verdict', ''),
        ))
    else:
        summaries.append(ApproachSummary(
            approach='6: Distributional Isomorphism',
            ran=False,
            n_changes=0, dict_hit=0.0, signal_rate=0.0, bigram_z=0.0,
            key_finding='NOT_RUN',
        ))

    return summaries


# ---------------------------------------------------------------------------
# Cross-approach agreement matrix and consensus
# ---------------------------------------------------------------------------

def _build_consensus(
    phase15_assignment: Dict[str, str],
    signal_assignment: Optional[Dict[str, str]],
    ppl_assignment: Optional[Dict[str, str]],
    suffix_assignment: Optional[Dict[str, str]],
    crib_assignment: Optional[Dict[str, str]],
    distrib_assignment: Optional[Dict[str, str]],
) -> Tuple[List[TripleConsensus], Dict[str, str]]:
    """Build per-triple consensus and the consensus-corrected table.

    Rules:
      - >=3 approaches agree on a change from Phase 15 -> HIGH confidence, apply
      - >=2 approaches agree AND one is long-crib cross-folio  -> MEDIUM, apply
      - Otherwise keep Phase 15 original -> UNCHANGED
    """
    all_triples = sorted(phase15_assignment.keys())
    consensus_table = dict(phase15_assignment)
    consensus_list: List[TripleConsensus] = []

    for tk in all_triples:
        p15_syl = phase15_assignment.get(tk, '??')

        # Get each approach's recommendation for this triple
        sig_syl = signal_assignment.get(tk, p15_syl) if signal_assignment else p15_syl
        ppl_syl = ppl_assignment.get(tk, p15_syl) if ppl_assignment else p15_syl
        suf_syl = suffix_assignment.get(tk, p15_syl) if suffix_assignment else p15_syl
        # Crib assignment is partial — only has newly confirmed triples
        crib_syl = crib_assignment.get(tk, p15_syl) if crib_assignment else p15_syl
        dist_syl = distrib_assignment.get(tk, p15_syl) if distrib_assignment else p15_syl

        # Count how many approaches propose a change from Phase 15
        changes: Dict[str, List[str]] = defaultdict(list)  # syllable -> [approach_names]
        if sig_syl != p15_syl:
            changes[sig_syl].append('signal')
        if ppl_syl != p15_syl:
            changes[ppl_syl].append('perplexity')
        if suf_syl != p15_syl:
            changes[suf_syl].append('suffix')
        if crib_syl != p15_syl:
            changes[crib_syl].append('crib')
        if dist_syl != p15_syl:
            changes[dist_syl].append('distrib')

        # Find the change with the most agreement
        best_change_syl = p15_syl
        best_change_count = 0
        best_change_approaches: List[str] = []
        for syl, approaches in changes.items():
            if len(approaches) > best_change_count:
                best_change_count = len(approaches)
                best_change_syl = syl
                best_change_approaches = approaches

        # Determine confidence and whether to apply
        confidence = 'UNCHANGED'
        consensus_syl = p15_syl

        if best_change_count >= 3:
            confidence = 'HIGH'
            consensus_syl = best_change_syl
        elif best_change_count >= 2 and 'crib' in best_change_approaches:
            confidence = 'MEDIUM'
            consensus_syl = best_change_syl
        elif best_change_count >= 2:
            # Two approaches agree but neither is crib -- LOW confidence,
            # do not apply (keep Phase 15)
            confidence = 'LOW'
        # else: 0 or 1 approach wants a change -> UNCHANGED

        consensus_table[tk] = consensus_syl

        consensus_list.append(TripleConsensus(
            triple_key=tk,
            phase15_syllable=p15_syl,
            signal_syllable=sig_syl,
            ppl_syllable=ppl_syl,
            suffix_syllable=suf_syl,
            crib_syllable=crib_syl,
            distrib_syllable=dist_syl,
            n_agree_on_change=best_change_count,
            consensus_syllable=consensus_syl,
            confidence=confidence,
        ))

    return consensus_list, consensus_table


# ---------------------------------------------------------------------------
# Signal classification and bigram z computation
# ---------------------------------------------------------------------------

def _classify_tokens(
    real_hits: List[bool],
    null_hits_lists: List[List[bool]],
    n_tokens: int,
) -> Tuple[int, int, List[str]]:
    """Classify every token.

    Returns (n_signal, n_anti_signal, classifications).
    """
    n_signal = 0
    n_anti_signal = 0
    classifications: List[str] = []

    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_lists if nh[idx])

        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
            n_signal += 1
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
            n_anti_signal += 1
        else:
            classifications.append('SHARED_MISS')

    return n_signal, n_anti_signal, classifications


def _compute_bigram_z(
    classifications: List[str],
    decoded: List[str],
    folios: List[str],
    ref_bigrams: Set[Tuple[str, str]],
    n_perms: int = 1000,
    seed: int = 42,
) -> float:
    """Compute bigram z-score for SIGNAL-SIGNAL consecutive pairs."""
    n_tokens = len(classifications)
    n_signal = sum(1 for c in classifications if c == 'SIGNAL')

    n_pairs = 0
    n_hits = 0
    for i in range(n_tokens - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and folios[i] == folios[i + 1]):
            n_pairs += 1
            if (decoded[i], decoded[i + 1]) in ref_bigrams:
                n_hits += 1

    hit_rate = n_hits / n_pairs if n_pairs > 0 else 0.0

    if n_pairs == 0 or n_signal < 2:
        return 0.0

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
    null_var = (
        sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)
        if null_rates else 0.0
    )
    null_std = null_var ** 0.5

    if null_std > 0:
        return (hit_rate - null_mean) / null_std
    return float('inf') if hit_rate > null_mean else 0.0


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------

def _validate_assignment(
    assignment: Dict[str, str],
    all_tokens: List[str],
    token_folios: List[str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    ref_bigrams: Set[Tuple[str, str]],
    null_seeds: List[int],
) -> Tuple[float, float, float]:
    """Decode corpus, classify, compute bigram z.

    Returns (dict_hit, signal_rate, bigram_z).
    """
    n_tokens = len(all_tokens)

    # Decode real corpus
    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]
    dict_hit = sum(real_hits) / n_tokens if n_tokens > 0 else 0.0

    # Regenerate null corpora
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )
    null_hits_lists: List[List[bool]] = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_hits_lists.append([w in ref_word_set for w in null_decoded])

    # Classify
    n_signal, n_anti_signal, classifications = _classify_tokens(
        real_hits, null_hits_lists, n_tokens,
    )
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0

    # Bigram z
    bigram_z = _compute_bigram_z(
        classifications, real_decoded, token_folios,
        ref_bigrams, n_perms=1000, seed=42,
    )
    if bigram_z == float('inf'):
        bigram_z = 999.0

    return dict_hit, signal_rate, bigram_z


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase33_integrate() -> None:
    """Step 33.16: Integrate all Phase 33 approaches into a unified assessment."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 33.16: Phase 33 Integration")
    print("=" * 70)

    rd = _results_dir()

    # ==================================================================
    # 1. Load results from all 6 approaches
    # ==================================================================
    print("\n  1. Loading results from all 6 approaches ...")

    sig_decode = _load_json(rd, 'signal_corrected_decode.json')
    sig_swap = _load_json(rd, 'signal_guided_swap.json')
    ppl_val = _load_json(rd, 'perplexity_validate.json')
    ppl_search = _load_json(rd, 'perplexity_search.json')
    suffix_data = _load_json(rd, 'suffix_constrained_search.json')
    crib_prop = _load_json(rd, 'long_crib_propagate.json')
    crib_csp = _load_json(rd, 'long_crib_csp.json')
    distrib_val = _load_json(rd, 'distributional_validate.json')
    baseline_data = _load_json(rd, 'combined_refine.json')

    file_status = {
        'signal_corrected_decode.json': sig_decode is not None,
        'signal_guided_swap.json': sig_swap is not None,
        'perplexity_validate.json': ppl_val is not None,
        'perplexity_search.json': ppl_search is not None,
        'suffix_constrained_search.json': suffix_data is not None,
        'long_crib_propagate.json': crib_prop is not None,
        'long_crib_csp.json': crib_csp is not None,
        'distributional_validate.json': distrib_val is not None,
        'combined_refine.json': baseline_data is not None,
    }

    for fname, found in file_status.items():
        marker = '[OK]' if found else '[MISS]'
        print(f"     {marker:6s} {fname}")

    n_loaded = sum(1 for v in file_status.values() if v)
    print(f"\n     Loaded {n_loaded}/{len(file_status)} result files")

    # Phase 15 baseline assignment (required)
    if baseline_data is None:
        print("  [ERROR] combined_refine.json not found — cannot proceed")
        return
    phase15_assignment = dict(baseline_data.get('best_assignment', {}))
    if not phase15_assignment:
        print("  [ERROR] No best_assignment in combined_refine.json")
        return
    print(f"     Phase 15 baseline: {len(phase15_assignment)} triples")

    # ==================================================================
    # 2. Per-approach verdict table
    # ==================================================================
    print("\n  2. Per-approach verdict table ...")

    summaries = _build_approach_summaries(rd)

    print(f"\n     {'Approach':<35s} {'Ran':>4s} {'Changes':>8s} "
          f"{'Dict-Hit':>9s} {'Signal':>8s} {'Bigram z':>9s}")
    print(f"     {'-' * 35} {'-' * 4} {'-' * 8} "
          f"{'-' * 9} {'-' * 8} {'-' * 9}")
    for s in summaries:
        ran_str = 'YES' if s.ran else 'NO'
        dh = f"{s.dict_hit:.4f}" if s.ran else '—'
        sr = f"{s.signal_rate:.4f}" if s.ran and s.signal_rate > 0 else '—'
        bz = f"{s.bigram_z:.2f}" if s.ran and s.bigram_z > 0 else '—'
        print(f"     {s.approach:<35s} {ran_str:>4s} {s.n_changes:>8d} "
              f"{dh:>9s} {sr:>8s} {bz:>9s}")
        if s.ran and s.key_finding:
            finding = s.key_finding[:70]
            print(f"       -> {finding}")

    n_approaches_ran = sum(1 for s in summaries if s.ran)
    print(f"\n     {n_approaches_ran}/6 approaches ran")

    # ==================================================================
    # 3. Cross-approach agreement matrix (25 x 6)
    # ==================================================================
    print("\n  3. Building cross-approach agreement matrix ...")

    signal_assignment = _extract_signal_assignment(rd)
    ppl_assignment = _extract_ppl_assignment(rd)
    suffix_assignment = _extract_suffix_assignment(rd)
    crib_assignment = _extract_crib_assignment(rd)
    distrib_assignment = _extract_distrib_proposals(rd)

    approach_assignments = {
        'signal': signal_assignment,
        'perplexity': ppl_assignment,
        'suffix': suffix_assignment,
        'crib': crib_assignment,
        'distrib': distrib_assignment,
    }

    n_have = sum(1 for v in approach_assignments.values() if v is not None)
    print(f"     Extracted assignments from {n_have}/5 approaches")

    for name, asgn in approach_assignments.items():
        if asgn is not None:
            n_diff = sum(
                1 for tk in phase15_assignment
                if asgn.get(tk, phase15_assignment[tk]) != phase15_assignment[tk]
            )
            print(f"       {name:12s}: {n_diff} changes from Phase 15")
        else:
            print(f"       {name:12s}: NOT AVAILABLE")

    # ==================================================================
    # 4. Build consensus-corrected table
    # ==================================================================
    print("\n  4. Building consensus-corrected table ...")

    consensus_list, consensus_assignment = _build_consensus(
        phase15_assignment,
        signal_assignment,
        ppl_assignment,
        suffix_assignment,
        crib_assignment,
        distrib_assignment,
    )

    n_high = sum(1 for c in consensus_list if c.confidence == 'HIGH')
    n_medium = sum(1 for c in consensus_list if c.confidence == 'MEDIUM')
    n_low = sum(1 for c in consensus_list if c.confidence == 'LOW')
    n_unchanged = sum(1 for c in consensus_list if c.confidence == 'UNCHANGED')
    n_consensus_changes = sum(
        1 for c in consensus_list
        if c.consensus_syllable != c.phase15_syllable
    )

    print(f"     HIGH confidence (>=3 agree):   {n_high}")
    print(f"     MEDIUM confidence (2+crib):    {n_medium}")
    print(f"     LOW confidence (2 agree):      {n_low}")
    print(f"     UNCHANGED:                     {n_unchanged}")
    print(f"     Total consensus changes:       {n_consensus_changes}")

    # Print the agreement matrix
    print(f"\n     {'Triple':>35s} {'P15':>5s} {'SIG':>5s} {'PPL':>5s} "
          f"{'SUF':>5s} {'CRB':>5s} {'DST':>5s} {'#Chg':>5s} {'Cons':>5s} {'Conf':>10s}")
    print(f"     {'-' * 35} {'-' * 5} {'-' * 5} {'-' * 5} "
          f"{'-' * 5} {'-' * 5} {'-' * 5} {'-' * 5} {'-' * 5} {'-' * 10}")
    for c in consensus_list:
        # Highlight changes from Phase 15
        sig_m = '*' if c.signal_syllable != c.phase15_syllable else ' '
        ppl_m = '*' if c.ppl_syllable != c.phase15_syllable else ' '
        suf_m = '*' if c.suffix_syllable != c.phase15_syllable else ' '
        crb_m = '*' if c.crib_syllable != c.phase15_syllable else ' '
        dst_m = '*' if c.distrib_syllable != c.phase15_syllable else ' '
        con_m = '!' if c.consensus_syllable != c.phase15_syllable else ' '

        print(f"     {c.triple_key:>35s} {c.phase15_syllable:>5s} "
              f"{sig_m}{c.signal_syllable:>4s} {ppl_m}{c.ppl_syllable:>4s} "
              f"{suf_m}{c.suffix_syllable:>4s} {crb_m}{c.crib_syllable:>4s} "
              f"{dst_m}{c.distrib_syllable:>4s} {c.n_agree_on_change:>5d} "
              f"{con_m}{c.consensus_syllable:>4s} {c.confidence:>10s}")

    # ==================================================================
    # 5. Final readability test on consensus table
    # ==================================================================
    print("\n  5. Final readability test on consensus table ...")

    # Load baseline bigram z from Phase 29
    bg_data = _load_json(rd, 'signal_bigrams.json')
    baseline_bigram_z = 6.14
    if bg_data is not None:
        baseline_bigram_z = bg_data.get('bigram_z_score', 6.14)

    if n_consensus_changes == 0:
        # No changes — report Phase 15 baseline metrics directly
        print("     No consensus changes; using Phase 15 baseline metrics")

        # Load Phase 28/29 baseline metrics
        baseline_dict_hit = 0.436
        baseline_signal_rate = 0.165
        if sig_decode is not None:
            baseline_dict_hit = sig_decode.get('baseline_dict_hit', 0.436)
            baseline_signal_rate = sig_decode.get('baseline_signal_rate', 0.165)

        consensus_dict_hit = baseline_dict_hit
        consensus_signal_rate = baseline_signal_rate
        consensus_bigram_z = baseline_bigram_z
    else:
        # Run full validation on the consensus table
        print("     Loading corpus and reference data ...")

        # Load modifier rules
        mod_data = _load_json(rd, 'modifier_integrate.json')
        if mod_data is None:
            print("  [ERROR] modifier_integrate.json not found — cannot validate")
            consensus_dict_hit = 0.0
            consensus_signal_rate = 0.0
            consensus_bigram_z = 0.0
        else:
            modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

            # Null seeds
            null_seeds = [100, 101, 102, 103, 104]
            null_data = _load_json(rd, 'null_corpus.json')
            if null_data is not None:
                null_seeds = [
                    r['seed'] for r in null_data.get('null_runs', [])
                ]

            # Reference word set
            ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
            base_words = set(
                w.lower() for w in ref_corpus.get_combined_tokens('latin')
                if len(w) >= 2
            )
            expanded, _ = build_expanded_word_set(base_words)
            ref_word_set = base_words | expanded

            # Reference bigrams
            ref_tokens = [
                w.lower() for w in ref_corpus.get_combined_tokens('latin')
                if len(w) >= 2
            ]
            ref_bigrams: Set[Tuple[str, str]] = set()
            for i in range(len(ref_tokens) - 1):
                ref_bigrams.add((ref_tokens[i], ref_tokens[i + 1]))

            # Corpus
            corpus = load_corpus(verbose=False)
            eva_to_triple = build_eva_to_triple_lookup()

            all_tokens: List[str] = []
            token_folios: List[str] = []
            for folio, page in corpus.pages.items():
                for token in page.all_tokens:
                    all_tokens.append(token)
                    token_folios.append(folio)

            print(f"     {len(all_tokens)} tokens, "
                  f"{len(ref_word_set)} ref words, "
                  f"{len(ref_bigrams)} ref bigrams")

            print("     Validating consensus table ...")
            consensus_dict_hit, consensus_signal_rate, consensus_bigram_z = \
                _validate_assignment(
                    consensus_assignment,
                    all_tokens, token_folios,
                    eva_to_triple,
                    modifier_chars, modifier_rules,
                    ref_word_set, ref_bigrams,
                    null_seeds,
                )

    delta_bigram_z = consensus_bigram_z - baseline_bigram_z

    print(f"\n     Consensus dict_hit:    {consensus_dict_hit:.4f}")
    print(f"     Consensus signal_rate: {consensus_signal_rate:.4f}")
    print(f"     Consensus bigram_z:    {consensus_bigram_z:.2f}")
    print(f"     Baseline bigram_z:     {baseline_bigram_z:.2f}")
    print(f"     Delta bigram_z:        {delta_bigram_z:+.2f}")

    # ==================================================================
    # 6. Progression table
    # ==================================================================
    print("\n  6. Progression table ...")

    progression = [
        {
            'phase': '16',
            'dict_hit': 0.436,
            'signal': '—',
            'bigram_z': '—',
            'note': 'Full-corpus baseline',
        },
        {
            'phase': '29',
            'dict_hit': 0.436,
            'signal': '16.5%',
            'bigram_z': '6.14',
            'note': 'SIGNAL bigram discovery',
        },
        {
            'phase': '30',
            'dict_hit': 0.436,
            'signal': '16.5%',
            'bigram_z': '6.14',
            'note': '2 words confirmed',
        },
        {
            'phase': '33',
            'dict_hit': round(consensus_dict_hit, 4),
            'signal': f"{consensus_signal_rate:.1%}",
            'bigram_z': f"{consensus_bigram_z:.2f}",
            'note': f"{n_consensus_changes} consensus changes",
        },
    ]

    print(f"     {'Phase':>6s} | {'Dict-Hit':>9s} | {'Signal':>8s} | "
          f"{'Bigram z':>9s} | Note")
    print(f"     {'-' * 6} | {'-' * 9} | {'-' * 8} | {'-' * 9} | {'-' * 30}")
    for p in progression:
        dh = p['dict_hit']
        dh_str = f"{dh:.4f}" if isinstance(dh, float) else str(dh)
        print(f"     {p['phase']:>6s} | {dh_str:>9s} | {p['signal']:>8s} | "
              f"{p['bigram_z']:>9s} | {p['note']}")

    # ==================================================================
    # 7. Gap analysis
    # ==================================================================
    print("\n  7. Gap analysis ...")

    # Unresolved triples: those where no approach made a recommendation
    # (all approaches either didn't run or kept Phase 15's assignment)
    unresolved = []
    for c in consensus_list:
        if c.n_agree_on_change == 0:
            unresolved.append(c.triple_key)

    print(f"     {len(unresolved)}/{len(consensus_list)} triples unresolved "
          f"(no approach recommended a change)")
    if unresolved:
        for tk in unresolved:
            p15_syl = phase15_assignment.get(tk, '??')
            print(f"       {tk}: stays as '{p15_syl}'")

    # ==================================================================
    # 8. Verdict
    # ==================================================================
    print("\n  8. Verdict ...")

    # Determine which single approach had the best bigram_z
    best_approach = None
    best_approach_z = baseline_bigram_z
    for s in summaries:
        if s.ran and s.bigram_z > best_approach_z:
            best_approach = s.approach
            best_approach_z = s.bigram_z

    if n_consensus_changes > 0 and consensus_bigram_z > baseline_bigram_z + 0.5:
        verdict = (
            f"CONSENSUS_IMPROVED: {n_consensus_changes} consensus changes "
            f"raised bigram z from {baseline_bigram_z:.2f} to "
            f"{consensus_bigram_z:.2f} ({delta_bigram_z:+.2f}). "
            f"dict_hit={consensus_dict_hit:.4f}, "
            f"signal_rate={consensus_signal_rate:.4f}. "
            f"{n_high} high-confidence, {n_medium} medium-confidence changes."
        )
    elif best_approach is not None and best_approach_z > baseline_bigram_z + 0.5:
        verdict = (
            f"SINGLE_APPROACH_BEST: No consensus improvement, but "
            f"'{best_approach}' achieved bigram z={best_approach_z:.2f} "
            f"(baseline={baseline_bigram_z:.2f}). "
            f"Consensus had {n_consensus_changes} changes but "
            f"z={consensus_bigram_z:.2f}."
        )
    else:
        verdict = (
            f"TABLE_CONFIRMED: Phase 15 table is confirmed as best available. "
            f"{n_approaches_ran}/6 approaches ran; "
            f"{n_consensus_changes} consensus changes; "
            f"bigram z={consensus_bigram_z:.2f} vs baseline {baseline_bigram_z:.2f}. "
            f"{len(unresolved)}/25 triples unresolved."
        )

    print(f"\n     {verdict}")

    # ==================================================================
    # 9. Save
    # ==================================================================
    result = Phase33IntegrateResult(
        approach_summaries=[_convert(asdict(s)) for s in summaries],
        triple_consensus=[_convert(asdict(c)) for c in consensus_list],
        n_high_confidence=n_high,
        n_medium_confidence=n_medium,
        n_unchanged=n_unchanged,
        consensus_assignment=consensus_assignment,
        n_consensus_changes=n_consensus_changes,
        consensus_dict_hit=round(consensus_dict_hit, 6),
        consensus_signal_rate=round(consensus_signal_rate, 6),
        consensus_bigram_z=round(consensus_bigram_z, 2),
        baseline_bigram_z=round(baseline_bigram_z, 2),
        delta_bigram_z=round(delta_bigram_z, 2),
        progression=[_convert(p) for p in progression],
        unresolved_triples=unresolved,
        n_unresolved=len(unresolved),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase33_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"PHASE 33 VERDICT")
    print(f"{'=' * 70}")
    print(f"  {verdict}")
    print(f"\n  Approaches: {n_approaches_ran}/6 ran")
    for s in summaries:
        status = 'RAN' if s.ran else 'SKIP'
        finding_short = s.key_finding[:60] if s.key_finding else '—'
        print(f"    [{status}] {s.approach}: {finding_short}")
    print(f"\n  Agreement matrix:")
    print(f"    HIGH (>=3 agree):    {n_high}")
    print(f"    MEDIUM (2+crib):     {n_medium}")
    print(f"    LOW (2 agree):       {n_low}")
    print(f"    UNCHANGED:           {n_unchanged}")
    print(f"\n  Consensus: {n_consensus_changes} changes applied")
    print(f"  dict_hit:    {consensus_dict_hit:.4f}")
    print(f"  signal_rate: {consensus_signal_rate:.4f}")
    print(f"  bigram_z:    {consensus_bigram_z:.2f} "
          f"(baseline={baseline_bigram_z:.2f}, delta={delta_bigram_z:+.2f})")
    print(f"\n  Unresolved:  {len(unresolved)}/25 triples")
    print(f"\n  Saved -> {out_path}")
    print(f"  ({elapsed:.1f}s)")
