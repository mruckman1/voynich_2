"""
Phase 47 Track D – Manuscript Sequence Analysis
=================================================
Test whether internal evidence from decoded text suggests page misordering.
Folio-to-folio vocabulary overlap, cross-folio continuity, anomalous
boundaries, and local reordering tests.

Dependency chain:
    signal_bigrams.json          (Phase 29 parallel arrays)
    signal_10k.json              (Phase 36 signal words)
    final_decode_summary.json    (Phase 46 folio summaries)
        -> seq_overlap.json      (Step 47D.1)
        -> seq_continuity.json   (Step 47D.2)
        -> seq_boundaries.json   (Step 47D.3)
        -> seq_reorder.json      (Step 47D.4)
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _load_folio_sections() -> Dict[str, str]:
    corpus = load_corpus(verbose=False)
    return {
        folio: (page.section if hasattr(page, 'section') else 'unknown')
        for folio, page in corpus.pages.items()
    }


# ---------------------------------------------------------------------------
# Step 47D.1 — Folio-to-folio vocabulary overlap
# ---------------------------------------------------------------------------

@dataclass
class SeqOverlapResult:
    n_folios: int
    mean_consecutive_jaccard: float
    mean_same_section_jaccard: float
    mean_cross_section_jaccard: float
    consecutive_vs_random_ratio: float
    top_similar_pairs: List[Dict]
    per_section_stats: Dict[str, Dict[str, float]]
    runtime_seconds: float


def run_seq_overlap() -> None:
    """Step 47D.1: folio-to-folio SIGNAL vocabulary overlap matrix."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47D.1: Folio-to-Folio Vocabulary Overlap")
    print("=" * 70)

    rd = _results_dir()

    # Load parallel arrays
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_decoded = sb.get('token_decoded', [])
    token_classifications = sb.get('token_classifications', [])

    if not token_folios:
        print("  [SKIP] No data")
        return

    folio_sections = _load_folio_sections()

    # Build per-folio SIGNAL word sets
    folio_signal_words: Dict[str, set] = defaultdict(set)
    for i in range(len(token_folios)):
        if token_classifications[i] == 'SIGNAL':
            folio_signal_words[token_folios[i]].add(token_decoded[i])

    folio_order = list(dict.fromkeys(token_folios))
    n_folios = len(folio_order)
    print(f"\n  {n_folios} folios")

    # Compute Jaccard for all pairs (226^2/2 ~ 25K pairs — fast)
    consecutive_jaccards = []
    same_section_jaccards = []
    cross_section_jaccards = []
    all_jaccards = []
    top_pairs = []

    for i in range(n_folios):
        for j in range(i + 1, n_folios):
            f1, f2 = folio_order[i], folio_order[j]
            jac = _jaccard(folio_signal_words[f1], folio_signal_words[f2])
            all_jaccards.append(jac)

            s1 = folio_sections.get(f1, 'unknown')
            s2 = folio_sections.get(f2, 'unknown')

            if j == i + 1:
                consecutive_jaccards.append(jac)
            if s1 == s2:
                same_section_jaccards.append(jac)
            else:
                cross_section_jaccards.append(jac)

            # Track top similar non-consecutive pairs
            if j > i + 1 and jac > 0.3:
                top_pairs.append({
                    'folio_a': f1, 'folio_b': f2,
                    'jaccard': round(jac, 4),
                    'same_section': s1 == s2,
                    'distance': j - i,
                })

    top_pairs.sort(key=lambda x: -x['jaccard'])
    top_pairs = top_pairs[:20]

    mean_consec = sum(consecutive_jaccards) / len(consecutive_jaccards) if consecutive_jaccards else 0.0
    mean_same = sum(same_section_jaccards) / len(same_section_jaccards) if same_section_jaccards else 0.0
    mean_cross = sum(cross_section_jaccards) / len(cross_section_jaccards) if cross_section_jaccards else 0.0
    mean_all = sum(all_jaccards) / len(all_jaccards) if all_jaccards else 0.0
    consec_vs_random = mean_consec / mean_all if mean_all > 0 else 0.0

    print(f"\n  Mean Jaccard:")
    print(f"    Consecutive:   {mean_consec:.4f}")
    print(f"    Same section:  {mean_same:.4f}")
    print(f"    Cross section: {mean_cross:.4f}")
    print(f"    All pairs:     {mean_all:.4f}")
    print(f"    Consecutive/random ratio: {consec_vs_random:.2f}x")

    # Per-section internal vs external
    per_section_stats: Dict[str, Dict[str, float]] = {}
    for section in set(folio_sections.values()):
        sec_folios = [f for f in folio_order if folio_sections.get(f) == section]
        internal = []
        external = []
        for i, f1 in enumerate(sec_folios):
            for j, f2 in enumerate(sec_folios):
                if j <= i:
                    continue
                internal.append(_jaccard(folio_signal_words[f1], folio_signal_words[f2]))
        for f1 in sec_folios:
            for f2 in folio_order:
                if folio_sections.get(f2) != section:
                    external.append(_jaccard(folio_signal_words[f1], folio_signal_words[f2]))
        per_section_stats[section] = {
            'n_folios': len(sec_folios),
            'mean_internal': round(sum(internal) / len(internal), 4) if internal else 0.0,
            'mean_external': round(sum(external) / len(external), 4) if external else 0.0,
        }

    result = SeqOverlapResult(
        n_folios=n_folios,
        mean_consecutive_jaccard=round(mean_consec, 4),
        mean_same_section_jaccard=round(mean_same, 4),
        mean_cross_section_jaccard=round(mean_cross, 4),
        consecutive_vs_random_ratio=round(consec_vs_random, 4),
        top_similar_pairs=top_pairs,
        per_section_stats=per_section_stats,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'seq_overlap.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47D.2 — Cross-folio word continuity
# ---------------------------------------------------------------------------

@dataclass
class SeqContinuityResult:
    n_boundaries: int
    n_plausible_joins: int
    plausibility_rate: float
    n_signal_continuous: int
    signal_continuity_rate: float
    within_section_plausibility: float
    between_section_plausibility: float
    boundary_details: List[Dict]
    runtime_seconds: float


def run_seq_continuity() -> None:
    """Step 47D.2: cross-folio word continuity at boundaries."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47D.2: Cross-Folio Word Continuity")
    print("=" * 70)

    rd = _results_dir()

    # Load parallel arrays
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_decoded = sb.get('token_decoded', [])
    token_classifications = sb.get('token_classifications', [])

    if not token_folios:
        print("  [SKIP] No data")
        return

    folio_sections = _load_folio_sections()

    # Build reference bigram set
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens_raw = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ]
    ref_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens_raw) - 1):
        ref_bigrams.add((ref_tokens_raw[i], ref_tokens_raw[i + 1]))

    # Group tokens by folio
    folio_token_indices: Dict[str, List[int]] = defaultdict(list)
    for i, f in enumerate(token_folios):
        folio_token_indices[f].append(i)

    folio_order = list(dict.fromkeys(token_folios))
    n_boundaries = len(folio_order) - 1

    boundary_details = []
    n_plausible = 0
    n_signal_cont = 0
    within_plaus = []
    between_plaus = []

    for bi in range(n_boundaries):
        f_a = folio_order[bi]
        f_b = folio_order[bi + 1]
        idx_a = folio_token_indices[f_a]
        idx_b = folio_token_indices[f_b]

        last_5 = [token_decoded[i] for i in idx_a[-5:]] if len(idx_a) >= 5 else [token_decoded[i] for i in idx_a]
        first_5 = [token_decoded[i] for i in idx_b[:5]] if len(idx_b) >= 5 else [token_decoded[i] for i in idx_b]

        # Check boundary bigram
        bigram_plausible = False
        if last_5 and first_5:
            bigram_plausible = (last_5[-1], first_5[0]) in ref_bigrams

        # Signal continuity: any SIGNAL in last 3 of A or first 3 of B
        last_3_cls = [token_classifications[i] for i in idx_a[-3:]] if len(idx_a) >= 3 else [token_classifications[i] for i in idx_a]
        first_3_cls = [token_classifications[i] for i in idx_b[:3]] if len(idx_b) >= 3 else [token_classifications[i] for i in idx_b]
        signal_continuous = (
            'SIGNAL' in last_3_cls and 'SIGNAL' in first_3_cls
        )

        if bigram_plausible:
            n_plausible += 1
        if signal_continuous:
            n_signal_cont += 1

        same_section = folio_sections.get(f_a) == folio_sections.get(f_b)
        if same_section:
            within_plaus.append(1 if bigram_plausible else 0)
        else:
            between_plaus.append(1 if bigram_plausible else 0)

        boundary_details.append({
            'folio_a': f_a,
            'folio_b': f_b,
            'section_a': folio_sections.get(f_a, 'unknown'),
            'section_b': folio_sections.get(f_b, 'unknown'),
            'same_section': same_section,
            'last_5': last_5,
            'first_5': first_5,
            'bigram_plausible': bigram_plausible,
            'signal_continuous': signal_continuous,
        })

    plaus_rate = n_plausible / n_boundaries if n_boundaries else 0.0
    sig_rate = n_signal_cont / n_boundaries if n_boundaries else 0.0
    within_rate = sum(within_plaus) / len(within_plaus) if within_plaus else 0.0
    between_rate = sum(between_plaus) / len(between_plaus) if between_plaus else 0.0

    print(f"\n  {n_boundaries} folio boundaries")
    print(f"  Plausible joins: {n_plausible} ({plaus_rate:.1%})")
    print(f"  Signal continuous: {n_signal_cont} ({sig_rate:.1%})")
    print(f"  Within-section plausibility: {within_rate:.1%}")
    print(f"  Between-section plausibility: {between_rate:.1%}")

    result = SeqContinuityResult(
        n_boundaries=n_boundaries,
        n_plausible_joins=n_plausible,
        plausibility_rate=round(plaus_rate, 4),
        n_signal_continuous=n_signal_cont,
        signal_continuity_rate=round(sig_rate, 4),
        within_section_plausibility=round(within_rate, 4),
        between_section_plausibility=round(between_rate, 4),
        boundary_details=boundary_details,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'seq_continuity.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47D.3 — Anomalous boundaries
# ---------------------------------------------------------------------------

@dataclass
class SeqBoundaryResult:
    n_boundaries: int
    anomaly_threshold: float
    n_anomalous: int
    within_section_anomalies: List[Dict]
    between_section_anomalies: List[Dict]
    mean_within_score: float
    mean_between_score: float
    runtime_seconds: float


def run_seq_boundary() -> None:
    """Step 47D.3: detect anomalous sequence boundaries."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47D.3: Anomalous Sequence Boundaries")
    print("=" * 70)

    rd = _results_dir()

    # Load continuity data
    cont = _safe_load(os.path.join(rd, 'seq_continuity.json'))
    boundaries = cont.get('boundary_details', [])
    if not boundaries:
        print("  [SKIP] No continuity data")
        return

    # Load overlap data for Jaccard
    overlap = _safe_load(os.path.join(rd, 'seq_overlap.json'))

    # Build per-boundary continuity score
    # Score = 0.5 * bigram_plausible + 0.3 * signal_continuous + 0.2 * word_overlap
    # (word_overlap approximated by shared SIGNAL words — from overlap matrix indirectly)
    for bd in boundaries:
        score = 0.0
        if bd.get('bigram_plausible'):
            score += 0.5
        if bd.get('signal_continuous'):
            score += 0.3
        # Shared words between last_5 and first_5
        shared = len(set(bd.get('last_5', [])) & set(bd.get('first_5', [])))
        if shared > 0:
            score += 0.2
        bd['continuity_score'] = round(score, 3)

    # Separate within-section and between-section
    within = [b for b in boundaries if b.get('same_section')]
    between = [b for b in boundaries if not b.get('same_section')]

    # Z-score within each category
    def _zscore_boundaries(bds: List[Dict]) -> List[Dict]:
        scores = [b['continuity_score'] for b in bds]
        if len(scores) < 2:
            return []
        mean_s = sum(scores) / len(scores)
        std_s = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5
        anomalies = []
        for b in bds:
            z = (b['continuity_score'] - mean_s) / std_s if std_s > 0 else 0.0
            if z < -1.5:  # below 1.5 std
                anomalies.append({
                    'folio_a': b['folio_a'],
                    'folio_b': b['folio_b'],
                    'section_a': b['section_a'],
                    'section_b': b['section_b'],
                    'continuity_score': b['continuity_score'],
                    'z_score': round(z, 2),
                })
        return anomalies

    within_anomalies = _zscore_boundaries(within)
    between_anomalies = _zscore_boundaries(between)

    mean_within = sum(b['continuity_score'] for b in within) / len(within) if within else 0.0
    mean_between = sum(b['continuity_score'] for b in between) / len(between) if between else 0.0

    n_anom = len(within_anomalies) + len(between_anomalies)

    print(f"\n  Within-section boundaries: {len(within)} (mean score={mean_within:.3f})")
    print(f"  Between-section boundaries: {len(between)} (mean score={mean_between:.3f})")
    print(f"  Anomalous (z < -1.5): {n_anom}")
    print(f"    Within-section anomalies: {len(within_anomalies)}")
    print(f"    Between-section anomalies: {len(between_anomalies)}")

    if within_anomalies:
        print("\n  Within-section anomalies (unexpected breaks):")
        for a in within_anomalies[:5]:
            print(f"    {a['folio_a']} -> {a['folio_b']}  "
                  f"score={a['continuity_score']:.3f}  z={a['z_score']:.2f}")

    result = SeqBoundaryResult(
        n_boundaries=len(boundaries),
        anomaly_threshold=-1.5,
        n_anomalous=n_anom,
        within_section_anomalies=within_anomalies,
        between_section_anomalies=between_anomalies,
        mean_within_score=round(mean_within, 4),
        mean_between_score=round(mean_between, 4),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'seq_boundaries.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 47D.4 — Local reordering test
# ---------------------------------------------------------------------------

@dataclass
class SeqReorderResult:
    n_anomalous_tested: int
    swaps_tested: List[Dict]
    n_improvements: int
    best_swap: Optional[Dict]
    verdict: str
    runtime_seconds: float


def run_seq_reorder() -> None:
    """Step 47D.4: test local reorderings at anomalous boundaries."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 47D.4: Local Reordering Test")
    print("=" * 70)

    rd = _results_dir()

    # Load boundary data
    bd_data = _safe_load(os.path.join(rd, 'seq_boundaries.json'))
    within_anomalies = bd_data.get('within_section_anomalies', [])

    # Load overlap data for finding high-overlap partners
    overlap_data = _safe_load(os.path.join(rd, 'seq_overlap.json'))
    top_pairs = overlap_data.get('top_similar_pairs', [])

    # Load continuity data for boundary scoring
    cont = _safe_load(os.path.join(rd, 'seq_continuity.json'))
    boundary_details = cont.get('boundary_details', [])

    # Load parallel arrays for decoded words
    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_folios = sb.get('token_folios', [])
    token_decoded = sb.get('token_decoded', [])
    token_classifications = sb.get('token_classifications', [])

    folio_sections = _load_folio_sections()

    if not within_anomalies:
        print("\n  No within-section anomalies found. No reordering needed.")
        result = SeqReorderResult(
            n_anomalous_tested=0,
            swaps_tested=[],
            n_improvements=0,
            best_swap=None,
            verdict='NO_REORDER',
            runtime_seconds=round(time.time() - t0, 2),
        )
        out_path = os.path.join(rd, 'seq_reorder.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(asdict(result)), f, indent=2)
        print(f"\n  Saved -> {out_path}")
        return

    # Build per-folio SIGNAL word sets for overlap computation
    folio_signal_words: Dict[str, set] = defaultdict(set)
    for i in range(len(token_folios)):
        if token_classifications[i] == 'SIGNAL':
            folio_signal_words[token_folios[i]].add(token_decoded[i])

    folio_order = list(dict.fromkeys(token_folios))
    folio_idx = {f: i for i, f in enumerate(folio_order)}

    # For each anomalous boundary, test swapping with nearest high-overlap partner
    swaps_tested = []
    n_improvements = 0

    for anom in within_anomalies[:10]:  # cap at 10
        f_a = anom['folio_a']
        f_b = anom['folio_b']
        section = anom.get('section_a', 'unknown')

        # Find best swap candidate: folio in same section with highest overlap to f_b
        section_folios = [f for f in folio_order if folio_sections.get(f) == section]
        best_swap_folio = None
        best_swap_jac = 0.0

        for candidate in section_folios:
            if candidate in (f_a, f_b):
                continue
            jac = _jaccard(folio_signal_words[candidate], folio_signal_words[f_b])
            if jac > best_swap_jac:
                best_swap_jac = jac
                best_swap_folio = candidate

        if best_swap_folio is None:
            continue

        # Score original boundary
        original_score = anom.get('continuity_score', 0.0)

        # Score swapped boundary (approximation: use Jaccard overlap)
        swapped_jac = _jaccard(folio_signal_words[best_swap_folio], folio_signal_words[f_b])
        original_jac = _jaccard(folio_signal_words[f_a], folio_signal_words[f_b])
        improvement = swapped_jac - original_jac

        entry = {
            'folio_a': f_a,
            'folio_b': f_b,
            'swap_candidate': best_swap_folio,
            'original_jaccard': round(original_jac, 4),
            'swapped_jaccard': round(swapped_jac, 4),
            'improvement': round(improvement, 4),
            'beneficial': improvement > 0.1,
        }
        swaps_tested.append(entry)
        if improvement > 0.1:
            n_improvements += 1

    best_swap = None
    if swaps_tested:
        best_swap = max(swaps_tested, key=lambda x: x['improvement'])

    if n_improvements >= 2:
        verdict = 'REORDER_BENEFICIAL'
    elif n_improvements >= 1:
        verdict = 'REORDER_MARGINAL'
    else:
        verdict = 'NO_REORDER'

    print(f"\n  Anomalies tested: {len(swaps_tested)}")
    print(f"  Improvements (>0.1 Jaccard): {n_improvements}")
    print(f"  Verdict: {verdict}")
    if best_swap:
        print(f"  Best swap: {best_swap['folio_a']} -> {best_swap['swap_candidate']} "
              f"(improvement={best_swap['improvement']:.4f})")

    result = SeqReorderResult(
        n_anomalous_tested=len(swaps_tested),
        swaps_tested=swaps_tested,
        n_improvements=n_improvements,
        best_swap=best_swap,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'seq_reorder.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Track D orchestrator
# ---------------------------------------------------------------------------

def run_track_d_47() -> None:
    """Run all Track D steps."""
    run_seq_overlap()
    print()
    run_seq_continuity()
    print()
    run_seq_boundary()
    print()
    run_seq_reorder()
