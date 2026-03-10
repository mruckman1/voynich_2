"""
Phase 33.2 – Per-Triple SIGNAL vs ANTI_SIGNAL Rates
=====================================================
Computes a continuous confidence score for every triple assignment by
measuring how often each triple appears in SIGNAL tokens versus
ANTI_SIGNAL tokens.  This tells us which triples are "helping"
(contributing to genuine Latin) and which are "hurting" (generating
dictionary collisions).

Dependency chain:
    signal_bigrams.json        (Phase 29 — per-token classifications)
    combined_refine.json       (Phase 15 — best_assignment)
    bootstrap_loop.json        (Phase 30 — confirmed/unconfirmed triples)
    anti_signal_diagnosis.json (Step 33.1, optional — confirmed status)
        → triple_signal_rates.json (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TripleSignalProfile:
    triple_key: str
    assigned_syllable: str
    is_confirmed: bool
    total_tokens: int
    signal_tokens: int
    anti_signal_tokens: int
    shared_hit_tokens: int
    shared_miss_tokens: int
    signal_rate: float
    anti_rate: float
    net_signal: float
    confidence_rank: int
    # Positional rates
    initial_signal_rate: float
    medial_signal_rate: float
    final_signal_rate: float


@dataclass
class TripleSignalRatesResult:
    n_triples: int
    triple_profiles: List[Dict]
    # Comparison
    confirmed_mean_net_signal: float
    unconfirmed_mean_net_signal: float
    # Interaction matrix (top pairs)
    top_interactions: List[Dict]  # (triple_a, triple_b, joint_signal_rate, n_co_occur)
    # Verdict
    n_positive_net: int  # triples with net_signal > 0
    n_negative_net: int  # triples with net_signal < 0
    swap_candidates: List[str]  # triples with net_signal < -0.2 and not confirmed
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_triple_signal_rates() -> None:
    """Step 33.2: Per-triple SIGNAL vs ANTI_SIGNAL rates."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 33.2: Per-Triple SIGNAL vs ANTI_SIGNAL Rates")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading input files ...")

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        sb = json.load(f)
    token_evas = sb['token_evas']
    token_classifications = sb['token_classifications']
    n_tokens = sb['n_tokens']
    print(f"     signal_bigrams.json: {n_tokens} tokens, "
          f"{sb['n_signal']} SIGNAL")

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        cr = json.load(f)
    best_assignment: Dict[str, str] = cr['best_assignment']
    print(f"     combined_refine.json: {len(best_assignment)} triple assignments")

    with open(os.path.join(rd, 'bootstrap_loop.json')) as f:
        bl = json.load(f)
    confirmed_triple_list: List[str] = bl.get('confirmed_triples', [])
    confirmed_set: Set[str] = set(confirmed_triple_list)
    print(f"     bootstrap_loop.json: {len(confirmed_set)} confirmed triples")

    # Optional: load anti_signal_diagnosis.json for refined confirmed status
    diag_path = os.path.join(rd, 'anti_signal_diagnosis.json')
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            diag = json.load(f)
        # Override confirmed status from diagnosis if available
        triple_diagnoses = diag.get('triple_diagnoses', [])
        if triple_diagnoses:
            confirmed_set = set()
            for td in triple_diagnoses:
                if td.get('confirmed', False):
                    confirmed_set.add(td['triple_key'])
            print(f"     anti_signal_diagnosis.json: overriding to "
                  f"{len(confirmed_set)} confirmed triples")
    else:
        print("     anti_signal_diagnosis.json: not found (using bootstrap_loop)")

    # ── 2. Map tokens to triples ──
    print("\n  2. Mapping tokens to triples ...")
    eva_to_triple = build_eva_to_triple_lookup()
    all_triple_keys: Set[str] = set(best_assignment.keys())

    # Per-triple accumulators
    triple_total: Counter = Counter()
    triple_signal: Counter = Counter()
    triple_anti: Counter = Counter()
    triple_shared_hit: Counter = Counter()
    triple_shared_miss: Counter = Counter()

    # Positional accumulators: {triple_key: [total, signal]}
    pos_initial_total: Counter = Counter()
    pos_initial_signal: Counter = Counter()
    pos_medial_total: Counter = Counter()
    pos_medial_signal: Counter = Counter()
    pos_final_total: Counter = Counter()
    pos_final_signal: Counter = Counter()

    # Co-occurrence accumulators: {(triple_a, triple_b): [total, signal]}
    pair_total: Counter = Counter()
    pair_signal: Counter = Counter()

    n_mapped = 0
    n_unmapped = 0

    for idx in range(n_tokens):
        token = token_evas[idx]
        cls = token_classifications[idx]
        triples = token_to_triples(token, eva_to_triple)

        if not triples:
            n_unmapped += 1
            continue
        n_mapped += 1

        # Deduplicate triples within this token for counting
        unique_triples = set(triples)
        for tk in unique_triples:
            triple_total[tk] += 1
            if cls == 'SIGNAL':
                triple_signal[tk] += 1
            elif cls == 'ANTI_SIGNAL':
                triple_anti[tk] += 1
            elif cls == 'SHARED_HIT':
                triple_shared_hit[tk] += 1
            elif cls == 'SHARED_MISS':
                triple_shared_miss[tk] += 1

        # Positional analysis
        n_triples = len(triples)
        for pos_idx, tk in enumerate(triples):
            if n_triples == 1:
                # Single triple: count as initial only
                pos_initial_total[tk] += 1
                if cls == 'SIGNAL':
                    pos_initial_signal[tk] += 1
            elif n_triples == 2:
                if pos_idx == 0:
                    pos_initial_total[tk] += 1
                    if cls == 'SIGNAL':
                        pos_initial_signal[tk] += 1
                else:
                    pos_final_total[tk] += 1
                    if cls == 'SIGNAL':
                        pos_final_signal[tk] += 1
            else:
                if pos_idx == 0:
                    pos_initial_total[tk] += 1
                    if cls == 'SIGNAL':
                        pos_initial_signal[tk] += 1
                elif pos_idx == n_triples - 1:
                    pos_final_total[tk] += 1
                    if cls == 'SIGNAL':
                        pos_final_signal[tk] += 1
                else:
                    pos_medial_total[tk] += 1
                    if cls == 'SIGNAL':
                        pos_medial_signal[tk] += 1

        # Co-occurrence pairs (sorted to avoid duplicates)
        unique_sorted = sorted(unique_triples)
        for i in range(len(unique_sorted)):
            for j in range(i + 1, len(unique_sorted)):
                pair_key = (unique_sorted[i], unique_sorted[j])
                pair_total[pair_key] += 1
                if cls == 'SIGNAL':
                    pair_signal[pair_key] += 1

    print(f"     Mapped: {n_mapped} tokens, unmapped: {n_unmapped}")
    print(f"     Triples with data: {len(triple_total)}")

    # ── 3. Compute per-triple profiles ──
    print("\n  3. Computing per-triple signal profiles ...")
    profiles: List[TripleSignalProfile] = []

    for tk in sorted(all_triple_keys):
        total = triple_total.get(tk, 0)
        sig = triple_signal.get(tk, 0)
        anti = triple_anti.get(tk, 0)
        sh = triple_shared_hit.get(tk, 0)
        sm = triple_shared_miss.get(tk, 0)

        signal_rate = sig / total if total > 0 else 0.0
        anti_rate = anti / total if total > 0 else 0.0
        net_signal = signal_rate - anti_rate

        # Positional rates
        init_total = pos_initial_total.get(tk, 0)
        init_sig = pos_initial_signal.get(tk, 0)
        initial_sr = init_sig / init_total if init_total > 0 else 0.0

        med_total = pos_medial_total.get(tk, 0)
        med_sig = pos_medial_signal.get(tk, 0)
        medial_sr = med_sig / med_total if med_total > 0 else 0.0

        fin_total = pos_final_total.get(tk, 0)
        fin_sig = pos_final_signal.get(tk, 0)
        final_sr = fin_sig / fin_total if fin_total > 0 else 0.0

        profiles.append(TripleSignalProfile(
            triple_key=tk,
            assigned_syllable=best_assignment.get(tk, '?'),
            is_confirmed=tk in confirmed_set,
            total_tokens=total,
            signal_tokens=sig,
            anti_signal_tokens=anti,
            shared_hit_tokens=sh,
            shared_miss_tokens=sm,
            signal_rate=round(signal_rate, 6),
            anti_rate=round(anti_rate, 6),
            net_signal=round(net_signal, 6),
            confidence_rank=0,  # filled below
            initial_signal_rate=round(initial_sr, 6),
            medial_signal_rate=round(medial_sr, 6),
            final_signal_rate=round(final_sr, 6),
        ))

    # Assign confidence ranks (1 = best net_signal)
    profiles.sort(key=lambda p: -p.net_signal)
    for rank, p in enumerate(profiles, 1):
        p.confidence_rank = rank

    print(f"     {'Triple':<42s} {'Syl':>4s} {'Conf':>5s} "
          f"{'Total':>6s} {'SIG':>5s} {'ANTI':>5s} {'Net':>8s} {'Rank':>5s}")
    print(f"     {'─' * 42} {'─' * 4} {'─' * 5} "
          f"{'─' * 6} {'─' * 5} {'─' * 5} {'─' * 8} {'─' * 5}")
    for p in profiles:
        conf_mark = "Y" if p.is_confirmed else "N"
        print(f"     {p.triple_key:<42s} {p.assigned_syllable:>4s} "
              f"{conf_mark:>5s} {p.total_tokens:>6d} "
              f"{p.signal_tokens:>5d} {p.anti_signal_tokens:>5d} "
              f"{p.net_signal:>+8.4f} {p.confidence_rank:>5d}")

    # ── 4. Confirmed vs unconfirmed comparison ──
    print("\n  4. Confirmed vs unconfirmed comparison ...")
    confirmed_nets = [p.net_signal for p in profiles if p.is_confirmed]
    unconfirmed_nets = [p.net_signal for p in profiles if not p.is_confirmed]

    confirmed_mean = (
        sum(confirmed_nets) / len(confirmed_nets)
        if confirmed_nets else 0.0
    )
    unconfirmed_mean = (
        sum(unconfirmed_nets) / len(unconfirmed_nets)
        if unconfirmed_nets else 0.0
    )

    print(f"     Confirmed   ({len(confirmed_nets):2d} triples): "
          f"mean net_signal = {confirmed_mean:+.4f}")
    print(f"     Unconfirmed ({len(unconfirmed_nets):2d} triples): "
          f"mean net_signal = {unconfirmed_mean:+.4f}")
    print(f"     Delta (confirmed - unconfirmed): "
          f"{confirmed_mean - unconfirmed_mean:+.4f}")

    # ── 5. Positional signal breakdown ──
    print("\n  5. Positional signal rates ...")
    print(f"     {'Triple':<42s} {'Init':>7s} {'Med':>7s} {'Final':>7s}")
    print(f"     {'─' * 42} {'─' * 7} {'─' * 7} {'─' * 7}")
    for p in profiles:
        print(f"     {p.triple_key:<42s} "
              f"{p.initial_signal_rate:>7.3f} "
              f"{p.medial_signal_rate:>7.3f} "
              f"{p.final_signal_rate:>7.3f}")

    # ── 6. Interaction effects ──
    print("\n  6. Computing interaction effects (co-occurring pairs) ...")
    interactions: List[Dict] = []
    for (ta, tb), total in pair_total.most_common():
        if total < 10:
            continue
        sig = pair_signal.get((ta, tb), 0)
        joint_sr = sig / total if total > 0 else 0.0
        interactions.append({
            'triple_a': ta,
            'triple_b': tb,
            'joint_signal_rate': round(joint_sr, 6),
            'n_co_occur': total,
            'n_joint_signal': sig,
        })

    # Sort by joint signal rate descending
    interactions.sort(key=lambda x: -x['joint_signal_rate'])
    top_interactions = interactions[:30]

    print(f"     Pairs with >= 10 co-occurrences: {len(interactions)}")
    print(f"\n     Top 10 by joint signal rate:")
    for ix in top_interactions[:10]:
        print(f"       {ix['triple_a']:<38s} + "
              f"{ix['triple_b']:<38s}  "
              f"rate={ix['joint_signal_rate']:.3f} "
              f"(n={ix['n_co_occur']})")

    # ── 7. Identify swap candidates and verdict ──
    print("\n  7. Verdict ...")
    n_positive = sum(1 for p in profiles if p.net_signal > 0)
    n_negative = sum(1 for p in profiles if p.net_signal < 0)
    n_zero = sum(1 for p in profiles if p.net_signal == 0)

    swap_candidates = [
        p.triple_key for p in profiles
        if p.net_signal < -0.02 and not p.is_confirmed
    ]

    print(f"     Positive net_signal: {n_positive}")
    print(f"     Negative net_signal: {n_negative}")
    print(f"     Zero net_signal:     {n_zero}")
    print(f"     Swap candidates (net < -0.02, unconfirmed): "
          f"{len(swap_candidates)}")
    if swap_candidates:
        for sc in swap_candidates:
            p_obj = next(p for p in profiles if p.triple_key == sc)
            print(f"       {sc} = {p_obj.assigned_syllable} "
                  f"(net={p_obj.net_signal:+.4f})")

    if len(swap_candidates) == 0:
        verdict = "ALL_STABLE"
    elif len(swap_candidates) <= 3:
        verdict = "FEW_SWAPS"
    else:
        verdict = "MANY_SWAPS"

    verdict_msg = (
        f"{verdict}: {n_positive}/{len(profiles)} triples positive, "
        f"{len(swap_candidates)} swap candidates. "
        f"Confirmed mean={confirmed_mean:+.4f}, "
        f"unconfirmed mean={unconfirmed_mean:+.4f}."
    )
    print(f"\n     {verdict_msg}")

    # ── 8. Save output ──
    print("\n  8. Saving triple_signal_rates.json ...")

    result = TripleSignalRatesResult(
        n_triples=len(profiles),
        triple_profiles=[_convert(asdict(p)) for p in profiles],
        confirmed_mean_net_signal=round(confirmed_mean, 6),
        unconfirmed_mean_net_signal=round(unconfirmed_mean, 6),
        top_interactions=top_interactions,
        n_positive_net=n_positive,
        n_negative_net=n_negative,
        swap_candidates=swap_candidates,
        verdict=verdict_msg,
        runtime_seconds=round(time.time() - t0, 1),
    )

    out_path = os.path.join(rd, 'triple_signal_rates.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
    print(f"\n  Step 33.2 completed in {time.time() - t0:.1f}s")
