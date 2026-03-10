"""
Step 37.7 – Joint Swap Targeting
==================================
Identify which pairs of unconfirmed triples are most likely to jointly
produce content words if both are simultaneously corrected.

Dependency chain:
    signal_10k.json            (Step 36.2)
    combined_refine.json       (Phase 15)
    tachygraphic_stroke.json   (Phase 19.5)
    bootstrap_10k.json         (Step 36.5, for confirmed triples)
        → joint_target.json    (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, token_to_triples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_joint_target() -> None:
    """Step 37.7: Joint Swap Targeting."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.7: Joint Swap Targeting")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    stroke_data = _safe_load(os.path.join(rd, 'tachygraphic_stroke.json'))
    boot_data = _safe_load(os.path.join(rd, 'bootstrap_10k.json'))

    assignment = refine_data.get('best_assignment', {})
    token_evas = signal_data.get('token_evas', [])
    token_classifications = signal_data.get('token_classifications', [])
    token_hits = signal_data.get('token_hits_10k', [])
    families = stroke_data.get('sign_families', [])

    # Identify confirmed triples from bootstrap
    confirmed_triples = set()
    boot_confirmed = boot_data.get('confirmed_triples', [])
    if isinstance(boot_confirmed, list):
        confirmed_triples = set(boot_confirmed)
    elif isinstance(boot_confirmed, dict):
        confirmed_triples = set(boot_confirmed.keys())

    # If no bootstrap data, try to identify from signal words
    if not confirmed_triples:
        # Fall back: all triples in the assignment are "potentially confirmed"
        # Mark none as confirmed to treat all as unconfirmed
        pass

    all_triples = set(assignment.keys())
    unconfirmed = all_triples - confirmed_triples

    print(f"     {len(all_triples)} total triples")
    print(f"     {len(confirmed_triples)} confirmed triples")
    print(f"     {len(unconfirmed)} unconfirmed triples")

    # ── 2. Find co-occurring unconfirmed pairs ──
    print("  2. Finding co-occurring unconfirmed triple pairs …")
    eva_to_triple = build_eva_to_triple_lookup()

    # For each token, find which unconfirmed triples it contains
    pair_cooccurrence: Counter = Counter()
    pair_miss_count: Counter = Counter()
    pair_token_examples: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for i, eva in enumerate(token_evas):
        triples = token_to_triples(eva, eva_to_triple)
        token_unconfirmed = [t for t in triples if t in unconfirmed]

        if len(token_unconfirmed) >= 2:
            # Record all pairs of unconfirmed triples in this token
            for a in range(len(token_unconfirmed)):
                for b in range(a + 1, len(token_unconfirmed)):
                    pair = tuple(sorted([token_unconfirmed[a], token_unconfirmed[b]]))
                    pair_cooccurrence[pair] += 1
                    # Is this token a SHARED_MISS? (currently not hitting dictionary)
                    if (i < len(token_classifications) and
                            token_classifications[i] == 'SHARED_MISS'):
                        pair_miss_count[pair] += 1
                    if len(pair_token_examples[pair]) < 5:
                        pair_token_examples[pair].append(eva)

    print(f"     {len(pair_cooccurrence)} unique unconfirmed triple pairs found")

    # ── 3. Rank by content-word potential ──
    print("  3. Ranking by content-word potential …")
    ranked_pairs = []
    for pair, co_count in pair_cooccurrence.most_common():
        miss_count = pair_miss_count.get(pair, 0)
        t1, t2 = pair
        s1 = assignment.get(t1, '?')
        s2 = assignment.get(t2, '?')

        # Content word potential: high co-occurrence + high miss rate
        # = tokens that currently fail to match dictionary
        miss_fraction = miss_count / co_count if co_count > 0 else 0.0
        potential_score = co_count * miss_fraction

        ranked_pairs.append({
            'triple1': t1,
            'triple2': t2,
            'syllable1': s1,
            'syllable2': s2,
            'co_occurrence_count': co_count,
            'miss_count': miss_count,
            'miss_fraction': round(miss_fraction, 3),
            'potential_score': round(potential_score, 1),
            'example_tokens': pair_token_examples.get(pair, []),
        })

    ranked_pairs.sort(key=lambda x: x['potential_score'], reverse=True)

    print("     Top 10 pairs by content-word potential:")
    for rp in ranked_pairs[:10]:
        print(f"       {rp['syllable1']}+{rp['syllable2']:<6s} "
              f"co={rp['co_occurrence_count']:>4d} "
              f"miss={rp['miss_count']:>4d} "
              f"({rp['miss_fraction']:.0%}) "
              f"score={rp['potential_score']:.1f}")

    # ── 4. Constrain by family ──
    print("  4. Constraining by sign family …")
    # Build triple → family mapping
    triple_to_family: Dict[str, str] = {}
    for fam in families:
        gc = fam.get('glyph_class', '')
        for member in fam.get('members', []):
            triple_key = eva_to_triple.get(member, '')
            if triple_key:
                triple_to_family[triple_key] = gc

    for rp in ranked_pairs:
        f1 = triple_to_family.get(rp['triple1'], 'unknown')
        f2 = triple_to_family.get(rp['triple2'], 'unknown')
        rp['family1'] = f1
        rp['family2'] = f2
        rp['same_family'] = f1 == f2

    n_same_family = sum(1 for rp in ranked_pairs[:20] if rp['same_family'])
    print(f"     Top 20: {n_same_family} same-family pairs, "
          f"{20 - n_same_family} cross-family pairs")

    # ── 5. Select top 10 for joint swap search ──
    top_10 = ranked_pairs[:10]
    print("  5. Selected top 10 pairs for joint swap search")

    # ── 6. Save ──
    elapsed = time.time() - t0

    output = {
        'n_total_triples': len(all_triples),
        'n_confirmed': len(confirmed_triples),
        'n_unconfirmed': len(unconfirmed),
        'confirmed_triples': sorted(confirmed_triples),
        'unconfirmed_triples': sorted(unconfirmed),
        'n_pairs_found': len(ranked_pairs),
        'ranked_pairs': ranked_pairs[:50],
        'top_10_for_swap': top_10,
        'n_same_family_top20': n_same_family,
        'verdict': (
            f"Joint targeting: {len(unconfirmed)} unconfirmed triples, "
            f"{len(ranked_pairs)} co-occurring pairs. "
            f"Top pair: {top_10[0]['syllable1']}+{top_10[0]['syllable2']} "
            f"(score={top_10[0]['potential_score']:.1f})"
            if top_10 else "No pairs found"
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'joint_target.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
