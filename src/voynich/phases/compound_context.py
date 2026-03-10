"""
Phase 32.4 – Compound-Sign Context Analysis
==============================================
Re-run PMI context analysis on the compound-sign decoded corpus with
the expanded signal vocabulary.

Dependency chain:
    compound_bigrams.json      (Step 32.3 — per-token cache)
    compound_signal.json       (Step 32.2 — signal words)
    signal_context.json        (Phase 29 baseline)
        → compound_context.json (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.compound_decode import SUFFIX_ENDING_MAP


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
# Main
# ---------------------------------------------------------------------------

def run_compound_context() -> None:
    """Step 32.4: Compound-sign context analysis."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 32.4: Compound-Sign Context Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")
    with open(os.path.join(rd, 'compound_bigrams.json')) as f:
        cb = json.load(f)
    with open(os.path.join(rd, 'compound_signal.json')) as f:
        cs = json.load(f)

    decoded = cb['token_decoded']
    classifications = cb['token_classifications']
    folios = cb['token_folios']
    dict_hits = cb['token_dict_hits']
    n_tokens = cb['n_tokens']

    # Get signal words (sigma > 2.0)
    signal_words = [ws['word'] for ws in cs['word_signals']
                    if ws.get('is_genuine', False)]
    print(f"     {len(signal_words)} signal words, {n_tokens} tokens")

    # Build reference word set
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Build reference bigrams for confirmed-confirmed pair check
    ref_tokens = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                  if len(w) >= 2]
    ref_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(ref_tokens) - 1):
        ref_bigrams.add((ref_tokens[i], ref_tokens[i + 1]))

    # ── 2. Context windows ──
    print("\n  2. Extracting context windows ...")
    word_freq = Counter(decoded)
    total_tokens = n_tokens

    pair_freq: Counter = Counter()
    for i in range(n_tokens - 1):
        if folios[i] == folios[i + 1]:
            pair_freq[(decoded[i], decoded[i + 1])] += 1
    total_pairs = sum(pair_freq.values())

    context_windows = []
    for sw in signal_words:
        left_counts: Counter = Counter()
        right_counts: Counter = Counter()
        n_occ = 0

        for i in range(n_tokens):
            if decoded[i] == sw and classifications[i] == 'SIGNAL':
                n_occ += 1
                if i > 0 and folios[i - 1] == folios[i]:
                    left_counts[decoded[i - 1]] += 1
                if i < n_tokens - 1 and folios[i] == folios[i + 1]:
                    right_counts[decoded[i + 1]] += 1

        top_left = []
        for word, count in left_counts.most_common(15):
            p_pair = pair_freq.get((word, sw), 0) / total_pairs if total_pairs else 0
            p_w = word_freq[word] / total_tokens
            p_sw = word_freq[sw] / total_tokens
            pmi = math.log2(p_pair / (p_w * p_sw)) if p_pair > 0 and p_w > 0 and p_sw > 0 else 0.0
            top_left.append({
                'word': word, 'count': count, 'pmi': round(pmi, 3),
                'is_dict_hit': word in ref_word_set,
            })

        top_right = []
        for word, count in right_counts.most_common(15):
            p_pair = pair_freq.get((sw, word), 0) / total_pairs if total_pairs else 0
            p_w = word_freq[word] / total_tokens
            p_sw = word_freq[sw] / total_tokens
            pmi = math.log2(p_pair / (p_w * p_sw)) if p_pair > 0 and p_w > 0 and p_sw > 0 else 0.0
            top_right.append({
                'word': word, 'count': count, 'pmi': round(pmi, 3),
                'is_dict_hit': word in ref_word_set,
            })

        all_neighbors = list(left_counts.elements()) + list(right_counts.elements())
        ctx_hits = sum(1 for w in all_neighbors if w in ref_word_set)
        ctx_rate = ctx_hits / len(all_neighbors) if all_neighbors else 0.0

        context_windows.append({
            'signal_word': sw, 'n_occurrences': n_occ,
            'top_left': top_left, 'top_right': top_right,
            'context_dict_hit_rate': round(ctx_rate, 4),
        })

        if n_occ > 0:
            print(f"     {sw:12s} occ={n_occ:4d} ctx_hit={ctx_rate:.1%}")

    # ── 3. New crib candidates ──
    print("\n  3. Identifying new crib candidates ...")
    signal_word_set = set(signal_words)
    word_evidence: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)

    for cw in context_windows:
        for neighbor in cw['top_left'] + cw['top_right']:
            word = neighbor['word']
            if word in signal_word_set:
                continue
            if neighbor['is_dict_hit']:
                word_evidence[word].append((
                    cw['signal_word'], neighbor['pmi'], neighbor['count'],
                ))

    cribs = []
    for word, evidence_list in word_evidence.items():
        assoc_sws = set(e[0] for e in evidence_list)
        if len(assoc_sws) < 2:
            continue
        mean_pmi = sum(e[1] for e in evidence_list) / len(evidence_list)
        if mean_pmi < 0.5:
            continue
        total_count = sum(e[2] for e in evidence_list)
        cribs.append({
            'word': word,
            'n_signal_word_associations': len(assoc_sws),
            'mean_pmi': round(mean_pmi, 3),
            'total_count': total_count,
            'is_dict_hit': True,
        })

    cribs.sort(key=lambda c: (-c['n_signal_word_associations'], -c['mean_pmi']))
    print(f"     {len(cribs)} new crib candidates")
    for c in cribs[:10]:
        print(f"       {c['word']:12s} assoc={c['n_signal_word_associations']} "
              f"PMI={c['mean_pmi']:.2f} count={c['total_count']}")

    # ── 4. Chain extension ──
    print("\n  4. Finding chains (≥3 dict-hit tokens with ≥1 SIGNAL) ...")
    chains = []
    i = 0
    while i < n_tokens:
        if not dict_hits[i]:
            i += 1
            continue
        start = i
        folio = folios[i]
        while i < n_tokens and dict_hits[i] and folios[i] == folio:
            i += 1
        length = i - start
        if length >= 3:
            n_sig = sum(1 for j in range(start, i)
                        if classifications[j] == 'SIGNAL')
            if n_sig >= 1:
                chains.append({
                    'words': decoded[start:i],
                    'folio': folio,
                    'start_idx': start,
                    'length': length,
                    'n_signal': n_sig,
                })

    chains.sort(key=lambda c: -c['length'])
    longest = chains[0]['length'] if chains else 0
    print(f"     {len(chains)} chains, longest = {longest}")
    for ch in chains[:5]:
        print(f"       {ch['folio']} len={ch['length']} sig={ch['n_signal']}: "
              f"{' '.join(ch['words'][:8])}")

    # ── 5. Confirmed-confirmed inflected pairs ──
    print("\n  5. Confirmed-confirmed inflected pair check ...")
    inflected_matches = []
    for i in range(n_tokens - 1):
        if (classifications[i] == 'SIGNAL' and
                classifications[i + 1] == 'SIGNAL' and
                folios[i] == folios[i + 1]):
            w1, w2 = decoded[i], decoded[i + 1]
            if (w1, w2) in ref_bigrams:
                inflected_matches.append({
                    'folio': folios[i], 'pos': i,
                    'w1': w1, 'w2': w2, 'match_type': 'exact',
                })

    print(f"     {len(inflected_matches)} confirmed-confirmed inflected bigram matches")
    for m in inflected_matches[:10]:
        print(f"       {m['folio']} pos={m['pos']}: {m['w1']} {m['w2']}")

    # ── 6. Load Phase 29 baseline ──
    phase29_path = os.path.join(rd, 'signal_context.json')
    phase29_n_cribs = 0
    phase29_n_chains = 0
    if os.path.exists(phase29_path):
        with open(phase29_path) as f:
            p29 = json.load(f)
        phase29_n_cribs = p29.get('n_new_crib_candidates', 0)
        phase29_n_chains = p29.get('n_chains_found', 0)

    # ── 7. Save ──
    print("\n  7. Saving compound_context.json ...")
    output = {
        'context_windows': context_windows,
        'n_new_crib_candidates': len(cribs),
        'new_crib_candidates': cribs[:50],
        'n_chains_found': len(chains),
        'longest_chain': longest,
        'chain_candidates': [_convert(ch) for ch in chains[:50]],
        'inflected_confirmed_pairs': inflected_matches,
        'n_inflected_pairs': len(inflected_matches),
        'phase29_n_cribs': phase29_n_cribs,
        'phase29_n_chains': phase29_n_chains,
        'delta_cribs': len(cribs) - phase29_n_cribs,
        'delta_chains': len(chains) - phase29_n_chains,
        'gate_passed': len(cribs) > 0 or len(chains) > 0,
        'verdict': (f"{len(cribs)} crib candidates, {len(chains)} chains "
                    f"(longest={longest}), {len(inflected_matches)} inflected pairs"),
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(rd, 'compound_context.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Step 32.4 completed in {time.time() - t0:.1f}s")
