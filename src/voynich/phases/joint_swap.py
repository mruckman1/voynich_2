"""
Step 37.8 – Exhaustive 2-Triple and Targeted 3-Triple Search
===============================================================
For the top triple pairs, test all joint syllable swaps that produce
content words at 10K.

Dependency chain:
    joint_target.json          (Step 37.7)
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
    signal_10k.json            (Step 36.2)
        → joint_swap.json      (this step)
"""

import json
import os
import random
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
)
from voynich.core.reference import (
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
    load_reference_corpus,
)


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


_FUNCTION_WORDS = {
    'de', 'in', 'ad', 'et', 'cum', 'per', 'pro', 'ex', 'ab', 'sub',
    'non', 'si', 'ut', 'sed', 'ne', 'ac', 'at', 'an', 'se', 'te',
    'me', 'nos', 'iam', 'est', 'sunt', 'hoc', 'id', 'ea', 'is',
    'di', 'du', 'ce', 'ci', 'co', 'cu', 'bi', 'bo', 'be', 'da',
    'la', 'le', 'li', 'lo', 'ni', 'no', 'nu', 'ra', 're', 'ri',
    'ro', 'sa', 'so', 'su', 'ta', 'ti', 'to',
}


def _is_content_word(word: str) -> bool:
    return word.lower() not in _FUNCTION_WORDS and len(word) >= 4


def _generate_candidates(triple_key: str) -> List[str]:
    """Generate candidate syllables for a triple using phoneme maps."""
    parts = triple_key.split(',')
    if len(parts) != 3:
        return []
    first_stroke, last_stroke, _ = parts

    consonants = PHONEME_PLACE_MAP.get(first_stroke, [''])
    vowels = PHONEME_NUCLEUS_MAP.get(last_stroke, ['a', 'e', 'i', 'o'])

    candidates = []
    for c in consonants:
        for v in vowels:
            syl = c + v
            if len(syl) >= 2:
                candidates.append(syl)

    # Also try vowel-only for vowel-initial triples
    for v in vowels:
        if len(v) >= 1:
            candidates.append(v)

    return list(set(candidates))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_joint_swap() -> None:
    """Step 37.8: Exhaustive 2-Triple and Targeted 3-Triple Search."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 37.8: Joint Swap Search")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    target_data = _safe_load(os.path.join(rd, 'joint_target.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    signal_data = _safe_load(os.path.join(rd, 'signal_10k.json'))

    assignment = refine_data.get('best_assignment', {})
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('eva_char') in modifier_chars:
            modifier_rules[c['eva_char']] = 'silent'

    top_pairs = target_data.get('top_10_for_swap', [])
    token_evas = signal_data.get('token_evas', [])
    unconfirmed = set(target_data.get('unconfirmed_triples', []))

    eva_to_triple = build_eva_to_triple_lookup()

    print(f"     {len(top_pairs)} target pairs")
    print(f"     {len(unconfirmed)} unconfirmed triples")

    # Build 10K dictionary
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    word_freq = Counter(w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2)
    dict_10k = set(w for w, _ in word_freq.most_common(10000))

    # Build reference bigrams for z-score validation
    ref_tokens = [w.lower() for w in ref.get_combined_tokens('latin') if len(w) >= 2]
    ref_bigrams = set()
    for i in range(len(ref_tokens) - 1):
        if ref_tokens[i] in dict_10k and ref_tokens[i + 1] in dict_10k:
            ref_bigrams.add((ref_tokens[i], ref_tokens[i + 1]))

    # Use a sample of tokens for speed
    sample_size = min(5000, len(token_evas))
    rng = random.Random(42)
    sample_indices = sorted(rng.sample(range(len(token_evas)), sample_size))
    sample_evas = [token_evas[i] for i in sample_indices]

    def _decode_and_score(table: Dict[str, str]) -> Tuple[float, int, int]:
        """Decode sample, return (dict_hit, content_word_count, content_bigram_count)."""
        decoded = []
        for eva in sample_evas:
            d = decode_token_modifier_aware(
                eva, table, eva_to_triple, modifier_chars, modifier_rules)
            decoded.append(d.lower())
        hits = sum(1 for w in decoded if w in dict_10k)
        content = sum(1 for w in decoded if w in dict_10k and _is_content_word(w))
        # Content-content bigrams
        cc_bigrams = 0
        for i in range(len(decoded) - 1):
            w1, w2 = decoded[i], decoded[i + 1]
            if (w1 in dict_10k and w2 in dict_10k and
                    _is_content_word(w1) and _is_content_word(w2) and
                    (w1, w2) in ref_bigrams):
                cc_bigrams += 1
        return hits / len(decoded), content, cc_bigrams

    # Baseline
    baseline_hit, baseline_content, baseline_cc = _decode_and_score(assignment)
    print(f"     Baseline: hit={baseline_hit:.3%}, content={baseline_content}, "
          f"CC_bigrams={baseline_cc}")

    # ── 2. Exhaustive 2-triple search ──
    print("  2. Exhaustive 2-triple search …")
    pair_results = []

    for pair_idx, pair in enumerate(top_pairs):
        t1 = pair['triple1']
        t2 = pair['triple2']
        cands1 = _generate_candidates(t1)
        cands2 = _generate_candidates(t2)

        if not cands1 or not cands2:
            continue

        best_hit = baseline_hit
        best_content = baseline_content
        best_cc = baseline_cc
        best_s1 = assignment.get(t1, '')
        best_s2 = assignment.get(t2, '')
        n_tested = 0

        for s1 in cands1:
            for s2 in cands2:
                new_table = dict(assignment)
                new_table[t1] = s1
                new_table[t2] = s2
                hit, content, cc = _decode_and_score(new_table)
                n_tested += 1

                if (content > best_content or
                        (content == best_content and hit > best_hit)):
                    best_hit = hit
                    best_content = content
                    best_cc = cc
                    best_s1 = s1
                    best_s2 = s2

        delta_hit = best_hit - baseline_hit
        delta_content = best_content - baseline_content

        pair_results.append({
            'triple1': t1,
            'triple2': t2,
            'original_s1': assignment.get(t1, ''),
            'original_s2': assignment.get(t2, ''),
            'best_s1': best_s1,
            'best_s2': best_s2,
            'n_tested': n_tested,
            'best_hit': round(best_hit, 4),
            'best_content': best_content,
            'best_cc_bigrams': best_cc,
            'delta_hit': round(delta_hit, 4),
            'delta_content': delta_content,
            'improved': delta_content > 0 or delta_hit > 0.005,
        })

        status = "IMPROVED" if delta_content > 0 or delta_hit > 0.005 else "no change"
        print(f"     Pair {pair_idx + 1}: {assignment.get(t1, '')}+{assignment.get(t2, '')} "
              f"→ {best_s1}+{best_s2} [{status}] "
              f"({n_tested} tested, Δcontent={delta_content:+d})")

    # ── 3. 3-triple targeted search on top 3 improved pairs ──
    print("  3. 3-triple targeted search …")
    improved_pairs = [p for p in pair_results if p['improved']]
    three_triple_results = []

    for pair in improved_pairs[:3]:
        t1 = pair['triple1']
        t2 = pair['triple2']
        s1 = pair['best_s1']
        s2 = pair['best_s2']

        # Try adding each remaining unconfirmed triple
        remaining = unconfirmed - {t1, t2}
        for t3 in sorted(remaining):
            cands3 = _generate_candidates(t3)
            if not cands3:
                continue

            best3_content = pair['best_content']
            best3_s3 = assignment.get(t3, '')

            for s3 in cands3:
                new_table = dict(assignment)
                new_table[t1] = s1
                new_table[t2] = s2
                new_table[t3] = s3
                _, content, cc = _decode_and_score(new_table)

                if content > best3_content:
                    best3_content = content
                    best3_s3 = s3

            if best3_content > pair['best_content']:
                three_triple_results.append({
                    'triple1': t1, 'triple2': t2, 'triple3': t3,
                    's1': s1, 's2': s2, 's3': best3_s3,
                    'content': best3_content,
                    'delta_from_pair': best3_content - pair['best_content'],
                })

    print(f"     {len(three_triple_results)} 3-triple improvements found")

    # ── 4. Greedy accumulation ──
    print("  4. Greedy accumulation of independent swaps …")
    accumulated_table = dict(assignment)
    swap_sequence = []

    for pair in sorted(pair_results, key=lambda p: p['delta_content'], reverse=True):
        if not pair['improved']:
            continue

        # Check if these triples are already modified
        t1, t2 = pair['triple1'], pair['triple2']
        if (accumulated_table.get(t1) != assignment.get(t1) or
                accumulated_table.get(t2) != assignment.get(t2)):
            continue  # Already modified by earlier swap

        # Apply swap
        test_table = dict(accumulated_table)
        test_table[t1] = pair['best_s1']
        test_table[t2] = pair['best_s2']
        hit, content, cc = _decode_and_score(test_table)

        if content > baseline_content or hit > baseline_hit + 0.005:
            accumulated_table[t1] = pair['best_s1']
            accumulated_table[t2] = pair['best_s2']
            swap_sequence.append({
                'triple1': t1, 'triple2': t2,
                's1': pair['best_s1'], 's2': pair['best_s2'],
                'cumulative_hit': round(hit, 4),
                'cumulative_content': content,
            })

    # Final score
    final_hit, final_content, final_cc = _decode_and_score(accumulated_table)
    print(f"     {len(swap_sequence)} swaps accumulated")
    print(f"     Final: hit={final_hit:.3%}, content={final_content}, "
          f"CC_bigrams={final_cc}")

    # ── 5. Save ──
    elapsed = time.time() - t0

    output = {
        'n_pairs_tested': len(pair_results),
        'pair_results': pair_results,
        'n_improved_pairs': len(improved_pairs),
        'three_triple_results': three_triple_results,
        'swap_sequence': swap_sequence,
        'n_swaps': len(swap_sequence),
        'accumulated_table': accumulated_table,
        'baseline_hit': round(baseline_hit, 4),
        'baseline_content': baseline_content,
        'final_hit': round(final_hit, 4),
        'final_content': final_content,
        'final_cc_bigrams': final_cc,
        'delta_hit': round(final_hit - baseline_hit, 4),
        'delta_content': final_content - baseline_content,
        'verdict': (
            f"Joint swap: {len(swap_sequence)} swaps, "
            f"hit={final_hit:.3%} (Δ={final_hit - baseline_hit:+.3%}), "
            f"content={final_content} (Δ={final_content - baseline_content:+d}), "
            f"CC={final_cc}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'joint_swap.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
