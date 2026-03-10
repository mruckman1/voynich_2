"""
Step 38.7 – Macaronic Concatenation Test
=========================================
Re-run Phase 37's concatenation test matching against the merged dictionary
instead of the 17K Latin dictionary. Selective merging to preserve bigram
structure.

Dependency chain:
    merged_signal.json         (Step 38.3)
    merged_dict.json           (Step 38.1)
    pair_concat.json           (Step 37.4 — confirmed pairs)
    decode_10k.json            (Step 36.1)
        → merged_concat.json   (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir


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
# Core functions
# ---------------------------------------------------------------------------

def _expanded_pair_pool(
    signal_words: Set[str],
    decoded_lower: List[str],
    token_folios: List[str],
    classifications: List[str],
) -> List[Tuple[str, str, int]]:
    """Build expanded pair pool including Italian-only signal pairs."""
    pairs = []
    n = len(decoded_lower)
    for i in range(n - 1):
        if i >= len(token_folios) - 1:
            break
        if token_folios[i] != token_folios[i + 1]:
            continue
        w1, w2 = decoded_lower[i], decoded_lower[i + 1]
        if (w1 in signal_words and w2 in signal_words and
            classifications[i] == 'SIGNAL' and classifications[i + 1] == 'SIGNAL'):
            pairs.append((w1, w2, i))
    return pairs


def _concat_against_merged(
    pairs: List[Tuple[str, str, int]],
    merged_dict: Set[str],
    latin_10k: Set[str],
    italian_10k: Set[str],
) -> Tuple[List[Dict], List[Dict]]:
    """Concatenate signal pairs and match against merged dict."""
    all_matches = []
    new_italian = []
    pair_counts = Counter((w1, w2) for w1, w2, _ in pairs)

    for (w1, w2), count in pair_counts.items():
        concat = w1 + w2
        if concat in merged_dict:
            in_lat = concat in latin_10k
            in_ita = concat in italian_10k
            source = ('SHARED' if in_lat and in_ita
                      else 'LATIN_ONLY' if in_lat
                      else 'ITALIAN_ONLY')

            function_words = {'de', 'in', 'se', 'ne', 'ad', 'la', 'le', 'di',
                             'da', 'si', 'cum', 'per', 'pro', 'sub', 'que', 'et'}
            is_content = len(concat) >= 4 and concat not in function_words

            entry = {
                'w1': w1,
                'w2': w2,
                'concatenated': concat,
                'source': source,
                'is_content': is_content,
                'pair_count': count,
            }
            all_matches.append(entry)
            if source == 'ITALIAN_ONLY':
                new_italian.append(entry)

    all_matches.sort(key=lambda x: x['pair_count'], reverse=True)
    return all_matches, new_italian


def _selective_merge_rules(
    matches: List[Dict],
    min_freq: int = 3,
    min_length: int = 5,
) -> List[Dict]:
    """Filter merge rules: content words only, freq ≥ 3, length ≥ 5."""
    rules = []
    for m in matches:
        if (m['is_content'] and
            m['pair_count'] >= min_freq and
            len(m['concatenated']) >= min_length):
            rules.append({
                'w1': m['w1'],
                'w2': m['w2'],
                'merged': m['concatenated'],
                'source': m['source'],
                'count': m['pair_count'],
            })
    return rules


def _retokenize_selective(
    decoded_lower: List[str],
    token_folios: List[str],
    classifications: List[str],
    merge_rules: List[Dict],
) -> Tuple[List[str], List[str], List[str]]:
    """Apply selective merges to produce a new token stream."""
    merge_map = {(r['w1'], r['w2']): r['merged'] for r in merge_rules}
    new_decoded = []
    new_folios = []
    new_class = []
    n = len(decoded_lower)
    i = 0

    while i < n:
        if i < n - 1:
            pair = (decoded_lower[i], decoded_lower[i + 1])
            folio_match = (i < len(token_folios) - 1 and
                          token_folios[i] == token_folios[i + 1])
            if pair in merge_map and folio_match:
                new_decoded.append(merge_map[pair])
                new_folios.append(token_folios[i])
                # Merged token inherits SIGNAL if either was SIGNAL
                if (classifications[i] == 'SIGNAL' or
                    classifications[i + 1] == 'SIGNAL'):
                    new_class.append('SIGNAL')
                else:
                    new_class.append(classifications[i])
                i += 2
                continue
        new_decoded.append(decoded_lower[i])
        new_folios.append(token_folios[i] if i < len(token_folios) else 'unknown')
        new_class.append(classifications[i] if i < len(classifications) else 'UNKNOWN')
        i += 1

    return new_decoded, new_folios, new_class


def _bigram_test_selective(
    new_decoded: List[str],
    new_folios: List[str],
    new_class: List[str],
    bigram_set: Set[Tuple[str, str]],
    merged_dict: Set[str],
    latin_10k: Set[str],
    italian_10k: Set[str],
    n_perms: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run bigram test on selectively merged stream."""
    # Find SIGNAL-SIGNAL pairs in merged stream
    pairs = []
    signal_words_set = set()
    n = len(new_decoded)

    for i in range(n - 1):
        if i >= len(new_folios) - 1:
            break
        if new_folios[i] != new_folios[i + 1]:
            continue
        if new_class[i] == 'SIGNAL' and new_class[i + 1] == 'SIGNAL':
            pairs.append((new_decoded[i], new_decoded[i + 1]))
            signal_words_set.add(new_decoded[i])
            signal_words_set.add(new_decoded[i + 1])

    exact_hits = sum(1 for w1, w2 in pairs if (w1, w2) in bigram_set)

    # Content-content bigrams
    function_words = {'de', 'in', 'se', 'ne', 'ad', 'la', 'le', 'di',
                     'da', 'si', 'cum', 'per', 'pro', 'sub', 'que', 'et'}
    cc_hits = 0
    cc_catalog = []
    for w1, w2 in pairs:
        if (w1, w2) in bigram_set:
            w1_content = len(w1) >= 3 and w1 not in function_words
            w2_content = len(w2) >= 3 and w2 not in function_words
            if w1_content and w2_content:
                cc_hits += 1
                cc_catalog.append({'w1': w1, 'w2': w2})

    # Null permutation
    rng = random.Random(seed)
    signal_list = list(signal_words_set)
    null_counts = []
    for _ in range(n_perms):
        shuffled = list(signal_list) * 5
        rng.shuffle(shuffled)
        nc = sum(1 for j in range(len(shuffled) - 1)
                if (shuffled[j], shuffled[j + 1]) in bigram_set)
        null_counts.append(nc)

    null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
    null_var = (sum((c - null_mean) ** 2 for c in null_counts) / len(null_counts)
                if null_counts else 0.0)
    null_std = null_var ** 0.5
    z = ((exact_hits - null_mean) / null_std
         if null_std > 0 else (10.0 if exact_hits > null_mean else 0.0))

    return {
        'n_merged_tokens': n,
        'n_signal_pairs': len(pairs),
        'exact_hits': exact_hits,
        'cc_hits': cc_hits,
        'cc_catalog': cc_catalog[:20],
        'null_mean': round(null_mean, 2),
        'null_std': round(null_std, 2),
        'z_score': round(z, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_merged_concat() -> None:
    """Step 38.7: Macaronic Concatenation Test."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 38.7: Macaronic Concatenation Test")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    signal_data = _safe_load(os.path.join(rd, 'merged_signal.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))

    classifications = signal_data.get('token_classifications', [])
    decoded_lower = signal_data.get('token_decoded', [])
    token_folios = signal_data.get('token_folios', [])

    merged_dict = set(dict_data.get('merged_words', []))
    latin_10k = set(dict_data.get('latin_10k_words', []))
    italian_10k = set(dict_data.get('italian_10k_words', []))

    signal_words = set(w['word'] for w in signal_data.get('word_signals', []))

    bigram_list = dict_data.get('bigram_list', [])
    bigram_set: Set[Tuple[str, str]] = set()
    for pair in bigram_list:
        if len(pair) == 2:
            bigram_set.add((pair[0], pair[1]))

    print(f"     {len(decoded_lower)} tokens, {len(signal_words)} signal words")
    print(f"     {len(merged_dict)} merged dict, {len(bigram_set)} bigrams")

    # ── 2. Build expanded pair pool ──
    print("  2. Building pair pool …")
    pairs = _expanded_pair_pool(
        signal_words, decoded_lower, token_folios, classifications,
    )
    print(f"     {len(pairs)} SIGNAL-SIGNAL pairs")

    # ── 3. Concatenation against merged dict ──
    print("  3. Concatenating pairs …")
    all_matches, new_italian = _concat_against_merged(
        pairs, merged_dict, latin_10k, italian_10k,
    )
    print(f"     {len(all_matches)} concatenation matches")
    print(f"     {len(new_italian)} Italian-only matches")

    # Source breakdown
    sources = Counter(m['source'] for m in all_matches)
    for src, cnt in sources.most_common():
        print(f"       {src}: {cnt}")

    content_matches = [m for m in all_matches if m['is_content']]
    print(f"     Content words (len≥4): {len(content_matches)}")

    if all_matches:
        print("     Top concatenation matches:")
        for m in all_matches[:10]:
            print(f"       {m['w1']}+{m['w2']} → {m['concatenated']:>12s} "
                  f"({m['source']}, count={m['pair_count']})")

    # ── 4. Selective merge rules ──
    print("  4. Selective merge rules …")
    rules = _selective_merge_rules(all_matches)
    print(f"     {len(rules)} selective merge rules (freq≥3, len≥5, content)")
    for r in rules[:10]:
        print(f"       {r['w1']}+{r['w2']} → {r['merged']} "
              f"({r['source']}, count={r['count']})")

    # ── 5. Retokenize ──
    print("  5. Retokenizing with selective merges …")
    new_decoded, new_folios, new_class = _retokenize_selective(
        decoded_lower, token_folios, classifications, rules,
    )
    n_merged_tokens = sum(1 for w in new_decoded
                         if any(r['merged'] == w for r in rules))
    print(f"     Original tokens: {len(decoded_lower)}")
    print(f"     After merging: {len(new_decoded)}")
    print(f"     Merged tokens: {n_merged_tokens}")

    # ── 6. Bigram test on merged stream ──
    print("  6. Bigram test on merged stream …")
    bigram_result = _bigram_test_selective(
        new_decoded, new_folios, new_class,
        bigram_set, merged_dict, latin_10k, italian_10k,
    )

    print(f"     Signal pairs: {bigram_result['n_signal_pairs']}")
    print(f"     Exact bigram hits: {bigram_result['exact_hits']}")
    print(f"     Content-content: {bigram_result['cc_hits']}")
    print(f"     z-score: {bigram_result['z_score']}")

    if bigram_result['cc_catalog']:
        print("     Content-content bigrams:")
        for cc in bigram_result['cc_catalog'][:10]:
            print(f"       {cc['w1']} — {cc['w2']}")

    # ── 7. Comparison: no-merge vs selective merge ──
    print("  7. Comparison …")
    # No-merge bigram z from step 38.4
    bigram_data = _safe_load(os.path.join(rd, 'merged_bigrams.json'))
    original_z = bigram_data.get('bigram_z', 0.0)
    selective_z = bigram_result['z_score']

    print(f"     No-merge z: {original_z:.2f}")
    print(f"     Selective merge z: {selective_z:.2f}")
    print(f"     {'IMPROVED' if selective_z > original_z else 'NOT IMPROVED'}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'n_pairs': len(pairs),
        'n_concat_matches': len(all_matches),
        'n_content_matches': len(content_matches),
        'n_italian_only_matches': len(new_italian),
        'concat_source_counts': dict(sources),
        'concat_matches': all_matches[:100],
        'italian_only_matches': new_italian[:50],
        'n_selective_rules': len(rules),
        'selective_rules': rules,
        'n_original_tokens': len(decoded_lower),
        'n_merged_tokens': len(new_decoded),
        'bigram_result': bigram_result,
        'no_merge_z': round(original_z, 2),
        'selective_merge_z': round(selective_z, 2),
        'merge_improved': selective_z > original_z,
        'verdict': (
            f"{len(all_matches)} concat matches "
            f"({len(new_italian)} Italian-only). "
            f"{len(rules)} selective merge rules. "
            f"Selective z={selective_z:.2f} vs no-merge z={original_z:.2f}. "
            f"CC bigrams: {bigram_result['cc_hits']}."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'merged_concat.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
