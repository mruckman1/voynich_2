"""
Step 39.4 – Corrected Table Signal Pipeline
=============================================
Run the full signal pipeline (decode, signal isolation, bigram test)
on the vowel-corrected table from Step 39.3.

Dependency chain:
    targeted_vowel_fix.json    (Step 39.3)
    merged_dict.json           (Step 38.1)
    null_corpus.json           (Step 17)
    modifier_integrate.json    (Step 16)
    decode_10k.json            (Step 36.1)
    merged_signal.json         (Step 38.3 — for comparison)
    merged_bigrams.json        (Step 38.4 — for comparison)
        → corrected_signal.json (this step)
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
    load_corpus,
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


def _edit_distance_1(w1: str, w2: str) -> bool:
    """Check if edit distance <= 1."""
    if abs(len(w1) - len(w2)) > 1:
        return False
    if w1 == w2:
        return True
    if len(w1) == len(w2):
        return sum(a != b for a, b in zip(w1, w2)) <= 1
    short, long_ = (w1, w2) if len(w1) < len(w2) else (w2, w1)
    i = j = diffs = 0
    while i < len(short) and j < len(long_):
        if short[i] != long_[j]:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


# ---------------------------------------------------------------------------
# Signal isolation (replicates merged_signal.py logic)
# ---------------------------------------------------------------------------

def _classify_signal(
    real_hits: List[bool],
    null_hits_list: List[List[bool]],
) -> List[str]:
    """4-class signal classification."""
    n = len(real_hits)
    classifications = []
    for i in range(n):
        r_hit = real_hits[i]
        n_null = sum(1 for nh in null_hits_list if i < len(nh) and nh[i])
        if r_hit and n_null <= 1:
            classifications.append('SIGNAL')
        elif r_hit and n_null >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and n_null >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')
    return classifications


def _per_word_signal(
    decoded: List[str],
    classifications: List[str],
    null_decoded_list: List[List[str]],
    merged_dict: Set[str],
) -> List[Dict]:
    """Compute per-word sigma for genuine signal detection."""
    word_real_counts = Counter()
    for w, cls in zip(decoded, classifications):
        if cls in ('SIGNAL', 'SHARED_HIT'):
            word_real_counts[w] += 1

    word_null_counts: Dict[str, List[int]] = {}
    for w in word_real_counts:
        word_null_counts[w] = []

    for null_dec in null_decoded_list:
        null_wc = Counter(null_dec)
        for w in word_real_counts:
            word_null_counts[w].append(null_wc.get(w, 0))

    word_signals = []
    for w, real_count in word_real_counts.most_common():
        if real_count < 5:
            continue
        null_vals = word_null_counts.get(w, [0])
        null_mean = sum(null_vals) / len(null_vals) if null_vals else 0.0
        null_var = sum((v - null_mean) ** 2 for v in null_vals) / len(null_vals) if null_vals else 0.0
        null_std = null_var ** 0.5
        sigma = (real_count - null_mean) / null_std if null_std > 0 else float('inf')
        selectivity = real_count / null_mean if null_mean > 0 else float('inf')

        word_signals.append({
            'word': w,
            'real_count': real_count,
            'null_mean': round(null_mean, 2),
            'sigma': round(sigma, 2),
            'selectivity': round(selectivity, 2),
            'is_genuine_signal': sigma > 2.0 and real_count >= 5,
        })

    word_signals.sort(key=lambda x: x['sigma'], reverse=True)
    return word_signals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_corrected_signal() -> None:
    """Step 39.4: Corrected Table Signal Pipeline."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.4: Corrected Table Signal Pipeline")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    fix_data = _safe_load(os.path.join(rd, 'targeted_vowel_fix.json'))
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    decode_data = _safe_load(os.path.join(rd, 'decode_10k.json'))
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    baseline_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))
    baseline_bigrams = _safe_load(os.path.join(rd, 'merged_bigrams.json'))

    # Use corrected assignment if available, else fall back to combined_refine
    corrected_assignment = fix_data.get('corrected_assignment')
    if not corrected_assignment:
        refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
        corrected_assignment = refine_data.get('best_assignment', {})
        print("     WARNING: No corrected assignment, using Phase 15 baseline")

    merged_words = set(dict_data.get('merged_words', []))
    latin_10k = set(dict_data.get('latin_10k_words', []))
    italian_10k = set(dict_data.get('italian_10k_words', []))
    bigram_list = dict_data.get('bigram_list', [])
    bigram_set = {(b[0], b[1]) for b in bigram_list}

    token_evas = decode_data.get('token_evas', [])
    token_folios = decode_data.get('token_folios', [])

    from voynich.phases.null_corpus import _reconstruct_modifier_rules
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    eva_to_triple = build_eva_to_triple_lookup()

    # Null corpus seeds
    null_seeds = [nr.get('seed', 100 + i)
                  for i, nr in enumerate(null_data.get('null_runs', []))]
    if not null_seeds:
        null_seeds = [100, 101, 102, 103, 104]

    print(f"     Corrected assignment triples: {len(corrected_assignment)}")
    print(f"     Merged dict size: {len(merged_words)}")
    print(f"     Null seeds: {null_seeds}")

    # ── 2. Decode real corpus ──
    print("\n  2. Decoding real corpus with corrected assignment …")
    decoded = []
    for eva in token_evas:
        d = decode_token_modifier_aware(
            eva, corrected_assignment, eva_to_triple,
            modifier_chars, modifier_rules)
        decoded.append(d.lower())
    real_hits = [w in merged_words for w in decoded]
    real_hit_rate = sum(real_hits) / len(real_hits) if real_hits else 0.0
    print(f"     Dict hit rate: {real_hit_rate:.4f}")

    # ── 3. Decode null corpora ──
    print("\n  3. Decoding null corpora …")
    from voynich.phases.null_corpus import _build_eva_bigram_model, _generate_null_corpus
    corpus = load_corpus(verbose=False)
    all_tokens = []
    for _folio, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_decoded_list = []
    null_hits_list = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(token_evas), seed)
        null_dec = []
        for nt in null_tokens:
            d = decode_token_modifier_aware(
                nt, corrected_assignment, eva_to_triple,
                modifier_chars, modifier_rules)
            null_dec.append(d.lower())
        null_decoded_list.append(null_dec)
        null_hits = [w in merged_words for w in null_dec]
        null_hits_list.append(null_hits)

    null_hit_rates = [sum(nh) / len(nh) if nh else 0.0 for nh in null_hits_list]
    null_mean_hit = sum(null_hit_rates) / len(null_hit_rates) if null_hit_rates else 0.0
    print(f"     Null mean hit rate: {null_mean_hit:.4f}")

    # ── 4. Signal classification ──
    print("\n  4. Signal classification …")
    classifications = _classify_signal(real_hits, null_hits_list)
    class_counts = Counter(classifications)
    n_signal = class_counts.get('SIGNAL', 0)
    signal_rate = n_signal / len(classifications) if classifications else 0.0
    print(f"     SIGNAL: {n_signal} ({signal_rate:.4f})")
    print(f"     SHARED_HIT: {class_counts.get('SHARED_HIT', 0)}")
    print(f"     SHARED_MISS: {class_counts.get('SHARED_MISS', 0)}")
    print(f"     ANTI_SIGNAL: {class_counts.get('ANTI_SIGNAL', 0)}")

    # ── 5. Per-word signal ──
    print("\n  5. Per-word signal scoring …")
    word_signals = _per_word_signal(
        decoded, classifications, null_decoded_list, merged_words)
    n_genuine = sum(1 for ws in word_signals if ws.get('is_genuine_signal'))
    print(f"     Genuine signal words: {n_genuine}")

    # ── 6. Bigram test ──
    print("\n  6. Bigram plausibility test …")
    function_words = {'de', 'in', 'se', 'ne', 'ad', 'la', 'le', 'di',
                      'da', 'si', 'no', 'et', 'a', 'e', 'i', 'o', 'u',
                      'cum', 'per', 'pro', 'sub', 'que'}

    # Find SIGNAL-SIGNAL pairs
    signal_pairs = []
    for i in range(len(classifications) - 1):
        if (classifications[i] == 'SIGNAL' and
                classifications[i + 1] == 'SIGNAL' and
                i < len(token_folios) - 1 and
                token_folios[i] == token_folios[i + 1]):
            signal_pairs.append((token_folios[i], i, decoded[i], decoded[i + 1]))

    exact_bigram_hits = 0
    relaxed_bigram_hits = 0
    n_exact_cc = 0
    n_relaxed_cc = 0

    for folio, pos, w1, w2 in signal_pairs:
        is_exact = (w1, w2) in bigram_set
        if is_exact:
            exact_bigram_hits += 1
            w1c = w1 not in function_words and len(w1) >= 3 and w1 in merged_words
            w2c = w2 not in function_words and len(w2) >= 3 and w2 in merged_words
            if w1c and w2c:
                n_exact_cc += 1
        else:
            for bw1, bw2 in bigram_set:
                if _edit_distance_1(w1, bw1) and _edit_distance_1(w2, bw2):
                    relaxed_bigram_hits += 1
                    w1c = w1 not in function_words and len(w1) >= 3 and w1 in merged_words
                    w2c = w2 not in function_words and len(w2) >= 3 and w2 in merged_words
                    if w1c and w2c:
                        n_relaxed_cc += 1
                    break

    # Null permutation for z-score
    signal_words_set = {w for w, cls in zip(decoded, classifications) if cls == 'SIGNAL'}
    signal_list = list(signal_words_set)
    rng = random.Random(42)
    null_bigram_counts = []
    for _ in range(500):
        shuffled = list(signal_list) * 5
        rng.shuffle(shuffled)
        nc = sum(1 for j in range(len(shuffled) - 1)
                 if (shuffled[j], shuffled[j + 1]) in bigram_set)
        null_bigram_counts.append(nc)
    null_mean = sum(null_bigram_counts) / len(null_bigram_counts)
    null_var = sum((c - null_mean) ** 2 for c in null_bigram_counts) / len(null_bigram_counts)
    null_std = null_var ** 0.5
    bigram_z = (exact_bigram_hits - null_mean) / null_std if null_std > 0 else 0.0

    print(f"     Exact bigram hits: {exact_bigram_hits}")
    print(f"     Relaxed bigram hits: {relaxed_bigram_hits}")
    print(f"     Bigram z: {bigram_z:.2f}")
    print(f"     Exact CC: {n_exact_cc}")
    print(f"     Relaxed CC: {n_relaxed_cc}")

    # ── 7. Compare with Phase 38 baseline ──
    print("\n  7. Comparison with Phase 38 …")
    p38_signal_rate = baseline_signal.get('signal_rate', 0.0)
    p38_n_signal = baseline_signal.get('n_signal', 0)
    p38_bigram_z = baseline_bigrams.get('bigram_z', 0.0)
    p38_n_cc = baseline_bigrams.get('n_content_content', 0)

    print(f"     Signal rate: {p38_signal_rate:.4f} → {signal_rate:.4f}")
    print(f"     Bigram z: {p38_bigram_z:.2f} → {bigram_z:.2f}")
    print(f"     CC bigrams: {p38_n_cc} → {n_exact_cc + n_relaxed_cc}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': len(decoded),
        'dict_hit_rate': round(real_hit_rate, 4),
        'null_mean_hit_rate': round(null_mean_hit, 4),
        'n_signal': n_signal,
        'signal_rate': round(signal_rate, 4),
        'n_shared_hit': class_counts.get('SHARED_HIT', 0),
        'n_shared_miss': class_counts.get('SHARED_MISS', 0),
        'n_anti_signal': class_counts.get('ANTI_SIGNAL', 0),
        'n_genuine_signal_words': n_genuine,
        'word_signals': word_signals[:100],  # limit
        'exact_bigram_hits': exact_bigram_hits,
        'relaxed_bigram_hits': relaxed_bigram_hits,
        'bigram_z': round(bigram_z, 2),
        'n_exact_cc': n_exact_cc,
        'n_relaxed_cc': n_relaxed_cc,
        'n_signal_pairs': len(signal_pairs),
        'delta_vs_phase38': {
            'signal_rate_delta': round(signal_rate - p38_signal_rate, 4),
            'bigram_z_delta': round(bigram_z - p38_bigram_z, 2),
            'n_signal_delta': n_signal - p38_n_signal,
        },
        'token_classifications': classifications,
        'token_decoded': decoded,
        'verdict': (
            f"Corrected signal: {n_signal} SIGNAL ({signal_rate:.4f}), "
            f"z={bigram_z:.2f}, {n_exact_cc} exact CC, "
            f"{n_relaxed_cc} relaxed CC, "
            f"{n_genuine} genuine signal words."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'corrected_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
