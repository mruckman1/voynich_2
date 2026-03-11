"""
Step 41.1 – Null Corpus Venetian Decode
========================================
Decode all 5 null corpora through the Phase 15/16 pipeline and match
against the Venetian extended word set (29,207 words).  This produces
the proper null baseline that was missing from Phase 40.

Dependency chain:
    venetian_forms.json      (Step 40.1 — Venetian extended word set)
    combined_refine.json     (Phase 15 — best_assignment)
    modifier_integrate.json  (Phase 16 — modifier chars/rules)
    null_corpus.json         (Phase 17 — null seeds)
        → null_venetian_decode.json  (this step)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import (
    data_dir as _data_dir,
    results_dir as _results_dir,
)
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_isolation import _decode_corpus_r3


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


def _edit_distance_1(a: str, b: str) -> bool:
    """Check if two words are within edit distance 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    diffs = 0
    i = j = 0
    while i < len(longer) and j < len(shorter):
        if longer[i] != shorter[j]:
            diffs += 1
            i += 1
        else:
            i += 1
            j += 1
    return diffs + (len(longer) - i) <= 1


# ---------------------------------------------------------------------------
# Venetian bigram reference (copied from venetian_bigrams.py)
# ---------------------------------------------------------------------------

def _build_venetian_reference_bigrams(
    latin_text: str,
    anonimo_text: str,
) -> Set[Tuple[str, str]]:
    """Build a bigram set from Venetian-transformed Latin text + Anonimo."""
    from voynich.core.reference import apply_venetian_sound_changes

    bigrams: Set[Tuple[str, str]] = set()

    # 1. Transform Latin reference text to synthetic Venetian
    latin_words = re.findall(r'[a-z]+', latin_text.lower())
    venetian_words = []
    for w in latin_words:
        variants = apply_venetian_sound_changes(w)
        if variants:
            venetian_words.append(next(iter(variants)))
        else:
            venetian_words.append(w)
    for i in range(len(venetian_words) - 1):
        bigrams.add((venetian_words[i], venetian_words[i + 1]))

    # 2. Add bigrams from Anonimo Veneziano
    anonimo_words = re.findall(
        r'[a-zA-ZàèéìòùÀÈÉÌÒÙ]+', anonimo_text.lower(),
    )
    for i in range(len(anonimo_words) - 1):
        bigrams.add((anonimo_words[i], anonimo_words[i + 1]))

    return bigrams


# ---------------------------------------------------------------------------
# Core: null bigram counting
# ---------------------------------------------------------------------------

def _count_null_bigram_hits(
    null_decoded: List[str],
    venetian_set: Set[str],
    ref_bigrams: Set[Tuple[str, str]],
    ref_words: Set[str],
) -> Tuple[int, int]:
    """Count exact and relaxed bigram hits for a null corpus.

    Finds consecutive pairs where both words are in venetian_set,
    then checks against the reference bigram table.

    Returns: (n_exact, n_relaxed)
    """
    n_exact = 0
    n_relaxed = 0

    for i in range(len(null_decoded) - 1):
        w1 = null_decoded[i]
        w2 = null_decoded[i + 1]
        if not (w1 and w2 and w1 in venetian_set and w2 in venetian_set):
            continue

        if (w1, w2) in ref_bigrams:
            n_exact += 1
        else:
            # Relaxed: check edit-distance-1
            found = False
            for rw in ref_words:
                if len(rw) > len(w1) + 1 or len(rw) < len(w1) - 1:
                    continue
                if _edit_distance_1(w1, rw):
                    # Check if any partner of rw is close to w2
                    for rw1, rw2 in ref_bigrams:
                        if rw1 == rw and _edit_distance_1(w2, rw2):
                            found = True
                            break
                    if found:
                        break
            if found:
                n_relaxed += 1

    return n_exact, n_relaxed


def _count_null_bigram_hits_fast(
    null_decoded: List[str],
    venetian_set: Set[str],
    ref_bigrams: Set[Tuple[str, str]],
    word_index: Dict[str, Set[str]],
    partner_cache: Dict[str, Set[str]],
) -> Tuple[int, int]:
    """Fast bigram counting using precomputed partner cache.

    partner_cache: for each decoded word, set of ref words within edit-1.
    word_index: ref_word → set of ref partners (from bigram table).
    """
    n_exact = 0
    n_relaxed = 0

    for i in range(len(null_decoded) - 1):
        w1 = null_decoded[i]
        w2 = null_decoded[i + 1]
        if not (w1 and w2 and w1 in venetian_set and w2 in venetian_set):
            continue

        if (w1, w2) in ref_bigrams:
            n_exact += 1
            continue

        # Relaxed: use partner cache
        found = False
        w1_partners = partner_cache.get(w1, set())
        w2_partners = partner_cache.get(w2, set())

        # Check (p1, w2) or (p1, p2) in ref_bigrams
        for p1 in w1_partners:
            if p1 in word_index:
                for rp in word_index[p1]:
                    if rp == w2 or rp in w2_partners:
                        found = True
                        break
            if found:
                break

        if not found:
            # Also check (w1, p2)
            if w1 in word_index:
                for rp in word_index[w1]:
                    if rp in w2_partners:
                        found = True
                        break

        if found:
            n_relaxed += 1

    return n_exact, n_relaxed


def _build_word_index(
    reference_bigrams: Set[Tuple[str, str]],
) -> Dict[str, Set[str]]:
    """Build word→partner index for fast bigram lookup."""
    index: Dict[str, Set[str]] = {}
    for w1, w2 in reference_bigrams:
        if w1 not in index:
            index[w1] = set()
        index[w1].add(w2)
    return index


def _build_partner_cache(
    words: Set[str],
    ref_words: Set[str],
    max_word_len: int = 8,
) -> Dict[str, Set[str]]:
    """For each decoded word, find all ref words within edit distance 1.

    Filters by length for performance — only checks ref words
    within ±1 character length.
    """
    cache: Dict[str, Set[str]] = {}
    for w in words:
        if not w or len(w) > max_word_len:
            continue
        partners = set()
        for rw in ref_words:
            if abs(len(rw) - len(w)) <= 1 and _edit_distance_1(w, rw):
                partners.add(rw)
        cache[w] = partners
    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_null_venetian_decode() -> None:
    """Step 41.1: Decode null corpora and match against Venetian word set."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 41.1: Null Corpus Venetian Decode")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    # Venetian extended word set
    ven_forms = _safe_load(os.path.join(rd, 'venetian_forms.json'))
    venetian_set = set(ven_forms.get('venetian_extended_set', []))
    print(f"    Venetian extended set: {len(venetian_set):,} words")

    # Phase 15 assignment
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    print(f"    Assignment: {len(assignment)} triples")

    # Phase 16 modifiers
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    print(f"    Modifiers: {len(modifier_chars)} chars")

    # Null seeds
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]
    if not null_seeds:
        null_seeds = [100, 101, 102, 103, 104]
    print(f"    Null seeds: {null_seeds}")

    # ── 2. Decode real corpus ──
    print("\n  2. Decoding real corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)

    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, venetian_set,
    )
    real_ven_hits = sum(1 for w in real_decoded if w in venetian_set)
    real_ven_rate = real_ven_hits / n_tokens if n_tokens > 0 else 0.0
    print(f"    {n_tokens:,} tokens, Venetian dict-hit = {real_ven_rate:.4f}")

    # ── 3. Build Venetian reference bigrams ──
    print("\n  3. Building Venetian reference bigram set …")
    latin_dir = os.path.join(_data_dir(), 'reference', 'latin')
    latin_text = ''
    if os.path.isdir(latin_dir):
        for fn in sorted(os.listdir(latin_dir)):
            fpath = os.path.join(latin_dir, fn)
            if os.path.isfile(fpath) and fn.endswith('.txt'):
                with open(fpath) as f:
                    latin_text += f.read() + ' '

    anonimo_path = os.path.join(
        _data_dir(), 'reference', 'italian', 'anonimo_veneziano.txt',
    )
    anonimo_text = ''
    if os.path.exists(anonimo_path):
        with open(anonimo_path) as f:
            anonimo_text = f.read()

    ven_ref_bigrams = _build_venetian_reference_bigrams(latin_text, anonimo_text)
    print(f"    Venetian reference bigrams: {len(ven_ref_bigrams):,}")

    # Build indexes for fast bigram lookup
    word_index = _build_word_index(ven_ref_bigrams)
    ref_words = set()
    for w1, w2 in ven_ref_bigrams:
        ref_words.add(w1)
        ref_words.add(w2)
    print(f"    Reference word vocabulary: {len(ref_words):,}")

    # ── 4. Regenerate and decode null corpora ──
    print("\n  4. Regenerating and decoding null corpora …")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )

    # Collect all unique decoded words (real + will add null) for partner cache
    all_decoded_words = set(w for w in real_decoded if w)

    null_results = []
    null_decoded_lists = []

    for i, seed in enumerate(null_seeds):
        print(f"    Null corpus {i + 1}/{len(null_seeds)} (seed={seed}) …")
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, venetian_set,
        )
        null_decoded_lists.append(null_decoded)

        # Venetian hit rate
        null_ven_hits = sum(1 for w in null_decoded if w in venetian_set)
        null_ven_rate = null_ven_hits / len(null_decoded) if null_decoded else 0.0
        print(f"      Venetian dict-hit = {null_ven_rate:.4f}")

        # Collect words for partner cache
        all_decoded_words.update(w for w in null_decoded if w)

        null_results.append({
            'seed': seed,
            'n_tokens': len(null_decoded),
            'venetian_hit_count': null_ven_hits,
            'venetian_hit_rate': round(null_ven_rate, 6),
        })

    # ── 5. Build partner cache for fast relaxed bigram checking ──
    print("\n  5. Building partner cache for relaxed bigram checking …")
    # Only cache words that actually appear in venetian_set (both decoded & ref)
    words_in_venetian = {w for w in all_decoded_words if w in venetian_set}
    partner_cache = _build_partner_cache(words_in_venetian, ref_words)
    print(f"    Cached partners for {len(partner_cache):,} words")

    # ── 6. Count bigram hits for real corpus ──
    print("\n  6. Counting real corpus bigram hits …")
    real_exact, real_relaxed = _count_null_bigram_hits_fast(
        real_decoded, venetian_set, ven_ref_bigrams, word_index, partner_cache,
    )
    real_total_bigram = real_exact + real_relaxed
    print(f"    Real: exact={real_exact}, relaxed={real_relaxed}, "
          f"total={real_total_bigram}")

    # ── 7. Count bigram hits for each null corpus ──
    print("\n  7. Counting null corpus bigram hits …")
    for i, null_decoded in enumerate(null_decoded_lists):
        null_exact, null_relaxed = _count_null_bigram_hits_fast(
            null_decoded, venetian_set, ven_ref_bigrams,
            word_index, partner_cache,
        )
        null_results[i]['bigram_exact'] = null_exact
        null_results[i]['bigram_relaxed'] = null_relaxed
        null_results[i]['bigram_total'] = null_exact + null_relaxed
        print(f"    Null {i + 1}: exact={null_exact}, "
              f"relaxed={null_relaxed}, total={null_exact + null_relaxed}")

    # ── 8. Summary statistics ──
    print("\n  8. Summary …")
    null_ven_rates = [r['venetian_hit_rate'] for r in null_results]
    null_mean_rate = sum(null_ven_rates) / len(null_ven_rates)
    null_std_rate = (sum((r - null_mean_rate) ** 2 for r in null_ven_rates)
                     / len(null_ven_rates)) ** 0.5

    selectivity = real_ven_rate / null_mean_rate if null_mean_rate > 0.001 else 999.0
    z_hit_rate = ((real_ven_rate - null_mean_rate) / null_std_rate
                  if null_std_rate > 0.001 else 0.0)

    null_bigram_totals = [r['bigram_total'] for r in null_results]
    null_bigram_mean = (sum(null_bigram_totals) / len(null_bigram_totals)
                        if null_bigram_totals else 0.0)
    null_bigram_std = (sum((b - null_bigram_mean) ** 2 for b in null_bigram_totals)
                       / len(null_bigram_totals)) ** 0.5 if null_bigram_totals else 0.001

    bigram_z_raw = ((real_total_bigram - null_bigram_mean) / null_bigram_std
                    if null_bigram_std > 0.001 else 0.0)

    print(f"    Real Venetian hit rate: {real_ven_rate:.4f}")
    print(f"    Null mean hit rate: {null_mean_rate:.4f} ± {null_std_rate:.4f}")
    print(f"    Selectivity: {selectivity:.2f}×")
    print(f"    Hit-rate z: {z_hit_rate:.2f}")
    print(f"    Real bigram total: {real_total_bigram}")
    print(f"    Null bigram mean: {null_bigram_mean:.1f} ± {null_bigram_std:.1f}")
    print(f"    Bigram z (raw, 5-null): {bigram_z_raw:.2f}")

    # ── 9. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': n_tokens,
        'n_venetian_set': len(venetian_set),
        'n_ref_bigrams': len(ven_ref_bigrams),
        'real_venetian_hit_rate': round(real_ven_rate, 6),
        'real_venetian_hit_count': real_ven_hits,
        'real_bigram_exact': real_exact,
        'real_bigram_relaxed': real_relaxed,
        'real_bigram_total': real_total_bigram,
        'null_results': null_results,
        'null_mean_venetian_hit_rate': round(null_mean_rate, 6),
        'null_std_venetian_hit_rate': round(null_std_rate, 6),
        'selectivity': round(selectivity, 4),
        'z_hit_rate': round(z_hit_rate, 4),
        'null_bigram_mean': round(null_bigram_mean, 2),
        'null_bigram_std': round(null_bigram_std, 2),
        'bigram_z_raw': round(bigram_z_raw, 4),
        'null_decoded_stored': True,
        'null_decoded_tokens': [nd for nd in null_decoded_lists],
        'real_decoded_tokens': real_decoded,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'null_venetian_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
