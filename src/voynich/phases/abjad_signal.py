"""
Step 34.4 – Abjad Signal Isolation (Track A)
=============================================
Runs the full Phase 28-29 signal pipeline on the abjad-decoded corpus.

Dependency chain:
    abjad_csp.json            (34.2: abjad table)
    sigla_dictionary.json     (34.1: skeleton dict)
    modifier_integrate.json   (Phase 16)
    null_corpus.json          (Phase 17)
    signal_bigrams.json       (Phase 29: baseline)
        → abjad_signal.json   (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import load_reference_corpus
from voynich.phases.morpheme_grid import decompose_token_morphemes
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.sigla_dictionary import _strip_vowels


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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AbjadSignalResult:
    n_tokens: int
    n_signal: int
    signal_rate: float
    n_anti: int
    anti_rate: float
    n_shared_hit: int
    n_shared_miss: int
    dict_hit_rate: float

    # Bigram test
    n_signal_pairs: int
    n_bigram_hits: int
    bigram_hit_rate: float
    bigram_z: float
    null_bigram_mean: float
    null_bigram_std: float

    # Baseline comparison
    phase29_signal_rate: float
    phase29_bigram_z: float
    signal_rate_delta: float
    bigram_z_delta: float
    verdict: str  # ABJAD_SIGNAL_BETTER / CV_SIGNAL_BETTER

    runtime_seconds: float


# ---------------------------------------------------------------------------
# Abjad decode
# ---------------------------------------------------------------------------

def _abjad_decode_token(
    token: str,
    eva_to_triple: Dict[str, str],
    abjad_table: Dict[str, str],
) -> str:
    """Decode a token's root through the abjad table → consonant string."""
    decomp = decompose_token_morphemes(token)
    stem = decomp.stem if hasattr(decomp, 'stem') else token
    stem_chars = tokenize_eva_chars(stem)
    consonants = []
    for ch in stem_chars:
        triple = eva_to_triple.get(ch)
        if triple and triple in abjad_table:
            consonants.append(abjad_table[triple])
    return ''.join(consonants)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_abjad_signal() -> None:
    """Step 34.4: Abjad signal isolation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.4: Abjad Signal Isolation (Track A)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load abjad assignment ──
    print("\n  1. Loading abjad assignment …")
    abjad_path = os.path.join(rd, 'abjad_csp.json')
    if not os.path.exists(abjad_path):
        print("  [SKIP] abjad_csp.json not found")
        return
    with open(abjad_path) as f:
        abjad_data = json.load(f)
    abjad_table = abjad_data.get('best_assignment', {})

    # ── 2. Build consonant skeleton dictionary ──
    print("\n  2. Building skeleton dictionary …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    skeleton_dict: Set[str] = set()
    for word in base_words:
        skel = _strip_vowels(word)
        if len(skel) >= 2:
            skeleton_dict.add(skel)
    print(f"     {len(skeleton_dict)} reference skeletons")

    # Build consonant-skeleton bigrams from reference
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_skel_bigrams: Set[Tuple[str, str]] = set()
    prev_skel = None
    for word in ref_tokens:
        skel = _strip_vowels(word.lower())
        if len(skel) >= 2:
            if prev_skel is not None:
                ref_skel_bigrams.add((prev_skel, skel))
            prev_skel = skel
        else:
            prev_skel = None
    print(f"     {len(ref_skel_bigrams)} reference skeleton bigrams")

    # ── 3. Load corpus ──
    print("\n  3. Loading corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    all_tokens: List[str] = []
    token_folios: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens")

    # ── 4. Decode real corpus ──
    print("\n  4. Decoding real corpus through abjad table …")
    real_decoded = [_abjad_decode_token(t, eva_to_triple, abjad_table)
                    for t in all_tokens]
    real_hits = [s in skeleton_dict and len(s) >= 2 for s in real_decoded]
    dict_hit_rate = sum(real_hits) / n_tokens if n_tokens > 0 else 0.0
    print(f"     Dict hit rate: {dict_hit_rate:.3f}")

    # ── 5. Decode null corpora ──
    print("\n  5. Decoding null corpora …")
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    null_hits_list: List[List[bool]] = []

    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = [_abjad_decode_token(t, eva_to_triple, abjad_table)
                        for t in null_tokens]
        null_hits_list.append([s in skeleton_dict and len(s) >= 2
                               for s in null_decoded])

    # ── 6. Classify tokens ──
    print("\n  6. Classifying tokens …")
    classifications: List[str] = []
    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if nh[idx])
        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')

    n_signal = classifications.count('SIGNAL')
    n_anti = classifications.count('ANTI_SIGNAL')
    n_shared_hit = classifications.count('SHARED_HIT')
    n_shared_miss = classifications.count('SHARED_MISS')
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    anti_rate = n_anti / n_tokens if n_tokens > 0 else 0.0

    print(f"     SIGNAL: {n_signal} ({signal_rate:.3f})")
    print(f"     ANTI:   {n_anti} ({anti_rate:.3f})")

    # ── 7. Bigram z-score ──
    print("\n  7. Computing bigram z-score …")
    # Find SIGNAL-SIGNAL pairs
    signal_pairs = []
    for i in range(n_tokens - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and token_folios[i] == token_folios[i + 1]):
            signal_pairs.append((token_folios[i], i,
                                 real_decoded[i], real_decoded[i + 1]))

    n_bigram_hits = sum(
        1 for _, _, w1, w2 in signal_pairs
        if (w1, w2) in ref_skel_bigrams
    )
    bigram_hit_rate = n_bigram_hits / len(signal_pairs) if signal_pairs else 0.0

    # Null permutation test
    rng = random.Random(42)
    indices = list(range(n_tokens))
    null_rates: List[float] = []
    for _ in range(500):
        fake_signal = set(rng.sample(indices, min(n_signal, n_tokens)))
        n_pairs = 0
        n_hits = 0
        for i in range(n_tokens - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and token_folios[i] == token_folios[i + 1]):
                n_pairs += 1
                if (real_decoded[i], real_decoded[i + 1]) in ref_skel_bigrams:
                    n_hits += 1
        rate = n_hits / n_pairs if n_pairs > 0 else 0.0
        null_rates.append(rate)

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates) if null_rates else 0.0
    null_std = null_var ** 0.5
    bigram_z = (bigram_hit_rate - null_mean) / null_std if null_std > 0 else 0.0

    print(f"     Signal pairs: {len(signal_pairs)}")
    print(f"     Bigram hits: {n_bigram_hits} ({bigram_hit_rate:.4f})")
    print(f"     Bigram z: {bigram_z:.2f}")

    # ── 8. Compare to Phase 29 baseline ──
    print("\n  8. Comparing to Phase 29 baseline …")
    phase29_signal_rate = 0.165
    phase29_bigram_z = 6.14
    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(bg_path):
        with open(bg_path) as f:
            bg = json.load(f)
        phase29_signal_rate = bg.get('signal_rate', 0.165)
        phase29_bigram_z = bg.get('bigram_z_score', 6.14)

    verdict = ('ABJAD_SIGNAL_BETTER'
               if signal_rate > phase29_signal_rate and bigram_z > phase29_bigram_z
               else 'CV_SIGNAL_BETTER')

    print(f"     Phase 29: SIGNAL={phase29_signal_rate:.3f}, z={phase29_bigram_z:.2f}")
    print(f"     Abjad:    SIGNAL={signal_rate:.3f}, z={bigram_z:.2f}")
    print(f"     Verdict: {verdict}")

    elapsed = time.time() - t0

    result = AbjadSignalResult(
        n_tokens=n_tokens,
        n_signal=n_signal,
        signal_rate=round(signal_rate, 4),
        n_anti=n_anti,
        anti_rate=round(anti_rate, 4),
        n_shared_hit=n_shared_hit,
        n_shared_miss=n_shared_miss,
        dict_hit_rate=round(dict_hit_rate, 4),
        n_signal_pairs=len(signal_pairs),
        n_bigram_hits=n_bigram_hits,
        bigram_hit_rate=round(bigram_hit_rate, 4),
        bigram_z=round(bigram_z, 2),
        null_bigram_mean=round(null_mean, 6),
        null_bigram_std=round(null_std, 6),
        phase29_signal_rate=round(phase29_signal_rate, 4),
        phase29_bigram_z=round(phase29_bigram_z, 2),
        signal_rate_delta=round(signal_rate - phase29_signal_rate, 4),
        bigram_z_delta=round(bigram_z - phase29_bigram_z, 2),
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'abjad_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"\n  Completed in {elapsed:.1f}s")
