"""
Step 34.10 – Dialect Signal Pipeline (Track C)
================================================
Standard signal pipeline on the dialect-optimized decode from Step 34.9.
Same architecture as abjad_signal.py: decode real + 5 null corpora
through the dialect table, classify tokens, compute bigram z-score,
and compare to Phase 29 baselines.

Dependency chain:
    dialect_decode.json       (34.9: dialect-optimized table)
    modifier_integrate.json   (Phase 16 modifiers)
    null_corpus.json          (Phase 17 seeds)
    signal_bigrams.json       (Phase 29 baseline)
        -> dialect_signal.json (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.mixed_lm import _apply_sound_changes, _preprocess_text


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
# Dialect decode helper
# ---------------------------------------------------------------------------

def _dialect_decode_token(
    token: str,
    eva_to_triple: Dict[str, str],
    dialect_table: Dict[str, str],
    modifier_chars: Set[str],
) -> str:
    """Decode a single token through the dialect table.

    Tokenizes EVA chars, skips modifier chars, maps remaining to triples,
    looks up syllables in dialect_table.
    """
    chars = tokenize_eva_chars(token)
    syllables = []
    for ch in chars:
        if ch in modifier_chars:
            continue
        triple = eva_to_triple.get(ch)
        if triple and triple in dialect_table:
            syllables.append(dialect_table[triple])
    return ''.join(syllables).lower()


def _dialect_decode_corpus(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    dialect_table: Dict[str, str],
    modifier_chars: Set[str],
) -> List[str]:
    """Decode a full token list through the dialect table."""
    return [
        _dialect_decode_token(t, eva_to_triple, dialect_table, modifier_chars)
        for t in tokens
    ]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DialectSignalResult:
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

    # Baseline comparison (Phase 29)
    phase29_signal_rate: float
    phase29_bigram_z: float
    signal_rate_delta: float
    bigram_z_delta: float

    # Per-folio top signal folios
    top_signal_folios: List[Dict]

    verdict: str  # DIALECT_SIGNAL_BETTER / CV_SIGNAL_BETTER
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dialect_signal() -> None:
    """Step 34.10: Dialect signal pipeline."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.10: Dialect Signal Pipeline (Track C)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load dialect assignment ──
    print("\n  1. Loading dialect assignment …")
    dialect_path = os.path.join(rd, 'dialect_decode.json')
    if not os.path.exists(dialect_path):
        print("  [SKIP] dialect_decode.json not found — run dialect-decode first")
        return
    with open(dialect_path) as f:
        dialect_data = json.load(f)
    dialect_table = dialect_data.get('best_assignment', {})
    print(f"     {len(dialect_table)} triple assignments loaded")

    # ── 2. Build dialect reference word set ──
    print("\n  2. Building dialect reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    raw_latin = ref_corpus.get_combined_text('latin')
    latin_text = _preprocess_text(raw_latin)
    dialect_text = _apply_sound_changes(latin_text)
    dialect_words = set(w for w in dialect_text.split() if len(w) >= 2)
    expanded_dialect, _ = build_expanded_word_set(dialect_words)
    dialect_word_set = dialect_words | expanded_dialect
    print(f"     {len(dialect_word_set)} dialect reference words")

    # Build word-level bigrams for dialect reference
    dialect_tokens_ref = [w for w in dialect_text.split() if len(w) >= 2]
    dialect_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(dialect_tokens_ref) - 1):
        dialect_bigrams.add((dialect_tokens_ref[i], dialect_tokens_ref[i + 1]))
    print(f"     {len(dialect_bigrams)} dialect word bigrams")

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

    # ── 4. Decode real corpus through dialect table ──
    print("\n  4. Decoding real corpus …")
    real_decoded = _dialect_decode_corpus(
        all_tokens, eva_to_triple, dialect_table, modifier_chars,
    )
    real_hits = [w in dialect_word_set and len(w) >= 2 for w in real_decoded]
    dict_hit_rate = sum(real_hits) / n_tokens if n_tokens > 0 else 0.0
    print(f"     Dict hit rate: {dict_hit_rate:.3f} ({sum(real_hits)} hits)")

    # ── 5. Decode null corpora ──
    print("\n  5. Decoding 5 null corpora …")
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    null_hits_list: List[List[bool]] = []

    for i, seed in enumerate(null_seeds):
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _dialect_decode_corpus(
            null_tokens, eva_to_triple, dialect_table, modifier_chars,
        )
        null_hits = [w in dialect_word_set and len(w) >= 2 for w in null_decoded]
        null_hits_list.append(null_hits)
        null_rate = sum(null_hits) / len(null_hits) if null_hits else 0.0
        print(f"     Null {i + 1} (seed={seed}): dict_hit={null_rate:.3f}")

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

    print(f"     SIGNAL:      {n_signal} ({signal_rate:.3f})")
    print(f"     ANTI_SIGNAL: {n_anti} ({anti_rate:.3f})")
    print(f"     SHARED_HIT:  {n_shared_hit}")
    print(f"     SHARED_MISS: {n_shared_miss}")

    # ── 7. Bigram z-score ──
    print("\n  7. Computing bigram z-score …")
    # Find SIGNAL-SIGNAL pairs respecting folio boundaries
    signal_pairs: List[Tuple[str, int, str, str]] = []
    for i in range(n_tokens - 1):
        if (classifications[i] == 'SIGNAL'
                and classifications[i + 1] == 'SIGNAL'
                and token_folios[i] == token_folios[i + 1]):
            signal_pairs.append((
                token_folios[i], i,
                real_decoded[i], real_decoded[i + 1],
            ))

    n_bigram_hits = sum(
        1 for _, _, w1, w2 in signal_pairs
        if (w1, w2) in dialect_bigrams
    )
    bigram_hit_rate = n_bigram_hits / len(signal_pairs) if signal_pairs else 0.0

    # Null permutation test (500 relabelings)
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
                if (real_decoded[i], real_decoded[i + 1]) in dialect_bigrams:
                    n_hits += 1
        rate = n_hits / n_pairs if n_pairs > 0 else 0.0
        null_rates.append(rate)

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = (
        sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)
        if null_rates else 0.0
    )
    null_std = null_var ** 0.5
    bigram_z = (bigram_hit_rate - null_mean) / null_std if null_std > 0 else 0.0

    print(f"     Signal pairs: {len(signal_pairs)}")
    print(f"     Bigram hits: {n_bigram_hits} ({bigram_hit_rate:.4f})")
    print(f"     Null mean: {null_mean:.6f}, std: {null_std:.6f}")
    print(f"     Bigram z: {bigram_z:.2f}")

    # ── 8. Per-folio signal ranking (top 10) ──
    print("\n  8. Per-folio signal ranking …")
    folio_n: Counter = Counter(token_folios)
    folio_n_signal: Counter = Counter()
    for folio, cls in zip(token_folios, classifications):
        if cls == 'SIGNAL':
            folio_n_signal[folio] += 1

    folio_stats = []
    for folio in sorted(folio_n.keys()):
        n_tok = folio_n[folio]
        n_sig = folio_n_signal.get(folio, 0)
        sr = n_sig / n_tok if n_tok > 0 else 0.0
        folio_stats.append({
            'folio': folio,
            'n_tokens': n_tok,
            'n_signal': n_sig,
            'signal_rate': round(sr, 4),
        })
    folio_stats.sort(key=lambda x: -x['signal_rate'])
    for fs in folio_stats[:10]:
        print(f"     {fs['folio']:8s}: {fs['n_signal']:3d}/{fs['n_tokens']:3d} "
              f"({fs['signal_rate']:.1%})")

    # ── 9. Compare to Phase 29 baseline ──
    print("\n  9. Comparing to Phase 29 baseline …")
    phase29_signal_rate = 0.165
    phase29_bigram_z = 6.14
    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(bg_path):
        with open(bg_path) as f:
            bg = json.load(f)
        phase29_signal_rate = bg.get('signal_rate', 0.165)
        phase29_bigram_z = bg.get('bigram_z_score', 6.14)

    signal_rate_delta = signal_rate - phase29_signal_rate
    bigram_z_delta = bigram_z - phase29_bigram_z

    verdict = (
        'DIALECT_SIGNAL_BETTER'
        if signal_rate > phase29_signal_rate and bigram_z > phase29_bigram_z
        else 'CV_SIGNAL_BETTER'
    )

    print(f"     Phase 29 (CV):   signal={phase29_signal_rate:.3f}, z={phase29_bigram_z:.2f}")
    print(f"     Dialect:         signal={signal_rate:.3f}, z={bigram_z:.2f}")
    print(f"     Delta:           signal={signal_rate_delta:+.3f}, z={bigram_z_delta:+.2f}")
    print(f"     Verdict: {verdict}")

    # ── 10. Save ──
    elapsed = round(time.time() - t0, 2)

    result = DialectSignalResult(
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
        signal_rate_delta=round(signal_rate_delta, 4),
        bigram_z_delta=round(bigram_z_delta, 2),
        top_signal_folios=folio_stats[:20],
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'dialect_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved -> {out_path}  ({elapsed:.1f}s)")
