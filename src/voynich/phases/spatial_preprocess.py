"""
Step 35.1 – Spatial Gallows Preprocessing
==========================================
Classify every gallows occurrence in the corpus by spatial type
(INTERSECTING / PRECEDING / FOLLOWING / STANDALONE), then build the
spatial-conditioned token representation used by all Phase 35 steps.

INTERSECTING ligatures (cth/ckh/cph/cfh) are retained as single units.
PRECEDING and FOLLOWING gallows are stripped from the phonetic stream.
STANDALONE gallows (the entire token) produce no phonetic output.

Also applies a heuristic spatial conditioning to the 5 null corpora.

Dependency chain:
    gallows_geometry.py            (spatial classification logic)
    null_corpus.json               (Phase 17 seeds)
    data/corpus/                   (EVA transcription)
        → spatial_preprocess.json  (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.phases.gallows_geometry import (
    _classify_gallows_spatial,
    GALLOWS_CHARS,
    GALLOWS_BENCH_LIGATURES,
    SPATIAL_INTERSECTING,
    SPATIAL_PRECEDING,
    SPATIAL_FOLLOWING,
    SPATIAL_STANDALONE,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
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
# Spatial conditioning
# ---------------------------------------------------------------------------

def _apply_spatial_conditioning(
    token: str,
    eva_chars: List[str],
    classifications: List[Tuple[str, int, str, List[str]]],
) -> Tuple[str, str, Optional[str]]:
    """Apply spatial conditioning to a single token.

    Returns (conditioned_token, strategy, determinative_char_or_None).

    Strategy labels:
      'identity'          — no gallows in token, pass-through
      'intersecting_kept' — has INTERSECTING ligature, kept as-is
      'strip_preceding'   — PRECEDING gallows stripped
      'strip_following'   — FOLLOWING gallows stripped
      'strip_mixed'       — mixture of PRECEDING/FOLLOWING stripped
      'standalone_silent' — entire token is a standalone gallows
    """
    if not classifications:
        return token, 'identity', None

    # Build set of char indices to strip and collect determinative info
    strip_indices: Set[int] = set()
    has_intersecting = False
    has_preceding = False
    has_following = False
    determinative = None

    for eva_char, pos, spatial_type, adj in classifications:
        if spatial_type == SPATIAL_INTERSECTING:
            has_intersecting = True
            # Ligatures stay — don't strip
        elif spatial_type == SPATIAL_PRECEDING:
            strip_indices.add(pos)
            has_preceding = True
            if eva_char in GALLOWS_CHARS:
                determinative = eva_char
        elif spatial_type == SPATIAL_FOLLOWING:
            strip_indices.add(pos)
            has_following = True
            if eva_char in GALLOWS_CHARS:
                determinative = eva_char
        elif spatial_type == SPATIAL_STANDALONE:
            strip_indices.add(pos)
            if eva_char in GALLOWS_CHARS:
                determinative = eva_char

    # If all chars would be stripped, this is a standalone-silent token
    if len(strip_indices) == len(eva_chars):
        return '', 'standalone_silent', determinative

    # Build conditioned token from remaining chars
    remaining = [ch for i, ch in enumerate(eva_chars) if i not in strip_indices]
    conditioned = ''.join(remaining)

    # Determine strategy
    if not strip_indices:
        if has_intersecting:
            strategy = 'intersecting_kept'
        else:
            strategy = 'identity'
    elif has_preceding and has_following:
        strategy = 'strip_mixed'
    elif has_preceding:
        strategy = 'strip_preceding'
    elif has_following:
        strategy = 'strip_following'
    else:
        strategy = 'strip_preceding'  # fallback

    return conditioned, strategy, determinative


def _condition_null_token(token: str) -> str:
    """Apply spatial conditioning heuristic to a null-corpus token.

    Null corpora lack spatial markup. Heuristic:
      - Detect cth/ckh/cph/cfh as INTERSECTING → keep
      - Strip remaining bare k/t/p/f chars

    This is conservative: strips ALL bare gallows, slightly inflating
    null dict-hit rate (making selectivity conservative).
    """
    chars = tokenize_eva_chars(token)
    remaining = []
    for ch in chars:
        if ch in GALLOWS_BENCH_LIGATURES:
            # INTERSECTING — keep as-is
            remaining.append(ch)
        elif ch in GALLOWS_CHARS:
            # Strip bare gallows
            continue
        else:
            remaining.append(ch)
    return ''.join(remaining) if remaining else ''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_spatial_preprocess() -> None:
    """Step 35.1: Spatial gallows preprocessing."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 35.1: Spatial Gallows Preprocessing")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load corpus ──
    print("\n  1. Loading corpus ...")
    corpus = load_corpus(verbose=False)

    all_tokens: List[str] = []
    token_folios: List[str] = []
    token_sections: List[str] = []

    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
            token_sections.append(page.section)

    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens")

    # ── 2. Classify gallows spatially and condition tokens ──
    print("\n  2. Classifying gallows and conditioning tokens ...")
    token_conditioned: List[str] = []
    token_strategies: List[str] = []
    token_determinatives: List[Optional[str]] = []

    strategy_counter: Counter = Counter()

    for idx, token in enumerate(all_tokens):
        eva_chars = tokenize_eva_chars(token)
        classifications = _classify_gallows_spatial(token, eva_chars)

        conditioned, strategy, determinative = _apply_spatial_conditioning(
            token, eva_chars, classifications,
        )

        token_conditioned.append(conditioned)
        token_strategies.append(strategy)
        token_determinatives.append(determinative)
        strategy_counter[strategy] += 1

    conditioning_rate = 1.0 - strategy_counter.get('identity', 0) / n_tokens

    print(f"     Conditioning rate: {conditioning_rate:.1%}")
    for strat in ['identity', 'intersecting_kept', 'strip_preceding',
                   'strip_following', 'strip_mixed', 'standalone_silent']:
        n = strategy_counter.get(strat, 0)
        print(f"       {strat:20s}: {n:6d} ({n / n_tokens:.1%})")

    # Token length stats
    orig_lens = [len(tokenize_eva_chars(t)) for t in all_tokens]
    cond_lens = [len(tokenize_eva_chars(t)) if t else 0 for t in token_conditioned]
    mean_orig = sum(orig_lens) / len(orig_lens) if orig_lens else 0
    mean_cond = sum(cond_lens) / len(cond_lens) if cond_lens else 0
    print(f"\n     Mean EVA-char length: {mean_orig:.2f} → {mean_cond:.2f}")

    # Determinative stats
    det_counter: Counter = Counter()
    for d in token_determinatives:
        if d:
            det_counter[d] += 1
    print(f"     Determinatives: {dict(det_counter)}")

    # ── 3. Condition null corpora ──
    print("\n  3. Conditioning null corpora ...")
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )

    null_conditioned: List[List[str]] = []
    for i, seed in enumerate(null_seeds):
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        conditioned_null = [_condition_null_token(t) for t in null_tokens]
        null_conditioned.append(conditioned_null)
        n_empty = sum(1 for t in conditioned_null if t == '')
        print(f"     Null {i + 1} (seed={seed}): {n_empty} empty tokens")

    # ── 4. Save ──
    print("\n  4. Saving spatial_preprocess.json ...")
    elapsed = round(time.time() - t0, 2)

    spatial_stats = {
        'n_tokens_total': n_tokens,
        'n_identity': strategy_counter.get('identity', 0),
        'n_intersecting_kept': strategy_counter.get('intersecting_kept', 0),
        'n_strip_preceding': strategy_counter.get('strip_preceding', 0),
        'n_strip_following': strategy_counter.get('strip_following', 0),
        'n_strip_mixed': strategy_counter.get('strip_mixed', 0),
        'n_standalone_silent': strategy_counter.get('standalone_silent', 0),
        'mean_len_original': round(mean_orig, 3),
        'mean_len_conditioned': round(mean_cond, 3),
        'determinative_counts': dict(det_counter),
    }

    output = {
        'spatial_stats': spatial_stats,
        'n_tokens': n_tokens,
        'token_folios': token_folios,
        'token_evas': all_tokens,
        'token_conditioned': token_conditioned,
        'token_strategies': token_strategies,
        'token_determinatives': token_determinatives,
        'token_sections': token_sections,
        'conditioning_rate': round(conditioning_rate, 6),
        'null_conditioned': null_conditioned,
        'null_seeds': null_seeds,
        'verdict': (
            f"Conditioning rate {conditioning_rate:.1%}: "
            f"{strategy_counter.get('strip_preceding', 0)} PRECEDING stripped, "
            f"{strategy_counter.get('intersecting_kept', 0)} INTERSECTING kept, "
            f"{strategy_counter.get('standalone_silent', 0)} STANDALONE silent"
        ),
        'runtime_seconds': elapsed,
    }

    out_path = os.path.join(rd, 'spatial_preprocess.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {elapsed:.1f}s")
