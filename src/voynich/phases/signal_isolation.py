"""
Phase 28.4 – Signal Isolation
================================
Identifies which confirmed crib words carry genuine real-vs-null signal
by comparing their frequency in the real corpus vs. 5 regenerated null
corpora.  Also classifies each token position as SIGNAL (real hit, null
miss) or SHARED_HIT / SHARED_MISS / ANTI_SIGNAL.

Independent of Steps 28.2–28.3; depends only on 28.1 + Phase 15/16/17.

Dependency chain:
    crib_extraction.json     (Step 28.1)
    combined_refine.json     (Phase 15 assignment)
    modifier_integrate.json  (Phase 16 modifiers)
    null_corpus.json         (Phase 17 seeds)
        → signal_isolation.json  (this step)
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
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WordSignal:
    word: str
    real_count: int
    null_mean_count: float
    null_std_count: float
    signal_sigma: float
    real_rate: float
    null_rate_mean: float
    selectivity: float
    is_genuine_signal: bool


@dataclass
class TokenClassification:
    """Aggregate counts for each token classification."""
    n_signal: int              # real hit, ≥4/5 null miss
    n_shared_hit: int          # real hit, ≥3/5 null also hit
    n_shared_miss: int         # real miss, null miss
    n_anti_signal: int         # real miss, ≥3/5 null hit


@dataclass
class FolioSignal:
    folio: str
    n_tokens: int
    n_signal: int
    signal_rate: float


@dataclass
class SignalIsolationResult:
    # Per-word signal
    n_words_tested: int
    n_genuine_signals: int
    n_artifacts: int
    word_signals: List[Dict]
    # Token-level classification
    token_classification: Dict
    n_signal_tokens: int
    signal_token_rate: float
    # Folio distribution
    top_signal_folios: List[Dict]
    # Null corpora info
    null_n_corpora: int
    null_seeds: List[int]
    # Overlap with crib pool
    n_crib_at_signal: int
    n_crib_at_shared: int
    # Summary
    mean_selectivity: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 strategy: try alteration, then strip, then raw."""
    decoded = []
    for token in tokens:
        # Alteration
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        # Strip
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        # Raw
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_signal_isolation() -> None:
    """Step 28.4: Signal isolation — real vs null per-word and per-token."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 28.4: Signal Isolation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    # Phase 15 assignment (used by Phase 16)
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    # Phase 16 modifiers
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Crib extraction — confirmed words
    crib_path = os.path.join(rd, 'crib_extraction.json')
    crib_words: List[str] = []
    if os.path.exists(crib_path):
        with open(crib_path) as f:
            crib_data = json.load(f)
        crib_words = [c['word'] for c in crib_data.get('cribs', [])]

    # Null corpus seeds
    null_path = os.path.join(rd, 'null_corpus.json')
    null_seeds = [100, 101, 102, 103, 104]
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]
    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")
    print(f"     Crib words: {len(crib_words)}")
    print(f"     Null seeds: {null_seeds}")

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 3. Decode real corpus ──
    print("\n  3. Decoding real corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)

    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]
    real_hit_rate = sum(real_hits) / n_tokens
    print(f"     {n_tokens} tokens, dict_hit = {real_hit_rate:.3f}")

    # ── 4. Regenerate and decode null corpora ──
    print("\n  4. Regenerating and decoding null corpora …")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_decoded_list: List[List[str]] = []
    null_hits_list: List[List[bool]] = []

    for i, seed in enumerate(null_seeds):
        print(f"     Null corpus {i + 1}/{len(null_seeds)} (seed={seed}) …")
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_hits = [w in ref_word_set for w in null_decoded]
        null_decoded_list.append(null_decoded)
        null_hits_list.append(null_hits)
        null_rate = sum(null_hits) / len(null_hits)
        print(f"       dict_hit = {null_rate:.3f}")

    # ── 5. Per-word signal analysis ──
    print("\n  5. Per-word signal analysis …")
    real_word_counts = Counter(w for w, hit in zip(real_decoded, real_hits) if hit)

    # Count each crib word in null corpora
    word_signals: List[WordSignal] = []
    test_words = sorted(set(crib_words) | set(real_word_counts.keys()))
    # Focus on crib words only to keep output manageable
    test_words = sorted(set(crib_words))

    for word in test_words:
        real_count = real_word_counts.get(word, 0)
        null_counts = []
        for null_decoded in null_decoded_list:
            null_word_counts = Counter(null_decoded)
            null_counts.append(null_word_counts.get(word, 0))

        null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
        null_var = (sum((c - null_mean) ** 2 for c in null_counts)
                    / len(null_counts) if null_counts else 0.0)
        null_std = null_var ** 0.5

        sigma = ((real_count - null_mean) / null_std) if null_std > 0 else (
            float('inf') if real_count > null_mean else 0.0
        )

        real_rate = real_count / n_tokens
        null_rate_mean = null_mean / n_tokens
        selectivity = real_rate / null_rate_mean if null_rate_mean > 0 else float('inf')

        is_genuine = sigma > 2.0

        word_signals.append(WordSignal(
            word=word,
            real_count=real_count,
            null_mean_count=round(null_mean, 2),
            null_std_count=round(null_std, 2),
            signal_sigma=round(sigma, 2) if sigma != float('inf') else 999.0,
            real_rate=round(real_rate, 6),
            null_rate_mean=round(null_rate_mean, 6),
            selectivity=round(selectivity, 2) if selectivity != float('inf') else 999.0,
            is_genuine_signal=is_genuine,
        ))

    word_signals.sort(key=lambda w: -w.signal_sigma)
    n_genuine = sum(1 for w in word_signals if w.is_genuine_signal)
    n_artifacts = sum(1 for w in word_signals if not w.is_genuine_signal)

    for ws in word_signals:
        tag = '★' if ws.is_genuine_signal else '○'
        print(f"    {tag} {ws.word:12s}  real={ws.real_count:4d}  "
              f"null={ws.null_mean_count:6.1f}±{ws.null_std_count:4.1f}  "
              f"σ={ws.signal_sigma:6.1f}  sel={ws.selectivity:.2f}")

    # ── 6. Token-level classification ──
    print("\n  6. Token-level classification …")
    n_signal = 0
    n_shared_hit = 0
    n_shared_miss = 0
    n_anti_signal = 0

    # Track which folio each token is on
    folio_tokens: Dict[str, List[int]] = defaultdict(list)
    token_idx = 0
    for folio, page in corpus.pages.items():
        page_tokens = page.all_tokens
        for _ in page_tokens:
            folio_tokens[folio].append(token_idx)
            token_idx += 1

    signal_positions: Set[int] = set()

    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if nh[idx])

        if r_hit and null_hit_count <= 1:
            n_signal += 1
            signal_positions.add(idx)
        elif r_hit and null_hit_count >= 3:
            n_shared_hit += 1
        elif not r_hit and null_hit_count <= 1:
            n_shared_miss += 1
        elif not r_hit and null_hit_count >= 3:
            n_anti_signal += 1
        else:
            n_shared_miss += 1  # ambiguous → shared miss

    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    print(f"     SIGNAL:      {n_signal:6d} ({n_signal / n_tokens:.1%})")
    print(f"     SHARED_HIT:  {n_shared_hit:6d}")
    print(f"     SHARED_MISS: {n_shared_miss:6d}")
    print(f"     ANTI_SIGNAL: {n_anti_signal:6d}")

    # ── 7. Folio distribution of SIGNAL tokens ──
    print("\n  7. Folio signal distribution (top 10) …")
    folio_signal: List[FolioSignal] = []
    for folio, indices in sorted(folio_tokens.items()):
        n_f = len(indices)
        n_s = sum(1 for i in indices if i in signal_positions)
        if n_f > 0:
            folio_signal.append(FolioSignal(
                folio=folio, n_tokens=n_f, n_signal=n_s,
                signal_rate=round(n_s / n_f, 4),
            ))
    folio_signal.sort(key=lambda f: -f.signal_rate)

    for fs in folio_signal[:10]:
        print(f"     {fs.folio:8s}  {fs.n_signal:3d}/{fs.n_tokens:3d}  "
              f"({fs.signal_rate:.1%})")

    # ── 8. Crib pool overlap with signal positions ──
    print("\n  8. Crib pool ↔ signal overlap …")
    crib_word_set = set(crib_words)
    n_crib_at_signal = 0
    n_crib_at_shared = 0
    for idx in range(n_tokens):
        w = real_decoded[idx]
        if w in crib_word_set:
            if idx in signal_positions:
                n_crib_at_signal += 1
            elif real_hits[idx]:
                n_crib_at_shared += 1
    print(f"     Crib tokens at SIGNAL positions: {n_crib_at_signal}")
    print(f"     Crib tokens at SHARED_HIT positions: {n_crib_at_shared}")

    # ── 9. Gate and verdict ──
    finite_selectivities = [
        ws.selectivity for ws in word_signals
        if ws.selectivity < 900 and ws.is_genuine_signal
    ]
    mean_sel = (sum(finite_selectivities) / len(finite_selectivities)
                if finite_selectivities else 0.0)

    gate_passed = n_genuine >= 5
    verdict = (
        f"PASS: {n_genuine} genuine signal words (σ>2.0), "
        f"{n_signal} SIGNAL tokens ({signal_rate:.1%}), "
        f"mean selectivity {mean_sel:.2f}×"
        if gate_passed
        else f"FAIL: Only {n_genuine} genuine signal words (need ≥5)"
    )
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 10. Save ──
    result = SignalIsolationResult(
        n_words_tested=len(word_signals),
        n_genuine_signals=n_genuine,
        n_artifacts=n_artifacts,
        word_signals=[_convert(asdict(ws)) for ws in word_signals],
        token_classification={
            'n_signal': n_signal,
            'n_shared_hit': n_shared_hit,
            'n_shared_miss': n_shared_miss,
            'n_anti_signal': n_anti_signal,
        },
        n_signal_tokens=n_signal,
        signal_token_rate=round(signal_rate, 4),
        top_signal_folios=[_convert(asdict(fs)) for fs in folio_signal[:20]],
        null_n_corpora=len(null_seeds),
        null_seeds=null_seeds,
        n_crib_at_signal=n_crib_at_signal,
        n_crib_at_shared=n_crib_at_shared,
        mean_selectivity=round(mean_sel, 2),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'signal_isolation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
