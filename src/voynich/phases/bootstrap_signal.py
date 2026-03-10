"""
Phase 30.2 – Post-Bootstrap Signal Re-Isolation
===================================================
Re-runs the full signal isolation (Phase 28.4 methodology) with the
expanded confirmed vocabulary and any new triple assignments from the
bootstrap loop.

Dependency chain:
    bootstrap_loop.json        (Step 30.1 — evolved assignment)
    combined_refine.json       (Phase 15 — fallback assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 seeds)
    crib_extraction.json       (Phase 28.1 — crib words)
    signal_isolation.json      (Phase 28.4 — baseline for comparison)
        → bootstrap_signal.json   (this step)
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
    load_corpus,
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
from voynich.phases.signal_isolation import _decode_corpus_r3


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
class FolioSignal:
    folio: str
    n_tokens: int
    n_signal: int
    signal_rate: float


@dataclass
class BootstrapSignalResult:
    # Per-word signal
    n_words_tested: int
    n_genuine_signals: int
    n_artifacts: int
    word_signals: List[Dict]
    # Token-level classification
    n_signal_tokens: int
    signal_token_rate: float
    n_shared_hit: int
    n_shared_miss: int
    n_anti_signal: int
    # Folio distribution
    top_signal_folios: List[Dict]
    # Null corpora info
    null_n_corpora: int
    null_seeds: List[int]
    # Bootstrap comparison
    baseline_n_genuine: int
    baseline_signal_rate: float
    delta_n_genuine: int
    delta_signal_rate: float
    new_signal_words: List[str]
    lost_signal_words: List[str]
    # Assignment info
    assignment_source: str
    n_triples_changed: int
    # Summary
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bootstrap_signal() -> None:
    """Step 30.2: Re-isolate signal with post-bootstrap assignment."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 30.2: Post-Bootstrap Signal Re-Isolation")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load assignment ──
    print("\n  1. Loading assignment …")

    boot_path = os.path.join(rd, 'bootstrap_loop.json')
    refine_path = os.path.join(rd, 'combined_refine.json')

    if os.path.exists(boot_path):
        with open(boot_path) as f:
            boot_data = json.load(f)
        assignment = boot_data.get('final_assignment', {})
        n_changed = boot_data.get('n_total_accepted', 0)
        accepted_words = set(boot_data.get('accepted_words', []))
        source = 'bootstrap_loop.json'
        print(f"     Using bootstrap assignment ({n_changed} changes)")
    elif os.path.exists(refine_path):
        with open(refine_path) as f:
            refine_data = json.load(f)
        assignment = refine_data.get('best_assignment', {})
        n_changed = 0
        accepted_words = set()
        source = 'combined_refine.json (fallback)'
        print(f"     Using Phase 15 assignment (no bootstrap)")
    else:
        print("  [SKIP] No assignment found")
        return

    # ── 2. Load modifiers ──
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Null seeds
    null_path = os.path.join(rd, 'null_corpus.json')
    null_seeds = [100, 101, 102, 103, 104]
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # Crib words (extend with bootstrap-accepted words)
    crib_path = os.path.join(rd, 'crib_extraction.json')
    crib_words: List[str] = []
    if os.path.exists(crib_path):
        with open(crib_path) as f:
            crib_data = json.load(f)
        crib_words = [c['word'] for c in crib_data.get('cribs', [])]
    test_words = sorted(set(crib_words) | accepted_words)

    # Baseline signal isolation
    sig_path = os.path.join(rd, 'signal_isolation.json')
    baseline_genuine = 0
    baseline_signal_rate = 0.0
    baseline_signal_words: Set[str] = set()
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        baseline_genuine = sig_data.get('n_genuine_signals', 0)
        baseline_signal_rate = sig_data.get('signal_token_rate', 0.0)
        baseline_signal_words = {
            ws['word'] for ws in sig_data.get('word_signals', [])
            if ws.get('is_genuine_signal', False)
        }

    print(f"     Test words: {len(test_words)}")
    print(f"     Baseline genuine signals: {baseline_genuine}")

    # ── 3. Build reference word set ──
    print("\n  2. Building reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 4. Decode real corpus ──
    print("\n  3. Decoding real corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens: List[str] = []
    all_folios: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            all_folios.append(folio)
    n_tokens = len(all_tokens)

    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]
    real_hit_rate = sum(real_hits) / n_tokens
    print(f"     {n_tokens} tokens, dict_hit = {real_hit_rate:.4f}")

    # ── 5. Regenerate and decode null corpora ──
    print("\n  4. Regenerating and decoding null corpora …")
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

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
        null_hits_list.append(null_hits)
        null_rate = sum(null_hits) / len(null_hits)
        print(f"       dict_hit = {null_rate:.4f}")

    # ── 6. Per-word signal analysis ──
    print("\n  5. Per-word signal analysis …")
    real_word_counts = Counter(w for w, hit in zip(real_decoded, real_hits) if hit)

    null_decoded_list: List[List[str]] = []
    # Re-decode nulls to get word-level counts
    for i, seed in enumerate(null_seeds):
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_decoded = _decode_corpus_r3(
            null_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        null_decoded_list.append(null_decoded)

    word_signals: List[WordSignal] = []
    for word in test_words:
        real_count = real_word_counts.get(word, 0)
        null_counts = []
        for nd in null_decoded_list:
            null_counts.append(Counter(nd).get(word, 0))

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
        boot_tag = ' [BOOT]' if ws.word in accepted_words else ''
        print(f"    {tag} {ws.word:12s}  real={ws.real_count:4d}  "
              f"null={ws.null_mean_count:6.1f}±{ws.null_std_count:4.1f}  "
              f"σ={ws.signal_sigma:6.1f}  sel={ws.selectivity:.2f}{boot_tag}")

    # ── 7. Token-level classification ──
    print("\n  6. Token-level classification …")
    n_signal = 0
    n_shared_hit = 0
    n_shared_miss = 0
    n_anti_signal = 0

    folio_token_indices: Dict[str, List[int]] = defaultdict(list)
    for idx in range(n_tokens):
        folio_token_indices[all_folios[idx]].append(idx)

    signal_positions: Set[int] = set()
    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if nh[idx])

        if r_hit and null_hit_count <= 1:
            n_signal += 1
            signal_positions.add(idx)
        elif r_hit and null_hit_count >= 3:
            n_shared_hit += 1
        elif not r_hit and null_hit_count >= 3:
            n_anti_signal += 1
        else:
            n_shared_miss += 1

    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    print(f"     SIGNAL:      {n_signal:6d} ({signal_rate:.1%})")
    print(f"     SHARED_HIT:  {n_shared_hit:6d}")
    print(f"     SHARED_MISS: {n_shared_miss:6d}")
    print(f"     ANTI_SIGNAL: {n_anti_signal:6d}")

    # ── 8. Folio distribution ──
    print("\n  7. Folio signal distribution (top 10) …")
    folio_signals: List[FolioSignal] = []
    for folio, indices in sorted(folio_token_indices.items()):
        n_f = len(indices)
        n_s = sum(1 for i in indices if i in signal_positions)
        if n_f > 0:
            folio_signals.append(FolioSignal(
                folio=folio, n_tokens=n_f, n_signal=n_s,
                signal_rate=round(n_s / n_f, 4),
            ))
    folio_signals.sort(key=lambda f: -f.signal_rate)
    for fs in folio_signals[:10]:
        print(f"     {fs.folio:8s}  {fs.n_signal:3d}/{fs.n_tokens:3d}  "
              f"({fs.signal_rate:.1%})")

    # ── 9. Comparison ──
    new_signal_words_set = {
        ws.word for ws in word_signals if ws.is_genuine_signal
    }
    new_signals = sorted(new_signal_words_set - baseline_signal_words)
    lost_signals = sorted(baseline_signal_words - new_signal_words_set)

    delta_genuine = n_genuine - baseline_genuine
    delta_rate = signal_rate - baseline_signal_rate

    print(f"\n  8. Comparison to Phase 28 baseline …")
    print(f"     Genuine signals: {baseline_genuine} → {n_genuine} (Δ={delta_genuine:+d})")
    print(f"     Signal rate: {baseline_signal_rate:.4f} → {signal_rate:.4f} "
          f"(Δ={delta_rate:+.4f})")
    if new_signals:
        print(f"     New: {', '.join(new_signals)}")
    if lost_signals:
        print(f"     Lost: {', '.join(lost_signals)}")

    # ── 10. Gate ──
    gate = n_genuine >= 8 and delta_rate >= -0.005
    if gate:
        verdict = f"SIGNAL_MAINTAINED ({n_genuine} genuine, Δrate={delta_rate:+.4f})"
    else:
        verdict = f"SIGNAL_REGRESSION ({n_genuine} genuine, Δrate={delta_rate:+.4f})"

    print(f"\n     Verdict: {verdict}")
    print(f"     Gate: {'PASS' if gate else 'FAIL'}")

    result = BootstrapSignalResult(
        n_words_tested=len(test_words),
        n_genuine_signals=n_genuine,
        n_artifacts=n_artifacts,
        word_signals=[_convert(asdict(ws)) for ws in word_signals],
        n_signal_tokens=n_signal,
        signal_token_rate=round(signal_rate, 6),
        n_shared_hit=n_shared_hit,
        n_shared_miss=n_shared_miss,
        n_anti_signal=n_anti_signal,
        top_signal_folios=[_convert(asdict(fs)) for fs in folio_signals[:20]],
        null_n_corpora=len(null_seeds),
        null_seeds=null_seeds,
        baseline_n_genuine=baseline_genuine,
        baseline_signal_rate=round(baseline_signal_rate, 6),
        delta_n_genuine=delta_genuine,
        delta_signal_rate=round(delta_rate, 6),
        new_signal_words=new_signals,
        lost_signal_words=lost_signals,
        assignment_source=source,
        n_triples_changed=n_changed,
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    out_path = os.path.join(rd, 'bootstrap_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
