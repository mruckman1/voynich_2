"""
Phase 30.4 – Post-Bootstrap Context Analysis
================================================
Re-runs Phase 29.2's context exploitation with the expanded confirmed
vocabulary from the bootstrap.  Feeds back into bootstrap_loop.py for
potential further iterations.

Dependency chain:
    bootstrap_bigrams.json     (Step 30.3 — new per-token cache)
    bootstrap_signal.json      (Step 30.2 — expanded signal words)
    bootstrap_loop.json        (Step 30.1 — accepted words)
    signal_context.json        (Phase 29.2 — baseline for comparison)
    signal_isolation.json      (Phase 28.4 — original signal words)
        → bootstrap_context.json  (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.signal_context import (
    _extract_context_windows,
    _find_chains,
    _identify_new_cribs,
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
class ConfirmedPair:
    word1: str
    word2: str
    count: int
    folios: List[str]


@dataclass
class BootstrapContextResult:
    context_windows: List[Dict]
    n_new_crib_candidates: int
    new_crib_candidates: List[Dict]
    chain_candidates: List[Dict]
    n_chains_found: int
    longest_chain: int
    # Confirmed-confirmed pairs
    confirmed_pairs: List[Dict]
    n_confirmed_pairs: int
    # Bootstrap comparison
    baseline_n_new_cribs: int
    baseline_longest_chain: int
    delta_n_new_cribs: int
    delta_longest_chain: int
    # Bootstrap-accepted word context
    accepted_word_context: List[Dict]
    # Summary
    signal_words_used: List[str]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bootstrap_context() -> None:
    """Step 30.4: Post-bootstrap context analysis."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 30.4: Post-Bootstrap Context Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load per-token cache (from bootstrap_bigrams or signal_bigrams) ──
    print("\n  1. Loading per-token cache …")
    boot_bg_path = os.path.join(rd, 'bootstrap_bigrams.json')
    bg_path = os.path.join(rd, 'signal_bigrams.json')

    if os.path.exists(boot_bg_path):
        with open(boot_bg_path) as f:
            bg_data = json.load(f)
        print("     Using bootstrap_bigrams.json")
    elif os.path.exists(bg_path):
        with open(bg_path) as f:
            bg_data = json.load(f)
        print("     Using signal_bigrams.json (fallback)")
    else:
        print("  [SKIP] No per-token cache found")
        return

    decoded = bg_data['token_decoded']
    classifications = bg_data['token_classifications']
    folios = bg_data['token_folios']
    dict_hits = bg_data['token_dict_hits']

    # ── 2. Load expanded signal words ──
    print("\n  2. Loading signal words …")

    # Original signal words from Phase 28
    sig_path = os.path.join(rd, 'signal_isolation.json')
    original_signal_words: Set[str] = set()
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        original_signal_words = {
            ws['word'] for ws in sig_data.get('word_signals', [])
            if ws.get('is_genuine_signal', False)
        }

    # Bootstrap signal words (may include new ones)
    boot_sig_path = os.path.join(rd, 'bootstrap_signal.json')
    bootstrap_signal_words: Set[str] = set()
    if os.path.exists(boot_sig_path):
        with open(boot_sig_path) as f:
            boot_sig_data = json.load(f)
        bootstrap_signal_words = {
            ws['word'] for ws in boot_sig_data.get('word_signals', [])
            if ws.get('is_genuine_signal', False)
        }

    # Bootstrap-accepted words
    boot_loop_path = os.path.join(rd, 'bootstrap_loop.json')
    accepted_words: Set[str] = set()
    if os.path.exists(boot_loop_path):
        with open(boot_loop_path) as f:
            boot_loop_data = json.load(f)
        accepted_words = set(boot_loop_data.get('accepted_words', []))

    # Combine: original signals + bootstrap signals + accepted words
    all_signal_words = sorted(original_signal_words | bootstrap_signal_words | accepted_words)
    print(f"     Original signals: {len(original_signal_words)}")
    print(f"     Bootstrap signals: {len(bootstrap_signal_words)}")
    print(f"     Accepted words: {len(accepted_words)}")
    print(f"     Total signal words for context: {len(all_signal_words)}")

    # ── 3. Load baseline for comparison ──
    ctx_path = os.path.join(rd, 'signal_context.json')
    baseline_n_cribs = 0
    baseline_longest = 0
    if os.path.exists(ctx_path):
        with open(ctx_path) as f:
            ctx_baseline = json.load(f)
        baseline_n_cribs = ctx_baseline.get('n_new_crib_candidates', 0)
        baseline_longest = ctx_baseline.get('longest_chain', 0)

    # ── 4. Build reference word set ──
    print("\n  3. Building reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # ── 5. Extract context windows ──
    print("\n  4. Extracting context windows …")
    windows = _extract_context_windows(
        all_signal_words, decoded, classifications, folios, ref_word_set,
    )
    for w in windows:
        if w.n_occurrences > 0:
            print(f"     {w.signal_word:12s}  occ={w.n_occurrences:4d}  "
                  f"ctx_hit={w.context_dict_hit_rate:.3f}")

    # ── 6. Identify new cribs ──
    print("\n  5. Identifying new crib candidates …")
    # Exclude already-confirmed words from candidates
    exclude_words = set(all_signal_words)
    new_cribs = _identify_new_cribs(
        windows, ref_word_set, list(exclude_words),
    )
    print(f"     {len(new_cribs)} new crib candidates")
    for nc in new_cribs[:10]:
        print(f"       {nc.word:12s}  assoc={nc.n_signal_word_associations}  "
              f"PMI={nc.mean_pmi:.2f}  count={nc.total_count}")

    # ── 7. Find chains ──
    print("\n  6. Finding chains …")
    chains = _find_chains(decoded, classifications, folios, dict_hits)
    longest_chain = max((c.length for c in chains), default=0)
    print(f"     {len(chains)} chains, longest={longest_chain}")
    for ch in chains[:5]:
        print(f"       {ch.folio}  len={ch.length}  sig={ch.n_signal}  "
              f"words={' '.join(ch.words[:8])}")

    # ── 8. Confirmed-confirmed pairs ──
    print("\n  7. Finding confirmed-confirmed pairs …")
    confirmed_set = set(all_signal_words)
    conf_pairs: List[ConfirmedPair] = []
    pair_counter: Counter = Counter()
    pair_folios: Dict[Tuple[str, str], List[str]] = {}

    for i in range(len(decoded) - 1):
        if folios[i] != folios[i + 1]:
            continue
        w1, w2 = decoded[i], decoded[i + 1]
        if w1 in confirmed_set and w2 in confirmed_set:
            key = (w1, w2)
            pair_counter[key] += 1
            if key not in pair_folios:
                pair_folios[key] = []
            if folios[i] not in pair_folios[key]:
                pair_folios[key].append(folios[i])

    for (w1, w2), count in pair_counter.most_common(20):
        conf_pairs.append(ConfirmedPair(
            word1=w1, word2=w2, count=count,
            folios=pair_folios[(w1, w2)][:5],
        ))
    print(f"     {len(conf_pairs)} confirmed-confirmed pairs")
    for cp in conf_pairs[:5]:
        print(f"       {cp.word1} {cp.word2}  count={cp.count}  "
              f"folios={','.join(cp.folios[:3])}")

    # ── 9. Comparison ──
    delta_cribs = len(new_cribs) - baseline_n_cribs
    delta_chain = longest_chain - baseline_longest

    print(f"\n  8. Comparison to Phase 29 baseline …")
    print(f"     New cribs: {baseline_n_cribs} → {len(new_cribs)} (Δ={delta_cribs:+d})")
    print(f"     Longest chain: {baseline_longest} → {longest_chain} (Δ={delta_chain:+d})")

    # Gate
    gate = len(new_cribs) >= 1 or longest_chain >= baseline_longest
    if len(new_cribs) >= 5 and longest_chain > baseline_longest:
        verdict = f"CONTEXT_EXPANDED ({len(new_cribs)} new cribs, chain={longest_chain})"
    elif len(new_cribs) >= 1:
        verdict = f"CONTEXT_STABLE ({len(new_cribs)} new cribs)"
    else:
        verdict = "CONTEXT_EXHAUSTED (no new cribs)"

    print(f"\n     Verdict: {verdict}")
    print(f"     Gate: {'PASS' if gate else 'FAIL'}")

    result = BootstrapContextResult(
        context_windows=[_convert(asdict(w)) for w in windows],
        n_new_crib_candidates=len(new_cribs),
        new_crib_candidates=[_convert(asdict(nc)) for nc in new_cribs[:30]],
        chain_candidates=[_convert(asdict(ch)) for ch in chains[:50]],
        n_chains_found=len(chains),
        longest_chain=longest_chain,
        confirmed_pairs=[_convert(asdict(cp)) for cp in conf_pairs],
        n_confirmed_pairs=len(conf_pairs),
        baseline_n_new_cribs=baseline_n_cribs,
        baseline_longest_chain=baseline_longest,
        delta_n_new_cribs=delta_cribs,
        delta_longest_chain=delta_chain,
        accepted_word_context=[
            _convert(asdict(w)) for w in windows
            if w.signal_word in accepted_words
        ],
        signal_words_used=all_signal_words,
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    out_path = os.path.join(rd, 'bootstrap_context.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
