"""
Step 43.13 – Viterbi Decoding
================================
Decode the entire corpus using the trained HMM via the Viterbi algorithm,
producing the most probable syllable sequence for each token.

Dependency chain:
    results/baum_welch_training.json  (Step 43.12: trained A, B, pi)
    results/hmm_architecture.json     (Step 43.10: state labels, obs vocab)
    results/combined_refine.json      (Phase 15: for comparison)
    results/modifier_integrate.json   (Phase 16: modifier chars)
    data/corpus/                      (EVA transcription)
        → viterbi_decode.json         (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ViterbiDecodeResult:
    n_tokens_decoded: int
    n_char_positions: int
    # Decoded text samples
    decoded_words_sample: List[str]
    eva_tokens_sample: List[str]
    # Dictionary matching
    dict_hit_rate: float
    dict_hit_count: int
    dict_size: int
    # Comparison with Phase 15
    agreement_rate: float
    n_agree: int
    n_disagree: int
    n_improved: int
    n_degraded: int
    changed_examples: List[Dict]
    # Per-folio results
    folio_results: List[Dict]
    best_folio: str
    best_folio_dict_hit: float
    # Consecutive hits
    max_consecutive_hits: int
    n_runs_ge_3: int
    n_runs_ge_5: int
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Viterbi algorithm
# ---------------------------------------------------------------------------

def _viterbi_token(
    obs: np.ndarray,
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    is_anchored: Optional[np.ndarray] = None,
    anchor_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Viterbi decoding in log-space for a single token's char sequence.

    Returns array of state indices (one per character position).
    """
    T = len(obs)
    K = len(log_pi)

    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=np.int32)

    # Initialization
    delta[0] = log_pi + log_B[:, obs[0]]

    # Anchor forcing at t=0
    if is_anchored is not None and is_anchored[0]:
        forced = anchor_state[0]
        mask = np.full(K, -np.inf)
        mask[forced] = 0.0
        delta[0] += mask

    # Recursion
    for t in range(1, T):
        for j in range(K):
            scores = delta[t - 1] + log_A[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] + log_B[j, obs[t]]

        # Anchor forcing
        if is_anchored is not None and is_anchored[t]:
            forced = anchor_state[t]
            mask = np.full(K, -np.inf)
            mask[forced] = 0.0
            delta[t] += mask

    # Backtracking
    states = np.zeros(T, dtype=np.int32)
    states[T - 1] = np.argmax(delta[T - 1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def _decode_corpus_hmm(
    corpus,
    state_labels: List[str],
    obs_vocab: List[str],
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    assignment: Dict[str, str],
    modifier_chars: Set[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Decode entire corpus using Viterbi.

    Returns (hmm_decoded_words, phase15_decoded_words, eva_tokens, folios).
    """
    char_to_idx = {ch: i for i, ch in enumerate(obs_vocab)}
    unk_idx = len(obs_vocab) - 1
    eva_to_triple = build_eva_to_triple_lookup()

    hmm_decoded: List[str] = []
    p15_decoded: List[str] = []
    eva_tokens: List[str] = []
    folios: List[str] = []

    for folio_id, page in corpus.pages.items():
        for token in page.all_tokens:
            chars = tokenize_eva_chars(token)
            if not chars:
                continue

            # Encode characters to observation indices
            obs = np.array([char_to_idx.get(ch, unk_idx) for ch in chars],
                          dtype=np.int32)

            # Viterbi decode
            states = _viterbi_token(obs, log_pi, log_A, log_B)

            # Map states to syllables
            syllables = [state_labels[s] for s in states]
            hmm_word = ''.join(syllables)

            # Phase 15 decode for comparison
            p15_word = decode_token_modifier_aware(
                token, assignment, eva_to_triple, modifier_chars
            )

            hmm_decoded.append(hmm_word)
            p15_decoded.append(p15_word)
            eva_tokens.append(token)
            folios.append(folio_id)

    return hmm_decoded, p15_decoded, eva_tokens, folios


def _count_consecutive_runs(hits: List[bool]) -> Tuple[int, int, int]:
    """Count maximum consecutive run and runs >= 3 and >= 5."""
    max_run = 0
    runs_ge_3 = 0
    runs_ge_5 = 0
    current = 0
    for h in hits:
        if h:
            current += 1
        else:
            if current >= 3:
                runs_ge_3 += 1
            if current >= 5:
                runs_ge_5 += 1
            max_run = max(max_run, current)
            current = 0
    # Handle trailing run
    if current >= 3:
        runs_ge_3 += 1
    if current >= 5:
        runs_ge_5 += 1
    max_run = max(max_run, current)
    return max_run, runs_ge_3, runs_ge_5


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_viterbi_decode() -> None:
    """Step 43.13: decode corpus via Viterbi with trained HMM."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.13: Viterbi Decoding")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load trained HMM ──
    print("\n  1. Loading trained HMM …")
    bw = _safe_load(os.path.join(rd, 'baum_welch_training.json'))
    hmm_arch = _safe_load(os.path.join(rd, 'hmm_architecture.json'))

    state_labels = hmm_arch.get('state_labels', [])
    obs_vocab = hmm_arch.get('observation_vocab', [])
    K = len(state_labels)
    V = len(obs_vocab)

    pi = np.array(bw.get('pi', [1.0 / K] * K))
    A = np.array(bw.get('A', [[1.0 / K] * K] * K))
    B = np.array(bw.get('B', [[1.0 / V] * V] * K))

    # Convert to log space
    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_B = np.log(B + 1e-300)

    print(f"     K={K}, V={V}")
    print(f"     Trained LL: {bw.get('final_log_likelihood', 'N/A')}")

    # ── 2. Load Phase 15 assignment and modifiers ──
    print("\n  2. Loading Phase 15 assignment …")
    combined = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = combined.get('best_assignment', {})
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))

    # ── 3. Build dictionary ──
    print("\n  3. Building dictionary …")
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(ref.get_combined_tokens('latin'))
        expanded, _ = build_expanded_word_set(base_words)
    except Exception:
        expanded = set()
    print(f"     Dictionary size: {len(expanded):,}")

    # ── 4. Decode corpus ──
    print("\n  4. Decoding corpus via Viterbi …")
    corpus = load_corpus(verbose=False)
    hmm_decoded, p15_decoded, eva_tokens, folios = _decode_corpus_hmm(
        corpus, state_labels, obs_vocab,
        log_pi, log_A, log_B,
        assignment, modifier_chars,
    )
    n_tokens = len(hmm_decoded)
    print(f"     Decoded {n_tokens:,} tokens")

    # ── 5. Dictionary matching ──
    print("\n  5. Dictionary matching …")
    hmm_hits = [w.lower() in expanded for w in hmm_decoded]
    p15_hits = [w.lower() in expanded for w in p15_decoded]

    hmm_hit_count = sum(hmm_hits)
    p15_hit_count = sum(p15_hits)
    hmm_hit_rate = hmm_hit_count / n_tokens if n_tokens > 0 else 0.0
    p15_hit_rate = p15_hit_count / n_tokens if n_tokens > 0 else 0.0

    print(f"     HMM dict-hit: {hmm_hit_count:,} ({hmm_hit_rate:.1%})")
    print(f"     Phase 15 dict-hit: {p15_hit_count:,} ({p15_hit_rate:.1%})")

    # ── 6. Compare HMM vs Phase 15 ──
    print("\n  6. Comparing with Phase 15 …")
    n_agree = sum(1 for h, p in zip(hmm_decoded, p15_decoded) if h == p)
    n_disagree = n_tokens - n_agree
    agreement_rate = n_agree / n_tokens if n_tokens > 0 else 0.0

    n_improved = sum(1 for hh, ph in zip(hmm_hits, p15_hits) if hh and not ph)
    n_degraded = sum(1 for hh, ph in zip(hmm_hits, p15_hits) if not hh and ph)

    print(f"     Agreement: {n_agree:,} ({agreement_rate:.1%})")
    print(f"     Improved (HMM hit, P15 miss): {n_improved:,}")
    print(f"     Degraded (HMM miss, P15 hit): {n_degraded:,}")

    # Collect changed examples
    changed_examples = []
    for i, (h, p, e) in enumerate(zip(hmm_decoded, p15_decoded, eva_tokens)):
        if h != p and len(changed_examples) < 20:
            changed_examples.append({
                'eva': e,
                'hmm': h,
                'phase15': p,
                'hmm_hit': hmm_hits[i],
                'p15_hit': p15_hits[i],
            })

    # ── 7. Per-folio analysis ──
    print("\n  7. Per-folio analysis …")
    folio_stats: Dict[str, Dict[str, int]] = {}
    for i, f in enumerate(folios):
        if f not in folio_stats:
            folio_stats[f] = {'total': 0, 'hits': 0}
        folio_stats[f]['total'] += 1
        if hmm_hits[i]:
            folio_stats[f]['hits'] += 1

    folio_results = []
    for f_id, stats in sorted(folio_stats.items()):
        rate = stats['hits'] / stats['total'] if stats['total'] > 0 else 0.0
        folio_results.append({
            'folio': f_id,
            'n_tokens': stats['total'],
            'dict_hit': round(rate, 4),
        })

    best_folio_entry = max(folio_results, key=lambda x: x['dict_hit']) if folio_results else {}
    best_folio = best_folio_entry.get('folio', '')
    best_folio_dict_hit = best_folio_entry.get('dict_hit', 0.0)

    print(f"     Best folio: {best_folio} ({best_folio_dict_hit:.1%})")

    # ── 8. Consecutive hit analysis ──
    print("\n  8. Consecutive hit analysis …")
    max_run, runs_3, runs_5 = _count_consecutive_runs(hmm_hits)
    print(f"     Max consecutive hits: {max_run}")
    print(f"     Runs ≥ 3: {runs_3}")
    print(f"     Runs ≥ 5: {runs_5}")

    # ── 9. Save ──
    elapsed = time.time() - t0

    result = ViterbiDecodeResult(
        n_tokens_decoded=n_tokens,
        n_char_positions=sum(len(tokenize_eva_chars(t)) for t in eva_tokens),
        decoded_words_sample=hmm_decoded[:100],
        eva_tokens_sample=eva_tokens[:100],
        dict_hit_rate=round(hmm_hit_rate, 6),
        dict_hit_count=hmm_hit_count,
        dict_size=len(expanded),
        agreement_rate=round(agreement_rate, 4),
        n_agree=n_agree,
        n_disagree=n_disagree,
        n_improved=n_improved,
        n_degraded=n_degraded,
        changed_examples=changed_examples,
        folio_results=folio_results[:30],
        best_folio=best_folio,
        best_folio_dict_hit=round(best_folio_dict_hit, 4),
        max_consecutive_hits=max_run,
        n_runs_ge_3=runs_3,
        n_runs_ge_5=runs_5,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'viterbi_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path} ({elapsed:.1f}s)")
