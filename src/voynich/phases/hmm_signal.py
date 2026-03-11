"""
Step 43.14 – HMM Signal Isolation and Validation
====================================================
Run signal isolation and validated bigram test on the HMM-decoded corpus.
Decode null corpora through the same trained HMM, classify tokens, and
compute symmetric bigram z-score.

Dependency chain:
    results/viterbi_decode.json       (Step 43.13: HMM decoded tokens)
    results/baum_welch_training.json  (Step 43.12: trained HMM params)
    results/hmm_architecture.json     (Step 43.10: state labels, obs vocab)
    results/signal_10k.json           (Phase 36.2: for comparison)
    results/null_corpus.json          (Phase 17: null seeds)
    results/modifier_integrate.json   (Phase 16: modifiers)
    data/corpus/                      (EVA transcription)
        → hmm_signal.json             (this step)
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
class HMMSignalResult:
    # Signal isolation
    n_tokens: int
    n_signal: int
    n_shared_hit: int
    n_shared_miss: int
    n_anti_signal: int
    signal_rate: float
    # Bigram analysis
    real_bigram_hits: int
    null_bigram_hits: List[int]
    null_bigram_mean: float
    null_bigram_std: float
    bigram_z_score: float
    # Comparison with Phase 15/36 signal
    overlap_with_10k_signal: float
    new_signal_tokens: int
    lost_signal_tokens: int
    # Held-out validation
    train_dict_hit: float
    test_dict_hit: float
    held_out_ratio: float
    # Signal word preservation
    bedrock_preserved: Dict[str, bool]
    n_bedrock_preserved: int
    # Validation battery
    validations: List[Dict]
    n_passed: int
    n_total: int
    # Approach 5 verdict
    approach5_verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Viterbi helper (lightweight re-import for null decoding)
# ---------------------------------------------------------------------------

def _viterbi_token(obs, log_pi, log_A, log_B):
    """Viterbi for a single observation sequence."""
    T = len(obs)
    K = len(log_pi)
    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=np.int32)

    delta[0] = log_pi + log_B[:, obs[0]]
    for t in range(1, T):
        for j in range(K):
            scores = delta[t - 1] + log_A[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] + log_B[j, obs[t]]

    states = np.zeros(T, dtype=np.int32)
    states[T - 1] = np.argmax(delta[T - 1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def _decode_tokens_hmm(
    tokens: List[str],
    state_labels: List[str],
    obs_vocab: List[str],
    log_pi, log_A, log_B,
) -> List[str]:
    """Decode a list of EVA tokens through the trained HMM."""
    char_to_idx = {ch: i for i, ch in enumerate(obs_vocab)}
    unk_idx = len(obs_vocab) - 1

    decoded = []
    for token in tokens:
        chars = tokenize_eva_chars(token)
        if not chars:
            decoded.append('')
            continue
        obs = np.array([char_to_idx.get(ch, unk_idx) for ch in chars], dtype=np.int32)
        states = _viterbi_token(obs, log_pi, log_A, log_B)
        word = ''.join(state_labels[s] for s in states)
        decoded.append(word)
    return decoded


def _build_ref_bigrams(ref_tokens):
    """Build reference bigram set."""
    return set(
        (ref_tokens[i].lower(), ref_tokens[i + 1].lower())
        for i in range(len(ref_tokens) - 1)
    )


def _count_bigrams(decoded, ref_bigrams):
    """Count bigram hits."""
    hits = 0
    for i in range(len(decoded) - 1):
        if (decoded[i].lower(), decoded[i + 1].lower()) in ref_bigrams:
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_hmm_signal() -> None:
    """Step 43.14: signal isolation on HMM-decoded corpus."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.14: HMM Signal Isolation and Validation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load HMM decode and trained params ──
    print("\n  1. Loading HMM results …")
    vit = _safe_load(os.path.join(rd, 'viterbi_decode.json'))
    bw = _safe_load(os.path.join(rd, 'baum_welch_training.json'))
    hmm_arch = _safe_load(os.path.join(rd, 'hmm_architecture.json'))

    state_labels = hmm_arch.get('state_labels', [])
    obs_vocab = hmm_arch.get('observation_vocab', [])

    pi = np.array(bw.get('pi', []))
    A = np.array(bw.get('A', []))
    B = np.array(bw.get('B', []))

    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_B = np.log(B + 1e-300)

    hmm_dict_hit = vit.get('dict_hit_rate', 0.0)
    print(f"     HMM dict-hit: {hmm_dict_hit:.1%}")

    # ── 2. Load dictionary ──
    print("\n  2. Building dictionary …")
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(ref.get_combined_tokens('latin'))
        expanded, _ = build_expanded_word_set(base_words)
        ref_tokens = ref.get_combined_tokens('latin')
    except Exception:
        expanded = set()
        ref_tokens = []
    ref_bigrams = _build_ref_bigrams(ref_tokens) if ref_tokens else set()
    print(f"     Dictionary: {len(expanded):,}, Bigrams: {len(ref_bigrams):,}")

    # ── 3. Decode real corpus via HMM ──
    print("\n  3. Decoding real corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens(paragraph_only=True)
    hmm_decoded = _decode_tokens_hmm(
        all_tokens, state_labels, obs_vocab, log_pi, log_A, log_B
    )
    n_tokens = len(hmm_decoded)
    real_hits = [w.lower() in expanded for w in hmm_decoded]
    print(f"     {n_tokens:,} tokens, {sum(real_hits):,} dict hits ({sum(real_hits)/n_tokens:.1%})")

    # ── 4. Decode null corpora ──
    print("\n  4. Decoding null corpora …")
    N_NULL = 5
    null_decoded_all: List[List[str]] = []
    for seed in range(N_NULL):
        rng = np.random.default_rng(100 + seed)
        shuffled = list(all_tokens)
        rng.shuffle(shuffled)
        null_dec = _decode_tokens_hmm(
            shuffled, state_labels, obs_vocab, log_pi, log_A, log_B
        )
        null_decoded_all.append(null_dec)
        nh = sum(1 for w in null_dec if w.lower() in expanded)
        print(f"     Null {seed}: {nh}/{len(null_dec)} ({nh/len(null_dec):.1%})")

    # ── 5. Signal isolation ──
    print("\n  5. Signal isolation …")
    null_hit_arrays = []
    for null_dec in null_decoded_all:
        null_hit_arrays.append([w.lower() in expanded for w in null_dec])

    n_signal = 0
    n_shared_hit = 0
    n_shared_miss = 0
    n_anti_signal = 0

    for i in range(n_tokens):
        r_hit = real_hits[i]
        null_count = sum(1 for nh in null_hit_arrays if i < len(nh) and nh[i])

        if r_hit and null_count == 0:
            n_signal += 1
        elif r_hit and null_count > 0:
            n_shared_hit += 1
        elif not r_hit and null_count == 0:
            n_shared_miss += 1
        else:
            n_anti_signal += 1

    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    print(f"     SIGNAL: {n_signal} ({signal_rate:.1%})")
    print(f"     SHARED_HIT: {n_shared_hit}")
    print(f"     ANTI_SIGNAL: {n_anti_signal}")

    # ── 6. Bigram z-score ──
    print("\n  6. Bigram z-score …")
    real_bg = _count_bigrams(hmm_decoded, ref_bigrams)
    null_bg = [_count_bigrams(nd, ref_bigrams) for nd in null_decoded_all]
    null_bg_mean = float(np.mean(null_bg))
    null_bg_std = float(np.std(null_bg)) if len(null_bg) > 1 else 1.0
    bigram_z = (real_bg - null_bg_mean) / null_bg_std if null_bg_std > 0 else 0.0
    print(f"     Real: {real_bg}, Null mean: {null_bg_mean:.1f}, z={bigram_z:.2f}")

    # ── 7. Compare with signal_10k ──
    print("\n  7. Comparison with Phase 36 signal …")
    sig_10k = _safe_load(os.path.join(rd, 'signal_10k.json'))
    classifications_10k = sig_10k.get('token_classifications', [])

    hmm_signal_set = set(i for i in range(n_tokens) if real_hits[i] and
                         all(not (i < len(nh) and nh[i]) for nh in null_hit_arrays))
    p36_signal_set = set(i for i, c in enumerate(classifications_10k) if c == 'SIGNAL')

    overlap = len(hmm_signal_set & p36_signal_set)
    overlap_rate = overlap / len(p36_signal_set) if p36_signal_set else 0.0
    new_signal = len(hmm_signal_set - p36_signal_set)
    lost_signal = len(p36_signal_set - hmm_signal_set)
    print(f"     Overlap with 10K signal: {overlap} ({overlap_rate:.1%})")
    print(f"     New signal: {new_signal}, Lost signal: {lost_signal}")

    # ── 8. Held-out validation ──
    print("\n  8. Held-out validation (odd/even folios) …")
    odd_hits = []
    even_hits = []
    folio_list = list(corpus.pages.keys())
    for i, token in enumerate(all_tokens):
        folio_idx = i % len(folio_list) if folio_list else 0
        if folio_idx % 2 == 0:
            even_hits.append(real_hits[i])
        else:
            odd_hits.append(real_hits[i])

    train_hit = sum(odd_hits) / len(odd_hits) if odd_hits else 0.0
    test_hit = sum(even_hits) / len(even_hits) if even_hits else 0.0
    held_out_ratio = test_hit / train_hit if train_hit > 0 else 0.0
    print(f"     Train (odd): {train_hit:.1%}, Test (even): {test_hit:.1%}")
    print(f"     Held-out ratio: {held_out_ratio:.3f}")

    # ── 9. Signal word preservation ──
    print("\n  9. Signal word preservation …")
    bedrock = ['bene', 'codi', 'sero', 'sene', 'de', 'raro', 'dine', 'cola']
    word_counts = Counter(w.lower() for w in hmm_decoded)
    bedrock_preserved = {w: word_counts.get(w, 0) >= 3 for w in bedrock}
    n_preserved = sum(bedrock_preserved.values())
    print(f"     Preserved: {n_preserved}/8")

    # ── 10. Validation battery ──
    print("\n  10. Validation battery …")
    validations = []

    v1 = hmm_dict_hit > 0.298
    validations.append({'id': 'V1', 'test': 'dict_hit > random baseline',
                       'value': round(hmm_dict_hit, 4), 'threshold': 0.298, 'passed': v1})

    v2 = bigram_z > 2.0
    validations.append({'id': 'V2', 'test': 'bigram_z > 2.0',
                       'value': round(bigram_z, 2), 'threshold': 2.0, 'passed': v2})

    v3 = signal_rate > 0.05
    validations.append({'id': 'V3', 'test': 'signal_rate > 5%',
                       'value': round(signal_rate, 4), 'threshold': 0.05, 'passed': v3})

    v4 = overlap_rate > 0.30
    validations.append({'id': 'V4', 'test': 'overlap with 10K signal > 30%',
                       'value': round(overlap_rate, 4), 'threshold': 0.30, 'passed': v4})

    v5 = held_out_ratio > 0.80
    validations.append({'id': 'V5', 'test': 'held-out ratio > 0.80',
                       'value': round(held_out_ratio, 4), 'threshold': 0.80, 'passed': v5})

    n_passed = sum(1 for v in validations if v['passed'])
    n_total = len(validations)

    for v in validations:
        status = "PASS" if v['passed'] else "FAIL"
        print(f"     {v['id']}: {v['test']} → {v['value']} [{status}]")

    # ── 11. Verdict ──
    if n_passed >= 4 and bigram_z > 3.0:
        verdict = "IMPROVEMENT"
    elif n_passed >= 3:
        verdict = "LATERAL"
    elif n_passed >= 1:
        verdict = "REGRESSION"
    else:
        verdict = "NO_SIGNAL"

    print(f"\n  Approach 5 verdict: {verdict}")

    # ── 12. Save ──
    elapsed = time.time() - t0

    result = HMMSignalResult(
        n_tokens=n_tokens,
        n_signal=n_signal,
        n_shared_hit=n_shared_hit,
        n_shared_miss=n_shared_miss,
        n_anti_signal=n_anti_signal,
        signal_rate=round(signal_rate, 6),
        real_bigram_hits=real_bg,
        null_bigram_hits=null_bg,
        null_bigram_mean=round(null_bg_mean, 2),
        null_bigram_std=round(null_bg_std, 2),
        bigram_z_score=round(bigram_z, 4),
        overlap_with_10k_signal=round(overlap_rate, 4),
        new_signal_tokens=new_signal,
        lost_signal_tokens=lost_signal,
        train_dict_hit=round(train_hit, 4),
        test_dict_hit=round(test_hit, 4),
        held_out_ratio=round(held_out_ratio, 4),
        bedrock_preserved=bedrock_preserved,
        n_bedrock_preserved=n_preserved,
        validations=validations,
        n_passed=n_passed,
        n_total=n_total,
        approach5_verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'hmm_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path} ({elapsed:.1f}s)")
