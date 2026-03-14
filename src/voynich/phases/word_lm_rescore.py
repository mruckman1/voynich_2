"""
Phase 50 Track B – Word-Level LM Rescoring
============================================
Two-stage decode: (1) per-token ED1+charLM with length>=4 restriction,
(2) whole-sequence Viterbi with external word bigram LM.

Dependency chain:
    combined_refine.json        (Phase 15 best table)
    signal_bigrams.json         (Phase 29 parallel arrays)
        -> word_lm_rescore.json
"""

from __future__ import annotations

import json
import math
import os
import time
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import build_ngram_lm, cross_entropy_lm


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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_convert(item) for item in obj.tolist()]
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


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# ED1 generation (length-restricted)
# ---------------------------------------------------------------------------

def _generate_ed1(
    word: str,
    min_len: int = 4,
    vocab_chars: str = 'abcdefghijklmnopqrstuvwxyz',
) -> Set[str]:
    """Generate all edit-distance-1 variants of *word*.

    Only generate variants if ``len(word) >= min_len``; otherwise return
    ``{word}`` to avoid inflation on short decoded strings.
    """
    if len(word) < min_len:
        return {word}

    results: Set[str] = {word}
    for i in range(len(word)):
        # Deletion
        results.add(word[:i] + word[i + 1:])
        # Substitution
        for c in vocab_chars:
            if c != word[i]:
                results.add(word[:i] + c + word[i + 1:])
    # Insertion
    for i in range(len(word) + 1):
        for c in vocab_chars:
            results.add(word[:i] + c + word[i:])
    return results


# ---------------------------------------------------------------------------
# Token decode
# ---------------------------------------------------------------------------

def _decode_token(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> str:
    """Decode a single EVA token string to a syllable string."""
    chars = tokenize_eva_chars(token)
    parts: List[str] = []
    for ch in chars:
        triple_key = eva_to_triple.get(ch)
        if triple_key is None:
            continue
        syl = assignment.get(triple_key)
        if syl is not None:
            parts.append(syl)
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Char LM + 10K word set builders
# ---------------------------------------------------------------------------

def _build_char_lm() -> Dict:
    """Build a character 5-gram LM from Latin+Italian reference corpora."""
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_tokens.extend(ref.get_combined_tokens(lang))
    words = [w.lower() for w in all_tokens if w.isalpha() and 2 <= len(w) <= 15]
    return build_ngram_lm(words, order=5, smoothing=0.01)


def _build_10k_word_set() -> Set[str]:
    """Build a 10K word set from the most frequent Latin+Italian words."""
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_tokens.extend(ref.get_combined_tokens(lang))
    freq = Counter(w.lower() for w in all_tokens if w.isalpha() and len(w) >= 2)
    return {w for w, _ in freq.most_common(10000)}


# ---------------------------------------------------------------------------
# Word bigram LM
# ---------------------------------------------------------------------------

def _build_word_bigram_lm() -> Dict:
    """Build a word-level bigram LM from Latin+Italian reference text.

    Splits tokens into chunks of 50 (synthetic sentences) for bigram
    counting. Returns Counter-based storage for fast lookup.
    """
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    all_tokens: List[str] = []
    for lang in ('latin', 'italian'):
        all_tokens.extend(ref.get_combined_tokens(lang))
    words = [w.lower() for w in all_tokens if w.isalpha() and 2 <= len(w) <= 15]

    BOS = '<BOS>'
    bigram_counts: Counter = Counter()
    unigram_counts: Counter = Counter()

    # Split into "sentences" of 50 words
    for i in range(0, len(words), 50):
        chunk = [BOS] + words[i:i + 50]
        for w in chunk:
            unigram_counts[w] += 1
        for j in range(len(chunk) - 1):
            bigram_counts[(chunk[j], chunk[j + 1])] += 1

    total = sum(unigram_counts.values())
    vocab_size = len(unigram_counts)

    return {
        'bigram_counts': bigram_counts,
        'unigram_counts': unigram_counts,
        'total': total,
        'vocab_size': vocab_size,
    }


def _word_log_prob(word: str, prev_word: str, word_lm: Dict) -> float:
    """Log probability of *word* given *prev_word* (add-k smoothing)."""
    k = 1e-6
    V = word_lm['vocab_size']
    bi_count = word_lm['bigram_counts'].get((prev_word, word), 0)
    prev_count = word_lm['unigram_counts'].get(prev_word, 0)
    prob = (bi_count + k) / (prev_count + k * V)
    return math.log(prob + 1e-300)


# ---------------------------------------------------------------------------
# Stratified subsample
# ---------------------------------------------------------------------------

def _subsample_tokens(
    token_evas: List[str],
    token_folios: List[str],
    n: int = 5000,
    seed: int = 42,
) -> List[int]:
    """Stratified subsample of token indices, balanced across folios."""
    folio_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, f in enumerate(token_folios):
        folio_indices[f].append(idx)

    rng = random.Random(seed)
    n_folios = len(folio_indices)
    if n_folios == 0:
        return []

    per_folio = max(1, n // n_folios)
    selected: List[int] = []

    for folio in sorted(folio_indices.keys()):
        idxs = folio_indices[folio]
        if len(idxs) <= per_folio:
            selected.extend(idxs)
        else:
            selected.extend(rng.sample(idxs, per_folio))

    # If we overshot, trim; if under, top up randomly
    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    elif len(selected) < n:
        remaining = [i for i in range(len(token_evas)) if i not in set(selected)]
        deficit = n - len(selected)
        if deficit <= len(remaining):
            selected.extend(rng.sample(remaining, deficit))
        else:
            selected.extend(remaining)

    selected.sort()
    return selected


# ---------------------------------------------------------------------------
# Stage 1: per-token ED1 + charLM decode
# ---------------------------------------------------------------------------

def _stage1_decode(
    token_evas: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    char_lm: Dict,
    word_set: Set[str],
    ed1_min_len: int = 4,
) -> List[Tuple[List[Tuple[str, float, bool]], str, bool]]:
    """Per-token scoring with length-restricted ED1.

    Returns a list of (top5_candidates, best_word, is_hit) per token.
    Each candidate in top5 is (word, char_lm_score, is_hit).
    """
    results: List[Tuple[List[Tuple[str, float, bool]], str, bool]] = []

    for token in token_evas:
        base = _decode_token(token, assignment, eva_to_triple)
        if not base:
            results.append(([('', 99.0, False)], '', False))
            continue

        # Generate ED1 candidates with length restriction
        candidates = _generate_ed1(base, min_len=ed1_min_len)

        # Partition into dict hits and non-hits
        hit_candidates = [w for w in candidates if w in word_set]
        non_hit_candidates = [w for w in candidates if w not in word_set]

        # Score all with char LM
        def _score(w: str) -> float:
            if not w:
                return 99.0
            return cross_entropy_lm('_' + w + '_', char_lm, per_char=True)

        if hit_candidates:
            scored_hits = [(w, _score(w), True) for w in hit_candidates]
            scored_hits.sort(key=lambda x: x[1])
            best_word = scored_hits[0][0]
            is_hit = True
            # Top 5 from hits
            top5 = scored_hits[:5]
        else:
            scored_all = [(w, _score(w), w in word_set) for w in candidates]
            scored_all.sort(key=lambda x: x[1])
            best_word = scored_all[0][0]
            is_hit = False
            top5 = scored_all[:5]

        results.append((top5, best_word, is_hit))

    return results


# ---------------------------------------------------------------------------
# Stage 2: word-level Viterbi
# ---------------------------------------------------------------------------

def _stage2_viterbi(
    folio_indices: List[int],
    candidates_per_token: List[List[Tuple[str, float]]],
    word_lm: Dict,
    alpha: float = 0.7,
) -> List[str]:
    """Word-level Viterbi over a folio's token sequence.

    Parameters
    ----------
    folio_indices : list of int
        Indices into the global token array for this folio.
    candidates_per_token : list of list of (word, char_ce_score)
        Per token position, top-5 candidates from Stage 1.
    word_lm : dict
        Word bigram LM from :func:`_build_word_bigram_lm`.
    alpha : float
        Weight for the word bigram transition (1-alpha for char LM emission).

    Returns
    -------
    list of str
        Selected word for each position in the folio.
    """
    T = len(candidates_per_token)
    if T == 0:
        return []

    BOS = '<BOS>'

    # dp[t] = list of (score, backptr) for each candidate at position t
    dp: List[List[Tuple[float, int]]] = []

    # t = 0: initialise from BOS
    cands_0 = candidates_per_token[0]
    dp_0: List[Tuple[float, int]] = []
    for j, (wj, ce_j) in enumerate(cands_0):
        emission = (1.0 - alpha) * (-ce_j)
        transition = alpha * _word_log_prob(wj, BOS, word_lm)
        dp_0.append((emission + transition, -1))
    dp.append(dp_0)

    # t = 1..T-1
    for t in range(1, T):
        cands_t = candidates_per_token[t]
        dp_t: List[Tuple[float, int]] = []
        prev_cands = candidates_per_token[t - 1]

        for j, (wj, ce_j) in enumerate(cands_t):
            emission = (1.0 - alpha) * (-ce_j)
            best_score = -math.inf
            best_k = 0

            for k, (wk, _) in enumerate(prev_cands):
                transition = alpha * _word_log_prob(wj, wk, word_lm)
                score = dp[t - 1][k][0] + transition + emission
                if score > best_score:
                    best_score = score
                    best_k = k

            dp_t.append((best_score, best_k))
        dp.append(dp_t)

    # Backtrace
    # Find best final state
    best_final_score = -math.inf
    best_final_j = 0
    for j, (score, _) in enumerate(dp[T - 1]):
        if score > best_final_score:
            best_final_score = score
            best_final_j = j

    path: List[int] = [0] * T
    path[T - 1] = best_final_j
    for t in range(T - 2, -1, -1):
        path[t] = dp[t + 1][path[t + 1]][1]

    selected: List[str] = []
    for t in range(T):
        word, _ = candidates_per_token[t][path[t]]
        selected.append(word)

    return selected


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WordLMRescoreResult:
    phase16_baseline_rate: float
    stage1_rate: float
    stage1_improvement: float
    viterbi_rate: float
    viterbi_improvement: float
    words_changed_by_viterbi: int
    per_length: Dict[str, Dict]
    cc_bigrams_total: int
    cc_bigrams_ref_matches: int
    cc_bigrams_rate: float
    scrambled_mean: float
    scramble_selectivity: float
    bigram_real_matches: int
    bigram_z_score: float
    n_tokens: int
    alpha: float
    ed1_min_length: int
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_word_lm_rescore() -> None:
    """Phase 50 Track B: word-level LM rescoring."""
    t0 = time.time()
    print("=" * 70)
    print("PHASE 50 TRACK B: Word-Level LM Rescoring")
    print("=" * 70)

    rd = _results_dir()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n  [1/11] Loading data ...")
    cr = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment: Dict[str, str] = cr.get('best_assignment', cr.get('assignment', {}))
    if not assignment:
        print("  [ABORT] No assignment table found in combined_refine.json")
        return

    sb = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    token_evas: List[str] = sb.get('token_evas', [])
    token_folios: List[str] = sb.get('token_folios', [])
    token_decoded: List[str] = sb.get('token_decoded', [])
    n_total = len(token_evas)
    print(f"       Loaded {n_total} tokens, {len(assignment)} triple assignments")

    if n_total == 0:
        print("  [ABORT] No token data in signal_bigrams.json")
        return

    # ------------------------------------------------------------------
    # 2. Build infrastructure
    # ------------------------------------------------------------------
    print("\n  [2/11] Building infrastructure ...")
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"       EVA-to-triple lookup: {len(eva_to_triple)} glyphs")

    print("       Building char 5-gram LM ...")
    char_lm = _build_char_lm()

    print("       Building 10K word set ...")
    word_set = _build_10k_word_set()
    print(f"       10K word set: {len(word_set)} words")

    print("       Building word bigram LM ...")
    word_lm = _build_word_bigram_lm()
    print(f"       Word bigram LM: {word_lm['vocab_size']} vocab, "
          f"{sum(word_lm['bigram_counts'].values())} bigram events")

    # ------------------------------------------------------------------
    # 3. Subsample to 5000 tokens
    # ------------------------------------------------------------------
    print("\n  [3/11] Subsampling to 5000 tokens ...")
    sub_indices = _subsample_tokens(token_evas, token_folios, n=5000, seed=42)
    sub_evas = [token_evas[i] for i in sub_indices]
    sub_folios = [token_folios[i] for i in sub_indices]
    sub_decoded = [token_decoded[i] for i in sub_indices]
    n_sub = len(sub_evas)
    print(f"       Subsampled {n_sub} tokens across "
          f"{len(set(sub_folios))} folios")

    # ------------------------------------------------------------------
    # 4. Phase 16 baseline (raw decode, no ED1)
    # ------------------------------------------------------------------
    print("\n  [4/11] Computing Phase 16 baseline (raw decode) ...")
    baseline_hits = 0
    baseline_decoded: List[str] = []
    for eva in sub_evas:
        dec = _decode_token(eva, assignment, eva_to_triple)
        baseline_decoded.append(dec)
        if dec in word_set:
            baseline_hits += 1
    phase16_baseline_rate = baseline_hits / n_sub if n_sub > 0 else 0.0
    print(f"       Phase 16 baseline: {baseline_hits}/{n_sub} = "
          f"{phase16_baseline_rate:.4f}")

    # ------------------------------------------------------------------
    # 5. Stage 1: per-token ED1 + charLM (len>=4 restriction)
    # ------------------------------------------------------------------
    ED1_MIN_LEN = 4
    print(f"\n  [5/11] Stage 1: ED1 (min_len={ED1_MIN_LEN}) + charLM ...")
    stage1_results = _stage1_decode(
        sub_evas, assignment, eva_to_triple,
        char_lm, word_set, ed1_min_len=ED1_MIN_LEN,
    )

    stage1_hits = sum(1 for _, _, hit in stage1_results if hit)
    stage1_rate = stage1_hits / n_sub if n_sub > 0 else 0.0
    stage1_improvement = stage1_rate - phase16_baseline_rate
    print(f"       Stage 1: {stage1_hits}/{n_sub} = {stage1_rate:.4f} "
          f"(delta={stage1_improvement:+.4f})")

    # ------------------------------------------------------------------
    # 6. Group by folio and run Stage 2 Viterbi
    # ------------------------------------------------------------------
    ALPHA = 0.7
    print(f"\n  [6/11] Stage 2: Viterbi (alpha={ALPHA}) ...")

    # Group indices by folio (preserving order)
    folio_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, f in enumerate(sub_folios):
        folio_groups[f].append(idx)

    sorted_folios = sorted(folio_groups.keys())
    n_folios = len(sorted_folios)
    report_interval = max(1, n_folios // 5)

    viterbi_words: List[str] = [''] * n_sub

    for fi, folio in enumerate(sorted_folios):
        if fi % report_interval == 0:
            pct = fi / n_folios * 100
            print(f"       Folio {fi}/{n_folios} ({pct:.0f}%) ...")

        f_indices = folio_groups[folio]
        # Build candidates per token for this folio
        cands_for_folio: List[List[Tuple[str, float]]] = []
        for idx in f_indices:
            top5, _, _ = stage1_results[idx]
            cands_for_folio.append([(w, ce) for w, ce, _ in top5])

        selected = _stage2_viterbi(
            f_indices, cands_for_folio, word_lm, alpha=ALPHA,
        )

        for local_pos, global_idx in enumerate(f_indices):
            viterbi_words[global_idx] = selected[local_pos]

    print(f"       Folio {n_folios}/{n_folios} (100%) done.")

    viterbi_hits = sum(1 for w in viterbi_words if w in word_set)
    viterbi_rate = viterbi_hits / n_sub if n_sub > 0 else 0.0
    viterbi_improvement = viterbi_rate - phase16_baseline_rate

    words_changed = sum(
        1 for i in range(n_sub)
        if viterbi_words[i] != stage1_results[i][1]
    )

    print(f"       Viterbi: {viterbi_hits}/{n_sub} = {viterbi_rate:.4f} "
          f"(delta={viterbi_improvement:+.4f})")
    print(f"       Words changed by Viterbi: {words_changed}")

    # ------------------------------------------------------------------
    # 7. Compare baselines
    # ------------------------------------------------------------------
    print("\n  [7/11] Comparison summary ...")
    print(f"       Phase 16 raw:   {phase16_baseline_rate:.4f}")
    print(f"       Stage 1 ED1:    {stage1_rate:.4f} "
          f"({stage1_improvement:+.4f})")
    print(f"       Stage 2 Viterbi:{viterbi_rate:.4f} "
          f"({viterbi_improvement:+.4f})")

    # ------------------------------------------------------------------
    # 8. Per-length breakdown
    # ------------------------------------------------------------------
    print("\n  [8/11] Per-length breakdown ...")
    per_length: Dict[str, Dict] = {}
    len_groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n_sub):
        wlen = len(baseline_decoded[i])
        len_groups[wlen].append(i)

    for wlen in sorted(len_groups.keys()):
        idxs = len_groups[wlen]
        if not idxs:
            continue
        s1_h = sum(1 for idx in idxs if stage1_results[idx][2])
        vit_h = sum(1 for idx in idxs if viterbi_words[idx] in word_set)
        n_len = len(idxs)
        per_length[str(wlen)] = {
            'n_tokens': n_len,
            'stage1_hits': s1_h,
            'stage1_rate': s1_h / n_len if n_len > 0 else 0.0,
            'viterbi_hits': vit_h,
            'viterbi_rate': vit_h / n_len if n_len > 0 else 0.0,
        }
        if wlen <= 8:
            print(f"       len={wlen}: n={n_len}, "
                  f"S1={s1_h / n_len:.3f}, Vit={vit_h / n_len:.3f}")

    # ------------------------------------------------------------------
    # 9. CC Bigram analysis
    # ------------------------------------------------------------------
    print("\n  [9/11] CC Bigram analysis ...")
    # Build reference bigram set
    ref = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    ref_tokens_all: List[str] = []
    for lang in ('latin', 'italian'):
        ref_tokens_all.extend(ref.get_combined_tokens(lang))
    ref_words_lc = [w.lower() for w in ref_tokens_all if w.isalpha() and len(w) >= 2]
    ref_bigram_set: Set[Tuple[str, str]] = set()
    for i in range(len(ref_words_lc) - 1):
        ref_bigram_set.add((ref_words_lc[i], ref_words_lc[i + 1]))

    cc_total = 0
    cc_ref_matches = 0
    for i in range(n_sub - 1):
        # Same folio?
        if sub_folios[i] != sub_folios[i + 1]:
            continue
        w1 = viterbi_words[i]
        w2 = viterbi_words[i + 1]
        if len(w1) >= 3 and len(w2) >= 3 and w1 in word_set and w2 in word_set:
            cc_total += 1
            if (w1, w2) in ref_bigram_set:
                cc_ref_matches += 1

    cc_rate = cc_ref_matches / cc_total if cc_total > 0 else 0.0
    print(f"       CC bigrams: {cc_ref_matches}/{cc_total} = {cc_rate:.4f}")

    # ------------------------------------------------------------------
    # 10. Scrambled-corpus null (100 trials)
    # ------------------------------------------------------------------
    N_SCRAMBLE = 100
    print(f"\n  [10/11] Scrambled-corpus null ({N_SCRAMBLE} trials) ...")
    rng = random.Random(42)
    scrambled_rates: List[float] = []

    for trial in range(N_SCRAMBLE):
        # Shuffle token indices within each folio
        scrambled_viterbi_hits = 0
        for folio in sorted_folios:
            f_indices = folio_groups[folio]
            shuffled = list(f_indices)
            rng.shuffle(shuffled)

            # Build candidates using shuffled order
            cands_for_folio: List[List[Tuple[str, float]]] = []
            for idx in shuffled:
                top5, _, _ = stage1_results[idx]
                cands_for_folio.append([(w, ce) for w, ce, _ in top5])

            selected = _stage2_viterbi(
                shuffled, cands_for_folio, word_lm, alpha=ALPHA,
            )

            for w in selected:
                if w in word_set:
                    scrambled_viterbi_hits += 1

        s_rate = scrambled_viterbi_hits / n_sub if n_sub > 0 else 0.0
        scrambled_rates.append(s_rate)

    scrambled_mean = float(np.mean(scrambled_rates))
    scramble_selectivity = viterbi_rate / scrambled_mean if scrambled_mean > 0 else float('inf')
    print(f"       Scrambled mean: {scrambled_mean:.4f}")
    print(f"       Scramble selectivity: {scramble_selectivity:.3f}")

    # ------------------------------------------------------------------
    # 11. Bigram permutation test (1000 trials)
    # ------------------------------------------------------------------
    N_PERM = 1000
    print(f"\n  [11/11] Bigram permutation test ({N_PERM} trials) ...")

    # Real CC bigram matches (already computed above)
    real_cc_matches = cc_ref_matches

    perm_matches: List[int] = []
    rng2 = random.Random(123)

    for trial in range(N_PERM):
        perm_total_matches = 0
        for folio in sorted_folios:
            f_indices = folio_groups[folio]
            folio_words = [viterbi_words[idx] for idx in f_indices]
            # Permute word positions within this folio
            perm_words = list(folio_words)
            rng2.shuffle(perm_words)

            for i in range(len(perm_words) - 1):
                w1 = perm_words[i]
                w2 = perm_words[i + 1]
                if (len(w1) >= 3 and len(w2) >= 3
                        and w1 in word_set and w2 in word_set):
                    if (w1, w2) in ref_bigram_set:
                        perm_total_matches += 1

        perm_matches.append(perm_total_matches)

    perm_mean = float(np.mean(perm_matches))
    perm_std = float(np.std(perm_matches))
    bigram_z = ((real_cc_matches - perm_mean) / perm_std
                if perm_std > 0 else 0.0)
    print(f"       Real CC matches: {real_cc_matches}")
    print(f"       Null mean: {perm_mean:.2f}, std: {perm_std:.2f}")
    print(f"       z-score: {bigram_z:.2f}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    improvement = viterbi_rate - stage1_rate
    if improvement > 0.05 and scramble_selectivity > 1.1:
        verdict = 'WORD_LM_IMPROVES'
    elif improvement > 0.02:
        verdict = 'MARGINAL_IMPROVEMENT'
    elif improvement > -0.01:
        verdict = 'NO_IMPROVEMENT'
    else:
        verdict = 'REGRESSION'

    runtime = time.time() - t0

    print("\n" + "=" * 70)
    print(f"  VERDICT: {verdict}")
    print(f"  Phase 16 baseline: {phase16_baseline_rate:.4f}")
    print(f"  Stage 1 (ED1 len>={ED1_MIN_LEN}): {stage1_rate:.4f}")
    print(f"  Stage 2 (Viterbi alpha={ALPHA}): {viterbi_rate:.4f}")
    print(f"  Viterbi improvement over Stage 1: {improvement:+.4f}")
    print(f"  Scramble selectivity: {scramble_selectivity:.3f}")
    print(f"  Bigram z-score: {bigram_z:.2f}")
    print(f"  Runtime: {runtime:.1f}s")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------------
    result = WordLMRescoreResult(
        phase16_baseline_rate=phase16_baseline_rate,
        stage1_rate=stage1_rate,
        stage1_improvement=stage1_improvement,
        viterbi_rate=viterbi_rate,
        viterbi_improvement=viterbi_improvement,
        words_changed_by_viterbi=words_changed,
        per_length=per_length,
        cc_bigrams_total=cc_total,
        cc_bigrams_ref_matches=cc_ref_matches,
        cc_bigrams_rate=cc_rate,
        scrambled_mean=scrambled_mean,
        scramble_selectivity=scramble_selectivity,
        bigram_real_matches=real_cc_matches,
        bigram_z_score=bigram_z,
        n_tokens=n_sub,
        alpha=ALPHA,
        ed1_min_length=ED1_MIN_LEN,
        verdict=verdict,
        runtime_seconds=runtime,
    )

    _save_json(rd, 'word_lm_rescore.json', asdict(result))
    print(f"\n  Saved -> {os.path.join(rd, 'word_lm_rescore.json')}")
