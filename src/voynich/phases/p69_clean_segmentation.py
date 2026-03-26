"""
Phase 69, Track 1: Clean Subset Segmentation
===============================================
Re-run Harris MI and LM Viterbi segmentation on the decoded character
streams from clean runs only.  Phase 65 achieved F1 > 0.65 on Latin but
near-zero on Voynich due to 56% decode error.  The clean subset has 0%.

Requires Track 0 >= PARTIAL.

Dependency chain:
    results/p69_clean_corpus.json        (Step 0)
    results/p69_clean_validation.json    (Track 0, must be >= PARTIAL)
    results/combined_refine.json         (Phase 15)
    results/triple_tiers.json            (Phase 28/53)
        -> results/p69_clean_segmentation.json
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
)


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
    if isinstance(obj, set):
        return sorted(obj)
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
# Confirmed triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()
    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))
    if not confirmed_keys:
        return dict(assignment), {}
    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Harris MI boundary detection
# ---------------------------------------------------------------------------

def _compute_mi_profile(stream: str, context_len: int = 2) -> List[float]:
    """Compute MI between context and next char at each position."""
    if len(stream) <= context_len:
        return []

    # Build joint and marginal counts
    joint_counts: Counter = Counter()
    context_counts: Counter = Counter()
    char_counts: Counter = Counter()

    for i in range(context_len, len(stream)):
        ctx = stream[i - context_len:i]
        ch = stream[i]
        joint_counts[(ctx, ch)] += 1
        context_counts[ctx] += 1
        char_counts[ch] += 1

    total = sum(joint_counts.values())
    if total == 0:
        return [0.0] * (len(stream) - context_len)

    # Compute MI at each position
    mi_values = []
    for i in range(context_len, len(stream)):
        ctx = stream[i - context_len:i]
        ch = stream[i]
        p_joint = joint_counts.get((ctx, ch), 0) / total
        p_ctx = context_counts.get(ctx, 0) / total
        p_ch = char_counts.get(ch, 0) / total

        if p_joint > 0 and p_ctx > 0 and p_ch > 0:
            mi = math.log2(p_joint / (p_ctx * p_ch))
        else:
            mi = 0.0
        mi_values.append(mi)

    return mi_values


def _find_mi_boundaries(mi_values: List[float], sigma: float = 1.5,
                         min_depth: float = 0.3) -> List[int]:
    """Find local minima in MI profile as word boundary candidates."""
    if len(mi_values) < 3:
        return []

    mean_mi = np.mean(mi_values)
    std_mi = np.std(mi_values)
    threshold = mean_mi - sigma * std_mi

    boundaries = []
    for i in range(1, len(mi_values) - 1):
        if (mi_values[i] < mi_values[i - 1] and
                mi_values[i] < mi_values[i + 1] and
                mi_values[i] < threshold):
            depth = min(mi_values[i - 1] - mi_values[i],
                       mi_values[i + 1] - mi_values[i])
            if depth >= min_depth:
                boundaries.append(i)

    return boundaries


def _segment_at_boundaries(stream: str, boundaries: List[int],
                            context_len: int = 2) -> List[str]:
    """Segment a string at given boundary positions."""
    # Adjust boundaries to account for context offset
    adjusted = [b + context_len for b in boundaries]
    # Add start and end
    cuts = [0] + sorted(adjusted) + [len(stream)]
    words = []
    for i in range(len(cuts) - 1):
        w = stream[cuts[i]:cuts[i + 1]]
        if w:
            words.append(w)
    return words


# ---------------------------------------------------------------------------
# Viterbi LM segmentation
# ---------------------------------------------------------------------------

def _build_word_unigram_model(ref_word_set: Set[str],
                               ref_tokens: List[str]) -> Dict[str, float]:
    """Build unigram log-probability model from reference corpus."""
    word_counts: Counter = Counter()
    for token in ref_tokens:
        w = token.lower()
        if len(w) >= 2 and w in ref_word_set:
            word_counts[w] += 1
    total = sum(word_counts.values())
    if total == 0:
        return {}
    return {w: math.log(c / total) for w, c in word_counts.items()}


def _viterbi_segment(stream: str, word_log_probs: Dict[str, float],
                      min_len: int = 2, max_len: int = 10) -> List[str]:
    """Viterbi segmentation minimizing -log P(word sequence)."""
    n = len(stream)
    if n == 0:
        return []

    unk_penalty = -20.0
    best = [(float('-inf'), -1)] * (n + 1)
    best[0] = (0.0, -1)

    for i in range(n):
        if best[i][0] == float('-inf'):
            continue
        for length in range(min_len, min(max_len, n - i) + 1):
            word = stream[i:i + length]
            lp = word_log_probs.get(word, unk_penalty)
            score = best[i][0] + lp
            if score > best[i + length][0]:
                best[i + length] = (score, i)

    if best[n][0] == float('-inf'):
        return [stream]

    words = []
    pos = n
    while pos > 0:
        prev = best[pos][1]
        if prev < 0:
            break
        words.append(stream[prev:pos])
        pos = prev
    words.reverse()
    return words


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CleanSegmentResult:
    phase: str = "69"
    step: str = "69.2"
    experiment: str = "clean_segmentation"
    validation_status: str = ""
    n_runs_processed: int = 0
    # Harris MI results
    harris_results: List[Dict[str, Any]] = field(default_factory=list)
    harris_mean_dict_hit: float = 0.0
    harris_mean_word_length: float = 0.0
    # LM Viterbi results
    lm_results: List[Dict[str, Any]] = field(default_factory=list)
    lm_mean_dict_hit: float = 0.0
    lm_mean_word_length: float = 0.0
    # EVA baseline
    eva_baseline_dict_hit: float = 0.0
    # Gates
    gate_cs1: bool = False    # LM dict_hit > 15%
    gate_cs2: bool = False    # beats EVA baseline (15.4%)
    gate_cs3: bool = False    # mean word length 4.0-8.0
    gate_cs4: bool = False    # >= 5 runs with > 20% dict_hit
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_clean_segment():
    """Track 1: Harris MI + LM segmentation on clean decoded runs."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 69.2 — Clean Subset Segmentation")
    print("=" * 40)

    # --- Check validation gate ---
    val_data = _safe_load(os.path.join(rd, 'p69_clean_validation.json'))
    val_verdict = val_data.get('verdict', 'FAILED')
    if val_verdict == 'FAILED':
        print("  SKIPPED: Validation gate = FAILED")
        result = CleanSegmentResult(
            validation_status='SKIPPED_FAILED_VALIDATION',
            runtime_seconds=round(time.time() - t0, 1),
        )
        _save_json(rd, 'p69_clean_segmentation.json', result)
        return
    print(f"  Validation: {val_verdict}")

    # --- Load clean corpus ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    if not clean_data:
        print("  ERROR: p69_clean_corpus.json not found.")
        return

    runs = clean_data.get('top_runs', [])
    clean_indices = clean_data.get('clean_indices', [])
    clean_decoded = clean_data.get('clean_decoded', [])
    print(f"  Clean runs >= 5: {len(runs)}")

    # Build index → decoded lookup
    idx_to_decoded: Dict[int, str] = {}
    for i, ci in enumerate(clean_indices):
        if i < len(clean_decoded):
            idx_to_decoded[ci] = clean_decoded[i]

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    full_assignment = {**confirmed, **unresolved}
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    base_words = set(w.lower() for w in ref_tokens if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Build LM
    word_log_probs = _build_word_unigram_model(ref_word_set, ref_tokens)
    print(f"  LM vocabulary: {len(word_log_probs)} words")

    # --- Process runs ---
    # Re-build run streams from the stored run data
    processed_runs = []
    for run_info in runs[:50]:
        start = run_info['start_idx']
        length = run_info['length']

        # Build decoded stream from clean tokens in this run
        stream_chars = []
        eva_decoded_tokens = []
        for idx in range(start, start + length):
            d = idx_to_decoded.get(idx)
            if d is None:
                # Decode on the fly
                result = decode_token_cvc_v2(
                    all_tokens[idx], full_assignment, eva_to_triple, coda_table)
                d = result.decoded_cvc if result.decoded_cvc else ''
            if d and '?' not in d:
                stream_chars.append(d)
                eva_decoded_tokens.append(d)

        if not stream_chars:
            continue

        stream = ''.join(stream_chars)
        if len(stream) < 10:
            continue

        processed_runs.append({
            'start': start,
            'length': length,
            'stream': stream,
            'eva_decoded_tokens': eva_decoded_tokens,
            'folio': run_info.get('folio', '?'),
        })

    print(f"  Processable runs: {len(processed_runs)}")

    # --- Harris MI segmentation ---
    print("\n  Harris MI segmentation...")
    harris_results = []
    for run in processed_runs:
        mi_profile = _compute_mi_profile(run['stream'], context_len=2)
        boundaries = _find_mi_boundaries(mi_profile, sigma=1.5, min_depth=0.2)
        words = _segment_at_boundaries(run['stream'], boundaries, context_len=2)

        dict_hits = sum(1 for w in words if w in ref_word_set)
        dict_hit_rate = dict_hits / len(words) if words else 0.0
        mean_len = np.mean([len(w) for w in words]) if words else 0.0

        harris_results.append({
            'start': run['start'],
            'folio': run['folio'],
            'length': run['length'],
            'n_boundaries': len(boundaries),
            'n_words': len(words),
            'mean_word_length': round(float(mean_len), 1),
            'dict_hit_rate': round(dict_hit_rate, 3),
            'sample': ' '.join(words[:15]),
        })

    harris_rates = [r['dict_hit_rate'] for r in harris_results]
    harris_mean_hit = float(np.mean(harris_rates)) if harris_rates else 0.0
    harris_lengths = [r['mean_word_length'] for r in harris_results]
    harris_mean_len = float(np.mean(harris_lengths)) if harris_lengths else 0.0
    print(f"  Harris mean dict hit: {harris_mean_hit:.1%}")
    print(f"  Harris mean word length: {harris_mean_len:.1f}")

    # --- LM Viterbi segmentation ---
    print("\n  LM Viterbi segmentation...")
    lm_results = []
    for run in processed_runs:
        words = _viterbi_segment(run['stream'], word_log_probs)

        dict_hits = sum(1 for w in words if w in ref_word_set)
        dict_hit_rate = dict_hits / len(words) if words else 0.0
        mean_len = np.mean([len(w) for w in words]) if words else 0.0

        lm_results.append({
            'start': run['start'],
            'folio': run['folio'],
            'length': run['length'],
            'n_words': len(words),
            'mean_word_length': round(float(mean_len), 1),
            'dict_hit_rate': round(dict_hit_rate, 3),
            'sample': ' '.join(words[:15]),
        })

    lm_rates = [r['dict_hit_rate'] for r in lm_results]
    lm_mean_hit = float(np.mean(lm_rates)) if lm_rates else 0.0
    lm_lengths = [r['mean_word_length'] for r in lm_results]
    lm_mean_len = float(np.mean(lm_lengths)) if lm_lengths else 0.0
    print(f"  LM mean dict hit: {lm_mean_hit:.1%}")
    print(f"  LM mean word length: {lm_mean_len:.1f}")

    # --- EVA baseline ---
    print("\n  EVA baseline (token boundaries as words)...")
    eva_dict_hits = 0
    eva_total = 0
    for run in processed_runs:
        for d in run['eva_decoded_tokens']:
            eva_total += 1
            if d in ref_word_set:
                eva_dict_hits += 1
    eva_baseline = eva_dict_hits / eva_total if eva_total else 0.0
    print(f"  EVA baseline dict hit: {eva_baseline:.1%}")

    # --- Gates ---
    n_runs_above_20 = sum(1 for r in lm_results if r['dict_hit_rate'] > 0.20)

    gate_cs1 = lm_mean_hit > 0.15
    gate_cs2 = lm_mean_hit > eva_baseline
    gate_cs3 = 4.0 <= lm_mean_len <= 8.0
    gate_cs4 = n_runs_above_20 >= 5
    gates_passed = sum([gate_cs1, gate_cs2, gate_cs3, gate_cs4])

    result = CleanSegmentResult(
        validation_status=val_verdict,
        n_runs_processed=len(processed_runs),
        harris_results=harris_results[:30],
        harris_mean_dict_hit=round(harris_mean_hit, 4),
        harris_mean_word_length=round(harris_mean_len, 2),
        lm_results=lm_results[:30],
        lm_mean_dict_hit=round(lm_mean_hit, 4),
        lm_mean_word_length=round(lm_mean_len, 2),
        eva_baseline_dict_hit=round(eva_baseline, 4),
        gate_cs1=gate_cs1,
        gate_cs2=gate_cs2,
        gate_cs3=gate_cs3,
        gate_cs4=gate_cs4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p69_clean_segmentation.json', result)

    print(f"\n  Summary")
    print(f"  -------")
    print(f"  LM dict hit:     {lm_mean_hit:.1%} ({'PASS' if gate_cs1 else 'FAIL'} > 15%)")
    print(f"  Beats EVA:       {lm_mean_hit:.1%} vs {eva_baseline:.1%} "
          f"({'PASS' if gate_cs2 else 'FAIL'})")
    print(f"  Word length:     {lm_mean_len:.1f} ({'PASS' if gate_cs3 else 'FAIL'} 4.0-8.0)")
    print(f"  Runs > 20%:      {n_runs_above_20} ({'PASS' if gate_cs4 else 'FAIL'} >= 5)")
    print(f"  Gates: {gates_passed}/4")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
