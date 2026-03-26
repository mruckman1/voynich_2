"""
Phase 62, Investigation 2: Cross-Token Word Reconstruction
===========================================================
Strip EVA token boundaries from the decoded syllable stream. Slide a
character window and check for dictionary words.  Compare cross-boundary
hit rate to null (shuffled token order).

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/phase62_cross_token.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrossTokenResult:
    phase: str = "62"
    step: str = "62.2"
    experiment: str = "cross_token"
    stream_length: int = 0
    n_tokens_used: int = 0
    # Sliding window results
    total_hits: int = 0
    cross_boundary_hits: int = 0
    within_token_hits: int = 0
    cross_boundary_fraction: float = 0.0
    mean_span: float = 0.0
    n_cross_5plus: int = 0            # cross-boundary words of length >= 5
    # Per-window-size breakdown
    per_window: List[Dict] = field(default_factory=list)
    # Null comparison
    null_cross_mean: float = 0.0
    null_cross_std: float = 0.0
    z_score: float = 0.0
    p_value: float = 1.0
    # Top words found
    top_words: List[List] = field(default_factory=list)
    top_cross_words: List[Dict] = field(default_factory=list)
    # Gates
    g1_cross_rate: bool = False        # cross-boundary > 20%
    g2_z_score: bool = False           # z > 2.0 vs shuffled null
    g3_mean_span: bool = False         # mean span 1.5-3.0
    g4_cross_5plus: bool = False       # >= 10 cross-boundary words len >= 5
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _build_stream(decoded_tokens):
    """Concatenate decoded tokens into a character stream with boundaries."""
    stream = []
    boundaries = []  # (stream_position, token_index)
    pos = 0
    for idx, d in enumerate(decoded_tokens):
        if not d or d == '?':
            continue
        boundaries.append((pos, idx))
        for ch in d:
            stream.append(ch)
            pos += 1
    return ''.join(stream), boundaries


def _find_token_at_pos(pos, boundaries):
    """Binary search for which token index a stream position belongs to."""
    lo, hi = 0, len(boundaries) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if boundaries[mid][0] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return boundaries[lo][1]


def _sliding_window_search(stream, boundaries, dictionary, min_len=4, max_len=10):
    """Slide window across stream, find dictionary words, classify boundary-crossing."""
    hits = []
    stream_len = len(stream)

    # For efficiency, use a set and check substrings
    for start in range(stream_len - min_len + 1):
        for length in range(min(max_len, stream_len - start), min_len - 1, -1):
            candidate = stream[start:start + length]
            if candidate in dictionary:
                start_tok = _find_token_at_pos(start, boundaries)
                end_tok = _find_token_at_pos(start + length - 1, boundaries)
                crosses = start_tok != end_tok
                hits.append({
                    'word': candidate,
                    'length': length,
                    'crosses_boundary': crosses,
                    'n_tokens_spanned': end_tok - start_tok + 1,
                })
                break  # longest match wins at this position

    return hits


def _analyze_hits(hits):
    total = len(hits)
    cross = sum(1 for h in hits if h['crosses_boundary'])
    within = total - cross
    cross_frac = cross / total if total > 0 else 0.0
    spans = [h['n_tokens_spanned'] for h in hits]
    mean_span = float(np.mean(spans)) if spans else 0.0
    cross_5plus = sum(1 for h in hits if h['crosses_boundary'] and h['length'] >= 5)

    word_counts = Counter(h['word'] for h in hits)
    cross_words = Counter(h['word'] for h in hits if h['crosses_boundary'])

    return {
        'total': total,
        'cross': cross,
        'within': within,
        'cross_frac': cross_frac,
        'mean_span': mean_span,
        'cross_5plus': cross_5plus,
        'top_words': word_counts.most_common(20),
        'top_cross': [{'word': w, 'count': c} for w, c in cross_words.most_common(15)],
    }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_cross_token():
    """Phase 62.2: Cross-token word reconstruction."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 2: Cross-Token Word Reconstruction")
    print("=" * 70)

    # Load
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    coda_table = build_coda_table_v2()

    # Load dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    # Filter to length >= 4 for sliding window
    dict_4plus = set(w for w in ref_word_set if len(w) >= 4)

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    decoded = decode_corpus_cvc_v2(all_tokens, assignment, eva_to_triple, coda_table)

    print(f"  Tokens: {len(all_tokens)}  Dict (len>=4): {len(dict_4plus)}")

    # Build stream and search
    stream, boundaries = _build_stream(decoded)
    print(f"  Stream length: {len(stream)} chars")
    print(f"  Running sliding window search (4-10 chars)...")

    hits = _sliding_window_search(stream, boundaries, dict_4plus, min_len=4, max_len=10)
    stats = _analyze_hits(hits)

    print(f"  Total hits: {stats['total']}")
    print(f"  Cross-boundary: {stats['cross']} ({stats['cross_frac']:.1%})")
    print(f"  Within-token: {stats['within']}")
    print(f"  Mean span: {stats['mean_span']:.2f}")
    print(f"  Cross-boundary len>=5: {stats['cross_5plus']}")

    # Null comparison: shuffle token order
    N_SHUFFLES = 100
    print(f"  Running {N_SHUFFLES} null shuffles...")
    null_cross_rates = []
    rng = np.random.default_rng(42)
    for seed in range(N_SHUFFLES):
        shuffled = list(decoded)
        rng.shuffle(shuffled)
        null_stream, null_bounds = _build_stream(shuffled)
        null_hits = _sliding_window_search(null_stream, null_bounds, dict_4plus,
                                            min_len=4, max_len=10)
        null_cross = sum(1 for h in null_hits if h['crosses_boundary'])
        null_total = len(null_hits)
        null_cross_rates.append(null_cross / null_total if null_total > 0 else 0.0)

    null_mean = float(np.mean(null_cross_rates))
    null_std = float(np.std(null_cross_rates))
    z = (stats['cross_frac'] - null_mean) / null_std if null_std > 0 else 0.0
    from scipy.stats import norm
    p_val = 1 - norm.cdf(z)

    print(f"  Null cross rate: {null_mean:.3f} ± {null_std:.3f}")
    print(f"  z-score: {z:.2f}  p={p_val:.4f}")

    # Gates
    g1 = stats['cross_frac'] > 0.20
    g2 = z > 2.0
    g3 = 1.5 <= stats['mean_span'] <= 3.0
    g4 = stats['cross_5plus'] >= 10
    gates_passed = sum([g1, g2, g3, g4])

    if g1 and g2:
        verdict = "WORDS_SPAN_TOKENS"
    elif g2:
        verdict = "WEAK_CROSS_BOUNDARY_SIGNAL"
    else:
        verdict = "TOKENS_ARE_SELF_CONTAINED"

    result = CrossTokenResult(
        stream_length=len(stream),
        n_tokens_used=len(boundaries),
        total_hits=stats['total'],
        cross_boundary_hits=stats['cross'],
        within_token_hits=stats['within'],
        cross_boundary_fraction=round(stats['cross_frac'], 4),
        mean_span=round(stats['mean_span'], 3),
        n_cross_5plus=stats['cross_5plus'],
        null_cross_mean=round(null_mean, 4),
        null_cross_std=round(null_std, 4),
        z_score=round(z, 3),
        p_value=round(p_val, 6),
        top_words=stats['top_words'],
        top_cross_words=stats['top_cross'],
        g1_cross_rate=g1,
        g2_z_score=g2,
        g3_mean_span=g3,
        g4_cross_5plus=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    if stats['top_cross']:
        print(f"\n  Top cross-boundary words:")
        for cw in stats['top_cross'][:10]:
            print(f"    {cw['word']:12s} ({cw['count']})")

    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} G4={'PASS' if g4 else 'FAIL'} ({gates_passed}/4)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_cross_token.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
