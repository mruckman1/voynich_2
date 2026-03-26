"""
Phase 69, Track 2: Clean LLM Reading (with Phase 66 Controls)
================================================================
Select the best consecutive clean runs for reading analysis.
Apply hallucination controls: compare real passages against shuffled
and null controls using dictionary-hit scoring.

Requires Track 0 >= PARTIAL.

Dependency chain:
    results/p69_clean_corpus.json        (Step 0)
    results/p69_clean_validation.json    (Track 0, must be >= PARTIAL)
    results/combined_refine.json         (Phase 15)
    results/triple_tiers.json            (Phase 28/53)
        -> results/p69_clean_llm.json
"""

import json
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
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51


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


def _get_confirmed_and_unresolved(rd: str) -> Tuple[Dict[str, str], Dict[str, str]]:
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
# Control generation
# ---------------------------------------------------------------------------

def _generate_shuffled_control(decoded_tokens: List[str], seed: int) -> List[str]:
    """Shuffle token order (preserves character distribution but destroys sequence)."""
    rng = np.random.default_rng(seed=seed)
    shuffled = list(decoded_tokens)
    rng.shuffle(shuffled)
    return shuffled


def _generate_null_control(decoded_tokens: List[str], char_dist: Dict[str, float],
                            seed: int) -> List[str]:
    """Generate null tokens matching clean-subset character distribution and lengths."""
    rng = np.random.default_rng(seed=seed)
    chars = sorted(char_dist.keys())
    probs = np.array([char_dist[c] for c in chars])
    probs = probs / probs.sum()

    null_tokens = []
    for d in decoded_tokens:
        length = len(d)
        if length == 0:
            null_tokens.append('')
            continue
        generated = ''.join(rng.choice(chars, size=length, p=probs))
        null_tokens.append(generated)
    return null_tokens


def _compute_char_distribution(decoded_tokens: List[str]) -> Dict[str, float]:
    """Compute character frequency distribution from decoded tokens."""
    char_counts: Counter = Counter()
    total = 0
    for d in decoded_tokens:
        for c in d:
            char_counts[c] += 1
            total += 1
    if total == 0:
        return {}
    return {c: count / total for c, count in char_counts.items()}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_passage(decoded_tokens: List[str], ref_word_set: Set[str],
                    signal_words: Set[str]) -> Dict[str, float]:
    """Score a passage by dict hit rate, signal word count, etc."""
    n = len(decoded_tokens)
    if n == 0:
        return {'dict_hit': 0.0, 'signal_hits': 0, 'n_tokens': 0}

    dict_hits = sum(1 for d in decoded_tokens if d and d in ref_word_set)
    signal_hits = sum(1 for d in decoded_tokens if d and d in signal_words)

    return {
        'dict_hit': dict_hits / n,
        'signal_hits': signal_hits,
        'n_tokens': n,
        'dict_hit_count': dict_hits,
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CleanLLMResult:
    phase: str = "69"
    step: str = "69.3"
    experiment: str = "clean_llm_reading"
    validation_status: str = ""
    n_passages: int = 0
    # Scores
    real_mean_dict_hit: float = 0.0
    shuffled_mean_dict_hit: float = 0.0
    null_mean_dict_hit: float = 0.0
    real_to_shuffled_ratio: float = 0.0
    real_to_null_ratio: float = 0.0
    # Signal
    real_mean_signal: float = 0.0
    # T1 preservation
    t1_preservation_rate: float = 0.0
    # Per-passage results
    passages: List[Dict[str, Any]] = field(default_factory=list)
    n_valid_readings: int = 0
    # Gates
    gate_cl1: bool = False    # dict-hit baseline >= 20%
    gate_cl2: bool = False    # real > 2x shuffled
    gate_cl3: bool = False    # real > 1.5x null
    gate_cl4: bool = False    # T1 preservation > 70%
    gate_cl5: bool = False    # >= 1 valid reading (dict_hit > 40%)
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_clean_llm_read():
    """Track 2: LLM reading of clean passages with controls."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 69.3 — Clean LLM Reading (with Controls)")
    print("=" * 49)

    # --- Check validation gate ---
    val_data = _safe_load(os.path.join(rd, 'p69_clean_validation.json'))
    val_verdict = val_data.get('verdict', 'FAILED')
    if val_verdict == 'FAILED':
        print("  SKIPPED: Validation gate = FAILED")
        result = CleanLLMResult(
            validation_status='SKIPPED_FAILED_VALIDATION',
            runtime_seconds=round(time.time() - t0, 1),
        )
        _save_json(rd, 'p69_clean_llm.json', result)
        return
    print(f"  Validation: {val_verdict}")

    # --- Load clean corpus ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    if not clean_data:
        print("  ERROR: p69_clean_corpus.json not found.")
        return

    t1_catalogue = clean_data.get('t1_catalogue', [])
    top_runs = clean_data.get('top_runs', [])
    clean_indices = clean_data.get('clean_indices', [])
    clean_decoded = clean_data.get('clean_decoded', [])

    t1_type_to_word = {e['eva_type']: e['matched_word'] for e in t1_catalogue}
    t1_types = set(t1_type_to_word.keys())

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
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    signal_words = set(SIGNAL_WORDS_51.keys())

    # --- Select passages from top runs ---
    print(f"\n  Selecting passages from {len(top_runs)} top runs...")

    passages = []
    for run_info in top_runs[:15]:
        start = run_info['start_idx']
        length = run_info['length']

        # Decode all tokens in run
        run_decoded = []
        run_t1_count = 0
        run_t1_preserved = 0

        for idx in range(start, start + length):
            token = all_tokens[idx]
            d = idx_to_decoded.get(idx)
            if d is None:
                result_tok = decode_token_cvc_v2(
                    token, full_assignment, eva_to_triple, coda_table)
                d = result_tok.decoded_cvc if result_tok.decoded_cvc else '?'

            run_decoded.append(d)
            if token in t1_types:
                run_t1_count += 1
                # Check if T1 word is preserved in decode
                expected = t1_type_to_word[token]
                if d == expected or (d and expected in d):
                    run_t1_preserved += 1

        passages.append({
            'start': start,
            'length': length,
            'folio': run_info.get('folio', '?'),
            'decoded': run_decoded,
            'n_t1': run_t1_count,
            't1_preserved': run_t1_preserved,
        })

    print(f"  Passages selected: {len(passages)}")

    # --- Compute character distribution for null controls ---
    all_clean_decoded = [d for d in clean_decoded if d and '?' not in d]
    char_dist = _compute_char_distribution(all_clean_decoded)

    # --- Score real, shuffled, and null controls ---
    print("\n  Scoring passages with controls...")

    passage_results = []
    all_real_hits = []
    all_shuffled_hits = []
    all_null_hits = []
    t1_total = 0
    t1_preserved_total = 0

    for p in passages:
        decoded = [d for d in p['decoded'] if d and '?' not in d]
        if len(decoded) < 5:
            continue

        # Real score
        real_score = _score_passage(decoded, ref_word_set, signal_words)

        # Shuffled control (3 trials, take mean)
        shuffled_scores = []
        for seed in range(3):
            shuffled = _generate_shuffled_control(decoded, seed=seed + 100)
            shuffled_scores.append(
                _score_passage(shuffled, ref_word_set, signal_words)['dict_hit'])
        mean_shuffled = float(np.mean(shuffled_scores))

        # Null control (3 trials, take mean)
        null_scores = []
        for seed in range(3):
            null_toks = _generate_null_control(decoded, char_dist, seed=seed + 200)
            null_scores.append(
                _score_passage(null_toks, ref_word_set, signal_words)['dict_hit'])
        mean_null = float(np.mean(null_scores))

        all_real_hits.append(real_score['dict_hit'])
        all_shuffled_hits.append(mean_shuffled)
        all_null_hits.append(mean_null)
        t1_total += p['n_t1']
        t1_preserved_total += p['t1_preserved']

        passage_results.append({
            'folio': p['folio'],
            'start': p['start'],
            'length': p['length'],
            'n_tokens': len(decoded),
            'real_dict_hit': round(real_score['dict_hit'], 3),
            'shuffled_dict_hit': round(mean_shuffled, 3),
            'null_dict_hit': round(mean_null, 3),
            'signal_hits': real_score['signal_hits'],
            'n_t1': p['n_t1'],
            't1_preserved': p['t1_preserved'],
            'sample': ' '.join(decoded[:20]),
        })

    # --- Aggregate ---
    real_mean = float(np.mean(all_real_hits)) if all_real_hits else 0.0
    shuffled_mean = float(np.mean(all_shuffled_hits)) if all_shuffled_hits else 0.0
    null_mean = float(np.mean(all_null_hits)) if all_null_hits else 0.0
    real_to_shuffled = real_mean / shuffled_mean if shuffled_mean > 0 else 0.0
    real_to_null = real_mean / null_mean if null_mean > 0 else 0.0
    t1_pres_rate = t1_preserved_total / t1_total if t1_total > 0 else 0.0
    real_signal_mean = float(np.mean([p['signal_hits'] for p in passage_results])) \
        if passage_results else 0.0
    n_valid = sum(1 for r in all_real_hits if r > 0.40)

    print(f"  Real mean dict hit:     {real_mean:.1%}")
    print(f"  Shuffled mean dict hit: {shuffled_mean:.1%}")
    print(f"  Null mean dict hit:     {null_mean:.1%}")
    print(f"  Real/shuffled ratio:    {real_to_shuffled:.2f}×")
    print(f"  Real/null ratio:        {real_to_null:.2f}×")
    print(f"  T1 preservation:        {t1_pres_rate:.1%}")

    # --- Gates ---
    gate_cl1 = real_mean >= 0.20
    gate_cl2 = real_to_shuffled > 2.0
    gate_cl3 = real_to_null > 1.5
    gate_cl4 = t1_pres_rate > 0.70
    gate_cl5 = n_valid >= 1
    gates_passed = sum([gate_cl1, gate_cl2, gate_cl3, gate_cl4, gate_cl5])

    result = CleanLLMResult(
        validation_status=val_verdict,
        n_passages=len(passage_results),
        real_mean_dict_hit=round(real_mean, 4),
        shuffled_mean_dict_hit=round(shuffled_mean, 4),
        null_mean_dict_hit=round(null_mean, 4),
        real_to_shuffled_ratio=round(real_to_shuffled, 3),
        real_to_null_ratio=round(real_to_null, 3),
        real_mean_signal=round(real_signal_mean, 2),
        t1_preservation_rate=round(t1_pres_rate, 3),
        passages=passage_results,
        n_valid_readings=n_valid,
        gate_cl1=gate_cl1,
        gate_cl2=gate_cl2,
        gate_cl3=gate_cl3,
        gate_cl4=gate_cl4,
        gate_cl5=gate_cl5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p69_clean_llm.json', result)

    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Dict hit:        {real_mean:.1%} ({'PASS' if gate_cl1 else 'FAIL'} >= 20%)")
    print(f"  Real/shuffled:   {real_to_shuffled:.2f}× ({'PASS' if gate_cl2 else 'FAIL'} > 2.0×)")
    print(f"  Real/null:       {real_to_null:.2f}× ({'PASS' if gate_cl3 else 'FAIL'} > 1.5×)")
    print(f"  T1 preserved:    {t1_pres_rate:.1%} ({'PASS' if gate_cl4 else 'FAIL'} > 70%)")
    print(f"  Valid readings:  {n_valid} ({'PASS' if gate_cl5 else 'FAIL'} >= 1)")
    print(f"  Gates: {gates_passed}/5")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
