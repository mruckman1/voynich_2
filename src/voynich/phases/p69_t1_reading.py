"""
Phase 69, Track 5: T1-Anchored Passage Reading
=================================================
Find passages with dense T1 word clusters, use them as fixed anchors,
and attempt to fill gaps with LM-based segmentation.

Dependency chain:
    results/p69_clean_corpus.json        (Step 0)
    results/combined_refine.json         (Phase 15)
    results/triple_tiers.json            (Phase 28/53)
        -> results/p69_t1_reading.json
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

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
from voynich.phases.cvc_permutation import PHARMA_REGISTER


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
    """Return (confirmed_12, unresolved_13)."""
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
# Simple Viterbi word segmentation
# ---------------------------------------------------------------------------

def _build_word_unigram_model(ref_word_set: Set[str],
                               ref_corpus_tokens: List[str]) -> Dict[str, float]:
    """Build unigram log-probability model from reference corpus."""
    word_counts: Counter = Counter()
    for token in ref_corpus_tokens:
        w = token.lower()
        if len(w) >= 2 and w in ref_word_set:
            word_counts[w] += 1

    total = sum(word_counts.values())
    if total == 0:
        return {}

    log_probs = {}
    for word, count in word_counts.items():
        log_probs[word] = math.log(count / total)

    return log_probs


def _viterbi_segment(stream: str, word_log_probs: Dict[str, float],
                      min_len: int = 2, max_len: int = 10) -> List[str]:
    """Simple Viterbi segmentation of a character stream into words."""
    n = len(stream)
    if n == 0:
        return []

    # Unknown word penalty
    unk_penalty = -20.0

    # DP: best[i] = (best_log_prob to reach position i, backpointer)
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

    # Backtrack
    if best[n][0] == float('-inf'):
        return [stream]  # fallback: whole stream as one "word"

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
class T1ReadingResult:
    phase: str = "69"
    step: str = "69.6"
    experiment: str = "t1_anchored_reading"
    n_windows_found: int = 0
    n_passages_attempted: int = 0
    # Per-passage results
    passages: List[Dict[str, Any]] = field(default_factory=list)
    mean_known_fraction: float = 0.0
    mean_dict_hit: float = 0.0
    n_passages_above_30pct: int = 0
    n_pharma_readings: int = 0
    # Gates
    gate_ar1: bool = False    # >= 20 passages found
    gate_ar2: bool = False    # mean known_fraction > 0.50
    gate_ar3: bool = False    # >= 3 passages dict_hit > 30%
    gate_ar4: bool = False    # >= 1 pharma reading
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_t1_read():
    """Track 5: T1-anchored passage reading."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 69.6 — T1-Anchored Passage Reading")
    print("=" * 43)

    # --- Load T1 catalogue ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    if not clean_data:
        print("  ERROR: p69_clean_corpus.json not found.")
        return

    t1_catalogue = clean_data.get('t1_catalogue', [])
    clean_indices_set = set(clean_data.get('clean_indices', []))
    print(f"  T1 words: {len(t1_catalogue)}")

    # Build T1 type → word lookup
    t1_type_to_word: Dict[str, str] = {}
    t1_type_to_gloss: Dict[str, str] = {}
    for entry in t1_catalogue:
        t1_type_to_word[entry['eva_type']] = entry['matched_word']
        t1_type_to_gloss[entry['eva_type']] = entry.get('matched_word', '')
    t1_types = set(t1_type_to_word.keys())

    # --- Load corpus and assignment ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    full_assignment = {**confirmed, **unresolved}
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Build folio list
    folio_list: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folio_list.append(folio)

    # Build dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    base_words = set(w.lower() for w in ref_tokens if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Build unigram model for Viterbi
    word_log_probs = _build_word_unigram_model(ref_word_set, ref_tokens)
    print(f"  Words with log-probs: {len(word_log_probs)}")

    # --- Find T1-dense windows ---
    print("\n  Scanning for T1-dense windows (size=15, min 3 T1)...")
    WINDOW_SIZE = 15
    MIN_T1 = 3

    windows = []
    for start in range(len(all_tokens) - WINDOW_SIZE + 1):
        t1_count = sum(1 for i in range(start, start + WINDOW_SIZE)
                      if all_tokens[i] in t1_types)
        if t1_count >= MIN_T1:
            windows.append({
                'start': start,
                'end': start + WINDOW_SIZE - 1,
                'n_t1': t1_count,
                'folio': folio_list[start] if start < len(folio_list) else '?',
            })

    # Deduplicate overlapping windows (keep highest T1 count)
    deduped: List[Dict] = []
    used_starts: Set[int] = set()
    for w in sorted(windows, key=lambda w: -w['n_t1']):
        if any(abs(w['start'] - s) < WINDOW_SIZE // 2 for s in used_starts):
            continue
        deduped.append(w)
        used_starts.add(w['start'])

    deduped.sort(key=lambda w: -w['n_t1'])
    n_windows = len(deduped)
    print(f"  Windows found: {n_windows}")

    # --- Build and analyze passages ---
    print("\n  Analyzing passages...")
    passages = []

    for window in deduped[:30]:
        tokens_in_window = []
        for idx in range(window['start'], window['end'] + 1):
            token = all_tokens[idx]
            result = decode_token_cvc_v2(
                token, full_assignment, eva_to_triple, coda_table)
            decoded = result.decoded_cvc if result.decoded_cvc else '?'

            is_t1 = token in t1_types
            is_clean = idx in clean_indices_set

            tokens_in_window.append({
                'position': idx,
                'eva': token,
                'decoded': decoded,
                'is_t1': is_t1,
                't1_word': t1_type_to_word.get(token) if is_t1 else None,
                'is_clean': is_clean,
                'confidence': 'T1' if is_t1 else ('CLEAN' if is_clean else 'PARTIAL'),
            })

        n_t1 = sum(1 for t in tokens_in_window if t['is_t1'])
        n_clean = sum(1 for t in tokens_in_window if t['is_clean'])
        n_total = len(tokens_in_window)
        known_fraction = (n_t1 + n_clean) / n_total if n_total else 0.0

        # Attempt gap filling: concatenate non-T1 decoded chars, segment
        gap_decoded = ''.join(t['decoded'] for t in tokens_in_window
                             if not t['is_t1'] and t['decoded'] != '?')
        gap_words = _viterbi_segment(gap_decoded, word_log_probs) if gap_decoded else []
        gap_dict_hits = sum(1 for w in gap_words if w in ref_word_set)
        gap_dict_hit_rate = gap_dict_hits / len(gap_words) if gap_words else 0.0

        # Overall dict hit (T1 words count as hits + gap hits)
        total_words = n_t1 + len(gap_words)
        total_hits = n_t1 + gap_dict_hits  # T1 words are known hits
        overall_dict_hit = total_hits / total_words if total_words else 0.0

        # Check pharmaceutical content
        all_words_in_passage = set()
        for t in tokens_in_window:
            if t['is_t1'] and t['t1_word']:
                all_words_in_passage.add(t['t1_word'])
        all_words_in_passage.update(gap_words)
        has_pharma = len(all_words_in_passage & PHARMA_REGISTER) >= 1

        # Build readable representation
        readable_parts = []
        for t in tokens_in_window:
            if t['is_t1']:
                readable_parts.append(f"[{t['t1_word']}]")
            elif t['is_clean']:
                readable_parts.append(t['decoded'])
            else:
                readable_parts.append(f"({t['decoded']})")

        passages.append({
            'folio': window['folio'],
            'start_idx': window['start'],
            'n_tokens': n_total,
            'n_t1': n_t1,
            'n_clean': n_clean,
            'known_fraction': round(known_fraction, 3),
            'readable': ' '.join(readable_parts),
            't1_words': [t['t1_word'] for t in tokens_in_window if t['is_t1']],
            'gap_words': gap_words[:20],
            'gap_dict_hit_rate': round(gap_dict_hit_rate, 3),
            'overall_dict_hit': round(overall_dict_hit, 3),
            'has_pharma': has_pharma,
        })

    # --- Aggregate metrics ---
    known_fractions = [p['known_fraction'] for p in passages]
    dict_hits = [p['overall_dict_hit'] for p in passages]
    mean_known = sum(known_fractions) / len(known_fractions) if known_fractions else 0.0
    mean_dict_hit = sum(dict_hits) / len(dict_hits) if dict_hits else 0.0
    n_above_30 = sum(1 for d in dict_hits if d > 0.30)
    n_pharma = sum(1 for p in passages if p['has_pharma'])

    # --- Gates ---
    gate_ar1 = n_windows >= 20
    gate_ar2 = mean_known > 0.50
    gate_ar3 = n_above_30 >= 3
    gate_ar4 = n_pharma >= 1
    gates_passed = sum([gate_ar1, gate_ar2, gate_ar3, gate_ar4])

    result = T1ReadingResult(
        n_windows_found=n_windows,
        n_passages_attempted=len(passages),
        passages=passages,
        mean_known_fraction=round(mean_known, 3),
        mean_dict_hit=round(mean_dict_hit, 3),
        n_passages_above_30pct=n_above_30,
        n_pharma_readings=n_pharma,
        gate_ar1=gate_ar1,
        gate_ar2=gate_ar2,
        gate_ar3=gate_ar3,
        gate_ar4=gate_ar4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p69_t1_reading.json', result)

    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Windows:        {n_windows} ({'PASS' if gate_ar1 else 'FAIL'} >= 20)")
    print(f"  Known fraction: {mean_known:.1%} ({'PASS' if gate_ar2 else 'FAIL'} > 50%)")
    print(f"  Passages >30%:  {n_above_30} ({'PASS' if gate_ar3 else 'FAIL'} >= 3)")
    print(f"  Pharma:         {n_pharma} ({'PASS' if gate_ar4 else 'FAIL'} >= 1)")
    print(f"  Gates: {gates_passed}/4")

    # Show top 5 passages
    if passages:
        print(f"\n  Top passages:")
        for p in sorted(passages, key=lambda p: -p['overall_dict_hit'])[:5]:
            print(f"    {p['folio']} (T1={p['n_t1']}, hit={p['overall_dict_hit']:.0%}): "
                  f"{p['readable'][:80]}")

    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
