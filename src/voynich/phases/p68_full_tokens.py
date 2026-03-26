"""
Phase 68, Track 1: Fully-Decoded Token Exploitation
=====================================================
Identify tokens where 100% of EVA characters map to confirmed triples
or validated coda markers.  These tokens have 0% decode error and
provide gold-standard anchors for constraining unresolved triples
via their neighbors.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
        -> results/p68_full_tokens.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import CodaTable, get_coda
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
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
# Confirmed / unresolved triple separation
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FullTokenResult:
    phase: str = "68"
    step: str = "68.1"
    experiment: str = "fully_decoded_tokens"
    # Corpus stats
    n_corpus_tokens: int = 0
    n_fully_decoded: int = 0
    fully_decoded_fraction: float = 0.0
    # Dict hit on fully-decoded subset
    n_dict_hits: int = 0
    dict_hit_rate: float = 0.0
    # Vocabulary
    n_distinct_words: int = 0
    top_words: List[Dict[str, Any]] = field(default_factory=list)
    # Consecutive runs
    n_runs_ge3: int = 0
    longest_run: int = 0
    readable_sequences: List[Dict[str, Any]] = field(default_factory=list)
    # Anchor context analysis
    n_partial_with_anchors: int = 0
    n_triples_with_context: int = 0
    # Triple candidates from context scoring
    triple_candidates: Dict[str, str] = field(default_factory=dict)
    triple_details: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_n_fully: bool = False        # FD1: >= 5000 fully decoded
    g2_dict_hit: bool = False       # FD2: dict-hit > 40%
    g3_runs: bool = False           # FD3: >= 10 runs of length >= 3
    g4_context: bool = False        # FD4: >= 5 triples with context
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _classify_token_confidence(
    token: str,
    eva_to_triple: Dict[str, str],
    confirmed_keys: Set[str],
    coda_table: CodaTable,
) -> Tuple[int, int, int]:
    """Return (n_confirmed, n_coda, n_unresolved) for a token."""
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return 0, 0, 0

    classified = classify_token_chars_v2(eva_chars, coda_table)
    n_confirmed = 0
    n_coda = 0
    n_unresolved = 0

    for role, char in classified:
        if role == 'SYLLABIC':
            triple_key = eva_to_triple.get(char, '')
            if triple_key in confirmed_keys:
                n_confirmed += 1
            else:
                n_unresolved += 1
        elif role == 'CODA_MARKER':
            n_coda += 1

    return n_confirmed, n_coda, n_unresolved


def _find_consecutive_runs(indices: List[int]) -> List[List[int]]:
    """Find runs of consecutive integers."""
    if not indices:
        return []
    runs = []
    current_run = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_run.append(indices[i])
        else:
            runs.append(current_run)
            current_run = [indices[i]]
    runs.append(current_run)
    return runs


def _get_unresolved_triples_in_token(
    token: str,
    eva_to_triple: Dict[str, str],
    confirmed_keys: Set[str],
    coda_table: CodaTable,
) -> List[str]:
    """Return list of unresolved triple_keys in this token."""
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return []
    classified = classify_token_chars_v2(eva_chars, coda_table)
    triples = []
    for role, char in classified:
        if role == 'SYLLABIC':
            triple_key = eva_to_triple.get(char, '')
            if triple_key and triple_key not in confirmed_keys:
                triples.append(triple_key)
    return triples


def _score_candidates_in_context(
    anchor_contexts: Dict[str, List[Dict]],
    unresolved: Dict[str, str],
    confirmed: Dict[str, str],
    all_tokens: List[str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
    ref_word_set: Set[str],
) -> Dict[str, str]:
    """For each unresolved triple with anchor context, try candidate syllables.

    Score by how many of its partial-token neighbors produce dict hits when
    the candidate replaces the current assignment.
    """
    # Possible syllable values: all confirmed syllable values + common CV
    candidate_syllables = sorted(set(confirmed.values()) | set(unresolved.values()) |
                                  {'ba', 'be', 'bi', 'bo', 'bu',
                                   'ca', 'ce', 'ci', 'co', 'cu',
                                   'da', 'de', 'di', 'do', 'du',
                                   'fa', 'fe', 'fi', 'fo', 'fu',
                                   'la', 'le', 'li', 'lo', 'lu',
                                   'ma', 'me', 'mi', 'mo', 'mu',
                                   'na', 'ne', 'ni', 'no', 'nu',
                                   'pa', 'pe', 'pi', 'po', 'pu',
                                   'ra', 're', 'ri', 'ro', 'ru',
                                   'sa', 'se', 'si', 'so', 'su',
                                   'ta', 'te', 'ti', 'to', 'tu',
                                   'va', 've', 'vi', 'vo', 'vu'})

    best_candidates: Dict[str, str] = {}

    for triple_key, contexts in anchor_contexts.items():
        if not contexts:
            continue

        scores: Counter = Counter()
        for candidate in candidate_syllables:
            # Build test assignment with this candidate
            test_assignment = {**confirmed, **unresolved}
            test_assignment[triple_key] = candidate

            hits = 0
            for ctx in contexts:
                token_idx = ctx['token_idx']
                token = all_tokens[token_idx]
                result = decode_token_cvc_v2(
                    token, test_assignment, eva_to_triple, coda_table)
                decoded = result.decoded_cvc
                if decoded and '?' not in decoded and decoded in ref_word_set:
                    hits += 1
            scores[candidate] = hits

        if scores:
            ranked = scores.most_common()
            top_syl, top_hits = ranked[0]
            if top_hits > 0:
                best_candidates[triple_key] = top_syl

    return best_candidates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_full_tokens():
    """Track 1: Fully-decoded token exploitation."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 68.1 — Fully-Decoded Token Exploitation")
    print("=" * 50)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    full_assignment = {**confirmed, **unresolved}
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    print(f"  Corpus tokens: {len(all_tokens)}")

    # Build dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"  Dictionary size: {len(ref_word_set)}")

    # --- Step 1: Classify all tokens ---
    print("\n  Classifying token confidence...")
    fully_decoded_indices: List[int] = []
    partial_indices: List[int] = []

    for idx, token in enumerate(all_tokens):
        n_conf, n_coda, n_unres = _classify_token_confidence(
            token, eva_to_triple, confirmed_keys, coda_table)
        if n_unres == 0 and (n_conf + n_coda) > 0:
            fully_decoded_indices.append(idx)
        elif n_unres > 0:
            partial_indices.append(idx)

    n_fully = len(fully_decoded_indices)
    frac = n_fully / len(all_tokens) if all_tokens else 0.0
    print(f"  Fully decoded: {n_fully} ({frac:.1%})")
    print(f"  Partial:       {len(partial_indices)}")

    # --- Step 2: Analyze fully-decoded tokens ---
    print("\n  Analyzing fully-decoded subset...")
    decoded_words: List[str] = []
    n_dict_hits = 0
    vocab: Counter = Counter()

    for idx in fully_decoded_indices:
        result = decode_token_cvc_v2(
            all_tokens[idx], full_assignment, eva_to_triple, coda_table)
        d = result.decoded_cvc if result.decoded_cvc else ''
        decoded_words.append(d)
        if d and '?' not in d and d in ref_word_set:
            n_dict_hits += 1
        if d:
            vocab[d] += 1

    dict_hit_rate = n_dict_hits / n_fully if n_fully > 0 else 0.0
    print(f"  Dict hits: {n_dict_hits}/{n_fully} ({dict_hit_rate:.1%})")
    print(f"  Distinct words: {len(vocab)}")

    top_words = [{'word': w, 'count': c} for w, c in vocab.most_common(50)]

    # --- Step 3: Find consecutive runs ---
    runs = _find_consecutive_runs(fully_decoded_indices)
    runs_ge3 = [r for r in runs if len(r) >= 3]
    longest_run = max((len(r) for r in runs), default=0)
    print(f"  Runs of length >= 3: {len(runs_ge3)}")
    print(f"  Longest run: {longest_run}")

    # Build readable sequences for runs >= 3
    readable_sequences = []
    for run in sorted(runs_ge3, key=lambda r: -len(r))[:20]:
        seq_decoded = []
        for idx in run:
            result = decode_token_cvc_v2(
                all_tokens[idx], full_assignment, eva_to_triple, coda_table)
            seq_decoded.append(result.decoded_cvc if result.decoded_cvc else '?')
        seq_hits = sum(1 for w in seq_decoded if w in ref_word_set)
        readable_sequences.append({
            'start_idx': run[0],
            'length': len(run),
            'decoded': seq_decoded,
            'joined': ' '.join(seq_decoded),
            'dict_hits': seq_hits,
        })

    # --- Step 4: Build anchor contexts ---
    print("\n  Building anchor contexts for unresolved triples...")
    fully_set = set(fully_decoded_indices)
    anchor_contexts: Dict[str, List[Dict]] = {}
    n_partial_with_anchors = 0
    window = 2

    for idx in partial_indices:
        # Check for fully-decoded neighbors
        neighbors = []
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            nidx = idx + offset
            if nidx in fully_set:
                result = decode_token_cvc_v2(
                    all_tokens[nidx], full_assignment, eva_to_triple, coda_table)
                neighbors.append({
                    'offset': offset,
                    'token_idx': nidx,
                    'decoded': result.decoded_cvc if result.decoded_cvc else '',
                })

        if not neighbors:
            continue

        n_partial_with_anchors += 1

        # Which unresolved triples are in this partial token?
        unresolved_triples = _get_unresolved_triples_in_token(
            all_tokens[idx], eva_to_triple, confirmed_keys, coda_table)

        for triple_key in unresolved_triples:
            if triple_key not in anchor_contexts:
                anchor_contexts[triple_key] = []
            anchor_contexts[triple_key].append({
                'token_idx': idx,
                'eva_token': all_tokens[idx],
                'neighbors': neighbors,
            })

    n_triples_with_ctx = len(anchor_contexts)
    print(f"  Partial tokens with anchors: {n_partial_with_anchors}")
    print(f"  Unresolved triples with context: {n_triples_with_ctx}")

    for tk, ctxs in sorted(anchor_contexts.items()):
        print(f"    {tk}: {len(ctxs)} context windows")

    # --- Step 5: Score candidates ---
    print("\n  Scoring candidates from context...")
    # Limit context windows per triple for performance
    limited_contexts: Dict[str, List[Dict]] = {}
    for tk, ctxs in anchor_contexts.items():
        limited_contexts[tk] = ctxs[:200]

    triple_candidates = _score_candidates_in_context(
        limited_contexts, unresolved, confirmed,
        all_tokens, eva_to_triple, coda_table, ref_word_set)

    triple_details = []
    for tk in sorted(unresolved.keys()):
        current = unresolved[tk]
        proposed = triple_candidates.get(tk)
        n_ctx = len(anchor_contexts.get(tk, []))
        detail = {
            'triple_key': tk,
            'current_value': current,
            'proposed_value': proposed if proposed else current,
            'n_context_windows': n_ctx,
            'changed': proposed is not None and proposed != current,
        }
        triple_details.append(detail)
        if proposed and proposed != current:
            print(f"    {tk}: {current} -> {proposed} (from {n_ctx} windows)")

    # --- Gates ---
    g1 = n_fully >= 5000
    g2 = dict_hit_rate > 0.40
    g3 = len(runs_ge3) >= 10
    g4 = n_triples_with_ctx >= 5
    gates_passed = sum([g1, g2, g3, g4])

    result = FullTokenResult(
        n_corpus_tokens=len(all_tokens),
        n_fully_decoded=n_fully,
        fully_decoded_fraction=round(frac, 4),
        n_dict_hits=n_dict_hits,
        dict_hit_rate=round(dict_hit_rate, 4),
        n_distinct_words=len(vocab),
        top_words=top_words,
        n_runs_ge3=len(runs_ge3),
        longest_run=longest_run,
        readable_sequences=readable_sequences,
        n_partial_with_anchors=n_partial_with_anchors,
        n_triples_with_context=n_triples_with_ctx,
        triple_candidates=triple_candidates,
        triple_details=triple_details,
        g1_n_fully=g1,
        g2_dict_hit=g2,
        g3_runs=g3,
        g4_context=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p68_full_tokens.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Fully decoded:   {n_fully} ({frac:.1%}) ({'PASS' if g1 else 'FAIL'} >= 5000)")
    print(f"  Dict hit rate:   {dict_hit_rate:.1%} ({'PASS' if g2 else 'FAIL'} > 40%)")
    print(f"  Runs >= 3:       {len(runs_ge3)} ({'PASS' if g3 else 'FAIL'} >= 10)")
    print(f"  Triples w/ ctx:  {n_triples_with_ctx} ({'PASS' if g4 else 'FAIL'} >= 5)")
    print(f"  Candidates:      {len(triple_candidates)}")
    print(f"  Gates: {gates_passed}/4")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
