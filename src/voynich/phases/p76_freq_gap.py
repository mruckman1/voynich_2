"""
Phase 76, Track 3: Frequency-Identification Gap Analysis
=========================================================
Identify which unresolved triples block the most frequent unidentified
token types. Ranks triples by "blocking impact" -- resolving the
top-blocking triple would identify the most new frequent types.

Dependency chain:
    results/p75_t1.json               (Phase 75 Track 3)
    results/p75_redecode.json         (Phase 75 Step 0)
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
    results/p76_wildcard_prop.json    (optional, Track 1 cross-ref)
        -> results/p76_freq_gap.json
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
from voynich.phases.corrected_coda import classify_token_chars_v2, decode_token_cvc_v2
from voynich.phases.p69_clean_validation import _get_confirmed_and_unresolved
from voynich.phases.p75_redecode import _build_3coda_table


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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FreqGapResult:
    phase: str = "76"
    step: str = "76.3"
    experiment: str = "frequency_gap_analysis"
    # Top-500 analysis
    n_top_types: int = 0
    n_identified: int = 0
    n_dict_hit: int = 0
    n_unidentified: int = 0
    identification_rate_top500: float = 0.0
    # Triple priority ranking
    triple_priority: List[Dict[str, Any]] = field(default_factory=list)
    n_blocking_triples: int = 0
    # Top unidentified types
    top_unidentified: List[Dict[str, Any]] = field(default_factory=list)
    # Track 1 cross-reference
    track1_available: bool = False
    n_top5_with_constraints: int = 0
    top5_overlap: List[str] = field(default_factory=list)
    # Gates
    fg1_id_rate: bool = False
    fg2_top_blocking_resolved: bool = False
    fg3_top5_constrained: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helper: get unresolved triples in a token
# ---------------------------------------------------------------------------

def _get_unresolved_triples_in_token(
    token: str,
    eva_to_triple: Dict[str, str],
    unresolved_keys: Set[str],
    coda_table,
) -> List[str]:
    """Return list of unresolved triple_keys present in the token."""
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return []

    classified = classify_token_chars_v2(eva_chars, coda_table)
    unresolved_in_token: List[str] = []

    for role, char in classified:
        if role == 'SYLLABIC':
            triple_key = eva_to_triple.get(char, '')
            if triple_key and triple_key in unresolved_keys:
                if triple_key not in unresolved_in_token:
                    unresolved_in_token.append(triple_key)

    return unresolved_in_token


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_freq_gap() -> FreqGapResult:
    """Track 3: Frequency-identification gap analysis."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 76.3 -- Frequency-Identification Gap Analysis")
    print("=" * 54)

    # --- Load shared data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    unresolved_keys = set(unresolved.keys())
    full_assignment = {**confirmed, **unresolved}

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = _build_3coda_table()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")
    print(f"  Corpus tokens: {len(all_tokens)}")
    print(f"  Dictionary size: {len(ref_word_set)}")

    # --- Load T1 identifications ---
    t1_data = _safe_load(os.path.join(rd, 'p75_t1.json'))
    t1_identifications = t1_data.get('identifications', [])
    t1_identified_types: Set[str] = set()
    for ident in t1_identifications:
        tok = ident.get('token', '')
        if tok:
            t1_identified_types.add(tok)
    print(f"  T1 identified types: {len(t1_identified_types)}")

    # --- Load decoded corpus from p75_redecode or decode fresh ---
    redecode_data = _safe_load(os.path.join(rd, 'p75_redecode.json'))
    decoded_tokens = redecode_data.get('decoded_tokens', [])

    if len(decoded_tokens) != len(all_tokens):
        print("  Decoded corpus not available or mismatched; decoding fresh...")
        decoded_tokens = []
        for token in all_tokens:
            result = decode_token_cvc_v2(
                token, full_assignment, eva_to_triple, coda_table)
            decoded_tokens.append(result.decoded_cvc)

    # --- Build type frequency table ---
    type_counter: Counter = Counter(all_tokens)
    top_types = type_counter.most_common(500)
    n_top_types = len(top_types)
    print(f"  Top-{n_top_types} token types (by frequency)")

    # --- Build type -> decoded mapping ---
    type_to_decoded: Dict[str, str] = {}
    for tok, dec in zip(all_tokens, decoded_tokens):
        if tok not in type_to_decoded:
            type_to_decoded[tok] = dec

    # --- Classify each top type ---
    identified_types: List[str] = []
    dict_hit_types: List[str] = []
    unidentified_types: List[Tuple[str, int]] = []  # (type, freq)

    for token_type, freq in top_types:
        decoded = type_to_decoded.get(token_type, '')
        is_t1 = token_type in t1_identified_types
        is_dict_hit = (decoded and decoded.lower() in ref_word_set)

        if is_t1:
            identified_types.append(token_type)
        elif is_dict_hit:
            dict_hit_types.append(token_type)
        else:
            unidentified_types.append((token_type, freq))

    n_identified = len(identified_types)
    n_dict_hit = len(dict_hit_types)
    n_unidentified = len(unidentified_types)
    identification_rate = (n_identified + n_dict_hit) / n_top_types if n_top_types > 0 else 0.0

    print(f"\n  Top-500 classification:")
    print(f"    T1-identified:  {n_identified}")
    print(f"    Dict-hit only:  {n_dict_hit}")
    print(f"    Unidentified:   {n_unidentified}")
    print(f"    ID rate:        {identification_rate:.1%}")

    # --- For unidentified types: find blocking triples ---
    print("\n  Analyzing blocking triples for unidentified types...")

    # triple_key -> list of (type, freq) that it blocks
    blocking_map: Dict[str, List[Tuple[str, int]]] = {}

    for token_type, freq in unidentified_types:
        unres_triples = _get_unresolved_triples_in_token(
            token_type, eva_to_triple, unresolved_keys, coda_table)
        for tk in unres_triples:
            if tk not in blocking_map:
                blocking_map[tk] = []
            blocking_map[tk].append((token_type, freq))

    # Rank by number of blocked types
    triple_priority: List[Dict[str, Any]] = []
    for triple_key in sorted(blocking_map.keys(),
                             key=lambda k: -len(blocking_map[k])):
        blocked = blocking_map[triple_key]
        current_value = full_assignment.get(triple_key, '?')
        triple_priority.append({
            'triple_key': triple_key,
            'n_blocked_types': len(blocked),
            'total_blocked_tokens': sum(f for _, f in blocked),
            'current_value': current_value,
            'sample_blocked': [t for t, _ in blocked[:5]],
        })

    n_blocking = len(triple_priority)
    print(f"  Blocking triples: {n_blocking}")

    if triple_priority:
        print(f"\n  Top-10 blocking triples:")
        for entry in triple_priority[:10]:
            print(f"    {entry['triple_key']}: blocks {entry['n_blocked_types']} types "
                  f"({entry['total_blocked_tokens']} tokens), "
                  f"current='{entry['current_value']}'")

    # --- Top-30 unidentified types ---
    top_unidentified: List[Dict[str, Any]] = []
    for token_type, freq in unidentified_types[:30]:
        decoded = type_to_decoded.get(token_type, '')
        unres_triples = _get_unresolved_triples_in_token(
            token_type, eva_to_triple, unresolved_keys, coda_table)
        top_unidentified.append({
            'eva_type': token_type,
            'frequency': freq,
            'decoded': decoded,
            'blocking_triples': unres_triples,
            'n_blocking': len(unres_triples),
        })

    if top_unidentified:
        print(f"\n  Top-10 unidentified types:")
        for entry in top_unidentified[:10]:
            print(f"    {entry['eva_type']} (freq={entry['frequency']}, "
                  f"decoded='{entry['decoded']}', "
                  f"blocked by {entry['blocking_triples']})")

    # --- Cross-reference with Track 1 (if available) ---
    track1_data = _safe_load(os.path.join(rd, 'p76_wildcard_prop.json'))
    track1_available = bool(track1_data)
    n_top5_with_constraints = 0
    top5_overlap: List[str] = []

    if track1_available:
        print("\n  Cross-referencing with Track 1 (wildcard propagation)...")
        track1_resolution = track1_data.get('per_triple_resolution', {})

        top5_keys = [entry['triple_key'] for entry in triple_priority[:5]]
        for tk in top5_keys:
            if tk in track1_resolution:
                confidence = track1_resolution[tk].get('confidence', '')
                if confidence in ('RESOLVED', 'LIKELY', 'TENTATIVE'):
                    n_top5_with_constraints += 1
                    top5_overlap.append(tk)

        print(f"  Top-5 blocking triples with Track 1 constraints: "
              f"{n_top5_with_constraints}/5")
        if top5_overlap:
            for tk in top5_overlap:
                info = track1_resolution[tk]
                print(f"    {tk}: '{info.get('best_syllable', '?')}' "
                      f"({info.get('confidence', '?')})")

        # Check if the most-blocking triple is RESOLVED in Track 1
        most_blocking_key = triple_priority[0]['triple_key'] if triple_priority else ''
        most_blocking_resolved = (
            most_blocking_key in track1_resolution and
            track1_resolution[most_blocking_key].get('confidence', '') == 'RESOLVED'
        )
    else:
        print("\n  Track 1 results not available; skipping cross-reference.")
        most_blocking_resolved = False

    # ===================================================================
    # Gates
    # ===================================================================
    fg1 = identification_rate > 0.40
    fg2 = most_blocking_resolved
    fg3 = n_top5_with_constraints >= 3

    gates_passed = sum([fg1, fg2, fg3])

    if fg1 and fg2 and fg3:
        verdict = 'GAP_ACTIONABLE'
    elif fg1 and (fg2 or fg3):
        verdict = 'GAP_PARTIAL'
    elif fg1:
        verdict = 'GAP_IDENTIFIED'
    else:
        verdict = 'GAP_WIDE'

    print(f"\n  Gates:")
    print(f"    FG1 top-500 ID rate > 40%:          {fg1} ({identification_rate:.1%})")
    print(f"    FG2 most-blocking = RESOLVED (T1):  {fg2}")
    print(f"    FG3 >= 3 of top-5 have constraints: {fg3} ({n_top5_with_constraints}/5)")
    print(f"  Gates passed: {gates_passed}/3")
    print(f"  Verdict: {verdict}")

    # ===================================================================
    # Build and save result
    # ===================================================================
    result = FreqGapResult(
        n_top_types=n_top_types,
        n_identified=n_identified,
        n_dict_hit=n_dict_hit,
        n_unidentified=n_unidentified,
        identification_rate_top500=round(identification_rate, 4),
        triple_priority=triple_priority,
        n_blocking_triples=n_blocking,
        top_unidentified=top_unidentified,
        track1_available=track1_available,
        n_top5_with_constraints=n_top5_with_constraints,
        top5_overlap=top5_overlap,
        fg1_id_rate=fg1,
        fg2_top_blocking_resolved=fg2,
        fg3_top5_constrained=fg3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p76_freq_gap.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
