"""
Phase 16.4 – Minimal Pair Subtraction (Approach E)
===================================================
Finds token pairs that differ by exactly one EVA character and tests
whether removing that character preserves or destroys dictionary-hit
status.  If two tokens differ by one char and both decode to recognised
Latin words, that char is likely a modifier (it doesn't carry independent
syllabic content).

Dependency chain:
    combined_refine.json  (Phase 15 best_assignment)
    corpus (IVTFF)
        → modifier_minimal_pairs.json (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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


def _levenshtein(s1: str, s2: str) -> int:
    """Simple Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MinimalPairEntry:
    token_long: str             # token with extra char
    token_short: str            # token without
    differing_char: str         # EVA char present in long, absent in short
    differing_position: str     # 'initial', 'medial', 'final'
    decoded_long: str
    decoded_short: str
    long_is_hit: bool
    short_is_hit: bool
    both_hits: bool
    removal_preserves_hit: bool   # short is hit when long is hit
    removal_creates_hit: bool     # short is hit when long is NOT hit
    edit_distance: int


@dataclass
class MinimalPairsResult:
    n_unique_tokens: int
    n_pairs_found: int
    n_both_hits: int
    n_removal_preserves: int
    n_removal_creates: int
    pairs_sample: List[Dict]          # top 50 pairs
    per_char_evidence: List[Dict]     # aggregated per EVA char
    modifier_candidates: List[str]    # chars where removal often helps
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def find_minimal_pairs(
    unique_tokens: Set[str],
) -> List[Tuple[str, str, str, str]]:
    """Find all token pairs differing by exactly one EVA character.

    Returns list of (token_long, token_short, differing_char, position).
    token_long has the extra character; token_short is the shorter form.
    """
    pairs: List[Tuple[str, str, str, str]] = []

    # Build a set for O(1) lookup
    token_set = set(unique_tokens)

    for token in unique_tokens:
        chars = tokenize_eva_chars(token)
        n = len(chars)
        if n < 2:
            continue

        # Try removing each character position
        for ci in range(n):
            remaining = chars[:ci] + chars[ci + 1:]
            short_token = ''.join(remaining)

            if short_token in token_set and short_token != token:
                removed_char = chars[ci]
                if ci == 0:
                    pos = 'initial'
                elif ci == n - 1:
                    pos = 'final'
                else:
                    pos = 'medial'
                pairs.append((token, short_token, removed_char, pos))

    return pairs


def analyze_pairs(
    pairs: List[Tuple[str, str, str, str]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
) -> List[MinimalPairEntry]:
    """Decode both tokens in each pair and check dict-hit status."""
    entries: List[MinimalPairEntry] = []

    for token_long, token_short, diff_char, pos in pairs:
        decoded_long = decode_token(token_long, assignment, eva_to_triple)
        decoded_short = decode_token(token_short, assignment, eva_to_triple)

        long_hit = decoded_long.lower() in ref_word_set
        short_hit = decoded_short.lower() in ref_word_set

        entries.append(MinimalPairEntry(
            token_long=token_long,
            token_short=token_short,
            differing_char=diff_char,
            differing_position=pos,
            decoded_long=decoded_long,
            decoded_short=decoded_short,
            long_is_hit=long_hit,
            short_is_hit=short_hit,
            both_hits=long_hit and short_hit,
            removal_preserves_hit=long_hit and short_hit,
            removal_creates_hit=(not long_hit) and short_hit,
            edit_distance=_levenshtein(decoded_long, decoded_short),
        ))

    return entries


def aggregate_per_char(
    entries: List[MinimalPairEntry],
) -> List[Dict]:
    """Aggregate minimal pair evidence per EVA character."""
    char_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            'n_pairs': 0,
            'n_removal_preserves': 0,
            'n_removal_creates': 0,
            'n_removal_destroys': 0,
            'n_both_hits': 0,
            'n_long_hit_only': 0,
            'n_short_hit_only': 0,
            'n_neither_hit': 0,
        }
    )

    for e in entries:
        s = char_stats[e.differing_char]
        s['n_pairs'] += 1
        if e.removal_preserves_hit:
            s['n_removal_preserves'] += 1
        if e.removal_creates_hit:
            s['n_removal_creates'] += 1
        if e.long_is_hit and not e.short_is_hit:
            s['n_removal_destroys'] += 1
            s['n_long_hit_only'] += 1
        if e.both_hits:
            s['n_both_hits'] += 1
        if e.short_is_hit and not e.long_is_hit:
            s['n_short_hit_only'] += 1
        if not e.long_is_hit and not e.short_is_hit:
            s['n_neither_hit'] += 1

    result = []
    for ch, s in sorted(char_stats.items(), key=lambda x: -x[1]['n_pairs']):
        n = s['n_pairs']
        # Modifier score: how often removal helps vs hurts
        helps = s['n_removal_preserves'] + s['n_removal_creates']
        hurts = s['n_removal_destroys']
        modifier_score = helps / (helps + hurts) if (helps + hurts) > 0 else 0.0

        result.append({
            'eva_char': ch,
            **s,
            'modifier_score': round(modifier_score, 4),
        })

    # Sort by modifier_score descending
    result.sort(key=lambda x: -x['modifier_score'])
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_modifier_minimal_pairs() -> None:
    """Step 16.4: Minimal pair modifier evidence (Approach E)."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 16.4: Minimal Pair Subtraction (Approach E)")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load Phase 15 best assignment ───
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found — run combined-refine first")
        return

    with open(refine_path) as f:
        refine_data = json.load(f)

    assignment = refine_data.get('best_assignment', {})
    print(f"\n  1. Loaded Phase 15 best assignment ({len(assignment)} triples)")

    # ─── Load corpus ───
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    unique_tokens = set(tokens)
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"      {len(tokens)} tokens, {len(unique_tokens)} unique")

    # ─── Build reference word set ───
    print("\n  3. Building expanded reference word set …")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()

    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"      {len(ref_word_set)} words in reference set")

    # ─── Find minimal pairs ───
    print("\n  4. Finding minimal pairs …")
    pairs = find_minimal_pairs(unique_tokens)
    print(f"      {len(pairs)} minimal pairs found")

    if not pairs:
        print("  [SKIP] No minimal pairs found")
        result = MinimalPairsResult(
            n_unique_tokens=len(unique_tokens),
            n_pairs_found=0,
            n_both_hits=0,
            n_removal_preserves=0,
            n_removal_creates=0,
            pairs_sample=[],
            per_char_evidence=[],
            modifier_candidates=[],
            gate_passed=False,
            verdict="FAIL: No minimal pairs found in corpus.",
            runtime_seconds=round(time.time() - t0, 2),
        )
        out_path = os.path.join(rd, 'modifier_minimal_pairs.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(result), f, indent=2)
        print(f"\n  → {out_path}")
        return

    # ─── Analyse pairs ───
    print("\n  5. Analysing pair semantics …")
    entries = analyze_pairs(pairs, assignment, eva_to_triple, ref_word_set)

    n_both = sum(1 for e in entries if e.both_hits)
    n_preserves = sum(1 for e in entries if e.removal_preserves_hit)
    n_creates = sum(1 for e in entries if e.removal_creates_hit)

    print(f"      Both tokens are dict hits: {n_both}")
    print(f"      Removal preserves hit: {n_preserves}")
    print(f"      Removal creates new hit: {n_creates}")

    # ─── Aggregate per character ───
    print("\n  6. Aggregating per-character evidence …")
    per_char = aggregate_per_char(entries)

    # Modifier candidates: chars where removal helps more than it hurts
    modifier_candidates = [
        p['eva_char'] for p in per_char
        if p['modifier_score'] >= 0.5 and p['n_pairs'] >= 3
    ]

    print(f"      Modifier candidates (score >= 0.5, pairs >= 3): {modifier_candidates}")

    print(f"\n  7. Per-character summary (top 15):")
    print(f"      {'Char':<8} {'Pairs':>6} {'Preserv':>8} {'Creates':>8} "
          f"{'Destroy':>8} {'Score':>7}")
    print("      " + "-" * 55)
    for p in per_char[:15]:
        print(f"      {p['eva_char']:<8} {p['n_pairs']:>6} "
              f"{p['n_removal_preserves']:>8} {p['n_removal_creates']:>8} "
              f"{p['n_removal_destroys']:>8} {p['modifier_score']:>7.3f}")

    # ─── Sample pairs ───
    sample_entries = sorted(entries, key=lambda e: (
        -(1 if e.removal_preserves_hit or e.removal_creates_hit else 0),
        e.edit_distance,
    ))[:50]

    # ─── Gate ───
    total_helps = n_preserves + n_creates
    gate_passed = total_helps >= 5
    verdict = (
        f"PASS: {total_helps} pairs where removal preserves/creates dict hit. "
        f"{len(modifier_candidates)} modifier candidates."
        if gate_passed
        else f"FAIL: Only {total_helps} helpful removals (need >= 5)."
    )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ─── Save ───
    result = MinimalPairsResult(
        n_unique_tokens=len(unique_tokens),
        n_pairs_found=len(pairs),
        n_both_hits=n_both,
        n_removal_preserves=n_preserves,
        n_removal_creates=n_creates,
        pairs_sample=[_convert(asdict(e)) for e in sample_entries],
        per_char_evidence=per_char,
        modifier_candidates=modifier_candidates,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'modifier_minimal_pairs.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
