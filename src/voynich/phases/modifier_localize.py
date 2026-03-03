"""
Phase 16.5 – Dictionary Hit Localization (Approach C)
=====================================================
Among decoded tokens that are dictionary hits, localizes which EVA
characters contribute to the matching portion of the decoded string
and which produce "padding" syllables that don't appear in the matched
Latin word.

Characters that consistently appear in padding positions are modifier
candidates — they produce extra syllables not present in the target word.

Dependency chain:
    combined_refine.json  (Phase 15 best_assignment)
    corpus (IVTFF)
        → modifier_localize.json (this step)
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LocalizationEntry:
    token: str
    eva_chars: List[str]
    decoded: str
    matched_word: str
    syllable_sequence: List[str]    # per-char syllable output
    matching_char_indices: List[int]
    padding_char_indices: List[int]
    padding_chars: List[str]
    matching_chars: List[str]


@dataclass
class LocalizationResult:
    n_hit_tokens: int
    n_tokens_with_padding: int
    n_tokens_analyzed: int
    padding_char_counts: Dict[str, int]
    match_char_counts: Dict[str, int]
    padding_ratio_per_char: Dict[str, float]
    modifier_candidates: List[str]
    modifier_threshold: float
    sample_localizations: List[Dict]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _best_substring_alignment(
    decoded: str,
    target: str,
    syllables: List[str],
) -> Tuple[int, int]:
    """Find the best alignment of target within decoded.

    Returns (start_syl_idx, end_syl_idx) indicating which syllables
    form the best match with the target word.
    """
    # Build cumulative character positions for each syllable boundary
    boundaries = [0]
    for syl in syllables:
        boundaries.append(boundaries[-1] + len(syl))

    decoded_lower = decoded.lower()
    target_lower = target.lower()

    best_start = 0
    best_end = len(syllables)
    best_score = -1

    # Try all contiguous syllable spans
    for si in range(len(syllables)):
        for ei in range(si + 1, len(syllables) + 1):
            substring = decoded_lower[boundaries[si]:boundaries[ei]]
            # Score: longest common subsequence length
            score = _lcs_length(substring, target_lower)
            if score > best_score:
                best_score = score
                best_start = si
                best_end = ei

    return best_start, best_end


def _lcs_length(s1: str, s2: str) -> int:
    """Length of the longest common subsequence."""
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if s1[i] == s2[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev = curr
    return prev[n]


def _find_best_substring_match(
    decoded: str,
    ref_word_set: set,
) -> Optional[str]:
    """Find the longest dictionary word that is a substring of decoded.

    If the full decoded string is a match AND is <= 4 chars (short word),
    return None (no padding to detect).  Otherwise, find the longest
    proper substring match.
    """
    decoded_lower = decoded.lower()
    best: Optional[str] = None
    best_len = 0

    # Check all substrings of decoded_lower (length >= 2)
    n = len(decoded_lower)
    for slen in range(n, 1, -1):  # longest first
        for start in range(n - slen + 1):
            sub = decoded_lower[start:start + slen]
            if sub in ref_word_set:
                if slen == n:
                    # Full match — only interesting if the decoded string
                    # is long enough that some syllables could be padding
                    if n <= 4:
                        continue  # short full match, skip
                if slen > best_len:
                    best = sub
                    best_len = slen
        if best:
            break  # found at this length, no need for shorter

    return best


def localize_hits(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    max_tokens: int = 2000,
) -> List[LocalizationEntry]:
    """For each token, find the best dictionary substring in its decoded form.

    Tokens where the decoded string is longer than the matched word have
    "padding" characters — EVA chars that produce extra syllables not
    part of the matched Latin word.
    """
    entries: List[LocalizationEntry] = []

    for token in tokens[:max_tokens]:
        decoded = decode_token(token, assignment, eva_to_triple)
        decoded_lower = decoded.lower()

        # Skip very short decoded strings
        if len(decoded_lower) < 3:
            continue

        chars = tokenize_eva_chars(token)
        syllables: List[str] = []
        for ch in chars:
            triple = eva_to_triple.get(ch)
            if triple and triple in assignment:
                syllables.append(assignment[triple])
            else:
                syllables.append('?')

        if not syllables:
            continue

        # Try to find a dictionary word that is a substring of the decoded
        matched_word = _find_best_substring_match(decoded, ref_word_set)
        if not matched_word:
            continue

        # Align: which syllable span best matches the dictionary word?
        start_idx, end_idx = _best_substring_alignment(
            decoded, matched_word, syllables,
        )

        # Only keep entries where there IS padding
        matching_indices = list(range(start_idx, end_idx))
        padding_indices = [
            i for i in range(len(chars))
            if i not in matching_indices
        ]

        if not padding_indices:
            continue  # no padding = no signal

        entries.append(LocalizationEntry(
            token=token,
            eva_chars=chars,
            decoded=decoded,
            matched_word=matched_word,
            syllable_sequence=syllables,
            matching_char_indices=matching_indices,
            padding_char_indices=padding_indices,
            padding_chars=[chars[i] for i in padding_indices],
            matching_chars=[chars[i] for i in matching_indices],
        ))

    return entries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_modifier_localize() -> None:
    """Step 16.5: Hit localization padding analysis (Approach C)."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 16.5: Dictionary Hit Localization (Approach C)")
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
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"      {len(tokens)} tokens")

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

    # ─── Localize hits ───
    print("\n  4. Localizing dictionary hits …")
    entries = localize_hits(tokens, assignment, eva_to_triple, ref_word_set)
    n_with_padding = sum(1 for e in entries if e.padding_char_indices)
    print(f"      {len(entries)} hit tokens analyzed")
    print(f"      {n_with_padding} tokens have padding characters")

    # ─── Aggregate per character ───
    print("\n  5. Aggregating per-character padding statistics …")
    padding_counts: Counter = Counter()
    match_counts: Counter = Counter()

    for e in entries:
        for ch in e.padding_chars:
            padding_counts[ch] += 1
        for ch in e.matching_chars:
            match_counts[ch] += 1

    # Compute padding ratio per char
    all_chars = set(padding_counts.keys()) | set(match_counts.keys())
    padding_ratio: Dict[str, float] = {}
    for ch in all_chars:
        p = padding_counts.get(ch, 0)
        m = match_counts.get(ch, 0)
        total = p + m
        padding_ratio[ch] = p / total if total > 0 else 0.0

    # Sort by padding ratio descending
    sorted_chars = sorted(padding_ratio.items(), key=lambda x: -x[1])

    threshold = 0.6
    modifier_candidates = [
        ch for ch, ratio in sorted_chars
        if ratio >= threshold and (padding_counts.get(ch, 0) + match_counts.get(ch, 0)) >= 5
    ]

    print(f"\n  6. Per-character padding ratios (top 15):")
    print(f"      {'Char':<8} {'Padding':>8} {'Match':>8} {'Total':>8} {'Ratio':>7}")
    print("      " + "-" * 45)
    for ch, ratio in sorted_chars[:15]:
        p = padding_counts.get(ch, 0)
        m = match_counts.get(ch, 0)
        print(f"      {ch:<8} {p:>8} {m:>8} {p + m:>8} {ratio:>7.3f}")

    print(f"\n      Modifier candidates (ratio >= {threshold}, count >= 5): "
          f"{modifier_candidates}")

    # ─── Sample localizations ───
    sample = entries[:30]

    print(f"\n  7. Sample localizations (first 10):")
    for e in entries[:10]:
        padding_str = ','.join(e.padding_chars) if e.padding_chars else '(none)'
        print(f"      {e.token:>15} → {e.decoded:<15} "
              f"match={e.matched_word:<12} padding=[{padding_str}]")

    # ─── Gate ───
    gate_passed = len(modifier_candidates) >= 3
    verdict = (
        f"PASS: {len(modifier_candidates)} chars with padding ratio >= {threshold}. "
        f"{n_with_padding}/{len(entries)} hit tokens have padding."
        if gate_passed
        else f"FAIL: Only {len(modifier_candidates)} chars with high padding ratio "
        f"(need >= 3 with ratio >= {threshold} and count >= 5)."
    )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ─── Save ───
    result = LocalizationResult(
        n_hit_tokens=len(entries),
        n_tokens_with_padding=n_with_padding,
        n_tokens_analyzed=min(len(tokens), 2000),
        padding_char_counts=dict(padding_counts.most_common()),
        match_char_counts=dict(match_counts.most_common()),
        padding_ratio_per_char={ch: round(r, 4) for ch, r in sorted_chars},
        modifier_candidates=modifier_candidates,
        modifier_threshold=threshold,
        sample_localizations=[_convert(asdict(e)) for e in sample],
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'modifier_localize.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
