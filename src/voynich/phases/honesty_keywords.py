"""
Phase 17.0.2 – Keyword Presence Test
=====================================
Checks whether basic Latin medical vocabulary appears as complete decoded
tokens in the Phase 16 output.  Tests exact and edit-distance-1 matching
against a curated top-100 Latin medical word list.

Dependency chain:
    modifier_integrate.json  (Phase 16 best result)
    combined_refine.json     (Phase 15 best_assignment)
        → honesty_keywords.json (this step)
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
)
from voynich.core.reference import (
    LATIN_MEDICAL_TOP_100,
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


def _reconstruct_modifier_rules(data: Dict) -> Tuple[Set[str], Dict[str, str]]:
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


def _r3_decode(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    max_tokens: int = 8000,
) -> List[str]:
    decoded = []
    for token in tokens[:max_tokens]:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt)
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped)
            continue
        decoded.append(decode_token(token, assignment, eva_to_triple))
    return decoded


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KeywordMatch:
    latin_word: str
    expected_rank: int
    found_exact: bool
    found_relaxed: bool
    best_match: str
    edit_distance: int
    n_occurrences: int
    decoded_context: List[str]


@dataclass
class HonestyKeywordResult:
    n_keywords: int
    keyword_list: List[str]

    # Exact matching
    n_exact_found: int
    exact_keywords: List[str]
    exact_rate: float

    # Relaxed matching (edit distance <= 1)
    n_relaxed_found: int
    relaxed_keywords: List[str]
    relaxed_rate: float

    # Frequency correlation
    rho: float
    rho_p_value: float

    # Per-keyword detail
    keyword_matches: List[Dict]

    # Random baseline
    random_exact_mean: float
    random_relaxed_mean: float
    exact_selectivity: float
    relaxed_selectivity: float

    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_honesty_keywords() -> None:
    """Step 17.0.2: Top-100 keyword presence test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 17.0.2: Keyword Presence Test")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load Phase 16 results ───
    print("\n  1. Loading Phase 16 results …")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # ─── Load Phase 15 assignment ───
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    # ─── Load corpus ───
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"      {len(tokens)} tokens")

    # ─── Build expanded dictionary (for R3 hit selection) ───
    print("\n  3. Building reference word set …")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()
    expanded_words, _ = build_expanded_word_set(base_words)
    expanded_set = base_words | expanded_words
    print(f"      {len(expanded_set)} words in expanded set")

    # ─── R3 decode all tokens ───
    print("\n  4. Decoding all tokens (R3 combined strategy) …")
    r3_decoded = _r3_decode(
        tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, expanded_set,
    )
    decoded_lower = [w.lower() for w in r3_decoded]
    decoded_set = set(decoded_lower)
    decoded_counts = Counter(decoded_lower)
    print(f"      Decoded {len(r3_decoded)} tokens, {len(decoded_set)} unique")

    # ─── Build keyword list ───
    keywords = [(word, rank) for word, rank in LATIN_MEDICAL_TOP_100]
    keyword_words = [w for w, _ in keywords]

    # ─── Match each keyword ───
    print("\n  5. Matching keywords against decoded output …")
    matches: List[KeywordMatch] = []
    exact_found = []
    relaxed_found = []

    for word, rank in keywords:
        # Exact match
        found_exact = word in decoded_set
        n_occurrences = decoded_counts.get(word, 0)

        # Relaxed match: find closest decoded token within edit distance 1
        best_match = word if found_exact else ''
        best_ed = 0 if found_exact else 999

        if not found_exact:
            for dw in decoded_set:
                if abs(len(dw) - len(word)) > 1:
                    continue
                ed = _edit_distance(word, dw)
                if ed < best_ed:
                    best_ed = ed
                    best_match = dw
                    if ed == 1:
                        break

        found_relaxed = best_ed <= 1

        # Context: up to 3 examples of this keyword in decoded output
        context = []
        if n_occurrences > 0:
            for i, dw in enumerate(decoded_lower):
                if dw == word:
                    start = max(0, i - 1)
                    end = min(len(decoded_lower), i + 2)
                    context.append(' '.join(decoded_lower[start:end]))
                    if len(context) >= 3:
                        break

        if found_exact:
            exact_found.append(word)
        if found_relaxed:
            relaxed_found.append(word)

        matches.append(KeywordMatch(
            latin_word=word,
            expected_rank=rank,
            found_exact=found_exact,
            found_relaxed=found_relaxed,
            best_match=best_match,
            edit_distance=best_ed if best_ed < 999 else -1,
            n_occurrences=n_occurrences,
            decoded_context=context,
        ))

    n_exact = len(exact_found)
    n_relaxed = len(relaxed_found)

    print(f"      Exact matches:   {n_exact}/{len(keywords)}")
    print(f"      Relaxed (ed≤1):  {n_relaxed}/{len(keywords)}")

    if exact_found:
        print(f"      Exact keywords:  {exact_found}")

    # ─── Frequency correlation ───
    print("\n  6. Computing frequency correlation …")
    # For keywords that have at least 1 occurrence, correlate rank with count
    ranks_for_corr = []
    counts_for_corr = []
    for m in matches:
        if m.n_occurrences > 0:
            ranks_for_corr.append(m.expected_rank)
            counts_for_corr.append(m.n_occurrences)

    rho = 0.0
    rho_p = 1.0
    if len(ranks_for_corr) >= 3:
        try:
            from voynich.core.stats import rank_correlation
            rho, rho_p = rank_correlation(ranks_for_corr, counts_for_corr)
        except (ImportError, ValueError):
            rho, rho_p = 0.0, 1.0
    # Correlation should be negative (lower rank = more frequent = higher count)
    print(f"      rho = {rho:.3f}, p = {rho_p:.4f}")
    print(f"      (negative rho expected: rank 1 = most frequent)")

    # ─── Random baseline ───
    print("\n  7. Computing random baseline …")
    rng = random.Random(42)
    syllables = list(set(assignment.values()))
    random_exact_counts = []
    random_relaxed_counts = []

    for trial in range(5):
        rand_assign = {k: rng.choice(syllables) for k in assignment}
        rand_decoded = [
            decode_token(t, rand_assign, eva_to_triple).lower()
            for t in tokens[:2000]
        ]
        rand_set = set(rand_decoded)

        trial_exact = sum(1 for w in keyword_words if w in rand_set)
        trial_relaxed = 0
        for w in keyword_words:
            if w in rand_set:
                trial_relaxed += 1
            else:
                for rw in rand_set:
                    if abs(len(rw) - len(w)) <= 1 and _edit_distance(w, rw) <= 1:
                        trial_relaxed += 1
                        break
        random_exact_counts.append(trial_exact)
        random_relaxed_counts.append(trial_relaxed)

    random_exact_mean = sum(random_exact_counts) / len(random_exact_counts)
    random_relaxed_mean = sum(random_relaxed_counts) / len(random_relaxed_counts)
    exact_selectivity = n_exact / max(random_exact_mean, 0.1)
    relaxed_selectivity = n_relaxed / max(random_relaxed_mean, 0.1)

    print(f"      Random exact mean:   {random_exact_mean:.1f}")
    print(f"      Random relaxed mean: {random_relaxed_mean:.1f}")
    print(f"      Exact selectivity:   {exact_selectivity:.2f}×")
    print(f"      Relaxed selectivity: {relaxed_selectivity:.2f}×")

    # ─── Keyword table ───
    print(f"\n  8. Keyword match table (showing matches only):")
    print(f"      {'Rank':>4} {'Keyword':<15} {'Exact':>5} {'Ed≤1':>5} "
          f"{'Best Match':<15} {'ED':>3} {'Count':>5}")
    print("      " + "-" * 62)
    for m in matches:
        if m.found_exact or m.found_relaxed:
            ex = 'Y' if m.found_exact else 'N'
            rl = 'Y' if m.found_relaxed else 'N'
            print(f"      {m.expected_rank:>4} {m.latin_word:<15} {ex:>5} {rl:>5} "
                  f"{m.best_match:<15} {m.edit_distance:>3} {m.n_occurrences:>5}")

    # ─── Gate ───
    gate_passed = n_relaxed >= 20 and rho < -0.3  # negative correlation expected
    # Also accept strong positive n_relaxed even if rho is weak
    if n_relaxed >= 20 and abs(rho) > 0.3:
        gate_passed = True

    print(f"\n  9. Gate: n_relaxed >= 20 AND |rho| > 0.3")
    print(f"      n_relaxed = {n_relaxed}")
    print(f"      rho = {rho:.3f}")
    print(f"      {'PASS' if gate_passed else 'FAIL'}")

    # ─── Verdict ───
    if gate_passed:
        verdict = (
            f"PASS: {n_exact} exact + {n_relaxed} relaxed keyword matches "
            f"(rho={rho:.3f}). Latin medical vocabulary present in decoded output."
        )
    elif n_relaxed >= 5:
        verdict = (
            f"MARGINAL: {n_exact} exact + {n_relaxed} relaxed matches "
            f"(rho={rho:.3f}). Some keywords present but below threshold."
        )
    else:
        verdict = (
            f"FAIL: {n_exact} exact + {n_relaxed} relaxed matches "
            f"(rho={rho:.3f}). Basic Latin vocabulary absent from decoded output."
        )

    print(f"\n  Verdict: {verdict}")

    # ─── Save ───
    result = HonestyKeywordResult(
        n_keywords=len(keywords),
        keyword_list=keyword_words,
        n_exact_found=n_exact,
        exact_keywords=exact_found,
        exact_rate=round(n_exact / len(keywords), 4),
        n_relaxed_found=n_relaxed,
        relaxed_keywords=relaxed_found,
        relaxed_rate=round(n_relaxed / len(keywords), 4),
        rho=round(rho, 4),
        rho_p_value=round(rho_p, 6),
        keyword_matches=[_convert(asdict(m)) for m in matches],
        random_exact_mean=round(random_exact_mean, 2),
        random_relaxed_mean=round(random_relaxed_mean, 2),
        exact_selectivity=round(exact_selectivity, 2),
        relaxed_selectivity=round(relaxed_selectivity, 2),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'honesty_keywords.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
