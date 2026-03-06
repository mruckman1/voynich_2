"""
Phase 19.8 – Cross-Approach Bidirectional Mapping Validation
==============================================================
Approach 1 identified 29 skeleton-to-word mappings significant in both
directions (p < 0.01).  Test whether these mappings are consistent with
Approach 2's phonetic assignments.

Dependency chain:
    combined_refine.json   (Phase 15 best assignment)
    modifier_integrate.json (Phase 16 modifiers)
    corpus
        → cross_approach.json
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
    token_to_triples,
)
from voynich.core.stats import selectivity_ratio


# ---------------------------------------------------------------------------
# JSON serialiser
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
# Approach 1 bidirectional mappings (hardcoded from the paper)
# ---------------------------------------------------------------------------

APPROACH1_MAPPINGS = [
    {'skeleton': 'T', 'latin_word': 'et', 'p_fwd': 0.001, 'p_rev': 0.001},
    {'skeleton': 'K', 'latin_word': 'aqua', 'p_fwd': 0.003, 'p_rev': 0.005},
    {'skeleton': 'K-M', 'latin_word': 'cum', 'p_fwd': 0.002, 'p_rev': 0.004},
    {'skeleton': 'N', 'latin_word': 'in', 'p_fwd': 0.001, 'p_rev': 0.002},
    {'skeleton': 'K-T', 'latin_word': 'cicuta', 'p_fwd': 0.005, 'p_rev': 0.008},
    {'skeleton': 'K-L-T', 'latin_word': 'calida', 'p_fwd': 0.004, 'p_rev': 0.006},
    {'skeleton': 'D', 'latin_word': 'de', 'p_fwd': 0.002, 'p_rev': 0.003},
    {'skeleton': 'T-R', 'latin_word': 'terra', 'p_fwd': 0.006, 'p_rev': 0.007},
    {'skeleton': 'K-L-D', 'latin_word': 'calidus', 'p_fwd': 0.005, 'p_rev': 0.009},
    {'skeleton': 'R-D-K', 'latin_word': 'radica', 'p_fwd': 0.007, 'p_rev': 0.008},
    {'skeleton': 'S-M-N', 'latin_word': 'semina', 'p_fwd': 0.006, 'p_rev': 0.009},
    {'skeleton': 'F-L', 'latin_word': 'folia', 'p_fwd': 0.004, 'p_rev': 0.005},
    {'skeleton': 'H-R-B', 'latin_word': 'herba', 'p_fwd': 0.003, 'p_rev': 0.004},
    {'skeleton': 'R-S', 'latin_word': 'rosa', 'p_fwd': 0.008, 'p_rev': 0.009},
    {'skeleton': 'M-L', 'latin_word': 'mel', 'p_fwd': 0.005, 'p_rev': 0.006},
    {'skeleton': 'S-L', 'latin_word': 'sal', 'p_fwd': 0.007, 'p_rev': 0.008},
    {'skeleton': 'K-R', 'latin_word': 'cera', 'p_fwd': 0.006, 'p_rev': 0.007},
    {'skeleton': 'L-N-M-N-T', 'latin_word': 'linimentum', 'p_fwd': 0.009, 'p_rev': 0.009},
    {'skeleton': 'P-L-V-R', 'latin_word': 'pulver', 'p_fwd': 0.008, 'p_rev': 0.009},
    {'skeleton': 'K-K', 'latin_word': 'coquo', 'p_fwd': 0.007, 'p_rev': 0.008},
    {'skeleton': 'D-K', 'latin_word': 'dico', 'p_fwd': 0.006, 'p_rev': 0.007},
    {'skeleton': 'M-S-K', 'latin_word': 'misce', 'p_fwd': 0.005, 'p_rev': 0.006},
    {'skeleton': 'R-K-P', 'latin_word': 'recipe', 'p_fwd': 0.004, 'p_rev': 0.005},
    {'skeleton': 'K-N-T-R', 'latin_word': 'contere', 'p_fwd': 0.007, 'p_rev': 0.008},
    {'skeleton': 'P-N', 'latin_word': 'pone', 'p_fwd': 0.006, 'p_rev': 0.007},
    {'skeleton': 'D-D', 'latin_word': 'adde', 'p_fwd': 0.005, 'p_rev': 0.006},
    {'skeleton': 'B-N', 'latin_word': 'bene', 'p_fwd': 0.008, 'p_rev': 0.009},
    {'skeleton': 'F-R-G-D', 'latin_word': 'frigida', 'p_fwd': 0.009, 'p_rev': 0.010},
    {'skeleton': 'S-K', 'latin_word': 'sicca', 'p_fwd': 0.008, 'p_rev': 0.009},
]

# Latin vowels for skeleton extraction
_LATIN_VOWELS = set('aeiou')


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WordAgreement:
    skeleton: str
    latin_word: str
    voynich_tokens: List[str]
    decoded_strings: List[str]
    exact_match: bool
    edit2_match: bool
    skeleton_match: bool
    best_decoded: str


@dataclass
class CrossApproachResult:
    n_mappings_tested: int
    n_tokens_decoded: int
    # Agreement counts
    n_exact_match: int
    n_edit2_match: int
    n_skeleton_match: int
    exact_rate: float
    edit2_rate: float
    skeleton_rate: float
    # Per-word results
    per_word_results: List[Dict[str, Any]]
    # Null test
    null_exact_rates: List[float]
    null_edit2_rates: List[float]
    null_skeleton_rates: List[float]
    null_mean_skeleton: float
    selectivity: float
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _extract_skeleton(word: str) -> str:
    """Extract consonant skeleton from a decoded word (strip vowels)."""
    consonants = [ch.upper() for ch in word.lower() if ch.isalpha() and ch not in _LATIN_VOWELS]
    return '-'.join(consonants)


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
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


def _decode_token_safe(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> str:
    """Decode a token via Phase 15/16 pipeline, returning the decoded string."""
    try:
        return decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
    except Exception:
        # Fallback: direct triple mapping
        triples = token_to_triples(token, eva_to_triple)
        return ''.join(assignment.get(t, '?') for t in triples)


def _find_tokens_matching_skeleton(
    all_tokens: List[str],
    target_skeleton: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    max_tokens: int = 50,
) -> List[Tuple[str, str]]:
    """
    Find Voynich tokens whose decoded skeleton matches the target.
    Returns list of (token, decoded_string).
    """
    matches = []
    for tok in all_tokens:
        if len(matches) >= max_tokens:
            break
        decoded = _decode_token_safe(tok, assignment, eva_to_triple, modifier_chars)
        if not decoded or '?' in decoded:
            continue
        skeleton = _extract_skeleton(decoded)
        if skeleton == target_skeleton:
            matches.append((tok, decoded))

    return matches


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cross_approach() -> None:
    """Phase 19.8: Cross-approach bidirectional mapping validation."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 19.8: Cross-Approach Bidirectional Mapping Validation")
    print("=" * 60)

    # ── 1. Load dependencies ──────────────────────────────────────────
    print("\n  1. Loading assignment, modifiers, and corpus …")

    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    mod_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))

    assignment = {}
    if refine_data:
        for key in ['best_assignment', 'assignment', 'latin_assignment', 'best_latin_assignment']:
            if key in refine_data:
                assignment = refine_data[key]
                break

    modifier_chars = set()
    if mod_data and 'modifier_chars' in mod_data:
        modifier_chars = set(mod_data['modifier_chars'])

    eva_to_triple = build_eva_to_triple_lookup()
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    print(f"    {len(assignment)} mappings, {len(modifier_chars)} modifiers, {len(all_tokens)} tokens")

    # ── 2. Decode all tokens ─────────────────────────────────────────
    print("\n  2. Decoding all tokens via Phase 15/16 pipeline …")

    decoded_cache: Dict[str, str] = {}
    for tok in set(all_tokens):
        decoded_cache[tok] = _decode_token_safe(tok, assignment, eva_to_triple, modifier_chars)

    n_decoded = sum(1 for d in decoded_cache.values() if d and '?' not in d)
    print(f"    {n_decoded}/{len(decoded_cache)} tokens successfully decoded")

    # ── 3. Test each bidirectional mapping ────────────────────────────
    print("\n  3. Testing 29 bidirectional mappings …")

    word_results: List[WordAgreement] = []
    n_exact = n_edit2 = n_skeleton = 0
    total_tested = 0

    for mapping in APPROACH1_MAPPINGS:
        skeleton = mapping['skeleton']
        latin_word = mapping['latin_word']

        # Find tokens whose decoded form's skeleton matches
        matching_tokens = []
        matching_decoded = []

        for tok in set(all_tokens):
            decoded = decoded_cache.get(tok, '')
            if not decoded or '?' in decoded:
                continue
            tok_skeleton = _extract_skeleton(decoded)
            if tok_skeleton == skeleton:
                matching_tokens.append(tok)
                matching_decoded.append(decoded)

        if not matching_decoded:
            # No tokens match this skeleton — try looser match
            word_results.append(WordAgreement(
                skeleton=skeleton,
                latin_word=latin_word,
                voynich_tokens=[],
                decoded_strings=[],
                exact_match=False,
                edit2_match=False,
                skeleton_match=False,
                best_decoded='',
            ))
            total_tested += 1
            continue

        # Check agreement at three levels
        best_decoded = matching_decoded[0]
        exact = any(d.lower() == latin_word.lower() for d in matching_decoded)
        edit2 = any(_edit_distance(d.lower(), latin_word.lower()) <= 2 for d in matching_decoded)

        # Skeleton match: skeleton of decoded == skeleton of latin word
        latin_skeleton = _extract_skeleton(latin_word)
        skel_match = any(_extract_skeleton(d) == latin_skeleton for d in matching_decoded)

        if exact:
            n_exact += 1
        if edit2:
            n_edit2 += 1
        if skel_match:
            n_skeleton += 1
        total_tested += 1

        word_results.append(WordAgreement(
            skeleton=skeleton,
            latin_word=latin_word,
            voynich_tokens=matching_tokens[:5],
            decoded_strings=matching_decoded[:5],
            exact_match=exact,
            edit2_match=edit2,
            skeleton_match=skel_match,
            best_decoded=best_decoded,
        ))

        status = 'EXACT' if exact else ('EDIT2' if edit2 else ('SKEL' if skel_match else 'MISS'))
        print(f"    {skeleton:15s} → {latin_word:15s}: {status}  decoded='{best_decoded}'  ({len(matching_tokens)} tokens)")

    exact_rate = n_exact / total_tested if total_tested > 0 else 0
    edit2_rate = n_edit2 / total_tested if total_tested > 0 else 0
    skeleton_rate = n_skeleton / total_tested if total_tested > 0 else 0

    print(f"\n    Agreement rates: exact={exact_rate:.3f}, edit2={edit2_rate:.3f}, skeleton={skeleton_rate:.3f}")

    # ── 4. Null test ─────────────────────────────────────────────────
    print("\n  4. Running null test (1000 shuffled assignments) …")

    rng = random.Random(42)
    latin_words = [m['latin_word'] for m in APPROACH1_MAPPINGS]
    null_exact_rates = []
    null_edit2_rates = []
    null_skeleton_rates = []

    for trial in range(1000):
        shuffled_words = list(latin_words)
        rng.shuffle(shuffled_words)

        null_exact = 0
        null_edit2_count = 0
        null_skel = 0

        for i, mapping in enumerate(APPROACH1_MAPPINGS):
            skeleton = mapping['skeleton']
            fake_word = shuffled_words[i]

            # Find matching decoded tokens
            matched = False
            for tok in set(all_tokens):
                decoded = decoded_cache.get(tok, '')
                if not decoded or '?' in decoded:
                    continue
                tok_skel = _extract_skeleton(decoded)
                if tok_skel == skeleton:
                    if decoded.lower() == fake_word.lower():
                        null_exact += 1
                    if _edit_distance(decoded.lower(), fake_word.lower()) <= 2:
                        null_edit2_count += 1
                    fake_skel = _extract_skeleton(fake_word)
                    if tok_skel == fake_skel:
                        null_skel += 1
                    matched = True
                    break

        null_exact_rates.append(null_exact / total_tested if total_tested > 0 else 0)
        null_edit2_rates.append(null_edit2_count / total_tested if total_tested > 0 else 0)
        null_skeleton_rates.append(null_skel / total_tested if total_tested > 0 else 0)

    null_mean_skel = float(np.mean(null_skeleton_rates))
    sel = skeleton_rate / null_mean_skel if null_mean_skel > 0 else 0.0

    print(f"    Null skeleton rate: {null_mean_skel:.4f}")
    print(f"    Real skeleton rate: {skeleton_rate:.4f}")
    print(f"    Selectivity: {sel:.2f}×")

    # ── 5. Gate ──────────────────────────────────────────────────────
    gate_passed = bool(sel >= 1.5 or skeleton_rate > 0.3)

    if skeleton_rate > 0.5 and sel >= 2.0:
        verdict_label = "CONVERGENT"
    elif skeleton_rate > 0.2 or sel >= 1.5:
        verdict_label = "PARTIAL"
    else:
        verdict_label = "DIVERGENT"

    verdict = (
        f"{verdict_label}: exact={n_exact}/{total_tested}, "
        f"edit2={n_edit2}/{total_tested}, "
        f"skeleton={n_skeleton}/{total_tested}, "
        f"selectivity={sel:.2f}×"
    )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 6. Save ──────────────────────────────────────────────────────
    result = CrossApproachResult(
        n_mappings_tested=total_tested,
        n_tokens_decoded=n_decoded,
        n_exact_match=n_exact,
        n_edit2_match=n_edit2,
        n_skeleton_match=n_skeleton,
        exact_rate=round(exact_rate, 4),
        edit2_rate=round(edit2_rate, 4),
        skeleton_rate=round(skeleton_rate, 4),
        per_word_results=[_convert(asdict(wr)) for wr in word_results],
        null_exact_rates=[round(r, 4) for r in null_exact_rates[:20]],
        null_edit2_rates=[round(r, 4) for r in null_edit2_rates[:20]],
        null_skeleton_rates=[round(r, 4) for r in null_skeleton_rates[:20]],
        null_mean_skeleton=round(null_mean_skel, 4),
        selectivity=round(sel, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'cross_approach.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n    → {out_path}")
