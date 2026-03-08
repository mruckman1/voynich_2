"""
Step 24.9 – EVA Ligature Hypothesis Test
=========================================
Test whether specific EVA character sequences (ch, sh, cth, ckh, cph, cfh)
are ligatures (single signs) rather than character sequences.

Dependency chain:
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        -> ligature_test.json (this step)
"""

import json
import math
import os
import random
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
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.core.stats import first_order_entropy


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
# Ligature candidates
# ---------------------------------------------------------------------------

LIGATURE_CANDIDATES = [
    ('ch', ['c', 'h']),
    ('sh', ['s', 'h']),
    ('cth', ['c', 't', 'h']),
    ('ckh', ['c', 'k', 'h']),
    ('cph', ['c', 'p', 'h']),
    ('cfh', ['c', 'f', 'h']),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LigatureCandidate:
    pair: str  # e.g., "ch"
    components: List[str]
    mi_score: float
    mi_z_score: float  # z-score relative to all pairs
    pair_frequency: int
    component_frequencies: List[int]
    frequency_ratio: float  # pair_freq / (comp1_freq * comp2_freq) * total
    positional_profile: Dict[str, float]  # p_initial, p_medial, p_final
    behaves_as_unit: bool  # positional profile consistent with single char


@dataclass
class RetokenizationProfile:
    ligatures_merged: List[str]
    original_inventory_size: int
    new_inventory_size: int
    original_mean_word_length: float
    new_mean_word_length: float
    original_entropy: float
    new_entropy: float
    latin_inventory_size: int
    latin_mean_word_length: float
    inventory_closer_to_latin: bool
    word_length_closer_to_latin: bool


@dataclass
class LigatureTestResult:
    timestamp: str
    n_candidates: int
    candidates: List[Dict]
    # MI analysis
    mean_mi_all_pairs: float
    std_mi_all_pairs: float
    n_pairs_analyzed: int
    # Strong ligatures (z > 2)
    strong_ligatures: List[str]
    n_strong: int
    # Re-tokenization
    retokenization: Dict
    # Verdict
    ligature_hypothesis_supported: bool
    recommended_merges: List[str]
    effective_inventory_size: int
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# MI computation
# ---------------------------------------------------------------------------

def _compute_adjacent_mi(
    corpus_tokens: List[str],
) -> Tuple[Dict[Tuple[str, str], float], Counter, Counter]:
    """Compute MI for all adjacent character pairs.

    Characters are split at the individual EVA letter level (no ligature
    grouping) so that we can measure co-occurrence statistics for the
    raw characters that make up the candidate ligatures.
    """
    pair_counts: Counter = Counter()
    char_counts: Counter = Counter()
    total_pairs = 0

    for token in corpus_tokens:
        # Split at individual character level (length-1 only)
        chars = list(token)
        for i in range(len(chars) - 1):
            pair_counts[(chars[i], chars[i + 1])] += 1
            total_pairs += 1
        for ch in chars:
            char_counts[ch] += 1

    total_chars = sum(char_counts.values())
    mi_scores: Dict[Tuple[str, str], float] = {}
    for (c1, c2), count in pair_counts.items():
        p_pair = count / total_pairs if total_pairs > 0 else 0
        p_c1 = char_counts[c1] / total_chars if total_chars > 0 else 0
        p_c2 = char_counts[c2] / total_chars if total_chars > 0 else 0
        if p_c1 > 0 and p_c2 > 0 and p_pair > 0:
            mi_scores[(c1, c2)] = math.log2(p_pair / (p_c1 * p_c2))

    return mi_scores, pair_counts, char_counts


# ---------------------------------------------------------------------------
# Positional profile
# ---------------------------------------------------------------------------

def _positional_profile(
    ligature: str,
    corpus_tokens: List[str],
) -> Dict[str, float]:
    """Compute P(initial), P(medial), P(final) for *ligature* treated as
    a single unit within tokens.

    We scan raw tokens for the substring and classify each occurrence by
    its position (initial = starts at index 0, final = ends at last index,
    medial = everything else).
    """
    initial = 0
    medial = 0
    final = 0

    for token in corpus_tokens:
        start = 0
        while True:
            idx = token.find(ligature, start)
            if idx == -1:
                break
            at_start = idx == 0
            at_end = idx + len(ligature) == len(token)
            if at_start and at_end:
                # token IS the ligature — count as both initial and final
                initial += 1
                final += 1
            elif at_start:
                initial += 1
            elif at_end:
                final += 1
            else:
                medial += 1
            start = idx + 1  # slide forward to find overlapping matches

    total = initial + medial + final
    if total == 0:
        return {'p_initial': 0.0, 'p_medial': 0.0, 'p_final': 0.0}
    return {
        'p_initial': initial / total,
        'p_medial': medial / total,
        'p_final': final / total,
    }


def _single_char_positional_profiles(
    corpus_tokens: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute positional profiles for all single EVA characters."""
    profiles: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])

    for token in corpus_tokens:
        chars = list(token)
        n = len(chars)
        for i, ch in enumerate(chars):
            if i == 0:
                profiles[ch][0] += 1  # initial
            elif i == n - 1:
                profiles[ch][2] += 1  # final
            else:
                profiles[ch][1] += 1  # medial

    result: Dict[str, Dict[str, float]] = {}
    for ch, counts in profiles.items():
        total = sum(counts)
        if total > 0:
            result[ch] = {
                'p_initial': counts[0] / total,
                'p_medial': counts[1] / total,
                'p_final': counts[2] / total,
            }
    return result


def _profile_is_unit_like(
    lig_profile: Dict[str, float],
    single_char_profiles: Dict[str, Dict[str, float]],
) -> bool:
    """Decide whether a ligature's positional profile is consistent with
    being a single character.

    Criterion: the ligature's profile has entropy >= median entropy of all
    single-character profiles.  A single-sign character should show up in
    multiple positions (not exclusively one position), so its positional
    entropy should be at least moderate.
    """
    def _entropy(prof: Dict[str, float]) -> float:
        vals = [v for v in prof.values() if v > 0]
        return -sum(p * math.log2(p) for p in vals) if vals else 0.0

    lig_ent = _entropy(lig_profile)

    char_entropies = sorted(_entropy(p) for p in single_char_profiles.values())
    if not char_entropies:
        return False
    median_ent = char_entropies[len(char_entropies) // 2]
    return lig_ent >= median_ent


# ---------------------------------------------------------------------------
# Re-tokenization
# ---------------------------------------------------------------------------

def _retokenize_corpus(
    corpus_tokens: List[str],
    merges: List[str],
) -> List[List[str]]:
    """Re-tokenize corpus treating each merge target as a single unit.

    Returns list of token-char-lists, where each element is either a merge
    target (e.g. 'ch') or a single EVA letter.
    """
    # Sort merges longest-first for greedy matching
    merges_sorted = sorted(merges, key=len, reverse=True)

    retokenized: List[List[str]] = []
    for token in corpus_tokens:
        chars: List[str] = []
        i = 0
        while i < len(token):
            matched = False
            for m in merges_sorted:
                if token[i:i + len(m)] == m:
                    chars.append(m)
                    i += len(m)
                    matched = True
                    break
            if not matched:
                chars.append(token[i])
                i += 1
        retokenized.append(chars)
    return retokenized


def _compute_retokenization_profile(
    corpus_tokens: List[str],
    merges: List[str],
    latin_tokens: List[str],
) -> RetokenizationProfile:
    """Build a RetokenizationProfile comparing original vs re-tokenized
    corpus vs Latin reference."""

    # --- Original stats (character-level = individual letters) ---
    orig_chars_per_token: List[int] = []
    orig_inventory: Set[str] = set()
    orig_char_stream: List[str] = []
    for token in corpus_tokens:
        chars = list(token)
        orig_chars_per_token.append(len(chars))
        orig_inventory.update(chars)
        orig_char_stream.extend(chars)

    orig_inventory_size = len(orig_inventory)
    orig_mean_wl = (
        sum(orig_chars_per_token) / len(orig_chars_per_token)
        if orig_chars_per_token
        else 0.0
    )
    orig_entropy = first_order_entropy(''.join(orig_char_stream))

    # --- New stats (after merging ligatures) ---
    retok = _retokenize_corpus(corpus_tokens, merges)
    new_chars_per_token: List[int] = []
    new_inventory: Set[str] = set()
    new_char_stream: List[str] = []
    for chars in retok:
        new_chars_per_token.append(len(chars))
        new_inventory.update(chars)
        new_char_stream.extend(chars)

    new_inventory_size = len(new_inventory)
    new_mean_wl = (
        sum(new_chars_per_token) / len(new_chars_per_token)
        if new_chars_per_token
        else 0.0
    )
    # For entropy, join using a separator that won't collide, then compute.
    # We use the merged units as single "characters" by mapping to unique
    # single-char surrogates.
    surrogates: Dict[str, str] = {}
    next_code = 0x0100  # start above ASCII
    for unit in sorted(new_inventory):
        if len(unit) == 1:
            surrogates[unit] = unit
        else:
            surrogates[unit] = chr(next_code)
            next_code += 1
    surrogate_stream = ''.join(surrogates.get(u, u) for u in new_char_stream)
    new_entropy = first_order_entropy(surrogate_stream)

    # --- Latin reference stats ---
    latin_chars: Set[str] = set()
    latin_lens: List[int] = []
    for tok in latin_tokens:
        latin_chars.update(tok.lower())
        latin_lens.append(len(tok))
    latin_inventory_size = len(latin_chars) if latin_chars else 23
    latin_mean_wl = (
        sum(latin_lens) / len(latin_lens) if latin_lens else 5.5
    )

    inv_closer = abs(new_inventory_size - latin_inventory_size) < abs(
        orig_inventory_size - latin_inventory_size
    )
    wl_closer = abs(new_mean_wl - latin_mean_wl) < abs(
        orig_mean_wl - latin_mean_wl
    )

    return RetokenizationProfile(
        ligatures_merged=list(merges),
        original_inventory_size=orig_inventory_size,
        new_inventory_size=new_inventory_size,
        original_mean_word_length=round(orig_mean_wl, 4),
        new_mean_word_length=round(new_mean_wl, 4),
        original_entropy=round(orig_entropy, 4),
        new_entropy=round(new_entropy, 4),
        latin_inventory_size=latin_inventory_size,
        latin_mean_word_length=round(latin_mean_wl, 4),
        inventory_closer_to_latin=inv_closer,
        word_length_closer_to_latin=wl_closer,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ligature_test() -> None:
    """Step 24.9 — EVA Ligature Hypothesis Test.

    Tests whether ch, sh, cth, ckh, cph, cfh are ligatures (single signs)
    rather than sequences of individual characters, using mutual information,
    positional profiling, and re-tokenization comparison to Latin.

    Saves results to ``results/ligature_test.json``.
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load corpus
    # ------------------------------------------------------------------
    print("Step 24.9.1: Loading Voynich corpus ...")
    corpus = load_corpus(verbose=False)
    full_text = corpus.get_text()
    corpus_tokens = [t for t in full_text.split() if t]
    print(f"  Loaded {len(corpus_tokens):,} tokens")

    # ------------------------------------------------------------------
    # 2. Load Latin reference for comparison
    # ------------------------------------------------------------------
    print("Step 24.9.2: Loading Latin reference corpus ...")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        latin_tokens = ref_corpus.get_combined_tokens('latin')
    except Exception:
        # Fallback: empty list — we will use defaults in the profile
        latin_tokens = []
    print(f"  Latin reference: {len(latin_tokens):,} tokens")

    # ------------------------------------------------------------------
    # 3. Compute MI for all adjacent character pairs
    # ------------------------------------------------------------------
    print("Step 24.9.3: Computing mutual information for adjacent pairs ...")
    mi_scores, pair_counts, char_counts = _compute_adjacent_mi(corpus_tokens)
    all_mi_values = list(mi_scores.values())

    if all_mi_values:
        mean_mi = sum(all_mi_values) / len(all_mi_values)
        variance = sum((v - mean_mi) ** 2 for v in all_mi_values) / len(
            all_mi_values
        )
        std_mi = math.sqrt(variance) if variance > 0 else 1e-9
    else:
        mean_mi = 0.0
        std_mi = 1e-9

    print(f"  Analyzed {len(mi_scores):,} unique pairs")
    print(f"  Mean MI = {mean_mi:.4f}, Std MI = {std_mi:.4f}")

    # ------------------------------------------------------------------
    # 4. Compute positional profiles for single chars
    # ------------------------------------------------------------------
    print("Step 24.9.4: Computing positional profiles ...")
    single_char_profiles = _single_char_positional_profiles(corpus_tokens)

    # ------------------------------------------------------------------
    # 5. Evaluate each ligature candidate
    # ------------------------------------------------------------------
    print("Step 24.9.5: Evaluating ligature candidates ...")
    candidates: List[LigatureCandidate] = []

    for lig_str, components in LIGATURE_CANDIDATES:
        # --- MI score ---
        # For multi-component ligatures (e.g. cth = c+t+h), we compute
        # the average MI across consecutive component pairs.
        component_pair_mis: List[float] = []
        for j in range(len(components) - 1):
            key = (components[j], components[j + 1])
            if key in mi_scores:
                component_pair_mis.append(mi_scores[key])
        mi_val = (
            sum(component_pair_mis) / len(component_pair_mis)
            if component_pair_mis
            else 0.0
        )
        z_score = (mi_val - mean_mi) / std_mi if std_mi > 0 else 0.0

        # --- Frequency ---
        pair_freq = 0
        for token in corpus_tokens:
            start = 0
            while True:
                idx = token.find(lig_str, start)
                if idx == -1:
                    break
                pair_freq += 1
                start = idx + 1

        comp_freqs = [char_counts.get(c, 0) for c in components]
        total_chars = sum(char_counts.values())
        # Frequency ratio: observed / expected under independence
        if all(f > 0 for f in comp_freqs) and total_chars > 0:
            expected = 1.0
            for f in comp_freqs:
                expected *= f / total_chars
            expected *= total_chars  # scale to count
            freq_ratio = pair_freq / expected if expected > 0 else 0.0
        else:
            freq_ratio = 0.0

        # --- Positional profile ---
        pos_profile = _positional_profile(lig_str, corpus_tokens)
        behaves = _profile_is_unit_like(pos_profile, single_char_profiles)

        cand = LigatureCandidate(
            pair=lig_str,
            components=components,
            mi_score=round(mi_val, 4),
            mi_z_score=round(z_score, 4),
            pair_frequency=pair_freq,
            component_frequencies=comp_freqs,
            frequency_ratio=round(freq_ratio, 4),
            positional_profile={k: round(v, 4) for k, v in pos_profile.items()},
            behaves_as_unit=behaves,
        )
        candidates.append(cand)
        print(
            f"    {lig_str:4s}  MI={mi_val:+.3f}  z={z_score:+.2f}  "
            f"freq={pair_freq:6d}  ratio={freq_ratio:.2f}  "
            f"unit={'Y' if behaves else 'N'}"
        )

    # ------------------------------------------------------------------
    # 6. Identify strong ligatures (z > 2)
    # ------------------------------------------------------------------
    print("Step 24.9.6: Identifying strong ligatures (MI z-score > 2) ...")
    strong = [c.pair for c in candidates if c.mi_z_score > 2.0]
    print(f"  Strong ligatures: {strong if strong else '(none)'}")

    # ------------------------------------------------------------------
    # 7. Re-tokenization test
    # ------------------------------------------------------------------
    print("Step 24.9.7: Running re-tokenization test ...")
    # Merge all candidates that have z > 0 (any positive MI signal)
    merges_to_test = [c.pair for c in candidates if c.mi_z_score > 0]
    if not merges_to_test:
        # Fall back to all candidates if none have positive z
        merges_to_test = [c.pair for c in candidates]

    retok_profile = _compute_retokenization_profile(
        corpus_tokens, merges_to_test, latin_tokens
    )

    print(f"  Merges tested: {retok_profile.ligatures_merged}")
    print(
        f"  Inventory: {retok_profile.original_inventory_size} -> "
        f"{retok_profile.new_inventory_size}  "
        f"(Latin: {retok_profile.latin_inventory_size})"
    )
    print(
        f"  Mean word length: {retok_profile.original_mean_word_length:.2f} -> "
        f"{retok_profile.new_mean_word_length:.2f}  "
        f"(Latin: {retok_profile.latin_mean_word_length:.2f})"
    )
    print(
        f"  Entropy: {retok_profile.original_entropy:.4f} -> "
        f"{retok_profile.new_entropy:.4f}"
    )
    print(
        f"  Inventory closer to Latin: {retok_profile.inventory_closer_to_latin}"
    )
    print(
        f"  Word length closer to Latin: "
        f"{retok_profile.word_length_closer_to_latin}"
    )

    # ------------------------------------------------------------------
    # 8. Verdict
    # ------------------------------------------------------------------
    print("Step 24.9.8: Computing verdict ...")
    # Hypothesis is supported if at least 2 candidates are strong AND
    # re-tokenization moves at least one metric closer to Latin.
    n_strong = len(strong)
    retok_improvement = (
        retok_profile.inventory_closer_to_latin
        or retok_profile.word_length_closer_to_latin
    )
    hypothesis_supported = n_strong >= 2 and retok_improvement

    # Recommended merges: strong MI AND behaves as positional unit
    recommended = [
        c.pair
        for c in candidates
        if c.mi_z_score > 2.0 and c.behaves_as_unit
    ]
    # If none meet both criteria, recommend strong MI alone
    if not recommended:
        recommended = strong

    effective_inventory = retok_profile.new_inventory_size

    if hypothesis_supported:
        verdict = (
            f"SUPPORTED: {n_strong} ligature(s) with MI z > 2 "
            f"({', '.join(strong)}). Re-tokenization improves fit to Latin. "
            f"Effective inventory = {effective_inventory}."
        )
    else:
        reasons = []
        if n_strong < 2:
            reasons.append(f"only {n_strong} strong candidate(s)")
        if not retok_improvement:
            reasons.append("re-tokenization does not improve Latin fit")
        verdict = (
            f"NOT SUPPORTED: {'; '.join(reasons)}. "
            f"Candidates may still function as digraphs but lack strong "
            f"statistical evidence for single-sign status."
        )

    print(f"  Verdict: {verdict}")

    # ------------------------------------------------------------------
    # 9. Assemble and save results
    # ------------------------------------------------------------------
    runtime = time.time() - t0

    result = LigatureTestResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_candidates=len(candidates),
        candidates=[_convert(c) for c in candidates],
        mean_mi_all_pairs=round(mean_mi, 6),
        std_mi_all_pairs=round(std_mi, 6),
        n_pairs_analyzed=len(mi_scores),
        strong_ligatures=strong,
        n_strong=n_strong,
        retokenization=_convert(retok_profile),
        ligature_hypothesis_supported=hypothesis_supported,
        recommended_merges=recommended,
        effective_inventory_size=effective_inventory,
        verdict=verdict,
        runtime_seconds=round(runtime, 2),
    )

    out_path = _results_dir() / 'ligature_test.json'
    with open(out_path, 'w') as fh:
        json.dump(_convert(result), fh, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Runtime: {runtime:.1f}s")
