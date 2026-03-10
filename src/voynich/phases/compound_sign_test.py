"""
Phase 31.6: Compound Sign Hypothesis Test
============================================
Test whether Voynich "words" are compound signs where the prefix encodes a
semantic category, the root encodes the pronunciation, and the suffix encodes
grammar — like Egyptian hieroglyphs or Maya signs.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    morpheme_grid.json         (Phase 4 decomposition)
        → compound_sign_test.json  (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

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
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes,
    KNOWN_PREFIXES,
    KNOWN_SUFFIXES,
)
from voynich.phases.null_corpus import _reconstruct_modifier_rules
from voynich.phases.signal_isolation import _decode_corpus_r3


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
class DecompStats:
    """Statistics about morpheme decomposition."""
    n_tokens: int
    n_with_prefix: int
    n_with_suffix: int
    n_with_both: int
    n_stem_only: int
    prefix_distribution: Dict[str, int]
    suffix_distribution: Dict[str, int]
    mean_stem_length: float


@dataclass
class PrefixSemanticProfile:
    """Semantic profile for tokens with a given prefix."""
    prefix: str
    n_tokens: int
    top_decoded_words: List[Tuple[str, int]]
    dict_hit_rate: float


@dataclass
class SuffixGrammaticalProfile:
    """Grammatical profile for tokens with a given suffix."""
    suffix: str
    n_tokens: int
    top_decoded_words: List[Tuple[str, int]]
    dict_hit_rate: float
    latin_ending_matches: Dict[str, int]


@dataclass
class CompoundSignResult:
    """Full Step 31.6 output."""
    # Decomposition stats
    decomp_stats: Dict
    # Root-only decoding
    full_dict_hit: float
    root_dict_hit: float
    root_delta: float
    # Prefix semantic classification
    prefix_profiles: List[Dict]
    prefix_chi_sq: float
    prefix_p_value: float
    # Suffix grammatical classification
    suffix_profiles: List[Dict]
    suffix_chi_sq: float
    suffix_p_value: float
    # Mixed decode
    mixed_dict_hit: float
    mixed_delta: float
    # Verdict
    root_improves: bool
    prefix_semantic: bool
    suffix_grammatical: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _decompose_all_tokens(
    all_tokens: List[str],
) -> Tuple[List[Tuple[str, str, str, str]], DecompStats]:
    """Decompose all tokens into (token, prefix, root, suffix)."""
    decomps = []
    prefix_counts = Counter()
    suffix_counts = Counter()
    stem_lengths = []
    n_with_prefix = 0
    n_with_suffix = 0
    n_with_both = 0
    n_stem_only = 0

    for token in all_tokens:
        d = decompose_token_morphemes(token)
        decomps.append((token, d.prefix, d.stem, d.suffix))
        if d.prefix:
            prefix_counts[d.prefix] += 1
            n_with_prefix += 1
        if d.suffix:
            suffix_counts[d.suffix] += 1
            n_with_suffix += 1
        if d.prefix and d.suffix:
            n_with_both += 1
        if not d.prefix and not d.suffix:
            n_stem_only += 1
        if d.stem:
            stem_lengths.append(len(d.stem))

    stats = DecompStats(
        n_tokens=len(all_tokens),
        n_with_prefix=n_with_prefix,
        n_with_suffix=n_with_suffix,
        n_with_both=n_with_both,
        n_stem_only=n_stem_only,
        prefix_distribution=dict(prefix_counts.most_common()),
        suffix_distribution=dict(suffix_counts.most_common()),
        mean_stem_length=round(sum(stem_lengths) / max(len(stem_lengths), 1), 2),
    )

    return decomps, stats


def _root_only_decode(
    decomps: List[Tuple[str, str, str, str]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[float, float]:
    """Decode only roots (stripping prefixes and suffixes). Return (root_dict_hit, full_dict_hit)."""
    roots = [stem for _, prefix, stem, suffix in decomps]
    full_tokens = [token for token, _, _, _ in decomps]
    n = len(decomps)

    # Full token decode
    full_decoded = _decode_corpus_r3(
        full_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    full_hits = sum(1 for w in full_decoded if w in ref_word_set)
    full_dict_hit = full_hits / n

    # Root-only decode
    root_decoded = _decode_corpus_r3(
        roots, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    root_hits = sum(1 for w in root_decoded if w in ref_word_set)
    root_dict_hit = root_hits / n

    return root_dict_hit, full_dict_hit


def _prefix_semantic_test(
    decomps: List[Tuple[str, str, str, str]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[List[PrefixSemanticProfile], float, float]:
    """Group by prefix, decode roots, compute chi-squared on decoded-word distributions."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for token, prefix, stem, suffix in decomps:
        key = prefix if prefix else 'none'
        groups[key].append(stem)

    profiles = []
    word_distributions: Dict[str, Counter] = {}

    for prefix in sorted(groups.keys()):
        stems = groups[prefix]
        decoded = _decode_corpus_r3(
            stems, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        word_counts = Counter(w for w in decoded if w in ref_word_set)
        hits = sum(word_counts.values())
        dict_hit = hits / max(len(stems), 1)
        word_distributions[prefix] = word_counts

        profiles.append(PrefixSemanticProfile(
            prefix=prefix,
            n_tokens=len(stems),
            top_decoded_words=word_counts.most_common(10),
            dict_hit_rate=round(dict_hit, 4),
        ))

    # Chi-squared test
    all_words = Counter()
    for wc in word_distributions.values():
        all_words.update(wc)
    top_words = [w for w, _ in all_words.most_common(25)]

    if len(top_words) < 2 or len(word_distributions) < 2:
        return profiles, 0.0, 1.0

    group_names = sorted(word_distributions.keys())
    observed = np.zeros((len(group_names), len(top_words)))
    for i, gn in enumerate(group_names):
        for j, w in enumerate(top_words):
            observed[i, j] = word_distributions[gn].get(w, 0)

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    if total == 0:
        return profiles, 0.0, 1.0

    expected = row_sums * col_sums / total
    mask = expected > 0
    chi_sq = float(np.sum(((observed[mask] - expected[mask]) ** 2) / expected[mask]))
    df = (len(group_names) - 1) * (len(top_words) - 1)

    if df > 0:
        z = (chi_sq - df) / max((2 * df) ** 0.5, 1)
        if z > 3:
            p_value = 0.001
        elif z > 2:
            p_value = 0.01
        elif z > 1.5:
            p_value = 0.05
        else:
            p_value = min(1.0, max(0.1, 0.5 - z * 0.1))
    else:
        p_value = 1.0

    return profiles, chi_sq, p_value


def _suffix_grammatical_test(
    decomps: List[Tuple[str, str, str, str]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[List[SuffixGrammaticalProfile], float, float]:
    """Group by suffix, decode roots, check if different suffixes produce different endings."""
    # Latin inflection endings to check
    latin_endings = {
        'nominative_sg': {'a', 'us', 'um', 'is', 'es'},
        'genitive_sg': {'ae', 'i', 'is'},
        'accusative_sg': {'am', 'um', 'em'},
        'ablative_sg': {'a', 'o', 'e'},
        'verb_inf': {'re', 'ri'},
        'verb_imp': {'a', 'e', 'i'},
    }

    groups: Dict[str, List[str]] = defaultdict(list)
    for token, prefix, stem, suffix in decomps:
        key = suffix if suffix else 'none'
        groups[key].append(stem)

    profiles = []
    word_distributions: Dict[str, Counter] = {}

    for suffix in sorted(groups.keys()):
        stems = groups[suffix]
        decoded = _decode_corpus_r3(
            stems, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        word_counts = Counter(w for w in decoded if w in ref_word_set)
        hits = sum(word_counts.values())
        dict_hit = hits / max(len(stems), 1)
        word_distributions[suffix] = word_counts

        # Check Latin ending distribution
        ending_counts: Dict[str, int] = Counter()
        for word, count in word_counts.items():
            for category, endings in latin_endings.items():
                for end in endings:
                    if word.endswith(end):
                        ending_counts[category] += count
                        break

        profiles.append(SuffixGrammaticalProfile(
            suffix=suffix,
            n_tokens=len(stems),
            top_decoded_words=word_counts.most_common(10),
            dict_hit_rate=round(dict_hit, 4),
            latin_ending_matches=dict(ending_counts),
        ))

    # Chi-squared on ending distributions
    all_endings = Counter()
    ending_distributions: Dict[str, Counter] = {}
    for sp in profiles:
        ending_distributions[sp.suffix] = Counter(sp.latin_ending_matches)
        all_endings.update(sp.latin_ending_matches)

    categories = sorted(all_endings.keys())
    suffixes = sorted(ending_distributions.keys())

    if len(categories) < 2 or len(suffixes) < 2:
        return profiles, 0.0, 1.0

    observed = np.zeros((len(suffixes), len(categories)))
    for i, s in enumerate(suffixes):
        for j, c in enumerate(categories):
            observed[i, j] = ending_distributions[s].get(c, 0)

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    if total == 0:
        return profiles, 0.0, 1.0

    expected = row_sums * col_sums / total
    mask = expected > 0
    chi_sq = float(np.sum(((observed[mask] - expected[mask]) ** 2) / expected[mask]))
    df = (len(suffixes) - 1) * (len(categories) - 1)

    if df > 0:
        z = (chi_sq - df) / max((2 * df) ** 0.5, 1)
        if z > 3:
            p_value = 0.001
        elif z > 2:
            p_value = 0.01
        elif z > 1.5:
            p_value = 0.05
        else:
            p_value = min(1.0, max(0.1, 0.5 - z * 0.1))
    else:
        p_value = 1.0

    return profiles, chi_sq, p_value


def _mixed_decode(
    decomps: List[Tuple[str, str, str, str]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> float:
    """Mixed decode: roots phonetically, try appending suffix-derived Latin endings."""
    # Simple suffix → Latin ending mapping (from Phase 19.3 patterns)
    suffix_ending_map = {
        'dy': 'a', 'y': 'i', 'ey': 'e', 'aiin': 'um',
        'ol': 'is', 'al': 'ae', 'in': 'em', 'am': 'am',
        'iin': 'en', 'm': 'um', 'aiiin': 'ium', 'iiin': 'ium',
        'an': 'an', 'n': 'n',
    }

    roots = [stem for _, prefix, stem, suffix in decomps]
    suffixes = [suffix for _, prefix, stem, suffix in decomps]

    # Decode roots
    root_decoded = _decode_corpus_r3(
        roots, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    # Try appending Latin endings
    n = len(decomps)
    hits = 0
    for i in range(n):
        root_word = root_decoded[i]
        suffix = suffixes[i]

        # Try root alone
        if root_word in ref_word_set:
            hits += 1
            continue

        # Try root + Latin ending
        ending = suffix_ending_map.get(suffix, '')
        if ending:
            combined = root_word + ending
            if combined in ref_word_set:
                hits += 1
                continue
            # Try trimming last char of root and adding ending
            if len(root_word) > 2:
                trimmed = root_word[:-1] + ending
                if trimmed in ref_word_set:
                    hits += 1
                    continue

    return hits / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_compound_sign() -> None:
    """Step 31.6: Test compound sign hypothesis."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 31.6: Compound Sign Hypothesis")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs...")

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    print(f"     {len(all_tokens)} tokens, {len(ref_word_set)} reference words")

    # ── 2. Decompose all tokens ──
    print("\n  2. Decomposing all tokens...")
    decomps, stats = _decompose_all_tokens(all_tokens)
    print(f"     With prefix: {stats.n_with_prefix} ({stats.n_with_prefix / stats.n_tokens:.1%})")
    print(f"     With suffix: {stats.n_with_suffix} ({stats.n_with_suffix / stats.n_tokens:.1%})")
    print(f"     With both: {stats.n_with_both} ({stats.n_with_both / stats.n_tokens:.1%})")
    print(f"     Stem only: {stats.n_stem_only} ({stats.n_stem_only / stats.n_tokens:.1%})")
    print(f"     Mean stem length: {stats.mean_stem_length} chars")

    # ── 3. Root-only decoding ──
    print("\n  3. Root-only vs full-token decoding...")
    root_dict_hit, full_dict_hit = _root_only_decode(
        decomps, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    root_delta = root_dict_hit - full_dict_hit
    print(f"     Full-token dict_hit: {full_dict_hit:.4f}")
    print(f"     Root-only dict_hit:  {root_dict_hit:.4f}")
    print(f"     Delta: {root_delta:+.4f}")

    # ── 4. Prefix semantic classification ──
    print("\n  4. Prefix semantic classification...")
    prefix_profiles, prefix_chi_sq, prefix_p = _prefix_semantic_test(
        decomps, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    for pp in prefix_profiles:
        top_3 = pp.top_decoded_words[:3]
        top_str = ', '.join(f'{w}({c})' for w, c in top_3)
        print(f"     {pp.prefix:5s}: {pp.n_tokens:6d} tokens, "
              f"dict_hit={pp.dict_hit_rate:.3f}, top: {top_str}")
    print(f"     Chi-squared: {prefix_chi_sq:.1f}, p≈{prefix_p:.4f}")

    # ── 5. Suffix grammatical classification ──
    print("\n  5. Suffix grammatical classification...")
    suffix_profiles, suffix_chi_sq, suffix_p = _suffix_grammatical_test(
        decomps, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    for sp in suffix_profiles[:10]:  # Top 10
        top_3 = sp.top_decoded_words[:3]
        top_str = ', '.join(f'{w}({c})' for w, c in top_3)
        endings_str = ', '.join(f'{k}={v}' for k, v in
                                sorted(sp.latin_ending_matches.items(), key=lambda x: -x[1])[:3])
        print(f"     {sp.suffix:5s}: {sp.n_tokens:6d} tokens, "
              f"dict_hit={sp.dict_hit_rate:.3f}, endings: {endings_str}")
    print(f"     Chi-squared: {suffix_chi_sq:.1f}, p≈{suffix_p:.4f}")

    # ── 6. Mixed decoding ──
    print("\n  6. Mixed decoding (root + suffix → Latin ending)...")
    mixed_dict_hit = _mixed_decode(
        decomps, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    mixed_delta = mixed_dict_hit - full_dict_hit
    print(f"     Mixed dict_hit: {mixed_dict_hit:.4f}")
    print(f"     Delta vs full: {mixed_delta:+.4f}")

    # ── 7. Verdict ──
    root_improves = root_delta > 0.01
    prefix_semantic = prefix_p < 0.05
    suffix_grammatical = suffix_p < 0.05

    if root_improves and (prefix_semantic or suffix_grammatical):
        verdict = "COMPOUND_SIGN_SUPPORTED"
    elif root_improves:
        verdict = "COMPOUND_SIGN_POSSIBLE"
    elif prefix_semantic or suffix_grammatical:
        verdict = "AFFIX_SEMANTIC_ONLY"
    else:
        verdict = "COMPOUND_SIGN_UNSUPPORTED"

    print(f"\n  Verdict: {verdict}")
    print(f"     Root-only improves: {root_improves} (Δ={root_delta:+.4f})")
    print(f"     Prefix semantic: {prefix_semantic} (p={prefix_p:.4f})")
    print(f"     Suffix grammatical: {suffix_grammatical} (p={suffix_p:.4f})")

    # ── 8. Save ──
    result = CompoundSignResult(
        decomp_stats=_convert(asdict(stats)),
        full_dict_hit=round(full_dict_hit, 4),
        root_dict_hit=round(root_dict_hit, 4),
        root_delta=round(root_delta, 4),
        prefix_profiles=[_convert(asdict(pp)) for pp in prefix_profiles],
        prefix_chi_sq=round(prefix_chi_sq, 2),
        prefix_p_value=round(prefix_p, 6),
        suffix_profiles=[_convert(asdict(sp)) for sp in suffix_profiles],
        suffix_chi_sq=round(suffix_chi_sq, 2),
        suffix_p_value=round(suffix_p, 6),
        mixed_dict_hit=round(mixed_dict_hit, 4),
        mixed_delta=round(mixed_delta, 4),
        root_improves=root_improves,
        prefix_semantic=prefix_semantic,
        suffix_grammatical=suffix_grammatical,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'compound_sign_test.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")
