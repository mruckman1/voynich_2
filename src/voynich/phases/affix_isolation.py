"""
Phase 19.3 – Affix Layer Isolation and Independent Decoding
============================================================
Both approaches found ~4 prefixes and ~14 suffixes forming a closed,
low-entropy grammatical layer.  Decode this layer independently of stems
by matching Voynich affix distributional patterns to Latin inflectional
patterns.

Dependency chain:
    morpheme_grid.json    (Phase 4.5B)
    combined_refine.json  (Phase 15 best assignment)
    modifier_integrate.json (Phase 16 modifiers)
    reference corpus
        → affix_isolation.json
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
    token_to_triples,
)
from voynich.core.reference import (
    LATIN_DECLENSION_SUFFIXES,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.core.stats import (
    jensen_shannon_divergence,
    selectivity_ratio,
)


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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AffixProfile:
    affix: str
    affix_type: str  # 'prefix' or 'suffix'
    frequency: int
    position_entropy: float
    section_distribution: Dict[str, float]
    co_occurring_affixes: List[str]
    decoded_phonetic: str


@dataclass
class AffixIsolationResult:
    n_tokens: int
    n_with_prefix: int
    n_with_suffix: int
    n_stems: int
    prefix_inventory: Dict[str, int]
    suffix_inventory: Dict[str, int]
    # Affix profiles
    affix_profiles: List[Dict[str, Any]]
    # Hungarian assignment
    best_mapping: Dict[str, str]  # Voynich affix → Latin ending
    mapping_score: float
    # Paradigm consistency
    n_paradigm_violations: int
    n_paradigm_total: int
    paradigm_consistency: float
    # Null test
    null_scores: List[float]
    selectivity: float
    # Cross-validation
    odd_folio_score: float
    even_folio_score: float
    cv_ratio: float
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


def _strip_affix(eva_chars: List[str], prefixes: List[str],
                 suffixes: List[str]) -> Tuple[str, str, List[str]]:
    """
    Strip prefix and suffix from an EVA-char sequence.
    Returns (prefix, suffix, stem_chars).
    """
    prefix_found = ''
    suffix_found = ''
    chars = list(eva_chars)

    # Check prefixes (first char)
    if chars and chars[0] in prefixes:
        prefix_found = chars[0]
        chars = chars[1:]

    # Check suffixes (match longest suffix from end)
    for suf_len in range(min(3, len(chars)), 0, -1):
        candidate = ''.join(chars[-suf_len:])
        if candidate in suffixes:
            suffix_found = candidate
            chars = chars[:-suf_len]
            break
        # Also check individual EVA chars as suffix
        if suf_len == 1 and chars[-1] in suffixes:
            suffix_found = chars[-1]
            chars = chars[:-1]
            break

    return prefix_found, suffix_found, chars


def _entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values() if c > 0
    )


def _build_latin_ending_profiles(ref_corpus) -> Dict[str, Dict[str, Any]]:
    """
    Build distributional profiles for Latin inflectional endings.
    """
    profiles = {}

    # Gather all endings from LATIN_DECLENSION_SUFFIXES
    all_endings = set()
    paradigm_map = {}  # ending → paradigm class
    for paradigm, endings in LATIN_DECLENSION_SUFFIXES.items():
        for ending in endings:
            all_endings.add(ending)
            if ending not in paradigm_map:
                paradigm_map[ending] = []
            paradigm_map[ending].append(paradigm)

    # Get Latin word frequencies
    latin_tokens = ref_corpus.get_combined_tokens('latin')
    if not latin_tokens:
        return {}

    ending_freq = Counter()
    for word in latin_tokens:
        word = word.lower().strip()
        for ending in sorted(all_endings, key=len, reverse=True):
            if word.endswith(ending) and len(word) > len(ending):
                ending_freq[ending] += 1
                break

    # Build profiles
    total_matched = sum(ending_freq.values())
    for ending in all_endings:
        freq = ending_freq.get(ending, 0)
        profiles[ending] = {
            'frequency': freq,
            'frequency_rank': 0,  # Will be filled below
            'paradigm_classes': paradigm_map.get(ending, []),
            'relative_frequency': freq / total_matched if total_matched > 0 else 0,
        }

    # Assign frequency ranks
    sorted_endings = sorted(profiles.keys(),
                            key=lambda e: profiles[e]['frequency'], reverse=True)
    for rank, ending in enumerate(sorted_endings):
        profiles[ending]['frequency_rank'] = rank

    return profiles


def _build_compatibility_matrix(
    voynich_profiles: Dict[str, Dict],
    latin_profiles: Dict[str, Dict],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build compatibility matrix between Voynich affixes and Latin endings.
    Score based on frequency-rank proximity and paradigm alignment.
    """
    v_keys = sorted(voynich_profiles.keys())
    l_keys = sorted(latin_profiles.keys())

    matrix = np.zeros((len(v_keys), len(l_keys)))

    for i, vk in enumerate(v_keys):
        vp = voynich_profiles[vk]
        for j, lk in enumerate(l_keys):
            lp = latin_profiles[lk]

            # Frequency rank proximity (0-1, higher = better match)
            v_rank = vp.get('frequency_rank', 0)
            l_rank = lp.get('frequency_rank', 0)
            max_rank = max(len(v_keys), len(l_keys))
            rank_score = 1.0 - abs(v_rank - l_rank) / max_rank if max_rank > 0 else 0

            # Type alignment: prefix should match prefix-like endings, suffix → suffix
            type_score = 0.5
            if vp.get('affix_type') == 'suffix' and lp.get('paradigm_classes'):
                type_score = 0.8  # Suffixes align with inflectional endings

            # Frequency magnitude similarity
            v_freq = vp.get('relative_frequency', 0)
            l_freq = lp.get('relative_frequency', 0)
            if v_freq > 0 and l_freq > 0:
                freq_ratio = min(v_freq, l_freq) / max(v_freq, l_freq)
            else:
                freq_ratio = 0.0

            matrix[i, j] = 0.4 * rank_score + 0.3 * type_score + 0.3 * freq_ratio

    return matrix, v_keys, l_keys


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_affix_isolation() -> None:
    """Phase 19.3: Affix layer isolation and independent decoding."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 19.3: Affix Layer Isolation and Independent Decoding")
    print("=" * 60)

    # ── 1. Load dependencies ──────────────────────────────────────────
    print("\n  1. Loading morpheme grid, assignment, and corpus …")

    morph_data = _load_json(os.path.join(rd, 'morpheme_grid.json'))
    refine_data = _load_json(os.path.join(rd, 'combined_refine.json'))
    mod_data = _load_json(os.path.join(rd, 'modifier_integrate.json'))

    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()

    # Extract prefix/suffix inventories
    if morph_data and 'stats' in morph_data:
        stats = morph_data['stats']
        prefix_inv = stats.get('prefix_distribution', {})
        suffix_inv = stats.get('suffix_distribution', {})
    else:
        prefix_inv = {'o': 0, 'd': 0, 'y': 0, 's': 0}
        suffix_inv = {'dy': 0, 'y': 0, 'ey': 0, 'aiin': 0, 'ol': 0,
                      'in': 0, 'al': 0, 'am': 0, 'm': 0, 'iin': 0,
                      'an': 0, 'aiiin': 0, 'iiin': 0, 'n': 0}
        print("  [WARN] morpheme_grid.json missing; using default affixes")

    prefix_list = sorted(prefix_inv.keys())
    suffix_list = sorted(suffix_inv.keys(), key=lambda s: suffix_inv.get(s, 0), reverse=True)

    # Load assignment for decoding
    best_assignment = {}
    if refine_data and 'best_assignment' in refine_data:
        best_assignment = refine_data['best_assignment']
    elif refine_data:
        # Try alternate key names
        for key in ['assignment', 'latin_assignment', 'best_latin_assignment']:
            if key in refine_data:
                best_assignment = refine_data[key]
                break

    modifier_chars = set()
    if mod_data and 'modifier_chars' in mod_data:
        modifier_chars = set(mod_data['modifier_chars'])

    eva_to_triple = build_eva_to_triple_lookup()

    print(f"    {len(prefix_list)} prefixes, {len(suffix_list)} suffixes, {len(tokens)} tokens")

    # ── 2. Build Voynich affix profiles ───────────────────────────────
    print("\n  2. Building Voynich affix distributional profiles …")

    # Decompose all tokens
    prefix_counts = Counter()
    suffix_counts = Counter()
    stem_tokens = []
    n_with_prefix = 0
    n_with_suffix = 0
    affix_cooccurrence = defaultdict(Counter)

    sections = ['herbal_a', 'herbal_b', 'pharmaceutical', 'biological',
                'astronomical', 'cosmological', 'recipes']
    section_affix_counts: Dict[str, Counter] = {s: Counter() for s in sections}

    for page in corpus.pages.values():
        section = page.section if hasattr(page, 'section') else 'unknown'
        for tok in page.all_tokens:
            chars = tokenize_eva_chars(tok)
            pref, suf, stem_chars = _strip_affix(chars, prefix_list, suffix_list)

            if pref:
                prefix_counts[pref] += 1
                n_with_prefix += 1
                if section in section_affix_counts:
                    section_affix_counts[section][pref] += 1
            if suf:
                suffix_counts[suf] += 1
                n_with_suffix += 1
                if section in section_affix_counts:
                    section_affix_counts[section][suf] += 1
            if pref and suf:
                affix_cooccurrence[pref][suf] += 1
                affix_cooccurrence[suf][pref] += 1

            stem_tokens.append(''.join(stem_chars) if stem_chars else tok)

    # Build per-affix profiles
    all_affixes = {**{p: 'prefix' for p in prefix_list},
                   **{s: 'suffix' for s in suffix_list}}
    voynich_profiles = {}
    total_affixes = sum(prefix_counts.values()) + sum(suffix_counts.values())

    for affix, atype in all_affixes.items():
        freq = prefix_counts.get(affix, 0) if atype == 'prefix' else suffix_counts.get(affix, 0)
        section_dist = {}
        for sec in sections:
            sec_total = sum(section_affix_counts[sec].values())
            section_dist[sec] = section_affix_counts[sec].get(affix, 0) / sec_total if sec_total > 0 else 0

        # Decode the affix phonetically
        decoded = ''
        if best_assignment:
            affix_triples = token_to_triples(affix, eva_to_triple)
            decoded = ''.join(best_assignment.get(t, '?') for t in affix_triples)

        co_affixes = sorted(affix_cooccurrence.get(affix, {}).keys(),
                            key=lambda a: affix_cooccurrence[affix][a], reverse=True)

        voynich_profiles[affix] = {
            'affix_type': atype,
            'frequency': freq,
            'frequency_rank': 0,
            'relative_frequency': freq / total_affixes if total_affixes > 0 else 0,
            'section_distribution': section_dist,
            'co_occurring_affixes': co_affixes[:5],
            'decoded_phonetic': decoded,
        }

    # Assign frequency ranks
    sorted_v = sorted(voynich_profiles.keys(),
                      key=lambda a: voynich_profiles[a]['frequency'], reverse=True)
    for rank, affix in enumerate(sorted_v):
        voynich_profiles[affix]['frequency_rank'] = rank

    print(f"    Built profiles for {len(voynich_profiles)} affixes")
    for a in sorted_v[:6]:
        p = voynich_profiles[a]
        print(f"      {a:6s} ({p['affix_type']:6s}) freq={p['frequency']:5d}  decoded='{p['decoded_phonetic']}'")

    # ── 3. Build Latin ending profiles ────────────────────────────────
    print("\n  3. Building Latin inflectional ending profiles …")

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_profiles = _build_latin_ending_profiles(ref_corpus)
    print(f"    Built profiles for {len(latin_profiles)} Latin endings")

    # ── 4. Hungarian assignment ───────────────────────────────────────
    print("\n  4. Computing optimal affix-to-ending assignment (Hungarian) …")

    if latin_profiles and voynich_profiles:
        compat_matrix, v_keys, l_keys = _build_compatibility_matrix(
            voynich_profiles, latin_profiles,
        )

        # Hungarian: maximize → minimize negated
        row_idx, col_idx = linear_sum_assignment(-compat_matrix)

        best_mapping = {}
        mapping_score = 0.0
        for r, c in zip(row_idx, col_idx):
            best_mapping[v_keys[r]] = l_keys[c]
            mapping_score += compat_matrix[r, c]

        mapping_score /= len(row_idx) if len(row_idx) > 0 else 1
        print(f"    Mean compatibility score: {mapping_score:.4f}")
        for vk in sorted_v[:8]:
            lk = best_mapping.get(vk, '?')
            print(f"      {vk:6s} → {lk}")
    else:
        best_mapping = {}
        mapping_score = 0.0
        print("  [WARN] Cannot compute assignment — missing data")

    # ── 5. Paradigm consistency check ─────────────────────────────────
    print("\n  5. Checking paradigm consistency …")

    n_violations = 0
    n_total = 0

    # Group Voynich affixes by decoded stem class
    paradigm_groups: Dict[str, List[str]] = defaultdict(list)
    for affix, latin_end in best_mapping.items():
        if latin_end in latin_profiles:
            for paradigm in latin_profiles[latin_end].get('paradigm_classes', []):
                paradigm_groups[paradigm].append(affix)
                n_total += 1

    # Check: affixes mapped to the same paradigm should co-occur on the same stems
    for paradigm, affixes in paradigm_groups.items():
        for i in range(len(affixes)):
            for j in range(i + 1, len(affixes)):
                a1, a2 = affixes[i], affixes[j]
                co = affix_cooccurrence.get(a1, {}).get(a2, 0)
                if co == 0:
                    n_violations += 1

    consistency = 1.0 - (n_violations / n_total) if n_total > 0 else 0.0
    print(f"    Violations: {n_violations}/{n_total}, consistency: {consistency:.3f}")

    # ── 6. Null test ─────────────────────────────────────────────────
    print("\n  6. Running null test (100 shuffled assignments) …")

    rng = random.Random(42)
    null_scores = []

    for trial in range(100):
        shuffled_v_keys = list(v_keys)
        rng.shuffle(shuffled_v_keys)
        trial_score = 0.0
        for r, c in zip(range(len(shuffled_v_keys)), col_idx):
            if r < len(shuffled_v_keys) and c < len(l_keys):
                # Find the row in the matrix corresponding to the shuffled key
                orig_r = v_keys.index(shuffled_v_keys[r]) if shuffled_v_keys[r] in v_keys else r
                trial_score += compat_matrix[orig_r, c]
        trial_score /= len(row_idx) if len(row_idx) > 0 else 1
        null_scores.append(trial_score)

    null_arr = np.array(null_scores)
    sel = selectivity_ratio(mapping_score, null_arr)
    print(f"    Real score: {mapping_score:.4f}, null mean: {np.mean(null_arr):.4f}")
    print(f"    Selectivity: {sel:.2f}×")

    # ── 7. Cross-validation ──────────────────────────────────────────
    print("\n  7. Cross-validation (odd/even folios) …")

    # Simple proxy: compare affix frequency ranks on odd vs even pages
    odd_counts = Counter()
    even_counts = Counter()
    for page in corpus.pages.values():
        folio = page.folio
        try:
            num = int(''.join(c for c in folio if c.isdigit()))
            is_odd = num % 2 == 1
        except ValueError:
            is_odd = True

        for tok in page.all_tokens:
            chars = tokenize_eva_chars(tok)
            _, suf, _ = _strip_affix(chars, prefix_list, suffix_list)
            if suf:
                if is_odd:
                    odd_counts[suf] += 1
                else:
                    even_counts[suf] += 1

    # Rank correlation between odd and even
    common_sufs = sorted(set(odd_counts) & set(even_counts))
    if len(common_sufs) >= 3:
        odd_ranks = [sorted(odd_counts, key=odd_counts.get, reverse=True).index(s)
                     for s in common_sufs]
        even_ranks = [sorted(even_counts, key=even_counts.get, reverse=True).index(s)
                      for s in common_sufs]
        from scipy.stats import spearmanr
        rho, _ = spearmanr(odd_ranks, even_ranks)
        cv_ratio = float(rho)
    else:
        cv_ratio = 0.0

    print(f"    Cross-validation rank correlation: {cv_ratio:.3f}")

    # ── 8. Gate ──────────────────────────────────────────────────────
    gate_passed = bool(sel >= 1.5 and consistency >= 0.5)

    if gate_passed:
        verdict = f"PASS: selectivity={sel:.2f}×, paradigm consistency={consistency:.3f}"
    else:
        verdict = f"FAIL: selectivity={sel:.2f}×, paradigm consistency={consistency:.3f}"

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 9. Save ──────────────────────────────────────────────────────
    result = AffixIsolationResult(
        n_tokens=len(tokens),
        n_with_prefix=n_with_prefix,
        n_with_suffix=n_with_suffix,
        n_stems=len(set(stem_tokens)),
        prefix_inventory=dict(prefix_counts),
        suffix_inventory=dict(suffix_counts),
        affix_profiles=[_convert(v) for v in voynich_profiles.values()],
        best_mapping=best_mapping,
        mapping_score=round(mapping_score, 4),
        n_paradigm_violations=n_violations,
        n_paradigm_total=n_total,
        paradigm_consistency=round(consistency, 4),
        null_scores=[round(s, 4) for s in null_scores],
        selectivity=round(sel, 4),
        odd_folio_score=round(cv_ratio, 4),
        even_folio_score=round(cv_ratio, 4),
        cv_ratio=round(cv_ratio, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'affix_isolation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n    → {out_path}")
