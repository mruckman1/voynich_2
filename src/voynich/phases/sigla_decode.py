"""
Step 34.3 – Sigla-Specific Decode (Track A)
=============================================
Tests whether abjad-decoded consonant sequences match known medieval
medical abbreviations from Cappelli.

Dependency chain:
    sigla_dictionary.json  (34.1)
    abjad_csp.json         (34.2)
        → sigla_decode.json  (this step)
"""

import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import load_reference_corpus
from voynich.phases.morpheme_grid import decompose_token_morphemes
from voynich.phases.sigla_dictionary import _find_cappelli_path, _load_cappelli_sigla, _strip_vowels


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
class SiglaDecodeResult:
    # Cappelli matching
    n_decoded_roots: int
    n_cappelli_matches: int
    cappelli_match_rate: float
    match_by_pattern_type: Dict[str, int]  # suspension/contraction/standard
    top_matches: List[Dict]

    # Frequency rank correlation
    spearman_rho: float
    freq_p_value: float
    n_freq_pairs: int

    # Section-specific abbreviation coherence
    section_coherence: Dict[str, Dict]

    # Botanical label re-test
    n_herbal_folios_tested: int
    n_label_matches: int
    label_matches: List[Dict]

    runtime_seconds: float


# ---------------------------------------------------------------------------
# Abbreviation pattern matching
# ---------------------------------------------------------------------------

def _match_cappelli(
    consonant_seq: str,
    sigla: List[Dict],
) -> List[Dict]:
    """Match a consonant sequence against Cappelli abbreviations.

    Tries three pattern types:
    1. Standard: exact match of clean_form or skeleton
    2. Suspension: consonant_seq is a prefix of some sigla skeleton
    3. Contraction: consonant_seq matches first + last consonants
    """
    matches = []
    if len(consonant_seq) < 1:
        return matches

    for entry in sigla:
        skel = entry['skeleton']
        clean = entry['clean_form']

        # Standard match
        if consonant_seq == skel or consonant_seq == clean:
            matches.append({
                'expansion': entry['expansion'],
                'pattern': 'standard',
                'domain': entry['domain'],
                'abbreviated_form': entry['abbreviated_form'],
            })
            continue

        # Suspension: consonant_seq is a prefix of the skeleton
        if len(consonant_seq) >= 2 and skel.startswith(consonant_seq):
            matches.append({
                'expansion': entry['expansion'],
                'pattern': 'suspension',
                'domain': entry['domain'],
                'abbreviated_form': entry['abbreviated_form'],
            })
            continue

        # Contraction: first and last consonants match
        if (len(consonant_seq) >= 2 and len(skel) >= 2
                and consonant_seq[0] == skel[0]
                and consonant_seq[-1] == skel[-1]):
            matches.append({
                'expansion': entry['expansion'],
                'pattern': 'contraction',
                'domain': entry['domain'],
                'abbreviated_form': entry['abbreviated_form'],
            })

    return matches


def _frequency_rank_correlation(
    decoded_roots: List[str],
    sigla: List[Dict],
    ref_corpus,
) -> Tuple[float, float, int]:
    """Compute Spearman rank correlation between EVA root frequency
    and Cappelli abbreviation frequency in reference corpus.
    """
    # Count EVA root frequency
    root_freq = Counter(decoded_roots)

    # Match each decoded root to Cappelli and get the expansion
    root_to_expansion: Dict[str, str] = {}
    for root, _ in root_freq.most_common():
        matches = _match_cappelli(root, sigla)
        if matches:
            root_to_expansion[root] = matches[0]['expansion']

    if len(root_to_expansion) < 5:
        return 0.0, 1.0, len(root_to_expansion)

    # Get reference corpus word frequencies
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_freq = Counter(w.lower() for w in ref_tokens)

    # Build paired ranks
    pairs = []
    for root, expansion in root_to_expansion.items():
        # Clean expansion to get base word
        exp_clean = re.sub(r'[^a-zA-Z]', '', expansion).lower()
        if exp_clean in ref_freq:
            pairs.append((root_freq[root], ref_freq[exp_clean]))

    if len(pairs) < 5:
        return 0.0, 1.0, len(pairs)

    # Spearman rank correlation
    n = len(pairs)
    rank_x = _rank([ p[0] for p in pairs])
    rank_y = _rank([p[1] for p in pairs])
    d_sq = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1)) if n > 1 else 0.0

    # Approximate p-value
    if n > 3:
        t_stat = rho * ((n - 2) / (1 - rho ** 2 + 1e-10)) ** 0.5
        p_value = max(0.001, 2.0 / (1 + abs(t_stat)))  # rough approximation
    else:
        p_value = 1.0

    return round(rho, 4), round(p_value, 4), n


def _rank(values: List[float]) -> List[float]:
    """Compute ranks with average tie-breaking."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sigla_decode() -> None:
    """Step 34.3: Sigla-specific decode."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.3: Sigla-Specific Decode (Track A)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load abjad assignment ──
    print("\n  1. Loading abjad assignment …")
    abjad_path = os.path.join(rd, 'abjad_csp.json')
    if not os.path.exists(abjad_path):
        print("  [SKIP] abjad_csp.json not found — run abjad-csp first")
        return
    with open(abjad_path) as f:
        abjad_data = json.load(f)
    abjad_table = abjad_data.get('best_assignment', {})
    print(f"     Loaded {len(abjad_table)} triple → consonant assignments")

    # ── 2. Load Cappelli sigla ──
    print("\n  2. Loading Cappelli sigla …")
    cappelli_path = _find_cappelli_path()
    sigla = _load_cappelli_sigla(cappelli_path)
    print(f"     {len(sigla)} entries")

    # ── 3. Load corpus + decode roots ──
    print("\n  3. Decoding roots through abjad table …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    all_tokens: List[str] = []
    token_folios: List[str] = []
    token_sections: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
            token_sections.append(page.section)

    decoded_roots: List[str] = []
    for token in all_tokens:
        decomp = decompose_token_morphemes(token)
        stem = decomp.stem if hasattr(decomp, 'stem') else token
        stem_chars = tokenize_eva_chars(stem)
        consonants = []
        for ch in stem_chars:
            triple = eva_to_triple.get(ch)
            if triple and triple in abjad_table:
                consonants.append(abjad_table[triple])
        decoded_roots.append(''.join(consonants))

    print(f"     {len(decoded_roots)} roots decoded")

    # ── 4. Match against Cappelli ──
    print("\n  4. Matching against Cappelli abbreviations …")
    n_matches = 0
    pattern_counts: Counter = Counter()
    top_matches: List[Dict] = []

    for i, (token, root) in enumerate(zip(all_tokens, decoded_roots)):
        if len(root) < 2:
            continue
        matches = _match_cappelli(root, sigla)
        if matches:
            n_matches += 1
            for m in matches:
                pattern_counts[m['pattern']] += 1
            if len(top_matches) < 30:
                top_matches.append({
                    'token': token,
                    'decoded_consonants': root,
                    'matches': matches[:3],
                    'folio': token_folios[i],
                })

    match_rate = n_matches / len(decoded_roots) if decoded_roots else 0.0
    print(f"     {n_matches} matches ({match_rate:.1%})")
    print(f"     By pattern: {dict(pattern_counts)}")

    # ── 5. Frequency rank correlation ──
    print("\n  5. Computing frequency rank correlation …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    rho, p_val, n_pairs = _frequency_rank_correlation(decoded_roots, sigla, ref_corpus)
    print(f"     Spearman ρ = {rho:.4f}, p = {p_val:.4f}, n = {n_pairs}")

    # ── 6. Section-specific abbreviation coherence ──
    print("\n  6. Section abbreviation coherence …")
    section_coherence: Dict[str, Dict] = {}
    sections = sorted(set(token_sections))
    for section in sections:
        section_roots = [r for r, s in zip(decoded_roots, token_sections) if s == section]
        if not section_roots:
            continue
        domain_hits: Counter = Counter()
        for root in section_roots:
            if len(root) < 2:
                continue
            matches = _match_cappelli(root, sigla)
            for m in matches:
                domain_hits[m['domain']] += 1
        total_domain = sum(domain_hits.values())
        section_coherence[section] = {
            'n_roots': len(section_roots),
            'n_domain_hits': total_domain,
            'domain_breakdown': dict(domain_hits.most_common(5)),
        }
        print(f"     {section}: {total_domain} domain hits")

    # ── 7. Botanical label re-test ──
    print("\n  7. Botanical label re-test …")
    # Load plant names
    from voynich.phases.sigla_dictionary import _strip_vowels
    plant_names = [
        'rosmarinus', 'calendula', 'salvia', 'plantago', 'artemisia',
        'mentha', 'urtica', 'verbena', 'sambucus', 'centaurea',
        'borago', 'melissa', 'viola', 'drosera', 'ranunculus',
        'helleborus', 'hyoscyamus', 'aconitum', 'mandragora', 'papaver',
    ]
    plant_skeletons = {_strip_vowels(name): name for name in plant_names}

    herbal_folios = [f for f, s in zip(token_folios, token_sections)
                     if 'herbal' in s.lower()]
    herbal_folios_unique = sorted(set(herbal_folios))

    label_matches: List[Dict] = []
    for folio in herbal_folios_unique[:20]:
        folio_roots = [r for r, f in zip(decoded_roots, token_folios) if f == folio]
        for root in folio_roots[:5]:  # Check first few tokens as potential labels
            if len(root) < 2:
                continue
            # Check suspension match with plant skeletons
            for skel, name in plant_skeletons.items():
                if skel.startswith(root) or root == skel[:len(root)]:
                    label_matches.append({
                        'folio': folio,
                        'decoded_consonants': root,
                        'plant_skeleton': skel,
                        'plant_name': name,
                        'match_type': 'suspension',
                    })

    print(f"     {len(herbal_folios_unique)} herbal folios tested")
    print(f"     {len(label_matches)} botanical label matches")

    elapsed = time.time() - t0

    result = SiglaDecodeResult(
        n_decoded_roots=len(decoded_roots),
        n_cappelli_matches=n_matches,
        cappelli_match_rate=round(match_rate, 4),
        match_by_pattern_type=dict(pattern_counts),
        top_matches=top_matches,
        spearman_rho=rho,
        freq_p_value=p_val,
        n_freq_pairs=n_pairs,
        section_coherence=section_coherence,
        n_herbal_folios_tested=len(herbal_folios_unique),
        n_label_matches=len(label_matches),
        label_matches=label_matches[:20],
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'sigla_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"\n  Completed in {elapsed:.1f}s")
