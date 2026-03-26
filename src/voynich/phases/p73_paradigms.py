"""
Phase 73, Track 4: Corrected Paradigm Mapping
===============================================
Re-run Phase 70 Track 2 paradigm mapping and Phase 71 Track 2 root
identification with corrected decode. Uses corrected T1 catalogue
from Track 3.

With connector→null, decoded strings are shorter, paradigm families
should be smaller (no -r inflation), and coda-to-case mapping uses
only 3 genuine codas (n, s, t) plus descender-r.

Dependency chain:
    results/p73_redecode.json          (Step 0)
    results/p73_t1.json                (Track 3 — corrected T1 catalogue)
    results/combined_refine.json       (Phase 15)
    results/phase70_paradigms.json     (Phase 70, for comparison)
    results/phase71_root_identification.json (Phase 71, for comparison)
        -> results/p73_paradigms.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import (
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.inflectional_catalog import CODA_GRAMMAR
from voynich.phases.p72_connector import _build_coda_table_with_connector
from voynich.phases.suffix_grammar import _classify_latin_ending


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
# Known roots (from Phase 71)
# ---------------------------------------------------------------------------

_KNOWN_ROOTS: Dict[str, Dict[str, str]] = {
    'cor': {'meaning': 'heart', 'class': 'BODY_PART'},
    'rad': {'meaning': 'root', 'class': 'INGREDIENT'},
    'herb': {'meaning': 'herb', 'class': 'INGREDIENT'},
    'aqu': {'meaning': 'water', 'class': 'INGREDIENT'},
    'sen': {'meaning': 'senna', 'class': 'INGREDIENT'},
    'bel': {'meaning': 'beautiful', 'class': 'QUALITY'},
    'ser': {'meaning': 'serum/evening', 'class': 'OTHER'},
    'col': {'meaning': 'strain', 'class': 'PREPARATION'},
    'ben': {'meaning': 'good/well', 'class': 'FUNCTION'},
    'dic': {'meaning': 'say', 'class': 'FUNCTION'},
    'di': {'meaning': 'of/from', 'class': 'FUNCTION'},
    'ne': {'meaning': 'not/nor', 'class': 'FUNCTION'},
    'se': {'meaning': 'if/self', 'class': 'FUNCTION'},
    'co': {'meaning': 'with/together', 'class': 'FUNCTION'},
    'ra': {'meaning': 'root/radical', 'class': 'FUNCTION'},
    'con': {'meaning': 'with', 'class': 'FUNCTION'},
    'ber': {'meaning': 'berry', 'class': 'INGREDIENT'},
    'din': {'meaning': '?', 'class': 'OTHER'},
    'cos': {'meaning': '?', 'class': 'OTHER'},
    'ter': {'meaning': 'grind', 'class': 'PREPARATION'},
    'coc': {'meaning': 'cook', 'class': 'PREPARATION'},
    'mis': {'meaning': 'mix', 'class': 'PREPARATION'},
    'rec': {'meaning': 'take/receive', 'class': 'PREPARATION'},
    'pis': {'meaning': 'pound', 'class': 'PREPARATION'},
}


# ---------------------------------------------------------------------------
# Paradigm discovery
# ---------------------------------------------------------------------------

def _discover_paradigms(
    decoded_types: List[str],
    min_family_size: int = 3,
    min_root_len: int = 2,
    max_root_len: int = 6,
) -> List[Dict[str, Any]]:
    """Find morphological paradigms in decoded vocabulary.

    Longest-prefix-first grouping (root_len 6→2).
    Requires >= min_family_size members with different suffixes.
    """
    type_set = set(decoded_types)
    assigned: Set[str] = set()
    paradigms = []

    for root_len in range(max_root_len, min_root_len - 1, -1):
        prefix_groups: Dict[str, List[str]] = {}
        for word in type_set - assigned:
            if len(word) >= root_len:
                prefix = word[:root_len]
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(word)

        for prefix, members in sorted(prefix_groups.items(),
                                       key=lambda x: -len(x[1])):
            if len(members) < min_family_size:
                continue

            suffixes = set(m[root_len:] for m in members)
            if len(suffixes) < 2:
                continue

            # Check if root is known
            root_info = None
            for known_root, info in _KNOWN_ROOTS.items():
                if prefix.startswith(known_root) or known_root.startswith(prefix):
                    root_info = info
                    break

            paradigms.append({
                'root': prefix,
                'root_len': root_len,
                'n_forms': len(members),
                'n_suffixes': len(suffixes),
                'members': sorted(members)[:20],
                'meaning': root_info['meaning'] if root_info else '?',
                'pharma_class': root_info['class'] if root_info else 'UNKNOWN',
            })

            assigned.update(members)

    paradigms.sort(key=lambda p: -p['n_forms'])
    return paradigms


def _map_coda_to_case(
    t1_catalogue: List[Dict[str, Any]],
    all_tokens: List[str],
    decoded_tokens: List[str],
    corrected_coda,
    eva_to_triple: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Map coda consonants to Latin case endings using T1-identified words."""
    # Build T1 map: EVA type → matched word
    t1_map = {i['token']: i['matched_word']
              for i in t1_catalogue if 'token' in i and 'matched_word' in i}

    coda_case_obs: Dict[str, Counter] = {}

    for idx, token in enumerate(all_tokens):
        if token not in t1_map:
            continue

        matched = t1_map[token]
        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, corrected_coda)

        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, corrected_coda)
                if coda_val:
                    codas.append(coda_val)

        if not codas:
            continue

        last_coda = codas[-1]
        pos, case_ending = _classify_latin_ending(matched)
        if not pos or pos == 'UNCLEAR':
            continue

        if last_coda not in coda_case_obs:
            coda_case_obs[last_coda] = Counter()
        coda_case_obs[last_coda][f"{pos}_{case_ending}" if case_ending else pos] += 1

    # Summarize
    result = {}
    for coda, counts in sorted(coda_case_obs.items()):
        total = sum(counts.values())
        dominant = counts.most_common(1)[0] if counts else ('UNKNOWN', 0)
        result[coda] = {
            'total_observations': total,
            'dominant_case': dominant[0],
            'dominance_fraction': dominant[1] / total if total > 0 else 0.0,
            'distribution': dict(counts.most_common()),
            'is_consistent': (dominant[1] / total) > 0.50 if total > 0 else False,
        }

    return result


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorrectedParadigmsResult:
    phase: str = "73"
    step: str = "73.4"
    experiment: str = "corrected_paradigms"
    # Paradigm stats
    n_paradigms: int = 0
    largest_paradigm_size: int = 0
    mean_paradigm_size: float = 0.0
    paradigm_details: List[Dict[str, Any]] = field(default_factory=list)
    # Root dictionary
    n_roots_identified: int = 0
    n_roots_unknown: int = 0
    pharma_distribution: Dict[str, int] = field(default_factory=dict)
    # Coda-to-case mapping
    coda_case_mapping: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_consistent_codas: int = 0
    # Comparison
    old_n_paradigms: int = 0
    old_largest: int = 0
    # Gates
    gate_p1: bool = False  # Largest ≤ 30
    gate_p2: bool = False  # Mean size 3-10
    gate_p3: bool = False  # ≥ 3 consistent codas
    gate_p4: bool = False  # ≥ 5 PREPARATION roots
    gate_p5: bool = False  # ≥ 40% identified
    gates_passed: int = 0
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_corrected_paradigms() -> CorrectedParadigmsResult:
    """Track 4: Paradigm mapping with corrected decode."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 73.4 — Corrected Paradigm Mapping")
    print("=" * 45)

    # --- Load corrected decode ---
    redecode_data = _safe_load(os.path.join(rd, 'p73_redecode.json'))
    decoded_tokens = redecode_data.get('decoded_tokens', [])
    if not decoded_tokens:
        print("  ERROR: p73_redecode.json not found. Run redecode first.")
        return CorrectedParadigmsResult()

    # --- Load corrected T1 ---
    t1_data = _safe_load(os.path.join(rd, 'p73_t1.json'))
    t1_catalogue = t1_data.get('identifications', [])
    print(f"  Corrected T1 identifications: {len(t1_catalogue)}")

    # --- Load old data for comparison ---
    old_paradigm_data = _safe_load(os.path.join(rd, 'phase71_root_identification.json'))
    old_n_paradigms = old_paradigm_data.get('n_paradigms', 342)
    old_largest = 117  # from Phase 71 docs

    # --- Load corpus ---
    eva_to_triple = build_eva_to_triple_lookup()
    corrected_coda = _build_coda_table_with_connector('')

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    print(f"  Tokens: {len(all_tokens)}")

    # --- Discover paradigms from corrected decoded types ---
    decoded_types = sorted(set(d for d in decoded_tokens if d and '?' not in d))
    print(f"  Decoded vocabulary types: {len(decoded_types)}")

    paradigms = _discover_paradigms(decoded_types)
    n_paradigms = len(paradigms)
    largest = paradigms[0]['n_forms'] if paradigms else 0
    sizes = [p['n_forms'] for p in paradigms]
    mean_size = float(np.mean(sizes)) if sizes else 0.0

    print(f"  Paradigms found: {n_paradigms} (was {old_n_paradigms})")
    print(f"  Largest: {largest} (was {old_largest})")
    print(f"  Mean size: {mean_size:.1f}")

    # --- Root classification ---
    pharma_dist = Counter(p['pharma_class'] for p in paradigms)
    n_identified = sum(1 for p in paradigms if p['meaning'] != '?')
    n_unknown = n_paradigms - n_identified
    id_fraction = n_identified / n_paradigms if n_paradigms > 0 else 0.0

    print(f"  Identified roots: {n_identified}/{n_paradigms} ({100*id_fraction:.1f}%)")
    print(f"  Pharmaceutical classification: {dict(pharma_dist.most_common())}")

    # --- Coda-to-case mapping ---
    print("  Mapping codas to case endings...")
    coda_case = _map_coda_to_case(
        t1_catalogue, all_tokens, decoded_tokens, corrected_coda, eva_to_triple)

    n_consistent = sum(1 for v in coda_case.values() if v.get('is_consistent'))
    for coda, info in sorted(coda_case.items()):
        print(f"    -{coda}: {info['dominant_case']} ({info['dominance_fraction']:.0%} "
              f"of {info['total_observations']} obs)")

    # --- Gates ---
    n_prep = pharma_dist.get('PREPARATION', 0)
    gate_p1 = largest <= 30
    gate_p2 = 3 <= mean_size <= 10
    gate_p3 = n_consistent >= 3
    gate_p4 = n_prep >= 5
    gate_p5 = id_fraction >= 0.40
    gates_passed = sum([gate_p1, gate_p2, gate_p3, gate_p4, gate_p5])

    if gates_passed >= 4:
        verdict = 'PARADIGMS_VALIDATED'
    elif gates_passed >= 2:
        verdict = 'PARADIGMS_PARTIAL'
    else:
        verdict = 'PARADIGMS_FAILED'

    result = CorrectedParadigmsResult(
        n_paradigms=n_paradigms,
        largest_paradigm_size=largest,
        mean_paradigm_size=round(mean_size, 2),
        paradigm_details=paradigms[:50],
        n_roots_identified=n_identified,
        n_roots_unknown=n_unknown,
        pharma_distribution=dict(pharma_dist.most_common()),
        coda_case_mapping=coda_case,
        n_consistent_codas=n_consistent,
        old_n_paradigms=old_n_paradigms,
        old_largest=old_largest,
        gate_p1=gate_p1,
        gate_p2=gate_p2,
        gate_p3=gate_p3,
        gate_p4=gate_p4,
        gate_p5=gate_p5,
        gates_passed=gates_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p73_paradigms.json', asdict(result))
    print(f"\n  Verdict: {verdict} ({gates_passed}/5)")
    print(f"  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
