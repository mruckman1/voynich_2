"""
Phase 59, Investigation 10: The "aiin" Family Deep Dive
=========================================================
hook→n coda means every "-aiin" suffix adds "n" to the preceding syllable.
This is the most common suffix (3,837 tokens).  Under CVC decode, "-aiin"
becomes coda "n" instead of an independent syllable.  This module checks
whether the resulting CVC endings cluster into recognizable Latin declension
patterns.

Dependency chain:
    results/coda_table.json           (Phase 57.1)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/cvc_aiin_family.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.phases.coda_markers import (
    build_coda_table,
    decode_token_cvc,
)
from voynich.phases.coda_markers import decode_corpus_cv_strip


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
# Constants
# ---------------------------------------------------------------------------

HOOK_SUFFIXES = {'aiin', 'aiiin', 'iin', 'iiin', 'n'}

# Latin declension patterns for -n endings
DECLENSION_MAP = {
    'an': '1st decl accusative (-am/-an)',
    'en': '3rd decl accusative/ablative (-em/-en)',
    'in': 'prepositional/locative (-in)',
    'on': '2nd decl accusative (Gallo-Italic -um→-on)',
    'un': '2nd decl accusative (-um/-un)',
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CvcAiinResult:
    """Full Investigation 10 output."""
    phase: str = "59"
    investigation: str = "10"
    experiment: str = "cvc_aiin_family"
    total_aiin_tokens: int = 0
    # Ending distribution
    ending_distribution: Dict[str, int] = field(default_factory=dict)
    top_3_cover: float = 0.0         # fraction covered by top 3 endings
    # Declension analysis
    declension_analysis: Dict[str, int] = field(default_factory=dict)
    latin_ending_fraction: float = 0.0
    # Sample tokens
    sample_tokens: List[Dict[str, Any]] = field(default_factory=list)
    # Per-suffix breakdown
    per_suffix_counts: Dict[str, int] = field(default_factory=dict)
    # Gates
    g1_enough_data: bool = False       # ≥ 1000 tokens
    g2_clustering: bool = False        # top 3 cover ≥ 60%
    g3_latin_endings: bool = False     # Latin ending fraction ≥ 40%
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def map_aiin_tokens(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
) -> List[Dict[str, Any]]:
    """Map all tokens ending in hook-group suffixes."""
    aiin_tokens = []

    for idx, token in enumerate(all_tokens):
        chars = tokenize_eva_chars(token)
        if not chars:
            continue

        # Check if last char is a hook suffix
        last_char = chars[-1]
        if last_char not in HOOK_SUFFIXES:
            continue

        result = decode_token_cvc(token, assignment, eva_to_triple, coda_table)
        cvc = result.decoded_cvc
        cv = result.decoded_cv

        # What does the CVC form end in?
        if cvc and cvc[-1] == 'n':
            ending = cvc[-2:] if len(cvc) >= 2 else cvc
        else:
            ending = cvc[-2:] if len(cvc) >= 2 else cvc

        aiin_tokens.append({
            'token_idx': idx,
            'eva': token,
            'cv_decode': cv,
            'cvc_decode': cvc,
            'cvc_ending': ending,
            'hook_suffix': last_char,
        })

    return aiin_tokens


def classify_endings(aiin_tokens: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Classify CVC endings by Latin declension pattern."""
    ending_counts = Counter(t['cvc_ending'] for t in aiin_tokens)

    declension_hits: Dict[str, int] = {}
    for ending, count in ending_counts.items():
        # Check last 2 chars against declension map
        suffix_2 = ending[-2:] if len(ending) >= 2 else ending
        if suffix_2 in DECLENSION_MAP:
            decl = DECLENSION_MAP[suffix_2]
            declension_hits[decl] = declension_hits.get(decl, 0) + count

    return dict(ending_counts.most_common(30)), declension_hits


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_aiin():
    """Investigation 10: The aiin family deep dive."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 10: The 'aiin' Family Deep Dive")
    print("=" * 70)

    rd = str(_results_dir())

    # Load data
    eva_to_triple = build_eva_to_triple_lookup()
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    coda_table = build_coda_table('primary')

    # Map aiin tokens
    print("\n  Mapping tokens ending in hook-group suffixes ...")
    aiin_tokens = map_aiin_tokens(all_tokens, assignment, eva_to_triple, coda_table)
    print(f"  Total -aiin tokens: {len(aiin_tokens)}")

    if not aiin_tokens:
        result = CvcAiinResult(runtime_seconds=round(time.time() - t0, 2))
        _save_json(rd, 'cvc_aiin_family.json', result)
        return

    # Per-suffix breakdown
    per_suffix = Counter(t['hook_suffix'] for t in aiin_tokens)
    print(f"\n  Per-suffix breakdown:")
    for suffix, count in per_suffix.most_common():
        print(f"    {suffix:8s} {count:>6}")

    # Classify endings
    ending_dist, declension_hits = classify_endings(aiin_tokens)

    total = len(aiin_tokens)
    top_3_endings = list(ending_dist.values())[:3]
    top_3_cover = sum(top_3_endings) / total if total > 0 else 0

    latin_ending_count = sum(declension_hits.values())
    latin_frac = latin_ending_count / total if total > 0 else 0

    print(f"\n  Ending distribution (top 15):")
    for ending, count in list(ending_dist.items())[:15]:
        frac = count / total
        decl = DECLENSION_MAP.get(ending[-2:] if len(ending) >= 2 else ending, '')
        marker = ' ← ' + decl if decl else ''
        print(f"    {ending:8s} {count:>6} ({frac:.1%}){marker}")

    print(f"\n  Top 3 endings cover: {top_3_cover:.1%}")
    print(f"\n  Latin declension analysis:")
    for decl, count in sorted(declension_hits.items(), key=lambda x: -x[1]):
        print(f"    {decl:40s} {count:>6}")
    print(f"  Latin ending fraction: {latin_frac:.1%}")

    # Sample tokens
    samples = []
    for tok in aiin_tokens[:20]:
        samples.append({
            'eva': tok['eva'],
            'cv': tok['cv_decode'],
            'cvc': tok['cvc_decode'],
            'ending': tok['cvc_ending'],
            'suffix': tok['hook_suffix'],
        })

    print(f"\n  Sample tokens:")
    print(f"  {'EVA':<16} {'CV decode':<14} {'CVC decode':<14} {'Ending':<8} {'Suffix'}")
    print(f"  {'-'*16} {'-'*14} {'-'*14} {'-'*8} {'-'*8}")
    for s in samples[:10]:
        print(f"  {s['eva']:<16} {s['cv']:<14} {s['cvc']:<14} "
              f"{s['ending']:<8} {s['suffix']}")

    # Gates
    g1 = len(aiin_tokens) >= 1000
    g2 = top_3_cover >= 0.60
    g3 = latin_frac >= 0.40
    gates_passed = sum([g1, g2, g3])

    print(f"\n  Validation Gates:")
    print(f"    G1 ≥ 1000 tokens:           {'PASS' if g1 else 'FAIL'} ({len(aiin_tokens)})")
    print(f"    G2 top 3 cover ≥ 60%:       {'PASS' if g2 else 'FAIL'} ({top_3_cover:.1%})")
    print(f"    G3 Latin ending frac ≥ 40%: {'PASS' if g3 else 'FAIL'} ({latin_frac:.1%})")
    print(f"    Gates passed: {gates_passed}/3")

    result = CvcAiinResult(
        total_aiin_tokens=len(aiin_tokens),
        ending_distribution=ending_dist,
        top_3_cover=round(top_3_cover, 4),
        declension_analysis=declension_hits,
        latin_ending_fraction=round(latin_frac, 4),
        sample_tokens=samples,
        per_suffix_counts=dict(per_suffix),
        g1_enough_data=g1,
        g2_clustering=g2,
        g3_latin_endings=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_aiin_family.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 10 completed in {time.time() - t0:.1f}s")
