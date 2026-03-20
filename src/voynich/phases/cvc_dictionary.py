"""
Phase 59, Investigation 2: CVC-Aware Dictionary
=================================================
The 10K dictionary was built for CV-length words.  CVC decode produces
longer words that don't match.  This module builds a proper CVC-aware
evaluation dictionary from multiple sources and re-scores the CVC decode.

Dependency chain:
    results/coda_table.json           (Phase 57.1)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    results/null_corpus.json          (Phase 17)
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
        -> results/cvc_dictionary.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import data_dir, results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import build_coda_table, decode_corpus_cvc
from voynich.phases.cvc_coda_signal import _load_costamagna_syllables
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DictSourceStats:
    """Statistics for one dictionary source."""
    name: str
    n_words: int
    n_new: int  # words not already in prior sources
    sample: List[str] = field(default_factory=list)


@dataclass
class CvcDictResult:
    """Full Investigation 2 output."""
    phase: str = "59"
    investigation: str = "2"
    experiment: str = "cvc_dictionary"
    # Dictionary stats
    cvc_dict_size: int = 0
    old_dict_size: int = 0
    sources: List[DictSourceStats] = field(default_factory=list)
    # Scoring
    cvc_vs_old_dict: float = 0.0      # CVC decoded vs old 10K dict
    cvc_vs_cvc_dict: float = 0.0      # CVC decoded vs CVC-aware dict
    cv_vs_old_dict: float = 0.0       # CV decoded vs old 10K dict (baseline)
    # Null comparison
    null_vs_cvc_dict_mean: float = 0.0
    null_vs_cvc_dict_rates: List[float] = field(default_factory=list)
    cvc_selectivity: float = 0.0
    # Gates
    g1_dict_hit: bool = False          # CVC dict-hit ≥ 35%
    g2_selectivity: bool = False       # ≥ 1.5×
    g3_dict_size: bool = False         # < 50K
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Dictionary building
# ---------------------------------------------------------------------------

def build_cvc_dictionary(
    latin_ref_words: Set[str],
    italian_ref_words: Set[str],
    costamagna_all: Set[str],
) -> Tuple[Set[str], List[DictSourceStats]]:
    """Build CVC-aware evaluation dictionary from multiple sources.

    Sources:
    1. Costamagna syllables (all attested forms)
    2. Latin words of length 3-8
    3. Italian words of length 3-8
    4. The existing expanded Latin dictionary (baseline)
    """
    cvc_dict: Set[str] = set()
    sources: List[DictSourceStats] = []

    # Source 1: Costamagna syllables
    s1 = set()
    for syl in costamagna_all:
        sl = syl.lower()
        if sl and len(sl) >= 2:
            s1.add(sl)
    cvc_dict |= s1
    sources.append(DictSourceStats(
        name='costamagna_syllables',
        n_words=len(s1),
        n_new=len(s1),
        sample=sorted(s1)[:10],
    ))

    # Source 2: Latin words of length 3-8
    s2 = set()
    for word in latin_ref_words:
        wl = word.lower()
        if 3 <= len(wl) <= 8:
            s2.add(wl)
    new_2 = s2 - cvc_dict
    cvc_dict |= s2
    sources.append(DictSourceStats(
        name='latin_3_8',
        n_words=len(s2),
        n_new=len(new_2),
        sample=sorted(new_2)[:10],
    ))

    # Source 3: Italian words of length 3-8
    s3 = set()
    for word in italian_ref_words:
        wl = word.lower()
        if 3 <= len(wl) <= 8:
            s3.add(wl)
    new_3 = s3 - cvc_dict
    cvc_dict |= s3
    sources.append(DictSourceStats(
        name='italian_3_8',
        n_words=len(s3),
        n_new=len(new_3),
        sample=sorted(new_3)[:10],
    ))

    # Source 4: Full expanded Latin dictionary (the old 131K)
    expanded, _ = build_expanded_word_set(latin_ref_words)
    s4 = latin_ref_words | expanded
    new_4 = s4 - cvc_dict
    cvc_dict |= s4
    sources.append(DictSourceStats(
        name='expanded_latin',
        n_words=len(s4),
        n_new=len(new_4),
        sample=sorted(new_4)[:10],
    ))

    return cvc_dict, sources


def compute_dict_hit(decoded: List[str], word_set: Set[str]) -> float:
    """Compute dictionary hit rate."""
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_dict():
    """Investigation 2: Build CVC-aware dictionary and re-score."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 2: CVC-Aware Dictionary")
    print("=" * 70)

    rd = str(_results_dir())

    # Load reference corpora
    print("\n  Loading reference corpora ...")
    ref_corpus = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    latin_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                      if len(w) >= 2)
    italian_words = set()
    try:
        italian_words = set(w.lower() for w in ref_corpus.get_combined_tokens('italian')
                            if len(w) >= 2)
    except (KeyError, AttributeError):
        print("  (No Italian reference corpus available)")

    # Load Costamagna
    _, _, costamagna_all = _load_costamagna_syllables()
    print(f"  Latin words:      {len(latin_words)}")
    print(f"  Italian words:    {len(italian_words)}")
    print(f"  Costamagna syls:  {len(costamagna_all)}")

    # Build CVC dictionary
    print("\n  Building CVC-aware dictionary ...")
    cvc_dict, sources = build_cvc_dictionary(latin_words, italian_words, costamagna_all)
    print(f"  CVC dictionary size: {len(cvc_dict)}")

    for src in sources:
        print(f"    {src.name}: {src.n_words} words ({src.n_new} new)")

    # Old dictionary (expanded Latin)
    expanded, _ = build_expanded_word_set(latin_words)
    old_dict = latin_words | expanded
    print(f"  Old dictionary size: {len(old_dict)}")

    # Load corpus and decode
    print("\n  Loading corpus and decoding ...")
    eva_to_triple = build_eva_to_triple_lookup()
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    coda_table = build_coda_table('primary')

    # CVC decode
    cvc_decoded = decode_corpus_cvc(all_tokens, assignment, eva_to_triple, coda_table)

    # CV decode (for baseline comparison)
    from voynich.phases.coda_markers import decode_corpus_cv_strip
    cv_decoded = decode_corpus_cv_strip(all_tokens, assignment, eva_to_triple, coda_table)

    # Score
    cvc_vs_old = compute_dict_hit(cvc_decoded, old_dict)
    cvc_vs_cvc = compute_dict_hit(cvc_decoded, cvc_dict)
    cv_vs_old = compute_dict_hit(cv_decoded, old_dict)

    print(f"\n  Scoring:")
    print(f"    CVC decoded vs old dict:     {cvc_vs_old:.4f}")
    print(f"    CVC decoded vs CVC-aware:    {cvc_vs_cvc:.4f}")
    print(f"    CV decoded vs old dict:       {cv_vs_old:.4f} (baseline)")

    # Null comparison
    print("\n  Running null comparison ...")
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_seeds = ([r['seed'] for r in null_data.get('null_runs', [])]
                  if null_data else [100, 101, 102, 103, 104])

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    null_rates: List[float] = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed)
        null_decoded = decode_corpus_cvc(
            null_tokens, assignment, eva_to_triple, coda_table)
        rate = compute_dict_hit(null_decoded, cvc_dict)
        null_rates.append(rate)
        print(f"    Null seed {seed}: {rate:.4f}")

    null_mean = float(np.mean(null_rates)) if null_rates else 0.0
    selectivity = cvc_vs_cvc / null_mean if null_mean > 0 else float('inf')
    print(f"  Null mean: {null_mean:.4f}")
    print(f"  CVC selectivity: {selectivity:.2f}×")

    # Gates
    g1 = cvc_vs_cvc >= 0.35
    g2 = selectivity >= 1.5
    g3 = len(cvc_dict) < 50000
    gates_passed = sum([g1, g2, g3])

    print(f"\n  Validation Gates:")
    print(f"    G1 CVC dict-hit ≥ 35%:     {'PASS' if g1 else 'FAIL'} ({cvc_vs_cvc:.1%})")
    print(f"    G2 selectivity ≥ 1.5×:     {'PASS' if g2 else 'FAIL'} ({selectivity:.2f}×)")
    print(f"    G3 dict size < 50K:        {'PASS' if g3 else 'FAIL'} ({len(cvc_dict)})")
    print(f"    Gates passed: {gates_passed}/3")

    result = CvcDictResult(
        cvc_dict_size=len(cvc_dict),
        old_dict_size=len(old_dict),
        sources=sources,
        cvc_vs_old_dict=round(cvc_vs_old, 4),
        cvc_vs_cvc_dict=round(cvc_vs_cvc, 4),
        cv_vs_old_dict=round(cv_vs_old, 4),
        null_vs_cvc_dict_mean=round(null_mean, 4),
        null_vs_cvc_dict_rates=[round(r, 4) for r in null_rates],
        cvc_selectivity=round(selectivity, 2),
        g1_dict_hit=g1,
        g2_selectivity=g2,
        g3_dict_size=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_dictionary.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 2 completed in {time.time() - t0:.1f}s")
