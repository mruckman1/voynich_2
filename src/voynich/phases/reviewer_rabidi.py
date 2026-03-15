"""
Reviewer Analysis 2: Results With and Without *rabidi*
======================================================

Tests whether the 22 T1 word-level identifications are robust to
removing the single most frequent word (rabidi, 5 of 22 entries).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Set

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus


def _convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _compute_stats(
    entries: List[Dict],
    signal_token_count: int,
    n_corpus_tokens: int,
    ci_vocab: Set[str],
    label: str,
) -> Dict:
    """Compute statistics for a set of T1 catalog entries."""
    eva_types = set(e["eva_type"] for e in entries)
    latin_words = set(e["latin_word"] for e in entries)
    folios: Set[str] = set()
    for e in entries:
        folios.update(e.get("folios", []))

    # Catalog token count from pre-computed total_corpus_count per entry
    catalog_token_count = sum(e.get("total_corpus_count", 0) for e in entries)

    # Coverage: signal word tokens + catalog tokens (may overlap slightly)
    total_glossed = signal_token_count + catalog_token_count
    coverage = total_glossed / n_corpus_tokens if n_corpus_tokens else 0.0

    # Circa Instans overlap
    ci_overlap = (len(latin_words & ci_vocab) / len(latin_words)
                  if latin_words else 0.0)

    # Domain coherence: classify each latin word
    pharma_terms = {
        "ratione", "stercora", "radicom", "diasene", "coralli",
        "commune", "codex", "secundi",
    }
    pharma_count = len(latin_words & pharma_terms)
    pharma_fraction = pharma_count / len(latin_words) if latin_words else 0.0

    return {
        "label": label,
        "n_entries": len(entries),
        "n_distinct_words": len(latin_words),
        "distinct_words": sorted(latin_words),
        "n_eva_types": len(eva_types),
        "eva_types": sorted(eva_types),
        "n_folios": len(folios),
        "catalog_token_count": catalog_token_count,
        "signal_token_count": signal_token_count,
        "corpus_coverage": coverage,
        "ci_overlap": ci_overlap,
        "pharma_fraction": pharma_fraction,
    }


def run_reviewer_rabidi() -> None:
    t0 = time.time()
    rd = _results_dir()

    # ── Load word catalog ──
    catalog_path = os.path.join(rd, "word_catalog.json")
    if not os.path.exists(catalog_path):
        print("ERROR: word_catalog.json not found")
        return
    with open(catalog_path) as f:
        catalog = json.load(f)

    t1_entries = [e for e in catalog["single_token_ids"] if e.get("tier") == "T1"]
    signal_words = catalog.get("signal_words", [])
    signal_decoded = set(sw["decoded"] for sw in signal_words)

    # ── Load paradigms ──
    valid_path = os.path.join(rd, "word_validation.json")
    paradigms = []
    if os.path.exists(valid_path):
        with open(valid_path) as f:
            valid_data = json.load(f)
        paradigms = valid_data.get("paradigms", [])

    # ── Corpus size and signal token count ──
    n_corpus_tokens = catalog.get("n_corpus_tokens", 36238)
    # Signal word tokens: sum of real_count across all 70 signal words
    signal_token_count = sum(sw.get("real_count", 0) for sw in signal_words)

    # ── Load Circa Instans vocabulary ──
    try:
        ref_corpus = load_reference_corpus(languages=["latin"], verbose=False)
        ci_vocab = set(
            w.lower()
            for w in ref_corpus.get_combined_tokens("latin")
            if len(w) >= 2
        )
    except Exception:
        ci_vocab = set()

    # ── Partition ──
    rabidi_entries = [e for e in t1_entries if e["latin_word"] == "rabidi"]
    non_rabidi_entries = [e for e in t1_entries if e["latin_word"] != "rabidi"]

    # ── Compute stats ──
    # Signal token count is the same for both (signal words are independent
    # of the T1 catalog — they come from Phase 36)
    stats_all = _compute_stats(
        t1_entries, signal_token_count, n_corpus_tokens, ci_vocab, "all_22"
    )
    stats_no_rabidi = _compute_stats(
        non_rabidi_entries, signal_token_count, n_corpus_tokens, ci_vocab,
        "without_rabidi"
    )

    # ── Paradigm impact ──
    rabidi_eva = set(e["eva_type"] for e in rabidi_entries)
    paradigms_with_rabidi = [
        p for p in paradigms
        if any(m in rabidi_eva for m in p.get("eva_types", []))
    ]
    paradigms_without = [
        p for p in paradigms
        if not any(m in rabidi_eva for m in p.get("eva_types", []))
    ]

    # ── Impact summary ──
    coverage_delta = stats_no_rabidi["corpus_coverage"] - stats_all["corpus_coverage"]
    folio_delta = stats_no_rabidi["n_folios"] - stats_all["n_folios"]
    paradigm_delta = len(paradigms_without) - len(paradigms)

    # ── Verdict ──
    coverage_drop = abs(coverage_delta)
    ci_ok = stats_no_rabidi["ci_overlap"] >= 0.60
    words_ok = stats_no_rabidi["n_distinct_words"] >= 7
    if coverage_drop < 0.05 and ci_ok and words_ok:
        verdict = "ROBUST"
    else:
        verdict = "FRAGILE"

    result = {
        "test": "rabidi_sensitivity",
        "with_rabidi": stats_all,
        "without_rabidi": stats_no_rabidi,
        "rabidi_entries": {
            "n_entries": len(rabidi_entries),
            "eva_types": sorted(rabidi_eva),
            "folios": sorted(set().union(*(set(e.get("folios", []))
                                           for e in rabidi_entries))),
        },
        "paradigm_impact": {
            "total_paradigms": len(paradigms),
            "paradigms_with_rabidi": len(paradigms_with_rabidi),
            "paradigms_without_rabidi": len(paradigms_without),
            "affected_stems": [p["stem"] for p in paradigms_with_rabidi],
        },
        "impact": {
            "coverage_delta": coverage_delta,
            "folio_delta": folio_delta,
            "paradigm_delta": paradigm_delta,
            "words_lost": sorted(
                set(e["latin_word"] for e in rabidi_entries)
            ),
        },
        "verdict": verdict,
        "interpretation": (
            f"Removing all {len(rabidi_entries)} rabidi identifications "
            f"reduces the T1 catalog to {len(non_rabidi_entries)} entries "
            f"across {stats_no_rabidi['n_distinct_words']} distinct words. "
            f"Corpus coverage changes by {coverage_delta:+.1%} "
            f"({stats_no_rabidi['corpus_coverage']:.1%} vs "
            f"{stats_all['corpus_coverage']:.1%}). "
            f"CI overlap: {stats_no_rabidi['ci_overlap']:.1%} vs "
            f"{stats_all['ci_overlap']:.1%}. "
            f"Verdict: {verdict}."
        ),
        "runtime_seconds": time.time() - t0,
    }

    # ── Print summary ──
    print("=" * 70)
    print("REVIEWER ANALYSIS 2: Rabidi Sensitivity")
    print("=" * 70)
    print(f"\n{'Metric':<30s}  {'All 22 T1':>12s}  {'Without rabidi':>14s}")
    print("-" * 60)
    print(f"{'Entries':<30s}  {stats_all['n_entries']:>12d}  "
          f"{stats_no_rabidi['n_entries']:>14d}")
    print(f"{'Distinct Latin words':<30s}  {stats_all['n_distinct_words']:>12d}  "
          f"{stats_no_rabidi['n_distinct_words']:>14d}")
    print(f"{'EVA types':<30s}  {stats_all['n_eva_types']:>12d}  "
          f"{stats_no_rabidi['n_eva_types']:>14d}")
    print(f"{'Folios covered':<30s}  {stats_all['n_folios']:>12d}  "
          f"{stats_no_rabidi['n_folios']:>14d}")
    print(f"{'Corpus coverage':<30s}  {stats_all['corpus_coverage']:>11.1%}  "
          f"{stats_no_rabidi['corpus_coverage']:>13.1%}")
    print(f"{'CI overlap':<30s}  {stats_all['ci_overlap']:>11.1%}  "
          f"{stats_no_rabidi['ci_overlap']:>13.1%}")
    print(f"{'Paradigms':<30s}  {len(paradigms):>12d}  "
          f"{len(paradigms_without):>14d}")
    print(f"\nVerdict: {verdict}")
    print(f"\nRabidi entries removed: {len(rabidi_entries)}")
    print(f"Rabidi EVA types: {sorted(rabidi_eva)}")
    print(f"Words in catalog: {sorted(stats_all['distinct_words'])}")
    print(f"Words remaining:  {sorted(stats_no_rabidi['distinct_words'])}")

    # ── Save ──
    out_path = os.path.join(rd, "reviewer_rabidi.json")
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\nSaved to {out_path}")
