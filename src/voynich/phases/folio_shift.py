"""
Phase 10.3 — Folio-Level Encoding Shift Test
===============================================

Rationale
---------
A keyed cipher (H3) might operate at a period longer than line-level.  The key
might change per folio, per quire, or per section, producing systematic
differences between folios' statistical properties beyond what topic explains.

Section strategy:
  This MUST be per-section.  Cross-section comparisons (herbal vs pharma) show
  huge JSD that's entirely topical.  The signal is within-section, across-folio
  comparison: herbal folio 1 vs herbal folio 2, pharma folio 88 vs pharma
  folio 89.  The residual JSD after controlling for topic is the H3 signal.

Sub-analyses
------------
10.3a  Inter-folio bigram matrix comparison (within-section only)
10.3b  Folio-specific frequency shifts (function-word CV)
10.3c  Quire-boundary analysis (controlling for section)
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import (
    bootstrap_ci,
    coefficient_of_variation,
    compare_bigram_matrices,
    word_transition_matrix,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FolioStats:
    folio: str
    section: str
    quire: int
    n_tokens: int


@dataclass
class SectionJSDAnalysis:
    section: str
    n_folios: int
    mean_within_section_jsd: float
    jsd_std: float
    jsd_values: List[float]


@dataclass
class FolioJSDResult:
    per_section: List[Dict]
    overall_within_section_jsd: float
    residual_significant: bool


@dataclass
class FunctionWordCV:
    voynich_mean_cv: float
    voynich_per_stem_cv: Dict[str, float]
    reference_mean_cv: Dict[str, float]
    cv_inflated: bool


@dataclass
class QuireBoundaryResult:
    within_quire_jsd: float
    between_quire_jsd: float
    quire_effect: bool


@dataclass
class FolioShiftResult:
    n_folios_analyzed: int
    jsd_analysis: Dict
    function_word_cv: Dict
    quire_boundary: Dict
    h3_supported: bool
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Convert dataclass/numpy types to JSON-serializable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


MIN_FOLIO_TOKENS = 80  # Minimum tokens per folio for inclusion


def _build_folio_data(corpus) -> Dict[str, List[Dict]]:
    """
    Build per-folio token lists grouped by section.

    Returns {section: [{folio, tokens, quire}, ...]}
    Only includes Language A folios with >= MIN_FOLIO_TOKENS paragraph tokens.
    """
    section_folios: Dict[str, List[Dict]] = defaultdict(list)

    for page in corpus.pages.values():
        if page.language != 'A':
            continue
        tokens = page.paragraph_text.split()
        if len(tokens) < MIN_FOLIO_TOKENS:
            continue
        section_folios[page.section].append({
            'folio': page.folio,
            'tokens': tokens,
            'quire': page.quire,
            'section': page.section,
        })

    return dict(section_folios)


def _folio_bigram_jsd(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute JSD between word-level bigram matrices of two folio token lists."""
    mat_a, alph_a = word_transition_matrix(tokens_a)
    mat_b, alph_b = word_transition_matrix(tokens_b)
    return compare_bigram_matrices(mat_a, mat_b, alph_a, alph_b)


def _within_section_jsd(section_folios: Dict[str, List[Dict]]) -> FolioJSDResult:
    """Compute all-pairs JSD within each section."""
    per_section = []
    all_jsds = []

    for section, folios in section_folios.items():
        if len(folios) < 2:
            continue

        jsds = []
        for i, j in combinations(range(len(folios)), 2):
            jsd = _folio_bigram_jsd(folios[i]['tokens'], folios[j]['tokens'])
            jsds.append(jsd)

        if jsds:
            analysis = SectionJSDAnalysis(
                section=section,
                n_folios=len(folios),
                mean_within_section_jsd=float(np.mean(jsds)),
                jsd_std=float(np.std(jsds)),
                jsd_values=jsds,
            )
            per_section.append(analysis)
            all_jsds.extend(jsds)

    overall = float(np.mean(all_jsds)) if all_jsds else 0.0

    # Bootstrap null: shuffle tokens across folios within section, recompute JSD
    null_jsds = []
    n_boot = 100
    rng = np.random.RandomState(42)
    for section, folios in section_folios.items():
        if len(folios) < 2:
            continue
        all_tokens = []
        folio_sizes = []
        for f in folios:
            all_tokens.extend(f['tokens'])
            folio_sizes.append(len(f['tokens']))

        for _ in range(n_boot):
            rng.shuffle(all_tokens)
            # Redistribute into folio-sized chunks
            fake_folios = []
            idx = 0
            for sz in folio_sizes:
                fake_folios.append(all_tokens[idx:idx + sz])
                idx += sz
            # Compute pairwise JSD for first pair only (for efficiency)
            if len(fake_folios) >= 2:
                jsd_null = _folio_bigram_jsd(fake_folios[0], fake_folios[1])
                null_jsds.append(jsd_null)

    # Is the observed JSD significantly higher than null?
    if null_jsds and all_jsds:
        null_mean = float(np.mean(null_jsds))
        null_std = float(np.std(null_jsds))
        z = (overall - null_mean) / null_std if null_std > 0 else 0.0
        residual_significant = z > 2.0
    else:
        residual_significant = False

    return FolioJSDResult(
        per_section=[_convert(asdict(s)) for s in per_section],
        overall_within_section_jsd=overall,
        residual_significant=residual_significant,
    )


def _identify_function_stems(corpus, top_n: int = 20) -> List[str]:
    """
    Identify function-like stems: those appearing uniformly across sections.
    These are high-frequency stems with low CV across sections.
    """
    tokens_a = corpus.get_tokens(language='A')
    total_counts = Counter(tokens_a)

    # Get per-section counts
    sections = set()
    for page in corpus.pages.values():
        if page.language == 'A':
            sections.add(page.section)

    section_counts: Dict[str, Counter] = {}
    for section in sections:
        section_tokens = corpus.get_tokens(section=section, paragraph_only=True)
        if section_tokens:
            section_counts[section] = Counter(section_tokens)

    # Compute CV across sections for frequent stems
    candidates = [t for t, c in total_counts.most_common(100)]
    stem_cvs = {}
    for stem in candidates:
        freqs = []
        for section, sc in section_counts.items():
            total_in_section = sum(sc.values())
            if total_in_section > 0:
                freqs.append(sc[stem] / total_in_section)
        if len(freqs) >= 2:
            stem_cvs[stem] = coefficient_of_variation(freqs)

    # Function-like = low CV (uniform across sections)
    sorted_stems = sorted(stem_cvs.items(), key=lambda x: x[1])
    return [s for s, _ in sorted_stems[:top_n]]


def _function_word_cv(
    corpus,
    ref_corpus,
    section_folios: Dict[str, List[Dict]],
) -> FunctionWordCV:
    """
    Compare function-word CV across folios (within same section) between
    Voynich and reference languages.
    """
    function_stems = _identify_function_stems(corpus)

    # Compute per-folio frequency of each function stem, WITHIN SECTIONS
    per_stem_cv: Dict[str, float] = {}
    for stem in function_stems:
        freqs = []
        for section, folios in section_folios.items():
            for f in folios:
                total = len(f['tokens'])
                if total > 0:
                    count = f['tokens'].count(stem)
                    freqs.append(count / total)
        if len(freqs) >= 3:
            per_stem_cv[stem] = coefficient_of_variation(freqs)

    voynich_mean_cv = float(np.mean(list(per_stem_cv.values()))) if per_stem_cv else 0.0

    # Reference language function-word CV
    # Use the top 20 most frequent words as "function words"
    ref_mean_cvs: Dict[str, float] = {}
    for lang in ref_corpus.languages:
        ref_texts = ref_corpus.get_texts(lang)
        if len(ref_texts) < 2:
            continue

        # Get per-text token lists
        text_token_lists = [t.tokens for t in ref_texts if len(t.tokens) > 50]
        if len(text_token_lists) < 2:
            continue

        all_ref_tokens = []
        for tl in text_token_lists:
            all_ref_tokens.extend(tl)
        top_ref = [t for t, _ in Counter(all_ref_tokens).most_common(20)]

        cvs = []
        for word in top_ref:
            freqs = []
            for tl in text_token_lists:
                total = len(tl)
                if total > 0:
                    freqs.append(tl.count(word) / total)
            if len(freqs) >= 2:
                cvs.append(coefficient_of_variation(freqs))

        if cvs:
            ref_mean_cvs[lang] = float(np.mean(cvs))

    # CV inflated if Voynich > 1.5x the average reference CV
    ref_avg = float(np.mean(list(ref_mean_cvs.values()))) if ref_mean_cvs else 1.0
    cv_inflated = voynich_mean_cv > 1.5 * ref_avg

    return FunctionWordCV(
        voynich_mean_cv=voynich_mean_cv,
        voynich_per_stem_cv=per_stem_cv,
        reference_mean_cv=ref_mean_cvs,
        cv_inflated=cv_inflated,
    )


def _quire_boundary_test(section_folios: Dict[str, List[Dict]]) -> QuireBoundaryResult:
    """
    Compare within-quire vs between-quire JSD, controlling for section.
    Only compare folios in the same section.
    """
    within_quire_jsds = []
    between_quire_jsds = []

    for section, folios in section_folios.items():
        if len(folios) < 2:
            continue

        for i, j in combinations(range(len(folios)), 2):
            jsd = _folio_bigram_jsd(folios[i]['tokens'], folios[j]['tokens'])
            if folios[i]['quire'] == folios[j]['quire']:
                within_quire_jsds.append(jsd)
            else:
                between_quire_jsds.append(jsd)

    within_mean = float(np.mean(within_quire_jsds)) if within_quire_jsds else -1.0
    between_mean = float(np.mean(between_quire_jsds)) if between_quire_jsds else -1.0

    # Quire effect if between-quire JSD > within-quire JSD (same section)
    if within_quire_jsds and between_quire_jsds:
        quire_effect = between_mean > within_mean * 1.2
    else:
        # Insufficient cross-quire data within same section
        quire_effect = False

    return QuireBoundaryResult(
        within_quire_jsd=within_mean,
        between_quire_jsd=between_mean,
        quire_effect=quire_effect,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_folio_shift() -> Dict[str, Any]:
    """Run Phase 10.3: folio-level encoding shift test."""
    print("=" * 60)
    print("Phase 10.3 — Folio-Level Encoding Shift Test")
    print("=" * 60)

    # --- Load data ---
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)

    # --- Build folio data ---
    print("\n  Building per-folio data (Language A, paragraph text)...")
    section_folios = _build_folio_data(corpus)
    total_folios = sum(len(f) for f in section_folios.values())
    for section, folios in section_folios.items():
        print(f"    {section}: {len(folios)} folios")
    print(f"    Total: {total_folios} folios (>= {MIN_FOLIO_TOKENS} tokens each)")

    # --- Within-section JSD ---
    print("\n  Computing within-section folio pairwise JSD...")
    jsd_result = _within_section_jsd(section_folios)
    print(f"    Overall within-section JSD: {jsd_result.overall_within_section_jsd:.5f}")
    print(f"    Residual significant (vs null): {jsd_result.residual_significant}")
    for s in jsd_result.per_section:
        print(f"      {s['section']}: {s['n_folios']} folios, "
              f"mean JSD={s['mean_within_section_jsd']:.5f}")

    # --- Function-word CV ---
    print("\n  Computing function-word CV...")
    cv_result = _function_word_cv(corpus, ref_corpus, section_folios)
    print(f"    Voynich mean CV: {cv_result.voynich_mean_cv:.3f}")
    for lang, cv in cv_result.reference_mean_cv.items():
        print(f"    {lang} mean CV: {cv:.3f}")
    print(f"    CV inflated: {cv_result.cv_inflated}")

    # --- Quire boundary ---
    print("\n  Testing quire boundary effects...")
    quire_result = _quire_boundary_test(section_folios)
    print(f"    Within-quire JSD:  {quire_result.within_quire_jsd:.5f}")
    print(f"    Between-quire JSD: {quire_result.between_quire_jsd:.5f}")
    print(f"    Quire effect: {quire_result.quire_effect}")

    # --- H3 verdict ---
    h3_evidence = [
        jsd_result.residual_significant,
        cv_result.cv_inflated,
        quire_result.quire_effect,
    ]
    h3_supported = sum(h3_evidence) >= 2  # At least 2 of 3 indicators

    gate_passed = h3_supported or (not any(h3_evidence))  # Clear signal either way

    if h3_supported:
        verdict = (f"folio_shift_supports_H3: residual_jsd={jsd_result.residual_significant}, "
                   f"cv_inflated={cv_result.cv_inflated}, "
                   f"quire_effect={quire_result.quire_effect}")
    elif not any(h3_evidence):
        verdict = "folio_shift_rejects_H3: no systematic encoding shifts detected"
    else:
        verdict = (f"folio_shift_ambiguous: "
                   f"residual_jsd={jsd_result.residual_significant}, "
                   f"cv_inflated={cv_result.cv_inflated}, "
                   f"quire_effect={quire_result.quire_effect}")

    print(f"\n  H3 supported: {h3_supported}")
    print(f"  Gate passed: {gate_passed}")
    print(f"  Verdict: {verdict}")

    # --- Save ---
    result = FolioShiftResult(
        n_folios_analyzed=total_folios,
        jsd_analysis=_convert(asdict(jsd_result)),
        function_word_cv=_convert(asdict(cv_result)),
        quire_boundary=_convert(asdict(quire_result)),
        h3_supported=h3_supported,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'folio_shift.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return out
