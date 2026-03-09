"""
Phase 28.5 – Crib Localization
=================================
Tests whether confirmed crib words cluster on domain-appropriate folios
(plant terms on herbal pages, pharmaceutical verbs on recipe pages, etc.).
Also extracts context windows around confirmed words for inspection.

Dependency chain:
    crib_extraction.json     (Step 28.1)
    signal_isolation.json    (Step 28.4)
    combined_refine.json     (Phase 15 assignment)
    modifier_integrate.json  (Phase 16 modifiers)
        → crib_localization.json  (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import _reconstruct_modifier_rules


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
# Domain expectations
# ---------------------------------------------------------------------------

DOMAIN_EXPECTATIONS: Dict[str, List[str]] = {
    # Plant / botanical terms → herbal sections
    'radi': ['herbal_a', 'herbal_b', 'pharmaceutical'],
    'rami': ['herbal_a', 'herbal_b'],
    'sene': ['herbal_a', 'herbal_b', 'pharmaceutical'],  # senna-related
    'seni': ['herbal_a', 'herbal_b', 'pharmaceutical'],
    # Pharmaceutical verbs → recipes/pharmaceutical
    'cola': ['pharmaceutical', 'recipes'],
    'codi': ['pharmaceutical', 'recipes'],
    'dedi': ['pharmaceutical', 'recipes'],
    # Quality/description terms → everywhere (non-diagnostic)
    'bene': [],      # function word, no section preference
    'de': [],        # preposition
    'sera': [],      # generic adjective
    'sero': [],      # generic
    'rara': [],      # adjective
    'raro': [],      # adverb
    'dira': [],      # adjective
    # Medical / anatomy
    'nera': ['biological', 'pharmaceutical'],
    'hane': [],
    'comi': ['pharmaceutical', 'recipes'],
    'coni': ['pharmaceutical', 'recipes'],
    'dine': [],
    # Zodiac terms
    'sec': ['astronomical'],
    'cor': ['astronomical', 'biological'],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LocalizationEntry:
    word: str
    total_count: int
    section_counts: Dict[str, int]
    peak_section: str
    peak_section_rate: float
    expected_sections: List[str]
    domain_correct: bool            # peak is in expected sections
    chi_sq: float                   # uniformity chi-squared
    context_windows: List[str]      # decoded 3-token windows around hits


@dataclass
class CribLocalizationResult:
    n_words_tested: int
    n_domain_correct: int
    n_domain_diagnostic: int        # words with non-empty expected_sections
    domain_accuracy: float          # correct / diagnostic
    localization_entries: List[Dict]
    best_passage_folio: str
    best_passage_decoded: List[str]
    best_passage_n_hits: int
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _chi_squared_uniformity(counts: Dict[str, int]) -> float:
    """Chi-squared test for uniformity across sections."""
    values = [v for v in counts.values() if v > 0]
    if not values or len(values) <= 1:
        return 0.0
    total = sum(values)
    expected = total / len(values)
    return sum((v - expected) ** 2 / expected for v in values)


def _decode_corpus_r3_with_positions(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 strategy."""
    decoded = []
    for token in tokens:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_crib_localization() -> None:
    """Step 28.5: Crib localization — domain-appropriate placement test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 28.5: Crib Localization")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    crib_path = os.path.join(rd, 'crib_extraction.json')
    if not os.path.exists(crib_path):
        print("  [SKIP] crib_extraction.json not found")
        return
    with open(crib_path) as f:
        crib_data = json.load(f)
    crib_words = [c['word'] for c in crib_data.get('cribs', [])]

    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    print(f"     {len(crib_words)} crib words, {len(assignment)} triples")

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # ── 3. Decode corpus with section info ──
    print("\n  3. Decoding corpus with section tracking …")
    corpus = load_corpus(verbose=False)

    # Build token list with section and folio labels
    token_list: List[str] = []
    section_list: List[str] = []
    folio_list: List[str] = []

    for folio, page in corpus.pages.items():
        from voynich.core.corpus import _infer_section
        section = _infer_section(folio)
        for token in page.all_tokens:
            token_list.append(token)
            section_list.append(section)
            folio_list.append(folio)

    decoded = _decode_corpus_r3_with_positions(
        token_list, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    print(f"     {len(decoded)} tokens decoded")

    # ── 4. Per-word section analysis ──
    print("\n  4. Per-word section analysis …")
    entries: List[LocalizationEntry] = []
    crib_word_set = set(crib_words)

    for word in sorted(crib_word_set):
        # Find all positions where this word appears
        positions = [i for i, w in enumerate(decoded) if w == word]
        total_count = len(positions)

        if total_count == 0:
            entries.append(LocalizationEntry(
                word=word, total_count=0, section_counts={},
                peak_section='none', peak_section_rate=0.0,
                expected_sections=DOMAIN_EXPECTATIONS.get(word, []),
                domain_correct=False, chi_sq=0.0, context_windows=[],
            ))
            continue

        # Count by section
        section_counts: Dict[str, int] = Counter()
        for idx in positions:
            section_counts[section_list[idx]] += 1

        peak_section = max(section_counts, key=section_counts.get)
        peak_rate = section_counts[peak_section] / total_count

        expected = DOMAIN_EXPECTATIONS.get(word, [])
        domain_correct = (peak_section in expected) if expected else True

        chi_sq = _chi_squared_uniformity(dict(section_counts))

        # Extract 3-token context windows (up to 5 examples)
        context_windows = []
        for idx in positions[:5]:
            start = max(0, idx - 1)
            end = min(len(decoded), idx + 2)
            window = decoded[start:end]
            context_windows.append(' '.join(window))

        entries.append(LocalizationEntry(
            word=word,
            total_count=total_count,
            section_counts=dict(section_counts),
            peak_section=peak_section,
            peak_section_rate=round(peak_rate, 3),
            expected_sections=expected,
            domain_correct=domain_correct,
            chi_sq=round(chi_sq, 2),
            context_windows=context_windows,
        ))

    # Print results
    for e in entries:
        tag = '✓' if e.domain_correct else ('○' if not e.expected_sections else '✗')
        print(f"    {tag} {e.word:12s}  n={e.total_count:4d}  "
              f"peak={e.peak_section:16s} ({e.peak_section_rate:.0%})  "
              f"χ²={e.chi_sq:6.1f}")

    # ── 5. Find best passage ──
    print("\n  5. Finding best passage …")
    # Group by folio, find folio with most consecutive dict hits
    best_folio = ''
    best_run = 0
    best_decoded_passage: List[str] = []
    best_passage_hits = 0

    current_folio = ''
    folio_decoded: List[str] = []

    for idx in range(len(decoded)):
        if folio_list[idx] != current_folio:
            # Check previous folio
            if folio_decoded:
                run = 0
                max_run = 0
                run_start = 0
                for j, w in enumerate(folio_decoded):
                    if w in ref_word_set:
                        run += 1
                        if run > max_run:
                            max_run = run
                            run_start = j - run + 1
                    else:
                        run = 0
                if max_run > best_run:
                    best_run = max_run
                    best_folio = current_folio
                    start = max(0, run_start)
                    best_decoded_passage = folio_decoded[start:start + max_run + 2]
                    best_passage_hits = sum(
                        1 for w in folio_decoded if w in ref_word_set
                    )

            current_folio = folio_list[idx]
            folio_decoded = []
        folio_decoded.append(decoded[idx])

    # Check last folio
    if folio_decoded:
        run = 0
        max_run = 0
        for w in folio_decoded:
            if w in ref_word_set:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run > best_run:
            best_run = max_run
            best_folio = current_folio
            best_decoded_passage = folio_decoded[:20]
            best_passage_hits = sum(1 for w in folio_decoded if w in ref_word_set)

    print(f"     Best passage: {best_folio} ({best_run} consecutive hits)")
    if best_decoded_passage:
        print(f"     → {' '.join(best_decoded_passage[:15])}")

    # ── 6. Gate and verdict ──
    diagnostic_entries = [e for e in entries if e.expected_sections]
    n_diagnostic = len(diagnostic_entries)
    n_correct = sum(1 for e in diagnostic_entries if e.domain_correct)
    accuracy = n_correct / n_diagnostic if n_diagnostic > 0 else 0.0

    gate_passed = n_diagnostic == 0 or accuracy >= 0.40
    verdict = (
        f"PASS: {n_correct}/{n_diagnostic} diagnostic words on expected sections "
        f"({accuracy:.0%}). Best passage: {best_folio} ({best_run} consecutive)"
        if gate_passed
        else f"FAIL: Only {n_correct}/{n_diagnostic} diagnostic words correct "
             f"({accuracy:.0%})"
    )
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 7. Save ──
    result = CribLocalizationResult(
        n_words_tested=len(entries),
        n_domain_correct=n_correct,
        n_domain_diagnostic=n_diagnostic,
        domain_accuracy=round(accuracy, 4),
        localization_entries=[_convert(asdict(e)) for e in entries],
        best_passage_folio=best_folio,
        best_passage_decoded=best_decoded_passage[:20],
        best_passage_n_hits=best_passage_hits,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'crib_localization.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
