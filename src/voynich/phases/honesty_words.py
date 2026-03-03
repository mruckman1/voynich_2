"""
Phase 17.0.5 – Minimum Viable Word Test
========================================
Tests whether ANY specific Voynich token demonstrably decodes to the Latin
word it should represent based on independent (non-phonetic) evidence.

Test categories:
  1. Rosetta folio plant names (8 folios with plant identifications)
  2. Position-1 verb stems (15 verb candidates → imperatives)
  3. Astronomical section tokens (month/zodiac names)
  4. High-frequency tokens (most common tokens → common Latin words)

Dependency chain:
    rosetta_selection.json    (Phase 6 – plant IDs)
    verb_identification.json  (Phase 9 – verb stems)
    modifier_integrate.json   (Phase 16 modifiers)
    combined_refine.json      (Phase 15 best_assignment)
        → honesty_words.json  (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
)
from voynich.core.reference import (
    LATIN_MONTH_NAMES,
    LATIN_ZODIAC_NAMES,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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


def _reconstruct_modifier_rules(data: Dict) -> Tuple[Set[str], Dict[str, str]]:
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _decode_stem(
    stem: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
) -> Tuple[str, str]:
    """Decode a stem with both modifier-aware and stripped methods."""
    decoded = decode_token_modifier_aware(
        stem, assignment, eva_to_triple, modifier_chars,
        modifier_rules=modifier_rules,
    )
    stripped = decode_token_modifier_aware(
        stem, assignment, eva_to_triple, modifier_chars,
    )
    return decoded, stripped


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WordTest:
    category: str
    voynich_token: str
    expected_latin: str
    decoded_token: str
    decoded_stripped: str
    edit_distance: int
    is_match: bool
    evidence_source: str
    confidence: float


@dataclass
class HonestyWordResult:
    categories: List[str]
    n_tests_total: int

    n_rosetta: int
    rosetta_matches: int
    rosetta_tests: List[Dict]

    n_verb_tests: int
    verb_matches: int
    verb_tests: List[Dict]

    n_astro_tests: int
    astro_matches: int
    astro_tests: List[Dict]

    n_freq_tests: int
    freq_matches: int
    freq_tests: List[Dict]

    total_matches: int
    total_tests: int
    match_rate: float

    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_honesty_words() -> None:
    """Step 17.0.5: Minimum viable word test."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 17.0.5: Minimum Viable Word Test")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load Phase 16 ───
    print("\n  1. Loading Phase 16 results …")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # ─── Load Phase 15 assignment ───
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()

    # ─── Load corpus ───
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    print(f"      {len(tokens)} tokens")

    # ─── Build expanded dictionary ───
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()
    expanded_words, _ = build_expanded_word_set(base_words)
    expanded_set = base_words | expanded_words

    all_tests: List[WordTest] = []

    # ═══════════════════════════════════════════════════════
    # Category 1: Rosetta plant names
    # ═══════════════════════════════════════════════════════
    print("\n  3. Category 1: Rosetta plant names …")
    rosetta_tests: List[WordTest] = []
    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    if os.path.exists(rosetta_path):
        with open(rosetta_path) as f:
            rosetta_data = json.load(f)

        for folio in rosetta_data.get('folio_scores', []):
            stem = folio.get('dominant_stem', '')
            medieval_name = folio.get('medieval_name', '').lower()
            medieval_stem = folio.get('medieval_stem', '').lower()
            folio_id = folio.get('folio', '')

            if not stem or not medieval_name:
                continue

            decoded, stripped = _decode_stem(
                stem, assignment, eva_to_triple, modifier_chars, modifier_rules,
            )

            # Compare against both full name and stem
            # Use proportional threshold for short decoded strings
            best_ed = min(
                _edit_distance(decoded.lower(), medieval_name),
                _edit_distance(decoded.lower(), medieval_stem),
                _edit_distance(stripped.lower(), medieval_name),
                _edit_distance(stripped.lower(), medieval_stem),
            )
            threshold = max(1, len(medieval_stem) // 3)
            is_match = best_ed <= threshold

            wt = WordTest(
                category='rosetta_plant',
                voynich_token=stem,
                expected_latin=f"{medieval_name} (stem: {medieval_stem})",
                decoded_token=decoded,
                decoded_stripped=stripped,
                edit_distance=best_ed,
                is_match=is_match,
                evidence_source=f"{folio_id}: {medieval_name}",
                confidence=folio.get('id_confidence', 0.0),
            )
            rosetta_tests.append(wt)

            marker = '*' if is_match else ' '
            print(f"    {marker} {folio_id}: {stem} → {decoded} / {stripped} "
                  f"(expected: {medieval_name}, ED={best_ed}, threshold={threshold})")
    else:
        print("      [SKIP] rosetta_selection.json not found")

    rosetta_matches = sum(1 for t in rosetta_tests if t.is_match)
    all_tests.extend(rosetta_tests)

    # ═══════════════════════════════════════════════════════
    # Category 2: Position-1 verb stems
    # ═══════════════════════════════════════════════════════
    print("\n  4. Category 2: Position-1 verb stems …")
    verb_tests: List[WordTest] = []
    verb_path = os.path.join(rd, 'verb_identification.json')
    if os.path.exists(verb_path):
        with open(verb_path) as f:
            verb_data = json.load(f)

        for a in verb_data.get('assignments', []):
            stem = a.get('voynich_stem', '')
            imperative = a.get('latin_verb', '')
            if not stem or not imperative:
                continue

            decoded, stripped = _decode_stem(
                stem, assignment, eva_to_triple, modifier_chars, modifier_rules,
            )

            best_ed = min(
                _edit_distance(decoded.lower(), imperative),
                _edit_distance(stripped.lower(), imperative),
            )
            is_match = best_ed <= 1

            wt = WordTest(
                category='verb',
                voynich_token=stem,
                expected_latin=imperative,
                decoded_token=decoded,
                decoded_stripped=stripped,
                edit_distance=best_ed,
                is_match=is_match,
                evidence_source=f"Phase 9 verb: {imperative} ({a.get('latin_meaning', '')})",
                confidence=a.get('total_score', 0.0),
            )
            verb_tests.append(wt)

            marker = '*' if is_match else ' '
            print(f"    {marker} {stem:<12} → {decoded:<15} / {stripped:<15} "
                  f"(expected: {imperative}, ED={best_ed})")
    else:
        print("      [SKIP] verb_identification.json not found")

    verb_matches = sum(1 for t in verb_tests if t.is_match)
    all_tests.extend(verb_tests)

    # ═══════════════════════════════════════════════════════
    # Category 3: Astronomical section tokens
    # ═══════════════════════════════════════════════════════
    print("\n  5. Category 3: Astronomical section tokens …")
    astro_tests: List[WordTest] = []
    expected_names = LATIN_MONTH_NAMES + LATIN_ZODIAC_NAMES

    # Get astronomical section tokens
    astro_tokens = corpus.get_tokens(section='astronomical')
    if not astro_tokens:
        astro_tokens = corpus.get_tokens(section='cosmological')

    if astro_tokens:
        # Get unique tokens sorted by frequency
        astro_freq = Counter(astro_tokens)
        top_astro = [t for t, _ in astro_freq.most_common(50)]

        for token in top_astro:
            decoded, stripped = _decode_stem(
                token, assignment, eva_to_triple, modifier_chars, modifier_rules,
            )

            # Find closest expected name
            best_name = ''
            best_ed = 999
            for name in expected_names:
                ed = min(
                    _edit_distance(decoded.lower(), name),
                    _edit_distance(stripped.lower(), name),
                )
                if ed < best_ed:
                    best_ed = ed
                    best_name = name

            # Only report if edit distance is reasonable
            if best_ed <= max(2, len(best_name) // 3):
                is_match = best_ed <= 1
                wt = WordTest(
                    category='astronomical',
                    voynich_token=token,
                    expected_latin=best_name,
                    decoded_token=decoded,
                    decoded_stripped=stripped,
                    edit_distance=best_ed,
                    is_match=is_match,
                    evidence_source=f"Astronomical section, freq={astro_freq[token]}",
                    confidence=0.5,
                )
                astro_tests.append(wt)

                marker = '*' if is_match else ' '
                print(f"    {marker} {token:<12} → {decoded:<15} / {stripped:<15} "
                      f"(closest: {best_name}, ED={best_ed})")

        if not astro_tests:
            print("      No close matches found in astronomical tokens")
    else:
        print("      No astronomical section tokens found")

    astro_matches = sum(1 for t in astro_tests if t.is_match)
    all_tests.extend(astro_tests)

    # ═══════════════════════════════════════════════════════
    # Category 4: High-frequency tokens
    # ═══════════════════════════════════════════════════════
    print("\n  6. Category 4: High-frequency tokens …")
    freq_tests: List[WordTest] = []

    # Common Latin words expected in any Latin text
    common_latin = [
        'et', 'in', 'est', 'ad', 'de', 'cum', 'non', 'per',
        'aqua', 'oleum', 'folia', 'radix', 'herba', 'mel',
        'recipe', 'misce', 'bene', 'male',
    ]

    token_freq = Counter(tokens)
    top_20 = [t for t, _ in token_freq.most_common(20)]

    for token in top_20:
        decoded, stripped = _decode_stem(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )

        # Check against common Latin words
        best_word = ''
        best_ed = 999
        for word in common_latin:
            ed = min(
                _edit_distance(decoded.lower(), word),
                _edit_distance(stripped.lower(), word),
            )
            if ed < best_ed:
                best_ed = ed
                best_word = word

        # Also check if decoded word is in expanded dictionary
        in_dict = decoded.lower() in expanded_set or stripped.lower() in expanded_set
        is_match = best_ed <= 1

        wt = WordTest(
            category='high_frequency',
            voynich_token=token,
            expected_latin=best_word,
            decoded_token=decoded,
            decoded_stripped=stripped,
            edit_distance=best_ed,
            is_match=is_match,
            evidence_source=f"Rank {top_20.index(token) + 1}, freq={token_freq[token]}",
            confidence=0.3,
        )
        freq_tests.append(wt)

        dict_mark = 'D' if in_dict else ' '
        match_mark = '*' if is_match else ' '
        print(f"    {match_mark}{dict_mark} {token:<12} → {decoded:<15} / {stripped:<15} "
              f"(closest: {best_word}, ED={best_ed}, freq={token_freq[token]})")

    freq_matches = sum(1 for t in freq_tests if t.is_match)
    all_tests.extend(freq_tests)

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    total_matches = sum(1 for t in all_tests if t.is_match)
    total_tests = len(all_tests)

    print(f"\n  7. Summary:")
    print(f"      Rosetta plants:   {rosetta_matches}/{len(rosetta_tests)}")
    print(f"      Verb stems:       {verb_matches}/{len(verb_tests)}")
    print(f"      Astronomical:     {astro_matches}/{len(astro_tests)}")
    print(f"      High-frequency:   {freq_matches}/{len(freq_tests)}")
    print(f"      TOTAL:            {total_matches}/{total_tests}")

    # ─── Gate ───
    gate_passed = total_matches >= 3
    print(f"\n  8. Gate: total_matches >= 3")
    print(f"      total_matches = {total_matches}")
    print(f"      {'PASS' if gate_passed else 'FAIL'}")

    # ─── Verdict ───
    matched_words = [
        f"{t.voynich_token}→{t.decoded_token}≈{t.expected_latin}"
        for t in all_tests if t.is_match
    ]
    if gate_passed:
        verdict = (
            f"PASS: {total_matches} words decoded correctly: "
            f"{', '.join(matched_words[:5])}."
        )
    elif total_matches >= 1:
        verdict = (
            f"MARGINAL: {total_matches} match(es): {', '.join(matched_words)}. "
            f"Below threshold of 3."
        )
    else:
        verdict = (
            f"FAIL: 0 independently-motivated words decoded correctly. "
            f"The phonetic table cannot decode any specific word."
        )

    print(f"\n  Verdict: {verdict}")

    # ─── Save ───
    result = HonestyWordResult(
        categories=['rosetta_plant', 'verb', 'astronomical', 'high_frequency'],
        n_tests_total=total_tests,
        n_rosetta=len(rosetta_tests),
        rosetta_matches=rosetta_matches,
        rosetta_tests=[_convert(asdict(t)) for t in rosetta_tests],
        n_verb_tests=len(verb_tests),
        verb_matches=verb_matches,
        verb_tests=[_convert(asdict(t)) for t in verb_tests],
        n_astro_tests=len(astro_tests),
        astro_matches=astro_matches,
        astro_tests=[_convert(asdict(t)) for t in astro_tests],
        n_freq_tests=len(freq_tests),
        freq_matches=freq_matches,
        freq_tests=[_convert(asdict(t)) for t in freq_tests],
        total_matches=total_matches,
        total_tests=total_tests,
        match_rate=round(total_matches / max(total_tests, 1), 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'honesty_words.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
