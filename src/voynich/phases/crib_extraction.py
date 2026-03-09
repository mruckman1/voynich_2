"""
Phase 28.1 – Crib Extraction from Confirmed Words
====================================================
Extracts character-level EVA→syllable assignments from all confirmed
Latin dictionary hits across Phase 14 (feature CSP), Phase 19.8
(cross-approach), and Phase 26 (zodiac quality terms).

Each crib word is decomposed into EVA characters, mapped through the
stroke-triple lookup, and aligned with the Latin syllables the
assignment table produced.  The result is a tiered crib pool that
downstream steps use for consistency testing and signal isolation.

Dependency chain:
    feature_csp.json          (Phase 14 assignment)
    feature_decode.json       (Phase 14 confirmed hits)
    cross_approach.json       (Phase 19.8 bidirectional mappings)
    astro_crib.json           (Phase 26 zodiac quality terms)
        → crib_extraction.json  (this step)
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
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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
class CharAlignment:
    """Single EVA character → syllable alignment within a crib word."""
    eva_char: str
    triple_key: str
    syllable: str


@dataclass
class CribEntry:
    """A single confirmed crib word with its provenance and alignments."""
    word: str
    sources: List[str]
    n_sources: int
    tier: int                          # 1, 2, or 3
    alignments: List[CharAlignment]
    triples_covered: List[str]
    syllables_covered: List[str]
    corpus_count: int
    example_tokens: List[str]          # EVA tokens that decode to this word
    aligned: bool                      # True if char↔syllable alignment succeeded
    notes: str


@dataclass
class CribExtractionResult:
    n_cribs: int
    n_tier1: int
    n_tier2: int
    n_tier3: int
    cribs: List[Dict]
    all_triples_covered: List[str]     # union of triples across all Tier 1+2 cribs
    all_triples_unconfirmed: List[str] # 25 triples minus covered
    char_to_syllable: Dict[str, str]   # eva_char → syllable (from Phase 14 table)
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Crib loading helpers
# ---------------------------------------------------------------------------

def _decode_token_basic(token: str, assignment: Dict[str, str],
                        eva_to_triple: Dict[str, str]) -> str:
    """Decode an EVA token using the triple→syllable assignment (no modifiers)."""
    chars = tokenize_eva_chars(token)
    parts = []
    for ch in chars:
        triple = eva_to_triple.get(ch)
        if triple:
            parts.append(assignment.get(triple, '?'))
        else:
            parts.append('?')
    return ''.join(parts)


def _find_tokens_for_word(
    word: str,
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> List[str]:
    """Find all EVA tokens in the corpus that decode to the given word."""
    hits = []
    for token in set(all_tokens):
        decoded = _decode_token_basic(token, assignment, eva_to_triple)
        if decoded.lower() == word.lower():
            hits.append(token)
    return sorted(hits)


def _align_chars_to_syllables(
    token: str,
    word: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> Optional[List[CharAlignment]]:
    """Align EVA characters in a token to syllables in the decoded word.

    Returns a list of CharAlignment if the number of syllabic (triple-mapped)
    chars matches the number of syllables, else None.
    """
    chars = tokenize_eva_chars(token)
    # Each char with a triple produces one syllable
    syllabic_chars = []
    for ch in chars:
        triple = eva_to_triple.get(ch)
        if triple and triple in assignment:
            syllabic_chars.append((ch, triple))

    # The decoded word is the concatenation of syllables from the assignment
    # Reconstruct syllable list from the assignment for this token
    syllables = []
    for ch, triple in syllabic_chars:
        syllables.append(assignment[triple])

    decoded = ''.join(syllables)
    if decoded.lower() != word.lower():
        return None

    alignments = []
    for (ch, triple), syl in zip(syllabic_chars, syllables):
        alignments.append(CharAlignment(eva_char=ch, triple_key=triple, syllable=syl))
    return alignments


def _load_phase14_hits(rd: str) -> Tuple[List[str], Dict[str, str]]:
    """Load Phase 14 confirmed hits and assignment.

    Returns:
        (confirmed_words, phase14_assignment)
    """
    # Load confirmed hits
    decode_path = os.path.join(rd, 'feature_decode.json')
    if not os.path.exists(decode_path):
        print("  [WARN] feature_decode.json not found")
        return [], {}
    with open(decode_path) as f:
        decode_data = json.load(f)

    confirmed = decode_data.get('vocabulary_catalog', {}).get('confirmed_hits', [])

    # Load Phase 14 assignment
    csp_path = os.path.join(rd, 'feature_csp.json')
    if not os.path.exists(csp_path):
        print("  [WARN] feature_csp.json not found")
        return confirmed, {}
    with open(csp_path) as f:
        csp_data = json.load(f)

    assignment = {}
    lang_results = csp_data.get('language_results', {}).get('latin', {})
    assignment = lang_results.get('best_assignment', {})
    if not assignment:
        assignment = csp_data.get('best_assignment', {})

    return confirmed, assignment


def _load_cross_approach(rd: str) -> List[Dict]:
    """Load Phase 19.8 cross-approach per-word results."""
    path = os.path.join(rd, 'cross_approach.json')
    if not os.path.exists(path):
        print("  [WARN] cross_approach.json not found")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get('per_word_results', [])


def _load_zodiac_hits(rd: str) -> List[Dict]:
    """Load Phase 26 astro crib vocab hits."""
    path = os.path.join(rd, 'astro_crib.json')
    if not os.path.exists(path):
        print("  [WARN] astro_crib.json not found")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get('vocab_hits', [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_crib_extraction() -> None:
    """Step 28.1: Extract character-level assignments from confirmed cribs."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 28.1: Crib Extraction from Confirmed Words")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load Phase 14 confirmed hits ──
    print("\n  1. Loading Phase 14 confirmed hits …")
    confirmed_words, phase14_assignment = _load_phase14_hits(str(rd))
    print(f"     {len(confirmed_words)} confirmed words, "
          f"{len(phase14_assignment)} triple assignments")

    if not phase14_assignment:
        print("  [ABORT] No Phase 14 assignment available")
        return

    # ── 2. Load corpus and find tokens for each hit ──
    print("\n  2. Scanning corpus for crib tokens …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    token_counter = Counter(all_tokens)

    # Build word→source mapping
    word_sources: Dict[str, Set[str]] = {}
    for w in confirmed_words:
        word_sources.setdefault(w, set()).add('phase14')

    # ── 3. Load Phase 19.8 cross-approach ──
    print("\n  3. Loading Phase 19.8 cross-approach words …")
    cross_results = _load_cross_approach(str(rd))
    n_exact = 0
    n_edit2 = 0
    for entry in cross_results:
        w = entry.get('latin_word', '')
        if entry.get('exact_match'):
            word_sources.setdefault(w, set()).add('phase19_exact')
            n_exact += 1
        elif entry.get('edit2_match'):
            word_sources.setdefault(w, set()).add('phase19_edit2')
            n_edit2 += 1
    print(f"     {n_exact} exact, {n_edit2} edit≤2 matches")

    # ── 4. Load Phase 26 zodiac quality hits ──
    print("\n  4. Loading Phase 26 zodiac quality hits …")
    zodiac_hits = _load_zodiac_hits(str(rd))
    zodiac_words = set()
    for hit in zodiac_hits:
        variant = hit.get('found_variant', '')
        if variant and hit.get('on_correct_folio'):
            zodiac_words.add(variant)
            word_sources.setdefault(variant, set()).add('phase26_zodiac')
    print(f"     {len(zodiac_words)} unique zodiac-confirmed variants: "
          f"{sorted(zodiac_words)}")

    # ── 5. Build crib pool ──
    print("\n  5. Building crib pool with character alignments …")
    cribs: List[CribEntry] = []
    all_triples_covered: Set[str] = set()

    for word, sources in sorted(word_sources.items()):
        # Find tokens that decode to this word
        tokens_for_word = _find_tokens_for_word(
            word, all_tokens, phase14_assignment, eva_to_triple,
        )
        corpus_count = sum(token_counter[t] for t in tokens_for_word)

        # Determine tier
        n_independent = len(sources - {'phase19_edit2'})
        if n_independent >= 2:
            tier = 1
        elif 'phase14' in sources and corpus_count >= 5:
            tier = 2
        else:
            tier = 3

        # Align characters to syllables using first matching token
        alignments: List[CharAlignment] = []
        aligned = False
        for tok in tokens_for_word:
            aligns = _align_chars_to_syllables(tok, word, phase14_assignment,
                                                eva_to_triple)
            if aligns is not None:
                alignments = aligns
                aligned = True
                break

        triples = [a.triple_key for a in alignments]
        syllables = [a.syllable for a in alignments]
        if tier <= 2:
            all_triples_covered.update(triples)

        notes_parts = []
        if not tokens_for_word:
            notes_parts.append("no corpus tokens decode to this word")
        if not aligned and tokens_for_word:
            notes_parts.append("alignment failed (syllable count mismatch)")

        cribs.append(CribEntry(
            word=word,
            sources=sorted(sources),
            n_sources=len(sources),
            tier=tier,
            alignments=alignments,
            triples_covered=triples,
            syllables_covered=syllables,
            corpus_count=corpus_count,
            example_tokens=tokens_for_word[:5],
            aligned=aligned,
            notes='; '.join(notes_parts) if notes_parts else '',
        ))

    # Sort by tier then word
    cribs.sort(key=lambda c: (c.tier, c.word))

    n_tier1 = sum(1 for c in cribs if c.tier == 1)
    n_tier2 = sum(1 for c in cribs if c.tier == 2)
    n_tier3 = sum(1 for c in cribs if c.tier == 3)

    # Unconfirmed triples
    all_25 = set(phase14_assignment.keys())
    unconfirmed = sorted(all_25 - all_triples_covered)

    # Build char→syllable map from Phase 14
    char_to_syl: Dict[str, str] = {}
    for glyph, components in EVA_VISUAL_COMPONENTS.items():
        triple = (components['first_stroke'] + ',' +
                  components['last_stroke'] + ',' +
                  components['glyph_class'])
        if triple in phase14_assignment:
            char_to_syl[glyph] = phase14_assignment[triple]

    # ── 6. Report ──
    print(f"\n  Crib pool: {len(cribs)} words")
    print(f"    Tier 1 (cross-source): {n_tier1}")
    print(f"    Tier 2 (Phase 14, freq≥5): {n_tier2}")
    print(f"    Tier 3 (low-freq or edit2-only): {n_tier3}")
    print(f"    Triples covered by Tier 1+2: {len(all_triples_covered)}/25")
    print(f"    Unconfirmed triples: {len(unconfirmed)}")

    for c in cribs:
        tag = {1: '★', 2: '●', 3: '○'}[c.tier]
        src = ','.join(c.sources)
        print(f"      {tag} {c.word:12s} tier={c.tier}  "
              f"count={c.corpus_count:4d}  aligned={c.aligned}  [{src}]")

    gate_passed = n_tier1 + n_tier2 >= 10
    verdict = (
        f"PASS: {n_tier1} Tier-1 + {n_tier2} Tier-2 cribs, "
        f"{len(all_triples_covered)}/25 triples covered"
        if gate_passed
        else f"FAIL: Only {n_tier1 + n_tier2} Tier-1+2 cribs (need ≥10)"
    )
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 7. Save ──
    result = CribExtractionResult(
        n_cribs=len(cribs),
        n_tier1=n_tier1,
        n_tier2=n_tier2,
        n_tier3=n_tier3,
        cribs=[_convert(asdict(c)) for c in cribs],
        all_triples_covered=sorted(all_triples_covered),
        all_triples_unconfirmed=unconfirmed,
        char_to_syllable=char_to_syl,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'crib_extraction.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
