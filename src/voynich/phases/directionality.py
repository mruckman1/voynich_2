"""
Step 24.10 – Reading Direction and Line-Wrap Analysis
=====================================================
Test whether the manuscript is read strictly left-to-right, or whether
any sections use boustrophedon, right-to-left, or other reading directions.

Dependency chain:
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → directionality.json (this step)
"""

import json
import math
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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SectionDirectionality:
    section: str
    n_lines: int
    n_words: int
    forward_bigram: float
    reversed_bigram: float
    boustrophedon_bigram: float
    best_direction: str  # "forward", "reversed", "boustrophedon"
    forward_vs_reversed_ratio: float
    initial_entropy: float
    final_entropy: float
    medial_entropy: float


@dataclass
class DirectionalityResult:
    timestamp: str
    n_sections: int
    n_total_lines: int
    n_total_words: int
    # Per-section results
    section_results: List[Dict]
    # Corpus-wide
    corpus_forward_bigram: float
    corpus_reversed_bigram: float
    corpus_boustrophedon_bigram: float
    corpus_best_direction: str
    # Entropy analysis
    corpus_initial_entropy: float
    corpus_final_entropy: float
    corpus_medial_entropy: float
    # Verdict
    any_non_forward: bool  # True if any section best != forward
    non_forward_sections: List[str]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Section inference by folio number
# ---------------------------------------------------------------------------

SECTION_RANGES = {
    'herbal_a': (1, 56),
    'pharmaceutical': (57, 66),
    'astronomical': (67, 73),
    'biological': (74, 84),
    'cosmological': (85, 86),
    'zodiac': (87, 101),
    'herbal_b': (102, 116),
}


def _folio_to_section(folio: str) -> str:
    """Map a folio ID (e.g. 'f1r', 'f57v') to a section name."""
    import re
    m = re.search(r'\d+', folio)
    if not m:
        return 'unknown'
    num = int(m.group())
    for section_name, (lo, hi) in SECTION_RANGES.items():
        if lo <= num <= hi:
            return section_name
    return 'unknown'


# ---------------------------------------------------------------------------
# Bigram plausibility helpers
# ---------------------------------------------------------------------------

def _bigram_plausibility(words: List[str], ref_bigrams: set) -> float:
    """Fraction of consecutive word pairs that appear in reference bigrams."""
    if len(words) < 2:
        return 0.0
    hits = sum(1 for i in range(len(words) - 1)
               if (words[i], words[i + 1]) in ref_bigrams)
    return hits / (len(words) - 1)


def _build_ref_bigrams(ref_words: List[str]) -> set:
    """Build a set of (word_i, word_{i+1}) bigrams from reference text."""
    return {(ref_words[i].lower(), ref_words[i + 1].lower())
            for i in range(len(ref_words) - 1)}


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def _entropy(items: List[str]) -> float:
    """Shannon entropy in bits over a list of string items."""
    counts = Counter(items)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total)
                for c in counts.values() if c > 0)


# ---------------------------------------------------------------------------
# R3 combined decode (reused from Phase 16 / Phase 23)
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    max_tokens: int = 0,
) -> List[str]:
    """Decode corpus using R3 combined strategy (alter -> strip -> original)."""
    limit = max_tokens if max_tokens > 0 else len(tokens)
    decoded: List[str] = []
    for token in tokens[:limit]:
        # Try alteration
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt)
            continue

        # Try stripping
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped)
            continue

        # Fall back to original decoding
        original = decode_token(token, assignment, eva_to_triple)
        decoded.append(original)

    return decoded


# ---------------------------------------------------------------------------
# Corpus line extraction
# ---------------------------------------------------------------------------

def _extract_lines_by_section(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Dict[str, List[List[str]]]:
    """
    Extract decoded word-lines grouped by manuscript section.

    Returns a dict: section_name -> list of lines, where each line is a
    list of decoded words.
    """
    section_lines: Dict[str, List[List[str]]] = defaultdict(list)

    for folio_id, page in corpus.pages.items():
        section = page.section
        if section == 'unknown':
            section = _folio_to_section(folio_id)

        # Each locus (paragraph line, label, etc.) is treated as a "line"
        for locus in page.loci:
            text = locus.clean_text
            if not text:
                continue
            raw_tokens = text.split()
            if not raw_tokens:
                continue

            # Decode each token using R3 strategy
            decoded_line: List[str] = []
            for token in raw_tokens:
                alt = decode_token_modifier_aware(
                    token, assignment, eva_to_triple, modifier_chars,
                    modifier_rules=modifier_rules,
                )
                if alt.lower() in ref_word_set:
                    decoded_line.append(alt.lower())
                    continue
                stripped = decode_token_modifier_aware(
                    token, assignment, eva_to_triple, modifier_chars,
                )
                if stripped.lower() in ref_word_set:
                    decoded_line.append(stripped.lower())
                    continue
                original = decode_token(token, assignment, eva_to_triple)
                decoded_line.append(original.lower())

            if decoded_line:
                section_lines[section].append(decoded_line)

    return dict(section_lines)


# ---------------------------------------------------------------------------
# Direction modes: flatten lines into word sequences
# ---------------------------------------------------------------------------

def _flatten_forward(lines: List[List[str]]) -> List[str]:
    """Forward reading: lines in order, words in order within each line."""
    words: List[str] = []
    for line in lines:
        words.extend(line)
    return words


def _flatten_reversed(lines: List[List[str]]) -> List[str]:
    """Reversed reading: each line's word order is reversed (RTL), lines in order."""
    words: List[str] = []
    for line in lines:
        words.extend(reversed(line))
    return words


def _flatten_boustrophedon(lines: List[List[str]]) -> List[str]:
    """Boustrophedon reading: odd lines forward, even lines reversed (0-indexed)."""
    words: List[str] = []
    for i, line in enumerate(lines):
        if i % 2 == 0:
            words.extend(line)
        else:
            words.extend(reversed(line))
    return words


# ---------------------------------------------------------------------------
# Per-section analysis
# ---------------------------------------------------------------------------

def _analyse_section(
    section: str,
    lines: List[List[str]],
    ref_bigrams: set,
) -> SectionDirectionality:
    """Compute directionality metrics for a single section."""

    n_lines = len(lines)
    n_words = sum(len(line) for line in lines)

    # Flatten under each reading mode
    forward_words = _flatten_forward(lines)
    reversed_words = _flatten_reversed(lines)
    boustro_words = _flatten_boustrophedon(lines)

    # Bigram plausibility for each mode
    fwd_bg = _bigram_plausibility(forward_words, ref_bigrams)
    rev_bg = _bigram_plausibility(reversed_words, ref_bigrams)
    bst_bg = _bigram_plausibility(boustro_words, ref_bigrams)

    # Best direction
    scores = {'forward': fwd_bg, 'reversed': rev_bg, 'boustrophedon': bst_bg}
    best_dir = max(scores, key=scores.get)

    # Forward-vs-reversed ratio
    fwd_vs_rev = fwd_bg / rev_bg if rev_bg > 0 else (
        float('inf') if fwd_bg > 0 else 1.0
    )

    # Entropy of line-initial, line-final, and medial tokens
    initial_tokens: List[str] = []
    final_tokens: List[str] = []
    medial_tokens: List[str] = []

    for line in lines:
        if len(line) == 0:
            continue
        initial_tokens.append(line[0])
        if len(line) >= 2:
            final_tokens.append(line[-1])
        if len(line) >= 3:
            medial_tokens.extend(line[1:-1])

    init_ent = _entropy(initial_tokens)
    final_ent = _entropy(final_tokens)
    med_ent = _entropy(medial_tokens) if medial_tokens else 0.0

    return SectionDirectionality(
        section=section,
        n_lines=n_lines,
        n_words=n_words,
        forward_bigram=round(fwd_bg, 6),
        reversed_bigram=round(rev_bg, 6),
        boustrophedon_bigram=round(bst_bg, 6),
        best_direction=best_dir,
        forward_vs_reversed_ratio=round(fwd_vs_rev, 4),
        initial_entropy=round(init_ent, 4),
        final_entropy=round(final_ent, 4),
        medial_entropy=round(med_ent, 4),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_directionality() -> None:
    """Step 24.10: Reading direction and line-wrap analysis."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.10: Reading Direction and Line-Wrap Analysis")
    print("=" * 70)

    rdir = _results_dir()

    # ── 1. Load Phase 15/16 pipeline ──────────────────────────────────────
    print("\n  1. Loading Phase 15/16 pipeline …")

    combined = _load_json(str(rdir / "combined_refine.json"))
    if combined is None:
        print("    [SKIP] combined_refine.json not found — run combined-refine first")
        return
    assignment = combined.get("best_assignment", {})

    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    modifier_chars = set(mod_data.get("modifier_chars", []))
    modifier_rules: Dict[str, str] = {}
    for cls in mod_data.get("classifications", []):
        if cls.get("final_classification") == "modifier":
            modifier_rules[cls["eva_char"]] = cls.get("modifier_type", "silent")

    print(f"    Assignment: {len(assignment)} triple-keys")
    print(f"    Modifiers: {len(modifier_chars)} chars")

    # ── 2. Load corpus ────────────────────────────────────────────────────
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()
    all_tokens = corpus.get_tokens()
    print(f"    {len(all_tokens)} tokens, {len(corpus.pages)} folios")

    # ── 3. Build expanded reference word set ──────────────────────────────
    print("\n  3. Building expanded reference word set …")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()

    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"    {len(ref_word_set)} words in reference set")

    # ── 4. Build reference bigrams ────────────────────────────────────────
    print("\n  4. Building reference bigrams …")
    try:
        ref_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2]
    except Exception:
        ref_words = sorted(base_words)
    ref_bigrams = _build_ref_bigrams(ref_words[:10000])
    print(f"    {len(ref_bigrams)} reference bigrams")

    # ── 5. Extract lines by section ───────────────────────────────────────
    print("\n  5. Extracting and decoding lines by section …")
    section_lines = _extract_lines_by_section(
        corpus, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    total_lines = sum(len(lines) for lines in section_lines.values())
    total_words = sum(
        sum(len(line) for line in lines) for lines in section_lines.values()
    )
    print(f"    {len(section_lines)} sections, {total_lines} lines, "
          f"{total_words} words")

    for sec, lines in sorted(section_lines.items()):
        n_w = sum(len(l) for l in lines)
        print(f"      {sec:<20} {len(lines):>5} lines, {n_w:>6} words")

    # ── 6. Per-section directionality analysis ────────────────────────────
    print("\n  6. Per-section directionality analysis …")
    section_results: List[SectionDirectionality] = []

    for section in sorted(section_lines.keys()):
        lines = section_lines[section]
        result = _analyse_section(section, lines, ref_bigrams)
        section_results.append(result)

        marker = " ***" if result.best_direction != "forward" else ""
        print(f"    {section:<20}  fwd={result.forward_bigram:.6f}  "
              f"rev={result.reversed_bigram:.6f}  "
              f"bst={result.boustrophedon_bigram:.6f}  "
              f"best={result.best_direction}{marker}")
        print(f"      {'':20}  H_init={result.initial_entropy:.2f}  "
              f"H_final={result.final_entropy:.2f}  "
              f"H_medial={result.medial_entropy:.2f}  "
              f"fwd/rev={result.forward_vs_reversed_ratio:.2f}")

    # ── 7. Corpus-wide analysis ───────────────────────────────────────────
    print("\n  7. Corpus-wide analysis …")

    # Combine all lines across sections (preserving section order)
    all_lines: List[List[str]] = []
    for section in sorted(section_lines.keys()):
        all_lines.extend(section_lines[section])

    corpus_result = _analyse_section("corpus_wide", all_lines, ref_bigrams)

    print(f"    Forward bigram:       {corpus_result.forward_bigram:.6f}")
    print(f"    Reversed bigram:      {corpus_result.reversed_bigram:.6f}")
    print(f"    Boustrophedon bigram: {corpus_result.boustrophedon_bigram:.6f}")
    print(f"    Best direction:       {corpus_result.best_direction}")
    print(f"    Forward/Reversed:     {corpus_result.forward_vs_reversed_ratio:.2f}")
    print(f"    H(initial):           {corpus_result.initial_entropy:.2f}")
    print(f"    H(final):             {corpus_result.final_entropy:.2f}")
    print(f"    H(medial):            {corpus_result.medial_entropy:.2f}")

    # ── 8. Identify non-forward sections ──────────────────────────────────
    non_forward = [
        r.section for r in section_results if r.best_direction != "forward"
    ]
    any_non_forward = len(non_forward) > 0

    # ── 9. Verdict ────────────────────────────────────────────────────────
    if not any_non_forward:
        if corpus_result.best_direction == "forward":
            verdict = (
                "UNIFORM FORWARD: All sections and the corpus as a whole "
                "show forward (left-to-right) reading as optimal. No evidence "
                "for boustrophedon or right-to-left reading in any section."
            )
        else:
            verdict = (
                f"CORPUS NON-FORWARD ({corpus_result.best_direction}): "
                f"No individual section favours non-forward reading, but the "
                f"corpus-wide analysis prefers {corpus_result.best_direction}. "
                f"This may indicate subtle inter-section ordering effects."
            )
    else:
        if len(non_forward) == len(section_results):
            verdict = (
                f"ALL SECTIONS NON-FORWARD: Every section prefers non-forward "
                f"reading. Sections: {', '.join(non_forward)}. "
                f"The manuscript may use a non-standard reading direction "
                f"throughout."
            )
        else:
            verdict = (
                f"MIXED DIRECTIONALITY: {len(non_forward)}/{len(section_results)} "
                f"sections show non-forward reading: {', '.join(non_forward)}. "
                f"Forward-reading sections may differ in genre or scribal hand."
            )

    elapsed = time.time() - t0

    print(f"\n  8. Verdict: {verdict}")
    print(f"    Non-forward sections: {non_forward if non_forward else '(none)'}")

    # ── 10. Build and save result ─────────────────────────────────────────
    result = DirectionalityResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        n_sections=len(section_results),
        n_total_lines=total_lines,
        n_total_words=total_words,
        section_results=[_convert(asdict(r)) for r in section_results],
        corpus_forward_bigram=corpus_result.forward_bigram,
        corpus_reversed_bigram=corpus_result.reversed_bigram,
        corpus_boustrophedon_bigram=corpus_result.boustrophedon_bigram,
        corpus_best_direction=corpus_result.best_direction,
        corpus_initial_entropy=corpus_result.initial_entropy,
        corpus_final_entropy=corpus_result.final_entropy,
        corpus_medial_entropy=corpus_result.medial_entropy,
        any_non_forward=any_non_forward,
        non_forward_sections=non_forward,
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = rdir / "directionality.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  → {out_path} ({elapsed:.1f}s)")
