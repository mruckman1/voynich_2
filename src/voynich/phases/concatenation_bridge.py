"""
Phase 51 Track B: Concatenation Bridge Search
==============================================
Use signal words as anchors, partially decode adjacent dark tokens
via confirmed-triple characters, and search pharmaceutical dictionaries
for pattern matches that constrain free-triple assignments.

Dependency chain:
    signal_bigrams.json        (Step 29.1 -- per-token decoded + classifications)
    combined_refine.json       (Step 15   -- best_assignment)
    modifier_integrate.json    (Step 16   -- modifier chars)
    triple_tiers.json          (Step 44   -- confirmed / landscape tiers)
        -> concatenation_bridge.json  (this step)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHARMACEUTICAL_VOCABULARY,
    generate_inflected_forms,
    generate_medieval_variants,
    load_reference_corpus,
)

from voynich.phases.suffix_calibration import SIGNAL_WORDS_SET


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


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CoverageAudit:
    n_dark_tokens: int
    n_with_1plus_confirmed: int
    n_with_2plus_confirmed: int
    coverage_rate: float
    mean_confirmed_fraction: float
    distribution: Dict[str, int]    # "0%", "1-25%", "26-50%", "51-75%", "76-100%"
    gate_passed: bool


@dataclass
class BridgeMatch:
    token_idx: int
    token_eva: str
    pattern: str
    matched_word: str
    n_confirmed_chars: int
    n_free_chars: int
    implied_assignments: Dict[str, str]   # triple_key -> syllable
    anchor_word: str
    anchor_position: str       # 'before' or 'after'
    distance: int              # 1 = adjacent, 2 = skip-one
    folio: str
    n_total_matches: int       # how many dict words matched this pattern


@dataclass
class ConsensusAssignment:
    triple_key: str
    implied_syllable: str
    consensus: float
    n_observations: int
    agrees_with_phase15: bool
    source_words: List[str]


@dataclass
class ConcatenationBridgeResult:
    # Coverage audit
    coverage: Dict[str, Any]
    # Search results
    n_signal_anchors_used: int
    n_dark_tokens_examined: int
    n_with_partial_decode: int
    n_bridge_matches: int
    n_unique_words_found: int
    top_bridge_matches: List[Dict]
    # Concatenation search
    n_concat_pairs_tested: int
    n_concat_matches: int
    top_concat_matches: List[Dict]
    # Consensus
    consensus_assignments: List[Dict]
    n_new_assignments: int
    # Null test
    null_mean_matches: float
    null_std_matches: float
    bridge_selectivity: float
    bridge_z_score: float
    # Dictionary info
    pharma_dict_size: int
    confirmed_triple_count: int
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Dictionary building
# ---------------------------------------------------------------------------

def _build_pharma_dict() -> Set[str]:
    """Build a small focused pharmaceutical dictionary (~5K words)."""
    words: Set[str] = set()

    # 1. Direct pharmaceutical vocabulary
    for domain, terms in PHARMACEUTICAL_VOCABULARY.items():
        for w in terms:
            words.add(w.lower())

    # 2. Inflected forms of pharmaceutical stems
    pharma_stems = {
        # verbs (1st conjugation stems)
        'col': 'verb1', 'dist': 'verb1', 'misc': 'verb2',
        'add': 'verb3', 'solv': 'verb3', 'coqu': 'verb3',
        'ter': 'verb3', 'lav': 'verb1', 'pon': 'verb3',
        # nouns
        'herb': 'noun1', 'foli': 'noun2n', 'radic': 'noun3',
        'flo': 'noun3', 'semin': 'noun3', 'cortic': 'noun3',
        'aqu': 'noun1', 'ole': 'noun2n', 'vin': 'noun2n',
        'mel': 'noun3', 'pulv': 'noun3', 'unguent': 'noun2n',
        'sirup': 'noun2', 'decoct': 'noun2n', 'infus': 'noun2n',
        'caput': 'noun3', 'stomach': 'noun2', 'ocul': 'noun2',
        'pector': 'noun3', 'ventr': 'noun3',
    }
    for stem, pos in pharma_stems.items():
        for form in generate_inflected_forms(stem, pos):
            words.add(form.lower())

    # 3. Medieval spelling variants of base words
    base_words = list(words)
    for w in base_words:
        for variant in generate_medieval_variants(w).keys():
            words.add(variant.lower())

    # 4. Reference corpus words (Latin, length 3-10, common)
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        word_freq: Counter = Counter()
        for text in ref.get_texts('latin'):
            for tok in text.tokens:
                w = tok.lower()
                if 3 <= len(w) <= 10 and w.isalpha():
                    word_freq[w] += 1
        for w, freq in word_freq.items():
            if freq >= 2:
                words.add(w)
    except (FileNotFoundError, Exception):
        pass  # Reference corpus not available; proceed with pharma vocab only

    # Filter to reasonable lengths
    words = {w for w in words if 2 <= len(w) <= 12}
    return words


# ---------------------------------------------------------------------------
# Partial decode
# ---------------------------------------------------------------------------

def _build_partial_decode(
    eva_token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    confirmed_triples: Set[str],
) -> Tuple[str, List[Tuple[int, str, str, bool]]]:
    """Partially decode a token using only confirmed triples.

    Returns (pattern_string, char_details) where:
    - pattern_string: e.g. "di??ne" with ? for free-triple chars
    - char_details: [(pos, eva_char, syllable_or_?, is_confirmed)]
    """
    chars = tokenize_eva_chars(eva_token)
    details = []
    pattern_parts = []
    pos = 0

    for ch in chars:
        if ch in modifier_chars:
            continue  # Skip modifiers

        triple = eva_to_triple.get(ch)
        if triple is None:
            pattern_parts.append('?')
            details.append((pos, ch, '?', False))
        elif triple in confirmed_triples:
            syl = assignment.get(triple, '?')
            pattern_parts.append(syl)
            details.append((pos, ch, syl, True))
        else:
            pattern_parts.append('?')
            details.append((pos, ch, '?', False))
        pos += 1

    return ''.join(pattern_parts), details


def _pattern_to_regex(pattern: str) -> Optional[re.Pattern]:
    """Convert a partial decode pattern to a regex.

    Each '?' represents one unknown CV syllable (1-3 chars).
    """
    if '?' not in pattern:
        return None  # Fully decoded, no search needed

    # Split on ? marks, escape literal parts
    parts = pattern.split('?')
    regex_parts = []
    for i, part in enumerate(parts):
        if part:
            regex_parts.append(re.escape(part))
        if i < len(parts) - 1:
            regex_parts.append('.{1,3}')

    regex_str = '^' + ''.join(regex_parts) + '$'
    try:
        return re.compile(regex_str)
    except re.error:
        return None


def _search_dict(
    pattern: str,
    pharma_dict: Set[str],
    max_matches: int = 50,
) -> List[str]:
    """Search the dictionary for words matching a partial decode pattern."""
    regex = _pattern_to_regex(pattern)
    if regex is None:
        # Fully decoded — check if it's in the dict
        return [pattern] if pattern in pharma_dict else []

    matches = []
    # Estimate min/max word lengths from the pattern
    n_wildcards = pattern.count('?')
    known_len = len(pattern) - n_wildcards
    min_len = known_len + n_wildcards * 1
    max_len = known_len + n_wildcards * 3

    for word in pharma_dict:
        if min_len <= len(word) <= max_len:
            if regex.match(word):
                matches.append(word)
                if len(matches) >= max_matches:
                    break

    return matches


def _extract_implied_assignments(
    pattern: str,
    matched_word: str,
    char_details: List[Tuple[int, str, str, bool]],
    eva_to_triple: Dict[str, str],
) -> Dict[str, str]:
    """Extract implied triple→syllable assignments from a pattern match.

    Align the pattern (with ?s) to the matched word and extract what
    each ? must be.
    """
    # This is tricky because ? can match 1-3 chars.
    # Simple approach: if the pattern has exactly 1 ?, the implied syllable
    # is unambiguous. For 2+ ?s, alignment is ambiguous — skip.
    n_wildcards = pattern.count('?')
    if n_wildcards != 1:
        return {}  # Only handle unambiguous single-wildcard cases

    # Find the position and length of the wildcard
    q_pos = pattern.index('?')
    prefix = pattern[:q_pos]
    suffix = pattern[q_pos + 1:]

    # The matched word must start with prefix and end with suffix
    if not matched_word.startswith(prefix) or not matched_word.endswith(suffix):
        return {}
    if suffix:
        implied = matched_word[len(prefix):-len(suffix)]
    else:
        implied = matched_word[len(prefix):]

    if not implied or len(implied) > 3:
        return {}

    # Find which EVA char was the wildcard
    free_chars = [(pos, ch, syl, conf) for pos, ch, syl, conf in char_details
                  if not conf]
    if len(free_chars) != 1:
        return {}

    _, eva_ch, _, _ = free_chars[0]
    triple = eva_to_triple.get(eva_ch)
    if triple is None:
        return {}

    return {triple: implied}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_concatenation_bridge() -> None:
    """Phase 51 Track B: Concatenation Bridge Search."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 51 TRACK B: Concatenation Bridge Search")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ────────────────────────────────────────────────
    print("\n  B.1  Loading inputs...")

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        bigram_data = json.load(f)
    token_evas = bigram_data['token_evas']
    token_decoded = bigram_data['token_decoded']
    token_classifications = bigram_data['token_classifications']
    token_folios = bigram_data['token_folios']
    n_tokens = len(token_evas)

    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = refine_data['best_assignment']

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars = set(mod_data['modifier_chars'])

    with open(os.path.join(rd, 'triple_tiers.json')) as f:
        tiers_data = json.load(f)

    # Extract confirmed triple keys
    confirmed_triples: Set[str] = set()
    landscape_triples: Set[str] = set()
    for entry in tiers_data['tiers'].get('CONFIRMED', []):
        confirmed_triples.add(entry['triple_key'])
    for entry in tiers_data['tiers'].get('LANDSCAPE_CONFIRMED', []):
        landscape_triples.add(entry['triple_key'])

    eva_to_triple = build_eva_to_triple_lookup()

    print(f"       {n_tokens} tokens")
    print(f"       {len(assignment)} triple assignments")
    print(f"       {len(modifier_chars)} modifier chars")
    print(f"       {len(confirmed_triples)} CONFIRMED triples")
    print(f"       {len(landscape_triples)} LANDSCAPE_CONFIRMED triples")

    # ── 2. Coverage audit ─────────────────────────────────────────────
    print("\n  B.2  Confirmed-triple coverage audit...")

    n_dark = 0
    n_with_1 = 0
    n_with_2 = 0
    fractions = []
    buckets = {"0%": 0, "1-25%": 0, "26-50%": 0, "51-75%": 0, "76-100%": 0}

    for i in range(n_tokens):
        cls = token_classifications[i]
        if cls == 'SIGNAL':
            continue  # Only audit dark tokens
        n_dark += 1

        chars = tokenize_eva_chars(token_evas[i])
        n_conf = 0
        n_free = 0
        for ch in chars:
            if ch in modifier_chars:
                continue
            triple = eva_to_triple.get(ch)
            if triple and triple in confirmed_triples:
                n_conf += 1
            else:
                n_free += 1

        total = n_conf + n_free
        frac = n_conf / total if total > 0 else 0.0
        fractions.append(frac)

        if n_conf >= 1:
            n_with_1 += 1
        if n_conf >= 2:
            n_with_2 += 1

        if frac == 0:
            buckets["0%"] += 1
        elif frac <= 0.25:
            buckets["1-25%"] += 1
        elif frac <= 0.50:
            buckets["26-50%"] += 1
        elif frac <= 0.75:
            buckets["51-75%"] += 1
        else:
            buckets["76-100%"] += 1

    coverage_rate = n_with_1 / n_dark if n_dark > 0 else 0.0
    mean_frac = sum(fractions) / len(fractions) if fractions else 0.0
    gate_passed = coverage_rate >= 0.30

    coverage_audit = CoverageAudit(
        n_dark_tokens=n_dark,
        n_with_1plus_confirmed=n_with_1,
        n_with_2plus_confirmed=n_with_2,
        coverage_rate=round(coverage_rate, 4),
        mean_confirmed_fraction=round(mean_frac, 4),
        distribution=buckets,
        gate_passed=gate_passed,
    )

    print(f"       Dark tokens: {n_dark}")
    print(f"       ≥1 confirmed char: {n_with_1} ({coverage_rate:.1%})")
    print(f"       ≥2 confirmed chars: {n_with_2} ({n_with_2/max(n_dark,1):.1%})")
    print(f"       Mean confirmed fraction: {mean_frac:.3f}")
    print(f"       Distribution: {buckets}")
    print(f"       Gate: {'PASS' if gate_passed else 'FAIL'}")

    if not gate_passed:
        print("\n  *** Coverage too low — Track B infeasible ***")
        runtime = round(time.time() - t0, 2)
        result = ConcatenationBridgeResult(
            coverage=_convert(asdict(coverage_audit)),
            n_signal_anchors_used=0,
            n_dark_tokens_examined=0,
            n_with_partial_decode=0,
            n_bridge_matches=0,
            n_unique_words_found=0,
            top_bridge_matches=[],
            n_concat_pairs_tested=0,
            n_concat_matches=0,
            top_concat_matches=[],
            consensus_assignments=[],
            n_new_assignments=0,
            null_mean_matches=0.0,
            null_std_matches=0.0,
            bridge_selectivity=0.0,
            bridge_z_score=0.0,
            pharma_dict_size=0,
            confirmed_triple_count=len(confirmed_triples),
            verdict='BRIDGE_EMPTY',
            runtime_seconds=runtime,
        )
        out_path = _save_json(rd, 'concatenation_bridge.json', asdict(result))
        print(f"\n  Saved → {out_path}")
        return

    # ── 3. Build pharmaceutical dictionary ────────────────────────────
    print("\n  B.3  Building pharmaceutical dictionary...")

    pharma_dict = _build_pharma_dict()
    print(f"       Dictionary size: {len(pharma_dict)} words")

    # ── 4. Signal-adjacent dark token search ──────────────────────────
    print("\n  B.4  Signal-adjacent bridge search...")

    # Find all SIGNAL token positions
    signal_positions = set()
    for i in range(n_tokens):
        if token_decoded[i] in SIGNAL_WORDS_SET:
            signal_positions.add(i)

    bridge_matches: List[BridgeMatch] = []
    n_anchors_used = 0
    n_dark_examined = 0
    n_with_partial = 0
    seen_dark_tokens: Set[int] = set()

    for sig_idx in sorted(signal_positions):
        anchor_word = token_decoded[sig_idx]
        n_anchors_used += 1

        # Check neighbors at distance 1 and 2
        for dist in [1, 2]:
            for offset, position in [(-dist, 'before'), (dist, 'after')]:
                nbr_idx = sig_idx + offset
                if nbr_idx < 0 or nbr_idx >= n_tokens:
                    continue
                if nbr_idx in signal_positions:
                    continue  # Neighbor is also a signal word
                if nbr_idx in seen_dark_tokens:
                    continue

                seen_dark_tokens.add(nbr_idx)
                n_dark_examined += 1
                dark_eva = token_evas[nbr_idx]

                # Build partial decode
                pattern, details = _build_partial_decode(
                    dark_eva, assignment, eva_to_triple,
                    modifier_chars, confirmed_triples,
                )

                n_conf = sum(1 for _, _, _, c in details if c)
                n_free = sum(1 for _, _, _, c in details if not c)

                if n_conf < 1 or n_free < 1 or n_free > 3:
                    continue
                n_with_partial += 1

                # Search dictionary
                matches = _search_dict(pattern, pharma_dict)
                if not matches:
                    continue

                for mword in matches:
                    implied = _extract_implied_assignments(
                        pattern, mword, details, eva_to_triple,
                    )
                    bridge_matches.append(BridgeMatch(
                        token_idx=nbr_idx,
                        token_eva=dark_eva,
                        pattern=pattern,
                        matched_word=mword,
                        n_confirmed_chars=n_conf,
                        n_free_chars=n_free,
                        implied_assignments=implied,
                        anchor_word=anchor_word,
                        anchor_position=position,
                        distance=dist,
                        folio=token_folios[nbr_idx],
                        n_total_matches=len(matches),
                    ))

    n_bridge_matches = len(bridge_matches)
    unique_words = set(bm.matched_word for bm in bridge_matches)

    print(f"       Signal anchors: {n_anchors_used}")
    print(f"       Dark tokens examined: {n_dark_examined}")
    print(f"       With usable partial decode: {n_with_partial}")
    print(f"       Bridge matches: {n_bridge_matches}")
    print(f"       Unique words found: {len(unique_words)}")

    # Show top matches (by fewest total alternatives = highest confidence)
    sorted_matches = sorted(bridge_matches, key=lambda m: m.n_total_matches)
    for bm in sorted_matches[:15]:
        print(f"       {bm.token_eva:12s} pattern={bm.pattern:10s} "
              f"→ {bm.matched_word:12s} "
              f"anchor={bm.anchor_word} ({bm.anchor_position}) "
              f"n_alt={bm.n_total_matches} folio={bm.folio}")

    # ── 5. Concatenation-enhanced search ──────────────────────────────
    print("\n  B.5  Concatenation bridge search...")

    concat_matches: List[BridgeMatch] = []
    n_concat_tested = 0

    for sig_idx in sorted(signal_positions):
        anchor_word = token_decoded[sig_idx]

        for offset, position in [(-1, 'before'), (1, 'after')]:
            nbr_idx = sig_idx + offset
            if nbr_idx < 0 or nbr_idx >= n_tokens:
                continue
            if nbr_idx in signal_positions:
                continue

            dark_eva = token_evas[nbr_idx]
            pattern, details = _build_partial_decode(
                dark_eva, assignment, eva_to_triple,
                modifier_chars, confirmed_triples,
            )

            if not any(not c for _, _, _, c in details):
                continue  # No free chars to search

            n_concat_tested += 1

            # Concatenate: signal + dark or dark + signal
            if position == 'after':
                concat_pattern = anchor_word + pattern
            else:
                concat_pattern = pattern + anchor_word

            concat_hits = _search_dict(concat_pattern, pharma_dict)
            for mword in concat_hits:
                concat_matches.append(BridgeMatch(
                    token_idx=nbr_idx,
                    token_eva=dark_eva,
                    pattern=concat_pattern,
                    matched_word=mword,
                    n_confirmed_chars=sum(1 for _, _, _, c in details if c),
                    n_free_chars=sum(1 for _, _, _, c in details if not c),
                    implied_assignments={},
                    anchor_word=anchor_word,
                    anchor_position=position,
                    distance=1,
                    folio=token_folios[nbr_idx],
                    n_total_matches=len(concat_hits),
                ))

    print(f"       Concat pairs tested: {n_concat_tested}")
    print(f"       Concat matches: {len(concat_matches)}")
    for cm in sorted(concat_matches, key=lambda m: m.n_total_matches)[:10]:
        print(f"       {cm.pattern:20s} → {cm.matched_word:15s} "
              f"anchor={cm.anchor_word} folio={cm.folio}")

    # ── 6. Cross-position consensus ───────────────────────────────────
    print("\n  B.6  Cross-position consensus for free triples...")

    triple_implications: Dict[str, Counter] = defaultdict(Counter)
    for bm in bridge_matches:
        for triple_key, syllable in bm.implied_assignments.items():
            weight = 1.0 / max(bm.n_total_matches, 1)
            triple_implications[triple_key][syllable] += weight

    consensus_list: List[ConsensusAssignment] = []
    for triple_key, syllable_counts in triple_implications.items():
        total = sum(syllable_counts.values())
        if total < 3:
            continue  # Not enough evidence
        top_syl, top_count = syllable_counts.most_common(1)[0]
        consensus = top_count / total if total > 0 else 0.0

        agrees = (assignment.get(triple_key) == top_syl)
        source_words = list(set(
            bm.matched_word for bm in bridge_matches
            if triple_key in bm.implied_assignments
            and bm.implied_assignments[triple_key] == top_syl
        ))

        consensus_list.append(ConsensusAssignment(
            triple_key=triple_key,
            implied_syllable=top_syl,
            consensus=round(consensus, 4),
            n_observations=int(round(total)),
            agrees_with_phase15=agrees,
            source_words=source_words[:10],
        ))

        status = "AGREES" if agrees else "NEW"
        print(f"       {triple_key:35s} → {top_syl:4s}  "
              f"consensus={consensus:.1%}  n={int(round(total))}  "
              f"P15={'=' if agrees else '≠'}{assignment.get(triple_key, '?')}  "
              f"[{status}]")

    # Filter to high-confidence consensus assignments
    strong_consensus = [c for c in consensus_list
                        if c.consensus > 0.5 and c.n_observations >= 5]
    n_new = sum(1 for c in strong_consensus if not c.agrees_with_phase15)

    print(f"\n       Strong consensus: {len(strong_consensus)} "
          f"({n_new} new, {len(strong_consensus) - n_new} agree with P15)")

    # ── 7. Null test ──────────────────────────────────────────────────
    print("\n  B.7  Null test (shuffled syllables)...")

    rng = random.Random(42)
    n_null = 20  # Fewer iterations since bridge search is slower
    all_syllables = list(set(assignment.values()))
    null_match_counts = []

    for null_i in range(n_null):
        # Create shuffled assignment for confirmed triples
        shuffled_assignment = dict(assignment)
        confirmed_list = list(confirmed_triples)
        shuffled_syls = [shuffled_assignment[t] for t in confirmed_list]
        rng.shuffle(shuffled_syls)
        for t, s in zip(confirmed_list, shuffled_syls):
            shuffled_assignment[t] = s

        # Count bridge matches with shuffled assignment
        null_count = 0
        for sig_idx in sorted(signal_positions):
            for offset in [-1, 1]:
                nbr_idx = sig_idx + offset
                if nbr_idx < 0 or nbr_idx >= n_tokens:
                    continue
                if nbr_idx in signal_positions:
                    continue

                pattern, _ = _build_partial_decode(
                    token_evas[nbr_idx], shuffled_assignment, eva_to_triple,
                    modifier_chars, confirmed_triples,
                )
                n_conf = pattern.count('?') < len(
                    [c for c in tokenize_eva_chars(token_evas[nbr_idx])
                     if c not in modifier_chars]
                )
                if '?' in pattern and pattern != '?' * len(pattern):
                    matches = _search_dict(pattern, pharma_dict)
                    null_count += len(matches)

        null_match_counts.append(null_count)

    null_mean = (sum(null_match_counts) / len(null_match_counts)
                 if null_match_counts else 0.0)
    null_std = (sum((x - null_mean) ** 2 for x in null_match_counts)
                / max(len(null_match_counts), 1)) ** 0.5
    bridge_z = ((n_bridge_matches - null_mean) / null_std
                if null_std > 0 else 0.0)
    bridge_sel = (n_bridge_matches / null_mean
                  if null_mean > 0 else float('inf'))

    print(f"       Real matches:  {n_bridge_matches}")
    print(f"       Null mean:     {null_mean:.1f} ± {null_std:.1f}")
    print(f"       Z-score:       {bridge_z:.2f}")
    print(f"       Selectivity:   {bridge_sel:.2f}×")

    # ── 8. Verdict ────────────────────────────────────────────────────

    if len(strong_consensus) >= 3 and bridge_sel > 1.5:
        verdict = 'BRIDGE_PRODUCTIVE'
    elif len(strong_consensus) >= 1 or bridge_sel > 1.2:
        verdict = 'BRIDGE_MARGINAL'
    else:
        verdict = 'BRIDGE_EMPTY'

    print(f"\n  Verdict: {verdict}")

    # ── 9. Save ───────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = ConcatenationBridgeResult(
        coverage=_convert(asdict(coverage_audit)),
        n_signal_anchors_used=n_anchors_used,
        n_dark_tokens_examined=n_dark_examined,
        n_with_partial_decode=n_with_partial,
        n_bridge_matches=n_bridge_matches,
        n_unique_words_found=len(unique_words),
        top_bridge_matches=[
            _convert(asdict(bm))
            for bm in sorted_matches[:50]
        ],
        n_concat_pairs_tested=n_concat_tested,
        n_concat_matches=len(concat_matches),
        top_concat_matches=[
            _convert(asdict(cm))
            for cm in sorted(concat_matches, key=lambda m: m.n_total_matches)[:30]
        ],
        consensus_assignments=[
            _convert(asdict(c)) for c in consensus_list
        ],
        n_new_assignments=n_new,
        null_mean_matches=round(null_mean, 2),
        null_std_matches=round(null_std, 2),
        bridge_selectivity=round(bridge_sel, 4),
        bridge_z_score=round(bridge_z, 2),
        pharma_dict_size=len(pharma_dict),
        confirmed_triple_count=len(confirmed_triples),
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'concatenation_bridge.json', asdict(result))
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
