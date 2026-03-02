"""
Phase 11 – Constraint layers for CSP phonetic decoding
========================================================
Six constraint layers that progressively prune the search space for mapping
14 Voynich grid cells to CV syllables in a target Romance language.

Layer 1: Phoneme inventory         (domain initialisation)
Layer 2: Frequency matching        (rank-based pruning)
Layer 3: Phonotactic legality      (CV pair legality)
Layer 4: Word-structure validity   (decoded-token scoring)
Layer 5: Illustration anchors      (Rosetta folio matching)
Layer 6: Cross-entropy scoring     (LM scoring)
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import tokenize_eva_chars, token_to_grid_cells
from voynich.core.reference import (
    PHONEME_INVENTORIES,
    ROMANCE_PHONOTACTICS,
    build_cv_syllable_table,
    build_cvc_syllable_table,
    build_syllable_frequency_table,
    get_phoneme_inventory,
    LATIN_IMPERATIVE_SYLLABIFICATIONS,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PhonemeInventory:
    """Phoneme inventory for a target language with ranked CV syllables."""
    language: str
    consonants: List[str]
    vowels: List[str]
    cv_syllables: List[str]
    frequency_ranked: List[str]
    phonotactic_onsets: List[str]
    phonotactic_rimes: List[str]
    forbidden_onsets: set = field(default_factory=set)


@dataclass
class VerbConstraint:
    """Hard/soft constraint derived from Phase 9 verb-to-imperative assignments."""
    voynich_stem: str
    latin_verb: str
    latin_syllables: List[str]   # from LATIN_IMPERATIVE_SYLLABIFICATIONS
    voynich_cells: List[str]     # grid cell keys for each EVA char in stem
    confidence: float
    is_hard: bool                # True when confidence >= 0.65 AND length-matched


@dataclass
class AnchorConstraint:
    """One illustration-anchor mapping between a Voynich stem and a plant."""
    folio: str
    voynich_stem: str
    voynich_cells: List[str]
    target_word: str
    target_syllables: List[str]
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Inventory builder
# ---------------------------------------------------------------------------

def build_phoneme_inventory(
    language: str,
    ref_corpus: Optional[Any] = None,
    relaxation_level: int = 0,
    inherent_vowel: Optional[str] = None,
) -> PhonemeInventory:
    """Build a :class:`PhonemeInventory` for *language*.

    Generates all legal CV (and optionally CVC/CCV) syllables and ranks them
    by frequency in the reference corpus (falls back to uniform if none).

    Parameters
    ----------
    relaxation_level:
        0 = strict CV only; 1–5 = progressively expanded inventory
        (see :func:`~voynich.core.reference.build_cvc_syllable_table`).
    inherent_vowel:
        Vowel appended to consonant singletons at relaxation level 1+.
    """
    inv = get_phoneme_inventory(language)
    if relaxation_level > 0:
        cv_syllables = build_cvc_syllable_table(
            language, relaxation_level=relaxation_level,
            inherent_vowel=inherent_vowel,
        )
    else:
        cv_syllables = build_cv_syllable_table(language)
    freq_table = build_syllable_frequency_table(language, ref_corpus)

    # Rank syllables by frequency descending
    frequency_ranked = sorted(
        cv_syllables,
        key=lambda s: freq_table.get(s, 0.0),
        reverse=True,
    )

    # Phonotactic info
    phono = ROMANCE_PHONOTACTICS.get(language, ROMANCE_PHONOTACTICS.get('latin', {}))
    onsets = phono.get('onsets', [''])
    rimes = phono.get('rimes', inv['vowels'])
    forbidden = phono.get('forbidden_onsets', set())

    return PhonemeInventory(
        language=language,
        consonants=inv['consonants'],
        vowels=inv['vowels'],
        cv_syllables=cv_syllables,
        frequency_ranked=frequency_ranked,
        phonotactic_onsets=onsets,
        phonotactic_rimes=rimes,
        forbidden_onsets=forbidden,
    )


# ---------------------------------------------------------------------------
# Anchor builder
# ---------------------------------------------------------------------------

def build_anchor_constraints(
    rosetta_data: Dict,
    cv_labels: Dict,
) -> List[AnchorConstraint]:
    """Build :class:`AnchorConstraint` objects from Rosetta selection data.

    For each selected Rosetta folio, decompose the dominant Voynich stem
    into grid cells and syllabify the medieval plant name.
    """
    from voynich.core.corpus import build_eva_to_cell_lookup
    from voynich.core.stats import syllabify_latin

    eva_to_cell = build_eva_to_cell_lookup(cv_labels)
    anchors: List[AnchorConstraint] = []

    for folio_info in rosetta_data.get('folio_scores', []):
        folio = folio_info.get('folio', '')
        selected = rosetta_data.get('selected_rosetta_folios', [])
        if folio not in selected:
            continue

        stem = folio_info.get('dominant_stem', '')
        target = folio_info.get('medieval_name', '')
        weight = folio_info.get('combined_score', 0.5)

        if not stem or not target:
            continue

        # Decompose EVA stem into grid cells
        cells = token_to_grid_cells(stem, eva_to_cell)

        # Syllabify the target plant name
        target_syls = syllabify_latin(target.split()[0])  # first word only
        if not target_syls:
            target_syls = [target]

        anchors.append(AnchorConstraint(
            folio=folio,
            voynich_stem=stem,
            voynich_cells=cells,
            target_word=target,
            target_syllables=target_syls,
            weight=weight,
        ))

    return anchors


# ---------------------------------------------------------------------------
# Layer 1: Phoneme inventory constraint (domain initialisation)
# ---------------------------------------------------------------------------

def prune_by_inventory(
    cell_domains: Dict[str, List[str]],
    inventory: PhonemeInventory,
) -> Dict[str, List[str]]:
    """Restrict each cell's domain to legal CV syllables.

    Each cell can only map to a syllable that exists in the target
    language's phoneme inventory.
    """
    legal = set(inventory.cv_syllables)
    pruned: Dict[str, List[str]] = {}
    for cell_key, domain in cell_domains.items():
        pruned[cell_key] = [s for s in domain if s in legal]
        # Keep at least the full CV table if nothing matched
        if not pruned[cell_key]:
            pruned[cell_key] = list(inventory.cv_syllables)
    return pruned


# ---------------------------------------------------------------------------
# Layer 2: Frequency matching constraint (rank-based pruning)
# ---------------------------------------------------------------------------

def prune_by_frequency(
    cell_domains: Dict[str, List[str]],
    cell_frequencies: Dict[str, int],
    inventory: PhonemeInventory,
    slack: int = 3,
) -> Dict[str, List[str]]:
    """Prune domains by frequency rank matching.

    Cells are ranked by corpus frequency; syllables are ranked by
    reference corpus frequency.  Cell at rank *k* can only map to
    syllables at ranks [k - slack, k + slack].
    """
    # Rank cells by frequency (descending)
    ranked_cells = sorted(
        cell_frequencies.keys(),
        key=lambda c: cell_frequencies.get(c, 0),
        reverse=True,
    )
    cell_rank = {c: i for i, c in enumerate(ranked_cells)}

    # Syllable rank from inventory
    syl_rank = {s: i for i, s in enumerate(inventory.frequency_ranked)}

    n_cells = len(ranked_cells)
    n_syls = len(inventory.frequency_ranked)

    pruned: Dict[str, List[str]] = {}
    for cell_key, domain in cell_domains.items():
        rank_c = cell_rank.get(cell_key, n_cells - 1)
        # Map cell rank to proportional syllable rank range
        # Cell rank / n_cells ≈ syllable rank / n_syls
        centre = int(rank_c * n_syls / max(n_cells, 1))
        lo = max(0, centre - slack * (n_syls // n_cells + 1))
        hi = min(n_syls, centre + slack * (n_syls // n_cells + 1) + 1)

        allowed_syls = set(inventory.frequency_ranked[lo:hi])
        filtered = [s for s in domain if s in allowed_syls]
        # Fall back to original domain if over-pruned
        pruned[cell_key] = filtered if filtered else domain

    return pruned


# ---------------------------------------------------------------------------
# Layer 3: Phonotactic legality
# ---------------------------------------------------------------------------

def check_phonotactic_legality(
    syllable: str,
    inventory: PhonemeInventory,
) -> bool:
    """Return True if *syllable* is a phonotactically legal CV combo."""
    vowels = set(inventory.vowels)

    # Split syllable into onset + nucleus
    onset = ''
    rest = syllable
    while rest and rest[0] not in vowels:
        onset += rest[0]
        rest = rest[1:]

    if onset in inventory.forbidden_onsets:
        return False

    return True


def prune_by_phonotactics(
    cell_domains: Dict[str, List[str]],
    inventory: PhonemeInventory,
) -> Dict[str, List[str]]:
    """Remove phonotactically illegal syllables from each cell's domain."""
    pruned: Dict[str, List[str]] = {}
    for cell_key, domain in cell_domains.items():
        filtered = [
            s for s in domain
            if check_phonotactic_legality(s, inventory)
        ]
        pruned[cell_key] = filtered if filtered else domain
    return pruned


# ---------------------------------------------------------------------------
# Layer 4: Word-structure validity
# ---------------------------------------------------------------------------

def score_word_validity(
    decoded_tokens: List[str],
    inventory: PhonemeInventory,
) -> float:
    """Return the fraction of decoded tokens that are phonotactically legal.

    A decoded word is "legal" if:
    - Its final character is in the word-final-legal set
    - It contains at least one vowel
    - Its length is reasonable (2–20 chars)
    """
    if not decoded_tokens:
        return 0.0

    inv = get_phoneme_inventory(inventory.language)
    final_legal = inv.get('word_final_legal', set(inventory.vowels))
    vowel_set = set(inventory.vowels)

    n_legal = 0
    for word in decoded_tokens:
        if not word or len(word) < 2 or len(word) > 20:
            continue
        has_vowel = any(ch in vowel_set for ch in word)
        final_ok = word[-1] in final_legal
        if has_vowel and final_ok:
            n_legal += 1

    return n_legal / len(decoded_tokens)


# ---------------------------------------------------------------------------
# Layer 5: Illustration anchor matching
# ---------------------------------------------------------------------------

def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def score_anchor_match(
    assignment: Dict[str, str],
    anchors: List[AnchorConstraint],
    eva_to_cell: Dict[str, str],
) -> Tuple[float, int]:
    """Score how well an assignment matches illustration anchors.

    For each anchor, decode the Voynich stem using the assignment and
    compare to the target plant name syllables via edit distance.

    Returns (total_penalty, n_matched) where n_matched counts anchors
    with edit distance ≤ 3.
    """
    total_penalty = 0.0
    n_matched = 0

    for anchor in anchors:
        # Decode the stem
        decoded_parts: List[str] = []
        for cell_key in anchor.voynich_cells:
            syl = assignment.get(cell_key, '?')
            decoded_parts.append(syl)
        decoded = ''.join(decoded_parts)

        target = ''.join(anchor.target_syllables)

        dist = _edit_distance(decoded.lower(), target.lower())

        # Normalise by target length
        norm_dist = dist / max(len(target), 1)
        total_penalty += norm_dist * anchor.weight

        if dist <= 3:
            n_matched += 1

    return total_penalty, n_matched


# ---------------------------------------------------------------------------
# Layer 6: Cross-entropy scoring
# ---------------------------------------------------------------------------

def _decode_voynich_tokens(
    assignment: Dict[str, str],
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    max_tokens: int = 2000,
) -> List[str]:
    """Decode Voynich tokens to a list of syllable strings."""
    decoded_words: List[str] = []
    for token in voynich_tokens[:max_tokens]:
        chars = tokenize_eva_chars(token)
        parts: List[str] = []
        for ch in chars:
            cell = eva_to_cell.get(ch)
            if cell and cell in assignment:
                parts.append(assignment[cell])
            # unknown chars are dropped (not appended as '?')
        if parts:
            decoded_words.append(''.join(parts))
    return decoded_words


def score_cross_entropy(
    assignment: Dict[str, str],
    lm: Dict,
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    max_tokens: int = 2000,
) -> float:
    """Score an assignment by cross-entropy of decoded text.

    Decodes *max_tokens* Voynich tokens and scores the resulting
    character sequence against the language model.

    The LM must have been built from a **word token list** via
    ``build_ngram_lm(tokens, ...)``.  Decoded words are joined with
    '_' (the same boundary marker the LM uses) before scoring.
    """
    from voynich.core.stats import cross_entropy_lm

    decoded_words = _decode_voynich_tokens(
        assignment, voynich_tokens, eva_to_cell, max_tokens,
    )
    if not decoded_words:
        return 99.0

    # Format to match build_ngram_lm encoding: '_' between words,
    # plus leading and trailing '_' word-boundary markers.
    decoded_text = '_' + '_'.join(decoded_words) + '_'
    return cross_entropy_lm(decoded_text, lm, per_char=True)


def score_dict_hit_rate(
    assignment: Dict[str, str],
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    max_tokens: int = 2000,
) -> float:
    """Fraction of decoded tokens that appear in the reference word set.

    A high dictionary hit rate means the decoded text contains many
    recognised Romance words — the most direct evidence of correct decoding.
    """
    decoded = _decode_voynich_tokens(
        assignment, voynich_tokens, eva_to_cell, max_tokens,
    )
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w in ref_word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def composite_score(
    cross_entropy: float,
    word_validity: float,
    anchor_penalty: float,
    anchor_match_count: int,
    n_anchors: int,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 1.5,
    verb_penalty: float = 0.0,
    gamma_verb: float = 1.0,
    n_distinct_syllables: int = 0,
    n_cells: int = 14,
) -> float:
    """Compute a combined score (lower is better).

    ``score = α·CE + β·(1-validity) + γ·anchor_penalty + γv·verb_penalty``

    A soft diversity reward discourages degenerate solutions where many
    cells collapse to the same syllable:
    ``score -= 0.2 * (n_distinct_syllables / n_cells)``
    """
    base = (
        alpha * cross_entropy
        + beta * (1.0 - word_validity)
        + gamma * anchor_penalty
        + gamma_verb * verb_penalty
    )
    if n_cells > 0 and n_distinct_syllables > 0:
        base -= 0.2 * (n_distinct_syllables / n_cells)
    return base


# ---------------------------------------------------------------------------
# Phase 11.5 – Verb constraint layer (Layer 7)
# ---------------------------------------------------------------------------

def build_verb_constraints(
    verb_data: Dict,
    cv_labels: Dict,
    eva_to_cell: Dict[str, str],
) -> List[VerbConstraint]:
    """Build verb constraints from Phase 9 results.

    For each confident assignment (score >= 0.5), decompose the Voynich
    stem into grid cells.  Only create a constraint when the cell count
    exactly matches the syllable count from
    :data:`~voynich.core.reference.LATIN_IMPERATIVE_SYLLABIFICATIONS`.

    In practice only 1–3 stems will be length-matched.
    """
    constraints: List[VerbConstraint] = []
    assignments = verb_data.get('assignments', [])

    for asgn in assignments:
        voynich_stem = asgn.get('voynich_stem', '')
        latin_verb = asgn.get('latin_verb', '')
        confidence = float(asgn.get('total_score', 0.0))

        if confidence < 0.5 or not voynich_stem or not latin_verb:
            continue

        latin_syls = LATIN_IMPERATIVE_SYLLABIFICATIONS.get(latin_verb)
        if not latin_syls:
            continue

        # Decompose stem into grid cells
        cells = token_to_grid_cells(voynich_stem, eva_to_cell)

        # Only use when cell count matches syllable count (direct alignment)
        if len(cells) != len(latin_syls):
            continue

        constraints.append(VerbConstraint(
            voynich_stem=voynich_stem,
            latin_verb=latin_verb,
            latin_syllables=list(latin_syls),
            voynich_cells=cells,
            confidence=confidence,
            is_hard=(confidence >= 0.65),
        ))

    return constraints


def apply_verb_constraints(
    cell_domains: Dict[str, List[str]],
    verb_constraints: List[VerbConstraint],
    inventory: PhonemeInventory,
) -> Dict[str, List[str]]:
    """Narrow domains for cells with hard verb constraints.

    For each hard constraint (confidence >= 0.65), if the target syllable
    is in the inventory, the cell's domain is narrowed to just that value.
    If the target syllable is NOT in the inventory (e.g. a CVC form not
    yet included), it is added to the domain without narrowing (soft mode).
    """
    legal_set = set(inventory.cv_syllables)
    result = {k: list(v) for k, v in cell_domains.items()}

    for vc in verb_constraints:
        if not vc.is_hard:
            continue
        for cell_key, target_syl in zip(vc.voynich_cells, vc.latin_syllables):
            if cell_key not in result:
                continue
            if target_syl in legal_set:
                result[cell_key] = [target_syl]
            else:
                # Add as soft option — don't narrow
                existing = set(result[cell_key])
                existing.add(target_syl)
                result[cell_key] = list(existing)

    return result


def score_verb_consistency(
    assignment: Dict[str, str],
    verb_constraints: List[VerbConstraint],
    eva_to_cell: Dict[str, str],
) -> Tuple[float, int]:
    """Score how well an assignment matches verb constraints.

    Mirrors :func:`score_anchor_match`:  for each verb constraint, decode
    the Voynich stem and compare edit distance to the full Latin verb string.

    Returns (total_penalty, n_matched) where n_matched counts constraints
    with edit distance <= 2.
    """
    total_penalty = 0.0
    n_matched = 0

    for vc in verb_constraints:
        decoded_parts: List[str] = []
        for cell_key in vc.voynich_cells:
            syl = assignment.get(cell_key, '?')
            decoded_parts.append(syl)
        decoded = ''.join(decoded_parts)
        target = ''.join(vc.latin_syllables)
        dist = _edit_distance(decoded.lower(), target.lower())
        norm_dist = dist / max(len(target), 1)
        total_penalty += norm_dist * vc.confidence
        if dist <= 2:
            n_matched += 1

    return total_penalty, n_matched


def extract_confirmed_hits(
    assignment: Dict[str, str],
    voynich_tokens: List[str],
    eva_to_cell: Dict[str, str],
    ref_word_set: set,
    min_frequency: int = 3,
) -> List[AnchorConstraint]:
    """Extract high-confidence decoded tokens as new anchor constraints.

    A token qualifies when:
    1. It always decodes to the same string (consistency).
    2. That string is an exact hit in *ref_word_set*.
    3. The token appears >= *min_frequency* times in *voynich_tokens*.

    Returns a list of :class:`AnchorConstraint` objects with
    ``folio='confirmed_hit'`` so they are distinguishable from
    illustration anchors.
    """
    # Count token frequencies and collect unique decodings per token
    freq: Counter = Counter(voynich_tokens)
    token_decoded: Dict[str, set] = {}

    for token in set(voynich_tokens):
        chars = tokenize_eva_chars(token)
        parts: List[str] = []
        for ch in chars:
            cell = eva_to_cell.get(ch)
            if cell and cell in assignment:
                parts.append(assignment[cell])
        if parts:
            decoded = ''.join(parts)
            token_decoded[token] = {decoded}

    confirmed: List[AnchorConstraint] = []
    for token, decoded_set in token_decoded.items():
        if len(decoded_set) != 1:
            continue
        decoded_str = next(iter(decoded_set))
        if decoded_str not in ref_word_set:
            continue
        if freq[token] < min_frequency:
            continue

        # Build cells for this token
        cells = token_to_grid_cells(token, eva_to_cell)
        # Syllabify decoded_str (treat as one unit per cell for simplicity)
        n_cells = len(cells)
        if n_cells == 0:
            continue
        chunk = max(1, len(decoded_str) // n_cells)
        target_syls = [
            decoded_str[i * chunk:(i + 1) * chunk]
            for i in range(n_cells)
        ]
        # Pad or truncate
        while len(target_syls) < n_cells:
            target_syls.append(target_syls[-1])
        target_syls = target_syls[:n_cells]

        confirmed.append(AnchorConstraint(
            folio='confirmed_hit',
            voynich_stem=token,
            voynich_cells=cells,
            target_word=decoded_str,
            target_syllables=target_syls,
            weight=0.8,  # slightly lower weight than illustration anchors
        ))

    return confirmed
