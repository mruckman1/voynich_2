"""
Step 24.11 – Known-Plaintext Crib Search
========================================
Search for known Latin medical formulae encoded in the Voynich text
using the Phase 16 table as a partial key.

Dependency chain:
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → known_text_search.json (this step)
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
    EVA_VISUAL_COMPONENTS,
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


# ---------------------------------------------------------------------------
# Medical formulae and null controls
# ---------------------------------------------------------------------------

MEDICAL_FORMULAE = [
    ('recipe', 'recipe'),
    ('accipe', 'accipe'),
    ('est calida', 'est calida'),
    ('est frigida', 'est frigida'),
    ('calida et sicca', 'calida et sicca'),
    ('frigida et humida', 'frigida et humida'),
    ('valet contra', 'valet contra'),
    ('coque in aqua', 'coque in aqua'),
    ('misce cum melle', 'misce cum melle'),
    ('fac emplastrum', 'fac emplastrum'),
    ('tere et cola', 'tere et cola'),
    ('bibe cum vino', 'bibe cum vino'),
    ('in primo gradu', 'in primo gradu'),
    ('in secundo gradu', 'in secundo gradu'),
    ('herba calida', 'herba calida'),
    ('herba frigida', 'herba frigida'),
    ('contra dolorem', 'contra dolorem'),
    ('destilla per', 'destilla per'),
    ('pone in vase', 'pone in vase'),
    ('cum aqua rosa', 'cum aqua rosa'),
]

# Null control: non-medical Latin phrases
NULL_PHRASES = [
    ('senatus populusque', 'senatus populusque'),
    ('anno domini', 'anno domini'),
    ('post meridiem', 'post meridiem'),
    ('de rerum natura', 'de rerum natura'),
    ('veni vidi vici', 'veni vidi vici'),
    ('carpe diem', 'carpe diem'),
    ('in vino veritas', 'in vino veritas'),
    ('ars longa vita', 'ars longa vita'),
    ('cogito ergo sum', 'cogito ergo sum'),
    ('tempus fugit', 'tempus fugit'),
    ('amor vincit', 'amor vincit'),
    ('caveat emptor', 'caveat emptor'),
    ('deus ex machina', 'deus ex machina'),
    ('et tu brute', 'et tu brute'),
    ('habeas corpus', 'habeas corpus'),
    ('magna carta', 'magna carta'),
    ('modus operandi', 'modus operandi'),
    ('persona non grata', 'persona non grata'),
    ('status quo ante', 'status quo ante'),
    ('terra incognita', 'terra incognita'),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FormulaMatch:
    formula_name: str
    formula_text: str
    position: int  # word index in decoded sequence
    agreement_rate: float
    matched_words: List[str]
    mismatched_words: List[Dict]  # {position, expected, actual}
    folio: str  # which folio this appears on (if traceable)


@dataclass
class ImpliedCorrection:
    triple_key: str
    current_syllable: str
    implied_syllable: str
    source_formula: str
    n_supporting_matches: int


@dataclass
class KnownTextSearchResult:
    timestamp: str
    n_medical_formulae: int
    n_null_phrases: int
    # Medical matches
    medical_matches: List[Dict]
    n_medical_matches: int
    medical_match_rate: float  # avg agreement across best matches
    # Null matches
    null_matches: List[Dict]
    n_null_matches: int
    null_match_rate: float
    # Selectivity
    medical_vs_null_ratio: float
    # Implied corrections
    implied_corrections: List[Dict]
    n_corrections: int
    # Verdict
    is_medical: bool  # medical >> null matches
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict]:
    """Load a JSON file if it exists, else return None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _syllabify_latin(word: str) -> List[str]:
    """Simple Latin syllabification by greedy CV splitting.

    Splits a Latin word into approximate syllables using basic rules:
    - Vowels are: a, e, i, o, u
    - A syllable has at most one vowel nucleus
    - Consonant clusters between vowels are split: last consonant goes with
      the following vowel, remaining consonants stay with the preceding vowel
    """
    vowels = set('aeiou')
    word = word.lower().strip()
    if not word:
        return []

    # Identify vowel positions
    vowel_positions = [i for i, ch in enumerate(word) if ch in vowels]
    if not vowel_positions:
        return [word] if word else []

    syllables = []
    start = 0

    for vi in range(len(vowel_positions)):
        vpos = vowel_positions[vi]

        if vi < len(vowel_positions) - 1:
            next_vpos = vowel_positions[vi + 1]
            # Consonants between this vowel and the next
            consonant_span = next_vpos - vpos - 1

            if consonant_span <= 0:
                # Adjacent vowels: split right after this vowel
                end = vpos + 1
            elif consonant_span == 1:
                # Single consonant goes with next syllable
                end = vpos + 1
            else:
                # Multiple consonants: split before the last one
                end = next_vpos - 1

            syllables.append(word[start:end])
            start = end
        else:
            # Last vowel: take everything remaining
            syllables.append(word[start:])

    return syllables


def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode all tokens using the R3 combined strategy from Phase 16.

    R3: tries alteration first, then stripping, then original.
    Picks whichever gets a dict hit.
    """
    decoded = []
    for token in tokens:
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


def _build_folio_token_index(corpus) -> List[Tuple[str, int]]:
    """Build a list of (folio, token_index_in_folio) for each token in order.

    Returns a list parallel to corpus.get_tokens() where each entry is
    (folio_id, local_token_index).
    """
    folio_map: List[Tuple[str, int]] = []
    for page in corpus.pages.values():
        page_tokens = page.paragraph_text.split() if page.paragraph_text else []
        for local_idx in range(len(page_tokens)):
            folio_map.append((page.folio, local_idx))
    return folio_map


def _search_formula_in_decoded(
    formula_words: List[str],
    decoded_words: List[str],
    folio_index: List[Tuple[str, int]],
    agreement_threshold: float = 0.6,
) -> List[FormulaMatch]:
    """Slide a window over decoded words, looking for the formula.

    Parameters
    ----------
    formula_words : list of str
        The expected Latin words in the formula.
    decoded_words : list of str
        The decoded Voynich words.
    folio_index : list of (folio, local_index)
        Folio mapping parallel to decoded_words.
    agreement_threshold : float
        Minimum fraction of formula words that must match.

    Returns
    -------
    List of FormulaMatch instances where agreement >= threshold.
    """
    n_formula = len(formula_words)
    n_decoded = len(decoded_words)
    matches: List[FormulaMatch] = []

    if n_formula == 0 or n_decoded == 0:
        return matches

    for pos in range(n_decoded - n_formula + 1):
        window = decoded_words[pos:pos + n_formula]
        matched = []
        mismatched = []

        for wi, (expected, actual) in enumerate(zip(formula_words, window)):
            if actual.lower() == expected.lower():
                matched.append(expected)
            else:
                mismatched.append({
                    'position': wi,
                    'expected': expected,
                    'actual': actual,
                })

        agreement = len(matched) / n_formula
        if agreement >= agreement_threshold:
            # Determine folio
            folio = 'unknown'
            if pos < len(folio_index):
                folio = folio_index[pos][0]

            matches.append(FormulaMatch(
                formula_name='',  # filled in by caller
                formula_text=' '.join(formula_words),
                position=pos,
                agreement_rate=round(agreement, 4),
                matched_words=matched,
                mismatched_words=mismatched,
                folio=folio,
            ))

    return matches


def _extract_corrections(
    matches: List[FormulaMatch],
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> List[ImpliedCorrection]:
    """For each mismatch in a formula match, infer what the correct mapping
    should be by comparing expected vs actual decoded output.

    Where the formula says the word should be X but we decoded Y, the
    difference implies a correction to the triple -> syllable table.
    """
    # Aggregate correction evidence: (triple_key, implied_syllable) -> count + sources
    correction_evidence: Dict[Tuple[str, str, str], Dict] = defaultdict(
        lambda: {'count': 0, 'sources': set()}
    )

    for match in matches:
        for mismatch in match.mismatched_words:
            word_pos = match.position + mismatch['position']
            expected_word = mismatch['expected']
            actual_word = mismatch['actual']

            if word_pos >= len(tokens):
                continue

            token = tokens[word_pos]
            chars = tokenize_eva_chars(token)

            # Syllabify the expected word
            expected_syls = _syllabify_latin(expected_word)
            # Syllabify the actual decoded word
            actual_syls = _syllabify_latin(actual_word)

            # Map token chars to triples (excluding modifiers)
            syllabic_chars = [
                ch for ch in chars
                if ch not in modifier_chars and ch in eva_to_triple
            ]
            triple_keys = [eva_to_triple[ch] for ch in syllabic_chars]

            # Compare expected vs actual syllables at each triple position
            for ti, triple_key in enumerate(triple_keys):
                current_syl = assignment.get(triple_key, '?')

                if ti < len(expected_syls):
                    implied_syl = expected_syls[ti]
                else:
                    continue

                # Only record if different from current assignment
                if implied_syl.lower() != current_syl.lower():
                    key = (triple_key, current_syl, implied_syl.lower())
                    correction_evidence[key]['count'] += 1
                    correction_evidence[key]['sources'].add(match.formula_name)

    # Convert to list, sorted by support count
    corrections: List[ImpliedCorrection] = []
    for (triple_key, current_syl, implied_syl), evidence in correction_evidence.items():
        corrections.append(ImpliedCorrection(
            triple_key=triple_key,
            current_syllable=current_syl,
            implied_syllable=implied_syl,
            source_formula=', '.join(sorted(evidence['sources'])),
            n_supporting_matches=evidence['count'],
        ))

    corrections.sort(key=lambda c: -c.n_supporting_matches)
    return corrections


def _search_all_formulae(
    formulae: List[Tuple[str, str]],
    decoded_words: List[str],
    folio_index: List[Tuple[str, int]],
    agreement_threshold: float = 0.6,
    label: str = 'medical',
) -> List[FormulaMatch]:
    """Search for all formulae in the decoded corpus.

    Returns all matches above the agreement threshold.
    """
    all_matches: List[FormulaMatch] = []

    for name, text in formulae:
        formula_words = text.lower().split()
        matches = _search_formula_in_decoded(
            formula_words, decoded_words, folio_index, agreement_threshold,
        )
        for m in matches:
            m.formula_name = name
        all_matches.extend(matches)

    return all_matches


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_known_text_search() -> None:
    """Step 24.11: Known-Plaintext Crib Search."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.11: Known-Plaintext Crib Search")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load Phase 15 best assignment ──
    print("\n  1. Loading Phase 15 best assignment ...")
    refine_path = os.path.join(rd, 'combined_refine.json')
    refine_data = _load_json(refine_path)
    if refine_data is None:
        print("     [SKIP] combined_refine.json not found -- run combined-refine first")
        return
    assignment = refine_data.get('best_assignment', {})
    print(f"     {len(assignment)} triple -> syllable mappings loaded")

    # ── 2. Load Phase 16 modifier info ──
    print("\n  2. Loading Phase 16 modifier classification ...")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    mod_data = _load_json(mod_path)
    if mod_data is None:
        print("     [SKIP] modifier_integrate.json not found -- run mod-integrate first")
        return

    modifier_chars = set(mod_data.get('modifier_chars', []))
    # Reconstruct modifier_rules from classifications
    modifier_rules: Dict[str, str] = {}
    for cl in mod_data.get('classifications', []):
        if cl.get('final_classification') == 'modifier':
            modifier_rules[cl['eva_char']] = cl.get('modifier_type', 'silent')
    print(f"     {len(modifier_chars)} modifier chars, {len(modifier_rules)} modifier rules")

    # ── 3. Load corpus ──
    print("\n  3. Loading corpus ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"     {len(tokens)} tokens, {len(eva_to_triple)} EVA->triple mappings")

    # ── 4. Build expanded reference word set ──
    print("\n  4. Building expanded reference word set ...")
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
    print(f"     {len(ref_word_set)} words in reference set")

    # ── 5. Decode corpus using R3 combined strategy ──
    print("\n  5. Decoding corpus with R3 combined strategy ...")
    decoded_words = _decode_corpus_r3(
        tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    decoded_lower = [w.lower() for w in decoded_words]
    n_hits = sum(1 for w in decoded_lower if w in ref_word_set)
    dict_hit = n_hits / len(decoded_lower) if decoded_lower else 0.0
    print(f"     {len(decoded_words)} decoded words, dict_hit={dict_hit:.4f}")

    # ── 6. Build folio index ──
    print("\n  6. Building folio-token index ...")
    folio_index = _build_folio_token_index(corpus)
    # Pad or truncate to match decoded_words length
    while len(folio_index) < len(decoded_words):
        folio_index.append(('unknown', 0))
    folio_index = folio_index[:len(decoded_words)]
    print(f"     {len(folio_index)} folio mappings")

    # ── 7. Search for medical formulae ──
    print("\n  7. Searching for medical formulae ...")
    medical_matches = _search_all_formulae(
        MEDICAL_FORMULAE, decoded_lower, folio_index,
        agreement_threshold=0.6, label='medical',
    )
    n_medical = len(medical_matches)

    # Best match per formula
    best_medical: Dict[str, FormulaMatch] = {}
    for m in medical_matches:
        if m.formula_name not in best_medical or m.agreement_rate > best_medical[m.formula_name].agreement_rate:
            best_medical[m.formula_name] = m

    medical_match_rate = 0.0
    if best_medical:
        medical_match_rate = sum(m.agreement_rate for m in best_medical.values()) / len(best_medical)

    print(f"     {n_medical} total matches across {len(best_medical)} formulae")
    print(f"     Mean agreement (best per formula): {medical_match_rate:.4f}")

    if best_medical:
        print(f"\n     Top medical matches:")
        for name, m in sorted(best_medical.items(), key=lambda x: -x[1].agreement_rate)[:10]:
            print(f"       {name:<25} agreement={m.agreement_rate:.2f}  "
                  f"folio={m.folio}  matched={m.matched_words}")

    # ── 8. Search for null (non-medical) phrases ──
    print("\n  8. Searching for null (non-medical) phrases ...")
    null_matches = _search_all_formulae(
        NULL_PHRASES, decoded_lower, folio_index,
        agreement_threshold=0.6, label='null',
    )
    n_null = len(null_matches)

    best_null: Dict[str, FormulaMatch] = {}
    for m in null_matches:
        if m.formula_name not in best_null or m.agreement_rate > best_null[m.formula_name].agreement_rate:
            best_null[m.formula_name] = m

    null_match_rate = 0.0
    if best_null:
        null_match_rate = sum(m.agreement_rate for m in best_null.values()) / len(best_null)

    print(f"     {n_null} total matches across {len(best_null)} formulae")
    print(f"     Mean agreement (best per formula): {null_match_rate:.4f}")

    if best_null:
        print(f"\n     Top null matches:")
        for name, m in sorted(best_null.items(), key=lambda x: -x[1].agreement_rate)[:10]:
            print(f"       {name:<25} agreement={m.agreement_rate:.2f}  "
                  f"folio={m.folio}  matched={m.matched_words}")

    # ── 9. Compute selectivity ──
    print("\n  9. Computing medical vs null selectivity ...")
    if null_match_rate > 0:
        medical_vs_null_ratio = medical_match_rate / null_match_rate
    elif medical_match_rate > 0:
        medical_vs_null_ratio = float('inf')
    else:
        medical_vs_null_ratio = 1.0

    # Also compare by count
    n_med_formulae_matched = len(best_medical)
    n_null_formulae_matched = len(best_null)
    print(f"     Medical formulae with matches: {n_med_formulae_matched}/{len(MEDICAL_FORMULAE)}")
    print(f"     Null phrases with matches:     {n_null_formulae_matched}/{len(NULL_PHRASES)}")
    ratio_display = f"{medical_vs_null_ratio:.2f}" if medical_vs_null_ratio != float('inf') else "INF"
    print(f"     Medical/Null agreement ratio:   {ratio_display}")

    # ── 10. Extract implied corrections from medical matches ──
    print("\n  10. Extracting implied corrections from matches ...")
    corrections = _extract_corrections(
        medical_matches, tokens, assignment, eva_to_triple, modifier_chars,
    )
    n_corrections = len(corrections)
    print(f"      {n_corrections} implied corrections extracted")

    if corrections:
        print(f"\n      Top implied corrections:")
        for c in corrections[:15]:
            print(f"        {c.triple_key:<35} {c.current_syllable:>6} -> {c.implied_syllable:<6}  "
                  f"(n={c.n_supporting_matches}, from: {c.source_formula})")

    # ── 11. Verdict ──
    is_medical = (
        n_med_formulae_matched > n_null_formulae_matched
        and medical_match_rate > null_match_rate
        and n_med_formulae_matched >= 3
    )

    if is_medical:
        verdict = (
            f"MEDICAL SIGNAL: {n_med_formulae_matched}/{len(MEDICAL_FORMULAE)} medical formulae "
            f"found (agreement={medical_match_rate:.2f}) vs "
            f"{n_null_formulae_matched}/{len(NULL_PHRASES)} null phrases "
            f"(agreement={null_match_rate:.2f}). "
            f"Ratio={ratio_display}. "
            f"{n_corrections} implied table corrections extracted."
        )
    elif n_medical > 0 or n_null > 0:
        verdict = (
            f"INCONCLUSIVE: {n_med_formulae_matched}/{len(MEDICAL_FORMULAE)} medical vs "
            f"{n_null_formulae_matched}/{len(NULL_PHRASES)} null phrases matched. "
            f"Agreement rates medical={medical_match_rate:.2f}, null={null_match_rate:.2f}. "
            f"No clear medical signal above null baseline."
        )
    else:
        verdict = (
            f"NO MATCHES: Neither medical formulae nor null phrases matched "
            f"at >=60% agreement. The Phase 16 table does not produce "
            f"recognizable Latin phrases at this threshold."
        )

    print(f"\n  Verdict: {verdict}")

    # ── 12. Save results ──
    runtime = round(time.time() - t0, 2)

    # Handle inf for JSON serialization
    ratio_for_json = medical_vs_null_ratio
    if ratio_for_json == float('inf'):
        ratio_for_json = 999.0

    result = KnownTextSearchResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_medical_formulae=len(MEDICAL_FORMULAE),
        n_null_phrases=len(NULL_PHRASES),
        medical_matches=[_convert(asdict(m)) for m in medical_matches],
        n_medical_matches=n_medical,
        medical_match_rate=round(medical_match_rate, 4),
        null_matches=[_convert(asdict(m)) for m in null_matches],
        n_null_matches=n_null,
        null_match_rate=round(null_match_rate, 4),
        medical_vs_null_ratio=round(ratio_for_json, 4),
        implied_corrections=[_convert(asdict(c)) for c in corrections],
        n_corrections=n_corrections,
        is_medical=is_medical,
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, 'known_text_search.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
    print(f"  Runtime: {runtime:.1f}s")
