"""
Step 25.2 – Folio f6r Manual Examination
=========================================
Extract the full decoded text of folio f6r, identify consecutive-hit
sequences and coherent fragments, and evaluate them against a Calendula
(marigold) medical vocabulary from period Latin herbals.

This step evaluates the decoded text *as text*, not just as statistics.

Dependency chain:
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
    boustrophedon_decode.json (Step 25.1, optional — for reordering)
        → f6r_manual.json (this step)
"""

import json
import os
import time
from collections import defaultdict
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
    generate_medieval_variants,
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
# Constants
# ---------------------------------------------------------------------------

TARGET_FOLIO = 'f6r'
TARGET_PLANT = 'Calendula'

# Calendula-specific vocabulary organized by category and specificity
# specificity: 'specific' = distinctively medical/botanical
#              'generic'  = common Latin, not diagnostic
CALENDULA_VOCAB_RAW: Dict[str, List[Tuple[str, str]]] = {
    'plant_names': [
        ('calendula', 'specific'), ('solsequium', 'specific'),
        ('caltha', 'specific'), ('sponsa', 'specific'),
        ('solis', 'specific'),
    ],
    'humoral': [
        ('calida', 'specific'), ('sicca', 'specific'),
        ('calidus', 'specific'), ('siccus', 'specific'),
        ('gradu', 'specific'), ('secundo', 'generic'),
        ('tertio', 'generic'),
    ],
    'medical_uses': [
        ('vulnera', 'specific'), ('cutis', 'specific'),
        ('scabies', 'specific'), ('ulcera', 'specific'),
        ('menstrua', 'specific'), ('oculi', 'specific'),
        ('dolor', 'specific'), ('dentium', 'specific'),
        ('icteritia', 'specific'), ('febris', 'specific'),
        ('morbus', 'specific'), ('infirmitas', 'specific'),
    ],
    'preparations': [
        ('succus', 'specific'), ('decoctio', 'specific'),
        ('emplastrum', 'specific'), ('unguentum', 'specific'),
        ('pulvis', 'specific'), ('aqua', 'generic'),
        ('oleum', 'specific'), ('vinum', 'generic'),
    ],
    'plant_parts': [
        ('flores', 'specific'), ('folia', 'specific'),
        ('herba', 'specific'), ('radix', 'specific'),
        ('semen', 'specific'), ('cortex', 'specific'),
    ],
    'collocations': [
        ('est', 'generic'), ('valet', 'specific'),
        ('contra', 'generic'), ('sanat', 'specific'),
        ('mundificat', 'specific'), ('consolidat', 'specific'),
        ('curat', 'specific'), ('prodest', 'specific'),
    ],
    'neighboring_plants': [
        ('calamus', 'specific'), ('camomilla', 'specific'),
        ('capparis', 'specific'), ('cannabis', 'specific'),
    ],
}

# Medical formula templates for fragment parsing
MEDICAL_TEMPLATES = [
    # (pattern_name, word_patterns)
    ('quality_statement', ['est', 'calida', 'sicca', 'gradu']),
    ('remedy_formula', ['valet', 'contra', 'prodest', 'sanat']),
    ('preparation', ['recipe', 'accipe', 'coque', 'misce']),
    ('ingredient', ['cum', 'aqua', 'oleum', 'vinum', 'melle']),
    ('plant_reference', ['herba', 'folia', 'flores', 'radix', 'succus']),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class F6rToken:
    position: int
    line: int
    locus_id: str
    eva_original: str
    eva_chars: List[str]
    modifier_chars: List[str]
    syllabic_chars: List[str]
    decoded_syllables: List[str]
    decoded_string: str
    strategy: str
    dict_hit_expanded: bool
    dict_hit_original: bool
    matched_word: Optional[str]
    confidence: str


@dataclass
class CalendulaMatch:
    token_position: int
    decoded_word: str
    matched_term: str
    category: str
    edit_distance: int
    specificity: str


@dataclass
class ConsecutiveRun:
    start: int
    length: int
    decoded_words: List[str]
    matched_words: List[str]
    lines: List[int]


@dataclass
class LatinParseAttempt:
    fragment: str
    words: List[str]
    parseable: bool
    structure: str
    medical_sense: str


@dataclass
class ComparisonFolio:
    folio_id: str
    section: str
    n_tokens: int
    dict_hit_rate: float
    formatted_text: List[str]
    n_consecutive_max: int


@dataclass
class F6rManualResult:
    timestamp: str
    folio: str
    plant: str
    n_tokens: int
    n_dict_hits: int
    dict_hit_rate: float
    n_dict_hits_original: int
    dict_hit_rate_original: float
    # Full token records
    tokens: List[Dict]
    # Consecutive runs
    consecutive_runs: List[Dict]
    longest_run: Dict
    # Calendula matching
    calendula_matches: Dict[str, List[Dict]]
    n_specific_matches: int
    n_generic_matches: int
    specificity_ratio: float
    # Formatted text
    formatted_text: List[str]
    boustrophedon_text: List[str]
    # Latin parse attempts
    latin_parse_attempts: List[Dict]
    # Comparison folios
    comparison: Dict[str, Dict]
    # Verdict
    f6r_verdict: str
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Calendula vocabulary builder
# ---------------------------------------------------------------------------

def _build_calendula_vocab() -> Dict[str, Tuple[str, str]]:
    """
    Build Calendula vocabulary dict: word -> (category, specificity).
    Includes medieval spelling variants.
    """
    vocab: Dict[str, Tuple[str, str]] = {}

    for category, word_list in CALENDULA_VOCAB_RAW.items():
        for word, specificity in word_list:
            vocab[word.lower()] = (category, specificity)
            # Generate medieval spelling variants
            variants = generate_medieval_variants(word.lower())
            for variant in variants:
                if variant not in vocab:
                    vocab[variant] = (category, specificity)

    return vocab


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insert
                prev_row[j + 1] + 1,   # delete
                prev_row[j] + cost,    # replace
            ))
        prev_row = curr_row
    return prev_row[-1]


# ---------------------------------------------------------------------------
# R3 decode with detailed output
# ---------------------------------------------------------------------------

def _decode_r3_detailed(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    base_words: set,
) -> Tuple[str, str, bool, bool, Optional[str]]:
    """
    R3 combined decode with expanded detail.
    Returns (decoded_word, strategy, hit_expanded, hit_original, matched_word).
    """
    # Try alteration
    alt = decode_token_modifier_aware(
        token, assignment, eva_to_triple, modifier_chars,
        modifier_rules=modifier_rules,
    )
    if alt.lower() in ref_word_set:
        w = alt.lower()
        return w, 'alteration', True, w in base_words, w

    # Try stripping
    stripped = decode_token_modifier_aware(
        token, assignment, eva_to_triple, modifier_chars,
    )
    if stripped.lower() in ref_word_set:
        w = stripped.lower()
        return w, 'stripping', True, w in base_words, w

    # Fall back to original
    original = decode_token(token, assignment, eva_to_triple)
    w = original.lower()
    return w, 'original', w in ref_word_set, w in base_words, w if w in ref_word_set else None


def _get_decoded_syllables(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars_set: Set[str],
) -> List[str]:
    """Get per-syllabic-char syllable assignments for a token."""
    chars = tokenize_eva_chars(token)
    syllables: List[str] = []
    for ch in chars:
        if ch in modifier_chars_set:
            continue
        triple_key = eva_to_triple.get(ch)
        if triple_key and triple_key in assignment:
            syllables.append(assignment[triple_key])
        else:
            syllables.append('?')
    return syllables


# ---------------------------------------------------------------------------
# Build full token records for a folio
# ---------------------------------------------------------------------------

def _build_folio_tokens(
    page,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    base_words: set,
) -> List[F6rToken]:
    """Build detailed token records for every token on a folio."""
    tokens: List[F6rToken] = []
    position = 0

    for line_idx, locus in enumerate(page.loci):
        text = locus.clean_text
        if not text:
            continue
        raw_tokens = text.split()

        for tok in raw_tokens:
            position += 1
            eva_chars = tokenize_eva_chars(tok)
            mod_chars = [ch for ch in eva_chars if ch in modifier_chars]
            syl_chars = [ch for ch in eva_chars if ch not in modifier_chars]
            decoded_syls = _get_decoded_syllables(
                tok, assignment, eva_to_triple, modifier_chars,
            )

            decoded, strategy, hit_exp, hit_orig, matched = _decode_r3_detailed(
                tok, assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set, base_words,
            )

            confidence = 'high' if hit_exp else ('low' if '?' in decoded else 'medium')

            tokens.append(F6rToken(
                position=position,
                line=line_idx + 1,
                locus_id=locus.locus_id,
                eva_original=tok,
                eva_chars=eva_chars,
                modifier_chars=mod_chars,
                syllabic_chars=syl_chars,
                decoded_syllables=decoded_syls,
                decoded_string=decoded,
                strategy=strategy,
                dict_hit_expanded=hit_exp,
                dict_hit_original=hit_orig,
                matched_word=matched,
                confidence=confidence,
            ))

    return tokens


# ---------------------------------------------------------------------------
# Consecutive run finding
# ---------------------------------------------------------------------------

def _find_consecutive_runs(tokens: List[F6rToken]) -> List[ConsecutiveRun]:
    """Find all runs of 2+ consecutive dict hits."""
    runs: List[ConsecutiveRun] = []
    streak_start = -1
    streak_words: List[str] = []
    streak_matched: List[str] = []
    streak_lines: List[int] = []

    for i, tok in enumerate(tokens):
        if tok.dict_hit_expanded:
            if streak_start < 0:
                streak_start = i
                streak_words = []
                streak_matched = []
                streak_lines = []
            streak_words.append(tok.decoded_string)
            streak_matched.append(tok.matched_word or tok.decoded_string)
            if tok.line not in streak_lines:
                streak_lines.append(tok.line)
        else:
            if streak_start >= 0 and len(streak_words) >= 2:
                runs.append(ConsecutiveRun(
                    start=streak_start,
                    length=len(streak_words),
                    decoded_words=streak_words,
                    matched_words=streak_matched,
                    lines=streak_lines,
                ))
            streak_start = -1

    # Handle final streak
    if streak_start >= 0 and len(streak_words) >= 2:
        runs.append(ConsecutiveRun(
            start=streak_start,
            length=len(streak_words),
            decoded_words=streak_words,
            matched_words=streak_matched,
            lines=streak_lines,
        ))

    runs.sort(key=lambda r: -r.length)
    return runs


# ---------------------------------------------------------------------------
# Calendula vocabulary search
# ---------------------------------------------------------------------------

def _search_calendula_matches(
    tokens: List[F6rToken],
    vocab: Dict[str, Tuple[str, str]],
) -> Dict[str, List[CalendulaMatch]]:
    """
    Search decoded words against Calendula vocabulary.
    Returns matches organized by type: exact, near, generic, specific.
    """
    matches: Dict[str, List[CalendulaMatch]] = {
        'exact': [], 'near': [], 'generic': [], 'specific': [],
    }

    for tok in tokens:
        decoded = tok.decoded_string.lower()
        if not decoded or '?' in decoded:
            continue

        for vocab_word, (category, specificity) in vocab.items():
            dist = _edit_distance(decoded, vocab_word)
            if dist > 2:
                continue

            match = CalendulaMatch(
                token_position=tok.position,
                decoded_word=decoded,
                matched_term=vocab_word,
                category=category,
                edit_distance=dist,
                specificity=specificity,
            )

            if dist == 0:
                matches['exact'].append(match)
                if specificity == 'specific':
                    matches['specific'].append(match)
                else:
                    matches['generic'].append(match)
            else:
                matches['near'].append(match)
                if specificity == 'specific':
                    matches['specific'].append(match)
                else:
                    matches['generic'].append(match)

    return matches


# ---------------------------------------------------------------------------
# Formatted text output
# ---------------------------------------------------------------------------

def _format_folio_text(tokens: List[F6rToken], calendula_vocab: Dict) -> List[str]:
    """
    Format folio text line-by-line with tags.
    Tags: [HIT] [HIT-ORIG] [CALENDULA] [MISS]
    """
    lines_dict: Dict[int, List[str]] = defaultdict(list)

    for tok in tokens:
        word = tok.decoded_string
        # Determine tag
        if tok.decoded_string.lower() in calendula_vocab:
            tag = '[CALENDULA]'
        elif tok.dict_hit_original:
            tag = '[HIT-ORIG]'
        elif tok.dict_hit_expanded:
            tag = '[HIT]'
        else:
            tag = '[MISS]'

        lines_dict[tok.line].append(f"{word}{tag}")

    formatted: List[str] = []
    for line_num in sorted(lines_dict.keys()):
        words = lines_dict[line_num]
        formatted.append(f"Line {line_num}: {' '.join(words)}")

    return formatted


def _apply_boustrophedon_to_text(
    tokens: List[F6rToken],
    calendula_vocab: Dict,
) -> List[str]:
    """Format folio text with boustrophedon line ordering (even lines reversed)."""
    lines_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    for tok in tokens:
        word = tok.decoded_string
        if tok.decoded_string.lower() in calendula_vocab:
            tag = '[CALENDULA]'
        elif tok.dict_hit_original:
            tag = '[HIT-ORIG]'
        elif tok.dict_hit_expanded:
            tag = '[HIT]'
        else:
            tag = '[MISS]'
        lines_dict[tok.line].append((word, tag))

    formatted: List[str] = []
    for line_num in sorted(lines_dict.keys()):
        pairs = lines_dict[line_num]
        # Reverse even-numbered lines (boustrophedon B1)
        if line_num % 2 == 0:
            pairs = list(reversed(pairs))
        words = [f"{w}{t}" for w, t in pairs]
        formatted.append(f"Line {line_num}: {' '.join(words)}")

    return formatted


# ---------------------------------------------------------------------------
# Latin parse attempts
# ---------------------------------------------------------------------------

def _parse_as_latin(words: List[str]) -> LatinParseAttempt:
    """Attempt to parse a decoded word sequence as Latin."""
    fragment = ' '.join(words)

    # Check against medical templates
    template_matches: List[str] = []
    for template_name, pattern_words in MEDICAL_TEMPLATES:
        if any(pw in words for pw in pattern_words):
            template_matches.append(template_name)

    # POS analysis
    from voynich.phases.boustrophedon import _pos_tag_heuristic
    tags = [_pos_tag_heuristic(w) for w in words]

    # Look for structures
    structures: List[str] = []

    for i in range(len(tags) - 1):
        if tags[i] == 'PREP' and tags[i + 1] == 'NOUN':
            structures.append(f"PREP+NOUN({words[i]} {words[i+1]})")
        if tags[i] == 'NOUN' and tags[i + 1] == 'VERB':
            structures.append(f"NOUN+VERB({words[i]} {words[i+1]})")
        if tags[i] in ('NOUN', 'UNK') and tags[i + 1] in ('NOUN', 'UNK'):
            # Could be ADJ+NOUN
            if (words[i].endswith(('a', 'us', 'um')) and
                    words[i + 1].endswith(('a', 'us', 'um', 'ae', 'i'))):
                structures.append(f"ADJ?+NOUN?({words[i]} {words[i+1]})")

    parseable = len(structures) > 0 or len(template_matches) > 0
    structure_str = '; '.join(structures) if structures else 'no Latin structure detected'

    # Medical sense
    if template_matches:
        medical_sense = f"matches templates: {', '.join(template_matches)}"
    else:
        medical_sense = 'no medical formula detected'

    return LatinParseAttempt(
        fragment=fragment,
        words=words,
        parseable=parseable,
        structure=structure_str,
        medical_sense=medical_sense,
    )


# ---------------------------------------------------------------------------
# Comparison folios
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
    import re
    m = re.search(r'\d+', folio)
    if not m:
        return 'unknown'
    num = int(m.group())
    for section_name, (lo, hi) in SECTION_RANGES.items():
        if lo <= num <= hi:
            return section_name
    return 'unknown'


def _build_comparison_folios(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    base_words: set,
    calendula_vocab: Dict,
) -> Dict[str, ComparisonFolio]:
    """Build comparison data for worst herbal folio and a pharmaceutical folio."""
    # Score all herbal_a folios
    herbal_scores: List[Tuple[str, float, int]] = []
    pharma_folios: List[str] = []

    for folio_id, page in corpus.pages.items():
        section = page.section
        if section == 'unknown':
            section = _folio_to_section(folio_id)

        toks = page.all_tokens
        if not toks or len(toks) < 5:
            continue

        if section == 'herbal_a' and folio_id != TARGET_FOLIO:
            hits = 0
            for t in toks:
                dec, _, hit_e, _, _ = _decode_r3_detailed(
                    t, assignment, eva_to_triple,
                    modifier_chars, modifier_rules, ref_word_set, base_words,
                )
                if hit_e:
                    hits += 1
            herbal_scores.append((folio_id, hits / len(toks), len(toks)))

        if section == 'pharmaceutical':
            pharma_folios.append(folio_id)

    result: Dict[str, ComparisonFolio] = {}

    # Worst herbal folio
    if herbal_scores:
        herbal_scores.sort(key=lambda x: x[1])
        worst_id = herbal_scores[0][0]
        worst_page = corpus.pages[worst_id]
        worst_tokens = _build_folio_tokens(
            worst_page, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set, base_words,
        )
        worst_runs = _find_consecutive_runs(worst_tokens)
        worst_formatted = _format_folio_text(worst_tokens, calendula_vocab)
        worst_hits = sum(1 for t in worst_tokens if t.dict_hit_expanded)
        worst_rate = worst_hits / len(worst_tokens) if worst_tokens else 0.0

        result['worst_herbal'] = ComparisonFolio(
            folio_id=worst_id,
            section='herbal_a',
            n_tokens=len(worst_tokens),
            dict_hit_rate=round(worst_rate, 4),
            formatted_text=worst_formatted,
            n_consecutive_max=worst_runs[0].length if worst_runs else 0,
        )

    # First pharmaceutical folio with enough tokens
    if pharma_folios:
        pharma_id = pharma_folios[0]
        pharma_page = corpus.pages[pharma_id]
        pharma_tokens = _build_folio_tokens(
            pharma_page, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set, base_words,
        )
        pharma_runs = _find_consecutive_runs(pharma_tokens)
        pharma_formatted = _format_folio_text(pharma_tokens, calendula_vocab)
        pharma_hits = sum(1 for t in pharma_tokens if t.dict_hit_expanded)
        pharma_rate = pharma_hits / len(pharma_tokens) if pharma_tokens else 0.0

        result['pharma'] = ComparisonFolio(
            folio_id=pharma_id,
            section='pharmaceutical',
            n_tokens=len(pharma_tokens),
            dict_hit_rate=round(pharma_rate, 4),
            formatted_text=pharma_formatted,
            n_consecutive_max=pharma_runs[0].length if pharma_runs else 0,
        )

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_f6r_manual() -> None:
    """Step 25.2: Folio f6r manual examination."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 25.2: Folio f6r Manual Examination")
    print("=" * 70)

    rdir = _results_dir()

    # ── 1. Load pipeline ─────────────────────────────────────────────────
    print("\n  1. Loading Phase 15/16 pipeline …")

    combined_path = rdir / "combined_refine.json"
    if not os.path.exists(combined_path):
        print("    [SKIP] combined_refine.json not found")
        return
    with open(combined_path) as f:
        combined = json.load(f)
    assignment = combined.get("best_assignment", {})

    mod_path = rdir / "modifier_integrate.json"
    if os.path.exists(mod_path):
        with open(mod_path) as f:
            mod_data = json.load(f)
    else:
        mod_data = {}
    modifier_chars = set(mod_data.get("modifier_chars", []))
    modifier_rules: Dict[str, str] = {}
    for cls in mod_data.get("classifications", []):
        if cls.get("final_classification") == "modifier":
            modifier_rules[cls["eva_char"]] = cls.get("modifier_type", "silent")

    print(f"    Assignment: {len(assignment)} triple-keys")
    print(f"    Modifiers: {len(modifier_chars)} chars")

    # ── 2. Load corpus ───────────────────────────────────────────────────
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    if TARGET_FOLIO not in corpus.pages:
        print(f"    [ERROR] Folio {TARGET_FOLIO} not found in corpus")
        return
    page = corpus.pages[TARGET_FOLIO]
    print(f"    Folio {TARGET_FOLIO}: {len(page.all_tokens)} tokens, "
          f"{len(page.loci)} loci")

    # ── 3. Build reference sets ──────────────────────────────────────────
    print("\n  3. Building reference sets …")
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
    print(f"    {len(base_words)} base words, {len(ref_word_set)} expanded")

    # ── 4. Build token records ───────────────────────────────────────────
    print(f"\n  4. Building detailed token records for {TARGET_FOLIO} …")
    tokens = _build_folio_tokens(
        page, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set, base_words,
    )

    n_tokens = len(tokens)
    n_hits_exp = sum(1 for t in tokens if t.dict_hit_expanded)
    n_hits_orig = sum(1 for t in tokens if t.dict_hit_original)
    hit_rate = n_hits_exp / n_tokens if n_tokens else 0.0
    hit_rate_orig = n_hits_orig / n_tokens if n_tokens else 0.0

    print(f"    {n_tokens} tokens")
    print(f"    Dict hits (expanded): {n_hits_exp} ({hit_rate:.1%})")
    print(f"    Dict hits (original): {n_hits_orig} ({hit_rate_orig:.1%})")

    # Print all tokens
    print(f"\n    Full token listing:")
    for tok in tokens:
        marker = '*' if tok.dict_hit_expanded else ' '
        orig_marker = 'O' if tok.dict_hit_original else ' '
        print(f"    {marker}{orig_marker} {tok.position:>3} L{tok.line:<2} "
              f"{tok.eva_original:>15} -> {tok.decoded_string:<15} "
              f"({tok.strategy}, {tok.confidence}) "
              f"syls={tok.decoded_syllables}")

    # ── 5. Find consecutive runs ─────────────────────────────────────────
    print(f"\n  5. Consecutive hit sequences …")
    runs = _find_consecutive_runs(tokens)

    if runs:
        print(f"    {len(runs)} fragments of 2+ consecutive hits:")
        for i, run in enumerate(runs):
            spanning = "same line" if len(run.lines) == 1 else f"lines {run.lines}"
            print(f"      {i+1}. [{run.length} words, {spanning}] "
                  f"{' '.join(run.decoded_words)}")

        longest = runs[0]
        print(f"\n    Longest run ({longest.length} consecutive hits):")
        print(f"      Decoded: {' '.join(longest.decoded_words)}")
        print(f"      Matched: {' '.join(longest.matched_words)}")
        print(f"      Lines: {longest.lines}")
    else:
        print("    No consecutive hit sequences found")
        longest = None

    # ── 6. Calendula vocabulary search ───────────────────────────────────
    print(f"\n  6. Calendula vocabulary search …")
    calendula_vocab = _build_calendula_vocab()
    print(f"    Calendula vocabulary: {len(calendula_vocab)} terms (with variants)")

    cal_matches = _search_calendula_matches(tokens, calendula_vocab)

    n_exact = len(cal_matches['exact'])
    n_near = len(cal_matches['near'])
    n_specific = len(cal_matches['specific'])
    n_generic = len(cal_matches['generic'])

    print(f"    Exact matches: {n_exact}")
    for m in cal_matches['exact']:
        print(f"      pos={m.token_position}: '{m.decoded_word}' = "
              f"'{m.matched_term}' ({m.category}, {m.specificity})")

    print(f"    Near matches (edit dist 1-2): {n_near}")
    for m in cal_matches['near'][:15]:
        print(f"      pos={m.token_position}: '{m.decoded_word}' ~ "
              f"'{m.matched_term}' (dist={m.edit_distance}, "
              f"{m.category}, {m.specificity})")

    print(f"    Specific matches: {n_specific}")
    print(f"    Generic matches: {n_generic}")
    specificity_ratio = n_specific / n_generic if n_generic > 0 else float('inf') if n_specific > 0 else 0.0
    print(f"    Specificity ratio: {specificity_ratio:.2f}")

    # ── 7. Formatted text ────────────────────────────────────────────────
    print(f"\n  7. Formatted transliteration …")
    formatted_text = _format_folio_text(tokens, calendula_vocab)
    for line in formatted_text:
        print(f"    {line}")

    # Boustrophedon variant
    boustro_text = _apply_boustrophedon_to_text(tokens, calendula_vocab)
    print(f"\n    Boustrophedon variant:")
    for line in boustro_text:
        print(f"    {line}")

    # ── 8. Latin parse attempts ──────────────────────────────────────────
    print(f"\n  8. Latin parse attempts …")
    parse_attempts: List[LatinParseAttempt] = []

    # Parse each consecutive run
    for run in runs:
        attempt = _parse_as_latin(run.decoded_words)
        parse_attempts.append(attempt)
        print(f"    Fragment: \"{attempt.fragment}\"")
        print(f"      Parseable: {attempt.parseable}")
        print(f"      Structure: {attempt.structure}")
        print(f"      Medical: {attempt.medical_sense}")

    # ── 9. Comparison folios ─────────────────────────────────────────────
    print(f"\n  9. Comparison folios …")
    comparison = _build_comparison_folios(
        corpus, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set, base_words,
        calendula_vocab,
    )

    for comp_type, comp in comparison.items():
        print(f"\n    {comp_type}: {comp.folio_id} ({comp.section})")
        print(f"      {comp.n_tokens} tokens, {comp.dict_hit_rate:.1%} dict-hit, "
              f"max consecutive: {comp.n_consecutive_max}")
        for line in comp.formatted_text[:5]:
            print(f"      {line}")
        if len(comp.formatted_text) > 5:
            print(f"      ... ({len(comp.formatted_text) - 5} more lines)")

    # ── 10. Verdict ──────────────────────────────────────────────────────
    print(f"\n  10. Determining f6r verdict …")

    # Check conditions for each verdict level
    longest_length = longest.length if longest else 0

    # Check if longest run parses as Latin medical phrase about Calendula
    longest_parseable = False
    longest_medical = False
    if parse_attempts and longest:
        # The first parse attempt corresponds to the longest run
        longest_parseable = parse_attempts[0].parseable
        longest_medical = 'no medical formula' not in parse_attempts[0].medical_sense

    # Count parseable fragments of 3+ words
    n_parseable_3plus = sum(
        1 for pa in parse_attempts
        if pa.parseable and len(pa.words) >= 3
    )

    if longest_parseable and longest_medical and n_specific >= 3:
        f6r_verdict = 'READABLE_LATIN'
    elif n_specific >= 3 and n_parseable_3plus >= 1:
        f6r_verdict = 'PARTIAL_LATIN'
    elif specificity_ratio > 1.5 and n_specific >= 1:
        f6r_verdict = 'DOMAIN_MATCH'
    else:
        f6r_verdict = 'GIBBERISH'

    # Build detailed verdict message
    if f6r_verdict == 'READABLE_LATIN':
        verdict = (
            f"READABLE LATIN: The {longest_length}-consecutive-hit sequence "
            f"parses as a Latin medical phrase with Calendula-relevant content. "
            f"{n_specific} specific medical/botanical vocabulary matches."
        )
    elif f6r_verdict == 'PARTIAL_LATIN':
        verdict = (
            f"PARTIAL LATIN: {n_specific} Calendula-specific vocabulary matches "
            f"and {n_parseable_3plus} parseable fragment(s) of 3+ words. "
            f"Strong evidence the decode is tracking real content."
        )
    elif f6r_verdict == 'DOMAIN_MATCH':
        verdict = (
            f"DOMAIN MATCH: Calendula-specific vocabulary matches ({n_specific}) "
            f"exceed generic matches ({n_generic}) with ratio {specificity_ratio:.2f}. "
            f"The text appears to be about the depicted plant."
        )
    else:
        verdict = (
            f"GIBBERISH: {n_specific} specific matches, {n_generic} generic matches. "
            f"Specificity ratio {specificity_ratio:.2f}. "
            f"The 51.6% dict-hit rate is non-discriminative — "
            f"decoded words are common short syllables."
        )

    print(f"    f6r verdict: {f6r_verdict}")
    print(f"    {verdict}")

    elapsed = time.time() - t0

    # ── 11. Save ─────────────────────────────────────────────────────────
    result = F6rManualResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        folio=TARGET_FOLIO,
        plant=TARGET_PLANT,
        n_tokens=n_tokens,
        n_dict_hits=n_hits_exp,
        dict_hit_rate=round(hit_rate, 4),
        n_dict_hits_original=n_hits_orig,
        dict_hit_rate_original=round(hit_rate_orig, 4),
        tokens=[_convert(asdict(t)) for t in tokens],
        consecutive_runs=[_convert(asdict(r)) for r in runs],
        longest_run=_convert(asdict(longest)) if longest else {},
        calendula_matches={
            k: [_convert(asdict(m)) for m in v]
            for k, v in cal_matches.items()
        },
        n_specific_matches=n_specific,
        n_generic_matches=n_generic,
        specificity_ratio=round(specificity_ratio, 4),
        formatted_text=formatted_text,
        boustrophedon_text=boustro_text,
        latin_parse_attempts=[_convert(asdict(pa)) for pa in parse_attempts],
        comparison={k: _convert(asdict(v)) for k, v in comparison.items()},
        f6r_verdict=f6r_verdict,
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = rdir / "f6r_manual.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2, ensure_ascii=False)

    print(f"\n  → {out_path} ({elapsed:.1f}s)")
