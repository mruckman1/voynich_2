"""
Step 25.1 – Boustrophedon Re-Ordering and Readability Test
==========================================================
Re-order Phase 16's decoded text under four reading-direction variants
(Forward, Reversed, Boustrophedon-odd-first, Boustrophedon-even-first)
and measure whether bigram plausibility, trigram plausibility, POS trigram
validity, phrase detection, and function-word adjacency improve.

Dependency chain:
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → boustrophedon_decode.json (this step)
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
)
from voynich.core.reference import (
    LATIN_PHRASE_PATTERNS,
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
# Constants
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

VARIANTS = ['F', 'R', 'B1', 'B2']

FUNCTION_WORDS = {'et', 'in', 'de', 'cum', 'ad', 'per', 'sed', 'quia',
                  'quod', 'ab', 'ex', 'sub', 'pro', 'ne', 'ut', 'si'}

PREPOSITIONS = {'in', 'de', 'cum', 'ad', 'per', 'ab', 'ex', 'sub', 'pro',
                'contra', 'super', 'inter', 'ante', 'post'}

CONJUNCTIONS = {'et', 'sed', 'quia', 'quod', 'ut', 'si', 'ne', 'vel',
                'aut', 'atque', 'nec', 'neque'}

# Valid Latin POS trigram patterns (simplified)
VALID_POS_TRIGRAMS = {
    ('PREP', 'NOUN', 'VERB'), ('PREP', 'NOUN', 'NOUN'),
    ('PREP', 'NOUN', 'ADJ'), ('PREP', 'ADJ', 'NOUN'),
    ('NOUN', 'VERB', 'NOUN'), ('NOUN', 'VERB', 'ADJ'),
    ('NOUN', 'VERB', 'PREP'), ('NOUN', 'ADJ', 'VERB'),
    ('NOUN', 'CONJ', 'NOUN'), ('NOUN', 'NOUN', 'VERB'),
    ('ADJ', 'NOUN', 'VERB'), ('ADJ', 'NOUN', 'PREP'),
    ('ADJ', 'NOUN', 'CONJ'), ('ADJ', 'NOUN', 'NOUN'),
    ('VERB', 'NOUN', 'PREP'), ('VERB', 'PREP', 'NOUN'),
    ('VERB', 'ADJ', 'NOUN'), ('VERB', 'NOUN', 'CONJ'),
    ('VERB', 'NOUN', 'NOUN'), ('VERB', 'CONJ', 'VERB'),
    ('CONJ', 'NOUN', 'VERB'), ('CONJ', 'VERB', 'NOUN'),
    ('CONJ', 'ADJ', 'NOUN'), ('CONJ', 'NOUN', 'NOUN'),
    ('CONJ', 'NOUN', 'ADJ'), ('CONJ', 'PREP', 'NOUN'),
    ('NOUN', 'VERB', 'CONJ'), ('NOUN', 'PREP', 'NOUN'),
    ('VERB', 'NOUN', 'ADJ'), ('VERB', 'CONJ', 'NOUN'),
}

# Medical formula keywords for phrase detection
MEDICAL_KEYWORDS = {
    'recipe', 'accipe', 'coque', 'misce', 'contere', 'adde',
    'calida', 'frigida', 'sicca', 'humida', 'valet', 'contra',
    'sanat', 'mundificat', 'consolidat',
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VariantMetrics:
    bigram: float
    trigram: float
    pos_validity: float
    n_phrases: int
    func_word_adj: float


@dataclass
class BoustrophedonResult:
    timestamp: str
    n_sections: int
    n_total_lines: int
    n_total_words: int
    # Per-section × per-variant metrics
    section_variant_matrix: Dict[str, Dict[str, Dict]]
    section_rankings: Dict[str, Dict[str, List[str]]]
    # Per-folio analysis for herbal_a and biological
    per_folio_best: Dict[str, str]
    # Control sections
    control_results: Dict[str, Dict[str, Dict]]
    # Null test
    null_test: Dict[str, Any]
    # Magnitude
    magnitude_assessment: Dict[str, Any]
    # Verdict
    boustrophedon_verdict: str
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Section inference
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Line extraction (section + per-folio)
# ---------------------------------------------------------------------------

def _extract_lines(
    corpus,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> Tuple[Dict[str, List[List[str]]], Dict[str, Dict[str, List[List[str]]]]]:
    """
    Extract decoded word-lines grouped by section and by folio-within-section.

    Returns:
        section_lines: section -> list of lines
        section_folio_lines: section -> folio_id -> list of lines
    """
    section_lines: Dict[str, List[List[str]]] = defaultdict(list)
    section_folio_lines: Dict[str, Dict[str, List[List[str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for folio_id, page in corpus.pages.items():
        section = page.section
        if section == 'unknown':
            section = _folio_to_section(folio_id)

        for locus in page.loci:
            text = locus.clean_text
            if not text:
                continue
            raw_tokens = text.split()
            if not raw_tokens:
                continue

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
                section_folio_lines[section][folio_id].append(decoded_line)

    return dict(section_lines), {
        s: dict(f) for s, f in section_folio_lines.items()
    }


# ---------------------------------------------------------------------------
# Variant application
# ---------------------------------------------------------------------------

def _apply_variant(lines: List[List[str]], variant: str) -> List[str]:
    """
    Flatten lines into a word sequence under the given reading variant.

    Variants:
      F:  all lines forward
      R:  all lines reversed
      B1: odd lines (1-indexed) forward, even lines reversed
      B2: even lines (1-indexed) forward, odd lines reversed
    """
    words: List[str] = []
    for i, line in enumerate(lines):
        line_num = i + 1  # 1-indexed
        if variant == 'F':
            words.extend(line)
        elif variant == 'R':
            words.extend(reversed(line))
        elif variant == 'B1':
            # Odd lines forward, even lines reversed
            if line_num % 2 == 1:
                words.extend(line)
            else:
                words.extend(reversed(line))
        elif variant == 'B2':
            # Even lines forward, odd lines reversed
            if line_num % 2 == 0:
                words.extend(line)
            else:
                words.extend(reversed(line))
    return words


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _build_ref_bigrams(ref_words: List[str]) -> Set[Tuple[str, str]]:
    return {(ref_words[i], ref_words[i + 1])
            for i in range(len(ref_words) - 1)}


def _build_ref_trigrams(ref_words: List[str]) -> Set[Tuple[str, str, str]]:
    return {(ref_words[i], ref_words[i + 1], ref_words[i + 2])
            for i in range(len(ref_words) - 2)}


def _bigram_plausibility(
    words: List[str],
    ref_bigrams: Set[Tuple[str, str]],
    mode: str = 'strict',
) -> float:
    """Fraction of consecutive word pairs that appear in reference bigrams."""
    if len(words) < 2:
        return 0.0
    if mode == 'strict':
        hits = sum(1 for i in range(len(words) - 1)
                   if (words[i], words[i + 1]) in ref_bigrams)
    else:
        # Relaxed: edit distance 1 from any reference bigram word
        hits = 0
        for i in range(len(words) - 1):
            if (words[i], words[i + 1]) in ref_bigrams:
                hits += 1
            else:
                # Check edit distance 1 variants
                for rw1, rw2 in ref_bigrams:
                    if (_edit_dist_le1(words[i], rw1) and
                            _edit_dist_le1(words[i + 1], rw2)):
                        hits += 1
                        break
    return hits / (len(words) - 1)


def _edit_dist_le1(a: str, b: str) -> bool:
    """Check if two strings are within edit distance 1."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    # Insertion/deletion
    short, long = (a, b) if len(a) < len(b) else (b, a)
    diffs = 0
    si = li = 0
    while si < len(short) and li < len(long):
        if short[si] != long[li]:
            diffs += 1
            if diffs > 1:
                return False
            li += 1
        else:
            si += 1
            li += 1
    return True


def _trigram_plausibility(
    words: List[str],
    ref_trigrams: Set[Tuple[str, str, str]],
) -> float:
    if len(words) < 3:
        return 0.0
    hits = sum(1 for i in range(len(words) - 2)
               if (words[i], words[i + 1], words[i + 2]) in ref_trigrams)
    return hits / (len(words) - 2)


def _pos_tag_heuristic(word: str) -> str:
    """Assign a rough Latin POS tag based on suffix heuristics."""
    w = word.lower()
    if w in PREPOSITIONS:
        return 'PREP'
    if w in CONJUNCTIONS:
        return 'CONJ'
    # Verb infinitive endings
    if w.endswith(('are', 'ere', 'ire', 'ari', 'eri', 'iri')):
        return 'VERB'
    # Verb 3sg present endings
    if w.endswith(('at', 'et', 'it', 'ant', 'ent', 'unt')):
        return 'VERB'
    # Verb perfect
    if w.endswith(('avit', 'evit', 'ivit')):
        return 'VERB'
    # Adjective / noun endings (Latin declension)
    if w.endswith(('us', 'um', 'ae', 'orum', 'arum', 'ibus',
                   'onis', 'inis', 'atis')):
        return 'NOUN'
    if w.endswith(('a', 'am', 'as', 'os')):
        return 'NOUN'
    return 'UNK'


def _pos_trigram_validity(words: List[str]) -> float:
    """Fraction of POS trigrams that are valid Latin grammar sequences."""
    if len(words) < 3:
        return 0.0
    tags = [_pos_tag_heuristic(w) for w in words]
    n_trigrams = len(tags) - 2
    if n_trigrams <= 0:
        return 0.0
    # Skip trigrams with UNK — they can't be validated
    valid_count = 0
    testable_count = 0
    for i in range(n_trigrams):
        tri = (tags[i], tags[i + 1], tags[i + 2])
        if 'UNK' in tri:
            continue
        testable_count += 1
        if tri in VALID_POS_TRIGRAMS:
            valid_count += 1
    return valid_count / testable_count if testable_count > 0 else 0.0


def _phrase_detection(
    words: List[str],
    ref_word_set: set,
) -> List[Dict]:
    """
    Sliding window phrase detection. Returns list of candidate phrases
    with their scores.
    """
    phrases: List[Dict] = []
    n = len(words)

    for window_size in range(3, 9):
        for start in range(n - window_size + 1):
            window = words[start:start + window_size]

            # Dict hit density in window
            n_hits = sum(1 for w in window if w in ref_word_set)
            hit_density = n_hits / window_size

            # Medical keyword presence
            n_medical = sum(1 for w in window if w in MEDICAL_KEYWORDS)

            # Check against LATIN_PHRASE_PATTERNS
            pattern_match = False
            for pattern_name, pattern_words in LATIN_PHRASE_PATTERNS:
                if any(pw in window for pw in pattern_words):
                    pattern_match = True
                    break

            # Score: require high hit density + at least one pattern/medical match
            score = hit_density * 0.5
            if n_medical > 0:
                score += 0.3
            if pattern_match:
                score += 0.2

            if hit_density >= 0.8 and (n_medical > 0 or pattern_match):
                phrases.append({
                    'start': start,
                    'words': window,
                    'hit_density': round(hit_density, 3),
                    'n_medical': n_medical,
                    'pattern_match': pattern_match,
                    'score': round(score, 3),
                })

    # Deduplicate overlapping phrases: keep highest-scoring
    if not phrases:
        return []
    phrases.sort(key=lambda p: -p['score'])
    kept: List[Dict] = []
    used_positions: Set[int] = set()
    for p in phrases:
        positions = set(range(p['start'], p['start'] + len(p['words'])))
        if not positions & used_positions:
            kept.append(p)
            used_positions |= positions
    return kept


def _function_word_adjacency(words: List[str]) -> float:
    """
    Fraction of function words followed by a content word (noun/verb/adj).
    """
    if len(words) < 2:
        return 0.0
    n_func = 0
    n_func_before_content = 0
    content_tags = {'NOUN', 'VERB'}
    for i in range(len(words) - 1):
        if words[i] in FUNCTION_WORDS:
            n_func += 1
            next_tag = _pos_tag_heuristic(words[i + 1])
            if next_tag in content_tags:
                n_func_before_content += 1
    return n_func_before_content / n_func if n_func > 0 else 0.0


def _compute_variant_metrics(
    lines: List[List[str]],
    variant: str,
    ref_bigrams: Set[Tuple[str, str]],
    ref_trigrams: Set[Tuple[str, str, str]],
    ref_word_set: set,
) -> VariantMetrics:
    """Compute all 5 metrics for a given reading variant."""
    words = _apply_variant(lines, variant)
    return VariantMetrics(
        bigram=round(_bigram_plausibility(words, ref_bigrams, mode='strict'), 8),
        trigram=round(_trigram_plausibility(words, ref_trigrams), 8),
        pos_validity=round(_pos_trigram_validity(words), 6),
        n_phrases=len(_phrase_detection(words, ref_word_set)),
        func_word_adj=round(_function_word_adjacency(words), 6),
    )


# ---------------------------------------------------------------------------
# Per-folio analysis
# ---------------------------------------------------------------------------

def _per_folio_analysis(
    folio_lines: Dict[str, List[List[str]]],
    ref_bigrams: Set[Tuple[str, str]],
) -> Dict[str, str]:
    """For each folio, determine which variant gives highest bigram plausibility."""
    result: Dict[str, str] = {}
    for folio_id, lines in folio_lines.items():
        if len(lines) < 2:
            result[folio_id] = 'F'
            continue
        best_variant = 'F'
        best_score = -1.0
        for variant in VARIANTS:
            words = _apply_variant(lines, variant)
            score = _bigram_plausibility(words, ref_bigrams)
            if score > best_score:
                best_score = score
                best_variant = variant
        result[folio_id] = best_variant
    return result


# ---------------------------------------------------------------------------
# Null shuffle test
# ---------------------------------------------------------------------------

def _null_shuffle_test(
    section_lines: List[List[str]],
    ref_bigrams: Set[Tuple[str, str]],
    n_shuffles: int = 100,
) -> Dict[str, Any]:
    """
    Randomly shuffle line order and compare boustrophedon bigram plausibility
    against the shuffled distribution.
    """
    # Compute boustrophedon scores for B1 and B2
    b1_words = _apply_variant(section_lines, 'B1')
    b2_words = _apply_variant(section_lines, 'B2')
    b1_score = _bigram_plausibility(b1_words, ref_bigrams)
    b2_score = _bigram_plausibility(b2_words, ref_bigrams)
    best_boustro = max(b1_score, b2_score)

    # Generate null distribution
    null_scores: List[float] = []
    for _ in range(n_shuffles):
        shuffled = list(section_lines)
        random.shuffle(shuffled)
        # Apply boustrophedon to shuffled lines
        words = _apply_variant(shuffled, 'B1')
        null_scores.append(_bigram_plausibility(words, ref_bigrams))

    # Rank and p-value
    n_above = sum(1 for ns in null_scores if ns >= best_boustro)
    p_value = (n_above + 1) / (n_shuffles + 1)  # +1 for the observation itself

    return {
        'best_boustro_score': round(best_boustro, 8),
        'null_mean': round(sum(null_scores) / len(null_scores), 8) if null_scores else 0.0,
        'null_max': round(max(null_scores), 8) if null_scores else 0.0,
        'boustro_rank': n_above + 1,
        'n_shuffles': n_shuffles,
        'p_value': round(p_value, 4),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_boustrophedon() -> None:
    """Step 25.1: Boustrophedon re-ordering and readability test."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 25.1: Boustrophedon Re-Ordering and Readability Test")
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
    print(f"    {len(corpus.pages)} folios")

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
    print(f"    {len(ref_word_set)} words in reference set")

    # Build word-level bigrams and trigrams from reference corpus
    try:
        ref_word_list = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                         if len(w) >= 2]
    except Exception:
        ref_word_list = sorted(base_words)

    ref_bigrams = _build_ref_bigrams(ref_word_list[:10000])
    ref_trigrams = _build_ref_trigrams(ref_word_list[:10000])
    print(f"    {len(ref_bigrams)} reference bigrams, {len(ref_trigrams)} reference trigrams")

    # ── 4. Extract lines by section and folio ────────────────────────────
    print("\n  4. Extracting and decoding lines …")
    section_lines, section_folio_lines = _extract_lines(
        corpus, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    total_lines = sum(len(lines) for lines in section_lines.values())
    total_words = sum(
        sum(len(line) for line in lines) for lines in section_lines.values()
    )
    print(f"    {len(section_lines)} sections, {total_lines} lines, {total_words} words")

    for sec, lines in sorted(section_lines.items()):
        n_w = sum(len(l) for l in lines)
        print(f"      {sec:<20} {len(lines):>5} lines, {n_w:>6} words")

    # ── 5. Per-section × per-variant metrics ─────────────────────────────
    print("\n  5. Computing per-section × per-variant metrics …")

    section_variant_matrix: Dict[str, Dict[str, Dict]] = {}
    section_rankings: Dict[str, Dict[str, List[str]]] = {}

    for section in sorted(section_lines.keys()):
        lines = section_lines[section]
        variant_metrics: Dict[str, VariantMetrics] = {}

        for variant in VARIANTS:
            metrics = _compute_variant_metrics(
                lines, variant, ref_bigrams, ref_trigrams, ref_word_set,
            )
            variant_metrics[variant] = metrics

        # Store matrix
        section_variant_matrix[section] = {
            v: _convert(asdict(m)) for v, m in variant_metrics.items()
        }

        # Rank variants per metric
        rankings: Dict[str, List[str]] = {}
        for metric_name in ['bigram', 'trigram', 'pos_validity', 'n_phrases', 'func_word_adj']:
            ranked = sorted(
                VARIANTS,
                key=lambda v: getattr(variant_metrics[v], metric_name),
                reverse=True,
            )
            rankings[metric_name] = ranked
        section_rankings[section] = rankings

        # Print summary
        f_bg = variant_metrics['F'].bigram
        b1_bg = variant_metrics['B1'].bigram
        b2_bg = variant_metrics['B2'].bigram
        r_bg = variant_metrics['R'].bigram
        best_v = rankings['bigram'][0]
        marker = " ***" if best_v != 'F' else ""
        print(f"    {section:<20}  F={f_bg:.8f}  R={r_bg:.8f}  "
              f"B1={b1_bg:.8f}  B2={b2_bg:.8f}  best={best_v}{marker}")

        # Print other metrics
        for v in ['F', 'B1', 'B2']:
            vm = variant_metrics[v]
            print(f"      {v}: tri={vm.trigram:.8f}  pos={vm.pos_validity:.4f}  "
                  f"phrases={vm.n_phrases}  func={vm.func_word_adj:.4f}")

    # ── 6. Per-folio analysis for herbal_a and biological ────────────────
    print("\n  6. Per-folio analysis for herbal_a and biological …")
    per_folio_best: Dict[str, str] = {}

    for section in ['herbal_a', 'biological']:
        if section not in section_folio_lines:
            print(f"    {section}: not found in corpus")
            continue
        folio_lines = section_folio_lines[section]
        folio_best = _per_folio_analysis(folio_lines, ref_bigrams)
        per_folio_best.update(folio_best)

        # Count preferred variants
        variant_counts = Counter(folio_best.values())
        print(f"    {section}: {len(folio_best)} folios — "
              f"{dict(variant_counts.most_common())}")

    # ── 7. Control sections ──────────────────────────────────────────────
    print("\n  7. Control sections (pharmaceutical, recipes) …")
    control_results: Dict[str, Dict[str, Dict]] = {}

    for section in ['pharmaceutical', 'recipes']:
        if section not in section_lines:
            print(f"    {section}: not found")
            continue
        lines = section_lines[section]
        variant_metrics_ctrl: Dict[str, Dict] = {}
        for variant in VARIANTS:
            metrics = _compute_variant_metrics(
                lines, variant, ref_bigrams, ref_trigrams, ref_word_set,
            )
            variant_metrics_ctrl[variant] = _convert(asdict(metrics))
        control_results[section] = variant_metrics_ctrl

        # Which variant wins?
        best_ctrl = max(VARIANTS, key=lambda v: variant_metrics_ctrl[v]['bigram'])
        print(f"    {section}: best={best_ctrl} "
              f"(F={variant_metrics_ctrl['F']['bigram']:.8f}, "
              f"B1={variant_metrics_ctrl['B1']['bigram']:.8f})")

    # ── 8. Null shuffle test ─────────────────────────────────────────────
    print("\n  8. Null shuffle test …")

    # Run on the section with the strongest boustrophedon signal
    target_sections = ['herbal_a', 'biological']
    best_null_result: Dict[str, Any] = {}
    for section in target_sections:
        if section not in section_lines:
            continue
        null_result = _null_shuffle_test(
            section_lines[section], ref_bigrams, n_shuffles=100,
        )
        print(f"    {section}: boustro={null_result['best_boustro_score']:.8f}, "
              f"null_mean={null_result['null_mean']:.8f}, "
              f"p={null_result['p_value']:.4f}")
        if (not best_null_result or
                null_result['best_boustro_score'] > best_null_result.get('best_boustro_score', 0)):
            best_null_result = null_result
            best_null_result['section'] = section

    # ── 9. Magnitude assessment ──────────────────────────────────────────
    print("\n  9. Magnitude assessment …")

    # Find best boustrophedon bigram score across all sections
    best_boustro_bigram = 0.0
    best_boustro_section = ''
    for section, variants in section_variant_matrix.items():
        for v in ['B1', 'B2']:
            bg = variants[v]['bigram']
            if bg > best_boustro_bigram:
                best_boustro_bigram = bg
                best_boustro_section = section

    threshold = 0.01
    above_threshold = best_boustro_bigram > threshold

    magnitude = {
        'best_boustro_bigram': round(best_boustro_bigram, 8),
        'best_boustro_section': best_boustro_section,
        'threshold': threshold,
        'above_threshold': above_threshold,
    }
    print(f"    Best boustrophedon bigram: {best_boustro_bigram:.8f} "
          f"({'ABOVE' if above_threshold else 'BELOW'} threshold {threshold})")

    # ── 10. Verdict ──────────────────────────────────────────────────────
    print("\n  10. Determining verdict …")

    # Count how many sections prefer boustrophedon
    n_boustro_wins = 0
    for section in section_variant_matrix:
        rankings = section_rankings[section]
        if rankings['bigram'][0] in ('B1', 'B2'):
            n_boustro_wins += 1

    # Check null test
    null_significant = best_null_result.get('p_value', 1.0) < 0.05

    if above_threshold and null_significant:
        boustrophedon_verdict = 'CONFIRMED'
    elif (best_boustro_bigram > 0 and
          n_boustro_wins >= 1 and
          not above_threshold):
        # Beats forward by some amount but absolute level too low
        boustrophedon_verdict = 'SUGGESTIVE'
    else:
        boustrophedon_verdict = 'NOT_CONFIRMED'

    if boustrophedon_verdict == 'CONFIRMED':
        verdict = (
            f"CONFIRMED: Boustrophedon reading in {best_boustro_section} with "
            f"bigram plausibility {best_boustro_bigram:.6f} (above threshold "
            f"{threshold}), p={best_null_result.get('p_value', 'N/A')}. "
            f"Re-run all readability tests on reordered text."
        )
    elif boustrophedon_verdict == 'SUGGESTIVE':
        verdict = (
            f"SUGGESTIVE: {n_boustro_wins} section(s) prefer boustrophedon, "
            f"but absolute bigram plausibility ({best_boustro_bigram:.8f}) "
            f"is below threshold {threshold}. Direction preference is real but "
            f"decode accuracy is the bottleneck."
        )
    else:
        verdict = (
            f"NOT CONFIRMED: Forward reading is optimal or tied. "
            f"Best boustrophedon bigram: {best_boustro_bigram:.8f}. "
            f"Phase 24.10 direction signal was likely noise or an artifact."
        )

    print(f"    Verdict: {boustrophedon_verdict}")
    print(f"    {verdict}")

    elapsed = time.time() - t0

    # ── 11. Save ─────────────────────────────────────────────────────────
    result = BoustrophedonResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_sections=len(section_lines),
        n_total_lines=total_lines,
        n_total_words=total_words,
        section_variant_matrix=section_variant_matrix,
        section_rankings=section_rankings,
        per_folio_best=per_folio_best,
        control_results=control_results,
        null_test=best_null_result,
        magnitude_assessment=magnitude,
        boustrophedon_verdict=boustrophedon_verdict,
        verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = rdir / "boustrophedon_decode.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2, ensure_ascii=False)

    print(f"\n  → {out_path} ({elapsed:.1f}s)")
