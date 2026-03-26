"""
Phase 66: Shared Validation Infrastructure (V1-V5)
====================================================
Provides blind null controls, known-answer calibration, anchor word
verification, research-constrained prompt templates, and cross-folio
consistency checking for all Tier 1 (LLM-based) tracks.

This is a library module -- no run_*() entry point, no CLI command.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/corrected_coda.json       (Phase 60A)
    results/word_catalog.json         (Phase 52)
    data/reference/latin/             (Latin reference corpus)
"""

import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.core.stats import syllabify_latin
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
    decode_token_cvc_v2,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ControlScores:
    real_mean: float = 0.0
    shuffled_mean: float = 0.0
    null_mean: float = 0.0
    real_std: float = 0.0
    shuffled_std: float = 0.0
    null_std: float = 0.0
    z_vs_shuffled: float = 0.0
    z_vs_null: float = 0.0
    shuffled_ratio: float = 0.0
    null_ratio: float = 0.0
    v1_passed: bool = False


@dataclass
class KnownAnswerResult:
    n_tested: int = 0
    n_passed: int = 0
    mean_char_accuracy: float = 0.0
    mean_word_accuracy: float = 0.0
    mean_boundary_f1: float = 0.0
    v2_passed: bool = False
    per_passage: List[Dict] = field(default_factory=list)


@dataclass
class AnchorResult:
    n_testable: int = 0
    n_preserved: int = 0
    n_broken: int = 0
    preservation_rate: float = 0.0
    preserved_words: List[Dict] = field(default_factory=list)
    broken_words: List[Dict] = field(default_factory=list)
    v3_passed: bool = False


@dataclass
class ConsistencyResult:
    n_shared_sequences: int = 0
    n_consistent: int = 0
    n_inconsistent: int = 0
    consistency_rate: float = 0.0
    v5_passed: bool = False


# ---------------------------------------------------------------------------
# V0: Edit distance
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


# ---------------------------------------------------------------------------
# V1: Blind null controls
# ---------------------------------------------------------------------------

def generate_shuffled_control(stream: str, seed: int) -> str:
    """Character-level shuffle of a decoded character stream.

    Preserves character frequencies but destroys all sequential structure.
    """
    chars = list(stream)
    rng = random.Random(seed)
    rng.shuffle(chars)
    return ''.join(chars)


def generate_null_control(
    n_chars: int,
    corpus_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    seed: int,
) -> str:
    """Generate a null decoded stream by creating null EVA tokens and CVC-decoding.

    Uses the bigram model from null_corpus.py to generate synthetic tokens
    that match Voynich's EVA-character statistics, then decodes them through
    the same CVC pipeline.
    """
    from voynich.phases.null_corpus import (
        _build_eva_bigram_model,
        _generate_null_corpus,
    )

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        corpus_tokens)

    # Generate enough null tokens to fill n_chars of decoded output
    # Overshoot by 2x to ensure we have enough after decode
    null_tokens = _generate_null_corpus(
        bigram_probs, initial_probs, token_lengths,
        n_tokens=max(100, n_chars), seed=seed,
    )

    # CVC decode the null tokens
    decoded = decode_corpus_cvc_v2(null_tokens, assignment, eva_to_triple,
                                   coda_table)

    # Concatenate and trim to target length
    stream = ''.join(d for d in decoded if d and '?' not in d)
    return stream[:n_chars] if len(stream) >= n_chars else stream


def generate_controls(
    real_passages: List[Dict[str, Any]],
    corpus_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    base_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate shuffled and null controls for a list of real passages.

    Each passage dict must have a 'stream' key with the decoded text.
    Returns two lists (shuffled, null), each with the same metadata as
    the real passages but with substituted streams.
    """
    shuffled = []
    nulls = []

    for i, passage in enumerate(real_passages):
        stream = passage['stream']

        # Shuffled control
        shuf = dict(passage)
        shuf['stream'] = generate_shuffled_control(stream, base_seed + i)
        shuf['control_type'] = 'SHUFFLED'
        shuffled.append(shuf)

        # Null control
        null_stream = generate_null_control(
            len(stream), corpus_tokens, assignment,
            eva_to_triple, coda_table, base_seed + 1000 + i,
        )
        nul = dict(passage)
        nul['stream'] = null_stream
        nul['control_type'] = 'NULL'
        nulls.append(nul)

    return shuffled, nulls


def score_against_controls(
    real_scores: List[float],
    shuffled_scores: List[float],
    null_scores: List[float],
) -> ControlScores:
    """Compare real scores against shuffled and null control scores.

    Gate: z >= 2.0 vs both controls AND ratio >= 2.0.
    """
    real_arr = np.array(real_scores) if real_scores else np.array([0.0])
    shuf_arr = np.array(shuffled_scores) if shuffled_scores else np.array([0.0])
    null_arr = np.array(null_scores) if null_scores else np.array([0.0])

    real_mean = float(np.mean(real_arr))
    shuf_mean = float(np.mean(shuf_arr))
    null_mean = float(np.mean(null_arr))
    real_std = float(np.std(real_arr))
    shuf_std = float(np.std(shuf_arr))
    null_std = float(np.std(null_arr))

    z_shuf = ((real_mean - shuf_mean) / shuf_std) if shuf_std > 0 else 0.0
    z_null = ((real_mean - null_mean) / null_std) if null_std > 0 else 0.0

    shuf_ratio = real_mean / (shuf_mean + 1e-6)
    null_ratio = real_mean / (null_mean + 1e-6)

    passed = (
        z_shuf >= 2.0 and z_null >= 2.0
        and shuf_ratio >= 2.0 and null_ratio >= 2.0
    )

    return ControlScores(
        real_mean=real_mean,
        shuffled_mean=shuf_mean,
        null_mean=null_mean,
        real_std=real_std,
        shuffled_std=shuf_std,
        null_std=null_std,
        z_vs_shuffled=round(z_shuf, 3),
        z_vs_null=round(z_null, 3),
        shuffled_ratio=round(shuf_ratio, 3),
        null_ratio=round(null_ratio, 3),
        v1_passed=passed,
    )


# ---------------------------------------------------------------------------
# V2: Known-answer calibration
# ---------------------------------------------------------------------------

def _load_t1_words() -> List[Dict[str, Any]]:
    """Load T1 identifications from word_catalog.json."""
    rd = str(_results_dir())
    path = os.path.join(rd, 'word_catalog.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    ids = data.get('single_token_ids', [])
    return [x for x in ids if x.get('tier') == 'T1']


def _build_reverse_assignment(
    assignment: Dict[str, str],
) -> Dict[str, List[str]]:
    """Build syllable -> list of triples reverse lookup."""
    rev: Dict[str, List[str]] = {}
    for triple, syllable in assignment.items():
        if syllable not in rev:
            rev[syllable] = []
        rev[syllable].append(triple)
    return rev


def _build_triple_to_eva(
    eva_to_triple: Dict[str, str],
) -> Dict[str, List[str]]:
    """Build triple -> list of EVA chars reverse lookup."""
    rev: Dict[str, List[str]] = {}
    for eva_char, triple in eva_to_triple.items():
        if triple not in rev:
            rev[triple] = []
        rev[triple].append(eva_char)
    return rev


def forward_encode_word(
    latin_word: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
) -> Optional[str]:
    """Forward-encode a Latin word through the tachygraphic model.

    Syllabify -> find matching triple -> EVA char -> CVC decode.
    Returns the decoded form, or None if encoding fails.

    NOTE: This is approximate because the assignment is many-to-one.
    We pick the first matching triple/EVA char found.
    """
    syllables = syllabify_latin(latin_word.lower())
    if not syllables:
        return None

    rev_assignment = _build_reverse_assignment(assignment)
    triple_to_eva = _build_triple_to_eva(eva_to_triple)

    encoded_evas = []
    for syl in syllables:
        # Find a triple that maps to this syllable (or close)
        triples = rev_assignment.get(syl, [])
        if not triples:
            # Try first 2 chars as approximation
            triples = rev_assignment.get(syl[:2], [])
        if not triples:
            return None

        # Find an EVA char for the first matching triple
        triple = triples[0]
        eva_chars = triple_to_eva.get(triple, [])
        if not eva_chars:
            return None
        encoded_evas.append(eva_chars[0])

    # CVC decode the constructed EVA sequence
    eva_token = ''.join(encoded_evas)
    result = decode_token_cvc_v2(eva_token, assignment, eva_to_triple,
                                  coda_table)
    return result.decoded_cvc if result.decoded_cvc else None


def build_known_answer_passages(
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    n: int = 10,
) -> List[Dict[str, Any]]:
    """Build known-answer calibration passages from Circa Instans.

    Takes Latin pharmaceutical text, forward-encodes it through the
    tachygraphic model, and produces passages where we know the answer.
    """
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    latin_texts = ref.get_combined_tokens('latin')

    if not latin_texts:
        return []

    passages = []
    # Take consecutive word chunks
    chunk_size = 8  # 8 words per passage
    rng = random.Random(42)
    indices = list(range(0, len(latin_texts) - chunk_size, chunk_size))
    rng.shuffle(indices)

    for idx in indices[:n * 5]:  # try 5x to find n good passages
        if len(passages) >= n:
            break

        words = latin_texts[idx:idx + chunk_size]
        # Forward-encode each word
        encoded_parts = []
        success = True
        for w in words:
            w_clean = w.lower().strip('.,;:!?()[]')
            if len(w_clean) < 3:
                continue
            encoded = forward_encode_word(w_clean, assignment, eva_to_triple,
                                           coda_table)
            if encoded:
                encoded_parts.append((w_clean, encoded))
            else:
                success = False
                break

        if not success or len(encoded_parts) < 3:
            continue

        known_latin = ' '.join(w for w, _ in encoded_parts)
        decoded_stream = ''.join(e for _, e in encoded_parts)

        passages.append({
            'stream': decoded_stream,
            'known_latin': known_latin,
            'known_words': [w for w, _ in encoded_parts],
            'folio': 'CALIBRATION',
            'section': 'pharmaceutical',
            'control_type': None,
        })

    return passages


def score_known_answer(
    llm_result: Dict[str, Any],
    known_answer: Dict[str, Any],
) -> Dict[str, Any]:
    """Score LLM reading accuracy against known plaintext.

    Returns character accuracy, word accuracy, and boundary F1.
    """
    proposed = llm_result.get('segmented_text', '')
    known_text = known_answer['known_latin']
    known_words = known_answer.get('known_words', [])

    # Character-level accuracy (ignoring spaces)
    proposed_flat = proposed.replace(' ', '').lower()
    known_flat = known_text.replace(' ', '').lower()
    if not known_flat:
        return {'char_accuracy': 0.0, 'word_accuracy': 0.0,
                'boundary_f1': 0.0}

    char_ed = _edit_distance(proposed_flat, known_flat)
    char_accuracy = 1.0 - char_ed / max(len(proposed_flat), len(known_flat), 1)

    # Word-level accuracy
    proposed_words = [w.lower() for w in proposed.split() if w]
    word_matches = sum(1 for pw in proposed_words if pw in known_words)
    word_accuracy = (word_matches / len(known_words)
                     if known_words else 0.0)

    # Boundary F1 (simplified: check word boundary positions)
    def _find_boundaries(text: str) -> Set[int]:
        pos = 0
        boundaries = set()
        for word in text.split():
            pos += len(word)
            boundaries.add(pos)
        return boundaries

    proposed_bounds = _find_boundaries(proposed)
    known_bounds = _find_boundaries(known_text)

    if not known_bounds or not proposed_bounds:
        boundary_f1 = 0.0
    else:
        # Allow tolerance of ±2 characters
        tp = sum(1 for pb in proposed_bounds
                 if any(abs(pb - kb) <= 2 for kb in known_bounds))
        precision = tp / len(proposed_bounds) if proposed_bounds else 0.0
        recall = tp / len(known_bounds) if known_bounds else 0.0
        boundary_f1 = (2 * precision * recall / (precision + recall)
                       if (precision + recall) > 0 else 0.0)

    return {
        'char_accuracy': round(char_accuracy, 4),
        'word_accuracy': round(word_accuracy, 4),
        'boundary_f1': round(boundary_f1, 4),
        'n_proposed_words': len(proposed_words),
        'n_known_words': len(known_words),
    }


# ---------------------------------------------------------------------------
# V3: Anchor word verification
# ---------------------------------------------------------------------------

SIGNAL_WORDS_SET = set(SIGNAL_WORDS_51.keys())


def verify_anchor_preservation(
    llm_result: Dict[str, Any],
    passage_stream: str,
    signal_words: Optional[Set[str]] = None,
) -> AnchorResult:
    """Check whether confirmed signal words are preserved in LLM reading.

    The 70 signal words are statistically validated. The LLM's readings
    MUST preserve them — breaking "cola" into "co" + "la" contradicts
    established findings.
    """
    if signal_words is None:
        signal_words = SIGNAL_WORDS_SET

    proposed_text = llm_result.get('segmented_text', '')
    # Reconstruct the stream from proposed text (remove spaces)
    proposed_flat = proposed_text.replace(' ', '').lower()

    preserved = []
    broken = []

    for sw in sorted(signal_words, key=len, reverse=True):
        if sw not in passage_stream:
            continue  # not in this passage

        # Check if the LLM's segmentation keeps it intact
        # Find the word in the proposed text
        if sw in proposed_text.lower().split():
            preserved.append({'signal_word': sw, 'status': 'PRESERVED'})
        elif sw in proposed_flat:
            # Present in flat text but broken across word boundaries
            # Check if any single proposed word contains it
            found_in_word = any(
                sw in pw.lower()
                for pw in proposed_text.split()
            )
            if found_in_word:
                preserved.append({'signal_word': sw, 'status': 'PRESERVED'})
            else:
                broken.append({'signal_word': sw, 'status': 'BROKEN'})
        else:
            # Not found at all in the proposed text
            broken.append({'signal_word': sw, 'status': 'MISSING'})

    n_testable = len(preserved) + len(broken)
    rate = len(preserved) / n_testable if n_testable > 0 else 1.0

    return AnchorResult(
        n_testable=n_testable,
        n_preserved=len(preserved),
        n_broken=len(broken),
        preservation_rate=round(rate, 4),
        preserved_words=preserved,
        broken_words=broken,
        v3_passed=rate >= 0.70 and len(broken) == 0,
    )


# ---------------------------------------------------------------------------
# V4: Research-constrained prompt
# ---------------------------------------------------------------------------

RESEARCH_PROMPT_TEMPLATE = """\
You are examining a decoded medieval Latin pharmaceutical text. The text was \
encoded in Italian syllabic tachygraphy and has been partially decoded at the \
syllable level by a computational pipeline. The syllable-level decode has been \
statistically validated (p=0.006 coherence, 83% Costamagna attestation).

DECODED TEXT (no word boundaries):
{decoded_stream}

═══════════════════════════════════════════════════════════════════
HARD CONSTRAINTS — You MUST follow these. They are statistically confirmed.
═══════════════════════════════════════════════════════════════════

CONFIRMED SIGNAL WORDS (p=0.006 — these ARE real words in the text):
  Function words: di, se, ne, co, de, la, li, ha, fa, si, ci, te, ti, tu, ni, \
bi, du, su
  Pharmaceutical: sero, cola, sene, codi, tere, raso, cor, din, cone, bene, \
decor
  Italian verbs: dise, dice, dico, dicu, diga (forms of "dire" = to say)

  If any of these appear as substrings in the decoded text above, they MUST be \
preserved as whole words or as part of longer words. Do NOT split "cola" into \
"co" + "la". Do NOT split "bene" into "be" + "ne".

CONFIRMED T1 IDENTIFICATIONS (p=0.009 — full words identified in the text):
  ratione, coralli, diasene, stercora, radicom, commune, secundi, codex, rabidi

LANGUAGE: Macaronic Latin-Italian with Gallo-Italic phonological features:
  - Degemination: bela (not bella), sene (not senna), corali (not coralli)
  - Northern Italian -on for Latin -um in some accusatives
  - Tuscan function words (ci, si, tu) with northern content vocabulary

CODA CONSONANTS (confirmed: 70.4% word-final position):
  Words ending in n, r, s, t are common (from coda markers)
  Common endings: -en (3rd decl acc), -in (prepositional), -an (1st decl acc), \
-on (2nd decl acc, Gallo-Italic), -er, -or, -es, -is

MANUSCRIPT CONTEXT:
  - Section: {section}
  - Folio: {folio}
  - Content type: pharmaceutical recipes, herbal descriptions, medical \
instructions

{anchor_section}

═══════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════

1. Insert word boundaries into the decoded text
2. For each proposed word, provide the Latin/Italian word and confidence
3. Attempt a translation
4. Mark genuinely unreadable sections with [?]
5. Do NOT force a reading where you are not confident — [?] is preferable \
to guessing

Respond ONLY with JSON:
{{
  "segmented_text": "...",
  "words": [
    {{"decoded": "...", "latin": "...", "meaning": "...", \
"confidence": "HIGH|MEDIUM|LOW"}},
  ],
  "translation": "...",
  "notes": "...",
  "readable_fraction": 0.0,
  "uncertain_regions": ["..."]
}}
"""


def build_anchor_section(
    passage_stream: str,
    signal_words: Optional[Dict[str, Dict]] = None,
    t1_words: Optional[List[Dict]] = None,
    folio: str = '?',
) -> str:
    """Build the ANCHOR WORDS section of the prompt for a specific passage."""
    if signal_words is None:
        signal_words = SIGNAL_WORDS_51
    if t1_words is None:
        t1_words = _load_t1_words()

    found = []

    for sw, info in signal_words.items():
        if sw in passage_stream:
            count = passage_stream.count(sw)
            gloss = info.get('gloss', '?')
            found.append(f"  '{sw}' ({gloss}) — appears {count}x")

    for t1 in t1_words:
        t1_folios = t1.get('folios', [])
        if folio in t1_folios:
            found.append(f"  '{t1['latin_word']}' — T1 identification on "
                         f"this folio")

    if found:
        return ("ANCHOR WORDS CONFIRMED IN THIS PASSAGE:\n"
                + '\n'.join(found))
    return ("ANCHOR WORDS: No confirmed signal words detected in this "
            "specific passage.")


# ---------------------------------------------------------------------------
# V5: Cross-folio consistency
# ---------------------------------------------------------------------------

def check_cross_folio_consistency(
    all_readings: List[Dict[str, Any]],
    min_subseq_len: int = 6,
) -> ConsistencyResult:
    """Check whether the same decoded sequence gets the same reading everywhere.

    Extract all decoded substrings of length >= min_subseq_len that appear
    in 2+ passages. For each, check whether the LLM's proposed segmentation
    is consistent.
    """
    # Collect segmentations keyed by decoded substring
    sequence_readings: Dict[str, List[Dict]] = {}

    for reading in all_readings:
        if reading.get('control_type'):
            continue
        stream = reading.get('stream', '')
        segmented = reading.get('segmented_text', '')
        folio = reading.get('folio', '?')

        if not stream or not segmented:
            continue

        # Build a position map: stream position -> segmented position
        # (segmented has spaces, stream doesn't)
        seg_flat = segmented.replace(' ', '')

        for start in range(len(stream) - min_subseq_len + 1):
            subseq = stream[start:start + min_subseq_len]

            # Find how this substring was segmented
            idx = seg_flat.find(subseq)
            if idx == -1:
                continue

            # Find the segmentation around this position
            pos = 0
            seg_start_word = 0
            for wi, word in enumerate(segmented.split()):
                if pos + len(word) > idx:
                    seg_start_word = wi
                    break
                pos += len(word)

            seg_end_word = seg_start_word
            pos2 = pos
            for wi in range(seg_start_word, len(segmented.split())):
                word = segmented.split()[wi]
                pos2 += len(word)
                if pos2 >= idx + min_subseq_len:
                    seg_end_word = wi + 1
                    break

            seg_words = ' '.join(
                segmented.split()[seg_start_word:seg_end_word])

            if subseq not in sequence_readings:
                sequence_readings[subseq] = []
            sequence_readings[subseq].append({
                'folio': folio,
                'segmentation': seg_words,
            })

    # Check consistency for sequences appearing 2+ times
    consistent = 0
    inconsistent = 0
    total = 0

    for subseq, readings in sequence_readings.items():
        if len(readings) < 2:
            continue
        total += 1
        segmentations = set(r['segmentation'] for r in readings)
        if len(segmentations) == 1:
            consistent += 1
        else:
            inconsistent += 1

    rate = consistent / total if total > 0 else 0.0

    return ConsistencyResult(
        n_shared_sequences=total,
        n_consistent=consistent,
        n_inconsistent=inconsistent,
        consistency_rate=round(rate, 4),
        v5_passed=rate >= 0.4 and total >= 3,
    )


# ---------------------------------------------------------------------------
# Helper: build 10K dictionary
# ---------------------------------------------------------------------------

def build_10k_dict() -> Set[str]:
    """Build a 10K-word Latin pharmaceutical dictionary."""
    ref = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref.get_combined_tokens('latin'))
    word_set, _ = build_expanded_word_set(base_words)
    # The expanded set is ~131K. For 10K, take the most frequent.
    all_tokens = ref.get_combined_tokens('latin')
    freq = Counter(w.lower() for w in all_tokens)
    top_10k = {w for w, _ in freq.most_common(10000)}
    return top_10k | SIGNAL_WORDS_SET


def compute_dict_hit_for_words(
    words: List[str],
    dict_set: Set[str],
) -> float:
    """Fraction of words that appear in the dictionary."""
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.lower() in dict_set)
    return hits / len(words)
