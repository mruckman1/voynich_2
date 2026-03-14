"""
Phase 51 Track A: Reverse Suffix Calibration
=============================================
Use 70 confirmed signal words as ground truth to build a calibrated
EVA suffix -> Latin ending map, then POS-tag the entire 36K-token corpus.

Dependency chain:
    signal_bigrams.json        (Step 29.1 -- per-token decoded + classifications)
    combined_refine.json       (Step 15   -- best_assignment)
    modifier_integrate.json    (Step 16   -- modifier chars)
        -> suffix_calibration.json  (this step)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars

from voynich.phases.morpheme_grid import (
    decompose_token_morphemes,
    KNOWN_PREFIXES,
    KNOWN_SUFFIXES,
)
from voynich.phases.suffix_grammar import (
    LATIN_NOUN_ENDINGS,
    LATIN_VERB_ENDINGS,
    _classify_latin_ending,
)

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
# 70 Hardcoded Signal Words (ground truth)
# ---------------------------------------------------------------------------
# From README: 51 Latin-10K (Phase 36) + 22 Italian-only (Phases 37-38)
# minus 3 overlap (dise, cu, dedi) = 70 unique.

SIGNAL_WORDS_51: Dict[str, Dict[str, Any]] = {
    # ── 51 Latin-10K signal words (Phase 36) ──
    'di':    {'sigma': 129.71, 'real_count': 1353, 'type': 'function', 'lang': 'shared', 'gloss': 'of'},
    'se':    {'sigma': 105.12, 'real_count': 592,  'type': 'function', 'lang': 'shared', 'gloss': 'if/self'},
    'ne':    {'sigma': 93.52,  'real_count': 1470, 'type': 'function', 'lang': 'shared', 'gloss': 'not/nor'},
    'dise':  {'sigma': 77.77,  'real_count': 71,   'type': 'content',  'lang': 'italian', 'gloss': 'says'},
    'sero':  {'sigma': 70.12,  'real_count': 135,  'type': 'pharm',    'lang': 'shared', 'gloss': 'serum/evening'},
    'bi':    {'sigma': 63.23,  'real_count': 342,  'type': 'function', 'lang': 'shared', 'gloss': 'twice'},
    'ce':    {'sigma': 61.19,  'real_count': 353,  'type': 'function', 'lang': 'shared', 'gloss': 'here/this'},
    'co':    {'sigma': 52.53,  'real_count': 490,  'type': 'function', 'lang': 'shared', 'gloss': 'with'},
    'ni':    {'sigma': 51.38,  'real_count': 494,  'type': 'function', 'lang': 'shared', 'gloss': 'nor'},
    'rati':  {'sigma': 50.44,  'real_count': 156,  'type': 'content',  'lang': 'latin',  'gloss': 'reckoning'},
    'sene':  {'sigma': 47.71,  'real_count': 242,  'type': 'botanical','lang': 'shared', 'gloss': 'senna'},
    'de':    {'sigma': 47.34,  'real_count': 471,  'type': 'function', 'lang': 'shared', 'gloss': 'of/from'},
    'bene':  {'sigma': 46.41,  'real_count': 152,  'type': 'quality',  'lang': 'shared', 'gloss': 'well/good'},
    'du':    {'sigma': 46.10,  'real_count': 189,  'type': 'function', 'lang': 'shared', 'gloss': 'two/of the'},
    'ci':    {'sigma': 37.82,  'real_count': 64,   'type': 'function', 'lang': 'shared', 'gloss': 'there/to it'},
    'te':    {'sigma': 36.57,  'real_count': 122,  'type': 'function', 'lang': 'shared', 'gloss': 'you/thee'},
    'bo':    {'sigma': 32.57,  'real_count': 124,  'type': 'function', 'lang': 'shared', 'gloss': 'function'},
    'dira':  {'sigma': 32.41,  'real_count': 50,   'type': 'quality',  'lang': 'shared', 'gloss': 'dire/harsh'},
    'la':    {'sigma': 32.06,  'real_count': 117,  'type': 'function', 'lang': 'shared', 'gloss': 'the (fem.)'},
    'si':    {'sigma': 29.44,  'real_count': 170,  'type': 'function', 'lang': 'shared', 'gloss': 'yes/self'},
    'sere':  {'sigma': 28.53,  'real_count': 73,   'type': 'quality',  'lang': 'shared', 'gloss': 'serene'},
    'nera':  {'sigma': 27.82,  'real_count': 62,   'type': 'quality',  'lang': 'italian','gloss': 'black (fem.)'},
    'ra':    {'sigma': 23.28,  'real_count': 121,  'type': 'function', 'lang': 'shared', 'gloss': 'function'},
    'sera':  {'sigma': 21.69,  'real_count': 166,  'type': 'content',  'lang': 'shared', 'gloss': 'evening'},
    'do':    {'sigma': 21.61,  'real_count': 29,   'type': 'function', 'lang': 'shared', 'gloss': 'I give'},
    're':    {'sigma': 21.11,  'real_count': 21,   'type': 'function', 'lang': 'shared', 'gloss': 'thing/about'},
    'so':    {'sigma': 21.07,  'real_count': 242,  'type': 'function', 'lang': 'shared', 'gloss': 'I am/above'},
    'cu':    {'sigma': 20.19,  'real_count': 144,  'type': 'function', 'lang': 'italian','gloss': 'with (dialectal)'},
    'ti':    {'sigma': 19.95,  'real_count': 65,   'type': 'function', 'lang': 'shared', 'gloss': 'you (dat.)'},
    'su':    {'sigma': 19.75,  'real_count': 46,   'type': 'function', 'lang': 'shared', 'gloss': 'on/above'},
    'diri':  {'sigma': 19.46,  'real_count': 31,   'type': 'content',  'lang': 'italian','gloss': 'to say (inf.)'},
    'ru':    {'sigma': 18.47,  'real_count': 59,   'type': 'function', 'lang': 'shared', 'gloss': 'function'},
    'cola':  {'sigma': 16.73,  'real_count': 68,   'type': 'pharm',    'lang': 'shared', 'gloss': 'strain (v.)'},
    'nu':    {'sigma': 16.39,  'real_count': 47,   'type': 'function', 'lang': 'shared', 'gloss': 'function'},
    'ha':    {'sigma': 15.50,  'real_count': 7,    'type': 'function', 'lang': 'shared', 'gloss': 'has (It.)'},
    'li':    {'sigma': 15.45,  'real_count': 94,   'type': 'function', 'lang': 'shared', 'gloss': 'the (pl.)'},
    'dedi':  {'sigma': 15.20,  'real_count': 68,   'type': 'content',  'lang': 'italian','gloss': 'I gave'},
    'ga':    {'sigma': 11.02,  'real_count': 6,    'type': 'function', 'lang': 'shared', 'gloss': 'function'},
    'tere':  {'sigma': 10.96,  'real_count': 10,   'type': 'content',  'lang': 'latin',  'gloss': 'to rub'},
    'sede':  {'sigma': 10.76,  'real_count': 19,   'type': 'content',  'lang': 'shared', 'gloss': 'seat/see'},
    'tela':  {'sigma': 10.61,  'real_count': 20,   'type': 'content',  'lang': 'shared', 'gloss': 'cloth/web'},
    'tu':    {'sigma': 10.03,  'real_count': 15,   'type': 'function', 'lang': 'shared', 'gloss': 'you'},
    'dico':  {'sigma': 9.88,   'real_count': 48,   'type': 'content',  'lang': 'shared', 'gloss': 'I say'},
    'ge':    {'sigma': 9.66,   'real_count': 18,   'type': 'function', 'lang': 'shared', 'gloss': 'function'},
    'sese':  {'sigma': 9.50,   'real_count': 18,   'type': 'function', 'lang': 'latin',  'gloss': 'themselves'},
    'hi':    {'sigma': 8.22,   'real_count': 11,   'type': 'function', 'lang': 'shared', 'gloss': 'these'},
    'raro':  {'sigma': 7.62,   'real_count': 15,   'type': 'quality',  'lang': 'shared', 'gloss': 'rarely'},
    'fe':    {'sigma': 6.32,   'real_count': 5,    'type': 'function', 'lang': 'shared', 'gloss': 'made/faith'},
    'fa':    {'sigma': 5.58,   'real_count': 10,   'type': 'function', 'lang': 'shared', 'gloss': 'does/makes'},
    'raso':  {'sigma': 3.39,   'real_count': 6,    'type': 'content',  'lang': 'latin',  'gloss': 'scraped'},
    'dici':  {'sigma': 2.51,   'real_count': 5,    'type': 'content',  'lang': 'shared', 'gloss': 'to be said'},
    # ── 22 Italian-only signal words (Phases 37-38) ──
    # (dise, cu, dedi already above — 19 new entries)
    'be':      {'sigma': 134.65, 'real_count': 547,  'type': 'function', 'lang': 'italian', 'gloss': 'well (It. variant)'},
    'cora':    {'sigma': 98.68,  'real_count': 1114, 'type': 'content',  'lang': 'italian', 'gloss': 'heart'},
    'bela':    {'sigma': 43.75,  'real_count': 400,  'type': 'quality',  'lang': 'italian', 'gloss': 'beautiful'},
    'cedi':    {'sigma': 23.48,  'real_count': 24,   'type': 'content',  'lang': 'italian', 'gloss': 'yield'},
    'didi':    {'sigma': 18.82,  'real_count': 136,  'type': 'content',  'lang': 'italian', 'gloss': 'gave (pl.)'},
    'dice':    {'sigma': 18.44,  'real_count': 51,   'type': 'content',  'lang': 'italian', 'gloss': 'says'},
    'deco':    {'sigma': 17.98,  'real_count': 65,   'type': 'content',  'lang': 'italian', 'gloss': 'I decorate'},
    'cose':    {'sigma': 16.30,  'real_count': 14,   'type': 'content',  'lang': 'italian', 'gloss': 'things'},
    'beri':    {'sigma': 15.52,  'real_count': 20,   'type': 'content',  'lang': 'italian', 'gloss': 'to drink'},
    'code':    {'sigma': 15.46,  'real_count': 68,   'type': 'content',  'lang': 'italian', 'gloss': 'tails/codes'},
    'dicu':    {'sigma': 14.12,  'real_count': 17,   'type': 'content',  'lang': 'italian', 'gloss': 'I say (dialectal)'},
    'corali':  {'sigma': 13.47,  'real_count': 8,    'type': 'content',  'lang': 'italian', 'gloss': 'corals'},
    'diga':    {'sigma': 13.47,  'real_count': 8,    'type': 'content',  'lang': 'italian', 'gloss': 'say (subj.)'},
    'dido':    {'sigma': 11.02,  'real_count': 13,   'type': 'content',  'lang': 'italian', 'gloss': 'I gave (var.)'},
    'deri':    {'sigma': 7.12,   'real_count': 11,   'type': 'function', 'lang': 'italian', 'gloss': 'of the (pl.)'},
    'dere':    {'sigma': 6.28,   'real_count': 8,    'type': 'content',  'lang': 'italian', 'gloss': 'to give'},
    'gi':      {'sigma': 4.31,   'real_count': 6,    'type': 'function', 'lang': 'italian', 'gloss': 'already'},
    'cela':    {'sigma': 3.53,   'real_count': 5,    'type': 'content',  'lang': 'italian', 'gloss': 'hides'},
    'decore':  {'sigma': 3.25,   'real_count': 7,    'type': 'content',  'lang': 'italian', 'gloss': 'decorate'},
}

SIGNAL_WORDS_SET = set(SIGNAL_WORDS_51.keys())

# ---------------------------------------------------------------------------
# Latin ending detection for signal words
# ---------------------------------------------------------------------------

# For short words (<=2 chars), the ending IS the whole word.
# For longer words, use Latin declension/conjugation endings.

_LATIN_ENDINGS_SORTED = sorted(
    list(LATIN_NOUN_ENDINGS.keys()) + list(LATIN_VERB_ENDINGS.keys()),
    key=lambda x: -len(x),
)

def _get_word_ending(word: str) -> str:
    """Determine the Latin ending of a decoded signal word."""
    if len(word) <= 2:
        return word
    w = word.lower()
    for ending in _LATIN_ENDINGS_SORTED:
        bare = ending.lstrip('-')
        if w.endswith(bare) and len(w) > len(bare):
            return bare
    # Fallback: last character (covers -a, -e, -i, -o, -u)
    return w[-1]


# ---------------------------------------------------------------------------
# POS tag table: Latin ending -> possible POS tags
# ---------------------------------------------------------------------------

ENDING_TO_POS: Dict[str, List[str]] = {
    'a':    ['NOUN_NOM_F1', 'VERB_IMP_2S'],
    'ae':   ['NOUN_GEN_F1'],
    'am':   ['NOUN_ACC_F1'],
    'um':   ['NOUN_ACC_M2', 'ADJ_NEUT'],
    'us':   ['NOUN_NOM_M2', 'ADJ_MASC'],
    'i':    ['NOUN_GEN_M2', 'VERB_IMP'],
    'is':   ['NOUN_GEN_M3'],
    'em':   ['NOUN_ACC_M3'],
    'o':    ['NOUN_DAT_M2', 'VERB_1S'],
    'e':    ['NOUN_ABL_M3', 'VERB_IMP', 'ADV'],
    'u':    ['NOUN_ABL_4', 'PRONOUN'],
    're':   ['VERB_INF'],
    't':    ['VERB_3S'],
    'nt':   ['VERB_3P'],
    'ns':   ['VERB_PRES_PART'],
}

# Particles: uninflected function words
_PARTICLES = {
    'in', 'de', 'ad', 'cum', 'per', 'pro', 'sub', 'ex', 'ab',
    'et', 'vel', 'aut', 'sed', 'si', 'ne', 'ut', 'non', 'iam',
    'sic', 'ita', 'tunc', 'ergo', 'ibi', 'ubi', 'nunc',
    'di', 'se', 'ce', 'bi', 'ni', 'ci', 'ti', 'du', 'su', 'nu',
    'ru', 'la', 'li', 'ra', 're', 'tu', 'co', 'do', 'so', 'bo',
    'ha', 'fa', 'fe', 'ga', 'ge', 'hi', 'gi', 'be', 'cu',
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SuffixCalEntry:
    eva_suffix: str
    n_signal_tokens: int
    latin_ending_votes: Dict[str, int]
    dominant_ending: str
    confidence: float
    agreement_rate: float
    n_signal_words_using: int
    example_pairs: List[Dict]


@dataclass
class SuffixCalibrationResult:
    n_signal_words: int
    n_signal_tokens_found: int
    n_with_suffix: int
    n_without_suffix: int
    n_eva_suffixes_calibrated: int
    suffix_map: Dict[str, str]
    suffix_entries: List[Dict]
    # Phase 33 comparison
    agreement_with_phase33: Dict[str, Any]
    # POS tagging
    pos_tag_coverage: float
    pos_distribution: Dict[str, int]
    section_pos_profiles: Dict[str, Dict[str, int]]
    # Null test
    null_mean_agreement: float
    null_std_agreement: float
    agreement_z_score: float
    null_selectivity: float
    # Cross-validation
    cross_val_accuracy: float
    cross_val_folds: int
    # Paradigm
    paradigm_table: Dict[str, str]
    paradigm_coherence: float
    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return h


def _concentration(counter: Counter) -> float:
    n_classes = len(counter)
    if n_classes <= 1:
        return 1.0
    max_ent = math.log2(n_classes)
    if max_ent == 0:
        return 1.0
    return max(0.0, 1.0 - _entropy(counter) / max_ent)


def _build_suffix_ending_votes(
    signal_source_map: Dict[str, List[Dict]],
) -> Dict[str, Counter]:
    """For each EVA suffix, tally votes: which Latin ending does it encode?

    Returns {eva_suffix: Counter({latin_ending: count})}
    """
    suffix_votes: Dict[str, Counter] = defaultdict(Counter)

    for decoded_word, records in signal_source_map.items():
        word_ending = _get_word_ending(decoded_word)
        for rec in records:
            suffix = rec['suffix']
            if suffix:
                suffix_votes[suffix][word_ending] += 1

    return dict(suffix_votes)


def _compute_real_agreement(suffix_votes: Dict[str, Counter]) -> float:
    """Mean agreement rate across all suffixes (dominant / total)."""
    if not suffix_votes:
        return 0.0
    rates = []
    for counter in suffix_votes.values():
        total = sum(counter.values())
        if total > 0:
            top = counter.most_common(1)[0][1]
            rates.append(top / total)
    return sum(rates) / len(rates) if rates else 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_suffix_calibration() -> None:
    """Phase 51 Track A: Reverse Suffix Calibration."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 51 TRACK A: Reverse Suffix Calibration")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ────────────────────────────────────────────────
    print("\n  A.1  Loading inputs...")

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        bigram_data = json.load(f)
    token_evas = bigram_data['token_evas']
    token_decoded = bigram_data['token_decoded']
    token_classifications = bigram_data['token_classifications']
    token_folios = bigram_data['token_folios']
    n_tokens = len(token_evas)

    print(f"       {n_tokens} tokens loaded")
    print(f"       {len(SIGNAL_WORDS_51)} hardcoded signal words")

    # ── 2. Build signal-word source map ───────────────────────────────
    print("\n  A.2  Building signal-word source map...")

    # For each signal word, collect all EVA tokens that decode to it
    signal_source_map: Dict[str, List[Dict]] = defaultdict(list)
    n_signal_tokens = 0

    for i in range(n_tokens):
        decoded = token_decoded[i]
        if decoded in SIGNAL_WORDS_SET:
            n_signal_tokens += 1
            decomp = decompose_token_morphemes(token_evas[i])
            signal_source_map[decoded].append({
                'eva': token_evas[i],
                'prefix': decomp.prefix,
                'stem': decomp.stem,
                'suffix': decomp.suffix,
                'folio': token_folios[i],
                'classification': token_classifications[i],
            })

    n_words_found = len(signal_source_map)
    n_with_suffix = sum(
        1 for recs in signal_source_map.values()
        for r in recs if r['suffix']
    )
    n_without_suffix = n_signal_tokens - n_with_suffix

    print(f"       {n_signal_tokens} signal tokens found across {n_words_found} words")
    print(f"       {n_with_suffix} with EVA suffix, {n_without_suffix} without")

    # Show top signal words
    for word in sorted(signal_source_map.keys(),
                       key=lambda w: -len(signal_source_map[w]))[:10]:
        recs = signal_source_map[word]
        suffixes = Counter(r['suffix'] for r in recs if r['suffix'])
        sfx_str = ', '.join(f"{s}={c}" for s, c in suffixes.most_common(3))
        print(f"       {word:10s}: {len(recs):5d} tokens  suffixes: {sfx_str or '(none)'}")

    # ── 3. Build calibrated suffix → ending map ──────────────────────
    print("\n  A.3  Building calibrated suffix → ending map...")

    suffix_votes = _build_suffix_ending_votes(signal_source_map)

    suffix_map: Dict[str, str] = {}
    suffix_entries: List[SuffixCalEntry] = []

    for sfx in sorted(suffix_votes.keys(),
                       key=lambda s: -sum(suffix_votes[s].values())):
        counter = suffix_votes[sfx]
        total = sum(counter.values())
        top_ending, top_count = counter.most_common(1)[0]
        confidence = _concentration(counter)
        agreement_rate = top_count / total if total > 0 else 0.0

        # Which signal words use this suffix?
        words_using = set()
        examples = []
        for word, recs in signal_source_map.items():
            for r in recs:
                if r['suffix'] == sfx:
                    words_using.add(word)
                    if len(examples) < 5:
                        examples.append({
                            'signal_word': word,
                            'eva_token': r['eva'],
                            'folio': r['folio'],
                        })

        entry = SuffixCalEntry(
            eva_suffix=sfx,
            n_signal_tokens=total,
            latin_ending_votes=dict(counter.most_common()),
            dominant_ending=top_ending,
            confidence=round(confidence, 4),
            agreement_rate=round(agreement_rate, 4),
            n_signal_words_using=len(words_using),
            example_pairs=examples,
        )
        suffix_entries.append(entry)
        suffix_map[sfx] = top_ending

        vote_str = ', '.join(f"{e}={c}" for e, c in counter.most_common(3))
        print(f"       {sfx:6s} → {top_ending:4s}  "
              f"agree={agreement_rate:.1%}  conf={confidence:.3f}  "
              f"n={total}  [{vote_str}]")

    # ── 4. Compare to Phase 33 mappings ──────────────────────────────
    print("\n  A.4  Comparing to Phase 33 suffix mappings...")

    phase33_map = {
        'dy': 'a', 'y': 'i', 'ey': 'e', 'aiin': 'um',
        'ol': 'us', 'al': 'is', 'in': 'em', 'am': 'am',
    }
    n_agreed = 0
    n_disagreed = 0
    comparison_details = []
    for sfx, p33_ending in phase33_map.items():
        p51_ending = suffix_map.get(sfx, '?')
        agrees = (p51_ending == p33_ending)
        if agrees:
            n_agreed += 1
        else:
            n_disagreed += 1
        comparison_details.append({
            'suffix': sfx,
            'phase33': p33_ending,
            'phase51': p51_ending,
            'agrees': agrees,
        })
        status = "AGREE" if agrees else "DISAGREE"
        print(f"       {sfx:6s}: P33={p33_ending:4s}  P51={p51_ending:4s}  {status}")

    agreement_p33 = {
        'agreed': n_agreed,
        'disagreed': n_disagreed,
        'rate': round(n_agreed / max(n_agreed + n_disagreed, 1), 4),
        'details': comparison_details,
    }

    # ── 5. Apply suffix map to full corpus → POS tags ────────────────
    print("\n  A.5  POS-tagging full corpus...")

    corpus = load_corpus()
    # Build folio→section lookup
    folio_section: Dict[str, str] = {}
    for folio_id, page in corpus.pages.items():
        folio_section[folio_id] = page.section

    pos_distribution: Counter = Counter()
    section_pos: Dict[str, Counter] = defaultdict(Counter)
    n_tagged = 0

    for i in range(n_tokens):
        decomp = decompose_token_morphemes(token_evas[i])
        suffix = decomp.suffix

        if not suffix or suffix not in suffix_map:
            pos_distribution['UNTAGGED'] += 1
            section = folio_section.get(token_folios[i], 'unknown')
            section_pos[section]['UNTAGGED'] += 1
            continue

        ending = suffix_map[suffix]
        decoded = token_decoded[i]

        # Check if it's a particle
        if decoded in _PARTICLES:
            pos = 'PARTICLE'
        elif ending in ENDING_TO_POS:
            # Use first (most common) POS for the distribution
            pos = ENDING_TO_POS[ending][0]
        else:
            pos = f'ENDING_{ending}'

        pos_distribution[pos] += 1
        n_tagged += 1
        section = folio_section.get(token_folios[i], 'unknown')
        section_pos[section][pos] += 1

    pos_coverage = n_tagged / n_tokens if n_tokens > 0 else 0.0

    print(f"       POS coverage: {n_tagged}/{n_tokens} = {pos_coverage:.1%}")
    print("       POS distribution:")
    for pos, count in pos_distribution.most_common(15):
        print(f"         {pos:20s}: {count:6d} ({count/n_tokens:.1%})")

    print("\n       Per-section profiles:")
    for section in sorted(section_pos.keys()):
        sec_total = sum(section_pos[section].values())
        top3 = section_pos[section].most_common(3)
        top_str = ', '.join(f"{p}={c}" for p, c in top3)
        print(f"         {section:20s}: n={sec_total:5d}  [{top_str}]")

    # ── 6. Null test ─────────────────────────────────────────────────
    print("\n  A.6  Null test (shuffled endings)...")

    real_agreement = _compute_real_agreement(suffix_votes)
    rng = random.Random(42)
    n_null = 100
    null_agreements = []

    # Collect all (word, ending) pairs for shuffling
    word_endings = {w: _get_word_ending(w) for w in SIGNAL_WORDS_SET}
    all_endings = list(word_endings.values())

    for _ in range(n_null):
        # Shuffle word→ending assignments
        shuffled_endings = list(all_endings)
        rng.shuffle(shuffled_endings)
        shuffled_map = dict(zip(word_endings.keys(), shuffled_endings))

        # Rebuild suffix votes with shuffled endings
        null_votes: Dict[str, Counter] = defaultdict(Counter)
        for decoded_word, records in signal_source_map.items():
            word_ending = shuffled_map.get(decoded_word, decoded_word[-1])
            for rec in records:
                if rec['suffix']:
                    null_votes[rec['suffix']][word_ending] += 1

        null_agreement = _compute_real_agreement(dict(null_votes))
        null_agreements.append(null_agreement)

    null_mean = sum(null_agreements) / len(null_agreements)
    null_std = (sum((x - null_mean) ** 2 for x in null_agreements)
                / len(null_agreements)) ** 0.5
    z_score = ((real_agreement - null_mean) / null_std
               if null_std > 0 else 0.0)
    null_selectivity = (real_agreement / null_mean
                        if null_mean > 0 else float('inf'))

    print(f"       Real agreement:  {real_agreement:.4f}")
    print(f"       Null mean:       {null_mean:.4f} ± {null_std:.4f}")
    print(f"       Z-score:         {z_score:.2f}")
    print(f"       Selectivity:     {null_selectivity:.2f}×")

    # ── 7. Cross-validation ──────────────────────────────────────────
    print("\n  A.7  5-fold cross-validation...")

    signal_words_list = sorted(SIGNAL_WORDS_SET)
    rng2 = random.Random(123)
    rng2.shuffle(signal_words_list)
    n_folds = 5
    fold_size = len(signal_words_list) // n_folds
    fold_accuracies = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(signal_words_list)
        test_words = set(signal_words_list[start:end])
        train_words = SIGNAL_WORDS_SET - test_words

        # Build suffix map from training words only
        train_votes: Dict[str, Counter] = defaultdict(Counter)
        for word, records in signal_source_map.items():
            if word not in train_words:
                continue
            word_ending = _get_word_ending(word)
            for rec in records:
                if rec['suffix']:
                    train_votes[rec['suffix']][word_ending] += 1

        train_suffix_map = {}
        for sfx, counter in train_votes.items():
            if counter:
                train_suffix_map[sfx] = counter.most_common(1)[0][0]

        # Test: do held-out signal words' suffixes predict the correct ending?
        n_correct = 0
        n_test = 0
        for word in test_words:
            if word not in signal_source_map:
                continue
            word_ending = _get_word_ending(word)
            for rec in signal_source_map[word]:
                if rec['suffix'] and rec['suffix'] in train_suffix_map:
                    n_test += 1
                    if train_suffix_map[rec['suffix']] == word_ending:
                        n_correct += 1

        fold_acc = n_correct / n_test if n_test > 0 else 0.0
        fold_accuracies.append(fold_acc)
        print(f"       Fold {fold+1}: {n_correct}/{n_test} = {fold_acc:.1%}")

    cv_accuracy = (sum(fold_accuracies) / len(fold_accuracies)
                   if fold_accuracies else 0.0)
    print(f"       Mean CV accuracy: {cv_accuracy:.1%}")

    # ── 8. Paradigm analysis ─────────────────────────────────────────
    print("\n  A.8  Paradigm analysis...")

    # Build paradigm table from suffix map
    paradigm_table: Dict[str, str] = {}
    for sfx, ending in suffix_map.items():
        bare = ending.lstrip('-')
        # Check noun endings
        if f'-{bare}' in LATIN_NOUN_ENDINGS:
            paradigm_table[sfx] = LATIN_NOUN_ENDINGS[f'-{bare}']
        elif f'-{bare}' in LATIN_VERB_ENDINGS:
            paradigm_table[sfx] = LATIN_VERB_ENDINGS[f'-{bare}']
        else:
            paradigm_table[sfx] = f'ending_{bare}'

    # Simple coherence: count how many suffix→ending mappings produce
    # distinct grammatical categories
    unique_forms = set(paradigm_table.values())
    n_suffixes_mapped = len(paradigm_table)
    paradigm_coherence = (len(unique_forms) / n_suffixes_mapped
                          if n_suffixes_mapped > 0 else 0.0)

    print(f"       {n_suffixes_mapped} suffixes mapped to "
          f"{len(unique_forms)} distinct forms")
    print(f"       Paradigm coherence: {paradigm_coherence:.4f}")
    for sfx, form in sorted(paradigm_table.items()):
        print(f"         {sfx:6s} → {form}")

    # ── 9. Verdict ───────────────────────────────────────────────────

    n_calibrated = len(suffix_map)
    n_total_suffixes = len(KNOWN_SUFFIXES)
    coverage = n_calibrated / n_total_suffixes if n_total_suffixes > 0 else 0.0
    mean_confidence = (sum(e.confidence for e in suffix_entries) / len(suffix_entries)
                       if suffix_entries else 0.0)

    if mean_confidence > 0.6 and coverage > 0.5 and null_selectivity > 1.5:
        verdict = 'SUFFIX_MAP_VALID'
    elif mean_confidence > 0.4 or coverage > 0.3:
        verdict = 'SUFFIX_MAP_PARTIAL'
    else:
        verdict = 'SUFFIX_MAP_INVALID'

    print(f"\n  Verdict: {verdict}")
    print(f"       Mean confidence: {mean_confidence:.4f}")
    print(f"       Coverage: {n_calibrated}/{n_total_suffixes} = {coverage:.1%}")
    print(f"       Null selectivity: {null_selectivity:.2f}×")
    print(f"       CV accuracy: {cv_accuracy:.1%}")

    # ── 10. Save ─────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = SuffixCalibrationResult(
        n_signal_words=len(SIGNAL_WORDS_51),
        n_signal_tokens_found=n_signal_tokens,
        n_with_suffix=n_with_suffix,
        n_without_suffix=n_without_suffix,
        n_eva_suffixes_calibrated=n_calibrated,
        suffix_map=suffix_map,
        suffix_entries=[_convert(asdict(e)) for e in suffix_entries],
        agreement_with_phase33=agreement_p33,
        pos_tag_coverage=round(pos_coverage, 4),
        pos_distribution=dict(pos_distribution.most_common()),
        section_pos_profiles={
            sec: dict(counter.most_common())
            for sec, counter in section_pos.items()
        },
        null_mean_agreement=round(null_mean, 4),
        null_std_agreement=round(null_std, 4),
        agreement_z_score=round(z_score, 2),
        null_selectivity=round(null_selectivity, 4),
        cross_val_accuracy=round(cv_accuracy, 4),
        cross_val_folds=n_folds,
        paradigm_table=paradigm_table,
        paradigm_coherence=round(paradigm_coherence, 4),
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'suffix_calibration.json', asdict(result))
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
