"""
Phase 48 Track C: f1r Multispectral Annotations (Marci Decipherment Attempt)
==============================================================================
Analyze the three columns of hidden letters revealed by the September 2024
Lazarus Project multispectral imaging on f1r.  These have been identified as
a decipherment attempt by Johannes Marcus Marci.

Dependency chain:
    combined_refine.json       (Phase 15 — T_P15 assignment)
    modifier_integrate.json    (Phase 16 — modifier chars)
        → marci_source.json          (48C.1)
        → marci_correspondences.json (48C.2)
        → marci_comparison.json      (48C.3)
        → marci_test.json            (48C.4)
"""

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir


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


def _load_json(rd: str, filename: str) -> Optional[Dict]:
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarciSource:
    """Step 48C.1 output."""
    data_available: bool
    source_description: str
    transcription_sources: List[Dict]
    n_characters_visible: int
    column_descriptions: List[str]
    scholarly_references: List[str]
    notes: List[str]
    runtime_seconds: float


@dataclass
class MarciCorrespondence:
    """One EVA→Roman letter correspondence from Marci."""
    eva_char: str
    roman_letter: str
    triple_key: str
    confidence: str           # HIGH / MEDIUM / LOW / UNCERTAIN
    source_note: str


@dataclass
class MarciCorrespondences:
    """Step 48C.2 output."""
    correspondences: List[Dict]
    n_correspondences: int
    is_alphabetic: bool       # character-level substitution?
    is_syllabic: bool         # syllable-level?
    data_quality: str         # SUFFICIENT / INSUFFICIENT / UNAVAILABLE
    notes: List[str]
    runtime_seconds: float


@dataclass
class MarciComparison:
    """Step 48C.3 output."""
    per_character: List[Dict]
    consonant_ari: float
    syllable_ari: float
    n_consonant_match: int
    n_vowel_match: int
    n_no_match: int
    interpretation: str
    runtime_seconds: float


@dataclass
class MarciTest:
    """Step 48C.4 output."""
    n_correspondences_used: int
    marci_table_dict_hit: float
    t_p15_dict_hit_f1r: float
    random_baseline: float
    selectivity: float
    performance_class: str     # EXCEEDS_T_P15 / PARTIAL / RANDOM
    notes: List[str]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 48C.1 — Source Data
# ---------------------------------------------------------------------------

def run_marci_source() -> None:
    """Step 48C.1: Obtain and transcribe multispectral data."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48C.1: Marci Multispectral Source Data")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Document what is known ──
    print("\n  1. Documenting known information about f1r multispectral annotations...")

    # The Lazarus Project multispectral imaging (Sept 2024) revealed hidden
    # columns on f1r.  Lisa Fagin Davis identified them as a decipherment
    # attempt by Johannes Marcus Marci (1640s-1660s).
    #
    # The actual character-by-character transcription has NOT been published
    # in machine-readable form.  What we know comes from:
    # - Davis's social media posts (Twitter/X)
    # - News coverage of the imaging project
    # - The images themselves (Google Drive, visual inspection required)

    transcription_sources = [
        {
            'source': 'Lisa Fagin Davis (2024)',
            'description': 'Identified three columns in right margin of f1r: '
                           'two in Roman alphabet, one in Voynichese. Attributed '
                           'to Johannes Marcus Marci as a decipherment attempt.',
            'format': 'Social media posts and academic presentations',
            'machine_readable': False,
            'url_note': 'Google Drive images available (visual inspection required)',
        },
        {
            'source': 'Lazarus Project (Rochester Institute of Technology)',
            'description': 'Multispectral imaging of Beinecke MS 408. '
                           'Multiple spectral bands reveal text invisible under '
                           'normal lighting.',
            'format': 'TIFF/raw multispectral images',
            'machine_readable': False,
            'url_note': 'Publicly released on Google Drive',
        },
    ]

    column_descriptions = [
        'Column 1 (Roman alphabet): Letters visible in right margin of f1r',
        'Column 2 (Roman alphabet): Second column of Roman letters',
        'Column 3 (Voynichese): Column of Voynich characters',
        'Arrangement suggests a correspondence table: Voynichese ↔ Roman letters',
    ]

    scholarly_refs = [
        'Davis, Lisa Fagin (2024). Social media posts on Lazarus Project f1r imaging.',
        'Lazarus Project, Rochester Institute of Technology (2024). '
        'Multispectral imaging of Beinecke MS 408.',
        'Marci, Johannes Marcus (1665/1666). Letter to Athanasius Kircher '
        '(accompanying the manuscript).',
    ]

    notes = [
        'NO machine-readable transcription of the Marci annotations has been published',
        'Character-by-character extraction requires visual inspection of multispectral images',
        'The images are publicly available but require specialized image processing',
        'Davis identified ~20-30 characters visible across the three columns',
        'Attribution to Marci is based on handwriting comparison with his known letters',
        'Marci owned the manuscript from ~1640s until sending it to Kircher in 1665/1666',
        'This is the only known historical decipherment attempt',
    ]

    data_available = False  # No machine-readable transcription exists

    print(f"     Data available in machine-readable form: {data_available}")
    print(f"     Transcription sources: {len(transcription_sources)}")
    for src in transcription_sources:
        print(f"       • {src['source']}: {src['description'][:80]}...")
    print(f"\n     Column arrangement:")
    for col in column_descriptions:
        print(f"       {col}")
    print(f"\n     Notes:")
    for note in notes:
        print(f"       • {note}")

    # ── 2. Save ──
    result = MarciSource(
        data_available=data_available,
        source_description='Multispectral imaging by Lazarus Project (Sept 2024) '
                           'revealed hidden annotations on f1r attributed to '
                           'Johannes Marcus Marci. No machine-readable transcription '
                           'has been published.',
        transcription_sources=transcription_sources,
        n_characters_visible=0,  # Unknown without visual inspection
        column_descriptions=column_descriptions,
        scholarly_references=scholarly_refs,
        notes=notes,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'marci_source.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48C.2 — Extract Correspondences
# ---------------------------------------------------------------------------

def run_marci_extract() -> None:
    """Step 48C.2: Extract Voynichese-to-Roman letter correspondences."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48C.2: Marci Correspondence Extraction")
    print("=" * 70)

    rd = _results_dir()

    source_data = _load_json(rd, 'marci_source.json')
    if not source_data:
        print("     ERROR: marci_source.json not found. Run marci-source first.")
        return

    # ── 1. Check data availability ──
    print("\n  1. Checking data availability...")

    data_available = source_data.get('data_available', False)

    if not data_available:
        print("     No machine-readable Marci transcription available.")
        print("     Cannot extract correspondences without visual inspection of images.")
        print()
        print("     BLOCKED: This step requires either:")
        print("       (a) A published transcription of the Marci columns, or")
        print("       (b) Visual inspection and manual transcription of the")
        print("           multispectral images from the Lazarus Project.")
        print()
        print("     Proceeding with UNAVAILABLE status.")

        result = MarciCorrespondences(
            correspondences=[],
            n_correspondences=0,
            is_alphabetic=False,
            is_syllabic=False,
            data_quality='UNAVAILABLE',
            notes=[
                'No machine-readable Marci transcription exists',
                'Extraction blocked until published transcription or manual image analysis',
                'The multispectral images are publicly available on Google Drive',
                'Future work: visual inspection of images to extract character pairs',
            ],
            runtime_seconds=round(time.time() - t0, 2),
        )

        out = _save_json(rd, 'marci_correspondences.json', asdict(result))
        print(f"\n  Saved → {out}")
        print(f"  Completed in {time.time() - t0:.1f}s")
        return

    # If data were available, we would extract correspondences here
    # This code path is for future use when a transcription becomes available


# ---------------------------------------------------------------------------
# Step 48C.3 — Compare to T_P15
# ---------------------------------------------------------------------------

def run_marci_compare() -> None:
    """Step 48C.3: Compare Marci's guesses to T_P15."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48C.3: Marci vs T_P15 Comparison")
    print("=" * 70)

    rd = _results_dir()

    corr_data = _load_json(rd, 'marci_correspondences.json')
    if not corr_data:
        print("     ERROR: marci_correspondences.json not found.")
        return

    # ── 1. Check data quality ──
    data_quality = corr_data.get('data_quality', 'UNAVAILABLE')

    if data_quality == 'UNAVAILABLE':
        print("\n  Marci correspondences UNAVAILABLE — no comparison possible.")
        print("  Recording null result.")

        result = MarciComparison(
            per_character=[],
            consonant_ari=0.0,
            syllable_ari=0.0,
            n_consonant_match=0,
            n_vowel_match=0,
            n_no_match=0,
            interpretation='DATA_UNAVAILABLE: Cannot compare without Marci transcription. '
                           'The multispectral images exist but no machine-readable '
                           'extraction has been performed.',
            runtime_seconds=round(time.time() - t0, 2),
        )

        out = _save_json(rd, 'marci_comparison.json', asdict(result))
        print(f"\n  Saved → {out}")
        return

    # If correspondences were available, comparison logic would go here
    combined = _load_json(rd, 'combined_refine.json')
    if not combined:
        print("     ERROR: combined_refine.json not found.")
        return

    from voynich.core.corpus import build_eva_to_triple_lookup

    assignment = combined.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()
    correspondences = corr_data.get('correspondences', [])

    per_char = []
    n_consonant = 0
    n_vowel = 0
    n_no = 0

    for corr in correspondences:
        eva_char = corr.get('eva_char', '')
        roman = corr.get('roman_letter', '')
        tk = eva_to_triple.get(eva_char, '')
        t_p15_syl = assignment.get(tk, '')

        consonant_match = len(t_p15_syl) >= 1 and t_p15_syl[0] == roman
        vowel_match = len(t_p15_syl) >= 2 and t_p15_syl[1] == roman

        if consonant_match:
            n_consonant += 1
            match_type = 'consonant'
        elif vowel_match:
            n_vowel += 1
            match_type = 'vowel'
        else:
            n_no += 1
            match_type = 'none'

        per_char.append({
            'eva_char': eva_char,
            'marci_value': roman,
            'triple_key': tk,
            't_p15_syllable': t_p15_syl,
            'match_type': match_type,
        })

    # Compute ARI (Adjusted Rand Index) approximation
    n_total = len(correspondences)
    consonant_ari = n_consonant / n_total if n_total > 0 else 0.0
    syllable_ari = (n_consonant + n_vowel) / n_total if n_total > 0 else 0.0

    if consonant_ari > 0.5:
        interp = "STRONG: Marci's guesses consistently match T_P15 consonants"
    elif consonant_ari > 0.3:
        interp = "MODERATE: Marci detected consonant dimension but not vowels"
    elif syllable_ari > 0.3:
        interp = "PARTIAL: Some agreement at vowel level"
    else:
        interp = "NONE: Marci's guesses appear random with respect to T_P15"

    result = MarciComparison(
        per_character=per_char,
        consonant_ari=round(consonant_ari, 4),
        syllable_ari=round(syllable_ari, 4),
        n_consonant_match=n_consonant,
        n_vowel_match=n_vowel,
        n_no_match=n_no,
        interpretation=interp,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'marci_comparison.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48C.4 — Test Marci's Assignments on Corpus
# ---------------------------------------------------------------------------

def run_marci_test() -> None:
    """Step 48C.4: Test Marci's assignments on the corpus."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48C.4: Marci Assignment Corpus Test")
    print("=" * 70)

    rd = _results_dir()

    corr_data = _load_json(rd, 'marci_correspondences.json')
    comp_data = _load_json(rd, 'marci_comparison.json')

    if not corr_data:
        print("     ERROR: marci_correspondences.json not found.")
        return

    data_quality = corr_data.get('data_quality', 'UNAVAILABLE')

    if data_quality == 'UNAVAILABLE':
        print("\n  Marci correspondences UNAVAILABLE — cannot test on corpus.")
        print("  Recording null result.")

        result = MarciTest(
            n_correspondences_used=0,
            marci_table_dict_hit=0.0,
            t_p15_dict_hit_f1r=0.0,
            random_baseline=0.0,
            selectivity=0.0,
            performance_class='DATA_UNAVAILABLE',
            notes=[
                'Cannot test Marci assignments — no transcription available',
                'This step will become executable when a transcription is published',
            ],
            runtime_seconds=round(time.time() - t0, 2),
        )

        out = _save_json(rd, 'marci_test.json', asdict(result))
        print(f"\n  Saved → {out}")
        return

    # If data were available, we would:
    # 1. Build a Marci table (EVA char → Marci value)
    # 2. Decode f1r main text using Marci's table
    # 3. Compute dict-hit against Latin and Italian dictionaries
    # 4. Compare to T_P15 performance on f1r
    print("  Data available — running corpus test...")

    from voynich.core.corpus import (
        build_eva_to_triple_lookup,
        decode_token_modifier_aware,
        load_corpus,
        tokenize_eva_chars,
    )
    from voynich.core.reference import build_expanded_word_set, load_reference_corpus

    corpus = load_corpus(verbose=False)
    combined = _load_json(rd, 'combined_refine.json')
    mod_data = _load_json(rd, 'modifier_integrate.json')

    assignment = combined.get('best_assignment', {})
    modifier_chars = set(mod_data.get('modifier_chars', [])) if mod_data else set()
    eva_to_triple = build_eva_to_triple_lookup()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    dict_131k = base_words | expanded

    # Build Marci table: EVA char → single letter (→ substitute for triple's syllable)
    correspondences = corr_data.get('correspondences', [])
    marci_assignment = dict(assignment)  # Start from T_P15

    for corr in correspondences:
        eva_char = corr.get('eva_char', '')
        roman = corr.get('roman_letter', '')
        tk = eva_to_triple.get(eva_char, '')
        if tk and roman:
            # Replace the syllable assignment with Marci's letter
            marci_assignment[tk] = roman

    # Decode f1r
    f1r = corpus.pages.get('f1r')
    if not f1r:
        print("     ERROR: f1r not found in corpus")
        return

    f1r_tokens = f1r.all_tokens
    n_tokens = len(f1r_tokens)

    # T_P15 decode
    t_p15_hits = 0
    for tok in f1r_tokens:
        decoded = decode_token_modifier_aware(tok, assignment, eva_to_triple, modifier_chars)
        if decoded.lower() in dict_131k:
            t_p15_hits += 1
    t_p15_hit = t_p15_hits / n_tokens if n_tokens > 0 else 0.0

    # Marci decode
    marci_hits = 0
    for tok in f1r_tokens:
        decoded = decode_token_modifier_aware(tok, marci_assignment, eva_to_triple, modifier_chars)
        if decoded.lower() in dict_131k:
            marci_hits += 1
    marci_hit = marci_hits / n_tokens if n_tokens > 0 else 0.0

    # Random baseline (shuffle assignment values)
    rng = random.Random(42)
    values = list(assignment.values())
    random_hits_total = 0
    n_random = 5
    for _ in range(n_random):
        rng.shuffle(values)
        rand_assign = dict(zip(assignment.keys(), values))
        hits = 0
        for tok in f1r_tokens:
            decoded = decode_token_modifier_aware(tok, rand_assign, eva_to_triple, modifier_chars)
            if decoded.lower() in dict_131k:
                hits += 1
        random_hits_total += hits
    random_baseline = random_hits_total / (n_random * n_tokens) if n_tokens > 0 else 0.0

    selectivity = marci_hit / random_baseline if random_baseline > 0 else 0.0

    if marci_hit >= t_p15_hit:
        perf_class = 'EXCEEDS_T_P15'
    elif marci_hit > random_baseline * 1.5:
        perf_class = 'PARTIAL'
    else:
        perf_class = 'RANDOM'

    print(f"     Marci dict-hit: {marci_hit:.4f}")
    print(f"     T_P15 dict-hit: {t_p15_hit:.4f}")
    print(f"     Random baseline: {random_baseline:.4f}")
    print(f"     Performance: {perf_class}")

    result = MarciTest(
        n_correspondences_used=len(correspondences),
        marci_table_dict_hit=round(marci_hit, 4),
        t_p15_dict_hit_f1r=round(t_p15_hit, 4),
        random_baseline=round(random_baseline, 4),
        selectivity=round(selectivity, 4),
        performance_class=perf_class,
        notes=[
            f'Tested on f1r ({n_tokens} tokens)',
            f'{len(correspondences)} Marci correspondences used',
        ],
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'marci_test.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Track C runner
# ---------------------------------------------------------------------------

def run_track_c_48() -> None:
    """Run all Track C steps sequentially."""
    print("\n" + "█" * 70)
    print("  PHASE 48 TRACK C: f1r Marci Annotations")
    print("█" * 70)

    run_marci_source()
    run_marci_extract()
    run_marci_compare()
    run_marci_test()

    print("\n" + "█" * 70)
    print("  TRACK C COMPLETE")
    print("█" * 70)
