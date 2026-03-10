"""
Step 34.1 – Medieval Abbreviation Dictionary (Track A)
=======================================================
Builds two specialised dictionaries for the abjad consonant-only hypothesis:
1. Consonant-skeleton dictionary (Latin words stripped of vowels)
2. Medical sigla dictionary (from Cappelli)
Then merges them into a combined target dict for consonant-sequence matching.

Dependency chain:
    data/reference/latin/                  (reference corpus)
    data/2Translate/Cappelli.../extracted.json  (2,678 abbreviation entries)
        → sigla_dictionary.json  (this step)
"""

import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    load_reference_corpus,
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


VOWELS = set('aeiouy')


def _strip_vowels(word: str) -> str:
    """Strip all vowels from a word, leaving only consonants."""
    return ''.join(c for c in word.lower() if c not in VOWELS and c.isalpha())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SiglaDictionaryResult:
    # Consonant skeleton dictionary
    n_base_words: int
    n_unique_skeletons: int
    skeleton_length_distribution: Dict[str, int]  # length → count
    most_ambiguous_skeletons: List[Dict]  # skeleton → list of words

    # Cappelli sigla dictionary
    cappelli_path: str
    n_cappelli_entries: int
    n_cappelli_latin: int
    n_cappelli_pharmaceutical: int
    n_cappelli_medical: int
    cappelli_domain_breakdown: Dict[str, int]
    sample_sigla: List[Dict]

    # Combined dictionary
    n_combined_entries: int
    overlap_count: int  # skeletons that appear in both dicts

    # Null hit rates by consonant sequence length
    null_hit_rates: Dict[str, float]  # length → probability of random hit
    random_baseline_hit_rate: float   # overall null for random 2-4 char sequences

    runtime_seconds: float


# ---------------------------------------------------------------------------
# Cappelli loading
# ---------------------------------------------------------------------------

def _find_cappelli_path() -> str:
    """Locate the Cappelli extracted JSON file."""
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))),
            'data', '2Translate',
            'Cappelli_Lexicon Abbreviaturarum_DONE',
            'Cappelli_Lexicon Abbreviaturarum_extracted.json',
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Cannot find Cappelli_Lexicon Abbreviaturarum_extracted.json"
    )


def _load_cappelli_sigla(cappelli_path: str) -> List[Dict[str, str]]:
    """Load Cappelli abbreviation entries.

    Returns list of dicts with keys:
        abbreviated_form, expansion, skeleton, domain
    """
    with open(cappelli_path) as f:
        data = json.load(f)

    entries = data.get('entries', [])
    sigla: List[Dict[str, str]] = []

    for entry in entries:
        abbr = entry.get('abbreviated_form', '')
        expansion = entry.get('latin_expansion', '')
        domain = entry.get('semantic_domain', 'general')
        lang = entry.get('language', 'latin')

        if not abbr or not expansion:
            continue

        # Clean the abbreviated form — remove periods, parentheses
        clean_abbr = re.sub(r'[^a-zA-Z]', '', abbr).lower()
        if not clean_abbr:
            continue

        # Get consonant skeleton of the abbreviation
        skeleton = _strip_vowels(clean_abbr)

        sigla.append({
            'abbreviated_form': abbr,
            'clean_form': clean_abbr,
            'expansion': expansion,
            'skeleton': skeleton,
            'domain': domain,
            'language': lang,
        })

    return sigla


# ---------------------------------------------------------------------------
# Dictionary building
# ---------------------------------------------------------------------------

def _build_consonant_skeletons(
    latin_dict: Set[str],
) -> Dict[str, List[str]]:
    """Build consonant-skeleton dictionary: skeleton → list of source words."""
    skeletons: Dict[str, List[str]] = defaultdict(list)
    for word in sorted(latin_dict):
        skel = _strip_vowels(word)
        if len(skel) >= 1:
            skeletons[skel].append(word)
    return dict(skeletons)


def _merge_dictionaries(
    skeletons: Dict[str, List[str]],
    sigla: List[Dict[str, str]],
) -> Dict[str, List[str]]:
    """Merge consonant skeletons + Cappelli sigla into combined dict."""
    combined: Dict[str, List[str]] = defaultdict(list)

    # Add all consonant skeletons
    for skel, words in skeletons.items():
        combined[skel].extend(words)

    # Add Cappelli entries by their skeleton
    for entry in sigla:
        skel = entry['skeleton']
        if skel and len(skel) >= 1:
            expansion = entry['expansion']
            if expansion not in combined.get(skel, []):
                combined[skel].append(f"[sigla]{expansion}")
        # Also add by their clean abbreviated form
        clean = entry['clean_form']
        if clean and clean not in combined.get(clean, []):
            combined[clean].append(f"[abbr]{entry['expansion']}")

    return dict(combined)


def _null_hit_rate(
    combined_dict: Dict[str, List[str]],
    n_random: int = 10000,
    seed: int = 42,
) -> Dict[int, float]:
    """Compute null hit rate: probability that a random consonant sequence
    of length N matches something in the combined dictionary.

    Returns dict of length → probability.
    """
    consonants = list('bcdfghlmnpqrstvxz')
    rng = random.Random(seed)
    results: Dict[int, float] = {}

    for length in range(1, 7):
        hits = 0
        for _ in range(n_random):
            seq = ''.join(rng.choice(consonants) for _ in range(length))
            if seq in combined_dict:
                hits += 1
        results[length] = hits / n_random

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sigla_dictionary() -> None:
    """Step 34.1: Build medieval abbreviation dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.1: Medieval Abbreviation Dictionary (Track A)")
    print("=" * 70)

    # ── 1. Build consonant skeleton dictionary from 17K base words ──
    print("\n  1. Building consonant skeleton dictionary …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    print(f"     Base dictionary: {len(base_words)} words")

    skeletons = _build_consonant_skeletons(base_words)
    print(f"     Unique consonant skeletons: {len(skeletons)}")

    # Length distribution
    length_dist: Dict[int, int] = Counter()
    for skel in skeletons:
        length_dist[len(skel)] += 1

    # Most ambiguous skeletons (most words mapping to same skeleton)
    ambiguous = sorted(skeletons.items(), key=lambda x: len(x[1]), reverse=True)
    top_ambiguous = [
        {'skeleton': skel, 'n_words': len(words), 'words': words[:10]}
        for skel, words in ambiguous[:20]
    ]
    print(f"     Most ambiguous: '{ambiguous[0][0]}' → {len(ambiguous[0][1])} words")

    # ── 2. Load Cappelli sigla ──
    print("\n  2. Loading Cappelli sigla …")
    cappelli_path = _find_cappelli_path()
    sigla = _load_cappelli_sigla(cappelli_path)
    print(f"     Loaded {len(sigla)} Cappelli entries from {cappelli_path}")

    n_latin = sum(1 for s in sigla if s['language'] == 'latin')
    n_pharma = sum(1 for s in sigla if s['domain'] == 'pharmaceutical')
    n_medical = sum(1 for s in sigla if s['domain'] == 'medical')
    domain_breakdown = Counter(s['domain'] for s in sigla)

    print(f"     Latin: {n_latin}, pharmaceutical: {n_pharma}, medical: {n_medical}")

    # ── 3. Merge dictionaries ──
    print("\n  3. Merging dictionaries …")
    combined = _merge_dictionaries(skeletons, sigla)
    print(f"     Combined dictionary: {len(combined)} unique consonant sequences")

    # Count overlap
    sigla_keys = set()
    for s in sigla:
        sigla_keys.add(s['skeleton'])
        sigla_keys.add(s['clean_form'])
    overlap = len(sigla_keys & set(skeletons.keys()))
    print(f"     Overlap (in both dicts): {overlap}")

    # ── 4. Null hit rates ──
    print("\n  4. Computing null hit rates …")
    null_rates = _null_hit_rate(combined, n_random=10000)
    for length, rate in sorted(null_rates.items()):
        print(f"     Length {length}: {rate:.4f} ({rate*100:.2f}%)")

    # Overall baseline for length 2-4 (typical root lengths)
    random_baseline = sum(null_rates.get(l, 0) for l in [2, 3, 4]) / 3

    elapsed = time.time() - t0

    result = SiglaDictionaryResult(
        n_base_words=len(base_words),
        n_unique_skeletons=len(skeletons),
        skeleton_length_distribution={str(k): v for k, v in sorted(length_dist.items())},
        most_ambiguous_skeletons=top_ambiguous,
        cappelli_path=cappelli_path,
        n_cappelli_entries=len(sigla),
        n_cappelli_latin=n_latin,
        n_cappelli_pharmaceutical=n_pharma,
        n_cappelli_medical=n_medical,
        cappelli_domain_breakdown=dict(domain_breakdown),
        sample_sigla=sigla[:20],
        n_combined_entries=len(combined),
        overlap_count=overlap,
        null_hit_rates={str(k): round(v, 6) for k, v in sorted(null_rates.items())},
        random_baseline_hit_rate=round(random_baseline, 6),
        runtime_seconds=round(elapsed, 1),
    )

    rd = _results_dir()
    out_path = os.path.join(rd, 'sigla_dictionary.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved → {out_path}")

    print(f"\n  Completed in {elapsed:.1f}s")
