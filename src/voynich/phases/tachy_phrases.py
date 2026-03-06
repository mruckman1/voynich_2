"""
Phase 20.6 – Latin Phrase Detection and Botanical Cross-Check
=============================================================
Extract candidate Latin phrases from the decoded text using a sliding window
and cross-check botanical phrases against the illustration database.

Dependency chain:
    tachy_decode.json + Latin reference corpus
        → tachy_phrases.json
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    LATIN_PHRASE_PATTERNS,
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.core.stats import compute_phrase_selectivity


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CandidatePhrase:
    words: List[str]
    position: int           # token index in corpus
    folio: str
    dict_hit_fraction: float
    pattern_match: str      # matched pattern name or ''
    category: str           # recipe, humoral, application, botanical, other


@dataclass
class TachyPhrasesResult:
    n_phrases_detected: int
    phrase_categories: Dict[str, int]
    phrases: List[Dict]
    null_phrase_mean: float
    null_phrase_std: float
    phrase_selectivity: float
    phrase_z_score: float
    n_botanical_folios_checked: int
    n_botanical_matches: int
    botanical_matches: List[Dict]
    botanical_p_value: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_json(rd: str, fname: str) -> Dict:
    path = os.path.join(rd, fname)
    if not os.path.exists(path):
        print(f"    [WARN] {fname} not found")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phrase classification
# ---------------------------------------------------------------------------

_RECIPE_WORDS = {'recipe', 'accipe', 'misce', 'coque', 'fac', 'tere',
                 'cola', 'destilla', 'solve', 'adde', 'pone'}
_HUMORAL_WORDS = {'calidus', 'calida', 'calidum', 'frigidus', 'frigida',
                  'humidus', 'humida', 'siccus', 'sicca', 'gradus',
                  'temperatus', 'temperata'}
_APPLICATION_WORDS = {'applica', 'bibe', 'da', 'super', 'loco', 'impone',
                      'unge', 'lava', 'gargariza'}
_BOTANICAL_WORDS = {'radix', 'folia', 'folium', 'semen', 'cortex', 'herba',
                    'flos', 'flores', 'succus', 'ramus', 'planta'}


def _classify_phrase(words: List[str]) -> str:
    """Classify a phrase by its dominant vocabulary domain."""
    word_set = set(w.lower() for w in words)
    if word_set & _RECIPE_WORDS:
        return 'recipe'
    if word_set & _HUMORAL_WORDS:
        return 'humoral'
    if word_set & _APPLICATION_WORDS:
        return 'application'
    if word_set & _BOTANICAL_WORDS:
        return 'botanical'
    return 'other'


# ---------------------------------------------------------------------------
# Phrase extraction
# ---------------------------------------------------------------------------

def _extract_phrases(
    decoded_words: List[str],
    ref_word_set: set,
    phrase_patterns: List[Tuple[str, List[str]]],
    min_window: int = 3,
    max_window: int = 8,
    min_dict_fraction: float = 0.5,
) -> List[CandidatePhrase]:
    """Sliding-window phrase extraction."""
    candidates = []

    # Build pattern lookup for fast matching
    pattern_strings: Dict[str, str] = {}
    for name, templates in phrase_patterns:
        for t in templates:
            pattern_strings[t.lower()] = name

    for window_size in range(min_window, max_window + 1):
        for i in range(len(decoded_words) - window_size + 1):
            window = decoded_words[i:i + window_size]

            # Dict hit fraction
            hits = sum(1 for w in window if w.lower() in ref_word_set)
            frac = hits / len(window)
            if frac < min_dict_fraction:
                continue

            # Pattern match
            window_text = ' '.join(w.lower() for w in window)
            matched_pattern = ''
            for pattern, name in pattern_strings.items():
                if pattern in window_text:
                    matched_pattern = name
                    break

            # At least one known medical term
            word_set = set(w.lower() for w in window)
            has_medical = bool(
                word_set & _RECIPE_WORDS
                | word_set & _HUMORAL_WORDS
                | word_set & _APPLICATION_WORDS
                | word_set & _BOTANICAL_WORDS
            )

            if frac >= 0.6 or matched_pattern or has_medical:
                category = _classify_phrase(window)
                candidates.append(CandidatePhrase(
                    words=window,
                    position=i,
                    folio='',  # filled in later if available
                    dict_hit_fraction=frac,
                    pattern_match=matched_pattern,
                    category=category,
                ))

    # Deduplicate overlapping phrases — keep highest dict_hit_fraction
    if not candidates:
        return []

    candidates.sort(key=lambda c: (-c.dict_hit_fraction, c.position))
    filtered = []
    used_positions = set()
    for c in candidates:
        pos_range = set(range(c.position, c.position + len(c.words)))
        if pos_range & used_positions:
            continue
        filtered.append(c)
        used_positions |= pos_range

    return filtered


# ---------------------------------------------------------------------------
# Null phrase detection
# ---------------------------------------------------------------------------

def _null_phrase_counts(
    ref_word_set: set,
    n_words: int,
    phrase_patterns: List[Tuple[str, List[str]]],
    n_trials: int = 20,
) -> List[int]:
    """Count phrases in random word sequences."""
    rng = random.Random(42)
    word_list = sorted(ref_word_set)
    if not word_list:
        return [0] * n_trials

    counts = []
    for _ in range(n_trials):
        random_words = [rng.choice(word_list) for _ in range(n_words)]
        phrases = _extract_phrases(random_words, ref_word_set, phrase_patterns)
        counts.append(len(phrases))
    return counts


# ---------------------------------------------------------------------------
# Botanical cross-check
# ---------------------------------------------------------------------------

# Known botanical identifications for select folios
_BOTANICAL_IDS = {
    'f1v': 'centaurea',
    'f2r': 'plantago',
    'f3r': 'viola',
    'f4r': 'salvia',
    'f5r': 'rosa',
    'f6r': 'artemisia',
    'f9v': 'helleborus',
    'f11r': 'mentha',
    'f13r': 'calendula',
    'f15r': 'papaver',
    'f16r': 'urtica',
    'f17r': 'sambucus',
    'f22r': 'melissa',
    'f25r': 'rosmarinus',
    'f33v': 'cannabis',
    'f34r': 'borago',
    'f35r': 'malva',
    'f38r': 'linum',
    'f41r': 'ruta',
    'f43r': 'nymphaea',
    'f44r': 'verbena',
    'f47r': 'coriandrum',
    'f49r': 'cuminum',
    'f50r': 'foeniculum',
    'f52r': 'anethum',
    'f53r': 'apium',
    'f55r': 'petroselinum',
    'f56r': 'levisticum',
}


def _botanical_cross_check(
    per_folio_data: List[Dict],
    ref_word_set: set,
) -> List[Dict]:
    """Check if decoded text on botanical folios contains plant names."""
    matches = []
    for folio_entry in per_folio_data:
        folio = folio_entry.get('folio', '')
        # Normalise folio name
        folio_key = folio.lower().replace(' ', '')

        if folio_key not in _BOTANICAL_IDS:
            continue

        expected_plant = _BOTANICAL_IDS[folio_key]
        sample = folio_entry.get('sample', [])

        # Check if any decoded word contains the plant name stem
        for voynich, decoded in sample:
            decoded_lower = decoded.lower() if decoded else ''
            # Check stem match (first 4+ chars of plant name)
            stem = expected_plant[:4]
            if stem in decoded_lower:
                matches.append({
                    'folio': folio,
                    'expected_plant': expected_plant,
                    'decoded_word': decoded,
                    'voynich_token': voynich,
                    'match_type': 'stem',
                })
            # Also check if the plant name is in the reference and decoded
            elif expected_plant in decoded_lower:
                matches.append({
                    'folio': folio,
                    'expected_plant': expected_plant,
                    'decoded_word': decoded,
                    'voynich_token': voynich,
                    'match_type': 'exact',
                })

    return matches


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tachy_phrases() -> None:
    """Step 20.6: Phrase detection and botanical cross-check."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 20.6: Latin Phrase Detection + Botanical Cross-Check")
    print("=" * 70)

    rd = _results_dir()

    # ─── 1. Load dependencies ───
    print("\n  1. Loading dependencies …")
    decode_data = _load_json(rd, 'tachy_decode.json')

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words

    # Extract decoded word sequence
    decoded_words: List[str] = []
    for entry in decode_data.get('decoded_sample', []):
        if len(entry) >= 2:
            decoded = entry[1]
            if decoded and decoded != '?':
                decoded_words.append(decoded.lower())

    # Also use top_decoded_words for a larger sample
    for word, count in decode_data.get('top_decoded_words', []):
        for _ in range(min(count, 5)):
            decoded_words.append(word.lower())

    print(f"      Decoded words for analysis: {len(decoded_words)}")

    # ─── 2. Extract phrases ───
    print("\n  2. Extracting candidate phrases …")
    phrases = _extract_phrases(
        decoded_words, ref_word_set, LATIN_PHRASE_PATTERNS,
    )
    n_phrases = len(phrases)

    # Categorise
    cat_counts: Counter = Counter()
    for p in phrases:
        cat_counts[p.category] += 1

    print(f"      Phrases detected: {n_phrases}")
    for cat, count in cat_counts.most_common():
        print(f"        {cat}: {count}")

    for p in phrases[:15]:
        print(f"        [{p.category}] {' '.join(p.words)} "
              f"(dict={p.dict_hit_fraction:.0%}"
              f"{', pattern=' + p.pattern_match if p.pattern_match else ''})")

    # ─── 3. Null comparison ───
    print("\n  3. Null phrase comparison …")
    null_counts = _null_phrase_counts(
        ref_word_set, len(decoded_words), LATIN_PHRASE_PATTERNS,
    )
    null_mean = float(np.mean(null_counts))
    null_std = float(np.std(null_counts)) if len(null_counts) > 1 else 1.0

    if null_mean > 0:
        phrase_selectivity = n_phrases / null_mean
    else:
        phrase_selectivity = float('inf') if n_phrases > 0 else 1.0

    z_score = (n_phrases - null_mean) / null_std if null_std > 0 else 0.0
    print(f"      Null phrases: {null_mean:.1f} ± {null_std:.1f}")
    print(f"      Selectivity: {phrase_selectivity:.2f}×")
    print(f"      Z-score: {z_score:.2f}")

    # ─── 4. Botanical cross-check ───
    print("\n  4. Botanical cross-check …")
    per_folio = decode_data.get('per_folio_summary', [])
    botanical_matches = _botanical_cross_check(per_folio, ref_word_set)
    n_botanical = len(botanical_matches)
    n_botanical_folios = len(_BOTANICAL_IDS)

    for m in botanical_matches[:10]:
        print(f"      {m['folio']}: expected '{m['expected_plant']}' "
              f"→ decoded '{m['decoded_word']}' ({m['match_type']})")

    # Permutation p-value for botanical matches
    rng = random.Random(42)
    n_perms = 1000
    perm_counts = []
    folio_list = [f.get('folio', '') for f in per_folio]
    for _ in range(n_perms):
        shuffled = list(folio_list)
        rng.shuffle(shuffled)
        # Reassign folio labels and re-check
        shuffled_data = []
        for i, entry in enumerate(per_folio):
            shuffled_entry = dict(entry)
            shuffled_entry['folio'] = shuffled[i] if i < len(shuffled) else ''
            shuffled_data.append(shuffled_entry)
        perm_matches = _botanical_cross_check(shuffled_data, ref_word_set)
        perm_counts.append(len(perm_matches))

    p_value = sum(1 for c in perm_counts if c >= n_botanical) / n_perms
    print(f"      Botanical matches: {n_botanical}/{n_botanical_folios}")
    print(f"      Permutation p-value: {p_value:.4f}")

    # ─── 5. Gate ───
    gate_passed = n_phrases >= 3 and phrase_selectivity > 2.0
    if gate_passed:
        verdict = (f"PASS: {n_phrases} phrases (selectivity={phrase_selectivity:.2f}×). "
                   f"{n_botanical} botanical matches (p={p_value:.4f}).")
    else:
        verdict = (f"FAIL: {n_phrases} phrases (need ≥3, "
                   f"selectivity={phrase_selectivity:.2f}× need >2.0). "
                   f"{n_botanical} botanical matches.")

    print(f"\n  5. Gate: {verdict}")

    # ─── 6. Save ───
    result = TachyPhrasesResult(
        n_phrases_detected=n_phrases,
        phrase_categories=dict(cat_counts),
        phrases=[asdict(p) for p in phrases[:100]],
        null_phrase_mean=null_mean,
        null_phrase_std=null_std,
        phrase_selectivity=phrase_selectivity,
        phrase_z_score=z_score,
        n_botanical_folios_checked=n_botanical_folios,
        n_botanical_matches=n_botanical,
        botanical_matches=botanical_matches,
        botanical_p_value=p_value,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out_path = os.path.join(rd, 'tachy_phrases.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
