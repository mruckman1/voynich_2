"""
Step 39.11 – Venetian Lexicon
==============================
Build a standalone Venetian word list from the Anonimo Veneziano,
classified relative to the existing Latin and Italian 10K dictionaries.

Dependency chain:
    data/reference/italian/anonimo_veneziano.txt
    merged_dict.json           (Step 38.1)
        → venetian_lexicon.json (this step)
"""

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Set

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def _normalize_venetian(text: str) -> str:
    """Normalize Venetian text (lowercase, strip accents partially)."""
    text = text.lower()
    # Keep accented chars for accurate representation but also provide
    # deaccented form for matching
    return text


def _tokenize(text: str) -> List[str]:
    """Tokenize text into words (alphabetic + accented chars)."""
    return re.findall(r'[a-zA-ZàèéìòùÀÈÉÌÒÙ]+', text.lower())


def _deaccent(word: str) -> str:
    """Remove accents for dictionary matching."""
    accent_map = {
        'à': 'a', 'è': 'e', 'é': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
    }
    return ''.join(accent_map.get(ch, ch) for ch in word)


# ---------------------------------------------------------------------------
# Venetian phonological markers
# ---------------------------------------------------------------------------

# Venetian-specific features to search for in decoded corpus
VENETIAN_MARKERS = {
    'el_article': re.compile(r'\bel\b'),          # Venetian definite article
    'degemination': re.compile(r'(?<=[aeiou])([bcdfglmnprstvz])\1'),  # double→single
    'aro_ending': re.compile(r'aro\b'),            # -aro instead of -aio
    'ero_ending': re.compile(r'ero\b'),            # -ero instead of -iere
    'intervocalic_d_loss': re.compile(r'(?<=[aeiou])(?=[aeiou])'),  # V_V where d dropped
}

# Venetian preparation verbs (from cookbook traditions)
VENETIAN_PREPARATION_VOCAB = {
    'verbs': [
        'toy', 'toi',           # take (Venetian imperative)
        'fa', 'far',            # make, do
        'meti', 'mete',         # put
        'meschola', 'mescola',  # mix
        'boli', 'bolire',       # boil
        'cola', 'colar',        # strain
        'pesta', 'pestar',      # grind
        'taglia', 'taiar',      # cut
        'frize', 'frixere',     # fry
        'lessa', 'lessar',      # boil (less common)
        'destempera',           # dissolve/temper
    ],
    'containers': [
        'pignata', 'pignatta',  # pot
        'padella',              # pan
        'caldiera',             # cauldron
        'mortaro',              # mortar
        'scudella',             # bowl
    ],
    'ingredients': [
        'specie', 'spezie',     # spices
        'pevere', 'pepe',       # pepper
        'zengevro', 'zenzevro', # ginger
        'canella',              # cinnamon
        'garofali',             # cloves
        'zucharo', 'zucaro',    # sugar
        'mandole', 'amandole',  # almonds
        'lacte', 'late',        # milk
        'ovi', 'ova',           # eggs
        'brodo',                # broth
    ],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_venetian_lexicon() -> None:
    """Step 39.11: Venetian Lexicon."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.11: Venetian Lexicon")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    dict_data = _safe_load(os.path.join(rd, 'merged_dict.json'))
    latin_10k = set(dict_data.get('latin_10k_words', []))
    italian_10k = set(dict_data.get('italian_10k_words', []))

    anonimo_path = os.path.join(str(_data_dir()), 'reference', 'italian',
                                'anonimo_veneziano.txt')
    if not os.path.exists(anonimo_path):
        print(f"     ERROR: {anonimo_path} not found")
        return

    with open(anonimo_path, encoding='utf-8', errors='replace') as f:
        text = f.read()

    print(f"     Latin 10K size: {len(latin_10k)}")
    print(f"     Italian 10K size: {len(italian_10k)}")
    print(f"     Anonimo text length: {len(text)} chars")

    # ── 2. Tokenize and count ──
    print("\n  2. Tokenizing Anonimo Veneziano …")
    tokens = _tokenize(text)
    word_counts = Counter(tokens)
    n_types_raw = len(word_counts)
    print(f"     Total tokens: {len(tokens)}")
    print(f"     Unique types: {n_types_raw}")

    # ── 3. Frequency filter (≥2) ──
    print("\n  3. Applying frequency filter (≥2) …")
    freq2_words = {w for w, c in word_counts.items() if c >= 2}
    print(f"     Types with freq ≥ 2: {len(freq2_words)}")

    # ── 4. Classify relative to existing dictionaries ──
    print("\n  4. Classifying vocabulary …")
    shared_with_latin = set()
    shared_with_italian = set()
    venetian_specific = set()
    shared_both = set()

    for word in freq2_words:
        deacc = _deaccent(word)
        in_latin = word in latin_10k or deacc in latin_10k
        in_italian = word in italian_10k or deacc in italian_10k

        if in_latin and in_italian:
            shared_both.add(word)
        elif in_latin:
            shared_with_latin.add(word)
        elif in_italian:
            shared_with_italian.add(word)
        else:
            venetian_specific.add(word)

    print(f"     Shared with both: {len(shared_both)}")
    print(f"     Shared with Latin only: {len(shared_with_latin)}")
    print(f"     Shared with Italian only: {len(shared_with_italian)}")
    print(f"     Venetian-specific: {len(venetian_specific)}")

    # ── 5. Extract preparation vocabulary ──
    print("\n  5. Extracting preparation vocabulary …")
    # Find Venetian preparation words that appear in the text
    found_prep_verbs = []
    for v in VENETIAN_PREPARATION_VOCAB['verbs']:
        if v in word_counts:
            found_prep_verbs.append({'word': v, 'count': word_counts[v]})

    found_containers = []
    for c in VENETIAN_PREPARATION_VOCAB['containers']:
        if c in word_counts:
            found_containers.append({'word': c, 'count': word_counts[c]})

    found_ingredients = []
    for ing in VENETIAN_PREPARATION_VOCAB['ingredients']:
        if ing in word_counts:
            found_ingredients.append({'word': ing, 'count': word_counts[ing]})

    print(f"     Preparation verbs found: {len(found_prep_verbs)}")
    print(f"     Containers found: {len(found_containers)}")
    print(f"     Ingredients found: {len(found_ingredients)}")

    # ── 6. Build supplement dictionary ──
    print("\n  6. Building Venetian supplement dictionary …")

    # The supplement includes Venetian-specific words + preparation vocab
    supplement = set(venetian_specific)
    for entry in found_prep_verbs + found_containers + found_ingredients:
        supplement.add(entry['word'])

    # Also add deaccented forms for matching
    supplement_with_deaccent = set()
    for w in supplement:
        supplement_with_deaccent.add(w)
        deacc = _deaccent(w)
        if deacc != w:
            supplement_with_deaccent.add(deacc)

    print(f"     Supplement size: {len(supplement_with_deaccent)}")

    # ── 7. Top Venetian-specific words ──
    print("\n  7. Top Venetian-specific words (by frequency) …")
    ven_by_freq = sorted(venetian_specific,
                         key=lambda w: word_counts[w], reverse=True)
    for w in ven_by_freq[:20]:
        print(f"     {w}: {word_counts[w]}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens_total': len(tokens),
        'n_types_raw': n_types_raw,
        'n_types_freq2plus': len(freq2_words),
        'n_shared_both': len(shared_both),
        'n_shared_latin_only': len(shared_with_latin),
        'n_shared_italian_only': len(shared_with_italian),
        'n_venetian_specific': len(venetian_specific),
        'venetian_words': sorted(venetian_specific),
        'supplement_words': sorted(supplement_with_deaccent),
        'n_supplement': len(supplement_with_deaccent),
        'preparation_verbs': found_prep_verbs,
        'preparation_containers': found_containers,
        'preparation_ingredients': found_ingredients,
        'top_venetian_words': [
            {'word': w, 'count': word_counts[w]}
            for w in ven_by_freq[:50]
        ],
        'verdict': (
            f"{len(venetian_specific)} Venetian-specific words, "
            f"{len(supplement_with_deaccent)} supplement (with deaccented), "
            f"{len(found_prep_verbs)} prep verbs, "
            f"{len(found_ingredients)} ingredients."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'venetian_lexicon.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
