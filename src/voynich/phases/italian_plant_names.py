"""
Step 39.8 – Italian Plant Name Dictionary
==========================================
Build Italian and Venetian vernacular plant names for the botanical
concordance folios.  Phase 33 tested Latin names and found zero valid
alignments; Italian names have different syllable structures that may
be compatible with confirmed triples.

Dependency chain:
    data/reference/voynich_plant/medieval_latin_names.json
    data/reference/voynich_plant/Voynich_Herbal_Multi-Source_Identification_Concordance.csv
    data/reference/italian/anonimo_veneziano.txt
        → italian_plant_names.json   (this step)
"""

import csv
import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

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
# Italian plant name table (hardcoded domain knowledge)
# ---------------------------------------------------------------------------

# Linnaean name → { 'italian': [names], 'venetian': [names] }
ITALIAN_PLANT_NAMES: Dict[str, Dict[str, List[str]]] = {
    'Viola odorata': {
        'italian': ['viola', 'violetta', 'viole'],
        'venetian': ['viola'],
    },
    'Rosmarinus officinalis': {
        'italian': ['rosmarino', 'ramerino'],
        'venetian': ['rosmarin'],
    },
    'Papaver somniferum': {
        'italian': ['papavero', 'papavere'],
        'venetian': ['papavero'],
    },
    'Malva sylvestris': {
        'italian': ['malva'],
        'venetian': ['malva'],
    },
    'Cannabis sativa': {
        'italian': ['canapa', 'canape'],
        'venetian': ['caneva'],
    },
    'Anagallis arvensis': {
        'italian': ['anagallide', 'centonchio'],
        'venetian': [],
    },
    'Carthamus tinctorius': {
        'italian': ['cartamo', 'zafferanone'],
        'venetian': ['zafferanone'],
    },
    'Pulmonaria officinalis': {
        'italian': ['polmonaria'],
        'venetian': [],
    },
    'Calendula officinalis': {
        'italian': ['calendola', 'fiorrancio'],
        'venetian': ['calendola'],
    },
    'Salvia officinalis': {
        'italian': ['salvia'],
        'venetian': ['salvia'],
    },
    'Artemisia absinthium': {
        'italian': ['assenzio'],
        'venetian': ['assenzo'],
    },
    'Mentha piperita': {
        'italian': ['menta', 'mentuccia'],
        'venetian': ['menta'],
    },
    'Plantago major': {
        'italian': ['piantaggine', 'plantago'],
        'venetian': [],
    },
    'Borago officinalis': {
        'italian': ['borragine', 'boragine'],
        'venetian': ['borana'],
    },
    'Urtica dioica': {
        'italian': ['ortica'],
        'venetian': ['ortiga'],
    },
    'Foeniculum vulgare': {
        'italian': ['finocchio'],
        'venetian': ['fenoccio', 'fenochio'],
    },
    'Petroselinum crispum': {
        'italian': ['prezzemolo', 'petrosello'],
        'venetian': ['petroselo'],
    },
    'Ocimum basilicum': {
        'italian': ['basilico'],
        'venetian': ['basilico'],
    },
    'Lavandula angustifolia': {
        'italian': ['lavanda', 'lavandola'],
        'venetian': ['lavanda'],
    },
    'Crocus sativus': {
        'italian': ['zafferano'],
        'venetian': ['safran', 'zafferan'],
    },
    'Cinnamomum verum': {
        'italian': ['cannella'],
        'venetian': ['canella'],
    },
    'Zingiber officinale': {
        'italian': ['zenzero'],
        'venetian': ['zengevro', 'zenzevro'],
    },
    'Syzygium aromaticum': {
        'italian': ['garofano', 'chiodo di garofano'],
        'venetian': ['garofali', 'garofalo'],
    },
    'Piper nigrum': {
        'italian': ['pepe'],
        'venetian': ['pevere'],
    },
    'Rosa gallica': {
        'italian': ['rosa'],
        'venetian': ['rosa'],
    },
    'Lilium candidum': {
        'italian': ['giglio'],
        'venetian': ['zigio'],
    },
    'Drosera rotundifolia': {
        'italian': ['drosera', 'rosolida'],
        'venetian': [],
    },
    'Euphorbia lathyris': {
        'italian': ['euforbia', 'catapuzia'],
        'venetian': [],
    },
    'Ricinus communis': {
        'italian': ['ricino'],
        'venetian': [],
    },
    'Helianthus annuus': {
        'italian': ['girasole'],
        'venetian': [],
    },
    'Nymphaea alba': {
        'italian': ['ninfea', 'nenufaro'],
        'venetian': ['nenufaro'],
    },
    'Centaurea cyanus': {
        'italian': ['fiordaliso', 'centaurea'],
        'venetian': [],
    },
    'Aconitum napellus': {
        'italian': ['aconito', 'napello'],
        'venetian': [],
    },
    'Mandragora officinarum': {
        'italian': ['mandragora', 'mandragola'],
        'venetian': ['mandragora'],
    },
    'Helleborus niger': {
        'italian': ['elleboro'],
        'venetian': [],
    },
    'Digitalis purpurea': {
        'italian': ['digitale'],
        'venetian': [],
    },
    'Aloe vera': {
        'italian': ['aloe'],
        'venetian': ['aloe'],
    },
    'Cucumis sativus': {
        'italian': ['cocomero', 'cetriolo'],
        'venetian': ['cedriolo'],
    },
    'Linum usitatissimum': {
        'italian': ['lino'],
        'venetian': ['lin'],
    },
    'Thymus vulgaris': {
        'italian': ['timo'],
        'venetian': ['timo'],
    },
    'Origanum vulgare': {
        'italian': ['origano'],
        'venetian': ['origano'],
    },
    'Matricaria chamomilla': {
        'italian': ['camomilla'],
        'venetian': ['camomilla'],
    },
    'Verbena officinalis': {
        'italian': ['verbena'],
        'venetian': ['verbena'],
    },
    'Sambucus nigra': {
        'italian': ['sambuco'],
        'venetian': ['sambuco'],
    },
    'Achillea millefolium': {
        'italian': ['millefoglio', 'achillea'],
        'venetian': [],
    },
    'Hypericum perforatum': {
        'italian': ['iperico', 'erba di san giovanni'],
        'venetian': [],
    },
    'Taraxacum officinale': {
        'italian': ['tarassaco', 'dente di leone'],
        'venetian': [],
    },
    'Melissa officinalis': {
        'italian': ['melissa', 'cedronella'],
        'venetian': ['melissa'],
    },
    'Ruta graveolens': {
        'italian': ['ruta'],
        'venetian': ['ruda'],
    },
}


# ---------------------------------------------------------------------------
# Syllabification
# ---------------------------------------------------------------------------

def _syllabify_cv(word: str) -> List[str]:
    """Simple CV syllabification for Italian words."""
    vowels = set('aeiou')
    word = word.lower()
    syllables = []
    current = []

    i = 0
    while i < len(word):
        ch = word[i]
        if ch in vowels:
            current.append(ch)
            syllables.append(''.join(current))
            current = []
        else:
            if current and any(c in vowels for c in current):
                # Previous syllable had a vowel, start new
                syllables.append(''.join(current))
                current = [ch]
            else:
                current.append(ch)
        i += 1

    if current:
        if syllables:
            syllables[-1] += ''.join(current)
        else:
            syllables.append(''.join(current))

    return syllables


# ---------------------------------------------------------------------------
# Venetian ingredient extraction
# ---------------------------------------------------------------------------

def _extract_venetian_ingredients(anonimo_path: str) -> List[Dict]:
    """Extract ingredient/plant names from Anonimo Veneziano."""
    if not os.path.exists(anonimo_path):
        return []

    with open(anonimo_path, encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Tokenize
    words = re.findall(r'[a-zA-ZàèéìòùÀÈÉÌÒÙ]+', text.lower())
    word_counts = Counter(words)

    # Known ingredient/plant-related terms
    botanical_seeds = {
        'erba', 'erbe', 'fior', 'fiore', 'fiori', 'foglia', 'foglie',
        'radice', 'radici', 'seme', 'semi', 'scorza', 'corteccia',
        'frutto', 'frutti', 'olio', 'aqua', 'vino', 'aceto',
        'sale', 'miele', 'zucaro', 'zucharo', 'pevere', 'pepe',
        'canella', 'garofali', 'zengevro', 'zenzevro', 'noce',
        'mandole', 'amandole', 'rosa', 'viole', 'specie', 'spezie',
    }

    ingredients = []
    for word, count in word_counts.most_common():
        if count >= 3 and (word in botanical_seeds or len(word) >= 5):
            if word in botanical_seeds:
                ingredients.append({
                    'word': word,
                    'count': count,
                    'type': 'botanical_term',
                })
    return ingredients


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_italian_plant_names() -> None:
    """Step 39.8: Italian Plant Name Dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.8: Italian Plant Name Dictionary")
    print("=" * 70)

    rd = _results_dir()
    plant_dir = os.path.join(str(_data_dir()), 'reference', 'voynich_plant')
    italian_dir = os.path.join(str(_data_dir()), 'reference', 'italian')

    # ── 1. Load botanical concordance ──
    print("\n  1. Loading botanical concordance …")
    concordance_path = os.path.join(
        plant_dir, 'Voynich_Herbal_Multi-Source_Identification_Concordance.csv')
    medieval_path = os.path.join(plant_dir, 'medieval_latin_names.json')

    folio_to_species: Dict[str, List[str]] = {}
    if os.path.exists(concordance_path):
        with open(concordance_path, encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                folio = row.get('Folio', '').strip()
                species = row.get('Proposed Botanical Identification', '').strip()
                if folio and species:
                    folio_to_species.setdefault(folio, []).append(species)
    print(f"     Folios with identifications: {len(folio_to_species)}")

    medieval_names = _safe_load(medieval_path)
    print(f"     Medieval Latin names: {len(medieval_names)}")

    # ── 2. Build folio → Italian plant name table ──
    print("\n  2. Building folio → Italian name table …")

    plant_name_table = []
    n_with_italian = 0

    for folio, species_list in sorted(folio_to_species.items()):
        for species in species_list:
            # Try to match species to ITALIAN_PLANT_NAMES
            italian_names = []
            venetian_names = []
            medieval_latin = ''

            # Look up by exact Linnaean name
            if species in ITALIAN_PLANT_NAMES:
                italian_names = ITALIAN_PLANT_NAMES[species]['italian']
                venetian_names = ITALIAN_PLANT_NAMES[species]['venetian']

            # Try partial match (genus only)
            if not italian_names:
                genus = species.split()[0] if species else ''
                for key, val in ITALIAN_PLANT_NAMES.items():
                    if key.startswith(genus + ' '):
                        italian_names = val['italian']
                        venetian_names = val['venetian']
                        break

            # Medieval Latin name
            if species in medieval_names:
                medieval_latin = medieval_names[species].get('medieval_name', '')

            if italian_names:
                n_with_italian += 1

            # Syllabify all names
            syllabified = {}
            for name in italian_names + venetian_names:
                if ' ' not in name:  # skip multi-word names
                    syllabified[name] = _syllabify_cv(name)

            plant_name_table.append({
                'folio': folio,
                'latin_binomial': species,
                'medieval_latin': medieval_latin,
                'italian_names': italian_names,
                'venetian_names': venetian_names,
                'syllabified': syllabified,
            })

    print(f"     Plant entries: {len(plant_name_table)}")
    print(f"     With Italian names: {n_with_italian}")

    # ── 3. Extract Venetian ingredients from Anonimo ──
    print("\n  3. Extracting Venetian ingredients …")
    anonimo_path = os.path.join(italian_dir, 'anonimo_veneziano.txt')
    venetian_ingredients = _extract_venetian_ingredients(anonimo_path)
    print(f"     Venetian ingredients found: {len(venetian_ingredients)}")

    # ── 4. Build unified Italian botanical vocabulary ──
    print("\n  4. Building unified vocabulary …")
    all_italian_plant_words: Set[str] = set()
    for entry in plant_name_table:
        for name in entry['italian_names'] + entry['venetian_names']:
            if ' ' not in name:
                all_italian_plant_words.add(name.lower())

    for ing in venetian_ingredients:
        all_italian_plant_words.add(ing['word'])

    print(f"     Total unique plant/ingredient words: {len(all_italian_plant_words)}")

    # ── 5. Save ──
    elapsed = time.time() - t0

    output = {
        'n_concordance_folios': len(folio_to_species),
        'n_plant_entries': len(plant_name_table),
        'n_with_italian_name': n_with_italian,
        'n_venetian_ingredients': len(venetian_ingredients),
        'n_unique_plant_words': len(all_italian_plant_words),
        'plant_name_table': plant_name_table,
        'venetian_ingredients': venetian_ingredients,
        'all_plant_words': sorted(all_italian_plant_words),
        'verdict': (
            f"{len(plant_name_table)} plant entries, "
            f"{n_with_italian} with Italian names, "
            f"{len(venetian_ingredients)} Venetian ingredients, "
            f"{len(all_italian_plant_words)} unique plant words."
        ),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'italian_plant_names.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
