"""
Step 39.6 – Phrase Template Alignment
======================================
Align medical phrases against standardized macaronic pharmaceutical
templates.  Build templates from Circa Instans (Latin) and Anonimo
Veneziano (Italian) patterns.  For each phrase, try to match against
templates.  Template-predicted words for MISS positions are used to
extract corrections.

Dependency chain:
    phrase_cribs.json          (Step 39.5)
    reference corpora
        → phrase_alignment.json (this step)
"""

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir


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
# Template definitions
# ---------------------------------------------------------------------------

# Pharmaceutical templates modeled on Circa Instans (Latin) and
# Anonimo Veneziano (Italian) patterns.  Each template is a list of
# slot entries.  Slot types:
#   VERB       – pharma verb (recipe, cola, misce, …)
#   PLANT      – plant/ingredient name
#   LIQUID     – liquid medium (vino, olio, …)
#   BODY_PART  – anatomical term
#   QUALITY    – adjective (calida, frigida, …)
#   PREP       – preposition (in, de, cum, per, …)
#   CONJ       – conjunction (et, e)
#   ARTICLE    – article (la, le)
#   DEGREE     – degree marker (primo, secundo, …)
#   LINK       – linking verb (est, fa, sit, fit)
#   ANY        – any word

TEMPLATES = [
    # ── Latin pharmaceutical templates ──
    {
        'name': 'L1_recipe_coque_cola',
        'language': 'Latin',
        'source': 'Circa Instans',
        'slots': [
            {'type': 'VERB', 'words': {'recipe'}},
            {'type': 'PLANT', 'words': None},
            {'type': 'PREP', 'words': {'in', 'cum', 'de'}},
            {'type': 'LIQUID', 'words': None},
            {'type': 'VERB', 'words': {'coque', 'cola'}},
        ],
    },
    {
        'name': 'L2_est_quality_gradu',
        'language': 'Latin',
        'source': 'Circa Instans',
        'slots': [
            {'type': 'LINK', 'words': {'est', 'sit'}},
            {'type': 'QUALITY', 'words': None},
            {'type': 'PREP', 'words': {'in'}},
            {'type': 'DEGREE', 'words': {'primo', 'secundo', 'tertio', 'quarto'}},
        ],
    },
    {
        'name': 'L3_cura_body_cum',
        'language': 'Latin',
        'source': 'Circa Instans',
        'slots': [
            {'type': 'VERB', 'words': {'cura', 'sana'}},
            {'type': 'BODY_PART', 'words': None},
            {'type': 'PREP', 'words': {'cum', 'per', 'de'}},
            {'type': 'PLANT', 'words': None},
        ],
    },
    {
        'name': 'L4_misce_et_cola',
        'language': 'Latin',
        'source': 'Circa Instans',
        'slots': [
            {'type': 'VERB', 'words': {'misce'}},
            {'type': 'PLANT', 'words': None},
            {'type': 'CONJ', 'words': {'et', 'cum'}},
            {'type': 'PLANT', 'words': None},
            {'type': 'VERB', 'words': {'cola'}},
        ],
    },
    {
        'name': 'L5_bibe_prep_body',
        'language': 'Latin',
        'source': 'Circa Instans',
        'slots': [
            {'type': 'VERB', 'words': {'bibe', 'recipe'}},
            {'type': 'PREP', 'words': {'per', 'ad', 'in', 'de'}},
            {'type': 'BODY_PART', 'words': None},
        ],
    },
    {
        'name': 'L6_plant_est_quality',
        'language': 'Latin',
        'source': 'Circa Instans',
        'slots': [
            {'type': 'PLANT', 'words': None},
            {'type': 'LINK', 'words': {'est'}},
            {'type': 'QUALITY', 'words': None},
            {'type': 'CONJ', 'words': {'et'}},
            {'type': 'QUALITY', 'words': None},
        ],
    },
    # ── Italian pharmaceutical templates ──
    {
        'name': 'I1_toy_ingredient_verb',
        'language': 'Italian',
        'source': 'Anonimo Veneziano',
        'slots': [
            {'type': 'VERB', 'words': {'dice', 'beni'}},
            {'type': 'PLANT', 'words': None},
            {'type': 'CONJ', 'words': {'e', 'et'}},
            {'type': 'VERB', 'words': {'cola', 'misce', 'coque'}},
            {'type': 'PREP', 'words': {'con', 'in'}},
            {'type': 'LIQUID', 'words': None},
        ],
    },
    {
        'name': 'I2_quality_body',
        'language': 'Italian',
        'source': 'Anonimo Veneziano',
        'slots': [
            {'type': 'ARTICLE', 'words': {'la', 'le'}},
            {'type': 'PLANT', 'words': None},
            {'type': 'LINK', 'words': {'est', 'fa'}},
            {'type': 'QUALITY', 'words': None},
        ],
    },
    {
        'name': 'I3_cola_con_liquid',
        'language': 'Italian',
        'source': 'Anonimo Veneziano',
        'slots': [
            {'type': 'VERB', 'words': {'cola', 'coque'}},
            {'type': 'PREP', 'words': {'con', 'in'}},
            {'type': 'LIQUID', 'words': None},
        ],
    },
    # ── Macaronic mixing templates ──
    {
        'name': 'M1_recipe_plant_con',
        'language': 'Macaronic',
        'source': 'Mixed',
        'slots': [
            {'type': 'VERB', 'words': {'recipe'}},
            {'type': 'PLANT', 'words': None},
            {'type': 'PREP', 'words': {'con', 'cum', 'e', 'et'}},
            {'type': 'LIQUID', 'words': None},
        ],
    },
    {
        'name': 'M2_bene_quality',
        'language': 'Macaronic',
        'source': 'Mixed',
        'slots': [
            {'type': 'PLANT', 'words': None},
            {'type': 'LINK', 'words': {'est', 'fa'}},
            {'type': 'QUALITY', 'words': None},
            {'type': 'PREP', 'words': {'de', 'per', 'in'}},
            {'type': 'BODY_PART', 'words': None},
        ],
    },
    {
        'name': 'M3_verb_plant_plant',
        'language': 'Macaronic',
        'source': 'Mixed',
        'slots': [
            {'type': 'VERB', 'words': None},
            {'type': 'PLANT', 'words': None},
            {'type': 'CONJ', 'words': {'et', 'e', 'cum', 'con'}},
            {'type': 'PLANT', 'words': None},
        ],
    },
]


# Medical vocabulary sets for slot matching
PHARMA_VERBS = {'cola', 'recipe', 'misce', 'coque', 'dice', 'cura',
                'sana', 'bibe', 'beni'}
BODY_PARTS = {'cora', 'core', 'corpo', 'carne', 'ossa', 'pede',
              'manu', 'dente', 'naso'}
INGREDIENTS = {'rosa', 'sale', 'vino', 'olio', 'bene', 'sene',
               'calce', 'suco'}
QUALITIES = {'bela', 'bona', 'calida', 'frigida', 'sicca',
             'dulce', 'rara', 'nova'}
LIQUIDS = {'vino', 'olio', 'suco'}
PREPS = {'de', 'in', 'cum', 'per', 'ad', 'con', 'super', 'sub'}
CONJS = {'et', 'e', 'cum', 'con'}
ARTICLES = {'la', 'le', 'lo', 'il'}
LINKS = {'est', 'fa', 'sit', 'fit'}
DEGREES = {'primo', 'secundo', 'tertio', 'quarto'}

SLOT_VOCAB = {
    'VERB': PHARMA_VERBS,
    'PLANT': INGREDIENTS,
    'LIQUID': LIQUIDS,
    'BODY_PART': BODY_PARTS,
    'QUALITY': QUALITIES,
    'PREP': PREPS,
    'CONJ': CONJS,
    'ARTICLE': ARTICLES,
    'LINK': LINKS,
    'DEGREE': DEGREES,
}


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------

def _match_word_to_slot(
    word: str,
    slot: Dict,
    classification: str,
) -> Tuple[bool, str]:
    """Check if a word matches a template slot.

    Returns (matched, match_type) where match_type is one of:
    'EXACT' – word is in the slot's explicit word set
    'TYPE'  – word belongs to the slot's type vocabulary
    'MISS'  – word is a MISS token and slot predicts a word
    'NO'    – no match
    """
    slot_type = slot['type']
    explicit_words = slot.get('words')

    # Check explicit word match
    if explicit_words is not None and word in explicit_words:
        return True, 'EXACT'

    # Check type-based match
    type_vocab = SLOT_VOCAB.get(slot_type, set())
    if word in type_vocab:
        return True, 'TYPE'

    # If word is MISS, slot could predict it
    if classification == 'MISS':
        return True, 'MISS'

    return False, 'NO'


def _try_match_phrase_to_template(
    words: List[str],
    classifications: List[str],
    template: Dict,
) -> Optional[Dict]:
    """Try to align a phrase against a template using sliding window.

    Returns match details if successful, None otherwise.
    """
    slots = template['slots']
    n_slots = len(slots)
    n_words = len(words)

    if n_words < n_slots:
        return None

    best_match = None
    best_score = 0

    # Slide template across phrase
    for offset in range(n_words - n_slots + 1):
        window_words = words[offset:offset + n_slots]
        window_cls = classifications[offset:offset + n_slots]

        slot_matches = []
        n_confirmed = 0
        n_miss_predicted = 0

        all_matched = True
        for i, (w, cls, slot) in enumerate(zip(window_words, window_cls, slots)):
            matched, match_type = _match_word_to_slot(w, slot, cls)
            if not matched:
                all_matched = False
                break
            slot_matches.append({
                'word': w,
                'classification': cls,
                'slot_type': slot['type'],
                'match_type': match_type,
                'slot_index': i,
            })
            if match_type in ('EXACT', 'TYPE'):
                n_confirmed += 1
            elif match_type == 'MISS':
                n_miss_predicted += 1

        if not all_matched:
            continue

        # Score: CONFIRMED matches are worth more
        score = n_confirmed * 2 + n_miss_predicted

        if score > best_score and n_confirmed >= 2:
            best_score = score
            best_match = {
                'template_name': template['name'],
                'template_language': template['language'],
                'template_source': template['source'],
                'offset': offset,
                'n_confirmed': n_confirmed,
                'n_miss_predicted': n_miss_predicted,
                'score': score,
                'slot_matches': slot_matches,
            }

    return best_match


def _extract_template_predictions(
    match: Dict,
) -> List[Dict]:
    """Extract predicted words for MISS slots from a template match."""
    predictions = []

    for sm in match.get('slot_matches', []):
        if sm['match_type'] != 'MISS':
            continue

        slot_type = sm['slot_type']
        predicted_words = []

        # Use explicit words from the slot if available
        # (we need to look up the template again)
        type_vocab = SLOT_VOCAB.get(slot_type, set())
        if type_vocab:
            predicted_words = sorted(type_vocab)

        predictions.append({
            'miss_word': sm['word'],
            'slot_type': slot_type,
            'slot_index': sm['slot_index'],
            'predicted_words': predicted_words,
            'template_name': match['template_name'],
        })

    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phrase_alignment() -> None:
    """Step 39.6: Phrase Template Alignment."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 39.6: Phrase Template Alignment")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    crib_data = _safe_load(os.path.join(rd, 'phrase_cribs.json'))

    phrase_annotations = crib_data.get('phrase_annotations', [])
    flanked_misses = crib_data.get('flanked_misses', [])

    print(f"     Phrases from phrase_cribs: {len(phrase_annotations)}")
    print(f"     Templates defined: {len(TEMPLATES)}")

    # ── 2. Match phrases against templates ──
    print("\n  2. Matching phrases against templates …")
    all_matches: List[Dict] = []
    phrases_with_match = 0

    for phrase in phrase_annotations:
        words = phrase.get('words', [])
        classifications = phrase.get('classifications', [])
        folio = phrase.get('folio', 'unknown')

        if not words or not classifications:
            continue

        phrase_matches = []
        for template in TEMPLATES:
            match = _try_match_phrase_to_template(
                words, classifications, template,
            )
            if match is not None:
                match['folio'] = folio
                match['phrase_words'] = words
                phrase_matches.append(match)

        if phrase_matches:
            phrases_with_match += 1
            # Keep best match per phrase
            phrase_matches.sort(key=lambda m: m['score'], reverse=True)
            all_matches.extend(phrase_matches)

    print(f"     Phrases with >=1 template match: {phrases_with_match}")
    print(f"     Total template matches: {len(all_matches)}")

    # ── 3. Summarize by template ──
    print("\n  3. Template match summary …")
    template_counts: Dict[str, int] = defaultdict(int)
    for m in all_matches:
        template_counts[m['template_name']] += 1

    for tname, count in sorted(template_counts.items(),
                                key=lambda x: -x[1]):
        print(f"       {tname}: {count} matches")

    # ── 4. Extract template-predicted corrections ──
    print("\n  4. Extracting template-predicted corrections …")
    all_predictions: List[Dict] = []

    for match in all_matches:
        predictions = _extract_template_predictions(match)
        for pred in predictions:
            pred['folio'] = match.get('folio', 'unknown')
            pred['phrase_words'] = match.get('phrase_words', [])
            pred['template_score'] = match.get('score', 0)
        all_predictions.extend(predictions)

    print(f"     Template-predicted MISS corrections: {len(all_predictions)}")

    # Group by miss_word for summary
    miss_word_predictions: Dict[str, List[str]] = defaultdict(list)
    for pred in all_predictions:
        miss_word_predictions[pred['miss_word']].extend(
            pred['predicted_words'][:5])

    for mw, preds in sorted(miss_word_predictions.items(),
                             key=lambda x: -len(x[1]))[:10]:
        unique_preds = sorted(set(preds))[:5]
        print(f"       '{mw}' → predicted: {unique_preds}")

    # ── 5. Build template-predicted corrections list ──
    print("\n  5. Building correction list …")
    template_corrections: List[Dict] = []

    for pred in all_predictions:
        for pw in pred.get('predicted_words', [])[:5]:
            template_corrections.append({
                'miss_word': pred['miss_word'],
                'predicted_word': pw,
                'slot_type': pred['slot_type'],
                'template_name': pred['template_name'],
                'folio': pred.get('folio', 'unknown'),
                'template_score': pred.get('template_score', 0),
            })

    # Deduplicate
    seen: Set[Tuple[str, str]] = set()
    unique_corrections: List[Dict] = []
    for tc in template_corrections:
        key = (tc['miss_word'], tc['predicted_word'])
        if key not in seen:
            seen.add(key)
            unique_corrections.append(tc)

    print(f"     Unique template-predicted corrections: {len(unique_corrections)}")

    # ── 6. Match detail examples ──
    print("\n  6. Top match details …")
    top_matches = sorted(all_matches, key=lambda m: m['score'], reverse=True)[:10]
    match_details: List[Dict] = []

    for m in top_matches:
        detail = {
            'template_name': m['template_name'],
            'language': m['template_language'],
            'folio': m.get('folio', 'unknown'),
            'phrase_words': m.get('phrase_words', []),
            'score': m['score'],
            'n_confirmed': m['n_confirmed'],
            'n_miss_predicted': m['n_miss_predicted'],
            'slot_matches': m.get('slot_matches', []),
        }
        match_details.append(detail)
        print(f"       [{m['template_name']}] {m.get('folio', '?')}: "
              f"score={m['score']} confirmed={m['n_confirmed']} "
              f"miss={m['n_miss_predicted']}")
        print(f"         Words: {' '.join(m.get('phrase_words', [])[:8])}")

    # ── 7. Save ──
    elapsed = time.time() - t0

    verdict_parts = [
        f"{len(TEMPLATES)} templates tested",
        f"{phrases_with_match}/{len(phrase_annotations)} phrases matched",
        f"{len(all_matches)} total matches",
        f"{len(unique_corrections)} template-predicted corrections",
    ]

    output = {
        'n_templates': len(TEMPLATES),
        'n_phrases_tested': len(phrase_annotations),
        'n_template_matches': len(all_matches),
        'template_match_details': match_details,
        'template_match_counts': dict(template_counts),
        'template_predicted_corrections': unique_corrections[:200],
        'all_predictions': all_predictions[:200],
        'n_predictions': len(all_predictions),
        'n_unique_corrections': len(unique_corrections),
        'verdict': '. '.join(verdict_parts) + '.',
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'phrase_alignment.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
