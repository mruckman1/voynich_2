"""
Phase 70, Track 3: Phrase Fragment Assembly
============================================
Gloss the 888 sequential proximity pairs from Phase 69, classify them
syntactically, extend to trigrams, and check against pharmaceutical formulae.

Re-scans the corpus for ORDERED pairs (Phase 69 stored sorted pairs,
losing word order; word order matters for syntactic classification).

Dependency chain:
    results/p69_t1_network.json          (Track 4: proximity pair types)
    results/p69_clean_corpus.json        (Step 0: t1_catalogue)
    results/phase70_pharma_dict.json     (Track 1: expanded dict — optional)
    results/combined_refine.json         (Phase 15: best_assignment)
        -> results/phase70_phrases.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import (
    LATIN_PHRASE_PATTERNS,
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.corrected_coda import build_coda_table_v2, decode_token_cvc_v2
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51
from voynich.phases.suffix_grammar import _classify_latin_ending


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Pharmaceutical word sets for classification
# ---------------------------------------------------------------------------

_VERB_SET = set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('verbs', []))
_INGREDIENT_SET = (
    set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('plant_parts', []))
    | set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('preparations', []))
)
_QUALITY_SET = set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('qualities', []))
_BODY_SET = set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('body_parts', []))

# CI recipe templates — verb + typical object patterns
_CI_TEMPLATES = [
    {'name': 'recipe_ingredient', 'verb_words': {'recipe', 'accipe'},
     'requires': 'ingredient', 'description': 'Take [ingredient]'},
    {'name': 'strain_substance', 'verb_words': {'cola', 'colare', 'colat'},
     'requires': 'any', 'description': 'Strain [substance]'},
    {'name': 'grind_substance', 'verb_words': {'tere', 'terere', 'contere'},
     'requires': 'any', 'description': 'Grind [substance]'},
    {'name': 'mix_substances', 'verb_words': {'misce', 'miscere'},
     'requires': 'any', 'description': 'Mix [substances]'},
    {'name': 'cook_substance', 'verb_words': {'coque', 'coquere'},
     'requires': 'any', 'description': 'Cook [substance]'},
    {'name': 'add_substance', 'verb_words': {'adde', 'addere'},
     'requires': 'any', 'description': 'Add [substance]'},
    {'name': 'dissolve_substance', 'verb_words': {'solve', 'solvere'},
     'requires': 'any', 'description': 'Dissolve [substance]'},
    {'name': 'prep_with_medium', 'verb_words': {'cum', 'in', 'per'},
     'requires': 'any', 'description': 'With/in [medium]'},
    {'name': 'quality_statement', 'verb_words': {'est', 'sit'},
     'requires': 'quality', 'description': 'Is [quality]'},
]


def _build_gloss_lookup(
    t1_catalogue: List[Dict],
    expanded_dict: Set[str],
) -> Dict[str, Dict[str, str]]:
    """Build master gloss lookup: decoded_word → {gloss, class, source}.

    Priority: T1 > signal_word > pharmaceutical_vocab > dict_only.
    """
    lookup: Dict[str, Dict[str, str]] = {}

    # Layer 1: T1 catalogue
    for entry in t1_catalogue:
        w = entry.get('matched_word', '')
        if w:
            lookup[w] = {
                'gloss': w,  # T1 doesn't always have a gloss field
                'class': 'T1',
                'source': 'T1_catalogue',
            }

    # Layer 2: Signal words (have explicit glosses)
    for word, info in SIGNAL_WORDS_51.items():
        if word not in lookup:
            lookup[word] = {
                'gloss': info.get('gloss', word),
                'class': info.get('type', 'signal'),
                'source': 'signal',
            }
        else:
            # Upgrade T1 entry with signal gloss
            lookup[word]['gloss'] = info.get('gloss', lookup[word]['gloss'])
            lookup[word]['class'] = info.get('type', lookup[word]['class'])

    # Layer 3: Pharmaceutical vocabulary (known category)
    for category, words in PHARMACEUTICAL_VOCABULARY.items():
        for w in words:
            wl = w.lower()
            if wl not in lookup:
                lookup[wl] = {
                    'gloss': wl,
                    'class': category,
                    'source': 'pharma_vocab',
                }

    # Layer 4: Expanded dict (only mark as 'dict' if in dictionary)
    for w in expanded_dict:
        if w not in lookup:
            lookup[w] = {
                'gloss': w,
                'class': 'latin',
                'source': 'dict',
            }

    return lookup


def _classify_pair_syntax(
    gloss_a: Dict[str, str],
    gloss_b: Dict[str, str],
    word_a: str,
    word_b: str,
) -> str:
    """Classify a glossed pair by syntactic relationship."""
    class_a = gloss_a.get('class', '?')
    class_b = gloss_b.get('class', '?')

    # Check POS from Latin ending analysis
    pos_a, _ = _classify_latin_ending(word_a)
    pos_b, _ = _classify_latin_ending(word_b)

    # Verb + anything
    if class_a == 'verbs' or pos_a == 'VERB' or word_a in _VERB_SET:
        if class_b in ('plant_parts', 'preparations', 'body_parts', 'T1', 'latin'):
            return 'VERB_OBJECT'
        return 'VERB_OTHER'

    # Preposition/function + noun
    if class_a in ('function', 'function_words') or pos_a == 'PARTICLE':
        if pos_b in ('NOUN', 'VERB') or class_b in ('T1', 'latin', 'plant_parts',
                                                      'preparations', 'body_parts'):
            return 'PREP_NOUN'
        return 'FUNCTION_OTHER'

    # Noun + noun
    if (class_a in ('plant_parts', 'preparations', 'body_parts', 'T1') and
            class_b in ('plant_parts', 'preparations', 'body_parts', 'T1')):
        return 'NOUN_NOUN'

    # Quality + noun
    if (class_a in ('qualities', 'quality') and
            class_b in ('plant_parts', 'preparations', 'body_parts', 'T1', 'latin')):
        return 'ADJ_NOUN'

    # Noun + function
    if (class_a in ('plant_parts', 'preparations', 'body_parts', 'T1', 'latin') and
            (class_b in ('function', 'function_words') or pos_b == 'PARTICLE')):
        return 'NOUN_PREP'

    return f'{class_a}_{class_b}'


def _rescan_ordered_pairs(
    all_tokens: List[str],
    t1_types: Set[str],
    cvc_decoded: List[str],
    window: int = 5,
    min_count: int = 3,
) -> List[Dict[str, Any]]:
    """Re-scan corpus for ORDERED proximity pairs.

    Unlike Phase 69 which stored sorted pairs, this preserves word order.
    """
    pair_counter: Counter = Counter()
    pair_examples: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for idx in range(len(all_tokens)):
        token_a = all_tokens[idx]
        decoded_a = cvc_decoded[idx] if idx < len(cvc_decoded) else ''
        if token_a not in t1_types or not decoded_a:
            continue

        for offset in range(1, window + 1):
            j = idx + offset
            if j >= len(all_tokens):
                break
            token_b = all_tokens[j]
            decoded_b = cvc_decoded[j] if j < len(cvc_decoded) else ''
            if token_b not in t1_types or not decoded_b:
                continue

            pair = (decoded_a, decoded_b)
            pair_counter[pair] += 1
            if len(pair_examples[pair]) < 3:
                pair_examples[pair].append(idx)

    # Filter by min_count and sort
    ordered_pairs = []
    for (word_a, word_b), count in pair_counter.most_common():
        if count < min_count:
            break
        ordered_pairs.append({
            'word_a': word_a,
            'word_b': word_b,
            'count': count,
            'example_positions': pair_examples[(word_a, word_b)],
        })

    return ordered_pairs


def _gloss_proximity_pairs(
    ordered_pairs: List[Dict],
    gloss_lookup: Dict[str, Dict[str, str]],
    top_n: int = 200,
) -> List[Dict[str, Any]]:
    """Annotate top proximity pairs with glosses and classify syntax."""
    glossed = []

    for pair in ordered_pairs[:top_n]:
        word_a = pair['word_a']
        word_b = pair['word_b']

        ga = gloss_lookup.get(word_a, {'gloss': '?', 'class': '?', 'source': None})
        gb = gloss_lookup.get(word_b, {'gloss': '?', 'class': '?', 'source': None})

        both_glossed = ga['gloss'] != '?' and gb['gloss'] != '?'
        # "dict" source still counts as glossed (we know it's a Latin word)
        both_in_dict = word_a in gloss_lookup and word_b in gloss_lookup

        pair_type = _classify_pair_syntax(ga, gb, word_a, word_b)

        # Check against CI templates
        formula_match = None
        for template in _CI_TEMPLATES:
            if (word_a in template['verb_words'] or word_b in template['verb_words']):
                formula_match = template['name']
                break

        glossed.append({
            'word_a': word_a,
            'word_b': word_b,
            'gloss_a': ga['gloss'],
            'gloss_b': gb['gloss'],
            'class_a': ga['class'],
            'class_b': gb['class'],
            'pair_type': pair_type,
            'both_glossed': both_in_dict,
            'count': pair['count'],
            'formula_match': formula_match,
            'reading': f"{ga['gloss']} + {gb['gloss']}" if both_in_dict else '?',
        })

    return glossed


def _extend_to_trigrams(
    all_tokens: List[str],
    cvc_decoded: List[str],
    gloss_lookup: Dict[str, Dict[str, str]],
    t1_types: Set[str],
    top_n: int = 50,
) -> Dict[str, Any]:
    """Find 3-token sequences where all 3 are T1/signal words."""
    trigram_counter: Counter = Counter()
    trigram_glosses: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {}

    for i in range(len(all_tokens) - 2):
        if all_tokens[i] not in t1_types:
            continue
        if all_tokens[i + 1] not in t1_types:
            continue
        if all_tokens[i + 2] not in t1_types:
            continue

        d1 = cvc_decoded[i] if i < len(cvc_decoded) else ''
        d2 = cvc_decoded[i + 1] if i + 1 < len(cvc_decoded) else ''
        d3 = cvc_decoded[i + 2] if i + 2 < len(cvc_decoded) else ''

        if not all([d1, d2, d3]):
            continue

        trigram = (d1, d2, d3)
        trigram_counter[trigram] += 1

        if trigram not in trigram_glosses:
            g1 = gloss_lookup.get(d1, {}).get('gloss', '?')
            g2 = gloss_lookup.get(d2, {}).get('gloss', '?')
            g3 = gloss_lookup.get(d3, {}).get('gloss', '?')
            trigram_glosses[trigram] = (g1, g2, g3)

    top_trigrams = []
    for trigram, count in trigram_counter.most_common(top_n):
        glosses = trigram_glosses.get(trigram, ('?', '?', '?'))
        all_glossed = all(g != '?' for g in glosses)

        top_trigrams.append({
            'words': list(trigram),
            'glosses': list(glosses),
            'count': count,
            'all_glossed': all_glossed,
            'reading': ' '.join(glosses) if all_glossed else '?',
        })

    n_fully_glossed = sum(1 for t in top_trigrams if t['all_glossed'])

    return {
        'n_unique_trigrams': len(trigram_counter),
        'n_total_occurrences': sum(trigram_counter.values()),
        'n_fully_glossed': n_fully_glossed,
        'top_trigrams': top_trigrams,
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PhraseAssemblyResult:
    phase: str = "70"
    step: str = "70.3"
    experiment: str = "phrase_assembly"
    # Pair stats
    n_ordered_pairs: int = 0
    n_pairs_glossed: int = 0
    glossed_fraction: float = 0.0
    pair_type_distribution: Dict[str, int] = field(default_factory=dict)
    n_verb_object: int = 0
    n_prep_noun: int = 0
    n_formula_matches: int = 0
    top_pairs: List[Dict] = field(default_factory=list)
    # Trigram stats
    n_unique_trigrams: int = 0
    n_fully_glossed_trigrams: int = 0
    top_trigrams: List[Dict] = field(default_factory=list)
    # Gates
    gate_ph1: bool = False  # >= 40% of top-200 glossed
    gate_ph2: bool = False  # >= 10 VERB_OBJECT or PREP_NOUN pairs
    gate_ph3: bool = False  # >= 3 CI formula matches
    gate_ph4: bool = False  # >= 20 fully-glossed trigrams
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phrase_assemble():
    """Track 3: Gloss proximity pairs, classify syntax, find trigrams."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 70.3 — Phrase Fragment Assembly")
    print("=" * 38)

    # --- Load dependencies ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])
    t1_types = set(entry['eva_type'] for entry in t1_catalogue if entry.get('eva_type'))
    print(f"  T1 types: {len(t1_types)}")

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Load expanded dict from Track 1 if available, else build base
    pharma_data = _safe_load(os.path.join(rd, 'phase70_pharma_dict.json'))
    if pharma_data.get('combined_word_list'):
        expanded_dict = set(pharma_data['combined_word_list'])
        print(f"  Using Track 1 expanded dict: {len(expanded_dict)} words")
    else:
        print("  Track 1 not available; building base expanded dict...")
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                         if len(w) >= 2)
        expanded_dict, _ = build_expanded_word_set(base_words)
        expanded_dict = base_words | expanded_dict
        print(f"  Base expanded dict: {len(expanded_dict)} words")

    # --- Decode all tokens (CVC) ---
    print("\n  Decoding corpus (CVC)...")
    cvc_decoded = []
    for token in all_tokens:
        try:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
            cvc_decoded.append(result.decoded_cvc)
        except Exception:
            cvc_decoded.append('')
    print(f"    Decoded {len(cvc_decoded)} tokens")

    # --- Build gloss lookup ---
    print("  Building gloss lookup...")
    gloss_lookup = _build_gloss_lookup(t1_catalogue, expanded_dict)
    print(f"    Gloss entries: {len(gloss_lookup)}")

    # --- Step 3.1: Re-scan for ordered pairs ---
    print("\n  Scanning for ordered proximity pairs...")
    ordered_pairs = _rescan_ordered_pairs(
        all_tokens, t1_types, cvc_decoded, window=5, min_count=3)
    print(f"    Ordered pairs (count ≥ 3): {len(ordered_pairs)}")

    # --- Step 3.2: Gloss and classify pairs ---
    print("  Glossing top 200 pairs...")
    glossed_pairs = _gloss_proximity_pairs(ordered_pairs, gloss_lookup, top_n=200)

    n_both_glossed = sum(1 for p in glossed_pairs if p['both_glossed'])
    glossed_fraction = n_both_glossed / len(glossed_pairs) if glossed_pairs else 0.0

    type_dist = Counter(p['pair_type'] for p in glossed_pairs)
    n_vo = type_dist.get('VERB_OBJECT', 0)
    n_pn = type_dist.get('PREP_NOUN', 0)
    n_formula = sum(1 for p in glossed_pairs if p.get('formula_match'))

    print(f"    Both glossed: {n_both_glossed}/{len(glossed_pairs)} ({glossed_fraction:.0%})")
    print(f"    VERB_OBJECT: {n_vo}, PREP_NOUN: {n_pn}")
    print(f"    Formula matches: {n_formula}")
    print(f"    Type distribution: {dict(type_dist.most_common(8))}")

    # Show top 10 glossed pairs
    print("\n    Top glossed pairs:")
    for p in glossed_pairs[:10]:
        tag = f" [{p['formula_match']}]" if p['formula_match'] else ""
        print(f"      {p['word_a']} + {p['word_b']} → "
              f"{p['reading']} ({p['pair_type']}, n={p['count']}){tag}")

    # --- Step 3.3: Extend to trigrams ---
    print("\n  Finding trigram phrases...")
    trigram_results = _extend_to_trigrams(
        all_tokens, cvc_decoded, gloss_lookup, t1_types, top_n=50)
    print(f"    Unique trigrams: {trigram_results['n_unique_trigrams']}")
    print(f"    Fully glossed: {trigram_results['n_fully_glossed']}")

    for t in trigram_results['top_trigrams'][:10]:
        print(f"      {' '.join(t['words'])} → {t['reading']} (n={t['count']})")

    # --- Gates ---
    g1 = glossed_fraction >= 0.40
    g2 = (n_vo + n_pn) >= 10
    g3 = n_formula >= 3
    g4 = trigram_results['n_fully_glossed'] >= 20

    gates_passed = sum([g1, g2, g3, g4])

    print(f"\n  Gates: {gates_passed}/4")
    print(f"    PH1 (≥40% glossed): {'PASS' if g1 else 'FAIL'} ({glossed_fraction:.0%})")
    print(f"    PH2 (≥10 VERB_OBJ/PREP_NOUN): {'PASS' if g2 else 'FAIL'} ({n_vo + n_pn})")
    print(f"    PH3 (≥3 formula matches): {'PASS' if g3 else 'FAIL'} ({n_formula})")
    print(f"    PH4 (≥20 glossed trigrams): {'PASS' if g4 else 'FAIL'} ({trigram_results['n_fully_glossed']})")

    if gates_passed >= 3:
        verdict = 'PHRASES_FOUND'
    elif gates_passed >= 1:
        verdict = 'PARTIAL_PHRASES'
    else:
        verdict = 'NO_PHRASES'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = PhraseAssemblyResult(
        n_ordered_pairs=len(ordered_pairs),
        n_pairs_glossed=n_both_glossed,
        glossed_fraction=glossed_fraction,
        pair_type_distribution=dict(type_dist.most_common()),
        n_verb_object=n_vo,
        n_prep_noun=n_pn,
        n_formula_matches=n_formula,
        top_pairs=glossed_pairs[:50],
        n_unique_trigrams=trigram_results['n_unique_trigrams'],
        n_fully_glossed_trigrams=trigram_results['n_fully_glossed'],
        top_trigrams=trigram_results['top_trigrams'],
        gate_ph1=g1,
        gate_ph2=g2,
        gate_ph3=g3,
        gate_ph4=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out = _save_json(rd, 'phase70_phrases.json', asdict(result))
    print(f"\n  Saved: {out}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
