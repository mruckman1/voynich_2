"""
Phase 71, Track 3: Grammatically-Annotated Passage Reading
==========================================================
Combine Track 1 (inflectional catalog) and Track 2 (root dictionary) to
produce passage readings where every token has both a grammatical label
and (where possible) a lexical identification.

6-layer annotation:
  Layer 1: EVA token
  Layer 2: CVC decoded
  Layer 3: Root (from root dictionary)
  Layer 4: Suffix / coda
  Layer 5: Grammar label (from inflectional catalog)
  Layer 6: Lexical gloss (T1 > signal > dict > root > ?)

Dependency chain:
    results/phase71_inflectional_catalog.json  (Track 1)
    results/phase71_root_identification.json   (Track 2)
    results/combined_refine.json               (Phase 15)
    results/p69_clean_corpus.json              (T1 catalogue)
    results/phase70_pharma_dict.json           (expanded dict — optional)
        -> results/phase71_grammatical_reading.json
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import (
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
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
# CI grammatical templates
# ---------------------------------------------------------------------------

_GRAMMATICAL_TEMPLATES = [
    {
        'name': 'recipe_instruction',
        'pattern': ['VERBAL', 'NOMINAL'],
        'description': 'VERB + NOUN_ACC: process ingredient',
        'example': 'cola sennam (strain senna)',
    },
    {
        'name': 'property_description',
        'pattern': ['UNMARKED', 'VERBAL', 'UNMARKED'],
        'description': 'NOUN + VERB_3SG + NOUN: X has/is Y',
        'example': 'senna valet ad stomachum',
    },
    {
        'name': 'passive_instruction',
        'pattern': ['VERBAL', 'NOMINAL'],
        'description': 'PASSIVE + NOUN: let X be processed',
        'example': 'colatur senna (let senna be strained)',
    },
    {
        'name': 'ingredient_list',
        'pattern': ['UNMARKED', 'UNMARKED', 'UNMARKED'],
        'description': 'NOUN + NOUN + NOUN: ingredient sequence',
        'example': 'rosa senna cera',
    },
    {
        'name': 'prep_phrase',
        'pattern': ['FUNCTION_STEM', 'NOMINAL'],
        'description': 'PREP + NOUN_ACC/ABL: prepositional phrase',
        'example': 'cum aqua (with water)',
    },
]

# Pharma verbs and ingredients for interpretation
_PHARMA_VERBS = set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('verbs', []))
_INGREDIENTS = (
    set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('plant_parts', []))
    | set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('preparations', []))
)
_QUALITIES = set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('qualities', []))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_folio_list(corpus) -> List[str]:
    folios: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folios.append(folio)
    return folios


def _build_section_list(corpus) -> List[str]:
    sections: List[str] = []
    for _folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            sections.append(getattr(page, 'section', 'unknown'))
    return sections


def _build_master_gloss(
    t1_catalogue: List[Dict],
    root_dict_sample: List[Dict],
) -> Dict[str, Dict[str, str]]:
    """Build master gloss lookup combining all sources."""
    lookup: Dict[str, Dict[str, str]] = {}

    # Signal words (highest priority for glosses)
    for word, info in SIGNAL_WORDS_51.items():
        lookup[word] = {
            'gloss': info.get('gloss', word),
            'class': info.get('type', 'signal'),
            'source': 'signal',
        }

    # T1 catalogue
    for entry in t1_catalogue:
        w = entry.get('matched_word', '')
        if w and w not in lookup:
            lookup[w] = {
                'gloss': w,
                'class': 'T1',
                'source': 'T1_catalogue',
            }

    # Root dictionary paradigm forms
    for root_entry in root_dict_sample:
        meaning = root_entry.get('meaning', '?')
        if meaning == '?':
            continue
        for form, suffix in root_entry.get('example_forms', {}).items():
            if form not in lookup:
                suffix_str = f" ({suffix})" if suffix and suffix != '∅' else ""
                lookup[form] = {
                    'gloss': f"{meaning}{suffix_str}",
                    'class': 'paradigm',
                    'source': 'root_dict',
                }

    # Pharmaceutical vocabulary
    for category, words in PHARMACEUTICAL_VOCABULARY.items():
        for w in words:
            wl = w.lower()
            if wl not in lookup:
                lookup[wl] = {
                    'gloss': wl,
                    'class': category,
                    'source': 'pharma_vocab',
                }

    return lookup


def _select_passages(
    all_tokens: List[str],
    cvc_decoded: List[str],
    folio_list: List[str],
    section_list: List[str],
    gram_categories: List[str],
    clean_indices: Set[int],
    expanded_dict: Set[str],
    gloss_lookup: Dict[str, Dict[str, str]],
    n: int = 20,
    window: int = 15,
) -> List[Dict[str, Any]]:
    """Select top passages scored by grammatical + lexical coverage."""
    windows = []

    for start in range(len(all_tokens) - window):
        if folio_list[start] != folio_list[start + window - 1]:
            continue

        n_gram = sum(1 for i in range(start, start + window)
                     if gram_categories[i] not in ('UNMARKED', 'MULTI_CODA', 'DOUBLE_CODA'))
        n_lex = sum(1 for i in range(start, start + window)
                    if cvc_decoded[i] and (
                        cvc_decoded[i] in gloss_lookup
                        or cvc_decoded[i] in expanded_dict))
        n_clean = sum(1 for i in range(start, start + window)
                      if i in clean_indices)

        section = section_list[start] if start < len(section_list) else 'unknown'
        section_bonus = (1.0 if section == 'pharmaceutical' else
                         0.5 if section in ('herbal_a', 'herbal_b') else 0.0)

        score = (
            2.0 * n_gram / window +
            3.0 * n_lex / window +
            1.0 * n_clean / window +
            section_bonus
        )

        windows.append({
            'start': start,
            'end': start + window - 1,
            'folio': folio_list[start],
            'section': section,
            'n_gram': n_gram,
            'n_lex': n_lex,
            'n_clean': n_clean,
            'score': score,
        })

    windows.sort(key=lambda w: -w['score'])

    selected = []
    used = set()
    for w in windows:
        positions = set(range(w['start'], w['end'] + 1))
        if not positions & used:
            selected.append(w)
            used.update(positions)
        if len(selected) >= n:
            break

    return selected


def _annotate_passage(
    window: Dict,
    all_tokens: List[str],
    cvc_decoded: List[str],
    gram_categories: List[str],
    gram_functions: List[str],
    gram_codas: List[List[str]],
    clean_indices: Set[int],
    gloss_lookup: Dict[str, Dict[str, str]],
    expanded_dict: Set[str],
    root_lookup: Dict[str, Dict],
) -> List[Dict[str, Any]]:
    """Produce 6-layer annotation for each token in a passage."""
    annotated = []

    for idx in range(window['start'], window['end'] + 1):
        eva = all_tokens[idx]
        decoded = cvc_decoded[idx] if idx < len(cvc_decoded) else ''
        is_clean = idx in clean_indices
        gram_cat = gram_categories[idx] if idx < len(gram_categories) else 'UNKNOWN'
        gram_func = gram_functions[idx] if idx < len(gram_functions) else 'UNKNOWN'
        codas = gram_codas[idx] if idx < len(gram_codas) else []

        # Layer 3: Root
        root_info = None
        root_str = '?'
        suffix_str = '?'
        if decoded:
            for root_entry in root_lookup.values():
                root = root_entry.get('root', '')
                if decoded.startswith(root) and len(decoded) >= len(root):
                    root_info = root_entry
                    root_str = root
                    suffix_str = decoded[len(root):] if len(decoded) > len(root) else '∅'
                    break

        # Layer 4: Suffix / coda
        coda_str = ','.join(codas) if codas else '∅'

        # Layer 6: Gloss
        gloss_info = gloss_lookup.get(decoded, {})
        if gloss_info:
            gloss = gloss_info.get('gloss', decoded)
            gloss_source = gloss_info.get('source', 'unknown')
        elif decoded and decoded in expanded_dict:
            gloss = decoded
            gloss_source = 'dict'
        elif root_info and root_info.get('meaning', '?') != '?':
            gloss = root_info['meaning']
            gloss_source = 'root'
        elif decoded:
            gloss = f'[{decoded}]'
            gloss_source = 'decoded'
        else:
            gloss = '?'
            gloss_source = 'unknown'

        annotated.append({
            'position': idx,
            'layer_1_eva': eva,
            'layer_2_decoded': decoded,
            'layer_3_root': root_str,
            'layer_3_root_meaning': root_info.get('meaning', '?') if root_info else '?',
            'layer_4_suffix': suffix_str,
            'layer_4_codas': coda_str,
            'layer_5_grammar': gram_func,
            'layer_5_category': gram_cat,
            'layer_6_gloss': gloss,
            'layer_6_source': gloss_source,
            'is_clean': is_clean,
        })

    return annotated


def _match_templates(
    annotated: List[Dict],
) -> List[Dict[str, Any]]:
    """Match grammatical sequence against CI templates."""
    categories = [tok['layer_5_category'] for tok in annotated]
    matches = []

    for template in _GRAMMATICAL_TEMPLATES:
        pattern = template['pattern']
        plen = len(pattern)

        # Slide pattern across the category sequence
        n_matches = 0
        match_positions = []
        for i in range(len(categories) - plen + 1):
            window = categories[i:i + plen]
            if window == pattern:
                n_matches += 1
                match_positions.append(i)

        if n_matches > 0:
            score = n_matches / (len(categories) - plen + 1) if len(categories) > plen else 0
            matches.append({
                'template': template['name'],
                'description': template['description'],
                'n_matches': n_matches,
                'score': min(1.0, score * 5),  # Normalize
                'positions': match_positions[:5],
            })

    return sorted(matches, key=lambda m: -m['score'])


def _format_structural_reading(annotated: List[Dict]) -> str:
    """GRAMMAR(gloss) format."""
    parts = []
    for tok in annotated:
        gram = tok['layer_5_grammar']
        gloss = tok['layer_6_gloss']
        parts.append(f"{gram}({gloss})")
    return ' '.join(parts)


def _format_natural_reading(annotated: List[Dict]) -> str:
    """Human-readable format."""
    parts = []
    for tok in annotated:
        source = tok['layer_6_source']
        gloss = tok['layer_6_gloss']
        gram = tok['layer_5_grammar']

        if source in ('signal', 'T1_catalogue', 'root_dict', 'pharma_vocab', 'dict'):
            parts.append(gloss)
        elif gram.startswith('VERB'):
            parts.append(f"[?verb, {gram.split('_')[-1] if '_' in gram else '?'}]")
        elif gram.startswith('NOUN'):
            parts.append(f"[?noun, acc]")
        elif gram == 'FUNCTION_OR_SHORT_STEM':
            parts.append("[func?]")
        else:
            parts.append(f"[{tok['layer_2_decoded']}]")

    return ' · '.join(parts)


def _build_interpretation(annotated: List[Dict]) -> Optional[str]:
    """Attempt pharmaceutical interpretation."""
    verbs = [t for t in annotated if t['layer_2_decoded'] in _PHARMA_VERBS]
    ingredients = [t for t in annotated if t['layer_2_decoded'] in _INGREDIENTS]
    qualities = [t for t in annotated if t['layer_2_decoded'] in _QUALITIES]

    parts = []
    if verbs:
        parts.append('/'.join(set(t['layer_6_gloss'] for t in verbs)))
    if ingredients:
        parts.append(' + '.join(set(t['layer_6_gloss'] for t in ingredients)))
    if qualities:
        parts.append(f"({', '.join(set(t['layer_6_gloss'] for t in qualities))})")

    return ' '.join(parts) if parts else None


def _run_null_controls(
    all_tokens: List[str],
    cvc_decoded: List[str],
    folio_list: List[str],
    gram_categories: List[str],
    gram_functions: List[str],
    gram_codas: List[List[str]],
    clean_indices: Set[int],
    gloss_lookup: Dict[str, Dict[str, str]],
    expanded_dict: Set[str],
    root_lookup: Dict[str, Dict],
    real_passages: List[Dict],
    n_random: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Null controls: random passages + shuffled grammar."""
    rng = random.Random(seed)

    # Real passage metrics
    real_gram_fracs = []
    real_lex_fracs = []
    real_template_counts = []
    for p in real_passages:
        tokens = p.get('tokens', [])
        n = len(tokens)
        if n == 0:
            continue
        n_gram = sum(1 for t in tokens
                     if t['layer_5_category'] not in ('UNMARKED', 'MULTI_CODA', 'DOUBLE_CODA'))
        n_lex = sum(1 for t in tokens
                    if t['layer_6_source'] in ('signal', 'T1_catalogue', 'root_dict',
                                                'pharma_vocab', 'dict'))
        real_gram_fracs.append(n_gram / n)
        real_lex_fracs.append(n_lex / n)
        real_template_counts.append(len(p.get('template_matches', [])))

    # Random passages
    all_starts = list(range(len(all_tokens) - 15))
    rng.shuffle(all_starts)

    random_gram_fracs = []
    random_lex_fracs = []
    random_template_counts = []
    used = set()

    for start in all_starts:
        if len(random_gram_fracs) >= n_random:
            break
        end = start + 14
        if end >= len(all_tokens):
            continue
        if folio_list[start] != folio_list[end]:
            continue
        positions = set(range(start, end + 1))
        if positions & used:
            continue
        used.update(positions)

        window = {'start': start, 'end': end, 'folio': folio_list[start]}
        ann = _annotate_passage(
            window, all_tokens, cvc_decoded, gram_categories,
            gram_functions, gram_codas, clean_indices,
            gloss_lookup, expanded_dict, root_lookup)

        n = len(ann)
        n_gram = sum(1 for t in ann
                     if t['layer_5_category'] not in ('UNMARKED', 'MULTI_CODA', 'DOUBLE_CODA'))
        n_lex = sum(1 for t in ann
                    if t['layer_6_source'] in ('signal', 'T1_catalogue', 'root_dict',
                                                'pharma_vocab', 'dict'))
        random_gram_fracs.append(n_gram / n if n > 0 else 0)
        random_lex_fracs.append(n_lex / n if n > 0 else 0)
        templates = _match_templates(ann)
        random_template_counts.append(len(templates))

    real_gram_mean = float(np.mean(real_gram_fracs)) if real_gram_fracs else 0.0
    random_gram_mean = float(np.mean(random_gram_fracs)) if random_gram_fracs else 0.0
    real_lex_mean = float(np.mean(real_lex_fracs)) if real_lex_fracs else 0.0
    random_lex_mean = float(np.mean(random_lex_fracs)) if random_lex_fracs else 0.0
    real_template_mean = float(np.mean(real_template_counts)) if real_template_counts else 0.0
    random_template_mean = float(np.mean(random_template_counts)) if random_template_counts else 0.0

    template_selectivity = (real_template_mean / random_template_mean
                            if random_template_mean > 0 else float('inf'))
    lex_selectivity = (real_lex_mean / random_lex_mean
                       if random_lex_mean > 0 else float('inf'))

    return {
        'real_gram_mean': real_gram_mean,
        'random_gram_mean': random_gram_mean,
        'real_lex_mean': real_lex_mean,
        'random_lex_mean': random_lex_mean,
        'lex_selectivity': lex_selectivity,
        'real_template_mean': real_template_mean,
        'random_template_mean': random_template_mean,
        'template_selectivity': template_selectivity,
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GrammaticalReadingResult:
    phase: str = "71"
    step: str = "71.3"
    experiment: str = "grammatical_reading"
    # Passage stats
    n_passages: int = 0
    mean_grammatical_fraction: float = 0.0
    mean_identified_fraction: float = 0.0
    n_high_grammatical: int = 0   # gram_fraction > 70%
    n_high_identified: int = 0    # identified > 50%
    n_template_matches: int = 0
    n_interpretable: int = 0
    # Null controls
    null_controls: Dict[str, Any] = field(default_factory=dict)
    # Passages
    passages: List[Dict] = field(default_factory=list)
    # Gates
    gate_g1: bool = False  # >= 10 passages with gram > 70%
    gate_g2: bool = False  # >= 5 passages with identified > 50%
    gate_g3: bool = False  # >= 3 passages with template matches > 0.4
    gate_g4: bool = False  # template selectivity > 1.3x
    gate_g5: bool = False  # lex selectivity > 1.5x
    gate_g6: bool = False  # >= 1 passage pharmaceutically interpretable
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_gram_read():
    """Track 3: Grammatically-annotated passage reading."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 71.3 — Grammatically-Annotated Passage Reading")
    print("=" * 53)

    # --- Load dependencies ---
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])
    clean_indices = set(clean_data.get('clean_indices', []))
    print(f"  T1 catalogue: {len(t1_catalogue)}, Clean: {len(clean_indices)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folio_list = _build_folio_list(corpus)
    section_list = _build_section_list(corpus)
    print(f"  Tokens: {len(all_tokens)}")

    # Expanded dict
    pharma_data = _safe_load(os.path.join(rd, 'phase70_pharma_dict.json'))
    if pharma_data.get('combined_word_list'):
        expanded_dict = set(pharma_data['combined_word_list'])
    else:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                         if len(w) >= 2)
        expanded_dict, _ = build_expanded_word_set(base_words)
        expanded_dict = base_words | expanded_dict

    # Track 1: inflectional catalog
    t1_data = _safe_load(os.path.join(rd, 'phase71_inflectional_catalog.json'))
    if not t1_data:
        print("  WARNING: Track 1 not available — running inline classification")

    # Track 2: root dictionary
    t2_data = _safe_load(os.path.join(rd, 'phase71_root_identification.json'))
    root_dict_sample = t2_data.get('root_dictionary_sample', [])
    root_lookup = {e['root']: e for e in root_dict_sample if e.get('root')}
    print(f"  Root dictionary: {len(root_lookup)} roots")

    # Build master gloss
    gloss_lookup = _build_master_gloss(t1_catalogue, root_dict_sample)
    print(f"  Gloss entries: {len(gloss_lookup)}")

    # --- Decode corpus ---
    print("\n  Decoding corpus (CVC)...")
    cvc_decoded = []
    for token in all_tokens:
        try:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
            cvc_decoded.append(result.decoded_cvc)
        except Exception:
            cvc_decoded.append('')

    # --- Build grammatical category arrays ---
    print("  Building grammatical classification arrays...")
    gram_categories: List[str] = []
    gram_functions: List[str] = []
    gram_codas: List[List[str]] = []

    for idx, token in enumerate(all_tokens):
        eva_chars = tokenize_eva_chars(token)
        classified = classify_token_chars_v2(eva_chars, coda_table)

        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:
                    codas.append(coda_val)

        decoded = cvc_decoded[idx]

        # Determine grammar
        from voynich.phases.inflectional_catalog import _determine_gram_function, CODA_GRAMMAR
        coda_dicts = [{'eva_char': '', 'coda_consonant': c} for c in codas]
        gram = _determine_gram_function(coda_dicts, decoded)
        gram_categories.append(gram['category'])
        gram_functions.append(gram['function'])
        gram_codas.append(codas)

    # --- Select passages ---
    print("\n  Selecting top 20 passages...")
    passages = _select_passages(
        all_tokens, cvc_decoded, folio_list, section_list,
        gram_categories, clean_indices, expanded_dict, gloss_lookup,
        n=20, window=15)
    print(f"  Selected: {len(passages)} passages")

    # --- Annotate passages ---
    print("\n  Annotating passages...")
    annotated_passages = []
    for i, window in enumerate(passages):
        ann = _annotate_passage(
            window, all_tokens, cvc_decoded, gram_categories,
            gram_functions, gram_codas, clean_indices,
            gloss_lookup, expanded_dict, root_lookup)

        templates = _match_templates(ann)
        structural = _format_structural_reading(ann)
        natural = _format_natural_reading(ann)
        interpretation = _build_interpretation(ann)

        n_tok = len(ann)
        n_gram = sum(1 for t in ann
                     if t['layer_5_category'] not in ('UNMARKED', 'MULTI_CODA', 'DOUBLE_CODA'))
        n_lex = sum(1 for t in ann
                    if t['layer_6_source'] in ('signal', 'T1_catalogue', 'root_dict',
                                                'pharma_vocab', 'dict'))

        gram_frac = n_gram / n_tok if n_tok > 0 else 0
        lex_frac = n_lex / n_tok if n_tok > 0 else 0

        passage_data = {
            'folio': window['folio'],
            'section': window.get('section', '?'),
            'start': window['start'],
            'end': window['end'],
            'n_tokens': n_tok,
            'grammatical_fraction': gram_frac,
            'identified_fraction': lex_frac,
            'structural_reading': structural,
            'natural_reading': natural,
            'interpretation': interpretation,
            'template_matches': templates,
            'tokens': ann,
        }
        annotated_passages.append(passage_data)

        template_str = f", templates: {len(templates)}" if templates else ""
        interp_str = f"\n      Interpretation: {interpretation}" if interpretation else ""
        print(f"    [{i+1}] {window['folio']} ({window.get('section', '?')}) "
              f"— gram={gram_frac:.0%} lex={lex_frac:.0%}{template_str}")
        print(f"      Natural: {natural[:100]}...")
        if interp_str:
            print(interp_str)

    # --- Statistics ---
    gram_fracs = [p['grammatical_fraction'] for p in annotated_passages]
    lex_fracs = [p['identified_fraction'] for p in annotated_passages]
    mean_gram = float(np.mean(gram_fracs)) if gram_fracs else 0.0
    mean_lex = float(np.mean(lex_fracs)) if lex_fracs else 0.0
    n_high_gram = sum(1 for f in gram_fracs if f > 0.70)
    n_high_lex = sum(1 for f in lex_fracs if f > 0.50)
    n_template = sum(1 for p in annotated_passages if p['template_matches'])
    n_template_strong = sum(
        1 for p in annotated_passages
        if p['template_matches'] and p['template_matches'][0].get('score', 0) > 0.4)
    n_interp = sum(1 for p in annotated_passages if p['interpretation'])

    print(f"\n  Mean grammatical: {mean_gram:.1%}")
    print(f"  Mean identified: {mean_lex:.1%}")
    print(f"  High grammatical (>70%): {n_high_gram}")
    print(f"  High identified (>50%): {n_high_lex}")
    print(f"  Template matches (any): {n_template}")
    print(f"  Template matches (>0.4): {n_template_strong}")
    print(f"  Interpretable: {n_interp}")

    # --- Null controls ---
    print("\n  Running null controls...")
    null_results = _run_null_controls(
        all_tokens, cvc_decoded, folio_list,
        gram_categories, gram_functions, gram_codas,
        clean_indices, gloss_lookup, expanded_dict, root_lookup,
        annotated_passages, n_random=20)

    print(f"    Real gram mean: {null_results['real_gram_mean']:.3f}")
    print(f"    Random gram mean: {null_results['random_gram_mean']:.3f}")
    print(f"    Real lex mean: {null_results['real_lex_mean']:.3f}")
    print(f"    Random lex mean: {null_results['random_lex_mean']:.3f}")
    print(f"    Lex selectivity: {null_results['lex_selectivity']:.2f}×")
    print(f"    Template selectivity: {null_results['template_selectivity']:.2f}×")

    # --- Gates ---
    g1 = n_high_gram >= 10
    g2 = n_high_lex >= 5
    g3 = n_template_strong >= 3
    g4 = null_results['template_selectivity'] > 1.3
    g5 = null_results['lex_selectivity'] > 1.5
    g6 = n_interp >= 1

    gates_passed = sum([g1, g2, g3, g4, g5, g6])

    print(f"\n  Gates: {gates_passed}/6")
    print(f"    G1 (≥10 gram >70%): {'PASS' if g1 else 'FAIL'} ({n_high_gram})")
    print(f"    G2 (≥5 lex >50%): {'PASS' if g2 else 'FAIL'} ({n_high_lex})")
    print(f"    G3 (≥3 template >0.4): {'PASS' if g3 else 'FAIL'} ({n_template_strong})")
    print(f"    G4 (template sel >1.3×): {'PASS' if g4 else 'FAIL'} "
          f"({null_results['template_selectivity']:.2f}×)")
    print(f"    G5 (lex sel >1.5×): {'PASS' if g5 else 'FAIL'} "
          f"({null_results['lex_selectivity']:.2f}×)")
    print(f"    G6 (≥1 interpretable): {'PASS' if g6 else 'FAIL'} ({n_interp})")

    if gates_passed >= 5:
        verdict = 'GRAMMATICAL_READING'
    elif gates_passed >= 3:
        verdict = 'PARTIAL_READING'
    else:
        verdict = 'NO_READING'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    passages_for_json = []
    for p in annotated_passages:
        p_copy = dict(p)
        p_copy['tokens'] = p_copy.get('tokens', [])[:5]  # truncate for size
        passages_for_json.append(p_copy)

    result = GrammaticalReadingResult(
        n_passages=len(annotated_passages),
        mean_grammatical_fraction=mean_gram,
        mean_identified_fraction=mean_lex,
        n_high_grammatical=n_high_gram,
        n_high_identified=n_high_lex,
        n_template_matches=n_template_strong,
        n_interpretable=n_interp,
        null_controls=null_results,
        passages=passages_for_json,
        gate_g1=g1,
        gate_g2=g2,
        gate_g3=g3,
        gate_g4=g4,
        gate_g5=g5,
        gate_g6=g6,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 5,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out = _save_json(rd, 'phase71_grammatical_reading.json', asdict(result))
    print(f"\n  Saved: {out}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
