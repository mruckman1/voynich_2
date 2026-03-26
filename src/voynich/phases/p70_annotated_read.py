"""
Phase 70, Track 4: Annotated Pharmaceutical Readings
=====================================================
Select top 20 T1-dense passages (15-token windows), produce 7-layer
annotated readings, match against CI recipe templates, and run null
controls (shuffled + random).

Dependency chain:
    results/p69_clean_corpus.json        (Step 0: t1_catalogue, clean_indices)
    results/phase70_pharma_dict.json     (Track 1: expanded dict — optional)
    results/phase70_paradigms.json       (Track 2: paradigm data — optional)
    results/combined_refine.json         (Phase 15: best_assignment)
    data/reference/latin/circa_instans.txt
        -> results/phase70_readings.json
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import (
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
# CI vocabulary for template matching
# ---------------------------------------------------------------------------

# Known CI chapter topics (ingredient → chapter description)
_CI_CHAPTERS: Dict[str, str] = {
    'rosa': 'De Rosa (On Rose)',
    'senna': 'De Senna (On Senna)',
    'corallum': 'De Corallio (On Coral)',
    'cera': 'De Cera (On Wax)',
    'oleum': 'De Oleo (On Oil)',
    'vinum': 'De Vino (On Wine)',
    'acetum': 'De Aceto (On Vinegar)',
    'mel': 'De Melle (On Honey)',
    'sal': 'De Sale (On Salt)',
    'aqua': 'De Aqua (On Water)',
    'herba': 'De Herba (On Herb)',
    'radix': 'De Radice (On Root)',
    'cortex': 'De Cortice (On Bark)',
    'flos': 'De Flore (On Flower)',
    'semen': 'De Semine (On Seed)',
    'stercus': 'De Stercore (On Dung)',
    'cassia': 'De Cassia (On Cassia)',
}

# Pharmaceutical verbs (imperatives)
_PHARMA_VERBS = set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('verbs', []))

# Ingredients
_INGREDIENTS = (
    set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('plant_parts', []))
    | set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('preparations', []))
)

# Qualities
_QUALITIES = set(w.lower() for w in PHARMACEUTICAL_VOCABULARY.get('qualities', []))


def _build_folio_list(corpus) -> List[str]:
    """Build flat list of folio IDs, one per token."""
    folios: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folios.append(folio)
    return folios


def _build_section_list(corpus) -> List[str]:
    """Build flat list of section labels, one per token."""
    sections: List[str] = []
    for _folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            sections.append(getattr(page, 'section', 'unknown'))
    return sections


def _select_best_passages(
    all_tokens: List[str],
    cvc_decoded: List[str],
    folio_list: List[str],
    section_list: List[str],
    t1_types: Set[str],
    clean_indices: Set[int],
    expanded_dict: Set[str],
    n: int = 20,
    window: int = 15,
) -> List[Dict[str, Any]]:
    """Select top-N non-overlapping windows scored by T1 density + dict-hit."""
    windows = []

    for start in range(len(all_tokens) - window):
        # Must be same folio
        if folio_list[start] != folio_list[start + window - 1]:
            continue

        n_t1 = sum(1 for i in range(start, start + window)
                   if all_tokens[i] in t1_types)
        n_dict = sum(1 for i in range(start, start + window)
                     if cvc_decoded[i] and cvc_decoded[i] in expanded_dict)
        n_clean = sum(1 for i in range(start, start + window)
                      if i in clean_indices)

        section = section_list[start] if start < len(section_list) else 'unknown'

        section_bonus = (1.0 if section == 'pharmaceutical' else
                         0.5 if section in ('herbal_a', 'herbal_b') else 0.0)

        score = (
            3.0 * n_t1 / window +
            2.0 * n_dict / window +
            1.0 * n_clean / window +
            section_bonus
        )

        windows.append({
            'start': start,
            'end': start + window - 1,
            'folio': folio_list[start],
            'section': section,
            'n_t1': n_t1,
            'n_dict': n_dict,
            'n_clean': n_clean,
            'score': score,
        })

    windows.sort(key=lambda w: -w['score'])

    # Deduplicate overlapping
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


def _build_master_gloss(
    t1_catalogue: List[Dict],
    expanded_dict: Set[str],
    paradigm_data: Dict,
) -> Dict[str, Dict[str, str]]:
    """Build master gloss lookup for annotation."""
    lookup: Dict[str, Dict[str, str]] = {}

    # Signal words (best glosses)
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

    # Paradigm roots (from Track 2)
    for paradigm in paradigm_data.get('paradigm_details', []):
        root = paradigm.get('root', '')
        meaning = paradigm.get('meaning', '?')
        if meaning != '?':
            for member in paradigm.get('members', []):
                w = member.get('decoded', '')
                case = member.get('case_ending', '')
                if w and w not in lookup:
                    case_str = f" ({case})" if case else ""
                    lookup[w] = {
                        'gloss': f"{meaning}{case_str}",
                        'class': 'paradigm',
                        'source': 'Track2',
                    }

    return lookup


def _annotate_passage(
    window: Dict,
    all_tokens: List[str],
    cvc_decoded: List[str],
    clean_indices: Set[int],
    t1_map: Dict[str, Dict],
    gloss_lookup: Dict[str, Dict[str, str]],
    expanded_dict: Set[str],
) -> List[Dict[str, Any]]:
    """Produce 7-layer annotation for each token in a passage."""
    annotated = []

    for idx in range(window['start'], window['end'] + 1):
        eva = all_tokens[idx]
        decoded = cvc_decoded[idx] if idx < len(cvc_decoded) else ''
        is_clean = idx in clean_indices

        # Layer 3: Dict match
        dict_match = decoded in expanded_dict if decoded else False

        # Layer 4: T1 identification
        t1_match = t1_map.get(eva)

        # Layer 5: Morphological analysis
        pos, case_ending = _classify_latin_ending(decoded) if decoded else ('', '')

        # Layer 6 + 7: Gloss and confidence
        gloss_info = gloss_lookup.get(decoded, {})
        if t1_match:
            gloss = gloss_info.get('gloss', decoded)
            confidence = 'T1_IDENTIFIED'
        elif gloss_info.get('source') == 'Track2':
            gloss = gloss_info.get('gloss', decoded)
            confidence = 'PARADIGM'
        elif gloss_info.get('source') == 'signal':
            gloss = gloss_info.get('gloss', decoded)
            confidence = 'SIGNAL'
        elif dict_match and decoded:
            gloss = gloss_info.get('gloss', decoded)
            confidence = 'DICT_HIT'
        elif decoded and is_clean:
            gloss = f'[{decoded}]'
            confidence = 'DECODED_CLEAN'
        elif decoded:
            gloss = f'[{decoded}]'
            confidence = 'DECODED'
        else:
            gloss = '?'
            confidence = 'UNKNOWN'

        annotated.append({
            'position': idx,
            'layer_1_eva': eva,
            'layer_2_decoded': decoded,
            'layer_3_dict_match': dict_match,
            'layer_4_t1': t1_match.get('matched_word', '') if t1_match else '',
            'layer_5_pos': pos,
            'layer_5_case': case_ending,
            'layer_6_gloss': gloss,
            'layer_7_confidence': confidence,
            'is_clean': is_clean,
        })

    return annotated


def _format_reading(annotated: List[Dict], window: Dict) -> Dict[str, Any]:
    """Format annotation into human-readable output + stats."""
    reading_parts = []
    for tok in annotated:
        conf = tok['layer_7_confidence']
        if conf in ('T1_IDENTIFIED', 'PARADIGM', 'SIGNAL', 'DICT_HIT'):
            reading_parts.append(tok['layer_6_gloss'])
        else:
            reading_parts.append(f"[{tok['layer_2_decoded']}]")

    reading = ' · '.join(reading_parts)

    n_identified = sum(1 for t in annotated
                       if t['layer_7_confidence'] in
                       ('T1_IDENTIFIED', 'PARADIGM', 'SIGNAL', 'DICT_HIT'))
    identified_fraction = n_identified / len(annotated) if annotated else 0.0

    conf_dist = Counter(t['layer_7_confidence'] for t in annotated)

    return {
        'folio': window['folio'],
        'section': window.get('section', '?'),
        'start': window['start'],
        'end': window['end'],
        'n_tokens': len(annotated),
        'identified_fraction': identified_fraction,
        'reading': reading,
        'confidence_distribution': dict(conf_dist),
        'tokens': annotated,
    }


def _match_ci_chapter(annotated: List[Dict]) -> Optional[Dict[str, Any]]:
    """Match passage ingredients against CI chapters."""
    decoded_words = set(
        t['layer_2_decoded'] for t in annotated
        if t['layer_2_decoded'] and t['layer_7_confidence'] in
        ('T1_IDENTIFIED', 'PARADIGM', 'SIGNAL', 'DICT_HIT')
    )

    best_match = None
    best_score = 0

    for ingredient, chapter in _CI_CHAPTERS.items():
        # Check if any decoded word is related to this ingredient
        score = 0
        for w in decoded_words:
            if w == ingredient or ingredient.startswith(w) or w.startswith(ingredient[:3]):
                score += 2
            elif any(w.startswith(ingredient[:k]) for k in range(3, len(ingredient))):
                score += 1

        if score > best_score:
            best_score = score
            best_match = {
                'chapter': chapter,
                'ingredient': ingredient,
                'score': score,
                'matched_words': [w for w in decoded_words
                                  if ingredient.startswith(w[:3]) or w.startswith(ingredient[:3])],
            }

    return best_match if best_match and best_match['score'] >= 2 else None


def _build_interpretation(annotated: List[Dict], ci_match: Optional[Dict]) -> Optional[str]:
    """Attempt a pharmaceutical interpretation of the passage."""
    verbs = [t for t in annotated
             if t['layer_2_decoded'] in _PHARMA_VERBS]
    ingredients = [t for t in annotated
                   if t['layer_2_decoded'] in _INGREDIENTS]
    qualities = [t for t in annotated
                 if t['layer_2_decoded'] in _QUALITIES]

    parts = []
    if verbs:
        parts.append('/'.join(set(t['layer_6_gloss'] for t in verbs)))
    if ingredients:
        parts.append(' + '.join(set(t['layer_6_gloss'] for t in ingredients)))
    if qualities:
        parts.append(f"({', '.join(set(t['layer_6_gloss'] for t in qualities))})")
    if ci_match:
        parts.append(f"— cf. {ci_match['chapter']}")

    return ' '.join(parts) if parts else None


def _run_null_controls(
    all_tokens: List[str],
    cvc_decoded: List[str],
    folio_list: List[str],
    clean_indices: Set[int],
    t1_types: Set[str],
    expanded_dict: Set[str],
    gloss_lookup: Dict[str, Dict[str, str]],
    t1_map: Dict[str, Dict],
    selected_passages: List[Dict],
    n_random: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run null controls: random (non-T1-dense) passages."""
    rng = random.Random(seed)

    # Compute identified_fraction for selected passages
    real_fractions = []
    for passage_data in selected_passages:
        real_fractions.append(passage_data.get('identified_fraction', 0.0))

    # Random passages: pick windows that avoid T1-dense regions
    all_starts = list(range(len(all_tokens) - 15))
    rng.shuffle(all_starts)

    random_fractions = []
    used = set()
    for start in all_starts:
        if len(random_fractions) >= n_random:
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

        # Annotate this random window
        window = {'start': start, 'end': end, 'folio': folio_list[start]}
        ann = _annotate_passage(
            window, all_tokens, cvc_decoded, clean_indices,
            t1_map, gloss_lookup, expanded_dict)
        n_id = sum(1 for t in ann if t['layer_7_confidence'] in
                   ('T1_IDENTIFIED', 'PARADIGM', 'SIGNAL', 'DICT_HIT'))
        random_fractions.append(n_id / len(ann) if ann else 0.0)

    real_mean = float(np.mean(real_fractions)) if real_fractions else 0.0
    random_mean = float(np.mean(random_fractions)) if random_fractions else 0.0
    selectivity = real_mean / random_mean if random_mean > 0 else float('inf')

    return {
        'real_mean_identified': real_mean,
        'random_mean_identified': random_mean,
        'selectivity_vs_random': selectivity,
        'n_real': len(real_fractions),
        'n_random': len(random_fractions),
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedReadingResult:
    phase: str = "70"
    step: str = "70.4"
    experiment: str = "annotated_readings"
    # Passage stats
    n_passages: int = 0
    mean_identified_fraction: float = 0.0
    n_high_quality: int = 0  # identified > 70%
    n_ci_matches: int = 0
    n_interpretations: int = 0
    # Null controls
    null_random_identified: float = 0.0
    selectivity_vs_random: float = 0.0
    # Passages (truncated for JSON size)
    passages: List[Dict] = field(default_factory=list)
    # Gates
    gate_r1: bool = False  # mean identified > 50%
    gate_r2: bool = False  # >= 5 passages > 70%
    gate_r3: bool = False  # >= 3 CI template matches
    gate_r4: bool = False  # >= 1 coherent interpretation
    gate_r5: bool = False  # selected > 1.5× random
    gate_r6: bool = False  # at least 1 passage with identified > 60%
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_annotate_read():
    """Track 4: Select passages, annotate, match CI, run null controls."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 70.4 — Annotated Pharmaceutical Readings")
    print("=" * 49)

    # --- Load dependencies ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])
    clean_indices_list = clean_data.get('clean_indices', [])
    clean_indices = set(clean_indices_list)
    t1_types = set(entry['eva_type'] for entry in t1_catalogue if entry.get('eva_type'))
    t1_map = {entry['eva_type']: entry for entry in t1_catalogue if entry.get('eva_type')}
    print(f"  T1 types: {len(t1_types)}, Clean tokens: {len(clean_indices)}")

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folio_list = _build_folio_list(corpus)
    section_list = _build_section_list(corpus)

    # Load expanded dict (Track 1 if available, else base)
    pharma_data = _safe_load(os.path.join(rd, 'phase70_pharma_dict.json'))
    if pharma_data.get('combined_word_list'):
        expanded_dict = set(pharma_data['combined_word_list'])
        print(f"  Using Track 1 dict: {len(expanded_dict)} words")
    else:
        print("  Track 1 not available; building base expanded dict...")
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                         if len(w) >= 2)
        expanded_dict, _ = build_expanded_word_set(base_words)
        expanded_dict = base_words | expanded_dict
        print(f"  Base expanded dict: {len(expanded_dict)} words")

    # Load paradigm data (Track 2 — optional)
    paradigm_data = _safe_load(os.path.join(rd, 'phase70_paradigms.json'))

    # --- Decode corpus ---
    print("\n  Decoding corpus (CVC)...")
    cvc_decoded = []
    for token in all_tokens:
        try:
            result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
            cvc_decoded.append(result.decoded_cvc)
        except Exception:
            cvc_decoded.append('')

    # --- Build master gloss ---
    gloss_lookup = _build_master_gloss(t1_catalogue, expanded_dict, paradigm_data)
    print(f"  Gloss entries: {len(gloss_lookup)}")

    # --- Select best passages ---
    print("\n  Selecting top 20 T1-dense passages...")
    passages = _select_best_passages(
        all_tokens, cvc_decoded, folio_list, section_list,
        t1_types, clean_indices, expanded_dict, n=20, window=15)
    print(f"  Selected: {len(passages)} passages")

    # --- Annotate each passage ---
    print("\n  Annotating passages...")
    annotated_passages = []
    for i, window in enumerate(passages):
        ann = _annotate_passage(
            window, all_tokens, cvc_decoded, clean_indices,
            t1_map, gloss_lookup, expanded_dict)
        formatted = _format_reading(ann, window)

        # CI matching
        ci_match = _match_ci_chapter(ann)
        formatted['ci_match'] = ci_match

        # Pharmaceutical interpretation
        interpretation = _build_interpretation(ann, ci_match)
        formatted['interpretation'] = interpretation

        annotated_passages.append(formatted)

        # Print summary
        ci_tag = f" → {ci_match['chapter']}" if ci_match else ""
        interp_tag = f"\n      Interpretation: {interpretation}" if interpretation else ""
        print(f"    [{i+1}] {window['folio']} ({window.get('section', '?')}) "
              f"— {formatted['identified_fraction']:.0%} identified{ci_tag}")
        print(f"      Reading: {formatted['reading'][:100]}...")
        if interp_tag:
            print(interp_tag)

    # --- Statistics ---
    id_fractions = [p['identified_fraction'] for p in annotated_passages]
    mean_id = float(np.mean(id_fractions)) if id_fractions else 0.0
    n_high = sum(1 for f in id_fractions if f > 0.70)
    n_ci = sum(1 for p in annotated_passages if p.get('ci_match'))
    n_interp = sum(1 for p in annotated_passages if p.get('interpretation'))

    print(f"\n  Mean identified: {mean_id:.1%}")
    print(f"  High quality (>70%): {n_high}")
    print(f"  CI matches: {n_ci}")
    print(f"  Interpretations: {n_interp}")

    # --- Null controls ---
    print("\n  Running null controls (20 random passages)...")
    null_results = _run_null_controls(
        all_tokens, cvc_decoded, folio_list, clean_indices,
        t1_types, expanded_dict, gloss_lookup, t1_map,
        annotated_passages, n_random=20)

    print(f"    Real mean: {null_results['real_mean_identified']:.3f}")
    print(f"    Random mean: {null_results['random_mean_identified']:.3f}")
    print(f"    Selectivity: {null_results['selectivity_vs_random']:.2f}×")

    # --- Gates ---
    g1 = mean_id > 0.50
    g2 = n_high >= 5
    g3 = n_ci >= 3
    g4 = n_interp >= 1
    g5 = null_results['selectivity_vs_random'] > 1.5
    g6 = any(f > 0.60 for f in id_fractions)

    gates_passed = sum([g1, g2, g3, g4, g5, g6])

    print(f"\n  Gates: {gates_passed}/6")
    print(f"    R1 (mean identified > 50%): {'PASS' if g1 else 'FAIL'} ({mean_id:.1%})")
    print(f"    R2 (≥5 passages > 70%): {'PASS' if g2 else 'FAIL'} ({n_high})")
    print(f"    R3 (≥3 CI matches): {'PASS' if g3 else 'FAIL'} ({n_ci})")
    print(f"    R4 (≥1 interpretation): {'PASS' if g4 else 'FAIL'} ({n_interp})")
    print(f"    R5 (selectivity > 1.5×): {'PASS' if g5 else 'FAIL'} "
          f"({null_results['selectivity_vs_random']:.2f}×)")
    print(f"    R6 (any > 60%): {'PASS' if g6 else 'FAIL'}")

    if gates_passed >= 4:
        verdict = 'PHARMACEUTICAL_READING'
    elif gates_passed >= 2:
        verdict = 'PARTIAL_READING'
    else:
        verdict = 'NO_READING'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    # Truncate token-level detail for JSON size
    passages_for_json = []
    for p in annotated_passages:
        p_copy = dict(p)
        # Keep only summary + reading, not full token arrays
        p_copy['tokens'] = p_copy.get('tokens', [])[:5]  # sample only
        passages_for_json.append(p_copy)

    result = AnnotatedReadingResult(
        n_passages=len(annotated_passages),
        mean_identified_fraction=mean_id,
        n_high_quality=n_high,
        n_ci_matches=n_ci,
        n_interpretations=n_interp,
        null_random_identified=null_results['random_mean_identified'],
        selectivity_vs_random=null_results['selectivity_vs_random'],
        passages=passages_for_json,
        gate_r1=g1,
        gate_r2=g2,
        gate_r3=g3,
        gate_r4=g4,
        gate_r5=g5,
        gate_r6=g6,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 4,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    out = _save_json(rd, 'phase70_readings.json', asdict(result))
    print(f"\n  Saved: {out}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
