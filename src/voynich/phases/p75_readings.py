"""
Phase 75, Track 5: Corrected Annotated Readings (3-Coda Model)
================================================================
Produce the best annotated pharmaceutical readings using ALL corrected
data from the 3-coda model: corrected decode (connector→null,
descender→null), corrected T1 vocabulary, corrected paradigms, and
corrected grammar labels.

NEW in Phase 75: integrates Phase 74 distributional vocabulary as an
additional confidence tier (DISTRIBUTIONAL) between DICT_HIT and
DECODED_CLEAN.

Key test: template selectivity — do grammatical templates match
selectively (> 1.3x) now that connector-passive AND descender noise
are removed?

Dependency chain:
    results/p75_redecode.json          (Step 0)
    results/p75_t1.json                (Track 3)
    results/p75_grammar.json           (Track 2)
    results/p75_paradigms.json         (Track 4)
    results/p74_patterns.json          (Phase 74 distributional vocab)
    results/combined_refine.json       (Phase 15)
    results/p69_clean_corpus.json      (Phase 69)
        -> results/p75_readings.json
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import classify_token_chars_v2
from voynich.phases.inflectional_catalog import CODA_GRAMMAR, _determine_gram_function
from voynich.phases.p75_redecode import _build_3coda_table
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
# Distributional vocabulary integration (Phase 74)
# ---------------------------------------------------------------------------

def _build_distributional_lookup(
    rd: str,
    min_sim: float = 0.50,
) -> Dict[str, Dict[str, Any]]:
    """Build EVA type -> distributional match lookup from p74_patterns.json.

    Only includes identifications with similarity > min_sim.
    Returns dict mapping EVA type to {matched_word, similarity, source}.
    """
    patterns_data = _safe_load(os.path.join(rd, 'p74_patterns.json'))
    distributional_ids = patterns_data.get('distributional_ids', [])

    lookup: Dict[str, Dict[str, Any]] = {}
    for entry in distributional_ids:
        sim = entry.get('similarity', 0.0)
        if sim > min_sim:
            eva_type = entry.get('eva_type', '')
            if eva_type:
                lookup[eva_type] = {
                    'matched_word': entry.get('matched_word', '?'),
                    'matched_t1_type': entry.get('matched_t1_type', ''),
                    'similarity': sim,
                    'source': 'distributional',
                }

    return lookup


# ---------------------------------------------------------------------------
# Grammatical templates (corrected — no connector/descender noise)
# ---------------------------------------------------------------------------

_TEMPLATES = [
    {
        'name': 'recipe_instruction',
        'pattern': ['VERBAL', 'NOMINAL'],
        'english': '[verb-imperative] [ingredient, accusative]',
    },
    {
        'name': 'property_statement',
        'pattern': ['FUNCTION_STEM', 'VERBAL', 'FUNCTION_STEM'],
        'english': '[subject] [is/has] [property]',
    },
    {
        'name': 'prep_phrase',
        'pattern': ['FUNCTION_STEM', 'NOMINAL'],
        'english': '[preposition] [noun, accusative]',
    },
    {
        'name': 'compound_instruction',
        'pattern': ['VERBAL', 'NOMINAL', 'FUNCTION_STEM', 'NOMINAL'],
        'english': '[verb] [ingredient] [prep] [medium]',
    },
    {
        'name': 'passive_instruction',
        'pattern': ['VERBAL', 'NOMINAL'],
        'english': '[is processed] [ingredient]',
    },
]


def _match_template(gram_sequence: List[str], pattern: List[str]) -> float:
    """Score how well a grammatical sequence matches a template pattern.

    Returns fraction of pattern found as subsequence.
    """
    if not pattern or not gram_sequence:
        return 0.0

    # Broad category mapping
    broad_map = {}
    for g in gram_sequence:
        if g in ('VERB_2SG', 'VERB_3SG', 'VERB_PASSIVE', 'VERB_3PL',
                 'VERB_EST', 'PARTICIPLE'):
            broad_map[g] = 'VERBAL'
        elif g in ('NOUN_ACC', 'NOUN_NOM_PL'):
            broad_map[g] = 'NOMINAL'
        elif g in ('FUNCTION_OR_SHORT_STEM',):
            broad_map[g] = 'FUNCTION_STEM'
        else:
            broad_map[g] = g

    broad_seq = [broad_map.get(g, g) for g in gram_sequence]

    # Subsequence match
    matched = 0
    seq_idx = 0
    for pat_item in pattern:
        while seq_idx < len(broad_seq):
            if broad_seq[seq_idx] == pat_item:
                matched += 1
                seq_idx += 1
                break
            seq_idx += 1

    return matched / len(pattern)


# ---------------------------------------------------------------------------
# Passage selection and annotation
# ---------------------------------------------------------------------------

def _select_passages(
    all_tokens: List[str],
    decoded_tokens: List[str],
    folio_list: List[str],
    section_list: List[str],
    t1_types: Set[str],
    clean_indices: Set[int],
    ref_word_set: Set[str],
    distrib_lookup: Dict[str, Dict[str, Any]],
    n: int = 20,
    window: int = 15,
) -> List[Dict[str, Any]]:
    """Score all 15-token windows, select top non-overlapping passages.

    Distributional matches contribute 1.5 points (between dict 2.0 and
    clean 1.0) to make them influential but not dominant.
    """
    windows = []
    for start in range(len(all_tokens) - window):
        end = start + window - 1
        if folio_list[start] != folio_list[end]:
            continue

        n_t1 = sum(1 for i in range(start, end + 1)
                   if all_tokens[i] in t1_types)
        n_dict = sum(1 for i in range(start, end + 1)
                    if decoded_tokens[i] and decoded_tokens[i].lower() in ref_word_set)
        n_clean = sum(1 for i in range(start, end + 1) if i in clean_indices)
        n_distrib = sum(1 for i in range(start, end + 1)
                       if all_tokens[i] in distrib_lookup
                       and all_tokens[i] not in t1_types)

        section = section_list[start]
        section_bonus = (1.0 if section == 'pharmaceutical' else
                        0.5 if section in ('herbal_a', 'herbal_b') else 0.0)

        score = (3.0 * n_t1/window + 2.0 * n_dict/window +
                 1.5 * n_distrib/window + 1.0 * n_clean/window +
                 section_bonus)
        windows.append({
            'start': start, 'end': end, 'score': score,
            'folio': folio_list[start], 'section': section,
            'n_t1': n_t1, 'n_dict': n_dict, 'n_distrib': n_distrib,
        })

    windows.sort(key=lambda w: -w['score'])

    selected = []
    used: Set[int] = set()
    for w in windows:
        positions = set(range(w['start'], w['end'] + 1))
        if not (positions & used):
            selected.append(w)
            used.update(positions)
            if len(selected) >= n:
                break

    return selected


def _annotate_passage(
    window: Dict,
    all_tokens: List[str],
    decoded_tokens: List[str],
    t1_map: Dict[str, str],
    ref_word_set: Set[str],
    clean_indices: Set[int],
    distrib_lookup: Dict[str, Dict[str, Any]],
    coda_table,
    eva_to_triple: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Produce 7-layer annotation for each token in a passage.

    Confidence tiers (in priority order):
        T1_IDENTIFIED   — in corrected T1 catalogue
        DICT_HIT        — decoded form in reference dictionary
        DISTRIBUTIONAL  — Phase 74 distributional match (sim > 0.50)
        DECODED_CLEAN   — in clean subset
        DECODED         — decoded but no external match
        UNKNOWN         — could not decode
    """
    annotated = []
    for idx in range(window['start'], window['end'] + 1):
        eva = all_tokens[idx]
        decoded = decoded_tokens[idx]
        is_clean = idx in clean_indices
        dict_match = bool(decoded and decoded.lower() in ref_word_set)
        t1_word = t1_map.get(eva, '')

        # Distributional match
        distrib_info = distrib_lookup.get(eva)
        distrib_word = distrib_info['matched_word'] if distrib_info else ''
        distrib_sim = distrib_info['similarity'] if distrib_info else 0.0

        # Morphology
        pos, case_end = _classify_latin_ending(decoded) if decoded else ('', '')

        # Grammar from coda
        eva_chars = tokenize_eva_chars(eva)
        classified = classify_token_chars_v2(eva_chars, coda_table)
        codas = []
        for role, char in classified:
            if role == 'CODA_MARKER':
                coda_val = get_coda(char, coda_table)
                if coda_val:
                    codas.append({'coda_consonant': coda_val})
        gram = _determine_gram_function(codas, decoded)

        # Confidence tier (with DISTRIBUTIONAL between DICT_HIT and DECODED_CLEAN)
        if t1_word:
            confidence = 'T1_IDENTIFIED'
        elif dict_match and decoded:
            confidence = 'DICT_HIT'
        elif distrib_info and not t1_word:
            confidence = 'DISTRIBUTIONAL'
        elif is_clean:
            confidence = 'DECODED_CLEAN'
        elif decoded:
            confidence = 'DECODED'
        else:
            confidence = 'UNKNOWN'

        entry = {
            'position': idx,
            'eva': eva,
            'decoded': decoded,
            'dict_match': dict_match,
            't1_word': t1_word,
            'pos': pos,
            'gram_category': gram['category'],
            'gram_function': gram['function'],
            'confidence': confidence,
            'is_clean': is_clean,
        }

        # Add distributional info when available
        if distrib_info:
            entry['distrib_word'] = distrib_word
            entry['distrib_similarity'] = round(distrib_sim, 4)

        annotated.append(entry)

    return annotated


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorrectedReadingsResult:
    phase: str = "75"
    step: str = "75.5"
    experiment: str = "readings_3coda"
    n_passages: int = 0
    mean_identified_fraction: float = 0.0
    n_high_quality: int = 0  # passages with > 70% identified
    template_selectivity: float = 0.0
    lexical_selectivity: float = 0.0
    n_template_matches: int = 0
    n_interpretable: int = 0
    # Distributional vocabulary integration
    n_distributional_types: int = 0
    distributional_coverage: float = 0.0  # fraction of tokens matched distributionally
    passages: List[Dict[str, Any]] = field(default_factory=list)
    null_controls: Dict[str, Any] = field(default_factory=dict)
    # Gates
    gate_r1: bool = False  # mean identified > 60%
    gate_r2: bool = False  # >= 5 passages > 70%
    gate_r3: bool = False  # template selectivity > 1.3x
    gate_r4: bool = False  # >= 5 CI template matches > 0.4
    gate_r5: bool = False  # >= 1 interpretable
    gate_r6: bool = False  # lexical selectivity > 1.5x
    gates_passed: int = 0
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_readings_3coda() -> CorrectedReadingsResult:
    """Track 5: Annotated readings with 3-coda decode + distributional vocab."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 75.5 — Corrected Annotated Readings (3-Coda Model)")
    print("=" * 58)

    # --- Load 3-coda decoded data ---
    redecode_data = _safe_load(os.path.join(rd, 'p75_redecode.json'))
    decoded_tokens = redecode_data.get('decoded_tokens', [])
    folio_list = redecode_data.get('folio_list', [])
    section_list = redecode_data.get('section_list', [])

    if not decoded_tokens:
        print("  ERROR: p75_redecode.json not found.")
        return CorrectedReadingsResult()

    t1_data = _safe_load(os.path.join(rd, 'p75_t1.json'))
    t1_catalogue = t1_data.get('identifications', [])
    t1_map = {i['token']: i['matched_word']
              for i in t1_catalogue if 'token' in i and 'matched_word' in i}
    t1_types = set(t1_map.keys())

    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    clean_indices = set(clean_data.get('clean_indices', []))

    # --- Load distributional vocabulary from Phase 74 ---
    distrib_lookup = _build_distributional_lookup(rd, min_sim=0.50)
    print(f"  Distributional vocab (sim > 0.50): {len(distrib_lookup)} types")

    # --- Load dictionary ---
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = _build_3coda_table()

    print(f"  Tokens: {len(all_tokens)}")
    print(f"  T1 types: {len(t1_types)}")

    # --- Compute distributional coverage ---
    n_distrib_hits = sum(1 for t in all_tokens if t in distrib_lookup)
    distributional_coverage = n_distrib_hits / len(all_tokens) if all_tokens else 0.0
    print(f"  Distributional coverage: {100*distributional_coverage:.1f}% "
          f"({n_distrib_hits}/{len(all_tokens)} tokens)")

    # --- Select passages ---
    print("  Selecting best passages...")
    passages = _select_passages(
        all_tokens, decoded_tokens, folio_list, section_list,
        t1_types, clean_indices, ref_word_set, distrib_lookup,
        n=20, window=15)

    # --- Annotate each passage ---
    print(f"  Annotating {len(passages)} passages...")
    annotated_passages = []
    for window in passages:
        ann = _annotate_passage(
            window, all_tokens, decoded_tokens, t1_map,
            ref_word_set, clean_indices, distrib_lookup,
            coda_table, eva_to_triple)

        # Count identified = T1 + DICT_HIT + DISTRIBUTIONAL
        n_identified = sum(1 for t in ann
                          if t['confidence'] in ('T1_IDENTIFIED', 'DICT_HIT',
                                                  'DISTRIBUTIONAL'))
        id_fraction = n_identified / len(ann) if ann else 0.0

        # Template matching
        gram_seq = [t['gram_function'] for t in ann]
        template_matches = []
        for tmpl in _TEMPLATES:
            score = _match_template(gram_seq, tmpl['pattern'])
            if score > 0.3:
                template_matches.append({
                    'template': tmpl['name'],
                    'score': round(score, 3),
                })

        # Interpretation attempt
        has_verb = any(t['gram_category'] == 'VERBAL' for t in ann)
        has_noun = any(t['gram_category'] == 'NOMINAL' for t in ann)
        has_t1 = any(t['confidence'] == 'T1_IDENTIFIED' for t in ann)
        is_interpretable = has_verb and has_noun and has_t1 and id_fraction > 0.50

        annotated_passages.append({
            'folio': window['folio'],
            'section': window['section'],
            'start': window['start'],
            'end': window['end'],
            'score': round(window['score'], 3),
            'identified_fraction': round(id_fraction, 3),
            'n_distributional': sum(1 for t in ann
                                    if t['confidence'] == 'DISTRIBUTIONAL'),
            'template_matches': sorted(template_matches, key=lambda m: -m['score']),
            'is_interpretable': is_interpretable,
            'tokens': ann,
        })

    # --- Null controls (random passages) ---
    print("  Running null controls (20 random passages)...")
    rng = random.Random(42)
    all_starts = list(range(len(all_tokens) - 15))
    rng.shuffle(all_starts)

    used_positions: Set[int] = set()
    for p in passages:
        used_positions.update(range(p['start'], p['end'] + 1))

    random_id_fractions = []
    random_template_counts = []
    for start in all_starts:
        if len(random_id_fractions) >= 20:
            break
        end = start + 14
        if end >= len(folio_list) or folio_list[start] != folio_list[end]:
            continue
        positions = set(range(start, end + 1))
        if positions & used_positions:
            continue
        used_positions.update(positions)

        window = {'start': start, 'end': end, 'folio': folio_list[start]}
        ann = _annotate_passage(
            window, all_tokens, decoded_tokens, t1_map,
            ref_word_set, clean_indices, distrib_lookup,
            coda_table, eva_to_triple)

        n_id = sum(1 for t in ann
                   if t['confidence'] in ('T1_IDENTIFIED', 'DICT_HIT',
                                           'DISTRIBUTIONAL'))
        random_id_fractions.append(n_id / len(ann) if ann else 0.0)

        gram_seq = [t['gram_function'] for t in ann]
        n_tmpl = sum(1 for tmpl in _TEMPLATES
                    if _match_template(gram_seq, tmpl['pattern']) > 0.4)
        random_template_counts.append(n_tmpl)

    # --- Compute metrics ---
    real_id_fractions = [p['identified_fraction'] for p in annotated_passages]
    mean_identified = float(np.mean(real_id_fractions)) if real_id_fractions else 0.0
    n_high_quality = sum(1 for f in real_id_fractions if f > 0.70)

    real_template_count = sum(
        1 for p in annotated_passages
        if p['template_matches'] and p['template_matches'][0]['score'] > 0.4)

    n_interpretable = sum(1 for p in annotated_passages if p['is_interpretable'])

    random_mean_id = float(np.mean(random_id_fractions)) if random_id_fractions else 0.0
    random_mean_tmpl = float(np.mean(random_template_counts)) if random_template_counts else 0.0

    lexical_sel = mean_identified / random_mean_id if random_mean_id > 0 else float('inf')
    template_sel = (real_template_count / len(annotated_passages)) / (
        random_mean_tmpl if random_mean_tmpl > 0 else 0.001) if annotated_passages else 0.0

    null_controls = {
        'random_mean_identified': round(random_mean_id, 4),
        'random_mean_templates': round(random_mean_tmpl, 2),
        'lexical_selectivity': round(lexical_sel, 3),
        'template_selectivity': round(template_sel, 3),
    }

    print(f"\n  Results:")
    print(f"    Mean identified: {100*mean_identified:.1f}% "
          f"(random: {100*random_mean_id:.1f}%)")
    print(f"    Lexical selectivity: {lexical_sel:.2f}x")
    print(f"    Template selectivity: {template_sel:.2f}x")
    print(f"    High quality (>70%): {n_high_quality}")
    print(f"    Interpretable: {n_interpretable}")
    print(f"    Distributional types used: {len(distrib_lookup)}")
    print(f"    Distributional coverage: {100*distributional_coverage:.1f}%")

    # --- Gates ---
    gate_r1 = mean_identified > 0.60
    gate_r2 = n_high_quality >= 5
    gate_r3 = template_sel > 1.3
    gate_r4 = real_template_count >= 5
    gate_r5 = n_interpretable >= 1
    gate_r6 = lexical_sel > 1.5
    gates_passed = sum([gate_r1, gate_r2, gate_r3, gate_r4, gate_r5, gate_r6])

    print(f"\n  Gates:")
    print(f"    R1 (mean > 60%):           {'PASS' if gate_r1 else 'FAIL'} "
          f"({100*mean_identified:.1f}%)")
    print(f"    R2 (>=5 high quality):     {'PASS' if gate_r2 else 'FAIL'} "
          f"({n_high_quality})")
    print(f"    R3 (template sel > 1.3x):  {'PASS' if gate_r3 else 'FAIL'} "
          f"({template_sel:.2f}x)")
    print(f"    R4 (>=5 template matches): {'PASS' if gate_r4 else 'FAIL'} "
          f"({real_template_count})")
    print(f"    R5 (>=1 interpretable):    {'PASS' if gate_r5 else 'FAIL'} "
          f"({n_interpretable})")
    print(f"    R6 (lex sel > 1.5x):       {'PASS' if gate_r6 else 'FAIL'} "
          f"({lexical_sel:.2f}x)")

    if gates_passed >= 4:
        verdict = 'CORRECTED_READING'
    elif gates_passed >= 2:
        verdict = 'PARTIAL_READING'
    else:
        verdict = 'NO_READING'

    result = CorrectedReadingsResult(
        n_passages=len(annotated_passages),
        mean_identified_fraction=round(mean_identified, 4),
        n_high_quality=n_high_quality,
        template_selectivity=round(template_sel, 3),
        lexical_selectivity=round(lexical_sel, 3),
        n_template_matches=real_template_count,
        n_interpretable=n_interpretable,
        n_distributional_types=len(distrib_lookup),
        distributional_coverage=round(distributional_coverage, 4),
        passages=annotated_passages,
        null_controls=null_controls,
        gate_r1=gate_r1,
        gate_r2=gate_r2,
        gate_r3=gate_r3,
        gate_r4=gate_r4,
        gate_r5=gate_r5,
        gate_r6=gate_r6,
        gates_passed=gates_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p75_readings.json', asdict(result))
    print(f"\n  Verdict: {verdict} ({gates_passed}/6)")
    print(f"  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
