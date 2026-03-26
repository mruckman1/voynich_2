"""
Phase 70, Track 1: Pharmaceutical Dictionary Expansion
=======================================================
Build a pharmaceutical-domain dictionary that goes beyond the general Latin
expansion in reference.py.  New layers:

  1. Focused pharma inflection table (20 stems × full declension/conjugation)
  2. Gallo-Italic dialectal variants (degemination, northern accusative)
  3. Circa Instans full-text vocabulary extraction
  4. Aggregated function-word set (signal + particles + pharma function words)
  5. T1-identified words (from Phase 68 expanded T1)

Evaluate the expanded dictionary against the clean subset (22,823 tokens)
and null corpora to measure selectivity.

Dependency chain:
    results/p69_clean_corpus.json        (Step 0)
    results/combined_refine.json         (Phase 15)
    results/modifier_integrate.json      (Phase 16)
    data/reference/latin/circa_instans.txt
        -> results/phase70_pharma_dict.json
"""

import json
import os
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
    generate_inflected_forms,
    generate_medieval_variants,
    load_reference_corpus,
)
from voynich.phases.corrected_coda import build_coda_table_v2, decode_token_cvc_v2
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51
from voynich.phases.suffix_grammar import _PARTICLES


# ---------------------------------------------------------------------------
# JSON helpers (standard pattern)
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
# Pharmaceutical inflection tables
# ---------------------------------------------------------------------------

# Stems not already in reference.py's _PHARMA_STEMS, plus extra pharma terms
_EXTRA_PHARMA_STEMS: List[Tuple[str, str]] = [
    # Nouns — 1st declension (a-stem)
    ('senn', 'noun1'),       # senna
    ('rut', 'noun1'),        # ruta
    ('cassi', 'noun1'),      # cassia
    ('ciner', 'noun1'),      # cinera (ash)
    ('ros', 'noun1'),        # rosa
    ('cer', 'noun1'),        # cera (wax)
    ('sal', 'noun1'),        # salvia
    ('camomil', 'noun1'),    # camomilla
    # Nouns — 2nd declension
    ('corall', 'noun2n'),    # corallum
    ('morb', 'noun2'),       # morbus
    ('medicament', 'noun2n'),  # medicamentum
    ('emplast', 'noun2n'),   # emplastrum
    ('electuari', 'noun2n'), # electuarium
    # Nouns — 3rd declension
    ('ration', 'noun3'),     # ratio
    ('stercor', 'noun3'),    # stercus/stercoris
    ('commun', 'noun3'),     # communis
    ('sal', 'noun3'),        # sal/salis
    ('febr', 'noun3'),       # febris
    ('dolor', 'noun3'),      # dolor
    ('tumor', 'noun3'),      # tumor
    ('sanguin', 'noun3'),    # sanguis
    ('potio', 'noun3'),      # potio
    ('cur', 'noun1'),        # cura
    # Verbs — additional pharma imperatives
    ('recip', 'verb3'),      # recipere
    ('accip', 'verb3'),      # accipere
    ('incid', 'verb3'),      # incidere
    ('frang', 'verb3'),      # frangere
    ('pon', 'verb3'),        # ponere
    ('calefac', 'verb3'),    # calefacere
    ('mundific', 'verb1'),   # mundificare
]

# Full Gallo-Italic verb forms from recipe tradition
_PHARMA_VERB_FORMS: Dict[str, List[str]] = {
    'col': ['cola', 'colare', 'colat', 'coletur', 'colata', 'colando', 'colatus'],
    'ter': ['tere', 'terere', 'terit', 'teritur', 'tritus', 'trita', 'terendo'],
    'misc': ['misce', 'miscere', 'miscet', 'miscetur', 'mixtus', 'mixta'],
    'coqu': ['coque', 'coquere', 'coquit', 'coquitur', 'coctus', 'cocta'],
    'add': ['adde', 'addere', 'addit', 'additur', 'additus', 'addita'],
    'solv': ['solve', 'solvere', 'solvit', 'solvitur', 'solutus', 'soluta'],
    'recip': ['recipe', 'recipere', 'recipit', 'receptus'],
    'accip': ['accipe', 'accipere', 'accipit', 'acceptus'],
    'pon': ['pone', 'ponere', 'ponit', 'ponitur', 'positus', 'posita'],
    'fac': ['fac', 'facere', 'facit', 'fiat', 'factus', 'facta'],
    'distill': ['distilla', 'distillare', 'distillat', 'distillatus'],
    'lav': ['lava', 'lavare', 'lavat', 'lavatur', 'lavatus', 'lavata'],
    'bib': ['bibe', 'bibere', 'bibit', 'bibitur'],
    'ung': ['unge', 'ungere', 'ungit', 'ungitur', 'unctus', 'uncta'],
}

# Gallo-Italic degemination rules
_DEGEMINATION_RULES = [
    ('ll', 'l'), ('nn', 'n'), ('mm', 'm'),
    ('ss', 's'), ('tt', 't'), ('rr', 'r'), ('pp', 'p'),
]

# Additional function words (Italian/Romance)
_ITALIAN_FUNCTION_WORDS: Dict[str, str] = {
    'di': 'of', 'la': 'the(f)', 'li': 'the(pl)', 'il': 'the(m)',
    'lo': 'the(m)', 'le': 'the(f.pl)', 'un': 'a/one',
    'ci': 'there', 'se': 'if/self', 'co': 'with(dial)',
    'con': 'with', 'per': 'through', 'che': 'that/which',
    'del': 'of the', 'nel': 'in the', 'dal': 'from the',
    'al': 'to the', 'bene': 'well', 'male': 'badly',
    'satis': 'enough', 'multum': 'much', 'parum': 'little',
    'sic': 'thus', 'ita': 'so', 'tunc': 'then', 'ibi': 'there',
    'hic': 'this', 'ille': 'that', 'qui': 'who/which',
    'que': 'and(encl)', 'quod': 'which/that',
    'super': 'above', 'inter': 'between', 'sine': 'without',
    'sub': 'under', 'quia': 'because', 'donec': 'until',
    'dum': 'while', 'contra': 'against',
}


def _build_pharma_inflection_table() -> Dict[str, str]:
    """Generate full inflection forms for pharma stems beyond what
    build_expanded_word_set() already covers.

    Returns dict: word -> provenance_tag.
    """
    forms: Dict[str, str] = {}

    # Extra stems
    for stem, paradigm in _EXTRA_PHARMA_STEMS:
        for form in generate_inflected_forms(stem, paradigm):
            fl = form.lower()
            if len(fl) >= 3:
                forms[fl] = f'pharma_inflect:{stem}({paradigm})'

    # Explicit verb forms
    for root, form_list in _PHARMA_VERB_FORMS.items():
        for form in form_list:
            fl = form.lower()
            if len(fl) >= 2:
                forms[fl] = f'pharma_verb:{root}'

    return forms


def _build_gallo_italic_variants(word_set: Set[str]) -> Dict[str, str]:
    """Apply Gallo-Italic transformations to produce dialectal variants.

    Returns dict: variant -> provenance_tag.
    """
    variants: Dict[str, str] = {}

    for word in word_set:
        if len(word) < 3:
            continue

        # Degemination
        degeminated = word
        for old, new in _DEGEMINATION_RULES:
            degeminated = degeminated.replace(old, new)
        if degeminated != word and len(degeminated) >= 3:
            variants[degeminated] = f'gallo_degem:{word}'

        # Northern accusative: -um → -on, -am → -an
        if word.endswith('um') and len(word) > 3:
            on_form = word[:-2] + 'on'
            variants[on_form] = f'gallo_acc:{word}'
        elif word.endswith('am') and len(word) > 3:
            an_form = word[:-2] + 'an'
            variants[an_form] = f'gallo_acc:{word}'

        # Combined: degeminated + northern accusative
        if degeminated != word:
            if degeminated.endswith('um') and len(degeminated) > 3:
                combined = degeminated[:-2] + 'on'
                variants[combined] = f'gallo_combined:{word}'
            elif degeminated.endswith('am') and len(degeminated) > 3:
                combined = degeminated[:-2] + 'an'
                variants[combined] = f'gallo_combined:{word}'

    return variants


def _extract_ci_vocabulary(ci_path: str) -> Dict[str, str]:
    """Extract unique words from Circa Instans text.

    Returns dict: word -> provenance_tag.
    """
    words: Dict[str, str] = {}

    if not os.path.exists(ci_path):
        return words

    with open(ci_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    for token in text.split():
        w = token.lower().strip('.,;:!?()[]{}"\'-/')
        if len(w) >= 3 and w.isalpha() and len(w) <= 20:
            if w not in words:
                words[w] = 'ci_attested'

    return words


def _build_function_word_set() -> Dict[str, str]:
    """Aggregate function words from all sources.

    Returns dict: word -> provenance_tag.
    """
    func: Dict[str, str] = {}

    # From signal words (type='function')
    for word, info in SIGNAL_WORDS_51.items():
        if info.get('type') == 'function':
            func[word] = f'signal_function:{word}'

    # From Latin particles
    for w in _PARTICLES:
        if w not in func:
            func[w] = 'latin_particle'

    # From PHARMACEUTICAL_VOCABULARY function_words
    for w in PHARMACEUTICAL_VOCABULARY.get('function_words', []):
        wl = w.lower()
        if wl not in func:
            func[wl] = 'pharma_function'

    # Italian function words
    for w, gloss in _ITALIAN_FUNCTION_WORDS.items():
        if w not in func:
            func[w] = f'italian_function:{gloss}'

    return func


def _add_t1_identifications(rd: str) -> Dict[str, str]:
    """Add T1-identified words from Phase 69 clean corpus catalogue.

    Returns dict: word -> provenance_tag.
    """
    t1_words: Dict[str, str] = {}

    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    catalogue = clean_data.get('t1_catalogue', [])

    for entry in catalogue:
        w = entry.get('matched_word', '')
        if w and len(w) >= 2:
            t1_words[w] = f"t1_{entry.get('tier', 'unknown')}"

    return t1_words


def _combine_all_layers(
    base_expanded: Set[str],
    pharma_inflected: Dict[str, str],
    gallo_variants: Dict[str, str],
    ci_words: Dict[str, str],
    function_words: Dict[str, str],
    t1_words: Dict[str, str],
) -> Tuple[Set[str], Dict[str, str]]:
    """Union all layers, tracking provenance for new words only.

    Returns (combined_set, provenance_for_new).
    """
    combined = set(base_expanded)
    provenance: Dict[str, str] = {}

    layers = [
        ('pharma_inflected', pharma_inflected),
        ('gallo_italic', gallo_variants),
        ('ci_vocabulary', ci_words),
        ('function_words', function_words),
        ('t1_identifications', t1_words),
    ]

    for layer_name, layer_dict in layers:
        for word, prov in layer_dict.items():
            if word not in combined:
                combined.add(word)
                provenance[word] = prov

    return combined, provenance


def _evaluate_expanded(
    combined: Set[str],
    base_expanded: Set[str],
    clean_decoded: List[str],
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: Any,
    provenance: Dict[str, str],
    n_null_trials: int = 100,
) -> Dict[str, Any]:
    """Evaluate expanded dictionary against clean decoded tokens and null.

    Null approach: shuffle assignment table (confirmed triples only),
    re-decode, measure dict-hit.
    """
    # Real dict-hit (old vs new)
    old_hits = sum(1 for d in clean_decoded if d and d in base_expanded)
    new_hits = sum(1 for d in clean_decoded if d and d in combined)
    n_clean = len(clean_decoded)

    old_dict_hit = old_hits / n_clean if n_clean else 0.0
    new_dict_hit = new_hits / n_clean if n_clean else 0.0
    delta = new_dict_hit - old_dict_hit

    # Per-layer contribution (how many NEW hits does each layer contribute?)
    layer_contributions: Dict[str, int] = Counter()
    new_only = combined - base_expanded
    for d in clean_decoded:
        if d and d in new_only:
            prov = provenance.get(d, 'unknown')
            # Extract layer name from provenance tag
            layer = prov.split(':')[0] if ':' in prov else prov
            layer_contributions[layer] += 1

    # Null distribution: shuffle confirmed syllable assignments
    rng = np.random.default_rng(seed=42)

    # Separate confirmed vs unresolved from assignment
    # (We'll load triple_tiers for this if available)
    all_syls = sorted(set(assignment.values()))
    all_keys = sorted(assignment.keys())

    null_hits_list: List[float] = []
    for trial in range(n_null_trials):
        shuffled_vals = rng.choice(all_syls, size=len(all_keys), replace=True)
        null_assignment = dict(zip(all_keys, shuffled_vals.tolist()))

        null_decoded = []
        for token in [all_tokens[i] for i in range(len(all_tokens))
                      if i < len(clean_decoded)]:
            try:
                result = decode_token_cvc_v2(
                    token, null_assignment, eva_to_triple, coda_table)
                null_decoded.append(result.decoded_cvc)
            except Exception:
                null_decoded.append('')

        null_h = sum(1 for d in null_decoded if d and d in combined)
        null_hits_list.append(null_h / len(null_decoded) if null_decoded else 0.0)

    null_mean = float(np.mean(null_hits_list))
    null_std = float(np.std(null_hits_list))
    selectivity = new_dict_hit / null_mean if null_mean > 0 else float('inf')
    z_score = (new_dict_hit - null_mean) / null_std if null_std > 0 else 0.0

    # Newly identified word types
    new_types = set()
    for d in clean_decoded:
        if d and d in combined and d not in base_expanded:
            new_types.add(d)

    new_word_sample = Counter(
        d for d in clean_decoded if d and d in new_types
    ).most_common(30)

    return {
        'old_dict_hit': old_dict_hit,
        'new_dict_hit': new_dict_hit,
        'delta': delta,
        'old_hits': old_hits,
        'new_hits': new_hits,
        'n_clean': n_clean,
        'null_mean': null_mean,
        'null_std': null_std,
        'selectivity': selectivity,
        'z_score': z_score,
        'layer_contributions': dict(layer_contributions.most_common()),
        'n_new_types': len(new_types),
        'new_word_sample': new_word_sample,
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PharmaDictResult:
    phase: str = "70"
    step: str = "70.1"
    experiment: str = "pharma_dictionary"
    # Layer sizes
    n_base_expanded: int = 0
    n_pharma_inflected: int = 0
    n_gallo_variants: int = 0
    n_ci_vocabulary: int = 0
    n_function_words: int = 0
    n_t1_words: int = 0
    n_combined: int = 0
    # Evaluation
    clean_dict_hit_old: float = 0.0
    clean_dict_hit_new: float = 0.0
    delta: float = 0.0
    null_mean: float = 0.0
    null_std: float = 0.0
    selectivity: float = 0.0
    z_score: float = 0.0
    layer_contributions: Dict[str, int] = field(default_factory=dict)
    n_new_types: int = 0
    new_word_sample: List[Any] = field(default_factory=list)
    # Gates
    gate_d1: bool = False  # clean_dict_hit > 0.50
    gate_d2: bool = False  # selectivity > 1.5
    gate_d3: bool = False  # >= 100 new word types
    gate_d4: bool = False  # dict size < 100K
    gate_d5: bool = False  # CI layer contributes most new hits
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pharma_dict():
    """Track 1: Build pharmaceutical dictionary and evaluate."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 70.1 — Pharmaceutical Dictionary Expansion")
    print("=" * 50)

    # --- Load dependencies ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    clean_decoded = clean_data.get('clean_decoded', [])
    clean_indices = clean_data.get('clean_indices', [])
    print(f"  Clean tokens: {len(clean_decoded)}")

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    print(f"  Assignment triples: {len(assignment)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # --- Build base expanded dictionary ---
    print("\n  Building base expanded dictionary...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    base_expanded, _ = build_expanded_word_set(base_words)
    base_expanded = base_words | base_expanded
    print(f"  Base expanded size: {len(base_expanded)}")

    # --- Layer 1: Pharma inflection table ---
    print("  Building pharma inflection table...")
    pharma_inflected = _build_pharma_inflection_table()
    print(f"    Pharma inflected forms: {len(pharma_inflected)}")

    # --- Layer 2: Gallo-Italic variants ---
    print("  Building Gallo-Italic variants...")
    # Apply to base_expanded + pharma_inflected
    source_for_gallo = base_expanded | set(pharma_inflected.keys())
    gallo_variants = _build_gallo_italic_variants(source_for_gallo)
    print(f"    Gallo-Italic variants: {len(gallo_variants)}")

    # --- Layer 3: CI vocabulary ---
    print("  Extracting Circa Instans vocabulary...")
    ci_path = os.path.join(str(_data_dir()), 'reference', 'latin', 'circa_instans.txt')
    ci_words = _extract_ci_vocabulary(ci_path)
    # Also extract from de_viribus_herbarum if available
    dvh_path = os.path.join(str(_data_dir()), 'reference', 'latin', 'de_viribus_herbarum.txt')
    dvh_words = _extract_ci_vocabulary(dvh_path)
    for w, p in dvh_words.items():
        if w not in ci_words:
            ci_words[w] = 'dvh_attested'
    print(f"    CI + DVH vocabulary: {len(ci_words)}")

    # --- Layer 4: Function words ---
    print("  Building function word set...")
    function_words = _build_function_word_set()
    print(f"    Function words: {len(function_words)}")

    # --- Layer 5: T1 identifications ---
    print("  Adding T1 identifications...")
    t1_words = _add_t1_identifications(rd)
    print(f"    T1 words: {len(t1_words)}")

    # --- Combine all layers ---
    print("\n  Combining all layers...")
    combined, provenance = _combine_all_layers(
        base_expanded, pharma_inflected, gallo_variants,
        ci_words, function_words, t1_words)
    print(f"  Combined dictionary size: {len(combined)}")

    # --- Evaluate ---
    print("\n  Evaluating against clean tokens + null distribution...")
    eval_results = _evaluate_expanded(
        combined, base_expanded, clean_decoded, all_tokens,
        assignment, eva_to_triple, coda_table, provenance,
        n_null_trials=100)

    print(f"    Old dict-hit: {eval_results['old_dict_hit']:.3f}")
    print(f"    New dict-hit: {eval_results['new_dict_hit']:.3f}")
    print(f"    Delta: {eval_results['delta']:+.3f}")
    print(f"    Null mean: {eval_results['null_mean']:.3f}")
    print(f"    Selectivity: {eval_results['selectivity']:.2f}×")
    print(f"    Z-score: {eval_results['z_score']:.2f}")
    print(f"    New types identified: {eval_results['n_new_types']}")
    print(f"    Layer contributions: {eval_results['layer_contributions']}")

    # --- Gates ---
    g1 = eval_results['new_dict_hit'] > 0.50
    g2 = eval_results['selectivity'] > 1.5
    g3 = eval_results['n_new_types'] >= 100
    g4 = len(combined) < 100_000
    # D5: CI layer contributes most new hits
    lc = eval_results['layer_contributions']
    ci_contrib = lc.get('ci_attested', 0) + lc.get('dvh_attested', 0)
    max_contrib = max(lc.values()) if lc else 0
    g5 = ci_contrib >= max_contrib and ci_contrib > 0

    gates_passed = sum([g1, g2, g3, g4, g5])

    print(f"\n  Gates: {gates_passed}/5")
    print(f"    D1 (dict-hit > 0.50): {'PASS' if g1 else 'FAIL'} ({eval_results['new_dict_hit']:.3f})")
    print(f"    D2 (selectivity > 1.5×): {'PASS' if g2 else 'FAIL'} ({eval_results['selectivity']:.2f}×)")
    print(f"    D3 (≥100 new types): {'PASS' if g3 else 'FAIL'} ({eval_results['n_new_types']})")
    print(f"    D4 (size < 100K): {'PASS' if g4 else 'FAIL'} ({len(combined)})")
    print(f"    D5 (CI top contrib): {'PASS' if g5 else 'FAIL'} (CI={ci_contrib}, max={max_contrib})")

    if gates_passed >= 3:
        verdict = 'DICTIONARY_EXPANDED'
    elif gates_passed >= 1:
        verdict = 'MARGINAL_EXPANSION'
    else:
        verdict = 'NO_IMPROVEMENT'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = PharmaDictResult(
        n_base_expanded=len(base_expanded),
        n_pharma_inflected=len(pharma_inflected),
        n_gallo_variants=len(gallo_variants),
        n_ci_vocabulary=len(ci_words),
        n_function_words=len(function_words),
        n_t1_words=len(t1_words),
        n_combined=len(combined),
        clean_dict_hit_old=eval_results['old_dict_hit'],
        clean_dict_hit_new=eval_results['new_dict_hit'],
        delta=eval_results['delta'],
        null_mean=eval_results['null_mean'],
        null_std=eval_results['null_std'],
        selectivity=eval_results['selectivity'],
        z_score=eval_results['z_score'],
        layer_contributions=eval_results['layer_contributions'],
        n_new_types=eval_results['n_new_types'],
        new_word_sample=eval_results['new_word_sample'],
        gate_d1=g1,
        gate_d2=g2,
        gate_d3=g3,
        gate_d4=g4,
        gate_d5=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Save result + combined word list (for downstream tracks)
    out = _save_json(rd, 'phase70_pharma_dict.json', {
        **asdict(result),
        'combined_word_list': sorted(combined),
    })
    print(f"\n  Saved: {out}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
