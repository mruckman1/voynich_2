"""
Phase 61, Track A: Deep Pharmaceutical Recipe Reading
======================================================
Selects the 5 pharmaceutically richest recipes, produces 6-layer deep
annotation (EVA / CVC / segments / gloss / deep interpretation / confidence),
applies concatenation recognition, declension analysis, CI recipe template
matching, and generates human-readable reading attempts.

Extends Phase 60 Track D's automated glossing into actual reading.

Dependency chain:
    results/recipe_annotation.json    (Phase 60D)
    results/corrected_coda.json       (Phase 60A)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
    data/reference/latin/circa_instans.txt
        -> results/phase61_deep_recipes.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import data_dir, results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
    decode_token_cvc_v2,
)
from voynich.phases.cvc_recipes import (
    PHARMA_VOCAB,
    extract_recipes,
    find_cvc_boundary_markers,
)
from voynich.phases.cvc_segmentation import (
    _load_segmentation_inventory,
    segment_decoded_word,
)
from voynich.phases.recipe_annotation import (
    CONNECTORS,
    ENDING_GLOSSES,
    INGREDIENTS,
    QUALITIES,
    QUANTITIES,
    VERBS,
    _build_annotation_vocab,
    _annotate_token,
    _classify_role,
)


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
# Extended vocabulary for deep reading
# ---------------------------------------------------------------------------

# T1 identifications (words confirmed by word-level permutation test)
T1_IDENTIFICATIONS = {
    'ratione': 'by method/reason',
    'coralli': 'coral(s)',
    'diasene': 'senna compound',
    'stercora': 'medicinal dung',
    'radicom': 'root (acc.)',
    'commune': 'common',
    'secundi': 'of the second',
    'balsamo': 'balsam',
    'decoctum': 'decoction',
}

# Extended pharmaceutical vocabulary for concatenation matching
PHARMA_EXTENDED = {
    'aqua': 'water', 'herba': 'herb', 'radice': 'root', 'semen': 'seed',
    'cortex': 'bark', 'folia': 'leaves', 'flores': 'flowers', 'oleum': 'oil',
    'pulvis': 'powder', 'succo': 'juice', 'cera': 'wax', 'mel': 'honey',
    'vinum': 'wine', 'acetum': 'vinegar', 'morbo': 'disease', 'dolor': 'pain',
    'febre': 'fever', 'cura': 'cure', 'sana': 'healthy', 'calida': 'hot',
    'frigida': 'cold', 'siccus': 'dry', 'humidus': 'moist',
    'pannum': 'cloth', 'mortario': 'mortar', 'patella': 'dish',
    'decoctione': 'by decoction', 'infusum': 'infusion',
    'electuarium': 'electuary', 'pilula': 'pill', 'sirupus': 'syrup',
    'unguentum': 'ointment', 'emplastrum': 'plaster',
}

# Declension map for CVC coda endings
DECLENSION_MAP = {
    'en': {'case': 'acc/abl.3rd', 'example': 'herbam→herben'},
    'in': {'case': 'prep/loc', 'example': 'in aqua'},
    'an': {'case': 'acc.1st', 'example': 'herbam→herban'},
    'on': {'case': 'acc.2nd', 'example': 'vinum→vinon'},
    'un': {'case': 'acc.2nd', 'example': 'fructum→fructun'},
    'er': {'case': 'agent/comp', 'example': '-tor, -ter'},
    'ar': {'case': 'adj', 'example': '-aris'},
    'or': {'case': 'agent', 'example': '-tor, -sor'},
    'es': {'case': 'nom.pl', 'example': 'radices'},
    'is': {'case': 'gen.sg', 'example': 'radicis'},
}

# CI recipe templates
CI_TEMPLATES = [
    {
        'name': 'simple_decoction',
        'slots': ['VERB', 'INGREDIENT', 'PREP', 'MEDIUM'],
        'verb_set': {'coque', 'decoque', 'cola', 'colar'},
        'prep_set': {'in', 'co', 'con', 'de', 'per'},
        'example': 'decoque radicem in aqua',
        'translation': 'boil the root in water',
    },
    {
        'name': 'straining',
        'slots': ['VERB', 'PREP', 'FILTER'],
        'verb_set': {'cola', 'colar', 'colat'},
        'prep_set': {'per', 'de'},
        'example': 'cola per pannum',
        'translation': 'strain through cloth',
    },
    {
        'name': 'grinding',
        'slots': ['VERB', 'INGREDIENT', 'PREP', 'TOOL'],
        'verb_set': {'tere', 'terer', 'contere'},
        'prep_set': {'in', 'co', 'con'},
        'example': 'tere in mortario',
        'translation': 'grind in a mortar',
    },
    {
        'name': 'mixing',
        'slots': ['VERB', 'INGREDIENT', 'PREP', 'INGREDIENT'],
        'verb_set': {'misce', 'miser', 'adde'},
        'prep_set': {'co', 'con', 'cu'},
        'example': 'misce cum melle',
        'translation': 'mix with honey',
    },
    {
        'name': 'dosage',
        'slots': ['VERB', 'QUANTITY', 'PREP', 'VEHICLE'],
        'verb_set': {'recipe', 'bibe', 'da', 'pone'},
        'prep_set': {'co', 'con', 'cu', 'in', 'de'},
        'example': 'da cum aqua calida',
        'translation': 'give with warm water',
    },
    {
        'name': 'compound_naming',
        'slots': ['PREFIX', 'INGREDIENT'],
        'prefix_set': {'dia'},
        'example': 'diasene',
        'translation': 'compound of senna',
    },
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DeepAnnotation:
    """6-layer annotation for a single token."""
    position: int
    eva: str
    decoded: str
    segments: List[str]
    basic_gloss: str
    gloss_source: str
    structural_role: str
    # Deep interpretation
    concatenations: List[Dict[str, Any]] = field(default_factory=list)
    declension: str = ""
    case_gloss: str = ""
    formula_role: str = ""
    ci_reference: str = ""
    confidence: str = "LOW"


@dataclass
class MergedUnit:
    """A merged reading unit (possibly spanning multiple tokens)."""
    positions: List[int]
    eva: str
    decoded: str
    word: str
    merge_type: str    # 'MERGED' or 'SINGLE'
    confidence: str


@dataclass
class RecipeReading:
    """Complete reading of one recipe."""
    recipe_idx: int
    folio: str
    n_tokens: int
    n_eva_tokens: int
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    merged: List[Dict[str, Any]] = field(default_factory=list)
    n_concatenations: int = 0
    template_matches: List[Dict[str, Any]] = field(default_factory=list)
    reading_text: str = ""
    reading_confidence: float = 0.0
    glossed_fraction: float = 0.0
    high_conf_fraction: float = 0.0
    pharma_terms: List[str] = field(default_factory=list)
    declension_tokens: List[str] = field(default_factory=list)
    formatted_display: str = ""


@dataclass
class DeepRecipeResult:
    phase: str = "61"
    step: str = "61.1"
    experiment: str = "deep_recipe_reading"
    n_candidates: int = 0
    n_selected: int = 0
    recipes: List[Dict[str, Any]] = field(default_factory=list)
    n_with_verb: int = 0
    n_with_ingredient: int = 0
    n_with_template: int = 0
    n_with_concatenation: int = 0
    mean_reading_confidence: float = 0.0
    ci_entries_found: List[str] = field(default_factory=list)
    # Gates
    g1_verbs: bool = False          # >= 3/5 with pharma verb
    g2_ingredients: bool = False    # >= 2/5 with ingredient
    g3_template: bool = False       # >= 1 CI template match >= 0.5
    g4_concatenation: bool = False  # >= 1 concatenation produces T1 word
    g5_confidence: bool = False     # mean reading confidence >= 0.3
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Circa Instans loader
# ---------------------------------------------------------------------------

def _load_circa_instans() -> Dict[str, List[str]]:
    """Load Circa Instans text and extract ingredient entries."""
    ci_path = os.path.join(str(data_dir('reference/latin')), 'circa_instans.txt')
    if not os.path.exists(ci_path):
        return {}

    entries: Dict[str, List[str]] = {}
    try:
        with open(ci_path, encoding='utf-8', errors='replace') as f:
            text = f.read()

        # Split into chunks and extract keywords
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            pl = para.lower()
            # Check for known ingredient names
            for ingredient in T1_IDENTIFICATIONS:
                if ingredient in pl:
                    entries.setdefault(ingredient, []).append(
                        para[:200].strip()
                    )
            for ingredient in PHARMA_EXTENDED:
                if ingredient in pl:
                    entries.setdefault(ingredient, []).append(
                        para[:200].strip()
                    )
    except Exception:
        pass

    return entries


# ---------------------------------------------------------------------------
# Recipe selection
# ---------------------------------------------------------------------------

def _select_deep_reading_candidates(
    recipe_annotation_data: Dict,
    n: int = 5,
) -> List[Dict[str, Any]]:
    """Select the n best recipes for deep reading by pharmaceutical richness."""
    top_recipes = recipe_annotation_data.get('top_recipes', [])
    if not top_recipes:
        return []

    scored = []
    for recipe in top_recipes:
        # Phase 60D stores tokens under 'tokens' key (not 'annotations')
        tokens = recipe.get('tokens', recipe.get('annotations', []))
        if not tokens:
            continue

        n_tokens = len(tokens)
        if n_tokens < 5:
            continue

        cvc_tokens = [t.get('decoded', '').lower() for t in tokens]
        glossed_frac = recipe.get('glossed_fraction', 0)

        has_verb = any(t in VERBS for t in cvc_tokens)
        has_ingredient = any(t in INGREDIENTS for t in cvc_tokens)
        has_measure = any(t in QUANTITIES for t in cvc_tokens)
        has_quality = any(t in QUALITIES for t in cvc_tokens)

        pharma_score = (
            2.0 * has_verb +
            2.0 * has_ingredient +
            1.0 * has_measure +
            1.0 * has_quality +
            0.5 * min(n_tokens / 20.0, 1.0) +
            1.0 * glossed_frac
        )

        scored.append({
            **recipe,
            'pharma_score': pharma_score,
            'has_verb': has_verb,
            'has_ingredient': has_ingredient,
            'has_measure': has_measure,
            'has_quality': has_quality,
        })

    scored.sort(key=lambda r: -r['pharma_score'])
    return scored[:n]


# ---------------------------------------------------------------------------
# Deep annotation
# ---------------------------------------------------------------------------

def _deep_annotate_recipe(
    recipe: Dict[str, Any],
    vocab: Dict[str, Dict[str, str]],
    costamagna_inv: Set[str],
    ci_entries: Dict[str, List[str]],
) -> List[DeepAnnotation]:
    """Produce 6-layer deep annotation for a recipe."""
    # Phase 60D stores tokens under 'tokens' key (not 'annotations')
    tokens = recipe.get('tokens', recipe.get('annotations', []))
    if not tokens:
        return []

    n = len(tokens)
    deep_annots = []

    for idx in range(n):
        ann = tokens[idx]
        eva = ann.get('eva', '')
        decoded = ann.get('decoded', '').lower()
        segments = ann.get('segments', [])
        basic_gloss = ann.get('gloss', '?')
        gloss_source = ann.get('gloss_source', ann.get('confidence', 'none'))
        role = ann.get('structural_role', ann.get('role', 'UNKNOWN'))

        # Deep: concatenation check
        concat_matches = []
        for ahead in range(1, min(4, n - idx)):
            combined = decoded
            for k in range(1, ahead + 1):
                combined += tokens[idx + k].get('decoded', '').lower()
            # Check T1 identifications
            if combined in T1_IDENTIFICATIONS:
                concat_matches.append({
                    'tokens_combined': ahead + 1,
                    'combined_string': combined,
                    'word': T1_IDENTIFICATIONS[combined],
                    'confidence': 'HIGH',
                })
            elif combined in PHARMA_EXTENDED:
                concat_matches.append({
                    'tokens_combined': ahead + 1,
                    'combined_string': combined,
                    'word': PHARMA_EXTENDED[combined],
                    'confidence': 'MEDIUM',
                })

        # Deep: declension analysis
        declension = ""
        case_gloss = ""
        if len(decoded) >= 3:
            ending = decoded[-2:]
            if ending in DECLENSION_MAP:
                declension = DECLENSION_MAP[ending]['case']
                case_gloss = DECLENSION_MAP[ending]['example']

        # Deep: formula role
        formula_role = ""
        if idx == 0 and decoded in VERBS:
            formula_role = 'RECIPE_VERB (imperative opening)'
        elif decoded in CONNECTORS:
            formula_role = f'PREPOSITION ({basic_gloss})'
        elif role == 'INGREDIENT':
            if declension and 'acc' in declension:
                formula_role = 'DIRECT_OBJECT (ingredient in accusative)'
            else:
                formula_role = 'INGREDIENT'

        # Deep: CI cross-reference
        ci_ref = ""
        if decoded in ci_entries:
            ci_ref = ci_entries[decoded][0][:100]
        for concat in concat_matches:
            cs = concat['combined_string']
            if cs in ci_entries:
                ci_ref = ci_entries[cs][0][:100]

        # Confidence
        if basic_gloss != '?' and concat_matches:
            confidence = 'HIGH'
        elif basic_gloss != '?':
            confidence = 'MEDIUM'
        elif concat_matches:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        deep_annots.append(DeepAnnotation(
            position=idx,
            eva=eva,
            decoded=decoded,
            segments=segments if isinstance(segments, list) else [],
            basic_gloss=basic_gloss,
            gloss_source=gloss_source,
            structural_role=role,
            concatenations=concat_matches,
            declension=declension,
            case_gloss=case_gloss,
            formula_role=formula_role,
            ci_reference=ci_ref,
            confidence=confidence,
        ))

    return deep_annots


# ---------------------------------------------------------------------------
# Merge concatenations
# ---------------------------------------------------------------------------

def _merge_concatenations(
    deep_annots: List[DeepAnnotation],
) -> List[MergedUnit]:
    """Greedily merge consecutive tokens via concatenation matches."""
    merged = []
    consumed: Set[int] = set()
    n = len(deep_annots)

    for idx in range(n):
        if idx in consumed:
            continue

        ann = deep_annots[idx]

        # Find longest concatenation match
        best_concat = None
        for concat in sorted(ann.concatenations,
                             key=lambda c: -c['tokens_combined']):
            span = range(idx, idx + concat['tokens_combined'])
            if not any(i in consumed for i in span) and all(i < n for i in span):
                best_concat = concat
                break

        if best_concat:
            span = list(range(idx, idx + best_concat['tokens_combined']))
            merged_eva = ' '.join(deep_annots[i].eva for i in span)
            merged.append(MergedUnit(
                positions=span,
                eva=merged_eva,
                decoded=best_concat['combined_string'],
                word=best_concat['word'],
                merge_type='MERGED',
                confidence=best_concat['confidence'],
            ))
            for i in span:
                consumed.add(i)
        else:
            merged.append(MergedUnit(
                positions=[idx],
                eva=ann.eva,
                decoded=ann.decoded,
                word=ann.basic_gloss,
                merge_type='SINGLE',
                confidence=ann.confidence,
            ))
            consumed.add(idx)

    return merged


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------

def _match_templates(
    deep_annots: List[DeepAnnotation],
    merged: List[MergedUnit],
) -> List[Dict[str, Any]]:
    """Match recipe against CI templates."""
    matches = []
    tokens = [a.decoded for a in deep_annots]
    roles = [a.structural_role for a in deep_annots]

    for template in CI_TEMPLATES:
        slots = template['slots']
        n_slots = len(slots)
        matched_slots = 0

        for slot in slots:
            if slot == 'VERB':
                if any(t in template.get('verb_set', set()) for t in tokens):
                    matched_slots += 1
            elif slot == 'INGREDIENT':
                if any(r == 'INGREDIENT' for r in roles):
                    matched_slots += 1
            elif slot == 'PREP':
                if any(t in template.get('prep_set', set()) for t in tokens):
                    matched_slots += 1
            elif slot == 'QUANTITY':
                if any(r == 'QUANTITY' for r in roles):
                    matched_slots += 1
            elif slot in ('MEDIUM', 'FILTER', 'TOOL', 'VEHICLE'):
                # These are typically nouns after a preposition
                for i in range(len(tokens) - 1):
                    if tokens[i] in template.get('prep_set', set()):
                        if roles[i + 1] in ('INGREDIENT', 'UNKNOWN'):
                            matched_slots += 1
                            break
            elif slot == 'PREFIX':
                if any(t.startswith('dia') for t in tokens):
                    matched_slots += 1

        score = matched_slots / n_slots if n_slots > 0 else 0.0
        if score >= 0.4:
            matches.append({
                'template': template['name'],
                'score': round(score, 2),
                'matched_slots': matched_slots,
                'total_slots': n_slots,
                'example': template['example'],
                'translation': template['translation'],
            })

    return sorted(matches, key=lambda m: -m['score'])


# ---------------------------------------------------------------------------
# Reading generation
# ---------------------------------------------------------------------------

def _generate_reading(
    deep_annots: List[DeepAnnotation],
    merged: List[MergedUnit],
    folio: str,
) -> Tuple[str, float]:
    """Generate a human-readable reading attempt."""
    parts = []
    n_glossed = 0

    for unit in merged:
        if unit.word and unit.word != '?':
            parts.append(unit.word)
            n_glossed += 1
        else:
            parts.append(f'[{unit.decoded}]')

    reading = ' '.join(parts)
    confidence = n_glossed / len(merged) if merged else 0.0
    return reading, round(confidence, 3)


def _format_recipe_display(
    recipe_idx: int,
    folio: str,
    deep_annots: List[DeepAnnotation],
    merged: List[MergedUnit],
    reading: str,
    template_matches: List[Dict[str, Any]],
) -> str:
    """Produce human-readable multi-line display."""
    lines = []
    lines.append(f"RECIPE #{recipe_idx} (folio {folio}, {len(deep_annots)} tokens)")
    lines.append("=" * 60)

    # EVA line
    evas = [a.eva for a in deep_annots]
    lines.append("EVA:  " + "  ".join(f"{e:>8}" for e in evas[:12]))

    # CVC line
    cvcs = [a.decoded for a in deep_annots]
    lines.append("CVC:  " + "  ".join(f"{c:>8}" for c in cvcs[:12]))

    # Gloss line
    glosses = [a.basic_gloss for a in deep_annots]
    lines.append("Gloss:" + "  ".join(f"{g:>8}" for g in glosses[:12]))

    # Role line
    roles = [a.structural_role[:4] for a in deep_annots]
    lines.append("Role: " + "  ".join(f"{r:>8}" for r in roles[:12]))

    if len(deep_annots) > 12:
        lines.append("  ... (+ {} more tokens)".format(len(deep_annots) - 12))

    # Concatenations
    concats = [(a.decoded, a.concatenations) for a in deep_annots if a.concatenations]
    if concats:
        lines.append("\nConcatenations found:")
        for token, cats in concats:
            for c in cats:
                lines.append(f"  {c['combined_string']} = {c['word']} ({c['confidence']})")

    # Template matches
    if template_matches:
        lines.append("\nCI template matches:")
        for tm in template_matches[:3]:
            lines.append(f"  {tm['template']} (score={tm['score']}): "
                         f"{tm['example']} = \"{tm['translation']}\"")

    # Reading
    lines.append(f"\nREADING ATTEMPT:")
    lines.append(f"  {reading}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_deep_recipes():
    t0 = time.time()
    print("=" * 70)
    print("Phase 61, Track A: Deep Pharmaceutical Recipe Reading")
    print("=" * 70)

    rd = str(_results_dir())

    # Load Phase 60 recipe annotation
    print("\n  Loading Phase 60 recipe annotations ...")
    ra_data = _safe_load(os.path.join(rd, 'recipe_annotation.json'))
    if not ra_data:
        print("  ERROR: recipe_annotation.json not found. Run Phase 60D first.")
        result = DeepRecipeResult(runtime_seconds=round(time.time() - t0, 1))
        _save_json(rd, 'phase61_deep_recipes.json', result)
        return

    # Load reference data
    print("  Loading reference data ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    vocab = _build_annotation_vocab(ref_word_set)

    costamagna_inv, syl_to_struct = _load_segmentation_inventory()

    # Load Circa Instans
    print("  Loading Circa Instans ...")
    ci_entries = _load_circa_instans()
    print(f"  CI entries with known ingredients: {len(ci_entries)}")

    # Select best recipes
    print("\n  1. Selecting 5 best recipes for deep reading ...")
    candidates = _select_deep_reading_candidates(ra_data, n=5)
    n_candidates = len(ra_data.get('top_recipes', []))
    print(f"     {n_candidates} total recipes, {len(candidates)} selected")

    if not candidates:
        print("  WARNING: No suitable recipes found.")
        result = DeepRecipeResult(
            n_candidates=n_candidates,
            runtime_seconds=round(time.time() - t0, 1),
        )
        _save_json(rd, 'phase61_deep_recipes.json', result)
        return

    for i, c in enumerate(candidates):
        folio = c.get('folio', '?')
        n_tok = len(c.get('tokens', c.get('annotations', [])))
        gf = c.get('glossed_fraction', 0)
        ps = c.get('pharma_score', 0)
        print(f"     #{i}: {folio}, {n_tok} tokens, "
              f"glossed={gf:.1%}, pharma_score={ps:.1f}")

    # Deep annotate each recipe
    print("\n  2. Deep annotation + concatenation + templates ...")
    recipe_readings: List[RecipeReading] = []
    n_with_verb = 0
    n_with_ingredient = 0
    n_with_template = 0
    n_with_concat = 0
    all_ci_found: Set[str] = set()

    for idx, recipe in enumerate(candidates):
        folio = recipe.get('folio', '?')
        recipe_tokens = recipe.get('tokens', recipe.get('annotations', []))
        n_tok = len(recipe_tokens)

        # Deep annotate
        deep_annots = _deep_annotate_recipe(recipe, vocab, costamagna_inv, ci_entries)

        # Merge concatenations
        merged = _merge_concatenations(deep_annots)
        n_concats = sum(1 for m in merged if m.merge_type == 'MERGED')

        # Template matching
        template_matches = _match_templates(deep_annots, merged)

        # Reading
        reading, confidence = _generate_reading(deep_annots, merged, folio)

        # Format display
        display = _format_recipe_display(
            idx, folio, deep_annots, merged, reading, template_matches,
        )

        # Gather metrics
        has_verb = any(a.structural_role == 'VERB' for a in deep_annots)
        has_ingredient = any(a.structural_role == 'INGREDIENT' for a in deep_annots)
        has_template = len(template_matches) > 0 and template_matches[0]['score'] >= 0.5
        has_concat = n_concats > 0

        if has_verb:
            n_with_verb += 1
        if has_ingredient:
            n_with_ingredient += 1
        if has_template:
            n_with_template += 1
        if has_concat:
            n_with_concat += 1

        # CI references
        for a in deep_annots:
            if a.ci_reference:
                all_ci_found.add(a.decoded)

        # Pharma terms and declensions
        pharma = [a.decoded for a in deep_annots
                  if a.decoded in PHARMA_EXTENDED or a.decoded in T1_IDENTIFICATIONS]
        decl_tokens = [f"{a.decoded} ({a.declension})" for a in deep_annots
                       if a.declension]

        glossed_frac = sum(1 for m in merged if m.word != '?') / len(merged) if merged else 0.0
        high_conf_frac = sum(1 for a in deep_annots
                             if a.confidence == 'HIGH') / len(deep_annots) if deep_annots else 0.0

        rr = RecipeReading(
            recipe_idx=idx,
            folio=folio,
            n_tokens=n_tok,
            n_eva_tokens=n_tok,
            annotations=[_convert(asdict(a)) for a in deep_annots],
            merged=[_convert(asdict(m)) for m in merged],
            n_concatenations=n_concats,
            template_matches=template_matches,
            reading_text=reading,
            reading_confidence=confidence,
            glossed_fraction=round(glossed_frac, 3),
            high_conf_fraction=round(high_conf_frac, 3),
            pharma_terms=pharma,
            declension_tokens=decl_tokens,
            formatted_display=display,
        )
        recipe_readings.append(rr)

        print(f"\n     Recipe #{idx} ({folio}):")
        print(f"       Tokens: {n_tok}, Concatenations: {n_concats}")
        print(f"       Templates: {len(template_matches)} "
              f"(best={template_matches[0]['score']:.2f} {template_matches[0]['template']}"
              if template_matches else "       Templates: 0")
        print(f"       Reading confidence: {confidence:.3f}")
        print(f"       {display[:200]}...")

    # Compute aggregate metrics
    mean_conf = (sum(r.reading_confidence for r in recipe_readings)
                 / len(recipe_readings) if recipe_readings else 0.0)

    # Gates
    g1 = n_with_verb >= 3
    g2 = n_with_ingredient >= 2
    g3 = n_with_template >= 1
    g4 = n_with_concat >= 1
    g5 = mean_conf >= 0.3
    gates = sum([g1, g2, g3, g4, g5])

    result = DeepRecipeResult(
        n_candidates=n_candidates,
        n_selected=len(candidates),
        recipes=[_convert(asdict(r)) for r in recipe_readings],
        n_with_verb=n_with_verb,
        n_with_ingredient=n_with_ingredient,
        n_with_template=n_with_template,
        n_with_concatenation=n_with_concat,
        mean_reading_confidence=round(mean_conf, 3),
        ci_entries_found=sorted(all_ci_found),
        g1_verbs=g1,
        g2_ingredients=g2,
        g3_template=g3,
        g4_concatenation=g4,
        g5_confidence=g5,
        gates_passed=gates,
        gate_passed=gates >= 3,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'phase61_deep_recipes.json', result)

    # Summary
    print("\n" + "=" * 70)
    print("  TRACK A SUMMARY: Deep Pharmaceutical Recipe Reading")
    print("=" * 70)
    print(f"  Recipes selected:        {result.n_selected}")
    print(f"  With verb:               {n_with_verb}/5")
    print(f"  With ingredient:         {n_with_ingredient}/5")
    print(f"  With CI template match:  {n_with_template}/5")
    print(f"  With concatenation:      {n_with_concat}/5")
    print(f"  Mean reading confidence: {mean_conf:.3f}")
    print(f"  CI ingredients found:    {sorted(all_ci_found)}")
    print(f"\n  Gates: {gates}/5 passed")
    print(f"    G1 (>=3 with verb):      {g1}")
    print(f"    G2 (>=2 with ingred):    {g2}")
    print(f"    G3 (>=1 template>=0.5):  {g3}")
    print(f"    G4 (>=1 concatenation):  {g4}")
    print(f"    G5 (mean conf>=0.3):     {g5}")

    # Print formatted readings
    for rr in recipe_readings:
        print(f"\n{'─' * 60}")
        print(rr.formatted_display)

    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
