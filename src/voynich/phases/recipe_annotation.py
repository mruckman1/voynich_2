"""
Phase 60, Track D: Recipe Annotation + Reading Attempts
=======================================================
Selects the top 10 CVC recipes by glossed fraction, produces 4-layer
annotation (EVA / CVC decoded / segments / gloss), classifies structural
roles (VERB, INGREDIENT, QUANTITY, QUALITY, CONNECTOR, UNKNOWN), and
cross-references against pharmaceutical vocabulary.

This is the project's first attempt at reading connected text.

Dependency chain:
    results/corrected_coda.json       (Track A)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
        -> results/recipe_annotation.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import build_coda_table
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_corpus_cvc_v2,
    decode_token_cvc_v2,
)
from voynich.phases.cvc_coda_signal import _build_folio_list
from voynich.phases.cvc_recipes import (
    CV_BOUNDARY_MARKERS,
    PHARMA_VOCAB,
    extract_recipes,
    find_cvc_boundary_markers,
    score_recipes,
)
from voynich.phases.cvc_segmentation import (
    _load_segmentation_inventory,
    segment_decoded_word,
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
# Annotation vocabulary
# ---------------------------------------------------------------------------

# Structural role vocabularies
VERBS = {
    'cola', 'colar', 'colat', 'tere', 'terer', 'misce', 'miser',
    'recipe', 'adde', 'coque', 'pone', 'bibe', 'solve', 'distilla',
}

INGREDIENTS = {
    'sene', 'senen', 'sener', 'coralli', 'stercora', 'radicom',
    'diasene', 'sal', 'mel', 'cer', 'cera', 'aqua', 'herba',
    'radice', 'semen', 'cortex', 'folia', 'flores', 'balsamo',
    'gummi', 'oleum', 'pulvis', 'succo',
}

QUANTITIES = {
    'bes', 'ses', 'din', 'bis', 'ter', 'semi', 'duo', 'tres',
}

QUALITIES = {
    'bene', 'benen', 'bon', 'fort', 'decor', 'nera', 'bela',
    'sana', 'bona', 'calda', 'fredda', 'nova',
}

CONNECTORS = {
    'di', 'de', 'co', 'con', 'ne', 'se', 'in', 'la', 'li',
    'si', 'ni', 'ci', 'te', 'ti', 'da', 'du', 'ad', 'et',
    'le', 'lo', 'cu', 'ce', 'per', 'non',
}

# Latin declension ending glosses
ENDING_GLOSSES = {
    'en': 'acc/abl.3rd',
    'in': 'prep/loc',
    'an': 'acc.1st',
    'on': 'acc.2nd',
    'un': 'acc.2nd',
    'er': 'agent/comp',
    'ar': 'adj',
    'or': 'agent/quality',
    'es': 'nom.pl',
    'is': 'gen.sg',
    'us': 'nom.2nd',
    'um': 'acc.2nd',
    'am': 'acc.1st',
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TokenAnnotation:
    """4-layer annotation for a single token."""
    position: int
    eva: str
    decoded: str
    segments: List[str]
    seg_attested: List[bool]
    gloss: str
    gloss_source: str        # 'signal', 'pharma', 'function', 'segmented', 'none'
    structural_role: str     # VERB, INGREDIENT, QUANTITY, QUALITY, CONNECTOR, UNKNOWN
    confidence: str          # HIGH, MEDIUM, LOW


@dataclass
class AnnotatedRecipe:
    """Full annotation of one recipe."""
    recipe_idx: int
    folio: str
    n_tokens: int
    glossed_fraction: float
    annotations: List[TokenAnnotation] = field(default_factory=list)
    role_distribution: Dict[str, int] = field(default_factory=dict)
    reading_attempt: str = ""
    pharma_matches: List[str] = field(default_factory=list)
    formatted_display: str = ""


@dataclass
class RecipeAnnotationResult:
    """Full Track D output."""
    phase: str = "60"
    step: str = "60.4"
    experiment: str = "recipe_annotation"
    n_recipes_annotated: int = 0
    mean_glossed_fraction: float = 0.0
    n_with_verb_ingredient: int = 0
    n_with_pharma_match: int = 0
    max_consecutive_glossed: int = 0
    ingredient_inventory: List[str] = field(default_factory=list)
    verb_inventory: List[str] = field(default_factory=list)
    top_recipes: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_glossed_gt40: bool = False       # >= 5 recipes with glossed > 40%
    g2_verb_ingredient: bool = False    # >= 3 recipes with VERB + INGREDIENT
    g3_pharma_match: bool = False       # >= 1 pharma cross-reference
    g4_mean_glossed: bool = False       # mean glossed > 35%
    g5_consecutive: bool = False        # >= 1 recipe with 5+ consecutive glossed
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Annotation vocabulary builder
# ---------------------------------------------------------------------------

def _build_annotation_vocab(ref_word_set: Set[str]) -> Dict[str, Dict[str, str]]:
    """Build merged annotation vocabulary from all available sources."""
    vocab: Dict[str, Dict[str, str]] = {}

    # Load signal words if available
    try:
        from voynich.phases.suffix_calibration import SIGNAL_WORDS_51
        for word, info in SIGNAL_WORDS_51.items():
            vocab[word] = {
                'gloss': info.get('gloss', word),
                'source': 'signal',
                'type': info.get('type', 'unknown'),
            }
    except (ImportError, AttributeError):
        pass

    # Pharma vocabulary
    pharma_set = set(PHARMA_VOCAB.keys()) if isinstance(PHARMA_VOCAB, dict) else set(PHARMA_VOCAB)
    for term in pharma_set:
        if term not in vocab:
            vocab[term] = {
                'gloss': term,
                'source': 'pharma',
                'type': 'pharma',
            }

    # Role-based glosses
    role_glosses = {
        'cola': 'strain', 'colar': 'strain(inf)', 'tere': 'grind',
        'terer': 'grind(inf)', 'misce': 'mix', 'miser': 'mix(inf)',
        'recipe': 'take', 'adde': 'add', 'coque': 'cook',
        'pone': 'place', 'bibe': 'drink', 'solve': 'dissolve',
        'di': 'of', 'de': 'of/from', 'in': 'in', 'con': 'with',
        'se': 'if/self', 'ne': 'not/nor', 'la': 'the(f)',
        'li': 'the(pl)', 'si': 'if', 'et': 'and', 'ad': 'to',
        'per': 'through', 'non': 'not',
        'bene': 'well', 'bon': 'good', 'fort': 'strong',
        'bela': 'beautiful', 'sana': 'healthy',
        'sene': 'senna', 'coralli': 'coral', 'mel': 'honey',
        'sal': 'salt', 'cer': 'wax', 'cera': 'wax',
        'aqua': 'water', 'oleum': 'oil', 'herba': 'herb',
        'radice': 'root', 'semen': 'seed',
        'din': 'daily', 'cor': 'heart', 'ser': 'serum',
        'decor': 'beauty/grace',
    }
    for word, gloss in role_glosses.items():
        if word not in vocab:
            vocab[word] = {
                'gloss': gloss,
                'source': 'role_vocab',
                'type': 'known',
            }
        elif vocab[word]['gloss'] == word:
            vocab[word]['gloss'] = gloss

    return vocab


# ---------------------------------------------------------------------------
# Token annotation
# ---------------------------------------------------------------------------

def _classify_role(decoded: str) -> str:
    """Classify a decoded token's structural role."""
    dl = decoded.lower()
    if dl in VERBS:
        return 'VERB'
    if dl in INGREDIENTS:
        return 'INGREDIENT'
    if dl in QUANTITIES:
        return 'QUANTITY'
    if dl in QUALITIES:
        return 'QUALITY'
    if dl in CONNECTORS:
        return 'CONNECTOR'
    return 'UNKNOWN'


def _annotate_token(
    position: int,
    eva_token: str,
    decoded: str,
    costamagna_inv: Set[str],
    vocab: Dict[str, Dict[str, str]],
) -> TokenAnnotation:
    """Produce 4-layer annotation for a single token."""
    # Segment
    segments = segment_decoded_word(decoded, costamagna_inv)
    seg_texts = [s['text'] for s in segments]
    seg_attested = [s['attested'] for s in segments]

    # Gloss lookup
    dl = decoded.lower()
    gloss = '?'
    gloss_source = 'none'

    if dl in vocab:
        gloss = vocab[dl]['gloss']
        gloss_source = vocab[dl]['source']
    else:
        # Try each segment
        seg_glosses = []
        any_glossed = False
        for s in segments:
            st = s['text'].lower()
            if st in vocab:
                seg_glosses.append(vocab[st]['gloss'])
                any_glossed = True
            else:
                seg_glosses.append(st)
        if any_glossed:
            gloss = '+'.join(seg_glosses)
            gloss_source = 'segmented'

    # Structural role
    role = _classify_role(decoded)

    # Confidence
    if gloss_source == 'signal':
        confidence = 'HIGH'
    elif gloss_source in ('pharma', 'role_vocab'):
        confidence = 'MEDIUM'
    elif gloss_source == 'segmented':
        confidence = 'LOW'
    else:
        confidence = 'LOW'

    return TokenAnnotation(
        position=position,
        eva=eva_token,
        decoded=decoded,
        segments=seg_texts,
        seg_attested=seg_attested,
        gloss=gloss,
        gloss_source=gloss_source,
        structural_role=role,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Recipe formatting
# ---------------------------------------------------------------------------

def _format_recipe(recipe_idx: int, folio: str, annotations: List[TokenAnnotation]) -> str:
    """Produce human-readable multi-line recipe display."""
    lines = []
    lines.append(f"RECIPE #{recipe_idx+1}: {folio} "
                 f"({len(annotations)} tokens, "
                 f"{sum(1 for a in annotations if a.gloss != '?')}/{len(annotations)} glossed)")
    lines.append("-" * 70)

    # Build aligned columns
    eva_row = []
    cvc_row = []
    seg_row = []
    gloss_row = []
    role_row = []

    for ann in annotations:
        width = max(
            len(ann.eva), len(ann.decoded),
            len('|'.join(ann.segments)),
            len(ann.gloss[:12]),
            len(ann.structural_role[:4]),
        ) + 1
        eva_row.append(ann.eva.ljust(width))
        cvc_row.append(ann.decoded.ljust(width))
        seg_row.append('|'.join(ann.segments).ljust(width))
        gloss_row.append(ann.gloss[:12].ljust(width))
        role_row.append(ann.structural_role[:4].ljust(width))

    lines.append("EVA:   " + ''.join(eva_row))
    lines.append("CVC:   " + ''.join(cvc_row))
    lines.append("Segs:  " + ''.join(seg_row))
    lines.append("Gloss: " + ''.join(gloss_row))
    lines.append("Role:  " + ''.join(role_row))
    lines.append("-" * 70)

    # Reading attempt
    reading_parts = []
    for ann in annotations:
        if ann.gloss != '?':
            reading_parts.append(ann.gloss)
        else:
            reading_parts.append(f'[{ann.decoded}]')
    lines.append("Reading: " + ' + '.join(reading_parts))

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_recipe_annotate():
    """Track D: Annotate top 10 CVC recipes with reading attempts."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 60, Track D: Recipe Annotation + Reading Attempts")
    print("=" * 70)

    rd = str(_results_dir())

    # Load data
    print("\n  Loading corpus and decoding ...")
    eva_to_triple = build_eva_to_triple_lookup()
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folios = _build_folio_list(corpus)

    # Decode with corrected CVC
    coda_corrected = build_coda_table_v2()
    cvc_decoded = decode_corpus_cvc_v2(
        all_tokens, assignment, eva_to_triple, coda_corrected)

    # Reference dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Load Costamagna inventory
    costamagna_inv, syl_to_struct = _load_segmentation_inventory()

    # Extract recipes
    print("\n  Extracting recipes ...")
    boundary_markers = find_cvc_boundary_markers(cvc_decoded, CV_BOUNDARY_MARKERS)
    print(f"  Boundary markers: {sorted(boundary_markers)}")
    recipes = extract_recipes(cvc_decoded, folios, boundary_markers)
    print(f"  Extracted {len(recipes)} recipes")

    # Score recipes
    pharma_vocab = set(PHARMA_VOCAB.keys()) if isinstance(PHARMA_VOCAB, dict) else set(PHARMA_VOCAB)
    scored = score_recipes(recipes, ref_word_set, pharma_vocab)
    scored.sort(key=lambda r: (-r.glossed_fraction, -r.max_consecutive_glossed))

    # Select top 10 with minimum 8 tokens
    eligible = [r for r in scored if r.length >= 8]
    top10 = eligible[:10]
    print(f"  Selected top {len(top10)} recipes (min 8 tokens)")

    # Build annotation vocabulary
    vocab = _build_annotation_vocab(ref_word_set)
    print(f"  Annotation vocabulary: {len(vocab)} entries")

    # Annotate each recipe
    annotated_recipes: List[AnnotatedRecipe] = []
    all_ingredients: Set[str] = set()
    all_verbs: Set[str] = set()

    for recipe_data in top10:
        annotations: List[TokenAnnotation] = []
        for pos, decoded_token in enumerate(recipe_data.tokens):
            # Find original EVA token (approximate: use decoded as fallback)
            eva_token = decoded_token  # EVA originals not stored in scored recipes

            ann = _annotate_token(
                pos, eva_token, decoded_token, costamagna_inv, vocab)
            annotations.append(ann)

        # Role distribution
        role_dist = Counter(a.structural_role for a in annotations)

        # Collect ingredients and verbs
        for ann in annotations:
            if ann.structural_role == 'INGREDIENT' and ann.gloss != '?':
                all_ingredients.add(ann.gloss)
            if ann.structural_role == 'VERB' and ann.gloss != '?':
                all_verbs.add(ann.gloss)

        # Pharma matches
        pharma_matches = [
            ann.decoded for ann in annotations
            if ann.decoded.lower() in pharma_vocab
        ]

        # Format display
        formatted = _format_recipe(recipe_data.recipe_idx, recipe_data.folio, annotations)

        # Reading attempt
        reading_parts = []
        for ann in annotations:
            if ann.gloss != '?':
                reading_parts.append(ann.gloss)
            else:
                reading_parts.append(f'[{ann.decoded}]')
        reading = ' + '.join(reading_parts)

        glossed_frac = sum(1 for a in annotations if a.gloss != '?') / len(annotations)

        annotated_recipes.append(AnnotatedRecipe(
            recipe_idx=recipe_data.recipe_idx,
            folio=recipe_data.folio,
            n_tokens=len(annotations),
            glossed_fraction=round(glossed_frac, 3),
            annotations=annotations,
            role_distribution=dict(role_dist),
            reading_attempt=reading,
            pharma_matches=pharma_matches,
            formatted_display=formatted,
        ))

    # Print annotated recipes
    for ar in annotated_recipes:
        print(f"\n{ar.formatted_display}")

    # Aggregate stats
    mean_glossed = (
        sum(ar.glossed_fraction for ar in annotated_recipes) / len(annotated_recipes)
        if annotated_recipes else 0.0)

    n_with_verb_ingredient = sum(
        1 for ar in annotated_recipes
        if ar.role_distribution.get('VERB', 0) >= 1
        and ar.role_distribution.get('INGREDIENT', 0) >= 1)

    n_with_pharma = sum(
        1 for ar in annotated_recipes
        if len(ar.pharma_matches) >= 1)

    max_consec = 0
    for ar in annotated_recipes:
        run = 0
        for ann in ar.annotations:
            if ann.gloss != '?':
                run += 1
                max_consec = max(max_consec, run)
            else:
                run = 0

    n_glossed_gt40 = sum(1 for ar in annotated_recipes if ar.glossed_fraction > 0.40)

    print(f"\n  Summary:")
    print(f"    Recipes annotated: {len(annotated_recipes)}")
    print(f"    Mean glossed fraction: {mean_glossed:.1%}")
    print(f"    With VERB + INGREDIENT: {n_with_verb_ingredient}")
    print(f"    With pharma match: {n_with_pharma}")
    print(f"    Max consecutive glossed: {max_consec}")
    print(f"    Ingredient inventory: {sorted(all_ingredients)}")
    print(f"    Verb inventory: {sorted(all_verbs)}")

    # Gates
    g1 = n_glossed_gt40 >= 5
    g2 = n_with_verb_ingredient >= 3
    g3 = n_with_pharma >= 1
    g4 = mean_glossed > 0.35
    g5 = max_consec >= 5
    gates_passed = sum([g1, g2, g3, g4, g5])

    print(f"\n  Validation Gates:")
    print(f"    G1 >= 5 with glossed > 40%:      {'PASS' if g1 else 'FAIL'} ({n_glossed_gt40})")
    print(f"    G2 >= 3 with VERB+INGREDIENT:    {'PASS' if g2 else 'FAIL'} ({n_with_verb_ingredient})")
    print(f"    G3 >= 1 pharma match:            {'PASS' if g3 else 'FAIL'} ({n_with_pharma})")
    print(f"    G4 mean glossed > 35%:           {'PASS' if g4 else 'FAIL'} ({mean_glossed:.1%})")
    print(f"    G5 >= 1 recipe with 5+ consec:   {'PASS' if g5 else 'FAIL'} ({max_consec})")
    print(f"    Gates passed: {gates_passed}/5")

    # Build top recipes for JSON (without the formatted display text)
    top_recipe_dicts = []
    for ar in annotated_recipes:
        top_recipe_dicts.append({
            'recipe_idx': ar.recipe_idx,
            'folio': ar.folio,
            'n_tokens': ar.n_tokens,
            'glossed_fraction': ar.glossed_fraction,
            'role_distribution': ar.role_distribution,
            'reading_attempt': ar.reading_attempt,
            'pharma_matches': ar.pharma_matches,
            'tokens': [
                {
                    'position': ann.position,
                    'decoded': ann.decoded,
                    'segments': ann.segments,
                    'gloss': ann.gloss,
                    'role': ann.structural_role,
                    'confidence': ann.confidence,
                }
                for ann in ar.annotations
            ],
        })

    result = RecipeAnnotationResult(
        n_recipes_annotated=len(annotated_recipes),
        mean_glossed_fraction=round(mean_glossed, 3),
        n_with_verb_ingredient=n_with_verb_ingredient,
        n_with_pharma_match=n_with_pharma,
        max_consecutive_glossed=max_consec,
        ingredient_inventory=sorted(all_ingredients),
        verb_inventory=sorted(all_verbs),
        top_recipes=top_recipe_dicts,
        g1_glossed_gt40=g1,
        g2_verb_ingredient=g2,
        g3_pharma_match=g3,
        g4_mean_glossed=g4,
        g5_consecutive=g5,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'recipe_annotation.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Track D completed in {time.time() - t0:.1f}s")
    print(f"  Verdict: {'PASS' if result.gate_passed else 'FAIL'} "
          f"({gates_passed}/5 gates)")
