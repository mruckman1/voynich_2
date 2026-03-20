"""
Phase 59, Investigation 9: Recipe Reading Under CVC Decode
============================================================
Phase 47 extracted 89 recipes using "cola"/"codi" boundaries.
This module re-extracts recipes from the CVC-decoded corpus and
scores their readability.

Dependency chain:
    results/coda_table.json           (Phase 57.1)
    results/combined_refine.json      (Phase 15)
    results/cvc_glossing.json         (Investigation 4, optional)
        -> results/cvc_recipes.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import build_coda_table, decode_corpus_cvc


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
# Constants
# ---------------------------------------------------------------------------

# CV boundary markers from prior phases
CV_BOUNDARY_MARKERS = {'cola', 'codi', 'bene', 'sene', 'tere', 'dine'}

# Pharmaceutical vocabulary (approximate)
PHARMA_VOCAB = {
    'recipe', 'accipe', 'misce', 'cola', 'tere', 'solve', 'distilla',
    'aqua', 'oleum', 'herba', 'radice', 'semen', 'pulvis', 'succo',
    'cortex', 'folia', 'flores', 'balsamo', 'gummi', 'cera', 'mel',
    'dosi', 'cura', 'morbo', 'dolor', 'sana', 'bene', 'coralli',
    'ratione', 'commune', 'diasene', 'stercora', 'secundi',
    # CVC variants
    'colar', 'codin', 'benen', 'senen', 'terer', 'diner',
    'cors', 'corr', 'corn', 'radin', 'coran',
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScoredRecipe:
    """A recipe extracted from CVC-decoded corpus."""
    recipe_idx: int
    folio: str
    tokens: List[str]
    length: int
    boundary_marker: str
    glossed_fraction: float
    pharma_density: float
    max_consecutive_glossed: int
    annotated_text: str


@dataclass
class CvcRecipeResult:
    """Full Investigation 9 output."""
    phase: str = "59"
    investigation: str = "9"
    experiment: str = "cvc_recipes"
    n_recipes: int = 0
    mean_length: float = 0.0
    mean_glossed_fraction: float = 0.0
    mean_pharma_density: float = 0.0
    n_with_long_runs: int = 0   # recipes with max_consecutive_glossed ≥ 5
    boundary_markers_found: List[str] = field(default_factory=list)
    top_recipes: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_enough_recipes: bool = False    # ≥ 30 recipes
    g2_glossed_frac: bool = False      # mean glossed fraction > 30%
    g3_long_runs: bool = False         # ≥ 3 recipes with max_run ≥ 5
    g4_pharma_density: bool = False    # mean pharma density > 10%
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def find_cvc_boundary_markers(
    cvc_decoded: List[str],
    cv_markers: Set[str],
) -> Set[str]:
    """Find CVC equivalents of known CV boundary markers."""
    cvc_markers: Set[str] = set()
    for decoded in cvc_decoded:
        dl = decoded.lower()
        # Check if stripping a trailing consonant gives a known marker
        if dl in cv_markers:
            cvc_markers.add(dl)
        elif len(dl) >= 4 and dl[:-1] in cv_markers:
            cvc_markers.add(dl)
    return cvc_markers


def extract_recipes(
    cvc_decoded: List[str],
    folios: List[str],
    boundary_markers: Set[str],
    min_length: int = 3,
    max_length: int = 50,
) -> List[Dict[str, Any]]:
    """Segment CVC-decoded corpus into recipes using boundary markers."""
    recipes: List[Dict[str, Any]] = []
    current_tokens: List[str] = []
    current_folio: Optional[str] = None
    current_marker: str = ''

    for idx, decoded in enumerate(cvc_decoded):
        folio = folios[idx] if idx < len(folios) else '?'
        dl = decoded.lower()

        if dl in boundary_markers and current_tokens:
            if min_length <= len(current_tokens) <= max_length:
                recipes.append({
                    'tokens': list(current_tokens),
                    'folio': current_folio or folio,
                    'length': len(current_tokens),
                    'boundary_marker': current_marker,
                })
            current_tokens = [decoded]
            current_folio = folio
            current_marker = dl
        else:
            current_tokens.append(decoded)
            if current_folio is None:
                current_folio = folio

    # Final recipe
    if current_tokens and min_length <= len(current_tokens) <= max_length:
        recipes.append({
            'tokens': list(current_tokens),
            'folio': current_folio or '?',
            'length': len(current_tokens),
            'boundary_marker': current_marker,
        })

    return recipes


def score_recipes(
    recipes: List[Dict[str, Any]],
    ref_word_set: Set[str],
    pharma_vocab: Set[str],
) -> List[ScoredRecipe]:
    """Score readability of each recipe."""
    scored = []

    for i, recipe in enumerate(recipes):
        tokens = recipe['tokens']
        n = len(tokens)

        # Glossed fraction (hit in any dictionary)
        n_glossed = sum(1 for t in tokens if t.lower() in ref_word_set)
        glossed_frac = n_glossed / n if n > 0 else 0

        # Pharma density
        n_pharma = sum(1 for t in tokens if t.lower() in pharma_vocab)
        pharma_dens = n_pharma / n if n > 0 else 0

        # Longest consecutive glossed run
        max_run = 0
        current_run = 0
        for t in tokens:
            if t.lower() in ref_word_set:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        # Annotated text
        parts = []
        for t in tokens:
            if t.lower() in ref_word_set:
                parts.append(t)
            else:
                parts.append(f"[{t}]")
        annotated = ' '.join(parts)

        scored.append(ScoredRecipe(
            recipe_idx=i,
            folio=recipe['folio'],
            tokens=tokens,
            length=n,
            boundary_marker=recipe['boundary_marker'],
            glossed_fraction=round(glossed_frac, 3),
            pharma_density=round(pharma_dens, 3),
            max_consecutive_glossed=max_run,
            annotated_text=annotated,
        ))

    return scored


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_recipe():
    """Investigation 9: Recipe reading under CVC decode."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 9: Recipe Reading Under CVC Decode")
    print("=" * 70)

    rd = str(_results_dir())

    # Load data
    print("\n  Loading corpus and decoding ...")
    eva_to_triple = build_eva_to_triple_lookup()
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    coda_table = build_coda_table('primary')

    # Build folio list
    folios: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folios.append(folio)

    # CVC decode
    cvc_decoded = decode_corpus_cvc(all_tokens, assignment, eva_to_triple, coda_table)

    # Reference dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # Find boundary markers
    print("  Finding CVC boundary markers ...")
    cvc_markers = find_cvc_boundary_markers(cvc_decoded, CV_BOUNDARY_MARKERS)
    print(f"  CVC boundary markers: {sorted(cvc_markers)}")

    # If no CVC markers found, fall back to CV markers present in CVC decode
    if not cvc_markers:
        cvc_markers = CV_BOUNDARY_MARKERS & set(d.lower() for d in cvc_decoded)
        print(f"  Fallback CV markers found in CVC decode: {sorted(cvc_markers)}")

    # Extract recipes
    print("\n  Extracting recipes ...")
    recipes = extract_recipes(cvc_decoded, folios, cvc_markers)
    print(f"  Recipes extracted: {len(recipes)}")

    if not recipes:
        print("  No recipes found.")
        result = CvcRecipeResult(runtime_seconds=round(time.time() - t0, 2))
        _save_json(rd, 'cvc_recipes.json', result)
        return

    # Score readability
    print("  Scoring recipe readability ...")
    scored = score_recipes(recipes, ref_word_set, PHARMA_VOCAB)

    # Sort by glossed fraction
    scored.sort(key=lambda r: -r.glossed_fraction)

    # Aggregate stats
    mean_len = float(np.mean([r.length for r in scored]))
    mean_glossed = float(np.mean([r.glossed_fraction for r in scored]))
    mean_pharma = float(np.mean([r.pharma_density for r in scored]))
    n_long_runs = sum(1 for r in scored if r.max_consecutive_glossed >= 5)

    print(f"\n  Recipe Statistics:")
    print(f"    Mean length:           {mean_len:.1f} tokens")
    print(f"    Mean glossed fraction: {mean_glossed:.1%}")
    print(f"    Mean pharma density:   {mean_pharma:.1%}")
    print(f"    Recipes with run ≥ 5:  {n_long_runs}")

    # Top recipes
    print(f"\n  Top recipes by glossed fraction:")
    for r in scored[:8]:
        print(f"    [{r.folio}] ({r.length} tok, {r.glossed_fraction:.0%} glossed, "
              f"run={r.max_consecutive_glossed})")
        print(f"      {r.annotated_text[:80]}")

    # Gates
    g1 = len(recipes) >= 30
    g2 = mean_glossed > 0.30
    g3 = n_long_runs >= 3
    g4 = mean_pharma > 0.10
    gates_passed = sum([g1, g2, g3, g4])

    print(f"\n  Validation Gates:")
    print(f"    G1 ≥ 30 recipes:           {'PASS' if g1 else 'FAIL'} ({len(recipes)})")
    print(f"    G2 mean glossed > 30%:     {'PASS' if g2 else 'FAIL'} ({mean_glossed:.1%})")
    print(f"    G3 ≥ 3 with run ≥ 5:       {'PASS' if g3 else 'FAIL'} ({n_long_runs})")
    print(f"    G4 mean pharma > 10%:      {'PASS' if g4 else 'FAIL'} ({mean_pharma:.1%})")
    print(f"    Gates passed: {gates_passed}/4")

    # Prepare output
    top_recipes = [_convert(r) for r in scored[:30]]

    result = CvcRecipeResult(
        n_recipes=len(recipes),
        mean_length=round(mean_len, 1),
        mean_glossed_fraction=round(mean_glossed, 4),
        mean_pharma_density=round(mean_pharma, 4),
        n_with_long_runs=n_long_runs,
        boundary_markers_found=sorted(cvc_markers),
        top_recipes=top_recipes,
        g1_enough_recipes=g1,
        g2_glossed_frac=g2,
        g3_long_runs=g3,
        g4_pharma_density=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_recipes.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 9 completed in {time.time() - t0:.1f}s")
