"""
Phase 14.3 – Feature-Level CSP Solver
=======================================
Replaces Phase 11's 14 cell-level CSP variables with ~23 feature-level
variables, one per attested (first_stroke, last_stroke, glyph_class) triple.
Each EVA character gets its own phoneme assignment instead of sharing a slot
with 2–8 other characters in the same grid cell.

Architecture
------------
``FeatureVariable`` duck-types to ``CSPVariable``: it has the same
``.cell_key``, ``.domain``, ``.frequency`` attributes.  The "cell_key" for a
FeatureVariable is the triple_key string ``"first_stroke,last_stroke,glyph_class"``.

All of ``beam_search()``, ``score_assignment_full()``, and ``ac3_propagate()``
from ``csp_solver.py`` are reused unchanged — they operate on the triple_key
namespace transparently.

The only bridge: ``build_eva_to_triple_lookup()`` from ``corpus.py`` is
passed as ``eva_to_cell`` to every function that accepts that argument.

Dependency chain:
    stroke_features.json (Step 14.2)
    rosetta_selection.json, verb_identification.json (Phase 11 anchors/verbs)
        → feature_csp.json (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_cv_syllable_table,
    build_triple_phoneme_hypotheses,
    load_reference_corpus,
)
from voynich.core.stats import build_ngram_lm
from voynich.phases.csp_constraints import (
    AnchorConstraint,
    PhonemeInventory,
    VerbConstraint,
    build_phoneme_inventory,
    prune_by_frequency,
    prune_by_inventory,
    prune_by_phonotactics,
    score_cross_entropy,
)
from voynich.phases.csp_solver import (
    CSPVariable,
    _convert,
    ac3_propagate,
    beam_search,
    decode_corpus,
    decode_token,
    score_assignment_full,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureVariable:
    """One attested (first_stroke, last_stroke, glyph_class) triple.

    Duck-types to :class:`~voynich.phases.csp_solver.CSPVariable` so that
    :func:`~voynich.phases.csp_solver.beam_search` and
    :func:`~voynich.phases.csp_solver.score_assignment_full` work without
    modification.
    """
    cell_key: str           # triple_key = "first_stroke,last_stroke,glyph_class"
    cv_label: str           # human-readable label (same as cell_key for clarity)
    eva_glyphs: List[str]   # EVA characters sharing this triple
    frequency: int          # total corpus frequency across all glyphs
    domain: List[str] = field(default_factory=list)
    # Extra Phase 14 metadata (not used by beam_search, for reporting only)
    first_stroke: str = ''
    last_stroke: str = ''
    glyph_class: str = ''
    onset_candidates: List[str] = field(default_factory=list)
    nucleus_candidates: List[str] = field(default_factory=list)


@dataclass
class FeatureCSPResult:
    """Full Phase 14 feature CSP solution output."""
    language: str
    n_feature_variables: int
    n_phase11_variables: int            # 14 (for comparison)
    domain_sizes_initial: Dict[str, int]
    domain_sizes_after_propagation: Dict[str, int]
    best_assignment: Dict[str, str]     # triple_key -> syllable
    best_dict_hit: float
    best_cross_entropy: float
    best_word_validity: float
    best_anchor_matches: int
    best_selectivity: float
    phase11_baseline_dict_hit: float    # 11.1%
    improvement: float                  # best_dict_hit - 11.1%
    top_k_assignments: List[Dict]
    decoded_sample: List[Any]
    runtime_seconds: float
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Build feature variables
# ---------------------------------------------------------------------------

def build_feature_variables(
    eva_to_triple: Dict[str, str],
    token_freqs: Counter,
    inventory: PhonemeInventory,
    hypothesis_map: Optional[Dict[str, List[str]]] = None,
) -> List[FeatureVariable]:
    """Build one :class:`FeatureVariable` per attested stroke triple.

    Parameters
    ----------
    eva_to_triple:
        Lookup from :func:`~voynich.core.corpus.build_eva_to_triple_lookup`.
    token_freqs:
        Per-glyph corpus frequency counts.
    inventory:
        The target-language phoneme inventory.
    hypothesis_map:
        Optional per-triple candidate syllable list from
        :func:`~voynich.core.reference.build_triple_phoneme_hypotheses`.
        If *None*, domains are initialised from the full inventory.
    """
    # Collect attested triples and their glyphs
    triple_to_glyphs: Dict[str, List[str]] = {}
    for glyph, triple_key in eva_to_triple.items():
        if triple_key not in triple_to_glyphs:
            triple_to_glyphs[triple_key] = []
        triple_to_glyphs[triple_key].append(glyph)

    variables: List[FeatureVariable] = []
    for triple_key, glyphs in triple_to_glyphs.items():
        freq = sum(token_freqs.get(g, 0) for g in glyphs)
        parts = triple_key.split(',')
        fs = parts[0] if len(parts) > 0 else ''
        ls = parts[1] if len(parts) > 1 else ''
        gc = parts[2] if len(parts) > 2 else ''

        from voynich.core.reference import PHONEME_PLACE_MAP, PHONEME_NUCLEUS_MAP
        fv = FeatureVariable(
            cell_key=triple_key,
            cv_label=triple_key,
            eva_glyphs=glyphs,
            frequency=freq,
            first_stroke=fs,
            last_stroke=ls,
            glyph_class=gc,
            onset_candidates=PHONEME_PLACE_MAP.get(fs, []),
            nucleus_candidates=PHONEME_NUCLEUS_MAP.get(ls, []),
        )
        variables.append(fv)

    # Sort by frequency descending (mirrors CSPVariable ordering)
    variables.sort(key=lambda v: v.frequency, reverse=True)
    return variables


# ---------------------------------------------------------------------------
# Domain initialisation
# ---------------------------------------------------------------------------

def _build_anchor_constraints_triple(
    rosetta_data: Dict,
    eva_to_triple: Dict[str, str],
) -> List[AnchorConstraint]:
    """Build anchor constraints using triple_keys instead of cell_keys.

    Same logic as :func:`~voynich.phases.csp_constraints.build_anchor_constraints`
    but decomposes Voynich stems via triple lookup.
    """
    from voynich.core.stats import syllabify_latin
    from voynich.core.corpus import tokenize_eva_chars

    anchors: List[AnchorConstraint] = []
    for folio_info in rosetta_data.get('folio_scores', []):
        folio = folio_info.get('folio', '')
        selected = rosetta_data.get('selected_rosetta_folios', [])
        if folio not in selected:
            continue
        stem = folio_info.get('dominant_stem', '')
        target = folio_info.get('medieval_name', '')
        weight = folio_info.get('combined_score', 0.5)
        if not stem or not target:
            continue
        # Decompose EVA stem into triple_keys
        triple_cells = token_to_triples(stem, eva_to_triple)
        target_syls = syllabify_latin(target.split()[0])
        if not target_syls:
            target_syls = [target]
        anchors.append(AnchorConstraint(
            folio=folio,
            voynich_stem=stem,
            voynich_cells=triple_cells,
            target_word=target,
            target_syllables=target_syls,
            weight=weight,
        ))
    return anchors


def initialise_feature_domains(
    variables: List[FeatureVariable],
    inventory: PhonemeInventory,
    hypothesis_map: Optional[Dict[str, List[str]]],
    anchors: List[AnchorConstraint],
    frequency_slack: int = 3,
) -> List[FeatureVariable]:
    """Apply constraint layers to seed each FeatureVariable's domain.

    Layers applied (in order):
    1. Inventory constraint: restrict to legal CV syllables
    2. Frequency rank matching: narrow by syllable frequency rank
    3. Phonotactic legality: remove forbidden onset combinations
    4. Stroke-guidance: intersect with hypothesis_map candidates
    5. Anchor hint expansion: ensure anchor-suggested syllables are searchable

    Falls back to the full inventory for any variable whose domain becomes
    empty after any layer, so the CSP is never unsolvable.
    """
    cell_frequencies = {v.cell_key: v.frequency for v in variables}

    # Start: every triple gets the full CV syllable list
    cell_domains: Dict[str, List[str]] = {
        v.cell_key: list(inventory.cv_syllables) for v in variables
    }

    # Layer 1: inventory
    cell_domains = prune_by_inventory(cell_domains, inventory)

    # Layer 2: frequency
    cell_domains = prune_by_frequency(
        cell_domains, cell_frequencies, inventory, slack=frequency_slack,
    )

    # Layer 3: phonotactics
    cell_domains = prune_by_phonotactics(cell_domains, inventory)

    # Layer NEW: stroke-guidance — intersect with hypothesis candidates
    if hypothesis_map:
        inv_set = set(inventory.cv_syllables)
        for triple_key, domain in list(cell_domains.items()):
            hints = [s for s in hypothesis_map.get(triple_key, []) if s in inv_set]
            if hints:
                intersection = [s for s in domain if s in set(hints)]
                if intersection:
                    cell_domains[triple_key] = intersection
                # else: fallback to unpruned domain (don't let it go empty)

    # Layer 5 (partial): anchor hints — add anchor-suggested syllables so beam
    # search can discover them even if they were pruned by earlier layers.
    legal_cv = set(inventory.cv_syllables)
    for anchor in anchors:
        if len(anchor.voynich_cells) != len(anchor.target_syllables):
            continue
        for triple_key, target_syl in zip(anchor.voynich_cells, anchor.target_syllables):
            syl = target_syl.lower()
            if triple_key in cell_domains and syl in legal_cv:
                if syl not in cell_domains[triple_key]:
                    cell_domains[triple_key].append(syl)

    # Write back
    for v in variables:
        domain = cell_domains.get(v.cell_key, list(inventory.cv_syllables))
        if not domain:
            domain = list(inventory.cv_syllables)
        v.domain = domain

    return variables


# ---------------------------------------------------------------------------
# Per-language feature CSP run
# ---------------------------------------------------------------------------

def run_feature_csp_for_language(
    language: str,
    variables: List[FeatureVariable],
    lm: Dict,
    voynich_tokens: List[str],
    eva_to_triple: Dict[str, str],
    anchors: List[AnchorConstraint],
    inventory: PhonemeInventory,
    ref_word_set: set,
    verb_constraints: Optional[List[VerbConstraint]] = None,
    beam_width: int = 80,
    max_solutions: int = 20,
) -> FeatureCSPResult:
    """Run beam search with FeatureVariables for one target language.

    FeatureVariable is duck-typed to CSPVariable so beam_search() and
    score_assignment_full() operate transparently on triple_keys.
    """
    import copy

    phase11_baseline = 0.111

    domain_sizes_initial = {v.cell_key: len(v.domain) for v in variables}

    # AC-3 propagation
    solvable, variables = ac3_propagate(variables)  # type: ignore[arg-type]
    if not solvable:
        return FeatureCSPResult(
            language=language,
            n_feature_variables=len(variables),
            n_phase11_variables=14,
            domain_sizes_initial=domain_sizes_initial,
            domain_sizes_after_propagation={v.cell_key: len(v.domain) for v in variables},
            best_assignment={},
            best_dict_hit=0.0,
            best_cross_entropy=99.0,
            best_word_validity=0.0,
            best_anchor_matches=0,
            best_selectivity=0.0,
            phase11_baseline_dict_hit=phase11_baseline,
            improvement=0.0,
            top_k_assignments=[],
            decoded_sample=[],
            runtime_seconds=0.0,
            gate_passed=False,
            verdict="CSP unsolvable after AC-3",
        )

    domain_sizes_ac3 = {v.cell_key: len(v.domain) for v in variables}

    t0 = time.time()
    solutions = beam_search(
        variables=variables,  # type: ignore[arg-type]
        lm=lm,
        voynich_tokens=voynich_tokens,
        eva_to_cell=eva_to_triple,   # triple_key lookup passed as eva_to_cell
        anchors=anchors,
        inventory=inventory,
        ref_word_set=ref_word_set,
        verb_constraints=verb_constraints,
        relaxation_level=0,
        beam_width=beam_width,
        max_solutions=max_solutions,
    )
    elapsed = time.time() - t0

    if not solutions:
        return FeatureCSPResult(
            language=language,
            n_feature_variables=len(variables),
            n_phase11_variables=14,
            domain_sizes_initial=domain_sizes_initial,
            domain_sizes_after_propagation=domain_sizes_ac3,
            best_assignment={},
            best_dict_hit=0.0,
            best_cross_entropy=99.0,
            best_word_validity=0.0,
            best_anchor_matches=0,
            best_selectivity=0.0,
            phase11_baseline_dict_hit=phase11_baseline,
            improvement=0.0,
            top_k_assignments=[],
            decoded_sample=[],
            runtime_seconds=elapsed,
            gate_passed=False,
            verdict="No solutions found",
        )

    best = solutions[0]
    improvement = best.dict_hit_rate - phase11_baseline

    # Selectivity vs random baseline
    import random
    rng = random.Random(42)
    all_syls = list(inventory.cv_syllables)
    random_hits: List[float] = []
    for _ in range(50):
        rand_map = {v.cell_key: rng.choice(all_syls) for v in variables}
        decoded = decode_corpus(voynich_tokens, rand_map, eva_to_triple, max_tokens=500)
        hits = sum(1 for w in decoded if w in ref_word_set)
        random_hits.append(hits / len(decoded) if decoded else 0.0)
    random_baseline = sum(random_hits) / len(random_hits) if random_hits else 0.001
    selectivity = best.dict_hit_rate / max(random_baseline, 0.001)

    gate_passed = best.dict_hit_rate > phase11_baseline and selectivity >= 1.5

    if gate_passed:
        if best.dict_hit_rate > 0.25:
            verdict = f"BREAKTHROUGH: {best.dict_hit_rate:.1%} dict_hit ({selectivity:.2f}x selectivity). Feature model resolves cell conflation ceiling."
        else:
            verdict = f"IMPROVEMENT: {best.dict_hit_rate:.1%} dict_hit ({selectivity:.2f}x) vs Phase 11 11.1%. Partial resolution of cell conflation."
    else:
        verdict = (
            f"No improvement: {best.dict_hit_rate:.1%} dict_hit vs Phase 11 11.1%. "
            f"Selectivity: {selectivity:.2f}x. "
            "The 11.1%% ceiling may have a cause beyond cell-level ambiguity."
        )

    return FeatureCSPResult(
        language=language,
        n_feature_variables=len(variables),
        n_phase11_variables=14,
        domain_sizes_initial=domain_sizes_initial,
        domain_sizes_after_propagation=domain_sizes_ac3,
        best_assignment=best.mapping,
        best_dict_hit=best.dict_hit_rate,
        best_cross_entropy=best.cross_entropy,
        best_word_validity=best.word_validity,
        best_anchor_matches=best.anchor_match_count,
        best_selectivity=selectivity,
        phase11_baseline_dict_hit=phase11_baseline,
        improvement=improvement,
        top_k_assignments=[_convert(s) for s in solutions[:10]],
        decoded_sample=best.decoded_sample,
        runtime_seconds=round(elapsed, 2),
        gate_passed=gate_passed,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feature_csp() -> None:
    """Step 14.3: feature-level CSP solver on the Voynich corpus."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 14.3: Feature-Level CSP Solver")
    print("=" * 70)

    rd = _results_dir()

    # Load stroke features (Step 14.2 output)
    sf_path = os.path.join(rd, 'stroke_features.json')
    if not os.path.exists(sf_path):
        print("  [SKIP] stroke_features.json not found — run stroke-features first")
        return

    # Load corpus
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(language='A', paragraph_only=True)
    if not tokens:
        print("  [SKIP] No Language A tokens found")
        return

    # Build triple lookup
    eva_to_triple = build_eva_to_triple_lookup()

    # Glyph frequencies
    glyph_freq: Counter = Counter()
    for token in tokens:
        for ch in tokenize_eva_chars(token):
            glyph_freq[ch] += 1

    # Load reference corpora
    ref_corpus = load_reference_corpus(verbose=False)

    # Load anchor constraints
    rosetta_path = os.path.join(rd, 'rosetta_selection.json')
    anchors: List[AnchorConstraint] = []
    if os.path.exists(rosetta_path):
        with open(rosetta_path) as f:
            rosetta_data = json.load(f)
        anchors = _build_anchor_constraints_triple(rosetta_data, eva_to_triple)
        print(f"\n  Anchor constraints: {len(anchors)}")
    else:
        print("\n  [INFO] No rosetta_selection.json found — running without anchors")

    languages = ['latin', 'occitan', 'italian', 'german']
    all_results: Dict[str, Dict] = {}
    best_lang = ''
    best_dict_hit = 0.0

    for language in languages:
        print(f"\n  ── Language: {language.upper()} ──")

        # Build inventory and LM
        ref_tokens = ref_corpus.get_combined_tokens(language)
        if not ref_tokens:
            print(f"  [SKIP] No reference corpus for {language}")
            continue

        inventory = build_phoneme_inventory(language, ref_corpus)
        lm = build_ngram_lm(ref_tokens[:10000], order=3, smoothing=0.01)
        ref_word_set = set(w.lower() for w in ref_tokens if len(w) >= 2)

        # Build hypothesis map for this language
        hypothesis_map = build_triple_phoneme_hypotheses(
            language, build_cv_syllable_table(language)
        )

        # Build feature variables
        variables = build_feature_variables(
            eva_to_triple, glyph_freq, inventory, hypothesis_map
        )

        # Initialise domains
        variables = initialise_feature_domains(
            variables, inventory, hypothesis_map, anchors
        )

        domain_avg = sum(len(v.domain) for v in variables) / len(variables) if variables else 0
        print(f"  Variables: {len(variables)}, avg domain: {domain_avg:.1f}")

        # Run feature CSP
        lang_result = run_feature_csp_for_language(
            language=language,
            variables=variables,
            lm=lm,
            voynich_tokens=tokens,
            eva_to_triple=eva_to_triple,
            anchors=anchors,
            inventory=inventory,
            ref_word_set=ref_word_set,
            beam_width=80,
        )

        all_results[language] = _convert(lang_result)
        print(f"  dict_hit: {lang_result.best_dict_hit:.3f}  CE: {lang_result.best_cross_entropy:.3f}  selectivity: {lang_result.best_selectivity:.2f}x")
        print(f"  Gate: {'PASS' if lang_result.gate_passed else 'FAIL'}  | {lang_result.verdict}")

        if lang_result.best_dict_hit > best_dict_hit:
            best_dict_hit = lang_result.best_dict_hit
            best_lang = language

    if not all_results:
        print("\n  [ERROR] No language results produced")
        return

    # Combined output
    best_result = all_results.get(best_lang, {})
    out = {
        'best_language': best_lang,
        'best_dict_hit': best_dict_hit,
        'phase11_baseline': 0.111,
        'improvement': best_dict_hit - 0.111,
        'gate_passed': best_result.get('gate_passed', False),
        'language_results': all_results,
        'runtime_seconds': round(time.time() - t0, 2),
    }

    out_path = os.path.join(rd, 'feature_csp.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  ── Final Summary ──")
    print(f"  Best language: {best_lang}  |  dict_hit: {best_dict_hit:.3f}")
    print(f"  Phase 11 baseline: 11.1%  |  Improvement: {best_dict_hit - 0.111:+.3f}")
    print(f"  Overall gate: {'PASS' if out['gate_passed'] else 'FAIL'}")
    print(f"\n  Results saved → {out_path}")
