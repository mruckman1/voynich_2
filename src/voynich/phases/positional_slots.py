"""
Phase 7 / Approach 9: Pharmaceutical Positional Slot Analysis
==============================================================
Medieval pharmaceutical recipes follow rigid formulaic structure. The position
of a word in a recipe instruction predicts its grammatical and semantic class
with high accuracy. If the Voynich pharmaceutical sections follow this structure,
positional analysis reveals word classes without any phonetic decoding.

Sub-analyses:
  9.1 — Latin recipe structure analysis (reference slot profiles)
  9.2 — Voynich pharmaceutical section positional statistics
  9.3 — Cross-validate position × paradigm classification
  9.4 — Verb identification and frequency matching
  9.5 — Ingredient slot analysis

Output:
  results/positional_slots.json
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus
from voynich.core.stats import (
    jensen_shannon_divergence, chi_squared_goodness,
    rank_correlation, selectivity_ratio, bootstrap_ci,
    cohens_kappa,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    load_reference_corpus, ReferenceCorpus,
    segment_latin_recipes, compute_slot_profile,
    label_word_class, SlotProfile, RecipeSegment,
    LATIN_RECIPE_VERBS, LATIN_RECIPE_CONNECTORS,
)
from voynich.phases.morpheme_grid import (
    decompose_token_morphemes, decompose_corpus_morphemes,
    MorphemeDecomposition, KNOWN_PREFIXES, KNOWN_SUFFIXES,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VoynichRecipeSegment:
    """One segmented recipe-like unit from the pharmaceutical section."""
    folio: str
    tokens: List[str]
    stems: List[str]
    affix_patterns: List[str]
    n_tokens: int


@dataclass
class VoynichSlotProfile:
    """Positional slot statistics for Voynich pharmaceutical text."""
    n_segments: int
    mean_segment_length: float
    segment_length_distribution: Dict[str, int]
    # Per-position stem concentration
    position_stem_counts: Dict[int, Dict[str, int]]
    position_affix_counts: Dict[int, Dict[str, int]]
    # Mutual information
    mi_stem_position: float
    mi_affix_position: float
    # Inferred candidates
    verb_candidate_stems: List[str]
    noun_candidate_stems: List[str]
    connector_candidate_stems: List[str]


@dataclass
class PositionParadigmCross:
    """Cross-validation of position × paradigm predictions."""
    contingency_table: Dict[str, Dict[str, int]]
    kappa: float
    chi2: float
    chi2_p: float
    latin_kappa: float
    kappa_ratio: float
    verdict: str


@dataclass
class VerbCandidate:
    """A stem identified as a potential recipe verb."""
    stem: str
    token_count: int
    position_1_pct: float
    n_suffix_types: int
    n_forms: int
    frequency_rank: int


@dataclass
class IngredientCandidate:
    """A stem identified as a potential ingredient noun."""
    stem: str
    token_count: int
    medial_pct: float
    n_suffix_types: int
    appears_on_herbal_folios: bool
    herbal_folio_count: int


@dataclass
class PositionalSlotsResult:
    """Full Approach 9 output."""
    # 9.1: Latin reference
    latin_n_recipes: int
    latin_mean_recipe_length: float
    latin_verb_initial_ratio: float
    latin_slot_entropy: List[float]
    # 9.2: Voynich pharmaceutical
    voynich_n_segments: int
    voynich_mean_segment_length: float
    voynich_mi_stem_position: float
    voynich_mi_affix_position: float
    n_verb_candidates: int
    n_noun_candidates: int
    n_connector_candidates: int
    # 9.3: Cross-validation
    position_paradigm_kappa: float
    position_paradigm_chi2: float
    position_paradigm_chi2_p: float
    latin_position_paradigm_kappa: float
    # 9.4: Verb identification
    verb_candidates: List[Dict]
    verb_frequency_rho: float
    verb_frequency_p: float
    # 9.5: Ingredient identification
    ingredient_candidates: List[Dict]
    n_ingredients_on_herbal_folios: int
    # Null tests
    null_mi_mean: float
    null_mi_std: float
    mi_selectivity: float
    null_kappa_mean: float
    null_kappa_std: float
    kappa_selectivity: float
    # Gate
    mi_gate: bool
    kappa_gate: bool
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# 9.1 — Latin recipe structure analysis
# ---------------------------------------------------------------------------

def analyze_latin_recipes(ref_corpus: ReferenceCorpus) -> Tuple[List[RecipeSegment], SlotProfile]:
    """Segment Circa Instans and compute positional slot profile."""
    recipes = segment_latin_recipes(ref_corpus, language='latin', min_tokens=3)
    profile = compute_slot_profile(recipes, max_position=10)
    return recipes, profile


# ---------------------------------------------------------------------------
# 9.2 — Voynich pharmaceutical section analysis
# ---------------------------------------------------------------------------

def segment_voynich_pharmaceutical(
    corpus: VoynichCorpus,
) -> List[VoynichRecipeSegment]:
    """
    Segment the Voynich pharmaceutical + recipes sections into units.

    Uses paragraph line boundaries from the IVTFF transcription. Each
    paragraph locus typically corresponds to one recipe entry or
    instruction block.
    """
    segments = []
    for section in ('pharmaceutical', 'recipes'):
        pages = corpus.get_pages_by_section(section)
        for page in pages:
            for locus in page.loci:
                text = locus.clean_text
                if not text or not text.strip():
                    continue
                tokens = text.split()
                if len(tokens) < 3:
                    continue
                # Decompose each token
                stems = []
                affix_patterns = []
                for tok in tokens:
                    d = decompose_token_morphemes(tok)
                    stems.append(d.stem if d.stem else tok)
                    patt = (d.prefix or '') + '...' + (d.suffix or '')
                    affix_patterns.append(patt)
                segments.append(VoynichRecipeSegment(
                    folio=page.folio,
                    tokens=tokens,
                    stems=stems,
                    affix_patterns=affix_patterns,
                    n_tokens=len(tokens),
                ))
    return segments


def compute_mutual_information(
    segments: List[VoynichRecipeSegment],
    max_position: int = 8,
    use_stems: bool = True,
) -> float:
    """
    Compute MI(token_class, position) across segments.

    High MI = rigid positional structure (consistent with recipes).
    Low MI = free word order.
    """
    # Build joint distribution P(class, position)
    joint_counts: Dict[Tuple[str, int], int] = Counter()
    position_counts: Dict[int, int] = Counter()
    class_counts: Dict[str, int] = Counter()
    total = 0

    for seg in segments:
        items = seg.stems if use_stems else seg.affix_patterns
        for pos in range(min(max_position, len(items))):
            item = items[pos]
            joint_counts[(item, pos)] += 1
            position_counts[pos] += 1
            class_counts[item] += 1
            total += 1

    if total == 0:
        return 0.0

    mi = 0.0
    for (item, pos), count in joint_counts.items():
        p_joint = count / total
        p_item = class_counts[item] / total
        p_pos = position_counts[pos] / total
        if p_item > 0 and p_pos > 0:
            mi += p_joint * math.log2(p_joint / (p_item * p_pos))
    return mi


def classify_stems_by_position(
    segments: List[VoynichRecipeSegment],
    threshold_pct: float = 60.0,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Classify stems into verb/noun/connector candidates by positional behavior.

    - Verb candidates: stems with >= threshold_pct at position 0
    - Connector candidates: stems appearing primarily at segment boundaries
      (first or last 2 positions) and high frequency
    - Noun candidates: stems primarily at medial positions (2-6)
    """
    # Count stem occurrences at each relative position
    stem_pos_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    stem_total: Dict[str, int] = Counter()

    for seg in segments:
        n = len(seg.stems)
        for i, stem in enumerate(seg.stems):
            stem_total[stem] += 1
            if i == 0:
                stem_pos_counts[stem]['initial'] += 1
            elif i == n - 1:
                stem_pos_counts[stem]['final'] += 1
            else:
                stem_pos_counts[stem]['medial'] += 1

    verb_candidates = []
    noun_candidates = []
    connector_candidates = []

    for stem, total in stem_total.items():
        if total < 3:
            continue
        counts = stem_pos_counts[stem]
        initial_pct = 100.0 * counts.get('initial', 0) / total
        medial_pct = 100.0 * counts.get('medial', 0) / total
        final_pct = 100.0 * counts.get('final', 0) / total

        if initial_pct >= threshold_pct:
            verb_candidates.append(stem)
        elif medial_pct >= threshold_pct:
            noun_candidates.append(stem)
        elif (initial_pct + final_pct) >= threshold_pct and total >= 5:
            connector_candidates.append(stem)

    return verb_candidates, noun_candidates, connector_candidates


def build_voynich_slot_profile(
    segments: List[VoynichRecipeSegment],
    max_position: int = 8,
) -> VoynichSlotProfile:
    """Build full positional slot profile for Voynich pharmaceutical text."""
    # Per-position stem and affix counts
    pos_stem: Dict[int, Counter] = defaultdict(Counter)
    pos_affix: Dict[int, Counter] = defaultdict(Counter)

    for seg in segments:
        for pos in range(min(max_position, seg.n_tokens)):
            pos_stem[pos][seg.stems[pos]] += 1
            pos_affix[pos][seg.affix_patterns[pos]] += 1

    # Segment length distribution
    lengths = [seg.n_tokens for seg in segments]
    len_dist = Counter()
    for l in lengths:
        bucket = str(min(l, 20))
        len_dist[bucket] += 1

    mi_stem = compute_mutual_information(segments, max_position, use_stems=True)
    mi_affix = compute_mutual_information(segments, max_position, use_stems=False)

    verbs, nouns, connectors = classify_stems_by_position(segments)

    return VoynichSlotProfile(
        n_segments=len(segments),
        mean_segment_length=sum(lengths) / max(len(lengths), 1),
        segment_length_distribution=dict(len_dist),
        position_stem_counts={k: dict(v) for k, v in pos_stem.items()},
        position_affix_counts={k: dict(v) for k, v in pos_affix.items()},
        mi_stem_position=mi_stem,
        mi_affix_position=mi_affix,
        verb_candidate_stems=verbs,
        noun_candidate_stems=nouns,
        connector_candidate_stems=connectors,
    )


# ---------------------------------------------------------------------------
# 9.3 — Cross-validate position × paradigm
# ---------------------------------------------------------------------------

def _load_paradigm_data() -> Optional[Dict]:
    """Load paradigm discovery results from Phase 5.1."""
    path = _results_dir() / 'paradigm_discovery.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _build_paradigm_lookup(paradigm_data: Dict) -> Dict[str, str]:
    """
    Build a stem -> paradigm_class lookup from Phase 5.1 cluster data.

    Classifies clusters by relative paradigm richness:
    - The cluster with highest mean_n_forms+mean_n_suffixes = verb_paradigm
    - Next highest = noun_paradigm
    - Next = adj_paradigm
    - Lowest = particle_paradigm
    """
    if paradigm_data is None:
        return {}

    clusters = paradigm_data.get('clusters', [])
    if not clusters:
        return {}

    # Rank clusters by morphological richness
    ranked = sorted(clusters, key=lambda c: (
        c.get('mean_n_forms', 0) + c.get('mean_n_suffixes', 0)
    ), reverse=True)

    class_names = ['verb_paradigm', 'noun_paradigm', 'adj_paradigm', 'particle_paradigm']
    lookup = {}
    for i, cluster in enumerate(ranked):
        cls = class_names[min(i, len(class_names) - 1)]
        for stem in cluster.get('member_stems', []):
            lookup[stem] = cls

    return lookup


def _classify_paradigm(stem: str, paradigm_lookup: Dict[str, str]) -> str:
    """Classify a stem's paradigm shape using pre-built lookup."""
    return paradigm_lookup.get(stem, 'unknown')


def cross_validate_position_paradigm(
    segments: List[VoynichRecipeSegment],
    paradigm_lookup: Dict[str, str],
    latin_recipes: List[RecipeSegment],
) -> PositionParadigmCross:
    """
    Cross-validate positional and paradigmatic word-class predictions.

    Builds a contingency table of (positional_class × paradigm_class)
    and computes Cohen's kappa. High kappa means the two independent
    classification systems agree — strong evidence for real word classes.
    """
    # Map each stem to positional class and paradigm class
    # Only include stems that have BOTH a non-trivial positional class
    # AND a known paradigm class (to avoid swamping with content/unknown)
    positional_labels = []
    paradigm_labels = []

    verbs, nouns, connectors = classify_stems_by_position(segments)
    verb_set = set(verbs)
    noun_set = set(nouns)
    conn_set = set(connectors)

    seen_stems = set()
    for seg in segments:
        for stem in seg.stems:
            if stem in seen_stems:
                continue
            seen_stems.add(stem)

            # Paradigm class
            para_class = _classify_paradigm(stem, paradigm_lookup)

            # Positional class
            if stem in verb_set:
                pos_class = 'verb'
            elif stem in noun_set:
                pos_class = 'noun'
            elif stem in conn_set:
                pos_class = 'connector'
            else:
                pos_class = 'content'

            # Include all stems with known paradigm class
            # (unknown paradigm provides no signal)
            if para_class != 'unknown':
                positional_labels.append(pos_class)
                paradigm_labels.append(para_class)

    pos_arr = np.array(positional_labels)
    par_arr = np.array(paradigm_labels)
    kappa = cohens_kappa(pos_arr, par_arr) if len(pos_arr) > 0 else 0.0

    # Chi-squared on contingency table
    contingency: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    for p, g in zip(positional_labels, paradigm_labels):
        contingency[p][g] += 1

    # Build matrix for chi²
    pos_classes = sorted(set(positional_labels))
    par_classes = sorted(set(paradigm_labels))
    if len(pos_classes) >= 2 and len(par_classes) >= 2:
        table = np.zeros((len(pos_classes), len(par_classes)))
        for i, pc in enumerate(pos_classes):
            for j, gc in enumerate(par_classes):
                table[i, j] = contingency[pc].get(gc, 0)
        from scipy.stats import chi2_contingency
        chi2, p_val, _, _ = chi2_contingency(table)
    else:
        chi2, p_val = 0.0, 1.0

    # Latin comparison: compute kappa for Latin recipes
    latin_pos_labels = []
    latin_para_labels = []
    for recipe in latin_recipes:
        for i, (tok, cls) in enumerate(zip(recipe.tokens, recipe.word_classes)):
            if i == 0:
                lat_pos = 'verb' if cls == 'verb' else 'content'
            elif i == recipe.n_tokens - 1:
                lat_pos = 'content'
            else:
                lat_pos = 'noun' if cls == 'other' else cls
            latin_pos_labels.append(lat_pos)
            # Latin paradigm: classify by suffix
            if cls == 'verb':
                latin_para_labels.append('verb_paradigm')
            elif cls == 'connector':
                latin_para_labels.append('particle_paradigm')
            else:
                latin_para_labels.append('noun_paradigm')

    latin_kappa = 0.0
    if latin_pos_labels:
        latin_kappa = cohens_kappa(
            np.array(latin_pos_labels), np.array(latin_para_labels),
        )

    kappa_ratio = kappa / latin_kappa if abs(latin_kappa) > 1e-10 else 0.0
    if kappa > 0.3:
        verdict = 'substantial_agreement'
    elif kappa > 0.1:
        verdict = 'moderate_agreement'
    else:
        verdict = 'weak_or_no_agreement'

    return PositionParadigmCross(
        contingency_table={k: dict(v) for k, v in contingency.items()},
        kappa=kappa,
        chi2=chi2,
        chi2_p=p_val,
        latin_kappa=latin_kappa,
        kappa_ratio=kappa_ratio,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# 9.4 — Verb identification
# ---------------------------------------------------------------------------

def identify_verb_candidates(
    segments: List[VoynichRecipeSegment],
    threshold_pct: float = 60.0,
) -> List[VerbCandidate]:
    """
    Identify stems that behave like recipe-initial verbs.

    Criteria: high concentration at position 0, verb-like paradigm shape,
    occurrence across multiple folios.
    """
    stem_pos0: Counter = Counter()
    stem_total: Counter = Counter()
    stem_forms: Dict[str, set] = defaultdict(set)
    stem_suffixes: Dict[str, set] = defaultdict(set)

    for seg in segments:
        if not seg.stems:
            continue
        stem_pos0[seg.stems[0]] += 1
        for i, stem in enumerate(seg.stems):
            stem_total[stem] += 1
            stem_forms[stem].add(seg.tokens[i])
            d = decompose_token_morphemes(seg.tokens[i])
            if d.suffix:
                stem_suffixes[stem].add(d.suffix)

    candidates = []
    for stem, total in stem_total.most_common():
        if total < 3:
            continue
        pos0_pct = 100.0 * stem_pos0.get(stem, 0) / total
        if pos0_pct < threshold_pct:
            continue
        n_suf = len(stem_suffixes.get(stem, set()))
        n_forms = len(stem_forms.get(stem, set()))
        candidates.append(VerbCandidate(
            stem=stem,
            token_count=total,
            position_1_pct=pos0_pct,
            n_suffix_types=n_suf,
            n_forms=n_forms,
            frequency_rank=0,
        ))

    # Assign frequency ranks
    candidates.sort(key=lambda c: -c.token_count)
    for i, c in enumerate(candidates):
        c.frequency_rank = i + 1

    return candidates


def match_verb_frequency(
    voynich_verbs: List[VerbCandidate],
    n_latin_verbs: int = 10,
) -> Tuple[float, float]:
    """
    Score whether verb candidate frequency ranking matches
    expected Latin recipe verb ranking via Spearman correlation.
    """
    n = min(len(voynich_verbs), n_latin_verbs)
    if n < 3:
        return 0.0, 1.0
    # Expected: the top Latin verbs are recipe, accipe, misce, contere, coque...
    # Their frequency ranking is 1, 2, 3, ...
    # If Voynich verb candidates mirror this zipfian drop, rho > 0
    voynich_freqs = np.array([v.token_count for v in voynich_verbs[:n]], dtype=float)
    latin_ranks = np.arange(1, n + 1, dtype=float)
    # Compare frequency magnitude to expected rank order
    rho, p = rank_correlation(voynich_freqs, -latin_ranks)
    return rho, p


# ---------------------------------------------------------------------------
# 9.5 — Ingredient slot analysis
# ---------------------------------------------------------------------------

def identify_ingredient_candidates(
    segments: List[VoynichRecipeSegment],
    corpus: VoynichCorpus,
    threshold_pct: float = 50.0,
) -> List[IngredientCandidate]:
    """
    Identify stems that behave like ingredient nouns.

    Post-verb medial stems with noun-like paradigms. Cross-reference
    with herbal folios to identify plant names.
    """
    stem_medial: Counter = Counter()
    stem_total: Counter = Counter()
    stem_suffixes: Dict[str, set] = defaultdict(set)

    for seg in segments:
        n = len(seg.stems)
        for i, stem in enumerate(seg.stems):
            stem_total[stem] += 1
            if 0 < i < n - 1:
                stem_medial[stem] += 1
            d = decompose_token_morphemes(seg.tokens[i])
            if d.suffix:
                stem_suffixes[stem].add(d.suffix)

    # Find stems that also appear on herbal folios
    herbal_stems: Counter = Counter()
    for page in corpus.get_pages_by_section('herbal_a'):
        for tok in page.all_tokens:
            d = decompose_token_morphemes(tok)
            if d.stem:
                herbal_stems[d.stem] += 1

    candidates = []
    for stem, total in stem_total.most_common():
        if total < 3:
            continue
        medial_pct = 100.0 * stem_medial.get(stem, 0) / total
        if medial_pct < threshold_pct:
            continue
        n_suf = len(stem_suffixes.get(stem, set()))
        herbal_count = herbal_stems.get(stem, 0)
        candidates.append(IngredientCandidate(
            stem=stem,
            token_count=total,
            medial_pct=medial_pct,
            n_suffix_types=n_suf,
            appears_on_herbal_folios=herbal_count > 0,
            herbal_folio_count=herbal_count,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Null tests
# ---------------------------------------------------------------------------

def null_test_mi(
    segments: List[VoynichRecipeSegment],
    real_mi: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test: shuffle tokens within each segment, recompute MI.

    Returns (null_mean, null_std, selectivity).
    """
    rng = random.Random(seed)
    null_values = []

    for _ in range(n_trials):
        shuffled = []
        for seg in segments:
            stems_copy = list(seg.stems)
            rng.shuffle(stems_copy)
            shuffled.append(VoynichRecipeSegment(
                folio=seg.folio,
                tokens=seg.tokens,
                stems=stems_copy,
                affix_patterns=seg.affix_patterns,
                n_tokens=seg.n_tokens,
            ))
        null_mi = compute_mutual_information(shuffled, max_position=8, use_stems=True)
        null_values.append(null_mi)

    null_arr = np.array(null_values)
    sel = selectivity_ratio(real_mi, null_arr)
    return float(np.mean(null_arr)), float(np.std(null_arr)), sel


def null_test_kappa(
    segments: List[VoynichRecipeSegment],
    paradigm_lookup: Dict[str, str],
    latin_recipes: List[RecipeSegment],
    real_kappa: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Null test: shuffle stems within each segment, recompute kappa.

    Returns (null_mean, null_std, selectivity).
    """
    rng = random.Random(seed)
    null_values = []

    for _ in range(n_trials):
        shuffled = []
        for seg in segments:
            stems_copy = list(seg.stems)
            rng.shuffle(stems_copy)
            shuffled.append(VoynichRecipeSegment(
                folio=seg.folio,
                tokens=seg.tokens,
                stems=stems_copy,
                affix_patterns=seg.affix_patterns,
                n_tokens=seg.n_tokens,
            ))
        cross = cross_validate_position_paradigm(shuffled, paradigm_lookup, latin_recipes)
        null_values.append(cross.kappa)

    null_arr = np.array(null_values)
    # For kappa, higher is better, so selectivity = real / null_mean
    # But null kappas can be near 0 or negative, so use absolute comparison
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))
    if null_std > 1e-10:
        z = (real_kappa - null_mean) / null_std
        sel = max(0, z / 1.5)  # Normalize so 1.5σ = selectivity 1.0
    else:
        sel = float('inf') if real_kappa > null_mean else 0.0
    return null_mean, null_std, sel


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_results(result: PositionalSlotsResult):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print("APPROACH 9: PHARMACEUTICAL POSITIONAL SLOT ANALYSIS")
    print("=" * 70)

    print("\n--- 9.1: Latin Recipe Reference ---")
    print(f"  Recipes segmented:        {result.latin_n_recipes}")
    print(f"  Mean recipe length:       {result.latin_mean_recipe_length:.1f} tokens")
    print(f"  Verb-initial ratio:       {result.latin_verb_initial_ratio:.1%}")
    print(f"  Slot entropy (pos 0-4):   {' '.join(f'{e:.2f}' for e in result.latin_slot_entropy[:5])}")

    print("\n--- 9.2: Voynich Pharmaceutical Section ---")
    print(f"  Segments found:           {result.voynich_n_segments}")
    print(f"  Mean segment length:      {result.voynich_mean_segment_length:.1f} tokens")
    print(f"  MI(stem, position):       {result.voynich_mi_stem_position:.4f}")
    print(f"  MI(affix, position):      {result.voynich_mi_affix_position:.4f}")
    print(f"  Verb candidates:          {result.n_verb_candidates}")
    print(f"  Noun candidates:          {result.n_noun_candidates}")
    print(f"  Connector candidates:     {result.n_connector_candidates}")

    print("\n--- 9.3: Position × Paradigm Cross-Validation ---")
    print(f"  Cohen's kappa:            {result.position_paradigm_kappa:.3f}")
    print(f"  Chi²:                     {result.position_paradigm_chi2:.2f}  (p = {result.position_paradigm_chi2_p:.4f})")
    print(f"  Latin reference kappa:    {result.latin_position_paradigm_kappa:.3f}")

    print("\n--- 9.4: Verb Identification ---")
    if result.verb_candidates:
        print(f"  {'Rank':<6} {'Stem':<12} {'Count':<8} {'Pos-1%':<10} {'Suffixes':<10}")
        for v in result.verb_candidates[:10]:
            print(f"  {v['frequency_rank']:<6} {v['stem']:<12} {v['token_count']:<8} "
                  f"{v['position_1_pct']:<10.1f} {v['n_suffix_types']:<10}")
    print(f"  Frequency match rho:      {result.verb_frequency_rho:.3f}  (p = {result.verb_frequency_p:.4f})")

    print("\n--- 9.5: Ingredient Identification ---")
    if result.ingredient_candidates:
        print(f"  {'Stem':<12} {'Count':<8} {'Medial%':<10} {'Herbal?':<10}")
        for ing in result.ingredient_candidates[:10]:
            herb = 'yes' if ing['appears_on_herbal_folios'] else 'no'
            print(f"  {ing['stem']:<12} {ing['token_count']:<8} "
                  f"{ing['medial_pct']:<10.1f} {herb:<10}")
    print(f"  On herbal folios:         {result.n_ingredients_on_herbal_folios} / {len(result.ingredient_candidates)}")

    print("\n--- Null Tests ---")
    print(f"  MI selectivity:           {result.mi_selectivity:.2f}×  "
          f"(null μ={result.null_mi_mean:.4f} σ={result.null_mi_std:.4f})")
    print(f"  Kappa selectivity:        {result.kappa_selectivity:.2f}×  "
          f"(null μ={result.null_kappa_mean:.4f} σ={result.null_kappa_std:.4f})")

    print(f"\n  MI gate:                  {'PASS' if result.mi_gate else 'FAIL'}")
    print(f"  Kappa gate:               {'PASS' if result.kappa_gate else 'FAIL'}")
    print(f"  Overall gate:             {'PASS' if result.gate_passed else 'FAIL'}")
    print(f"  Verdict:                  {result.verdict}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _convert(obj):
    """Convert dataclass/numpy types to JSON-serializable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


def run_positional_slots() -> Dict:
    """
    Run Approach 9: Pharmaceutical Positional Slot Analysis.

    Flow:
    1. Load corpus + reference
    2. Load Phase 5.1 paradigm data
    3. Analyze Latin recipe structure (9.1)
    4. Segment and profile Voynich pharmaceutical text (9.2)
    5. Cross-validate position × paradigm (9.3)
    6. Identify verb candidates (9.4)
    7. Identify ingredient candidates (9.5)
    8. Null tests
    9. Gate checks
    10. Save to results/positional_slots.json
    """
    print("Loading corpus and reference data...")
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus()

    paradigm_data = _load_paradigm_data()
    if paradigm_data is None:
        print("  WARNING: Phase 5.1 results not found. Paradigm cross-validation "
              "will be limited. Run 'voynich paradigms' first.")
    paradigm_lookup = _build_paradigm_lookup(paradigm_data) if paradigm_data else {}

    # 9.1 — Latin recipe analysis
    print("9.1: Analyzing Latin recipe structure...")
    latin_recipes, latin_profile = analyze_latin_recipes(ref_corpus)

    # 9.2 — Voynich pharmaceutical section
    print("9.2: Segmenting Voynich pharmaceutical sections...")
    segments = segment_voynich_pharmaceutical(corpus)
    voynich_profile = build_voynich_slot_profile(segments)

    # 9.3 — Cross-validate position × paradigm
    print("9.3: Cross-validating position × paradigm...")
    cross = cross_validate_position_paradigm(segments, paradigm_lookup, latin_recipes)

    # 9.4 — Verb identification
    print("9.4: Identifying verb candidates...")
    verb_candidates = identify_verb_candidates(segments)
    verb_rho, verb_p = match_verb_frequency(verb_candidates)

    # 9.5 — Ingredient identification
    print("9.5: Identifying ingredient candidates...")
    ingredient_candidates = identify_ingredient_candidates(segments, corpus)

    # Null tests
    print("Running null tests (100 trials each)...")
    null_mi_mean, null_mi_std, mi_sel = null_test_mi(
        segments, voynich_profile.mi_stem_position,
    )
    null_kappa_mean, null_kappa_std, kappa_sel = null_test_kappa(
        segments, paradigm_lookup, latin_recipes, cross.kappa,
    )

    # Gate checks
    mi_gate = mi_sel > 1.5
    kappa_gate = cross.kappa > 0.1 and kappa_sel > 1.0
    gate_passed = mi_gate  # MI gate is the primary criterion

    if gate_passed and kappa_gate:
        verdict = 'positional_structure_confirmed'
    elif gate_passed:
        verdict = 'positional_structure_detected_paradigm_weak'
    elif kappa_gate:
        verdict = 'paradigm_agreement_without_positional_structure'
    else:
        verdict = 'no_significant_positional_structure'

    result = PositionalSlotsResult(
        latin_n_recipes=latin_profile.n_recipes,
        latin_mean_recipe_length=latin_profile.mean_recipe_length,
        latin_verb_initial_ratio=latin_profile.verb_initial_ratio,
        latin_slot_entropy=latin_profile.slot_entropy_by_position,
        voynich_n_segments=voynich_profile.n_segments,
        voynich_mean_segment_length=voynich_profile.mean_segment_length,
        voynich_mi_stem_position=voynich_profile.mi_stem_position,
        voynich_mi_affix_position=voynich_profile.mi_affix_position,
        n_verb_candidates=len(verb_candidates),
        n_noun_candidates=len(voynich_profile.noun_candidate_stems),
        n_connector_candidates=len(voynich_profile.connector_candidate_stems),
        position_paradigm_kappa=cross.kappa,
        position_paradigm_chi2=cross.chi2,
        position_paradigm_chi2_p=cross.chi2_p,
        latin_position_paradigm_kappa=cross.latin_kappa,
        verb_candidates=[_convert(asdict(v)) for v in verb_candidates[:15]],
        verb_frequency_rho=verb_rho,
        verb_frequency_p=verb_p,
        ingredient_candidates=[_convert(asdict(ing)) for ing in ingredient_candidates[:20]],
        n_ingredients_on_herbal_folios=sum(
            1 for ing in ingredient_candidates if ing.appears_on_herbal_folios
        ),
        null_mi_mean=null_mi_mean,
        null_mi_std=null_mi_std,
        mi_selectivity=mi_sel,
        null_kappa_mean=null_kappa_mean,
        null_kappa_std=null_kappa_std,
        kappa_selectivity=kappa_sel,
        mi_gate=mi_gate,
        kappa_gate=kappa_gate,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    _print_results(result)

    out = _convert(asdict(result))
    out_path = _results_dir() / 'positional_slots.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out
