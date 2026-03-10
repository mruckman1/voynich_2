"""
Step 34.15 – Spatial-Metadata Decode (Track E)
===============================================
Tags each token with its gallows spatial metadata from Step 34.14,
then re-decodes with spatial-domain constraints.  Tests whether tokens
with specific gallows configurations decode preferentially to words
from specific semantic domains (chi-squared test).

Dependency chain:
    gallows_geometry.json      (Step 34.14)
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
    signal_bigrams.json        (Phase 29 classifications)
        → spatial_decode.json  (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.gallows_geometry import (
    GALLOWS_CHARS,
    GALLOWS_BENCH_LIGATURES,
    SPATIAL_INTERSECTING,
    SPATIAL_PRECEDING,
    SPATIAL_FOLLOWING,
    SPATIAL_STANDALONE,
    _classify_gallows_spatial,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_isolation import _decode_corpus_r3


# ---------------------------------------------------------------------------
# Helpers
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Semantic domain sets (crude Latin domain proxies)
# ---------------------------------------------------------------------------

DOMAIN_BOTANICAL = {
    'herba', 'folia', 'radix', 'semen', 'flos', 'cortex', 'planta',
    'folium', 'ramus', 'truncus', 'succus', 'oleum', 'aqua', 'terra',
    'arbor', 'rosa', 'viola', 'salvia', 'mentha', 'calendula',
}

DOMAIN_MEDICAL = {
    'dolor', 'morbus', 'cura', 'remedium', 'febris', 'sanguis',
    'vulnus', 'pestis', 'medicina', 'potio', 'unguentum', 'pillula',
    'dosis', 'corpus', 'caput', 'manus', 'pes', 'cor', 'stomachus',
}

DOMAIN_ASTRO = {
    'stella', 'luna', 'sol', 'planeta', 'caelum', 'mundus',
    'signum', 'zodiacus', 'aries', 'taurus', 'gemini', 'cancer',
    'leo', 'virgo', 'libra', 'scorpio', 'sagittarius', 'aquarius',
}

DOMAIN_FUNCTION = {
    'de', 'in', 'ad', 'per', 'cum', 'et', 'est', 'non', 'si',
    'ut', 'aut', 'vel', 'sed', 'que', 'pro', 'sub', 'super',
}

ALL_DOMAINS = {
    'botanical': DOMAIN_BOTANICAL,
    'medical': DOMAIN_MEDICAL,
    'astro': DOMAIN_ASTRO,
    'function': DOMAIN_FUNCTION,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpatialDomainCorrelation:
    """Chi-squared test result for spatial_type vs semantic domain."""
    spatial_type: str
    domain: str
    observed: int
    expected: float
    chi2_contribution: float
    enriched: bool  # observed > expected


@dataclass
class SpatialConstrainedResult:
    """Result of spatially constrained dictionary matching."""
    spatial_type: str
    n_tokens: int
    unconstrained_dict_hit: float
    constrained_dict_hit: float
    delta: float
    preferred_domains: List[str]


@dataclass
class SpatialDecodeResult:
    # Corpus overview
    n_tokens: int
    n_gallows_tokens: int
    n_no_gallows_tokens: int

    # Phase 16 baseline
    baseline_dict_hit: float
    baseline_signal_rate: float

    # Spatial-domain correlation
    chi2_statistic: float
    chi2_p_value: float
    chi2_df: int
    domain_correlations: List[Dict]

    # Spatial-constrained decode
    constrained_results: List[Dict]
    overall_constrained_dict_hit: float

    # Signal comparison
    spatial_signal_rate: float
    spatial_signal_delta: float

    # Null comparison
    null_chi2_mean: float
    null_chi2_std: float
    chi2_z_score: float

    # Verdict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------

def _chi_squared_spatial_domain(
    token_spatial_types: List[str],
    decoded: List[str],
    dict_hits: List[bool],
) -> Tuple[float, int, List[SpatialDomainCorrelation]]:
    """Compute chi-squared statistic for spatial_type x domain contingency.

    Returns (chi2, df, correlations).
    """
    # Build contingency: for each (spatial_type, domain) pair, count tokens
    spatial_types = [SPATIAL_INTERSECTING, SPATIAL_PRECEDING,
                     SPATIAL_FOLLOWING, SPATIAL_STANDALONE, 'NONE']
    domain_names = list(ALL_DOMAINS.keys()) + ['other']

    def _classify_domain(word: str) -> str:
        w = word.lower()
        for domain_name, domain_set in ALL_DOMAINS.items():
            if w in domain_set:
                return domain_name
        return 'other'

    # Build observed counts
    observed: Dict[Tuple[str, str], int] = Counter()
    row_totals: Counter = Counter()
    col_totals: Counter = Counter()
    total = 0

    for i, word in enumerate(decoded):
        if not dict_hits[i]:
            continue
        st = token_spatial_types[i]
        dom = _classify_domain(word)
        observed[(st, dom)] += 1
        row_totals[st] += 1
        col_totals[dom] += 1
        total += 1

    if total == 0:
        return 0.0, 0, []

    # Compute chi-squared
    chi2 = 0.0
    correlations: List[SpatialDomainCorrelation] = []

    active_rows = [s for s in spatial_types if row_totals[s] > 0]
    active_cols = [d for d in domain_names if col_totals[d] > 0]

    for st in active_rows:
        for dom in active_cols:
            obs = observed.get((st, dom), 0)
            exp = (row_totals[st] * col_totals[dom]) / total if total > 0 else 0.001
            if exp > 0:
                contrib = ((obs - exp) ** 2) / exp
            else:
                contrib = 0.0
            chi2 += contrib
            correlations.append(SpatialDomainCorrelation(
                spatial_type=st,
                domain=dom,
                observed=obs,
                expected=round(exp, 2),
                chi2_contribution=round(contrib, 4),
                enriched=obs > exp,
            ))

    df = max((len(active_rows) - 1) * (len(active_cols) - 1), 1)
    return chi2, df, correlations


def _chi2_p_value(chi2: float, df: int) -> float:
    """Approximate chi-squared p-value using Chernoff bound.

    For a rough approximation; not a full gamma incomplete function.
    """
    if df <= 0 or chi2 <= 0:
        return 1.0
    # Simple approximation: use normal approximation for large df
    z = ((chi2 / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / math.sqrt(
        2.0 / (9 * df)
    )
    # Standard normal CDF approximation
    if z > 6:
        return 0.0
    if z < -6:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-z * z / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (
            1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )
    if z > 0:
        return p
    return 1.0 - p


# ---------------------------------------------------------------------------
# Spatial-constrained decoding
# ---------------------------------------------------------------------------

def _spatial_constrained_decode(
    token_spatial_types: List[str],
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    domain_correlations: List[SpatialDomainCorrelation],
) -> Tuple[List[str], Dict[str, SpatialConstrainedResult]]:
    """Re-decode with spatial-type-dependent dictionary constraint.

    For each spatial type, identify enriched domains and prefer matches
    from those domains.
    """
    # Build domain preference per spatial type
    spatial_preferred: Dict[str, Set[str]] = defaultdict(set)
    for corr in domain_correlations:
        if corr.enriched and corr.chi2_contribution > 1.0:
            domain_set = ALL_DOMAINS.get(corr.domain, set())
            spatial_preferred[corr.spatial_type] |= domain_set

    # Decode all tokens normally first
    normal_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    # For gallows tokens with spatial preferences, check if spatial-constrained
    # dict matching changes anything
    constrained_decoded = list(normal_decoded)
    spatial_type_indices: Dict[str, List[int]] = defaultdict(list)

    for i, st in enumerate(token_spatial_types):
        spatial_type_indices[st].append(i)
        preferred = spatial_preferred.get(st, set())
        if preferred and normal_decoded[i] not in preferred:
            # Check if any preferred word matches this decoded form
            # via edit distance 1
            word = normal_decoded[i]
            for pw in preferred:
                if len(pw) == len(word) and sum(
                    1 for a, b in zip(pw, word) if a != b
                ) <= 1:
                    constrained_decoded[i] = pw
                    break

    # Compute per-spatial-type results
    results: Dict[str, SpatialConstrainedResult] = {}
    for st in [SPATIAL_INTERSECTING, SPATIAL_PRECEDING,
               SPATIAL_FOLLOWING, SPATIAL_STANDALONE, 'NONE']:
        indices = spatial_type_indices.get(st, [])
        n = len(indices)
        if n == 0:
            results[st] = SpatialConstrainedResult(
                spatial_type=st, n_tokens=0,
                unconstrained_dict_hit=0.0, constrained_dict_hit=0.0,
                delta=0.0, preferred_domains=[],
            )
            continue

        unc_hits = sum(1 for i in indices if normal_decoded[i] in ref_word_set)
        con_hits = sum(1 for i in indices
                       if constrained_decoded[i] in ref_word_set)
        unc_rate = unc_hits / n
        con_rate = con_hits / n

        pref_domains = []
        for corr in domain_correlations:
            if (corr.spatial_type == st and corr.enriched
                    and corr.chi2_contribution > 1.0):
                pref_domains.append(corr.domain)

        results[st] = SpatialConstrainedResult(
            spatial_type=st,
            n_tokens=n,
            unconstrained_dict_hit=round(unc_rate, 4),
            constrained_dict_hit=round(con_rate, 4),
            delta=round(con_rate - unc_rate, 4),
            preferred_domains=pref_domains,
        )

    return constrained_decoded, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_spatial_decode() -> None:
    """Step 34.15: Spatial-metadata decode."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.15: Spatial-Metadata Decode (Track E)")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Gallows geometry (optional — we can recompute if absent)
    geom_path = os.path.join(rd, 'gallows_geometry.json')
    geom_data = None
    if os.path.exists(geom_path):
        with open(geom_path) as f:
            geom_data = json.load(f)

    # Signal classifications
    signal_classifications: List[str] = []
    sig_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        signal_classifications = sig_data.get('token_classifications', [])

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Geometry data: {'loaded' if geom_data else 'will recompute'}")

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 3. Load corpus and tag spatial types ──
    print("\n  3. Loading corpus and tagging spatial types ...")
    corpus = load_corpus(verbose=False)

    all_tokens: List[str] = []
    token_folios: List[str] = []
    token_sections: List[str] = []

    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
            token_sections.append(page.section)

    n_tokens = len(all_tokens)

    # Classify each token's spatial type
    token_spatial_types: List[str] = []
    n_gallows_tokens = 0

    for token in all_tokens:
        eva_chars = tokenize_eva_chars(token)
        classifications = _classify_gallows_spatial(token, eva_chars)
        if classifications:
            # Use the first gallows classification as the token's type
            token_spatial_types.append(classifications[0][2])
            n_gallows_tokens += 1
        else:
            token_spatial_types.append('NONE')

    n_no_gallows = n_tokens - n_gallows_tokens
    print(f"     {n_tokens} tokens: {n_gallows_tokens} with gallows, "
          f"{n_no_gallows} without")

    # ── 4. Baseline decode ──
    print("\n  4. Baseline Phase 16 decode ...")
    baseline_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_hits = [w in ref_word_set for w in baseline_decoded]
    baseline_dict_hit = sum(baseline_hits) / n_tokens if n_tokens > 0 else 0.0

    baseline_signal_rate = 0.0
    if signal_classifications:
        n_sig = sum(1 for c in signal_classifications if c == 'SIGNAL')
        baseline_signal_rate = n_sig / len(signal_classifications) \
            if signal_classifications else 0.0

    print(f"     Baseline dict_hit: {baseline_dict_hit:.3f}")
    print(f"     Baseline signal_rate: {baseline_signal_rate:.3f}")

    # ── 5. Chi-squared spatial-domain test ──
    print("\n  5. Chi-squared spatial-domain correlation test ...")
    chi2, df, correlations = _chi_squared_spatial_domain(
        token_spatial_types, baseline_decoded, baseline_hits,
    )
    p_value = _chi2_p_value(chi2, df)

    print(f"     chi2 = {chi2:.2f}, df = {df}, p = {p_value:.4f}")
    sig_corrs = [c for c in correlations
                 if c.enriched and c.chi2_contribution > 1.0]
    if sig_corrs:
        print("     Enriched spatial-domain pairs:")
        for c in sorted(sig_corrs, key=lambda x: -x.chi2_contribution)[:10]:
            print(f"       {c.spatial_type:14s} x {c.domain:10s}  "
                  f"obs={c.observed} exp={c.expected:.1f} "
                  f"chi2_contrib={c.chi2_contribution:.2f}")
    else:
        print("     No significantly enriched pairs found")

    # ── 6. Spatial-constrained decode ──
    print("\n  6. Spatial-constrained decode ...")
    constrained_decoded, constrained_results = _spatial_constrained_decode(
        token_spatial_types, all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set, correlations,
    )

    con_hits = sum(1 for w in constrained_decoded if w in ref_word_set)
    overall_con_hit = con_hits / n_tokens if n_tokens > 0 else 0.0

    for st in [SPATIAL_INTERSECTING, SPATIAL_PRECEDING,
               SPATIAL_FOLLOWING, SPATIAL_STANDALONE, 'NONE']:
        cr = constrained_results.get(st)
        if cr and cr.n_tokens > 0:
            print(f"     {st:14s}  n={cr.n_tokens:5d}  "
                  f"unc={cr.unconstrained_dict_hit:.3f}  "
                  f"con={cr.constrained_dict_hit:.3f}  "
                  f"delta={cr.delta:+.3f}  "
                  f"domains={cr.preferred_domains}")

    print(f"     Overall constrained dict_hit: {overall_con_hit:.3f} "
          f"(baseline: {baseline_dict_hit:.3f})")

    # ── 7. Signal comparison ──
    print("\n  7. Signal isolation on spatial-constrained decode ...")
    # Recompute signal using constrained decode
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(
        all_tokens,
    )

    # Decode one null corpus for quick SIGNAL classification
    null_tokens = _generate_null_corpus(
        bigram_probs, initial_probs, token_lengths, n_tokens, null_seeds[0],
    )
    null_decoded = _decode_corpus_r3(
        null_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    null_hits = [w in ref_word_set for w in null_decoded]

    # Simple signal classification: real hit + null miss = SIGNAL
    con_dict_hits = [w in ref_word_set for w in constrained_decoded]
    n_spatial_signal = sum(
        1 for rh, nh in zip(con_dict_hits, null_hits)
        if rh and not nh
    )
    spatial_signal_rate = n_spatial_signal / n_tokens if n_tokens > 0 else 0.0
    spatial_signal_delta = spatial_signal_rate - baseline_signal_rate

    print(f"     Spatial signal_rate: {spatial_signal_rate:.3f}")
    print(f"     Delta vs baseline: {spatial_signal_delta:+.3f}")

    # ── 8. Null chi2 comparison ──
    print("\n  8. Null chi2 comparison (100 permutations) ...")
    rng = random.Random(42)
    null_chi2s: List[float] = []
    spatial_labels = list(set(token_spatial_types))

    for _ in range(100):
        shuffled_types = list(token_spatial_types)
        rng.shuffle(shuffled_types)
        nc2, _, _ = _chi_squared_spatial_domain(
            shuffled_types, baseline_decoded, baseline_hits,
        )
        null_chi2s.append(nc2)

    null_chi2_mean = sum(null_chi2s) / len(null_chi2s) if null_chi2s else 0.0
    null_chi2_var = (
        sum((c - null_chi2_mean) ** 2 for c in null_chi2s) / len(null_chi2s)
        if null_chi2s else 0.0
    )
    null_chi2_std = null_chi2_var ** 0.5
    chi2_z = ((chi2 - null_chi2_mean) / null_chi2_std
              if null_chi2_std > 0 else 0.0)

    print(f"     Real chi2: {chi2:.2f}")
    print(f"     Null chi2: mean={null_chi2_mean:.2f}, std={null_chi2_std:.2f}")
    print(f"     z-score: {chi2_z:.2f}")

    # ── 9. Verdict ──
    spatial_is_significant = chi2_z > 2.0 and p_value < 0.05
    decode_improved = overall_con_hit > baseline_dict_hit + 0.005
    signal_improved = spatial_signal_delta > 0.005

    verdict = (
        f"Chi2 = {chi2:.2f} (z={chi2_z:.2f}, p={p_value:.4f}). "
        f"{'SIGNIFICANT' if spatial_is_significant else 'NOT significant'} "
        f"spatial-domain correlation. "
        f"Constrained dict_hit = {overall_con_hit:.3f} vs "
        f"baseline = {baseline_dict_hit:.3f} "
        f"({'IMPROVED' if decode_improved else 'NO improvement'}). "
        f"Signal delta = {spatial_signal_delta:+.3f} "
        f"({'IMPROVED' if signal_improved else 'NO improvement'})."
    )
    print(f"\n  VERDICT: {verdict}")

    # ── 10. Save ──
    elapsed = round(time.time() - t0, 2)

    result = SpatialDecodeResult(
        n_tokens=n_tokens,
        n_gallows_tokens=n_gallows_tokens,
        n_no_gallows_tokens=n_no_gallows,
        baseline_dict_hit=round(baseline_dict_hit, 4),
        baseline_signal_rate=round(baseline_signal_rate, 4),
        chi2_statistic=round(chi2, 4),
        chi2_p_value=round(p_value, 6),
        chi2_df=df,
        domain_correlations=[_convert(asdict(c)) for c in correlations],
        constrained_results=[
            _convert(asdict(constrained_results[st]))
            for st in [SPATIAL_INTERSECTING, SPATIAL_PRECEDING,
                       SPATIAL_FOLLOWING, SPATIAL_STANDALONE, 'NONE']
            if st in constrained_results
        ],
        overall_constrained_dict_hit=round(overall_con_hit, 4),
        spatial_signal_rate=round(spatial_signal_rate, 4),
        spatial_signal_delta=round(spatial_signal_delta, 4),
        null_chi2_mean=round(null_chi2_mean, 2),
        null_chi2_std=round(null_chi2_std, 2),
        chi2_z_score=round(chi2_z, 2),
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'spatial_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {elapsed:.1f}s")
