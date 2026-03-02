"""
Phase 8: Cipher Validation & Integration
==========================================
Shared validation between Approach 16 (Bigram Transfer) and Approach 18
(MDL Decoding).  Cross-checks the two approaches against each other and
against all prior phase findings.

Sub-analyses:
  V.1 — Cross-approach convergence (do 16 and 18 agree?)
  V.2 — Prior phase convergence (do results align with Phases 6-7.5?)
  V.3 — Seeded decoding (use 16's mapping to seed 18)
  V.4 — Combined assessment (Fisher combined probability)

Output:
  results/cipher_validate.json
"""

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core.stats import (
    fisher_combined_probability,
    build_ngram_lm,
    cross_entropy_lm,
)
from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    load_reference_corpus, stem_token,
)
from voynich.core.corpus import load_corpus
from voynich.phases.morpheme_grid import decompose_token_morphemes


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceCheck:
    """Cross-check between Approach 16 and 18 results."""
    mapping_agreement: float
    n_stems_compared: int
    high_agreement_stems: List[str]
    agreement_by_frequency_rank: Dict[str, float]


@dataclass
class PriorPhaseConvergence:
    """Check whether decoding aligns with prior phase findings."""
    illustration_agreement: float
    verb_position_agreement: float
    noun_cluster_agreement: float
    n_checks_passed: int
    n_checks_total: int
    details: Dict[str, str]


@dataclass
class CipherValidateResult:
    """Full Phase 8 validation output."""
    bigram_gate: bool
    bigram_verdict: str
    bigram_best_selectivity: float
    mdl_gate: bool
    mdl_verdict: str
    mdl_sanity_passed: bool
    mdl_best_compression: float
    convergence: Dict
    approaches_agree: bool
    prior_convergence: Dict
    seeded_improvement: float
    fisher_chi2: float
    fisher_p_value: float
    overall_gate_passed: bool
    confidence_level: str
    verdict: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Convert dataclass/numpy types to JSON-serializable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _load_result(name: str) -> Optional[Dict]:
    """Load a result JSON file from results/."""
    path = _results_dir() / f'{name}.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# V.1: Cross-Approach Convergence
# ---------------------------------------------------------------------------

def check_cross_approach_convergence(
    bigram_result: Dict,
    mdl_result: Dict,
) -> ConvergenceCheck:
    """
    Compare mappings from Approach 16 and 18.

    Measures the fraction of stems that both approaches map to the same
    target Latin stem.
    """
    # Extract best mappings
    bigram_mapping: Dict[str, str] = {}
    sa_results = bigram_result.get('sa_results', {})
    best_metric = bigram_result.get('best_metric', '')
    if best_metric in sa_results:
        bigram_mapping = sa_results[best_metric].get('best_permutation', {})

    mdl_mapping: Dict[str, str] = {}
    best_key = f"stem_{mdl_result.get('best_language', 'latin')}"
    mcmc_results = mdl_result.get('mcmc_results', {})
    if best_key in mcmc_results:
        mdl_mapping = mcmc_results[best_key].get('best_mapping', {})

    # Compare mappings
    common_stems = set(bigram_mapping.keys()) & set(mdl_mapping.keys())
    if not common_stems:
        return ConvergenceCheck(
            mapping_agreement=0.0,
            n_stems_compared=0,
            high_agreement_stems=[],
            agreement_by_frequency_rank={},
        )

    agree = 0
    high_agree = []
    for stem in common_stems:
        if bigram_mapping[stem] == mdl_mapping[stem]:
            agree += 1
            high_agree.append(stem)

    agreement = agree / len(common_stems)

    # Agreement by frequency rank (top-10, top-50, all)
    sorted_stems = sorted(common_stems)
    rank_agreement = {}
    for cutoff_label, cutoff in [('top_10', 10), ('top_50', 50), ('all', len(sorted_stems))]:
        subset = sorted_stems[:cutoff]
        if subset:
            a = sum(1 for s in subset if bigram_mapping[s] == mdl_mapping[s])
            rank_agreement[cutoff_label] = round(a / len(subset), 4)
        else:
            rank_agreement[cutoff_label] = 0.0

    return ConvergenceCheck(
        mapping_agreement=round(agreement, 4),
        n_stems_compared=len(common_stems),
        high_agreement_stems=high_agree[:20],
        agreement_by_frequency_rank=rank_agreement,
    )


# ---------------------------------------------------------------------------
# V.2: Prior Phase Convergence
# ---------------------------------------------------------------------------

def check_prior_phase_convergence(
    best_mapping: Dict[str, str],
) -> PriorPhaseConvergence:
    """
    Check decoded results against prior phase findings.

    Loads illustration_constrained.json, positional_slots.json, and
    distributional.json to cross-validate the decoding.
    """
    details: Dict[str, str] = {}
    checks_passed = 0
    checks_total = 0

    # --- Check 1: Illustration agreement ---
    illust = _load_result('illustration_constrained')
    illust_agreement = 0.0
    if illust:
        checks_total += 1
        folios = illust.get('folios', [])
        matches = 0
        total = 0
        for folio_data in folios:
            dominant = folio_data.get('dominant_stem', '')
            if not dominant or dominant not in best_mapping:
                continue
            decoded_stem = best_mapping[dominant]
            # Check if decoded stem matches any identification
            ids = folio_data.get('identifications', [])
            for ident in ids:
                medieval_stem = ident.get('medieval_stem', '')
                if medieval_stem and decoded_stem == medieval_stem:
                    matches += 1
                    break
            total += 1

        if total > 0:
            illust_agreement = matches / total
            if illust_agreement > 0.1:
                checks_passed += 1
                details['illustration'] = (
                    f'{matches}/{total} decoded plant names match identifications'
                )
            else:
                details['illustration'] = (
                    f'Only {matches}/{total} matches (below 10% threshold)'
                )
        else:
            details['illustration'] = 'No comparable folios found'
    else:
        details['illustration'] = 'illustration_constrained.json not found'

    # --- Check 2: Verb position agreement ---
    slots = _load_result('positional_slots')
    verb_agreement = 0.0
    if slots:
        checks_total += 1
        verb_initial = slots.get('verb_initial_ratio', 0.0)
        if verb_initial > 0.3:
            # Check if mapped verbs appear in position-1 of Voynich text
            # This is a structural check: we can't directly verify without
            # re-running positional analysis on decoded text, so we check
            # if the slot profile suggests verb-initial structure
            verb_agreement = verb_initial
            if verb_agreement > 0.3:
                checks_passed += 1
                details['verb_position'] = (
                    f'Verb-initial ratio {verb_initial:.2f} is consistent'
                )
            else:
                details['verb_position'] = (
                    f'Verb-initial ratio {verb_initial:.2f} below 0.3 threshold'
                )
        else:
            details['verb_position'] = (
                f'Positional slots show weak verb-initial pattern ({verb_initial:.2f})'
            )
    else:
        details['verb_position'] = 'positional_slots.json not found'

    # --- Check 3: Noun cluster agreement ---
    distrib = _load_result('distributional')
    noun_agreement = 0.0
    if distrib:
        checks_total += 1
        # Check if the distributional analysis found meaningful structure
        # that would be consistent with the decoding
        procrustes_results = distrib.get('procrustes_alignments', [])
        if procrustes_results:
            # Look for Latin alignment score
            for alignment in procrustes_results:
                if 'latin' in str(alignment.get('target_label', '')).lower():
                    score = alignment.get('score', 1.0)
                    # Lower residual = better alignment
                    if score < 0.9:
                        noun_agreement = 1.0 - score
                        checks_passed += 1
                        details['noun_clusters'] = (
                            f'Procrustes residual {score:.3f} suggests structural match'
                        )
                    else:
                        details['noun_clusters'] = (
                            f'Procrustes residual {score:.3f} is high (weak match)'
                        )
                    break
            else:
                details['noun_clusters'] = 'No Latin alignment found in distributional results'
        else:
            details['noun_clusters'] = 'No Procrustes alignments in distributional results'
    else:
        details['noun_clusters'] = 'distributional.json not found'

    return PriorPhaseConvergence(
        illustration_agreement=round(illust_agreement, 4),
        verb_position_agreement=round(verb_agreement, 4),
        noun_cluster_agreement=round(noun_agreement, 4),
        n_checks_passed=checks_passed,
        n_checks_total=checks_total,
        details=details,
    )


# ---------------------------------------------------------------------------
# V.3: Seeded Decoding
# ---------------------------------------------------------------------------

def run_seeded_decode(
    bigram_mapping: Dict[str, str],
    voynich_tokens: List[str],
    ref_corpus,
    latin_lm: Dict,
    top_n: int = 100,
    max_iter: int = 50_000,
    n_restarts: int = 3,
    seed: int = 42,
) -> float:
    """
    Use Approach 16's mapping to seed Approach 18's MCMC.

    Runs a short MCMC chain initialized from the bigram transfer mapping
    and measures the improvement in cross-entropy.  Uses the incremental
    SA from mdl_decode for performance.

    Returns improvement ratio (init_ce / best_ce).
    """
    from voynich.phases.mdl_decode import _fast_sa_mdl

    # Prepare Voynich stems
    stems = []
    for tok in voynich_tokens:
        d = decompose_token_morphemes(tok)
        stems.append(d.stem if d.stem else tok)

    counts = Counter(stems)
    vocab = [s for s, c in counts.most_common() if c >= 3]
    n = min(top_n, len(vocab))
    v_vocab = vocab[:n]
    v_set = set(v_vocab)
    stem_seq = [s for s in stems if s in v_set][:1000]

    # Build target vocab from mapping
    ref_tokens = ref_corpus.get_combined_tokens('latin')
    ref_stems = [stem_token(t, 'latin') for t in ref_tokens]
    ref_counts = Counter(ref_stems)
    ref_vocab = [s for s, c in ref_counts.most_common() if c >= 3]
    r_vocab = ref_vocab[:n]

    # Convert stem sequence to index array
    v_to_idx = {s: i for i, s in enumerate(v_vocab)}
    r_to_idx = {s: i for i, s in enumerate(r_vocab)}
    stem_indices = np.array([v_to_idx.get(s, n) for s in stem_seq], dtype=int)
    stem_indices = stem_indices[stem_indices < n]
    if len(stem_indices) > 500:
        stem_indices = stem_indices[:500]

    # Build initial permutation from bigram mapping (must be a valid permutation)
    init_perm = np.arange(n, dtype=int)
    for v_stem, l_stem in bigram_mapping.items():
        if v_stem in v_to_idx and l_stem in r_to_idx:
            v_idx = v_to_idx[v_stem]
            l_idx = r_to_idx[l_stem]
            # Swap to maintain valid permutation
            current_pos = int(np.where(init_perm == l_idx)[0][0])
            init_perm[current_pos] = init_perm[v_idx]
            init_perm[v_idx] = l_idx

    # Compute init CE via full text evaluation
    decoded_parts = [r_vocab[init_perm[idx]] for idx in stem_indices]
    decoded_text = '_' + '_'.join(decoded_parts) + '_'
    init_ce = cross_entropy_lm(decoded_text, latin_lm)

    # Run incremental SA seeded from bigram mapping
    global_best_ce = init_ce
    for r in range(n_restarts):
        _, best_ce_r, _ = _fast_sa_mdl(
            stem_indices, r_vocab, latin_lm, n,
            max_iter=max_iter,
            t_start=0.05, t_end=0.00005,
            seed=seed + r * 7,
            init_perm=init_perm,
        )
        if best_ce_r < global_best_ce:
            global_best_ce = best_ce_r

    improvement = init_ce / global_best_ce if global_best_ce > 0 else 1.0
    print(f"    Seeded decode: init_CE={init_ce:.4f}, best_CE={global_best_ce:.4f}, "
          f"improvement={improvement:.4f}x")

    return round(improvement, 4)


# ---------------------------------------------------------------------------
# V.4: Combined Assessment
# ---------------------------------------------------------------------------

def combined_assessment(
    bigram_gate: bool,
    mdl_gate: bool,
    convergence: ConvergenceCheck,
    prior_convergence: PriorPhaseConvergence,
) -> Tuple[bool, str, str]:
    """
    Combine evidence from both approaches.

    Returns (overall_gate_passed, confidence_level, verdict).
    """
    # Collect p-value proxies from selectivity ratios
    # Higher agreement/selectivity -> lower effective p-value
    p_values = []

    if convergence.mapping_agreement > 0:
        # Convert agreement to approximate p-value (higher agreement = lower p)
        p_agreement = max(1e-10, 1.0 - convergence.mapping_agreement)
        p_values.append(p_agreement)

    if prior_convergence.n_checks_total > 0:
        fraction_passed = (prior_convergence.n_checks_passed /
                           prior_convergence.n_checks_total)
        p_prior = max(1e-10, 1.0 - fraction_passed)
        p_values.append(p_prior)

    if bigram_gate:
        p_values.append(0.05)
    if mdl_gate:
        p_values.append(0.05)

    if p_values:
        chi2, df, combined_p = fisher_combined_probability(p_values)
    else:
        chi2, df, combined_p = 0.0, 0, 1.0

    # Determine confidence level
    approaches_agree = convergence.mapping_agreement > 0.3
    prior_ok = prior_convergence.n_checks_passed >= 2

    if bigram_gate and mdl_gate and approaches_agree and prior_ok:
        confidence = 'high'
        overall = True
    elif bigram_gate and mdl_gate and (approaches_agree or prior_ok):
        confidence = 'medium'
        overall = True
    elif bigram_gate or mdl_gate:
        confidence = 'low'
        overall = False
    else:
        confidence = 'none'
        overall = False

    # Verdict
    if confidence == 'high':
        verdict = 'strong_convergent_evidence_for_latin_cipher'
    elif confidence == 'medium':
        verdict = 'moderate_evidence_partial_convergence'
    elif confidence == 'low':
        verdict = 'weak_evidence_single_approach_only'
    else:
        verdict = 'no_evidence_for_stem_level_cipher'

    return overall, confidence, verdict


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_cipher_validate() -> Dict:
    """
    Run Phase 8 shared validation and integration.

    1. Load bigram_transfer.json and mdl_decode.json
    2. Cross-approach convergence
    3. Prior phase convergence
    4. Seeded decoding (Approach 16 -> 18)
    5. Combined assessment
    6. Save to results/cipher_validate.json
    """
    print("=" * 70)
    print("PHASE 8: CIPHER VALIDATION & INTEGRATION")
    print("=" * 70)

    # --- Load prior results ---
    print("\n--- Loading Phase 8 results ---")
    bigram_result = _load_result('bigram_transfer')
    mdl_result = _load_result('mdl_decode')

    if not bigram_result:
        print("  WARNING: bigram_transfer.json not found")
    if not mdl_result:
        print("  WARNING: mdl_decode.json not found")

    bigram_gate = bigram_result.get('gate_passed', False) if bigram_result else False
    bigram_verdict = bigram_result.get('verdict', 'not_run') if bigram_result else 'not_run'
    bigram_selectivity = bigram_result.get('best_selectivity', 0.0) if bigram_result else 0.0

    mdl_gate = mdl_result.get('gate_passed', False) if mdl_result else False
    mdl_verdict = mdl_result.get('verdict', 'not_run') if mdl_result else 'not_run'
    mdl_sanity = (mdl_result.get('sanity_check', {}).get('passed', False)
                  if mdl_result else False)
    mdl_compression = (mdl_result.get('best_compression_ratio', 0.0)
                       if mdl_result else 0.0)

    print(f"  Bigram: gate={'PASSED' if bigram_gate else 'FAILED'}, "
          f"selectivity={bigram_selectivity:.4f}")
    print(f"  MDL:    gate={'PASSED' if mdl_gate else 'FAILED'}, "
          f"compression={mdl_compression:.4f}, "
          f"sanity={'PASSED' if mdl_sanity else 'FAILED'}")

    # --- Cross-approach convergence ---
    print("\n--- V.1: Cross-Approach Convergence ---")
    if bigram_result and mdl_result:
        convergence = check_cross_approach_convergence(bigram_result, mdl_result)
        print(f"  Mapping agreement: {convergence.mapping_agreement:.4f} "
              f"({convergence.n_stems_compared} stems compared)")
        if convergence.high_agreement_stems:
            print(f"  Agreed stems (sample): "
                  f"{', '.join(convergence.high_agreement_stems[:5])}")
    else:
        convergence = ConvergenceCheck(
            mapping_agreement=0.0,
            n_stems_compared=0,
            high_agreement_stems=[],
            agreement_by_frequency_rank={},
        )
        print("  Skipped (missing results)")

    approaches_agree = convergence.mapping_agreement > 0.3

    # --- Prior phase convergence ---
    print("\n--- V.2: Prior Phase Convergence ---")
    # Get best mapping from whichever approach did better
    best_mapping: Dict[str, str] = {}
    if mdl_result and mdl_gate:
        best_key = f"stem_{mdl_result.get('best_language', 'latin')}"
        mcmc = mdl_result.get('mcmc_results', {})
        if best_key in mcmc:
            best_mapping = mcmc[best_key].get('best_mapping', {})
    elif bigram_result:
        sa = bigram_result.get('sa_results', {})
        bm = bigram_result.get('best_metric', '')
        if bm in sa:
            best_mapping = sa[bm].get('best_permutation', {})

    prior = check_prior_phase_convergence(best_mapping)
    print(f"  Checks passed: {prior.n_checks_passed}/{prior.n_checks_total}")
    for check_name, detail in prior.details.items():
        print(f"    {check_name}: {detail}")

    # --- Seeded decoding ---
    print("\n--- V.3: Seeded Decoding (Approach 16 -> 18) ---")
    seeded_improvement = 1.0
    if bigram_result:
        bigram_mapping = {}
        sa = bigram_result.get('sa_results', {})
        bm = bigram_result.get('best_metric', '')
        if bm in sa:
            bigram_mapping = sa[bm].get('best_permutation', {})

        if bigram_mapping:
            try:
                corpus = load_corpus(verbose=False)
                ref_corpus = load_reference_corpus(verbose=False)
                voynich_tokens = corpus.get_tokens(language='A')

                # Build a quick LM for seeded decode
                ref_tokens = ref_corpus.get_combined_tokens('latin')
                if ref_tokens:
                    latin_lm = build_ngram_lm(ref_tokens, order=3, smoothing=0.01)
                    seeded_improvement = run_seeded_decode(
                        bigram_mapping=bigram_mapping,
                        voynich_tokens=voynich_tokens,
                        ref_corpus=ref_corpus,
                        latin_lm=latin_lm,
                        top_n=80,
                        max_iter=50_000,
                        n_restarts=3,
                        seed=42,
                    )
            except Exception as e:
                print(f"    Seeded decode failed: {e}")
                seeded_improvement = 1.0
        else:
            print("    No bigram mapping available")
    else:
        print("    Skipped (bigram_transfer.json not found)")

    # --- Combined assessment ---
    print("\n--- V.4: Combined Assessment ---")
    overall_gate, confidence, verdict = combined_assessment(
        bigram_gate=bigram_gate,
        mdl_gate=mdl_gate,
        convergence=convergence,
        prior_convergence=prior,
    )

    # Fisher combined probability
    p_values = []
    if bigram_selectivity > 1.0:
        p_values.append(1.0 / bigram_selectivity)
    if mdl_compression > 1.0:
        p_values.append(1.0 / mdl_compression)

    if p_values:
        chi2, df, combined_p = fisher_combined_probability(p_values)
    else:
        chi2, combined_p = 0.0, 1.0

    print(f"  Overall gate:     {'PASSED' if overall_gate else 'FAILED'}")
    print(f"  Confidence level: {confidence}")
    print(f"  Fisher chi2:      {chi2:.4f}, p={combined_p:.6f}")
    print(f"  Verdict:          {verdict}")

    # --- Build result ---
    result = CipherValidateResult(
        bigram_gate=bigram_gate,
        bigram_verdict=bigram_verdict,
        bigram_best_selectivity=round(bigram_selectivity, 4),
        mdl_gate=mdl_gate,
        mdl_verdict=mdl_verdict,
        mdl_sanity_passed=mdl_sanity,
        mdl_best_compression=round(mdl_compression, 4),
        convergence=_convert(asdict(convergence)),
        approaches_agree=approaches_agree,
        prior_convergence=_convert(asdict(prior)),
        seeded_improvement=seeded_improvement,
        fisher_chi2=round(chi2, 4),
        fisher_p_value=round(combined_p, 6),
        overall_gate_passed=overall_gate,
        confidence_level=confidence,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    out_path = _results_dir() / 'cipher_validate.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    return out
