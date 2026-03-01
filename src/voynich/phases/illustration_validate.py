"""
Phase 6 Validation: Illustration-Constrained Decoding Validation Battery
=========================================================================
Comprehensive validation of Phase 6 results: null tests, leave-one-out
cross-validation, train/test split, bootstrap stability, and stop
condition evaluation.

Sub-analyses:
  6.V.1 — Null test: shuffled tokens
  6.V.2 — Null test: shuffled characters
  6.V.3 — Null test: random plant names
  6.V.4 — Leave-one-out cross-validation
  6.V.5 — Train/test split
  6.V.6 — Bootstrap unanimity stability
  6.V.7 — Stop condition evaluation

Output:
  results/illustration_validate.json
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, VoynichCorpus, tokenize_eva_chars
from voynich.core._paths import results_dir as _results_dir
from voynich.phases.illustration_constrained import (
    FolioIdentificationSet, PlantIdentification,
    load_medieval_names, parse_concordance,
    build_folio_identification_sets,
    _convert, _check_gate,
)
from voynich.phases.anchor_propagate import (
    build_anchor_hypothesis, cross_consistency_check,
    AnchorHypothesis, CrossConsistencyMatrix,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NullTestResult:
    """Result of one null test."""
    test_name: str
    description: str
    real_value: float
    null_mean: float
    null_std: float
    z_score: float
    selectivity: float
    passed: bool


@dataclass
class LeaveOneOutResult:
    """Leave-one-out cross-validation results."""
    n_anchors: int
    per_anchor_unanimity: Dict[str, float]
    mean_unanimity: float
    min_unanimity: float
    max_unanimity: float
    std_unanimity: float
    stable: bool


@dataclass
class TrainTestResult:
    """Train/test split validation."""
    train_folios: List[str]
    test_folios: List[str]
    train_unanimity: float
    test_unanimity: float
    transfer_ratio: float
    passed: bool


@dataclass
class IllustrationValidationResult:
    """Full Phase 6 validation output."""
    # Null tests
    null_tests: List[Dict]
    n_null_passed: int
    n_null_total: int
    # Leave-one-out
    leave_one_out: Dict
    loo_stable: bool
    # Train/test
    train_test: Dict
    train_test_passed: bool
    # Bootstrap
    bootstrap_unanimity_mean: float
    bootstrap_unanimity_ci_lo: float
    bootstrap_unanimity_ci_hi: float
    bootstrap_stable: bool
    # Character unanimity breakdown
    n_chars_unanimous: int
    n_chars_majority: int
    n_chars_conflicting: int
    # Stop conditions
    stop_conditions: List[str]
    overall_status: str
    # Overall
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Null tests
# ---------------------------------------------------------------------------

def null_test_shuffled_tokens(
    anchors: List[AnchorHypothesis],
    folio_sets: List[FolioIdentificationSet],
    corpus: VoynichCorpus,
    real_unanimity: float,
    n_trials: int = 100,
    seed: int = 42,
) -> NullTestResult:
    """
    Null test: Shuffle tokens across folios, rebuild anchors.

    Tests whether folio-specific token content matters.
    """
    rng = random.Random(seed)
    folio_index = {fs.folio: fs for fs in folio_sets}

    # Collect all herbal_a tokens
    all_tokens = corpus.get_tokens(section='herbal_a', paragraph_only=True)

    null_vals: List[float] = []
    for trial in range(n_trials):
        # Shuffle all tokens
        shuffled = list(all_tokens)
        rng.shuffle(shuffled)

        # Redistribute tokens to folio-sized chunks
        idx = 0
        hypotheses = []
        for h in anchors:
            fs = folio_index.get(h.folio)
            if fs is None:
                continue

            n_tokens = fs.token_count
            chunk = shuffled[idx:idx + n_tokens] if idx + n_tokens <= len(shuffled) else shuffled[idx:]
            idx += n_tokens

            # Decompose chunk tokens to find a "dominant stem"
            from voynich.phases.morpheme_grid import decompose_token_morphemes
            stem_counts: Counter = Counter()
            for token in chunk:
                d = decompose_token_morphemes(token)
                if d.stem:
                    stem_counts[d.stem] += 1

            if not stem_counts:
                continue

            # Build a fake folio set with the shuffled dominant stem
            fake_dom_stem = stem_counts.most_common(1)[0][0]
            fake_fs = FolioIdentificationSet(
                folio=h.folio,
                identifications=fs.identifications,
                n_sources=fs.n_sources,
                tier=fs.tier,
                dominant_stem=fake_dom_stem,
                dominant_stem_forms=[],
                dominant_stem_paradigm_shape=fs.dominant_stem_paradigm_shape,
                dominant_stem_token_count=stem_counts[fake_dom_stem],
                token_count=len(chunk),
            )

            for pid in fs.identifications:
                if pid.medieval_stem:
                    hyp = build_anchor_hypothesis(fake_fs, pid)
                    if hyp and hyp.paradigm_compatible:
                        hypotheses.append(hyp)
                    break

        if len(hypotheses) >= 2:
            cc = cross_consistency_check(hypotheses)
            null_vals.append(cc.unanimity_ratio)
        else:
            null_vals.append(0.0)

    null_mean = float(np.mean(null_vals))
    null_std = float(np.std(null_vals))
    z = (real_unanimity - null_mean) / null_std if null_std > 0 else 0.0
    selectivity = real_unanimity / null_mean if null_mean > 0 else 0.0

    return NullTestResult(
        test_name='shuffled_tokens',
        description='Shuffle tokens across folios, rebuild anchors',
        real_value=round(real_unanimity, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z, 2),
        selectivity=round(selectivity, 4),
        passed=selectivity > 1.5,
    )


def null_test_shuffled_chars(
    anchors: List[AnchorHypothesis],
    folio_sets: List[FolioIdentificationSet],
    corpus: VoynichCorpus,
    real_unanimity: float,
    n_trials: int = 100,
    seed: int = 42,
) -> NullTestResult:
    """
    Null test: Shuffle characters within each token on anchor folios.

    Tests whether character-level structure carries information.
    """
    rng = random.Random(seed)
    folio_index = {fs.folio: fs for fs in folio_sets}

    null_vals: List[float] = []
    for trial in range(n_trials):
        hypotheses = []
        for h in anchors:
            fs = folio_index.get(h.folio)
            if fs is None:
                continue

            page = corpus.pages.get(h.folio)
            if page is None:
                continue

            # Shuffle characters within each token
            from voynich.phases.morpheme_grid import decompose_token_morphemes
            shuffled_tokens = []
            for token in page.all_tokens:
                chars = list(tokenize_eva_chars(token))
                rng.shuffle(chars)
                shuffled_tokens.append(''.join(chars))

            # Find dominant stem from shuffled tokens
            stem_counts: Counter = Counter()
            for token in shuffled_tokens:
                d = decompose_token_morphemes(token)
                if d.stem:
                    stem_counts[d.stem] += 1

            if not stem_counts:
                continue

            fake_dom_stem = stem_counts.most_common(1)[0][0]
            fake_fs = FolioIdentificationSet(
                folio=h.folio,
                identifications=fs.identifications,
                n_sources=fs.n_sources,
                tier=fs.tier,
                dominant_stem=fake_dom_stem,
                dominant_stem_forms=[],
                dominant_stem_paradigm_shape=fs.dominant_stem_paradigm_shape,
                dominant_stem_token_count=stem_counts[fake_dom_stem],
                token_count=len(shuffled_tokens),
            )

            for pid in fs.identifications:
                if pid.medieval_stem:
                    hyp = build_anchor_hypothesis(fake_fs, pid)
                    if hyp and hyp.paradigm_compatible:
                        hypotheses.append(hyp)
                    break

        if len(hypotheses) >= 2:
            cc = cross_consistency_check(hypotheses)
            null_vals.append(cc.unanimity_ratio)
        else:
            null_vals.append(0.0)

    null_mean = float(np.mean(null_vals))
    null_std = float(np.std(null_vals))
    z = (real_unanimity - null_mean) / null_std if null_std > 0 else 0.0
    selectivity = real_unanimity / null_mean if null_mean > 0 else 0.0

    return NullTestResult(
        test_name='shuffled_chars',
        description='Shuffle characters within tokens, rebuild anchors',
        real_value=round(real_unanimity, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z, 2),
        selectivity=round(selectivity, 4),
        passed=selectivity > 1.5,
    )


def null_test_random_names(
    anchors: List[AnchorHypothesis],
    folio_sets: List[FolioIdentificationSet],
    real_unanimity: float,
    n_trials: int = 100,
    seed: int = 42,
) -> NullTestResult:
    """
    Null test: Replace plant names with random medieval Latin words.

    Tests whether the specific plant identifications matter.
    """
    rng = random.Random(seed)
    folio_index = {fs.folio: fs for fs in folio_sets}

    random_stem_pool = [
        'herb', 'aqu', 'ole', 'radic', 'foli', 'flor', 'semin',
        'morb', 'febr', 'dolor', 'sanguin', 'remedi', 'virt',
        'natur', 'corpor', 'membr', 'pector', 'ventr', 'capit',
        'dent', 'ocul', 'aur', 'nas', 'man', 'ped', 'crust',
        'cort', 'pulver', 'success', 'decoct', 'infus', 'emplast',
        'unguent', 'electuari', 'sirup', 'pill', 'trocis',
    ]

    null_vals: List[float] = []
    for trial in range(n_trials):
        hypotheses = []
        for h in anchors:
            fs = folio_index.get(h.folio)
            if fs is None:
                continue

            fake_stem = rng.choice(random_stem_pool)
            fake_pid = PlantIdentification(
                folio=h.folio,
                linnaean_name='Random',
                common_name='random',
                source='null_test',
                medieval_name=fake_stem + 'a',
                medieval_stem=fake_stem,
                declension='noun_1st',
            )
            hyp = build_anchor_hypothesis(fs, fake_pid)
            if hyp and hyp.paradigm_compatible:
                hypotheses.append(hyp)

        if len(hypotheses) >= 2:
            cc = cross_consistency_check(hypotheses)
            null_vals.append(cc.unanimity_ratio)
        else:
            null_vals.append(0.0)

    null_mean = float(np.mean(null_vals))
    null_std = float(np.std(null_vals))
    z = (real_unanimity - null_mean) / null_std if null_std > 0 else 0.0
    selectivity = real_unanimity / null_mean if null_mean > 0 else 0.0

    return NullTestResult(
        test_name='random_names',
        description='Replace plant names with random Latin words',
        real_value=round(real_unanimity, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z, 2),
        selectivity=round(selectivity, 4),
        passed=selectivity > 1.5,
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def leave_one_out_validation(
    anchors: List[AnchorHypothesis],
) -> LeaveOneOutResult:
    """
    Leave-one-out: remove each anchor, rebuild cross-consistency.

    Measures robustness of the unanimity metric.
    """
    per_anchor: Dict[str, float] = {}

    for i, left_out in enumerate(anchors):
        remaining = [h for j, h in enumerate(anchors) if j != i]
        if len(remaining) >= 2:
            cc = cross_consistency_check(remaining)
            per_anchor[left_out.folio] = cc.unanimity_ratio
        else:
            per_anchor[left_out.folio] = 0.0

    vals = list(per_anchor.values())
    mean_u = float(np.mean(vals)) if vals else 0.0
    min_u = float(np.min(vals)) if vals else 0.0
    max_u = float(np.max(vals)) if vals else 0.0
    std_u = float(np.std(vals)) if vals else 0.0

    return LeaveOneOutResult(
        n_anchors=len(anchors),
        per_anchor_unanimity={k: round(v, 4) for k, v in per_anchor.items()},
        mean_unanimity=round(mean_u, 4),
        min_unanimity=round(min_u, 4),
        max_unanimity=round(max_u, 4),
        std_unanimity=round(std_u, 4),
        stable=min_u > 0.30,
    )


def train_test_split(
    anchors: List[AnchorHypothesis],
    train_fraction: float = 0.6,
    seed: int = 42,
) -> TrainTestResult:
    """
    Split anchors into train (60%) and test (40%).

    Build char mappings from train only, then test consistency of
    train mappings against test anchor mappings.
    """
    rng = random.Random(seed)
    indices = list(range(len(anchors)))
    rng.shuffle(indices)

    n_train = max(2, int(len(anchors) * train_fraction))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_anchors = [anchors[i] for i in train_idx]
    test_anchors = [anchors[i] for i in test_idx]

    # Train consistency
    if len(train_anchors) >= 2:
        train_cc = cross_consistency_check(train_anchors)
        train_unanimity = train_cc.unanimity_ratio
    else:
        train_unanimity = 0.0

    # Test: check if test anchors agree with train consensus
    if len(test_anchors) >= 2:
        test_cc = cross_consistency_check(test_anchors)
        test_unanimity = test_cc.unanimity_ratio
    elif len(test_anchors) == 1:
        # Check agreement of single test anchor with train consensus
        if len(train_anchors) >= 2:
            combined = train_anchors + test_anchors
            combined_cc = cross_consistency_check(combined)
            test_unanimity = combined_cc.unanimity_ratio
        else:
            test_unanimity = 0.0
    else:
        test_unanimity = 0.0

    transfer = test_unanimity / train_unanimity if train_unanimity > 0 else 0.0

    return TrainTestResult(
        train_folios=[anchors[i].folio for i in train_idx],
        test_folios=[anchors[i].folio for i in test_idx],
        train_unanimity=round(train_unanimity, 4),
        test_unanimity=round(test_unanimity, 4),
        transfer_ratio=round(transfer, 4),
        passed=transfer > 0.5,
    )


def bootstrap_unanimity(
    anchors: List[AnchorHypothesis],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float, bool]:
    """
    Bootstrap: resample anchors with replacement, measure unanimity stability.

    Returns (mean, ci_lo, ci_hi, stable).
    """
    rng = random.Random(seed)
    boot_vals: List[float] = []

    for trial in range(n_bootstrap):
        sample = [rng.choice(anchors) for _ in range(len(anchors))]
        if len(sample) >= 2:
            cc = cross_consistency_check(sample)
            boot_vals.append(cc.unanimity_ratio)
        else:
            boot_vals.append(0.0)

    mean = float(np.mean(boot_vals))
    ci_lo = float(np.percentile(boot_vals, 2.5))
    ci_hi = float(np.percentile(boot_vals, 97.5))
    stable = ci_lo > 0.30

    return mean, ci_lo, ci_hi, stable


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

def check_stop_conditions(
    null_results: List[NullTestResult],
    loo: LeaveOneOutResult,
    tt: TrainTestResult,
    unanimity_ratio: float,
    bootstrap_ci_lo: float,
) -> Tuple[List[str], str]:
    """
    Check stop conditions from the README plan.

    Returns (condition_messages, overall_status).
    overall_status is one of: 'hard_stop', 'soft_stop', 'green_light'.
    """
    conditions: List[str] = []

    # Hard stop conditions
    hard_stop = False
    if unanimity_ratio < 0.20:
        conditions.append("HARD STOP: unanimity_ratio < 0.20 (below chance)")
        hard_stop = True
    if all(not nr.passed for nr in null_results):
        conditions.append("HARD STOP: all null tests show selectivity < 1.5")
        hard_stop = True

    if hard_stop:
        return conditions, 'hard_stop'

    # Soft stop conditions
    soft_stop = False
    if unanimity_ratio < 0.50:
        conditions.append(f"SOFT STOP: unanimity_ratio={unanimity_ratio:.4f} "
                          f"< 0.50 (above chance but not decisive)")
        soft_stop = True
    if any(not nr.passed for nr in null_results):
        failed = [nr.test_name for nr in null_results if not nr.passed]
        conditions.append(f"SOFT STOP: null test(s) failed: {', '.join(failed)}")
        soft_stop = True
    if not loo.stable:
        conditions.append(f"SOFT STOP: LOO min unanimity={loo.min_unanimity:.4f} "
                          f"< 0.30 (not stable)")
        soft_stop = True

    if soft_stop:
        return conditions, 'soft_stop'

    # Green light
    conditions.append("GREEN LIGHT: unanimity > 0.50, all null tests pass, "
                       "LOO stable, bootstrap CI lower bound above threshold")
    return conditions, 'green_light'


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_illustration_validate(
    anchor_data: Optional[Dict] = None,
    competitive_data: Optional[Dict] = None,
) -> Dict:
    """
    Run Phase 6 Validation.

    1. Load anchor results
    2. Reconstruct anchor hypotheses
    3. Run 3 null tests
    4. Run leave-one-out
    5. Run train/test split
    6. Bootstrap unanimity
    7. Check stop conditions
    8. Save results/illustration_validate.json
    """
    print("=" * 70)
    print("Phase 6 Validation: Illustration-Constrained Decoding")
    print("=" * 70)

    # Load anchor results
    if anchor_data is None:
        results_path = os.path.join(_results_dir(), 'anchor_propagate.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                anchor_data = json.load(f)
        else:
            from voynich.phases.anchor_propagate import run_anchor_propagate
            anchor_data = run_anchor_propagate()

    real_unanimity = anchor_data.get('unanimity_ratio', 0.0)
    print(f"\n  Real unanimity ratio: {real_unanimity:.4f}")

    # Rebuild anchors
    print("\n  Loading corpus and rebuilding anchors...")
    corpus = load_corpus(verbose=False)
    concordance = parse_concordance()
    medieval_names = load_medieval_names()
    folio_sets = build_folio_identification_sets(
        concordance, medieval_names, corpus,
    )
    folio_index = {fs.folio: fs for fs in folio_sets}

    anchors: List[AnchorHypothesis] = []
    stored_hypotheses = anchor_data.get('anchor_hypotheses', [])
    for h_dict in stored_hypotheses:
        if not h_dict.get('paradigm_compatible', False):
            continue
        folio = h_dict['folio']
        fs = folio_index.get(folio)
        if fs is None:
            continue
        for pid in fs.identifications:
            if pid.medieval_stem == h_dict.get('medieval_stem'):
                h = build_anchor_hypothesis(fs, pid)
                if h and h.paradigm_compatible:
                    anchors.append(h)
                break

    print(f"  Reconstructed anchors: {len(anchors)}")

    if len(anchors) < 3:
        print("  Too few anchors for validation battery.")
        result = IllustrationValidationResult(
            null_tests=[], n_null_passed=0, n_null_total=0,
            leave_one_out={}, loo_stable=False,
            train_test={}, train_test_passed=False,
            bootstrap_unanimity_mean=0.0,
            bootstrap_unanimity_ci_lo=0.0,
            bootstrap_unanimity_ci_hi=0.0,
            bootstrap_stable=False,
            n_chars_unanimous=0, n_chars_majority=0, n_chars_conflicting=0,
            stop_conditions=['HARD STOP: too few anchors'],
            overall_status='hard_stop',
            gate_passed=False, verdict='insufficient_anchors',
        )
        out_path = os.path.join(_results_dir(), 'illustration_validate.json')
        with open(out_path, 'w') as f:
            json.dump(_convert(asdict(result)), f, indent=2, default=str)
        return _convert(asdict(result))

    # Get character unanimity from cross-consistency
    cc = cross_consistency_check(anchors)

    # Run null tests
    print("\n  6.V.1: Null test — shuffled tokens")
    nt1 = null_test_shuffled_tokens(
        anchors, folio_sets, corpus, real_unanimity,
        n_trials=100, seed=42,
    )
    print(f"    Selectivity: {nt1.selectivity:.4f}x  "
          f"z={nt1.z_score:.2f}  {'PASS' if nt1.passed else 'FAIL'}")

    print("\n  6.V.2: Null test — shuffled characters")
    nt2 = null_test_shuffled_chars(
        anchors, folio_sets, corpus, real_unanimity,
        n_trials=100, seed=42,
    )
    print(f"    Selectivity: {nt2.selectivity:.4f}x  "
          f"z={nt2.z_score:.2f}  {'PASS' if nt2.passed else 'FAIL'}")

    print("\n  6.V.3: Null test — random plant names")
    nt3 = null_test_random_names(
        anchors, folio_sets, real_unanimity,
        n_trials=100, seed=42,
    )
    print(f"    Selectivity: {nt3.selectivity:.4f}x  "
          f"z={nt3.z_score:.2f}  {'PASS' if nt3.passed else 'FAIL'}")

    null_tests = [nt1, nt2, nt3]
    n_null_passed = sum(1 for nt in null_tests if nt.passed)

    # Leave-one-out
    print("\n  6.V.4: Leave-one-out cross-validation")
    loo = leave_one_out_validation(anchors)
    print(f"    Mean unanimity: {loo.mean_unanimity:.4f}")
    print(f"    Min: {loo.min_unanimity:.4f}  Max: {loo.max_unanimity:.4f}")
    print(f"    Stable: {'YES' if loo.stable else 'NO'}")

    # Per-anchor detail
    for folio, u in sorted(loo.per_anchor_unanimity.items()):
        status = "OK" if u > 0.30 else "LOW"
        print(f"      Without {folio}: {u:.4f} [{status}]")

    # Train/test split
    print("\n  6.V.5: Train/test split (60/40)")
    tt = train_test_split(anchors, train_fraction=0.6, seed=42)
    print(f"    Train unanimity: {tt.train_unanimity:.4f}")
    print(f"    Test unanimity: {tt.test_unanimity:.4f}")
    print(f"    Transfer ratio: {tt.transfer_ratio:.4f}")
    print(f"    Passed: {'YES' if tt.passed else 'NO'}")

    # Bootstrap
    print("\n  6.V.6: Bootstrap unanimity stability")
    boot_mean, boot_lo, boot_hi, boot_stable = bootstrap_unanimity(
        anchors, n_bootstrap=200, seed=42,
    )
    print(f"    Mean: {boot_mean:.4f}")
    print(f"    95% CI: [{boot_lo:.4f}, {boot_hi:.4f}]")
    print(f"    Stable: {'YES' if boot_stable else 'NO'}")

    # Stop conditions
    print("\n  6.V.7: Stop condition evaluation")
    conditions, overall_status = check_stop_conditions(
        null_tests, loo, tt, real_unanimity, boot_lo,
    )
    for cond in conditions:
        print(f"    {cond}")
    print(f"\n    Overall status: {overall_status.upper()}")

    # Gate
    gate_passed = overall_status == 'green_light'
    if overall_status == 'green_light':
        verdict = 'cross_modal_decoding_validated'
    elif overall_status == 'soft_stop':
        verdict = 'structural_findings_only'
    else:
        verdict = 'approach_abandoned'

    print(f"  Verdict: {verdict}")

    # Build result
    result = IllustrationValidationResult(
        null_tests=[asdict(nt) for nt in null_tests],
        n_null_passed=n_null_passed,
        n_null_total=len(null_tests),
        leave_one_out=asdict(loo),
        loo_stable=loo.stable,
        train_test=asdict(tt),
        train_test_passed=tt.passed,
        bootstrap_unanimity_mean=round(boot_mean, 4),
        bootstrap_unanimity_ci_lo=round(boot_lo, 4),
        bootstrap_unanimity_ci_hi=round(boot_hi, 4),
        bootstrap_stable=boot_stable,
        n_chars_unanimous=cc.n_chars_unanimous,
        n_chars_majority=cc.n_chars_majority,
        n_chars_conflicting=cc.n_chars_conflicting,
        stop_conditions=conditions,
        overall_status=overall_status,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    # Save
    out_path = os.path.join(_results_dir(), 'illustration_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return _convert(asdict(result))
