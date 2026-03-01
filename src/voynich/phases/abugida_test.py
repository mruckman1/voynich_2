"""
Phase 4 Step 3: Abugida Hypothesis Test
=========================================
Test whether the Voynich script is an abugida (consonant base + vowel modifier)
rather than a pure syllabary or alphabet.

Sub-analyses:
  3A — Decompose each glyph into onset/nucleus strokes, compute positional entropy
  3B — Conditional entropy H(nucleus|onset) vs H(nucleus), compute MI
  3C — Compare against reference profiles for alphabet, syllabary, abugida, abjad

Key diagnostic: R = 1 - H(nucleus|onset) / H(nucleus)
  R near 0   -> onset and nucleus are independent (alphabet or pure syllabary)
  R > 0.2    -> onset partially predicts nucleus (abugida-like)
  R > 0.5    -> strong dependency (unusual, needs investigation)

Output:
  abugida_test.json — entropy decomposition, script type classification
"""

import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core._paths import results_dir as _results_dir
from voynich.analysis.strokes import decompose_glyph, Stroke


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OnsetNucleusEntropy:
    """Entropy decomposition of onset and nucleus components."""
    h_onset: float
    h_nucleus: float
    h_joint: float
    h_onset_given_nucleus: float
    h_nucleus_given_onset: float
    mi_onset_nucleus: float
    reduction_r: float          # 1 - H(nucleus|onset)/H(nucleus)
    reverse_r: float            # 1 - H(onset|nucleus)/H(onset)
    n_onset_types: int
    n_nucleus_types: int
    n_observations: int


@dataclass
class PositionalEntropyCurve:
    """Entropy of onset/nucleus at each glyph position within token."""
    position: int
    h_onset: float
    h_nucleus: float
    n_observations: int


@dataclass
class ScriptTypeReference:
    """Expected entropy profile for a script type."""
    script_type: str
    description: str
    expected_r_low: float
    expected_r_high: float
    expected_ratio_low: float   # H(onset)/H(nucleus) range
    expected_ratio_high: float


@dataclass
class ScriptTypeComparison:
    """Comparison of Voynich against a script type reference."""
    script_type: str
    r_in_range: bool
    r_distance: float
    overall_match: str  # 'match', 'partial', 'mismatch'


@dataclass
class AbugidaTestResult:
    """Full abugida hypothesis test output."""
    entropy: OnsetNucleusEntropy
    positional_curves: List[PositionalEntropyCurve]
    script_comparisons: List[ScriptTypeComparison]
    best_match_script: str
    mi_ci_lower: float
    mi_ci_upper: float
    r_ci_lower: float
    r_ci_upper: float
    null_mi_mean: float
    null_mi_std: float
    mi_z_score: float
    verdict: str


# ---------------------------------------------------------------------------
# Phase 3A: Onset/Nucleus decomposition
# ---------------------------------------------------------------------------

def decompose_tokens_onset_nucleus(
    tokens: List[str],
) -> List[Tuple[str, str]]:
    """
    Decompose each glyph into (onset_stroke, nucleus_stroke) pairs.

    For each EVA character in each token:
    - onset = first stroke in the decomposition
    - nucleus = last stroke in the decomposition
    - If the glyph has only one stroke, onset == nucleus
    """
    pairs = []
    for token in tokens:
        eva_chars = tokenize_eva_chars(token)
        for ec in eva_chars:
            strokes = decompose_glyph(ec)
            if strokes:
                onset = strokes[0].name
                nucleus = strokes[-1].name
                pairs.append((onset, nucleus))
    return pairs


def decompose_by_position(
    tokens: List[str],
) -> Dict[int, List[Tuple[str, str]]]:
    """Group onset/nucleus pairs by glyph position within token."""
    by_pos: Dict[int, List[Tuple[str, str]]] = {}
    for token in tokens:
        eva_chars = tokenize_eva_chars(token)
        for pos, ec in enumerate(eva_chars):
            strokes = decompose_glyph(ec)
            if strokes:
                onset = strokes[0].name
                nucleus = strokes[-1].name
                by_pos.setdefault(pos, []).append((onset, nucleus))
    return by_pos


# ---------------------------------------------------------------------------
# Phase 3B: Conditional entropy computation
# ---------------------------------------------------------------------------

def _entropy(counts: Counter) -> float:
    """Shannon entropy from a Counter."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def compute_onset_nucleus_entropy(
    pairs: List[Tuple[str, str]],
) -> OnsetNucleusEntropy:
    """Compute H(onset), H(nucleus), H(joint), conditional entropies, MI."""
    onset_counts = Counter(o for o, _ in pairs)
    nucleus_counts = Counter(n for _, n in pairs)
    joint_counts = Counter(pairs)

    h_onset = _entropy(onset_counts)
    h_nucleus = _entropy(nucleus_counts)
    h_joint = _entropy(joint_counts)

    h_onset_given_nucleus = h_joint - h_nucleus
    h_nucleus_given_onset = h_joint - h_onset
    mi = h_onset + h_nucleus - h_joint

    # Clamp small negative values from floating point
    h_onset_given_nucleus = max(0.0, h_onset_given_nucleus)
    h_nucleus_given_onset = max(0.0, h_nucleus_given_onset)
    mi = max(0.0, mi)

    r = 1.0 - (h_nucleus_given_onset / h_nucleus) if h_nucleus > 0 else 0.0
    reverse_r = 1.0 - (h_onset_given_nucleus / h_onset) if h_onset > 0 else 0.0

    return OnsetNucleusEntropy(
        h_onset=round(h_onset, 4),
        h_nucleus=round(h_nucleus, 4),
        h_joint=round(h_joint, 4),
        h_onset_given_nucleus=round(h_onset_given_nucleus, 4),
        h_nucleus_given_onset=round(h_nucleus_given_onset, 4),
        mi_onset_nucleus=round(mi, 4),
        reduction_r=round(r, 4),
        reverse_r=round(reverse_r, 4),
        n_onset_types=len(onset_counts),
        n_nucleus_types=len(nucleus_counts),
        n_observations=len(pairs),
    )


def compute_positional_entropy(
    by_pos: Dict[int, List[Tuple[str, str]]],
    max_pos: int = 10,
) -> List[PositionalEntropyCurve]:
    """Compute onset and nucleus entropy at each glyph position."""
    curves = []
    for pos in range(max_pos):
        pairs = by_pos.get(pos, [])
        if len(pairs) < 10:
            break
        onset_counts = Counter(o for o, _ in pairs)
        nucleus_counts = Counter(n for _, n in pairs)
        curves.append(PositionalEntropyCurve(
            position=pos,
            h_onset=round(_entropy(onset_counts), 4),
            h_nucleus=round(_entropy(nucleus_counts), 4),
            n_observations=len(pairs),
        ))
    return curves


# ---------------------------------------------------------------------------
# Phase 3C: Script type references and comparison
# ---------------------------------------------------------------------------

def build_script_type_references() -> List[ScriptTypeReference]:
    """
    Reference profiles for script types based on typological properties.

    Alphabet (Latin): Onset and nucleus strokes are independent letters.
      R near 0, H(onset)/H(nucleus) near 1.0.

    Pure syllabary (Linear B, kana): Each glyph is an atomic CV unit.
      If decomposed into strokes, onset and nucleus are constrained by the
      fixed glyph inventory. R moderate (0.2-0.5), ratio near 1.0.

    Abugida (Devanagari, Ethiopic): Consonant base with vowel modifier.
      Onset predicts nucleus partially. R > 0.2, onset H > nucleus H.

    Abjad (Arabic, Hebrew): Consonants primary, vowels secondary.
      H(onset) >> H(nucleus), R variable.
    """
    return [
        ScriptTypeReference(
            script_type='alphabet',
            description='Independent letters (Latin, Greek)',
            expected_r_low=-0.05,
            expected_r_high=0.15,
            expected_ratio_low=0.7,
            expected_ratio_high=1.3,
        ),
        ScriptTypeReference(
            script_type='syllabary',
            description='Atomic CV glyphs (Linear B, Kana)',
            expected_r_low=0.15,
            expected_r_high=0.55,
            expected_ratio_low=0.8,
            expected_ratio_high=1.2,
        ),
        ScriptTypeReference(
            script_type='abugida',
            description='Consonant base + vowel modifier (Devanagari)',
            expected_r_low=0.20,
            expected_r_high=0.70,
            expected_ratio_low=1.2,
            expected_ratio_high=3.0,
        ),
        ScriptTypeReference(
            script_type='abjad',
            description='Consonant-primary (Arabic, Hebrew)',
            expected_r_low=0.0,
            expected_r_high=0.30,
            expected_ratio_low=1.5,
            expected_ratio_high=5.0,
        ),
    ]


def compare_to_script_types(
    entropy: OnsetNucleusEntropy,
    references: List[ScriptTypeReference],
) -> List[ScriptTypeComparison]:
    """Compare Voynich R and H-ratio against each script type reference."""
    r = entropy.reduction_r
    comparisons = []

    for ref in references:
        r_in = ref.expected_r_low <= r <= ref.expected_r_high

        if r < ref.expected_r_low:
            r_dist = ref.expected_r_low - r
        elif r > ref.expected_r_high:
            r_dist = r - ref.expected_r_high
        else:
            r_dist = 0.0

        if r_in:
            match = 'match'
        elif r_dist < 0.1:
            match = 'partial'
        else:
            match = 'mismatch'

        comparisons.append(ScriptTypeComparison(
            script_type=ref.script_type,
            r_in_range=r_in,
            r_distance=round(r_dist, 4),
            overall_match=match,
        ))

    return comparisons


# ---------------------------------------------------------------------------
# Bootstrap and null testing
# ---------------------------------------------------------------------------

def bootstrap_onset_nucleus_ci(
    tokens: List[str],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Bootstrap CIs for MI and R by resampling tokens."""
    rng = random.Random(seed)
    mis = []
    rs = []

    for _ in range(n_bootstrap):
        sample = rng.choices(tokens, k=len(tokens))
        pairs = decompose_tokens_onset_nucleus(sample)
        ent = compute_onset_nucleus_entropy(pairs)
        mis.append(ent.mi_onset_nucleus)
        rs.append(ent.reduction_r)

    mi_arr = np.array(mis)
    r_arr = np.array(rs)
    mi_ci = (float(np.percentile(mi_arr, 2.5)), float(np.percentile(mi_arr, 97.5)))
    r_ci = (float(np.percentile(r_arr, 2.5)), float(np.percentile(r_arr, 97.5)))
    return mi_ci, r_ci


def null_baseline_mi(
    pairs: List[Tuple[str, str]],
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Null baseline MI: shuffle nucleus labels independently.
    If MI is real, it should exceed this null.
    """
    rng = random.Random(seed)
    null_mis = []

    onsets = [o for o, _ in pairs]
    nuclei = [n for _, n in pairs]

    for _ in range(n_trials):
        shuffled_nuclei = list(nuclei)
        rng.shuffle(shuffled_nuclei)
        shuffled_pairs = list(zip(onsets, shuffled_nuclei))
        ent = compute_onset_nucleus_entropy(shuffled_pairs)
        null_mis.append(ent.mi_onset_nucleus)

    arr = np.array(null_mis)
    return float(np.mean(arr)), float(np.std(arr))


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_results(result: AbugidaTestResult) -> None:
    """Print formatted abugida test results."""
    ent = result.entropy

    print("\n--- Phase 3A: Onset/Nucleus Decomposition ---")
    print(f"  Observations: {ent.n_observations}")
    print(f"  Onset types:  {ent.n_onset_types}")
    print(f"  Nucleus types: {ent.n_nucleus_types}")

    print("\n--- Phase 3B: Conditional Entropy ---")
    print(f"  H(onset):              {ent.h_onset:.4f} bits")
    print(f"  H(nucleus):            {ent.h_nucleus:.4f} bits")
    print(f"  H(onset, nucleus):     {ent.h_joint:.4f} bits")
    print(f"  H(nucleus | onset):    {ent.h_nucleus_given_onset:.4f} bits")
    print(f"  H(onset | nucleus):    {ent.h_onset_given_nucleus:.4f} bits")
    print(f"  MI(onset; nucleus):    {ent.mi_onset_nucleus:.4f} bits")
    print(f"    95% CI: [{result.mi_ci_lower:.4f}, {result.mi_ci_upper:.4f}]")
    print(f"    Null MI: {result.null_mi_mean:.4f} +/- {result.null_mi_std:.4f}")
    print(f"    z-score: {result.mi_z_score:.1f}")
    print(f"\n  Reduction R = 1 - H(nuc|ons)/H(nuc): {ent.reduction_r:.4f}")
    print(f"    95% CI: [{result.r_ci_lower:.4f}, {result.r_ci_upper:.4f}]")
    print(f"    Abugida threshold (>0.20):  {'PASS' if ent.reduction_r > 0.20 else 'FAIL'}")
    print(f"\n  Reverse R = 1 - H(ons|nuc)/H(ons):   {ent.reverse_r:.4f}")
    asymmetry = abs(ent.reduction_r - ent.reverse_r)
    print(f"  Asymmetry |R - R_rev|:                {asymmetry:.4f}")
    if ent.reduction_r > ent.reverse_r + 0.1:
        print("    -> Onset predicts nucleus more than reverse (abugida pattern)")
    elif ent.reverse_r > ent.reduction_r + 0.1:
        print("    -> Nucleus predicts onset more than reverse (unusual)")
    else:
        print("    -> Roughly symmetric mutual constraint")

    if result.positional_curves:
        print("\n  Positional entropy (onset vs nucleus):")
        print(f"  {'Pos':>4s} {'H(onset)':>10s} {'H(nucleus)':>12s} {'Ratio':>7s} {'N':>7s}")
        for pc in result.positional_curves:
            ratio = pc.h_onset / pc.h_nucleus if pc.h_nucleus > 0 else 0
            print(f"  {pc.position:>4d} {pc.h_onset:>10.4f} {pc.h_nucleus:>12.4f} "
                  f"{ratio:>7.2f} {pc.n_observations:>7d}")

    print("\n--- Phase 3C: Script Type Classification ---")
    print(f"  {'Type':<15s} {'R in range':>10s} {'Distance':>10s} {'Match':<10s}")
    print("  " + "-" * 47)
    for sc in result.script_comparisons:
        print(f"  {sc.script_type:<15s} {'YES' if sc.r_in_range else 'NO':>10s} "
              f"{sc.r_distance:>10.4f} {sc.overall_match.upper():<10s}")

    print(f"\n  Best match: {result.best_match_script.upper()}")
    print(f"\n  VERDICT: {result.verdict.upper()}")

    if result.verdict == 'abugida':
        print("  -> The script shows abugida-like structure.")
        print("  -> Map onset strokes to consonants and nucleus strokes to vowels.")
    elif result.verdict == 'syllabary':
        print("  -> The script is a pure syllabary.")
        print("  -> Continue with whole-cell syllable mapping.")
    elif result.verdict == 'alphabet':
        print("  -> The script is alphabetic.")
        print("  -> The grid may be imposing artificial syllabic structure.")
    else:
        print("  -> Evidence is inconclusive between script types.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_abugida_test() -> Dict:
    """Run the abugida hypothesis test and save results."""
    print("=" * 70)
    print("PHASE 4 STEP 3: ABUGIDA HYPOTHESIS TEST")
    print("=" * 70)

    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)

    # Phase 3A: Decompose
    print("\nDecomposing glyphs into onset/nucleus strokes...")
    pairs = decompose_tokens_onset_nucleus(tokens)
    by_pos = decompose_by_position(tokens)

    # Phase 3B: Compute entropy
    print("Computing conditional entropy...")
    entropy = compute_onset_nucleus_entropy(pairs)
    pos_curves = compute_positional_entropy(by_pos)

    # Bootstrap CIs
    print("Bootstrapping CIs (500 iterations)...")
    mi_ci, r_ci = bootstrap_onset_nucleus_ci(tokens, n_bootstrap=500)

    # Null baseline
    print("Computing null baseline MI (100 trials)...")
    null_mean, null_std = null_baseline_mi(pairs, n_trials=100)
    z = (entropy.mi_onset_nucleus - null_mean) / null_std if null_std > 0 else 0.0

    # Phase 3C: Compare to script types
    refs = build_script_type_references()
    comparisons = compare_to_script_types(entropy, refs)

    # Find best match
    matches = [c for c in comparisons if c.overall_match == 'match']
    if matches:
        best = min(matches, key=lambda c: c.r_distance).script_type
    else:
        partials = [c for c in comparisons if c.overall_match == 'partial']
        if partials:
            best = min(partials, key=lambda c: c.r_distance).script_type
        else:
            best = min(comparisons, key=lambda c: c.r_distance).script_type

    # Verdict
    r = entropy.reduction_r
    if r > 0.20 and entropy.reduction_r > entropy.reverse_r:
        verdict = 'abugida'
    elif 0.15 <= r <= 0.55 and abs(entropy.reduction_r - entropy.reverse_r) < 0.1:
        verdict = 'syllabary'
    elif r < 0.15:
        verdict = 'alphabet'
    else:
        verdict = 'inconclusive'

    result = AbugidaTestResult(
        entropy=entropy,
        positional_curves=pos_curves,
        script_comparisons=comparisons,
        best_match_script=best,
        mi_ci_lower=round(mi_ci[0], 4),
        mi_ci_upper=round(mi_ci[1], 4),
        r_ci_lower=round(r_ci[0], 4),
        r_ci_upper=round(r_ci[1], 4),
        null_mi_mean=round(null_mean, 4),
        null_mi_std=round(null_std, 4),
        mi_z_score=round(z, 1),
        verdict=verdict,
    )

    _print_results(result)

    # Save
    rd = _results_dir()
    out_data = {
        'entropy': asdict(entropy),
        'positional_curves': [asdict(pc) for pc in pos_curves],
        'script_comparisons': [asdict(sc) for sc in comparisons],
        'best_match_script': best,
        'mi_ci': [result.mi_ci_lower, result.mi_ci_upper],
        'r_ci': [result.r_ci_lower, result.r_ci_upper],
        'null_mi_mean': result.null_mi_mean,
        'null_mi_std': result.null_mi_std,
        'mi_z_score': result.mi_z_score,
        'verdict': verdict,
    }
    out_path = os.path.join(rd, 'abugida_test.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return out_data
