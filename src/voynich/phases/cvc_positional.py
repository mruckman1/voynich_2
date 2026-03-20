"""
Phase 59, Investigation 6: Positional Distribution of Coda Markers
===================================================================
If modifier characters are coda consonants, they should appear word-finally
or word-medially — not word-initially.  This module tests that prediction
by computing positional profiles for each modifier stroke group.

Dependency chain:
    results/modifier_integrate.json   (Phase 16)
    EVA_VISUAL_COMPONENTS             (reference.py)
        -> results/cvc_positional.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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
# Stroke group definitions (from coda_markers.py)
# ---------------------------------------------------------------------------

CODA_GROUPS = {
    'hook':       ['aiin', 'aiiin', 'iin', 'iiin', 'n'],
    'descender':  ['dy', 'ey', 'y'],
    'sigmoid':    ['ar', 'or'],
    'vertical':   ['al', 'ol', 'am', 'i', 'm', 'g'],
    'connector':  ['b', 'h', 'ckh', 'u'],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GroupPositionStats:
    """Positional statistics for one modifier stroke group."""
    group: str
    initial: int = 0
    medial: int = 0
    final: int = 0
    solo: int = 0
    total: int = 0
    initial_frac: float = 0.0
    medial_frac: float = 0.0
    final_frac: float = 0.0
    solo_frac: float = 0.0
    mean_relative_position: float = 0.0
    compatible: Optional[bool] = None
    reason: str = ''


@dataclass
class NullPositionalComparison:
    """Bootstrap comparison of modifier vs non-modifier positions."""
    modifier_mean_position: float = 0.0
    nonmodifier_mean_position: float = 0.0
    difference: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    significant: bool = False


@dataclass
class CvcPositionalResult:
    """Full Investigation 6 output."""
    phase: str = "59"
    investigation: str = "6"
    experiment: str = "cvc_positional"
    # Per-group stats
    group_stats: List[GroupPositionStats] = field(default_factory=list)
    overall_stats: Optional[GroupPositionStats] = None
    # Null comparison
    null_comparison: Optional[NullPositionalComparison] = None
    # Gates
    g1_initial_frac: bool = False      # < 0.15
    g2_mean_position: bool = False     # > 0.55
    g3_groups_compatible: bool = False  # ≥ 4/5 compatible
    g4_significant_diff: bool = False   # mod vs non-mod significant
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _build_modifier_set(mod_data: Dict) -> Set[str]:
    """Build set of all modifier + ambiguous EVA characters."""
    modifiers: Set[str] = set()
    for cls in mod_data.get('classifications', []):
        if cls['final_classification'] in ('modifier', 'ambiguous'):
            modifiers.add(cls['eva_char'])
    return modifiers


def compute_coda_positions(
    corpus_tokens: List[str],
    modifier_set: Set[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[float]]]:
    """Compute positional profiles for each modifier group.

    Returns (position_data, relative_positions).
    """
    # Build flat modifier -> group lookup
    all_modifiers: Set[str] = set()
    modifier_to_group: Dict[str, str] = {}
    for group, chars in CODA_GROUPS.items():
        for c in chars:
            if c in modifier_set:
                all_modifiers.add(c)
                modifier_to_group[c] = group

    # Initialize counters
    groups = list(CODA_GROUPS.keys()) + ['ALL']
    position_data: Dict[str, Dict[str, int]] = {
        g: {'initial': 0, 'medial': 0, 'final': 0, 'solo': 0, 'total': 0}
        for g in groups
    }
    relative_positions: Dict[str, List[float]] = {g: [] for g in groups}
    # Also track non-modifier positions for comparison
    nonmod_positions: List[float] = []

    for token in corpus_tokens:
        chars = tokenize_eva_chars(token)
        n = len(chars)
        if n == 0:
            continue

        for idx, char in enumerate(chars):
            # Relative position
            rel_pos = idx / (n - 1) if n > 1 else 0.5

            if char in all_modifiers:
                group = modifier_to_group[char]
                position_data[group]['total'] += 1
                position_data['ALL']['total'] += 1

                if n == 1:
                    pos_class = 'solo'
                elif idx == 0:
                    pos_class = 'initial'
                elif idx == n - 1:
                    pos_class = 'final'
                else:
                    pos_class = 'medial'

                position_data[group][pos_class] += 1
                position_data['ALL'][pos_class] += 1
                relative_positions[group].append(rel_pos)
                relative_positions['ALL'].append(rel_pos)
            else:
                nonmod_positions.append(rel_pos)

    return position_data, relative_positions, nonmod_positions


def score_coda_compatibility(
    position_data: Dict[str, Dict[str, int]],
    relative_positions: Dict[str, List[float]],
) -> List[GroupPositionStats]:
    """Score each group on coda compatibility."""
    results: List[GroupPositionStats] = []

    for group in list(CODA_GROUPS.keys()) + ['ALL']:
        data = position_data[group]
        total = data['total']

        if total == 0:
            results.append(GroupPositionStats(
                group=group, compatible=None, reason='no_data'))
            continue

        initial_frac = data['initial'] / total
        medial_frac = data['medial'] / total
        final_frac = data['final'] / total
        solo_frac = data['solo'] / total
        mean_pos = float(np.mean(relative_positions[group])) if relative_positions[group] else 0.5

        # Coda compatibility scoring
        if initial_frac > 0.20:
            compatible = False
            reason = f'too_many_initial ({initial_frac:.1%})'
        elif solo_frac > 0.50:
            compatible = False
            reason = f'too_many_solo ({solo_frac:.1%})'
        elif (final_frac + medial_frac) > 0.70:
            compatible = True
            reason = f'final+medial={final_frac + medial_frac:.1%}'
        else:
            compatible = None  # MARGINAL
            reason = 'mixed distribution'

        results.append(GroupPositionStats(
            group=group,
            initial=data['initial'],
            medial=data['medial'],
            final=data['final'],
            solo=data['solo'],
            total=total,
            initial_frac=round(initial_frac, 4),
            medial_frac=round(medial_frac, 4),
            final_frac=round(final_frac, 4),
            solo_frac=round(solo_frac, 4),
            mean_relative_position=round(mean_pos, 4),
            compatible=compatible,
            reason=reason,
        ))

    return results


def null_positional_comparison(
    modifier_positions: List[float],
    nonmod_positions: List[float],
    n_boot: int = 1000,
) -> NullPositionalComparison:
    """Compare modifier vs non-modifier mean positions with bootstrap CI."""
    if not modifier_positions or not nonmod_positions:
        return NullPositionalComparison()

    mod_arr = np.array(modifier_positions)
    non_arr = np.array(nonmod_positions)
    mod_mean = float(np.mean(mod_arr))
    non_mean = float(np.mean(non_arr))

    rng = np.random.RandomState(42)
    boot_diffs: List[float] = []
    for _ in range(n_boot):
        sample_mod = rng.choice(mod_arr, size=len(mod_arr), replace=True)
        sample_non = rng.choice(non_arr, size=min(len(non_arr), 50000), replace=True)
        boot_diffs.append(float(np.mean(sample_mod) - np.mean(sample_non)))

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return NullPositionalComparison(
        modifier_mean_position=round(mod_mean, 4),
        nonmodifier_mean_position=round(non_mean, 4),
        difference=round(mod_mean - non_mean, 4),
        ci_95_lower=round(ci_lower, 4),
        ci_95_upper=round(ci_upper, 4),
        significant=ci_lower > 0,  # modifier mean significantly higher
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_positional():
    """Investigation 6: Positional distribution of coda markers."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 6: Coda Marker Positional Distribution")
    print("=" * 70)

    rd = str(_results_dir())

    # Load modifier classification
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_set = _build_modifier_set(mod_data)
    print(f"\n  Modifier characters: {len(modifier_set)}")

    # Load corpus
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    print(f"  Corpus tokens: {len(all_tokens)}")

    # Compute positions
    print("\n  Computing positional profiles ...")
    position_data, relative_positions, nonmod_positions = compute_coda_positions(
        all_tokens, modifier_set)

    # Score compatibility
    group_stats = score_coda_compatibility(position_data, relative_positions)

    # Find overall stats
    overall = next((g for g in group_stats if g.group == 'ALL'), None)
    per_group = [g for g in group_stats if g.group != 'ALL']

    # Print per-group table
    print(f"\n  {'Group':<12} {'Initial':>8} {'Medial':>8} {'Final':>8} "
          f"{'Solo':>8} {'Total':>8} {'MeanPos':>8} {'Compat'}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for g in group_stats:
        compat_str = ('YES' if g.compatible is True
                      else 'NO' if g.compatible is False
                      else 'MARGINAL' if g.compatible is None and g.total > 0
                      else '-')
        print(f"  {g.group:<12} {g.initial_frac:>7.1%} {g.medial_frac:>7.1%} "
              f"{g.final_frac:>7.1%} {g.solo_frac:>7.1%} {g.total:>8d} "
              f"{g.mean_relative_position:>8.3f} {compat_str}")

    # Null comparison
    print("\n  Running bootstrap comparison (modifier vs non-modifier) ...")
    null_comp = null_positional_comparison(
        relative_positions.get('ALL', []), nonmod_positions)

    print(f"  Modifier mean position:     {null_comp.modifier_mean_position:.4f}")
    print(f"  Non-modifier mean position: {null_comp.nonmodifier_mean_position:.4f}")
    print(f"  Difference:                 {null_comp.difference:.4f}")
    print(f"  95% CI:                     [{null_comp.ci_95_lower:.4f}, "
          f"{null_comp.ci_95_upper:.4f}]")
    print(f"  Significant:                {null_comp.significant}")

    # Gates
    n_compatible = sum(1 for g in per_group if g.compatible is True)

    g1 = overall.initial_frac < 0.15 if overall else False
    g2 = overall.mean_relative_position > 0.55 if overall else False
    g3 = n_compatible >= 4
    g4 = null_comp.significant

    gates_passed = sum([g1, g2, g3, g4])

    print(f"\n  Validation Gates:")
    print(f"    G1 initial_frac < 15%:     {'PASS' if g1 else 'FAIL'} "
          f"({overall.initial_frac:.1%})" if overall else "    G1: no data")
    print(f"    G2 mean_position > 0.55:   {'PASS' if g2 else 'FAIL'} "
          f"({overall.mean_relative_position:.3f})" if overall else "    G2: no data")
    print(f"    G3 ≥4/5 groups compatible: {'PASS' if g3 else 'FAIL'} "
          f"({n_compatible}/5)")
    print(f"    G4 mod vs non-mod signif:  {'PASS' if g4 else 'FAIL'}")
    print(f"    Gates passed: {gates_passed}/4")

    result = CvcPositionalResult(
        group_stats=per_group,
        overall_stats=overall,
        null_comparison=null_comp,
        g1_initial_frac=g1,
        g2_mean_position=g2,
        g3_groups_compatible=g3,
        g4_significant_diff=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_positional.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 6 completed in {time.time() - t0:.1f}s")
