"""
Phase 19.4 – Modifier Character Distributional Validation
==========================================================
Phase 16 identified 15 modifier characters via 5 convergent methods.
This test validates those modifiers against 6 independent distributional
predictions that true modifier systems (Devanagari virama, Arabic shadda)
should satisfy.

Dependency chain:
    modifier_integrate.json  (Phase 16 classification)
    corpus
        → modifier_validation.json
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as sp_stats

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars


# ---------------------------------------------------------------------------
# JSON serialiser (project convention)
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    name: str
    description: str
    passed: bool
    metric_real: float
    metric_null: float
    p_value: float
    detail: str


@dataclass
class ModifierValidationResult:
    n_modifiers_tested: int
    n_syllabic_tested: int
    modifier_chars: List[str]
    syllabic_chars: List[str]
    # Per-prediction results
    predictions: List[Dict[str, Any]]
    n_confirmed: int
    n_neutral: int
    n_violated: int
    # Null comparison
    null_mean_confirmed: float
    null_std_confirmed: float
    null_n_trials: int
    real_vs_null_sigma: float
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _char_bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    """Extract all consecutive EVA-char pairs within tokens."""
    bigrams = []
    for tok in tokens:
        chars = tokenize_eva_chars(tok)
        for i in range(len(chars) - 1):
            bigrams.append((chars[i], chars[i + 1]))
    return bigrams


def _entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy of a frequency dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values() if c > 0
    )


def _char_mi(bigrams: List[Tuple[str, str]], char_set: Set[str],
             partner_set: Set[str]) -> float:
    """
    Compute mean mutual information between chars in char_set and their
    adjacent partners in partner_set.
    """
    # Count joint and marginal frequencies
    joint = Counter()
    marginal_a = Counter()
    marginal_b = Counter()
    total = 0
    for a, b in bigrams:
        if a in char_set and b in partner_set:
            joint[(a, b)] += 1
            marginal_a[a] += 1
            marginal_b[b] += 1
            total += 1
        if b in char_set and a in partner_set:
            joint[(b, a)] += 1
            marginal_a[b] += 1
            marginal_b[a] += 1
            total += 1

    if total == 0:
        return 0.0

    mi = 0.0
    for (a, b), count in joint.items():
        p_ab = count / total
        p_a = marginal_a[a] / total
        p_b = marginal_b[b] / total
        if p_a > 0 and p_b > 0 and p_ab > 0:
            mi += p_ab * math.log2(p_ab / (p_a * p_b))
    return mi


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

def _test_adjacency_asymmetry(
    bigrams: List[Tuple[str, str]],
    modifiers: Set[str],
    syllabics: Set[str],
) -> PredictionResult:
    """
    P1: Modifiers should have HIGHER MI with their adjacent syllabic
    neighbours than syllabic-syllabic pairs (more selective adjacency).
    """
    mi_mod = _char_mi(bigrams, modifiers, syllabics)
    mi_syl = _char_mi(bigrams, syllabics, syllabics)

    passed = mi_mod > mi_syl
    return PredictionResult(
        name="P1_adjacency_asymmetry",
        description="MI(modifier,syllabic) > MI(syllabic,syllabic)",
        passed=passed,
        metric_real=round(mi_mod, 4),
        metric_null=round(mi_syl, 4),
        p_value=-1.0,  # Comparative, not statistical
        detail=f"MI_mod={mi_mod:.4f}, MI_syl={mi_syl:.4f}, ratio={mi_mod / mi_syl:.2f}" if mi_syl > 0 else "MI_syl=0",
    )


def _test_no_modifier_pairs(
    bigrams: List[Tuple[str, str]],
    modifiers: Set[str],
    all_chars: Set[str],
) -> PredictionResult:
    """
    P2: Modifier-modifier bigrams should be significantly rarer than
    expected by chance (like virama-virama in Devanagari).
    """
    mod_mod_count = sum(1 for a, b in bigrams if a in modifiers and b in modifiers)
    total = len(bigrams)

    # Expected: P(mod) * P(mod) * total
    mod_unigram = sum(1 for a, _ in bigrams if a in modifiers)
    p_mod = mod_unigram / total if total > 0 else 0
    expected = p_mod * p_mod * total

    ratio = mod_mod_count / expected if expected > 0 else 0.0
    passed = ratio < 0.5  # Significantly less than expected

    return PredictionResult(
        name="P2_no_modifier_pairs",
        description="Modifier-modifier bigrams << expected",
        passed=passed,
        metric_real=round(ratio, 4),
        metric_null=1.0,  # Expected ratio under independence
        p_value=-1.0,
        detail=f"observed={mod_mod_count}, expected={expected:.1f}, obs/exp={ratio:.3f}",
    )


def _test_position_clustering(
    tokens: List[str],
    modifiers: Set[str],
) -> PredictionResult:
    """
    P3: Modifiers should cluster at specific positions within tokens
    (non-uniform positional distribution). Chi-squared test for
    non-uniformity.
    """
    position_counts: Dict[str, Counter] = defaultdict(Counter)

    for tok in tokens:
        chars = tokenize_eva_chars(tok)
        n = len(chars)
        if n < 2:
            continue
        for i, ch in enumerate(chars):
            if ch in modifiers:
                # Normalize position to 0-2: initial, medial, final
                if i == 0:
                    pos = 'initial'
                elif i == n - 1:
                    pos = 'final'
                else:
                    pos = 'medial'
                position_counts[ch][pos] += 1

    # Aggregate across all modifiers
    agg = Counter()
    for ch_counts in position_counts.values():
        agg.update(ch_counts)

    positions = ['initial', 'medial', 'final']
    observed = np.array([agg.get(p, 0) for p in positions], dtype=float)
    total = observed.sum()
    if total < 10:
        return PredictionResult(
            name="P3_position_clustering",
            description="Modifiers cluster at specific token positions",
            passed=False,
            metric_real=0.0,
            metric_null=0.0,
            p_value=1.0,
            detail="Insufficient data",
        )

    expected = np.full(3, total / 3.0)
    chi2, p_val = sp_stats.chisquare(observed, expected)

    passed = p_val < 0.01  # Highly non-uniform

    return PredictionResult(
        name="P3_position_clustering",
        description="Modifiers cluster at specific token positions",
        passed=passed,
        metric_real=round(float(chi2), 2),
        metric_null=0.0,
        p_value=round(float(p_val), 6),
        detail=f"initial={int(observed[0])}, medial={int(observed[1])}, final={int(observed[2])}, chi2={chi2:.2f}, p={p_val:.2e}",
    )


def _test_length_effect(
    tokens: List[str],
    modifiers: Set[str],
) -> PredictionResult:
    """
    P4: Tokens containing modifiers should be longer (in EVA chars) than
    those without, by approximately 1 char per modifier.
    """
    with_mod = []
    without_mod = []

    for tok in tokens:
        chars = tokenize_eva_chars(tok)
        n = len(chars)
        has_mod = any(ch in modifiers for ch in chars)
        if has_mod:
            with_mod.append(n)
        else:
            without_mod.append(n)

    if len(with_mod) < 10 or len(without_mod) < 10:
        return PredictionResult(
            name="P4_length_effect",
            description="Modifier tokens are longer than non-modifier tokens",
            passed=False,
            metric_real=0.0,
            metric_null=0.0,
            p_value=1.0,
            detail="Insufficient data",
        )

    mean_with = float(np.mean(with_mod))
    mean_without = float(np.mean(without_mod))
    diff = mean_with - mean_without

    ks_stat, ks_p = sp_stats.ks_2samp(with_mod, without_mod)

    # Modifier tokens should be longer
    passed = diff > 0.5 and ks_p < 0.01

    return PredictionResult(
        name="P4_length_effect",
        description="Modifier tokens are longer than non-modifier tokens",
        passed=passed,
        metric_real=round(diff, 3),
        metric_null=0.0,
        p_value=round(float(ks_p), 6),
        detail=f"mean_with={mean_with:.2f}, mean_without={mean_without:.2f}, diff={diff:.2f}, KS_p={ks_p:.2e}",
    )


def _test_bigram_preservation(
    tokens: List[str],
    modifiers: Set[str],
    all_chars: Set[str],
    rng: random.Random,
) -> PredictionResult:
    """
    P5: Removing modifiers should preserve (or improve) bigram entropy
    better than removing a random set of characters.
    """
    # Full text bigram entropy
    full_text = ' '.join(tokens)
    h2_full = _bigram_entropy(tokens)

    # Strip modifiers
    stripped_tokens = []
    for tok in tokens:
        chars = tokenize_eva_chars(tok)
        kept = [ch for ch in chars if ch not in modifiers]
        if kept:
            stripped_tokens.append(''.join(kept))
    h2_stripped = _bigram_entropy(stripped_tokens) if stripped_tokens else 0.0

    # Strip random characters (same count as modifiers)
    n_mod = len(modifiers)
    non_mod_chars = list(all_chars - modifiers)
    if len(non_mod_chars) >= n_mod:
        random_set = set(rng.sample(non_mod_chars, n_mod))
    else:
        random_set = set(non_mod_chars)
    random_stripped = []
    for tok in tokens:
        chars = tokenize_eva_chars(tok)
        kept = [ch for ch in chars if ch not in random_set]
        if kept:
            random_stripped.append(''.join(kept))
    h2_random = _bigram_entropy(random_stripped) if random_stripped else 0.0

    # Modifier removal should decrease H2 less than random removal
    delta_mod = abs(h2_full - h2_stripped)
    delta_rand = abs(h2_full - h2_random)

    passed = delta_mod < delta_rand

    return PredictionResult(
        name="P5_bigram_preservation",
        description="Modifier removal preserves bigram structure better than random removal",
        passed=passed,
        metric_real=round(delta_mod, 4),
        metric_null=round(delta_rand, 4),
        p_value=-1.0,
        detail=f"H2_full={h2_full:.4f}, H2_mod_stripped={h2_stripped:.4f}, H2_rand_stripped={h2_random:.4f}, delta_mod={delta_mod:.4f}, delta_rand={delta_rand:.4f}",
    )


def _bigram_entropy(tokens: List[str]) -> float:
    """Compute character-level bigram conditional entropy."""
    text = ' '.join(tokens)
    if len(text) < 2:
        return 0.0

    bigram_counts: Dict[str, Counter] = defaultdict(Counter)
    for i in range(len(text) - 1):
        bigram_counts[text[i]][text[i + 1]] += 1

    total_entropy = 0.0
    total_contexts = 0
    for context, followers in bigram_counts.items():
        context_total = sum(followers.values())
        h = -sum(
            (c / context_total) * math.log2(c / context_total)
            for c in followers.values() if c > 0
        )
        total_entropy += h * context_total
        total_contexts += context_total

    return total_entropy / total_contexts if total_contexts > 0 else 0.0


def _test_section_independence(
    corpus,
    modifiers: Set[str],
    syllabics: Set[str],
) -> PredictionResult:
    """
    P6: Modifier frequencies should be more consistent across manuscript
    sections than syllabic character frequencies (modifiers are part of
    the script, not the vocabulary).
    """
    sections = ['herbal_a', 'herbal_b', 'pharmaceutical', 'biological',
                'astronomical', 'cosmological', 'recipes']

    mod_freqs_by_section: Dict[str, Dict[str, float]] = {}
    syl_freqs_by_section: Dict[str, Dict[str, float]] = {}

    for section in sections:
        pages = corpus.get_pages_by_section(section)
        if not pages:
            continue
        section_tokens = []
        for page in pages:
            section_tokens.extend(page.all_tokens)

        if len(section_tokens) < 50:
            continue

        # Count EVA chars
        char_counts = Counter()
        for tok in section_tokens:
            for ch in tokenize_eva_chars(tok):
                char_counts[ch] += 1

        total = sum(char_counts.values())
        if total == 0:
            continue

        mod_freqs_by_section[section] = {
            ch: char_counts.get(ch, 0) / total for ch in modifiers
        }
        syl_freqs_by_section[section] = {
            ch: char_counts.get(ch, 0) / total for ch in syllabics
        }

    if len(mod_freqs_by_section) < 3:
        return PredictionResult(
            name="P6_section_independence",
            description="Modifier frequencies more consistent across sections than syllabic",
            passed=False,
            metric_real=0.0,
            metric_null=0.0,
            p_value=1.0,
            detail="Insufficient sections with data",
        )

    # Compute CV of each character's frequency across sections
    mod_cvs = []
    for ch in modifiers:
        freqs = [sf.get(ch, 0.0) for sf in mod_freqs_by_section.values()]
        mean = np.mean(freqs)
        if mean > 0:
            mod_cvs.append(float(np.std(freqs) / mean))

    syl_cvs = []
    for ch in syllabics:
        freqs = [sf.get(ch, 0.0) for sf in syl_freqs_by_section.values()]
        mean = np.mean(freqs)
        if mean > 0:
            syl_cvs.append(float(np.std(freqs) / mean))

    mean_mod_cv = float(np.mean(mod_cvs)) if mod_cvs else 999.0
    mean_syl_cv = float(np.mean(syl_cvs)) if syl_cvs else 999.0

    # Modifiers should have LOWER CV (more consistent)
    passed = mean_mod_cv < mean_syl_cv

    return PredictionResult(
        name="P6_section_independence",
        description="Modifier frequencies more consistent across sections than syllabic",
        passed=passed,
        metric_real=round(mean_mod_cv, 4),
        metric_null=round(mean_syl_cv, 4),
        p_value=-1.0,
        detail=f"mean_CV_modifiers={mean_mod_cv:.4f}, mean_CV_syllabic={mean_syl_cv:.4f}",
    )


# ---------------------------------------------------------------------------
# Null baseline
# ---------------------------------------------------------------------------

def _run_null_trial(
    tokens: List[str],
    all_chars_list: List[str],
    n_select: int,
    corpus,
    rng: random.Random,
) -> int:
    """Run one null trial: pick n_select random chars, test 6 predictions."""
    random_mods = set(rng.sample(all_chars_list, min(n_select, len(all_chars_list))))
    random_syls = set(all_chars_list) - random_mods

    bigrams = _char_bigrams(tokens)

    results = [
        _test_adjacency_asymmetry(bigrams, random_mods, random_syls),
        _test_no_modifier_pairs(bigrams, random_mods, set(all_chars_list)),
        _test_position_clustering(tokens, random_mods),
        _test_length_effect(tokens, random_mods),
        _test_bigram_preservation(tokens, random_mods, set(all_chars_list), rng),
        _test_section_independence(corpus, random_mods, random_syls),
    ]
    return sum(1 for r in results if r.passed)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_modifier_validation() -> None:
    """Phase 19.4: Modifier character distributional validation."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 19.4: Modifier Character Distributional Validation")
    print("=" * 60)

    # ── 1. Load dependencies ──────────────────────────────────────────
    print("\n  1. Loading modifier classifications and corpus …")

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    mod_data = _load_json(mod_path)
    if mod_data is None:
        print(f"  [WARN] Missing {mod_path} — cannot run validation")
        result = ModifierValidationResult(
            n_modifiers_tested=0, n_syllabic_tested=0,
            modifier_chars=[], syllabic_chars=[],
            predictions=[], n_confirmed=0, n_neutral=0, n_violated=0,
            null_mean_confirmed=0, null_std_confirmed=0, null_n_trials=0,
            real_vs_null_sigma=0,
            gate_passed=False, verdict="SKIP: missing modifier_integrate.json",
            runtime_seconds=round(time.time() - t0, 2),
        )
        out = os.path.join(rd, 'modifier_validation.json')
        with open(out, 'w') as f:
            json.dump(_convert(result), f, indent=2)
        print(f"\n    → {out}")
        return

    modifier_chars = set(mod_data['modifier_chars'])
    syllabic_chars = set(mod_data['syllabic_chars'])

    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    print(f"    {len(modifier_chars)} modifiers, {len(syllabic_chars)} syllabic, {len(tokens)} tokens")

    # Build full EVA char inventory
    all_chars_set: Set[str] = set()
    for tok in tokens:
        for ch in tokenize_eva_chars(tok):
            all_chars_set.add(ch)
    all_chars_list = sorted(all_chars_set)

    # ── 2. Run 6 prediction tests ────────────────────────────────────
    print("\n  2. Running 6 distributional predictions …")

    bigrams = _char_bigrams(tokens)
    rng = random.Random(42)

    p1 = _test_adjacency_asymmetry(bigrams, modifier_chars, syllabic_chars)
    print(f"    P1 adjacency asymmetry: {'PASS' if p1.passed else 'FAIL'} — {p1.detail}")

    p2 = _test_no_modifier_pairs(bigrams, modifier_chars, all_chars_set)
    print(f"    P2 no modifier pairs:   {'PASS' if p2.passed else 'FAIL'} — {p2.detail}")

    p3 = _test_position_clustering(tokens, modifier_chars)
    print(f"    P3 position clustering:  {'PASS' if p3.passed else 'FAIL'} — {p3.detail}")

    p4 = _test_length_effect(tokens, modifier_chars)
    print(f"    P4 length effect:        {'PASS' if p4.passed else 'FAIL'} — {p4.detail}")

    p5 = _test_bigram_preservation(tokens, modifier_chars, all_chars_set, rng)
    print(f"    P5 bigram preservation:  {'PASS' if p5.passed else 'FAIL'} — {p5.detail}")

    p6 = _test_section_independence(corpus, modifier_chars, syllabic_chars)
    print(f"    P6 section independence: {'PASS' if p6.passed else 'FAIL'} — {p6.detail}")

    predictions = [p1, p2, p3, p4, p5, p6]
    n_confirmed = sum(1 for p in predictions if p.passed)
    n_violated = 6 - n_confirmed
    n_neutral = 0  # All tests produce binary pass/fail

    print(f"\n    Confirmed: {n_confirmed}/6")

    # ── 3. Null baseline ─────────────────────────────────────────────
    print("\n  3. Running null baseline (100 trials with random 'modifiers') …")

    n_trials = 100
    null_confirmed = []
    for trial in range(n_trials):
        trial_rng = random.Random(1000 + trial)
        n_pass = _run_null_trial(tokens, all_chars_list, len(modifier_chars),
                                 corpus, trial_rng)
        null_confirmed.append(n_pass)

    null_mean = float(np.mean(null_confirmed))
    null_std = float(np.std(null_confirmed))
    sigma = (n_confirmed - null_mean) / null_std if null_std > 0 else 0.0

    print(f"    Null mean: {null_mean:.2f} ± {null_std:.2f}")
    print(f"    Real: {n_confirmed}, sigma: {sigma:.2f}")

    # ── 4. Gate ──────────────────────────────────────────────────────
    # Gate: real modifiers pass more predictions than null mean + 2σ
    gate_passed = bool(n_confirmed > null_mean + 2 * null_std and n_confirmed >= 4)

    if gate_passed:
        verdict = f"VALIDATED: {n_confirmed}/6 predictions confirmed ({sigma:.1f}σ above null)"
    elif n_confirmed >= 3:
        verdict = f"PARTIAL: {n_confirmed}/6 predictions confirmed ({sigma:.1f}σ above null)"
    else:
        verdict = f"REFUTED: only {n_confirmed}/6 predictions confirmed ({sigma:.1f}σ above null)"

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 5. Save ──────────────────────────────────────────────────────
    result = ModifierValidationResult(
        n_modifiers_tested=len(modifier_chars),
        n_syllabic_tested=len(syllabic_chars),
        modifier_chars=sorted(modifier_chars),
        syllabic_chars=sorted(syllabic_chars),
        predictions=[_convert(asdict(p)) for p in predictions],
        n_confirmed=n_confirmed,
        n_neutral=n_neutral,
        n_violated=n_violated,
        null_mean_confirmed=round(null_mean, 3),
        null_std_confirmed=round(null_std, 3),
        null_n_trials=n_trials,
        real_vs_null_sigma=round(sigma, 3),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'modifier_validation.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n    → {out_path}")
