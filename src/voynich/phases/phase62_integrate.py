"""
Phase 62: Exhaustive Computational Analysis — Integration
==========================================================
Combines all 11 investigation results into three tier verdicts:
  Q1 (Token Identity): Inv 1-4 → WORDS / SYLLABLES / MIXED
  Q2 (Classification Quality): Inv 5-8 → GOOD / MINOR / MAJOR
  Q3 (Corpus-Level Structure): Inv 9-11 → insights on A/B, hands, entropy

Dependency chain:
    results/phase62_t1_reverse.json       (Inv 1)
    results/phase62_cross_token.json      (Inv 2)
    results/phase62_gallows_initial.json  (Inv 3)
    results/phase62_decoded_bigram.json   (Inv 4)
    results/phase62_orphaned_coda.json    (Inv 5)
    results/phase62_double_modifier.json  (Inv 6)
    results/phase62_token_length.json     (Inv 7)
    results/phase62_syllable_entropy.json (Inv 8)
    results/phase62_lang_ab_cvc.json      (Inv 9)
    results/phase62_hand_cvc.json         (Inv 10)
    results/phase62_multi_entropy.json    (Inv 11)
        -> results/phase62_integrate.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


from voynich.core._paths import results_dir as _results_dir


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
# Experiment files
# ---------------------------------------------------------------------------

EXPERIMENT_FILES = [
    (1, 'phase62_t1_reverse.json', 'T1 Reverse Engineering'),
    (2, 'phase62_cross_token.json', 'Cross-Token Word Reconstruction'),
    (3, 'phase62_gallows_initial.json', 'Gallows as Word-Initial Markers'),
    (4, 'phase62_decoded_bigram.json', 'Decoded Bigram Frequency vs Latin'),
    (5, 'phase62_orphaned_coda.json', 'Orphaned Coda Investigation'),
    (6, 'phase62_double_modifier.json', 'Double-Modifier Sequences'),
    (7, 'phase62_token_length.json', 'Token Length Distribution'),
    (8, 'phase62_syllable_entropy.json', 'Syllable-Level Entropy'),
    (9, 'phase62_lang_ab_cvc.json', 'Language A/B Under CVC'),
    (10, 'phase62_hand_cvc.json', 'Hand-by-Hand CVC'),
    (11, 'phase62_multi_entropy.json', 'Multi-Level Entropy'),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TierVerdict:
    tier: str
    question: str
    investigations: List[int]
    n_run: int = 0
    n_passed: int = 0
    verdicts: List[Dict] = field(default_factory=list)
    answer: str = ""


@dataclass
class Phase62Result:
    phase: str = "62"
    experiment: str = "phase62_integrate"
    n_investigations_run: int = 0
    n_investigations_passed: int = 0
    tier1: Dict = field(default_factory=dict)
    tier2: Dict = field(default_factory=dict)
    tier3: Dict = field(default_factory=dict)
    overall_verdict: str = ""
    summary: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Tier logic
# ---------------------------------------------------------------------------

def _evaluate_tier1(results: Dict[int, Dict]) -> TierVerdict:
    """Q1: Are EVA tokens words or syllables?"""
    tier = TierVerdict(
        tier='Tier 1',
        question='Token Identity: words or syllables?',
        investigations=[1, 2, 3, 4],
    )

    word_votes = 0
    syllable_votes = 0

    for inv_num in tier.investigations:
        r = results.get(inv_num)
        if not r:
            continue
        tier.n_run += 1
        passed = r.get('gate_passed', False)
        verdict = r.get('verdict', '')
        tier.verdicts.append({'inv': inv_num, 'passed': passed, 'verdict': verdict})
        if passed:
            tier.n_passed += 1

        # Vote logic
        if inv_num == 1:
            if r.get('tokens_are_words'):
                word_votes += 1
            elif r.get('tokens_are_syllables'):
                syllable_votes += 1
        elif inv_num == 2:
            if r.get('cross_boundary_fraction', 0) > 0.20:
                syllable_votes += 1
            else:
                word_votes += 1
        elif inv_num == 3:
            if r.get('best_concat_ratio', 0) > 1.5:
                word_votes += 1  # gallows mark word starts → tokens are syllables
                syllable_votes += 1
        elif inv_num == 4:
            if r.get('within_word_fraction', 0) > 0.40:
                syllable_votes += 1
            else:
                word_votes += 1

    if syllable_votes > word_votes:
        tier.answer = 'TOKENS_ARE_SYLLABLES'
    elif word_votes > syllable_votes:
        tier.answer = 'TOKENS_ARE_WORDS'
    else:
        tier.answer = 'MIXED'

    return tier


def _evaluate_tier2(results: Dict[int, Dict]) -> TierVerdict:
    """Q2: Is the coda classification correct?"""
    tier = TierVerdict(
        tier='Tier 2',
        question='Classification quality: coda and modifier correctness',
        investigations=[5, 6, 7, 8],
    )

    for inv_num in tier.investigations:
        r = results.get(inv_num)
        if not r:
            continue
        tier.n_run += 1
        passed = r.get('gate_passed', False)
        verdict = r.get('verdict', '')
        tier.verdicts.append({'inv': inv_num, 'passed': passed, 'verdict': verdict})
        if passed:
            tier.n_passed += 1

    if tier.n_passed >= 3:
        tier.answer = 'CLASSIFICATION_GOOD'
    elif tier.n_passed >= 2:
        tier.answer = 'MINOR_CORRECTIONS_NEEDED'
    else:
        tier.answer = 'MAJOR_RECLASSIFICATION_NEEDED'

    return tier


def _evaluate_tier3(results: Dict[int, Dict]) -> TierVerdict:
    """Q3: What additional structure exists?"""
    tier = TierVerdict(
        tier='Tier 3',
        question='Corpus-level structure: A/B split, hands, entropy',
        investigations=[9, 10, 11],
    )

    findings = []
    for inv_num in tier.investigations:
        r = results.get(inv_num)
        if not r:
            continue
        tier.n_run += 1
        passed = r.get('gate_passed', False)
        verdict = r.get('verdict', '')
        tier.verdicts.append({'inv': inv_num, 'passed': passed, 'verdict': verdict})
        if passed:
            tier.n_passed += 1
            findings.append(verdict)

    if tier.n_passed >= 2:
        tier.answer = 'SIGNIFICANT_STRUCTURE'
    elif tier.n_passed == 1:
        tier.answer = 'PARTIAL_STRUCTURE'
    else:
        tier.answer = 'NO_ADDITIONAL_STRUCTURE'

    return tier


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------

def run_phase62_verdict():
    """Phase 62 verdict: integrate all 11 investigation results."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62: Integration — Exhaustive Pre-Visual Analysis")
    print("=" * 70)

    # Load all results
    results = {}
    for inv_num, filename, title in EXPERIMENT_FILES:
        r = _safe_load(os.path.join(rd, filename))
        if r:
            results[inv_num] = r
            status = 'PASS' if r.get('gate_passed') else 'FAIL'
            verdict = r.get('verdict', '?')
            gp = r.get('gates_passed', '?')
            print(f"  Inv {inv_num:2d}: {title:40s} [{status}] {gp} gates  {verdict}")
        else:
            print(f"  Inv {inv_num:2d}: {title:40s} [NOT RUN]")

    n_run = len(results)
    n_passed = sum(1 for r in results.values() if r.get('gate_passed'))

    # Evaluate tiers
    tier1 = _evaluate_tier1(results)
    tier2 = _evaluate_tier2(results)
    tier3 = _evaluate_tier3(results)

    # Overall verdict
    answers = [tier1.answer, tier2.answer, tier3.answer]
    tier_passes = sum(1 for t in [tier1, tier2, tier3] if t.n_passed > t.n_run // 2)

    if tier_passes == 3:
        overall = 'ALL_TIERS_PASS'
    elif tier_passes >= 1:
        overall = 'PARTIAL'
    else:
        overall = 'NO_TIERS_PASS'

    summary = (f"Q1={tier1.answer} | Q2={tier2.answer} | Q3={tier3.answer} | "
               f"{n_run}/11 run, {n_passed}/11 passed")

    result = Phase62Result(
        n_investigations_run=n_run,
        n_investigations_passed=n_passed,
        tier1=_convert(asdict(tier1)),
        tier2=_convert(asdict(tier2)),
        tier3=_convert(asdict(tier3)),
        overall_verdict=overall,
        summary=summary,
        runtime_seconds=time.time() - t0,
    )

    print(f"\n  Tier 1 ({tier1.question}):")
    print(f"    {tier1.n_passed}/{tier1.n_run} passed → {tier1.answer}")
    print(f"  Tier 2 ({tier2.question}):")
    print(f"    {tier2.n_passed}/{tier2.n_run} passed → {tier2.answer}")
    print(f"  Tier 3 ({tier3.question}):")
    print(f"    {tier3.n_passed}/{tier3.n_run} passed → {tier3.answer}")
    print(f"\n  Overall: {overall}")
    print(f"  Summary: {summary}")

    path = _save_json(rd, 'phase62_integrate.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result


def run_phase62():
    """Run all Phase 62 investigations + integration."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 62: Full Pipeline — Exhaustive Pre-Visual Analysis")
    print("=" * 70)

    from voynich.phases.p62_t1_reverse import run_t1_reverse
    from voynich.phases.p62_cross_token import run_cross_token
    from voynich.phases.p62_gallows_initial import run_gallows_initial
    from voynich.phases.p62_decoded_bigram import run_decoded_bigram
    from voynich.phases.p62_orphaned_coda import run_orphaned_coda
    from voynich.phases.p62_double_modifier import run_double_modifier
    from voynich.phases.p62_token_length import run_token_length
    from voynich.phases.p62_syllable_entropy import run_syllable_entropy
    from voynich.phases.p62_lang_ab_cvc import run_lang_ab_cvc
    from voynich.phases.p62_hand_cvc import run_hand_cvc
    from voynich.phases.p62_multi_entropy import run_multi_entropy

    runners = [
        ('Inv 1', run_t1_reverse),
        ('Inv 2', run_cross_token),
        ('Inv 3', run_gallows_initial),
        ('Inv 4', run_decoded_bigram),
        ('Inv 5', run_orphaned_coda),
        ('Inv 6', run_double_modifier),
        ('Inv 7', run_token_length),
        ('Inv 8', run_syllable_entropy),
        ('Inv 9', run_lang_ab_cvc),
        ('Inv 10', run_hand_cvc),
        ('Inv 11', run_multi_entropy),
    ]

    for label, runner in runners:
        print(f"\n{'─' * 70}")
        print(f"  Running {label}...")
        print(f"{'─' * 70}")
        try:
            runner()
        except Exception as e:
            print(f"  ERROR in {label}: {e}")

    print(f"\n{'─' * 70}")
    print(f"  Running Integration...")
    print(f"{'─' * 70}")
    result = run_phase62_verdict()

    total_time = time.time() - t0
    print(f"\n  Phase 62 total runtime: {total_time:.1f}s ({total_time / 60:.1f} min)")
    return result
