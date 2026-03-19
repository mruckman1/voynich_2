"""
Phase 54.2 – Lenition Pattern Test
====================================
Tests whether decoded Voynich signal words show Gallo-Italic lenition
(voicing of Latin intervocalic voiceless stops: P→b, T→d, K→g) or
Tuscan preservation.  The lenition rate discriminates Northern Italian
dialects (high lenition) from Tuscan (near-zero lenition).

Dependency chain:
    signal_10k.json          (Phase 17 — Latin signal words)
    italian_signal.json      (Phase 19 — Italian-only signal words)
    word_catalog.json        (Phase 24 — T1 identified words)
        → phase54_lenition.json   (this step)
"""

import json
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Tuple

from voynich.core._paths import results_dir as _results_dir


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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class LenitionResult:
    phase: str  # "54.2"
    experiment: str  # "lenition_pattern"
    n_testable: int
    n_lenited: int
    n_preserved: int
    n_spirantized: int
    n_other: int
    lenition_rate: float
    spirantization_rate: float
    preservation_rate: float
    per_word_details: List[Dict]  # word, latin, stop, outcome
    dialect_scores: Dict[str, float]
    null_mean: float
    null_std: float
    z_score: float
    selectivity: float
    gates: Dict[str, bool]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Lenition etyma database
# ---------------------------------------------------------------------------

# Each entry describes the intervocalic-stop situation for a decoded word.
# Fields:
#   latin       — Latin etymon
#   stop        — the original Latin stop consonant (or None if no stop)
#   latin_pos   — index of the stop in the Latin word
#   decoded_pos — index of the corresponding consonant in the decoded word
#   decoded_consonant — what consonant actually appears
#   outcome     — 'lenited', 'preserved', 'spirantized'
#   intervocalic — whether the stop is truly intervocalic (default True)
#   already_voiced — whether the Latin stop was already voiced (default False)
#   cluster    — whether the stop is part of a cluster (default False)

LENITION_ETYMA: Dict[str, Dict[str, Any]] = {
    'diga': {
        'latin': 'dicat',
        'stop': 'c',  # Latin /k/
        'latin_pos': 2,
        'decoded_pos': 2,
        'decoded_consonant': 'g',
        'outcome': 'lenited',  # k→g
    },
    'dise': {
        'latin': 'dicit',
        'stop': 'c',
        'latin_pos': 2,
        'decoded_pos': 2,
        'decoded_consonant': 's',
        'outcome': 'spirantized',  # k→s (via palatalization)
    },
    'dice': {
        'latin': 'dicit',
        'stop': 'c',
        'latin_pos': 2,
        'decoded_pos': 2,
        'decoded_consonant': 'c',
        'outcome': 'preserved',  # k→c (Tuscan-like)
    },
    'dico': {
        'latin': 'dico',
        'stop': 'c',
        'latin_pos': 2,
        'decoded_pos': 2,
        'decoded_consonant': 'c',
        'outcome': 'preserved',  # but NOT intervocalic in 1sg (word-final)
        'intervocalic': False,  # Not intervocalic in this form
    },
    'codi': {
        'latin': 'cocti',  # from coquere/coctio
        'stop': 'c',
        'latin_pos': 2,
        'decoded_pos': 2,
        'decoded_consonant': 'd',
        'outcome': 'lenited',  # ct→d (cluster simplification + voicing)
    },
    'sede': {
        'latin': 'sedem',
        'stop': 'd',
        'latin_pos': 2,
        'decoded_pos': 2,
        'decoded_consonant': 'd',
        'outcome': 'preserved',  # d→d (already voiced, no change expected)
        'already_voiced': True,
    },
    'cose': {
        'latin': 'causas',
        'stop': 's',
        'latin_pos': 3,  # Not a stop - skip
        'intervocalic': False,
    },
    'cora': {
        'latin': 'cor',
        'stop': None,  # no intervocalic stop
        'intervocalic': False,
    },
    'ratione': {
        'latin': 'rationem',
        'stop': 't',
        'latin_pos': 2,
        'decoded_pos': 2,  # 'ratione' - 't' at position 2
        'decoded_consonant': 't',
        'outcome': 'preserved',  # but this is in a -tion- cluster
        'cluster': True,
    },
    'secundi': {
        'latin': 'secundum',
        'stop': 'c',
        'latin_pos': 2,
        'decoded_pos': 2,
        'decoded_consonant': 'c',
        'outcome': 'preserved',
        'intervocalic': True,
    },
}


# ---------------------------------------------------------------------------
# Consonant classification sets
# ---------------------------------------------------------------------------

VOICED_STOPS = {'b', 'd', 'g', 'v'}
VOICELESS_STOPS = {'p', 't', 'k', 'c', 'q'}
SPIRANTS = {'s', 'z', 'x', 'f', 'h'}

ALL_CONSONANTS = 'bcdfghjklmnpqrstvwxyz'


# ---------------------------------------------------------------------------
# Dialect lenition profiles
# ---------------------------------------------------------------------------

# Each profile gives (lo, hi) ranges for lenition, spirantization,
# and preservation rates.  A dialect scores highest when the observed
# rates fall inside these ranges.

DIALECT_LENITION_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    'venetian':  {'lenition': (0.3, 0.6), 'spirant': (0.2, 0.5), 'preservation': (0.1, 0.3)},
    'lombard':   {'lenition': (0.5, 0.8), 'spirant': (0.1, 0.3), 'preservation': (0.0, 0.2)},
    'ligurian':  {'lenition': (0.4, 0.7), 'spirant': (0.1, 0.4), 'preservation': (0.1, 0.2)},
    'emilian':   {'lenition': (0.5, 0.8), 'spirant': (0.1, 0.2), 'preservation': (0.0, 0.2)},
    'tuscan':    {'lenition': (0.0, 0.1), 'spirant': (0.0, 0.1), 'preservation': (0.8, 1.0)},
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_dialect(
    lenition_rate: float,
    spirant_rate: float,
    preservation_rate: float,
    profile: Dict[str, Tuple[float, float]],
) -> float:
    """Score a single dialect profile against observed rates.

    For each rate dimension, the score is a triangle function centered
    on the profile midpoint with half-width equal to the profile range.
    The final score is the mean of the three dimension scores.
    """
    scores = []
    for rate, key in [
        (lenition_rate, 'lenition'),
        (spirant_rate, 'spirant'),
        (preservation_rate, 'preservation'),
    ]:
        lo, hi = profile[key]
        mid = (lo + hi) / 2
        rng = (hi - lo) / 2
        if rng > 0:
            s = max(0.0, 1.0 - abs(rate - mid) / rng)
        else:
            s = 1.0 if abs(rate - mid) < 0.1 else 0.0
        scores.append(s)
    return sum(scores) / len(scores)


def _classify_consonant(c: str) -> str:
    """Classify a consonant as lenited, preserved, spirantized, or other."""
    if c in VOICED_STOPS:
        return 'lenited'
    if c in VOICELESS_STOPS:
        return 'preserved'
    if c in SPIRANTS:
        return 'spirantized'
    return 'other'


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_lenition() -> None:
    """Run the lenition pattern test (Phase 54.2)."""
    t0 = time.time()
    rd = _results_dir()

    print("=" * 70)
    print("PHASE 54.2: Lenition Pattern Test")
    print("=" * 70)

    # ── 1. Load signal words ──
    print("\n  1. Loading signal words ...")
    decoded_words: set = set()

    # signal_10k.json → genuine signal words
    sig10k_path = os.path.join(rd, "signal_10k.json")
    if os.path.exists(sig10k_path):
        with open(sig10k_path) as f:
            sig10k = json.load(f)
        for ws in sig10k.get("word_signals", []):
            if ws.get("is_genuine_signal", False):
                decoded_words.add(ws["word"])
        print(f"     signal_10k.json: {len([w for w in sig10k.get('word_signals', []) if w.get('is_genuine_signal')])} genuine signals")
    else:
        print(f"     signal_10k.json: NOT FOUND")

    # italian_signal.json → italian-only signal words
    ital_path = os.path.join(rd, "italian_signal.json")
    if os.path.exists(ital_path):
        with open(ital_path) as f:
            ital = json.load(f)
        for ws in ital.get("italian_only_signals", []):
            decoded_words.add(ws["word"])
        print(f"     italian_signal.json: {len(ital.get('italian_only_signals', []))} Italian-only signals")
    else:
        print(f"     italian_signal.json: NOT FOUND")

    # word_catalog.json → T1 words
    cat_path = os.path.join(rd, "word_catalog.json")
    if os.path.exists(cat_path):
        with open(cat_path) as f:
            cat = json.load(f)
        t1_words = set()
        for entry in cat.get("single_token_ids", []):
            if entry.get("tier") == "T1":
                t1_words.add(entry["latin_word"])
        decoded_words.update(t1_words)
        print(f"     word_catalog.json: {len(t1_words)} unique T1 words")
    else:
        print(f"     word_catalog.json: NOT FOUND")

    print(f"     Total unique decoded words: {len(decoded_words)}")

    # ── 2. Filter to testable entries ──
    print("\n  2. Filtering lenition etyma to testable entries ...")

    testable_entries: List[Dict[str, Any]] = []
    all_entries: List[Dict[str, Any]] = []

    for word, info in sorted(LENITION_ETYMA.items()):
        entry = {'word': word}
        entry.update(info)

        # Skip if word not in our decoded signal set
        if word not in decoded_words:
            entry['skip_reason'] = 'not_in_signals'
            all_entries.append(entry)
            continue

        # Filter criteria
        intervocalic = info.get('intervocalic', True)
        already_voiced = info.get('already_voiced', False)
        cluster = info.get('cluster', False)
        stop = info.get('stop')

        if not intervocalic:
            entry['skip_reason'] = 'not_intervocalic'
            all_entries.append(entry)
            continue
        if already_voiced:
            entry['skip_reason'] = 'already_voiced'
            all_entries.append(entry)
            continue
        if cluster:
            entry['skip_reason'] = 'cluster'
            all_entries.append(entry)
            continue
        if stop is None:
            entry['skip_reason'] = 'no_stop'
            all_entries.append(entry)
            continue

        entry['skip_reason'] = None
        all_entries.append(entry)
        testable_entries.append(entry)

    n_testable = len(testable_entries)
    print(f"     Total etyma: {len(LENITION_ETYMA)}")
    print(f"     In signal set: {sum(1 for e in all_entries if e.get('skip_reason') != 'not_in_signals')}")
    print(f"     Testable (intervocalic, not already voiced, not cluster): {n_testable}")

    # ── 3. Classify each testable entry ──
    print("\n  3. Classifying intervocalic stop outcomes ...")
    per_word_details: List[Dict] = []

    for entry in testable_entries:
        decoded_consonant = entry.get('decoded_consonant', '')
        outcome = _classify_consonant(decoded_consonant)
        detail = {
            'word': entry['word'],
            'latin': entry['latin'],
            'stop': entry['stop'],
            'decoded_consonant': decoded_consonant,
            'outcome': outcome,
        }
        per_word_details.append(detail)
        print(f"     {entry['word']:12s} < {entry['latin']:12s}  "
              f"({entry['stop']} -> {decoded_consonant}) = {outcome}")

    # ── 4. Compute rates ──
    n_lenited = sum(1 for d in per_word_details if d['outcome'] == 'lenited')
    n_preserved = sum(1 for d in per_word_details if d['outcome'] == 'preserved')
    n_spirantized = sum(1 for d in per_word_details if d['outcome'] == 'spirantized')
    n_other = n_testable - n_lenited - n_preserved - n_spirantized

    lenition_rate = n_lenited / n_testable if n_testable > 0 else 0.0
    spirant_rate = n_spirantized / n_testable if n_testable > 0 else 0.0
    preservation_rate = n_preserved / n_testable if n_testable > 0 else 0.0

    print(f"\n  4. Rates:")
    print(f"     Lenition:       {lenition_rate:.3f} ({n_lenited}/{n_testable})")
    print(f"     Spirantization: {spirant_rate:.3f} ({n_spirantized}/{n_testable})")
    print(f"     Preservation:   {preservation_rate:.3f} ({n_preserved}/{n_testable})")
    print(f"     Other:          {n_other}/{n_testable}")

    # ── 5. Dialect scoring ──
    print("\n  5. Dialect scoring:")
    dialect_scores: Dict[str, float] = {}
    for dname, profile in DIALECT_LENITION_PROFILES.items():
        score = _score_dialect(lenition_rate, spirant_rate, preservation_rate, profile)
        dialect_scores[dname] = round(score, 4)

    for dialect, score in sorted(dialect_scores.items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 20)
        print(f"     {dialect:12s}  {score:.3f}  {bar}")

    # ── 6. Null test (1000 iterations, random consonant replacement) ──
    print("\n  6. Null test (1000 iterations, random consonant replacement) ...")
    rng = random.Random(42)
    n_null = 1000
    null_best_scores: List[float] = []

    if n_testable > 0:
        for _ in range(n_null):
            # For each testable word, replace decoded consonant with random
            null_n_lenited = 0
            null_n_preserved = 0
            null_n_spirantized = 0
            null_n_other = 0

            for _entry in testable_entries:
                rand_c = rng.choice(ALL_CONSONANTS)
                outcome = _classify_consonant(rand_c)
                if outcome == 'lenited':
                    null_n_lenited += 1
                elif outcome == 'preserved':
                    null_n_preserved += 1
                elif outcome == 'spirantized':
                    null_n_spirantized += 1
                else:
                    null_n_other += 1

            null_len_rate = null_n_lenited / n_testable
            null_spi_rate = null_n_spirantized / n_testable
            null_pres_rate = null_n_preserved / n_testable

            # Score each dialect and take the best
            best_null_score = 0.0
            for _dname, profile in DIALECT_LENITION_PROFILES.items():
                s = _score_dialect(null_len_rate, null_spi_rate, null_pres_rate, profile)
                if s > best_null_score:
                    best_null_score = s
            null_best_scores.append(best_null_score)

        null_mean = sum(null_best_scores) / len(null_best_scores)
        null_std = (sum((s - null_mean) ** 2 for s in null_best_scores) / len(null_best_scores)) ** 0.5
    else:
        null_mean = 0.0
        null_std = 0.0

    # Real best score
    real_best = max(dialect_scores.values()) if dialect_scores else 0.0

    if null_std > 0:
        z_score = (real_best - null_mean) / null_std
    else:
        z_score = 0.0
    selectivity = real_best / null_mean if null_mean > 0 else 0.0

    print(f"     Null mean: {null_mean:.3f}")
    print(f"     Null std:  {null_std:.3f}")
    print(f"     Real best: {real_best:.3f}")
    print(f"     z-score:   {z_score:.2f}")
    print(f"     Selectivity: {selectivity:.2f}x")

    # ── 7. Gates ──
    g1 = n_testable >= 3
    g2 = (lenition_rate + spirant_rate) > 0.3 or preservation_rate > 0.7
    g3 = selectivity >= 1.5

    gates = {
        "G1_n_testable_ge_3": g1,
        "G2_rate_not_ambiguous": g2,
        "G3_selectivity_ge_1_5": g3,
    }

    print("\n  7. Gates:")
    for gate_name, passed in gates.items():
        print(f"     {gate_name}: {'PASS' if passed else 'FAIL'}")

    all_gates = g1 and g2 and g3
    if all_gates:
        best_dialect = max(dialect_scores, key=dialect_scores.get)
        verdict = (f"LENITION_DETECTED (lenition={lenition_rate:.3f}, "
                   f"spirant={spirant_rate:.3f}, "
                   f"best_fit={best_dialect}, z={z_score:.2f})")
    elif not g1:
        verdict = f"INSUFFICIENT_DATA (n_testable={n_testable})"
    elif not g2:
        verdict = (f"AMBIGUOUS_RATES (lenition={lenition_rate:.3f}, "
                   f"spirant={spirant_rate:.3f}, "
                   f"preservation={preservation_rate:.3f})")
    else:
        verdict = f"LOW_SELECTIVITY (selectivity={selectivity:.2f}x)"

    print(f"\n  Verdict: {verdict}")

    # ── 8. Save results ──
    runtime = round(time.time() - t0, 1)

    result = LenitionResult(
        phase="54.2",
        experiment="lenition_pattern",
        n_testable=n_testable,
        n_lenited=n_lenited,
        n_preserved=n_preserved,
        n_spirantized=n_spirantized,
        n_other=n_other,
        lenition_rate=round(lenition_rate, 4),
        spirantization_rate=round(spirant_rate, 4),
        preservation_rate=round(preservation_rate, 4),
        per_word_details=per_word_details,
        dialect_scores=dialect_scores,
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z_score, 2),
        selectivity=round(selectivity, 2),
        gates=gates,
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, "phase54_lenition.json")
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}  ({runtime}s)")
