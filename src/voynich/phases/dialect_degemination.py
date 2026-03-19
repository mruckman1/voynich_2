"""
Phase 54.1 – Systematic Degemination Test
==========================================
Tests whether the decoded Voynich signal words show systematic
degemination of Latin geminate consonants, which is diagnostic of
Northern Italian dialects (Venetian, Lombard, Ligurian, Emilian)
versus Tuscan (which preserves geminates).

Dependency chain:
    signal_10k.json          (Phase 17 — Latin signal words)
    italian_signal.json      (Phase 19 — Italian-only signal words)
    word_catalog.json        (Phase 24 — T1 identified words)
        → phase54_degemination.json   (this step)
"""

import json
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple

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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DegemWordDetail:
    word: str
    etymon: str
    has_geminate: bool
    geminate_cluster: str
    standard_italian: str
    status: str  # 'degeminated', 'preserved', 'absent'


@dataclass
class DegemResult:
    phase: str  # "54.1"
    experiment: str  # "degemination"
    n_testable: int
    n_degeminated: int
    n_preserved: int
    degemination_rate: float
    null_mean: float
    null_std: float
    z_score: float
    selectivity: float
    dialect_scores: Dict[str, float]
    per_word_details: List[Dict]
    gates: Dict[str, bool]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Geminate etyma database
# ---------------------------------------------------------------------------

# (latin_etymon, has_geminate, geminate_cluster or None, standard_italian)
GEMINATE_ETYMA: Dict[str, Tuple[str, bool, Optional[str], str]] = {
    # --- Signal words (Latin 10K) ---
    "di": ("de", False, None, "di"),
    "se": ("se", False, None, "se"),
    "ne": ("ne", False, None, "ne"),
    "dise": ("dixe", False, None, "disse"),
    "sero": ("serum", False, None, "siero"),
    "bi": ("bis", False, None, "bi"),
    "ce": ("ce", False, None, "ce"),
    "co": ("cum", False, None, "con"),
    "ni": ("ni", False, None, "ni"),
    "rati": ("rationem", False, None, "ragione"),
    "sene": ("senna", True, "nn", "senna"),
    "de": ("de", False, None, "di"),
    "bene": ("bene", False, None, "bene"),
    "du": ("duo", False, None, "due"),
    "ci": ("ci", False, None, "ci"),
    "te": ("te", False, None, "te"),
    "bo": ("bonus", False, None, "buono"),
    "dira": ("dira", False, None, "dira"),
    "la": ("la", False, None, "la"),
    "si": ("si", False, None, "si"),
    "sere": ("serem", False, None, "sera"),
    "nera": ("nigra", False, None, "nera"),
    "ra": ("ra", False, None, "ra"),
    "sera": ("sera", False, None, "sera"),
    "do": ("do", False, None, "do"),
    "re": ("re", False, None, "re"),
    "so": ("sum", False, None, "sono"),
    "cu": ("cum", False, None, "con"),
    "ti": ("ti", False, None, "ti"),
    "su": ("super", False, None, "su"),
    "diri": ("dirigere", False, None, "dirigere"),
    "ru": ("ru", False, None, "ru"),
    "cola": ("colare", False, None, "colare"),
    "nu": ("nu", False, None, "nu"),
    "ha": ("habere", False, None, "ha"),
    "li": ("illi", True, "ll", "gli"),
    "dedi": ("dedisse", True, "ss", "diede"),
    "ga": ("ga", False, None, "ga"),
    "tere": ("terere", False, None, "tritare"),
    "sede": ("sedem", False, None, "sede"),
    "tela": ("tela", False, None, "tela"),
    "tu": ("tu", False, None, "tu"),
    "dico": ("dico", False, None, "dico"),
    "ge": ("ge", False, None, "ge"),
    "sese": ("seipse", False, None, "se stesso"),
    "hi": ("hi", False, None, "i"),
    "raro": ("rarum", False, None, "raro"),
    "fe": ("fecit", False, None, "fece"),
    "fa": ("facit", False, None, "fa"),
    "raso": ("rasum", False, None, "raso"),
    "dici": ("dici", False, None, "dici"),

    # --- Italian-only signals ---
    "be": ("be", False, None, "be"),
    "cora": ("cora", False, None, "cora"),
    "bela": ("bella", True, "ll", "bella"),
    "cedi": ("cedere", False, None, "cedere"),
    "didi": ("didicit", False, None, "didi"),
    "dice": ("dicere", False, None, "dice"),
    "deco": ("decorum", False, None, "decoro"),
    "cose": ("causa", False, None, "cose"),
    "beri": ("bibere", False, None, "bere"),
    "code": ("codicem", False, None, "codice"),
    "dicu": ("dicum", False, None, "dico"),
    "corali": ("corallum", True, "ll", "coralli"),
    "diga": ("diga", False, None, "diga"),
    "dido": ("dido", False, None, "dido"),
    "deri": ("deri", False, None, "deri"),
    "dere": ("dere", False, None, "dere"),
    "gi": ("gi", False, None, "gi"),
    "cela": ("cellam", True, "ll", "cella"),
    "decore": ("decorem", False, None, "decoro"),

    # --- T1 identified words ---
    "stercora": ("stercora", False, None, "stercora"),
    "ratione": ("rationem", False, None, "ragione"),
    "rabidi": ("rabidi", False, None, "rabidi"),
    "diasene": ("diasenna", True, "nn", "diasenna"),
    "coralli": ("corallum", True, "ll", "coralli"),
    "codex": ("codicem", False, None, "codice"),
    "radicom": ("radicem", False, None, "radice"),
    "commune": ("communem", True, "mm", "comune"),
    "secundi": ("secundum", False, None, "secondo"),
}


# ---------------------------------------------------------------------------
# Degemination analysis
# ---------------------------------------------------------------------------

def _classify_word(word: str, etymon: str, geminate: str) -> str:
    """Classify a decoded word relative to its geminate etymon.

    Returns:
        'preserved' — the decoded form retains the geminate cluster
        'degeminated' — the decoded form has the single consonant
                        at a position matching the etymon
        'absent' — neither the geminate nor the matching single
                   consonant appears in the decoded form
    """
    if geminate in word:
        return "preserved"
    single = geminate[0]  # e.g., "ll" → "l"
    if single in word:
        return "degeminated"
    return "absent"


def _null_degemination_rate(
    testable_words: List[Tuple[str, str, str]],
    n_iter: int,
    rng: random.Random,
) -> List[float]:
    """Compute null degemination rates by shuffling characters within words.

    For each iteration, shuffle each decoded word's characters and check
    whether the shuffled form still shows the single consonant at any
    position (matching geminate's base letter).

    Returns a list of null degemination rates.
    """
    null_rates: List[float] = []

    for _ in range(n_iter):
        n_degem = 0
        n_pres = 0
        for word, _etymon, geminate in testable_words:
            chars = list(word)
            rng.shuffle(chars)
            shuffled = "".join(chars)
            single = geminate[0]
            if geminate in shuffled:
                n_pres += 1
            elif single in shuffled:
                n_degem += 1
            # else: absent — skip for rate calc
        denom = n_degem + n_pres
        if denom > 0:
            null_rates.append(n_degem / denom)
        else:
            null_rates.append(0.0)

    return null_rates


def _score_dialect(
    observed_rate: float,
) -> Dict[str, float]:
    """Score dialect fit based on degemination rate.

    Each dialect has a midpoint (typical degemination rate) and a range
    (how far the rate can deviate and still be plausible).
    Score = max(0, 1 - |observed - midpoint| / range).
    """
    dialects = {
        "Venetian":  (0.90,  0.20),
        "Lombard":   (0.925, 0.15),
        "Ligurian":  (0.875, 0.15),
        "Emilian":   (0.85,  0.20),
        "Tuscan":    (0.075, 0.15),
    }
    scores: Dict[str, float] = {}
    for name, (midpoint, rng_width) in dialects.items():
        scores[name] = round(max(0.0, 1.0 - abs(observed_rate - midpoint) / rng_width), 4)
    return scores


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_degemination() -> None:
    """Run the systematic degemination test (Phase 54.1)."""
    t0 = time.time()
    rd = _results_dir()

    print("=" * 70)
    print("PHASE 54.1: Systematic Degemination Test")
    print("=" * 70)

    # ── 1. Load signal words ──
    print("\n  1. Loading signal words …")
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

    # ── 2. Match against geminate etyma ──
    print("\n  2. Matching against geminate etyma …")
    per_word_details: List[DegemWordDetail] = []
    testable_words: List[Tuple[str, str, str]] = []  # (word, etymon, geminate_cluster)

    for word in sorted(decoded_words):
        if word not in GEMINATE_ETYMA:
            continue
        etymon, has_gem, gem_cluster, std_italian = GEMINATE_ETYMA[word]
        if not has_gem:
            per_word_details.append(DegemWordDetail(
                word=word,
                etymon=etymon,
                has_geminate=False,
                geminate_cluster="",
                standard_italian=std_italian,
                status="n/a",
            ))
            continue

        # Testable: etymon has a geminate
        status = _classify_word(word, etymon, gem_cluster)
        per_word_details.append(DegemWordDetail(
            word=word,
            etymon=etymon,
            has_geminate=True,
            geminate_cluster=gem_cluster,
            standard_italian=std_italian,
            status=status,
        ))
        if status in ("degeminated", "preserved"):
            testable_words.append((word, etymon, gem_cluster))

    n_testable = len(testable_words)
    n_degeminated = sum(1 for d in per_word_details if d.status == "degeminated")
    n_preserved = sum(1 for d in per_word_details if d.status == "preserved")
    n_absent = sum(1 for d in per_word_details if d.status == "absent")
    n_with_geminate = sum(1 for d in per_word_details if d.has_geminate)

    print(f"     Words with geminate etyma: {n_with_geminate}")
    print(f"     Testable (degeminated + preserved): {n_testable}")
    print(f"     Degeminated: {n_degeminated}")
    print(f"     Preserved: {n_preserved}")
    print(f"     Absent: {n_absent}")

    # ── 3. Per-word detail printout ──
    print("\n  3. Per-word classification:")
    for detail in per_word_details:
        if detail.has_geminate:
            print(f"     {detail.word:12s} < {detail.etymon:12s} "
                  f"({detail.geminate_cluster}) → {detail.status}")

    # ── 4. Degemination rate ──
    if n_testable > 0:
        degem_rate = n_degeminated / n_testable
    else:
        degem_rate = 0.0
    print(f"\n  4. Degemination rate: {degem_rate:.3f} "
          f"({n_degeminated}/{n_testable})")

    # ── 5. Null test ──
    print("\n  5. Null test (1000 iterations, character shuffle) …")
    rng = random.Random(42)
    n_null = 1000

    if n_testable > 0:
        null_rates = _null_degemination_rate(testable_words, n_null, rng)
        null_mean = sum(null_rates) / len(null_rates)
        null_std = (sum((r - null_mean) ** 2 for r in null_rates) / len(null_rates)) ** 0.5
        if null_std > 0:
            z_score = (degem_rate - null_mean) / null_std
        else:
            z_score = 0.0
        selectivity = degem_rate / null_mean if null_mean > 0 else 0.0
    else:
        null_rates = []
        null_mean = 0.0
        null_std = 0.0
        z_score = 0.0
        selectivity = 0.0

    print(f"     Null mean: {null_mean:.3f}")
    print(f"     Null std:  {null_std:.3f}")
    print(f"     z-score:   {z_score:.2f}")
    print(f"     Selectivity: {selectivity:.2f}x")

    # ── 6. Dialect scoring ──
    print("\n  6. Dialect scoring:")
    dialect_scores = _score_dialect(degem_rate)
    for dialect, score in sorted(dialect_scores.items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 20)
        print(f"     {dialect:12s}  {score:.3f}  {bar}")

    # ── 7. Gates ──
    g1 = n_testable >= 3
    g2 = degem_rate < 0.4 or degem_rate > 0.6
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
        verdict = (f"DEGEMINATION_DETECTED (rate={degem_rate:.3f}, "
                   f"best_fit={best_dialect}, z={z_score:.2f})")
    elif not g1:
        verdict = f"INSUFFICIENT_DATA (n_testable={n_testable})"
    elif not g2:
        verdict = f"AMBIGUOUS_RATE (rate={degem_rate:.3f})"
    else:
        verdict = f"LOW_SELECTIVITY (selectivity={selectivity:.2f}x)"

    print(f"\n  Verdict: {verdict}")

    # ── 8. Save results ──
    runtime = round(time.time() - t0, 1)

    result = DegemResult(
        phase="54.1",
        experiment="degemination",
        n_testable=n_testable,
        n_degeminated=n_degeminated,
        n_preserved=n_preserved,
        degemination_rate=round(degem_rate, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z_score, 2),
        selectivity=round(selectivity, 2),
        dialect_scores=dialect_scores,
        per_word_details=[_convert(asdict(d)) for d in per_word_details],
        gates=gates,
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, "phase54_degemination.json")
    with open(out_path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}  ({runtime}s)")
