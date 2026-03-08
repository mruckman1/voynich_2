"""
Phase 24.1 -- Leave-One-Out Triple Sensitivity Analysis
========================================================
For each of the 25 feature-triples in the Phase 16 assignment, removes
that triple's mapping and re-decodes the corpus to measure the impact
on dict-hit rate and bigram plausibility.  Triples whose removal causes
a large drop are ``probably_correct``; triples whose removal causes no
change or an improvement are ``probably_wrong``.

Dependency chain:
    combined_refine.json   (Phase 15 best_assignment)
    modifier_integrate.json (Phase 16 modifier chars + rules)
        -> triple_sensitivity.json (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TripleSensitivity:
    triple_key: str
    eva_chars: List[str]
    assigned_syllable: str
    tokens_affected: int
    dict_hit_delta: float
    bigram_delta: float
    sensitivity_rank: int
    classification: str  # "probably_correct" | "uncertain" | "probably_wrong"
    anchor_backed: bool


@dataclass
class SensitivityResult:
    timestamp: str
    baseline_dict_hit: float
    baseline_bigram_hits: int
    baseline_n_tokens: int
    n_triples: int
    sensitivities: List[Dict]
    n_probably_correct: int
    n_uncertain: int
    n_probably_wrong: int
    probably_correct_triples: List[str]
    uncertain_triples: List[str]
    probably_wrong_triples: List[str]
    anchor_overrides: List[str]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Bigram helpers (from readability_delta.py)
# ---------------------------------------------------------------------------

def _build_ref_bigrams(ref_words: List[str]) -> set:
    return {(ref_words[i].lower(), ref_words[i + 1].lower())
            for i in range(len(ref_words) - 1)}


def _count_bigram_hits(decoded_words: List[str], ref_bigrams: set) -> int:
    """Count absolute number of bigram hits."""
    if len(decoded_words) < 2:
        return 0
    return sum(1 for i in range(len(decoded_words) - 1)
               if (decoded_words[i].lower(), decoded_words[i + 1].lower())
               in ref_bigrams)


# ---------------------------------------------------------------------------
# R3 combined decode
# ---------------------------------------------------------------------------

def _decode_r3_combined(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 combined strategy (alter -> strip -> original)."""
    decoded = []
    for token in tokens:
        # Try alteration
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt)
            continue

        # Try stripping
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped)
            continue

        # Fall back to original decoding
        original = decode_token(token, assignment, eva_to_triple)
        decoded.append(original)

    return decoded


def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    """Fraction of decoded tokens that are dictionary hits."""
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# Build triple -> EVA chars reverse map
# ---------------------------------------------------------------------------

def _build_triple_to_eva_chars(eva_to_triple: Dict[str, str]) -> Dict[str, List[str]]:
    """Map each triple_key to the list of EVA chars that produce it."""
    result: Dict[str, List[str]] = defaultdict(list)
    for eva_char, triple_key in eva_to_triple.items():
        result[triple_key].append(eva_char)
    return dict(result)


# ---------------------------------------------------------------------------
# Count tokens affected by a triple
# ---------------------------------------------------------------------------

def _count_tokens_affected(
    tokens: List[str],
    triple_key: str,
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> int:
    """Count how many tokens contain at least one EVA char mapping to triple_key."""
    # Build set of EVA chars for this triple (excluding modifier chars)
    target_chars = set()
    for eva_char, tk in eva_to_triple.items():
        if tk == triple_key and eva_char not in modifier_chars:
            target_chars.add(eva_char)

    count = 0
    for token in tokens:
        chars = tokenize_eva_chars(token)
        if any(ch in target_chars for ch in chars):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Phase 19.8 anchor syllables
# ---------------------------------------------------------------------------

ANCHOR_SYLLABLES = frozenset({
    'de', 'be', 'ne', 'et', 'te', 'in', 'ter', 'ra',
    'ro', 'sa', 'se', 'la', 'ad', 'di',
})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_triple_sensitivity() -> None:
    """Step 24.1: Leave-one-out sensitivity analysis for all 25 triples."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 24.1: Leave-One-Out Triple Sensitivity Analysis")
    print("=" * 70)

    rdir = _results_dir()

    # ---- 1. Load Phase 16 pipeline ----------------------------------------
    print("\n  1. Loading Phase 16 pipeline ...")

    combined = _load_json(str(rdir / "combined_refine.json"))
    if combined is None:
        print("    [SKIP] combined_refine.json not found -- run combined-refine first")
        return
    assignment = combined.get("best_assignment", {})
    print(f"      Assignment: {len(assignment)} triples")

    mod_data = _load_json(str(rdir / "modifier_integrate.json"))
    if mod_data is None:
        print("    [SKIP] modifier_integrate.json not found -- run mod-integrate first")
        return
    modifier_chars = set(mod_data.get("modifier_chars", []))
    print(f"      Modifier chars: {len(modifier_chars)}")

    # Build modifier rules from classifications
    modifier_rules: Dict[str, str] = {}
    for cls in mod_data.get("classifications", []):
        if cls.get("final_classification") == "modifier":
            modifier_rules[cls["eva_char"]] = cls.get("modifier_type", "silent")

    # ---- 2. Load corpus and reference set ---------------------------------
    print("\n  2. Loading corpus and reference word set ...")

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"      Corpus: {len(all_tokens)} total tokens")

    # Seeded sample of 5000 tokens
    rng = random.Random(42)
    sample_tokens = rng.sample(all_tokens, min(5000, len(all_tokens)))
    print(f"      Sample: {len(sample_tokens)} tokens (seed=42)")

    # Build expanded reference word set
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()

    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"      Reference set: {len(ref_word_set)} words")

    # Build reference bigrams for bigram plausibility
    try:
        ref_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2]
    except (NameError, KeyError):
        ref_words = sorted(base_words)
    ref_bigrams = _build_ref_bigrams(ref_words[:5000])
    print(f"      Reference bigrams: {len(ref_bigrams)}")

    # ---- 3. Establish baseline --------------------------------------------
    print("\n  3. Establishing baseline (R3 combined) ...")

    baseline_decoded = _decode_r3_combined(
        sample_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_dict_hit = _compute_dict_hit(baseline_decoded, ref_word_set)
    baseline_bigram_hits = _count_bigram_hits(baseline_decoded, ref_bigrams)

    print(f"      Baseline dict_hit: {baseline_dict_hit:.4f} ({baseline_dict_hit:.1%})")
    print(f"      Baseline bigram hits: {baseline_bigram_hits}")

    # ---- 4. Build reverse maps --------------------------------------------
    triple_to_eva = _build_triple_to_eva_chars(eva_to_triple)

    # ---- 5. Leave-one-out for each triple ---------------------------------
    triple_keys = sorted(assignment.keys())
    n_triples = len(triple_keys)
    print(f"\n  4. Running leave-one-out for {n_triples} triples ...")

    sensitivities: List[TripleSensitivity] = []

    for idx, tk in enumerate(triple_keys):
        assigned_syl = assignment[tk]
        eva_chars = triple_to_eva.get(tk, [])

        # Count tokens affected
        tokens_affected = _count_tokens_affected(
            sample_tokens, tk, eva_to_triple, modifier_chars,
        )

        # Create modified assignment with this triple removed
        loo_assignment = {k: v for k, v in assignment.items() if k != tk}

        # Re-decode
        loo_decoded = _decode_r3_combined(
            sample_tokens, loo_assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        loo_dict_hit = _compute_dict_hit(loo_decoded, ref_word_set)
        loo_bigram_hits = _count_bigram_hits(loo_decoded, ref_bigrams)

        # Deltas (negative = removal hurts performance = triple is useful)
        dict_hit_delta = loo_dict_hit - baseline_dict_hit
        bigram_delta = loo_bigram_hits - baseline_bigram_hits

        sensitivities.append(TripleSensitivity(
            triple_key=tk,
            eva_chars=eva_chars,
            assigned_syllable=assigned_syl,
            tokens_affected=tokens_affected,
            dict_hit_delta=round(dict_hit_delta, 6),
            bigram_delta=bigram_delta,
            sensitivity_rank=0,  # filled in after sorting
            classification='',   # filled in below
            anchor_backed=False,  # filled in below
        ))

        progress = f"[{idx + 1}/{n_triples}]"
        direction = "drop" if dict_hit_delta < 0 else ("gain" if dict_hit_delta > 0 else "flat")
        print(f"      {progress} {tk:<45} -> {assigned_syl:<6} "
              f"delta={dict_hit_delta:+.4f} ({direction}), "
              f"bigram_delta={bigram_delta:+d}, "
              f"affected={tokens_affected}")

    # ---- 6. Rank and classify ---------------------------------------------
    print(f"\n  5. Ranking and classifying triples ...")

    # Sort by dict_hit_delta ascending (most negative = most impact = rank 1)
    sensitivities.sort(key=lambda s: s.dict_hit_delta)
    for rank, s in enumerate(sensitivities, 1):
        s.sensitivity_rank = rank

    # Classify
    for s in sensitivities:
        if s.dict_hit_delta < -0.03 and s.bigram_delta < 0:
            s.classification = 'probably_correct'
        elif s.dict_hit_delta > -0.005 and s.tokens_affected > 100:
            # dict-hit increase or negligible drop AND triple covers many tokens
            s.classification = 'probably_wrong'
        else:
            s.classification = 'uncertain'

    # ---- 7. Cross-reference Phase 19.8 anchors ----------------------------
    print(f"\n  6. Cross-referencing with anchor syllables ...")

    anchor_overrides: List[str] = []
    for s in sensitivities:
        if s.assigned_syllable.lower() in ANCHOR_SYLLABLES:
            s.anchor_backed = True
            if s.classification != 'probably_correct':
                anchor_overrides.append(s.triple_key)
                s.classification = 'probably_correct'

    if anchor_overrides:
        print(f"      Overridden to probably_correct by anchor match: "
              f"{len(anchor_overrides)}")
        for tk in anchor_overrides:
            s_match = next(s for s in sensitivities if s.triple_key == tk)
            print(f"        {tk} -> {s_match.assigned_syllable}")
    else:
        print(f"      No anchor overrides needed")

    # ---- 8. Summary -------------------------------------------------------
    probably_correct = [s for s in sensitivities if s.classification == 'probably_correct']
    uncertain = [s for s in sensitivities if s.classification == 'uncertain']
    probably_wrong = [s for s in sensitivities if s.classification == 'probably_wrong']

    print(f"\n  7. Classification summary:")
    print(f"      probably_correct: {len(probably_correct)}")
    print(f"      uncertain:       {len(uncertain)}")
    print(f"      probably_wrong:  {len(probably_wrong)}")

    print(f"\n      --- Probably Correct ---")
    for s in probably_correct:
        anchor = " [anchor]" if s.anchor_backed else ""
        print(f"        rank {s.sensitivity_rank:>2}: {s.triple_key:<45} "
              f"-> {s.assigned_syllable:<6} delta={s.dict_hit_delta:+.4f}{anchor}")

    print(f"\n      --- Uncertain ---")
    for s in uncertain:
        print(f"        rank {s.sensitivity_rank:>2}: {s.triple_key:<45} "
              f"-> {s.assigned_syllable:<6} delta={s.dict_hit_delta:+.4f}")

    print(f"\n      --- Probably Wrong ---")
    for s in probably_wrong:
        print(f"        rank {s.sensitivity_rank:>2}: {s.triple_key:<45} "
              f"-> {s.assigned_syllable:<6} delta={s.dict_hit_delta:+.4f} "
              f"(affected={s.tokens_affected})")

    # ---- 9. Verdict -------------------------------------------------------
    elapsed = time.time() - t0

    if len(probably_correct) >= len(probably_wrong) * 2:
        verdict = (
            f"STRONG SIGNAL: {len(probably_correct)}/{n_triples} triples "
            f"probably_correct ({len(probably_wrong)} probably_wrong). "
            f"Majority of assignments are load-bearing."
        )
    elif len(probably_correct) > len(probably_wrong):
        verdict = (
            f"MODERATE SIGNAL: {len(probably_correct)}/{n_triples} triples "
            f"probably_correct vs {len(probably_wrong)} probably_wrong. "
            f"Assignment partially correct."
        )
    else:
        verdict = (
            f"WEAK SIGNAL: only {len(probably_correct)}/{n_triples} triples "
            f"probably_correct vs {len(probably_wrong)} probably_wrong. "
            f"Assignment may be largely spurious."
        )

    print(f"\n  Verdict: {verdict}")

    # ---- 10. Save ---------------------------------------------------------
    result = SensitivityResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        baseline_dict_hit=round(baseline_dict_hit, 6),
        baseline_bigram_hits=baseline_bigram_hits,
        baseline_n_tokens=len(sample_tokens),
        n_triples=n_triples,
        sensitivities=[_convert(asdict(s)) for s in sensitivities],
        n_probably_correct=len(probably_correct),
        n_uncertain=len(uncertain),
        n_probably_wrong=len(probably_wrong),
        probably_correct_triples=[s.triple_key for s in probably_correct],
        uncertain_triples=[s.triple_key for s in uncertain],
        probably_wrong_triples=[s.triple_key for s in probably_wrong],
        anchor_overrides=anchor_overrides,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = rdir / "triple_sensitivity.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2, ensure_ascii=False)

    print(f"\n  -> {out_path} ({elapsed:.1f}s)")
