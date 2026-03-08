"""
Phase 24.5 -- Corrected Table Assembly
=======================================
Assembles the final corrected triple-to-syllable table by integrating
evidence from steps 24.1 (sensitivity), 24.2 (targeted swap), and
24.3 (bigram filter).  For each of the 25 feature triples, determines
whether the Phase 16 original or the swapped assignment should be used,
assigns a confidence tier, and computes table-level quality metrics.

If the bigram filter recommends "phase16" (indicating overfitting by
the targeted swaps), the Phase 16 table is used as the final table.

Dependency chain:
    combined_refine.json      (Phase 15 best_assignment)
    triple_sensitivity.json   (Step 24.1 classifications)
    targeted_swap.json        (Step 24.2 corrected assignments)
    bigram_filter.json        (Step 24.3 overfitting verdict)
    cross_approach.json       (anchor word mappings)
        -> corrected_table.json (this step)
"""

import json
import math
import os
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
    build_cv_syllable_table,
    build_syllable_frequency_table,
    load_reference_corpus,
)


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


def _jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two distributions (as dicts)."""
    all_keys = set(p) | set(q)
    p_arr = [p.get(k, 1e-10) for k in all_keys]
    q_arr = [q.get(k, 1e-10) for k in all_keys]
    # Normalize
    p_sum = sum(p_arr)
    q_sum = sum(q_arr)
    p_arr = [x / p_sum for x in p_arr]
    q_arr = [x / q_sum for x in q_arr]
    m_arr = [(pi + qi) / 2 for pi, qi in zip(p_arr, q_arr)]

    # KL divergence
    def kl(a: List[float], b: List[float]) -> float:
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 0)

    return (kl(p_arr, m_arr) + kl(q_arr, m_arr)) / 2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TripleProvenance:
    triple_key: str
    eva_chars: List[str]
    original_syllable: str
    final_syllable: str
    was_swapped: bool
    confidence_tier: str  # CONFIRMED, CORRECTED, ORIGINAL, UNCERTAIN
    sensitivity_classification: str  # from 24.1
    dict_hit_delta: float
    swap_improvement: float
    evidence: str


@dataclass
class CorrectedTableResult:
    timestamp: str
    # Table
    final_assignment: Dict[str, str]
    n_triples: int
    # Provenance
    provenance: List[Dict]
    n_confirmed: int
    n_corrected: int
    n_original: int
    n_uncertain: int
    # Quality metrics
    frequency_jsd: float
    family_coherence: float
    grid_shape_score: float
    # Diff from Phase 16
    n_swaps: int
    swapped_triples: List[Dict]  # list of {triple_key, old, new}
    # Recommendation
    bigram_filter_passed: bool
    recommended_table: str  # "corrected" or "phase16"
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Build triple -> EVA chars reverse map
# ---------------------------------------------------------------------------

def _build_triple_to_eva_chars(
    eva_to_triple: Dict[str, str],
) -> Dict[str, List[str]]:
    """Map each triple_key to the list of EVA chars that produce it."""
    result: Dict[str, List[str]] = defaultdict(list)
    for eva_char, triple_key in eva_to_triple.items():
        result[triple_key].append(eva_char)
    return dict(result)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def _compute_frequency_jsd(
    assignment: Dict[str, str],
    triple_corpus_freq: Dict[str, float],
    ref_syllable_freq: Dict[str, float],
) -> float:
    """JSD between assigned-syllable frequency distribution (weighted by
    triple corpus frequency) and reference Latin syllable frequencies."""
    # Build the assigned frequency distribution
    assigned_freq: Dict[str, float] = defaultdict(float)
    for triple_key, syllable in assignment.items():
        weight = triple_corpus_freq.get(triple_key, 1e-6)
        assigned_freq[syllable] += weight

    # Normalise
    total = sum(assigned_freq.values())
    if total > 0:
        assigned_freq = {k: v / total for k, v in assigned_freq.items()}

    return _jsd(assigned_freq, ref_syllable_freq)


def _compute_family_coherence(assignment: Dict[str, str]) -> float:
    """Fraction of first_stroke families where all member triples share
    the same onset consonant in their assigned syllable."""
    # Group triples by first_stroke
    families: Dict[str, List[str]] = defaultdict(list)
    for triple_key, syllable in assignment.items():
        parts = triple_key.split(',')
        if len(parts) >= 1:
            first_stroke = parts[0]
            families[first_stroke].append(syllable)

    if not families:
        return 0.0

    coherent = 0
    for first_stroke, syllables in families.items():
        # Extract onset consonant from each syllable
        onsets = set()
        for syl in syllables:
            # Onset = leading consonant(s); if starts with vowel, onset is ''
            vowels = set('aeiou')
            onset = ''
            for ch in syl.lower():
                if ch in vowels:
                    break
                onset += ch
            onsets.add(onset)
        if len(onsets) == 1:
            coherent += 1

    return coherent / len(families)


def _compute_grid_shape_score(assignment: Dict[str, str]) -> float:
    """Check if the triple -> syllable table has consonant x vowel structure.

    Organise triples into a grid: first_stroke (rows) x last_stroke (columns).
    For each row, check if all syllables share the same onset consonant.
    For each column, check if all syllables share the same nucleus vowel.
    Score = fraction of rows+columns that are pure."""
    vowel_set = set('aeiou')

    # Build grid cells
    grid: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for triple_key, syllable in assignment.items():
        parts = triple_key.split(',')
        if len(parts) >= 2:
            first_stroke = parts[0]
            last_stroke = parts[1]
            grid[(first_stroke, last_stroke)].append(syllable)

    # Extract rows and columns
    rows: Dict[str, List[str]] = defaultdict(list)
    cols: Dict[str, List[str]] = defaultdict(list)
    for (fs, ls), syls in grid.items():
        rows[fs].extend(syls)
        cols[ls].extend(syls)

    # Only consider rows/columns with 2+ members
    n_checked = 0
    n_pure = 0

    for row_key, syls in rows.items():
        if len(syls) < 2:
            continue
        n_checked += 1
        # Extract onsets
        onsets = set()
        for syl in syls:
            onset = ''
            for ch in syl.lower():
                if ch in vowel_set:
                    break
                onset += ch
            onsets.add(onset)
        if len(onsets) == 1:
            n_pure += 1

    for col_key, syls in cols.items():
        if len(syls) < 2:
            continue
        n_checked += 1
        # Extract nucleus vowels
        nuclei = set()
        for syl in syls:
            for ch in syl.lower():
                if ch in vowel_set:
                    nuclei.add(ch)
                    break
        if len(nuclei) == 1:
            n_pure += 1

    if n_checked == 0:
        return 0.0
    return n_pure / n_checked


def _compute_triple_corpus_freq(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
) -> Dict[str, float]:
    """Compute the corpus frequency of each triple_key (fraction of all
    triple occurrences in the corpus)."""
    counts: Counter = Counter()
    for token in tokens:
        chars = tokenize_eva_chars(token)
        for ch in chars:
            tk = eva_to_triple.get(ch)
            if tk:
                counts[tk] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_corrected_table() -> None:
    """Step 24.5: Assemble corrected triple-to-syllable table."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 24.5: Corrected Table Assembly")
    print("=" * 70)

    rdir = _results_dir()

    # ---- 1. Load Phase 16 original assignment ---------------------------------
    print("\n  1. Loading Phase 16 original assignment ...")

    combined = _load_json(str(rdir / "combined_refine.json"))
    if combined is None:
        print("    [SKIP] combined_refine.json not found -- run combined-refine first")
        return
    original_assignment = combined.get("best_assignment", {})
    print(f"      Original assignment: {len(original_assignment)} triples")

    # ---- 2. Load corrected assignment from targeted swap ----------------------
    print("\n  2. Loading targeted swap results ...")

    swap_data = _load_json(str(rdir / "targeted_swap.json"))
    if swap_data is not None:
        corrected_assignment = swap_data.get("final_assignment", {})
        print(f"      Corrected assignment: {len(corrected_assignment)} triples")
    else:
        print("    [INFO] targeted_swap.json not found -- using original assignment")
        corrected_assignment = dict(original_assignment)

    # ---- 3. Load bigram filter verdict ----------------------------------------
    print("\n  3. Loading bigram filter results ...")

    bigram_data = _load_json(str(rdir / "bigram_filter.json"))
    if bigram_data is not None:
        bigram_filter_passed = bigram_data.get("gate_passed", True)
        bigram_recommendation = bigram_data.get("recommended_table", "corrected")
        bigram_verdict = bigram_data.get("verdict", "")
        print(f"      Bigram filter passed: {bigram_filter_passed}")
        print(f"      Recommendation: {bigram_recommendation}")
    else:
        print("    [INFO] bigram_filter.json not found -- assuming corrected table OK")
        bigram_filter_passed = True
        bigram_recommendation = "corrected"
        bigram_verdict = "no bigram filter data"

    # ---- 4. Load sensitivity classifications ----------------------------------
    print("\n  4. Loading sensitivity analysis ...")

    sensitivity_data = _load_json(str(rdir / "triple_sensitivity.json"))
    sensitivity_map: Dict[str, Dict] = {}
    if sensitivity_data is not None:
        for s in sensitivity_data.get("sensitivities", []):
            tk = s.get("triple_key", "")
            sensitivity_map[tk] = s
        print(f"      Loaded {len(sensitivity_map)} triple classifications")
    else:
        print("    [INFO] triple_sensitivity.json not found -- classifications unavailable")

    # ---- 5. Load cross_approach anchors ----------------------------------------
    print("\n  5. Loading cross-approach anchor data ...")

    cross_data = _load_json(str(rdir / "cross_approach.json"))
    anchor_triples: Set[str] = set()
    if cross_data is not None:
        # Extract triples involved in confirmed word mappings (exact or edit2)
        per_word = cross_data.get("per_word_results", [])
        confirmed_words = [
            pw for pw in per_word
            if pw.get("exact_match") or pw.get("edit2_match")
        ]
        print(f"      Confirmed word mappings: {len(confirmed_words)}")

        # For each confirmed word, find which triples are used in its tokens
        eva_to_triple = build_eva_to_triple_lookup()
        for pw in confirmed_words:
            for token in pw.get("voynich_tokens", []):
                chars = tokenize_eva_chars(token)
                for ch in chars:
                    tk = eva_to_triple.get(ch)
                    if tk and tk in original_assignment:
                        anchor_triples.add(tk)
        print(f"      Anchor-backed triples: {len(anchor_triples)}")
    else:
        print("    [INFO] cross_approach.json not found -- no anchor data")
        eva_to_triple = build_eva_to_triple_lookup()

    # ---- 6. Determine final table based on bigram filter -----------------------
    print("\n  6. Determining final table ...")

    use_phase16 = (bigram_recommendation == "phase16")
    if use_phase16:
        final_assignment = dict(original_assignment)
        recommended_table = "phase16"
        print("      -> Using Phase 16 table (bigram filter flagged overfitting)")
    else:
        final_assignment = dict(corrected_assignment)
        recommended_table = "corrected"
        print("      -> Using corrected table")

    # Ensure final_assignment has all triples from original
    for tk in original_assignment:
        if tk not in final_assignment:
            final_assignment[tk] = original_assignment[tk]

    # ---- 7. Build provenance for each triple ----------------------------------
    print("\n  7. Building provenance records ...")

    triple_to_eva = _build_triple_to_eva_chars(eva_to_triple)
    triple_keys = sorted(original_assignment.keys())
    provenance_list: List[TripleProvenance] = []
    swapped_triples: List[Dict[str, str]] = []

    for tk in triple_keys:
        original_syl = original_assignment.get(tk, "???")
        corrected_syl = corrected_assignment.get(tk, original_syl)
        final_syl = final_assignment.get(tk, original_syl)
        was_swapped = (corrected_syl != original_syl)
        eva_chars = triple_to_eva.get(tk, [])

        # Sensitivity info
        sens = sensitivity_map.get(tk, {})
        sens_classification = sens.get("classification", "unknown")
        dict_hit_delta = sens.get("dict_hit_delta", 0.0)

        # Swap improvement (from targeted_swap data if available)
        swap_improvement = 0.0
        if swap_data is not None and was_swapped:
            # Look for per-swap details
            swap_details = swap_data.get("swap_details", [])
            for sd in swap_details:
                if sd.get("triple_key") == tk:
                    swap_improvement = sd.get("combined_score", 0.0)
                    break

        # Determine confidence tier
        is_anchor = tk in anchor_triples
        is_probably_correct = (sens_classification == "probably_correct")
        is_probably_wrong = (sens_classification == "probably_wrong")
        is_uncertain = (sens_classification == "uncertain")

        if is_anchor and (is_probably_correct or sens_classification == "unknown"):
            confidence_tier = "CONFIRMED"
            evidence = f"Anchor-backed ({sens_classification}); delta={dict_hit_delta:+.4f}"
        elif was_swapped and bigram_filter_passed:
            confidence_tier = "CORRECTED"
            evidence = (
                f"Swapped {original_syl}->{corrected_syl}; "
                f"improvement={swap_improvement:.4f}; "
                f"sensitivity={sens_classification}"
            )
            swapped_triples.append({
                "triple_key": tk,
                "old": original_syl,
                "new": corrected_syl,
            })
        elif not was_swapped and (is_probably_correct or is_uncertain):
            confidence_tier = "ORIGINAL"
            evidence = f"Kept original; sensitivity={sens_classification}; delta={dict_hit_delta:+.4f}"
        else:
            confidence_tier = "UNCERTAIN"
            if is_probably_wrong and not was_swapped:
                evidence = (
                    f"Probably wrong but no swap improved both metrics; "
                    f"delta={dict_hit_delta:+.4f}"
                )
            elif was_swapped and not bigram_filter_passed:
                evidence = (
                    f"Swap reverted (bigram filter flagged overfitting); "
                    f"original={original_syl}, attempted={corrected_syl}"
                )
            else:
                evidence = f"Sensitivity={sens_classification}; delta={dict_hit_delta:+.4f}"

        provenance_list.append(TripleProvenance(
            triple_key=tk,
            eva_chars=eva_chars,
            original_syllable=original_syl,
            final_syllable=final_syl,
            was_swapped=was_swapped,
            confidence_tier=confidence_tier,
            sensitivity_classification=sens_classification,
            dict_hit_delta=round(dict_hit_delta, 6),
            swap_improvement=round(swap_improvement, 6),
            evidence=evidence,
        ))

        marker = ""
        if confidence_tier == "CONFIRMED":
            marker = " [ANCHOR]"
        elif confidence_tier == "CORRECTED":
            marker = " [SWAPPED]"
        elif confidence_tier == "UNCERTAIN":
            marker = " [?]"
        print(f"      {tk:<45} {original_syl:<6} -> {final_syl:<6} "
              f"{confidence_tier}{marker}")

    # Count tiers
    n_confirmed = sum(1 for p in provenance_list if p.confidence_tier == "CONFIRMED")
    n_corrected = sum(1 for p in provenance_list if p.confidence_tier == "CORRECTED")
    n_original = sum(1 for p in provenance_list if p.confidence_tier == "ORIGINAL")
    n_uncertain = sum(1 for p in provenance_list if p.confidence_tier == "UNCERTAIN")

    print(f"\n      Tier counts: CONFIRMED={n_confirmed}, CORRECTED={n_corrected}, "
          f"ORIGINAL={n_original}, UNCERTAIN={n_uncertain}")

    # ---- 8. Compute quality metrics -------------------------------------------
    print("\n  8. Computing table quality metrics ...")

    # 8a. Frequency JSD
    print("      8a. Computing frequency JSD ...")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    triple_corpus_freq = _compute_triple_corpus_freq(all_tokens, eva_to_triple)

    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    except (FileNotFoundError, KeyError):
        ref_corpus = None

    ref_syllable_freq = build_syllable_frequency_table('latin', ref_corpus)
    frequency_jsd = _compute_frequency_jsd(
        final_assignment, triple_corpus_freq, ref_syllable_freq,
    )
    print(f"      Frequency JSD: {frequency_jsd:.6f}")

    # 8b. Family coherence
    print("      8b. Computing family coherence ...")
    family_coherence = _compute_family_coherence(final_assignment)
    print(f"      Family coherence: {family_coherence:.4f} "
          f"({family_coherence:.1%} of first_stroke families are pure)")

    # 8c. Grid shape score
    print("      8c. Computing grid shape score ...")
    grid_shape_score = _compute_grid_shape_score(final_assignment)
    print(f"      Grid shape score: {grid_shape_score:.4f} "
          f"({grid_shape_score:.1%} of rows+columns are pure)")

    # ---- 9. Diff report -------------------------------------------------------
    print("\n  9. Diff report (Phase 16 -> final):")

    n_swaps = sum(1 for tk in triple_keys
                  if final_assignment.get(tk) != original_assignment.get(tk))
    if n_swaps == 0:
        print("      No changes from Phase 16 table.")
    else:
        print(f"      {n_swaps} triple(s) changed:")
        for tk in triple_keys:
            orig = original_assignment.get(tk, "???")
            final = final_assignment.get(tk, "???")
            if orig != final:
                print(f"        {tk}: {orig} -> {final}")

    # ---- 10. Final verdict ----------------------------------------------------
    elapsed = time.time() - t0

    if use_phase16:
        verdict = (
            f"PHASE16 RETAINED: bigram filter flagged overfitting in targeted swaps. "
            f"Using Phase 16 table unchanged. "
            f"Quality: JSD={frequency_jsd:.4f}, "
            f"family_coherence={family_coherence:.2f}, "
            f"grid_shape={grid_shape_score:.2f}."
        )
    elif n_swaps == 0:
        verdict = (
            f"NO CHANGES: targeted swaps produced no improvements over Phase 16. "
            f"Quality: JSD={frequency_jsd:.4f}, "
            f"family_coherence={family_coherence:.2f}, "
            f"grid_shape={grid_shape_score:.2f}."
        )
    else:
        verdict = (
            f"CORRECTED: {n_swaps} triple(s) swapped. "
            f"Tiers: {n_confirmed} confirmed, {n_corrected} corrected, "
            f"{n_original} original, {n_uncertain} uncertain. "
            f"Quality: JSD={frequency_jsd:.4f}, "
            f"family_coherence={family_coherence:.2f}, "
            f"grid_shape={grid_shape_score:.2f}."
        )

    print(f"\n  Verdict: {verdict}")

    # ---- 11. Save -------------------------------------------------------------
    result = CorrectedTableResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        final_assignment=final_assignment,
        n_triples=len(final_assignment),
        provenance=[_convert(asdict(p)) for p in provenance_list],
        n_confirmed=n_confirmed,
        n_corrected=n_corrected,
        n_original=n_original,
        n_uncertain=n_uncertain,
        frequency_jsd=round(frequency_jsd, 6),
        family_coherence=round(family_coherence, 4),
        grid_shape_score=round(grid_shape_score, 4),
        n_swaps=n_swaps,
        swapped_triples=swapped_triples,
        bigram_filter_passed=bigram_filter_passed,
        recommended_table=recommended_table,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = rdir / "corrected_table.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2, ensure_ascii=False)

    print(f"\n  -> {out_path} ({elapsed:.1f}s)")
