"""
Phase B.0 -- Ligature Analysis
===============================
Process ligature observation data and quantify EVA mis-segmentation.

Loads the ligature observation dataset and classifies each EVA pair as
confirmed ligature (connection_rate >= 0.8), possible ligature (0.5-0.8),
or non-ligature (< 0.5).  Then re-tokenizes the corpus with confirmed
ligatures merged and measures the impact on character inventory size and
mean token length.

Severity classification:
    >30% connected  -> "severe"  (major re-segmentation needed)
    10-30% connected -> "moderate" (targeted merges recommended)
    <10% connected  -> "minimal"  (current segmentation adequate)

Dependency chain:
    data/reference/ligature/ligature_observations.json
    corpus (IVTFF)
        -> ligature_analysis.json (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir, data_dir as _data_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.reference import EVA_VISUAL_COMPONENTS, load_ligature_observations


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
class LigatureAnalysisResult:
    """Full ligature mis-segmentation analysis."""
    n_pairs_examined: int
    n_confirmed_ligatures: int
    n_possible_ligatures: int
    n_non_ligatures: int
    overall_connection_rate: float
    confirmed_pairs: List[str]
    original_char_inventory_size: int
    new_char_inventory_size: int
    original_mean_token_length: float
    new_mean_token_length: float
    severity: str
    gate_result: str
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _classify_pairs(
    pair_summaries: List[Dict],
) -> Tuple[List[str], List[str], List[str], float]:
    """Classify pairs into confirmed/possible/non-ligatures.

    Returns (confirmed, possible, non_ligatures, overall_connection_rate).
    """
    confirmed: List[str] = []
    possible: List[str] = []
    non_ligatures: List[str] = []

    total_connected = 0
    total_examined = 0

    for ps in pair_summaries:
        pair_label = ps.get('pair', ps.get('pair_label', ''))
        rate = ps.get('connection_rate', 0.0)
        n_connected = ps.get('n_connected', 0)
        n_examined = ps.get('n_examined', 0)

        total_connected += n_connected
        total_examined += n_examined

        if rate >= 0.8:
            confirmed.append(pair_label)
        elif rate >= 0.5:
            possible.append(pair_label)
        else:
            non_ligatures.append(pair_label)

    overall_rate = total_connected / total_examined if total_examined > 0 else 0.0
    return confirmed, possible, non_ligatures, overall_rate


def _retokenize_with_ligatures(
    tokens: List[str],
    confirmed_pairs: List[str],
) -> Tuple[List[List[str]], int]:
    """Re-tokenize corpus merging confirmed ligature pairs.

    For each token, tokenize into EVA chars, then scan for consecutive
    pairs that appear in confirmed_pairs and merge them into a single unit.

    Returns (list of retokenized char sequences, new inventory size).
    """
    # Build set of confirmed pair strings for fast lookup
    confirmed_set = set(confirmed_pairs)

    all_retokenized: List[List[str]] = []
    new_chars_seen: set = set()

    for token in tokens:
        chars = tokenize_eva_chars(token)
        merged: List[str] = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1:
                pair = chars[i] + '+' + chars[i + 1]
                pair_concat = chars[i] + chars[i + 1]
                # Check both formats: "a+b" and "ab"
                if pair in confirmed_set or pair_concat in confirmed_set:
                    merged.append(pair_concat)
                    new_chars_seen.add(pair_concat)
                    i += 2
                    continue
            merged.append(chars[i])
            new_chars_seen.add(chars[i])
            i += 1

        all_retokenized.append(merged)

    return all_retokenized, len(new_chars_seen)


def _mean_token_length(tokens: List[str]) -> float:
    """Mean number of EVA characters per token."""
    if not tokens:
        return 0.0
    lengths = [len(tokenize_eva_chars(t)) for t in tokens]
    return sum(lengths) / len(lengths)


def _mean_retokenized_length(retokenized: List[List[str]]) -> float:
    """Mean number of units per retokenized token."""
    if not retokenized:
        return 0.0
    lengths = [len(chars) for chars in retokenized]
    return sum(lengths) / len(lengths)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ligature_analysis() -> None:
    """Phase B.0: Ligature observation analysis and mis-segmentation quantification."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE B.0: Ligature Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ---- Step 1: Load ligature observations ----
    print("\n  1. Loading ligature observations ...")
    obs = load_ligature_observations()

    if obs is None:
        print("      [WARN] ligature_observations.json not found.")
        print("      Creating result with empty observation data.")
        pair_summaries = []
    else:
        pair_summaries = obs.get('pair_summaries', [])
        print(f"      Loaded {len(pair_summaries)} pair summaries")

    # ---- Step 2: Classify pairs ----
    print("\n  2. Analyzing pair summaries ...")
    confirmed, possible, non_ligs, overall_rate = _classify_pairs(pair_summaries)

    print(f"      Confirmed ligatures (rate >= 0.80): {len(confirmed)}")
    print(f"      Possible ligatures  (rate 0.50-0.80): {len(possible)}")
    print(f"      Non-ligatures       (rate < 0.50): {len(non_ligs)}")
    print(f"      Overall connection rate: {overall_rate:.4f}")

    if confirmed:
        print(f"      Confirmed pairs: {confirmed}")

    # ---- Step 3: Compute overall connection rate ----
    print(f"\n  3. Overall connection rate: {overall_rate:.4f} "
          f"({overall_rate:.1%} of examined transitions are connected)")

    # ---- Step 4: Re-tokenize corpus with confirmed ligatures merged ----
    print("\n  4. Re-tokenizing corpus with confirmed ligatures merged ...")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    print(f"      {len(tokens)} tokens loaded")

    original_mean_len = _mean_token_length(tokens)
    print(f"      Original mean token length: {original_mean_len:.2f} EVA chars")

    retokenized, new_inventory_size = _retokenize_with_ligatures(tokens, confirmed)
    new_mean_len = _mean_retokenized_length(retokenized)
    print(f"      New char inventory size: {new_inventory_size}")
    print(f"      New mean token length: {new_mean_len:.2f} units")

    # ---- Step 5: Compare to original ----
    original_inventory = len(EVA_VISUAL_COMPONENTS)
    print(f"\n  5. Inventory comparison:")
    print(f"      Original: {original_inventory} characters")
    print(f"      After ligature merge: {new_inventory_size} characters")
    reduction_pct = (
        (original_inventory - new_inventory_size) / original_inventory * 100
        if original_inventory > 0 else 0.0
    )
    print(f"      Reduction: {reduction_pct:.1f}%")

    # ---- Step 6: Gate B.0 ----
    print("\n  6. Gate B.0: Severity classification ...")
    if overall_rate > 0.30:
        severity = "severe"
        gate_result = (
            "SEVERE: >30% of examined transitions are connected. "
            "Major re-segmentation needed before phonemic analysis. "
            "Merge all confirmed ligatures and rebuild feature triples."
        )
    elif overall_rate > 0.10:
        severity = "moderate"
        gate_result = (
            "MODERATE: 10-30% connected. Targeted merges recommended "
            "for confirmed pairs. Re-run feature decomposition with "
            "merged inventory."
        )
    else:
        severity = "minimal"
        gate_result = (
            "MINIMAL: <10% connected. Current EVA segmentation is "
            "adequate. Ligature effects are within noise floor."
        )

    print(f"      Severity: {severity}")
    print(f"      Gate result: {gate_result}")

    # ---- Verdict ----
    verdict = (
        f"Phase B.0 complete. {len(pair_summaries)} pairs examined: "
        f"{len(confirmed)} confirmed ligatures, {len(possible)} possible, "
        f"{len(non_ligs)} non-ligatures. "
        f"Overall connection rate {overall_rate:.1%} -> severity '{severity}'. "
        f"Inventory {original_inventory} -> {new_inventory_size} "
        f"({reduction_pct:.1f}% reduction). "
        f"Mean token length {original_mean_len:.2f} -> {new_mean_len:.2f}."
    )
    print(f"\n  Verdict: {verdict}")

    # ---- Save ----
    result = LigatureAnalysisResult(
        n_pairs_examined=len(pair_summaries),
        n_confirmed_ligatures=len(confirmed),
        n_possible_ligatures=len(possible),
        n_non_ligatures=len(non_ligs),
        overall_connection_rate=round(overall_rate, 4),
        confirmed_pairs=confirmed,
        original_char_inventory_size=original_inventory,
        new_char_inventory_size=new_inventory_size,
        original_mean_token_length=round(original_mean_len, 4),
        new_mean_token_length=round(new_mean_len, 4),
        severity=severity,
        gate_result=gate_result,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'ligature_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  Results saved -> {out_path}")
