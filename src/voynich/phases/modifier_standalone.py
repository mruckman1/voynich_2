"""
Phase 16.1 – Modifier Standalone Analysis (Approach B)
======================================================
Identifies modifier candidates by analysing distributional properties of
each EVA character/ligature: standalone frequency, positional distribution
(initial/medial/final), positional entropy, and adjacency entropy.

Characters that never appear as single-character tokens and show strong
positional restrictions are strong modifier candidates — like Devanagari
virama, which never stands alone and always follows a consonant.

Dependency chain:
    corpus (IVTFF)
        → modifier_standalone.json (this step)
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import EVA_VISUAL_COMPONENTS


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


def _entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy (base-2) of a count dictionary."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EVACharProfile:
    """Distributional profile for a single EVA character/ligature."""
    eva_char: str
    triple_key: str
    corpus_frequency: int
    appears_as_solo_token: bool
    solo_token_count: int
    n_tokens_containing: int
    position_initial_count: int
    position_medial_count: int
    position_final_count: int
    position_initial_pct: float
    position_medial_pct: float
    position_final_pct: float
    positional_entropy: float
    left_neighbour_entropy: float
    right_neighbour_entropy: float
    n_distinct_left_neighbours: int
    n_distinct_right_neighbours: int
    modifier_score: float
    position_bias: str  # 'initial', 'medial', 'final', 'uniform'


@dataclass
class StandaloneResult:
    n_eva_chars: int
    n_never_solo: int
    n_modifier_candidates: int
    modifier_threshold: float
    char_profiles: List[Dict]
    modifier_candidates: List[str]   # EVA chars scoring above threshold
    modifier_triple_keys: List[str]  # triple_keys of modifier candidates
    solo_chars: List[str]            # chars that DO appear as solo tokens
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def build_char_profiles(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
) -> List[EVACharProfile]:
    """Build distributional profile for each EVA character/ligature."""

    # --- Gather character-level statistics ---
    char_freq: Counter = Counter()
    solo_chars: Counter = Counter()
    # Positional counts: initial / medial / final
    pos_initial: Counter = Counter()
    pos_medial: Counter = Counter()
    pos_final: Counter = Counter()
    # Adjacency counts
    left_neighbours: Dict[str, Counter] = {}
    right_neighbours: Dict[str, Counter] = {}
    # Tokens containing each char
    tokens_containing: Counter = Counter()

    for token in tokens:
        chars = tokenize_eva_chars(token)
        n = len(chars)

        if n == 1:
            solo_chars[chars[0]] += 1

        seen_in_token: Set[str] = set()
        for ci, ch in enumerate(chars):
            char_freq[ch] += 1

            if ch not in seen_in_token:
                tokens_containing[ch] += 1
                seen_in_token.add(ch)

            # Position
            if n == 1:
                # Single-char token counts as all positions
                pos_initial[ch] += 1
                pos_final[ch] += 1
            elif ci == 0:
                pos_initial[ch] += 1
            elif ci == n - 1:
                pos_final[ch] += 1
            else:
                pos_medial[ch] += 1

            # Adjacency
            if ch not in left_neighbours:
                left_neighbours[ch] = Counter()
                right_neighbours[ch] = Counter()

            if ci > 0:
                left_neighbours[ch][chars[ci - 1]] += 1
            if ci < n - 1:
                right_neighbours[ch][chars[ci + 1]] += 1

    # --- Build profiles for all known EVA chars ---
    all_eva_chars = sorted(EVA_VISUAL_COMPONENTS.keys())
    profiles: List[EVACharProfile] = []

    for ch in all_eva_chars:
        freq = char_freq.get(ch, 0)
        if freq == 0:
            continue  # Skip unattested chars

        triple_key = eva_to_triple.get(ch, '?')
        solo_count = solo_chars.get(ch, 0)
        appears_solo = solo_count > 0

        ini = pos_initial.get(ch, 0)
        med = pos_medial.get(ch, 0)
        fin = pos_final.get(ch, 0)
        total_pos = ini + med + fin

        ini_pct = ini / total_pos if total_pos > 0 else 0.0
        med_pct = med / total_pos if total_pos > 0 else 0.0
        fin_pct = fin / total_pos if total_pos > 0 else 0.0

        pos_ent = _entropy({'initial': ini, 'medial': med, 'final': fin})

        left_ent = _entropy(left_neighbours.get(ch, Counter()))
        right_ent = _entropy(right_neighbours.get(ch, Counter()))
        n_left = len(left_neighbours.get(ch, Counter()))
        n_right = len(right_neighbours.get(ch, Counter()))

        # Determine position bias
        if total_pos > 0:
            max_pct = max(ini_pct, med_pct, fin_pct)
            if max_pct > 0.6:
                if ini_pct == max_pct:
                    bias = 'initial'
                elif med_pct == max_pct:
                    bias = 'medial'
                else:
                    bias = 'final'
            else:
                bias = 'uniform'
        else:
            bias = 'uniform'

        # Compute composite modifier score
        score = _score_modifier(
            appears_solo=appears_solo,
            pos_entropy=pos_ent,
            left_adj_entropy=left_ent,
            right_adj_entropy=right_ent,
            freq=freq,
            median_freq=0,  # will be filled in after all chars are computed
        )

        profiles.append(EVACharProfile(
            eva_char=ch,
            triple_key=triple_key,
            corpus_frequency=freq,
            appears_as_solo_token=appears_solo,
            solo_token_count=solo_count,
            n_tokens_containing=tokens_containing.get(ch, 0),
            position_initial_count=ini,
            position_medial_count=med,
            position_final_count=fin,
            position_initial_pct=round(ini_pct, 4),
            position_medial_pct=round(med_pct, 4),
            position_final_pct=round(fin_pct, 4),
            positional_entropy=round(pos_ent, 4),
            left_neighbour_entropy=round(left_ent, 4),
            right_neighbour_entropy=round(right_ent, 4),
            n_distinct_left_neighbours=n_left,
            n_distinct_right_neighbours=n_right,
            modifier_score=0.0,  # placeholder
            position_bias=bias,
        ))

    # Second pass: compute modifier scores with median frequency
    if profiles:
        freqs_sorted = sorted(p.corpus_frequency for p in profiles)
        median_freq = freqs_sorted[len(freqs_sorted) // 2]

        # Max entropy for normalisation
        max_pos_entropy = math.log2(3)  # 3 positions
        all_chars_count = len(profiles)
        max_adj_entropy = math.log2(max(all_chars_count, 2))

        for p in profiles:
            p.modifier_score = round(_score_modifier(
                appears_solo=p.appears_as_solo_token,
                pos_entropy=p.positional_entropy,
                left_adj_entropy=p.left_neighbour_entropy,
                right_adj_entropy=p.right_neighbour_entropy,
                freq=p.corpus_frequency,
                median_freq=median_freq,
                max_pos_entropy=max_pos_entropy,
                max_adj_entropy=max_adj_entropy,
            ), 4)

    # Sort by modifier_score descending
    profiles.sort(key=lambda p: p.modifier_score, reverse=True)
    return profiles


def _score_modifier(
    appears_solo: bool,
    pos_entropy: float,
    left_adj_entropy: float,
    right_adj_entropy: float,
    freq: int,
    median_freq: int,
    max_pos_entropy: float = 1.585,
    max_adj_entropy: float = 5.0,
) -> float:
    """Composite modifier score in [0, 1].

    Components:
      0.30 * (never appears as standalone token)
      0.25 * (low positional entropy → positionally restricted)
      0.15 * (low left-adjacency entropy → restricted left context)
      0.15 * (low right-adjacency entropy → restricted right context)
      0.15 * (high frequency → modifiers tend to be common)
    """
    s = 0.0

    # Never-standalone: strongest signal
    s += 0.30 * (0.0 if appears_solo else 1.0)

    # Low positional entropy (normalised to [0, 1], inverted)
    if max_pos_entropy > 0:
        s += 0.25 * (1.0 - min(pos_entropy / max_pos_entropy, 1.0))

    # Low adjacency entropy
    if max_adj_entropy > 0:
        s += 0.15 * (1.0 - min(left_adj_entropy / max_adj_entropy, 1.0))
        s += 0.15 * (1.0 - min(right_adj_entropy / max_adj_entropy, 1.0))

    # High frequency (modifiers like 'i' are very common)
    if median_freq > 0:
        s += 0.15 * min(freq / (2 * median_freq), 1.0)

    return s


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_modifier_standalone() -> None:
    """Step 16.1: Standalone modifier candidate analysis (Approach B)."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 16.1: Modifier Standalone Analysis (Approach B)")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load corpus ───
    print("\n  1. Loading corpus …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    print(f"      {len(tokens)} tokens loaded")

    # ─── Build lookups ───
    eva_to_triple = build_eva_to_triple_lookup()

    # ─── Build character profiles ───
    print("\n  2. Building EVA character distributional profiles …")
    profiles = build_char_profiles(tokens, eva_to_triple)
    print(f"      {len(profiles)} attested EVA chars/ligatures profiled")

    # ─── Identify modifier candidates ───
    threshold = 0.6
    modifier_candidates = [p.eva_char for p in profiles if p.modifier_score >= threshold]
    modifier_triple_keys = list(dict.fromkeys(
        p.triple_key for p in profiles if p.modifier_score >= threshold
    ))
    solo_chars = [p.eva_char for p in profiles if p.appears_as_solo_token]
    never_solo = [p.eva_char for p in profiles if not p.appears_as_solo_token]

    print(f"\n  3. Results:")
    print(f"      Solo chars (appear as 1-char tokens): {solo_chars}")
    print(f"      Never-solo chars: {never_solo}")
    print(f"      Modifier candidates (score >= {threshold}): {modifier_candidates}")
    print(f"      Corresponding triple_keys: {modifier_triple_keys}")

    # ─── Print top profiles ───
    print(f"\n  4. Top 15 profiles by modifier score:")
    print(f"      {'Char':<8} {'Triple':<40} {'Freq':>6} {'Solo':>5} "
          f"{'Pos.Ent':>8} {'L.Ent':>6} {'R.Ent':>6} {'Score':>6} {'Bias':<8}")
    print("      " + "-" * 105)
    for p in profiles[:15]:
        print(f"      {p.eva_char:<8} {p.triple_key:<40} {p.corpus_frequency:>6} "
              f"{'Y' if p.appears_as_solo_token else 'N':>5} "
              f"{p.positional_entropy:>8.3f} "
              f"{p.left_neighbour_entropy:>6.3f} "
              f"{p.right_neighbour_entropy:>6.3f} "
              f"{p.modifier_score:>6.3f} {p.position_bias:<8}")

    # ─── Gate ───
    gate_passed = len(modifier_candidates) >= 5
    verdict = (
        f"PASS: {len(modifier_candidates)} modifier candidates identified "
        f"(threshold {threshold}). {len(never_solo)} chars never appear standalone."
        if gate_passed
        else f"FAIL: Only {len(modifier_candidates)} modifier candidates "
        f"(need >= 5 with score >= {threshold})."
    )

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ─── Save ───
    result = StandaloneResult(
        n_eva_chars=len(profiles),
        n_never_solo=len(never_solo),
        n_modifier_candidates=len(modifier_candidates),
        modifier_threshold=threshold,
        char_profiles=[_convert(asdict(p)) for p in profiles],
        modifier_candidates=modifier_candidates,
        modifier_triple_keys=modifier_triple_keys,
        solo_chars=solo_chars,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'modifier_standalone.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
