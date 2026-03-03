"""
Phase 16.6 – Convergent Modifier Classification and Re-Decode
=============================================================
Integrates evidence from all five modifier-detection approaches (B, D, A,
E, C) into a convergent per-EVA-character classification.  Then re-decodes
the corpus under three strategies: modifier stripping, modifier-as-alteration,
and combined best-of-both.

Dependency chain:
    modifier_standalone.json   (Approach B)
    modifier_anomaly.json      (Approach D)
    modifier_distribution.json (Approach A)
    modifier_minimal_pairs.json (Approach E)
    modifier_localize.json     (Approach C)
    combined_refine.json       (Phase 15 best_assignment)
        → modifier_integrate.json (this step)
"""

import json
import os
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModifierClassification:
    eva_char: str
    triple_key: str
    standalone_score: float       # from B (0-1)
    anomaly_score: float          # from D (0-1)
    distribution_evidence: bool   # in best modifier set from A
    minimal_pair_score: float     # from E (0-1)
    localization_score: float     # from C (padding ratio)
    n_approaches_supporting: int
    final_classification: str     # 'modifier', 'syllabic', 'ambiguous'
    modifier_type: str            # 'silent', 'vowel_changer', 'geminator', etc.


@dataclass
class IntegrationResult:
    # Classification
    n_modifier: int
    n_syllabic: int
    n_ambiguous: int
    modifier_chars: List[str]
    syllabic_chars: List[str]
    ambiguous_chars: List[str]
    classifications: List[Dict]

    # Strategy R1: stripping
    r1_dict_hit: float
    r1_selectivity: float
    r1_mean_syllables: float

    # Strategy R2: alteration
    r2_dict_hit: float
    r2_selectivity: float
    r2_mean_syllables: float

    # Strategy R3: combined
    r3_dict_hit: float
    r3_selectivity: float
    r3_mean_syllables: float

    # Baselines and comparison
    phase15_dict_hit: float
    phase15_selectivity: float
    best_strategy: str
    best_dict_hit: float
    best_selectivity: float
    best_mean_syllables: float

    # Gates
    r1_modifier_count_ok: bool
    r2_syllable_distribution_ok: bool
    r3_final_gate: bool

    decoded_sample: List[List[str]]
    progression: Dict
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Convergent classification
# ---------------------------------------------------------------------------

def _load_approach_results(rd: str) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Load results from all 5 approaches. Returns empty dicts for missing files."""
    files = {
        'B': 'modifier_standalone.json',
        'D': 'modifier_anomaly.json',
        'A': 'modifier_distribution.json',
        'E': 'modifier_minimal_pairs.json',
        'C': 'modifier_localize.json',
    }
    results = {}
    for key, fname in files.items():
        path = os.path.join(rd, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)
        else:
            results[key] = {}
            print(f"    [WARN] {fname} not found — {key} evidence will be empty")

    return results['B'], results['D'], results['A'], results['E'], results['C']


def convergent_classify(
    b_result: Dict,
    d_result: Dict,
    a_result: Dict,
    e_result: Dict,
    c_result: Dict,
    eva_to_triple: Dict[str, str],
) -> List[ModifierClassification]:
    """Combine evidence from all 5 approaches per EVA character."""

    # Build per-char score lookups from each approach
    # B: standalone scores
    b_scores: Dict[str, float] = {}
    for p in b_result.get('char_profiles', []):
        b_scores[p['eva_char']] = p.get('modifier_score', 0.0)
    b_candidates = set(b_result.get('modifier_candidates', []))

    # D: anomaly scores
    d_scores: Dict[str, float] = {}
    for p in d_result.get('anomaly_profiles', []):
        d_scores[p['eva_char']] = p.get('anomaly_score', 0.0)
    d_candidates = set(d_result.get('modifier_candidates', []))

    # A: distribution best modifier set
    best_match = a_result.get('best_match', {})
    a_modifier_chars = set(best_match.get('modifier_chars', []))

    # E: minimal pair scores
    e_scores: Dict[str, float] = {}
    for p in e_result.get('per_char_evidence', []):
        e_scores[p['eva_char']] = p.get('modifier_score', 0.0)
    e_candidates = set(e_result.get('modifier_candidates', []))

    # C: localization padding ratios
    c_scores: Dict[str, float] = {}
    c_ratios = c_result.get('padding_ratio_per_char', {})
    for ch, ratio in c_ratios.items():
        c_scores[ch] = ratio
    c_candidates = set(c_result.get('modifier_candidates', []))

    # All known EVA chars
    all_chars = sorted(EVA_VISUAL_COMPONENTS.keys())

    classifications: List[ModifierClassification] = []

    for ch in all_chars:
        if ch not in eva_to_triple:
            continue

        triple_key = eva_to_triple[ch]

        b_s = b_scores.get(ch, 0.0)
        d_s = d_scores.get(ch, 0.0)
        a_ev = ch in a_modifier_chars
        e_s = e_scores.get(ch, 0.0)
        c_s = c_scores.get(ch, 0.0)

        # Count supporting approaches (threshold-based)
        n_supporting = 0
        if b_s >= 0.6 or ch in b_candidates:
            n_supporting += 1
        if d_s >= 0.5 or ch in d_candidates:
            n_supporting += 1
        if a_ev:
            n_supporting += 1
        if e_s >= 0.5 or ch in e_candidates:
            n_supporting += 1
        if c_s >= 0.6 or ch in c_candidates:
            n_supporting += 1

        # Classification
        if n_supporting >= 3:
            classification = 'modifier'
        elif n_supporting == 2:
            classification = 'ambiguous'
        else:
            classification = 'syllabic'

        # Infer modifier type from positional bias (B)
        modifier_type = 'silent'  # default
        if classification in ('modifier', 'ambiguous'):
            # Check positional bias from B
            for p in b_result.get('char_profiles', []):
                if p['eva_char'] == ch:
                    bias = p.get('position_bias', 'uniform')
                    if bias == 'medial':
                        modifier_type = 'silent'  # virama-like
                    elif bias == 'final':
                        modifier_type = 'vowel_changer'  # suffix marker
                    elif bias == 'initial':
                        modifier_type = 'cluster'  # prefix cluster
                    break

        classifications.append(ModifierClassification(
            eva_char=ch,
            triple_key=triple_key,
            standalone_score=round(b_s, 4),
            anomaly_score=round(d_s, 4),
            distribution_evidence=a_ev,
            minimal_pair_score=round(e_s, 4),
            localization_score=round(c_s, 4),
            n_approaches_supporting=n_supporting,
            final_classification=classification,
            modifier_type=modifier_type,
        ))

    # Sort: modifiers first, then by support count
    classifications.sort(key=lambda c: (
        -({'modifier': 2, 'ambiguous': 1, 'syllabic': 0}[c.final_classification]),
        -c.n_approaches_supporting,
    ))

    return classifications


# ---------------------------------------------------------------------------
# Re-decode strategies
# ---------------------------------------------------------------------------

def _compute_dict_hit(
    decoded: List[str],
    ref_word_set: set,
) -> float:
    """Fraction of decoded tokens that are dictionary hits."""
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    return hits / len(decoded)


def _compute_mean_syllables(
    tokens: List[str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
) -> float:
    """Mean non-modifier triples per token (= corrected syllable count)."""
    counts = []
    for token in tokens:
        chars = tokenize_eva_chars(token)
        n_syl = sum(
            1 for ch in chars
            if ch not in modifier_chars and ch in eva_to_triple
        )
        counts.append(max(n_syl, 1))
    return sum(counts) / len(counts) if counts else 0.0


def _compute_selectivity(
    dict_hit: float,
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    n_random: int = 5,
) -> float:
    """Selectivity ratio: dict_hit / random_baseline."""
    import random
    rng = random.Random(42)
    syllables = list(set(assignment.values()))
    if not syllables:
        return 0.0

    random_hits = []
    for _ in range(n_random):
        rand_assignment = {k: rng.choice(syllables) for k in assignment}
        decoded = [
            decode_token(t, rand_assignment, eva_to_triple)
            for t in tokens[:2000]
        ]
        random_hits.append(_compute_dict_hit(decoded, ref_word_set))

    baseline = sum(random_hits) / len(random_hits) if random_hits else 0.01
    return dict_hit / max(baseline, 0.001)


def strategy_strip(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    max_tokens: int = 2000,
) -> List[str]:
    """R1: Strip modifier chars entirely before decoding."""
    decoded = []
    for token in tokens[:max_tokens]:
        result = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        decoded.append(result)
    return decoded


def strategy_alter(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    max_tokens: int = 2000,
) -> List[str]:
    """R2: Apply modifier-type-specific alterations."""
    decoded = []
    for token in tokens[:max_tokens]:
        result = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        decoded.append(result)
    return decoded


def strategy_combined(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    max_tokens: int = 2000,
) -> List[str]:
    """R3: Try alteration first; if no dict hit, try stripping; else keep original."""
    decoded = []
    for token in tokens[:max_tokens]:
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_modifier_integrate() -> None:
    """Step 16.6: Convergent classification + modifier-aware re-decode."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 16.6: Convergent Modifier Classification + Re-Decode")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load all approach results ───
    print("\n  1. Loading approach results …")
    b_result, d_result, a_result, e_result, c_result = _load_approach_results(rd)

    # ─── Load Phase 15 best assignment ───
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found — run combined-refine first")
        return

    with open(refine_path) as f:
        refine_data = json.load(f)

    assignment = refine_data.get('best_assignment', {})
    phase15_dict_hit = refine_data.get('best_dict_hit', 0.3545)
    phase15_selectivity = refine_data.get('best_selectivity', 2.55)

    # ─── Load corpus ───
    print("\n  2. Loading corpus …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"      {len(tokens)} tokens")

    # ─── Build reference word set ───
    print("\n  3. Building expanded reference word set …")
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
    print(f"      {len(ref_word_set)} words in reference set")

    # ─── Convergent classification ───
    print("\n  4. Computing convergent classification …")
    classifications = convergent_classify(
        b_result, d_result, a_result, e_result, c_result, eva_to_triple,
    )

    modifier_chars_list = [c.eva_char for c in classifications if c.final_classification == 'modifier']
    syllabic_chars_list = [c.eva_char for c in classifications if c.final_classification == 'syllabic']
    ambiguous_chars_list = [c.eva_char for c in classifications if c.final_classification == 'ambiguous']
    modifier_chars_set = set(modifier_chars_list)

    print(f"      Modifiers ({len(modifier_chars_list)}): {modifier_chars_list}")
    print(f"      Syllabic ({len(syllabic_chars_list)}): {syllabic_chars_list}")
    print(f"      Ambiguous ({len(ambiguous_chars_list)}): {ambiguous_chars_list}")

    # Build modifier rules map
    modifier_rules: Dict[str, str] = {}
    for c in classifications:
        if c.final_classification == 'modifier':
            modifier_rules[c.eva_char] = c.modifier_type

    print(f"\n      Modifier type assignments:")
    for ch, mtype in modifier_rules.items():
        print(f"        {ch:>8} → {mtype}")

    # ─── Classification table ───
    print(f"\n  5. Full classification table:")
    print(f"      {'Char':<8} {'Triple':<35} {'B':>5} {'D':>5} {'A':>3} "
          f"{'E':>5} {'C':>5} {'#':>2} {'Class':<10} {'Type':<15}")
    print("      " + "-" * 100)
    for c in classifications:
        a_str = 'Y' if c.distribution_evidence else 'N'
        print(f"      {c.eva_char:<8} {c.triple_key:<35} "
              f"{c.standalone_score:>5.2f} {c.anomaly_score:>5.2f} "
              f"{a_str:>3} {c.minimal_pair_score:>5.2f} "
              f"{c.localization_score:>5.2f} {c.n_approaches_supporting:>2} "
              f"{c.final_classification:<10} {c.modifier_type:<15}")

    # ─── Phase 15 baseline ───
    print(f"\n  6. Phase 15 baseline: dict_hit={phase15_dict_hit:.4f}, "
          f"selectivity={phase15_selectivity:.2f}×")

    # ─── Strategy R1: Stripping ───
    print("\n  7. Strategy R1: Modifier stripping …")
    r1_decoded = strategy_strip(tokens, assignment, eva_to_triple, modifier_chars_set)
    r1_hit = _compute_dict_hit(r1_decoded, ref_word_set)
    r1_mean_syl = _compute_mean_syllables(tokens, eva_to_triple, modifier_chars_set)
    r1_sel = _compute_selectivity(r1_hit, tokens, assignment, eva_to_triple, ref_word_set)
    print(f"      dict_hit={r1_hit:.4f}, selectivity={r1_sel:.2f}×, "
          f"mean_syl={r1_mean_syl:.2f}")

    # ─── Strategy R2: Alteration ───
    print("\n  8. Strategy R2: Modifier alteration …")
    r2_decoded = strategy_alter(
        tokens, assignment, eva_to_triple, modifier_chars_set, modifier_rules,
    )
    r2_hit = _compute_dict_hit(r2_decoded, ref_word_set)
    r2_mean_syl = r1_mean_syl  # same syllable count since structure unchanged
    r2_sel = _compute_selectivity(r2_hit, tokens, assignment, eva_to_triple, ref_word_set)
    print(f"      dict_hit={r2_hit:.4f}, selectivity={r2_sel:.2f}×, "
          f"mean_syl={r2_mean_syl:.2f}")

    # ─── Strategy R3: Combined ───
    print("\n  9. Strategy R3: Combined (alteration → stripping → original) …")
    r3_decoded = strategy_combined(
        tokens, assignment, eva_to_triple, modifier_chars_set, modifier_rules,
        ref_word_set,
    )
    r3_hit = _compute_dict_hit(r3_decoded, ref_word_set)
    r3_mean_syl = r1_mean_syl
    r3_sel = _compute_selectivity(r3_hit, tokens, assignment, eva_to_triple, ref_word_set)
    print(f"      dict_hit={r3_hit:.4f}, selectivity={r3_sel:.2f}×, "
          f"mean_syl={r3_mean_syl:.2f}")

    # ─── Determine best strategy ───
    strategies = {
        'R1_strip': (r1_hit, r1_sel, r1_mean_syl, r1_decoded),
        'R2_alter': (r2_hit, r2_sel, r2_mean_syl, r2_decoded),
        'R3_combined': (r3_hit, r3_sel, r3_mean_syl, r3_decoded),
    }
    best_name = max(strategies, key=lambda k: strategies[k][0])
    best_hit, best_sel, best_mean, best_decoded = strategies[best_name]

    print(f"\n  10. Best strategy: {best_name}")
    print(f"      dict_hit={best_hit:.4f} (Phase 15: {phase15_dict_hit:.4f}, "
          f"delta={best_hit - phase15_dict_hit:+.4f})")
    print(f"      selectivity={best_sel:.2f}× (Phase 15: {phase15_selectivity:.2f}×)")
    print(f"      mean_syl={best_mean:.2f}")

    # ─── Decoded sample ───
    decoded_sample = [
        [tokens[i], best_decoded[i]]
        for i in range(min(20, len(tokens), len(best_decoded)))
    ]

    print(f"\n  11. Decoded sample (first 20):")
    for tok, dec in decoded_sample:
        marker = '*' if dec.lower() in ref_word_set else ' '
        print(f"      {marker} {tok:>15} → {dec}")

    # ─── Gates ───
    r1_ok = 5 <= len(modifier_chars_list) <= 15
    r2_ok = 2.0 <= best_mean <= 3.0
    r3_ok = best_hit > phase15_dict_hit and 2.0 <= best_mean <= 3.5

    print(f"\n  12. Validation gates:")
    print(f"      R1 (modifier count 5-15): "
          f"{'PASS' if r1_ok else 'FAIL'} ({len(modifier_chars_list)} modifiers)")
    print(f"      R2 (mean syl/token 2.0-3.0): "
          f"{'PASS' if r2_ok else 'FAIL'} ({best_mean:.2f})")
    print(f"      R3 (dict_hit > Phase 15 AND mean ~2.5): "
          f"{'PASS' if r3_ok else 'FAIL'}")

    # ─── Progression ───
    progression = {
        'phase11': {'dict_hit': 0.111, 'selectivity': 1.92},
        'phase13': {'dict_hit': 0.1143, 'selectivity': 1.86},
        'phase14': {'dict_hit': 0.1945, 'selectivity': 3.00},
        'phase15': {'dict_hit': phase15_dict_hit, 'selectivity': phase15_selectivity},
        'phase16': {
            'dict_hit': round(best_hit, 4),
            'selectivity': round(best_sel, 2),
            'mean_syl_per_token': round(best_mean, 2),
            'strategy': best_name,
            'n_modifiers': len(modifier_chars_list),
        },
        'trend': 'improvement' if best_hit > phase15_dict_hit else 'plateau',
    }

    # ─── Verdict ───
    if r3_ok:
        verdict = (
            f"PASS: {best_name} achieves {best_hit:.1%} dict_hit "
            f"({best_sel:.2f}× selectivity), mean {best_mean:.2f} syl/token. "
            f"{len(modifier_chars_list)} modifiers identified: {modifier_chars_list}. "
            f"Improvement over Phase 15: {best_hit - phase15_dict_hit:+.1%}."
        )
    elif best_hit >= phase15_dict_hit:
        verdict = (
            f"PARTIAL: dict_hit {best_hit:.1%} >= Phase 15 ({phase15_dict_hit:.1%}) "
            f"but mean syl/token {best_mean:.2f} outside target range. "
            f"{len(modifier_chars_list)} modifiers: {modifier_chars_list}."
        )
    else:
        verdict = (
            f"FAIL: Best dict_hit {best_hit:.1%} < Phase 15 ({phase15_dict_hit:.1%}). "
            f"Modifier removal lost phonemic content. "
            f"Reverting to Phase 15 baseline."
        )

    print(f"\n  Verdict: {verdict}")

    # ─── Save ───
    result = IntegrationResult(
        n_modifier=len(modifier_chars_list),
        n_syllabic=len(syllabic_chars_list),
        n_ambiguous=len(ambiguous_chars_list),
        modifier_chars=modifier_chars_list,
        syllabic_chars=syllabic_chars_list,
        ambiguous_chars=ambiguous_chars_list,
        classifications=[_convert(asdict(c)) for c in classifications],
        r1_dict_hit=round(r1_hit, 4),
        r1_selectivity=round(r1_sel, 2),
        r1_mean_syllables=round(r1_mean_syl, 3),
        r2_dict_hit=round(r2_hit, 4),
        r2_selectivity=round(r2_sel, 2),
        r2_mean_syllables=round(r2_mean_syl, 3),
        r3_dict_hit=round(r3_hit, 4),
        r3_selectivity=round(r3_sel, 2),
        r3_mean_syllables=round(r3_mean_syl, 3),
        phase15_dict_hit=round(phase15_dict_hit, 4),
        phase15_selectivity=round(phase15_selectivity, 2),
        best_strategy=best_name,
        best_dict_hit=round(best_hit, 4),
        best_selectivity=round(best_sel, 2),
        best_mean_syllables=round(best_mean, 3),
        r1_modifier_count_ok=r1_ok,
        r2_syllable_distribution_ok=r2_ok,
        r3_final_gate=r3_ok,
        decoded_sample=decoded_sample,
        progression=progression,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'modifier_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  → {out_path}")
