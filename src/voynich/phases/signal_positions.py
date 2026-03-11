"""
Step 43.6 -- Signal Word Position Mapping
=========================================
Map all positions of the 8 bedrock signal words across the entire corpus
with precise positional metadata.

Dependency chain:
    results/signal_word_revalidate.json  (Phase 42.3: bedrock signal words)
    results/combined_refine.json         (Phase 15: 25-triple table)
    results/modifier_integrate.json      (Phase 16: modifier handling)
    data/corpus/                         (EVA transcription)
        -> signal_positions.json          (this step)
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
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEDROCK_WORDS = ['bene', 'codi', 'sero', 'sene', 'de', 'raro', 'dine', 'cola']


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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SignalOccurrence:
    word: str
    folio: str
    section: str
    token_index_in_folio: int
    total_tokens_in_folio: int
    is_line_initial: bool
    relative_position: float
    preceding_decoded: str
    following_decoded: str


@dataclass
class SignalPositionResult:
    n_signal_words: int
    signal_words: List[str]
    n_total_occurrences: int
    per_word_summary: Dict[str, Dict]
    per_section_summary: Dict[str, Dict]
    inter_signal_distances: Dict[str, List[int]]
    mean_inter_distance: float
    folio_heat_map: Dict[str, Dict[str, int]]
    uniformity_tests: Dict[str, Dict]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------

def _chi_squared_test(
    observed: List[int],
    expected: List[float],
) -> Tuple[float, float]:
    """Compute chi-squared statistic and p-value.

    Uses scipy if available, otherwise computes the statistic manually
    and approximates the p-value via the regularised upper incomplete
    gamma function.
    """
    # Filter out categories with zero expected
    obs_filt = []
    exp_filt = []
    for o, e in zip(observed, expected):
        if e > 0:
            obs_filt.append(o)
            exp_filt.append(e)

    if len(obs_filt) < 2:
        return 0.0, 1.0

    chi2 = sum((o - e) ** 2 / e for o, e in zip(obs_filt, exp_filt))
    df = len(obs_filt) - 1

    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = 1.0 - chi2_dist.cdf(chi2, df)
    except ImportError:
        # Manual approximation using the Wilson-Hilferty normal approximation
        if df <= 0:
            p_value = 1.0
        else:
            # Wilson-Hilferty transformation
            z = ((chi2 / df) ** (1.0 / 3.0)
                 - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
            # Standard normal survival function approximation
            if z > 6.0:
                p_value = 0.0
            elif z < -6.0:
                p_value = 1.0
            else:
                # Use error function approximation
                p_value = 0.5 * math.erfc(z / math.sqrt(2.0))

    return chi2, p_value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_signal_positions() -> None:
    """Step 43.6: Signal Word Position Mapping."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.6: Signal Word Position Mapping")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

    # Phase 15 assignment
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    if not refine_data:
        print("  [SKIP] combined_refine.json not found")
        return
    assignment = refine_data.get('best_assignment', {})

    # Phase 16 modifiers
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    if not mod_data:
        print("  [SKIP] modifier_integrate.json not found")
        return
    modifier_chars = mod_data.get('modifier_chars', [])
    modifier_set = set(modifier_chars)

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_set)} chars")
    print(f"     Bedrock words: {BEDROCK_WORDS}")

    # ── 2. Load corpus and decode ──
    print("\n  2. Loading corpus and decoding tokens ...")

    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()
    bedrock_set = set(BEDROCK_WORDS)

    # Collect all occurrences
    occurrences: List[SignalOccurrence] = []

    # We need per-folio decoded token lists for context
    folio_data: List[Tuple[str, str, List[str], List[str]]] = []
    # (folio_id, section, raw_tokens, decoded_tokens)

    for folio_id, page in corpus.pages.items():
        tokens = page.all_tokens
        section = page.section
        n_tokens = len(tokens)

        # Decode all tokens on this folio
        decoded_tokens = []
        for token in tokens:
            decoded = decode_token_modifier_aware(
                token, assignment, eva_to_triple, modifier_set,
            )
            decoded_tokens.append(decoded.lower())

        folio_data.append((folio_id, section, tokens, decoded_tokens))

        # Find bedrock word occurrences
        for i, dec_word in enumerate(decoded_tokens):
            if dec_word in bedrock_set:
                # Preceding decoded word
                preceding = decoded_tokens[i - 1] if i > 0 else ''
                # Following decoded word
                following = decoded_tokens[i + 1] if i < n_tokens - 1 else ''
                # Relative position
                rel_pos = i / n_tokens if n_tokens > 0 else 0.0

                occurrences.append(SignalOccurrence(
                    word=dec_word,
                    folio=folio_id,
                    section=section,
                    token_index_in_folio=i,
                    total_tokens_in_folio=n_tokens,
                    is_line_initial=(i == 0),
                    relative_position=round(rel_pos, 4),
                    preceding_decoded=preceding,
                    following_decoded=following,
                ))

    print(f"     Total occurrences of bedrock words: {len(occurrences)}")

    # ── 3. Per-word summary ──
    print("\n  3. Building per-word summary ...")

    per_word_summary: Dict[str, Dict] = {}
    for word in BEDROCK_WORDS:
        word_occs = [o for o in occurrences if o.word == word]
        count = len(word_occs)
        folios_seen = set(o.folio for o in word_occs)
        n_folios = len(folios_seen)

        # Sections breakdown
        section_counts: Dict[str, int] = Counter(o.section for o in word_occs)

        # Relative position stats
        positions = [o.relative_position for o in word_occs]
        if positions:
            mean_pos = sum(positions) / len(positions)
            var_pos = sum((p - mean_pos) ** 2 for p in positions) / len(positions)
            std_pos = var_pos ** 0.5
        else:
            mean_pos = 0.0
            std_pos = 0.0

        per_word_summary[word] = {
            'count': count,
            'n_folios': n_folios,
            'sections': dict(section_counts),
            'mean_relative_pos': round(mean_pos, 4),
            'std_relative_pos': round(std_pos, 4),
        }

        print(f"     {word:8s}: {count:4d} occurrences across {n_folios} folios, "
              f"mean_pos={mean_pos:.3f}, std_pos={std_pos:.3f}")

    # ── 4. Per-section summary ──
    print("\n  4. Building per-section summary ...")

    per_section_summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for occ in occurrences:
        per_section_summary[occ.section][occ.word] += 1

    # Convert to regular dicts
    per_section_summary = {
        section: dict(word_counts)
        for section, word_counts in sorted(per_section_summary.items())
    }

    for section, word_counts in per_section_summary.items():
        total = sum(word_counts.values())
        print(f"     {section:15s}: {total:4d} total  "
              f"({', '.join(f'{w}={c}' for w, c in sorted(word_counts.items()))})")

    # ── 5. Folio heat map ──
    print("\n  5. Building folio heat map ...")

    folio_heat_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for occ in occurrences:
        folio_heat_map[occ.folio][occ.word] += 1

    # Convert to regular dicts
    folio_heat_map = {
        folio: dict(word_counts)
        for folio, word_counts in sorted(folio_heat_map.items())
    }

    # Show top folios by total count
    folio_totals = {
        folio: sum(wc.values())
        for folio, wc in folio_heat_map.items()
    }
    top_folios = sorted(folio_totals.items(), key=lambda x: -x[1])[:15]
    for folio, total in top_folios:
        words_str = ', '.join(
            f'{w}={c}' for w, c in sorted(folio_heat_map[folio].items())
        )
        print(f"     {folio:8s}: {total:3d} ({words_str})")

    # ── 6. Inter-signal distances ──
    print("\n  6. Computing inter-signal distances ...")

    inter_signal_distances: Dict[str, List[int]] = defaultdict(list)
    all_distances: List[int] = []

    for folio_id, section, tokens, decoded_tokens in folio_data:
        # Collect positions of bedrock word occurrences on this folio
        signal_positions: List[Tuple[int, str]] = []
        for i, dec_word in enumerate(decoded_tokens):
            if dec_word in bedrock_set:
                signal_positions.append((i, dec_word))

        # Compute distances between consecutive signal occurrences
        for j in range(len(signal_positions) - 1):
            pos1, word1 = signal_positions[j]
            pos2, word2 = signal_positions[j + 1]
            distance = pos2 - pos1
            pair_key = f"{word1}->{word2}"
            inter_signal_distances[pair_key].append(distance)
            all_distances.append(distance)

    # Convert to regular dict
    inter_signal_distances = dict(inter_signal_distances)

    mean_inter_distance = (
        sum(all_distances) / len(all_distances)
        if all_distances else 0.0
    )

    print(f"     {len(all_distances)} inter-signal gaps measured")
    print(f"     Mean inter-signal distance: {mean_inter_distance:.2f} tokens")

    # Show top pair types
    pair_counts = {
        pair: len(dists)
        for pair, dists in sorted(
            inter_signal_distances.items(),
            key=lambda x: -len(x[1]),
        )
    }
    for pair, count in list(pair_counts.items())[:10]:
        dists = inter_signal_distances[pair]
        mean_d = sum(dists) / len(dists) if dists else 0.0
        print(f"     {pair:20s}: {count:4d} gaps, mean distance={mean_d:.1f}")

    # ── 7. Uniformity tests ──
    print("\n  7. Testing uniformity across sections ...")

    # Count tokens per section for expected proportions
    section_token_counts: Dict[str, int] = Counter()
    for folio_id, page in corpus.pages.items():
        section = page.section
        section_token_counts[section] += len(page.all_tokens)

    total_tokens = sum(section_token_counts.values())
    sections_ordered = sorted(section_token_counts.keys())

    uniformity_tests: Dict[str, Dict] = {}

    for word in BEDROCK_WORDS:
        word_occs = [o for o in occurrences if o.word == word]
        word_count = len(word_occs)

        if word_count < 5:
            # Too few occurrences for a meaningful test
            uniformity_tests[word] = {
                'chi2': 0.0,
                'p_value': 1.0,
                'uniform': True,
                'note': f'too few occurrences ({word_count})',
            }
            continue

        # Observed counts per section
        obs_by_section = Counter(o.section for o in word_occs)
        observed = [obs_by_section.get(s, 0) for s in sections_ordered]

        # Expected counts proportional to section size
        expected = [
            word_count * section_token_counts[s] / total_tokens
            for s in sections_ordered
        ]

        chi2, p_value = _chi_squared_test(observed, expected)
        is_uniform = p_value > 0.05

        uniformity_tests[word] = {
            'chi2': round(chi2, 4),
            'p_value': round(p_value, 6),
            'uniform': is_uniform,
        }

        tag = 'UNIFORM' if is_uniform else 'NON-UNIFORM'
        print(f"     {word:8s}: chi2={chi2:8.2f}, p={p_value:.4f} -> {tag}")

    # ── 8. Summary ──
    print("\n  " + "=" * 66)
    print(f"  Total bedrock signal words: {len(BEDROCK_WORDS)}")
    print(f"  Total occurrences: {len(occurrences)}")
    print(f"  Mean inter-signal distance: {mean_inter_distance:.2f} tokens")
    n_uniform = sum(1 for v in uniformity_tests.values() if v.get('uniform'))
    n_nonuniform = len(uniformity_tests) - n_uniform
    print(f"  Uniformity: {n_uniform} uniform, {n_nonuniform} non-uniform")

    # ── 9. Save ──
    result = SignalPositionResult(
        n_signal_words=len(BEDROCK_WORDS),
        signal_words=list(BEDROCK_WORDS),
        n_total_occurrences=len(occurrences),
        per_word_summary=per_word_summary,
        per_section_summary=per_section_summary,
        inter_signal_distances=inter_signal_distances,
        mean_inter_distance=round(mean_inter_distance, 4),
        folio_heat_map=folio_heat_map,
        uniformity_tests=uniformity_tests,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'signal_positions.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
