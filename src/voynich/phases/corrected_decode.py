"""
Step 24.6 – Full Corpus Decode with Corrected Table
=====================================================
Decodes the entire Voynich corpus using the corrected assignment table
from Step 24.5 (or falls back to Phase 16 if bigram filter didn't pass).
Reports corpus-wide and per-section dict-hit rates, selectivity, and
a decoded sample.

Dependency chain:
    corrected_table.json (Step 24.5)  — or —  combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → corrected_decode.json (this step)
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
# R3 combined decode
# ---------------------------------------------------------------------------

def _decode_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode corpus using R3 combined strategy (alter -> strip -> original)."""
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


# ---------------------------------------------------------------------------
# Selectivity computation
# ---------------------------------------------------------------------------

def _compute_selectivity(
    dict_hit: float,
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    ref_word_set: set,
    n_random: int = 5,
) -> float:
    """Selectivity ratio: dict_hit / random_baseline."""
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
        hits = sum(1 for w in decoded if w.lower() in ref_word_set)
        random_hits.append(hits / len(decoded) if decoded else 0.0)

    baseline = sum(random_hits) / len(random_hits) if random_hits else 0.01
    return dict_hit / max(baseline, 0.001)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorrectedDecodeResult:
    timestamp: str
    table_source: str                   # "corrected" or "phase16"
    n_tokens_decoded: int
    dict_hit_rate: float
    selectivity: float
    n_unique_decoded: int
    n_dict_hits: int
    decoded_sample: List[List[str]]     # [[eva_token, decoded], ...]
    per_section_hits: Dict[str, float]
    phase16_comparison: float           # Phase 16 dict_hit for reference
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_corrected_decode() -> None:
    """Step 24.6: Full corpus decode with corrected table."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.6: Full Corpus Decode with Corrected Table")
    print("=" * 70)

    rdir = _results_dir()

    # ─── 1. Load assignment table ───
    print("\n  1. Loading assignment table …")
    t1 = time.time()

    table_source = "corrected"
    corrected_data = _load_json(str(rdir / "corrected_table.json"))

    if corrected_data is not None and corrected_data.get("bigram_filter_passed", False):
        assignment = corrected_data.get("final_assignment", {})
        print(f"     Using corrected table ({len(assignment)} triples)")
    else:
        # Fall back to Phase 16 table
        table_source = "phase16"
        combined_data = _load_json(str(rdir / "combined_refine.json"))
        if combined_data is None:
            print("  [SKIP] Neither corrected_table.json nor combined_refine.json found")
            return
        assignment = combined_data.get("best_assignment", {})
        reason = "not found" if corrected_data is None else "bigram filter not passed"
        print(f"     Corrected table {reason} — falling back to Phase 16 ({len(assignment)} triples)")

    print(f"     Table source: {table_source} ({time.time() - t1:.1f}s)")

    # ─── 2. Load modifiers ───
    print("\n  2. Loading modifier info …")
    t2 = time.time()

    mod_data = _load_json(str(rdir / "modifier_integrate.json")) or {}
    modifier_chars = set(mod_data.get("modifier_chars", []))

    modifier_rules: Dict[str, str] = {}
    for cls in mod_data.get("classifications", []):
        if cls.get("final_classification") == "modifier":
            modifier_rules[cls["eva_char"]] = cls.get("modifier_type", "silent")

    print(f"     {len(modifier_chars)} modifier chars, {len(modifier_rules)} rules ({time.time() - t2:.1f}s)")

    # ─── 3. Load corpus ───
    print("\n  3. Loading corpus …")
    t3 = time.time()

    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"     {len(tokens)} tokens ({time.time() - t3:.1f}s)")

    # ─── 4. Build reference word set ───
    print("\n  4. Building expanded reference word set …")
    t4 = time.time()

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
    print(f"     {len(ref_word_set)} words in reference set ({time.time() - t4:.1f}s)")

    # ─── 5. Decode ALL tokens with R3 combined strategy ───
    print("\n  5. Decoding full corpus (R3 combined) …")
    t5 = time.time()

    decoded = _decode_r3(
        tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )

    n_dict_hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    dict_hit_rate = n_dict_hits / len(decoded) if decoded else 0.0
    n_unique = len(set(w.lower() for w in decoded))

    print(f"     {len(decoded)} tokens decoded")
    print(f"     Dict hits: {n_dict_hits}/{len(decoded)} = {dict_hit_rate:.1%}")
    print(f"     Unique decoded forms: {n_unique}")
    print(f"     ({time.time() - t5:.1f}s)")

    # ─── 6. Decoded sample (first 50 tokens) ───
    print("\n  6. Recording decoded sample (first 50 tokens) …")
    t6 = time.time()

    decoded_sample = [
        [tokens[i], decoded[i]]
        for i in range(min(50, len(decoded)))
    ]

    for i, (eva, dec) in enumerate(decoded_sample[:10]):
        hit = "*" if dec.lower() in ref_word_set else " "
        print(f"     {i+1:3d}. {eva:>15} → {dec:<15} {hit}")
    if len(decoded_sample) > 10:
        print(f"     ... ({len(decoded_sample) - 10} more in output)")

    print(f"     ({time.time() - t6:.1f}s)")

    # ─── 7. Per-section dict-hit rates ───
    print("\n  7. Computing per-section dict-hit rates …")
    t7 = time.time()

    sections = ['herbal_a', 'astronomical', 'biological', 'cosmological',
                'herbal_b', 'pharmaceutical', 'recipes']

    per_section_hits: Dict[str, float] = {}
    for section in sections:
        section_tokens = corpus.get_tokens(section=section)
        if not section_tokens:
            per_section_hits[section] = 0.0
            continue

        section_decoded = _decode_r3(
            section_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )
        section_hits = sum(1 for w in section_decoded if w.lower() in ref_word_set)
        section_rate = section_hits / len(section_decoded) if section_decoded else 0.0
        per_section_hits[section] = round(section_rate, 4)
        print(f"     {section:<20} {section_hits:>5}/{len(section_decoded):<5} = {section_rate:.1%}")

    print(f"     ({time.time() - t7:.1f}s)")

    # ─── 8. Selectivity ───
    print("\n  8. Computing selectivity (5 random baselines) …")
    t8 = time.time()

    selectivity = _compute_selectivity(
        dict_hit_rate, tokens, assignment, eva_to_triple, ref_word_set,
        n_random=5,
    )
    print(f"     Selectivity: {selectivity:.2f}x ({time.time() - t8:.1f}s)")

    # ─── Phase 16 comparison ───
    combined_data = _load_json(str(rdir / "combined_refine.json")) or {}
    phase16_dict_hit = combined_data.get("best_dict_hit", 0.0)
    # Use modifier_integrate if available for the R3 result
    mod_result = _load_json(str(rdir / "modifier_integrate.json")) or {}
    if mod_result.get("r3_dict_hit"):
        phase16_dict_hit = mod_result["r3_dict_hit"]

    elapsed = time.time() - t0

    # ─── Build result ───
    result = CorrectedDecodeResult(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        table_source=table_source,
        n_tokens_decoded=len(decoded),
        dict_hit_rate=round(dict_hit_rate, 4),
        selectivity=round(selectivity, 2),
        n_unique_decoded=n_unique,
        n_dict_hits=n_dict_hits,
        decoded_sample=decoded_sample,
        per_section_hits=per_section_hits,
        phase16_comparison=round(phase16_dict_hit, 4),
        runtime_seconds=round(elapsed, 2),
    )

    out_path = rdir / "corrected_decode.json"
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2, ensure_ascii=False)

    print(f"\n  SUMMARY")
    print(f"  {'='*50}")
    print(f"  Table source:    {table_source}")
    print(f"  Dict-hit rate:   {dict_hit_rate:.1%}")
    print(f"  Selectivity:     {selectivity:.2f}x")
    print(f"  Phase 16 ref:    {phase16_dict_hit:.1%}")
    delta = dict_hit_rate - phase16_dict_hit
    print(f"  Delta vs Ph16:   {delta:+.1%}")
    print(f"  → {out_path} ({elapsed:.1f}s)")
