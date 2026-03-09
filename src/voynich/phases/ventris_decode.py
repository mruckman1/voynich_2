"""
Phase 28.7 – Ventris Corpus Decode
=====================================
Decodes the full Voynich corpus using the Ventris tiered table
(from Step 28.6) with modifier handling, and compares dict_hit
to Phase 16 baseline.

Dependency chain:
    ventris_table.json        (Step 28.6)
    modifier_integrate.json   (Phase 16 modifiers)
        → ventris_decode.json   (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    _infer_section,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import _reconstruct_modifier_rules


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
class SectionStats:
    section: str
    n_tokens: int
    n_hits: int
    dict_hit: float


@dataclass
class VentrisDecodeResult:
    corpus_n_tokens: int
    corpus_dict_hit: float
    corpus_dict_hit_base: float
    phase16_baseline: float
    improvement_vs_phase16: float
    section_stats: List[Dict]
    longest_consecutive: int
    best_passage_folio: str
    best_passage_text: str
    sample_decoded_herbal: List[List[str]]
    sample_decoded_pharma: List[List[str]]
    n_tier1_triples_active: int
    n_tier2_triples_active: int
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ventris_decode() -> None:
    """Step 28.7: Decode corpus with Ventris tiered table."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 28.7: Ventris Corpus Decode")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    vtab_path = os.path.join(rd, 'ventris_table.json')
    if not os.path.exists(vtab_path):
        print("  [SKIP] ventris_table.json not found — run ventris-tab first")
        return
    with open(vtab_path) as f:
        vtab_data = json.load(f)
    assignment = vtab_data.get('merged_assignment', {})

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    # Phase 16 R3 was computed on max_tokens=2000 (subsample).
    # We decode the full corpus, so use Phase 16's subsample rate for reference
    # but also note it's not directly comparable.
    phase16_subsample_hit = mod_data.get('r3_dict_hit', mod_data.get('best_dict_hit', 0.0))

    # Count active tiers
    n_t1 = vtab_data.get('n_tier1', 0)
    n_t2 = vtab_data.get('n_tier2', 0)

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")
    print(f"     Phase 16 R3 (2000-token subsample): {phase16_subsample_hit:.3f}")

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} expanded, {len(base_words)} base words")

    # ── 3. Decode corpus ──
    print("\n  3. Decoding full corpus …")
    corpus = load_corpus(verbose=False)

    decoded_tokens: List[str] = []
    hit_flags: List[bool] = []
    section_labels: List[str] = []
    folio_labels: List[str] = []
    eva_tokens: List[str] = []

    for folio, page in corpus.pages.items():
        section = _infer_section(folio)
        for token in page.all_tokens:
            eva_tokens.append(token)
            folio_labels.append(folio)
            section_labels.append(section)

            # R3 strategy
            alt = decode_token_modifier_aware(
                token, assignment, eva_to_triple,
                modifier_chars, modifier_rules,
            )
            if alt.lower() in ref_word_set:
                decoded_tokens.append(alt.lower())
                hit_flags.append(True)
                continue

            stripped = decode_token_modifier_aware(
                token, assignment, eva_to_triple, modifier_chars,
            )
            if stripped.lower() in ref_word_set:
                decoded_tokens.append(stripped.lower())
                hit_flags.append(True)
                continue

            raw = decode_token(token, assignment, eva_to_triple)
            decoded_tokens.append(raw.lower())
            hit_flags.append(raw.lower() in ref_word_set)

    n_tokens = len(decoded_tokens)
    n_hits = sum(hit_flags)
    dict_hit = n_hits / n_tokens if n_tokens > 0 else 0.0

    # Base dict hit (original 17K)
    n_base_hits = sum(1 for w in decoded_tokens if w in base_words)
    dict_hit_base = n_base_hits / n_tokens if n_tokens > 0 else 0.0

    # Compute Phase 16 baseline on same full corpus for fair comparison
    # (Phase 16's reported 51.6% was on 2000-token subsample)
    phase16_path = os.path.join(rd, 'combined_refine.json')
    with open(phase16_path) as f:
        p16_data = json.load(f)
    phase16_assignment = p16_data.get('best_assignment', {})

    # If Ventris table is unchanged (0 corrections), baseline = current
    n_changed = vtab_data.get('n_changed_vs_phase16', 0)
    if n_changed == 0:
        phase16_full_hit = dict_hit  # same table → same result
    else:
        # Recompute Phase 16 on same token set
        p16_hits = 0
        for i, token in enumerate(eva_tokens):
            alt = decode_token_modifier_aware(
                token, phase16_assignment, eva_to_triple,
                modifier_chars, modifier_rules,
            )
            if alt.lower() in ref_word_set:
                p16_hits += 1
                continue
            stripped = decode_token_modifier_aware(
                token, phase16_assignment, eva_to_triple, modifier_chars,
            )
            if stripped.lower() in ref_word_set:
                p16_hits += 1
                continue
            raw = decode_token(token, phase16_assignment, eva_to_triple)
            if raw.lower() in ref_word_set:
                p16_hits += 1
        phase16_full_hit = p16_hits / n_tokens if n_tokens > 0 else 0.0

    improvement = dict_hit - phase16_full_hit

    print(f"     {n_tokens} tokens, {n_hits} hits")
    print(f"     dict_hit (expanded): {dict_hit:.4f}")
    print(f"     dict_hit (base):     {dict_hit_base:.4f}")
    print(f"     Phase 16 (subsample): {phase16_subsample_hit:.4f} (2000 tokens)")
    print(f"     Phase 16 (full):      {phase16_full_hit:.4f} (same {n_tokens} tokens)")
    print(f"     Improvement:          {improvement:+.4f}")

    # ── 4. Section stats ──
    print("\n  4. Per-section breakdown …")
    section_counter: Dict[str, List[bool]] = {}
    for sec, hit in zip(section_labels, hit_flags):
        section_counter.setdefault(sec, []).append(hit)

    section_stats: List[SectionStats] = []
    for sec in sorted(section_counter.keys()):
        hits_list = section_counter[sec]
        ss = SectionStats(
            section=sec,
            n_tokens=len(hits_list),
            n_hits=sum(hits_list),
            dict_hit=round(sum(hits_list) / len(hits_list), 4),
        )
        section_stats.append(ss)
        print(f"     {sec:20s}  {ss.n_hits:5d}/{ss.n_tokens:5d}  "
              f"({ss.dict_hit:.1%})")

    # ── 5. Longest consecutive hit run ──
    print("\n  5. Longest consecutive hit run …")
    max_run = 0
    run = 0
    best_run_end = 0
    for i, hit in enumerate(hit_flags):
        if hit:
            run += 1
            if run > max_run:
                max_run = run
                best_run_end = i
        else:
            run = 0

    best_run_start = best_run_end - max_run + 1
    best_folio = folio_labels[best_run_start] if best_run_start < len(folio_labels) else ''
    passage = decoded_tokens[best_run_start:best_run_end + 1]
    print(f"     {max_run} consecutive hits on {best_folio}")
    print(f"     → {' '.join(passage[:15])}")

    # ── 6. Sample decoded text ──
    sample_herbal: List[List[str]] = []
    sample_pharma: List[List[str]] = []
    for i in range(len(decoded_tokens)):
        if section_labels[i] == 'herbal_a' and len(sample_herbal) < 10:
            sample_herbal.append([eva_tokens[i], decoded_tokens[i]])
        elif section_labels[i] == 'pharmaceutical' and len(sample_pharma) < 10:
            sample_pharma.append([eva_tokens[i], decoded_tokens[i]])

    # ── 7. Gate and verdict ──
    gate_passed = dict_hit >= phase16_full_hit - 0.02
    verdict = (
        f"PASS: dict_hit={dict_hit:.4f} (Δ={improvement:+.4f} vs Phase 16). "
        f"Longest run={max_run} on {best_folio}."
        if gate_passed
        else f"FAIL: dict_hit={dict_hit:.4f} regressed by "
             f"{-improvement:.4f} vs Phase 16 ({phase16_full_hit:.4f})"
    )
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 8. Save ──
    result = VentrisDecodeResult(
        corpus_n_tokens=n_tokens,
        corpus_dict_hit=round(dict_hit, 4),
        corpus_dict_hit_base=round(dict_hit_base, 4),
        phase16_baseline=round(phase16_full_hit, 4),
        improvement_vs_phase16=round(improvement, 4),
        section_stats=[_convert(asdict(s)) for s in section_stats],
        longest_consecutive=max_run,
        best_passage_folio=best_folio,
        best_passage_text=' '.join(passage[:20]),
        sample_decoded_herbal=sample_herbal,
        sample_decoded_pharma=sample_pharma,
        n_tier1_triples_active=n_t1,
        n_tier2_triples_active=n_t2,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'ventris_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
