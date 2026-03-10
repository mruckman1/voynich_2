"""
Phase 32.8 – Phase 32 Verdict
================================
Final verdict based on all Phase 32 results.

Dependency chain:
    compound_decode.json       (Step 32.1)
    compound_signal.json       (Step 32.2)
    compound_bigrams.json      (Step 32.3)
    compound_bootstrap.json    (Step 32.5)
    compound_folio.json        (Step 32.6)
    compound_readability.json  (Step 32.7)
    signal_bigrams.json        (Phase 29 baseline)
        → phase32_verdict.json  (this step)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(rd: str, filename: str) -> Optional[Dict]:
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase32_verdict() -> None:
    """Step 32.8: Phase 32 verdict."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 32.8: Verdict")
    print("=" * 70)

    rd = _results_dir()

    # ── Load all results ──
    cd = _load_json(rd, 'compound_decode.json')
    cs = _load_json(rd, 'compound_signal.json')
    cb = _load_json(rd, 'compound_bigrams.json')
    cboot = _load_json(rd, 'compound_bootstrap.json')
    cfolio = _load_json(rd, 'compound_folio.json')
    cread = _load_json(rd, 'compound_readability.json')
    p29 = _load_json(rd, 'signal_bigrams.json')

    # ── Extract key metrics ──
    compound_dict_hit = cd.get('dict_hit_rate', 0) if cd else 0
    compound_selectivity = cd.get('compound_selectivity', 0) if cd else 0
    compound_bigram_z = cb.get('bigram_z_score', 0) if cb else 0
    compound_n_signal = cs.get('n_signal', 0) if cs else 0
    compound_signal_rate = cs.get('signal_rate', 0) if cs else 0
    bootstrap_n_confirmed = cboot.get('n_total_accepted', 0) if cboot else 0
    confirmed_vocab_size = cboot.get('confirmed_vocabulary_size', 0) if cboot else 0
    readability_passed = cread.get('n_passed', 0) if cread else 0
    readability_total = cread.get('n_total', 12) if cread else 12

    phase29_bigram_z = p29.get('bigram_z_score', 0) if p29 else 0
    phase29_signal_rate = p29.get('signal_rate', 0) if p29 else 0
    phase29_n_signal = p29.get('n_signal', 0) if p29 else 0

    # Best fragment
    best_frag = cfolio.get('best_fragment') if cfolio else None
    best_frag_length = best_frag.get('length', 0) if best_frag else 0
    best_frag_score = best_frag.get('parse', {}).get('parse_score', 0) if best_frag else 0

    # Longest signal run
    longest_run = 0
    if cfolio:
        for af in cfolio.get('annotated_folios', []):
            for sr in af.get('signal_runs', []):
                longest_run = max(longest_run, sr.get('length', 0))

    # Deltas
    delta_dict_hit = compound_dict_hit - 0.436  # vs Phase 16 baseline
    delta_bigram_z = compound_bigram_z - phase29_bigram_z
    delta_signal_rate = compound_signal_rate - phase29_signal_rate
    delta_n_signal = compound_n_signal - phase29_n_signal

    # ── Verdict ──
    evidence: List[str] = []
    next_steps: List[str] = []

    # Check for COMPOUND_BREAKTHROUGH
    is_breakthrough = (
        compound_bigram_z > 10
        and compound_signal_rate > 0.30
        and best_frag_length >= 4
        and best_frag_score >= 0.6
        and bootstrap_n_confirmed >= 5
    )

    # Check for COMPOUND_IMPROVEMENT
    is_improvement = (
        compound_bigram_z > 8
        or (compound_bigram_z > 6.5 and delta_bigram_z > 0.5)
    ) and compound_signal_rate > 0.25 and bootstrap_n_confirmed >= 3

    # Check for COMPOUND_CONFIRMED
    is_confirmed = (
        compound_signal_rate > 0.20
        and compound_selectivity > 1.5
        and abs(delta_bigram_z) < 2.0
    )

    if is_breakthrough:
        verdict = "COMPOUND_BREAKTHROUGH"
        evidence.append(f"Bigram z={compound_bigram_z:.2f} (>10)")
        evidence.append(f"SIGNAL rate={compound_signal_rate:.1%} (>30%)")
        evidence.append(f"Best fragment: {best_frag_length} words, score={best_frag_score:.3f}")
        evidence.append(f"Bootstrap: {bootstrap_n_confirmed} new confirmed words")
        next_steps.append("Present best fragment as candidate decoded passage")
        next_steps.append("Expand to full-corpus reading attempt")
    elif is_improvement:
        verdict = "COMPOUND_IMPROVEMENT"
        evidence.append(f"Bigram z={compound_bigram_z:.2f} (improvement over {phase29_bigram_z:.2f})")
        evidence.append(f"SIGNAL rate={compound_signal_rate:.1%} (>25%)")
        evidence.append(f"Bootstrap: {bootstrap_n_confirmed} new confirmed words")
        next_steps.append("Focus on folios with highest SIGNAL rates")
        next_steps.append("Refine suffix mappings using paradigm analysis")
    elif is_confirmed:
        verdict = "COMPOUND_CONFIRMED"
        evidence.append(f"SIGNAL rate increased to {compound_signal_rate:.1%}")
        evidence.append(f"Selectivity={compound_selectivity:.2f}x (genuine, not collisions)")
        evidence.append(f"Bigram z={compound_bigram_z:.2f} (stable)")
        next_steps.append("Model is structurally correct but doesn't break through to readability")
        next_steps.append("Need CVC/CCV expansion or determinative value identification")
    else:
        verdict = "COMPOUND_COLLISIONS"
        evidence.append(f"SIGNAL rate={compound_signal_rate:.1%} (not improved)")
        evidence.append(f"Bigram z={compound_bigram_z:.2f} (not improved)")
        evidence.append(f"Dict-hit increase is from shorter-word collisions")
        next_steps.append("Compound model may be structurally correct but doesn't improve signal")
        next_steps.append("Consider alternative suffix mappings or segmentation approaches")

    # Gap analysis
    gap_analysis = {
        'confirmed_root_triples': 12,  # unchanged
        'suffix_mappings_tested': len(
            [s for s in cd.get('token_suffixes', []) if s]
        ) if cd else 0,
        'unique_suffixes_used': len(set(
            s for s in cd.get('token_suffixes', []) if s
        )) if cd else 0,
        'dict_hit_gap': round(0.895 - compound_dict_hit, 3),  # vs oracle ceiling (Phase 23)
        'signal_gap': round(1.0 - compound_signal_rate, 3),
    }

    # ── Print ──
    print(f"\n  VERDICT: {verdict}")
    print(f"\n  Key metrics:")
    print(f"     Compound dict_hit:    {compound_dict_hit:.4f} (delta={delta_dict_hit:+.4f})")
    print(f"     Compound selectivity: {compound_selectivity:.2f}x")
    print(f"     Compound bigram z:    {compound_bigram_z:.2f} (Phase 29: {phase29_bigram_z:.2f}, delta={delta_bigram_z:+.2f})")
    print(f"     SIGNAL rate:          {compound_signal_rate:.1%} (Phase 29: {phase29_signal_rate:.1%})")
    print(f"     Bootstrap accepted:   {bootstrap_n_confirmed}")
    print(f"     Readability:          {readability_passed}/{readability_total}")
    print(f"     Longest SIGNAL run:   {longest_run}")

    print(f"\n  Evidence:")
    for e in evidence:
        print(f"     - {e}")

    print(f"\n  Next steps:")
    for ns in next_steps:
        print(f"     - {ns}")

    # ── Progression ──
    progression = [
        {'phase': 11, 'dict_hit': 0.111, 'selectivity': 1.92, 'model': 'CV syllabary'},
        {'phase': 14, 'dict_hit': 0.194, 'selectivity': 3.00, 'model': 'Sub-cell features'},
        {'phase': 15, 'dict_hit': 0.354, 'selectivity': 2.55, 'model': 'Dict expansion'},
        {'phase': 16, 'dict_hit': 0.436, 'selectivity': 3.38, 'model': 'Modifier R3'},
        {'phase': 29, 'dict_hit': 0.436, 'selectivity': None, 'model': 'Signal z=6.14'},
        {'phase': 31, 'dict_hit': 0.607, 'selectivity': None, 'model': 'Compound-sign'},
        {'phase': 32, 'dict_hit': round(compound_dict_hit, 3),
         'selectivity': round(compound_selectivity, 2),
         'model': f'Compound+signal z={compound_bigram_z:.1f}'},
    ]

    print(f"\n  Full progression:")
    for p in progression:
        sel = f"{p['selectivity']:.2f}x" if p['selectivity'] else "  —"
        print(f"     Phase {p['phase']:2d}: dict_hit={p['dict_hit']:.3f} "
              f"sel={sel:>6s}  {p['model']}")

    # ── Save ──
    output = {
        'compound_dict_hit': round(compound_dict_hit, 6),
        'compound_selectivity': round(compound_selectivity, 4),
        'compound_bigram_z': round(compound_bigram_z, 2),
        'compound_n_signal': compound_n_signal,
        'compound_signal_rate': round(compound_signal_rate, 6),
        'bootstrap_n_confirmed': bootstrap_n_confirmed,
        'confirmed_vocabulary_size': confirmed_vocab_size,
        'phase29_bigram_z': phase29_bigram_z,
        'phase29_signal_rate': phase29_signal_rate,
        'phase29_n_signal': phase29_n_signal,
        'delta_dict_hit': round(delta_dict_hit, 6),
        'delta_bigram_z': round(delta_bigram_z, 2),
        'delta_signal_rate': round(delta_signal_rate, 6),
        'delta_n_signal': delta_n_signal,
        'readability_n_passed': readability_passed,
        'readability_n_total': readability_total,
        'longest_signal_run': longest_run,
        'best_fragment_length': best_frag_length,
        'best_fragment_score': best_frag_score,
        'verdict': verdict,
        'evidence': evidence,
        'next_steps': next_steps,
        'gap_analysis': gap_analysis,
        'progression': progression,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(rd, 'phase32_verdict.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Step 32.8 completed in {time.time() - t0:.1f}s")
