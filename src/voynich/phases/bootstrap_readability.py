"""
Phase 30.6 – Post-Bootstrap Full Readability Battery
=======================================================
Runs the complete readability battery on the post-bootstrap decoded corpus,
comparing to all prior baselines.

Dependency chain:
    bootstrap_bigrams.json       (Step 30.3 — per-token cache + bigram z)
    bootstrap_signal.json        (Step 30.2 — signal analysis)
    bootstrap_folio.json         (Step 30.5 — folio runs)
    bootstrap_loop.json          (Step 30.1 — accepted words)
    modifier_integrate.json      (Phase 16)
    signal_bigrams.json          (Phase 29.1 — baseline)
    signal_isolation.json        (Phase 28.4 — baseline)
        → bootstrap_readability.json  (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.ventris_readability import _bigram_jsd, _section_chi_sq


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
class Validation:
    id: str
    name: str
    value: float
    threshold: float
    passed: bool
    note: str


@dataclass
class BootstrapReadabilityResult:
    validations: List[Dict]
    n_passed: int
    n_total: int
    pass_rate: float
    # Cross-phase comparison
    phase28_dict_hit: float
    phase29_z_score: float
    bootstrap_dict_hit: float
    bootstrap_z_score: float
    dict_hit_delta_vs_p28: float
    z_score_delta_vs_p29: float
    # Progression table
    progression: Dict
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bootstrap_readability() -> None:
    """Step 30.6: Full readability battery post-bootstrap."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 30.6: Post-Bootstrap Readability Battery")
    print("=" * 70)

    rd = _results_dir()

    # ── Load all inputs ──
    def _load(name):
        path = os.path.join(rd, name)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    boot_bg = _load('bootstrap_bigrams.json')
    boot_sig = _load('bootstrap_signal.json')
    boot_folio = _load('bootstrap_folio.json')
    boot_loop = _load('bootstrap_loop.json')
    boot_ctx = _load('bootstrap_context.json')
    mod_data = _load('modifier_integrate.json')
    sig_bg = _load('signal_bigrams.json')
    sig_iso = _load('signal_isolation.json')

    # Baselines
    phase28_dict_hit = 0.436  # known from memory
    phase29_z = sig_bg.get('bigram_z_score', 6.14)

    # Bootstrap values — use bootstrap_loop's final_dict_hit (computed from
    # word-in-ref_word_set, the true dict_hit rate).  token_dict_hits in
    # bootstrap_bigrams.json uses SIGNAL|SHARED_HIT classification which
    # excludes ambiguous tokens (r_hit=True, null_count=2 → SHARED_MISS).
    boot_n_tokens = boot_bg.get('n_tokens', 1)
    boot_dict_hit_rate = boot_loop.get('final_dict_hit', 0.0)
    boot_z = boot_bg.get('bigram_z_score', 0.0)

    # Build ref words for JSD
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_words = [
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    ][:5000]

    # ── Run validations ──
    print("\n  Running 10 validations …\n")
    validations: List[Validation] = []

    # V1: dict_hit >= 0.43
    v1_val = boot_dict_hit_rate
    validations.append(Validation(
        id='V1', name='dict_hit >= 0.43',
        value=round(v1_val, 4), threshold=0.43,
        passed=v1_val >= 0.43,
        note=f"Full corpus dict_hit = {v1_val:.4f}",
    ))

    # V2: bigram JSD vs Latin < 0.5
    best_run_text = boot_folio.get('best_run_text', '')
    run_words = best_run_text.split() if best_run_text else []
    # Use all signal-decoded words for JSD if best run is too short
    if len(run_words) < 20:
        decoded = boot_bg.get('token_decoded', [])
        classifications = boot_bg.get('token_classifications', [])
        run_words = [d for d, c in zip(decoded, classifications) if c == 'SIGNAL']
    v2_val = _bigram_jsd(run_words, ref_words) if run_words else 1.0
    validations.append(Validation(
        id='V2', name='bigram JSD < 0.5',
        value=round(v2_val, 4), threshold=0.5,
        passed=v2_val < 0.5,
        note=f"JSD({len(run_words)} SIGNAL words vs {len(ref_words)} ref words) = {v2_val:.4f}",
    ))

    # V3: section variation chi_sq > 3.84
    section_stats = []
    if boot_folio:
        from voynich.core.corpus import _infer_section
        decoded = boot_bg.get('token_decoded', [])
        folios = boot_bg.get('token_folios', [])
        dict_hits = boot_bg.get('token_dict_hits', [])
        section_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {'n': 0, 'hits': 0})
        for f, d, h in zip(folios, decoded, dict_hits):
            sec = _infer_section(f)
            section_counts[sec]['n'] += 1
            if h:
                section_counts[sec]['hits'] += 1
        section_stats = [
            {'section': s, 'n_tokens': v['n'], 'dict_hit': v['hits'] / v['n'] if v['n'] > 0 else 0}
            for s, v in section_counts.items()
        ]
    v3_val = _section_chi_sq(section_stats) if section_stats else 0.0
    validations.append(Validation(
        id='V3', name='section chi_sq > 3.84',
        value=round(v3_val, 2), threshold=3.84,
        passed=v3_val > 3.84,
        note=f"Chi-sq across {len(section_stats)} sections = {v3_val:.2f}",
    ))

    # V4: signal sigma mean >= 2.0
    word_sigs = boot_sig.get('word_signals', [])
    genuine_sigs = [ws for ws in word_sigs if ws.get('is_genuine_signal', False)]
    v4_val = (
        sum(ws.get('signal_sigma', 0) for ws in genuine_sigs) / len(genuine_sigs)
        if genuine_sigs else 0.0
    )
    validations.append(Validation(
        id='V4', name='signal sigma mean >= 2.0',
        value=round(v4_val, 2), threshold=2.0,
        passed=v4_val >= 2.0,
        note=f"Mean sigma of {len(genuine_sigs)} genuine signals = {v4_val:.2f}",
    ))

    # V5: n_genuine_signals >= 8
    v5_val = boot_sig.get('n_genuine_signals', 0)
    validations.append(Validation(
        id='V5', name='n_genuine >= 8',
        value=float(v5_val), threshold=8.0,
        passed=v5_val >= 8,
        note=f"{v5_val} genuine signal words",
    ))

    # V6: longest SIGNAL run > 4
    v6_val = boot_folio.get('longest_run', 0)
    validations.append(Validation(
        id='V6', name='longest run > 4',
        value=float(v6_val), threshold=4.0,
        passed=v6_val > 4,
        note=f"Longest consecutive SIGNAL run = {v6_val}",
    ))

    # V7: modifier fraction in 0.20–0.50
    mod_chars_count = len(mod_data.get('modifier_chars', mod_data.get('final_modifiers', [])))
    total_eva = 44  # 44 EVA glyphs
    v7_val = mod_chars_count / total_eva if total_eva > 0 else 0.0
    validations.append(Validation(
        id='V7', name='modifier frac 0.20-0.50',
        value=round(v7_val, 3), threshold=0.20,
        passed=0.20 <= v7_val <= 0.50,
        note=f"{mod_chars_count}/{total_eva} modifier chars = {v7_val:.3f}",
    ))

    # V8: bigram z-score >= 4.0
    v8_val = boot_z
    validations.append(Validation(
        id='V8', name='bigram z >= 4.0',
        value=round(v8_val, 2), threshold=4.0,
        passed=v8_val >= 4.0,
        note=f"SIGNAL bigram z = {v8_val:.2f}",
    ))

    # V9: no regression vs Phase 28 (dict_hit delta >= -0.005)
    v9_val = boot_dict_hit_rate - phase28_dict_hit
    validations.append(Validation(
        id='V9', name='no regression vs P28',
        value=round(v9_val, 4), threshold=-0.005,
        passed=v9_val >= -0.005,
        note=f"dict_hit delta = {v9_val:+.4f} (P28={phase28_dict_hit:.3f})",
    ))

    # V10: any new signal word or bigram hit
    new_sigs = boot_sig.get('new_signal_words', [])
    delta_hits = boot_bg.get('delta_n_bigram_hits', 0)
    v10_val = len(new_sigs) + max(delta_hits, 0)
    validations.append(Validation(
        id='V10', name='new signal/bigram >= 1',
        value=float(v10_val), threshold=1.0,
        passed=v10_val >= 1,
        note=f"{len(new_sigs)} new signal words, {delta_hits:+d} bigram hits",
    ))

    # ── Summary ──
    n_passed = sum(1 for v in validations if v.passed)
    n_total = len(validations)
    pass_rate = n_passed / n_total if n_total > 0 else 0.0

    for v in validations:
        tag = '✓' if v.passed else '✗'
        print(f"  {tag} {v.id}: {v.name:28s}  value={v.value:8.4f}  "
              f"thr={v.threshold:8.4f}  {v.note}")

    # Cross-phase progression
    boot_confirmed = len(boot_loop.get('accepted_words', []))
    boot_triples = len(boot_loop.get('confirmed_triples', []))

    progression = {
        'phase16': {'dict_hit': 0.436, 'signal_rate': None, 'bigram_z': None,
                    'confirmed_words': None, 'triples_confirmed': None},
        'phase28': {'dict_hit': 0.436, 'signal_rate': 0.165, 'bigram_z': None,
                    'confirmed_words': 8, 'triples_confirmed': 12},
        'phase29': {'dict_hit': 0.436, 'signal_rate': 0.165, 'bigram_z': phase29_z,
                    'confirmed_words': 8, 'triples_confirmed': 12},
        'phase30': {
            'dict_hit': round(boot_dict_hit_rate, 4),
            'signal_rate': round(boot_bg.get('signal_rate', 0), 4),
            'bigram_z': round(boot_z, 2),
            'confirmed_words': 8 + boot_confirmed,
            'triples_confirmed': boot_triples,
        },
    }

    print(f"\n  ── Cross-Phase Progression ──")
    print(f"  {'Phase':8s} {'dict_hit':>9s} {'sig_rate':>9s} {'z':>7s} "
          f"{'words':>6s} {'triples':>8s}")
    for phase, vals in progression.items():
        dh = f"{vals['dict_hit']:.3f}" if vals['dict_hit'] is not None else '—'
        sr = f"{vals['signal_rate']:.3f}" if vals['signal_rate'] is not None else '—'
        z = f"{vals['bigram_z']:.2f}" if vals['bigram_z'] is not None else '—'
        cw = str(vals['confirmed_words']) if vals['confirmed_words'] is not None else '—'
        ct = str(vals['triples_confirmed']) if vals['triples_confirmed'] is not None else '—'
        print(f"  {phase:8s} {dh:>9s} {sr:>9s} {z:>7s} {cw:>6s} {ct:>8s}")

    # Gate
    gate = n_passed >= 7
    if n_passed >= 9:
        verdict_str = f"READABILITY_STRONG ({n_passed}/{n_total})"
    elif n_passed >= 7:
        verdict_str = f"READABILITY_PASS ({n_passed}/{n_total})"
    else:
        verdict_str = f"READABILITY_FAIL ({n_passed}/{n_total})"

    print(f"\n     {n_passed}/{n_total} validations passed")
    print(f"     Verdict: {verdict_str}")
    print(f"     Gate: {'PASS' if gate else 'FAIL'}")

    result = BootstrapReadabilityResult(
        validations=[_convert(asdict(v)) for v in validations],
        n_passed=n_passed,
        n_total=n_total,
        pass_rate=round(pass_rate, 3),
        phase28_dict_hit=phase28_dict_hit,
        phase29_z_score=phase29_z,
        bootstrap_dict_hit=round(boot_dict_hit_rate, 4),
        bootstrap_z_score=round(boot_z, 2),
        dict_hit_delta_vs_p28=round(boot_dict_hit_rate - phase28_dict_hit, 4),
        z_score_delta_vs_p29=round(boot_z - phase29_z, 2),
        progression=progression,
        gate_passed=gate,
        verdict=verdict_str,
        runtime_seconds=round(time.time() - t0, 1),
    )

    out_path = os.path.join(rd, 'bootstrap_readability.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
