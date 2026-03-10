"""
Phase 30.7 – Bootstrap Verdict and Convergence Analysis
==========================================================
Reads all bootstrap iteration results, computes convergence trajectory,
performs gap analysis on unconfirmed triples, and produces the final
Phase 30 verdict.

Dependency chain:
    bootstrap_iter_N.json        (Step 30.1 — per-iteration results)
    bootstrap_loop.json          (Step 30.1 — summary)
    bootstrap_signal.json        (Step 30.2)
    bootstrap_bigrams.json       (Step 30.3)
    bootstrap_context.json       (Step 30.4)
    bootstrap_folio.json         (Step 30.5)
    bootstrap_readability.json   (Step 30.6)
    combined_refine.json         (Phase 15 — full assignment for gap analysis)
    signal_bigrams.json          (Phase 29.1 — baseline)
        → phase30_verdict.json      (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConvergencePoint:
    iteration: int
    dict_hit: float
    n_accepted: int
    confirmed_vocab_size: int
    triples_confirmed: int


@dataclass
class UnconfirmedTriple:
    triple_key: str
    first_stroke: str
    last_stroke: str
    glyph_class: str
    current_syllable: str
    eva_glyphs: List[str]
    corpus_frequency: int


@dataclass
class Phase30VerdictResult:
    # Loop summary
    n_iterations_run: int
    converged: bool
    convergence_reason: str
    # Final metrics
    final_dict_hit: float
    final_signal_rate: float
    final_bigram_z: float
    final_longest_run: int
    final_n_genuine_signals: int
    # Words
    n_new_words_confirmed: int
    new_words: List[str]
    # Convergence trajectory
    convergence_curve: List[Dict]
    trajectory_shape: str
    # Cross-phase progression
    progression: Dict
    # Gap analysis
    n_confirmed_triples: int
    n_total_triples: int
    n_unconfirmed_triples: int
    unconfirmed_triples: List[Dict]
    dark_token_fraction: float
    dark_token_count: int
    # Readability
    n_validations_passed: int
    n_validations_total: int
    # Verdict
    outcome: str
    outcome_description: str
    key_findings: List[str]
    next_steps: List[str]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def _classify_trajectory(curve: List[ConvergencePoint]) -> str:
    """Classify the convergence trajectory shape."""
    if not curve:
        return 'empty'
    if len(curve) == 1:
        if curve[0].n_accepted == 0:
            return 'immediate_stall'
        return 'single_iteration'

    accepted_counts = [c.n_accepted for c in curve]

    # Check for immediate stall
    if accepted_counts[0] == 0:
        return 'immediate_stall'

    # Check for stall after first iteration
    if all(a == 0 for a in accepted_counts[1:]):
        return 'single_burst'

    # Compare first half to second half
    mid = len(accepted_counts) // 2
    first_half_avg = sum(accepted_counts[:mid]) / mid if mid > 0 else 0
    second_half_avg = sum(accepted_counts[mid:]) / (len(accepted_counts) - mid) if (len(accepted_counts) - mid) > 0 else 0

    if second_half_avg > first_half_avg * 1.2:
        return 'accelerating'
    elif second_half_avg < first_half_avg * 0.5:
        return 'decelerating'
    else:
        return 'linear'


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def _gap_analysis(
    confirmed_triples: Set[str],
    all_triples: Dict[str, str],
    corpus_tokens: List[str],
    eva_to_triple: Dict[str, str],
) -> Tuple[List[UnconfirmedTriple], int, float]:
    """Analyze unconfirmed triples and dark vocabulary."""
    # Build reverse map: triple → EVA glyphs
    triple_to_glyphs: Dict[str, List[str]] = defaultdict(list)
    for glyph, triple_key in eva_to_triple.items():
        triple_to_glyphs[triple_key].append(glyph)

    # Count triple usage in corpus
    triple_freq: Counter = Counter()
    for token in corpus_tokens:
        from voynich.core.corpus import tokenize_eva_chars
        chars = tokenize_eva_chars(token)
        for ch in chars:
            if ch in eva_to_triple:
                triple_freq[eva_to_triple[ch]] += 1

    # Identify unconfirmed triples
    unconfirmed: List[UnconfirmedTriple] = []
    for triple_key, syllable in sorted(all_triples.items()):
        if triple_key in confirmed_triples:
            continue
        parts = triple_key.split(',')
        if len(parts) != 3:
            continue
        first_stroke, last_stroke, glyph_class = parts
        unconfirmed.append(UnconfirmedTriple(
            triple_key=triple_key,
            first_stroke=first_stroke,
            last_stroke=last_stroke,
            glyph_class=glyph_class,
            current_syllable=syllable,
            eva_glyphs=triple_to_glyphs.get(triple_key, []),
            corpus_frequency=triple_freq.get(triple_key, 0),
        ))

    unconfirmed.sort(key=lambda u: -u.corpus_frequency)

    # Dark tokens: tokens containing at least one unconfirmed triple
    unconfirmed_set = {u.triple_key for u in unconfirmed}
    dark_count = 0
    for token in corpus_tokens:
        from voynich.core.corpus import tokenize_eva_chars
        chars = tokenize_eva_chars(token)
        for ch in chars:
            if ch in eva_to_triple and eva_to_triple[ch] in unconfirmed_set:
                dark_count += 1
                break

    dark_fraction = dark_count / len(corpus_tokens) if corpus_tokens else 0.0

    return unconfirmed, dark_count, dark_fraction


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase30_verdict() -> None:
    """Step 30.7: Phase 30 verdict and convergence analysis."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 30.7: Bootstrap Verdict and Convergence Analysis")
    print("=" * 70)

    rd = _results_dir()

    def _load(name):
        path = os.path.join(rd, name)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    # ── 1. Load all Phase 30 results ──
    print("\n  1. Loading Phase 30 results …")

    boot_loop = _load('bootstrap_loop.json')
    boot_sig = _load('bootstrap_signal.json')
    boot_bg = _load('bootstrap_bigrams.json')
    boot_ctx = _load('bootstrap_context.json')
    boot_folio = _load('bootstrap_folio.json')
    boot_read = _load('bootstrap_readability.json')
    sig_bg = _load('signal_bigrams.json')

    # ── 2. Load iteration results ──
    iteration_results = []
    for i in range(1, 6):
        iter_data = _load(f'bootstrap_iter_{i}.json')
        if iter_data:
            iteration_results.append(iter_data)

    n_iter = boot_loop.get('n_iterations_run', len(iteration_results))
    converged = boot_loop.get('converged', False)
    convergence_reason = boot_loop.get('convergence_reason', 'unknown')

    print(f"     Iterations: {n_iter}")
    print(f"     Converged: {converged} ({convergence_reason})")

    # ── 3. Convergence trajectory ──
    print("\n  2. Convergence trajectory …")
    curve: List[ConvergencePoint] = []
    for ir in iteration_results:
        curve.append(ConvergencePoint(
            iteration=ir.get('iteration', 0),
            dict_hit=ir.get('dict_hit_after', 0.0),
            n_accepted=len(ir.get('confirmed', [])),
            confirmed_vocab_size=ir.get('confirmed_vocab_size', 0),
            triples_confirmed=ir.get('triples_after', 0),
        ))

    trajectory = _classify_trajectory(curve)
    print(f"     Trajectory shape: {trajectory}")
    for cp in curve:
        print(f"       Iter {cp.iteration}: accepted={cp.n_accepted}  "
              f"dict_hit={cp.dict_hit:.4f}  vocab={cp.confirmed_vocab_size}  "
              f"triples={cp.triples_confirmed}")

    # ── 4. Final metrics ──
    print("\n  3. Final metrics …")
    n_new = boot_loop.get('n_total_accepted', 0)
    new_words = boot_loop.get('accepted_words', [])
    final_dict_hit = boot_loop.get('final_dict_hit', 0.0)
    final_signal_rate = boot_sig.get('signal_token_rate', 0.0)
    final_z = boot_bg.get('bigram_z_score', 0.0)
    final_longest = boot_folio.get('longest_run', 0)
    final_n_genuine = boot_sig.get('n_genuine_signals', 0)

    baseline_signal_rate = 0.165  # Phase 28
    baseline_z = sig_bg.get('bigram_z_score', 6.14)

    print(f"     New words confirmed: {n_new}")
    if new_words:
        print(f"     Words: {', '.join(new_words)}")
    print(f"     dict_hit: {final_dict_hit:.4f}")
    print(f"     Signal rate: {final_signal_rate:.4f}")
    print(f"     Bigram z: {final_z:.2f}")
    print(f"     Longest run: {final_longest}")
    print(f"     Genuine signals: {final_n_genuine}")

    # ── 5. Gap analysis ──
    print("\n  4. Gap analysis …")
    assignment = boot_loop.get('final_assignment', {})
    confirmed_triples_list = boot_loop.get('confirmed_triples', [])
    confirmed_triples_set = set(confirmed_triples_list)

    eva_to_triple = build_eva_to_triple_lookup()
    corpus = load_corpus(verbose=False)
    corpus_tokens = corpus.get_tokens()

    unconfirmed, dark_count, dark_fraction = _gap_analysis(
        confirmed_triples_set, assignment, corpus_tokens, eva_to_triple,
    )

    print(f"     Confirmed triples: {len(confirmed_triples_set)}/{len(assignment)}")
    print(f"     Unconfirmed triples: {len(unconfirmed)}")
    print(f"     Dark tokens: {dark_count}/{len(corpus_tokens)} ({dark_fraction:.1%})")

    if unconfirmed:
        print(f"     Top unconfirmed by frequency:")
        for u in unconfirmed[:5]:
            print(f"       {u.triple_key:30s}  syl='{u.current_syllable}'  "
                  f"freq={u.corpus_frequency:5d}  glyphs={u.eva_glyphs}")

    # ── 6. Outcome classification ──
    print("\n  5. Outcome classification …")

    signal_delta = final_signal_rate - baseline_signal_rate
    z_delta = final_z - baseline_z
    n_val_passed = boot_read.get('n_passed', 0)
    n_val_total = boot_read.get('n_total', 10)

    if (n_new >= 5
            and signal_delta >= 0.02
            and z_delta >= 1.0
            and final_longest >= 5):
        outcome = 'BOOTSTRAP_SUCCESS'
        desc = (f"{n_new} new words confirmed, signal rate +{signal_delta:.1%}, "
                f"bigram z +{z_delta:.1f}, longest run {final_longest}")
    elif (n_new >= 3
            and signal_delta >= 0.01
            and z_delta >= 0.5):
        outcome = 'BOOTSTRAP_PARTIAL'
        desc = (f"{n_new} new words confirmed, signal rate +{signal_delta:.1%}, "
                f"bigram z +{z_delta:.1f}")
    elif n_new >= 1:
        outcome = 'BOOTSTRAP_MARGINAL'
        desc = (f"{n_new} new words confirmed but metrics barely changed "
                f"(Δsignal={signal_delta:+.1%}, Δz={z_delta:+.1f})")
    else:
        outcome = 'BOOTSTRAP_STALLED'
        desc = ("0 candidates passed all 4 checks — the 16 context candidates "
                "are not independently confirmable through the bootstrap mechanism")

    print(f"     Outcome: {outcome}")
    print(f"     {desc}")

    # ── 7. Key findings ──
    key_findings = []
    if n_new > 0:
        key_findings.append(
            f"Bootstrap confirmed {n_new} new words: {', '.join(new_words)}"
        )
    key_findings.append(
        f"Convergence trajectory: {trajectory} "
        f"(converged={'yes' if converged else 'no'}, reason={convergence_reason})"
    )
    key_findings.append(
        f"Confirmed {len(confirmed_triples_set)}/{len(assignment)} triples "
        f"({len(unconfirmed)} remain unconfirmed)"
    )
    key_findings.append(
        f"Dark vocabulary: {dark_fraction:.1%} of corpus tokens contain "
        f"unconfirmed triples"
    )
    key_findings.append(
        f"Readability: {n_val_passed}/{n_val_total} validations passed"
    )
    if final_z >= 6.0:
        key_findings.append(
            f"SIGNAL bigram plausibility remains strong (z={final_z:.2f})"
        )

    # ── 8. Next steps ──
    next_steps = []
    if outcome == 'BOOTSTRAP_SUCCESS':
        next_steps.append("Paper: present confirmed vocabulary, convergence analysis, "
                          "and best decoded fragment")
        next_steps.append("Gap analysis specifies exactly which external evidence "
                          "is needed for full decipherment")
    elif outcome in ('BOOTSTRAP_PARTIAL', 'BOOTSTRAP_MARGINAL'):
        next_steps.append("Paper: present Phase 29 results as primary, "
                          "bootstrap as secondary evidence")
        next_steps.append(f"External sources needed for {len(unconfirmed)} "
                          f"unconfirmed triples")
    else:
        next_steps.append("Paper: stand with Phase 29 results (z=6.14)")
        next_steps.append("Bootstrap stall is informative: signal words are "
                          "isolated peaks that don't propagate to neighbors")
        next_steps.append(f"External evidence needed for "
                          f"{len(unconfirmed)} unconfirmed triples")

    # Progression
    progression = boot_read.get('progression', {
        'phase16': {'dict_hit': 0.436},
        'phase28': {'dict_hit': 0.436, 'bigram_z': None},
        'phase29': {'dict_hit': 0.436, 'bigram_z': baseline_z},
        'phase30': {'dict_hit': final_dict_hit, 'bigram_z': final_z},
    })

    gate = outcome in ('BOOTSTRAP_SUCCESS', 'BOOTSTRAP_PARTIAL')

    print(f"\n     Gate: {'PASS' if gate else 'FAIL'}")

    result = Phase30VerdictResult(
        n_iterations_run=n_iter,
        converged=converged,
        convergence_reason=convergence_reason,
        final_dict_hit=round(final_dict_hit, 4),
        final_signal_rate=round(final_signal_rate, 4),
        final_bigram_z=round(final_z, 2),
        final_longest_run=final_longest,
        final_n_genuine_signals=final_n_genuine,
        n_new_words_confirmed=n_new,
        new_words=new_words,
        convergence_curve=[_convert(asdict(cp)) for cp in curve],
        trajectory_shape=trajectory,
        progression=progression,
        n_confirmed_triples=len(confirmed_triples_set),
        n_total_triples=len(assignment),
        n_unconfirmed_triples=len(unconfirmed),
        unconfirmed_triples=[_convert(asdict(u)) for u in unconfirmed],
        dark_token_fraction=round(dark_fraction, 4),
        dark_token_count=dark_count,
        n_validations_passed=n_val_passed,
        n_validations_total=n_val_total,
        outcome=outcome,
        outcome_description=desc,
        key_findings=key_findings,
        next_steps=next_steps,
        gate_passed=gate,
        verdict=outcome,
        runtime_seconds=round(time.time() - t0, 1),
    )

    # Print key findings
    print(f"\n  ── Key Findings ──")
    for finding in key_findings:
        print(f"     • {finding}")
    print(f"\n  ── Next Steps ──")
    for step in next_steps:
        print(f"     → {step}")

    out_path = os.path.join(rd, 'phase30_verdict.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
