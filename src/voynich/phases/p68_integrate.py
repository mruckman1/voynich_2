"""
Phase 68: Rare Syllable Recovery — Integration
================================================
Collects evidence from all 7 tracks, votes on each unresolved triple,
builds a final assignment table, and evaluates it against T_P15.

Dependency chain:
    results/p68_full_tokens.json      (Track 1)
    results/p68_within_token.json     (Track 2)
    results/p68_paradigmatic.json     (Track 3)
    results/p68_expanded_t1.json      (Track 4)
    results/p68_formulaic.json        (Track 5)
    results/p68_distributional.json   (Track 6)
    results/p68_ed_lattice.json       (Track 7)
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
        -> results/p68_integrate.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
)
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _compute_bigram_z,
    _run_signal_isolation,
)
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13)."""
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TripleEvidence:
    triple_key: str
    votes: Dict[str, str]       # {track_name: predicted_syllable}
    consensus_syllable: str
    consensus_count: int
    status: str                  # RESOLVED / LIKELY / UNRESOLVED
    tp15_value: str
    changed: bool


@dataclass
class Phase68IntegrateResult:
    phase: str = "68"
    step: str = "68.8"
    experiment: str = "phase68_integrate"
    # Track availability
    n_tracks_run: int = 0
    tracks_available: List[str] = field(default_factory=list)
    # Per-triple evidence
    per_triple: List[TripleEvidence] = field(default_factory=list)
    n_resolved: int = 0
    n_likely: int = 0
    n_unresolved: int = 0
    # Final assignment comparison
    final_assignment: Dict[str, str] = field(default_factory=dict)
    n_changed_from_tp15: int = 0
    changes: List[Dict[str, str]] = field(default_factory=list)
    # Evaluation
    tp15_dict_hit: float = 0.0
    final_dict_hit: float = 0.0
    delta_dict_hit: float = 0.0
    tp15_signal_words: int = 0
    final_signal_words: int = 0
    tp15_bigram_z: float = 0.0
    final_bigram_z: float = 0.0
    # Verdict
    verdict: str = ""
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

TRACK_FILES = {
    'full_tokens': 'p68_full_tokens.json',
    'within_token': 'p68_within_token.json',
    'paradigmatic': 'p68_paradigmatic.json',
    'expanded_t1': 'p68_expanded_t1.json',
    'formulaic': 'p68_formulaic.json',
    'distributional': 'p68_distributional.json',
    'ed_lattice': 'p68_ed_lattice.json',
}


def _collect_evidence(
    rd: str,
    unresolved_keys: List[str],
) -> Dict[str, Dict[str, str]]:
    """Collect predictions from all 7 tracks.

    Returns {triple_key: {track_name: predicted_syllable}}.
    """
    evidence: Dict[str, Dict[str, str]] = {tk: {} for tk in unresolved_keys}

    for track_name, filename in TRACK_FILES.items():
        data = _safe_load(os.path.join(rd, filename))
        if not data:
            continue

        # Each track stores triple_candidates: Dict[str, str]
        candidates = data.get('triple_candidates', {})
        if not candidates:
            continue

        for tk, syllable in candidates.items():
            if tk in evidence and syllable:
                evidence[tk][track_name] = syllable

    return evidence


def _vote(
    evidence: Dict[str, Dict[str, str]],
    unresolved: Dict[str, str],
) -> List[TripleEvidence]:
    """Majority vote across tracks for each unresolved triple."""
    results = []

    for tk in sorted(evidence.keys()):
        votes = evidence[tk]
        tp15_val = unresolved.get(tk, '?')

        if not votes:
            results.append(TripleEvidence(
                triple_key=tk,
                votes=votes,
                consensus_syllable=tp15_val,
                consensus_count=0,
                status='UNRESOLVED',
                tp15_value=tp15_val,
                changed=False,
            ))
            continue

        # Count syllable votes
        syllable_counter = Counter(votes.values())
        top_syl, top_count = syllable_counter.most_common(1)[0]

        if top_count >= 3:
            status = 'RESOLVED'
        elif top_count >= 2:
            status = 'LIKELY'
        else:
            status = 'UNRESOLVED'

        # Use consensus syllable if RESOLVED or LIKELY, else keep T_P15
        consensus = top_syl if status in ('RESOLVED', 'LIKELY') else tp15_val

        results.append(TripleEvidence(
            triple_key=tk,
            votes=votes,
            consensus_syllable=consensus,
            consensus_count=top_count,
            status=status,
            tp15_value=tp15_val,
            changed=consensus != tp15_val,
        ))

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _decode_and_evaluate(
    assignment: Dict[str, str],
    all_tokens: List[str],
    eva_to_triple: Dict[str, str],
    coda_table,
    ref_word_set: Set[str],
) -> Tuple[List[str], float]:
    """Decode corpus and compute dict_hit."""
    decoded = []
    n_hits = 0
    for token in all_tokens:
        result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
        d = result.decoded_cvc
        decoded.append(d if d else '')
        if d and '?' not in d and d in ref_word_set:
            n_hits += 1

    dict_hit = n_hits / len(all_tokens) if all_tokens else 0.0
    return decoded, dict_hit


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_phase68_verdict():
    """Integration: combine all 7 tracks and evaluate."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 68 — Integration Verdict")
    print("=" * 35)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    unresolved_keys = sorted(unresolved.keys())
    print(f"  Confirmed: {len(confirmed)}, Unresolved: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folios = _build_folio_list(corpus)

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # --- Check which tracks have run ---
    tracks_available = []
    for name, fname in TRACK_FILES.items():
        if os.path.exists(os.path.join(rd, fname)):
            tracks_available.append(name)
    print(f"  Tracks available: {len(tracks_available)} — {', '.join(tracks_available)}")

    # --- Collect evidence ---
    evidence = _collect_evidence(rd, unresolved_keys)

    # --- Vote ---
    voted = _vote(evidence, unresolved)

    n_resolved = sum(1 for v in voted if v.status == 'RESOLVED')
    n_likely = sum(1 for v in voted if v.status == 'LIKELY')
    n_unresolved = sum(1 for v in voted if v.status == 'UNRESOLVED')

    print(f"\n  Voting results:")
    print(f"    RESOLVED:   {n_resolved}")
    print(f"    LIKELY:     {n_likely}")
    print(f"    UNRESOLVED: {n_unresolved}")

    for v in voted:
        n_votes = len(v.votes)
        change_mark = " ** CHANGED" if v.changed else ""
        print(f"    {v.triple_key}: {v.consensus_syllable} "
              f"({v.status}, {v.consensus_count}/{n_votes} votes, "
              f"T_P15={v.tp15_value}){change_mark}")

    # --- Build final assignment ---
    final_assignment = dict(confirmed)
    for v in voted:
        final_assignment[v.triple_key] = v.consensus_syllable

    changes = [{'triple_key': v.triple_key, 'from': v.tp15_value,
                'to': v.consensus_syllable}
               for v in voted if v.changed]
    n_changed = len(changes)

    # --- Evaluate T_P15 ---
    print("\n  Evaluating T_P15 baseline...")
    tp15_full = dict(confirmed)
    tp15_full.update(unresolved)
    tp15_decoded, tp15_dict_hit = _decode_and_evaluate(
        tp15_full, all_tokens, eva_to_triple, coda_table, ref_word_set)
    print(f"    T_P15 dict_hit: {tp15_dict_hit:.4f}")

    # --- Evaluate final ---
    print("  Evaluating Phase 68 assignment...")
    final_decoded, final_dict_hit = _decode_and_evaluate(
        final_assignment, all_tokens, eva_to_triple, coda_table, ref_word_set)
    delta = final_dict_hit - tp15_dict_hit
    print(f"    Final dict_hit: {final_dict_hit:.4f} (Δ = {delta:+.4f})")

    # --- Signal isolation ---
    print("  Running signal isolation...")
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_seeds = ([r['seed'] for r in null_data.get('null_runs', [])]
                  if null_data else [100, 101, 102, 103, 104])

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    null_decoded_tp15: List[List[str]] = []
    null_decoded_final: List[List[str]] = []
    for seed in null_seeds[:3]:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed)

        nd_tp15 = []
        nd_final = []
        for nt in null_tokens:
            r1 = decode_token_cvc_v2(nt, tp15_full, eva_to_triple, coda_table)
            nd_tp15.append(r1.decoded_cvc if r1.decoded_cvc else '')
            r2 = decode_token_cvc_v2(nt, final_assignment, eva_to_triple, coda_table)
            nd_final.append(r2.decoded_cvc if r2.decoded_cvc else '')

        null_decoded_tp15.append(nd_tp15)
        null_decoded_final.append(nd_final)

    tp15_signal = _run_signal_isolation(
        tp15_decoded, null_decoded_tp15, ref_word_set, len(all_tokens))
    final_signal = _run_signal_isolation(
        final_decoded, null_decoded_final, ref_word_set, len(all_tokens))

    tp15_sw = tp15_signal.n_signal_words
    final_sw = final_signal.n_signal_words
    print(f"    T_P15 signal words: {tp15_sw}")
    print(f"    Final signal words: {final_sw}")

    # --- Bigram z ---
    print("  Computing bigram z-scores...")
    tp15_bz = _compute_bigram_z(
        tp15_decoded, null_decoded_tp15, ref_word_set, folios, n_perms=200)
    final_bz = _compute_bigram_z(
        final_decoded, null_decoded_final, ref_word_set, folios, n_perms=200)
    print(f"    T_P15 bigram z: {tp15_bz:.2f}")
    print(f"    Final bigram z: {final_bz:.2f}")

    # --- Verdict ---
    if n_resolved >= 5 and delta > 0.02:
        verdict = 'RESOLVED_IMPROVEMENT'
    elif n_resolved >= 2 or delta > 0:
        verdict = 'PARTIAL_RESOLUTION'
    elif n_likely >= 3:
        verdict = 'WEAK_CONSENSUS'
    else:
        verdict = 'NO_CONSENSUS'

    gates_passed = sum([
        delta >= 0,               # G1: no regression
        n_resolved >= 1,          # G2: at least 1 resolved
        final_sw >= tp15_sw,      # G3: signal preserved
        final_bz >= tp15_bz * 0.9,  # G4: bigram z not regressed
        n_changed >= 1,           # G5: at least 1 change proposed
    ])

    result = Phase68IntegrateResult(
        n_tracks_run=len(tracks_available),
        tracks_available=tracks_available,
        per_triple=voted,
        n_resolved=n_resolved,
        n_likely=n_likely,
        n_unresolved=n_unresolved,
        final_assignment=final_assignment,
        n_changed_from_tp15=n_changed,
        changes=changes,
        tp15_dict_hit=round(tp15_dict_hit, 4),
        final_dict_hit=round(final_dict_hit, 4),
        delta_dict_hit=round(delta, 4),
        tp15_signal_words=tp15_sw,
        final_signal_words=final_sw,
        tp15_bigram_z=round(tp15_bz, 2),
        final_bigram_z=round(final_bz, 2),
        verdict=verdict,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p68_integrate.json', result)

    # --- Summary ---
    print(f"\n  {'=' * 40}")
    print(f"  VERDICT: {verdict}")
    print(f"  {'=' * 40}")
    print(f"  Resolved: {n_resolved}, Likely: {n_likely}, Unresolved: {n_unresolved}")
    print(f"  Dict hit: {tp15_dict_hit:.4f} → {final_dict_hit:.4f} (Δ = {delta:+.4f})")
    print(f"  Signal:   {tp15_sw} → {final_sw}")
    print(f"  Bigram z: {tp15_bz:.2f} → {final_bz:.2f}")
    print(f"  Changes:  {n_changed}")
    print(f"  Gates: {gates_passed}/5")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")


def run_phase68():
    """Run the full Phase 68 pipeline: all 7 tracks + integration."""
    t0 = time.time()

    print("Phase 68 — Rare Syllable Recovery (Full Pipeline)")
    print("=" * 52)
    print()

    # Track 1: Fully-decoded tokens
    from voynich.phases.p68_full_tokens import run_full_tokens
    run_full_tokens()
    print()

    # Track 2: Within-token co-occurrence
    from voynich.phases.p68_within_token import run_within_token
    run_within_token()
    print()

    # Track 3: Paradigmatic analysis
    from voynich.phases.p68_paradigmatic import run_paradigmatic
    run_paradigmatic()
    print()

    # Track 4: Expanded T1
    from voynich.phases.p68_expanded_t1 import run_expanded_t1
    run_expanded_t1()
    print()

    # Track 5: Formulaic patterns
    from voynich.phases.p68_formulaic import run_formula_decode
    run_formula_decode()
    print()

    # Track 6: Distributional constraints
    from voynich.phases.p68_distributional import run_distrib_constrain
    run_distrib_constrain()
    print()

    # Track 7: ED lattice
    from voynich.phases.p68_ed_lattice import run_ed_lattice
    run_ed_lattice()
    print()

    # Integration
    run_phase68_verdict()

    print(f"\nPhase 68 total time: {time.time() - t0:.1f}s")
