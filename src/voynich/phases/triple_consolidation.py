"""
Phase 45 – Track C: Triple Confidence Consolidation
=====================================================
Establish a definitive 3-tier confidence partition of the 25-triple
assignment table using converging evidence from Phase 28, Phase 33,
Phase 44 (MaxSAT + CSA), and Phase 45 Track A (SBM forensics).

Dependency chain:
    crib_extraction.json       (Phase 28)
    bootstrap_loop.json        (Phase 30)
    maxsat_landscape.json      (Phase 44A.3)
    kperm_analysis.json        (Phase 44C.3)
    sbm_predictions.json       (Phase 44B.4)
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
        -> triple_tiers.json       (Step 45C.1)
        -> triple_ambiguous.json   (Step 45C.2)
        -> canonical_table.json    (Step 45C.3)
        -> triple_impact.json      (Step 45C.4)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
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


def _build_ref_word_set() -> set:
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    return base_words | expanded


def _compute_dict_hit_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    word_set: set,
) -> float:
    from voynich.phases.signal_isolation import _decode_corpus_r3
    decoded = _decode_corpus_r3(
        tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )
    hits = sum(1 for w in decoded if w in word_set)
    return hits / len(tokens) if tokens else 0.0


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class TripleEvidence:
    triple_key: str
    tier: str
    current_assignment: str
    maxsat_top: str
    maxsat_confidence: float
    csa_assignment: str
    sbm_community: int
    sbm_prediction: str
    n_sources_agree: int
    evidence_sources: List[str]


@dataclass
class TripleTierResult:
    tiers: Dict[str, List[Dict]]
    n_confirmed: int
    n_landscape_confirmed: int
    n_ambiguous: int
    evidence_table: List[Dict]
    runtime_seconds: float


@dataclass
class AmbiguousDossier:
    triple_key: str
    current_assignment: str
    maxsat_candidates: List[Dict]
    csa_candidate: str
    sbm_community: int
    sbm_prediction: str
    token_count: int
    signal_word_count: int
    signal_words_using: List[str]
    dict_hit_deltas: Dict[str, float]
    best_candidate: str
    best_candidate_source: str


@dataclass
class AmbiguousDossierResult:
    dossiers: List[Dict]
    n_ambiguous: int
    n_with_clear_alternative: int
    runtime_seconds: float


@dataclass
class CanonicalTableResult:
    table: Dict[str, str]
    tier_annotations: Dict[str, str]
    n_changes_from_p15: int
    changes: List[Dict]
    dict_hit: float
    baseline_dict_hit: float
    runtime_seconds: float


@dataclass
class ImpactResult:
    ambiguity_budget: float
    worst_case_dict_hit: float
    best_case_dict_hit: float
    baseline_dict_hit: float
    per_triple_impact: List[Dict]
    total_token_coverage: float
    signal_word_vulnerability: int
    gate_label: str
    runtime_seconds: float


# ══════════════════════════════════════════════════════════════════════
#  Step 45C.1 — Three-Tier Confidence Partition
# ══════════════════════════════════════════════════════════════════════

# Signal words from Phase 28/29
SIGNAL_WORDS = ['bene', 'codi', 'sero', 'sene', 'de', 'raro', 'dine', 'cola']

# MaxSAT consensus threshold for landscape-confirmed
LANDSCAPE_THRESHOLD = 0.60


def run_triple_tiers() -> None:
    """Step 45C.1: build 3-tier confidence partition."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 45C.1: Three-Tier Confidence Partition")
    print("=" * 70)

    rd = _results_dir()

    # Load all upstream evidence
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    if not assignment:
        print("  [SKIP] combined_refine.json not found")
        return

    maxsat_data = _safe_load(os.path.join(rd, 'maxsat_landscape.json'))
    per_triple_consensus = maxsat_data.get('per_triple_consensus', {})

    # CSA: load best_assignment from kperm_search.json
    kperm_search = _safe_load(os.path.join(rd, 'kperm_search.json'))
    csa_best_assignment = kperm_search.get('best_assignment', {})

    sbm_pred = _safe_load(os.path.join(rd, 'sbm_predictions.json'))
    sbm_predictions_raw = sbm_pred.get('predictions', [])
    if isinstance(sbm_predictions_raw, list):
        sbm_predictions = {p['triple_key']: p for p in sbm_predictions_raw
                           if isinstance(p, dict) and 'triple_key' in p}
    else:
        sbm_predictions = sbm_predictions_raw if isinstance(sbm_predictions_raw, dict) else {}

    sbm_comms = _safe_load(os.path.join(rd, 'sbm_communities.json'))
    communities = sbm_comms.get('communities', {})

    crib_data = _safe_load(os.path.join(rd, 'crib_extraction.json'))
    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))

    eva_to_triple = build_eva_to_triple_lookup()

    # Identify confirmed triples (Tier 1)
    confirmed_triples: Set[str] = set()
    for crib in crib_data.get('cribs', []):
        if crib.get('tier', 99) <= 2:
            for tk in crib.get('triples_covered', []):
                confirmed_triples.add(tk)
    for tk in boot_data.get('confirmed_triples', []):
        confirmed_triples.add(tk)

    # Map EVA chars to triples for community lookup
    triple_to_comm: Dict[str, int] = {}
    for ch, cid in communities.items():
        tk = eva_to_triple.get(ch)
        if tk:
            triple_to_comm[tk] = cid

    # Classify all 25 triples
    tier1 = []  # CONFIRMED
    tier2 = []  # LANDSCAPE_CONFIRMED
    tier3 = []  # GENUINELY_AMBIGUOUS
    evidence_table = []

    for tk, syl in assignment.items():
        # Get MaxSAT consensus
        consensus = per_triple_consensus.get(tk, {})
        if consensus:
            top_syl = max(consensus, key=consensus.get)
            top_conf = consensus[top_syl]
        else:
            top_syl = syl
            top_conf = 0.0

        # Get CSA assignment (single best from kperm_search)
        csa_syl = csa_best_assignment.get(tk, '')

        # Get SBM prediction (field is 'predicted_onset', not 'predicted_syllable')
        sbm_pred_entry = sbm_predictions.get(tk, {})
        sbm_pred_syl = sbm_pred_entry.get('current_assignment', '')
        comm_id = triple_to_comm.get(tk, -1)

        # Count agreeing sources
        sources_agree = []
        if top_syl == syl and top_conf > 0.5:
            sources_agree.append('maxsat')
        if csa_syl == syl:
            sources_agree.append('csa')
        if sbm_pred_syl == syl:
            sources_agree.append('sbm')

        evidence = TripleEvidence(
            triple_key=tk,
            tier='',
            current_assignment=syl,
            maxsat_top=top_syl,
            maxsat_confidence=round(top_conf, 4),
            csa_assignment=csa_syl,
            sbm_community=comm_id,
            sbm_prediction=sbm_pred_syl,
            n_sources_agree=len(sources_agree),
            evidence_sources=sources_agree,
        )

        if tk in confirmed_triples:
            evidence.tier = 'CONFIRMED'
            tier1.append(evidence)
        elif top_conf >= LANDSCAPE_THRESHOLD:
            evidence.tier = 'LANDSCAPE_CONFIRMED'
            tier2.append(evidence)
        else:
            evidence.tier = 'GENUINELY_AMBIGUOUS'
            tier3.append(evidence)

        evidence_table.append(evidence)

    print(f"\n  Tier 1 (CONFIRMED): {len(tier1)} triples")
    print(f"  Tier 2 (LANDSCAPE_CONFIRMED): {len(tier2)} triples")
    print(f"  Tier 3 (GENUINELY_AMBIGUOUS): {len(tier3)} triples")

    for ev in tier2:
        print(f"    {ev.triple_key}: {ev.current_assignment} "
              f"(MaxSAT top: {ev.maxsat_top} @ {ev.maxsat_confidence:.0%})")

    for ev in tier3:
        print(f"    {ev.triple_key}: {ev.current_assignment} "
              f"(MaxSAT: {ev.maxsat_confidence:.0%})")

    result = TripleTierResult(
        tiers={
            'CONFIRMED': [_convert(asdict(e)) for e in tier1],
            'LANDSCAPE_CONFIRMED': [_convert(asdict(e)) for e in tier2],
            'GENUINELY_AMBIGUOUS': [_convert(asdict(e)) for e in tier3],
        },
        n_confirmed=len(tier1),
        n_landscape_confirmed=len(tier2),
        n_ambiguous=len(tier3),
        evidence_table=[_convert(asdict(e)) for e in evidence_table],
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'triple_tiers.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Step 45C.2 — Ambiguous Triple Characterization
# ══════════════════════════════════════════════════════════════════════

def run_triple_ambig() -> None:
    """Step 45C.2: characterize the genuinely ambiguous triples."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 45C.2: Ambiguous Triple Characterization")
    print("=" * 70)

    rd = _results_dir()

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    if not tier_data:
        print("  [SKIP] triple_tiers.json not found; run triple-tiers first")
        return

    ambiguous = tier_data.get('tiers', {}).get('GENUINELY_AMBIGUOUS', [])
    if not ambiguous:
        print("  No ambiguous triples found — all classified.")
        return

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    maxsat_data = _safe_load(os.path.join(rd, 'maxsat_landscape.json'))
    per_triple_consensus = maxsat_data.get('per_triple_consensus', {})

    kperm_data = _safe_load(os.path.join(rd, 'kperm_search.json'))
    csa_assignment = kperm_data.get('best_assignment', {})

    sbm_pred = _safe_load(os.path.join(rd, 'sbm_predictions.json'))
    sbm_predictions_raw = sbm_pred.get('predictions', [])
    if isinstance(sbm_predictions_raw, list):
        sbm_predictions = {p['triple_key']: p for p in sbm_predictions_raw
                           if isinstance(p, dict) and 'triple_key' in p}
    else:
        sbm_predictions = sbm_predictions_raw if isinstance(sbm_predictions_raw, dict) else {}

    sbm_comms = _safe_load(os.path.join(rd, 'sbm_communities.json'))
    communities = sbm_comms.get('communities', {})

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')

    eva_to_triple = build_eva_to_triple_lookup()

    # Count tokens per triple
    print("\n  Loading corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    triple_token_count: Counter = Counter()
    for tok in all_tokens:
        triples = token_to_triples(tok, eva_to_triple)
        for tk in set(triples):  # count each triple once per token
            triple_token_count[tk] += 1

    # Check which signal words use which triples
    signal_word_triples: Dict[str, Set[str]] = {}
    for sw in SIGNAL_WORDS:
        sw_triples = set()
        for tok in all_tokens:
            chars = tokenize_eva_chars(tok)
            triples = [eva_to_triple.get(ch) for ch in chars]
            triples = [tk for tk in triples if tk]
            decoded = ''.join(assignment.get(tk, '?') for tk in triples)
            if decoded == sw:
                sw_triples.update(triples)
                break  # one example is enough
        signal_word_triples[sw] = sw_triples

    # Build word set for dict-hit delta computation
    print("  Building reference word set …")
    word_set = _build_ref_word_set()

    # Baseline dict-hit
    baseline_hit = _compute_dict_hit_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )

    dossiers = []
    for amb in ambiguous:
        tk = amb['triple_key']
        current = amb['current_assignment']

        # MaxSAT candidates
        consensus = per_triple_consensus.get(tk, {})
        maxsat_cands = sorted(
            [{'syllable': s, 'confidence': round(c, 4)}
             for s, c in consensus.items()],
            key=lambda x: -x['confidence'],
        )[:5]

        # CSA candidate
        csa_cand = csa_assignment.get(tk, '')

        # SBM prediction
        sbm_entry = sbm_predictions.get(tk, {})
        sbm_pred_syl = sbm_entry.get('current_assignment', '')

        # Community
        triple_chars = [ch for ch, t in eva_to_triple.items() if t == tk]
        comm_id = -1
        if triple_chars:
            comm_id = communities.get(triple_chars[0], -1)

        # Token count
        tok_count = triple_token_count.get(tk, 0)

        # Signal word vulnerability
        sw_using = [sw for sw, sw_tks in signal_word_triples.items() if tk in sw_tks]

        # Dict-hit delta for each candidate
        candidates = set()
        candidates.add(current)
        for mc in maxsat_cands:
            candidates.add(mc['syllable'])
        if csa_cand:
            candidates.add(csa_cand)
        if sbm_pred_syl:
            candidates.add(sbm_pred_syl)

        deltas = {}
        for cand in sorted(candidates):
            test_assign = dict(assignment)
            test_assign[tk] = cand
            dh = _compute_dict_hit_r3(
                all_tokens, test_assign, eva_to_triple,
                modifier_chars, modifier_rules, word_set,
            )
            deltas[cand] = round(dh - baseline_hit, 6)

        best_cand = max(deltas, key=deltas.get)
        best_source = 'current'
        if best_cand == csa_cand:
            best_source = 'csa'
        elif best_cand == sbm_pred_syl:
            best_source = 'sbm'
        elif any(mc['syllable'] == best_cand for mc in maxsat_cands):
            best_source = 'maxsat'

        dossier = AmbiguousDossier(
            triple_key=tk,
            current_assignment=current,
            maxsat_candidates=maxsat_cands,
            csa_candidate=csa_cand,
            sbm_community=comm_id,
            sbm_prediction=sbm_pred_syl,
            token_count=tok_count,
            signal_word_count=len(sw_using),
            signal_words_using=sw_using,
            dict_hit_deltas=deltas,
            best_candidate=best_cand,
            best_candidate_source=best_source,
        )
        dossiers.append(dossier)

        print(f"\n  {tk}:")
        print(f"    Current: {current}, Best: {best_cand} (Δ={deltas.get(best_cand, 0):.4f})")
        print(f"    MaxSAT: {maxsat_cands[:3]}")
        print(f"    CSA: {csa_cand}, SBM: {sbm_pred_syl}")
        print(f"    Tokens: {tok_count}, Signal words: {sw_using}")

    n_clear = sum(1 for d in dossiers
                  if d.best_candidate != d.current_assignment
                  and d.dict_hit_deltas.get(d.best_candidate, 0) > 0.005)

    result = AmbiguousDossierResult(
        dossiers=[_convert(asdict(d)) for d in dossiers],
        n_ambiguous=len(dossiers),
        n_with_clear_alternative=n_clear,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'triple_ambiguous.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  {n_clear} triples with clear alternative (Δ>0.5%)")
    print(f"  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Step 45C.3 — Canonical Table Assembly
# ══════════════════════════════════════════════════════════════════════

def run_triple_lock() -> None:
    """Step 45C.3: assemble canonical table with locked tiers."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 45C.3: Canonical Table Assembly")
    print("=" * 70)

    rd = _results_dir()

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    if not tier_data:
        print("  [SKIP] triple_tiers.json not found")
        return

    ambig_data = _safe_load(os.path.join(rd, 'triple_ambiguous.json'))
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    if not assignment:
        print("  [SKIP] combined_refine.json not found")
        return

    maxsat_data = _safe_load(os.path.join(rd, 'maxsat_landscape.json'))
    per_triple_consensus = maxsat_data.get('per_triple_consensus', {})

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')

    eva_to_triple = build_eva_to_triple_lookup()

    # Build canonical table
    canonical = dict(assignment)
    tier_annotations: Dict[str, str] = {}
    changes: List[Dict] = []

    # Tier 1: CONFIRMED — keep Phase 15 values
    for entry in tier_data.get('tiers', {}).get('CONFIRMED', []):
        tk = entry['triple_key']
        tier_annotations[tk] = 'CONFIRMED'

    # Tier 2: LANDSCAPE_CONFIRMED — use MaxSAT consensus value
    for entry in tier_data.get('tiers', {}).get('LANDSCAPE_CONFIRMED', []):
        tk = entry['triple_key']
        consensus = per_triple_consensus.get(tk, {})
        if consensus:
            maxsat_syl = max(consensus, key=consensus.get)
            old_syl = canonical.get(tk, '')
            if maxsat_syl != old_syl:
                canonical[tk] = maxsat_syl
                changes.append({
                    'triple': tk,
                    'old': old_syl,
                    'new': maxsat_syl,
                    'reason': f'MaxSAT consensus {consensus[maxsat_syl]:.0%}',
                })
        tier_annotations[tk] = 'LANDSCAPE_CONFIRMED'

    # Tier 3: GENUINELY_AMBIGUOUS — keep Phase 15 values by default
    # but apply changes if dossier shows clear improvement
    for entry in tier_data.get('tiers', {}).get('GENUINELY_AMBIGUOUS', []):
        tk = entry['triple_key']
        tier_annotations[tk] = 'GENUINELY_AMBIGUOUS'

        # Check if the dossier suggests a clear alternative
        if ambig_data:
            for dossier in ambig_data.get('dossiers', []):
                if dossier['triple_key'] == tk:
                    best = dossier['best_candidate']
                    delta = dossier['dict_hit_deltas'].get(best, 0)
                    if best != dossier['current_assignment'] and delta > 0.005:
                        old_syl = canonical.get(tk, '')
                        canonical[tk] = best
                        changes.append({
                            'triple': tk,
                            'old': old_syl,
                            'new': best,
                            'reason': f'Dossier best candidate Δ={delta:.4f}',
                        })
                    break

    # Any remaining triples not in tiers (shouldn't happen, but safety)
    for tk in assignment:
        if tk not in tier_annotations:
            tier_annotations[tk] = 'UNCLASSIFIED'

    print(f"\n  Canonical table: {len(canonical)} triples")
    print(f"  Changes from Phase 15: {len(changes)}")
    for ch in changes:
        print(f"    {ch['triple']}: {ch['old']} → {ch['new']} ({ch['reason']})")

    # Decode full corpus with canonical table
    print("\n  Building reference word set …")
    word_set = _build_ref_word_set()

    print("  Loading corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    canonical_hit = _compute_dict_hit_r3(
        all_tokens, canonical, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )

    baseline_hit = _compute_dict_hit_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )

    print(f"\n  Baseline dict-hit: {baseline_hit:.4f}")
    print(f"  Canonical dict-hit: {canonical_hit:.4f}")
    print(f"  Delta: {canonical_hit - baseline_hit:+.4f}")

    result = CanonicalTableResult(
        table=canonical,
        tier_annotations=tier_annotations,
        n_changes_from_p15=len(changes),
        changes=changes,
        dict_hit=round(canonical_hit, 4),
        baseline_dict_hit=round(baseline_hit, 4),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'canonical_table.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Step 45C.4 — Impact Analysis
# ══════════════════════════════════════════════════════════════════════

def run_triple_impact() -> None:
    """Step 45C.4: compute ambiguity budget and impact analysis."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 45C.4: Impact Analysis — Ambiguity Budget")
    print("=" * 70)

    rd = _results_dir()

    ambig_data = _safe_load(os.path.join(rd, 'triple_ambiguous.json'))
    if not ambig_data:
        print("  [SKIP] triple_ambiguous.json not found")
        return

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    canonical_data = _safe_load(os.path.join(rd, 'canonical_table.json'))

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')

    eva_to_triple = build_eva_to_triple_lookup()

    print("\n  Building reference word set …")
    word_set = _build_ref_word_set()

    print("  Loading corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Baseline
    base_table = canonical_data.get('table', assignment)
    baseline_hit = _compute_dict_hit_r3(
        all_tokens, base_table, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )

    # Count token coverage for ambiguous triples
    triple_token_count: Counter = Counter()
    total_chars = 0
    for tok in all_tokens:
        triples = token_to_triples(tok, eva_to_triple)
        total_chars += len(triples)
        for tk in set(triples):
            triple_token_count[tk] += 1

    dossiers = ambig_data.get('dossiers', [])

    per_triple_impact = []
    worst_case_assign = dict(base_table)
    best_case_assign = dict(base_table)

    for dossier in dossiers:
        tk = dossier['triple_key']
        deltas = dossier.get('dict_hit_deltas', {})

        if not deltas:
            per_triple_impact.append({
                'triple_key': tk,
                'token_count': triple_token_count.get(tk, 0),
                'best_delta': 0.0,
                'worst_delta': 0.0,
                'best_candidate': dossier.get('current_assignment', ''),
                'worst_candidate': dossier.get('current_assignment', ''),
            })
            continue

        best_cand = max(deltas, key=deltas.get)
        worst_cand = min(deltas, key=deltas.get)
        best_delta = deltas[best_cand]
        worst_delta = deltas[worst_cand]

        best_case_assign[tk] = best_cand
        worst_case_assign[tk] = worst_cand

        per_triple_impact.append({
            'triple_key': tk,
            'token_count': triple_token_count.get(tk, 0),
            'best_delta': round(best_delta, 6),
            'worst_delta': round(worst_delta, 6),
            'best_candidate': best_cand,
            'worst_candidate': worst_cand,
        })

    # Compute worst/best case dict-hit
    worst_hit = _compute_dict_hit_r3(
        all_tokens, worst_case_assign, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )
    best_hit = _compute_dict_hit_r3(
        all_tokens, best_case_assign, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )

    budget = best_hit - worst_hit

    # Token coverage of ambiguous triples
    ambig_triple_keys = {d['triple_key'] for d in dossiers}
    ambig_chars = 0
    for tok in all_tokens:
        triples = token_to_triples(tok, eva_to_triple)
        for tk in triples:
            if tk in ambig_triple_keys:
                ambig_chars += 1
    coverage = ambig_chars / total_chars if total_chars > 0 else 0.0

    # Signal word vulnerability
    signal_vuln = sum(d.get('signal_word_count', 0) for d in dossiers)

    # Gate label
    if budget < 0.02:
        gate = 'LOW_LEVERAGE'
    elif budget < 0.05:
        gate = 'MODERATE_LEVERAGE'
    else:
        gate = 'HIGH_LEVERAGE'

    print(f"\n  Baseline dict-hit: {baseline_hit:.4f}")
    print(f"  Worst case: {worst_hit:.4f}")
    print(f"  Best case: {best_hit:.4f}")
    print(f"  Ambiguity budget: {budget:.4f} ({budget:.1%})")
    print(f"  Token coverage: {coverage:.1%}")
    print(f"  Signal word vulnerability: {signal_vuln}")
    print(f"  Gate: {gate}")

    result = ImpactResult(
        ambiguity_budget=round(budget, 4),
        worst_case_dict_hit=round(worst_hit, 4),
        best_case_dict_hit=round(best_hit, 4),
        baseline_dict_hit=round(baseline_hit, 4),
        per_triple_impact=per_triple_impact,
        total_token_coverage=round(coverage, 4),
        signal_word_vulnerability=signal_vuln,
        gate_label=gate,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'triple_impact.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Track C Runner
# ══════════════════════════════════════════════════════════════════════

def run_track_c_45() -> None:
    """Run all Track C steps."""
    run_triple_tiers()
    print("\n" + "=" * 70 + "\n")
    run_triple_ambig()
    print("\n" + "=" * 70 + "\n")
    run_triple_lock()
    print("\n" + "=" * 70 + "\n")
    run_triple_impact()
