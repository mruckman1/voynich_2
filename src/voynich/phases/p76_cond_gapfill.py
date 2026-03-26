"""
Phase 76, Track 4: Conditional LLM Gap-Fill Re-Run
=====================================================
Only runs if Track 1 resolves >=3 triples AND new_clean_fraction > 70%.
Re-decodes corpus with updated assignment, then re-runs Phase 74's
LLM gap-filling with all hallucination controls.

Dependency chain:
    results/p76_wildcard_prop.json       (Track 1 — wildcard propagation)
    results/combined_refine.json         (Phase 15 — base assignment)
    results/p69_clean_corpus.json        (Phase 69 — T1 catalogue)
        -> results/p76_gapfill.json
"""

import asyncio
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
)
from voynich.phases.p75_redecode import _build_3coda_table
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51


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
# Decode helper
# ---------------------------------------------------------------------------

def _decode_corpus_with_assignment(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> List[str]:
    """Decode corpus with 3-coda model and given assignment table."""
    coda_table = _build_3coda_table()
    decoded = []
    for token in all_tokens:
        result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
        decoded.append(result.decoded_cvc)
    return decoded


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CondGapFillResult:
    phase: str = "76"
    step: str = "76.4"
    experiment: str = "conditional_gapfill"
    # Precondition state
    skipped: bool = True
    skip_reason: str = ""
    n_resolved_triples: int = 0
    n_likely_triples: int = 0
    new_clean_fraction: float = 0.0
    # Gap-fill results (populated only if not skipped)
    n_passages: int = 0
    n_known_answer: int = 0
    ka_accuracy: float = 0.0
    confidence_selectivity: float = 0.0
    consistency: float = 0.0
    decode_agreement: float = 0.0
    n_accepted: int = 0
    n_rejected: int = 0
    accepted_proposals: List[Dict[str, Any]] = field(default_factory=list)
    n_fully_filled: int = 0
    # Updated decode stats
    updated_dict_hit: float = 0.0
    updated_signal_count: int = 0
    # Gates
    gate_gf1: bool = False   # not skipped (preconditions met)
    gate_gf2: bool = False   # KA accuracy >= 30%
    gate_gf3: bool = False   # >= 1 accepted proposal
    gates_passed: int = 0
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_cond_gapfill() -> CondGapFillResult:
    """Track 4: Conditional LLM gap-fill re-run."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 76.4 -- Conditional LLM Gap-Fill Re-Run")
    print("=" * 47)

    # --- Check preconditions from Track 1 ---
    track1 = _safe_load(os.path.join(rd, 'p76_wildcard_prop.json'))

    if not track1:
        reason = "p76_wildcard_prop.json not found"
        print(f"  SKIP: {reason}")
        result = CondGapFillResult(
            skipped=True,
            skip_reason=reason,
            verdict='SKIPPED',
            runtime_seconds=time.time() - t0,
        )
        path = _save_json(rd, 'p76_gapfill.json', asdict(result))
        print(f"  Saved: {path}")
        return result

    n_resolved = track1.get('n_resolved', 0)
    n_likely = track1.get('n_likely', 0)
    new_clean_fraction = track1.get('new_clean_fraction', 0.0)

    print(f"  Track 1 resolved: {n_resolved}")
    print(f"  Track 1 likely: {n_likely}")
    print(f"  New clean fraction: {100*new_clean_fraction:.1f}%")

    precond_triples = (n_resolved + n_likely) >= 3
    precond_clean = new_clean_fraction > 0.70

    if not precond_triples:
        reason = (f"Insufficient triples: {n_resolved} resolved + "
                  f"{n_likely} likely = {n_resolved + n_likely} < 3")
        print(f"  SKIP: {reason}")
        result = CondGapFillResult(
            skipped=True,
            skip_reason=reason,
            n_resolved_triples=n_resolved,
            n_likely_triples=n_likely,
            new_clean_fraction=new_clean_fraction,
            verdict='SKIPPED',
            runtime_seconds=time.time() - t0,
        )
        path = _save_json(rd, 'p76_gapfill.json', asdict(result))
        print(f"  Saved: {path}")
        return result

    if not precond_clean:
        reason = (f"Clean fraction {100*new_clean_fraction:.1f}% <= 70%")
        print(f"  SKIP: {reason}")
        result = CondGapFillResult(
            skipped=True,
            skip_reason=reason,
            n_resolved_triples=n_resolved,
            n_likely_triples=n_likely,
            new_clean_fraction=new_clean_fraction,
            verdict='SKIPPED',
            runtime_seconds=time.time() - t0,
        )
        path = _save_json(rd, 'p76_gapfill.json', asdict(result))
        print(f"  Saved: {path}")
        return result

    print("  Preconditions MET -- proceeding with gap-fill re-run")

    # --- Build updated assignment table ---
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    base_assignment = refine_data.get('best_assignment', {})

    # Merge Track 1 resolved triples into assignment
    updated_assignment = dict(base_assignment)
    resolved_triples = track1.get('resolved_triples', [])
    for entry in resolved_triples:
        triple = entry.get('triple', '')
        value = entry.get('assigned_value', '')
        if triple and value:
            updated_assignment[triple] = value

    likely_triples = track1.get('likely_triples', [])
    for entry in likely_triples:
        triple = entry.get('triple', '')
        value = entry.get('assigned_value', '')
        if triple and value:
            updated_assignment[triple] = value

    n_new_assignments = len(updated_assignment) - len(base_assignment)
    print(f"  Updated assignment: {len(base_assignment)} -> "
          f"{len(updated_assignment)} (+{n_new_assignments})")

    # --- Load shared data ---
    eva_to_triple = build_eva_to_triple_lookup()

    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])
    t1_types: Dict[str, Dict] = {}
    for entry in t1_catalogue:
        eva_type = entry.get('eva_type', '')
        if eva_type:
            t1_types[eva_type] = entry

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    folio_list: List[str] = []
    for folio_id, page in corpus.pages.items():
        for _ in page.all_tokens:
            folio_list.append(folio_id)

    print(f"  T1 types: {len(t1_types)}, tokens: {len(all_tokens)}")

    # --- Re-decode with updated assignment ---
    print("  Decoding corpus with updated assignment...")
    decoded = _decode_corpus_with_assignment(
        all_tokens, updated_assignment, eva_to_triple)

    dict_hits = sum(1 for d in decoded if d and d.lower() in ref_word_set)
    updated_dict_hit = dict_hits / len(decoded) if decoded else 0.0
    print(f"  Updated dict-hit: {100*updated_dict_hit:.1f}%")

    # --- Import gap-fill functions from Phase 74 ---
    from voynich.phases.p74_llm_gapfill import (
        _build_known_answer_passages,
        _build_random_passage,
        _build_shuffled_control,
        _run_gap_filling_async,
        _score_gap_filling,
        _score_known_answers,
        _select_gap_fill_passages,
        _validate_proposals,
    )

    # --- Select passages ---
    print("  Selecting gap-fill passages...")
    real_passages = _select_gap_fill_passages(
        all_tokens, decoded, t1_types, ref_word_set, folio_list, n=15)
    print(f"    Selected {len(real_passages)} passages")

    if not real_passages:
        print("  WARNING: No suitable passages found.")
        result = CondGapFillResult(
            skipped=False,
            n_resolved_triples=n_resolved,
            n_likely_triples=n_likely,
            new_clean_fraction=new_clean_fraction,
            n_passages=0,
            updated_dict_hit=updated_dict_hit,
            gate_gf1=True,
            gates_passed=1,
            verdict='NO_PASSAGES',
            runtime_seconds=time.time() - t0,
        )
        path = _save_json(rd, 'p76_gapfill.json', asdict(result))
        print(f"  Saved: {path}")
        return result

    mean_id = float(np.mean([p['id_fraction'] for p in real_passages]))
    print(f"    Mean identification: {100*mean_id:.1f}%")

    # --- Build controls ---
    print("  Building controls...")
    shuffled = [_build_shuffled_control(p, seed=i * 31)
                for i, p in enumerate(real_passages)]

    random_passages: List[Dict[str, Any]] = []
    for i, p in enumerate(real_passages):
        rp = _build_random_passage(
            all_tokens, decoded, t1_types, ref_word_set, folio_list,
            n_gaps=p['n_gaps'], seed=2000 + i)
        if rp:
            random_passages.append(rp)
    print(f"    Random passages: {len(random_passages)}")

    known_answer = _build_known_answer_passages(
        all_tokens, decoded, t1_types, ref_word_set, folio_list, n=10)
    print(f"    Known-answer passages: {len(known_answer)}")

    n_api_calls = (len(real_passages) * 3 + len(shuffled) +
                   len(random_passages) + len(known_answer))
    print(f"    Total API calls: {n_api_calls}")

    # --- Run LLM queries ---
    print("\n  Running LLM gap-filling...")
    all_results, ka_results = asyncio.run(
        _run_gap_filling_async(
            real_passages, known_answer, shuffled, random_passages))

    n_failures = sum(1 for r in all_results
                     if r.get('control_type') is None
                     and not any(r.get('runs', [])))
    n_failures += sum(1 for r in ka_results if not r.get('result'))
    print(f"    API failures: {n_failures}")

    # --- Score known answers ---
    print("\n  Scoring known-answer calibration...")
    ka_scores = _score_known_answers(ka_results, known_answer)
    ka_accuracy = ka_scores.get('accuracy', 0.0)
    print(f"    KA accuracy: {100*ka_accuracy:.1f}% "
          f"({ka_scores.get('n_correct', 0)}/{ka_scores.get('n_total', 0)})")

    # --- Score gap filling ---
    print("  Scoring gap-filling quality...")
    scores = _score_gap_filling(all_results, ka_scores, ref_word_set)
    conf_sel = scores.get('confidence_selectivity', 0.0)
    consistency = scores.get('consistency', 0.0)
    decode_agree = scores.get('decode_agreement', 0.0)
    print(f"    Confidence selectivity: {conf_sel:.2f}x")
    print(f"    Consistency: {100*consistency:.1f}%")
    print(f"    Decode agreement: {100*decode_agree:.1f}%")

    # --- Validate proposals ---
    print("\n  Validating proposals...")
    validation = _validate_proposals(all_results, scores, ref_word_set)
    n_accepted = validation.get('n_accepted', 0)
    n_rejected = validation.get('n_rejected', 0)
    accepted_list = validation.get('accepted', [])
    print(f"    Accepted: {n_accepted}")
    print(f"    Rejected: {n_rejected}")

    if accepted_list:
        print(f"    Accepted proposals:")
        for p in accepted_list[:10]:
            print(f"      pos {p.get('position', '?')}: "
                  f"'{p.get('proposed_word', '?')}' "
                  f"({p.get('proposed_gloss', '?')}) -- "
                  f"decoded='{p.get('decoded_string', '?')}', "
                  f"ED={p.get('ed_with_decoded', '?')}")

    # Count fully-filled passages
    filled_by_passage: Dict[int, int] = Counter()
    gaps_by_passage: Dict[int, int] = Counter()
    for r in [r for r in all_results if r.get('control_type') is None]:
        passage_info = r.get('passage', {})
        start = passage_info.get('start', -1)
        gaps_by_passage[start] = passage_info.get('n_gaps', 0)
    for p in accepted_list:
        filled_by_passage[p.get('passage_start', -1)] += 1

    n_fully_filled = sum(1 for start, n_filled in filled_by_passage.items()
                         if n_filled >= gaps_by_passage.get(start, 999))

    # --- Gates ---
    gate_gf1 = True  # preconditions passed (we got here)
    gate_gf2 = ka_accuracy >= 0.30
    gate_gf3 = n_accepted >= 1

    gates_passed = sum([gate_gf1, gate_gf2, gate_gf3])

    print(f"\n  Gates:")
    print(f"    GF1 (not skipped): PASS")
    print(f"    GF2 (KA accuracy >=30%): {'PASS' if gate_gf2 else 'FAIL'} "
          f"({100*ka_accuracy:.1f}%)")
    print(f"    GF3 (>=1 accepted): {'PASS' if gate_gf3 else 'FAIL'} "
          f"({n_accepted})")
    print(f"    Total: {gates_passed}/3")

    # --- Verdict ---
    if gate_gf1 and gate_gf2 and gate_gf3:
        verdict = 'GAPFILL_PRODUCTIVE'
    elif gate_gf1 and gate_gf2:
        verdict = 'GAPFILL_CALIBRATED'
    elif gate_gf1:
        verdict = 'GAPFILL_ATTEMPTED'
    else:
        verdict = 'GAPFILL_FAILED'

    result = CondGapFillResult(
        skipped=False,
        skip_reason='',
        n_resolved_triples=n_resolved,
        n_likely_triples=n_likely,
        new_clean_fraction=new_clean_fraction,
        n_passages=len(real_passages),
        n_known_answer=len(known_answer),
        ka_accuracy=round(ka_accuracy, 4),
        confidence_selectivity=round(conf_sel, 4),
        consistency=round(consistency, 4),
        decode_agreement=round(decode_agree, 4),
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        accepted_proposals=accepted_list[:20],
        n_fully_filled=n_fully_filled,
        updated_dict_hit=round(updated_dict_hit, 4),
        updated_signal_count=0,
        gate_gf1=gate_gf1,
        gate_gf2=gate_gf2,
        gate_gf3=gate_gf3,
        gates_passed=gates_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p76_gapfill.json', asdict(result))
    print(f"\n  Verdict: {verdict} ({gates_passed}/3)")
    print(f"  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
