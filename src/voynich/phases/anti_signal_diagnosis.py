"""
Phase 33.1 – Anti-Signal Diagnosis
=====================================
Diagnoses which triples are responsible for generating anti-signal words
(words that appear MORE in null corpora than in real Voynich).  Builds
per-triple participation counts for SIGNAL vs ANTI_SIGNAL tokens to
identify WRONG triples.

Dependency chain:
    signal_bigrams.json        (Phase 29.1 — per-token classifications)
    signal_isolation.json      (Phase 28.4 — per-word sigma)
    combined_refine.json       (Phase 15 — best_assignment)
    bootstrap_loop.json        (Phase 30 — confirmed triples)
    compound_sign_test.json    (Phase 31.6 — prefix/suffix positions)
        → anti_signal_diagnosis.json  (this step)
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
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
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
class TripleDiagnosis:
    triple_key: str
    current_assignment: str
    signal_token_count: int
    anti_signal_token_count: int
    shared_hit_count: int
    shared_miss_count: int
    total_tokens: int
    signal_ratio: float
    confirmed: bool
    diagnosis: str  # 'CORRECT', 'SUSPECT', 'WRONG'


@dataclass
class AntiSignalWord:
    word: str
    sigma: float
    token_count: int
    triples_used: List[str]
    problem_triples: List[str]  # triples with low signal_ratio


@dataclass
class AntiSignalDiagnosisResult:
    # Per-triple diagnosis
    n_triples: int
    n_correct: int
    n_suspect: int
    n_wrong: int
    triple_diagnoses: List[Dict]
    # Anti-signal word analysis
    n_anti_signal_words: int
    anti_signal_words: List[Dict]
    # Summary
    confirmed_mean_signal_ratio: float
    unconfirmed_mean_signal_ratio: float
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_anti_signal_diagnosis() -> None:
    """Step 33.1: Diagnose which triples cause anti-signal words."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 33.1: Anti-Signal Diagnosis")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs ...")

    # Signal bigrams (per-token classifications)
    sig_bi_path = os.path.join(rd, 'signal_bigrams.json')
    if not os.path.exists(sig_bi_path):
        print("  [SKIP] signal_bigrams.json not found")
        return
    with open(sig_bi_path) as f:
        sig_bi_data = json.load(f)

    token_evas: List[str] = sig_bi_data.get('token_evas', [])
    token_decoded: List[str] = sig_bi_data.get('token_decoded', [])
    token_classifications: List[str] = sig_bi_data.get('token_classifications', [])
    token_folios: List[str] = sig_bi_data.get('token_folios', [])
    n_tokens = len(token_evas)

    print(f"     {n_tokens} tokens loaded from signal_bigrams.json")

    # Signal isolation (per-word sigma)
    sig_iso_path = os.path.join(rd, 'signal_isolation.json')
    if not os.path.exists(sig_iso_path):
        print("  [SKIP] signal_isolation.json not found")
        return
    with open(sig_iso_path) as f:
        sig_iso_data = json.load(f)

    word_signals = sig_iso_data.get('word_signals', [])

    # Assignment
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    # Confirmed triples from bootstrap_loop or crib_extraction
    confirmed_triple_set: Set[str] = set()
    boot_path = os.path.join(rd, 'bootstrap_loop.json')
    crib_path = os.path.join(rd, 'crib_extraction.json')
    if os.path.exists(boot_path):
        with open(boot_path) as f:
            boot_data = json.load(f)
        confirmed_triple_set = set(boot_data.get('confirmed_triples', []))
        print(f"     Confirmed triples (bootstrap): {len(confirmed_triple_set)}")
    elif os.path.exists(crib_path):
        with open(crib_path) as f:
            crib_data = json.load(f)
        confirmed_triple_set = set(crib_data.get('all_triples_covered', []))
        print(f"     Confirmed triples (crib extraction): {len(confirmed_triple_set)}")
    else:
        print("     No confirmed triple source found — all treated as unconfirmed")

    # Compound sign test (prefix/suffix positions)
    compound_path = os.path.join(rd, 'compound_sign_test.json')
    compound_data: Optional[Dict] = None
    if os.path.exists(compound_path):
        with open(compound_path) as f:
            compound_data = json.load(f)
        print("     Loaded compound_sign_test.json for position cross-reference")
    else:
        print("     compound_sign_test.json not found — skipping position cross-reference")

    print(f"     Assignment: {len(assignment)} triples")

    # ── 2. Extract anti-signal words ──
    print("\n  2. Extracting anti-signal words (sigma < -2.0) ...")

    anti_signal_words_raw: List[Dict] = []
    for ws in word_signals:
        sigma = ws.get('signal_sigma', 0.0)
        if sigma < -2.0:
            anti_signal_words_raw.append(ws)

    anti_signal_words_raw.sort(key=lambda w: w.get('signal_sigma', 0.0))
    anti_signal_word_set = {w['word'] for w in anti_signal_words_raw}

    print(f"     {len(anti_signal_words_raw)} anti-signal words found")
    for ws in anti_signal_words_raw:
        print(f"       {ws['word']:15s}  sigma={ws['signal_sigma']:7.2f}")

    # ── 3. Build per-triple participation matrix ──
    print("\n  3. Building per-triple participation matrix ...")

    triple_signal_counts: Dict[str, int] = defaultdict(int)
    triple_anti_signal_counts: Dict[str, int] = defaultdict(int)
    triple_shared_hit_counts: Dict[str, int] = defaultdict(int)
    triple_shared_miss_counts: Dict[str, int] = defaultdict(int)
    triple_total_counts: Dict[str, int] = defaultdict(int)

    for idx in range(n_tokens):
        eva_token = token_evas[idx]
        classification = token_classifications[idx]
        triples = token_to_triples(eva_token, eva_to_triple)

        for triple_key in triples:
            triple_total_counts[triple_key] += 1
            if classification == 'SIGNAL':
                triple_signal_counts[triple_key] += 1
            elif classification == 'ANTI_SIGNAL':
                triple_anti_signal_counts[triple_key] += 1
            elif classification == 'SHARED_HIT':
                triple_shared_hit_counts[triple_key] += 1
            elif classification == 'SHARED_MISS':
                triple_shared_miss_counts[triple_key] += 1

    # Report classification totals
    n_signal_tok = sum(1 for c in token_classifications if c == 'SIGNAL')
    n_anti_tok = sum(1 for c in token_classifications if c == 'ANTI_SIGNAL')
    n_shared_hit_tok = sum(1 for c in token_classifications if c == 'SHARED_HIT')
    n_shared_miss_tok = sum(1 for c in token_classifications if c == 'SHARED_MISS')
    print(f"     Token classifications:")
    print(f"       SIGNAL:      {n_signal_tok:6d}")
    print(f"       ANTI_SIGNAL: {n_anti_tok:6d}")
    print(f"       SHARED_HIT:  {n_shared_hit_tok:6d}")
    print(f"       SHARED_MISS: {n_shared_miss_tok:6d}")

    # ── 4. Compute signal_ratio and classify triples ──
    print("\n  4. Computing signal_ratio per triple ...")

    all_triple_keys = sorted(set(assignment.keys()) | set(triple_total_counts.keys()))

    diagnoses: List[TripleDiagnosis] = []
    for triple_key in all_triple_keys:
        sig_count = triple_signal_counts.get(triple_key, 0)
        anti_count = triple_anti_signal_counts.get(triple_key, 0)
        sh_count = triple_shared_hit_counts.get(triple_key, 0)
        sm_count = triple_shared_miss_counts.get(triple_key, 0)
        total = triple_total_counts.get(triple_key, 0)
        current_syl = assignment.get(triple_key, '???')
        is_confirmed = triple_key in confirmed_triple_set

        denominator = sig_count + anti_count
        if denominator > 0:
            signal_ratio = sig_count / denominator
        else:
            # No SIGNAL or ANTI_SIGNAL tokens use this triple — neutral
            signal_ratio = 0.5

        # Classify
        if signal_ratio > 0.7 and is_confirmed:
            diagnosis = 'CORRECT'
        elif signal_ratio < 0.3:
            diagnosis = 'WRONG'
        else:
            diagnosis = 'SUSPECT'

        diagnoses.append(TripleDiagnosis(
            triple_key=triple_key,
            current_assignment=current_syl,
            signal_token_count=sig_count,
            anti_signal_token_count=anti_count,
            shared_hit_count=sh_count,
            shared_miss_count=sm_count,
            total_tokens=total,
            signal_ratio=round(signal_ratio, 4),
            confirmed=is_confirmed,
            diagnosis=diagnosis,
        ))

    diagnoses.sort(key=lambda d: d.signal_ratio)

    n_correct = sum(1 for d in diagnoses if d.diagnosis == 'CORRECT')
    n_suspect = sum(1 for d in diagnoses if d.diagnosis == 'SUSPECT')
    n_wrong = sum(1 for d in diagnoses if d.diagnosis == 'WRONG')

    print(f"     CORRECT: {n_correct}")
    print(f"     SUSPECT: {n_suspect}")
    print(f"     WRONG:   {n_wrong}")
    print()
    for d in diagnoses:
        conf_tag = ' [confirmed]' if d.confirmed else ''
        print(f"     {d.triple_key:40s} → {d.current_assignment:4s}  "
              f"sig={d.signal_token_count:5d}  anti={d.anti_signal_token_count:5d}  "
              f"ratio={d.signal_ratio:.3f}  {d.diagnosis}{conf_tag}")

    # ── 5. Diagnose anti-signal words ──
    print("\n  5. Diagnosing anti-signal words ...")

    # Build a lookup from decoded word → list of token indices
    word_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx in range(n_tokens):
        decoded = token_decoded[idx]
        if decoded in anti_signal_word_set:
            word_to_indices[decoded].append(idx)

    # Build signal_ratio lookup
    triple_ratio_lookup: Dict[str, float] = {
        d.triple_key: d.signal_ratio for d in diagnoses
    }

    anti_words_result: List[AntiSignalWord] = []
    for ws in anti_signal_words_raw:
        word = ws['word']
        sigma = ws['signal_sigma']
        indices = word_to_indices.get(word, [])
        token_count = len(indices)

        # Collect all triples used across all tokens decoding to this word
        all_triples: Set[str] = set()
        for idx in indices:
            eva_token = token_evas[idx]
            triples = token_to_triples(eva_token, eva_to_triple)
            all_triples.update(triples)

        # Identify problem triples (signal_ratio < 0.5)
        problem_triples = sorted(
            t for t in all_triples
            if triple_ratio_lookup.get(t, 0.5) < 0.5
        )

        anti_words_result.append(AntiSignalWord(
            word=word,
            sigma=sigma,
            token_count=token_count,
            triples_used=sorted(all_triples),
            problem_triples=problem_triples,
        ))

        problem_str = ', '.join(
            f'{t}={assignment.get(t, "?")}' for t in problem_triples
        ) if problem_triples else '(none identified)'
        print(f"     {word:15s}  sigma={sigma:7.2f}  tokens={token_count:4d}  "
              f"triples={len(all_triples)}  problems: {problem_str}")

    # ── 6. Cross-reference with compound_sign_test positions ──
    print("\n  6. Cross-referencing with positional data ...")

    if compound_data is not None:
        decomp_stats = compound_data.get('decomp_stats', {})
        prefix_dist = decomp_stats.get('prefix_distribution', {})
        suffix_dist = decomp_stats.get('suffix_distribution', {})
        print(f"     Top prefixes: {list(prefix_dist.keys())[:5]}")
        print(f"     Top suffixes: {list(suffix_dist.keys())[:5]}")

        # For each WRONG/SUSPECT triple, check if it appears predominantly
        # in prefix or suffix positions
        for d in diagnoses:
            if d.diagnosis in ('WRONG', 'SUSPECT') and d.total_tokens > 0:
                # Find EVA chars that map to this triple
                eva_chars_for_triple = [
                    ch for ch in EVA_VISUAL_COMPONENTS
                    if eva_to_triple.get(ch) == d.triple_key
                ]
                # Check if any of those chars are common prefixes or suffixes
                prefix_overlap = [ch for ch in eva_chars_for_triple
                                  if ch in prefix_dist]
                suffix_overlap = [ch for ch in eva_chars_for_triple
                                  if ch in suffix_dist]
                if prefix_overlap or suffix_overlap:
                    pos_info = []
                    if prefix_overlap:
                        pos_info.append(f"prefix({','.join(prefix_overlap)})")
                    if suffix_overlap:
                        pos_info.append(f"suffix({','.join(suffix_overlap)})")
                    print(f"     {d.triple_key:40s}  positions: {', '.join(pos_info)}")
    else:
        print("     (skipped — no compound_sign_test data)")

    # ── 7. Summary statistics ──
    print("\n  7. Summary statistics ...")

    confirmed_ratios = [
        d.signal_ratio for d in diagnoses if d.confirmed
    ]
    unconfirmed_ratios = [
        d.signal_ratio for d in diagnoses if not d.confirmed
    ]
    confirmed_mean = (
        sum(confirmed_ratios) / len(confirmed_ratios)
        if confirmed_ratios else 0.0
    )
    unconfirmed_mean = (
        sum(unconfirmed_ratios) / len(unconfirmed_ratios)
        if unconfirmed_ratios else 0.0
    )

    print(f"     Confirmed triples mean signal_ratio:   {confirmed_mean:.4f}")
    print(f"     Unconfirmed triples mean signal_ratio: {unconfirmed_mean:.4f}")

    # ── 8. Verdict ──
    if n_wrong == 0 and n_suspect <= 5:
        verdict = (f"TABLE_HEALTHY: {n_correct} correct, {n_suspect} suspect, "
                   f"0 wrong triples")
    elif n_wrong == 0:
        verdict = (f"TABLE_CAUTION: {n_correct} correct, {n_suspect} suspect, "
                   f"0 wrong — many suspect triples need review")
    elif n_wrong <= 3:
        wrong_keys = [d.triple_key for d in diagnoses if d.diagnosis == 'WRONG']
        verdict = (f"FEW_WRONG: {n_wrong} wrong triple(s) ({', '.join(wrong_keys)}), "
                   f"{n_suspect} suspect, {n_correct} correct")
    else:
        verdict = (f"TABLE_DEGRADED: {n_wrong} wrong, {n_suspect} suspect, "
                   f"{n_correct} correct — significant reassignment needed")

    print(f"\n  Verdict: {verdict}")

    # ── 9. Save ──
    result = AntiSignalDiagnosisResult(
        n_triples=len(diagnoses),
        n_correct=n_correct,
        n_suspect=n_suspect,
        n_wrong=n_wrong,
        triple_diagnoses=[_convert(asdict(d)) for d in diagnoses],
        n_anti_signal_words=len(anti_words_result),
        anti_signal_words=[_convert(asdict(w)) for w in anti_words_result],
        confirmed_mean_signal_ratio=round(confirmed_mean, 4),
        unconfirmed_mean_signal_ratio=round(unconfirmed_mean, 4),
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'anti_signal_diagnosis.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)

    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")
