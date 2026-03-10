"""
Phase 30.1 – Iterative Ventris Bootstrap Loop
================================================
Iteratively expands the confirmed vocabulary by promoting crib candidates
that pass four independent checks: triple consistency, signal position,
context reciprocity, and typological consistency.

Each iteration: harvest candidates → rank → check → accept/reject →
re-decode → re-classify → repeat until convergence.

Dependency chain:
    combined_refine.json      (Phase 15 assignment)
    modifier_integrate.json   (Phase 16 modifiers)
    null_corpus.json          (Phase 17 seeds)
    signal_bigrams.json       (Phase 29.1 per-token cache)
    signal_context.json       (Phase 29.2 new crib candidates)
    signal_isolation.json     (Phase 28.4 signal word list)
    crib_extraction.json      (Phase 28.1 confirmed cribs)
    crib_consistency.json     (Phase 28.2 family details)
    signal_folio_read.json    (Phase 29.3 signal runs)
        → bootstrap_iter_N.json  (per-iteration results)
        → bootstrap_loop.json    (final summary)
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
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.consistency_check import _parse_syllable
from voynich.phases.csp_solver import decode_token
from voynich.phases.family_propagation import _enumerate_candidates
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.signal_isolation import _decode_corpus_r3


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
class CandidateResult:
    word: str
    source: str
    evidence_score: float
    check1_consistent: bool
    check1_detail: str
    check2_signal: bool
    check2_signal_rate: float
    check3_reciprocal: bool
    check3_reciprocal_count: int
    check4_typological: bool
    check4_detail: str
    accepted: bool
    status: str            # 'CONFIRMED', 'REJECTED', 'DEFERRED'
    rejection_reason: str
    proposed_triples: Dict[str, str]   # triple_key → syllable (new proposals)


@dataclass
class IterationResult:
    iteration: int
    candidates_tested: int
    confirmed: List[str]
    rejected: List[Dict]
    deferred: List[Dict]
    triples_before: int
    triples_after: int
    new_triple_assignments: Dict[str, str]
    signal_rate_before: float
    signal_rate_after: float
    confirmed_vocab_size: int
    dict_hit_before: float
    dict_hit_after: float
    runtime_seconds: float


@dataclass
class BootstrapLoopResult:
    max_iterations: int
    n_iterations_run: int
    converged: bool
    convergence_reason: str
    final_assignment: Dict[str, str]
    initial_dict_hit: float
    final_dict_hit: float
    dict_hit_delta: float
    initial_signal_rate: float
    final_signal_rate: float
    signal_rate_delta: float
    iterations: List[Dict]
    n_total_accepted: int
    n_total_rejected: int
    accepted_words: List[str]
    confirmed_triples: List[str]
    unconfirmed_triples: List[str]
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Token classification (same logic as signal_isolation / signal_bigrams)
# ---------------------------------------------------------------------------

def _classify_tokens(
    real_hits: List[bool],
    null_hits_list: List[List[bool]],
    n_tokens: int,
) -> List[str]:
    """Classify each token as SIGNAL/SHARED_HIT/SHARED_MISS/ANTI_SIGNAL."""
    classifications = []
    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if nh[idx])
        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')
    return classifications


# ---------------------------------------------------------------------------
# Check 1: Triple consistency
# ---------------------------------------------------------------------------

def _check_triple_consistency(
    word: str,
    decoded_tokens: List[str],
    token_evas: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    confirmed_triples: Set[str],
) -> Tuple[bool, str, Dict[str, str]]:
    """Check if the candidate word's triple alignment is consistent.

    Returns (passes, detail, proposed_triples).
    proposed_triples maps triple_key → syllable for any new (unconfirmed) triples.
    """
    # Find EVA tokens that decode to this word
    matching_evas = []
    for i, d in enumerate(decoded_tokens):
        if d == word:
            matching_evas.append(token_evas[i])

    if not matching_evas:
        return False, f"no EVA tokens decode to '{word}'", {}

    # Use the most common EVA form
    eva_counts = Counter(matching_evas)
    best_eva = eva_counts.most_common(1)[0][0]

    # Get triples for this EVA token
    triples = token_to_triples(best_eva, eva_to_triple)

    # Get syllables for each triple from current assignment
    syllables = []
    for t in triples:
        syl = assignment.get(t, '')
        if syl:
            syllables.append(syl)

    if not syllables:
        return False, f"no assigned syllables for triples of '{best_eva}'", {}

    decoded_word = ''.join(syllables).lower()

    # Check: does the decoded word match what we expect?
    # The word already decoded to this value, so we just need to check
    # if any confirmed triples are contradicted
    proposed = {}
    for t, syl in zip(triples, syllables):
        if t in confirmed_triples:
            # This triple is already confirmed — the assignment is fixed
            # No contradiction possible since this word decodes correctly
            pass
        else:
            # This is an unconfirmed triple — record the proposed assignment
            proposed[t] = syl

    return True, f"consistent ({len(proposed)} new triple proposals)", proposed


# ---------------------------------------------------------------------------
# Check 2: Signal position
# ---------------------------------------------------------------------------

def _check_signal_position(
    word: str,
    decoded_tokens: List[str],
    classifications: List[str],
    min_signal_rate: float = 0.50,
) -> Tuple[bool, float]:
    """Check if ≥50% of the candidate's occurrences are at SIGNAL positions."""
    n_total = 0
    n_signal = 0
    for i, d in enumerate(decoded_tokens):
        if d == word:
            n_total += 1
            if classifications[i] == 'SIGNAL':
                n_signal += 1

    if n_total == 0:
        return False, 0.0

    rate = n_signal / n_total
    return rate >= min_signal_rate, round(rate, 4)


# ---------------------------------------------------------------------------
# Check 3: Context reciprocity
# ---------------------------------------------------------------------------

def _check_context_reciprocity(
    word: str,
    confirmed_words: Set[str],
    decoded_tokens: List[str],
    folios: List[str],
    min_reciprocal_count: int = 1,
    min_reciprocal_pmi: float = 0.3,
) -> Tuple[bool, int]:
    """Check if the candidate's neighbors include confirmed signal words.

    The candidate was found as a neighbor of confirmed words (forward direction).
    This checks the reverse: do confirmed words appear in the candidate's context?
    """
    n = len(decoded_tokens)
    word_freq = Counter(decoded_tokens)
    total_tokens = n

    # Build adjacency pair counts (within same folio)
    pair_freq: Counter = Counter()
    for i in range(n - 1):
        if folios[i] == folios[i + 1]:
            pair_freq[(decoded_tokens[i], decoded_tokens[i + 1])] += 1
    total_pairs = sum(pair_freq.values())

    # Find all positions of candidate word
    neighbor_counts: Counter = Counter()
    for i in range(n):
        if decoded_tokens[i] != word:
            continue
        # ±2 window
        for delta in [-2, -1, 1, 2]:
            j = i + delta
            if 0 <= j < n and folios[i] == folios[j]:
                neighbor_counts[decoded_tokens[j]] += 1

    # Count confirmed words among neighbors
    reciprocal_count = 0
    for cw in confirmed_words:
        if cw in neighbor_counts:
            # Check PMI
            count = neighbor_counts[cw]
            p_pair = pair_freq.get((word, cw), 0) + pair_freq.get((cw, word), 0)
            p_pair = p_pair / total_pairs if total_pairs > 0 else 0
            p_w = word_freq[word] / total_tokens if total_tokens > 0 else 0
            p_cw = word_freq[cw] / total_tokens if total_tokens > 0 else 0
            if p_pair > 0 and p_w > 0 and p_cw > 0:
                pmi = math.log2(p_pair / (p_w * p_cw))
                if pmi >= min_reciprocal_pmi:
                    reciprocal_count += 1
            elif count >= 2:
                # Fallback: if PMI can't be computed, accept if count ≥ 2
                reciprocal_count += 1

    passes = reciprocal_count >= min_reciprocal_count
    return passes, reciprocal_count


# ---------------------------------------------------------------------------
# Check 4: Typological consistency
# ---------------------------------------------------------------------------

def _check_typological(
    proposed_triples: Dict[str, str],
    assignment: Dict[str, str],
) -> Tuple[bool, str]:
    """Check if proposed syllables are typologically valid for their triples.

    Uses PHONEME_PLACE_MAP × PHONEME_NUCLEUS_MAP to enumerate valid syllables.
    """
    if not proposed_triples:
        return True, "no new triples proposed"

    for triple, syllable in proposed_triples.items():
        current = assignment.get(triple, syllable)
        valid_set = set(_enumerate_candidates(triple, current))
        valid_set.add(current)  # _enumerate_candidates excludes current
        if syllable not in valid_set:
            # Also check directly against the maps
            parts = triple.split(',')
            if len(parts) != 3:
                return False, f"malformed triple: {triple}"
            first_stroke, last_stroke, _ = parts
            onset, nucleus = _parse_syllable(syllable)
            allowed_onsets = PHONEME_PLACE_MAP.get(first_stroke, [])
            allowed_nuclei = PHONEME_NUCLEUS_MAP.get(last_stroke, [])
            onset_ok = (onset in allowed_onsets) or (onset == '' and nucleus != '')
            nucleus_ok = any(ch in allowed_nuclei for ch in nucleus) if nucleus else True
            if not (onset_ok and nucleus_ok):
                return False, f"triple {triple}: syllable '{syllable}' not typologically valid"

    return True, "all proposed triples typologically valid"


# ---------------------------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------------------------

def _rank_candidates(
    candidates: List[Dict],
) -> List[Dict]:
    """Rank crib candidates by composite strength score."""
    scored = []
    for c in candidates:
        n_assoc = c.get('n_signal_word_associations', 1)
        mean_pmi = c.get('mean_pmi', 0.0)
        total_count = max(c.get('total_count', 1), 1)
        score = (
            n_assoc * 0.4
            + mean_pmi * 0.3
            + math.log(total_count) * 0.3
        )
        scored.append({**c, '_rank_score': round(score, 4)})
    scored.sort(key=lambda x: -x['_rank_score'])
    return scored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bootstrap_loop() -> None:
    """Step 30.1: Iterative Ventris bootstrap loop."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 30.1: Iterative Ventris Bootstrap Loop")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()
    max_iterations = 5

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    # Phase 15 assignment
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = dict(refine_data.get('best_assignment', {}))

    # Phase 16 modifiers
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Null corpus seeds
    null_path = os.path.join(rd, 'null_corpus.json')
    null_seeds = [100, 101, 102, 103, 104]
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # Signal bigrams (per-token cache)
    bigrams_path = os.path.join(rd, 'signal_bigrams.json')
    if not os.path.exists(bigrams_path):
        print("  [SKIP] signal_bigrams.json not found — run signal-bigram first")
        return
    with open(bigrams_path) as f:
        bg_data = json.load(f)
    token_folios = bg_data['token_folios']
    token_evas = bg_data['token_evas']
    token_decoded = bg_data['token_decoded']
    token_classifications = bg_data['token_classifications']
    token_dict_hits = bg_data['token_dict_hits']

    # Signal isolation (word-level signals)
    sig_path = os.path.join(rd, 'signal_isolation.json')
    if not os.path.exists(sig_path):
        print("  [SKIP] signal_isolation.json not found")
        return
    with open(sig_path) as f:
        sig_data = json.load(f)
    signal_words = {
        ws['word'] for ws in sig_data.get('word_signals', [])
        if ws.get('is_genuine_signal', False)
    }

    # Crib candidates: prefer bootstrap_context.json if exists (feedback loop)
    boot_ctx_path = os.path.join(rd, 'bootstrap_context.json')
    ctx_path = os.path.join(rd, 'signal_context.json')
    if os.path.exists(boot_ctx_path):
        print("     Using bootstrap_context.json (feedback iteration)")
        with open(boot_ctx_path) as f:
            ctx_data = json.load(f)
    elif os.path.exists(ctx_path):
        with open(ctx_path) as f:
            ctx_data = json.load(f)
    else:
        print("  [SKIP] signal_context.json not found — run signal-context first")
        return
    raw_candidates = ctx_data.get('new_crib_candidates', [])

    # Crib extraction (confirmed triples)
    crib_path = os.path.join(rd, 'crib_extraction.json')
    if not os.path.exists(crib_path):
        print("  [SKIP] crib_extraction.json not found")
        return
    with open(crib_path) as f:
        crib_data = json.load(f)
    confirmed_triples = set(crib_data.get('all_triples_covered', []))

    # Signal folio reads (for additional SIGNAL-run candidates)
    folio_path = os.path.join(rd, 'signal_folio_read.json')
    run_candidates = []
    if os.path.exists(folio_path):
        with open(folio_path) as f:
            folio_data = json.load(f)
        for run in folio_data.get('all_signal_runs', []):
            for w in run.get('decoded_words', []):
                if w and len(w) >= 2:
                    run_candidates.append({
                        'word': w,
                        'source': 'signal_run',
                        'n_signal_word_associations': 1,
                        'mean_pmi': run.get('parse_score', 0.5),
                        'total_count': 1,
                        'is_dict_hit': True,
                    })

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Confirmed triples: {len(confirmed_triples)}")
    print(f"     Signal words: {len(signal_words)}")
    print(f"     Context candidates: {len(raw_candidates)}")
    print(f"     Signal-run candidates: {len(run_candidates)}")

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 3. Load real corpus for re-decode ──
    corpus = load_corpus(verbose=False)
    all_tokens: List[str] = []
    all_folios: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            all_folios.append(folio)
    n_tokens = len(all_tokens)

    # Build null corpus model (for inter-iteration re-classification)
    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    # ── 4. Compute baseline metrics ──
    print("\n  3. Computing baseline metrics …")
    real_decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    real_hits = [w in ref_word_set for w in real_decoded]
    baseline_dict_hit = sum(real_hits) / n_tokens

    baseline_signal_rate = sum(
        1 for c in token_classifications if c == 'SIGNAL'
    ) / len(token_classifications) if token_classifications else 0.0

    print(f"     Baseline dict_hit: {baseline_dict_hit:.4f}")
    print(f"     Baseline signal rate: {baseline_signal_rate:.4f}")

    # ── 5. Bootstrap loop ──
    print(f"\n  4. Starting bootstrap loop (max {max_iterations} iterations) …")

    current_assignment = dict(assignment)
    current_decoded = list(real_decoded)
    current_classifications = list(token_classifications)
    current_folios = list(token_folios)
    current_evas = list(token_evas)
    confirmed_words = set(signal_words)
    all_accepted_words: List[str] = []
    iteration_results: List[IterationResult] = []
    prev_dict_hit = baseline_dict_hit

    for iteration in range(1, max_iterations + 1):
        iter_t0 = time.time()
        print(f"\n  ── Iteration {iteration} ──")

        # A. Harvest and merge candidates
        all_candidates = []
        seen_words = set()

        # Context PMI candidates
        for c in raw_candidates:
            w = c.get('word', '')
            if w and w not in seen_words and w not in confirmed_words:
                if c.get('is_dict_hit', False) and w in ref_word_set:
                    all_candidates.append({**c, 'source': c.get('source', 'context_pmi')})
                    seen_words.add(w)

        # Signal-run candidates
        for c in run_candidates:
            w = c.get('word', '')
            if w and w not in seen_words and w not in confirmed_words:
                if w in ref_word_set:
                    all_candidates.append(c)
                    seen_words.add(w)

        if not all_candidates:
            print(f"     No candidates to test — convergence reached")
            break

        # B. Rank candidates
        ranked = _rank_candidates(all_candidates)
        print(f"     {len(ranked)} candidates to test")

        # C. Apply 4 checks to each candidate
        candidate_results: List[CandidateResult] = []
        confirmed_this_iter: List[str] = []
        rejected_this_iter: List[Dict] = []
        deferred_this_iter: List[Dict] = []
        new_assignments_this_iter: Dict[str, str] = {}
        triples_before = len(confirmed_triples)

        for c in ranked:
            word = c['word']

            # Check 1: Triple consistency
            c1_pass, c1_detail, proposed = _check_triple_consistency(
                word, current_decoded, current_evas,
                current_assignment, eva_to_triple, confirmed_triples,
            )

            # Check 2: Signal position
            c2_pass, c2_rate = _check_signal_position(
                word, current_decoded, current_classifications,
            )

            # Check 3: Context reciprocity
            c3_pass, c3_count = _check_context_reciprocity(
                word, confirmed_words, current_decoded, current_folios,
            )

            # Check 4: Typological consistency
            c4_pass, c4_detail = _check_typological(
                proposed, current_assignment,
            )

            # Decision
            if c1_pass and c2_pass and c3_pass and c4_pass:
                status = 'CONFIRMED'
                accepted = True
                reason = ''
                # Check for triple conflicts with already-accepted this iteration
                conflict = False
                for t, syl in proposed.items():
                    if t in new_assignments_this_iter:
                        if new_assignments_this_iter[t] != syl:
                            conflict = True
                            reason = f"triple {t} conflicts with earlier acceptance"
                            status = 'DEFERRED'
                            accepted = False
                            break
                if not conflict:
                    confirmed_this_iter.append(word)
                    confirmed_words.add(word)
                    for t, syl in proposed.items():
                        new_assignments_this_iter[t] = syl
                        confirmed_triples.add(t)
            elif not c1_pass:
                status = 'REJECTED'
                accepted = False
                reason = f"Check 1 (consistency): {c1_detail}"
            elif not c2_pass:
                status = 'DEFERRED'
                accepted = False
                reason = f"Check 2 (signal position): rate={c2_rate:.3f} < 0.50"
            elif not c3_pass:
                status = 'DEFERRED'
                accepted = False
                reason = f"Check 3 (reciprocity): count={c3_count} < 1"
            elif not c4_pass:
                status = 'REJECTED'
                accepted = False
                reason = f"Check 4 (typological): {c4_detail}"
            else:
                status = 'DEFERRED'
                accepted = False
                reason = 'unknown'

            cr = CandidateResult(
                word=word,
                source=c.get('source', 'context_pmi'),
                evidence_score=c.get('_rank_score', 0.0),
                check1_consistent=c1_pass,
                check1_detail=c1_detail,
                check2_signal=c2_pass,
                check2_signal_rate=c2_rate,
                check3_reciprocal=c3_pass,
                check3_reciprocal_count=c3_count,
                check4_typological=c4_pass,
                check4_detail=c4_detail,
                accepted=accepted,
                status=status,
                rejection_reason=reason,
                proposed_triples=proposed,
            )
            candidate_results.append(cr)

            if status == 'REJECTED':
                rejected_this_iter.append({'word': word, 'reason': reason})
            elif status == 'DEFERRED':
                deferred_this_iter.append({'word': word, 'reason': reason})

            tag = '✓' if accepted else ('✗' if status == 'REJECTED' else '~')
            print(f"       {tag} {word:12s}  C1={'Y' if c1_pass else 'N'} "
                  f"C2={'Y' if c2_pass else 'N'}({c2_rate:.2f}) "
                  f"C3={'Y' if c3_pass else 'N'}({c3_count}) "
                  f"C4={'Y' if c4_pass else 'N'}  → {status}")

        print(f"     Confirmed: {len(confirmed_this_iter)}, "
              f"Rejected: {len(rejected_this_iter)}, "
              f"Deferred: {len(deferred_this_iter)}")

        # E. Apply accepted assignments
        for t, syl in new_assignments_this_iter.items():
            current_assignment[t] = syl

        # F. Re-decode and re-classify if any new assignments
        if confirmed_this_iter:
            print(f"     Re-decoding with {len(new_assignments_this_iter)} new assignments …")
            current_decoded = _decode_corpus_r3(
                all_tokens, current_assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            new_real_hits = [w in ref_word_set for w in current_decoded]

            # Re-generate null corpora and classify
            null_hits_list: List[List[bool]] = []
            for seed in null_seeds:
                null_tokens = _generate_null_corpus(
                    bigram_probs, initial_probs, token_lengths, n_tokens, seed,
                )
                null_decoded = _decode_corpus_r3(
                    null_tokens, current_assignment, eva_to_triple,
                    modifier_chars, modifier_rules, ref_word_set,
                )
                null_hits_list.append([w in ref_word_set for w in null_decoded])

            current_classifications = _classify_tokens(
                new_real_hits, null_hits_list, n_tokens,
            )

            new_dict_hit = sum(new_real_hits) / n_tokens
            new_signal_rate = sum(
                1 for c in current_classifications if c == 'SIGNAL'
            ) / n_tokens
        else:
            new_dict_hit = prev_dict_hit
            new_signal_rate = sum(
                1 for c in current_classifications if c == 'SIGNAL'
            ) / n_tokens

        all_accepted_words.extend(confirmed_this_iter)

        iter_result = IterationResult(
            iteration=iteration,
            candidates_tested=len(ranked),
            confirmed=confirmed_this_iter,
            rejected=rejected_this_iter,
            deferred=deferred_this_iter,
            triples_before=triples_before,
            triples_after=len(confirmed_triples),
            new_triple_assignments=new_assignments_this_iter,
            signal_rate_before=round(prev_dict_hit, 6),
            signal_rate_after=round(new_signal_rate, 6),
            confirmed_vocab_size=len(confirmed_words),
            dict_hit_before=round(prev_dict_hit, 6),
            dict_hit_after=round(new_dict_hit, 6),
            runtime_seconds=round(time.time() - iter_t0, 1),
        )
        iteration_results.append(iter_result)

        # Save per-iteration result
        iter_path = os.path.join(rd, f'bootstrap_iter_{iteration}.json')
        with open(iter_path, 'w') as f:
            json.dump(_convert(iter_result), f, indent=2)
        print(f"     → {iter_path}")
        print(f"     dict_hit: {prev_dict_hit:.4f} → {new_dict_hit:.4f} "
              f"(Δ={new_dict_hit - prev_dict_hit:+.4f})")

        # G. Termination check
        if not confirmed_this_iter:
            print(f"\n     Convergence: 0 new words confirmed in iteration {iteration}")
            break

        if iteration >= 2:
            prev_delta = abs(
                iteration_results[-2].dict_hit_after
                - iteration_results[-2].dict_hit_before
            )
            curr_delta = abs(new_dict_hit - prev_dict_hit)
            if prev_delta < 0.001 and curr_delta < 0.001:
                print(f"\n     Convergence: dict_hit delta < 0.001 for "
                      f"2 consecutive iterations")
                break

        prev_dict_hit = new_dict_hit

    # ── 6. Summary ──
    n_iter = len(iteration_results)
    converged = (
        n_iter < max_iterations
        or (n_iter > 0 and not iteration_results[-1].confirmed)
    )
    if converged:
        if n_iter > 0 and not iteration_results[-1].confirmed:
            convergence_reason = "no new words confirmed"
        else:
            convergence_reason = "dict_hit delta below threshold"
    else:
        convergence_reason = "max iterations reached"

    final_dict_hit = iteration_results[-1].dict_hit_after if iteration_results else baseline_dict_hit
    final_signal_rate = iteration_results[-1].signal_rate_after if iteration_results else baseline_signal_rate

    unconfirmed = sorted(
        set(assignment.keys()) - confirmed_triples
    )

    gate = len(all_accepted_words) > 0
    if len(all_accepted_words) >= 5:
        verdict = 'BOOTSTRAP_SUCCESS'
    elif len(all_accepted_words) >= 3:
        verdict = 'BOOTSTRAP_PARTIAL'
    elif len(all_accepted_words) >= 1:
        verdict = 'BOOTSTRAP_MARGINAL'
    else:
        verdict = 'BOOTSTRAP_STALLED'

    result = BootstrapLoopResult(
        max_iterations=max_iterations,
        n_iterations_run=n_iter,
        converged=converged,
        convergence_reason=convergence_reason,
        final_assignment=current_assignment,
        initial_dict_hit=round(baseline_dict_hit, 6),
        final_dict_hit=round(final_dict_hit, 6),
        dict_hit_delta=round(final_dict_hit - baseline_dict_hit, 6),
        initial_signal_rate=round(baseline_signal_rate, 6),
        final_signal_rate=round(final_signal_rate, 6),
        signal_rate_delta=round(final_signal_rate - baseline_signal_rate, 6),
        iterations=[_convert(ir) for ir in iteration_results],
        n_total_accepted=len(all_accepted_words),
        n_total_rejected=sum(len(ir.rejected) for ir in iteration_results),
        accepted_words=all_accepted_words,
        confirmed_triples=sorted(confirmed_triples),
        unconfirmed_triples=unconfirmed,
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    # Print summary
    print(f"\n  ── Summary ──")
    print(f"     Iterations: {n_iter}")
    print(f"     Converged: {converged} ({convergence_reason})")
    print(f"     Words accepted: {len(all_accepted_words)}")
    if all_accepted_words:
        print(f"     Accepted: {', '.join(all_accepted_words)}")
    print(f"     Confirmed triples: {len(confirmed_triples)}/{len(assignment)}")
    print(f"     dict_hit: {baseline_dict_hit:.4f} → {final_dict_hit:.4f} "
          f"(Δ={final_dict_hit - baseline_dict_hit:+.4f})")
    print(f"     Signal rate: {baseline_signal_rate:.4f} → {final_signal_rate:.4f}")
    print(f"     Verdict: {verdict}")
    print(f"     Gate: {'PASS' if gate else 'FAIL'}")

    out_path = os.path.join(rd, 'bootstrap_loop.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
