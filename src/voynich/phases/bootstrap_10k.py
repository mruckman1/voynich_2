"""
Step 36.5 – Ventris Bootstrap at 10K
======================================
Re-runs the iterative bootstrap under the 10K dictionary.  Phase 30
stalled because 31/33 candidates had signal position rates 0.31–0.45 at
131K.  The same candidates may have entirely different signal position
rates at 10K.

Dependency chain:
    signal_10k.json           (Step 36.2)
    context_10k.json          (Step 36.4)
    decode_10k.json           (Step 36.1)
    combined_refine.json      (Phase 15 assignment)
    modifier_integrate.json   (Phase 16 modifiers)
    null_corpus.json          (Phase 17 seeds)
    bootstrap_loop.json       (Phase 30 — previous candidates)
    crib_extraction.json      (Phase 28.1 — confirmed triples)
        → bootstrap_10k.json (this step)
"""

import json
import math
import os
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
)
from voynich.core.reference import (
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.dict_calibration import (
    _build_dict_variants,
    _classify_tokens,
)
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
        from dataclasses import asdict
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
# 4-check protocol (adapted from bootstrap_loop.py)
# ---------------------------------------------------------------------------

def _check_triple_consistency(
    word: str,
    decoded_tokens: List[str],
    token_evas: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    confirmed_triples: Set[str],
) -> Tuple[bool, str, Dict[str, str]]:
    """Check 1: triple alignment is consistent."""
    matching_evas = []
    for i, d in enumerate(decoded_tokens):
        if d == word:
            matching_evas.append(token_evas[i])

    if not matching_evas:
        return False, f"no EVA tokens decode to '{word}'", {}

    eva_counts = Counter(matching_evas)
    best_eva = eva_counts.most_common(1)[0][0]
    triples = token_to_triples(best_eva, eva_to_triple)

    syllables = []
    for t in triples:
        syl = assignment.get(t, '')
        if syl:
            syllables.append(syl)

    if not syllables:
        return False, f"no assigned syllables for triples of '{best_eva}'", {}

    proposed = {}
    for t, syl in zip(triples, syllables):
        if t not in confirmed_triples:
            proposed[t] = syl

    return True, f"consistent ({len(proposed)} new triple proposals)", proposed


def _check_signal_position(
    word: str,
    decoded_tokens: List[str],
    classifications: List[str],
    min_signal_rate: float = 0.50,
) -> Tuple[bool, float]:
    """Check 2: ≥50% of occurrences are at SIGNAL positions (10K classification)."""
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


def _check_context_reciprocity(
    word: str,
    confirmed_words: Set[str],
    decoded_tokens: List[str],
    folios: List[str],
    min_reciprocal_count: int = 1,
    min_reciprocal_pmi: float = 0.3,
) -> Tuple[bool, int]:
    """Check 3: confirmed signal words appear in candidate's context."""
    n = len(decoded_tokens)
    word_freq = Counter(decoded_tokens)
    total_tokens = n

    pair_freq: Counter = Counter()
    for i in range(n - 1):
        if folios[i] == folios[i + 1]:
            pair_freq[(decoded_tokens[i], decoded_tokens[i + 1])] += 1
    total_pairs = sum(pair_freq.values())

    neighbor_counts: Counter = Counter()
    for i in range(n):
        if decoded_tokens[i] != word:
            continue
        for delta in [-2, -1, 1, 2]:
            j = i + delta
            if 0 <= j < n and folios[i] == folios[j]:
                neighbor_counts[decoded_tokens[j]] += 1

    reciprocal_count = 0
    for cw in confirmed_words:
        if cw in neighbor_counts:
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
                reciprocal_count += 1

    return reciprocal_count >= min_reciprocal_count, reciprocal_count


def _check_typological(
    proposed_triples: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 4: proposed syllables are typologically valid."""
    if not proposed_triples:
        return True, "no new triples proposed"

    for triple, syllable in proposed_triples.items():
        parts = triple.split(',')
        if len(parts) != 3:
            return False, f"malformed triple: {triple}"
        first_stroke, last_stroke, _ = parts
        allowed_onsets = PHONEME_PLACE_MAP.get(first_stroke, [])
        allowed_nuclei = PHONEME_NUCLEUS_MAP.get(last_stroke, [])

        # Parse syllable: onset is consonant(s), nucleus is vowel(s)
        onset = ''
        nucleus = ''
        for i, ch in enumerate(syllable):
            if ch in 'aeiou':
                onset = syllable[:i]
                nucleus = syllable[i:]
                break
        else:
            onset = syllable
            nucleus = ''

        onset_ok = (onset in allowed_onsets) or (onset == '' and nucleus != '')
        nucleus_ok = any(ch in allowed_nuclei for ch in nucleus) if nucleus else True
        if not (onset_ok and nucleus_ok):
            return False, f"triple {triple}: syllable '{syllable}' not typologically valid"

    return True, "all proposed triples typologically valid"


def _rank_candidates(candidates: List[Dict]) -> List[Dict]:
    """Rank candidates by composite strength score."""
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

def run_bootstrap_10k() -> None:
    """Step 36.5: Ventris bootstrap at 10K dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 36.5: Ventris Bootstrap at 10K")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()
    max_iterations = 5

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    # Phase 15 assignment
    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = dict(refine_data.get('best_assignment', {}))

    # Phase 16 modifiers
    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    # Null seeds
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    # Signal 10K (per-token cache)
    with open(os.path.join(rd, 'signal_10k.json')) as f:
        sig_data = json.load(f)
    token_folios = sig_data['token_folios']
    token_evas = sig_data['token_evas']
    token_decoded = sig_data['token_decoded']
    token_classifications = sig_data['token_classifications']
    n_tokens = sig_data['n_tokens']

    # 10K signal words
    signal_words = set(
        w['word'] for w in sig_data.get('word_signals', [])
        if w.get('is_genuine_signal', False)
    )

    # Context 10K candidates
    ctx_path = os.path.join(rd, 'context_10k.json')
    raw_candidates = []
    if os.path.exists(ctx_path):
        with open(ctx_path) as f:
            ctx_data = json.load(f)
        raw_candidates = ctx_data.get('new_crib_candidates', [])

    # Phase 30 candidates (for re-evaluation at 10K)
    phase30_candidates = []
    boot_path = os.path.join(rd, 'bootstrap_loop.json')
    if os.path.exists(boot_path):
        with open(boot_path) as f:
            boot_data = json.load(f)
        for it in boot_data.get('iterations', []):
            for d in it.get('deferred', []):
                phase30_candidates.append({
                    'word': d['word'],
                    'source': 'phase30_deferred',
                    'n_signal_word_associations': 1,
                    'mean_pmi': 0.5,
                    'total_count': 1,
                })

    # Confirmed triples from crib extraction
    confirmed_triples: Set[str] = set()
    crib_path = os.path.join(rd, 'crib_extraction.json')
    if os.path.exists(crib_path):
        with open(crib_path) as f:
            crib_data = json.load(f)
        confirmed_triples = set(crib_data.get('all_triples_covered', []))

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Confirmed triples: {len(confirmed_triples)}")
    print(f"     Signal words (10K): {len(signal_words)}")
    print(f"     Context 10K candidates: {len(raw_candidates)}")
    print(f"     Phase 30 deferred candidates: {len(phase30_candidates)}")

    # ── 2. Build reference word sets ──
    print("\n  2. Building reference word sets …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    dict_131k = base_words | expanded

    dict_variants = _build_dict_variants(base_words, ref_corpus, [10000])
    dict_10k = dict_variants[0][1]
    print(f"     131K dict: {len(dict_131k)} words")
    print(f"     10K dict: {len(dict_10k)} words")

    # ── 3. Load corpus + build null model ──
    corpus = load_corpus(verbose=False)
    all_tokens: List[str] = []
    all_folios: List[str] = []
    for folio, page in corpus.pages.items():
        for token in page.all_tokens:
            all_tokens.append(token)
            all_folios.append(folio)

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)

    # ── 4. Baseline metrics at 10K ──
    print("\n  3. Computing baseline metrics at 10K …")
    baseline_signal_rate = sig_data['signal_rate']
    baseline_dict_hit = sig_data.get('n_signal', 0) / n_tokens if n_tokens > 0 else 0.0
    # Use decode_10k's dict_hit_10k as the proper baseline
    with open(os.path.join(rd, 'decode_10k.json')) as f:
        decode_data = json.load(f)
    baseline_dict_hit_10k = decode_data['dict_hit_10k']
    print(f"     Baseline dict_hit (10K): {baseline_dict_hit_10k:.4f}")
    print(f"     Baseline signal rate (10K): {baseline_signal_rate:.4f}")

    # ── 5. Bootstrap loop ──
    print(f"\n  4. Starting bootstrap loop (max {max_iterations} iterations) …")

    current_assignment = dict(assignment)
    current_decoded = list(token_decoded)
    current_classifications = list(token_classifications)
    confirmed_words_set = set(signal_words)
    all_accepted: List[str] = []
    iteration_results: List[Dict] = []
    prev_dict_hit = baseline_dict_hit_10k

    for iteration in range(1, max_iterations + 1):
        iter_t0 = time.time()
        print(f"\n  ── Iteration {iteration} ──")

        # A. Harvest candidates
        all_candidates = []
        seen = set()

        # 10K context candidates
        for c in raw_candidates:
            w = c.get('word', '')
            if w and w not in seen and w not in confirmed_words_set and w in dict_10k:
                all_candidates.append({**c, 'source': c.get('source', 'context_10k')})
                seen.add(w)

        # Phase 30 deferred candidates (if in 10K dict)
        for c in phase30_candidates:
            w = c.get('word', '')
            if w and w not in seen and w not in confirmed_words_set and w in dict_10k:
                all_candidates.append(c)
                seen.add(w)

        # 10K signal words themselves (self-confirmation)
        for w in signal_words:
            if w not in seen and w not in confirmed_words_set and w in dict_10k:
                all_candidates.append({
                    'word': w,
                    'source': 'signal_word_10k',
                    'n_signal_word_associations': 1,
                    'mean_pmi': 1.0,
                    'total_count': 10,
                })
                seen.add(w)

        if not all_candidates:
            print("     No candidates to test — convergence")
            break

        ranked = _rank_candidates(all_candidates)
        print(f"     {len(ranked)} candidates to test")

        # B. Apply 4 checks
        confirmed_this_iter: List[str] = []
        rejected_this_iter: List[Dict] = []
        deferred_this_iter: List[Dict] = []
        new_assignments: Dict[str, str] = {}
        triples_before = len(confirmed_triples)

        candidate_details: List[Dict] = []
        for c in ranked:
            word = c['word']

            c1_pass, c1_detail, proposed = _check_triple_consistency(
                word, current_decoded, token_evas,
                current_assignment, eva_to_triple, confirmed_triples,
            )
            c2_pass, c2_rate = _check_signal_position(
                word, current_decoded, current_classifications,
            )
            c3_pass, c3_count = _check_context_reciprocity(
                word, confirmed_words_set, current_decoded, token_folios,
            )
            c4_pass, c4_detail = _check_typological(proposed)

            # Decision
            if c1_pass and c2_pass and c3_pass and c4_pass:
                conflict = False
                reason = ''
                for t, syl in proposed.items():
                    if t in new_assignments and new_assignments[t] != syl:
                        conflict = True
                        reason = f"triple {t} conflicts"
                        break
                if not conflict:
                    status = 'CONFIRMED'
                    confirmed_this_iter.append(word)
                    confirmed_words_set.add(word)
                    for t, syl in proposed.items():
                        new_assignments[t] = syl
                        confirmed_triples.add(t)
                else:
                    status = 'DEFERRED'
            elif not c1_pass:
                status = 'REJECTED'
                reason = f"C1: {c1_detail}"
            elif not c2_pass:
                status = 'DEFERRED'
                reason = f"C2: rate={c2_rate:.3f}"
            elif not c3_pass:
                status = 'DEFERRED'
                reason = f"C3: count={c3_count}"
            elif not c4_pass:
                status = 'REJECTED'
                reason = f"C4: {c4_detail}"
            else:
                status = 'DEFERRED'
                reason = 'unknown'

            candidate_details.append({
                'word': word,
                'source': c.get('source', ''),
                'c1_pass': c1_pass, 'c1_detail': c1_detail,
                'c2_pass': c2_pass, 'c2_rate': c2_rate,
                'c3_pass': c3_pass, 'c3_count': c3_count,
                'c4_pass': c4_pass, 'c4_detail': c4_detail,
                'status': status,
                'proposed_triples': proposed,
            })

            if status == 'REJECTED':
                rejected_this_iter.append({'word': word, 'reason': reason})
            elif status == 'DEFERRED':
                deferred_this_iter.append({'word': word, 'reason': reason})

            tag = '✓' if status == 'CONFIRMED' else ('✗' if status == 'REJECTED' else '~')
            print(f"       {tag} {word:12s}  C1={'Y' if c1_pass else 'N'} "
                  f"C2={'Y' if c2_pass else 'N'}({c2_rate:.2f}) "
                  f"C3={'Y' if c3_pass else 'N'}({c3_count}) "
                  f"C4={'Y' if c4_pass else 'N'}  → {status}")

        print(f"     Confirmed: {len(confirmed_this_iter)}, "
              f"Rejected: {len(rejected_this_iter)}, "
              f"Deferred: {len(deferred_this_iter)}")

        # C. Apply new assignments and re-decode/re-classify
        for t, syl in new_assignments.items():
            current_assignment[t] = syl

        if confirmed_this_iter:
            print(f"     Re-decoding with {len(new_assignments)} new assignments …")
            # Re-decode with 131K for R3 selection
            full_decoded = _decode_corpus_r3(
                all_tokens, current_assignment, eva_to_triple,
                modifier_chars, modifier_rules, dict_131k,
            )
            current_decoded = full_decoded

            # Re-classify at 10K
            real_hits_10k = [w.lower() in dict_10k for w in current_decoded]
            null_hits_10k: List[List[bool]] = []
            for seed in null_seeds:
                null_tokens = _generate_null_corpus(
                    bigram_probs, initial_probs, token_lengths, n_tokens, seed,
                )
                null_decoded = _decode_corpus_r3(
                    null_tokens, current_assignment, eva_to_triple,
                    modifier_chars, modifier_rules, dict_131k,
                )
                null_hits_10k.append([w.lower() in dict_10k for w in null_decoded])

            current_classifications = _classify_tokens(real_hits_10k, null_hits_10k)
            new_dict_hit = sum(real_hits_10k) / n_tokens
            new_signal_rate = current_classifications.count('SIGNAL') / n_tokens
        else:
            new_dict_hit = prev_dict_hit
            new_signal_rate = current_classifications.count('SIGNAL') / n_tokens

        all_accepted.extend(confirmed_this_iter)

        iteration_results.append({
            'iteration': iteration,
            'candidates_tested': len(ranked),
            'confirmed': confirmed_this_iter,
            'rejected': rejected_this_iter,
            'deferred': deferred_this_iter,
            'candidate_details': candidate_details[:50],  # cap for JSON size
            'triples_before': triples_before,
            'triples_after': len(confirmed_triples),
            'new_assignments': new_assignments,
            'dict_hit_before': round(prev_dict_hit, 6),
            'dict_hit_after': round(new_dict_hit, 6),
            'signal_rate': round(new_signal_rate, 4),
            'confirmed_vocab_size': len(confirmed_words_set),
            'runtime_seconds': round(time.time() - iter_t0, 1),
        })

        print(f"     dict_hit (10K): {prev_dict_hit:.4f} → {new_dict_hit:.4f} "
              f"(Δ={new_dict_hit - prev_dict_hit:+.4f})")
        print(f"     SIGNAL rate: {new_signal_rate:.4f}")

        # Termination
        if not confirmed_this_iter:
            print(f"\n     Convergence: 0 new words confirmed")
            break
        if iteration >= 2:
            curr_delta = abs(new_dict_hit - prev_dict_hit)
            prev_delta = abs(
                iteration_results[-2]['dict_hit_after'] - iteration_results[-2]['dict_hit_before']
            )
            if prev_delta < 0.001 and curr_delta < 0.001:
                print(f"\n     Convergence: dict_hit delta < 0.001 for 2 iterations")
                break

        prev_dict_hit = new_dict_hit

    # ── 6. Compare to Phase 30 ──
    print("\n  5. Comparing to Phase 30 …")
    phase30_n_accepted = 0
    phase30_n_tested = 0
    if os.path.exists(boot_path):
        with open(boot_path) as f:
            p30 = json.load(f)
        phase30_n_accepted = p30.get('n_total_accepted', 0)
        phase30_n_tested = sum(it.get('candidates_tested', 0) for it in p30.get('iterations', []))
    print(f"     Phase 30 (131K): {phase30_n_accepted} accepted / {phase30_n_tested} tested")
    print(f"     Phase 36 (10K):  {len(all_accepted)} accepted / "
          f"{sum(ir['candidates_tested'] for ir in iteration_results)} tested")

    # ── 7. Save ──
    elapsed = time.time() - t0
    n_iter = len(iteration_results)
    converged = (
        n_iter < max_iterations
        or (n_iter > 0 and not iteration_results[-1]['confirmed'])
    )

    final_dict_hit = iteration_results[-1]['dict_hit_after'] if iteration_results else baseline_dict_hit_10k
    final_signal_rate = iteration_results[-1]['signal_rate'] if iteration_results else baseline_signal_rate

    # Determine verdict
    n_accepted = len(all_accepted)
    if n_accepted >= 5:
        verdict = 'BOOTSTRAP_SUCCESS'
    elif n_accepted >= 3:
        verdict = 'BOOTSTRAP_PARTIAL'
    elif n_accepted >= 1:
        verdict = 'BOOTSTRAP_MARGINAL'
    else:
        verdict = 'BOOTSTRAP_STALLED'

    output = {
        'max_iterations': max_iterations,
        'n_iterations_run': n_iter,
        'converged': converged,
        'convergence_reason': (
            'no new words confirmed' if converged and n_iter > 0 and not iteration_results[-1]['confirmed']
            else 'dict_hit delta below threshold' if converged
            else 'max iterations reached'
        ),
        'final_assignment': current_assignment,
        'initial_dict_hit_10k': round(baseline_dict_hit_10k, 4),
        'final_dict_hit_10k': round(final_dict_hit, 4),
        'dict_hit_delta': round(final_dict_hit - baseline_dict_hit_10k, 4),
        'initial_signal_rate': round(baseline_signal_rate, 4),
        'final_signal_rate': round(final_signal_rate, 4),
        'signal_rate_delta': round(final_signal_rate - baseline_signal_rate, 4),
        'iterations': iteration_results,
        'n_total_accepted': n_accepted,
        'n_total_rejected': sum(len(ir['rejected']) for ir in iteration_results),
        'n_total_deferred': sum(len(ir['deferred']) for ir in iteration_results),
        'accepted_words': all_accepted,
        'confirmed_triples': sorted(confirmed_triples),
        'unconfirmed_triples': sorted(
            set(assignment.keys()) - confirmed_triples
        ),
        # Comparison to Phase 30
        'phase30_n_accepted': phase30_n_accepted,
        'phase30_n_tested': phase30_n_tested,
        'verdict': verdict,
        'gate_passed': n_accepted >= 1,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'bootstrap_10k.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("BOOTSTRAP 10K SUMMARY")
    print("=" * 70)
    print(f"\n  Iterations: {n_iter}")
    print(f"  Accepted words: {n_accepted} — {all_accepted}")
    print(f"  Dict-hit (10K): {baseline_dict_hit_10k:.4f} → {final_dict_hit:.4f}")
    print(f"  Signal rate: {baseline_signal_rate:.4f} → {final_signal_rate:.4f}")
    print(f"  Confirmed triples: {len(confirmed_triples)}/{len(assignment)}")
    print(f"  Verdict: {verdict}")
    print(f"\n  Runtime: {elapsed:.1f}s")
