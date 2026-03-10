"""
Step 33.6 – Perplexity-Minimizing Coordinate Descent
=====================================================
For each triple, find the syllable assignment that minimises the perplexity of
the decoded text under a Latin character 5-gram LM.  Uses coordinate descent:
fix all triples except one, optimise that one, cycle through all until
convergence (or 3 full passes).

Dependency chain:
    latin_lm.json             (Step 33.5 – character LM)
    combined_refine.json      (Phase 15 best_assignment)
    modifier_integrate.json   (Phase 16 modifiers)
        -> perplexity_search.json  (this step)
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
    decode_token_modifier_aware,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    PHONEME_NUCLEUS_MAP,
    PHONEME_PLACE_MAP,
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


def _folio_number(folio: str) -> int:
    """Extract numeric part from folio name (e.g. 'f1r' -> 1, 'f70v2' -> 70)."""
    return int(''.join(c for c in folio if c.isdigit()))


# ---------------------------------------------------------------------------
# Character n-gram language model
# ---------------------------------------------------------------------------

class CharNgramLM:
    """Smoothed character n-gram LM reconstructed from saved counts."""

    def __init__(
        self,
        order: int,
        alpha: float,
        counts: Dict[str, Dict[str, int]],
        vocab_size: int,
    ):
        self.order = order
        self.alpha = alpha
        self.counts = counts  # {context_str: {char: count}}
        self.vocab_size = vocab_size
        self.context_totals: Dict[str, int] = {
            ctx: sum(chars.values()) for ctx, chars in counts.items()
        }

    def bits_per_char(self, text: str) -> float:
        """Compute cross-entropy in bits per character."""
        padded = '^' * (self.order - 1) + text + '$'
        total_log = 0.0
        n = 0
        for i in range(len(padded) - self.order + 1):
            context = padded[i:i + self.order - 1]
            next_char = padded[i + self.order - 1]
            count = self.counts.get(context, {}).get(next_char, 0)
            total_count = self.context_totals.get(context, 0)
            prob = (count + self.alpha) / (total_count + self.alpha * self.vocab_size)
            total_log += math.log2(prob)
            n += 1
        return -total_log / n if n > 0 else float('inf')


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TriplePerplexityResult:
    triple_key: str
    original_syllable: str
    best_syllable: str
    original_bpc: float
    best_bpc: float
    delta_bpc: float
    changed: bool
    n_candidates_tested: int


@dataclass
class PerplexitySearchResult:
    # Search config
    n_passes: int
    n_triples_searched: int
    n_candidates_evaluated: int
    # Results
    n_changes: int
    changed_triples: List[Dict]  # TriplePerplexityResult as dicts
    best_assignment: Dict[str, str]
    # Perplexity metrics
    baseline_train_bpc: float
    optimized_train_bpc: float
    delta_train_bpc: float
    baseline_val_bpc: float
    optimized_val_bpc: float
    delta_val_bpc: float
    # Multi-objective
    baseline_dict_hit: float
    optimized_dict_hit: float
    delta_dict_hit: float
    # Verdict
    verdict: str  # 'PERPLEXITY_IMPROVED', 'NO_IMPROVEMENT', 'OVERFITTING'
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _decode_token_r3(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> str:
    """Decode a single token with R3 strategy (alter/strip/raw)."""
    # Alteration
    alt = decode_token_modifier_aware(
        token, assignment, eva_to_triple, modifier_chars, modifier_rules,
    )
    if alt.lower() in ref_word_set:
        return alt.lower()
    # Strip
    stripped = decode_token_modifier_aware(
        token, assignment, eva_to_triple, modifier_chars,
    )
    if stripped.lower() in ref_word_set:
        return stripped.lower()
    # Raw
    raw = decode_token(token, assignment, eva_to_triple)
    return raw.lower()


def _decode_all_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode all tokens using R3 strategy."""
    return [
        _decode_token_r3(t, assignment, eva_to_triple,
                         modifier_chars, modifier_rules, ref_word_set)
        for t in tokens
    ]


def _text_to_clean(decoded: List[str]) -> str:
    """Join decoded words and clean to a-z and space only."""
    text = ' '.join(decoded)
    return ''.join(c for c in text if c.isalpha() or c == ' ')


def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    """Fraction of decoded tokens that match the reference dictionary."""
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w in ref_word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# Fast-path re-decode for coordinate descent
# ---------------------------------------------------------------------------

def _fast_redecode_bpc(
    all_tokens: List[str],
    affected_indices: List[int],
    current_decoded: List[str],
    triple_key: str,
    new_syllable: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    lm: CharNgramLM,
) -> Tuple[float, List[str]]:
    """Re-decode only affected tokens and compute perplexity."""
    new_assignment = dict(assignment)
    new_assignment[triple_key] = new_syllable

    new_decoded = list(current_decoded)
    for idx in affected_indices:
        token = all_tokens[idx]
        new_decoded[idx] = _decode_token_r3(
            token, new_assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )

    clean = _text_to_clean(new_decoded)
    bpc = lm.bits_per_char(clean)
    return bpc, new_decoded


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_perplexity_search() -> None:
    """Coordinate-descent perplexity search over triple assignments."""
    t0 = time.time()
    rd = _results_dir()

    # ── 1. Load Latin LM ──────────────────────────────────────────────────
    lm_path = os.path.join(rd, 'latin_lm.json')
    if not os.path.exists(lm_path):
        print("  [SKIP] latin_lm.json not found — run Step 33.5 first.")
        return

    print("  1. Loading Latin character LM …")
    with open(lm_path) as f:
        lm_data = json.load(f)

    lm_counts = lm_data['lm_counts_fivegram']
    lm_order = lm_data.get('lm_order', 5)
    lm_alpha = lm_data.get('lm_alpha', 1.0)
    # 26 letters + space + ^ + $ = 28 (but $ and ^ are boundary, space is real)
    lm_vocab_size = 28

    lm = CharNgramLM(
        order=lm_order,
        alpha=lm_alpha,
        counts=lm_counts,
        vocab_size=lm_vocab_size,
    )
    print(f"     order={lm_order}, alpha={lm_alpha}, "
          f"vocab_size={lm_vocab_size}, "
          f"{len(lm_counts)} contexts loaded")

    # ── 2. Load assignment and modifiers ──────────────────────────────────
    print("  2. Loading assignment + modifiers …")
    with open(os.path.join(rd, 'combined_refine.json')) as f:
        refine_data = json.load(f)
    assignment = dict(refine_data['best_assignment'])
    print(f"     {len(assignment)} triple assignments")

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    print(f"     {len(modifier_chars)} modifier chars, "
          f"{len(modifier_rules)} modifier rules")

    # ── 3. Load corpus and reference dictionary ───────────────────────────
    print("  3. Loading corpus + reference dictionary …")
    corpus = load_corpus()
    eva_to_triple = build_eva_to_triple_lookup()

    ref_corpus = load_reference_corpus()
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 4. Split into train (odd folios) and validate (even folios) ───────
    print("  4. Splitting corpus into train/validate by folio parity …")
    all_tokens: List[str] = []
    token_folios: List[str] = []

    for folio, page in corpus.pages.items():
        page_tokens = page.all_tokens
        for t in page_tokens:
            all_tokens.append(t)
            token_folios.append(folio)

    n_total = len(all_tokens)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for i in range(n_total):
        fnum = _folio_number(token_folios[i])
        if fnum % 2 == 1:
            train_indices.append(i)
        else:
            val_indices.append(i)

    train_set = set(train_indices)
    val_set = set(val_indices)

    train_tokens = [all_tokens[i] for i in train_indices]
    val_tokens = [all_tokens[i] for i in val_indices]
    print(f"     {len(train_tokens)} train tokens (odd folios), "
          f"{len(val_tokens)} validate tokens (even folios)")

    # ── 5. Pre-compute triple → affected train token indices ──────────────
    print("  5. Building triple → token index map …")
    triple_to_train_indices: Dict[str, List[int]] = defaultdict(list)
    triple_freq: Counter = Counter()

    for local_idx, global_idx in enumerate(train_indices):
        token = all_tokens[global_idx]
        triples = token_to_triples(token, eva_to_triple)
        for tk in triples:
            triple_to_train_indices[tk].append(local_idx)
            triple_freq[tk] += 1

    # Order triples by frequency (most common first)
    sorted_triples = [tk for tk, _ in triple_freq.most_common()]
    print(f"     {len(sorted_triples)} active triples, "
          f"top: {sorted_triples[0]} ({triple_freq[sorted_triples[0]]} occ)")

    # ── 6. Baseline decode on train half ──────────────────────────────────
    print("  6. Baseline decode …")
    train_decoded = _decode_all_r3(
        train_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_train_clean = _text_to_clean(train_decoded)
    baseline_train_bpc = lm.bits_per_char(baseline_train_clean)
    baseline_train_dict_hit = _compute_dict_hit(train_decoded, ref_word_set)
    print(f"     Train: bpc={baseline_train_bpc:.4f}, "
          f"dict_hit={baseline_train_dict_hit:.4f}")

    # ── 7. Coordinate descent ─────────────────────────────────────────────
    max_passes = 3
    min_improvement = 0.01  # bits/char

    current_assignment = dict(assignment)
    current_decoded = list(train_decoded)
    current_bpc = baseline_train_bpc

    all_changes: List[TriplePerplexityResult] = []
    total_candidates_evaluated = 0
    total_changes = 0

    print(f"\n  7. Coordinate descent ({max_passes} passes, "
          f"{len(sorted_triples)} triples) …")

    for pass_num in range(1, max_passes + 1):
        pass_changes = 0
        print(f"\n     ── Pass {pass_num}/{max_passes} ──")

        for ti, triple_key in enumerate(sorted_triples):
            original_syl = current_assignment.get(triple_key, '?')

            # Parse triple_key → first_stroke, last_stroke, glyph_class
            parts = triple_key.split(',')
            if len(parts) != 3:
                continue
            first_stroke, last_stroke, _ = parts

            # Generate candidate syllables
            onset_candidates = PHONEME_PLACE_MAP.get(first_stroke, [])
            nucleus_candidates = PHONEME_NUCLEUS_MAP.get(last_stroke, [])

            candidates: List[str] = []
            for onset in onset_candidates:
                for nucleus in nucleus_candidates:
                    syl = onset + nucleus
                    if 2 <= len(syl) <= 3:
                        candidates.append(syl)
            # Also include pure vowels
            for nucleus in nucleus_candidates:
                if len(nucleus) >= 1 and nucleus not in candidates:
                    candidates.append(nucleus)

            # Filter: all-different (not used by another triple)
            used_by_others = set()
            for tk, syl in current_assignment.items():
                if tk != triple_key:
                    used_by_others.add(syl)
            candidates = [c for c in candidates if c not in used_by_others]

            if not candidates:
                continue

            # Always include original if not filtered
            if original_syl not in candidates and original_syl != '?':
                candidates.append(original_syl)

            # Affected indices in train_decoded (local indices)
            affected = triple_to_train_indices.get(triple_key, [])
            if not affected:
                continue

            best_syl = original_syl
            best_bpc = current_bpc
            n_tested = 0

            for candidate in candidates:
                if candidate == current_assignment.get(triple_key):
                    continue  # skip current (already have its bpc)
                n_tested += 1

                bpc, _ = _fast_redecode_bpc(
                    train_tokens, affected, current_decoded,
                    triple_key, candidate, current_assignment,
                    eva_to_triple, modifier_chars, modifier_rules,
                    ref_word_set, lm,
                )

                if bpc < best_bpc - min_improvement:
                    best_bpc = bpc
                    best_syl = candidate

            total_candidates_evaluated += n_tested

            changed = best_syl != original_syl
            result = TriplePerplexityResult(
                triple_key=triple_key,
                original_syllable=original_syl,
                best_syllable=best_syl,
                original_bpc=round(current_bpc, 6),
                best_bpc=round(best_bpc, 6),
                delta_bpc=round(best_bpc - current_bpc, 6),
                changed=changed,
                n_candidates_tested=n_tested,
            )

            if changed:
                # Accept the change: re-decode affected tokens with new syl
                current_assignment[triple_key] = best_syl
                _, new_decoded = _fast_redecode_bpc(
                    train_tokens, affected, current_decoded,
                    triple_key, best_syl, current_assignment,
                    eva_to_triple, modifier_chars, modifier_rules,
                    ref_word_set, lm,
                )
                current_decoded = new_decoded
                current_bpc = best_bpc
                pass_changes += 1
                total_changes += 1
                all_changes.append(result)

                print(f"       [{ti+1:2d}/{len(sorted_triples)}] "
                      f"{triple_key}: {original_syl} -> {best_syl}  "
                      f"bpc {result.original_bpc:.4f} -> {result.best_bpc:.4f} "
                      f"({result.delta_bpc:+.4f})")

        print(f"     Pass {pass_num}: {pass_changes} changes, "
              f"bpc={current_bpc:.4f}")

        # Convergence check
        if pass_changes == 0:
            print(f"     Converged after pass {pass_num} — no changes.")
            break

    optimized_train_bpc = current_bpc
    optimized_train_dict_hit = _compute_dict_hit(current_decoded, ref_word_set)

    print(f"\n     Train result: bpc {baseline_train_bpc:.4f} -> "
          f"{optimized_train_bpc:.4f} "
          f"(delta={optimized_train_bpc - baseline_train_bpc:+.4f})")
    print(f"     Train dict_hit: {baseline_train_dict_hit:.4f} -> "
          f"{optimized_train_dict_hit:.4f}")

    # ── 8. Validation half ────────────────────────────────────────────────
    print("\n  8. Validation on held-out even folios …")
    # Baseline on validation
    val_decoded_baseline = _decode_all_r3(
        val_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    baseline_val_clean = _text_to_clean(val_decoded_baseline)
    baseline_val_bpc = lm.bits_per_char(baseline_val_clean)
    baseline_val_dict_hit = _compute_dict_hit(val_decoded_baseline, ref_word_set)

    # Optimized on validation
    val_decoded_opt = _decode_all_r3(
        val_tokens, current_assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    optimized_val_clean = _text_to_clean(val_decoded_opt)
    optimized_val_bpc = lm.bits_per_char(optimized_val_clean)
    optimized_val_dict_hit = _compute_dict_hit(val_decoded_opt, ref_word_set)

    delta_val_bpc = optimized_val_bpc - baseline_val_bpc

    print(f"     Val bpc: {baseline_val_bpc:.4f} -> {optimized_val_bpc:.4f} "
          f"(delta={delta_val_bpc:+.4f})")
    print(f"     Val dict_hit: {baseline_val_dict_hit:.4f} -> "
          f"{optimized_val_dict_hit:.4f}")

    # ── 9. Verdict ────────────────────────────────────────────────────────
    delta_train_bpc = optimized_train_bpc - baseline_train_bpc
    delta_dict_hit = optimized_val_dict_hit - baseline_val_dict_hit

    if total_changes == 0:
        verdict = 'NO_IMPROVEMENT'
    elif delta_val_bpc < -min_improvement:
        verdict = 'PERPLEXITY_IMPROVED'
    elif delta_val_bpc > min_improvement:
        verdict = 'OVERFITTING'
    else:
        # Val perplexity roughly unchanged
        if delta_dict_hit > 0.005:
            verdict = 'PERPLEXITY_IMPROVED'
        elif total_changes > 0 and delta_train_bpc < -min_improvement:
            verdict = 'OVERFITTING'
        else:
            verdict = 'NO_IMPROVEMENT'

    print(f"\n     Verdict: {verdict}")
    print(f"     Total changes: {total_changes}, "
          f"candidates evaluated: {total_candidates_evaluated}")

    # ── 10. Build and save result ─────────────────────────────────────────
    elapsed = time.time() - t0
    n_passes_done = min(pass_num, max_passes) if 'pass_num' in dir() else max_passes

    result = PerplexitySearchResult(
        n_passes=n_passes_done,
        n_triples_searched=len(sorted_triples),
        n_candidates_evaluated=total_candidates_evaluated,
        n_changes=total_changes,
        changed_triples=[_convert(asdict(c)) for c in all_changes],
        best_assignment=current_assignment,
        baseline_train_bpc=round(baseline_train_bpc, 6),
        optimized_train_bpc=round(optimized_train_bpc, 6),
        delta_train_bpc=round(delta_train_bpc, 6),
        baseline_val_bpc=round(baseline_val_bpc, 6),
        optimized_val_bpc=round(optimized_val_bpc, 6),
        delta_val_bpc=round(delta_val_bpc, 6),
        baseline_dict_hit=round(baseline_val_dict_hit, 6),
        optimized_dict_hit=round(optimized_val_dict_hit, 6),
        delta_dict_hit=round(delta_dict_hit, 6),
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = os.path.join(rd, 'perplexity_search.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved -> {out_path}  ({elapsed:.1f}s)")
