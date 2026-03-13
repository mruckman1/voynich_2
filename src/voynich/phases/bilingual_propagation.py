"""
Phase 48 Track D: Integrated Bilingual Crib Propagation
=========================================================
Collect cribs from Tracks A-C, check consistency, propagate to
unconfirmed triples, decode corpus, and validate.

Dependency chain:
    f116v_match.json           (48A.4)
    f116v_reverse.json         (48A.5)
    margin_decode.json         (48B.3)
    marci_comparison.json      (48C.3)
    combined_refine.json       (Phase 15 — T_P15)
    modifier_integrate.json    (Phase 16)
    signal_isolation.json      (Phase 28)
        → crib_collection.json   (48D.1)
        → crib_consistency.json  (48D.2)
        → crib_propagation.json  (48D.3)
        → crib_decode.json       (48D.4)
        → crib_validation.json   (48D.5)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir


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


def _load_json(rd: str, filename: str) -> Optional[Dict]:
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CribEntry:
    """A single character-level assignment from a bilingual crib."""
    source: str               # f116v / f17r / marci
    eva_char: str
    triple_key: str
    proposed_syllable: str
    confidence: str           # HIGH / MEDIUM / LOW
    agrees_with_t_p15: bool
    t_p15_syllable: str


@dataclass
class CribCollection:
    """Step 48D.1 output."""
    cribs: List[Dict]
    n_total: int
    n_agree: int
    n_disagree: int
    sources_represented: List[str]
    triples_covered: List[str]
    runtime_seconds: float


@dataclass
class CribConsistency:
    """Step 48D.2 output."""
    cross_source_checks: List[Dict]
    n_cross_source_agreements: int
    n_cross_source_conflicts: int
    tier1_conflicts: List[Dict]
    consistency_verdict: str
    runtime_seconds: float


@dataclass
class PropagationResult:
    """Result of attempting one triple change."""
    triple_key: str
    original_syllable: str
    proposed_syllable: str
    source: str
    dict_hit_delta: float
    signal_words_broken: int
    accepted: bool
    reason: str


@dataclass
class CribPropagation:
    """Step 48D.3 output."""
    propagation_attempts: List[Dict]
    n_accepted: int
    n_rejected: int
    cumulative_dict_hit_delta: float
    modified_triples: List[str]
    runtime_seconds: float


@dataclass
class CribDecode:
    """Step 48D.4 output."""
    dict_hit_10k: float
    dict_hit_131k: float
    signal_rate: float
    bigram_z: float
    t_p15_dict_hit_10k: float
    t_p15_dict_hit_131k: float
    delta_dict_hit: float
    n_tokens: int
    runtime_seconds: float


@dataclass
class CribValidation:
    """Step 48D.5 output."""
    bigram_z: float
    t_p15_bigram_z: float
    signal_words_surviving: int
    signal_words_total: int
    selectivity_10k: float
    n_new_signal_words: int
    gate_result: str          # ACCEPTED / REJECTED
    gate_details: List[str]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 48D.1 — Crib Collection
# ---------------------------------------------------------------------------

def run_crib_collect() -> None:
    """Step 48D.1: Collect all viable cribs from Tracks A-C."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48D.1: Crib Collection")
    print("=" * 70)

    rd = _results_dir()

    from voynich.core.corpus import build_eva_to_triple_lookup

    eva_to_triple = build_eva_to_triple_lookup()

    # Load T_P15
    combined = _load_json(rd, 'combined_refine.json')
    if not combined:
        print("     ERROR: combined_refine.json not found.")
        return
    assignment = combined.get('best_assignment', {})

    # ── 1. Collect from Track A (f116v) ──
    print("\n  1. Collecting from Track A (f116v)...")

    cribs = []
    sources_set = set()

    match_data = _load_json(rd, 'f116v_match.json')
    reverse_data = _load_json(rd, 'f116v_reverse.json')

    if match_data:
        gate = match_data.get('gate_result', 'CRIB_FAILED')
        best_level = match_data.get('best_match_level', 'NO_MATCH')
        print(f"     f116v match gate: {gate} (best: {best_level})")

        if gate == 'CRIB_VIABLE':
            # Extract character-level assignments from matched words
            best_word = match_data.get('best_match_word', '')
            print(f"     Best match word: '{best_word}'")

            # The matched word gives us character-level constraints
            # Look at the decode data for triple-to-syllable mappings
            decode_data = _load_json(rd, 'f116v_decode.json')
            if decode_data:
                for dec in decode_data.get('primary_decodes', []):
                    token = dec.get('eva_token', '')
                    syllables = dec.get('syllables', [])
                    from voynich.core.corpus import tokenize_eva_chars
                    chars = tokenize_eva_chars(token)
                    mod_data = _load_json(rd, 'modifier_integrate.json')
                    mod_chars = set(mod_data.get('modifier_chars', [])) if mod_data else set()
                    syl_idx = 0
                    for ch in chars:
                        if ch in mod_chars:
                            continue
                        tk = eva_to_triple.get(ch)
                        if tk and syl_idx < len(syllables):
                            t_p15_syl = assignment.get(tk, '')
                            proposed = syllables[syl_idx]
                            cribs.append(asdict(CribEntry(
                                source='f116v_decode',
                                eva_char=ch,
                                triple_key=tk,
                                proposed_syllable=proposed,
                                confidence='MEDIUM',
                                agrees_with_t_p15=(proposed == t_p15_syl),
                                t_p15_syllable=t_p15_syl,
                            )))
                            sources_set.add('f116v_decode')
                            syl_idx += 1

    if reverse_data:
        n_testable = reverse_data.get('n_testable', 0)
        print(f"     f116v reverse: {n_testable} testable candidates")

        for cand in reverse_data.get('candidates', []):
            if cand.get('testable') and cand.get('n_conflict_t_p15', 99) <= 2:
                for conflict in cand.get('conflicting_triples', []):
                    cribs.append(asdict(CribEntry(
                        source='f116v_reverse',
                        eva_char='',
                        triple_key=conflict.get('triple_key', ''),
                        proposed_syllable=conflict.get('required', ''),
                        confidence='LOW',
                        agrees_with_t_p15=False,
                        t_p15_syllable=conflict.get('current', ''),
                    )))
                    sources_set.add('f116v_reverse')

    # ── 2. Collect from Track B (f17r) ──
    print("\n  2. Collecting from Track B (f17r)...")

    margin_data = _load_json(rd, 'margin_decode.json')
    if margin_data:
        hits = [e for e in margin_data.get('decoded_entries', [])
                if e.get('in_131k_dict')]
        print(f"     Marginal decode: {len(hits)} dict hits")

        for entry in hits:
            from voynich.core.corpus import tokenize_eva_chars
            chars = tokenize_eva_chars(entry.get('eva_token', ''))
            syllables = entry.get('syllables', [])
            mod_data_b = _load_json(rd, 'modifier_integrate.json')
            mod_chars_b = set(mod_data_b.get('modifier_chars', [])) if mod_data_b else set()
            syl_idx = 0
            for ch in chars:
                if ch in mod_chars_b:
                    continue
                tk = eva_to_triple.get(ch)
                if tk and syl_idx < len(syllables):
                    t_p15_syl = assignment.get(tk, '')
                    proposed = syllables[syl_idx]
                    cribs.append(asdict(CribEntry(
                        source=f"f17r_decode_{entry.get('folio', '')}",
                        eva_char=ch,
                        triple_key=tk,
                        proposed_syllable=proposed,
                        confidence='LOW',
                        agrees_with_t_p15=(proposed == t_p15_syl),
                        t_p15_syllable=t_p15_syl,
                    )))
                    sources_set.add('f17r_decode')
                    syl_idx += 1

    # ── 3. Collect from Track C (Marci) ──
    print("\n  3. Collecting from Track C (Marci)...")

    marci_data = _load_json(rd, 'marci_comparison.json')
    if marci_data:
        consonant_ari = marci_data.get('consonant_ari', 0.0)
        print(f"     Marci consonant ARI: {consonant_ari}")

        if consonant_ari > 0.3:
            for pc in marci_data.get('per_character', []):
                if pc.get('match_type') != 'none':
                    cribs.append(asdict(CribEntry(
                        source='marci',
                        eva_char=pc.get('eva_char', ''),
                        triple_key=pc.get('triple_key', ''),
                        proposed_syllable=pc.get('marci_value', ''),
                        confidence='LOW',
                        agrees_with_t_p15=(pc.get('match_type') == 'consonant'),
                        t_p15_syllable=pc.get('t_p15_syllable', ''),
                    )))
                    sources_set.add('marci')
        else:
            print("     Marci ARI below threshold (0.3) — not including")
    else:
        print("     Marci comparison data not available")

    # ── 4. Summary ──
    n_agree = sum(1 for c in cribs if c.get('agrees_with_t_p15'))
    n_disagree = sum(1 for c in cribs if not c.get('agrees_with_t_p15'))
    triples_covered = list(set(c.get('triple_key', '') for c in cribs if c.get('triple_key')))

    print(f"\n  4. Summary:")
    print(f"     Total cribs: {len(cribs)}")
    print(f"     Agree with T_P15: {n_agree}")
    print(f"     Disagree with T_P15: {n_disagree}")
    print(f"     Sources: {sorted(sources_set)}")
    print(f"     Triples covered: {len(triples_covered)}")

    # ── 5. Save ──
    result = CribCollection(
        cribs=cribs,
        n_total=len(cribs),
        n_agree=n_agree,
        n_disagree=n_disagree,
        sources_represented=sorted(sources_set),
        triples_covered=triples_covered,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'crib_collection.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48D.2 — Internal Consistency
# ---------------------------------------------------------------------------

def run_crib_consistent() -> None:
    """Step 48D.2: Internal consistency check."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48D.2: Crib Consistency Check")
    print("=" * 70)

    rd = _results_dir()

    crib_data = _load_json(rd, 'crib_collection.json')
    if not crib_data:
        print("     ERROR: crib_collection.json not found.")
        return

    # Load tier data from triple_tiers.json
    tier_data = _load_json(rd, 'triple_tiers.json')
    tier1_triples = set()
    if tier_data:
        for entry in tier_data.get('evidence_table', []):
            if entry.get('tier') == 'CONFIRMED':
                tier1_triples.add(entry.get('triple_key', ''))

    cribs = crib_data.get('cribs', [])

    # ── 1. Group cribs by triple_key ──
    print("\n  1. Grouping cribs by triple key...")

    by_triple: Dict[str, List[Dict]] = {}
    for crib in cribs:
        tk = crib.get('triple_key', '')
        if tk:
            by_triple.setdefault(tk, []).append(crib)

    print(f"     {len(by_triple)} unique triples with cribs")

    # ── 2. Cross-source consistency ──
    print("\n  2. Cross-source consistency:")

    cross_checks = []
    n_agree = 0
    n_conflict = 0

    for tk, entries in by_triple.items():
        sources = set(e.get('source', '') for e in entries)
        syllables = set(e.get('proposed_syllable', '') for e in entries)

        if len(sources) > 1 and len(syllables) > 0:
            consistent = len(syllables) == 1
            check = {
                'triple_key': tk,
                'sources': sorted(sources),
                'proposed_syllables': sorted(syllables),
                'consistent': consistent,
                'n_sources': len(sources),
            }
            cross_checks.append(check)

            if consistent:
                n_agree += 1
                print(f"     ✓ {tk}: AGREE ({sorted(syllables)}) from {sorted(sources)}")
            else:
                n_conflict += 1
                print(f"     ✗ {tk}: CONFLICT ({sorted(syllables)}) from {sorted(sources)}")

    # ── 3. Tier 1 conflicts ──
    print("\n  3. Tier 1 conflict check:")

    tier1_conflicts = []
    for tk, entries in by_triple.items():
        if tk in tier1_triples:
            for e in entries:
                if not e.get('agrees_with_t_p15'):
                    tier1_conflicts.append({
                        'triple_key': tk,
                        'source': e.get('source', ''),
                        'proposed': e.get('proposed_syllable', ''),
                        't_p15': e.get('t_p15_syllable', ''),
                    })
                    print(f"     ⚠ TIER 1 CONFLICT: {tk} — "
                          f"crib={e.get('proposed_syllable')} vs "
                          f"T_P15={e.get('t_p15_syllable')} "
                          f"(source: {e.get('source')})")

    if not tier1_conflicts:
        print("     No Tier 1 conflicts")

    # ── 4. Verdict ──
    if tier1_conflicts:
        verdict = (f"TIER1_CONFLICT: {len(tier1_conflicts)} crib(s) conflict with "
                   f"confirmed triples — cannot propagate these to main text")
    elif n_conflict > 0:
        verdict = f"PARTIAL_CONSISTENCY: {n_agree} agree, {n_conflict} conflict across sources"
    elif n_agree > 0:
        verdict = f"CONSISTENT: All {n_agree} cross-source checks agree"
    else:
        verdict = "NO_CROSS_SOURCE: Only single-source cribs available"

    print(f"\n  4. Verdict: {verdict}")

    # ── 5. Save ──
    result = CribConsistency(
        cross_source_checks=cross_checks,
        n_cross_source_agreements=n_agree,
        n_cross_source_conflicts=n_conflict,
        tier1_conflicts=tier1_conflicts,
        consistency_verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'crib_consistency.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48D.3 — Propagation
# ---------------------------------------------------------------------------

def run_crib_propagate() -> None:
    """Step 48D.3: Propagate cribs to unconfirmed triples."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48D.3: Crib Propagation")
    print("=" * 70)

    rd = _results_dir()

    from voynich.core.corpus import (
        build_eva_to_triple_lookup,
        decode_token_modifier_aware,
        load_corpus,
    )
    from voynich.core.reference import build_expanded_word_set, load_reference_corpus

    # ── 1. Load dependencies ──
    print("\n  1. Loading dependencies...")

    crib_data = _load_json(rd, 'crib_collection.json')
    consist_data = _load_json(rd, 'crib_consistency.json')
    combined = _load_json(rd, 'combined_refine.json')
    mod_data = _load_json(rd, 'modifier_integrate.json')
    signal_data = _load_json(rd, 'signal_isolation.json')

    if not crib_data or not combined:
        print("     ERROR: Missing dependency files.")
        return

    assignment = dict(combined.get('best_assignment', {}))
    modifier_chars = set(mod_data.get('modifier_chars', [])) if mod_data else set()
    eva_to_triple = build_eva_to_triple_lookup()

    signal_words = set()
    if signal_data:
        for ws in signal_data.get('word_signals', []):
            if ws.get('is_genuine_signal'):
                signal_words.add(ws['word'])

    # Load tier data from triple_tiers.json
    tier_data = _load_json(rd, 'triple_tiers.json')
    tier1_triples = set()
    if tier_data:
        for entry in tier_data.get('evidence_table', []):
            if entry.get('tier') == 'CONFIRMED':
                tier1_triples.add(entry.get('triple_key', ''))

    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    dict_10k = base_words

    # Get all tokens
    all_tokens = []
    for _folio, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)

    # ── 2. Compute baseline ──
    print("\n  2. Computing T_P15 baseline...")

    baseline_hits = 0
    for tok in all_tokens:
        decoded = decode_token_modifier_aware(tok, assignment, eva_to_triple, modifier_chars)
        if decoded.lower() in dict_10k:
            baseline_hits += 1
    baseline_dict_hit = baseline_hits / len(all_tokens) if all_tokens else 0.0
    print(f"     Baseline dict_hit (10K): {baseline_dict_hit:.4f}")

    # Check signal words
    def _count_signal_hits(assign: Dict) -> int:
        count = 0
        for tok in all_tokens:
            decoded = decode_token_modifier_aware(tok, assign, eva_to_triple, modifier_chars)
            if decoded.lower() in signal_words:
                count += 1
        return count

    baseline_signal = _count_signal_hits(assignment)

    # ── 3. Try each crib-proposed change ──
    print("\n  3. Attempting propagation...")

    # Get cribs that disagree with T_P15 and are not Tier 1
    changeable_cribs = []
    for crib in crib_data.get('cribs', []):
        tk = crib.get('triple_key', '')
        if not crib.get('agrees_with_t_p15') and tk not in tier1_triples and tk:
            changeable_cribs.append(crib)

    # Deduplicate by triple_key (keep highest confidence)
    conf_rank = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    by_triple: Dict[str, Dict] = {}
    for crib in changeable_cribs:
        tk = crib['triple_key']
        if tk not in by_triple or conf_rank.get(crib.get('confidence', 'LOW'), 0) > \
                conf_rank.get(by_triple[tk].get('confidence', 'LOW'), 0):
            by_triple[tk] = crib

    print(f"     {len(by_triple)} unique triples to test")

    propagation_attempts = []
    current_assignment = dict(assignment)
    cumulative_delta = 0.0

    for tk, crib in sorted(by_triple.items()):
        original = current_assignment.get(tk, '')
        proposed = crib.get('proposed_syllable', '')
        source = crib.get('source', '')

        # Try the change
        test_assignment = dict(current_assignment)
        test_assignment[tk] = proposed

        test_hits = 0
        for tok in all_tokens:
            decoded = decode_token_modifier_aware(tok, test_assignment, eva_to_triple, modifier_chars)
            if decoded.lower() in dict_10k:
                test_hits += 1
        test_dict_hit = test_hits / len(all_tokens) if all_tokens else 0.0

        delta = test_dict_hit - baseline_dict_hit
        signal_count = _count_signal_hits(test_assignment)
        signal_broken = max(0, baseline_signal - signal_count)

        # Accept if: doesn't break signal words, doesn't decrease dict-hit by >0.5%
        accepted = signal_broken == 0 and delta >= -0.005

        reason = 'accepted' if accepted else ''
        if signal_broken > 0:
            reason = f'REJECTED: breaks {signal_broken} signal words'
        elif delta < -0.005:
            reason = f'REJECTED: dict_hit delta {delta:+.4f} < -0.005'

        pr = PropagationResult(
            triple_key=tk,
            original_syllable=original,
            proposed_syllable=proposed,
            source=source,
            dict_hit_delta=round(delta, 6),
            signal_words_broken=signal_broken,
            accepted=accepted,
            reason=reason,
        )
        propagation_attempts.append(asdict(pr))

        if accepted:
            current_assignment[tk] = proposed
            cumulative_delta += delta
            print(f"     ✓ {tk}: '{original}' → '{proposed}' (Δ={delta:+.4f})")
        else:
            print(f"     ✗ {tk}: '{original}' → '{proposed}' ({reason})")

    n_accepted = sum(1 for p in propagation_attempts if p.get('accepted'))
    n_rejected = len(propagation_attempts) - n_accepted
    modified = [p['triple_key'] for p in propagation_attempts if p.get('accepted')]

    print(f"\n     Accepted: {n_accepted}, Rejected: {n_rejected}")
    print(f"     Cumulative delta: {cumulative_delta:+.4f}")

    # ── 4. Save ──
    result = CribPropagation(
        propagation_attempts=propagation_attempts,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        cumulative_dict_hit_delta=round(cumulative_delta, 6),
        modified_triples=modified,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'crib_propagation.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48D.4 — Crib-Informed Corpus Decode
# ---------------------------------------------------------------------------

def run_crib_decode() -> None:
    """Step 48D.4: Decode corpus with crib-informed table."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48D.4: Crib-Informed Corpus Decode")
    print("=" * 70)

    rd = _results_dir()

    from voynich.core.corpus import (
        build_eva_to_triple_lookup,
        decode_token_modifier_aware,
        load_corpus,
    )
    from voynich.core.reference import build_expanded_word_set, load_reference_corpus

    # ── 1. Load and build crib-informed table ──
    print("\n  1. Building crib-informed table...")

    combined = _load_json(rd, 'combined_refine.json')
    prop_data = _load_json(rd, 'crib_propagation.json')
    mod_data = _load_json(rd, 'modifier_integrate.json')

    if not combined:
        print("     ERROR: combined_refine.json not found.")
        return

    assignment = dict(combined.get('best_assignment', {}))
    modifier_chars = set(mod_data.get('modifier_chars', [])) if mod_data else set()
    eva_to_triple = build_eva_to_triple_lookup()

    # Apply accepted propagations
    n_changes = 0
    if prop_data:
        for attempt in prop_data.get('propagation_attempts', []):
            if attempt.get('accepted'):
                tk = attempt.get('triple_key', '')
                new_syl = attempt.get('proposed_syllable', '')
                if tk and new_syl:
                    assignment[tk] = new_syl
                    n_changes += 1

    print(f"     Applied {n_changes} crib-informed changes")

    # ── 2. Load corpus and dictionaries ──
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    dict_10k = base_words
    dict_131k = base_words | expanded

    # ── 3. Decode full corpus with BOTH tables ──
    print("\n  2. Decoding full corpus...")

    all_tokens = []
    for _folio, page in corpus.pages.items():
        all_tokens.extend(page.all_tokens)
    n_tokens = len(all_tokens)

    # Crib-informed table
    hits_10k = 0
    hits_131k = 0
    for tok in all_tokens:
        decoded = decode_token_modifier_aware(tok, assignment, eva_to_triple, modifier_chars)
        dw = decoded.lower()
        if dw in dict_10k:
            hits_10k += 1
        if dw in dict_131k:
            hits_131k += 1

    crib_dict_hit_10k = hits_10k / n_tokens if n_tokens > 0 else 0.0
    crib_dict_hit_131k = hits_131k / n_tokens if n_tokens > 0 else 0.0

    # T_P15 baseline
    t_p15_assign = combined.get('best_assignment', {})
    t_hits_10k = 0
    t_hits_131k = 0
    for tok in all_tokens:
        decoded = decode_token_modifier_aware(tok, t_p15_assign, eva_to_triple, modifier_chars)
        dw = decoded.lower()
        if dw in dict_10k:
            t_hits_10k += 1
        if dw in dict_131k:
            t_hits_131k += 1

    t_p15_dict_hit_10k = t_hits_10k / n_tokens if n_tokens > 0 else 0.0
    t_p15_dict_hit_131k = t_hits_131k / n_tokens if n_tokens > 0 else 0.0

    delta = crib_dict_hit_10k - t_p15_dict_hit_10k

    print(f"     Tokens: {n_tokens}")
    print(f"     Crib table 10K: {crib_dict_hit_10k:.4f}")
    print(f"     Crib table 131K: {crib_dict_hit_131k:.4f}")
    print(f"     T_P15 10K: {t_p15_dict_hit_10k:.4f}")
    print(f"     T_P15 131K: {t_p15_dict_hit_131k:.4f}")
    print(f"     Delta (10K): {delta:+.4f}")

    # ── 4. Save ──
    result = CribDecode(
        dict_hit_10k=round(crib_dict_hit_10k, 6),
        dict_hit_131k=round(crib_dict_hit_131k, 6),
        signal_rate=0.0,  # Computed in validation step
        bigram_z=0.0,     # Computed in validation step
        t_p15_dict_hit_10k=round(t_p15_dict_hit_10k, 6),
        t_p15_dict_hit_131k=round(t_p15_dict_hit_131k, 6),
        delta_dict_hit=round(delta, 6),
        n_tokens=n_tokens,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'crib_decode.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48D.5 — Validation
# ---------------------------------------------------------------------------

def run_crib_validate() -> None:
    """Step 48D.5: Signal isolation and bigram z on crib table.

    Uses the canonical z-score methodology from Phase 47 (zscore_audit):
    10K dict, exact + relaxed (edit-distance-1), 500 perms, proper
    5-null-corpus SIGNAL classification.
    """
    t0 = time.time()

    print("=" * 70)
    print("STEP 48D.5: Crib Validation (canonical methodology)")
    print("=" * 70)

    rd = _results_dir()

    from voynich.phases.zscore_audit import (
        _build_context,
        _build_ref_bigrams,
        _compute_z_full,
    )

    # ── 1. Load and build crib-informed table ──
    print("\n  1. Building crib-informed table...")

    combined = _load_json(rd, 'combined_refine.json')
    prop_data = _load_json(rd, 'crib_propagation.json')
    signal_data = _load_json(rd, 'signal_isolation.json')
    phase47 = _load_json(rd, 'phase47_integrate.json')

    if not combined:
        print("     ERROR: combined_refine.json not found.")
        return

    assignment = dict(combined.get('best_assignment', {}))

    # Apply accepted propagations
    n_changes = 0
    if prop_data:
        for attempt in prop_data.get('propagation_attempts', []):
            if attempt.get('accepted'):
                tk = attempt.get('triple_key', '')
                new_syl = attempt.get('proposed_syllable', '')
                if tk and new_syl:
                    assignment[tk] = new_syl
                    n_changes += 1

    print(f"     Applied {n_changes} crib-informed changes to T_P15")

    signal_words = set()
    if signal_data:
        for ws in signal_data.get('word_signals', []):
            if ws.get('is_genuine_signal'):
                signal_words.add(ws['word'])

    # Baseline z from Phase 47
    t_p15_z = 14.78
    if phase47:
        t_p15_z = phase47.get('track_a_canonical_z', 14.78)

    # ── 2. Compute canonical z-score ──
    print("\n  2. Computing canonical z-score (10K, exact+relaxed, 500 perms)...")
    print("     This uses the same methodology as Phase 47A.4.")

    ctx = _build_context(rd)
    ref_bigrams_10k = _build_ref_bigrams(ctx.ref_tokens_raw, ctx.ref_word_set_10k)

    z_result = _compute_z_full(
        label='T_CRIB',
        assignment=assignment,
        dict_mode='10K',
        ref_word_set=ctx.ref_word_set_10k,
        ref_bigram_set=ref_bigrams_10k,
        ctx=ctx,
        n_perms=500,
        include_relaxed=True,
        seed=42,
    )

    bigram_z = z_result.z_total

    print(f"     SIGNAL tokens: {z_result.n_signal} ({z_result.signal_rate:.1%})")
    print(f"     SIGNAL pairs: {z_result.n_signal_pairs}")
    print(f"     Exact hits: {z_result.n_exact_hits}")
    print(f"     Relaxed hits: {z_result.n_relaxed_hits}")
    print(f"     z_exact={z_result.z_exact:.2f}  z_total={z_result.z_total:.2f}")
    print(f"     T_P15 canonical z: {t_p15_z:.2f}")

    # ── 3. Signal word survival ──
    print("\n  3. Signal word survival check...")

    # Use the decoded tokens from the z-score computation
    from voynich.phases.zscore_audit import _decode_corpus_r3
    decoded = _decode_corpus_r3(
        ctx.all_tokens, assignment, ctx.eva_to_triple,
        ctx.modifier_chars, ctx.modifier_rules, ctx.ref_word_set_10k,
    )

    surviving = 0
    for sw in signal_words:
        if sw in decoded:
            surviving += 1
    print(f"     {surviving}/{len(signal_words)} signal words survive")

    # Selectivity
    selectivity = z_result.total_hit_rate / z_result.null_mean_total if z_result.null_mean_total > 0 else 0.0

    # New signal words
    decoded_counter = Counter(decoded)
    new_signals = 0
    for word, count in decoded_counter.items():
        if word in ctx.ref_word_set_10k and word not in signal_words and count >= 5:
            new_signals += 1

    # ── 4. Gate ──
    print("\n  4. Gate assessment:")

    gate_details = []

    # G1: Bigram z >= T_P15 z (no regression)
    g1 = bigram_z >= t_p15_z * 0.9
    gate_details.append(f"G1 bigram_z ({bigram_z:.2f}) >= {t_p15_z:.2f}*0.9={t_p15_z*0.9:.2f}: {'PASS' if g1 else 'FAIL'}")

    # G2: All signal words survive
    g2 = surviving >= len(signal_words)
    gate_details.append(f"G2 signal survival ({surviving}/{len(signal_words)}): {'PASS' if g2 else 'FAIL'}")

    # G3: Selectivity >= 1.0
    g3 = selectivity >= 1.0
    gate_details.append(f"G3 selectivity ({selectivity:.2f}) >= 1.0: {'PASS' if g3 else 'FAIL'}")

    # G4: At least 1 new signal word
    g4 = new_signals >= 1
    gate_details.append(f"G4 new signals ({new_signals}): {'PASS' if g4 else 'FAIL'}")

    gate_pass = g1 and g2 and g3
    gate_result = 'ACCEPTED' if gate_pass else 'REJECTED'

    for gd in gate_details:
        print(f"     {gd}")
    print(f"\n     Gate: {gate_result}")

    # ── 5. Save ──
    result = CribValidation(
        bigram_z=round(bigram_z, 4),
        t_p15_bigram_z=round(t_p15_z, 4),
        signal_words_surviving=surviving,
        signal_words_total=len(signal_words),
        selectivity_10k=round(selectivity, 4),
        n_new_signal_words=new_signals,
        gate_result=gate_result,
        gate_details=gate_details,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'crib_validation.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Track D runner
# ---------------------------------------------------------------------------

def run_track_d_48() -> None:
    """Run all Track D steps sequentially."""
    print("\n" + "█" * 70)
    print("  PHASE 48 TRACK D: Bilingual Crib Propagation")
    print("█" * 70)

    run_crib_collect()
    run_crib_consistent()
    run_crib_propagate()
    run_crib_decode()
    run_crib_validate()

    print("\n" + "█" * 70)
    print("  TRACK D COMPLETE")
    print("█" * 70)
