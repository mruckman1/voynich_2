"""
Step 43.4 – Inversion Decode
===============================
Invert the best encoding table from Step 43.3 to produce a decoding table,
and decode the Voynich corpus.

Dependency chain:
    results/encoding_search.json     (Step 43.3: best encoding table)
    results/combined_refine.json     (Phase 15: original assignment)
    results/modifier_integrate.json  (Phase 16: modifier chars)
    results/null_corpus.json         (Phase 17: null seeds)
    data/corpus/                     (EVA transcription)
        → inversion_decode.json      (this step)
"""

import json
import os
import random
import time

import numpy as np
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    load_corpus,
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
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
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InversionDecodeResult:
    # Inverted table
    inverted_assignment: Dict[str, str]  # triple_key → syllable
    n_assigned_triples: int
    n_unassigned_triples: int
    # Comparison with Phase 15
    n_agreements: int
    n_consonant_only: int
    n_disagreements: int
    agreement_rate: float
    disagreements: List[Dict]
    # Decoding results
    n_tokens_decoded: int
    dict_hit_rate: float
    dict_hit_count: int
    decoded_sample: List[str]  # first 50 decoded tokens
    # Null comparison
    null_dict_hit_mean: float
    null_dict_hit_std: float
    selectivity_ratio: float
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helper: reconstruct modifier rules from modifier_integrate.json
# ---------------------------------------------------------------------------

def _reconstruct_modifier_rules(data: Dict) -> Tuple[Set[str], Dict[str, str]]:
    """Extract modifier chars and modifier rules from modifier_integrate.json."""
    modifier_chars = set(data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')
    return modifier_chars, modifier_rules


# ---------------------------------------------------------------------------
# Helper: decode a full corpus with R3 (combined) strategy
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    expanded_set: set,
) -> List[str]:
    """Decode tokens using the R3 combined strategy from Phase 16.

    For each token: try alteration first, then stripping, then plain decode.
    Pick whichever gets a dictionary hit.
    """
    from voynich.phases.csp_solver import decode_token

    decoded: List[str] = []
    for token in tokens:
        # R2: alteration (modifier_rules applied)
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in expanded_set:
            decoded.append(alt)
            continue

        # R1: stripping (modifiers silently dropped)
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in expanded_set:
            decoded.append(stripped)
            continue

        # Fallback: plain decode (no modifier awareness)
        plain = decode_token(token, assignment, eva_to_triple)
        decoded.append(plain)

    return decoded


# ---------------------------------------------------------------------------
# Helper: compute dict hit rate
# ---------------------------------------------------------------------------

def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# Helper: build syllable frequency from reference corpus
# ---------------------------------------------------------------------------

def _build_syllable_frequencies(
    base_words: set,
) -> Dict[str, int]:
    """Count approximate syllable frequency by treating each 2-char CV
    substring of reference words as a 'syllable'."""
    freq: Counter = Counter()
    for word in base_words:
        w = word.lower()
        # Count all 2-char substrings as rough syllable proxies
        for i in range(0, len(w) - 1):
            freq[w[i:i+2]] += 1
    return dict(freq)


# ---------------------------------------------------------------------------
# Null corpus generation (with fallback)
# ---------------------------------------------------------------------------

def _generate_null_corpora(
    tokens: List[str],
    seeds: List[int],
    n_null: int,
) -> List[List[str]]:
    """Generate null corpora using the same method as null_corpus.py.

    Falls back to shuffling the real token order if the import fails.
    """
    try:
        from voynich.phases.null_corpus import (
            _build_eva_bigram_model,
            _generate_null_corpus,
        )

        bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(tokens)
        null_corpora: List[List[str]] = []
        for i in range(n_null):
            seed = seeds[i] if i < len(seeds) else 200 + i
            null_tokens = _generate_null_corpus(
                bigram_probs, initial_probs, token_lengths,
                n_tokens=len(tokens), seed=seed,
            )
            null_corpora.append(null_tokens)
        return null_corpora

    except (ImportError, Exception):
        # Fallback: shuffle real token order
        null_corpora = []
        for i in range(n_null):
            seed = seeds[i] if i < len(seeds) else 200 + i
            rng = random.Random(seed)
            shuffled = list(tokens)
            rng.shuffle(shuffled)
            null_corpora.append(shuffled)
        return null_corpora


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_inversion_decode() -> None:
    """Step 43.4: Invert encoding table and decode the Voynich corpus."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.4: Inversion Decode")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load encoding_search.json (Step 43.3) ──
    print("\n  1. Loading encoding search results …")
    enc_path = os.path.join(rd, 'encoding_search.json')
    enc_data = _safe_load(enc_path)
    if not enc_data:
        raise FileNotFoundError(
            f"encoding_search.json not found at {enc_path}. "
            f"Run Step 43.3 (encoding-search) first."
        )
    best_syl_to_triple = enc_data.get('best_syllable_to_triple', {})
    print(f"     Encoding table: {len(best_syl_to_triple)} syllable→triple mappings")
    if not best_syl_to_triple:
        raise ValueError("encoding_search.json has empty best_syllable_to_triple")

    # ── 2. Invert the table: syllable→triple → triple→syllable ──
    print("\n  2. Inverting encoding table …")

    # Build syllable frequency from reference corpus for collision resolution
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()
    syl_freq = _build_syllable_frequencies(base_words)

    # Group by triple to detect collisions
    triple_to_syllables: Dict[str, List[str]] = {}
    for syl, triple_key in best_syl_to_triple.items():
        triple_to_syllables.setdefault(triple_key, []).append(syl)

    # Resolve collisions: keep the most frequent syllable
    triple_to_syllable: Dict[str, str] = {}
    collision_count = 0
    for triple_key, syls in triple_to_syllables.items():
        if len(syls) == 1:
            triple_to_syllable[triple_key] = syls[0]
        else:
            collision_count += 1
            # Pick the syllable with highest reference frequency
            best_syl = max(syls, key=lambda s: syl_freq.get(s, 0))
            triple_to_syllable[triple_key] = best_syl
            print(f"     Collision on {triple_key}: {syls} → picked '{best_syl}' "
                  f"(freq={syl_freq.get(best_syl, 0)})")

    print(f"     Inverted table: {len(triple_to_syllable)} triple→syllable mappings")
    print(f"     Collisions resolved: {collision_count}")

    # Identify all known triples from EVA_VISUAL_COMPONENTS
    all_triples = set()
    for components in EVA_VISUAL_COMPONENTS.values():
        triple_key = (
            f"{components['first_stroke']},"
            f"{components['last_stroke']},"
            f"{components['glyph_class']}"
        )
        all_triples.add(triple_key)

    n_assigned = len(set(triple_to_syllable.keys()) & all_triples)
    n_unassigned = len(all_triples) - n_assigned
    print(f"     Assigned triples: {n_assigned}/{len(all_triples)}")
    print(f"     Unassigned triples: {n_unassigned}")

    # ── 3. Load Phase 15 assignment for comparison ──
    print("\n  3. Loading Phase 15 assignment for comparison …")
    refine_path = os.path.join(rd, 'combined_refine.json')
    refine_data = _safe_load(refine_path)
    phase15_assignment = refine_data.get('best_assignment', {})
    print(f"     Phase 15 assignment: {len(phase15_assignment)} triples")

    # ── 4. Compare inverted table with Phase 15 ──
    print("\n  4. Comparing inverted table vs Phase 15 assignment …")
    n_agreements = 0
    n_consonant_only = 0
    n_disagreements = 0
    disagreements: List[Dict] = []

    for triple_key, p15_syl in phase15_assignment.items():
        inv_syl = triple_to_syllable.get(triple_key)
        if inv_syl is None:
            # Triple not in inverted table — count as disagreement
            n_disagreements += 1
            disagreements.append({
                'triple_key': triple_key,
                'phase15_syllable': p15_syl,
                'inverted_syllable': None,
                'match_type': 'missing',
            })
            continue

        if inv_syl == p15_syl:
            n_agreements += 1
        elif len(inv_syl) >= 1 and len(p15_syl) >= 1 and inv_syl[0] == p15_syl[0]:
            # Same consonant, different vowel
            n_consonant_only += 1
            disagreements.append({
                'triple_key': triple_key,
                'phase15_syllable': p15_syl,
                'inverted_syllable': inv_syl,
                'match_type': 'consonant_only',
            })
        else:
            n_disagreements += 1
            disagreements.append({
                'triple_key': triple_key,
                'phase15_syllable': p15_syl,
                'inverted_syllable': inv_syl,
                'match_type': 'different',
            })

    n_compared = n_agreements + n_consonant_only + n_disagreements
    agreement_rate = n_agreements / max(n_compared, 1)

    print(f"     Agreements (exact):      {n_agreements}")
    print(f"     Consonant-only matches:  {n_consonant_only}")
    print(f"     Disagreements:           {n_disagreements}")
    print(f"     Agreement rate:          {agreement_rate:.1%}")

    if disagreements:
        print("     Sample disagreements:")
        for d in disagreements[:5]:
            print(f"       {d['triple_key']}: "
                  f"P15={d['phase15_syllable']} vs Inv={d['inverted_syllable']} "
                  f"({d['match_type']})")

    # ── 5. Load corpus and modifiers ──
    print("\n  5. Loading corpus and modifier rules …")
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens()
    eva_to_triple = build_eva_to_triple_lookup()
    print(f"     Tokens: {len(tokens)}")

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    mod_data = _safe_load(mod_path)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)
    print(f"     Modifier chars: {len(modifier_chars)}")

    # ── 6. Build expanded dictionary ──
    print("\n  6. Building expanded dictionary …")
    expanded_words, _ = build_expanded_word_set(base_words)
    expanded_set = base_words | expanded_words
    print(f"     Base words: {len(base_words):,}")
    print(f"     Expanded set: {len(expanded_set):,}")

    # ── 7. Decode the full corpus with the inverted table ──
    print("\n  7. Decoding full corpus with inverted table …")
    decoded_tokens = _decode_corpus_r3(
        tokens, triple_to_syllable, eva_to_triple,
        modifier_chars, modifier_rules, expanded_set,
    )

    dict_hit_count = sum(1 for w in decoded_tokens if w.lower() in expanded_set)
    dict_hit_rate = dict_hit_count / max(len(decoded_tokens), 1)
    decoded_sample = decoded_tokens[:50]

    print(f"     Decoded tokens: {len(decoded_tokens)}")
    print(f"     Dict hits: {dict_hit_count} ({dict_hit_rate:.1%})")
    print(f"     Sample: {decoded_sample[:10]}")

    # ── 8. Null comparison ──
    print("\n  8. Null corpus comparison (5 null corpora) …")
    N_NULL = 5

    # Get seeds from null_corpus.json if available
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_runs = null_data.get('null_runs', [])
    seeds = [r.get('seed', 100 + i) for i, r in enumerate(null_runs)]
    # Pad to N_NULL if needed
    while len(seeds) < N_NULL:
        seeds.append(200 + len(seeds))

    null_corpora = _generate_null_corpora(tokens, seeds, N_NULL)

    null_dict_hits: List[float] = []
    for i, null_tokens in enumerate(null_corpora):
        null_decoded = _decode_corpus_r3(
            null_tokens, triple_to_syllable, eva_to_triple,
            modifier_chars, modifier_rules, expanded_set,
        )
        null_hit = _compute_dict_hit(null_decoded, expanded_set)
        null_dict_hits.append(null_hit)
        print(f"     Null {i+1} (seed={seeds[i]}): {null_hit:.1%}")

    null_mean = float(np.mean(null_dict_hits))
    null_std = float(np.std(null_dict_hits))
    selectivity_ratio = dict_hit_rate / max(null_mean, 0.001)

    print(f"\n     Real dict_hit:     {dict_hit_rate:.1%}")
    print(f"     Null mean ± std:   {null_mean:.1%} ± {null_std:.1%}")
    print(f"     Selectivity ratio: {selectivity_ratio:.2f}×")

    # ── 9. Gate and verdict ──
    gate_passed = selectivity_ratio > 1.5
    if gate_passed:
        verdict = (
            f"PASS: Inverted table achieves {dict_hit_rate:.1%} dict_hit "
            f"({selectivity_ratio:.2f}× selectivity vs null). "
            f"{n_agreements}/{n_compared} triples agree with Phase 15 "
            f"({agreement_rate:.0%}). "
            f"{n_assigned}/{len(all_triples)} triples assigned."
        )
    else:
        verdict = (
            f"FAIL: Inverted table achieves {dict_hit_rate:.1%} dict_hit "
            f"({selectivity_ratio:.2f}× selectivity vs null, below 1.5× threshold). "
            f"{n_agreements}/{n_compared} triples agree with Phase 15 "
            f"({agreement_rate:.0%})."
        )

    print(f"\n  GATE: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Verdict: {verdict}")

    # ── 10. Save ──
    elapsed = time.time() - t0

    result = InversionDecodeResult(
        inverted_assignment=triple_to_syllable,
        n_assigned_triples=n_assigned,
        n_unassigned_triples=n_unassigned,
        n_agreements=n_agreements,
        n_consonant_only=n_consonant_only,
        n_disagreements=n_disagreements,
        agreement_rate=round(agreement_rate, 4),
        disagreements=disagreements,
        n_tokens_decoded=len(decoded_tokens),
        dict_hit_rate=round(dict_hit_rate, 4),
        dict_hit_count=dict_hit_count,
        decoded_sample=decoded_sample,
        null_dict_hit_mean=round(null_mean, 4),
        null_dict_hit_std=round(null_std, 4),
        selectivity_ratio=round(selectivity_ratio, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(elapsed, 2),
    )

    out_path = os.path.join(rd, 'inversion_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path} ({elapsed:.1f}s)")
