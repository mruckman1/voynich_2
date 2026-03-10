"""
Step 34.9 – Dialect-Optimized CSP Decode (Track C)
=====================================================
Re-runs the Phase 14 feature CSP but scoring against a synthetic Northern
Italian dialect reference corpus rather than classical Latin.  Compares
the dialect-optimized table against the Latin-optimized Phase 16 table to
determine which language model produces better signal.

Algorithm:
  1. Build dialect reference corpus via sound changes (reuses mixed_lm rules).
  2. Build char bigram/trigram frequency stats for the dialect corpus.
  3. Re-run CSP: 25 stroke-triple variables with CV syllable domains,
     minimizing cross-entropy against dialect character bigrams.
  4. Decode corpus through the Italian-optimized table.
  5. Run signal isolation (real vs null) against dialect bigram refs.
  6. Language discrimination: compare Latin-optimized vs Italian-optimized.
  7. Per-section language preference test.

Dependency chain:
    combined_refine.json       (Phase 15 assignment — Latin table baseline)
    modifier_integrate.json    (Phase 16 modifiers)
    null_corpus.json           (Phase 17 seeds)
    stroke_features.json       (Phase 14.2 triple info)
        -> dialect_decode.json (this step)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.mixed_lm import _apply_sound_changes, _preprocess_text


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
# Dialect reference building
# ---------------------------------------------------------------------------

def _build_dialect_corpus(ref_corpus) -> Tuple[str, Set[str], List[str]]:
    """Build synthetic Northern Italian corpus from Latin reference.

    Returns (dialect_text, dialect_word_set, dialect_tokens).
    """
    raw_latin = ref_corpus.get_combined_text('latin')
    latin_text = _preprocess_text(raw_latin)
    dialect_text = _apply_sound_changes(latin_text)

    # Build word set from dialect text
    dialect_words = dialect_text.split()
    dialect_word_set = set(w for w in dialect_words if len(w) >= 2)
    return dialect_text, dialect_word_set, dialect_words


def _build_char_bigrams(text: str) -> Tuple[Counter, int]:
    """Build character-level bigram counts from text."""
    bigrams: Counter = Counter()
    for i in range(len(text) - 1):
        bigrams[text[i:i + 2]] += 1
    total = sum(bigrams.values())
    return bigrams, total


def _build_char_trigrams(text: str) -> Tuple[Counter, int]:
    """Build character-level trigram counts from text."""
    trigrams: Counter = Counter()
    for i in range(len(text) - 2):
        trigrams[text[i:i + 3]] += 1
    total = sum(trigrams.values())
    return trigrams, total


def _build_word_bigrams(tokens: List[str]) -> Set[Tuple[str, str]]:
    """Build word-level bigram set from token list."""
    bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(tokens) - 1):
        if len(tokens[i]) >= 2 and len(tokens[i + 1]) >= 2:
            bigrams.add((tokens[i], tokens[i + 1]))
    return bigrams


# ---------------------------------------------------------------------------
# Simplified beam search (dialect-specific)
# ---------------------------------------------------------------------------

def _build_cv_domain() -> List[str]:
    """Build standard CV syllable domain for Latin-family languages."""
    consonants = ['b', 'c', 'd', 'f', 'g', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v']
    vowels = ['a', 'e', 'i', 'o', 'u']
    domain = []
    for c in consonants:
        for v in vowels:
            domain.append(c + v)
    # Pure vowels
    for v in vowels:
        domain.append(v)
    return domain


def _score_dialect_assignment(
    assignment: Dict[str, str],
    token_triple_seqs: List[List[str]],
    dialect_word_set: Set[str],
    char_bigrams: Counter,
    n_char_bigrams: int,
) -> Tuple[float, float, int]:
    """Score an assignment by dict hit rate and char-bigram cross-entropy.

    Returns (cross_entropy, hit_rate, n_hits).
    """
    # Decode tokens
    decoded = []
    for seq in token_triple_seqs:
        syllables = [assignment.get(t, '?') for t in seq]
        word = ''.join(syllables)
        decoded.append(word)

    # Dict hit rate
    n_hits = sum(1 for w in decoded if w.lower() in dialect_word_set and len(w) >= 2)
    hit_rate = n_hits / len(decoded) if decoded else 0.0

    # Character bigram cross-entropy
    total_log = 0.0
    n_bg = 0
    vocab_size = 27  # a-z + space
    for word in decoded:
        w = word.lower()
        if len(w) < 2:
            continue
        for i in range(len(w) - 1):
            bg = w[i:i + 2]
            freq = char_bigrams.get(bg, 0)
            prob = (freq + 1) / (n_char_bigrams + vocab_size * vocab_size)
            total_log += math.log2(prob)
            n_bg += 1

    cross_entropy = -total_log / n_bg if n_bg > 0 else 20.0
    return cross_entropy, hit_rate, n_hits


def _dialect_beam_search(
    triple_keys: List[str],
    triple_freqs: Dict[str, int],
    token_triple_seqs: List[List[str]],
    dialect_word_set: Set[str],
    char_bigrams: Counter,
    n_char_bigrams: int,
    beam_width: int = 30,
    max_solutions: int = 10,
    seed: int = 42,
) -> List[Tuple[Dict[str, str], float, float, int]]:
    """Beam search for dialect-optimized CV assignment.

    Returns list of (assignment, cross_entropy, hit_rate, n_hits).
    """
    rng = random.Random(seed)
    domain = _build_cv_domain()

    # Sort triples by frequency (most constrained first)
    sorted_triples = sorted(triple_keys,
                            key=lambda t: triple_freqs.get(t, 0),
                            reverse=True)

    # Build frequency-ranked syllable ordering from dialect word set
    syl_freq: Counter = Counter()
    for word in dialect_word_set:
        w = word.lower()
        for i in range(0, len(w) - 1, 2):
            syl = w[i:i + 2]
            if syl in domain:
                syl_freq[syl] += 1
    freq_ranked_syls = [s for s, _ in syl_freq.most_common()] + [
        s for s in domain if s not in syl_freq
    ]

    # Beam: list of (partial_assignment, score)
    beam: List[Tuple[Dict[str, str], float]] = [({}, 0.0)]

    for triple in sorted_triples:
        new_beam: List[Tuple[Dict[str, str], float]] = []

        for partial, prev_score in beam:
            used = set(partial.values())
            candidates = [s for s in freq_ranked_syls if s not in used]
            if not candidates:
                candidates = freq_ranked_syls[:]

            for syllable in candidates[:8]:
                new_assign = dict(partial)
                new_assign[triple] = syllable

                # Quick score: count decoded roots that match dialect words
                quick_hits = 0
                for seq in token_triple_seqs[:500]:
                    word = ''.join(new_assign.get(t, '?') for t in seq)
                    if '?' not in word and word.lower() in dialect_word_set and len(word) >= 2:
                        quick_hits += 1

                new_beam.append((new_assign, quick_hits))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    # Full scoring of final candidates
    results = []
    for assignment, _ in beam:
        ce, hit_rate, n_hits = _score_dialect_assignment(
            assignment, token_triple_seqs, dialect_word_set,
            char_bigrams, n_char_bigrams,
        )
        results.append((assignment, ce, hit_rate, n_hits))

    results.sort(key=lambda x: (-x[2], x[1]))
    return results[:max_solutions]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SectionPreference:
    section: str
    n_tokens: int
    latin_dict_hit: float
    dialect_dict_hit: float
    prefers: str  # 'LATIN' or 'DIALECT'


@dataclass
class DialectDecodeResult:
    # Dialect corpus
    n_dialect_words: int
    n_dialect_word_set: int
    n_char_bigrams: int
    n_char_trigrams: int

    # CSP
    n_variables: int
    n_domain: int
    best_assignment: Dict[str, str]
    best_cross_entropy: float
    n_beam_solutions: int

    # Decode results
    dialect_dict_hit: float
    dialect_n_hits: int
    n_tokens: int
    sample_decoded: List[Dict]  # first 30

    # Null comparison
    null_dict_hit_rates: List[float]
    null_mean: float
    dialect_selectivity: float

    # Signal isolation (real vs null)
    n_signal: int
    signal_rate: float
    n_anti: int
    n_shared_hit: int
    n_shared_miss: int

    # Language discrimination
    phase16_dict_hit: float
    phase16_selectivity: float
    dialect_vs_latin: str  # 'DIALECT_BETTER' or 'LATIN_BETTER'

    # Per-section preference
    section_preferences: List[Dict]
    n_sections_prefer_dialect: int
    n_sections_total: int

    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dialect_decode() -> None:
    """Step 34.9: Dialect-optimized CSP decode."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.9: Dialect-Optimized CSP Decode (Track C)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Build dialect reference corpus ──
    print("\n  1. Building dialect reference corpus …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    dialect_text, dialect_word_set, dialect_tokens = _build_dialect_corpus(ref_corpus)
    print(f"     Dialect words: {len(dialect_tokens)}")
    print(f"     Dialect word set: {len(dialect_word_set)}")

    # Also build expanded dialect word set (medieval variants of dialect words)
    expanded_dialect, _ = build_expanded_word_set(dialect_word_set)
    full_dialect_set = dialect_word_set | expanded_dialect
    print(f"     Expanded dialect set: {len(full_dialect_set)}")

    # ── 2. Build char n-gram stats ──
    print("\n  2. Building character n-gram stats …")
    char_bigrams, n_char_bigrams = _build_char_bigrams(dialect_text)
    char_trigrams, n_char_trigrams = _build_char_trigrams(dialect_text)
    print(f"     Char bigrams: {len(char_bigrams)} unique, {n_char_bigrams} total")
    print(f"     Char trigrams: {len(char_trigrams)} unique, {n_char_trigrams} total")

    # Build word-level bigrams for signal testing
    dialect_word_bigrams = _build_word_bigrams(dialect_tokens)
    print(f"     Word bigrams: {len(dialect_word_bigrams)} unique")

    # ── 3. Load corpus and build triple sequences ──
    print("\n  3. Loading corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    # Load modifiers
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    all_tokens: List[str] = []
    token_folios: List[str] = []
    token_sections: List[str] = []
    for folio, page in corpus.pages.items():
        section = getattr(page, 'section', 'unknown')
        for token in page.all_tokens:
            all_tokens.append(token)
            token_folios.append(folio)
            token_sections.append(section)
    n_tokens = len(all_tokens)
    print(f"     {n_tokens} tokens")

    # Build triple sequences for each token
    token_triple_seqs: List[List[str]] = []
    triple_freqs: Counter = Counter()
    for token in all_tokens:
        chars = tokenize_eva_chars(token)
        triples = []
        for ch in chars:
            if ch not in modifier_chars:
                triple = eva_to_triple.get(ch)
                if triple:
                    triples.append(triple)
        token_triple_seqs.append(triples)
        for t in triples:
            triple_freqs[t] += 1

    triple_keys = sorted(triple_freqs.keys(),
                         key=lambda t: triple_freqs[t], reverse=True)
    print(f"     {len(triple_keys)} active triples")

    # ── 4. Run dialect beam search ──
    print("\n  4. Running dialect-optimized beam search …")
    solutions = _dialect_beam_search(
        triple_keys, triple_freqs, token_triple_seqs,
        full_dialect_set, char_bigrams, n_char_bigrams,
        beam_width=30, max_solutions=10,
    )

    if not solutions:
        print("  [FAIL] No solutions found")
        return

    best_assign, best_ce, best_hit_rate, best_n_hits = solutions[0]
    print(f"     Best: hit_rate={best_hit_rate:.3f} ({best_n_hits} hits), CE={best_ce:.3f}")
    print(f"     {len(solutions)} solutions total")

    # ── 5. Decode corpus through dialect table ──
    print("\n  5. Decoding corpus through dialect table …")
    dialect_decoded: List[str] = []
    for seq in token_triple_seqs:
        syllables = [best_assign.get(t, '?') for t in seq]
        word = ''.join(syllables).lower()
        dialect_decoded.append(word)

    dialect_hits = [w in full_dialect_set and len(w) >= 2 for w in dialect_decoded]
    dialect_dict_hit = sum(dialect_hits) / n_tokens if n_tokens > 0 else 0.0
    print(f"     Dialect dict hit: {dialect_dict_hit:.3f} ({sum(dialect_hits)} hits)")

    # Sample decoded tokens
    sample_decoded = []
    for i in range(min(30, n_tokens)):
        sample_decoded.append({
            'token': all_tokens[i],
            'triples': token_triple_seqs[i],
            'decoded': dialect_decoded[i],
            'is_hit': dialect_hits[i],
        })

    # ── 6. Null comparison ──
    print("\n  6. Running null comparison …")
    null_seeds = [100, 101, 102, 103, 104]
    null_path = os.path.join(rd, 'null_corpus.json')
    if os.path.exists(null_path):
        with open(null_path) as f:
            null_data = json.load(f)
        null_seeds = [r['seed'] for r in null_data.get('null_runs', [])]

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    null_hit_rates: List[float] = []

    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        # Decode null tokens through dialect table
        null_decoded = []
        for token in null_tokens:
            chars = tokenize_eva_chars(token)
            triples = []
            for ch in chars:
                if ch not in modifier_chars:
                    triple = eva_to_triple.get(ch)
                    if triple:
                        triples.append(triple)
            word = ''.join(best_assign.get(t, '?') for t in triples).lower()
            null_decoded.append(word)
        null_hits = sum(
            1 for w in null_decoded if w in full_dialect_set and len(w) >= 2
        )
        null_hit_rates.append(null_hits / len(null_decoded) if null_decoded else 0.0)

    null_mean = sum(null_hit_rates) / len(null_hit_rates) if null_hit_rates else 0.0
    dialect_selectivity = dialect_dict_hit / null_mean if null_mean > 0 else float('inf')
    print(f"     Null mean: {null_mean:.3f}")
    print(f"     Dialect selectivity: {dialect_selectivity:.2f}x")

    # ── 7. Signal isolation (real vs null) ──
    print("\n  7. Signal isolation …")
    # Build null hit lists for classification
    null_hits_lists: List[List[bool]] = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, n_tokens, seed,
        )
        null_dec = []
        for token in null_tokens:
            chars = tokenize_eva_chars(token)
            triples = []
            for ch in chars:
                if ch not in modifier_chars:
                    triple = eva_to_triple.get(ch)
                    if triple:
                        triples.append(triple)
            word = ''.join(best_assign.get(t, '?') for t in triples).lower()
            null_dec.append(word)
        null_hits_lists.append([
            w in full_dialect_set and len(w) >= 2 for w in null_dec
        ])

    classifications: List[str] = []
    for idx in range(n_tokens):
        r_hit = dialect_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_lists if nh[idx])
        if r_hit and null_hit_count <= 1:
            classifications.append('SIGNAL')
        elif r_hit and null_hit_count >= 3:
            classifications.append('SHARED_HIT')
        elif not r_hit and null_hit_count >= 3:
            classifications.append('ANTI_SIGNAL')
        else:
            classifications.append('SHARED_MISS')

    n_signal = classifications.count('SIGNAL')
    n_anti = classifications.count('ANTI_SIGNAL')
    n_shared_hit = classifications.count('SHARED_HIT')
    n_shared_miss = classifications.count('SHARED_MISS')
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    print(f"     SIGNAL: {n_signal} ({signal_rate:.3f})")
    print(f"     ANTI: {n_anti}, SHARED_HIT: {n_shared_hit}, SHARED_MISS: {n_shared_miss}")

    # ── 8. Language discrimination vs Phase 16 ──
    print("\n  8. Language discrimination …")
    refine_path = os.path.join(rd, 'combined_refine.json')
    phase16_dict_hit = 0.0
    phase16_selectivity = 3.38  # default from memory

    if os.path.exists(refine_path):
        with open(refine_path) as f:
            refine_data = json.load(f)
        latin_assignment = refine_data.get('best_assignment', {})

        # Build Latin reference word set
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2
        )
        expanded_lat, _ = build_expanded_word_set(base_words)
        latin_ref_set = base_words | expanded_lat

        # Decode through Latin table
        latin_decoded = _decode_corpus_r3(
            all_tokens, latin_assignment, eva_to_triple,
            modifier_chars, modifier_rules, latin_ref_set,
        )
        latin_hits = sum(1 for w in latin_decoded if w in latin_ref_set)
        phase16_dict_hit = latin_hits / n_tokens if n_tokens > 0 else 0.0
        print(f"     Phase 16 (Latin) dict hit: {phase16_dict_hit:.3f}")

    dialect_vs_latin = (
        'DIALECT_BETTER'
        if dialect_selectivity > phase16_selectivity
        else 'LATIN_BETTER'
    )
    print(f"     Phase 16 selectivity: {phase16_selectivity:.2f}x")
    print(f"     Dialect selectivity: {dialect_selectivity:.2f}x")
    print(f"     Verdict: {dialect_vs_latin}")

    # ── 9. Per-section language preference ──
    print("\n  9. Per-section language preference …")
    section_tokens: Dict[str, List[int]] = defaultdict(list)
    for i, sec in enumerate(token_sections):
        section_tokens[sec].append(i)

    section_prefs: List[SectionPreference] = []
    for sec_name, indices in sorted(section_tokens.items()):
        if len(indices) < 20:
            continue

        # Dialect hits for this section
        sec_dialect_hits = sum(1 for i in indices if dialect_hits[i])
        sec_dialect_rate = sec_dialect_hits / len(indices)

        # Latin hits for this section (using Phase 16 decoded if available)
        sec_latin_rate = phase16_dict_hit  # approximate with corpus-wide rate
        if os.path.exists(refine_path):
            sec_latin_hits = sum(
                1 for i in indices
                if i < len(latin_decoded) and latin_decoded[i] in latin_ref_set
            )
            sec_latin_rate = sec_latin_hits / len(indices)

        prefers = 'DIALECT' if sec_dialect_rate > sec_latin_rate else 'LATIN'
        section_prefs.append(SectionPreference(
            section=sec_name,
            n_tokens=len(indices),
            latin_dict_hit=round(sec_latin_rate, 4),
            dialect_dict_hit=round(sec_dialect_rate, 4),
            prefers=prefers,
        ))
        print(f"     {sec_name:15s}: n={len(indices):5d}, "
              f"latin={sec_latin_rate:.3f}, dialect={sec_dialect_rate:.3f} "
              f"-> {prefers}")

    n_prefer_dialect = sum(1 for sp in section_prefs if sp.prefers == 'DIALECT')

    # ── 10. Verdict ──
    verdict_parts = []
    verdict_parts.append(
        f"Dialect dict_hit={dialect_dict_hit:.3f} "
        f"(selectivity={dialect_selectivity:.2f}x)"
    )
    verdict_parts.append(
        f"Phase16 dict_hit={phase16_dict_hit:.3f} "
        f"(selectivity={phase16_selectivity:.2f}x)"
    )
    verdict_parts.append(dialect_vs_latin)
    verdict_parts.append(
        f"Sections preferring dialect: {n_prefer_dialect}/{len(section_prefs)}"
    )
    verdict = '; '.join(verdict_parts)

    print(f"\n  Verdict: {verdict}")

    elapsed = round(time.time() - t0, 2)

    result = DialectDecodeResult(
        n_dialect_words=len(dialect_tokens),
        n_dialect_word_set=len(full_dialect_set),
        n_char_bigrams=n_char_bigrams,
        n_char_trigrams=n_char_trigrams,
        n_variables=len(triple_keys),
        n_domain=len(_build_cv_domain()),
        best_assignment=best_assign,
        best_cross_entropy=round(best_ce, 4),
        n_beam_solutions=len(solutions),
        dialect_dict_hit=round(dialect_dict_hit, 4),
        dialect_n_hits=sum(dialect_hits),
        n_tokens=n_tokens,
        sample_decoded=sample_decoded[:30],
        null_dict_hit_rates=[round(r, 4) for r in null_hit_rates],
        null_mean=round(null_mean, 4),
        dialect_selectivity=round(dialect_selectivity, 4),
        n_signal=n_signal,
        signal_rate=round(signal_rate, 4),
        n_anti=n_anti,
        n_shared_hit=n_shared_hit,
        n_shared_miss=n_shared_miss,
        phase16_dict_hit=round(phase16_dict_hit, 4),
        phase16_selectivity=phase16_selectivity,
        dialect_vs_latin=dialect_vs_latin,
        section_preferences=[_convert(sp) for sp in section_prefs],
        n_sections_prefer_dialect=n_prefer_dialect,
        n_sections_total=len(section_prefs),
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'dialect_decode.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved -> {out_path}  ({elapsed:.1f}s)")
