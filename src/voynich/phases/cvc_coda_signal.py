"""
Phase 57, Steps 3-5: CVC Validation + Signal Isolation + Comparison
====================================================================
Validates CVC decoded output against Costamagna's attested inventory,
runs signal isolation comparing real vs null, and compares 4 decode
strategies: cv_strip, r3_combined, cvc_primary, cvc_alternate.

Dependency chain:
    results/coda_table.json       (Step 57.1)
    results/combined_refine.json  (Phase 15)
    results/modifier_integrate.json  (Phase 16)
    results/null_corpus.json      (Phase 17)
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
        -> results/cvc_coda_signal.json  (Step 57.4)
        -> results/cvc_compare.json      (Step 57.5)
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import data_dir, results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)
from voynich.phases.coda_markers import (
    build_coda_table,
    decode_corpus_cvc,
    decode_token_cvc,
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CvcValidation:
    total_cvc_tokens: int
    unique_cvc_types: int
    attested_in_costamagna: int
    unattested: int
    attestation_rate_type: float
    attestation_rate_token: float
    top_attested: List[str]
    top_unattested: List[str]


@dataclass
class SignalStats:
    n_signal: int
    n_shared_hit: int
    n_shared_miss: int
    n_anti_signal: int
    signal_rate: float
    n_signal_words: int
    mean_selectivity: float
    top_signal_words: List[Dict[str, Any]]


@dataclass
class StrategyResult:
    name: str
    desc: str
    dict_hit: float
    n_signal_words: int
    mean_selectivity: float
    bigram_z: float
    mean_word_length: float
    net_signal: int
    cvc_validation: Optional[CvcValidation] = None


@dataclass
class CvcSignalResult:
    phase: str = "57"
    step: str = "57.4"
    experiment: str = "cvc_coda_signal"
    dict_hit_real: float = 0.0
    dict_hit_null_mean: float = 0.0
    selectivity: float = 0.0
    signal: Optional[SignalStats] = None
    cvc_validation: Optional[CvcValidation] = None
    runtime_seconds: float = 0.0


@dataclass
class CvcCompareResult:
    phase: str = "57"
    step: str = "57.5"
    experiment: str = "cvc_compare"
    strategies: List[StrategyResult] = field(default_factory=list)
    winner: str = ""
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _decode_corpus_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """Decode tokens using R3 strategy (alteration -> strip -> raw)."""
    decoded = []
    for token in tokens:
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars, modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt.lower())
            continue
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped.lower())
            continue
        raw = decode_token(token, assignment, eva_to_triple)
        decoded.append(raw.lower())
    return decoded


def _decode_corpus_cv_strip(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
) -> List[str]:
    """Decode tokens using simple CV strip (drop modifiers)."""
    return [
        decode_token_modifier_aware(
            tok, assignment, eva_to_triple, modifier_chars,
        ).lower()
        for tok in tokens
    ]


# ---------------------------------------------------------------------------
# Costamagna CVC validation
# ---------------------------------------------------------------------------

def _load_costamagna_syllables() -> Tuple[Set[str], Set[str], Set[str]]:
    """Load Costamagna syllabary and return (cv_set, cvc_set, all_set)."""
    syl_path = os.path.join(
        str(data_dir('GL.S.III.MISC.12/extraction')),
        'syllabary_table.json',
    )
    if not os.path.exists(syl_path):
        return set(), set(), set()

    with open(syl_path) as f:
        entries = json.load(f)

    cv_set: Set[str] = set()
    cvc_set: Set[str] = set()
    all_set: Set[str] = set()

    for entry in entries:
        syl = entry.get('syllable', '')
        struct = entry.get('structure', '')
        parts = [s.strip() for s in syl.split('-')] if '-' in syl else [syl]
        for part in parts:
            pl = part.lower()
            all_set.add(pl)
            if struct in ('CV', 'VC', 'V'):
                cv_set.add(pl)
            elif struct in ('CVC', 'CCV', 'VCC', 'CVCC'):
                cvc_set.add(pl)
            elif struct == 'shared_sign':
                if len(pl) <= 2:
                    cv_set.add(pl)
                else:
                    cvc_set.add(pl)
            else:
                if len(pl) <= 2:
                    cv_set.add(pl)
                else:
                    cvc_set.add(pl)

    return cv_set, cvc_set, all_set


def _extract_cvc_syllables(decoded_word: str) -> List[str]:
    """Extract CVC-pattern substrings from a decoded word.

    A CVC syllable = consonant + vowel + consonant (3 chars, ends in consonant).
    We scan the decoded word for such patterns.
    """
    vowels = set('aeiou')
    cvcs = []
    # Split into syllables by grouping C*V+C* patterns
    current = ''
    for ch in decoded_word:
        current += ch
        if ch in vowels:
            cvcs.append(current)
            current = ''
    if current and cvcs:
        cvcs[-1] += current
    elif current:
        cvcs.append(current)

    result = []
    for syl in cvcs:
        if len(syl) >= 3 and syl[-1] not in vowels:
            result.append(syl)
    return result


def _validate_cvc_costamagna(decoded_tokens: List[str]) -> CvcValidation:
    """Validate CVC decoded output against Costamagna's inventory."""
    _, cvc_set, _ = _load_costamagna_syllables()

    produced_cvcs: List[str] = []
    for word in decoded_tokens:
        if not word or word == '?':
            continue
        for syl in _extract_cvc_syllables(word):
            produced_cvcs.append(syl.lower())

    total = len(produced_cvcs)
    unique = set(produced_cvcs)
    attested = unique & cvc_set
    unattested = unique - cvc_set

    cvc_counts = Counter(produced_cvcs)
    attested_tokens = sum(cvc_counts[s] for s in attested)

    return CvcValidation(
        total_cvc_tokens=total,
        unique_cvc_types=len(unique),
        attested_in_costamagna=len(attested),
        unattested=len(unattested),
        attestation_rate_type=len(attested) / len(unique) if unique else 0.0,
        attestation_rate_token=attested_tokens / total if total else 0.0,
        top_attested=sorted(attested)[:20],
        top_unattested=sorted(unattested)[:20],
    )


# ---------------------------------------------------------------------------
# Signal isolation on decoded corpus
# ---------------------------------------------------------------------------

def _run_signal_isolation(
    real_decoded: List[str],
    null_decoded_list: List[List[str]],
    ref_word_set: Set[str],
    n_tokens: int,
) -> SignalStats:
    """Per-word and per-token signal isolation."""
    real_hits = [w in ref_word_set for w in real_decoded]
    real_word_counts = Counter(w for w, hit in zip(real_decoded, real_hits) if hit)

    test_words = sorted(real_word_counts.keys())

    word_signals = []
    for word in test_words:
        real_count = real_word_counts.get(word, 0)
        null_counts = [Counter(nd).get(word, 0) for nd in null_decoded_list]
        null_mean = sum(null_counts) / len(null_counts) if null_counts else 0.0
        null_var = (sum((c - null_mean) ** 2 for c in null_counts)
                    / len(null_counts) if null_counts else 0.0)
        null_std = null_var ** 0.5

        sigma = ((real_count - null_mean) / null_std) if null_std > 0 else (
            float('inf') if real_count > null_mean else 0.0
        )
        selectivity = (real_count / null_mean) if null_mean > 0 else float('inf')

        if sigma > 2.0:
            word_signals.append({
                'word': word,
                'sigma': round(sigma, 2) if sigma != float('inf') else 999.0,
                'real_count': real_count,
                'null_mean': round(null_mean, 2),
                'selectivity': round(selectivity, 2) if selectivity != float('inf') else 999.0,
            })

    word_signals.sort(key=lambda w: -w['sigma'])

    # Per-token classification
    null_hits_list = [[w in ref_word_set for w in nd] for nd in null_decoded_list]

    n_signal = n_shared_hit = n_shared_miss = n_anti_signal = 0
    for idx in range(min(n_tokens, len(real_decoded))):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if idx < len(nh) and nh[idx])

        if r_hit and null_hit_count <= 1:
            n_signal += 1
        elif r_hit and null_hit_count >= 3:
            n_shared_hit += 1
        elif not r_hit and null_hit_count <= 1:
            n_shared_miss += 1
        elif not r_hit and null_hit_count >= 3:
            n_anti_signal += 1
        else:
            n_shared_miss += 1

    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0

    finite_sels = [w['selectivity'] for w in word_signals if w['selectivity'] < 900]
    mean_sel = sum(finite_sels) / len(finite_sels) if finite_sels else 0.0

    return SignalStats(
        n_signal=n_signal,
        n_shared_hit=n_shared_hit,
        n_shared_miss=n_shared_miss,
        n_anti_signal=n_anti_signal,
        signal_rate=round(signal_rate, 4),
        n_signal_words=len(word_signals),
        mean_selectivity=round(mean_sel, 2),
        top_signal_words=word_signals[:20],
    )


# ---------------------------------------------------------------------------
# Bigram z-score (simplified permutation test)
# ---------------------------------------------------------------------------

def _compute_bigram_z(
    real_decoded: List[str],
    null_decoded_list: List[List[str]],
    ref_word_set: Set[str],
    folios: List[str],
    n_perms: int = 500,
) -> float:
    """Compute bigram z-score using positional permutation test."""
    n_tokens = len(real_decoded)
    if n_tokens < 10:
        return 0.0

    # Classify tokens as SIGNAL
    real_hits = [w in ref_word_set for w in real_decoded]
    null_hits_list = [[w in ref_word_set for w in nd] for nd in null_decoded_list]

    signal_positions: Set[int] = set()
    for idx in range(n_tokens):
        r_hit = real_hits[idx]
        null_hit_count = sum(1 for nh in null_hits_list if idx < len(nh) and nh[idx])
        if r_hit and null_hit_count <= 1:
            signal_positions.add(idx)

    n_signal = len(signal_positions)
    if n_signal < 5:
        return 0.0

    # Build reference bigram set from top-frequency words
    real_word_counts = Counter(real_decoded)
    top_words = {w for w, c in real_word_counts.most_common(500) if w in ref_word_set}
    ref_bigrams: Set[Tuple[str, str]] = set()
    for w1 in top_words:
        for w2 in top_words:
            if w1 != w2:
                ref_bigrams.add((w1, w2))

    # Observed SIGNAL-SIGNAL bigram hit rate
    n_pairs = 0
    n_hits = 0
    for i in range(n_tokens - 1):
        if (i in signal_positions and (i + 1) in signal_positions
                and i < len(folios) and (i + 1) < len(folios)
                and folios[i] == folios[i + 1]):
            n_pairs += 1
            if (real_decoded[i], real_decoded[i + 1]) in ref_bigrams:
                n_hits += 1

    if n_pairs == 0:
        return 0.0
    observed_rate = n_hits / n_pairs

    # Null permutation test
    rng = random.Random(42)
    indices = list(range(n_tokens))
    null_rates: List[float] = []

    for _ in range(n_perms):
        fake_signal = set(rng.sample(indices, min(n_signal, n_tokens)))
        np_ = 0
        nh_ = 0
        for i in range(n_tokens - 1):
            if (i in fake_signal and (i + 1) in fake_signal
                    and i < len(folios) and (i + 1) < len(folios)
                    and folios[i] == folios[i + 1]):
                np_ += 1
                if (real_decoded[i], real_decoded[i + 1]) in ref_bigrams:
                    nh_ += 1
        rate = nh_ / np_ if np_ > 0 else 0.0
        null_rates.append(rate)

    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    null_var = (sum((r - null_mean) ** 2 for r in null_rates)
                / len(null_rates) if null_rates else 0.0)
    null_std = null_var ** 0.5

    if null_std == 0:
        return 0.0 if observed_rate <= null_mean else float('inf')
    return (observed_rate - null_mean) / null_std


# ---------------------------------------------------------------------------
# Build folio list
# ---------------------------------------------------------------------------

def _build_folio_list(corpus) -> List[str]:
    """Build a flat list of folio IDs, one per token."""
    folios: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folios.append(folio)
    return folios


# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------

def _load_shared_data():
    """Load all data needed by both cvc_signal and cvc_compare."""
    rd = str(_results_dir())
    eva_to_triple = build_eva_to_triple_lookup()

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folios = _build_folio_list(corpus)

    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_seeds = ([r['seed'] for r in null_data.get('null_runs', [])]
                  if null_data else [100, 101, 102, 103, 104])

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    null_token_lists = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed,
        )
        null_token_lists.append(null_tokens)

    coda_primary = build_coda_table('primary')
    coda_alternate = build_coda_table('alternate')

    return {
        'rd': rd,
        'eva_to_triple': eva_to_triple,
        'assignment': assignment,
        'modifier_chars': modifier_chars,
        'modifier_rules': modifier_rules,
        'ref_word_set': ref_word_set,
        'corpus': corpus,
        'all_tokens': all_tokens,
        'folios': folios,
        'null_seeds': null_seeds,
        'null_token_lists': null_token_lists,
        'coda_primary': coda_primary,
        'coda_alternate': coda_alternate,
    }


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def _run_strategy(
    name: str,
    desc: str,
    all_tokens: List[str],
    null_token_lists: List[List[str]],
    ref_word_set: Set[str],
    folios: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    coda_primary,
    coda_alternate,
    n_perms: int = 500,
) -> StrategyResult:
    """Decode real + null with a strategy, compute all metrics."""
    n_tokens = len(all_tokens)

    # Decode real corpus
    if name == 'cv_strip':
        real_decoded = _decode_corpus_cv_strip(
            all_tokens, assignment, eva_to_triple, modifier_chars)
    elif name == 'r3_combined':
        real_decoded = _decode_corpus_r3(
            all_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set)
    elif name == 'cvc_primary':
        real_decoded = decode_corpus_cvc(
            all_tokens, assignment, eva_to_triple, coda_primary)
    elif name == 'cvc_alternate':
        real_decoded = decode_corpus_cvc(
            all_tokens, assignment, eva_to_triple, coda_alternate)
    else:
        raise ValueError(f"Unknown strategy: {name}")

    # Decode null corpora
    null_decoded_list = []
    for null_tokens in null_token_lists:
        if name == 'cv_strip':
            nd = _decode_corpus_cv_strip(
                null_tokens, assignment, eva_to_triple, modifier_chars)
        elif name == 'r3_combined':
            nd = _decode_corpus_r3(
                null_tokens, assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set)
        elif name == 'cvc_primary':
            nd = decode_corpus_cvc(
                null_tokens, assignment, eva_to_triple, coda_primary)
        elif name == 'cvc_alternate':
            nd = decode_corpus_cvc(
                null_tokens, assignment, eva_to_triple, coda_alternate)
        null_decoded_list.append(nd)

    # Dict hit
    real_hits = sum(1 for w in real_decoded if w in ref_word_set)
    dict_hit = real_hits / n_tokens if n_tokens > 0 else 0.0

    # Signal isolation
    signal = _run_signal_isolation(
        real_decoded, null_decoded_list, ref_word_set, n_tokens)

    # Bigram z
    bigram_z = _compute_bigram_z(
        real_decoded, null_decoded_list, ref_word_set, folios, n_perms=n_perms)

    # Mean word length
    lengths = [len(w) for w in real_decoded if w and w != '?']
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0

    # Net signal
    net_signal = signal.n_signal - signal.n_anti_signal

    # CVC validation (only for CVC strategies)
    cvc_val = None
    if name.startswith('cvc_'):
        cvc_val = _validate_cvc_costamagna(real_decoded)

    return StrategyResult(
        name=name,
        desc=desc,
        dict_hit=round(dict_hit, 4),
        n_signal_words=signal.n_signal_words,
        mean_selectivity=signal.mean_selectivity,
        bigram_z=round(bigram_z, 2),
        mean_word_length=round(mean_len, 2),
        net_signal=net_signal,
        cvc_validation=cvc_val,
    )


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_cvc_coda_signal():
    """Step 57.4: Signal isolation on CVC decoded corpus."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 57, Step 4: CVC Signal Isolation")
    print("=" * 70)

    data = _load_shared_data()
    rd = data['rd']

    print("\n  Decoding real corpus with CVC primary ...")
    real_decoded = decode_corpus_cvc(
        data['all_tokens'], data['assignment'],
        data['eva_to_triple'], data['coda_primary'])

    n_tokens = len(data['all_tokens'])
    real_hits = sum(1 for w in real_decoded if w in data['ref_word_set'])
    dict_hit = real_hits / n_tokens
    print(f"  Dict hit: {dict_hit:.4f} ({real_hits}/{n_tokens})")

    print("\n  Decoding null corpora ...")
    null_decoded_list = []
    for i, null_tokens in enumerate(data['null_token_lists']):
        nd = decode_corpus_cvc(
            null_tokens, data['assignment'],
            data['eva_to_triple'], data['coda_primary'])
        null_decoded_list.append(nd)
        null_hits = sum(1 for w in nd if w in data['ref_word_set'])
        print(f"    Null {i+1}: dict_hit = {null_hits / len(nd):.4f}")

    null_hit_rates = [
        sum(1 for w in nd if w in data['ref_word_set']) / len(nd)
        for nd in null_decoded_list
    ]
    null_mean = sum(null_hit_rates) / len(null_hit_rates)
    selectivity = dict_hit / null_mean if null_mean > 0 else float('inf')
    print(f"\n  Selectivity: {selectivity:.2f}x")

    print("\n  Running signal isolation ...")
    signal = _run_signal_isolation(
        real_decoded, null_decoded_list, data['ref_word_set'], n_tokens)

    print(f"  SIGNAL tokens: {signal.n_signal} ({signal.signal_rate:.1%})")
    print(f"  SHARED_HIT:    {signal.n_shared_hit}")
    print(f"  SHARED_MISS:   {signal.n_shared_miss}")
    print(f"  ANTI_SIGNAL:   {signal.n_anti_signal}")
    print(f"  Signal words (sigma>2): {signal.n_signal_words}")
    print(f"  Mean selectivity: {signal.mean_selectivity:.2f}x")

    if signal.top_signal_words:
        print("\n  Top signal words:")
        for ws in signal.top_signal_words[:10]:
            print(f"    {ws['word']:12s} sigma={ws['sigma']:6.1f} "
                  f"real={ws['real_count']:4d} sel={ws['selectivity']:.1f}x")

    print("\n  Validating against Costamagna CVC inventory ...")
    cvc_val = _validate_cvc_costamagna(real_decoded)
    print(f"  CVC types produced: {cvc_val.unique_cvc_types}")
    print(f"  Attested in Costamagna: {cvc_val.attested_in_costamagna} "
          f"({cvc_val.attestation_rate_type:.1%})")
    print(f"  Token attestation: {cvc_val.attestation_rate_token:.1%}")

    result = CvcSignalResult(
        dict_hit_real=round(dict_hit, 4),
        dict_hit_null_mean=round(null_mean, 4),
        selectivity=round(selectivity, 2),
        signal=signal,
        cvc_validation=cvc_val,
        runtime_seconds=round(time.time() - t0, 2),
    )
    path = _save_json(rd, 'cvc_coda_signal.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Step 57.4 completed in {time.time() - t0:.1f}s")


def run_cvc_compare():
    """Step 57.5: Compare 4 decode strategies."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 57, Step 5: CVC Decode Comparison Battery")
    print("=" * 70)

    data = _load_shared_data()
    rd = data['rd']

    strategies_spec = [
        ('cv_strip', 'Phase 16: strip modifiers, CV decode'),
        ('r3_combined', 'Phase 16 R3: alteration -> strip -> raw'),
        ('cvc_primary', 'CVC: CV + coda (vertical->t)'),
        ('cvc_alternate', 'CVC: CV + coda (vertical->m)'),
    ]

    strategies = []
    for name, desc in strategies_spec:
        print(f"\n  Strategy: {name}")
        print(f"    {desc}")
        sr = _run_strategy(
            name, desc,
            data['all_tokens'], data['null_token_lists'],
            data['ref_word_set'], data['folios'],
            data['assignment'], data['eva_to_triple'],
            data['modifier_chars'], data['modifier_rules'],
            data['coda_primary'], data['coda_alternate'],
            n_perms=500,
        )
        strategies.append(sr)
        print(f"    dict_hit={sr.dict_hit:.4f}  signal_words={sr.n_signal_words}  "
              f"bigram_z={sr.bigram_z:.2f}  mean_len={sr.mean_word_length:.2f}  "
              f"net_signal={sr.net_signal}")
        if sr.cvc_validation:
            print(f"    CVC attestation: {sr.cvc_validation.attestation_rate_type:.1%} "
                  f"(type), {sr.cvc_validation.attestation_rate_token:.1%} (token)")

    # Summary table
    print("\n" + "=" * 70)
    print("  Summary Comparison")
    print(f"  {'Strategy':<16} {'DictHit':>8} {'Signal':>8} "
          f"{'Sel':>6} {'Bi-z':>8} {'Len':>6} {'Net':>6}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*6}")
    for sr in strategies:
        print(f"  {sr.name:<16} {sr.dict_hit:>8.4f} {sr.n_signal_words:>8d} "
              f"{sr.mean_selectivity:>6.1f} {sr.bigram_z:>8.2f} "
              f"{sr.mean_word_length:>6.2f} {sr.net_signal:>6d}")

    # Pick winner
    winner = max(strategies, key=lambda s: (s.net_signal, s.bigram_z))
    verdict = (f"WINNER: {winner.name} (net_signal={winner.net_signal}, "
               f"bigram_z={winner.bigram_z})")

    result = CvcCompareResult(
        strategies=strategies,
        winner=winner.name,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )
    path = _save_json(rd, 'cvc_compare.json', result)
    print(f"\n  Winner: {winner.name}")
    print(f"  Saved: {path}")
    print(f"  Step 57.5 completed in {time.time() - t0:.1f}s")
