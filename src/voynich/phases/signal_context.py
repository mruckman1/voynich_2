"""
Phase 29.2 – Context of Confirmed Signal Words
==================================================
For each of the 8 confirmed signal words, examines decoded words at
positions ±1 and ±2, computes PMI, identifies new crib candidates,
and attempts chain extension to find multi-word Latin fragments.

Dependency chain:
    signal_bigrams.json   (Step 29.1 — per-token cache)
    signal_isolation.json (Step 28.4 — signal word list)
        → signal_context.json   (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)


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
class ContextNeighbor:
    word: str
    count: int
    pmi: float
    is_dict_hit: bool
    pos_tag: str


@dataclass
class ContextWindow:
    signal_word: str
    n_occurrences: int
    top_left: List[Dict]
    top_right: List[Dict]
    context_dict_hit_rate: float


@dataclass
class NewCrib:
    word: str
    evidence: str
    total_count: int
    n_signal_word_associations: int
    mean_pmi: float
    is_dict_hit: bool


@dataclass
class ChainCandidate:
    words: List[str]
    folio: str
    start_idx: int
    length: int
    n_signal: int
    n_dict_hits: int


@dataclass
class SignalContextResult:
    context_windows: List[Dict]
    n_new_crib_candidates: int
    new_crib_candidates: List[Dict]
    chain_candidates: List[Dict]
    n_chains_found: int
    longest_chain: int
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# POS heuristic
# ---------------------------------------------------------------------------

_SUFFIX_POS = [
    # Verb endings (longer first)
    ('ntur', 'VERB'), ('tur', 'VERB'), ('nt', 'VERB'),
    ('mus', 'VERB'), ('tis', 'VERB'),
    ('are', 'VERB'), ('ere', 'VERB'), ('ire', 'VERB'),
    ('ans', 'VERB'), ('ens', 'VERB'),
    # Noun / adjective endings
    ('orum', 'GEN_PL'), ('arum', 'GEN_PL'),
    ('ibus', 'DAT_ABL_PL'),
    ('ium', 'GEN_PL'), ('uum', 'GEN_PL'),
    ('um', 'NOUN_ACC'), ('am', 'NOUN_ACC'), ('em', 'NOUN_ACC'),
    ('us', 'NOUN_NOM'), ('er', 'NOUN_NOM'),
    ('ae', 'GEN_DAT'), ('is', 'GEN_DAT'),
    ('os', 'NOUN_ACC_PL'), ('as', 'NOUN_ACC_PL'), ('es', 'NOUN_NOM_PL'),
    ('a', 'NOUN_NOM'),
    ('i', 'GEN_DAT'),
    ('o', 'ABL_DAT'),
    ('e', 'ABL'),
]

# Known prepositions / function words
_PREPOSITIONS = {'de', 'in', 'ad', 'cum', 'per', 'pro', 'sub', 'ex', 'ab'}
_CONJUNCTIONS = {'et', 'vel', 'aut', 'sed', 'si', 'ne', 'ut'}


def _suffix_pos_heuristic(word: str) -> str:
    """Simple Latin POS tagger based on suffixes."""
    if word in _PREPOSITIONS:
        return 'PREP'
    if word in _CONJUNCTIONS:
        return 'CONJ'
    for suffix, pos in _SUFFIX_POS:
        if word.endswith(suffix) and len(word) > len(suffix):
            return pos
    return 'UNKNOWN'


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def _extract_context_windows(
    signal_words: List[str],
    decoded: List[str],
    classifications: List[str],
    folios: List[str],
    ref_word_set: set,
) -> List[ContextWindow]:
    """Extract ±1 context for each signal word at SIGNAL positions."""
    n = len(decoded)

    # Corpus-wide word frequencies (for PMI)
    word_freq = Counter(decoded)
    total_tokens = n

    # Adjacency pair counts (for PMI denominator)
    pair_freq: Counter = Counter()
    for i in range(n - 1):
        if folios[i] == folios[i + 1]:
            pair_freq[(decoded[i], decoded[i + 1])] += 1

    total_pairs = sum(pair_freq.values())

    windows = []
    for sw in signal_words:
        left_counts: Counter = Counter()
        right_counts: Counter = Counter()
        n_occ = 0

        for i in range(n):
            if decoded[i] == sw and classifications[i] == 'SIGNAL':
                n_occ += 1
                # Left neighbor
                if i > 0 and folios[i - 1] == folios[i]:
                    left_counts[decoded[i - 1]] += 1
                # Right neighbor
                if i < n - 1 and folios[i] == folios[i + 1]:
                    right_counts[decoded[i + 1]] += 1

        if n_occ == 0:
            windows.append(ContextWindow(
                signal_word=sw, n_occurrences=0,
                top_left=[], top_right=[],
                context_dict_hit_rate=0.0,
            ))
            continue

        # Compute PMI for top left neighbors
        top_left = []
        for word, count in left_counts.most_common(15):
            p_pair = pair_freq.get((word, sw), 0) / total_pairs if total_pairs > 0 else 0
            p_w = word_freq[word] / total_tokens
            p_sw = word_freq[sw] / total_tokens
            pmi = math.log2(p_pair / (p_w * p_sw)) if p_pair > 0 and p_w > 0 and p_sw > 0 else 0.0
            top_left.append(_convert(asdict(ContextNeighbor(
                word=word, count=count, pmi=round(pmi, 3),
                is_dict_hit=word in ref_word_set,
                pos_tag=_suffix_pos_heuristic(word),
            ))))

        # Compute PMI for top right neighbors
        top_right = []
        for word, count in right_counts.most_common(15):
            p_pair = pair_freq.get((sw, word), 0) / total_pairs if total_pairs > 0 else 0
            p_w = word_freq[word] / total_tokens
            p_sw = word_freq[sw] / total_tokens
            pmi = math.log2(p_pair / (p_w * p_sw)) if p_pair > 0 and p_w > 0 and p_sw > 0 else 0.0
            top_right.append(_convert(asdict(ContextNeighbor(
                word=word, count=count, pmi=round(pmi, 3),
                is_dict_hit=word in ref_word_set,
                pos_tag=_suffix_pos_heuristic(word),
            ))))

        # Context dict hit rate
        all_neighbors = list(left_counts.elements()) + list(right_counts.elements())
        ctx_hits = sum(1 for w in all_neighbors if w in ref_word_set)
        ctx_rate = ctx_hits / len(all_neighbors) if all_neighbors else 0.0

        windows.append(ContextWindow(
            signal_word=sw,
            n_occurrences=n_occ,
            top_left=top_left,
            top_right=top_right,
            context_dict_hit_rate=round(ctx_rate, 4),
        ))

    return windows


# ---------------------------------------------------------------------------
# New crib identification
# ---------------------------------------------------------------------------

def _identify_new_cribs(
    context_windows: List[ContextWindow],
    ref_word_set: set,
    signal_words: List[str],
    min_associations: int = 2,
    min_pmi: float = 0.5,
) -> List[NewCrib]:
    """Identify new crib candidates from context patterns.

    A new crib is a word that:
    - Appears as a neighbor of ≥2 different signal words
    - Is a dict hit
    - Has mean PMI > min_pmi
    - Is not already a signal word
    """
    signal_word_set = set(signal_words)
    # Collect: word -> list of (signal_word, pmi, count)
    word_evidence: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)

    for cw in context_windows:
        for neighbor in cw.top_left + cw.top_right:
            word = neighbor['word']
            if word in signal_word_set:
                continue
            if neighbor['is_dict_hit']:
                word_evidence[word].append((
                    cw.signal_word, neighbor['pmi'], neighbor['count'],
                ))

    cribs = []
    for word, evidence_list in word_evidence.items():
        # Unique signal words this neighbor is associated with
        assoc_sws = set(e[0] for e in evidence_list)
        if len(assoc_sws) < min_associations:
            continue
        mean_pmi = sum(e[1] for e in evidence_list) / len(evidence_list)
        if mean_pmi < min_pmi:
            continue
        total_count = sum(e[2] for e in evidence_list)
        cribs.append(NewCrib(
            word=word,
            evidence=', '.join(f'{sw}(PMI={pmi:.1f})' for sw, pmi, _ in evidence_list),
            total_count=total_count,
            n_signal_word_associations=len(assoc_sws),
            mean_pmi=round(mean_pmi, 3),
            is_dict_hit=word in ref_word_set,
        ))

    cribs.sort(key=lambda c: (-c.n_signal_word_associations, -c.mean_pmi))
    return cribs


# ---------------------------------------------------------------------------
# Chain extension
# ---------------------------------------------------------------------------

def _find_chains(
    decoded: List[str],
    classifications: List[str],
    folios: List[str],
    dict_hits: List[bool],
    min_length: int = 3,
) -> List[ChainCandidate]:
    """Find maximal consecutive runs of dict-hit tokens containing ≥1 SIGNAL.

    A chain is a maximal consecutive sequence of dictionary-hit tokens
    (regardless of classification) that contains at least one SIGNAL token,
    within the same folio.
    """
    n = len(decoded)
    chains: List[ChainCandidate] = []

    i = 0
    while i < n:
        if not dict_hits[i]:
            i += 1
            continue
        # Start of a potential chain
        start = i
        folio = folios[i]
        while i < n and dict_hits[i] and folios[i] == folio:
            i += 1
        end = i  # exclusive

        length = end - start
        if length < min_length:
            continue

        words = decoded[start:end]
        n_sig = sum(
            1 for j in range(start, end)
            if classifications[j] == 'SIGNAL'
        )
        if n_sig == 0:
            continue

        chains.append(ChainCandidate(
            words=words,
            folio=folio,
            start_idx=start,
            length=length,
            n_signal=n_sig,
            n_dict_hits=length,
        ))

    chains.sort(key=lambda c: (-c.length, -c.n_signal))
    return chains


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_signal_context() -> None:
    """Step 29.2: Context of confirmed signal words."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 29.2: Context of Confirmed Signal Words")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if not os.path.exists(bg_path):
        print("  [SKIP] signal_bigrams.json not found — run signal-bigram first")
        return
    with open(bg_path) as f:
        bg_data = json.load(f)

    decoded = bg_data['token_decoded']
    classifications = bg_data['token_classifications']
    folios = bg_data['token_folios']
    dict_hits = bg_data['token_dict_hits']

    # Load signal words from Phase 28.4
    sig_path = os.path.join(rd, 'signal_isolation.json')
    if not os.path.exists(sig_path):
        print("  [SKIP] signal_isolation.json not found")
        return
    with open(sig_path) as f:
        sig_data = json.load(f)
    signal_words = [
        ws['word'] for ws in sig_data.get('word_signals', [])
        if ws.get('is_genuine_signal')
    ]
    print(f"     {len(signal_words)} signal words: {signal_words}")
    print(f"     {len(decoded)} tokens loaded from signal_bigrams.json")

    # Build reference word set
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    # ── 2. Extract context windows ──
    print("\n  2. Extracting context windows …")
    context_windows = _extract_context_windows(
        signal_words, decoded, classifications, folios, ref_word_set,
    )

    for cw in context_windows:
        print(f"\n     {cw.signal_word} ({cw.n_occurrences} SIGNAL occurrences)")
        if cw.top_left:
            top_l = cw.top_left[0]
            print(f"       Top left:  {top_l['word']:12s} "
                  f"(n={top_l['count']}, PMI={top_l['pmi']:.2f}, "
                  f"POS={top_l['pos_tag']})")
        if cw.top_right:
            top_r = cw.top_right[0]
            print(f"       Top right: {top_r['word']:12s} "
                  f"(n={top_r['count']}, PMI={top_r['pmi']:.2f}, "
                  f"POS={top_r['pos_tag']})")
        print(f"       Context dict_hit rate: {cw.context_dict_hit_rate:.1%}")

    # ── 3. Identify new cribs ──
    print("\n  3. Identifying new crib candidates …")
    new_cribs = _identify_new_cribs(
        context_windows, ref_word_set, signal_words,
    )

    print(f"     {len(new_cribs)} new crib candidates")
    for nc in new_cribs[:10]:
        print(f"       {nc.word:12s}  assoc={nc.n_signal_word_associations}  "
              f"PMI={nc.mean_pmi:.2f}  count={nc.total_count}  "
              f"evidence: {nc.evidence}")

    # ── 4. Chain extension ──
    print("\n  4. Finding dict-hit chains containing SIGNAL tokens …")
    chains = _find_chains(decoded, classifications, folios, dict_hits)

    print(f"     {len(chains)} chains of length ≥ 3")
    longest_chain = chains[0].length if chains else 0
    for ch in chains[:10]:
        sig_frac = ch.n_signal / ch.length
        print(f"       {ch.folio} idx={ch.start_idx:5d} "
              f"len={ch.length:3d} signal={ch.n_signal:2d} "
              f"({sig_frac:.0%}): "
              f"{' '.join(ch.words[:8])}{'…' if ch.length > 8 else ''}")

    # ── 5. Gate and verdict ──
    n_chains = len(chains)
    gate_passed = len(new_cribs) >= 2 or longest_chain >= 5
    verdict = (
        f"{len(new_cribs)} new crib candidates, "
        f"{n_chains} chains (longest={longest_chain}). "
        f"{'Context provides new evidence.' if gate_passed else 'Limited contextual signal.'}"
    )
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 6. Save ──
    result = SignalContextResult(
        context_windows=[_convert(asdict(cw)) for cw in context_windows],
        n_new_crib_candidates=len(new_cribs),
        new_crib_candidates=[_convert(asdict(nc)) for nc in new_cribs[:30]],
        chain_candidates=[_convert(asdict(ch)) for ch in chains[:50]],
        n_chains_found=n_chains,
        longest_chain=longest_chain,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'signal_context.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
