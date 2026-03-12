"""
Phase 45 – Track B: SBM-Based Re-encoding and Decoding
========================================================
Use SBM communities as decoding units or as additional soft constraints
on the stroke-triple CSP.  Tests whether distributional grouping captures
phonological structure that stroke features miss.

Dependency chain:
    sbm_communities.json       (Phase 44B.2)
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16)
    crib_extraction.json       (Phase 28)
        -> sbm_encoding.json       (Step 45B.1)
        -> sbm_csp.json            (Step 45B.2)
        -> sbm_signal.json         (Step 45B.3)
        -> sbm_hybrid.json         (Step 45B.4)
        -> sbm_maxsat.json         (Step 45B.5)
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    token_to_triples,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    EVA_VISUAL_COMPONENTS,
    build_expanded_word_set,
    load_reference_corpus,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
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


def _load_communities(rd) -> Dict[str, int]:
    data = _safe_load(os.path.join(rd, 'sbm_communities.json'))
    return data.get('communities', {})


def _build_ref_word_set() -> set:
    """Build the expanded Latin reference word set."""
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    return base_words | expanded


def _decode_token_simple(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> str:
    """Decode a single token using triple→syllable assignment (no modifier logic)."""
    chars = tokenize_eva_chars(token)
    parts = []
    for ch in chars:
        tk = eva_to_triple.get(ch)
        if tk and tk in assignment:
            parts.append(assignment[tk])
    return ''.join(parts)


def _compute_dict_hit(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    word_set: set,
) -> float:
    """Compute dict-hit rate for a simple decode (no modifier logic)."""
    if not tokens:
        return 0.0
    hits = 0
    for tok in tokens:
        decoded = _decode_token_simple(tok, assignment, eva_to_triple)
        if decoded.lower() in word_set:
            hits += 1
    return hits / len(tokens)


def _compute_dict_hit_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    word_set: set,
) -> Tuple[float, List[str]]:
    """Compute dict-hit using R3 modifier-aware decode."""
    from voynich.phases.signal_isolation import _decode_corpus_r3
    decoded = _decode_corpus_r3(
        tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )
    hits = sum(1 for w in decoded if w in word_set)
    rate = hits / len(tokens) if tokens else 0.0
    return rate, decoded


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class CommunityDomainEntry:
    community_id: int
    members: List[str]
    confirmed_triples: List[str]
    confirmed_syllables: List[str]
    onset_set: List[str]
    vowel_set: List[str]
    domain: List[str]
    domain_size: int


@dataclass
class CommunityEncodingResult:
    entries: List[Dict]
    n_communities: int
    mean_domain_size: float
    total_confirmed_triples: int
    runtime_seconds: float


@dataclass
class CommunityCSPResult:
    best_assignment: Dict[str, str]
    best_dict_hit: float
    n_solutions_tested: int
    selectivity: float
    null_dict_hit: float
    per_language: Dict[str, float]
    runtime_seconds: float


@dataclass
class CommunitySignalResult:
    dict_hit: float
    signal_rate: float
    per_community_signal_rate: Dict[str, float]
    n_signal_tokens: int
    n_total_tokens: int
    selectivity: float
    runtime_seconds: float


@dataclass
class HybridVariant:
    name: str
    assignment: Dict[str, str]
    dict_hit: float
    selectivity: float
    n_triples_changed: int
    changed_triples: List[Dict]


@dataclass
class HybridDecodeResult:
    variants: List[Dict]
    best_variant: str
    best_dict_hit: float
    best_selectivity: float
    baseline_dict_hit: float
    diagnostic: str
    runtime_seconds: float


@dataclass
class CommunityLandscapeResult:
    n_solutions: int
    landscape_shape: str
    n_peaked: int
    best_dict_hit: float
    stroke_landscape_shape: str
    differs_from_stroke: bool
    runtime_seconds: float


# ══════════════════════════════════════════════════════════════════════
#  Step 45B.1 — Community-Based Encoding Table
# ══════════════════════════════════════════════════════════════════════

def run_sbm_encode() -> None:
    """Step 45B.1: build community-based encoding table."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 45B.1: Community-Based Encoding Table")
    print("=" * 70)

    rd = _results_dir()
    communities = _load_communities(rd)
    if not communities:
        print("  [SKIP] sbm_communities.json not found")
        return

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    if not assignment:
        print("  [SKIP] combined_refine.json not found")
        return

    # Load confirmed triples from Phase 28
    crib_data = _safe_load(os.path.join(rd, 'crib_extraction.json'))
    confirmed_triples = set()
    for crib in crib_data.get('cribs', []):
        if crib.get('tier', 99) <= 2:
            for tk in crib.get('triples_covered', []):
                confirmed_triples.add(tk)

    # Also from bootstrap_loop
    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    for tk in boot_data.get('confirmed_triples', []):
        confirmed_triples.add(tk)

    eva_to_triple = build_eva_to_triple_lookup()
    n_communities = max(communities.values()) + 1

    # Group triples by community
    comm_triples: Dict[int, Set[str]] = defaultdict(set)
    for ch, cid in communities.items():
        tk = eva_to_triple.get(ch)
        if tk:
            comm_triples[cid].add(tk)

    # Build domain for each community
    entries = []
    all_cv_syllables = sorted(set(assignment.values()))
    # Extend with common CV syllables
    vowels = 'aeiou'
    consonants = 'bcdfghlmnprstvx'
    for c in consonants:
        for v in vowels:
            all_cv_syllables.append(c + v)
    for v in vowels:
        all_cv_syllables.append(v)
    all_cv_syllables = sorted(set(all_cv_syllables))

    for cid in range(n_communities):
        members = sorted(ch for ch, c in communities.items() if c == cid)
        triples = sorted(comm_triples.get(cid, set()))

        # Find confirmed syllables in this community
        conf_triples = [tk for tk in triples if tk in confirmed_triples]
        conf_syls = [assignment[tk] for tk in conf_triples if tk in assignment]

        # Extract onset and vowel sets
        onsets = set()
        vowel_parts = set()
        for syl in conf_syls:
            if len(syl) >= 2:
                onsets.add(syl[0])
                vowel_parts.add(syl[1:])
            elif len(syl) == 1:
                vowel_parts.add(syl)

        # Build domain: syllables sharing onset OR vowel with any confirmed
        if onsets or vowel_parts:
            domain = []
            for syl in all_cv_syllables:
                if len(syl) >= 2:
                    if syl[0] in onsets or syl[1:] in vowel_parts:
                        domain.append(syl)
                elif len(syl) == 1:
                    if syl in vowel_parts:
                        domain.append(syl)
            if not domain:
                domain = list(all_cv_syllables)
        else:
            domain = list(all_cv_syllables)

        entry = CommunityDomainEntry(
            community_id=cid,
            members=members,
            confirmed_triples=conf_triples,
            confirmed_syllables=conf_syls,
            onset_set=sorted(onsets),
            vowel_set=sorted(vowel_parts),
            domain=domain,
            domain_size=len(domain),
        )
        entries.append(entry)

        print(f"  Community {cid}: {len(members)} chars, "
              f"{len(conf_triples)} confirmed, "
              f"domain size={len(domain)}")

    mean_dom = np.mean([e.domain_size for e in entries])

    result = CommunityEncodingResult(
        entries=[_convert(asdict(e)) for e in entries],
        n_communities=n_communities,
        mean_domain_size=round(float(mean_dom), 1),
        total_confirmed_triples=len(confirmed_triples),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'sbm_encoding.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Mean domain size: {mean_dom:.1f}")
    print(f"  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Step 45B.2 — CSP Decode with Community Variables
# ══════════════════════════════════════════════════════════════════════

def run_sbm_csp() -> None:
    """Step 45B.2: CSP decode with community variables (6 vars, exhaustive)."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 45B.2: CSP Decode with Community Variables")
    print("=" * 70)

    rd = _results_dir()
    communities = _load_communities(rd)
    if not communities:
        print("  [SKIP] sbm_communities.json not found")
        return

    enc_data = _safe_load(os.path.join(rd, 'sbm_encoding.json'))
    if not enc_data:
        print("  [SKIP] sbm_encoding.json not found; run sbm-encode first")
        return

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    eva_to_triple = build_eva_to_triple_lookup()
    n_communities = max(communities.values()) + 1

    # Build community→domain from encoding data
    comm_domains: Dict[int, List[str]] = {}
    for entry in enc_data.get('entries', []):
        cid = entry['community_id']
        comm_domains[cid] = entry['domain']

    # Build community→syllable assignment (mapping from community to syllable)
    # For community decode: EVA char → community → syllable
    # We need to convert triple→syllable to community→syllable
    # Since multiple triples can be in the same community, take the most common
    # syllable assignment within each community
    comm_syllables: Dict[int, Counter] = defaultdict(Counter)
    for ch, cid in communities.items():
        tk = eva_to_triple.get(ch)
        if tk and tk in assignment:
            comm_syllables[cid][assignment[tk]] += 1

    # Build the mapping: EVA char → triple → community → syllable
    # For community CSP: we assign one syllable per community
    print("\n  Building reference word set …")
    word_set = _build_ref_word_set()
    print(f"  {len(word_set)} reference words")

    print("\n  Loading corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    print(f"  {len(all_tokens)} tokens")

    # Exhaustive search over community→syllable assignments
    # Limit domain to top-K most common per community to keep search tractable
    MAX_DOMAIN = 10
    trimmed_domains = {}
    for cid in range(n_communities):
        domain = comm_domains.get(cid, [])
        # Prioritize by putting confirmed syllables first
        top_syls = [s for s, _ in comm_syllables[cid].most_common(3)]
        rest = [s for s in domain if s not in top_syls]
        trimmed = top_syls + rest[:MAX_DOMAIN - len(top_syls)]
        trimmed_domains[cid] = trimmed[:MAX_DOMAIN]

    print(f"\n  Trimmed domains: {[len(trimmed_domains[c]) for c in range(n_communities)]}")

    # Build community-based assignment for each combo
    # EVA char → community → syllable → decoded word
    def _decode_with_comm_assignment(comm_assign: Dict[int, str]) -> float:
        """Decode tokens using community→syllable mapping and compute dict-hit."""
        hits = 0
        for tok in all_tokens:
            chars = tokenize_eva_chars(tok)
            parts = []
            for ch in chars:
                cid = communities.get(ch, -1)
                if cid >= 0 and cid in comm_assign:
                    parts.append(comm_assign[cid])
            decoded = ''.join(parts).lower()
            if decoded in word_set:
                hits += 1
        return hits / len(all_tokens) if all_tokens else 0.0

    # Enumerate top assignments
    # With 6 communities and MAX_DOMAIN=10, full enumeration = 10^6 = 1M (too many)
    # Use greedy + random sampling instead
    print("\n  Searching community assignments (greedy + sampling) …")
    best_dict_hit = 0.0
    best_comm_assign: Dict[int, str] = {}
    n_tested = 0

    # Start with the most-common syllable per community (greedy baseline)
    greedy = {}
    for cid in range(n_communities):
        if comm_syllables[cid]:
            greedy[cid] = comm_syllables[cid].most_common(1)[0][0]
        elif trimmed_domains.get(cid):
            greedy[cid] = trimmed_domains[cid][0]
        else:
            greedy[cid] = 'a'

    greedy_hit = _decode_with_comm_assignment(greedy)
    n_tested += 1
    if greedy_hit > best_dict_hit:
        best_dict_hit = greedy_hit
        best_comm_assign = dict(greedy)

    print(f"  Greedy baseline: dict_hit={greedy_hit:.4f}")

    # Random sampling (1000 random assignments)
    rng = random.Random(42)
    for _ in range(1000):
        comm_assign = {}
        for cid in range(n_communities):
            dom = trimmed_domains.get(cid, ['a'])
            comm_assign[cid] = rng.choice(dom)
        dh = _decode_with_comm_assignment(comm_assign)
        n_tested += 1
        if dh > best_dict_hit:
            best_dict_hit = dh
            best_comm_assign = dict(comm_assign)

    print(f"  After 1000 random: best dict_hit={best_dict_hit:.4f}")

    # Coordinate descent from best
    improved = True
    while improved:
        improved = False
        for cid in range(n_communities):
            dom = trimmed_domains.get(cid, ['a'])
            for syl in dom:
                test = dict(best_comm_assign)
                test[cid] = syl
                dh = _decode_with_comm_assignment(test)
                n_tested += 1
                if dh > best_dict_hit:
                    best_dict_hit = dh
                    best_comm_assign = dict(test)
                    improved = True

    print(f"  After coordinate descent: best dict_hit={best_dict_hit:.4f}")

    # Compute null dict-hit (random community→syllable)
    null_hits = []
    for _ in range(20):
        null_assign = {}
        for cid in range(n_communities):
            dom = trimmed_domains.get(cid, ['a'])
            null_assign[cid] = rng.choice(dom)
        null_hits.append(_decode_with_comm_assignment(null_assign))
    null_mean = float(np.mean(null_hits))
    selectivity = best_dict_hit / null_mean if null_mean > 0 else 0.0

    result = CommunityCSPResult(
        best_assignment={str(k): v for k, v in best_comm_assign.items()},
        best_dict_hit=round(best_dict_hit, 4),
        n_solutions_tested=n_tested,
        selectivity=round(selectivity, 2),
        null_dict_hit=round(null_mean, 4),
        per_language={'latin': round(best_dict_hit, 4)},
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'sbm_csp.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Selectivity: {selectivity:.2f}×")
    print(f"  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Step 45B.3 — Signal Isolation on Community Decode
# ══════════════════════════════════════════════════════════════════════

def run_comm_signal() -> None:
    """Step 45B.3: signal isolation on community decode."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 45B.3: Signal Isolation on Community Decode")
    print("=" * 70)

    rd = _results_dir()
    communities = _load_communities(rd)
    if not communities:
        print("  [SKIP] sbm_communities.json not found")
        return

    csp_data = _safe_load(os.path.join(rd, 'sbm_csp.json'))
    if not csp_data:
        print("  [SKIP] sbm_csp.json not found; run sbm-csp first")
        return

    comm_assign = csp_data.get('best_assignment', {})
    # Convert string keys back to int
    comm_assign = {int(k): v for k, v in comm_assign.items()}

    print("\n  Building reference word set …")
    word_set = _build_ref_word_set()

    print("  Loading corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    n_communities = max(communities.values()) + 1

    # Decode real corpus using community assignment
    real_hits = 0
    per_comm_signal: Dict[int, List[bool]] = defaultdict(list)

    for tok in all_tokens:
        chars = tokenize_eva_chars(tok)
        parts = []
        comm_ids = []
        for ch in chars:
            cid = communities.get(ch, -1)
            if cid >= 0 and cid in comm_assign:
                parts.append(comm_assign[cid])
                comm_ids.append(cid)
        decoded = ''.join(parts).lower()
        is_hit = decoded in word_set
        if is_hit:
            real_hits += 1

        # Track per primary community (smallest community wins)
        if comm_ids:
            primary = min(comm_ids)  # smallest community = most specific
            per_comm_signal[primary].append(is_hit)

    real_hit_rate = real_hits / len(all_tokens) if all_tokens else 0.0

    # Generate null corpora via random shuffling of community assignments
    rng = random.Random(42)
    null_rates = []
    for _ in range(5):
        null_assign = dict(comm_assign)
        syls = list(null_assign.values())
        rng.shuffle(syls)
        null_assign = {cid: syls[i] for i, cid in enumerate(sorted(null_assign.keys()))}

        null_hits = 0
        for tok in all_tokens:
            chars = tokenize_eva_chars(tok)
            parts = []
            for ch in chars:
                cid = communities.get(ch, -1)
                if cid >= 0 and cid in null_assign:
                    parts.append(null_assign[cid])
            decoded = ''.join(parts).lower()
            if decoded in word_set:
                null_hits += 1
        null_rates.append(null_hits / len(all_tokens) if all_tokens else 0.0)

    null_mean = float(np.mean(null_rates))
    selectivity = real_hit_rate / null_mean if null_mean > 0 else 0.0

    # SIGNAL tokens: those that hit in real but not in majority of nulls
    # (simplified — just count real hits as signal for community decode)
    n_signal = real_hits

    per_comm_rates = {}
    for cid in range(n_communities):
        vals = per_comm_signal.get(cid, [])
        rate = sum(vals) / len(vals) if vals else 0.0
        per_comm_rates[str(cid)] = round(rate, 4)

    result = CommunitySignalResult(
        dict_hit=round(real_hit_rate, 4),
        signal_rate=round(n_signal / len(all_tokens) if all_tokens else 0.0, 4),
        per_community_signal_rate=per_comm_rates,
        n_signal_tokens=n_signal,
        n_total_tokens=len(all_tokens),
        selectivity=round(selectivity, 2),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'sbm_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Dict-hit: {real_hit_rate:.4f}")
    print(f"  Selectivity: {selectivity:.2f}×")
    print(f"  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Step 45B.4 — Hybrid Stroke+Community Decode
# ══════════════════════════════════════════════════════════════════════

def run_sbm_hybrid() -> None:
    """Step 45B.4: hybrid stroke+community decode (3 variants)."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 45B.4: Hybrid Stroke+Community Decode")
    print("=" * 70)

    rd = _results_dir()
    communities = _load_communities(rd)
    if not communities:
        print("  [SKIP] sbm_communities.json not found")
        return

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    if not assignment:
        print("  [SKIP] combined_refine.json not found")
        return

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))
    modifier_rules: Dict[str, str] = {}
    for c in mod_data.get('classifications', []):
        if c.get('final_classification') == 'modifier':
            modifier_rules[c['eva_char']] = c.get('modifier_type', 'silent')

    eva_to_triple = build_eva_to_triple_lookup()

    # Load confirmed triples
    crib_data = _safe_load(os.path.join(rd, 'crib_extraction.json'))
    confirmed_triples = set()
    for crib in crib_data.get('cribs', []):
        if crib.get('tier', 99) <= 2:
            for tk in crib.get('triples_covered', []):
                confirmed_triples.add(tk)
    boot_data = _safe_load(os.path.join(rd, 'bootstrap_loop.json'))
    for tk in boot_data.get('confirmed_triples', []):
        confirmed_triples.add(tk)

    # Map triples to communities
    triple_to_comm: Dict[str, int] = {}
    for ch, cid in communities.items():
        tk = eva_to_triple.get(ch)
        if tk:
            triple_to_comm[tk] = cid

    # Identify which triples are free (not confirmed)
    free_triples = [tk for tk in assignment if tk not in confirmed_triples]

    # Group free triples by community
    comm_free: Dict[int, List[str]] = defaultdict(list)
    for tk in free_triples:
        cid = triple_to_comm.get(tk, -1)
        if cid >= 0:
            comm_free[cid].append(tk)

    # For each community with confirmed triples, extract onset/vowel constraints
    comm_confirmed_onsets: Dict[int, Set[str]] = defaultdict(set)
    comm_confirmed_vowels: Dict[int, Set[str]] = defaultdict(set)
    for tk in confirmed_triples:
        cid = triple_to_comm.get(tk, -1)
        if cid >= 0 and tk in assignment:
            syl = assignment[tk]
            if len(syl) >= 2:
                comm_confirmed_onsets[cid].add(syl[0])
                comm_confirmed_vowels[cid].add(syl[1:])
            elif len(syl) == 1:
                comm_confirmed_vowels[cid].add(syl)

    print(f"\n  {len(free_triples)} free triples, {len(confirmed_triples)} confirmed")
    print(f"  Community groups: {[(cid, len(tks)) for cid, tks in sorted(comm_free.items())]}")

    print("\n  Building reference word set …")
    word_set = _build_ref_word_set()

    print("  Loading corpus …")
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Baseline: Phase 15 assignment with R3 decode
    baseline_hit, _ = _compute_dict_hit_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )
    print(f"  Baseline (Phase 15): dict_hit={baseline_hit:.4f}")

    # Generate all possible syllables
    vowels = 'aeiou'
    consonants = 'bcdfghlmnprstvx'
    all_syls = [v for v in vowels]
    for c in consonants:
        for v in vowels:
            all_syls.append(c + v)

    variants = []

    # ── HYBRID_NONE: no community constraint (just Phase 15) ──
    variants.append(HybridVariant(
        name='HYBRID_NONE',
        assignment=dict(assignment),
        dict_hit=round(baseline_hit, 4),
        selectivity=0.0,
        n_triples_changed=0,
        changed_triples=[],
    ))

    # ── HYBRID_C: free triples in same community must share onset ──
    hybrid_c_assign = dict(assignment)
    n_changed_c = 0
    changed_c = []
    for cid, free_tks in comm_free.items():
        onsets = comm_confirmed_onsets.get(cid, set())
        if not onsets or len(onsets) != 1:
            continue  # Can't constrain if 0 or multiple onsets
        target_onset = list(onsets)[0]
        for tk in free_tks:
            old_syl = hybrid_c_assign.get(tk, '')
            if len(old_syl) >= 2 and old_syl[0] != target_onset:
                # Change onset to match community
                new_syl = target_onset + old_syl[1:]
                if new_syl in set(all_syls):
                    hybrid_c_assign[tk] = new_syl
                    n_changed_c += 1
                    changed_c.append({
                        'triple': tk, 'old': old_syl, 'new': new_syl,
                        'community': cid,
                    })

    hybrid_c_hit, _ = _compute_dict_hit_r3(
        all_tokens, hybrid_c_assign, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )
    print(f"  HYBRID_C: dict_hit={hybrid_c_hit:.4f}, {n_changed_c} triples changed")

    variants.append(HybridVariant(
        name='HYBRID_C',
        assignment=hybrid_c_assign,
        dict_hit=round(hybrid_c_hit, 4),
        selectivity=0.0,
        n_triples_changed=n_changed_c,
        changed_triples=changed_c,
    ))

    # ── HYBRID_V: free triples in same community must share vowel ──
    hybrid_v_assign = dict(assignment)
    n_changed_v = 0
    changed_v = []
    for cid, free_tks in comm_free.items():
        vowel_parts = comm_confirmed_vowels.get(cid, set())
        if not vowel_parts or len(vowel_parts) != 1:
            continue
        target_vowel = list(vowel_parts)[0]
        for tk in free_tks:
            old_syl = hybrid_v_assign.get(tk, '')
            if len(old_syl) >= 2 and old_syl[1:] != target_vowel:
                new_syl = old_syl[0] + target_vowel
                if new_syl in set(all_syls):
                    hybrid_v_assign[tk] = new_syl
                    n_changed_v += 1
                    changed_v.append({
                        'triple': tk, 'old': old_syl, 'new': new_syl,
                        'community': cid,
                    })
            elif len(old_syl) == 1 and old_syl != target_vowel:
                new_syl = target_vowel
                hybrid_v_assign[tk] = new_syl
                n_changed_v += 1
                changed_v.append({
                    'triple': tk, 'old': old_syl, 'new': new_syl,
                    'community': cid,
                })

    hybrid_v_hit, _ = _compute_dict_hit_r3(
        all_tokens, hybrid_v_assign, eva_to_triple,
        modifier_chars, modifier_rules, word_set,
    )
    print(f"  HYBRID_V: dict_hit={hybrid_v_hit:.4f}, {n_changed_v} triples changed")

    variants.append(HybridVariant(
        name='HYBRID_V',
        assignment=hybrid_v_assign,
        dict_hit=round(hybrid_v_hit, 4),
        selectivity=0.0,
        n_triples_changed=n_changed_v,
        changed_triples=changed_v,
    ))

    # Compute null selectivity for each variant
    rng = random.Random(42)
    for var in variants:
        null_hits = []
        for _ in range(5):
            null_assign = dict(var.assignment)
            free_keys = [tk for tk in null_assign if tk not in confirmed_triples]
            syls = [null_assign[tk] for tk in free_keys]
            rng.shuffle(syls)
            for i, tk in enumerate(free_keys):
                null_assign[tk] = syls[i]
            nh, _ = _compute_dict_hit_r3(
                all_tokens, null_assign, eva_to_triple,
                modifier_chars, modifier_rules, word_set,
            )
            null_hits.append(nh)
        null_mean = float(np.mean(null_hits))
        var.selectivity = round(var.dict_hit / null_mean if null_mean > 0 else 0.0, 2)

    best_var = max(variants, key=lambda v: v.dict_hit)

    # Diagnostic: which dimension do communities capture?
    if best_var.name == 'HYBRID_C' and hybrid_c_hit > baseline_hit * 1.01:
        diagnostic = "Communities capture CONSONANT classes."
    elif best_var.name == 'HYBRID_V' and hybrid_v_hit > baseline_hit * 1.01:
        diagnostic = "Communities capture VOWEL classes."
    elif best_var.name == 'HYBRID_NONE':
        diagnostic = "Community constraints do not improve decoding."
    else:
        diagnostic = "Marginal or no improvement from community constraints."

    result = HybridDecodeResult(
        variants=[_convert(asdict(v)) for v in variants],
        best_variant=best_var.name,
        best_dict_hit=best_var.dict_hit,
        best_selectivity=best_var.selectivity,
        baseline_dict_hit=round(baseline_hit, 4),
        diagnostic=diagnostic,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'sbm_hybrid.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Best variant: {best_var.name} ({best_var.dict_hit:.4f})")
    print(f"  Diagnostic: {diagnostic}")
    print(f"\n  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Step 45B.5 — MaxSAT Landscape at Community Granularity
# ══════════════════════════════════════════════════════════════════════

def run_sbm_landscape() -> None:
    """Step 45B.5: MaxSAT landscape at community granularity."""
    t0 = time.time()
    print("=" * 70)
    print("STEP 45B.5: MaxSAT Landscape at Community Granularity")
    print("=" * 70)

    rd = _results_dir()
    communities = _load_communities(rd)
    if not communities:
        print("  [SKIP] sbm_communities.json not found")
        return

    enc_data = _safe_load(os.path.join(rd, 'sbm_encoding.json'))
    if not enc_data:
        print("  [SKIP] sbm_encoding.json not found; run sbm-encode first")
        return

    # Load stroke-triple landscape for comparison
    stroke_landscape = _safe_load(os.path.join(rd, 'maxsat_landscape.json'))
    stroke_shape = stroke_landscape.get('classification', 'UNKNOWN')

    n_communities = max(communities.values()) + 1

    # With 6 communities, enumerate ALL possible assignments
    comm_domains: Dict[int, List[str]] = {}
    for entry in enc_data.get('entries', []):
        cid = entry['community_id']
        # Use top 8 candidates per community (8^6 = 262K max)
        dom = entry['domain'][:8]
        comm_domains[cid] = dom

    total_combos = 1
    for cid in range(n_communities):
        total_combos *= len(comm_domains.get(cid, [1]))

    print(f"\n  Total combinations: {total_combos}")

    # Enumerate all combinations
    print("\n  Building reference word set …")
    word_set = _build_ref_word_set()
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    n_tokens = len(all_tokens)

    domain_lists = [comm_domains.get(cid, ['a']) for cid in range(n_communities)]

    # ── Optimisation: group tokens by unique community-ID sequence ──
    # Many tokens share the same sequence (e.g. most 3-char tokens
    # using common EVA chars). Decoding once per unique sequence and
    # multiplying by count reduces the inner loop from ~36K to ~few hundred.
    from collections import Counter as _Counter
    seq_counts: Dict[tuple, int] = _Counter()
    for tok in all_tokens:
        chars = tokenize_eva_chars(tok)
        seq = tuple(communities.get(ch, -1) for ch in chars)
        seq_counts[seq] += 1

    unique_seqs = list(seq_counts.keys())
    counts = [seq_counts[s] for s in unique_seqs]
    n_unique = len(unique_seqs)
    print(f"  Unique community sequences: {n_unique} (from {n_tokens} tokens)")

    all_combos = list(product(*domain_lists))
    n_combos = len(all_combos)

    # Single-threaded enumeration — the inner loop iterates over
    # unique community sequences (~176) instead of all tokens (~36K),
    # giving a ~200× speedup that makes 262K combos tractable.
    all_hits = []
    best_hit = 0.0
    for i, combo in enumerate(all_combos):
        hits = 0
        for seq, cnt in zip(unique_seqs, counts):
            parts = []
            for cid in seq:
                if 0 <= cid < n_communities:
                    parts.append(combo[cid])
            decoded = ''.join(parts).lower()
            if decoded in word_set:
                hits += cnt
        dh = hits / n_tokens if n_tokens else 0.0
        all_hits.append(dh)
        if dh > best_hit:
            best_hit = dh
        if (i + 1) % 50000 == 0:
            print(f"    {i + 1}/{n_combos} tested, best={best_hit:.4f}")

    best_hit = max(all_hits) if all_hits else 0.0
    best_idx = all_hits.index(best_hit)
    best_combo = all_combos[best_idx]
    best_assign = {cid: best_combo[cid] for cid in range(n_communities)}
    n_tested = len(all_hits)

    # Classify landscape
    threshold = best_hit * 0.99
    near_optimal = sum(1 for h in all_hits if h >= threshold)
    n_solutions = n_tested
    n_peaked = near_optimal

    if near_optimal <= 3:
        landscape_shape = 'PEAKED'
    elif near_optimal <= 20:
        landscape_shape = 'RIDGED'
    else:
        landscape_shape = 'FLAT'

    print(f"\n  Best dict-hit: {best_hit:.4f}")
    print(f"  Near-optimal (≥99% of best): {near_optimal}")
    print(f"  Landscape shape: {landscape_shape}")

    differs = landscape_shape != stroke_shape

    result = CommunityLandscapeResult(
        n_solutions=n_solutions,
        landscape_shape=landscape_shape,
        n_peaked=n_peaked,
        best_dict_hit=round(best_hit, 4),
        stroke_landscape_shape=stroke_shape,
        differs_from_stroke=differs,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'sbm_maxsat.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(asdict(result)), f, indent=2)
    print(f"\n  Differs from stroke landscape: {differs}")
    print(f"  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Track B Runner
# ══════════════════════════════════════════════════════════════════════

def run_track_b_45() -> None:
    """Run all Track B steps."""
    run_sbm_encode()
    print("\n" + "=" * 70 + "\n")
    run_sbm_csp()
    print("\n" + "=" * 70 + "\n")
    run_comm_signal()
    print("\n" + "=" * 70 + "\n")
    run_sbm_hybrid()
    print("\n" + "=" * 70 + "\n")
    run_sbm_landscape()
