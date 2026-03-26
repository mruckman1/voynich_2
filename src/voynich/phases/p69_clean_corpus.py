"""
Phase 69, Step 0: Build Clean Corpus Partition + T1 Catalogue
==============================================================
Partition the corpus into CLEAN tokens (every EVA character maps to a
confirmed triple or validated coda marker, 0% decode error) and PARTIAL
tokens (1+ unresolved characters).  Also build the expanded T1 word
catalogue from Phase 68 Track 4.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
    results/p68_expanded_t1.json      (Phase 68 Track 4)
        -> results/p69_clean_corpus.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_token_cvc_v2,
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
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13)."""
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CleanCorpusResult:
    phase: str = "69"
    step: str = "69.0"
    experiment: str = "build_clean_corpus"
    # Corpus partition
    n_corpus_tokens: int = 0
    n_clean: int = 0
    n_partial: int = 0
    clean_fraction: float = 0.0
    # Clean subset metrics
    clean_dict_hit: float = 0.0
    n_clean_dict_hits: int = 0
    n_clean_types: int = 0
    # Consecutive clean runs
    n_runs_ge3: int = 0
    n_runs_ge5: int = 0
    n_runs_ge10: int = 0
    longest_run: int = 0
    top_runs: List[Dict[str, Any]] = field(default_factory=list)
    # Folio density
    folio_density: Dict[str, float] = field(default_factory=dict)
    best_folios: List[str] = field(default_factory=list)
    # Section density
    section_density: Dict[str, float] = field(default_factory=dict)
    # T1 catalogue
    n_t1_identifications: int = 0
    n_t1_tier1: int = 0
    n_t1_tier2: int = 0
    n_t1_tier3: int = 0
    t1_catalogue: List[Dict[str, Any]] = field(default_factory=list)
    t1_coverage_tokens: int = 0
    t1_coverage_fraction: float = 0.0
    # Clean token indices (per folio, for downstream use)
    clean_indices: List[int] = field(default_factory=list)
    # Decoded clean tokens (for downstream use)
    clean_decoded: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Token confidence classification (from p68_full_tokens pattern)
# ---------------------------------------------------------------------------

def _classify_token_confidence(
    token: str,
    eva_to_triple: Dict[str, str],
    confirmed_keys: Set[str],
    coda_table,
) -> Tuple[int, int, int]:
    """Return (n_confirmed, n_coda, n_unresolved) for a token."""
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return 0, 0, 0

    classified = classify_token_chars_v2(eva_chars, coda_table)
    n_confirmed = 0
    n_coda = 0
    n_unresolved = 0

    for role, char in classified:
        if role == 'SYLLABIC':
            triple_key = eva_to_triple.get(char, '')
            if triple_key in confirmed_keys:
                n_confirmed += 1
            else:
                n_unresolved += 1
        elif role == 'CODA_MARKER':
            n_coda += 1

    return n_confirmed, n_coda, n_unresolved


def _find_consecutive_runs(indices: List[int]) -> List[List[int]]:
    """Find runs of consecutive integers."""
    if not indices:
        return []
    runs = []
    current_run = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_run.append(indices[i])
        else:
            runs.append(current_run)
            current_run = [indices[i]]
    runs.append(current_run)
    return runs


def _build_folio_list(corpus) -> List[str]:
    """Build a flat list of folio IDs, one per token."""
    folios: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folios.append(folio)
    return folios


def _build_section_list(corpus) -> List[str]:
    """Build a flat list of section labels, one per token."""
    sections: List[str] = []
    for _folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            sections.append(getattr(page, 'section', 'unknown'))
    return sections


# ---------------------------------------------------------------------------
# T1 catalogue builder
# ---------------------------------------------------------------------------

ORIGINAL_T1_TYPES: Set[str] = set()  # Populated from Phase 52 if available


def _build_t1_catalogue(
    rd: str,
    all_tokens: List[str],
    folio_list: List[str],
) -> Tuple[List[Dict[str, Any]], int]:
    """Build T1 catalogue from Phase 68 expanded T1 results.

    Returns (catalogue_list, total_t1_token_occurrences).
    """
    t1_data = _safe_load(os.path.join(rd, 'p68_expanded_t1.json'))
    identifications = t1_data.get('identifications', [])

    if not identifications:
        return [], 0

    # Load original T1 from Phase 52 if available
    catalog_data = _safe_load(os.path.join(rd, 'word_catalog.json'))
    original_types: Set[str] = set()
    if catalog_data:
        for entry in catalog_data.get('catalog', []):
            if entry.get('unique_match'):
                original_types.add(entry.get('eva_type', ''))

    # Build token → folio mapping for distribution
    type_to_folios: Dict[str, Set[str]] = {}
    for idx, token in enumerate(all_tokens):
        if token not in type_to_folios:
            type_to_folios[token] = set()
        type_to_folios[token].add(folio_list[idx])

    type_to_count: Counter = Counter(all_tokens)

    catalogue = []
    total_occurrences = 0

    for ident in identifications:
        eva_type = ident.get('token', '')
        matched_word = ident.get('matched_word', '')
        n_folios = ident.get('n_folios', 0)

        if not eva_type or not matched_word:
            continue

        count = type_to_count.get(eva_type, 0)
        folios = sorted(type_to_folios.get(eva_type, set()))

        # Tier assignment
        if eva_type in original_types:
            tier = 'TIER_1'
        elif n_folios >= 5:
            tier = 'TIER_2'
        else:
            tier = 'TIER_3'

        catalogue.append({
            'eva_type': eva_type,
            'matched_word': matched_word,
            'n_folios': n_folios,
            'frequency': count,
            'tier': tier,
            'folios': folios,
        })
        total_occurrences += count

    # Sort by tier priority then frequency
    tier_order = {'TIER_1': 0, 'TIER_2': 1, 'TIER_3': 2}
    catalogue.sort(key=lambda c: (tier_order.get(c['tier'], 9), -c['frequency']))

    return catalogue, total_occurrences


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_build_clean():
    """Step 0: Build clean corpus partition and T1 catalogue."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 69.0 — Build Clean Corpus Partition")
    print("=" * 44)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    confirmed_keys = set(confirmed.keys())
    full_assignment = {**confirmed, **unresolved}
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folio_list = _build_folio_list(corpus)
    section_list = _build_section_list(corpus)
    print(f"  Corpus tokens: {len(all_tokens)}")

    # Build dictionary
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"  Dictionary size: {len(ref_word_set)}")

    # --- Step 1: Classify all tokens ---
    print("\n  Classifying token confidence...")
    clean_indices: List[int] = []
    partial_indices: List[int] = []

    for idx, token in enumerate(all_tokens):
        n_conf, n_coda, n_unres = _classify_token_confidence(
            token, eva_to_triple, confirmed_keys, coda_table)
        if n_unres == 0 and (n_conf + n_coda) > 0:
            clean_indices.append(idx)
        elif n_unres > 0:
            partial_indices.append(idx)

    n_clean = len(clean_indices)
    n_partial = len(partial_indices)
    clean_frac = n_clean / len(all_tokens) if all_tokens else 0.0
    print(f"  Clean tokens:   {n_clean} ({clean_frac:.1%})")
    print(f"  Partial tokens: {n_partial}")

    # --- Step 2: Decode clean tokens and compute dict hit ---
    print("\n  Decoding clean subset...")
    clean_decoded: List[str] = []
    n_dict_hits = 0
    clean_vocab: Counter = Counter()

    for idx in clean_indices:
        result = decode_token_cvc_v2(
            all_tokens[idx], full_assignment, eva_to_triple, coda_table)
        d = result.decoded_cvc if result.decoded_cvc else ''
        clean_decoded.append(d)
        if d and '?' not in d and d in ref_word_set:
            n_dict_hits += 1
        if d:
            clean_vocab[d] += 1

    clean_dict_hit = n_dict_hits / n_clean if n_clean > 0 else 0.0
    print(f"  Clean dict hits: {n_dict_hits}/{n_clean} ({clean_dict_hit:.1%})")
    print(f"  Clean distinct words: {len(clean_vocab)}")

    # --- Step 3: Consecutive clean runs ---
    runs = _find_consecutive_runs(clean_indices)
    runs_ge3 = [r for r in runs if len(r) >= 3]
    runs_ge5 = [r for r in runs if len(r) >= 5]
    runs_ge10 = [r for r in runs if len(r) >= 10]
    longest_run = max((len(r) for r in runs), default=0)
    print(f"  Runs >= 3: {len(runs_ge3)}")
    print(f"  Runs >= 5: {len(runs_ge5)}")
    print(f"  Runs >= 10: {len(runs_ge10)}")
    print(f"  Longest run: {longest_run}")

    # Top runs with decoded text
    top_runs = []
    for run in sorted(runs_ge5, key=lambda r: -len(r))[:30]:
        run_decoded = []
        for idx in run:
            ci = clean_indices.index(idx) if idx in clean_indices else -1
            if ci >= 0 and ci < len(clean_decoded):
                run_decoded.append(clean_decoded[ci])
            else:
                result = decode_token_cvc_v2(
                    all_tokens[idx], full_assignment, eva_to_triple, coda_table)
                run_decoded.append(result.decoded_cvc if result.decoded_cvc else '?')
        n_hits = sum(1 for w in run_decoded if w in ref_word_set)
        top_runs.append({
            'start_idx': run[0],
            'length': len(run),
            'folio': folio_list[run[0]] if run[0] < len(folio_list) else '?',
            'decoded': ' '.join(run_decoded),
            'dict_hits': n_hits,
            'dict_hit_rate': round(n_hits / len(run_decoded), 3) if run_decoded else 0,
        })

    # --- Step 4: Folio density ---
    folio_clean_counts: Counter = Counter()
    folio_total_counts: Counter = Counter()
    for idx in range(len(all_tokens)):
        folio = folio_list[idx] if idx < len(folio_list) else '?'
        folio_total_counts[folio] += 1
    for idx in clean_indices:
        folio = folio_list[idx] if idx < len(folio_list) else '?'
        folio_clean_counts[folio] += 1

    folio_density = {}
    for folio in folio_total_counts:
        if folio_total_counts[folio] > 0:
            folio_density[folio] = round(
                folio_clean_counts.get(folio, 0) / folio_total_counts[folio], 3)

    folio_density_sorted = dict(sorted(folio_density.items(), key=lambda x: -x[1]))
    best_folios = [f for f, d in folio_density_sorted.items() if d >= 0.70][:20]

    # --- Step 5: Section density ---
    section_clean_counts: Counter = Counter()
    section_total_counts: Counter = Counter()
    for idx in range(len(all_tokens)):
        sec = section_list[idx] if idx < len(section_list) else 'unknown'
        section_total_counts[sec] += 1
    for idx in clean_indices:
        sec = section_list[idx] if idx < len(section_list) else 'unknown'
        section_clean_counts[sec] += 1

    section_density = {}
    for sec in section_total_counts:
        if section_total_counts[sec] > 0:
            section_density[sec] = round(
                section_clean_counts.get(sec, 0) / section_total_counts[sec], 3)

    # --- Step 6: Build T1 catalogue ---
    print("\n  Building T1 catalogue...")
    t1_catalogue, t1_coverage_tokens = _build_t1_catalogue(
        rd, all_tokens, folio_list)

    tier_counts: Counter = Counter(c['tier'] for c in t1_catalogue)
    t1_coverage_frac = t1_coverage_tokens / len(all_tokens) if all_tokens else 0.0
    print(f"  T1 identifications: {len(t1_catalogue)}")
    print(f"    Tier 1 (original): {tier_counts.get('TIER_1', 0)}")
    print(f"    Tier 2 (≥5 folios): {tier_counts.get('TIER_2', 0)}")
    print(f"    Tier 3 (3-4 folios): {tier_counts.get('TIER_3', 0)}")
    print(f"  T1 coverage: {t1_coverage_tokens} tokens ({t1_coverage_frac:.1%})")

    # --- Build result ---
    result = CleanCorpusResult(
        n_corpus_tokens=len(all_tokens),
        n_clean=n_clean,
        n_partial=n_partial,
        clean_fraction=round(clean_frac, 4),
        clean_dict_hit=round(clean_dict_hit, 4),
        n_clean_dict_hits=n_dict_hits,
        n_clean_types=len(clean_vocab),
        n_runs_ge3=len(runs_ge3),
        n_runs_ge5=len(runs_ge5),
        n_runs_ge10=len(runs_ge10),
        longest_run=longest_run,
        top_runs=top_runs,
        folio_density=folio_density_sorted,
        best_folios=best_folios,
        section_density=section_density,
        n_t1_identifications=len(t1_catalogue),
        n_t1_tier1=tier_counts.get('TIER_1', 0),
        n_t1_tier2=tier_counts.get('TIER_2', 0),
        n_t1_tier3=tier_counts.get('TIER_3', 0),
        t1_catalogue=t1_catalogue,
        t1_coverage_tokens=t1_coverage_tokens,
        t1_coverage_fraction=round(t1_coverage_frac, 4),
        clean_indices=clean_indices,
        clean_decoded=clean_decoded,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p69_clean_corpus.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Clean tokens:    {n_clean} ({clean_frac:.1%})")
    print(f"  Clean dict hit:  {clean_dict_hit:.1%}")
    print(f"  Runs >= 5:       {len(runs_ge5)}")
    print(f"  Longest run:     {longest_run}")
    print(f"  Best folios:     {len(best_folios)} with ≥70% clean")
    print(f"  T1 words:        {len(t1_catalogue)}")
    print(f"  Sections:")
    for sec, density in sorted(section_density.items(), key=lambda x: -x[1]):
        print(f"    {sec}: {density:.1%}")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
