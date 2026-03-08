"""
Step 24.13 – Section-Trained Decode Transfer Test
=================================================
Train separate decoding tables for each manuscript section and test
whether they transfer to other sections.

Dependency chain:
    combined_refine.json (Phase 15)
    modifier_integrate.json (Phase 16)
        → cross_section_transfer.json (this step)
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_cv_syllable_table,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.csp_solver import decode_token


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
# Section ranges
# ---------------------------------------------------------------------------

SECTION_RANGES = {
    'herbal_a': (1, 56),
    'pharmaceutical': (57, 66),
    'astronomical': (67, 73),
    'biological': (74, 84),
    'cosmological': (85, 86),
    'zodiac': (87, 101),
    'herbal_b': (102, 116),
}

MIN_SECTION_TOKENS = 200


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SectionTable:
    section: str
    n_tokens: int
    self_dict_hit: float
    n_triples_changed: int  # compared to Phase 16
    changed_triples: List[Dict]  # [{triple_key, phase16_syl, section_syl}]


@dataclass
class CrossSectionResult:
    timestamp: str
    n_sections: int
    sections_used: List[str]
    section_token_counts: Dict[str, int]
    # Per-section tables
    section_tables: List[Dict]
    # Transfer matrix
    transfer_matrix: Dict[str, Dict[str, float]]  # train -> test -> dict_hit
    # Analysis
    mean_self_hit: float
    mean_transfer_hit: float
    transfer_ratio: float  # mean_transfer / mean_self
    # Clustering
    table_distances: Dict[str, Dict[str, int]]  # section pairs -> distance
    clusters: List[List[str]]  # groups of similar sections
    # Verdict
    encoding_is_uniform: bool  # transfer_ratio > 0.8
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re


def _extract_folio_number(folio_id: str) -> Optional[int]:
    """Extract numeric folio number from folio ID like 'f1r', 'f102v'."""
    m = re.search(r'(\d+)', folio_id)
    if m:
        return int(m.group(1))
    return None


def _assign_section(folio_id: str) -> Optional[str]:
    """Assign a section name based on folio number ranges."""
    num = _extract_folio_number(folio_id)
    if num is None:
        return None
    for section, (lo, hi) in SECTION_RANGES.items():
        if lo <= num <= hi:
            return section
    return None


def _group_tokens_by_section(corpus) -> Dict[str, List[str]]:
    """Group all corpus tokens by manuscript section.

    Uses each page's folio ID to assign a section, then collects
    all paragraph tokens from that page.
    """
    section_tokens: Dict[str, List[str]] = defaultdict(list)

    for folio_id, page in corpus.pages.items():
        section = _assign_section(folio_id)
        if section is None:
            # Use the page's own section inference as fallback
            section = page.section
        if section and section != 'unknown':
            tokens = page.paragraph_text.split()
            tokens = [t for t in tokens if len(t) >= 2]
            section_tokens[section].extend(tokens)

    return dict(section_tokens)


def _decode_r3(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
) -> List[str]:
    """R3 combined decode: try alteration -> stripping -> original."""
    decoded = []
    for token in tokens:
        # Try alteration
        alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
            modifier_rules=modifier_rules,
        )
        if alt.lower() in ref_word_set:
            decoded.append(alt)
            continue

        # Try stripping
        stripped = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        if stripped.lower() in ref_word_set:
            decoded.append(stripped)
            continue

        # Fall back to original decoding
        original = decode_token(token, assignment, eva_to_triple)
        decoded.append(original)

    return decoded


def _dict_hit_rate(decoded: List[str], ref_word_set: set) -> float:
    """Fraction of decoded tokens that are dictionary hits."""
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    return hits / len(decoded)


def _train_section_table(
    section_tokens: List[str],
    base_assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    modifier_rules: Dict[str, str],
    ref_word_set: set,
    cv_inventory: List[str],
) -> Tuple[Dict[str, str], float]:
    """Greedy hill-climbing to optimize assignment for a specific section.

    Uses at most 500 tokens and does a single pass through all triple keys
    to keep runtime manageable.
    """
    current = dict(base_assignment)
    sample = section_tokens[:500]

    # Compute baseline hit rate
    decoded = _decode_r3(
        sample, current, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    current_hit = _dict_hit_rate(decoded, ref_word_set)

    # Single pass through all triple keys (no outer loop)
    triple_keys = list(current.keys())
    random.shuffle(triple_keys)  # randomize traversal order

    for triple_key in triple_keys:
        old_syl = current[triple_key]
        best_syl = old_syl
        best_hit = current_hit

        # Values currently in use by OTHER triples
        used_by_others = set(
            v for k, v in current.items() if k != triple_key
        )

        for candidate in cv_inventory:
            if candidate == old_syl:
                continue
            if candidate in used_by_others:
                continue  # all-different constraint

            current[triple_key] = candidate
            decoded = _decode_r3(
                sample, current, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            hit = _dict_hit_rate(decoded, ref_word_set)
            if hit > best_hit:
                best_hit = hit
                best_syl = candidate

            current[triple_key] = old_syl

        if best_syl != old_syl:
            current[triple_key] = best_syl
            current_hit = best_hit

    return current, current_hit


def _table_distance(table_a: Dict[str, str], table_b: Dict[str, str]) -> int:
    """Number of triple keys where two tables assign different syllables."""
    all_keys = set(table_a.keys()) | set(table_b.keys())
    return sum(
        1 for k in all_keys
        if table_a.get(k) != table_b.get(k)
    )


def _cluster_sections(
    sections: List[str],
    distances: Dict[str, Dict[str, int]],
    threshold: int = 5,
) -> List[List[str]]:
    """Group sections where pairwise table distance < threshold.

    Uses simple single-linkage clustering.
    """
    # Start with each section in its own cluster
    clusters: List[Set[str]] = [set([s]) for s in sections]

    # Merge clusters that have members within threshold distance
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                # Check if any pair across the two clusters is within threshold
                should_merge = False
                for a in clusters[i]:
                    for b in clusters[j]:
                        d = distances.get(a, {}).get(b, 999)
                        if d < threshold:
                            should_merge = True
                            break
                    if should_merge:
                        break
                if should_merge:
                    clusters[i] = clusters[i] | clusters[j]
                    clusters.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    return [sorted(list(c)) for c in clusters]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_cross_section() -> None:
    """Step 24.13: Section-trained decode transfer test."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 24.13: Section-Trained Decode Transfer Test")
    print("=" * 70)

    rd = _results_dir()

    # ─── Load Phase 16 results (modifier_integrate.json) ───
    print("\n  1. Loading Phase 16 results ...")
    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found — run phase16 first")
        return

    with open(mod_path) as f:
        mod_data = json.load(f)

    modifier_chars_list = mod_data.get('modifier_chars', [])
    modifier_chars = set(modifier_chars_list)
    print(f"      {len(modifier_chars)} modifier chars: {modifier_chars_list}")

    # Build modifier rules from classifications
    modifier_rules: Dict[str, str] = {}
    for cl in mod_data.get('classifications', []):
        if cl.get('final_classification') == 'modifier':
            modifier_rules[cl['eva_char']] = cl.get('modifier_type', 'silent')

    # ─── Load Phase 15 best assignment ───
    print("\n  2. Loading Phase 15 best assignment ...")
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found — run combined-refine first")
        return

    with open(refine_path) as f:
        refine_data = json.load(f)

    base_assignment = refine_data.get('best_assignment', {})
    print(f"      {len(base_assignment)} triple -> syllable mappings")

    # ─── Load corpus ───
    print("\n  3. Loading corpus ...")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    # ─── Build reference word set ───
    print("\n  4. Building expanded reference word set ...")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for w in ref_corpus.get_combined_tokens('latin')
            if len(w) >= 2
        )
    except (FileNotFoundError, KeyError):
        base_words = set()

    expanded_words, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded_words
    print(f"      {len(ref_word_set)} words in reference set")

    # ─── Build CV syllable inventory ───
    cv_inventory = build_cv_syllable_table('latin')
    print(f"      {len(cv_inventory)} CV syllables in inventory")

    # ─── Split corpus by section ───
    print("\n  5. Splitting corpus by section ...")
    section_tokens = _group_tokens_by_section(corpus)

    section_token_counts: Dict[str, int] = {}
    for section in sorted(section_tokens.keys()):
        n = len(section_tokens[section])
        section_token_counts[section] = n
        marker = "OK" if n >= MIN_SECTION_TOKENS else "SKIP (< 200)"
        print(f"      {section:>20}: {n:>6} tokens  [{marker}]")

    # Filter to sections with enough tokens
    usable_sections = [
        s for s, n in section_token_counts.items()
        if n >= MIN_SECTION_TOKENS
    ]
    print(f"\n      {len(usable_sections)} sections with >= {MIN_SECTION_TOKENS} tokens")

    if len(usable_sections) < 2:
        print("  [SKIP] Need at least 2 usable sections for transfer test")
        return

    # ─── Train section-specific tables ───
    print(f"\n  6. Training section-specific tables ({len(usable_sections)} sections) ...")
    section_tables_data: List[SectionTable] = []
    trained_tables: Dict[str, Dict[str, str]] = {}

    for idx, section in enumerate(usable_sections, 1):
        tokens = section_tokens[section]
        print(f"\n      [{idx}/{len(usable_sections)}] Training {section} "
              f"({len(tokens)} tokens, using {min(len(tokens), 500)}) ...")

        t_start = time.time()
        trained_assignment, self_hit = _train_section_table(
            tokens, base_assignment, eva_to_triple,
            modifier_chars, modifier_rules,
            ref_word_set, cv_inventory,
        )
        t_elapsed = time.time() - t_start

        # Compare to base assignment
        changed = []
        for k in base_assignment:
            if trained_assignment.get(k) != base_assignment.get(k):
                changed.append({
                    'triple_key': k,
                    'phase16_syl': base_assignment.get(k, '?'),
                    'section_syl': trained_assignment.get(k, '?'),
                })

        trained_tables[section] = trained_assignment
        section_tables_data.append(SectionTable(
            section=section,
            n_tokens=len(tokens),
            self_dict_hit=round(self_hit, 4),
            n_triples_changed=len(changed),
            changed_triples=changed,
        ))

        print(f"          self_dict_hit={self_hit:.4f}, "
              f"triples changed={len(changed)}/{len(base_assignment)}, "
              f"time={t_elapsed:.1f}s")

    # ─── Cross-apply: build transfer matrix ───
    print(f"\n  7. Cross-applying tables ({len(usable_sections)}x{len(usable_sections)} matrix) ...")
    transfer_matrix: Dict[str, Dict[str, float]] = {}

    for train_section in usable_sections:
        transfer_matrix[train_section] = {}
        train_table = trained_tables[train_section]

        for test_section in usable_sections:
            test_tokens = section_tokens[test_section][:500]
            decoded = _decode_r3(
                test_tokens, train_table, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            hit = _dict_hit_rate(decoded, ref_word_set)
            transfer_matrix[train_section][test_section] = round(hit, 4)

    # Print the matrix
    print(f"\n      Transfer matrix (train=row, test=col):")
    header = f"{'':>20}"
    for s in usable_sections:
        header += f" {s[:8]:>10}"
    print(f"      {header}")
    print(f"      {'':>20}{'-' * (10 * len(usable_sections) + len(usable_sections))}")

    for train_s in usable_sections:
        row = f"{train_s:>20}"
        for test_s in usable_sections:
            val = transfer_matrix[train_s][test_s]
            marker = "*" if train_s == test_s else " "
            row += f" {val:>9.4f}{marker}"
        print(f"      {row}")

    # ─── Transfer quality analysis ───
    print(f"\n  8. Transfer quality analysis ...")

    # Diagonal (self-application) values
    self_hits = [
        transfer_matrix[s][s] for s in usable_sections
    ]
    mean_self_hit = sum(self_hits) / len(self_hits) if self_hits else 0.0

    # Off-diagonal (transfer) values
    transfer_hits = []
    for train_s in usable_sections:
        for test_s in usable_sections:
            if train_s != test_s:
                transfer_hits.append(transfer_matrix[train_s][test_s])
    mean_transfer_hit = (
        sum(transfer_hits) / len(transfer_hits) if transfer_hits else 0.0
    )

    transfer_ratio = (
        mean_transfer_hit / mean_self_hit if mean_self_hit > 0 else 0.0
    )

    print(f"      Mean self-hit (diagonal):    {mean_self_hit:.4f}")
    print(f"      Mean transfer-hit (off-diag): {mean_transfer_hit:.4f}")
    print(f"      Transfer ratio:               {transfer_ratio:.4f}")

    # ─── Clustering by table distance ───
    print(f"\n  9. Computing table distances and clustering ...")

    table_distances: Dict[str, Dict[str, int]] = {}
    for s1 in usable_sections:
        table_distances[s1] = {}
        for s2 in usable_sections:
            dist = _table_distance(trained_tables[s1], trained_tables[s2])
            table_distances[s1][s2] = dist

    # Print distance matrix
    print(f"\n      Table distance matrix (# triples differing):")
    header = f"{'':>20}"
    for s in usable_sections:
        header += f" {s[:8]:>10}"
    print(f"      {header}")

    for s1 in usable_sections:
        row = f"{s1:>20}"
        for s2 in usable_sections:
            row += f" {table_distances[s1][s2]:>10}"
        print(f"      {row}")

    clusters = _cluster_sections(usable_sections, table_distances, threshold=5)
    print(f"\n      Clusters (distance threshold < 5): {clusters}")

    # ─── Verdict ───
    encoding_is_uniform = transfer_ratio > 0.8

    if encoding_is_uniform:
        verdict = (
            f"UNIFORM ENCODING: Transfer ratio {transfer_ratio:.3f} > 0.8. "
            f"Section-trained tables generalize well across sections, "
            f"indicating the manuscript uses a single consistent encoding system. "
            f"Mean self-hit={mean_self_hit:.4f}, mean transfer={mean_transfer_hit:.4f}."
        )
    else:
        # Check for section-specific patterns
        best_self = max(self_hits) if self_hits else 0.0
        worst_transfer = min(transfer_hits) if transfer_hits else 0.0
        verdict = (
            f"NON-UNIFORM ENCODING: Transfer ratio {transfer_ratio:.3f} <= 0.8. "
            f"Section-specific tables do NOT transfer well, suggesting "
            f"encoding differences between sections. "
            f"Best self-hit={best_self:.4f}, worst transfer={worst_transfer:.4f}. "
            f"{len(clusters)} cluster(s) found."
        )

    print(f"\n  10. Verdict: {verdict}")

    runtime = round(time.time() - t0, 2)
    print(f"\n  Runtime: {runtime}s")

    # ─── Save ───
    result = CrossSectionResult(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        n_sections=len(usable_sections),
        sections_used=usable_sections,
        section_token_counts=section_token_counts,
        section_tables=[_convert(asdict(st)) for st in section_tables_data],
        transfer_matrix=transfer_matrix,
        mean_self_hit=round(mean_self_hit, 4),
        mean_transfer_hit=round(mean_transfer_hit, 4),
        transfer_ratio=round(transfer_ratio, 4),
        table_distances=table_distances,
        clusters=clusters,
        encoding_is_uniform=encoding_is_uniform,
        verdict=verdict,
        runtime_seconds=runtime,
    )

    out_path = os.path.join(rd, 'cross_section_transfer.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
