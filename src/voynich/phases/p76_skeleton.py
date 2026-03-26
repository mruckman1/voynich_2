"""
Phase 76, Track 2: Grammatical Skeleton Parsing + Parallel Passages
=====================================================================
Use the validated 3-coda grammar (bootstrap p=0.0000) to build
grammatical skeletons for all 15-token windows, find parallel passages
(same skeleton, different tokens), and test recipe template selectivity.

Dependency chain:
    results/combined_refine.json         (Phase 15 — assignment table)
    results/modifier_integrate.json      (Phase 16 — modifiers)
    results/p69_clean_corpus.json        (Phase 69 — T1 catalogue)
        -> results/p76_skeleton.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import classify_token_chars_v2, decode_token_cvc_v2
from voynich.phases.p75_redecode import _build_3coda_table


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
# Coda-to-grammar label mapping
# ---------------------------------------------------------------------------

CODA_TO_LABEL = {
    'n':  'ACC',     # accusative case (direct objects)
    's':  'V2',      # 2nd person verbs (imperative instructions)
    't':  'V3',      # 3rd person verbs
    'nt': 'V3PL',    # 3rd person plural
    'ns': 'PART',    # participle
}


# ---------------------------------------------------------------------------
# Recipe templates
# ---------------------------------------------------------------------------

RECIPE_TEMPLATES = [
    {
        'name': 'imperative_recipe',
        'pattern': ['V2', 'ACC'],
        'english': '[verb-imperative] [ingredient, accusative]',
    },
    {
        'name': 'compound_recipe',
        'pattern': ['V2', 'ACC', 'STEM', 'ACC'],
        'english': '[verb] [ingredient] [prep/modifier] [ingredient]',
    },
    {
        'name': 'property_statement',
        'pattern': ['STEM', 'V3'],
        'english': '[subject] [is/has/does]',
    },
    {
        'name': 'passive_instruction',
        'pattern': ['V3', 'ACC'],
        'english': '[is processed] [ingredient]',
    },
    {
        'name': 'ingredient_list',
        'pattern': ['STEM', 'STEM', 'STEM'],
        'english': '[item] [item] [item]',
    },
    {
        'name': 'negated_instruction',
        'pattern': ['STEM', 'V2', 'ACC'],
        'english': '[ne/non] [verb] [ingredient]',
    },
]


# ---------------------------------------------------------------------------
# Helpers: folio / section lists
# ---------------------------------------------------------------------------

def _build_folio_list(corpus) -> List[str]:
    """Build flat list of folio IDs, one per token."""
    folios: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folios.append(folio)
    return folios


def _build_section_list(corpus) -> List[str]:
    """Build flat list of section labels, one per token."""
    sections: List[str] = []
    for _folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            sections.append(getattr(page, 'section', 'unknown'))
    return sections


# ---------------------------------------------------------------------------
# Step 2a: Build grammatical label for a single token
# ---------------------------------------------------------------------------

def _token_to_grammar_label(
    token: str,
    coda_table,
) -> str:
    """Classify a token into a grammatical label based on its coda(s).

    Returns one of: STEM, ACC, V2, V3, V3PL, PART.
    """
    eva_chars = tokenize_eva_chars(token)
    classified = classify_token_chars_v2(eva_chars, coda_table)

    # Extract codas from CODA_MARKER characters
    coda_consonants: List[str] = []
    for role, char in classified:
        if role == 'CODA_MARKER':
            coda_val = get_coda(char, coda_table)
            if coda_val:
                coda_consonants.append(coda_val)

    if not coda_consonants:
        return 'STEM'

    # Join codas to check for compound endings (nt, ns)
    coda_str = ''.join(coda_consonants)

    # Check compound codas first
    if 'nt' in coda_str:
        return 'V3PL'
    if 'ns' in coda_str:
        return 'PART'

    # Use the last coda consonant for simple cases
    last_coda = coda_consonants[-1]
    return CODA_TO_LABEL.get(last_coda, 'STEM')


# ---------------------------------------------------------------------------
# Step 2a: Build skeletons for all 15-token windows
# ---------------------------------------------------------------------------

def _build_skeletons(
    all_tokens: List[str],
    folio_list: List[str],
    coda_table,
    window: int = 15,
) -> List[Dict[str, Any]]:
    """Build grammatical skeletons for all 15-token windows within folios."""
    n = len(all_tokens)
    skeletons: List[Dict[str, Any]] = []

    # Precompute grammar labels for all tokens
    print("    Pre-computing grammar labels for all tokens...")
    token_labels = [_token_to_grammar_label(tok, coda_table) for tok in all_tokens]

    print(f"    Scanning {n - window + 1} possible windows...")
    for start in range(n - window + 1):
        end = start + window - 1
        # Must be within same folio
        if folio_list[start] != folio_list[end]:
            continue

        labels = token_labels[start:start + window]
        skeleton_str = ' '.join(labels)

        skeletons.append({
            'start': start,
            'end': end,
            'folio': folio_list[start],
            'labels': labels,
            'skeleton_str': skeleton_str,
        })

    return skeletons


# ---------------------------------------------------------------------------
# Step 2b: Find parallel passages
# ---------------------------------------------------------------------------

def _find_parallel_passages(
    skeletons: List[Dict[str, Any]],
    all_tokens: List[str],
    t1_types: Set[str],
    window: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Group skeletons by skeleton_str, find parallel pairs with diffs.

    Returns (parallel_pairs, substitution_map).
    """
    # Group by skeleton string
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, skel in enumerate(skeletons):
        groups[skel['skeleton_str']].append(idx)

    parallel_pairs: List[Dict[str, Any]] = []

    # Track substitutions: unknown_token -> [{substitutes_for, role}]
    substitution_entries: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for skeleton_str, indices in groups.items():
        if len(indices) < 2:
            continue

        # Check all pairs within the group
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                skel_a = skeletons[indices[i]]
                skel_b = skeletons[indices[j]]

                # Non-overlapping: |start_a - start_b| >= window
                if abs(skel_a['start'] - skel_b['start']) < window:
                    continue

                # Find positions where tokens differ
                tokens_a = all_tokens[skel_a['start']:skel_a['start'] + window]
                tokens_b = all_tokens[skel_b['start']:skel_b['start'] + window]

                diff_positions: List[int] = []
                for pos in range(window):
                    if tokens_a[pos] != tokens_b[pos]:
                        diff_positions.append(pos)

                if not diff_positions:
                    continue

                # Diagnostic diffs: one token is T1, other is not
                diagnostic_diffs: List[Dict[str, Any]] = []
                for pos in diff_positions:
                    a_is_t1 = tokens_a[pos] in t1_types
                    b_is_t1 = tokens_b[pos] in t1_types

                    if a_is_t1 != b_is_t1:
                        t1_tok = tokens_a[pos] if a_is_t1 else tokens_b[pos]
                        unk_tok = tokens_b[pos] if a_is_t1 else tokens_a[pos]
                        gram_role = skel_a['labels'][pos]

                        diagnostic_diffs.append({
                            't1_token': t1_tok,
                            'unknown_token': unk_tok,
                            'position': pos,
                            'grammatical_role': gram_role,
                        })

                        # Record substitution
                        substitution_entries[unk_tok].append({
                            'substitutes_for': t1_tok,
                            'grammatical_role': gram_role,
                        })

                pair = {
                    'skeleton_str': skeleton_str,
                    'start_a': skel_a['start'],
                    'start_b': skel_b['start'],
                    'folio_a': skel_a['folio'],
                    'folio_b': skel_b['folio'],
                    'n_diffs': len(diff_positions),
                    'n_diagnostic': len(diagnostic_diffs),
                    'diagnostic_diffs': diagnostic_diffs,
                }
                parallel_pairs.append(pair)

    # Sort by number of diagnostic diffs (desc)
    parallel_pairs.sort(key=lambda p: -p['n_diagnostic'])

    return parallel_pairs, dict(substitution_entries)


# ---------------------------------------------------------------------------
# Step 2c: Recipe template matching
# ---------------------------------------------------------------------------

def _match_template_subsequence(labels: List[str], pattern: List[str]) -> bool:
    """Check if pattern is a subsequence of labels."""
    if not pattern:
        return True
    pat_idx = 0
    for label in labels:
        if label == pattern[pat_idx]:
            pat_idx += 1
            if pat_idx == len(pattern):
                return True
    return False


def _compute_template_rates(
    skeletons: List[Dict[str, Any]],
    section_list: List[str],
) -> Dict[str, float]:
    """Compute template match rate by section."""
    section_counts: Dict[str, int] = Counter()
    section_matches: Dict[str, int] = Counter()

    for skel in skeletons:
        section = section_list[skel['start']]
        section_counts[section] += 1

        for tmpl in RECIPE_TEMPLATES:
            if _match_template_subsequence(skel['labels'], tmpl['pattern']):
                section_matches[section] += 1
                break  # count each skeleton at most once

    rates: Dict[str, float] = {}
    for section, count in sorted(section_counts.items()):
        if count > 0:
            rates[section] = section_matches.get(section, 0) / count
        else:
            rates[section] = 0.0

    return rates


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SkeletonResult:
    phase: str = "76"
    step: str = "76.2"
    experiment: str = "skeleton_parse"
    # Step 2a: skeleton counts
    n_skeletons: int = 0
    n_unique_skeletons: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    # Step 2b: parallel passages
    n_parallel_pairs: int = 0
    n_diagnostic_diffs: int = 0
    top_parallel_pairs: List[Dict[str, Any]] = field(default_factory=list)
    substitution_map: Dict[str, Any] = field(default_factory=dict)
    n_substitution_tokens: int = 0
    # Step 2c: template rates
    section_template_rates: Dict[str, float] = field(default_factory=dict)
    template_selectivity: float = 0.0
    recipe_sections_higher: bool = False
    template_match_counts: Dict[str, int] = field(default_factory=dict)
    # Gates
    gate_s1: bool = False  # >= 100 parallel pairs
    gate_s2: bool = False  # >= 20 diagnostic diffs
    gate_s3: bool = False  # pharma > astro template rate
    gate_s4: bool = False  # >= 10 substitution assignments
    gates_passed: int = 0
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_skeleton_parse() -> SkeletonResult:
    """Track 2: Grammatical skeleton parsing + parallel passages."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 76.2 -- Grammatical Skeleton Parsing + Parallel Passages")
    print("=" * 62)

    # --- Load shared data ---
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])
    t1_types: Set[str] = set()
    for entry in t1_catalogue:
        eva_type = entry.get('eva_type', '')
        if eva_type:
            t1_types.add(eva_type)

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    folio_list = _build_folio_list(corpus)
    section_list = _build_section_list(corpus)

    coda_table = _build_3coda_table()

    print(f"  Corpus: {len(all_tokens)} tokens")
    print(f"  T1 types: {len(t1_types)}")

    # --- Step 2a: Build skeletons ---
    print("\n  Step 2a: Building grammatical skeletons (window=15)...")
    skeletons = _build_skeletons(all_tokens, folio_list, coda_table, window=15)
    unique_strs = set(s['skeleton_str'] for s in skeletons)

    # Count label distribution across all tokens
    all_labels: List[str] = []
    for tok in all_tokens:
        all_labels.append(_token_to_grammar_label(tok, coda_table))
    label_dist = dict(Counter(all_labels).most_common())

    print(f"    Total skeletons: {len(skeletons)}")
    print(f"    Unique skeletons: {len(unique_strs)}")
    print(f"    Label distribution:")
    for label, count in label_dist.items():
        print(f"      {label}: {count} ({100*count/len(all_tokens):.1f}%)")

    # --- Step 2b: Find parallel passages ---
    print("\n  Step 2b: Finding parallel passages...")
    parallel_pairs, substitution_map = _find_parallel_passages(
        skeletons, all_tokens, t1_types, window=15)

    total_diagnostic = sum(p['n_diagnostic'] for p in parallel_pairs)

    print(f"    Parallel pairs: {len(parallel_pairs)}")
    print(f"    Total diagnostic diffs: {total_diagnostic}")
    print(f"    Tokens with substitutions: {len(substitution_map)}")

    if parallel_pairs:
        print(f"    Top 5 parallel pairs:")
        for pp in parallel_pairs[:5]:
            print(f"      {pp['folio_a']}[{pp['start_a']}] vs "
                  f"{pp['folio_b']}[{pp['start_b']}]: "
                  f"{pp['n_diagnostic']} diagnostic diffs")

    # Truncate substitution_map to top 30 by number of entries
    sub_map_sorted = sorted(
        substitution_map.items(),
        key=lambda kv: -len(kv[1]),
    )
    sub_map_top30 = {
        tok: entries[:5]  # keep up to 5 substitution records per token
        for tok, entries in sub_map_sorted[:30]
    }

    # --- Step 2c: Recipe template matching ---
    print("\n  Step 2c: Recipe template matching by section...")
    section_rates = _compute_template_rates(skeletons, section_list)

    # Count template matches globally
    tmpl_counts: Dict[str, int] = Counter()
    for skel in skeletons:
        for tmpl in RECIPE_TEMPLATES:
            if _match_template_subsequence(skel['labels'], tmpl['pattern']):
                tmpl_counts[tmpl['name']] += 1

    for section, rate in sorted(section_rates.items()):
        print(f"    {section}: {100*rate:.1f}%")

    # Selectivity: pharmaceutical or herbal vs astronomical
    pharma_rate = section_rates.get('pharmaceutical', 0.0)
    herbal_a_rate = section_rates.get('herbal_a', 0.0)
    astro_rate = section_rates.get('astronomical', 0.0)

    # Use whichever recipe section is higher
    recipe_rate = max(pharma_rate, herbal_a_rate)
    template_selectivity = recipe_rate / astro_rate if astro_rate > 0 else (
        float('inf') if recipe_rate > 0 else 0.0)

    recipe_higher = recipe_rate > astro_rate

    print(f"\n    Recipe section rate: {100*recipe_rate:.1f}%")
    print(f"    Astro section rate: {100*astro_rate:.1f}%")
    print(f"    Template selectivity: {template_selectivity:.2f}x")
    print(f"    Recipe sections higher: {recipe_higher}")

    # --- Gates ---
    gate_s1 = len(parallel_pairs) >= 100
    gate_s2 = total_diagnostic >= 20
    gate_s3 = recipe_higher
    gate_s4 = len(substitution_map) >= 10
    gates_passed = sum([gate_s1, gate_s2, gate_s3, gate_s4])

    print(f"\n  Gates:")
    print(f"    S1 (>=100 parallel pairs): {'PASS' if gate_s1 else 'FAIL'} "
          f"({len(parallel_pairs)})")
    print(f"    S2 (>=20 diagnostic diffs): {'PASS' if gate_s2 else 'FAIL'} "
          f"({total_diagnostic})")
    print(f"    S3 (pharma/herbal > astro): {'PASS' if gate_s3 else 'FAIL'}")
    print(f"    S4 (>=10 substitution tokens): {'PASS' if gate_s4 else 'FAIL'} "
          f"({len(substitution_map)})")
    print(f"    Total: {gates_passed}/4")

    # --- Verdict ---
    if gates_passed >= 3:
        verdict = 'SKELETON_SELECTIVE'
    elif gates_passed >= 2:
        verdict = 'SKELETON_PARTIAL'
    else:
        verdict = 'SKELETON_INSUFFICIENT'

    result = SkeletonResult(
        n_skeletons=len(skeletons),
        n_unique_skeletons=len(unique_strs),
        label_distribution=label_dist,
        n_parallel_pairs=len(parallel_pairs),
        n_diagnostic_diffs=total_diagnostic,
        top_parallel_pairs=parallel_pairs[:20],
        substitution_map=sub_map_top30,
        n_substitution_tokens=len(substitution_map),
        section_template_rates=section_rates,
        template_selectivity=round(template_selectivity, 4),
        recipe_sections_higher=recipe_higher,
        template_match_counts=dict(tmpl_counts),
        gate_s1=gate_s1,
        gate_s2=gate_s2,
        gate_s3=gate_s3,
        gate_s4=gate_s4,
        gates_passed=gates_passed,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    path = _save_json(rd, 'p76_skeleton.json', asdict(result))
    print(f"\n  Verdict: {verdict} ({gates_passed}/4)")
    print(f"  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
