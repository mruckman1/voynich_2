"""
Phase 69, Track 4: T1 Vocabulary Network
==========================================
Build a network of relationships between the 223 T1-identified words:
  - Folio co-occurrence (which T1 words appear on the same folio?)
  - Sequential proximity (which appear within 5 tokens of each other?)
  - Morphological families (shared Latin roots)
  - Semantic CI overlap (shared Circa Instans entries)

Dependency chain:
    results/p69_clean_corpus.json        (Step 0)
    results/combined_refine.json         (Phase 15)
    results/triple_tiers.json            (Phase 28/53)
    data/reference/latin/circa_instans.txt
        -> results/p69_t1_network.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
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
# CI loader (from p66_collocations pattern)
# ---------------------------------------------------------------------------

def _load_ci_entries(ci_path: str) -> List[Set[str]]:
    """Load Circa Instans text and split into entries (paragraphs)."""
    if not os.path.exists(ci_path):
        return []
    with open(ci_path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    entries = []
    for para in paragraphs:
        words = set()
        for token in para.lower().split():
            cleaned = ''.join(c for c in token if c.isalpha())
            if len(cleaned) >= 2:
                words.add(cleaned)
        if words:
            entries.append(words)
    return entries


def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein edit distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


# ---------------------------------------------------------------------------
# Morphological analysis
# ---------------------------------------------------------------------------

LATIN_ENDINGS_MORPH = [
    ('ione', 'ablative', '3rd_abstract'),
    ('onis', 'genitive', '3rd_abstract'),
    ('arum', 'gen_pl', '1st'),
    ('orum', 'gen_pl', '2nd'),
    ('ae', 'gen_dat', '1st'),
    ('am', 'accusative', '1st'),
    ('um', 'accusative', '2nd'),
    ('em', 'accusative', '3rd'),
    ('is', 'genitive', '3rd'),
    ('us', 'nominative', '2nd'),
    ('es', 'nom_pl', '3rd'),
    ('or', 'agent', '3rd'),
    ('er', 'agent', '2nd'),
    ('ar', 'adjective', '3rd'),
    ('a', 'nom_abl', '1st'),
    ('o', 'dat_abl', '2nd'),
    ('e', 'ablative', '3rd'),
    ('i', 'genitive', '2nd'),
]


def _extract_root(word: str) -> str:
    """Extract Latin root by stripping longest matching ending."""
    for ending, _, _ in LATIN_ENDINGS_MORPH:
        if word.endswith(ending) and len(word) > len(ending) + 2:
            return word[:-len(ending)]
    return word


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class T1NetworkResult:
    phase: str = "69"
    step: str = "69.5"
    experiment: str = "t1_vocabulary_network"
    n_t1_words: int = 0
    # Folio co-occurrence
    n_cooccurrence_pairs: int = 0
    top_cooccurrence: List[Dict[str, Any]] = field(default_factory=list)
    # Sequential proximity
    n_sequential_pairs: int = 0
    top_sequential: List[Dict[str, Any]] = field(default_factory=list)
    # Morphological families
    n_paradigms: int = 0
    paradigms: List[Dict[str, Any]] = field(default_factory=list)
    # Semantic CI overlap
    n_ci_pairs: int = 0
    ci_pairs: List[Dict[str, Any]] = field(default_factory=list)
    n_t1_in_ci: int = 0
    # Gates
    gate_vn1: bool = False    # >= 10 paradigms
    gate_vn2: bool = False    # >= 20 sequential pairs with count >= 3
    gate_vn3: bool = False    # >= 5 CI-matching pairs
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_t1_network():
    """Track 4: T1 vocabulary network analysis."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 69.5 — T1 Vocabulary Network")
    print("=" * 36)

    # --- Load T1 catalogue ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    if not clean_data:
        print("  ERROR: p69_clean_corpus.json not found. Run build-clean first.")
        return

    t1_catalogue = clean_data.get('t1_catalogue', [])
    if not t1_catalogue:
        print("  ERROR: No T1 identifications found.")
        return

    print(f"  T1 words: {len(t1_catalogue)}")

    # Build T1 type → matched word lookup
    t1_type_to_word: Dict[str, str] = {}
    for entry in t1_catalogue:
        t1_type_to_word[entry['eva_type']] = entry['matched_word']

    t1_types = set(t1_type_to_word.keys())
    t1_words = set(t1_type_to_word.values())

    # --- Load corpus ---
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Build folio list
    folio_list: List[str] = []
    for folio, page in corpus.pages.items():
        for _ in page.all_tokens:
            folio_list.append(folio)

    # --- 1. Folio co-occurrence ---
    print("\n  1. Folio co-occurrence...")
    folio_t1_types: Dict[str, Set[str]] = {}
    for idx, token in enumerate(all_tokens):
        if token in t1_types:
            folio = folio_list[idx] if idx < len(folio_list) else '?'
            if folio not in folio_t1_types:
                folio_t1_types[folio] = set()
            folio_t1_types[folio].add(token)

    cooc_counter: Counter = Counter()
    for folio, types_on_folio in folio_t1_types.items():
        type_list = sorted(types_on_folio)
        for i in range(len(type_list)):
            for j in range(i + 1, len(type_list)):
                pair = (type_list[i], type_list[j])
                cooc_counter[pair] += 1

    top_cooc = []
    for (a, b), count in cooc_counter.most_common(50):
        n_folios_a = len(folio_t1_types.get(a, set()) if a in folio_t1_types else set())
        n_folios_b = len(folio_t1_types.get(b, set()) if b in folio_t1_types else set())
        # Folios containing a
        folios_a = set(f for f, types in folio_t1_types.items() if a in types)
        folios_b = set(f for f, types in folio_t1_types.items() if b in types)
        jaccard = len(folios_a & folios_b) / len(folios_a | folios_b) if (folios_a | folios_b) else 0

        top_cooc.append({
            'eva_a': a, 'word_a': t1_type_to_word.get(a, '?'),
            'eva_b': b, 'word_b': t1_type_to_word.get(b, '?'),
            'shared_folios': count,
            'jaccard': round(jaccard, 3),
        })

    n_cooc_pairs = len(cooc_counter)
    print(f"    Co-occurrence pairs: {n_cooc_pairs}")

    # --- 2. Sequential proximity ---
    print("\n  2. Sequential proximity (window=5)...")
    seq_counter: Counter = Counter()
    WINDOW = 5

    for idx, token in enumerate(all_tokens):
        if token not in t1_types:
            continue
        for offset in range(1, WINDOW + 1):
            j = idx + offset
            if j < len(all_tokens) and all_tokens[j] in t1_types:
                pair = tuple(sorted([token, all_tokens[j]]))
                seq_counter[pair] += 1

    top_seq = []
    for (a, b), count in seq_counter.most_common(50):
        top_seq.append({
            'eva_a': a, 'word_a': t1_type_to_word.get(a, '?'),
            'eva_b': b, 'word_b': t1_type_to_word.get(b, '?'),
            'count': count,
        })

    n_seq_pairs = sum(1 for c in seq_counter.values() if c >= 3)
    print(f"    Sequential pairs (count >= 3): {n_seq_pairs}")

    # --- 3. Morphological families ---
    print("\n  3. Morphological families...")
    root_groups: Dict[str, List[Dict[str, str]]] = {}
    for entry in t1_catalogue:
        word = entry['matched_word']
        root = _extract_root(word)
        if len(root) >= 2:
            if root not in root_groups:
                root_groups[root] = []
            root_groups[root].append({
                'word': word,
                'eva_type': entry['eva_type'],
                'tier': entry['tier'],
            })

    paradigms = []
    for root, members in sorted(root_groups.items()):
        if len(members) >= 2:
            paradigms.append({
                'root': root,
                'n_members': len(members),
                'members': members,
            })

    paradigms.sort(key=lambda p: -p['n_members'])
    n_paradigms = len(paradigms)
    print(f"    Paradigms (shared root, 2+ members): {n_paradigms}")
    for p in paradigms[:10]:
        words = [m['word'] for m in p['members']]
        print(f"      {p['root']}: {', '.join(words)}")

    # --- 4. Semantic CI overlap ---
    print("\n  4. Semantic CI overlap...")
    ci_path = os.path.join(str(_data_dir()), 'reference', 'latin', 'circa_instans.txt')
    ci_entries = _load_ci_entries(ci_path)
    print(f"    CI entries: {len(ci_entries)}")

    # Map T1 words to CI entries
    word_to_ci_indices: Dict[str, List[int]] = {}
    n_t1_in_ci = 0
    for word in t1_words:
        matching = []
        for ci_idx, entry_words in enumerate(ci_entries):
            # Check exact or ED <= 1
            for ew in entry_words:
                if word == ew or (len(word) >= 3 and _edit_distance(word, ew) <= 1):
                    matching.append(ci_idx)
                    break
        if matching:
            word_to_ci_indices[word] = matching
            n_t1_in_ci += 1

    print(f"    T1 words in CI: {n_t1_in_ci}/{len(t1_words)}")

    # Find word pairs sharing CI entries
    ci_pairs = []
    words_in_ci = sorted(word_to_ci_indices.keys())
    for i in range(len(words_in_ci)):
        for j in range(i + 1, len(words_in_ci)):
            shared = set(word_to_ci_indices[words_in_ci[i]]) & set(word_to_ci_indices[words_in_ci[j]])
            if shared:
                ci_pairs.append({
                    'word_a': words_in_ci[i],
                    'word_b': words_in_ci[j],
                    'shared_ci_entries': len(shared),
                })

    ci_pairs.sort(key=lambda p: -p['shared_ci_entries'])
    n_ci_pairs = len(ci_pairs)
    print(f"    CI-matching pairs: {n_ci_pairs}")

    # --- Gates ---
    gate_vn1 = n_paradigms >= 10
    gate_vn2 = n_seq_pairs >= 20
    gate_vn3 = n_ci_pairs >= 5
    gates_passed = sum([gate_vn1, gate_vn2, gate_vn3])

    result = T1NetworkResult(
        n_t1_words=len(t1_catalogue),
        n_cooccurrence_pairs=n_cooc_pairs,
        top_cooccurrence=top_cooc[:50],
        n_sequential_pairs=n_seq_pairs,
        top_sequential=top_seq[:50],
        n_paradigms=n_paradigms,
        paradigms=paradigms[:30],
        n_ci_pairs=n_ci_pairs,
        ci_pairs=ci_pairs[:50],
        n_t1_in_ci=n_t1_in_ci,
        gate_vn1=gate_vn1,
        gate_vn2=gate_vn2,
        gate_vn3=gate_vn3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p69_t1_network.json', result)

    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Paradigms:     {n_paradigms} ({'PASS' if gate_vn1 else 'FAIL'} >= 10)")
    print(f"  Seq pairs:     {n_seq_pairs} ({'PASS' if gate_vn2 else 'FAIL'} >= 20)")
    print(f"  CI pairs:      {n_ci_pairs} ({'PASS' if gate_vn3 else 'FAIL'} >= 5)")
    print(f"  Gates: {gates_passed}/3")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
