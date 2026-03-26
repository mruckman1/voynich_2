"""
Phase 69, Track 6: T1 × Circa Instans Cross-Reference
========================================================
Map each T1 word to CI entries that contain it, then build
folio → topic assignments based on T1 word clustering.
Validate with a permutation test.

Dependency chain:
    results/p69_clean_corpus.json        (Step 0)
    data/reference/latin/circa_instans.txt
        -> results/p69_t1_ci.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.corpus import load_corpus


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
# CI loader
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


def _ci_entry_title(ci_entries: List[Set[str]], idx: int) -> str:
    """Get the longest word in a CI entry as a title proxy."""
    if idx >= len(ci_entries):
        return '?'
    words = ci_entries[idx]
    if not words:
        return '?'
    return max(words, key=len)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class T1CIResult:
    phase: str = "69"
    step: str = "69.7"
    experiment: str = "t1_ci_crossreference"
    n_t1_words: int = 0
    n_ci_entries: int = 0
    n_t1_in_ci: int = 0
    t1_in_ci_fraction: float = 0.0
    # Per-word CI matches
    word_ci_matches: List[Dict[str, Any]] = field(default_factory=list)
    # Folio-topic assignments
    folio_topics: List[Dict[str, Any]] = field(default_factory=list)
    n_folio_assignments: int = 0
    # Permutation test
    real_overlap_score: float = 0.0
    null_mean: float = 0.0
    null_std: float = 0.0
    perm_z: float = 0.0
    perm_p: float = 1.0
    n_trials: int = 0
    # Section coverage
    section_topics: Dict[str, List[str]] = field(default_factory=dict)
    # Gates
    gate_ci1: bool = False    # >= 50 T1 words found in CI
    gate_ci2: bool = False    # >= 10 folio-topic assignments
    gate_ci3: bool = False    # perm p < 0.05
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_t1_ci_crossref():
    """Track 6: T1 × CI cross-reference."""
    t0 = time.time()
    rd = str(_results_dir())
    N_TRIALS = 1000

    print("Phase 69.7 — T1 × Circa Instans Cross-Reference")
    print("=" * 50)

    # --- Load T1 catalogue ---
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    if not clean_data:
        print("  ERROR: p69_clean_corpus.json not found.")
        return

    t1_catalogue = clean_data.get('t1_catalogue', [])
    print(f"  T1 words: {len(t1_catalogue)}")

    # Build T1 type → word and T1 type → folios
    t1_type_to_word: Dict[str, str] = {}
    t1_type_to_folios: Dict[str, List[str]] = {}
    for entry in t1_catalogue:
        t1_type_to_word[entry['eva_type']] = entry['matched_word']
        t1_type_to_folios[entry['eva_type']] = entry.get('folios', [])

    t1_words = sorted(set(t1_type_to_word.values()))

    # --- Load corpus for folio info ---
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Build folio → section mapping
    folio_to_section: Dict[str, str] = {}
    folio_list: List[str] = []
    for folio, page in corpus.pages.items():
        folio_to_section[folio] = getattr(page, 'section', 'unknown')
        for _ in page.all_tokens:
            folio_list.append(folio)

    # Build folio → T1 words present
    folio_t1_words: Dict[str, Set[str]] = {}
    for idx, token in enumerate(all_tokens):
        if token in t1_type_to_word:
            folio = folio_list[idx] if idx < len(folio_list) else '?'
            if folio not in folio_t1_words:
                folio_t1_words[folio] = set()
            folio_t1_words[folio].add(t1_type_to_word[token])

    # --- Load CI entries ---
    ci_path = os.path.join(str(_data_dir()), 'reference', 'latin', 'circa_instans.txt')
    ci_entries = _load_ci_entries(ci_path)
    print(f"  CI entries: {len(ci_entries)}")

    if not ci_entries:
        print("  WARNING: No CI data found. Proceeding with empty results.")

    # --- Map T1 words to CI entries ---
    print("\n  Mapping T1 words to CI entries...")
    word_to_ci: Dict[str, List[int]] = {}

    for word in t1_words:
        matching = []
        for ci_idx, entry_words in enumerate(ci_entries):
            for ew in entry_words:
                if word == ew or (len(word) >= 3 and _edit_distance(word, ew) <= 1):
                    matching.append(ci_idx)
                    break
        if matching:
            word_to_ci[word] = matching

    n_t1_in_ci = len(word_to_ci)
    print(f"  T1 words found in CI: {n_t1_in_ci}/{len(t1_words)}")

    word_ci_matches = []
    for word in sorted(word_to_ci.keys()):
        ci_indices = word_to_ci[word]
        titles = [_ci_entry_title(ci_entries, i) for i in ci_indices[:5]]
        word_ci_matches.append({
            'word': word,
            'n_ci_entries': len(ci_indices),
            'ci_entry_titles': titles,
        })

    # --- Build folio → topic assignments ---
    print("\n  Building folio-topic assignments...")
    folio_topics = []

    for folio, t1_on_folio in sorted(folio_t1_words.items()):
        # For each pair of T1 words on this folio, check if they share a CI entry
        words_on_folio = sorted(t1_on_folio & set(word_to_ci.keys()))
        if len(words_on_folio) < 2:
            continue

        # Find CI entries that contain 2+ of these words
        ci_hit_counts: Counter = Counter()
        for word in words_on_folio:
            for ci_idx in word_to_ci.get(word, []):
                ci_hit_counts[ci_idx] += 1

        shared_entries = [(ci_idx, count)
                         for ci_idx, count in ci_hit_counts.most_common()
                         if count >= 2]

        if shared_entries:
            best_ci, best_count = shared_entries[0]
            topic_words = [w for w in words_on_folio
                          if best_ci in word_to_ci.get(w, [])]
            folio_topics.append({
                'folio': folio,
                'section': folio_to_section.get(folio, 'unknown'),
                'ci_entry_idx': best_ci,
                'ci_entry_title': _ci_entry_title(ci_entries, best_ci),
                'n_matching_words': best_count,
                'matching_words': topic_words,
                'n_t1_on_folio': len(t1_on_folio),
            })

    n_folio_assignments = len(folio_topics)
    folio_topics.sort(key=lambda f: -f['n_matching_words'])
    print(f"  Folio-topic assignments: {n_folio_assignments}")

    # Section coverage
    section_topics_dict: Dict[str, List[str]] = {}
    for ft in folio_topics:
        sec = ft['section']
        if sec not in section_topics_dict:
            section_topics_dict[sec] = []
        section_topics_dict[sec].append(ft['ci_entry_title'])

    # --- Permutation test ---
    print(f"\n  Permutation test ({N_TRIALS} trials)...")

    # Real overlap score = total n_matching_words across all folio assignments
    real_overlap = sum(ft['n_matching_words'] for ft in folio_topics)
    print(f"    Real overlap score: {real_overlap}")

    # Null: shuffle which folios T1 words appear on
    all_folios = sorted(folio_t1_words.keys())
    all_folio_word_sets = [folio_t1_words[f] for f in all_folios]

    null_overlaps: List[float] = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(seed=trial)

        # Shuffle the word-set assignments across folios
        shuffled_indices = rng.permutation(len(all_folio_word_sets))
        shuffled_folio_words: Dict[str, Set[str]] = {}
        for i, folio in enumerate(all_folios):
            shuffled_folio_words[folio] = all_folio_word_sets[shuffled_indices[i]]

        # Recompute overlap
        trial_overlap = 0
        for folio, t1_on_folio in shuffled_folio_words.items():
            words_on_folio = sorted(t1_on_folio & set(word_to_ci.keys()))
            if len(words_on_folio) < 2:
                continue
            ci_hit_counts: Counter = Counter()
            for word in words_on_folio:
                for ci_idx in word_to_ci.get(word, []):
                    ci_hit_counts[ci_idx] += 1
            shared = [(ci_idx, count) for ci_idx, count in ci_hit_counts.most_common()
                     if count >= 2]
            trial_overlap += sum(count for _, count in shared)

        null_overlaps.append(trial_overlap)

    null_mean = float(np.mean(null_overlaps))
    null_std = float(np.std(null_overlaps))
    perm_z = (real_overlap - null_mean) / null_std if null_std > 0 else 0.0
    perm_p = sum(1 for n in null_overlaps if n >= real_overlap) / N_TRIALS

    print(f"    Null mean: {null_mean:.1f} ± {null_std:.1f}")
    print(f"    z = {perm_z:.2f}, p = {perm_p:.4f}")

    # --- Gates ---
    gate_ci1 = n_t1_in_ci >= 50
    gate_ci2 = n_folio_assignments >= 10
    gate_ci3 = perm_p < 0.05
    gates_passed = sum([gate_ci1, gate_ci2, gate_ci3])

    result = T1CIResult(
        n_t1_words=len(t1_catalogue),
        n_ci_entries=len(ci_entries),
        n_t1_in_ci=n_t1_in_ci,
        t1_in_ci_fraction=round(n_t1_in_ci / len(t1_words), 3) if t1_words else 0.0,
        word_ci_matches=word_ci_matches[:100],
        folio_topics=folio_topics[:50],
        n_folio_assignments=n_folio_assignments,
        real_overlap_score=real_overlap,
        null_mean=round(null_mean, 2),
        null_std=round(null_std, 2),
        perm_z=round(perm_z, 2),
        perm_p=round(perm_p, 4),
        n_trials=N_TRIALS,
        section_topics=section_topics_dict,
        gate_ci1=gate_ci1,
        gate_ci2=gate_ci2,
        gate_ci3=gate_ci3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p69_t1_ci.json', result)

    print(f"\n  Summary")
    print(f"  -------")
    print(f"  T1 in CI:        {n_t1_in_ci} ({'PASS' if gate_ci1 else 'FAIL'} >= 50)")
    print(f"  Folio topics:    {n_folio_assignments} ({'PASS' if gate_ci2 else 'FAIL'} >= 10)")
    print(f"  Perm p:          {perm_p:.4f} ({'PASS' if gate_ci3 else 'FAIL'} < 0.05)")
    print(f"  Gates: {gates_passed}/3")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
