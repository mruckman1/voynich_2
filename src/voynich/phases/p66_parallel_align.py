"""
Phase 66, Track 5: Circa Instans Parallel Corpus Alignment
============================================================
N-gram overlap between syllabified CI entries and Voynich folio decoded
profiles. Null: 100 shuffled CI entry labels.

Dependency chain:
    data/reference/latin/circa_instans.txt
    results/combined_refine.json      (Phase 15)
        -> results/p66_parallel_align.json
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import data_dir as _data_dir, results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.stats import syllabify_latin
from voynich.phases.corrected_coda import build_coda_table_v2, decode_corpus_cvc_v2


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
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParallelAlignResult:
    phase: str = "66"
    step: str = "66.5"
    experiment: str = "ci_parallel_alignment"
    n_ci_entries: int = 0
    n_folios: int = 0
    real_mean_score: float = 0.0
    null_mean_score: float = 0.0
    null_std_score: float = 0.0
    selectivity: float = 0.0
    n_above_95th: int = 0
    top_alignments: List[Dict] = field(default_factory=list)
    p1_selectivity: bool = False   # selectivity > 1.5
    p2_above_95th: bool = False    # >= 5 above 95th pctile
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_circa_instans(path: str) -> List[Dict[str, str]]:
    """
    Parse circa_instans.txt into entries.

    Entries are numbered (e.g. '2. Accacia , sucus prunellarum :')
    and separated by blank lines or the next numbered entry.
    Returns list of {'name': ..., 'text': ...}.
    """
    if not os.path.exists(path):
        return []

    with open(path, encoding='utf-8', errors='replace') as f:
        raw = f.read()

    # Split on numbered entry headers: "N. Name ..."
    entry_pattern = re.compile(r'(?:^|\n)(\d+)\.\s+(.+?)(?:\s*:|$)', re.MULTILINE)
    matches = list(entry_pattern.finditer(raw))

    entries = []
    for i, m in enumerate(matches):
        name = m.group(2).strip().split(',')[0].strip()  # first name before comma
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        # Clean up quotation marks and extra whitespace
        body = re.sub(r"['']+", '', body)
        body = re.sub(r'\s+', ' ', body)
        if name and body:
            entries.append({'name': name.lower(), 'text': body.lower()})

    # If no numbered entries found, treat the whole file as entry 1
    if not entries:
        text = re.sub(r'\s+', ' ', raw).strip().lower()
        if text:
            entries.append({'name': 'iarus', 'text': text})

    return entries


def _syllabify_text(text: str) -> str:
    """Syllabify all words in text and concatenate syllables."""
    words = re.findall(r'[a-z]+', text)
    syllables = []
    for w in words:
        syls = syllabify_latin(w)
        syllables.extend(syls)
    return ''.join(syllables)


def _char_trigram_profile(text: str) -> Set[str]:
    """Build character trigram set from text."""
    trigrams = set()
    for i in range(len(text) - 2):
        trigrams.add(text[i:i + 3])
    return trigrams


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_parallel_align() -> ParallelAlignResult:
    t0 = time.time()
    rd = str(_results_dir())
    result = ParallelAlignResult()

    print("=" * 70)
    print("Phase 66, Track 5: CI Parallel Corpus Alignment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and parse Circa Instans
    # ------------------------------------------------------------------
    ci_path = str(_data_dir('reference') / 'latin' / 'circa_instans.txt')
    entries = _parse_circa_instans(ci_path)
    if not entries:
        print(f"[WARN] circa_instans.txt not found or empty at {ci_path}")
        result.verdict = "INSUFFICIENT_DATA"
        result.runtime_seconds = round(time.time() - t0, 2)
        _save_json(rd, 'p66_parallel_align.json', result)
        return result

    result.n_ci_entries = len(entries)
    print(f"  CI entries parsed: {len(entries)}")

    # ------------------------------------------------------------------
    # 2. Syllabify CI entries and build trigram profiles
    # ------------------------------------------------------------------
    print("  Syllabifying CI entries ...")
    ci_profiles: List[Tuple[str, Set[str]]] = []
    for entry in entries:
        syl_text = _syllabify_text(entry['text'])
        profile = _char_trigram_profile(syl_text)
        ci_profiles.append((entry['name'], profile))

    # ------------------------------------------------------------------
    # 3. Load corpus, CVC decode, build folio profiles
    # ------------------------------------------------------------------
    print("  Loading corpus and decoding tokens ...")
    corpus = load_corpus(verbose=False)
    assignment = _safe_load(os.path.join(rd, 'combined_refine.json')).get(
        'best_assignment', {}
    )
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    folio_profiles: Dict[str, Set[str]] = {}
    folio_ids: List[str] = []

    for folio_id, page in corpus.pages.items():
        tokens = page.all_tokens
        if not tokens:
            continue
        decoded = decode_corpus_cvc_v2(tokens, assignment, eva_to_triple, coda_table)
        decoded_clean = [d for d in decoded if d and d != '???']
        concat = ''.join(decoded_clean).lower()
        profile = _char_trigram_profile(concat)
        if profile:
            folio_profiles[folio_id] = profile
            folio_ids.append(folio_id)

    result.n_folios = len(folio_profiles)
    print(f"  Folio profiles built: {result.n_folios}")

    if not folio_profiles:
        print("[WARN] No folio profiles produced.")
        result.verdict = "INSUFFICIENT_DATA"
        result.runtime_seconds = round(time.time() - t0, 2)
        _save_json(rd, 'p66_parallel_align.json', result)
        return result

    # ------------------------------------------------------------------
    # 4. Compute best-match Jaccard for each CI entry
    # ------------------------------------------------------------------
    print("  Computing Jaccard overlaps ...")
    real_best_scores: List[float] = []
    top_alignments: List[Dict] = []

    for ci_name, ci_profile in ci_profiles:
        best_score = 0.0
        best_folio = ''
        for fid in folio_ids:
            score = _jaccard(ci_profile, folio_profiles[fid])
            if score > best_score:
                best_score = score
                best_folio = fid
        real_best_scores.append(best_score)
        top_alignments.append({
            'ci_entry': ci_name,
            'best_folio': best_folio,
            'jaccard': round(best_score, 6),
        })

    real_scores_arr = np.array(real_best_scores)
    real_mean = float(np.mean(real_scores_arr))
    result.real_mean_score = round(real_mean, 6)

    # Sort top alignments by score descending, keep top 20
    top_alignments.sort(key=lambda x: x['jaccard'], reverse=True)
    result.top_alignments = top_alignments[:20]

    print(f"  Real mean best-match Jaccard: {result.real_mean_score}")

    # ------------------------------------------------------------------
    # 5. Null: 100 shuffles of CI entry labels
    # ------------------------------------------------------------------
    print("  Running 100-trial shuffle null ...")
    rng = random.Random(42)
    null_means: List[float] = []
    all_null_scores: List[float] = []

    for _ in range(100):
        # Shuffle CI profiles: reassign names to different profiles
        shuffled_profiles = [p for _, p in ci_profiles]
        rng.shuffle(shuffled_profiles)
        trial_scores = []
        for ci_profile in shuffled_profiles:
            best_score = 0.0
            for fid in folio_ids:
                score = _jaccard(ci_profile, folio_profiles[fid])
                if score > best_score:
                    best_score = score
            trial_scores.append(best_score)
            all_null_scores.append(best_score)
        null_means.append(float(np.mean(trial_scores)))

    null_means_arr = np.array(null_means)
    null_mean = float(np.mean(null_means_arr))
    null_std = float(np.std(null_means_arr))
    result.null_mean_score = round(null_mean, 6)
    result.null_std_score = round(null_std, 6)

    # Selectivity
    if null_mean > 0:
        result.selectivity = round(real_mean / null_mean, 4)
    else:
        result.selectivity = float(real_mean) if real_mean > 0 else 0.0

    # 95th percentile of individual null best-match scores
    if all_null_scores:
        pctile_95 = float(np.percentile(all_null_scores, 95))
        result.n_above_95th = int(np.sum(real_scores_arr > pctile_95))
    else:
        result.n_above_95th = 0

    print(f"  Null mean: {result.null_mean_score}, std: {result.null_std_score}")
    print(f"  Selectivity: {result.selectivity}x")
    print(f"  Entries above 95th pctile: {result.n_above_95th}")

    # ------------------------------------------------------------------
    # 6. Gate evaluation
    # ------------------------------------------------------------------
    result.p1_selectivity = result.selectivity > 1.5
    result.p2_above_95th = result.n_above_95th >= 5
    result.gates_passed = sum([result.p1_selectivity, result.p2_above_95th])
    result.gate_passed = result.gates_passed >= 2

    if result.gate_passed:
        result.verdict = "CI_ALIGNMENT_SIGNIFICANT"
    elif result.gates_passed == 1:
        result.verdict = "CI_ALIGNMENT_MARGINAL"
    else:
        result.verdict = "CI_ALIGNMENT_NOT_FOUND"

    # ------------------------------------------------------------------
    # 7. Print summary and save
    # ------------------------------------------------------------------
    result.runtime_seconds = round(time.time() - t0, 2)

    print()
    print("-" * 50)
    print(f"  P1 selectivity > 1.5x:    {result.p1_selectivity}  ({result.selectivity}x)")
    print(f"  P2 >= 5 above 95th pctile: {result.p2_above_95th}  ({result.n_above_95th})")
    print(f"  Gates passed:             {result.gates_passed}/2")
    print(f"  Verdict:                  {result.verdict}")
    print(f"  Runtime:                  {result.runtime_seconds}s")
    print("-" * 50)

    path = _save_json(rd, 'p66_parallel_align.json', result)
    print(f"  Saved: {path}")
    return result
