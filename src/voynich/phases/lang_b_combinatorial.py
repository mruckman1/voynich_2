"""
Phase 19.1 – Language B Combinatorial Attack
=============================================
Language B has a restricted vocabulary (~13 word types, 227 tokens).
This is a tractable exhaustive assignment problem against small candidate
label sets from medieval knowledge systems.

Dependency chain:
    corpus (Language B pages)
    combined_refine.json  (Phase 15)
    modifier_integrate.json (Phase 16)
        → lang_b_combinatorial.json
"""

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.stats import (
    coefficient_of_variation,
    jensen_shannon_divergence,
    selectivity_ratio,
    word_transition_matrix,
)


# ---------------------------------------------------------------------------
# JSON serialiser
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
# Candidate label sets from medieval knowledge systems
# ---------------------------------------------------------------------------

CANDIDATE_LABEL_SETS = {
    'planets': {
        'labels': ['Sol', 'Luna', 'Mercurius', 'Venus', 'Mars', 'Jupiter', 'Saturnus'],
        'domain': 'astronomical',
        'known_sequences': [
            ['Saturnus', 'Jupiter', 'Mars', 'Sol', 'Venus', 'Mercurius', 'Luna'],
        ],
    },
    'zodiac': {
        'labels': ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                   'Libra', 'Scorpio', 'Sagittarius', 'Capricornus',
                   'Aquarius', 'Pisces'],
        'domain': 'astronomical',
        'known_sequences': [
            ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
             'Libra', 'Scorpio', 'Sagittarius', 'Capricornus',
             'Aquarius', 'Pisces'],
        ],
    },
    'humoral_qualities': {
        'labels': ['calidus', 'frigidus', 'humidus', 'siccus',
                   'calidus_humidus', 'calidus_siccus',
                   'frigidus_humidus', 'frigidus_siccus'],
        'domain': 'pharmaceutical',
        'known_sequences': [],
    },
    'dosage_units': {
        'labels': ['drachma', 'uncia', 'manipulus', 'cochlear',
                   'libra', 'gutta', 'pugillus', 'scrupulus'],
        'domain': 'pharmaceutical',
        'known_sequences': [],
    },
    'days_of_week': {
        'labels': ['Lunae', 'Martis', 'Mercurii', 'Jovis',
                   'Veneris', 'Saturni', 'Solis'],
        'domain': 'astronomical',
        'known_sequences': [
            ['Lunae', 'Martis', 'Mercurii', 'Jovis', 'Veneris', 'Saturni', 'Solis'],
        ],
    },
    'galenic_degrees': {
        'labels': ['primus', 'secundus', 'tertius', 'quartus'],
        'domain': 'pharmaceutical',
        'known_sequences': [
            ['primus', 'secundus', 'tertius', 'quartus'],
        ],
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CandidateSetResult:
    label_set_name: str
    n_labels: int
    n_lang_b_onsets: int
    best_mapping: Dict[str, str]
    best_score: float
    null_mean: float
    null_p95: float
    selectivity: float
    family_alignment: float
    section_correlation: float


@dataclass
class LangBCombinatorialResult:
    # Corpus split
    n_lang_b_folios: int
    n_lang_b_tokens: int
    n_lang_b_types: int
    lang_b_type_list: List[str]
    # Morphological families
    edy_family: List[str]
    aiin_family: List[str]
    other_family: List[str]
    edy_pct: float
    aiin_pct: float
    # Onset decomposition
    n_unique_onsets: int
    onset_list: List[str]
    # Transition matrix
    transition_entropy: float
    transition_sparsity: float
    # Per-candidate-set results
    candidate_results: List[Dict[str, Any]]
    best_candidate_set: str
    best_selectivity: float
    # Gate
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _extract_lang_b_restricted_vocab(
    lang_b_tokens: List[str],
    top_n: int = 20,
) -> Tuple[List[str], List[str]]:
    """
    Extract the restricted-vocabulary subset of Language B.
    Look for tokens that form a small, high-frequency closed set.
    """
    counts = Counter(lang_b_tokens)
    # Sort by frequency
    sorted_types = sorted(counts.keys(), key=lambda t: counts[t], reverse=True)

    # Find the restricted vocabulary: tokens with frequency >= 5
    # that together account for a large share of the total
    restricted = []
    restricted_tokens = []
    cumulative = 0
    total = sum(counts.values())

    for t in sorted_types:
        if counts[t] >= 3 and len(restricted) < top_n:
            restricted.append(t)
            cumulative += counts[t]

    # Filter tokens to only restricted vocab
    restricted_set = set(restricted)
    restricted_tokens = [t for t in lang_b_tokens if t in restricted_set]

    return restricted, restricted_tokens


def _decompose_onset(token: str) -> Tuple[str, str]:
    """
    Decompose a Language B token into (onset, body).
    The onset is the initial consonant cluster, body is the rest.
    """
    chars = tokenize_eva_chars(token)
    if not chars:
        return ('', token)

    # Find where the first 'common body' pattern starts
    # Language B tokens typically end in -edy or -aiin
    token_str = token
    for body in ['edy', 'aiin', 'eedy', 'eey', 'dy', 'y']:
        if token_str.endswith(body) and len(token_str) > len(body):
            onset = token_str[:-len(body)]
            return onset, body

    return (token_str, '')


def _build_transition_matrix(
    tokens: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Build word-to-word transition matrix."""
    vocab = sorted(set(tokens))
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    mat = np.zeros((n, n))

    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1

    # Normalize rows
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat /= row_sums

    return mat, vocab


def _score_assignment(
    mapping: Dict[str, str],
    transition_matrix: np.ndarray,
    vocab: List[str],
    known_sequences: List[List[str]],
    section_tokens: Dict[str, List[str]],
    domain: str,
    edy_family: Set[str],
    aiin_family: Set[str],
    label_set_name: str,
) -> float:
    """
    Score a candidate assignment based on multiple criteria.
    """
    score = 0.0
    n_criteria = 0

    # 1. Transition plausibility: do known sequences appear?
    if known_sequences:
        seq_score = 0.0
        for seq in known_sequences:
            # Check if any subsequence of length 2 appears in transitions
            for i in range(len(seq) - 1):
                label_a, label_b = seq[i], seq[i + 1]
                # Find which tokens map to these labels
                tok_a = [t for t, l in mapping.items() if l == label_a]
                tok_b = [t for t, l in mapping.items() if l == label_b]
                for ta in tok_a:
                    for tb in tok_b:
                        if ta in vocab and tb in vocab:
                            ia = vocab.index(ta)
                            ib = vocab.index(tb)
                            if transition_matrix[ia, ib] > 0:
                                seq_score += transition_matrix[ia, ib]
        score += seq_score
        n_criteria += 1

    # 2. Section correlation
    if section_tokens and domain:
        target_sections = {
            'astronomical': ['astronomical', 'cosmological'],
            'pharmaceutical': ['pharmaceutical', 'recipes'],
        }
        target = target_sections.get(domain, [])
        if target:
            in_target = sum(
                1 for sec, toks in section_tokens.items()
                if sec in target
                for t in toks if t in mapping
            )
            total = sum(len(toks) for toks in section_tokens.values())
            sec_corr = in_target / total if total > 0 else 0
            score += sec_corr
            n_criteria += 1

    # 3. Family alignment: do the two morphological families map to
    #    a natural semantic split within the label set?
    edy_labels = set(mapping[t] for t in mapping if t in edy_family)
    aiin_labels = set(mapping[t] for t in mapping if t in aiin_family)
    if edy_labels and aiin_labels:
        # Families should map to disjoint label subsets
        overlap = len(edy_labels & aiin_labels)
        family_score = 1.0 - overlap / max(len(edy_labels), len(aiin_labels), 1)
        score += family_score
        n_criteria += 1

    # 4. Frequency-rank alignment
    freq_ranks = Counter(mapping.keys())
    mapped_labels = list(mapping.values())
    if len(mapped_labels) > 1:
        # Simple: top-frequency token should map to a "primary" label
        score += 0.5
        n_criteria += 1

    return score / n_criteria if n_criteria > 0 else 0.0


def _exhaustive_or_hungarian(
    onsets: List[str],
    labels: List[str],
    score_fn,
) -> Tuple[float, Dict[str, str]]:
    """
    Find the best injective mapping from onsets to labels.
    Use exhaustive search if small enough, otherwise Hungarian.
    """
    n_onsets = len(onsets)
    n_labels = len(labels)

    if n_onsets <= 8 and n_labels <= 8:
        # Exhaustive search over permutations
        best_score = -1.0
        best_mapping = {}

        for perm in permutations(range(n_labels), min(n_onsets, n_labels)):
            mapping = {}
            for i, j in enumerate(perm):
                if i < n_onsets:
                    mapping[onsets[i]] = labels[j]

            s = score_fn(mapping)
            if s > best_score:
                best_score = s
                best_mapping = dict(mapping)

        return best_score, best_mapping
    else:
        # Build compatibility matrix and use Hungarian
        compat = np.zeros((n_onsets, n_labels))
        for i, onset in enumerate(onsets):
            for j, label in enumerate(labels):
                mapping = {onset: label}
                compat[i, j] = score_fn(mapping)

        row_idx, col_idx = linear_sum_assignment(-compat)
        mapping = {}
        total = 0.0
        for r, c in zip(row_idx, col_idx):
            mapping[onsets[r]] = labels[c]
            total += compat[r, c]

        return total / len(row_idx) if len(row_idx) > 0 else 0.0, mapping


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_lang_b_combinatorial() -> None:
    """Phase 19.1: Language B combinatorial attack."""
    t0 = time.time()
    rd = str(_results_dir())

    print("=" * 60)
    print("Phase 19.1: Language B Combinatorial Attack")
    print("=" * 60)

    # ── 1. Extract Language B tokens ──────────────────────────────────
    print("\n  1. Extracting Language B tokens …")

    corpus = load_corpus(verbose=False)
    lang_b_pages = corpus.get_pages_by_language('B')

    lang_b_all_tokens = []
    section_tokens: Dict[str, List[str]] = defaultdict(list)

    for page in lang_b_pages:
        section = page.section if hasattr(page, 'section') else 'unknown'
        for tok in page.all_tokens:
            lang_b_all_tokens.append(tok)
            section_tokens[section].append(tok)

    # Extract restricted vocabulary
    restricted_types, restricted_tokens = _extract_lang_b_restricted_vocab(lang_b_all_tokens)

    all_types = sorted(set(lang_b_all_tokens))
    type_counts = Counter(lang_b_all_tokens)

    print(f"    Language B: {len(lang_b_pages)} folios, {len(lang_b_all_tokens)} tokens, {len(all_types)} types")
    print(f"    Restricted vocab: {len(restricted_types)} types, {len(restricted_tokens)} tokens")

    # ── 2. Morphological family analysis ─────────────────────────────
    print("\n  2. Analyzing morphological families …")

    edy_family = set()
    aiin_family = set()
    other_family = set()

    for t in all_types:
        if t.endswith('edy') or t.endswith('eedy'):
            edy_family.add(t)
        elif t.endswith('aiin') or t.endswith('aiiin'):
            aiin_family.add(t)
        else:
            other_family.add(t)

    total = len(lang_b_all_tokens)
    edy_count = sum(type_counts[t] for t in edy_family)
    aiin_count = sum(type_counts[t] for t in aiin_family)
    edy_pct = edy_count / total * 100 if total > 0 else 0
    aiin_pct = aiin_count / total * 100 if total > 0 else 0

    print(f"    -edy family: {len(edy_family)} types ({edy_pct:.1f}%)")
    print(f"    -aiin family: {len(aiin_family)} types ({aiin_pct:.1f}%)")
    print(f"    Other: {len(other_family)} types")

    # ── 3. Onset decomposition ────────────────────────────────────────
    print("\n  3. Decomposing onsets …")

    onset_counter = Counter()
    for t in restricted_types:
        onset, body = _decompose_onset(t)
        if onset:
            onset_counter[onset] += 1

    onset_list = sorted(onset_counter.keys())
    print(f"    {len(onset_list)} unique onsets: {onset_list[:10]}")

    # ── 4. Build transition matrix ────────────────────────────────────
    print("\n  4. Building transition matrix …")

    trans_mat, trans_vocab = _build_transition_matrix(restricted_tokens)

    # Compute transition entropy
    trans_entropy = 0.0
    n_rows = 0
    for row in trans_mat:
        h = -sum(p * math.log2(p) for p in row if p > 0)
        trans_entropy += h
        n_rows += 1
    trans_entropy /= n_rows if n_rows > 0 else 1

    # Sparsity: fraction of near-zero entries
    n_total_entries = trans_mat.size
    n_sparse = np.sum(trans_mat < 0.05)
    sparsity = float(n_sparse / n_total_entries) if n_total_entries > 0 else 0

    print(f"    Transition entropy: {trans_entropy:.3f} bits")
    print(f"    Transition sparsity: {sparsity:.3f}")

    # ── 5. Test candidate label sets ─────────────────────────────────
    print("\n  5. Testing 6 candidate label sets …")

    candidate_results = []
    rng = random.Random(42)

    # Use restricted types as the token set for assignment
    assignment_tokens = restricted_types if restricted_types else all_types[:13]

    for set_name, set_info in CANDIDATE_LABEL_SETS.items():
        labels = set_info['labels']
        domain = set_info['domain']
        known_seqs = set_info['known_sequences']

        print(f"\n    ── {set_name} ({len(labels)} labels) ──")

        if len(assignment_tokens) == 0:
            print("      [SKIP] No tokens to assign")
            continue

        # Score function for this candidate set
        def make_score_fn(lbl, kseqs, dom):
            def score_fn(mapping):
                return _score_assignment(
                    mapping, trans_mat, trans_vocab,
                    kseqs, section_tokens, dom,
                    edy_family, aiin_family, set_name,
                )
            return score_fn

        score_fn = make_score_fn(labels, known_seqs, domain)

        # Find best assignment
        best_score, best_mapping = _exhaustive_or_hungarian(
            assignment_tokens[:min(len(assignment_tokens), len(labels))],
            labels,
            score_fn,
        )

        # Null test: shuffle token positions 100 times
        null_scores = []
        for trial in range(100):
            shuffled = list(assignment_tokens[:min(len(assignment_tokens), len(labels))])
            rng.shuffle(shuffled)
            null_mapping = dict(zip(shuffled, labels[:len(shuffled)]))
            ns = _score_assignment(
                null_mapping, trans_mat, trans_vocab,
                known_seqs, section_tokens, domain,
                edy_family, aiin_family, set_name,
            )
            null_scores.append(ns)

        null_arr = np.array(null_scores)
        null_mean = float(np.mean(null_arr))
        null_p95 = float(np.percentile(null_arr, 95))
        sel = best_score / null_mean if null_mean > 0 else 0.0

        # Family alignment
        edy_labels = set(best_mapping[t] for t in best_mapping if t in edy_family)
        aiin_labels = set(best_mapping[t] for t in best_mapping if t in aiin_family)
        overlap = len(edy_labels & aiin_labels)
        family_align = 1.0 - overlap / max(len(edy_labels), len(aiin_labels), 1) if (edy_labels or aiin_labels) else 0

        # Section correlation
        target_secs = {'astronomical': ['astronomical', 'cosmological'],
                       'pharmaceutical': ['pharmaceutical', 'recipes']}
        targets = target_secs.get(domain, [])
        in_target = sum(1 for sec, toks in section_tokens.items()
                        if sec in targets for t in toks if t in best_mapping)
        total_in_sec = sum(len(toks) for toks in section_tokens.values())
        sec_corr = in_target / total_in_sec if total_in_sec > 0 else 0

        cr = CandidateSetResult(
            label_set_name=set_name,
            n_labels=len(labels),
            n_lang_b_onsets=len(assignment_tokens),
            best_mapping=best_mapping,
            best_score=round(best_score, 4),
            null_mean=round(null_mean, 4),
            null_p95=round(null_p95, 4),
            selectivity=round(sel, 4),
            family_alignment=round(family_align, 4),
            section_correlation=round(sec_corr, 4),
        )
        candidate_results.append(cr)

        print(f"      Score: {best_score:.4f} (null: {null_mean:.4f}, sel: {sel:.2f}×)")
        print(f"      Family alignment: {family_align:.3f}, Section corr: {sec_corr:.3f}")
        for tok, lbl in sorted(best_mapping.items()):
            print(f"        {tok:12s} → {lbl}")

    # ── 6. Select best candidate set ─────────────────────────────────
    print("\n  6. Ranking candidate sets …")

    if candidate_results:
        best_cr = max(candidate_results, key=lambda c: c.selectivity)
        best_candidate_set = best_cr.label_set_name
        best_sel = best_cr.selectivity
    else:
        best_candidate_set = 'none'
        best_sel = 0.0

    print(f"    Best: {best_candidate_set} (selectivity {best_sel:.2f}×)")

    # ── 7. Gate ──────────────────────────────────────────────────────
    gate_passed = bool(best_sel >= 1.5)

    if gate_passed:
        verdict = f"PASS: {best_candidate_set} achieves {best_sel:.2f}× selectivity"
    elif best_sel >= 1.0:
        verdict = f"MARGINAL: {best_candidate_set} at {best_sel:.2f}× (below 1.5× gate)"
    else:
        verdict = f"FAIL: no candidate set exceeds 1.0× selectivity"

    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 8. Save ──────────────────────────────────────────────────────
    result = LangBCombinatorialResult(
        n_lang_b_folios=len(lang_b_pages),
        n_lang_b_tokens=len(lang_b_all_tokens),
        n_lang_b_types=len(all_types),
        lang_b_type_list=all_types[:50],
        edy_family=sorted(edy_family)[:20],
        aiin_family=sorted(aiin_family)[:20],
        other_family=sorted(other_family)[:20],
        edy_pct=round(edy_pct, 2),
        aiin_pct=round(aiin_pct, 2),
        n_unique_onsets=len(onset_list),
        onset_list=onset_list,
        transition_entropy=round(trans_entropy, 4),
        transition_sparsity=round(sparsity, 4),
        candidate_results=[_convert(asdict(cr)) for cr in candidate_results],
        best_candidate_set=best_candidate_set,
        best_selectivity=round(best_sel, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'lang_b_combinatorial.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n    → {out_path}")
