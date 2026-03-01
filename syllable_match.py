"""
Workstream F: Syllable-Level Retranscription and Language Matching
===================================================================
Convert Voynich text through the syllabary grid into abstract C_iV_j labels,
compute syllable-level statistics, and match against candidate languages.

Steps:
  F.1 — Provisional CV labeling of grid cells
  F.2 — Full-corpus syllabic retranscription
  F.3 — Syllable bigram matching against candidate languages
  F.4 — Mutual information between Voynich and Latin syllable sequences
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from corpus import load_corpus, VoynichCorpus, tokenize_eva_chars
from stats import (
    syllabify_latin, syllabify_latin_text,
    bigram_transition_matrix, cosine_similarity, jensen_shannon_divergence,
    frobenius_distance, bootstrap_ci, pearson_correlation, first_order_entropy,
)
from strokes import (
    SyllabaryGrid, build_ventris_grid, decompose_glyph,
    segment_token_as_syllables,
)
from reference import (
    load_reference_corpus, get_reference_text,
    get_reference_syllable_stats, ReferenceCorpus,
)
from grid_validate import build_grid_from_tokens


# ---------------------------------------------------------------------------
# F.1: Provisional CV Labeling
# ---------------------------------------------------------------------------

@dataclass
class CVLabel:
    """Abstract consonant-vowel label for a grid cell."""
    onset_class: str
    nucleus_class: str
    cv_label: str
    glyphs: List[str]
    frequency: int


def assign_cv_labels(
    grid: SyllabaryGrid,
    tokens: List[str],
) -> Dict[str, CVLabel]:
    """
    Assign abstract C_iV_j labels to each filled grid cell.

    Convention:
    - Rows (onsets) -> C1, C2, ..., Cn (ordered by total frequency)
    - Columns (nuclei) -> V1, V2, ..., Vm (ordered by total frequency)
    - Cell label = C_i + V_j
    """
    from grid_refine import segment_token_merged

    # Build merge maps for this grid
    onset_merge, nucleus_merge = _build_merge_maps(grid)

    # Count usage of each grid cell using merged labels
    cell_counts: Counter = Counter()
    for token in tokens:
        syls = segment_token_merged(token, onset_merge, nucleus_merge)
        cell_counts.update(syls)

    # Count onset and nucleus totals
    onset_totals: Counter = Counter()
    nucleus_totals: Counter = Counter()
    for cell_key, count in cell_counts.items():
        parts = cell_key.split(',', 1)
        if len(parts) != 2:
            continue
        onset, nucleus = parts
        onset_totals[onset] += count
        nucleus_totals[nucleus] += count

    # Rank by frequency
    onset_rank = {label: i + 1 for i, (label, _)
                  in enumerate(onset_totals.most_common())}
    nucleus_rank = {label: i + 1 for i, (label, _)
                    in enumerate(nucleus_totals.most_common())}

    # Build CV labels
    cv_labels: Dict[str, CVLabel] = {}
    for cell_key, glyphs in grid.cells.items():
        parts = cell_key.split(',', 1)
        if len(parts) != 2:
            continue
        onset, nucleus = parts

        c_idx = onset_rank.get(onset, len(onset_rank) + 1)
        v_idx = nucleus_rank.get(nucleus, len(nucleus_rank) + 1)
        cv_str = f"C{c_idx}V{v_idx}"

        cv_labels[cell_key] = CVLabel(
            onset_class=onset,
            nucleus_class=nucleus,
            cv_label=cv_str,
            glyphs=glyphs,
            frequency=cell_counts.get(cell_key, 0),
        )

    return cv_labels


# ---------------------------------------------------------------------------
# F.2: Full-Corpus Syllabic Retranscription
# ---------------------------------------------------------------------------

@dataclass
class RetranscriptionResult:
    """Statistics on the retranscribed (CV-labeled) corpus."""
    n_tokens: int
    n_cv_types: int
    n_cv_tokens: int
    cv_ttr: float
    cv_h1: float
    cv_h2: float
    mean_cv_per_word: float
    std_cv_per_word: float
    ambiguity_rate: float
    top_20_cv_labels: List[Tuple[str, int]]
    sample_retranscriptions: List[Tuple[str, str]]


def _build_merge_maps(grid: SyllabaryGrid) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build merge maps from raw stroke names to merged grid labels.

    A merged label like 'ascender+vertical' in row_labels means that both
    'ascender' and 'vertical' map to this merged category.
    """
    onset_merge: Dict[str, str] = {}
    for label in grid.row_labels:
        for part in label.split('+'):
            onset_merge[part] = label

    nucleus_merge: Dict[str, str] = {}
    for label in grid.col_labels:
        for part in label.split('+'):
            nucleus_merge[part] = label

    return onset_merge, nucleus_merge


def retranscribe_token(
    token: str,
    cv_labels: Dict[str, CVLabel],
    onset_merge: Optional[Dict[str, str]] = None,
    nucleus_merge: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Convert an EVA token into a sequence of CV labels."""
    result = []
    glyphs = tokenize_eva_chars(token)
    for g in glyphs:
        strokes = decompose_glyph(g)
        if strokes:
            raw_onset = strokes[0].value
            raw_nucleus = strokes[-1].value
            # Apply merge mapping if provided
            onset = onset_merge.get(raw_onset, raw_onset) if onset_merge else raw_onset
            nucleus = nucleus_merge.get(raw_nucleus, raw_nucleus) if nucleus_merge else raw_nucleus
            key = f"{onset},{nucleus}"
            if key in cv_labels:
                result.append(cv_labels[key].cv_label)
            else:
                result.append('?')
        else:
            result.append('?')
    return result


def retranscribe_corpus(
    tokens: List[str],
    cv_labels: Dict[str, CVLabel],
    grid: Optional[SyllabaryGrid] = None,
) -> RetranscriptionResult:
    """Retranscribe entire corpus and compute syllable-level statistics."""
    # Build merge maps from the grid
    onset_merge, nucleus_merge = None, None
    if grid is not None:
        onset_merge, nucleus_merge = _build_merge_maps(grid)

    all_cv: List[str] = []
    cv_per_word: List[int] = []
    n_ambiguous = 0
    n_total_glyphs = 0
    samples: List[Tuple[str, str]] = []

    for i, token in enumerate(tokens):
        cv_seq = retranscribe_token(token, cv_labels, onset_merge, nucleus_merge)
        all_cv.extend(cv_seq)
        cv_per_word.append(len(cv_seq))

        n_total_glyphs += len(cv_seq)
        n_ambiguous += sum(1 for c in cv_seq if c == '?')

        if i < 20:
            samples.append((token, '.'.join(cv_seq)))

    # Stats
    cv_counts = Counter(all_cv)
    n_types = len(cv_counts)
    n_tokens_cv = len(all_cv)

    # H1: Shannon entropy of CV labels
    h1 = 0.0
    for count in cv_counts.values():
        p = count / n_tokens_cv
        if p > 0:
            h1 -= p * math.log2(p)

    # H2: conditional entropy of CV bigrams
    bigrams: Counter = Counter()
    unigrams: Counter = Counter()
    for token in tokens:
        cv_seq = retranscribe_token(token, cv_labels, onset_merge, nucleus_merge)
        for k in range(len(cv_seq) - 1):
            bigrams[(cv_seq[k], cv_seq[k + 1])] += 1
            unigrams[cv_seq[k]] += 1

    h2 = 0.0
    total_bi = sum(bigrams.values())
    total_uni = sum(unigrams.values())
    if total_bi > 0 and total_uni > 0:
        h_joint = -sum((c / total_bi) * math.log2(c / total_bi)
                       for c in bigrams.values() if c > 0)
        h_ctx = -sum((c / total_uni) * math.log2(c / total_uni)
                     for c in unigrams.values() if c > 0)
        h2 = h_joint - h_ctx

    ambiguity_rate = n_ambiguous / n_total_glyphs if n_total_glyphs > 0 else 0

    return RetranscriptionResult(
        n_tokens=len(tokens),
        n_cv_types=n_types,
        n_cv_tokens=n_tokens_cv,
        cv_ttr=n_types / n_tokens_cv if n_tokens_cv > 0 else 0,
        cv_h1=round(h1, 4),
        cv_h2=round(h2, 4),
        mean_cv_per_word=float(np.mean(cv_per_word)),
        std_cv_per_word=float(np.std(cv_per_word)),
        ambiguity_rate=round(ambiguity_rate, 4),
        top_20_cv_labels=cv_counts.most_common(20),
        sample_retranscriptions=samples,
    )


# ---------------------------------------------------------------------------
# F.3: Syllable Bigram Matching Against Candidate Languages
# ---------------------------------------------------------------------------

@dataclass
class LanguageMatchResult:
    """Result of matching Voynich syllable bigrams to a language."""
    language: str
    optimal_distance: float
    jsd: float
    n_ref_syllable_types: int
    selectivity: float


def build_cv_bigram_matrix(
    tokens: List[str],
    cv_labels: Dict[str, CVLabel],
    grid: Optional[SyllabaryGrid] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build bigram transition matrix from CV-labeled tokens."""
    onset_merge, nucleus_merge = None, None
    if grid is not None:
        onset_merge, nucleus_merge = _build_merge_maps(grid)

    all_labels = sorted(set(lab.cv_label for lab in cv_labels.values()))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)

    counts = np.zeros((n, n), dtype=float)
    for token in tokens:
        cv_seq = retranscribe_token(token, cv_labels, onset_merge, nucleus_merge)
        for k in range(len(cv_seq) - 1):
            a, b = cv_seq[k], cv_seq[k + 1]
            if a in label_to_idx and b in label_to_idx:
                counts[label_to_idx[a]][label_to_idx[b]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = counts / row_sums

    return matrix, all_labels


def _optimal_mapping_distance(
    mat_v: np.ndarray,
    mat_r: np.ndarray,
) -> float:
    """Find optimal permutation of mat_r to minimize Frobenius distance to mat_v."""
    nv = mat_v.shape[0]
    nr = mat_r.shape[0]
    n = max(nv, nr)

    padded_v = np.zeros((n, n))
    padded_r = np.zeros((n, n))
    padded_v[:nv, :nv] = mat_v
    padded_r[:nr, :nr] = mat_r

    # Cost matrix
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.sum((padded_v[i] - padded_r[j]) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost)
    perm_r = padded_r[col_ind][:, col_ind]

    return float(np.linalg.norm(padded_v - perm_r))


def match_against_languages(
    tokens: List[str],
    cv_labels: Dict[str, CVLabel],
    reference_corpus: Optional[ReferenceCorpus] = None,
    languages: Optional[List[str]] = None,
    seed: int = 42,
) -> List[LanguageMatchResult]:
    """
    For each candidate language, find optimal mapping and score.
    """
    if reference_corpus is None:
        reference_corpus = load_reference_corpus(verbose=False)

    if languages is None:
        languages = list(reference_corpus.languages) if reference_corpus.languages else ['latin']

    # Build Voynich CV bigram matrix (need grid for merge maps)
    # Reconstruct grid for merge maps
    grid_for_maps = build_grid_from_tokens(tokens)
    v_mat, v_labels = build_cv_bigram_matrix(tokens, cv_labels, grid_for_maps)

    results = []
    for lang in languages:
        try:
            ref_stats = get_reference_syllable_stats(
                lang, corpus=reference_corpus, n_words=5000, seed=seed)
        except Exception:
            continue

        s_mat, s_alph = ref_stats['syllable_bigrams']
        n_ref_types = len(s_alph)

        # Trim reference matrix to top-N matching Voynich size
        n_v = v_mat.shape[0]
        if s_mat.shape[0] > n_v:
            syl_freqs = s_mat.sum(axis=1)
            top_idx = np.argsort(syl_freqs)[-n_v:]
            s_mat_trimmed = s_mat[np.ix_(top_idx, top_idx)]
        else:
            s_mat_trimmed = s_mat

        dist = _optimal_mapping_distance(v_mat, s_mat_trimmed)

        # JSD
        v_flat = v_mat.flatten() + 1e-10
        s_flat = s_mat_trimmed.flatten() + 1e-10
        max_len = max(len(v_flat), len(s_flat))
        vp = np.zeros(max_len)
        sp = np.zeros(max_len)
        vp[:len(v_flat)] = v_flat
        sp[:len(s_flat)] = s_flat
        vp /= vp.sum()
        sp /= sp.sum()
        jsd = jensen_shannon_divergence(vp, sp)

        results.append(LanguageMatchResult(
            language=lang,
            optimal_distance=dist,
            jsd=jsd,
            n_ref_syllable_types=n_ref_types,
            selectivity=0.0,  # filled in after null baseline
        ))

    # Null baseline for selectivity
    rng = random.Random(seed)
    null_dists: Dict[str, List[float]] = defaultdict(list)
    for _ in range(20):
        shuffled = list(tokens)
        rng.shuffle(shuffled)
        sh_mat, _ = build_cv_bigram_matrix(shuffled, cv_labels, grid_for_maps)
        for r in results:
            ref_stats = get_reference_syllable_stats(
                r.language, corpus=reference_corpus, n_words=5000, seed=seed)
            s_mat, s_alph = ref_stats['syllable_bigrams']
            n_v = sh_mat.shape[0]
            if s_mat.shape[0] > n_v:
                syl_freqs = s_mat.sum(axis=1)
                top_idx = np.argsort(syl_freqs)[-n_v:]
                s_trimmed = s_mat[np.ix_(top_idx, top_idx)]
            else:
                s_trimmed = s_mat
            null_dists[r.language].append(
                _optimal_mapping_distance(sh_mat, s_trimmed))

    for r in results:
        null_mean = float(np.mean(null_dists[r.language]))
        r.selectivity = null_mean / r.optimal_distance if r.optimal_distance > 0 else 0

    results.sort(key=lambda r: r.optimal_distance)
    return results


# ---------------------------------------------------------------------------
# F.4: Pointwise Mutual Information Under Best Mapping
# ---------------------------------------------------------------------------

@dataclass
class PMIResult:
    """PMI comparison between Voynich and reference syllable sequences."""
    language: str
    mean_pmi_voynich: float
    mean_pmi_reference: float
    pmi_correlation: float
    pmi_correlation_p: float
    n_common_bigrams: int
    significant: bool


def compute_syllable_pmi(
    syllable_sequences: List[List[str]],
    top_n: int = 50,
) -> Dict[Tuple[str, str], float]:
    """Compute PMI for top-n most frequent syllable bigrams."""
    bigrams: Counter = Counter()
    unigrams: Counter = Counter()

    for seq in syllable_sequences:
        for k in range(len(seq) - 1):
            bigrams[(seq[k], seq[k + 1])] += 1
            unigrams[seq[k]] += 1

    total_bi = sum(bigrams.values())
    total_uni = sum(unigrams.values())

    if total_bi == 0 or total_uni == 0:
        return {}

    # Top-n bigrams
    top_bigrams = bigrams.most_common(top_n)

    pmi = {}
    for (a, b), count in top_bigrams:
        p_ab = count / total_bi
        p_a = unigrams[a] / total_uni
        p_b = unigrams.get(b, 1) / total_uni
        if p_a > 0 and p_b > 0 and p_ab > 0:
            pmi[(a, b)] = math.log2(p_ab / (p_a * p_b))

    return pmi


def pmi_comparison(
    tokens: List[str],
    cv_labels: Dict[str, CVLabel],
    best_language: str,
    reference_corpus: Optional[ReferenceCorpus] = None,
    grid: Optional[SyllabaryGrid] = None,
    seed: int = 42,
) -> PMIResult:
    """
    Compare PMI structure between Voynich CV-labeled text and reference.
    """
    if reference_corpus is None:
        reference_corpus = load_reference_corpus(verbose=False)

    onset_merge, nucleus_merge = None, None
    if grid is not None:
        onset_merge, nucleus_merge = _build_merge_maps(grid)

    # Voynich PMI (using CV labels)
    voynich_seqs = []
    for token in tokens:
        cv_seq = retranscribe_token(token, cv_labels, onset_merge, nucleus_merge)
        voynich_seqs.append(cv_seq)

    v_pmi = compute_syllable_pmi(voynich_seqs, top_n=50)

    # Reference PMI (using syllabified text)
    ref_text = get_reference_text(best_language, n_words=5000, seed=seed,
                                  corpus=reference_corpus)
    ref_syl_seqs = syllabify_latin_text(ref_text)
    r_pmi = compute_syllable_pmi(ref_syl_seqs, top_n=50)

    # We can't directly correlate different label sets, so compare PMI distributions
    v_values = np.array(list(v_pmi.values()))
    r_values = np.array(list(r_pmi.values()))

    # Pad to same length for correlation
    n = min(len(v_values), len(r_values))
    if n < 3:
        return PMIResult(
            language=best_language,
            mean_pmi_voynich=float(np.mean(v_values)) if len(v_values) > 0 else 0,
            mean_pmi_reference=float(np.mean(r_values)) if len(r_values) > 0 else 0,
            pmi_correlation=0.0,
            pmi_correlation_p=1.0,
            n_common_bigrams=n,
            significant=False,
        )

    # Sort by magnitude to align distributions
    v_sorted = np.sort(v_values)[-n:]
    r_sorted = np.sort(r_values)[-n:]

    r_corr, p_corr = pearson_correlation(v_sorted, r_sorted)

    # Null baseline
    rng = random.Random(seed)
    null_corrs = []
    for _ in range(100):
        rng.shuffle(v_sorted)
        r_null, _ = pearson_correlation(v_sorted, r_sorted)
        null_corrs.append(r_null)
    v_sorted = np.sort(v_values)[-n:]  # restore order

    significant = bool(r_corr > np.mean(null_corrs) + 2 * np.std(null_corrs))

    return PMIResult(
        language=best_language,
        mean_pmi_voynich=float(np.mean(v_values)),
        mean_pmi_reference=float(np.mean(r_values)),
        pmi_correlation=r_corr,
        pmi_correlation_p=p_corr,
        n_common_bigrams=n,
        significant=significant,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_syllable_matching() -> Dict:
    """Run all Workstream F tests and print/save results."""
    os.makedirs('results', exist_ok=True)

    print("=" * 70)
    print("WORKSTREAM F: SYLLABLE-LEVEL RETRANSCRIPTION AND MATCHING")
    print("=" * 70)

    # Load data
    corpus = load_corpus(verbose=False)
    tokens = corpus.get_tokens(paragraph_only=True)
    ref_corpus = load_reference_corpus(verbose=False)
    grid = build_grid_from_tokens(tokens)

    # F.1: CV Labeling
    print("\n--- F.1: Provisional CV Labeling ---")
    cv_labels = assign_cv_labels(grid, tokens)
    print(f"  Grid cells labeled: {len(cv_labels)}")
    print(f"  CV labels:")
    for key, label in sorted(cv_labels.items(),
                              key=lambda x: x[1].frequency, reverse=True):
        print(f"    {label.cv_label} ({key}): {label.frequency:,} occurrences, "
              f"glyphs: {label.glyphs}")

    with open('results/cv_labels.json', 'w') as f:
        json.dump({k: asdict(v) for k, v in cv_labels.items()}, f, indent=2)

    # F.2: Retranscription
    print("\n--- F.2: Full-Corpus Syllabic Retranscription ---")
    retrans = retranscribe_corpus(tokens, cv_labels, grid)
    print(f"  Tokens: {retrans.n_tokens}")
    print(f"  CV types: {retrans.n_cv_types}")
    print(f"  CV tokens: {retrans.n_cv_tokens}")
    print(f"  CV TTR: {retrans.cv_ttr:.4f}")
    print(f"  CV H1: {retrans.cv_h1:.4f}")
    print(f"  CV H2: {retrans.cv_h2:.4f}")
    print(f"  Mean CV/word: {retrans.mean_cv_per_word:.2f} "
          f"(std {retrans.std_cv_per_word:.2f})")
    print(f"  Ambiguity rate: {retrans.ambiguity_rate:.1%}")
    print(f"  Top 10 CV labels:")
    for label, count in retrans.top_20_cv_labels[:10]:
        print(f"    {label}: {count:,}")
    print(f"  Sample retranscriptions:")
    for orig, retrans_str in retrans.sample_retranscriptions[:5]:
        print(f"    {orig} -> {retrans_str}")

    with open('results/retranscription_stats.json', 'w') as f:
        json.dump(asdict(retrans), f, indent=2)

    # F.3: Language Matching
    print("\n--- F.3: Syllable Bigram Matching Against Languages ---")
    print("  Computing optimal mappings for each language...")
    lang_results = match_against_languages(tokens, cv_labels, ref_corpus)
    print(f"\n  {'Language':<15} {'Distance':>10} {'JSD':>8} "
          f"{'Selectivity':>12} {'Syl Types':>10}")
    print(f"  {'-' * 58}")
    for r in lang_results:
        print(f"  {r.language:<15} {r.optimal_distance:>10.4f} "
              f"{r.jsd:>8.4f} {r.selectivity:>11.2f}x "
              f"{r.n_ref_syllable_types:>10}")

    with open('results/syllable_language_ranking.json', 'w') as f:
        json.dump([asdict(r) for r in lang_results], f, indent=2)

    # F.4: PMI Comparison
    best_lang = lang_results[0].language if lang_results else 'latin'
    print(f"\n--- F.4: PMI Correlation with {best_lang} ---")
    pmi_result = pmi_comparison(tokens, cv_labels, best_lang, ref_corpus, grid)
    print(f"  Mean PMI (Voynich):    {pmi_result.mean_pmi_voynich:.4f}")
    print(f"  Mean PMI ({best_lang}): {pmi_result.mean_pmi_reference:.4f}")
    print(f"  PMI correlation:       {pmi_result.pmi_correlation:.4f} "
          f"(p={pmi_result.pmi_correlation_p:.4f})")
    print(f"  Common bigrams:        {pmi_result.n_common_bigrams}")
    print(f"  >> Significant: {pmi_result.significant}")

    with open('results/syllable_pmi.json', 'w') as f:
        json.dump(asdict(pmi_result), f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("WORKSTREAM F SUMMARY")
    print(f"  CV types: {retrans.n_cv_types}, "
          f"H1={retrans.cv_h1:.2f}, H2={retrans.cv_h2:.2f}")
    print(f"  Ambiguity rate: {retrans.ambiguity_rate:.1%}")
    if lang_results:
        print(f"  Best language: {lang_results[0].language} "
              f"(distance={lang_results[0].optimal_distance:.4f}, "
              f"selectivity={lang_results[0].selectivity:.2f}x)")
    print(f"  PMI correlation: {pmi_result.pmi_correlation:.4f} "
          f"(significant={pmi_result.significant})")
    print(f"{'=' * 70}")

    return {
        'cv_labels': {k: asdict(v) for k, v in cv_labels.items()},
        'retranscription': asdict(retrans),
        'language_ranking': [asdict(r) for r in lang_results],
        'pmi': asdict(pmi_result),
    }
