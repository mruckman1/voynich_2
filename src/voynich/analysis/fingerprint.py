"""
Information-Theoretic Fingerprinting (Approach 2)
==================================================
Compute a comprehensive entropy profile of the Voynich text and match it
against reference profiles for candidate language+encoding combinations.

Phases:
  2.1 — Voynich entropy profile (EntropyProfile dataclass)
  2.2 — Reference library construction (ReferenceLibrary class)
  2.3 — Profile matching (cosine similarity, confusion matrix)
  2.4 — Section-level differentiation
  2.5 — Discriminant validation (null text comparison)
"""

import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from voynich.core.corpus import VoynichCorpus, load_corpus, VOYNICH_SECTIONS
from voynich.core.stats import (
    compute_all_entropy, first_order_entropy, conditional_entropy,
    word_unigram_entropy, word_conditional_entropy,
    zipf_analysis, word_positional_entropy,
    bigram_transition_matrix,
    mutual_information_lag, intra_token_mi,
    token_length_entropy, type_token_ratio_at_n,
    cosine_similarity, jensen_shannon_divergence,
)
from voynich.core.ciphers import (
    ENCODING_SCHEMES, REFERENCE_LANGUAGES,
    generate_reference_text, apply_encoding,
)
from voynich.core.reference import load_reference_corpus, get_reference_text, ReferenceCorpus
from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Phase 2.1: Entropy Profile
# ---------------------------------------------------------------------------

@dataclass
class EntropyProfile:
    """Complete information-theoretic fingerprint of a text."""
    label: str

    # Character-level
    h1: float = 0.0               # Shannon entropy
    h2: float = 0.0               # Conditional entropy order 1
    h3: float = 0.0               # Conditional entropy order 2

    # Word-level
    word_h1: float = 0.0          # Word unigram entropy
    word_h2: float = 0.0          # Word bigram conditional entropy

    # Long-range mutual information
    mi_lags: Dict[int, float] = field(default_factory=dict)

    # Intra-token structure
    intra_mi: float = 0.0         # MI(first_char, last_char)

    # Word-length distribution
    word_length_entropy: float = 0.0
    mean_word_length: float = 0.0
    std_word_length: float = 0.0

    # Vocabulary growth
    ttr_curve: Dict[int, float] = field(default_factory=dict)

    # Positional entropy (positions 0-9)
    positional_entropy: List[float] = field(default_factory=list)

    # Zipf's law
    zipf_exponent: float = 0.0
    zipf_r_squared: float = 0.0

    # Corpus stats
    vocabulary_size: int = 0
    total_tokens: int = 0

    # Bigram matrix digest
    bigram_matrix_entropy: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert profile to a fixed-length numeric vector for comparison."""
        v = [
            self.h1, self.h2, self.h3,
            self.word_h1, self.word_h2,
            self.intra_mi,
            self.word_length_entropy,
            self.mean_word_length, self.std_word_length,
            self.zipf_exponent, self.zipf_r_squared,
            self.bigram_matrix_entropy,
        ]

        # MI lags 1-10
        for k in range(1, 11):
            v.append(self.mi_lags.get(k, 0.0))

        # Positional entropy positions 0-9
        for i in range(10):
            v.append(self.positional_entropy[i] if i < len(self.positional_entropy) else 0.0)

        # TTR at key sizes
        for n in [100, 500, 1000, 5000, 10000]:
            v.append(self.ttr_curve.get(n, 0.0))

        return np.array(v, dtype=float)

    @staticmethod
    def vector_labels() -> List[str]:
        """Human-readable labels for each vector dimension."""
        labels = [
            'H1_char', 'H2_char', 'H3_char',
            'H1_word', 'H2_word',
            'MI_intra_token',
            'H_word_length',
            'mean_word_length', 'std_word_length',
            'zipf_exponent', 'zipf_r_squared',
            'bigram_matrix_entropy',
        ]
        for k in range(1, 11):
            labels.append(f'MI_lag_{k}')
        for i in range(10):
            labels.append(f'positional_entropy_{i}')
        for n in [100, 500, 1000, 5000, 10000]:
            labels.append(f'TTR_at_{n}')
        return labels

    def to_dict(self) -> Dict:
        """Serialize to a plain dict for JSON output."""
        return asdict(self)


def compute_profile(text: str, tokens: List[str], label: str = "text") -> EntropyProfile:
    """Compute the full entropy profile from text and tokens."""

    # Character-level entropy
    ent = compute_all_entropy(text)

    # Word-level entropy
    w_h1 = word_unigram_entropy(tokens)
    w_h2 = word_conditional_entropy(tokens, order=1)

    # Mutual information at various lags
    mi = mutual_information_lag(tokens, max_lag=10)

    # Intra-token MI
    i_mi = intra_token_mi(tokens)

    # Word length stats
    wl_entropy = token_length_entropy(tokens)
    lengths = [len(t) for t in tokens]
    mean_wl = float(np.mean(lengths)) if lengths else 0.0
    std_wl = float(np.std(lengths)) if lengths else 0.0

    # TTR curve
    ttr = type_token_ratio_at_n(tokens)

    # Positional entropy
    pos_ent = word_positional_entropy(tokens)
    pos_list = [pos_ent.get(f'pos_{i}', 0.0) for i in range(10)]

    # Zipf's law
    zipf = zipf_analysis(tokens)

    # Bigram matrix entropy (entropy of the flattened probability matrix)
    bmat, _ = bigram_transition_matrix(text)
    flat = bmat.flatten()
    flat = flat[flat > 0]
    if len(flat) > 0:
        flat_norm = flat / flat.sum()
        bmat_entropy = float(-np.sum(flat_norm * np.log2(flat_norm)))
    else:
        bmat_entropy = 0.0

    return EntropyProfile(
        label=label,
        h1=ent['H1'], h2=ent['H2'], h3=ent['H3'],
        word_h1=w_h1, word_h2=w_h2,
        mi_lags=mi,
        intra_mi=i_mi,
        word_length_entropy=wl_entropy,
        mean_word_length=mean_wl,
        std_word_length=std_wl,
        ttr_curve=ttr,
        positional_entropy=pos_list,
        zipf_exponent=zipf.get('zipf_exponent', 0.0),
        zipf_r_squared=zipf.get('r_squared', 0.0),
        vocabulary_size=zipf.get('vocabulary_size', 0),
        total_tokens=len(tokens),
        bigram_matrix_entropy=bmat_entropy,
    )


def compute_voynich_profile(
    corpus: VoynichCorpus,
    section: Optional[str] = None,
    language: Optional[str] = None,
) -> EntropyProfile:
    """Compute the entropy profile for the Voynich text (or a filtered subset)."""
    text = corpus.get_text(section=section, language=language, paragraph_only=True)
    tokens = text.split()

    label_parts = ['voynich']
    if section:
        label_parts.append(section)
    if language:
        label_parts.append(f'lang_{language}')
    label = '_'.join(label_parts)

    return compute_profile(text, tokens, label=label)


# ---------------------------------------------------------------------------
# Phase 2.2: Reference Library
# ---------------------------------------------------------------------------

@dataclass
class ReferenceEntry:
    """Aggregated reference profile for one language+encoding combination."""
    language: str
    encoding: str
    mean_vector: np.ndarray
    std_vector: np.ndarray
    n_samples: int
    sample_profiles: List[EntropyProfile] = field(default_factory=list)


class ReferenceLibrary:
    """
    Builds and stores reference profiles for all language x encoding combos.
    """

    def __init__(self, n_samples: int = 50, n_words: int = 500, verbose: bool = True,
                 reference_corpus: Optional[ReferenceCorpus] = None):
        self.n_samples = n_samples
        self.n_words = n_words
        self.verbose = verbose
        self.reference_corpus = reference_corpus
        self.entries: Dict[Tuple[str, str], ReferenceEntry] = {}

    def build(self,
              languages: Optional[List[str]] = None,
              encodings: Optional[List[str]] = None):
        """Build reference profiles for all requested combinations."""
        if languages is None:
            languages = REFERENCE_LANGUAGES
        if encodings is None:
            encodings = list(ENCODING_SCHEMES.keys())

        total = len(languages) * len(encodings)
        count = 0

        for lang in languages:
            for enc in encodings:
                count += 1
                if self.verbose:
                    print(f"  [{count}/{total}] {lang} x {enc}...", end='', flush=True)

                vectors = []
                profiles = []

                for i in range(self.n_samples):
                    seed = i * 7 + hash(lang) % 10000 + hash(enc) % 10000
                    plaintext = get_reference_text(
                        lang, n_words=self.n_words, seed=seed,
                        corpus=self.reference_corpus,
                    )

                    encoded = apply_encoding(plaintext, enc, seed=seed)
                    if not encoded or len(encoded.split()) < 10:
                        continue

                    tokens = encoded.split()
                    profile = compute_profile(encoded, tokens,
                                              label=f'{lang}_{enc}_sample{i}')
                    profiles.append(profile)
                    vectors.append(profile.to_vector())

                if vectors:
                    arr = np.array(vectors)
                    entry = ReferenceEntry(
                        language=lang,
                        encoding=enc,
                        mean_vector=arr.mean(axis=0),
                        std_vector=arr.std(axis=0),
                        n_samples=len(vectors),
                    )
                    self.entries[(lang, enc)] = entry

                if self.verbose:
                    print(f" {len(vectors)} samples")

    def match(self, query: EntropyProfile, metric: str = 'cosine') -> List[Dict]:
        """
        Rank all reference profiles by similarity to the query.
        Returns sorted list of dicts: {language, encoding, similarity, distance}.
        """
        q_vec = query.to_vector()
        results = []

        for (lang, enc), entry in self.entries.items():
            ref_vec = entry.mean_vector

            if metric == 'cosine':
                sim = cosine_similarity(q_vec, ref_vec)
                dist = 1.0 - sim
            elif metric == 'euclidean':
                dist = float(np.linalg.norm(q_vec - ref_vec))
                sim = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            results.append({
                'language': lang,
                'encoding': enc,
                'similarity': round(sim, 6),
                'distance': round(dist, 6),
            })

        results.sort(key=lambda x: x['distance'])
        return results

    def confusion_matrix(self, metric: str = 'cosine') -> Tuple[np.ndarray, List[str]]:
        """
        Build pairwise distance matrix between all reference profiles.
        Returns (matrix, labels) where labels are 'language_encoding'.
        """
        keys = sorted(self.entries.keys())
        labels = [f'{lang}_{enc}' for lang, enc in keys]
        n = len(keys)
        mat = np.zeros((n, n))

        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                v1 = self.entries[k1].mean_vector
                v2 = self.entries[k2].mean_vector
                if metric == 'cosine':
                    mat[i, j] = 1.0 - cosine_similarity(v1, v2)
                elif metric == 'euclidean':
                    mat[i, j] = float(np.linalg.norm(v1 - v2))

        return mat, labels


# ---------------------------------------------------------------------------
# Phase 2.4: Section-Level Profiling
# ---------------------------------------------------------------------------

def compute_section_profiles(corpus: VoynichCorpus) -> Dict[str, EntropyProfile]:
    """Compute per-section entropy profiles."""
    profiles = {}
    for section in VOYNICH_SECTIONS:
        text = corpus.get_text(section=section, paragraph_only=True)
        tokens = text.split()
        if len(tokens) < 50:
            continue
        profiles[section] = compute_profile(text, tokens, label=f'voynich_{section}')
    return profiles


# ---------------------------------------------------------------------------
# Phase 2.5: Discriminant Validation
# ---------------------------------------------------------------------------

def generate_null_text(tokens: List[str], method: str = 'shuffle',
                       seed: int = 42) -> str:
    """
    Generate null text for validation.

    Methods:
        'shuffle' — shuffle characters within each token
        'random'  — frequency-matched random character generation
        'markov'  — bigram-level Markov chain generation
    """
    rng = random.Random(seed)

    if method == 'shuffle':
        scrambled = []
        for t in tokens:
            chars = list(t)
            rng.shuffle(chars)
            scrambled.append(''.join(chars))
        return ' '.join(scrambled)

    elif method == 'random':
        # Build character frequency distribution
        all_chars = ''.join(tokens)
        char_counts = {}
        for c in all_chars:
            char_counts[c] = char_counts.get(c, 0) + 1
        chars = list(char_counts.keys())
        weights = [char_counts[c] for c in chars]

        random_tokens = []
        for t in tokens:
            length = len(t)
            new_chars = rng.choices(chars, weights=weights, k=length)
            random_tokens.append(''.join(new_chars))
        return ' '.join(random_tokens)

    elif method == 'markov':
        # Build character bigram model from tokens
        all_text = ' '.join(tokens)
        chars = [c for c in all_text if c != ' ']
        bigrams = {}
        for i in range(len(chars) - 1):
            c1, c2 = chars[i], chars[i + 1]
            if c1 not in bigrams:
                bigrams[c1] = {}
            bigrams[c1][c2] = bigrams[c1].get(c2, 0) + 1

        # Generate tokens matching original lengths
        markov_tokens = []
        all_chars_list = list(set(chars))
        for t in tokens:
            length = len(t)
            current = rng.choice(all_chars_list)
            generated = [current]
            for _ in range(length - 1):
                if current in bigrams:
                    nexts = list(bigrams[current].keys())
                    wts = list(bigrams[current].values())
                    current = rng.choices(nexts, weights=wts, k=1)[0]
                else:
                    current = rng.choice(all_chars_list)
                generated.append(current)
            markov_tokens.append(''.join(generated))
        return ' '.join(markov_tokens)

    else:
        raise ValueError(f"Unknown null method: {method}")


def discriminant_validation(
    voynich_profile: EntropyProfile,
    voynich_tokens: List[str],
    library: ReferenceLibrary,
    n_trials: int = 20,
) -> Dict:
    """
    Test whether profile matching discriminates real Voynich from null text.

    For each null method, generate trials, compute profiles, match against
    the reference library, and compare the quality of match to the real
    Voynich profile's match.
    """
    # Real match
    real_match = library.match(voynich_profile)
    real_best = real_match[0] if real_match else None

    results = {'real': {'best_match': real_best, 'top_5': real_match[:5]}}

    for method in ['shuffle', 'random', 'markov']:
        null_distances = []
        null_best_matches = []

        for trial in range(n_trials):
            null_text = generate_null_text(voynich_tokens, method=method,
                                           seed=42 + trial)
            null_tokens = null_text.split()
            null_profile = compute_profile(null_text, null_tokens,
                                           label=f'null_{method}_{trial}')
            null_match = library.match(null_profile)
            if null_match:
                null_distances.append(null_match[0]['distance'])
                null_best_matches.append(null_match[0])

        results[method] = {
            'mean_best_distance': float(np.mean(null_distances)) if null_distances else 0.0,
            'std_best_distance': float(np.std(null_distances)) if null_distances else 0.0,
            'sample_best_matches': null_best_matches[:3],
        }

    # Discrimination score: how much better does real Voynich match vs null?
    if real_best:
        for method in ['shuffle', 'random', 'markov']:
            null_mean = results[method]['mean_best_distance']
            if null_mean > 0:
                results[method]['discrimination_ratio'] = null_mean / max(real_best['distance'], 1e-10)
            else:
                results[method]['discrimination_ratio'] = 0.0

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_fingerprint_analysis():
    """Run the full information-theoretic fingerprinting pipeline."""
    print("=" * 70)
    print("APPROACH 2: INFORMATION-THEORETIC FINGERPRINTING")
    print("=" * 70)

    # --- Phase 2.1: Voynich Entropy Profile ---
    print("\n--- Phase 2.1: Computing Voynich Entropy Profile ---")
    corpus = load_corpus(verbose=False)
    voynich = compute_voynich_profile(corpus)

    print(f"\n  Full Corpus Profile ({voynich.total_tokens} tokens, "
          f"{voynich.vocabulary_size} types):")
    print(f"    H1 (char entropy):     {voynich.h1:.4f}")
    print(f"    H2 (char|prev):        {voynich.h2:.4f}")
    print(f"    H3 (char|prev2):       {voynich.h3:.4f}")
    print(f"    H1 (word unigram):     {voynich.word_h1:.4f}")
    print(f"    H2 (word bigram):      {voynich.word_h2:.4f}")
    print(f"    MI(first,last char):   {voynich.intra_mi:.4f}")
    print(f"    H(word length):        {voynich.word_length_entropy:.4f}")
    print(f"    Mean word length:      {voynich.mean_word_length:.2f}")
    print(f"    Zipf exponent:         {voynich.zipf_exponent:.4f}")
    print(f"    Zipf R^2:              {voynich.zipf_r_squared:.4f}")
    print(f"    Bigram matrix entropy: {voynich.bigram_matrix_entropy:.4f}")

    # MI decay
    print(f"\n    MI decay (word-level):")
    for k in [1, 2, 3, 5, 10]:
        if k in voynich.mi_lags:
            print(f"      lag {k:2d}: {voynich.mi_lags[k]:.4f}")

    # TTR curve
    print(f"\n    TTR curve:")
    for n, ttr in sorted(voynich.ttr_curve.items()):
        print(f"      N={n:5d}: {ttr:.4f}")

    # --- Phase 2.4: Section-Level Profiles ---
    print("\n--- Phase 2.4: Section-Level Profiles ---")
    section_profiles = compute_section_profiles(corpus)
    for section, prof in sorted(section_profiles.items()):
        print(f"\n  {section} ({prof.total_tokens} tokens):")
        print(f"    H1={prof.h1:.3f}  H2={prof.h2:.3f}  H3={prof.h3:.3f}  "
              f"Zipf={prof.zipf_exponent:.3f}  MWL={prof.mean_word_length:.2f}")

    # --- Phase 2.2: Reference Library ---
    print("\n--- Phase 2.2: Building Reference Library ---")

    # Load real reference corpus if available
    try:
        ref_corpus = load_reference_corpus(verbose=True)
        real_langs = ref_corpus.languages
        print(f"\n  Real corpus loaded for: {', '.join(real_langs)}")
        for lang in real_langs:
            total = sum(t.token_count for t in ref_corpus.get_texts(lang))
            print(f"    {lang}: {total:,} tokens")
    except FileNotFoundError:
        ref_corpus = None
        print("  No real reference corpus found; using synthetic text only.")

    print()
    library = ReferenceLibrary(
        n_samples=30, n_words=500, verbose=True,
        reference_corpus=ref_corpus,
    )
    library.build()

    # --- Phase 2.3: Profile Matching ---
    print("\n--- Phase 2.3: Profile Matching ---")
    matches = library.match(voynich, metric='cosine')

    print(f"\n  Top 15 matches (cosine similarity):")
    print(f"  {'Rank':<6} {'Language':<12} {'Encoding':<22} {'Similarity':>12} {'Distance':>10}")
    print(f"  {'-'*62}")
    for i, m in enumerate(matches[:15]):
        print(f"  {i+1:<6} {m['language']:<12} {m['encoding']:<22} "
              f"{m['similarity']:>12.6f} {m['distance']:>10.6f}")

    # Best match assessment
    if len(matches) >= 2:
        best = matches[0]
        second = matches[1]
        gap = second['distance'] - best['distance']
        print(f"\n  Best match: {best['language']} + {best['encoding']} "
              f"(similarity={best['similarity']:.4f})")
        print(f"  Gap to second: {gap:.6f}")
        if best['similarity'] > 0.9 and second['similarity'] < 0.8:
            print(f"  ** STRONG MATCH — clear winner")
        elif best['similarity'] > 0.85:
            print(f"  ** MODERATE MATCH — top candidate but not decisive")
        else:
            print(f"  ** WEAK MATCH — no clear language+encoding winner")

    # Section-level matching
    print("\n  Section-level best matches:")
    for section, prof in sorted(section_profiles.items()):
        sec_matches = library.match(prof, metric='cosine')
        if sec_matches:
            best = sec_matches[0]
            print(f"    {section:<18} -> {best['language']:<10} + "
                  f"{best['encoding']:<20} (sim={best['similarity']:.4f})")

    # --- Phase 2.5: Discriminant Validation ---
    print("\n--- Phase 2.5: Discriminant Validation ---")
    voynich_tokens = corpus.get_tokens(paragraph_only=True)
    disc = discriminant_validation(voynich, voynich_tokens, library, n_trials=10)

    real_dist = disc['real']['best_match']['distance'] if disc['real']['best_match'] else float('inf')
    print(f"\n  Real Voynich best-match distance: {real_dist:.6f}")
    for method in ['shuffle', 'random', 'markov']:
        md = disc[method]
        print(f"  {method:<10} null mean distance:  {md['mean_best_distance']:.6f} "
              f"(+/- {md['std_best_distance']:.6f})  "
              f"discrimination={md.get('discrimination_ratio', 0):.2f}x")

    # Verdict
    print("\n  Discrimination verdict:")
    all_discriminating = True
    for method in ['shuffle', 'random', 'markov']:
        ratio = disc[method].get('discrimination_ratio', 0)
        if ratio < 1.5:
            print(f"    {method}: FAIL (ratio {ratio:.2f} < 1.5x)")
            all_discriminating = False
        else:
            print(f"    {method}: PASS (ratio {ratio:.2f}x)")

    if all_discriminating:
        print("  ** PROFILE MATCHING IS DISCRIMINATIVE — real Voynich matches "
              "significantly better than null text")
    else:
        print("  ** PROFILE MATCHING IS NOT YET DISCRIMINATIVE — some null "
              "methods match nearly as well")

    # --- Save results ---
    rd = _results_dir()

    # Save Voynich profile
    with open(os.path.join(rd, 'voynich_profile.json'), 'w') as f:
        json.dump(voynich.to_dict(), f, indent=2, default=str)

    # Save section profiles
    section_data = {s: p.to_dict() for s, p in section_profiles.items()}
    with open(os.path.join(rd, 'section_profiles.json'), 'w') as f:
        json.dump(section_data, f, indent=2, default=str)

    # Save match rankings
    with open(os.path.join(rd, 'match_rankings.json'), 'w') as f:
        json.dump(matches, f, indent=2)

    # Save discriminant results
    disc_serializable = {}
    for k, v in disc.items():
        if isinstance(v, dict):
            disc_serializable[k] = {
                kk: (vv if not isinstance(vv, np.floating) else float(vv))
                for kk, vv in v.items()
            }
        else:
            disc_serializable[k] = v
    with open(os.path.join(rd, 'discriminant_validation.json'), 'w') as f:
        json.dump(disc_serializable, f, indent=2, default=str)

    print(f"\n  Results saved to {rd}/")
    print("=" * 70)

    return {
        'voynich_profile': voynich,
        'section_profiles': section_profiles,
        'matches': matches,
        'discriminant': disc,
    }


if __name__ == '__main__':
    run_fingerprint_analysis()
