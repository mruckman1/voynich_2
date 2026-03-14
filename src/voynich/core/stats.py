"""
Statistical Analysis Module
============================
Character/word entropy, Zipf's law, bigram transition matrices,
positional glyph analysis, mutual information, and profile comparison.

Carried over from voynich/modules/statistical_analysis.py with extensions:
- mutual_information_lag(): long-range token MI
- intra_token_mi(): MI(first_char, last_char)
- token_length_entropy(): H(word_length)
- type_token_ratio_at_n(): TTR at multiple corpus sizes
"""

import math
import random
import numpy as np
from collections import Counter, defaultdict
from typing import Any, List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Character-Level Entropy
# ---------------------------------------------------------------------------

def char_frequencies(text: str) -> Dict[str, float]:
    """Compute character frequency distribution (ignoring spaces)."""
    chars = [c for c in text if c != ' ']
    total = len(chars)
    if total == 0:
        return {}
    counts = Counter(chars)
    return {c: n / total for c, n in counts.items()}


def first_order_entropy(text: str) -> float:
    """H1: Shannon entropy of individual characters."""
    freqs = char_frequencies(text)
    if not freqs:
        return 0.0
    return -sum(p * math.log2(p) for p in freqs.values() if p > 0)


def conditional_entropy(text: str, order: int = 2) -> float:
    """
    Compute conditional character entropy of given order.
    H2 = conditional entropy given 1 preceding character (order=1)
    H3 = conditional entropy given 2 preceding characters (order=2)
    """
    chars = [c for c in text if c != ' ']
    n = len(chars)
    if n <= order:
        return 0.0

    ngram_counts = Counter()
    context_counts = Counter()

    for i in range(n - order):
        context = tuple(chars[i:i + order])
        ngram = tuple(chars[i:i + order + 1])
        context_counts[context] += 1
        ngram_counts[ngram] += 1

    total = sum(ngram_counts.values())
    total_ctx = sum(context_counts.values())

    if total == 0 or total_ctx == 0:
        return 0.0

    h_joint = -sum((c / total) * math.log2(c / total)
                   for c in ngram_counts.values() if c > 0)
    h_ctx = -sum((c / total_ctx) * math.log2(c / total_ctx)
                 for c in context_counts.values() if c > 0)

    return h_joint - h_ctx


def compute_all_entropy(text: str) -> Dict[str, float]:
    """Compute H1, H2, H3 for a text."""
    return {
        'H1': first_order_entropy(text),
        'H2': conditional_entropy(text, order=1),
        'H3': conditional_entropy(text, order=2),
    }


# ---------------------------------------------------------------------------
# Word-Level Entropy
# ---------------------------------------------------------------------------

def word_conditional_entropy(tokens: List[str], order: int = 1) -> float:
    """
    Compute word-level conditional entropy.
    H(W_n | W_{n-1}) for order=1, etc.
    """
    n = len(tokens)
    if n <= order:
        return 0.0

    ngram_counts: Counter = Counter()
    context_counts: Counter = Counter()

    for i in range(n - order):
        context = tuple(tokens[i:i + order])
        ngram = tuple(tokens[i:i + order + 1])
        context_counts[context] += 1
        ngram_counts[ngram] += 1

    total = sum(ngram_counts.values())
    total_ctx = sum(context_counts.values())

    if total == 0 or total_ctx == 0:
        return 0.0

    h_joint = -sum((c / total) * math.log2(c / total)
                   for c in ngram_counts.values() if c > 0)
    h_ctx = -sum((c / total_ctx) * math.log2(c / total_ctx)
                 for c in context_counts.values() if c > 0)

    return h_joint - h_ctx


def word_unigram_entropy(tokens: List[str]) -> float:
    """H(W): Shannon entropy of word unigram distribution."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c / total) * math.log2(c / total)
                for c in counts.values() if c > 0)


# ---------------------------------------------------------------------------
# Zipf's Law
# ---------------------------------------------------------------------------

def zipf_analysis(tokens: List[str]) -> Dict[str, Any]:
    """
    Analyze Zipf's law conformance.
    Returns rank-frequency data and R^2 fit.
    """
    counts = Counter(tokens)
    ranked = sorted(counts.values(), reverse=True)
    n = len(ranked)
    if n < 2:
        return {'r_squared': 0.0, 'ranks': [], 'frequencies': []}

    ranks = np.arange(1, n + 1, dtype=float)
    freqs = np.array(ranked, dtype=float)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs + 1e-10)

    A = np.vstack([log_ranks, np.ones(n)]).T
    try:
        result = np.linalg.lstsq(A, log_freqs, rcond=None)
        slope, intercept = result[0]
    except np.linalg.LinAlgError:
        slope, intercept = -1.0, 0.0

    predicted = slope * log_ranks + intercept
    ss_res = np.sum((log_freqs - predicted) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'zipf_exponent': -slope,
        'r_squared': r_squared,
        'ranks': ranks.tolist(),
        'frequencies': freqs.tolist(),
        'top_20': counts.most_common(20),
        'vocabulary_size': n,
        'total_tokens': len(tokens),
        'type_token_ratio': n / len(tokens) if tokens else 0,
    }


# ---------------------------------------------------------------------------
# Bigram Transition Matrices
# ---------------------------------------------------------------------------

def bigram_transition_matrix(text: str) -> Tuple[np.ndarray, List[str]]:
    """
    Build a character bigram transition probability matrix.
    Returns: (matrix, alphabet) where matrix[i][j] = P(char_j | char_i)
    """
    chars = [c for c in text if c != ' ']
    alphabet = sorted(set(chars))
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    n = len(alphabet)

    counts = np.zeros((n, n), dtype=float)
    for i in range(len(chars) - 1):
        c1, c2 = chars[i], chars[i + 1]
        counts[char_to_idx[c1]][char_to_idx[c2]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = counts / row_sums

    return matrix, alphabet


def word_transition_matrix(tokens: list, top_n: int = None) -> Tuple[np.ndarray, List[str]]:
    """
    Build a word-level bigram transition probability matrix.
    """
    freqs = Counter(tokens)
    if top_n:
        vocab = [w for w, _ in freqs.most_common(top_n)]
    else:
        vocab = sorted(set(tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    counts = np.zeros((n, n), dtype=float)
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if w1 in word_to_idx and w2 in word_to_idx:
            counts[word_to_idx[w1]][word_to_idx[w2]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums, vocab


def compare_bigram_matrices(mat_a: np.ndarray, mat_b: np.ndarray,
                            alph_a: List[str], alph_b: List[str]) -> float:
    """
    Compute Jensen-Shannon divergence between two bigram matrices.
    Only compares characters present in both alphabets.
    Lower value = more similar.
    """
    common = sorted(set(alph_a) & set(alph_b))
    if len(common) < 2:
        return float('inf')

    idx_a = [alph_a.index(c) for c in common]
    idx_b = [alph_b.index(c) for c in common]

    sub_a = mat_a[np.ix_(idx_a, idx_a)]
    sub_b = mat_b[np.ix_(idx_b, idx_b)]

    p = sub_a.flatten() + 1e-10
    q = sub_b.flatten() + 1e-10
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------------------------------
# Positional Analysis
# ---------------------------------------------------------------------------

def positional_glyph_distribution(tokens: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Analyze character positions within words.
    Returns: {char: {'initial': n, 'medial': n, 'final': n, 'singleton': n}}
    """
    positions = defaultdict(lambda: {'initial': 0, 'medial': 0, 'final': 0, 'singleton': 0})

    for word in tokens:
        if len(word) == 1:
            positions[word[0]]['singleton'] += 1
        elif len(word) >= 2:
            positions[word[0]]['initial'] += 1
            positions[word[-1]]['final'] += 1
            for c in word[1:-1]:
                positions[c]['medial'] += 1

    return dict(positions)


def classify_glyphs_by_position(tokens: List[str],
                                threshold: float = 0.7) -> Dict[str, str]:
    """
    Classify each glyph as PREFIX (P), SUFFIX (S), MEDIAL (M), or ANY (A)
    based on positional dominance.
    """
    dist = positional_glyph_distribution(tokens)
    classifications = {}

    for char, pos_counts in dist.items():
        total = sum(pos_counts.values())
        if total == 0:
            classifications[char] = 'A'
            continue

        ratios = {k: v / total for k, v in pos_counts.items()}

        if ratios['initial'] >= threshold:
            classifications[char] = 'P'
        elif ratios['final'] >= threshold:
            classifications[char] = 'S'
        elif ratios['medial'] >= threshold:
            classifications[char] = 'M'
        else:
            classifications[char] = 'A'

    return classifications


def word_positional_entropy(tokens: List[str]) -> Dict[str, float]:
    """
    Compute entropy at each character position within words.
    Groups words by length, then computes entropy at each position.
    Returns: {'pos_0': H, 'pos_1': H, ...}
    """
    by_length = defaultdict(list)
    for t in tokens:
        by_length[len(t)].append(t)

    position_entropies = {}
    for pos in range(10):
        chars_at_pos = []
        for length, words in by_length.items():
            if length > pos:
                chars_at_pos.extend(w[pos] for w in words)
        if chars_at_pos:
            counts = Counter(chars_at_pos)
            total = len(chars_at_pos)
            h = -sum((c / total) * math.log2(c / total)
                     for c in counts.values() if c > 0)
            position_entropies[f'pos_{pos}'] = round(h, 4)

    return position_entropies


# ---------------------------------------------------------------------------
# Statistical Profile & Distance
# ---------------------------------------------------------------------------

def full_statistical_profile(text: str, label: str = "text") -> Dict:
    """
    Compute a comprehensive statistical profile of a text.
    This is the core comparison fingerprint.
    """
    tokens = text.split()

    profile = {
        'label': label,
        'char_count': len(text.replace(' ', '')),
        'token_count': len(tokens),
        'entropy': compute_all_entropy(text),
        'zipf': zipf_analysis(tokens),
        'positional_entropy': word_positional_entropy(tokens),
        'glyph_classes': classify_glyphs_by_position(tokens),
        'positional_distribution': positional_glyph_distribution(tokens),
    }

    lengths = [len(t) for t in tokens]
    if lengths:
        profile['mean_word_length'] = np.mean(lengths)
        profile['std_word_length'] = np.std(lengths)
        profile['word_length_dist'] = dict(Counter(lengths))

    return profile


def profile_distance(prof_a: Dict, prof_b: Dict) -> float:
    """
    Compute a composite distance metric between two statistical profiles.
    Lower = more similar.
    """
    distance = 0.0
    weights = {'H1': 1.0, 'H2': 3.0, 'H3': 3.0}

    for key, w in weights.items():
        ea = prof_a.get('entropy', {}).get(key, 0)
        eb = prof_b.get('entropy', {}).get(key, 0)
        distance += w * (ea - eb) ** 2

    za = prof_a.get('zipf', {}).get('zipf_exponent', 1.0)
    zb = prof_b.get('zipf', {}).get('zipf_exponent', 1.0)
    distance += 2.0 * (za - zb) ** 2

    mla = prof_a.get('mean_word_length', 4.0)
    mlb = prof_b.get('mean_word_length', 4.0)
    distance += 1.0 * (mla - mlb) ** 2

    ttr_a = prof_a.get('zipf', {}).get('type_token_ratio', 0.5)
    ttr_b = prof_b.get('zipf', {}).get('type_token_ratio', 0.5)
    distance += 1.5 * (ttr_a - ttr_b) ** 2

    pe_a = prof_a.get('positional_entropy', {})
    pe_b = prof_b.get('positional_entropy', {})
    common_pos = set(pe_a.keys()) & set(pe_b.keys())
    if common_pos:
        pos_dist = sum((pe_a[k] - pe_b[k]) ** 2 for k in common_pos)
        distance += 2.0 * pos_dist / len(common_pos)

    return math.sqrt(distance)


def word_length_distribution(tokens: List[str]) -> Dict:
    """Compute word-length distribution statistics."""
    lengths = [len(t) for t in tokens if t]
    if not lengths:
        return {'lengths': [], 'mean': 0, 'std': 0, 'histogram': {}}
    arr = np.array(lengths)
    hist = Counter(lengths)
    return {
        'lengths': arr,
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'histogram': {int(k): int(v) for k, v in sorted(hist.items())},
        'total': len(lengths),
    }


# ---------------------------------------------------------------------------
# NEW: Extended Metrics for Information-Theoretic Fingerprinting
# ---------------------------------------------------------------------------

def mutual_information_lag(tokens: List[str], max_lag: int = 10) -> Dict[int, float]:
    """
    Compute MI(token_i, token_{i+k}) for k=1..max_lag.
    Measures long-range word-level sequential dependence.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    n = len(tokens)
    if n < max_lag + 1:
        return {}

    results = {}
    for lag in range(1, max_lag + 1):
        # Count joint and marginal frequencies
        joint_counts = Counter()
        x_counts = Counter()
        y_counts = Counter()

        for i in range(n - lag):
            x, y = tokens[i], tokens[i + lag]
            joint_counts[(x, y)] += 1
            x_counts[x] += 1
            y_counts[y] += 1

        total = n - lag
        if total == 0:
            results[lag] = 0.0
            continue

        # MI = sum p(x,y) * log2(p(x,y) / (p(x) * p(y)))
        mi = 0.0
        for (x, y), count in joint_counts.items():
            p_xy = count / total
            p_x = x_counts[x] / total
            p_y = y_counts[y] / total
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))

        results[lag] = mi

    return results


def intra_token_mi(tokens: List[str]) -> float:
    """
    Compute MI(first_char, last_char) within tokens.
    Measures intra-word structural coupling.
    Tokens of length 1 are excluded (first=last is trivial).
    """
    pairs = [(t[0], t[-1]) for t in tokens if len(t) >= 2]
    if not pairs:
        return 0.0

    total = len(pairs)
    joint_counts = Counter(pairs)
    first_counts = Counter(p[0] for p in pairs)
    last_counts = Counter(p[1] for p in pairs)

    mi = 0.0
    for (f, l), count in joint_counts.items():
        p_fl = count / total
        p_f = first_counts[f] / total
        p_l = last_counts[l] / total
        if p_fl > 0 and p_f > 0 and p_l > 0:
            mi += p_fl * math.log2(p_fl / (p_f * p_l))

    return mi


def token_length_entropy(tokens: List[str]) -> float:
    """
    H(word_length): entropy of the word-length distribution.
    Measures how regular/irregular word lengths are.
    """
    if not tokens:
        return 0.0
    lengths = [len(t) for t in tokens]
    counts = Counter(lengths)
    total = len(lengths)
    return -sum((c / total) * math.log2(c / total)
                for c in counts.values() if c > 0)


def type_token_ratio_at_n(tokens: List[str],
                          n_values: List[int] = None) -> Dict[int, float]:
    """
    Compute type-token ratio at different corpus sizes.
    Measures vocabulary growth curve (Heaps' law).
    """
    if n_values is None:
        n_values = [100, 500, 1000, 5000, 10000]

    results = {}
    for n in n_values:
        if n > len(tokens):
            break
        subset = tokens[:n]
        results[n] = len(set(subset)) / n

    return results


# ---------------------------------------------------------------------------
# Cosine Similarity / JSD for Profile Comparison
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two probability distributions.
    Returns value in [0, 1] (using log base 2).
    """
    p = np.asarray(p, dtype=float) + 1e-10
    q = np.asarray(q, dtype=float) + 1e-10
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    return float(0.5 * (kl_pm + kl_qm))


# ---------------------------------------------------------------------------
# Latin Syllabifier
# ---------------------------------------------------------------------------

LATIN_VOWELS = set('aeiouy')
LATIN_CONSONANTS = set('bcdfghjklmnpqrstvwxz')
LATIN_DIGRAPHS = ('qu', 'ph', 'th', 'ch', 'rh', 'gn')
LATIN_DIPHTHONGS = ('ae', 'oe', 'au', 'eu')
MUTA_CUM_LIQUIDA = frozenset({
    'pr', 'br', 'tr', 'dr', 'cr', 'gr',
    'pl', 'bl', 'tl', 'dl', 'cl', 'gl',
    'fr', 'fl',
})


def _tokenize_latin(word: str) -> List[str]:
    """
    Split a Latin word into phonological units (letters/digraphs/diphthongs).
    Each unit is tagged as 'V' (vowel) or 'C' (consonant).
    Returns list of (unit_string, type) tuples.
    """
    word = word.lower().strip()
    units: List[Tuple[str, str]] = []
    i = 0
    while i < len(word):
        if not word[i].isalpha():
            i += 1
            continue
        # Try diphthongs first (vowel pairs)
        if i + 1 < len(word):
            pair = word[i:i + 2]
            if pair in LATIN_DIPHTHONGS:
                units.append((pair, 'V'))
                i += 2
                continue
            # Try consonant digraphs
            if pair in LATIN_DIGRAPHS:
                units.append((pair, 'C'))
                i += 2
                continue
        # Single character
        c = word[i]
        if c in LATIN_VOWELS:
            units.append((c, 'V'))
        elif c in LATIN_CONSONANTS:
            units.append((c, 'C'))
        else:
            units.append((c, 'C'))  # treat unknown as consonant
        i += 1
    return units


def syllabify_latin(word: str) -> List[str]:
    """
    Rule-based Latin syllabification.

    Rules (in priority order):
    1. Treat digraphs (qu, ph, th, ch, rh, gn) as single consonants
    2. Treat diphthongs (ae, oe, au, eu) as single vowels
    3. V.CV: single consonant between vowels goes to next syllable
    4. VC.CV: two consonants split, unless muta cum liquida
    5. VCC.CV or VC.CCV for 3+ consonants

    Returns list of syllable strings. Returns [word] if no vowels found.
    """
    units = _tokenize_latin(word)
    if not units:
        return [word] if word else []

    # Check if there are any vowels
    vowel_indices = [i for i, (_, t) in enumerate(units) if t == 'V']
    if not vowel_indices:
        return [''.join(u for u, _ in units)]

    # Build syllables by finding consonant clusters between vowels
    # and deciding where to split them
    syllables: List[List[str]] = [[]]

    i = 0
    while i < len(units):
        unit_str, unit_type = units[i]

        if unit_type == 'V':
            # Add vowel to current syllable
            syllables[-1].append(unit_str)

            # Look ahead: collect consonants until next vowel
            consonants: List[str] = []
            j = i + 1
            while j < len(units) and units[j][1] == 'C':
                consonants.append(units[j][0])
                j += 1

            if j >= len(units):
                # No more vowels — remaining consonants are coda
                syllables[-1].extend(consonants)
                i = j
                continue

            # We have consonants between two vowels — decide split point
            n_cons = len(consonants)
            if n_cons == 0:
                # V.V — start new syllable at next vowel
                syllables.append([])
                i = j
                continue
            elif n_cons == 1:
                # V.CV — consonant goes to next syllable
                syllables.append([consonants[0]])
                i = j
                continue
            elif n_cons == 2:
                pair = consonants[0] + consonants[1]
                if pair in MUTA_CUM_LIQUIDA:
                    # V.CCV — cluster stays together as onset
                    syllables.append([consonants[0], consonants[1]])
                else:
                    # VC.CV — split between consonants
                    syllables[-1].append(consonants[0])
                    syllables.append([consonants[1]])
                i = j
                continue
            else:
                # 3+ consonants: try to give last 2 to next syllable
                # if they form a valid onset cluster
                last_two = consonants[-2] + consonants[-1]
                if last_two in MUTA_CUM_LIQUIDA:
                    # VCC...C.CCV
                    syllables[-1].extend(consonants[:-2])
                    syllables.append(list(consonants[-2:]))
                else:
                    # VCC...CC.CV — only last consonant to next syllable
                    syllables[-1].extend(consonants[:-1])
                    syllables.append([consonants[-1]])
                i = j
                continue
        else:
            # Initial consonant(s) before first vowel
            syllables[-1].append(unit_str)
            i += 1

    return [''.join(parts) for parts in syllables if parts]


def syllabify_latin_text(text: str) -> List[List[str]]:
    """Syllabify all words in a Latin text. Returns list of syllable lists."""
    words = text.split()
    return [syllabify_latin(w) for w in words if w]


def syllable_count_distribution(text: str) -> Dict[int, int]:
    """Distribution of syllable counts per word in Latin text."""
    syllabified = syllabify_latin_text(text)
    counts = Counter(len(s) for s in syllabified)
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# DTW (Dynamic Time Warping) Distance
# ---------------------------------------------------------------------------

def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Compute DTW distance between two 1D sequences.
    Uses O(n*m) dynamic programming. Pure numpy implementation.
    """
    a = np.asarray(seq_a, dtype=float)
    b = np.asarray(seq_b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float('inf')

    dtw = np.full((n + 1, m + 1), float('inf'))
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


# ---------------------------------------------------------------------------
# Frobenius Norm (Matrix Distance)
# ---------------------------------------------------------------------------

def frobenius_distance(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
    """
    Frobenius norm of (A - B). Pads smaller matrix with zeros if sizes differ.
    """
    a = np.asarray(mat_a, dtype=float)
    b = np.asarray(mat_b, dtype=float)

    max_rows = max(a.shape[0], b.shape[0])
    max_cols = max(a.shape[1], b.shape[1])

    padded_a = np.zeros((max_rows, max_cols))
    padded_b = np.zeros((max_rows, max_cols))
    padded_a[:a.shape[0], :a.shape[1]] = a
    padded_b[:b.shape[0], :b.shape[1]] = b

    return float(np.linalg.norm(padded_a - padded_b))


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Generic bootstrap confidence interval.

    Args:
        data: 1D array of observations
        statistic_fn: function that takes an array and returns a scalar
        n_bootstrap: number of bootstrap resamples
        ci_level: confidence level (e.g., 0.95 for 95% CI)
        seed: random seed for reproducibility

    Returns:
        (point_estimate, lower_ci, upper_ci)
    """
    data = np.asarray(data)
    point = float(statistic_fn(data))

    rng = np.random.RandomState(seed)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.randint(0, len(data), size=len(data))]
        boot_stats[i] = statistic_fn(sample)

    alpha = (1 - ci_level) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))

    return point, lo, hi


# ---------------------------------------------------------------------------
# Effect Size and Statistical Testing
# ---------------------------------------------------------------------------

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Cohen's d effect size between two groups.
    d = (mean_a - mean_b) / pooled_std
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def log_bayes_factor(
    data: np.ndarray,
    null_mean: float,
    alt_mean: float,
    shared_std: float,
) -> float:
    """
    Approximate log Bayes factor (BF10) using Savage-Dickey density ratio.
    Positive = evidence for alternative; negative = evidence for null.
    """
    data = np.asarray(data, dtype=float)
    if shared_std <= 0 or len(data) == 0:
        return 0.0

    obs_mean = np.mean(data)
    obs_se = shared_std / math.sqrt(len(data))

    # Log-likelihood of observed mean under null vs alternative
    def _log_normal_density(x: float, mu: float, sigma: float) -> float:
        if sigma <= 0:
            return -float('inf')
        return -0.5 * math.log(2 * math.pi * sigma ** 2) - (x - mu) ** 2 / (2 * sigma ** 2)

    ll_null = _log_normal_density(obs_mean, null_mean, obs_se)
    ll_alt = _log_normal_density(obs_mean, alt_mean, obs_se)

    return ll_alt - ll_null


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Pearson correlation coefficient with bootstrap p-value.
    Returns (r, p_bootstrap).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return 0.0, 1.0

    x, y = x[:n], y[:n]
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return 0.0, 1.0

    r = float(np.mean((x - mx) * (y - my)) / (sx * sy))

    # Bootstrap p-value: fraction of shuffled correlations >= |r|
    rng = np.random.RandomState(42)
    n_boot = 1000
    count_extreme = 0
    for _ in range(n_boot):
        y_shuffled = rng.permutation(y)
        r_null = float(np.mean((x - mx) * (y_shuffled - np.mean(y_shuffled)))
                        / (sx * np.std(y_shuffled))) if np.std(y_shuffled) > 0 else 0.0
        if abs(r_null) >= abs(r):
            count_extreme += 1

    p = count_extreme / n_boot
    return r, p


# ---------------------------------------------------------------------------
# Paradigm-Specific Metrics (Phase 5)
# ---------------------------------------------------------------------------

def chi_squared_goodness(
    observed: np.ndarray, expected: np.ndarray,
) -> Tuple[float, float]:
    """
    Chi-squared goodness-of-fit test.

    Returns (chi2_statistic, p_value).
    """
    from scipy.stats import chi2 as chi2_dist
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    # Pseudocount to avoid division by zero
    expected = expected + 1e-10
    chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
    df = max(1, len(observed) - 1)
    p_val = 1.0 - float(chi2_dist.cdf(chi2_stat, df))
    return chi2_stat, p_val


def rank_correlation(
    x: np.ndarray, y: np.ndarray,
) -> Tuple[float, float]:
    """Spearman rank correlation between two arrays."""
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return 0.0, 1.0
    rho, p = spearmanr(x[:n], y[:n])
    return float(rho), float(p)


def selectivity_ratio(real_value: float, null_values: np.ndarray) -> float:
    """
    Compute selectivity ratio: real / mean(null).

    A ratio > 1.5 indicates the real signal exceeds the null meaningfully.
    Returns inf if null_mean is near zero and real_value > 0.
    """
    null_values = np.asarray(null_values, dtype=float)
    null_mean = float(np.mean(null_values))
    if abs(null_mean) < 1e-10:
        return float('inf') if real_value > 0 else 0.0
    return real_value / null_mean


def paradigm_shape_vector(
    n_forms: int, suffix_set: set, prefix_set: set,
) -> np.ndarray:
    """
    Encode a paradigm shape as a 7-dim feature vector for clustering.

    Features: [n_forms, n_suffixes, n_prefixes, has_prefix,
               max_suffix_len, min_suffix_len, suffix_diversity_ratio]
    """
    suffixes = list(suffix_set)
    prefixes = list(prefix_set)
    n_suf = len(suffixes)
    n_pre = len(prefixes)
    max_suf_len = max((len(s) for s in suffixes), default=0)
    min_suf_len = min((len(s) for s in suffixes), default=0)
    suf_diversity = n_suf / max(n_forms, 1)
    return np.array([
        n_forms, n_suf, n_pre, float(n_pre > 0),
        max_suf_len, min_suf_len, suf_diversity,
    ], dtype=float)


# ---------------------------------------------------------------------------
# Co-occurrence, Embeddings, and Alignment (Phase 7)
# ---------------------------------------------------------------------------

def build_cooccurrence_matrix(
    tokens: List[str],
    vocab: List[str],
    window: int = 2,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build symmetric word-word co-occurrence count matrix.

    Counts how often each pair of vocabulary items co-occurs within
    a sliding window of *window* tokens on each side.

    Returns (n_vocab × n_vocab) dense count matrix and {word: index} mapping.
    """
    word2idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    matrix = np.zeros((n, n), dtype=float)
    for pos, token in enumerate(tokens):
        if token not in word2idx:
            continue
        i = word2idx[token]
        lo = max(0, pos - window)
        hi = min(len(tokens), pos + window + 1)
        for j_pos in range(lo, hi):
            if j_pos == pos:
                continue
            ctx = tokens[j_pos]
            if ctx in word2idx:
                j = word2idx[ctx]
                matrix[i, j] += 1.0
    return matrix, word2idx


def ppmi_matrix(
    cooccurrence: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """
    Compute Positive Pointwise Mutual Information matrix.

    PPMI(i,j) = max(0, log2(P(i,j) / (P(i) * P(j)^alpha)))

    Context distribution smoothing (alpha < 1) reduces bias against
    rare words, critical for small corpora.
    """
    total = cooccurrence.sum()
    if total < 1e-10:
        return np.zeros_like(cooccurrence)

    row_sums = cooccurrence.sum(axis=1)            # P(i) marginal
    col_sums = cooccurrence.sum(axis=0)            # P(j) marginal

    # Context smoothing: raise context frequencies to alpha, renormalize
    col_smooth = col_sums ** alpha
    col_smooth_sum = col_smooth.sum()

    n = cooccurrence.shape[0]
    result = np.zeros_like(cooccurrence)
    for i in range(n):
        if row_sums[i] < 1e-10:
            continue
        for j in range(n):
            if cooccurrence[i, j] < 1e-10 or col_smooth[j] < 1e-10:
                continue
            p_ij = cooccurrence[i, j] / total
            p_i = row_sums[i] / total
            p_j_smooth = col_smooth[j] / col_smooth_sum
            pmi = math.log2(p_ij / (p_i * p_j_smooth))
            if pmi > 0:
                result[i, j] = pmi
    return result


def truncated_svd(
    matrix: np.ndarray,
    n_components: int = 50,
) -> np.ndarray:
    """
    Truncated SVD dimensionality reduction.

    Returns U[:, :k] * sqrt(S[:k]) — the standard word embedding matrix.
    Clamps n_components to min(matrix.shape) - 1 if needed.
    """
    k = min(n_components, min(matrix.shape) - 1)
    if k < 1:
        return np.zeros((matrix.shape[0], 1))
    U, S, _ = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :k] * np.sqrt(S[:k])


def procrustes_alignment(
    source: np.ndarray,
    target: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Orthogonal Procrustes alignment between two embedding spaces.

    Finds rotation R minimizing ||source[src_idx] @ R - target[tgt_idx]||_F
    using scipy.linalg.orthogonal_procrustes.

    Returns (source @ R, residual_frobenius_norm).
    """
    from scipy.linalg import orthogonal_procrustes as _oprocruste
    src_idx = np.asarray(src_idx, dtype=int)
    tgt_idx = np.asarray(tgt_idx, dtype=int)
    A = source[src_idx]
    B = target[tgt_idx]
    R, scale = _oprocruste(A, B)
    aligned = source @ R
    residual = float(np.linalg.norm(aligned[src_idx] - B))
    return aligned, residual


def gromov_wasserstein_distance(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    p: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    epsilon: float = 0.1,
    n_iter: int = 100,
) -> float:
    """
    Approximate Gromov-Wasserstein distance between two metric spaces.

    Compares internal distance structure without requiring point
    correspondences. Uses entropic regularization with Sinkhorn-like
    projection for tractability.

    Args:
        dist_a: (n × n) pairwise distance matrix for space A
        dist_b: (m × m) pairwise distance matrix for space B
        p: Weight vector for space A (uniform if None)
        q: Weight vector for space B (uniform if None)
        epsilon: Entropic regularization strength
        n_iter: Number of projection iterations
    """
    n = dist_a.shape[0]
    m = dist_b.shape[0]
    if p is None:
        p = np.ones(n) / n
    if q is None:
        q = np.ones(m) / m

    # Initialize transport plan as outer product of marginals
    T = np.outer(p, q)

    for _ in range(n_iter):
        # Cost matrix: C(i,j) = sum_{i',j'} (D_a(i,i') - D_b(j,j'))^2 * T(i',j')
        # Efficient: C = Da^2 p 1^T + 1 p^T Db^2 - 2 Da T Db
        da2 = dist_a ** 2
        db2 = dist_b ** 2
        term1 = da2 @ T @ np.ones((m, m))
        term2 = np.ones((n, n)) @ T @ db2
        term3 = 2.0 * dist_a @ T @ dist_b
        C = term1 + term2 - term3

        # Sinkhorn step with entropic regularization
        K = np.exp(-C / epsilon)
        K = np.maximum(K, 1e-300)  # numerical stability

        # Row normalization
        for _ in range(10):
            row_sum = K.sum(axis=1)
            row_sum = np.maximum(row_sum, 1e-300)
            K = K * (p / row_sum)[:, None]
            col_sum = K.sum(axis=0)
            col_sum = np.maximum(col_sum, 1e-300)
            K = K * (q / col_sum)[None, :]

        T = K

    # Compute final GW objective
    da2 = dist_a ** 2
    db2 = dist_b ** 2
    cost = (da2 @ T @ np.ones((m, m)) + np.ones((n, n)) @ T @ db2
            - 2.0 * dist_a @ T @ dist_b)
    gw = float(np.sum(cost * T))
    return gw


def cohens_kappa(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """
    Cohen's kappa inter-rater agreement coefficient.

    kappa = (p_observed - p_expected) / (1 - p_expected)
    Returns value in [-1, 1] where 1 = perfect agreement.
    """
    labels_a = np.asarray(labels_a, dtype=str)
    labels_b = np.asarray(labels_b, dtype=str)
    n = len(labels_a)
    if n == 0:
        return 0.0
    categories = sorted(set(labels_a) | set(labels_b))
    # Build confusion matrix
    cat2idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        conf[cat2idx[a], cat2idx[b]] += 1.0
    p_o = np.trace(conf) / n
    row_sums = conf.sum(axis=1) / n
    col_sums = conf.sum(axis=0) / n
    p_e = float(np.sum(row_sums * col_sums))
    if abs(1.0 - p_e) < 1e-10:
        return 1.0 if abs(p_o - 1.0) < 1e-10 else 0.0
    return float((p_o - p_e) / (1.0 - p_e))


def adjusted_rand_index(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """
    Adjusted Rand Index for comparing two clusterings.

    ARI = (RI - expected_RI) / (max_RI - expected_RI)
    Returns value in [-1, 1] where 1 = perfect agreement.
    """
    labels_a = np.asarray(labels_a, dtype=str)
    labels_b = np.asarray(labels_b, dtype=str)
    n = len(labels_a)
    if n < 2:
        return 0.0
    classes_a = sorted(set(labels_a))
    classes_b = sorted(set(labels_b))
    a2i = {c: i for i, c in enumerate(classes_a)}
    b2i = {c: i for i, c in enumerate(classes_b)}
    # Build contingency table
    nij = np.zeros((len(classes_a), len(classes_b)), dtype=float)
    for a, b in zip(labels_a, labels_b):
        nij[a2i[a], b2i[b]] += 1.0

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_comb_nij = sum(comb2(nij[i, j]) for i in range(nij.shape[0])
                       for j in range(nij.shape[1]))
    sum_comb_a = sum(comb2(nij[i, :].sum()) for i in range(nij.shape[0]))
    sum_comb_b = sum(comb2(nij[:, j].sum()) for j in range(nij.shape[1]))
    comb_n = comb2(n)
    if comb_n < 1e-10:
        return 0.0
    expected = sum_comb_a * sum_comb_b / comb_n
    max_idx = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_idx - expected
    if abs(denom) < 1e-10:
        return 1.0 if abs(sum_comb_nij - expected) < 1e-10 else 0.0
    return float((sum_comb_nij - expected) / denom)


def silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute mean silhouette coefficient for a clustering.

    For each point i:
      a(i) = mean distance to other points in same cluster
      b(i) = min over other clusters of mean distance to that cluster
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Returns mean s(i) over all points.  Value in [-1, 1].
    """
    from scipy.spatial.distance import cdist

    labels = np.asarray(labels)
    n = len(labels)
    if n < 2:
        return 0.0

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Pairwise Euclidean distances
    dists = cdist(embeddings, embeddings, metric='euclidean')

    silhouettes = np.zeros(n)
    for i in range(n):
        same_mask = labels == labels[i]
        same_mask[i] = False  # exclude self
        n_same = same_mask.sum()
        if n_same == 0:
            silhouettes[i] = 0.0
            continue
        a_i = dists[i, same_mask].mean()

        b_i = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other_mask = labels == lbl
            if other_mask.sum() == 0:
                continue
            mean_dist = dists[i, other_mask].mean()
            if mean_dist < b_i:
                b_i = mean_dist

        denom = max(a_i, b_i)
        silhouettes[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(silhouettes))


def hungarian_assignment(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Optimal assignment maximizing total score.

    Wraps scipy.optimize.linear_sum_assignment (which minimizes),
    so we negate the matrix before solving.

    Returns (row_indices, col_indices, total_score).
    """
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(-np.asarray(cost_matrix, dtype=float))
    total = float(cost_matrix[row_ind, col_ind].sum())
    return row_ind, col_ind, total


def fisher_combined_probability(
    p_values: List[float],
) -> Tuple[float, int, float]:
    """
    Fisher's method for combining independent p-values.

    chi2 = -2 * sum(ln(p_i))
    Under H0: chi2 ~ chi-squared(2k) where k = number of p-values.

    Returns (chi2_statistic, degrees_of_freedom, combined_p_value).
    """
    from scipy.stats import chi2 as chi2_dist

    clamped = [max(p, 1e-300) for p in p_values if 0 < p <= 1]
    if not clamped:
        return 0.0, 0, 1.0
    chi2_stat = -2.0 * sum(math.log(p) for p in clamped)
    df = 2 * len(clamped)
    combined_p = 1.0 - float(chi2_dist.cdf(chi2_stat, df))
    return chi2_stat, df, combined_p


# ---------------------------------------------------------------------------
# N-gram Language Model (Phase 8)
# ---------------------------------------------------------------------------

def build_ngram_lm(
    tokens: List[str],
    order: int = 3,
    smoothing: float = 0.01,
) -> Dict:
    """
    Build a character-level n-gram language model with add-k smoothing.

    Joins tokens with '_' as word boundary marker, then builds a dictionary
    mapping (order-1)-character context tuples to {next_char: count} dicts.

    Args:
        tokens: list of words (each word is a string of characters)
        order: n-gram order (e.g. 3 for trigram)
        smoothing: add-k smoothing constant for unseen n-grams

    Returns dict with:
        'order': int,
        'vocab': sorted list of characters (including '_'),
        'vocab_size': int,
        'counts': dict mapping context_tuple -> {next_char: count},
        'smoothing': float,
    """
    text = '_'.join(tokens)
    text = '_' + text + '_'

    vocab = sorted(set(text))
    vocab_size = len(vocab)

    counts: Dict[Tuple, Dict[str, int]] = {}
    ctx_len = order - 1

    for i in range(ctx_len, len(text)):
        context = tuple(text[i - ctx_len:i])
        char = text[i]
        if context not in counts:
            counts[context] = {}
        counts[context][char] = counts[context].get(char, 0) + 1

    return {
        'order': order,
        'vocab': vocab,
        'vocab_size': vocab_size,
        'counts': counts,
        'smoothing': smoothing,
    }


def cross_entropy_lm(
    text: str,
    lm: Dict,
    per_char: bool = True,
) -> float:
    """
    Compute cross-entropy of text under a character-level n-gram LM.

    Uses add-k smoothing for unseen n-grams and backs off to shorter
    contexts when a full context has never been seen.

    Args:
        text: string to score (should use '_' as word boundary)
        lm: language model dict from build_ngram_lm()
        per_char: if True return bits/char, else return total bits

    Returns cross-entropy in bits.
    """
    order = lm['order']
    counts = lm['counts']
    k = lm['smoothing']
    V = lm['vocab_size']
    ctx_len = order - 1

    if len(text) <= ctx_len:
        return math.log2(V) if per_char else math.log2(V) * max(len(text), 1)

    total_log_prob = 0.0
    n_chars = 0

    for i in range(ctx_len, len(text)):
        context = tuple(text[i - ctx_len:i])
        char = text[i]

        # Try full context, then back off
        prob = None
        for backoff in range(ctx_len + 1):
            ctx = context[backoff:]
            if ctx in counts:
                ctx_counts = counts[ctx]
                total_count = sum(ctx_counts.values())
                char_count = ctx_counts.get(char, 0)
                prob = (char_count + k) / (total_count + k * V)
                break

        if prob is None or prob <= 0:
            prob = 1.0 / V

        total_log_prob += math.log2(prob)
        n_chars += 1

    ce = -total_log_prob / n_chars if n_chars > 0 and per_char else -total_log_prob
    return ce


def simulated_annealing(
    cost_fn,
    init_state,
    propose_fn,
    max_iter: int = 500_000,
    t_start: float = 1.0,
    t_end: float = 0.001,
    n_restarts: int = 10,
    seed: int = 42,
    verbose: bool = False,
    checkpoint_interval: int = 10_000,
) -> Tuple:
    """
    General simulated annealing optimizer.

    Args:
        cost_fn: state -> float (lower is better)
        init_state: initial state (deep-copied for each restart)
        propose_fn: (state, rng) -> new_state
        max_iter: iterations per restart
        t_start, t_end: temperature schedule endpoints (exponential cooling)
        n_restarts: number of independent restarts
        seed: base random seed (incremented per restart)
        verbose: print progress
        checkpoint_interval: record best cost every N iterations

    Returns (best_state, best_cost, convergence_history)
    where convergence_history records best cost at each checkpoint.
    """
    import copy

    cooling_rate = (t_end / t_start) ** (1.0 / max_iter) if max_iter > 0 else 1.0

    global_best_state = None
    global_best_cost = float('inf')
    convergence_history: List[float] = []

    for restart in range(n_restarts):
        rng = random.Random(seed + restart)
        state = copy.deepcopy(init_state)
        current_cost = cost_fn(state)
        best_cost = current_cost
        best_state = copy.deepcopy(state)
        temp = t_start

        for it in range(max_iter):
            new_state = propose_fn(state, rng)
            new_cost = cost_fn(new_state)
            delta = new_cost - current_cost

            if delta < 0 or rng.random() < math.exp(-delta / temp):
                state = new_state
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = copy.deepcopy(state)

            temp *= cooling_rate

            if it % checkpoint_interval == 0:
                convergence_history.append(best_cost)

        if verbose:
            print(f"    Restart {restart + 1}/{n_restarts}: "
                  f"best cost = {best_cost:.6f}")

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_state = copy.deepcopy(best_state)

    return global_best_state, global_best_cost, convergence_history


# ---------------------------------------------------------------------------
# Phase 9: Piecewise Zipf & Entropy Curves
# ---------------------------------------------------------------------------

def piecewise_zipf_fit(
    ranks: np.ndarray,
    freqs: np.ndarray,
    min_segment_size: int = 10,
) -> Dict[str, Any]:
    """
    Fit a two-segment (piecewise) power law to rank-frequency data.

    Sweeps all possible breakpoints and selects the one that minimizes
    total sum-of-squared-errors in log-log space.  Also fits a single
    power law for comparison.

    Returns dict with breakpoint_rank, per-segment exponents and R²,
    and SSE values for both models.
    """
    ranks = np.asarray(ranks, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    n = len(ranks)
    if n < 2 * min_segment_size:
        return {
            'breakpoint_rank': n // 2,
            'segment1_exponent': 0.0, 'segment1_r_squared': 0.0,
            'segment2_exponent': 0.0, 'segment2_r_squared': 0.0,
            'single_exponent': 0.0, 'single_r_squared': 0.0,
            'sse_single': 0.0, 'sse_piecewise': 0.0, 'n_data': n,
        }

    log_r = np.log(ranks)
    log_f = np.log(freqs + 1e-10)

    # --- single fit ---
    A_full = np.vstack([log_r, np.ones(n)]).T
    try:
        sol = np.linalg.lstsq(A_full, log_f, rcond=None)
        slope_s, intercept_s = sol[0]
    except np.linalg.LinAlgError:
        slope_s, intercept_s = -1.0, 0.0
    pred_s = slope_s * log_r + intercept_s
    sse_single = float(np.sum((log_f - pred_s) ** 2))
    ss_tot = float(np.sum((log_f - np.mean(log_f)) ** 2))
    r2_single = 1 - sse_single / ss_tot if ss_tot > 0 else 0.0

    # --- sweep breakpoints ---
    best_sse_pw = float('inf')
    best_bp = min_segment_size
    best_slopes = (-1.0, 0.0, -1.0, 0.0)

    for bp in range(min_segment_size, n - min_segment_size + 1):
        lr1, lf1 = log_r[:bp], log_f[:bp]
        lr2, lf2 = log_r[bp:], log_f[bp:]
        A1 = np.vstack([lr1, np.ones(bp)]).T
        A2 = np.vstack([lr2, np.ones(n - bp)]).T
        try:
            s1, i1 = np.linalg.lstsq(A1, lf1, rcond=None)[0]
            s2, i2 = np.linalg.lstsq(A2, lf2, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        sse = float(np.sum((lf1 - (s1 * lr1 + i1)) ** 2)
                     + np.sum((lf2 - (s2 * lr2 + i2)) ** 2))
        if sse < best_sse_pw:
            best_sse_pw = sse
            best_bp = bp
            best_slopes = (s1, i1, s2, i2)

    # per-segment R²
    def _seg_r2(log_r_seg, log_f_seg, slope, intercept):
        pred = slope * log_r_seg + intercept
        ss_res = float(np.sum((log_f_seg - pred) ** 2))
        ss_t = float(np.sum((log_f_seg - np.mean(log_f_seg)) ** 2))
        return 1 - ss_res / ss_t if ss_t > 0 else 0.0

    s1, i1, s2, i2 = best_slopes
    r2_seg1 = _seg_r2(log_r[:best_bp], log_f[:best_bp], s1, i1)
    r2_seg2 = _seg_r2(log_r[best_bp:], log_f[best_bp:], s2, i2)

    return {
        'breakpoint_rank': int(best_bp),
        'segment1_exponent': float(-s1),
        'segment1_r_squared': float(r2_seg1),
        'segment2_exponent': float(-s2),
        'segment2_r_squared': float(r2_seg2),
        'single_exponent': float(-slope_s),
        'single_r_squared': float(r2_single),
        'sse_single': sse_single,
        'sse_piecewise': float(best_sse_pw),
        'n_data': n,
    }


def aic_bic_compare(
    n_data: int,
    sse_model1: float,
    k_model1: int,
    sse_model2: float,
    k_model2: int,
) -> Dict[str, Any]:
    """
    Compare two models using AIC and BIC.

    AIC = n * ln(SSE / n) + 2k
    BIC = n * ln(SSE / n) + k * ln(n)

    Lower is better.  delta < 0 means model 2 is preferred.
    """
    if n_data <= 0 or sse_model1 <= 0 or sse_model2 <= 0:
        return {
            'aic_model1': 0.0, 'aic_model2': 0.0,
            'bic_model1': 0.0, 'bic_model2': 0.0,
            'delta_aic': 0.0, 'delta_bic': 0.0,
            'preferred_model': 'neither',
        }
    ln_n = math.log(n_data)
    aic1 = n_data * math.log(sse_model1 / n_data) + 2 * k_model1
    aic2 = n_data * math.log(sse_model2 / n_data) + 2 * k_model2
    bic1 = n_data * math.log(sse_model1 / n_data) + k_model1 * ln_n
    bic2 = n_data * math.log(sse_model2 / n_data) + k_model2 * ln_n
    d_aic = aic2 - aic1
    d_bic = bic2 - bic1
    if d_aic < -2 and d_bic < -2:
        preferred = 'model2'
    elif d_aic > 2 and d_bic > 2:
        preferred = 'model1'
    else:
        preferred = 'ambiguous'
    return {
        'aic_model1': float(aic1), 'aic_model2': float(aic2),
        'bic_model1': float(bic1), 'bic_model2': float(bic2),
        'delta_aic': float(d_aic), 'delta_bic': float(d_bic),
        'preferred_model': preferred,
    }


def entropy_curve(text: str, max_order: int = 8) -> Dict[int, float]:
    """
    Compute conditional character entropy at context orders 0 … max_order.

    Order 0 is the unconditional (first-order) character entropy.
    """
    curve: Dict[int, float] = {0: first_order_entropy(text)}
    for order in range(1, max_order + 1):
        curve[order] = conditional_entropy(text, order=order)
    return curve


# ---------------------------------------------------------------------------
# Token-Level Entropy with Smoothing & Back-off (Phase 10)
# ---------------------------------------------------------------------------

def token_conditional_entropy(
    tokens: List[str],
    order: int,
) -> float:
    """
    Token-level conditional entropy H(W_t | W_{t-1}, ..., W_{t-order}).

    For order 0, returns word_unigram_entropy(tokens).
    For order >= 1, uses plug-in MLE: H(ngram) - H(context).
    For order > 3 where sparsity causes upward bias, applies
    interpolated back-off across sub-orders.
    """
    if order == 0:
        return word_unigram_entropy(tokens)

    n = len(tokens)
    if n <= order:
        return 0.0

    if order <= 3:
        # Direct plug-in (same formula as word_conditional_entropy)
        return word_conditional_entropy(tokens, order=order)

    # Interpolated back-off for order > 3:
    # Higher-order MLE estimates suffer from positive bias when most
    # contexts are seen only once.  Interpolate across sub-orders,
    # weighting by how much data each order has.
    h_estimates = []
    weights = []
    for sub_order in range(1, order + 1):
        h_sub = word_conditional_entropy(tokens, order=sub_order)

        # Weight: mean context count determines reliability
        n_contexts = n - sub_order
        if n_contexts <= 0:
            continue
        ngram_counts: Counter = Counter()
        for i in range(n_contexts):
            ctx = tuple(tokens[i:i + sub_order])
            ngram_counts[ctx] += 1
        n_unique_ctx = len(ngram_counts)
        mean_count = n_contexts / max(n_unique_ctx, 1)

        # Jelinek-Mercer: trust higher-order more when contexts have
        # enough observations
        lam = mean_count / (mean_count + 5.0)
        w = lam ** (sub_order - 1)  # order 1 always has weight ~1
        h_estimates.append(h_sub)
        weights.append(w)

    if not h_estimates:
        return 0.0

    total_w = sum(weights)
    return sum(h * w for h, w in zip(h_estimates, weights)) / total_w


def token_entropy_curve(
    tokens: List[str],
    orders: Tuple[int, ...] = (0, 1, 2, 3, 5, 10),
) -> Dict[int, float]:
    """
    Compute token-level conditional entropy at each context order.

    Returns {order: H} dict.  Order 0 is the unconditional word entropy.
    Token-level analog of entropy_curve() (which is character-level).
    """
    curve: Dict[int, float] = {}
    for order in orders:
        curve[order] = token_conditional_entropy(tokens, order)
    return curve


def fit_exponential_decay(
    x_vals: List[float],
    y_vals: List[float],
) -> Tuple[float, float, float]:
    """
    Fit y = A * exp(-x / tau) via log-linear regression on ln(y) = ln(A) - x/tau.

    Filters out y <= 0 entries before fitting.
    Returns (A, tau, r_squared).  If fitting fails, returns (0, 0, 0).
    """
    xs, ys = [], []
    for x, y in zip(x_vals, y_vals):
        if y > 0:
            xs.append(float(x))
            ys.append(float(y))
    if len(xs) < 2:
        return (0.0, 0.0, 0.0)

    log_y = [math.log(y) for y in ys]
    x_arr = np.array(xs)
    log_y_arr = np.array(log_y)

    # Linear regression: log_y = intercept + slope * x
    A_mat = np.vstack([x_arr, np.ones(len(x_arr))]).T
    result = np.linalg.lstsq(A_mat, log_y_arr, rcond=None)
    slope, intercept = result[0]

    A = math.exp(intercept)
    tau = -1.0 / slope if slope != 0 else float('inf')

    # R-squared
    y_pred = intercept + slope * x_arr
    ss_res = float(np.sum((log_y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((log_y_arr - np.mean(log_y_arr)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return (A, tau, r_sq)


def coefficient_of_variation(values: List[float]) -> float:
    """Coefficient of variation: std / mean.  Returns 0 if mean is near zero."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    m = float(np.mean(arr))
    if abs(m) < 1e-12:
        return 0.0
    return float(np.std(arr) / m)


# ---------------------------------------------------------------------------
# Phase B-C: Paleographic Comparison Utilities
# ---------------------------------------------------------------------------

def cosine_similarity_triples(
    v_triple: Dict[str, str],
    t_triple: Dict[str, str],
) -> float:
    """Compute cosine similarity between two stroke-feature triples.

    Each triple is a dict with keys: first_stroke, last_stroke, glyph_class.
    Similarity is computed as fraction of matching components (0, 1/3, 2/3, or 1).
    """
    matches = 0
    total = 0
    for key in ('first_stroke', 'last_stroke', 'glyph_class'):
        v_val = v_triple.get(key, '')
        t_val = t_triple.get(key, '')
        if v_val and t_val:
            total += 1
            if v_val == t_val:
                matches += 1

    if total == 0:
        return 0.0
    return matches / total


def stroke_similarity(
    eva_strokes: Dict[str, str],
    hist_strokes: Dict[str, str],
    include_class: bool = True,
) -> float:
    """Two-tier stroke similarity for Phase 21 paleographic comparison.

    Level 1: canonical exact match → 1.0 per component.
    Level 2: category match → 0.5 per component.
    Level 3: no match → 0.0.

    Normalizes by the number of available (non-empty) components.

    Parameters
    ----------
    eva_strokes : dict
        Keys: first_stroke, last_stroke, and optionally glyph_class.
        Values should already be in canonical form.
    hist_strokes : dict
        Same key structure. May lack glyph_class.
    include_class : bool
        If True, include glyph_class in comparison (3 components).
        If False, compare only first_stroke and last_stroke (2 components).
    """
    from voynich.core.reference import normalize_stroke, stroke_category

    keys = ['first_stroke', 'last_stroke']
    if include_class:
        keys.append('glyph_class')

    score = 0.0
    available = 0

    for key in keys:
        e_val = eva_strokes.get(key, '')
        h_val = hist_strokes.get(key, '')
        if not e_val or not h_val:
            continue

        available += 1

        # Normalize both sides
        e_canon = normalize_stroke(e_val)
        h_canon = normalize_stroke(h_val)

        if e_canon == h_canon:
            score += 1.0
        elif key != 'glyph_class':
            # Category-level fuzzy match (not applicable to glyph_class)
            e_cat = stroke_category(e_canon)
            h_cat = stroke_category(h_canon)
            if e_cat != 'unknown' and e_cat == h_cat:
                score += 0.5

    if available == 0:
        return 0.0
    return score / available


def compute_phrase_selectivity(
    n_phrases: int,
    null_phrase_counts: List[int],
) -> Dict[str, float]:
    """Compute phrase detection selectivity against null baseline.

    Parameters
    ----------
    n_phrases : int
        Number of phrases detected in the real decoded output.
    null_phrase_counts : list of int
        Number of phrases detected in each null/random decode.

    Returns
    -------
    Dict with:
        'selectivity': float (n_phrases / null_mean, or inf if null_mean=0)
        'p_value': float (empirical p-value)
        'null_mean': float
        'null_std': float
        'z_score': float
    """
    if not null_phrase_counts:
        return {
            'selectivity': float('inf') if n_phrases > 0 else 0.0,
            'p_value': 0.0 if n_phrases > 0 else 1.0,
            'null_mean': 0.0,
            'null_std': 0.0,
            'z_score': float('inf') if n_phrases > 0 else 0.0,
        }

    null_arr = np.array(null_phrase_counts, dtype=float)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))

    if null_mean > 0:
        selectivity = n_phrases / null_mean
    else:
        selectivity = float('inf') if n_phrases > 0 else 0.0

    if null_std > 0:
        z_score = (n_phrases - null_mean) / null_std
    else:
        z_score = float('inf') if n_phrases > null_mean else 0.0

    # Empirical p-value: fraction of null runs >= real count
    n_exceeding = sum(1 for nc in null_phrase_counts if nc >= n_phrases)
    p_value = n_exceeding / len(null_phrase_counts)

    return {
        'selectivity': selectivity,
        'p_value': p_value,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_score': z_score,
    }


def overlap_rate_with_null(
    real_overlap: int,
    total_items: int,
    null_overlaps: List[int],
) -> Dict[str, float]:
    """Compute overlap selectivity against null baseline.

    Parameters
    ----------
    real_overlap : int
        Number of items that overlap in the real comparison.
    total_items : int
        Total number of items compared.
    null_overlaps : list of int
        Overlap counts from null/random comparisons.

    Returns
    -------
    Dict with rate, null_mean_rate, selectivity, p_value.
    """
    real_rate = real_overlap / total_items if total_items > 0 else 0.0

    if not null_overlaps:
        return {
            'rate': real_rate,
            'null_mean_rate': 0.0,
            'selectivity': float('inf') if real_rate > 0 else 0.0,
            'p_value': 0.0 if real_overlap > 0 else 1.0,
        }

    null_rates = [n / total_items for n in null_overlaps] if total_items > 0 else [0.0] * len(null_overlaps)
    null_mean = float(np.mean(null_rates))

    selectivity = real_rate / null_mean if null_mean > 0 else (float('inf') if real_rate > 0 else 0.0)
    n_exceeding = sum(1 for nr in null_rates if nr >= real_rate)
    p_value = n_exceeding / len(null_rates)

    return {
        'rate': real_rate,
        'null_mean_rate': null_mean,
        'selectivity': selectivity,
        'p_value': p_value,
    }


# ---------------------------------------------------------------------------
# Word-Level N-Gram Language Model (Phase 49)
# ---------------------------------------------------------------------------

def build_word_ngram_lm(
    word_sequences: List[List[str]],
    order: int = 3,
    smoothing: float = 1.0,
) -> Dict:
    """
    Build a word-level n-gram language model with add-k smoothing.

    Args:
        word_sequences: list of word sequences (each is a list of word strings)
        order: n-gram order (e.g. 3 for trigram)
        smoothing: add-k smoothing constant

    Returns dict with:
        'order': int,
        'vocab': sorted list of words (including <BOS>, <EOS>),
        'vocab_size': int,
        'counts': dict mapping context_tuple_key -> {next_word: count},
        'smoothing': float,
    """
    BOS = '<BOS>'
    EOS = '<EOS>'
    ctx_len = order - 1

    vocab_set: set = {BOS, EOS}
    counts: Dict[str, Dict[str, int]] = {}

    for seq in word_sequences:
        padded = [BOS] * ctx_len + list(seq) + [EOS]
        vocab_set.update(seq)
        for i in range(ctx_len, len(padded)):
            context_key = '|'.join(padded[i - ctx_len:i])
            word = padded[i]
            if context_key not in counts:
                counts[context_key] = {}
            counts[context_key][word] = counts[context_key].get(word, 0) + 1

    vocab = sorted(vocab_set)
    return {
        'order': order,
        'vocab': vocab,
        'vocab_size': len(vocab),
        'counts': counts,
        'smoothing': smoothing,
    }


def cross_entropy_word_lm(
    word_sequence: List[str],
    lm: Dict,
    per_word: bool = True,
) -> float:
    """
    Compute cross-entropy of a word sequence under a word-level n-gram LM.

    Args:
        word_sequence: list of words to score
        lm: language model dict from build_word_ngram_lm()
        per_word: if True return bits/word, else total bits

    Returns cross-entropy in bits.
    """
    order = lm['order']
    counts = lm['counts']
    k = lm['smoothing']
    V = lm['vocab_size']
    ctx_len = order - 1

    BOS = '<BOS>'
    EOS = '<EOS>'
    padded = [BOS] * ctx_len + list(word_sequence) + [EOS]

    total_log_prob = 0.0
    n_words = 0

    for i in range(ctx_len, len(padded)):
        word = padded[i]

        # Try full context, then back off
        prob = None
        for backoff in range(ctx_len + 1):
            ctx_key = '|'.join(padded[i - ctx_len + backoff:i])
            if ctx_key in counts:
                ctx_counts = counts[ctx_key]
                total_count = sum(ctx_counts.values())
                word_count = ctx_counts.get(word, 0)
                prob = (word_count + k) / (total_count + k * V)
                break

        if prob is None or prob <= 0:
            prob = 1.0 / V

        total_log_prob += math.log2(prob)
        n_words += 1

    ce = -total_log_prob / n_words if n_words > 0 and per_word else -total_log_prob
    return ce


# ---------------------------------------------------------------------------
# Sinkhorn Optimal Transport (Phase 49)
# ---------------------------------------------------------------------------

def sinkhorn_ot(
    cost_matrix: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Compute entropic optimal transport plan via log-domain Sinkhorn.

    Args:
        cost_matrix: (n, m) cost matrix C
        a: (n,) source marginal (must sum to 1)
        b: (m,) target marginal (must sum to 1)
        reg: entropic regularization strength epsilon
        max_iter: maximum Sinkhorn iterations
        tol: convergence tolerance on marginal error

    Returns:
        (transport_plan, wasserstein_distance)
        transport_plan: (n, m) coupling matrix gamma
        wasserstein_distance: <C, gamma> = sum(C * gamma)
    """
    n, m = cost_matrix.shape
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)

    # Log-domain kernel: log K = -C / reg
    log_K = -cost_matrix / reg

    # Initialize dual variables
    log_u = np.zeros(n)
    log_v = np.zeros(m)

    for _it in range(max_iter):
        # Update u: log_u = log_a - logsumexp(log_K + log_v[None, :], axis=1)
        log_Kv = log_K + log_v[None, :]
        log_u_new = log_a - _logsumexp_rows(log_Kv)

        # Update v: log_v = log_b - logsumexp(log_K.T + log_u[None, :], axis=1)
        log_Ku = log_K.T + log_u_new[None, :]
        log_v_new = log_b - _logsumexp_rows(log_Ku)

        # Check convergence
        if np.max(np.abs(log_u_new - log_u)) < tol and np.max(np.abs(log_v_new - log_v)) < tol:
            log_u, log_v = log_u_new, log_v_new
            break
        log_u, log_v = log_u_new, log_v_new

    # Recover transport plan: gamma = diag(u) K diag(v)
    log_gamma = log_u[:, None] + log_K + log_v[None, :]
    gamma = np.exp(log_gamma)

    # Wasserstein distance
    w_dist = float(np.sum(gamma * cost_matrix))
    return gamma, w_dist


def _logsumexp_rows(log_matrix: np.ndarray) -> np.ndarray:
    """Numerically stable logsumexp along axis=1."""
    row_max = np.max(log_matrix, axis=1, keepdims=True)
    return (row_max.squeeze(1) +
            np.log(np.sum(np.exp(log_matrix - row_max), axis=1)))


# ---------------------------------------------------------------------------
# Gromov-Wasserstein Distance (Phase 49)
# ---------------------------------------------------------------------------

def gromov_wasserstein(
    D1: np.ndarray,
    D2: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, float]:
    """
    Compute entropic Gromov-Wasserstein distance between two metric spaces.

    Finds coupling gamma minimizing:
        sum_{i,j,k,l} |D1(i,k) - D2(j,l)|^2 * gamma(i,j) * gamma(k,l)
    with entropic regularization.

    Args:
        D1: (n, n) intra-distance matrix for source space
        D2: (m, m) intra-distance matrix for target space
        p: (n,) source marginal
        q: (m,) target marginal
        reg: entropic regularization
        max_iter: maximum outer iterations
        tol: convergence tolerance

    Returns:
        (coupling, gw_distance)
    """
    n = len(p)
    m = len(q)

    # Initialize coupling as outer product of marginals
    gamma = np.outer(p, q)

    # Precompute D1^2 and D2^2 row/column sums for efficient cost computation
    D1_sq = D1 ** 2
    D2_sq = D2 ** 2

    prev_gw = float('inf')

    for _it in range(max_iter):
        # Compute linearized cost matrix:
        # L(i,j) = sum_k D1(i,k)^2 * p(k) + sum_l D2(j,l)^2 * q(l) - 2 * D1 @ gamma @ D2.T
        term1 = (D1_sq @ p)[:, None]           # (n, 1)
        term2 = (D2_sq @ q)[None, :]           # (1, m)
        term3 = 2.0 * D1 @ gamma @ D2.T        # (n, m)
        cost = term1 + term2 - term3

        # Solve entropic OT with this linearized cost
        gamma_new, _ = sinkhorn_ot(cost, p, q, reg=reg, max_iter=100)

        # Compute GW objective
        gw_val = float(np.sum(cost * gamma_new))

        if abs(prev_gw - gw_val) < tol:
            gamma = gamma_new
            break
        gamma = gamma_new
        prev_gw = gw_val

    # Final GW distance
    gw_distance = float(np.sum(cost * gamma))
    return gamma, gw_distance


# ---------------------------------------------------------------------------
# NetLSD Spectral Signature (Phase 49)
# ---------------------------------------------------------------------------

def netlsd_signature(
    eigenvalues: np.ndarray,
    timescales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute NetLSD heat kernel trace signature from graph Laplacian eigenvalues.

    h(t) = (1/n) * sum_i exp(-t * lambda_i)

    Args:
        eigenvalues: sorted eigenvalues of the normalized Laplacian
        timescales: array of time values t; defaults to logspace(-2, 2, 50)

    Returns:
        signature array of shape (len(timescales),)
    """
    if timescales is None:
        timescales = np.logspace(-2, 2, 50)

    eigs = np.asarray(eigenvalues, dtype=np.float64)
    n = len(eigs)
    if n == 0:
        return np.zeros(len(timescales))

    # h(t) = (1/n) sum_i exp(-t * lambda_i)
    # Shape: (T, n) -> sum over n -> (T,)
    sig = np.sum(np.exp(-timescales[:, None] * eigs[None, :]), axis=1) / n
    return sig


def spectral_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """L2 distance between two spectral signatures (e.g. from netlsd_signature)."""
    return float(np.sqrt(np.sum((sig1 - sig2) ** 2)))
