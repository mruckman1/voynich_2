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
