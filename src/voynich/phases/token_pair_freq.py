"""
Phase 33.13 – Token Pair Frequency Tables
==========================================
Build frequency tables of consecutive word pairs in the Voynich corpus
(herbal_a section), the Latin reference corpus, and the decoded Voynich
corpus.  These distributional tables feed Step 33.14 (Hungarian-algorithm
matching).

Dependency chain:
    combined_refine.json     (Phase 15 assignment)
    modifier_integrate.json  (Phase 16 modifiers)
        → token_pair_freq.json  (this step)
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.null_corpus import _reconstruct_modifier_rules


# ---------------------------------------------------------------------------
# Helpers
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


def _zipf_exponent(counts: List[int]) -> float:
    """Fit Zipf exponent via log-log linear regression."""
    if len(counts) < 2:
        return 0.0
    sorted_counts = sorted(counts, reverse=True)
    n = min(len(sorted_counts), 200)  # use top 200
    log_ranks = [math.log(i + 1) for i in range(n)]
    log_freqs = [math.log(max(c, 1)) for c in sorted_counts[:n]]
    # Simple linear regression: log_freq = -alpha * log_rank + b
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_freqs))
    den = sum((x - mean_x) ** 2 for x in log_ranks)
    slope = num / den if den > 0 else 0.0
    return -slope  # Zipf exponent is positive


def _concentration(pair_counts: Counter, n: int) -> float:
    """Fraction of total pair occurrences covered by the top-n pairs."""
    total = sum(pair_counts.values())
    if total == 0:
        return 0.0
    top_n = pair_counts.most_common(n)
    return sum(c for _, c in top_n) / total


def _spearman_rank_correlation(x_counts: Dict[Tuple[str, str], int],
                                y_counts: Dict[Tuple[str, str], int]) -> float:
    """
    Compute Spearman rank correlation between two pair-frequency
    distributions over their shared keys.

    Falls back to a manual implementation if scipy is unavailable.
    """
    shared_keys = sorted(set(x_counts.keys()) & set(y_counts.keys()))
    if len(shared_keys) < 3:
        return 0.0

    x_vals = [x_counts[k] for k in shared_keys]
    y_vals = [y_counts[k] for k in shared_keys]

    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x_vals, y_vals)
        return float(rho) if rho == rho else 0.0
    except ImportError:
        pass

    # Manual Spearman: rank both, then Pearson on ranks
    def _rank(vals: List[int]) -> List[float]:
        indexed = sorted(enumerate(vals), key=lambda iv: -iv[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j + 1) / 2.0  # 1-based average rank
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rx = _rank(x_vals)
    ry = _rank(y_vals)
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = sum((a - mean_rx) ** 2 for a in rx)
    den_y = sum((b - mean_ry) ** 2 for b in ry)
    den = (den_x * den_y) ** 0.5
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairStats:
    n_total_pairs: int
    n_unique_pairs: int
    zipf_exponent: float
    concentration_top20: float   # fraction of total covered by top 20 pairs
    concentration_top50: float
    top_pairs: List[Dict]        # [{pair: [w1, w2], count: N, rank: R}, ...]


@dataclass
class TokenPairFreqResult:
    # EVA token pairs (herbal_a)
    eva_pair_stats: Dict         # PairStats as dict
    # Latin word pairs (reference)
    latin_pair_stats: Dict       # PairStats as dict
    # Decoded pairs (herbal_a decoded)
    decoded_pair_stats: Dict     # PairStats as dict
    # Top-N inventories
    top_eva_tokens: List[Dict]   # [{token, count, rank}, ...]
    top_latin_words: List[Dict]
    top_decoded_words: List[Dict]
    # Rank correlation between decoded pairs and Latin pairs
    decoded_latin_rank_correlation: float
    # Metadata
    n_herbal_a_tokens: int
    n_latin_tokens: int
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Pair extraction helpers
# ---------------------------------------------------------------------------

def _extract_pairs_from_loci(pages, per_locus: bool = True) -> Tuple[Counter, List[str]]:
    """
    Extract consecutive token pairs from a list of VoynichPage objects.
    Respects locus boundaries so pairs never cross lines.

    Returns:
        pair_counts: Counter of (token1, token2) tuples
        all_tokens:  flat list of all tokens encountered
    """
    pair_counts: Counter = Counter()
    all_tokens: List[str] = []

    for page in pages:
        for locus in page.loci:
            tokens = tokenize(locus.clean_text)
            all_tokens.extend(tokens)
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += 1

    return pair_counts, all_tokens


def _extract_pairs_from_token_list(tokens: List[str]) -> Counter:
    """Extract consecutive pairs from a flat token list (no boundary awareness)."""
    pair_counts: Counter = Counter()
    for i in range(len(tokens) - 1):
        pair_counts[(tokens[i], tokens[i + 1])] += 1
    return pair_counts


def _build_pair_stats(pair_counts: Counter, top_n: int = 200) -> PairStats:
    """Build PairStats from a Counter of pair tuples."""
    counts_list = sorted(pair_counts.values(), reverse=True)
    total_pairs = sum(counts_list)
    unique_pairs = len(counts_list)

    zipf = _zipf_exponent(counts_list)
    conc_20 = _concentration(pair_counts, 20)
    conc_50 = _concentration(pair_counts, 50)

    top_pairs = []
    for rank, ((w1, w2), count) in enumerate(pair_counts.most_common(top_n), 1):
        top_pairs.append({'pair': [w1, w2], 'count': count, 'rank': rank})

    return PairStats(
        n_total_pairs=total_pairs,
        n_unique_pairs=unique_pairs,
        zipf_exponent=round(zipf, 4),
        concentration_top20=round(conc_20, 4),
        concentration_top50=round(conc_50, 4),
        top_pairs=top_pairs,
    )


def _build_top_tokens(tokens: List[str], top_n: int = 50) -> List[Dict]:
    """Build a ranked frequency list of individual tokens."""
    counts = Counter(tokens)
    result = []
    for rank, (token, count) in enumerate(counts.most_common(top_n), 1):
        result.append({'token': token, 'count': count, 'rank': rank})
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_token_pair_freq() -> None:
    """Step 33.13: Token pair frequency tables for distributional matching."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 33.13: Token Pair Frequency Tables")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load assignment and modifiers ──
    print("\n  1. Loading assignment and modifiers ...")

    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    print(f"     Assignment: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")

    # ── 2. Voynich EVA pair frequencies (herbal_a) ──
    print("\n  2. Building Voynich EVA pair frequencies (herbal_a) ...")
    corpus = load_corpus(verbose=False)
    herbal_pages = corpus.get_pages_by_section('herbal_a')
    print(f"     {len(herbal_pages)} herbal_a pages")

    eva_pair_counts, herbal_tokens = _extract_pairs_from_loci(herbal_pages)
    n_herbal_a_tokens = len(herbal_tokens)
    print(f"     {n_herbal_a_tokens} tokens, "
          f"{sum(eva_pair_counts.values())} total pairs, "
          f"{len(eva_pair_counts)} unique pairs")

    eva_pair_stats = _build_pair_stats(eva_pair_counts)

    # Top EVA token inventories (top-50 for JSON, report top-20 on screen)
    top_eva_tokens = _build_top_tokens(herbal_tokens, top_n=50)
    print("     Top-10 EVA tokens:")
    for entry in top_eva_tokens[:10]:
        print(f"       {entry['rank']:3d}. {entry['token']:12s}  count={entry['count']}")

    # ── 3. Latin reference pair frequencies ──
    print("\n  3. Building Latin reference pair frequencies ...")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    latin_tokens_raw = ref_corpus.get_combined_tokens('latin')
    # Lowercase and filter short tokens
    latin_tokens = [w.lower() for w in latin_tokens_raw if len(w) >= 2]
    n_latin_tokens = len(latin_tokens)
    print(f"     {n_latin_tokens} Latin reference tokens")

    latin_pair_counts = _extract_pairs_from_token_list(latin_tokens)
    print(f"     {sum(latin_pair_counts.values())} total pairs, "
          f"{len(latin_pair_counts)} unique pairs")

    latin_pair_stats = _build_pair_stats(latin_pair_counts)

    top_latin_words = _build_top_tokens(latin_tokens, top_n=50)
    print("     Top-10 Latin words:")
    for entry in top_latin_words[:10]:
        print(f"       {entry['rank']:3d}. {entry['token']:12s}  count={entry['count']}")

    # ── 4. Decoded pair frequencies (herbal_a through Phase 16 R3) ──
    print("\n  4. Decoding herbal_a tokens and building decoded pair frequencies ...")

    # Build reference word set for R3 decoding
    base_words = set(w.lower() for w in latin_tokens_raw if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     Reference word set: {len(ref_word_set)} words")

    # Decode per-locus to preserve boundaries for pair extraction
    decoded_pair_counts: Counter = Counter()
    all_decoded_tokens: List[str] = []

    for page in herbal_pages:
        for locus in page.loci:
            tokens = tokenize(locus.clean_text)
            if not tokens:
                continue
            decoded = _decode_corpus_r3(
                tokens, assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set,
            )
            all_decoded_tokens.extend(decoded)
            for i in range(len(decoded) - 1):
                decoded_pair_counts[(decoded[i], decoded[i + 1])] += 1

    n_decoded_hits = sum(1 for w in all_decoded_tokens if w in ref_word_set)
    dict_hit_rate = n_decoded_hits / len(all_decoded_tokens) if all_decoded_tokens else 0.0
    print(f"     {len(all_decoded_tokens)} decoded tokens, "
          f"dict_hit = {dict_hit_rate:.3f}")
    print(f"     {sum(decoded_pair_counts.values())} total decoded pairs, "
          f"{len(decoded_pair_counts)} unique decoded pairs")

    decoded_pair_stats = _build_pair_stats(decoded_pair_counts)

    top_decoded_words = _build_top_tokens(all_decoded_tokens, top_n=50)
    print("     Top-10 decoded words:")
    for entry in top_decoded_words[:10]:
        in_dict = " *" if entry['token'] in ref_word_set else ""
        print(f"       {entry['rank']:3d}. {entry['token']:12s}  "
              f"count={entry['count']}{in_dict}")

    # ── 5. Rank correlation between decoded and Latin pair frequencies ──
    print("\n  5. Computing rank correlation (decoded vs Latin pairs) ...")

    rho = _spearman_rank_correlation(decoded_pair_counts, latin_pair_counts)
    n_shared = len(set(decoded_pair_counts.keys()) & set(latin_pair_counts.keys()))
    print(f"     Shared pair types: {n_shared}")
    print(f"     Spearman rho: {rho:.4f}")

    # ── 6. Summary and verdict ──
    print("\n  6. Summary ...")
    print(f"     EVA pairs:     {eva_pair_stats.n_unique_pairs} unique, "
          f"Zipf={eva_pair_stats.zipf_exponent:.3f}, "
          f"conc20={eva_pair_stats.concentration_top20:.3f}")
    print(f"     Latin pairs:   {latin_pair_stats.n_unique_pairs} unique, "
          f"Zipf={latin_pair_stats.zipf_exponent:.3f}, "
          f"conc20={latin_pair_stats.concentration_top20:.3f}")
    print(f"     Decoded pairs: {decoded_pair_stats.n_unique_pairs} unique, "
          f"Zipf={decoded_pair_stats.zipf_exponent:.3f}, "
          f"conc20={decoded_pair_stats.concentration_top20:.3f}")

    verdict = (
        f"TABLES_BUILT: {eva_pair_stats.n_unique_pairs} EVA pairs, "
        f"{latin_pair_stats.n_unique_pairs} Latin pairs, "
        f"{decoded_pair_stats.n_unique_pairs} decoded pairs; "
        f"decoded-Latin rho={rho:.4f} over {n_shared} shared pairs"
    )
    print(f"\n  Verdict: {verdict}")

    # ── 7. Save results ──
    result = TokenPairFreqResult(
        eva_pair_stats=_convert(asdict(eva_pair_stats)),
        latin_pair_stats=_convert(asdict(latin_pair_stats)),
        decoded_pair_stats=_convert(asdict(decoded_pair_stats)),
        top_eva_tokens=top_eva_tokens,
        top_latin_words=top_latin_words,
        top_decoded_words=top_decoded_words,
        decoded_latin_rank_correlation=round(rho, 4),
        n_herbal_a_tokens=n_herbal_a_tokens,
        n_latin_tokens=n_latin_tokens,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'token_pair_freq.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}")
