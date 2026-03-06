"""
Phase 18.5 – Lempel-Ziv Complexity Growth Curve
=================================================

The ultimate tie-breaker between H1 / H2 / H3.

  H1 (Hoax)       → LZ dictionary flatlines once all table combinations
                     are seen; compression ratio plateaus early.
  H2 (Cipher)     → growth curve matches natural-language Latin, scaled
                     by a constant verbose factor.
  H3 (Taxonomic)  → ultra-compressible due to systematic prefix reuse;
                     compression ratio is lower than natural language.

Uses stdlib ``zlib`` and ``lzma`` plus a pure-Python LZ78 factoriser.

Dependency chain:
    (none — reads corpus directly)
        -> lz_complexity.json
"""

import json
import lzma
import math
import os
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars
from voynich.core.reference import load_reference_corpus
from voynich.core.stats import fit_exponential_decay


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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LZComplexityResult:
    voynich_byte_length: int
    latin_byte_length: int
    occitan_byte_length: int
    cardan_byte_length: int
    sample_sizes: List[int]
    voynich_zlib_ratios: List[Optional[float]]
    voynich_lzma_ratios: List[Optional[float]]
    voynich_lz78_counts: List[Optional[int]]
    latin_zlib_ratios: List[Optional[float]]
    latin_lz78_counts: List[Optional[int]]
    occitan_zlib_ratios: List[Optional[float]]
    cardan_zlib_ratios: List[Optional[float]]
    cardan_lz78_counts: List[Optional[int]]
    voynich_asymptotic_zlib: Optional[float]
    latin_asymptotic_zlib: Optional[float]
    cardan_asymptotic_zlib: Optional[float]
    lz78_growth_rate_voynich: Optional[float]
    lz78_growth_rate_latin: Optional[float]
    lz78_growth_rate_cardan: Optional[float]
    voynich_vs_cardan_ratio: Optional[float]
    voynich_vs_latin_ratio: Optional[float]
    hypothesis_support: Dict[str, float]
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _lz78_phrase_count(data: bytes) -> int:
    """Count phrases in the LZ78 factorisation of *data*.

    Each phrase is the shortest prefix not yet seen.  The count of
    distinct phrases is the LZ78 complexity of the string.
    """
    seen: set = set()
    current = b''
    count = 0
    for byte in data:
        current += bytes([byte])
        if current not in seen:
            seen.add(current)
            count += 1
            current = b''
    if current:
        count += 1
    return count


def _build_token_bytes(tokens: List[str]) -> bytes:
    """Encode a token list as a UTF-8 byte string (space-separated)."""
    return ' '.join(tokens).encode('utf-8')


def _compute_compression_profile(
    data: bytes,
    sample_sizes: List[int],
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[int]]]:
    """Compute zlib ratio, lzma ratio, and LZ78 phrase count at each sample size."""
    zlib_ratios: List[Optional[float]] = []
    lzma_ratios: List[Optional[float]] = []
    lz78_counts: List[Optional[int]] = []

    for n in sample_sizes:
        chunk = data[:n]
        if len(chunk) < 10:
            zlib_ratios.append(None)
            lzma_ratios.append(None)
            lz78_counts.append(None)
            continue

        actual_n = len(chunk)
        z_compressed = zlib.compress(chunk, 9)
        zlib_ratios.append(round(len(z_compressed) / actual_n, 4))

        l_compressed = lzma.compress(chunk)
        lzma_ratios.append(round(len(l_compressed) / actual_n, 4))

        lz78_counts.append(_lz78_phrase_count(chunk))

    return zlib_ratios, lzma_ratios, lz78_counts


def _fit_asymptote(sample_sizes: List[int], ratios: List[Optional[float]]) -> Optional[float]:
    """Fit exponential decay A*exp(-x/tau) to compression ratios and return the asymptotic value."""
    valid = [(n, r) for n, r in zip(sample_sizes, ratios) if r is not None and n > 0]
    if len(valid) < 3:
        return None
    x = np.array([v[0] for v in valid], dtype=float)
    y = np.array([v[1] for v in valid], dtype=float)
    try:
        A, tau, r_sq = fit_exponential_decay(x, y)
        # Asymptotic value as x → ∞: the fitted curve → 0, but the residual
        # floor is better estimated as the last observed ratio
        asymptote = float(y[-1])
        return round(asymptote, 4)
    except Exception:
        return round(float(y[-1]), 4) if len(y) > 0 else None


def _lz78_growth_rate(sample_sizes: List[int], counts: List[Optional[int]]) -> Optional[float]:
    """Slope of log(phrase_count) vs log(N) — the LZ78 growth exponent."""
    valid = [(n, c) for n, c in zip(sample_sizes, counts) if c is not None and n > 0 and c > 0]
    if len(valid) < 3:
        return None
    log_n = np.log(np.array([v[0] for v in valid], dtype=float))
    log_c = np.log(np.array([v[1] for v in valid], dtype=float))
    # linear regression in log-log space
    coeffs = np.polyfit(log_n, log_c, 1)
    return round(float(coeffs[0]), 4)


def _generate_cardan_bytes(
    tokens: List[str],
    n_tokens: int = 5000,
    seed: int = 42,
) -> bytes:
    """Generate a Cardan Grille null text by randomly recombining EVA
    characters according to unigram frequencies and word-length distribution
    from the real corpus.
    """
    rng = np.random.default_rng(seed)

    # Collect EVA character frequencies
    char_counts: Counter = Counter()
    length_counts: Counter = Counter()
    for tok in tokens:
        chars = tokenize_eva_chars(tok)
        for ch in chars:
            char_counts[ch] += 1
        length_counts[len(chars)] += 1

    chars_list = list(char_counts.keys())
    char_probs = np.array([char_counts[c] for c in chars_list], dtype=float)
    char_probs /= char_probs.sum()

    lengths = list(length_counts.keys())
    length_probs = np.array([length_counts[l] for l in lengths], dtype=float)
    length_probs /= length_probs.sum()

    fake_tokens: List[str] = []
    for _ in range(n_tokens):
        wlen = rng.choice(lengths, p=length_probs)
        chars = rng.choice(chars_list, size=wlen, p=char_probs)
        fake_tokens.append(''.join(chars))

    return _build_token_bytes(fake_tokens)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_lz_complexity() -> None:
    """Phase 18.5: Lempel-Ziv complexity growth curve."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 18.5: Lempel-Ziv Complexity Growth Curve")
    print("=" * 70)

    rd = _results_dir()
    sample_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    # ── 1. Voynich ────────────────────────────────────────────────────
    print("\n  1. Encoding Voynich corpus …")
    corpus = load_corpus(verbose=False)
    tokens_a = corpus.get_tokens(language='A', paragraph_only=True)
    voynich_bytes = _build_token_bytes(tokens_a)
    # Ensure 'all' is included
    sizes_v = [s for s in sample_sizes if s <= len(voynich_bytes)] + [len(voynich_bytes)]
    print(f"     {len(voynich_bytes):,} bytes")

    v_zlib, v_lzma, v_lz78 = _compute_compression_profile(voynich_bytes, sizes_v)
    print(f"     zlib ratios: {[r for r in v_zlib if r is not None]}")

    # ── 2. Latin ──────────────────────────────────────────────────────
    print("\n  2. Latin reference …")
    latin_bytes = b''
    l_zlib: List[Optional[float]] = []
    l_lz78: List[Optional[int]] = []
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        lat_tokens = ref.get_combined_tokens('latin')
        if lat_tokens:
            latin_bytes = _build_token_bytes(lat_tokens)
            sizes_l = [s for s in sample_sizes if s <= len(latin_bytes)] + [len(latin_bytes)]
            l_zlib, _, l_lz78 = _compute_compression_profile(latin_bytes, sizes_l)
            print(f"     {len(latin_bytes):,} bytes  |  zlib ratios: {[r for r in l_zlib if r is not None]}")
    except Exception as e:
        print(f"     WARNING: Latin unavailable ({e})")

    # ── 3. Occitan ────────────────────────────────────────────────────
    print("\n  3. Occitan reference …")
    occitan_bytes = b''
    oc_zlib: List[Optional[float]] = []
    try:
        ref_oc = load_reference_corpus(languages=['occitan'], verbose=False)
        oc_tokens = ref_oc.get_combined_tokens('occitan')
        if oc_tokens:
            occitan_bytes = _build_token_bytes(oc_tokens)
            sizes_oc = [s for s in sample_sizes if s <= len(occitan_bytes)] + [len(occitan_bytes)]
            oc_zlib, _, _ = _compute_compression_profile(occitan_bytes, sizes_oc)
            print(f"     {len(occitan_bytes):,} bytes")
    except Exception:
        print("     WARNING: Occitan unavailable")

    # ── 4. Cardan Grille null ─────────────────────────────────────────
    print("\n  4. Generating Cardan Grille null …")
    cardan_bytes = _generate_cardan_bytes(tokens_a, n_tokens=len(tokens_a))
    sizes_c = [s for s in sample_sizes if s <= len(cardan_bytes)] + [len(cardan_bytes)]
    c_zlib, _, c_lz78 = _compute_compression_profile(cardan_bytes, sizes_c)
    print(f"     {len(cardan_bytes):,} bytes  |  zlib ratios: {[r for r in c_zlib if r is not None]}")

    # ── 5. Fit asymptotes and growth rates ────────────────────────────
    print("\n  5. Fitting growth curves …")
    v_asymp = _fit_asymptote(sizes_v, v_zlib)
    l_asymp = _fit_asymptote([s for s in sample_sizes if s <= len(latin_bytes)] + [len(latin_bytes)], l_zlib) if l_zlib else None
    c_asymp = _fit_asymptote(sizes_c, c_zlib)

    v_growth = _lz78_growth_rate(sizes_v, v_lz78)
    l_growth = _lz78_growth_rate([s for s in sample_sizes if s <= len(latin_bytes)] + [len(latin_bytes)], l_lz78) if l_lz78 else None
    c_growth = _lz78_growth_rate(sizes_c, c_lz78)

    vs_cardan = round(v_asymp / c_asymp, 4) if v_asymp and c_asymp and c_asymp > 0 else None
    vs_latin = round(v_asymp / l_asymp, 4) if v_asymp and l_asymp and l_asymp > 0 else None

    print(f"     Voynich asymptotic zlib = {v_asymp}  |  LZ78 growth = {v_growth}")
    print(f"     Latin   asymptotic zlib = {l_asymp}  |  LZ78 growth = {l_growth}")
    print(f"     Cardan  asymptotic zlib = {c_asymp}  |  LZ78 growth = {c_growth}")
    print(f"     Voynich / Cardan = {vs_cardan}  |  Voynich / Latin = {vs_latin}")

    # ── 6. Hypothesis scoring ─────────────────────────────────────────
    print("\n  6. Scoring hypotheses …")

    # H1: compression profile similar to Cardan null
    h1 = _sigmoid(-(abs((vs_cardan or 1.0) - 1.0)) / 0.15) if vs_cardan else 0.3

    # H2: growth rate similar to Latin (scaled)
    h2 = _sigmoid(-abs((vs_latin or 1.0) - 1.0) / 0.2) if vs_latin else 0.3

    # H3: ultra-compressible (low asymptotic ratio)
    h3 = _sigmoid(-(((v_asymp or 0.5) - 0.15)) / 0.1) if v_asymp else 0.3

    total = h1 + h2 + h3
    if total > 0:
        h1, h2, h3 = h1 / total, h2 / total, h3 / total

    hypothesis_support = {'H1': round(h1, 4), 'H2': round(h2, 4), 'H3': round(h3, 4)}
    print(f"     H1={h1:.3f}  H2={h2:.3f}  H3={h3:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────
    if vs_cardan is not None and abs(vs_cardan - 1.0) < 0.15:
        verdict = (f"CARDAN-LIKE: Voynich compression profile matches Cardan Grille null "
                   f"(ratio = {vs_cardan:.3f}). Consistent with H1 (hoax).")
    elif vs_latin is not None and abs(vs_latin - 1.0) < 0.2:
        verdict = (f"NATURAL-LIKE: Voynich compression profile matches Latin "
                   f"(ratio = {vs_latin:.3f}). Consistent with H2 (verbose cipher over "
                   "natural language).")
    elif v_asymp is not None and v_asymp < 0.2:
        verdict = (f"ULTRA-COMPRESSIBLE: Voynich asymptotic zlib ratio = {v_asymp:.3f} — "
                   "very low, suggesting systematic prefix reuse. Consistent with H3 "
                   "(taxonomic language).")
    else:
        verdict = (f"MIXED: Voynich asymptotic ratio = {v_asymp}, vs Cardan = {vs_cardan}, "
                   f"vs Latin = {vs_latin}. No single hypothesis clearly dominant.")

    print(f"\n  Verdict: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────
    result = LZComplexityResult(
        voynich_byte_length=len(voynich_bytes),
        latin_byte_length=len(latin_bytes),
        occitan_byte_length=len(occitan_bytes),
        cardan_byte_length=len(cardan_bytes),
        sample_sizes=sizes_v,
        voynich_zlib_ratios=v_zlib,
        voynich_lzma_ratios=v_lzma,
        voynich_lz78_counts=v_lz78,
        latin_zlib_ratios=l_zlib,
        latin_lz78_counts=l_lz78,
        occitan_zlib_ratios=oc_zlib,
        cardan_zlib_ratios=c_zlib,
        cardan_lz78_counts=c_lz78,
        voynich_asymptotic_zlib=v_asymp,
        latin_asymptotic_zlib=l_asymp,
        cardan_asymptotic_zlib=c_asymp,
        lz78_growth_rate_voynich=v_growth,
        lz78_growth_rate_latin=l_growth,
        lz78_growth_rate_cardan=c_growth,
        voynich_vs_cardan_ratio=vs_cardan,
        voynich_vs_latin_ratio=vs_latin,
        hypothesis_support=hypothesis_support,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'lz_complexity.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)

    print(f"\n  -> {out_path}")
