"""
Phase 49 Track B – Fourier/Spectral Periodicity Analysis
=========================================================
Systematic spectral analysis across all folios to detect periodic structures
(formulaic text, recipe boundaries, repeating patterns). Uses FFT,
autocorrelation, and STFT windowed analysis.

Dependency chain:
    signal_bigrams.json         (Phase 29 parallel arrays)
        -> spectral_signals.json    (Step 49B.1)
        -> spectral_fft.json        (Step 49B.2)
        -> spectral_stft.json       (Step 49B.3)
        -> spectral_crossfolio.json (Step 49B.4)
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, stft
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus, tokenize_eva_chars


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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_convert(x) for x in obj.tolist()]
    if isinstance(obj, float) and (obj != obj):
        return None
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
# Constants
# ---------------------------------------------------------------------------

GALLOWS_CHARS = {'t', 'k', 'p', 'f'}
CHANNELS = ['freq_rank', 'signal_indicator', 'gallows_indicator', 'dict_hit',
            'token_length']


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpectralSignalsResult:
    n_folios: int
    n_tokens_total: int
    channels: List[str]
    mean_signal_length: float
    min_signal_length: int
    max_signal_length: int
    runtime_seconds: float


@dataclass
class SpectralFFTResult:
    n_folios_analyzed: int
    n_periodic_folios: int
    periodic_folio_ids: List[str]
    dominant_periods: Dict[str, List[float]]
    mean_autocorr_peak: float
    channel_periodicity: Dict[str, int]
    runtime_seconds: float


@dataclass
class SpectralSTFTResult:
    n_folios_analyzed: int
    n_formula_boundaries: int
    formula_folios: List[str]
    n_folios_with_transitions: int
    mean_transition_sharpness: float
    stft_window_size: int
    runtime_seconds: float


@dataclass
class SpectralCrossfolioResult:
    section_spectral_signatures: Dict[str, Dict]
    section_discriminability: float
    periodic_sections: List[str]
    n_folio_pairs_correlated: int
    spectral_clustering_silhouette: float
    n_clusters: int
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 49B.1  Token-to-Signal Conversion
# ---------------------------------------------------------------------------

def run_spectral_signals() -> None:
    """Convert per-token data into 5 signal channels per folio."""
    t0 = time.time()
    rd = _results_dir()

    print("=" * 60)
    print("Step 49B.1 – Token-to-Signal Conversion")
    print("=" * 60)

    # Load signal_bigrams.json for parallel arrays
    sb_path = os.path.join(rd, 'signal_bigrams.json')
    sb = _safe_load(sb_path)
    if not sb:
        raise FileNotFoundError(f"Missing {sb_path}")

    token_evas: List[str] = sb['token_evas']
    token_decoded: List[str] = sb.get('token_decoded', [])
    token_folios: List[str] = sb['token_folios']
    token_classifications: List[str] = sb['token_classifications']
    token_dict_hits: List[bool] = sb.get('token_dict_hits', [])

    n_tokens = len(token_evas)
    print(f"  Loaded {n_tokens} tokens from signal_bigrams.json")

    # Build corpus-wide frequency rank
    freq_counter = Counter(token_evas)
    # Rank: most common = 1
    sorted_tokens = freq_counter.most_common()
    token_to_rank: Dict[str, int] = {}
    for rank, (tok, _count) in enumerate(sorted_tokens, start=1):
        token_to_rank[tok] = rank
    max_rank = len(token_to_rank)

    # Group indices by folio
    folio_indices: Dict[str, List[int]] = defaultdict(list)
    for i, fol in enumerate(token_folios):
        folio_indices[fol].append(i)

    # Build 5 channels per folio
    folio_signals: Dict[str, Dict[str, List]] = {}
    folio_lengths: List[int] = []

    for fol in sorted(folio_indices.keys()):
        indices = folio_indices[fol]
        n_fol = len(indices)
        folio_lengths.append(n_fol)

        ch_freq_rank = []
        ch_signal = []
        ch_gallows = []
        ch_dict_hit = []
        ch_length = []

        for idx in indices:
            eva = token_evas[idx]

            # Channel 1: freq_rank
            ch_freq_rank.append(token_to_rank.get(eva, max_rank))

            # Channel 2: signal_indicator
            ch_signal.append(1 if token_classifications[idx] == 'SIGNAL' else 0)

            # Channel 3: gallows_indicator
            eva_chars = tokenize_eva_chars(eva)
            is_gallows = 1 if (eva_chars and eva_chars[0] in GALLOWS_CHARS) else 0
            ch_gallows.append(is_gallows)

            # Channel 4: dict_hit
            if token_dict_hits:
                ch_dict_hit.append(1 if token_dict_hits[idx] else 0)
            else:
                ch_dict_hit.append(0)

            # Channel 5: token_length
            ch_length.append(len(eva))

        folio_signals[fol] = {
            'freq_rank': ch_freq_rank,
            'signal_indicator': ch_signal,
            'gallows_indicator': ch_gallows,
            'dict_hit': ch_dict_hit,
            'token_length': ch_length,
        }

    n_folios = len(folio_signals)
    mean_len = float(np.mean(folio_lengths)) if folio_lengths else 0.0
    min_len = int(min(folio_lengths)) if folio_lengths else 0
    max_len = int(max(folio_lengths)) if folio_lengths else 0

    print(f"  {n_folios} folios, mean length {mean_len:.1f} tokens")
    print(f"  Range: {min_len} – {max_len} tokens per folio")
    print(f"  Channels: {CHANNELS}")

    # Save full signal data separately (can be large)
    _save_json(rd, 'spectral_signals_data.json', folio_signals)
    print(f"  Saved spectral_signals_data.json")

    result = SpectralSignalsResult(
        n_folios=n_folios,
        n_tokens_total=n_tokens,
        channels=CHANNELS,
        mean_signal_length=round(mean_len, 2),
        min_signal_length=min_len,
        max_signal_length=max_len,
        runtime_seconds=round(time.time() - t0, 3),
    )

    path = _save_json(rd, 'spectral_signals.json', result)
    print(f"  Saved {path}")
    print(f"  Runtime: {result.runtime_seconds:.2f}s\n")


# ---------------------------------------------------------------------------
# Step 49B.2  FFT + Autocorrelation
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= max(n, 64)."""
    n = max(n, 64)
    return 1 << (n - 1).bit_length()


def _autocorrelation(signal: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Compute normalized autocorrelation for positive lags."""
    signal = signal - np.mean(signal)
    if np.std(signal) < 1e-10:
        return np.zeros(max_lag)
    acf = np.correlate(signal, signal, mode='full')
    acf = acf[len(signal) - 1:]  # positive lags only
    acf = acf / (acf[0] + 1e-20)  # normalize
    return acf[:max_lag]


def run_spectral_fft() -> None:
    """FFT and autocorrelation analysis per folio per channel."""
    t0 = time.time()
    rd = _results_dir()

    print("=" * 60)
    print("Step 49B.2 – FFT + Autocorrelation")
    print("=" * 60)

    data_path = os.path.join(rd, 'spectral_signals_data.json')
    folio_signals = _safe_load(data_path)
    if not folio_signals:
        raise FileNotFoundError(f"Missing {data_path}")

    MIN_TOKENS = 8
    periodic_folio_ids: List[str] = []
    dominant_periods: Dict[str, List[float]] = {}
    channel_periodicity: Dict[str, int] = {ch: 0 for ch in CHANNELS}
    all_autocorr_peaks: List[float] = []
    n_analyzed = 0

    for fol in sorted(folio_signals.keys()):
        channels = folio_signals[fol]
        n_tok = len(channels[CHANNELS[0]])

        if n_tok < MIN_TOKENS:
            continue

        n_analyzed += 1
        folio_is_periodic = False
        folio_periods: List[float] = []

        for ch_name in CHANNELS:
            signal = np.array(channels[ch_name], dtype=np.float64)
            N = len(signal)
            N_padded = _next_power_of_2(N)

            # Detrend
            signal_centered = signal - np.mean(signal)

            # Zero-pad
            padded = np.zeros(N_padded)
            padded[:N] = signal_centered

            # FFT
            fft_vals = rfft(padded)
            power = np.abs(fft_vals) ** 2

            # Find peaks in power spectrum (skip DC component at index 0)
            if len(power) > 2:
                median_power = np.median(power[1:])
                prominence_thresh = 3.0 * median_power if median_power > 0 else 1.0
                peaks, properties = find_peaks(
                    power[1:], prominence=prominence_thresh
                )
                if len(peaks) > 0:
                    folio_is_periodic = True
                    channel_periodicity[ch_name] += 1
                    # Convert to periods (peaks are 0-indexed offset by 1)
                    for pk in peaks:
                        freq_idx = pk + 1  # actual index in power array
                        if freq_idx > 0:
                            period = float(N_padded) / float(freq_idx)
                            folio_periods.append(round(period, 2))

            # Autocorrelation
            max_lag = min(50, N // 2)
            if max_lag > 1:
                acf = _autocorrelation(signal, max_lag=max_lag)
                # Find peaks in ACF above 0.2 (skip lag 0)
                acf_above = acf[1:]
                acf_peaks, _ = find_peaks(acf_above, height=0.2)
                for ap in acf_peaks:
                    all_autocorr_peaks.append(float(acf_above[ap]))

        if folio_is_periodic:
            periodic_folio_ids.append(fol)
            # Keep top 5 periods by magnitude
            dominant_periods[fol] = sorted(set(folio_periods))[:5]

    mean_acf_peak = float(np.mean(all_autocorr_peaks)) if all_autocorr_peaks else 0.0

    print(f"  Analyzed {n_analyzed} folios (≥{MIN_TOKENS} tokens)")
    print(f"  Periodic folios: {len(periodic_folio_ids)}/{n_analyzed}")
    print(f"  Mean autocorrelation peak: {mean_acf_peak:.4f}")
    print(f"  Channel periodicity counts:")
    for ch, cnt in channel_periodicity.items():
        print(f"    {ch}: {cnt} folios")

    # Top periodic folios
    if periodic_folio_ids:
        print(f"  Top periodic folios (by # dominant periods):")
        top_periodic = sorted(
            periodic_folio_ids,
            key=lambda f: len(dominant_periods.get(f, [])),
            reverse=True,
        )[:10]
        for fol in top_periodic:
            periods = dominant_periods.get(fol, [])
            print(f"    {fol}: periods = {periods}")

    result = SpectralFFTResult(
        n_folios_analyzed=n_analyzed,
        n_periodic_folios=len(periodic_folio_ids),
        periodic_folio_ids=periodic_folio_ids,
        dominant_periods=dominant_periods,
        mean_autocorr_peak=round(mean_acf_peak, 6),
        channel_periodicity=channel_periodicity,
        runtime_seconds=round(time.time() - t0, 3),
    )

    path = _save_json(rd, 'spectral_fft.json', result)
    print(f"  Saved {path}")
    print(f"  Runtime: {result.runtime_seconds:.2f}s\n")


# ---------------------------------------------------------------------------
# Step 49B.3  Windowed Spectrogram (STFT)
# ---------------------------------------------------------------------------

def run_spectral_stft() -> None:
    """STFT on dict_hit channel to detect formula boundaries."""
    t0 = time.time()
    rd = _results_dir()

    print("=" * 60)
    print("Step 49B.3 – Windowed Spectrogram (STFT)")
    print("=" * 60)

    data_path = os.path.join(rd, 'spectral_signals_data.json')
    folio_signals = _safe_load(data_path)
    if not folio_signals:
        raise FileNotFoundError(f"Missing {data_path}")

    MIN_TOKENS_STFT = 32
    WINDOW_SIZE = 16
    n_analyzed = 0
    total_boundaries = 0
    formula_folios: List[str] = []
    folios_with_transitions: List[str] = []
    all_sharpness: List[float] = []

    for fol in sorted(folio_signals.keys()):
        channels = folio_signals[fol]
        signal = np.array(channels['dict_hit'], dtype=np.float64)
        N = len(signal)

        if N < MIN_TOKENS_STFT:
            continue

        n_analyzed += 1

        # Compute STFT
        nperseg = min(WINDOW_SIZE, N // 2)
        noverlap = min(nperseg // 2, N // 4)

        # Ensure nperseg >= 2
        if nperseg < 2:
            continue

        freqs, times, Zxx = stft(
            signal, fs=1.0, nperseg=nperseg, noverlap=noverlap
        )
        power_stft = np.abs(Zxx) ** 2

        # Compute spectral centroid at each time frame
        n_frames = power_stft.shape[1]
        centroids = np.zeros(n_frames)
        for t_idx in range(n_frames):
            frame_power = power_stft[:, t_idx]
            total_power = np.sum(frame_power)
            if total_power > 1e-12:
                centroids[t_idx] = np.sum(freqs * frame_power) / total_power
            else:
                centroids[t_idx] = 0.0

        # Detect transitions: centroid change > 1 std dev between adjacent
        if n_frames > 1 and np.std(centroids) > 1e-10:
            diffs = np.abs(np.diff(centroids))
            std_centroid = np.std(centroids)
            transition_mask = diffs > std_centroid
            n_transitions = int(np.sum(transition_mask))

            if n_transitions > 0:
                folios_with_transitions.append(fol)
                sharpness_vals = diffs[transition_mask] / (std_centroid + 1e-20)
                all_sharpness.extend(sharpness_vals.tolist())

                # Formula boundary: clusters of transitions
                # Count transition clusters as boundaries
                boundaries = 0
                in_transition = False
                for m in transition_mask:
                    if m and not in_transition:
                        boundaries += 1
                        in_transition = True
                    elif not m:
                        in_transition = False

                if boundaries > 0:
                    total_boundaries += boundaries
                    formula_folios.append(fol)
        else:
            pass  # no transitions detectable

    mean_sharpness = float(np.mean(all_sharpness)) if all_sharpness else 0.0

    print(f"  Analyzed {n_analyzed} folios (≥{MIN_TOKENS_STFT} tokens)")
    print(f"  Folios with transitions: {len(folios_with_transitions)}")
    print(f"  Formula boundary folios: {len(formula_folios)}")
    print(f"  Total formula boundaries: {total_boundaries}")
    print(f"  Mean transition sharpness: {mean_sharpness:.4f}")

    result = SpectralSTFTResult(
        n_folios_analyzed=n_analyzed,
        n_formula_boundaries=total_boundaries,
        formula_folios=formula_folios,
        n_folios_with_transitions=len(folios_with_transitions),
        mean_transition_sharpness=round(mean_sharpness, 6),
        stft_window_size=WINDOW_SIZE,
        runtime_seconds=round(time.time() - t0, 3),
    )

    path = _save_json(rd, 'spectral_stft.json', result)
    print(f"  Saved {path}")
    print(f"  Runtime: {result.runtime_seconds:.2f}s\n")


# ---------------------------------------------------------------------------
# Step 49B.4  Cross-Folio Section Analysis
# ---------------------------------------------------------------------------

def _normalize_spectrum(power: np.ndarray) -> np.ndarray:
    """L2-normalize a power spectrum."""
    norm = np.linalg.norm(power)
    if norm > 1e-12:
        return power / norm
    return power


def run_spectral_crossfolio() -> None:
    """Cross-folio spectral analysis grouped by manuscript section."""
    t0 = time.time()
    rd = _results_dir()

    print("=" * 60)
    print("Step 49B.4 – Cross-Folio Section Analysis")
    print("=" * 60)

    data_path = os.path.join(rd, 'spectral_signals_data.json')
    folio_signals = _safe_load(data_path)
    if not folio_signals:
        raise FileNotFoundError(f"Missing {data_path}")

    # Load corpus for section mapping
    corpus = load_corpus(verbose=False)

    MIN_TOKENS = 8
    SPECTRUM_LEN = 32  # fixed length for cross-folio comparison

    # Map folio -> section
    folio_section: Dict[str, str] = {}
    for fol_id, page in corpus.pages.items():
        folio_section[fol_id] = page.section

    # Compute normalized power spectra for dict_hit channel per folio
    folio_spectra: Dict[str, np.ndarray] = {}
    section_spectra: Dict[str, List[np.ndarray]] = defaultdict(list)

    for fol in sorted(folio_signals.keys()):
        channels = folio_signals[fol]
        signal = np.array(channels['dict_hit'], dtype=np.float64)
        N = len(signal)

        if N < MIN_TOKENS:
            continue

        # Compute FFT with fixed output length
        N_padded = _next_power_of_2(max(N, SPECTRUM_LEN * 2))
        signal_centered = signal - np.mean(signal)
        padded = np.zeros(N_padded)
        padded[:N] = signal_centered

        fft_vals = rfft(padded)
        power = np.abs(fft_vals) ** 2

        # Truncate/pad to SPECTRUM_LEN for uniform comparison
        if len(power) >= SPECTRUM_LEN:
            spectrum = power[1:SPECTRUM_LEN + 1]  # skip DC
        else:
            spectrum = np.zeros(SPECTRUM_LEN)
            n_copy = min(len(power) - 1, SPECTRUM_LEN)
            spectrum[:n_copy] = power[1:1 + n_copy]

        spectrum_normed = _normalize_spectrum(spectrum)
        folio_spectra[fol] = spectrum_normed

        section = folio_section.get(fol, 'unknown')
        section_spectra[section].append(spectrum_normed)

    n_folios_with_spectra = len(folio_spectra)
    print(f"  Computed spectra for {n_folios_with_spectra} folios")
    print(f"  Sections found: {sorted(section_spectra.keys())}")

    # Per-section spectral signatures
    section_sigs: Dict[str, Dict] = {}
    periodic_sections: List[str] = []

    for section in sorted(section_spectra.keys()):
        spectra_list = section_spectra[section]
        n_sec = len(spectra_list)
        if n_sec == 0:
            continue

        avg_spectrum = np.mean(spectra_list, axis=0)

        # Dominant frequency (index of max power in avg spectrum)
        dom_freq_idx = int(np.argmax(avg_spectrum))
        dom_power = float(avg_spectrum[dom_freq_idx])

        # Bandwidth (std of power distribution)
        total_power = np.sum(avg_spectrum)
        if total_power > 1e-12:
            freq_indices = np.arange(len(avg_spectrum))
            mean_freq = np.sum(freq_indices * avg_spectrum) / total_power
            bandwidth = float(np.sqrt(
                np.sum(((freq_indices - mean_freq) ** 2) * avg_spectrum)
                / total_power
            ))
        else:
            mean_freq = 0.0
            bandwidth = 0.0

        # Periodicity score: ratio of peak to mean power
        mean_power = float(np.mean(avg_spectrum))
        periodicity_score = dom_power / (mean_power + 1e-20)

        section_sigs[section] = {
            'n_folios': n_sec,
            'dominant_freq_index': dom_freq_idx,
            'dominant_power': round(dom_power, 6),
            'bandwidth': round(bandwidth, 4),
            'periodicity_score': round(periodicity_score, 4),
        }

        # A section is "periodic" if its periodicity_score > 3
        if periodicity_score > 3.0:
            periodic_sections.append(section)

        print(f"  Section '{section}': {n_sec} folios, "
              f"periodicity={periodicity_score:.2f}, "
              f"bandwidth={bandwidth:.2f}")

    # Spectral distance matrix and clustering
    folio_ids = sorted(folio_spectra.keys())
    n_fol = len(folio_ids)
    n_pairs = 0
    spectral_clustering_sil = 0.0
    n_clusters = 0

    if n_fol >= 5:
        # Build feature matrix
        X = np.array([folio_spectra[f] for f in folio_ids])

        # Determine number of clusters (min of 5 or unique sections)
        unique_sections = list(set(
            folio_section.get(f, 'unknown') for f in folio_ids
        ))
        n_clusters = min(5, len(unique_sections), n_fol - 1)
        n_clusters = max(2, n_clusters)

        # KMeans clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # Silhouette score requires at least 2 clusters with samples
            n_unique_labels = len(set(labels))
            if n_unique_labels >= 2:
                spectral_clustering_sil = float(silhouette_score(X, labels))
            else:
                spectral_clustering_sil = 0.0
        except Exception:
            spectral_clustering_sil = 0.0

        n_pairs = n_fol * (n_fol - 1) // 2

    # Section discriminability: how well does spectral clustering match sections?
    # Compute as adjusted Rand index approximation using silhouette
    section_discriminability = spectral_clustering_sil

    print(f"\n  Spectral clustering silhouette: {spectral_clustering_sil:.4f}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Periodic sections: {periodic_sections}")
    print(f"  Folio pairs analyzed: {n_pairs}")

    result = SpectralCrossfolioResult(
        section_spectral_signatures=section_sigs,
        section_discriminability=round(section_discriminability, 6),
        periodic_sections=periodic_sections,
        n_folio_pairs_correlated=n_pairs,
        spectral_clustering_silhouette=round(spectral_clustering_sil, 6),
        n_clusters=n_clusters,
        runtime_seconds=round(time.time() - t0, 3),
    )

    path = _save_json(rd, 'spectral_crossfolio.json', result)
    print(f"  Saved {path}")
    print(f"  Runtime: {result.runtime_seconds:.2f}s\n")


# ---------------------------------------------------------------------------
# Track B runner
# ---------------------------------------------------------------------------

def run_track_b_49() -> None:
    """Run all Track B steps sequentially."""
    run_spectral_signals()
    run_spectral_fft()
    run_spectral_stft()
    run_spectral_crossfolio()
