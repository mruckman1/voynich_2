"""
Step 34.19 – Phase 34 Integration
====================================
Compares all encoding model tracks and identifies which model best fits
the Voynich manuscript.  The decisive metric is SIGNAL bigram z-score.

Dependency chain:
    dict_calibration.json     (Track G)
    abjad_signal.json         (Track A)
    slot_signal.json          (Track B)
    dialect_signal.json       (Track C)
    resegment_signal.json     (Track D)
    spatial_decode.json       (Track E)
    vowel_decode.json         (Track F)
    signal_bigrams.json       (Phase 29 baseline)
        → phase34_integrate.json  (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voynich.core._paths import results_dir as _results_dir


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackSummary:
    track: str
    model: str
    dict_hit: float
    selectivity: float
    signal_rate: float
    bigram_z: float
    key_finding: str
    beats_baseline: bool


@dataclass
class Phase34IntegrateResult:
    # Baseline
    baseline_dict_hit: float
    baseline_selectivity: float
    baseline_signal_rate: float
    baseline_bigram_z: float

    # Per-track summaries
    track_summaries: List[Dict]

    # Track ranking
    best_track: str
    best_model: str
    best_bigram_z: float
    best_signal_rate: float

    # Track interactions tested
    interactions_tested: List[Dict]

    # Model selection
    verdict: str  # TRACK_X_WINS / BASELINE_HOLDS
    improvement_over_baseline: Dict[str, float]

    # Gap analysis
    n_encoding_variables_confirmed: int
    signal_fraction: float
    next_steps: List[str]

    runtime_seconds: float


# ---------------------------------------------------------------------------
# Track result loading
# ---------------------------------------------------------------------------

def _load_track_result(
    rd: str,
    filename: str,
    signal_keys: Tuple[str, ...] = ('signal_rate', 'bigram_z'),
    dict_hit_key: str = 'dict_hit_rate',
    selectivity_key: str = 'selectivity',
) -> Optional[Dict[str, float]]:
    """Load a track's key metrics from its JSON result file."""
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)

    metrics: Dict[str, float] = {}
    for key in (dict_hit_key, selectivity_key) + signal_keys:
        val = data.get(key)
        if val is not None:
            metrics[key] = float(val)

    # Also try nested keys
    if 'dict_hit' not in metrics and 'dict_hit_rate' in data:
        metrics['dict_hit'] = float(data['dict_hit_rate'])
    if 'selectivity' not in metrics:
        metrics['selectivity'] = float(data.get('selectivity', 0.0))
    if 'signal_rate' not in metrics:
        metrics['signal_rate'] = float(data.get('signal_rate', 0.0))
    if 'bigram_z' not in metrics:
        for key in ('bigram_z', 'bigram_z_score'):
            if key in data:
                metrics['bigram_z'] = float(data[key])
                break

    # Extract key finding / verdict
    metrics['verdict'] = data.get('verdict', data.get('abjad_vs_cv', ''))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase34_integrate() -> None:
    """Step 34.19: Phase 34 integration."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.19: Phase 34 Integration")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load baseline ──
    print("\n  1. Loading baseline (Phase 29) …")
    baseline = {
        'dict_hit': 0.436,
        'selectivity': 1.37,
        'signal_rate': 0.165,
        'bigram_z': 6.14,
    }
    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(bg_path):
        with open(bg_path) as f:
            bg = json.load(f)
        baseline['signal_rate'] = bg.get('signal_rate', 0.165)
        baseline['bigram_z'] = bg.get('bigram_z_score', 6.14)
        baseline['dict_hit'] = bg.get('n_signal', 0) / max(bg.get('n_tokens', 1), 1)

    print(f"     Baseline: dict_hit={baseline['dict_hit']:.3f}"
          f"  SIGNAL={baseline['signal_rate']:.3f}  z={baseline['bigram_z']:.2f}")

    # ── 2. Load track results ──
    print("\n  2. Loading track results …")

    track_configs = [
        ('G', 'Dict right-sized', 'dict_calibration.json',
         'optimal_signal_rate', 'optimal_bigram_z', 'optimal_dict_hit'),
        ('A', 'Abjad consonant', 'abjad_signal.json',
         'signal_rate', 'bigram_z', 'dict_hit_rate'),
        ('B', 'Slot-conditioned', 'slot_signal.json',
         'signal_rate', 'bigram_z_score', 'dict_hit_rate'),
        ('C', 'Latin-Italian dialect', 'dialect_signal.json',
         'signal_rate', 'bigram_z', 'dict_hit_rate'),
        ('D', 'Scripta continua', 'resegment_signal.json',
         'real_signal_rate', 'bigram_z_score', 'real_dict_hit'),
        ('E', '2D spatial', 'spatial_decode.json',
         'spatial_signal_rate', 'chi2_z_score', 'overall_constrained_dict_hit'),
        ('F', 'Vowel pointers', 'vowel_decode.json',
         'signal_rate', 'bigram_z', 'merged_dict_hit'),
    ]

    summaries: List[TrackSummary] = []

    for track, model, filename, sig_key, z_key, dh_key in track_configs:
        path = os.path.join(rd, filename)
        if not os.path.exists(path):
            print(f"     Track {track} ({model}): [NOT FOUND] {filename}")
            summaries.append(TrackSummary(
                track=track, model=model,
                dict_hit=0.0, selectivity=0.0,
                signal_rate=0.0, bigram_z=0.0,
                key_finding='NOT_RUN',
                beats_baseline=False,
            ))
            continue

        with open(path) as f:
            data = json.load(f)

        signal_rate = float(data.get(sig_key, data.get('signal_rate', 0.0)))
        bigram_z = float(data.get(z_key, data.get('bigram_z', data.get('bigram_z_score', 0.0))))
        dict_hit = float(data.get(dh_key, data.get('dict_hit_rate', data.get('optimal_dict_hit', 0.0))))
        selectivity = float(data.get('selectivity', data.get('optimal_selectivity', 0.0)))
        verdict = data.get('verdict', data.get('abjad_vs_cv', ''))

        # Cap degenerate z-scores (e.g. z=999 from div-by-zero)
        if bigram_z > 100.0:
            bigram_z = min(bigram_z, 100.0)

        beats = bigram_z > baseline['bigram_z'] and signal_rate >= 0.05

        summaries.append(TrackSummary(
            track=track, model=model,
            dict_hit=round(dict_hit, 4),
            selectivity=round(selectivity, 4),
            signal_rate=round(signal_rate, 4),
            bigram_z=round(bigram_z, 2),
            key_finding=str(verdict),
            beats_baseline=beats,
        ))

        status = "BEATS BASELINE" if beats else "below baseline"
        print(f"     Track {track} ({model}): dict_hit={dict_hit:.3f}"
              f"  SIGNAL={signal_rate:.3f}  z={bigram_z:.2f}  [{status}]")

    # ── 3. Find best track ──
    print("\n  3. Model selection …")
    valid_summaries = [s for s in summaries if s.key_finding != 'NOT_RUN']
    # Only consider tracks with meaningful signal (>= 5% signal rate)
    meaningful = [s for s in valid_summaries if s.signal_rate >= 0.05]
    if meaningful:
        best = max(meaningful, key=lambda s: s.bigram_z)
    elif valid_summaries:
        best = max(valid_summaries, key=lambda s: s.bigram_z)
    else:
        best = TrackSummary(
            track='BASELINE', model='CV syllable',
            dict_hit=baseline['dict_hit'],
            selectivity=baseline['selectivity'],
            signal_rate=baseline['signal_rate'],
            bigram_z=baseline['bigram_z'],
            key_finding='NO_TRACKS_RUN',
            beats_baseline=False,
        )

    if best.bigram_z > baseline['bigram_z']:
        verdict = f'TRACK_{best.track}_WINS'
        print(f"     WINNER: Track {best.track} ({best.model})"
              f" — z={best.bigram_z:.2f} > baseline {baseline['bigram_z']:.2f}")
    else:
        verdict = 'BASELINE_HOLDS'
        print(f"     BASELINE HOLDS — no track exceeds z={baseline['bigram_z']:.2f}")
        print(f"     Best was Track {best.track} ({best.model}) z={best.bigram_z:.2f}")

    # ── 4. Track interactions ──
    print("\n  4. Track interactions …")
    interactions: List[Dict] = []
    # Note: interaction testing would require re-decoding with combined models
    # For now, identify promising combinations based on individual results
    winning_tracks = [s for s in summaries if s.beats_baseline]
    if len(winning_tracks) >= 2:
        interactions.append({
            'combination': f"{winning_tracks[0].track}+{winning_tracks[1].track}",
            'models': f"{winning_tracks[0].model} + {winning_tracks[1].model}",
            'note': 'Both beat baseline individually — combination worth testing',
        })

    # Always suggest A+F if both have results
    a_summary = next((s for s in summaries if s.track == 'A' and s.key_finding != 'NOT_RUN'), None)
    f_summary = next((s for s in summaries if s.track == 'F' and s.key_finding != 'NOT_RUN'), None)
    if a_summary and f_summary:
        interactions.append({
            'combination': 'A+F',
            'models': 'Abjad consonants + Vowel pointers',
            'note': 'Complementary models: consonants from roots + vowels from pointers',
            'combined_potential': 'HIGH' if a_summary.signal_rate > 0.05 else 'LOW',
        })

    print(f"     {len(interactions)} potential combinations identified")

    # ── 5. Gap analysis ──
    print("\n  5. Gap analysis …")
    best_signal = max((s.signal_rate for s in summaries if s.key_finding != 'NOT_RUN'), default=0.0)
    n_confirmed = sum(1 for s in summaries if s.beats_baseline)

    next_steps = []
    if verdict == 'BASELINE_HOLDS':
        next_steps.append("CV syllable model remains best — focus on improving within current model")
        next_steps.append("Consider hybrid models combining successful track elements")
    else:
        next_steps.append(f"Track {best.track} ({best.model}) exceeded baseline")
        next_steps.append("Test track combinations for further improvement")
        next_steps.append("Run full readability battery on winning model")

    if any(s.track == 'G' and s.key_finding != 'NOT_RUN' for s in summaries):
        g_summary = next(s for s in summaries if s.track == 'G')
        if g_summary.signal_rate > baseline['signal_rate']:
            next_steps.append(
                f"Apply optimal dictionary (Track G) to all future analyses"
            )

    improvement = {
        'bigram_z_delta': round(best.bigram_z - baseline['bigram_z'], 2),
        'signal_rate_delta': round(best.signal_rate - baseline['signal_rate'], 4),
        'dict_hit_delta': round(best.dict_hit - baseline['dict_hit'], 4),
    }

    elapsed = time.time() - t0

    result = Phase34IntegrateResult(
        baseline_dict_hit=baseline['dict_hit'],
        baseline_selectivity=baseline['selectivity'],
        baseline_signal_rate=baseline['signal_rate'],
        baseline_bigram_z=baseline['bigram_z'],
        track_summaries=[asdict(s) for s in summaries],
        best_track=best.track,
        best_model=best.model,
        best_bigram_z=best.bigram_z,
        best_signal_rate=best.signal_rate,
        interactions_tested=interactions,
        verdict=verdict,
        improvement_over_baseline=improvement,
        n_encoding_variables_confirmed=n_confirmed,
        signal_fraction=round(best_signal, 4),
        next_steps=next_steps,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'phase34_integrate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  Saved → {out_path}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("PHASE 34 SUMMARY — ENCODING MODEL COMPARISON")
    print("=" * 70)
    print(f"\n  {'Track':<8s} {'Model':<22s} {'DictHit':>8s} {'SIGNAL':>8s}"
          f" {'z':>7s} {'Status':<15s}")
    print("  " + "-" * 68)
    print(f"  {'BASE':<8s} {'CV syllable':<22s} {baseline['dict_hit']:>8.3f}"
          f" {baseline['signal_rate']:>8.3f} {baseline['bigram_z']:>7.2f}"
          f" {'─── baseline ───':<15s}")

    for s in summaries:
        if s.key_finding == 'NOT_RUN':
            status = 'NOT RUN'
        elif s.beats_baseline:
            status = 'BEATS BASELINE'
        else:
            status = 'below'
        print(f"  {s.track:<8s} {s.model:<22s} {s.dict_hit:>8.3f}"
              f" {s.signal_rate:>8.3f} {s.bigram_z:>7.2f} {status:<15s}")

    print(f"\n  VERDICT: {verdict}")
    print(f"  Completed in {elapsed:.1f}s")
