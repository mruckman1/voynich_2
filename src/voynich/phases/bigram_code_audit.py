"""
Step 42.1 – Bigram Code Audit
==============================
Inspect every script that computed a bigram z-score to determine whether
each used symmetric comparison (same hit-counting method for real and null).

Phase 41 found that Phase 40's venetian_bigrams.py compared real
(exact + relaxed) hits against null (exact-only) permutation hits,
inflating z from -0.47 to 319.76.  This step audits all 8 bigram
scripts for the same asymmetry pattern.

Dependency chain:
    signal_bigrams.json, combined_bigrams.json, bigrams_10k.json,
    concat_bigrams.json, merged_bigrams.json, corrected_signal.json,
    amplified_bigrams.json, venetian_bigrams.json
        → bigram_code_audit.json  (this step)
"""

import json
import os
import time
from typing import Any, Dict, List

from voynich.core._paths import results_dir as _results_dir


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
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


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────
# Audit definitions
# ─────────────────────────────────────────────────────────────────

# Each entry encodes the findings from manual code reading of the script.
# Fields:
#   script        – source file name
#   phase         – phase label (e.g. "29")
#   step          – step label (e.g. "29.1")
#   results_file  – JSON output of that script
#   z_field       – key in results JSON containing the z-score
#   real_counting – how real bigram hits are counted
#   null_counting – how null permutation hits are counted
#   null_model    – the null model type (relabel or shuffle)
#   z_formula     – what goes into the z numerator
#   symmetric     – whether the z comparison is apples-to-apples
#   status        – VALID | NEEDS_INSPECTION | BUGGED
#   notes         – explanation

_AUDIT_DEFINITIONS = [
    {
        'script': 'signal_bigrams.py',
        'phase': '29',
        'step': '29.1',
        'results_file': 'signal_bigrams.json',
        'z_field': 'bigram_z_score',
        'real_counting': 'exact_only',
        'null_counting': 'exact_only',
        'null_model': 'relabel_rate',
        'z_formula': '(exact_rate - null_mean_rate) / null_std_rate',
        'symmetric': True,
        'status': 'VALID',
        'notes': (
            'Real counts exact-only bigram hits among SIGNAL-SIGNAL pairs. '
            'Null relabels n_signal random positions, counts exact-only hits '
            'among consecutive same-folio SIGNAL pairs. Both use rate '
            '(hits/pairs). Symmetric.'
        ),
    },
    {
        'script': 'combined_bigrams.py',
        'phase': '35',
        'step': '35.4',
        'results_file': 'combined_bigrams.json',
        'z_field': 'bigram_z_score',
        'real_counting': 'exact_only',
        'null_counting': 'exact_only',
        'null_model': 'relabel_rate',
        'z_formula': '(exact_rate - null_mean_rate) / null_std_rate',
        'symmetric': True,
        'status': 'VALID',
        'notes': (
            'Reuses _null_permutation_test() from signal_bigrams.py. '
            'Both real and null count exact-only, rate-based. Symmetric.'
        ),
    },
    {
        'script': 'bigrams_10k.py',
        'phase': '36',
        'step': '36.3',
        'results_file': 'bigrams_10k.json',
        'z_field': 'bigram_z',
        'real_counting': 'exact_only',
        'null_counting': 'exact_only',
        'null_model': 'relabel_rate',
        'z_formula': '(exact_rate - null_mean_rate) / null_std_rate',
        'symmetric': True,
        'status': 'VALID',
        'notes': (
            'Uses _null_permutation_test() from signal_bigrams.py with '
            '10K Latin dictionary. Both real and null count exact-only, '
            'rate-based. Symmetric.'
        ),
    },
    {
        'script': 'concat_bigrams.py',
        'phase': '37.6',
        'step': '37.6',
        'results_file': 'concat_bigrams.json',
        'z_field': 'merged_bigram_z',
        'real_counting': 'exact_only',
        'null_counting': 'exact_only',
        'null_model': 'shuffle_count',
        'z_formula': '(exact_count - null_mean_count) / null_std_count',
        'symmetric': True,
        'status': 'VALID',
        'notes': (
            'Different null model (word-shuffle instead of relabel) but '
            'both real and null count exact-only. Symmetric within its '
            'own methodology.'
        ),
    },
    {
        'script': 'merged_bigrams.py',
        'phase': '38',
        'step': '38.4',
        'results_file': 'merged_bigrams.json',
        'z_field': 'bigram_z',
        'real_counting': 'exact_plus_relaxed_tallied_separately',
        'null_counting': 'exact_only',
        'null_model': 'shuffle_count',
        'z_formula': '(exact_hits - null_mean_count) / null_std_count',
        'symmetric': True,
        'status': 'NEEDS_INSPECTION',
        'notes': (
            'Real tallies exact (12) and relaxed (1759) separately. '
            'The z formula uses exact_hits only (line 334). Null shuffle '
            'also counts exact-only. So z itself IS symmetric for exact '
            'counting. But the large relaxed count alongside could mislead '
            'interpretation, and the shuffle null model differs from the '
            'relabel model used in Phases 29/35/36. Recompute to verify.'
        ),
    },
    {
        'script': 'corrected_signal.py',
        'phase': '39.4',
        'step': '39.4',
        'results_file': 'corrected_signal.json',
        'z_field': 'bigram_z',
        'real_counting': 'exact_plus_relaxed_tallied_separately',
        'null_counting': 'exact_only',
        'null_model': 'shuffle_count',
        'z_formula': '(exact_bigram_hits - null_mean_count) / null_std_count',
        'symmetric': True,
        'status': 'NEEDS_INSPECTION',
        'notes': (
            'Same pattern as merged_bigrams.py. Real tallies exact (10) '
            'and relaxed (807) separately. z uses exact_bigram_hits only '
            '(line 316). Null counts exact-only. Symmetric for exact but '
            'uses shuffle null model. Recompute to verify.'
        ),
    },
    {
        'script': 'amplified_bigrams.py',
        'phase': '39.16',
        'step': '39.16',
        'results_file': 'amplified_bigrams.json',
        'z_field': 'bigram_z',
        'real_counting': 'exact_plus_relaxed_tallied_separately',
        'null_counting': 'exact_only',
        'null_model': 'shuffle_count',
        'z_formula': '(exact_hits - null_mean_count) / null_std_count',
        'symmetric': True,
        'status': 'NEEDS_INSPECTION',
        'notes': (
            'Same pattern. Real: exact=17, relaxed=2922. z uses exact_hits '
            'only (line 239). Null counts exact-only. Symmetric for exact '
            'but z=19.89 seems high for only 17 exact hits — need to verify '
            'null baseline (calibrated bigram set may be very small, making '
            'random matches extremely rare). Recompute to verify.'
        ),
    },
    {
        'script': 'venetian_bigrams.py',
        'phase': '40',
        'step': '40.3',
        'results_file': 'venetian_bigrams.json',
        'z_field': 'bigram_z',
        'real_counting': 'exact_plus_relaxed_combined',
        'null_counting': 'exact_only',
        'null_model': 'shuffle_count',
        'z_formula': '(exact + relaxed - null_mean_count) / null_std_count',
        'symmetric': False,
        'status': 'BUGGED',
        'notes': (
            'CONFIRMED BUG (Phase 41). real_hits = exact_hits + relaxed_hits '
            '(line 203), but null permutation only counts exact (lines '
            '211-215). z = (real_total - null_mean) / null_std = 319.76. '
            'Phase 41 recomputed with symmetric null: z_exact=1.33, '
            'z_total=-0.47. The 319.76 was entirely artifactual.'
        ),
    },
]


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def run_bigram_code_audit() -> None:
    """Step 42.1: Audit all bigram z-score computation scripts."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 42.1: Bigram Code Audit")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load reported z-scores from results files ──
    print("\n  1. Loading reported z-scores from results files …")

    audits: List[Dict] = []
    for defn in _AUDIT_DEFINITIONS:
        data = _safe_load(os.path.join(rd, defn['results_file']))
        reported_z = data.get(defn['z_field'], None)
        if reported_z is not None:
            reported_z = round(float(reported_z), 4)

        audit = {
            'script': defn['script'],
            'phase': defn['phase'],
            'step': defn['step'],
            'results_file': defn['results_file'],
            'z_field': defn['z_field'],
            'reported_z': reported_z,
            'real_counting': defn['real_counting'],
            'null_counting': defn['null_counting'],
            'null_model': defn['null_model'],
            'z_formula': defn['z_formula'],
            'symmetric': defn['symmetric'],
            'status': defn['status'],
            'notes': defn['notes'],
        }
        audits.append(audit)

        status_str = defn['status']
        sym_str = "symmetric" if defn['symmetric'] else "ASYMMETRIC"
        z_str = f"z={reported_z}" if reported_z is not None else "z=N/A"
        print(f"    Phase {defn['phase']:>5s}  {defn['script']:<28s}  "
              f"{z_str:>14s}  {sym_str:<12s}  {status_str}")

    # ── 2. Summary ──
    print("\n  2. Summary")

    n_valid = sum(1 for a in audits if a['status'] == 'VALID')
    n_bugged = sum(1 for a in audits if a['status'] == 'BUGGED')
    n_needs_inspection = sum(1 for a in audits
                            if a['status'] == 'NEEDS_INSPECTION')

    print(f"    VALID:            {n_valid}")
    print(f"    NEEDS_INSPECTION: {n_needs_inspection}")
    print(f"    BUGGED:           {n_bugged}")
    print(f"    Total:            {len(audits)}")

    # ── 3. Key findings ──
    print("\n  3. Key findings")

    findings = []

    # Phases 29, 35, 36 use relabel + exact-only → should be unaffected
    findings.append(
        "Phases 29, 35, 36: Use relabel null model with exact-only "
        "counting for both real and null. These z-scores (6.14, 6.88, "
        "12.66) should be unaffected by the asymmetry bug."
    )

    # Phase 37.6 uses shuffle + exact-only → also unaffected
    findings.append(
        "Phase 37.6: Uses shuffle null model with exact-only counting. "
        "z=-6.67 (no signal). Unaffected."
    )

    # Phases 38, 39.4, 39.16 tally relaxed but z uses exact-only
    findings.append(
        "Phases 38, 39.4, 39.16: Tally relaxed hits separately but "
        "z formula uses exact_hits only. The z-score computation itself "
        "appears symmetric, but uses a different null model (shuffle vs "
        "relabel). Step 42.2 will recompute with canonical methodology."
    )

    # Phase 40 is the confirmed bug
    findings.append(
        "Phase 40: CONFIRMED BUG. z=319.76 was real(exact+relaxed) vs "
        "null(exact-only). Phase 41 corrected to z_total=-0.47."
    )

    for i, f in enumerate(findings, 1):
        print(f"    [{i}] {f}")

    # ── 4. Verdict ──
    if n_bugged > 0:
        verdict = "BUGS_FOUND"
    elif n_needs_inspection > 0:
        verdict = "INSPECTION_NEEDED"
    else:
        verdict = "ALL_VALID"

    print(f"\n  4. VERDICT: {verdict}")

    # ── 5. Save ──
    elapsed = time.time() - t0

    output = {
        'script_audits': audits,
        'n_scripts_audited': len(audits),
        'n_valid': n_valid,
        'n_bugged': n_bugged,
        'n_needs_inspection': n_needs_inspection,
        'key_findings': findings,
        'verdict': verdict,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'bigram_code_audit.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
