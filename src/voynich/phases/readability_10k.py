"""
Step 36.7 – Full Readability Battery at 10K
=============================================
The definitive cross-phase comparison table using the project's strongest
signal configuration (10K dictionary, unconditioned Phase 16 decode).

Dependency chain:
    decode_10k.json           (Step 36.1)
    signal_10k.json           (Step 36.2)
    bigrams_10k.json          (Step 36.3)
    context_10k.json          (Step 36.4)
    bootstrap_10k.json        (Step 36.5)
    folio_10k.json            (Step 36.6)
        → readability_10k.json  (this step)
"""

import json
import os
import time
from typing import Any, Dict, List

from voynich.core._paths import results_dir as _results_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Load JSON if it exists, otherwise return empty dict."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_readability_10k() -> None:
    """Step 36.7: Full readability battery at 10K dictionary."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 36.7: Full Readability Battery at 10K")
    print("=" * 70)

    rd = _results_dir()

    # ── Load all Phase 36 results ──
    print("\n  Loading Phase 36 results …")
    decode = _safe_load(os.path.join(rd, 'decode_10k.json'))
    signal = _safe_load(os.path.join(rd, 'signal_10k.json'))
    bigrams = _safe_load(os.path.join(rd, 'bigrams_10k.json'))
    context = _safe_load(os.path.join(rd, 'context_10k.json'))
    bootstrap = _safe_load(os.path.join(rd, 'bootstrap_10k.json'))
    folio = _safe_load(os.path.join(rd, 'folio_10k.json'))

    # ── 12 readability tests ──
    tests = []

    # V1: Dict-hit rate at 10K
    dict_hit_10k = decode.get('dict_hit_10k', 0.0)
    tests.append({
        'id': 'V1', 'name': 'Dict-hit rate (10K)',
        'value': round(dict_hit_10k, 4),
        'threshold': 0.20,
        'passed': dict_hit_10k >= 0.20,
    })

    # V2: SIGNAL rate at 10K
    signal_rate = signal.get('signal_rate', 0.0)
    tests.append({
        'id': 'V2', 'name': 'SIGNAL rate (10K)',
        'value': round(signal_rate, 4),
        'threshold': 0.15,
        'passed': signal_rate >= 0.15,
    })

    # V3: Bigram z at 10K
    bigram_z = bigrams.get('bigram_z', 0.0)
    tests.append({
        'id': 'V3', 'name': 'Bigram z-score (10K)',
        'value': round(bigram_z, 2),
        'threshold': 10.0,
        'passed': bigram_z >= 10.0,
    })

    # V4: Trigram hits
    n_trigram = bigrams.get('n_trigram_hits', 0)
    tests.append({
        'id': 'V4', 'name': 'Trigram hits',
        'value': n_trigram,
        'threshold': 0,
        'passed': True,  # informational
    })

    # V5: Confirmed vocabulary size at 10K
    n_genuine = signal.get('n_genuine_signal_words', 0)
    tests.append({
        'id': 'V5', 'name': 'Confirmed vocabulary (10K)',
        'value': n_genuine,
        'threshold': 5,
        'passed': n_genuine >= 5,
    })

    # V6: Longest SIGNAL run
    longest_run = folio.get('longest_run_overall', 0)
    tests.append({
        'id': 'V6', 'name': 'Longest SIGNAL run',
        'value': longest_run,
        'threshold': 3,
        'passed': longest_run >= 3,
    })

    # V7: Parseable Latin fragments
    n_parseable = folio.get('n_parseable_fragments', 0)
    tests.append({
        'id': 'V7', 'name': 'Parseable Latin fragments',
        'value': n_parseable,
        'threshold': 1,
        'passed': n_parseable >= 1,
    })

    # V8: Net signal (SIGNAL - ANTI at 10K)
    net_signal = signal.get('net_signal', 0.0)
    tests.append({
        'id': 'V8', 'name': 'Net signal (10K)',
        'value': round(net_signal, 4),
        'threshold': 0.10,
        'passed': net_signal >= 0.10,
    })

    # V9: Selectivity at 10K
    sel_10k = decode.get('selectivity_10k', 0.0)
    tests.append({
        'id': 'V9', 'name': 'Selectivity (10K)',
        'value': round(sel_10k, 2),
        'threshold': 1.3,
        'passed': sel_10k >= 1.3,
    })

    # V10: Content-content bigrams
    n_cc = bigrams.get('n_content_content', 0)
    tests.append({
        'id': 'V10', 'name': 'Content-content bigrams',
        'value': n_cc,
        'threshold': 1,
        'passed': n_cc >= 1,
    })

    # V11: Confirmed-confirmed pairs
    n_conf_pairs = context.get('n_confirmed_pairs', 0)
    tests.append({
        'id': 'V11', 'name': 'Confirmed-confirmed pairs',
        'value': n_conf_pairs,
        'threshold': 1,
        'passed': n_conf_pairs >= 1,
    })

    # V12: Medical collocations
    n_medical = context.get('n_medical_total', 0)
    tests.append({
        'id': 'V12', 'name': 'Medical collocations',
        'value': n_medical,
        'threshold': 1,
        'passed': n_medical >= 1,
    })

    n_passed = sum(1 for t in tests if t['passed'])

    # ── Print test results ──
    print("\n  Readability Tests:")
    print(f"  {'ID':<5s} {'Test':<30s} {'Value':>8s} {'Thresh':>8s} {'Result':>8s}")
    print("  " + "-" * 59)
    for t in tests:
        result = 'PASS' if t['passed'] else 'FAIL'
        val_str = str(t['value'])
        thr_str = str(t['threshold'])
        print(f"  {t['id']:<5s} {t['name']:<30s} {val_str:>8s} {thr_str:>8s} {result:>8s}")
    print(f"\n  {n_passed}/{len(tests)} passed")

    # ── Progression table ──
    print("\n  DEFINITIVE PROGRESSION TABLE:")
    print(f"  {'Phase':<8s} {'Dict-hit':>8s} {'Dict':>6s} {'SIGNAL':>8s} {'Bigram z':>9s}"
          f" {'Confirm':>8s} {'Key advance':<30s}")
    print("  " + "-" * 80)

    progression = [
        ('14', '19.4%', '17K', '—', '—', '—', 'Stroke-triple model'),
        ('16', '43.6%', '131K', '—', '—', '—', 'Modifier handling'),
        ('28', '43.6%', '131K', '16.5%', '—', '8', 'Signal isolation'),
        ('29', '43.6%', '131K', '16.5%', '6.14', '8', 'Bigram discovery'),
        ('30', '43.6%', '131K', '16.5%', '6.14', '10', 'Bootstrap (+2)'),
        ('34G', '22.7%', '10K', '18.7%', '13.12', '—', 'Dict right-sizing'),
    ]
    for row in progression:
        print(f"  {row[0]:<8s} {row[1]:>8s} {row[2]:>6s} {row[3]:>8s} {row[4]:>9s}"
              f" {row[5]:>8s} {row[6]:<30s}")

    # Phase 36 row
    n_boot = bootstrap.get('n_total_accepted', 0)
    n_confirmed = n_genuine + n_boot
    print(f"  {'36':>8s} {dict_hit_10k*100:>7.1f}% {'10K':>6s} {signal_rate*100:>7.1f}%"
          f" {bigram_z:>9.2f} {n_confirmed:>8d} {'Full 10K pipeline':<30s}")

    # ── Bootstrap comparison ──
    boot_verdict = bootstrap.get('verdict', 'N/A')
    boot_accepted = bootstrap.get('accepted_words', [])
    print(f"\n  Bootstrap verdict: {boot_verdict}")
    if boot_accepted:
        print(f"  Bootstrap words: {boot_accepted}")

    # ── Save ──
    elapsed = time.time() - t0

    output = {
        'tests': tests,
        'n_passed': n_passed,
        'n_total': len(tests),
        # Key metrics
        'dict_hit_10k': round(dict_hit_10k, 4),
        'signal_rate_10k': round(signal_rate, 4),
        'bigram_z_10k': round(bigram_z, 2),
        'net_signal_10k': round(net_signal, 4),
        'selectivity_10k': round(sel_10k, 2),
        'n_genuine_signal_words': n_genuine,
        'n_bootstrap_words': n_boot,
        'n_confirmed_total': n_confirmed,
        'n_content_content_bigrams': n_cc,
        'n_medical_collocations': n_medical,
        'n_confirmed_pairs': n_conf_pairs,
        'longest_run': longest_run,
        'n_trigram_hits': n_trigram,
        'n_parseable_fragments': n_parseable,
        'bootstrap_verdict': boot_verdict,
        'bootstrap_words': boot_accepted,
        # For verdict
        'best_fragment': folio.get('best_fragment'),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'readability_10k.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"  Runtime: {elapsed:.1f}s")
