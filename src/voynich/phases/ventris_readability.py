"""
Phase 28.8 – Ventris Readability Battery
==========================================
8-point validation battery for the Ventris decode output.

Dependency chain:
    ventris_decode.json       (Step 28.7)
    signal_isolation.json     (Step 28.4)
    crib_localization.json    (Step 28.5)
    modifier_integrate.json   (Phase 16)
        → ventris_readability.json  (this step)
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import load_reference_corpus


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
# Validation helpers
# ---------------------------------------------------------------------------

def _bigram_jsd(decoded_words: List[str], ref_words: List[str]) -> float:
    """Jensen-Shannon divergence between character bigram distributions."""
    def _bigram_dist(words):
        counts = Counter()
        total = 0
        for w in words:
            for i in range(len(w) - 1):
                counts[w[i:i+2]] += 1
                total += 1
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    p = _bigram_dist(decoded_words)
    q = _bigram_dist(ref_words)

    all_keys = set(p) | set(q)
    if not all_keys:
        return 1.0

    # M = (P + Q) / 2
    m = {}
    for k in all_keys:
        m[k] = (p.get(k, 0) + q.get(k, 0)) / 2

    def _kl(a, b):
        total = 0.0
        for k in a:
            if a[k] > 0 and b.get(k, 0) > 0:
                total += a[k] * math.log2(a[k] / b[k])
        return total

    return (_kl(p, m) + _kl(q, m)) / 2


def _section_chi_sq(section_stats: List[Dict]) -> float:
    """Chi-squared test for section variation in dict_hit."""
    rates = [s.get('dict_hit', 0) for s in section_stats if s.get('n_tokens', 0) > 50]
    if len(rates) < 2:
        return 0.0
    mean_rate = sum(rates) / len(rates)
    if mean_rate == 0:
        return 0.0
    # Use n_tokens as weights
    chi_sq = 0.0
    for s in section_stats:
        n = s.get('n_tokens', 0)
        if n < 50:
            continue
        observed = s.get('dict_hit', 0) * n
        expected = mean_rate * n
        if expected > 0:
            chi_sq += (observed - expected) ** 2 / expected
    return chi_sq


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Validation:
    id: str
    name: str
    value: float
    threshold: float
    passed: bool
    note: str


@dataclass
class ReadabilityResult:
    validations: List[Dict]
    n_passed: int
    n_total: int
    pass_rate: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ventris_readability() -> None:
    """Step 28.8: Readability battery for Ventris decode."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 28.8: Ventris Readability Battery")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    decode_path = os.path.join(rd, 'ventris_decode.json')
    if not os.path.exists(decode_path):
        print("  [SKIP] ventris_decode.json not found")
        return
    with open(decode_path) as f:
        decode_data = json.load(f)

    signal_path = os.path.join(rd, 'signal_isolation.json')
    signal_data = {}
    if os.path.exists(signal_path):
        with open(signal_path) as f:
            signal_data = json.load(f)

    local_path = os.path.join(rd, 'crib_localization.json')
    local_data = {}
    if os.path.exists(local_path):
        with open(local_path) as f:
            local_data = json.load(f)

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    mod_data = {}
    if os.path.exists(mod_path):
        with open(mod_path) as f:
            mod_data = json.load(f)

    # ── 2. Prepare reference data ──
    print("\n  2. Preparing reference data …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    ref_words = [w.lower() for w in ref_corpus.get_combined_tokens('latin')
                 if len(w) >= 2]

    # Extract decoded words from sample
    decoded_words = []
    for pair in decode_data.get('sample_decoded_herbal', []):
        if len(pair) >= 2:
            decoded_words.append(pair[1])
    for pair in decode_data.get('sample_decoded_pharma', []):
        if len(pair) >= 2:
            decoded_words.append(pair[1])

    # For bigram JSD, use the best passage text as a proxy
    passage_text = decode_data.get('best_passage_text', '')
    passage_words = passage_text.split() if passage_text else decoded_words

    # ── 3. Run validations ──
    print("\n  3. Running validations …")
    validations: List[Validation] = []

    # V1: dict_hit ≥ 0.40
    dict_hit = decode_data.get('corpus_dict_hit', 0.0)
    validations.append(Validation(
        id='V1', name='dict_hit threshold',
        value=round(dict_hit, 4), threshold=0.40,
        passed=dict_hit >= 0.40,
        note=f"Expanded dict hit rate: {dict_hit:.4f}",
    ))

    # V2: bigram JSD vs Latin < 0.5
    jsd = _bigram_jsd(passage_words, ref_words[:5000])
    validations.append(Validation(
        id='V2', name='bigram JSD vs Latin',
        value=round(jsd, 4), threshold=0.5,
        passed=jsd < 0.5,
        note=f"JSD between decoded and Latin bigrams: {jsd:.4f}",
    ))

    # V3: section variation chi_sq > 3.84
    section_stats = decode_data.get('section_stats', [])
    chi_sq = _section_chi_sq(section_stats)
    validations.append(Validation(
        id='V3', name='section variation',
        value=round(chi_sq, 2), threshold=3.84,
        passed=chi_sq > 3.84,
        note=f"Chi-squared for section dict_hit variation: {chi_sq:.2f}",
    ))

    # V4: mean signal sigma > 2.0
    word_signals = signal_data.get('word_signals', [])
    sigmas = [ws.get('signal_sigma', 0) for ws in word_signals
              if ws.get('signal_sigma', 0) < 900]
    mean_sigma = sum(sigmas) / len(sigmas) if sigmas else 0.0
    validations.append(Validation(
        id='V4', name='signal sigma',
        value=round(mean_sigma, 2), threshold=2.0,
        passed=mean_sigma > 2.0,
        note=f"Mean signal sigma across crib words: {mean_sigma:.2f}",
    ))

    # V5: domain accuracy ≥ 0.50
    domain_acc = local_data.get('domain_accuracy', 0.0)
    validations.append(Validation(
        id='V5', name='domain accuracy',
        value=round(domain_acc, 4), threshold=0.50,
        passed=domain_acc >= 0.50,
        note=f"Domain placement accuracy: {domain_acc:.4f}",
    ))

    # V6: longest consecutive hit run > 5
    longest = decode_data.get('longest_consecutive', 0)
    validations.append(Validation(
        id='V6', name='consecutive hit run',
        value=float(longest), threshold=5.0,
        passed=longest > 5,
        note=f"Longest consecutive dict hit run: {longest}",
    ))

    # V7: modifier fraction in 0.20–0.50 range
    n_mod = len(mod_data.get('modifier_chars', []))
    n_total_chars = n_mod + len(mod_data.get('syllabic_chars', [])) + len(
        mod_data.get('ambiguous_chars', []))
    mod_frac = n_mod / n_total_chars if n_total_chars > 0 else 0.0
    validations.append(Validation(
        id='V7', name='modifier fraction',
        value=round(mod_frac, 4), threshold=0.0,
        passed=0.20 <= mod_frac <= 0.50,
        note=f"Modifier chars: {n_mod}/{n_total_chars} ({mod_frac:.1%}), "
             f"expected 20–50%",
    ))

    # V8: no regression vs Phase 16
    phase16 = decode_data.get('phase16_baseline', 0.0)
    regression = dict_hit - phase16
    validations.append(Validation(
        id='V8', name='no regression vs Phase 16',
        value=round(regression, 4), threshold=-0.02,
        passed=regression >= -0.02,
        note=f"dict_hit delta vs Phase 16: {regression:+.4f}",
    ))

    # ── 4. Report ──
    print()
    for v in validations:
        tag = '✓' if v.passed else '✗'
        print(f"    {tag} {v.id}: {v.name:30s}  "
              f"value={v.value:8.4f}  thr={v.threshold:8.4f}  → "
              f"{'PASS' if v.passed else 'FAIL'}")

    n_passed = sum(1 for v in validations if v.passed)
    n_total = len(validations)
    pass_rate = n_passed / n_total if n_total > 0 else 0.0

    gate_passed = n_passed >= 6
    verdict = (
        f"PASS: {n_passed}/{n_total} validations passed ({pass_rate:.0%})"
        if gate_passed
        else f"FAIL: Only {n_passed}/{n_total} validations passed ({pass_rate:.0%})"
    )
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 5. Save ──
    result = ReadabilityResult(
        validations=[_convert(asdict(v)) for v in validations],
        n_passed=n_passed,
        n_total=n_total,
        pass_rate=round(pass_rate, 4),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'ventris_readability.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
