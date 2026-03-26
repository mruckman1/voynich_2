"""
Phase 62, Investigation 7: Token Length Distribution Analysis
==============================================================
Correlate EVA character count per token to decoded CVC output length
and Costamagna segmentation. If each EVA token = one syllable, tokens
should produce 2-4 char decoded strings (CV=2, CVC=3).

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
        -> results/phase62_token_length.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    classify_token_chars_v2,
    decode_corpus_cvc_v2,
)


# ---------------------------------------------------------------------------
# JSON helpers
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
    if isinstance(obj, set):
        return sorted(obj)
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TokenLengthResult:
    phase: str = "62"
    step: str = "62.7"
    experiment: str = "token_length"
    n_tokens: int = 0
    # Correlations
    corr_eva_decoded: float = 0.0
    corr_syllabic_decoded: float = 0.0
    spearman_eva_decoded: float = 0.0
    spearman_syllabic_decoded: float = 0.0
    # Mean stats
    mean_eva_chars: float = 0.0
    mean_decoded_chars: float = 0.0
    mean_syllabic_chars: float = 0.0
    mean_coda_chars: float = 0.0
    mean_chars_per_syllabic: float = 0.0
    # Distribution
    decoded_length_dist: Dict[int, int] = field(default_factory=dict)
    eva_length_dist: Dict[int, int] = field(default_factory=dict)
    fraction_6plus: float = 0.0
    # Per-bin breakdown
    bins: List[Dict] = field(default_factory=list)
    # Gates
    g1_corr_syllabic: bool = False     # correlation > 0.7
    g2_chars_per_syl: bool = False     # mean chars/syllabic = 2.0 ± 0.5
    g3_short_tokens: bool = False      # < 30% with 6+ decoded chars
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_token_length():
    """Phase 62.7: Token length distribution analysis."""
    t0 = time.time()
    rd = str(_results_dir())
    print("=" * 70)
    print("Phase 62, Investigation 7: Token Length Distribution")
    print("=" * 70)

    # Load
    eva_to_triple = build_eva_to_triple_lookup()
    refine = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine.get('best_assignment', {})
    coda_table = build_coda_table_v2()

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()
    decoded = decode_corpus_cvc_v2(all_tokens, assignment, eva_to_triple, coda_table)

    print(f"  Tokens: {len(all_tokens)}")

    # Collect per-token measurements
    eva_lens = []
    decoded_lens = []
    syllabic_lens = []
    coda_lens = []
    chars_per_syllabic = []

    for token, dec in zip(all_tokens, decoded):
        if not dec or dec == '?':
            continue

        eva_chars = tokenize_eva_chars(token)
        n_eva = len(eva_chars)
        n_decoded = len(dec)

        classified = classify_token_chars_v2(eva_chars, coda_table)
        n_syllabic = sum(1 for role, _ in classified if role == 'SYLLABIC')
        n_coda = sum(1 for role, _ in classified if role == 'CODA_MARKER')

        eva_lens.append(n_eva)
        decoded_lens.append(n_decoded)
        syllabic_lens.append(n_syllabic)
        coda_lens.append(n_coda)
        if n_syllabic > 0:
            chars_per_syllabic.append(n_decoded / n_syllabic)

    n = len(eva_lens)
    eva_arr = np.array(eva_lens)
    dec_arr = np.array(decoded_lens)
    syl_arr = np.array(syllabic_lens)

    # Correlations
    corr_eva = float(np.corrcoef(eva_arr, dec_arr)[0, 1]) if n >= 3 else 0.0
    corr_syl = float(np.corrcoef(syl_arr, dec_arr)[0, 1]) if n >= 3 else 0.0
    spearman_eva = float(sp_stats.spearmanr(eva_arr, dec_arr).statistic) if n >= 3 else 0.0
    spearman_syl = float(sp_stats.spearmanr(syl_arr, dec_arr).statistic) if n >= 3 else 0.0

    # Fix NaN
    for val_name in ['corr_eva', 'corr_syl', 'spearman_eva', 'spearman_syl']:
        v = locals()[val_name]
        if np.isnan(v):
            locals()[val_name] = 0.0
    corr_eva = 0.0 if np.isnan(corr_eva) else corr_eva
    corr_syl = 0.0 if np.isnan(corr_syl) else corr_syl
    spearman_eva = 0.0 if np.isnan(spearman_eva) else spearman_eva
    spearman_syl = 0.0 if np.isnan(spearman_syl) else spearman_syl

    mean_cps = float(np.mean(chars_per_syllabic)) if chars_per_syllabic else 0.0
    frac_6plus = sum(1 for d in decoded_lens if d >= 6) / n if n > 0 else 0.0

    # Per-bin breakdown (by EVA char count)
    bins_data = {}
    for i in range(n):
        ev = eva_lens[i]
        if ev not in bins_data:
            bins_data[ev] = {'decoded': [], 'syllabic': []}
        bins_data[ev]['decoded'].append(decoded_lens[i])
        bins_data[ev]['syllabic'].append(syllabic_lens[i])

    bins = []
    for ev in sorted(bins_data.keys()):
        d = bins_data[ev]
        bins.append({
            'eva_len': ev,
            'n_tokens': len(d['decoded']),
            'mean_decoded_len': round(float(np.mean(d['decoded'])), 2),
            'std_decoded_len': round(float(np.std(d['decoded'])), 2),
            'mean_syllabic': round(float(np.mean(d['syllabic'])), 2),
        })

    # Gates
    g1 = corr_syl > 0.7
    g2 = 1.5 <= mean_cps <= 2.5
    g3 = frac_6plus < 0.30
    gates_passed = sum([g1, g2, g3])

    if g1 and g3:
        verdict = "SYLLABLE_MODEL_CONSISTENT"
    elif not g3:
        verdict = "TOKENS_CONTAIN_MULTIPLE_SYLLABLES"
    else:
        verdict = "WEAK_CORRELATION"

    result = TokenLengthResult(
        n_tokens=n,
        corr_eva_decoded=round(corr_eva, 4),
        corr_syllabic_decoded=round(corr_syl, 4),
        spearman_eva_decoded=round(spearman_eva, 4),
        spearman_syllabic_decoded=round(spearman_syl, 4),
        mean_eva_chars=round(float(np.mean(eva_lens)), 2),
        mean_decoded_chars=round(float(np.mean(decoded_lens)), 2),
        mean_syllabic_chars=round(float(np.mean(syllabic_lens)), 2),
        mean_coda_chars=round(float(np.mean(coda_lens)), 2),
        mean_chars_per_syllabic=round(mean_cps, 3),
        decoded_length_dist=dict(Counter(decoded_lens).most_common()),
        eva_length_dist=dict(Counter(eva_lens).most_common()),
        fraction_6plus=round(frac_6plus, 4),
        bins=bins,
        g1_corr_syllabic=g1,
        g2_chars_per_syl=g2,
        g3_short_tokens=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        verdict=verdict,
        runtime_seconds=time.time() - t0,
    )

    # Print summary
    print(f"\n  Correlations:")
    print(f"    EVA chars vs decoded length: r={corr_eva:.3f} (Spearman={spearman_eva:.3f})")
    print(f"    Syllabic chars vs decoded length: r={corr_syl:.3f} (Spearman={spearman_syl:.3f})")
    print(f"  Means: EVA={np.mean(eva_lens):.1f} chars, decoded={np.mean(decoded_lens):.1f} chars")
    print(f"  Mean chars per syllabic char: {mean_cps:.2f}")
    print(f"  Fraction with 6+ decoded chars: {frac_6plus:.1%}")
    print(f"\n  Per-bin breakdown:")
    for b in bins[:8]:
        print(f"    EVA={b['eva_len']} chars: n={b['n_tokens']:5d}  "
              f"decoded={b['mean_decoded_len']:.1f}±{b['std_decoded_len']:.1f}  "
              f"syllabic={b['mean_syllabic']:.1f}")
    print(f"\n  Gates: G1={'PASS' if g1 else 'FAIL'} G2={'PASS' if g2 else 'FAIL'} "
          f"G3={'PASS' if g3 else 'FAIL'} ({gates_passed}/3)")
    print(f"  Verdict: {verdict}")

    path = _save_json(rd, 'phase62_token_length.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result
