"""
Step 43.5 – Inversion Validation
===================================
Validate the inversion-decoded corpus using the symmetric methodology
from Phase 42: decode null corpora, compute signal isolation, compute
symmetric bigram z-score.

Dependency chain:
    results/inversion_decode.json     (Step 43.4: inverted table + decode)
    results/combined_refine.json      (Phase 15: baseline comparison)
    results/symmetric_recompute.json  (Phase 42.2: baseline z)
    results/null_corpus.json          (Phase 17: null seeds)
    results/modifier_integrate.json   (Phase 16: modifiers)
    data/corpus/                      (EVA transcription)
        → inversion_validate.json     (this step)
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import build_expanded_word_set, load_reference_corpus


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
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InversionValidateResult:
    # Dict-hit comparison
    real_dict_hit: float
    phase15_dict_hit: float
    delta_dict_hit: float
    # Null comparison
    null_dict_hits: List[float]
    null_dict_hit_mean: float
    null_dict_hit_std: float
    dict_selectivity: float
    # Signal isolation
    n_signal: int
    n_shared_hit: int
    n_shared_miss: int
    n_anti_signal: int
    signal_rate: float
    # Bigram test
    real_bigram_hits: int
    null_bigram_hits_mean: float
    bigram_z_score: float
    # Signal word preservation
    bedrock_word_hits: Dict[str, bool]
    n_bedrock_preserved: int
    # Validation battery
    validations: List[Dict]
    n_passed: int
    n_total: int
    # Verdict
    approach1_verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_reference_bigrams(ref_tokens: List[str]) -> set:
    """Build set of word bigrams from reference corpus."""
    bigrams = set()
    for i in range(len(ref_tokens) - 1):
        bigrams.add((ref_tokens[i].lower(), ref_tokens[i + 1].lower()))
    return bigrams


def _count_bigram_hits(decoded: List[str], ref_bigrams: set) -> int:
    """Count how many consecutive decoded word pairs match reference bigrams."""
    hits = 0
    for i in range(len(decoded) - 1):
        pair = (decoded[i].lower(), decoded[i + 1].lower())
        if pair in ref_bigrams:
            hits += 1
    return hits


def _generate_null_decoded(
    corpus, inv_assignment, modifier_chars, n_null=5,
) -> List[List[str]]:
    """Generate null-decoded corpora by shuffling token order."""
    rng = np.random.default_rng(42)
    eva_to_triple = build_eva_to_triple_lookup()

    # Get all tokens
    all_tokens = corpus.get_tokens(paragraph_only=True)

    null_decoded_list = []
    for seed in range(n_null):
        rng_s = np.random.default_rng(100 + seed)
        shuffled = list(all_tokens)
        rng_s.shuffle(shuffled)

        decoded = []
        for token in shuffled:
            w = decode_token_modifier_aware(
                token, inv_assignment, eva_to_triple, modifier_chars
            )
            decoded.append(w)
        null_decoded_list.append(decoded)

    return null_decoded_list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_inversion_validate() -> None:
    """Step 43.5: validate the inversion-decoded corpus."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 43.5: Inversion Validation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    inv_data = _safe_load(os.path.join(rd, 'inversion_decode.json'))
    inv_assignment = inv_data.get('inverted_assignment', {})
    real_dict_hit = inv_data.get('dict_hit_rate', 0.0)
    print(f"     Inversion dict-hit: {real_dict_hit:.1%}")

    combined = _safe_load(os.path.join(rd, 'combined_refine.json'))
    p15_assignment = combined.get('best_assignment', {})

    sym = _safe_load(os.path.join(rd, 'symmetric_recompute.json'))
    baseline_z = sym.get('best_surviving_z_exact', 3.80)
    print(f"     Phase 42 baseline z: {baseline_z}")

    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    modifier_chars = set(mod_data.get('modifier_chars', []))

    # ── 2. Build dictionary ──
    print("\n  2. Building dictionary …")
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(ref.get_combined_tokens('latin'))
        expanded, _ = build_expanded_word_set(base_words)
        ref_tokens = ref.get_combined_tokens('latin')
    except Exception:
        expanded = set()
        ref_tokens = []
    print(f"     Dictionary: {len(expanded):,} words")

    ref_bigrams = _build_reference_bigrams(ref_tokens) if ref_tokens else set()
    print(f"     Reference bigrams: {len(ref_bigrams):,}")

    # ── 3. Decode real corpus ──
    print("\n  3. Decoding real corpus …")
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    real_decoded = []
    for folio_id, page in corpus.pages.items():
        for token in page.all_tokens:
            w = decode_token_modifier_aware(
                token, inv_assignment, eva_to_triple, modifier_chars
            )
            real_decoded.append(w)

    real_hits = [w.lower() in expanded for w in real_decoded]
    real_hit_rate = sum(real_hits) / len(real_hits) if real_hits else 0.0
    print(f"     Real dict-hit: {sum(real_hits):,} / {len(real_decoded):,} ({real_hit_rate:.1%})")

    # Phase 15 decode for comparison
    p15_decoded = []
    for folio_id, page in corpus.pages.items():
        for token in page.all_tokens:
            w = decode_token_modifier_aware(
                token, p15_assignment, eva_to_triple, modifier_chars
            )
            p15_decoded.append(w)
    p15_hits = [w.lower() in expanded for w in p15_decoded]
    p15_hit_rate = sum(p15_hits) / len(p15_hits) if p15_hits else 0.0
    print(f"     Phase 15 dict-hit: {sum(p15_hits):,} / {len(p15_decoded):,} ({p15_hit_rate:.1%})")

    # ── 4. Null corpus comparison ──
    print("\n  4. Null corpus comparison …")
    null_decoded_list = _generate_null_decoded(corpus, inv_assignment, modifier_chars, n_null=5)

    null_dict_hits = []
    for i, null_dec in enumerate(null_decoded_list):
        nh = sum(1 for w in null_dec if w.lower() in expanded)
        null_rate = nh / len(null_dec) if null_dec else 0.0
        null_dict_hits.append(null_rate)
        print(f"     Null {i}: {null_rate:.1%}")

    null_mean = float(np.mean(null_dict_hits))
    null_std = float(np.std(null_dict_hits)) if len(null_dict_hits) > 1 else 0.01
    dict_selectivity = real_hit_rate / null_mean if null_mean > 0 else 0.0

    # ── 5. Signal isolation ──
    print("\n  5. Signal isolation …")
    null_hit_sets = []
    for null_dec in null_decoded_list:
        null_hit_sets.append(set(i for i, w in enumerate(null_dec) if w.lower() in expanded))

    n_signal = 0
    n_shared_hit = 0
    n_shared_miss = 0
    n_anti_signal = 0

    for i in range(len(real_decoded)):
        r_hit = real_hits[i] if i < len(real_hits) else False
        null_count = sum(1 for ns in null_hit_sets if i < len(null_decoded_list[0]) and i in ns)

        if r_hit and null_count == 0:
            n_signal += 1
        elif r_hit and null_count > 0:
            n_shared_hit += 1
        elif not r_hit and null_count == 0:
            n_shared_miss += 1
        else:
            n_anti_signal += 1

    signal_rate = n_signal / len(real_decoded) if real_decoded else 0.0
    print(f"     SIGNAL: {n_signal} ({signal_rate:.1%})")
    print(f"     SHARED_HIT: {n_shared_hit}")
    print(f"     SHARED_MISS: {n_shared_miss}")
    print(f"     ANTI_SIGNAL: {n_anti_signal}")

    # ── 6. Bigram z-score ──
    print("\n  6. Bigram z-score …")
    real_bg_hits = _count_bigram_hits(real_decoded, ref_bigrams)

    null_bg_hits = []
    for null_dec in null_decoded_list:
        null_bg_hits.append(_count_bigram_hits(null_dec, ref_bigrams))

    null_bg_mean = float(np.mean(null_bg_hits)) if null_bg_hits else 0.0
    null_bg_std = float(np.std(null_bg_hits)) if len(null_bg_hits) > 1 else 1.0
    bigram_z = (real_bg_hits - null_bg_mean) / null_bg_std if null_bg_std > 0 else 0.0

    print(f"     Real bigram hits: {real_bg_hits}")
    print(f"     Null mean: {null_bg_mean:.1f} ± {null_bg_std:.1f}")
    print(f"     Bigram z: {bigram_z:.2f}")

    # ── 7. Signal word preservation ──
    print("\n  7. Signal word check …")
    bedrock = ['bene', 'codi', 'sero', 'sene', 'de', 'raro', 'dine', 'cola']
    real_word_set = Counter(w.lower() for w in real_decoded)
    bedrock_hits = {w: w in real_word_set and real_word_set[w] >= 3 for w in bedrock}
    n_preserved = sum(bedrock_hits.values())
    print(f"     Preserved: {n_preserved}/8")
    for w, hit in bedrock_hits.items():
        if hit:
            print(f"       {w}: {real_word_set[w]} occurrences ✓")

    # ── 8. Validation battery ──
    print("\n  8. Validation battery …")
    validations = []

    # V1: dict-hit above random baseline
    v1 = real_hit_rate > 0.30
    validations.append({'id': 'V1', 'test': 'dict_hit > 30%',
                       'value': round(real_hit_rate, 4), 'threshold': 0.30, 'passed': v1})

    # V2: bigram z > 2.0
    v2 = bigram_z > 2.0
    validations.append({'id': 'V2', 'test': 'bigram_z > 2.0',
                       'value': round(bigram_z, 2), 'threshold': 2.0, 'passed': v2})

    # V3: agreement with Phase 15 > 40%
    agreement = inv_data.get('agreement_rate', 0.0)
    v3 = agreement > 0.40
    validations.append({'id': 'V3', 'test': 'agreement_rate > 40%',
                       'value': round(agreement, 4), 'threshold': 0.40, 'passed': v3})

    # V4: selectivity > 1.2
    v4 = dict_selectivity > 1.2
    validations.append({'id': 'V4', 'test': 'selectivity > 1.2',
                       'value': round(dict_selectivity, 4), 'threshold': 1.2, 'passed': v4})

    # V5: signal rate > 5%
    v5 = signal_rate > 0.05
    validations.append({'id': 'V5', 'test': 'signal_rate > 5%',
                       'value': round(signal_rate, 4), 'threshold': 0.05, 'passed': v5})

    n_passed = sum(1 for v in validations if v['passed'])
    n_total = len(validations)

    for v in validations:
        status = "PASS" if v['passed'] else "FAIL"
        print(f"     {v['id']}: {v['test']} → {v['value']} [{status}]")

    # ── 9. Verdict ──
    delta = real_hit_rate - p15_hit_rate
    if n_passed >= 4 and bigram_z > baseline_z:
        verdict = "IMPROVEMENT"
    elif n_passed >= 3 and abs(delta) < 0.05:
        verdict = "LATERAL"
    else:
        verdict = "REGRESSION"

    print(f"\n  Verdict: {verdict}")
    print(f"  Delta dict-hit vs Phase 15: {delta:+.1%}")
    print(f"  Bigram z vs baseline: {bigram_z:.2f} vs {baseline_z:.2f}")

    # ── 10. Save ──
    elapsed = time.time() - t0

    result = InversionValidateResult(
        real_dict_hit=round(real_hit_rate, 6),
        phase15_dict_hit=round(p15_hit_rate, 6),
        delta_dict_hit=round(delta, 6),
        null_dict_hits=[round(x, 4) for x in null_dict_hits],
        null_dict_hit_mean=round(null_mean, 4),
        null_dict_hit_std=round(null_std, 4),
        dict_selectivity=round(dict_selectivity, 4),
        n_signal=n_signal,
        n_shared_hit=n_shared_hit,
        n_shared_miss=n_shared_miss,
        n_anti_signal=n_anti_signal,
        signal_rate=round(signal_rate, 6),
        real_bigram_hits=real_bg_hits,
        null_bigram_hits_mean=round(null_bg_mean, 2),
        bigram_z_score=round(bigram_z, 4),
        bedrock_word_hits=bedrock_hits,
        n_bedrock_preserved=n_preserved,
        validations=validations,
        n_passed=n_passed,
        n_total=n_total,
        approach1_verdict=verdict,
        runtime_seconds=round(elapsed, 1),
    )

    out_path = os.path.join(rd, 'inversion_validate.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path} ({elapsed:.1f}s)")
