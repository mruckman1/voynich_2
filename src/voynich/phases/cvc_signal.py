"""
Step 40.7 – CVC Signal Isolation
==================================
Full-corpus signal isolation using the CVC assignment from Step 40.6.

Dependency chain:
    cvc_csp.json            (Step 40.6)
    merged_signal.json      (Step 38.3)
    null_corpus.json        (Step 17)
    merged_dict.json        (Step 38.1)
        → cvc_signal.json   (this step)
"""

import json
import math
import os
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
)


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
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cvc_signal() -> None:
    """Step 40.7: CVC Signal Isolation."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 40.7: CVC Signal Isolation")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")
    cvc_data = _safe_load(os.path.join(rd, 'cvc_csp.json'))
    merged_signal = _safe_load(os.path.join(rd, 'merged_signal.json'))
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    merged_dict = _safe_load(os.path.join(rd, 'merged_dict.json'))

    cvc_assignment = cvc_data.get('best_assignment_cvc', {})
    cv_assignment = cvc_data.get('best_cv_assignment', {})
    # Use the CVC assignment if it improved, otherwise fall back to CV
    verdict = cvc_data.get('verdict', 'CVC_NEUTRAL')
    assignment = cvc_assignment if verdict == 'CVC_IMPROVES' else cv_assignment
    print(f"    Using assignment: {'CVC' if verdict == 'CVC_IMPROVES' else 'CV'}")
    print(f"    Assignment size: {len(assignment)} triples")

    # Build reference word set
    ref_words = set(merged_dict.get('latin_10k_words', []))
    ref_words.update(merged_dict.get('italian_10k_words', []))
    ven_lex = _safe_load(os.path.join(rd, 'venetian_lexicon.json'))
    for entry in ven_lex.get('supplement_words', []):
        if isinstance(entry, str):
            ref_words.add(entry)
        elif isinstance(entry, dict):
            ref_words.add(entry.get('word', ''))
    ven_forms = _safe_load(os.path.join(rd, 'venetian_forms.json'))
    for w in ven_forms.get('venetian_extended_set', []):
        ref_words.add(w)
    ref_words.discard('')
    print(f"    Reference word set: {len(ref_words):,}")

    # ── 2. Decode full corpus with selected assignment ──
    print("\n  2. Decoding full corpus …")
    decoded_tokens = merged_signal.get('token_decoded', [])
    token_folios = merged_signal.get('token_folios', [])
    n_tokens = len(decoded_tokens)

    # If CVC improved, re-decode; otherwise reuse existing decoded tokens
    if verdict == 'CVC_IMPROVES' and cvc_assignment:
        from voynich.phases.csp_solver import decode_token
        eva_to_triple = build_eva_to_triple_lookup()
        corpus = load_corpus(verbose=False)
        all_tokens = []
        all_folios = []
        for folio, page in corpus.pages.items():
            for token in page.all_tokens:
                all_tokens.append(token)
                all_folios.append(folio)

        cvc_decoded = []
        for token in all_tokens:
            d = decode_token(token, cvc_assignment, eva_to_triple)
            cvc_decoded.append(d)
        decoded_tokens = cvc_decoded
        token_folios = all_folios
        n_tokens = len(decoded_tokens)
        print(f"    Re-decoded {n_tokens:,} tokens with CVC assignment")
    else:
        print(f"    Reusing merged decoded tokens ({n_tokens:,})")

    # ── 3. Dict-hit ──
    print("\n  3. Computing dict-hit …")
    cvc_hits = sum(1 for w in decoded_tokens if w and w in ref_words)
    cvc_dict_hit = cvc_hits / n_tokens if n_tokens > 0 else 0.0
    print(f"    CVC dict-hit: {cvc_hits:,}/{n_tokens:,} = {cvc_dict_hit:.4f}")

    # ── 4. Null comparison ──
    print("\n  4. Null corpus comparison …")
    null_decoded_lists = []
    for run in null_data.get('null_runs', []):
        nd = run.get('decoded_tokens', [])
        if nd:
            null_decoded_lists.append(nd)

    null_rates = []
    for nd in null_decoded_lists:
        nh = sum(1 for w in nd if w and w in ref_words)
        null_rates.append(nh / len(nd) if nd else 0.0)
    null_mean = sum(null_rates) / len(null_rates) if null_rates else 0.0
    selectivity = cvc_dict_hit / null_mean if null_mean > 0.001 else 999.0
    print(f"    Null mean: {null_mean:.4f}")
    print(f"    CVC selectivity: {selectivity:.2f}×")

    # ── 5. 4-class classification ──
    print("\n  5. Classifying tokens …")
    classifications = []
    counts = Counter()
    for i in range(n_tokens):
        word = decoded_tokens[i] if i < len(decoded_tokens) else ''
        real_hit = bool(word and word in ref_words)
        null_hit_count = 0
        for nd in null_decoded_lists:
            if i < len(nd) and nd[i] and nd[i] in ref_words:
                null_hit_count += 1

        if real_hit and null_hit_count <= 1:
            cls = 'SIGNAL'
        elif real_hit and null_hit_count >= 3:
            cls = 'SHARED_HIT'
        elif not real_hit and null_hit_count >= 3:
            cls = 'ANTI_SIGNAL'
        else:
            cls = 'SHARED_MISS'
        classifications.append(cls)
        counts[cls] += 1

    n_signal = counts.get('SIGNAL', 0)
    signal_rate = n_signal / n_tokens if n_tokens > 0 else 0.0
    print(f"    SIGNAL: {n_signal:,} ({signal_rate:.4f})")
    for cls in ['SHARED_HIT', 'SHARED_MISS', 'ANTI_SIGNAL']:
        print(f"    {cls}: {counts.get(cls, 0):,}")

    # ── 6. Per-word signal ──
    print("\n  6. Computing per-word signal …")
    word_real: Counter = Counter()
    for i, w in enumerate(decoded_tokens):
        if w and w in ref_words and classifications[i] == 'SIGNAL':
            word_real[w] += 1
    top_words = word_real.most_common(30)
    print(f"    Top CVC signal words: {[w for w, c in top_words[:10]]}")

    # ── 7. Compare to merged signal ──
    merged_signal_rate = merged_signal.get('signal_rate', 0.0)
    print(f"\n  7. Comparison:")
    print(f"    Merged signal rate: {merged_signal_rate:.4f}")
    print(f"    CVC signal rate: {signal_rate:.4f}")
    print(f"    Delta: {signal_rate - merged_signal_rate:+.4f}")

    # ── 8. Save ──
    elapsed = time.time() - t0

    output = {
        'n_tokens': n_tokens,
        'cvc_dict_hit': round(cvc_dict_hit, 6),
        'null_mean': round(null_mean, 6),
        'cvc_selectivity': round(selectivity, 4),
        'n_cvc_signal': n_signal,
        'cvc_signal_rate': round(signal_rate, 6),
        'class_counts': dict(counts),
        'token_classifications_cvc': classifications,
        'top_signal_words': [{'word': w, 'count': c} for w, c in top_words],
        'delta_signal_vs_merged': round(signal_rate - merged_signal_rate, 6),
        'assignment_used': 'CVC' if verdict == 'CVC_IMPROVES' else 'CV',
        'verdict': ('CVC_SIGNAL_IMPROVES' if signal_rate > merged_signal_rate + 0.005
                    else 'CVC_SIGNAL_NEUTRAL'),
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(rd, 'cvc_signal.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n  Saved → {out_path} ({elapsed:.1f}s)")
