"""
Phase 34.11 – Scripta Continua Stream Construction
=====================================================
Strips all EVA spaces and creates continuous character streams at both
the raw EVA level and the decoded (Phase 16) level.  Computes entropy
profiles for the continuous vs. space-segmented versions to quantify
how much information word boundaries carry.

Dependency chain:
    combined_refine.json       (Phase 15 assignment)
    modifier_integrate.json    (Phase 16 modifiers)
        → continua_stream.json   (this step)
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.null_corpus import _reconstruct_modifier_rules
from voynich.phases.signal_isolation import _decode_corpus_r3


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


def _char_entropy(stream: str) -> float:
    """Compute per-character Shannon entropy (bits) of a string."""
    if not stream:
        return 0.0
    counts = Counter(stream)
    total = len(stream)
    h = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FolioStreamStats:
    folio: str
    section: str
    n_tokens: int
    n_decoded_chars: int
    n_eva_chars: int
    mean_word_length: float
    continuous_entropy: float
    segmented_entropy: float
    entropy_delta: float


@dataclass
class ContinuaStreamResult:
    n_folios: int
    total_decoded_chars: int
    mean_original_word_length: float
    continuous_entropy: float
    segmented_entropy: float
    entropy_delta: float
    sample_streams: List[Dict]
    per_folio_stats: List[Dict]
    total_eva_chars: int
    n_total_tokens: int
    n_unique_decoded_chars: int
    n_unique_eva_chars: int
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_continua_stream() -> None:
    """Step 34.11: Build continuous character streams and compute entropy."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 34.11: Scripta Continua Stream Construction")
    print("=" * 70)

    rd = _results_dir()
    eva_to_triple = build_eva_to_triple_lookup()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

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

    # ── 2. Build reference word set ──
    print("\n  2. Building reference word set …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin')
        if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded
    print(f"     {len(ref_word_set)} reference words")

    # ── 3. Iterate folios, build streams ──
    print("\n  3. Building continuous streams per folio …")
    corpus = load_corpus(verbose=False)

    per_folio: List[FolioStreamStats] = []
    sample_streams: List[Dict] = []

    all_decoded_chars_global: List[str] = []
    all_segmented_words_global: List[str] = []
    all_eva_chars_global: List[str] = []
    total_word_lengths: List[int] = []
    n_total_tokens = 0

    for folio, page in corpus.pages.items():
        tokens = page.all_tokens
        if not tokens:
            continue

        n_total_tokens += len(tokens)

        # Decode tokens via R3
        decoded_words = _decode_corpus_r3(
            tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set,
        )

        # Record original word lengths (decoded)
        for w in decoded_words:
            total_word_lengths.append(len(w))

        # Build raw EVA continuous stream (character-level, no spaces)
        eva_chars_folio: List[str] = []
        for token in tokens:
            chars = tokenize_eva_chars(token)
            eva_chars_folio.extend(chars)
        eva_continuous = ''.join(eva_chars_folio)
        all_eva_chars_global.extend(eva_chars_folio)

        # Build decoded continuous stream (no spaces)
        decoded_continuous = ''.join(decoded_words)
        all_decoded_chars_global.extend(list(decoded_continuous))

        # Build decoded segmented stream (space-separated)
        decoded_segmented = ' '.join(decoded_words)
        all_segmented_words_global.extend(decoded_words)

        # Entropy of continuous decoded stream
        cont_h = _char_entropy(decoded_continuous)
        # Entropy of segmented stream (including space character)
        seg_h = _char_entropy(decoded_segmented)
        # Delta: how much entropy changes when spaces are removed
        entropy_delta = cont_h - seg_h

        mean_wl = (
            sum(len(w) for w in decoded_words) / len(decoded_words)
            if decoded_words else 0.0
        )

        fs = FolioStreamStats(
            folio=folio,
            section=page.section,
            n_tokens=len(tokens),
            n_decoded_chars=len(decoded_continuous),
            n_eva_chars=len(eva_continuous),
            mean_word_length=round(mean_wl, 2),
            continuous_entropy=round(cont_h, 4),
            segmented_entropy=round(seg_h, 4),
            entropy_delta=round(entropy_delta, 4),
        )
        per_folio.append(fs)

        # Save sample streams for first 5 folios
        if len(sample_streams) < 5:
            sample_streams.append({
                'folio': folio,
                'section': page.section,
                'n_tokens': len(tokens),
                'decoded_continuous': decoded_continuous[:200],
                'eva_continuous': eva_continuous[:200],
                'decoded_segmented': decoded_segmented[:200],
            })

    # ── 4. Global statistics ──
    print("\n  4. Computing global statistics …")

    global_decoded_continuous = ''.join(all_decoded_chars_global)
    global_decoded_segmented = ' '.join(all_segmented_words_global)

    global_cont_h = _char_entropy(global_decoded_continuous)
    global_seg_h = _char_entropy(global_decoded_segmented)
    global_entropy_delta = global_cont_h - global_seg_h

    mean_original_word_length = (
        sum(total_word_lengths) / len(total_word_lengths)
        if total_word_lengths else 0.0
    )

    n_unique_decoded = len(set(global_decoded_continuous))
    n_unique_eva = len(set(all_eva_chars_global))

    print(f"     Folios: {len(per_folio)}")
    print(f"     Total tokens: {n_total_tokens}")
    print(f"     Total decoded chars: {len(global_decoded_continuous)}")
    print(f"     Total EVA chars: {len(all_eva_chars_global)}")
    print(f"     Mean original word length: {mean_original_word_length:.2f}")
    print(f"     Continuous entropy: {global_cont_h:.4f} bits/char")
    print(f"     Segmented entropy:  {global_seg_h:.4f} bits/char")
    print(f"     Entropy delta:      {global_entropy_delta:+.4f} bits/char")
    print(f"     Unique decoded chars: {n_unique_decoded}")
    print(f"     Unique EVA chars: {n_unique_eva}")

    # ── 5. Per-folio ranking ──
    print("\n  5. Top folios by entropy delta (top 10) …")
    per_folio.sort(key=lambda f: -abs(f.entropy_delta))
    for fs in per_folio[:10]:
        print(f"     {fs.folio:8s}  section={fs.section:12s}  "
              f"cont_H={fs.continuous_entropy:.3f}  "
              f"seg_H={fs.segmented_entropy:.3f}  "
              f"Δ={fs.entropy_delta:+.3f}")

    # ── 6. Verdict ──
    if abs(global_entropy_delta) > 0.3:
        verdict = (
            f"ENTROPY_SHIFT: Removing spaces changes character entropy by "
            f"{global_entropy_delta:+.4f} bits/char — word boundaries carry "
            f"significant information."
        )
    elif abs(global_entropy_delta) > 0.1:
        verdict = (
            f"ENTROPY_MODERATE: Removing spaces changes character entropy by "
            f"{global_entropy_delta:+.4f} bits/char — word boundaries carry "
            f"moderate information."
        )
    else:
        verdict = (
            f"ENTROPY_FLAT: Removing spaces changes character entropy by "
            f"{global_entropy_delta:+.4f} bits/char — word boundaries carry "
            f"minimal information (consistent with scripta continua)."
        )

    print(f"\n  Verdict: {verdict}")

    # ── 7. Save ──
    # Sort per_folio back to original folio order for output
    per_folio.sort(key=lambda f: f.folio)

    result = ContinuaStreamResult(
        n_folios=len(per_folio),
        total_decoded_chars=len(global_decoded_continuous),
        mean_original_word_length=round(mean_original_word_length, 2),
        continuous_entropy=round(global_cont_h, 4),
        segmented_entropy=round(global_seg_h, 4),
        entropy_delta=round(global_entropy_delta, 4),
        sample_streams=sample_streams,
        per_folio_stats=[_convert(asdict(fs)) for fs in per_folio],
        total_eva_chars=len(all_eva_chars_global),
        n_total_tokens=n_total_tokens,
        n_unique_decoded_chars=n_unique_decoded,
        n_unique_eva_chars=n_unique_eva,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'continua_stream.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
