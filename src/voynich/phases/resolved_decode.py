"""
Phase 53 Track C: Re-Decode Corpus and Measure Improvement
============================================================
Decode all tokens with the corrected (or original) assignment table,
measure dict-hit improvement, find content runs, and produce updated
folio readings.

Dependency chain:
    triple_resolution.json     (Track B)
    combined_refine.json       (Phase 15)
    modifier_integrate.json    (Phase 16)
    signal_bigrams.json        (Phase 29)
    word_catalog.json          (Phase 52)
        -> resolved_decode.json (this step)
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    decode_token_modifier_aware,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.core.reference import (
    PHARMACEUTICAL_VOCABULARY,
    build_expanded_word_set,
    load_reference_corpus,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_SET


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
    if isinstance(obj, set):
        return sorted(_convert(item) for item in obj)
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


def _compute_dict_hit(decoded: List[str], ref_word_set: set) -> float:
    if not decoded:
        return 0.0
    hits = sum(1 for w in decoded if w.lower() in ref_word_set)
    return hits / len(decoded)


# ---------------------------------------------------------------------------
# Content words: pharmaceutical/botanical terms
# ---------------------------------------------------------------------------

def _build_content_word_set() -> Set[str]:
    """Build set of content words (pharmaceutical/botanical, NOT function words)."""
    function_words = {
        'de', 'di', 'se', 'ne', 'la', 'le', 'si', 'ni', 'bi', 'co',
        'ce', 'ci', 'cu', 'du', 'ra', 'ro', 're', 'do', 'su', 'tu',
        'ad', 'in', 'et', 'a', 'e', 'i', 'o', 'u',
        'te', 'be', 'bo', 'so', 'ha', 'hi', 'fa', 'ga', 'ge', 'fe',
        'mi', 'nu', 'ru', 'li', 'ti',
    }

    content_words: Set[str] = set()
    for domain, terms in PHARMACEUTICAL_VOCABULARY.items():
        for w in terms:
            w_lower = w.lower()
            if w_lower not in function_words and len(w_lower) >= 4:
                content_words.add(w_lower)

    return content_words


# ---------------------------------------------------------------------------
# Content run detection
# ---------------------------------------------------------------------------

def _find_content_runs(
    decoded_words: List[str],
    folios: List[str],
    eva_tokens: List[str],
    ref_word_set: Set[str],
    content_words: Set[str],
    min_run_length: int = 5,
    min_content_words: int = 2,
) -> List[Dict]:
    """Find consecutive runs of decoded words with content words."""
    n = len(decoded_words)
    runs = []

    # Find all dict-hit positions
    is_hit = [decoded_words[i].lower() in ref_word_set for i in range(n)]

    # Find consecutive runs of dict hits
    i = 0
    while i < n:
        if not is_hit[i]:
            i += 1
            continue

        # Start of a run
        j = i
        while j < n and is_hit[j]:
            j += 1

        run_length = j - i
        if run_length >= min_run_length:
            # Count content words in this run
            run_decoded = [decoded_words[k].lower() for k in range(i, j)]
            run_content = [w for w in run_decoded if w in content_words]

            if len(run_content) >= min_content_words:
                run_folios = [folios[k] for k in range(i, j)]
                run_evas = [eva_tokens[k] for k in range(i, j)]
                runs.append({
                    'start_idx': i,
                    'length': run_length,
                    'n_content_words': len(run_content),
                    'content_words': run_content,
                    'folio': run_folios[0],
                    'text': ' '.join(run_decoded),
                    'eva_text': ' '.join(run_evas),
                    'all_folios': list(set(run_folios)),
                    'score': run_length * len(run_content),
                })

        i = j

    # Sort by score descending
    runs.sort(key=lambda r: r['score'], reverse=True)
    return runs


def _find_all_runs(
    decoded_words: List[str],
    folios: List[str],
    eva_tokens: List[str],
    ref_word_set: Set[str],
    min_length: int = 3,
) -> List[Dict]:
    """Find all consecutive dict-hit runs (no content word requirement)."""
    n = len(decoded_words)
    runs = []
    is_hit = [decoded_words[i].lower() in ref_word_set for i in range(n)]

    i = 0
    while i < n:
        if not is_hit[i]:
            i += 1
            continue
        j = i
        while j < n and is_hit[j]:
            j += 1
        run_length = j - i
        if run_length >= min_length:
            run_decoded = [decoded_words[k].lower() for k in range(i, j)]
            runs.append({
                'start_idx': i,
                'length': run_length,
                'folio': folios[i],
                'text': ' '.join(run_decoded),
            })
        i = j

    runs.sort(key=lambda r: r['length'], reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Per-section breakdown
# ---------------------------------------------------------------------------

def _section_breakdown(
    decoded_words: List[str],
    folios: List[str],
    ref_word_set: Set[str],
) -> Dict[str, Dict]:
    """Compute dict-hit rate per manuscript section."""
    # Simple section mapping from folio numbers
    section_counts: Dict[str, int] = defaultdict(int)
    section_hits: Dict[str, int] = defaultdict(int)

    for decoded, folio in zip(decoded_words, folios):
        # Extract folio number for rough section assignment
        section = _folio_to_section(folio)
        section_counts[section] += 1
        if decoded.lower() in ref_word_set:
            section_hits[section] += 1

    result = {}
    for section in sorted(section_counts.keys()):
        total = section_counts[section]
        hits = section_hits[section]
        result[section] = {
            'n_tokens': total,
            'n_hits': hits,
            'dict_hit': round(hits / total, 4) if total > 0 else 0.0,
        }
    return result


def _folio_to_section(folio: str) -> str:
    """Map a folio ID to a manuscript section name."""
    # Extract numeric part
    num_str = ''
    for ch in folio:
        if ch.isdigit():
            num_str += ch
        elif num_str:
            break
    if not num_str:
        return 'unknown'
    num = int(num_str)

    if num <= 11:
        return 'herbal_a'
    elif num <= 56:
        return 'herbal_b'
    elif num <= 67:
        return 'pharma'
    elif num <= 73:
        return 'zodiac'
    elif num <= 84:
        return 'cosmo'
    elif num <= 86:
        return 'recipes'
    elif num <= 102:
        return 'herbal_c'
    elif num <= 116:
        return 'stars'
    else:
        return 'other'


# ---------------------------------------------------------------------------
# Dark-to-decoded comparison
# ---------------------------------------------------------------------------

def _count_newly_decoded(
    decoded_pre: List[str],
    decoded_post: List[str],
    ref_word_set: Set[str],
) -> int:
    """Count tokens that were DARK (no dict hit) before but decoded now."""
    n = 0
    for pre, post in zip(decoded_pre, decoded_post):
        was_dark = pre.lower() not in ref_word_set
        now_hit = post.lower() in ref_word_set
        if was_dark and now_hit:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_resolved_decode() -> None:
    """Phase 53 Track C: Re-decode corpus and measure improvement."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 53 TRACK C: Resolved Decode and Content Runs")
    print("=" * 70)

    rd = _results_dir()

    # ── Load data ─────────────────────────────────────────────────────
    print("\n  C.1  Loading data...")

    resolution_data = _safe_load(os.path.join(rd, 'triple_resolution.json'))
    if not resolution_data:
        print("  *** triple_resolution.json not found — run Track B first ***")
        return

    corrected_assignment = resolution_data.get('corrected_assignment', {})
    corrections_applied = resolution_data.get('corrections_applied', 0)

    # If no corrected assignment, fall back to original
    if not corrected_assignment:
        with open(os.path.join(rd, 'combined_refine.json')) as f:
            corrected_assignment = json.load(f)['best_assignment']

    # Also load original for comparison
    with open(os.path.join(rd, 'combined_refine.json')) as f:
        original_assignment = json.load(f)['best_assignment']

    with open(os.path.join(rd, 'modifier_integrate.json')) as f:
        mod_data = json.load(f)
    modifier_chars = set(mod_data.get('modifier_chars', []))

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        bigram_data = json.load(f)
    token_evas = bigram_data['token_evas']
    token_folios = bigram_data['token_folios']

    eva_to_triple = build_eva_to_triple_lookup()

    # Build reference word set
    print("       Building reference dictionary...")
    try:
        ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
        base_words = set(
            w.lower() for text in ref_corpus.get_texts('latin')
            for w in [t.lower() for t in text.tokens]
            if len(w) >= 2 and w.isalpha()
        )
    except Exception:
        base_words = set()
    expanded_dict, _ = build_expanded_word_set(base_words)
    ref_10k = base_words
    ref_131k = base_words | expanded_dict

    content_words = _build_content_word_set()
    print(f"       {len(ref_10k)} base words, {len(ref_131k)} expanded")
    print(f"       {len(content_words)} content words")
    print(f"       {len(token_evas)} tokens to decode")

    # ── Full corpus decode ────────────────────────────────────────────
    print("\n  C.2  Full corpus decode...")

    # Pre-correction decode (for dark-to-decoded comparison)
    decoded_pre = []
    for eva in token_evas:
        d = decode_token_modifier_aware(
            eva, original_assignment, eva_to_triple, modifier_chars,
        )
        decoded_pre.append(d.lower())

    # Post-correction decode
    decoded_post = []
    for eva in token_evas:
        d = decode_token_modifier_aware(
            eva, corrected_assignment, eva_to_triple, modifier_chars,
        )
        decoded_post.append(d.lower())

    dict_hit_10k = _compute_dict_hit(decoded_post, ref_10k)
    dict_hit_131k = _compute_dict_hit(decoded_post, ref_131k)

    # Signal token count
    n_signal = sum(1 for d in decoded_post if d in SIGNAL_WORDS_SET)
    signal_rate = n_signal / len(decoded_post) if decoded_post else 0.0

    print(f"       Dict-hit (10K):  {dict_hit_10k:.4f}")
    print(f"       Dict-hit (131K): {dict_hit_131k:.4f}")
    print(f"       Signal tokens: {n_signal} ({signal_rate:.1%})")

    # ── Dark-to-decoded count ─────────────────────────────────────────
    newly_decoded = _count_newly_decoded(decoded_pre, decoded_post, ref_131k)
    print(f"\n  C.3  Newly decoded tokens: {newly_decoded}")

    # ── Per-section breakdown ─────────────────────────────────────────
    print("\n  C.4  Per-section breakdown...")
    sections = _section_breakdown(decoded_post, token_folios, ref_131k)
    for section, stats in sections.items():
        print(f"         {section:12s}: {stats['dict_hit']:.1%} "
              f"({stats['n_hits']}/{stats['n_tokens']})")

    # ── Content run detection ─────────────────────────────────────────
    print("\n  C.5  Content run detection...")

    # First find all runs (no content requirement)
    all_runs = _find_all_runs(
        decoded_post, token_folios, token_evas, ref_131k, min_length=3,
    )
    longest_run = all_runs[0]['length'] if all_runs else 0
    print(f"       Total runs (>=3): {len(all_runs)}")
    print(f"       Longest run: {longest_run}")

    if all_runs:
        top_run = all_runs[0]
        print(f"       Best run ({top_run['length']} tokens, "
              f"folio {top_run['folio']}): {top_run['text'][:100]}")

    # Content runs (pharmaceutical/botanical)
    content_runs = _find_content_runs(
        decoded_post, token_folios, token_evas, ref_131k, content_words,
        min_run_length=5, min_content_words=2,
    )
    print(f"       Content runs (>=5, >=2 content words): {len(content_runs)}")

    if content_runs:
        for i, cr in enumerate(content_runs[:5]):
            print(f"         #{i+1}: {cr['folio']} ({cr['length']} tokens, "
                  f"{cr['n_content_words']} content) -> {cr['text'][:80]}")
    else:
        # Relax to find any content-adjacent runs
        relaxed_runs = _find_content_runs(
            decoded_post, token_folios, token_evas, ref_131k, content_words,
            min_run_length=3, min_content_words=1,
        )
        if relaxed_runs:
            print(f"       Relaxed content runs (>=3, >=1 content): "
                  f"{len(relaxed_runs)}")
            for i, cr in enumerate(relaxed_runs[:3]):
                print(f"         #{i+1}: {cr['folio']} ({cr['length']} tokens, "
                      f"{cr['n_content_words']} content) -> {cr['text'][:80]}")

    # ── Best content run ──────────────────────────────────────────────
    best_content_run = {}
    if content_runs:
        bcr = content_runs[0]
        best_content_run = {
            'length': bcr['length'],
            'n_content_words': bcr['n_content_words'],
            'content_words': bcr['content_words'],
            'folio': bcr['folio'],
            'text': bcr['text'],
            'eva_text': bcr.get('eva_text', ''),
        }

    # ── Save ──────────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    result = {
        'corrections_applied': corrections_applied,
        'corpus_stats': {
            'dict_hit_10k': round(dict_hit_10k, 6),
            'dict_hit_131k': round(dict_hit_131k, 6),
            'signal_tokens': n_signal,
            'signal_rate': round(signal_rate, 4),
            'n_total_tokens': len(decoded_post),
        },
        'newly_decoded_tokens': newly_decoded,
        'section_breakdown': sections,
        'longest_run': longest_run,
        'n_all_runs': len(all_runs),
        'n_content_runs': len(content_runs),
        'best_content_run': best_content_run,
        'top_10_content_runs': content_runs[:10],
        'top_10_all_runs': [
            {'length': r['length'], 'folio': r['folio'], 'text': r['text'][:200]}
            for r in all_runs[:10]
        ],
        'runtime_seconds': runtime,
    }

    out_path = _save_json(rd, 'resolved_decode.json', result)
    print(f"\n  Saved -> {out_path}")
    print(f"  Completed in {runtime:.1f}s")
