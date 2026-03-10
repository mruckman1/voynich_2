"""
Phase 29.3 – SIGNAL Folio Deep Examination
=============================================
For the top 4 SIGNAL folios, produces annotated transliterations showing
which tokens are SIGNAL, extracts maximal SIGNAL runs, attempts Latin
parses, and generates plain-text-with-gaps output for human reading.

Dependency chain:
    signal_bigrams.json   (Step 29.1 — per-token cache)
    signal_context.json   (Step 29.2 — new crib candidates)
        → signal_folio_read.json   (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import _infer_section


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
class SignalRun:
    folio: str
    start_local_idx: int
    length: int
    eva_tokens: List[str]
    decoded_words: List[str]
    latin_parse: str
    parse_score: float


@dataclass
class FolioAnnotation:
    folio: str
    section: str
    n_tokens: int
    n_signal: int
    signal_rate: float
    n_shared_hit: int
    n_anti_signal: int
    annotated_text: str
    plain_text_with_gaps: str
    signal_runs: List[Dict]
    max_run_length: int
    dict_hit_rate: float


@dataclass
class SignalFolioResult:
    top_folios_analyzed: List[str]
    folio_annotations: List[Dict]
    all_signal_runs: List[Dict]
    n_runs_total: int
    n_runs_length_ge3: int
    longest_run: int
    best_run_folio: str
    best_run_text: str
    f6r_comparison: Dict
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# POS heuristic (shared with signal_context)
# ---------------------------------------------------------------------------

_PREPOSITIONS = {'de', 'in', 'ad', 'cum', 'per', 'pro', 'sub', 'ex', 'ab'}
_CONJUNCTIONS = {'et', 'vel', 'aut', 'sed', 'si', 'ne', 'ut'}

_SUFFIX_POS = [
    ('ntur', 'VERB'), ('tur', 'VERB'), ('nt', 'VERB'),
    ('mus', 'VERB'), ('tis', 'VERB'),
    ('are', 'VERB'), ('ere', 'VERB'), ('ire', 'VERB'),
    ('um', 'NOUN_ACC'), ('am', 'NOUN_ACC'), ('em', 'NOUN_ACC'),
    ('us', 'NOUN_NOM'), ('er', 'NOUN_NOM'),
    ('ae', 'GEN'), ('is', 'GEN'),
    ('a', 'NOUN_NOM'), ('i', 'GEN'), ('o', 'ABL'), ('e', 'ABL'),
]


def _suffix_pos(word: str) -> str:
    if word in _PREPOSITIONS:
        return 'PREP'
    if word in _CONJUNCTIONS:
        return 'CONJ'
    for suffix, pos in _SUFFIX_POS:
        if word.endswith(suffix) and len(word) > len(suffix):
            return pos
    return 'UNK'


# ---------------------------------------------------------------------------
# Latin parse heuristic
# ---------------------------------------------------------------------------

# Pairs of POS tags that form plausible Latin sequences
_GOOD_POS_PAIRS = {
    ('PREP', 'NOUN_ACC'), ('PREP', 'NOUN_NOM'), ('PREP', 'ABL'),
    ('PREP', 'GEN'), ('PREP', 'UNK'),
    ('NOUN_NOM', 'VERB'), ('NOUN_ACC', 'VERB'),
    ('CONJ', 'NOUN_NOM'), ('CONJ', 'NOUN_ACC'), ('CONJ', 'VERB'),
    ('VERB', 'NOUN_ACC'), ('VERB', 'ABL'),
    ('NOUN_NOM', 'NOUN_NOM'),  # apposition
    ('NOUN_NOM', 'GEN'),       # noun + genitive
    ('UNK', 'VERB'), ('VERB', 'UNK'),
    ('UNK', 'NOUN_NOM'), ('NOUN_NOM', 'UNK'),
}


def _attempt_latin_parse(words: List[str]) -> Tuple[str, float]:
    """Heuristic Latin parsing of a word sequence.

    Returns (parsed_text, score) where score is 0.0–1.0 based on
    how many consecutive POS pairs are grammatically plausible.
    """
    if len(words) < 2:
        return ' '.join(words), 0.0

    pos_tags = [_suffix_pos(w) for w in words]
    n_pairs = len(words) - 1
    n_good = 0
    for i in range(n_pairs):
        if (pos_tags[i], pos_tags[i + 1]) in _GOOD_POS_PAIRS:
            n_good += 1

    score = n_good / n_pairs if n_pairs > 0 else 0.0
    parsed = ' '.join(f'{w}[{p}]' for w, p in zip(words, pos_tags))
    return parsed, round(score, 3)


# ---------------------------------------------------------------------------
# Folio annotation
# ---------------------------------------------------------------------------

def _build_folio_data(
    folio: str,
    token_folios: List[str],
    token_evas: List[str],
    token_decoded: List[str],
    token_classifications: List[str],
    token_dict_hits: List[bool],
    signal_words: Set[str],
    new_crib_words: Set[str],
) -> FolioAnnotation:
    """Build annotated data for a single folio."""
    # Extract indices for this folio
    indices = [i for i, f in enumerate(token_folios) if f == folio]
    if not indices:
        return FolioAnnotation(
            folio=folio, section=_infer_section(folio),
            n_tokens=0, n_signal=0, signal_rate=0.0,
            n_shared_hit=0, n_anti_signal=0,
            annotated_text='', plain_text_with_gaps='',
            signal_runs=[], max_run_length=0, dict_hit_rate=0.0,
        )

    n_tokens = len(indices)
    cls_list = [token_classifications[i] for i in indices]
    dec_list = [token_decoded[i] for i in indices]
    eva_list = [token_evas[i] for i in indices]
    hit_list = [token_dict_hits[i] for i in indices]

    n_signal = sum(1 for c in cls_list if c == 'SIGNAL')
    n_shared_hit = sum(1 for c in cls_list if c == 'SHARED_HIT')
    n_anti = sum(1 for c in cls_list if c == 'ANTI_SIGNAL')
    n_hits = sum(hit_list)

    # Annotated text
    ann_parts = []
    gap_parts = []
    for j in range(n_tokens):
        cls = cls_list[j]
        word = dec_list[j]
        if cls == 'SIGNAL':
            if word in signal_words:
                ann_parts.append(f'[CONFIRMED:{word}]')
            elif word in new_crib_words:
                ann_parts.append(f'[CONTEXT:{word}]')
            else:
                ann_parts.append(f'[SIGNAL:{word}]')
            gap_parts.append(word)
        elif cls == 'SHARED_HIT':
            ann_parts.append(f'[SHARED:{word}]')
            gap_parts.append('[…]')
        else:
            ann_parts.append(f'[---]')
            gap_parts.append('[…]')

    annotated_text = ' '.join(ann_parts)
    plain_text = ' '.join(gap_parts)

    # Collapse consecutive [...] in plain text
    collapsed = []
    prev_gap = False
    for part in gap_parts:
        if part == '[…]':
            if not prev_gap:
                collapsed.append('[…]')
                prev_gap = True
        else:
            collapsed.append(part)
            prev_gap = False
    plain_text = ' '.join(collapsed)

    # Signal runs
    runs: List[SignalRun] = []
    i = 0
    while i < n_tokens:
        if cls_list[i] != 'SIGNAL':
            i += 1
            continue
        start = i
        while i < n_tokens and cls_list[i] == 'SIGNAL':
            i += 1
        length = i - start
        if length >= 2:
            run_words = dec_list[start:i]
            run_evas = eva_list[start:i]
            parsed, score = _attempt_latin_parse(run_words)
            runs.append(SignalRun(
                folio=folio,
                start_local_idx=start,
                length=length,
                eva_tokens=run_evas,
                decoded_words=run_words,
                latin_parse=parsed,
                parse_score=score,
            ))

    max_run = max((r.length for r in runs), default=0)

    return FolioAnnotation(
        folio=folio,
        section=_infer_section(folio),
        n_tokens=n_tokens,
        n_signal=n_signal,
        signal_rate=round(n_signal / n_tokens, 4) if n_tokens > 0 else 0.0,
        n_shared_hit=n_shared_hit,
        n_anti_signal=n_anti,
        annotated_text=annotated_text,
        plain_text_with_gaps=plain_text,
        signal_runs=[_convert(asdict(r)) for r in runs],
        max_run_length=max_run,
        dict_hit_rate=round(n_hits / n_tokens, 4) if n_tokens > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_signal_folio_read() -> None:
    """Step 29.3: SIGNAL folio deep examination."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 29.3: SIGNAL Folio Deep Examination")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    bg_path = os.path.join(rd, 'signal_bigrams.json')
    if not os.path.exists(bg_path):
        print("  [SKIP] signal_bigrams.json not found — run signal-bigram first")
        return
    with open(bg_path) as f:
        bg_data = json.load(f)

    token_folios = bg_data['token_folios']
    token_evas = bg_data['token_evas']
    token_decoded = bg_data['token_decoded']
    token_classifications = bg_data['token_classifications']
    token_dict_hits = bg_data['token_dict_hits']

    # Load signal words
    sig_path = os.path.join(rd, 'signal_isolation.json')
    signal_words: Set[str] = set()
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        signal_words = {
            ws['word'] for ws in sig_data.get('word_signals', [])
            if ws.get('is_genuine_signal')
        }

    # Load new crib candidates from 29.2
    ctx_path = os.path.join(rd, 'signal_context.json')
    new_crib_words: Set[str] = set()
    if os.path.exists(ctx_path):
        with open(ctx_path) as f:
            ctx_data = json.load(f)
        new_crib_words = {
            nc['word'] for nc in ctx_data.get('new_crib_candidates', [])
        }

    print(f"     {len(token_decoded)} tokens, "
          f"{len(signal_words)} signal words, "
          f"{len(new_crib_words)} new crib words")

    # ── 2. Select top folios by signal rate ──
    print("\n  2. Selecting top SIGNAL folios …")
    folio_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'total': 0, 'signal': 0},
    )
    for folio, cls in zip(token_folios, token_classifications):
        folio_counts[folio]['total'] += 1
        if cls == 'SIGNAL':
            folio_counts[folio]['signal'] += 1

    folio_ranked = sorted(
        folio_counts.items(),
        key=lambda kv: -(kv[1]['signal'] / kv[1]['total']
                         if kv[1]['total'] > 10 else 0),
    )

    # Top 4 with at least 20 tokens
    top_folios = [
        f for f, c in folio_ranked
        if c['total'] >= 20
    ][:4]

    for f in top_folios:
        c = folio_counts[f]
        rate = c['signal'] / c['total']
        print(f"     {f:8s}  {c['signal']:3d}/{c['total']:3d}  ({rate:.1%})")

    # ── 3. Annotate each top folio ──
    print("\n  3. Building folio annotations …")
    annotations: List[FolioAnnotation] = []
    all_runs: List[Dict] = []

    for folio in top_folios:
        ann = _build_folio_data(
            folio, token_folios, token_evas, token_decoded,
            token_classifications, token_dict_hits,
            signal_words, new_crib_words,
        )
        annotations.append(ann)
        all_runs.extend(ann.signal_runs)

        print(f"\n     ── {folio} ({ann.section}) ──")
        print(f"     {ann.n_tokens} tokens, {ann.n_signal} SIGNAL "
              f"({ann.signal_rate:.1%}), "
              f"dict_hit={ann.dict_hit_rate:.1%}")
        print(f"     {len(ann.signal_runs)} SIGNAL runs "
              f"(max length {ann.max_run_length})")

        # Print plain-text-with-gaps (truncated)
        plain = ann.plain_text_with_gaps
        if len(plain) > 300:
            plain = plain[:300] + ' …'
        print(f"     Plain text: {plain}")

        # Print top runs
        for run in sorted(ann.signal_runs,
                          key=lambda r: -r['length'])[:3]:
            print(f"       Run len={run['length']}: "
                  f"{' '.join(run['decoded_words'][:10])}"
                  f"{'…' if run['length'] > 10 else ''}"
                  f"  parse_score={run['parse_score']}")

    # ── 4. f6r comparison ──
    print("\n  4. f6r comparison …")
    f6r_counts = folio_counts.get('f6r', {'total': 0, 'signal': 0})
    f6r_signal_rate = (
        f6r_counts['signal'] / f6r_counts['total']
        if f6r_counts['total'] > 0 else 0.0
    )
    f6r_dict_hits = sum(
        1 for f, h in zip(token_folios, token_dict_hits)
        if f == 'f6r' and h
    )
    f6r_dict_rate = (
        f6r_dict_hits / f6r_counts['total']
        if f6r_counts['total'] > 0 else 0.0
    )

    f6r_comparison = {
        'f6r_n_tokens': f6r_counts['total'],
        'f6r_n_signal': f6r_counts['signal'],
        'f6r_signal_rate': round(f6r_signal_rate, 4),
        'f6r_dict_hit_rate': round(f6r_dict_rate, 4),
        'top_folio': top_folios[0] if top_folios else '',
        'top_folio_signal_rate': (
            annotations[0].signal_rate if annotations else 0.0
        ),
    }
    print(f"     f6r: signal_rate={f6r_signal_rate:.1%}, "
          f"dict_hit={f6r_dict_rate:.1%}")
    if top_folios:
        print(f"     {top_folios[0]}: signal_rate="
              f"{annotations[0].signal_rate:.1%}")

    # ── 5. Summary ──
    n_runs = len(all_runs)
    n_runs_ge3 = sum(1 for r in all_runs if r['length'] >= 3)
    longest_run = max((r['length'] for r in all_runs), default=0)
    best_run = max(all_runs, key=lambda r: r['length']) if all_runs else None
    best_run_folio = best_run['folio'] if best_run else ''
    best_run_text = (
        ' '.join(best_run['decoded_words'])
        if best_run else ''
    )

    gate_passed = longest_run >= 3 or n_runs >= 5
    verdict = (
        f"{n_runs} SIGNAL runs across {len(top_folios)} folios, "
        f"{n_runs_ge3} of length ≥ 3, "
        f"longest={longest_run} on {best_run_folio}. "
        f"{'Fragmentary signal sequences found.' if gate_passed else 'No substantial runs.'}"
    )
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 6. Save ──
    result = SignalFolioResult(
        top_folios_analyzed=top_folios,
        folio_annotations=[_convert(asdict(a)) for a in annotations],
        all_signal_runs=all_runs[:100],
        n_runs_total=n_runs,
        n_runs_length_ge3=n_runs_ge3,
        longest_run=longest_run,
        best_run_folio=best_run_folio,
        best_run_text=best_run_text,
        f6r_comparison=f6r_comparison,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'signal_folio_read.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
