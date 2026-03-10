"""
Phase 30.5 – Post-Bootstrap Folio Examination
=================================================
Re-examines the top SIGNAL folios with the expanded vocabulary from the
bootstrap, producing annotated transliterations with additional tags for
bootstrap-confirmed words.

Dependency chain:
    bootstrap_bigrams.json     (Step 30.3 — new per-token cache)
    bootstrap_signal.json      (Step 30.2 — signal words)
    bootstrap_loop.json        (Step 30.1 — accepted words)
    bootstrap_context.json     (Step 30.4 — new crib candidates)
    signal_folio_read.json     (Phase 29.3 — baseline)
    signal_isolation.json      (Phase 28.4 — original signal words)
        → bootstrap_folio.json    (this step)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import _infer_section
from voynich.phases.signal_folio_read import (
    SignalRun,
    _attempt_latin_parse,
)


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


# Medical / pharmaceutical domain words for domain checking
_MEDICAL_STEMS = {
    'herba', 'foli', 'radi', 'flor', 'semen', 'cort', 'bals',
    'calid', 'frigid', 'humid', 'sicc',
    'recipe', 'coqu', 'misc', 'cola', 'distill', 'filt',
    'aqua', 'oleum', 'vinum', 'mel', 'acetum',
    'febre', 'dolor', 'morb', 'infirm',
    'bene', 'male', 'bon', 'optim',
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    n_confirmed_orig: int
    n_confirmed_boot: int
    n_candidates: int
    domain_words: List[str]
    domain_match_rate: float


@dataclass
class BootstrapFolioResult:
    top_folios_analyzed: List[str]
    folio_annotations: List[Dict]
    all_signal_runs: List[Dict]
    n_runs_total: int
    n_runs_length_ge3: int
    longest_run: int
    best_run_folio: str
    best_run_text: str
    best_run_parse_score: float
    f6r_comparison: Dict
    # Bootstrap comparison
    baseline_longest_run: int
    baseline_n_runs_ge3: int
    delta_longest_run: int
    delta_n_runs_ge3: int
    new_top_folios: List[str]
    # Best fragment
    best_fragment_words: List[str]
    best_fragment_folio: str
    best_fragment_parse: str
    best_fragment_score: float
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Folio annotation with bootstrap tags
# ---------------------------------------------------------------------------

def _annotate_folio_bootstrap(
    folio: str,
    token_folios: List[str],
    token_evas: List[str],
    token_decoded: List[str],
    token_classifications: List[str],
    token_dict_hits: List[bool],
    confirmed_orig: Set[str],
    confirmed_boot: Set[str],
    candidate_words: Set[str],
) -> FolioAnnotation:
    """Build annotated data for a single folio with bootstrap-specific tags."""
    indices = [i for i, f in enumerate(token_folios) if f == folio]
    section = _infer_section(folio)

    if not indices:
        return FolioAnnotation(
            folio=folio, section=section,
            n_tokens=0, n_signal=0, signal_rate=0.0,
            n_shared_hit=0, n_anti_signal=0,
            annotated_text='', plain_text_with_gaps='',
            signal_runs=[], max_run_length=0, dict_hit_rate=0.0,
            n_confirmed_orig=0, n_confirmed_boot=0, n_candidates=0,
            domain_words=[], domain_match_rate=0.0,
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

    n_conf_orig = 0
    n_conf_boot = 0
    n_cand = 0

    # Annotated text with bootstrap tags
    ann_parts = []
    gap_parts = []
    for j in range(n_tokens):
        cls = cls_list[j]
        word = dec_list[j]
        if cls == 'SIGNAL':
            if word in confirmed_orig:
                ann_parts.append(f'[CONFIRMED-ORIG:{word}]')
                n_conf_orig += 1
            elif word in confirmed_boot:
                ann_parts.append(f'[CONFIRMED-BOOT:{word}]')
                n_conf_boot += 1
            elif word in candidate_words:
                ann_parts.append(f'[CANDIDATE:{word}]')
                n_cand += 1
            else:
                ann_parts.append(f'[SIGNAL:{word}]')
            gap_parts.append(word)
        elif cls == 'SHARED_HIT':
            ann_parts.append(f'[SHARED:{word}]')
            gap_parts.append('[...]')
        elif cls == 'ANTI_SIGNAL':
            ann_parts.append(f'[ANTI:{word}]')
            gap_parts.append('[...]')
        else:
            ann_parts.append('[MISS]')
            gap_parts.append('[...]')

    annotated_text = ' '.join(ann_parts)

    # Collapse consecutive [...] in plain text
    collapsed = []
    prev_gap = False
    for part in gap_parts:
        if part == '[...]':
            if not prev_gap:
                collapsed.append('[...]')
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

    # Domain check
    domain_words = []
    for w in set(dec_list):
        for stem in _MEDICAL_STEMS:
            if w.startswith(stem) or stem.startswith(w):
                domain_words.append(w)
                break
    domain_rate = len(domain_words) / n_tokens if n_tokens > 0 else 0.0

    return FolioAnnotation(
        folio=folio,
        section=section,
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
        n_confirmed_orig=n_conf_orig,
        n_confirmed_boot=n_conf_boot,
        n_candidates=n_cand,
        domain_words=domain_words,
        domain_match_rate=round(domain_rate, 4),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bootstrap_folio() -> None:
    """Step 30.5: Post-bootstrap annotated folio examination."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 30.5: Post-Bootstrap Folio Examination")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load per-token cache ──
    print("\n  1. Loading per-token cache …")
    boot_bg_path = os.path.join(rd, 'bootstrap_bigrams.json')
    bg_path = os.path.join(rd, 'signal_bigrams.json')

    if os.path.exists(boot_bg_path):
        with open(boot_bg_path) as f:
            bg_data = json.load(f)
        print("     Using bootstrap_bigrams.json")
    elif os.path.exists(bg_path):
        with open(bg_path) as f:
            bg_data = json.load(f)
        print("     Using signal_bigrams.json (fallback)")
    else:
        print("  [SKIP] No per-token cache found")
        return

    token_folios = bg_data['token_folios']
    token_evas = bg_data['token_evas']
    token_decoded = bg_data['token_decoded']
    token_classifications = bg_data['token_classifications']
    token_dict_hits = bg_data['token_dict_hits']

    # ── 2. Load word sets ──
    print("\n  2. Loading word sets …")

    # Original signal words
    sig_path = os.path.join(rd, 'signal_isolation.json')
    confirmed_orig: Set[str] = set()
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        confirmed_orig = {
            ws['word'] for ws in sig_data.get('word_signals', [])
            if ws.get('is_genuine_signal', False)
        }

    # Bootstrap-accepted words
    boot_path = os.path.join(rd, 'bootstrap_loop.json')
    confirmed_boot: Set[str] = set()
    if os.path.exists(boot_path):
        with open(boot_path) as f:
            boot_data = json.load(f)
        confirmed_boot = set(boot_data.get('accepted_words', []))

    # Context candidates
    ctx_path = os.path.join(rd, 'bootstrap_context.json')
    if not os.path.exists(ctx_path):
        ctx_path = os.path.join(rd, 'signal_context.json')
    candidate_words: Set[str] = set()
    if os.path.exists(ctx_path):
        with open(ctx_path) as f:
            ctx_data = json.load(f)
        candidate_words = {
            c['word'] for c in ctx_data.get('new_crib_candidates', [])
        }

    print(f"     Confirmed-orig: {len(confirmed_orig)}")
    print(f"     Confirmed-boot: {len(confirmed_boot)}")
    print(f"     Candidates: {len(candidate_words)}")

    # Baseline
    folio_baseline_path = os.path.join(rd, 'signal_folio_read.json')
    baseline_longest = 0
    baseline_n_ge3 = 0
    baseline_top_folios: List[str] = []
    if os.path.exists(folio_baseline_path):
        with open(folio_baseline_path) as f:
            folio_baseline = json.load(f)
        baseline_longest = folio_baseline.get('longest_run', 0)
        baseline_n_ge3 = folio_baseline.get('n_runs_length_ge3', 0)
        baseline_top_folios = folio_baseline.get('top_folios_analyzed', [])

    # ── 3. Rank folios by signal rate ──
    print("\n  3. Ranking folios by signal rate …")
    folio_indices: Dict[str, List[int]] = defaultdict(list)
    for i, f in enumerate(token_folios):
        folio_indices[f].append(i)

    folio_rates = []
    for folio, indices in folio_indices.items():
        n = len(indices)
        if n < 20:
            continue
        n_sig = sum(1 for i in indices if token_classifications[i] == 'SIGNAL')
        folio_rates.append((folio, n_sig / n, n))
    folio_rates.sort(key=lambda x: -x[1])

    top_folios = [f for f, _, _ in folio_rates[:4]]
    # Always include f6r for comparison
    if 'f6r' not in top_folios:
        top_folios.append('f6r')

    print(f"     Top folios: {', '.join(top_folios)}")

    # ── 4. Annotate each folio ──
    print("\n  4. Annotating folios …")
    annotations: List[FolioAnnotation] = []
    all_runs: List[SignalRun] = []

    for folio in top_folios:
        ann = _annotate_folio_bootstrap(
            folio, token_folios, token_evas, token_decoded,
            token_classifications, token_dict_hits,
            confirmed_orig, confirmed_boot, candidate_words,
        )
        annotations.append(ann)

        print(f"     {folio:8s}  tokens={ann.n_tokens:3d}  "
              f"signal={ann.n_signal:3d} ({ann.signal_rate:.1%})  "
              f"orig={ann.n_confirmed_orig}  boot={ann.n_confirmed_boot}  "
              f"runs={len(ann.signal_runs)}  max_run={ann.max_run_length}")

        # Collect runs
        for r in ann.signal_runs:
            all_runs.append(SignalRun(**{k: v for k, v in r.items()}))

    # Also collect runs from ALL folios for the global ranking
    print("\n  5. Collecting SIGNAL runs across all folios …")
    for folio in folio_indices:
        if folio in top_folios:
            continue
        ann = _annotate_folio_bootstrap(
            folio, token_folios, token_evas, token_decoded,
            token_classifications, token_dict_hits,
            confirmed_orig, confirmed_boot, candidate_words,
        )
        for r in ann.signal_runs:
            all_runs.append(SignalRun(**{k: v for k, v in r.items()}))

    # Sort all runs by length descending
    all_runs.sort(key=lambda r: (-r.length, -r.parse_score))
    n_ge3 = sum(1 for r in all_runs if r.length >= 3)
    longest = all_runs[0].length if all_runs else 0
    best_run_folio = all_runs[0].folio if all_runs else ''
    best_run_text = ' '.join(all_runs[0].decoded_words) if all_runs else ''

    print(f"     Total runs: {len(all_runs)}, ≥3: {n_ge3}, longest: {longest}")

    # ── 6. Best fragment selection ──
    print("\n  6. Selecting best fragment …")
    best_frag_words: List[str] = []
    best_frag_folio = ''
    best_frag_parse = ''
    best_frag_score = 0.0

    for r in all_runs[:20]:
        if r.length >= 3 and r.parse_score >= best_frag_score:
            best_frag_words = r.decoded_words
            best_frag_folio = r.folio
            best_frag_parse = r.latin_parse
            best_frag_score = r.parse_score

    if best_frag_words:
        print(f"     Best fragment: {' '.join(best_frag_words)}")
        print(f"     Folio: {best_frag_folio}, parse_score: {best_frag_score:.3f}")
        print(f"     Parse: {best_frag_parse}")

    # f6r comparison
    f6r_ann = next((a for a in annotations if a.folio == 'f6r'), None)
    f6r_comparison = {}
    if f6r_ann:
        f6r_comparison = {
            'signal_rate': f6r_ann.signal_rate,
            'dict_hit_rate': f6r_ann.dict_hit_rate,
            'n_confirmed_orig': f6r_ann.n_confirmed_orig,
            'n_confirmed_boot': f6r_ann.n_confirmed_boot,
            'max_run_length': f6r_ann.max_run_length,
        }

    # ── 7. Comparison ──
    delta_longest = longest - baseline_longest
    delta_ge3 = n_ge3 - baseline_n_ge3
    new_top = [f for f in top_folios[:4] if f not in baseline_top_folios]

    print(f"\n  7. Comparison to Phase 29 baseline …")
    print(f"     Longest run: {baseline_longest} → {longest} (Δ={delta_longest:+d})")
    print(f"     Runs ≥3: {baseline_n_ge3} → {n_ge3} (Δ={delta_ge3:+d})")
    if new_top:
        print(f"     New top folios: {', '.join(new_top)}")

    # Gate
    gate = longest >= 3 or len(all_runs) >= 5
    if longest >= 5:
        verdict = f"FOLIO_STRONG (longest_run={longest})"
    elif longest >= 3:
        verdict = f"FOLIO_MAINTAINED (longest_run={longest})"
    else:
        verdict = f"FOLIO_WEAK (longest_run={longest})"

    print(f"\n     Verdict: {verdict}")
    print(f"     Gate: {'PASS' if gate else 'FAIL'}")

    result = BootstrapFolioResult(
        top_folios_analyzed=top_folios,
        folio_annotations=[_convert(asdict(a)) for a in annotations],
        all_signal_runs=[_convert(asdict(r)) for r in all_runs[:100]],
        n_runs_total=len(all_runs),
        n_runs_length_ge3=n_ge3,
        longest_run=longest,
        best_run_folio=best_run_folio,
        best_run_text=best_run_text,
        best_run_parse_score=best_frag_score,
        f6r_comparison=f6r_comparison,
        baseline_longest_run=baseline_longest,
        baseline_n_runs_ge3=baseline_n_ge3,
        delta_longest_run=delta_longest,
        delta_n_runs_ge3=delta_ge3,
        new_top_folios=new_top,
        best_fragment_words=best_frag_words,
        best_fragment_folio=best_frag_folio,
        best_fragment_parse=best_frag_parse,
        best_fragment_score=best_frag_score,
        gate_passed=gate,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    out_path = os.path.join(rd, 'bootstrap_folio.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
