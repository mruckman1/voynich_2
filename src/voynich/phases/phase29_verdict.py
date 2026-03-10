"""
Phase 29.5 – Phase 29 Verdict
================================
Aggregates results from Steps 29.1–29.4 and classifies the outcome as
PHRASE_FOUND / FRAGMENTS_FOUND / SIGNAL_CONFIRMED_ONLY / WORD_LEVEL_ONLY.

Dependency chain:
    signal_bigrams.json     (Step 29.1)
    signal_context.json     (Step 29.2)
    signal_folio_read.json  (Step 29.3)
    signal_phrases.json     (Step 29.4)
        → phase29_verdict.json   (this step)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from voynich.core._paths import results_dir as _results_dir


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
class Phase29VerdictResult:
    # Signal filtering summary
    n_signal_tokens: int
    signal_rate: float
    n_signal_pairs: int

    # Bigram plausibility
    bigram_hit_rate: float
    bigram_p_value: float
    bigram_z_score: float
    bigram_above_null: bool
    n_trigram_hits: int
    n_relaxed_hits: int

    # Context exploitation
    n_new_cribs: int
    n_chains: int
    longest_chain: int

    # Folio analysis
    top_folio: str
    top_folio_signal_rate: float
    n_signal_runs: int
    n_runs_ge3: int
    longest_run: int

    # Phrases
    n_candidates: int
    n_all_signal: int
    top_phrase: str
    top_phrase_score: float

    # Verdict
    outcome: str
    outcome_description: str
    key_findings: List[str]
    progression: Dict
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

def _classify_outcome(
    bigrams: Dict,
    context: Dict,
    folio: Dict,
    phrases: Dict,
) -> tuple:
    """Classify outcome and return (outcome, description)."""
    bigram_z = bigrams.get('bigram_z_score', 0.0)
    bigram_p = bigrams.get('bigram_p_value', 1.0)
    n_bigram_hits = bigrams.get('n_bigram_hits', 0)
    n_trigram_hits = bigrams.get('n_trigram_hits', 0)

    n_new_cribs = context.get('n_new_crib_candidates', 0)
    longest_chain = context.get('longest_chain', 0)
    n_chains = context.get('n_chains_found', 0)

    n_runs_ge3 = folio.get('n_runs_length_ge3', 0)
    longest_run = folio.get('longest_run', 0)

    top_score = phrases.get('top_phrase_score', 0.0)
    n_candidates = phrases.get('n_candidates', 0)

    # Check candidates for confirmed-word phrases
    has_confirmed_phrase = False
    for c in phrases.get('candidates', []):
        if c.get('length', 0) >= 3 and c.get('n_confirmed', 0) >= 2:
            has_confirmed_phrase = True
            break

    # PHRASE_FOUND: strong multi-word signal
    if (n_trigram_hits > 0
            or (has_confirmed_phrase and bigram_z > 2.0 and bigram_p < 0.05)):
        return (
            'PHRASE_FOUND',
            'At least one multi-word Latin phrase found with statistical '
            'significance above null.  This represents the first decoded '
            'passage from the Voynich manuscript.',
        )

    # FRAGMENTS_FOUND: sequential structure detected
    if (bigram_z > 2.0
            or n_runs_ge3 >= 3
            or longest_chain >= 5
            or (n_bigram_hits > 0 and bigram_p < 0.10)):
        return (
            'FRAGMENTS_FOUND',
            'Sequential structure detected in SIGNAL tokens — bigram '
            'plausibility above null or multiple SIGNAL runs of 3+ tokens. '
            'Fragmentary Latin can be read with gaps.',
        )

    # SIGNAL_CONFIRMED_ONLY: context adds evidence but no phrases
    if (bigram_z > 1.0
            or n_new_cribs >= 2
            or longest_chain >= 3):
        return (
            'SIGNAL_CONFIRMED_ONLY',
            'Context analysis confirms additional crib words beyond the '
            'original 8, but no multi-word phrases emerge above noise. '
            'The signal is at the word level with some sequential hints.',
        )

    # WORD_LEVEL_ONLY: signal is individual words only
    return (
        'WORD_LEVEL_ONLY',
        'The 8 confirmed signal words stand as isolated correct '
        'decodings in a stream of noise.  No sequential structure '
        '(bigrams, phrases, chains) is detectable above the null '
        'baseline.  The signal is real but word-level only.',
    )


# ---------------------------------------------------------------------------
# Progression
# ---------------------------------------------------------------------------

def _build_progression() -> Dict:
    return {
        'Phase 11': '11.1% dict_hit (1.92×)',
        'Phase 14': '19.4% dict_hit (3.00×) — feature model breakthrough',
        'Phase 15': '35.4% dict_hit (2.55×) — dictionary expansion',
        'Phase 16': '43.6% dict_hit (3.38×, full corpus) — modifier detection',
        'Phase 23': 'No historical permutation bridge',
        'Phase 24': '43.6% — error correction degraded; Phase 16 table stands',
        'Phase 26': '39.1% — zodiac NO_SIGNAL',
        'Phase 28': '43.6% — Ventris crib propagation confirms 8 signal words',
        'Phase 29': 'Signal-filtered readability (this phase)',
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase29_verdict() -> None:
    """Step 29.5: Phase 29 verdict."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 29.5: Phase 29 Verdict")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load all upstream results ──
    print("\n  1. Loading upstream results …")

    def _load(name: str) -> Dict:
        path = os.path.join(rd, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        print(f"     [WARN] {name} not found")
        return {}

    bigrams = _load('signal_bigrams.json')
    context = _load('signal_context.json')
    folio = _load('signal_folio_read.json')
    phrases = _load('signal_phrases.json')

    # ── 2. Extract key metrics ──
    print("\n  2. Key metrics:")

    n_signal = bigrams.get('n_signal', 0)
    signal_rate = bigrams.get('signal_rate', 0.0)
    n_signal_pairs = bigrams.get('n_signal_pairs', 0)
    bigram_hit_rate = bigrams.get('bigram_hit_rate', 0.0)
    bigram_p = bigrams.get('bigram_p_value', 1.0)
    bigram_z = bigrams.get('bigram_z_score', 0.0)
    bigram_above = bigram_z > 2.0 and bigram_p < 0.05
    n_trigram_hits = bigrams.get('n_trigram_hits', 0)
    n_relaxed = bigrams.get('n_relaxed_bigram_hits', 0)

    n_new_cribs = context.get('n_new_crib_candidates', 0)
    n_chains = context.get('n_chains_found', 0)
    longest_chain = context.get('longest_chain', 0)

    top_folio = ''
    top_folio_rate = 0.0
    folio_stats = folio.get('folio_signal_pair_stats',
                            folio.get('folio_annotations', []))
    if folio_stats:
        top_folio = folio_stats[0].get('folio', '')
        top_folio_rate = folio_stats[0].get('signal_rate', 0.0)
    top_folios = folio.get('top_folios_analyzed', [])
    if top_folios:
        top_folio = top_folios[0]
    n_runs = folio.get('n_runs_total', 0)
    n_runs_ge3 = folio.get('n_runs_length_ge3', 0)
    longest_run = folio.get('longest_run', 0)

    n_candidates = phrases.get('n_candidates', 0)
    n_all_signal = phrases.get('n_all_signal', 0)
    top_phrase = phrases.get('top_phrase_text', '')
    top_phrase_score = phrases.get('top_phrase_score', 0.0)

    print(f"     SIGNAL tokens: {n_signal} ({signal_rate:.1%})")
    print(f"     SIGNAL pairs:  {n_signal_pairs}")
    print(f"     Bigram hits:   {bigrams.get('n_bigram_hits', 0)} "
          f"(rate={bigram_hit_rate:.6f}, z={bigram_z:.2f}, p={bigram_p:.4f})")
    print(f"     Trigram hits:  {n_trigram_hits}")
    print(f"     Relaxed hits:  {n_relaxed}")
    print(f"     New cribs:     {n_new_cribs}")
    print(f"     Chains:        {n_chains} (longest={longest_chain})")
    print(f"     SIGNAL runs:   {n_runs} ({n_runs_ge3} of length ≥ 3, "
          f"longest={longest_run})")
    print(f"     Phrases:       {n_candidates} ({n_all_signal} all-signal)")
    print(f"     Top phrase:    '{top_phrase}' (score={top_phrase_score:.3f})")

    # ── 3. Classify outcome ──
    print("\n  3. Classifying outcome …")
    outcome, description = _classify_outcome(bigrams, context, folio, phrases)

    print(f"     Outcome: {outcome}")
    print(f"     {description}")

    # ── 4. Key findings ──
    findings: List[str] = []

    if bigram_above:
        findings.append(
            f'SIGNAL bigram plausibility above null (z={bigram_z:.1f}, '
            f'p={bigram_p:.4f})')
    else:
        findings.append(
            f'SIGNAL bigram plausibility NOT above null (z={bigram_z:.1f})')

    if n_trigram_hits > 0:
        findings.append(f'{n_trigram_hits} Latin trigram matches in SIGNAL tokens')

    if n_new_cribs > 0:
        findings.append(f'{n_new_cribs} new crib candidates from context analysis')

    if longest_chain >= 3:
        findings.append(f'Longest dict-hit chain with SIGNAL: {longest_chain} tokens')

    if longest_run >= 3:
        findings.append(
            f'Longest consecutive SIGNAL run: {longest_run} tokens '
            f'on {folio.get("best_run_folio", "")}')

    if n_relaxed > 0:
        findings.append(f'{n_relaxed} relaxed bigram matches (edit distance 1)')

    progression = _build_progression()

    # ── 5. Gate and verdict ──
    gate_passed = outcome in ('PHRASE_FOUND', 'FRAGMENTS_FOUND')
    verdict = (
        f"{outcome}: {description[:100]}… "
        f"Bigram z={bigram_z:.2f}, {n_new_cribs} new cribs, "
        f"longest_chain={longest_chain}, longest_run={longest_run}."
    )
    print(f"\n  Outcome: {outcome}")
    print(f"  Gate: {'PASS' if gate_passed else 'FAIL'}")

    # ── 6. Save ──
    result = Phase29VerdictResult(
        n_signal_tokens=n_signal,
        signal_rate=signal_rate,
        n_signal_pairs=n_signal_pairs,
        bigram_hit_rate=bigram_hit_rate,
        bigram_p_value=bigram_p,
        bigram_z_score=bigram_z,
        bigram_above_null=bigram_above,
        n_trigram_hits=n_trigram_hits,
        n_relaxed_hits=n_relaxed,
        n_new_cribs=n_new_cribs,
        n_chains=n_chains,
        longest_chain=longest_chain,
        top_folio=top_folio,
        top_folio_signal_rate=top_folio_rate,
        n_signal_runs=n_runs,
        n_runs_ge3=n_runs_ge3,
        longest_run=longest_run,
        n_candidates=n_candidates,
        n_all_signal=n_all_signal,
        top_phrase=top_phrase,
        top_phrase_score=top_phrase_score,
        outcome=outcome,
        outcome_description=description,
        key_findings=findings,
        progression=progression,
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'phase29_verdict.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
