"""
Phase 29.4 – SIGNAL Phrase Extraction
========================================
Combines bigram matches (29.1), context chains (29.2), and SIGNAL runs
(29.3) into a scored catalog of candidate Latin phrases.  Cross-validates
against null (SIGNAL tokens already exclude null hits by definition).

Dependency chain:
    signal_bigrams.json     (Step 29.1)
    signal_context.json     (Step 29.2)
    signal_folio_read.json  (Step 29.3)
        → signal_phrases.json   (this step)
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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


# Pharmaceutical / medical vocabulary for domain scoring
_PHARMA_STEMS = {
    'herb', 'rad', 'foli', 'flor', 'sem', 'cort', 'aqua', 'vino',
    'ole', 'mel', 'sal', 'cera', 'bene', 'mal', 'bon', 'fort',
    'cal', 'frig', 'sicc', 'humid', 'tere', 'misc', 'cola',
    'coque', 'pulver', 'ung', 'emplastr', 'poti', 'infus',
    'decoc', 'bib', 'sume', 'recipe', 'accipe',
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CandidatePhrase:
    words: List[str]
    source: str
    folio: str
    length: int
    n_confirmed: int
    n_dict_hits: int
    parse_score: float
    domain_score: float
    all_signal: bool
    composite_score: float


@dataclass
class SignalPhraseResult:
    n_candidates: int
    candidates: List[Dict]
    n_unique_phrases: int
    n_all_signal: int
    top_phrase_text: str
    top_phrase_score: float
    top_phrase_source: str
    phrase_sources: Dict
    gate_passed: bool
    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------

def _collect_candidates(
    bigrams_data: Dict,
    context_data: Dict,
    folio_data: Dict,
    signal_words: Set[str],
) -> List[CandidatePhrase]:
    """Merge candidate phrases from all three upstream steps."""
    candidates: List[CandidatePhrase] = []

    # Source 1: bigram matches from 29.1
    for pair in bigrams_data.get('bigram_hit_pairs', []):
        if len(pair) == 2:
            candidates.append(_build_candidate(
                words=pair, source='bigram_match', folio='corpus-wide',
                signal_words=signal_words,
            ))

    # Source 2: trigram matches from 29.1
    for tri in bigrams_data.get('trigram_hit_triples', []):
        if len(tri) == 3:
            candidates.append(_build_candidate(
                words=tri, source='trigram_match', folio='corpus-wide',
                signal_words=signal_words,
            ))

    # Source 3: chains from 29.2
    for chain in context_data.get('chain_candidates', []):
        words = chain.get('words', [])
        if len(words) >= 3:
            candidates.append(_build_candidate(
                words=words, source='context_chain',
                folio=chain.get('folio', ''),
                signal_words=signal_words,
                n_signal=chain.get('n_signal', 0),
            ))

    # Source 4: SIGNAL runs from 29.3
    for run in folio_data.get('all_signal_runs', []):
        words = run.get('decoded_words', [])
        parse_score = run.get('parse_score', 0.0)
        if len(words) >= 2:
            candidates.append(_build_candidate(
                words=words, source='signal_run',
                folio=run.get('folio', ''),
                signal_words=signal_words,
                parse_score_override=parse_score,
                all_signal=True,
            ))

    return candidates


def _build_candidate(
    words: List[str],
    source: str,
    folio: str,
    signal_words: Set[str],
    n_signal: int = 0,
    parse_score_override: Optional[float] = None,
    all_signal: bool = False,
) -> CandidatePhrase:
    """Build a scored candidate phrase."""
    length = len(words)
    n_confirmed = sum(1 for w in words if w in signal_words)

    # Domain score: how many words have pharmaceutical stems
    n_domain = sum(
        1 for w in words
        if any(w.startswith(stem) for stem in _PHARMA_STEMS)
    )
    domain_score = n_domain / length if length > 0 else 0.0

    # Parse score (simple heuristic if not provided)
    if parse_score_override is not None:
        parse_score = parse_score_override
    else:
        parse_score = 0.3  # default for bigram/trigram matches

    # Composite scoring
    length_score = min(length / 5.0, 1.0)
    confirmed_score = n_confirmed / length if length > 0 else 0.0
    dict_score = 1.0  # all candidates are dict hits by construction

    composite = (
        0.20 * length_score
        + 0.25 * confirmed_score
        + 0.20 * dict_score
        + 0.15 * domain_score
        + 0.20 * parse_score
    )

    return CandidatePhrase(
        words=words,
        source=source,
        folio=folio,
        length=length,
        n_confirmed=n_confirmed,
        n_dict_hits=length,
        parse_score=round(parse_score, 3),
        domain_score=round(domain_score, 3),
        all_signal=all_signal,
        composite_score=round(composite, 4),
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(
    candidates: List[CandidatePhrase],
) -> List[CandidatePhrase]:
    """Remove duplicate phrases (same word sequence)."""
    seen: Set[str] = set()
    unique: List[CandidatePhrase] = []
    for c in candidates:
        key = ' '.join(c.words)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_signal_phrases() -> None:
    """Step 29.4: SIGNAL phrase extraction."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 29.4: SIGNAL Phrase Extraction")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load inputs ──
    print("\n  1. Loading inputs …")

    def _load(name: str) -> Dict:
        path = os.path.join(rd, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        print(f"     [WARN] {name} not found")
        return {}

    bigrams_data = _load('signal_bigrams.json')
    context_data = _load('signal_context.json')
    folio_data = _load('signal_folio_read.json')

    # Load signal words
    sig_data = _load('signal_isolation.json')
    signal_words: Set[str] = {
        ws['word'] for ws in sig_data.get('word_signals', [])
        if ws.get('is_genuine_signal')
    }
    print(f"     Signal words: {signal_words}")

    # ── 2. Collect candidates ──
    print("\n  2. Collecting phrase candidates …")
    candidates = _collect_candidates(
        bigrams_data, context_data, folio_data, signal_words,
    )
    print(f"     {len(candidates)} raw candidates")

    # ── 3. Sort and deduplicate ──
    candidates.sort(key=lambda c: -c.composite_score)
    candidates = _deduplicate(candidates)
    print(f"     {len(candidates)} unique candidates")

    # Source breakdown
    source_counts = Counter(c.source for c in candidates)
    print(f"     Sources: {dict(source_counts)}")

    # ── 4. Report top candidates ──
    print("\n  4. Top candidate phrases:")
    n_all_signal = sum(1 for c in candidates if c.all_signal)

    for i, c in enumerate(candidates[:20]):
        tag = '★' if c.n_confirmed > 0 else '○'
        sig_tag = '[ALL_SIGNAL]' if c.all_signal else ''
        print(f"    {tag} #{i + 1}  score={c.composite_score:.3f}  "
              f"len={c.length}  confirmed={c.n_confirmed}  "
              f"parse={c.parse_score:.2f}  domain={c.domain_score:.2f}  "
              f"src={c.source}  {sig_tag}")
        print(f"         {' '.join(c.words)}")

    # ── 5. Gate and verdict ──
    top = candidates[0] if candidates else None
    top_text = ' '.join(top.words) if top else ''
    top_score = top.composite_score if top else 0.0
    top_source = top.source if top else ''

    has_long_phrase = any(c.length >= 3 and c.composite_score >= 0.5
                         for c in candidates)
    has_confirmed_phrase = any(c.n_confirmed >= 2 for c in candidates)

    gate_passed = has_long_phrase or has_confirmed_phrase
    verdict = (
        f"{len(candidates)} unique phrases. "
        f"Top: '{top_text}' (score={top_score:.3f}, src={top_source}). "
        f"{n_all_signal} are all-SIGNAL. "
        f"{'Candidate phrases found.' if gate_passed else 'No strong phrases.'}"
    )
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  {verdict}")

    # ── 6. Save ──
    result = SignalPhraseResult(
        n_candidates=len(candidates),
        candidates=[_convert(asdict(c)) for c in candidates[:100]],
        n_unique_phrases=len(candidates),
        n_all_signal=n_all_signal,
        top_phrase_text=top_text,
        top_phrase_score=top_score,
        top_phrase_source=top_source,
        phrase_sources=dict(source_counts),
        gate_passed=gate_passed,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out_path = os.path.join(rd, 'signal_phrases.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  → {out_path}")
