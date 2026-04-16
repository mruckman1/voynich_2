"""
Phase 82: Decoding Pipeline Trace (Reviewer 3.8)
=================================================
Show the complete EVA -> decoded word pipeline for specific tokens.
Addresses reviewer complaint that EVA source text is not shown alongside
decoded passages, making decodings unverifiable.

Produces traces for:
  - The 3 words the reviewer specifically asks about: chedy, daiin, qokeedy
  - Tokens from example passages (f54r, f57v)
  - The word-level identifications from Table 5

Each trace shows: EVA token -> EVA chars -> char roles -> triples ->
syllables -> decoded CV / decoded CVC.

Output: results/p82_decode_trace.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import (
    build_eva_to_triple_lookup,
    load_corpus,
    tokenize_eva_chars,
)
from voynich.phases.coda_markers import get_coda
from voynich.phases.corrected_coda import (
    classify_token_chars_v2,
    decode_token_cvc_v2,
)
from voynich.phases.p75_redecode import _build_3coda_table


# ---------------------------------------------------------------------------
# JSON helpers (standard pattern)
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
class CharTrace:
    """Trace for a single EVA character within a token."""
    eva_char: str
    role: str                    # SYLLABIC, CODA_MARKER
    triple: Optional[str]        # e.g. "open_curve,connector,bench" or None
    syllable: Optional[str]      # e.g. "co" or None (if unresolved/modifier)
    coda: Optional[str]          # e.g. "n" or None


@dataclass
class TokenTrace:
    """Complete decode trace for one EVA token."""
    eva_token: str
    eva_chars: List[str]
    char_traces: List[CharTrace]
    decoded_cv: str
    decoded_cvc: str
    matched_word: Optional[str]  # from T1 identifications, if any
    folio: Optional[str]         # folio where this token appears (first)


@dataclass
class PassageTrace:
    """A sequence of traced tokens from a folio."""
    folio: str
    section: str
    token_traces: List[TokenTrace]
    decoded_cv_sequence: str     # space-separated CV decodes
    decoded_cvc_sequence: str    # space-separated CVC decodes
    eva_sequence: str            # space-separated EVA tokens


@dataclass
class DecodeTraceResult:
    phase: str = "82"
    experiment: str = "decode_trace"
    # Reviewer-requested specific words
    reviewer_words: List[TokenTrace] = field(default_factory=list)
    # Example passages from the paper
    passage_traces: List[PassageTrace] = field(default_factory=list)
    # Table 5 word-level identifications with full traces
    table5_traces: List[TokenTrace] = field(default_factory=list)
    # Summary stats
    n_reviewer_words: int = 0
    n_passages: int = 0
    n_table5: int = 0
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core trace function
# ---------------------------------------------------------------------------

def trace_token(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
    matched_word: Optional[str] = None,
    folio: Optional[str] = None,
) -> TokenTrace:
    """Produce a complete decode trace for one EVA token."""
    eva_chars = tokenize_eva_chars(token)
    classified = classify_token_chars_v2(eva_chars, coda_table)

    char_traces = []
    for role, char in classified:
        triple = None
        syllable = None
        coda = None

        if role == 'SYLLABIC':
            triple = eva_to_triple.get(char)
            if triple:
                syllable = assignment.get(triple, '?')
        elif role == 'CODA_MARKER':
            coda = get_coda(char, coda_table)

        char_traces.append(CharTrace(
            eva_char=char,
            role=role,
            triple=triple,
            syllable=syllable,
            coda=coda,
        ))

    # Get CV and CVC decodes
    result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)

    return TokenTrace(
        eva_token=token,
        eva_chars=eva_chars,
        char_traces=char_traces,
        decoded_cv=result.decoded_cv,
        decoded_cvc=result.decoded_cvc,
        matched_word=matched_word,
        folio=folio,
    )


# ---------------------------------------------------------------------------
# Table 5 identifications (from the paper)
# ---------------------------------------------------------------------------

TABLE_5_ENTRIES = [
    ('otol', 'ratione', 'by method'),
    ('oty', 'rabidi', 'of the fierce'),
    ('qopchedy', 'stercora', 'dung (med.)'),
    ('ytol', 'diasene', 'senna cpd.'),
    ('chotar', 'coralli', 'of corals'),
    ('chkain', 'codex', 'codex'),
    ('otcham', 'radicom', 'root (acc.)'),
    ('chtol', 'commune', 'common'),
    ('shty', 'secundi', 'of the second'),
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_decode_trace():
    """Phase 82: Produce decode traces for reviewer response."""
    t0 = time.time()
    rd = _results_dir()
    print("Phase 82: Decoding Pipeline Trace")
    print("=" * 60)

    # Load resources
    combined = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = combined.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = _build_3coda_table()
    corpus = load_corpus(verbose=False)

    # Build token -> first folio lookup
    token_folios: Dict[str, str] = {}
    for folio, page in corpus.pages.items():
        for tok in page.all_tokens:
            if tok not in token_folios:
                token_folios[tok] = folio

    # ------------------------------------------------------------------
    # 1. Reviewer-requested specific words
    # ------------------------------------------------------------------
    reviewer_tokens = ['chedy', 'daiin', 'qokeedy']
    reviewer_traces = []
    print("\n--- Reviewer-requested words ---")
    for tok in reviewer_tokens:
        trace = trace_token(
            tok, assignment, eva_to_triple, coda_table,
            folio=token_folios.get(tok),
        )
        reviewer_traces.append(trace)
        _print_trace(trace)

    # ------------------------------------------------------------------
    # 2. Example passages from the paper
    # ------------------------------------------------------------------
    passage_traces = []

    # Find passages on specific folios
    passage_folios = ['f54r', 'f57v']
    for target_folio in passage_folios:
        page = corpus.pages.get(target_folio)
        if not page:
            print(f"\n  [Folio {target_folio} not found]")
            continue

        tokens = page.all_tokens
        if not tokens:
            continue

        # Get section
        section = getattr(page, 'illustration', '') or 'unknown'

        # Trace all tokens on the folio
        tok_traces = []
        for tok in tokens:
            trace = trace_token(
                tok, assignment, eva_to_triple, coda_table,
                folio=target_folio,
            )
            tok_traces.append(trace)

        passage = PassageTrace(
            folio=target_folio,
            section=section,
            token_traces=tok_traces,
            decoded_cv_sequence=' '.join(t.decoded_cv for t in tok_traces),
            decoded_cvc_sequence=' '.join(t.decoded_cvc for t in tok_traces),
            eva_sequence=' '.join(t.eva_token for t in tok_traces),
        )
        passage_traces.append(passage)

        print(f"\n--- Passage: {target_folio} ({len(tokens)} tokens) ---")
        print(f"  EVA: {passage.eva_sequence[:120]}...")
        print(f"  CV:  {passage.decoded_cv_sequence[:120]}...")
        print(f"  CVC: {passage.decoded_cvc_sequence[:120]}...")

    # ------------------------------------------------------------------
    # 3. Table 5 word-level identifications
    # ------------------------------------------------------------------
    table5_traces = []
    print("\n--- Table 5 word-level identifications ---")
    for eva_type, latin_word, gloss in TABLE_5_ENTRIES:
        trace = trace_token(
            eva_type, assignment, eva_to_triple, coda_table,
            matched_word=latin_word,
            folio=token_folios.get(eva_type),
        )
        table5_traces.append(trace)
        _print_trace(trace, gloss=gloss)

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    result = DecodeTraceResult(
        reviewer_words=reviewer_traces,
        passage_traces=passage_traces,
        table5_traces=table5_traces,
        n_reviewer_words=len(reviewer_traces),
        n_passages=len(passage_traces),
        n_table5=len(table5_traces),
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'p82_decode_trace.json', result)
    print(f"\n  Saved -> {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    return result


def _print_trace(trace: TokenTrace, gloss: str = ''):
    """Pretty-print a single token trace."""
    chars_str = ' | '.join(trace.eva_chars)
    print(f"\n  Token: {trace.eva_token}")
    print(f"    EVA chars: [{chars_str}]")
    for ct in trace.char_traces:
        if ct.role == 'SYLLABIC':
            triple_short = ct.triple.replace(',', '/') if ct.triple else '?'
            print(f"      {ct.eva_char:8s} SYLLABIC  triple={triple_short:40s} -> {ct.syllable or '?'}")
        elif ct.role == 'CODA_MARKER':
            print(f"      {ct.eva_char:8s} CODA      coda={ct.coda or 'null'}")
    print(f"    Decoded CV:  {trace.decoded_cv}")
    print(f"    Decoded CVC: {trace.decoded_cvc}")
    if trace.matched_word:
        extra = f" ({gloss})" if gloss else ''
        print(f"    Matched:     {trace.matched_word}{extra}")
    if trace.folio:
        print(f"    First folio: {trace.folio}")
