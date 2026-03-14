"""
Phase 52 Track C: Word-Level Structural Reading
================================================
Merge signal words with bridge-identified word IDs to produce
annotated folio-level readings of the Voynich manuscript.

Dependency chain:
    word_catalog.json          (Track A)
    signal_bigrams.json        (Step 29.1)
        -> word_reading.json   (this step)
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51


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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedToken:
    position: int
    folio: str
    eva_token: str
    decoded: str
    source: str           # TIER_0, T1, T2, T3, DARK
    gloss: str
    confidence: float


@dataclass
class FolioReading:
    folio: str
    section: str
    n_tokens: int
    n_glossed: int
    coverage: float
    max_consecutive: int
    reading_text: str
    tokens: Optional[List[Dict]]  # full detail only for top folios


@dataclass
class ConsecutiveRun:
    start_pos: int
    length: int
    folio: str
    text: str
    n_content_words: int   # non-function-word glossed tokens


@dataclass
class WordReadingResult:
    # Vocabulary merge
    total_vocabulary: int
    n_tier0: int
    n_tier1: int
    n_tier2: int
    n_tier3: int
    # Corpus annotation
    n_tokens: int
    n_glossed: int
    overall_coverage: float
    # Folio readings
    n_folios: int
    folio_readings: List[Dict]
    best_folio: str
    best_folio_coverage: float
    best_folio_max_consecutive: int
    best_folio_reading: str
    # Consecutive runs
    longest_run_length: int
    longest_run_folio: str
    longest_run_text: str
    top_10_runs: List[Dict]
    # Content analysis
    n_content_words_in_vocab: int
    circa_instans_overlap: float
    domain_distribution: Dict[str, int]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Section inference from folio name
# ---------------------------------------------------------------------------

def _folio_section(folio: str) -> str:
    """Infer manuscript section from folio ID."""
    if not folio:
        return 'unknown'
    num_str = folio.lstrip('f').rstrip('rv')
    try:
        num = int(num_str.split('v')[0].split('r')[0])
    except ValueError:
        return 'unknown'

    if num <= 25:
        return 'herbal_a'
    elif num <= 56:
        return 'herbal_b'
    elif num <= 67:
        return 'pharmaceutical'
    elif num <= 73:
        return 'zodiac'
    elif num <= 84:
        return 'biological'
    elif num <= 86:
        return 'cosmological'
    elif num <= 102:
        return 'herbal_c'
    elif num <= 116:
        return 'stars'
    else:
        return 'unknown'


# ---------------------------------------------------------------------------
# Circa Instans overlap
# ---------------------------------------------------------------------------

def _circa_instans_overlap(catalog_words: Set[str]) -> float:
    """Compute fraction of catalog words that appear in reference corpora."""
    from voynich.core.reference import load_reference_corpus
    try:
        ref = load_reference_corpus(languages=['latin'], verbose=False)
        ref_words: Set[str] = set()
        for text in ref.get_texts('latin'):
            for tok in text.tokens:
                ref_words.add(tok.lower())
        if not catalog_words:
            return 0.0
        overlap = catalog_words & ref_words
        return len(overlap) / len(catalog_words)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_word_reading() -> None:
    """Phase 52 Track C: Word-Level Structural Reading."""
    t0 = time.time()

    print("=" * 70)
    print("PHASE 52 TRACK C: Word-Level Structural Reading")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load catalog ──────────────────────────────────────────────
    print("\n  C.1  Loading catalog and corpus data...")

    catalog_data = _safe_load(os.path.join(rd, 'word_catalog.json'))
    if not catalog_data:
        print("  *** word_catalog.json not found — run Track A first ***")
        return

    with open(os.path.join(rd, 'signal_bigrams.json')) as f:
        bigram_data = json.load(f)
    token_evas = bigram_data['token_evas']
    token_decoded = bigram_data['token_decoded']
    token_folios = bigram_data['token_folios']
    n_tokens = len(token_evas)

    # ── 2. Build merged vocabulary ───────────────────────────────────
    print("\n  C.2  Building merged vocabulary...")

    # TIER_0: signal words (decoded form → info)
    signal_lookup: Dict[str, Dict] = {}
    for word, info in SIGNAL_WORDS_51.items():
        signal_lookup[word] = {
            'gloss': info['gloss'],
            'type': info['type'],
            'source': 'TIER_0',
            'confidence': 1.0,
        }

    # T1/T2/T3: catalog entries (EVA type → best match)
    # If multiple words for same EVA type, pick highest confidence
    catalog_lookup: Dict[str, Dict] = {}
    for wid in catalog_data.get('single_token_ids', []):
        eva = wid['eva_type']
        if eva not in catalog_lookup or wid['confidence'] > catalog_lookup[eva].get('confidence', 0):
            catalog_lookup[eva] = {
                'latin_word': wid['latin_word'],
                'tier': wid['tier'],
                'confidence': wid['confidence'],
                'is_ambiguous': wid.get('is_ambiguous', False),
                'competing_words': wid.get('competing_words', []),
            }

    n_tier0 = len(signal_lookup)
    n_tier1 = sum(1 for w in catalog_data.get('single_token_ids', [])
                  if w['tier'] == 'T1')
    n_tier2 = sum(1 for w in catalog_data.get('single_token_ids', [])
                  if w['tier'] == 'T2')
    n_tier3 = sum(1 for w in catalog_data.get('single_token_ids', [])
                  if w['tier'] == 'T3')
    total_vocab = n_tier0 + len(catalog_lookup)

    print(f"       TIER_0 (signal): {n_tier0}")
    print(f"       T1 catalog: {n_tier1}")
    print(f"       T2 catalog: {n_tier2}")
    print(f"       T3 catalog: {n_tier3}")
    print(f"       Total vocabulary: {total_vocab}")

    # ── 3. Annotate every token ──────────────────────────────────────
    print("\n  C.3  Annotating corpus...")

    # Function word set for content word detection
    function_words = set(w for w, info in SIGNAL_WORDS_51.items()
                         if info['type'] == 'function')

    annotated: List[AnnotatedToken] = []
    n_glossed = 0

    for i in range(n_tokens):
        decoded = token_decoded[i]
        eva = token_evas[i]
        folio = token_folios[i]

        if decoded in signal_lookup:
            info = signal_lookup[decoded]
            tok = AnnotatedToken(
                position=i, folio=folio, eva_token=eva,
                decoded=decoded, source='TIER_0',
                gloss=info['gloss'], confidence=1.0,
            )
            n_glossed += 1
        elif eva in catalog_lookup:
            cat = catalog_lookup[eva]
            tok = AnnotatedToken(
                position=i, folio=folio, eva_token=eva,
                decoded=cat['latin_word'], source=cat['tier'],
                gloss=cat['latin_word'], confidence=cat['confidence'],
            )
            n_glossed += 1
        else:
            tok = AnnotatedToken(
                position=i, folio=folio, eva_token=eva,
                decoded='', source='DARK',
                gloss='', confidence=0.0,
            )

        annotated.append(tok)

    overall_coverage = n_glossed / n_tokens if n_tokens > 0 else 0.0
    print(f"       Glossed: {n_glossed} / {n_tokens} ({overall_coverage:.1%})")

    # ── 4. Folio-level readings ──────────────────────────────────────
    print("\n  C.4  Building folio-level readings...")

    folio_tokens: Dict[str, List[AnnotatedToken]] = defaultdict(list)
    for tok in annotated:
        folio_tokens[tok.folio].append(tok)

    folio_readings: List[FolioReading] = []

    for folio, tokens in folio_tokens.items():
        n_tok = len(tokens)
        n_gl = sum(1 for t in tokens if t.source != 'DARK')
        cov = n_gl / n_tok if n_tok > 0 else 0.0

        # Max consecutive glossed
        max_consec = 0
        cur_consec = 0
        for t in tokens:
            if t.source != 'DARK':
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 0

        # Build reading text
        parts = []
        in_gap = False
        for t in tokens:
            if t.source != 'DARK':
                if in_gap:
                    parts.append('[...]')
                    in_gap = False
                parts.append(t.decoded)
            else:
                in_gap = True
        if in_gap:
            parts.append('[...]')
        reading_text = ' '.join(parts)

        section = _folio_section(folio)

        folio_readings.append(FolioReading(
            folio=folio,
            section=section,
            n_tokens=n_tok,
            n_glossed=n_gl,
            coverage=round(cov, 4),
            max_consecutive=max_consec,
            reading_text=reading_text,
            tokens=None,
        ))

    # Sort by coverage * sqrt(n_tokens)
    folio_readings.sort(
        key=lambda fr: fr.coverage * (fr.n_tokens ** 0.5),
        reverse=True,
    )

    # Fill full token detail for top 5
    for fr in folio_readings[:5]:
        fr.tokens = [asdict(t) for t in folio_tokens[fr.folio]]

    print(f"       {len(folio_readings)} folios annotated")
    print(f"       Top 5 folios:")
    for fr in folio_readings[:5]:
        print(f"         {fr.folio} ({fr.section}): "
              f"{fr.coverage:.1%} coverage, "
              f"max_consec={fr.max_consecutive}, "
              f"{fr.n_tokens} tokens")

    best = folio_readings[0] if folio_readings else None

    # ── 5. Find longest consecutive glossed runs ─────────────────────
    print("\n  C.5  Finding longest consecutive glossed runs...")

    runs: List[ConsecutiveRun] = []
    cur_start = -1
    cur_len = 0

    for i, tok in enumerate(annotated):
        if tok.source != 'DARK':
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len >= 3:
                run_tokens = annotated[cur_start:cur_start + cur_len]
                text = ' '.join(t.decoded for t in run_tokens)
                n_content = sum(1 for t in run_tokens
                                if t.decoded not in function_words
                                and t.source != 'TIER_0')
                runs.append(ConsecutiveRun(
                    start_pos=cur_start, length=cur_len,
                    folio=annotated[cur_start].folio, text=text,
                    n_content_words=n_content,
                ))
            cur_len = 0

    # Capture final run
    if cur_len >= 3:
        run_tokens = annotated[cur_start:cur_start + cur_len]
        text = ' '.join(t.decoded for t in run_tokens)
        n_content = sum(1 for t in run_tokens
                        if t.decoded not in function_words
                        and t.source != 'TIER_0')
        runs.append(ConsecutiveRun(
            start_pos=cur_start, length=cur_len,
            folio=annotated[cur_start].folio, text=text,
            n_content_words=n_content,
        ))

    runs.sort(key=lambda r: r.length, reverse=True)
    top_runs = runs[:10]

    print(f"       Runs of length >= 3: {len(runs)}")
    if top_runs:
        print(f"       Longest: {top_runs[0].length} tokens on {top_runs[0].folio}")
        for r in top_runs[:5]:
            print(f"         len={r.length} folio={r.folio} "
                  f"content={r.n_content_words}: {r.text[:80]}")

    # ── 6. Content vocabulary analysis ───────────────────────────────
    print("\n  C.6  Content vocabulary analysis...")

    domain_dist: Dict[str, int] = Counter()
    content_words: Set[str] = set()

    for wid in catalog_data.get('single_token_ids', []):
        content_words.add(wid['latin_word'].lower())

    for word, info in SIGNAL_WORDS_51.items():
        domain_dist[info['type']] += 1

    from voynich.core.reference import PHARMACEUTICAL_VOCABULARY
    pharma_terms: Set[str] = set()
    for terms in PHARMACEUTICAL_VOCABULARY.values():
        for t in terms:
            pharma_terms.add(t.lower())

    n_pharma = sum(1 for w in content_words if w in pharma_terms)
    domain_dist['catalog_pharmaceutical'] = n_pharma
    domain_dist['catalog_other'] = len(content_words) - n_pharma
    n_content_in_vocab = len(content_words)

    circa_overlap = _circa_instans_overlap(content_words)
    print(f"       Content words in catalog: {n_content_in_vocab}")
    print(f"       Pharmaceutical: {n_pharma}")
    print(f"       Circa Instans overlap: {circa_overlap:.1%}")

    # ── 7. Save ──────────────────────────────────────────────────────
    runtime = round(time.time() - t0, 2)

    longest = top_runs[0] if top_runs else None

    result = WordReadingResult(
        total_vocabulary=total_vocab,
        n_tier0=n_tier0,
        n_tier1=n_tier1,
        n_tier2=n_tier2,
        n_tier3=n_tier3,
        n_tokens=n_tokens,
        n_glossed=n_glossed,
        overall_coverage=round(overall_coverage, 4),
        n_folios=len(folio_readings),
        folio_readings=[asdict(fr) for fr in folio_readings[:20]],
        best_folio=best.folio if best else '',
        best_folio_coverage=best.coverage if best else 0.0,
        best_folio_max_consecutive=best.max_consecutive if best else 0,
        best_folio_reading=best.reading_text[:500] if best else '',
        longest_run_length=longest.length if longest else 0,
        longest_run_folio=longest.folio if longest else '',
        longest_run_text=longest.text[:300] if longest else '',
        top_10_runs=[asdict(r) for r in top_runs],
        n_content_words_in_vocab=n_content_in_vocab,
        circa_instans_overlap=round(circa_overlap, 4),
        domain_distribution=dict(domain_dist),
        runtime_seconds=runtime,
    )

    out_path = _save_json(rd, 'word_reading.json', asdict(result))
    print(f"\n  Saved → {out_path}")
    print(f"  Completed in {runtime:.1f}s")
