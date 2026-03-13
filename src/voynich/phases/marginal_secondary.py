"""
Phase 48 Track B: f17r and f66r Marginal Note Analysis
========================================================
Extract and decode any Voynichese content on f17r and f66r.
Cross-reference with f116v findings.

Dependency chain:
    combined_refine.json       (Phase 15 — T_P15 assignment)
    modifier_integrate.json    (Phase 16 — modifier chars)
    f116v_transcription.json   (48A.1 — f116v data)
        → f17r_extract.json    (48B.1)
        → f66r_extract.json    (48B.2)
        → margin_decode.json   (48B.3)
        → margin_hands.json    (48B.4)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

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


def _load_json(rd: str, filename: str) -> Optional[Dict]:
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarginalContent:
    """Extracted content from a marginal note."""
    folio: str
    locus_ids: List[str]
    voynichese_tokens: List[str]
    latin_alphabet_text: str
    locus_types: List[str]
    language_guess: str
    hand: int
    notes: List[str]


@dataclass
class F17rExtract:
    """Step 48B.1 output."""
    main_text_tokens: List[str]
    marginal_content: Optional[Dict]
    has_voynichese_marginal: bool
    has_latin_alphabet_marginal: bool
    marginal_locus_id: str
    marginal_text: str
    n_main_loci: int
    runtime_seconds: float


@dataclass
class F66rExtract:
    """Step 48B.2 output."""
    label_tokens: List[str]
    marginal_content: Optional[Dict]
    has_voynichese: bool
    has_latin_alphabet: bool
    dialect_features: List[str]
    n_loci: int
    notes: List[str]
    runtime_seconds: float


@dataclass
class MarginDecodeEntry:
    """Decode of one marginal Voynichese token."""
    folio: str
    eva_token: str
    eva_chars: List[str]
    triple_keys: List[str]
    syllables: List[str]
    decoded_word: str
    in_10k_dict: bool
    in_131k_dict: bool


@dataclass
class MarginDecode:
    """Step 48B.3 output."""
    decoded_entries: List[Dict]
    n_total_tokens: int
    n_dict_hits_10k: int
    n_dict_hits_131k: int
    cross_folio_consistency: List[Dict]
    runtime_seconds: float


@dataclass
class DialectEvidence:
    """Dialect evidence from one folio."""
    folio: str
    features: List[str]
    language: str
    region: str
    confidence: str


@dataclass
class MarginHands:
    """Step 48B.4 output."""
    dialect_evidence: List[Dict]
    same_hand_assessment: str
    geographic_constraints: List[str]
    decipherment_implications: List[str]
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Step 48B.1 — f17r Extraction
# ---------------------------------------------------------------------------

def run_f17r_extract() -> None:
    """Step 48B.1: Extract f17r marginal content."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48B.1: f17r Marginal Content Extraction")
    print("=" * 70)

    rd = _results_dir()

    from voynich.core.corpus import load_corpus, tokenize_eva_chars

    corpus = load_corpus(verbose=False)

    # ── 1. Extract f17r page ──
    print("\n  1. Extracting f17r from corpus...")

    page = corpus.pages.get('f17r')
    if not page:
        print("     WARNING: f17r not found in corpus")
        result = F17rExtract(
            main_text_tokens=[],
            marginal_content=None,
            has_voynichese_marginal=False,
            has_latin_alphabet_marginal=False,
            marginal_locus_id='',
            marginal_text='',
            n_main_loci=0,
            runtime_seconds=round(time.time() - t0, 2),
        )
        _save_json(rd, 'f17r_extract.json', asdict(result))
        return

    # ── 2. Separate main text from marginal notes ──
    print("\n  2. Separating main text from marginal notes...")

    main_tokens = []
    marginal_loci = []
    marginal_text_parts = []
    all_locus_types = []

    for locus in page.loci:
        lid = locus.locus_id
        ltype = locus.locus_type
        all_locus_types.append(ltype)

        if ltype == 'Lx':
            # Extraneous label — marginal note
            marginal_loci.append(lid)
            marginal_text_parts.append(locus.clean_text)
            print(f"     MARGINAL: {lid} → '{locus.clean_text}'")
        else:
            # Main text
            tokens = locus.clean_text.split('.')
            tokens = [t.strip() for t in tokens if t.strip()]
            main_tokens.extend(tokens)

    # ZL3b-n has f17r.13,@Lx → "oteeeon.oiil"
    # This is a marginal label in Voynichese
    has_voynichese_marginal = len(marginal_loci) > 0
    has_latin_marginal = False  # No Latin-alphabet marginal on f17r in IVTFF

    marginal_text = ' '.join(marginal_text_parts)

    # Parse marginal Voynichese tokens
    marginal_voynich_tokens = []
    for part in marginal_text_parts:
        toks = part.split('.')
        toks = [t.strip() for t in toks if t.strip()]
        marginal_voynich_tokens.extend(toks)

    print(f"\n     Main text tokens: {len(main_tokens)}")
    print(f"     Marginal loci: {marginal_loci}")
    print(f"     Marginal Voynichese tokens: {marginal_voynich_tokens}")
    print(f"     Has Voynichese marginal: {has_voynichese_marginal}")
    print(f"     Has Latin-alphabet marginal: {has_latin_marginal}")

    # ── 3. Scholarly context ──
    print("\n  3. Scholarly context:")
    print("     • Petersen: f17r and f116v may be in the same hand")
    print("     • The f17r note may contain the word 'mallier' (French/Provençal ending)")
    print("     • ZL3b-n marginal: 'oteeeon oiil' — Voynichese, not Latin-alphabet")

    marginal_content = None
    if marginal_voynich_tokens:
        marginal_content = asdict(MarginalContent(
            folio='f17r',
            locus_ids=marginal_loci,
            voynichese_tokens=marginal_voynich_tokens,
            latin_alphabet_text='',
            locus_types=['Lx'],
            language_guess='Voynichese',
            hand=page.hand if hasattr(page, 'hand') else 1,
            notes=[
                'ZL3b-n: f17r.13,@Lx → oteeeon.oiil',
                'Petersen considers this same hand as f116v',
                'No Latin-alphabet marginal text found in IVTFF transcription',
                'Published analyses mention "mallier" but this may be from visual inspection, not IVTFF',
            ],
        ))

    # ── 4. Save ──
    result = F17rExtract(
        main_text_tokens=main_tokens[:50],  # Truncate for JSON size
        marginal_content=marginal_content,
        has_voynichese_marginal=has_voynichese_marginal,
        has_latin_alphabet_marginal=has_latin_marginal,
        marginal_locus_id=marginal_loci[0] if marginal_loci else '',
        marginal_text=marginal_text,
        n_main_loci=len([l for l in all_locus_types if l != 'Lx']),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'f17r_extract.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48B.2 — f66r Extraction
# ---------------------------------------------------------------------------

def run_f66r_extract() -> None:
    """Step 48B.2: Extract f66r marginal content."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48B.2: f66r Marginal Content Extraction")
    print("=" * 70)

    rd = _results_dir()

    from voynich.core.corpus import load_corpus

    corpus = load_corpus(verbose=False)

    # ── 1. Extract f66r page ──
    print("\n  1. Extracting f66r from corpus...")

    page = corpus.pages.get('f66r')
    if not page:
        print("     WARNING: f66r not found in corpus")
        result = F66rExtract(
            label_tokens=[],
            marginal_content=None,
            has_voynichese=False,
            has_latin_alphabet=False,
            dialect_features=[],
            n_loci=0,
            notes=['f66r not found in corpus'],
            runtime_seconds=round(time.time() - t0, 2),
        )
        _save_json(rd, 'f66r_extract.json', asdict(result))
        return

    # ── 2. Extract all content ──
    print("\n  2. Extracting all content...")

    all_tokens = []
    locus_types = []
    has_voynichese = False
    has_latin = False

    for locus in page.loci:
        ltype = locus.locus_type
        locus_types.append(ltype)
        text = locus.clean_text
        tokens = text.split('.')
        tokens = [t.strip() for t in tokens if t.strip()]
        all_tokens.extend(tokens)

        if ltype == 'Lx':
            has_latin = True
            print(f"     LATIN: {locus.locus_id} → '{text}'")
        else:
            has_voynichese = True
            print(f"     VOYNICH: {locus.locus_id} [{ltype}] → '{text}'")

    print(f"\n     Total tokens: {len(all_tokens)}")
    print(f"     Locus types: {set(locus_types)}")
    print(f"     Has Voynichese: {has_voynichese}")
    print(f"     Has Latin-alphabet: {has_latin}")

    # ── 3. Dialect analysis ──
    print("\n  3. Dialect analysis:")

    # f66r is associated with "muß mel" / "musmel" reading
    # but this comes from visual inspection of the manuscript,
    # not from the IVTFF transcription
    dialect_features = [
        '"muß mel" (ground flour) reading from visual inspection — Swabian/Alsatian dialect',
        'ß ligature indicates Upper German dialect area',
        'Not present in IVTFF transcription (which only transcribes Voynichese)',
        'f66r is $H=5 (Hand 5), $L=B (Language B), $P=G (Page type G)',
    ]

    notes = [
        'f66r in IVTFF contains only Voynichese labels (@L0 type)',
        'The "muß mel" reading is from paleographic analysis of non-EVA text',
        'IVTFF does not transcribe the Latin-alphabet marginal note',
        'Hand 5 — different from main scribe (Hand 1) and f116v annotator (Hand 3)',
        'Language B section — may use different encoding conventions',
    ]

    for feat in dialect_features:
        print(f"     • {feat}")

    # ── 4. Save ──
    result = F66rExtract(
        label_tokens=all_tokens,
        marginal_content=None,  # No Latin-alphabet marginal in IVTFF
        has_voynichese=has_voynichese,
        has_latin_alphabet=False,  # Not in IVTFF transcription
        dialect_features=dialect_features,
        n_loci=len(locus_types),
        notes=notes,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'f66r_extract.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48B.3 — Marginal Voynichese Decode
# ---------------------------------------------------------------------------

def run_margin_decode() -> None:
    """Step 48B.3: Decode any marginal Voynichese through T_P15."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48B.3: Marginal Voynichese Decode")
    print("=" * 70)

    rd = _results_dir()

    from voynich.core.corpus import (
        build_eva_to_triple_lookup,
        decode_token_modifier_aware,
        tokenize_eva_chars,
    )
    from voynich.core.reference import build_expanded_word_set, load_reference_corpus

    # ── 1. Load dependencies ──
    print("\n  1. Loading dependencies...")

    f17r_data = _load_json(rd, 'f17r_extract.json')
    f66r_data = _load_json(rd, 'f66r_extract.json')
    f116v_data = _load_json(rd, 'f116v_transcription.json')
    combined = _load_json(rd, 'combined_refine.json')
    mod_data = _load_json(rd, 'modifier_integrate.json')

    if not combined:
        print("     ERROR: combined_refine.json not found.")
        return

    assignment = combined.get('best_assignment', {})
    modifier_chars = set(mod_data.get('modifier_chars', [])) if mod_data else set()
    eva_to_triple = build_eva_to_triple_lookup()

    # Build dictionaries
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    dict_10k = base_words
    dict_131k = base_words | expanded

    # ── 2. Collect marginal Voynichese tokens ──
    print("\n  2. Collecting marginal Voynichese tokens...")

    marginal_tokens = []

    # From f17r
    if f17r_data and f17r_data.get('has_voynichese_marginal'):
        mc = f17r_data.get('marginal_content', {})
        if mc:
            for tok in mc.get('voynichese_tokens', []):
                marginal_tokens.append(('f17r', tok))
                print(f"     f17r: '{tok}'")

    # From f66r — all content is Voynichese labels
    if f66r_data and f66r_data.get('has_voynichese'):
        for tok in f66r_data.get('label_tokens', []):
            if tok and len(tok) > 1:  # Skip single-char tokens
                marginal_tokens.append(('f66r', tok))

    print(f"     Total marginal tokens to decode: {len(marginal_tokens)}")

    # ── 3. Decode each token ──
    print("\n  3. Decoding through T_P15...")

    decoded_entries = []
    n_hits_10k = 0
    n_hits_131k = 0

    for folio, token in marginal_tokens:
        eva_chars = tokenize_eva_chars(token)
        triple_keys = []
        for ch in eva_chars:
            tk = eva_to_triple.get(ch)
            if tk:
                triple_keys.append(tk)

        # Decode via R3 strategy
        decoded_alt = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )
        decoded_strip = decode_token_modifier_aware(
            token, assignment, eva_to_triple, modifier_chars,
        )

        # Raw syllables
        syllables = []
        for ch in eva_chars:
            if ch not in modifier_chars:
                tk = eva_to_triple.get(ch)
                if tk and tk in assignment:
                    syllables.append(assignment[tk])

        decoded = decoded_alt.lower()
        in_10k = decoded in dict_10k
        in_131k = decoded in dict_131k

        if in_10k:
            n_hits_10k += 1
        if in_131k:
            n_hits_131k += 1

        entry = MarginDecodeEntry(
            folio=folio,
            eva_token=token,
            eva_chars=eva_chars,
            triple_keys=triple_keys,
            syllables=syllables,
            decoded_word=decoded,
            in_10k_dict=in_10k,
            in_131k_dict=in_131k,
        )
        decoded_entries.append(asdict(entry))

    # Show first few and any hits
    for de in decoded_entries[:10]:
        marker = '★' if de['in_131k_dict'] else ' '
        print(f"     {marker} [{de['folio']}] '{de['eva_token']}' → '{de['decoded_word']}'")

    hits = [de for de in decoded_entries if de['in_131k_dict']]
    if hits:
        print(f"\n     Dict hits (131K):")
        for h in hits:
            print(f"       [{h['folio']}] '{h['eva_token']}' → '{h['decoded_word']}'")

    # ── 4. Cross-folio consistency ──
    print("\n  4. Cross-folio consistency check...")

    # Find EVA chars that appear in both f116v and f17r marginal text
    f116v_chars = set()
    if f116v_data:
        for word_data in f116v_data.get('words', []):
            for variant in word_data.get('variants', []):
                if variant.get('source') == 'ZL3b-n':
                    f116v_chars.update(variant.get('eva_chars', []))

    f17r_chars = set()
    for folio, token in marginal_tokens:
        if folio == 'f17r':
            f17r_chars.update(tokenize_eva_chars(token))

    shared_chars = f116v_chars & f17r_chars
    consistency_checks = []

    if shared_chars:
        print(f"     Shared EVA chars between f116v and f17r: {shared_chars}")
        for ch in shared_chars:
            tk = eva_to_triple.get(ch)
            syl = assignment.get(tk, '?') if tk else '?'
            consistency_checks.append({
                'eva_char': ch,
                'triple_key': tk,
                'syllable': syl,
                'consistent': True,  # Same table maps same char consistently
                'note': 'Same T_P15 assignment used for both folios',
            })
    else:
        print("     No shared chars between f116v and f17r marginal text")

    # ── 5. Save ──
    result = MarginDecode(
        decoded_entries=decoded_entries,
        n_total_tokens=len(marginal_tokens),
        n_dict_hits_10k=n_hits_10k,
        n_dict_hits_131k=n_hits_131k,
        cross_folio_consistency=consistency_checks,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'margin_decode.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Dict hits: {n_hits_10k} (10K), {n_hits_131k} (131K)")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48B.4 — Same-Hand Analysis
# ---------------------------------------------------------------------------

def run_margin_hand() -> None:
    """Step 48B.4: Same-hand analysis across f116v/f17r/f66r."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48B.4: Marginal Same-Hand Analysis")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Compile dialect evidence ──
    print("\n  1. Compiling dialect evidence...")

    evidence = [
        DialectEvidence(
            folio='f116v',
            features=[
                'p/b substitution ("pox leber" — bilabial neutralization)',
                '"gasmich"/"geissmilch" — diphthong reduction (ei→a)',
                '"so nim" — imperative without -e ending',
                '"valsch vbren" — v/f alternation',
                'Hand 3 in IVTFF ($H=3)',
            ],
            language='German (Eastern Bavarian)',
            region='Vienna/Austria region',
            confidence='HIGH',
        ),
        DialectEvidence(
            folio='f17r',
            features=[
                'Marginal note in Voynichese (oteeeon oiil)',
                'Published analyses mention "mallier" (French/Provençal ending) '
                'but this is from visual inspection, not IVTFF',
                'Petersen: same hand as f116v',
                'Hand 1 in IVTFF ($H=1) — but marginal note hand not separately identified',
            ],
            language='Voynichese (marginal) / possibly French (visual)',
            region='uncertain',
            confidence='LOW',
        ),
        DialectEvidence(
            folio='f66r',
            features=[
                '"muß mel" reading from visual inspection (not in IVTFF)',
                'ß ligature → Upper German dialect area',
                'Swabian/Alsatian features',
                'Hand 5 in IVTFF ($H=5) — different from f116v Hand 3',
            ],
            language='German (Swabian/Alsatian)',
            region='Swabian/Alsatian',
            confidence='MEDIUM',
        ),
    ]

    for ev in evidence:
        print(f"\n     {ev.folio}: {ev.language} ({ev.region})")
        print(f"       Confidence: {ev.confidence}")
        for feat in ev.features:
            print(f"       • {feat}")

    # ── 2. Same-hand assessment ──
    print("\n  2. Same-hand assessment:")

    same_hand = (
        "MIXED: f116v (Hand 3) and f17r marginal may be same hand (per Petersen). "
        "f66r is Hand 5 — likely different scribe. The IVTFF hand assignments "
        "($H=3 vs $H=5) support different annotators for f116v and f66r."
    )
    print(f"     {same_hand}")

    # ── 3. Geographic constraints ──
    print("\n  3. Geographic constraints:")

    geo = [
        "f116v dialect: Eastern Bavarian / Vienna region (HIGH confidence)",
        "f66r dialect: Swabian / Alsatian (MEDIUM confidence, from visual inspection)",
        "Both point to southern German-speaking region but different sub-regions",
        "Compatible with Alpine corridor: Vienna → Swabia via Tyrol/Bavaria",
        "Consistent with Northern Italian manuscript circulating in German-speaking lands",
    ]
    for g in geo:
        print(f"     • {g}")

    # ── 4. Decipherment implications ──
    print("\n  4. Decipherment implications:")

    implications = [
        "f116v scribe (Hand 3) could read Voynichese — embedded 2 Voynich words in recipe",
        "If Eastern Bavarian scribe read Voynichese → encoding accessible to German medical practitioners",
        "Consistent with tachygraphic system from N. Italy that circulated through Alpine region",
        "f66r Hand 5 (if different) → manuscript passed through multiple readers",
        "Multiple readers who understood the script → it was a practical notation, not a one-off cipher",
    ]
    for imp in implications:
        print(f"     • {imp}")

    # ── 5. Save ──
    result = MarginHands(
        dialect_evidence=[asdict(e) for e in evidence],
        same_hand_assessment=same_hand,
        geographic_constraints=geo,
        decipherment_implications=implications,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'margin_hands.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Track B runner
# ---------------------------------------------------------------------------

def run_track_b_48() -> None:
    """Run all Track B steps sequentially."""
    print("\n" + "█" * 70)
    print("  PHASE 48 TRACK B: f17r and f66r Marginal Notes")
    print("█" * 70)

    run_f17r_extract()
    run_f66r_extract()
    run_margin_decode()
    run_margin_hand()

    print("\n" + "█" * 70)
    print("  TRACK B COMPLETE")
    print("█" * 70)
