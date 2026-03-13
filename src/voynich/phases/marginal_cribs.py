"""
Phase 48 Track A: f116v Voynichese Word Decoding and Contextual Matching
=========================================================================
Decode the two embedded Voynichese words on f116v through the locked T_P15
table.  Test decoded output against all competing interpretations of the
surrounding Latin/German text.

Dependency chain:
    combined_refine.json       (Phase 15 — T_P15 assignment)
    modifier_integrate.json    (Phase 16 — modifier chars)
    signal_isolation.json      (Phase 28 — signal words)
        → f116v_transcription.json   (48A.1)
        → f116v_decode.json          (48A.2)
        → f116v_context.json         (48A.3)
        → f116v_match.json           (48A.4)
        → f116v_reverse.json         (48A.5)
"""

import json
import os
import time
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


def _load_json(rd: str, filename: str) -> Optional[Dict]:
    path = os.path.join(rd, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data) if hasattr(data, '__dataclass_fields__') else _convert(data), f, indent=2)
    return path


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionVariant:
    """One transcription of a Voynichese word on f116v."""
    source: str                    # e.g. "ZL3b-n", "RF1b-e"
    eva_token: str                 # e.g. "oror"
    eva_chars: List[str]           # e.g. ["o","r","o","r"]
    triple_keys: List[str]         # stroke triples for each char
    triple_tiers: List[str]        # "tier1"/"tier2"/"tier3" per triple
    n_unknown_chars: int           # chars not in standard EVA inventory


@dataclass
class F116vTranscription:
    """Step 48A.1 output."""
    words: List[Dict]              # per-word: list of TranscriptionVariant dicts
    positional_context: Dict       # where in f116v text the words appear
    surrounding_text: str          # Latin/German context from IVTFF comments
    n_transcription_sources: int
    runtime_seconds: float


@dataclass
class DecodeResult:
    """Decode of one word variant through T_P15."""
    source: str
    eva_token: str
    syllables: List[str]           # per-triple syllable from T_P15
    decoded_word: str              # concatenated syllables
    in_10k_dict: bool
    in_131k_dict: bool
    in_italian_dict: bool
    is_signal_word: bool
    lattice_alternatives: List[str]  # other possible decodings


@dataclass
class F116vDecode:
    """Step 48A.2 output."""
    primary_decodes: List[Dict]    # one per word (best transcription)
    all_variant_decodes: List[Dict]  # all transcription variants
    n_dict_hits_10k: int
    n_dict_hits_131k: int
    n_signal_matches: int
    runtime_seconds: float


@dataclass
class ContextualReading:
    """One scholarly reading of the f116v surrounding text."""
    scholar: str
    year: int
    genre: str                     # "recipe", "charm", "statement", etc.
    language: str                  # "Latin+German", "Italian", etc.
    summary: str
    voynichese_role: str           # "ingredient", "verb", "charm_word", etc.
    semantic_field: List[str]      # expected word categories
    candidate_words: List[str]     # words that would fit the position


@dataclass
class F116vContext:
    """Step 48A.3 output."""
    readings: List[Dict]
    n_readings: int
    consensus_role: str            # most common role across readings
    union_candidates: List[str]    # all candidate words across readings
    runtime_seconds: float


@dataclass
class MatchResult:
    """Match of one decoded word against one reading."""
    decoded_word: str
    reading_scholar: str
    match_level: str               # STRONG_MATCH / PARTIAL_MATCH / WEAK_MATCH / NO_MATCH
    best_candidate: str
    edit_distance: int
    semantic_match: bool
    is_signal_word: bool


@dataclass
class F116vMatch:
    """Step 48A.4 output."""
    matches: List[Dict]
    best_match_level: str
    best_match_word: str
    best_match_reading: str
    gate_result: str               # CRIB_VIABLE / CRIB_FAILED
    runtime_seconds: float


@dataclass
class ReverseCandidate:
    """Reverse-engineered assignment for one candidate word."""
    candidate_word: str
    syllables: List[str]
    required_assignments: Dict     # triple_key -> required syllable
    n_agree_t_p15: int
    n_conflict_t_p15: int
    conflicting_triples: List[Dict]  # which triples need changing
    conflicts_tier1: bool
    testable: bool                 # True if ≤2 non-tier1 conflicts


@dataclass
class F116vReverse:
    """Step 48A.5 output."""
    candidates: List[Dict]
    n_testable: int
    best_candidate: str
    best_n_conflicts: int
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Known f116v scholarly readings
# ---------------------------------------------------------------------------

CONTEXTUAL_READINGS = [
    ContextualReading(
        scholar="Albus",
        year=2012,
        genre="medical_recipe",
        language="Latin+German",
        summary="Medical recipe for wound plaster. 'poxleber umen[do] putriter' = "
                "'Billy goat liver for wet rot'. Voynichese words appear between "
                "Latin recipe instructions and German continuation.",
        voynichese_role="ingredient",
        semantic_field=["pharmaceutical", "ingredient", "preparation"],
        candidate_words=[
            "oleum", "aqua", "pulvis", "herba", "radix", "semen",
            "mel", "cera", "sal", "vinum", "acetum", "gummi",
            "folia", "cortex", "succus", "emplastrum", "unguentum",
        ],
    ),
    ContextualReading(
        scholar="Petersen",
        year=2016,
        genre="healing_charm",
        language="Latin+German",
        summary="Healing charm / medical incantation. Middle lines with crosses "
                "are 'abracadabra-style' charm words. Voynichese words may be "
                "magical/power words in the charm formula.",
        voynichese_role="charm_word",
        semantic_field=["charm", "incantation", "magical"],
        candidate_words=[
            "abra", "sator", "agla", "tetra", "alpha", "omega",
            "amen", "pax", "lux", "via", "vita", "rosa",
        ],
    ),
    ContextualReading(
        scholar="Winkler",
        year=2018,
        genre="statement",
        language="Latin+German",
        summary="'pox leber primum putrefacit' = 'roe buck liver is the first "
                "thing that putrefies'. Different parsing of first line — not a "
                "recipe title but a statement about decomposition.",
        voynichese_role="ingredient",
        semantic_field=["pharmaceutical", "animal_product", "preparation"],
        candidate_words=[
            "oleum", "pulvis", "fel", "sanguis", "adeps", "medulla",
            "butyrum", "caseus", "lac", "sebum", "caro",
        ],
    ),
    ContextualReading(
        scholar="Sherwood",
        year=2015,
        genre="recipe",
        language="Italian",
        summary="Italian interpretation (no added letters). Different language "
                "identification entirely. Voynichese words are Italian vocabulary.",
        voynichese_role="ingredient",
        semantic_field=["pharmaceutical", "Italian"],
        candidate_words=[
            "olio", "erba", "acqua", "sale", "miele", "cera",
            "vino", "aceto", "fiore", "foglia", "radice", "seme",
        ],
    ),
    ContextualReading(
        scholar="VoynichTemple",
        year=2023,
        genre="cookbook_recipe",
        language="German_dialect",
        summary="Eastern Bavarian dialect, Vienna region. 'so nim' = 'so take' "
                "(common in cookbooks). Voynichese words may be ingredient names "
                "the annotator did not know how to write in Latin/German.",
        voynichese_role="ingredient",
        semantic_field=["culinary", "ingredient", "spice"],
        candidate_words=[
            "piper", "crocus", "sal", "mel", "butyrum", "caseus",
            "caro", "farina", "ova", "lac", "vinum", "oleum",
            "zingiber", "cynamomum", "galanga",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Step 48A.1 — EVA Transcription Extraction
# ---------------------------------------------------------------------------

def run_f116v_transcribe() -> None:
    """Step 48A.1: Extract and verify EVA transcription of Voynichese words on f116v."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48A.1: f116v EVA Transcription Extraction")
    print("=" * 70)

    rd = _results_dir()

    from voynich.core.corpus import (
        build_eva_to_triple_lookup,
        load_corpus,
        tokenize_eva_chars,
    )

    # Load corpus to get f116v data
    corpus = load_corpus(verbose=False)
    eva_to_triple = build_eva_to_triple_lookup()

    # Load tier information from triple_tiers.json (Phase 45C)
    tier_data = _load_json(rd, 'triple_tiers.json')
    tier_map = {}
    if tier_data:
        for entry in tier_data.get('evidence_table', []):
            tk = entry.get('triple_key', '')
            tier = entry.get('tier', '')
            if tier == 'CONFIRMED':
                tier_map[tk] = 'tier1'
            elif tier == 'LANDSCAPE_CONFIRMED':
                tier_map[tk] = 'tier2'
            else:
                tier_map[tk] = 'tier3'

    # ── 1. Extract f116v from corpus ──
    print("\n  1. Extracting f116v from corpus...")

    f116v_page = corpus.pages.get('f116v')
    if not f116v_page:
        print("     WARNING: f116v not found in corpus")

    # Known transcriptions of the Voynichese words
    # ZL3b-n: <f116v.1,@Lx> oror.sheey<!valsch vbren so nim gaf mich o>
    # RF1b-e: <f116v.1,@Lx> oror.sheey
    transcription_sources = {
        'ZL3b-n': {
            'tokens': ['oror', 'sheey'],
            'surrounding': 'valsch vbren so nim gaf mich o',
            'locus': 'f116v.1,@Lx',
        },
        'RF1b-e': {
            'tokens': ['oror', 'sheey'],
            'surrounding': '',
            'locus': 'f116v.1,@Lx',
        },
        'Palmer2004': {
            'tokens': ['oror', 'sheey'],
            'surrounding': '',
            'locus': 'f116v (published reading)',
        },
    }

    # ── 2. Tokenize and map to triples ──
    print("\n  2. Tokenizing EVA characters and mapping to triples...")

    all_words = []  # one entry per Voynichese word position
    word_tokens = ['oror', 'sheey']  # the two embedded words

    for word_idx, token in enumerate(word_tokens):
        variants = []
        for src_name, src_data in transcription_sources.items():
            src_token = src_data['tokens'][word_idx] if word_idx < len(src_data['tokens']) else token

            eva_chars = tokenize_eva_chars(src_token)
            triple_keys = []
            triple_tiers = []
            n_unknown = 0

            for ch in eva_chars:
                tk = eva_to_triple.get(ch)
                if tk:
                    triple_keys.append(tk)
                    triple_tiers.append(tier_map.get(tk, 'tier3'))
                else:
                    triple_keys.append(f'UNKNOWN:{ch}')
                    triple_tiers.append('unknown')
                    n_unknown += 1

            variant = TranscriptionVariant(
                source=src_name,
                eva_token=src_token,
                eva_chars=eva_chars,
                triple_keys=triple_keys,
                triple_tiers=triple_tiers,
                n_unknown_chars=n_unknown,
            )
            variants.append(asdict(variant))

            print(f"     [{src_name}] '{src_token}' → chars={eva_chars} "
                  f"→ triples={triple_keys}")

        all_words.append({
            'word_index': word_idx,
            'position': f"f116v line 1 (between Latin recipe and German text)",
            'variants': variants,
        })

    # ── 3. Positional context ──
    print("\n  3. Positional context...")

    # The IVTFF comment gives us the Latin-alphabet text surrounding the words:
    # "valsch vbren so nim gaf mich o"
    # Palmer, Albus, etc. provide expanded readings of the full page
    positional = {
        'folio': 'f116v',
        'locus_type': '@Lx',  # label, extraneous (non-standard text)
        'line': 1,
        'preceding_text': '(Latin recipe text, lines 1-3)',
        'following_text': 'valsch vbren so nim gaf mich o (German, line 4 continuation)',
        'structural_role': 'Embedded between Latin recipe and German instructions',
        'ivtff_comment': 'valsch vbren so nim gaf mich o',
        'hand': 3,  # $H=3 in IVTFF header
        'notes': [
            'Hand 3 (not main scribe) — later annotator',
            'Classified as @Lx (extraneous label) in IVTFF',
            'Goat sketch accompanies the text',
            'All three transcription sources agree on reading: oror sheey',
        ],
    }

    print(f"     Locus: {positional['locus_type']} (Hand {positional['hand']})")
    print(f"     Following text: {positional['ivtff_comment']}")
    print(f"     All sources agree: oror sheey")

    # ── 4. Cross-reference transcriptions ──
    print("\n  4. Cross-referencing transcriptions...")
    all_agree = True
    for word_data in all_words:
        tokens_set = set(v['eva_token'] for v in word_data['variants'])
        if len(tokens_set) > 1:
            all_agree = False
            print(f"     Word {word_data['word_index']}: DISAGREEMENT — {tokens_set}")
        else:
            print(f"     Word {word_data['word_index']}: AGREE — {tokens_set.pop()}")

    # ── 5. Save ──
    surrounding = "valsch vbren so nim gaf mich o"
    result = F116vTranscription(
        words=all_words,
        positional_context=positional,
        surrounding_text=surrounding,
        n_transcription_sources=len(transcription_sources),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'f116v_transcription.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48A.2 — T_P15 Decode
# ---------------------------------------------------------------------------

def run_f116v_decode() -> None:
    """Step 48A.2: Decode f116v Voynichese words through T_P15 table."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48A.2: f116v T_P15 Decode")
    print("=" * 70)

    rd = _results_dir()

    from voynich.core.corpus import (
        build_eva_to_triple_lookup,
        decode_token_modifier_aware,
        tokenize_eva_chars,
    )
    from voynich.core.reference import (
        build_expanded_word_set,
        load_reference_corpus,
    )

    # ── 1. Load dependencies ──
    print("\n  1. Loading dependencies...")

    transcription = _load_json(rd, 'f116v_transcription.json')
    if not transcription:
        print("     ERROR: f116v_transcription.json not found. Run f116v-transcribe first.")
        return

    combined = _load_json(rd, 'combined_refine.json')
    if not combined:
        print("     ERROR: combined_refine.json not found.")
        return
    assignment = combined.get('best_assignment', {})

    mod_data = _load_json(rd, 'modifier_integrate.json')
    modifier_chars = set(mod_data.get('modifier_chars', [])) if mod_data else set()

    signal_data = _load_json(rd, 'signal_isolation.json')
    signal_words = set()
    if signal_data:
        for ws in signal_data.get('word_signals', []):
            if ws.get('is_genuine_signal'):
                signal_words.add(ws['word'])

    eva_to_triple = build_eva_to_triple_lookup()

    # Build dictionaries
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    dict_10k = base_words
    dict_131k = base_words | expanded

    # Italian dictionary
    try:
        ref_it = load_reference_corpus(languages=['italian'], verbose=False)
        italian_words = set(w.lower() for w in ref_it.get_combined_tokens('italian') if len(w) >= 2)
    except Exception:
        italian_words = set()

    print(f"     T_P15: {len(assignment)} triples")
    print(f"     Modifiers: {len(modifier_chars)} chars")
    print(f"     Signal words: {signal_words}")
    print(f"     Dict 10K: {len(dict_10k)}, Dict 131K: {len(dict_131k)}, Italian: {len(italian_words)}")

    # ── 2. Decode each word ──
    print("\n  2. Decoding Voynichese words...")

    primary_decodes = []
    all_variant_decodes = []
    n_hits_10k = 0
    n_hits_131k = 0
    n_signal = 0

    for word_data in transcription.get('words', []):
        print(f"\n     Word {word_data['word_index']}:")

        for variant in word_data.get('variants', []):
            token = variant['eva_token']
            source = variant['source']

            # R3 strategy: try alteration, then strip, then raw
            decoded_alt = decode_token_modifier_aware(
                token, assignment, eva_to_triple, modifier_chars,
                modifier_rules=None,
            )
            decoded_strip = decode_token_modifier_aware(
                token, assignment, eva_to_triple, modifier_chars,
            )

            # Raw decode (no modifier processing)
            raw_chars = tokenize_eva_chars(token)
            raw_syllables = []
            for ch in raw_chars:
                tk = eva_to_triple.get(ch)
                if tk and tk in assignment:
                    raw_syllables.append(assignment[tk])
            decoded_raw = ''.join(raw_syllables)

            # Pick best via R3 strategy
            decoded = decoded_alt
            if decoded_alt.lower() in dict_131k:
                decoded = decoded_alt
            elif decoded_strip.lower() in dict_131k:
                decoded = decoded_strip
            else:
                decoded = decoded_raw

            decoded_lower = decoded.lower()

            in_10k = decoded_lower in dict_10k
            in_131k = decoded_lower in dict_131k
            in_it = decoded_lower in italian_words
            is_sig = decoded_lower in signal_words

            # Generate lattice alternatives
            lattice = _generate_decode_lattice(
                token, assignment, eva_to_triple, modifier_chars,
                dict_10k, dict_131k,
            )

            result = DecodeResult(
                source=source,
                eva_token=token,
                syllables=raw_syllables,
                decoded_word=decoded_lower,
                in_10k_dict=in_10k,
                in_131k_dict=in_131k,
                in_italian_dict=in_it,
                is_signal_word=is_sig,
                lattice_alternatives=lattice[:20],
            )

            all_variant_decodes.append(asdict(result))
            if source == 'ZL3b-n':
                primary_decodes.append(asdict(result))

            if in_10k:
                n_hits_10k += 1
            if in_131k:
                n_hits_131k += 1
            if is_sig:
                n_signal += 1

            print(f"       [{source}] '{token}' → syllables={raw_syllables} "
                  f"→ '{decoded_lower}'")
            print(f"         10K={in_10k} 131K={in_131k} Italian={in_it} "
                  f"Signal={is_sig}")
            if lattice:
                print(f"         Lattice ({len(lattice)} alternatives): "
                      f"{lattice[:5]}")

    # ── 3. Save ──
    output = F116vDecode(
        primary_decodes=primary_decodes,
        all_variant_decodes=all_variant_decodes,
        n_dict_hits_10k=n_hits_10k,
        n_dict_hits_131k=n_hits_131k,
        n_signal_matches=n_signal,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'f116v_decode.json', asdict(output))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


def _generate_decode_lattice(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: Set[str],
    dict_10k: Set[str],
    dict_131k: Set[str],
) -> List[str]:
    """Generate alternative decodings for Tier 2/3 triples."""
    from voynich.core.corpus import tokenize_eva_chars
    from voynich.core.reference import (
        PHONEME_NUCLEUS_MAP,
        PHONEME_PLACE_MAP,
    )

    chars = tokenize_eva_chars(token)
    # Get triples for non-modifier chars
    syllable_chars = [ch for ch in chars if ch not in modifier_chars]
    triples = []
    for ch in syllable_chars:
        tk = eva_to_triple.get(ch)
        if tk:
            triples.append(tk)

    if not triples:
        return []

    # For each triple, get the assigned syllable + alternatives
    options_per_position = []
    for tk in triples:
        assigned = assignment.get(tk, '')
        parts = tk.split(',')
        first_stroke = parts[0] if len(parts) > 0 else ''
        last_stroke = parts[1] if len(parts) > 1 else ''

        consonants = PHONEME_PLACE_MAP.get(first_stroke, [''])
        vowels = PHONEME_NUCLEUS_MAP.get(last_stroke, [''])

        alternatives = set()
        if assigned:
            alternatives.add(assigned)
        for c in consonants[:3]:
            for v in vowels[:3]:
                syl = c + v
                if len(syl) >= 2:
                    alternatives.add(syl)

        options_per_position.append(list(alternatives))

    # Generate combinations (limit to manageable number)
    lattice_words = set()
    _lattice_recurse(options_per_position, 0, [], lattice_words, dict_131k, limit=500)

    # Sort: dict hits first, then alphabetical
    hits = sorted([w for w in lattice_words if w in dict_131k])
    non_hits = sorted([w for w in lattice_words if w not in dict_131k])
    return hits + non_hits[:10]


def _lattice_recurse(
    options: List[List[str]],
    pos: int,
    current: List[str],
    results: Set[str],
    dictionary: Set[str],
    limit: int,
) -> None:
    """Recursively generate lattice combinations."""
    if len(results) >= limit:
        return
    if pos >= len(options):
        word = ''.join(current)
        if word:
            results.add(word)
        return
    for opt in options[pos]:
        _lattice_recurse(options, pos + 1, current + [opt], results, dictionary, limit)


# ---------------------------------------------------------------------------
# Step 48A.3 — Compile Competing Contextual Readings
# ---------------------------------------------------------------------------

def run_f116v_context() -> None:
    """Step 48A.3: Compile all competing readings of f116v surrounding text."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48A.3: f116v Contextual Readings")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Compile readings ──
    print(f"\n  1. Compiling {len(CONTEXTUAL_READINGS)} scholarly readings...")

    readings_out = []
    all_candidates = set()
    role_counts: Dict[str, int] = {}

    for reading in CONTEXTUAL_READINGS:
        rd_dict = asdict(reading)
        readings_out.append(rd_dict)
        all_candidates.update(reading.candidate_words)
        role_counts[reading.voynichese_role] = role_counts.get(reading.voynichese_role, 0) + 1

        print(f"\n     {reading.scholar} ({reading.year}): {reading.genre}")
        print(f"       Role: {reading.voynichese_role}")
        print(f"       Field: {reading.semantic_field}")
        print(f"       Candidates: {len(reading.candidate_words)} words")

    # ── 2. Consensus ──
    consensus_role = max(role_counts, key=role_counts.get) if role_counts else 'unknown'
    print(f"\n  2. Consensus role: {consensus_role} ({role_counts})")
    print(f"     Total unique candidates: {len(all_candidates)}")

    # ── 3. Save ──
    result = F116vContext(
        readings=readings_out,
        n_readings=len(readings_out),
        consensus_role=consensus_role,
        union_candidates=sorted(all_candidates),
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'f116v_context.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48A.4 — Contextual Matching
# ---------------------------------------------------------------------------

def run_f116v_match() -> None:
    """Step 48A.4: Test decoded words against contextual constraints."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48A.4: f116v Contextual Matching")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load dependencies ──
    print("\n  1. Loading dependencies...")

    decode_data = _load_json(rd, 'f116v_decode.json')
    context_data = _load_json(rd, 'f116v_context.json')

    if not decode_data:
        print("     ERROR: f116v_decode.json not found. Run f116v-decode first.")
        return
    if not context_data:
        print("     ERROR: f116v_context.json not found. Run f116v-context first.")
        return

    signal_data = _load_json(rd, 'signal_isolation.json')
    signal_words = set()
    if signal_data:
        for ws in signal_data.get('word_signals', []):
            if ws.get('is_genuine_signal'):
                signal_words.add(ws['word'])

    # ── 2. Match each decoded word against each reading ──
    print("\n  2. Testing decoded words against readings...")

    matches = []
    best_level = 'NO_MATCH'
    best_word = ''
    best_reading = ''

    level_rank = {'STRONG_MATCH': 4, 'PARTIAL_MATCH': 3, 'WEAK_MATCH': 2, 'NO_MATCH': 1}

    # Use primary decodes (ZL3b-n source)
    decoded_words = []
    for dec in decode_data.get('primary_decodes', []):
        decoded_words.append(dec['decoded_word'])
        # Also add lattice alternatives that are dict hits
        for alt in dec.get('lattice_alternatives', []):
            if alt not in decoded_words:
                decoded_words.append(alt)

    # Also collect decoded words from all variants
    for dec in decode_data.get('all_variant_decodes', []):
        w = dec['decoded_word']
        if w not in decoded_words:
            decoded_words.append(w)

    # Deduplicate
    decoded_words = list(dict.fromkeys(decoded_words))

    print(f"     Testing {len(decoded_words)} decoded words against "
          f"{len(context_data.get('readings', []))} readings...")

    for dw in decoded_words:
        for reading in context_data.get('readings', []):
            candidates = reading.get('candidate_words', [])
            semantic_field = set(reading.get('semantic_field', []))

            # Test 1: Direct dictionary match (already done in decode step)
            # Test 2: Semantic field match
            semantic_match = any(
                dw == c or _edit_distance(dw, c) <= 1
                for c in candidates
            )

            # Test 3: Edit distance to predicted words
            best_ed = float('inf')
            best_cand = ''
            for c in candidates:
                ed = _edit_distance(dw, c)
                if ed < best_ed:
                    best_ed = ed
                    best_cand = c

            # Test 4: Signal word cross-reference
            is_sig = dw in signal_words

            # Determine match level
            if best_ed == 0:
                match_level = 'STRONG_MATCH'
            elif best_ed <= 2 and semantic_match:
                match_level = 'PARTIAL_MATCH'
            elif best_ed <= 2:
                match_level = 'WEAK_MATCH'
            else:
                match_level = 'NO_MATCH'

            match = MatchResult(
                decoded_word=dw,
                reading_scholar=reading.get('scholar', ''),
                match_level=match_level,
                best_candidate=best_cand,
                edit_distance=best_ed if best_ed != float('inf') else -1,
                semantic_match=semantic_match,
                is_signal_word=is_sig,
            )
            matches.append(asdict(match))

            if level_rank.get(match_level, 0) > level_rank.get(best_level, 0):
                best_level = match_level
                best_word = dw
                best_reading = reading.get('scholar', '')

            if match_level in ('STRONG_MATCH', 'PARTIAL_MATCH'):
                print(f"     ★ {match_level}: '{dw}' ↔ '{best_cand}' "
                      f"(ED={best_ed}, {reading.get('scholar', '')})")

    # ── 3. Gate ──
    gate = 'CRIB_VIABLE' if best_level in ('STRONG_MATCH', 'PARTIAL_MATCH') else 'CRIB_FAILED'
    print(f"\n  3. Best match: {best_level} — '{best_word}' ({best_reading})")
    print(f"     Gate: {gate}")

    # ── 4. Save ──
    result = F116vMatch(
        matches=matches,
        best_match_level=best_level,
        best_match_word=best_word,
        best_match_reading=best_reading,
        gate_result=gate,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'f116v_match.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 48A.5 — Reverse Engineering
# ---------------------------------------------------------------------------

def run_f116v_reverse() -> None:
    """Step 48A.5: Reverse engineer — what assignment would produce plausible readings?"""
    t0 = time.time()

    print("=" * 70)
    print("STEP 48A.5: f116v Reverse Engineering")
    print("=" * 70)

    rd = _results_dir()

    from voynich.core.corpus import build_eva_to_triple_lookup, tokenize_eva_chars

    # ── 1. Load dependencies ──
    print("\n  1. Loading dependencies...")

    transcription = _load_json(rd, 'f116v_transcription.json')
    context_data = _load_json(rd, 'f116v_context.json')
    combined = _load_json(rd, 'combined_refine.json')

    if not transcription or not context_data or not combined:
        print("     ERROR: Missing dependency files.")
        return

    assignment = combined.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()

    # Load tier data from triple_tiers.json
    tier_data = _load_json(rd, 'triple_tiers.json')
    tier1_triples = set()
    if tier_data:
        for entry in tier_data.get('evidence_table', []):
            if entry.get('tier') == 'CONFIRMED':
                tier1_triples.add(entry.get('triple_key', ''))

    # Get all candidate words from readings
    all_candidates = context_data.get('union_candidates', [])
    print(f"     {len(all_candidates)} candidate words to test")
    print(f"     {len(assignment)} T_P15 assignments")
    print(f"     {len(tier1_triples)} Tier 1 (confirmed) triples")

    # ── 2. Get the EVA chars for each Voynichese word ──
    word_triples_list = []
    for word_data in transcription.get('words', []):
        # Use ZL3b-n variant
        for variant in word_data.get('variants', []):
            if variant.get('source') == 'ZL3b-n':
                chars = variant.get('eva_chars', [])
                triples = []
                for ch in chars:
                    tk = eva_to_triple.get(ch)
                    if tk:
                        triples.append(tk)
                word_triples_list.append({
                    'token': variant['eva_token'],
                    'chars': chars,
                    'triples': triples,
                })
                break

    print(f"\n  2. Voynichese words:")
    for wt in word_triples_list:
        print(f"     '{wt['token']}' → triples: {wt['triples']}")

    # ── 3. For each candidate, check required assignments ──
    print("\n  3. Reverse-engineering candidate alignments...")

    candidates_out = []

    for candidate in all_candidates:
        # Syllabify candidate (simple CV decomposition)
        syllables = _cv_syllabify(candidate)

        # Try to align syllables with each word's triples
        for wt in word_triples_list:
            triples = wt['triples']
            if len(syllables) != len(triples):
                continue  # Length mismatch

            required = {}
            n_agree = 0
            n_conflict = 0
            conflicts = []
            has_tier1_conflict = False

            for i, (tk, syl) in enumerate(zip(triples, syllables)):
                required[tk] = syl
                current = assignment.get(tk, '')
                if current == syl:
                    n_agree += 1
                else:
                    n_conflict += 1
                    is_tier1 = tk in tier1_triples
                    if is_tier1:
                        has_tier1_conflict = True
                    conflicts.append({
                        'triple_key': tk,
                        'current': current,
                        'required': syl,
                        'is_tier1': is_tier1,
                    })

            testable = n_conflict <= 2 and not has_tier1_conflict

            rc = ReverseCandidate(
                candidate_word=candidate,
                syllables=syllables,
                required_assignments=required,
                n_agree_t_p15=n_agree,
                n_conflict_t_p15=n_conflict,
                conflicting_triples=conflicts,
                conflicts_tier1=has_tier1_conflict,
                testable=testable,
            )
            candidates_out.append(asdict(rc))

            if testable or n_conflict <= 2:
                marker = '★' if testable else '○'
                print(f"     {marker} '{candidate}' → {syllables} | "
                      f"agree={n_agree} conflict={n_conflict} "
                      f"tier1_conflict={has_tier1_conflict} testable={testable}")

    # Also try with combined words (both Voynichese words → one candidate)
    combined_triples = []
    for wt in word_triples_list:
        combined_triples.extend(wt['triples'])

    for candidate in all_candidates:
        syllables = _cv_syllabify(candidate)
        if len(syllables) != len(combined_triples):
            continue

        required = {}
        n_agree = 0
        n_conflict = 0
        conflicts = []
        has_tier1_conflict = False

        for tk, syl in zip(combined_triples, syllables):
            required[tk] = syl
            current = assignment.get(tk, '')
            if current == syl:
                n_agree += 1
            else:
                n_conflict += 1
                is_tier1 = tk in tier1_triples
                if is_tier1:
                    has_tier1_conflict = True
                conflicts.append({
                    'triple_key': tk,
                    'current': current,
                    'required': syl,
                    'is_tier1': is_tier1,
                })

        testable = n_conflict <= 2 and not has_tier1_conflict
        rc = ReverseCandidate(
            candidate_word=f"{candidate} (combined)",
            syllables=syllables,
            required_assignments=required,
            n_agree_t_p15=n_agree,
            n_conflict_t_p15=n_conflict,
            conflicting_triples=conflicts,
            conflicts_tier1=has_tier1_conflict,
            testable=testable,
        )
        candidates_out.append(asdict(rc))

    # ── 4. Summary ──
    n_testable = sum(1 for c in candidates_out if c.get('testable'))
    best = min(candidates_out, key=lambda c: c.get('n_conflict_t_p15', 99)) if candidates_out else None

    print(f"\n  4. Summary:")
    print(f"     Total candidates tested: {len(candidates_out)}")
    print(f"     Testable (≤2 conflicts, no Tier 1): {n_testable}")
    if best:
        print(f"     Best: '{best['candidate_word']}' ({best['n_conflict_t_p15']} conflicts)")

    # ── 5. Save ──
    result = F116vReverse(
        candidates=candidates_out,
        n_testable=n_testable,
        best_candidate=best['candidate_word'] if best else '',
        best_n_conflicts=best['n_conflict_t_p15'] if best else -1,
        runtime_seconds=round(time.time() - t0, 2),
    )

    out = _save_json(rd, 'f116v_reverse.json', asdict(result))
    print(f"\n  Saved → {out}")
    print(f"  Completed in {time.time() - t0:.1f}s")


def _cv_syllabify(word: str) -> List[str]:
    """Simple CV syllabification of a Latin/Italian word."""
    vowels = set('aeiou')
    syllables = []
    current = ''

    for i, ch in enumerate(word):
        current += ch
        if ch in vowels:
            # Check if next char is a consonant followed by a vowel (CV boundary)
            if i + 1 < len(word) and word[i + 1] not in vowels:
                # Look ahead: is there a vowel after the consonant?
                if i + 2 < len(word) and word[i + 2] in vowels:
                    syllables.append(current)
                    current = ''
                elif i + 1 == len(word) - 1:
                    # Final consonant — attach to current syllable
                    pass
                else:
                    syllables.append(current)
                    current = ''
            elif i + 1 >= len(word):
                # End of word
                syllables.append(current)
                current = ''
            else:
                # Next char is also a vowel — break here
                syllables.append(current)
                current = ''

    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)

    return syllables


# ---------------------------------------------------------------------------
# Track A runner
# ---------------------------------------------------------------------------

def run_track_a_48() -> None:
    """Run all Track A steps sequentially."""
    print("\n" + "█" * 70)
    print("  PHASE 48 TRACK A: f116v Voynichese Word Decoding")
    print("█" * 70)

    run_f116v_transcribe()
    run_f116v_decode()
    run_f116v_context()
    run_f116v_match()
    run_f116v_reverse()

    print("\n" + "█" * 70)
    print("  TRACK A COMPLETE")
    print("█" * 70)
