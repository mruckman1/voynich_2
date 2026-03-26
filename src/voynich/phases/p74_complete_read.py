"""
Phase 74, Track B3: Assemble Complete Readings
================================================
Combine T1 identifications + dictionary matches + accepted LLM gap-fills
into complete annotated readings.

For passages where ALL gaps have ACCEPTED proposals, every token has a
proposed Latin word. This is the culmination: complete passage readings
of the Voynich manuscript.

Dependency chain:
    results/p74_llm_gapfill.json         (Track B2)
    results/p69_clean_corpus.json        (Phase 69 — T1 catalogue)
    results/combined_refine.json         (Phase 15 — assignment table)
        -> results/p74_complete_readings.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51
from voynich.phases.suffix_grammar import _classify_latin_ending


# ---------------------------------------------------------------------------
# JSON helpers
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
# CI chapter matching (from p70_annotated_read pattern)
# ---------------------------------------------------------------------------

CI_CHAPTERS = {
    'De Corallio': ['coral', 'cor', 'cora', 'cordi', 'rubeus', 'mare'],
    'De Senna': ['senna', 'sene', 'sena', 'purgare', 'laxare'],
    'De Rosa': ['rosa', 'ros', 'rosae', 'rosaceus'],
    'De Viola': ['viola', 'violas', 'violaceus'],
    'De Cassia': ['cassia', 'cassiae'],
    'De Cera': ['cera', 'cerae', 'cereus'],
    'De Sale': ['sal', 'salis', 'salinus'],
    'De Melle': ['mel', 'mellis', 'melleus'],
    'De Oleo': ['oleum', 'olei', 'oleosus'],
    'De Radice': ['radix', 'radi', 'rade', 'radicis'],
    'De Pipere': ['piper', 'piperis'],
    'De Balsamo': ['balsamum', 'balsami'],
    'De Aloe': ['aloe', 'aloes'],
    'De Camphora': ['camphora', 'camphorae'],
    'De Stercora': ['stercus', 'stercora', 'stercore'],
    'De Diasene': ['diasene', 'diasena'],
}


def _match_ci_chapter(words: List[str]) -> Optional[str]:
    """Match a list of decoded words against CI chapter ingredients."""
    best_chapter = None
    best_score = 0

    for chapter, keywords in CI_CHAPTERS.items():
        score = sum(1 for w in words if w.lower() in keywords)
        if score > best_score:
            best_score = score
            best_chapter = chapter

    return best_chapter if best_score >= 2 else None


# ---------------------------------------------------------------------------
# Assemble readings
# ---------------------------------------------------------------------------

def _assemble_readings(
    gapfill_data: Dict,
    all_tokens: List[str],
    decoded: List[str],
    t1_types: Dict[str, Dict],
    ref_word_set: Set[str],
    folio_list: List[str],
) -> List[Dict[str, Any]]:
    """Assemble complete readings from T1 + dict + accepted gap-fills."""
    accepted = gapfill_data.get('accepted_proposals', [])

    # Index accepted proposals by (passage_start, position)
    accepted_by_pos: Dict[Tuple[int, int], Dict] = {}
    for p in accepted:
        key = (p['passage_start'], p['position'])
        accepted_by_pos[key] = p

    # Reconstruct the real passages from gapfill results
    # We need the original passage info — re-select from corpus
    from voynich.phases.p74_llm_gapfill import _select_gap_fill_passages

    real_passages = _select_gap_fill_passages(
        all_tokens, decoded, t1_types, ref_word_set, folio_list, n=15)

    readings = []

    for passage in real_passages:
        start = passage['start']
        folio = passage['folio']
        gap_positions = set(g['position'] for g in passage['gaps'])

        tokens = []
        for pos in range(15):
            idx = start + pos

            # Check: identified from passage selection
            identified = next(
                (t for t in passage['identified'] if t['position'] == pos),
                None)

            # Check: accepted gap-fill
            filled = accepted_by_pos.get((start, pos))

            if identified:
                # Morphological analysis
                word = identified['word']
                pos_ending, ending_detail = _classify_latin_ending(word)

                tokens.append({
                    'position': pos,
                    'eva': all_tokens[idx] if idx < len(all_tokens) else '?',
                    'decoded': decoded[idx] if idx < len(decoded) else '?',
                    'word': word,
                    'gloss': identified['gloss'],
                    'source': identified['source'],
                    'confidence': 'VALIDATED',
                    'pos_tag': pos_ending or '',
                })
            elif filled and filled.get('status') == 'ACCEPTED':
                word = filled['proposed_word']
                pos_ending, _ = _classify_latin_ending(word)

                tokens.append({
                    'position': pos,
                    'eva': all_tokens[idx] if idx < len(all_tokens) else '?',
                    'decoded': decoded[idx] if idx < len(decoded) else '?',
                    'word': word,
                    'gloss': filled.get('proposed_gloss', '?'),
                    'source': 'LLM_GAP_FILL',
                    'confidence': 'ACCEPTED',
                    'reasoning': filled.get('reasoning', ''),
                    'pos_tag': pos_ending or '',
                })
            else:
                dec = decoded[idx] if idx < len(decoded) else '?'
                tokens.append({
                    'position': pos,
                    'eva': all_tokens[idx] if idx < len(all_tokens) else '?',
                    'decoded': dec,
                    'word': f"[{dec}]",
                    'gloss': '?',
                    'source': 'UNFILLED',
                    'confidence': 'UNKNOWN',
                    'pos_tag': '',
                })

        # Compute metrics
        n_validated = sum(1 for t in tokens if t['confidence'] == 'VALIDATED')
        n_accepted = sum(1 for t in tokens if t['confidence'] == 'ACCEPTED')
        n_unfilled = sum(1 for t in tokens if t['confidence'] == 'UNKNOWN')
        complete_fraction = (n_validated + n_accepted) / 15

        # Build reading string
        reading_parts = []
        for t in tokens:
            if t['gloss'] != '?':
                reading_parts.append(t['gloss'])
            else:
                reading_parts.append(f"[{t['word']}]")
        reading = ' · '.join(reading_parts)

        # CI chapter matching
        all_words = [t['word'] for t in tokens if t['confidence'] != 'UNKNOWN']
        ci_match = _match_ci_chapter(all_words)

        # Pharmaceutical interpretation
        verbs = [t for t in tokens if t.get('pos_tag') == 'VERB']
        nouns = [t for t in tokens if t.get('pos_tag') == 'NOUN']

        interpretation = None
        if verbs and nouns:
            verb_str = ', '.join(t['word'] for t in verbs[:3])
            noun_str = ', '.join(t['word'] for t in nouns[:3])
            interpretation = f"Verbs: {verb_str}; Nouns: {noun_str}"
            if ci_match:
                interpretation += f" — cf. {ci_match}"

        readings.append({
            'folio': folio,
            'start': start,
            'all_filled': n_unfilled == 0,
            'n_validated': n_validated,
            'n_accepted': n_accepted,
            'n_unfilled': n_unfilled,
            'complete_fraction': round(complete_fraction, 3),
            'tokens': tokens,
            'reading': reading,
            'ci_match': ci_match,
            'interpretation': interpretation,
        })

    readings.sort(key=lambda r: -r['complete_fraction'])
    return readings


# ---------------------------------------------------------------------------
# Null control: random passage readings for comparison
# ---------------------------------------------------------------------------

def _random_passage_readings(
    all_tokens: List[str],
    decoded: List[str],
    t1_types: Dict[str, Dict],
    ref_word_set: Set[str],
    folio_list: List[str],
    n: int = 20,
) -> Dict[str, float]:
    """Score random 15-token passages for comparison."""
    rng = np.random.default_rng(42)
    t1_set = set(t1_types.keys())
    window = 15

    fractions = []
    for _ in range(n):
        start = rng.integers(0, max(1, len(all_tokens) - window))
        w_tokens = all_tokens[start:start + window]
        w_decoded = decoded[start:start + window]

        n_id = sum(
            1 for tok, dec in zip(w_tokens, w_decoded)
            if tok in t1_set or (dec and dec.lower() in ref_word_set)
        )
        fractions.append(n_id / window)

    return {
        'random_mean_identified': round(float(np.mean(fractions)), 4),
        'random_std': round(float(np.std(fractions)), 4),
        'n_random': n,
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompleteReadingResult:
    phase: str = "74"
    step: str = "74.B3"
    experiment: str = "complete_readings"
    n_readings: int = 0
    n_fully_filled: int = 0
    n_near_complete: int = 0  # complete_fraction > 0.90
    best_complete_fraction: float = 0.0
    mean_complete_fraction: float = 0.0
    readings: List[Dict[str, Any]] = field(default_factory=list)
    # Null comparison
    random_mean_identified: float = 0.0
    selectivity: float = 0.0
    # CI matching
    n_ci_matches: int = 0
    n_interpretable: int = 0
    # Gates
    gate_b3_1: bool = False   # ≥1 fully filled
    gate_b3_2: bool = False   # ≥5 with fraction > 0.90
    gate_b3_3: bool = False   # Best reading pharmaceutically interpretable
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_complete_read():
    """Track B3: Assemble complete readings."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 74.B3 — Assemble Complete Readings")
    print("=" * 41)

    # --- Load gap-fill results ---
    print("  Loading gap-fill results...")
    gapfill_data = _safe_load(os.path.join(rd, 'p74_llm_gapfill.json'))
    n_accepted = gapfill_data.get('n_accepted', 0)
    print(f"    Accepted proposals: {n_accepted}")

    # --- Load data ---
    print("  Loading corpus data...")
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])
    t1_types: Dict[str, Dict] = {}
    for entry in t1_catalogue:
        eva_type = entry.get('eva_type', '')
        if eva_type:
            t1_types[eva_type] = entry

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})
    eva_to_triple = build_eva_to_triple_lookup()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    folio_list = []
    for folio_id, page in sorted(corpus.pages.items()):
        for _ in page.all_tokens:
            folio_list.append(folio_id)

    # --- Decode ---
    print("  Decoding corpus...")
    coda_table = build_coda_table_v2()
    coda_table.stroke_to_coda['connector'] = ''
    decoded = []
    for token in all_tokens:
        result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
        decoded.append(result.decoded_cvc)

    # --- Assemble readings ---
    print("  Assembling readings...")
    readings = _assemble_readings(
        gapfill_data, all_tokens, decoded, t1_types, ref_word_set, folio_list)

    n_fully = sum(1 for r in readings if r['all_filled'])
    n_near = sum(1 for r in readings if r['complete_fraction'] > 0.90)
    best_frac = max((r['complete_fraction'] for r in readings), default=0.0)
    mean_frac = float(np.mean([r['complete_fraction'] for r in readings])) \
        if readings else 0.0

    print(f"    Total readings: {len(readings)}")
    print(f"    Fully filled (100%): {n_fully}")
    print(f"    Near-complete (>90%): {n_near}")
    print(f"    Best fraction: {best_frac:.1%}")
    print(f"    Mean fraction: {mean_frac:.1%}")

    # --- Print top readings ---
    for i, r in enumerate(readings[:5]):
        fill_tag = " ** COMPLETE **" if r['all_filled'] else ""
        print(f"\n    [{i + 1}] {r['folio']} — {r['complete_fraction']:.0%} "
              f"({r['n_validated']}V + {r['n_accepted']}A + "
              f"{r['n_unfilled']}U){fill_tag}")
        print(f"        {r['reading']}")
        if r['ci_match']:
            print(f"        CI: {r['ci_match']}")
        if r['interpretation']:
            print(f"        Interp: {r['interpretation']}")

    # --- Null comparison ---
    print("\n  Running null comparison...")
    null_stats = _random_passage_readings(
        all_tokens, decoded, t1_types, ref_word_set, folio_list)
    selectivity = mean_frac / (null_stats['random_mean_identified'] + 0.001)
    print(f"    Random mean identified: {null_stats['random_mean_identified']:.1%}")
    print(f"    Selectivity: {selectivity:.2f}×")

    # --- CI and interpretation counts ---
    n_ci = sum(1 for r in readings if r['ci_match'])
    n_interp = sum(1 for r in readings if r['interpretation'])

    # --- Gates ---
    g1 = n_fully >= 1
    g2 = n_near >= 5
    best_reading = readings[0] if readings else None
    g3 = bool(best_reading and best_reading.get('interpretation')
              and best_reading.get('all_filled'))

    gates_passed = sum([g1, g2, g3])

    print(f"\n  Gates:")
    print(f"    B3_1 (≥1 fully filled): {'PASS' if g1 else 'FAIL'} ({n_fully})")
    print(f"    B3_2 (≥5 near-complete): {'PASS' if g2 else 'FAIL'} ({n_near})")
    print(f"    B3_3 (best interpretable): {'PASS' if g3 else 'FAIL'}")
    print(f"    Total: {gates_passed}/3")

    # --- Verdict ---
    if g1 and g3:
        verdict = 'COMPLETE_READING'
    elif g1:
        verdict = 'COMPLETE_BUT_UNCLEAR'
    elif g2:
        verdict = 'NEAR_COMPLETE'
    else:
        verdict = 'FRAGMENTARY'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    # Truncate readings for JSON size (keep top 10 with full tokens,
    # rest with summary only)
    truncated_readings = []
    for i, r in enumerate(readings):
        if i < 10:
            truncated_readings.append(r)
        else:
            truncated_readings.append({
                'folio': r['folio'],
                'start': r['start'],
                'all_filled': r['all_filled'],
                'complete_fraction': r['complete_fraction'],
                'reading': r['reading'],
                'ci_match': r['ci_match'],
            })

    result = CompleteReadingResult(
        n_readings=len(readings),
        n_fully_filled=n_fully,
        n_near_complete=n_near,
        best_complete_fraction=round(best_frac, 4),
        mean_complete_fraction=round(mean_frac, 4),
        readings=truncated_readings,
        random_mean_identified=null_stats['random_mean_identified'],
        selectivity=round(selectivity, 4),
        n_ci_matches=n_ci,
        n_interpretable=n_interp,
        gate_b3_1=g1,
        gate_b3_2=g2,
        gate_b3_3=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 1,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p74_complete_readings.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
