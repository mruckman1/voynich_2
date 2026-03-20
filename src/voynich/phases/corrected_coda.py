"""
Phase 60, Track A: Corrected Coda Mapping + Re-Decode + Comparison
==================================================================
Applies two corrections from Phase 59:
  1. connector -> 'r' (was 'l'): Phase 59 Inv 7 found 23.4% vs 0.5%
  2. EVA 'i' -> SYLLABIC in non-final position: Phase 59 Inv 3 found
     0 meaningful coda hits from 'i' (2,807 tokens)

Re-decodes the full corpus (36,238 tokens) and 5 null corpora with the
corrected mapping, compares 6 decode strategies on 10 metrics, and
provides a per-correction diagnostic impact report.

Dependency chain:
    results/coda_table.json           (Phase 57.1)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    results/null_corpus.json          (Phase 17)
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
        -> results/corrected_coda.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, tokenize_eva_chars
from voynich.phases.coda_markers import (
    CodaTable,
    CvcDecodeResult,
    SIMPLE_GALLOWS,
    build_coda_table,
    classify_token_chars,
    decode_corpus_cvc,
    decode_corpus_cv_strip,
    get_coda,
)
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _compute_bigram_z,
    _decode_corpus_cv_strip,
    _decode_corpus_r3,
    _load_shared_data,
    _run_signal_isolation,
    _validate_cvc_costamagna,
)
from voynich.phases.cvc_segmentation import (
    _load_segmentation_inventory,
    segment_corpus,
)


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
class CorrectionImpact:
    """Impact of a single correction on affected tokens."""
    correction: str
    n_affected: int
    old_dict_hit: float
    new_dict_hit: float
    delta: float
    sample_changes: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class StrategyMetrics:
    """Metrics for a single decode strategy."""
    name: str
    desc: str
    dict_hit: float
    n_signal_words: int
    mean_selectivity: float
    bigram_z: float
    net_signal: int
    mean_word_length: float
    attestation_rate: float
    cvc_fraction: float
    latin_ending_fraction: float
    content_word_fraction: float


@dataclass
class CorrectedCodaResult:
    """Full Track A output."""
    phase: str = "60"
    step: str = "60.1"
    experiment: str = "corrected_coda"
    # Strategy comparison
    strategies: List[StrategyMetrics] = field(default_factory=list)
    best_strategy: str = ""
    # Per-correction impact
    connector_impact: Optional[CorrectionImpact] = None
    i_impact: Optional[CorrectionImpact] = None
    # Gates
    g1_bigram_z: bool = False         # corrected bigram_z >= 96.19
    g2_net_signal: bool = False       # corrected net_signal >= 3855
    g3_attestation: bool = False      # corrected attestation >= 79.9%
    g4_connector: bool = False        # connector-affected tokens improved
    g5_i_neutral: bool = False        # i-affected no attestation regression
    g6_new_signal: bool = False       # >= 1 new signal word from corrections
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Corrected CVC functions
# ---------------------------------------------------------------------------

def build_coda_table_v2() -> CodaTable:
    """Build corrected coda table: connector->r, keep other mappings."""
    table = build_coda_table('primary')
    table.stroke_to_coda['connector'] = 'r'
    return table


def build_coda_table_v2_alt() -> CodaTable:
    """Build corrected alternate table: connector->r, vertical->m."""
    table = build_coda_table('alternate')
    table.stroke_to_coda['connector'] = 'r'
    return table


def classify_token_chars_v2(
    eva_chars: List[str],
    coda_table: CodaTable,
) -> List[Tuple[str, str]]:
    """Corrected character classification.

    Calls the original classify_token_chars, then reclassifies EVA 'i'
    as SYLLABIC in non-final positions.  Phase 59 Inv 3 found that 'i'
    produces 0 meaningful coda hits (5 total hits out of 2,807 tokens).
    """
    classified = classify_token_chars(eva_chars, coda_table)

    # Post-process: 'i' at non-final position -> SYLLABIC
    corrected = []
    for idx, (role, char) in enumerate(classified):
        if char == 'i' and role == 'CODA_MARKER' and idx < len(classified) - 1:
            corrected.append(('SYLLABIC', char))
        else:
            corrected.append((role, char))

    return corrected


def decode_token_cvc_v2(
    token: str,
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
) -> CvcDecodeResult:
    """Decode an EVA token using corrected CVC rules.

    Same algorithm as decode_token_cvc but uses classify_token_chars_v2.
    """
    eva_chars = tokenize_eva_chars(token)
    if not eva_chars:
        return CvcDecodeResult(
            token=token, eva_chars=[], char_roles=[],
            decoded_cv='', decoded_cvc='',
        )

    classified = classify_token_chars_v2(eva_chars, coda_table)
    roles = [role for role, _ in classified]

    # Build CVC output
    output_parts: List[Tuple[str, str]] = []
    for role, char in classified:
        if role == 'SYLLABIC':
            triple = eva_to_triple.get(char)
            syl = assignment.get(triple, '?') if triple else '?'
            output_parts.append(('CV', syl))
        elif role == 'CODA_MARKER':
            coda = get_coda(char, coda_table)
            if coda and output_parts and output_parts[-1][0] in ('CV', 'CVC'):
                prev_type, prev_val = output_parts[-1]
                output_parts[-1] = ('CVC', prev_val + coda)
            elif coda:
                output_parts.append(('ORPHAN', coda))

    decoded_cvc = ''.join(val for _, val in output_parts)

    # CV-only decode (strip modifiers)
    cv_parts = []
    for role, char in classified:
        if role == 'SYLLABIC':
            triple = eva_to_triple.get(char)
            syl = assignment.get(triple, '?') if triple else '?'
            cv_parts.append(syl)
    decoded_cv = ''.join(cv_parts)

    return CvcDecodeResult(
        token=token,
        eva_chars=eva_chars,
        char_roles=roles,
        decoded_cv=decoded_cv,
        decoded_cvc=decoded_cvc,
    )


def decode_corpus_cvc_v2(
    tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table: CodaTable,
) -> List[str]:
    """Decode a list of EVA tokens using corrected CVC rules."""
    return [
        decode_token_cvc_v2(tok, assignment, eva_to_triple, coda_table).decoded_cvc
        for tok in tokens
    ]


# ---------------------------------------------------------------------------
# Latin ending analysis
# ---------------------------------------------------------------------------

LATIN_ENDINGS = {
    '-en': 'acc/abl 3rd',
    '-in': 'prep/loc',
    '-an': 'acc 1st',
    '-on': 'acc 2nd',
    '-un': 'acc 2nd',
    '-er': 'agent/comp',
    '-ar': 'adj',
    '-or': 'agent/quality',
    '-es': 'nom pl',
    '-is': 'gen sg',
    '-us': 'nom sg 2nd',
    '-um': 'acc sg 2nd',
    '-am': 'acc sg 1st',
}


def _latin_ending_fraction(decoded_tokens: List[str]) -> float:
    """Fraction of tokens whose last 2-3 chars match a Latin ending."""
    n_matched = 0
    n_total = 0
    for w in decoded_tokens:
        if not w or w == '?' or len(w) < 3:
            continue
        n_total += 1
        suffix2 = '-' + w[-2:]
        suffix3 = '-' + w[-3:] if len(w) >= 4 else ''
        if suffix2 in LATIN_ENDINGS or suffix3 in LATIN_ENDINGS:
            n_matched += 1
    return n_matched / n_total if n_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Content word fraction
# ---------------------------------------------------------------------------

FUNCTION_WORDS = {
    'di', 'de', 'da', 'du', 'in', 'ad', 'et', 'se', 'si', 'cu', 'ce',
    'la', 'le', 'lo', 'li', 'ne', 'no', 'ni', 'con', 'per', 'non', 'bene',
    'co', 'te', 'ti',
}


def _content_word_fraction(signal_words: List[Dict[str, Any]]) -> float:
    """Fraction of signal words that are content (not function) words."""
    if not signal_words:
        return 0.0
    n_content = sum(1 for w in signal_words
                    if w['word'] not in FUNCTION_WORDS and len(w['word']) >= 3)
    return n_content / len(signal_words)


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def _run_strategy_v2(
    name: str,
    desc: str,
    all_tokens: List[str],
    null_token_lists: List[List[str]],
    ref_word_set: Set[str],
    folios: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    modifier_chars: set,
    modifier_rules: Dict[str, str],
    coda_primary: CodaTable,
    coda_alternate: CodaTable,
    coda_corrected: CodaTable,
    coda_corrected_alt: CodaTable,
    costamagna_inv: Set[str],
    syl_to_struct: Dict[str, str],
    n_perms: int = 500,
) -> StrategyMetrics:
    """Decode real + null with a strategy, compute all 10 metrics."""
    n_tokens = len(all_tokens)

    # Decode real corpus
    if name == 'cv_strip':
        real_decoded = _decode_corpus_cv_strip(
            all_tokens, assignment, eva_to_triple, modifier_chars)
    elif name == 'r3_combined':
        real_decoded = _decode_corpus_r3(
            all_tokens, assignment, eva_to_triple,
            modifier_chars, modifier_rules, ref_word_set)
    elif name == 'cvc_primary':
        real_decoded = decode_corpus_cvc(
            all_tokens, assignment, eva_to_triple, coda_primary)
    elif name == 'cvc_alternate':
        real_decoded = decode_corpus_cvc(
            all_tokens, assignment, eva_to_triple, coda_alternate)
    elif name == 'cvc_corrected':
        real_decoded = decode_corpus_cvc_v2(
            all_tokens, assignment, eva_to_triple, coda_corrected)
    elif name == 'cvc_corr_alt':
        real_decoded = decode_corpus_cvc_v2(
            all_tokens, assignment, eva_to_triple, coda_corrected_alt)
    else:
        raise ValueError(f"Unknown strategy: {name}")

    # Decode null corpora with same strategy
    null_decoded_list = []
    for null_tokens in null_token_lists:
        if name == 'cv_strip':
            nd = _decode_corpus_cv_strip(
                null_tokens, assignment, eva_to_triple, modifier_chars)
        elif name == 'r3_combined':
            nd = _decode_corpus_r3(
                null_tokens, assignment, eva_to_triple,
                modifier_chars, modifier_rules, ref_word_set)
        elif name == 'cvc_primary':
            nd = decode_corpus_cvc(
                null_tokens, assignment, eva_to_triple, coda_primary)
        elif name == 'cvc_alternate':
            nd = decode_corpus_cvc(
                null_tokens, assignment, eva_to_triple, coda_alternate)
        elif name == 'cvc_corrected':
            nd = decode_corpus_cvc_v2(
                null_tokens, assignment, eva_to_triple, coda_corrected)
        elif name == 'cvc_corr_alt':
            nd = decode_corpus_cvc_v2(
                null_tokens, assignment, eva_to_triple, coda_corrected_alt)
        null_decoded_list.append(nd)

    # 1. Dict hit
    real_hits = sum(1 for w in real_decoded if w in ref_word_set)
    dict_hit = real_hits / n_tokens if n_tokens > 0 else 0.0

    # 2-4. Signal isolation
    signal = _run_signal_isolation(
        real_decoded, null_decoded_list, ref_word_set, n_tokens)
    net_signal = signal.n_signal - signal.n_anti_signal

    # 5. Bigram z
    bigram_z = _compute_bigram_z(
        real_decoded, null_decoded_list, ref_word_set, folios, n_perms=n_perms)

    # 6. Mean word length
    lengths = [len(w) for w in real_decoded if w and w != '?']
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0

    # 7-8. Segmentation attestation + CVC fraction
    attestation_rate = 0.0
    cvc_frac = 0.0
    if name.startswith('cvc_'):
        per_token, all_segs = segment_corpus(real_decoded, costamagna_inv, syl_to_struct)
        n_att = sum(1 for s in all_segs if s['attested'])
        n_seg = len(all_segs)
        attestation_rate = n_att / n_seg if n_seg > 0 else 0.0
        cvc_structs = sum(1 for s in all_segs
                          if s['attested'] and syl_to_struct.get(s['text'], '') in
                          ('CVC', 'CCV', 'VCC', 'CVCC'))
        cvc_frac = cvc_structs / n_seg if n_seg > 0 else 0.0

    # 9. Latin ending fraction
    latin_end = _latin_ending_fraction(real_decoded)

    # 10. Content word fraction
    content_frac = _content_word_fraction(signal.top_signal_words)

    return StrategyMetrics(
        name=name,
        desc=desc,
        dict_hit=round(dict_hit, 4),
        n_signal_words=signal.n_signal_words,
        mean_selectivity=signal.mean_selectivity,
        bigram_z=round(bigram_z, 2),
        net_signal=net_signal,
        mean_word_length=round(mean_len, 2),
        attestation_rate=round(attestation_rate, 4),
        cvc_fraction=round(cvc_frac, 4),
        latin_ending_fraction=round(latin_end, 4),
        content_word_fraction=round(content_frac, 4),
    )


# ---------------------------------------------------------------------------
# Diagnostic impact report
# ---------------------------------------------------------------------------

def _compute_correction_impact(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_old: CodaTable,
    coda_new: CodaTable,
    ref_word_set: Set[str],
    correction_name: str,
    affected_test,
) -> CorrectionImpact:
    """Measure the impact of a specific correction on affected tokens."""
    affected_indices = []
    for idx, token in enumerate(all_tokens):
        if affected_test(token):
            affected_indices.append(idx)

    n_affected = len(affected_indices)
    if n_affected == 0:
        return CorrectionImpact(
            correction=correction_name, n_affected=0,
            old_dict_hit=0.0, new_dict_hit=0.0, delta=0.0,
        )

    old_hits = 0
    new_hits = 0
    samples = []
    for idx in affected_indices:
        token = all_tokens[idx]
        old_result = decode_corpus_cvc([token], assignment, eva_to_triple, coda_old)[0]
        new_result = decode_corpus_cvc_v2([token], assignment, eva_to_triple, coda_new)[0]

        old_hit = old_result in ref_word_set
        new_hit = new_result in ref_word_set
        if old_hit:
            old_hits += 1
        if new_hit:
            new_hits += 1

        if old_result != new_result and len(samples) < 20:
            samples.append({
                'token': token,
                'old_decode': old_result,
                'new_decode': new_result,
                'old_hit': old_hit,
                'new_hit': new_hit,
            })

    old_rate = old_hits / n_affected
    new_rate = new_hits / n_affected

    return CorrectionImpact(
        correction=correction_name,
        n_affected=n_affected,
        old_dict_hit=round(old_rate, 4),
        new_dict_hit=round(new_rate, 4),
        delta=round(new_rate - old_rate, 4),
        sample_changes=samples,
    )


def _has_connector_modifier(token: str) -> bool:
    """Check if a token contains a connector-group modifier character."""
    from voynich.core.reference import EVA_VISUAL_COMPONENTS
    eva_chars = tokenize_eva_chars(token)
    for ch in eva_chars[1:]:  # skip first (always SYLLABIC)
        comp = EVA_VISUAL_COMPONENTS.get(ch)
        if comp and comp.get('last_stroke') == 'connector':
            return True
    return False


def _has_medial_i(token: str) -> bool:
    """Check if a token contains EVA 'i' in a non-final position."""
    eva_chars = tokenize_eva_chars(token)
    for idx in range(1, len(eva_chars) - 1):
        if eva_chars[idx] == 'i':
            return True
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_corrected_coda():
    """Track A: Apply Phase 59 corrections and compare 6 strategies."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 60, Track A: Corrected Coda Mapping")
    print("=" * 70)

    # Load shared data
    print("\n  Loading shared data ...")
    data = _load_shared_data()
    rd = data['rd']

    all_tokens = data['all_tokens']
    assignment = data['assignment']
    eva_to_triple = data['eva_to_triple']
    ref_word_set = data['ref_word_set']
    folios = data['folios']
    null_token_lists = data['null_token_lists']
    modifier_chars = data['modifier_chars']
    modifier_rules = data['modifier_rules']
    coda_primary = data['coda_primary']
    coda_alternate = data['coda_alternate']

    # Build corrected tables
    coda_corrected = build_coda_table_v2()
    coda_corrected_alt = build_coda_table_v2_alt()

    print(f"\n  Corrected mapping: connector -> r (was l)")
    print(f"  Corrected: EVA 'i' -> SYLLABIC in non-final position")

    # Load Costamagna inventory for segmentation metrics
    costamagna_inv, syl_to_struct = _load_segmentation_inventory()
    print(f"  Costamagna inventory: {len(costamagna_inv)} syllables")

    # Run 6 strategies
    strategies_spec = [
        ('cv_strip', 'Phase 16: strip modifiers, CV decode'),
        ('r3_combined', 'Phase 16 R3: alteration -> strip -> raw'),
        ('cvc_primary', 'Phase 57 CVC: vertical->t, connector->l'),
        ('cvc_alternate', 'Phase 57 CVC: vertical->m, connector->l'),
        ('cvc_corrected', 'Phase 60 CVC: vertical->t, connector->r, i=syllabic'),
        ('cvc_corr_alt', 'Phase 60 CVC: vertical->m, connector->r, i=syllabic'),
    ]

    strategies: List[StrategyMetrics] = []
    for i, (name, desc) in enumerate(strategies_spec):
        print(f"\n  [{i+1}/6] Running {name} ...")
        result = _run_strategy_v2(
            name, desc,
            all_tokens, null_token_lists, ref_word_set, folios,
            assignment, eva_to_triple, modifier_chars, modifier_rules,
            coda_primary, coda_alternate, coda_corrected, coda_corrected_alt,
            costamagna_inv, syl_to_struct,
            n_perms=500,
        )
        strategies.append(result)
        print(f"    dict_hit={result.dict_hit:.4f}  signal={result.n_signal_words}  "
              f"bigram_z={result.bigram_z:.2f}  net_signal={result.net_signal}  "
              f"attest={result.attestation_rate:.4f}")

    # Find best strategy by composite ranking (bigram_z primary, net_signal secondary)
    best = max(strategies, key=lambda s: (s.bigram_z, s.net_signal))

    # Comparison table
    print("\n  " + "=" * 100)
    print(f"  {'Strategy':<18} {'DictHit':>8} {'Signal':>7} {'Sel':>6} "
          f"{'BigZ':>8} {'NetSig':>7} {'WdLen':>6} {'Attest':>8} "
          f"{'CVC%':>6} {'LatEnd':>7} {'Cont%':>6}")
    print("  " + "-" * 100)
    for s in strategies:
        marker = " <-- BEST" if s.name == best.name else ""
        print(f"  {s.name:<18} {s.dict_hit:>8.4f} {s.n_signal_words:>7} "
              f"{s.mean_selectivity:>6.2f} {s.bigram_z:>8.2f} "
              f"{s.net_signal:>7} {s.mean_word_length:>6.2f} "
              f"{s.attestation_rate:>8.4f} {s.cvc_fraction:>6.4f} "
              f"{s.latin_ending_fraction:>7.4f} "
              f"{s.content_word_fraction:>6.4f}{marker}")
    print("  " + "=" * 100)

    # Per-correction diagnostics
    print("\n  Computing per-correction impact ...")

    connector_impact = _compute_correction_impact(
        all_tokens, assignment, eva_to_triple,
        coda_primary, coda_corrected, ref_word_set,
        'connector_l_to_r', _has_connector_modifier,
    )
    print(f"\n  Connector (l->r): {connector_impact.n_affected} tokens affected")
    print(f"    Old dict_hit: {connector_impact.old_dict_hit:.4f}")
    print(f"    New dict_hit: {connector_impact.new_dict_hit:.4f}")
    print(f"    Delta: {connector_impact.delta:+.4f}")

    i_impact = _compute_correction_impact(
        all_tokens, assignment, eva_to_triple,
        coda_primary, coda_corrected, ref_word_set,
        'i_coda_to_syllabic', _has_medial_i,
    )
    print(f"\n  EVA 'i' (coda->syllabic): {i_impact.n_affected} tokens affected")
    print(f"    Old dict_hit: {i_impact.old_dict_hit:.4f}")
    print(f"    New dict_hit: {i_impact.new_dict_hit:.4f}")
    print(f"    Delta: {i_impact.delta:+.4f}")

    # Get reference metrics for gates (Phase 57 CVC primary)
    cvc57 = next(s for s in strategies if s.name == 'cvc_primary')
    corr = next(s for s in strategies if s.name == 'cvc_corrected')

    # Collect new vs old signal words
    old_decoded = decode_corpus_cvc(all_tokens, assignment, eva_to_triple, coda_primary)
    new_decoded = decode_corpus_cvc_v2(all_tokens, assignment, eva_to_triple, coda_corrected)
    old_words = set(w for w in old_decoded if w in ref_word_set)
    new_words = set(w for w in new_decoded if w in ref_word_set)
    new_signal_words = new_words - old_words

    # Validation gates
    g1 = corr.bigram_z >= 96.19
    g2 = corr.net_signal >= 3855
    g3 = corr.attestation_rate >= 0.799
    g4 = connector_impact.delta > 0
    g5 = i_impact.delta >= -0.01  # allow small regression
    g6 = len(new_signal_words) >= 1
    gates_passed = sum([g1, g2, g3, g4, g5, g6])

    print(f"\n  Validation Gates:")
    print(f"    G1 bigram_z >= 96.19:           {'PASS' if g1 else 'FAIL'} "
          f"({corr.bigram_z:.2f})")
    print(f"    G2 net_signal >= 3855:          {'PASS' if g2 else 'FAIL'} "
          f"({corr.net_signal})")
    print(f"    G3 attestation >= 79.9%:        {'PASS' if g3 else 'FAIL'} "
          f"({corr.attestation_rate:.1%})")
    print(f"    G4 connector improved:          {'PASS' if g4 else 'FAIL'} "
          f"(delta={connector_impact.delta:+.4f})")
    print(f"    G5 i-tokens neutral/positive:   {'PASS' if g5 else 'FAIL'} "
          f"(delta={i_impact.delta:+.4f})")
    print(f"    G6 new signal words >= 1:       {'PASS' if g6 else 'FAIL'} "
          f"({len(new_signal_words)} new)")
    print(f"    Gates passed: {gates_passed}/6")

    result = CorrectedCodaResult(
        strategies=[s for s in strategies],
        best_strategy=best.name,
        connector_impact=connector_impact,
        i_impact=i_impact,
        g1_bigram_z=g1,
        g2_net_signal=g2,
        g3_attestation=g3,
        g4_connector=g4,
        g5_i_neutral=g5,
        g6_new_signal=g6,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 4,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'corrected_coda.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Track A completed in {time.time() - t0:.1f}s")
    print(f"  Best strategy: {best.name}")
    print(f"  Verdict: {'PASS' if result.gate_passed else 'FAIL'} "
          f"({gates_passed}/6 gates)")
