"""
Phase 59, Investigation 1: Syllable Segmentation of CVC Output
===============================================================
Phase 57's Costamagna attestation was 4.3% (type-level) because the
evaluation checked multi-syllable strings like "corar" against single-syllable
CVC entries like "cor".  This module segments CVC-decoded output into
Costamagna syllables using greedy maximal-munch, then re-measures attestation.

Dependency chain:
    results/coda_table.json           (Phase 57.1)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    results/null_corpus.json          (Phase 17)
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
        -> results/cvc_segmentation.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import data_dir, results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.coda_markers import build_coda_table, decode_corpus_cvc
from voynich.phases.null_corpus import (
    _build_eva_bigram_model,
    _generate_null_corpus,
    _reconstruct_modifier_rules,
)


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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SegmentInfo:
    """A single segment from maximal-munch segmentation."""
    text: str
    length: int
    attested: bool
    start: int
    structure: str = ''   # CV, CVC, CCV, VC, V, etc.


@dataclass
class TokenSegmentation:
    """Segmentation result for one decoded token."""
    decoded: str
    segments: List[Dict[str, Any]]
    n_syllables: int
    n_attested: int
    attestation_rate: float


@dataclass
class NullComparison:
    """Comparison of real vs null segmentation rates."""
    real_rate: float
    null_mean: float
    null_std: float
    z_score: float
    selectivity: float
    null_rates: List[float]


@dataclass
class CvcSegmentationResult:
    """Full Investigation 1 output."""
    phase: str = "59"
    investigation: str = "1"
    experiment: str = "cvc_segmentation"
    # Aggregate stats
    total_tokens_segmented: int = 0
    total_syllable_tokens: int = 0
    attested_syllable_tokens: int = 0
    attestation_rate_token: float = 0.0
    unique_syllables: int = 0
    unique_attested: int = 0
    attestation_rate_type: float = 0.0
    mean_syllables_per_token: float = 0.0
    # Structure distribution
    structure_distribution: Dict[str, int] = field(default_factory=dict)
    # CVC/CCV fraction among attested
    cvc_ccv_fraction: float = 0.0
    # Null comparison
    null_comparison: Optional[NullComparison] = None
    # Gates
    g1_attestation: bool = False       # ≥ 40%
    g2_selectivity: bool = False       # ≥ 1.5×
    g3_mean_syl: bool = False          # 2.0–4.0
    g4_cvc_fraction: bool = False      # CVC+CCV ≥ 20%
    gates_passed: int = 0
    gate_passed: bool = False
    # Top examples
    top_attested_syllables: List[Dict[str, Any]] = field(default_factory=list)
    top_unmatched_syllables: List[Dict[str, Any]] = field(default_factory=list)
    sample_segmentations: List[Dict[str, Any]] = field(default_factory=list)
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Costamagna inventory loader
# ---------------------------------------------------------------------------

def _load_segmentation_inventory() -> Tuple[Set[str], Dict[str, str]]:
    """Load Costamagna syllabary and build inventory for segmentation.

    Returns (all_syllables_set, syllable_to_structure_map).
    """
    syl_path = os.path.join(
        str(data_dir('GL.S.III.MISC.12/extraction')),
        'syllabary_table.json',
    )
    if not os.path.exists(syl_path):
        return set(), {}

    with open(syl_path) as f:
        entries = json.load(f)

    inventory: Set[str] = set()
    syl_to_struct: Dict[str, str] = {}

    for entry in entries:
        syl = entry.get('syllable', '')
        struct = entry.get('structure', '')

        if struct in ('sigla',):
            continue

        if '-' in syl:
            # Shared sign — add each alternative
            for alt in syl.split('-'):
                alt = alt.strip().lower()
                if alt:
                    inventory.add(alt)
                    syl_to_struct[alt] = 'shared_sign'
        else:
            sl = syl.lower()
            if sl:
                inventory.add(sl)
                syl_to_struct[sl] = struct

    return inventory, syl_to_struct


# ---------------------------------------------------------------------------
# Segmentation functions
# ---------------------------------------------------------------------------

def segment_decoded_word(
    decoded_string: str,
    inventory: Set[str],
    max_syllable_len: int = 5,
) -> List[Dict[str, Any]]:
    """Greedy left-to-right maximal munch segmentation.

    Try longest match first, then shorter.  If no match at any length,
    consume 1 character as 'unmatched'.

    Example: "corar" → try "cora"(no) → "cor"(yes!) → "ar"(yes!) → ["cor","ar"]
    """
    syllables: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(decoded_string):
        matched = False
        for length in range(min(max_syllable_len, len(decoded_string) - pos), 0, -1):
            candidate = decoded_string[pos:pos + length]
            if candidate in inventory:
                syllables.append({
                    'text': candidate,
                    'length': length,
                    'attested': True,
                    'start': pos,
                })
                pos += length
                matched = True
                break
        if not matched:
            syllables.append({
                'text': decoded_string[pos],
                'length': 1,
                'attested': False,
                'start': pos,
            })
            pos += 1

    return syllables


def segment_corpus(
    cvc_decoded_tokens: List[str],
    inventory: Set[str],
    syl_to_struct: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Segment all CVC-decoded tokens.

    Returns (per_token_results, all_segments).
    """
    all_segments: List[Dict[str, Any]] = []
    per_token_results: List[Dict[str, Any]] = []

    for token_idx, decoded in enumerate(cvc_decoded_tokens):
        if not decoded or decoded == '?':
            continue
        segments = segment_decoded_word(decoded, inventory)
        n_attested = sum(1 for s in segments if s['attested'])
        n_total = len(segments)

        per_token_results.append({
            'token_idx': token_idx,
            'decoded': decoded,
            'segments': segments,
            'n_syllables': n_total,
            'n_attested': n_attested,
            'attestation_rate': n_attested / n_total if n_total > 0 else 0,
        })
        all_segments.extend(segments)

    return per_token_results, all_segments


def compute_structure_distribution(
    all_segments: List[Dict[str, Any]],
    syl_to_struct: Dict[str, str],
) -> Dict[str, int]:
    """Classify each attested segment by structure type."""
    counts: Dict[str, int] = {}
    for seg in all_segments:
        if seg['attested']:
            struct = syl_to_struct.get(seg['text'], 'unknown')
            counts[struct] = counts.get(struct, 0) + 1
        else:
            counts['unmatched'] = counts.get('unmatched', 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Null comparison
# ---------------------------------------------------------------------------

def null_segmentation_comparison(
    real_rate: float,
    null_token_lists: List[List[str]],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
    coda_table,
    inventory: Set[str],
    syl_to_struct: Dict[str, str],
) -> NullComparison:
    """Segment null corpora and compare attestation rates."""
    null_rates: List[float] = []

    for null_tokens in null_token_lists:
        null_decoded = decode_corpus_cvc(
            null_tokens, assignment, eva_to_triple, coda_table)
        _, null_segments = segment_corpus(null_decoded, inventory, syl_to_struct)
        total = len(null_segments)
        attested = sum(1 for s in null_segments if s['attested'])
        rate = attested / total if total > 0 else 0.0
        null_rates.append(rate)

    null_mean = float(np.mean(null_rates)) if null_rates else 0.0
    null_std = float(np.std(null_rates)) if null_rates else 0.0
    z_score = (real_rate - null_mean) / null_std if null_std > 0 else 0.0
    selectivity = real_rate / null_mean if null_mean > 0 else float('inf')

    return NullComparison(
        real_rate=round(real_rate, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        z_score=round(z_score, 2),
        selectivity=round(selectivity, 2),
        null_rates=[round(r, 4) for r in null_rates],
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_segmentation():
    """Investigation 1: Syllable segmentation of CVC decoded output."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 1: CVC Syllable Segmentation")
    print("=" * 70)

    rd = str(_results_dir())

    # Load Costamagna inventory
    print("\n  Loading Costamagna syllable inventory ...")
    inventory, syl_to_struct = _load_segmentation_inventory()
    print(f"  Inventory: {len(inventory)} syllables")

    # Load shared data
    print("  Loading corpus and assignment table ...")
    eva_to_triple = build_eva_to_triple_lookup()
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    from voynich.core.corpus import load_corpus
    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    coda_table = build_coda_table('primary')

    # Decode corpus with CVC
    print("  Decoding corpus (CVC primary) ...")
    cvc_decoded = decode_corpus_cvc(all_tokens, assignment, eva_to_triple, coda_table)

    # Segment
    print("  Segmenting decoded output ...")
    per_token_results, all_segments = segment_corpus(
        cvc_decoded, inventory, syl_to_struct)

    total_syllables = len(all_segments)
    attested_syllables = sum(1 for s in all_segments if s['attested'])
    attestation_rate_token = attested_syllables / total_syllables if total_syllables else 0.0

    unique_syls = set(s['text'] for s in all_segments)
    unique_attested = set(s['text'] for s in all_segments if s['attested'])
    attestation_rate_type = len(unique_attested) / len(unique_syls) if unique_syls else 0.0

    mean_syl_per_tok = float(np.mean(
        [r['n_syllables'] for r in per_token_results]
    )) if per_token_results else 0.0

    # Structure distribution
    struct_dist = compute_structure_distribution(all_segments, syl_to_struct)

    # CVC+CCV fraction among attested
    cvc_ccv_count = sum(v for k, v in struct_dist.items()
                        if k in ('CVC', 'CCV', 'CCVC', 'CCCVC', 'CVCC', 'VCC'))
    cvc_ccv_frac = cvc_ccv_count / attested_syllables if attested_syllables > 0 else 0.0

    print(f"\n  Total syllable tokens: {total_syllables}")
    print(f"  Attested:              {attested_syllables} "
          f"({attestation_rate_token:.1%})")
    print(f"  Unique syllables:      {len(unique_syls)}")
    print(f"  Unique attested:       {len(unique_attested)} "
          f"({attestation_rate_type:.1%})")
    print(f"  Mean syllables/token:  {mean_syl_per_tok:.2f}")
    print(f"  CVC+CCV fraction:      {cvc_ccv_frac:.1%}")

    # Structure distribution
    print(f"\n  Structure distribution:")
    for struct, count in sorted(struct_dist.items(), key=lambda x: -x[1]):
        frac = count / total_syllables if total_syllables else 0
        print(f"    {struct:<14} {count:>6} ({frac:.1%})")

    # Top attested syllables
    attested_counts = Counter(s['text'] for s in all_segments if s['attested'])
    top_attested = [{'syllable': syl, 'count': cnt, 'structure': syl_to_struct.get(syl, '?')}
                    for syl, cnt in attested_counts.most_common(20)]

    # Top unmatched
    unmatched_counts = Counter(s['text'] for s in all_segments if not s['attested'])
    top_unmatched = [{'text': txt, 'count': cnt}
                     for txt, cnt in unmatched_counts.most_common(20)]

    print(f"\n  Top attested syllables:")
    for ta in top_attested[:10]:
        print(f"    {ta['syllable']:8s} ({ta['structure']:4s}) count={ta['count']}")

    print(f"\n  Top unmatched segments:")
    for tu in top_unmatched[:10]:
        print(f"    {tu['text']:8s} count={tu['count']}")

    # Sample segmentations
    samples = []
    for ptr in per_token_results[:30]:
        seg_str = '+'.join(
            s['text'] if s['attested'] else f"[{s['text']}]"
            for s in ptr['segments']
        )
        samples.append({
            'decoded': ptr['decoded'],
            'segmented': seg_str,
            'n_syl': ptr['n_syllables'],
            'n_att': ptr['n_attested'],
            'rate': round(ptr['attestation_rate'], 2),
        })

    print(f"\n  Sample segmentations:")
    for s in samples[:10]:
        print(f"    {s['decoded']:16s} -> {s['segmented']:24s} "
              f"({s['n_att']}/{s['n_syl']} attested)")

    # Null comparison
    print("\n  Running null comparison ...")
    mod_data = _safe_load(os.path.join(rd, 'modifier_integrate.json'))
    null_data = _safe_load(os.path.join(rd, 'null_corpus.json'))
    null_seeds = ([r['seed'] for r in null_data.get('null_runs', [])]
                  if null_data else [100, 101, 102, 103, 104])

    bigram_probs, initial_probs, token_lengths = _build_eva_bigram_model(all_tokens)
    null_token_lists = []
    for seed in null_seeds:
        null_tokens = _generate_null_corpus(
            bigram_probs, initial_probs, token_lengths, len(all_tokens), seed)
        null_token_lists.append(null_tokens)

    null_comp = null_segmentation_comparison(
        attestation_rate_token, null_token_lists,
        assignment, eva_to_triple, coda_table,
        inventory, syl_to_struct)

    print(f"  Real attestation:  {null_comp.real_rate:.4f}")
    print(f"  Null mean:         {null_comp.null_mean:.4f}")
    print(f"  Null std:          {null_comp.null_std:.4f}")
    print(f"  Z-score:           {null_comp.z_score:.2f}")
    print(f"  Selectivity:       {null_comp.selectivity:.2f}x")

    # Evaluate gates
    g1 = attestation_rate_token >= 0.40
    g2 = null_comp.selectivity >= 1.5
    g3 = 2.0 <= mean_syl_per_tok <= 4.0
    g4 = cvc_ccv_frac >= 0.20
    gates_passed = sum([g1, g2, g3, g4])

    print(f"\n  Validation Gates:")
    print(f"    G1 attestation ≥ 40%:     {'PASS' if g1 else 'FAIL'} "
          f"({attestation_rate_token:.1%})")
    print(f"    G2 selectivity ≥ 1.5×:    {'PASS' if g2 else 'FAIL'} "
          f"({null_comp.selectivity:.2f}×)")
    print(f"    G3 mean syl/tok 2.0–4.0:  {'PASS' if g3 else 'FAIL'} "
          f"({mean_syl_per_tok:.2f})")
    print(f"    G4 CVC+CCV ≥ 20%:         {'PASS' if g4 else 'FAIL'} "
          f"({cvc_ccv_frac:.1%})")
    print(f"    Gates passed: {gates_passed}/4")

    result = CvcSegmentationResult(
        total_tokens_segmented=len(per_token_results),
        total_syllable_tokens=total_syllables,
        attested_syllable_tokens=attested_syllables,
        attestation_rate_token=round(attestation_rate_token, 4),
        unique_syllables=len(unique_syls),
        unique_attested=len(unique_attested),
        attestation_rate_type=round(attestation_rate_type, 4),
        mean_syllables_per_token=round(mean_syl_per_tok, 2),
        structure_distribution=struct_dist,
        cvc_ccv_fraction=round(cvc_ccv_frac, 4),
        null_comparison=null_comp,
        g1_attestation=g1,
        g2_selectivity=g2,
        g3_mean_syl=g3,
        g4_cvc_fraction=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        top_attested_syllables=top_attested,
        top_unmatched_syllables=top_unmatched,
        sample_segmentations=samples,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_segmentation.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 1 completed in {time.time() - t0:.1f}s")
