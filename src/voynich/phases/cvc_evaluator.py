"""
Phase 60, Track C: Unified CVC Evaluation Framework
====================================================
Replaces fragmented dict-hit / signal / bigram-z comparisons with a
single CVCEvaluator that scores decode strategies across 5 weighted
categories and produces a definitive comparison table.

Categories:
  1. Segmentation (0.25): Costamagna attestation, syllable structure
  2. Signal (0.25): signal word count, selectivity, net signal
  3. Sequential (0.20): bigram z-score
  4. Morphology (0.15): Latin ending fraction, ending diversity
  5. Pharma (0.15): pharmaceutical vocabulary overlap

Dependency chain:
    results/corrected_coda.json       (Track A)
    results/combined_refine.json      (Phase 15)
    results/modifier_integrate.json   (Phase 16)
    data/GL.S.III.MISC.12/extraction/syllabary_table.json
        -> results/cvc_evaluator.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.phases.cvc_coda_signal import (
    _build_folio_list,
    _compute_bigram_z,
    _load_shared_data,
    _run_signal_isolation,
)
from voynich.phases.cvc_segmentation import (
    _load_segmentation_inventory,
    segment_corpus,
)
from voynich.phases.corrected_coda import (
    FUNCTION_WORDS,
    LATIN_ENDINGS,
    _content_word_fraction,
    _latin_ending_fraction,
    build_coda_table_v2,
    build_coda_table_v2_alt,
    decode_corpus_cvc_v2,
)
from voynich.phases.coda_markers import decode_corpus_cvc
from voynich.phases.cvc_recipes import PHARMA_VOCAB


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
class CategoryScore:
    """Score for one evaluation category."""
    name: str
    weight: float
    sub_metrics: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class StrategyEvaluation:
    """Full evaluation of one decode strategy."""
    name: str
    desc: str
    categories: List[CategoryScore] = field(default_factory=list)
    composite: float = 0.0
    # Legacy metrics for backward compat
    dict_hit: float = 0.0
    n_signal_words: int = 0
    bigram_z: float = 0.0
    net_signal: int = 0


@dataclass
class CvcEvaluatorResult:
    """Full Track C output."""
    phase: str = "60"
    step: str = "60.3"
    experiment: str = "cvc_evaluator"
    evaluations: List[StrategyEvaluation] = field(default_factory=list)
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    best_strategy: str = ""
    # Gates
    g1_corrected_gt_r3: bool = False
    g2_corrected_gt_p57: bool = False
    g3_all_above_min: bool = False
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# CVCEvaluator
# ---------------------------------------------------------------------------

class CVCEvaluator:
    """Unified evaluation framework for CVC-decoded Voynich output."""

    CATEGORY_WEIGHTS = {
        'segmentation': 0.25,
        'signal': 0.25,
        'sequential': 0.20,
        'morphology': 0.15,
        'pharma': 0.15,
    }

    def __init__(
        self,
        ref_word_set: Set[str],
        costamagna_inv: Set[str],
        syl_to_struct: Dict[str, str],
        pharma_vocab: Set[str],
    ):
        self.ref_word_set = ref_word_set
        self.costamagna_inv = costamagna_inv
        self.syl_to_struct = syl_to_struct
        self.pharma_vocab = pharma_vocab

    def evaluate(
        self,
        name: str,
        desc: str,
        real_decoded: List[str],
        null_decoded_list: List[List[str]],
        folios: List[str],
        is_cvc: bool = True,
    ) -> StrategyEvaluation:
        """Evaluate a decode strategy across all 5 categories."""
        n_tokens = len(real_decoded)

        # Signal isolation
        signal = _run_signal_isolation(
            real_decoded, null_decoded_list, self.ref_word_set, n_tokens)
        net_signal = signal.n_signal - signal.n_anti_signal

        # Bigram z
        bigram_z = _compute_bigram_z(
            real_decoded, null_decoded_list, self.ref_word_set, folios,
            n_perms=500)

        # Dict hit (legacy)
        real_hits = sum(1 for w in real_decoded if w in self.ref_word_set)
        dict_hit = real_hits / n_tokens if n_tokens > 0 else 0.0

        # --- Category 1: Segmentation ---
        seg_metrics = {}
        if is_cvc:
            per_token, all_segs = segment_corpus(
                real_decoded, self.costamagna_inv, self.syl_to_struct)
            n_att = sum(1 for s in all_segs if s['attested'])
            n_seg = len(all_segs)
            seg_metrics['attestation_rate'] = n_att / n_seg if n_seg > 0 else 0.0
            syl_counts = [len(r['segments']) for r in per_token]
            seg_metrics['mean_syllables'] = (
                sum(syl_counts) / len(syl_counts) if syl_counts else 0.0)
            cvc_structs = sum(1 for s in all_segs
                              if s['attested'] and self.syl_to_struct.get(
                                  s['text'], '') in ('CVC', 'CCV', 'VCC', 'CVCC'))
            seg_metrics['cvc_fraction'] = cvc_structs / n_seg if n_seg > 0 else 0.0
        else:
            seg_metrics['attestation_rate'] = 0.0
            seg_metrics['mean_syllables'] = 0.0
            seg_metrics['cvc_fraction'] = 0.0

        # --- Category 2: Signal ---
        sig_metrics = {
            'n_signal_words': float(signal.n_signal_words),
            'mean_selectivity': signal.mean_selectivity,
            'net_signal': float(net_signal),
            'content_fraction': _content_word_fraction(signal.top_signal_words),
        }

        # --- Category 3: Sequential ---
        seq_metrics = {
            'bigram_z': bigram_z,
        }

        # --- Category 4: Morphology ---
        morph_metrics = {
            'latin_ending_fraction': _latin_ending_fraction(real_decoded),
            'ending_diversity': self._ending_diversity(real_decoded),
        }

        # --- Category 5: Pharma ---
        pharma_metrics = {
            'pharma_overlap': self._pharma_overlap(real_decoded),
            'n_pharma_terms': float(self._count_pharma(real_decoded)),
        }

        categories = [
            CategoryScore(
                name='segmentation', weight=0.25,
                sub_metrics=seg_metrics,
            ),
            CategoryScore(
                name='signal', weight=0.25,
                sub_metrics=sig_metrics,
            ),
            CategoryScore(
                name='sequential', weight=0.20,
                sub_metrics=seq_metrics,
            ),
            CategoryScore(
                name='morphology', weight=0.15,
                sub_metrics=morph_metrics,
            ),
            CategoryScore(
                name='pharma', weight=0.15,
                sub_metrics=pharma_metrics,
            ),
        ]

        return StrategyEvaluation(
            name=name, desc=desc,
            categories=categories,
            dict_hit=round(dict_hit, 4),
            n_signal_words=signal.n_signal_words,
            bigram_z=round(bigram_z, 2),
            net_signal=net_signal,
        )

    def normalize_and_score(
        self,
        evaluations: List[StrategyEvaluation],
    ) -> List[StrategyEvaluation]:
        """Normalize sub-metrics across strategies and compute composites."""
        # Collect all metric values per (category, metric)
        metric_ranges: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for ev in evaluations:
            for cat in ev.categories:
                for metric_name, value in cat.sub_metrics.items():
                    key = (cat.name, metric_name)
                    lo, hi = metric_ranges.get(key, (value, value))
                    metric_ranges[key] = (min(lo, value), max(hi, value))

        # Normalize and compute category scores
        for ev in evaluations:
            composite = 0.0
            for cat in ev.categories:
                normalized_values = []
                for metric_name, value in cat.sub_metrics.items():
                    key = (cat.name, metric_name)
                    lo, hi = metric_ranges[key]
                    if hi - lo < 1e-9:
                        norm = 0.5  # all strategies equal
                    else:
                        norm = (value - lo) / (hi - lo)
                    normalized_values.append(norm)

                cat.score = round(
                    sum(normalized_values) / len(normalized_values)
                    if normalized_values else 0.0, 4)
                composite += cat.weight * cat.score

            ev.composite = round(composite, 4)

        return evaluations

    def _ending_diversity(self, decoded_tokens: List[str]) -> float:
        """Count unique Latin ending types found."""
        endings_found = set()
        for w in decoded_tokens:
            if not w or w == '?' or len(w) < 3:
                continue
            for ending in LATIN_ENDINGS:
                suffix = ending[1:]  # strip the '-'
                if w.endswith(suffix):
                    endings_found.add(ending)
        return float(len(endings_found))

    def _pharma_overlap(self, decoded_tokens: List[str]) -> float:
        """Fraction of decoded types that appear in pharmaceutical vocabulary."""
        decoded_types = set(w for w in decoded_tokens if w and w != '?')
        if not decoded_types:
            return 0.0
        overlap = decoded_types & self.pharma_vocab
        return len(overlap) / len(decoded_types)

    def _count_pharma(self, decoded_tokens: List[str]) -> int:
        """Count pharmaceutical term hits (token level)."""
        return sum(1 for w in decoded_tokens if w in self.pharma_vocab)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_eval():
    """Track C: Evaluate all strategies through unified CVC framework."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 60, Track C: Unified CVC Evaluation Framework")
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

    # Load Costamagna inventory
    costamagna_inv, syl_to_struct = _load_segmentation_inventory()
    pharma_vocab = set(PHARMA_VOCAB.keys()) if isinstance(PHARMA_VOCAB, dict) else set(PHARMA_VOCAB)

    # Build evaluator
    evaluator = CVCEvaluator(
        ref_word_set=ref_word_set,
        costamagna_inv=costamagna_inv,
        syl_to_struct=syl_to_struct,
        pharma_vocab=pharma_vocab,
    )

    # Define strategies and decode each
    from voynich.phases.cvc_coda_signal import (
        _decode_corpus_cv_strip,
        _decode_corpus_r3,
    )

    strategies_spec = [
        ('cv_strip', 'Phase 16: strip modifiers, CV decode', False),
        ('r3_combined', 'Phase 16 R3: alteration -> strip -> raw', False),
        ('cvc_primary', 'Phase 57 CVC: vertical->t, connector->l', True),
        ('cvc_corrected', 'Phase 60 CVC: connector->r, i=syllabic', True),
    ]

    evaluations: List[StrategyEvaluation] = []

    for i, (name, desc, is_cvc) in enumerate(strategies_spec):
        print(f"\n  [{i+1}/{len(strategies_spec)}] Evaluating {name} ...")

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
        elif name == 'cvc_corrected':
            real_decoded = decode_corpus_cvc_v2(
                all_tokens, assignment, eva_to_triple, coda_corrected)
        else:
            raise ValueError(f"Unknown strategy: {name}")

        # Decode null corpora
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
            elif name == 'cvc_corrected':
                nd = decode_corpus_cvc_v2(
                    null_tokens, assignment, eva_to_triple, coda_corrected)
            null_decoded_list.append(nd)

        ev = evaluator.evaluate(
            name, desc, real_decoded, null_decoded_list, folios, is_cvc=is_cvc)
        evaluations.append(ev)
        print(f"    dict_hit={ev.dict_hit:.4f}  signal={ev.n_signal_words}  "
              f"bigram_z={ev.bigram_z:.2f}  net_signal={ev.net_signal}")

    # Normalize and compute composites
    evaluations = evaluator.normalize_and_score(evaluations)

    # Sort by composite (descending)
    evaluations.sort(key=lambda e: -e.composite)

    # Ranking table
    ranking = []
    print("\n  " + "=" * 95)
    print(f"  {'Strategy':<18} {'DictHit':>8} {'Signal':>7} {'BigZ':>8} "
          f"{'NetSig':>7} {'Seg':>6} {'Sig':>6} {'Seq':>6} "
          f"{'Morph':>6} {'Pharma':>6} {'Comp':>7}")
    print("  " + "-" * 95)
    for ev in evaluations:
        cat_scores = {c.name: c.score for c in ev.categories}
        marker = " <-- BEST" if ev == evaluations[0] else ""
        print(f"  {ev.name:<18} {ev.dict_hit:>8.4f} {ev.n_signal_words:>7} "
              f"{ev.bigram_z:>8.2f} {ev.net_signal:>7} "
              f"{cat_scores.get('segmentation', 0):>6.3f} "
              f"{cat_scores.get('signal', 0):>6.3f} "
              f"{cat_scores.get('sequential', 0):>6.3f} "
              f"{cat_scores.get('morphology', 0):>6.3f} "
              f"{cat_scores.get('pharma', 0):>6.3f} "
              f"{ev.composite:>7.4f}{marker}")
        ranking.append({
            'rank': len(ranking) + 1,
            'strategy': ev.name,
            'composite': ev.composite,
            'dict_hit': ev.dict_hit,
            'n_signal': ev.n_signal_words,
            'bigram_z': ev.bigram_z,
            'net_signal': ev.net_signal,
        })
    print("  " + "=" * 95)

    # Gates
    ev_corr = next((e for e in evaluations if e.name == 'cvc_corrected'), None)
    ev_r3 = next((e for e in evaluations if e.name == 'r3_combined'), None)
    ev_p57 = next((e for e in evaluations if e.name == 'cvc_primary'), None)

    g1 = ev_corr.composite > ev_r3.composite if ev_corr and ev_r3 else False
    g2 = ev_corr.composite > ev_p57.composite if ev_corr and ev_p57 else False
    g3 = all(
        c.score >= 0.3
        for c in (ev_corr.categories if ev_corr else [])
    ) if ev_corr else False
    gates_passed = sum([g1, g2, g3])

    print(f"\n  Validation Gates:")
    print(f"    G1 corrected > R3:           {'PASS' if g1 else 'FAIL'} "
          f"({ev_corr.composite:.4f} vs {ev_r3.composite:.4f})"
          if ev_corr and ev_r3 else "    G1: N/A")
    print(f"    G2 corrected > Phase 57:     {'PASS' if g2 else 'FAIL'} "
          f"({ev_corr.composite:.4f} vs {ev_p57.composite:.4f})"
          if ev_corr and ev_p57 else "    G2: N/A")
    print(f"    G3 all components >= 0.3:    {'PASS' if g3 else 'FAIL'}")
    print(f"    Gates passed: {gates_passed}/3")

    result = CvcEvaluatorResult(
        evaluations=evaluations,
        ranking=ranking,
        best_strategy=evaluations[0].name if evaluations else '',
        g1_corrected_gt_r3=g1,
        g2_corrected_gt_p57=g2,
        g3_all_above_min=g3,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_evaluator.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Track C completed in {time.time() - t0:.1f}s")
    print(f"  Best strategy: {evaluations[0].name} "
          f"(composite={evaluations[0].composite:.4f})")
