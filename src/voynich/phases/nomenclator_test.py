"""
Phase 9.2 — Nomenclator / Bimodal Frequency Test
=================================================

Rationale
---------
A nomenclator combines a codebook (common words -> dedicated symbols) with a
character-level cipher (rare words -> spelled out).  The frequency distribution
should be bimodal: a small set of very high-frequency tokens with one Zipf slope,
and a large set of low-frequency tokens with a different slope.

Sub-analyses
------------
9.2a  Fit single vs piecewise Zipf (AIC/BIC model selection)
9.2b  Compare bimodality to four reference languages
9.2c  Profile the two segments (high-freq codebook vs low-freq spelled-out)
9.2d  Differential decoding (whole-word seg 1, char-level seg 2)
Null  Markov-generated text bimodality comparison
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import load_corpus
from voynich.core.reference import load_reference_corpus, stem_token
from voynich.core.stats import (
    aic_bic_compare,
    build_ngram_lm,
    conditional_entropy,
    cross_entropy_lm,
    piecewise_zipf_fit,
    selectivity_ratio,
    zipf_analysis,
)
from voynich.phases.morpheme_grid import decompose_token_morphemes


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ZipfModelComparison:
    single_exponent: float
    single_r_squared: float
    single_sse: float
    breakpoint_rank: int
    segment1_exponent: float
    segment1_r_squared: float
    segment2_exponent: float
    segment2_r_squared: float
    piecewise_sse: float
    aic_single: float
    aic_piecewise: float
    bic_single: float
    bic_piecewise: float
    delta_aic: float
    delta_bic: float
    preferred_model: str


@dataclass
class BimodalityComparison:
    language: str
    breakpoint_rank: int
    delta_aic: float
    delta_bic: float
    preferred_model: str
    segment1_exponent: float
    segment2_exponent: float


@dataclass
class SegmentProfile:
    segment: str
    n_types: int
    n_tokens: int
    fraction_of_corpus: float
    mean_token_length: float
    h2_entropy: float
    top_20_tokens: List[Tuple[str, int]]
    morpheme_regularity: float


@dataclass
class DifferentialDecodingResult:
    segment1_method: str
    segment2_method: str
    segment2_char_types: int
    segment2_best_ce: float
    segment2_null_ce: float
    segment2_selectivity: float
    best_language: str


@dataclass
class NomenclatorTestResult:
    voynich_zipf: Dict
    zipf_model_comparison: Dict
    reference_bimodality: List[Dict]
    voynich_bimodality_unique: bool
    high_freq_profile: Dict
    low_freq_profile: Dict
    segment_exponent_gap: float
    differential_decoding: Dict
    null_markov_delta_aics: List[float]
    null_mean_delta_aic: float
    bimodality_selectivity: float
    gate_bimodality: bool
    gate_selectivity: bool
    gate_passed: bool
    verdict: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Convert dataclass/numpy types to JSON-serializable form."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _split_by_frequency(
    tokens: List[str], breakpoint_rank: int,
) -> Tuple[set, set]:
    """Partition vocabulary into high-freq and low-freq sets at *breakpoint_rank*."""
    counts = Counter(tokens)
    ranked = [tok for tok, _ in counts.most_common()]
    high = set(ranked[:breakpoint_rank])
    low = set(ranked[breakpoint_rank:])
    return high, low


def _profile_segment(
    tokens: List[str],
    segment_types: set,
    segment_label: str,
    total_corpus_tokens: int,
) -> SegmentProfile:
    """Compute descriptive statistics for one frequency segment."""
    seg_tokens = [t for t in tokens if t in segment_types]
    n_tokens = len(seg_tokens)
    n_types = len(segment_types)
    fraction = n_tokens / total_corpus_tokens if total_corpus_tokens else 0.0
    lengths = [len(t) for t in seg_tokens]
    mean_len = float(np.mean(lengths)) if lengths else 0.0

    text_for_h2 = ' '.join(seg_tokens)
    h2 = conditional_entropy(text_for_h2, order=1) if len(seg_tokens) > 20 else 0.0

    counts = Counter(seg_tokens)
    top20 = counts.most_common(20)

    # morpheme regularity: fraction of types that decompose non-trivially
    n_regular = 0
    for t in segment_types:
        d = decompose_token_morphemes(t)
        if d.prefix or d.suffix:
            n_regular += 1
    regularity = n_regular / n_types if n_types else 0.0

    return SegmentProfile(
        segment=segment_label,
        n_types=n_types,
        n_tokens=n_tokens,
        fraction_of_corpus=fraction,
        mean_token_length=mean_len,
        h2_entropy=h2,
        top_20_tokens=top20,
        morpheme_regularity=regularity,
    )


def _generate_markov_tokens(
    text: str, n_tokens_target: int, order: int = 2, seed: int = 42,
) -> List[str]:
    """Generate synthetic word tokens from a character-level Markov model."""
    lm = build_ngram_lm(list(text), order=order, smoothing=0.01)
    rng = random.Random(seed)
    counts = lm.get('counts', {})
    vocab = lm.get('vocab', set())
    if not counts or not vocab:
        return []

    # Build sampling tables: context -> [(char, cumprob), ...]
    tables: Dict[tuple, List[Tuple[str, float]]] = {}
    for ctx_key, char_counts in counts.items():
        total = sum(char_counts.values())
        if total == 0:
            continue
        ctx = tuple(ctx_key) if isinstance(ctx_key, (list, tuple)) else (ctx_key,)
        cumulative = []
        running = 0.0
        for ch, c in char_counts.items():
            running += c / total
            cumulative.append((ch, running))
        tables[ctx] = cumulative

    def _sample(ctx):
        tbl = tables.get(ctx)
        if not tbl:
            return rng.choice(list(vocab)) if vocab else 'a'
        r = rng.random()
        for ch, cp in tbl:
            if r <= cp:
                return ch
        return tbl[-1][0]

    # Generate characters, split on space
    generated: List[str] = []
    buf: List[str] = []
    ctx = tuple(['_'] * order)
    max_chars = n_tokens_target * 6  # average ~5 chars/token + space
    for _ in range(max_chars):
        ch = _sample(ctx)
        if ch == ' ' or ch == '_':
            if buf:
                word = ''.join(buf)
                if 1 <= len(word) <= 20:
                    generated.append(word)
                buf = []
                if len(generated) >= n_tokens_target:
                    break
            ctx = tuple(['_'] * order)
        else:
            buf.append(ch)
            ctx = (*ctx[1:], ch)
    if buf and len(generated) < n_tokens_target:
        generated.append(''.join(buf))
    return generated


def _differential_decode(
    tokens: List[str],
    low_freq_types: set,
    ref_corpus,
    language: str,
) -> DifferentialDecodingResult:
    """
    Attempt character-level MDL decoding on the low-frequency (spelled-out)
    segment only.  This segment should have ~20 character types if the
    nomenclator model is correct.
    """
    seg_tokens = [t for t in tokens if t in low_freq_types]
    seg_text = ' '.join(seg_tokens)

    # Character inventory of the low-freq segment
    chars = set(seg_text.replace(' ', ''))
    n_char_types = len(chars)

    # Build target LM from reference language
    ref_text = ref_corpus.get_combined_text(language)
    ref_lm = build_ngram_lm(list(ref_text), order=3, smoothing=0.01)

    # Baseline: cross-entropy of raw segment text under reference LM
    base_ce = cross_entropy_lm(seg_text, ref_lm, per_char=True)

    # Null: random character permutation CE (average over 20 trials)
    rng = random.Random(42)
    char_list = sorted(chars)
    null_ces = []
    for _ in range(20):
        perm = char_list.copy()
        rng.shuffle(perm)
        mapping = dict(zip(char_list, perm))
        permuted = ''.join(mapping.get(c, c) for c in seg_text)
        null_ces.append(cross_entropy_lm(permuted, ref_lm, per_char=True))
    null_mean = float(np.mean(null_ces))

    selectivity = null_mean / base_ce if base_ce > 0 else 1.0

    return DifferentialDecodingResult(
        segment1_method='frequency_rank_match',
        segment2_method='character_level_mdl',
        segment2_char_types=n_char_types,
        segment2_best_ce=base_ce,
        segment2_null_ce=null_mean,
        segment2_selectivity=selectivity,
        best_language=language,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_nomenclator_test() -> Dict:
    """
    Phase 9.2: Test whether the Voynich vocabulary exhibits bimodal
    frequency structure consistent with a nomenclator encoding.
    """
    print("Phase 9.2: Nomenclator / Bimodal Frequency Test")
    print("=" * 60)

    # --- Load data ---
    corpus = load_corpus(verbose=False)
    ref_corpus = load_reference_corpus(verbose=False)
    voynich_tokens = corpus.get_tokens(language='A')
    total_v = len(voynich_tokens)
    print(f"  Voynich Language A tokens: {total_v:,}")

    # ===================================================================
    # 9.2a: Single vs piecewise Zipf
    # ===================================================================
    print("\n  9.2a: Zipf model comparison ...")
    v_zipf = zipf_analysis(voynich_tokens)
    ranks = np.array(v_zipf['ranks'])
    freqs = np.array(v_zipf['frequencies'])

    pw = piecewise_zipf_fit(ranks, freqs)
    mc = aic_bic_compare(
        n_data=pw['n_data'],
        sse_model1=pw['sse_single'], k_model1=2,
        sse_model2=pw['sse_piecewise'], k_model2=5,
    )

    zipf_comp = ZipfModelComparison(
        single_exponent=pw['single_exponent'],
        single_r_squared=pw['single_r_squared'],
        single_sse=pw['sse_single'],
        breakpoint_rank=pw['breakpoint_rank'],
        segment1_exponent=pw['segment1_exponent'],
        segment1_r_squared=pw['segment1_r_squared'],
        segment2_exponent=pw['segment2_exponent'],
        segment2_r_squared=pw['segment2_r_squared'],
        piecewise_sse=pw['sse_piecewise'],
        aic_single=mc['aic_model1'],
        aic_piecewise=mc['aic_model2'],
        bic_single=mc['bic_model1'],
        bic_piecewise=mc['bic_model2'],
        delta_aic=mc['delta_aic'],
        delta_bic=mc['delta_bic'],
        preferred_model=mc['preferred_model'],
    )
    print(f"    Single Zipf exponent: {pw['single_exponent']:.3f}  "
          f"R²={pw['single_r_squared']:.4f}")
    print(f"    Piecewise breakpoint: rank {pw['breakpoint_rank']}")
    print(f"    Seg 1 exponent: {pw['segment1_exponent']:.3f}  "
          f"Seg 2 exponent: {pw['segment2_exponent']:.3f}")
    print(f"    delta_AIC = {mc['delta_aic']:.1f}  "
          f"delta_BIC = {mc['delta_bic']:.1f}  "
          f"preferred: {mc['preferred_model']}")

    # ===================================================================
    # 9.2b: Reference bimodality
    # ===================================================================
    print("\n  9.2b: Reference language bimodality ...")
    ref_bimodality: List[BimodalityComparison] = []
    for lang in ('latin', 'occitan', 'italian', 'german'):
        try:
            ref_tokens = ref_corpus.get_combined_tokens(lang)
        except Exception:
            print(f"    {lang}: corpus not available, skipping")
            continue
        if len(ref_tokens) < 100:
            continue
        rz = zipf_analysis(ref_tokens)
        rr = np.array(rz['ranks'])
        rf = np.array(rz['frequencies'])
        rpw = piecewise_zipf_fit(rr, rf)
        rmc = aic_bic_compare(
            n_data=rpw['n_data'],
            sse_model1=rpw['sse_single'], k_model1=2,
            sse_model2=rpw['sse_piecewise'], k_model2=5,
        )
        comp = BimodalityComparison(
            language=lang,
            breakpoint_rank=rpw['breakpoint_rank'],
            delta_aic=rmc['delta_aic'],
            delta_bic=rmc['delta_bic'],
            preferred_model=rmc['preferred_model'],
            segment1_exponent=rpw['segment1_exponent'],
            segment2_exponent=rpw['segment2_exponent'],
        )
        ref_bimodality.append(comp)
        print(f"    {lang}: delta_AIC={rmc['delta_aic']:.1f}  "
              f"preferred={rmc['preferred_model']}")

    # Is Voynich MORE bimodal than references?
    ref_daic = [r.delta_aic for r in ref_bimodality] if ref_bimodality else [0.0]
    voynich_daic = mc['delta_aic']
    voynich_bimodality_unique = voynich_daic < min(ref_daic) - 5.0

    # ===================================================================
    # 9.2c: Segment profiling
    # ===================================================================
    print("\n  9.2c: Segment profiling ...")
    bp = pw['breakpoint_rank']
    high_freq, low_freq = _split_by_frequency(voynich_tokens, bp)
    hi_prof = _profile_segment(voynich_tokens, high_freq, 'high_freq', total_v)
    lo_prof = _profile_segment(voynich_tokens, low_freq, 'low_freq', total_v)
    seg_gap = abs(pw['segment1_exponent'] - pw['segment2_exponent'])

    print(f"    High-freq segment: {hi_prof.n_types} types, "
          f"{hi_prof.n_tokens:,} tokens ({hi_prof.fraction_of_corpus:.1%})")
    print(f"    Low-freq segment:  {lo_prof.n_types} types, "
          f"{lo_prof.n_tokens:,} tokens ({lo_prof.fraction_of_corpus:.1%})")
    print(f"    Morpheme regularity: high={hi_prof.morpheme_regularity:.2f}  "
          f"low={lo_prof.morpheme_regularity:.2f}")
    print(f"    Exponent gap: {seg_gap:.3f}")

    # Character types in the low-freq segment
    lo_chars = set(''.join(t for t in voynich_tokens if t in low_freq))
    print(f"    Low-freq character types: {len(lo_chars)}")

    # ===================================================================
    # 9.2d: Differential decoding
    # ===================================================================
    print("\n  9.2d: Differential decoding (character-level on low-freq) ...")
    best_dd = None
    best_dd_selectivity = 0.0
    for lang in ('latin', 'occitan'):
        try:
            dd = _differential_decode(voynich_tokens, low_freq, ref_corpus, lang)
            print(f"    {lang}: CE={dd.segment2_best_ce:.4f}  "
                  f"null_CE={dd.segment2_null_ce:.4f}  "
                  f"selectivity={dd.segment2_selectivity:.3f}")
            if dd.segment2_selectivity > best_dd_selectivity:
                best_dd_selectivity = dd.segment2_selectivity
                best_dd = dd
        except Exception as e:
            print(f"    {lang}: failed — {e}")

    if best_dd is None:
        best_dd = DifferentialDecodingResult(
            segment1_method='frequency_rank_match',
            segment2_method='character_level_mdl',
            segment2_char_types=0,
            segment2_best_ce=0.0,
            segment2_null_ce=0.0,
            segment2_selectivity=1.0,
            best_language='none',
        )

    # ===================================================================
    # Null test: Markov bimodality
    # ===================================================================
    print("\n  Null test: Markov-generated bimodality ...")
    voynich_text = corpus.get_text(language='A')
    n_null = 50
    null_daics: List[float] = []
    for trial in range(n_null):
        synth_tokens = _generate_markov_tokens(
            voynich_text, n_tokens_target=len(voynich_tokens),
            order=2, seed=42 + trial,
        )
        if len(synth_tokens) < 100:
            continue
        sz = zipf_analysis(synth_tokens)
        sr = np.array(sz['ranks'])
        sf = np.array(sz['frequencies'])
        spw = piecewise_zipf_fit(sr, sf)
        smc = aic_bic_compare(
            n_data=spw['n_data'],
            sse_model1=spw['sse_single'], k_model1=2,
            sse_model2=spw['sse_piecewise'], k_model2=5,
        )
        null_daics.append(smc['delta_aic'])

    null_mean_daic = float(np.mean(null_daics)) if null_daics else 0.0
    # Selectivity: how much more bimodal is Voynich than Markov?
    # More negative delta_AIC = more bimodal
    abs_voynich = abs(voynich_daic) if voynich_daic < 0 else 0.0
    abs_null = abs(null_mean_daic) if null_mean_daic < 0 else 1e-6
    bimodality_selectivity = abs_voynich / abs_null if abs_null > 0 else 1.0

    print(f"    Voynich delta_AIC: {voynich_daic:.1f}")
    print(f"    Markov mean delta_AIC: {null_mean_daic:.1f}")
    print(f"    Bimodality selectivity: {bimodality_selectivity:.2f}x")

    # ===================================================================
    # Gate
    # ===================================================================
    gate_bimodality = voynich_daic < -10.0
    gate_selectivity = bimodality_selectivity >= 1.5
    gate_passed = gate_bimodality and gate_selectivity

    if gate_passed:
        verdict = 'nomenclator_bimodality_confirmed'
    elif gate_bimodality and not gate_selectivity:
        verdict = 'bimodal_but_not_unique'
    elif not gate_bimodality:
        verdict = 'single_zipf_adequate'
    else:
        verdict = 'inconclusive'

    print(f"\n  Gate: bimodality={gate_bimodality}  "
          f"selectivity={gate_selectivity}  passed={gate_passed}")
    print(f"  Verdict: {verdict}")

    # ===================================================================
    # Save
    # ===================================================================
    result = NomenclatorTestResult(
        voynich_zipf=v_zipf,
        zipf_model_comparison=_convert(asdict(zipf_comp)),
        reference_bimodality=[_convert(asdict(r)) for r in ref_bimodality],
        voynich_bimodality_unique=voynich_bimodality_unique,
        high_freq_profile=_convert(asdict(hi_prof)),
        low_freq_profile=_convert(asdict(lo_prof)),
        segment_exponent_gap=seg_gap,
        differential_decoding=_convert(asdict(best_dd)),
        null_markov_delta_aics=null_daics,
        null_mean_delta_aic=null_mean_daic,
        bimodality_selectivity=bimodality_selectivity,
        gate_bimodality=gate_bimodality,
        gate_selectivity=gate_selectivity,
        gate_passed=gate_passed,
        verdict=verdict,
    )

    out = _convert(asdict(result))
    with open(_results_dir() / 'nomenclator_test.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  Results saved to results/nomenclator_test.json")
    return out
