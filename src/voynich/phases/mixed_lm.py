"""
Step 34.8 – Latin-Italian Interpolated Language Model (Track C)
================================================================
Builds a Latin character 5-gram LM and a synthetic Northern-Italian LM
(derived from Latin via sound changes), then scores the Phase 16 decoded
corpus under several interpolation weights to test whether a mixed
Latin-Italian model fits better than pure Latin.

Algorithm:
  1. Build Latin char-5gram LM from reference corpus.
  2. Generate synthetic Northern Italian via sound-change rules
     (intervocalic voicing, geminate lenition, final consonant loss,
      palatalization).
  3. Build Italian LM on the synthetic corpus.
  4. Interpolated LM: alpha * Latin + (1-alpha) * Italian for
     alpha in {0.3, 0.5, 0.7, 0.9}.
  5. Score decoded Voynich corpus under each LM variant.
  6. Code-switching test: SIGNAL vs non-SIGNAL tokens under each LM.

Dependency chain:
    combined_refine.json     (Phase 15 assignment)
    modifier_integrate.json  (Phase 16 modifiers)
    signal_bigrams.json      (Phase 29 token classifications)
    latin_lm.json            (Step 33.5: calibration baseline)
        -> mixed_lm.json     (this step)
"""

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus, EVA_VISUAL_COMPONENTS
from voynich.phases.signal_isolation import _decode_corpus_r3
from voynich.phases.null_corpus import _reconstruct_modifier_rules
from voynich.phases.latin_lm import CharNgramLM


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


def _preprocess_text(raw: str) -> str:
    """Lowercase, keep only a-z and space, collapse whitespace."""
    out = []
    for ch in raw.lower():
        if 'a' <= ch <= 'z' or ch == ' ':
            out.append(ch)
    text = ''.join(out)
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


# ---------------------------------------------------------------------------
# Sound-change rules: Latin -> synthetic Northern Italian
# ---------------------------------------------------------------------------

_VOWELS = set('aeiou')


def _apply_sound_changes(text: str) -> str:
    """Apply Northern Italian sound changes to a Latin text string.

    Transformations (applied in order):
      1. Intervocalic voicing: p->b, t->d, c->g between vowels
      2. Geminate lenition: ll->l, ss->s, tt->t, pp->p, cc->c, mm->m, nn->n
      3. Final consonant loss: drop -m, -t, -s at end of words when preceded
         by an unstressed (non-initial) vowel
      4. Palatalization: ct->tt, cl->chi, pl->pi
    """
    # Work on lowercase
    text = text.lower()

    # --- 4. Palatalization (before voicing, to avoid ct->cd) ---
    text = text.replace('ct', 'tt')
    text = text.replace('cl', 'chi')
    text = text.replace('pl', 'pi')

    # --- 1. Intervocalic voicing ---
    result = list(text)
    voicing_map = {'p': 'b', 't': 'd', 'c': 'g'}
    for i in range(1, len(result) - 1):
        ch = result[i]
        if ch in voicing_map:
            prev = result[i - 1]
            nxt = result[i + 1]
            if prev in _VOWELS and nxt in _VOWELS:
                result[i] = voicing_map[ch]
    text = ''.join(result)

    # --- 2. Geminate lenition ---
    for gem in ['ll', 'ss', 'tt', 'pp', 'cc', 'mm', 'nn', 'rr', 'ff']:
        text = text.replace(gem, gem[0])

    # --- 3. Final consonant loss (-m, -t, -s) in words ---
    words = text.split()
    processed = []
    for word in words:
        if len(word) >= 3 and word[-1] in ('m', 't', 's'):
            # Only drop if preceded by a vowel (unstressed position)
            if word[-2] in _VOWELS:
                word = word[:-1]
        processed.append(word)
    text = ' '.join(processed)

    return text


# ---------------------------------------------------------------------------
# Interpolated LM
# ---------------------------------------------------------------------------

class InterpolatedLM:
    """Linearly interpolated character-level LM: alpha * lm_a + (1-alpha) * lm_b."""

    def __init__(self, lm_a: CharNgramLM, lm_b: CharNgramLM, alpha: float):
        self.lm_a = lm_a
        self.lm_b = lm_b
        self.alpha = alpha

    def bits_per_char(self, text: str) -> float:
        """Compute cross-entropy via log of interpolated probabilities."""
        order = self.lm_a.order
        padded = '^' * (order - 1) + text + '$'
        vocab_a = len(self.lm_a.vocab) + 2
        vocab_b = len(self.lm_b.vocab) + 2
        total_log = 0.0
        n = 0
        for i in range(len(padded) - order + 1):
            context = padded[i:i + order - 1]
            next_char = padded[i + order - 1]
            # LM A probability
            count_a = self.lm_a.counts[context][next_char]
            total_a = self.lm_a.context_totals[context]
            prob_a = (count_a + self.lm_a.alpha) / (total_a + self.lm_a.alpha * vocab_a)
            # LM B probability
            count_b = self.lm_b.counts[context][next_char]
            total_b = self.lm_b.context_totals[context]
            prob_b = (count_b + self.lm_b.alpha) / (total_b + self.lm_b.alpha * vocab_b)
            # Interpolate
            prob = self.alpha * prob_a + (1.0 - self.alpha) * prob_b
            if prob > 0:
                total_log += math.log2(prob)
            n += 1
        return -total_log / n if n > 0 else float('inf')


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LMVariantScore:
    alpha: float
    corpus_bpc: float
    signal_bpc: Optional[float]
    non_signal_bpc: Optional[float]
    signal_delta: float  # signal_bpc - corpus_bpc


@dataclass
class CodeSwitchTest:
    alpha: float
    signal_bpc: Optional[float]
    non_signal_bpc: Optional[float]
    signal_prefers_italian: bool  # True if signal_bpc lower at lower alpha


@dataclass
class MixedLMResult:
    # Corpus info
    n_latin_chars: int
    n_italian_chars: int
    latin_bpc_baseline: float
    italian_bpc_baseline: float

    # Sound change stats
    sound_change_sample: List[Dict]  # 10 before/after examples
    n_words_changed: int
    pct_words_changed: float

    # Per-alpha scores
    variant_scores: List[Dict]

    # Best variant
    best_alpha: float
    best_corpus_bpc: float
    best_signal_bpc: Optional[float]
    improvement_over_latin: float  # latin_bpc - best_bpc (positive = improvement)

    # Code-switching analysis
    code_switch_tests: List[Dict]
    signal_prefers_lower_alpha: bool  # SIGNAL tokens prefer more Italian weight
    signal_italian_trend: str  # 'ITALIAN_PREFERRED' or 'LATIN_PREFERRED' or 'NO_TREND'

    # Per-section language preference
    section_bpc: Dict[str, Dict]  # section -> {latin_bpc, italian_bpc, best_alpha}

    verdict: str
    runtime_seconds: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mixed_lm() -> None:
    """Step 34.8: Latin-Italian interpolated language model."""
    t0 = time.time()

    print("=" * 70)
    print("STEP 34.8: Latin-Italian Interpolated Language Model (Track C)")
    print("=" * 70)

    rd = _results_dir()

    # ── 1. Load Latin reference corpus ──
    print("\n  1. Loading Latin reference corpus …")
    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    raw_latin = ref_corpus.get_combined_text('latin')
    latin_text = _preprocess_text(raw_latin)
    print(f"     Latin text: {len(latin_text)} chars")

    # ── 2. Generate synthetic Northern Italian ──
    print("\n  2. Applying sound changes to generate synthetic Italian …")
    italian_text = _apply_sound_changes(latin_text)
    print(f"     Italian text: {len(italian_text)} chars")

    # Count changed words
    latin_words = latin_text.split()
    italian_words = italian_text.split()
    n_changed = sum(
        1 for lw, iw in zip(latin_words, italian_words) if lw != iw
    )
    pct_changed = n_changed / len(latin_words) if latin_words else 0.0
    print(f"     Changed words: {n_changed}/{len(latin_words)} ({pct_changed:.1%})")

    # Sample before/after
    sound_change_sample = []
    seen = set()
    for lw, iw in zip(latin_words, italian_words):
        if lw != iw and lw not in seen and len(sound_change_sample) < 10:
            sound_change_sample.append({'latin': lw, 'italian': iw})
            seen.add(lw)

    # ── 3. Train Latin and Italian LMs (5-gram) ──
    print("\n  3. Training character 5-gram LMs …")

    # Train/test split (80/20)
    split_lat = int(len(latin_text) * 0.8)
    latin_train = latin_text[:split_lat]
    latin_test = latin_text[split_lat:]

    split_ita = int(len(italian_text) * 0.8)
    italian_train = italian_text[:split_ita]
    italian_test = italian_text[split_ita:]

    lm_latin = CharNgramLM(order=5, alpha=1.0)
    lm_latin.train(latin_train)
    print(f"     Latin LM: {len(lm_latin.counts)} contexts")

    lm_italian = CharNgramLM(order=5, alpha=1.0)
    lm_italian.train(italian_train)
    print(f"     Italian LM: {len(lm_italian.counts)} contexts")

    # Baseline bpc on held-out text
    latin_bpc_baseline = lm_latin.bits_per_char(latin_test)
    italian_bpc_baseline = lm_italian.bits_per_char(italian_test)
    print(f"     Latin baseline bpc (held-out): {latin_bpc_baseline:.4f}")
    print(f"     Italian baseline bpc (held-out): {italian_bpc_baseline:.4f}")

    # ── 4. Load decoded Voynich corpus ──
    print("\n  4. Decoding Voynich corpus (R3 strategy) …")
    refine_path = os.path.join(rd, 'combined_refine.json')
    if not os.path.exists(refine_path):
        print("  [SKIP] combined_refine.json not found")
        return
    with open(refine_path) as f:
        refine_data = json.load(f)
    assignment = refine_data.get('best_assignment', {})

    mod_path = os.path.join(rd, 'modifier_integrate.json')
    if not os.path.exists(mod_path):
        print("  [SKIP] modifier_integrate.json not found")
        return
    with open(mod_path) as f:
        mod_data = json.load(f)
    modifier_chars, modifier_rules = _reconstruct_modifier_rules(mod_data)

    base_words = set(
        w.lower() for w in ref_corpus.get_combined_tokens('latin') if len(w) >= 2
    )
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    eva_to_triple = build_eva_to_triple_lookup()
    corpus = load_corpus(verbose=False)

    # Collect tokens with section info
    all_tokens: List[str] = []
    token_sections: List[str] = []
    for folio, page in corpus.pages.items():
        section = getattr(page, 'section', 'unknown')
        for token in page.all_tokens:
            all_tokens.append(token)
            token_sections.append(section)

    decoded = _decode_corpus_r3(
        all_tokens, assignment, eva_to_triple,
        modifier_chars, modifier_rules, ref_word_set,
    )
    n_tokens = len(decoded)
    print(f"     {n_tokens} tokens decoded")

    # ── 5. Load SIGNAL classifications ──
    print("\n  5. Loading SIGNAL classifications …")
    classifications: List[str] = []
    sig_path = os.path.join(rd, 'signal_bigrams.json')
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_data = json.load(f)
        classifications = sig_data.get('token_classifications', [])
        if len(classifications) != len(decoded):
            print(f"     [WARN] Classification length mismatch: "
                  f"{len(classifications)} vs {len(decoded)}")
            classifications = []
    if not classifications:
        print("     [WARN] No valid classifications — SIGNAL/non-SIGNAL split unavailable")

    signal_words = [
        decoded[i] for i in range(len(decoded))
        if i < len(classifications) and classifications[i] == 'SIGNAL'
    ] if classifications else []
    non_signal_words = [
        decoded[i] for i in range(len(decoded))
        if i < len(classifications) and classifications[i] != 'SIGNAL'
    ] if classifications else []

    print(f"     SIGNAL: {len(signal_words)}, non-SIGNAL: {len(non_signal_words)}")

    # ── 6. Score under interpolated LMs ──
    print("\n  6. Scoring under interpolated LMs …")
    corpus_text = _preprocess_text(' '.join(decoded))
    signal_text = _preprocess_text(' '.join(signal_words)) if signal_words else ''
    non_signal_text = _preprocess_text(' '.join(non_signal_words)) if non_signal_words else ''

    alphas = [0.3, 0.5, 0.7, 0.9, 1.0]  # 1.0 = pure Latin
    variant_scores: List[LMVariantScore] = []
    code_switch_tests: List[CodeSwitchTest] = []

    for alpha in alphas:
        if alpha >= 1.0:
            # Pure Latin
            c_bpc = lm_latin.bits_per_char(corpus_text)
            s_bpc = lm_latin.bits_per_char(signal_text) if signal_text else None
            ns_bpc = lm_latin.bits_per_char(non_signal_text) if non_signal_text else None
        else:
            interp = InterpolatedLM(lm_latin, lm_italian, alpha)
            c_bpc = interp.bits_per_char(corpus_text)
            s_bpc = interp.bits_per_char(signal_text) if signal_text else None
            ns_bpc = interp.bits_per_char(non_signal_text) if non_signal_text else None

        sig_delta = (s_bpc - c_bpc) if s_bpc is not None else 0.0

        vs = LMVariantScore(
            alpha=alpha,
            corpus_bpc=round(c_bpc, 6),
            signal_bpc=round(s_bpc, 6) if s_bpc is not None else None,
            non_signal_bpc=round(ns_bpc, 6) if ns_bpc is not None else None,
            signal_delta=round(sig_delta, 6),
        )
        variant_scores.append(vs)

        # Code-switching test
        signal_prefers = False
        if s_bpc is not None and ns_bpc is not None:
            # SIGNAL prefers Italian if its bpc is lower at lower alpha
            signal_prefers = s_bpc < ns_bpc
        code_switch_tests.append(CodeSwitchTest(
            alpha=alpha,
            signal_bpc=round(s_bpc, 6) if s_bpc is not None else None,
            non_signal_bpc=round(ns_bpc, 6) if ns_bpc is not None else None,
            signal_prefers_italian=signal_prefers,
        ))

        print(f"     alpha={alpha:.1f}: corpus_bpc={c_bpc:.4f}"
              f"{f', signal_bpc={s_bpc:.4f}' if s_bpc is not None else ''}"
              f"{f', non_signal_bpc={ns_bpc:.4f}' if ns_bpc is not None else ''}")

    # Find best alpha
    best_vs = min(variant_scores, key=lambda v: v.corpus_bpc)
    best_alpha = best_vs.alpha
    best_corpus_bpc = best_vs.corpus_bpc
    latin_only_bpc = next(v.corpus_bpc for v in variant_scores if v.alpha >= 1.0)
    improvement = latin_only_bpc - best_corpus_bpc

    print(f"\n     Best alpha: {best_alpha}")
    print(f"     Best corpus bpc: {best_corpus_bpc:.4f}")
    print(f"     Improvement over pure Latin: {improvement:+.4f}")

    # ── 7. Signal Italian trend ──
    print("\n  7. Code-switching analysis …")
    # Check if SIGNAL bpc decreases as alpha decreases (more Italian)
    signal_bpc_at_alphas = [
        (v.alpha, v.signal_bpc) for v in variant_scores
        if v.signal_bpc is not None
    ]
    signal_bpc_at_alphas.sort(key=lambda x: x[0])

    signal_prefers_lower_alpha = False
    signal_italian_trend = 'NO_TREND'
    if len(signal_bpc_at_alphas) >= 2:
        # Compare lowest alpha (most Italian) vs highest alpha (most Latin)
        lowest_alpha_bpc = signal_bpc_at_alphas[0][1]
        highest_alpha_bpc = signal_bpc_at_alphas[-1][1]
        if lowest_alpha_bpc < highest_alpha_bpc:
            signal_prefers_lower_alpha = True
            signal_italian_trend = 'ITALIAN_PREFERRED'
        elif lowest_alpha_bpc > highest_alpha_bpc:
            signal_italian_trend = 'LATIN_PREFERRED'
        print(f"     SIGNAL at alpha={signal_bpc_at_alphas[0][0]}: "
              f"{lowest_alpha_bpc:.4f}")
        print(f"     SIGNAL at alpha={signal_bpc_at_alphas[-1][0]}: "
              f"{highest_alpha_bpc:.4f}")
        print(f"     Trend: {signal_italian_trend}")

    # ── 8. Per-section language preference ──
    print("\n  8. Per-section language preference …")
    sections: Dict[str, List[str]] = defaultdict(list)
    for i, word in enumerate(decoded):
        sec = token_sections[i] if i < len(token_sections) else 'unknown'
        sections[sec].append(word)

    section_bpc: Dict[str, Dict] = {}
    for sec_name, sec_words in sorted(sections.items()):
        if len(sec_words) < 20:
            continue
        sec_text = _preprocess_text(' '.join(sec_words))
        if not sec_text:
            continue
        lat_bpc = lm_latin.bits_per_char(sec_text)
        ita_bpc = lm_italian.bits_per_char(sec_text)
        # Find best alpha for this section
        best_sec_alpha = 1.0
        best_sec_bpc = lat_bpc
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            interp = InterpolatedLM(lm_latin, lm_italian, alpha)
            bpc = interp.bits_per_char(sec_text)
            if bpc < best_sec_bpc:
                best_sec_bpc = bpc
                best_sec_alpha = alpha
        section_bpc[sec_name] = {
            'n_tokens': len(sec_words),
            'latin_bpc': round(lat_bpc, 4),
            'italian_bpc': round(ita_bpc, 4),
            'best_alpha': best_sec_alpha,
            'best_bpc': round(best_sec_bpc, 4),
        }
        print(f"     {sec_name:15s}: n={len(sec_words):5d}, "
              f"lat={lat_bpc:.3f}, ita={ita_bpc:.3f}, "
              f"best_alpha={best_sec_alpha}")

    # ── 9. Verdict ──
    verdict_parts = []

    if improvement > 0.05:
        verdict_parts.append(
            f"MIXED_BETTER: best alpha={best_alpha}, "
            f"improvement={improvement:+.4f} bpc over pure Latin"
        )
    elif improvement > 0:
        verdict_parts.append(
            f"MIXED_MARGINAL: best alpha={best_alpha}, "
            f"improvement={improvement:+.4f} bpc (marginal)"
        )
    else:
        verdict_parts.append(
            f"LATIN_BETTER: pure Latin wins by {-improvement:.4f} bpc"
        )

    verdict_parts.append(f"Code-switching: {signal_italian_trend}")

    # Count sections preferring Italian (best_alpha < 1.0)
    n_sec_italian = sum(
        1 for info in section_bpc.values() if info['best_alpha'] < 1.0
    )
    n_sec_total = len(section_bpc)
    verdict_parts.append(
        f"Sections preferring Italian: {n_sec_italian}/{n_sec_total}"
    )

    verdict = '; '.join(verdict_parts)

    print(f"\n  Verdict: {verdict}")

    # ── 10. Save ──
    elapsed = round(time.time() - t0, 2)

    result = MixedLMResult(
        n_latin_chars=len(latin_text),
        n_italian_chars=len(italian_text),
        latin_bpc_baseline=round(latin_bpc_baseline, 6),
        italian_bpc_baseline=round(italian_bpc_baseline, 6),
        sound_change_sample=sound_change_sample,
        n_words_changed=n_changed,
        pct_words_changed=round(pct_changed, 4),
        variant_scores=[_convert(v) for v in variant_scores],
        best_alpha=best_alpha,
        best_corpus_bpc=round(best_corpus_bpc, 6),
        best_signal_bpc=round(best_vs.signal_bpc, 6) if best_vs.signal_bpc is not None else None,
        improvement_over_latin=round(improvement, 6),
        code_switch_tests=[_convert(t) for t in code_switch_tests],
        signal_prefers_lower_alpha=signal_prefers_lower_alpha,
        signal_italian_trend=signal_italian_trend,
        section_bpc=section_bpc,
        verdict=verdict,
        runtime_seconds=elapsed,
    )

    out_path = os.path.join(rd, 'mixed_lm.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(result), f, indent=2)
    print(f"\n  -> {out_path}  ({elapsed:.1f}s)")
