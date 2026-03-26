"""
Phase 66, Track 12: Astronomical Deep Dive
============================================
Separate analysis of astronomical/cosmological sections vs pharmaceutical:
coda distribution, vocabulary overlap, dict hit, entropy profile.

Dependency chain:
    results/combined_refine.json      (Phase 15)
        -> results/p66_astro_deep.json
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import build_coda_table_v2, decode_corpus_cvc_v2, decode_token_cvc_v2
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51


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
class AstroDeepResult:
    phase: str = "66"
    step: str = "66.12"
    experiment: str = "astronomical_deep_dive"
    n_astro_tokens: int = 0
    n_pharma_tokens: int = 0
    n_astro_pages: int = 0
    n_pharma_pages: int = 0
    astro_dict_hit: float = 0.0
    pharma_dict_hit: float = 0.0
    dict_hit_diff_pp: float = 0.0
    astro_coda_dist: Dict[str, int] = field(default_factory=dict)
    pharma_coda_dist: Dict[str, int] = field(default_factory=dict)
    coda_chi2: float = 0.0
    coda_chi2_p: float = 1.0
    vocab_overlap_jaccard: float = 0.0
    astro_entropy: float = 0.0
    pharma_entropy: float = 0.0
    n_astro_types: int = 0
    n_pharma_types: int = 0
    # Gates
    a1_coda_chi2: bool = False     # chi2 p < 0.05
    a2_dict_diff: bool = False     # astro dict hit < pharma by >= 10pp
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CODA_CONSONANTS = set('nrstlm')

ASTRO_SECTIONS = {'astronomical', 'cosmological'}
PHARMA_SECTIONS = {'pharmaceutical'}

# Fallback folio ranges for section matching
ASTRO_FOLIO_PREFIXES = tuple(f'f{n}' for n in range(67, 75))  # f67-f74
ASTRO_FOLIO_EXTRA = tuple(f'f{n}' for n in range(85, 87))     # f85-f86
PHARMA_FOLIO_PREFIXES = tuple(f'f{n}' for n in range(88, 103)) # f88-f102


def _is_astro_page(page) -> bool:
    """Check if a page belongs to the astronomical/cosmological section."""
    if page.section in ASTRO_SECTIONS:
        return True
    folio = page.folio
    for prefix in ASTRO_FOLIO_PREFIXES + ASTRO_FOLIO_EXTRA:
        if folio.startswith(prefix):
            return True
    return False


def _is_pharma_page(page) -> bool:
    """Check if a page belongs to the pharmaceutical section."""
    if page.section in PHARMA_SECTIONS:
        return True
    folio = page.folio
    for prefix in PHARMA_FOLIO_PREFIXES:
        if folio.startswith(prefix):
            return True
    return False


def _get_coda_char(decoded_word: str) -> Optional[str]:
    """Return the coda consonant if the word ends in one, else None."""
    if not decoded_word:
        return None
    last = decoded_word[-1].lower()
    if last in CODA_CONSONANTS:
        return last
    return None


def _shannon_entropy(counter: Counter) -> float:
    """Compute Shannon entropy H1 = -sum(p * log2(p))."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return h


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity of two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_astro_deep() -> AstroDeepResult:
    """Phase 66, Track 12: Astronomical Deep Dive."""
    t0 = time.time()
    rd = _results_dir()
    result = AstroDeepResult()

    print("=" * 70)
    print("Phase 66, Track 12: Astronomical Deep Dive")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load corpus and dependencies
    # ------------------------------------------------------------------
    print("\n[1] Loading corpus and CVC decode resources ...")
    corpus = load_corpus(verbose=False)

    refine_path = os.path.join(rd, "combined_refine.json")
    refine = _safe_load(refine_path)
    if not refine:
        print("  WARNING: combined_refine.json not found; using empty assignment")
    assignment = refine.get("best_assignment", {})

    eva_to_triple = build_eva_to_triple_lookup()
    coda_table = build_coda_table_v2()

    # Load 10K reference dictionary
    print("  Loading reference dictionary ...")
    ref_corpus = load_reference_corpus(languages=["latin"], verbose=False)
    all_ref_tokens = ref_corpus.get_combined_tokens("latin")
    dict_10k: Set[str] = {w.lower() for w in all_ref_tokens if 2 <= len(w) <= 15}
    print(f"  Reference dictionary: {len(dict_10k)} words")

    # ------------------------------------------------------------------
    # 2. Split corpus by section
    # ------------------------------------------------------------------
    print("\n[2] Splitting corpus into astro vs pharma sections ...")
    astro_tokens: List[str] = []
    pharma_tokens: List[str] = []
    astro_pages_set: Set[str] = set()
    pharma_pages_set: Set[str] = set()

    for folio, page in corpus.pages.items():
        if _is_astro_page(page):
            astro_tokens.extend(page.all_tokens)
            astro_pages_set.add(folio)
        elif _is_pharma_page(page):
            pharma_tokens.extend(page.all_tokens)
            pharma_pages_set.add(folio)

    result.n_astro_tokens = len(astro_tokens)
    result.n_pharma_tokens = len(pharma_tokens)
    result.n_astro_pages = len(astro_pages_set)
    result.n_pharma_pages = len(pharma_pages_set)

    print(f"  Astro:  {result.n_astro_tokens} tokens, {result.n_astro_pages} pages")
    print(f"  Pharma: {result.n_pharma_tokens} tokens, {result.n_pharma_pages} pages")

    if result.n_astro_tokens == 0 or result.n_pharma_tokens == 0:
        print("  WARNING: One or both sections are empty. Aborting analysis.")
        result.verdict = "INSUFFICIENT_DATA"
        result.runtime_seconds = round(time.time() - t0, 2)
        _save_json(rd, "p66_astro_deep.json", asdict(result))
        return result

    # ------------------------------------------------------------------
    # 3. CVC decode both subsections
    # ------------------------------------------------------------------
    print("\n[3] CVC decoding both sections ...")
    astro_decoded = decode_corpus_cvc_v2(
        astro_tokens, assignment, eva_to_triple, coda_table
    )
    pharma_decoded = decode_corpus_cvc_v2(
        pharma_tokens, assignment, eva_to_triple, coda_table
    )
    print(f"  Astro decoded:  {len(astro_decoded)} words")
    print(f"  Pharma decoded: {len(pharma_decoded)} words")

    # ------------------------------------------------------------------
    # 4. Dict hit for each section
    # ------------------------------------------------------------------
    print("\n[4] Computing dict hit rates ...")
    astro_hits = sum(1 for w in astro_decoded if w.lower() in dict_10k and w != '?')
    pharma_hits = sum(1 for w in pharma_decoded if w.lower() in dict_10k and w != '?')

    astro_valid = sum(1 for w in astro_decoded if w and w != '?')
    pharma_valid = sum(1 for w in pharma_decoded if w and w != '?')

    result.astro_dict_hit = (astro_hits / astro_valid * 100) if astro_valid > 0 else 0.0
    result.pharma_dict_hit = (pharma_hits / pharma_valid * 100) if pharma_valid > 0 else 0.0
    result.dict_hit_diff_pp = result.astro_dict_hit - result.pharma_dict_hit

    print(f"  Astro dict hit:  {result.astro_dict_hit:.1f}% ({astro_hits}/{astro_valid})")
    print(f"  Pharma dict hit: {result.pharma_dict_hit:.1f}% ({pharma_hits}/{pharma_valid})")
    print(f"  Difference:      {result.dict_hit_diff_pp:+.1f} pp")

    # ------------------------------------------------------------------
    # 5. Coda distribution
    # ------------------------------------------------------------------
    print("\n[5] Computing coda distributions ...")
    astro_coda_counts: Counter = Counter()
    pharma_coda_counts: Counter = Counter()

    for w in astro_decoded:
        coda = _get_coda_char(w)
        if coda:
            astro_coda_counts[coda] += 1

    for w in pharma_decoded:
        coda = _get_coda_char(w)
        if coda:
            pharma_coda_counts[coda] += 1

    result.astro_coda_dist = dict(sorted(astro_coda_counts.items()))
    result.pharma_coda_dist = dict(sorted(pharma_coda_counts.items()))

    print(f"  Astro codas:  {dict(astro_coda_counts.most_common())}")
    print(f"  Pharma codas: {dict(pharma_coda_counts.most_common())}")

    # ------------------------------------------------------------------
    # 6. Chi-squared test on coda contingency table
    # ------------------------------------------------------------------
    print("\n[6] Chi-squared test on coda contingency table ...")
    all_codas = sorted(set(astro_coda_counts.keys()) | set(pharma_coda_counts.keys()))

    if len(all_codas) >= 2:
        # Build contingency table: rows = [astro, pharma], cols = codas
        astro_row = [astro_coda_counts.get(c, 0) for c in all_codas]
        pharma_row = [pharma_coda_counts.get(c, 0) for c in all_codas]
        contingency = [astro_row, pharma_row]

        try:
            from scipy.stats import chi2_contingency
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            result.coda_chi2 = float(chi2)
            result.coda_chi2_p = float(p_val)
        except ImportError:
            print("  WARNING: scipy not available; skipping chi-squared test")
            result.coda_chi2 = 0.0
            result.coda_chi2_p = 1.0
    else:
        print("  WARNING: Fewer than 2 coda categories; skipping chi-squared test")
        result.coda_chi2 = 0.0
        result.coda_chi2_p = 1.0

    print(f"  Chi2 = {result.coda_chi2:.2f}, p = {result.coda_chi2_p:.6f}")

    # ------------------------------------------------------------------
    # 7. Vocabulary overlap (Jaccard)
    # ------------------------------------------------------------------
    print("\n[7] Computing vocabulary overlap ...")
    astro_types: Set[str] = {w.lower() for w in astro_decoded if w and w != '?'}
    pharma_types: Set[str] = {w.lower() for w in pharma_decoded if w and w != '?'}

    result.n_astro_types = len(astro_types)
    result.n_pharma_types = len(pharma_types)
    result.vocab_overlap_jaccard = _jaccard(astro_types, pharma_types)

    shared = astro_types & pharma_types
    print(f"  Astro types:  {result.n_astro_types}")
    print(f"  Pharma types: {result.n_pharma_types}")
    print(f"  Shared types: {len(shared)}")
    print(f"  Jaccard:      {result.vocab_overlap_jaccard:.4f}")

    # ------------------------------------------------------------------
    # 8. Shannon entropy of word frequency distributions
    # ------------------------------------------------------------------
    print("\n[8] Computing Shannon entropy H1 ...")
    astro_freq = Counter(w.lower() for w in astro_decoded if w and w != '?')
    pharma_freq = Counter(w.lower() for w in pharma_decoded if w and w != '?')

    result.astro_entropy = round(_shannon_entropy(astro_freq), 4)
    result.pharma_entropy = round(_shannon_entropy(pharma_freq), 4)

    print(f"  Astro H1:  {result.astro_entropy:.4f} bits")
    print(f"  Pharma H1: {result.pharma_entropy:.4f} bits")

    # ------------------------------------------------------------------
    # 9. Evaluate gates
    # ------------------------------------------------------------------
    print("\n[9] Evaluating gates ...")

    # A1: coda chi2 p < 0.05
    result.a1_coda_chi2 = result.coda_chi2_p < 0.05
    print(f"  A1 coda chi2 p < 0.05: {result.a1_coda_chi2}  (p={result.coda_chi2_p:.6f})")

    # A2: astro dict_hit < pharma dict_hit by >= 10pp
    result.a2_dict_diff = result.dict_hit_diff_pp <= -10.0
    print(f"  A2 dict diff >= 10pp:  {result.a2_dict_diff}  "
          f"(diff={result.dict_hit_diff_pp:+.1f}pp)")

    result.gates_passed = sum([result.a1_coda_chi2, result.a2_dict_diff])
    result.gate_passed = result.gates_passed >= 2

    if result.gate_passed:
        result.verdict = "SECTIONS_DIVERGENT"
    elif result.gates_passed == 1:
        result.verdict = "PARTIAL_DIVERGENCE"
    else:
        result.verdict = "SECTIONS_UNIFORM"

    # ------------------------------------------------------------------
    # 10. Summary and save
    # ------------------------------------------------------------------
    result.runtime_seconds = round(time.time() - t0, 2)

    print("\n" + "=" * 70)
    print(f"VERDICT: {result.verdict}")
    print(f"  Gates passed: {result.gates_passed}/2")
    print(f"  Astro:  {result.n_astro_tokens} tokens, {result.n_astro_pages} pages, "
          f"dict_hit={result.astro_dict_hit:.1f}%, H1={result.astro_entropy:.2f}")
    print(f"  Pharma: {result.n_pharma_tokens} tokens, {result.n_pharma_pages} pages, "
          f"dict_hit={result.pharma_dict_hit:.1f}%, H1={result.pharma_entropy:.2f}")
    print(f"  Coda chi2={result.coda_chi2:.2f} (p={result.coda_chi2_p:.6f})")
    print(f"  Vocab Jaccard={result.vocab_overlap_jaccard:.4f}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")
    print("=" * 70)

    saved = _save_json(rd, "p66_astro_deep.json", asdict(result))
    print(f"  Saved -> {saved}")

    return result
