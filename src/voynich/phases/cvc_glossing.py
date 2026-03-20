"""
Phase 59, Investigation 4: CVC Signal Word Glossing
=====================================================
Phase 57 found 64 CVC signal words but didn't identify what they mean.
This module attempts to gloss each CVC signal word by looking it up in
Latin and Italian dictionaries, classifying them by semantic domain.

Dependency chain:
    results/cvc_coda_signal.json      (Phase 57.4)
    Latin + Italian reference corpora
        -> results/cvc_glossing.json
"""

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import build_expanded_word_set, load_reference_corpus


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
# Known vocabulary sets
# ---------------------------------------------------------------------------

ROMANCE_FUNCTION_WORDS = {
    'de', 'di', 'du', 'da', 'in', 'ad', 'et', 'se', 'si', 'cu', 'ce',
    'la', 'le', 'lo', 'li', 'un', 'il', 'al', 'el', 'ne', 'no', 'ni',
    'co', 'con', 'per', 'pro', 'sed', 'non', 'que', 'qui',
}

PHARMA_TERMS = {
    'recipe', 'accipe', 'misce', 'cola', 'distilla', 'tere', 'solve',
    'coralli', 'diasene', 'stercora', 'radicom', 'commune', 'secundi',
    'ratione', 'balsamo', 'radice', 'herba', 'aqua', 'oleum', 'pulvis',
    'semen', 'cortex', 'folia', 'flores', 'succo', 'gummi',
    'dosi', 'cura', 'morbo', 'febre', 'dolor', 'sana',
}

BOTANICAL_TERMS = {
    'rosa', 'viola', 'salvia', 'calendula', 'urtica', 'plantago',
    'camomilla', 'menta', 'basilico', 'rosmarino', 'finocchio',
    'radix', 'folium', 'flos', 'semen', 'cortex', 'herba',
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GlossedWord:
    """A CVC signal word with attempted gloss."""
    word: str
    sigma: float
    selectivity: float
    count: int
    latin_match: Optional[str] = None
    latin_ed: int = -1
    italian_match: Optional[str] = None
    italian_ed: int = -1
    category: str = 'UNKNOWN'
    gloss: str = '?'


@dataclass
class VocabProfile:
    """Comparison of CV vs CVC vocabulary profiles."""
    cv_n_words: int = 0
    cvc_n_words: int = 0
    cv_function_frac: float = 0.0
    cvc_function_frac: float = 0.0
    cv_content_frac: float = 0.0
    cvc_content_frac: float = 0.0
    cv_mean_length: float = 0.0
    cvc_mean_length: float = 0.0
    overlap: int = 0


@dataclass
class CvcGlossingResult:
    """Full Investigation 4 output."""
    phase: str = "59"
    investigation: str = "4"
    experiment: str = "cvc_glossing"
    n_glossed: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    function_fraction: float = 0.0
    content_fraction: float = 0.0
    pharma_fraction: float = 0.0
    vocab_profile: Optional[VocabProfile] = None
    words: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    g1_glossed: bool = False           # ≥ 20 with glosses
    g2_content: bool = False           # content fraction > 30%
    g3_pharma: bool = False            # ≥ 5 pharma/botanical
    g4_longer: bool = False            # CVC mean length > CV mean length
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein edit distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def find_closest(word: str, dictionary: Set[str], max_ed: int = 1) -> Optional[Dict]:
    """Find closest dictionary match within edit distance."""
    wl = word.lower()
    if wl in dictionary:
        return {'word': wl, 'ed': 0}

    if max_ed >= 1:
        best = None
        best_ed = max_ed + 1
        # Only check words of similar length for speed
        for dw in dictionary:
            if abs(len(dw) - len(wl)) > max_ed:
                continue
            ed = _edit_distance(wl, dw)
            if ed <= max_ed and ed < best_ed:
                best = dw
                best_ed = ed
                if ed == 1:
                    break  # good enough
        if best:
            return {'word': best, 'ed': best_ed}

    return None


def classify_word(
    word: str,
    latin_match: Optional[Dict],
    italian_match: Optional[Dict],
) -> Tuple[str, str]:
    """Classify a signal word and return (category, gloss)."""
    wl = word.lower()

    if wl in ROMANCE_FUNCTION_WORDS:
        return 'FUNCTION', wl

    if wl in PHARMA_TERMS:
        return 'PHARMACEUTICAL', wl

    if wl in BOTANICAL_TERMS:
        return 'BOTANICAL', wl

    if latin_match and latin_match['ed'] == 0:
        if latin_match['word'] in PHARMA_TERMS:
            return 'PHARMACEUTICAL', latin_match['word']
        return 'LATIN_EXACT', latin_match['word']

    if italian_match and italian_match['ed'] == 0:
        if italian_match['word'] in PHARMA_TERMS:
            return 'PHARMACEUTICAL', italian_match['word']
        return 'ITALIAN_EXACT', italian_match['word']

    if latin_match and latin_match['ed'] <= 1:
        return 'LATIN_ED1', latin_match['word']

    if italian_match and italian_match['ed'] <= 1:
        return 'ITALIAN_ED1', italian_match['word']

    return 'UNKNOWN', '?'


def gloss_signal_words(
    signal_words: List[Dict[str, Any]],
    latin_dict: Set[str],
    italian_dict: Set[str],
) -> List[GlossedWord]:
    """Gloss all CVC signal words."""
    glossed = []
    for ws in signal_words:
        word = ws['word']
        sigma = ws.get('sigma', 0)
        sel = ws.get('selectivity', 0)
        count = ws.get('real_count', 0)

        latin_match = find_closest(word, latin_dict, max_ed=1)
        italian_match = find_closest(word, italian_dict, max_ed=1)

        category, gloss = classify_word(word, latin_match, italian_match)

        glossed.append(GlossedWord(
            word=word,
            sigma=sigma,
            selectivity=sel,
            count=count,
            latin_match=latin_match['word'] if latin_match else None,
            latin_ed=latin_match['ed'] if latin_match else -1,
            italian_match=italian_match['word'] if italian_match else None,
            italian_ed=italian_match['ed'] if italian_match else -1,
            category=category,
            gloss=gloss,
        ))

    return glossed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cvc_gloss():
    """Investigation 4: Gloss CVC signal words."""
    t0 = time.time()
    print("=" * 70)
    print("Phase 59, Investigation 4: CVC Signal Word Glossing")
    print("=" * 70)

    rd = str(_results_dir())

    # Load CVC signal words from Phase 57
    cvc_signal_data = _safe_load(os.path.join(rd, 'cvc_coda_signal.json'))
    signal_section = cvc_signal_data.get('signal', {})
    cvc_signal_words = signal_section.get('top_signal_words', [])

    if not cvc_signal_words:
        print("  No CVC signal words found in cvc_coda_signal.json")
        result = CvcGlossingResult(runtime_seconds=round(time.time() - t0, 2))
        _save_json(rd, 'cvc_glossing.json', result)
        return

    print(f"\n  CVC signal words to gloss: {len(cvc_signal_words)}")

    # Load reference dictionaries
    print("  Loading reference dictionaries ...")
    ref_corpus = load_reference_corpus(languages=['latin', 'italian'], verbose=False)
    latin_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                      if len(w) >= 2)
    italian_words = set()
    try:
        italian_words = set(w.lower() for w in ref_corpus.get_combined_tokens('italian')
                            if len(w) >= 2)
    except (KeyError, AttributeError):
        pass

    # Expand Latin
    expanded, _ = build_expanded_word_set(latin_words)
    latin_dict = latin_words | expanded
    print(f"  Latin dict: {len(latin_dict)}, Italian dict: {len(italian_words)}")

    # Gloss
    print("\n  Glossing signal words ...")
    glossed = gloss_signal_words(cvc_signal_words, latin_dict, italian_words)

    # Categorize
    categories = Counter(g.category for g in glossed)
    n_function = categories.get('FUNCTION', 0)
    n_unknown = categories.get('UNKNOWN', 0)
    n_pharma = categories.get('PHARMACEUTICAL', 0) + categories.get('BOTANICAL', 0)
    n_total = len(glossed)
    function_frac = n_function / n_total if n_total > 0 else 0
    content_frac = 1 - function_frac - (n_unknown / n_total if n_total > 0 else 0)
    pharma_frac = n_pharma / n_total if n_total > 0 else 0

    print(f"\n  Categories:")
    for cat, cnt in categories.most_common():
        print(f"    {cat:<20s} {cnt:>4}")
    print(f"  Function fraction: {function_frac:.1%}")
    print(f"  Content fraction:  {content_frac:.1%}")
    print(f"  Pharma fraction:   {pharma_frac:.1%}")

    # Print glossed words
    print(f"\n  {'Word':<14} {'Sigma':>6} {'Sel':>6} {'Count':>6} {'Category':<20} {'Gloss'}")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*6} {'-'*20} {'-'*14}")
    for g in glossed:
        print(f"  {g.word:<14} {g.sigma:>6.1f} {g.selectivity:>6.1f} "
              f"{g.count:>6} {g.category:<20} {g.gloss}")

    # Compare with CV signal words (from Phase 28/29 if available)
    cv_signal_data = _safe_load(os.path.join(rd, 'signal_bigrams.json'))
    cv_signal_words = []
    if cv_signal_data:
        # Phase 29 stores signal word info differently
        for ws in cv_signal_data.get('signal_words', []):
            cv_signal_words.append(ws)

    # Build vocab profile
    cvc_lengths = [len(g.word) for g in glossed]
    cv_lengths = [len(w.get('word', '')) for w in cv_signal_words] if cv_signal_words else []
    cv_set = set(w.get('word', '') for w in cv_signal_words)
    cvc_set = set(g.word for g in glossed)

    cv_func = sum(1 for w in cv_signal_words
                  if w.get('word', '').lower() in ROMANCE_FUNCTION_WORDS)

    profile = VocabProfile(
        cv_n_words=len(cv_signal_words),
        cvc_n_words=len(glossed),
        cv_function_frac=cv_func / len(cv_signal_words) if cv_signal_words else 0,
        cvc_function_frac=function_frac,
        cv_content_frac=1 - (cv_func / len(cv_signal_words) if cv_signal_words else 0),
        cvc_content_frac=content_frac,
        cv_mean_length=float(np.mean(cv_lengths)) if cv_lengths else 0,
        cvc_mean_length=float(np.mean(cvc_lengths)) if cvc_lengths else 0,
        overlap=len(cv_set & cvc_set),
    )

    print(f"\n  Vocabulary Profile Comparison:")
    print(f"    CV signal words:  {profile.cv_n_words}")
    print(f"    CVC signal words: {profile.cvc_n_words}")
    print(f"    CV mean length:   {profile.cv_mean_length:.1f}")
    print(f"    CVC mean length:  {profile.cvc_mean_length:.1f}")
    print(f"    Overlap:          {profile.overlap}")

    # Gates
    n_with_gloss = n_total - n_unknown
    g1 = n_with_gloss >= 20
    g2 = content_frac > 0.30
    g3 = n_pharma >= 5
    g4 = profile.cvc_mean_length > profile.cv_mean_length if profile.cv_mean_length > 0 else False
    gates_passed = sum([g1, g2, g3, g4])

    print(f"\n  Validation Gates:")
    print(f"    G1 ≥ 20 glossed:           {'PASS' if g1 else 'FAIL'} ({n_with_gloss})")
    print(f"    G2 content fraction > 30%: {'PASS' if g2 else 'FAIL'} ({content_frac:.1%})")
    print(f"    G3 ≥ 5 pharma/botanical:   {'PASS' if g3 else 'FAIL'} ({n_pharma})")
    print(f"    G4 CVC longer than CV:     {'PASS' if g4 else 'FAIL'}")
    print(f"    Gates passed: {gates_passed}/4")

    result = CvcGlossingResult(
        n_glossed=n_total,
        categories=dict(categories),
        function_fraction=round(function_frac, 4),
        content_fraction=round(content_frac, 4),
        pharma_fraction=round(pharma_frac, 4),
        vocab_profile=profile,
        words=[_convert(g) for g in glossed],
        g1_glossed=g1,
        g2_content=g2,
        g3_pharma=g3,
        g4_longer=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 2),
    )

    path = _save_json(rd, 'cvc_glossing.json', result)
    print(f"\n  Saved: {path}")
    print(f"  Investigation 4 completed in {time.time() - t0:.1f}s")
