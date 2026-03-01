"""
Reference Corpus Loading and Cleaning
======================================
Load real historical texts for comparison with Voynich fingerprints.

Directory layout:
    data/reference/<language>/<text_name>.txt

Each .txt file is a cleaned plain-text corpus. The language is inferred
from the parent directory name.
"""

import json
import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from voynich.core._paths import data_dir as _data_dir


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReferenceText:
    """A single real-world reference text."""
    language: str
    name: str
    text: str
    tokens: List[str]
    token_count: int
    filepath: Path


@dataclass
class ReferenceCorpus:
    """Collection of reference texts organized by language."""
    texts: Dict[str, List[ReferenceText]] = field(default_factory=dict)

    @property
    def languages(self) -> List[str]:
        return sorted(lang for lang, txts in self.texts.items() if txts)

    def get_texts(self, language: str) -> List[ReferenceText]:
        return self.texts.get(language, [])

    def get_combined_text(self, language: str) -> str:
        return ' '.join(t.text for t in self.get_texts(language))

    def get_combined_tokens(self, language: str) -> List[str]:
        out: List[str] = []
        for t in self.get_texts(language):
            out.extend(t.tokens)
        return out

    def summary(self) -> Dict:
        info: Dict = {}
        for lang in self.languages:
            txts = self.get_texts(lang)
            info[lang] = {
                'texts': len(txts),
                'total_tokens': sum(t.token_count for t in txts),
                'files': [t.name for t in txts],
            }
        return info


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _strip_rtf(raw_bytes: bytes) -> str:
    """Convert RTF bytes to plain text, handling Unicode escapes."""
    text = raw_bytes.decode('latin-1')

    # Decode \uc0\uNNNN and standalone \uNNNN escapes
    def _unicode_repl(m: re.Match) -> str:
        try:
            return chr(int(m.group(1)))
        except (ValueError, OverflowError):
            return ''

    text = re.sub(r'\\uc0\\u(\d+)\s?', _unicode_repl, text)
    text = re.sub(r'\\u(\d+)\s?', _unicode_repl, text)

    # Decode hex escapes (\'xx) as Windows-1252
    def _hex_repl(m: re.Match) -> str:
        try:
            return bytes([int(m.group(1), 16)]).decode('cp1252')
        except (ValueError, UnicodeDecodeError):
            return ''

    text = re.sub(r"\\'([0-9a-fA-F]{2})", _hex_repl, text)

    # Remove RTF header groups
    text = re.sub(r'\{\\fonttbl[^}]*\}', '', text)
    text = re.sub(r'\{\\colortbl[^}]*\}', '', text)
    text = re.sub(r'\{\\\*\\expandedcolortbl[^}]*\}', '', text)
    text = re.sub(r'\{\\\*[^}]*\}', '', text)

    # Remove RTF control words, keeping trailing space
    text = re.sub(r'\\[a-z]+\d*\s?', ' ', text)

    # Remove braces
    text = text.replace('{', '').replace('}', '')

    # Join continuation lines (backslash-newline)
    text = text.replace(' \n', ' ')

    return text


def _is_apparatus_line(line: str) -> bool:
    """Heuristic: detect critical apparatus lines (manuscript variant notes)."""
    s = line.strip()
    if len(s) < 10:
        return False

    # Greek characters are almost exclusively in apparatus
    greek_count = sum(1 for c in s if '\u0370' <= c <= '\u03FF'
                      or '\u1F00' <= c <= '\u1FFF')
    if greek_count > 2:
        return True

    # Apparatus keywords
    if re.search(r'\b(codd|edd|lect|nostram|habent|omitt|legit)\b', s, re.I):
        return True

    # Dense manuscript sigla: many single/double uppercase letters separated
    # by commas or semicolons (e.g. "BB, yy, nn, 99, xx")
    sigla_hits = re.findall(r'\b[A-Z]{1,2}\b', s)
    semicolons = s.count(';')
    if len(sigla_hits) > 4 and semicolons > 2:
        return True

    # Starts with a line-number + fragment followed by sigla pattern
    # e.g. "289 Cum oleo misc. roseo nn, 9-9."
    if re.match(r'^\d{1,4}\s+\w', s):
        # Check if it's mostly apparatus (lots of commas + sigla)
        if semicolons > 1 and len(sigla_hits) > 2:
            return True

    return False


def _is_garbled(line: str) -> bool:
    """Detect OCR-garbled lines (short fragments, random chars)."""
    words = line.split()
    if not words:
        return True
    # Garbled lines have many very short tokens (1-2 chars)
    short = sum(1 for w in words if len(w) <= 2)
    if len(words) >= 3 and short / len(words) > 0.6:
        return True
    # Lines with very low ratio of Latin-letter content
    latin_alpha = sum(1 for c in line if c.isalpha() and ord(c) < 0x0250)
    if len(line) > 10 and latin_alpha / len(line) < 0.4:
        return True
    return False


def clean_reference_text(raw: str, language: str = 'latin') -> str:
    """
    Clean raw text for entropy analysis.

    Strips editorial markup, normalizes characters, lowercases, and returns
    a whitespace-separated string of words.
    """
    lines = raw.split('\n')
    kept: List[str] = []

    # Detect where sustained real text begins — skip garbled OCR preambles.
    # Look for a window of consecutive non-garbled, non-apparatus lines.
    start_idx = 0
    window_size = 5
    for i in range(len(lines)):
        s = lines[i].strip()
        if not s:
            continue
        if len(s) < 40:
            continue
        # Check a window of lines for sustained quality
        good = 0
        for j in range(i, min(i + window_size, len(lines))):
            ls = lines[j].strip()
            if ls and len(ls) > 30 and not _is_garbled(ls) and not _is_apparatus_line(ls):
                good += 1
        if good >= 3:
            start_idx = i
            break

    for line in lines[start_idx:]:
        s = line.strip()
        if not s:
            continue

        # Skip apparatus lines (critical edition variant notes)
        if _is_apparatus_line(s):
            continue

        # Skip garbled OCR fragments
        if _is_garbled(s):
            continue

        # Skip very short lines that are just noise
        alpha_count = sum(1 for c in s if c.isalpha() and ord(c) < 0x0250)
        if len(s) > 5 and alpha_count / len(s) < 0.3:
            continue

        kept.append(s)

    text = ' '.join(kept)

    # Strip '' used as section delimiters in Circa Instans
    text = text.replace("''", ' ')

    # Strip entry numbering like "2. Accacia" — keep the word, drop number+dot
    text = re.sub(r'\b\d{1,4}\.\s+', ' ', text)

    # Strip standalone numbers (verse line numbers like "585")
    text = re.sub(r'\b\d+\b', ' ', text)

    # Strip degree markers (3° → empty)
    text = text.replace('°', ' ')

    # Remove non-Latin-alphabet characters (keep accented Latin letters + spaces)
    # This strips Greek, punctuation, brackets, etc.
    cleaned_chars: List[str] = []
    for c in text:
        if c == ' ':
            cleaned_chars.append(c)
        elif c.isalpha():
            cat = unicodedata.category(c)
            # Keep Latin letters (basic + extended), drop Greek/Cyrillic/etc.
            if ord(c) < 0x0250 or cat.startswith('L'):
                # Double-check: skip Greek and Cyrillic blocks
                if not ('\u0370' <= c <= '\u03FF' or '\u0400' <= c <= '\u04FF'
                        or '\u1F00' <= c <= '\u1FFF'):
                    cleaned_chars.append(c)

    text = ''.join(cleaned_chars)

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Post-cleaning token filter: remove apparatus sigla that survived
    # line-level filtering (single letters used as manuscript codes,
    # double-letter sigla like nn/yy/bb/cc/xx/ee/ze)
    _SIGLA = frozenset({
        'nn', 'yy', 'bb', 'cc', 'xx', 'ee', 'ze', 'dd', 'gg', 'ff',
        'pp', 'rr', 'ss', 'tt', 'ww', 'zz', 'aa',
    })
    tokens = text.split()
    tokens = [w for w in tokens if len(w) >= 2 and w not in _SIGLA]
    text = ' '.join(tokens)

    return text


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------

def _default_reference_dir() -> str:
    return str(_data_dir('reference'))


def discover_reference_texts(
    data_dir: Optional[str] = None,
) -> Dict[str, List[Path]]:
    """
    Scan data/reference/ for available texts organized by language.
    Returns {language: [path1.txt, path2.txt, ...]}.
    """
    if data_dir is None:
        data_dir = _default_reference_dir()

    found: Dict[str, List[Path]] = {}
    ref_path = Path(data_dir)
    if not ref_path.is_dir():
        return found

    for lang_dir in sorted(ref_path.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name.startswith('.'):
            continue
        language = lang_dir.name
        txt_files = sorted(lang_dir.glob('*.txt'))
        if txt_files:
            found[language] = txt_files

    return found


def load_reference_text(filepath: Path, language: str) -> ReferenceText:
    """Load and clean a single reference text file."""
    raw_bytes = filepath.read_bytes()

    # Detect RTF and convert if needed
    if raw_bytes[:5] == b'{\\rtf':
        raw = _strip_rtf(raw_bytes)
    else:
        # Try UTF-8, fall back to latin-1
        try:
            raw = raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raw = raw_bytes.decode('latin-1')

    text = clean_reference_text(raw, language=language)
    tokens = text.split()

    return ReferenceText(
        language=language,
        name=filepath.stem,
        text=text,
        tokens=tokens,
        token_count=len(tokens),
        filepath=filepath,
    )


def load_reference_corpus(
    data_dir: Optional[str] = None,
    languages: Optional[List[str]] = None,
    min_tokens: int = 100,
    verbose: bool = True,
) -> ReferenceCorpus:
    """
    Load all reference texts, optionally filtered by language.

    Parameters:
        data_dir:   Override default data/reference/ path
        languages:  Only load these languages (None = all available)
        min_tokens: Skip texts with fewer tokens than this
        verbose:    Print loading summary
    """
    discovered = discover_reference_texts(data_dir)
    if not discovered:
        ref_dir = data_dir or _default_reference_dir()
        raise FileNotFoundError(
            f"No reference texts found in {ref_dir}/\n"
            f"Expected layout: {ref_dir}/<language>/<text>.txt"
        )

    corpus = ReferenceCorpus()

    for lang, paths in sorted(discovered.items()):
        if languages is not None and lang not in languages:
            continue

        loaded: List[ReferenceText] = []
        for path in paths:
            try:
                ref = load_reference_text(path, lang)
            except Exception as e:
                if verbose:
                    print(f"  WARNING: failed to load {path}: {e}")
                continue

            if ref.token_count < min_tokens:
                if verbose:
                    print(f"  Skipping {ref.name} ({ref.token_count} tokens < "
                          f"{min_tokens} minimum)")
                continue

            loaded.append(ref)
            if verbose:
                print(f"  Loaded {lang}/{ref.name}: "
                      f"{ref.token_count:,} tokens")

        if loaded:
            corpus.texts[lang] = loaded

    return corpus


# ---------------------------------------------------------------------------
# Bridge to fingerprint.py
# ---------------------------------------------------------------------------

def get_reference_text(
    language: str,
    n_words: int = 500,
    seed: int = 42,
    corpus: Optional[ReferenceCorpus] = None,
) -> str:
    """
    Get reference text for a language, preferring real corpus over synthetic.

    If a ReferenceCorpus is provided and has texts for this language, returns
    a random contiguous window of n_words from the real corpus. Falls back to
    generate_reference_text() from ciphers.py if no real corpus is available.
    """
    if corpus is not None:
        combined = corpus.get_combined_tokens(language)
        if len(combined) >= n_words:
            rng = random.Random(seed)
            max_start = len(combined) - n_words
            start = rng.randint(0, max_start)
            return ' '.join(combined[start:start + n_words])

    # Fall back to synthetic
    from voynich.core.ciphers import generate_reference_text as _synth
    return _synth(language, n_words=n_words, seed=seed)


# ---------------------------------------------------------------------------
# Syllable-Level Reference Statistics
# ---------------------------------------------------------------------------

def get_reference_syllable_stats(
    language: str,
    corpus: Optional[ReferenceCorpus] = None,
    n_words: int = 5000,
    seed: int = 42,
) -> Dict:
    """
    Compute syllable-level statistics for a reference language.

    Returns dict with:
        syllable_lengths: list of syllable counts per word
        char_lengths: list of character counts per word
        syllable_bigrams: (matrix, alphabet) for syllable-level bigrams
        char_bigrams: (matrix, alphabet) for character-level bigrams
        positional_entropy_char: H(char|position=k)
        positional_entropy_syl: H(syllable|position=k)
    """
    from voynich.core.stats import (syllabify_latin, syllabify_latin_text,
                                    bigram_transition_matrix, word_positional_entropy,
                                    first_order_entropy)
    from collections import Counter

    text = get_reference_text(language, n_words=n_words, seed=seed, corpus=corpus)
    tokens = text.split()

    # Character-level stats
    char_lengths = [len(w) for w in tokens]
    char_mat, char_alph = bigram_transition_matrix(text)
    pos_ent_char = word_positional_entropy(tokens)

    # Syllable-level stats
    syllabified = syllabify_latin_text(text)
    syl_lengths = [len(s) for s in syllabified]

    # Build syllable bigram matrix
    all_syllables: List[str] = []
    for syls in syllabified:
        all_syllables.extend(syls)

    syl_text = ' '.join(all_syllables)
    syl_alph = sorted(set(all_syllables))

    # Build syllable transition matrix directly
    import numpy as np
    syl_to_idx = {s: i for i, s in enumerate(syl_alph)}
    n_syls = len(syl_alph)
    syl_counts = np.zeros((n_syls, n_syls), dtype=float)

    for word_syls in syllabified:
        for k in range(len(word_syls) - 1):
            s1, s2 = word_syls[k], word_syls[k + 1]
            if s1 in syl_to_idx and s2 in syl_to_idx:
                syl_counts[syl_to_idx[s1]][syl_to_idx[s2]] += 1

    row_sums = syl_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    syl_mat = syl_counts / row_sums

    # Syllable positional entropy: H(syllable at position k in word)
    import math
    from collections import defaultdict
    pos_syls: Dict[int, List[str]] = defaultdict(list)
    for word_syls in syllabified:
        for k, s in enumerate(word_syls):
            if k < 10:
                pos_syls[k].append(s)

    pos_ent_syl: Dict[str, float] = {}
    for pos in range(10):
        if pos not in pos_syls or not pos_syls[pos]:
            break
        syls_at_pos = pos_syls[pos]
        counts = Counter(syls_at_pos)
        total = len(syls_at_pos)
        h = -sum((c / total) * math.log2(c / total)
                 for c in counts.values() if c > 0)
        pos_ent_syl[f'pos_{pos}'] = round(h, 4)

    return {
        'syllable_lengths': syl_lengths,
        'char_lengths': char_lengths,
        'syllable_bigrams': (syl_mat, syl_alph),
        'char_bigrams': (char_mat, char_alph),
        'positional_entropy_char': pos_ent_char,
        'positional_entropy_syl': pos_ent_syl,
        'n_syllable_types': n_syls,
        'n_words': len(tokens),
    }


# ---------------------------------------------------------------------------
# Morphological Reference Profiles (Phase 5)
# ---------------------------------------------------------------------------

# Latin declension/conjugation suffix inventories (medical Latin approximation)
LATIN_DECLENSION_SUFFIXES: Dict[str, List[str]] = {
    'noun_1st':      ['a', 'ae', 'am', 'arum', 'is', 'as'],
    'noun_2nd':      ['us', 'i', 'o', 'um', 'orum', 'os'],
    'noun_3rd':      ['is', 'em', 'e', 'ium', 'ibus', 'es'],
    'noun_4th':      ['us', 'ui', 'um', 'uum', 'ibus'],
    'adj_1st2nd':    ['us', 'a', 'um', 'i', 'ae', 'o', 'os', 'as'],
    'adj_3rd':       ['is', 'e', 'ium', 'ibus', 'es'],
    'verb_1st':      ['o', 'as', 'at', 'amus', 'atis', 'ant',
                      'a', 'ate', 'are'],
    'verb_2nd':      ['eo', 'es', 'et', 'emus', 'etis', 'ent',
                      'e', 'ete', 'ere'],
    'verb_3rd':      ['o', 'is', 'it', 'imus', 'itis', 'unt',
                      'e', 'ite', 'ere'],
    'verb_imperative': ['a', 'e', 'i', 'ate', 'ete', 'ite'],
}

LATIN_PARADIGM_PROFILES: Dict[str, Dict] = {
    'noun_declension':  {'mean_forms': 6, 'std_forms': 1.5, 'n_suffix_types': 6},
    'adj_declension':   {'mean_forms': 8, 'std_forms': 2.0, 'n_suffix_types': 8},
    'verb_conjugation': {'mean_forms': 10, 'std_forms': 3.0, 'n_suffix_types': 10},
    'invariable':       {'mean_forms': 1, 'std_forms': 0.5, 'n_suffix_types': 0},
}

# Occitan (Old Occitan) approximate suffix inventory
OCCITAN_DECLENSION_SUFFIXES: Dict[str, List[str]] = {
    'noun_fem':  ['a', 'as', 'e', 'es'],
    'noun_masc': ['', 's', 'on', 'ons'],
    'adj':       ['', 'a', 's', 'as', 'e', 'es'],
    'verb_ar':   ['i', 'as', 'a', 'am', 'atz', 'an', 'ar', 'at'],
    'verb_er':   ['i', 'es', 'e', 'em', 'etz', 'en', 'er', 'ut'],
    'verb_ir':   ['isc', 'is', 'is', 'im', 'itz', 'isson', 'ir', 'it'],
}

OCCITAN_PARADIGM_PROFILES: Dict[str, Dict] = {
    'noun_declension':  {'mean_forms': 4, 'std_forms': 1.0, 'n_suffix_types': 4},
    'adj_declension':   {'mean_forms': 6, 'std_forms': 1.5, 'n_suffix_types': 6},
    'verb_conjugation': {'mean_forms': 8, 'std_forms': 2.5, 'n_suffix_types': 8},
    'invariable':       {'mean_forms': 1, 'std_forms': 0.5, 'n_suffix_types': 0},
}

# Top-20 expected medical Latin stems (from Circa Instans / De Viribus Herbarum)
LATIN_MEDICAL_VOCABULARY: List[Tuple[str, str, str]] = [
    ('herba', 'noun', 'herb/plant'),
    ('aqua', 'noun', 'water'),
    ('oleum', 'noun', 'oil'),
    ('radix', 'noun', 'root'),
    ('folium', 'noun', 'leaf'),
    ('flos', 'noun', 'flower'),
    ('semen', 'noun', 'seed'),
    ('morbus', 'noun', 'disease'),
    ('febris', 'noun', 'fever'),
    ('dolor', 'noun', 'pain'),
    ('sanguis', 'noun', 'blood'),
    ('remedium', 'noun', 'remedy'),
    ('recipe', 'verb', 'take/receive'),
    ('accipe', 'verb', 'accept'),
    ('misce', 'verb', 'mix'),
    ('contere', 'verb', 'grind'),
    ('calida', 'adj', 'hot'),
    ('frigida', 'adj', 'cold'),
    ('sicca', 'adj', 'dry'),
    ('humida', 'adj', 'moist'),
]


def compute_suffix_inventory(
    language: str,
    corpus: Optional[ReferenceCorpus] = None,
    n_words: int = 5000,
) -> Dict:
    """
    Compute suffix inventory and paradigm shape profile for a language.

    Combines embedded declension tables with empirical suffix frequencies
    from the reference corpus.

    Returns dict with: suffix_types, suffix_distribution,
    mean_paradigm_size, paradigm_size_distribution, paradigm_profiles.
    """
    import numpy as np

    if language == 'latin':
        suffix_table = LATIN_DECLENSION_SUFFIXES
        profiles = LATIN_PARADIGM_PROFILES
    elif language == 'occitan':
        suffix_table = OCCITAN_DECLENSION_SUFFIXES
        profiles = OCCITAN_PARADIGM_PROFILES
    else:
        return {
            'suffix_types': [], 'suffix_distribution': {},
            'mean_paradigm_size': 0,
            'paradigm_size_distribution': {},
            'paradigm_profiles': {},
        }

    all_suffixes: set = set()
    for paradigm_suffixes in suffix_table.values():
        all_suffixes.update(paradigm_suffixes)
    all_suffixes.discard('')  # Remove empty string

    # Empirical suffix distribution from real corpus
    text = get_reference_text(language, n_words=n_words, corpus=corpus)
    tokens = text.split()
    ending_counts: Counter = Counter()
    for token in tokens:
        for end_len in range(1, min(5, len(token))):
            ending = token[-end_len:]
            if ending in all_suffixes:
                ending_counts[ending] += 1

    paradigm_sizes = [p['mean_forms'] for p in profiles.values()]

    return {
        'suffix_types': sorted(all_suffixes),
        'suffix_distribution': dict(ending_counts.most_common()),
        'mean_paradigm_size': float(np.mean(paradigm_sizes)),
        'paradigm_size_distribution': {
            k: v['mean_forms'] for k, v in profiles.items()
        },
        'paradigm_profiles': profiles,
    }


def get_paradigm_shape_profile(language: str) -> Dict[str, Dict]:
    """
    Return the expected paradigm shape profile for a language.

    Returns dict mapping paradigm_type -> {mean_forms, std_forms, n_suffix_types}.
    """
    if language == 'latin':
        return LATIN_PARADIGM_PROFILES
    elif language == 'occitan':
        return OCCITAN_PARADIGM_PROFILES
    return {}


# ---------------------------------------------------------------------------
# Plant Name Declension Lookup (Phase 6)
# ---------------------------------------------------------------------------

def infer_declension(nominative: str) -> str:
    """
    Infer Latin declension class from nominative singular form.

    Most medieval plant names are:
    - 1st declension (feminine -a): rosa, herba, viola, salvia, malva
    - 2nd declension (neuter -um / masc -us): absinthium, rosmarinus
    - 3rd declension: papaver, radix, cannabis, caulis

    Returns one of: noun_1st, noun_2nd, noun_3rd, noun_4th.
    """
    word = nominative.lower().strip()
    if word.endswith('um') or word.endswith('us'):
        return 'noun_2nd'
    elif word.endswith('a'):
        return 'noun_1st'
    elif word.endswith('is') or word.endswith('x') or word.endswith('en'):
        return 'noun_3rd'
    elif word.endswith('o'):
        return 'noun_3rd'
    elif word.endswith('er'):
        return 'noun_3rd'
    return 'noun_3rd'


def expected_paradigm_shape(declension: str) -> Tuple[int, int]:
    """
    Return expected (n_prefix_types, n_suffix_types) for a declension.

    Based on LATIN_DECLENSION_SUFFIXES counts:
    - noun_1st: (2, 6)  -- 0-2 Voynich prefix types, 6 case endings
    - noun_2nd: (2, 6)
    - noun_3rd: (2, 6)
    - noun_4th: (2, 5)

    The prefix count reflects Voynich determiners/prepositions rather
    than Latin prefixes proper.
    """
    suffix_counts = {
        'noun_1st': 6,
        'noun_2nd': 6,
        'noun_3rd': 6,
        'noun_4th': 5,
    }
    n_suf = suffix_counts.get(declension, 6)
    return (2, n_suf)


def extract_latin_stem(nominative: str, declension: str) -> str:
    """
    Extract the stem from a Latin nominative singular.

    Rules by declension:
    - noun_1st: drop -a (rosa -> ros, viola -> viol)
    - noun_2nd: drop -um or -us (absinthium -> absinthi, rosmarinus -> rosmarin)
    - noun_3rd: drop -is/-x/-en/-o/-er or return unchanged
    - noun_4th: drop -us (quercus -> querc)
    """
    word = nominative.lower().strip()
    if declension == 'noun_1st' and word.endswith('a'):
        return word[:-1]
    elif declension == 'noun_2nd':
        if word.endswith('um'):
            return word[:-2]
        elif word.endswith('us'):
            return word[:-2]
    elif declension == 'noun_3rd':
        if word.endswith('is'):
            return word[:-2]
        elif word.endswith('x'):
            return word[:-1]
        elif word.endswith('en'):
            return word[:-2]
        elif word.endswith('o'):
            return word[:-1]
        elif word.endswith('er'):
            return word
    elif declension == 'noun_4th' and word.endswith('us'):
        return word[:-2]
    return word
