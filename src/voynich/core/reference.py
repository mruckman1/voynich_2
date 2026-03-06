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
from typing import Any, Dict, List, Optional, Tuple

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

# Italian (Old/Middle Italian) suffix inventory — shares many Latin endings
ITALIAN_DECLENSION_SUFFIXES: Dict[str, List[str]] = {
    'noun_fem':   ['a', 'e', 'zione', 'zioni', 'ezza', 'ezze', 'ura', 'ure'],
    'noun_masc':  ['o', 'i', 'mento', 'menti', 'ore', 'ori', 'one', 'oni'],
    'adj':        ['o', 'a', 'i', 'e', 'oso', 'osa', 'osi', 'ose',
                   'ale', 'ali', 'ile', 'ili'],
    'verb_are':   ['o', 'i', 'a', 'iamo', 'ate', 'ano', 'are', 'ato',
                   'ata', 'ando', 'ava', 'avi', 'avano'],
    'verb_ere':   ['o', 'i', 'e', 'iamo', 'ete', 'ono', 'ere', 'uto',
                   'uta', 'endo', 'eva', 'evi', 'evano'],
    'verb_ire':   ['o', 'i', 'e', 'iamo', 'ite', 'ono', 'ire', 'ito',
                   'ita', 'endo', 'iva', 'ivi', 'ivano'],
}

# German (Middle High German / Early New High German) suffix inventory
GERMAN_DECLENSION_SUFFIXES: Dict[str, List[str]] = {
    'noun_strong': ['', 'es', 'em', 'en', 'er'],
    'noun_weak':   ['e', 'en', 'ens'],
    'noun_deriv':  ['ung', 'heit', 'keit', 'schaft', 'nis', 'sal', 'tum'],
    'adj':         ['', 'e', 'er', 'es', 'em', 'en',
                    'lich', 'isch', 'ig', 'bar', 'sam', 'haft'],
    'verb':        ['e', 'st', 't', 'en', 'et', 'te', 'ten',
                    'est', 'tet'],
    'verb_inf':    ['en', 'eln', 'ern'],
    'participle':  ['end', 'ent', 'ung'],
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

# Latin pharmaceutical imperative verbs — Phase 7.5 verb identification
# Frequency ranking based on Circa Instans / De Viribus Herbarum recipe structure
LATIN_PHARMACEUTICAL_IMPERATIVES: List[Dict[str, Any]] = [
    {'verb': 'recipe',   'meaning': 'take',    'frequency_rank': 1,
     'stem': 'recip',    'conjugation': 'verb_3rd', 'n_chars': 6,
     'imperative_forms': ['recipe', 'recipite'],
     'typical_objects': ['plant_names', 'plant_parts', 'preparations']},
    {'verb': 'accipe',   'meaning': 'accept',  'frequency_rank': 2,
     'stem': 'accip',    'conjugation': 'verb_3rd', 'n_chars': 6,
     'imperative_forms': ['accipe', 'accipite'],
     'typical_objects': ['plant_names', 'plant_parts', 'preparations']},
    {'verb': 'misce',    'meaning': 'mix',     'frequency_rank': 3,
     'stem': 'misc',     'conjugation': 'verb_2nd', 'n_chars': 5,
     'imperative_forms': ['misce', 'miscete'],
     'typical_objects': ['preparations', 'plant_parts']},
    {'verb': 'contere',  'meaning': 'grind',   'frequency_rank': 4,
     'stem': 'conter',   'conjugation': 'verb_3rd', 'n_chars': 7,
     'imperative_forms': ['contere', 'conterite'],
     'typical_objects': ['plant_parts', 'plant_names']},
    {'verb': 'coque',    'meaning': 'cook',    'frequency_rank': 5,
     'stem': 'coqu',     'conjugation': 'verb_3rd', 'n_chars': 5,
     'imperative_forms': ['coque', 'coquite'],
     'typical_objects': ['preparations']},
    {'verb': 'distilla', 'meaning': 'distil',  'frequency_rank': 6,
     'stem': 'distill',  'conjugation': 'verb_1st', 'n_chars': 8,
     'imperative_forms': ['distilla', 'distillate'],
     'typical_objects': ['preparations']},
    {'verb': 'pone',     'meaning': 'place',   'frequency_rank': 7,
     'stem': 'pon',      'conjugation': 'verb_3rd', 'n_chars': 4,
     'imperative_forms': ['pone', 'ponite'],
     'typical_objects': ['preparations', 'body_parts']},
    {'verb': 'applica',  'meaning': 'apply',   'frequency_rank': 8,
     'stem': 'applic',   'conjugation': 'verb_1st', 'n_chars': 7,
     'imperative_forms': ['applica', 'applicate'],
     'typical_objects': ['preparations', 'body_parts']},
    {'verb': 'adde',     'meaning': 'add',     'frequency_rank': 9,
     'stem': 'add',      'conjugation': 'verb_3rd', 'n_chars': 4,
     'imperative_forms': ['adde', 'addite'],
     'typical_objects': ['plant_names', 'plant_parts', 'preparations']},
    {'verb': 'cola',     'meaning': 'strain',  'frequency_rank': 10,
     'stem': 'col',      'conjugation': 'verb_1st', 'n_chars': 4,
     'imperative_forms': ['cola', 'colate'],
     'typical_objects': ['preparations']},
]

# Latin pharmaceutical noun semantic domains — Phase 7.5 subcluster matching
LATIN_PHARMACEUTICAL_DOMAINS: Dict[str, List[Tuple[str, str, int]]] = {
    'plant_names': [
        ('rosa', 'rose', 1), ('viola', 'violet', 2), ('salvia', 'sage', 3),
        ('malva', 'mallow', 4), ('absinthium', 'wormwood', 5),
        ('cannabis', 'hemp', 6), ('papaver', 'poppy', 7),
        ('rosmarinus', 'rosemary', 8), ('mentha', 'mint', 9),
        ('chamomilla', 'chamomile', 10), ('artemisia', 'mugwort', 11),
        ('urtica', 'nettle', 12), ('plantago', 'plantain', 13),
        ('ruta', 'rue', 14), ('verbena', 'vervain', 15),
    ],
    'plant_parts': [
        ('radix', 'root', 1), ('folium', 'leaf', 2), ('flos', 'flower', 3),
        ('semen', 'seed', 4), ('cortex', 'bark', 5), ('fructus', 'fruit', 6),
        ('succus', 'juice', 7), ('herba', 'herb/whole plant', 8),
    ],
    'preparations': [
        ('aqua', 'water', 1), ('oleum', 'oil', 2), ('vinum', 'wine', 3),
        ('mel', 'honey', 4), ('acetum', 'vinegar', 5),
        ('pulvis', 'powder', 6), ('unguentum', 'ointment', 7),
        ('emplastrum', 'plaster', 8), ('sirupus', 'syrup', 9),
    ],
    'body_parts': [
        ('caput', 'head', 1), ('stomachus', 'stomach', 2),
        ('oculus', 'eye', 3), ('dens', 'tooth', 4),
        ('pectus', 'chest', 5), ('venter', 'belly', 6),
        ('pes', 'foot', 7), ('manus', 'hand', 8),
    ],
    'qualities': [
        ('calida', 'hot', 1), ('frigida', 'cold', 2),
        ('sicca', 'dry', 3), ('humida', 'moist', 4),
    ],
}


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


# ---------------------------------------------------------------------------
# Latin Recipe Structure (Phase 7 / Approach 9)
# ---------------------------------------------------------------------------

LATIN_RECIPE_VERBS = [
    'recipe', 'accipe', 'misce', 'contere', 'coque', 'cola',
    'decoque', 'adde', 'tere', 'pone', 'fac', 'bibat',
    'detur', 'fiat', 'ponatur', 'lavetur', 'superponatur',
    'misceantur', 'terantur', 'coquantur', 'coletur',
    'teratur', 'decoquantur', 'iniiciatur', 'illiniatur',
    'instillatus', 'conficiatur', 'tritum',
]

LATIN_RECIPE_CONNECTORS = [
    'et', 'cum', 'in', 'de', 'ad', 'per', 'vel', 'sive',
    'aut', 'super', 'contra', 'idest', 'ex', 'ab', 'pro',
    'inde', 'si', 'ut', 'sic', 'quod', 'sicut',
]

# All known suffix endings, flattened for heuristic stemming (per language)
_ALL_LATIN_SUFFIXES = sorted(set(
    s for group in LATIN_DECLENSION_SUFFIXES.values() for s in group
), key=lambda x: -len(x))

_ALL_ITALIAN_SUFFIXES = sorted(set(
    s for group in ITALIAN_DECLENSION_SUFFIXES.values() for s in group
), key=lambda x: -len(x))

_ALL_GERMAN_SUFFIXES = sorted(set(
    s for group in GERMAN_DECLENSION_SUFFIXES.values() for s in group
), key=lambda x: -len(x))

_ALL_OCCITAN_SUFFIXES = sorted(set(
    s for group in OCCITAN_DECLENSION_SUFFIXES.values() for s in group
), key=lambda x: -len(x))

_SUFFIX_TABLE = {
    'latin': _ALL_LATIN_SUFFIXES,
    'occitan': _ALL_OCCITAN_SUFFIXES,
    'italian': _ALL_ITALIAN_SUFFIXES,
    'german': _ALL_GERMAN_SUFFIXES,
}


def stem_latin_token(token: str) -> str:
    """
    Heuristic Latin stemmer: strip longest matching inflectional suffix.

    Accepts the result only if the remaining stem is >= 3 characters.
    Falls back to the original token otherwise.
    """
    word = token.lower().strip()
    for suffix in _ALL_LATIN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def stem_token(token: str, language: str) -> str:
    """
    Language-aware heuristic stemmer: strip longest matching suffix.

    Uses the suffix inventory for the specified language.
    Falls back to stem_latin_token for unknown languages.
    """
    suffixes = _SUFFIX_TABLE.get(language)
    if suffixes is None:
        return stem_latin_token(token)
    word = token.lower().strip()
    for suffix in suffixes:
        if suffix and word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


@dataclass
class RecipeSegment:
    """One segmented recipe/instruction from a Latin herbal text."""
    entry_name: str
    tokens: List[str]
    n_tokens: int
    word_classes: List[str]


@dataclass
class SlotProfile:
    """Positional slot statistics for a corpus of recipes."""
    n_recipes: int
    mean_recipe_length: float
    max_position_analyzed: int
    position_class_probs: Dict[int, Dict[str, float]]
    slot_entropy_by_position: List[float]
    verb_initial_ratio: float


def label_word_class(token: str) -> str:
    """
    Label a Latin token as 'verb', 'connector', or 'other'.

    Uses the closed-class verb and connector lists. Tokens not matching
    either list are labeled 'other' (nouns, adjectives, quantities, etc.).
    """
    t = token.lower().strip()
    if t in LATIN_RECIPE_VERBS:
        return 'verb'
    if t in LATIN_RECIPE_CONNECTORS:
        return 'connector'
    # Heuristic: common imperative/subjunctive endings
    if t.endswith('atur') or t.endswith('antur') or t.endswith('etur'):
        return 'verb'
    return 'other'


def segment_latin_recipes(
    corpus: 'ReferenceCorpus',
    language: str = 'latin',
    min_tokens: int = 3,
    max_segment: int = 30,
) -> List[RecipeSegment]:
    """
    Segment Latin reference text into recipe/instruction units.

    Since the cleaned reference text is a flat token stream (all structure
    removed by clean_reference_text), segments at recipe-initial markers:
    imperative verbs, 'contra', 'ad' + body part patterns, 'item', 'confert'.

    Each segment runs from one marker to the next.
    """
    tokens = corpus.get_combined_tokens(language)
    if not tokens:
        return []

    # Recipe-initial words that start new segments
    segment_starters = set(LATIN_RECIPE_VERBS) | {
        'contra', 'confert', 'item', 'solvit', 'mundificat',
        'reducit', 'provocat', 'confortat', 'sanat', 'curat',
    }

    # Find segment boundaries
    boundaries = [0]
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        if tok in segment_starters:
            boundaries.append(i)
    boundaries.append(len(tokens))

    # Build segments
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        seg_tokens = tokens[start:end]
        # Cap segment length to avoid giant non-recipe blocks
        if len(seg_tokens) > max_segment:
            seg_tokens = seg_tokens[:max_segment]
        if len(seg_tokens) < min_tokens:
            continue
        classes = [label_word_class(t) for t in seg_tokens]
        segments.append(RecipeSegment(
            entry_name='',
            tokens=seg_tokens,
            n_tokens=len(seg_tokens),
            word_classes=classes,
        ))

    return segments


def compute_slot_profile(
    recipes: List[RecipeSegment],
    max_position: int = 10,
) -> SlotProfile:
    """
    Compute positional word-class statistics across all recipe segments.

    For each position k (0-indexed up to max_position), computes
    P(word_class | position=k) and per-position entropy over classes.
    """
    import math as _math
    if not recipes:
        return SlotProfile(
            n_recipes=0, mean_recipe_length=0.0, max_position_analyzed=0,
            position_class_probs={}, slot_entropy_by_position=[],
            verb_initial_ratio=0.0,
        )

    position_counts: Dict[int, Dict[str, int]] = {}
    for pos in range(max_position):
        position_counts[pos] = Counter()

    n_verb_initial = 0
    for recipe in recipes:
        for pos in range(min(max_position, recipe.n_tokens)):
            position_counts[pos][recipe.word_classes[pos]] += 1
        if recipe.n_tokens > 0 and recipe.word_classes[0] == 'verb':
            n_verb_initial += 1

    # Compute probabilities and entropy per position
    position_probs = {}
    entropies = []
    for pos in range(max_position):
        counts = position_counts[pos]
        total = sum(counts.values())
        if total == 0:
            position_probs[pos] = {}
            entropies.append(0.0)
            continue
        probs = {cls: c / total for cls, c in counts.items()}
        position_probs[pos] = probs
        h = -sum(p * _math.log2(p) for p in probs.values() if p > 0)
        entropies.append(h)

    mean_len = sum(r.n_tokens for r in recipes) / len(recipes)
    verb_init = n_verb_initial / len(recipes) if recipes else 0.0

    return SlotProfile(
        n_recipes=len(recipes),
        mean_recipe_length=mean_len,
        max_position_analyzed=max_position,
        position_class_probs=position_probs,
        slot_entropy_by_position=entropies,
        verb_initial_ratio=verb_init,
    )


# ---------------------------------------------------------------------------
# Latin Phrase Catalog (Phase 8)
# ---------------------------------------------------------------------------

def build_latin_phrase_catalog() -> Dict[str, List[str]]:
    """
    Build a catalog of common Latin pharmaceutical phrases for coherence testing.

    Returns dict mapping category -> list of stemmed 2-3 word phrases.
    Each phrase is a space-separated string of stemmed Latin words that
    commonly co-occur in medieval herbal/medical texts.

    Built from LATIN_RECIPE_VERBS, LATIN_RECIPE_CONNECTORS, and
    LATIN_PHARMACEUTICAL_DOMAINS already defined in this module.
    """
    catalog: Dict[str, List[str]] = {}

    # Recipe openings: imperative verb + object type
    openings = []
    recipe_verbs = ['recip', 'accip', 'misc', 'conter', 'coqu',
                    'distill', 'pon', 'applic', 'add', 'col']
    plant_stems = [stem_latin_token(p[0]) for p in
                   LATIN_PHARMACEUTICAL_DOMAINS.get('plant_names', [])]
    part_stems = [stem_latin_token(p[0]) for p in
                  LATIN_PHARMACEUTICAL_DOMAINS.get('plant_parts', [])]
    for verb in recipe_verbs[:5]:
        for obj in plant_stems[:5]:
            openings.append(f"{verb} {obj}")
        for obj in part_stems[:4]:
            openings.append(f"{verb} {obj}")
    catalog['recipe_openings'] = openings

    # Preparation instructions: verb + cum/in + preparation
    instructions = []
    prep_stems = [stem_latin_token(p[0]) for p in
                  LATIN_PHARMACEUTICAL_DOMAINS.get('preparations', [])]
    for verb in recipe_verbs[:5]:
        for prep in prep_stems[:5]:
            instructions.append(f"{verb} cum {prep}")
            instructions.append(f"{verb} in {prep}")
    catalog['preparation_instructions'] = instructions

    # Application phrases: verb + super/ad + body part
    applications = []
    body_stems = [stem_latin_token(p[0]) for p in
                  LATIN_PHARMACEUTICAL_DOMAINS.get('body_parts', [])]
    for verb in ['pon', 'applic', 'add']:
        for body in body_stems:
            applications.append(f"{verb} super {body}")
            applications.append(f"{verb} ad {body}")
    catalog['application_phrases'] = applications

    # Quality descriptions: est + quality
    qualities = []
    qual_stems = [stem_latin_token(q[0]) for q in
                  LATIN_PHARMACEUTICAL_DOMAINS.get('qualities', [])]
    for q1 in qual_stems:
        qualities.append(f"est {q1}")
        for q2 in qual_stems:
            if q1 != q2:
                qualities.append(f"{q1} et {q2}")
    catalog['quality_descriptions'] = qualities

    return catalog


# ---------------------------------------------------------------------------
# Constructed Script Grid Reference Data  (Phase 10.4)
# ---------------------------------------------------------------------------
# Pre-computed grid statistics for known combinatorial writing systems.
# Each entry captures the structural parameters of its onset × nucleus grid,
# sourced from standard orthographic descriptions.

SCRIPT_GRID_STATS: Dict[str, Dict[str, Any]] = {
    'hangul': {
        'description': 'Korean Hangul (initial × medial jamo)',
        'onset_types': 14,      # ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅅ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ
        'nucleus_types': 10,    # ㅏ ㅑ ㅓ ㅕ ㅗ ㅛ ㅜ ㅠ ㅡ ㅣ
        'occupancy': 0.95,      # nearly all onset-vowel pairs attested
        'r_forward': 0.12,      # R(vowel|consonant) — low, nearly independent
        'r_reverse': 0.14,      # R(consonant|vowel) — low
        'onset_entropy': 3.51,  # log2(14) ≈ 3.81, slightly compressed
        'nucleus_entropy': 3.15,  # log2(10) ≈ 3.32, slightly compressed
    },
    'devanagari': {
        'description': 'Devanagari (consonant × vowel diacritic)',
        'onset_types': 33,      # ka kha ga ... ha
        'nucleus_types': 12,    # a ā i ī u ū ṛ e ai o au (+ virama)
        'occupancy': 0.70,      # many C-V combinations rare in practice
        'r_forward': 0.28,      # R(vowel|consonant) — moderate
        'r_reverse': 0.35,      # R(consonant|vowel) — moderate
        'onset_entropy': 4.50,  # log2(33) ≈ 5.04, skewed by frequency
        'nucleus_entropy': 2.90,  # log2(12) ≈ 3.58, 'a' dominant
    },
    'ethiopic': {
        'description': 'Ethiopic / Ge\'ez (consonant × vowel order)',
        'onset_types': 26,      # base consonant forms
        'nucleus_types': 7,     # 7 vowel orders
        'occupancy': 0.85,      # most C × V cells filled
        'r_forward': 0.18,      # R(order|consonant) — low-moderate
        'r_reverse': 0.22,      # R(consonant|order) — low-moderate
        'onset_entropy': 4.20,  # log2(26) ≈ 4.70
        'nucleus_entropy': 2.60,  # log2(7) ≈ 2.81
    },
    'linear_b': {
        'description': 'Linear B syllabary (C × V, deciphered)',
        'onset_types': 15,      # approximate consonant series
        'nucleus_types': 5,     # a e i o u
        'occupancy': 0.60,      # many gaps in the grid
        'r_forward': 0.35,      # R(vowel|consonant) — moderate
        'r_reverse': 0.40,      # R(consonant|vowel) — moderate
        'onset_entropy': 3.50,  # log2(15) ≈ 3.91
        'nucleus_entropy': 2.10,  # log2(5) ≈ 2.32
    },
}


# ---------------------------------------------------------------------------
# Romance Phonotactic Constraints  (Phase 10.4c CSP)
# ---------------------------------------------------------------------------
# Allowed onset (C/CC) and rime (V/VC) inventories plus legal pairs for
# major Romance languages.  Used to prune the phonotactic constraint
# satisfaction search when mapping grid cells to phonemes/syllables.

ROMANCE_PHONOTACTICS: Dict[str, Dict[str, Any]] = {
    'latin': {
        'onsets': [
            '', 'b', 'c', 'd', 'f', 'g', 'h', 'l', 'm', 'n', 'p', 'qu',
            'r', 's', 't', 'v', 'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr',
            'gl', 'gr', 'pl', 'pr', 'sc', 'sp', 'st', 'str', 'tr',
        ],
        'rimes': [
            'a', 'e', 'i', 'o', 'u', 'ae', 'au', 'oe',
            'am', 'an', 'ar', 'as', 'at',
            'em', 'en', 'er', 'es', 'et',
            'im', 'in', 'ir', 'is', 'it',
            'om', 'on', 'or', 'os',
            'um', 'un', 'ur', 'us', 'ut',
        ],
        'forbidden_onsets': {'dl', 'tl', 'sr', 'nm'},
    },
    'italian': {
        'onsets': [
            '', 'b', 'c', 'ch', 'd', 'f', 'g', 'gh', 'gl', 'gn', 'l',
            'm', 'n', 'p', 'qu', 'r', 's', 'sc', 'sp', 'st', 'str', 't',
            'v', 'z', 'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gr',
            'pl', 'pr', 'tr',
        ],
        'rimes': [
            'a', 'e', 'i', 'o', 'u',
            'an', 'ar', 'al', 'at',
            'en', 'er', 'el', 'et',
            'in', 'ir', 'il', 'it',
            'on', 'or', 'ol',
            'un', 'ur',
        ],
        'forbidden_onsets': {'dl', 'tl', 'sr'},
    },
    'occitan': {
        'onsets': [
            '', 'b', 'c', 'ch', 'd', 'f', 'g', 'gl', 'gn', 'h', 'l',
            'lh', 'm', 'n', 'nh', 'p', 'qu', 'r', 's', 'sc', 'sp', 'st',
            't', 'v', 'z', 'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr',
            'gr', 'pl', 'pr', 'tr',
        ],
        'rimes': [
            'a', 'e', 'i', 'o', 'u',
            'an', 'ar', 'al', 'as', 'at',
            'en', 'er', 'el', 'es', 'et',
            'in', 'ir', 'il', 'is', 'it',
            'on', 'or', 'ol', 'os',
            'un', 'ur', 'us',
        ],
        'forbidden_onsets': {'dl', 'tl', 'sr'},
    },
    'german': {
        'onsets': [
            '', 'b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r',
            's', 'sch', 'sp', 'st', 'str', 't', 'v', 'w', 'z',
            'bl', 'br', 'dr', 'fl', 'fr', 'gl', 'gr', 'kl', 'kn', 'kr',
            'pf', 'pl', 'pr', 'tr', 'zw',
        ],
        'rimes': [
            'a', 'e', 'i', 'o', 'u',
            'an', 'ar', 'al', 'as', 'at',
            'en', 'er', 'el', 'es', 'et',
            'in', 'ir', 'il', 'is', 'it',
            'on', 'or', 'ol',
            'un', 'ur', 'us', 'ut',
            'ach', 'ich', 'uch',
        ],
        'forbidden_onsets': {'dl', 'tl', 'sr'},
    },
}


# ---------------------------------------------------------------------------
# Phoneme Inventories and Syllable Frequency Tools  (Phase 11 CSP)
# ---------------------------------------------------------------------------

PHONEME_INVENTORIES: Dict[str, Dict[str, Any]] = {
    'latin': {
        'consonants': ['b', 'c', 'd', 'f', 'g', 'h', 'l', 'm', 'n',
                        'p', 'r', 's', 't', 'v'],
        'vowels': ['a', 'e', 'i', 'o', 'u'],
        'word_final_legal': {'a', 'e', 'i', 'o', 'u', 'm', 'n', 'r',
                             's', 't', 'x'},
    },
    'occitan': {
        'consonants': ['b', 'c', 'd', 'f', 'g', 'h', 'l', 'm', 'n',
                        'p', 'r', 's', 't', 'v', 'z'],
        'vowels': ['a', 'e', 'i', 'o', 'u'],
        'word_final_legal': {'a', 'e', 'i', 'o', 'u', 'l', 'n', 'r',
                             's', 't', 'z'},
    },
    'italian': {
        'consonants': ['b', 'c', 'd', 'f', 'g', 'l', 'm', 'n', 'p',
                        'r', 's', 't', 'v', 'z'],
        'vowels': ['a', 'e', 'i', 'o', 'u'],
        'word_final_legal': {'a', 'e', 'i', 'o'},
    },
    'german': {
        'consonants': ['b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n',
                        'p', 'r', 's', 't', 'v', 'w', 'z'],
        'vowels': ['a', 'e', 'i', 'o', 'u'],
        'word_final_legal': {'a', 'e', 'i', 'o', 'u', 'b', 'd', 'f',
                             'g', 'k', 'l', 'm', 'n', 'p', 'r', 's',
                             't', 'z'},
    },
}


def get_phoneme_inventory(language: str) -> Dict[str, Any]:
    """Return phoneme inventory for a language, defaulting to Latin."""
    return PHONEME_INVENTORIES.get(language, PHONEME_INVENTORIES['latin'])


def build_cv_syllable_table(language: str) -> List[str]:
    """Build all legal CV syllables for *language*.

    Returns consonant+vowel combinations plus pure-vowel syllables.
    """
    inv = get_phoneme_inventory(language)
    syllables: List[str] = []
    for c in inv['consonants']:
        for v in inv['vowels']:
            syllables.append(c + v)
    # Pure-vowel onsets
    for v in inv['vowels']:
        syllables.append(v)
    return syllables


def build_syllable_frequency_table(
    language: str,
    ref_corpus: Optional['ReferenceCorpus'] = None,
    n_words: int = 10000,
) -> Dict[str, float]:
    """Compute CV syllable frequencies from reference corpus.

    Syllabifies reference text words, maps each syllable to its
    onset+nucleus (CV) pattern, and returns normalised frequencies.
    Falls back to uniform distribution when no corpus is available.
    """
    from voynich.core.stats import syllabify_latin  # avoid circular import

    cv_table = build_cv_syllable_table(language)
    counts: Counter = Counter()

    if ref_corpus is not None:
        tokens = ref_corpus.get_combined_tokens(language)
        if not tokens:
            # Try any available language
            for lang in ref_corpus.languages:
                tokens = ref_corpus.get_combined_tokens(lang)
                if tokens:
                    break
        for word in tokens[:n_words]:
            syls = syllabify_latin(word)
            for syl in syls:
                # Normalise to lowercase and strip to CV skeleton
                syl_lower = syl.lower()
                # Find the best matching CV pattern
                best_match = _match_cv_pattern(syl_lower, cv_table)
                if best_match:
                    counts[best_match] += 1

    # Ensure every legal syllable has at least a small count
    for syl in cv_table:
        if syl not in counts:
            counts[syl] = 1

    total = sum(counts.values())
    return {syl: cnt / total for syl, cnt in counts.items()}


def _match_cv_pattern(syllable: str, cv_table: List[str]) -> Optional[str]:
    """Find the best matching CV pattern for a syllable string.

    Extracts onset (leading consonants) and nucleus (first vowel)
    and returns the corresponding CV entry if it exists.
    """
    vowels = set('aeiou')
    onset = ''
    nucleus = ''
    rest = syllable.lower()

    # Extract onset consonants
    while rest and rest[0] not in vowels:
        onset += rest[0]
        rest = rest[1:]

    # Extract nucleus vowel
    if rest and rest[0] in vowels:
        nucleus = rest[0]

    if not nucleus:
        return None

    candidate = onset + nucleus
    if candidate in cv_table:
        return candidate

    # Try with simplified onset (just last consonant)
    if len(onset) > 1:
        candidate = onset[-1] + nucleus
        if candidate in cv_table:
            return candidate

    # Try pure vowel
    if nucleus in cv_table:
        return nucleus

    return None


# ---------------------------------------------------------------------------
# Phase 11.5: Extended syllable inventories (CVC, CCV, inherent vowel)
# ---------------------------------------------------------------------------

# Closed syllables (CVC) for each language, drawn from frequent medieval
# pharmaceutical patterns.
CVC_EXTENSIONS: Dict[str, List[str]] = {
    'latin': [
        'al', 'ar', 'an', 'as', 'at',
        'el', 'er', 'en', 'es', 'et',
        'il', 'ir', 'in', 'is', 'it',
        'ol', 'or', 'on', 'os',
        'ul', 'ur', 'un', 'us', 'ut',
        'cal', 'car', 'can', 'cas', 'cat',
        'tem', 'ter', 'ten', 'tes',
        'men', 'mer', 'mel',
        'nal', 'nar', 'nan',
        'sal', 'sar', 'san', 'sol', 'son',
        'pal', 'par', 'pan', 'pos', 'pon',
        'ram', 'ren', 'rin', 'ron',
        'mis', 'con', 'dis', 'ap', 'ad',
        'bis', 'bit', 'bin',
        'dis', 'dit', 'din',
        'lat', 'las', 'lar',
        'vat', 'vas', 'val',
        'til', 'col',
    ],
    'occitan': [
        'al', 'ar', 'an', 'as', 'at',
        'el', 'er', 'en', 'es', 'et',
        'il', 'ir', 'in', 'is', 'it',
        'ol', 'or', 'on', 'ul', 'ur',
        'ral', 'ran', 'ron',
        'cal', 'can', 'ten', 'men', 'nal', 'sol',
        'mis', 'con', 'dis',
    ],
    'italian': [
        'al', 'ar', 'an', 'el', 'er', 'en',
        'il', 'ir', 'in', 'ol', 'or', 'on',
        'ul', 'ur', 'un',
        'cal', 'car', 'can', 'ten', 'men', 'sol', 'sal', 'par',
        'con', 'dis',
    ],
    'german': [
        'al', 'an', 'ar', 'el', 'en', 'er',
        'in', 'un', 'ul',
        'ach', 'ich', 'uch',
        'kal', 'ken', 'hal', 'ren', 'sen',
        'con', 'dis',
    ],
}

# Complex onsets (CCV) — consonant cluster + vowel.
CCV_EXTENSIONS: Dict[str, List[str]] = {
    'latin': [
        'bra', 'bre', 'bri', 'bro', 'bru',
        'cla', 'cle', 'cli', 'clo', 'clu',
        'cra', 'cre', 'cri', 'cro', 'cru',
        'dra', 'dre', 'dri', 'dro', 'dru',
        'fla', 'fle', 'fli', 'flo', 'flu',
        'fra', 'fre', 'fri', 'fro', 'fru',
        'gla', 'gle', 'gli', 'glo', 'glu',
        'gra', 'gre', 'gri', 'gro', 'gru',
        'pla', 'ple', 'pli', 'plo', 'plu',
        'pra', 'pre', 'pri', 'pro', 'pru',
        'tra', 'tre', 'tri', 'tro', 'tru',
        'sta', 'ste', 'sti', 'sto', 'stu',
        'spa', 'spe', 'spi', 'spo', 'spu',
        'sca', 'sce', 'sci', 'sco', 'scu',
    ],
    'occitan': [
        'bra', 'bre', 'cla', 'cle', 'cra', 'cre',
        'dra', 'dre', 'fla', 'fle', 'fra', 'fre',
        'gla', 'gle', 'gra', 'gre', 'pla', 'ple',
        'pra', 'pre', 'tra', 'tre', 'sta', 'ste', 'spa',
    ],
    'italian': [
        'bra', 'bre', 'cla', 'cle', 'cra', 'cre',
        'dra', 'dre', 'fla', 'fle', 'fra', 'fre',
        'gla', 'gle', 'gra', 'gre', 'pla', 'ple',
        'pra', 'pre', 'tra', 'tre', 'sta', 'ste', 'spa', 'sca',
    ],
    'german': [
        'bra', 'bre', 'dra', 'dre', 'fra', 'fre',
        'gra', 'gre', 'kla', 'kle', 'kra', 'kre',
        'pla', 'ple', 'pra', 'pre', 'tra', 'tre',
        'sta', 'ste', 'scha', 'sche',
    ],
}

# Candidate inherent vowels (abugida model: onset-only cell carries this vowel)
INHERENT_VOWEL_CANDIDATES: List[str] = ['a', 'e', 'i']

# Phonological syllabification of the 10 Latin pharmaceutical imperatives.
# CVC syllables like 'mis', 'con', 'dis', 'ap', 'ad' require CVC_EXTENSIONS
# to be in the inventory (relaxation_level >= 2).
LATIN_IMPERATIVE_SYLLABIFICATIONS: Dict[str, List[str]] = {
    'recipe':   ['re', 'ci', 'pe'],
    'accipe':   ['ac', 'ci', 'pe'],
    'misce':    ['mis', 'ce'],
    'contere':  ['con', 'te', 're'],
    'coque':    ['co', 'que'],
    'distilla': ['dis', 'til', 'la'],
    'pone':     ['po', 'ne'],
    'applica':  ['ap', 'pli', 'ca'],
    'adde':     ['ad', 'de'],
    'cola':     ['co', 'la'],
}


def build_cvc_syllable_table(
    language: str,
    relaxation_level: int = 0,
    inherent_vowel: Optional[str] = None,
) -> List[str]:
    """Build an expanded syllable inventory for the given relaxation level.

    Level 0: CV only (same as :func:`build_cv_syllable_table`).
    Level 1: CV + consonant-only singletons with *inherent_vowel* appended.
    Level 2: CV + top-25 CVC entries from :data:`CVC_EXTENSIONS`.
    Level 3: CV + full :data:`CVC_EXTENSIONS`.
    Level 4: CV + CVC + top-25 CCV entries from :data:`CCV_EXTENSIONS`.
    Level 5: CV + full CVC + full CCV (~190 syllables for Latin).
    """
    base = build_cv_syllable_table(language)
    if relaxation_level == 0:
        return base

    result = list(base)
    seen = set(result)

    inv = get_phoneme_inventory(language)

    if relaxation_level >= 1 and inherent_vowel:
        # Add C+inherent_vowel for each consonant not already in base
        for c in inv['consonants']:
            syl = c + inherent_vowel
            if syl not in seen:
                result.append(syl)
                seen.add(syl)

    cvc = CVC_EXTENSIONS.get(language, CVC_EXTENSIONS.get('latin', []))
    ccv = CCV_EXTENSIONS.get(language, CCV_EXTENSIONS.get('latin', []))

    if relaxation_level == 2:
        additions = cvc[:25]
    elif relaxation_level == 3:
        additions = cvc
    elif relaxation_level == 4:
        additions = cvc + ccv[:25]
    elif relaxation_level >= 5:
        additions = cvc + ccv
    else:
        additions = []

    for syl in additions:
        if syl not in seen:
            result.append(syl)
            seen.add(syl)

    return result


# ---------------------------------------------------------------------------
# Romance Phonological Processes  (Phase 13)
# ---------------------------------------------------------------------------
# Catalogue of well-attested phonological rules in Latin and Romance languages,
# used to assess the linguistic plausibility of extracted reading rules.
# Format: produced_phoneme -> corrected_phoneme -> {process_name, languages, naturality}

ROMANCE_PHONOLOGICAL_PROCESSES: Dict[str, Dict[str, Dict]] = {
    # Final devoicing: voiced stops → voiceless word-finally
    'b': {'p': {'process': 'final_devoicing', 'languages': ['occitan', 'german', 'italian'],
                'naturality': 'high', 'description': 'Voiced stop /b/ devoiced to /p/ word-finally'}},
    'd': {'t': {'process': 'final_devoicing', 'languages': ['occitan', 'german', 'latin'],
                'naturality': 'high', 'description': 'Voiced stop /d/ devoiced to /t/ word-finally'}},
    'g': {'c': {'process': 'final_devoicing', 'languages': ['occitan', 'german'],
                'naturality': 'high', 'description': 'Voiced stop /g/ devoiced to /c/ word-finally'}},
    # Intervocalic voicing: voiceless → voiced between vowels
    'p': {'b': {'process': 'intervocalic_voicing', 'languages': ['occitan', 'italian', 'latin'],
                'naturality': 'high', 'description': '/p/ voiced to /b/ between vowels'},
          'v': {'process': 'intervocalic_spirantization', 'languages': ['occitan', 'italian'],
                'naturality': 'moderate', 'description': '/p/ spirantized to /v/ between vowels'}},
    't': {'d': {'process': 'intervocalic_voicing', 'languages': ['occitan', 'italian', 'latin'],
                'naturality': 'high', 'description': '/t/ voiced to /d/ between vowels'}},
    'c': {'g': {'process': 'intervocalic_voicing', 'languages': ['occitan', 'italian'],
                'naturality': 'high', 'description': '/c/ voiced to /g/ between vowels'},
          's': {'process': 'palatalization', 'languages': ['italian', 'occitan', 'latin'],
                'naturality': 'high', 'description': '/c/ → /s/ before front vowels (palatalization)'}},
    # Nasal assimilation
    'n': {'m': {'process': 'nasal_assimilation', 'languages': ['latin', 'italian', 'occitan'],
                'naturality': 'high', 'description': '/n/ assimilates to /m/ before labials (in/im, con/com)'}},
    'm': {'n': {'process': 'nasal_assimilation', 'languages': ['latin', 'italian'],
                'naturality': 'moderate', 'description': '/m/ assimilates to /n/ before dentals'}},
    # Schwa/vowel reduction word-finally
    'a': {'e': {'process': 'vowel_reduction', 'languages': ['occitan', 'latin'],
                'naturality': 'moderate', 'description': 'Final /a/ reduces to /e/ in unstressed position'},
          '': {'process': 'vowel_deletion', 'languages': ['italian', 'occitan'],
               'naturality': 'moderate', 'description': 'Final /a/ deleted word-finally in some dialects'}},
    'e': {'a': {'process': 'vowel_shift', 'languages': ['latin', 'occitan'],
                'naturality': 'moderate', 'description': 'Final /e/ shifts to /a/ in some paradigms'},
          '': {'process': 'schwa_deletion', 'languages': ['latin', 'occitan', 'italian'],
               'naturality': 'high', 'description': 'Unstressed /e/ deleted (analogous to Devanagari schwa deletion)'}},
    # Liquid alternation
    'l': {'r': {'process': 'liquid_alternation', 'languages': ['latin', 'occitan'],
                'naturality': 'moderate', 'description': '/l/→/r/ in certain consonant clusters (rhotacism)'}},
    'r': {'l': {'process': 'liquid_alternation', 'languages': ['latin', 'italian'],
                'naturality': 'moderate', 'description': '/r/→/l/ alternation in liquid clusters'}},
    # Sibilant variation
    's': {'z': {'process': 'sibilant_voicing', 'languages': ['italian', 'occitan'],
                'naturality': 'moderate', 'description': '/s/ → /z/ intervocalically in Italian/Occitan'},
          'c': {'process': 'sibilant_palatalization', 'languages': ['latin', 'italian'],
                'naturality': 'moderate', 'description': '/s/ → /c/ before front vowels in some Latin dialects'}},
    # Vowel harmony / back-front alternation
    'o': {'u': {'process': 'vowel_raising', 'languages': ['latin', 'occitan'],
                'naturality': 'moderate', 'description': '/o/ raises to /u/ in certain environments'},
          'a': {'process': 'vowel_lowering', 'languages': ['occitan', 'italian'],
                'naturality': 'moderate', 'description': '/o/ lowers to /a/ in pretonic position'}},
    'i': {'e': {'process': 'vowel_lowering', 'languages': ['latin', 'occitan'],
                'naturality': 'moderate', 'description': '/i/ lowers to /e/ in unstressed syllables'}},
    # Fricative variation
    'f': {'v': {'process': 'fricative_voicing', 'languages': ['italian', 'occitan'],
                'naturality': 'moderate', 'description': '/f/ → /v/ word-initially in some dialects'},
          'h': {'process': 'fricative_weakening', 'languages': ['italian'],
                'naturality': 'moderate', 'description': '/f/ weakens to /h/ in some Italian dialects'}},
    'v': {'b': {'process': 'betacism', 'languages': ['latin', 'occitan'],
                'naturality': 'high', 'description': '/v/→/b/ merger (Latin betacism, Occitan/Spanish)'}},
}

# Medieval Latin spelling variants for dictionary expansion (Step 13.6)
MEDIEVAL_LATIN_VARIANTS: Dict[str, str] = {
    'ae': 'e',    # Classical ae → medieval e
    'oe': 'e',    # Classical oe → medieval e
    'ph': 'f',    # Greek phi → medieval f
    'th': 't',    # Greek theta → medieval t
    'ch': 'c',    # Greek chi → medieval c
    'y': 'i',     # Greek upsilon → medieval i
    'hy': 'i',    # Hyper-correction
    'ci': 'ti',   # ti → ci before vowel (medieval spelling confusion)
    'ti': 'ci',   # ci → ti
    'qu': 'c',    # qu before non-u vowels
    'x': 'cs',    # x → cs
    'z': 's',     # z → s in some medieval texts
}


def expand_latin_word_set(word_set: set) -> set:
    """Expand a Latin word set with medieval variant spellings.

    For dictionary expansion testing in Phase 13.6.
    Adds variant-spelled forms of all words in the input set.
    """
    expanded = set(word_set)
    for word in list(word_set):
        for classical, medieval in MEDIEVAL_LATIN_VARIANTS.items():
            if classical in word:
                expanded.add(word.replace(classical, medieval))
        # Also add common abbreviation expansions
        if word.endswith('us'):
            expanded.add(word[:-2] + 'u')
        if word.endswith('um'):
            expanded.add(word[:-2] + 'u')
        if word.endswith('ae'):
            expanded.add(word[:-2] + 'e')
    return expanded


# ---------------------------------------------------------------------------
# EVA Visual Component Table  (Phase 12)
# ---------------------------------------------------------------------------
# Derived from EVA_STROKE_TABLE in voynich.analysis.strokes.
# Maps each EVA glyph to its first and last stroke primitives,
# which determine the (onset_class, nucleus_class) cell assignment.
# This static copy avoids a runtime import of strokes.py in the CSP pipeline.
#
# first_stroke → FIRST_STROKE_TO_ONSET mapping:
#   loop              → loop
#   open_curve        → open_curve+sigmoid
#   sigmoid           → open_curve+sigmoid
#   ascender          → ascender+vertical
#   crossbar          → crossbar
#   connector         → connector
#   vertical          → ascender+vertical
#
# last_stroke → LAST_STROKE_TO_NUCLEUS mapping:
#   tail              → loop+sigmoid+tail
#   sigmoid           → loop+sigmoid+tail
#   loop              → loop+sigmoid+tail
#   vertical          → vertical
#   plume             → ascender+crossbar+plume
#   crossbar          → ascender+crossbar+plume
#   connector         → connector+open_curve
#   descender         → descender
#   hook              → hook

EVA_VISUAL_COMPONENTS: Dict[str, Dict[str, str]] = {
    # bench / loop-onset glyphs
    'o':     {'first_stroke': 'loop',       'last_stroke': 'loop',       'glyph_class': 'bench'},
    'a':     {'first_stroke': 'loop',       'last_stroke': 'tail',       'glyph_class': 'bench'},
    'e':     {'first_stroke': 'loop',       'last_stroke': 'loop',       'glyph_class': 'bench'},
    'r':     {'first_stroke': 'loop',       'last_stroke': 'sigmoid',    'glyph_class': 'bench'},
    'l':     {'first_stroke': 'loop',       'last_stroke': 'vertical',   'glyph_class': 'bench'},
    # ligatures with loop onset
    'al':    {'first_stroke': 'loop',       'last_stroke': 'vertical',   'glyph_class': 'bench'},
    'ol':    {'first_stroke': 'loop',       'last_stroke': 'vertical',   'glyph_class': 'bench'},
    'ar':    {'first_stroke': 'loop',       'last_stroke': 'sigmoid',    'glyph_class': 'bench'},
    'or':    {'first_stroke': 'loop',       'last_stroke': 'sigmoid',    'glyph_class': 'bench'},
    'ey':    {'first_stroke': 'loop',       'last_stroke': 'descender',  'glyph_class': 'bench'},
    'aiin':  {'first_stroke': 'loop',       'last_stroke': 'hook',       'glyph_class': 'bench'},
    'aiiin': {'first_stroke': 'loop',       'last_stroke': 'hook',       'glyph_class': 'bench'},
    # gallows / ascender-onset glyphs
    'k':     {'first_stroke': 'ascender',   'last_stroke': 'ascender',   'glyph_class': 'gallows'},
    't':     {'first_stroke': 'ascender',   'last_stroke': 'crossbar',   'glyph_class': 'gallows'},
    'p':     {'first_stroke': 'ascender',   'last_stroke': 'plume',      'glyph_class': 'gallows'},
    'f':     {'first_stroke': 'ascender',   'last_stroke': 'crossbar',   'glyph_class': 'gallows'},
    'g':     {'first_stroke': 'vertical',   'last_stroke': 'ascender',   'glyph_class': 'minim'},
    # minim / vertical-onset glyphs
    'i':     {'first_stroke': 'vertical',   'last_stroke': 'vertical',   'glyph_class': 'minim'},
    'm':     {'first_stroke': 'vertical',   'last_stroke': 'vertical',   'glyph_class': 'minim'},
    'd':     {'first_stroke': 'vertical',   'last_stroke': 'vertical',   'glyph_class': 'minim'},
    'n':     {'first_stroke': 'vertical',   'last_stroke': 'hook',       'glyph_class': 'minim'},
    'iin':   {'first_stroke': 'vertical',   'last_stroke': 'hook',       'glyph_class': 'minim'},
    'iiin':  {'first_stroke': 'vertical',   'last_stroke': 'hook',       'glyph_class': 'minim'},
    # suffix / descender glyphs
    'y':     {'first_stroke': 'ascender',   'last_stroke': 'descender',  'glyph_class': 'suffix'},
    'dy':    {'first_stroke': 'vertical',   'last_stroke': 'descender',  'glyph_class': 'suffix'},
    'q':     {'first_stroke': 'ascender',   'last_stroke': 'descender',  'glyph_class': 'suffix'},
    # compound glyphs
    'qo':    {'first_stroke': 'ascender',   'last_stroke': 'loop',       'glyph_class': 'compound'},
    'qot':   {'first_stroke': 'ascender',   'last_stroke': 'crossbar',   'glyph_class': 'compound'},
    'qok':   {'first_stroke': 'ascender',   'last_stroke': 'ascender',   'glyph_class': 'compound'},
    # c/h family (open_curve / sigmoid onset)
    'c':     {'first_stroke': 'open_curve', 'last_stroke': 'open_curve', 'glyph_class': 'bench'},
    'h':     {'first_stroke': 'open_curve', 'last_stroke': 'connector',  'glyph_class': 'bench'},
    'ch':    {'first_stroke': 'open_curve', 'last_stroke': 'connector',  'glyph_class': 'bench'},
    'sh':    {'first_stroke': 'sigmoid',    'last_stroke': 'connector',  'glyph_class': 'bench'},
    'cth':   {'first_stroke': 'open_curve', 'last_stroke': 'connector',  'glyph_class': 'bench'},
    'ckh':   {'first_stroke': 'open_curve', 'last_stroke': 'connector',  'glyph_class': 'bench'},
    'cph':   {'first_stroke': 'open_curve', 'last_stroke': 'connector',  'glyph_class': 'bench'},
    'cfh':   {'first_stroke': 'open_curve', 'last_stroke': 'connector',  'glyph_class': 'bench'},
    's':     {'first_stroke': 'sigmoid',    'last_stroke': 'sigmoid',    'glyph_class': 'bench'},
    # rare glyphs
    'v':     {'first_stroke': 'open_curve', 'last_stroke': 'hook',       'glyph_class': 'rare'},
    'z':     {'first_stroke': 'sigmoid',    'last_stroke': 'hook',       'glyph_class': 'rare'},
    'x':     {'first_stroke': 'crossbar',   'last_stroke': 'crossbar',   'glyph_class': 'rare'},
    # connector-onset glyphs
    'b':     {'first_stroke': 'connector',  'last_stroke': 'connector',  'glyph_class': 'bench'},
    'j':     {'first_stroke': 'connector',  'last_stroke': 'connector',  'glyph_class': 'bench'},
    'u':     {'first_stroke': 'connector',  'last_stroke': 'connector',  'glyph_class': 'bench'},
}


# ---------------------------------------------------------------------------
# Phase 14: Articulatory hypotheses for stroke-type → phoneme mappings
# ---------------------------------------------------------------------------
# These are PRIORS for domain initialization in the feature CSP, not hard
# constraints.  Each entry maps one stroke type to a ranked list of candidate
# phonemes.  The intersection of PHONEME_PLACE_MAP[first_stroke] and
# PHONEME_NUCLEUS_MAP[last_stroke] seeds the domain for each feature triple.

PHONEME_PLACE_MAP: Dict[str, List[str]] = {
    # first_stroke -> candidate onset consonants (place of articulation)
    # Tall strokes (ascender) = stops (most common consonant class in Latin)
    'ascender':   ['t', 'k', 'p', 'd', 'g', 'b'],
    # Connected strokes = labials and bilabials
    'connector':  ['b', 'p', 'v', 'm', 'f'],
    # Crossbar = rare/fricative category
    'crossbar':   ['x', 'h', 'f', 'k'],
    # Loop onset = liquids, sonorants, open vowels
    'loop':       ['l', 'r', 'n', 'a', 'o', 'e'],
    # Open curve = sibilants / palatals
    'open_curve': ['c', 's', 'sc', 'h'],
    # Sigmoid = sibilant category
    'sigmoid':    ['s', 'z', 'sc'],
    # Vertical strokes = nasals and dentals
    'vertical':   ['m', 'n', 'd', 'l', 'i'],
}

PHONEME_NUCLEUS_MAP: Dict[str, List[str]] = {
    # last_stroke -> candidate vowel qualities / coda phonemes
    # Tall ascender end = open/low vowels
    'ascender':   ['a', 'e', 'i', 'o'],
    # Connector end = mid vowels
    'connector':  ['e', 'i', 'a', 'o'],
    # Crossbar end = coronal coda / dental closure
    'crossbar':   ['e', 'a', 't'],
    # Descender = high back vowels
    'descender':  ['u', 'i', 'o', 'y'],
    # Hook end = nasal coda
    'hook':       ['n', 'm', 'a', 'i'],
    # Loop end = round / back vowels
    'loop':       ['o', 'a', 'u', 'e'],
    # Open curve end = open vowels
    'open_curve': ['a', 'e', 'o'],
    # Plume end = labial coda
    'plume':      ['p', 'f', 'a', 'e'],
    # Sigmoid end = rhotic / sibilant coda
    'sigmoid':    ['r', 'a', 's', 'e'],
    # Tail end = unrounded front vowels
    'tail':       ['a', 'e', 'i'],
    # Vertical end = high front / lateral
    'vertical':   ['i', 'l', 'n', 'e'],
}


def build_triple_phoneme_hypotheses(
    language: str,
    inventory: Optional[Any] = None,
) -> Dict[str, List[str]]:
    """Build candidate syllable lists for each attested feature triple.

    For each of the ~23 attested ``(first_stroke, last_stroke, glyph_class)``
    triples found in :data:`EVA_VISUAL_COMPONENTS`, generates candidate
    syllables as the cross-product of
    ``PHONEME_PLACE_MAP[first_stroke]`` × ``PHONEME_NUCLEUS_MAP[last_stroke]``,
    filtered to syllables legal in the target language inventory.

    If the cross-product yields no legal syllables (e.g. for a rare triple),
    falls back to the full language inventory so the domain is never empty.

    Parameters
    ----------
    language:
        Target language code (e.g. ``'latin'``).
    inventory:
        Optional pre-built syllable list (from
        :func:`build_cv_syllable_table`).  If *None*, the table is built
        automatically for *language*.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from ``triple_key`` string to list of candidate syllables.
    """
    if inventory is None:
        inventory = build_cv_syllable_table(language)
    inv_set = set(inventory)

    hypotheses: Dict[str, List[str]] = {}

    # Collect all unique triples from EVA_VISUAL_COMPONENTS
    triples_seen: Dict[str, Tuple[str, str, str]] = {}
    for _glyph, comp in EVA_VISUAL_COMPONENTS.items():
        fs = comp['first_stroke']
        ls = comp['last_stroke']
        gc = comp['glyph_class']
        triple_key = f"{fs},{ls},{gc}"
        triples_seen[triple_key] = (fs, ls, gc)

    for triple_key, (fs, ls, _gc) in triples_seen.items():
        onset_candidates = PHONEME_PLACE_MAP.get(fs, [])
        nucleus_candidates = PHONEME_NUCLEUS_MAP.get(ls, [])

        candidates: List[str] = []
        # Cross-product: onset + nucleus
        for onset in onset_candidates:
            for nucleus in nucleus_candidates:
                syl = onset + nucleus
                if syl in inv_set:
                    candidates.append(syl)
        # Also include pure vowels that appear as nuclei
        for nucleus in nucleus_candidates:
            if nucleus in inv_set and nucleus not in candidates:
                candidates.append(nucleus)

        if not candidates:
            # Fallback: full inventory so the domain is never empty
            candidates = list(inventory)

        hypotheses[triple_key] = candidates

    return hypotheses


# ---------------------------------------------------------------------------
# Phase 15 – Medieval Latin dictionary expansion
# ---------------------------------------------------------------------------

MEDIEVAL_SPELLING_RULES: List[Tuple[str, str, str]] = [
    # (pattern, replacement, category)
    # ae/oe simplification (very common in medieval Latin)
    ('ae', 'e', 'ae_simplification'),
    ('oe', 'e', 'oe_simplification'),
    # vowel interchange
    ('e', 'i', 'vowel_interchange'),
    ('i', 'e', 'vowel_interchange'),
    ('o', 'u', 'vowel_interchange'),
    ('u', 'o', 'vowel_interchange'),
    # consonant voicing/devoicing
    ('t', 'd', 'voicing'),
    ('d', 't', 'voicing'),
    ('p', 'b', 'voicing'),
    ('b', 'p', 'voicing'),
    ('c', 'g', 'voicing'),
    ('g', 'c', 'voicing'),
    # h variation (universally unstable in medieval Latin)
    ('h', '', 'h_loss'),
    # gemination / degemination
    ('ll', 'l', 'degemination'),
    ('l', 'll', 'gemination'),
    ('ss', 's', 'degemination'),
    ('s', 'ss', 'gemination'),
    ('rr', 'r', 'degemination'),
    ('r', 'rr', 'gemination'),
    ('nn', 'n', 'degemination'),
    ('n', 'nn', 'gemination'),
    ('tt', 't', 'degemination'),
    ('t', 'tt', 'gemination'),
    ('pp', 'p', 'degemination'),
    ('p', 'pp', 'gemination'),
]

PHARMACEUTICAL_VOCABULARY: Dict[str, List[str]] = {
    'verbs': [
        'recipe', 'accipe', 'misce', 'contere', 'coque',
        'adde', 'pone', 'distilla', 'cola', 'solve',
        'funde', 'unge', 'bibe', 'lava', 'tere',
    ],
    'plant_parts': [
        'folia', 'radix', 'flos', 'semen', 'cortex',
        'herba', 'ramus', 'bacca', 'bulbus', 'tuber',
        'spica', 'gummi', 'resina', 'succus',
    ],
    'preparations': [
        'aqua', 'oleum', 'vinum', 'mel', 'acetum',
        'pulvis', 'emplastrum', 'unguentum', 'sirupus',
        'decoctum', 'infusum', 'electuarium', 'pilula',
    ],
    'body_parts': [
        'caput', 'stomachum', 'oculus', 'dens', 'pectus',
        'iecur', 'ren', 'venter', 'manus', 'pes',
    ],
    'qualities': [
        'calidus', 'frigidus', 'siccus', 'humidus',
        'bene', 'male', 'niger', 'albus', 'dulcis',
    ],
    'function_words': [
        'et', 'in', 'cum', 'est', 'ad', 'de',
        'per', 'pro', 'sine', 'non', 'vel', 'aut',
    ],
}

LATIN_PHRASE_PATTERNS: List[Tuple[str, List[str]]] = [
    ('recipe_formula', ['recipe', 'accipe']),
    ('medium', ['cum', 'aqua', 'in', 'aqua', 'oleum', 'vinum']),
    ('instruction', ['et', 'misce', 'contere', 'coque', 'adde']),
    ('heating', ['ad', 'ignem', 'coque']),
    ('quality', ['est', 'calidus', 'frigidus', 'siccus', 'humidus']),
    ('degree', ['in', 'primo', 'secundo', 'tertio', 'gradu']),
    ('efficacy', ['valet', 'contra', 'prodest']),
    ('plant_desc', ['folia', 'radix', 'flos', 'semen']),
]

# ── Phase 17 Step 0: Honesty Diagnostics Constants ──────────────────────

# Top-100 Latin medical words ranked by expected frequency in pharmaceutical text.
# Rank 1 = most expected. Built from PHARMACEUTICAL_VOCABULARY + high-frequency
# medical Latin terms from Circa Instans / De Viribus Herbarum.
LATIN_MEDICAL_TOP_100: List[Tuple[str, int]] = [
    # Function words (highest frequency)
    ('et', 1), ('in', 2), ('est', 3), ('ad', 4), ('de', 5),
    ('cum', 6), ('non', 7), ('per', 8), ('ut', 9), ('si', 10),
    ('quod', 11), ('sed', 12), ('vel', 13), ('pro', 14), ('ex', 15),
    ('ab', 16), ('qui', 17), ('que', 18), ('aut', 19), ('hoc', 20),
    # Pharmaceutical verbs (imperatives)
    ('recipe', 21), ('accipe', 22), ('misce', 23), ('contere', 24),
    ('coque', 25), ('adde', 26), ('pone', 27), ('distilla', 28),
    ('cola', 29), ('solve', 30), ('tere', 31), ('fac', 32),
    ('frange', 33), ('incide', 34), ('lava', 35),
    # Plant parts
    ('folia', 36), ('radix', 37), ('flos', 38), ('semen', 39),
    ('cortex', 40), ('herba', 41), ('ramus', 42), ('bacca', 43),
    # Preparations
    ('aqua', 44), ('oleum', 45), ('vinum', 46), ('mel', 47),
    ('acetum', 48), ('pulvis', 49), ('unguentum', 50), ('sirupus', 51),
    # Body parts
    ('caput', 52), ('cor', 53), ('stomachum', 54), ('oculus', 55),
    ('iecur', 56), ('ren', 57), ('pectus', 58), ('manus', 59),
    # Qualities
    ('calidus', 60), ('frigidus', 61), ('siccus', 62), ('humidus', 63),
    ('bonus', 64), ('magnus', 65), ('bene', 66), ('male', 67),
    # Degree and structure
    ('gradu', 68), ('primo', 69), ('secundo', 70), ('tertio', 71),
    ('natura', 72), ('virtus', 73), ('contra', 74), ('dies', 75),
    # Medical terms
    ('febris', 76), ('dolor', 77), ('morbus', 78), ('vulnus', 79),
    ('tumor', 80), ('sanguis', 81), ('venenum', 82), ('remedium', 83),
    # Quantities
    ('drachma', 84), ('uncia', 85), ('libra', 86), ('ana', 87),
    ('quantum', 88), ('partes', 89), ('modicum', 90), ('satis', 91),
    # Actions and additional terms
    ('calefacit', 92), ('mundificat', 93), ('valet', 94), ('prodest', 95),
    ('cura', 96), ('potio', 97), ('emplastrum', 98), ('nox', 99),
    ('mane', 100),
]

# Pharmaceutical imperative verbs ranked by expected frequency
LATIN_IMPERATIVE_RANKED: Dict[str, int] = {
    'recipe': 1, 'accipe': 2, 'misce': 3, 'contere': 4, 'coque': 5,
    'adde': 6, 'pone': 7, 'distilla': 8, 'cola': 9, 'solve': 10,
    'tere': 11, 'fac': 12, 'frange': 13, 'incide': 14, 'lava': 15,
}

# Calendar and zodiac names for astronomical section testing
LATIN_MONTH_NAMES: List[str] = [
    'ianuarius', 'februarius', 'martius', 'aprilis', 'maius', 'iunius',
    'iulius', 'augustus', 'september', 'october', 'november', 'december',
]

LATIN_ZODIAC_NAMES: List[str] = [
    'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
    'libra', 'scorpio', 'sagittarius', 'capricornus', 'aquarius', 'pisces',
]

# Latin nominal declension endings for inflected form generation
_LATIN_NOUN_ENDINGS: Dict[str, List[str]] = {
    'noun1': ['a', 'ae', 'am', 'arum', 'is'],            # 1st decl (rosa)
    'noun2': ['us', 'i', 'o', 'um', 'orum', 'is', 'os'], # 2nd decl (hortus)
    'noun2n': ['um', 'i', 'o', 'orum', 'is', 'a'],       # 2nd neut (oleum)
    'noun3': ['is', 'i', 'em', 'e', 'um', 'ibus', 'es'], # 3rd decl (radix)
    'noun4': ['us', 'ui', 'um', 'u', 'uum', 'ibus'],     # 4th decl (fructus)
    'noun5': ['es', 'ei', 'em', 'erum', 'ebus'],          # 5th decl (species)
}

_LATIN_VERB_ENDINGS: Dict[str, List[str]] = {
    'verb1': ['o', 'as', 'at', 'amus', 'atis', 'ant',     # pres ind
              'a', 'are', 'atur', 'ans', 'ando'],          # imp + inf + part
    'verb2': ['eo', 'es', 'et', 'emus', 'etis', 'ent',
              'e', 'ere', 'etur', 'ens', 'endo'],
    'verb3': ['o', 'is', 'it', 'imus', 'itis', 'unt',
              'e', 'ere', 'itur', 'ens', 'endo'],
    'verb4': ['io', 'is', 'it', 'imus', 'itis', 'iunt',
              'i', 'ire', 'itur', 'iens', 'iendo'],
}

_LATIN_ADJ_ENDINGS: List[str] = [
    'us', 'a', 'um', 'i', 'ae', 'o', 'am', 'os', 'as',  # 1st/2nd
    'is', 'e', 'em', 'ibus', 'es', 'ia',                  # 3rd
]


def generate_medieval_variants(word: str) -> Dict[str, List[str]]:
    """Generate medieval Latin spelling variants of a word.

    Returns dict mapping variant_string -> [rule_categories_applied].
    Only single-rule applications to avoid combinatorial explosion.
    """
    variants: Dict[str, List[str]] = {}

    for pattern, replacement, category in MEDIEVAL_SPELLING_RULES:
        idx = 0
        while True:
            pos = word.find(pattern, idx)
            if pos < 0:
                break
            variant = word[:pos] + replacement + word[pos + len(pattern):]
            if variant != word and len(variant) >= 2:
                if variant not in variants:
                    variants[variant] = []
                variants[variant].append(category)
            idx = pos + 1

    return variants


def generate_inflected_forms(stem: str, pos: str) -> List[str]:
    """Generate Latin inflected forms for a stem.

    Parameters
    ----------
    stem : str
        The base stem (without ending), e.g. 'ros' for 'rosa'.
    pos : str
        Paradigm key: 'noun1'–'noun5', 'noun2n', 'verb1'–'verb4', 'adj'.

    Returns a list of unique inflected forms.
    """
    if pos == 'adj':
        endings = _LATIN_ADJ_ENDINGS
    elif pos in _LATIN_NOUN_ENDINGS:
        endings = _LATIN_NOUN_ENDINGS[pos]
    elif pos in _LATIN_VERB_ENDINGS:
        endings = _LATIN_VERB_ENDINGS[pos]
    else:
        return [stem]

    forms = set()
    for ending in endings:
        form = stem + ending
        if len(form) >= 2:
            forms.add(form)
    return sorted(forms)


def build_expanded_word_set(
    base_word_set: set,
) -> Tuple[set, Dict[str, str]]:
    """Build an expanded Latin dictionary from the base reference word set.

    Expansion sources:
    1. Medieval spelling variants of every base word
    2. Pharmaceutical vocabulary terms (hand-curated)
    3. Inflected forms of pharmaceutical stems

    Returns
    -------
    (expanded_set, provenance_map)
        provenance_map maps each NEW word to the base word or source it
        derives from.
    """
    expanded = set(base_word_set)
    provenance: Dict[str, str] = {}

    # 1. Medieval spelling variants
    for word in list(base_word_set):
        for variant, _cats in generate_medieval_variants(word).items():
            if variant not in expanded:
                expanded.add(variant)
                provenance[variant] = f"variant:{word}"

    # 2. Pharmaceutical vocabulary (direct terms)
    for domain, words in PHARMACEUTICAL_VOCABULARY.items():
        for w in words:
            wl = w.lower()
            if wl not in expanded:
                expanded.add(wl)
                provenance[wl] = f"pharma:{domain}"

    # 3. Inflected forms of pharmaceutical vocabulary stems
    _PHARMA_STEMS: List[Tuple[str, str]] = [
        # (stem, paradigm)
        ('aqu', 'noun1'), ('ole', 'noun2n'), ('vin', 'noun2n'),
        ('herb', 'noun1'), ('foli', 'noun2n'), ('radic', 'noun3'),
        ('flor', 'noun3'), ('semin', 'noun3'), ('cortic', 'noun3'),
        ('ram', 'noun2'), ('bacc', 'noun1'), ('succus', 'noun2'),
        ('pulver', 'noun3'), ('sirup', 'noun2'),
        ('calid', 'adj'), ('frigid', 'adj'), ('sicc', 'adj'),
        ('humid', 'adj'), ('nigr', 'adj'), ('alb', 'adj'),
        ('dulc', 'noun3'),
        ('coqu', 'verb3'), ('misc', 'verb2'), ('add', 'verb3'),
        ('solv', 'verb3'), ('lav', 'verb1'), ('bib', 'verb3'),
        ('fund', 'verb3'), ('distill', 'verb1'), ('col', 'verb1'),
        ('ung', 'verb3'), ('ter', 'verb3'),
    ]
    for stem, paradigm in _PHARMA_STEMS:
        for form in generate_inflected_forms(stem, paradigm):
            fl = form.lower()
            if fl not in expanded:
                expanded.add(fl)
                provenance[fl] = f"inflection:{stem}({paradigm})"

    # 4. Medieval variants of pharmaceutical terms and inflections ONLY
    #    (NOT step-1 spelling variants — that would be variants-of-variants)
    pharma_direct: set = set()
    for domain, words in PHARMACEUTICAL_VOCABULARY.items():
        for w in words:
            pharma_direct.add(w.lower())
    for stem, paradigm in _PHARMA_STEMS:
        for form in generate_inflected_forms(stem, paradigm):
            pharma_direct.add(form.lower())

    for word in pharma_direct:
        for variant, _cats in generate_medieval_variants(word).items():
            if variant not in expanded:
                expanded.add(variant)
                provenance[variant] = f"pharma_variant:{word}"

    return expanded, provenance


# ---------------------------------------------------------------------------
# Phase A-D: Paleographic Reference Loading and Validation
# ---------------------------------------------------------------------------

VALID_FIRST_STROKES = {'ascender', 'connector', 'crossbar', 'loop', 'open_curve', 'sigmoid', 'vertical'}
VALID_LAST_STROKES = {'ascender', 'connector', 'crossbar', 'descender', 'hook', 'loop', 'open_curve', 'plume', 'sigmoid', 'tail', 'vertical'}
VALID_GLYPH_CLASSES = {'bench', 'compound', 'gallows', 'minim', 'rare', 'suffix'}
VALID_MODIFIER_MARKS = {'dot', 'tick', 'thickening', 'angle_change', 'serif', 'crossbar_added'}
VALID_CONFIDENCES = {'high', 'medium', 'low'}


def validate_stroke_fields(entry: Dict[str, Any]) -> List[str]:
    """Validate stroke vocabulary fields in a paleographic sign entry.

    Returns list of error messages (empty if valid).
    """
    errors: List[str] = []
    sid = entry.get('sign_id', '?')

    fs = entry.get('first_stroke')
    ls = entry.get('last_stroke')
    gc = entry.get('glyph_class')
    tk = entry.get('triple_key')

    if fs and fs != 'unclear' and fs not in VALID_FIRST_STROKES:
        errors.append(f"{sid}: invalid first_stroke '{fs}'")
    if ls and ls != 'unclear' and ls not in VALID_LAST_STROKES:
        errors.append(f"{sid}: invalid last_stroke '{ls}'")
    if gc and gc != 'unclear' and gc not in VALID_GLYPH_CLASSES:
        errors.append(f"{sid}: invalid glyph_class '{gc}'")

    # Check triple_key consistency
    if fs and ls and gc and tk:
        expected_tk = f"{fs},{ls},{gc}"
        if tk != expected_tk:
            errors.append(f"{sid}: triple_key '{tk}' != expected '{expected_tk}'")

    conf = entry.get('confidence')
    if conf and conf not in VALID_CONFIDENCES:
        errors.append(f"{sid}: invalid confidence '{conf}'")

    marks = entry.get('modifier_marks', [])
    for m in marks:
        if m not in VALID_MODIFIER_MARKS:
            errors.append(f"{sid}: invalid modifier_mark '{m}'")

    return errors


def _load_json_safe(path: str) -> Optional[Dict]:
    """Load a JSON file, returning None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_tironian_reference(source: str = 'all') -> List[Dict[str, Any]]:
    """Load Tironian reference signs from data/reference/tironian/.

    Parameters
    ----------
    source : str
        'schmitz', 'chatelain', or 'all' (default).

    Returns
    -------
    List of sign entry dicts, each tagged with 'source_file'.
    """
    base = str(_data_dir('reference/tironian'))
    signs: List[Dict[str, Any]] = []

    files_to_load: List[Tuple[str, str]] = []
    if source in ('schmitz', 'all'):
        files_to_load.append((os.path.join(base, 'schmitz_plates.json'), 'schmitz'))
    if source in ('chatelain', 'all'):
        files_to_load.append((os.path.join(base, 'chatelain_bobbio.json'), 'chatelain'))

    for fpath, tag in files_to_load:
        data = _load_json_safe(fpath)
        if data is None:
            continue
        for s in data.get('signs', []):
            s['source_file'] = tag
            signs.append(s)

    return signs


def load_cappelli_reference() -> List[Dict[str, Any]]:
    """Load Cappelli abbreviation entries from data/reference/cappelli/."""
    fpath = os.path.join(str(_data_dir('reference/cappelli')), 'cappelli_entries.json')
    data = _load_json_safe(fpath)
    if data is None:
        return []
    entries = data.get('entries', [])
    for e in entries:
        e['source_file'] = 'cappelli'
    return entries


def load_costamagna_reference() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load Costamagna sign tables from data/reference/costamagna/.

    Returns
    -------
    (family_signs, unaffiliated_signs) where family_signs is a flat list
    of all signs from all families (each tagged with family_id), and
    unaffiliated_signs is a list of signs without family membership.
    """
    fpath = os.path.join(str(_data_dir('reference/costamagna')), 'costamagna_signs.json')
    data = _load_json_safe(fpath)
    if data is None:
        return [], []

    family_signs: List[Dict[str, Any]] = []
    for fam in data.get('sign_families', []):
        fam_id = fam.get('family_id', '?')
        for s in fam.get('members', []):
            s['family_id'] = fam_id
            s['source_file'] = 'costamagna'
            family_signs.append(s)

    unaffiliated = data.get('unaffiliated_signs', [])
    for s in unaffiliated:
        s['source_file'] = 'costamagna'

    return family_signs, unaffiliated


def load_fontana_reference() -> List[Dict[str, Any]]:
    """Load Fontana cipher signs from data/reference/fontana/."""
    fpath = os.path.join(str(_data_dir('reference/fontana')), 'fontana_signs.json')
    data = _load_json_safe(fpath)
    if data is None:
        return []
    signs = data.get('signs', [])
    for s in signs:
        s['source_file'] = 'fontana'
    return signs


def load_milanese_reference() -> List[Dict[str, Any]]:
    """Load Milanese cipher keys from data/reference/milanese/."""
    fpath = os.path.join(str(_data_dir('reference/milanese')), 'milanese_cipher_keys.json')
    data = _load_json_safe(fpath)
    if data is None:
        return []
    return data.get('ciphers', [])


def load_ligature_observations() -> Optional[Dict[str, Any]]:
    """Load ligature observations from data/reference/ligature/."""
    fpath = os.path.join(str(_data_dir('reference/ligature')), 'ligature_observations.json')
    return _load_json_safe(fpath)


def load_master_reference() -> Optional[Dict[str, Any]]:
    """Load the merged master reference from data/reference/paleographic/."""
    fpath = os.path.join(str(_data_dir('reference/paleographic')), 'master_reference.json')
    return _load_json_safe(fpath)


def detect_sign_families(signs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group signs into families that share the same first_stroke.

    Within a family, members share first_stroke but differ by last_stroke
    and/or glyph_class — these are minimal-pair families analogous to the
    onset families in Phase 14's Voynich feature decomposition.

    Returns list of family dicts:
    {
        'family_id': str,
        'common_first_stroke': str,
        'n_members': int,
        'members': List[Dict],
        'triple_keys': List[str],
    }
    """
    from collections import defaultdict
    by_first: Dict[str, List[Dict]] = defaultdict(list)
    for s in signs:
        fs = s.get('first_stroke', 'unclear')
        if fs != 'unclear':
            by_first[fs].append(s)

    families: List[Dict[str, Any]] = []
    for idx, (fs, members) in enumerate(sorted(by_first.items()), 1):
        if len(members) < 2:
            continue
        triple_keys = sorted(set(s.get('triple_key', '') for s in members if s.get('triple_key')))
        families.append({
            'family_id': f'FAM_{idx:02d}',
            'common_first_stroke': fs,
            'n_members': len(members),
            'members': members,
            'triple_keys': triple_keys,
            'n_distinct_triples': len(triple_keys),
        })

    return families


def build_tironian_domain_priors(
    master_ref: Dict[str, Any],
    voynich_triples: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build CSP domain priors from Tironian sign matches.

    For each Voynich triple_key, finds Tironian signs with matching
    triple_key and returns their latin syllable values as domain candidates.

    Parameters
    ----------
    master_ref : dict
        The master reference JSON (loaded via load_master_reference()).
    voynich_triples : list of str
        The 25 Voynich triple_keys from stroke_features.json.

    Returns
    -------
    Dict mapping triple_key -> {
        'tironian_candidates': List[str],  # syllable values from matches
        'match_type': str,                 # 'exact', 'near', or 'none'
        'n_matches': int,
        'confidences': List[str],
    }
    """
    # Index all reference signs by triple_key
    ref_by_triple: Dict[str, List[Dict]] = {}
    for sign in master_ref.get('all_signs', []):
        tk = sign.get('triple_key', '')
        if tk:
            ref_by_triple.setdefault(tk, []).append(sign)

    # Also build a partial-match index (2-of-3 component matches)
    ref_components: List[Tuple[str, str, str, Dict]] = []
    for sign in master_ref.get('all_signs', []):
        fs = sign.get('first_stroke', '')
        ls = sign.get('last_stroke', '')
        gc = sign.get('glyph_class', '')
        if fs and ls and gc:
            ref_components.append((fs, ls, gc, sign))

    priors: Dict[str, Dict[str, Any]] = {}
    for vtk in voynich_triples:
        parts = vtk.split(',')
        if len(parts) != 3:
            priors[vtk] = {'tironian_candidates': [], 'match_type': 'none', 'n_matches': 0, 'confidences': []}
            continue

        vfs, vls, vgc = parts

        # Try exact match first
        exact = ref_by_triple.get(vtk, [])
        if exact:
            candidates = []
            confs = []
            for s in exact:
                val = s.get('latin_expansion') or s.get('syllable_value') or s.get('latin_value')
                if val:
                    candidates.append(val.lower())
                    confs.append(s.get('confidence', 'low'))
            priors[vtk] = {
                'tironian_candidates': sorted(set(candidates)),
                'match_type': 'exact',
                'n_matches': len(exact),
                'confidences': confs,
            }
            continue

        # Try near match (2 of 3 components match)
        near_candidates = []
        near_confs = []
        for rfs, rls, rgc, sign in ref_components:
            match_count = (rfs == vfs) + (rls == vls) + (rgc == vgc)
            if match_count >= 2:
                val = sign.get('latin_expansion') or sign.get('syllable_value') or sign.get('latin_value')
                if val:
                    near_candidates.append(val.lower())
                    near_confs.append(sign.get('confidence', 'low'))

        if near_candidates:
            priors[vtk] = {
                'tironian_candidates': sorted(set(near_candidates)),
                'match_type': 'near',
                'n_matches': len(near_candidates),
                'confidences': near_confs,
            }
        else:
            priors[vtk] = {
                'tironian_candidates': [],
                'match_type': 'none',
                'n_matches': 0,
                'confidences': [],
            }

    return priors
