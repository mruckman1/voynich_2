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
