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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
