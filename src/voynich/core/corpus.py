"""
Voynich Corpus Module
======================
IVTFF parser, EVA text cleaning, corpus access by section/language/hand.

Carried over from voynich/data/ivtff_parser.py with adaptations:
- Embedded EVA glyph metadata (no JSON dependency)
- Section inference from folio numbers
- Path handling via pathlib
"""

import re
import os
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

from voynich.core._paths import data_dir as _data_dir


# ---------------------------------------------------------------------------
# EVA Glyph Metadata (embedded — no JSON dependency)
# ---------------------------------------------------------------------------

EVA_GLYPHS = {
    'o': {'class': 'bench', 'description': 'bench-loop'},
    'a': {'class': 'bench', 'description': 'bench element'},
    'e': {'class': 'bench', 'description': 'bench variant'},
    'l': {'class': 'bench', 'description': 'loop element'},
    'r': {'class': 'bench', 'description': 'flourish stroke'},
    'c': {'class': 'bench', 'description': 'bench-c'},
    'h': {'class': 'bench', 'description': 'bench-h'},
    'x': {'class': 'bench', 'description': 'rare cross-stroke'},
    'i': {'class': 'minim', 'description': 'single minim'},
    'g': {'class': 'minim', 'description': 'rare medial'},
    'n': {'class': 'suffix', 'description': 'terminal flourish'},
    'm': {'class': 'suffix', 'description': 'terminal flourish variant'},
    'y': {'class': 'suffix', 'description': 'terminal descender'},
    'd': {'class': 'gallows', 'description': 'tall gallows prefix'},
    's': {'class': 'gallows', 'description': 'initial stroke'},
    'k': {'class': 'gallows', 'description': 'gallows glyph'},
    't': {'class': 'gallows', 'description': 'gallows glyph'},
    'p': {'class': 'gallows', 'description': 'gallows with plume'},
    'f': {'class': 'gallows', 'description': 'gallows with plume'},
    'q': {'class': 'gallows', 'description': 'initial-q'},
    'v': {'class': 'rare', 'description': 'rare bench variant'},
    'w': {'class': 'rare', 'description': 'rare bench variant'},
    'z': {'class': 'rare', 'description': 'rare element'},
}

EVA_LIGATURES = [
    'sh', 'ch', 'cth', 'ckh', 'cph', 'cfh',
    'iin', 'iiin', 'aiin', 'aiiin',
    'ol', 'or', 'al', 'ar',
    'qo', 'qok', 'qot',
    'dy', 'ey',
]

# Sorted longest-first for greedy matching
EVA_LIGATURES_SORTED = sorted(EVA_LIGATURES, key=len, reverse=True)


# ---------------------------------------------------------------------------
# Section / Language / Scribe Inference
# ---------------------------------------------------------------------------

VOYNICH_SECTIONS = {
    'herbal_a':       {'currier_lang': 'A', 'primary_scribe': 1},
    'herbal_b':       {'currier_lang': 'B', 'primary_scribe': 2},
    'astronomical':   {'currier_lang': 'B', 'primary_scribe': 3},
    'biological':     {'currier_lang': 'B', 'primary_scribe': 4},
    'cosmological':   {'currier_lang': 'B', 'primary_scribe': 3},
    'pharmaceutical': {'currier_lang': 'B', 'primary_scribe': 5},
    'recipes':        {'currier_lang': 'B', 'primary_scribe': 5},
}

def _infer_section(folio: str) -> str:
    """Infer section from folio number."""
    try:
        num = int(re.search(r'\d+', folio).group())
    except (AttributeError, ValueError):
        return 'unknown'
    if num <= 56:
        return 'herbal_a'
    elif 67 <= num <= 74:
        return 'astronomical'
    elif 75 <= num <= 84:
        return 'biological'
    elif 85 <= num <= 86:
        return 'cosmological'
    elif num == 87:
        return 'herbal_b'
    elif 88 <= num <= 102:
        return 'pharmaceutical'
    elif num >= 103:
        return 'recipes'
    return 'unknown'

def _infer_scribe(folio: str) -> int:
    """Infer scribe from folio number based on quire assignments."""
    try:
        num = int(re.search(r'\d+', folio).group())
    except (AttributeError, ValueError):
        return 0
    if num <= 56:
        return 1
    elif 87 <= num <= 102:
        return 2
    elif 67 <= num <= 74:
        return 3
    elif 75 <= num <= 84:
        return 4
    elif num >= 103:
        return 5
    return 0

def _infer_language(folio: str) -> str:
    """Infer Currier language from folio."""
    scribe = _infer_scribe(folio)
    return 'A' if scribe == 1 else 'B'


# ---------------------------------------------------------------------------
# EVA Text Cleaning
# ---------------------------------------------------------------------------

def clean_eva_text(raw: str) -> str:
    """
    Clean raw IVTFF text into pure EVA tokens.

    Conversions:
    - Remove <! ... > comments
    - Replace <-> (drawing breaks) with spaces
    - Remove <~> (alignment marks), <$> (line-end markers)
    - Resolve uncertain readings [a:o] -> take first option
    - Remove curly-brace ligature markers {ao} -> ao
    - Lowercase everything
    - Periods (word separators) -> spaces
    - Strip * ? , ' markers
    - Remove @NNN; glyph references
    - Remove all non-[a-z ] characters
    """
    text = raw

    # Remove inline comments <! ... >
    text = re.sub(r'<![^>]*>', '', text)

    # Drawing breaks -> space, alignment marks -> remove
    text = text.replace('<->', ' ')
    text = text.replace('<~>', ' ')
    text = text.replace('<$>', '')

    # Strip trailing line markers
    text = text.rstrip('-=')

    # Resolve uncertain readings: [a:o] -> a (first option)
    text = re.sub(r'\[([^:\]]*?):[^\]]*?\]', r'\1', text)
    text = re.sub(r'\[([^\]]*?)\]', r'\1', text)

    # Remove ligature markers: {ao} -> ao
    text = re.sub(r'\{([^}]*)\}', r'\1', text)

    # Lowercase
    text = text.lower()

    # Periods = word separators -> spaces
    text = text.replace('.', ' ')

    # Strip uncertainty/formatting markers
    text = text.replace('*', '')
    text = text.replace('?', '')
    text = text.replace(',', '')
    text = text.replace("'", '')

    # Remove glyph number references @254;
    text = re.sub(r'@\d+;?', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove any remaining non-alphabetic characters
    text = re.sub(r'[^a-z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text: str) -> List[str]:
    """Split EVA text into tokens (words)."""
    return [w for w in text.strip().split() if w]


def tokenize_eva_chars(token: str) -> List[str]:
    """
    Parse a single EVA token into its constituent EVA characters/ligatures.
    Uses longest-match-first to handle multi-character sequences like
    'sh', 'ch', 'cth', 'ckh', 'aiin', etc.

    Example: 'shody' -> ['sh', 'o', 'd', 'y']
             'cthres' -> ['cth', 'r', 'e', 's']
    """
    chars = []
    i = 0
    while i < len(token):
        matched = False
        # Try longest ligatures first
        for lig in EVA_LIGATURES_SORTED:
            if token[i:i+len(lig)] == lig:
                chars.append(lig)
                i += len(lig)
                matched = True
                break
        if not matched:
            chars.append(token[i])
            i += 1
    return chars


# ---------------------------------------------------------------------------
# IVTFF Data Structures
# ---------------------------------------------------------------------------

class VoynichPage:
    """Represents a single page (folio) of the manuscript."""

    def __init__(self, folio: str):
        self.folio = folio
        self.quire = 0
        self.language = ''
        self.hand = 0
        self.illustration = ''
        self.page_num = 0
        self.cluster = ''
        self.loci: List['VoynichLocus'] = []
        self.comments: List[str] = []

    @property
    def all_text(self) -> str:
        """All transliterated text on this page, space-separated."""
        return ' '.join(loc.clean_text for loc in self.loci if loc.clean_text)

    @property
    def all_tokens(self) -> List[str]:
        """All tokens on this page."""
        return self.all_text.split()

    @property
    def paragraph_text(self) -> str:
        """Only paragraph (running) text, excluding labels and circular text."""
        return ' '.join(
            loc.clean_text for loc in self.loci
            if loc.locus_type.startswith('P') and loc.clean_text
        )

    @property
    def section(self) -> str:
        """Infer section from illustration type and folio number."""
        # Use folio-number-based inference (more granular than illustration type)
        return _infer_section(self.folio)


class VoynichLocus:
    """A single text item (line/label/circular text) on a page."""

    def __init__(self, locus_id: str, locus_type: str, raw_text: str):
        self.locus_id = locus_id
        self.locus_type = locus_type
        self.raw_text = raw_text
        self._clean = None

    @property
    def clean_text(self) -> str:
        """Text with annotations stripped, periods->spaces, cleaned."""
        if self._clean is None:
            self._clean = clean_eva_text(self.raw_text)
        return self._clean


# ---------------------------------------------------------------------------
# IVTFF Parser
# ---------------------------------------------------------------------------

def parse_ivtff(filepath: str, verbose: bool = False) -> Dict[str, VoynichPage]:
    """
    Parse an IVTFF transliteration file into structured VoynichPage objects.

    Parameters:
        filepath: Path to .txt IVTFF file (e.g., ZL3b-n.txt)
        verbose: Print parsing statistics

    Returns:
        Dict[folio_id, VoynichPage]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"IVTFF file not found: {filepath}")

    pages = {}
    current_page = None
    line_count = 0
    locus_count = 0
    skipped = 0

    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        for raw_line in fh:
            line_count += 1
            line = raw_line.rstrip('\n\r')

            if not line.strip():
                continue

            # File header
            if line.startswith('#=IVTFF') or line.startswith('#='):
                continue

            # Comment
            if line.startswith('#'):
                if current_page:
                    current_page.comments.append(line[1:].strip())
                continue

            # Page header: <f1r>  <! $Q=A $P=A ... >
            page_match = re.match(r'^<(f\d+[rv]\d?)>\s*(.*)', line)
            if page_match:
                folio = page_match.group(1)
                rest = page_match.group(2)

                current_page = VoynichPage(folio)
                pages[folio] = current_page

                _parse_page_variables(current_page, rest)
                continue

            # Standalone variable line
            if line.strip().startswith('<!') and current_page:
                _parse_page_variables(current_page, line)
                continue

            # Locus line: <f1r.1,@P0>  text...
            locus_match = re.match(r'^<([^>]+)>\s*(.*)', line)
            if locus_match:
                locus_id = locus_match.group(1)
                text = locus_match.group(2)

                if not current_page:
                    skipped += 1
                    continue

                locus_type = _extract_locus_type(locus_id)

                locus = VoynichLocus(locus_id, locus_type, text)
                current_page.loci.append(locus)
                locus_count += 1
                continue

            skipped += 1

    if verbose:
        total_tokens = sum(len(p.all_tokens) for p in pages.values())
        total_chars = sum(len(p.all_text.replace(' ', '')) for p in pages.values())
        print(f"  Parsed {filepath}")
        print(f"  Pages: {len(pages)}")
        print(f"  Loci: {locus_count}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total characters: {total_chars}")
        print(f"  Lines read: {line_count}, skipped: {skipped}")

        lang_counts = Counter(p.language for p in pages.values() if p.language)
        print(f"  Language distribution: {dict(lang_counts)}")

        hand_counts = Counter(p.hand for p in pages.values() if p.hand)
        print(f"  Hand (scribe) distribution: {dict(hand_counts)}")

        section_counts = Counter(p.section for p in pages.values())
        print(f"  Section distribution: {dict(section_counts)}")

    return pages


def _parse_page_variables(page: VoynichPage, text: str):
    """Extract $Q, $L, $H, $I, $P, $C variables from page header."""
    var_patterns = {
        'Q': (r'\$Q=(\d+)', 'quire', int),
        'P': (r'\$P=(\d+)', 'page_num', int),
        'L': (r'\$L=([AB])', 'language', str),
        'H': (r'\$H=(\d+)', 'hand', int),
        'I': (r'\$I=(\w)', 'illustration', str),
        'C': (r'\$C=(\w+)', 'cluster', str),
    }

    for var_name, (pattern, attr, conv) in var_patterns.items():
        match = re.search(pattern, text)
        if match:
            setattr(page, attr, conv(match.group(1)))


def _extract_locus_type(locus_id: str) -> str:
    """
    Extract locus type from an IVTFF locus identifier.

    Examples:
        f1r.P1.1;H  -> P (paragraph)
        f1r.L1.1;H  -> L (label)
        f67r.C1.1;H -> C (circular)
        f67r.R1.1;H -> R (radial)
    """
    match = re.search(r'\.([PLCRTX])\d', locus_id)
    if match:
        return match.group(1)

    match = re.search(r'\.([A-Z])', locus_id)
    if match:
        return match.group(1)

    return 'P'


# ---------------------------------------------------------------------------
# High-Level Corpus Interface
# ---------------------------------------------------------------------------

class VoynichCorpus:
    """
    High-level corpus interface built from parsed IVTFF data.
    Provides filtered access by section, scribe, language, quire.
    """

    def __init__(self, pages: Dict[str, VoynichPage]):
        self.pages = pages

    @classmethod
    def from_file(cls, filepath: str, verbose: bool = False) -> 'VoynichCorpus':
        """Load corpus from an IVTFF file."""
        pages = parse_ivtff(filepath, verbose=verbose)
        return cls(pages)

    def get_text(self,
                 section: Optional[str] = None,
                 language: Optional[str] = None,
                 hand: Optional[int] = None,
                 quire: Optional[int] = None,
                 paragraph_only: bool = True) -> str:
        """Get filtered text as a single string."""
        parts = []
        for page in self.pages.values():
            if section and page.section != section:
                continue
            if language and page.language != language:
                continue
            if hand and page.hand != hand:
                continue
            if quire and page.quire != quire:
                continue

            text = page.paragraph_text if paragraph_only else page.all_text
            if text:
                parts.append(text)

        return ' '.join(parts)

    def get_tokens(self, **kwargs) -> List[str]:
        """Get filtered tokens."""
        return self.get_text(**kwargs).split()

    def get_page(self, folio: str) -> Optional[VoynichPage]:
        """Get a specific page."""
        return self.pages.get(folio)

    def get_pages_by_section(self, section: str) -> List[VoynichPage]:
        """Get all pages in a section."""
        return [p for p in self.pages.values() if p.section == section]

    def get_pages_by_hand(self, hand: int) -> List[VoynichPage]:
        """Get all pages by a specific scribe."""
        return [p for p in self.pages.values() if p.hand == hand]

    def get_pages_by_language(self, language: str) -> List[VoynichPage]:
        """Get all pages assigned to a Currier language ('A' or 'B')."""
        return [p for p in self.pages.values() if p.language == language]

    def get_folio_sequence(self, quire_order: Optional[List[int]] = None) -> List[VoynichPage]:
        """Get pages in quire ordering."""
        if quire_order is None:
            return sorted(
                self.pages.values(),
                key=lambda p: (p.quire, p.page_num)
            )

        ordered = []
        for q in quire_order:
            quire_pages = sorted(
                [p for p in self.pages.values() if p.quire == q],
                key=lambda p: p.page_num
            )
            ordered.extend(quire_pages)
        return ordered

    def summary(self) -> Dict:
        """Return corpus summary statistics."""
        all_tokens = self.get_tokens(paragraph_only=False)
        return {
            'total_pages': len(self.pages),
            'total_tokens': len(all_tokens),
            'total_characters': sum(len(t) for t in all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'type_token_ratio': len(set(all_tokens)) / max(1, len(all_tokens)),
            'languages': dict(Counter(
                p.language for p in self.pages.values() if p.language
            )),
            'hands': dict(Counter(
                p.hand for p in self.pages.values() if p.hand
            )),
            'sections': dict(Counter(
                p.section for p in self.pages.values()
            )),
            'quires': sorted(set(
                p.quire for p in self.pages.values() if p.quire
            )),
        }


def load_corpus(data_dir: str = None, verbose: bool = True) -> VoynichCorpus:
    """
    Load the best available corpus from the data directory.
    Prefers: ZL3b-n.txt > RF1b-e.txt > IT2a-n.txt
    """
    if data_dir is None:
        data_dir = str(_data_dir('corpus'))

    preferred = [
        'ZL3b-n.txt',
        'RF1b-e.txt',
        'IT2a-n.txt',
    ]

    for filename in preferred:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            if verbose:
                print(f"Loading corpus from {filepath}...")
            return VoynichCorpus.from_file(filepath, verbose=verbose)

    raise FileNotFoundError(
        f"No IVTFF files found in {data_dir}. "
        f"Copy ZL3b-n.txt into {data_dir}/"
    )


# ---------------------------------------------------------------------------
# Grid-cell decomposition helpers  (Phase 11 CSP)
# ---------------------------------------------------------------------------

def build_eva_to_cell_lookup(
    cv_labels: Dict[str, Dict],
) -> Dict[str, str]:
    """Build reverse lookup: EVA glyph -> grid cell key.

    *cv_labels* is the dict loaded from ``results/cv_labels.json`` where
    each key is a cell key (e.g. ``"loop,loop+sigmoid+tail"``) and the
    value contains a ``"glyphs"`` list.

    Returns e.g. ``{"a": "loop,loop+sigmoid+tail", ...}``.
    """
    lookup: Dict[str, str] = {}
    for cell_key, info in cv_labels.items():
        for glyph in info.get('glyphs', []):
            lookup[glyph] = cell_key
    return lookup


def token_to_grid_cells(
    token: str,
    eva_to_cell: Dict[str, str],
) -> List[str]:
    """Decompose a Voynich token into a sequence of grid cell keys.

    1. ``tokenize_eva_chars(token)`` -> list of EVA characters
    2. Each EVA char is mapped through *eva_to_cell*.
    3. Unknown chars (not in the lookup) are silently skipped.

    Returns a list of cell keys, one per recognised EVA character.
    """
    chars = tokenize_eva_chars(token)
    cells: List[str] = []
    for ch in chars:
        cell = eva_to_cell.get(ch)
        if cell is not None:
            cells.append(cell)
    return cells


def apply_character_moves(
    cv_labels: Dict[str, Any],
    moves: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Return a deep copy of *cv_labels* with glyph moves applied.

    Each move is ``{'eva_glyph': str, 'from_cell': str, 'to_cell': str}``.
    Glyphs are removed from ``from_cell['glyphs']`` and appended to
    ``to_cell['glyphs']``.  Cells that become empty are removed.
    Frequency counts are left unchanged (caller may recompute if needed).
    """
    import copy
    new_labels: Dict[str, Any] = copy.deepcopy(cv_labels)
    for move in moves:
        glyph = move['eva_glyph']
        from_cell = move['from_cell']
        to_cell = move['to_cell']
        if from_cell in new_labels and glyph in new_labels[from_cell].get('glyphs', []):
            new_labels[from_cell]['glyphs'].remove(glyph)
        if to_cell in new_labels and glyph not in new_labels[to_cell].get('glyphs', []):
            new_labels[to_cell]['glyphs'].append(glyph)
    # Remove cells that became empty
    empty = [k for k, v in new_labels.items() if not v.get('glyphs')]
    for k in empty:
        del new_labels[k]
    return new_labels


def token_to_grid_cells_alt(
    token: str,
    eva_to_cell: Dict[str, str],
    mode: str = 'default',
) -> List[str]:
    """Decompose a Voynich token into cell keys using an alternative mode.

    Modes
    -----
    'default'
        Same as :func:`token_to_grid_cells`.
    'split_aiin'
        Treat ``aiin`` / ``aiiin`` as two separate characters (``a`` + ``iin``
        / ``a`` + ``iiin``) rather than as a single ligature.
    'raw_chars'
        Tokenize without ligature merging — each raw EVA character is looked
        up individually.
    """
    if mode == 'default':
        return token_to_grid_cells(token, eva_to_cell)
    if mode == 'split_aiin':
        chars = tokenize_eva_chars(token)
        expanded: List[str] = []
        for ch in chars:
            if ch == 'aiin':
                expanded.extend(['a', 'iin'])
            elif ch == 'aiiin':
                expanded.extend(['a', 'iiin'])
            else:
                expanded.append(ch)
        return [eva_to_cell[c] for c in expanded if c in eva_to_cell]
    if mode == 'raw_chars':
        cells: List[str] = []
        for ch in token:
            cell = eva_to_cell.get(ch)
            if cell is not None:
                cells.append(cell)
        return cells
    raise ValueError(f"Unknown decomposition mode: {mode!r}")
