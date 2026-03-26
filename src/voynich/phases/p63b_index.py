"""Phase 63B Step B1: Extract and index folio images from zip."""

import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from voynich.core._paths import data_dir, results_dir


@dataclass
class FolioEntry:
    folio_id: str = ''
    image_path: str = ''
    image_filename: str = ''
    has_transcription: bool = False
    n_paragraph_lines: int = 0
    n_words: int = 0
    section: str = ''
    language: str = ''
    hand: int = 0


@dataclass
class IndexResult:
    n_images: int = 0
    n_with_transcription: int = 0
    n_paragraph_folios: int = 0
    folios_dir: str = ''
    entries: List[FolioEntry] = field(default_factory=list)
    missing_images: List[str] = field(default_factory=list)
    elapsed: float = 0.0


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
    return obj


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


def run_p63b_index():
    """Extract folio images from zip and build index."""
    t0 = time.time()
    rd = str(results_dir())

    raw_dir = str(data_dir('voynich_raw'))
    zip_path = os.path.join(raw_dir, 'voynich.zip')
    mapping_path = os.path.join(raw_dir, 'voynich_folio_mapping.json')
    folios_dir = os.path.join(raw_dir, 'folios')

    if not os.path.exists(zip_path):
        print(f"ERROR: voynich.zip not found at {zip_path}")
        return
    if not os.path.exists(mapping_path):
        print(f"ERROR: voynich_folio_mapping.json not found at {mapping_path}")
        return

    # Load mapping
    with open(mapping_path) as f:
        mapping = json.load(f)

    file_to_folio = mapping.get('file_to_folio', {})
    folio_to_file = mapping.get('folio_to_file', {})

    # Extract zip if not already done
    if os.path.exists(folios_dir) and len(os.listdir(folios_dir)) > 0:
        print(f"Phase 63B B1: Folios already extracted ({len(os.listdir(folios_dir))} files)")
    else:
        print(f"Phase 63B B1: Extracting {zip_path}...")
        os.makedirs(folios_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(folios_dir)
        print(f"  Extracted to {folios_dir}")

    # List extracted files
    extracted = set(os.listdir(folios_dir))
    n_images = len([f for f in extracted if f.endswith('.jpg')])
    print(f"  Found {n_images} JPG files")

    # Load IVTFF corpus for cross-reference
    from voynich.core.corpus import load_corpus
    corpus = load_corpus()

    # Build index: match images to transcription folios
    entries = []
    missing = []

    # Map corpus folio IDs (like 'f67r') to sub-page IDs in mapping (like 'f67r1', 'f67r2')
    corpus_folios = set(corpus.pages.keys())

    for folio_id, filename in folio_to_file.items():
        image_path = os.path.join(folios_dir, filename)

        if not os.path.exists(image_path):
            missing.append(folio_id)
            continue

        # Match to corpus: try exact match first, then base folio
        page = corpus.pages.get(folio_id)
        base_folio = folio_id.rstrip('0123456789') if folio_id[-1].isdigit() and not folio_id.endswith('r') and not folio_id.endswith('v') else folio_id
        if page is None and base_folio != folio_id:
            page = corpus.pages.get(base_folio)

        has_trans = page is not None
        n_para = 0
        n_words = 0
        section = ''
        language = ''
        hand = 0

        if page is not None:
            para_loci = [l for l in page.loci if l.locus_type.startswith('P')]
            n_para = len(para_loci)
            n_words = len(page.all_tokens)
            section = getattr(page, 'section', '')
            language = getattr(page, 'language', '')
            hand = getattr(page, 'hand', 0)

        entry = FolioEntry(
            folio_id=folio_id,
            image_path=image_path,
            image_filename=filename,
            has_transcription=has_trans,
            n_paragraph_lines=n_para,
            n_words=n_words,
            section=section,
            language=language,
            hand=hand,
        )
        entries.append(entry)

    with_trans = [e for e in entries if e.has_transcription]
    with_para = [e for e in entries if e.n_paragraph_lines > 0]

    result = IndexResult(
        n_images=len(entries),
        n_with_transcription=len(with_trans),
        n_paragraph_folios=len(with_para),
        folios_dir=folios_dir,
        entries=entries,
        missing_images=missing,
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63b_index.json', asdict(result))

    print(f"\n  Images indexed: {result.n_images}")
    print(f"  With transcription: {result.n_with_transcription}")
    print(f"  With paragraph text: {result.n_paragraph_folios}")
    print(f"  Missing images: {len(missing)}")
    if missing:
        print(f"  Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    print(f"  Elapsed: {result.elapsed:.1f}s")
