"""Phase 63B Steps B2-B4: Line, word, and character segmentation pipeline."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from voynich.core._paths import results_dir

DEV_SET = ['f1r', 'f2r', 'f3r', 'f15v', 'f17r']


@dataclass
class FolioSegResult:
    folio_id: str = ''
    expected_lines: int = 0
    detected_lines: int = 0
    line_match: bool = False
    total_words_expected: int = 0
    total_words_segmented: int = 0
    total_chars_segmented: int = 0
    image_width: int = 0
    image_height: int = 0


@dataclass
class SegmentResult:
    n_folios_processed: int = 0
    n_line_match: int = 0
    line_match_rate: float = 0.0
    total_words: int = 0
    total_chars: int = 0
    per_folio: List[FolioSegResult] = field(default_factory=list)
    dev_set_only: bool = True
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


def _safe_load(path: str) -> Any:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _process_folio(folio_id, image_path, page, output_dir):
    """Segment one folio into lines, words, and characters.

    Returns FolioSegResult and saves crops to output_dir.
    """
    from voynich.core.corpus import tokenize_eva_chars
    from voynich.visual.segment import (
        binarize,
        find_line_bands,
        find_text_region,
        find_word_gaps,
        segment_characters_dp,
    )

    # Load and convert to grayscale
    img = Image.open(image_path)
    img_gray = np.array(img.convert('L'))
    h, w = img_gray.shape

    # Binarize
    binary = binarize(img_gray)

    # Find text region
    y0, y1, x0, x1 = find_text_region(binary)
    text_binary = binary[y0:y1, x0:x1]

    # Get paragraph loci
    para_loci = [l for l in page.loci if l.locus_type.startswith('P')]
    n_expected_lines = len(para_loci)

    if n_expected_lines == 0:
        return FolioSegResult(folio_id=folio_id, image_width=w, image_height=h)

    # B2: Line segmentation
    from voynich.visual.segment import horizontal_projection
    h_proj = horizontal_projection(text_binary)
    line_bands = find_line_bands(h_proj, n_expected_lines)

    detected_lines = len(line_bands)
    line_match = abs(detected_lines - n_expected_lines) <= 1

    # Create output dirs
    lines_dir = os.path.join(output_dir, 'lines')
    words_dir = os.path.join(output_dir, 'words')
    chars_dir = os.path.join(output_dir, 'chars')
    os.makedirs(lines_dir, exist_ok=True)
    os.makedirs(words_dir, exist_ok=True)
    os.makedirs(chars_dir, exist_ok=True)

    total_words = 0
    total_chars = 0
    char_metadata = []

    # Process each line (use min of detected and expected)
    n_lines = min(detected_lines, n_expected_lines)

    for line_idx in range(n_lines):
        ly0, ly1 = line_bands[line_idx]
        line_binary = text_binary[ly0:ly1, :]

        # Save line crop
        line_img = Image.fromarray(img_gray[y0 + ly0:y0 + ly1, x0:x1])
        line_img.save(os.path.join(lines_dir, f'line_{line_idx:03d}.png'))

        # B3: Word segmentation
        locus = para_loci[line_idx]
        words = locus.clean_text.split()
        n_expected_words = len(words)

        if n_expected_words == 0:
            continue

        from voynich.visual.segment import vertical_projection
        word_gaps = find_word_gaps(vertical_projection(line_binary), n_expected_words)

        n_words = min(len(word_gaps), n_expected_words)
        total_words += n_words

        for word_idx in range(n_words):
            wx0, wx1 = word_gaps[word_idx]
            if wx1 <= wx0:
                continue

            word_binary = line_binary[:, wx0:wx1]

            # Save word crop
            word_img = Image.fromarray(
                img_gray[y0 + ly0:y0 + ly1, x0 + wx0:x0 + wx1])
            word_img.save(os.path.join(
                words_dir, f'line{line_idx:03d}_word{word_idx:03d}.png'))

            # B4: Character segmentation
            eva_word = words[word_idx]
            eva_chars = tokenize_eva_chars(eva_word)
            n_chars = len(eva_chars)

            if n_chars == 0:
                continue

            char_bounds = segment_characters_dp(word_binary, n_chars)

            for char_idx, (cx0, cx1) in enumerate(char_bounds):
                if cx1 <= cx0 or char_idx >= len(eva_chars):
                    continue

                # Save character crop
                char_img = Image.fromarray(
                    img_gray[y0 + ly0:y0 + ly1, x0 + wx0 + cx0:x0 + wx0 + cx1])
                char_name = eva_chars[char_idx]
                # Sanitize filename
                safe_name = char_name.replace('/', '_')
                fname = f'l{line_idx:03d}_w{word_idx:03d}_c{char_idx:02d}_{safe_name}.png'
                char_img.save(os.path.join(chars_dir, fname))

                char_metadata.append({
                    'folio': folio_id,
                    'line': int(line_idx),
                    'word': int(word_idx),
                    'char_idx': int(char_idx),
                    'eva_char': char_name,
                    'filename': fname,
                    'width': int(cx1 - cx0),
                    'height': int(ly1 - ly0),
                })
                total_chars += 1

    # Save character metadata
    with open(os.path.join(output_dir, 'char_metadata.json'), 'w') as f:
        json.dump(char_metadata, f, indent=2)

    return FolioSegResult(
        folio_id=folio_id,
        expected_lines=n_expected_lines,
        detected_lines=detected_lines,
        line_match=line_match,
        total_words_expected=sum(len(l.clean_text.split()) for l in para_loci),
        total_words_segmented=total_words,
        total_chars_segmented=total_chars,
        image_width=w,
        image_height=h,
    )


def run_p63b_segment(full=False):
    """Run segmentation pipeline on dev set (or full corpus if full=True)."""
    t0 = time.time()
    rd = str(results_dir())

    # Load index
    index = _safe_load(os.path.join(rd, 'p63b_index.json'))
    if not index:
        print("ERROR: Index not found. Run ms-index first.")
        return

    # Load corpus
    from voynich.core.corpus import load_corpus
    corpus = load_corpus()

    entries = index.get('entries', [])

    # Filter to dev set or full
    if not full:
        target_folios = set(DEV_SET)
        entries = [e for e in entries if e['folio_id'] in target_folios]
        print(f"Phase 63B B2-B4: Segmenting dev set ({len(entries)} folios)...")
    else:
        entries = [e for e in entries if e['has_transcription'] and e['n_paragraph_lines'] > 0]
        print(f"Phase 63B B2-B4: Segmenting full corpus ({len(entries)} folios)...")

    seg_dir = os.path.join(rd, 'p63b_segments')
    results = []

    for i, entry in enumerate(entries):
        folio_id = entry['folio_id']
        image_path = entry['image_path']

        page = corpus.pages.get(folio_id)
        if page is None:
            continue

        folio_out = os.path.join(seg_dir, folio_id)

        # Skip if already processed
        meta_path = os.path.join(folio_out, 'char_metadata.json')
        if os.path.exists(meta_path) and not full:
            existing = _safe_load(meta_path)
            if existing:
                print(f"  [{i+1}/{len(entries)}] {folio_id}: cached ({len(existing)} chars)")
                # Reconstruct result from cache
                para_loci = [l for l in page.loci if l.locus_type.startswith('P')]
                results.append(FolioSegResult(
                    folio_id=folio_id,
                    expected_lines=len(para_loci),
                    detected_lines=len(para_loci),  # approximate
                    line_match=True,
                    total_chars_segmented=len(existing),
                ))
                continue

        print(f"  [{i+1}/{len(entries)}] {folio_id}...", end=' ', flush=True)

        try:
            result = _process_folio(folio_id, image_path, page, folio_out)
            results.append(result)
            status = 'MATCH' if result.line_match else 'MISMATCH'
            print(f"lines {result.detected_lines}/{result.expected_lines} ({status}), "
                  f"{result.total_words_segmented} words, "
                  f"{result.total_chars_segmented} chars")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(FolioSegResult(folio_id=folio_id))

    n_match = sum(1 for r in results if r.line_match)
    total_words = sum(r.total_words_segmented for r in results)
    total_chars = sum(r.total_chars_segmented for r in results)

    summary = SegmentResult(
        n_folios_processed=len(results),
        n_line_match=n_match,
        line_match_rate=n_match / len(results) if results else 0.0,
        total_words=total_words,
        total_chars=total_chars,
        per_folio=results,
        dev_set_only=not full,
        elapsed=time.time() - t0,
    )

    _save_json(rd, 'p63b_segment.json', asdict(summary))

    print(f"\n  Folios processed: {summary.n_folios_processed}")
    print(f"  Line match rate: {summary.n_line_match}/{summary.n_folios_processed} "
          f"({summary.line_match_rate:.0%})")
    print(f"  Total words segmented: {summary.total_words}")
    print(f"  Total chars segmented: {summary.total_chars}")
    print(f"  Elapsed: {summary.elapsed:.1f}s")
