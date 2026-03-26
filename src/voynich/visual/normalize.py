"""Normalize sign images to 224x224 PNGs for embedding comparison."""

import os

from PIL import Image


def normalize_image(img, target_size=224):
    """Normalize a PIL Image to target_size x target_size RGB on white background.

    Steps:
    1. Convert any transparency to white background
    2. Convert to RGB
    3. Resize maintaining aspect ratio
    4. Center on white canvas
    """
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, 'white')
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize maintaining aspect ratio with margin
    max_dim = target_size - 20
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    # Center on white canvas
    final = Image.new('RGB', (target_size, target_size), 'white')
    offset = ((target_size - img.width) // 2,
              (target_size - img.height) // 2)
    final.paste(img, offset)

    return final


def normalize_and_save(input_path, output_path, target_size=224):
    """Load, normalize, and save a single image."""
    img = Image.open(input_path)
    normalized = normalize_image(img, target_size)
    normalized.save(output_path)
    return output_path


def normalize_costamagna_crops(crops_dir, metadata, syllabary, output_dir,
                                target_size=224):
    """Normalize all Costamagna crops and enrich metadata.

    Args:
        crops_dir: Path to costamagna_crops/ directory
        metadata: List of dicts from metadata.json
        syllabary: List of dicts from syllabary_table.json
        output_dir: Where to save normalized PNGs
        target_size: Output image dimension

    Returns:
        List of enriched metadata dicts with normalized image paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build syllabary lookup
    syl_lookup = {}
    for entry in syllabary:
        syl_lookup[entry['syllable']] = entry

    result = []
    for entry in metadata:
        sym_path = os.path.join(crops_dir, entry['image_symbol'])
        if not os.path.exists(sym_path):
            continue

        syl = entry['syllable']
        out_name = f"costa_{syl}.png"
        out_path = os.path.join(output_dir, out_name)

        try:
            normalize_and_save(sym_path, out_path, target_size)
        except Exception as e:
            print(f"  Warning: failed to normalize {syl}: {e}")
            continue

        enriched = {
            'syllable': syl,
            'image_path': out_path,
            'normalized_filename': out_name,
            'tavola': entry.get('tavola', '?'),
            'row': entry.get('row', '?'),
            'column': entry.get('column', '?'),
            'variant': entry.get('variant', False),
        }

        # Enrich with phonological data
        syl_info = syl_lookup.get(syl, {})
        enriched['structure'] = syl_info.get('structure', '?')
        enriched['onset'] = syl_info.get('initial_consonant', None)
        enriched['vowel'] = syl_info.get('vowel', None)
        enriched['coda'] = syl_info.get('final_consonant', None)

        result.append(enriched)

    return result


def normalize_eva_renders(renders_dir, eva_metadata, output_dir,
                           target_size=224):
    """Re-normalize EVA renders through the same pipeline as Costamagna.

    Ensures both sets go through identical normalization.
    """
    os.makedirs(output_dir, exist_ok=True)

    result = []
    for entry in eva_metadata:
        src_path = entry['image_path']
        if not os.path.exists(src_path):
            continue

        out_name = f"eva_{entry['eva_name']}.png"
        out_path = os.path.join(output_dir, out_name)

        try:
            normalize_and_save(src_path, out_path, target_size)
        except Exception as e:
            print(f"  Warning: failed to normalize EVA {entry['eva_name']}: {e}")
            continue

        enriched = dict(entry)
        enriched['normalized_path'] = out_path
        enriched['normalized_filename'] = out_name
        result.append(enriched)

    return result
