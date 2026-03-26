"""Render each EVA character from the TTF font as a 224x224 PNG.

The EVA font maps keyboard characters to Voynich glyphs.
Compound characters need specific rendering strings.
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Maps EVA character names to the string the font expects
EVA_RENDER_MAP = {
    # Single characters
    'a': 'a', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'F',
    'g': 'g', 'h': 'h', 'i': 'i', 'k': 'K', 'l': 'l',
    'm': 'm', 'n': 'n', 'o': 'o', 'p': 'P', 'q': 'q',
    'r': 'r', 's': 's', 't': 'T', 'y': 'y',
    # Compound characters
    'ch': 'ch', 'sh': 'Sh',
    'ckh': 'cKh', 'cth': 'cTh', 'cfh': 'cFh', 'cph': 'cPh',
    # Multi-minim
    'in': 'in', 'iin': 'iin', 'iiin': 'iiin',
    'aiin': 'aiin', 'aiiin': 'aiiin',
    # Others
    'dy': 'dy', 'ey': 'ey',
    'al': 'al', 'ol': 'ol', 'ar': 'ar', 'or': 'or', 'am': 'am',
}

# T_P15 syllable assignments for syllabic characters
T_P15 = {
    'a': 'ra', 'c': 'se', 'd': 'di', 'e': 'ne', 'f': 'fa',
    'g': 'de', 'h': 'ce', 'i': 'ni', 'k': 'de', 'l': 'la',
    'm': 'mi', 'n': 'ni', 'o': 'ro', 'p': 'be', 'q': 'cu',
    'r': 're', 's': 'so', 't': 'te', 'y': 'si',
    'ch': 'ca', 'sh': 'sa', 'ckh': 'da', 'cth': 'na',
    'cfh': 'ma', 'cph': 'pa',
}

# Coda assignments (Phase 60 corrected)
CODA_CHARS = {
    'aiin': 'n', 'aiiin': 'n', 'iin': 'n', 'iiin': 'n', 'n': 'n',
    'dy': 'r', 'ey': 'r', 'y': 'r',
    'ar': 's', 'or': 's',
    'al': 't', 'ol': 't', 'am': 't', 'm': 't',
    'h': 'r', 'ckh': 'r', 'u': 'r',
}


def render_eva_character(char_name, font_path, output_dir, size=200):
    """Render one EVA character to a 224x224 white-background PNG."""
    font_string = EVA_RENDER_MAP.get(char_name, char_name)

    font = ImageFont.truetype(font_path, size)

    # Render on large canvas
    canvas = Image.new('RGB', (size * 4, size * 3), 'white')
    draw = ImageDraw.Draw(canvas)

    bbox = draw.textbbox((0, 0), font_string, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (canvas.width - text_width) // 2 - bbox[0]
    y = (canvas.height - text_height) // 2 - bbox[1]
    draw.text((x, y), font_string, fill='black', font=font)

    # Crop to tight bounding box
    img_array = np.array(canvas)
    non_white = np.where(img_array < 240)
    if len(non_white[0]) == 0:
        return None

    y_min, y_max = non_white[0].min(), non_white[0].max()
    x_min, x_max = non_white[1].min(), non_white[1].max()

    # Add uniform padding
    pad = 20
    y_min = max(0, y_min - pad)
    y_max = min(canvas.height, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(canvas.width, x_max + pad)

    cropped = canvas.crop((x_min, y_min, x_max, y_max))

    # Resize to 224x224 maintaining aspect ratio, white padding
    target_size = 224
    cropped.thumbnail((target_size, target_size), Image.LANCZOS)
    final = Image.new('RGB', (target_size, target_size), 'white')
    offset = ((target_size - cropped.width) // 2,
              (target_size - cropped.height) // 2)
    final.paste(cropped, offset)

    output_path = os.path.join(output_dir, f'{char_name}.png')
    final.save(output_path)
    return output_path


def render_all_eva(font_path, output_dir):
    """Render all EVA characters and return metadata list."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for char_name in EVA_RENDER_MAP:
        path = render_eva_character(char_name, font_path, output_dir)
        if path:
            entry = {
                'eva_name': char_name,
                'image_path': path,
                'font_string': EVA_RENDER_MAP[char_name],
            }
            if char_name in T_P15:
                entry['t_p15_syllable'] = T_P15[char_name]
                entry['role'] = 'SYLLABIC'
            elif char_name in CODA_CHARS:
                entry['coda_value'] = CODA_CHARS[char_name]
                entry['role'] = 'CODA_MARKER'
            metadata.append(entry)

    return metadata
