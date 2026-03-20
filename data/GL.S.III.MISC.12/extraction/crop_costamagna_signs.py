#!/usr/bin/env python3
"""
crop_costamagna_signs.py

Extracts individual tachygraphic sign images from the plate photographs
in Costamagna (1953), using the structured catalog to label each crop.

The plates (Tavole I-XIII) use a regular 3-row × 6-column grid layout
inside a drawn rectangular border. Each cell contains a sign (upper ~60%)
and its syllable label in italics (lower ~40%).

Usage:
    python3 crop_costamagna_signs.py --input-dir /path/to/photos --output-dir /path/to/crops
    
Output structure:
    output_dir/
        alphabet/          # Individual letter signs from p.7
        signs/             # Individual syllable signs from Tavole I-XIII
            tav_01_r1c1_ac.png
            tav_01_r1c2_ad-at.png
            ...
        plates/            # Full plate crops (border region only)
        combination_rules/ # Example signs from pp.8-12
"""

import json
import os
import sys
import argparse
from pathlib import Path

try:
    from PIL import Image, ImageFilter, ImageOps
    import numpy as np
except ImportError:
    print("Requires: pip install Pillow numpy --break-system-packages")
    sys.exit(1)


# Plate image filenames mapped to tavola numbers
PLATE_MAP = {
    "GL_S_III_MISC_12_15.jpg": ("I", 1),
    "GL_S_III_MISC_12_16.jpg": ("II", 2),
    "GL_S_III_MISC_12_17.jpg": ("III", 3),
    "GL_S_III_MISC_12_18.jpg": ("IV", 4),
    "GL_S_III_MISC_12_19.jpg": ("V", 5),
    "GL_S_III_MISC_12_20.jpg": ("VI", 6),
    "GL_S_III_MISC_12_21.jpg": ("VII", 7),
    "GL_S_III_MISC_12_22.jpg": ("VIII", 8),
    "GL_S_III_MISC_12_23.jpg": ("IX", 9),
    "GL_S_III_MISC_12_24.jpg": ("X", 10),
    "GL_S_III_MISC_12_25.jpg": ("XI", 11),
    "GL_S_III_MISC_12_26.jpg": ("XII", 12),
    "GL_S_III_MISC_12_27.jpg": ("XIII", 13),
}

# Syllable values for each plate, row by row (from the catalog)
PLATE_SYLLABLES = {
    1:  [["ac","ad-at","al","an","ar","as"],
         ["atque","au","ba","bal","bar","bas"],
         ["be","bel","bem","ber","bi","bis"]],
    2:  [["bli","bo","bro","bru","bu","bus"],
         ["ca","cal","cam","can","car","cas"],
         ["ce","ce_v2","cen","ces","ci","cin"]],
    3:  [["cit","cle","co","con","cri","cu"],
         ["cum","cum_v2","cus","da","dal","dan"],
         ["de","del","des","dex","di","do"]],
    4:  [["don","dos","dra","dre","dri","du"],
         ["dul","dus","e","e_v2","er","est"],
         ["eu","fa","fe","fer","fi","flu"]],
    5:  [["fo","fre","fro","fu","ful","fun"],
         ["fus","ga","gau","ge","gel","gen"],
         ["ger","gi","gin","gle","gis","go"]],
    6:  [["gum","gun","hi","ho","ha","ia"],
         ["il","in","io","ip","is","iu"],
         ["la","lan","las","le","ler","lis"]],
    7:  [["lo","lon","ma","mal","mar","mas"],
         ["mau","me-mi","mel","men","mer","mo"],
         ["mu","mul","mus","na","nam","nar"]],
    8:  [["ne-ni","ner","nes","nes_v2","nit","nis"],
         ["no","nos","nu","nul","num","num_v2"],
         ["nus","oc","om","os","pa","par"]],
    9:  [["pau","pe","per","pi","pl","ple"],
         ["po","pon","por","post","pra","pri"],
         ["pro","pu","quar","que","qui","quin"]],
    10: [["quod","ra","rar","re","rer","res"],
         ["rex","ri","ri_v2","rim","ris","rit"],
         ["ro","ru","sa","san","sco","se"]],
    11: [["sel","sep","ser","ses","si","sis"],
         ["so","stat","ste","sti","sto","stri"],
         ["stris","stro","su","sum","sunt","sunt_v2"]],
    12: [["super","supra","ta","tar","te","tem"],
         ["tes","ter","ti","tis","to","tor"],
         ["tr","tra","tre","tres","tri","tru"]],
    13: [["tu","tul","tum","tum_v2","tur","tus"],
         ["u","ue","uel","uer","ui","uir"],
         ["um","uo","ur","us","uui","uua"],
         ["uual","zo"]],  # partial 4th row
}


def get_plate_border(img_w, img_h, tav_num):
    """
    Return (x_min, y_min, x_max, y_max) for the sign grid INTERIOR.
    
    Derived from peak detection on Tav I: column centers at 17.3%, 26.8%, 
    41.6%, 50.3%, 65.9%, 78.2% → evenly-spaced grid from ~11% to ~84%.
    Row positions estimated from visual inspection of plate images.
    """
    if tav_num == 13:
        return (int(img_w * 0.10), int(img_h * 0.14),
                int(img_w * 0.82), int(img_h * 0.84))
    else:
        return (int(img_w * 0.10), int(img_h * 0.14),
                int(img_w * 0.82), int(img_h * 0.80))


def crop_grid_cell(img, border, row, col, n_rows, n_cols, sign_fraction=0.65):
    """
    Crop a single cell from the grid, returning only the sign portion
    (upper part of the cell, excluding the label text below).
    
    sign_fraction: what fraction of each cell's height contains the sign
                   (vs. the label text below it)
    """
    x_min, y_min, x_max, y_max = border
    cell_w = (x_max - x_min) / n_cols
    cell_h = (y_max - y_min) / n_rows
    
    cx = x_min + col * cell_w
    cy = y_min + row * cell_h
    
    # Crop the sign portion (upper part of cell)
    left = int(cx)
    top = int(cy)
    right = int(cx + cell_w)
    bottom = int(cy + cell_h * sign_fraction)
    
    return img.crop((left, top, right, bottom))


def crop_plate(input_dir, output_dir, catalog_path=None):
    """Main cropping pipeline."""
    signs_dir = os.path.join(output_dir, "signs")
    plates_dir = os.path.join(output_dir, "plates")
    os.makedirs(signs_dir, exist_ok=True)
    os.makedirs(plates_dir, exist_ok=True)
    
    # Also produce a flat CSV for easy machine consumption
    csv_rows = [["filename", "tavola", "row", "col", "syllable", "is_variant", "notes"]]
    
    for filename, (tav_roman, tav_num) in PLATE_MAP.items():
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP: {filename} not found")
            continue
        
        print(f"Processing Tavola {tav_roman} ({filename})...")
        img = Image.open(filepath).convert("RGB")
        w, h = img.size
        
        # Get the plate border coordinates
        border = get_plate_border(w, h, tav_num)
        x_min, y_min, x_max, y_max = border
        
        # Save the full plate crop
        plate_crop = img.crop((x_min, y_min, x_max, y_max))
        plate_crop.save(os.path.join(plates_dir, f"tav_{tav_num:02d}_{tav_roman}.png"))
        
        # Get the syllable grid for this plate
        syllables = PLATE_SYLLABLES[tav_num]
        n_rows = len(syllables)
        n_cols = max(len(row) for row in syllables)
        
        for r, row_syllables in enumerate(syllables):
            for c, syl in enumerate(row_syllables):
                # Determine if this is a variant
                is_variant = "_v2" in syl
                clean_syl = syl.replace("_v2", "")
                
                # Crop the sign
                try:
                    sign_img = crop_grid_cell(img, border, r, c, n_rows, n_cols)
                except Exception as e:
                    print(f"  Error cropping {tav_roman} r{r+1}c{c+1} ({syl}): {e}")
                    continue
                
                # Build filename
                variant_tag = "_var" if is_variant else ""
                safe_syl = clean_syl.replace("-", "_").replace("·", "_")
                fname = f"tav_{tav_num:02d}_r{r+1}c{c+1}_{safe_syl}{variant_tag}.png"
                sign_img.save(os.path.join(signs_dir, fname))
                
                notes = ""
                if is_variant:
                    notes = "variant form"
                if clean_syl in ("atque", "est", "que", "qui", "quod", "super", "supra"):
                    notes = "Tironian sigla"
                
                csv_rows.append([fname, tav_roman, r+1, c+1, clean_syl, is_variant, notes])
    
    # Write the CSV index
    csv_path = os.path.join(output_dir, "sign_index.csv")
    with open(csv_path, "w") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    
    print(f"\nDone. {len(csv_rows)-1} signs cropped.")
    print(f"CSV index: {csv_path}")
    
    return csv_path


def build_syllabary_table(output_dir):
    """
    Build a flat lookup table of all unique syllable values
    with their structural properties, for use as CSP priors.
    """
    all_syllables = set()
    for tav_num, grid in PLATE_SYLLABLES.items():
        for row in grid:
            for syl in row:
                clean = syl.replace("_v2", "")
                all_syllables.add(clean)
    
    table = []
    for syl in sorted(all_syllables):
        entry = {
            "syllable": syl,
            "length": len(syl),
            "structure": classify_syllable(syl),
            "initial_consonant": get_onset(syl),
            "vowel": get_nucleus(syl),
            "final_consonant": get_coda(syl),
        }
        table.append(entry)
    
    path = os.path.join(output_dir, "syllabary_table.json")
    with open(path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Syllabary table ({len(table)} entries): {path}")
    return table


def classify_syllable(syl):
    """Classify syllable structure: CV, CVC, CCV, CCVC, V, VC, sigla, etc."""
    # Handle special cases
    sigla = {"atque", "est", "que", "qui", "quod", "super", "supra", "post", "stat", "sunt"}
    if syl in sigla:
        return "sigla"
    
    vowels = set("aeiou")
    chars = list(syl)
    
    # Build a C/V pattern
    pattern = ""
    for ch in chars:
        pattern += "V" if ch in vowels else "C"
    
    return pattern


def get_onset(syl):
    """Extract the onset (initial consonant cluster) of a syllable."""
    vowels = set("aeiou")
    onset = ""
    for ch in syl:
        if ch in vowels:
            break
        onset += ch
    return onset if onset else None


def get_nucleus(syl):
    """Extract the nucleus (vowel) of a syllable."""
    vowels = set("aeiou")
    nucleus = ""
    in_vowel = False
    for ch in syl:
        if ch in vowels:
            nucleus += ch
            in_vowel = True
        elif in_vowel:
            break
    return nucleus if nucleus else None


def get_coda(syl):
    """Extract the coda (final consonant cluster) of a syllable."""
    vowels = set("aeiou")
    # Find the last vowel position
    last_v = -1
    for i, ch in enumerate(syl):
        if ch in vowels:
            last_v = i
    if last_v == -1 or last_v == len(syl) - 1:
        return None
    return syl[last_v + 1:]


def main():
    parser = argparse.ArgumentParser(description="Crop tachygraphic signs from Costamagna (1953) plates")
    parser.add_argument("--input-dir", default="/mnt/user-data/uploads",
                        help="Directory containing the GL_S_III_MISC_12_*.jpg files")
    parser.add_argument("--output-dir", default="/home/claude/costamagna_crops",
                        help="Output directory for cropped signs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Costamagna (1953) Sign Extraction ===\n")
    
    # Crop individual signs from plates
    crop_plate(args.input_dir, args.output_dir)
    
    # Build the syllabary lookup table
    build_syllabary_table(args.output_dir)
    
    print("\nAll outputs in:", args.output_dir)


if __name__ == "__main__":
    main()
