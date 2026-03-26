# Phase 63: Visual Sign Comparison via Multimodal Embeddings

[← Phase Index](README.md) | [← Progression Table](../progression.md)

## Question

Do Voynich characters visually resemble the Costamagna tachygraphic signs they are proposed to encode under T_P15?

## Method

Two independent workstreams using Gemini Embedding 2 (`gemini-embedding-2-preview`, 768 dimensions) to compute cosine similarity between EVA character images and 236 Costamagna sign crops from the 1953 syllabary plates.

- **Workstream A (font-based):** 37 EVA characters rendered from the Voynich EVA Hand A TTF font at 224x224, compared against normalized Costamagna crops.
- **Workstream B (manuscript-based):** Real character crops segmented from high-res Beinecke 2014 folio scans (Internet Archive) via projection-profile forced alignment using IVTFF transcriptions, compared against the same Costamagna embeddings.

### Validation Framework

Five A-gates test whether T_P15 assignments correlate with visual similarity:

| Gate | Threshold |
|------|-----------|
| A-G1 | >= 8/25 T_P15 assignments rank in top-5 visually |
| A-G2 | >= 15/25 assignments rank in top-15 |
| A-G3 | Permutation test p < 0.05 |
| A-G4 | Family clustering z > 1.65 |
| A-G5 | >= 3 confirmed syllables in top-3 |

Four B-gates test segmentation quality and whether manuscript-based comparison improves over font-based:

| Gate | Threshold |
|------|-----------|
| B-G1 | >= 80% folios line count within +/-1 |
| B-G2 | >= 70% words correctly segmented |
| B-G3 | >= 60% character types have exemplars |
| B-G4 | >= 2 more A-gates than font, or Spearman r > 0.3 |

## Results

### Workstream A: Font-Based Comparison

**Similarity statistics:** Mean max cosine similarity 0.874 +/- 0.025. All 37x236 pairwise similarities fall in range 0.78-0.90. The embedding model does not discriminate between signs.

**T_P15 validation:** 0/25 STRONG, 1/25 MODERATE (EVA `cth` -> `na`, rank 9), 0/25 WEAK, 24/25 NONE (rank 236). The one moderate hit is a size/complexity artifact (compound gallows all match `atque`).

| Gate | Value | Result |
|------|-------|--------|
| A-G1: Top-5 | 0/25 | FAIL |
| A-G2: Top-15 | 1/25 | FAIL |
| A-G3: Permutation | p = 0.379 | FAIL |
| A-G4: Family clustering | z = 1.47 (p = 0.071) | FAIL |
| A-G5: Confirmed top-3 | 0 | FAIL |

**Verdict: VISUAL_NO_SIGNAL (0/5 gates)**

Family clustering z = 1.47 (p = 0.071) is near-significant, consistent with known EVA sign family structure (bench, gallows, minim groups share visual elements).

### Workstream B: Manuscript-Based Comparison

**Segmentation pipeline (dev set: f1r, f2r, f3r, f15v, f17r):**

| Folio | Expected Lines | Detected | Match | Words | Characters |
|-------|---------------|----------|-------|-------|------------|
| f1r | 28 | 23 | MISMATCH | 149 | 490 |
| f2r | 15 | 15 | MATCH | 81 | 249 |
| f3r | 20 | 20 | MATCH | 78 | 256 |
| f15v | 12 | 12 | MATCH | 50 | 132 |
| f17r | 13 | 13 | MATCH | 45 | 162 |

Line match rate 4/5 (80%). Word segmentation 403/555 (73%). 1,289 characters segmented.

**Exemplar selection:** 22/37 character types produced >= 3 exemplars (211 total). 15 rare types filtered out.

**Manuscript vs Costamagna comparison:** 0/16 STRONG, 0/16 MODERATE, 0/16 WEAK, 16/16 NONE. Manuscript exemplars perform worse than font renders (additional parchment/segmentation noise without discriminative detail).

| Gate | Value | Result |
|------|-------|--------|
| B-G1: Line segmentation | 80% (4/5) | PASS |
| B-G2: Word segmentation | 73% (403/555) | PASS |
| B-G3: Character types | 59% (22/37) | FAIL |
| B-G4: Informative | +0 A-gates, r = 0.0 | FAIL |

**Verdict: VISUAL_MISMATCH (B: 2/4, A-ms: 0/5)**

### Combined Verdict

| Workstream | A-gates | B-gates | Verdict |
|------------|---------|---------|---------|
| A (font) | 0/5 | -- | VISUAL_NO_SIGNAL |
| B (manuscript) | 0/5 | 2/4 | VISUAL_MISMATCH |

## Interpretation

The failure is **methodological, not data-quality**:

1. **Similarity compression:** All pairwise cosine similarities fall in 0.78-0.90 (spread 0.12, std 0.025). The embedding model treats all small handwritten marks as near-identical.

2. **Updated crops didn't help:** After improving Costamagna crops (tighter bounding boxes, better centering), the permutation test moved from p=0.75 to p=0.38 -- directionally better but still non-significant.

3. **Manuscript exemplars performed worse than font:** Real manuscript crops introduced parchment texture and segmentation noise without adding discriminative stroke detail. This rules out "font abstraction" as the primary cause.

**Conclusion:** Gemini Embedding 2 is a general-purpose multimodal embedding model. It does not capture fine-grained stroke-level features (pen direction, loop geometry, descender shape) that distinguish tachygraphic signs. The visual correspondence hypothesis remains untested -- this phase demonstrates that multimodal embeddings are the wrong tool, not that the correspondence is absent.

## What Would Work

1. **Explicit stroke feature extraction** -- count loops, measure ascender/descender ratios, classify curve directions, then compare feature vectors
2. **Vision LLM sign description** -- prompt a vision model to describe each sign's morphology in structured text, then compare descriptions
3. **Contrastive fine-tuning** -- train a small embedding model on known paleographic sign pairs

## Commands

```bash
# Workstream A
voynich vis-render        # A1: Render EVA glyphs from font
voynich vis-normalize     # A2: Normalize images
voynich vis-embed         # A3: Embed via Gemini
voynich vis-similarity    # A4: Compute similarity matrix
voynich vis-validate      # A5: Validate T_P15
voynich vis-report        # A6: Generate HTML report
voynich phase63-verdict   # Integration verdict
voynich phase63           # Full A pipeline
voynich vis-rerun         # Re-run A2-A6 with updated crops

# Workstream B
voynich ms-index          # B1: Extract and index folios
voynich ms-segment        # B2-B4: Segment lines/words/chars
voynich ms-exemplars      # B5: Select character exemplars
voynich ms-compare        # B6: Embed and compare
voynich phase63b-verdict  # Integration verdict
voynich phase63b          # Full B pipeline
```

## Files

### Visual utilities
- `src/voynich/visual/__init__.py`
- `src/voynich/visual/render_eva.py` -- EVA font rendering + T_P15/CODA tables
- `src/voynich/visual/normalize.py` -- Image normalization to 224x224
- `src/voynich/visual/embed.py` -- Gemini Embedding 2 client with retry + .env loading
- `src/voynich/visual/similarity.py` -- Cosine similarity, ranking, family cohesion, permutation test
- `src/voynich/visual/segment.py` -- Line/word/character segmentation via projection profiles + DP

### Phase modules
- `src/voynich/phases/p63_render.py` through `p63_integrate.py` (Workstream A, 7 files)
- `src/voynich/phases/p63b_index.py` through `p63b_integrate.py` (Workstream B, 5 files)

### Data
- `data/Voynich EVA Hand A.ttf` -- EVA font
- `data/GL.S.III.MISC.12/costamagna_crops/` -- 236 Costamagna sign crops + metadata
- `data/voynich_raw/voynich.zip` -- 213 high-res Beinecke folio scans
- `data/voynich_raw/voynich_folio_mapping.json` -- Folio-to-file mapping

### Results
- `results/p63_embeddings.npz` -- 509 embeddings (37 EVA + 236x2 Costamagna)
- `results/p63_visual_report.html` -- Workstream A visual report
- `results/p63b_visual_report.html` -- Workstream B visual report
- `results/p63_integrate.json` -- Workstream A verdict
- `results/p63b_integrate.json` -- Workstream B verdict

## API Cost

| Batch | Images | Cost |
|-------|--------|------|
| Workstream A (3 batches) | 509 | $0.061 |
| Workstream B (exemplars) | 211 | $0.025 |
| **Total** | **720** | **$0.086** |
