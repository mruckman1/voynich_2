# Phase 64: Multi-Method Visual Sign Comparison

[← Phase Index](README.md)

## Motivation

Phase 63 failed because Gemini Embedding 2 compressed all handwritten glyphs into a 0.874 +/- 0.025 cosine similarity band — the model cannot distinguish tachygraphic signs from each other. Phase 64 replaces holistic image embeddings with 7 independent methods that capture what actually distinguishes signs: stroke construction, topology, and geometry. A rank-fusion ensemble combines all methods.

## Methods

### Tier 1: LLM-Based (Gemini 3.1 Pro via OpenRouter)

| Method | File | API Calls | Captures |
|--------|------|-----------|----------|
| M1: Structured Morphology | `morphology_description.py` | 273 | Entry direction, stroke types, loops, terminals, complexity |
| M7: Direct Pairwise | `llm_pairwise.py` | 125 | Expert-like structural judgment with reasoning |

### Tier 2: Classical Computer Vision (No API)

| Method | File | Captures |
|--------|------|----------|
| M2: Skeleton Graphs | `stroke_extraction.py` | Endpoints, junctions, loops, stroke angles, direction histogram |
| M3: Shape Descriptors | `shape_descriptors.py` | 7 Hu moments + 20 Fourier descriptors + 6 geometric features |
| M4: Topology | `topological_features.py` | Components, holes, Euler number (coarse filter) |
| M5: HOG | `hog_features.py` | Local stroke orientations (9 bins, 32px cells) |
| M6: Hybrid | `hybrid_features.py` | All of M2-M5 combined, HOG PCA-reduced to 50 dims |

### Tier 3: Ensemble

Rank fusion across all available distance matrices. For each method, distances are converted to ranks; the ensemble is the weighted mean rank.

## Results

### Verdict: WEAK_SUPPORT (2/7 gates)

| Gate | Threshold | Value | Result |
|------|-----------|-------|--------|
| G1 Ensemble top-5 | >=5/25 | 1 | FAIL |
| G2 Ensemble top-15 | >=12/25 | 2 | FAIL |
| G3 LLM pairwise win rate | >50% | **56%** | **PASS** |
| G4 Method discrimination | >=2 methods spread >0.3 | **6/6** | **PASS** |
| G5 Permutation test | p<0.05 | p=0.138 | FAIL |
| G6 LLM structural match | >=10/25 same_basic_structure | 0 | FAIL |
| G7 Topology compatible | >=15/25 | 9 | FAIL |

### Per-Method Performance

| Method | Mean T_P15 Rank | Strong (<=5) | Moderate (<=15) | Spread |
|--------|----------------|-------------|----------------|--------|
| M2 Graph | **85.4** | **2** | **7** | 2.26 |
| M6 Hybrid | 99.4 | 1 | 1 | 3.37 |
| M5 HOG | 109.1 | 0 | 2 | 0.62 |
| M4 Topology | 117.4 | 1 | 3 | 67.0 |
| M3 Shape | 118.7 | 1 | 2 | 3.47 |
| M1 Morphology | 123.9 | 0 | 1 | 0.86 |

Graph features (M2) perform best: skeleton topology is the most style-invariant feature. LLM morphology (M1) performs worst — categorical descriptions don't discriminate between tachygraphic variants.

### Best and Worst T_P15 Matches (Mean Rank Across 6 Methods)

**Best-supported:**

| EVA | Syllable | Mean Rank | Best Individual |
|-----|----------|-----------|-----------------|
| y | si | **25.5** | Hybrid: rank 3 |
| a | ra | **41.2** | Morphology: rank 8 |
| cph | pa | **56.3** | HOG: rank 11 |
| sh | sa | **58.0** | Topology: rank 42 |
| o | ro | **61.5** | Topology: rank 6 |
| cth | na | **72.0** | Graph: rank 10 |
| q | cu | **72.2** | Shape: rank 3 |

**Worst-supported:**

| EVA | Syllable | Mean Rank |
|-----|----------|-----------|
| f | fa | 202.5 |
| n | ni | 199.8 |
| s | so | 186.5 |
| cfh | ma | 178.8 |
| l | la | 167.8 |

### LLM Pairwise Analysis (Method 7)

- 25 proposed + 100 random controls = 125 comparisons
- Mean proposed similarity: 0.070 (on 0-1 scale)
- Mean control similarity: 0.068
- Win rate: 14/25 (56%) — proposed beats mean control
- Same basic structure: **0/25** (LLM judges all pairs as fundamentally different)
- Best individual matches: h->ce (0.20), r->re (0.20), ckh->da (rank 1 among 5)
- LLM consistently describes EVA as "bold Gothic blackletter" vs Costamagna as "thin cursive tachygraphic strokes"

### Morphological Profile Comparison (Method 1)

| Property | EVA Signs | Costamagna Signs |
|----------|----------|-----------------|
| Complexity | simple=8, moderate=15, complex=14 | simple=61, moderate=153, complex=22 |
| Has loops | 65% (24/37) | 23% (55/236) |
| Mean strokes | 2.2 | 2.1 |

The EVA font renders appear significantly more complex and loop-heavy than Costamagna signs. This is largely a rendering artifact.

### Permutation Test

- Real mean rank: 102.6
- Null mean rank: 117.1
- z = -1.05, p = 0.138
- Real assignments rank 14% better than random, but not significant at p<0.05
- Real table beats 86.2% of random tables

## Comparison with Phase 63

| Metric | Phase 63 | Phase 64 |
|--------|----------|----------|
| Methods | 1 (Gemini Embedding 2) | 7 (2 LLM + 5 CV) |
| Discrimination | None (all sims 0.874 +/- 0.025) | All methods have real spread |
| Strong matches | 0/25 | 1/25 (y->si) |
| Moderate matches | 1/25 | 1/25 (a->ra) |
| Perm p-value | 0.379 | 0.138 |
| Gates passed | 0/5 | 2/7 |

Phase 64 is strictly better: methods discriminate, one sign achieves STRONG, and LLM pairwise finds proposed > controls 56% of the time. But the fundamental limitation remains.

## Key Finding

The multi-method approach works as methodology (all methods discriminate, spread is real), but T_P15's assignments do not have strong visual support from the Costamagna syllabary. The primary obstacle is the **domain gap**: font-rendered EVA characters are clean, heavy, and stylized; Costamagna crops are hand-written, thin, and cursive. Every method detects this style difference as the dominant signal, drowning out structural correspondence.

The ~7 best-matching signs (y, a, cph, sh, o, cth, q) may have genuine visual correspondence; the remaining ~18 do not. This is consistent with Phase 58's conclusion that visual matching requires either (a) manuscript-to-manuscript comparison at stroke level, or (b) that the Voynich glyphs have diverged sufficiently from Costamagna's 1953 reconstructions that visual correspondence cannot be established with current methods.

## CLI Commands

```bash
# Individual methods
voynich stroke-extract     # M2: Skeleton graph features
voynich shape-desc         # M3: Hu + Fourier descriptors
voynich topo-features      # M4: Topological features
voynich hog-compare        # M5: HOG features
voynich hybrid-features    # M6: Combined hybrid vector
voynich morph-describe     # M1: LLM morphology descriptions (requires OPENROUTER_API_KEY)
voynich llm-pairwise       # M7: LLM pairwise comparison (requires OPENROUTER_API_KEY)

# Integration
voynich visual-ensemble    # Rank fusion + validation
voynich phase64-verdict    # Evaluate gates
voynich phase64            # Full pipeline
```

## Output Files

```
results/
  p64_graph_features.json       # M2 per-sign features
  p64_graph_matrix.npz          # M2 distance matrix (37x236)
  p64_shape_matrix.npz          # M3 distance matrix
  p64_topo_features.json        # M4 per-sign features
  p64_topo_matrix.npz           # M4 distance matrix
  p64_hog_matrix.npz            # M5 distance matrix
  p64_hybrid_matrix.npz         # M6 combined matrix
  p64_morphology_eva.json       # M1 EVA descriptions
  p64_morphology_costa.json     # M1 Costamagna descriptions
  p64_morphology_matrix.npz     # M1 distance matrix
  p64_pairwise.json             # M7 LLM comparisons + scores
  p64_ensemble_matrix.npz       # Rank-fused matrix
  p64_diagnostics.json          # Per-method diagnostic scores
  p64_validation.json           # T_P15 validation + permutation test
  phase64.json                  # Final verdict (gates, metrics)
```
