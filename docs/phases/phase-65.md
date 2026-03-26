# Phase 65: Word Boundary Discovery in the Decoded Stream

[← Phase Index](README.md)

## Motivation

The CVC decode produces correct syllables (83% Costamagna-attested, p=0.006 coherence) but the decoded output doesn't form recognizable words. Phase 62 showed EVA tokens encode 2–3 Latin syllables each, meaning token boundaries are neither word boundaries nor syllable boundaries. Phase 65 strips all EVA token boundaries from the CVC-decoded corpus to produce a continuous character stream, then applies four word segmentation methods — organized from unsupervised to fully supervised — to discover Latin word boundaries.

## Input: The Decoded Stream

All methods operate on the same input: a continuous character stream produced by concatenating CVC-decoded tokens.

- **Full stream**: 218,035 characters from 36,238 tokens
- **8 section-level streams**: herbal_a (52K), recipes (65K), biological (36K), pharmaceutical (22K), astronomical (20K), cosmological (13K), unknown (8K), herbal_b (1K)
- **Voynich alphabet**: 17 characters (`abcdefghilmnorstu`)
- **Latin calibration stream**: 422,862 chars, 73,528 words, mean word length 5.75 — built by syllabifying Latin reference text and concatenating syllables (mimics CVC decode output)

## Methods and Results

### Method 1: Harris MI Boundary Detection (Step 65.2)

At word boundaries, the next character becomes less predictable. Computes pointwise mutual information between character context (length 1–5) and next character. Boundaries placed at MI drops.

**Latin calibration**: Best config = derivative method, context length 4. **F1 = 0.651** (P=0.608, R=0.700). The method works well on Latin.

**Voynich results**: 43,608 words, mean length 5.0, dict hit 7.5%, selectivity **0.45×** (below random). Recipes section best at 0.52×.

| Gate | Threshold | Result |
|------|-----------|--------|
| H1: Latin F1 > 0.3 | 0.651 | **PASS** |
| H2: Dict hit > 10% | 7.5% | FAIL |
| H3: Selectivity > 1.5× | 0.45× | FAIL |
| H4: Word length 4–8 | 5.0 | **PASS** |

**Verdict: HARRIS_PARTIAL (2/4)**

### Method 2: Bayesian (MDL) Word Segmentation (Step 65.3)

Iterative Viterbi DP minimizing description length: segment stream, count word frequencies, re-segment using updated frequencies. Equivalent to MAP estimate under a unigram Dirichlet-multinomial model.

**Latin calibration**: All alpha values tested (0.1–10.0) produced F1 = 0.000 on the Latin calibration stream. The MDL objective with the 17-char Voynich alphabet does not discriminate word boundaries.

**Voynich results**: 18,174 words, mean length 12.0 (too long), dict hit 0.0%, selectivity 0.00×.

| Gate | Threshold | Result |
|------|-----------|--------|
| B1: Word length 4–8 | 12.0 | FAIL |
| B2: Dict hit > 10% | 0.0% | FAIL |
| B3: Selectivity > 1.5× | 0.00× | FAIL |
| B4: Top-20 ≥ 5 in dict | 0 | FAIL |

**Verdict: BAYESIAN_FAIL (0/4)**

### Method 3: Character LM Perplexity Minimization (Step 65.4)

5-gram character LM trained on Latin pharmaceutical text (Circa Instans + De Viribus Herbarum, 449K chars) with `#` word boundary marker. Viterbi DP finds the segmentation minimizing total negative log-probability.

**Latin calibration**: **F1 = 0.654** (P=0.907, R=0.512). Perplexity drops from 36.3 (unsegmented) to 19.9 (segmented). Dict hit on held-out Latin = 25.5%. The method works excellently on Latin.

**Voynich results**: 20,051 words, mean length 10.5 (too long), dict hit 1.1%, selectivity **0.12×** (far below random). The LM prefers long "words" because the Voynich decoded stream has few character sequences that match Latin patterns.

| Gate | Threshold | Result |
|------|-----------|--------|
| L1: Latin F1 > 0.4 | 0.654 | **PASS** |
| L2: Dict hit > 15% | 1.1% | FAIL |
| L3: Selectivity > 2.0× | 0.12× | FAIL |
| L4: Perplexity reduction | — | FAIL |
| L5: ≥ 3 signal words in top-20 | 0 | FAIL |

**Verdict: LM_FAIL (1/5)**

### Method 4: Recipe Template-Constrained Segmentation (Step 65.5)

Dictionary-constrained DP segmentation of 30 recipe streams using a 5,000-word pharmaceutical dictionary. Also tries 6 recipe templates (simple_preparation, grinding, mixture, dosage, property_statement, compound_naming).

**Results**: 100% coverage (all recipe characters matched), selectivity **11.1×**, but mean word length 2.6 (too short). The DP finds short dictionary words (2–3 chars) that trivially tile the stream. No template matches; 0 distinct ingredients identified.

| Gate | Threshold | Result |
|------|-----------|--------|
| R1: ≥ 30% match | 100% | **PASS** |
| R2: Mean ED ≤ 2.0 | ∞ | FAIL |
| R3: ≥ 5 ingredients | 0 | FAIL |
| R4: ≥ 2 template types | 0 | FAIL |
| R5: ≥ 1 fully readable | 30 | **PASS** |

**Verdict: RECIPE_PARTIAL (2/5)**

## Integration (Step 65.6)

### Method Comparison

| Method | Dict Hit | Selectivity | Mean WL | Latin F1 | Gates | Verdict |
|--------|----------|-------------|---------|----------|-------|---------|
| Harris MI | 7.5% | 0.45× | 5.0 | **0.651** | 2/4 | PARTIAL |
| Bayesian MDL | 0.0% | 0.00× | 12.0 | 0.000 | 0/4 | FAIL |
| Character LM | 1.1% | 0.12× | 10.5 | **0.654** | 1/5 | FAIL |
| Recipe DP | 100% | **11.1×** | 2.6 | N/A | 2/5 | PARTIAL |

### EVA Baseline

Using EVA token boundaries as word boundaries: dict hit **15.4%**, selectivity **1.05×**. All four methods perform worse than this naive baseline.

### Pairwise Agreement

| | Harris | Bayesian | LM |
|---|--------|----------|-----|
| Harris | 1.000 | 0.000 | 0.000 |
| Bayesian | 0.000 | 1.000 | 0.661 |
| LM | 0.000 | 0.661 | 1.000 |

Mean pairwise agreement: 0.220. No cross-method consensus.

### Integration Gates

| Gate | Threshold | Result |
|------|-----------|--------|
| G1: ≥ 2 methods pass | 0 pass | FAIL |
| G2: Consensus selectivity > 1.5× | 0.000× | FAIL |
| G3: Word length 3.5–7.0 | 25.4 | FAIL |
| G4: Agreement > 0.3 | 0.220 | FAIL |

## Verdict: SEGMENTATION_FAILED (0/4)

The decoded character stream cannot be re-segmented into Latin words by any of the four tested methods.

## Key Findings

1. **Latin calibration succeeds, Voynich transfer fails.** Harris MI and the character LM both achieve F1 > 0.65 on Latin — the methods work. But the Voynich decoded stream has no Latin-like word boundary statistics.

2. **17-char alphabet too small.** The Voynich decoded stream uses only 17 characters (vs 38 in Latin). This reduces MI contrast at boundaries and limits the LM's discriminative power.

3. **All methods worse than EVA baseline.** Treating original EVA tokens as words (15.4% dict hit, 1.05× selectivity) outperforms every re-segmentation approach. The character-level statistics introduced by stripping token boundaries are less informative than the original token boundaries.

4. **Variable-length encoding confirmed.** The failure to find word boundaries in the decoded stream is consistent with Phase 53's finding that EVA characters map to 1–3 Latin characters (variable-length), not fixed 2-char CV syllables. The decode errors corrupt the character-level statistics that all four methods depend on.

5. **56% decode error rate is the bottleneck.** Phase 16's 43.6% dict hit means 56.4% of tokens decode incorrectly. Word boundary discovery requires a more accurate character-level decode.

## Dependency Chain

```
combined_refine.json (P15) + corrected_coda (P60) + corpus
    │
    ▼
p65_decoded_stream.json
    │
    ├──▶ p65_harris.json         (independent)
    ├──▶ p65_bayesian.json       (independent)
    ├──▶ p65_lm_segment.json     (independent)
    ├──▶ p65_recipe_segment.json (independent)
    │
    ▼
p65_integrate.json
```

## CLI Commands

```bash
voynich build-stream       # Step 65.1: Build decoded character streams
voynich harris-segment     # Step 65.2: Harris MI boundary detection
voynich bayesian-segment   # Step 65.3: Bayesian (MDL) word segmentation
voynich lm-segment         # Step 65.4: Character LM perplexity minimization
voynich recipe-segment     # Step 65.5: Recipe template-constrained segmentation
voynich phase65-verdict    # Step 65.6: Integration and verdict
voynich phase65            # Run full Phase 65 pipeline
```
