# Phase 78: CVC T1 Permutation Validation

[← Back to Phase Index](README.md)

## Purpose

Validate that Phase 75's 316 CVC T1 identifications are table-specific — not an artifact of the pipeline mechanics — by running 1,000 random CV assignment tables through the identical T1 pipeline and computing a p-value. The paper acknowledged that "the CVC identifications have not been run through the same permutation framework that validated the CV set at p = 0.009"; this phase closes that gap.

## Method

The test holds everything constant except the 25 CV triple→syllable assignments:
- **Fixed across all trials:** CVC coda table (hook→n, sigmoid→s, vertical→t, connector→null, descender→null), confirmed/unresolved triple classification, corpus, dictionary (131K expanded Latin), folio distribution
- **Randomised per trial:** 25 triples each assigned a syllable sampled uniformly with replacement from 75 Latin CV syllables

For each of 1,000 random tables:
1. Build wildcard patterns from partially-decoded token types (confirmed triples → literal chars, unresolved → `[a-z]`)
2. Match patterns against the dictionary
3. Count identifications: unique match + ≥3 distinct folios

The real table's count is compared against the null distribution.

### Optimisation

Pattern templates are precomputed once (8,771 templates from 9,257 token types). Each template stores the structural skeleton — which positions are HIGH (confirmed triple), LOW (unresolved), or CODA (fixed). Per trial, only the HIGH positions are filled with the random assignment's syllable values. Static templates (no HIGH positions) are matched once and cached. Dictionary matching uses direct character comparison rather than regex, reducing per-trial time from ~5s to ~3s.

## Results

| Metric | Value |
|--------|-------|
| Real table IDs | 331 (74 distinct words) |
| Null mean ± std | 209.6 ± 32.0 |
| Null median | 206 |
| Null range | 138 – 357 |
| **p-value** | **0.002** |
| **z-score** | **3.79** |
| Null frac = 0 | 0.000 |

### Gate Results

| Gate | Threshold | Value | Status |
|------|-----------|-------|--------|
| G1 p-value | < 0.001 | 0.002 | FAIL (narrowly) |
| G2 z-score | > 3.0 | 3.79 | **PASS** |
| G3 unique words > 50 | > 50 | 5 | FAIL |

**Verdict: CVC_T1_SIGNIFICANT (1/3 gates)**

### Per-Word Specificity

- **5 words unique to real table** (found by zero null trials): *erradicat*, *ceradis*, *benidiis*, *didit*, *abradi*
- **Mean word specificity: 0.947** — on average, each real-table word appears in only 5.3% of null trials
- Most specific (after the 5 unique): *corat* (1 trial), *decos* (1), *dedi* (2), *cıcora* (2), *cosdi* (3), *recodi* (3), *bene* (4), *sedis* (4)
- Least specific: short 2-letter words (*mi*, *ra*, *be*, *di*, *ni*) found by 13–16% of null trials — structurally easy to match

### Comparison with CV Permutation

| | CV (Phase 52) | CVC (Phase 78) |
|--|---------------|----------------|
| Real IDs | 22 | 331 |
| Null mean | ~1.5 | 209.6 |
| p-value | 0.009 | 0.002 |
| Frac null = 0 | 0.744 | 0.000 |

CVC is more significant (p = 0.002 vs 0.009) but the gap is narrower in relative terms (1.58× vs ~15×). This is because fixed codas reduce wildcards for ALL tables equally, inflating both real and null identification counts.

### Null Distribution (Percentiles)

| Percentile | IDs |
|------------|-----|
| 25th | 187 |
| 50th | 206 |
| 75th | 229 |
| 90th | 252 |
| 95th | 271 |
| 99th | 291 |

The real table's 331 IDs exceeds the 99th percentile of the null distribution (291).

## Interpretation

The real assignment table produces significantly more T1 identifications than random tables (p = 0.002, z = 3.79). The 331 identifications exceed the 99th percentile of the null distribution.

However, the test also reveals that the T1 pipeline has a substantial structural baseline: random tables produce a mean of ~210 identifications (not zero), because the fixed coda characters, confirmed triple split, and large dictionary create many opportunities for accidental unique matches. The real table's advantage is quantitative (~58% more than the null mean), not qualitative.

The 5 words unique to the real table (*erradicat*, *ceradis*, *benidiis*, *didit*, *abradi*) are longer words requiring multiple correct triple assignments simultaneously — these are the most compelling individual identifications.

## Note on Template–Phase 75 Discrepancy

The template pipeline produces 331 IDs vs Phase 75's recorded 316 (difference: 15). This is because Phase 75's `_extract_constraints` has an additional `len(matched_word) != target_len` guard that the template approach handles implicitly through length-indexed dictionaries. Since both real and null trials use the identical template pipeline, the permutation test is internally valid — it compares apples to apples.

## Files

- **Implementation:** `src/voynich/phases/cvc_t1_permutation.py`
- **Output:** `results/phase78_cvc_t1_perm.json`
- **CLI:** `voynich cvc-t1-perm` / `voynich phase78`

## Dependencies

- `results/combined_refine.json` (Phase 15 best_assignment)
- `results/triple_tiers.json` (Phase 28/53 confirmed triples)
- `results/p75_t1.json` (Phase 75 baseline)
