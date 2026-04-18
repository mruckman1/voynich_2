# Phase 80: Wildcard-Consistency Check

[← Back to Phase Index](README.md)

## Purpose

Reviewer Response v1 (Reviewer 3.7): test whether the word-level identifications with unresolved wildcards (15 of 316) are internally consistent — do the implied syllable values at unresolved triples agree with the T_P15 assignment? If wildcards resolve to stable, agreeing values, they support the model; if they contradict T_P15 or each other, they are pattern-matching artifacts.

## Method

- Collect all word-level identifications that matched at wildcard positions
- Extract the implied syllable value at each unresolved triple
- Compare against T_P15's assignment
- Null: 1,000 random CV tables through the same pipeline

## Results

| Metric | Value |
|--------|-------|
| Total identifications | 316 |
| Fully decoded (no wildcards) | 301 |
| With wildcards | 15 |
| Distinct unresolved triples observed | 5 |
| Triples agreeing with T_P15 | **0 / 5** |
| Mean consistency | 53.3% |
| Cross-identification conflicts | 8 |
| Null z-score | **−2.39** (worse than random) |

**Verdict: INCONSISTENCIES_FOUND** — the wildcard identifications disagree with T_P15 and with each other more than random tables do. They are pattern-matching artifacts, not implicit decodings.

## Interpretation

The wildcard identifications do not constitute additional evidence for the model — they are collateral matches produced by the dictionary search. Only the 301 fully-decoded identifications (no wildcards) carry word-level signal. This tightens the paper's evidence set and removes an ambiguity flagged by Reviewer 3.7.

## Files

- **Implementation:** [src/voynich/phases/p80_wildcard_consistency.py](../../src/voynich/phases/p80_wildcard_consistency.py)
- **Output:** `results/p80_wildcard_consistency.json`
- **CLI:** `voynich wildcard-check` / `voynich phase80`

## Dependencies

- `results/combined_refine.json`
- `results/p75_t1.json` (Phase 75 T1 identifications)
- `results/triple_tiers.json`
