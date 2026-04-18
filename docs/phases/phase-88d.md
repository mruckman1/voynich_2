# Phase 88d: Naibbe at Affix (Sub-Token) Granularity

[← Back to Phase Index](README.md)

## Purpose

Close the ``what if we measure Naibbe at finer granularity'' loophole. Phase 88c measured the tachygraphic simulation at two granularities (syllable and word); for symmetry, Phase 88d measures Naibbe at its sub-token unit — the prefix and suffix affixes produced by one half of a bigram encoding. The cross-boundary MI is computed only across original Naibbe token boundaries: between the *suffix* of token N (last affix) and the *prefix* of token N+1 (first affix). This is the direct analogue of how tachygraphic cross-boundary MI measures across word boundaries at syllable granularity.

## Method

For each of 20 seeds (Greshko defaults: `N_TABLES=6`, weights `5:2:2:2:1:1`, output alphabet 20, affix length 2–3):

1. Encode Latin plaintext (`data/reference/greshko/nathist_book16.txt`) and retain, per bigram, the prefix affix `P_i` (encrypting letter 1) and suffix affix `S_i` (encrypting letter 2).
2. Compute cross-boundary MI ratio on the pair sequence `(last_char(S_i), first_char(P_{i+1}))`, i.e. only suffix → next-prefix across original Naibbe token boundaries. In-token `(P_i, S_i)` pairs are excluded because they are structurally correlated by construction (same table, same slot grammar).
3. Compute frequency-connectivity Spearman ρ on the combined affix vocabulary (all prefix and suffix affix types as tokens, ED-1 neighbor counts vs log-frequency).

## Results

| System | Granularity | MI | ρ |
|---|---|---|---|
| **Voynich** | token (≡ syllable by hypothesis) | **1.448** | **+0.615** |
| **Tachygraphic** | syllable | **1.285** | **+0.585** |
| Tachygraphic | word | 1.061 | +0.235 |
| **Naibbe** | **affix (across original token boundary)** | **1.006 ± 0.004** | **+0.220 ± 0.130** |
| Naibbe | token | 1.006 | +0.176 |

**Verdict: NAIBBE_FAILS_AT_EVERY_GRANULARITY.** Cross-boundary MI stays at the 1.0 null at both token and sub-token granularity (1.006 in both cases). Frequency-connectivity ρ is slightly higher at affix granularity (+0.220 vs +0.176 at token level) because the smaller affix vocabulary produces more accidental edit-distance-1 collisions, but remains well below tachygraphic's +0.585 at syllable granularity and Voynich's observed +0.615.

## Interpretation

Cross-boundary MI at the 1.0 null is predicted by Naibbe's architecture: each plaintext bigram is encoded independently via random table selection, so the last glyph of one bigram's encoding carries no information about the first glyph of the next. This is a structural property the cipher cannot overcome without adding a cross-bigram dependency mechanism that isn't part of the Naibbe design.

Frequency-connectivity is weakly positive at both granularities but one-third the Voynich value. The rise from 0.176 (token) to 0.220 (affix) reflects purely combinatorial denseness (smaller type vocabulary, shorter strings → more accidental neighbors), not a systematic stroke-modification-family-like structure. Tachygraphy produces ρ ≈ 0.585 because its encoder systematically generates edit-distance-1 neighbors through the stroke-modification system; Naibbe produces nothing comparable because its affixes are sampled from slot-grammar-generated inventories with no cross-affix phonetic relationship.

## Four-cell table is now airtight

| System | Own granularity | Token-level (if different) |
|---|---|---|
| Voynich | 1.448 / +0.615 | — |
| Tachygraphic | 1.285 / +0.585 (syllable) | 1.061 / +0.235 (word) |
| Naibbe | 1.006 / +0.220 (affix, across boundary) | 1.006 / +0.176 (token) |

Every system is measured at both its own hypothesis-consistent granularity and at the next-coarser level. The pattern is consistent: tachygraphic tracks Voynich at its hypothesized granularity and falls to Naibbe-level values at the coarser one; Naibbe fails at every granularity measurable under its own design.

## Files

- **Implementation:** [src/voynich/phases/p88d_naibbe_affix.py](../../src/voynich/phases/p88d_naibbe_affix.py)
- **Output:** `results/p88d_naibbe_affix.json`
- **Runtime:** 7 s (20 seeds)

## Dependencies

- [src/voynich/phases/p88_naibbe_generalized.py](../../src/voynich/phases/p88_naibbe_generalized.py) — slot grammar, table construction, `_freq_conn_rho_plain`, `_cross_boundary_ratio_plain`
- `data/reference/greshko/nathist_book16.txt` — Latin plaintext
