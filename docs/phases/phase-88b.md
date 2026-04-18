# Phase 88b: Generalized Naibbe Parameter Grid Search

[← Back to Phase Index](README.md)

## Purpose

Close the "you only tested Greshko's published defaults" objection to [phase-88](phase-88.md). Greshko's own email noted that "these features could, of course, be changed and extended to a grid search." This phase runs a targeted 10-point grid × 20 seeds = 200 runs over the generalized Naibbe's tunable parameters — `N_TABLES`, `TABLE_WEIGHTS`, `OUTPUT_ALPHA_SIZE`, affix-length window — and measures whether any config simultaneously hits the tachygraphy-specific thresholds on cross-boundary MI (≥ 1.284×) or frequency-connectivity (ρ ≥ 0.5).

## Method

Grid of 10 configurations:

| Config | N_tables | Weights | α size | Affix |
|---|---|---|---|---|
| baseline_greshko | 6 | 5:2:2:2:1:1 | 20 | 2–3 |
| single_table | 1 | [1] | 20 | 2–3 |
| two_tables_equal | 2 | 1:1 | 20 | 2–3 |
| three_tables_equal | 3 | 1:1:1 | 20 | 2–3 |
| ten_tables_equal | 10 | 1×10 | 20 | 2–3 |
| extreme_skew | 6 | 10:1:1:1:1:1 | 20 | 2–3 |
| alpha_15 | 6 | 5:2:2:2:1:1 | 15 | 2–3 |
| alpha_26 | 6 | 5:2:2:2:1:1 | 26 | 2–3 |
| short_affixes | 6 | 5:2:2:2:1:1 | 20 | 1–2 |
| long_affixes | 6 | 5:2:2:2:1:1 | 20 | 3–4 |

For each (config, seed):
1. Encode the 52 639-char Latin plaintext (`nathist_book16.txt`) through the config.
2. Size-match output to Voynich-full character count; compute H₀–H₆.
3. Compute cosine of the shift vector vs Voynich-full and Voynich-B.
4. Compute cross-boundary MI ratio (plain character tokenisation).
5. Compute frequency-connectivity Spearman ρ (capped at 2 000 types).

Thresholds (for "reaches tachygraphic-specific"):
- MI ≥ 1.284× (tachygraphic reference, paper Section 4.4)
- ρ ≥ 0.5 (below Voynich's 0.618 but well above Naibbe's baseline ~0.23)

## Results

| Config | MI mean | MI max | ρ mean | ρ max | hits MI ≥1.284 | hits ρ ≥0.5 |
|---|---|---|---|---|---|---|
| baseline_greshko | 1.005 | 1.012 | +0.167 | +0.266 | **0/20** | **0/20** |
| single_table | **1.103** | **1.229** | +0.150 | +0.304 | 0/20 | 0/20 |
| two_tables_equal | 1.024 | 1.091 | +0.165 | +0.331 | 0/20 | 0/20 |
| three_tables_equal | 1.014 | 1.045 | +0.140 | +0.257 | 0/20 | 0/20 |
| ten_tables_equal | 1.002 | 1.005 | +0.135 | +0.263 | 0/20 | 0/20 |
| extreme_skew | 1.024 | 1.059 | +0.187 | +0.289 | 0/20 | 0/20 |
| alpha_15 | 1.005 | 1.012 | +0.171 | +0.298 | 0/20 | 0/20 |
| alpha_26 | 1.012 | 1.024 | +0.131 | +0.188 | 0/20 | 0/20 |
| short_affixes | 1.008 | 1.017 | **+0.203** | **+0.343** | 0/20 | 0/20 |
| long_affixes | 1.006 | 1.021 | +0.121 | +0.275 | 0/20 | 0/20 |

Reference values: Voynich-full MI = 1.448, ρ = +0.615.

### Best per diagnostic

- **Best MI**: `single_table` at 1.103 mean (max 1.229) — still 0.18 below tachygraphic 1.284 and 0.35 below Voynich 1.448. Also produces the *lowest* mean entropy-shift cosine (+0.840), so it doesn't simultaneously pass the other diagnostic either.
- **Best ρ**: `short_affixes` at +0.203 mean (max +0.343) — less than half the 0.5 threshold; one-third of Voynich's +0.615.

**Verdict: GRID_CONFIRMS_PHASE88** — 0 of 200 runs reach either tachygraphy-specific threshold.

## Interpretation

The Phase 88 NAIBBE_1_OF_3 verdict is robust to 10-point parameter variation with 20 independent seeds each. The `single_table` configuration — a pure monoalphabetic bigram substitution, the Naibbe family's closest approach to the Voynich's cross-boundary statistics — is still substantially below tachygraphic on MI and has the weakest entropy-shift match of the grid. No parameter choice produces a Naibbe variant that simultaneously reproduces the Voynich's entropy-shift shape, cross-boundary MI anomaly, and frequency-connectivity correlation.

This strengthens the rebuttal to Greshko's broader uniqueness-is-not-established claim: entropy-shift cosine on its own is genuinely non-specific (Phase 88), but the two token-adjacency discriminators are robust against the Naibbe cipher family — not just against Greshko's published default parameters.

The short-affix config's marginally elevated ρ (+0.203 mean, +0.343 max) is the only hint that the Naibbe family has *any* freq-connectivity structure, and it's still well below half the Voynich value — likely an artifact of shorter affixes creating denser ED-1 neighborhoods, not a genuine frequency-driven correlation.

## Files

- **Implementation:** [src/voynich/phases/p88b_grid_search.py](../../src/voynich/phases/p88b_grid_search.py)
- **Output:** `results/p88b_grid_search.json`
- **Runtime:** 190 s (200 runs)

## Dependencies

- `data/reference/greshko/nathist_book16.txt` (Latin plaintext)
- `results/p88_naibbe_generalized.json` (Voynich reference MI and ρ values)
- Helpers from [src/voynich/phases/p88_naibbe_generalized.py](../../src/voynich/phases/p88_naibbe_generalized.py) (slot-grammar, table construction, encoder, diagnostics)
