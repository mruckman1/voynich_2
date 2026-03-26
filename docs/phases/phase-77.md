# Phase 77: Timm-Schinner Self-Citation Discriminator Test

**Verdict: SELF_CITATION_ELIMINATED** (4/4 gates)

Closes the paper's most explicitly acknowledged gap: the Timm & Schinner (2020) self-citation algorithm had not been tested on the entropy shift discriminator or the cross-boundary MI test. Phase 77 connects Phase 27.1's existing generator to Phase 55A/B's existing discriminators. 540 corpora tested (27 configurations × 20 seeds). The self-citation algorithm is decisively eliminated on both tests.

## Test A: Entropy Shift — ELIMINATED

**CLI:** `voynich ts-test` | **Output:** `results/p77_timm_schinner.json`

The self-citation algorithm produces entropy shift cosines that are **negative** — anticorrelated with the Voynich's entropy signature.

| Mechanism | Cosine | Rank |
|-----------|--------|------|
| Schinner positional | +0.968 | 1 (scope limitation) |
| Schinner simple | +0.953 | 2 (scope limitation) |
| Tachygraphy | **+0.820** | 3 |
| Cardan 3-hole | +0.590 | 4 |
| Homophonic | +0.566 | 5 |
| Simple substitution | 0.000 | 8 |
| **Self-citation (best grid)** | **−0.108** | 9 |
| **Self-citation (default)** | **−0.153** | 10 |
| Naibbe cipher | −0.843 | 13 |

Default configuration (p_copy=0.7, p_mutate=0.10, buffer=100): cosine **−0.153** (CI: [−0.166, −0.140]). Gap from tachygraphy: **0.973** — the CI is entirely below zero and nowhere near +0.820.

Best grid configuration (p_copy=0.8, p_mutate=0.05, buffer=200): cosine **−0.108** — still negative, still anticorrelated.

The copy mechanism *reduces* high-order character entropy (repeated words create predictable patterns) rather than maintaining the high entropy floor that the Voynich shows and tachygraphy predicts.

## Test B: Cross-Boundary MI — ELIMINATED

| Mechanism | MI Ratio |
|-----------|----------|
| Voynich (reference) | **1.450×** |
| Tachygraphy (syllable-as-token) | 1.284× |
| **Self-citation (best grid)** | **1.054×** |
| Schinner stochastic | 1.044× |
| **Self-citation (default)** | **1.036×** |
| Shuffled null | ~1.00× |

Default: MI ratio **1.036×** (CI: [1.030, 1.047]) — indistinguishable from Schinner's null-level 1.044× and 29% below the Voynich's 1.450×.

Best grid (p_copy=0.8, p_mutate=0.05, buffer=50): **1.054×** — still at null level.

The copy mechanism preserves within-word character statistics but does not create meaningful between-word transitions. The Voynich's 1.450× requires systematic word-boundary structure — what syllable-as-token tachygraphy produces.

## Parameter Sensitivity

27 configurations tested: p_copy ∈ {0.6, 0.7, 0.8} × p_mutate ∈ {0.05, 0.10, 0.15} × buffer_size ∈ {50, 100, 200}.

- **Cosine range:** −0.169 to −0.108 (all negative, all anticorrelated)
- **MI ratio range:** 1.034 to 1.054 (all at null level)
- Higher p_copy slightly improves both metrics but never approaches either target
- p_mutate and buffer_size have negligible effect

## Gates

| Gate | Threshold | Result | Value |
|------|-----------|--------|-------|
| A1 | Entropy CI < tachygraphy (0.820) | **PASS** | CI high = −0.140 |
| A2 | Cosine < 0.5 (discriminated) | **PASS** | −0.153 |
| B1 | MI ratio < 1.10 (null level) | **PASS** | 1.036 |
| B2 | MI CI < Voynich (1.450) | **PASS** | CI high = 1.047 |

## Key Findings

1. **Self-citation is eliminated on both tests.** Entropy cosine is negative (−0.153), MI ratio is at null level (1.036×). No parameter configuration rescues it. This closes the paper's most significant acknowledged gap.

2. **Self-citation behaves like Schinner's stochastic model on MI** (1.036× vs 1.044×) — both produce null-level cross-boundary correlations. Neither can explain the Voynich's 1.450× anomaly.

3. **Self-citation is WORSE than Schinner on entropy shift** (−0.153 vs +0.953). Schinner's model is trained on Voynich data (circular), but self-citation's copy mechanism actively destroys the entropy signature that tachygraphy preserves.

4. **Tachygraphy remains the only mechanism that passes both tests** — cosine +0.820 AND MI ratio 1.284×. The paper's central claim is strengthened: "Against 13 encoding mechanisms, tachygraphy is the sole survivor of both discriminators."

## CLI Commands

```bash
voynich ts-test       # Full pipeline (~7 min)
voynich phase77       # Alias
```
