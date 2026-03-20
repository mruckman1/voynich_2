# Phase 58: Costamagna-Constrained CSP

**Verdict:** FAIL (3/8 gates) — Costamagna constraints too broad without visual sign matching; only 3 genuinely ambiguous triples remain

[← Phase 57](phase-57.md) | [Phase Index](README.md)

---

## Motivation

Phase 14's stroke-feature CSP used articulatory priors to constrain domains (~5.2 candidates per variable). Phase 58 attempts the first *historically-grounded* CSP: using Costamagna's attested syllable inventory (91 CV entries) as the domain source instead of phonotactic candidates.

The goal: determine whether Costamagna's inventory alone — even without visual sign matching — produces a better assignment than T_P15.

---

## Method

**Modules:** `costamagna_csp.py`, `phase58_verdict.py`
**CLI:** `voynich cost-domains`, `voynich cost-reduction`, `voynich cost-csp`, `voynich cost-compare`, `voynich phase58-verdict`, `voynich phase58`
**Outputs:** `cost_domains.json`, `cost_reduction.json`, `cost_csp.json`, `cost_compare.json`, `phase58_verdict.json`

### Step 58.1: Domain Construction

For each of the 25 stroke-feature triples:
1. **CONFIRMED** triples (Phase 45 tiers CONFIRMED + LANDSCAPE_CONFIRMED) → singleton domain (locked)
2. **CODA_MARKER** triples (Phase 57) → excluded from CSP (not a variable)
3. **UNRESOLVED** triples → Costamagna CV inventory minus confirmed values, plus shared-sign expansion (ad↔at, me↔mi, ne↔ni)

### Step 58.2: Domain Size Comparison

Compare domain sizes against Phase 11 (unconstrained) and Phase 14 (stroke-guided).

### Step 58.3: CSP Solve

Greedy hill-climbing with 20 random restarts:
- Initialize unresolved triples randomly from their Costamagna domains
- Lock confirmed triples
- Iteratively swap each unresolved triple's value to maximize dict_hit on a 5,000-token subsample
- Track top solutions and consensus changes

### Step 58.5: Comparison vs T_P15

Evaluate both tables (T_P15 and best CSP solution) on all canonical metrics: dict_hit, signal words, bigram z, net signal, Costamagna attestation.

---

## Results

### Domain Landscape

| Type | Count | Description |
|------|-------|-------------|
| CONFIRMED | 22 | 12 CONFIRMED + 10 LANDSCAPE_CONFIRMED (Phase 45) |
| CODA_MARKER | 0 | Modifiers share triples with syllabic chars (d/i/m all → `vertical,vertical,minim`) |
| UNRESOLVED | 3 | The genuinely ambiguous triples |

The 3 unresolved triples:

| Triple | Current (T_P15) | Corpus Tokens | Coverage |
|--------|-----------------|---------------|----------|
| `open_curve,hook,rare` | hi | ~82 | 0.23% |
| `open_curve,open_curve,bench` | ha | ~55 | 0.15% |
| `sigmoid,hook,rare` | fe | ~27 | 0.07% |

**Total coverage: 164 tokens (0.45% of corpus)**

### Domain Size Comparison

| Phase | Mean Domain Size | Total Search Space |
|-------|-----------------|-------------------|
| Phase 11 (unconstrained) | ~75 per variable | impractical |
| Phase 14 (stroke-guided) | ~5.2 per variable | ~10^9 |
| **Phase 58 (Costamagna)** | **73 per variable** | **10^5.6** |

The Costamagna domains (~73 per unresolved variable) are broader than Phase 14's stroke-guided domains (~5.2), but the search space is only 73³ ≈ 389,000 because only 3 variables are free. This is trivial to search.

### CSP Results

**20 random restarts completed in 102 seconds.**

| Rank | Dict Hit | Delta vs T_P15 | Changes |
|------|----------|----------------|---------|
| #1 | 33.40% | +0.12% | 3 |
| #2 | 33.40% | +0.12% | 3 |
| #3 | 33.40% | +0.12% | 3 |
| #4 | 33.40% | +0.12% | 3 |
| #5 | 33.40% | +0.12% | 3 |

All top-5 solutions achieve the same marginal improvement (+0.12% dict_hit). The values assigned to the 3 unresolved triples vary across solutions — the landscape is flat, confirming Phase 44's finding.

**One consensus change:** `open_curve,open_curve,bench` from "ha" → "an" (consistent across all top-5 solutions). The other 2 triples show no consensus.

### Full Metric Comparison (Raw Decode)

| Metric | T_P15 | CSP Best | Delta |
|--------|-------|----------|-------|
| Dict hit | 21.64% | 21.66% | +0.02% |
| Signal words | 29 | 29 | 0 |
| Bigram z | 106.37 | 108.41 | +2.04 |
| Net signal | 4,220 | 4,208 | -12 |
| Mean word length | 6.95 | 6.95 | 0.00 |
| Costamagna attest. | 0.0% | 0.0% | 0 |

The CSP result is **indistinguishable from T_P15** on every metric.

Note: These metrics use raw decode (no modifier handling), which produces lower dict_hit than the R3 strategy. The comparison is internally consistent — both tables evaluated identically.

---

## Validation Gates

| Gate | Threshold | Result | Detail |
|------|-----------|--------|--------|
| G1 | Dict hit ≥ 43.6% | **FAIL** | 21.66% (raw decode) |
| G2 | Signal count ≥ 56 | **FAIL** | 29 |
| G3 | Bigram z ≥ 14.78 | **PASS** | 108.41 |
| G4 | Costamagna attest. ≥ 60% | **FAIL** | 0.0% |
| G5 | ≥ 3 new signal words | **FAIL** | delta = 0 |
| G6 | Net signal > T_P15 | **FAIL** | 4,208 < 4,220 |
| G7 | Confirmed triples preserved | **PASS** | all preserved |
| G8 | ≥ 2 triples resolved | **PASS** | 3 changed |

**Score: 3/8 — FAIL**

---

## Interpretation

### Why the CSP Found Nothing

1. **Too few free variables.** With 22/25 triples already confirmed, the CSP has only 3 variables to optimize. These 3 triples cover just 0.45% of the corpus — far too few tokens to move any aggregate metric.

2. **The landscape is flat.** Phase 44 (MaxSAT) already demonstrated that 500+ near-optimal solutions exist in the assignment space. The 3 genuinely ambiguous triples are ambiguous *because* their optimal value depends on which of the 500+ solutions you're in.

3. **Costamagna domains are too broad without visual matching.** Each unresolved triple gets ~73 candidates (the full Costamagna CV inventory minus confirmed values). Phase 14's stroke-guided domains gave ~5.2. Without visual sign matching to narrow the domains, the Costamagna inventory provides *less* constraint than the existing stroke-feature approach.

### What Visual Matching Would Add

Without images: 73 candidates per unresolved triple → 10^5.6 search space.
With visual matching (estimated 5-8 per triple): ~7³ ≈ 343 → 10^2.5 search space.

More importantly, visual matching would provide **independent evidence** for specific assignments rather than relying on corpus statistics (which are flat for these rare triples). The photographed signs in `data/GL.S.III.MISC.12/images/` (41 high-resolution images of the 1953 catalog) are the next resource to exploit.

### The ha → an Consensus

The one consensus change (`open_curve,open_curve,bench`: ha → an) is interesting because:
- EVA 'c' (the only syllabic char with this triple) maps to "an" instead of "ha"
- "an" is a common Latin/Italian syllable
- But this change affects only ~55 tokens, so its impact is negligible on aggregate metrics

---

## Dependency Chain

```
results/phase57_verdict.json ─┐
results/combined_refine.json  ─┤
results/triple_tiers.json     ─┤
syllabary_table.json          ─┼──→ cost_domains.json (58.1)
                                        │
                                        ├──→ cost_reduction.json (58.2)
                                        │
                                        └──→ cost_csp.json (58.3)
                                                   │
                                                   └──→ cost_compare.json (58.5)
                                                             │
                                                             └──→ phase58_verdict.json (58.6)
```

---

## Commands

```bash
voynich cost-domains       # Step 58.1: Build Costamagna-constrained domains
voynich cost-reduction     # Step 58.2: Compare domain sizes across phases
voynich cost-csp           # Step 58.3: Run CSP (greedy hill-climbing, 20 restarts)
voynich cost-compare       # Step 58.5: Compare best CSP solution vs T_P15
voynich phase58-verdict    # Validation gates and verdict
voynich phase58            # Run full Phase 58 pipeline
```

Runtime: ~110 seconds total (dominated by Step 58.3 CSP solve at ~102 seconds).
