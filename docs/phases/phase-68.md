# Phase 68: Rare Syllable Recovery

[← Phase Index](README.md) · [← Progression](../progression.md)

## Verdict: **PARTIAL_RESOLUTION** (5/5 gates)

## Summary

Phase 67 revealed that statistical optimization converges on common syllables ('di', 'co', 'se') for unresolved triples because those defaults improve every corpus-level metric. Phase 68 reframes the strategy: instead of optimizing corpus-level metrics, find rare syllable values through **local, token-level constraints** — paradigmatic patterns, formulaic matching, and confirmed-context analysis.

Seven independent tracks produce `triple_candidates` that feed into a majority-vote integration. Result: 1 RESOLVED triple (first since Phase 14), 5 LIKELY, dict hit +1.57%.

## Track Results

### Track 1: Fully-Decoded Token Exploitation (3/4 gates)

**Finding:** 22,823 tokens (63%) contain ONLY confirmed triples and validated coda markers — zero decode error.

- Dict-hit on this clean subset: 35.9%
- Longest consecutive fully-decoded run: 29 tokens
- 3,204 runs of length ≥ 3
- 12,816 partial tokens have fully-decoded neighbors (within ±2 positions)
- All 13 unresolved triples have confirmed-context windows

Top decoded words: din (816×), cor (807×), corar (782×), ne (611×), berar (583×), serar (573×).

**CLI:** `voynich full-tokens` · **Output:** `results/p68_full_tokens.json`

### Track 2: Within-Token Co-Occurrence (1/3 gates)

**Finding:** 30,703 (confirmed, unresolved) triple pairs counted within tokens. All 13 unresolved triples have co-occurrence data. However, Latin syllable bigram frequencies don't produce clear winners — every candidate scores similarly, and LOO validation on confirmed triples is not meaningful.

**CLI:** `voynich within-token` · **Output:** `results/p68_within_token.json`

### Track 3: Paradigmatic Analysis — Minimal Pairs (3/4 gates)

**Finding:** 4,914 minimal pairs found among 9,257 token types (pairs differing by exactly 1 EVA char). 2,149 are diagnostic (one confirmed, one unresolved). 11 triples constrained, but candidate sets remain large (62-63 candidates per triple).

The paradigmatic track contributed to 5 of 6 final changes — the strongest individual contributor.

**CLI:** `voynich paradigmatic` · **Output:** `results/p68_paradigmatic.json`

### Track 4: Expanded T1 Identification Pipeline (3/3 gates)

**Finding:** CVC-enhanced wildcard matching produces **223 unique-match identifications** (up from Phase 52's 22). The additional known characters from coda markers reduce wildcard positions, enabling more unique matches.

6 triples constrained at 61.8% mean consistency. Best: `ascender,plume,gallows` → 'tu' with 100% consistency (4 observations at both syllable positions).

**CLI:** `voynich expanded-t1` · **Output:** `results/p68_expanded_t1.json`

### Track 5: Formulaic Pattern Decoding (3/3 gates)

**Finding:** 82 recurring n-gram patterns in recipe sections (20,110 tokens). 1,482 matches against Circa Instans formulae. 10 triples constrained.

Top patterns: "ne ne" (64×), "berar berar" (47×), "ne corar" (46×) — consistent with pharmaceutical recipe formulae. Formula matching identified 'di' and 'st' as frequent implied values.

**CLI:** `voynich formula-decode` · **Output:** `results/p68_formulaic.json`

### Track 6: Distributional Constraint Propagation (0/2 gates)

**Finding:** Only 2 triples receive constraints from Phase 67's 39 distributional anchors. Per-token match data is too sparse. Produced `loop,sigmoid,bench` → 'sa' and `vertical,descender,suffix` → 'bo'.

**CLI:** `voynich distrib-constrain` · **Output:** `results/p68_distributional.json`

### Track 7: Edit-Distance Lattice (1/3 gates)

**Finding:** 458 token types have dictionary neighbors within ED=2 (72,812 total neighbors). 11 triples constrained but zero with clear dominance — the lattice is too diffuse at ED=2 with a 130K dictionary. Runtime: 268s.

**CLI:** `voynich ed-lattice` · **Output:** `results/p68_ed_lattice.json`

## Integration

Majority vote across 7 tracks (3+ = RESOLVED, 2+ = LIKELY, else UNRESOLVED):

| Triple | T_P15 | Phase 68 | Status | Tracks |
|--------|-------|----------|--------|--------|
| `ascender,crossbar,compound` | be | **de** | **RESOLVED** | full_tokens, paradigmatic, ed_lattice |
| `ascender,plume,gallows` | ga | **di** | LIKELY | paradigmatic, within_token |
| `crossbar,crossbar,rare` | fa | **ba** | LIKELY | paradigmatic, ed_lattice |
| `loop,sigmoid,bench` | ne | **de** | LIKELY | paradigmatic, ed_lattice |
| `loop,tail,bench` | la | **ce** | LIKELY | full_tokens, paradigmatic |
| `open_curve,hook,rare` | hi | **ba** | LIKELY | full_tokens, paradigmatic |
| 7 triples | (unchanged) | — | UNRESOLVED | no 2-track agreement |

**Evaluation:**

| Metric | T_P15 | Phase 68 | Delta |
|--------|-------|----------|-------|
| Dict hit | 29.04% | 30.61% | +1.57% |
| Signal words | 80 | 81 | +1 |
| Bigram z | 117.69 | 112.77 | −4.92 |

All 5 validation gates pass: no dict-hit regression, at least 1 resolved, signal preserved, bigram z within tolerance, at least 1 change proposed.

## Why Token-Level Beats Corpus-Level

Phase 67's 5 tracks operated at the corpus level — metrics like dict-hit, bigram z, and composite score are dominated by high-frequency tokens. A rare syllable appearing 200 times in a 36,000-token corpus barely moves any aggregate metric.

Phase 68's 7 tracks operate at the **token level**:

- Track 1 finds tokens where 100% of characters are known — revealing the readable core
- Track 3 finds tokens that differ by ONE character from a fully-confirmed token — directly constraining rare syllables
- Track 4 extends T1 identification with more known characters — unique matches fill specific rare values
- Track 5 matches formulaic patterns against known CI recipes — the formula tells you what the rare syllable must be

Each approach finds rare values through LOCAL constraints rather than GLOBAL optimization.

## Dependencies

```
results/combined_refine.json      (Phase 15)
results/triple_tiers.json         (Phase 28/53)
results/p67_distributional.json   (Phase 67, Track 5)  [Track 6 only]
    -> results/p68_full_tokens.json
    -> results/p68_within_token.json
    -> results/p68_paradigmatic.json
    -> results/p68_expanded_t1.json
    -> results/p68_formulaic.json
    -> results/p68_distributional.json
    -> results/p68_ed_lattice.json
    -> results/p68_integrate.json
```

## CLI Commands

```bash
voynich full-tokens        # Track 1: Fully-decoded token exploitation
voynich within-token       # Track 2: Within-token co-occurrence
voynich paradigmatic       # Track 3: Minimal pair analysis
voynich expanded-t1        # Track 4: CVC-enhanced T1 pipeline
voynich formula-decode     # Track 5: Formulaic pattern decode
voynich distrib-constrain  # Track 6: Distributional constraint propagation
voynich ed-lattice         # Track 7: Edit-distance lattice
voynich phase68-verdict    # Integration verdict
voynich phase68            # Full pipeline (~5 min)
```
