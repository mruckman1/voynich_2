# Phase 67: Multi-Angle Triple Resolution

[← Phase Index](README.md)

**Verdict: PARTIAL_RESOLUTION** (3/5 gates) — 8/13 unresolved triples receive LIKELY consensus (2 tracks agree), 0 reach RESOLVED (3+ agree). Evolutionary search improves dict-hit by +1.3%; distributional mapping validates the decode approach (4/4 gates). Voting collapses ambiguous triples toward common confirmed syllables rather than discovering distinct values. 56% decode error rate persists.

## Motivation

The 13 unresolved triples (10 LANDSCAPE_CONFIRMED + 3 GENUINELY_AMBIGUOUS) produce ~56% character-level error, cascading into failure of every word-level method (Phase 65 segmentation, Phase 66 LLM reading). Phase 67 attacks this bottleneck from five independent angles, then combines results by majority vote.

## Track Summary

| Track | Method | Gates | Key Result |
|-------|--------|-------|------------|
| 1 | Wildcard matching | 3/5 | 25.3% unique rate; 16 signal words recovered; selectivity only 1.13× |
| 2 | Frequency matching | 2/3 | LOO recall 41.7%; 9/13 triples narrowed to < 20 candidates |
| 3 | Feature prediction | 2/4 | Vowel LOO 41.7%; 13/13 predictions in Costamagna; syllable LOO 8.3% |
| 4 | Evolutionary optimization | 4/5 | +1.3% dict hit; 13/13 top-5 consensus; 12/13 changed; signal regressed |
| 5 | Distributional mapping | **4/4** | 39 anchors; 23.1% exact hit; 46.2% related hit; Procrustes alignment works |

## Track 1: Confidence-Weighted Wildcard Matching

**CLI:** `voynich wildcard-match`

Marks characters from confirmed triples as HIGH-confidence literals and characters from unresolved triples as LOW-confidence wildcards. Matches wildcard patterns against the dictionary.

- **25.3% unique match rate** (8,490 / 33,526 attempted tokens)
- **Null comparison:** null corpora achieve 22.3% → selectivity only 1.13× (FAIL > 1.5×)
- **Constraints:** 9 triples received character-level constraints, but none reached 70% consistency
- **Signal recovery:** 16 of 70 signal words recovered via unique matches (PASS ≥ 5)
- **Quartile gradient:** Q4 (high confidence) unique rate > 2× Q1 (PASS)

The high unique-match rate is partly structural — short decoded words match many dictionary entries regardless of correctness.

## Track 2: Frequency-Matched Triple Resolution

**CLI:** `voynich freq-match`

Matches triple frequency ranks in the Voynich corpus to syllable frequency ranks in Latin reference text (±30% tolerance).

- **LOO recall:** 5/12 = 41.7% (FAIL ≥ 80%) — frequency rank is not a reliable predictor for ~58% of triples
- **Domain reduction:** mean 18.9% of inventory retained (PASS < 50%)
- **Narrow triples:** 9 of 13 got < 20 candidates (PASS ≥ 5)
- The two highest-frequency triples (`loop,sigmoid,bench` rank 5, `vertical,descender,suffix` rank 7) retained 90% of the inventory — frequency matching works best for rare triples
- Most T_P15 values fell outside frequency-matched domains

## Track 3: Feature-Based Prediction

**CLI:** `voynich feat-predict`

Trains classifiers (KNN, Decision Tree, Logistic Regression, Naive Bayes) on the 12 confirmed triples' stroke features → syllable values.

- **Syllable LOO:** 8.3% (expected: 12 training samples, 10 target classes)
- **Onset LOO:** 25.0% (FAIL > 50%)
- **Vowel LOO:** 41.7% (PASS > 40%) — vowels are partially predictable from stroke features
- **Costamagna validity:** 13/13 predictions are in the inventory (PASS ≥ 8)
- Predictions cluster on common syllables: 'di' (4 triples), 'co' (2), 'be' (2), 'de' (2), 'ro' (2), 'se' (1)

## Track 4: Evolutionary Optimization

**CLI:** `voynich evo-optimize`

200-individual × 500-generation evolutionary algorithm. Fitness = dict-hit on 5,000-token subsample + Track 2/3 agreement bonus. Uses pre-compiled token structures for ~100× speedup over naive per-token decode.

- **Search space:** 10^16.4 (constrained by Tracks 2+3 domains)
- **Converged at generation 100:** 0.4186 → 0.4596 subsample fitness
- **Full-corpus improvement:** 29.04% → 30.36% dict-hit (+1.32pp)
- **Top-5 consensus:** 13/13 (robust single optimum within constrained domains)
- **12/13 triples changed** from T_P15 — nearly all unresolved assignments suboptimal for dict-hit
- **Signal regression:** 2,933 → 2,277 signal tokens (FAIL) — optimizing dict-hit doesn't preserve signal words

## Track 5: Distributional Mapping

**CLI:** `voynich distrib-map`

Builds PPMI+SVD distributional vectors for EVA token types and Latin word types. Aligns via Procrustes using 39 anchor pairs (EVA tokens whose CVC decode matches a Latin dictionary word).

- **All 4 gates passed** — strongest validation results
- **39 anchor pairs** for Procrustes alignment
- **Mean anchor cosine:** 0.302 (weak but meaningful)
- **Signal word validation:** 23.1% exact hit rate (6/26), 46.2% related hit rate (12/26)
- Validates that EVA tokens and Latin words occupy structurally similar distributional positions

## Integration

**CLI:** `voynich phase67-verdict`

Collects predictions from all 5 tracks per unresolved triple, applies majority vote.

### Voting Results

| Triple | Consensus | Votes | Status | T_P15 | Changed |
|--------|-----------|-------|--------|-------|---------|
| ascender,crossbar,compound | be | 1/3 | UNRESOLVED | be | — |
| ascender,crossbar,gallows | te | 1/3 | UNRESOLVED | te | — |
| ascender,loop,compound | to | 1/3 | UNRESOLVED | to | — |
| ascender,plume,gallows | de | 2/3 | LIKELY | ga | Yes |
| connector,connector,bench | co | 2/3 | LIKELY | ba | Yes |
| crossbar,crossbar,rare | di | 2/4 | LIKELY | fa | Yes |
| loop,sigmoid,bench | ne | 1/3 | UNRESOLVED | ne | — |
| loop,tail,bench | la | 1/3 | UNRESOLVED | la | — |
| open_curve,hook,rare | di | 2/4 | LIKELY | hi | Yes |
| open_curve,open_curve,bench | co | 2/3 | LIKELY | ha | Yes |
| sigmoid,hook,rare | se | 2/3 | LIKELY | fe | Yes |
| vertical,ascender,minim | di | 2/2 | LIKELY | do | Yes |
| vertical,descender,suffix | di | 2/3 | LIKELY | du | Yes |

### Final Evaluation

| Metric | T_P15 | Phase 67 | Delta |
|--------|-------|----------|-------|
| Dict hit | 29.04% | 29.17% | +0.13% |
| Signal words | 80 | 71 | −9 |
| Bigram z | 117.69 | 125.86 | +8.17 |

## Key Findings

1. **Distributional mapping works:** The Procrustes-aligned distributional space meaningfully connects EVA tokens to Latin words. This independently validates the tachygraphic decode hypothesis.

2. **Evolutionary search finds marginal improvement:** +1.3% dict-hit on the full corpus by reassigning 12/13 unresolved triples, but at the cost of signal word regression.

3. **Voting collapses to common syllables:** The 8 proposed changes all shift unresolved triples toward the most frequent confirmed syllables ('di', 'co', 'se'). This is a nearest-neighbor artifact — with limited discriminative data, all methods converge on the mode.

4. **Wildcard matching reveals structural ambiguity:** 25.3% of tokens produce unique dictionary matches even with wildcards, but null corpora achieve 22.3% — the dictionary is too large relative to the decoded word length for wildcard matching to be discriminative.

5. **Computational approaches have reached their ceiling:** Five independent methods (dictionary matching, frequency ranking, ML prediction, evolutionary optimization, distributional semantics) all fail to resolve the 13 unresolved triples. The bottleneck is fundamentally a lack of discriminative information in the decoded character stream. Visual stroke-matching against historical tachygraphic specimens remains the most promising path forward.

## CLI Commands

```bash
voynich wildcard-match     # Track 1: Wildcard matching
voynich freq-match         # Track 2: Frequency matching
voynich feat-predict       # Track 3: Feature prediction
voynich evo-optimize       # Track 4: Evolutionary optimization
voynich distrib-map        # Track 5: Distributional mapping
voynich phase67-verdict    # Integration
voynich phase67            # Full pipeline
```

## Dependency Graph

```
Track 2 (freq-match) ──→ constrained domains ──→ Track 4 (evo-optimize)
                                              ↗
Track 3 (feat-predict) ──→ predictions ───────→ Track 4 (evo-optimize)

Track 1 (wildcard-match) ─────────────────────→ Integration
Track 5 (distrib-map) ────────────────────────→ Integration
Track 4 (evo-optimize) ───────────────────────→ Integration
                                                    ↓
                                              Phase 67 Verdict
```

Tracks 1, 2, 3, 5 run independently. Track 4 depends on Tracks 2+3.
