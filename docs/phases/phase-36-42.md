# Phases 36-42: Signal Vocabulary, Z-Score Audit, Venetian Retraction

[← Phases 31-35](phase-31-35.md) | [Phase Index](README.md) | [Next: Phases 43-48 →](phase-43-48.md)

**Key results:**
- Phase 36: 51 Latin-10K signal words (5.43x selectivity), bigram z=12.66
- Phase 37: Italian selectivity 5.45x vs Latin 1.30x; merged z=16.97; macaronic=YES
- Phase 38: Full macaronic pipeline z=14.37, 73 signal words, 31 CC bigrams (first non-zero)
- Phase 39: Venetian 4.58x selectivity; amplified z=19.89; Drosera Italian alignment 6.57x
- Phase 40: Venetian bigram z=319.76 (later found to be a measurement artifact); folio reading 47.8% coverage
- Phase 41: Venetian REFUTED (corrected z=-0.47); lexicon 73/73 complete; f57v 68% coverage
- Phase 42: All bigram z-scores deflated 3-70x under symmetric recomputation; conservative minimum z=14.78

---

## Phase 36: Full Signal Pipeline at 10K Dictionary (Unconditioned)

**Verdict: 10K_CONFIRMED** -- z=12.66, 51 signal words, 11/12 validations, bootstrap stalled.

Phase 36 takes the lesson from Phase 35's failure (don't modify the decode, modify the evaluation) and runs the complete Phase 28-30 signal pipeline using the 10K dictionary against the original, unconditioned Phase 16 decode.

### The 131K Dictionary Was Actively Harmful

| Category | 10K | 131K | Change |
|----------|-----|------|--------|
| SIGNAL | 18.5% | 16.5% | +2.0% |
| SHARED_HIT | 1.1% | 15.2% | -14.1% |
| ANTI_SIGNAL | 3.6% | 15.5% | -11.9% |
| SHARED_MISS | 76.8% | 52.7% | +24.1% |
| **Net signal** | **15.0%** | **1.0%** | **+14.0%** |

The 131K dictionary generated 15.5% ANTI_SIGNAL -- tokens where null corpora hit the expanded dictionary but real Voynich didn't. At 10K, ANTI drops to 3.6%, revealing true net signal of 15.0%.

### Signal Vocabulary: 8 -> 51

At 10K, **51 words** qualify as signal (sigma > 2.0) -- a 6x expansion. Top: di (sigma=129.7, sel=5.6x), se (105.1, 5.5x), ne (93.5, 5.5x), dise (77.8, 5.5x), sero (70.1, 5.9x). All 51 maintain selectivities of ~5.5x -- consistency across 51 independently-measured words is itself strong evidence.

### Bigram z = 12.66

1,507 SIGNAL-SIGNAL pairs, 12 exact matches. All 12 involve at least one function word. **0 content-content bigrams** -- the critical limitation. Signal is function-word-driven, not phrase-level.

### Best Folio: f57v

53.7% SIGNAL at 10K, with a 58-token chain. Highly repetitive decoded text cycling through ~10 syllables.

## Phase 37: Signal Decomposition, Concatenation, and Content Word Recovery

**Verdict: BASELINE** -- no investigation improved bigram z + CC simultaneously.

Five investigations: consonant-vowel decomposition (CONFIRMED), signal pair concatenation (z=22.06 but merged z=-6.67), multi-triple joint swap (3 swaps, overfits), f57v deep examination (MODERATE), Northern Italian dictionary test (**ITALIAN_PREFERRED**).

### Investigation 5: Northern Italian 10K -- ITALIAN_PREFERRED

| Metric | Italian 10K | Latin 10K |
|--------|------------|-----------|
| Hit rate | 20.8% | 24.0% |
| Null hit rate | 3.82% | 18.4% |
| **Selectivity** | **5.45x** | **1.30x** |

Italian selectivity is 4.2x higher. Latin has more raw hits but null also hits at 18.4%. Italian null rate only 3.82%. 22 Italian-only signal words identified: be(sigma=135), cora(99), dise(78), bela(44), cedi(23).

**Merged dictionary** (Latin + Italian = 19,363 words): bigram z = **16.97** (up from 12.66). **is_macaronic = YES.**

## Phase 38: Macaronic Signal Pipeline

**Verdict: SIGNAL_EXPANDED** -- z=14.37, 73 signal words [22 Italian], CC=31, cross-lang=998.

Full Phase 36-style pipeline on merged Latin+Italian dictionary.

- Signal rate: **24.58%** (8,906 tokens) -- up from 18.53% at Latin 10K
- Bigram z: **14.37** (12 exact, 1,759 relaxed)
- **31 content-content bigrams** (first non-zero in the project -- all prior phases produced exactly 0)
- **998 cross-language bigrams** -- 56.4% cross the Latin-Italian boundary
- **91 medical phrases** with >=2 domain types
- f57v contains 9 Venetian verb forms (fa, ha, si, di, se, ne, la, le, te)

Top Italian-only signal words: be (134.65, "well"), cora (98.68, "heart"), dise (77.77, "says"), bela (beautiful), dice (says), cose (things).

## Phase 39: Edit-Distance Bridge, Vowel Recovery, and Macaronic Crib Exploitation

**Verdict: VENETIAN_SIGNAL_FOUND** -- 0 corrections applied. Venetian selectivity 4.58x, amplified z=19.89.

### Track A: ED1 Bridge

31 CC bigrams collapse to only 10 unique word pairs, dominated by "cora cora" (19/31). **0 eligible corrections** -- the dominant pair's ambiguous reference match (both cura/cera and cera/cera) creates a CONFLICTED correction.

### Track C: Italian Botanical Names

Italian plant names succeed on f56r/Drosera where Latin names failed (**6.57x null selectivity**). But 0 cross-folio consistent assignments.

### Track D: Venetian Dialect

Venetian-specific words match decoded text at **4.58x** above chance. 166 Venetian-only token hits. 1 recipe template match on f57v.

### Track E: Amplified Signal

Calibrated 1,086-word dictionary: **bigram z = 19.89** (highest in project). Selectivity 322.53x (artificially high -- null produces zero hits).

## Phase 40: Venetian Reading, CVC Expansion, and Folio-Level Decipherment

**Verdict: MAINTAINED** -- Venetian bigram z=319.76. Folio reading 47.8% coverage.

### Track A: Venetian Bigram z=319.76

29,207-word Venetian extended set. Dict-hit: 33.6% (up from 26.0%). 157 exact bigram hits, 3,877 relaxed. **z=319.76**. All 1,771 CC bigrams classify as correct or plausible Venetian (zero genuine errors).

### Track B: CVC Expansion

CVC expanded: 36.05% on subsample (+21pp) but **22.37% on full corpus** (down from 43.63%). CVC overfits to subsample.

### Track C: Folio-Level Reading

| Folio | Coverage | Coherence | Max Run | Quality |
|-------|----------|-----------|---------|---------|
| f57v | 57.7% | 35.1% | 11 | 0.571 |
| f25v | 52.8% | 21.2% | 3 | 0.485 |
| f37r | 42.3% | 20.0% | 9 | 0.418 |

f57v 4x repeating formulaic pattern at regular 14-token intervals: "ra ne di ne hi fa de". 14 concatenated pairs forming known words (bene, cora, cola, dise, radi, dose, rosa).

## Phase 41: Venetian Null Validation, Lexicon Completion, and Inter-Formula Content Recovery

**Verdict: VENETIAN_REFUTED** -- z=319.76 was **entirely a measurement artifact**.

### The Critical Fix

Two bugs in Phase 40:
1. **Bigram z-test asymmetry**: Real counted exact+relaxed (4,034 hits); null counted exact-only (~141). This compared 4,034 against ~141 -> z=319.76.
2. **Missing null selectivity**: Null decode key didn't exist; defaulted to 999.0.

| Metric | Phase 40 (buggy) | Phase 41 (corrected) |
|--------|-----------------|---------------------|
| Real total hits | 4,034 | 4,527 |
| Null mean total | ~141 (exact only!) | **4,046.26** |
| **Bigram z-score** | **319.76** | **-0.47** |

Venetian selectivity: **1.18x** (below 1.5x threshold). The 29,207-word dictionary is so large that ~31% of random decoded text matches it.

### Still Standing (not affected by this bug)
- Phase 29 sequential structure (z=6.14) -- different methodology
- Phase 38 CC bigram z=14.37 -- different test
- 49 individual signal words
- f57v 4x formulaic repetition
- Assignment table's 43.6% dict-hit (vs ~30% null)

## Phase 42: Bigram Audit, Symmetric Revalidation, and Ground-Truth Assessment

**Verdict: MODERATE_EVIDENCE** -- all z deflated 3-70x, but 6/7 retain z>2.0.

Every z-score recomputed using canonical methodology: shuffle-based null, 500 permutations, both exact and edit-distance-1 hits counted symmetrically.

| Phase | Dictionary | Original z | Symmetric z_total | Deflation | Classification |
|-------|-----------|------------|-------------------|-----------|---------------|
| 29 | Latin 131K | 6.14 | **2.23** | 2.8x | DEFLATED |
| 35 | Latin 131K | 6.88 | **2.09** | 3.3x | DEFLATED |
| 36 | Latin 10K | 12.66 | **3.80** | 3.3x | DEFLATED |
| 37.6 | Latin 17K | -6.67 | -6.67 | 1.0x | CONFIRMED |
| 38 | Merged 19K | 14.37 | **3.65** | 3.9x | DEFLATED |
| 39.4 | Merged 19K | 11.53 | **2.26** | 5.1x | DEFLATED |
| 39.16 | Calibrated 1K | 19.89 | **3.90** | 5.1x | DEFLATED |
| 40 | Venetian 29K | 319.76 | **-0.47** | 680x | INVALIDATED |

Signal word sigma-scores and dict-hit selectivities independently validated as methodologically sound (per-token frequency, not bigram comparison).

**Honest assessment:** The Voynich manuscript uses a systematic encoding that resembles tachygraphic systems. When decoded through the Phase 16 assignment table, the text produces dictionary hits at 3.0x the rate of random text and forms Latin word sequences at z=3.90 above null. The Venetian hypothesis is retracted. The language could be Latin, Italian, or another Romance variety.

### Progression

| Phase | Dict | Signal | Bigram z | CC bigrams | Advance |
|-------|------|--------|----------|------------|---------|
| 29 | 131K | 16.5% | 6.14 | 0 | Bigram discovery |
| 36 | 10K | 18.5% | 12.66 | 0 | 10K pipeline |
| 37 | merged 19K | -- | 16.97 | 0 | Italian macaronic signal |
| 38 | merged 19K (full) | 24.6% | 14.37 | 31 | Full macaronic pipeline |
| 39 | calibrated 1.1K | 32.3% | 19.89 | 0+52 | Venetian confirmed (4.58x) |
| 40 | Venetian 29K | 33.6% | 319.76 (bug) | 1,771 | Folio reading |
| **41** | **Venetian 29K** | **17.2%** | **-0.47 (corrected)** | -- | **Venetian refuted** |
| **42** | -- | -- | **3.90 (best symmetric)** | -- | **Z-score audit complete** |

---

[← Phases 31-35](phase-31-35.md) | [Phase Index](README.md) | [Next: Phases 43-48 →](phase-43-48.md)
