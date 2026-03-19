[← Phases 2-3](phase-02-03.md) | [Phase Index](README.md) | [Next: Phases 6-7 →](phase-06-07.md)

# Phases 4-5: Audit, Section Diagnosis, Morpheme Decoding

## Phase 4: Discriminant Audit, Section Diagnosis, Abugida Test, Multi-Language Comparison

Phase 4 audits whether Phase 3 findings are publishable, diagnoses cross-section inconsistencies, classifies the script type, and expands language comparison beyond Latin. Each step has a decision gate that determines whether subsequent steps are worth pursuing.

### Step 1: Discriminant Audit

Cross-references all Phase 3 null test results with core metrics to determine which findings genuinely discriminate the Voynich signal from null baselines.

| Component | Description | Module |
|-----------|-------------|--------|
| Null test summary | Load `null_test_results.json` and classify each metric as discriminating, partial, or non-discriminating across 4 null types | `phases/discriminant_audit.py` |
| Hypothesis linkage | Cross-reference with pre-registered hypotheses (D1–F4) for pass/fail status | `phases/discriminant_audit.py` |
| Critical findings | Flag F.4 (PMI), F.3 (bigram ranking), D.1 (length), E.3 (stability) as gate metrics | `phases/discriminant_audit.py` |

### Step 2: Section Consistency Diagnosis

Diagnoses why E.4 cross-section grid consistency is only 0.14 Jaccard — is it a Currier A/B signal, a small-sample artifact, or grid instability?

| Component | Description | Module |
|-----------|-------------|--------|
| 2A: Per-section grids | Build a grid for each of the 7 manuscript sections, compute H1, H2, occupancy | `phases/section_diagnosis.py` |
| 2B: Sample-size calibration | Subsample at 200–10,000 tokens, build grids, measure Jaccard vs full grid to find the minimum reliable sample size | `phases/section_diagnosis.py` |
| 2C: Currier A/B test | Aggregate Currier A (herbal_a) vs Currier B (all others except herbal_b's 181 tokens), compare entropy profiles, grid Jaccard, bigram JSD, bootstrap CI on H2 difference | `phases/section_diagnosis.py` |

### Step 3: Abugida Hypothesis Test

Tests whether the script is an abugida (consonant base + vowel modifier) rather than a pure syllabary or alphabet.

| Component | Description | Module |
|-----------|-------------|--------|
| 3A: Onset/nucleus decomposition | Decompose each glyph into (first stroke, last stroke) pairs; compute positional entropy at each glyph position | `phases/abugida_test.py` |
| 3B: Conditional entropy | Compute H(onset), H(nucleus), H(nucleus\|onset), MI(onset; nucleus). Key diagnostic: R = 1 - H(nucleus\|onset)/H(nucleus) | `phases/abugida_test.py` |
| 3C: Script type classification | Compare R and H-ratio against reference profiles for alphabet (<0.15), syllabary (0.15–0.55), abugida (0.20–0.70), abjad | `phases/abugida_test.py` |

### Step 4: Multi-Language Comparison

Ranks 7 languages across three independent metrics with bootstrap confidence intervals.

| Component | Description | Module |
|-----------|-------------|--------|
| Fingerprint ranking | Cosine similarity of Voynich entropy profile vs reference profiles for each language (100 bootstrap samples) | `phases/multi_language.py` |
| Bigram JSD ranking | Character bigram matrix JSD between Voynich and reference text (100 bootstrap samples) | `phases/multi_language.py` |
| PMI correlation ranking | Syllable-level PMI distribution correlation (100 bootstrap samples) | `phases/multi_language.py` |
| Combined ranking | Mean rank across 3 metrics, CI overlap test between #1 and #2 | `phases/multi_language.py` |

Available languages: Latin and Occitan (real corpora), Italian, German, Spanish, Hebrew, Arabic (synthetic text from word lists in `core/ciphers.py`).

## Phase 4.5: Integrating Prior Findings

Phase 4.5 applies three validated findings from the prior project as filters on existing analyses: Language B is notation (not natural language), Voynich tokens have morpheme structure (prefix+stem+suffix), and qo- tokens are likely mechanical markers.

### Priority A: Language A Isolation

Splits the corpus by Currier Language A/B and builds independent profiles for each, rerunning fingerprint, grid, and abugida analyses on each language separately.

| Component | Description | Module |
|-----------|-------------|--------|
| A.1: Corpus split | Filter tokens by IVTFF `$L=A/B` metadata, report per-language folio/token/type counts | `phases/language_a_isolation.py` |
| A.2: Language profiles | Build independent 37-dim fingerprint, syllabary grid, and abugida R values for each language | `phases/language_a_isolation.py` |
| A.3: A/B comparison | Bigram JSD, grid Jaccard, H₂ difference with bootstrap CI, vocabulary overlap | `phases/language_a_isolation.py` |
| A.4: Null test | Shuffle tokens into random pools of same size, compare real JSD to null distribution | `phases/language_a_isolation.py` |

### Priority B: Morpheme Grid Reinterpretation

Decomposes Voynich tokens into prefix+stem+suffix morphemes based on known EVA affix inventories, then tests whether morpheme roles map to specific grid axes.

| Component | Description | Module |
|-----------|-------------|--------|
| B.1: Morpheme decomposition | Greedy longest-first prefix/suffix matching against known EVA affixes | `phases/morpheme_grid.py` |
| B.2: Grid axis mapping | Build 2×K contingency tables (affix vs stem stroke distributions) per grid axis, chi-squared and JSD tests | `phases/morpheme_grid.py` |
| B.3: Entropy cross-validation | Verify affix axis has lower entropy than stem axis | `phases/morpheme_grid.py` |
| B.4: R-value reinterpretation | Relabel onset/nucleus as affix/stem, check if R values become linguistically natural | `phases/morpheme_grid.py` |
| B.5: Entropy stripping | Compare H₂(full tokens) vs H₂(stems only) to test whether affixes carry predictable grammatical info | `phases/morpheme_grid.py` |

### Priority C: qo- Token Removal

Profiles qo- prefixed tokens (starting with EVA `qo`, `qok`, `qot` ligatures) and measures the effect of removing them on all metrics.

| Component | Description | Module |
|-----------|-------------|--------|
| C.1: qo- identification | Tokenize EVA chars, check if first char is in `{qo, qok, qot}` | `phases/language_a_isolation.py` |
| C.2: Removal analysis | Build profiles with/without qo-, compare grids, entropy deltas, grid cell clustering | `phases/language_a_isolation.py` |

## Phase 5: Morpheme-Based Decoding

Phase 4.5 established that the syllabary grid encodes morphological structure (stem + affix axes, z > 500, p < 0.001). Phase 5 inverts the prior project's failed whole-token-to-whole-word approach: discover inflectional paradigms first, match paradigm shapes to candidate languages, then attempt phonetic assignment — with strict selectivity gates (> 1.5x) at every step. Each gate failure stops downstream phases.

### Phase 5.1: Paradigm Discovery

Groups tokens by shared stems, catalogs affix variations, and clusters paradigm shapes.

| Component | Description | Module |
|-----------|-------------|--------|
| 5.1a: Stem grouping | Group morpheme decompositions by exact stem string and by grid-cell equivalence (merge allographic variants) | `phases/paradigm_discovery.py` |
| 5.1b: Shape classification | Classify paradigms by (n_prefix_types, n_suffix_types) shape tuples | `phases/paradigm_discovery.py` |
| 5.1c: Hierarchical clustering | Cluster paradigms into 5 groups by shape feature vectors using scipy.cluster.hierarchy | `phases/paradigm_discovery.py` |
| 5.1d: Null test | Shuffle characters within tokens, re-decompose, compare mean paradigm size. Gate: selectivity > 1.5x | `phases/paradigm_discovery.py` |

### Phase 5.2: Paradigm-to-Language Matching

Matches Voynich paradigm shapes against Latin/Occitan morphological profiles; aligns affixes.

| Component | Description | Module |
|-----------|-------------|--------|
| 5.2a: Morphological profiles | Build expected paradigm-size distributions from Latin/Occitan profiles (weighted Gaussians: 40% noun, 30% verb, 20% adj, 10% invariable) | `phases/paradigm_match.py` |
| 5.2b: Shape matching | Compare Voynich vs reference distributions via JSD, Spearman rho, chi-squared | `phases/paradigm_match.py` |
| 5.2c: Affix alignment | Rank-based alignment of Voynich suffixes to Latin/Occitan endings | `phases/paradigm_match.py` |
| 5.2d: Null test | Shuffle + re-match. Gates: JSD separation > 20%, alignment consistency > 50% | `phases/paradigm_match.py` |

### Phase 5.3: Frequency-Based Stem Identification

Identifies top Voynich stems against expected Latin medical vocabulary using four compatibility criteria + cross-consistency.

| Component | Description | Module |
|-----------|-------------|--------|
| 5.3a: Stem ranking | Sort stems by total token count, select top 20 | `phases/stem_identification.py` |
| 5.3b: Compatibility scoring | Four scores per candidate: paradigm, frequency, section, affix compatibility (each 0–1) | `phases/stem_identification.py` |
| 5.3c: Optimal assignment | Build cost matrix, solve via Hungarian algorithm (linear_sum_assignment) for 1-to-1 mapping | `phases/stem_identification.py` |
| 5.3d: Cross-consistency | Verify no duplicate Latin targets, POS-compatible suffix sharing, frequency order preserved | `phases/stem_identification.py` |
| 5.3e: Dual null controls | (1) Shuffled text control; (2) Random-word control (frequency-matched non-medical vocabulary). Gates: selectivity > 1.5x on both | `phases/stem_identification.py` |

### Phases 5.4+5.5: Phonetic Decode and Validation

Phonetic value assignment (gated on Phase 5.3) and comprehensive validation battery.

| Component | Description | Module |
|-----------|-------------|--------|
| 5.4a: Character mapping | Align EVA chars to Latin chars via positional matching; majority vote per EVA char | `phases/phonetic_decode.py` |
| 5.4b: Grid organization | Map phonetic values to grid cells via onset×nucleus structure | `phases/phonetic_decode.py` |
| 5.4c: Corpus decoding | Apply phonetic table to all tokens; compute decoded text entropy and bigram JSD with Latin | `phases/phonetic_decode.py` |
| 5.5a: Null discrimination | 7 tests: 4 null types × key metrics, each must show selectivity > 1.5× | `phases/phonetic_decode.py` |
| 5.5b: Phonetic table tests | 5 tests: coverage, consistency, bigram JSD, value cardinality, grid coherence | `phases/phonetic_decode.py` |
| 5.5c: Cross-validation | Train on herbal_a, test on herbal_b; check decoded bigram JSD transfer | `phases/phonetic_decode.py` |
| 5.5d: Bootstrap stability | Resample corpus 1000×, rebuild table, verify consistency > 0.60 in 95% of iterations | `phases/phonetic_decode.py` |

**Hard prerequisite:** Phase 5.3 `gate_passed == True` required before Phases 5.4+5.5 execute.

---
[← Phases 2-3](phase-02-03.md) | [Phase Index](README.md) | [Next: Phases 6-7 →](phase-06-07.md)
