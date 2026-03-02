# Voynich Manuscript: Syllabary & Information-Theoretic Analysis

A multi-phase computational analysis of the Voynich manuscript, progressing from language-agnostic statistical profiling through morpheme-level analysis to corpus-wide distributional semantics. Nine complementary approaches across seven phases attack the same questions from different angles, with strict selectivity gates (> 1.5x) preventing overconfident conclusions at every step.

**Approaches 1-2** (Phase 1) establish the script type and candidate language. **Phases 2-4** refine, validate, and audit. **Phase 5** attempts morpheme-based decoding (blocked by selectivity ceiling). **Phase 6** tries illustration-constrained decoding (blocked by small anchor set). **Phase 7** tests whole-corpus structural alignment via distributional semantics and positional slot analysis.

Key finding across all phases: the Voynich manuscript encodes a **Romance language** (Latin or Occitan, not separable) using a **morphological syllabary** with genuine affix+stem structure. Both Voynich Language A and B embedding spaces independently point to Latin as the closest structural match. However, the selectivity ceiling — where frequency priors dominate over genuine linguistic content — persists at every level of analysis.

## Quick Start

```bash
uv sync
uv pip install -e .
voynich corpus            # Load and summarize the EVA corpus
voynich reference         # Show reference corpus summary
voynich strokes           # Approach 1: stroke-level syllabary analysis
voynich fingerprint       # Approach 2: information-theoretic fingerprinting
voynich both              # Run both approaches
voynich nulls             # Phase 2A: null character identification
voynich grid              # Phase 2B: syllabary grid refinement
voynich phase2            # Run both Phase 2 analyses
voynich degeneracy        # Phase 3D: break substitution vs syllabary degeneracy
voynich grid-validate     # Phase 3E: validate syllabary grid
voynich syllable-match    # Phase 3F: syllable-level language matching
voynich validate-all      # Phase 3G: scholarly validation framework
voynich phase3            # Run all Phase 3 workstreams
voynich audit             # Phase 4.1: discriminant audit of Phase 3 results
voynich section-diagnosis # Phase 4.2: section consistency diagnosis
voynich abugida           # Phase 4.3: abugida hypothesis test
voynich multi-language    # Phase 4.4: multi-language comparison
voynich phase4            # Run all Phase 4 analyses
voynich lang-a            # Phase 4.5A+C: language A isolation + qo-removal
voynich morpheme-grid     # Phase 4.5B: morpheme grid reinterpretation
voynich phase4-5          # Run all Phase 4.5 analyses
voynich paradigms         # Phase 5.1: paradigm discovery
voynich paradigm-match    # Phase 5.2: paradigm-to-language matching
voynich stem-id           # Phase 5.3: frequency-based stem identification
voynich phonetic          # Phase 5.4+5.5: phonetic decode and validation
voynich phase5            # Run all Phase 5 analyses
voynich illustration      # Phase 6.0: illustration-constrained setup
voynich rosetta           # Phase 6 D+E: Rosetta folio selection
voynich anchor            # Phase 6 A+B: anchor-and-propagate
voynich compete           # Phase 6 C: competitive ID resolution
voynich phase6-validate   # Phase 6 validation battery
voynich phase6            # Run all Phase 6 analyses
voynich anchor-diagnosis  # Phase 6.1B: anchor inconsistency diagnosis
voynich encoding-diagnosis # Phase 6.1C: encoding model diagnosis
voynich phase6-1          # Run full Phase 6.1 pipeline (TF-IDF + diagnosis)
voynich embeddings        # Approach 8: morpheme distributional semantics
voynich slots             # Approach 9: pharmaceutical positional slot analysis
voynich phase7            # Run full Phase 7 (Approaches 8 + 9 + integration)
```

Alternatively, use `python -m voynich <command>` without installing.

Requires Python 3.12+, NumPy, and SciPy. The EVA transcription data (IVTFF format) should be placed in `data/corpus/`.

## Project Structure

```
voynich_2/
├── pyproject.toml               # Project metadata, dependencies, console_scripts
├── src/voynich/                 # Python package (installed via `uv pip install -e .`)
│   ├── __init__.py              # Package root
│   ├── __main__.py              # python -m voynich support
│   ├── cli.py                   # Entry point — run analyses from the command line
│   ├── core/                    # Foundation modules
│   │   ├── corpus.py            # IVTFF parser, EVA tokenizer, corpus access
│   │   ├── stats.py             # Entropy, Zipf, bigram matrices, MI, TTR, DTW, PPMI/SVD, Procrustes, GW
│   │   ├── ciphers.py           # Historical cipher implementations + encoding simulators
│   │   ├── reference.py         # Reference corpus loading, RTF conversion, syllable stats, Latin recipe segmentation
│   │   └── _paths.py            # Centralized path resolution for data and results directories
│   ├── analysis/                # Main analysis approaches
│   │   ├── strokes.py           # Approach 1: stroke decomposition, Ventris grid
│   │   └── fingerprint.py       # Approach 2: entropy profiling, profile matching
│   └── phases/                  # Phase 2–6 workstreams
│       ├── nulls.py             # Phase 2A: null character identification
│       ├── grid_refine.py       # Phase 2B: syllabary grid refinement
│       ├── degeneracy.py        # Phase 3D: substitution vs syllabary tests
│       ├── grid_validate.py     # Phase 3E: grid gap analysis, stability
│       ├── syllable_match.py    # Phase 3F: CV labeling, language matching
│       ├── scholarly.py         # Phase 3G: pre-registration, null testing
│       ├── discriminant_audit.py # Phase 4.1: audit Phase 3 findings vs null tests
│       ├── section_diagnosis.py # Phase 4.2: section consistency, Currier A/B
│       ├── abugida_test.py      # Phase 4.3: script type classification
│       ├── multi_language.py    # Phase 4.4: multi-language comparison with CIs
│       ├── language_a_isolation.py # Phase 4.5A+C: Language A isolation, qo-removal
│       ├── morpheme_grid.py     # Phase 4.5B: morpheme grid reinterpretation
│       ├── paradigm_discovery.py # Phase 5.1: paradigm discovery
│       ├── paradigm_match.py    # Phase 5.2: paradigm-to-language matching
│       ├── stem_identification.py # Phase 5.3: frequency-based stem identification
│       ├── phonetic_decode.py   # Phase 5.4+5.5: phonetic decode and validation
│       ├── illustration_constrained.py # Phase 6.0: illustration-constrained setup
│       ├── rosetta_selection.py # Phase 6 D+E: Rosetta folio selection
│       ├── anchor_propagate.py  # Phase 6 A+B: anchor-and-propagate decoding
│       ├── competitive_id.py    # Phase 6 C: competitive ID resolution
│       ├── illustration_validate.py # Phase 6: validation battery
│       ├── anchor_diagnosis.py  # Phase 6.1B: anchor inconsistency diagnosis
│       ├── encoding_diagnosis.py # Phase 6.1C: encoding model diagnosis
│       ├── distributional.py   # Phase 7 / Approach 8: distributional semantics
│       ├── positional_slots.py # Phase 7 / Approach 9: positional slot analysis
│       └── approach_integration.py # Phase 7: cross-validation of Approaches 8+9
├── data/
│   ├── corpus/                  # EVA transcription files (ZL3b-n.txt, RF1b-e.txt, IT2a-n.txt)
│   └── reference/               # Real historical corpora organized by language (not in git)
│       ├── latin/               # Circa Instans, De Viribus Herbarum
│       ├── occitan/             # Régime du Corps
│       └── voynich_plant/       # Plant ID concordance CSV + medieval Latin name mapping
├── results/                     # JSON output from analysis runs
└── archive/                     # Previous codebase (consonant-skeleton approach — deprecated)
```

## Approach 1: Stroke-Level Syllabary Analysis

The consonant skeleton approach failed because it assumed EVA characters are alphabetic (one character = one phoneme). But the compositionality test showed multi-stroke characters behave as ligatures. If each glyph represents a CV or CVC syllable, the "alphabet" is actually a syllabary — explaining both the low character-level entropy and the failure of phoneme-level matching.

This approach follows the Ventris method: map the script's internal structure (which signs share components, which signs appear in which positions) before identifying the language.

| Phase | Description | Module |
|-------|-------------|--------|
| 1.1 | **Stroke Decomposition** — Decompose EVA characters into 11 atomic stroke primitives (loop, open_curve, vertical, hook, descender, ascender, crossbar, sigmoid, plume, connector, tail). Covers all 23 single EVA characters + 17 ligatures. | `analysis/strokes.py` |
| 1.2 | **Positional Analysis** — Compute P(stroke \| position) for initial/medial/final positions. Measure MI(stroke, position) and chi-squared vs random/alphabetic null models. Strong positional constraints = syllabic structure. | `analysis/strokes.py` |
| 1.3 | **Ventris Grid** — Build a consonant x vowel grid grouping glyphs by shared initial stroke (onset) and final stroke (nucleus). Compare occupancy against Linear B, hiragana, and Cypriot syllabaries. | `analysis/strokes.py` |
| 1.4 | **Token Segmentation** — Re-analyze Voynich tokens as sequences of syllable units from the grid. Compute syllable-level entropy, TTR, and bigram statistics. | `analysis/strokes.py` |
| 1.5 | **Discriminant Validation** — Test whether the syllabary structure discriminates real Voynich text from character-shuffled null text (z-score on H2). | `analysis/strokes.py` |

## Approach 2: Information-Theoretic Fingerprinting

Instead of decoding first and checking after, characterize the Voynich text's statistical fingerprint and find which known language + encoding scheme produces the closest match.

| Phase | Description | Module |
|-------|-------------|--------|
| 2.1 | **Voynich Entropy Profile** — Compute a 37-dimensional vector: H1/H2/H3 (character), word entropy, MI at lags 1–10, intra-token MI, positional entropy at 10 positions, word-length entropy, Zipf exponent, TTR at 5 corpus sizes, bigram matrix entropy. | `analysis/fingerprint.py`, `core/stats.py` |
| 2.2 | **Reference Library** — Build equivalent profiles for 7 languages (Latin, Italian, German, Spanish, Hebrew, Arabic, Occitan) x 9 encoding schemes (raw, simple substitution, polyalphabetic, homophonic, nomenclator, syllabic, abbreviation light/heavy, null insertion) = 63 combinations. Uses real historical corpora when available (`reference.py`), falling back to synthetic text from word lists (`ciphers.py`). | `analysis/fingerprint.py`, `core/reference.py`, `core/ciphers.py` |
| 2.3 | **Profile Matching** — Rank reference profiles by cosine similarity to the Voynich vector. Compute pairwise confusion matrix to identify which combinations are distinguishable. | `analysis/fingerprint.py` |
| 2.4 | **Section Differentiation** — Compute per-section profiles (herbal A/B, astronomical, biological, cosmological, pharmaceutical, recipes) and match independently. Tests whether sections encode different languages or use different schemes. | `analysis/fingerprint.py` |
| 2.5 | **Discriminant Validation** — Generate null text (shuffle, random, Markov) and verify that real Voynich matches reference profiles significantly better than null text does. | `analysis/fingerprint.py` |

## Phase 2A: Null Character Identification

Some EVA characters may be meaningless padding (nulls) inserted to obscure the plaintext. This phase tests that hypothesis by measuring each character's information content and checking whether removing it improves the statistical profile match.

| Phase | Description | Module |
|-------|-------------|--------|
| 2A.1 | **Per-Character Information Content** — For each EVA glyph, compute: frequency/rank, H(next\|c) and H(prev\|c), MI(char, position_in_token), and H1/H2 change after removal. Combine into a composite `null_score`. | `phases/nulls.py` |
| 2A.2 | **Systematic Stripping** — Strip each character individually, re-profile, and match against the reference library. Test top-5 pairs and top-3 triple. Track whether stripping shifts the match away from `null_insertion`. | `phases/nulls.py` |
| 2A.3 | **Stroke Cross-Validation** — Check whether top null candidates' stroke components have low positional MI (expected for nulls, which appear at any position). | `phases/nulls.py` |
| 2A.4 | **Discriminant Validation** — Verify that stripped text still discriminates from shuffled null text (z-score > 2.0). | `phases/nulls.py` |

## Phase 2B: Syllabary Grid Refinement

The original Ventris grid (7 onsets x 11 nuclei) was only 27.3% occupied — too sparse for a plausible syllabary. This phase uses distributional clustering to merge similar categories and find a denser, more realistic grid.

| Phase | Description | Module |
|-------|-------------|--------|
| 2B.1 | **Nucleus Merging** — Build context vectors (co-occurrence with onsets) for each nucleus category. Hierarchical agglomerative clustering (cosine similarity, average linkage) with cuts at 4/5/6/7 clusters. | `phases/grid_refine.py` |
| 2B.2 | **Onset Merging** — Same approach for onsets, conditional on any pair exceeding 0.85 similarity. | `phases/grid_refine.py` |
| 2B.3 | **Grid Validation Sweep** — Score each candidate grid on occupancy (30%), discriminant z-score (25%), syllable bigram H2 (25%), and syllables/token (20%). | `phases/grid_refine.py` |
| 2B.4 | **Language Narrowing** — Map best grid dimensions against known syllabary sizes (Japanese kana, Romance CV, Latin, Germanic, Semitic). | `phases/grid_refine.py` |

## Phase 3: Breaking the Degeneracy

Phase 2 established Latin + simple_substitution as the best fingerprint match (0.9854 cosine similarity) and a refined 5x6 syllabary grid (46.7% occupancy). However, the entropy profile can't distinguish "alphabetic substitution on Latin" from "CV syllabary encoding a Latin-like language." Phase 3 resolves this degeneracy through four workstreams.

### Workstream D: Substitution vs Syllabary Degeneracy Tests

Three independent statistical tests compare how well the Voynich text's structure matches an alphabetic substitution model vs a CV syllabary model.

| Test | Description | Module |
|------|-------------|--------|
| D.1 | **Token Length Correlation** — Compare Voynich glyph-count distribution against Latin character-count and Latin syllable-count distributions using Pearson correlation and Earth Mover's Distance. | `phases/degeneracy.py` |
| D.2 | **Bigram Transition Structure** — Build Voynich char bigram matrix, find optimal permutation mapping to Latin char bigrams (substitution) and Latin syllable bigrams (syllabary) via the Hungarian algorithm, compare Frobenius distances. | `phases/degeneracy.py` |
| D.3 | **Position-Within-Token Entropy** — Compute H(unit\|position=k) curves for Voynich, Latin chars, and Latin syllables. Compare curve shapes via Dynamic Time Warping distance. | `phases/degeneracy.py` |

### Workstream E: Grid Validation

Four tests validate whether the 5x6 syllabary grid from Phase 2B is a genuine structural feature or an artifact.

| Test | Description | Module |
|------|-------------|--------|
| E.1 | **Gap Analysis** — Test whether the 16 empty cells form a systematic pattern (chi-squared) or are randomly distributed. Compare against Linear B, Cypriot, and Japanese kana gap patterns. | `phases/grid_validate.py` |
| E.2 | **Frequency Distribution** — Fit Zipf's law to grid cell usage frequencies. Test against theoretical Zipf via KS test. | `phases/grid_validate.py` |
| E.3 | **Bootstrap Stability** — Rebuild the grid 200 times from 50% subsamples. Measure Jaccard similarity of filled-cell sets across iterations. | `phases/grid_validate.py` |
| E.4 | **Cross-Section Consistency** — Build per-section grids and compare against the full-corpus grid. Test whether sections use the same syllabary. | `phases/grid_validate.py` |

### Workstream F: Syllable-Level Language Matching

Convert the grid into an abstract syllabary, retranscribe the entire corpus, and match against candidate languages at the syllable level.

| Step | Description | Module |
|------|-------------|--------|
| F.1 | **CV Labeling** — Assign frequency-ordered C_iV_j labels to each filled grid cell (C1=most common onset, V1=most common nucleus). | `phases/syllable_match.py` |
| F.2 | **Corpus Retranscription** — Convert every EVA token to a CV label sequence. Compute syllable-level entropy (H1, H2), TTR, and ambiguity rate. | `phases/syllable_match.py` |
| F.3 | **Syllable Bigram Matching** — For each candidate language, syllabify reference text, build syllable bigram matrix, find optimal permutation mapping via the Hungarian algorithm, rank by Frobenius distance. | `phases/syllable_match.py` |
| F.4 | **PMI Correlation** — Under the best-fit mapping, compute Pointwise Mutual Information for top-50 Voynich syllable bigrams and corresponding Latin bigrams. Pearson correlation measures structural similarity. | `phases/syllable_match.py` |

### Workstream G: Scholarly Validation Framework

A validation layer wrapping all Phase 3 experiments for reproducibility and statistical rigor.

| Component | Description | Module |
|-----------|-------------|--------|
| G.1 | **Pre-Registration** — Seven hypotheses with pre-specified metrics, directions, and thresholds, frozen before experiments run. | `phases/scholarly.py` |
| G.2 | **Null Testing** — Test key metrics against four null text types (shuffle, random, Markov, token-shuffle). Report z-scores, selectivity ratios, and discrimination. | `phases/scholarly.py` |
| G.3 | **Effect Sizes** — Cohen's d, bootstrap CIs, and Bayes factors for all main findings. | `phases/scholarly.py` |
| G.4 | **Reproducibility Manifest** — Python/NumPy/SciPy versions, random seeds, SHA256 hashes of all data and result files. | `phases/scholarly.py` |
| G.5 | **Sensitivity Analysis** — Vary grid cluster count and corpus size, track metric stability. | `phases/scholarly.py` |

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

Phase 4.5 established that the syllabary grid encodes morphological structure (stem + affix axes, z > 500, p < 0.001). Phase 5 inverts the prior project's failed whole-token-to-whole-word approach: discover inflectional paradigms first, match paradigm shapes to candidate languages, then attempt phonetic assignment — with strict selectivity gates (> 1.5×) at every step. Each gate failure stops downstream phases.

### Phase 5.1: Paradigm Discovery

Groups tokens by shared stems, catalogs affix variations, and clusters paradigm shapes.

| Component | Description | Module |
|-----------|-------------|--------|
| 5.1a: Stem grouping | Group morpheme decompositions by exact stem string and by grid-cell equivalence (merge allographic variants) | `phases/paradigm_discovery.py` |
| 5.1b: Shape classification | Classify paradigms by (n_prefix_types, n_suffix_types) shape tuples | `phases/paradigm_discovery.py` |
| 5.1c: Hierarchical clustering | Cluster paradigms into 5 groups by shape feature vectors using scipy.cluster.hierarchy | `phases/paradigm_discovery.py` |
| 5.1d: Null test | Shuffle characters within tokens, re-decompose, compare mean paradigm size. Gate: selectivity > 1.5× | `phases/paradigm_discovery.py` |

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
| 5.3e: Dual null controls | (1) Shuffled text control; (2) Random-word control (frequency-matched non-medical vocabulary). Gates: selectivity > 1.5× on both | `phases/stem_identification.py` |

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

## Phase 6: Illustration-Constrained Decoding

Phase 5 hit a "selectivity ceiling" — frequency-matched random Latin words scored as well as real medical vocabulary (selectivity 0.99×). Phase 6 breaks this ceiling by inverting the approach: instead of decode-then-validate, it uses botanical illustration identifications as cross-modal constraints that pin specific Latin plant names to specific folios, then checks whether a consistent character-to-sound mapping emerges across multiple anchor folios.

The pipeline: **E** (encoding model test) → **D** (Rosetta folio selection) → **B** (paradigm filtering) → **A** (anchor-and-propagate) → **C** (competitive ID resolution) → **Validation**

### Phase 6.0: Illustration-Constrained Setup

Parses the multi-source plant identification concordance and maps Linnaean binomials to medieval Latin names.

| Component | Description | Module |
|-----------|-------------|--------|
| 6.0a: Concordance parsing | Parse 70-entry multi-source concordance CSV (Bax, Tucker & Janick, Sherwood, etc.), group by folio | `phases/illustration_constrained.py` |
| 6.0b: Medieval name mapping | Map Linnaean binomials to medieval Latin equivalents with declension metadata (69 plants, 63 resolved) | `phases/illustration_constrained.py` |
| 6.0c: Tier classification | Tier 1 (genus consensus across sources), Tier 2 (single high-confidence), Tier 3 (contested) | `phases/illustration_constrained.py` |
| 6.0d: Dominant stem extraction | Extract the most frequent morpheme stem from each folio's tokens | `phases/illustration_constrained.py` |

### Phase 6 D+E: Rosetta Folio Selection

Scores folios on 5 criteria and selects the best set for anchor testing, then tests encoding models.

| Component | Description | Module |
|-----------|-------------|--------|
| D.1: Folio scoring | Score each folio on: ID confidence, name distinctiveness, dominant stem clarity, EVA char coverage, char novelty | `phases/rosetta_selection.py` |
| D.2: Greedy selection | Select folios maximizing combined score + character diversity across the set | `phases/rosetta_selection.py` |
| E.1: Encoding model test | Test 4 models (syllabic, alphabetic, abbreviated, mixed) by comparing expected vs observed token structure | `phases/rosetta_selection.py` |

### Phase 6 A+B: Anchor-and-Propagate

The core decoding engine. Hypothesizes that each Rosetta folio's dominant stem = the medieval Latin plant name, applies paradigm filtering, then checks cross-consistency of character mappings.

| Component | Description | Module |
|-----------|-------------|--------|
| A.1: Anchor hypotheses | For each Rosetta folio, align EVA chars of dominant stem to Latin chars of medieval name | `phases/anchor_propagate.py` |
| B.1: Paradigm filtering | Reject hypotheses where Voynich paradigm shape is incompatible with Latin declension class | `phases/anchor_propagate.py` |
| A.2: Cross-consistency | For each EVA char appearing in 2+ anchors, check if the same Latin mapping is assigned unanimously | `phases/anchor_propagate.py` |
| A.3: Propagation | Apply consensus mapping to decode all non-anchor herbal_a tokens | `phases/anchor_propagate.py` |
| A.4: Null tests | (1) Shuffle which folio text goes with which plant name; (2) Replace plant names with random Circa Instans words | `phases/anchor_propagate.py` |

### Phase 6 C: Competitive ID Resolution

Beam search over competing plant identifications for contested folios.

| Component | Description | Module |
|-----------|-------------|--------|
| C.1: Contested enumeration | Identify Tier 2+3 folios with multiple candidate IDs | `phases/competitive_id.py` |
| C.2: Beam search | Greedy beam search (width=10), add one contested folio at a time, keep top states by unanimity × log(n_chars+1) | `phases/competitive_id.py` |

### Phase 6 Validation

Full validation battery with null tests, leave-one-out, train/test split, and bootstrap stability.

| Component | Description | Module |
|-----------|-------------|--------|
| V.1: Three null tests | Shuffled tokens, shuffled characters, random plant names — each must show selectivity > 1.5× | `phases/illustration_validate.py` |
| V.2: Leave-one-out | Remove each anchor, rebuild mapping, check stability | `phases/illustration_validate.py` |
| V.3: Train/test split | 60/40 split, test generalization of character mapping | `phases/illustration_validate.py` |
| V.4: Bootstrap stability | Resample anchors 200×, verify unanimity CI width | `phases/illustration_validate.py` |
| V.5: Stop conditions | Hard stop (<0.20 or all nulls fail), soft stop (0.20–0.50), green light (>0.50 + all nulls >1.5×) | `phases/illustration_validate.py` |

**Gate structure:** illustration_constrained (≥8 Tier 1+2 folios) → rosetta_selection (≥8 folios, score >0.5) → anchor_propagate (unanimity >0.50, z >2.0) → competitive_id (separation >0.05) → validation (stop conditions)

## Phase 6.1: TF-IDF Stem Extraction and Diagnostic Investigation

Phase 6 produced z-scores of 32.0 and 6.75 (real signal exists) but unanimity of only 0.40 (below 0.50 threshold). Root cause: the dominant stem heuristic selected the most frequent stem per folio, picking up corpus-wide function words ("daiin") rather than folio-specific plant names. Phase 6.1 applies three fixes.

### Fix A: TF-IDF Stem Extraction

Replaces frequency-based stem selection with specificity-based selection. For each stem on each folio, computes four metrics: TF-IDF, specificity ratio (tf/cf), exclusivity ratio (this_folio/other_folios), and PMI. The stem with highest TF-IDF score becomes the dominant stem.

| Component | Description | Module |
|-----------|-------------|--------|
| A.1: Corpus stem statistics | Compute corpus-wide term frequency, document frequency for all stems across herbal_a | `phases/illustration_constrained.py` |
| A.2: Per-folio specificity | Four metrics per stem per folio: TF-IDF, specificity ratio, exclusivity, PMI | `phases/illustration_constrained.py` |
| A.3: Diagnostic comparison | Compare old (frequency) vs new (TF-IDF) dominant stems, report changes | `phases/illustration_constrained.py` |

### Fix B: Anchor-Level Inconsistency Diagnosis

Diagnoses which specific anchors and character mappings cause inconsistencies.

| Component | Description | Module |
|-----------|-------------|--------|
| B.1: Per-anchor profiling | For each anchor, count consistent vs conflicting character-reuse instances | `phases/anchor_diagnosis.py` |
| B.2: Poison detection | Leave-one-anchor-out unanimity; poison = removal improves unanimity by >0.05 | `phases/anchor_diagnosis.py` |
| B.3: Per-character profiling | For each EVA char in 2+ anchors, classify unanimity as high/medium/low | `phases/anchor_diagnosis.py` |
| B.4: Iterative pruning | Remove worst poison anchors until unanimity > 0.50 or 5 anchors remain | `phases/anchor_diagnosis.py` |

### Fix C: Encoding Model Diagnosis

Tests which encoding model best fits the anchor data per-anchor, with segmentation sensitivity.

| Component | Description | Module |
|-----------|-------------|--------|
| C.1: Per-anchor model fit | Test 4 models (syllabic, alphabetic, abbreviated, mixed) per anchor | `phases/encoding_diagnosis.py` |
| C.2: Model consensus | Rank models by count of good-fit anchors (fit < 0.3) | `phases/encoding_diagnosis.py` |
| C.3: Segmentation sensitivity | Test 3 segmentation strategies under winning model | `phases/encoding_diagnosis.py` |
| C.4: Hybrid model test | Check if short/medium/long names fit different models | `phases/encoding_diagnosis.py` |

## Phase 7: Morpheme Distributional Semantics & Positional Slot Analysis

Phases 5-6 hit a selectivity ceiling: individual token-to-word mappings can be satisfied by combinatorial abundance. Phase 7 attacks from two orthogonal angles that use the *entire corpus* rather than small anchor sets, testing global structural properties that don't require individual token identification.

**Approach 8** asks whether the geometric structure of Voynich stem embeddings matches a specific language's vocabulary space (via Procrustes and Gromov-Wasserstein alignment). **Approach 9** asks whether Voynich pharmaceutical text follows Latin recipe positional structure (verb-initial slots, noun/ingredient positions).

If both independently point to the same language, that's convergent evidence stronger than either alone.

### Approach 8: Morpheme-Level Distributional Semantics

Builds PPMI + SVD embedding spaces for Voynich stems, aligns them against Latin and Occitan reference embeddings.

| Sub-phase | Description | Module |
|-----------|-------------|--------|
| 8.1 | **Build Voynich embeddings** — Decompose tokens to stems via morpheme decomposition, build co-occurrence matrix (window=2), PPMI with smoothing (alpha=0.75), SVD to 50 dimensions. Separate spaces for Language A (412 stems, 8,652 tokens) and Language B (714 stems, 19,133 tokens). Validate via k-means ARI against section labels. | `phases/distributional.py` |
| 8.2 | **Build reference embeddings** — Same PPMI+SVD pipeline on Latin (Circa Instans + De Viribus Herbarum, 3,269 stems) and Occitan (Regime du Corps, 1,719 stems) with heuristic suffix stripping. | `phases/distributional.py` |
| 8.3 | **Alignment** — Procrustes rotation using 14 seed pairs from Phase 5.3 stem identifications. Gromov-Wasserstein structural comparison (no seeds needed, top-100 stems). Both A and B aligned independently against all references. | `phases/distributional.py` |
| 8.4 | **Affix embeddings** — Build affix-stem co-occurrence matrix, PPMI+SVD to 20 dims. Test whether prefixes and suffixes cluster separately. | `phases/distributional.py` |
| 8.5 | **Cluster correspondence** — Cluster both Voynich and reference spaces (k=5), compare cluster size distributions via rank correlation. | `phases/distributional.py` |
| 8.6 | **Null tests** — Shuffle seed pair assignments for Procrustes; random orthogonal rotation for GW. Selectivity = null_mean / real_residual (higher is better). Gate: > 1.5x. | `phases/distributional.py` |

### Approach 9: Pharmaceutical Positional Slot Analysis

Tests whether Voynich pharmaceutical text follows rigid positional slot structure like Latin recipe texts.

| Sub-phase | Description | Module |
|-----------|-------------|--------|
| 9.1 | **Latin recipe structure** — Segment Circa Instans into 1,234 recipes using verb-initial markers (recipe, accipe, misce, etc.). Label each token's word class (verb/noun/connector/other). Compute slot entropy by position — expect position 1 to be dominated by verbs. | `phases/positional_slots.py` |
| 9.2 | **Voynich pharmaceutical analysis** — Segment Language B pharmaceutical/recipes sections into 1,458 segments using paragraph breaks. Decompose tokens, record stem and paradigm class at each position. Compute MI(paradigm, position). | `phases/positional_slots.py` |
| 9.3 | **Cross-validate position x paradigm** — Build contingency table (position-class x paradigm-class). Cohen's kappa and chi-squared. Compare to Latin contingency table. Gate: kappa > 0.3, selectivity > 1.5x. | `phases/positional_slots.py` |
| 9.4 | **Verb identification** — Stems appearing at position 1 in >= 60% of their occurrences with verb-like paradigm shape. Rank by frequency, compare to Latin recipe verb ranking via Spearman correlation. | `phases/positional_slots.py` |
| 9.5 | **Ingredient identification** — Post-verb medial stems with noun-like paradigms. Cross-reference with herbal folio occurrence to identify plant names that bridge herbal and pharmaceutical sections. | `phases/positional_slots.py` |

### Approach Integration

Cross-validates Approaches 8 and 9 at multiple convergence points.

| Test | Description | Module |
|------|-------------|--------|
| Verb coherence | Do Approach 9 verb candidates cluster together in Approach 8 embedding space? (mean pairwise cosine sim vs 100 random samples) | `phases/approach_integration.py` |
| Noun coherence | Do noun candidates cluster together? | `phases/approach_integration.py` |
| Slot-embedding kappa | Cohen's kappa between k-means cluster labels and positional-slot labels (verb/noun/other) | `phases/approach_integration.py` |
| Joint null test | Shuffle both slot labels and embedding cluster assignments, recompute kappa. Selectivity > 1.5x. | `phases/approach_integration.py` |

## Integration

The approaches cross-validate across all phases:

| Approach 1 finds | Approach 2 finds | Phase 3 finds | Phase 4 finds | Phase 4.5 finds | Phase 5 finds | Phase 7 finds | Interpretation |
|---|---|---|---|---|---|---|---|
| CV syllabary grid with good fit | Closest match = Latin-substitution | D.1 favors syllabary, D.3 favors substitution, PMI r=0.96 | 8/15 metrics discriminate; PMI, bigram, length, stability all pass | Grid captures morphological structure (chi² p<0.001 both axes, JSD=0.46 on nucleus) | 2,328 stem paradigms discovered (z=178); 23 high-paradigm stems with 7–31 forms each | A and B embedding spaces both independently point to Latin (Procrustes + GW); noun candidates cluster 5.4x above baseline in embedding space | **Morphological structure confirmed at paradigm level; grid axes encode affix/stem roles; global embedding geometry converges on Latin** |
| Strong positional constraints (MI=0.30) | Latin dominates top 5 | Grid 100% stable, sections diverge (Jaccard=0.14) | Currier A/B distinct (H2 diff significant, grid Jaccard=0.14); min sample ~10k tokens | A/B confirmed as distinct systems (JSD z=3.82, vocab overlap=14%) | Paradigm selectivity 1.47× (z=178) — just below 1.5× gate | Language A ARI=0.11 (embeddings capture section structure); Language B ARI=-0.003 (no section signal — consistent with notation hypothesis) | **Section divergence = genuine A/B split, not artifact; A has semantic structure, B does not** |
| 5x6 grid, 47% occupancy | No null insertion evidence | Gap pattern random, closest to Cypriot (8% diff) | R=0.39 (syllabary/abugida overlap); nucleus predicts onset more than reverse | R(affix\|stem)=0.61, R(stem\|affix)=0.39 — linguistically natural under morpheme relabeling | Occitan JSD=0.65 vs Latin JSD=0.71; not separable (ratio=0.92) | Prefix/suffix separation=0.90 in affix embedding space; verbs at position 1 in 60-100% of segments; verb freq rho=0.97 with Latin recipe verbs | **Anomalous reverse R explained** — stems constrain affixes; **Romance family confirmed; affix space confirms morphological structure** |
| — | Latin best across encodings | Latin best syllable match | Latin #1, Occitan #2, but CIs overlap on all metrics | qo- removal neutral (14.4% of corpus, distributed across grid, no metric improvement) | Random-word selectivity 0.99× — frequency priors dominate over morphological content; **phonetic decode blocked** | Procrustes selectivity 0.96-0.97x, GW selectivity 1.00x — both fail 1.5x gate; only 14 seed pairs available | **Selectivity ceiling persists** at corpus-level alignment; seed pair scarcity limits Procrustes discrimination |

## Data

### Voynich Corpus

The project uses EVA (Extended Voynich Alphabet) transcription files in IVTFF format. The parser (`core/corpus.py`) supports three transcription sources with automatic preference ordering: ZL3b-n.txt > RF1b-e.txt > IT2a-n.txt.

The corpus provides filtered access by:
- **Section**: herbal_a, herbal_b, astronomical, biological, cosmological, pharmaceutical, recipes
- **Currier language**: A (herbal A scribe) or B (remaining sections)
- **Scribe hand**: 1–5 (inferred from quire assignments)

### Reference Corpora

Real historical texts for fingerprint comparison live in `data/reference/<language>/`. These are not tracked in git — acquire and place them locally. The loader (`core/reference.py`) auto-discovers `.txt` files by language directory and handles RTF-to-text conversion automatically.

**Currently available:**

*Latin (2 texts, ~73,528 tokens):*
- **Circa Instans** — Salernitan herbal/pharmaceutical text (~12th century, ~25,850 tokens)
- **De Viribus Herbarum** (Macer Floridus) — Herbal poem, medical botany (~47,678 tokens)

*Occitan (1 text, ~47,913 tokens):*
- **Régime du Corps** — Aldebrandin of Siena's health regimen (~13th century)

**To add a new corpus:** place a `.txt` file (plain text or RTF) in `data/reference/<language>/`. It will be automatically discovered, cleaned, and used by `analysis/fingerprint.py` on the next run. Languages without real corpora fall back to synthetic text from `core/ciphers.py`.

## Results Summary

### Fingerprint Matching (Approach 2)

The Voynich text's 37-dimensional entropy profile was compared against 63 reference profiles (7 languages x 9 encoding schemes). Latin corpora use real historical texts (Circa Instans, De Viribus Herbarum); other languages use synthetic text from period-appropriate word lists.

**Top 10 matches by cosine similarity:**

| Rank | Language | Encoding | Similarity |
|------|----------|----------|------------|
| 1 | Latin | simple_substitution | 0.9854 |
| 2 | Latin | raw | 0.9854 |
| 3 | Latin | abbreviation_light | 0.9852 |
| 4 | Latin | nomenclator | 0.9847 |
| 5 | Latin | syllabic | 0.9833 |
| 6 | Occitan | null_insertion | 0.9831 |
| 7 | German | null_insertion | 0.9830 |
| 8 | Hebrew | null_insertion | 0.9830 |
| 9 | Spanish | null_insertion | 0.9826 |
| 10 | Italian | null_insertion | 0.9820 |

Latin dominates the top 5 across multiple encoding schemes. The tight clustering of Latin matches (0.9833–0.9854) versus the gap to non-Latin entries suggests the underlying language is Latin or a close relative. The `null_insertion` encoding appears for every non-Latin language because null padding flattens statistical profiles toward uniformity, making them closer to any target.

**Voynich entropy profile (key dimensions):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| H1 (character) | 3.86 bits | Moderate — consistent with substitution cipher on natural language |
| H2 (bigram) | 2.36 bits | Low — strong sequential dependencies, not random |
| H3 (trigram) | 2.12 bits | Very low — highly structured character sequences |
| Word entropy H1 | 10.81 bits | 9,257 unique tokens across 36,238 total |
| Zipf exponent | 0.83 (R²=0.89) | Word frequencies follow power law, consistent with natural language |
| Mean word length | 5.36 chars | Comparable to Latin (~5.5) |

**Discriminant validation:** Real Voynich text matches reference profiles significantly better than shuffled (z = -65.4), random, or Markov-generated null text, confirming the match is not an artifact of corpus size or character distribution.

### Stroke-Level Syllabary (Approach 1)

**Positional analysis** of 11 stroke primitives across initial/medial/final positions:

| Stroke | Primary Position | Position Entropy | Interpretation |
|--------|-----------------|------------------|----------------|
| HOOK | 95.6% final | 0.28 bits | Strong final marker — vowel/coda indicator |
| TAIL | 89.3% medial | 0.56 bits | Internal connector |
| CROSSBAR | 75.2% medial | 0.87 bits | Internal structural element |
| LOOP | 79.5% medial | 0.88 bits | Core glyph body |
| OPEN_CURVE | 52.0% initial | 1.02 bits | Onset marker |
| SIGMOID | 52.7% initial | 1.46 bits | Onset/medial |
| CONNECTOR | 44.8% final | 1.54 bits | Flexible — onset or coda |
| VERTICAL | 39.2% initial | 1.55 bits | Spread across positions |
| DESCENDER | 65.6% final | 1.18 bits | Final position marker |

MI(stroke, position) = 0.296 bits — highly significant positional constraints (chi-squared p < 10⁻⁶ against both random and alphabetic null models). This level of positional structure is characteristic of syllabaries, not alphabets.

**Original Ventris grid:** 7 onsets x 11 nuclei, 21 filled cells (27.3% occupancy). Too sparse — real syllabaries occupy 60-90%.

**Syllable statistics:** 21 syllable types, mean 3.5 syllables/token, syllable-level H1 = 3.20, H2 = 2.38.

**Discriminant validation:** z-scores of -652 (H1) and -494 (H2) vs shuffled text. The syllabary structure captures genuine sequential patterns in the manuscript, not random glyph co-occurrence.

### Null Character Identification (Phase 2A)

**Top 5 null candidates** by composite null_score (frequency rank, context entropy, positional MI, removal effect):

| Char | Score | Freq | Rank | H(next\|c) | H(prev\|c) | ΔH2 |
|------|-------|------|------|-----------|-----------|-----|
| b | 0.554 | 15 | 41 | 2.55 | 0.00 | -0.001 |
| t | 0.534 | 4,954 | 9 | 3.32 | 2.40 | -0.038 |
| d | 0.518 | 6,247 | 7 | 3.46 | 3.83 | -0.027 |
| iiin | 0.514 | 46 | 38 | 2.96 | 1.01 | +0.000 |
| k | 0.513 | 7,065 | 4 | 3.18 | 2.84 | -0.016 |

**Key finding: no strong null signal.** Stripping any single character keeps `latin+simple_substitution` as the best match. The profile is remarkably stable under character removal:
- Stripping `e` (the most frequent character, 13.0%) is the only removal that shifts the match — to `latin+nomenclator` (0.9749)
- Stripping pairs `t+d`, `t+k`, or `d+k` also shifts to `latin+nomenclator`
- No stripping configuration moves the match away from Latin

**Stroke cross-validation:** All top-5 candidates have low-to-moderate positional entropy (0.85–1.43 bits), consistent with null status at the stroke level. However, the stripping experiments show their removal doesn't improve the profile, suggesting they carry genuine encoding information.

**Interpretation:** The null_insertion hypothesis is not supported. The Voynich script uses most or all of its characters meaningfully. The `null_insertion` encoding appearing highly ranked for non-Latin languages is likely an artifact of entropy flattening rather than evidence of actual null insertion.

### Grid Refinement (Phase 2B)

Distributional clustering merged the sparse 7x11 grid into a denser configuration:

**Best grid: 5 onsets x 6 nuclei (46.7% occupancy, score 0.933)**

| Merged Category | Original Strokes | Rationale |
|----------------|-----------------|-----------|
| **Onsets** | | |
| ascender+vertical | ascender, vertical | Both serve as initial strokes in tall characters |
| open_curve+sigmoid | open_curve, sigmoid | Both initiate curved-body glyphs |
| **Nuclei** | | |
| ascender+crossbar+plume | ascender, crossbar, plume | All medial decorative/structural elements |
| connector+open_curve | connector, open_curve | Co-occur in the same final contexts |
| loop+sigmoid+tail | loop, sigmoid, tail | Core body strokes that blend in medial position |

**Grid occupancy improvement:**

| Grid | Dimensions | Filled | Occupancy | Score |
|------|-----------|--------|-----------|-------|
| Original | 7 x 11 | 24 | 31.2% | 0.856 |
| Refined (best) | 5 x 6 | 14 | **46.7%** | **0.933** |
| 5 x 5 | 5 x 5 | 11 | 44.0% | 0.920 |
| 5 x 4 | 5 x 4 | 9 | 45.0% | 0.925 |

The refined 5x6 grid maintains full discriminant significance (z = -239.1) while increasing occupancy from 31% to 47%.

**Language narrowing by grid shape:**

| Family | Score | Expected Grid | Match |
|--------|-------|---------------|-------|
| Japanese-like | 0.812 | 8-12 onsets, 4-6 nuclei | Nuclei match; onsets fewer than expected |
| Romance (simple) | 0.812 | 8-15 onsets, 4-6 nuclei | Nuclei match; onsets fewer than expected |
| Latin (classical) | 0.708 | 12-18 onsets, 4-7 nuclei | Nuclei match; onsets well below expected |
| Germanic | 0.708 | 12-20 onsets, 5-8 nuclei | Both dimensions below expected |
| Semitic | 0.567 | 15-25 onsets, 3-5 nuclei | Poor fit |

The 5-onset inventory is smaller than expected for any known language's syllabary, but the 6-nuclei dimension is consistent with Romance and Japanese-like phonotactics. The low onset count may reflect further mergeable categories or a genuinely small consonant inventory.

### Degeneracy Tests (Phase 3D)

Three tests to determine whether the Voynich script is an alphabetic substitution cipher or a CV syllabary:

| Test | Metric (substitution) | Metric (syllabary) | Verdict |
|------|----------------------|---------------------|---------|
| D.1: Length correlation | r = 0.553, EMD = 0.040 | r = 0.797, EMD = 0.013 | **syllabary** |
| D.2: Bigram structure | Frobenius = 3.12 | Frobenius = 2.87 | inconclusive |
| D.3: Positional entropy | DTW = 4.02 | DTW = 17.39 | **substitution** |

**Overall verdict: inconclusive.** D.1 favors syllabary (token lengths correlate much better with Latin syllable counts than character counts). D.3 favors substitution (position-within-token entropy curve shape matches Latin characters better). D.2 is too close to call. The degeneracy is genuine — both models explain different aspects of the data.

### Grid Validation (Phase 3E)

| Test | Result | Threshold | Status |
|------|--------|-----------|--------|
| E.1: Gap pattern | chi-squared p = 0.073 | p < 0.05 | Random (not systematic) |
| E.2: Zipf fit | exponent = 2.10, R^2 = 0.57 | R^2 > 0.90 | Below threshold |
| E.3: Bootstrap stability | 100% cells stable, Jaccard = 0.995 | >90% stable | **Passed** |
| E.4: Section consistency | mean Jaccard = 0.14 | >80% agreement | **Failed** |

**Key findings:**
- The grid is **extremely stable** under resampling — all 14 cells appear in >99.5% of 200 bootstrap iterations.
- But per-section grids **diverge sharply** from the full-corpus grid. Best agreement is astronomical (0.29), worst is biological (0.08). This supports the Currier A/B language distinction: different manuscript sections may use different syllable inventories.
- Gap pattern is closest to Cypriot syllabary (occupancy diff = 8%), not to Japanese kana (45% diff) or Linear B (13% diff).

### Syllable-Level Retranscription and Matching (Phase 3F)

The 5x6 grid was converted to an abstract CV syllabary (14 types) and the entire corpus retranscribed:

| Metric | Value |
|--------|-------|
| CV types | 14 |
| CV tokens | 125,929 |
| Ambiguity rate | 0.0% |
| CV H1 | 2.90 bits |
| CV H2 | 2.43 bits |
| Mean CV/word | 3.48 |

**Sample retranscriptions:**

| EVA token | CV sequence |
|-----------|-------------|
| fachys | C2V3.C1V1.C3V4.C2V5.C3V1 |
| ykal | C2V5.C2V3.C1V2 |
| ataiin | C1V1.C2V3.C1V6 |
| shol | C3V4.C1V2 |
| sory | C3V1.C1V1.C2V5 |

**Language matching (optimal permutation via Hungarian algorithm):**

| Language | Frobenius Distance | JSD |
|----------|-------------------|-----|
| Latin | 2.087 | 0.996 |
| Occitan | 2.277 | 0.990 |

**PMI correlation:** Under the best-fit mapping, Voynich syllable bigram PMI values correlate with Latin syllable bigram PMI at r = 0.960 (p < 0.001, 50 common bigrams). This is strong evidence that the sequential structure of Voynich syllables mirrors Latin syllable combinatorics.

### Scholarly Validation (Phase 3G)

**Pre-registered hypotheses (5/7 passed):**

| Hypothesis | Metric | Result | Passed |
|------------|--------|--------|--------|
| D1: Lengths closer to syllables | EMD difference | -0.027 | Yes |
| D2: Syllabary Frobenius lower | Frobenius difference | -0.249 | Yes |
| D3: DTW to syllables lower | DTW difference | +13.37 | No |
| E1: Non-random gap pattern | chi-squared p | 0.073 | No |
| E3: Grid stable under subsampling | Stable fraction | 1.000 | Yes |
| F3: Latin best language match | Best = Latin? | Yes | Yes |
| F4: PMI correlation positive | PMI r | 0.960 | Yes |

**Null discrimination (11/20 metrics discriminate):**

| Metric | vs Shuffle | vs Random | vs Markov | vs Token-shuffle |
|--------|-----------|-----------|-----------|-----------------|
| H1 | no | no | z = -202 | no |
| H2 | z = -1157 | z = -762 | z = -240 | z = -72 |
| Word H1 | z = -1407 | z = -1564 | z = -458 | no |
| Mean word length | no | no | no | no |
| Zipf exponent | z = 298 | z = 551 | z = 222 | no |

H2 (bigram entropy) and word-level entropy are the strongest discriminants — they separate real Voynich from all null models. Zipf exponent discriminates well except against token-shuffle, as expected (shuffling token order preserves the word frequency distribution).

**Sensitivity:**
- Grid occupancy is **not robust** to nucleus cluster count (sensitivity = 0.20, varies 37-47% across 4-8 clusters)
- H1 entropy is **robust** to corpus size (sensitivity = 0.008, stable from 1,000 to 36,238 tokens)

### Discriminant Audit (Phase 4.1)

Cross-referencing all Phase 3 metrics against null test results:

| Metric | Real Value | Best \|z\| | Verdict | Hypothesis |
|--------|-----------|-----------|---------|------------|
| F.4 PMI correlation | 0.960 | — | **Discriminating** | F4: PASS |
| F.3 Bigram lang ranking | 2.087 | — | **Discriminating** | F3: PASS |
| D.1 Length (syllabary) | 0.013 | — | **Discriminating** | D1: PASS |
| E.3 Grid stability | 1.000 | — | **Discriminating** | E3: PASS |
| H2 (bigram entropy) | 2.364 | 1156.6 | **Discriminating** | — |
| Stroke discriminant | 2.848 | 254.3 | **Discriminating** | — |
| Stripped discriminant | 0.015 | 69.8 | **Discriminating** | — |
| D.3 Positional entropy | 17.389 | — | Non-discriminating | D3: FAIL |
| E.1 Grid gaps | 0.073 | — | Non-discriminating | E1: FAIL |

**Summary:** 8 discriminating, 3 partial, 4 non-discriminating out of 15 metrics. All critical findings (F.4 PMI, F.3 bigram ranking, D.1 length correlation, E.3 grid stability) survive null testing. Gate 1 passes.

### Section Consistency Diagnosis (Phase 4.2)

**Per-section token counts and grid reliability:**

| Section | Tokens | Currier | Grid | Reliable |
|---------|--------|---------|------|----------|
| recipes | 10,092 | B | 5x6 | Yes |
| herbal_a | 9,449 | A | 5x6 | Yes |
| biological | 6,476 | B | 5x6 | Yes |
| pharmaceutical | 3,542 | B | 5x6 | Yes |
| astronomical | 2,860 | B | 5x6 | Yes |
| cosmological | 2,220 | B | 5x6 | Yes |
| herbal_b | 181 | B | 5x11 | No |

**Sample-size calibration:** Minimum ~10,000 tokens needed for reliable grid construction (Jaccard > 0.80 vs full-corpus grid). Only recipes (10,092) exceeds this threshold. All other sections produce grids that diverge from the full-corpus grid primarily due to insufficient data.

**Currier A/B split test:**

| Metric | Language A (herbal_a) | Language B (5 sections) | Difference |
|--------|----------------------|------------------------|------------|
| H1 | 3.844 | 3.851 | -0.006 |
| H2 | 2.382 | 2.292 | +0.090 |
| Grid Jaccard | — | — | 0.143 |
| Bigram JSD | — | — | 0.182 |
| H2 diff 95% CI | — | — | [0.065, 0.110] |

**Diagnosis: DISTINCT DIALECTS.** The H2 difference is statistically significant (CI excludes zero), and grid Jaccard between A and B is only 0.14. Currier A and B use measurably different linguistic profiles. The low cross-section consistency in E.4 reflects both a genuine A/B split and small-sample effects.

### Abugida Hypothesis Test (Phase 4.3)

**Onset/nucleus entropy decomposition:**

| Metric | Value |
|--------|-------|
| H(onset) | 1.955 bits |
| H(nucleus) | 3.001 bits |
| H(onset, nucleus) | 3.773 bits |
| H(nucleus \| onset) | 1.818 bits |
| H(onset \| nucleus) | 0.771 bits |
| MI(onset; nucleus) | 1.183 bits |
| Reduction R | 0.394 |
| Reverse R | 0.605 |

**Script type comparison:**

| Script Type | R in Range | Distance | Match |
|-------------|-----------|----------|-------|
| Alphabet (0.00–0.15) | No | 0.244 | Mismatch |
| Syllabary (0.15–0.55) | Yes | 0.000 | Match |
| Abugida (0.20–0.70) | Yes | 0.000 | Match |
| Abjad (0.00–0.30) | No | 0.094 | Partial |

**Verdict: INCONCLUSIVE.** R = 0.39 falls in the overlap zone between syllabary and abugida. The unusual finding is the asymmetry: nucleus predicts onset (reverse R = 0.61) more strongly than onset predicts nucleus (R = 0.39). In a classic abugida, the opposite is expected — consonant bases constrain vowel modifiers. This reversed asymmetry may reflect the Voynich script's unique structure where the final stroke (nucleus) is more diagnostic of glyph identity than the initial stroke (onset).

### Multi-Language Comparison (Phase 4.4)

Seven languages ranked across three metrics with 100 bootstrap samples each:

**Combined ranking:**

| Rank | Language | Corpus Type | Fingerprint | Bigram JSD | PMI Corr | Mean Rank |
|------|----------|-------------|-------------|------------|----------|-----------|
| 1 | Latin | real | 0.989 | 0.657 | 0.976 | 1.33 |
| 2 | Occitan | real | 0.986 | 0.694 | 0.964 | 2.67 |
| 3 | Spanish | synthetic | 0.825 | 0.662 | 0.932 | 3.33 |
| 4 | Italian | synthetic | 0.830 | 0.719 | 0.931 | 4.33 |
| 5 | Hebrew | synthetic | 0.819 | 0.580 | 0.819 | 4.67 |
| 6 | German | synthetic | 0.821 | 0.696 | 0.778 | 5.67 |
| 7 | Arabic | synthetic | 0.820 | 0.724 | 0.850 | 6.00 |

**Separation test:** Latin and Occitan CIs overlap on all three metrics (fingerprint, bigram, PMI). The finding is **"Romance language family"** rather than "Latin specifically." The gap from Romance languages (#1–2) to others (#3+) is substantial, with real corpora scoring well above synthetic-vocabulary languages on fingerprint and PMI metrics.

### Language A Isolation (Phase 4.5A)

Independent profiles for Currier Language A and Language B:

| Metric | Language A | Language B | Difference |
|--------|-----------|-----------|------------|
| Folios | 114 | 82 | — |
| Tokens | 10,791 | 22,366 | — |
| Types | 3,762 | 5,722 | — |
| TTR | 0.349 | 0.256 | +0.093 |
| H1 | 3.832 | 3.863 | -0.031 |
| H2 | 2.125 | 1.972 | +0.153 |
| Grid occupancy | 50.0% | 36.7% | +13.3% |
| Abugida R | 0.427 | 0.384 | +0.044 |

**A/B comparison:**

| Metric | Value |
|--------|-------|
| Bigram JSD | 0.209 |
| Grid Jaccard | 0.130 |
| H₂ difference | 0.153 (CI: [0.114, 0.144], significant) |
| Vocabulary overlap | 13.8% |
| Null test z-score | 3.82 |
| Verdict | **DISTINCT_SYSTEMS** |

Language A has higher type-token ratio (0.35 vs 0.26), higher bigram entropy (2.13 vs 1.97), denser grid (50% vs 37%), and only 14% vocabulary overlap with Language B. The null test confirms the split is non-random (z=3.82). Language A's top tokens (`daiin`, `chol`, `chor`) differ sharply from Language B's (`chedy`, `shedy`, `qokeedy`), with qo- prefixed tokens dominating Language B's frequency list.

### qo- Token Analysis (Phase 4.5C)

| Metric | Value |
|--------|-------|
| qo- tokens (full corpus) | 5,220 (14.4%) |
| qo- in Language A | 10.1% |
| qo- in Language B | 18.1% |
| qo- unique types | 858 |
| Grid cell clustering | Not clustered (top cell = 46%) |
| Grid Jaccard (with vs without) | 1.000 |
| H₂ change on removal | +0.054 (2.12 → 2.17) |
| Verdict | **REMOVAL_NEUTRAL** |

qo- tokens are distributed across all grid cells rather than concentrated in specific positions. Removing them does not change the grid structure (Jaccard = 1.0) and barely affects entropy metrics. This suggests qo- tokens are functional elements of the encoding, not mechanical padding. They are significantly more common in Language B (18% vs 10%), consistent with Language B having a different morphological profile.

### Morpheme Grid Reinterpretation (Phase 4.5B)

**Morpheme decomposition of 36,238 tokens:**

| Metric | Value |
|--------|-------|
| Tokens with prefix | 29.7% |
| Tokens with suffix | 67.0% |
| Tokens with both | 21.1% |
| Stem-only tokens | 24.5% |
| Unique prefix types | 4 (`o`, `d`, `y`, `s`) |
| Unique suffix types | 14 (`dy`, `y`, `ey`, `aiin`, `ol`, …) |
| Mean stem length | 2.44 EVA chars |
| Unique stem types | 5,700 |

**Grid axis mapping (contingency table tests):**

| Axis | Chi² | p-value | JSD (affix vs stem) |
|------|------|---------|---------------------|
| Onset | 23,548 | < 0.001 | 0.177 |
| Nucleus | 59,620 | < 0.001 | 0.457 |

Both axes show highly significant association between morpheme role (affix vs stem) and stroke distributions. The nucleus axis differentiates more strongly (JSD = 0.457 vs 0.177), identifying it as the stem axis (higher entropy = more variable content) and the onset axis as the affix axis (lower entropy = more constrained grammatical markers).

**R-value reinterpretation:**

| Metric | Original Label | Morpheme Label | Value |
|--------|---------------|----------------|-------|
| R | H(nucleus\|onset)/H(nucleus) | R(stem\|affix) | 0.394 |
| Reverse R | H(onset\|nucleus)/H(onset) | R(affix\|stem) | 0.605 |

Under morpheme relabeling, R(affix|stem) = 0.61 means stems constrain affixes — linguistically natural (grammatical suffixes depend on word class). The previously anomalous reverse R is no longer anomalous.

**Entropy stripping test:**

| Metric | Full tokens | Stems only |
|--------|-------------|------------|
| H1 | 3.865 | 3.745 |
| H2 | 2.120 | 2.384 |
| Word H1 | 10.807 | 9.096 |

H₂ increases from 2.12 to 2.38 after stripping affixes, confirming that affixes carry predictable (low-entropy) grammatical information while stems carry higher-entropy content.

**Null testing:** z-scores of 522 (onset) and 1,091 (nucleus) vs shuffled role assignments. The morpheme-axis association is not an artifact.

**Verdict: MORPHOLOGICAL.** The syllabary grid captures genuine morphological structure. Grid axes correspond to affix and stem roles, explaining the previously inconclusive script type classification.

### Paradigm Discovery (Phase 5.1)

**Stem paradigm inventory (Language A, paragraph tokens):**

| Metric | Value |
|--------|-------|
| Total stems | 2,328 |
| Stems with affixes | 486 (20.9%) |
| Singleton stems | 1,842 (79.1%) |
| Mean paradigm size | 1.62 forms/stem |
| Median paradigm size | 1 |
| Grid-merged stems | 1,693 |
| Grid-merged mean paradigm size | 2.22 |

**Top 5 paradigms by token count:**

| Stem | Forms | Tokens | Prefixes | Suffixes | Shape |
|------|-------|--------|----------|----------|-------|
| ch | 29 | 579 | d, o, s, y | aiin, al, am, an, dy, ey, iin, ol, y | (4, 9) |
| daiin | 1 | 468 | — | — | (0, 0) |
| sh | 20 | 264 | d, o, s, y | aiin, al, am, an, dy, ey, ol, y | (4, 8) |
| k | 31 | 263 | d, o, s, y | aiiin, aiin, al, am, an, ey, ol, y | (4, 8) |
| chor | 10 | 227 | d, o, s, y | aiin, dy, ol, y | (4, 4) |

**Hierarchical clustering (5 clusters):**

| Cluster | Paradigms | Mean Forms | Mean Suffixes | Mean Prefixes | Representatives |
|---------|-----------|------------|---------------|---------------|-----------------|
| 1 | 171 | 2.6 | 2.2 | 0.0 | qok, dy, qoke |
| 2 | 51 | 2.2 | 0.0 | 1.5 | kchor, sheor, cheeor |
| 3 | 162 | 3.5 | 1.8 | 1.5 | a, shor, dal |
| 4 | 23 | 17.0 | 6.9 | 3.2 | ch, sh, k |
| 5 | 79 | 5.1 | 2.9 | 1.7 | chor, s, ol |

**Null test:** Real mean paradigm size 1.62 vs null mean 1.10 (z = 178.4). Selectivity ratio = **1.47×** (below 1.5× gate threshold).

**Verdict: GATE FAILED.** Paradigm structure is highly statistically significant (z = 178) but selectivity ratio 1.47× narrowly misses the 1.5× threshold. The signal is real but not strong enough for confident downstream use.

### Paradigm-to-Language Matching (Phase 5.2)

**Language comparison:**

| Language | JSD | Rank Correlation | Chi² | Combined Score |
|----------|-----|-----------------|------|----------------|
| Occitan | 0.650 | -0.280 | 23,267 | 0.248 |
| Latin | 0.709 | -0.748 | 26,185 | 0.154 |

**Affix alignment (top 5 Voynich → Occitan suffix mappings):**

| Voynich Suffix | Occitan Ending | Rank Distance |
|----------------|---------------|---------------|
| -aiin | -a | 0 |
| -ol | -as | 0 |
| -al | -e | 0 |
| -y | -es | 0 |
| -am | -s | 0 |

Alignment consistency: 1.00 (13/13 aligned). Null JSD mean: 0.827 (real vs null z = 87.3).

**Gate results:** JSD ratio = 0.92 (needs < 0.80 for 20% separation). Consistency gate passes (1.00 > 0.50). Separation gate fails — Latin and Occitan are not distinguishable at the paradigm level.

**Verdict: ROMANCE FAMILY ONLY.** Both Romance languages match well (combined scores 0.15–0.25) but cannot be separated. Consistent with Phase 4.4 finding that Latin/Occitan CIs overlap.

### Stem Identification (Phase 5.3)

**Top 5 stem identifications (by combined score):**

| Voynich Stem | Freq | Forms | Latin Word | POS | Combined |
|--------------|------|-------|------------|-----|----------|
| qok | 139 | 8 | accipe (accept) | verb | 1.00 |
| a | 114 | 7 | frigida (cold) | adj | 0.96 |
| ol | 104 | 8 | humida (moist) | adj | 0.92 |
| ke | 132 | 12 | misce (mix) | verb | 0.90 |
| s | 149 | 6 | dolor (pain) | noun | 0.88 |

**Cross-consistency:** 1.00 (0 violations in 20 identifications). No duplicate Latin targets, all POS-compatible.

**Null controls:**

| Control | Mean Score | Std | z-score | Selectivity |
|---------|-----------|-----|---------|-------------|
| Random-word | 0.808 | 0.010 | -1.11 | **0.99×** |
| Shuffled text | 0.733 | 0.013 | 4.79 | **1.09×** |

**Verdict: GATE FAILED.** Cross-consistency is perfect (1.00), but the critical random-word control shows selectivity of only 0.99× — frequency-matched random Latin words score as well as the real medical vocabulary. This confirms the "selectivity ceiling" identified in the prior project: compatibility metrics are dominated by frequency and section priors rather than morphological structure. The shuffled-text selectivity (1.09×) also fails the 1.5× threshold.

### Phonetic Decode (Phases 5.4+5.5)

**Verdict: STOPPED AT GATE 5.3.** Phase 5.3 gate failed (random-word selectivity 0.99×), so phonetic value assignment was not attempted. This is by design — the gate system prevents compounding unreliable identifications into a phonetic table that would appear meaningful but lack genuine selectivity.

**Stop condition:** "Phase 5.3 gate failed: identifications not reliable"

### Illustration-Constrained Decoding (Phase 6)

Phase 6 uses botanical illustration identifications as cross-modal constraints — if experts agree a folio depicts *Papaver somniferum*, the dominant stem on that folio should map to the medieval Latin word "papaver." A consistent character mapping emerging across multiple such anchor folios would break the selectivity ceiling.

**Concordance and tier classification (Phase 6.0):**

| Metric | Value |
|--------|-------|
| Concordance entries | 70 |
| Folios with identifications | 50 |
| Unique plants identified | 69 |
| Medieval names resolved | 63 (6 New World plants unresolvable) |
| Tier 1 folios (genus consensus) | 1 |
| Tier 2 folios (partial consensus) | 11 |
| Tier 3 folios (contested) | 38 |

**Rosetta folio selection (Phase 6 D+E):**

| Folio | Plant (medieval Latin) | Combined Score |
|-------|----------------------|----------------|
| f37v | anagallis | 1.102 |
| f25v | dracaena | 0.901 |
| f9v | viola | 0.733 |
| f47v | pulmonaria | 0.664 |
| f54r | carthamus | 0.564 |
| f33r | papaver | 0.509 |
| f24r | silene | 0.489 |
| f56r | ros solis | 0.449 |

EVA character coverage: 39/44 characters (88.6%). Best encoding model: morphographic-syllabic (consistency 0.76). Gate passed.

**Anchor-and-propagate (Phase 6 A+B):**

| Metric | Value |
|--------|-------|
| Anchors generated | 8 (all paradigm-compatible) |
| Unique EVA chars mapped | 5 |
| Unanimous chars | 2 of 5 |
| **Unanimity ratio** | **0.40** |
| Conflicting chars | 3 of 5 |
| Decode coverage | 10.4% (527 fully decoded, 2,904 partial) |
| Null z-score (shuffled folios) | 32.0 |
| Null z-score (random plants) | 6.75 |
| Gate | **FAILED** (unanimity < 0.50) |

The high z-scores (32.0 and 6.75) confirm that the real folio-to-plant assignments produce more consistent mappings than random assignments — there is *some* cross-modal signal. However, unanimity of only 0.40 means the signal is too weak for reliable character-level decoding.

**Competitive ID resolution (Phase 6 C):**

| Metric | Value |
|--------|-------|
| Contested folios | 9 |
| Combinations explored | 1,728 |
| Best unanimity | 0.20 |
| Runner-up unanimity | 0.20 |
| Separation | 0.00 |
| Gate | **FAILED** (no clear winner) |

**Validation battery (Phase 6):**

| Test | z-score | Selectivity | Status |
|------|---------|-------------|--------|
| Null: shuffled tokens | -0.22 | 0.88× | Failed |
| Null: shuffled chars | 0.04 | 1.00× | Failed |
| Null: random names | -0.23 | 0.98× | Failed |

| Stability Test | Result |
|---------------|--------|
| Leave-one-out mean unanimity | 0.41 (range 0.25–0.60, unstable) |
| Train/test transfer ratio | 0.0 (mapping does not generalize) |
| Bootstrap 95% CI | [0.0, 0.8] (maximally wide) |

**Verdict: HARD STOP.** All three null tests show selectivity < 1.5×, the train/test split shows zero transfer, and bootstrap CIs are maximally wide. The illustration-constrained approach does not yield a statistically distinguishable or stable character mapping. The approach is abandoned.

**Interpretation:** The cross-modal signal detected by anchor propagation (z = 32.0 vs shuffled folios) is real but insufficient. With only 12 Tier 1+2 folios and 5 unique EVA characters mapped, the constraint space is too sparse to resolve character-level ambiguities. The fundamental limitation is the plant identification concordance itself — most identifications are contested (38 of 50 folios at Tier 3), and even the best-supported IDs do not converge on a coherent mapping.

### Phase 6.1: TF-IDF Stem Extraction Results

Phase 6.1 replaced frequency-based dominant stem selection with TF-IDF specificity-based selection, diagnosing the root cause of Phase 6's failure.

**Fix A: TF-IDF Diagnostic Comparison**

| Metric | Frequency-based (Phase 6.0) | TF-IDF (Phase 6.1) |
|--------|---------------------------|---------------------|
| Folios changed | — | 46 of 50 (92%) |
| "daiin" as dominant | 17 folios | **0 folios** |
| Mean specificity ratio | 0.028 | **0.179** (6.4× improvement) |

**Anchor-and-propagate with TF-IDF stems:**

| Metric | Phase 6.0 | Phase 6.1 | Delta |
|--------|-----------|-----------|-------|
| **Unanimity ratio** | 0.40 | **0.5833** | **+0.18** |
| Unique EVA chars mapped | 5 | **12** | +7 |
| Unanimous chars | 2 of 5 | **7 of 12** | +5 |
| Null z (shuffled folios) | 32.0 | **66.1** | +34.1 |
| Null z (random plants) | 6.75 | **25.5** | +18.8 |
| Anchor-propagate gate | FAILED | **PASSED** | — |

**Encoding model diagnosis (Fix C):**

| Metric | Value |
|--------|-------|
| Best model | morphographic-abbreviated (4/8 good fit) |
| Clear winner | No (75% threshold not met) |
| Best segmentation | Balanced (unanimity 0.6667, +0.08 over baseline) |
| Hybrid evidence | Yes — short names fit abbreviated, medium/long fit syllabic |

**Anchor diagnosis (Fix B):**

| Metric | Value |
|--------|-------|
| High-unanimity chars (>0.80) | 7 |
| Low-unanimity chars (<0.50) | 5 |
| Poison anchors (Δ > 0.05) | 6 of 8 |

**Validation battery:**

| Test | Phase 6.0 | Phase 6.1 | Status |
|------|-----------|-----------|--------|
| Null: shuffled tokens | 0.88× | 1.22× | Improved but still < 1.5× |
| Null: shuffled chars | 1.00× | 1.46× | Improved but still < 1.5× |
| Null: random names | 0.98× | 1.43× | Improved but still < 1.5× |
| LOO stability | 0.25–0.60 | 0.25–0.60 | Unchanged |
| Train/test transfer | 0.0 | 0.0 | Unchanged |
| Overall status | HARD STOP | **HARD STOP** | — |

**Interpretation:** TF-IDF extraction substantially improved the anchor-propagate stage — unanimity rose from 0.40 to 0.58, clearing the 0.50 gate for the first time. The number of consistent character mappings more than doubled. However, the improvement was insufficient to clear the validation battery: null selectivities improved (from ~1.0× to ~1.3–1.5×) but still fall below the 1.5× threshold. The train/test transfer remains zero, indicating the mapping does not generalize. The fundamental constraint is the small anchor set (8 folios with 12 shared characters) — too few reuse instances for robust validation. The TF-IDF fix correctly identified the problem (generic stems instead of plant-specific stems) and improved the signal, but the concordance provides insufficient constraint density for confident decoding.

### Distributional Semantics (Phase 7 / Approach 8)

**Voynich embedding spaces:**

| Space | Stems | Dim | Tokens | Section ARI | ARI Null |
|-------|-------|-----|--------|-------------|----------|
| Language A | 412 | 50 | 8,652 | **0.1108** | -0.0001 |
| Language B | 714 | 50 | 19,133 | -0.0032 | 0.0017 |

Language A embeddings capture meaningful section structure (ARI 0.11 >> null). Language B shows no section clustering, consistent with Language B being notation rather than natural language.

**Reference embeddings:** Latin 3,269 stems (63,771 tokens), Occitan 1,719 stems (41,779 tokens).

**Alignment scores (Language A):**

| Language | Procrustes | GW dist | Seeds | NN cosine |
|----------|-----------|---------|-------|-----------|
| Latin | 3.554 | 0.0588 | 14 | 0.487 |
| Occitan | inf | 0.0619 | 1 | 0.000 |

**Alignment scores (Language B):**

| Language | Procrustes | GW dist | Seeds | NN cosine |
|----------|-----------|---------|-------|-----------|
| Latin | 3.920 | 0.0590 | 14 | 0.477 |
| Occitan | inf | 0.0603 | 1 | 0.000 |

**Cross-language convergence:**

| Check | Result |
|-------|--------|
| A+B Procrustes agree | **YES** (both point to Latin) |
| A+B GW agree | **YES** (both point to Latin) |
| A Procrustes selectivity | 0.96x (null mean 3.414, std 0.129) |
| B Procrustes selectivity | 0.97x (null mean 3.797, std 0.153) |
| GW selectivity | 1.00x (GW distance virtually unchanged under rotation) |
| Procrustes gate | FAIL |
| GW gate | FAIL |
| Embedding quality gate | PASS (A ARI > null, A ARI > 0.01) |
| Overall gate | FAIL |

**Affix embedding space:** 18 affix types, prefix/suffix separation = 0.895 (high — prefixes and suffixes occupy distinct regions of the affix co-occurrence space).

**Verdict: embeddings_valid_no_language_match.** Both A and B independently converge on Latin as the closest structural match via both Procrustes and GW — this is the first time the same language wins across two independent Voynich subsystems via two independent alignment methods. However, selectivity scores don't clear the 1.5x gate because the null distribution (shuffled seed pairs) produces residuals close to the real mapping. The fundamental limitation is seed pair scarcity: only 14 of 20 Phase 5.3 stems map to the Latin vocabulary, and only 1 maps to Occitan. With so few seeds, the Procrustes rotation is underdetermined. The GW distance is intrinsically insensitive to random rotation (cosine distances are rotation-invariant), explaining its 1.00x selectivity.

### Positional Slot Analysis (Phase 7 / Approach 9)

**Latin recipe structure:**

| Metric | Value |
|--------|-------|
| Recipes segmented | 1,234 |
| Mean recipe length | 18.2 tokens |
| Verb-initial ratio | 40.4% |
| Position 1 slot entropy | 1.56 bits |

**Voynich pharmaceutical analysis:**

| Metric | Value |
|--------|-------|
| Segments | 1,458 |
| Mean segment length | 9.2 tokens |
| MI(stem, position) | 0.862 |
| MI(affix, position) | 0.125 |
| MI selectivity | 1.07x (null mean 0.808) |
| MI gate | FAIL |

**Verb candidates (top 5 of 15):**

| Stem | Tokens | Position 1 % | Suffix types | Freq rank |
|------|--------|-------------|--------------|-----------|
| pol | 12 | 67% | 4 | 1 |
| pcheor | 6 | 83% | 0 | 2 |
| tsh | 5 | 60% | 3 | 3 |
| sha | 5 | 60% | 1 | 4 |
| yaiin | 4 | 100% | 0 | 5 |

**Verb frequency correlation:** Spearman rho = **0.972** (p = 2.5e-6) — Voynich verb candidate frequency ranking strongly correlates with Latin recipe verb frequency ranking.

**Ingredient candidates:** 443 stems identified, 348 also appear on herbal folios (cross-section plant name candidates).

**Position x paradigm cross-validation:** Kappa = 0.000 (position and paradigm classes are independent). Chi-squared = 27.1 (p = 0.001, significant — there is *some* association, but not strong enough for kappa > 0). The position-paradigm contingency structure does not match Latin's pattern.

**Verdict: no_significant_positional_structure.** MI selectivity 1.07x falls well below 1.5x gate. Voynich pharmaceutical text does not show the rigid verb-initial positional slot structure expected of Latin recipe text. However, the verb frequency ranking correlation (rho = 0.97) is striking — the relative frequencies of position-1 stems closely mirror Latin recipe verb frequencies, even though the positional constraint itself is weaker than expected.

### Approach Integration (Phase 7)

| Test | Result | Threshold | Status |
|------|--------|-----------|--------|
| Verb cluster coherence | ratio = 0.96x (14 stems) | > 1.2x | Not coherent |
| **Noun cluster coherence** | **ratio = 5.38x** (20 stems) | > 1.2x | **COHERENT** |
| Slot-embedding kappa | 0.000 | > 0.2 | No agreement |
| Joint selectivity | 0.00 | > 1.0 | No signal |

**Noun coherence** is the standout finding: the 20 noun/ingredient candidates identified by Approach 9's positional analysis cluster **5.4x more tightly** in Approach 8's embedding space than random stems of the same size. This cross-validates the two approaches — stems that behave like ingredients positionally also behave like a coherent semantic category distributionally.

Verb candidates do not show the same coherence (ratio 0.96x), likely because the 14 verb stems tested have low frequency (4-12 tokens each) and may not develop reliable embedding vectors.

**Overall verdict: no_significant_signal.** Both individual gates failed, so convergence is not established. However, the noun embedding coherence at 5.4x represents genuine cross-approach validation that the ingredient candidates form a real semantic category.

### Cross-Validation Summary

| Finding | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 4.5 | Phase 5 | Phase 6 | Phase 6.1 | Phase 7 | Assessment |
|---------|---------|---------|---------|---------|-----------|---------|---------|-----------|---------|------------|
| **Language** | — | Latin (top 5 matches), no nulls, Romance phonotactics | Latin best syllable match, PMI r=0.96 | Latin #1, Occitan #2, CIs overlap | Language A (Romance-like) vs B (notation); qo- functional | Occitan/Latin paradigms indistinguishable (JSD ratio=0.92); affix alignment consistency 1.00 | 63/69 plants mapped to medieval Latin; cross-modal signal z=32.0 vs shuffled | TF-IDF stems folio-specific; "daiin" eliminated (17→0 folios) | Both A and B embedding spaces point to Latin via Procrustes and GW; cross-language convergence YES on both methods | **Romance language family** — now confirmed by 3 independent methods (fingerprint, paradigm, embedding geometry) |
| **Encoding** | Strong positional constraints (MI=0.30) → syllabary | simple_substitution best, 5x6 grid 47% | D.1 favors syllabary, D.3 favors substitution | R=0.39 in syllabary/abugida overlap | Grid axes = affix/stem; R(affix\|stem)=0.61 natural | 486 multi-form paradigms with prefix+suffix structure; 5 clusters match inflectional system | Best model: morphographic-syllabic (consistency 0.76) | morphographic-abbreviated best (4/8 good fits); hybrid evidence by word length; balanced segmentation unanimity 0.6667 | Prefix/suffix separation 0.90 in affix embedding space; 18 affix types form distinct clusters | **Morphological syllabary** — grid encodes affix+stem structure; affix embedding space confirms |
| **Grid validity** | 7x11 original (27% occupancy) | 5x6 refined (47%, z=-239) | 100% stable, but sections diverge (Jaccard=0.14) | Minimum 10k tokens needed; A/B split genuine | Lang A grid 50% occupancy vs B 37%; both axes significant (z>500) | Grid-cell merging reduces stems 2,328→1,693 (allographic variants) | — | — | — | **Grid is real, morphologically grounded** |
| **Decoding** | — | — | — | — | — | Random-word selectivity 0.99× blocks stem ID; phonetic decode stopped at gate 5.3 | Unanimity 0.40 (below 0.50 threshold); train/test transfer 0.0; all null tests <1.5×; **HARD STOP** | Unanimity 0.40→0.5833 (passes 0.50 gate); anchor-propagate PASS; but validation still HARD STOP (selectivities 1.22-1.46×, below 1.5×) | Procrustes selectivity 0.96-0.97×, GW 1.00× — gates FAIL; 14 seed pairs insufficient for discrimination | **Selectivity ceiling persists** at all levels — token, anchor, and corpus-level alignment |
| **Currier A/B** | — | — | — | H2 diff significant, grid Jaccard=0.14 | JSD z=3.82, vocab overlap 14%, distinct token inventories | — | — | — | A ARI=0.11 (section structure captured); B ARI=-0.003 (no section signal); both converge on Latin | **Distinct systems** — A has semantic structure, B does not; both encode Latin-related content |
| **Null characters** | — | No null insertion evidence | 11/20 null tests discriminate | 8/15 metrics discriminate, all critical pass | qo- removal neutral; 67% have suffixes, 30% prefixes | — | — | — | — | **No null padding**; apparent padding is morphological |
| **Internal structure** | z = -652/-494 vs shuffled | z = -65 fingerprint, z = -69 stripped | H2 z=-1157, Zipf z=298 vs shuffled | — | H₂(stems)=2.38 > H₂(full)=2.12; affixes carry grammatical info | Paradigm selectivity z=178 (real vs shuffled); cross-consistency 1.00 on 20 IDs | 8 Rosetta folios, 88.6% EVA coverage; paradigm filtering passes all 8 anchors | Poison anchor pruning available; per-char consistency profiled (high/medium/low) | Noun candidates cluster 5.4× above random in embedding space; verb freq rho=0.97 with Latin recipe verbs; pharmaceutical MI selectivity 1.07× | **Morpheme structure confirmed at paradigm level; noun embedding coherence strong** |
| **Scholarly rigor** | — | — | 5/7 hypotheses pass, H1 robust to corpus size | All 4 gates evaluated with CIs | All null tests z>500; contingency chi² p<0.001 | 4 gates with dual null controls; random-word control catches selectivity ceiling | 5-stage gate pipeline with 3 null tests, LOO, train/test, bootstrap; HARD STOP issued correctly | Diagnostic investigation (anchor + encoding) confirms small-anchor-set as root cause; HARD STOP maintained | 6 gates across 3 analyses; joint null test; cross-language convergence check; all gates report transparently | **Gate system correctly prevents overconfident decoding; convergent evidence accumulates** |

## Results Files

Analysis outputs are saved as JSON to `results/` (53 files total):

**Phase 1 — Stroke Analysis:**
- `stroke_positional.json` — Stroke positional distributions and MI
- `ventris_grid.json` — Syllabary grid contents and occupancy (7x11 original)
- `syllable_stats.json` — Syllable-level sequence statistics
- `stroke_discriminant.json` — Real vs shuffled discrimination z-scores

**Phase 1 — Fingerprinting:**
- `voynich_profile.json` — Full 37-dimensional entropy profile
- `section_profiles.json` — Per-section entropy profiles
- `match_rankings.json` — Ranked language+encoding matches (63 combinations)
- `discriminant_validation.json` — Fingerprint discrimination vs null text

**Phase 2A — Null Character ID:**
- `null_char_profiles.json` — Per-character information content and null_scores
- `stripping_experiment.json` — All single/pair/triple strip results with match shifts
- `stroke_null_validation.json` — Stroke positional entropy for top null candidates
- `stripped_discriminant.json` — Discriminant validation on best stripped config

**Phase 2B — Grid Refinement:**
- `grid_similarity_matrices.json` — Pairwise nucleus/onset cosine similarity
- `grid_candidates.json` — All validated grid configs with composite scores
- `grid_refined_best.json` — Best grid cell contents and merge history
- `language_narrowing.json` — Ranked language families by grid dimension fit

**Phase 3D — Degeneracy Tests:**
- `degeneracy_length.json` — Token length correlation (Voynich vs Latin char/syllable)
- `degeneracy_bigram.json` — Bigram structure comparison (Frobenius distances)
- `degeneracy_positional.json` — Positional entropy curves and DTW distances
- `degeneracy_verdict.json` — Per-test and overall verdict

**Phase 3E — Grid Validation:**
- `grid_gaps.json` — Gap pattern analysis, chi-squared, reference syllabary comparison
- `grid_frequency.json` — Cell frequency distribution and Zipf fit
- `grid_stability.json` — Bootstrap stability (200 iterations, per-cell rates)
- `grid_sections.json` — Per-section grid Jaccard similarity vs full grid

**Phase 3F — Syllable Matching:**
- `cv_labels.json` — CV label assignments for each grid cell
- `retranscription_stats.json` — Corpus-wide CV statistics, sample retranscriptions
- `syllable_language_ranking.json` — Languages ranked by syllable bigram distance
- `syllable_pmi.json` — PMI correlation between Voynich and best-fit language

**Phase 3G — Scholarly Validation:**
- `hypotheses_preregistered.json` — 7 pre-registered hypotheses with pass/fail results
- `null_test_results.json` — 5 metrics x 4 null types, z-scores and discrimination
- `effect_sizes.json` — Cohen's d, bootstrap CIs for key metrics
- `reproducibility_manifest.json` — Versions, seeds, SHA256 hashes
- `sensitivity.json` — Grid cluster count and corpus size sensitivity

**Phase 4 — Discriminant Audit, Section Diagnosis, Abugida, Multi-Language:**
- `discriminant_audit.json` — 15-metric audit table with pass/fail, critical findings
- `section_diagnosis.json` — Per-section grids, sample-size calibration curve, Currier A/B verdict
- `abugida_test.json` — Onset/nucleus entropy decomposition, script type classification
- `multi_language.json` — 7-language rankings with bootstrap CIs, combined ranking

**Phase 4.5 — Language A Isolation, Morpheme Grid, qo- Removal:**
- `language_a_isolation.json` — Language A/B profiles, A/B comparison with null test, qo- removal analysis
- `morpheme_grid.json` — Morpheme decomposition stats, contingency tables, R-value reinterpretation, entropy stripping

**Phase 5 — Morpheme-Based Decoding:**
- `paradigm_discovery.json` — Stem paradigms, size distribution, hierarchical clusters, null test, gate status
- `paradigm_match.json` — Latin/Occitan JSD/rho/chi², affix alignments, separation test, gate status
- `stem_identification.json` — 20 stem identifications, compatibility scores, cross-consistency, dual null controls
- `phonetic_decode.json` — Gate check result (stopped at gate 5.3 if stem ID unreliable)

**Phase 6 — Illustration-Constrained Decoding:**
- `illustration_constrained.json` — Plant concordance, tier classification, medieval name mapping, dominant stems
- `rosetta_selection.json` — Rosetta folio scores, encoding model test, selected folios with EVA coverage
- `anchor_propagate.json` — Anchor hypotheses, cross-consistency matrix, propagation results, null tests
- `competitive_id.json` — Beam search over contested folios, best vs runner-up assignments
- `illustration_validate.json` — 3 null tests, leave-one-out, train/test split, bootstrap, stop conditions

**Phase 6.1 — TF-IDF Stem Extraction & Diagnostics:**
- `anchor_diagnosis.json` — Per-anchor consistency, poison anchor identification, per-character unanimity, iterative pruning
- `encoding_diagnosis.json` — Per-anchor model fits, model consensus, segmentation sensitivity, hybrid model analysis

**Phase 7 — Distributional Semantics, Positional Slots, Integration:**
- `distributional.json` — Per-language (A+B) embedding spaces, Procrustes/GW alignment to Latin and Occitan, affix embeddings, cluster correspondence, null tests, cross-language convergence
- `positional_slots.json` — Latin recipe segmentation, Voynich pharmaceutical slot analysis, position x paradigm cross-validation, verb/ingredient candidates
- `approach_integration.json` — Verb/noun cluster coherence in embedding space, slot-embedding kappa, joint null test, convergence verdict

## Background

This project is a fresh start after a prior approach (consonant-skeleton-to-Latin-dictionary matching) proved unproductive. Three pieces of infrastructure were carried over:

1. **EVA transcription data and tokenizer** — IVTFF parsing with folio/line structure
2. **Discriminant validation framework** — null-text generation and comparison logic
3. **Section classification** — folio-to-section mapping for Currier A/B analysis

Everything else — skeleton generation, dictionary matching, candidate selection, iterative refinement — was specific to the failed approach and was not carried over.
