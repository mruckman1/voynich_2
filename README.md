# Voynich Manuscript: Syllabary & Information-Theoretic Analysis

Two complementary language-agnostic approaches to the Voynich manuscript that don't assume a target language. **Approach 1** (Stroke-Level Syllabary) analyzes the script's internal structure. **Approach 2** (Information-Theoretic Fingerprinting) compares the text's statistical profile against candidate language+encoding combinations.

Together, they answer two questions: **what kind of script is this?** and **what kind of language does it encode?**

Both approaches are grounded in one confirmed empirical finding: EVA compositionality (47.9% output change when decomposing multi-stroke characters) confirms that sub-character structure carries information.

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
│   │   ├── stats.py             # Entropy, Zipf, bigram matrices, MI, TTR, DTW
│   │   ├── ciphers.py           # Historical cipher implementations + encoding simulators
│   │   └── reference.py         # Reference corpus loading, RTF conversion, syllable stats
│   ├── analysis/                # Main analysis approaches
│   │   ├── strokes.py           # Approach 1: stroke decomposition, Ventris grid
│   │   └── fingerprint.py       # Approach 2: entropy profiling, profile matching
│   └── phases/                  # Phase 2 and Phase 3 workstreams
│       ├── nulls.py             # Phase 2A: null character identification
│       ├── grid_refine.py       # Phase 2B: syllabary grid refinement
│       ├── degeneracy.py        # Phase 3D: substitution vs syllabary tests
│       ├── grid_validate.py     # Phase 3E: grid gap analysis, stability
│       ├── syllable_match.py    # Phase 3F: CV labeling, language matching
│       └── scholarly.py         # Phase 3G: pre-registration, null testing
├── data/
│   ├── corpus/                  # EVA transcription files (ZL3b-n.txt, RF1b-e.txt, IT2a-n.txt)
│   └── reference/               # Real historical corpora organized by language (not in git)
│       └── latin/               # Circa Instans, De Viribus Herbarum
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

## Integration

The approaches cross-validate across all phases:

| Approach 1 finds | Approach 2 finds | Phase 3 finds | Interpretation |
|---|---|---|---|
| CV syllabary grid with good fit | Closest match = Latin-substitution | D.1 favors syllabary, D.3 favors substitution, PMI r=0.96 | **The degeneracy is genuine** — both models fit the data |
| Strong positional constraints (MI=0.30) | Latin dominates top 5 | Grid 100% stable, sections diverge (Jaccard=0.14) | **Latin substrate likely**, but grid varies by section |
| 5x6 grid, 47% occupancy | No null insertion evidence | Gap pattern random, closest to Cypriot (8% diff) | **Grid structure is real but sparse** |

## Data

### Voynich Corpus

The project uses EVA (Extended Voynich Alphabet) transcription files in IVTFF format. The parser (`core/corpus.py`) supports three transcription sources with automatic preference ordering: ZL3b-n.txt > RF1b-e.txt > IT2a-n.txt.

The corpus provides filtered access by:
- **Section**: herbal_a, herbal_b, astronomical, biological, cosmological, pharmaceutical, recipes
- **Currier language**: A (herbal A scribe) or B (remaining sections)
- **Scribe hand**: 1–5 (inferred from quire assignments)

### Reference Corpora

Real historical texts for fingerprint comparison live in `data/reference/<language>/`. These are not tracked in git — acquire and place them locally. The loader (`core/reference.py`) auto-discovers `.txt` files by language directory and handles RTF-to-text conversion automatically.

**Currently available (Latin):**
- **Circa Instans** — Salernitan herbal/pharmaceutical text (~12th century, ~25,850 tokens)
- **De Viribus Herbarum** (Macer Floridus) — Herbal poem, medical botany (~47,678 tokens)

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

### Cross-Validation Summary

| Finding | Phase 1 | Phase 2 | Phase 3 | Assessment |
|---------|---------|---------|---------|------------|
| **Language** | — | Latin (top 5 matches), no nulls, Romance phonotactics | Latin best syllable match, PMI r=0.96 | **Latin or close relative** |
| **Encoding** | Strong positional constraints (MI=0.30) → syllabary | simple_substitution best, 5x6 grid 47% | D.1 favors syllabary, D.3 favors substitution | **Genuinely degenerate** — both models partially fit |
| **Grid validity** | 7x11 original (27% occupancy) | 5x6 refined (47%, z=-239) | 100% stable, but sections diverge (Jaccard=0.14) | **Grid is real but section-dependent** |
| **Null characters** | — | No null insertion evidence | 11/20 null tests discriminate | **No evidence of null padding** |
| **Internal structure** | z = -652/-494 vs shuffled | z = -65 fingerprint, z = -69 stripped | H2 z=-1157, Zipf z=298 vs shuffled | **Highly structured, not random** |
| **Scholarly rigor** | — | — | 5/7 hypotheses pass, H1 robust to corpus size | **Most findings replicate** |

## Results Files

Analysis outputs are saved as JSON to `results/` (33 files total):

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

## Background

This project is a fresh start after a prior approach (consonant-skeleton-to-Latin-dictionary matching) proved unproductive. Three pieces of infrastructure were carried over:

1. **EVA transcription data and tokenizer** — IVTFF parsing with folio/line structure
2. **Discriminant validation framework** — null-text generation and comparison logic
3. **Section classification** — folio-to-section mapping for Currier A/B analysis

Everything else — skeleton generation, dictionary matching, candidate selection, iterative refinement — was specific to the failed approach and was not carried over.
