[← Approaches](phase-01-approaches.md) | [Phase Index](README.md) | [Next: Phases 4-5 →](phase-04-05.md)

# Phases 2-3: Null Characters, Grid Refinement, Degeneracy Breaking

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

---
[← Approaches](phase-01-approaches.md) | [Phase Index](README.md) | [Next: Phases 4-5 →](phase-04-05.md)
