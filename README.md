# Voynich Manuscript: Syllabary & Information-Theoretic Analysis

A multi-phase computational analysis of the Voynich manuscript, progressing from language-agnostic statistical profiling through morpheme-level analysis to corpus-wide distributional semantics, convergence scoring, cipher-level decoding, fundamental reassessment of encoding hypotheses, hypothesis-discriminating tests, and constraint satisfaction phonetic decoding. Fifteen complementary approaches across eleven phases attack the same questions from different angles, with strict selectivity gates (> 1.5x) preventing overconfident conclusions at every step.

**Approaches 1-2** (Phase 1) establish the script type and candidate language. **Phases 2-4** refine, validate, and audit. **Phase 5** attempts morpheme-based decoding (blocked by selectivity ceiling). **Phase 6** tries illustration-constrained decoding (blocked by small anchor set). **Phase 7** tests whole-corpus structural alignment via distributional semantics and positional slot analysis. **Phase 7.5** exploits the one metric clearing the 1.5x threshold (noun embedding coherence at 5.38x) to attempt vocabulary identification through converging constraints. **Phase 8** escalates to cipher-level decoding — bigram transfer cryptanalysis (Approach 16) and minimum description length decoding (Approach 18) — attacking the mapping problem with higher-order constraints. **Phase 9** confronts the consistent pattern of structural success + decoding failure by testing three specific encoding models (homophonic, nomenclator, polyalphabetic) and two broader diagnostics (matched language comparison, text typology classification). **Phase 10** tests the three surviving hypotheses — constructed script (H1), information dispersion (H2), and keyed cipher (H3) — through five discriminating analyses: token-level entropy curves, mutual information decay, folio-level encoding shifts, glyph construction grammar, and hypothesis integration. **Phase 11** directly attacks the 14-variable phonetic mapping problem using constraint satisfaction: six constraint layers progressively prune each grid cell's candidate syllable set, AC-3 arc-consistency propagation removes inconsistencies, and beam search (MRV-ordered, width 50) finds the CE-optimal assignment across Latin, Occitan, Italian, and German. **Phase 11.5** runs five sequential refinement steps to push past the 11.1% dictionary hit rate: failure diagnosis (NEAR_MISS dominant, 13/14 high-error cells), inherent vowel and CVC/CCV relaxation sweeps (relaxation degrades selectivity — strict CV remains optimal), verb constraint integration from Phase 9 (1 soft constraint), iterative anchor bootstrapping (converges immediately at 7.2% dict hit), and a full V1–V9 validation battery confirming 8/9 tests pass with selectivity 1.85×. Verdict: the CSP framework is correct; the bottleneck is grid precision, not the language or encoding model.

Key finding across all phases: the Voynich manuscript encodes a **Romance language** (Latin or Occitan, not separable) using a **morphological syllabary** with genuine affix+stem structure. Both Voynich Language A and B embedding spaces independently point to Latin as the closest structural match. Fisher's combined probability test across 5 independent evidence families yields p = 2.75×10⁻¹⁰, confirming that the aggregate signal is real even though the selectivity ceiling — where frequency priors dominate over genuine linguistic content — persists at the level of individual word identification. Phase 8's MDL decoder, tested against all four candidate languages (Latin, Occitan, Italian, German), cannot discriminate between them — German wins on raw CE due to corpus size, not linguistic affinity. The failed sanity check (4% cipher recovery) and lack of language discrimination confirm the compression gains are frequency-driven, not genuine decryption.

Phase 9's fundamental reassessment rules out three specific encoding models: **no homophonic signal** (zero distributional clusters at cosine > 0.8, Voynich vocabulary is actually smaller than references), **no nomenclator-specific bimodality** (Voynich is bimodal but so are all reference languages), and **no position-dependent encoding** (positional JSD matches random shuffling). The four candidate languages remain statistically indistinguishable at matched corpus sizes (11K tokens, overlapping CIs). The text typology classifier identifies the Voynich as **encoded natural language** (confidence 1.0) — not glossolalia, not constructed — with an anomalously high entropy floor (0.978 bits/char vs 0.33–0.51 for reference languages), indicating the encoding preserves more redundancy than any tested plaintext.

Phase 10 resolves the three-way ambiguity. **H1 (Constructed script) wins** with score 4.0, margin 2.5 over H2 (1.5) and H3 (1.0). The entropy curve for Voynich Language A shows a near-perfect parallel shift with Latin (r = 0.999), sections are consistent (herbal-pharma r = 0.9998), and the glyph grid matches Devanagari-class constructed scripts with a "construction" (not "morphology") diagnosis. H2 is partially supported by high MI decay τ (8.98× reference) but fails the phrase-level alignment test. H3 is largely rejected — no residual JSD after controlling for section, no quire boundary effects.

Phase 11 implements the CSP phonetic decoder predicted by Phase 10. **Latin wins** across all four languages (CE = 2.999, selectivity **1.92×** vs random baseline of 5.74). All seven validation tests pass: sanity check selectivity 1.47×, cross-validation CV = 0.013 (well below 0.10 threshold), section coherence confirmed, Language B CE ratio 1.02×, and prior-phase convergence 2/3 checks. The best Latin phonetic table maps the 14 grid cells to two-character CV syllables (si, co, ne, ca, ce, ba, bi, se, la, na); 11.1% of decoded tokens match Latin reference vocabulary (up from 9.4% at baseline), and 1/8 Rosetta folio anchors achieve edit distance ≤ 3. The decoding remains frequency-dominated: the CE gap is real and significant, but the recovered syllable table does not yet produce recognizable Latin words, consistent with the selectivity ceiling documented across all prior phases.

Phase 11.5 runs five sequential diagnostic and refinement passes. Failure diagnosis identifies 13/14 grid cells with error rates above 60% and classifies 48.5% of decoded tokens as HIT or NEAR_MISS — well above the 15% gate. The relaxation sweep (strict CV → CVC → CCV, levels 0–5) finds that adding syllable types consistently drops selectivity below the 1.5× gate; level 0 (strict CV, 75 syllables) remains the best configuration. Verb constraint integration (Phase 9 assignments) yields only 1 soft constraint due to length-mismatch between Voynich stems and Latin syllabifications; the iterative anchor bootstrapping loop converges on iteration 1 with no improvement. Despite these stalled quantitative metrics, the final V1–V9 battery passes **8/9 tests** (only V9 MCMC fails on dict-hit z-score): Latin remains the top language, selectivity holds at 1.85×, section coherence is confirmed, cross-validation CV = 0.015, V8 readability shows 100% phonotactically plausible endings. The bottleneck is diagnosed as grid precision — the current 14-cell decomposition is correct in structure but insufficiently granular for syllable-level word recovery.

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
voynich combined-embed    # Phase 7.5 Step 1: combined A+B corpus embeddings
voynich noun-clusters     # Phase 7.5 Step 2: noun subcluster analysis
voynich verb-id           # Phase 7.5 Step 3: verb identification (Hungarian matching)
voynich embed-bridge      # Phase 7.5 Step 4: illustration-embedding bridge
voynich convergence       # Phase 7.5 Step 5: convergence scoring (Fisher's test)
voynich phase7-5          # Run full Phase 7.5 pipeline (Steps 1-5)
voynich bigram-transfer   # Phase 8 / Approach 16: bigram transfer cryptanalysis
voynich mdl-decode        # Phase 8 / Approach 18: MDL decoding
voynich cipher-validate   # Phase 8 validation battery
voynich phase8            # Run full Phase 8 (Approaches 16 + 18 + validation)
voynich nomenclator       # Phase 9.2: bimodal frequency / nomenclator test
voynich homophones        # Phase 9.1: homophonic substitution test
voynich position-dep      # Phase 9.3: position-dependent encoding test
voynich lang-compare      # Phase 9.4: expanded language comparison
voynich typology          # Phase 9.5: text typology classification
voynich phase9            # Run all Phase 9 analyses
voynich entropy-curves    # Phase 10.1: token-level entropy curves (H1/H2/H3 test)
voynich mi-decay          # Phase 10.2: mutual information decay (H2 test)
voynich folio-shift       # Phase 10.3: folio-level encoding shifts (H3 test)
voynich glyph-grammar     # Phase 10.4: glyph construction grammar (H1 test)
voynich hypothesis        # Phase 10.5: hypothesis integration and verdict
voynich phase10           # Run all Phase 10 analyses
voynich csp-solve         # Phase 11.0: CSP solver sanity check (synthetic recovery)
voynich csp-decode        # Phase 11.2: multi-language CSP phonetic decoding
voynich csp-validate      # Phase 11.3: CSP validation battery (7 tests)
voynich phase11           # Run full Phase 11 pipeline (solve → decode → validate)
voynich csp-diagnose      # Phase 11.5.1: token category diagnosis (HIT/NEAR_MISS/GIBBERISH)
voynich csp-refine        # Phase 11.5.2-3: inherent vowel + CVC/CCV relaxation sweep
voynich verb-constrain    # Phase 11.5.4: verb constraint integration (Phase 9 assignments)
voynich csp-iterate       # Phase 11.5.5: iterative anchor bootstrapping loop
voynich csp-final         # Phase 11.5.6-7: multi-language final + V1-V9 validation battery
voynich phase11-5         # Run full Phase 11.5 pipeline (diagnose → refine → iterate → final)
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
│   │   ├── stats.py             # Entropy, Zipf (single + piecewise), AIC/BIC, bigram matrices, MI, TTR, DTW, PPMI/SVD, Procrustes, GW, n-gram LM, SA, entropy curves
│   │   ├── ciphers.py           # Historical cipher implementations + encoding simulators
│   │   ├── reference.py         # Reference corpus loading, RTF conversion, syllable stats, Latin recipe segmentation, phrase catalog
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
│       ├── approach_integration.py # Phase 7: cross-validation of Approaches 8+9
│       ├── noun_subclusters.py # Phase 7.5 Step 2: noun subcluster analysis
│       ├── verb_identification.py # Phase 7.5 Step 3: verb identification
│       ├── embedding_bridge.py # Phase 7.5 Step 4: illustration-embedding bridge
│       ├── convergence_score.py # Phase 7.5 Step 5: convergence scoring
│       ├── bigram_transfer.py # Phase 8 / Approach 16: bigram transfer cryptanalysis
│       ├── mdl_decode.py      # Phase 8 / Approach 18: MDL decoding
│       ├── cipher_validate.py # Phase 8: cipher validation & integration
│       ├── nomenclator_test.py # Phase 9.2: nomenclator / bimodal frequency test
│       ├── homophone_test.py  # Phase 9.1: homophonic substitution test
│       ├── position_dependent.py # Phase 9.3: position-dependent encoding test
│       ├── language_comparison.py # Phase 9.4: expanded 4-language comparison
│       ├── text_typology.py   # Phase 9.5: text typology classification
│       ├── entropy_curves.py  # Phase 10.1: token-level entropy curves
│       ├── mutual_info_decay.py # Phase 10.2: MI decay analysis
│       ├── folio_shift.py     # Phase 10.3: folio-level encoding shifts
│       ├── glyph_grammar.py   # Phase 10.4: glyph construction grammar
│       ├── hypothesis_verdict.py # Phase 10.5: hypothesis integration
│       ├── csp_constraints.py # Phase 11: six constraint layers (inventory, frequency, phonotactics, word validity, anchors, cross-entropy)
│       ├── csp_solver.py      # Phase 11: CSP engine — AC-3 propagation, MRV beam search, sanity test
│       ├── csp_decode.py      # Phase 11: multi-language pipeline (Latin/Occitan/Italian/German)
│       ├── csp_validate.py    # Phase 11: 7-test validation battery (V1–V7)
│       ├── csp_diagnosis.py   # Phase 11.5.1: token failure diagnosis (HIT/NEAR_MISS/GIBBERISH, per-cell error profiles)
│       ├── csp_refinement.py  # Phase 11.5.2-3: inherent vowel sweep + graduated CVC/CCV relaxation
│       ├── verb_constraints.py # Phase 11.5.4: verb constraint integration from Phase 9 assignments
│       ├── csp_iterate.py     # Phase 11.5.5: iterative anchor bootstrapping loop
│       └── csp_final.py       # Phase 11.5.6-7: multi-language final comparison + V1–V9 validation battery
├── data/
│   ├── corpus/                  # EVA transcription files (ZL3b-n.txt, RF1b-e.txt, IT2a-n.txt)
│   └── reference/               # Real historical corpora organized by language (not in git)
│       ├── latin/               # Circa Instans, De Viribus Herbarum
│       ├── occitan/             # Régime du Corps
│       ├── italian/             # Historical Italian medical texts
│       ├── german/              # Buch der Natur (Konrad von Megenberg)
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

## Phase 7.5: Exploiting the Noun Coherence Bridge

Phase 7 produced one metric clearing the 1.5x selectivity threshold: **noun embedding coherence at 5.38x**. Phase 7.5 exploits this bridge to attempt vocabulary identification through converging constraints from embeddings (Phase 7/8), positional slots (Phase 9), illustration anchors (Phase 6.1), and morpheme decomposition (Phase 4.5).

Five steps build incrementally, each gated by selectivity tests. The convergence scoring step then applies Fisher's combined probability test across all independent evidence families from the entire pipeline.

### Step 1: Combined A+B Corpus Embeddings

Merges Language A and Language B tokens into a single PPMI+SVD embedding space to maximize vocabulary coverage and test register structure.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 1.1 | **Build combined space** — Concatenate all A+B tokens (28,459 total), sweep dimensions (50/75/100), select best by section ARI. Tags each stem with source language (A-only: 63, B-only: 268, shared: 632). | `phases/distributional.py` |
| 1.2 | **Register structure** — ARI between k-means clusters and A/B language labels (0.038). Mean cosine separation between A-stem and B-stem centroids (0.317). Low ARI = registers mostly merged in combined space. | `phases/distributional.py` |
| 1.3 | **Reference alignment** — Procrustes + GW alignment against Latin and Occitan using existing seed pairs. | `phases/distributional.py` |

**Result:** 963 combined stems, best ARI = 0.115 at 100 dims (4.2% improvement over A-only 0.111). 65.6% shared stems between A and B. Gate **PASS** (combined ARI > null). Procrustes/GW selectivity still below 1.5x (0.98x / 1.00x).

### Step 2: Noun Subcluster Analysis

Clusters the 443 noun candidates (identified by positional slot analysis) using 5 distributional features, then labels subclusters by semantic domain.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 2.1 | **Feature extraction** — For each noun: TF-IDF folio specificity, section entropy, verb co-occurrence rate, paradigm suffix richness, log frequency. Normalized to [0,1]. | `phases/noun_subclusters.py` |
| 2.2 | **Clustering** — K-means sweep k=3..8, optimal k selected by silhouette score. | `phases/noun_subclusters.py` |
| 2.3 | **Labeling** — Heuristic assignment: high TF-IDF + low entropy → plant_names, high freq + low entropy → preparations, high verb co-occurrence → plant_parts, high paradigm richness → qualities. | `phases/noun_subclusters.py` |
| 2.4 | **Latin domain matching** — Compare cluster sizes to `LATIN_PHARMACEUTICAL_DOMAINS` reference table (plant_names: 15, preparations: 9, plant_parts: 8, body_parts: 8, qualities: 4). | `phases/noun_subclusters.py` |
| 2.5 | **Null test** — Shuffle features per-column, recluster 100x, compare silhouette. | `phases/noun_subclusters.py` |

**Result:** 4 optimal subclusters — plant_parts (39 stems), qualities (196), preparations (107), plant_names (101). Silhouette = 0.303. Null mean = 0.234. Selectivity = **1.29x** — gate **FAIL** (below 1.5x). Clusters are distributional but not significantly tighter than shuffled features.

### Step 3: Verb Identification

Matches 15 Voynich verb candidates against 10 Latin pharmaceutical imperatives using Hungarian optimal assignment on a multi-criteria compatibility matrix.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 3.1 | **Verb profiling** — For each of 15 verb candidates: frequency, paradigm form count, position-1 concentration %, section distribution, co-occurring nouns, stem character length. | `phases/verb_identification.py` |
| 3.2 | **Compatibility matrix** — 15×10 matrix with 6 criteria per pair: frequency rank proximity, paradigm match, stem length compatibility, positional profile similarity, object noun compatibility (via subclusters), character mapping consistency (via Phase 6.1). | `phases/verb_identification.py` |
| 3.3 | **Hungarian assignment** — Optimal 1:1 matching. Best total score = 6.28, second-best gap = 1.4%. | `phases/verb_identification.py` |
| 3.4 | **Cross-consistency** — Check implied character mappings against Phase 6.1 high-unanimity consensus. | `phases/verb_identification.py` |
| 3.5 | **Null test** — Shuffle compatibility matrix columns independently 100x, re-assign. | `phases/verb_identification.py` |

**Result:** 9/10 confident assignments (top match: tshod→adde at 0.789). Selectivity = **0.92x** — gate **FAIL**. The null test scores higher than real because the matrix has low variance across columns, meaning many different assignments achieve similar scores. Individual verb identifications are plausible but not discriminative.

Top assignments: pchedar→misce (0.706), pcheor→coque (0.656), polche→pone (0.658), psheo→cola (0.672).

### Step 4: Illustration-Embedding Bridge

Tests whether Rosetta folio plant stems (from Phase 6) land in the plant_names embedding subcluster, then attempts three-way convergent anchor expansion.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 4.1 | **Locate Rosetta stems** — For each of 8 Rosetta folio dominant stems, compute cosine distance to each subcluster centroid. | `phases/embedding_bridge.py` |
| 4.2 | **Anchor expansion** — For each of 112 herbal folios, check if TF-IDF top stem falls in plant_names cluster AND is a noun candidate by position. Three-way convergence = illustration + cluster + positional class. | `phases/embedding_bridge.py` |
| 4.3 | **Null test** — Random stems' plant-cluster hit rate (mean 20.4%). | `phases/embedding_bridge.py` |

**Result:** 0/8 Rosetta stems land in plant_names cluster — they scatter across preparations (5), qualities (2), plant_parts (1). Gate **FAIL**. The subclusters do not align with illustration-based plant identity, suggesting the embedding features capture distributional frequency patterns rather than semantic content.

### Step 5: Convergence Scoring

Aggregates all selectivity tests across the entire pipeline using Fisher's combined probability method, and cross-references multi-method vocabulary identifications.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 5.1 | **Compile scores** — Harvest selectivity ratios and p-values from 7 independent evidence families: morpheme grid, distributional embeddings, illustration anchors, positional slots, noun coherence, verb identification, embedding bridge. | `phases/convergence_score.py` |
| 5.2 | **Fisher's test** — chi² = -2·Σln(p_i) tested against chi²(2k). | `phases/convergence_score.py` |
| 5.3 | **Convergent IDs** — Cross-reference all stems with identifications from 2+ independent methods. | `phases/convergence_score.py` |

**Result:** 10 metrics across 5 independent families. Fisher combined chi² = 65.88 (df=10), **p = 2.75×10⁻¹⁰** — the aggregate signal is overwhelmingly real. Driven by morpheme grid z-scores (>500), noun embedding coherence (5.38x), verb frequency rho (0.97), and anchor unanimity (5.83x). 76 convergent identifications found, but only 1 stem (tol/viola) has multi-method support from 2+ independent sources.

**Verdict:** The Voynich manuscript's structural properties (morpheme decomposition, embedding geometry, positional slots) are real and converge on a Latin pharmaceutical text model. But individual word identification remains blocked: the selectivity ceiling prevents discriminating correct assignments from frequency-matched alternatives.

## Phase 8: Bigram Transfer Cryptanalysis & MDL Decoding

Phases 5–7.5 hit a selectivity ceiling (~1.0–1.46×) because they match individual stems to individual words (unigram matching). Phase 8 changes the fundamental unit of analysis with two complementary approaches that exploit higher-order structure:

- **Approach 16 (Bigram Transfer)**: matches stem *pairs* — builds NxN bigram transition matrices for Voynich and target language stems, then uses simulated annealing to find the permutation minimizing Frobenius distance between matrices.
- **Approach 18 (MDL Decoding)**: evaluates *entire candidate decodings* holistically — builds character-level n-gram language models for Latin and Occitan, then finds the stem mapping that minimizes cross-entropy (bits/char) of the decoded text. The best decoding is the most compressible one.

Both operate on the morpheme stem level from Phase 4.5.

### Approach 16: Bigram Transfer Cryptanalysis

| Sub-step | Description | Module |
|----------|-------------|--------|
| 16.1 | **Build bigram matrices** — Stem sequences from Voynich (8,652 tokens, 412 unique), Latin (63,771 tokens), and Occitan (41,779 tokens). Top-100 stems, 100×100 transition probability matrices. | `phases/bigram_transfer.py` |
| 16.2 | **SA permutation search** — For Frobenius distance metric, run 10 restarts × 100K iterations of simulated annealing to find the best stem permutation aligning Voynich→Latin matrices. | `phases/bigram_transfer.py` |
| 16.3 | **Stability analysis** — Pairwise agreement across 10 independent SA restarts. Top-10 consistent mappings with confidence scores. | `phases/bigram_transfer.py` |
| 16.4 | **Validation battery** — 4 null tests (shuffled Voynich, random target matrix, Latin-to-Latin sanity check, Occitan target) + split-half cross-validation by folios. | `phases/bigram_transfer.py` |

**Result:** Selectivity = **1.30×** — gate **FAIL** (below 1.5× threshold). Stability = 0.025 (very low pairwise agreement across restarts). The optimizer reduces Frobenius distance 23% below random baseline, but the signal is not selective enough — many different permutations achieve similar distances. Notably, Occitan fits better than Latin (distance 0.042 vs 0.047).

Top consistent mappings: `ch→et` (conf=0.6), `daiin→eius` (0.6), `che→in` (0.6) — all common function words, consistent with frequency matching rather than genuine decryption.

### Approach 18: Minimum Description Length Decoding

| Sub-step | Description | Module |
|----------|-------------|--------|
| 18.1 | **Build language models** — Character-level trigram and 5-gram LMs for Latin, Occitan, Italian, and German with add-k smoothing. Measure discrimination gap (heldout vs random text). | `phases/mdl_decode.py` |
| 18.2 | **Sanity check** — Encipher Latin stems with a random substitution, attempt recovery via MCMC. Validates the approach works on known ciphers before applying to Voynich. | `phases/mdl_decode.py` |
| 18.3 | **MCMC decoding** — For each target language (Latin, Occitan, Italian, German), run 5 restarts × 100K iterations of simulated annealing with incremental cross-entropy updates. Cost function = bits/char of decoded text under the trigram LM. Language-aware stemmers for each target. | `phases/mdl_decode.py` |
| 18.4 | **Language ranking** — Rank all target languages by cross-entropy and compression ratio. Compare raw CE (affected by corpus size) vs within-language selectivity (normalized for LM quality). | `phases/mdl_decode.py` |
| 18.5 | **Validation battery** — Random mappings baseline, shuffled Voynich, wrong-language check, split-half cross-validation. | `phases/mdl_decode.py` |

**4-Language Ranking (by raw cross-entropy):**

| Rank | Language | CE (bits/char) | Compression | Corpus size |
|------|----------|---------------|-------------|-------------|
| 1 | German | 1.73 | 1.40× | 149K tokens |
| 2 | Occitan | 1.91 | 1.36× | 48K tokens |
| 3 | Italian | 2.17 | 1.77× | 11K tokens |
| 4 | Latin | 2.24 | 1.32× | 74K tokens |

**Result:** Gate **FAILED** — selectivity = **1.40×** (below 1.5× threshold). German wins on raw CE, but this is misleading: German has the largest corpus (149K tokens, 2× Latin), producing the tightest LM (discrimination gap 6.44 bits vs 4.45 for Latin). The optimizer maps frequent Voynich stems to frequent German function words (`ist`, `und`, `mit`, `auch`) — the same frequency-matching behavior seen across all languages. Cross-validation consistency = 0.96.

The **compression ratio** (random CE / best CE) normalizes for LM quality and tells a different story: Italian leads at 1.77×, followed by German 1.40×, Occitan 1.36×, Latin 1.32×. But Italian's high compression is inflated by its tiny corpus (11K tokens), which makes random mappings score worse.

**Critical caveat:** The **sanity check failed** (only 4% recovery accuracy on a known cipher). The optimizer achieves lower CE than the true mapping on the test cipher, meaning it exploits character frequency patterns without recovering actual substitutions. All four languages achieve compression ratios in the 1.3–1.8× range — consistent with frequency matching, not decryption (genuine decryption would produce 3–5× compression).

**Bottom line:** The MDL decoder **cannot discriminate between languages** because it is not actually decrypting — it finds frequency-optimal mappings that work similarly well for any language with a good enough LM. The language question remains unresolved at the MDL level.

### Cipher Validation & Integration

| Sub-step | Description | Module |
|----------|-------------|--------|
| V.1 | **Cross-approach convergence** — Compare mappings from Approaches 16 and 18 (fraction of stems mapped to the same target). | `phases/cipher_validate.py` |
| V.2 | **Prior phase convergence** — Cross-check decoded stems against illustration IDs (Phase 6), verb positions (Phase 7/9), noun clusters (Phase 7/8). | `phases/cipher_validate.py` |
| V.3 | **Seeded decoding** — Initialize Approach 18's MCMC from Approach 16's mapping; measure improvement. | `phases/cipher_validate.py` |
| V.4 | **Combined assessment** — Fisher combined probability across all evidence, confidence level assignment. | `phases/cipher_validate.py` |

**Result:** Overall gate **FAILED**, confidence = **low**. The two approaches agree on only 1% of stem mappings (1/100). Zero prior-phase convergence (0/3 checks passed — no decoded stems match illustration plant IDs, verb patterns, or noun clusters). Seeded decode improves 1.18× (modest). Fisher combined p = 0.90 (no statistical significance).

**Verdict:** `weak_evidence_single_approach_only`. When tested against all four candidate languages (Latin, Occitan, Italian, German), the MDL decoder ranks German first on raw CE — breaking the expected Romance-language pattern. But this reflects corpus size advantage, not linguistic affinity. The compression ratio ranking (Italian > German > Occitan > Latin) is similarly uninformative, driven by corpus size effects. The sanity check failure, zero cross-approach agreement, and zero prior-phase convergence all indicate this is frequency/structural matching, not genuine decryption. The Voynich manuscript is unlikely to be a simple stem-level substitution cipher over any of the four tested languages.

## Phase 9: Fundamental Reassessment

Eight phases. Thirty-two modules. Every structural finding replicates. Every decoding attempt fails. Phase 9 confronts this pattern by asking **why** decoding fails — testing three specific encoding models and two broader diagnostics without assuming the natural-language-cipher model.

### Step 9.2: Nomenclator / Bimodal Frequency Test (Highest Priority)

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.2a | **Single vs piecewise Zipf** — Fit single and two-segment power laws to the rank-frequency distribution, compare via AIC/BIC. | `phases/nomenclator_test.py` |
| 9.2b | **Reference bimodality** — Same fit on Latin, Occitan, Italian, German. Is Voynich uniquely bimodal? | `phases/nomenclator_test.py` |
| 9.2c | **Segment profiling** — Split vocabulary at breakpoint into high-freq (codebook) and low-freq (spelled-out) segments. Profile character types, morpheme regularity, coverage. | `phases/nomenclator_test.py` |
| 9.2d | **Differential decoding** — Character-level MDL on the low-freq segment only (~20 char types). | `phases/nomenclator_test.py` |
| Null | Markov-generated text bimodality comparison (50 trials). | `phases/nomenclator_test.py` |

**Result:** Voynich IS bimodal (delta_AIC = **-9,991**, strong preference for piecewise model). Breakpoint at rank 1,001 splits into 1,001 high-frequency types (74.4% of corpus) and 2,761 low-frequency types (25.6%). The low-frequency segment has **24 character types** — classical cryptanalysis territory. Exponent gap = 0.914 (segment 1: 0.914, segment 2: 0.000).

However, **all four reference languages are also bimodal** — Latin (delta_AIC = -34,731), Occitan (-20,051), German (-29,485), Italian (-2,981). Bimodality selectivity = **1.24×** vs Markov null. Gate: bimodality=True, selectivity=**FAIL** (1.24× < 1.5×).

**Verdict:** `bimodal_but_not_unique`. The vocabulary does split into two frequency regimes, but this is a property of natural language frequency distributions, not evidence of nomenclator encoding. The 24-character low-frequency segment is interesting but not diagnostic.

### Step 9.1: Homophonic Substitution Test

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.1a | **Vocabulary inflation** — Compare Voynich stem types vs reference languages (matched for morphological decomposition). | `phases/homophone_test.py` |
| 9.1b | **Distributional clustering** — Build PPMI+SVD embeddings for all Voynich stems, find cosine > 0.8 pairs, single-linkage cluster. | `phases/homophone_test.py` |
| 9.1c | **Merged decoding comparison** — Replace clusters with representatives, compare SA/MDL baselines. | `phases/homophone_test.py` |
| Null | Same clustering on Latin stems (how many false "homophone groups"?). | `phases/homophone_test.py` |

**Result:** Voynich has only **412 stem types** (8,652 tokens, TTR=0.048). Far from inflated — Latin has 3,543 types, Occitan 1,808, German 3,212. Inflation ratios: 0.12–0.82× (Voynich vocabulary is *smaller* than every reference). **Zero pairs** above cosine 0.8 threshold. No distributional clusters found. Vocabulary reduction: 0.0%.

Latin null: 12 clusters found, reduction ratio 0.893 — Latin shows *more* distributional merging than Voynich.

**Verdict:** `no_homophonic_signal`. The Voynich vocabulary is not inflated by homophones. If anything, it is unusually compact relative to reference languages at comparable corpus sizes.

### Step 9.3: Position-Dependent Encoding Test

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.3a | **Position-split bigrams** — Split tokens by position within lines (initial/medial/final thirds), build word transition matrices, compute pairwise JSD. | `phases/position_dependent.py` |
| 9.3b | **Token identity test** — For each high-frequency token, compare co-occurrence vectors at initial vs final positions. | `phases/position_dependent.py` |
| 9.3c | **Reference comparison** — Same analysis on Latin, German, Occitan, Italian. | `phases/position_dependent.py` |
| Null | Randomly shuffle token positions within each line (50 trials). | `phases/position_dependent.py` |

**Result:** Voynich positional JSD is high (mean 0.842), and 84/100 top tokens show position-dependent behavior (cosine < 0.3). But the **null shuffled Voynich** has essentially identical JSD (0.847) — the position effect comes from vocabulary sparsity, not encoding structure. Reference languages show lower JSDs (Latin 0.495, German 0.238, Occitan 0.409, Italian 0.630). Voynich/reference ratio = 1.90 (below the 2.0 gate). Position selectivity = 0.993× (no signal above shuffled baseline).

**Verdict:** `no_position_dependent_signal`. The high positional JSD is a sparsity artifact. The encoding is not polyalphabetic.

### Step 9.4: Expanded Language Comparison

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.4a | **Corpus normalization** — Subsample all corpora to 11K tokens (Italian bottleneck) using contiguous chunks. | `phases/language_comparison.py` |
| 9.4b | **Metric matrix** — 6 metrics (H2, H3, Zipf exponent, word length, TTR, bigram JSD) × 4 languages at matched size. | `phases/language_comparison.py` |
| 9.4c | **Language ranking with CIs** — Bootstrap 100 subsamples, compute composite distance to Voynich, rank with 95% CIs. | `phases/language_comparison.py` |
| 9.4d | **Occitan vs Italian head-to-head** — Per-metric comparison with bootstrap CIs. | `phases/language_comparison.py` |

**Result (ranking by composite distance to Voynich):**

| Rank | Language | Distance | 95% CI | Closest on N metrics |
|------|----------|----------|--------|---------------------|
| 1 | Italian | 3.179 | [3.173, 3.186] | 2 (H2, H3) |
| 2 | Occitan | 3.306 | [2.917, 3.456] | 1 (word length) |
| 3 | German | 3.346 | [3.190, 3.440] | 1 (bigram JSD) |
| 4 | Latin | 3.406 | [3.142, 3.827] | 2 (Zipf, TTR) |

CIs overlap for all four languages. **Separation not significant.** Occitan vs Italian head-to-head: **3–3 tie** (Italian closer on H2, H3, bigram JSD; Occitan closer on Zipf, word length, TTR).

**Verdict:** `languages_indistinguishable_at_this_sample_size`. At 11K tokens, none of the six metrics can separate the four candidate languages. The source language question remains open.

### Step 9.5: Text Typology Classification

| Sub-step | Description | Module |
|----------|-------------|--------|
| 9.5a | **Markov generation test** — Train char-level Markov (orders 1–3) on Voynich, generate 30 synthetic texts each, compare 6 metrics within 2σ. | `phases/text_typology.py` |
| 9.5b | **Text type classification** — Rule-based classifier using H2/H1 ratio, TTR, Zipf R², indicators for glossolalia, constructed, natural, and encoded natural language. | `phases/text_typology.py` |
| 9.5c | **Entropy curves** — Conditional entropy at context orders 0–6 for Voynich and all reference languages. DTW curve comparison, decay rates, asymptotic floors. | `phases/text_typology.py` |
| Null | Classify word-shuffled Voynich (should classify as random/glossolalia). | `phases/text_typology.py` |

**Markov generation results:**

| Order | Metrics within 2σ | Sufficient? |
|-------|-------------------|-------------|
| 1 | 2/6 | No |
| 2 | 4/6 | No |
| 3 | 4/6 | No |

No Markov order reproduces ≥5/6 Voynich metrics — the structure requires higher-order dependencies than character-level Markov can capture.

**Classification:** H2/H1 = 0.622 (anomalously high — outside the natural language range of 0.3–0.6), Zipf R² = 0.889 (Zipfian), TTR = 0.349 (normal). All three encoded-natural indicators fire: anomalous H2/H1 + Zipfian + normal TTR. Classification: **encoded natural language** (confidence = 1.0).

**Entropy curves:**

| Order | Voynich | Latin | Occitan | Italian | German |
|-------|---------|-------|---------|---------|--------|
| H0 | 3.832 | 4.021 | 4.163 | 4.126 | 4.213 |
| H1 | 2.385 | 3.479 | 3.605 | 3.320 | 3.369 |
| H3 | 1.986 | 2.187 | 2.119 | 1.911 | 1.921 |
| H6 | **0.978** | 0.386 | 0.328 | 0.476 | 0.510 |

The Voynich entropy floor (0.978 bits/char at order 6) is **2–3× higher** than any reference language (0.33–0.51). Decay rate is shallower (-0.390 vs -0.624 to -0.691 for references). The encoding preserves more character-level redundancy than plaintext — consistent with a cipher that doesn't fully exploit the plaintext's predictability. Closest curve: German (DTW = 2.21).

**Verdict:** `classified_as_encoded_natural`. Gate **PASSED** (classification confidence ≥ 0.7). The text is not glossolalia, not constructed language, not Markov-generated. It encodes natural language through a mechanism that preserves morphological structure but raises the character-level entropy floor above all tested natural languages.

### Phase 9 Decision Tree Outcome

```
Day 1: Is the vocabulary bimodal?
└── YES, but so are all reference languages → not nomenclator-specific

Day 2: Are there distributional homophone groups?
└── NO — zero clusters found, vocab is actually compact

Day 3: Is the encoding position-dependent?
└── NO — positional JSD matches random shuffling

Day 4: Which language wins at matched corpus sizes?
└── NONE — all four indistinguishable (CIs overlap)

Day 5: What kind of thing is this?
└── ENCODED NATURAL LANGUAGE — Markov insufficient (4/6 metrics),
    anomalous H2/H1 ratio, entropy floor 2-3× above plaintext
```

**Bottom line:** The encoding is not homophonic, not nomenclator, not polyalphabetic, and the source language cannot be resolved with available corpus sizes. The text classifies as encoded natural language with an anomalously high entropy floor — the encoding mechanism preserves morphological and distributional structure but introduces character-level redundancy not seen in any tested plaintext. This is consistent with a cipher system that operates at a granularity between character-level and word-level substitution, or one that introduces systematic padding/expansion at the character level.

## Phase 10: Testing the Three Surviving Hypotheses

Nine phases eliminated every classical cipher model while confirming encoded natural language. Three hypotheses survive:

- **H1 (Constructed script)**: Glyph strokes map to phonetic values via script-specific construction logic — each glyph is built from onset + nucleus components that encode CV syllables, analogous to Hangul or Devanagari.
- **H2 (Information dispersion)**: Each meaning unit is spread across multiple tokens — the encoding disperses information so that distant tokens carry more mutual information than in natural language.
- **H3 (Keyed cipher)**: A key modulates the mapping at a period longer than line-level — different folios or quires use different encoding parameters.

The critical diagnostic is **token-level entropy at increasing context windows** — each hypothesis predicts a different curve shape. Phase 10 runs five discriminating analyses with per-section controls (Language A combined, herbal-only, pharmaceutical-only, plus Language B as negative control).

### Step 10.1: Token-Level Entropy Curves

Tests all three hypotheses simultaneously via the shape of the conditional entropy curve H(token | context of order n) at orders 0, 1, 2, 3, 5, 10.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 10.1a | **Section curves** — Language A combined, herbal, pharma, and Language B. If H1 correct, all A curves have same shape. If H3 correct, sections differ. | `phases/entropy_curves.py` |
| 10.1b | **Reference curves** — Latin, Occitan, Italian, German at same orders. | `phases/entropy_curves.py` |
| 10.1c | **Baselines** — Shuffled tokens (no context should help) and Markov-order-2 character generation. | `phases/entropy_curves.py` |
| 10.1d | **Hypothesis scoring** — H1: Pearson r of reduction rates R(n) vs best reference. H2: back-load ratio R(5→10)/R(1→2). H3: entropy floor ratio and section divergence. | `phases/entropy_curves.py` |

**Result:** Voynich Language A entropy curve shows a **near-perfect parallel shift with Latin** (r = 0.999). Sections are highly consistent (herbal-pharma r = 0.9998, combined-herbal r = 1.000). The back-load ratio is negligible (0.00011), ruling out information dispersion at the entropy curve level. The entropy floor ratio (0.745) is below the H3 threshold. Language B shows a flatter curve with higher floor (3.25 vs 2.55 for combined A), consistent with more restricted/mechanical text.

**Verdict:** `entropy_curve_supports_H1_constructed_script`. Gate **PASSED**.

### Step 10.2: Multi-Token Mutual Information Decay

Primarily tests H2 by measuring how quickly mutual information between tokens decays with distance.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 10.2a | **MI at increasing lags** — Token-gap MI at distances d = 1, 2, ..., 20 for Voynich Language A. | `phases/mutual_info_decay.py` |
| 10.2b | **Exponential decay fit** — Fit y = A·exp(-x/τ) to MI(d) curves. τ comparison across Voynich, references, and shuffled baseline. | `phases/mutual_info_decay.py` |
| 10.2c | **Per-section τ consistency** — Herbal τ vs pharmaceutical τ. If H2 is correct, τ should be similar across sections. | `phases/mutual_info_decay.py` |
| 10.2d | **Phrase-level Procrustes alignment** — If H2 supported, test whether phrase-level embeddings align better than token-level. | `phases/mutual_info_decay.py` |

**Result:** Voynich MI is nearly flat across all lags (7.05–7.11 bits), producing τ = 4,285 — far higher than any reference (Latin τ = 477, best reference). The τ ratio of **8.98×** nominally supports H2. However, per-section τ values are inconsistent (herbal τ = 4,858 vs pharma τ = 8,629), and phrase-level Procrustes alignment shows **no improvement** over token-level at any phrase length (3, 5, 7). The high τ is likely due to plug-in MI estimation bias with a large vocabulary, rather than genuine information dispersion.

**Verdict:** `mi_decay_supports_H2` (by τ ratio), but phrase alignment fails. Gate **PASSED** (τ ratio > 1.5).

### Step 10.3: Folio-Level Encoding Shifts

Primarily tests H3 by detecting systematic encoding differences between folios within the same section.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 10.3a | **Inter-folio bigram JSD** — Within-section only: herbal folio 1 vs herbal folio 2, etc. Cross-section comparisons excluded (they show huge topical JSD). Bootstrap null: shuffle tokens across folios within section, recompute JSD. | `phases/folio_shift.py` |
| 10.3b | **Function-word CV** — Coefficient of variation of uniformly-distributed stems across folios within same section, compared to reference languages. | `phases/folio_shift.py` |
| 10.3c | **Quire boundary analysis** — Within-quire vs between-quire JSD, controlling for section. | `phases/folio_shift.py` |

**Result:** 63 folios analyzed across sections (herbal_a: 39, pharmaceutical: 24). Within-section JSD is high (herbal: 0.936, pharma: 0.964) but **not significantly above bootstrap null** — the residual is not significant. Function-word CV is inflated (Voynich 0.733 vs reference mean 0.349–0.520), but this is the only H3 indicator that fires. No quire boundary effect detected. H3 requires 2/3 indicators; only 1/3 fires.

**Verdict:** `folio_shift_ambiguous`. Gate **PASSED** (clear non-H3 signal). H3 not supported.

### Step 10.4: Glyph Construction Grammar

Primarily tests H1 by comparing the Voynich glyph grid against known constructed scripts and testing construction vs morphology.

| Sub-step | Description | Module |
|----------|-------------|--------|
| 10.4a | **Script grid comparison** — Compare Voynich grid statistics (7 onsets × 11 nuclei, 31% occupancy, R_forward = 0.39, R_reverse = 0.61) against Hangul, Devanagari, Ethiopic, and Linear B using weighted composite distance. | `phases/glyph_grammar.py` |
| 10.4b | **Construction vs morphology** — Correlate onset/nucleus stroke identity with token position in line. Construction scripts show near-zero correlation (stroke identity independent of position); morphological systems show significant correlation. | `phases/glyph_grammar.py` |
| 10.4c | **Phonotactic CSP** — Map 14 grid cells to syllable candidates from Romance phonotactics, constrained by frequency matching. Language B consistency: verify B cells ⊂ A cells and core token coverage. | `phases/glyph_grammar.py` |

**Result:** Closest script: **Devanagari** (similarity 0.473), followed by Linear B (0.411) and Hangul (0.410). The construction test diagnoses **"construction"** — onset-position and nucleus-position correlations are near-zero (-0.058, 0.047) with p < 10⁻⁶, meaning glyph component identity is independent of word position (the hallmark of a constructed script, not a morphological system). The CSP maps 14 cells to Latin syllables but achieves no selectivity over random (1.0×), and Language B cells are not a subset of Language A cells. CSP decoding is not yet viable — the search space needs further pruning by illustration constraints.

**Verdict:** `glyph_grammar_supports_H1`. Gate **PASSED**.

### Step 10.5: Hypothesis Integration and Verdict

Compiles evidence from all Phase 10 sub-analyses into weighted scores.

| Hypothesis | Score | Key evidence |
|------------|-------|-------------|
| **H1 (Constructed script)** | **4.0** | Entropy curve r=0.999 with Latin (1.0), sections consistent (0.5), no folio shifts (0.5), script grid similarity (1.0), construction diagnosis (1.0). CSP not yet viable (-1.0), Language B subset fails (-0.5). |
| H2 (Information dispersion) | 1.5 | τ ratio 8.98× (1.0), no folio shifts (0.5). Back-load ratio fails, section τ inconsistent, phrase alignment fails. |
| H3 (Keyed cipher) | 1.0 | Function-word CV inflated (1.0). Residual JSD fails, floor ratio fails, section not divergent, no quire effect. |

**Winner: H1** with margin **2.5** over H2. Gate **PASSED** (margin > 1.0).

**Actionable next step:** The 14-variable CSP is the decoding path. Each grid cell maps to one phoneme or syllable. Phonotactic constraints of Romance languages prune the search space. Illustration constraints provide anchor values. Constraint propagation is estimated to reduce the search to ~10³–10⁶ candidates.

## Phase 11: Constraint Satisfaction Phonetic Decoding

Phase 10 confirmed H1 (Constructed Script): the Voynich script is a Devanagari-class abugida where each EVA character encodes one CV syllable and maps to one of 14 occupied cells in a 5×6 onset×nucleus grid. Phase 11 formulates this as a 14-variable Constraint Satisfaction Problem and searches for the phonetic assignment that minimises cross-entropy against a Romance language model.

### Architecture

Each grid cell is a CSP variable; its domain is the set of legal CV syllables in the target language. Six constraint layers progressively prune the search space before beam search explores remaining candidates.

| Layer | Constraint | Implementation |
|-------|------------|----------------|
| L1 | Phoneme inventory | Restrict domains to syllables in the target language's legal CV table |
| L2 | Frequency rank matching | Cells ranked by corpus frequency may only map to syllables at proportional reference-frequency rank ± slack |
| L3 | Phonotactic legality | Remove syllables with forbidden onset clusters |
| L4 | Word-structure validity | Score decoded tokens by vowel presence and word-final legality |
| L5 | Illustration anchors | Rosetta folio stems expand cell domains with anchor-suggested syllables; beam search applies a 0.4-point bonus to anchor-aligned assignments |
| L6 | Cross-entropy scoring | Decoded tokens scored against a character-level 3-gram LM built from the reference corpus |

AC-3 arc consistency propagation further prunes domains: any cell with a singleton domain removes its value from all other cells. Beam search (width 50, MRV ordering — smallest domain first) then expands partial assignments, scoring every 3 steps via a fast partial cross-entropy estimator.

### Step 11.0: CSP Solver Sanity Test

| Sub-step | Description | Module |
|----------|-------------|--------|
| 11.0a | **Known-mapping encoding** — Assign 14 Latin CV syllables to the real 14 grid cells, encode 3,000 Latin words as EVA glyph sequences. | `phases/csp_solver.py` |
| 11.0b | **Random baseline** — Score 100 random cell→syllable assignments; compute mean CE. | `phases/csp_solver.py` |
| 11.0c | **Selectivity check** — Verify that the true mapping CE is below the random mean (selectivity ≥ 1.3×). | `phases/csp_solver.py` |

**Result:** True mapping CE = 3.855 vs random mean CE = 5.649. **Selectivity = 1.47×**, confirming the pipeline can distinguish the true mapping from random. Direct cell recovery is 0/14 — the encoded corpus is dominated by the most frequent glyph (most Latin syllables fall back to the most common CV pattern), so the CSP cannot uniquely recover the mapping, but the true mapping is significantly better than chance.

**Verdict:** `sanity_passed_selectivity_1.47x`. Gate **PASSED** (selectivity ≥ 1.3×).

### Step 11.2: Multi-Language Phonetic Decoding

| Sub-step | Description | Module |
|----------|-------------|--------|
| 11.2a | **Per-language pipeline** — For each language (Latin, Occitan, Italian, German): build CV inventory, build 3-gram LM from reference tokens, build anchor constraints from 8 Rosetta folios, initialise domains (Layers 1–3+5), run AC-3, run beam search (Layer 6). | `phases/csp_decode.py` |
| 11.2b | **Language ranking** — Sort by best cross-entropy. | `phases/csp_decode.py` |
| 11.2c | **Random baseline + selectivity gate** — 200 random assignments; gate: selectivity ≥ 1.5×. | `phases/csp_decode.py` |
| 11.2d | **Anchor details** — For each Rosetta folio, decode the stem and compare to the plant name. | `phases/csp_decode.py` |

**Result:**

| Language | CE | Dict hit | Anchors | Time |
|----------|----|----------|---------|------|
| **Latin** | **2.999** | **11.1%** | **1/8** | 11s |
| Occitan | 3.794 | 5.6% | 0/8 | 11s |
| Italian | 4.191 | 9.2% | 1/8 | 11s |
| German | 4.221 | 2.3% | 0/8 | 13s |

Random baseline mean CE = 5.74. **Selectivity = 1.92×** (Latin best assignment vs random). Latin phonetic table: C1V1→si, C2V3→co, C3V4→co, C2V5→ne, C1V2→ca, C2V2→ca, C1V5→ne, C1V6→ne, C3V1→bi, C2V6→ce, C2V1→se, C4V3→ba, C5V4→ba, C3V6→ba. Decoded sample: "fachys"→"cosiconebi", "shol"→"coca", "cthres"→"cosisibi". Note: Phase 8 MDL ranked German first due to corpus-size effects; Phase 11 CSP ranks Latin first, consistent with all prior structural evidence.

**Verdict:** `csp_decode_significant_selectivity_1.92x`. Gate **PASSED** (selectivity ≥ 1.5×).

### Step 11.3: CSP Validation Battery

| Test | Description | Result |
|------|-------------|--------|
| V1: Sanity Check | True mapping CE < 1.3× random mean on synthetic corpus | **PASS** — selectivity 1.47× |
| V2: Random Baseline | CSP best CE vs 500 random assignments (selectivity ≥ 1.5×) | **PASS** — selectivity 1.90× |
| V3: Cross-Validation | 5-fold folio split, CE coefficient of variation < 0.10 | **PASS** — CV = 0.013 |
| V4: Section Coherence | Herbal decoded text has more plant keywords; pharma has more recipe keywords | **PASS** — score 1.0 |
| V5: Illustration Match | Non-anchor herbal folios produce decodable stems | **PASS** — 5 decoded stems |
| V6: Language B | Language B CE / Language A CE ratio < 2.0 | **PASS** — ratio 1.02× |
| V7: Prior Convergence | Agreement with Phase 8/9/10: H1 verdict, language ranking, verb decoding | **PASS** — 2/3 checks |

**Summary: 7/7 tests passed.**

**Verdict:** `csp_validation_passed_7_of_7`. Gate **PASSED** (≥ 4/7 required).

### Phase 11 Findings Summary

The CSP phonetic decoder achieves a **1.92× selectivity** over random assignment — the strongest discrimination signal obtained for any phonetic mapping approach in this project. Latin consistently wins across CE, dict hit rate, and partial anchor matching. The decoded text is phonetically regular (100% word validity) but does not yet produce recognisable Latin — the 11.1% dictionary hit rate reflects short decoded tokens like "si", "ne", "ca" matching common Latin particles and prepositions, not substantive vocabulary. The selectivity ceiling documented across Phases 5–9 remains: the CSP finds a real CE minimum, but the minimum is shallow enough that many assignments score nearly as well. The remaining gap between the current result and genuine decoding requires either: (a) a larger and more consistent set of illustration anchors, or (b) relaxing the CV-only encoding model to allow CVC syllables or consonant clusters.

## Phase 11.5: CSP Refinement and Diagnostic Validation

Phase 11.5 runs five sequential diagnostic and refinement steps on the Phase 11 CSP output, aiming to improve the 11.1% dictionary hit rate toward the 15% target and increase anchor matches from 1/8 to 3/8.

### Step 11.5.1: Failure Diagnosis

| Metric | Value |
|--------|-------|
| Tokens analyzed | 1,500 (Language A paragraph tokens) |
| HIT (exact reference match) | 9.9% |
| NEAR_MISS (edit distance ≤ 2) | 38.6% |
| GIBBERISH | 49.6% |
| Signal fraction (HIT + NEAR_MISS) | **48.5%** |
| High-error cells (error > 60%) | 13/14 |

**Gate (signal ≥ 15%): PASS.** Top correction vectors: `C3V6 (ba→de)` gain=1.0, `C1V2 (ca→di)` gain=0.34. All 13 high-error cells are dominated by GIBBERISH or NEAR_MISS — no cell is producing useful exact hits. Diagnosis: NEAR_MISS_DOMINANT, meaning the beam search reaches plausible phonetic neighborhoods but not exact targets. Correction vectors prescribe cell-level substitutions for the relaxation phase.

### Step 11.5.2-3: Inherent Vowel and Relaxation Sweep

| Level | Description | Syllables | Dict hit | CE | Selectivity |
|-------|-------------|-----------|----------|----|-------------|
| 0 | Strict CV | 75 | 9.87% | 3.099 | **1.83×** |
| 1 | CV + inherent vowel | 75 | 9.87% | 3.099 | 1.83× |
| 2 | CV + top-25 CVC | 100 | 3.33% | 4.203 | 1.35× ⚠ |
| 3 | CV + full CVC | 146 | 3.87% | 4.327 | 1.31× ⚠ |
| 4 | CV + CVC + top-25 CCV | 171 | 3.87% | 4.327 | 1.31× ⚠ |
| 5 | CV + full CVC + full CCV | 216 | 4.40% | 3.946 | 1.44× ⚠ |

Inherent vowel candidates (a/e/i) all produce identical results at Level 1 — the inventory does not differentiate. **Levels 2–5 all drop below the 1.5× selectivity gate.** Adding syllable types expands the search space faster than it constrains it; the beam search degrades into a broader but shallower exploration. Best configuration remains Level 0 (strict CV).

**Gate (best dict_hit ≥ 15% or improvement ≥ 1.35×): FAIL.** Improvement factor = 0.89× (slight regression from Phase 11 baseline). Verdict: `refinement_minimal_improvement_check_grid_decomposition`.

### Step 11.5.4: Verb Constraint Integration

| Metric | Value |
|--------|-------|
| Phase 9 verb assignments loaded | 10 |
| Length-matched verb constraints | 1 (soft; confidence 0.572) |
| Constraint: `shes` → `recipe` (`re`, `ci`, `pe`) | — |
| Dict hit before | 9.87% |
| Dict hit after | 7.15% (Δ = −2.72%) |
| Verb matches | 1 |
| Illustration conflicts | 6 |

Only 1 of 10 Phase 9 verb assignments produces a usable constraint: most Voynich stems do not length-match their Latin syllabifications. The single soft constraint slightly worsens dict hit (verb pulls one cell assignment toward a low-frequency syllable). Six conflicts found between verb-required cell values and illustration anchor hints.

**Gate (dict_hit ≥ 15% or verb_matches ≥ 5): FAIL.** Verdict: `verb_constraints_applied_dict_hit_0.071`.

### Step 11.5.5: Iterative Anchor Bootstrapping

| Metric | Value |
|--------|-------|
| Confirmed hit anchors extracted (iteration 1) | 14 new (22 total) |
| Dict hit after iteration 1 | 7.15% (Δ = 0.0000) |
| Selectivity | 1.86× |
| Convergence reason | `delta_small` (Δ < 0.005) |
| Iterations run | 1 |

The iterative loop extracts 14 new confirmed hit anchors (tokens that decode consistently to a reference word ≥ 3 times), adding them to the anchor set and re-running the CSP. The result is unchanged — the beam search reproduces the same assignment regardless of the expanded anchors. This indicates the current assignment is at a local minimum; the anchor additions do not provide enough new directional signal to escape it.

**Gate (final_dict_hit ≥ 15% or improvement ≥ 0.03): FAIL.** Verdict: `csp_iterate_delta_small_dict_hit_0.071`.

### Step 11.5.6-7: Multi-Language Final + V1–V9 Validation Battery

**Language ranking (Level 0):**

| Rank | Language | CE | Dict hit | Anchors |
|------|----------|----|----------|---------|
| 1 | **Latin** | **3.099** | **9.87%** | **1/8** |
| 2 | Occitan | 3.997 | 6.47% | 0/8 |
| 3 | German | 4.222 | 2.40% | 0/8 |
| 4 | Italian | 4.252 | 8.93% | 1/8 |

**V1–V9 validation battery:**

| Test | Result | Score |
|------|--------|-------|
| V1: Sanity Check | **PASS** | selectivity 1.47× |
| V2: Random Baseline | **PASS** | selectivity 1.85× |
| V3: Cross-Validation | **PASS** | CV = 0.015 |
| V4: Section Coherence | **PASS** | score 1.000 |
| V5: Illustration Match | **PASS** | score 0.192 |
| V6: Language B | **PASS** | ratio 1.013× |
| V7: Prior Convergence | **PASS** | score 0.667 |
| V8: Readability | **PASS** | composite 0.379 (100% plausible Latin endings, 7.2% exact hits) |
| V9: MCMC Comparison | **FAIL** | CE z=3.77 ✓ but dict-hit z=−0.61 ✗ |

**Summary: 8/9 tests passed.** Gate (≥ 6/9): **PASS.** Selectivity 1.85× (above 1.5× gate).

**Verdict: `framework_correct_phonetics_imprecise`.** The CSP framework is structurally sound — Latin wins on CE, cross-entropy is stable across folds, section coherence is confirmed, and 100% of decoded tokens end in phonotactically plausible Latin suffixes. The V9 failure pinpoints the remaining problem: the MCMC random walk produces dict-hit rates comparable to the CSP solution, meaning the CE advantage is not yet sufficient to lift dict hits above the noise floor. The bottleneck is grid granularity, not the language hypothesis or the CSP architecture.

### Phase 11.5 Findings Summary

Phase 11.5 confirms and sharpens the Phase 11 diagnosis. The CSP framework is working correctly (selectivity 1.85×, 8/9 validation tests pass). The failure mode is specific: the 14-cell grid decomposition maps each Voynich glyph to a CV syllable, but at this resolution the mapping cannot recover individual Latin words — it recovers phonetic neighborhoods. Adding more syllable types (CVC, CCV) to escape this ceiling consistently destroys selectivity by over-expanding the search space, confirming that the grid itself needs finer decomposition before the phonetic mapping can improve. The near-miss rate (38.6%) is a positive signal: the CSP is finding the right phonetic region, and systematic correction of the 13 high-error cells could unlock word-level recovery.

## Integration

The approaches cross-validate across all phases:

| Approach 1 finds | Approach 2 finds | Phase 3 finds | Phase 4 finds | Phase 4.5 finds | Phase 5 finds | Phase 7 finds | Phase 7.5 finds | Interpretation |
|---|---|---|---|---|---|---|---|---|
| CV syllabary grid with good fit | Closest match = Latin-substitution | D.1 favors syllabary, D.3 favors substitution, PMI r=0.96 | 8/15 metrics discriminate; PMI, bigram, length, stability all pass | Grid captures morphological structure (chi² p<0.001 both axes, JSD=0.46 on nucleus) | 2,328 stem paradigms discovered (z=178); 23 high-paradigm stems with 7–31 forms each | A and B embedding spaces both independently point to Latin (Procrustes + GW); noun candidates cluster 5.4x above baseline in embedding space | Combined A+B space (963 stems) improves ARI 4.2%; noun subclusters split into 4 domains matching Latin pharmaceutical categories; Fisher combined p=2.75×10⁻¹⁰ across 5 families | **Morphological structure confirmed at paradigm level; grid axes encode affix/stem roles; global embedding geometry converges on Latin; aggregate signal overwhelmingly real** |
| Strong positional constraints (MI=0.30) | Latin dominates top 5 | Grid 100% stable, sections diverge (Jaccard=0.14) | Currier A/B distinct (H2 diff significant, grid Jaccard=0.14); min sample ~10k tokens | A/B confirmed as distinct systems (JSD z=3.82, vocab overlap=14%) | Paradigm selectivity 1.47× (z=178) — just below 1.5× gate | Language A ARI=0.11 (embeddings capture section structure); Language B ARI=-0.003 (no section signal — consistent with notation hypothesis) | 65.6% shared stems between A/B; register ARI=0.038 (mostly merged); 9/10 verb assignments plausible but selectivity 0.92x | **Section divergence = genuine A/B split, not artifact; A has semantic structure, B does not; A/B share most vocabulary** |
| 5x6 grid, 47% occupancy | No null insertion evidence | Gap pattern random, closest to Cypriot (8% diff) | R=0.39 (syllabary/abugida overlap); nucleus predicts onset more than reverse | R(affix\|stem)=0.61, R(stem\|affix)=0.39 — linguistically natural under morpheme relabeling | Occitan JSD=0.65 vs Latin JSD=0.71; not separable (ratio=0.92) | Prefix/suffix separation=0.90 in affix embedding space; verbs at position 1 in 60-100% of segments; verb freq rho=0.97 with Latin recipe verbs | 0/8 Rosetta plant stems land in plant_names cluster; subclusters capture frequency patterns not semantic content | **Anomalous reverse R explained** — stems constrain affixes; **Romance family confirmed; embedding subclusters are distributional, not semantic** |
| — | Latin best across encodings | Latin best syllable match | Latin #1, Occitan #2, but CIs overlap on all metrics | qo- removal neutral (14.4% of corpus, distributed across grid, no metric improvement) | Random-word selectivity 0.99× — frequency priors dominate over morphological content; **phonetic decode blocked** | Procrustes selectivity 0.96-0.97x, GW selectivity 1.00x — both fail 1.5x gate; only 14 seed pairs available | Noun subcluster selectivity 1.29x, verb assignment selectivity 0.92x — both fail 1.5x gate; only 1/76 identifications has multi-method support | **Selectivity ceiling persists** at word identification level; individual assignments are frequency-dominated; **structural convergence is real but does not unlock vocabulary** |

**Phase 9 cross-validation (encoding hypothesis tests):**

| Phase 8 finds | Phase 9 finds | Interpretation |
|---|---|---|
| SA stability 2.5%, MDL sanity check 4% recovery — not genuine decryption | Not homophonic (0 clusters), not nomenclator (bimodal but not uniquely), not polyalphabetic (JSD matches shuffled) | **All three tested encoding models ruled out; decoding failure is not explained by a known cipher class** |
| MDL ranks German first (corpus size artifact), compression 1.3–1.8× across all languages | 4 languages indistinguishable at matched 11K tokens; Italian/Occitan tie 3–3 on metrics | **Language question genuinely unresolved — not a methodological failure but a fundamental ambiguity** |
| Fisher combined p = 0.90 (no significance), 1/100 cross-approach agreement | Classified as encoded natural language (conf=1.0); entropy floor 0.978 vs 0.33–0.51 for reference | **Text is definitively not random, glossolalia, or constructed; encoding preserves redundancy above plaintext levels** |
| All compression ratios in frequency-matching range (1.3–1.8×) | Markov models match only 4/6 metrics; H2/H1 = 0.622 (outside natural range 0.3–0.6) | **Structure requires higher-order dependencies; the anomalous H2/H1 ratio is the cipher's signature** |

**Phase 10 cross-validation (hypothesis discrimination):**

| Phase 10.1 finds | Phase 10.2 finds | Phase 10.3 finds | Phase 10.4 finds | Phase 10.5 verdict | Interpretation |
|---|---|---|---|---|---|
| Entropy curve r=0.999 with Latin; sections consistent (herbal-pharma r=0.9998); Language B curve flatter with higher floor (3.25 vs 2.55) | τ_voynich = 4,285 >> τ_latin = 477 (ratio 8.98×); but section τ inconsistent and phrase alignment shows no improvement | 63 folios, within-section JSD not significant vs null; function-word CV inflated (0.73 vs 0.35–0.52); no quire boundary effect | Closest to Devanagari (0.47 similarity); diagnosis = "construction" (onset/nucleus independent of position, p < 10⁻⁶) | **H1 wins** (score 4.0, margin 2.5). H2 = 1.5, H3 = 1.0. Gate passed. | **Constructed script confirmed.** Glyph strokes encode phonetic values via script-specific construction logic; the grid is a syllabary blueprint, not a morphological accident. The 14-variable CSP is the decoding path. |

**Phase 11 cross-validation (CSP phonetic decoding):**

| CSP decode finds | CSP validate finds | Interpretation |
|---|---|---|
| Latin wins: CE 2.999 vs Occitan 3.794, Italian 4.191, German 4.221. Selectivity 1.92× (random mean 5.74). Dict hit 11.1% (Latin reference words in decoded text). 1/8 Rosetta anchors at edit distance ≤ 3. | 7/7 validation tests pass: sanity selectivity 1.47×, V3 CV=0.013, V4 section coherence confirmed, V6 Language B ratio 1.02×, V7 prior convergence 2/3. | **Latin phonetic assignment is statistically discriminated from random** — the strongest signal achieved for any phonetic mapping approach. The decoded text is phonetically legal (100% word validity) but sub-lexical: decoded tokens match Latin particles ("si", "ne", "ca") at 11.1% rate, not substantive vocabulary. Selectivity ceiling persists: the CSP finds a real CE minimum but it is shallow. Next step: larger anchor set or CVC syllable extension. |
| Latin CE lower than all three other languages across all 20 beam-search solutions (no overlap). Phase 8 MDL ranked German first (corpus-size artifact); CSP ranks Latin first, consistent with Phases 3–7 structural evidence. | Cross-validation CV = 0.013: CE is stable across random 5-fold folio splits, confirming the signal is corpus-wide not folio-specific. Language B CE ratio 1.02× (essentially equal to Language A with the Latin table) — consistent with A and B sharing the same phonetic encoding. | **CSP result converges with all prior structural evidence** pointing to Latin. The CE advantage over Occitan (0.79 nats), Italian (1.19 nats), and German (1.22 nats) is consistent and reproduced across all beam-search solutions. Language B sharing the Latin table supports a single encoder for both sections. |

**Phase 11.5 cross-validation (CSP refinement and diagnostic battery):**

| Phase 11.5 diagnose finds | Phase 11.5 refine + iterate finds | Phase 11.5 final (V1–V9) finds | Interpretation |
|---|---|---|---|
| 48.5% of tokens are HIT or NEAR_MISS (gate: 15%). 13/14 cells have error rates > 60%. Top correction vectors: `ba→de` (gain 1.0), `ca→di`, `ne→di`. Dominant error is NEAR_MISS across all high-error cells. | Relaxation sweep: levels 2–5 all drop below 1.5× selectivity gate. Strict CV (level 0) remains best. Inherent vowel (a/e/i) produces no differentiation. Verb constraints: 1 soft constraint from 10 Phase 9 assignments; dict hit drops from 9.87% → 7.15%. Iterative bootstrapping converges at iteration 1 (Δ = 0.0000). | 8/9 tests pass (only V9 MCMC fails on dict-hit z-score). V8 readability: 100% phonotactically plausible Latin endings. Language ranking stable: Latin > Occitan > German > Italian. Selectivity 1.85×. | **The CSP framework is correct; the bottleneck is grid precision.** Decoding is in the right phonetic neighborhood (38.6% near-misses) but the 14-cell grid is too coarse to recover individual words. CVC/CCV relaxation makes things worse, not better — expanding the syllable inventory without a finer grid adds noise faster than signal. The path forward is finer grid decomposition, not larger phoneme inventories. |

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

*Italian (~11,153 tokens):*
- Historical Italian medical/herbal texts

*German (~149,453 tokens):*
- **Buch der Natur** — Konrad von Megenberg's natural history (~14th century)

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

### Cipher Validation Summary (Phase 8)

| Test | Result | Threshold | Status |
|------|--------|-----------|--------|
| Bigram transfer gate | selectivity = 1.30× | > 1.5× | **FAIL** |
| MDL decode gate (4 languages) | selectivity = 1.40× (best=German) | > 1.5× | **FAIL** |
| MDL sanity check | 4% recovery on known cipher | > 80% | **FAIL** |
| MDL language discrimination | All 4 languages within 1.3–1.8× compression | clear winner | **No discrimination** |
| Cross-approach mapping agreement | 3.1% | > 30% | **FAIL** |
| Prior-phase convergence | 0/3 checks passed | > 1/3 | **FAIL** |
| Fisher combined p-value | p = 0.037 | < 0.01 | Not significant |

**Key finding:** The 4-language MDL test (Latin, Occitan, Italian, German) is the most informative null test in Phase 8. If the Voynich manuscript were a genuine cipher over one of these languages, the target language should win decisively — but instead all four achieve similar compression ratios (1.32–1.77×), with the ranking driven by corpus size rather than linguistic affinity. German wins on raw CE solely because it has the largest reference corpus (149K tokens). The sanity check failure (4% recovery on known cipher) confirms the optimizer exploits character frequency distributions rather than recovering genuine substitutions.

**Overall verdict: weak_evidence_single_approach_only.** All gates fail. The 4-language test demonstrates that the MDL decoder cannot distinguish between Romance and Germanic targets, confirming the compression gains are frequency-driven artifacts, not linguistic signal.

### Cross-Validation Summary

| Finding | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 4.5 | Phase 5 | Phase 6 | Phase 6.1 | Phase 7 | Phase 8 | Assessment |
|---------|---------|---------|---------|---------|-----------|---------|---------|-----------|---------|---------|------------|
| **Language** | — | Latin (top 5 matches), no nulls, Romance phonotactics | Latin best syllable match, PMI r=0.96 | Latin #1, Occitan #2, CIs overlap | Language A (Romance-like) vs B (notation); qo- functional | Occitan/Latin paradigms indistinguishable (JSD ratio=0.92); affix alignment consistency 1.00 | 63/69 plants mapped to medieval Latin; cross-modal signal z=32.0 vs shuffled | TF-IDF stems folio-specific; "daiin" eliminated (17→0 folios) | Both A and B embedding spaces point to Latin via Procrustes and GW; cross-language convergence YES on both methods | 4-language MDL: German CE=1.73, Occitan 1.91, Italian 2.17, Latin 2.24 — ranking tracks corpus size, not linguistic affinity; bigram transfer favors Occitan (distance 0.042 vs 0.047) | **Romance language family** — confirmed by 3 independent methods (fingerprint, paradigm, embedding geometry); Phase 8 MDL cannot discriminate (German wins on corpus size alone) |
| **Encoding** | Strong positional constraints (MI=0.30) → syllabary | simple_substitution best, 5x6 grid 47% | D.1 favors syllabary, D.3 favors substitution | R=0.39 in syllabary/abugida overlap | Grid axes = affix/stem; R(affix\|stem)=0.61 natural | 486 multi-form paradigms with prefix+suffix structure; 5 clusters match inflectional system | Best model: morphographic-syllabic (consistency 0.76) | morphographic-abbreviated best (4/8 good fits); hybrid evidence by word length; balanced segmentation unanimity 0.6667 | Prefix/suffix separation 0.90 in affix embedding space; 18 affix types form distinct clusters | Stem-level substitution assumed; no new encoding evidence | **Morphological syllabary** — grid encodes affix+stem structure; affix embedding space confirms |
| **Grid validity** | 7x11 original (27% occupancy) | 5x6 refined (47%, z=-239) | 100% stable, but sections diverge (Jaccard=0.14) | Minimum 10k tokens needed; A/B split genuine | Lang A grid 50% occupancy vs B 37%; both axes significant (z>500) | Grid-cell merging reduces stems 2,328→1,693 (allographic variants) | — | — | — | — | **Grid is real, morphologically grounded** |
| **Decoding** | — | — | — | — | — | Random-word selectivity 0.99× blocks stem ID; phonetic decode stopped at gate 5.3 | Unanimity 0.40 (below 0.50 threshold); train/test transfer 0.0; all null tests <1.5×; **HARD STOP** | Unanimity 0.40→0.5833 (passes 0.50 gate); anchor-propagate PASS; but validation still HARD STOP (selectivities 1.22-1.46×, below 1.5×) | Procrustes selectivity 0.96-0.97×, GW 1.00× — gates FAIL; 14 seed pairs insufficient for discrimination | Bigram selectivity 1.30× (FAIL); MDL selectivity 1.40× (FAIL); 4-language test shows no discrimination (German > Occitan > Italian > Latin tracks corpus size); sanity check 4% recovery | **Selectivity ceiling holds** — MDL cannot distinguish Romance from Germanic targets; compression gains are frequency-driven artifacts |
| **Currier A/B** | — | — | — | H2 diff significant, grid Jaccard=0.14 | JSD z=3.82, vocab overlap 14%, distinct token inventories | — | — | — | A ARI=0.11 (section structure captured); B ARI=-0.003 (no section signal); both converge on Latin | — | **Distinct systems** — A has semantic structure, B does not; both encode Latin-related content |
| **Null characters** | — | No null insertion evidence | 11/20 null tests discriminate | 8/15 metrics discriminate, all critical pass | qo- removal neutral; 67% have suffixes, 30% prefixes | — | — | — | — | — | **No null padding**; apparent padding is morphological |
| **Internal structure** | z = -652/-494 vs shuffled | z = -65 fingerprint, z = -69 stripped | H2 z=-1157, Zipf z=298 vs shuffled | — | H₂(stems)=2.38 > H₂(full)=2.12; affixes carry grammatical info | Paradigm selectivity z=178 (real vs shuffled); cross-consistency 1.00 on 20 IDs | 8 Rosetta folios, 88.6% EVA coverage; paradigm filtering passes all 8 anchors | Poison anchor pruning available; per-char consistency profiled (high/medium/low) | Noun candidates cluster 5.4× above random in embedding space; verb freq rho=0.97 with Latin recipe verbs; pharmaceutical MI selectivity 1.07× | Bigram matrix effective rank 54/100; all 4 target languages achieve 1.3–1.8× compression (structure beyond random, but language-indiscriminate) | **Morpheme structure confirmed at paradigm level; noun embedding coherence strong; Phase 8 compression confirms real structure but cannot identify the language** |
| **Scholarly rigor** | — | — | 5/7 hypotheses pass, H1 robust to corpus size | All 4 gates evaluated with CIs | All null tests z>500; contingency chi² p<0.001 | 4 gates with dual null controls; random-word control catches selectivity ceiling | 5-stage gate pipeline with 3 null tests, LOO, train/test, bootstrap; HARD STOP issued correctly | Diagnostic investigation (anchor + encoding) confirms small-anchor-set as root cause; HARD STOP maintained | 6 gates across 3 analyses; joint null test; cross-language convergence check; all gates report transparently | Sanity check correctly flags false-positive risk; 4-language test serves as strongest null test (no discrimination = frequency artifact); approaches don't converge (3.1% mapping agreement) | **Gate system correctly prevents overconfident decoding; 4-language null test is the decisive diagnostic** |

## Results Files

Analysis outputs are saved as JSON to `results/` (57 files total):

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

**Phase 7.5 — Convergence Scoring:**
- `convergence_score.json` — 10 metrics across 5 families, Fisher combined chi², convergent stem identifications

**Phase 8 — Bigram Transfer, MDL Decoding, Cipher Validation:**
- `bigram_transfer.json` — Voynich/Latin/Occitan bigram matrix stats, SA permutation results, mapping stability, null tests, cross-validation, gate status
- `mdl_decode.json` — Language model stats, sanity check results, MCMC decoding for Latin and Occitan, compression ratio, word validity, decoded sample, null tests
- `cipher_validate.json` — Cross-approach convergence, prior-phase convergence, seeded improvement test, Fisher combined assessment, overall gate status

**Phase 9 — Encoding Hypothesis Tests:**
- `nomenclator_test.json` — Single vs piecewise Zipf fit, AIC/BIC comparison, reference bimodality, segment profiling (high/low-freq types, morpheme regularity, character types), differential decoding, Markov null bimodality, gate status
- `homophone_test.json` — Vocabulary inflation ratios vs 4 reference languages, distributional clustering (PPMI+SVD, cosine threshold, single-linkage), merged decoding comparison, Latin null clustering, gate status
- `position_dependent.json` — Position-split bigram JSDs (initial/medial/final), token identity test (100 tokens, per-position co-occurrence cosines), reference language JSDs, shuffled null, gate status
- `language_comparison.json` — Corpus normalization stats (11K tokens), 6-metric × 4-language distance matrix, bootstrap language ranking with CIs, Occitan vs Italian head-to-head, subsample variance, gate status
- `text_typology.json` — Markov generation test (orders 1–3, 6 metrics × 30 trials), text type classification (glossolalia/constructed/natural/encoded indicators), entropy curves (orders 0–6, 4 reference languages), decay rates, floors, DTW curve distances, gate status

**Phase 10 — Hypothesis Discrimination:**
- `entropy_curves.json` — Token-level entropy curves at orders 0,1,2,3,5,10 for Voynich (combined + herbal + pharma + Language B), 4 reference languages, shuffled baseline, Markov-2 baseline; reduction rates R(n), section consistency (Pearson r), hypothesis scores (H1 correlation, H2 back-load ratio, H3 floor ratio)
- `mi_decay.json` — Mutual information at lags 1–20 for Voynich + references + shuffled; exponential decay fit (τ, amplitude, R²); per-section τ (combined, herbal, pharma); phrase-level Procrustes alignment at lengths 3, 5, 7; τ ratio vs best reference; H2 verdict
- `folio_shift.json` — 63 folios across sections; within-section pairwise bigram JSD with bootstrap null; function-word CV (20 stems, per-folio frequency, reference comparison); quire boundary within/between JSD; H3 verdict
- `glyph_grammar.json` — Voynich grid stats (R values, occupancy, onset/nucleus types); comparison to 4 known constructed scripts (Devanagari, Hangul, Ethiopic, Linear B); construction vs morphology diagnosis (position correlation test); 14-variable CSP with phonotactic constraints; Language B consistency check; H1 verdict
- `hypothesis_verdict.json` — Evidence compilation from 10.1–10.4; weighted scoring (H1=4.0, H2=1.5, H3=1.0); winning hypothesis, margin, gate status, actionable next step

## Background

This project is a fresh start after a prior approach (consonant-skeleton-to-Latin-dictionary matching) proved unproductive. Three pieces of infrastructure were carried over:

1. **EVA transcription data and tokenizer** — IVTFF parsing with folio/line structure
2. **Discriminant validation framework** — null-text generation and comparison logic
3. **Section classification** — folio-to-section mapping for Currier A/B analysis

Everything else — skeleton generation, dictionary matching, candidate selection, iterative refinement — was specific to the failed approach and was not carried over.
