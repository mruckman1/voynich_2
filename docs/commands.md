# Complete CLI Command Reference

All commands are invoked as `voynich <command>` (or `python -m voynich <command>`).

Commands named `phase<N>` run all steps for that phase sequentially. Individual step commands allow targeted execution.

## Setup

```bash
uv sync
uv pip install -e .
```

## Core Commands

```bash
voynich corpus            # Load and summarize the EVA corpus
voynich reference         # Show reference corpus summary
```

## Approaches 1-2 (Phase 1)

```bash
voynich strokes           # Approach 1: stroke-level syllabary analysis
voynich fingerprint       # Approach 2: information-theoretic fingerprinting
voynich both              # Run both approaches
```

## Phase 2: Null Characters & Grid Refinement

```bash
voynich nulls             # Phase 2A: null character identification
voynich grid              # Phase 2B: syllabary grid refinement
voynich phase2            # Run both Phase 2 analyses
```

## Phase 3: Breaking the Degeneracy

```bash
voynich degeneracy        # Phase 3D: break substitution vs syllabary degeneracy
voynich grid-validate     # Phase 3E: validate syllabary grid
voynich syllable-match    # Phase 3F: syllable-level language matching
voynich validate-all      # Phase 3G: scholarly validation framework
voynich phase3            # Run all Phase 3 workstreams
```

## Phase 4: Audit & Multi-Language Comparison

```bash
voynich audit             # Phase 4.1: discriminant audit of Phase 3 results
voynich section-diagnosis # Phase 4.2: section consistency diagnosis
voynich abugida           # Phase 4.3: abugida hypothesis test
voynich multi-language    # Phase 4.4: multi-language comparison
voynich phase4            # Run all Phase 4 analyses
voynich lang-a            # Phase 4.5A+C: language A isolation + qo-removal
voynich morpheme-grid     # Phase 4.5B: morpheme grid reinterpretation
voynich phase4-5          # Run all Phase 4.5 analyses
```

## Phase 5: Morpheme-Based Decoding

```bash
voynich paradigms         # Phase 5.1: paradigm discovery
voynich paradigm-match    # Phase 5.2: paradigm-to-language matching
voynich stem-id           # Phase 5.3: frequency-based stem identification
voynich phonetic          # Phase 5.4+5.5: phonetic decode and validation
voynich phase5            # Run all Phase 5 analyses
```

## Phase 6: Illustration-Constrained Decoding

```bash
voynich illustration      # Phase 6.0: illustration-constrained setup
voynich rosetta           # Phase 6 D+E: Rosetta folio selection
voynich anchor            # Phase 6 A+B: anchor-and-propagate
voynich compete           # Phase 6 C: competitive ID resolution
voynich phase6-validate   # Phase 6 validation battery
voynich phase6            # Run all Phase 6 analyses
voynich anchor-diagnosis  # Phase 6.1B: anchor inconsistency diagnosis
voynich encoding-diagnosis # Phase 6.1C: encoding model diagnosis
voynich phase6-1          # Run full Phase 6.1 pipeline (TF-IDF + diagnosis)
```

## Phase 7: Distributional Semantics & Positional Slots

```bash
voynich embeddings        # Approach 8: morpheme distributional semantics
voynich slots             # Approach 9: pharmaceutical positional slot analysis
voynich phase7            # Run full Phase 7 (Approaches 8 + 9 + integration)
voynich combined-embed    # Phase 7.5 Step 1: combined A+B corpus embeddings
voynich noun-clusters     # Phase 7.5 Step 2: noun subcluster analysis
voynich verb-id           # Phase 7.5 Step 3: verb identification (Hungarian matching)
voynich embed-bridge      # Phase 7.5 Step 4: illustration-embedding bridge
voynich convergence       # Phase 7.5 Step 5: convergence scoring (Fisher's test)
voynich phase7-5          # Run full Phase 7.5 pipeline (Steps 1-5)
```

## Phase 8: Cipher-Level Decoding

```bash
voynich bigram-transfer   # Phase 8 / Approach 16: bigram transfer cryptanalysis
voynich mdl-decode        # Phase 8 / Approach 18: MDL decoding
voynich cipher-validate   # Phase 8 validation battery
voynich phase8            # Run full Phase 8 (Approaches 16 + 18 + validation)
```

## Phase 9: Fundamental Reassessment

```bash
voynich nomenclator       # Phase 9.2: bimodal frequency / nomenclator test
voynich homophones        # Phase 9.1: homophonic substitution test
voynich position-dep      # Phase 9.3: position-dependent encoding test
voynich lang-compare      # Phase 9.4: expanded language comparison
voynich typology          # Phase 9.5: text typology classification
voynich phase9            # Run all Phase 9 analyses
```

## Phase 10: Hypothesis Testing

```bash
voynich entropy-curves    # Phase 10.1: token-level entropy curves (H1/H2/H3 test)
voynich mi-decay          # Phase 10.2: mutual information decay (H2 test)
voynich folio-shift       # Phase 10.3: folio-level encoding shifts (H3 test)
voynich glyph-grammar     # Phase 10.4: glyph construction grammar (H1 test)
voynich hypothesis        # Phase 10.5: hypothesis integration and verdict
voynich phase10           # Run all Phase 10 analyses
```

## Phase 11: CSP Phonetic Decoder

```bash
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

## Phase 12: Grid Recalibration

```bash
voynich grid-recal        # Phase 12.1-12.2: correction vector bias detection + character move proposal
voynich grid-alt          # Phase 12.4: stroke-alignment audit of all 44 EVA glyphs
voynich token-decomp      # Phase 12.5: digraph/ligature decomposition variant sweep (6 variants)
voynich recal-csp         # Phase 12.3+12.6: iterative CSP re-solve on recalibrated grid + V1-V10 validation
voynich phase12           # Run full Phase 12 pipeline
```

## Phase 13: Context-Dependent Rules

```bash
voynich error-patterns    # Phase 13.1: near-miss error catalog, NW alignment, MI gate (selectivity 20.11×)
voynich null-context      # Phase 13.6: cell conflation + dictionary expansion null tests
voynich extract-rules     # Phase 13.2: context-dependent rule formalization + power ranking
voynich context-csp       # Phase 13.3: context-aware CSP (Version A rule-constrained + Version B free search)
voynich rule-validate     # Phase 13.4: cross-validation + per-rule selectivity + linguistic plausibility
voynich context-decode    # Phase 13.5: full corpus decoding + V1-V11 validation battery
voynich phase13           # Run full Phase 13 pipeline
```

## Phase 14: Stroke-Feature Model (Breakthrough)

```bash
voynich cell-analysis     # Phase 14.1: within-cell distributional clustering (confirms 21 distinct phonemes from 14 cells)
voynich stroke-features   # Phase 14.2: enumerate 25 stroke triples + PHONEME_PLACE_MAP/PHONEME_NUCLEUS_MAP hypotheses
voynich feature-csp       # Phase 14.3: 25-variable feature CSP (19.4% dict_hit, 3.00x selectivity for Latin)
voynich feature-calibrate # Phase 14.4: synthetic abugida calibration (66.3% clean dict_hit, ~33% expected Voynich ceiling)
voynich feature-decode    # Phase 14.5-14.6: full multi-language decode + V1-V12 battery (7/12 pass, 18 confirmed Latin hits)
voynich subcell-split     # Phase 14.7: data-driven subcell fallback (8.3% dict_hit — feature model wins 19.4% vs 8.3%)
voynich phase14           # Run full Phase 14 pipeline
```

## Phase 15: Feature Model Refinement

```bash
voynich dict-expand       # Phase 15.1: medieval Latin dictionary expansion + near-miss catalog + selectivity validation
voynich artic-csp         # Phase 15.2: articulatory consistency scoring (delta grid search + hard constraints + per-onset descent)
voynich iter-hits         # Phase 15.3: iterative re-solving with confirmed dictionary hits as hard CSP constraints
voynich combined-refine   # Phase 15.4: 2^3 ablation study (dict × AC × hits) + combined optimization
voynich text-analysis     # Phase 15.5: decoded text analysis (phrase detection, section readability, vocabulary catalog)
voynich phase15-validate  # Phase 15.6: full V1-V14 validation battery + progression tracking
voynich phase15           # Run full Phase 15 pipeline
```

## Phase 16: Modifier Detection

```bash
voynich mod-standalone    # Phase 16.1: standalone distributional analysis (never-solo, positional/adjacency entropy)
voynich mod-anomaly       # Phase 16.2: frequency anomaly detection (Zipf residuals, obligatory co-occurrence, length correlation)
voynich mod-distrib       # Phase 16.3: syllable distribution matching (KS test of modifier subsets vs Latin syllable counts)
voynich mod-pairs         # Phase 16.4: minimal pair subtraction (token pairs differing by 1 char, dict-hit preservation)
voynich mod-localize      # Phase 16.5: dictionary hit localization (padding ratio per EVA char)
voynich mod-integrate     # Phase 16.6: convergent classification (≥3/5 agreement) + 3 re-decode strategies (strip/alter/combined)
voynich phase16           # Run full Phase 16 pipeline
```

## Phase 17: Honesty Diagnostics

```bash
voynich honesty-dict      # Phase 17.0.1: dictionary tier control test (original/expanded/core dict scoring)
voynich honesty-keywords  # Phase 17.0.2: top-100 Latin medical keyword presence test
voynich honesty-verbs     # Phase 17.0.3: positional verb decode test (15 stems vs Latin imperatives)
voynich null-corpus       # Phase 17.0.4: null corpus end-to-end control (5 synthetic bigram corpora)
voynich honesty-words     # Phase 17.0.5: minimum viable words test (rosetta plants, verbs, high-freq tokens)
voynich step0-integrate   # Phase 17.0.6: compile all 5 tests into GO/NO-GO verdict
voynich step0             # Run full Phase 17 Step 0 pipeline (all 6 honesty diagnostics)
```

## Phase 18: Hypothesis Discrimination Battery

```bash
voynich burstiness        # Phase 18.1: spatial autocorrelation / burstiness test
voynich stride-entropy    # Phase 18.2: stride-entropy decimation analysis
voynich trie-topology     # Phase 18.3: prefix trie topology & Colless imbalance
voynich hmm-pos           # Phase 18.4: unsupervised HMM POS induction
voynich lz-complexity     # Phase 18.5: Lempel-Ziv complexity growth curve
voynich hypothesis-disc   # Phase 18.6: weighted H1/H2/H3 aggregation and verdict
voynich phase18           # Run full Phase 18 pipeline (all 6 tests)
```

## Phase 19: Tachygraphic Identification

```bash
voynich modifier-validate # Phase 19.4: validate Phase 16 modifier classification (6 distributional predictions)
voynich affix-isolate     # Phase 19.3: affix-to-Latin ending mapping via Hungarian algorithm
voynich lang-b-attack     # Phase 19.1: Language B combinatorial label set attack (6 candidate sets)
voynich entropy-shift     # Phase 19.2: entropy shift cipher identification (9 mechanisms ranked by cosine similarity)
voynich tachy-stroke      # Phase 19.5: tachygraphic stroke-modification analysis (sign family phonetic regularity)
voynich cross-validate    # Phase 19.8: cross-approach convergence (29 mappings, skeleton agreement)
voynich illus-target      # Phase 19.7: illustration-targeted decoding (50 folios vs botanical IDs)
voynich stroke-sim        # Phase 19.6: tachygraphic simulation parameter sweep (24 variants vs Voynich fingerprint)
voynich phase19-integrate # Phase 19.9: aggregate all 8 tests into convergence/readiness verdict
voynich phase19           # Run full Phase 19 pipeline (all 9 tests)
```

## Phase 20: Tachygraphic Table Construction

```bash
voynich tachy-anchors     # Phase 20.1: extract per-EVA-char syllable anchors from cross-approach mappings + Phase 15 triples
voynich tachy-families    # Phase 20.2: map sign families to consonant classes, assign vowel variants within each family
voynich tachy-grid        # Phase 20.3: constrained grid solve — 29-variable CSP at EVA-character granularity
voynich tachy-decode      # Phase 20.4: full corpus decode (36K tokens) with tachygraphic table + R3 modifier strategy
voynich tachy-read        # Phase 20.5: readability assessment (bigram plausibility, POS validity, domain coherence, phrases)
voynich tachy-phrases     # Phase 20.6: Latin phrase detection + botanical cross-check (28 folios with plant IDs)
voynich tachy-validate    # Phase 20.7: 12-test validation battery (V1-V12) integrating all Phase 20 evidence
voynich phase20-integrate # Phase 20.8: compile verdict, tachygraphic table, progression tracking
voynich phase20           # Run full Phase 20 pipeline (all 8 steps)
```

## Phase 21: Paleographic Sign Comparison

```bash
voynich paleo-ingest      # Phase 21.1: normalize 5 historical sources → unified sign database
voynich fontana-families  # Phase 21.2: Fontana cipher families + gallows rotation test
voynich chatelain-families # Phase 21.3: Chatelain Bobbio families → reference syllable table
voynich eva-compare       # Phase 21.4: 44 EVA chars vs all historical signs (two-tier similarity)
voynich family-syllable   # Phase 21.5: map Voynich families → historical syllable families
voynich cappelli-mod      # Phase 21.6: modifier identification via Cappelli abbreviation marks
voynich paleo-table       # Phase 21.7: assemble paleographic decoding table
voynich paleo-decode      # Phase 21.8: decode full corpus with paleographic table
voynich paleo-validate    # Phase 21.9: 15-test validation battery (12 original + 3 paleographic)
voynich phase21-integrate # Phase 21.10: final verdict, progression, gap analysis
voynich phase21           # Run full Phase 21 pipeline (all 10 steps)
```

## Phase 22: First-Syllable Extraction

```bash
voynich first-syl         # Phase 22.1: extract first CV/CVC syllable from historical word matches
voynich fontana-phon      # Phase 22.2: map Fontana cipher key onto EVA chars via structural correspondences
voynich table-merge       # Phase 22.3: merge first-syllable + Fontana + anchors + Phase 15 fallbacks
voynich decode-22         # Phase 22.4: full corpus decode (36K tokens) with merged table + Viterbi segmentation
voynich read-22           # Phase 22.5: readability assessment (bigram plausibility, POS, domain coherence)
voynich phrases-22        # Phase 22.6: phrase detection + botanical cross-check (28 folios)
voynich validate-22       # Phase 22.7: 15-test validation battery (V1-V15)
voynich phase22-integrate # Phase 22.8: final verdict, mode comparison, progression, gap analysis
voynich phase22           # Run full Phase 22 pipeline (all 8 steps)
```

## Phase 23: Statistical Inversion Analysis

```bash
voynich ceiling           # Phase 23.1: oracle ceiling and efficiency analysis
voynich hist-invert       # Phase 23.2: historical inversion pattern search (5,199 signs)
voynich bench-split       # Phase 23.3: bench char subgroup remapping
voynich perm-search       # Phase 23.4: permutation search (222 candidates)
voynich read-delta        # Phase 23.5: readability delta comparison
voynich phase23           # Run full Phase 23 pipeline
```

## Phase 24: Error Correction + Exploratory Analysis

```bash
voynich triple-loo        # Phase 24.1: leave-one-out triple sensitivity
voynich error-id          # Phase 24.2: error candidate identification
voynich triple-swap       # Phase 24.3: greedy swap accumulation
voynich bigram-val        # Phase 24.4: bigram filter validation
voynich corrected-tab     # Phase 24.5: corrected table assembly
voynich corrected-decode  # Phase 24.6: corrected table corpus decode
voynich corrected-read    # Phase 24.7: corrected table readability battery
voynich word-bound        # Phase 24.8: word boundary analysis
voynich ligature-test     # Phase 24.9: ligature MI analysis
voynich direction         # Phase 24.10: directionality test
voynich crib-search       # Phase 24.11: known text search (medical formulae)
voynich folio-deep        # Phase 24.12: folio isolation and deep examination
voynich section-xfer      # Phase 24.13: cross-section transfer analysis
voynich reverse-eng       # Phase 24.14: reverse engineering from confirmed words
voynich token-gram        # Phase 24.15: token positional grammar
voynich phase24-integrate # Phase 24.16: integration and verdict
voynich phase24           # Run full Phase 24 pipeline
```

## Phase 25: Reading Direction

```bash
voynich boustro           # Phase 25.1: boustrophedon re-ordering test
voynich f6r-manual        # Phase 25.2: folio f6r manual examination
voynich phase25-verdict   # Phase 25.3: combined verdict
voynich phase25           # Run full Phase 25 pipeline
```

## Phase 26: Zodiac Known-Plaintext Attack

```bash
voynich zodiac-map        # Phase 26.1: zodiac folio cataloguing
voynich month-crib        # Phase 26.2: month name crib extraction (6 languages)
voynich astro-crib        # Phase 26.3: astrological vocabulary crib
voynich label-decode      # Phase 26.4: per-label exhaustive CSP decode
voynich zodiac-tab        # Phase 26.5: zodiac-derived assignment table
voynich zodiac-decode     # Phase 26.6: full corpus decode with zodiac table
voynich phase26-validate  # Phase 26.7: 12-test validation battery
voynich phase26-verdict   # Phase 26.8: final verdict
voynich phase26           # Run full Phase 26 pipeline
```

## Phase 27: Peer Review Controls

```bash
voynich gibberish         # Phase 27.1: gibberish/self-citation typology test
voynich naibbe            # Phase 27.2: Naibbe dice cipher entropy shift
voynich phase27-verdict   # Phase 27.3: combined verdict
voynich phase27           # Run full Phase 27 pipeline
```

## Phase 28: Ventris-Style Crib Propagation

```bash
voynich crib-extract      # Phase 28.1: crib word extraction (27 words, 3 sources)
voynich crib-consist      # Phase 28.2: internal consistency check
voynich family-prop       # Phase 28.3: family propagation (correction search)
voynich signal-iso        # Phase 28.4: signal isolation (real vs 5 null corpora)
voynich crib-local        # Phase 28.5: crib localization by section
voynich ventris-tab       # Phase 28.6: Ventris table assembly
voynich ventris-decode    # Phase 28.7: full corpus decode
voynich ventris-read      # Phase 28.8: readability battery (8 tests)
voynich phase28-verdict   # Phase 28.9: final verdict
voynich phase28           # Run full Phase 28 pipeline
```

## Phase 29: Signal-Filtered Readability

```bash
voynich signal-bigram     # Phase 29.1: SIGNAL-filtered bigram plausibility (z=6.14)
voynich signal-context    # Phase 29.2: context analysis (PMI, crib candidates)
voynich signal-folio      # Phase 29.3: SIGNAL folio deep examination
voynich signal-phrase     # Phase 29.4: phrase extraction and scoring
voynich phase29-verdict   # Phase 29.5: final verdict
voynich phase29           # Run full Phase 29 pipeline
```

## Phase 30: Iterative Ventris Bootstrap

```bash
voynich bootstrap         # Phase 30.1: bootstrap loop (4-check candidate confirmation)
voynich boot-signal       # Phase 30.2: post-bootstrap signal re-isolation
voynich boot-bigram       # Phase 30.3: post-bootstrap bigram plausibility
voynich boot-context      # Phase 30.4: post-bootstrap context analysis
voynich boot-folio        # Phase 30.5: post-bootstrap folio examination
voynich boot-read         # Phase 30.6: post-bootstrap readability battery (10 tests)
voynich phase30-verdict   # Phase 30.7: final verdict
voynich phase30           # Run full Phase 30 pipeline
```

## Phase 31: Botanical Anchors + Structural Reframing

```bash
voynich consensus-plants  # Phase 31.1: multi-source consensus plant identification
voynich plant-csp         # Phase 31.2: plant name CSP on folio labels
voynich plant-prop        # Phase 31.3: plant-derived assignment propagation
voynich bot-signal        # Phase 31.4: botanical signal validation
voynich determ-test       # Phase 31.5: gallows as determinatives test
voynich compound-test     # Phase 31.6: compound sign hypothesis test
voynich interleave-test   # Phase 31.7: Language A/B interleaved text separation
voynich reseg-test        # Phase 31.8: EVA re-segmentation (4 merge schemes)
voynich phase31-integrate # Phase 31.9: integration and combined verdict
voynich phase31           # Run full Phase 31 pipeline (all 9 steps)
```

## Phase 32: Compound-Sign Signal Pipeline

```bash
voynich comp-decode       # Phase 32.1: compound-sign corpus decode (real + 5 null corpora)
voynich comp-signal       # Phase 32.2: signal re-classification under compound decode
voynich comp-bigram       # Phase 32.3: bigram plausibility on compound SIGNAL pairs (z=-0.36)
voynich comp-context      # Phase 32.4: PMI context analysis on compound signal vocabulary
voynich comp-bootstrap    # Phase 32.5: bootstrap iteration under compound classifications
voynich comp-folio        # Phase 32.6: annotated folio examination (top SIGNAL folios)
voynich comp-read         # Phase 32.7: 12-test readability battery with cross-phase progression
voynich phase32-verdict   # Phase 32.8: final verdict (COMPOUND_COLLISIONS)
voynich phase32           # Run full Phase 32 pipeline (all 8 steps)
```

## Phase 33: Multi-Vector Error Correction

```bash
voynich anti-diag         # Step 33.1: anti-signal diagnosis (per-triple SIGNAL vs ANTI participation)
voynich triple-rates      # Step 33.2: per-triple signal rates with positional and interaction analysis
voynich signal-swap       # Step 33.3: signal-guided greedy swap (983 candidates, bigram z >= 6.14 gate)
voynich signal-correct    # Step 33.4: signal-corrected full decode + held-out validation
voynich latin-lm          # Step 33.5: Latin character-level n-gram LM (3-gram + 5-gram, add-1 smoothing)
voynich ppl-search        # Step 33.6: perplexity coordinate descent over 25 triples (3 passes)
voynich ppl-validate      # Step 33.7: three-table cross-validation (Phase 15 vs signal vs perplexity)
voynich suffix-gram       # Step 33.8: suffix grammar mapping (EVA suffixes → Latin inflectional endings)
voynich suffix-search     # Step 33.9: suffix-constrained root search for 13 unconfirmed triples
voynich long-crib         # Step 33.10: long botanical crib target identification (16 plants, 15 folios)
voynich long-csp          # Step 33.11: long crib CSP alignment (121 alignments tested, 0 valid)
voynich long-prop         # Step 33.12: long crib propagation (early exit if no new triples)
voynich pair-freq         # Step 33.13: token pair frequency tables (EVA + Latin + decoded)
voynich distrib-match     # Step 33.14: distributional match via Hungarian algorithm + null test
voynich distrib-validate  # Step 33.15: distributional cross-validation (skip if not significant)
voynich phase33-integrate # Step 33.16: cross-approach consensus matrix and final verdict
voynich phase33           # Run full Phase 33 pipeline (all 6 approaches + integration)
```

## Phase 34: Encoding Model Reformation (7 Tracks)

```bash
voynich dict-cal          # Step 34.18: dictionary right-sizing (Track G)
voynich sigla-dict        # Step 34.1: medieval abbreviation dictionary (Track A)
voynich abjad-csp         # Step 34.2: abjad consonant-only CSP (Track A)
voynich sigla-decode      # Step 34.3: sigla-specific decode (Track A)
voynich abjad-signal      # Step 34.4: abjad signal isolation (Track A)
voynich slot-vars         # Step 34.5: slot-conditioned variable fork (Track B)
voynich slot-csp          # Step 34.6: position-conditioned CSP solve (Track B)
voynich slot-signal       # Step 34.7: slot-conditioned signal isolation (Track B)
voynich mixed-lm          # Step 34.8: mixed Latin-Italian LM (Track C)
voynich dialect-decode    # Step 34.9: dialect-conditioned decode (Track C)
voynich dialect-signal    # Step 34.10: dialect signal isolation (Track C)
voynich continua          # Step 34.11: space stripping + character stream (Track D)
voynich reseg-decode      # Step 34.12: Viterbi re-segmentation (Track D)
voynich reseg-signal      # Step 34.13: re-segmented signal isolation (Track D)
voynich gallows-geom      # Step 34.14: gallows-bench spatial geometry (Track E)
voynich spatial-decode    # Step 34.15: spatial-tagged decode (Track E)
voynich vowel-ptr         # Step 34.16: vowel pointer hypothesis test (Track F)
voynich vowel-decode      # Step 34.17: vowel-pointed decode (Track F)
voynich phase34-integrate # Step 34.19: phase 34 integration
voynich phase34           # Run full Phase 34 pipeline
```

## Phase 35: Spatial Conditioning + 10K Dictionary

```bash
voynich spatial-pre       # Step 35.1: spatial gallows preprocessing
voynich comb-decode       # Step 35.2: combined spatial+10K decode
voynich comb-signal       # Step 35.3: combined signal isolation
voynich comb-bigram       # Step 35.4: combined bigram plausibility (THE PREDICTION TEST)
voynich comb-context      # Step 35.5: combined context analysis
voynich comb-bootstrap    # Step 35.6: combined Ventris bootstrap
voynich comb-folio        # Step 35.7: combined folio transliterations
voynich comb-read         # Step 35.8: combined readability battery
voynich phase35-verdict   # Step 35.9: Phase 35 verdict (NO_INTERACTION)
voynich phase35           # Run full Phase 35 pipeline
```

## Phase 46: Final Internal Consolidation

```bash
voynich arb-tables        # Step 46A.1: assemble 8 candidate tables from P15/MaxSAT/CSA/Canonical
voynich arb-bigram        # Step 46A.2: bigram z-score for all 8 tables (500 permutations)
voynich arb-signal        # Step 46A.3: signal word survival check per table
voynich arb-10k           # Step 46A.4: 10K dictionary hit + selectivity per table
voynich arb-select        # Step 46A.5: composite ranking and definitive table selection
voynich track-a-46        # Run full Track A (triple arbitration)
voynich freq-reference    # Step 46B.1: SBM profiles for Latin/Italian reference corpora
voynich freq-cipher       # Step 46B.2: SBM profiles for 5 synthetic ciphers
voynich freq-compare      # Step 46B.3: distance comparison (LANGUAGE_LIKE / CIPHER_LIKE)
voynich track-b-46        # Run full Track B (frequency diagnostic)
voynich final-decode      # Step 46C.1: definitive corpus decode (36K tokens, per-folio stats)
voynich final-annotate    # Step 46C.2: confidence annotations (GREEN/YELLOW/ORANGE/RED)
voynich final-map         # Step 46C.3: structured gap inventory (6 categories)
voynich final-summary     # Step 46C.4: project summary with progression table
voynich track-c-46        # Run full Track C (definitive decode + gap map)
voynich phase46-integrate # Integration: 6-validation battery + verdict
voynich phase46           # Run full Phase 46 pipeline (all 3 tracks + integration)
```

## Phase 47: Z-Score Audit & Structural Reading

```bash
voynich z-reproduce-42    # Step 47A.1: reproduce Phase 29 z=6.14 (131K, exact, 1000 perms)
voynich z-reproduce-46    # Step 47A.2: reproduce Phase 46 z=61.63 (10K, exact+relaxed, 500 perms)
voynich z-diff            # Step 47A.3: identify marginal impact of each methodology difference
voynich z-canonical       # Step 47A.4: canonical z for T_P15, T_MAX, T_CANONICAL, T_CSA
voynich z-sensitivity     # Step 47A.5: sensitivity analysis (dict size, ED threshold, perm count)
voynich track-a-47        # Run full Track A (z-score methodology audit)
voynich disamb-lattice    # Step 47B.1: build per-token decode lattice from tier alternatives
voynich disamb-bigram     # Step 47B.2: build decoded word bigram models (FULL/SIGNAL/GREEN)
voynich disamb-viterbi    # Step 47B.3: word-level Viterbi disambiguation (3 variants)
voynich disamb-eval       # Step 47B.4: evaluate disambiguation quality (dict-hit, bedrock survival)
voynich disamb-compare    # Step 47B.5: compare variants and select best (or declare non-beneficial)
voynich track-b-47        # Run full Track B (word-level disambiguation)
voynich read-ngrams       # Step 47C.1: repeated multi-word sequences (n=2..7)
voynich read-recipes      # Step 47C.2: recipe grammar extraction from 14 pharmaceutical folios
voynich read-topics       # Step 47C.3: folio-level topic clustering (K-means, PMI co-occurrence)
voynich read-star         # Step 47C.4: star folio readings (top 5 by GREEN rate)
voynich read-sections     # Step 47C.5: section-level vocabulary differentiation (JSD, chi-squared)
voynich track-c-47        # Run full Track C (structural reading)
voynich seq-overlap       # Step 47D.1: folio-to-folio vocabulary overlap (226x226 Jaccard)
voynich seq-continuity    # Step 47D.2: cross-folio word continuity at 225 boundaries
voynich seq-boundary      # Step 47D.3: anomalous sequence boundary detection
voynich seq-reorder       # Step 47D.4: local reordering test at anomalous boundaries
voynich track-d-47        # Run full Track D (sequence analysis)
voynich phase47-integrate # Integration: 8-validation battery + verdict
voynich phase47           # Run full Phase 47 pipeline (all 4 tracks + integration)
```

## Phase 48: Marginal Bilingual Crib Exploitation

```bash
voynich f116v-transcribe  # Step 48A.1: extract f116v Voynichese words from IVTFF marginal
voynich f116v-decode      # Step 48A.2: decode f116v words via T_P15 + lattice alternatives
voynich f116v-context     # Step 48A.3: compile 5 competing scholarly readings
voynich f116v-match       # Step 48A.4: match decoded output against contextual readings
voynich f116v-reverse     # Step 48A.5: reverse-engineer triple assignments from best matches
voynich track-a-48        # Run full Track A (f116v decode)
voynich f17r-extract      # Step 48B.1: extract f17r marginal content
voynich f66r-extract      # Step 48B.2: extract f66r marginal content
voynich margin-decode     # Step 48B.3: decode secondary marginals via T_P15
voynich margin-hand       # Step 48B.4: hand/dialect evidence compilation
voynich track-b-48        # Run full Track B (secondary marginals)
voynich marci-source      # Step 48C.1: locate Marci annotation source data
voynich marci-extract     # Step 48C.2: extract machine-readable transcription
voynich marci-compare     # Step 48C.3: compare Marci readings to T_P15 decode
voynich marci-test        # Step 48C.4: corpus-wide consistency test
voynich track-c-48        # Run full Track C (Marci annotations)
voynich crib-collect      # Step 48D.1: collect cribs from all tracks
voynich crib-consistent   # Step 48D.2: cross-track consistency matrix
voynich crib-propagate    # Step 48D.3: greedy propagation to assignment table
voynich crib-decode       # Step 48D.4: decode corpus with updated table
voynich crib-validate     # Step 48D.5: canonical bigram z-score validation
voynich track-d-48        # Run full Track D (crib propagation)
voynich phase48-integrate # Integration: 8-validation battery + verdict
voynich phase48           # Run full Phase 48 pipeline (all 4 tracks + integration)
```

## Reviewer Response Analyses

```bash
voynich reviewer-perm          # Random syllabary permutation test (1000 trials, ~30 min)
voynich reviewer-coherence     # Signal word coherence check (1000 trials, ~30 min)
voynich reviewer-family        # Within-family phonetic entropy test (1000 trials, <1 sec)
voynich reviewer-rabidi        # Rabidi sensitivity analysis
voynich reviewer-fingerprint   # Fingerprint cosine gap analysis
voynich reviewer-all           # All three + integration
```

## Phase 54: Gallo-Italic Dialect Identification

```bash
voynich degemination       # Phase 54.1: systematic degemination test (geminate simplification)
voynich lenition           # Phase 54.2: intervocalic voicing pattern test
voynich articles           # Phase 54.3: article and pronoun system matching
voynich pharma-region      # Phase 54.4: pharmaceutical terminology regionalization
voynich co-syntax          # Phase 54.5: 'co' syntactic validation (preposition test)
voynich verb-morph         # Phase 54.6: verb morphology deep dive (dire/dare paradigms)
voynich dialect-sim        # Phase 54.7: simulated macaronic text comparison
voynich zodiac-dialect     # Phase 54.8: zodiac label dialect decode
voynich dialect-verdict    # Phase 54: aggregate all experiments into final verdict
voynich phase54            # Run full Phase 54 pipeline (all 8 experiments + integration)
```

## Phase 55: Entropy Shift Generalization + Currier Self-Correlation

```bash
# Track A: extend entropy shift ranking to Schinner + Cardan
voynich schinner-gen       # Phase 55A.1: Schinner stochastic model (2 variants × 20 seeds × 36K tokens)
voynich cardan-gen         # Phase 55A.2: Rugg-Taylor Cardan grille (2 variants × 20 seeds × 36K tokens)
voynich entropy-extended   # Phase 55A.3: merge into 13-mechanism ranking; verdict SCHINNER_ABOVE_TACHYGRAPHY

# Track B: Currier cross-boundary self-correlation prediction
voynich currier-voynich    # Phase 55B.1: MI on real Voynich + 1000-shuffle null (ratio=1.450×, z=24.9σ)
voynich currier-tachy      # Phase 55B.2: tachygraphic simulation (syl=1.284×, word=1.061×)
voynich currier-controls   # Phase 55B.3: Latin (1.147×), Schinner (1.044×), Cardan (1.001×)
voynich currier-verdict    # Phase 55B.4: integrate Track B; verdict PREDICTION_CONFIRMED_UNIQUE

# Integration
voynich phase55-verdict    # Phase 55: integrate Track A + Track B (verdict PARTIAL, 3/6)
voynich phase55            # Run full Phase 55 pipeline (~15–25 min)
```

## Phase 56: Costamagna Structural Compatibility

```bash
voynich costamagna-compare  # Phase 56: 10 structural questions comparing Costamagna 1953 vs Voynich (verdict COMPATIBLE, 10/10)
```

## Phase 57: CVC Coda Decode

```bash
voynich coda-table         # Step 57.1: Build coda marker mapping table (stroke → coda consonant)
voynich cvc-coda-signal    # Step 57.4: Signal isolation on CVC decoded corpus
voynich cvc-compare        # Step 57.5: Compare CV vs CVC vs R3 decode strategies (4-way battery)
voynich cvc-tokens         # Step 57.8: Diagnostic detail on top-20 most frequent tokens
voynich phase57-verdict    # Step 57.7: Validation gates (7) and verdict
voynich phase57            # Run full Phase 57 pipeline
```

## Phase 58: Costamagna-Constrained CSP

```bash
voynich cost-domains       # Step 58.1: Build Costamagna-constrained CSP domains
voynich cost-reduction     # Step 58.2: Compare domain sizes across phases (11 vs 14 vs Costamagna)
voynich cost-csp           # Step 58.3: Run CSP with Costamagna domains (greedy hill-climbing)
voynich cost-compare       # Step 58.5: Compare best CSP solution vs T_P15
voynich phase58-verdict    # Step 58.6: Validation gates (8) and verdict
voynich phase58            # Run full Phase 58 pipeline
```

## Phase 59: CVC Refinement + Deep Investigation

```bash
# Tier 1: Foundational
voynich cvc-segment        # Inv 1: Syllable segmentation of CVC output (Costamagna maximal-munch)
voynich cvc-position       # Inv 6: Positional distribution of coda markers

# Tier 2: Mapping Refinement
voynich cvc-tm             # Inv 3: Resolve t/m coda ambiguity (vertical stroke group)
voynich cvc-connector      # Inv 7: Test 7 coda candidates for connector group (b,h,ckh,u)

# Tier 3: Content and Evaluation
voynich cvc-dict           # Inv 2: Build CVC-aware dictionary and re-score
voynich cvc-gloss          # Inv 4: Gloss CVC signal words (Latin/Italian lookup)
voynich cvc-recipe         # Inv 9: Recipe reading under CVC decode
voynich cvc-aiin           # Inv 10: The "aiin" family deep dive (hook→n Latin morphology)

# Tier 4: Validation and Prediction
voynich cvc-mi             # Inv 5: Cross-boundary MI under CVC decode
voynich cvc-combo          # Inv 8: Test Costamagna combination rules
voynich cvc-perm           # Inv 11: CVC permutation coherence test (1000 trials)

# Integration
voynich phase59-verdict    # Phase 59 verdict: integrate all 11 investigations
voynich phase59            # Run full Phase 59 pipeline
```

## Phase 60: Corrected CVC + Evaluation + Recipes

```bash
voynich corrected-coda     # Phase 60A: Corrected coda mapping (connector=r, i=syllabic)
voynich recal-coherence    # Phase 60B: Recalibrated CVC permutation coherence test
voynich cvc-eval           # Phase 60C: Unified CVC evaluation framework
voynich recipe-annotate    # Phase 60D: Recipe annotation + reading attempts
voynich phase60-verdict    # Phase 60 verdict: integrate all 4 tracks
voynich phase60            # Run full Phase 60 pipeline
```

## Phase 61: Deep Reading + Permutation + Sequences + Zodiac

```bash
voynich deep-recipes       # Phase 61A: Deep pharmaceutical recipe reading
voynich cvc-full-perm      # Phase 61B: Full CV permutation test under CVC decode
voynich cost-sequences     # Phase 61C: Costamagna sequence rule testing
voynich zodiac-cvc         # Phase 61D: Zodiac labels under CVC decode
voynich phase61-verdict    # Phase 61 verdict
voynich phase61            # Run full Phase 61 pipeline
```

## Phase 62: Exhaustive Pre-Visual Analysis

```bash
voynich t1-reverse         # Phase 62.1: T1 reverse engineering under CVC
voynich cross-token        # Phase 62.2: Cross-token word reconstruction
voynich gallows-initial    # Phase 62.3: Gallows as word-initial markers
voynich decoded-bigram     # Phase 62.4: Decoded bigram frequency vs Latin
voynich orphaned-coda      # Phase 62.5: Orphaned coda investigation
voynich double-mod         # Phase 62.6: Double-modifier sequences
voynich token-length       # Phase 62.7: Token length distribution
voynich syl-entropy        # Phase 62.8: Syllable-level entropy
voynich lang-ab-cvc        # Phase 62.9: Language A/B under CVC
voynich hand-cvc           # Phase 62.10: Hand-by-hand CVC analysis
voynich multi-entropy      # Phase 62.11: Multi-level entropy comparison
voynich phase62-verdict    # Phase 62 verdict
voynich phase62            # Run full Phase 62 pipeline
```

## Phase 63: Visual Sign Comparison

```bash
# Workstream A: Font-based comparison
voynich vis-render         # Phase 63 A1: Render EVA glyphs from font
voynich vis-normalize      # Phase 63 A2: Normalize EVA + Costamagna images
voynich vis-embed          # Phase 63 A3: Embed images via Gemini
voynich vis-similarity     # Phase 63 A4: Compute visual similarity matrix
voynich vis-validate       # Phase 63 A5: Validate T_P15 against visual rankings
voynich vis-report         # Phase 63 A6: Generate HTML visual comparison report
voynich vis-rerun          # Phase 63A: Clear caches and re-run Workstream A
voynich phase63-verdict    # Phase 63A verdict

# Workstream B: Manuscript segmentation
voynich ms-index           # Phase 63B B1: Extract and index folio images
voynich ms-segment         # Phase 63B B2-B4: Segment lines, words, characters
voynich ms-exemplars       # Phase 63B B5: Select character exemplars
voynich ms-compare         # Phase 63B B6: Embed exemplars and compare
voynich phase63b-verdict   # Phase 63B verdict
voynich phase63b           # Run full Phase 63B pipeline
```

## Phase 64: Multi-Method Visual Sign Comparison

```bash
voynich morph-describe     # Phase 64 M1: LLM morphology descriptions
voynich stroke-extract     # Phase 64 M2: Skeleton graph features
voynich shape-desc         # Phase 64 M3: Shape descriptors (Hu + Fourier)
voynich topo-features      # Phase 64 M4: Topological features
voynich hog-compare        # Phase 64 M5: HOG features
voynich hybrid-features    # Phase 64 M6: Hybrid combined features
voynich llm-pairwise       # Phase 64 M7: LLM pairwise comparison
voynich visual-ensemble    # Phase 64: Ensemble combination + validation
voynich phase64-verdict    # Phase 64 verdict
voynich phase64            # Run full Phase 64 pipeline
```

## Phase 65: Word Boundary Discovery

```bash
voynich build-stream       # Phase 65.1: Build decoded character streams
voynich harris-segment     # Phase 65.2: Harris MI boundary detection
voynich bayesian-segment   # Phase 65.3: Bayesian word segmentation
voynich lm-segment         # Phase 65.4: Character LM perplexity minimization
voynich recipe-segment     # Phase 65.5: Recipe template-constrained segmentation
voynich phase65-verdict    # Phase 65 verdict
voynich phase65            # Run full Phase 65 pipeline
```

## Phase 66: Multi-Vector Attack with Hallucination Controls

```bash
voynich llm-reading        # Phase 66.1: LLM pharmaceutical reading with controls
voynich reverse-sim        # Phase 66.2: Reverse simulation (Viterbi)
voynich f116v-crib         # Phase 66.3: f116v crib test
voynich illus-align        # Phase 66.4: Illustration-text alignment
voynich parallel-align     # Phase 66.5: CI parallel corpus alignment
voynich fontana-struct     # Phase 66.6: Fontana structural comparison
voynich lang-a-66          # Phase 66.7: Language A focus
voynich hand4              # Phase 66.8: Hand 4 focus
voynich collocations       # Phase 66.9: Collocational analysis
voynich ngram-freq         # Phase 66.10: N-gram frequency ranking
voynich metrical           # Phase 66.11: Metrical analysis
voynich astro-deep         # Phase 66.12: Astronomical deep dive
voynich phase66-verdict    # Phase 66 verdict
voynich phase66            # Run full Phase 66 pipeline
```

## Phase 67: Multi-Angle Triple Resolution

```bash
voynich wildcard-match     # Phase 67.1: Wildcard pattern matching
voynich freq-match         # Phase 67.2: Frequency rank matching
voynich feat-predict       # Phase 67.3: Feature-based prediction
voynich evo-optimize       # Phase 67.4: Evolutionary optimization
voynich distrib-map        # Phase 67.5: Distributional mapping
voynich phase67-verdict    # Phase 67 verdict
voynich phase67            # Run full Phase 67 pipeline
```

## Phase 68: Rare Syllable Recovery

```bash
voynich full-tokens        # Phase 68.1: Fully-decoded token exploitation
voynich within-token       # Phase 68.2: Within-token co-occurrence
voynich paradigmatic       # Phase 68.3: Minimal pair analysis
voynich expanded-t1        # Phase 68.4: CVC-enhanced T1 pipeline
voynich formula-decode     # Phase 68.5: Formulaic pattern decode
voynich distrib-constrain  # Phase 68.6: Distributional constraint propagation
voynich ed-lattice         # Phase 68.7: Edit-distance lattice
voynich phase68-verdict    # Phase 68 verdict
voynich phase68            # Run full Phase 68 pipeline
```

## Phase 69: The Clean Core — Validation, Exploitation, and Reading

```bash
voynich build-clean        # Phase 69.0: Build clean corpus partition
voynich validate-clean     # Phase 69.1: Validate clean subset (3 permutation tests)
voynich clean-segment      # Phase 69.2: Harris MI + LM segmentation on clean runs
voynich clean-llm-read     # Phase 69.3: LLM reading of clean passages
voynich clean-distrib      # Phase 69.4: Enhanced Procrustes with 200+ anchors
voynich t1-network         # Phase 69.5: T1 vocabulary network analysis
voynich t1-read            # Phase 69.6: T1-anchored passage reading
voynich t1-ci-crossref     # Phase 69.7: T1 × CI cross-reference
voynich phase69-verdict    # Phase 69 verdict
voynich phase69            # Run full Phase 69 pipeline
```

## Phase 70: Token-as-Word Exploitation

```bash
voynich pharma-dict        # Phase 70.1: Pharmaceutical dictionary expansion
voynich paradigm-map       # Phase 70.2: Morphological paradigm mapping
voynich phrase-assemble    # Phase 70.3: Phrase fragment assembly
voynich annotate-read      # Phase 70.4: Annotated pharmaceutical readings
voynich phase70-verdict    # Phase 70 verdict
voynich phase70            # Run full Phase 70 pipeline
```

## Phase 71: Inflectional Reverse Engineering

```bash
voynich inflect-catalog    # Phase 71.1: Inflectional catalog
voynich root-id            # Phase 71.2: Root-level paradigm identification
voynich gram-read          # Phase 71.3: Grammatically-annotated passage reading
voynich phase71-verdict    # Phase 71 verdict
voynich phase71            # Run full Phase 71 pipeline
```

## Phase 72: Decode Model Diagnosis and Revision

```bash
voynich connector-test     # Phase 72.1: Connector value investigation (13 values)
voynich xval-diagnosis     # Phase 72.2: Cross-validation failure diagnosis
voynich combo-models       # Phase 72.3: Alternative CVC combination models
voynich t1-expand72        # Phase 72.4: Tiered T1 vocabulary expansion
voynich var-length         # Phase 72.5: Variable-length encoding hypothesis
voynich phase72-verdict    # Phase 72 verdict
voynich phase72            # Run full Phase 72 pipeline
```

## Phase 73: Corrected Model Pipeline (Connector→Null)

```bash
voynich redecode           # Phase 73.0: Re-decode corpus with connector→null
voynich revalidate-clean   # Phase 73.1: Re-validate clean subset
voynich corrected-grammar  # Phase 73.2: Corrected inflectional catalog
voynich corrected-t1       # Phase 73.3: T1 re-identification + stability check
voynich corrected-paradigms # Phase 73.4: Paradigm mapping with corrected decode
voynich corrected-read     # Phase 73.5: Annotated readings with corrected data
voynich phase73-verdict    # Phase 73 verdict
voynich phase73            # Run full Phase 73 pipeline
```

## Phase 74: Descender Investigation + T1 Vocabulary Push

```bash
# Path A: Descender investigation
voynich descender-test     # Phase 74.A1: Exhaustive descender value testing (13 values)
voynich descender-context  # Phase 74.A2: Context-dependent descender analysis

# Path B: T1 vocabulary push
voynich eva-patterns       # Phase 74.B1: EVA-level distributional + positional expansion
voynich llm-gap-fill       # Phase 74.B2: LLM gap-filling with hallucination controls
voynich complete-read      # Phase 74.B3: Assemble complete readings

# Integration
voynich phase74-verdict    # Phase 74 verdict
voynich phase74            # Run full Phase 74 pipeline
```

## Phase 75: 3-Coda Model Pipeline (Connector→Null + Descender→Null)

```bash
voynich redecode-3coda     # Phase 75.0: Re-decode corpus with 3-coda model
voynich revalidate-3coda   # Phase 75.1: Re-validate clean subset under 3-coda model
voynich grammar-3coda      # Phase 75.2: Corrected grammatical analysis (3-coda model)
voynich t1-3coda           # Phase 75.3: Corrected T1 identification (3-coda model)
voynich paradigms-3coda    # Phase 75.4: Corrected paradigm mapping (3-coda model)
voynich read-3coda         # Phase 75.5: Corrected readings with distributional integration
voynich phase75-verdict    # Phase 75 verdict
voynich phase75            # Run full Phase 75 pipeline
```

## Phase 76: Triple Resolution from Vocabulary Convergence

```bash
voynich wildcard-prop      # Phase 76.1: T1 wildcard constraint extraction + LOO validation
voynich skeleton-parse     # Phase 76.2: Grammatical skeleton parsing + parallel passages
voynich freq-gap           # Phase 76.3: Frequency-identification gap analysis
voynich cond-gapfill       # Phase 76.4: Conditional LLM gap-fill re-run
voynich phase76-verdict    # Phase 76 verdict
voynich phase76            # Run full Phase 76 pipeline
```

## Phase 77: Timm-Schinner Self-Citation Discriminator Test

```bash
voynich ts-test            # Phase 77: Entropy shift + cross-boundary MI test (540 corpora)
voynich phase77            # Run Phase 77 (alias for ts-test)
```

## Phase 78: CVC T1 Permutation Validation

```bash
voynich cvc-t1-perm        # Phase 78: 1,000 random CV tables through T1 pipeline (p=0.002, z=3.79)
voynich phase78            # Run Phase 78 (alias for cvc-t1-perm)
```
