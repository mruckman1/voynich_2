# The Voice but Not the Song

**Voynich Manuscript computational analysis**: stroke-feature syllabary decoding, signal isolation, and Italian tachygraphic hypothesis testing.

Companion code for: Ruckman (2026), [*"The Voice But Not the Song: A Shorthand Hypothesis and the Statistical Fingerprint of the Voynich Manuscript"*](The_Voice_But_Not_the_Song.pdf).

## Overview

Two independent pipelines — one treating each Voynich character as a letter ([Approach 1](https://github.com/mruckman1/voynich)), the other as a syllable (this repository) — produce consistent structural conclusions:

- The source language is **Romance** (a mix of Latin and Italian)
- The content is **medieval medical/herbal**
- Two distinct subsystems coexist (Currier A/B)
- The morphological structure is **genuine**
- **Italian syllabic tachygraphy** is identified as the encoding mechanism, discriminated from 13 tested mechanisms; Schinner's stochastic model exposes a scope limitation of the discriminator (trained on Voynich data, see Phase 55); Rugg-Taylor's Cardan grille is clearly discriminated (cosine +0.49–+0.59 vs tachygraphy's +0.820); Timm & Schinner's self-citation algorithm is decisively eliminated (cosine −0.153, MI ratio 1.036× — anticorrelated and null-level, Phase 77)
- **56 decoded words** are validated under permutation testing (p = 0.001 for count; p = 0.011 for linguistic coherence)
- **22 word-level content identifications** are independently validated (p = 0.009)
- Individual words and syllables have been decoded, but **no connected passage of readable text** has been produced

## Quick Start

```bash
uv sync
uv pip install -e .

# Key analyses (aligned to paper sections):
voynich corpus            # Load and summarize the EVA corpus
voynich fingerprint       # Approach 2: information-theoretic fingerprinting (Paper Sec 2.2)
voynich feature-csp       # Phase 14: stroke-feature CSP — the breakthrough (Paper Sec 2.2)
voynich entropy-shift     # Phase 19: tachygraphic identification (Paper Sec 4.2)
voynich naibbe-test       # Phase 27: Naibbe cipher rejection (Paper Sec 4.2)
voynich signal-iso        # Phase 28: signal isolation methodology (Paper Sec 5)
voynich signal-bigram     # Phase 29: SIGNAL bigram test, z=6.14 (Paper Sec 6.3)
voynich reviewer-perm     # Permutation test: 1000 random tables (Paper Sec 6.2)
voynich reviewer-coherence # Coherence test: verb + function + pharma (Paper Sec 6.2)
```

For the complete reference of all commands, see [docs/commands.md](docs/commands.md).

Alternatively, use `python -m voynich <command>` without installing.

## Paper-to-Code Map

| Paper Section | Topic | Key Phases | Entry Points |
|---------------|-------|------------|--------------|
| Sec 2.1 | Consonant-skeleton matching | 1 | `voynich strokes` |
| Sec 2.2 | Syllabary analysis + signal isolation | 1, 14 | `voynich fingerprint`, `voynich feature-csp` |
| Sec 3.1 | Source language is Romance | 4, 9, 50D | `voynich lang-compare`, `voynich phase9` |
| Sec 3.2 | Medieval medical/herbal content | 4, 15, 47 | `voynich section-diagnosis`, `voynich text-analysis`, `voynich read-recipes` |
| Sec 3.3 | Genuine morphological structure | 5 | `voynich paradigms` |
| Sec 4.1 | Three-way ambiguity | 18 | `voynich hypothesis` |
| Sec 4.2 | Entropy shift discriminator (13 mechanisms) | 19, 55, 77 | `voynich entropy-shift`, `voynich entropy-extended`, `voynich naibbe-test`, `voynich ts-test` |
| Sec 4.3 | Sign family structure | 19 | `voynich tachy-stroke`, `voynich reviewer-family` |
| Sec 4.4 | Costamagna structural compatibility | 56 | `voynich costamagna-compare` |
| Sec 5 | Signal isolation methodology | 17, 28 | `voynich null-corpus`, `voynich signal-iso` |
| Sec 6.1 | Signal words | 36–38 | `voynich phase36` |
| Sec 6.2 | Permutation test | Reviewer | `voynich reviewer-perm`, `voynich reviewer-coherence` |
| Sec 6.3 | Word-level identifications | 52 | See [phase docs](docs/phases/phase-49-53.md) |
| Sec 6.4 | EVA words as syllables | 29, 55 | `voynich signal-bigram`, `voynich currier-voynich` |
| Sec 7 | CVC coda system | 57–60, 72–75, 78 | `voynich cvc-coda-signal`, `voynich phase75`, `voynich cvc-t1-perm` |
| Sec 8 | Encoding structure | 14–16, 31 | `voynich feature-csp`, `voynich determ-test` |
| Sec 9 | Limitations & failures | 17, 20–23, 33 | `voynich step0`, `voynich phase33` |
| Sec 10 | Additional properties | 24 | `voynich direction`, `voynich section-xfer` |
| App A | Phase narratives | 1–77 | [docs/phases/](docs/phases/) |
| App B | Complete signal vocabulary | 36–38 | [docs/signal-vocabulary.md](docs/signal-vocabulary.md) |
| App C | Z-score methodology audit | 47 | `voynich track-a-47` |
| App H | CVC signal vocabulary | 57–60, 73 | [docs/signal-vocabulary.md](docs/signal-vocabulary.md) |
| App I | Example decoded passages | 70, 75 | [docs/phases/phase-75.md](docs/phases/phase-75.md) |

## Key Results

### Encoding: Italian Syllabic Tachygraphy
- Entropy shift cosine similarity: **+0.820** (tachygraphy) vs **-0.843** (Naibbe cipher) — mirror-image signatures
- **13-mechanism ranking** (Phase 55): Rugg-Taylor Cardan grille discriminated (+0.49–+0.59, 20 seeds); Schinner's trained-on-Voynich Markov model scores +0.95–+0.97, revealing that the entropy shift test cannot distinguish tachygraphy from any model that memorizes the Voynich's own character statistics (discriminator scope is encoding mechanisms applied to independent plaintext)
- Resolves the three-way ambiguity: simultaneously a constructed system (H1), encoding natural language (H2), with systematic vocabulary (H3)
- Tested against the author's parameterized model, not historical specimens (none survive in original script form)
- **Currier cross-boundary self-correlation**: tachygraphic simulation (syllable-as-token) produces 1.284× predictability ratio vs Voynich's 1.450× (11% difference); word-as-token control gives only 1.061×, confirming the anomaly is driven by syllable-boundary structure; Schinner's model gives 1.044× (near null at 1.044×), showing the Markov model that outperforms tachygraphy on entropy shift cannot explain this independent anomaly

### Language: Macaronic Latin-Italian
- Italian selectivity **5.45×** vs Latin **1.30×** under signal isolation
- Size-matched language ID (all corpora subsampled to 11K tokens): Italian #1, Latin #2
- Earlier German ranking was entirely a corpus-size artifact

### Signal Words: 56 Permutation-Validated
- Real table produces 56 signal words vs random mean of ~33 (**p = 0.001**)
- Per-word selectivity (~3.8×) is structural, not table-specific (p = 0.26)
- Linguistic coherence is rare: only **1.1% of random tables** produce verb paradigms + function words + pharmaceutical terms simultaneously (**p = 0.011**)
- Two words (*de*, *bene*) produced independently by both pipelines

### Content Vocabulary: 22 Word-Level Identifications
- Pharmaceutical Latin: *ratione*, *coralli*, *diasene*, *stercora*, *radicom*, *commune*, *secundi*
- **p = 0.009** under word-level permutation test; 74.4% of random tables produce zero identifications
- 7 of 9 identified words appear in fewer than 1% of random trials

### What Remains Undecoded
- **13 free triples** (~59% of tokens) cannot be recovered computationally
- Solution landscape is formally FLAT (500+ near-optimal solutions)
- 6 independent correction methods propose different assignments with zero consensus
- Encoding granularity is variable-length (1-3 characters), not fixed 2-character CV
- **No connected passage of readable text** has been produced
- **Phase 62 confirms**: EVA tokens encode 2-3 syllables each (not single syllables); token boundaries ≠ word boundaries; decoded entropy matches Latin at all levels (overall similarity 0.89); Language A decodes 14pp better than Language B; geminate clusters (rr/ss) indicate remaining modifier classification issues
- **Phase 63 confirms**: General-purpose multimodal embeddings (Gemini Embedding 2) cannot resolve visual correspondence between Voynich and Costamagna signs — all signs collapse to 0.874 ± 0.025 similarity; manuscript segmentation pipeline works (80% line match, 73% word segmentation) but comparison method lacks stroke-level resolution
- **Phase 64 confirms**: Multi-method visual comparison (7 methods: skeleton graphs, shape descriptors, topology, HOG, LLM morphology, LLM pairwise) produces WEAK_SUPPORT (2/7 gates); LLM pairwise win rate 56% (proposed > controls); permutation p=0.138 (not significant); best matches: y→si (mean rank 25.5/236), a→ra (41.2), cph→pa (56.3); font-vs-handwriting domain gap is the primary obstacle
- **Phase 65 confirms**: Word boundary discovery in the decoded character stream FAILS (0/4 gates) — 4 methods (Harris MI, Bayesian MDL, character LM, recipe template DP) all achieve Latin calibration F1 > 0.65 but produce 0–7.5% dict hit on the Voynich (all below EVA baseline of 15.4%); the 17-character decoded alphabet and 56% decode error rate prevent character-level word boundary statistics from emerging
- **Phase 66 confirms**: Multi-vector attack with hallucination controls (12 tracks, 4 tiers) produces STRUCTURAL_INSIGHT (5/12 gates); LLM pharmaceutical reading with blind null controls: real passages score 78.7% dict_hit but null corpus scores 65.6% (ratio only 1.20×) — hallucination controls catch the LLM reading character-frequency patterns, not word-level content; Fontana tachygraphic system is STRUCTURALLY_SIMILAR (10 vs 12 sign families, rotation principle shared); Hand 4 biological section shows 280 recurring bigrams consistent with formulaic pharmaceutical text; f116v crib tokens decode plausibly (`oror`→`nes` ED=0, `sheey`→`sera` ED=1); illustration alignment (0/2), CI parallel (1/2), and frequency ranking (1/2) all fail — 56% decode error rate remains the bottleneck
- **Phase 67 confirms**: Multi-angle triple resolution (5 tracks: wildcard, frequency, features, evolutionary, distributional) produces PARTIAL_RESOLUTION (3/5 gates); 8/13 unresolved triples receive LIKELY consensus (2 tracks agree), 0 RESOLVED (3+ agree); evolutionary search finds +1.3% dict hit improvement; distributional mapping passes all 4 gates (39 Procrustes anchors, 23.1% exact hit, 46.2% related hit on signal words); wildcard matching recovers 16 signal words but selectivity only 1.13× vs null; voting collapses ambiguous triples toward common confirmed syllables (di, co, se) rather than discovering distinct values; bigram z improves (117.69→125.86) but signal words decrease (80→71); computational approaches exhaust their resolving power — visual stroke-matching remains the path forward
- **Phase 68 confirms**: Rare syllable recovery via 7 token-level tracks (fully-decoded tokens, within-token co-occurrence, minimal pairs, expanded T1, formulaic patterns, distributional constraints, ED lattice) produces PARTIAL_RESOLUTION (5/5 gates); 1 RESOLVED triple (`ascender,crossbar,compound`: be→de, 3/7 tracks agree), 5 LIKELY (2-track); dict hit improves +1.57% (29.0→30.6%); signal preserved (80→81); 63% of tokens (22,823) are fully decoded with 0% error (longest run: 29 tokens); CVC-enhanced T1 finds 223 identifications (up from 22); 4,914 minimal pairs yield 2,149 diagnostic constraints; 82 recurring formulaic patterns in recipe sections; token-level approaches find specific rare values where corpus-level optimization always converges on common defaults
- **Phase 69 confirms**: Clean core validation and exploitation (7 tracks, 22,823 tokens with 0% unresolved characters) produces CLEAN_CORE_PARTIAL (4/7 tracks, 15/25 gates); mandatory validation: coherence p=0.006 (PASS) but CV permutation p=0.092 and coda permutation p=0.318 (both FAIL) — confirmed triples are validated by *linguistic coherence*, not raw dict-hit alone; **segmentation FAILS even on 0%-error data** (LM Viterbi 13.2% << EVA baseline 40.6%) — EVA token boundaries are structurally essential and cannot be recovered from character streams; LLM reading produces 2.41× vs null but real/shuffled=1.00× — signal is at individual word level, not sequential phrase structure; **T1 vocabulary network**: 49 morphological paradigms (shared Latin roots), 888 sequential proximity pairs, 893 CI-matching pairs — decoded output preserves Latin inflectional morphology; 3,036 T1-dense passages at 88–93% dict-hit; 74% of T1 words (66/89) attested in Circa Instans; distributional mapping fails (43 anchors, 5.1% convergence) — EVA tokens don't correspond 1:1 to Latin words, consistent with tachygraphic encoding
- **Phase 70 confirms**: Token-as-word exploitation (4 tracks, 11/19 gates) produces PHARMACEUTICAL_READING; **dictionary expansion is NOT the bottleneck** — adding 20K+ pharmaceutical/dialectal words yields only +0.3% dict-hit and 9 new word types; **CVC codas encode Latin grammar**: coda -s→verb 2sg (100% of 20 obs), -t→verb 3sg (82% of 11 obs), -r→passive -tur; 11 EVA suffix→case mappings at 100% consistency; **phrase structure visible**: 818 ordered T1 pairs with 66 VERB+OBJ/PREP+NOUN classifications, 50 glossed trigrams; **annotated readings pass all 6 gates**: 20 passages at 80% mean identified (2.29× vs random), 18/20 above 70%, 20/20 CI chapter matches; T1-dense passages genuinely concentrate identified vocabulary; readings remain fragmentary — 56% decode error rate is the sole remaining bottleneck
- **Phase 71 confirms**: Inflectional reverse engineering (3 tracks, 11/16 gates) produces MARGINAL verdict; **coda -r massively overrepresented** (16,916 tokens = 47% of coda tokens) — connector→r + descender→r double-mapping creates 35% VERB_PASSIVE corpus-wide, far exceeding any natural Latin text (~15% expected); null validation FAILS (p=0.26, random coda→grammar permutations produce equally appropriate distributions); **section profiles highly significant** (chi² p=6.3×10⁻²¹³) — herbal/pharmaceutical sections show lower verbal fraction than astronomical/biological; **342 paradigms discovered** (79.5% roots identified, 47.9% corpus coverage) but paradigms are prefix-groupings of decoded strings, not genuine Latin inflectional families; **grammatical readings achieve 90% grammatical coverage** (mean across 20 passages) and 83% lexical identification (1.91× vs random), but template selectivity 0.95× (VERBAL→NOMINAL pattern is so pervasive that random passages match CI templates equally); cross-validation agreement only 24% (coda-based vs ending-based classification); the coda-to-grammar mapping from Phase 70 reflects real observations at the paradigm level but does not scale to corpus-wide grammatical classification
- **Phase 72 confirms**: Decode model diagnosis and revision (5 tracks, 16/23 gates) produces MODEL_REVISED verdict; **connector→null is the correct mapping** — exhaustive testing of 13 values (6 consonants, 5 vowels, null, boundary) shows null wins on composite score (0.612 vs r's 0.576), dict-hit (30.2% vs 29.0%), and cross-validation (90.5% vs 77.9%); **98.1% of connectors are token-medial** (not final), consistent with scribal ligatures not coda consonants; **per-coda cross-validation validates the other three codas**: -s 92.8%, -t 86.5%, -n 54.8% (small sample), -r 16.9% (catastrophic — the sole error source); **null_connector combination model wins** (composite 0.661 vs append 0.626, 4/5 gates); T1 expansion beyond Tier B (223 IDs) produces 100% false positive rate — relaxation adds only noise; variable-length encoding (7/12 triples prefer 1-char) produces +13.7pp dict-hit but bigram z drops 87→61 (collision artifact identical to Phase 32); **three tracks independently converge**: connector strokes are non-phonetic scribal features that should be dropped from the CVC decode
- **Phase 73 confirms**: Corrected model pipeline (6 tracks, 15/23 gates) produces CORRECTION_NEUTRAL verdict; **connector→null applied corpus-wide** — dict-hit 29.0%→30.2%, cross-validation 77.9%→90.5%, bigram z 87→90, signal 75→76; 2,662 tokens changed (7.3%), 419 new dict hits gained vs 2 lost; **T1 vocabulary robust** (89.7% stable, 223→243 identifications); **PREPARATION roots increase 2→15** (pharmaceutical verbs col-/ter-/coc-/mis- resolved); **verbal fraction unchanged at 57%** — descender→r (14,164 tokens) dominates coda production, not connector→r (removed ~2,752); grammar null test still fails; template selectivity 0.28× (grammar too uniform); coherence validated (p=0.027); **connector→null adopted as new CVC baseline**
- **Phase 74 confirms**: Descender investigation + T1 vocabulary push (5 tracks, 12/18 gates) produces DESCENDER_RESOLVED_AND_VOCAB_EXPANDED verdict; **descender→r ranks 10th of 13 values** — 'm' wins globally but **13/15 preceding triples independently prefer null** (same convergence as Phase 72 connector→null); verbal fraction drops 65.1%→24.2% (m) or 25.9% (null), approaching CI-expected ~15%; descender is **94.6% token-final** (genuine coda marker, unlike connector's 1.9%), but likely encodes a non-phonetic feature; **effective coda system may be just 3 consonants** (hook→n, sigmoid→s, vertical→t); **675 new EVA types identified** via distributional (595, mean cosine 0.474) and positional (173) patterns, total 898 types; **LLM gap-filling feasible but blocked** — 40% known-answer accuracy, 3.82× confidence selectivity, 70% consistency, but 0 accepted proposals (decode error prevents ED ≤ 2 validation); 6 passages >90% identified (3.25× vs random)
- **Phase 75 confirms**: 3-coda model pipeline (6 tracks, 12/23 gates) produces THREE_CODA_NEUTRAL verdict; **both corrections applied corpus-wide** (connector→null + descender→null) — dict-hit 30.2%→37.6% (+7.4pp), but signal 76→62 and bigram z 90.5→71.3 (shorter strings collide more with dictionary); **verbal fraction 57.3%→25.2%** — the central prediction confirmed; **bootstrap grammar null test passes at p=0.0000** (first significant coda-grammar validation in the project — zero of 500 random {n,s,t} shuffles produce a distribution as close to CI-expected); cross-validation doubles 26.2%→54.7%; exhaustive rank shows 6/6 due to code artifact (simplified classifier diverges from full pipeline — real CI distance 0.2951 is actually lower than all 6 permutation scores ≥0.3181); **T1 identifications increase 243→316** (+73, 88 gained vs 15 lost); **3 passages achieve 100% token identification** (project first, enabled by distributional vocabulary at 25.2% coverage); best passage f54r reads as pharmaceutical instruction: `ne · set · bes · cos · cone · se · sera · cone · din · tes · ne · dine · ne · cone · cone`; revalidation FAILS (0/3) — coherence degrades because n_pharma=0 signal words (pharmaceutical vocabulary lost distinctive forms); template selectivity 0.23× (47.7% UNMARKED dilutes grammatical patterns); descender correction is directionally right but trades sequential signal for dict-hit — unlike connector→null which improved everything
- **Phase 76 confirms**: Triple resolution from vocabulary convergence (4 tracks, 7/16 gates) produces NO_PROGRESS verdict; **wildcard propagation constrains 5/13 triples** (3 LIKELY: ascender,crossbar,compound→re, ascender,loop,compound→gu, loop,sigmoid,bench→re) but LOO cross-validation is structurally inapplicable (0/12 — confirmed triples are too frequent; removing one destroys the pattern space, producing 0 observations rather than wrong predictions); clean fraction rises 63%→71.5% with LIKELY triples applied; **10,493 parallel passage pairs** with identical grammatical skeletons prove massive formulaic repetition consistent with pharmaceutical recipe collections; **66,331 diagnostic diffs** and **1,478 substitution tokens** identified; template selectivity 1.00× (templates too permissive — need 4+ elements); **top-blocking triple is `ascender,crossbar,gallows`** (52 frequent types, 1,770 tokens); **LLM gap-filling produces first 2 accepted proposals** — "deinde" (then/next, from decoded "dinede", ED=2) and "rane" (frogs, from decoded "dirane", ED=2) — both pharmaceutically plausible and passing all 5 hallucination control layers, but session KA accuracy 26.7% and selectivity 1.03× undermine confidence; these are tentative leads, not validated identifications

## Assignment Table (T_P15)

25 stroke-feature triples → syllable assignments:
- **12 confirmed** (cross-source validation, Phases 14 + 19.8): these produce the 70 signal words
- **1 Phase 68 RESOLVED** (`ascender,crossbar,compound`: be→de, 3/7 tracks): first computationally resolved change
- **5 Phase 68 LIKELY** (2-track agreement): ga→di, fa→ba, ne→de, la→ce, hi→ba
- **4 remaining landscape-confirmed** (MaxSAT consensus >60%): statistically supported but landscape is flat
- **3 genuinely ambiguous** (no consensus): cover only 164 tokens (0.45% of corpus)

See [docs/signal-vocabulary.md](docs/signal-vocabulary.md) for the full 70-word vocabulary with tables.

## Project Structure

```
voynich_2/
├── We_Know_the_Voice_but_Not_the_Song_v18     # Companion paper (LaTeX)
├── pyproject.toml            # Dependencies, console_scripts (uv)
├── src/voynich/
│   ├── cli.py                # CLI entry point (all commands)
│   ├── core/                 # corpus.py, stats.py, reference.py, ciphers.py, _paths.py
│   ├── analysis/             # Approaches 1-2 (strokes.py, fingerprint.py)
│   ├── visual/               # Image rendering, embedding, similarity, segmentation (63), multi-method comparison (64)
│   └── phases/               # Phase modules (2-78 + reviewer analyses)
├── data/
│   ├── corpus/               # EVA transcription files (IVTFF format)
│   ├── 2Translate/           # Historical source transcriptions
│   └── reference/            # Language corpora by language (not in git)
├── results/                  # JSON output from analyses (not in git)
├── docs/                     # Detailed documentation
│   ├── commands.md           # Complete CLI reference
│   ├── signal-vocabulary.md  # 70 signal words + 22 T1 identifications
│   ├── progression.md        # Phase progression table
│   └── phases/               # Per-phase detailed documentation
└── archive/                  # Deprecated consonant-skeleton approach
```

## Reproducing Results

**Requirements:** Python 3.12+, uv package manager.

```bash
# Install
uv sync
uv pip install -e .

# Place EVA transcription files in data/corpus/
# (ZL3b-n.txt, RF1b-e.txt, IT2a-n.txt from voynich.nu)

# Run individual phases
voynich phase14           # Stroke-feature CSP (the breakthrough)
voynich phase16           # Modifier detection
voynich phase29           # Signal-filtered readability

# Run a full pipeline sequence
voynich phase11 && voynich phase14 && voynich phase15 && voynich phase16
```

Reference corpora (Latin, Italian, German, Occitan medical texts) should be placed in `data/reference/<language>/`. These are not distributed with the repository.

## Progression Summary

| Phase | Dict-Hit | Signal | Bigram z | Key Advance |
|-------|----------|--------|----------|-------------|
| 14 | 19.4% | — | — | Stroke-feature model (breakthrough) |
| 16 | 43.6% (131K) | — | — | Feature model + modifiers |
| 17 | — | — | — | NO-GO: null corpora hit 37.6% |
| 28 | 43.6% | 8 words | — | Signal isolation |
| 29 | 43.6% | — | 6.14 | SIGNAL bigram discovery |
| 36 | 24.1% (10K) | 51 words | 12.66 | 10K dictionary, validated pipeline |
| 42 | 43.6% | — | 14.78 | Z-score audit (conservative minimum) |
| 52 | 40.1% coverage | 22 T1 words | — | Word catalog; 74.8% CI overlap |
| 54 | — | INDETERMINATE | — | Dialect ID: macaronic (Tuscan grammar + northern phonology) |
| 55A | — | SCHINNER_ABOVE | — | Entropy shift extended to 13 mechanisms; Cardan discriminated (+0.49–+0.59); Schinner exposes discriminator scope limit |
| 55B | — | PREDICTION_CONFIRMED_UNIQUE | — | Currier anomaly: Voynich 1.45×, tachygraphic (syl) 1.28×, Schinner 1.04× — tachygraphy uniquely predicts the anomaly |
| 56 | — | COMPATIBLE (10/10) | — | Costamagna 1953 structural match: 21/21 syllables attested, 5 codas = 5 modifier types, 3 shared = 3 ambiguous |
| 57 | 27.5% (CVC) | PASS (5/7), 64 words | 96.19 | CVC coda decode: modifiers as coda consonants; dict_hit drops but bigram z and net signal dramatically increase |
| 58 | — | FAIL (3/8) | — | Costamagna CSP: 22/25 confirmed, 3 ambiguous; no improvement — visual matching essential |
| 59 | 79.9% (segmented) | CVC_VALIDATED (8/11) | — | CVC refinement: attestation artifact fixed (4.3%→79.9%); connector→r; vertical→t; -aiin=Latin declensions (62.3%) |
| 60 | 29.0% (corrected CVC) | 75 words, 4.51× | 87.74 | Corrected CVC: connector→r (+5.5%); i→syllabic (+4.0%); composite 0.94 (#1 strategy); coherence p=0.006 |
| 61 | — | p(CV coh)=0.001 | — | Deep reading (4/5 CI templates); full CV perm (p=0.013 count, p=0.006 CVC coh); zodiac NO_SIGNAL |
| 62 | — | PARTIAL (7/11) | — | Exhaustive pre-visual: Q1=TOKENS_ARE_SYLLABLES; Q2=MINOR_CORRECTIONS (rr/ss geminates); Q3=SIGNIFICANT_STRUCTURE (A/B chi²=222, entropy 0.89) |
| 63 | — | VISUAL_MISMATCH (0/5) | — | Multimodal embedding comparison (font + manuscript vs Costamagna): embedding model lacks stroke-level resolution; segmentation pipeline works (80% line, 73% word); methodology insufficient |
| 64 | — | WEAK_SUPPORT (2/7) | — | Multi-method visual (7 methods): LLM pairwise 56% win; perm p=0.138; graph features best (rank 85.4); y→si best match (25.5); domain gap dominates |
| 65 | — | SEGMENTATION_FAILED (0/4) | — | Word boundary discovery: 4 methods; Latin F1=0.65 (works); Voynich 0–7.5% dict hit (all below EVA baseline 15.4%); decode error rate is bottleneck |
| 66 | — | STRUCTURAL_INSIGHT (5/12) | — | Multi-vector attack: 12 tracks; LLM CONTROLS_DOMINATE (null≈real); Fontana SIMILAR (10 vs 12 families); Hand 4 STRUCTURED (280 bigrams); f116v CRIB_SUPPORTED; hallucination controls essential |
| 67 | 29.2% (voted) | PARTIAL_RESOLUTION (3/5) | 125.86 | Multi-angle triple resolution: 5 tracks; evo +1.3%; distributional 4/4 gates; 8/13 LIKELY, 0 RESOLVED; decode error persists |
| 68 | 30.6% (voted) | PARTIAL_RESOLUTION (5/5) | 112.77 | Rare syllable recovery: 7 tracks; 1 RESOLVED (be→de), 5 LIKELY; +1.57% dict hit; 63% fully decoded; 223 T1 IDs; token-level constraints |
| 69 | 35.9% (clean) | CLEAN_CORE_PARTIAL (4/7) | — | Clean core: coherence p=0.006; segmentation FAILS on clean (EVA 40.6% >> LM 13.2%); 49 paradigms; 888 seq pairs; 3,036 T1-dense passages |
| 70 | 36.2% (expanded) | PHARMACEUTICAL_READING (11/19) | — | Token-as-word: dict NOT bottleneck (+0.3%); coda -s→2sg (100%), -t→3sg (82%); 818 pairs, 50 trigrams; 80% identified passages (2.29× sel) |
| 71 | — | MARGINAL (11/16) | — | Inflectional reverse engineering: coda -r overrepresented (47% of coda tokens); null p=0.26; 342 paradigms (79.5% identified); 90% gram coverage but template sel 0.95×; section chi² p≈0 |
| 72 | 30.2% (null conn) | MODEL_REVISED (16/23) | 90.5 | Decode model diagnosis: connector→null WINS (xval 90.5% vs r 77.9%); coda -s 92.8%, -t 86.5%, -r 16.9%; null_connector model best; T1 expansion 100% FPR beyond Tier B; variable-length = collision artifact |
| 73 | 30.2% (corrected) | CORRECTION_NEUTRAL (15/23) | 90.48 | Corrected pipeline: connector→null applied; dict 29.0%→30.2%, xval 77.9%→90.5%; T1 stable (89.7%, 223→243); PREP roots 2→15; verbal 57.3% unchanged (descender-r dominates); new CVC baseline |
| 74 | 37.6% (desc→null) | DESCENDER_RESOLVED_AND_VOCAB_EXPANDED (12/18) | — | Descender→r ranks 10th/13; 13/15 triples prefer null; verbal 65.1%→24.2%; 94.6% token-final (genuine coda); 675 EVA pattern expansions (898 total); LLM gap-fill KA=40% but 0 accepted (decode error); 6 passages >90% (3.25× sel) |
| 75 | 37.6% (3-coda) | THREE_CODA_NEUTRAL (12/23) | 71.34 | 3-coda pipeline: dict +7.4pp, signal −14, bigram z −19.2; verbal 57.3%→25.2%; bootstrap p=0.0000 (first significant); xval 26.2%→54.7%; T1 243→316; 3 passages at 100% (first); distributional 25.2% coverage |
| 76 | 37.8% (w/ LIKELY) | NO_PROGRESS (7/16) | — | Triple resolution: 5/13 constrained (3 LIKELY), LOO inapplicable; 10,493 parallel pairs; top blocker: ascender,crossbar,gallows (52 types); 2 LLM gap-fills accepted ("deinde"+"rane") but KA 26.7% |
| 77 | — | SELF_CITATION_ELIMINATED (4/4) | — | Timm-Schinner: entropy cosine −0.153 (anticorrelated); MI 1.036× (null level); both tests eliminate; 13 mechanisms tested, tachygraphy sole survivor |
| 78 | — | CVC_T1_SIGNIFICANT (1/3) | — | CVC T1 permutation: 1,000 random tables; real 331 IDs vs null 210±32 (z=3.79, p=0.002); 5 unique words; closes T1 validation caveat |

Full table: [docs/progression.md](docs/progression.md)

## Detailed Documentation

- [Complete CLI Command Reference](docs/commands.md) — all commands grouped by phase
- [Phase-by-Phase Documentation](docs/phases/) — detailed results for all 78 phases
- [Signal Vocabulary Tables](docs/signal-vocabulary.md) — 70 signal words + 316 T1 word-level identifications
- [Progression Table](docs/progression.md) — metrics across all phases

## Citation

```bibtex
@article{ruckman2026voynich,
  title={We Know the Voice but Not the Song: Italian Tachygraphy and the Voynich Manuscript},
  author={Ruckman, Matthew},
  year={2026},
  month={3}
}
```

## License

[MIT](LICENSE)
