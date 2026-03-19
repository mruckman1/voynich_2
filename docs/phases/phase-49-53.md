# Phases 49-53: Novel Computational Approaches & Final Consolidation

[← Phases 43-48](phase-43-48.md) | [Phase Index](README.md) | [Next: Phase 54 →](phase-54.md)

**Phase 49:** IMPROVEMENT (50.4% dict-hit via external LM + ED1, but selectivity only 1.12x)
**Phase 50:** INSUFFICIENT_SIGNAL (any random table + ED1 ~ 28%; CC z=21.0 real; Italian #1 size-matched)
**Phase 51:** BOTH_PASS (suffix map partial z=4.23, bridge marginal z=4.57)
**Phase 52:** CATALOG_MARGINAL (22 T1 words, 40.1% coverage, 74.8% CI overlap)
**Phase 53:** CONSTRAINTS_FOUND_NO_CONSENSUS (230 constraints, z=0.02; variable-length encoding confirmed)

---

## Phase 49: Novel Computational Approaches

**Verdict: IMPROVEMENT** -- 7/8 validations pass. External LM breaks the flat landscape.

### Track A: External LM Lattice Decode

Char-level 5-gram and word-level 3-gram LMs trained on reference corpora (Latin 73K, Italian 11K, Occitan 48K, German 149K tokens). Per-token lattice expanded with ED1 variants against 10K dictionary. Per-folio beam search (width 10).

- **50.4% dict-hit on 10K** (up from 24.2% Phase 16 baseline)
- 28 CC bigrams, 63 consecutive hits (f116v)
- **But**: 98.6% of dict hits come from ED1-selected words
- Selectivity over random lattice: only **1.12x** (z=56)
- Char LM is the dominant scorer (alpha=0.67); word LM adds little

**Source decomposition**: 12,754 unchanged (35.2%), 5,481 from lattice alternatives (15.1%), **18,003 from ED1 expansion (49.7%)**.

**Short-word inflation**: 67.1% of dict hits are 3-letter words. Random 3-letter string has 68.2% chance of ED1 neighbor in 10K dictionary.

### Track B: Fourier/Spectral Periodicity

224/226 folios show periodic structure at 2.0-2.7 token periods -- ubiquitous, non-discriminating. Reflects trivial word-frequency alternation present in any natural language. Spectral clustering silhouette = 0.043.

### Tracks C-D: Optimal Transport and Spectral Graph Language ID

Both identify **German as structurally closest** via Sinkhorn OT, Gromov-Wasserstein, Laplacian eigenvalue spectra, and NetLSD. Unexpected -- likely reflects German's 2x corpus size advantage and monosyllabic decoded vocabulary character.

### Track E: RL Assessment

LM reward landscape is SHARP (dynamic range 4.96, variance 5.90). REINFORCE on 500-token sample: +7.0%. Opens multi-swap optimization path.

## Phase 50: WFST Validation, Word-Level LM Rescoring, Extended Null Battery

**Verdict: INSUFFICIENT_SIGNAL** -- 3/6 validations pass.

### Track A: Permuted-Table Null (THE CRITICAL TEST)

| Condition | Dict-Hit | Selectivity | z-score |
|-----------|----------|-------------|---------|
| Real table (T_P15) | 30.26% | -- | -- |
| Partial permutation (50 trials) | 29.56% +/- 1.22% | **1.02x** | 0.58 |
| Full permutation (50 trials) | 27.50% +/- 2.23% | **1.10x** | 1.24 |

**The ED1+charLM pipeline produces ~28-30% dict-hit regardless of what table you use.** The specific table adds only ~2 percentage points. ED1 approach invalidated as decipherment tool.

### Track B: Word-Level LM Rescoring

Restrict ED1 to words length >=4: dict-hit drops from 50.4% to **32.78%** (3-letter inflation = ~18pp). Word-level Viterbi adds **exactly zero** improvement. But CC bigram z=**21.0** -- real sequential structure confirmed. Scramble selectivity = 1.000 (word LM provides no sequential benefit).

### Track C: Extended Null Battery -- 1/5 pass

| Test | Result | Verdict |
|------|--------|---------|
| C.1: Wrong-Language LM | Latin=Italian=German=Occitan, all 30.22% | **FAIL** |
| C.2: Length-Matched Random | Random words: 97.6% vs real 30.2% | **FAIL** |
| C.3: Section-Specific | r=-0.67 (word length confound) | **FAIL** |
| C.4: Cross-Validated LM | CV ratio 1.006 | **PASS** |
| C.5: Bigram Attribution | 0 DIRECT CC bigrams, all from ED1 | **FAIL** |

### Track D: Size-Matched Language ID

All corpora subsampled to 11K tokens:

| Method | #1 | #2 | #3 | #4 |
|--------|----|----|----|----|
| Gromov-Wasserstein | Latin | Italian | Occitan | German |
| NetLSD Spectral | German | Occitan | Italian | Latin |
| Char N-Gram Profile | **Italian** | Latin | Occitan | German |
| **Consensus (Borda)** | **Italian (2.00)** | **Latin (2.33)** | Occitan (2.67) | German (3.00) |

**Italian ranks #1, German drops to last.** Phase 49's German ranking was entirely corpus-size artifact.

### Two Genuine Findings
1. **CC bigram z=21.0** -- real sequential structure (8.1% of consecutive-hit pairs match Latin bigrams)
2. **Italian #1** in size-matched language ID -- consistent with Phase 19 and Phase 46

## Phase 51: Reverse Suffix Calibration + Concatenation Bridge Search

**Overall Verdict: BOTH_PASS** -- 5/7 validations pass.

### Track A: Reverse Suffix Calibration -- SUFFIX_MAP_PARTIAL

70 signal words matched 11,228 tokens. 14 EVA suffixes calibrated to Latin endings. Null z=**4.23** (p<0.0001), selectivity 1.16x. But 5-fold CV accuracy only **6.9%** -- mapping too noisy and many-to-many.

POS coverage: 67.0%. Distribution: NOUN_NOM_F1 (23.7%), NOUN_GEN_M2 (12.0%), PARTICLE (7.2%).

Section profiles show structural variation: Biological highest NOUN_NOM_F1 (37.5%, descriptive anatomy), Recipes highest PARTICLE (8.4%, imperative-heavy).

### Track B: Concatenation Bridge Search -- BRIDGE_MARGINAL

Coverage: 93.6% of dark tokens have >=1 confirmed-triple character. **70,870 bridge matches** across 1,092 unique words. Null z=**4.57**, selectivity 1.44x.

High-confidence matches: `otol`->"ratione" (46 folios), `oty`->"rabidi" (60 folios), `qopchedy`->"stercora" (13 folios), `ytol`->"diasene" (10 folios), `chotar`->"coralli" (7 folios).

9 free triples received implied assignments but **0 reached strong consensus** (>50% agreement). The key triple `ascender,crossbar,gallows` received 903 observations scattered across incompatible syllables.

## Phase 52: Word-Level Identification Catalog

**Overall Verdict: CATALOG_MARGINAL** -- 5/7 validations pass. Gate FAIL.

### 22 Tier-1 Identifications

| EVA Type | Latin Word | Gloss | Folios | Confidence |
|----------|-----------|-------|--------|------------|
| `otol` | **ratione** | by method/reason | 46 | 0.980 |
| `oty` | **rabidi** | of the fierce | 60 | 0.960 |
| `qopchedy` | **stercora** | dung (medicinal) | 13 | 1.000 |
| `ytol` | **diasene** | diasenna compound | 10 | 0.930 |
| `chotar` | **coralli** | of corals | 7 | 0.880 |
| `chkain` | **codex** | codex/manuscript | 9 | 0.860 |
| `otcham` | **radicom** | root (acc.) | 4 | 0.770 |
| `chtol` | **commune** | common/shared | 4 | 0.770 |
| `shty` | **secundi** | of the second | 3 | 0.700 |

6 EVA types map to "coralli" (variant spellings). 5 to "rabidi". 3 to "diasene".

### Validation

- **Null test**: selectivity **0.56x** (FAIL) -- shuffled assignments produce MORE matches
- **56 morphological paradigms** found (radic- 7 forms, semin- 7 forms, codic- 6 forms, decoct- 5 forms)
- Signal adjacency enrichment: z=5.33 (significant)
- **74.8% Circa Instans overlap**
- 40.1% corpus coverage

### Structural Reading

Best content-bearing runs:
- f76v (14 tokens): "be cora tela sero bela cora bi cora ti cora ratione rabidi si rati"
- f111v (15 tokens): "so radiciss denns ra ne ne ne dives bi cora tela bela so bela ce"
- f116r (49.3% coverage): fragments including "oratione extrahendi" alongside roots, seeds, corals, branches

## Phase 53: Paradigm-Constrained Free Triple Resolution

**Overall Verdict: CONSTRAINTS_FOUND_NO_CONSENSUS** -- 3/7 validations pass. Gate FAIL.

230 constraints extracted from 15 paradigms across 5 of 13 free triples. No triple reached consensus (>0.5). Null test z=0.02 -- paradigm-derived constraints produce identical consensus on shuffled tables.

### Per-Triple Constraint Landscape

| Triple | Current | Top Implied | Consensus | N Obs | Recommendation |
|--------|---------|-------------|-----------|-------|----------------|
| `loop,tail,bench` | la | cis | 0.11 | 142 | NO_CONSENSUS |
| `ascender,loop,compound` | to | ra | 0.45 | 44 | NO_CONSENSUS |
| `ascender,crossbar,gallows` | te | r | 0.32 | 38 | NO_CONSENSUS |
| `loop,sigmoid,bench` | ne | rvi | 0.25 | 4 | NO_CONSENSUS |
| `ascender,plume,gallows` | ga | de | 1.00 | 2 | INSUFFICIENT |

`loop,tail,bench` is most revealing: 142 observations from 13 stems, yet consensus only 11%. Implied values wildly dispersed: "cis"(15), "ce"(15), "ces"(12), "cum"(12), "cem"(10), "ci"(10) plus 18 others. Different Latin case endings demand mutually exclusive outputs at the same position.

### Key Findings

1. **Ventris method fails because Latin morphology is incompatible with one-triple-one-syllable model.** Different inflectional endings demand mutually exclusive values from the same triple.
2. **Null test (z=0.02) proves constraints are not table-specific.** Shuffled tables produce equally convergent constraint landscapes.
3. **Variable-length encoding confirmed**: free triples encode 1-3 character substrings (55% three-char, 36% two-char, 8% one-char), not strictly 2-char CV syllables.
4. **8 of 13 free triples completely unconstrained** by paradigm evidence.

### Final Progression

| Phase | Dict Hit | Selectivity | Key Advance |
|-------|----------|-------------|-------------|
| Phase 16 | 43.6% (131K) | 3.38x | Feature model + modifiers |
| Phase 29 | 43.6% | -- | z=6.14 SIGNAL bigram discovery |
| Phase 36 | 24.1% (10K) | 1.31x | 51 signal words, validated pipeline |
| Phase 42 | 43.6% | -- | z=14.78 conservative minimum |
| Phase 44 | -- | -- | MaxSAT landscape FLAT |
| Phase 50A | 30.3% (ED1) | 1.10x | ED1 approach invalidated |
| Phase 50B | -- | 1.00x | CC z=21.0 real; word LM adds nothing |
| Phase 50D | -- | -- | Italian #1 size-matched |
| Phase 51A | -- | z=4.23 | Suffix map PARTIAL |
| Phase 51B | -- | z=4.57 | Bridge MARGINAL |
| Phase 52 | 40.1% cov | 0.56x | 22 T1 words, 56 paradigms |
| Phase 53 | -- | z=0.02 | Paradigm constraints not table-specific |

---

[← Phases 43-48](phase-43-48.md) | [Phase Index](README.md) | [Next: Phase 54 →](phase-54.md)
