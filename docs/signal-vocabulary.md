# Signal Vocabulary and Word-Level Identifications

[← Back to README](../README.md)

## Consolidated Signal Vocabulary (70 unique words)

Signal words are decoded Voynich tokens that appear significantly more often in real Voynich text than in null (permuted) corpora, measured as σ = (real_count − null_mean) / null_std, with threshold σ > 2.0. Selectivity = real_count / null_mean.

**Discovery progression:** Phase 28 (131K dict): 8 words → Phase 30 (bootstrap): +2 → Phase 36 (10K dict): 51 total → Phase 37-38 (Italian analysis): +22 Italian-only → **70 unique** (3 overlap: dise, cu, dedi).

## Current State of Decipherment

### What We Know

**Encoding mechanism:** Italian syllabic tachygraphy (cosine similarity 0.820 against the tachygraphic entropy-shift model, discriminated from 12 alternative encoding hypotheses including the Naibbe cipher at −0.843). The encoding uses a three-layer structure: gallows determinatives mark word boundaries or semantic categories, phonetic roots encode content via stroke-feature triples mapped to CV syllables, and grammatical suffixes encode inflectional endings. Each EVA character decomposes into a stroke-feature triple (first_stroke, last_stroke, glyph_class), and each triple maps to a syllable through the T_P15 assignment table.

**Source language:** Macaronic Latin-Italian (Italian selectivity 5.45× vs Latin 1.30×, confirmed by 4 independent methods: signal isolation, size-matched OT/spectral comparison, SBM profiling, and character n-gram analysis). Size-matched language ID (Phase 50D, all corpora subsampled to 11K tokens) places Italian #1, Latin #2, German #4 — Phase 49's German ranking was entirely a corpus-size artifact. Phase 54 dialect identification battery (8 experiments) returns DIALECT_INDETERMINATE: the signal words carry Tuscan morphological markers (*ci*, *si*, *tu*, *dice*, *dico*) alongside Gallo-Italic phonological features (degemination in *bela*/*sene*, lenition in *diga*/*dise*), consistent with a macaronic register mixing standard Italian grammar with northern Italian phonology.

**Sequential structure:** z = 14.78 (Phase 47 conservative minimum, exact-match-only). CC bigram z = 21.0 (Phase 50B, 32/397 consecutive-hit pairs match reference Latin bigrams at 8.1%). The decoded text contains genuine word-level sequential structure that matches Latin phrase patterns.

**Solution landscape:** Formally FLAT. Phase 44 enumerated 500+ near-optimal MaxSAT solutions. Phase 33 showed 6 independent correction methods propose different assignments for the same triples with zero consensus. Phase 53 confirmed: paradigm-derived constraints produce identical consensus landscapes on shuffled tables (z = 0.02).

**Encoding granularity:** Variable-length, not fixed CV. Phase 53 found that free triples encode 1–3 character substrings (distribution: 127 × 3-char, 84 × 2-char, 19 × 1-char), not strictly 2-character CV syllables as the C5×V4 model predicts. This is consistent with actual tachygraphic systems where stroke modifications encode variable-length phonetic units.

### Assignment Table (T_P15)

25 stroke-feature triples → syllable assignments:
- **12 confirmed** (cross-source validation, Phases 14 + 19.8): these produce the 70 signal words and are the ground truth of the project
- **10 landscape-confirmed** (MaxSAT consensus >60%, Phase 45): statistically supported but Phase 44 showed the landscape is flat, so these may not be uniquely correct
- **3 genuinely ambiguous** (no consensus): cover only 164 tokens (0.45% of corpus)

## 51 Latin-10K Signal Words (Phase 36, `results/signal_10k.json`)

| # | Word | σ | Real | Null Mean | Sel. | Language | Phase | Type |
|---|------|---|------|-----------|------|----------|-------|------|
| 1 | di | 129.71 | 1353 | 241.4 | 5.60× | Shared | 36 | function (of) |
| 2 | se | 105.12 | 592 | 108.0 | 5.48× | Shared | 36 | function (if/self) |
| 3 | ne | 93.52 | 1470 | 268.0 | 5.49× | Shared | 36 | function (not/nor) |
| 4 | dise | 77.77 | 71 | 12.8 | 5.55× | Italian-only | 36 | content (says) |
| 5 | sero | 70.12 | 135 | 22.8 | 5.92× | Shared | 28 | pharm. (serum/evening) |
| 6 | bi | 63.23 | 342 | 63.8 | 5.36× | Shared | 36 | function (twice) |
| 7 | ce | 61.19 | 353 | 66.0 | 5.35× | Shared | 36 | function (here/this) |
| 8 | co | 52.53 | 490 | 86.4 | 5.67× | Shared | 36 | function (with) |
| 9 | ni | 51.38 | 494 | 90.2 | 5.48× | Shared | 36 | function (nor) |
| 10 | rati | 50.44 | 156 | 26.8 | 5.82× | Latin | 36 | content (reckoning) |
| 11 | sene | 47.71 | 242 | 47.4 | 5.11× | Shared | 28 | botanical (senna) |
| 12 | de | 47.34 | 471 | 91.6 | 5.14× | Shared | 28 | function (of/from) |
| 13 | bene | 46.41 | 152 | 25.4 | 5.98× | Shared | 28 | quality (well/good) |
| 14 | du | 46.10 | 189 | 39.2 | 4.82× | Shared | 36 | function (two/of the) |
| 15 | ci | 37.82 | 64 | 7.4 | 8.65× | Shared | 30 | function (there/to it) |
| 16 | te | 36.57 | 122 | 22.8 | 5.35× | Shared | 36 | function (you/thee) |
| 17 | bo | 32.57 | 124 | 21.0 | 5.90× | Shared | 36 | function |
| 18 | dira | 32.41 | 50 | 12.2 | 4.10× | Shared | 36 | quality (dire/harsh) |
| 19 | la | 32.06 | 117 | 23.2 | 5.04× | Shared | 36 | function (the, fem.) |
| 20 | si | 29.44 | 170 | 32.4 | 5.25× | Shared | 36 | function (yes/self) |
| 21 | sere | 28.53 | 73 | 14.8 | 4.93× | Shared | 36 | quality (serene) |
| 22 | nera | 27.82 | 62 | 10.4 | 5.96× | Italian-only | 36 | quality (black, fem.) |
| 23 | ra | 23.28 | 121 | 21.8 | 5.55× | Shared | 36 | function |
| 24 | sera | 21.69 | 166 | 32.6 | 5.09× | Shared | 36 | content (evening) |
| 25 | do | 21.61 | 29 | 3.8 | 7.63× | Shared | 36 | function (I give) |
| 26 | re | 21.11 | 21 | 5.2 | 4.04× | Shared | 36 | function (thing/about) |
| 27 | so | 21.07 | 242 | 43.0 | 5.63× | Shared | 36 | function (I am/above) |
| 28 | cu | 20.19 | 144 | 28.8 | 5.00× | Italian-only | 36 | function |
| 29 | ti | 19.95 | 65 | 13.6 | 4.78× | Shared | 36 | function (you, dat.) |
| 30 | su | 19.75 | 46 | 9.8 | 4.69× | Shared | 36 | function (on/above) |
| 31 | diri | 19.46 | 31 | 4.6 | 6.74× | Italian-only | 36 | content (to say, inf.) |
| 32 | ru | 18.47 | 59 | 11.4 | 5.18× | Shared | 36 | function |
| 33 | cola | 16.73 | 68 | 12.0 | 5.67× | Shared | 28 | pharm. (strain, v.) |
| 34 | nu | 16.39 | 47 | 7.4 | 6.35× | Shared | 36 | function |
| 35 | ha | 15.50 | 7 | 0.8 | 8.75× | Shared | 36 | function (has, It.) |
| 36 | li | 15.45 | 94 | 14.0 | 6.71× | Shared | 36 | function (the, pl.) |
| 37 | dedi | 15.20 | 68 | 16.6 | 4.10× | Italian-only | 36 | content (I gave) |
| 38 | ga | 11.02 | 6 | 0.6 | 10.00× | Shared | 36 | function |
| 39 | tere | 10.96 | 10 | 1.8 | 5.56× | Latin | 36 | content (to rub) |
| 40 | sede | 10.76 | 19 | 4.4 | 4.32× | Shared | 36 | content (seat/see) |
| 41 | tela | 10.61 | 20 | 5.0 | 4.00× | Shared | 36 | content (cloth/web) |
| 42 | tu | 10.03 | 15 | 1.4 | 10.71× | Shared | 36 | function (you) |
| 43 | dico | 9.88 | 48 | 7.8 | 6.15× | Shared | 30 | content (I say) |
| 44 | ge | 9.66 | 18 | 3.8 | 4.74× | Shared | 36 | function |
| 45 | sese | 9.50 | 18 | 2.8 | 6.43× | Latin | 36 | function (themselves) |
| 46 | hi | 8.22 | 11 | 2.0 | 5.50× | Shared | 36 | function (these) |
| 47 | raro | 7.62 | 15 | 3.6 | 4.17× | Shared | 28 | quality (rarely) |
| 48 | fe | 6.32 | 5 | 1.0 | 5.00× | Shared | 36 | function (made/faith) |
| 49 | fa | 5.58 | 10 | 1.8 | 5.56× | Shared | 36 | function (does/makes) |
| 50 | raso | 3.39 | 6 | 1.4 | 4.29× | Latin | 36 | content (scraped) |
| 51 | dici | 2.51 | 5 | 1.6 | 3.12× | Shared | 36 | content (to be said) |

## 22 Italian-Only Signal Words (Phase 37-38, `results/italian_signal.json`)

| # | Word | σ | Real Count | Phase | Gloss |
|---|------|---|-----------|-------|-------|
| 1 | be | 134.65 | 547 | 37 | well (It. variant) |
| 2 | cora | 98.68 | 1114 | 37 | heart |
| 3 | dise | 77.77 | 71 | 36/37 | says |
| 4 | bela | 43.75 | 400 | 37 | beautiful |
| 5 | cedi | 23.48 | 24 | 37 | yield |
| 6 | cu | 20.19 | 144 | 36/37 | with (dialectal) |
| 7 | didi | 18.82 | 136 | 37 | gave (pl.) |
| 8 | dice | 18.44 | 51 | 37 | says |
| 9 | deco | 17.98 | 65 | 37 | I decorate |
| 10 | cose | 16.30 | 14 | 37 | things |
| 11 | beri | 15.52 | 20 | 37 | to drink |
| 12 | code | 15.46 | 68 | 37 | tails/codes |
| 13 | dedi | 15.20 | 68 | 36/37 | I gave |
| 14 | dicu | 14.12 | 17 | 37 | I say (dialectal) |
| 15 | corali | 13.47 | 8 | 37 | corals |
| 16 | diga | 13.47 | 8 | 37 | say (subj.) |
| 17 | dido | 11.02 | 13 | 37 | I gave (var.) |
| 18 | deri | 7.12 | 11 | 37 | of the (pl.) |
| 19 | dere | 6.28 | 8 | 37 | to give |
| 20 | gi | 4.31 | 6 | 37 | already |
| 21 | cela | 3.53 | 5 | 37 | hides |
| 22 | decore | 3.25 | 7 | 37 | decorate |

## Summary Statistics

- **51 Latin-10K signal words**: mean σ=31.4, mean selectivity=5.43×
- **22 Italian-only signal words**: mean σ=27.1
- **70 unique signal words** total (3 overlap: dise, cu, dedi appear in both lists)
- **Vocabulary composition**: ~65% function words, ~20% content/quality, ~15% pharmaceutical/botanical
- **Consistent selectivity**: ~5.5× across most words (matching CV tachygraphic model prediction of ~5.0×)
- **Language**: Shared (Latin+Italian) dominates; 4 Latin-only (rati, tere, sese, raso); 24 Italian-only
- **Italian verb paradigms**: 5 forms of "dire" (dise, dice, dico, dicu, diga) + 3 forms of "dare" (dedi, dido, dere) — internally consistent conjugation, not random dictionary collisions
- **Function word inventory**: complete Romance clause kit — articles (la, li), prepositions (di, de, co, su), pronouns (te, ti, tu, se, si, ci), auxiliaries (ha, fa)
- **Pharmaceutical register**: preparation verbs from the Circa Instans tradition (cola = strain, tere = grind, raso = scraped) and ingredients (sene = senna, corali = corals, sero = serum)

## 22 Word-Level Identifications (Phase 52 T1)

Whole-word identifications from the bridge search: EVA token types whose partial-decode pattern uniquely matches exactly one pharmaceutical dictionary word, recurring on 3+ independent folios. These are Ventris-style identifications — the EVA→word mapping is established without resolving individual character values for the free triples.

| EVA Type | Latin Word | Meaning | Folios | Corpus Freq |
|----------|-----------|---------|--------|-------------|
| otol | ratione | by method/reason | 46 | 74 |
| oty | rabidi | of the rabid/fierce | 60 | 111 |
| qopchedy | stercora | dung/manure | 13 | 30 |
| otaly | rabidi | of the rabid/fierce | 10 | 19 |
| ytol | diasene | diasenna (senna compound) | 10 | 15 |
| chotar | coralli | of corals | 7 | 10 |
| chotaiin | coralli | of corals | 7 | 9 |
| chkain | codex | codex/manuscript | 9 | 12 |
| chotal | coralli | of corals | 5 | 7 |
| opy | rabidi | of the rabid/fierce | 5 | 8 |
| tshol | diasene | diasenna | 6 | 6 |
| chetar | coralli | of corals | 5 | 6 |
| ety | rabidi | of the rabid/fierce | 5 | 6 |
| chotey | coralli | of corals | 4 | 8 |
| otcham | radicom | root (acc.) | 4 | 5 |
| chtol | commune | common/shared | 4 | 5 |
| ytoldy | diasene | diasenna | 4 | 4 |
| otary | rabidi | of the rabid/fierce | 4 | 6 |
| chetey | coralli | of corals | 3 | 7 |
| shty | secundi | of the second | 3 | 4 |
| chep | coralli | of corals | 3 | 4 |
| qofchedy | stercora | dung/manure | 5 | 8 |

6 different EVA types all map to "coralli" (variant spellings of the same word), 5 to "rabidi", 3 to "diasene". This morphological consistency across variant spellings is expected in a medieval manuscript and would be extraordinary if coincidental. These identifications survive the null test because their uniqueness (exactly one dictionary word matches each pattern) is immune to the dictionary-inflation problem that invalidated the 2,435 T2 identifications (null selectivity 0.56×).

## 56 Morphological Paradigms (Phase 52)

Different EVA token types mapping to different inflected forms of the same Latin stem — the hallmark of a genuine alphabetic encoding of inflected Latin:

- **radic-** stem (7 case forms from 20+ EVA types): radice / radicem / radices / radici / radicibus / radicis / radicum — "root" in full Latin declension
- **semin-** stem (7 forms): semin / semina / semine / seminem / semines / semini / seminis — "seed" in full declension
- **codic-** stem (6 forms): codice / codicem / codices... — "codex" declension
- **decoct-/dicoct-** stems (5 forms each): decocta / decocti / decocto... — "decoction"
- **secund-** stem (4 forms): secundi / secundo / secundum / secundus — "second/following"
- **divers-** stem (3 forms): diversas / diversi / diversis — "diverse/various"

Signal adjacency z = 5.33 — catalog words cluster near signal words significantly more than random tokens.

## Corpus Coverage and Structural Readings

**Coverage:** 40.1% of all 36,238 tokens are glossed (70 signal words + 22 T1 catalog entries + their corpus occurrences). Coverage rises to 74.9% on f57v (pharmaceutical section).

**Best structural reading** (f116r, 49.3% coverage): fragments including "oratione extrahendi" (by method of extracting) alongside roots (radecem), seeds (semen/seminne), corals (coralli/corallus), branches (ramis), with quality terms (bela/beautiful, amara/bitter, sere/serene, dives/rich). 74.8% overlap with the Circa Instans pharmaceutical reference corpus.

**Longest consecutive glossed runs:** 59 tokens on f57v (mostly function words); best content-bearing run: 15 tokens on f111v with "radicess", "dives", "tela"; 14 tokens on f76v mixing body parts (cora/heart), materials (tela/cloth), and method language (ratione/by method).

## Open Problems

**13 free triples** remain unresolved, accounting for ~59% of dark tokens. Resolving these requires approaches beyond the computational methods explored so far:

- Phase 44: the solution landscape contains 500+ near-optimal MaxSAT solutions — a purely score-based method needs additional constraints to discriminate among them
- Phase 50: character-level language models produce identical rankings across languages (selectivity 1.10×) — higher-order linguistic features or external evidence are needed for discrimination
- Phase 52: whole-word dictionary matching at this scale generates more false positives than true matches (null selectivity 0.56×) — targeted crib-based identification (as in the 22 T1 words) remains the viable path
- Phase 53: paradigm-derived constraints are not table-specific (z = 0.02), and the encoding granularity is variable-length (1–3 chars), not fixed 2-char CV — future models should accommodate variable-length mappings

The most promising directions for resolving these triples include: discovery of additional external cribs (plant identifications, astronomical labels), cross-manuscript comparison with other tachygraphic sources, and physical analysis (multispectral imaging, ink composition) that could reveal erased or faded text providing new anchor points.
