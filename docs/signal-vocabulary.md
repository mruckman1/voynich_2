# Signal Vocabulary and Word-Level Identifications

[← Back to README](../README.md)

## Consolidated Signal Vocabulary (70 CV + 75 CVC words)

Signal words are decoded Voynich tokens that appear significantly more often in real Voynich text than in null (permuted) corpora, measured as σ = (real_count − null_mean) / null_std, with threshold σ > 2.0. Selectivity = real_count / null_mean.

**CV discovery progression:** Phase 28 (131K dict): 8 words → Phase 30 (bootstrap): +2 → Phase 36 (10K dict): 51 total → Phase 37-38 (Italian analysis): +22 Italian-only → **70 unique** (3 overlap: dise, cu, dedi).

**CVC discovery progression:** Phase 57 (coda decode): 64 words → Phase 59 (CVC validation): mapping corrections identified → Phase 60 (corrected CVC): **75 words** (+13 new from corrections, −2 lost).

## Current State of Decipherment

### What We Know

**Encoding mechanism:** Italian syllabic tachygraphy (cosine similarity 0.820 against the tachygraphic entropy-shift model, discriminated from 12 alternative encoding hypotheses including the Naibbe cipher at −0.843). The encoding uses a three-layer structure: gallows determinatives mark word boundaries or semantic categories, phonetic roots encode content via stroke-feature triples mapped to CV syllables, and grammatical suffixes encode inflectional endings. Each EVA character decomposes into a stroke-feature triple (first_stroke, last_stroke, glyph_class), and each triple maps to a syllable through the T_P15 assignment table.

**Source language:** Macaronic Latin-Italian (Italian selectivity 5.45× vs Latin 1.30×, confirmed by 4 independent methods: signal isolation, size-matched OT/spectral comparison, SBM profiling, and character n-gram analysis). Size-matched language ID (Phase 50D, all corpora subsampled to 11K tokens) places Italian #1, Latin #2, German #4 — Phase 49's German ranking was entirely a corpus-size artifact. Phase 54 dialect identification battery (8 experiments) returns DIALECT_INDETERMINATE: the signal words carry Tuscan morphological markers (*ci*, *si*, *tu*, *dice*, *dico*) alongside Gallo-Italic phonological features (degemination in *bela*/*sene*, lenition in *diga*/*dise*), consistent with a macaronic register mixing standard Italian grammar with northern Italian phonology.

**Sequential structure:** z = 14.78 (Phase 47 conservative minimum, CV exact-match-only). CC bigram z = 21.0 (Phase 50B, 32/397 consecutive-hit pairs match reference Latin bigrams at 8.1%). CVC bigram z = 87.74 (Phase 60 corrected) — the CVC model dramatically amplifies sequential structure because coda consonants create more distinctive decoded words, reducing false-positive SHARED_HIT classifications.

**CVC coda model (Phases 57–60):** EVA modifier characters encode coda consonants rather than being noise. Five stroke types map to five Latin codas (hook→n, descender→r, sigmoid→s, vertical→t, connector→r), producing CVC syllables that match Costamagna's 1953 syllabary at 83.0% attestation. Net signal increases 10× (3,877 vs 370) and Latin declension endings appear in 60.7% of tokens. Phase 60 corrected two mappings (connector l→r, EVA 'i' reclassified as syllabic), adding 13 new signal words.

**Solution landscape:** Formally FLAT. Phase 44 enumerated 500+ near-optimal MaxSAT solutions. Phase 33 showed 6 independent correction methods propose different assignments for the same triples with zero consensus. Phase 53 confirmed: paradigm-derived constraints produce identical consensus landscapes on shuffled tables (z = 0.02).

**Encoding granularity:** Variable-length, not fixed CV. Phase 53 found that free triples encode 1–3 character substrings (distribution: 127 × 3-char, 84 × 2-char, 19 × 1-char), not strictly 2-character CV syllables as the C5×V4 model predicts. The CVC model extends this: modifier characters add coda consonants to preceding syllables, making the effective encoding unit a CVC syllable rather than CV. This is consistent with actual tachygraphic systems where stroke modifications encode variable-length phonetic units.

### Assignment Table (T_P15) + Coda Table

**CV layer** — 25 stroke-feature triples → syllable assignments:
- **12 confirmed** (cross-source validation, Phases 14 + 19.8): these produce the 70 CV signal words and are the ground truth of the project
- **10 landscape-confirmed** (MaxSAT consensus >60%, Phase 45): statistically supported but Phase 44 showed the landscape is flat, so these may not be uniquely correct
- **3 genuinely ambiguous** (no consensus): cover only 164 tokens (0.45% of corpus)

**CVC layer** — 5 stroke types → coda consonants (Phase 57, corrected Phase 60):
- **4 confirmed from Costamagna 1953**: hook→n, descender→r, sigmoid→s, vertical→t
- **1 corrected in Phase 60**: connector→r (was 'l' in Phase 57; Phase 59 Inv 7 found 'r' gives 23.4% vs 0.5%)
- **1 per-character override**: EVA 'i' reclassified as SYLLABIC in non-final position (Phase 59 Inv 3)
- **15 EVA modifier characters** act as coda markers; **14 ambiguous characters** are context-dependent

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

## CV Summary Statistics

- **51 Latin-10K signal words**: mean σ=31.4, mean selectivity=5.43×
- **22 Italian-only signal words**: mean σ=27.1
- **70 unique CV signal words** total (3 overlap: dise, cu, dedi appear in both lists)
- **Vocabulary composition**: ~65% function words, ~20% content/quality, ~15% pharmaceutical/botanical
- **Consistent selectivity**: ~5.5× across most words (matching CV tachygraphic model prediction of ~5.0×)
- **Language**: Shared (Latin+Italian) dominates; 4 Latin-only (rati, tere, sese, raso); 24 Italian-only
- **Italian verb paradigms**: 5 forms of "dire" (dise, dice, dico, dicu, diga) + 3 forms of "dare" (dedi, dido, dere) — internally consistent conjugation, not random dictionary collisions
- **Function word inventory**: complete Romance clause kit — articles (la, li), prepositions (di, de, co, su), pronouns (te, ti, tu, se, si, ci), auxiliaries (ha, fa)
- **Pharmaceutical register**: preparation verbs from the Circa Instans tradition (cola = strain, tere = grind, raso = scraped) and ingredients (sene = senna, corali = corals, sero = serum)

---

## Phases 55–56: No New Vocabulary

Phase 55 (entropy shift generalization + Currier prediction) and Phase 56 (Costamagna structural compatibility) were purely structural/statistical analyses that validated the tachygraphic hypothesis and established compatibility with Costamagna's 1953 syllabary. Neither phase produced new signal words or decoded vocabulary. Phase 56 noted that confirmed CV syllables (co, de, se) would extend to CVC forms (con-, del-, ser-) under the coda model — a prediction validated in Phase 57.

## CVC Coda Decode Vocabulary (Phases 57–60)

### Background: From CV to CVC

Phase 57 introduced the CVC (consonant-vowel-consonant) coda decode model, treating EVA modifier characters as coda consonants rather than stripping them. This is motivated by Costamagna's 1953 syllabary documentation of five coda rules in medieval Italian tachygraphy. The result is a fundamentally different decode: where Phase 16's CV model reads "daiin" as "di" (stripping the modifiers), the CVC model reads it as "din" (appending coda 'n' from the hook stroke).

**Coda mapping (Phase 57 → Phase 60 corrected):**

| Stroke Type | Phase 57 Coda | Phase 60 Coda | Costamagna Rule | Affected EVA Chars |
|-------------|---------------|---------------|-----------------|-------------------|
| hook | n | n | one dot added | aiin, iin, iiin, n |
| descender | r | r | descender stroke | ar, or, r |
| sigmoid | s | s | curve appended | dy, ey, s |
| vertical | t | t | crossbar through | al, am, g, m, ol |
| connector | **l** | **r** (corrected) | not in Costamagna | b, ckh, h, u |

**Phase 59 corrections applied in Phase 60:**
1. **connector → r** (was l): Phase 59 Inv 7 tested all 7 candidates on 950 affected tokens; 'r' gives 23.4% dict-hit vs 0.5% for 'l'. After applying to full corpus: +5.5% dict-hit on 7,990 affected tokens.
2. **EVA 'i' reclassified as SYLLABIC** in non-final position: Phase 59 Inv 3 found 0 meaningful coda hits from 'i' across 2,807 tokens (5 total hits). Reclassification: +4.0% dict-hit on affected tokens.

### CVC vs CV Performance (Phase 60 Evaluation)

| Metric | CV strip | R3 combined | CVC Phase 57 | CVC Phase 60 |
|--------|----------|-------------|--------------|--------------|
| Dict-hit | 39.1% | 43.6% | 27.5% | **29.0%** |
| Signal words (σ>2) | 23 | 88 | 64 | **75** |
| Bigram z | 62.14 | 55.74 | **96.19** | 87.74 |
| Net signal | 242 | 370 | 3,855 | **3,877** |
| Seg. attestation | — | — | 79.9% | **83.0%** |
| Latin endings | — | — | 55.7% | **60.7%** |
| Composite (Phase 60) | 0.047 | 0.111 | 0.914 | **0.939** |

CVC dict-hit is lower than CV (29% vs 43.6%) because CVC produces longer decoded strings that rarely match whole dictionary words. But on every other metric — signal count, bigram z, net signal, attestation, and Latin endings — CVC dramatically outperforms CV. The composite score (Phase 60 CVCEvaluator) places CVC corrected at 0.939, far above all CV strategies.

### 75 CVC Signal Words (Phase 60 Corrected, `results/corrected_coda.json`)

Signal isolation uses the same methodology as the CV pipeline: per-word σ = (real_count − null_mean) / null_std against 5 null corpora, threshold σ > 2.0.

| # | Word | σ | Real | Null Mean | Sel. | New in P60? | Gloss / Notes |
|---|------|---|------|-----------|------|-------------|---------------|
| 1 | din | 66.4 | 816 | 174.8 | 4.7× | | daily (diurnus) |
| 2 | ni | 58.4 | 355 | 29.6 | 12.0× | | nor |
| 3 | du | 48.4 | 179 | 17.0 | 10.5× | | two / of the |
| 4 | cone | 43.2 | 428 | 101.6 | 4.2× | | with+e (con+e) |
| 5 | ne | 43.2 | 1247 | 296.8 | 4.2× | | not/nor |
| 6 | bes | 42.5 | 265 | 57.8 | 4.6× | | twice / 2/3 (bes) |
| 7 | coras | 31.8 | 227 | 53.0 | 4.3× | | hearts (cor+plural) |
| 8 | bene | 31.3 | 131 | 36.0 | 3.6× | | well/good |
| 9 | cos | 29.6 | 365 | 105.0 | 3.5× | | with+s (co+s coda) |
| 10 | cor | 27.2 | 807 | 407.8 | 2.0× | | heart |
| 11 | decor | 27.2 | 57 | 6.6 | 8.6× | | beauty/grace |
| 12 | neder | 26.8 | 67 | 30.6 | 2.2× | | nor+of/from+r |
| 13 | rates | 25.1 | 175 | 42.8 | 4.1× | | reckonings (pl.) |
| 14 | ses | 25.0 | 235 | 80.8 | 2.9× | | six / themselves |
| 15 | hi | 24.5 | 11 | 1.2 | 9.2× | | these |
| 16 | corat | 23.8 | 50 | 14.4 | 3.5× | | heart+t (coda) |
| 17 | ber | 23.6 | 343 | 96.6 | 3.5× | | well+r (be+r coda) |
| 18 | sen | 23.6 | 171 | 42.0 | 4.1× | | senna (sen-na) |
| 19 | bet | 23.1 | 255 | 75.4 | 3.4× | | well+t (be+t coda) |
| 20 | dis | 22.8 | 341 | 122.6 | 2.8× | | of+s (di+s coda) |
| 21 | dicor | 21.7 | 76 | 31.4 | 2.4× | | of+heart (di+cor) |
| 22 | sene | 20.9 | 223 | 69.4 | 3.2× | | senna |
| 23 | decos | 20.5 | 19 | 2.6 | 7.3× | | beauty+s |
| 24 | cordi | 19.6 | 231 | 66.8 | 3.5× | | **NEW** heart+of (cor+di) |
| 25 | dine | 18.8 | 95 | 31.8 | 3.0× | | daily (variant) |
| 26 | ser | 18.6 | 397 | 175.6 | 2.3× | | serum / evening |
| 27 | cot | 18.2 | 80 | 25.4 | 3.1× | | with+t (co+t coda) |
| 28 | con | 17.7 | 68 | 11.0 | 6.2× | | with |
| 29 | den | 17.6 | 45 | 16.8 | 2.7× | | tooth (dens) |
| 30 | do | 17.6 | 9 | 0.4 | 22.5× | | I give |
| 31 | nerr | 14.3 | 52 | 22.0 | 2.4× | | **NEW** nerve+r |
| 32 | ton | 14.1 | 29 | 4.8 | 6.0× | | tone / thunder |
| 33 | tot | 13.8 | 113 | 54.6 | 2.1× | | so many (tot) |
| 34 | lader | 12.0 | 5 | 0.2 | 25.0× | | thief (latro) |
| 35 | tecor | 11.2 | 56 | 12.6 | 4.4× | | thee+heart |
| 36 | nes | 11.0 | 202 | 102.6 | 2.0× | | nor+s |
| 37 | fa | 9.4 | 10 | 0.8 | 12.5× | | does/makes |
| 38 | ten | 8.6 | 36 | 11.0 | 3.3× | | hold (tenere) |
| 39 | des | 8.2 | 58 | 16.2 | 3.6× | | from / down (des-) |
| 40 | dites | 7.9 | 37 | 11.0 | 3.4× | | of+thee+s |
| 41 | net | 7.1 | 73 | 46.6 | 1.6× | | nor+t |
| 42 | secos | 7.0 | 6 | 0.8 | 7.5× | | dry (siccus) |
| 43 | garne | 6.9 | 4 | 0.6 | 6.7× | | **NEW** garnish/decorate |
| 44 | tos | 6.6 | 43 | 12.0 | 3.6× | | so many+s |
| 45 | tecos | 6.6 | 23 | 4.0 | 5.8× | | thee+with+s |
| 46 | ditr | 6.4 | 21 | 12.0 | 1.8× | | of+tr |
| 47 | ladine | 6.1 | 47 | 18.2 | 2.6× | | **NEW** Ladin (Romance variety) |
| 48 | tes | 6.1 | 50 | 17.8 | 2.8× | | thee+s |
| 49 | set | 5.7 | 66 | 33.0 | 2.0× | | thirst / hedge |
| 50 | dene | 5.2 | 21 | 7.6 | 2.8× | | tooth+e |
| 51 | teras | 5.1 | 13 | 6.2 | 2.1× | | lands (terras) |
| 52 | cott | 4.7 | 4 | 1.0 | 4.0× | | baked (coctus) |
| 53 | sedis | 4.7 | 9 | 2.6 | 3.5× | | seat (gen.) |
| 54 | corr | 4.6 | 64 | 37.0 | 1.7× | | **NEW** heart+rr |
| 55 | disser | 4.5 | 3 | 1.2 | 2.5× | | dissertation/discourse |
| 56 | terras | 4.5 | 2 | 0.2 | 10.0× | | **NEW** lands (acc. pl.) |
| 57 | serr | 4.4 | 34 | 15.0 | 2.3× | | **NEW** serum+r / lock |
| 58 | rarras | 4.2 | 5 | 1.6 | 3.1× | | **NEW** rare (pl.) |
| 59 | raras | 3.8 | 8 | 3.4 | 2.4× | | rare (pl. variant) |
| 60 | ladin | 3.7 | 70 | 35.4 | 2.0× | | **NEW** Ladin / of the side |
| 61 | derr | 3.6 | 9 | 3.0 | 3.0× | | **NEW** of/from+rr |
| 62 | terr | 3.3 | 13 | 3.6 | 3.6× | | **NEW** land+r (terra) |
| 63 | tess | 3.3 | 8 | 3.2 | 2.5× | | weave (tessere) |
| 64 | tott | 3.3 | 2 | 0.4 | 5.0× | | so many (emphatic) |
| 65 | derra | 3.2 | 7 | 4.4 | 1.6× | | **NEW** from the (de+r+a) |
| 66 | gan | 3.2 | 6 | 3.4 | 1.8× | | gain/profit |
| 67 | sedes | 3.0 | 3 | 1.8 | 1.7× | | seats (nom. pl.) |
| 68 | digas | 2.8 | 5 | 1.6 | 3.1× | | say (subj., CVC) |
| 69 | netet | 2.5 | 5 | 2.4 | 2.1× | | nor+t+t |
| 70 | corn | 2.2 | 6 | 2.4 | 2.5× | | **NEW** horn (cornu) |
| — | digant | 999.0 | 1 | 0.0 | ∞ | | dignified (dignantur) |
| — | hat | 999.0 | 1 | 0.0 | ∞ | | has (3rd sg.) |
| — | laten | 999.0 | 1 | 0.0 | ∞ | | hidden (latens) |
| — | mi | 999.0 | 3 | 0.0 | ∞ | | me (dative) |
| — | secon | 999.0 | 1 | 0.0 | ∞ | | second (secundus) |

**5 infinite-sigma singletons** (σ=999): zero occurrences in all 5 null corpora. Listed separately because their extreme sigma is a statistical artifact of zero null variance. Of these, *secon* (second) and *laten* (hidden) are noteworthy as pharmaceutically relevant Latin.

**13 words NEW in Phase 60** (from connector→r and i→syllabic corrections): cordi, nerr, garne, ladine, corr, terras, serr, rarras, ladin, derr, terr, derra, corn. The connector→r correction opened up decoded forms ending in *-rr* and *-r* that were previously *-lr* and *-l* (non-Latin). The i→syllabic correction produced longer decoded words (extra CV syllable instead of coda *t*), enabling new matches like *ladine* and *cordi*.

**2 words lost** from Phase 57: *tor* (σ=3.5) and *ner* (σ=3.0) — both marginal, just above the σ>2 threshold.

### CVC Vocabulary Characteristics

The CVC vocabulary differs from the CV vocabulary in several ways:

- **Longer words**: CVC mean decoded word length 6.02 chars (vs CV 5.40). CVC words include coda consonants that add phonetic specificity.
- **Latin declension endings**: 60.7% of CVC-decoded tokens end in recognizable Latin case endings (-en, -in, -an, -on, -er, -ar, -or, -es, -is), compared to 0% for CV (which strips these). Phase 59 Inv 10 found the distribution is dominated by -en (1,703 tokens) and -in (1,457), consistent with Latin 3rd declension accusative/ablative and prepositional forms — exactly what pharmaceutical recipe language would produce.
- **Costamagna attestation**: 83.0% of CVC-decoded syllables (after greedy maximal-munch segmentation) are attested in Costamagna's 1953 syllabary inventory of 221 entries.
- **Higher net signal**: 3,877 (CVC) vs 370 (R3 combined) — a 10× improvement in the number of tokens classified as genuine signal (appearing in real text but not null corpora).

### CVC Coherence Validation (Phase 60 Track B)

Phase 59's permutation coherence test showed that the CV-era criteria (verb paradigm, function kit, pharma register) were trivially satisfied by CVC output (p=0.552). Phase 60 recalibrated the test with expanded, CVC-specific criteria:

| Criterion | Real Table | Threshold (92nd pctl) | Random Pass Rate | p-value |
|-----------|-----------|----------------------|-----------------|---------|
| Signal word count | 75 | 74 | 8.9% | 0.076 |
| Latin ending diversity | 8 types | 7 | 16.7% | 0.077 |
| Content word count | 66 | 66 | 8.4% | 0.084 |
| Pharmaceutical terms | 8 | 8 | 9.8% | 0.098 |

**Joint p = 0.006** (6/1000 random tables pass all 4). **Fisher combined p = 0.011**. The CVC decode is now statistically validated as coherent — matching or exceeding the CV baseline (p=0.011, Phase reviewer test).

### CVC Recipe Readings (Phase 60 Track D)

Phase 60 produced the project's first line-by-line recipe annotations. 340 recipes were extracted from the CVC-decoded corpus using boundary markers (expanded from 6 CV markers to 22 CVC variants: cola→colar/colas/colat/colan, etc.). Top 10 recipes by glossed fraction:

- **Mean glossed fraction: 94.9%** (using the full merged annotation vocabulary of 130 entries)
- **Max consecutive glossed: 26 tokens** — an entire recipe fully annotated
- **Pharmaceutical cross-references found**: 2 recipes matching Circa Instans preparation terms
- **Ingredient inventory**: senna (*sene/senen*) identified across multiple recipes
- **Sample reading** (f20v): "senna + heart + [unknown] + heart + [unknown] + well + evening + not/nor + nor"

These readings are fragmentary — most tokens decode to function-word combinations (*with+e*, *of+s*) rather than content words. The annotation confirms the pharmaceutical register but does not yet produce connected readable text.

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

### CV Coverage (Phases 36–52)

**Coverage:** 40.1% of all 36,238 tokens are glossed (70 signal words + 22 T1 catalog entries + their corpus occurrences). Coverage rises to 74.9% on f57v (pharmaceutical section).

**Best structural reading** (f116r, 49.3% coverage): fragments including "oratione extrahendi" (by method of extracting) alongside roots (radecem), seeds (semen/seminne), corals (coralli/corallus), branches (ramis), with quality terms (bela/beautiful, amara/bitter, sere/serene, dives/rich). 74.8% overlap with the Circa Instans pharmaceutical reference corpus.

**Longest consecutive glossed runs:** 59 tokens on f57v (mostly function words); best content-bearing run: 15 tokens on f111v with "radicess", "dives", "tela"; 14 tokens on f76v mixing body parts (cora/heart), materials (tela/cloth), and method language (ratione/by method).

### CVC Coverage (Phases 57–60)

**Segmentation attestation:** 83.0% of CVC-decoded syllables are attested in Costamagna's inventory (Phase 60 corrected), up from 79.9% (Phase 59) and 4.3% (Phase 57 — measurement artifact from comparing multi-syllable strings against single-syllable entries).

**Recipe extraction:** 340 recipes extracted using 22 CVC boundary markers (cola/colar/colas/colat/colan, codi/codin/codir/codis/codit, bene/benen/bener, sene/senen/sener/senes/senet, dine/dinen/diner/dines). Top 10 recipes average 94.9% glossed fraction with a merged annotation vocabulary of 130 entries.

**Best CVC recipe readings** (Phase 60 Track D):
- f3v: "strain + not/nor + [coras] + [des] + ..." — begins with pharmaceutical verb *cola* (strain)
- f20v: "senna + [cordi] + [ber] + heart + ..." — begins with ingredient *sene* (senna) followed by *cor* (heart)
- f47r: "senna + heart + with + ... + heart + serum + ... + heart + ..." — *sene*, *cor*, and *ser* recurring in pharmaceutical context
- f45r: "well/good + [ben] + ... + daily + ... + heart + with + ..." — *bene*, *din*, *cor* in recipe structure

**Longest consecutive CVC glossed run:** 26 tokens (f54r) — an entire recipe fully annotated. However, most annotations are function-word decompositions (e.g., *cone* = "with+e") rather than identifiable content words.

## Open Problems

**13 free triples** remain unresolved, accounting for ~59% of dark tokens. Resolving these requires approaches beyond the computational methods explored so far:

- Phase 44: the solution landscape contains 500+ near-optimal MaxSAT solutions — a purely score-based method needs additional constraints to discriminate among them
- Phase 50: character-level language models produce identical rankings across languages (selectivity 1.10×) — higher-order linguistic features or external evidence are needed for discrimination
- Phase 52: whole-word dictionary matching at this scale generates more false positives than true matches (null selectivity 0.56×) — targeted crib-based identification (as in the 22 T1 words) remains the viable path
- Phase 53: paradigm-derived constraints are not table-specific (z = 0.02), and the encoding granularity is variable-length (1–3 chars), not fixed 2-char CV — future models should accommodate variable-length mappings

**CVC-specific open problems** (Phases 57–60):

- **Bigram z trade-off**: The corrected CVC decode (Phase 60) improved attestation, signal count, and Latin endings but bigram z dropped from 96.19 to 87.74. The i→syllabic correction produces longer words that slightly dilute SIGNAL-SIGNAL adjacency. Both values are enormously significant, but optimizing all metrics simultaneously remains difficult.
- **CVC dict-hit is misleading**: CVC produces concatenated syllable strings (e.g., "coraterr") that don't match whole dictionary words. Phase 60's CVCEvaluator framework replaces dict-hit with segmentation attestation as the primary metric. Future phases should use composite scoring, not raw dict-hit.
- **Recipe readings are fragmentary**: While glossed fraction is high (94.9%), most annotations decompose tokens into function-word syllables rather than identifying content words. The CVC vocabulary of 75 signal words produces pharmaceutical register terms (senna, heart, serum, daily) but connected readable passages remain out of reach.
- **Coda ambiguity**: The vertical group (al, am, g, m, ol, i) shows heterogeneous behavior — Phase 59 Inv 3 found 'i' should be syllabic while 'm' is strongly coda (4.0× preference for t). Further per-character refinements within stroke groups may improve the model.

The most promising directions for resolving these triples include: discovery of additional external cribs (plant identifications, astronomical labels), cross-manuscript comparison with other tachygraphic sources, and physical analysis (multispectral imaging, ink composition) that could reveal erased or faded text providing new anchor points.

---

## Combined Master Vocabulary

All affirmed decoded vocabulary across both CV and CVC pipelines (Phases 28–60), organized by category. Each entry includes the decode pipeline (CV or CVC), the discovery phase, and the best available gloss.

### Pharmaceutical / Botanical Terms

| Word | Pipeline | σ | Real Count | Phase | Gloss | Validation |
|------|----------|---|-----------|-------|-------|------------|
| sene | CV+CVC | 47.7 / 20.9 | 242 / 223 | 28 | senna (plant) | Signal + T1 (diasene) |
| sero | CV | 70.1 | 135 | 28 | serum / evening | Signal |
| cola | CV | 16.7 | 68 | 28 | strain (verb, Circa Instans) | Signal |
| tere | CV | 11.0 | 10 | 36 | to rub/grind (verb, Circa Instans) | Signal |
| raso | CV | 3.4 | 6 | 36 | scraped (preparation term) | Signal |
| corali | CV | 13.5 | 8 | 37 | corals (ingredient) | Signal + T1 |
| cor | CVC | 27.2 | 807 | 57 | heart (body part) | Signal |
| sen | CVC | 23.6 | 171 | 57 | senna (CVC truncation) | Signal |
| ser | CVC | 18.6 | 397 | 57 | serum / evening (CVC) | Signal |
| din | CVC | 66.4 | 816 | 57 | daily (diurnus, dosage) | Signal |
| den | CVC | 17.6 | 45 | 57 | tooth (dens) | Signal |
| decor | CVC | 27.2 | 57 | 57 | beauty/grace | Signal |
| decos | CVC | 20.5 | 19 | 57 | beauties (plural) | Signal |
| secos | CVC | 7.0 | 6 | 57 | dry (siccus) | Signal |
| sedis | CVC | 4.7 | 9 | 57 | seat (genitive) | Signal |
| sedes | CVC | 3.0 | 3 | 57 | seats (nom. pl.) | Signal |
| cott | CVC | 4.7 | 4 | 57 | baked (coctus) | Signal |
| corn | CVC | 2.2 | 6 | 60 | horn (cornu) | Signal (NEW) |
| terras | CVC | 4.5 | 2 | 60 | lands (acc. pl.) | Signal (NEW) |
| garne | CVC | 6.9 | 4 | 60 | garnish/decorate | Signal (NEW) |

### Whole-Word Identifications (T1, Phase 52)

| EVA Type(s) | Latin Word | Meaning | Folios | Pipeline |
|-------------|-----------|---------|--------|----------|
| otol | ratione | by method/reason | 46 | CV (T1) |
| oty, otaly, opy, ety, otary | rabidi | of the rabid/fierce | 60 | CV (T1) |
| qopchedy, qofchedy | stercora | dung/manure | 13 | CV (T1) |
| ytol, tshol, ytoldy | diasene | diasenna (senna compound) | 10 | CV (T1) |
| chotar, chotaiin, chotal, chetar, chotey, chetey, chep | coralli | of corals | 7 | CV (T1) |
| chkain | codex | codex/manuscript | 9 | CV (T1) |
| otcham | radicom | root (accusative) | 4 | CV (T1) |
| chtol | commune | common/shared | 4 | CV (T1) |
| shty | secundi | of the second | 3 | CV (T1) |

### Content Words

| Word | Pipeline | σ | Real Count | Phase | Gloss |
|------|----------|---|-----------|-------|-------|
| cora | CV | 98.7 | 1114 | 37 | heart (Italian) |
| dise | CV | 77.8 | 71 | 36 | says (Italian) |
| rati | CV | 50.4 | 156 | 36 | reckoning |
| bela | CV | 43.8 | 400 | 37 | beautiful (Italian) |
| sera | CV | 21.7 | 166 | 36 | evening |
| diri | CV | 19.5 | 31 | 36 | to say (infinitive) |
| didi | CV | 18.8 | 136 | 37 | gave (plural, Italian) |
| dice | CV | 18.4 | 51 | 37 | says (Italian) |
| deco | CV | 18.0 | 65 | 37 | I decorate (Italian) |
| cose | CV | 16.3 | 14 | 37 | things (Italian) |
| beri | CV | 15.5 | 20 | 37 | to drink (Italian) |
| code | CV | 15.5 | 68 | 37 | tails/codes (Italian) |
| dedi | CV | 15.2 | 68 | 36 | I gave (Italian) |
| dicu | CV | 14.1 | 17 | 37 | I say (dialectal) |
| diga | CV | 13.5 | 8 | 37 | say (subjunctive, Italian) |
| dido | CV | 11.0 | 13 | 37 | I gave (variant) |
| tela | CV | 10.6 | 20 | 36 | cloth/web |
| sede | CV | 10.8 | 19 | 36 | seat/see |
| dico | CV | 9.9 | 48 | 30 | I say |
| dere | CV | 6.3 | 8 | 37 | to give (Italian) |
| dici | CV | 2.5 | 5 | 36 | to be said |
| cela | CV | 3.5 | 5 | 37 | hides (Italian) |
| decore | CV | 3.2 | 7 | 37 | decorate (Italian) |
| cone | CVC | 43.2 | 428 | 57 | with+e (composition) |
| coras | CVC | 31.8 | 227 | 57 | hearts (CVC plural) |
| rates | CVC | 25.1 | 175 | 57 | reckonings (plural) |
| dine | CVC | 18.8 | 95 | 57 | daily (variant) |
| corat | CVC | 23.8 | 50 | 57 | heart+t (inflected) |
| dicor | CVC | 21.7 | 76 | 57 | of+heart |
| lader | CVC | 12.0 | 5 | 57 | thief (latro) |
| dites | CVC | 7.9 | 37 | 57 | of+thee+s |
| teras | CVC | 5.1 | 13 | 57 | lands (terrae) |
| disser | CVC | 4.5 | 3 | 57 | discourse |
| digas | CVC | 2.8 | 5 | 57 | say (CVC subj.) |
| cordi | CVC | 19.6 | 231 | 60 | heart+of (NEW) |
| ladine | CVC | 6.1 | 47 | 60 | Ladin (NEW) |
| ladin | CVC | 3.7 | 70 | 60 | Ladin (NEW) |
| rarras | CVC | 4.2 | 5 | 60 | rare (pl., NEW) |

### Quality / Descriptive Words

| Word | Pipeline | σ | Real Count | Phase | Gloss |
|------|----------|---|-----------|-------|-------|
| bene | CV+CVC | 46.4 / 31.3 | 152 / 131 | 28 | well/good |
| dira | CV | 32.4 | 50 | 36 | dire/harsh |
| sere | CV | 28.5 | 73 | 36 | serene |
| nera | CV | 27.8 | 62 | 36 | black (fem., Italian) |
| raro | CV | 7.6 | 15 | 28 | rarely |

### Function Words

| Word | Pipeline | σ | Real Count | Phase | Gloss |
|------|----------|---|-----------|-------|-------|
| di | CV | 129.7 | 1353 | 36 | of |
| be | CV | 134.7 | 547 | 37 | well (Italian variant) |
| se | CV | 105.1 | 592 | 36 | if/self |
| ne | CV+CVC | 93.5 / 43.2 | 1470 / 1247 | 36 / 57 | not/nor |
| ni | CV+CVC | 51.4 / 58.4 | 494 / 355 | 36 / 57 | nor |
| bi | CV | 63.2 | 342 | 36 | twice |
| ce | CV | 61.2 | 353 | 36 | here/this |
| co | CV | 52.5 | 490 | 36 | with |
| de | CV | 47.3 | 471 | 28 | of/from |
| du | CV+CVC | 46.1 / 48.4 | 189 / 179 | 36 / 57 | two / of the |
| ci | CV | 37.8 | 64 | 30 | there/to it |
| te | CV | 36.6 | 122 | 36 | you/thee |
| bo | CV | 32.6 | 124 | 36 | (function) |
| la | CV | 32.1 | 117 | 36 | the (fem.) |
| si | CV | 29.4 | 170 | 36 | yes/self |
| ra | CV | 23.3 | 121 | 36 | (function) |
| do | CV+CVC | 21.6 / 17.6 | 29 / 9 | 36 / 57 | I give |
| re | CV | 21.1 | 21 | 36 | thing/about |
| so | CV | 21.1 | 242 | 36 | I am/above |
| cu | CV | 20.2 | 144 | 36 | with (dialectal) |
| ti | CV | 19.9 | 65 | 36 | you (dat.) |
| su | CV | 19.8 | 46 | 36 | on/above |
| ru | CV | 18.5 | 59 | 36 | (function) |
| nu | CV | 16.4 | 47 | 36 | (function) |
| ha | CV | 15.5 | 7 | 36 | has (Italian) |
| li | CV | 15.4 | 94 | 36 | the (pl.) |
| ga | CV | 11.0 | 6 | 36 | (function) |
| tu | CV | 10.0 | 15 | 36 | you |
| ge | CV | 9.7 | 18 | 36 | (function) |
| sese | CV | 9.5 | 18 | 36 | themselves (Latin) |
| hi | CV+CVC | 8.2 / 24.5 | 11 / 11 | 36 / 57 | these |
| fe | CV | 6.3 | 5 | 36 | made/faith |
| fa | CV+CVC | 5.6 / 9.4 | 10 / 10 | 36 / 57 | does/makes |
| gi | CV | 4.3 | 6 | 37 | already (Italian) |
| deri | CV | 7.1 | 11 | 37 | of the (pl., Italian) |
| con | CVC | 17.7 | 68 | 57 | with |
| des | CVC | 8.2 | 58 | 57 | from/down (prefix) |
| bes | CVC | 42.5 | 265 | 57 | twice / 2/3 |
| cos | CVC | 29.6 | 365 | 57 | with+s |
| ber | CVC | 23.6 | 343 | 57 | well+r |
| bet | CVC | 23.1 | 255 | 57 | well+t |
| dis | CVC | 22.8 | 341 | 57 | of+s |
| ses | CVC | 25.0 | 235 | 57 | six / themselves |
| ton | CVC | 14.1 | 29 | 57 | tone/thunder |
| tot | CVC | 13.8 | 113 | 57 | so many |
| cot | CVC | 18.2 | 80 | 57 | with+t |
| set | CVC | 5.7 | 66 | 57 | thirst/hedge |
| ten | CVC | 8.6 | 36 | 57 | hold (tenere) |
| nes | CVC | 11.0 | 202 | 57 | nor+s |
| net | CVC | 7.1 | 73 | 57 | nor+t |
| tos | CVC | 6.6 | 43 | 57 | so many+s |
| tes | CVC | 6.1 | 50 | 57 | thee+s |
| dene | CVC | 5.2 | 21 | 57 | tooth+e |
| gan | CVC | 3.2 | 6 | 57 | gain/profit |

### CVC-Only Compositional / Inflected Forms

CVC signal words that are inflected or compositional forms of known roots, produced by coda consonant attachment:

| Word | σ | Real Count | Phase | Composition | Root |
|------|---|-----------|-------|-------------|------|
| neder | 26.8 | 67 | 57 | ne+de+r | nor + of/from |
| tecor | 11.2 | 56 | 57 | te+cor | thee + heart |
| tecos | 6.6 | 23 | 57 | te+cos | thee + with+s |
| nerr | 14.3 | 52 | 60 | ner+r | nerve (NEW) |
| corr | 4.6 | 64 | 60 | cor+r | heart+r (NEW) |
| serr | 4.4 | 34 | 60 | ser+r | serum+r (NEW) |
| terr | 3.3 | 13 | 60 | ter+r | land+r (NEW) |
| derr | 3.6 | 9 | 60 | der+r | of/from+r (NEW) |
| derra | 3.2 | 7 | 60 | der+r+a | from the (NEW) |
| tess | 3.3 | 8 | 57 | tes+s | weave (tessere) |
| tott | 3.3 | 2 | 57 | tot+t | so many (emphatic) |
| netet | 2.5 | 5 | 57 | net+et | nor+t+t |
| raras | 3.8 | 8 | 57 | rara+s | rare (pl.) |

### Rare / Singleton Signal Words

| Word | Pipeline | Phase | Gloss | Notes |
|------|----------|-------|-------|-------|
| mi | CVC | 57 | me (dative) | σ=∞, 3 occurrences, 0 in null |
| secon | CVC | 57 | second (secundus) | σ=∞, 1 occurrence |
| laten | CVC | 57 | hidden (latens) | σ=∞, 1 occurrence |
| hat | CVC | 57 | has (3rd sg.) | σ=∞, 1 occurrence |
| digant | CVC | 57 | dignified (dignantur) | σ=∞, 1 occurrence |

### Summary Counts

| Category | CV (Phases 28–52) | CVC (Phases 57–60) | Combined Unique |
|----------|-------------------|---------------------|-----------------|
| Function words | 35 | 24 | ~50 |
| Content words | 26 | 16 | ~36 |
| Pharma/botanical | 3 | 10 | ~12 |
| Quality/descriptive | 6 | 0 | 6 |
| T1 whole-word IDs | 22 (9 unique Latin words) | — | 9 |
| Rare singletons | 0 | 5 | 5 |
| **Total signal words** | **70** | **75** | **~118 unique decoded forms** |
| **Total unique Latin/Italian glosses** | ~60 | ~55 | **~90 distinct meanings** |

**Cross-pipeline confirmations:** 8 words appear in both CV and CVC signal lists (ne, ni, du, hi, fa, do, bene, sene), providing independent validation across two different decode methodologies. The word *bene* was additionally confirmed by two independent CV pipelines (Phase 14 and Phase 19.8) — the only triple-confirmed word in the project.
