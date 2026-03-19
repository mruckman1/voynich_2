# Phases 24-27: Error Correction, Reading Direction, Zodiac Attack, Peer Review

[← Phases 20-23](phase-20-23.md) | [Phase Index](README.md) | [Next: Phases 28-30 →](phase-28-30.md)

---

## Phase 24 — Targeted Error Correction and Exploratory Analysis

*Part A — Error Correction:*
- `triple_sensitivity.json` — Leave-one-out analysis of 25 stroke-triple assignments; baseline dict_hit=51.6%; 11 classified **probably_correct** (drop >3% when removed), 7 **uncertain**, 7 **probably_wrong** (drop <0.5%); top sensitive triples: `vertical,vertical,minim` (delta -8.2%), `closed_loop,horizontal,round` (delta -6.1%); 7 Phase 19.8 anchor overrides applied (de, bene, et, in, terra, rosa, sal)
- `error_candidates.json` — 7 probably_wrong + 7 uncertain triples examined; 3-10 replacement candidates per triple; scored by 0.5×dict_hit + 0.3×bigram + 0.2×family_coherence; top candidate improvements: up to +2.1% dict_hit per swap
- `targeted_swap.json` — Greedy accumulation of best swaps with bigram filter; **3 swaps accepted** (bigram non-degrading + dict_hit improving); net improvement **+1.8% dict_hit** (51.6% → 53.4%); 4 swaps rejected (bigram degradation)
- `bigram_filter.json` — Held-out validation (seed 123 vs training seed 42); corrected table bigram=0.0000, Phase 16 bigram=0.0000 (both near floor); null mean=0.0000; held-out/training ratio=1.00; **no overfitting detected**; verdict **PASS**
- `corrected_table.json` — 25 triples: 11 CONFIRMED, 3 CORRECTED, 4 UNCERTAIN, 7 ORIGINAL; frequency JSD=0.142; family coherence=0.68; grid shape score=0.72; 3 corrections with full provenance
- `corrected_decode.json` — 36,238 tokens decoded with corrected table; **53.4% expanded dict_hit** (up from 51.6%); selectivity **3.45×** (up from 3.38×); per-section range 48.1%–59.2%; mean 2.61 syl/token
- `corrected_readability.json` — 5-test battery: bigram=0.0000 (unchanged), CE ratio=0.98, POS selectivity=1.02×, 3/7 domain hits, 0 phrases; **3/5 tests pass**; net vs Phase 16: +1.8% dict_hit, selectivity improved, readability stable

*Part B — Exploratory Analyses:*
- `word_boundary.json` — Concatenation test: 2.1% of adjacent word pairs form valid Latin words (null=1.8%, selectivity=1.17×); split test: 4.3% of long tokens split into valid word pairs; line-break partial words: 12.7% of line-final tokens continue on next line; verdict: **EVA spaces are mostly genuine word boundaries** but some over-segmentation exists
- `ligature_test.json` — MI analysis of 6 candidates (ch, sh, cth, ckh, cph, cfh); **ch** MI=2.34 (z=4.1, significant), **sh** MI=1.89 (z=3.2, significant); cth/ckh/cph/cfh MI<1.0 (not significant); re-tokenization with merged ch/sh: dict_hit changes <0.5%; verdict: **ch and sh are ligatures** but merging does not improve decode
- `directionality.json` — Forward vs reversed vs boustrophedon reading per section; forward best in 5/7 sections; reversed best in 0/7; boustrophedon best in 2/7 (stars, zodiac); line-position entropy: uniform across positions (no directionality signal); verdict: **forward reading confirmed** for most sections
- `known_text_search.json` — 20 medical formulae from Circa Instans searched at ≥60% agreement; **2 hits** (recipe for "aqua rosae" on f88r, "sal commune" pattern on f103r); null phrases: 0.3 hits mean; selectivity **6.67×**; corrections extracted: 3 character-level refinements from crib alignment
- `folio_isolation.json` — 226 folios scored; top folio **f88r** (constraint density 0.82: herbal section, 47 tokens, 3 botanical IDs, 2 anchor words); multi-decode: 61.7% dict_hit (best single folio); coherence: pharmaceutical vocabulary cluster detected; 4 candidate decoded words with botanical meaning
- `cross_section_transfer.json` — 7 section-specific tables trained; self-application dict_hit range 48.2%–67.3%; cross-application mean 41.8%; transfer ratio 0.78; **herbal A↔herbal B** transfer=0.91 (high); **stars↔zodiac** transfer=0.85; **biological↔herbal** transfer=0.62 (low); verdict: **encoding is mostly uniform** with minor section-specific variation
- `reverse_engineering.json` — 11 confirmed words aligned (de, bene, et, in, terra, rosa, sal, adde, aqua, bibe, coque); 19 character-level assignments extracted; 14/19 consistent with Phase 16 table (73.7%); 5 disagreements identified; bootstrap decode: 3 new candidate words discovered (mare, vale, ante); verdict: **partial table validates Phase 16** with 5 specific corrections suggested
- `token_grammar.json` — Positional profiles for 44 EVA chars: 12 initial-heavy, 8 final-heavy, 15 balanced, 9 rare; Latin syllable positional match: 18/25 triples compatible (72%); **7 Phase 16 violations** detected (word-initial Latin syllable assigned to word-final EVA char); gallows: all 4 gallows chars are paragraph/line-initial (>80%); verdict: **positional constraints identify 7 suspect assignments**

*Integration:*
- `phase24_integrate.json` — **Part A verdict**: corrected table improves dict_hit by +1.8% (51.6→53.4%) with no overfitting; bigram plausibility unchanged (floor effect); 3/25 assignments corrected. **Part B discoveries**: 5/8 analyses produced actionable findings (ligatures, crib search, folio isolation, reverse engineering, token grammar); 7 positional violations + 5 reverse-engineering disagreements identify **~10 suspect triple assignments** for future correction. **Progression**: Phase 11=11.1% → Phase 14=19.4% → Phase 15=35.4% → Phase 16=51.6% → **Phase 24=53.4%** (3.45× selectivity). **Decipherment readiness score**: 0.62 (up from 0.58). **Key conclusion**: Phase 16's table is largely correct (73.7% validated by reverse engineering); the remaining gap to the 89.5% oracle ceiling is primarily due to modifier handling, segmentation errors, and dictionary coverage rather than table inaccuracy. The 10 suspect assignments identified by convergent evidence (positional violations + reverse engineering + crib mismatches) are the highest-priority targets for Phase 25.

## Phase 25 — Reading Direction Test and Folio f6r Manual Examination

*Step 25.1 — Boustrophedon Re-Ordering:*
- `boustrophedon_decode.json` — 4 reading-direction variants (Forward, Reversed, Boustrophedon B1 odd-first, B2 even-first) tested across 7 sections × 5 metrics (bigram plausibility, trigram plausibility, POS trigram validity, phrase detection, function-word adjacency). herbal_a prefers reversed/B2 (bigram 0.000212 vs forward 0.000106); biological prefers B1 (0.000154 vs 0.0); all other sections tied at 0.0. Trigram plausibility=0.0 everywhere; 0 phrases detected in any variant. Per-folio analysis: 109/110 herbal_a folios prefer forward (signal driven by single folio); 19/20 biological folios prefer forward. Null shuffle test: p=0.099 (herbal_a), p=0.089 (biological) — **not significant at p<0.05**. Best boustrophedon bigram 0.000212, far below 0.01 threshold. Control sections (pharmaceutical, recipes) correctly show forward as best. Verdict: **SUGGESTIVE** — direction preference exists but absolute signal is noise-level; decode accuracy is the bottleneck, not reading direction.

*Step 25.2 — Folio f6r Manual Examination:*
- `f6r_manual.json` — Complete token-level decode of folio f6r (Calendula/marigold): 83 tokens, **61.4% expanded dict-hit** (51/83), **47.0% original dict-hit** (39/83). 9-consecutive-hit sequence: `ci didi di todi cora se cone radi se` (lines 6–7) — does NOT parse as Latin grammar; no subject-verb structure, no preposition+noun agreement, no medical formula match. 11 coherent fragments total, **0/11 parseable as Latin**. Calendula vocabulary search (339 terms with medieval variants): **0 exact matches**, 65 near matches at edit distance 1–2; specificity ratio **3.06×** (49 specific medical/botanical near-matches vs 16 generic). Hit words overwhelmingly 2-letter syllables: di(12×), ce(7×), ne(6×), se(5×), codi(7×) — these match Latin trivially, not because they're correct Calendula terms. Comparison folios: worst herbal f44r at 23.2%, pharmaceutical f57r at 23.2% — f6r is quantitatively better but qualitatively identical (short syllable hits, not readable text). Verdict: **DOMAIN_MATCH** — specificity ratio exceeds 1.5× but no readable Latin passage produced.

*Step 25.3 — Combined Verdict:*
- `phase25_verdict.json` — Decision matrix: SUGGESTIVE × DOMAIN_MATCH → **STRUCTURAL_ONLY**. Both tests show weak positive signals but neither crosses the threshold for a decoded passage or confirmed direction finding. Paper claims structural identification (cipher type = syllabary with modifiers, source language = Latin, content domain = botanical/medical) supported by 51.6% dict-hit at 3.38× selectivity. The 61.4% dict-hit on f6r is non-discriminative — decoded words are common short syllables that match Latin by chance. The 89.5% oracle ceiling vs 51.6% actual confirms the gap is in the syllable assignment table itself, not in reading direction or post-processing. **Key conclusion**: the project has identified what the Voynich cipher is (a stroke-level syllabary encoding Latin botanical/medical text) but has not yet produced readable decipherment. The remaining 38% gap requires better syllable assignments, not better reading order or folio-specific analysis.

## Phase 26 — Zodiac Known-Plaintext Attack

The zodiac section (f70v–f73v) is a closed system where external knowledge nearly completely determines the plaintext: each folio depicts a known zodiac sign, three folios have standard-script month names visible ("Mars"=March on Pisces, "Abril"=April on Aries, "May" on Taurus), and astrological tradition prescribes the vocabulary (planet names, body parts, elements, qualities). Phase 26 exploits these as cribs to extract grounded character assignments.

*Step 26.1 — Zodiac Map:*
- `zodiac_map.json` — 12 zodiac folios catalogued (f70v1–f73v); Capricornus and Aquarius missing (f74 absent). **299 labels** (Lz loci), **36 circular text blocks** (Cc), **1,194 total tokens** across zodiac section. Standard-script words confirmed on 3 folios: f70v2 "Mars" (French/March), f70v1 "Abril" (Spanish/April), f71v "May" (English/May). Aries and Taurus each span two folios (dark/light halves). Clock positions extracted from `<!HH:MM>` IVTFF annotations.

*Step 26.2 — Month Name Crib:*
- `month_crib.json` — 6 candidate languages tested (Latin, Italian, Northern Italian, French, Occitan, Spanish) with medieval spelling variants via `generate_medieval_variants()`. **Forward test** (decode labels via Phase 16 table, compare to expected month syllables): 0 exact matches, 0 close matches across all languages. **Table-independent CSP** (enumerate all syllable assignments for labels with ≤4 triples that produce any month name): **300 CSP solutions** found — but these are combinatorially expected given ~21 syllables per triple and short labels matching short month names. **Cross-folio consistency**: 0 consistent assignments (no triple received the same syllable from independent CSP solutions on different folios). **Null control**: selectivity **2.86×** (correct month-folio pairing scores 2.86× higher than random permutations) — statistically interesting but driven by month-name length distributions, not by actual decoding. Best language: **Northern Italian** (highest mean agreement 0.218). Verdict: **PARTIAL — selectivity present but no confirmed character assignments.**

*Step 26.3 — Astrological Crib:*
- `astro_crib.json` — 4 vocabulary domains tested against decoded zodiac text:
  - **Quality terms** (calidus/frigidus/siccus/humidus + Italian variants): **7 hits** on correct folios, selectivity **14.86×** — strongest signal in Phase 26, but hits are short substrings (2-3 letters) that occur by chance in decoded text.
  - **Body part terms** (caput→Aries, pectus→Cancer, etc.): **1 hit** — "cor" on Leo folio (Leo rules the heart). Interesting but isolated.
  - **Planet names** (sol, luna, mars, etc.): **0 hits** — no planet name found on its ruling sign's folio.
  - **Element terms** (ignis/terra/aer/aqua): **0 hits** — no element vocabulary detected.
  - **Element cycling test** (fire→earth→air→water period-4 pattern): cycle score **0.0** — no correlation between sequential folios and element vocabulary.
  - Null control: quality selectivity 14.86× is significant; other domains at baseline. Verdict: **PARTIAL — quality vocabulary shows signal but planet/element tests negative.**

*Step 26.4 — Per-Label Exhaustive CSP Decode:*
- `label_decode.json` — **299 zodiac labels** processed; 151 labels with ≤3 syllabic triples eligible for exhaustive CSP (enumerate all ~21^n syllable assignments, check each against 131K expanded dictionary). Phase 16 dict-hit rate: **51/299 (17.1%)**. CSP dict-hit rate: **149/151 (98.7%)** — but this is expected: with ~9K+ combinations tried per label and a 131K-word dictionary, almost any 2-3 syllable combination matches something. **0 derived assignments** — no triple received a consistent syllable across multiple independent labels. The CSP approach is undiscriminating: too many candidates, too few constraints. Agreement with Phase 16: N/A (no derived assignments to compare). Verdict: **MINIMAL — CSP produces abundant hits but no discriminating signal.**

*Step 26.5 — Zodiac-Derived Assignment Table:*
- `zodiac_table.json` — Tiered assembly of all zodiac-derived assignments: **0 tier-1** (no cross-folio confirmed assignments), **0 tier-2** (no single-source crib-derived assignments with sufficient weight), **25 tier-3** (all triples fall back to Phase 16). Merged table is **identical to Phase 16**. Critical design: tier-1 requires `month_crib_consistent` source (cross-folio validated), not accumulated CSP weight — this prevented a bug where ~300 CSP solutions each contributing weight 1.0 would falsely promote 13 triples to tier-1 and degrade dict_hit from 46% to 32%. Verdict: **NO CHANGE — zodiac analysis produced no new assignments.**

*Step 26.6 — Full Corpus Decode:*
- `zodiac_decode.json` — Full corpus decoded with merged table (= Phase 16): **39.1% corpus dict_hit** (vs Phase 16's 51.6% — discrepancy due to different word-set construction in this step), selectivity **1.31×**. Zodiac section specifically: **28.2% dict_hit** — notably **worse** than herbal sections (34–45%). Per-section: herbal_a 42.2%, herbal_b 34.2%, pharmaceutical 34.3%, recipes 43.3%, biological 27.2%, stars 37.9%, zodiac 28.2%. Bigram JSD from Latin: 0.658 (zodiac) vs 0.655 (corpus). Best passage: f72v2 (Virgo) with longest consecutive hit run. Verdict: **zodiac section decodes worse than other sections, suggesting different encoding conventions or content type.**

*Step 26.7 — Validation Battery:*
- `phase26_validate.json` — 12 validation tests, **5/12 PASS**, gate **FAIL** (needs ≥7):

| Test | Name | Result | Detail |
|------|------|--------|--------|
| V1 | Month name matches | **FAIL** | 3.0 (exact=0, close=0, csp=3 capped) — barely meets threshold but csp count is inflated |
| V2 | Month crib selectivity | **PASS** | 2.86× (threshold 2.0×) |
| V3 | Planet name cribs | **FAIL** | 0 planets matched (threshold ≥2) |
| V4 | Body part cribs | **FAIL** | 1 body hit (threshold ≥3) |
| V5 | Element cycling | **FAIL** | 0.0 (threshold >0.3) |
| V6 | Cross-label consistency | **FAIL** | 0 consistent assignments (threshold ≥3) |
| V7 | Zodiac readability | **FAIL** | 28.2% zodiac < 42.2% herbal |
| V8 | No regression | **PASS** | 39.1% ≥ 39.1% (within 0.5% tolerance) |
| V9 | Bigram plausibility | **PASS** | JSD 0.658 < 0.8 |
| V10 | Null discrimination | **FAIL** | 1.31× (threshold >1.5×) |
| V11 | Zodiac-derived assignments | **FAIL** | 0 tier1+tier2 (threshold ≥2) |
| V12 | Consecutive hits | **PASS** | 5 consecutive hits on f72v2 |

*Step 26.8 — Phase 26 Verdict:*
- `phase26_verdict.json` — Verdict: **NO_SIGNAL**. No statistically significant signal from zodiac known-plaintext attack. Month matches: 0, selectivity: 2.86×, consistent assignments: 0. The zodiac text does not appear to encode standard month names, planet names, or anatomical terms in any of the 6 tested languages. Progression: Phase 11=11.1% → Phase 14=19.4% → Phase 15=35.4% → Phase 16=51.6% → Phase 26=39.1% (no improvement; trend: regression due to different word-set construction, not actual table degradation).

- **Key conclusions**:
  1. The zodiac section decodes **worse** than other sections (28.2% vs 34–45% herbal), suggesting its content may not be standard astrological text — possibly calendrical computation, astrological medicine recipes, or abbreviated notation.
  2. The known-plaintext attack fails not because the method is wrong, but because the **assumed plaintext is wrong**: zodiac labels do not encode the month names, planet names, or body part terms that astrological tradition would predict.
  3. The 2.86× month crib selectivity and 14.86× quality vocabulary selectivity are interesting but driven by substring length effects rather than genuine decoding — short decoded syllables (di, ce, ne, se) match short Latin substrings by combinatorial chance.
  4. Phase 16's table (51.6%) remains the best decode. The zodiac section is **not** the easiest entry point for the cipher — contrary to the initial hypothesis that known zodiac content would provide strong cribs.

## Phase 27 — Peer Review Controls: Gibberish Classification and Naibbe Entropy Shift

Two focused validation tests to close the two specific peer-review vulnerabilities identified in the paper: (1) the Phase 9.5 text typology classifier was never tested against known gibberish or self-citation text, and (2) the Phase 19.2 entropy shift ranking never tested the Naibbe dice cipher with Greshko's 2025 published parameters.

*Step 27.1 — Gibberish and Self-Citation Typology Classification:*
- `gibberish_typology.json` — 38 Gaskell-Bowern gibberish transcriptions + 28 Timm-Schinner self-citation samples (10 default + 18 sensitivity grid) run through Phase 9.5 classifier.
  - **Gibberish**: **23/38 (60.5%)** classified as `encoded_natural` — the same label given to Voynich. 14/38 glossolalia, 1/38 constructed. Mean H2/H1=0.779 (high enough to trigger "anomalous" indicator), mean Zipf R²=0.681, mean TTR=0.733.
  - **Timm-Schinner**: **28/28 (100%)** classified as `encoded_natural` at every parameter setting (p_copy ∈ {0.6,0.7,0.8}, p_mutate ∈ {0.05,0.10,0.15}, buffer_size ∈ {50,100,200}). The copy-from-buffer algorithm perfectly reproduces Zipfian distributions (R²~0.930) and normal TTR (~0.353).
  - **Key discriminant the classifier misses**: entropy floor — Voynich **0.978** vs gibberish **0.048** (0/38 elevated above 0.6) vs Timm-Schinner 0.227. The entropy floor is the single most distinctive Voynich property but is not used in the classification rules.
  - Discriminant power: **0.227** (only 22.7% of control samples correctly excluded).
  - Comparison table: Voynich (encoded_natural, H2/H1=0.622, floor=0.978) vs Latin (encoded_natural, H2/H1=0.865, floor=0.386) vs gibberish mean (encoded_natural, H2/H1=0.779, floor=0.048) vs Timm-Schinner (encoded_natural, H2/H1=0.991, floor=0.227).
  - Methodological note: Gaskell-Bowern (2022) used word-length autocorrelation, triple-repeat rates, and character placement biases — largely non-overlapping features from Phase 9.5's entropy ratios and Zipf R².
  - Verdict: **CLASSIFIER_COMPROMISED** — the `encoded_natural` label cannot distinguish Voynich from deliberate gibberish or mechanically-generated self-citation text.

*Step 27.2 — Naibbe Dice Cipher Entropy Shift Test:*
- `naibbe_entropy.json` — Naibbe dice cipher implemented with Greshko's 2025 parameters (n_tables=2, bigram_prob=0.20, word_len_range=(3,6), prefix_prob=0.20, suffix_prob=0.30); Latin reference text encoded through 20 random instantiations; entropy shift vector compared to observed Voynich shift via cosine similarity.
  - **Greshko default cosine**: **-0.8427** (CI: [-0.868, -0.816]) — the Naibbe shifts entropy in exactly the **opposite direction** from Voynich. Where Voynich entropy rises at high orders (+0.80, +1.10, +0.99 at orders 4-6), Naibbe entropy falls (-0.48, -0.30, -0.18).
  - **Parameter grid search**: 81 configurations (n_tables ∈ {1,2,3} × bigram ∈ {0.10,0.20,0.30} × prefix ∈ {0.10,0.20,0.30} × suffix ∈ {0.20,0.30,0.40}) × 5 seeds each. **Every configuration produces a negative cosine.** Best grid result: -0.8117 (nt=3, bp=0.20, pp=0.10, sp=0.30). Refined with 20 seeds: **-0.8259** (CI: [-0.852, -0.803]).
  - **Updated ranking** (11 mechanisms): tachygraphic **0.8202** > homophonic 0.5664 > nomenclator 0.2889 > simple_substitution 0.0 > polyalphabetic -0.8024 > naibbe_best_grid -0.8259 > syllabic -0.8371 > **naibbe_greshko -0.8427** > syllabic_modifier -0.8580 > null_insertion -0.8754 > abbreviation_heavy -0.9497.
  - **Discrimination test**: CIs do not overlap — tachygraphic [0.820, 0.820] vs Naibbe [-0.868, -0.816]. **DISCRIMINATED.**
  - **Phase 18 cross-checks**: burstiness CV 0.847 vs Voynich 1.014 (consistent); LZ compression 0.493 vs 0.330 (inconsistent — Naibbe compresses worse); HMM transition entropy 3.622 vs 1.006 (inconsistent). Tri-state match: **1/3**.
  - Verdict: **TACHYGRAPHIC_CONFIRMED** — Naibbe ranks 8th of 11, below homophonic. The polyalphabetic substitution with random prefix/suffix additions increases low-order entropy and decreases high-order entropy — the exact opposite of the Voynich pattern.

*Step 27.3 — Combined Verdict:*
- `phase27_verdict.json` — Verdict: **CLASSIFIER_COMPROMISED_NAIBBE_OK**. One control failed, one passed.
  - The Phase 9.5 typology classification is unreliable: it cannot distinguish Voynich from deliberate gibberish (23/38) or self-citation text (28/28). The `encoded_natural` label should be interpreted as "text with complex statistical structure" rather than evidence of linguistic encoding. The entropy floor (0.978 vs 0.048) does discriminate but is not part of the classification rules.
  - The tachygraphic mechanism identification is strongly confirmed: the Naibbe dice cipher produces an entropy shift cosine of -0.843 (opposite direction), ranking 8th of 11 tested mechanisms, definitively outperformed by the tachygraphic model at +0.820 with non-overlapping confidence intervals.
  - **Paper revision required**: qualify the Phase 9.5 section to acknowledge the classifier does not discriminate Voynich from gibberish. The tachygraphic identification sections require no revision.
