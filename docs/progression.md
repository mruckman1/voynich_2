# Progression Table

[← Back to README](../README.md)

Key metrics tracked across phases. Dict-Hit = percentage of decoded tokens matching the reference dictionary. Signal = number of genuine signal words (hits on real Voynich that miss on null corpora). Bigram z = z-score measuring whether signal tokens form valid Latin word sequences above chance.

| Phase | Dict-Hit | Signal | Bigram z | Key Advance |
|-------|----------|--------|----------|-------------|
| 11 | 11.1% (17K) | — | — | CSP phonetic decoder, CV grid |
| 14 | 19.4% (17K) | — | — | Stroke-feature model (breakthrough) |
| 15 | 35.4% (131K) | — | — | Dictionary expansion + articulatory constraints |
| 16 | 43.6% (131K) | — | — | Feature model + modifiers |
| 17 | — | — | — | NO-GO: null corpora hit 37.6% |
| 28 | 43.6% | 8 words | — | Signal isolation, table tiering |
| 29 | 43.6% | — | 6.14 | SIGNAL bigram discovery |
| 36 | 24.1% (10K) | 51 words, 5.43× | 12.66 | 10K dictionary, validated pipeline |
| 37 | — | +22 Italian | — | Italian signal words, macaronic confirmation |
| 42 | 43.6% | — | 14.78 (conservative) | Z-score audit, canonical methodology |
| 44 | — | — | — | MaxSAT landscape: FLAT (500+ solutions) |
| 50A | 30.3% (ED1) | sel = 1.10× | — | ED1 approach invalidated |
| 50B | — | scramble = 1.00× | CC z = 21.0 | Word LM adds nothing; CC bigrams real |
| 50D | — | Italian #1 | — | Size-matched language ID confirmed |
| 51A | — | z = 4.23 | — | Suffix map PARTIAL; POS 67% coverage |
| 51B | — | z = 4.57 | — | Bridge MARGINAL; 93.6% confirmed-triple coverage |
| 52 | 40.1% coverage | 22 T1 words, 56 paradigms | — | Word catalog; 74.8% Circa Instans overlap |
| 53 | — | z = 0.02 (null) | — | Paradigm constraints not table-specific; variable-length encoding confirmed |
| 54 | — | INDETERMINATE | — | Dialect ID: Ligurian #1 (0.248), gap 0.012; 40% agreement; Fisher p=0.019 |
| 55A | — | SCHINNER_ABOVE | — | Entropy shift extended: 13 mechanisms; Cardan +0.49–0.59 (discriminated); Schinner +0.95–0.97 (above tachy, scope limitation exposed) |
| 55B | — | PREDICTION_CONFIRMED_UNIQUE | — | Currier cross-boundary MI: Voynich 0.190 bits / 1.450× (z=24.9σ); tachy-syl 1.284× (11% off); Schinner 1.044× (same as null) |
| 56 | — | COMPATIBLE (10/10) | — | Costamagna 1953 structural match: 21/21 syllables attested; 5 codas = 5 modifier stroke types; 3 shared pairs = 3 ambiguous triples |
| 57 | 27.5% (CVC) | 64 words, 4.76× | 96.19 | CVC coda decode: hook→n, descender→r, sigmoid→s, vertical→t, connector→l; dict_hit drops but bigram z and net signal dramatically increase |
| 58 | — | FAIL (3/8) | — | Costamagna CSP: 22/25 confirmed, 3 ambiguous; search space 10^5.6; no improvement — visual matching essential |
| 59 | 79.9% (segmented) | CVC_VALIDATED (8/11) | — | CVC refinement: 4.3%→79.9% attestation (measurement artifact); connector→r; vertical→t (1.57×, sub-groups p=0.003); -aiin→Latin declensions (62.3%); cross-MI increased (refuted absorption); permutation coherence degraded (p=0.552) |
| 60A | 29.0% (corrected CVC) | 75 words, 4.51× | 87.74 | Corrected CVC: connector→r applied (+5.5% affected tokens); i→syllabic (+4.0%); attestation 83.0%; 14 new signal words; composite 0.94 (#1 strategy) |
| 60B | — | COHERENCE_RARE (5/5) | — | Recalibrated coherence: p=0.006 joint (4 criteria), Fisher p=0.011; CVC p ≤ CV p (0.006 vs 0.011); content words, ending diversity, pharma, signal count discriminate |
| 60D | — | 4/5 gates | — | Recipe annotation: 340 recipes, top 10 at 94.9% glossed, max 26 consecutive glossed; pharmaceutical cross-references found |
| 61A | — | 3/5 gates | — | Deep recipe reading: 4/5 match CI templates (best 0.75), conf=0.94, 3/5 with ingredients; 0 concatenations; readings fragmentary |
| 61B | — | p(CV coh)=0.001 | — | Full CV permutation under CVC: 1000 trials; p(count)=0.013; p(CVC coh)=0.006 (beats CV 0.011); p(CV coh)=0.001; real 75 vs random 54±9 signal words |
| 61C | — | 2/4 gates | — | Costamagna sequence rules: 7 constraints; catalog attestation 17.0% vs null 20.1% (sel 1.18×, z=−22.9); coda inventory 8.5% vs Latin 34.4%; most constraints not discriminating |
| 61D | — | NO_SIGNAL (0/5) | — | Zodiac CVC: 1194 labels decoded; 0 matches at ED≤2 (decoded strings 7–12 chars vs 3–8 char names); confirms zodiac uses different encoding |
| 62 | — | PARTIAL (7/11) | — | Exhaustive pre-visual: Q1=TOKENS_ARE_SYLLABLES (2/4); Q2=MINOR_CORRECTIONS_NEEDED (2/4; rr/ss dominate illegal clusters); Q3=SIGNIFICANT_STRUCTURE (3/3; A/B coda chi²=222, hands chi²=1285, entropy 0.89 similar) |
| 63A | — | VISUAL_NO_SIGNAL (0/5) | — | Font-based visual comparison: Gemini Embedding 2 (768-dim); mean max sim 0.874±0.025; 0/25 top-5, 1/25 top-15; perm p=0.379; family z=1.47 (p=0.071 near-sig); embedding model lacks stroke-level resolution |
| 63B | — | VISUAL_MISMATCH (B:2/4, A:0/5) | — | Manuscript segmentation: 4/5 line match (80%); 73% word seg; 22/37 char types; 211 exemplars; manuscript embeddings 0/5 A-gates (worse than font); methodology insufficient, not data quality |
| 64 | — | WEAK_SUPPORT (2/7) | — | Multi-method visual comparison: 7 methods (2 LLM, 5 CV); graph features best (mean rank 85.4/236); LLM pairwise 56% win rate (PASS); all methods discriminate (PASS); perm p=0.138 (NS); 0/25 same_basic_structure; best: y→si (rank 25.5), a→ra (41.2); worst: f→fa (202.5); domain gap (font vs handwriting) dominates |
| 65 | — | SEGMENTATION_FAILED (0/4) | — | Word boundary discovery: 4 methods (Harris MI, Bayesian MDL, char LM, recipe DP); Latin calibration F1=0.651/0.654 (methods work on Latin); Voynich dict hit 0–7.5%, selectivity 0.00–0.45× (all below random); EVA baseline 15.4% dict hit outperforms all methods; 17-char alphabet; 56% decode error rate is bottleneck |
| 66 | — | STRUCTURAL_INSIGHT (5/12) | — | Multi-vector attack: 12 tracks, 4 tiers; LLM reading CONTROLS_DOMINATE (null 65.6% ≈ real 78.7%, ratio 1.20×); Fontana STRUCTURALLY_SIMILAR (10 vs 12 families); Hand 4 STRUCTURED (280 bigrams, `-ar` 2067×); f116v CRIB_SUPPORTED (`oror`→`nes` ED=0); collocations 3107 sig but selectivity 1.0×; hallucination controls caught LLM reading noise as content |
| 67 | 29.2% (voted) | PARTIAL_RESOLUTION (3/5) | 125.86 | Multi-angle triple resolution: 5 tracks (wildcard, frequency, features, evolutionary, distributional); 8/13 LIKELY (2-track agree), 0 RESOLVED (3-track); evo +1.3% dict hit; distributional 4/4 gates (39 anchors, 23% exact, 46% related); wildcard 25.3% unique (1.13× sel); voting collapses ambiguous triples toward common confirmed syllables; 56% decode error persists |
| 68 | 30.6% (voted) | PARTIAL_RESOLUTION (5/5) | 112.77 | Rare syllable recovery: 7 tracks (full tokens, within-token, paradigmatic, expanded T1, formulaic, distributional, ED lattice); 1 RESOLVED (be→de, 3-track), 5 LIKELY (2-track); dict hit +1.57%; signal 80→81; 63% of tokens fully decoded (0% error); 223 T1 identifications (up from 22); 4914 minimal pairs; 82 recurring formulaic patterns; token-level constraints find rare syllables that corpus-level metrics miss |
| 69 | 35.9% (clean) | CLEAN_CORE_PARTIAL (4/7) | — | Clean core validation + exploitation: 22,823 clean tokens (63%); coherence p=0.006 (PASS), CV perm p=0.092 (FAIL); segmentation FAILS on clean data (LM 13.2% << EVA 40.6%); LLM 2.41× vs null; T1 network 49 paradigms, 888 seq pairs; 3,036 T1-dense passages (88-93%); 74% T1 in CI; distrib 5.1% convergence; EVA boundaries are structurally essential |
| 70 | 36.2% (expanded) | PHARMACEUTICAL_READING (11/19) | — | Token-as-word exploitation: 4 tracks; dict expansion +0.3% (NOT bottleneck, 9 new types only); 32 paradigms, coda -s→2sg (100%), -t→3sg (82%); 818 ordered pairs (66 VERB+OBJ/PREP+NOUN), 50 glossed trigrams; 20 annotated passages at 80% identified (2.29× vs random); 20/20 CI chapter matches; all 6 reading gates PASS; decode error rate (56%) is sole remaining bottleneck |
| 71 | — | MARGINAL (11/16) | — | Inflectional reverse engineering: 3 tracks; inflectional catalog 2/5 (57.2% verbal, coda -r=47% of coda tokens, null p=0.26, section chi² p≈0); root identification 4/5 (342 paradigms, 79.5% identified, 47.9% coverage, 2 PREPARATION roots); grammatical reading 5/6 (90% gram, 83% lex, 1.91× lex sel, template sel 0.95×); xval agreement 24%; coda→grammar mapping does not scale from paradigm-level to corpus-wide |
| 72 | 30.2% (null conn) | MODEL_REVISED (16/23) | 90.5 | Decode model diagnosis: 5 tracks; connector→null WINS (30.2% dict, xval 90.5% vs r 77.9%, 98.1% medial); xval CONNECTOR_R_IS_PROBLEM (coda -s 92.8%, -t 86.5%, -r 16.9%); null_connector model best (0.661 vs append 0.626); T1 expansion 530 IDs (FPR 100% beyond Tier B); variable-length 7/12 prefer shorter but bigram z drops 87→61 (collisions); connector is scribal ligature not phonetic |
| 73 | 30.2% (corrected) | CORRECTION_NEUTRAL (15/23) | 90.48 | Corrected model pipeline: connector→null applied corpus-wide; dict 29.0%→30.2%, xval 77.9%→90.5%, bigram z 87→90; revalidation PARTIAL (coherence p=0.027 PASS, CV perm p=0.104 FAIL); grammar FAILED (verbal 57.3% unchanged, descender-r=14,164 tokens); T1 STABLE (89.7%, 223→243 IDs); paradigms VALIDATED (PREPARATION roots 2→15); readings 80% identified (2.20× sel); connector→null adopted as new baseline |
| 74 | 37.6% (desc→null) | DESCENDER_RESOLVED_AND_VOCAB_EXPANDED (12/18) | — | Descender investigation + T1 push: 2 paths, 5 tracks; descender→r ranks 10th/13 (verbal 65.1%→24.2%); 13/15 triples prefer null (same convergence as connector→null); descender 94.6% token-final (genuine coda, not ligature); 675 EVA-level pattern expansions (595 distributional, 173 positional); LLM gap-fill KA=40%, selectivity 3.82×, 0 accepted (decode error blocks); 6 passages >90% identified (3.25× selectivity); descender likely non-phonetic like connector |
| 75 | 37.6% (3-coda) | THREE_CODA_NEUTRAL (12/23) | 71.34 | 3-coda model pipeline: connector+descender→null applied; dict 30.2%→37.6% (+7.4pp), signal 76→62, bigram z 90.5→71.3; verbal 57.3%→25.2%; bootstrap p=0.0000 (FIRST SIGNIFICANT grammar null test); xval 26.2%→54.7%; T1 243→316 (+73); 3 passages at 100% identified (project first); distributional coverage 25.2%; revalidation FAILED (0/3); template sel 0.23× (UNMARKED 47.7% dilutes patterns); descender correction trades signal for dict-hit |
| 76 | 37.8% (w/ LIKELY) | NO_PROGRESS (7/16) | — | Triple resolution from vocabulary convergence: 4 tracks; wildcard propagation 5/13 triples constrained (3 LIKELY: re, gu, re), LOO 0/12 (structurally inapplicable), clean 63%→71.5%; 10,493 parallel passage pairs (66,331 diagnostic diffs, 1,478 substitution tokens); top blocker ascender,crossbar,gallows (52 types, 1,770 tokens); LLM gap-fill 2 ACCEPTED ("deinde"+"rane") but KA 26.7%, selectivity 1.03× — tentative leads only |
| 77 | — | SELF_CITATION_ELIMINATED (4/4) | — | Timm-Schinner self-citation discriminator test: 540 corpora (27 configs × 20 seeds); entropy cosine −0.153 (anticorrelated, rank 10/15); MI ratio 1.036× (null level, ≈ Schinner 1.044×); both tests eliminate; tachygraphy sole survivor of 13 mechanisms on both discriminators; closes paper's most significant gap |
| 78 | — | CVC_T1_SIGNIFICANT (1/3) | — | CVC T1 permutation validation: 1,000 random CV tables; real 331 IDs vs null 209.6±32.0 (z=3.79, p=0.002); real exceeds 99th percentile (291); 5 words unique to real table; mean word specificity 0.947; closes T1 validation caveat |

## Background

This project is a fresh start after a prior approach (consonant-skeleton-to-Latin-dictionary matching) proved unproductive. Three pieces of infrastructure were carried over:

1. **EVA transcription data and tokenizer** — IVTFF parsing with folio/line structure
2. **Discriminant validation framework** — null-text generation and comparison logic
3. **Section classification** — folio-to-section mapping for Currier A/B analysis

Everything else — skeleton generation, dictionary matching, candidate selection, iterative refinement — was specific to the failed approach and was not carried over.
