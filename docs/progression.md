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

## Background

This project is a fresh start after a prior approach (consonant-skeleton-to-Latin-dictionary matching) proved unproductive. Three pieces of infrastructure were carried over:

1. **EVA transcription data and tokenizer** — IVTFF parsing with folio/line structure
2. **Discriminant validation framework** — null-text generation and comparison logic
3. **Section classification** — folio-to-section mapping for Currier A/B analysis

Everything else — skeleton generation, dictionary matching, candidate selection, iterative refinement — was specific to the failed approach and was not carried over.
