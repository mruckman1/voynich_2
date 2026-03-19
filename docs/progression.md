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

## Background

This project is a fresh start after a prior approach (consonant-skeleton-to-Latin-dictionary matching) proved unproductive. Three pieces of infrastructure were carried over:

1. **EVA transcription data and tokenizer** — IVTFF parsing with folio/line structure
2. **Discriminant validation framework** — null-text generation and comparison logic
3. **Section classification** — folio-to-section mapping for Currier A/B analysis

Everything else — skeleton generation, dictionary matching, candidate selection, iterative refinement — was specific to the failed approach and was not carried over.
