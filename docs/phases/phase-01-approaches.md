[← Phase Index](README.md) | [Next: Phases 2-3 →](phase-02-03.md)

# Approaches 1-2: Stroke-Level Syllabary & Information-Theoretic Fingerprinting

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

---
[← Phase Index](README.md) | [Next: Phases 2-3 →](phase-02-03.md)
