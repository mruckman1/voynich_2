# Voynich Manuscript: Syllabary & Information-Theoretic Analysis

Two complementary language-agnostic approaches to the Voynich manuscript that don't assume a target language. **Approach 1** (Stroke-Level Syllabary) analyzes the script's internal structure. **Approach 2** (Information-Theoretic Fingerprinting) compares the text's statistical profile against candidate language+encoding combinations.

Together, they answer two questions: **what kind of script is this?** and **what kind of language does it encode?**

Both approaches are grounded in one confirmed empirical finding: EVA compositionality (47.9% output change when decomposing multi-stroke characters) confirms that sub-character structure carries information.

## Quick Start

```bash
uv sync
python cli.py corpus        # Load and summarize the EVA corpus
python cli.py strokes       # Approach 1: stroke-level syllabary analysis
python cli.py fingerprint   # Approach 2: information-theoretic fingerprinting
python cli.py both          # Run both approaches
```

Requires Python 3.12+ and NumPy. The EVA transcription data (IVTFF format) should be placed in `data/corpus/`.

## Project Structure

```
voynich_2/
├── cli.py               # Entry point — run analyses from the command line
├── corpus.py            # IVTFF parser, EVA tokenizer, corpus access by section/scribe/language
├── strokes.py           # Approach 1: stroke decomposition, positional analysis, Ventris grid
├── fingerprint.py       # Approach 2: entropy profiling, reference library, profile matching
├── stats.py             # Entropy calculations, Zipf's law, bigram matrices, MI, TTR
├── ciphers.py           # Historical cipher implementations + encoding simulators
├── data/
│   ├── corpus/          # EVA transcription files (ZL3b-n.txt, RF1b-e.txt, IT2a-n.txt)
│   └── LocalPulls/      # Local reference corpora
├── results/             # JSON output from analysis runs
└── archive/             # Previous codebase (consonant-skeleton approach — deprecated)
```

## Approach 1: Stroke-Level Syllabary Analysis

The consonant skeleton approach failed because it assumed EVA characters are alphabetic (one character = one phoneme). But the compositionality test showed multi-stroke characters behave as ligatures. If each glyph represents a CV or CVC syllable, the "alphabet" is actually a syllabary — explaining both the low character-level entropy and the failure of phoneme-level matching.

This approach follows the Ventris method: map the script's internal structure (which signs share components, which signs appear in which positions) before identifying the language.

| Phase | Description | Module |
|-------|-------------|--------|
| 1.1 | **Stroke Decomposition** — Decompose EVA characters into 11 atomic stroke primitives (loop, open_curve, vertical, hook, descender, ascender, crossbar, sigmoid, plume, connector, tail). Covers all 23 single EVA characters + 17 ligatures. | `strokes.py` |
| 1.2 | **Positional Analysis** — Compute P(stroke \| position) for initial/medial/final positions. Measure MI(stroke, position) and chi-squared vs random/alphabetic null models. Strong positional constraints = syllabic structure. | `strokes.py` |
| 1.3 | **Ventris Grid** — Build a consonant x vowel grid grouping glyphs by shared initial stroke (onset) and final stroke (nucleus). Compare occupancy against Linear B, hiragana, and Cypriot syllabaries. | `strokes.py` |
| 1.4 | **Token Segmentation** — Re-analyze Voynich tokens as sequences of syllable units from the grid. Compute syllable-level entropy, TTR, and bigram statistics. | `strokes.py` |
| 1.5 | **Discriminant Validation** — Test whether the syllabary structure discriminates real Voynich text from character-shuffled null text (z-score on H2). | `strokes.py` |

## Approach 2: Information-Theoretic Fingerprinting

Instead of decoding first and checking after, characterize the Voynich text's statistical fingerprint and find which known language + encoding scheme produces the closest match.

| Phase | Description | Module |
|-------|-------------|--------|
| 2.1 | **Voynich Entropy Profile** — Compute a 37-dimensional vector: H1/H2/H3 (character), word entropy, MI at lags 1–10, intra-token MI, positional entropy at 10 positions, word-length entropy, Zipf exponent, TTR at 5 corpus sizes, bigram matrix entropy. | `fingerprint.py`, `stats.py` |
| 2.2 | **Reference Library** — Build equivalent profiles for 7 languages (Latin, Italian, German, Spanish, Hebrew, Arabic, Occitan) x 9 encoding schemes (raw, simple substitution, polyalphabetic, homophonic, nomenclator, syllabic, abbreviation light/heavy, null insertion) = 63 combinations. | `fingerprint.py`, `ciphers.py` |
| 2.3 | **Profile Matching** — Rank reference profiles by cosine similarity to the Voynich vector. Compute pairwise confusion matrix to identify which combinations are distinguishable. | `fingerprint.py` |
| 2.4 | **Section Differentiation** — Compute per-section profiles (herbal A/B, astronomical, biological, cosmological, pharmaceutical, recipes) and match independently. Tests whether sections encode different languages or use different schemes. | `fingerprint.py` |
| 2.5 | **Discriminant Validation** — Generate null text (shuffle, random, Markov) and verify that real Voynich matches reference profiles significantly better than null text does. | `fingerprint.py` |

## Integration

The two approaches cross-validate:

| Approach 1 finds | Approach 2 finds | Interpretation |
|---|---|---|
| CV syllabary grid with good fit | Closest match = [lang]-syllabic | Strong convergent evidence for syllabary encoding |
| CV syllabary grid with good fit | Closest match = [lang]-substitution | Conflict — grid may be an artifact of glyph structure |
| No syllabary structure | Closest match = [lang]-substitution | Consistent — script is alphabetic |
| CV syllabary grid | No good match | Novel encoding or unknown language |

## Data

The project uses EVA (Extended Voynich Alphabet) transcription files in IVTFF format. The parser (`corpus.py`) supports three transcription sources with automatic preference ordering: ZL3b-n.txt > RF1b-e.txt > IT2a-n.txt.

The corpus provides filtered access by:
- **Section**: herbal_a, herbal_b, astronomical, biological, cosmological, pharmaceutical, recipes
- **Currier language**: A (herbal A scribe) or B (remaining sections)
- **Scribe hand**: 1–5 (inferred from quire assignments)

## Results

Analysis outputs are saved as JSON to `results/`:
- `voynich_profile.json` — Full 37-dimensional entropy profile
- `section_profiles.json` — Per-section entropy profiles
- `match_rankings.json` — Ranked language+encoding matches
- `stroke_positional.json` — Stroke positional distributions and MI
- `ventris_grid.json` — Syllabary grid contents and occupancy
- `syllable_stats.json` — Syllable-level sequence statistics
- `stroke_discriminant.json` — Real vs shuffled discrimination z-scores
- `discriminant_validation.json` — Fingerprint discrimination vs null text

## Background

This project is a fresh start after a prior approach (consonant-skeleton-to-Latin-dictionary matching) proved unproductive. Three pieces of infrastructure were carried over:

1. **EVA transcription data and tokenizer** — IVTFF parsing with folio/line structure
2. **Discriminant validation framework** — null-text generation and comparison logic
3. **Section classification** — folio-to-section mapping for Currier A/B analysis

Everything else — skeleton generation, dictionary matching, candidate selection, iterative refinement — was specific to the failed approach and was not carried over.
