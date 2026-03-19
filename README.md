# We Know the Voice but Not the Song

**Voynich Manuscript computational analysis**: stroke-feature syllabary decoding, signal isolation, and Italian tachygraphic hypothesis testing.

Companion code for: Ruckman (2026), *"We Know the Voice but Not the Song: Italian Tachygraphy and the Voynich Manuscript."*

## Overview

Two independent pipelines — one treating each Voynich character as a letter, the other as a syllable — produce consistent structural conclusions:

- The source language is **Romance** (a mix of Latin and Italian)
- The content is **medieval medical/herbal**
- Two distinct subsystems coexist (Currier A/B)
- The morphological structure is **genuine**
- **Italian syllabic tachygraphy** is identified as the encoding mechanism, discriminated from 13 tested mechanisms; Schinner's stochastic model is also testable but exposes a scope limitation of the discriminator (see Phase 55); Rugg-Taylor's Cardan grille is clearly discriminated (cosine +0.49–+0.59 vs tachygraphy's +0.820)
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
voynich naibbe            # Phase 27: Naibbe cipher rejection (Paper Sec 4.3)
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
| Sec 3.2 | Two distinct subsystems (A/B) | 4 | `voynich section-diagnosis` |
| Sec 3.3 | Medieval medical content | 15, 47 | `voynich text-analysis`, `voynich read-recipes` |
| Sec 3.4 | Genuine morphological structure | 5 | `voynich paradigms` |
| Sec 4.1 | Three-way ambiguity | 18 | `voynich hypothesis-disc` |
| Sec 4.2 | Entropy shift discriminator (9→13 mechanisms) | 19, 55 | `voynich entropy-shift`, `voynich entropy-extended` |
| Sec 4.3 | Tachygraphy vs Naibbe | 27 | `voynich naibbe` |
| Sec 4.4 | Sign family structure | 19 | `voynich tachy-stroke`, `voynich reviewer-family` |
| Sec 5 | Signal isolation methodology | 17, 28 | `voynich null-corpus`, `voynich signal-iso` |
| Sec 6.1 | Signal words | 36-38 | `voynich phase29` |
| Sec 6.2 | Permutation test | Reviewer | `voynich reviewer-perm`, `voynich reviewer-coherence` |
| Sec 6.3 | Word-level identifications | 52 | See [phase docs](docs/phases/phase-49-53.md) |
| Sec 6.4 | EVA words as syllables | 29, 55 | `voynich signal-bigram`, `voynich currier-voynich` |
| Sec 7 | Encoding structure | 14-16, 31 | `voynich feature-csp`, `voynich determ-test` |
| Sec 8 | Limitations & failures | 17, 20-23, 33 | `voynich step0`, `voynich phase33` |
| Sec 9 | Additional properties | 24 | `voynich direction`, `voynich section-xfer` |
| App A | Phase narratives | 1-53 | [docs/phases/](docs/phases/) |
| App B | Complete signal vocabulary | 36-38 | [docs/signal-vocabulary.md](docs/signal-vocabulary.md) |
| App C | Z-score methodology audit | 47 | `voynich track-a-47` |

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

## Assignment Table (T_P15)

25 stroke-feature triples → syllable assignments:
- **12 confirmed** (cross-source validation, Phases 14 + 19.8): these produce the 70 signal words
- **10 landscape-confirmed** (MaxSAT consensus >60%): statistically supported but landscape is flat
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
│   └── phases/               # Phase modules (2-55 + reviewer analyses)
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

Full table: [docs/progression.md](docs/progression.md)

## Detailed Documentation

- [Complete CLI Command Reference](docs/commands.md) — all commands grouped by phase
- [Phase-by-Phase Documentation](docs/phases/) — detailed results for all 54 phases
- [Signal Vocabulary Tables](docs/signal-vocabulary.md) — 70 signal words + 22 word-level identifications
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
