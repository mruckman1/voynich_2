"""
Voynich 2: Stroke-Level Syllabary & Information-Theoretic Fingerprinting
=========================================================================
Two complementary language-agnostic approaches to the Voynich manuscript.

Usage:
    python cli.py                 # Show corpus summary
    python cli.py corpus          # Load and summarize the EVA corpus
    python cli.py reference       # Show reference corpus summary
    python cli.py strokes         # Approach 1: stroke-level syllabary analysis
    python cli.py fingerprint     # Approach 2: information-theoretic fingerprinting
    python cli.py both            # Run both approaches sequentially
    python cli.py nulls           # Phase 2A: null character identification
    python cli.py grid            # Phase 2B: syllabary grid refinement
    python cli.py phase2          # Run both Phase 2 analyses
"""
import sys
import time


def cmd_corpus():
    """Load and display corpus summary."""
    from corpus import load_corpus
    corpus = load_corpus(verbose=True)
    summary = corpus.summary()
    print("\nCorpus Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Show section-level token counts
    print("\nTokens per section:")
    for section in sorted(set(
        p.section for p in corpus.pages.values()
    )):
        tokens = corpus.get_tokens(section=section, paragraph_only=True)
        print(f"  {section:<20s}: {len(tokens):6d} tokens")


def cmd_strokes():
    """Run Approach 1: stroke-level syllabary analysis."""
    from strokes import run_stroke_analysis
    t0 = time.time()
    run_stroke_analysis()
    print(f"\nStroke analysis completed in {time.time() - t0:.1f}s")


def cmd_fingerprint():
    """Run Approach 2: information-theoretic fingerprinting."""
    from fingerprint import run_fingerprint_analysis
    t0 = time.time()
    run_fingerprint_analysis()
    print(f"\nFingerprint analysis completed in {time.time() - t0:.1f}s")


def cmd_reference():
    """Show reference corpus summary."""
    from reference import load_reference_corpus
    try:
        corpus = load_reference_corpus(verbose=True)
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    print("\nReference Corpus Summary:")
    for lang, info in sorted(corpus.summary().items()):
        print(f"\n  {lang}:")
        print(f"    Texts:  {info['texts']}")
        print(f"    Tokens: {info['total_tokens']:,}")
        for name in info['files']:
            texts = [t for t in corpus.get_texts(lang) if t.name == name]
            if texts:
                print(f"      - {name}: {texts[0].token_count:,} tokens")


def cmd_nulls():
    """Run Phase 2A: null character identification."""
    from nulls import run_null_analysis
    t0 = time.time()
    run_null_analysis()
    print(f"\nNull analysis completed in {time.time() - t0:.1f}s")


def cmd_grid():
    """Run Phase 2B: syllabary grid refinement."""
    from grid_refine import run_grid_refinement
    t0 = time.time()
    run_grid_refinement()
    print(f"\nGrid refinement completed in {time.time() - t0:.1f}s")


def cmd_phase2():
    """Run both Phase 2 analyses sequentially."""
    cmd_nulls()
    print("\n" + "=" * 70 + "\n")
    cmd_grid()


def cmd_both():
    """Run both approaches sequentially."""
    cmd_strokes()
    print("\n" + "=" * 70 + "\n")
    cmd_fingerprint()


def main():
    commands = {
        'corpus': cmd_corpus,
        'reference': cmd_reference,
        'strokes': cmd_strokes,
        'fingerprint': cmd_fingerprint,
        'both': cmd_both,
        'nulls': cmd_nulls,
        'grid': cmd_grid,
        'phase2': cmd_phase2,
    }

    if len(sys.argv) < 2:
        cmd_corpus()
        return

    command = sys.argv[1]
    if command in commands:
        commands[command]()
    elif command in ('-h', '--help', 'help'):
        print(__doc__)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
