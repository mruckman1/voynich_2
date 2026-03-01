"""
Voynich 2: Stroke-Level Syllabary & Information-Theoretic Fingerprinting
=========================================================================
Two complementary language-agnostic approaches to the Voynich manuscript.

Usage:
    voynich                   # Show corpus summary
    voynich corpus            # Load and summarize the EVA corpus
    voynich reference         # Show reference corpus summary
    voynich strokes           # Approach 1: stroke-level syllabary analysis
    voynich fingerprint       # Approach 2: information-theoretic fingerprinting
    voynich both              # Run both approaches sequentially
    voynich nulls             # Phase 2A: null character identification
    voynich grid              # Phase 2B: syllabary grid refinement
    voynich phase2            # Run both Phase 2 analyses
    voynich degeneracy        # Phase 3D: break substitution vs syllabary degeneracy
    voynich grid-validate     # Phase 3E: validate syllabary grid
    voynich syllable-match    # Phase 3F: syllable-level language matching
    voynich validate-all      # Phase 3G: scholarly validation framework
    voynich phase3            # Run all Phase 3 workstreams
    voynich audit             # Phase 4.1: discriminant audit of Phase 3 results
    voynich section-diagnosis # Phase 4.2: section consistency diagnosis
    voynich abugida           # Phase 4.3: abugida hypothesis test
    voynich multi-language    # Phase 4.4: multi-language comparison
    voynich phase4            # Run all Phase 4 analyses
    voynich lang-a            # Phase 4.5A+C: language A isolation + qo-removal
    voynich morpheme-grid     # Phase 4.5B: morpheme grid reinterpretation
    voynich phase4-5          # Run all Phase 4.5 analyses
    voynich paradigms         # Phase 5.1: paradigm discovery
    voynich paradigm-match    # Phase 5.2: paradigm-to-language matching
    voynich stem-id           # Phase 5.3: frequency-based stem identification
    voynich phonetic          # Phase 5.4+5.5: phonetic decode and validation
    voynich phase5            # Run all Phase 5 analyses
"""
import sys
import time


def cmd_corpus():
    """Load and display corpus summary."""
    from voynich.core.corpus import load_corpus
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
    from voynich.analysis.strokes import run_stroke_analysis
    t0 = time.time()
    run_stroke_analysis()
    print(f"\nStroke analysis completed in {time.time() - t0:.1f}s")


def cmd_fingerprint():
    """Run Approach 2: information-theoretic fingerprinting."""
    from voynich.analysis.fingerprint import run_fingerprint_analysis
    t0 = time.time()
    run_fingerprint_analysis()
    print(f"\nFingerprint analysis completed in {time.time() - t0:.1f}s")


def cmd_reference():
    """Show reference corpus summary."""
    from voynich.core.reference import load_reference_corpus
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
    from voynich.phases.nulls import run_null_analysis
    t0 = time.time()
    run_null_analysis()
    print(f"\nNull analysis completed in {time.time() - t0:.1f}s")


def cmd_grid():
    """Run Phase 2B: syllabary grid refinement."""
    from voynich.phases.grid_refine import run_grid_refinement
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


def cmd_degeneracy():
    """Run Phase 3D: break substitution vs syllabary degeneracy."""
    from voynich.phases.degeneracy import run_degeneracy_analysis
    t0 = time.time()
    run_degeneracy_analysis()
    print(f"\nDegeneracy analysis completed in {time.time() - t0:.1f}s")


def cmd_grid_validate():
    """Run Phase 3E: validate syllabary grid."""
    from voynich.phases.grid_validate import run_grid_validation
    t0 = time.time()
    run_grid_validation()
    print(f"\nGrid validation completed in {time.time() - t0:.1f}s")


def cmd_syllable_match():
    """Run Phase 3F: syllable-level language matching."""
    from voynich.phases.syllable_match import run_syllable_matching
    t0 = time.time()
    run_syllable_matching()
    print(f"\nSyllable matching completed in {time.time() - t0:.1f}s")


def cmd_validate_all():
    """Run Phase 3G: scholarly validation framework."""
    from voynich.phases.scholarly import run_scholarly_validation
    t0 = time.time()
    run_scholarly_validation()
    print(f"\nScholarly validation completed in {time.time() - t0:.1f}s")


def cmd_phase3():
    """Run all Phase 3 workstreams sequentially."""
    cmd_degeneracy()
    print("\n" + "=" * 70 + "\n")
    cmd_grid_validate()
    print("\n" + "=" * 70 + "\n")
    cmd_syllable_match()
    print("\n" + "=" * 70 + "\n")
    cmd_validate_all()


def cmd_audit():
    """Run Phase 4.1: discriminant audit of Phase 3 results."""
    from voynich.phases.discriminant_audit import run_discriminant_audit
    t0 = time.time()
    run_discriminant_audit()
    print(f"\nDiscriminant audit completed in {time.time() - t0:.1f}s")


def cmd_section_diagnosis():
    """Run Phase 4.2: section consistency diagnosis."""
    from voynich.phases.section_diagnosis import run_section_diagnosis
    t0 = time.time()
    run_section_diagnosis()
    print(f"\nSection diagnosis completed in {time.time() - t0:.1f}s")


def cmd_abugida():
    """Run Phase 4.3: abugida hypothesis test."""
    from voynich.phases.abugida_test import run_abugida_test
    t0 = time.time()
    run_abugida_test()
    print(f"\nAbugida test completed in {time.time() - t0:.1f}s")


def cmd_multi_language():
    """Run Phase 4.4: multi-language comparison."""
    from voynich.phases.multi_language import run_multi_language
    t0 = time.time()
    run_multi_language()
    print(f"\nMulti-language comparison completed in {time.time() - t0:.1f}s")


def cmd_phase4():
    """Run all Phase 4 analyses sequentially."""
    cmd_audit()
    print("\n" + "=" * 70 + "\n")
    cmd_section_diagnosis()
    print("\n" + "=" * 70 + "\n")
    cmd_abugida()
    print("\n" + "=" * 70 + "\n")
    cmd_multi_language()


def cmd_lang_a():
    """Run Phase 4.5A+C: Language A isolation and qo-removal."""
    from voynich.phases.language_a_isolation import run_language_a_isolation
    t0 = time.time()
    run_language_a_isolation()
    print(f"\nLanguage A isolation completed in {time.time() - t0:.1f}s")


def cmd_morpheme_grid():
    """Run Phase 4.5B: morpheme grid reinterpretation."""
    from voynich.phases.morpheme_grid import run_morpheme_grid
    t0 = time.time()
    run_morpheme_grid()
    print(f"\nMorpheme grid analysis completed in {time.time() - t0:.1f}s")


def cmd_phase45():
    """Run all Phase 4.5 analyses sequentially."""
    cmd_lang_a()
    print("\n" + "=" * 70 + "\n")
    cmd_morpheme_grid()


def cmd_paradigms():
    """Run Phase 5.1: paradigm discovery."""
    from voynich.phases.paradigm_discovery import run_paradigm_discovery
    t0 = time.time()
    run_paradigm_discovery()
    print(f"\nParadigm discovery completed in {time.time() - t0:.1f}s")


def cmd_paradigm_match():
    """Run Phase 5.2: paradigm-to-language matching."""
    from voynich.phases.paradigm_match import run_paradigm_match
    t0 = time.time()
    run_paradigm_match()
    print(f"\nParadigm matching completed in {time.time() - t0:.1f}s")


def cmd_stem_id():
    """Run Phase 5.3: frequency-based stem identification."""
    from voynich.phases.stem_identification import run_stem_identification
    t0 = time.time()
    run_stem_identification()
    print(f"\nStem identification completed in {time.time() - t0:.1f}s")


def cmd_phonetic():
    """Run Phase 5.4+5.5: phonetic value assignment and validation."""
    from voynich.phases.phonetic_decode import run_phonetic_decode
    t0 = time.time()
    run_phonetic_decode()
    print(f"\nPhonetic decode completed in {time.time() - t0:.1f}s")


def cmd_phase5():
    """Run all Phase 5 analyses sequentially (with gate checking)."""
    cmd_paradigms()
    print("\n" + "=" * 70 + "\n")
    cmd_paradigm_match()
    print("\n" + "=" * 70 + "\n")
    cmd_stem_id()
    print("\n" + "=" * 70 + "\n")
    cmd_phonetic()


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
        'degeneracy': cmd_degeneracy,
        'grid-validate': cmd_grid_validate,
        'syllable-match': cmd_syllable_match,
        'validate-all': cmd_validate_all,
        'phase3': cmd_phase3,
        'audit': cmd_audit,
        'section-diagnosis': cmd_section_diagnosis,
        'abugida': cmd_abugida,
        'multi-language': cmd_multi_language,
        'phase4': cmd_phase4,
        'lang-a': cmd_lang_a,
        'morpheme-grid': cmd_morpheme_grid,
        'phase4-5': cmd_phase45,
        'paradigms': cmd_paradigms,
        'paradigm-match': cmd_paradigm_match,
        'stem-id': cmd_stem_id,
        'phonetic': cmd_phonetic,
        'phase5': cmd_phase5,
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
