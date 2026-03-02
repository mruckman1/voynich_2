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
    voynich illustration      # Phase 6.0: illustration-constrained setup
    voynich rosetta           # Phase 6 D+E: Rosetta folio selection
    voynich anchor            # Phase 6 A+B: anchor-and-propagate
    voynich compete           # Phase 6 C: competitive ID resolution
    voynich phase6-validate   # Phase 6 validation battery
    voynich phase6            # Run all Phase 6 analyses
    voynich anchor-diagnosis  # Phase 6.1B: anchor inconsistency diagnosis
    voynich encoding-diagnosis # Phase 6.1C: encoding model diagnosis
    voynich phase6-1          # Run full Phase 6.1 pipeline (TF-IDF + diagnosis)
    voynich embeddings        # Approach 8: morpheme distributional semantics
    voynich slots             # Approach 9: pharmaceutical positional slot analysis
    voynich phase7            # Run full Phase 7 (Approaches 8 + 9 + integration)
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


def cmd_illustration():
    """Run Phase 6.0: illustration-constrained setup."""
    from voynich.phases.illustration_constrained import run_illustration_constrained
    t0 = time.time()
    run_illustration_constrained()
    print(f"\nIllustration-constrained setup completed in {time.time() - t0:.1f}s")


def cmd_rosetta():
    """Run Phase 6 D+E: Rosetta folio selection and encoding model test."""
    from voynich.phases.rosetta_selection import run_rosetta_selection
    t0 = time.time()
    run_rosetta_selection()
    print(f"\nRosetta selection completed in {time.time() - t0:.1f}s")


def cmd_anchor():
    """Run Phase 6 A+B: anchor-and-propagate with paradigm filtering."""
    from voynich.phases.anchor_propagate import run_anchor_propagate
    t0 = time.time()
    run_anchor_propagate()
    print(f"\nAnchor-and-propagate completed in {time.time() - t0:.1f}s")


def cmd_compete():
    """Run Phase 6 C: competitive ID resolution."""
    from voynich.phases.competitive_id import run_competitive_id
    t0 = time.time()
    run_competitive_id()
    print(f"\nCompetitive ID resolution completed in {time.time() - t0:.1f}s")


def cmd_phase6_validate():
    """Run Phase 6 validation battery."""
    from voynich.phases.illustration_validate import run_illustration_validate
    t0 = time.time()
    run_illustration_validate()
    print(f"\nPhase 6 validation completed in {time.time() - t0:.1f}s")


def cmd_phase6():
    """Run all Phase 6 analyses sequentially (with gate checking)."""
    cmd_illustration()
    print("\n" + "=" * 70 + "\n")
    cmd_rosetta()
    print("\n" + "=" * 70 + "\n")
    cmd_anchor()
    print("\n" + "=" * 70 + "\n")
    cmd_compete()
    print("\n" + "=" * 70 + "\n")
    cmd_phase6_validate()


def cmd_anchor_diagnosis():
    """Run Phase 6.1B: anchor-level inconsistency diagnosis."""
    from voynich.phases.anchor_diagnosis import run_anchor_diagnosis
    t0 = time.time()
    run_anchor_diagnosis()
    print(f"\nAnchor diagnosis completed in {time.time() - t0:.1f}s")


def cmd_encoding_diagnosis():
    """Run Phase 6.1C: encoding model diagnosis."""
    from voynich.phases.encoding_diagnosis import run_encoding_diagnosis
    t0 = time.time()
    run_encoding_diagnosis()
    print(f"\nEncoding diagnosis completed in {time.time() - t0:.1f}s")


def cmd_phase61():
    """Run full Phase 6.1 pipeline: TF-IDF stems + encoding + anchor diagnosis."""
    # Step 1: TF-IDF stem extraction
    from voynich.phases.illustration_constrained import run_illustration_constrained
    t0 = time.time()
    run_illustration_constrained(use_tfidf=True)
    print(f"\nTF-IDF illustration setup completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 2: Rosetta selection (uses new stems)
    from voynich.phases.rosetta_selection import run_rosetta_selection
    t0 = time.time()
    run_rosetta_selection(use_tfidf=True)
    print(f"\nRosetta selection completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 3: Anchor-propagate with TF-IDF stems
    from voynich.phases.anchor_propagate import run_anchor_propagate
    t0 = time.time()
    run_anchor_propagate(use_tfidf=True)
    print(f"\nAnchor-and-propagate (TF-IDF) completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 4: Encoding model diagnosis
    from voynich.phases.encoding_diagnosis import run_encoding_diagnosis
    t0 = time.time()
    run_encoding_diagnosis(use_tfidf=True)
    print(f"\nEncoding diagnosis completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 5: Anchor diagnosis
    from voynich.phases.anchor_diagnosis import run_anchor_diagnosis
    t0 = time.time()
    run_anchor_diagnosis(use_tfidf=True)
    print(f"\nAnchor diagnosis completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 6: Competitive ID
    from voynich.phases.competitive_id import run_competitive_id
    t0 = time.time()
    run_competitive_id()
    print(f"\nCompetitive ID completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70 + "\n")

    # Step 7: Validation battery
    from voynich.phases.illustration_validate import run_illustration_validate
    t0 = time.time()
    run_illustration_validate()
    print(f"\nPhase 6.1 validation completed in {time.time() - t0:.1f}s")


def cmd_embeddings():
    """Run Approach 8: morpheme-level distributional semantics."""
    from voynich.phases.distributional import run_distributional
    t0 = time.time()
    run_distributional()
    print(f"\nDistributional analysis completed in {time.time() - t0:.1f}s")


def cmd_slots():
    """Run Approach 9: pharmaceutical positional slot analysis."""
    from voynich.phases.positional_slots import run_positional_slots
    t0 = time.time()
    run_positional_slots()
    print(f"\nPositional slot analysis completed in {time.time() - t0:.1f}s")


def cmd_phase7():
    """Run full Phase 7: Approaches 8 + 9 + integration."""
    cmd_embeddings()
    print("\n" + "=" * 70 + "\n")
    cmd_slots()
    print("\n" + "=" * 70 + "\n")
    from voynich.phases.approach_integration import run_approach_integration
    t0 = time.time()
    run_approach_integration()
    print(f"\nApproach integration completed in {time.time() - t0:.1f}s")


def cmd_combined_embed():
    """Run Phase 7.5 Step 1: combined A+B corpus embeddings."""
    from voynich.phases.distributional import run_combined_distributional
    t0 = time.time()
    run_combined_distributional()
    print(f"\nCombined embeddings completed in {time.time() - t0:.1f}s")


def cmd_noun_clusters():
    """Run Phase 7.5 Step 2: noun subcluster analysis."""
    from voynich.phases.noun_subclusters import run_noun_subclusters
    t0 = time.time()
    run_noun_subclusters()
    print(f"\nNoun subcluster analysis completed in {time.time() - t0:.1f}s")


def cmd_verb_id():
    """Run Phase 7.5 Step 3: verb identification."""
    from voynich.phases.verb_identification import run_verb_identification
    t0 = time.time()
    run_verb_identification()
    print(f"\nVerb identification completed in {time.time() - t0:.1f}s")


def cmd_embed_bridge():
    """Run Phase 7.5 Step 4: illustration-embedding bridge."""
    from voynich.phases.embedding_bridge import run_embedding_bridge
    t0 = time.time()
    run_embedding_bridge()
    print(f"\nEmbedding bridge completed in {time.time() - t0:.1f}s")


def cmd_convergence():
    """Run Phase 7.5 Step 5: convergence scoring."""
    from voynich.phases.convergence_score import run_convergence_score
    t0 = time.time()
    run_convergence_score()
    print(f"\nConvergence scoring completed in {time.time() - t0:.1f}s")


def cmd_phase75():
    """Run full Phase 7.5: Exploiting the Noun Coherence Bridge."""
    cmd_combined_embed()
    print("\n" + "=" * 70 + "\n")
    cmd_noun_clusters()
    print("\n" + "=" * 70 + "\n")
    cmd_verb_id()
    print("\n" + "=" * 70 + "\n")
    cmd_embed_bridge()
    print("\n" + "=" * 70 + "\n")
    cmd_convergence()


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
        'illustration': cmd_illustration,
        'rosetta': cmd_rosetta,
        'anchor': cmd_anchor,
        'compete': cmd_compete,
        'phase6-validate': cmd_phase6_validate,
        'phase6': cmd_phase6,
        'anchor-diagnosis': cmd_anchor_diagnosis,
        'encoding-diagnosis': cmd_encoding_diagnosis,
        'phase6-1': cmd_phase61,
        'embeddings': cmd_embeddings,
        'slots': cmd_slots,
        'phase7': cmd_phase7,
        'combined-embed': cmd_combined_embed,
        'noun-clusters': cmd_noun_clusters,
        'verb-id': cmd_verb_id,
        'embed-bridge': cmd_embed_bridge,
        'convergence': cmd_convergence,
        'phase7-5': cmd_phase75,
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
