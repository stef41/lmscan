"""Exhaustive edge-case tests for lmscan features, detector, report, and CLI."""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from lmscan.features import (
    _count_syllables,
    _flesch_kincaid_grade,
    _split_paragraphs,
    _split_sentences,
    _tokenize,
    bigram_repetition,
    burstiness,
    extract_features,
    hapax_legomena_ratio,
    punctuation_entropy,
    readability_consistency,
    sentence_length_variance,
    slop_word_score,
    transition_word_ratio,
    trigram_repetition,
    vocabulary_richness,
    word_entropy,
    zipf_deviation,
)
from lmscan._types import TextFeatures, ParagraphScore
from lmscan.report import (
    _signal_icon,
    _fmt_value,
    format_paragraph_report,
    format_report,
    format_json,
    format_directory_report,
    format_html,
)
from lmscan.scanner import scan, scan_file, scan_directory, scan_mixed
from lmscan.cli import main


# ═══════════════════════════════════════════════════════════════════════════════
#  _count_syllables — edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestCountSyllables:
    def test_empty(self):
        assert _count_syllables("") == 0

    def test_single_vowel_word(self):
        assert _count_syllables("a") == 1

    def test_two_char_word(self):
        assert _count_syllables("go") == 1

    def test_silent_e(self):
        # "make" → drop trailing e → "mak" → 1 vowel group → 1 syllable
        assert _count_syllables("make") == 1

    def test_le_ending(self):
        # "apple" → ends with "le" so NOT stripped → "apple" → 2 vowel groups
        assert _count_syllables("apple") == 2

    def test_monosyllabic(self):
        assert _count_syllables("cat") == 1
        assert _count_syllables("the") == 1

    def test_polysyllabic(self):
        assert _count_syllables("beautiful") >= 3
        assert _count_syllables("university") >= 4

    def test_contraction(self):
        # y counts as vowel
        assert _count_syllables("they") >= 1

    def test_consecutive_vowels(self):
        # "queue" — consecutive vowels count as one group
        result = _count_syllables("queue")
        assert result >= 1

    def test_word_becomes_empty_after_strip(self):
        # "e" → strip trailing e but not "le" → remove "e" → empty → return 1
        assert _count_syllables("e") == 1

    def test_whitespace(self):
        assert _count_syllables("   ") == 0

    def test_only_consonants(self):
        # "rhythm" has y as vowel
        assert _count_syllables("rhythm") >= 1

    def test_hyphenated(self):
        assert _count_syllables("self-contained") >= 3


# ═══════════════════════════════════════════════════════════════════════════════
#  _flesch_kincaid_grade — edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestFleschKincaid:
    def test_empty(self):
        assert _flesch_kincaid_grade("") == 0.0

    def test_single_word(self):
        # "Hello." → 1 sentence, 1 word, ~2 syllables
        grade = _flesch_kincaid_grade("Hello.")
        assert isinstance(grade, float)

    def test_simple_sentence(self):
        grade = _flesch_kincaid_grade("The cat sat on the mat.")
        assert grade < 5.0  # Simple text, low grade level

    def test_complex_sentence(self):
        grade = _flesch_kincaid_grade(
            "The epistemological ramifications of quantum entanglement "
            "transcend conventional phenomenological understanding."
        )
        # Complex words → higher grade level
        assert grade > 5.0

    def test_no_words(self):
        assert _flesch_kincaid_grade("!!!") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  trigram_repetition — direct tests (was coverage gap)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrigramRepetition:
    def test_empty(self):
        assert trigram_repetition("") == 0.0

    def test_too_short(self):
        assert trigram_repetition("one two three") == 0.0

    def test_no_repetition(self):
        text = "alpha beta gamma delta epsilon zeta eta"
        result = trigram_repetition(text)
        assert result == 0.0

    def test_high_repetition(self):
        text = "the cat sat the cat sat the cat sat the cat sat"
        result = trigram_repetition(text)
        assert result > 0.3

    def test_single_repeat(self):
        text = "we are here we are here and nothing else"
        result = trigram_repetition(text)
        assert result > 0.0

    def test_three_words_only(self):
        # Exactly 3 words — len < 4 check
        assert trigram_repetition("one two three") == 0.0

    def test_four_words_min(self):
        result = trigram_repetition("a b c d")
        assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════════════════════════
#  format_paragraph_report — was zero tests (coverage gap)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatParagraphReport:
    def test_empty(self):
        report = format_paragraph_report([])
        assert "Per-Paragraph" in report

    def test_single_paragraph(self):
        p = ParagraphScore(index=0, ai_probability=0.3, verdict="Likely Human", word_count=50, text="test")
        report = format_paragraph_report([p])
        assert "1" in report  # paragraph #1
        assert "30%" in report
        assert "Likely Human" in report

    def test_all_ai(self):
        paras = [
            ParagraphScore(index=0, ai_probability=0.8, verdict="Likely AI", word_count=30, text="t"),
            ParagraphScore(index=1, ai_probability=0.9, verdict="Likely AI", word_count=40, text="t"),
        ]
        report = format_paragraph_report(paras)
        assert "All paragraphs appear AI-generated" in report

    def test_mixed_content(self):
        paras = [
            ParagraphScore(index=0, ai_probability=0.2, verdict="Likely Human", word_count=30, text="t"),
            ParagraphScore(index=1, ai_probability=0.8, verdict="Likely AI", word_count=40, text="t"),
            ParagraphScore(index=2, ai_probability=0.1, verdict="Likely Human", word_count=35, text="t"),
        ]
        report = format_paragraph_report(paras)
        assert "Mixed content detected" in report
        assert "2" in report  # paragraph 2 flagged

    def test_no_ai_paragraphs(self):
        paras = [
            ParagraphScore(index=0, ai_probability=0.1, verdict="Likely Human", word_count=30, text="t"),
            ParagraphScore(index=1, ai_probability=0.2, verdict="Likely Human", word_count=40, text="t"),
        ]
        report = format_paragraph_report(paras)
        assert "Mixed content" not in report
        assert "All paragraphs" not in report


# ═══════════════════════════════════════════════════════════════════════════════
#  _signal_icon and _fmt_value — edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalIcon:
    def test_unknown_feature(self):
        icon, label = _signal_icon("unknown_feature", 999.0)
        assert "Normal" in label

    def test_burstiness_very_low(self):
        icon, label = _signal_icon("burstiness", 0.1)
        assert "AI" in label

    def test_burstiness_normal(self):
        icon, label = _signal_icon("burstiness", 0.6)
        assert "human" in label.lower() or "Normal" in label

    def test_slop_high(self):
        icon, label = _signal_icon("slop_word_score", 0.05)
        assert "AI" in label

    def test_slop_normal(self):
        icon, label = _signal_icon("slop_word_score", 0.001)
        assert "Normal" in label


class TestFmtValue:
    def test_percentage_format(self):
        result = _fmt_value("slop_word_score", 0.05)
        assert "5.0%" in result

    def test_decimal_format(self):
        result = _fmt_value("burstiness", 0.3456)
        assert "0.35" in result

    def test_trigram_format(self):
        result = _fmt_value("trigram_repetition", 0.123)
        assert "12.3%" in result


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLIEdgeCases:
    def test_html_format(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("This is sample text for testing the HTML output format of lmscan.")
        code = main(["--file", str(f), "--format", "html"])
        assert code == 0

    def test_mixed_flag(self, tmp_path):
        text = (
            "This is a human-written paragraph about cooking dinner.\n\n"
            "Furthermore, it is important to note that the comprehensive "
            "landscape of innovative solutions leverages synergy. "
            "This holistic approach fosters robust paradigms.\n\n"
            "Another normal paragraph about going to the park."
        )
        f = tmp_path / "mixed.txt"
        f.write_text(text)
        code = main(["--file", str(f), "--mixed"])
        assert code == 0

    def test_dir_json_format(self, tmp_path):
        (tmp_path / "a.txt").write_text("Hello world, this is a test file with enough words.")
        (tmp_path / "b.txt").write_text("Another test file for batch scanning with lmscan tool.")
        code = main(["--dir", str(tmp_path), "--format", "json"])
        assert code == 0

    def test_language_flag(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("This is a simple English text for testing language detection.")
        code = main(["--file", str(f), "--language", "en"])
        assert code == 0

    def test_dir_not_found(self):
        code = main(["--dir", "/nonexistent/path/xyz"])
        assert code == 1

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        code = main(["--file", str(f)])
        assert code == 1

    def test_threshold_pass(self, tmp_path):
        f = tmp_path / "human.txt"
        f.write_text("I went to the store yesterday. The weather was nice and sunny. "
                      "My dog ran around the yard while I cooked some pasta. "
                      "Nothing special happened but it was a good day overall.")
        code = main(["--file", str(f), "--threshold", "0.99"])
        assert code == 0

    def test_sentences_flag(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("First sentence here. Second one follows. Third for good measure.")
        code = main(["--file", str(f), "--sentences"])
        assert code == 0

    def test_direct_text_input(self):
        code = main(["This is a direct text input for testing the CLI with positional argument."])
        assert code == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  scan_mixed edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestScanMixed:
    def test_single_paragraph(self):
        result, paras = scan_mixed("Just a single paragraph of text here.")
        assert len(paras) <= 1

    def test_multi_paragraph(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result, paras = scan_mixed(text)
        assert len(paras) == 3

    def test_empty_text(self):
        result, paras = scan_mixed("")
        assert len(paras) == 0

    def test_paragraph_scores_sum_to_reasonable_range(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one too."
        _, paras = scan_mixed(text)
        for p in paras:
            assert 0.0 <= p.ai_probability <= 1.0

    def test_whitespace_only(self):
        result, paras = scan_mixed("   \n\n   \n\n   ")
        assert len(paras) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature extraction edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEdgeCases:
    def test_unicode_text(self):
        text = "日本語のテストです。これはテストです。"
        features = extract_features(text)
        assert features.word_count >= 0

    def test_very_long_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 500
        features = extract_features(text)
        assert features.word_count > 4000

    def test_single_word(self):
        features = extract_features("hello")
        assert features.word_count == 1
        assert features.sentence_count <= 1

    def test_all_punctuation(self):
        features = extract_features("!!! ??? ... ---")
        # Some punctuation combos tokenize as words (--- -> ---)
        assert features.word_count <= 1

    def test_numbers_only(self):
        features = extract_features("123 456 789")
        assert isinstance(features, TextFeatures)

    def test_newlines_only(self):
        features = extract_features("\n\n\n")
        assert features.word_count == 0

    def test_mixed_languages(self):
        text = "Hello world. Bonjour le monde. Hola mundo."
        features = extract_features(text)
        assert features.sentence_count >= 3

    def test_special_characters(self):
        text = "Email me at user@example.com! Call 555-1234."
        features = extract_features(text)
        assert features.word_count > 0

    def test_tab_separated(self):
        text = "word1\tword2\tword3"
        features = extract_features(text)
        assert features.word_count >= 3

    def test_empty_produces_zeros(self):
        f = extract_features("")
        assert f.entropy == 0.0
        assert f.burstiness == 0.0
        assert f.vocabulary_richness == 0.0
        assert f.word_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  _split_sentences edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitSentencesEdge:
    def test_abbreviation_handling(self):
        text = "Dr. Smith went to the store. He bought milk."
        sents = _split_sentences(text)
        assert len(sents) == 2  # "Dr." should not split

    def test_multiple_abbreviations(self):
        text = "Mr. and Mrs. Smith met Dr. Jones at St. Mary's."
        sents = _split_sentences(text)
        assert len(sents) == 1

    def test_ellipsis(self):
        text = "Well... I don't know. Maybe."
        sents = _split_sentences(text)
        assert len(sents) >= 1

    def test_exclamation_and_question(self):
        text = "Really? Yes! Definitely."
        sents = _split_sentences(text)
        assert len(sents) == 3

    def test_no_terminal_punctuation(self):
        text = "This sentence has no ending punctuation"
        sents = _split_sentences(text)
        assert len(sents) == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  _split_paragraphs edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitParagraphsEdge:
    def test_single_newline_no_split(self):
        text = "Line one.\nLine two."
        paras = _split_paragraphs(text)
        assert len(paras) == 1

    def test_triple_newline(self):
        text = "Para one.\n\n\nPara two."
        paras = _split_paragraphs(text)
        assert len(paras) == 2

    def test_whitespace_between_paragraphs(self):
        text = "Para one.\n  \nPara two."
        paras = _split_paragraphs(text)
        assert len(paras) == 2


# ═══════════════════════════════════════════════════════════════════════════════
#  Individual feature functions — boundary values
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureBoundaries:
    def test_entropy_single_unique_word(self):
        # All same word → entropy = 0
        assert word_entropy("hello hello hello") == 0.0

    def test_entropy_all_unique(self):
        text = "alpha beta gamma delta epsilon"
        e = word_entropy(text)
        assert e > 0.0

    def test_burstiness_single_sentence(self):
        assert burstiness("One sentence only.") == 0.0

    def test_vocabulary_richness_single_word(self):
        assert vocabulary_richness("hello") == 1.0

    def test_hapax_all_unique(self):
        text = "one two three four five"
        assert hapax_legomena_ratio(text) == 1.0

    def test_hapax_all_repeated(self):
        text = "one one two two three three"
        assert hapax_legomena_ratio(text) == 0.0

    def test_zipf_deviation_short(self):
        assert zipf_deviation("one two") == 0.0

    def test_sentence_length_variance_single(self):
        assert sentence_length_variance("Just one.") == 0.0

    def test_readability_consistency_single_para(self):
        assert readability_consistency("Just one paragraph.") == 0.0

    def test_bigram_repetition_short(self):
        assert bigram_repetition("one two") == 0.0

    def test_transition_word_ratio_none(self):
        assert transition_word_ratio("apple banana cherry") == 0.0

    def test_transition_word_ratio_high(self):
        text = "However, moreover, furthermore, additionally, consequently."
        ratio = transition_word_ratio(text)
        assert ratio > 0.0

    def test_punctuation_entropy_no_punct(self):
        assert punctuation_entropy("hello world") == 0.0

    def test_punctuation_entropy_single_type(self):
        assert punctuation_entropy("hello... world...") == 0.0  # only "." → entropy 0

    def test_slop_word_score_none(self):
        assert slop_word_score("apple banana cherry") == 0.0

    def test_slop_word_score_present(self):
        text = "We must delve into the tapestry of innovative solutions to leverage synergy."
        score = slop_word_score(text)
        assert score > 0.0

    def test_slop_phrases(self):
        text = "It's important to note that in today's world we must act. It goes without saying that this matters."
        score = slop_word_score(text)
        assert score > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  format_html — direct test
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatHTML:
    def test_produces_valid_html(self):
        result = scan("This is a test sentence for HTML report generation.")
        html = format_html(result)
        assert "<html" in html or "<!DOCTYPE" in html.lower() or "<div" in html
        assert "lmscan" in html.lower() or "AI" in html

    def test_html_contains_score(self):
        result = scan("Testing HTML generation with this sample text for lmscan.")
        html = format_html(result)
        assert "%" in html  # AI probability percentage


# ═══════════════════════════════════════════════════════════════════════════════
#  scan_file edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestScanFile:
    def test_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            scan_file("/nonexistent/path/to/file.txt")

    def test_utf8_file(self, tmp_path):
        f = tmp_path / "utf8.txt"
        f.write_text("Héllo wörld, this is a UTF-8 text file with accents.", encoding="utf-8")
        result = scan_file(str(f))
        assert result.features.word_count > 0

    def test_large_file(self, tmp_path):
        f = tmp_path / "large.txt"
        f.write_text("The quick brown fox jumps over the lazy dog. " * 1000)
        result = scan_file(str(f))
        assert result.features.word_count > 5000


# ═══════════════════════════════════════════════════════════════════════════════
#  scan_directory edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestScanDirectory:
    def test_empty_directory(self, tmp_path):
        results = scan_directory(str(tmp_path))
        assert results == []

    def test_non_text_files_skipped(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")
        (tmp_path / "hello.txt").write_text("Hello world test text.")
        results = scan_directory(str(tmp_path))
        fnames = [r[0] for r in results]
        assert "image.png" not in fnames

    def test_nested_directory(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested file test content.")
        results = scan_directory(str(tmp_path))
        # Should find nested files
        assert len(results) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
#  format_directory_report edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatDirectoryReport:
    def test_empty_results(self):
        report = format_directory_report([])
        assert "0 files scanned" in report

    def test_with_dirname(self):
        report = format_directory_report([], dirname="/my/dir")
        assert "/my/dir" in report


# ═══════════════════════════════════════════════════════════════════════════════
#  Detector edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectorEdge:
    def test_very_short_text(self):
        result = scan("Hi.")
        assert 0.0 <= result.ai_probability <= 1.0

    def test_repeated_text(self):
        text = "word " * 500
        result = scan(text)
        assert isinstance(result.ai_probability, float)

    def test_scan_returns_all_fields(self):
        result = scan("Testing that all fields are populated in scan result.")
        assert result.verdict is not None
        assert result.features is not None
        assert result.scan_time_s >= 0
        assert isinstance(result.sentence_scores, list)
        assert isinstance(result.model_attribution, list)
        assert isinstance(result.flags, list)

    def test_high_ai_text(self):
        text = (
            "It is important to note that the comprehensive landscape of "
            "innovative solutions leverages holistic synergy. Furthermore, "
            "the multifaceted paradigm fosters robust and cutting-edge "
            "transformative approaches. Moreover, this groundbreaking endeavor "
            "streamlines the pivotal interplay of nuanced methodologies."
        )
        result = scan(text)
        assert result.ai_probability > 0.3  # Should detect slop markers
