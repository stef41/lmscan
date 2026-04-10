from __future__ import annotations

from lmscan.detector import detect_paragraphs
from lmscan.scanner import scan_mixed
from lmscan._types import ParagraphScore, ScanResult


_HUMAN_PARA = (
    "I went to the grocery store yesterday and bought some apples. "
    "The cashier was really friendly and we chatted about the weather."
)
_AI_PARA = (
    "Let's delve into the multifaceted tapestry of innovation. Furthermore, "
    "it's important to note that a holistic approach is crucial for fostering "
    "comprehensive synergy across the entire landscape of modern endeavors."
)
_MIXED_TEXT = f"{_HUMAN_PARA}\n\n{_AI_PARA}\n\n{_HUMAN_PARA}"


def test_detect_paragraphs_basic():
    paras = detect_paragraphs(_MIXED_TEXT)
    assert len(paras) == 3
    assert all(isinstance(p, ParagraphScore) for p in paras)


def test_detect_paragraphs_mixed_content():
    paras = detect_paragraphs(_MIXED_TEXT)
    # Each paragraph should have a non-negative probability
    for p in paras:
        assert 0.0 <= p.ai_probability <= 1.0


def test_detect_paragraphs_have_verdicts():
    paras = detect_paragraphs(_MIXED_TEXT)
    for p in paras:
        assert isinstance(p.verdict, str)
        assert p.verdict != ""


def test_detect_paragraphs_indices():
    paras = detect_paragraphs(_MIXED_TEXT)
    indices = [p.index for p in paras]
    assert indices == [0, 1, 2]


def test_detect_paragraphs_word_count():
    paras = detect_paragraphs(_MIXED_TEXT)
    for p in paras:
        assert p.word_count > 0


def test_single_paragraph():
    paras = detect_paragraphs(_HUMAN_PARA)
    assert len(paras) == 1
    assert paras[0].index == 0


def test_empty_text():
    paras = detect_paragraphs("")
    assert paras == []


def test_whitespace_only():
    paras = detect_paragraphs("   \n\n   \n\n   ")
    assert paras == []


def test_scan_mixed_returns_both():
    result, paras = scan_mixed(_MIXED_TEXT)
    assert isinstance(result, ScanResult)
    assert isinstance(paras, list)
    assert len(paras) > 0


def test_scan_mixed_result_matches_scan():
    from lmscan.scanner import scan

    result_mixed, _ = scan_mixed(_MIXED_TEXT)
    result_plain = scan(_MIXED_TEXT)
    assert result_mixed.ai_probability == result_plain.ai_probability


def test_paragraph_scores_have_verdicts():
    _, paras = scan_mixed(_MIXED_TEXT)
    valid_verdicts = {"AI-generated", "Likely AI", "Mixed", "Likely human", "Human-written"}
    for p in paras:
        assert p.verdict in valid_verdicts


def test_paragraph_text_preserved():
    paras = detect_paragraphs(_MIXED_TEXT)
    for p in paras:
        assert len(p.text) > 0
        assert p.text.strip() == p.text
