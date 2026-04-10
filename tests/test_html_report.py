"""Tests for HTML report generation."""

from __future__ import annotations

from lmscan.scanner import scan
from lmscan.report import format_html
from lmscan._types import ScanResult, TextFeatures, ModelMatch, SentenceScore


def _make_result(ai_prob: float, **kwargs) -> ScanResult:
    """Create a ScanResult with the given AI probability."""
    return ScanResult(
        text="Test text for report.",
        ai_probability=ai_prob,
        verdict="AI-generated" if ai_prob >= 0.85 else "Likely AI" if ai_prob >= 0.65 else "Mixed" if ai_prob >= 0.40 else "Human-written",
        confidence="medium",
        features=TextFeatures(word_count=50, sentence_count=3),
        sentence_scores=kwargs.get("sentence_scores", []),
        model_attribution=kwargs.get("model_attribution", []),
        flags=kwargs.get("flags", []),
        scan_time_s=0.01,
    )


def test_format_html_contains_doctype():
    result = _make_result(0.75)
    html = format_html(result)
    assert "<!DOCTYPE html>" in html


def test_format_html_contains_probability():
    result = _make_result(0.82)
    html = format_html(result)
    assert "82%" in html


def test_format_html_contains_features():
    result = _make_result(0.50)
    html = format_html(result)
    assert "Feature Breakdown" in html
    assert "Burstiness" in html
    assert "Slop word density" in html


def test_format_html_verdict():
    result = _make_result(0.90)
    html = format_html(result)
    assert "AI-generated" in html


def test_format_html_model_attribution():
    match = ModelMatch(model="GPT-4 / ChatGPT", confidence=0.75, evidence=['"delve"', '"tapestry"'], marker_count=2)
    result = _make_result(0.80, model_attribution=[match])
    html = format_html(result)
    assert "Model Attribution" in html
    assert "GPT-4" in html
    assert "75%" in html


def test_format_html_color_coding_red():
    result = _make_result(0.85)
    html = format_html(result)
    # High AI probability should use red color
    assert "#f44336" in html


def test_format_html_color_coding_green():
    result = _make_result(0.20)
    html = format_html(result)
    # Low AI probability should use green color
    assert "#4caf50" in html


def test_format_html_no_external_deps():
    """HTML should not reference any external CSS or JS files."""
    result = _make_result(0.50)
    html = format_html(result)
    assert "http://" not in html
    assert "https://" not in html
    assert '<link rel="stylesheet"' not in html
    assert "<script src=" not in html


def test_format_html_sentences():
    ss = SentenceScore(text="This is AI text.", ai_probability=0.9, features={}, flags=["slop"])
    result = _make_result(0.80, sentence_scores=[ss])
    html = format_html(result)
    assert "Per-Sentence Analysis" in html
    assert "This is AI text." in html


def test_format_html_dark_theme():
    result = _make_result(0.50)
    html = format_html(result)
    # Check dark theme background color
    assert "#1a1a2e" in html
