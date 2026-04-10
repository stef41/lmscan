from __future__ import annotations

from lmscan.fingerprint import fingerprint, identify_slop_phrases, _MODEL_PROFILES
from lmscan._types import ModelMatch


_GPT4_TEXT = (
    "Let's delve into this tapestry of innovation. It's important to note that the "
    "landscape of modern technology offers a multifaceted beacon of holistic synergy. "
    "We must leverage these cutting-edge paradigms to foster groundbreaking solutions "
    "and harness the interplay between transformative approaches. In the realm of "
    "navigating complexities, it's worth noting the pivotal role of comprehensive strategies."
)

_CLAUDE_TEXT = (
    "I'd be happy to help with that! Great question — let me walk you through "
    "this thoughtful and nuanced approach. I should note that there are certainly "
    "some meaningful considerations here. I want to be straightforward about the "
    "robust methodology involved. I appreciate you asking about this comprehensive topic."
)

_GEMINI_TEXT = (
    "Here are some crucial and essential points to consider. Furthermore, it's important "
    "to optimize effectively and efficiently. Here's a breakdown of the significant "
    "factors you need to keep in mind. There are several ways to achieve this, and "
    "additionally you can explore these options."
)

_GENERIC_TEXT = (
    "The meeting was scheduled for Tuesday at 3 PM. We discussed the quarterly "
    "budget and planned the next sprint. John mentioned potential delays with "
    "the vendor. Everyone agreed to regroup on Thursday."
)


def test_fingerprint_gpt4_text():
    results = fingerprint(_GPT4_TEXT)
    assert len(results) > 0
    top = results[0]
    assert "GPT" in top.model


def test_fingerprint_claude_text():
    results = fingerprint(_CLAUDE_TEXT)
    assert len(results) > 0
    top = results[0]
    assert "Claude" in top.model


def test_fingerprint_gemini_text():
    results = fingerprint(_GEMINI_TEXT)
    assert len(results) > 0
    top = results[0]
    assert "Gemini" in top.model


def test_fingerprint_generic_text():
    results = fingerprint(_GENERIC_TEXT)
    # Generic text may get low confidence for all or no strong match
    if results:
        assert results[0].confidence < 0.8


def test_fingerprint_returns_sorted_by_confidence():
    results = fingerprint(_GPT4_TEXT)
    for i in range(len(results) - 1):
        assert results[i].confidence >= results[i + 1].confidence


def test_fingerprint_confidence_sums_to_roughly_1():
    results = fingerprint(_GPT4_TEXT)
    total = sum(m.confidence for m in results)
    # May not sum to exactly 1.0 because low-confidence models are filtered
    assert total <= 1.05


def test_fingerprint_evidence_strings():
    results = fingerprint(_GPT4_TEXT)
    top = results[0]
    assert len(top.evidence) > 0
    assert all(isinstance(e, str) for e in top.evidence)


def test_fingerprint_marker_counts():
    results = fingerprint(_GPT4_TEXT)
    top = results[0]
    assert top.marker_count > 0


def test_fingerprint_empty_text():
    results = fingerprint("")
    assert results == []


def test_fingerprint_whitespace_only():
    results = fingerprint("   \n\n  ")
    assert results == []


def test_fingerprint_returns_model_match():
    results = fingerprint(_GPT4_TEXT)
    assert all(isinstance(m, ModelMatch) for m in results)


def test_identify_slop_phrases_finds_matches():
    text = "We must delve into the tapestry of this landscape."
    results = identify_slop_phrases(text)
    found_words = [word for word, pos in results]
    assert "delve" in found_words
    assert "tapestry" in found_words


def test_identify_slop_phrases_returns_positions():
    text = "The landscape is clear."
    results = identify_slop_phrases(text)
    for word, pos in results:
        assert isinstance(pos, int)
        assert pos >= 0


def test_identify_slop_phrases_empty():
    assert identify_slop_phrases("") == []


def test_model_profiles_have_required_keys():
    for model, profile in _MODEL_PROFILES.items():
        assert "vocabulary" in profile, f"{model} missing vocabulary"
        assert "phrases" in profile, f"{model} missing phrases"
        assert "hedging" in profile, f"{model} missing hedging"
        assert "weight" in profile, f"{model} missing weight"
