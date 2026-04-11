from __future__ import annotations

from lmscan.detector import detect
from lmscan._types import ScanResult, TextFeatures


# ── AI detection patterns ────────────────────────────────────────────────────

_AI_TEXT = (
    "In today's rapidly evolving landscape, it's important to note that the multifaceted "
    "nature of modern technology requires a holistic approach. We must delve into the tapestry "
    "of innovation and leverage cutting-edge solutions to foster robust paradigms. Moreover, "
    "the interplay between synergy and transformative strategies underscores the pivotal role "
    "of comprehensive frameworks. Furthermore, it is essential to harness the groundbreaking "
    "potential of these innovative approaches. Additionally, navigating the complexities of "
    "this realm requires nuanced understanding and streamlined methodologies."
)

_HUMAN_TEXT = (
    "I went to the store yesterday. Got some milk and eggs. Weather was awful — "
    "rained the whole time. My kid wouldn't stop asking for candy so I caved and "
    "got him a chocolate bar. The cashier was really slow and the line was long. "
    "Drove home listening to the radio. Made scrambled eggs for dinner."
)

_MIXED_TEXT = (
    "The conference was held in Berlin last October. Several researchers presented their "
    "findings on climate modeling. One team showed promising results using satellite data. "
    "Another group focused on ocean temperature patterns. The discussions were lively and "
    "went on past the scheduled time. Coffee breaks were too short."
)


def test_detect_obvious_ai_text():
    result = detect(_AI_TEXT)
    assert result.ai_probability > 0.50


def test_detect_obvious_human_text():
    result = detect(_HUMAN_TEXT)
    assert result.ai_probability < 0.50


def test_detect_mixed_text():
    result = detect(_MIXED_TEXT)
    assert 0.20 <= result.ai_probability <= 0.80


def test_verdict_ai_generated():
    result = detect(_AI_TEXT)
    assert result.verdict in ("AI-generated", "Likely AI", "Mixed")


def test_verdict_human_written():
    result = detect(_HUMAN_TEXT)
    assert result.verdict in ("Human-written", "Likely human", "Mixed")


def test_verdict_thresholds():
    # Verify all verdict strings exist via the function
    from lmscan.detector import _verdict
    assert _verdict(0.90) == "AI-generated"
    assert _verdict(0.70) == "Likely AI"
    assert _verdict(0.50) == "Mixed"
    assert _verdict(0.30) == "Likely human"
    assert _verdict(0.10) == "Human-written"


def test_confidence_by_length():
    from lmscan.detector import _confidence
    short = TextFeatures(word_count=10)
    medium = TextFeatures(word_count=100)
    long_ = TextFeatures(word_count=500)
    assert _confidence(short) == "low"
    assert _confidence(medium) == "medium"
    assert _confidence(long_) == "high"


def test_flags_generated_for_ai_text():
    result = detect(_AI_TEXT)
    assert len(result.flags) > 0


def test_sentence_scores_length():
    result = detect(_AI_TEXT)
    assert len(result.sentence_scores) > 0


def test_detect_empty_text():
    result = detect("")
    assert isinstance(result, ScanResult)
    assert result.ai_probability >= 0.0


def test_detect_single_sentence():
    result = detect("Hello there.")
    assert isinstance(result, ScanResult)


def test_detect_returns_scan_result():
    result = detect("Some text here for testing purposes.")
    assert isinstance(result, ScanResult)
    assert isinstance(result.features, TextFeatures)


def test_scan_result_has_features():
    result = detect(_AI_TEXT)
    f = result.features
    assert f.word_count > 0
    assert f.sentence_count > 0


def test_scan_time_recorded():
    result = detect(_AI_TEXT)
    assert result.scan_time_s >= 0.0


def test_generate_flags_slop():
    from lmscan.detector import _generate_flags
    feats = TextFeatures(
        slop_word_score=0.05,
        burstiness=0.5,
        sentence_length_variance=0.5,
        transition_word_ratio=0.0,
        readability_consistency=3.0,
        bigram_repetition=0.0,
        hapax_ratio=0.7,
        word_count=100,
    )
    flags = _generate_flags(feats, 0.7)
    assert any("slop" in f.lower() for f in flags)


def test_generate_flags_empty():
    from lmscan.detector import _generate_flags
    feats = TextFeatures(
        slop_word_score=0.0,
        burstiness=0.8,
        sentence_length_variance=0.8,
        transition_word_ratio=0.0,
        readability_consistency=3.0,
        bigram_repetition=0.0,
        hapax_ratio=0.7,
        word_count=10,
    )
    flags = _generate_flags(feats, 0.3)
    assert isinstance(flags, list)
