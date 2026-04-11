"""Tests for the trained logistic regression classifier."""

from __future__ import annotations

import pytest

from lmscan.classifier import (
    ClassifierResult,
    classify,
    classify_text,
    _sigmoid,
    _isotonic_calibrate,
    _extract_vector,
    _standardise,
    _FEATURE_NAMES,
    _MEANS,
    _STDS,
    _WEIGHTS,
    _BIAS,
)
from lmscan.features import extract_features
from lmscan._types import TextFeatures


# ── Fixtures ─────────────────────────────────────────────────────────────────

HUMAN_TEXT = (
    "The cat sat on the mat, licking its paws. Outside, rain hammered "
    "the windows — fat drops that burst on the glass like tiny grenades. "
    "I poured another coffee. The third one today? Fourth? Didn't matter. "
    "What mattered was the deadline. Three days. Two chapters. And a brain "
    "that wouldn't cooperate."
)

AI_TEXT = (
    "In today's rapidly evolving landscape, it is important to note that "
    "artificial intelligence represents a transformative paradigm shift. "
    "This comprehensive analysis delves into the multifaceted implications "
    "of leveraging cutting-edge technology to foster innovative solutions. "
    "Furthermore, the holistic approach to harnessing these robust tools "
    "underscores the pivotal role of streamlined methodologies in driving "
    "groundbreaking advancements across the realm of modern computing."
)


# ── Unit tests ───────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_midpoint(self):
        assert _sigmoid(0.0) == pytest.approx(0.5, abs=1e-6)

    def test_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_overflow_protection(self):
        # Should not raise even with extreme values
        assert 0.0 <= _sigmoid(1000.0) <= 1.0
        assert 0.0 <= _sigmoid(-1000.0) <= 1.0

    def test_monotonic(self):
        vals = [_sigmoid(x) for x in range(-10, 11)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]


class TestIsotonicCalibrate:
    def test_boundary_low(self):
        assert _isotonic_calibrate(0.0) == pytest.approx(0.02, abs=0.01)

    def test_boundary_high(self):
        assert _isotonic_calibrate(1.0) == pytest.approx(0.98, abs=0.01)

    def test_midpoint(self):
        result = _isotonic_calibrate(0.5)
        assert 0.45 <= result <= 0.55

    def test_below_range(self):
        assert _isotonic_calibrate(-0.1) == pytest.approx(0.02, abs=0.01)

    def test_above_range(self):
        assert _isotonic_calibrate(1.1) == pytest.approx(0.98, abs=0.01)

    def test_interpolation(self):
        result = _isotonic_calibrate(0.15)
        assert 0.06 <= result <= 0.14

    def test_monotonic(self):
        vals = [_isotonic_calibrate(x / 100) for x in range(101)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]


class TestFeatureExtraction:
    def test_vector_length(self):
        features = extract_features("Hello world, this is a test.")
        vec = _extract_vector(features)
        assert len(vec) == 12

    def test_vector_matches_names(self):
        features = extract_features("some text here for testing purposes.")
        vec = _extract_vector(features)
        for i, name in enumerate(_FEATURE_NAMES):
            assert vec[i] == getattr(features, name)

    def test_standardise_zeros_at_mean(self):
        # If input equals the means, standardised should be all zeros
        z = _standardise(list(_MEANS))
        for zi in z:
            assert abs(zi) < 1e-6


class TestParameterConsistency:
    def test_means_length(self):
        assert len(_MEANS) == 12

    def test_stds_length(self):
        assert len(_STDS) == 12

    def test_weights_length(self):
        assert len(_WEIGHTS) == 12

    def test_stds_positive(self):
        for s in _STDS:
            assert s > 0

    def test_feature_names_match_textfeatures(self):
        f = TextFeatures()
        for name in _FEATURE_NAMES:
            assert hasattr(f, name), f"TextFeatures missing {name}"


class TestClassify:
    def test_returns_classifier_result(self):
        features = extract_features(HUMAN_TEXT)
        result = classify(features)
        assert isinstance(result, ClassifierResult)

    def test_probability_in_range(self):
        features = extract_features(HUMAN_TEXT)
        result = classify(features)
        assert 0.0 <= result.ai_probability <= 1.0

    def test_has_feature_contributions(self):
        features = extract_features(HUMAN_TEXT)
        result = classify(features)
        assert len(result.feature_contributions) == 12
        for name in _FEATURE_NAMES:
            assert name in result.feature_contributions

    def test_top_signals_ranked(self):
        features = extract_features(HUMAN_TEXT)
        result = classify(features)
        assert len(result.top_signals) <= 5
        # Check sorted by absolute value descending
        abs_vals = [abs(v) for _, v in result.top_signals]
        for i in range(1, len(abs_vals)):
            assert abs_vals[i] <= abs_vals[i - 1]

    def test_calibrated_flag(self):
        features = extract_features(HUMAN_TEXT)
        result = classify(features)
        assert result.calibrated is True

    def test_human_text_lower_probability(self):
        human_result = classify(extract_features(HUMAN_TEXT))
        ai_result = classify(extract_features(AI_TEXT))
        # AI-laden text should score higher
        assert ai_result.ai_probability > human_result.ai_probability

    def test_ai_text_higher_probability(self):
        result = classify(extract_features(AI_TEXT))
        # Text stuffed with slop words should score as AI
        assert result.ai_probability > 0.5


class TestClassifyText:
    def test_convenience_function(self):
        result = classify_text("Some text to classify for testing purposes.")
        assert isinstance(result, ClassifierResult)
        assert 0.0 <= result.ai_probability <= 1.0

    def test_empty_text(self):
        result = classify_text("")
        assert isinstance(result, ClassifierResult)

    def test_short_text(self):
        result = classify_text("Hello.")
        assert 0.0 <= result.ai_probability <= 1.0


class TestClassifierOnVariousTexts:
    """Test that the classifier produces reasonable outputs on diverse inputs."""

    @pytest.mark.parametrize("text", [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
    ])
    def test_pangrams(self, text):
        result = classify_text(text)
        assert 0.0 <= result.ai_probability <= 1.0

    def test_numeric_text(self):
        result = classify_text("123 456 789 0.12 3.45 67.89")
        assert 0.0 <= result.ai_probability <= 1.0

    def test_repeated_text(self):
        result = classify_text("hello " * 50)
        assert 0.0 <= result.ai_probability <= 1.0

    def test_mixed_case(self):
        result = classify_text(
            "NASA announced that the ISP for WiFi was updated. "
            "CEO of FAANG said OK."
        )
        assert 0.0 <= result.ai_probability <= 1.0
