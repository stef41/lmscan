"""Tests for lmscan.calibration module."""

from __future__ import annotations

from lmscan.calibration import (
    CalibrationResult,
    ThresholdConfig,
    calibrate,
    find_optimal_threshold,
)


# ── Sample texts ──────────────────────────────────────────────────────────────

# Clearly AI-like text (lots of slop words, uniform structure)
_AI_TEXT = (
    "It's important to note that leveraging holistic synergy is pivotal in today's world. "
    "Furthermore, the multifaceted landscape of innovation fosters comprehensive growth. "
    "Additionally, harnessing cutting-edge paradigms streamlines the transformative process. "
    "Moreover, delving into the nuanced tapestry of robust solutions underscores interplay. "
    "Consequently, groundbreaking endeavors facilitate the realm of game-changer strategies."
)

# Clearly human text (varied, casual, no slop)
_HUMAN_TEXT = (
    "I went to the store yesterday but forgot my wallet. Typical. "
    "The dog chased a squirrel up the old oak tree and barked for twenty minutes straight. "
    "My neighbor's kid drew a picture of what I think was a dinosaur? Hard to tell. "
    "Anyway, dinner was cold by the time I got home. Not great."
)


def test_calibrate_perfect_predictions():
    """With a well-separated dataset the metrics should be reasonable."""
    samples = [
        (_AI_TEXT, True),
        (_HUMAN_TEXT, False),
    ]
    result = calibrate(samples, threshold=0.5)
    total = result.true_positive + result.true_negative + result.false_positive + result.false_negative
    assert total == 2


def test_calibrate_returns_valid_metrics():
    samples = [(_AI_TEXT, True), (_HUMAN_TEXT, False)]
    result = calibrate(samples, threshold=0.5)
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.accuracy <= 1.0


def test_calibrate_empty_samples():
    result = calibrate([], threshold=0.5)
    assert result.true_positive == 0
    assert result.true_negative == 0
    assert result.false_positive == 0
    assert result.false_negative == 0
    assert result.accuracy == 0.0
    assert result.f1 == 0.0


def test_find_optimal_threshold():
    samples = [
        (_AI_TEXT, True),
        (_HUMAN_TEXT, False),
    ]
    threshold, result = find_optimal_threshold(samples, steps=10)
    assert 0.3 <= threshold <= 0.8
    assert isinstance(result, CalibrationResult)


def test_calibration_result_properties():
    cr = CalibrationResult(true_positive=5, true_negative=3, false_positive=2, false_negative=1)
    assert cr.precision == 5 / 7
    assert cr.recall == 5 / 6
    expected_f1 = 2 * (5 / 7) * (5 / 6) / ((5 / 7) + (5 / 6))
    assert abs(cr.f1 - expected_f1) < 1e-10
    assert cr.accuracy == 8 / 11


def test_threshold_config_defaults():
    config = ThresholdConfig()
    assert config.ai_threshold == 0.65
    assert config.feature_weights == {}


def test_threshold_config_custom():
    config = ThresholdConfig(ai_threshold=0.5, feature_weights={"burstiness": 0.3})
    assert config.ai_threshold == 0.5
    assert config.feature_weights["burstiness"] == 0.3


def test_calibrate_all_ai():
    samples = [(_AI_TEXT, True), (_AI_TEXT, True)]
    result = calibrate(samples, threshold=0.1)
    # With very low threshold, all should be predicted as AI
    assert result.true_positive + result.false_negative == 2
    assert result.false_positive + result.true_negative == 0


def test_calibrate_all_human():
    samples = [(_HUMAN_TEXT, False), (_HUMAN_TEXT, False)]
    result = calibrate(samples, threshold=0.5)
    assert result.true_positive + result.false_negative == 0
    total = result.true_negative + result.false_positive
    assert total == 2


def test_precision_recall_calculated():
    cr = CalibrationResult(true_positive=10, false_positive=5, false_negative=2, true_negative=8)
    assert cr.precision == 10 / 15
    assert cr.recall == 10 / 12


def test_f1_zero_when_no_positives():
    cr = CalibrationResult(true_positive=0, false_positive=0, false_negative=0, true_negative=5)
    assert cr.f1 == 0.0
    assert cr.precision == 0.0
    assert cr.recall == 0.0
