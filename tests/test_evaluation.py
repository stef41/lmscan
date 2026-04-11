"""Tests for the ROC/AUC evaluation module."""

from __future__ import annotations

import pytest

from lmscan.evaluation import (
    ROCResult,
    ROCPoint,
    compute_roc,
    _compute_roc_curve,
    _trapezoidal_auc,
    _delong_variance,
    _find_eer,
)


# ── Perfect classifier ──────────────────────────────────────────────────────

class TestPerfectClassifier:
    def test_auc_one(self):
        scores = [0.9, 0.8, 0.7, 0.1, 0.05, 0.01]
        labels = [1, 1, 1, 0, 0, 0]
        result = compute_roc(scores, labels)
        assert result.auc == pytest.approx(1.0, abs=0.01)

    def test_ci_near_one(self):
        scores = [0.95, 0.9, 0.85, 0.8, 0.1, 0.05, 0.03, 0.01]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        result = compute_roc(scores, labels)
        assert result.auc_ci_lower > 0.8

    def test_optimal_threshold(self):
        scores = [0.9, 0.8, 0.1, 0.05]
        labels = [1, 1, 0, 0]
        result = compute_roc(scores, labels)
        # Should pick a threshold between 0.1 and 0.8
        assert 0.1 <= result.optimal_threshold <= 0.9


# ── Random classifier ───────────────────────────────────────────────────────

class TestRandomClassifier:
    def test_auc_near_half(self):
        # Interleaved scores and labels → ~0.5 AUC
        scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]
        labels = [1, 1, 0, 0, 1, 1, 0, 0]
        result = compute_roc(scores, labels)
        assert 0.3 <= result.auc <= 0.7


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            compute_roc([0.5, 0.5], [1])

    def test_all_positive(self):
        result = compute_roc([0.9, 0.8, 0.7], [1, 1, 1])
        assert isinstance(result, ROCResult)

    def test_all_negative(self):
        result = compute_roc([0.1, 0.2, 0.3], [0, 0, 0])
        assert isinstance(result, ROCResult)

    def test_two_samples(self):
        result = compute_roc([0.9, 0.1], [1, 0])
        assert isinstance(result, ROCResult)
        assert result.auc == pytest.approx(1.0, abs=0.01)

    def test_tied_scores(self):
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [1, 0, 1, 0]
        result = compute_roc(scores, labels)
        assert isinstance(result, ROCResult)


# ── ROC curve properties ────────────────────────────────────────────────────

class TestROCCurve:
    def test_curve_not_empty(self):
        scores = [0.9, 0.8, 0.3, 0.1]
        labels = [1, 1, 0, 0]
        result = compute_roc(scores, labels)
        assert len(result.curve) > 0

    def test_tpr_fpr_in_range(self):
        scores = [0.9, 0.7, 0.4, 0.2]
        labels = [1, 1, 0, 0]
        result = compute_roc(scores, labels)
        for pt in result.curve:
            assert 0.0 <= pt.tpr <= 1.0
            assert 0.0 <= pt.fpr <= 1.0

    def test_precision_in_range(self):
        scores = [0.9, 0.7, 0.4, 0.2]
        labels = [1, 1, 0, 0]
        result = compute_roc(scores, labels)
        for pt in result.curve:
            assert 0.0 <= pt.precision <= 1.0

    def test_f1_in_range(self):
        scores = [0.9, 0.7, 0.4, 0.2]
        labels = [1, 1, 0, 0]
        result = compute_roc(scores, labels)
        for pt in result.curve:
            assert 0.0 <= pt.f1 <= 1.0


# ── AUC computation ─────────────────────────────────────────────────────────

class TestTrapezoidal:
    def test_perfect_auc(self):
        # Perfect ROC: (0,0) → (0,1) → (1,1)
        points = [
            ROCPoint(threshold=1.0, tpr=0.0, fpr=0.0, precision=1.0, f1=0.0),
            ROCPoint(threshold=0.5, tpr=1.0, fpr=0.0, precision=1.0, f1=1.0),
            ROCPoint(threshold=0.0, tpr=1.0, fpr=1.0, precision=0.5, f1=0.67),
        ]
        auc = _trapezoidal_auc(points)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_diagonal_auc(self):
        # Diagonal ROC: (0,0) → (1,1) → AUC = 0.5
        points = [
            ROCPoint(threshold=1.0, tpr=0.0, fpr=0.0, precision=1.0, f1=0.0),
            ROCPoint(threshold=0.0, tpr=1.0, fpr=1.0, precision=0.5, f1=0.67),
        ]
        auc = _trapezoidal_auc(points)
        assert auc == pytest.approx(0.5, abs=0.01)


# ── DeLong variance ─────────────────────────────────────────────────────────

class TestDeLongVariance:
    def test_perfect_separation_low_variance(self):
        scores = [0.95, 0.9, 0.85, 0.1, 0.05, 0.02]
        labels = [1, 1, 1, 0, 0, 0]
        var = _delong_variance(scores, labels, 1.0)
        assert var >= 0
        assert var < 0.1

    def test_insufficient_samples(self):
        var = _delong_variance([0.9], [1], 1.0)
        assert var == 0.0


# ── Equal Error Rate ─────────────────────────────────────────────────────────

class TestEER:
    def test_eer_in_range(self):
        scores = [0.9, 0.8, 0.6, 0.3, 0.2, 0.1]
        labels = [1, 1, 1, 0, 0, 0]
        result = compute_roc(scores, labels)
        assert 0.0 <= result.eer <= 0.5

    def test_perfect_eer_low(self):
        scores = [0.9, 0.8, 0.1, 0.05]
        labels = [1, 1, 0, 0]
        result = compute_roc(scores, labels)
        assert result.eer <= 0.25  # small sample, EER approximation


# ── Confidence intervals ────────────────────────────────────────────────────

class TestConfidenceIntervals:
    def test_ci_bounds(self):
        scores = [0.9, 0.85, 0.8, 0.75, 0.2, 0.15, 0.1, 0.05]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        result = compute_roc(scores, labels)
        assert result.auc_ci_lower <= result.auc
        assert result.auc_ci_upper >= result.auc
        assert 0.0 <= result.auc_ci_lower <= 1.0
        assert 0.0 <= result.auc_ci_upper <= 1.0

    def test_99_percent_wider(self):
        scores = [0.9, 0.85, 0.8, 0.2, 0.15, 0.1]
        labels = [1, 1, 1, 0, 0, 0]
        r95 = compute_roc(scores, labels, confidence=0.95)
        r99 = compute_roc(scores, labels, confidence=0.99)
        w95 = r95.auc_ci_upper - r95.auc_ci_lower
        w99 = r99.auc_ci_upper - r99.auc_ci_lower
        assert w99 >= w95

    def test_n_positive_negative(self):
        scores = [0.9, 0.8, 0.1, 0.05]
        labels = [1, 1, 0, 0]
        result = compute_roc(scores, labels)
        assert result.n_positive == 2
        assert result.n_negative == 2
