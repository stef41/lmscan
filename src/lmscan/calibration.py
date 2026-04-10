"""Calibration utilities for tuning detection thresholds."""

from __future__ import annotations
from dataclasses import dataclass, field

from ._types import ScanResult


@dataclass
class CalibrationResult:
    """Result of running calibration on labeled samples."""
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        return (self.true_positive + self.true_negative) / total if total else 0.0


@dataclass
class ThresholdConfig:
    """Custom threshold configuration."""
    ai_threshold: float = 0.65
    feature_weights: dict[str, float] = field(default_factory=dict)


def calibrate(samples: list[tuple[str, bool]], threshold: float = 0.65) -> CalibrationResult:
    """Run calibration with labeled samples.

    Args:
        samples: List of (text, is_ai) tuples where is_ai=True means the text IS AI-generated.
        threshold: AI probability threshold (above = AI, below = human).

    Returns:
        CalibrationResult with precision, recall, f1, accuracy.
    """
    from .scanner import scan

    result = CalibrationResult()
    for text, is_ai in samples:
        scan_result = scan(text)
        predicted_ai = scan_result.ai_probability >= threshold
        if is_ai and predicted_ai:
            result.true_positive += 1
        elif not is_ai and not predicted_ai:
            result.true_negative += 1
        elif not is_ai and predicted_ai:
            result.false_positive += 1
        else:
            result.false_negative += 1
    return result


def find_optimal_threshold(samples: list[tuple[str, bool]], steps: int = 20) -> tuple[float, CalibrationResult]:
    """Find the threshold that maximizes F1 score.

    Returns:
        Tuple of (best_threshold, best_calibration_result).
    """
    best_threshold = 0.5
    best_result = CalibrationResult()
    best_f1 = 0.0

    for i in range(steps + 1):
        threshold = 0.3 + (0.5 * i / steps)  # Range 0.3 to 0.8
        result = calibrate(samples, threshold)
        if result.f1 > best_f1:
            best_f1 = result.f1
            best_threshold = threshold
            best_result = result

    return best_threshold, best_result
