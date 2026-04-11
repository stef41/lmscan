"""ROC / AUC evaluation utilities for the lmscan detector.

Computes Receiver Operating Characteristic curves and Area Under Curve
from labeled text samples.  Provides proper statistical evaluation
including confidence intervals via DeLong's method (asymptotic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ROCPoint:
    """A single point on the ROC curve."""

    threshold: float
    tpr: float  # true positive rate (sensitivity / recall)
    fpr: float  # false positive rate (1 - specificity)
    precision: float
    f1: float


@dataclass
class ROCResult:
    """Full ROC analysis result."""

    auc: float
    auc_ci_lower: float  # 95% CI lower bound
    auc_ci_upper: float  # 95% CI upper bound
    curve: list[ROCPoint]
    optimal_threshold: float  # maximises Youden's J
    optimal_f1_threshold: float  # maximises F1
    n_positive: int
    n_negative: int
    eer: float  # equal error rate


def _compute_roc_curve(
    scores: Sequence[float],
    labels: Sequence[int],
) -> list[ROCPoint]:
    """Compute ROC curve from raw scores and binary labels.

    Parameters
    ----------
    scores:
        Predicted AI probabilities (higher = more AI-like).
    labels:
        Ground truth: 1 = AI, 0 = human.

    Returns
    -------
    List of ROCPoint at each unique threshold, sorted by threshold desc.
    """
    paired = sorted(zip(scores, labels), reverse=True)

    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return []

    points: list[ROCPoint] = []
    tp = 0
    fp = 0

    prev_score = float("inf")

    for score, label in paired:
        if score != prev_score:
            tpr = tp / total_pos
            fpr = fp / total_neg
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
            points.append(ROCPoint(
                threshold=prev_score if prev_score != float("inf") else score + 0.01,
                tpr=tpr,
                fpr=fpr,
                precision=precision,
                f1=f1,
            ))
            prev_score = score

        if label == 1:
            tp += 1
        else:
            fp += 1

    # Final point
    tpr = tp / total_pos
    fpr = fp / total_neg
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    points.append(ROCPoint(
        threshold=prev_score,
        tpr=tpr,
        fpr=fpr,
        precision=precision,
        f1=f1,
    ))

    return points


def _trapezoidal_auc(points: list[ROCPoint]) -> float:
    """Compute AUC using the trapezoidal rule."""
    if len(points) < 2:
        return 0.5

    auc = 0.0
    for i in range(1, len(points)):
        dx = points[i].fpr - points[i - 1].fpr
        avg_y = (points[i].tpr + points[i - 1].tpr) / 2
        auc += dx * avg_y

    return auc


def _delong_variance(
    scores: Sequence[float],
    labels: Sequence[int],
    auc: float,
) -> float:
    """Estimate AUC variance using DeLong's method (1988).

    Simplified implementation for confidence interval estimation.
    """
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    m = len(pos_scores)
    n = len(neg_scores)

    if m < 2 or n < 2:
        return 0.0

    # Placement values for positive samples
    v_pos: list[float] = []
    for ps in pos_scores:
        count = sum(1 for ns in neg_scores if ps > ns) + \
                0.5 * sum(1 for ns in neg_scores if ps == ns)
        v_pos.append(count / n)

    # Placement values for negative samples
    v_neg: list[float] = []
    for ns in neg_scores:
        count = sum(1 for ps in pos_scores if ps > ns) + \
                0.5 * sum(1 for ps in pos_scores if ps == ns)
        v_neg.append(count / m)

    # Variance components
    s_pos = sum((v - auc) ** 2 for v in v_pos) / (m - 1) if m > 1 else 0.0
    s_neg = sum((v - auc) ** 2 for v in v_neg) / (n - 1) if n > 1 else 0.0

    return s_pos / m + s_neg / n


def _find_eer(points: list[ROCPoint]) -> float:
    """Find the Equal Error Rate (where FPR ≈ FNR)."""
    best_eer = 1.0
    for pt in points:
        fnr = 1.0 - pt.tpr
        eer = (pt.fpr + fnr) / 2
        if abs(pt.fpr - fnr) < abs(best_eer * 2 - 1):
            best_eer = eer
    return best_eer


def compute_roc(
    scores: Sequence[float],
    labels: Sequence[int],
    confidence: float = 0.95,
) -> ROCResult:
    """Compute full ROC analysis with AUC and confidence intervals.

    Parameters
    ----------
    scores:
        Predicted probabilities (higher = more AI-like).
    labels:
        Ground truth binary labels (1 = AI, 0 = human).
    confidence:
        Confidence level for AUC CI (default 95%).

    Returns
    -------
    ROCResult
        AUC, confidence interval, ROC curve, optimal thresholds, EER.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    curve = _compute_roc_curve(scores, labels)

    if not curve:
        return ROCResult(
            auc=0.5,
            auc_ci_lower=0.0,
            auc_ci_upper=1.0,
            curve=[],
            optimal_threshold=0.5,
            optimal_f1_threshold=0.5,
            n_positive=n_pos,
            n_negative=n_neg,
            eer=0.5,
        )

    auc = _trapezoidal_auc(curve)
    variance = _delong_variance(scores, labels, auc)

    # z-score for confidence level
    # For 95%: z = 1.96
    alpha = 1.0 - confidence
    z = 1.96  # Approximate; exact for 95%
    if confidence == 0.99:
        z = 2.576
    elif confidence == 0.90:
        z = 1.645

    std_err = math.sqrt(variance) if variance > 0 else 0.0
    ci_lower = max(0.0, auc - z * std_err)
    ci_upper = min(1.0, auc + z * std_err)

    # Optimal threshold: maximises Youden's J = TPR - FPR
    best_j = -1.0
    optimal_thresh = 0.5
    for pt in curve:
        j = pt.tpr - pt.fpr
        if j > best_j:
            best_j = j
            optimal_thresh = pt.threshold

    # Optimal F1 threshold
    best_f1 = -1.0
    f1_thresh = 0.5
    for pt in curve:
        if pt.f1 > best_f1:
            best_f1 = pt.f1
            f1_thresh = pt.threshold

    eer = _find_eer(curve)

    return ROCResult(
        auc=round(auc, 4),
        auc_ci_lower=round(ci_lower, 4),
        auc_ci_upper=round(ci_upper, 4),
        curve=curve,
        optimal_threshold=round(optimal_thresh, 4),
        optimal_f1_threshold=round(f1_thresh, 4),
        n_positive=n_pos,
        n_negative=n_neg,
        eer=round(eer, 4),
    )
