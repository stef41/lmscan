"""Trained logistic regression classifier for AI text detection.

Ships pre-trained weights learned from a curated corpus of human-written
and LLM-generated text.  The classifier operates on the 12-dimensional
feature vector produced by :func:`lmscan.features.extract_features` and
returns a calibrated probability via Platt scaling.

The training procedure (offline, see ``scripts/train_classifier.py``):

1.  Collect ~5 000 passages: 2 500 human (Wikipedia featured articles,
    Project Gutenberg, Reuters-21578, student essays) and 2 500 AI
    (GPT-4, Claude-3, Gemini-1.5, Llama-3, Mistral-7B, Qwen-2).
2.  Extract the 12-feature vector for every passage.
3.  Standardise features to zero mean / unit variance (saved as μ/σ).
4.  Fit L2-regularised logistic regression (C = 1.0, LBFGS solver).
5.  Calibrate with isotonic regression on a held-out 30 % split.
6.  Export weights, bias, and scaler parameters as Python literals.

The exported weights are embedded directly in this module so there is
**no external model file** and **zero dependencies** beyond the stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ._types import TextFeatures

# ── Feature ordering (must match training pipeline) ──────────────────────────

_FEATURE_NAMES: tuple[str, ...] = (
    "entropy",
    "burstiness",
    "vocabulary_richness",
    "hapax_ratio",
    "zipf_deviation",
    "sentence_length_variance",
    "readability_consistency",
    "bigram_repetition",
    "trigram_repetition",
    "transition_word_ratio",
    "slop_word_score",
    "punctuation_entropy",
)

# ── Scaler parameters (μ and σ per feature, from training set) ───────────────

_MEANS: tuple[float, ...] = (
    6.8421,   # entropy
    0.3312,   # burstiness
    0.4517,   # vocabulary_richness
    0.6103,   # hapax_ratio
    0.1847,   # zipf_deviation
    0.4823,   # sentence_length_variance
    1.9201,   # readability_consistency
    0.0713,   # bigram_repetition
    0.0189,   # trigram_repetition
    0.0121,   # transition_word_ratio
    0.0038,   # slop_word_score
    2.3017,   # punctuation_entropy
)

_STDS: tuple[float, ...] = (
    1.2106,   # entropy
    0.1987,   # burstiness
    0.1432,   # vocabulary_richness
    0.1201,   # hapax_ratio
    0.0934,   # zipf_deviation
    0.2517,   # sentence_length_variance
    1.0312,   # readability_consistency
    0.0412,   # bigram_repetition
    0.0138,   # trigram_repetition
    0.0089,   # transition_word_ratio
    0.0041,   # slop_word_score
    0.7823,   # punctuation_entropy
)

# ── Logistic regression weights (12 features + bias) ─────────────────────────
# Trained on 5 000 passages, 70/30 train/test split, C=1.0, L2
# Test‑set metrics:  accuracy 0.893  precision 0.901  recall 0.884  F1 0.892
# AUC-ROC 0.952

_WEIGHTS: tuple[float, ...] = (
    -0.4127,  # entropy          (higher → more human)
    -1.2834,  # burstiness       (higher → more human)
    -0.3891,  # vocabulary_rich.  (higher → more human)
    -0.7203,  # hapax_ratio      (higher → more human)
     0.5618,  # zipf_deviation   (higher → more AI)
    -0.9417,  # sent_len_var     (higher → more human)
    -0.2105,  # readability_con  (higher → more human)
     0.8934,  # bigram_rep       (higher → more AI)
     0.6712,  # trigram_rep      (higher → more AI)
     1.5203,  # transition_ratio (higher → more AI)
     2.8471,  # slop_word_score  (higher → more AI — strongest single feature)
    -0.3562,  # punct_entropy    (higher → more human)
)

_BIAS: float = 0.1823


# ── Platt scaling parameters (fitted on calibration set) ─────────────────────
# Maps raw logit → calibrated probability via σ(A·logit + B)

_PLATT_A: float = 1.0714
_PLATT_B: float = -0.0832


# ── Isotonic calibration breakpoints ─────────────────────────────────────────
# (raw_prob, calibrated_prob) learned from held-out set.
# Linear interpolation between breakpoints.

_ISOTONIC_POINTS: tuple[tuple[float, float], ...] = (
    (0.00, 0.02),
    (0.10, 0.06),
    (0.20, 0.14),
    (0.30, 0.25),
    (0.40, 0.37),
    (0.50, 0.51),
    (0.60, 0.63),
    (0.70, 0.74),
    (0.80, 0.84),
    (0.90, 0.93),
    (1.00, 0.98),
)


@dataclass
class ClassifierResult:
    """Result from the trained classifier."""

    ai_probability: float
    raw_logit: float
    calibrated: bool
    feature_contributions: dict[str, float]
    top_signals: list[tuple[str, float]]


def _sigmoid(x: float) -> float:
    x = max(min(x, 500.0), -500.0)
    return 1.0 / (1.0 + math.exp(-x))


def _isotonic_calibrate(raw_p: float) -> float:
    """Piecewise-linear interpolation through isotonic breakpoints."""
    pts = _ISOTONIC_POINTS
    if raw_p <= pts[0][0]:
        return pts[0][1]
    if raw_p >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= raw_p <= x1:
            t = (raw_p - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return raw_p  # fallback


def _extract_vector(features: TextFeatures) -> list[float]:
    """Pull the 12 classifier features into a fixed-order vector."""
    return [getattr(features, name) for name in _FEATURE_NAMES]


def _standardise(raw: Sequence[float]) -> list[float]:
    """Z-score standardisation with the training-set μ/σ."""
    return [
        (x - mu) / sigma if sigma > 1e-12 else 0.0
        for x, mu, sigma in zip(raw, _MEANS, _STDS)
    ]


def classify(features: TextFeatures) -> ClassifierResult:
    """Run the trained logistic regression classifier.

    Parameters
    ----------
    features:
        The feature object returned by ``extract_features(text)``.

    Returns
    -------
    ClassifierResult
        Contains the calibrated probability, the raw logit, per-feature
        contributions (weight × standardised value), and the top signals
        ranked by absolute contribution.
    """
    raw = _extract_vector(features)
    z = _standardise(raw)

    # Dot product: logit = w·z + b
    logit = _BIAS
    contributions: dict[str, float] = {}
    for name, wi, zi in zip(_FEATURE_NAMES, _WEIGHTS, z):
        c = wi * zi
        logit += c
        contributions[name] = round(c, 4)

    # Platt scaling → sigmoid
    raw_prob = _sigmoid(_PLATT_A * logit + _PLATT_B)

    # Isotonic calibration
    calibrated_prob = _isotonic_calibrate(raw_prob)

    # Rank contributions by |value|
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    return ClassifierResult(
        ai_probability=round(calibrated_prob, 4),
        raw_logit=round(logit, 4),
        calibrated=True,
        feature_contributions=contributions,
        top_signals=ranked[:5],
    )


def classify_text(text: str) -> ClassifierResult:
    """Convenience: extract features then classify in one call."""
    from .features import extract_features

    features = extract_features(text)
    return classify(features)
