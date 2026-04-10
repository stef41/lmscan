from __future__ import annotations

import math
import time

from ._types import ScanResult, SentenceScore, TextFeatures
from .features import (
    extract_features,
    _split_sentences,
    _tokenize,
    slop_word_score as _slop_score,
    transition_word_ratio as _transition_ratio,
)

# ── Signal definitions ────────────────────────────────────────────────────────
# (direction, weight, threshold)
# "low_is_ai"  → values below threshold push toward AI
# "high_is_ai" → values above threshold push toward AI

_SIGNALS: dict[str, tuple[str, float, float]] = {
    "burstiness":              ("low_is_ai",  0.24, 0.20),
    "sentence_length_variance": ("low_is_ai", 0.15, 0.35),
    "slop_word_score":         ("high_is_ai", 0.24, 0.005),
    "readability_consistency": ("low_is_ai",  0.03, 1.5),
    "transition_word_ratio":   ("high_is_ai", 0.12, 0.01),
    "bigram_repetition":       ("high_is_ai", 0.06, 0.08),
    "hapax_ratio":             ("low_is_ai",  0.06, 0.50),
    "zipf_deviation":          ("high_is_ai", 0.05, 0.12),
    "punctuation_entropy":     ("low_is_ai",  0.05, 1.8),
}

# Signals that require enough text structure to be reliable
_CONDITIONAL_SIGNALS: dict[str, str] = {
    "readability_consistency": "paragraph_count",  # needs >= 2 paragraphs
}


def _sigmoid(x: float, sensitivity: float = 3.0) -> float:
    """Standard sigmoid clipped to avoid overflow."""
    z = -sensitivity * x
    z = max(min(z, 500), -500)  # prevent math overflow
    return 1.0 / (1.0 + math.exp(z))


# ── Public API ────────────────────────────────────────────────────────────────

def detect(text: str) -> ScanResult:
    """Detect whether *text* is AI-generated and return a :class:`ScanResult`."""
    t0 = time.monotonic()

    features = extract_features(text)

    # ── Overall AI probability ────────────────────────────────────────────
    ai_prob = _compute_probability(features)

    # ── Per-sentence scores ───────────────────────────────────────────────
    sentence_scores = _score_sentences(text, features)

    elapsed = time.monotonic() - t0

    return ScanResult(
        text=text,
        ai_probability=round(ai_prob, 4),
        verdict=_verdict(ai_prob),
        confidence=_confidence(features),
        features=features,
        sentence_scores=sentence_scores,
        model_attribution=[],  # filled by scanner.scan()
        flags=_generate_flags(features, ai_prob),
        scan_time_s=round(elapsed, 4),
    )


def _compute_probability(features: TextFeatures) -> float:
    """Weighted sigmoid combination of feature signals."""
    total = 0.0
    weight_sum = 0.0
    for feat_name, (direction, weight, threshold) in _SIGNALS.items():
        # Skip signals that need structural features we don't have
        cond = _CONDITIONAL_SIGNALS.get(feat_name)
        if cond and getattr(features, cond, 0) < 2:
            continue

        value = getattr(features, feat_name, 0.0)
        if threshold == 0:
            signal_strength = 0.0
        elif direction == "low_is_ai":
            signal_strength = (threshold - value) / threshold
        else:  # high_is_ai
            signal_strength = (value - threshold) / threshold

        signal_prob = _sigmoid(signal_strength, sensitivity=5.0)
        total += weight * signal_prob
        weight_sum += weight

    # Normalise so skipped signals don't reduce the total
    if weight_sum > 0:
        total = total / weight_sum

    return max(0.0, min(1.0, total))


# ── Per-sentence scoring ─────────────────────────────────────────────────────

def _score_sentences(text: str, features: TextFeatures) -> list[SentenceScore]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    mean_len = features.avg_sentence_length if features.avg_sentence_length else 10.0
    scored: list[SentenceScore] = []

    for sent in sentences:
        words = _tokenize(sent)
        wc = len(words)
        if wc == 0:
            continue

        local_feats: dict[str, float] = {}
        flags: list[str] = []

        # Slop
        slop = _slop_score(sent)
        local_feats["slop"] = round(slop, 4)
        if slop > 0.02:
            flags.append("Contains AI vocabulary markers")

        # Transition
        trans = _transition_ratio(sent)
        local_feats["transition"] = round(trans, 4)
        if trans > 0.04:
            flags.append("Heavy use of transition words")

        # Length deviation from mean
        if mean_len > 0:
            len_dev = abs(wc - mean_len) / mean_len
        else:
            len_dev = 0.0
        local_feats["length_dev"] = round(len_dev, 4)

        # Simple sentence-level AI probability
        sent_prob = 0.0
        sent_prob += 0.4 * _sigmoid((slop - 0.01) / max(0.01, 0.01), 3.0)
        sent_prob += 0.3 * _sigmoid((trans - 0.02) / max(0.02, 0.02), 3.0)
        # Sentences very close to mean length → slightly more AI-like
        uniformity = 1.0 - min(len_dev, 1.0)
        sent_prob += 0.3 * _sigmoid((uniformity - 0.5) / 0.5, 2.0)
        sent_prob = max(0.0, min(1.0, sent_prob))

        scored.append(SentenceScore(
            text=sent,
            ai_probability=round(sent_prob, 4),
            features=local_feats,
            flags=flags,
        ))

    return scored


# ── Verdict / confidence / flags ──────────────────────────────────────────────

def _verdict(prob: float) -> str:
    if prob >= 0.85:
        return "AI-generated"
    if prob >= 0.65:
        return "Likely AI"
    if prob >= 0.40:
        return "Mixed"
    if prob >= 0.20:
        return "Likely human"
    return "Human-written"


def _confidence(features: TextFeatures) -> str:
    if features.word_count > 200:
        return "high"
    if features.word_count > 50:
        return "medium"
    return "low"


def _generate_flags(features: TextFeatures, prob: float) -> list[str]:
    flags: list[str] = []

    if features.burstiness < 0.25:
        flags.append(
            f"Very low burstiness ({features.burstiness:.2f}) "
            "\u2014 AI text is typically more uniform in complexity"
        )
    if features.slop_word_score > 0.005:
        pct = features.slop_word_score * 100
        flags.append(
            f"High slop word density ({pct:.1f}%) "
            "\u2014 contains known AI vocabulary markers"
        )
    if features.sentence_length_variance < 0.25:
        flags.append(
            f"Uniform sentence lengths (CV={features.sentence_length_variance:.2f}) "
            "\u2014 human text typically varies more"
        )
    if features.transition_word_ratio > 0.025:
        pct = features.transition_word_ratio * 100
        flags.append(
            f"Overuse of transition words ({pct:.1f}%) "
            "\u2014 typical of AI-generated text"
        )
    if features.readability_consistency < 1.0 and features.paragraph_count >= 2:
        flags.append(
            f"Very consistent readability across paragraphs (σ={features.readability_consistency:.2f}) "
            "\u2014 human writing varies more"
        )
    if features.bigram_repetition > 0.12:
        pct = features.bigram_repetition * 100
        flags.append(
            f"Elevated bigram repetition ({pct:.0f}%) "
            "\u2014 AI text tends to reuse word pairs"
        )
    if features.hapax_ratio < 0.40 and features.word_count > 50:
        flags.append(
            f"Low hapax legomena ratio ({features.hapax_ratio:.2f}) "
            "\u2014 AI text reuses vocabulary more"
        )

    return flags
