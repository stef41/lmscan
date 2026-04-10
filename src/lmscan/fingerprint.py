from __future__ import annotations

import math
from collections import Counter

from ._types import ModelMatch
from .features import _tokenize

# ── Model vocabulary profiles ─────────────────────────────────────────────────

_MODEL_PROFILES: dict[str, dict] = {
    "GPT-4 / ChatGPT": {
        "vocabulary": [
            "delve", "tapestry", "beacon", "landscape", "foster", "leverage",
            "holistic", "synergy", "multifaceted", "nuanced", "pivotal",
            "cutting-edge", "groundbreaking", "game-changer", "unleash",
            "harness", "spearhead", "interplay", "underscores", "underscore",
            "comprehensive", "streamline", "realm",
        ],
        "phrases": [
            "it's important to note", "it's worth noting",
            "let's dive in", "in today's rapidly",
            "at the end of the day", "the landscape of",
            "navigate the complexities", "in the realm of",
        ],
        "hedging": [
            "it depends on", "there are several", "one approach is",
        ],
        "weight": 1.0,
    },
    "Claude (Anthropic)": {
        "vocabulary": [
            "certainly", "absolutely", "straightforward", "robust",
            "nuanced", "comprehensive", "thoughtful", "meaningful",
        ],
        "phrases": [
            "i'd be happy to", "great question", "that's a great",
            "let me", "here's", "i should note", "i want to be",
            "i appreciate", "it's worth", "happy to help",
            "i don't have", "i can't",
        ],
        "hedging": [
            "i should mention", "to be transparent", "it's worth noting",
        ],
        "weight": 1.0,
    },
    "Gemini (Google)": {
        "vocabulary": [
            "crucial", "essential", "furthermore", "additionally",
            "significantly", "effectively", "efficiently", "optimize",
        ],
        "phrases": [
            "here are some", "here's a breakdown",
            "you can", "to achieve this",
            "it's important to", "keep in mind",
            "there are several ways", "one way to",
        ],
        "hedging": [
            "i'm a large language model", "as an ai",
            "i don't have personal", "i can not",
        ],
        "weight": 1.0,
    },
    "Llama / Meta": {
        "vocabulary": [
            "awesome", "fantastic", "crucial", "vital",
            "robust", "scalable", "optimize", "implement",
        ],
        "phrases": [
            "here is", "here are", "note that",
            "make sure to", "don't forget to",
            "you can also", "for example",
        ],
        "hedging": [
            "i hope this helps", "let me know if",
            "hope that helps",
        ],
        "weight": 0.8,
    },
    "Mistral / Mixtral": {
        "vocabulary": [
            "indeed", "moreover", "therefore", "thus",
            "hence", "consequently", "noteworthy",
        ],
        "phrases": [
            "it is worth", "one should", "it should be noted",
            "in this context", "to this end",
        ],
        "hedging": [
            "as a language model", "i cannot",
        ],
        "weight": 0.7,
    },
}


def fingerprint(text: str) -> list[ModelMatch]:
    """Identify which LLM likely generated *text*.

    Returns a list of :class:`ModelMatch` sorted by confidence (highest first).
    Only models with confidence > 0.05 are included.
    """
    if not text or not text.strip():
        return []

    text_lower = text.lower()
    words = _tokenize(text)
    word_counts = Counter(words)

    raw_scores: dict[str, float] = {}
    evidences: dict[str, list[str]] = {}
    marker_counts: dict[str, int] = {}

    for model, profile in _MODEL_PROFILES.items():
        vocab_hits: list[tuple[str, int]] = []
        phrase_hits: list[str] = []
        hedge_hits: list[str] = []

        # Vocabulary matches (word-level, case-insensitive)
        for v in profile["vocabulary"]:
            v_lower = v.lower()
            # Handle hyphenated words: check both forms
            variants = [v_lower]
            if "-" in v_lower:
                variants.append(v_lower.replace("-", ""))
            for variant in variants:
                cnt = word_counts.get(variant, 0)
                if cnt > 0:
                    vocab_hits.append((v, cnt))
                    break

        # Phrase matches (substring, case-insensitive)
        for phrase in profile["phrases"]:
            if phrase.lower() in text_lower:
                phrase_hits.append(phrase)

        # Hedging matches
        for hedge in profile["hedging"]:
            if hedge.lower() in text_lower:
                hedge_hits.append(hedge)

        total_markers = len(vocab_hits) + len(phrase_hits) + len(hedge_hits)
        raw = (
            sum(cnt for _, cnt in vocab_hits) * 2
            + len(phrase_hits) * 3
            + len(hedge_hits) * 5
        ) * profile["weight"]

        raw_scores[model] = raw
        marker_counts[model] = total_markers

        # Build evidence strings
        ev: list[str] = []
        for word, cnt in vocab_hits:
            ev.append(f'"{word}" (\u00d7{cnt})' if cnt > 1 else f'"{word}"')
        for phrase in phrase_hits:
            ev.append(f'"{phrase}"')
        for hedge in hedge_hits:
            ev.append(f'[hedge] "{hedge}"')
        evidences[model] = ev

    # ── Softmax normalisation ─────────────────────────────────────────────
    total_raw = sum(raw_scores.values())
    if total_raw == 0:
        return []

    # Use softmax with temperature to spread out probabilities
    max_raw = max(raw_scores.values())
    exp_scores = {}
    for model, score in raw_scores.items():
        exp_scores[model] = math.exp((score - max_raw) / max(total_raw * 0.3, 1.0))

    exp_sum = sum(exp_scores.values())
    confidences = {m: e / exp_sum for m, e in exp_scores.items()}

    # ── Build results ─────────────────────────────────────────────────────
    results: list[ModelMatch] = []
    for model in sorted(confidences, key=confidences.get, reverse=True):  # type: ignore[arg-type]
        conf = confidences[model]
        if conf < 0.05:
            continue
        results.append(ModelMatch(
            model=model,
            confidence=round(conf, 4),
            evidence=evidences[model],
            marker_count=marker_counts[model],
        ))

    return results


def identify_slop_phrases(text: str) -> list[tuple[str, int]]:
    """Return all detected AI slop phrases with their positions in *text*."""
    from .features import _SLOP_PHRASES, _SLOP_SINGLE

    if not text:
        return []

    text_lower = text.lower()
    results: list[tuple[str, int]] = []

    # Multi-word phrases
    for phrase in _SLOP_PHRASES:
        start = 0
        while True:
            idx = text_lower.find(phrase, start)
            if idx == -1:
                break
            results.append((phrase, idx))
            start = idx + 1

    # Single words
    for match in __import__("re").finditer(r"\b\w+(?:-\w+)*\b", text_lower):
        word = match.group()
        if word in _SLOP_SINGLE:
            results.append((word, match.start()))

    results.sort(key=lambda x: x[1])
    return results
