from __future__ import annotations

import math
import re
import string
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
    "Qwen / Qwen2": {
        "vocabulary": [
            "elucidate", "paradigm", "multifaceted", "nuanced",
            "comprehensive", "facilitate", "leverage", "robust",
        ],
        "phrases": [
            "i'd be happy to help", "let me think about this step by step",
            "i hope this helps", "to provide a comprehensive",
        ],
        "hedging": [
            "to some extent", "it's worth noting",
            "it should be noted", "broadly speaking",
        ],
        "weight": 0.9,
    },
    "DeepSeek": {
        "vocabulary": [
            "synergistic", "holistic", "paradigmatic", "leverage",
            "optimize", "comprehensive", "robust", "facilitate",
        ],
        "phrases": [
            "let me analyze", "based on my analysis",
            "let me break this down", "here's my analysis",
        ],
        "hedging": [
            "it should be noted", "one could argue",
            "it's important to consider", "from a broader perspective",
        ],
        "weight": 0.9,
    },
    "Cohere / Command R": {
        "vocabulary": [
            "pivotal", "quintessential", "nuanced", "streamline",
            "comprehensive", "innovative", "facilitate",
        ],
        "phrases": [
            "here's what i found", "based on the information",
            "i can help with that", "here are the key points",
        ],
        "hedging": [
            "that said", "having said that",
            "it's worth mentioning", "on the other hand",
        ],
        "weight": 0.8,
    },
    "Phi / Phi-3": {
        "vocabulary": [
            "fundamentally", "essentially", "critically", "notably",
            "significantly", "effectively", "optimize",
        ],
        "phrases": [
            "the key point is", "to summarize",
            "in short", "the main idea",
        ],
        "hedging": [
            "in essence", "broadly speaking",
            "generally speaking", "for the most part",
        ],
        "weight": 0.7,
    },
}


# ── Structural patterns per model family ──────────────────────────────────────
# Based on empirical observation of LLM output tendencies.

_MODEL_STRUCTURE: dict[str, dict[str, float]] = {
    "GPT-4 / ChatGPT": {
        "avg_sentence_length": 22.0,   # GPT-4 favours long, balanced sentences
        "passive_bias": 0.25,          # moderate passive use
        "list_tendency": 0.3,          # sometimes uses lists
        "paragraph_uniformity": 0.8,   # very uniform paragraph sizes
    },
    "Claude (Anthropic)": {
        "avg_sentence_length": 18.0,   # shorter, more conversational
        "passive_bias": 0.15,
        "list_tendency": 0.2,
        "paragraph_uniformity": 0.6,   # more varied
    },
    "Gemini (Google)": {
        "avg_sentence_length": 20.0,
        "passive_bias": 0.2,
        "list_tendency": 0.5,          # loves bullet lists
        "paragraph_uniformity": 0.7,
    },
    "Llama / Meta": {
        "avg_sentence_length": 16.0,   # shorter, punchier
        "passive_bias": 0.1,
        "list_tendency": 0.4,
        "paragraph_uniformity": 0.5,
    },
    "Mistral / Mixtral": {
        "avg_sentence_length": 21.0,
        "passive_bias": 0.3,           # tends formal/passive
        "list_tendency": 0.2,
        "paragraph_uniformity": 0.75,
    },
    "Qwen / Qwen2": {
        "avg_sentence_length": 23.0,   # longest sentences of all
        "passive_bias": 0.25,
        "list_tendency": 0.3,
        "paragraph_uniformity": 0.7,
    },
    "DeepSeek": {
        "avg_sentence_length": 22.0,
        "passive_bias": 0.2,
        "list_tendency": 0.35,
        "paragraph_uniformity": 0.7,
    },
    "Cohere / Command R": {
        "avg_sentence_length": 19.0,
        "passive_bias": 0.15,
        "list_tendency": 0.5,
        "paragraph_uniformity": 0.65,
    },
    "Phi / Phi-3": {
        "avg_sentence_length": 17.0,
        "passive_bias": 0.15,
        "list_tendency": 0.3,
        "paragraph_uniformity": 0.6,
    },
}

# ── Punctuation and formatting style profiles per model ────────────────────────
# Each model has characteristic punctuation patterns:
# - em_dash_rate: usage of em-dashes (—) per 1000 chars
# - semicolon_rate: usage of semicolons per 1000 chars
# - colon_rate: usage of colons per 1000 chars
# - exclamation_rate: usage of exclamation marks per 1000 chars
# - parenthetical_rate: usage of parentheses per 1000 chars
# - comma_density: commas per sentence (avg)

_PUNCTUATION_PROFILES: dict[str, dict[str, float]] = {
    "GPT-4 / ChatGPT": {
        "em_dash_rate": 1.8,     # moderate em-dash usage
        "semicolon_rate": 0.8,
        "colon_rate": 1.5,
        "exclamation_rate": 0.3,
        "parenthetical_rate": 1.2,
        "comma_density": 2.8,
    },
    "Claude (Anthropic)": {
        "em_dash_rate": 3.2,     # Claude loves em-dashes
        "semicolon_rate": 0.5,
        "colon_rate": 2.0,
        "exclamation_rate": 0.4,
        "parenthetical_rate": 1.8,
        "comma_density": 2.5,
    },
    "Gemini (Google)": {
        "em_dash_rate": 0.5,
        "semicolon_rate": 0.3,
        "colon_rate": 2.5,
        "exclamation_rate": 0.5,
        "parenthetical_rate": 0.8,
        "comma_density": 2.2,
    },
    "Llama / Meta": {
        "em_dash_rate": 0.8,
        "semicolon_rate": 0.4,
        "colon_rate": 1.8,
        "exclamation_rate": 1.0,
        "parenthetical_rate": 0.6,
        "comma_density": 2.0,
    },
    "Mistral / Mixtral": {
        "em_dash_rate": 1.2,
        "semicolon_rate": 1.5,    # Mistral uses more semicolons
        "colon_rate": 1.2,
        "exclamation_rate": 0.2,
        "parenthetical_rate": 0.8,
        "comma_density": 3.0,
    },
    "Qwen / Qwen2": {
        "em_dash_rate": 0.6,
        "semicolon_rate": 0.6,
        "colon_rate": 1.5,
        "exclamation_rate": 0.3,
        "parenthetical_rate": 1.0,
        "comma_density": 2.6,
    },
    "DeepSeek": {
        "em_dash_rate": 0.4,
        "semicolon_rate": 0.5,
        "colon_rate": 2.2,
        "exclamation_rate": 0.2,
        "parenthetical_rate": 1.5,
        "comma_density": 2.4,
    },
    "Cohere / Command R": {
        "em_dash_rate": 2.0,
        "semicolon_rate": 0.6,
        "colon_rate": 1.8,
        "exclamation_rate": 0.5,
        "parenthetical_rate": 1.0,
        "comma_density": 2.3,
    },
    "Phi / Phi-3": {
        "em_dash_rate": 0.3,
        "semicolon_rate": 0.3,
        "colon_rate": 1.0,
        "exclamation_rate": 0.4,
        "parenthetical_rate": 0.5,
        "comma_density": 1.8,
    },
}


def _punctuation_style_score(text: str, model: str) -> float:
    """Score how well the text's punctuation style matches a model profile."""
    profile = _PUNCTUATION_PROFILES.get(model)
    if not profile or len(text) < 50:
        return 0.0

    from .features import _split_sentences

    n_chars = len(text)
    k = 1000.0 / max(n_chars, 1)

    # Count punctuation marks
    em_dashes = text.count("\u2014") + text.count(" -- ") + text.count(" - ")
    semicolons = text.count(";")
    colons = text.count(":")
    exclamations = text.count("!")
    parens = text.count("(") + text.count(")")

    actual = {
        "em_dash_rate": em_dashes * k,
        "semicolon_rate": semicolons * k,
        "colon_rate": colons * k,
        "exclamation_rate": exclamations * k,
        "parenthetical_rate": parens * k,
    }

    # Comma density per sentence
    sentences = _split_sentences(text)
    if sentences:
        comma_count = text.count(",")
        actual["comma_density"] = comma_count / len(sentences)
    else:
        actual["comma_density"] = 0.0

    # Score: inverse of normalized absolute difference
    total_match = 0.0
    n_metrics = 0
    for metric, expected in profile.items():
        act = actual.get(metric, 0.0)
        if expected > 0:
            match = max(0.0, 1.0 - abs(act - expected) / max(expected, 0.5))
        else:
            match = 1.0 if act < 0.5 else 0.0
        total_match += match
        n_metrics += 1

    return total_match / max(n_metrics, 1)


def _structural_score(text: str, model: str) -> float:
    """Score how well the text's structure matches a model's typical patterns."""
    profile = _MODEL_STRUCTURE.get(model)
    if not profile:
        return 0.0

    from .features import _split_sentences, _tokenize, passive_voice_ratio

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return 0.0

    # Sentence length match
    lengths = [len(_tokenize(s)) for s in sentences if _tokenize(s)]
    if not lengths:
        return 0.0
    actual_avg = sum(lengths) / len(lengths)
    expected_avg = profile["avg_sentence_length"]
    len_match = max(0.0, 1.0 - abs(actual_avg - expected_avg) / expected_avg)

    # Passive voice match
    actual_passive = passive_voice_ratio(text)
    expected_passive = profile["passive_bias"]
    passive_match = max(0.0, 1.0 - abs(actual_passive - expected_passive) / max(expected_passive, 0.1))

    # Paragraph uniformity match
    from .features import _split_paragraphs
    paras = _split_paragraphs(text)
    if len(paras) >= 2:
        para_lens = [len(_tokenize(p)) for p in paras if _tokenize(p)]
        if para_lens:
            mean_pl = sum(para_lens) / len(para_lens)
            if mean_pl > 0:
                cv = (sum((l - mean_pl)**2 for l in para_lens) / len(para_lens))**0.5 / mean_pl
                actual_uniformity = max(0.0, 1.0 - cv)
            else:
                actual_uniformity = 0.5
        else:
            actual_uniformity = 0.5
    else:
        actual_uniformity = 0.5
    expected_uniformity = profile["paragraph_uniformity"]
    uniformity_match = max(0.0, 1.0 - abs(actual_uniformity - expected_uniformity))

    # Weighted combination
    return 0.4 * len_match + 0.3 * passive_match + 0.3 * uniformity_match


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
    word_set = set(words)

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

        # Vocabulary scoring: TF-IDF-inspired weighting
        # Rare markers (appearing in fewer model profiles) are worth more
        vocab_score = 0.0
        for word, cnt in vocab_hits:
            # Count how many models share this marker
            shared = sum(
                1 for m, p in _MODEL_PROFILES.items()
                if word.lower() in [v.lower() for v in p["vocabulary"]]
            )
            idf = math.log(len(_MODEL_PROFILES) / max(shared, 1)) + 1.0
            vocab_score += cnt * idf

        raw = (
            vocab_score * 2
            + len(phrase_hits) * 4
            + len(hedge_hits) * 6
        ) * profile["weight"]

        # Add structural pattern bonus
        struct_bonus = _structural_score(text, model)
        raw += struct_bonus * 3.0 * profile["weight"]

        # Add punctuation style bonus
        punct_bonus = _punctuation_style_score(text, model)
        raw += punct_bonus * 2.0 * profile["weight"]

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
