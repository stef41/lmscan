from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextFeatures:
    """Statistical features extracted from text."""

    entropy: float = 0.0
    burstiness: float = 0.0
    vocabulary_richness: float = 0.0
    hapax_ratio: float = 0.0
    zipf_deviation: float = 0.0
    sentence_length_variance: float = 0.0
    readability_consistency: float = 0.0
    bigram_repetition: float = 0.0
    trigram_repetition: float = 0.0
    transition_word_ratio: float = 0.0
    slop_word_score: float = 0.0
    punctuation_entropy: float = 0.0
    # v0.4 advanced features
    passive_voice_ratio: float = 0.0
    sentence_opening_diversity: float = 0.0
    lexical_density: float = 0.0
    char_entropy: float = 0.0
    hedging_density: float = 0.0
    conjunction_start_ratio: float = 0.0
    # v0.6 features
    contraction_rate: float = 0.0
    first_person_ratio: float = 0.0
    question_ratio: float = 0.0
    list_pattern_density: float = 0.0
    long_ngram_repetition: float = 0.0
    chatbot_marker_score: float = 0.0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    paragraph_count: int = 0
    sentence_count: int = 0
    word_count: int = 0


@dataclass
class SentenceScore:
    """AI probability for a single sentence."""

    text: str
    ai_probability: float
    features: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


@dataclass
class ModelMatch:
    """Attribution result for a specific LLM."""

    model: str
    confidence: float
    evidence: list[str]
    marker_count: int = 0


@dataclass
class ParagraphScore:
    """AI probability for a single paragraph."""

    text: str
    index: int
    ai_probability: float
    verdict: str
    word_count: int


@dataclass
class ScanResult:
    """Complete scan result for a piece of text."""

    text: str
    ai_probability: float
    verdict: str
    confidence: str
    features: TextFeatures = field(default_factory=TextFeatures)
    sentence_scores: list[SentenceScore] = field(default_factory=list)
    model_attribution: list[ModelMatch] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    scan_time_s: float = 0.0


# ── Chat / dialog analysis ────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    """A single normalised chat turn."""

    text: str
    speaker: str
    index: int = 0
    role: str = ""


@dataclass
class MessageScore:
    """AI probability for a single chat message."""

    index: int
    speaker: str
    text: str
    ai_probability: float
    verdict: str = ""


@dataclass
class ParticipantScore:
    """Aggregated AI verdict for one chat participant."""

    name: str
    message_count: int = 0
    word_count: int = 0
    ai_probability: float = 0.0
    verdict: str = ""
    confidence: str = ""
    is_ai: bool = False
    model_attribution: list[ModelMatch] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


@dataclass
class ChatScanResult:
    """Complete scan result for a chat dialog transcript."""

    source: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    message_scores: list[MessageScore] = field(default_factory=list)
    participants: list[ParticipantScore] = field(default_factory=list)
    likely_bots: list[str] = field(default_factory=list)
    verdict: str = ""
    ai_probability: float = 0.0
    scan_time_s: float = 0.0
