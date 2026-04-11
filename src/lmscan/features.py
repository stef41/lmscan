from __future__ import annotations

import math
import re
import string
from collections import Counter

from ._types import TextFeatures

# ── Slop words / phrases heavily overused by LLMs ────────────────────────────

_SLOP_WORDS: set[str] = {
    # GPT-family markers
    "delve", "tapestry", "beacon", "landscape", "realm", "foster",
    "leverage", "holistic", "synergy", "paradigm", "multifaceted",
    "nuanced", "pivotal", "comprehensive", "streamline", "robust",
    "cutting-edge", "groundbreaking", "transformative", "innovative",
    "game-changer", "harness", "unleash", "empower", "navigate",
    "underscore", "underscores", "interplay", "spearhead",
    # Filler / hedging markers
    "it's important to note", "it's worth noting", "it is important to",
    "in today's world", "in the realm of", "at the end of the day",
    "it goes without saying", "needless to say",
    # Academic slop
    "elucidate", "aforementioned", "myriad", "plethora", "endeavor",
    "facilitate", "utilize", "commence", "subsequently", "pertaining to",
    "in light of", "with respect to",
}

# Pre-split into single-word set and multi-word list for efficient matching
_SLOP_SINGLE: set[str] = {w for w in _SLOP_WORDS if " " not in w and "'" not in w}
_SLOP_PHRASES: list[str] = [w for w in _SLOP_WORDS if " " in w or "'" in w]

# ── Transition words / phrases ────────────────────────────────────────────────

_TRANSITION_WORDS: set[str] = {
    "however", "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "specifically", "notably", "importantly", "significantly",
    "ultimately", "essentially", "fundamentally",
}
_TRANSITION_PHRASES: list[str] = [
    "in addition", "on the other hand", "as a result", "in conclusion",
]

# ── Abbreviations that should not trigger sentence splits ─────────────────────

_ABBREVIATIONS: set[str] = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "gen", "gov", "sgt", "cpl", "pvt", "capt", "lt", "col", "maj",
    "vs", "etc", "inc", "ltd", "co", "corp", "dept", "univ",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
    "nov", "dec", "fig", "approx", "vol", "no", "op",
    "i.e", "e.g", "cf",
}

# Pre-compiled patterns — avoids ~250 re.compile() calls per scan()
_ABBREVIATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf"\b({re.escape(abbr)})\.", re.IGNORECASE)
    for abbr in _ABBREVIATIONS
]


# ── Tokenisation helpers ──────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Keeps contractions, strips other punctuation."""
    if not text:
        return []
    words: list[str] = []
    for raw in text.lower().split():
        # Strip leading/trailing punctuation but keep internal apostrophes
        token = raw.strip(string.punctuation.replace("'", "").replace("-", ""))
        if token:
            words.append(token)
    return words


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, handling common abbreviations."""
    if not text or not text.strip():
        return []

    # Protect abbreviations: replace "Dr." → "Dr\x1f" temporarily
    sentinel = "\x1f"
    protected = text
    for pattern in _ABBREVIATION_PATTERNS:
        protected = pattern.sub(lambda m: m.group(1) + sentinel, protected)

    # Split on sentence-ending punctuation followed by space or end-of-string
    raw = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for s in raw:
        s = s.replace(sentinel, ".").strip()
        if s:
            sentences.append(s)
    return sentences


def _split_paragraphs(text: str) -> list[str]:
    """Split on double (or more) newlines."""
    if not text:
        return []
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


# ── Core feature functions ────────────────────────────────────────────────────

def word_entropy(text: str) -> float:
    """Shannon entropy of the word frequency distribution (bits)."""
    words = _tokenize(text)
    if not words:
        return 0.0
    counts = Counter(words)
    n = len(words)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def burstiness(text: str) -> float:
    """Coefficient of variation of per-sentence complexity scores."""
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return 0.0

    scores: list[float] = []
    for sent in sentences:
        words = _tokenize(sent)
        total = len(words)
        if total == 0:
            continue
        unique = len(set(words))
        complexity = (unique / total) * math.log(1 + total)
        scores.append(complexity)

    if len(scores) < 2:
        return 0.0

    mean = sum(scores) / len(scores)
    if mean == 0:
        return 0.0
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(var)
    return std / mean


def vocabulary_richness(text: str) -> float:
    """Type-token ratio with Yule's K correction for length, mapped to 0-1."""
    words = _tokenize(text)
    if not words:
        return 0.0

    n = len(words)
    if n == 1:
        return 1.0

    counts = Counter(words)
    # freq_spectrum: how many words appear exactly i times
    spectrum = Counter(counts.values())

    m1 = n
    m2 = sum(i * i * fi for i, fi in spectrum.items())

    denom = m1 * m1
    if denom == 0:
        return 0.0

    yule_k = 10000.0 * (m2 - m1) / denom

    # Map Yule's K to 0-1: lower K = richer vocabulary
    # Typical K: 0 (maximally rich) to ~200+ (very repetitive)
    richness = 1.0 / (1.0 + yule_k / 100.0)
    return richness


def hapax_legomena_ratio(text: str) -> float:
    """Fraction of words that appear exactly once."""
    words = _tokenize(text)
    if not words:
        return 0.0
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(counts) if counts else 0.0


def zipf_deviation(text: str) -> float:
    """Normalised MSE between actual and ideal Zipf-ranked frequencies."""
    words = _tokenize(text)
    if len(words) < 3:
        return 0.0

    counts = Counter(words)
    freqs = sorted(counts.values(), reverse=True)
    top_freq = freqs[0]
    if top_freq == 0:
        return 0.0

    n = len(freqs)
    mse = 0.0
    for rank_minus_one, actual in enumerate(freqs):
        rank = rank_minus_one + 1
        ideal = top_freq / rank
        diff = (actual - ideal) / top_freq  # normalise by top frequency
        mse += diff * diff
    mse /= n
    return math.sqrt(mse)


def sentence_length_variance(text: str) -> float:
    """Coefficient of variation of sentence lengths (in words)."""
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return 0.0
    lengths = [len(_tokenize(s)) for s in sentences]
    lengths = [l for l in lengths if l > 0]
    if len(lengths) < 2:
        return 0.0

    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    var = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return math.sqrt(var) / mean


def _count_syllables(word: str) -> int:
    """Heuristic syllable counter."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 2:
        return 1

    # Remove trailing silent-e
    if word.endswith("e") and not word.endswith("le"):
        word = word[:-1]
        if not word:
            return 1

    # Count vowel groups
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in "aeiouy"
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Handle -le endings (the original word)
    # Already handled by not stripping -le above

    return max(count, 1)


def _flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid grade level for a text passage."""
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    words = _tokenize(text)
    if not words:
        return 0.0

    total_words = len(words)
    total_sentences = max(len(sentences), 1)
    total_syllables = sum(_count_syllables(w) for w in words)

    return (
        0.39 * (total_words / total_sentences)
        + 11.8 * (total_syllables / total_words)
        - 15.59
    )


def readability_consistency(text: str) -> float:
    """Std dev of per-paragraph Flesch-Kincaid grade levels."""
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) < 2:
        return 0.0

    grades: list[float] = []
    for para in paragraphs:
        words = _tokenize(para)
        if len(words) < 5:
            continue
        grades.append(_flesch_kincaid_grade(para))

    if len(grades) < 2:
        return 0.0

    mean = sum(grades) / len(grades)
    var = sum((g - mean) ** 2 for g in grades) / len(grades)
    return math.sqrt(var)


def bigram_repetition(text: str) -> float:
    """Fraction of word bigrams that appear more than once."""
    words = _tokenize(text)
    if len(words) < 3:
        return 0.0
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    counts = Counter(bigrams)
    if not counts:
        return 0.0
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts)


def trigram_repetition(text: str) -> float:
    """Fraction of word trigrams that appear more than once."""
    words = _tokenize(text)
    if len(words) < 4:
        return 0.0
    trigrams = [
        (words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)
    ]
    counts = Counter(trigrams)
    if not counts:
        return 0.0
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts)


def transition_word_ratio(text: str) -> float:
    """Ratio of transition words / phrases to total words."""
    words = _tokenize(text)
    if not words:
        return 0.0

    text_lower = text.lower()
    total = len(words)
    count = 0

    # Single-word transitions
    word_set = set(words)
    for tw in _TRANSITION_WORDS:
        if tw in word_set:
            # Count actual occurrences
            count += words.count(tw)

    # Multi-word transitions: count substring occurrences
    for phrase in _TRANSITION_PHRASES:
        start = 0
        while True:
            idx = text_lower.find(phrase, start)
            if idx == -1:
                break
            count += len(phrase.split())  # each word in the phrase counts
            start = idx + 1

    return count / total


def punctuation_entropy(text: str) -> float:
    """Shannon entropy of punctuation character distribution."""
    puncts = [ch for ch in text if ch in string.punctuation]
    if not puncts:
        return 0.0
    counts = Counter(puncts)
    n = len(puncts)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def slop_word_score(text: str) -> float:
    """Count AI slop words and phrases, return as fraction of total words."""
    words = _tokenize(text)
    if not words:
        return 0.0

    text_lower = text.lower()
    total = len(words)
    hits = 0

    # Single-word slop
    for w in words:
        if w in _SLOP_SINGLE:
            hits += 1

    # Multi-word slop phrases
    for phrase in _SLOP_PHRASES:
        start = 0
        while True:
            idx = text_lower.find(phrase, start)
            if idx == -1:
                break
            hits += 1
            start = idx + 1

    return hits / total


# ── Master extraction function ────────────────────────────────────────────────

def extract_features(text: str) -> TextFeatures:
    """Extract all statistical features from *text*."""
    words = _tokenize(text)
    sentences = _split_sentences(text)
    paragraphs = _split_paragraphs(text)

    wc = len(words)
    sc = len(sentences)

    avg_wl = (sum(len(w) for w in words) / wc) if wc else 0.0
    avg_sl = (wc / sc) if sc else 0.0

    return TextFeatures(
        entropy=round(word_entropy(text), 6),
        burstiness=round(burstiness(text), 6),
        vocabulary_richness=round(vocabulary_richness(text), 6),
        hapax_ratio=round(hapax_legomena_ratio(text), 6),
        zipf_deviation=round(zipf_deviation(text), 6),
        sentence_length_variance=round(sentence_length_variance(text), 6),
        readability_consistency=round(readability_consistency(text), 6),
        bigram_repetition=round(bigram_repetition(text), 6),
        trigram_repetition=round(trigram_repetition(text), 6),
        transition_word_ratio=round(transition_word_ratio(text), 6),
        slop_word_score=round(slop_word_score(text), 6),
        punctuation_entropy=round(punctuation_entropy(text), 6),
        avg_word_length=round(avg_wl, 6),
        avg_sentence_length=round(avg_sl, 6),
        paragraph_count=len(paragraphs),
        sentence_count=sc,
        word_count=wc,
    )
