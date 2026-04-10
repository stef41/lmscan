from __future__ import annotations

import math

from lmscan.features import (
    _tokenize,
    _split_sentences,
    _split_paragraphs,
    word_entropy,
    burstiness,
    vocabulary_richness,
    hapax_legomena_ratio,
    zipf_deviation,
    sentence_length_variance,
    readability_consistency,
    bigram_repetition,
    trigram_repetition,
    transition_word_ratio,
    slop_word_score,
    punctuation_entropy,
    extract_features,
)
from lmscan._types import TextFeatures


# ── Tokeniser / splitter helpers ─────────────────────────────────────────────

def test_tokenize_basic():
    assert _tokenize("Hello world") == ["hello", "world"]


def test_tokenize_strips_punctuation():
    tokens = _tokenize("Well, it's fine!")
    assert "it's" in tokens
    assert "well" in tokens
    assert "fine" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []


def test_split_sentences_basic():
    sents = _split_sentences("Hello world. How are you? Fine!")
    assert len(sents) == 3


def test_split_sentences_handles_abbreviations():
    sents = _split_sentences("Dr. Smith went to Washington. He met Mr. Jones.")
    assert len(sents) == 2


def test_split_sentences_empty():
    assert _split_sentences("") == []


def test_split_paragraphs():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    paras = _split_paragraphs(text)
    assert len(paras) == 3


def test_split_paragraphs_empty():
    assert _split_paragraphs("") == []


# ── Word entropy ─────────────────────────────────────────────────────────────

def test_word_entropy_uniform():
    # All same word → 0 entropy
    text = "hello hello hello hello hello"
    assert word_entropy(text) == 0.0


def test_word_entropy_diverse():
    text = "the quick brown fox jumps over the lazy dog runs fast near lovely trees"
    e = word_entropy(text)
    assert e > 2.0  # high entropy for many unique words


def test_word_entropy_empty():
    assert word_entropy("") == 0.0


# ── Burstiness ───────────────────────────────────────────────────────────────

def test_burstiness_uniform_sentences():
    # All sentences approximately same length → low burstiness
    text = "I like dogs very much. She likes cats a lot. He loves fish tanks. We enjoy bird song."
    b = burstiness(text)
    assert b < 0.5


def test_burstiness_varied_sentences():
    text = (
        "Go. "
        "The extraordinarily magnificent and beautifully designed cathedral stood "
        "majestically upon the ancient hillside overlooking the entire sprawling valley below. "
        "Hi. "
        "Running across the field she noticed the purple flowers blooming near the old stone wall "
        "where butterflies danced in the warm afternoon breeze."
    )
    b = burstiness(text)
    assert b > 0.3


def test_burstiness_single_sentence():
    assert burstiness("Just one sentence here.") == 0.0


# ── Vocabulary richness ─────────────────────────────────────────────────────

def test_vocabulary_richness_low():
    # Repeated words → low richness
    text = "the the the the the the the the the the"
    r = vocabulary_richness(text)
    assert r < 0.3


def test_vocabulary_richness_high():
    # All unique words → high richness
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    r = vocabulary_richness(text)
    assert r > 0.7


def test_vocabulary_richness_empty():
    assert vocabulary_richness("") == 0.0


# ── Hapax ratio ──────────────────────────────────────────────────────────────

def test_hapax_ratio_low():
    # Many repeated words → low hapax ratio
    text = "dog dog cat cat bird bird fish fish fox fox"
    h = hapax_legomena_ratio(text)
    assert h == 0.0

def test_hapax_ratio_high():
    # Mostly unique words → high hapax ratio
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    h = hapax_legomena_ratio(text)
    assert h == 1.0  # all appear exactly once


# ── Zipf deviation ───────────────────────────────────────────────────────────

def test_zipf_deviation_realistic_text():
    text = (
        "the the the the cat cat cat dog dog bird "
        "fish tree house river mountain sky cloud rain sun moon"
    )
    z = zipf_deviation(text)
    assert z >= 0.0


def test_zipf_deviation_short():
    assert zipf_deviation("hi") == 0.0


# ── Sentence length variance ────────────────────────────────────────────────

def test_sentence_length_variance_low():
    text = "I like cats very much. She loves dogs a lot. He eats food every day. We see birds at dawn."
    v = sentence_length_variance(text)
    assert v < 0.4


def test_sentence_length_variance_high():
    text = (
        "Go. "
        "The magnificent cathedral upon the ancient hillside overlooking the sprawling valley "
        "with its breathtaking views was truly a sight to behold for everyone."
    )
    v = sentence_length_variance(text)
    assert v > 0.5


def test_sentence_length_variance_single():
    assert sentence_length_variance("Just one.") == 0.0


# ── Readability consistency ──────────────────────────────────────────────────

def test_readability_consistency_uniform():
    # Two similar paragraphs → low std dev
    text = (
        "The cat sat on the mat and looked around the room with interest.\n\n"
        "The dog lay on the rug and watched the door with mild curiosity."
    )
    rc = readability_consistency(text)
    assert rc < 3.0


def test_readability_consistency_single_paragraph():
    assert readability_consistency("Just one paragraph here.") == 0.0


# ── Bigram repetition ───────────────────────────────────────────────────────

def test_bigram_repetition_high():
    # Very repetitive text
    text = "I like I like I like I like I like I like"
    b = bigram_repetition(text)
    assert b > 0.3


def test_bigram_repetition_low():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo"
    b = bigram_repetition(text)
    assert b == 0.0  # all bigrams unique


# ── Transition word ratio ───────────────────────────────────────────────────

def test_transition_word_ratio_high():
    text = "However the result was good. Moreover it was fast. Furthermore it was cheap. Additionally it was robust."
    r = transition_word_ratio(text)
    assert r > 0.05


def test_transition_word_ratio_zero():
    text = "I went to the store. I bought some milk. I came home."
    r = transition_word_ratio(text)
    assert r == 0.0


# ── Slop word score ──────────────────────────────────────────────────────────

def test_slop_word_score_high():
    text = (
        "Let us delve into this tapestry of innovation. "
        "We must leverage holistic synergy to foster multifaceted paradigms."
    )
    s = slop_word_score(text)
    assert s > 0.1


def test_slop_word_score_zero():
    text = "I went to the store yesterday and bought some bread and eggs."
    s = slop_word_score(text)
    assert s == 0.0


# ── Punctuation entropy ─────────────────────────────────────────────────────

def test_punctuation_entropy():
    text = "Hello! How are you? Fine, thanks. Great — really!"
    pe = punctuation_entropy(text)
    assert pe > 0.0


def test_punctuation_entropy_no_punctuation():
    text = "hello world"
    pe = punctuation_entropy(text)
    assert pe == 0.0


# ── extract_features ─────────────────────────────────────────────────────────

def test_extract_features_returns_all_fields():
    text = "This is a test sentence. And another one here."
    f = extract_features(text)
    assert isinstance(f, TextFeatures)
    assert f.word_count > 0
    assert f.sentence_count >= 1
    assert f.avg_word_length > 0


def test_extract_features_empty_text():
    f = extract_features("")
    assert f.word_count == 0
    assert f.entropy == 0.0


def test_extract_features_single_word():
    f = extract_features("hello")
    assert f.word_count == 1
    assert f.sentence_count == 1
