"""Tests for the character-level n-gram perplexity estimator."""

from __future__ import annotations

import math

import pytest

from lmscan.perplexity import (
    PerplexityResult,
    compute_perplexity,
    _char_idx,
    _unigram_lp,
    _bigram_lp,
    _trigram_lp,
    _interpolated_logp,
    _VOCAB,
    _UNK_IDX,
    _UNK_LOGP,
    _LAMBDA_1,
    _LAMBDA_2,
    _LAMBDA_3,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

NATURAL_TEXT = (
    "The rain in Spain falls mainly on the plain. Yesterday, I went to "
    "the market and bought some apples. They were crisp and tart — exactly "
    "what I needed after a long walk through the muddy fields."
)

PREDICTABLE_TEXT = (
    "This is a comprehensive and multifaceted analysis that delves into "
    "the transformative landscape of modern technology. Furthermore, it is "
    "important to note that leveraging these innovative solutions provides "
    "a holistic approach to navigating the complexities of our world."
)


# ── Character index tests ────────────────────────────────────────────────────

class TestCharIdx:
    def test_space(self):
        assert _char_idx(" ") == 0

    def test_lowercase_a(self):
        assert _char_idx("a") == 1

    def test_lowercase_z(self):
        assert _char_idx("z") == 26

    def test_digit_0(self):
        idx = _char_idx("0")
        assert idx == _VOCAB.index("0")

    def test_unknown_char(self):
        # Characters not in vocab should return UNK index
        assert _char_idx("€") == _UNK_IDX
        assert _char_idx("→") == _UNK_IDX

    def test_uppercase_is_unk(self):
        # Our model is lowercase-only, so at the index level uppercase is UNK
        assert _char_idx("A") == _UNK_IDX


# ── Log-probability tests ───────────────────────────────────────────────────

class TestUnigramLP:
    def test_space_has_highest_prob(self):
        # Space should be the most common character
        assert _unigram_lp(" ") > _unigram_lp("z")

    def test_e_is_common(self):
        assert _unigram_lp("e") > _unigram_lp("q")

    def test_all_probs_negative(self):
        for ch in "abcdefghijklmnopqrstuvwxyz .,":
            assert _unigram_lp(ch) < 0

    def test_unknown_returns_unk(self):
        assert _unigram_lp("§") == _UNK_LOGP


class TestBigramLP:
    def test_th_is_common(self):
        # "th" is one of the most common English bigrams
        assert _bigram_lp("t", "h") > _bigram_lp("q", "z")

    def test_fallback_to_unigram(self):
        # A rare bigram should fall back to the unigram probability
        lp = _bigram_lp("x", "x")
        assert lp == _unigram_lp("x")

    def test_space_t_common(self):
        # " t" (start of "the", "that", etc.) should be common
        assert _bigram_lp(" ", "t") > _bigram_lp(" ", "z")


class TestTrigramLP:
    def test_the_is_common(self):
        # " th" is one of the most common English trigrams
        assert _trigram_lp(" ", "t", "h") > _trigram_lp("x", "z", "q")

    def test_fallback_chain(self):
        # Unknown trigram should still return a finite value
        lp = _trigram_lp("z", "z", "z")
        assert math.isfinite(lp)
        assert lp < 0


class TestInterpolatedLogP:
    def test_returns_finite(self):
        lp = _interpolated_logp("t", "h", "e")
        assert math.isfinite(lp)
        assert lp < 0

    def test_higher_than_unigram_for_common(self):
        # Interpolated prob of "the" should be higher than unigram of "e"
        lp_interp = _interpolated_logp(" ", "t", "h")
        lp_uni = _unigram_lp("h")
        assert lp_interp >= lp_uni

    def test_lambda_weights_sum_to_one(self):
        assert pytest.approx(_LAMBDA_1 + _LAMBDA_2 + _LAMBDA_3, abs=1e-6) == 1.0


# ── Perplexity computation tests ────────────────────────────────────────────

class TestComputePerplexity:
    def test_returns_result(self):
        result = compute_perplexity("hello world")
        assert isinstance(result, PerplexityResult)

    def test_perplexity_positive(self):
        result = compute_perplexity(NATURAL_TEXT)
        assert result.perplexity > 0

    def test_cross_entropy_positive(self):
        result = compute_perplexity(NATURAL_TEXT)
        assert result.cross_entropy > 0

    def test_log_likelihood_negative(self):
        result = compute_perplexity(NATURAL_TEXT)
        assert result.log_likelihood < 0

    def test_oov_rate_in_range(self):
        result = compute_perplexity(NATURAL_TEXT)
        assert 0.0 <= result.oov_rate <= 1.0

    def test_ai_signal_in_range(self):
        result = compute_perplexity(NATURAL_TEXT)
        assert 0.0 <= result.ai_signal <= 1.0

    def test_short_text_returns_inf(self):
        result = compute_perplexity("ab")
        assert result.perplexity == float("inf")

    def test_empty_text(self):
        result = compute_perplexity("")
        assert result.perplexity == float("inf")
        assert result.ai_signal == 0.5

    def test_num_chars_counting(self):
        text = "hello world"
        result = compute_perplexity(text)
        # num_chars = len(text) - 2 (first two chars have no full trigram context)
        assert result.num_chars == len(text) - 2

    def test_unicode_doesnt_crash(self):
        result = compute_perplexity("日本語のテキスト")
        assert isinstance(result, PerplexityResult)
        # High OOV rate expected for non-Latin text
        assert result.oov_rate > 0

    def test_repeated_text_lower_perplexity(self):
        # Highly repetitive text should have lower perplexity
        repetitive = "the the the the the the the the the the " * 5
        varied = "fox jumped quickly over the lazy brown dog sitting by fences"
        r1 = compute_perplexity(repetitive)
        r2 = compute_perplexity(varied)
        assert r1.perplexity < r2.perplexity

    def test_predictable_vs_natural(self):
        # Both should produce finite results
        r1 = compute_perplexity(NATURAL_TEXT)
        r2 = compute_perplexity(PREDICTABLE_TEXT)
        assert math.isfinite(r1.perplexity)
        assert math.isfinite(r2.perplexity)


class TestPerplexityEdgeCases:
    def test_only_spaces(self):
        result = compute_perplexity("      ")
        assert math.isfinite(result.perplexity)

    def test_only_punctuation(self):
        result = compute_perplexity("...!!??,,")
        assert math.isfinite(result.perplexity)

    def test_single_character_repeated(self):
        result = compute_perplexity("aaaaaaaaaa")
        assert math.isfinite(result.perplexity)

    def test_all_digits(self):
        result = compute_perplexity("1234567890" * 3)
        assert math.isfinite(result.perplexity)
