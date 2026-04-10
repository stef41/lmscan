"""Tests for model fingerprints including new models."""

from __future__ import annotations

from lmscan.fingerprint import fingerprint, _MODEL_PROFILES


def test_qwen_fingerprint_exists():
    assert "Qwen / Qwen2" in _MODEL_PROFILES


def test_deepseek_fingerprint_exists():
    assert "DeepSeek" in _MODEL_PROFILES


def test_cohere_fingerprint_exists():
    assert "Cohere / Command R" in _MODEL_PROFILES


def test_phi_fingerprint_exists():
    assert "Phi / Phi-3" in _MODEL_PROFILES


def test_all_profiles_have_required_fields():
    required = {"vocabulary", "phrases", "hedging", "weight"}
    for model, profile in _MODEL_PROFILES.items():
        for key in required:
            assert key in profile, f"{model} missing '{key}'"
        assert isinstance(profile["vocabulary"], list)
        assert isinstance(profile["phrases"], list)
        assert isinstance(profile["hedging"], list)
        assert isinstance(profile["weight"], (int, float))


def test_identify_qwen_text():
    text = (
        "I'd be happy to help you elucidate this multifaceted paradigm. "
        "Let me think about this step by step. "
        "To some extent, it's worth noting that the approach is comprehensive."
    )
    matches = fingerprint(text)
    model_names = [m.model for m in matches]
    assert "Qwen / Qwen2" in model_names


def test_identify_deepseek_text():
    text = (
        "Let me analyze this problem in a holistic and synergistic manner. "
        "Based on my analysis, the paradigmatic approach is optimal. "
        "It should be noted that one could argue for a comprehensive alternative."
    )
    matches = fingerprint(text)
    model_names = [m.model for m in matches]
    assert "DeepSeek" in model_names


def test_model_count_at_least_9():
    assert len(_MODEL_PROFILES) >= 9


def test_cohere_text_fingerprint():
    text = (
        "Here's what I found about this pivotal and quintessential topic. "
        "The nuanced approach helps streamline the process. "
        "That said, having said that, there are comprehensive alternatives."
    )
    matches = fingerprint(text)
    model_names = [m.model for m in matches]
    assert "Cohere / Command R" in model_names


def test_phi_text_fingerprint():
    text = (
        "Fundamentally, this is essentially a critically important problem. "
        "The key point is to optimize effectively and significantly. "
        "In essence, broadly speaking, the solution is notably simple."
    )
    matches = fingerprint(text)
    model_names = [m.model for m in matches]
    assert "Phi / Phi-3" in model_names
