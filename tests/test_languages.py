"""Tests for multilingual support."""
from __future__ import annotations

from lmscan.languages import (
    LanguageConfig,
    detect_language,
    get_language_config,
    get_slop_words,
    get_transition_words,
    is_cjk_text,
    list_languages,
)


class TestLanguageConfig:
    def test_list_languages(self) -> None:
        langs = list_languages()
        assert "en" in langs
        assert "fr" in langs
        assert "es" in langs
        assert "de" in langs
        assert "pt" in langs
        assert len(langs) >= 5

    def test_get_english(self) -> None:
        config = get_language_config("en")
        assert config is not None
        assert config.name == "English"
        assert config.code == "en"

    def test_get_french(self) -> None:
        config = get_language_config("fr")
        assert config is not None
        assert config.name == "French"
        assert len(config.slop_words) > 10

    def test_get_spanish(self) -> None:
        config = get_language_config("es")
        assert config is not None
        assert config.name == "Spanish"
        assert len(config.slop_words) > 10

    def test_get_german(self) -> None:
        config = get_language_config("de")
        assert config is not None
        assert config.name == "German"

    def test_get_portuguese(self) -> None:
        config = get_language_config("pt")
        assert config is not None
        assert config.name == "Portuguese"

    def test_unknown_language(self) -> None:
        config = get_language_config("xx")
        assert config is None

    def test_case_insensitive(self) -> None:
        config = get_language_config("FR")
        assert config is not None
        assert config.name == "French"


class TestDetectLanguage:
    def test_english_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. This is a simple English sentence that should be detected as English."
        assert detect_language(text) == "en"

    def test_french_text(self) -> None:
        text = "Le chat est sur la table. Les enfants sont dans le jardin avec une belle journée. Ce texte est en français."
        assert detect_language(text) == "fr"

    def test_spanish_text(self) -> None:
        text = "El gato está en la mesa. Los niños están en el jardín. Este texto es en español. Pero no importa."
        assert detect_language(text) == "es"

    def test_german_text(self) -> None:
        text = "Die Katze ist auf dem Tisch. Der Hund ist nicht im Garten. Das ist ein deutscher Text und er ist auch gut."
        assert detect_language(text) == "de"

    def test_portuguese_text(self) -> None:
        text = "O gato está na mesa. As crianças são do jardim para a escola. Este texto é em português mas não importa."
        assert detect_language(text) == "pt"

    def test_empty_text(self) -> None:
        assert detect_language("") == "en"
        assert detect_language("   ") == "en"

    def test_cjk_text(self) -> None:
        text = "这是一个测试句子。人工智能正在改变世界。机器学习是现代技术的核心。"
        assert detect_language(text) == "zh"

    def test_japanese_text(self) -> None:
        text = "これはテストです。人工知能が世界を変えています。" + "あいうえおかきくけこ" * 5
        lang = detect_language(text)
        assert lang in ("ja", "zh")  # CJK overlap is expected

    def test_korean_text(self) -> None:
        text = "이것은 테스트입니다. 인공지능이 세상을 바꾸고 있습니다." + "가" * 20
        assert detect_language(text) == "ko"


class TestSlopWords:
    def test_french_slop_words(self) -> None:
        single, phrases = get_slop_words("fr")
        assert len(single) > 5
        assert len(phrases) > 3
        assert "approfondir" in single or "approfondir" in " ".join(phrases)

    def test_spanish_slop_words(self) -> None:
        single, phrases = get_slop_words("es")
        assert len(single) > 5

    def test_german_slop_words(self) -> None:
        single, phrases = get_slop_words("de")
        assert len(single) > 5

    def test_english_returns_empty(self) -> None:
        single, phrases = get_slop_words("en")
        assert single == set()
        assert phrases == []

    def test_unknown_returns_empty(self) -> None:
        single, phrases = get_slop_words("xx")
        assert single == set()
        assert phrases == []


class TestTransitionWords:
    def test_french_transitions(self) -> None:
        words, phrases = get_transition_words("fr")
        assert len(words) > 3
        assert "cependant" in words

    def test_english_returns_empty(self) -> None:
        words, phrases = get_transition_words("en")
        assert words == set()
        assert phrases == []


class TestCJK:
    def test_is_cjk_chinese(self) -> None:
        assert is_cjk_text("这是中文测试文本，关于人工智能检测。")

    def test_is_not_cjk_english(self) -> None:
        assert not is_cjk_text("This is English text about AI detection.")

    def test_is_not_cjk_empty(self) -> None:
        assert not is_cjk_text("")

    def test_mixed_text_low_cjk(self) -> None:
        text = "This is mostly English with a tiny bit of 中文"
        assert not is_cjk_text(text)
