"""Multilingual AI text detection support.

Provides language-specific slop word dictionaries and detection configuration
for French, Spanish, German, Portuguese, and CJK character handling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LanguageConfig:
    """Configuration for language-specific AI text detection."""

    name: str
    code: str
    slop_words: set[str] = field(default_factory=set)
    slop_phrases: list[str] = field(default_factory=list)
    transition_words: set[str] = field(default_factory=set)
    transition_phrases: list[str] = field(default_factory=list)
    # Adjustment multipliers for detection thresholds
    burstiness_threshold: float = 0.20
    slop_threshold: float = 0.005


# ── Language dictionaries ─────────────────────────────────────────────────────

_FRENCH = LanguageConfig(
    name="French",
    code="fr",
    slop_words={
        "approfondir", "paysage", "phare", "levier", "holistique",
        "synergie", "paradigme", "multifacette", "nuancé", "fondamental",
        "incontournable", "primordial", "novateur", "innovant", "pionnier",
        "optimiser", "faciliter", "mettre en lumière", "souligner",
        "pertinent", "exhaustif", "catalyseur", "transversal",
        "écosystème", "résilience", "valoriser",
    },
    slop_phrases=[
        "il est important de noter", "il convient de souligner",
        "dans le monde d'aujourd'hui", "dans le cadre de",
        "force est de constater", "il va sans dire",
        "en fin de compte", "dans ce contexte",
    ],
    transition_words={
        "cependant", "néanmoins", "toutefois", "également",
        "notamment", "effectivement", "fondamentalement",
        "essentiellement", "par conséquent", "ainsi",
    },
    transition_phrases=[
        "de plus", "en outre", "d'autre part", "en conclusion",
        "par ailleurs", "en revanche",
    ],
)

_SPANISH = LanguageConfig(
    name="Spanish",
    code="es",
    slop_words={
        "profundizar", "panorama", "faro", "apalancamiento", "holístico",
        "sinergia", "paradigma", "multifacético", "matizado", "fundamental",
        "innovador", "pionero", "optimizar", "facilitar", "subrayar",
        "resaltar", "exhaustivo", "catalizador", "transversal",
        "ecosistema", "resiliencia", "poner de manifiesto",
    },
    slop_phrases=[
        "es importante señalar", "cabe destacar",
        "en el mundo actual", "en el ámbito de",
        "huelga decir", "en definitiva",
        "al fin y al cabo", "en este contexto",
    ],
    transition_words={
        "sin embargo", "no obstante", "además", "asimismo",
        "concretamente", "efectivamente", "fundamentalmente",
        "esencialmente", "por consiguiente", "así",
    },
    transition_phrases=[
        "por otra parte", "en conclusión", "por lo tanto",
        "a su vez", "en cambio",
    ],
)

_GERMAN = LanguageConfig(
    name="German",
    code="de",
    slop_words={
        "vertiefen", "landschaft", "leuchtturm", "hebel", "ganzheitlich",
        "synergie", "paradigma", "facettenreich", "nuanciert", "grundlegend",
        "wegweisend", "innovativ", "optimieren", "erleichtern", "unterstreichen",
        "hervorheben", "umfassend", "katalysator", "ökosystem", "resilienz",
    },
    slop_phrases=[
        "es ist wichtig zu beachten", "es sei darauf hingewiesen",
        "in der heutigen welt", "im rahmen von",
        "es versteht sich von selbst", "letzten endes",
        "in diesem zusammenhang",
    ],
    transition_words={
        "jedoch", "dennoch", "außerdem", "darüber hinaus",
        "insbesondere", "tatsächlich", "grundsätzlich",
        "im wesentlichen", "folglich", "somit",
    },
    transition_phrases=[
        "darüber hinaus", "andererseits", "zusammenfassend",
        "infolgedessen", "im gegensatz dazu",
    ],
)

_PORTUGUESE = LanguageConfig(
    name="Portuguese",
    code="pt",
    slop_words={
        "aprofundar", "paisagem", "farol", "alavancagem", "holístico",
        "sinergia", "paradigma", "multifacetado", "nuançado", "fundamental",
        "inovador", "pioneiro", "otimizar", "facilitar", "sublinhar",
        "destacar", "exaustivo", "catalisador", "ecossistema", "resiliência",
    },
    slop_phrases=[
        "é importante notar", "vale destacar",
        "no mundo atual", "no âmbito de",
        "escusado será dizer", "em última análise",
        "neste contexto",
    ],
    transition_words={
        "contudo", "todavia", "além disso", "ademais",
        "nomeadamente", "efetivamente", "fundamentalmente",
        "essencialmente", "consequentemente", "assim",
    },
    transition_phrases=[
        "por outro lado", "em conclusão", "portanto",
        "por sua vez", "em contrapartida",
    ],
)


# ── Registry ──────────────────────────────────────────────────────────────────

_LANGUAGES: dict[str, LanguageConfig] = {
    "en": LanguageConfig(name="English", code="en"),  # default, uses main slop lists
    "fr": _FRENCH,
    "es": _SPANISH,
    "de": _GERMAN,
    "pt": _PORTUGUESE,
}


def get_language_config(code: str) -> LanguageConfig | None:
    """Get language configuration by ISO 639-1 code."""
    return _LANGUAGES.get(code.lower())


def list_languages() -> list[str]:
    """Return list of supported language codes."""
    return sorted(_LANGUAGES.keys())


def detect_language(text: str) -> str:
    """Detect the primary language of text using character and word heuristics.

    Returns ISO 639-1 code. Falls back to 'en' for unrecognised text.
    """
    if not text or not text.strip():
        return "en"

    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return "en"

    # CJK detection
    cjk_count = sum(1 for ch in text if _is_cjk(ch))
    if cjk_count > len(text) * 0.15:
        return "zh"  # generic CJK — mainly Chinese

    # Hiragana/Katakana detection for Japanese
    jp_count = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff")
    if jp_count > len(text) * 0.10:
        return "ja"

    # Korean Hangul detection
    kr_count = sum(1 for ch in text if "\uac00" <= ch <= "\ud7a3")
    if kr_count > len(text) * 0.10:
        return "ko"

    # European language detection via common function words
    _MARKERS: dict[str, list[str]] = {
        "fr": ["le", "la", "les", "de", "des", "est", "dans", "pour", "avec", "une", "sont", "qui", "que", "pas", "ce", "cette"],
        "es": ["el", "la", "los", "las", "de", "del", "en", "es", "por", "con", "una", "que", "como", "pero", "más", "esto"],
        "de": ["der", "die", "das", "und", "ist", "ein", "eine", "für", "mit", "auf", "den", "dem", "nicht", "sich", "von", "auch"],
        "pt": ["o", "a", "os", "as", "de", "do", "da", "em", "para", "com", "uma", "que", "como", "mas", "mais", "são"],
    }

    word_set = set(words[:200])  # sample first 200 words
    scores: dict[str, int] = {}
    for lang, markers in _MARKERS.items():
        scores[lang] = sum(1 for m in markers if m in word_set)

    best_lang = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best_lang] >= 4:
        return best_lang

    return "en"


def _is_cjk(ch: str) -> bool:
    """Check if a character is in the CJK Unified Ideographs range."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0xF900 <= cp <= 0xFAFF
    )


def is_cjk_text(text: str) -> bool:
    """Check if text contains significant CJK characters."""
    if not text:
        return False
    cjk_count = sum(1 for ch in text if _is_cjk(ch))
    return cjk_count > len(text) * 0.15


def get_slop_words(language: str = "en") -> tuple[set[str], list[str]]:
    """Get slop single-words and phrases for a given language.

    Returns (single_words, phrases) tuple.
    """
    config = _LANGUAGES.get(language)
    if config is None or language == "en":
        return set(), []  # English uses the default lists in features.py

    single = {w for w in config.slop_words if " " not in w}
    phrases = [w for w in config.slop_words if " " in w] + config.slop_phrases
    return single, phrases


def get_transition_words(language: str = "en") -> tuple[set[str], list[str]]:
    """Get transition words and phrases for a given language.

    Returns (single_words, phrases) tuple.
    """
    config = _LANGUAGES.get(language)
    if config is None or language == "en":
        return set(), []  # English uses the default lists in features.py

    return config.transition_words, config.transition_phrases
