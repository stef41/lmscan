from __future__ import annotations

import json
from dataclasses import asdict

from ._types import ScanResult

_VERSION = "0.1.0"


def _signal_icon(feature: str, value: float) -> tuple[str, str]:
    """Return (icon, label) for a feature value indicating AI vs human."""
    thresholds: dict[str, tuple[str, float, float]] = {
        # feature: (direction, strong_thresh, moderate_thresh)
        # direction "low" means low value = AI
        "burstiness":              ("low",  0.20, 0.35),
        "sentence_length_variance": ("low", 0.20, 0.30),
        "vocabulary_richness":     ("low",  0.40, 0.55),
        "slop_word_score":         ("high", 0.02, 0.005),
        "transition_word_ratio":   ("high", 0.03, 0.02),
        "readability_consistency": ("low",  0.8,  1.5),
        "zipf_deviation":          ("high", 0.20, 0.15),
        "hapax_ratio":             ("low",  0.35, 0.45),
        "punctuation_entropy":     ("low",  1.5,  2.0),
        "bigram_repetition":       ("high", 0.15, 0.10),
    }
    if feature not in thresholds:
        return "\U0001f7e2", "Normal"

    direction, strong, moderate = thresholds[feature]

    if direction == "low":
        if value <= strong:
            return "\U0001f534", "Very low (AI)"
        if value <= moderate:
            return "\U0001f7e1", "Below average"
        return "\U0001f7e2", "Normal (human)"
    else:
        if value >= strong:
            return "\U0001f534", f"High (AI)"
        if value >= moderate:
            return "\U0001f7e1", "Elevated"
        return "\U0001f7e2", "Normal (human)"


def _fmt_value(feature: str, value: float) -> str:
    """Format a feature value for display."""
    if feature in ("slop_word_score", "transition_word_ratio", "bigram_repetition", "trigram_repetition"):
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def format_report(result: ScanResult, *, show_sentences: bool = False) -> str:
    """Generate a rich ASCII terminal report."""
    f = result.features
    lines: list[str] = []

    # Header
    lines.append(f"\U0001f50d lmscan v{_VERSION} \u2014 AI Text Forensics")
    lines.append("\u2550" * 50)
    lines.append("")

    # Verdict
    prob_pct = result.ai_probability * 100
    icon = "\U0001f916" if result.ai_probability >= 0.5 else "\u270d\ufe0f"
    lines.append(f"  Verdict:     {icon} {result.verdict} ({prob_pct:.0f}% confidence)")
    lines.append(f"  Words:       {f.word_count}")
    lines.append(f"  Sentences:   {f.sentence_count}")
    lines.append(f"  Scanned in {result.scan_time_s:.2f}s")
    lines.append("")

    # Feature table
    features_display = [
        ("Burstiness",              "burstiness",              f.burstiness),
        ("Sentence length variance", "sentence_length_variance", f.sentence_length_variance),
        ("Vocabulary richness",     "vocabulary_richness",     f.vocabulary_richness),
        ("Slop word density",       "slop_word_score",         f.slop_word_score),
        ("Transition word ratio",   "transition_word_ratio",   f.transition_word_ratio),
        ("Readability consistency", "readability_consistency",  f.readability_consistency),
        ("Zipf deviation",          "zipf_deviation",          f.zipf_deviation),
        ("Hapax legomena ratio",    "hapax_ratio",             f.hapax_ratio),
        ("Punctuation entropy",     "punctuation_entropy",     f.punctuation_entropy),
        ("Bigram repetition",       "bigram_repetition",       f.bigram_repetition),
    ]

    col1, col2, col3 = 28, 10, 20
    sep_line = f"\u251c{'─' * col1}\u253c{'─' * col2}\u253c{'─' * col3}\u2524"
    top_line = f"\u250c{'─' * col1}\u252c{'─' * col2}\u252c{'─' * col3}\u2510"
    bot_line = f"\u2514{'─' * col1}\u2534{'─' * col2}\u2534{'─' * col3}\u2518"
    hdr = f"\u2502 {'Feature':<{col1 - 2}} \u2502 {'Value':<{col2 - 2}} \u2502 {'Signal':<{col3 - 2}} \u2502"

    lines.append(top_line)
    lines.append(hdr)
    lines.append(sep_line)

    for label, feat_key, value in features_display:
        icon, signal_label = _signal_icon(feat_key, value)
        val_str = _fmt_value(feat_key, value)
        lines.append(
            f"\u2502 {label:<{col1 - 2}} "
            f"\u2502 {val_str:<{col2 - 2}} "
            f"\u2502 {icon} {signal_label:<{col3 - 4}} \u2502"
        )

    lines.append(bot_line)
    lines.append("")

    # Model attribution
    if result.model_attribution:
        lines.append("\U0001f50e Model Attribution")
        for i, m in enumerate(result.model_attribution, 1):
            ev_str = ", ".join(m.evidence[:6])
            if len(m.evidence) > 6:
                ev_str += f", +{len(m.evidence) - 6} more"
            lines.append(f"  {i}. {m.model:<22} {m.confidence:>3.0%} \u2014 {ev_str}")
        lines.append("")

    # Flags
    if result.flags:
        lines.append("\u26a0\ufe0f  Flags")
        for flag in result.flags:
            lines.append(f"  \u2022 {flag}")
        lines.append("")

    # Per-sentence breakdown
    if show_sentences and result.sentence_scores:
        lines.append("\U0001f4dd Per-Sentence Analysis")
        lines.append("\u2500" * 50)
        for i, ss in enumerate(result.sentence_scores, 1):
            prob_pct = ss.ai_probability * 100
            icon = "\U0001f916" if ss.ai_probability >= 0.5 else "\u270d\ufe0f"
            text_preview = ss.text[:60] + ("..." if len(ss.text) > 60 else "")
            lines.append(f"  {i}. {icon} {prob_pct:4.0f}%  {text_preview}")
            if ss.flags:
                for flag in ss.flags:
                    lines.append(f"       \u26a0 {flag}")
        lines.append("")

    return "\n".join(lines)


def format_json(result: ScanResult) -> str:
    """Serialise a :class:`ScanResult` to indented JSON."""
    data = asdict(result)
    # Drop the full text to keep output manageable
    data.pop("text", None)
    for s in data.get("sentence_scores", []):
        s.pop("text", None)
    return json.dumps(data, indent=2, ensure_ascii=False)
