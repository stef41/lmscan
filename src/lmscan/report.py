from __future__ import annotations

import json
from dataclasses import asdict

from ._types import ScanResult, ParagraphScore

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
        # v0.5/v0.6 features
        "passive_voice_ratio":     ("high", 0.30, 0.15),
        "sentence_opening_diversity": ("low", 0.55, 0.70),
        "hedging_density":         ("high", 0.02, 0.005),
        "conjunction_start_ratio": ("high", 0.25, 0.10),
        "contraction_rate":        ("low",  0.002, 0.01),
        "first_person_ratio":      ("low",  0.002, 0.01),
        "list_pattern_density":    ("high", 0.15, 0.05),
        "long_ngram_repetition":   ("high", 0.05, 0.02),
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
    pct_features = {
        "slop_word_score", "transition_word_ratio", "bigram_repetition",
        "trigram_repetition", "passive_voice_ratio", "hedging_density",
        "conjunction_start_ratio", "contraction_rate", "first_person_ratio",
        "list_pattern_density", "long_ngram_repetition",
    }
    if feature in pct_features:
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
        ("Passive voice ratio",     "passive_voice_ratio",     f.passive_voice_ratio),
        ("Opening diversity",       "sentence_opening_diversity", f.sentence_opening_diversity),
        ("Hedging density",         "hedging_density",         f.hedging_density),
        ("Conjunction starts",      "conjunction_start_ratio", f.conjunction_start_ratio),
        ("Contraction rate",        "contraction_rate",        f.contraction_rate),
        ("First-person pronouns",   "first_person_ratio",      f.first_person_ratio),
        ("List patterns",           "list_pattern_density",    f.list_pattern_density),
        ("Long n-gram repetition",  "long_ngram_repetition",   f.long_ngram_repetition),
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


def format_directory_report(
    results: list[tuple[str, ScanResult]], dirname: str = ""
) -> str:
    """Generate a summary table for a batch directory scan."""
    lines: list[str] = []
    header_label = dirname or "directory"
    lines.append(f"\U0001f4c1 Directory Scan: {header_label}")

    col_file, col_ai, col_verdict, col_model = 25, 8, 16, 22
    top = (
        f"\u250c{'─' * col_file}\u252c{'─' * col_ai}"
        f"\u252c{'─' * col_verdict}\u252c{'─' * col_model}\u2510"
    )
    sep = (
        f"\u251c{'─' * col_file}\u253c{'─' * col_ai}"
        f"\u253c{'─' * col_verdict}\u253c{'─' * col_model}\u2524"
    )
    bot = (
        f"\u2514{'─' * col_file}\u2534{'─' * col_ai}"
        f"\u2534{'─' * col_verdict}\u2534{'─' * col_model}\u2518"
    )
    hdr = (
        f"\u2502 {'File':<{col_file - 2}} "
        f"\u2502 {'AI %':<{col_ai - 2}} "
        f"\u2502 {'Verdict':<{col_verdict - 2}} "
        f"\u2502 {'Top Model':<{col_model - 2}} \u2502"
    )
    lines.append(top)
    lines.append(hdr)
    lines.append(sep)

    flagged = 0
    for fname, res in results:
        pct = f"{res.ai_probability * 100:.0f}%"
        top_model = res.model_attribution[0].model if res.model_attribution else ""
        lines.append(
            f"\u2502 {fname:<{col_file - 2}} "
            f"\u2502 {pct:>{col_ai - 3}} "
            f"\u2502 {res.verdict:<{col_verdict - 2}} "
            f"\u2502 {top_model:<{col_model - 2}} \u2502"
        )
        if res.ai_probability >= 0.5:
            flagged += 1

    lines.append(bot)
    lines.append(f"  {len(results)} files scanned, {flagged} flagged as AI")
    return "\n".join(lines)


def format_paragraph_report(paragraphs: list[ParagraphScore]) -> str:
    """Generate a per-paragraph analysis table."""
    lines: list[str] = []
    lines.append("\U0001f4dd Per-Paragraph Analysis")

    col_p, col_ai, col_verdict, col_words = 7, 8, 16, 7
    top = (
        f"\u250c{'─' * col_p}\u252c{'─' * col_ai}"
        f"\u252c{'─' * col_verdict}\u252c{'─' * col_words}\u2510"
    )
    sep = (
        f"\u251c{'─' * col_p}\u253c{'─' * col_ai}"
        f"\u253c{'─' * col_verdict}\u253c{'─' * col_words}\u2524"
    )
    bot = (
        f"\u2514{'─' * col_p}\u2534{'─' * col_ai}"
        f"\u2534{'─' * col_verdict}\u2534{'─' * col_words}\u2518"
    )
    hdr = (
        f"\u2502 {'¶':<{col_p - 2}} "
        f"\u2502 {'AI %':<{col_ai - 2}} "
        f"\u2502 {'Verdict':<{col_verdict - 2}} "
        f"\u2502 {'Words':<{col_words - 2}} \u2502"
    )
    lines.append(top)
    lines.append(hdr)
    lines.append(sep)

    ai_indices: list[int] = []
    for ps in paragraphs:
        pct = f"{ps.ai_probability * 100:.0f}%"
        lines.append(
            f"\u2502 {ps.index + 1:<{col_p - 2}} "
            f"\u2502 {pct:>{col_ai - 3}} "
            f"\u2502 {ps.verdict:<{col_verdict - 2}} "
            f"\u2502 {ps.word_count:>{col_words - 3}} \u2502"
        )
        if ps.ai_probability >= 0.5:
            ai_indices.append(ps.index + 1)

    lines.append(bot)

    if ai_indices and len(ai_indices) < len(paragraphs):
        idx_str = ", ".join(str(i) for i in ai_indices)
        lines.append(
            f"  \u26a0\ufe0f Mixed content detected: "
            f"paragraphs {idx_str} appear AI-generated"
        )
    elif ai_indices:
        lines.append("  \u26a0\ufe0f All paragraphs appear AI-generated")

    return "\n".join(lines)


def _html_signal_icon(feature: str, value: float) -> tuple[str, str, str]:
    """Return (color_class, circle_html, label) for a feature value."""
    thresholds: dict[str, tuple[str, float, float]] = {
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
        return "green", '<span class="dot green"></span>', "Normal"

    direction, strong, moderate = thresholds[feature]

    if direction == "low":
        if value <= strong:
            return "red", '<span class="dot red"></span>', "Very low (AI)"
        if value <= moderate:
            return "yellow", '<span class="dot yellow"></span>', "Below average"
        return "green", '<span class="dot green"></span>', "Normal (human)"
    else:
        if value >= strong:
            return "red", '<span class="dot red"></span>', "High (AI)"
        if value >= moderate:
            return "yellow", '<span class="dot yellow"></span>', "Elevated"
        return "green", '<span class="dot green"></span>', "Normal (human)"


def format_html(result: ScanResult) -> str:
    """Generate a self-contained HTML report with embedded CSS."""
    f = result.features
    prob_pct = result.ai_probability * 100

    # Gauge color
    if result.ai_probability < 0.40:
        gauge_color = "#4caf50"
        gauge_label = "Low"
    elif result.ai_probability < 0.65:
        gauge_color = "#ff9800"
        gauge_label = "Medium"
    else:
        gauge_color = "#f44336"
        gauge_label = "High"

    # Feature rows
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
        ("Passive voice ratio",     "passive_voice_ratio",     f.passive_voice_ratio),
        ("Opening diversity",       "sentence_opening_diversity", f.sentence_opening_diversity),
        ("Hedging density",         "hedging_density",         f.hedging_density),
        ("Conjunction starts",      "conjunction_start_ratio", f.conjunction_start_ratio),
        ("Contraction rate",        "contraction_rate",        f.contraction_rate),
        ("First-person pronouns",   "first_person_ratio",      f.first_person_ratio),
        ("List patterns",           "list_pattern_density",    f.list_pattern_density),
        ("Long n-gram repetition",  "long_ngram_repetition",   f.long_ngram_repetition),
    ]

    feature_rows = ""
    for label, feat_key, value in features_display:
        _color, dot_html, signal_label = _html_signal_icon(feat_key, value)
        val_str = _fmt_value(feat_key, value)
        feature_rows += (
            f"<tr><td>{label}</td><td>{val_str}</td>"
            f"<td>{dot_html} {signal_label}</td></tr>\n"
        )

    # Model attribution section
    model_section = ""
    if result.model_attribution:
        model_rows = ""
        for m in result.model_attribution:
            ev_str = ", ".join(m.evidence[:6])
            if len(m.evidence) > 6:
                ev_str += f", +{len(m.evidence) - 6} more"
            conf_pct = f"{m.confidence * 100:.0f}%"
            model_rows += f"<tr><td>{m.model}</td><td>{conf_pct}</td><td>{ev_str}</td></tr>\n"
        model_section = (
            "\n    <h2>Model Attribution</h2>\n"
            "    <table>\n"
            "      <thead><tr><th>Model</th><th>Confidence</th><th>Evidence</th></tr></thead>\n"
            f"      <tbody>{model_rows}</tbody>\n"
            "    </table>"
        )

    # Per-sentence section
    sentence_section = ""
    if result.sentence_scores:
        sentence_rows = ""
        for i, ss in enumerate(result.sentence_scores, 1):
            sp = ss.ai_probability * 100
            icon = "&#x1f916;" if ss.ai_probability >= 0.5 else "&#x270d;&#xfe0f;"
            text_escaped = ss.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            text_preview = text_escaped[:80] + ("..." if len(ss.text) > 80 else "")
            flags_str = ""
            if ss.flags:
                flags_str = "<br>".join(f"<small>&#x26a0; {fl}</small>" for fl in ss.flags)
            sentence_rows += (
                f"<tr><td>{i}</td><td>{icon} {sp:.0f}%</td>"
                f"<td>{text_preview}{' ' + flags_str if flags_str else ''}</td></tr>\n"
            )
        sentence_section = (
            "\n    <h2>Per-Sentence Analysis</h2>\n"
            "    <table>\n"
            "      <thead><tr><th>#</th><th>AI %</th><th>Text</th></tr></thead>\n"
            f"      <tbody>{sentence_rows}</tbody>\n"
            "    </table>"
        )

    # Flags section
    flags_section = ""
    if result.flags:
        flags_list = "".join(f"<li>{fl}</li>" for fl in result.flags)
        flags_section = f"<h2>Flags</h2><ul>{flags_list}</ul>"

    gauge_fill_width = f"{prob_pct:.1f}%"

    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>lmscan Report</title>\n'
        '<style>\n'
        '  * { margin: 0; padding: 0; box-sizing: border-box; }\n'
        f'  body {{ background: #1a1a2e; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 2rem; line-height: 1.6; }}\n'
        '  .container { max-width: 800px; margin: 0 auto; }\n'
        '  h1 { color: #00d4ff; font-size: 1.8rem; margin-bottom: 0.5rem; }\n'
        '  h1 small { font-size: 0.7em; color: #888; font-weight: normal; }\n'
        '  h2 { color: #00d4ff; font-size: 1.2rem; margin: 1.5rem 0 0.8rem; border-bottom: 1px solid #333; padding-bottom: 0.3rem; }\n'
        f'  .verdict-box {{ background: #16213e; border-radius: 8px; padding: 1.5rem; margin: 1.2rem 0; border-left: 4px solid {gauge_color}; }}\n'
        '  .verdict-box .label { color: #888; font-size: 0.85rem; }\n'
        '  .verdict-box .value { font-size: 1.4rem; font-weight: bold; }\n'
        '  .gauge { width: 100%; height: 24px; background: #2a2a4a; border-radius: 12px; overflow: hidden; margin: 0.8rem 0; }\n'
        f'  .gauge-fill {{ height: 100%; background: {gauge_color}; border-radius: 12px; width: {gauge_fill_width}; }}\n'
        '  .gauge-text { text-align: center; font-size: 0.85rem; color: #aaa; margin-top: 0.2rem; }\n'
        '  .meta { display: flex; gap: 2rem; margin-top: 0.8rem; font-size: 0.9rem; color: #aaa; }\n'
        '  table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }\n'
        '  th, td { padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid #2a2a4a; }\n'
        '  th { color: #00d4ff; font-size: 0.85rem; text-transform: uppercase; }\n'
        '  tr:hover { background: #16213e; }\n'
        '  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }\n'
        '  .dot.red { background: #f44336; }\n'
        '  .dot.yellow { background: #ff9800; }\n'
        '  .dot.green { background: #4caf50; }\n'
        '  ul { list-style: none; padding: 0; }\n'
        '  ul li { padding: 0.3rem 0; }\n'
        '  small { color: #888; }\n'
        '  .footer { margin-top: 2rem; text-align: center; color: #555; font-size: 0.8rem; }\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '<div class="container">\n'
        f'  <h1>lmscan <small>v{_VERSION} &mdash; AI Text Forensics</small></h1>\n'
        '\n'
        '  <div class="verdict-box">\n'
        '    <div class="label">Verdict</div>\n'
        f'    <div class="value" style="color: {gauge_color};">{result.verdict} &mdash; {prob_pct:.0f}% AI probability</div>\n'
        '    <div class="gauge"><div class="gauge-fill"></div></div>\n'
        f'    <div class="gauge-text">{gauge_label} confidence: {result.confidence}</div>\n'
        '    <div class="meta">\n'
        f'      <span>{f.word_count} words</span>\n'
        f'      <span>{f.sentence_count} sentences</span>\n'
        f'      <span>Scanned in {result.scan_time_s:.2f}s</span>\n'
        '    </div>\n'
        '  </div>\n'
        '\n'
        '  <h2>Feature Breakdown</h2>\n'
        '  <table>\n'
        '    <thead><tr><th>Feature</th><th>Value</th><th>Signal</th></tr></thead>\n'
        '    <tbody>\n'
        f'{feature_rows}'
        '    </tbody>\n'
        '  </table>\n'
        f'{model_section}\n'
        f'{flags_section}\n'
        f'{sentence_section}\n'
        '\n'
        f'  <div class="footer">Generated by lmscan v{_VERSION}</div>\n'
        '</div>\n'
        '</body>\n'
        '</html>'
    )
    return html
