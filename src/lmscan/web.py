"""Web UI for lmscan — Streamlit-based demo interface.

Launch with: ``streamlit run -m lmscan.web`` or ``lmscan --web``.

Requires the optional ``streamlit`` dependency:
``pip install lmscan[web]``
"""
from __future__ import annotations

import importlib
import sys

_STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


def check_streamlit() -> bool:
    """Check whether streamlit is importable."""
    return _STREAMLIT_AVAILABLE


def launch() -> None:
    """Launch the Streamlit web UI for lmscan.

    Raises RuntimeError if streamlit is not installed.
    """
    if not _STREAMLIT_AVAILABLE:
        raise RuntimeError(
            "Streamlit is required for the web UI. Install with: pip install lmscan[web]"
        )

    import streamlit as st

    from .scanner import scan, scan_mixed
    from .fingerprint import identify_slop_phrases
    from .languages import detect_language, list_languages
    from . import __version__

    st.set_page_config(page_title="lmscan", page_icon="🔍", layout="wide")
    st.title("🔍 lmscan — AI Text Forensics")
    st.caption(f"v{__version__} · Open-source GPTZero alternative")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Detection threshold", 0.0, 1.0, 0.5, 0.05)
        show_features = st.checkbox("Show feature breakdown", value=True)
        show_sentences = st.checkbox("Show per-sentence analysis", value=False)
        show_mixed = st.checkbox("Show per-paragraph analysis", value=False)
        st.divider()
        st.markdown(
            "[GitHub](https://github.com/stef41/lmscan) · "
            "[PyPI](https://pypi.org/project/lmscan/)"
        )

    # Main input
    text = st.text_area(
        "Paste text to analyse",
        height=250,
        placeholder="Paste any text here to check if it was written by AI...",
    )

    if st.button("🔍 Scan", type="primary") and text.strip():
        result = scan(text)
        lang = detect_language(text)

        # Result header
        col1, col2, col3 = st.columns(3)
        with col1:
            prob_pct = result.ai_probability * 100
            colour = "🔴" if prob_pct >= 65 else "🟡" if prob_pct >= 40 else "🟢"
            st.metric("AI Probability", f"{prob_pct:.0f}%", delta=None)
            st.markdown(f"{colour} **{result.verdict}**")
        with col2:
            st.metric("Confidence", result.confidence)
            st.caption(f"{result.features.word_count} words, {result.features.sentence_count} sentences")
        with col3:
            if result.model_attribution:
                top = result.model_attribution[0]
                st.metric("Likely Model", top.model)
                st.caption(f"{top.confidence:.0%} confidence")
            else:
                st.metric("Likely Model", "Unknown")
            if lang != "en":
                st.caption(f"Language: {lang}")

        # Model attribution chart
        if result.model_attribution:
            st.subheader("Model Attribution")
            chart_data = {
                m.model: m.confidence for m in result.model_attribution
            }
            st.bar_chart(chart_data)

            with st.expander("Evidence"):
                for m in result.model_attribution:
                    if m.evidence:
                        st.markdown(f"**{m.model}** ({m.confidence:.0%}): {', '.join(m.evidence[:5])}")

        # Feature breakdown
        if show_features:
            st.subheader("Feature Breakdown")
            feats = result.features
            feat_dict = {
                "Burstiness": feats.burstiness,
                "Entropy": feats.entropy,
                "Vocabulary Richness": feats.vocabulary_richness,
                "Hapax Ratio": feats.hapax_ratio,
                "Zipf Deviation": feats.zipf_deviation,
                "Sentence Length Variance": feats.sentence_length_variance,
                "Readability Consistency": feats.readability_consistency,
                "Bigram Repetition": feats.bigram_repetition,
                "Trigram Repetition": feats.trigram_repetition,
                "Transition Word Ratio": feats.transition_word_ratio,
                "Slop Word Score": feats.slop_word_score,
                "Punctuation Entropy": feats.punctuation_entropy,
            }
            st.bar_chart(feat_dict)

        # Slop phrases
        slop = identify_slop_phrases(text)
        if slop:
            st.subheader(f"AI Slop Phrases Detected ({len(slop)})")
            for phrase, positions in slop.items():
                st.markdown(f"- **\"{phrase}\"** (×{len(positions)})")

        # Per-sentence analysis
        if show_sentences and result.sentence_scores:
            st.subheader("Per-Sentence Analysis")
            for i, ss in enumerate(result.sentence_scores):
                prob = ss.ai_probability
                icon = "🔴" if prob >= 0.65 else "🟡" if prob >= 0.4 else "🟢"
                st.markdown(f"{icon} `{prob:.0%}` — {ss.text[:120]}...")

        # Mixed content (paragraph-level)
        if show_mixed:
            st.subheader("Per-Paragraph Analysis")
            _, paragraphs = scan_mixed(text)
            for p in paragraphs:
                prob = p.ai_probability
                icon = "🔴" if prob >= 0.65 else "🟡" if prob >= 0.4 else "🟢"
                st.markdown(f"{icon} **{p.verdict}** ({prob:.0%}) — ¶{p.index + 1} ({p.word_count} words)")
                with st.expander(f"Paragraph {p.index + 1}"):
                    st.text(p.text[:300])

        # Flags
        if result.flags:
            st.subheader("Flags")
            for flag in result.flags:
                st.warning(flag)

    elif not text.strip() and st.session_state.get("_scan_clicked"):
        st.warning("Please paste some text to analyse.")


def main() -> None:
    """Entry point for ``python -m lmscan.web``."""
    if not _STREAMLIT_AVAILABLE:
        print("Error: streamlit is required. Install with: pip install lmscan[web]", file=sys.stderr)
        sys.exit(1)

    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)


if __name__ == "__main__":
    main()
