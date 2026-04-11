from __future__ import annotations

import os

from ._types import ScanResult, ParagraphScore
from .detector import detect, detect_paragraphs
from .fingerprint import fingerprint
from .classifier import classify
from .perplexity import compute_perplexity


def _ensemble_probability(
    detector_prob: float,
    classifier_prob: float,
    perplexity_signal: float,
    word_count: int,
) -> float:
    """Combine detector, classifier, and perplexity into a single score.

    Uses confidence-weighted averaging with agreement bonus:
    - When all three agree, boost confidence
    - When they disagree, reduce confidence (more uncertain)
    """
    # Perplexity weight scales with text length
    if word_count >= 100:
        ppl_w = 0.15
    elif word_count >= 40:
        ppl_w = 0.08
    else:
        ppl_w = 0.0

    det_w = 0.50 - ppl_w / 2
    cls_w = 0.50 - ppl_w / 2

    base = det_w * detector_prob + cls_w * classifier_prob + ppl_w * perplexity_signal

    # Agreement bonus: if all signals agree (all >0.5 or all <0.5), push further
    signals = [detector_prob, classifier_prob]
    if ppl_w > 0:
        signals.append(perplexity_signal)

    all_ai = all(s >= 0.5 for s in signals)
    all_human = all(s < 0.5 for s in signals)

    if all_ai or all_human:
        # Agreement: reinforce signal by up to 8%
        agreement_factor = min(abs(s - 0.5) for s in signals) * 0.4
        if all_ai:
            base = min(1.0, base + agreement_factor)
        else:
            base = max(0.0, base - agreement_factor)

    return max(0.0, min(1.0, base))


def _verdict(prob: float) -> str:
    if prob >= 0.85:
        return "AI-generated"
    if prob >= 0.65:
        return "Likely AI"
    if prob >= 0.40:
        return "Mixed"
    if prob >= 0.20:
        return "Likely human"
    return "Human-written"


def scan(text: str) -> ScanResult:
    """Scan text for AI-generated content using ensemble detection.

    Combines three orthogonal signals:
    1. Signal-based detector (21 weighted linguistic features)
    2. Trained logistic regression classifier (23 features)
    3. Character-level n-gram perplexity model

    Then adds LLM model fingerprinting.
    """
    result = detect(text)
    clf_result = classify(result.features)
    ppl_result = compute_perplexity(text)

    # Ensemble combination
    ensemble_prob = _ensemble_probability(
        detector_prob=result.ai_probability,
        classifier_prob=clf_result.ai_probability,
        perplexity_signal=ppl_result.ai_signal,
        word_count=result.features.word_count,
    )

    result.ai_probability = round(ensemble_prob, 4)
    result.verdict = _verdict(ensemble_prob)
    result.model_attribution = fingerprint(text)
    return result


def scan_file(path: str, encoding: str = "utf-8") -> ScanResult:
    """Scan a text file."""
    with open(path, "r", encoding=encoding) as f:
        text = f.read()
    return scan(text)


def scan_directory(
    path: str,
    extensions: tuple[str, ...] = (".txt", ".md", ".rst", ".tex"),
    encoding: str = "utf-8",
) -> list[tuple[str, ScanResult]]:
    """Scan all text files in a directory."""
    results: list[tuple[str, ScanResult]] = []
    root = os.path.abspath(path)
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if any(fname.endswith(ext) for ext in extensions):
                fpath = os.path.join(dirpath, fname)
                try:
                    result = scan_file(fpath, encoding=encoding)
                    relpath = os.path.relpath(fpath, root)
                    results.append((relpath, result))
                except (OSError, UnicodeDecodeError):
                    continue
    return results


def scan_mixed(text: str) -> tuple[ScanResult, list[ParagraphScore]]:
    """Scan text with per-paragraph analysis for mixed content."""
    result = scan(text)
    paragraphs = detect_paragraphs(text)
    return result, paragraphs
