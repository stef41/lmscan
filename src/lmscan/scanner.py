from __future__ import annotations

import os

from ._types import ScanResult, ParagraphScore
from .detector import detect, detect_paragraphs
from .fingerprint import fingerprint


def scan(text: str) -> ScanResult:
    """Scan text for AI-generated content and identify the source model."""
    result = detect(text)
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
