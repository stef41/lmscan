from __future__ import annotations

from ._types import ScanResult
from .detector import detect
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
