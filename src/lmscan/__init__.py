from __future__ import annotations

__version__ = "0.1.0"

from .scanner import scan, scan_file
from ._types import ScanResult, TextFeatures, SentenceScore, ModelMatch

__all__ = [
    "scan",
    "scan_file",
    "ScanResult",
    "TextFeatures",
    "SentenceScore",
    "ModelMatch",
    "__version__",
]
