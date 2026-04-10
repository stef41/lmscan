from __future__ import annotations

__version__ = "0.1.0"

from .scanner import scan, scan_file, scan_directory, scan_mixed
from ._types import ScanResult, TextFeatures, SentenceScore, ModelMatch, ParagraphScore
from .calibration import calibrate, find_optimal_threshold, CalibrationResult, ThresholdConfig

__all__ = [
    "scan",
    "scan_file",
    "scan_directory",
    "scan_mixed",
    "ScanResult",
    "TextFeatures",
    "SentenceScore",
    "ModelMatch",
    "ParagraphScore",
    "calibrate",
    "find_optimal_threshold",
    "CalibrationResult",
    "ThresholdConfig",
    "__version__",
]
