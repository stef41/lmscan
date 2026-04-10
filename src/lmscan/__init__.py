from __future__ import annotations

__version__ = "0.3.0"

from .scanner import scan, scan_file, scan_directory, scan_mixed
from ._types import ScanResult, TextFeatures, SentenceScore, ModelMatch, ParagraphScore
from .calibration import calibrate, find_optimal_threshold, CalibrationResult, ThresholdConfig
from .languages import detect_language, list_languages, get_language_config, LanguageConfig
from .benchmark import run_benchmark, BenchmarkSample, BenchmarkResult, format_benchmark_report

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
    "detect_language",
    "list_languages",
    "get_language_config",
    "LanguageConfig",
    "run_benchmark",
    "BenchmarkSample",
    "BenchmarkResult",
    "format_benchmark_report",
    "__version__",
]
