from __future__ import annotations

import os
import tempfile

from lmscan.scanner import scan, scan_file
from lmscan._types import ScanResult


def test_scan_basic():
    result = scan("This is a test of the scanning functionality for AI detection.")
    assert isinstance(result, ScanResult)


def test_scan_returns_model_attribution():
    text = (
        "Let's delve into this tapestry of innovation. It's important to note that "
        "the landscape of modern technology requires a holistic approach to foster "
        "multifaceted synergy."
    )
    result = scan(text)
    assert isinstance(result.model_attribution, list)


def test_scan_file_with_tempfile():
    text = "This is content written in a file for testing purposes."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        f.flush()
        path = f.name
    try:
        result = scan_file(path)
        assert isinstance(result, ScanResult)
        assert result.text == text
    finally:
        os.unlink(path)


def test_scan_file_nonexistent():
    try:
        scan_file("/tmp/_nonexistent_lmscan_test_file_999.txt")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_scan_preserves_text():
    text = "Hello world, how are things going today?"
    result = scan(text)
    assert result.text == text


def test_scan_has_all_fields():
    result = scan("The quick brown fox jumps over the lazy dog near the river bank.")
    assert result.ai_probability >= 0.0
    assert result.verdict
    assert result.confidence
    assert result.features is not None
    assert isinstance(result.sentence_scores, list)
    assert isinstance(result.model_attribution, list)
    assert isinstance(result.flags, list)
    assert result.scan_time_s >= 0.0


def test_scan_empty():
    result = scan("")
    assert isinstance(result, ScanResult)


def test_scan_long_text():
    text = "This is a sentence. " * 50
    result = scan(text)
    assert result.features.word_count > 100
