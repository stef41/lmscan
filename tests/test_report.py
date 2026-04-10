from __future__ import annotations

import json

from lmscan.report import format_report, format_json
from lmscan.scanner import scan


_SAMPLE_TEXT = (
    "In today's rapidly evolving landscape, it's important to note that the multifaceted "
    "nature of modern technology requires a holistic approach. We must delve into the tapestry "
    "of innovation and leverage cutting-edge solutions to foster robust paradigms."
)


def _make_result():
    return scan(_SAMPLE_TEXT)


def test_format_report_contains_verdict():
    report = format_report(_make_result())
    assert "Verdict" in report


def test_format_report_contains_features():
    report = format_report(_make_result())
    assert "Burstiness" in report
    assert "Vocabulary richness" in report


def test_format_report_contains_model_attribution():
    result = _make_result()
    report = format_report(result)
    if result.model_attribution:
        assert "Model Attribution" in report


def test_format_report_contains_flags():
    result = _make_result()
    report = format_report(result)
    if result.flags:
        assert "Flags" in report


def test_format_json_valid():
    result = _make_result()
    j = format_json(result)
    data = json.loads(j)
    assert isinstance(data, dict)


def test_format_json_contains_fields():
    result = _make_result()
    data = json.loads(format_json(result))
    assert "ai_probability" in data
    assert "verdict" in data
    assert "features" in data
    assert "sentence_scores" in data


def test_format_json_no_raw_text():
    result = _make_result()
    data = json.loads(format_json(result))
    assert "text" not in data


def test_format_report_with_sentences():
    result = _make_result()
    report = format_report(result, show_sentences=True)
    assert "Per-Sentence" in report


def test_format_report_without_sentences():
    result = _make_result()
    report = format_report(result, show_sentences=False)
    assert "Per-Sentence" not in report


def test_format_report_scan_time():
    result = _make_result()
    report = format_report(result)
    assert "Scanned in" in report
