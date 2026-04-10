from __future__ import annotations

import os
import tempfile

from lmscan.scanner import scan_directory
from lmscan.report import format_directory_report
from lmscan._types import ScanResult


def _make_dir(files: dict[str, str]) -> str:
    """Create a temp directory with the given {filename: content} mapping."""
    d = tempfile.mkdtemp()
    for name, content in files.items():
        path = os.path.join(d, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
    return d


_HUMAN_TEXT = (
    "I went to the grocery store yesterday. The cashier was really friendly. "
    "My dog ran around the yard when I got home."
)
_AI_TEXT = (
    "Let's delve into the multifaceted tapestry of innovation. It's important to "
    "note that a holistic approach is crucial. Furthermore, this endeavor requires "
    "meticulous attention to detail and a comprehensive understanding of the landscape."
)


def test_scan_directory_basic():
    d = _make_dir({"a.txt": _HUMAN_TEXT, "b.txt": _AI_TEXT})
    results = scan_directory(d)
    assert len(results) == 2
    assert all(isinstance(r[1], ScanResult) for r in results)


def test_scan_directory_empty_dir():
    d = tempfile.mkdtemp()
    results = scan_directory(d)
    assert results == []


def test_scan_directory_skips_non_text():
    d = _make_dir({
        "essay.txt": _HUMAN_TEXT,
        "image.png": "binary data",
        "data.csv": "a,b,c",
    })
    results = scan_directory(d)
    assert len(results) == 1
    assert results[0][0] == "essay.txt"


def test_scan_directory_custom_extensions():
    d = _make_dir({
        "readme.md": _HUMAN_TEXT,
        "data.csv": "a,b,c",
    })
    results = scan_directory(d, extensions=(".csv",))
    assert len(results) == 1
    assert results[0][0] == "data.csv"


def test_scan_directory_nested():
    d = _make_dir({
        "top.txt": _HUMAN_TEXT,
        os.path.join("sub", "deep.txt"): _AI_TEXT,
    })
    results = scan_directory(d)
    assert len(results) == 2
    names = {r[0] for r in results}
    assert "top.txt" in names
    assert os.path.join("sub", "deep.txt") in names


def test_scan_directory_md_files():
    d = _make_dir({"readme.md": _HUMAN_TEXT})
    results = scan_directory(d)
    assert len(results) == 1


def test_scan_directory_sorted_order():
    d = _make_dir({"c.txt": "c", "a.txt": "a", "b.txt": "b"})
    results = scan_directory(d)
    names = [r[0] for r in results]
    assert names == sorted(names)


def test_format_directory_report_basic():
    d = _make_dir({"essay.txt": _HUMAN_TEXT})
    results = scan_directory(d)
    report = format_directory_report(results, dirname="test_dir")
    assert "test_dir" in report
    assert "essay.txt" in report
    assert "files scanned" in report


def test_format_directory_report_empty():
    report = format_directory_report([], dirname="empty")
    assert "0 files scanned" in report


def test_format_directory_report_flagged_count():
    d = _make_dir({"a.txt": _HUMAN_TEXT, "b.txt": _HUMAN_TEXT})
    results = scan_directory(d)
    report = format_directory_report(results, dirname="d")
    assert "2 files scanned" in report


def test_scan_directory_skips_bad_encoding():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "bad.txt")
    with open(path, "wb") as f:
        f.write(b"\xff\xfe" + b"\x00" * 100)
    results = scan_directory(d)
    # Should either skip or succeed, not crash
    assert isinstance(results, list)
