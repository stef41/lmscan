from __future__ import annotations

import os
import tempfile

from lmscan.cli import main


def test_cli_version(capsys):
    try:
        main(["--version"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "lmscan" in captured.out


def test_cli_direct_text(capsys):
    rc = main(["This is a test sentence for AI detection analysis."])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Verdict" in captured.out


def test_cli_file(capsys):
    text = "This is content in a file for CLI testing purposes."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        f.flush()
        path = f.name
    try:
        rc = main(["--file", path])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Verdict" in captured.out
    finally:
        os.unlink(path)


def test_cli_file_as_positional(capsys):
    text = "Testing positional file argument for CLI."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        f.flush()
        path = f.name
    try:
        rc = main([path])
        assert rc == 0
    finally:
        os.unlink(path)


def test_cli_json_format(capsys):
    rc = main(["--format", "json", "Some text to analyze for JSON output testing."])
    assert rc == 0
    captured = capsys.readouterr()
    import json
    data = json.loads(captured.out)
    assert "ai_probability" in data
    assert "verdict" in data


def test_cli_threshold_pass(capsys):
    # Human-like text with high threshold → should pass
    text = "I went to the store. Got milk. Weather was bad."
    rc = main(["--threshold", "0.99", text])
    assert rc == 0


def test_cli_threshold_fail(capsys):
    # AI-loaded text with very low threshold → should fail
    text = (
        "In today's rapidly evolving landscape, it's important to note that we must "
        "delve into the tapestry of innovation and leverage holistic synergy to foster "
        "multifaceted paradigms. Moreover, the interplay between transformative approaches "
        "underscores the pivotal role of comprehensive cutting-edge strategies."
    )
    rc = main(["--threshold", "0.01", text])
    assert rc == 1


def test_cli_nonexistent_file(capsys):
    rc = main(["--file", "/tmp/_nonexistent_lmscan_cli_test_999.txt"])
    assert rc == 1


def test_cli_no_input(capsys, monkeypatch):
    # Simulate a TTY so stdin path is skipped
    monkeypatch.setattr("sys.stdin", type("FakeTTY", (), {"isatty": lambda self: True, "read": lambda self: ""})())
    rc = main([])
    assert rc == 1


def test_cli_sentences_flag(capsys):
    text = (
        "This is the first sentence for testing. This is the second sentence for testing. "
        "And one more sentence to ensure per-sentence output works correctly."
    )
    rc = main(["--sentences", text])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Per-Sentence" in captured.out
