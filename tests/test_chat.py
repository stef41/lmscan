from __future__ import annotations

import json
import os
import tempfile

from lmscan._types import (
    ChatMessage,
    ChatScanResult,
    MessageScore,
    ParticipantScore,
)
from lmscan.chat import parse_chat, scan_chat, scan_chat_file
from lmscan.cli import main

# ── Fixtures ──────────────────────────────────────────────────────────────────

_AI_TEXT = (
    "Let's delve into the multifaceted tapestry of innovation. Furthermore, "
    "it's important to note that a holistic approach is crucial for fostering "
    "comprehensive synergy across the entire landscape of modern endeavors. "
    "Moreover, we must leverage cutting-edge paradigms to navigate the "
    "complexities of this realm."
)

_HUMAN_1 = (
    "I went to the grocery store yesterday and bought some apples. The "
    "cashier was really friendly and we chatted about the weather. It was "
    "raining pretty hard on the way home so I got soaked."
)

_HUMAN_2 = (
    "So my sister called me last night about the wedding plans. She wants to "
    "do it in the backyard which honestly sounds like a nightmare. We argued "
    "for like an hour about the caterer and then she hung up on me."
)

_HUMAN_3 = (
    "The old fishing boat had seen better days. Paint peeled off the hull in "
    "long strips. Every time a wave hit, the whole thing groaned like it was "
    "about to give up. My grandfather refused to sell it anyway."
)

_HUMAN_4 = (
    "I tried making sourdough again and it came out flat. Not sure what went "
    "wrong, maybe the starter was dead. Third time this month. Might just buy "
    "bread like a normal person from now on."
)


def _mixed_dialog() -> dict:
    return {
        "messages": [
            {"role": "user", "content": _HUMAN_1},
            {"role": "assistant", "content": _AI_TEXT},
            {"role": "user", "content": _HUMAN_2},
            {"role": "assistant", "content": _AI_TEXT},
        ]
    }


# ── Parser tests ──────────────────────────────────────────────────────────────

def test_parse_openai_style():
    msgs = parse_chat(_mixed_dialog())
    assert len(msgs) == 4
    assert all(isinstance(m, ChatMessage) for m in msgs)
    assert [m.speaker for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0].index == 0
    assert msgs[3].index == 3


def test_parse_generic_speaker_keys():
    data = [
        {"from": "Alice", "text": "one"},
        {"speaker": "Bob", "message": "two"},
        {"author": "Alice", "body": "three"},
        {"name": "Bob", "content": "four"},
        {"sender": "Alice", "utterance": "five"},
    ]
    msgs = parse_chat(data)
    assert [m.speaker for m in msgs] == ["Alice", "Bob", "Alice", "Bob", "Alice"]
    assert [m.text for m in msgs] == ["one", "two", "three", "four", "five"]


def test_parse_accepts_json_string():
    raw = json.dumps(_mixed_dialog())
    msgs = parse_chat(raw)
    assert len(msgs) == 4


def test_parse_participants_and_turns():
    data = {"participants": ["A", "B"], "turns": [{"text": "hello"}, {"text": "hi"}]}
    msgs = parse_chat(data)
    assert [m.speaker for m in msgs] == ["A", "B"]


def test_parse_plain_string_list():
    msgs = parse_chat(["first turn", "second turn", "third turn"])
    assert len(msgs) == 3
    assert [m.speaker for m in msgs] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_parse_single_message_object():
    msgs = parse_chat({"speaker": "A", "text": "just one message here"})
    assert len(msgs) == 1
    assert msgs[0].speaker == "A"


def test_parse_skips_system_messages():
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi there"},
        ]
    }
    msgs = parse_chat(data)
    assert len(msgs) == 1
    assert msgs[0].role == "user"


def test_parse_skips_empty_and_non_string():
    data = [
        {"speaker": "A", "text": "   "},
        {"speaker": "A", "text": 123},
        {"speaker": "A", "text": "real content here"},
    ]
    msgs = parse_chat(data)
    assert len(msgs) == 1
    assert msgs[0].text == "real content here"


def test_parse_generates_default_speaker():
    msgs = parse_chat([{"text": "no speaker given"}])
    assert msgs[0].speaker == "Speaker 1"


def test_parse_invalid_json_raises():
    try:
        parse_chat("this is not json")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_parse_unknown_shape_raises():
    for bad in ["{\"foo\": 1}", "[1, 2, 3]", "{}"]:
        try:
            parse_chat(bad)
            assert False, f"Expected ValueError for {bad!r}"
        except ValueError:
            pass


def test_parse_empty_list_raises():
    try:
        parse_chat([])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_parse_unsupported_type_raises():
    try:
        parse_chat(42)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ── Scan tests ────────────────────────────────────────────────────────────────

def test_scan_chat_returns_result():
    result = scan_chat(_mixed_dialog())
    assert isinstance(result, ChatScanResult)
    assert len(result.messages) == 4
    assert len(result.message_scores) == 4
    assert len(result.participants) == 2
    assert result.verdict
    assert result.scan_time_s >= 0.0


def test_scan_chat_detects_ai_participant():
    result = scan_chat(_mixed_dialog())
    assert "assistant" in result.likely_bots
    assert "user" not in result.likely_bots
    participants = {p.name: p for p in result.participants}
    assert participants["assistant"].is_ai is True
    assert participants["user"].is_ai is False


def test_scan_chat_participant_ordering():
    result = scan_chat(_mixed_dialog())
    probs = [p.ai_probability for p in result.participants]
    assert probs == sorted(probs, reverse=True)


def test_scan_chat_message_scores_flagged():
    result = scan_chat(_mixed_dialog())
    ai_msgs = [m for m in result.message_scores if m.speaker == "assistant"]
    human_msgs = [m for m in result.message_scores if m.speaker == "user"]
    assert min(m.ai_probability for m in ai_msgs) > max(
        m.ai_probability for m in human_msgs
    )


def test_scan_chat_human_only_no_bots():
    data = {
        "messages": [
            {"role": "user", "content": _HUMAN_1},
            {"role": "assistant", "content": _HUMAN_2},
            {"role": "user", "content": _HUMAN_3},
            {"role": "assistant", "content": _HUMAN_4},
        ]
    }
    result = scan_chat(data)
    assert result.likely_bots == []
    assert "No AI bot detected" in result.verdict


def test_scan_chat_participant_fields():
    result = scan_chat(_mixed_dialog())
    for p in result.participants:
        assert isinstance(p, ParticipantScore)
        assert p.message_count == 2
        assert p.word_count > 0
        assert 0.0 <= p.ai_probability <= 1.0
        assert p.verdict != ""
        assert p.confidence != ""
        assert isinstance(p.model_attribution, list)


def test_scan_chat_message_score_fields():
    result = scan_chat(_mixed_dialog())
    for m in result.message_scores:
        assert isinstance(m, MessageScore)
        assert m.verdict != ""
        assert 0.0 <= m.ai_probability <= 1.0


def test_scan_chat_accepts_preparsed_messages():
    msgs = parse_chat(["hello there friend", "another line of text"])
    result = scan_chat(msgs)
    assert len(result.messages) == 2


def test_scan_chat_threshold_override():
    # With threshold 0.0 every participant is flagged.
    result = scan_chat(_mixed_dialog(), threshold=0.0)
    assert len(result.likely_bots) == 2


def test_scan_chat_single_speaker():
    data = {"messages": [{"role": "user", "content": _HUMAN_1}]}
    result = scan_chat(data)
    assert len(result.participants) == 1
    assert result.participants[0].name == "user"


def test_scan_chat_short_messages():
    data = {
        "messages": [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "sure"},
        ]
    }
    result = scan_chat(data)
    assert len(result.participants) == 2


def test_scan_chat_source_name():
    result = scan_chat(_mixed_dialog(), source_name="demo.json")
    assert result.source == "demo.json"


def test_scan_chat_all_ai_dialog():
    data = {
        "messages": [
            {"role": "user", "content": _AI_TEXT},
            {"role": "assistant", "content": _AI_TEXT},
            {"role": "user", "content": _AI_TEXT},
            {"role": "assistant", "content": _AI_TEXT},
        ]
    }
    result = scan_chat(data)
    assert len(result.likely_bots) == 2


def test_scan_chat_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(_mixed_dialog(), f)
        path = f.name
    try:
        result = scan_chat_file(path)
        assert result.source == path
        assert "assistant" in result.likely_bots
    finally:
        os.unlink(path)


def test_scan_chat_file_nonexistent():
    try:
        scan_chat_file("/tmp/_nonexistent_lmscan_chat_999.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ── Report tests ──────────────────────────────────────────────────────────────

def test_format_chat_report():
    from lmscan.report import format_chat_report

    result = scan_chat(_mixed_dialog())
    out = format_chat_report(result)
    assert "Chat Dialog Forensics" in out
    assert "Participant Analysis" in out
    assert "assistant" in out


def test_format_chat_json():
    from lmscan.report import format_chat_json

    result = scan_chat(_mixed_dialog())
    data = json.loads(format_chat_json(result))
    assert "participants" in data
    assert "likely_bots" in data
    assert "message_scores" in data
    # Full message text should be dropped.
    assert "text" not in data["messages"][0]


# ── CLI tests ─────────────────────────────────────────────────────────────────

def _write_chat_file() -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(_mixed_dialog(), f)
        return f.name


def test_cli_chat_text_output(capsys):
    path = _write_chat_file()
    try:
        rc = main(["--chat", path])
        captured = capsys.readouterr()
        assert "Chat Dialog Forensics" in captured.out
        # Exit code 1 because a bot was detected.
        assert rc == 1
    finally:
        os.unlink(path)


def test_cli_chat_json_output(capsys):
    path = _write_chat_file()
    try:
        rc = main(["--chat", "--format", "json", path])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "participants" in data
        assert "likely_bots" in data
        assert rc == 1
    finally:
        os.unlink(path)


def test_cli_json_file_auto_chat(capsys):
    # A .json file should auto-enable chat mode without --chat.
    path = _write_chat_file()
    try:
        main(["--file", path])
        captured = capsys.readouterr()
        assert "Chat Dialog Forensics" in captured.out
    finally:
        os.unlink(path)


def test_cli_chat_human_only_exit_zero(capsys):
    data = {
        "messages": [
            {"role": "user", "content": _HUMAN_1},
            {"role": "assistant", "content": _HUMAN_2},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        rc = main(["--chat", path])
        assert rc == 0
    finally:
        os.unlink(path)


def test_cli_chat_threshold(capsys):
    data = {
        "messages": [
            {"role": "user", "content": _HUMAN_1},
            {"role": "assistant", "content": _HUMAN_2},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        # Very low threshold flags everything.
        rc = main(["--chat", "--threshold", "0.01", path])
        assert rc == 1
    finally:
        os.unlink(path)
