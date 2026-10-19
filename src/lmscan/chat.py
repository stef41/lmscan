"""Chat / dialog analysis for lmscan.

Given a chat transcript (a dialog between two or more parties) as JSON, this
module identifies whether one of the participants is an AI bot.

Design: individual messages are usually short and therefore noisy when scored
in isolation.  We therefore compute two levels of evidence:

1. **Per message** — the lightweight signal detector
   (:func:`lmscan.detector.detect`) is run on every turn.  This catches
   individual bot messages without paying for model fingerprinting.
2. **Per participant** — all turns belonging to one speaker are concatenated
   and run through the full :func:`lmscan.scanner.scan` ensemble (detector +
   trained classifier + perplexity) plus model fingerprinting.  The aggregate
   verdict is far more robust than any single message.

The conversation verdict flags every participant whose aggregate probability
crosses ``threshold`` as a likely AI bot.

Supported JSON shapes are auto-detected by :func:`parse_chat`; see its
docstring for the accepted formats.
"""
from __future__ import annotations

import json
import time

from ._types import (
    ChatMessage,
    ChatScanResult,
    MessageScore,
    ParticipantScore,
)
from .detector import _verdict, detect

# ── Flexible JSON parsing ─────────────────────────────────────────────────────

# Keys that may hold the list of messages inside a top-level object.
_MESSAGE_LIST_KEYS: tuple[str, ...] = (
    "messages",
    "turns",
    "conversation",
    "dialogue",
    "dialog",
    "chat",
    "transcript",
)

# Keys that may name the speaker of a message.
_SPEAKER_KEYS: tuple[str, ...] = (
    "speaker",
    "from",
    "author",
    "name",
    "sender",
    "user",
    "role",
    "participant",
)

# Keys that may hold the body text of a message.
_TEXT_KEYS: tuple[str, ...] = (
    "text",
    "content",
    "message",
    "body",
    "utterance",
)

_PARTICIPANT_KEYS: tuple[str, ...] = ("participants", "people", "users", "members")


def _default_speaker(index: int, participants: list[str]) -> str:
    """Pick a speaker name for an unlabelled message."""
    if participants:
        return participants[index % len(participants)]
    # Alternate between two generic speakers.
    return f"Speaker {index % 2 + 1}"


def _extract_text(item: dict[str, object]) -> str | None:
    for key in _TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str):
            return value
    return None


def _extract_speaker(item: dict[str, object]) -> str:
    for key in _SPEAKER_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_messages(
    raw_messages: list[object], participants: list[str]
) -> list[ChatMessage]:
    """Turn a raw list of message dicts/strings into :class:`ChatMessage` objects."""
    messages: list[ChatMessage] = []
    index = 0

    for item in raw_messages:
        text: str | None
        if isinstance(item, str):
            text = item
            speaker = ""
            role = ""
        elif isinstance(item, dict):
            text = _extract_text(item)
            if text is None:
                # A dict without a recognised text field cannot be scored.
                continue
            raw_role = item.get("role")
            role = raw_role if isinstance(raw_role, str) else ""
            speaker = _extract_speaker(item)
        else:
            continue

        if not text or not text.strip():
            continue

        # Skip system prompts — they are not spoken by a participant.
        if role.lower() == "system":
            continue

        if not speaker:
            speaker = _default_speaker(index, participants)

        messages.append(ChatMessage(
            text=text,
            speaker=speaker,
            index=index,
            role=role,
        ))
        index += 1

    return messages


def parse_chat(data: object) -> list[ChatMessage]:
    """Normalise *data* into a list of :class:`ChatMessage`.

    Accepts a JSON string, a ``dict``, or a ``list``.  Several common chat
    transcript shapes are auto-detected:

    - OpenAI-style:
      ``{"messages": [{"role": "user", "content": "..."}]}``
    - Generic message list:
      ``[{"speaker": "A", "text": "..."}, ...]``
      (speaker keys: ``speaker``/``from``/``author``/``name``/``sender``;
      text keys: ``text``/``content``/``message``/``body``/``utterance``)
    - Participants + turns:
      ``{"participants": ["A", "B"], "turns": [{"text": "..."}, ...]}``
    - Plain string list:
      ``["first turn", "second turn", ...]``

    ``system`` messages and empty turns are skipped.

    Raises
    ------
    ValueError
        If *data* is not valid JSON, or no recognisable messages are found.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Input is not valid JSON. Provide a chat transcript as JSON."
            ) from exc

    participants: list[str] = []

    if isinstance(data, dict):
        # Collect participant names, if provided.
        for key in _PARTICIPANT_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                participants = [str(v) for v in value if str(v).strip()]
                break

        raw_messages: list[object] | None = None
        for key in _MESSAGE_LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                raw_messages = value
                break

        if raw_messages is None:
            # Maybe the object itself is a single message.
            if _extract_text(data) is not None:
                raw_messages = [data]
            else:
                raise ValueError(
                    "Could not find a message list in the JSON object. Expected a "
                    "list under one of: "
                    + ", ".join(_MESSAGE_LIST_KEYS)
                    + ", or a single message object with a text field."
                )
    elif isinstance(data, list):
        raw_messages = data
    else:
        raise ValueError(
            "Unsupported chat input type "
            f"{type(data).__name__}; expected a JSON string, dict, or list."
        )

    messages = _normalize_messages(raw_messages, participants)

    if not messages:
        raise ValueError(
            "No scorable messages found. Expected non-empty message text; "
            "note that 'system' role messages are ignored."
        )

    return messages


# ── Grouping helpers ──────────────────────────────────────────────────────────

def _group_by_speaker(messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
    """Group messages by speaker, preserving first-appearance order."""
    grouped: dict[str, list[ChatMessage]] = {}
    for msg in messages:
        grouped.setdefault(msg.speaker, []).append(msg)
    return grouped


def _conversation_verdict(
    participants: list[ParticipantScore], likely_bots: list[str]
) -> str:
    if not participants:
        return "No participants"
    if likely_bots and len(likely_bots) == len(participants):
        return "All participants appear AI-generated"
    if likely_bots:
        names = ", ".join(likely_bots)
        return f"Likely AI bot detected: {names}"
    return "No AI bot detected"


# ── Public API ────────────────────────────────────────────────────────────────

def scan_chat(
    source: object, *, threshold: float = 0.5, source_name: str = ""
) -> ChatScanResult:
    """Analyse a chat transcript and identify any AI-bot participant.

    Parameters
    ----------
    source:
        A JSON string, a parsed ``dict``/``list``, or an already-parsed
        ``list[ChatMessage]``.
    threshold:
        Probability at or above which a participant is flagged as an AI bot.
    source_name:
        Optional label stored on the result (e.g. a file name).

    Returns
    -------
    ChatScanResult
        Message-level and participant-level scores plus the conversation
        verdict and the list of ``likely_bots``.
    """
    start = time.monotonic()

    if isinstance(source, list) and all(isinstance(m, ChatMessage) for m in source):
        messages = source
    else:
        messages = parse_chat(source)

    # ── Per-message scoring (lightweight detector) ────────────────────────
    message_scores: list[MessageScore] = []
    for msg in messages:
        prob = detect(msg.text).ai_probability
        message_scores.append(MessageScore(
            index=msg.index,
            speaker=msg.speaker,
            text=msg.text,
            ai_probability=round(prob, 4),
            verdict=_verdict(prob),
        ))

    # ── Per-participant scoring (full ensemble + fingerprint) ─────────────
    from .scanner import scan  # local import avoids a circular import

    grouped = _group_by_speaker(messages)
    participants: list[ParticipantScore] = []

    for name, msgs in grouped.items():
        joined = "\n".join(m.text for m in msgs)
        result = scan(joined)
        is_ai = result.ai_probability >= threshold
        participants.append(ParticipantScore(
            name=name,
            message_count=len(msgs),
            word_count=result.features.word_count,
            ai_probability=result.ai_probability,
            verdict=result.verdict,
            confidence=result.confidence,
            is_ai=is_ai,
            model_attribution=result.model_attribution,
            flags=result.flags,
        ))

    # Rank participants by AI probability (highest first) for stable output.
    participants.sort(key=lambda p: p.ai_probability, reverse=True)
    likely_bots = [p.name for p in participants if p.is_ai]

    conversation_prob = max((p.ai_probability for p in participants), default=0.0)

    elapsed = time.monotonic() - start

    return ChatScanResult(
        source=source_name,
        messages=messages,
        message_scores=message_scores,
        participants=participants,
        likely_bots=likely_bots,
        verdict=_conversation_verdict(participants, likely_bots),
        ai_probability=round(conversation_prob, 4),
        scan_time_s=round(elapsed, 4),
    )


def scan_chat_file(
    path: str,
    encoding: str = "utf-8",
    *,
    threshold: float = 0.5,
) -> ChatScanResult:
    """Read a JSON chat transcript from *path* and analyse it."""
    with open(path, encoding=encoding) as f:
        data = f.read()
    return scan_chat(data, threshold=threshold, source_name=path)
