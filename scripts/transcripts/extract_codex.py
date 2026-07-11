#!/usr/bin/env python3
"""Extract mainline user/agent text from a Codex JSONL transcript.

This uses event_msg user_message/agent_message records to avoid duplicated
response_item records and to ignore tool calls, tool outputs, system, and
developer messages.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GAP_SECONDS = 60 * 60


@dataclass
class Message:
    ts: datetime | None
    role: str
    text: str
    source_line: int
    model: str | None = None
    model_provider: str | None = None
    agent_runtime_version: str | None = None
    reasoning_effort: str | None = None
    transcript_scaffolding: bool = False
    subagent: str | None = None
    source_id: str | None = None


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text.strip())
    return text


def redact_visible_timestamps(text: str) -> str:
    # Keep calendar dates visible, but remove clock-time detail from any
    # summarizer-facing transcript text.
    text = re.sub(
        r"\b(20\d{2})-(\d{2})-(\d{2})T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b",
        r"\1-\2-\3",
        text,
    )
    text = re.sub(
        r"\b(20\d{2})-(\d{2})-(\d{2})\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s?(?:AM|PM|am|pm|EDT|EST|UTC|Z))?\b",
        r"\1-\2-\3",
        text,
    )
    text = re.sub(r"\b(20\d{2})(\d{2})(\d{2})\d{6}\b", r"\1-\2-\3", text)
    text = re.sub(r"\b(\d{2}-\d{2})\s+\d{1,2}:\d{2}(?::\d{2})?\b", r"\1", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?(?:AM|PM|am|pm|EDT|EST|UTC|Z))?\b", "[time redacted]", text)
    text = re.sub(r"\btimestamped\b", "date-stamped", text, flags=re.IGNORECASE)
    text = re.sub(r"\btimestamps\b", "dates", text, flags=re.IGNORECASE)
    text = re.sub(r"\btimestamp\b", "date", text, flags=re.IGNORECASE)
    return text


def is_noise_text(text: str) -> bool:
    stripped = text.lstrip()
    noise_prefixes = (
        "<task-notification>",
        "<subagent_notification>",
        "<heartbeat>",
    )
    if stripped.startswith(noise_prefixes):
        return True
    noise_markers = (
        "<task-id>",
        "<automation_id>",
        "<subagent_notification>",
    )
    return any(marker in stripped for marker in noise_markers)


def is_compaction_record(row: dict[str, Any]) -> bool:
    if row.get("type") == "compacted":
        return True
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return False
    return payload.get("type") in {"context_compacted", "compaction"}


def is_transcript_scaffolding_record(row: dict[str, Any]) -> bool:
    if row.get("isCompactSummary") or row.get("isVisibleInTranscriptOnly"):
        return True
    payload = row.get("payload")
    return isinstance(payload, dict) and bool(
        payload.get("isCompactSummary") or payload.get("isVisibleInTranscriptOnly")
    )


def response_item_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}:
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n\n".join(parts)


def subagent_session_metadata(parent_thread_id: str) -> dict[str, dict[str, str]]:
    """Map agent paths to child conversation/model metadata for one parent."""
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.is_dir():
        return {}
    result: dict[str, dict[str, str]] = {}
    for candidate in sorted(sessions_root.glob("**/*.jsonl")):
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                first = json.loads(handle.readline())
                payload = first.get("payload") if first.get("type") == "session_meta" else None
                if not isinstance(payload, dict) or payload.get("parent_thread_id") != parent_thread_id:
                    continue
                source = payload.get("source")
                subagent = source.get("subagent") if isinstance(source, dict) else None
                spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
                agent_path = spawn.get("agent_path") if isinstance(spawn, dict) else None
                if not isinstance(agent_path, str):
                    continue
                info = {
                    "source_id": str(payload.get("id") or candidate.stem),
                    "model_provider": str(payload.get("model_provider") or ""),
                    "agent_runtime_version": str(payload.get("cli_version") or ""),
                }
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("type") != "turn_context":
                        continue
                    context = row.get("payload")
                    if not isinstance(context, dict):
                        continue
                    if context.get("model"):
                        info["model"] = str(context["model"])
                    if context.get("effort"):
                        info["reasoning_effort"] = str(context["effort"])
                result[agent_path] = info
        except (OSError, json.JSONDecodeError):
            continue
    return result


def iter_messages(
    path: Path,
    *,
    include_transcript_scaffolding: bool = False,
    include_subagent_finals: bool = True,
) -> tuple[str, str | None, list[Message]]:
    thread_id = path.stem.rsplit("-", 1)[-1]
    cwd: str | None = None
    model_provider: str | None = None
    agent_runtime_version: str | None = None
    current_model: str | None = None
    current_effort: str | None = None
    out: list[Message] = []
    subagent_finals: list[Message] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if is_compaction_record(row):
                continue
            transcript_scaffolding = is_transcript_scaffolding_record(row)
            if transcript_scaffolding and not include_transcript_scaffolding:
                continue
            if row.get("type") == "session_meta":
                payload = row.get("payload") or {}
                thread_id = payload.get("id") or payload.get("session_id") or thread_id
                cwd = payload.get("cwd") or cwd
                model_provider = payload.get("model_provider") or model_provider
                agent_runtime_version = payload.get("cli_version") or agent_runtime_version
                continue
            if row.get("type") == "turn_context":
                payload = row.get("payload") or {}
                cwd = payload.get("cwd") or cwd
                current_model = payload.get("model") or current_model
                current_effort = payload.get("effort") or current_effort
                continue
            if row.get("type") == "response_item" and include_subagent_finals:
                payload = row.get("payload")
                if isinstance(payload, dict) and payload.get("type") == "agent_message":
                    author = payload.get("author")
                    recipient = payload.get("recipient")
                    text = clean_text(response_item_text(payload.get("content")))
                    if (
                        isinstance(author, str)
                        and author != "/root"
                        and recipient == "/root"
                        and text.startswith("Message Type: FINAL_ANSWER")
                    ):
                        subagent_finals.append(
                            Message(
                                parse_ts(row.get("timestamp")),
                                "assistant",
                                redact_visible_timestamps(text),
                                line_no,
                                model_provider=model_provider,
                                agent_runtime_version=agent_runtime_version,
                                subagent=author,
                            )
                        )
                continue
            if row.get("type") != "event_msg":
                continue

            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            kind = payload.get("type")
            if kind == "user_message":
                role = "user"
                text = payload.get("message")
            elif kind == "agent_message":
                role = "assistant"
                text = payload.get("message")
            else:
                continue

            if not isinstance(text, str):
                continue
            text = clean_text(text)
            if not text:
                continue
            if is_noise_text(text):
                continue
            text = redact_visible_timestamps(text)
            out.append(
                Message(
                    parse_ts(row.get("timestamp")),
                    role,
                    text,
                    line_no,
                    model=current_model,
                    model_provider=model_provider,
                    agent_runtime_version=agent_runtime_version,
                    reasoning_effort=current_effort,
                    transcript_scaffolding=transcript_scaffolding,
                )
            )
    if include_subagent_finals:
        metadata = subagent_session_metadata(thread_id)
        for message in subagent_finals:
            info = metadata.get(message.subagent or "", {})
            message.source_id = info.get("source_id") or message.subagent
            message.model = info.get("model") or message.model
            message.model_provider = info.get("model_provider") or message.model_provider
            message.agent_runtime_version = (
                info.get("agent_runtime_version") or message.agent_runtime_version
            )
            message.reasoning_effort = info.get("reasoning_effort") or message.reasoning_effort
        main_texts = {message.text for message in out}
        out.extend(message for message in subagent_finals if message.text not in main_texts)
    for message in out:
        if message.source_id is None:
            message.source_id = thread_id
    out = sorted(
        out,
        key=lambda m: (m.ts is None, m.ts or datetime.max.replace(tzinfo=timezone.utc), m.source_line),
    )
    return thread_id, cwd, out


def split_segments(messages: list[Message]) -> list[list[Message]]:
    segments: list[list[Message]] = []
    current: list[Message] = []
    previous_ts: datetime | None = None
    previous_day: str | None = None
    for msg in messages:
        day = msg.ts.date().isoformat() if msg.ts else previous_day
        gap = bool(msg.ts and previous_ts and (msg.ts - previous_ts).total_seconds() > GAP_SECONDS)
        day_changed = bool(current and day and previous_day and day != previous_day)
        if current and (gap or day_changed):
            segments.append(current)
            current = []
        current.append(msg)
        if msg.ts:
            previous_ts = msg.ts
            previous_day = msg.ts.date().isoformat()
    if current:
        segments.append(current)
    return segments


def slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "transcript"


def fmt_ts(ts: datetime | None) -> str:
    return ts.isoformat().replace("+00:00", "Z") if ts else "unknown"


def heading_metadata(msg: Message) -> str:
    if msg.role != "assistant":
        return ""
    fields: list[str] = []
    if msg.model:
        fields.append(f"model={msg.model}")
    if msg.model_provider:
        fields.append(f"provider={msg.model_provider}")
    if msg.agent_runtime_version:
        fields.append(f"codex_cli={msg.agent_runtime_version}")
    if msg.reasoning_effort:
        fields.append(f"effort={msg.reasoning_effort}")
    if msg.subagent:
        fields.append(f"subagent={msg.subagent}")
    return "  [" + "; ".join(fields) + "]" if fields else ""


def write_segment(path: Path, thread_id: str, cwd: str | None, date: str, sequence_in_date: int, segment: list[Message]) -> None:
    visible_segment = [msg for msg in segment if not msg.transcript_scaffolding]
    lines = [
        "---",
        "platform: codex",
        f"thread_id: {thread_id}",
        f"cwd: {cwd or 'unknown'}",
        f"date_utc: {date}",
        f"sequence_in_date: {sequence_in_date:03d}",
        f"messages: {len(visible_segment)}",
        "assistant_metadata: model, provider, Codex CLI version, and reasoning effort are included on assistant headings when present in the source JSONL.",
        "---",
        "",
    ]
    for message_idx, msg in enumerate(visible_segment, 1):
        lines.append(f"## Message {message_idx:03d} - {msg.role}{heading_metadata(msg)}")
        lines.append("")
        lines.append(msg.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--no-subagent-finals", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    thread_id, cwd, messages = iter_messages(
        args.source,
        include_subagent_finals=not args.no_subagent_finals,
    )
    segments = split_segments(messages)

    per_day_counts: dict[str, int] = {}
    for i, segment in enumerate(segments, 1):
        start = next((m.ts for m in segment if m.ts), None)
        day = start.date().isoformat() if start else "unknown-date"
        per_day_counts[day] = per_day_counts.get(day, 0) + 1
        out_path = args.out_dir / day / f"{per_day_counts[day]:03d}-codex.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_segment(out_path, thread_id, cwd, day, per_day_counts[day], segment)

    print(f"source={args.source}")
    print(f"thread_id={thread_id}")
    print(f"cwd={cwd}")
    print(f"messages={len(messages)}")
    print(f"segments={len(segments)}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
