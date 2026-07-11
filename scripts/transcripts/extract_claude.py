#!/usr/bin/env python3
"""Extract mainline user/assistant text from a Claude Code JSONL transcript.

This intentionally ignores tool use/results, attachments, system-ish meta records,
and Claude Code sidechain/subagent rows.
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
SUMMARY_WORKER_PROMPT_PREFIXES = (
    "You are summarizing mainline project conversation",
    "You are updating an existing mainline project conversation summary",
    "You are summarizing a contiguous segment of mainline project conversation",
    "You are writing notes/",
    "You are writing the sparse hierarchical",
)


@dataclass
class Message:
    ts: datetime | None
    role: str
    text: str
    source_line: int
    model: str | None = None
    agent_runtime_version: str | None = None
    transcript_scaffolding: bool = False
    subagent: str | None = None


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
        "<local-command-",
        "<command-name>",
        "<task-notification>",
        "<subagent_notification>",
        "<heartbeat>",
    )
    if stripped.startswith(noise_prefixes):
        return True
    noise_markers = (
        "<local-command-stdout>",
        "<local-command-stderr>",
        "<task-id>",
        "<automation_id>",
        "<subagent_notification>",
    )
    return any(marker in stripped for marker in noise_markers)


def is_transcript_scaffolding_row(row: dict[str, Any]) -> bool:
    return bool(row.get("isCompactSummary") or row.get("isVisibleInTranscriptOnly"))


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in {"text", "input_text", "output_text"} and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n\n".join(parts)
    return ""


def is_summary_worker_transcript(path: Path) -> bool:
    """Reject a standalone CLI summary session before extracting its answer.

    Print-mode Claude sessions contain a synthetic user message, so merely
    requiring a user row does not prevent recursive self-ingestion. Summary
    workers are separate sessions whose first real user prompt is our known
    summary instruction.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("isSidechain") or row.get("isMeta"):
                continue
            msg = row.get("message")
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            text = clean_text(text_from_content(msg.get("content")))
            return text.startswith(SUMMARY_WORKER_PROMPT_PREFIXES)
    return False


def subagent_final_messages(path: Path) -> list[Message]:
    """Return only the last nonempty assistant text from each Claude subagent."""
    subagents_dir = path.parent / path.stem / "subagents"
    if not subagents_dir.is_dir():
        return []
    finals: list[Message] = []
    for subagent_path in sorted(subagents_dir.glob("*.jsonl")):
        final: Message | None = None
        with subagent_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = row.get("message")
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                text = clean_text(text_from_content(msg.get("content")))
                if not text or is_noise_text(text):
                    continue
                agent_id = row.get("agentId") or subagent_path.stem.removeprefix("agent-")
                final = Message(
                    parse_ts(row.get("timestamp")),
                    "assistant",
                    redact_visible_timestamps(text),
                    line_no,
                    model=msg.get("model") if isinstance(msg.get("model"), str) else None,
                    agent_runtime_version=(
                        row.get("version") if isinstance(row.get("version"), str) else None
                    ),
                    subagent=str(agent_id),
                )
        if final is not None:
            finals.append(final)
    return finals


def iter_messages(
    path: Path,
    *,
    include_transcript_scaffolding: bool = False,
    include_subagent_finals: bool = True,
) -> list[Message]:
    if is_summary_worker_transcript(path):
        return []
    out: list[Message] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("isSidechain"):
                continue
            transcript_scaffolding = is_transcript_scaffolding_row(row)
            if transcript_scaffolding and not include_transcript_scaffolding:
                continue
            if row.get("type") not in {"user", "assistant"}:
                continue
            if row.get("isMeta"):
                continue

            msg = row.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in {"user", "assistant"}:
                continue

            text = clean_text(text_from_content(msg.get("content")))
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
                    model=msg.get("model") if isinstance(msg.get("model"), str) else None,
                    agent_runtime_version=row.get("version") if isinstance(row.get("version"), str) else None,
                    transcript_scaffolding=transcript_scaffolding,
                )
            )
    if include_subagent_finals:
        main_texts = {message.text for message in out}
        out.extend(message for message in subagent_final_messages(path) if message.text not in main_texts)
    return sorted(
        out,
        key=lambda m: (m.ts is None, m.ts or datetime.max.replace(tzinfo=timezone.utc), m.source_line),
    )


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
    if msg.agent_runtime_version:
        fields.append(f"claude_code_version={msg.agent_runtime_version}")
    if msg.subagent:
        fields.append(f"subagent={msg.subagent}")
    return "  [" + "; ".join(fields) + "]" if fields else ""


def write_segment(path: Path, session_id: str, date: str, sequence_in_date: int, segment: list[Message]) -> None:
    visible_segment = [msg for msg in segment if not msg.transcript_scaffolding]
    lines = [
        "---",
        "platform: claude-code",
        f"session_id: {session_id}",
        f"date_utc: {date}",
        f"sequence_in_date: {sequence_in_date:03d}",
        f"messages: {len(visible_segment)}",
        "assistant_metadata: model and Claude Code version are included on assistant headings when present in the source JSONL.",
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
    messages = iter_messages(args.source, include_subagent_finals=not args.no_subagent_finals)
    segments = split_segments(messages)

    session_id = args.source.stem
    per_day_counts: dict[str, int] = {}
    for i, segment in enumerate(segments, 1):
        start = next((m.ts for m in segment if m.ts), None)
        day = start.date().isoformat() if start else "unknown-date"
        per_day_counts[day] = per_day_counts.get(day, 0) + 1
        out_path = args.out_dir / day / f"{per_day_counts[day]:03d}-claude-code.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_segment(out_path, session_id, day, per_day_counts[day], segment)

    print(f"source={args.source}")
    print(f"messages={len(messages)}")
    print(f"segments={len(segments)}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
