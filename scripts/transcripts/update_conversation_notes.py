#!/usr/bin/env python3
"""Incrementally update conversation summary notes.

This script keeps a manifest of which raw transcript message ranges are covered
by each `notes/*-conversation-*.md` file. Future runs use the manifest to avoid
resummarizing already-covered material. If a covered conversation segment has
continued, the script builds an update prompt containing the existing summary,
the previously summarized messages, and the new messages after the cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notes_archive_naming import archive_day_start, compact_prefix, conversation_title, timestamp_from_full_prefix  # noqa: E402
from notes_archive_timestamps import ArchiveTimestampCache, TimestampInfo  # noqa: E402
from notes_summary_filters import (  # noqa: E402
    preemptive_prompt_for_forbidden_referents,
    resolve_forbid_patterns,
)
from summary_model import (  # noqa: E402
    DEFAULT_CODEX_REASONING,
    DEFAULT_PROVIDER,
    PROVIDERS,
    custom_command_provenance,
    resolve_spec,
    wrapper_argv,
)


SOURCE_ORDER = {"claude-code": "0-claude", "codex": "1-codex"}
DEFAULT_MANIFEST = Path("scripts/transcripts/conversation-summary-manifest.json")
DEFAULT_NOTES_DIR = Path("notes")
DEFAULT_WORK_DIR = Path("/tmp/valuegraft_transcript_incremental")
DEFAULT_SUMMARY_SHARDS_DIR = Path("/tmp/valuegraft_transcript_rebuild/summary_shards")
DEFAULT_CLAUDE_JSONL = (
    Path.home()
    / ".claude/projects/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370.jsonl"
)
DEFAULT_CODEX_JSONL = (
    Path.home()
    / ".codex/sessions/2026/07/04/rollout-2026-07-04T22-07-20-019f3007-bab0-7e50-b019-2625d1538f63.jsonl"
)
DEFAULT_FORBID_REGEX = [
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bASIA[0-9A-Z]{16}\b",
    r"\bsk-or-v1-[A-Za-z0-9_-]{32,}\b",
    r"\brpa_[A-Za-z0-9]{32,}\b",
    r"\bhf_[A-Za-z0-9]{20,}\b",
]
PARTICIPANTS_PREFIX = "**Participants:**"
OLD_PARTICIPANTS_SECTION_HEADING = "**Participants in this Conversation.**"
OLD_MODEL_SECTION_HEADING = "**Models in this Conversation.**"
RESERVED_NOTE_NAMES = {"AGENTS.md", "README.md"}
ARCHIVE_SUFFIXES = {".md", ".txt"}
CONVERSATION_SOURCES_HEADING = "## Conversation sources"
FORBIDDEN_RETRY_PROMPT = """\
A previous attempt at this summary used words or phrases that matched forbidden
output filters.

Forbidden matches seen across attempts so far:
{matches}

Rewrite the summary from scratch. Each match identifies an underlying referent
or subject that must become non-identifiable. This is
not a word-ban or synonym substitution exercise. Remove not only the matched term but also synonyms,
euphemisms, distinctive attributes, people, places, events, surrounding facts,
and narrative clues that would let a reader infer the specific subject. Rewrite
or remove the whole sentence or paragraph rather than playing taboo around one
word. If the subject is not essential to project state, omit it entirely. If it
is essential, reduce it to genuinely non-identifying language such as "an
unrelated sensitive-topic tangent" or "a notes-hygiene issue." Do not mention
the filtering rule, the forbidden list, or the previous attempt in the summary.

Original task:

{prompt}
"""

SUMMARY_PROMPT = """\
You are summarizing mainline project conversation for a future agent.

Write a concise but information-dense note body. Capture ideas, hypotheses,
methodology decisions, corrections, concrete results, caveats, operational
lessons, explicit timeline commitments, and handoff-relevant state. If the
conversation established an intended writing form, such as paper-style,
blog-style, article-style, or report-style, include that.

The first paragraph of your answer must be the italicized opening summary, or
at most two short italicized sentences, summarizing what this conversation
covers. Do
not put any title, heading, bold label, or preamble before that first italicized
paragraph. Then use prose paragraphs. If section labels help, use optional bold
paragraph-opening labels such as **Handoff State.** rather than Markdown `#` or
`##` headings. Use bullets only for compact lists of named results, rules, arms,
or open questions; do not turn the whole conversation into a bullet ledger.

Use neutral, professional prose focused on what changed and why. Do not preserve
every exchange. Preserve priority and urgency when it affects future work, but
express it as project priority, blocking status, or required follow-up rather
than participant mood. Do not flatten importance: if emphasis changes what a
future agent should do first, record that priority as a project fact or required
next action. Do not describe participant emotions, temperament, or interpersonal
tone; never use labels such as angry, frustrated, furious, annoyed, or upset.
Describe corrections, disagreements, and requirements as project facts. Do not
quote colorful or emotionally loaded user phrasing;
paraphrase it into neutral project terms. Omit side logistics unless they
directly affect repository workflow. If the transcript discusses arXiv, Zenodo,
ACM, DOI, uploading, posting, author rights, coauthor consent, or venue
selection, omit those details entirely unless a tracked repo artifact was
changed; at most preserve the intended document style or a concrete repo
workflow change. For transcript-note style discussions, record only the final
durable style rule in general terms. Do not retell the cleanup episode, mention
prohibited words, or quote examples of language to avoid. Return only the note
body, with no title. The script adds a deterministic participants paragraph and
an unlinked conversation-source ID list from source metadata; do not invent
model identifiers or source IDs. If the transcript headings
show a switch between assistant models, mention the switch at the relevant point
in the summary flow. Assistant messages labeled `subagent=...` are final
subagent assessments, not tool logs. Preserve their unique conclusions,
evidence, and proposed follow-ups when material, but synthesize them rather than
copying every detail. The deterministic participants paragraph names users and
contributing models from raw metadata. Attribute a subagent contribution to its
source-supported model identifier when available; never substitute its task
label for a participant identity or guess missing model metadata.

Preserve explicit scheduling commitments as handoff facts. If an agent or user
commits to a concrete ETA, deadline, duration, recurrence, check-back interval,
or expected completion window, record what was expected to happen and the
stated timing. If later messages revise, invalidate, or cancel that estimate,
record the update too. Do not infer timestamps or convert vague timing into
false precision; keep relative timing relative when that is all the transcript
provides.

{previous_context_block}

Transcript to summarize:

{transcript}
"""

PREVIOUS_CONTEXT_TEMPLATE = """\
Brief context from the previous {platform} conversation summary:

{context}
"""

REVISION_PROMPT = """\
You are updating an existing mainline project conversation summary.

Return a complete replacement note body. Preserve the useful content from the
existing summary, add the new material after the cutoff, and remove stale wording
if the new material changes the interpretation.

Focus on ideas, hypotheses, methodology decisions, corrections, concrete results,
caveats, operational lessons, explicit timeline commitments, and handoff-relevant
state. If the conversation established an intended writing form, such as
paper-style, blog-style, article-style, or report-style, include that.

The first paragraph of your answer must be the italicized opening summary, or
at most two short italicized sentences, summarizing what this conversation
covers. Do
not put any title, heading, bold label, or preamble before that first italicized
paragraph. Then use prose paragraphs. If section labels help, use optional bold
paragraph-opening labels such as **Handoff State.** rather than Markdown `#` or
`##` headings. Use bullets only for compact lists of named results, rules, arms,
or open questions; do not turn the whole conversation into a bullet ledger.

Use neutral, professional prose. Preserve priority and urgency when it affects
future work, but express it as project priority, blocking status, or required
follow-up rather than participant mood. Do not flatten importance: if emphasis
changes what a future agent should do first, record that priority as a project
fact or required next action. Do not describe participant emotions, temperament,
or interpersonal tone; never use labels such as angry, frustrated, furious,
annoyed, or upset. Describe corrections, disagreements, and
requirements as project facts. Do not quote colorful or emotionally loaded user
phrasing; paraphrase it into neutral project terms. Omit side logistics unless
they directly affect repository workflow. If the transcript discusses arXiv,
Zenodo, ACM, DOI, uploading, posting, author rights, coauthor consent, or venue
selection, omit those details entirely unless a tracked repo artifact was
changed; at most preserve the intended document style or a concrete repo
workflow change. For transcript-note style discussions, record only the final
durable style rule in general terms. Do not retell the cleanup episode, mention
prohibited words, or quote examples of language to avoid. Return only the note
body, with no title. The script adds a deterministic participants paragraph and
an unlinked conversation-source ID list from source metadata; do not invent
model identifiers or source IDs. If the transcript headings
show a switch between assistant models, mention the switch at the relevant point
in the summary flow. Assistant messages labeled `subagent=...` are final
subagent assessments, not tool logs. Preserve their unique conclusions,
evidence, and proposed follow-ups when material, but synthesize them rather than
copying every detail. The deterministic participants paragraph names users and
contributing models from raw metadata. Attribute a subagent contribution to its
source-supported model identifier when available; never substitute its task
label for a participant identity or guess missing model metadata.

Preserve explicit scheduling commitments as handoff facts. If an agent or user
commits to a concrete ETA, deadline, duration, recurrence, check-back interval,
or expected completion window, record what was expected to happen and the
stated timing. If later messages revise, invalidate, or cancel that estimate,
record the update too. Do not infer timestamps or convert vague timing into
false precision; keep relative timing relative when that is all the transcript
provides.

Existing summary:

{existing_summary}

Previously summarized transcript up to the cutoff:

{old_transcript}

New transcript after the cutoff:

{new_transcript}
"""

@dataclass
class MessageRecord:
    platform: str
    date: str
    sequence: int
    message_index: int
    timestamp: str
    role: str
    heading_metadata: str
    text: str
    source_line: int
    source_id: str | None = None
    transcript_scaffolding: bool = False


@dataclass
class SourceRange:
    platform: str
    date: str
    sequence: int
    first_message: int
    last_message: int


@dataclass
class NoteRecord:
    note: str
    source_ranges: list[SourceRange]
    first_timestamp: str
    last_timestamp: str
    input_hash: str
    summary_hash: str
    mode: str = "summary"
    models: list[str] = field(default_factory=list)
    summarizer: dict[str, str] | None = None


@dataclass(frozen=True)
class SplitDecision:
    index: int
    gap_seconds: float
    before_timestamp: str
    after_timestamp: str
    elapsed_hours: float
    reason: str


@dataclass(frozen=True)
class PlannedMessageChunk:
    messages: list[MessageRecord]
    split_after: SplitDecision | None = None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def discover_codex_sources(explicit_source: Path, repo_root: Path) -> list[Path]:
    """Find user-owned Codex sessions rooted in this repository.

    Subagent sessions are represented by their final answers in their parent
    transcript and must not also be summarized as independent conversations.
    The explicitly configured legacy source is retained even when its older
    metadata predates ``thread_source``.
    """
    sources = {explicit_source.expanduser().resolve()}
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.is_dir():
        return sorted(sources)
    expected_cwd = repo_root.resolve()
    for candidate in sessions_root.glob("**/*.jsonl"):
        try:
            first_line = candidate.open("r", encoding="utf-8").readline()
            row = json.loads(first_line)
        except (OSError, json.JSONDecodeError):
            continue
        payload = row.get("payload") if row.get("type") == "session_meta" else None
        if not isinstance(payload, dict) or payload.get("thread_source") != "user":
            continue
        cwd = payload.get("cwd")
        if not isinstance(cwd, str):
            continue
        try:
            same_repo = Path(cwd).expanduser().resolve() == expected_cwd
        except OSError:
            same_repo = False
        if same_repo:
            sources.add(candidate.resolve())
    return sorted(sources)


def load_segments(
    claude_jsonl: Path,
    codex_jsonl: Path,
    *,
    include_subagent_finals: bool = True,
    discover_codex_sessions: bool = False,
    repo_root: Path | None = None,
) -> dict[tuple[str, str, int], list[MessageRecord]]:
    claude = load_module(script_dir() / "extract_claude.py", "vg_incremental_claude")
    codex = load_module(script_dir() / "extract_codex.py", "vg_incremental_codex")
    segments: dict[tuple[str, str, int], list[MessageRecord]] = {}

    codex_sources = [codex_jsonl]
    if discover_codex_sessions:
        codex_sources = discover_codex_sources(codex_jsonl, repo_root or Path.cwd())
    sources = [("claude-code", claude, claude_jsonl)] + [
        ("codex", codex, source) for source in codex_sources
    ]
    raw_segments: dict[str, list[tuple[datetime, ModuleType, list[Any]]]] = {}
    for platform, mod, source in sources:
        if not source.exists():
            continue
        if platform == "codex":
            _thread_id, _cwd, messages = mod.iter_messages(
                source,
                include_transcript_scaffolding=True,
                include_subagent_finals=include_subagent_finals,
            )
        else:
            messages = mod.iter_messages(
                source,
                include_transcript_scaffolding=True,
                include_subagent_finals=include_subagent_finals,
            )
        for segment in mod.split_segments(messages):
            start = next((m.ts for m in segment if m.ts), None)
            if start is not None:
                raw_segments.setdefault(platform, []).append((start, mod, segment))

    for platform, platform_segments in raw_segments.items():
        per_day: dict[str, int] = {}
        for start, mod, segment in sorted(platform_segments, key=lambda item: item[0]):
            date = start.date().isoformat()
            per_day[date] = per_day.get(date, 0) + 1
            sequence = per_day[date]
            key = (platform, date, sequence)
            records: list[MessageRecord] = []
            for idx, msg in enumerate(segment, 1):
                if msg.ts is None:
                    continue
                heading_metadata = mod.heading_metadata(msg) if hasattr(mod, "heading_metadata") else ""
                records.append(
                    MessageRecord(
                        platform=platform,
                        date=date,
                        sequence=sequence,
                        message_index=idx,
                        timestamp=msg.ts.isoformat().replace("+00:00", "Z"),
                        role=msg.role,
                        heading_metadata=heading_metadata,
                        text=msg.text,
                        source_line=msg.source_line,
                        source_id=getattr(msg, "source_id", None),
                        transcript_scaffolding=getattr(msg, "transcript_scaffolding", False),
                    )
                )
            segments[key] = records
    return segments


def segment_sort_key(key: tuple[str, str, int]) -> tuple[str, int, str]:
    platform, date, sequence = key
    return (date, sequence, SOURCE_ORDER.get(platform, platform))


def run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def git_creation_timestamp(path: Path, root: Path) -> datetime | None:
    path = (root / path).resolve() if not path.is_absolute() else path
    rel = path.relative_to(root).as_posix()
    try:
        output = run_git(["log", "--follow", "--diff-filter=A", "--format=%cI", "--", rel], root)
    except subprocess.CalledProcessError:
        return None
    values = [
        datetime.fromisoformat(line.replace("Z", "+00:00")).astimezone(timezone.utc)
        for line in output.splitlines()
        if line
    ]
    if values:
        return values[0]

    try:
        output = run_git(["log", "--follow", "--format=%cI", "--", rel], root)
    except subprocess.CalledProcessError:
        return None
    values = [
        datetime.fromisoformat(line.replace("Z", "+00:00")).astimezone(timezone.utc)
        for line in output.splitlines()
        if line
    ]
    if not values:
        return None
    return min(values)


def filesystem_timestamp(path: Path) -> datetime:
    stat = path.stat()
    if hasattr(stat, "st_birthtime"):
        return datetime.fromtimestamp(stat.st_birthtime, timezone.utc)
    return datetime.fromtimestamp(stat.st_mtime, timezone.utc)


def archive_timestamp(path: Path, root: Path) -> datetime:
    return (
        git_creation_timestamp(path, root)
        or timestamp_from_full_prefix(path.name)
        or filesystem_timestamp(path)
    )


def cached_archive_timestamp(path: Path, root: Path, cache: ArchiveTimestampCache | None = None) -> datetime:
    if cache is not None:
        return cache.timestamp_for(path).value
    return archive_timestamp(path, root)


def archive_note_files(notes_dir: Path) -> list[Path]:
    if not notes_dir.exists():
        return []
    return sorted(
        path
        for path in notes_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in ARCHIVE_SUFFIXES
        and path.name not in RESERVED_NOTE_NAMES
    )


def compact_prefix_for_new_note(
    notes_dir: Path,
    root: Path,
    timestamp: datetime,
    cache: ArchiveTimestampCache | None = None,
) -> str:
    timestamp = timestamp.astimezone(timezone.utc)
    by_day: dict[str, list[datetime]] = {}
    existing_paths = archive_note_files(notes_dir)
    if cache is not None:
        cache.prepare(existing_paths)
    for path in existing_paths:
        existing = cached_archive_timestamp(path, root, cache)
        by_day.setdefault(existing.astimezone(timezone.utc).strftime("%Y%m%d"), []).append(existing)
    by_day.setdefault(timestamp.strftime("%Y%m%d"), []).append(timestamp)

    next_day_index = 1
    index: int | None = None
    for day in sorted(by_day):
        entries = sorted(by_day[day])
        day_start = archive_day_start(next_day_index, len(entries))
        if day == timestamp.strftime("%Y%m%d"):
            offset = sum(existing <= timestamp for existing in entries) - 1
            index = day_start + offset
            break
        next_day_index = day_start + len(entries)
    if index is None:
        raise RuntimeError(f"Could not allocate archive prefix for {timestamp.isoformat()}")
    return compact_prefix(timestamp, index)


def note_name_for_messages(prefix: str, messages: list[MessageRecord]) -> str:
    title = conversation_title(title_participant_entries_for_messages(messages))
    return f"{prefix}-{title}.md"


def provisional_note_path_for_messages(
    notes_dir: Path,
    messages: list[MessageRecord],
    first_ts: datetime,
) -> Path:
    prefix = first_ts.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    return notes_dir / note_name_for_messages(prefix, messages)


def existing_note_path_for_messages(
    notes_dir: Path,
    root: Path,
    messages: list[MessageRecord],
    first_timestamp: datetime,
) -> Path:
    title = conversation_title(title_participant_entries_for_messages(messages))
    cache = ArchiveTimestampCache.load(root)
    paths = archive_note_files(notes_dir)
    cache.prepare(paths)
    candidates = [
        path
        for path in paths
        if path.name.endswith(f"-{title}.md")
        and cached_archive_timestamp(path, root, cache).date() == first_timestamp.astimezone(timezone.utc).date()
    ]
    if len(candidates) != 1:
        names = ", ".join(str(path) for path in candidates) or "none"
        raise RuntimeError(f"Expected exactly one existing note for {title}: {names}")
    return candidates[0]


def render_messages(messages: list[MessageRecord]) -> str:
    if not messages:
        return ""
    parts: list[str] = []
    current: tuple[str, str, int] | None = None
    for msg in messages:
        if msg.transcript_scaffolding:
            continue
        key = (msg.platform, msg.date, msg.sequence)
        if key != current:
            current = key
            parts.append(
                f"# Conversation Chunk: {msg.platform} | date {msg.date} | "
                f"daily sequence {msg.sequence:03d}\n"
            )
        parts.append(
            f"## Message {msg.message_index:03d} - {msg.role}{msg.heading_metadata}\n\n"
            f"{msg.text.strip()}\n"
        )
    return ("\n".join(parts).strip() + "\n") if parts else ""


def messages_for_range(
    segments: dict[tuple[str, str, int], list[MessageRecord]],
    source_range: SourceRange,
) -> list[MessageRecord]:
    records = segments[(source_range.platform, source_range.date, source_range.sequence)]
    return [
        msg
        for msg in records
        if source_range.first_message <= msg.message_index <= source_range.last_message
    ]


def all_messages_for_ranges(
    segments: dict[tuple[str, str, int], list[MessageRecord]],
    ranges: list[SourceRange],
) -> list[MessageRecord]:
    messages: list[MessageRecord] = []
    for source_range in ranges:
        messages.extend(messages_for_range(segments, source_range))
    return messages


def parse_message_timestamp(message: MessageRecord) -> datetime:
    return datetime.fromisoformat(message.timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)


def message_duration_hours(messages: list[MessageRecord]) -> float:
    if len(messages) < 2:
        return 0.0
    return (parse_message_timestamp(messages[-1]) - parse_message_timestamp(messages[0])).total_seconds() / 3600


def parse_heading_fields(heading_metadata: str) -> dict[str, str]:
    match = re.search(r"\[(?P<body>.*)\]\s*$", heading_metadata)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for part in match.group("body").split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            fields[key] = value
    return fields


def is_real_model_id(model: str) -> bool:
    return not (model.startswith("<") and model.endswith(">"))


def model_entries_for_messages(
    messages: list[MessageRecord],
    *,
    include_effort: bool = True,
) -> list[str]:
    stats: dict[str, tuple[int, int]] = {}
    for index, msg in enumerate(messages):
        if msg.transcript_scaffolding:
            continue
        if msg.role != "assistant":
            continue
        fields = parse_heading_fields(msg.heading_metadata)
        model = fields.get("model")
        if not model or not is_real_model_id(model):
            continue
        if include_effort and (effort := fields.get("effort")):
            model = f"{model}-{effort}"
        chars, first_index = stats.get(model, (0, index))
        stats[model] = (chars + len(msg.text), first_index)
    return [
        entry
        for entry, (_chars, _first_index) in sorted(
            stats.items(),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
        )
    ]


def participant_entries_for_messages(messages: list[MessageRecord]) -> list[str]:
    visible_messages = [msg for msg in messages if not msg.transcript_scaffolding]
    entries: list[str] = []
    if any(msg.role == "user" for msg in visible_messages):
        entries.append("User")
    entries.extend(model_entries_for_messages(visible_messages))
    if not entries:
        entries.append("No user or assistant model metadata found.")
    return entries


def title_participant_entries_for_messages(messages: list[MessageRecord]) -> list[str]:
    """Return the established filename roster: user plus model IDs, no effort."""
    entries: list[str] = []
    visible_messages = [msg for msg in messages if not msg.transcript_scaffolding]
    if any(msg.role == "user" for msg in visible_messages):
        entries.append("User")
    entries.extend(model_entries_for_messages(visible_messages, include_effort=False))
    if not entries:
        entries.append("No user or assistant model metadata found.")
    return entries


def model_entries_for_ranges(
    segments: dict[tuple[str, str, int], list[MessageRecord]],
    ranges: list[SourceRange],
) -> list[str]:
    return model_entries_for_messages(all_messages_for_ranges(segments, ranges))


def required_model_ids(model_entries: list[str]) -> list[str]:
    return model_entries


def format_english_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def render_participants_block(messages: list[MessageRecord]) -> str:
    return f"{PARTICIPANTS_PREFIX} {format_english_list(participant_entries_for_messages(messages))}."


def remove_existing_participants_blocks(summary: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", summary.strip())]
    kept: list[str] = []
    skip_next = False
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if skip_next:
            skip_next = False
            continue
        if paragraph.startswith(PARTICIPANTS_PREFIX):
            continue
        if paragraph in {OLD_PARTICIPANTS_SECTION_HEADING, OLD_MODEL_SECTION_HEADING}:
            skip_next = True
            continue
        if paragraph.startswith(OLD_PARTICIPANTS_SECTION_HEADING) or paragraph.startswith(OLD_MODEL_SECTION_HEADING):
            continue
        kept.append(paragraph)
    return "\n\n".join(kept).strip()


def insert_participants_block(summary: str, messages: list[MessageRecord]) -> str:
    summary = remove_existing_participants_blocks(summary)
    block = render_participants_block(messages)
    if "\n\n" not in summary:
        return f"{summary}\n\n{block}\n"
    opening, rest = summary.split("\n\n", 1)
    return f"{opening.strip()}\n\n{block}\n\n{rest.strip()}\n"


def conversation_source_ids(messages: list[MessageRecord]) -> list[str]:
    source_ids: list[str] = []
    for message in messages:
        if message.transcript_scaffolding or not message.source_id:
            continue
        if message.source_id not in source_ids:
            source_ids.append(message.source_id)
    return source_ids


def remove_conversation_sources_footer(summary: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(CONVERSATION_SOURCES_HEADING)}\s*$")
    matches = list(pattern.finditer(summary.rstrip()))
    if not matches:
        return summary.rstrip()
    return summary[: matches[-1].start()].rstrip()


def insert_conversation_sources_footer(
    summary: str,
    messages: list[MessageRecord],
) -> str:
    body = remove_conversation_sources_footer(summary)
    source_ids = conversation_source_ids(messages)
    if not source_ids:
        return body + "\n"
    lines = [CONVERSATION_SOURCES_HEADING, ""]
    lines.extend(f"- `{source_id}`" for source_id in source_ids)
    return body + "\n\n" + "\n".join(lines) + "\n"


def validate_model_mentions(summary: str, model_entries: list[str]) -> list[str]:
    return [model_id for model_id in required_model_ids(model_entries) if model_id not in summary]


def parse_shard_ranges(shard_text: str) -> list[SourceRange]:
    ranges: list[SourceRange] = []
    chunks = re.finditer(
        r"^# Conversation Chunk: (claude-code|codex) "
        r"\| date (\d{4}-\d{2}-\d{2}) "
        r"\| daily sequence (\d{3})\s*\n(?P<body>.*?)(?=^# Conversation Chunk: |\Z)",
        shard_text,
        re.M | re.S,
    )
    for chunk in chunks:
        messages = [int(m) for m in re.findall(r"^## Message (\d{3}) - ", chunk.group("body"), re.M)]
        if not messages:
            continue
        ranges.append(
            SourceRange(
                platform=chunk.group(1),
                date=chunk.group(2),
                sequence=int(chunk.group(3)),
                first_message=min(messages),
                last_message=max(messages),
            )
        )
    if not ranges:
        raise RuntimeError("No source ranges found in shard text")
    return ranges


def manifest_dict(records: list[NoteRecord]) -> dict[str, Any]:
    return {
        "version": 1,
        "description": "Coverage manifest for scripts/transcripts/update_conversation_notes.py.",
        "notes": [
            {
                **asdict(record),
                "source_ranges": [asdict(source_range) for source_range in record.source_ranges],
            }
            for record in records
        ],
    }


def load_manifest(path: Path) -> list[NoteRecord]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[NoteRecord] = []
    for row in data.get("notes", []):
        ranges = [SourceRange(**source_range) for source_range in row["source_ranges"]]
        records.append(
            NoteRecord(
                note=row["note"],
                source_ranges=ranges,
                first_timestamp=row["first_timestamp"],
                last_timestamp=row["last_timestamp"],
                input_hash=row["input_hash"],
                summary_hash=row["summary_hash"],
                mode=row.get("mode", "summary"),
                models=row.get("models", []),
                summarizer=row.get("summarizer"),
            )
        )
    return records


def write_manifest(path: Path, records: list[NoteRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_dict(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def context_from_summary(summary: str, max_chars: int) -> str:
    summary = summary.strip()
    if not summary:
        return ""
    first_para = summary.split("\n\n", 1)[0].strip()
    if first_para.startswith("*") and first_para.endswith("*") and len(first_para) <= max_chars:
        return first_para
    compact = " ".join(first_para.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def timestamps_for_ranges(
    segments: dict[tuple[str, str, int], list[MessageRecord]],
    ranges: list[SourceRange],
) -> tuple[str, str]:
    messages = all_messages_for_ranges(segments, ranges)
    if not messages:
        raise RuntimeError("No messages for ranges")
    return messages[0].timestamp, messages[-1].timestamp


def init_manifest(args: argparse.Namespace) -> None:
    segments = load_segments(
        args.claude_jsonl,
        args.codex_jsonl,
        include_subagent_finals=not args.no_subagent_finals,
    )
    records: list[NoteRecord] = []
    for shard_path in sorted(args.summary_shards_dir.glob("shard-*.md")):
        shard_text = shard_path.read_text(encoding="utf-8")
        ranges = parse_shard_ranges(shard_text)
        first_ts, last_ts = timestamps_for_ranges(segments, ranges)
        messages = all_messages_for_ranges(segments, ranges)
        first_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        note_path = existing_note_path_for_messages(args.notes_dir, args.repo_root, messages, first_dt)
        if not note_path.exists():
            raise RuntimeError(f"Expected note not found for {shard_path.name}: {note_path}")
        summary_text = note_path.read_text(encoding="utf-8")
        models = model_entries_for_messages(messages)
        records.append(
            NoteRecord(
                note=str(note_path),
                source_ranges=ranges,
                first_timestamp=first_ts,
                last_timestamp=last_ts,
                input_hash=sha256_text(shard_text),
                summary_hash=sha256_text(summary_text),
                models=models,
            )
        )
    write_manifest(args.manifest, records)
    print(f"wrote {args.manifest} with {len(records)} records")


def covered_segments(records: list[NoteRecord]) -> dict[tuple[str, str, int], tuple[int, int]]:
    covered: dict[tuple[str, str, int], tuple[int, int]] = {}
    for record_idx, record in enumerate(records):
        for source_range in record.source_ranges:
            key = (source_range.platform, source_range.date, source_range.sequence)
            current = covered.get(key)
            if current is None or source_range.last_message > current[0]:
                covered[key] = (source_range.last_message, record_idx)
    return covered


def source_ranges_for_messages(messages: list[MessageRecord]) -> list[SourceRange]:
    ranges: list[SourceRange] = []
    current_key: tuple[str, str, int] | None = None
    first_message: int | None = None
    last_message: int | None = None
    for message in messages:
        key = (message.platform, message.date, message.sequence)
        if current_key != key:
            if current_key is not None and first_message is not None and last_message is not None:
                ranges.append(SourceRange(*current_key, first_message, last_message))
            current_key = key
            first_message = message.message_index
        last_message = message.message_index
    if current_key is not None and first_message is not None and last_message is not None:
        ranges.append(SourceRange(*current_key, first_message, last_message))
    return ranges


def choose_duration_split(
    messages: list[MessageRecord],
    max_note_duration_hours: float,
    split_window_start_hours: float,
    split_window_end_hours: float,
) -> SplitDecision | None:
    if len(messages) < 2 or message_duration_hours(messages) <= max_note_duration_hours:
        return None
    start_ts = parse_message_timestamp(messages[0])
    preferred: list[SplitDecision] = []
    fallback: list[SplitDecision] = []
    for index in range(1, len(messages)):
        before_ts = parse_message_timestamp(messages[index - 1])
        after_ts = parse_message_timestamp(messages[index])
        elapsed_hours = (before_ts - start_ts).total_seconds() / 3600
        if elapsed_hours <= 0:
            continue
        gap_seconds = (after_ts - before_ts).total_seconds()
        decision = SplitDecision(
            index=index,
            gap_seconds=gap_seconds,
            before_timestamp=before_ts.isoformat().replace("+00:00", "Z"),
            after_timestamp=after_ts.isoformat().replace("+00:00", "Z"),
            elapsed_hours=elapsed_hours,
            reason="preferred-window",
        )
        if split_window_start_hours <= elapsed_hours <= split_window_end_hours:
            preferred.append(decision)
        elif elapsed_hours < max_note_duration_hours:
            fallback.append(
                SplitDecision(
                    index=index,
                    gap_seconds=gap_seconds,
                    before_timestamp=decision.before_timestamp,
                    after_timestamp=decision.after_timestamp,
                    elapsed_hours=elapsed_hours,
                    reason="fallback-before-max",
                )
            )
    candidates = preferred or fallback
    if not candidates:
        return SplitDecision(
            index=1,
            gap_seconds=0,
            before_timestamp=messages[0].timestamp,
            after_timestamp=messages[1].timestamp,
            elapsed_hours=0,
            reason="fallback-first-message",
        )
    return max(candidates, key=lambda decision: (decision.gap_seconds, decision.elapsed_hours))


def split_messages_by_duration(
    messages: list[MessageRecord],
    max_note_duration_hours: float | None,
    split_window_start_hours: float,
    split_window_end_hours: float,
) -> list[PlannedMessageChunk]:
    if (
        max_note_duration_hours is None
        or max_note_duration_hours <= 0
        or not messages
        or message_duration_hours(messages) <= max_note_duration_hours
    ):
        return [PlannedMessageChunk(messages)]
    chunks: list[PlannedMessageChunk] = []
    current = list(messages)
    while current and message_duration_hours(current) > max_note_duration_hours:
        decision = choose_duration_split(
            current,
            max_note_duration_hours,
            split_window_start_hours,
            split_window_end_hours,
        )
        if decision is None:
            break
        chunks.append(PlannedMessageChunk(current[: decision.index], decision))
        current = current[decision.index :]
    if current:
        chunks.append(PlannedMessageChunk(current))
    return chunks


def build_new_ranges(
    segments: dict[tuple[str, str, int], list[MessageRecord]],
    covered: dict[tuple[str, str, int], tuple[int, int]],
    target_chars: int,
    max_coalesce_gap_hours: float | None,
    max_note_duration_hours: float | None = None,
    split_window_start_hours: float = 4.0,
    split_window_end_hours: float = 5.0,
) -> list[list[SourceRange]]:
    shards: list[list[SourceRange]] = []
    for platform in sorted({key[0] for key in segments}, key=lambda value: SOURCE_ORDER.get(value, value)):
        current_messages: list[MessageRecord] = []
        current_size = 0
        current_last_ts: datetime | None = None
        keys = [key for key in segments if key[0] == platform]
        for key in sorted(keys, key=segment_sort_key):
            if key in covered:
                continue
            messages = segments[key]
            if not messages:
                continue
            rendered = render_messages(messages)
            first_ts = parse_message_timestamp(messages[0])
            last_ts = parse_message_timestamp(messages[-1])
            gap_too_large = (
                max_coalesce_gap_hours is not None
                and current_last_ts is not None
                and (first_ts - current_last_ts).total_seconds() > max_coalesce_gap_hours * 3600
            )
            if current_messages and (current_size + len(rendered) > target_chars or gap_too_large):
                for chunk in split_messages_by_duration(
                    current_messages,
                    max_note_duration_hours,
                    split_window_start_hours,
                    split_window_end_hours,
                ):
                    shards.append(source_ranges_for_messages(chunk.messages))
                current_messages = []
                current_size = 0
                current_last_ts = None
            current_messages.extend(messages)
            current_size += len(rendered)
            current_last_ts = last_ts
        if current_messages:
            for chunk in split_messages_by_duration(
                current_messages,
                max_note_duration_hours,
                split_window_start_hours,
                split_window_end_hours,
            ):
                shards.append(source_ranges_for_messages(chunk.messages))
    return shards


def render_ranges_for_plan(ranges: list[SourceRange]) -> str:
    return ", ".join(
        f"{source_range.platform} {source_range.date}#{source_range.sequence} "
        f"{source_range.first_message}-{source_range.last_message}"
        for source_range in ranges
    )


def print_message_chunk_plan(
    chunks: list[PlannedMessageChunk],
    indent: str = "  ",
) -> None:
    for index, chunk in enumerate(chunks, 1):
        first_ts, last_ts = chunk.messages[0].timestamp, chunk.messages[-1].timestamp
        ranges = source_ranges_for_messages(chunk.messages)
        print(
            f"{indent}{index}. {first_ts} -> {last_ts} "
            f"({message_duration_hours(chunk.messages):.2f}h, {len(chunk.messages)} messages)"
        )
        print(f"{indent}   ranges: {render_ranges_for_plan(ranges)}")
        if chunk.split_after:
            decision = chunk.split_after
            print(
                f"{indent}   split after: {decision.before_timestamp} -> {decision.after_timestamp}; "
                f"gap {decision.gap_seconds / 60:.1f} min; "
                f"elapsed {decision.elapsed_hours:.2f}h; {decision.reason}"
            )


def dry_run_update_notes(
    records: list[NoteRecord],
    segments: dict[tuple[str, str, int], list[MessageRecord]],
    args: argparse.Namespace,
) -> None:
    max_coalesce_gap_hours = (
        None if args.max_coalesce_gap_hours is not None and args.max_coalesce_gap_hours < 0 else args.max_coalesce_gap_hours
    )
    max_note_duration_hours = (
        None
        if args.max_note_duration_hours is not None and args.max_note_duration_hours < 0
        else args.max_note_duration_hours
    )
    covered = covered_segments(records)
    print("DRY RUN: no prompts, summaries, notes, manifest, or git commits will be written.")
    print(f"manifest records: {len(records)}")

    split_candidates = []
    if max_note_duration_hours is not None and max_note_duration_hours > 0:
        for record in records:
            messages = all_messages_for_ranges(segments, record.source_ranges)
            if not messages:
                continue
            chunks = split_messages_by_duration(
                messages,
                max_note_duration_hours,
                args.split_window_start_hours,
                args.split_window_end_hours,
            )
            if len(chunks) > 1:
                split_candidates.append((record, messages, chunks))

    print(f"existing notes exceeding duration policy: {len(split_candidates)}")
    for record, messages, chunks in split_candidates:
        print(f"- {record.note}")
        print(
            f"  current: {messages[0].timestamp} -> {messages[-1].timestamp} "
            f"({message_duration_hours(messages):.2f}h, {len(messages)} messages)"
        )
        print(f"  current ranges: {render_ranges_for_plan(record.source_ranges)}")
        print(f"  would become {len(chunks)} notes:")
        print_message_chunk_plan(chunks, indent="    ")

    continuations: list[tuple[Path, tuple[str, str, int], int, int, int, int]] = []
    for key, (last_message, record_idx) in covered.items():
        messages = segments.get(key)
        if not messages:
            continue
        current_last = messages[-1].message_index
        if current_last <= last_message:
            continue
        new_range = SourceRange(key[0], key[1], key[2], last_message + 1, current_last)
        new_count = current_last - last_message
        new_chars = len(render_messages(messages_for_range(segments, new_range)))
        if (
            not args.force_small_continuations
            and new_count < args.min_continuation_messages
            and new_chars < args.min_continuation_chars
        ):
            continue
        continuations.append((Path(records[record_idx].note), key, last_message + 1, current_last, new_count, new_chars))
    print(f"large continuations that would revise existing notes: {len(continuations)}")
    for note_path, key, first_message, last_message, new_count, new_chars in continuations:
        print(
            f"- {note_path}: {key} messages {first_message}-{last_message} "
            f"({new_count} messages, {new_chars} chars)"
        )

    new_shards = build_new_ranges(
        segments,
        covered,
        args.target_chars,
        max_coalesce_gap_hours,
        max_note_duration_hours,
        args.split_window_start_hours,
        args.split_window_end_hours,
    )
    print(f"new conversation notes that would be created: {len(new_shards)}")
    for index, ranges in enumerate(new_shards, 1):
        messages = all_messages_for_ranges(segments, ranges)
        print(
            f"- new {index}: {messages[0].timestamp} -> {messages[-1].timestamp} "
            f"({message_duration_hours(messages):.2f}h, {len(messages)} messages)"
        )
        print(f"  ranges: {render_ranges_for_plan(ranges)}")


@dataclass(frozen=True)
class ForbiddenMatch:
    pattern: str
    text: str


def forbidden_matches(text: str, forbidden_patterns: list[str]) -> list[ForbiddenMatch]:
    matches: list[ForbiddenMatch] = []
    for pattern in forbidden_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            snippet = match.group(0).strip()
            if snippet:
                matches.append(ForbiddenMatch(pattern=pattern, text=snippet))
    return matches


def validate_summary(text: str, forbidden_patterns: list[str]) -> list[str]:
    return sorted({match.pattern for match in forbidden_matches(text, forbidden_patterns)})


def display_forbidden_match(match: ForbiddenMatch) -> str:
    if match.pattern in DEFAULT_FORBID_REGEX:
        return f"[redacted token-like match for pattern: {match.pattern}]"
    return match.text


def merge_forbidden_matches(
    existing: list[ForbiddenMatch],
    new_matches: list[ForbiddenMatch],
) -> list[ForbiddenMatch]:
    seen = {(match.pattern, match.text.lower()) for match in existing}
    merged = list(existing)
    for match in new_matches:
        key = (match.pattern, match.text.lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(match)
    return merged


def retry_prompt_for_forbidden_matches(prompt: str, matches: list[ForbiddenMatch]) -> str:
    display_values: list[str] = []
    seen: set[str] = set()
    for match in matches:
        value = display_forbidden_match(match)
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        display_values.append(value)
    rendered = "\n".join(f"- {value}" for value in display_values)
    return FORBIDDEN_RETRY_PROMPT.format(matches=rendered, prompt=prompt)


def scrub_forbidden_lines(text: str, forbidden_patterns: list[str]) -> str:
    lines = text.splitlines()
    kept = [
        line
        for line in lines
        if not any(re.search(pattern, line, re.IGNORECASE) for pattern in forbidden_patterns)
    ]
    scrubbed = "\n".join(kept).strip() + "\n"
    for pattern in forbidden_patterns:
        scrubbed = re.sub(pattern, "", scrubbed, flags=re.IGNORECASE)
    return scrubbed.strip() + "\n"


def run_command(command: list[str], prompt: str) -> str:
    proc = subprocess.run(
        command,
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"summary command failed with exit {proc.returncode}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip() + "\n"


def attempt_path(path: Path, attempt: int) -> Path:
    if attempt == 1:
        return path
    return path.with_name(f"{path.stem}.attempt-{attempt}{path.suffix}")


def run_summary_command(
    command: list[str],
    prompt: str,
    candidate_path: Path,
    retry_prompt_path: Path,
    forbidden_patterns: list[str],
    max_forbid_attempts: int,
) -> str:
    all_matches: list[ForbiddenMatch] = []
    source_matches = forbidden_matches(prompt, forbidden_patterns)
    current_prompt = (
        preemptive_prompt_for_forbidden_referents(prompt, source_matches)
        if source_matches
        else prompt
    )
    attempts = max(1, max_forbid_attempts)
    for attempt in range(1, attempts + 1):
        candidate = run_command(command, current_prompt)
        write_prompt(attempt_path(candidate_path, attempt), candidate)
        matches = forbidden_matches(candidate, forbidden_patterns)
        if not matches:
            return candidate
        all_matches = merge_forbidden_matches(all_matches, matches)
        print(
            f"forbidden summary output in attempt {attempt}/{attempts}: "
            f"{len(matches)} match(es), {len(all_matches)} unique accumulated"
        )
        if attempt == attempts:
            scrubbed = scrub_forbidden_lines(candidate, forbidden_patterns)
            write_prompt(candidate_path, scrubbed)
            remaining = forbidden_matches(scrubbed, forbidden_patterns)
            if remaining:
                raise RuntimeError(
                    f"summary still matches forbidden filters after scrub: "
                    f"{sorted({match.pattern for match in remaining})}"
                )
            print(f"scrubbed forbidden lines after {attempts} attempt(s): {candidate_path}")
            return scrubbed
        current_prompt = retry_prompt_for_forbidden_matches(prompt, all_matches)
        write_prompt(attempt_path(retry_prompt_path, attempt + 1), current_prompt)

    raise AssertionError("unreachable summary retry state")


def write_prompt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_markdown(paths: list[Path], cwd: Path) -> None:
    if not paths:
        return
    deno = shutil.which("deno")
    if deno is None:
        print("deno not found; skipping markdown formatting")
        return
    subprocess.check_call([deno, "fmt", *[str(path) for path in paths]], cwd=cwd)


def first_timestamp_for_ranges(
    segments: dict[tuple[str, str, int], list[MessageRecord]],
    ranges: list[SourceRange],
) -> datetime:
    first_ts, _last_ts = timestamps_for_ranges(segments, ranges)
    return datetime.fromisoformat(first_ts.replace("Z", "+00:00"))


def previous_context_for_new_range(
    records: list[NoteRecord],
    ranges: list[SourceRange],
    max_chars: int,
) -> str:
    if max_chars <= 0:
        return ""
    platforms = {source_range.platform for source_range in ranges}
    if len(platforms) != 1:
        return ""
    platform = next(iter(platforms))
    candidates = [
        record
        for record in records
        if all(source_range.platform == platform for source_range in record.source_ranges)
    ]
    if not candidates:
        return ""
    previous = max(candidates, key=lambda record: record.last_timestamp)
    note_path = Path(previous.note)
    if not note_path.exists():
        return ""
    context = context_from_summary(note_path.read_text(encoding="utf-8"), max_chars)
    if not context:
        return ""
    return PREVIOUS_CONTEXT_TEMPLATE.format(platform=platform, context=context).strip()


def git_commit(paths: list[Path], message: str, cwd: Path, iso_date: str | None = None) -> None:
    rels = [
        str(((cwd / path).resolve() if not path.is_absolute() else path).relative_to(cwd.resolve()))
        for path in paths
    ]
    subprocess.check_call(["git", "add", "--", *rels], cwd=cwd)
    env = os.environ.copy()
    if iso_date:
        env["GIT_AUTHOR_DATE"] = iso_date
        env["GIT_COMMITTER_DATE"] = iso_date
    subprocess.check_call(["git", "commit", "-m", message, "--", *rels], cwd=cwd, env=env)


def sync_model_blocks(
    records: list[NoteRecord],
    segments: dict[tuple[str, str, int], list[MessageRecord]],
    args: argparse.Namespace,
) -> tuple[list[Path], bool]:
    changed_paths: list[Path] = []
    manifest_changed = False
    for record in records:
        note_path = Path(record.note)
        if not note_path.exists():
            continue
        model_entries = model_entries_for_ranges(segments, record.source_ranges)
        if record.models != model_entries:
            record.models = model_entries
            manifest_changed = True
        original = note_path.read_text(encoding="utf-8")
        messages = all_messages_for_ranges(segments, record.source_ranges)
        updated = insert_participants_block(original, messages)
        updated = insert_conversation_sources_footer(updated, messages)
        if updated != original:
            note_path.write_text(updated, encoding="utf-8")
            format_markdown([note_path], args.repo_root)
            updated = note_path.read_text(encoding="utf-8")
            changed_paths.append(note_path)
            print(f"updated participant block: {note_path}")
        missing = validate_model_mentions(updated, model_entries)
        if missing:
            raise RuntimeError(f"model roster missing required model ids in {note_path}: {missing}")
        summary_hash = sha256_text(updated)
        if record.summary_hash != summary_hash:
            record.summary_hash = summary_hash
            manifest_changed = True
    return changed_paths, manifest_changed


def rebuild_all_records(
    old_records: list[NoteRecord],
    segments: dict[tuple[str, str, int], list[MessageRecord]],
    args: argparse.Namespace,
) -> None:
    max_coalesce_gap_hours = (
        None
        if args.max_coalesce_gap_hours is not None and args.max_coalesce_gap_hours < 0
        else args.max_coalesce_gap_hours
    )
    max_note_duration_hours = (
        None
        if args.max_note_duration_hours is not None and args.max_note_duration_hours < 0
        else args.max_note_duration_hours
    )
    shards = build_new_ranges(
        segments,
        {},
        args.target_chars,
        max_coalesce_gap_hours,
        max_note_duration_hours,
        args.split_window_start_hours,
        args.split_window_end_hours,
    )
    if not shards:
        raise RuntimeError("full conversation rebuild found no transcript ranges")

    print(f"full conversation rebuild: old_notes={len(old_records)} new_notes={len(shards)}")
    root = args.repo_root.resolve()

    def absolute_repo_path(path: Path) -> Path:
        return (root / path).resolve() if not path.is_absolute() else path.resolve()

    old_path_candidates = {absolute_repo_path(Path(record.note)) for record in old_records}
    old_path_candidates.update(
        path.resolve()
        for path in args.notes_dir.glob("*.md")
        if re.match(r"^\d{10,14}-conversation-.*\.md$", path.name)
    )
    old_paths = sorted(path for path in old_path_candidates if path.exists())
    old_path_set = set(old_paths)
    previous_context_by_platform: dict[str, str] = {}
    new_records: list[NoteRecord] = []
    generated: list[tuple[Path, Path]] = []
    final_targets: set[Path] = set()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conversation-rebuild-", dir=args.work_dir) as tmp:
        tmp_dir = Path(tmp)
        for index, ranges in enumerate(shards, 1):
            messages = all_messages_for_ranges(segments, ranges)
            transcript = render_messages(messages)
            platforms = {source_range.platform for source_range in ranges}
            platform = next(iter(platforms)) if len(platforms) == 1 else None
            previous_context_block = ""
            if platform and args.rolling_context_chars > 0:
                previous_context = previous_context_by_platform.get(platform, "")
                if previous_context:
                    previous_context_block = PREVIOUS_CONTEXT_TEMPLATE.format(
                        platform=platform,
                        context=previous_context,
                    ).strip()
            prompt = SUMMARY_PROMPT.format(
                previous_context_block=previous_context_block,
                transcript=transcript,
            )
            first_ts = first_timestamp_for_ranges(segments, ranges)
            final_path = provisional_note_path_for_messages(args.notes_dir, messages, first_ts)
            final_absolute = absolute_repo_path(final_path)
            if final_absolute in final_targets or (
                final_absolute.exists() and final_absolute not in old_path_set
            ):
                raise RuntimeError(f"full rebuild target collision: {final_path}")
            final_targets.add(final_absolute)
            candidate_path = tmp_dir / final_path.name
            prompt_path = args.work_dir / "prompts" / f"rebuild-{final_path.stem}.md"
            retry_prompt_path = args.work_dir / "prompts" / f"retry-rebuild-{final_path.stem}.md"
            write_prompt(prompt_path, prompt)
            print(
                f"rebuild {index}/{len(shards)}: {final_path.name} "
                f"messages={len(messages)} prompt_chars={len(prompt)}"
            )
            if not args.command:
                continue
            candidate = run_summary_command(
                args.command,
                prompt,
                candidate_path,
                retry_prompt_path,
                args.forbid_regex,
                args.max_forbid_attempts,
            )
            model_entries = model_entries_for_messages(messages)
            note_text = insert_participants_block(candidate, messages)
            note_text = insert_conversation_sources_footer(note_text, messages)
            candidate_path.write_text(note_text, encoding="utf-8")
            format_markdown([candidate_path], args.repo_root)
            formatted_summary = candidate_path.read_text(encoding="utf-8")
            missing_models = validate_model_mentions(formatted_summary, model_entries)
            if missing_models:
                raise RuntimeError(
                    f"model roster missing required model ids in {final_path}: {missing_models}"
                )
            first, last = timestamps_for_ranges(segments, ranges)
            new_records.append(
                NoteRecord(
                    note=str(final_path),
                    source_ranges=ranges,
                    first_timestamp=first,
                    last_timestamp=last,
                    input_hash=sha256_text(transcript),
                    summary_hash=sha256_text(formatted_summary),
                    models=model_entries,
                    summarizer=args.summarizer_provenance,
                )
            )
            generated.append((candidate_path, final_path))
            if platform and args.rolling_context_chars > 0:
                previous_context_by_platform[platform] = context_from_summary(
                    formatted_summary,
                    args.rolling_context_chars,
                )

        if not args.command:
            print(f"No summary command provided; wrote {len(shards)} full-rebuild prompts only.")
            return

        backup_dir = tmp_dir / "previous-notes"
        backup_dir.mkdir()
        manifest_path = absolute_repo_path(args.manifest)
        manifest_before = manifest_path.read_bytes() if manifest_path.exists() else None
        backups: list[tuple[Path, Path]] = []
        moved_new: list[Path] = []
        final_paths = [absolute_repo_path(final) for _candidate, final in generated]
        changed_paths = [*old_paths, *final_paths, manifest_path]
        try:
            for old_path in old_paths:
                backup_path = backup_dir / old_path.name
                shutil.move(str(old_path), backup_path)
                backups.append((backup_path, old_path))
            for candidate_path, final_path in generated:
                final_absolute = absolute_repo_path(final_path)
                final_absolute.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(candidate_path), final_absolute)
                moved_new.append(final_absolute)
            write_manifest(manifest_path, new_records)

            provider = args.summarizer_provenance.get("provider", "custom")
            model = args.summarizer_provenance.get("model")
            label = f"{provider}/{model}" if model else provider
            git_commit(changed_paths, f"Rebuild conversation summaries with {label}", root)
        except Exception:
            for final_path in moved_new:
                if final_path.exists():
                    final_path.unlink()
            for backup_path, old_path in backups:
                old_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup_path), old_path)
            if manifest_before is None:
                manifest_path.unlink(missing_ok=True)
            else:
                manifest_path.write_bytes(manifest_before)
            rels = sorted(
                {
                    path.resolve().relative_to(root).as_posix()
                    for path in changed_paths
                    if path.resolve().is_relative_to(root)
                }
            )
            if rels:
                subprocess.run(
                    ["git", "add", "--all", "--", *rels],
                    cwd=root,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            raise


def update_notes(args: argparse.Namespace) -> None:
    records = load_manifest(args.manifest)
    segments = load_segments(
        args.claude_jsonl,
        args.codex_jsonl,
        include_subagent_finals=not args.no_subagent_finals,
        discover_codex_sessions=args.discover_codex_sessions,
        repo_root=args.repo_root,
    )
    if args.dry_run and args.resummarize_all:
        max_gap = None if args.max_coalesce_gap_hours < 0 else args.max_coalesce_gap_hours
        max_duration = None if args.max_note_duration_hours < 0 else args.max_note_duration_hours
        shards = build_new_ranges(
            segments,
            {},
            args.target_chars,
            max_gap,
            max_duration,
            args.split_window_start_hours,
            args.split_window_end_hours,
        )
        print(f"FULL REBUILD DRY RUN: old_notes={len(records)} replacement_notes={len(shards)}")
        return
    if args.dry_run:
        dry_run_update_notes(records, segments, args)
        return
    covered = covered_segments(records)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    timestamp_cache = ArchiveTimestampCache.load(args.repo_root)

    changed_paths: list[Path] = []
    manifest_changed = False
    if args.command and not args.no_model_block_sync and not args.resummarize_all:
        synced_paths, synced_manifest = sync_model_blocks(records, segments, args)
        changed_paths.extend(synced_paths)
        manifest_changed = manifest_changed or synced_manifest

    if args.resummarize_all:
        rebuild_all_records(records, segments, args)
        return

    wrote_prompt = False
    continuations: dict[int, list[tuple[tuple[str, str, int], int, int]]] = {}
    for key, (last_message, record_idx) in covered.items():
        messages = segments.get(key)
        if not messages:
            continue
        current_last = messages[-1].message_index
        if current_last > last_message:
            new_range = SourceRange(key[0], key[1], key[2], last_message + 1, current_last)
            new_count = current_last - last_message
            new_chars = len(render_messages(messages_for_range(segments, new_range)))
            if (
                not args.force_small_continuations
                and new_count < args.min_continuation_messages
                and new_chars < args.min_continuation_chars
            ):
                print(
                    "deferred small continuation: "
                    f"{key} {last_message + 1}-{current_last} "
                    f"({new_count} messages, {new_chars} chars)"
                )
                continue
            continuations.setdefault(record_idx, []).append((key, last_message, current_last))

    for record_idx, items in continuations.items():
        items = sorted(items, key=lambda item: segment_sort_key(item[0]))
        record = records[record_idx]
        note_path = Path(record.note)
        old_messages = all_messages_for_ranges(segments, record.source_ranges)
        new_messages: list[MessageRecord] = []
        for key, old_last, current_last in items:
            platform, date, sequence = key
            new_range = SourceRange(platform, date, sequence, old_last + 1, current_last)
            new_messages.extend(messages_for_range(segments, new_range))
        new_messages = sorted(new_messages, key=lambda msg: (msg.timestamp, SOURCE_ORDER.get(msg.platform, msg.platform)))
        prompt = REVISION_PROMPT.format(
            existing_summary=note_path.read_text(encoding="utf-8").strip(),
            old_transcript=render_messages(old_messages),
            new_transcript=render_messages(new_messages),
        )
        prompt_path = args.work_dir / "prompts" / f"revise-{note_path.stem}.md"
        write_prompt(prompt_path, prompt)
        wrote_prompt = True
        for key, old_last, current_last in items:
            print(f"continuation: {note_path} {key} {old_last + 1}-{current_last}")
        print(f"prompt: {prompt_path}")
        if not args.command:
            continue
        candidate_path = args.work_dir / "candidates" / note_path.name
        retry_prompt_path = args.work_dir / "prompts" / f"retry-revise-{note_path.stem}.md"
        candidate = run_summary_command(
            args.command,
            prompt,
            candidate_path,
            retry_prompt_path,
            args.forbid_regex,
            args.max_forbid_attempts,
        )
        for key, _old_last, current_last in items:
            for source_range in record.source_ranges:
                if (source_range.platform, source_range.date, source_range.sequence) == key:
                    source_range.last_message = current_last
                    break
        messages = all_messages_for_ranges(segments, record.source_ranges)
        model_entries = model_entries_for_messages(messages)
        note_text = insert_participants_block(candidate, messages)
        note_text = insert_conversation_sources_footer(note_text, messages)
        note_path.write_text(note_text, encoding="utf-8")
        format_markdown([note_path], args.repo_root)
        formatted_summary = note_path.read_text(encoding="utf-8")
        missing_models = validate_model_mentions(formatted_summary, model_entries)
        if missing_models:
            raise RuntimeError(f"model roster missing required model ids in {note_path}: {missing_models}")
        first_ts, last_ts = timestamps_for_ranges(segments, record.source_ranges)
        record.first_timestamp = first_ts
        record.last_timestamp = last_ts
        record.input_hash = sha256_text(render_messages(all_messages_for_ranges(segments, record.source_ranges)))
        record.summary_hash = sha256_text(formatted_summary)
        record.models = model_entries
        record.summarizer = args.summarizer_provenance
        manifest_changed = True
        changed_paths.append(note_path)

    max_coalesce_gap_hours = (
        None if args.max_coalesce_gap_hours is not None and args.max_coalesce_gap_hours < 0 else args.max_coalesce_gap_hours
    )
    max_note_duration_hours = (
        None
        if args.max_note_duration_hours is not None and args.max_note_duration_hours < 0
        else args.max_note_duration_hours
    )
    new_shards = build_new_ranges(
        segments,
        covered,
        args.target_chars,
        max_coalesce_gap_hours,
        max_note_duration_hours,
        args.split_window_start_hours,
        args.split_window_end_hours,
    )
    for shard_idx, ranges in enumerate(new_shards, 1):
        messages = all_messages_for_ranges(segments, ranges)
        transcript = render_messages(messages)
        previous_context_block = previous_context_for_new_range(records, ranges, args.rolling_context_chars)
        prompt = SUMMARY_PROMPT.format(
            previous_context_block=previous_context_block,
            transcript=transcript,
        )
        first_ts = first_timestamp_for_ranges(segments, ranges)
        prefix = compact_prefix_for_new_note(args.notes_dir, args.repo_root, first_ts, timestamp_cache)
        note_path = args.notes_dir / note_name_for_messages(prefix, messages)
        if note_path.exists():
            provisional = provisional_note_path_for_messages(args.notes_dir, messages, first_ts)
            if provisional.exists():
                raise RuntimeError(
                    f"Refusing to overwrite existing note or provisional note: {note_path}, {provisional}"
                )
            print(f"archive name occupied; writing provisional note for later normalization: {provisional}")
            note_path = provisional
        prompt_path = args.work_dir / "prompts" / f"new-{note_path.name}"
        write_prompt(prompt_path, prompt)
        wrote_prompt = True
        print(f"new conversation note: {note_path} ({len(messages)} messages)")
        print(f"prompt: {prompt_path}")
        if not args.command:
            continue
        candidate_path = args.work_dir / "candidates" / note_path.name
        retry_prompt_path = args.work_dir / "prompts" / f"retry-new-{note_path.stem}.md"
        candidate = run_summary_command(
            args.command,
            prompt,
            candidate_path,
            retry_prompt_path,
            args.forbid_regex,
            args.max_forbid_attempts,
        )
        messages = all_messages_for_ranges(segments, ranges)
        model_entries = model_entries_for_messages(messages)
        note_text = insert_participants_block(candidate, messages)
        note_text = insert_conversation_sources_footer(note_text, messages)
        note_path.write_text(note_text, encoding="utf-8")
        format_markdown([note_path], args.repo_root)
        formatted_summary = note_path.read_text(encoding="utf-8")
        missing_models = validate_model_mentions(formatted_summary, model_entries)
        if missing_models:
            raise RuntimeError(f"model roster missing required model ids in {note_path}: {missing_models}")
        first, last = timestamps_for_ranges(segments, ranges)
        records.append(
            NoteRecord(
                note=str(note_path),
                source_ranges=ranges,
                first_timestamp=first,
                last_timestamp=last,
                input_hash=sha256_text(transcript),
                summary_hash=sha256_text(formatted_summary),
                models=model_entries,
                summarizer=args.summarizer_provenance,
            )
        )
        manifest_changed = True
        iso = first_ts.isoformat().replace("+00:00", "Z")
        git_commit([note_path], f"Archive conversation note {note_path.stem}", args.repo_root, iso)
        timestamp_cache.record_path(note_path, TimestampInfo(first_ts, "conversation note first message"))

    if args.command and (changed_paths or manifest_changed):
        write_manifest(args.manifest, records)
        if changed_paths:
            git_commit(changed_paths, "Update conversation summary notes", args.repo_root)
        if manifest_changed:
            git_commit([args.manifest], "Update conversation summary manifest", args.repo_root)
    elif not args.command:
        if wrote_prompt:
            print("No summary command provided; wrote prompts only.")
        else:
            print("No new or large enough transcript ranges found.")
    else:
        print("No new or continued transcript ranges found.")


def use_default_update_command(args: argparse.Namespace) -> None:
    if args.no_command or getattr(args, "dry_run", False):
        args.command = None
        return
    if args.command is not None:
        args.summarizer_provenance = custom_command_provenance(args.command)
        return
    spec = resolve_spec(args.summary_provider, args.summary_model, args.summary_reasoning)
    command = wrapper_argv(spec)
    if shutil.which(command[0]) is None:
        raise RuntimeError(
            f"Default summarizer command not found: {command[0]!r}. "
            "Install the selected provider CLI, pass --command explicitly, or use --no-command "
            "to write prompts without generating summaries."
        )
    args.command = command
    args.summarizer_provenance = spec.provenance()


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv.append("update")
    elif sys.argv[1].startswith("-") and sys.argv[1] not in {"-h", "--help"}:
        sys.argv.insert(1, "update")

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command_name", required=True)

    init_parser = sub.add_parser("init-manifest")
    init_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    init_parser.add_argument("--summary-shards-dir", type=Path, default=DEFAULT_SUMMARY_SHARDS_DIR)
    init_parser.add_argument("--notes-dir", type=Path, default=DEFAULT_NOTES_DIR)
    init_parser.add_argument("--claude-jsonl", type=Path, default=DEFAULT_CLAUDE_JSONL)
    init_parser.add_argument("--codex-jsonl", type=Path, default=DEFAULT_CODEX_JSONL)
    init_parser.add_argument("--no-subagent-finals", action="store_true")
    init_parser.set_defaults(func=init_manifest)

    update_parser = sub.add_parser("update")
    update_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    update_parser.add_argument("--notes-dir", type=Path, default=DEFAULT_NOTES_DIR)
    update_parser.add_argument("--claude-jsonl", type=Path, default=DEFAULT_CLAUDE_JSONL)
    update_parser.add_argument("--codex-jsonl", type=Path, default=DEFAULT_CODEX_JSONL)
    update_parser.add_argument(
        "--discover-codex-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include every user-owned Codex session whose working directory is this repository "
            "(default: enabled); subagent sessions remain excluded as independent sources."
        ),
    )
    update_parser.add_argument(
        "--no-subagent-finals",
        action="store_true",
        help="Exclude final subagent assessments from summarizer-facing transcripts.",
    )
    update_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    update_parser.add_argument("--target-chars", type=int, default=180_000)
    update_parser.add_argument(
        "--max-coalesce-gap-hours",
        type=float,
        default=2.0,
        help=(
            "Start a new conversation note when adjacent raw transcript segments "
            "are separated by more than this many hours; use a negative value to "
            "disable this boundary."
        ),
    )
    update_parser.add_argument(
        "--max-note-duration-hours",
        type=float,
        default=6.0,
        help=(
            "Prefer splitting generated conversation notes longer than this many hours; "
            "use a negative value to disable duration splitting."
        ),
    )
    update_parser.add_argument(
        "--split-window-start-hours",
        type=float,
        default=4.0,
        help="When duration splitting, prefer split points at least this many hours after the chunk start.",
    )
    update_parser.add_argument(
        "--split-window-end-hours",
        type=float,
        default=5.0,
        help="When duration splitting, prefer split points at most this many hours after the chunk start.",
    )
    update_parser.add_argument(
        "--min-continuation-messages",
        type=int,
        default=20,
        help="Do not revise an existing note for fewer new messages unless the continuation is large by chars.",
    )
    update_parser.add_argument(
        "--min-continuation-chars",
        type=int,
        default=8000,
        help="Do not revise an existing note for fewer new chars unless the continuation is large by messages.",
    )
    update_parser.add_argument(
        "--force-small-continuations",
        action="store_true",
        help="Revise existing notes even for tiny live-tail continuations.",
    )
    update_parser.add_argument(
        "--resummarize-all",
        action="store_true",
        help="Rebuild every conversation note and the coverage manifest from raw transcripts.",
    )
    update_parser.add_argument(
        "--no-model-block-sync",
        action="store_true",
        help="Skip deterministic model-roster block synchronization.",
    )
    update_parser.add_argument(
        "--rolling-context-chars",
        type=int,
        default=700,
        help="Carry this many chars from the previous same-source summary into a new summary prompt; 0 disables.",
    )
    update_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    update_parser.add_argument(
        "--no-command",
        action="store_true",
        help="Write prompts only instead of running the selected summarizer.",
    )
    update_parser.add_argument(
        "--summary-provider",
        choices=PROVIDERS,
        default=DEFAULT_PROVIDER,
        help="Summary CLI provider; default: codex.",
    )
    update_parser.add_argument(
        "--summary-model",
        help="Provider model id/alias; defaults to gpt-5.6-luna for Codex or sonnet for Claude.",
    )
    update_parser.add_argument(
        "--summary-reasoning",
        default=DEFAULT_CODEX_REASONING,
        help="Codex reasoning effort; ignored by Claude. Default: medium.",
    )
    update_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the update and duration-split plan without writing prompts, notes, manifests, or commits.",
    )
    update_parser.add_argument(
        "--forbid-regex",
        action="append",
        default=[],
        help=(
            "Additional regex identifying a referent that must not be identifiable in generated summaries; "
            "may be repeated."
        ),
    )
    update_parser.add_argument(
        "--dotenv",
        action="append",
        type=Path,
        help="dotenv file supplying notes forbid-regex env vars; defaults to .env and notes/.env.",
    )
    update_parser.add_argument(
        "--no-default-forbid-regex",
        action="store_true",
        help="Disable the built-in credential-shaped output filters.",
    )
    update_parser.add_argument(
        "--max-forbid-attempts",
        type=int,
        default=5,
        help="Retry a summary this many times when output matches --forbid-regex before scrubbing matching lines.",
    )
    update_parser.add_argument("--command", nargs=argparse.REMAINDER)
    update_parser.set_defaults(func=update_notes)

    args = parser.parse_args()
    if args.command_name == "update":
        args.forbid_regex = resolve_forbid_patterns(
            args.forbid_regex,
            args.repo_root,
            include_defaults=not args.no_default_forbid_regex,
            dotenv_paths=args.dotenv,
        )
        use_default_update_command(args)
    args.func(args)


if __name__ == "__main__":
    main()
