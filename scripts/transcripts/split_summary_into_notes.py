#!/usr/bin/env python3
"""Split a combined segment summary into dated conversation notes.

Each created note is committed with author/committer dates matching the first
raw message it covers, so git-history-based normalizers keep the files in the
intended chronological position.

Conversation summary notes are source-specific by default:
`YYYYMMDDNN-conversation-<participants>.md`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notes_archive_naming import archive_day_start, compact_prefix, conversation_title, timestamp_from_full_prefix  # noqa: E402


PARTICIPANTS_PREFIX = "**Participants:**"
OLD_PARTICIPANTS_SECTION_HEADING = "**Participants in this Conversation.**"
OLD_MODEL_SECTION_HEADING = "**Models in this Conversation.**"
RESERVED_NOTE_NAMES = {"AGENTS.md", "README.md"}
ARCHIVE_SUFFIXES = {".md", ".txt"}


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_message_timestamp_index(
    script_dir: Path,
    claude_jsonl: Path,
    codex_jsonl: Path,
) -> dict[tuple[str, str, int, int], datetime]:
    claude = load_module(script_dir / "extract_claude.py", "vg_extract_claude_for_split")
    codex = load_module(script_dir / "extract_codex.py", "vg_extract_codex_for_split")
    index: dict[tuple[str, str, int, int], datetime] = {}

    sources = [
        ("claude-code", claude, claude_jsonl),
        ("codex", codex, codex_jsonl),
    ]
    for platform, mod, source in sources:
        if platform == "codex":
            _thread_id, _cwd, messages = mod.iter_messages(source)
        else:
            messages = mod.iter_messages(source)
        segments = mod.split_segments(messages)
        per_day: dict[str, int] = {}
        for segment in segments:
            start = next((m.ts for m in segment if m.ts), None)
            if start is None:
                continue
            day = start.date().isoformat()
            per_day[day] = per_day.get(day, 0) + 1
            sequence = per_day[day]
            for message_idx, message in enumerate(segment, 1):
                if message.ts is not None:
                    index[(platform, day, sequence, message_idx)] = message.ts
    return index


def shard_start_metadata(
    shard_idx: int,
    summary_shards_dir: Path,
    timestamps: dict[tuple[str, str, int, int], datetime],
) -> tuple[str, datetime]:
    shard_path = summary_shards_dir / f"shard-{shard_idx:03d}.md"
    text = shard_path.read_text(encoding="utf-8", errors="replace")
    chunk = re.search(
        r"^# Conversation Chunk: (claude-code|codex) "
        r"\| date (\d{4}-\d{2}-\d{2}) "
        r"\| daily sequence (\d{3})\s*\n(?P<rest>.*?)(?=^# Conversation Chunk: |\Z)",
        text,
        re.M | re.S,
    )
    if not chunk:
        raise RuntimeError(f"No source chunk found for shard {shard_idx}")
    msg = re.search(r"^## Message (\d{3}) - ", chunk.group("rest"), re.M)
    if not msg:
        raise RuntimeError(f"No first message found for shard {shard_idx}")

    platform = chunk.group(1)
    key = (platform, chunk.group(2), int(chunk.group(3)), int(msg.group(1)))
    ts = timestamps.get(key)
    if ts is None:
        raise RuntimeError(f"No raw timestamp found for {key}")
    return platform, ts


def split_combined_summary(text: str) -> list[tuple[int, str]]:
    parts = re.split(r"(?m)^## Shard (\d+)\.\s*$", text)
    if len(parts) < 3:
        raise RuntimeError("Combined summary does not contain '## Shard N.' headings")
    out: list[tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        out.append((int(parts[i]), parts[i + 1].strip() + "\n"))
    return out


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


def shard_messages(text: str) -> list[tuple[str, str, str]]:
    messages: list[tuple[str, str, str]] = []
    for match in re.finditer(
        r"^## Message \d+ - (?P<role>user|assistant)(?P<meta>.*?)\n\n(?P<body>.*?)(?=^## Message \d+ - |\Z)",
        text,
        re.M | re.S,
    ):
        messages.append((match.group("role"), match.group("meta"), match.group("body").strip()))
    return messages


def model_entries_from_shard_text(text: str) -> list[str]:
    stats: dict[str, tuple[int, int]] = {}
    for index, (role, metadata, body) in enumerate(shard_messages(text)):
        if role != "assistant":
            continue
        fields = parse_heading_fields(metadata)
        model = fields.get("model")
        if not model or not is_real_model_id(model):
            continue
        if effort := fields.get("effort"):
            model = f"{model}-{effort}"
        chars, first_index = stats.get(model, (0, index))
        stats[model] = (chars + len(body), first_index)
    return [
        entry
        for entry, (_chars, _first_index) in sorted(
            stats.items(),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
        )
    ]


def participant_entries_from_shard_text(text: str) -> list[str]:
    entries: list[str] = []
    if any(role == "user" for role, _metadata, _body in shard_messages(text)):
        entries.append("User")
    entries.extend(model_entries_from_shard_text(text))
    if not entries:
        entries.append("No user or assistant model metadata found.")
    return entries


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
        return max(values)

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


def planned_conversation_paths(
    notes_dir: Path,
    repo_root: Path,
    items: list[tuple[int, datetime, str]],
) -> dict[int, Path]:
    existing = [
        ("existing", -1, archive_timestamp(path, repo_root), path.stem)
        for path in archive_note_files(notes_dir)
    ]
    planned = [("new", shard_idx, ts.astimezone(timezone.utc), title) for shard_idx, ts, title in items]
    by_day: dict[str, list[tuple[str, int, datetime, str]]] = {}
    for item in [*existing, *planned]:
        _kind, _idx, ts, _title = item
        by_day.setdefault(ts.strftime("%Y%m%d"), []).append(item)

    out: dict[int, Path] = {}
    next_day_index = 1
    for day in sorted(by_day):
        entries = sorted(by_day[day], key=lambda item: (item[2], item[0], item[3], item[1]))
        day_start = archive_day_start(next_day_index, len(entries))
        for offset, (kind, shard_idx, ts, title) in enumerate(entries):
            if kind != "new":
                continue
            day_index = day_start + offset
            path = notes_dir / f"{compact_prefix(ts, day_index)}-{title}.md"
            if path.exists():
                raise RuntimeError(f"Refusing to overwrite existing file: {path}")
            out[shard_idx] = path
        next_day_index = day_start + len(entries)
    return out


def format_english_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def render_participants_block(shard_text: str) -> str:
    return f"{PARTICIPANTS_PREFIX} {format_english_list(participant_entries_from_shard_text(shard_text))}."


def insert_participants_block(summary: str, shard_text: str) -> str:
    summary = summary.strip()
    for heading in (OLD_PARTICIPANTS_SECTION_HEADING, OLD_MODEL_SECTION_HEADING):
        summary = re.sub(
            rf"\n\n{re.escape(heading)}\n.*?(?=\n\n(?:#{{1,6}}\s|\*\*)|\Z)",
            "",
            summary,
            flags=re.S,
        )
    summary = re.sub(
        rf"\n\n{re.escape(PARTICIPANTS_PREFIX)}\s+.*?(?=\n\n(?:#{{1,6}}\s|\*\*)|\Z)",
        "",
        summary,
        flags=re.S,
    )
    block = render_participants_block(shard_text)
    if "\n\n" not in summary:
        return f"{summary}\n\n{block}\n"
    opening, rest = summary.split("\n\n", 1)
    return f"{opening.strip()}\n\n{block}\n\n{rest.strip()}\n"


def validate_model_mentions(summary: str, model_entries: list[str]) -> list[str]:
    return [model_id for model_id in model_entries if model_id not in summary]


def git_commit(path: Path, message: str, iso_date: str, cwd: Path) -> None:
    rel = str(path.relative_to(cwd))
    subprocess.check_call(["git", "add", "--", rel], cwd=cwd)
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = iso_date
    env["GIT_COMMITTER_DATE"] = iso_date
    subprocess.check_call(["git", "commit", "-m", message, "--", rel], cwd=cwd, env=env)


def format_markdown(paths: list[Path], cwd: Path) -> None:
    if not paths:
        return
    deno = shutil.which("deno")
    if deno is None:
        print("deno not found; skipping markdown formatting")
        return
    subprocess.check_call([deno, "fmt", *[str(path) for path in paths]], cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--summary-shards-dir", type=Path, required=True)
    parser.add_argument("--claude-jsonl", type=Path, required=True)
    parser.add_argument("--codex-jsonl", type=Path, required=True)
    parser.add_argument("--notes-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--delete-combined", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    timestamps = build_message_timestamp_index(script_dir, args.claude_jsonl, args.codex_jsonl)
    args.notes_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    combined_text = args.combined.read_text(encoding="utf-8")
    summaries = split_combined_summary(combined_text)
    planned_items: list[tuple[int, datetime, str]] = []
    shard_inputs: dict[int, tuple[str, datetime, str, list[str], str]] = {}
    for shard_idx, body in summaries:
        _platform, ts = shard_start_metadata(shard_idx, args.summary_shards_dir, timestamps)
        shard_text = (args.summary_shards_dir / f"shard-{shard_idx:03d}.md").read_text(encoding="utf-8")
        model_entries = model_entries_from_shard_text(shard_text)
        title = conversation_title(participant_entries_from_shard_text(shard_text))
        planned_items.append((shard_idx, ts, title))
        shard_inputs[shard_idx] = (body, ts, shard_text, model_entries, title)

    planned_paths = planned_conversation_paths(args.notes_dir, args.repo_root, planned_items)
    for shard_idx, _body in summaries:
        body, ts, shard_text, model_entries, _title = shard_inputs[shard_idx]
        path = planned_paths[shard_idx]
        body = insert_participants_block(body, shard_text)
        path.write_text(body, encoding="utf-8")
        format_markdown([path], args.repo_root)
        missing_models = validate_model_mentions(path.read_text(encoding="utf-8"), model_entries)
        if missing_models:
            raise RuntimeError(f"model roster missing required model ids in {path}: {missing_models}")
        created.append(path)
        print(f"wrote conversation note {shard_idx:03d}: {path}")
        iso = ts.isoformat().replace("+00:00", "Z")
        git_commit(path, f"Archive conversation note {path.stem}", iso, args.repo_root)

    if args.delete_combined:
        args.combined.unlink()

    print(f"created {len(created)} note files")


if __name__ == "__main__":
    main()
