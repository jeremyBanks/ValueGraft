#!/usr/bin/env python3
"""Split a combined shard summary into dated conversation notes.

The filename prefix for each shard is derived from the first raw message covered
by the corresponding summary shard. With --commit, each created note is committed
with matching author/committer dates so later git-history-based normalizers keep
the files in the intended chronological position.

Conversation summary notes are source-specific by default:
`YYYYMMDDHHMMSS-claude-conversation.md` or
`YYYYMMDDHHMMSS-codex-conversation.md`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType


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


def platform_slug(platform: str) -> str:
    if platform == "claude-code":
        return "claude"
    if platform == "codex":
        return "codex"
    return "conversation"


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
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--delete-combined", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    timestamps = build_message_timestamp_index(script_dir, args.claude_jsonl, args.codex_jsonl)
    args.notes_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    combined_text = args.combined.read_text(encoding="utf-8")
    for shard_idx, body in split_combined_summary(combined_text):
        platform, ts = shard_start_metadata(shard_idx, args.summary_shards_dir, timestamps)
        prefix = ts.strftime("%Y%m%d%H%M%S")
        path = args.notes_dir / f"{prefix}-{platform_slug(platform)}-conversation.md"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing file: {path}")
        path.write_text(body, encoding="utf-8")
        format_markdown([path], args.repo_root)
        created.append(path)
        print(f"wrote shard {shard_idx:03d}: {path}")
        if args.commit:
            iso = ts.isoformat().replace("+00:00", "Z")
            git_commit(path, f"Archive conversation shard {prefix}", iso, args.repo_root)

    if args.delete_combined:
        args.combined.unlink()

    print(f"created {len(created)} note files")


if __name__ == "__main__":
    main()
