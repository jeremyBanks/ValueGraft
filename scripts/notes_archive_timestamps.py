#!/usr/bin/env python3
"""Timestamp cache for notes archive filename planning."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notes_archive_naming import timestamp_from_full_prefix


DEFAULT_TIMESTAMP_CACHE = Path("scripts/notes-archive-timestamps.json")
CACHE_VERSION = 1


@dataclass(frozen=True)
class TimestampInfo:
    value: datetime
    source: str


def run_git(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def parse_git_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def git_creation_timestamp(path: Path, root: Path) -> TimestampInfo | None:
    rel = path.relative_to(root).as_posix()
    try:
        output = run_git(["log", "--follow", "--diff-filter=A", "--format=%cI", "--", rel], root)
    except subprocess.CalledProcessError:
        return None
    lines = [line for line in output.splitlines() if line]
    if lines:
        return TimestampInfo(parse_git_timestamp(lines[0]), "git current-file lifetime")

    try:
        output = run_git(["log", "--follow", "--format=%cI", "--", rel], root)
    except subprocess.CalledProcessError:
        return None
    lines = [line for line in output.splitlines() if line]
    if not lines:
        return None
    timestamps = [parse_git_timestamp(line) for line in lines]
    return TimestampInfo(min(timestamps), "git history")


def filesystem_timestamp(path: Path) -> TimestampInfo:
    stat = path.stat()
    if hasattr(stat, "st_birthtime"):
        return TimestampInfo(datetime.fromtimestamp(stat.st_birthtime, timezone.utc), "filesystem birth time")
    return TimestampInfo(datetime.fromtimestamp(stat.st_mtime, timezone.utc), "filesystem mtime")


def resolve_cache_path(root: Path, cache_path: Path) -> Path:
    return (root / cache_path).resolve() if not cache_path.is_absolute() else cache_path.resolve()


class ArchiveTimestampCache:
    def __init__(self, root: Path, cache_path: Path = DEFAULT_TIMESTAMP_CACHE, entries: dict[str, Any] | None = None):
        self.root = root.resolve()
        self.cache_path = resolve_cache_path(self.root, cache_path)
        self.entries: dict[str, dict[str, str]] = {
            path: entry
            for path, entry in (entries or {}).items()
            if isinstance(path, str) and isinstance(entry, dict)
        }
        self._object_ids: dict[str, str | None] = {}
        self._history: dict[str, TimestampInfo | None] = {}
        self.dirty = False

    @classmethod
    def load(cls, root: Path, cache_path: Path = DEFAULT_TIMESTAMP_CACHE) -> "ArchiveTimestampCache":
        resolved = resolve_cache_path(root.resolve(), cache_path)
        if not resolved.exists():
            return cls(root, cache_path)
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(root, cache_path)
        if data.get("version") != CACHE_VERSION:
            return cls(root, cache_path)
        entries = data.get("entries")
        return cls(root, cache_path, entries if isinstance(entries, dict) else {})

    def relative_path(self, path: Path) -> str:
        resolved = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        return resolved.relative_to(self.root).as_posix()

    def prepare(self, paths: list[Path]) -> None:
        missing = [self.relative_path(path) for path in paths if self.relative_path(path) not in self._object_ids]
        if not missing:
            return
        for rel in missing:
            self._object_ids[rel] = None
        try:
            output = run_git(["ls-files", "-s", "--", *missing], self.root)
        except subprocess.CalledProcessError:
            return
        for line in output.splitlines():
            parts = line.split(maxsplit=3)
            if len(parts) != 4:
                continue
            _mode, object_id, _stage, rel = parts
            self._object_ids[rel] = object_id

    def object_id_for_path(self, path: Path) -> str | None:
        rel = self.relative_path(path)
        if rel not in self._object_ids:
            self.prepare([path])
        return self._object_ids.get(rel)

    def cached_timestamp(self, path: Path, object_id: str | None) -> TimestampInfo | None:
        if not object_id:
            return None
        rel = self.relative_path(path)
        entry = self.entries.get(rel)
        if not entry or entry.get("object_id") != object_id:
            return None
        timestamp = entry.get("archive_timestamp_utc")
        source = entry.get("timestamp_source")
        if not timestamp or not source:
            return None
        try:
            return TimestampInfo(parse_git_timestamp(timestamp), source)
        except ValueError:
            return None

    def git_creation_timestamp(self, path: Path) -> TimestampInfo | None:
        rel = self.relative_path(path)
        if rel not in self._history:
            self._history[rel] = git_creation_timestamp((self.root / rel).resolve(), self.root)
        return self._history[rel]

    def timestamp_for(self, path: Path) -> TimestampInfo:
        object_id = self.object_id_for_path(path)
        cached = self.cached_timestamp(path, object_id)
        if cached is not None:
            return cached

        prefixed = timestamp_from_full_prefix(path.name)
        timestamp = (
            self.git_creation_timestamp(path)
            or (TimestampInfo(prefixed, "existing full filename prefix") if prefixed else None)
            or filesystem_timestamp(path)
        )
        self.record_path(path, timestamp)
        return timestamp

    def record_path(self, path: Path, timestamp: TimestampInfo | None = None) -> None:
        object_id = self.object_id_for_path(path)
        if not object_id:
            return
        rel = self.relative_path(path)
        timestamp = timestamp or self.timestamp_for(path)
        value = timestamp.value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        entry = {
            "object_id": object_id,
            "archive_timestamp_utc": value,
            "timestamp_source": timestamp.source,
        }
        if self.entries.get(rel) != entry:
            self.entries[rel] = entry
            self.dirty = True

    def record_rename(self, source: Path, target: Path, timestamp: TimestampInfo) -> None:
        source_rel = self.relative_path(source)
        if source_rel in self.entries:
            del self.entries[source_rel]
            self.dirty = True
        self._object_ids.pop(source_rel, None)
        self.record_path(target, timestamp)

    def prune(self, paths: list[Path]) -> None:
        keep = {self.relative_path(path) for path in paths}
        stale = [rel for rel in self.entries if rel not in keep]
        for rel in stale:
            del self.entries[rel]
            self.dirty = True

    def as_json(self) -> str:
        data = {
            "version": CACHE_VERSION,
            "description": "Cache of notes archive timestamps keyed by current repo path and git blob object id.",
            "entries": {rel: self.entries[rel] for rel in sorted(self.entries)},
        }
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    def save(self) -> bool:
        text = self.as_json()
        if self.cache_path.exists() and self.cache_path.read_text(encoding="utf-8") == text:
            self.dirty = False
            return False
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(text, encoding="utf-8")
        self.dirty = False
        return True
