#!/usr/bin/env python3
"""Normalize archive-note filenames in notes/.

Names become:

    YYYYMMDDNN-short-kebab-case-title.md

The date is UTC. NN is assigned by canonical timestamp and normally continues
across day boundaries. If carrying the prior day's suffix forward would push a
day past 99, that day starts at the corresponding 01..10 trailing digit instead.
For tracked files, the canonical timestamp is the first git commit timestamp for
the file, following renames. For migration from the older full-prefix scheme,
untracked or history-less files may fall back to an existing YYYYMMDDHHMMSS
filename prefix.
Markdown-like .txt notes are converted to .md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

from notes_archive_naming import (
    COMPACT_PREFIX_RE,
    DAILY_META_RE,
    FULL_PREFIX_RE,
    archive_day_start,
    compact_prefix,
    kebab_case,
    strip_known_prefix,
    timestamp_from_full_prefix,
)
from notes_archive_timestamps import (
    DEFAULT_TIMESTAMP_CACHE,
    ArchiveTimestampCache,
    TimestampInfo,
    git_creation_timestamp,
    parse_git_timestamp,
    run_git,
)

RESERVED_DOC_NAMES = {"AGENTS.md", "README.md"}
ARCHIVE_SUFFIXES = {".md", ".txt"}
DEFAULT_MANIFEST = Path("scripts/transcripts/conversation-summary-manifest.json")


@dataclass(frozen=True)
class Rename:
    source: Path
    target: Path
    timestamp: TimestampInfo
    day_index: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ManifestPathUpdate:
    source: str
    target: str


def git_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"]))


def timestamp_for(path: Path, root: Path, cache: ArchiveTimestampCache | None = None) -> TimestampInfo:
    if cache is not None:
        return cache.timestamp_for(path)
    prefixed = timestamp_from_full_prefix(path.name)
    return (
        git_creation_timestamp(path, root)
        or (TimestampInfo(prefixed, "existing full filename prefix") if prefixed else None)
        or ArchiveTimestampCache(root).timestamp_for(path)
    )


def archive_files(notes_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in notes_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in ARCHIVE_SUFFIXES
        and path.name not in RESERVED_DOC_NAMES
        and not DAILY_META_RE.match(path.name)
    )


def plan_renames(paths: list[Path], root: Path, cache: ArchiveTimestampCache | None = None) -> list[Rename]:
    if cache is not None:
        cache.prepare(paths)
    items: list[tuple[Path, TimestampInfo, str]] = []
    for source in paths:
        if source.name in RESERVED_DOC_NAMES or DAILY_META_RE.match(source.name):
            continue
        timestamp = timestamp_for(source, root, cache)
        title = kebab_case(strip_known_prefix(source.stem))
        items.append((source, timestamp, title))

    day_groups: dict[str, list[tuple[Path, TimestampInfo, str]]] = {}
    for item in items:
        _source, timestamp, _title = item
        day = timestamp.value.astimezone(timezone.utc).strftime("%Y%m%d")
        day_groups.setdefault(day, []).append(item)

    planned: list[Rename] = []
    targets: set[Path] = set()
    next_day_index = 1
    for day in sorted(day_groups):
        day_items = sorted(
            day_groups[day],
            key=lambda item: (item[1].value.astimezone(timezone.utc), item[0].name),
        )
        day_start = archive_day_start(next_day_index, len(day_items))
        for offset, (source, timestamp, title) in enumerate(day_items):
            day_index = day_start + offset
            prefix = compact_prefix(timestamp.value, day_index)
            target = source.with_name(f"{prefix}-{title}.md")
            if target in targets:
                raise SystemExit(f"internal collision while planning target: {target}")
            if target.exists() and target != source:
                raise SystemExit(f"target already exists; refusing ambiguous rename order: {target}")

            reasons: list[str] = []
            if not COMPACT_PREFIX_RE.match(source.name):
                if FULL_PREFIX_RE.match(source.name):
                    reasons.append("replace full UTC timestamp prefix with compact archive counter")
                else:
                    reasons.append("add compact UTC date/archive counter prefix")
            elif not source.name.startswith(f"{prefix}-"):
                reasons.append("correct compact archive counter prefix")
            if strip_known_prefix(source.stem) != title:
                reasons.append("kebab-case title")
            if source.suffix.lower() != ".md":
                reasons.append("convert suffix to .md")
            if source.name != target.name and not reasons:
                reasons.append("normalize filename")

            targets.add(target)
            if source == target:
                continue
            planned.append(
                Rename(
                    source=source,
                    target=target,
                    timestamp=timestamp,
                    day_index=day_index,
                    reasons=tuple(reasons),
                )
            )
        next_day_index = day_start + len(day_items)
    return planned


def git_is_tracked(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def stage_rename(rename: Rename, root: Path) -> list[Path]:
    source_rel = rename.source.relative_to(root).as_posix()
    target_rel = rename.target.relative_to(root).as_posix()
    if git_is_tracked(rename.source, root):
        subprocess.check_call(["git", "mv", "--", source_rel, target_rel], cwd=root)
        return [rename.source, rename.target]
    else:
        rename.source.rename(rename.target)
        subprocess.check_call(["git", "add", "--", target_rel], cwd=root)
        return [rename.target]


def apply_renames(renames: list[Rename], root: Path, cache: ArchiveTimestampCache | None = None) -> list[Path]:
    changed_paths: list[Path] = []
    for rename in renames:
        changed_paths.extend(stage_rename(rename, root))
        if cache is not None:
            cache.record_rename(rename.source, rename.target, rename.timestamp)
    return changed_paths


def resolve_manifest_path(path: Path, root: Path) -> Path:
    return (root / path).resolve() if not path.is_absolute() else path


def manifest_path_updates(
    renames: list[Rename],
    root: Path,
    manifest_path: Path,
) -> list[ManifestPathUpdate]:
    manifest_path = resolve_manifest_path(manifest_path, root)
    if not renames or not manifest_path.exists():
        return []
    rename_map = {
        rename.source.relative_to(root).as_posix(): rename.target.relative_to(root).as_posix()
        for rename in renames
    }
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    updates: list[ManifestPathUpdate] = []
    for row in data.get("notes", []):
        note = row.get("note")
        if note in rename_map:
            updates.append(ManifestPathUpdate(note, rename_map[note]))
    return updates


def apply_manifest_path_updates(
    updates: list[ManifestPathUpdate],
    root: Path,
    manifest_path: Path,
) -> Path | None:
    if not updates:
        return None
    manifest_path = resolve_manifest_path(manifest_path, root)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    update_map = {update.source: update.target for update in updates}
    for row in data.get("notes", []):
        note = row.get("note")
        if note in update_map:
            row["note"] = update_map[note]
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_rel = manifest_path.relative_to(root).as_posix()
    subprocess.check_call(["git", "add", "--", manifest_rel], cwd=root)
    return manifest_path


def commit_paths(paths: list[Path], root: Path, message: str) -> None:
    rels: list[str] = []
    seen: set[str] = set()
    for path in paths:
        resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
        rel = resolved.relative_to(root).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        rels.append(rel)
    if not rels:
        return
    subprocess.check_call(["git", "commit", "-m", message, "--", *rels], cwd=root)


def print_plan(renames: list[Rename], root: Path, mode: str, scanned_count: int) -> None:
    print(f"notes archive normalizer: mode={mode} scanned={scanned_count} planned={len(renames)}")
    if not renames:
        print("No archive filenames need normalization.")
        return

    for index, rename in enumerate(renames, 1):
        timestamp = rename.timestamp.value.strftime("%Y-%m-%dT%H:%M:%SZ")
        reasons = ", ".join(rename.reasons) if rename.reasons else "normalize filename"
        print(f"{index}. {rename.source.relative_to(root)}")
        print(f"   -> {rename.target.relative_to(root)}")
        print(f"   timestamp: {timestamp} ({rename.timestamp.source})")
        print(f"   day index: {rename.day_index}")
        print(f"   changes: {reasons}")


def print_manifest_plan(updates: list[ManifestPathUpdate], manifest_path: Path, root: Path) -> None:
    manifest_rel = resolve_manifest_path(manifest_path, root).relative_to(root)
    print(f"manifest path updates: {manifest_rel} planned={len(updates)}")
    for update in updates:
        print(f"   {update.source} -> {update.target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="specific archive-note files to normalize; defaults to notes/*.md and notes/*.txt",
    )
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=Path("notes"),
        help="notes archive directory; default: notes",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="conversation summary manifest to keep in sync; default: scripts/transcripts/conversation-summary-manifest.json",
    )
    parser.add_argument(
        "--timestamp-cache",
        type=Path,
        default=DEFAULT_TIMESTAMP_CACHE,
        help="path/blob keyed archive timestamp cache; default: scripts/notes-archive-timestamps.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned renames without changing the filesystem",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any file would be renamed",
    )
    args = parser.parse_args()

    root = git_root()
    notes_dir = (root / args.notes_dir).resolve()
    if args.paths:
        paths = [(root / path).resolve() if not path.is_absolute() else path for path in args.paths]
    else:
        paths = archive_files(notes_dir)

    for path in paths:
        if notes_dir not in path.parents:
            raise SystemExit(f"refusing to normalize file outside {notes_dir}: {path}")
        if path.name in RESERVED_DOC_NAMES or DAILY_META_RE.match(path.name):
            continue
        if path.suffix.lower() not in ARCHIVE_SUFFIXES:
            raise SystemExit(f"not an archive-note file: {path}")

    cache = ArchiveTimestampCache.load(root, args.timestamp_cache)
    renames = plan_renames(paths, root, cache)
    mode = "check" if args.check else "dry-run" if args.dry_run else "apply"
    print_plan(renames, root, mode, len(paths))
    manifest_updates = manifest_path_updates(renames, root, args.manifest)
    if renames or manifest_updates:
        print_manifest_plan(manifest_updates, args.manifest, root)

    if args.check and renames:
        print("Check failed: run without --check to apply these renames.")
        return 1
    if not args.dry_run:
        changed_paths = apply_renames(renames, root, cache)
        manifest_path = apply_manifest_path_updates(manifest_updates, root, args.manifest)
        if manifest_path is not None:
            changed_paths.append(manifest_path)
        final_paths = archive_files(notes_dir)
        cache.prepare(final_paths)
        cache.prune(final_paths)
        for path in final_paths:
            cache.record_path(path)
        if cache.save():
            subprocess.check_call(["git", "add", "--", cache.cache_path.relative_to(root).as_posix()], cwd=root)
            changed_paths.append(cache.cache_path)
        commit_paths(changed_paths, root, "Normalize note archive filenames")
        if renames:
            print(f"Applied {len(renames)} rename(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
