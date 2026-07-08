#!/usr/bin/env python3
"""Normalize archive-note filenames in notes/.

Names become:

    YYYYMMDDNN-short-kebab-case-title.md

The date is UTC. NN is a per-day counter assigned by canonical timestamp:
01..99, then A0, A1, ... if a day ever exceeds 99 files. For tracked files, the
canonical timestamp is the first git commit timestamp for the file, following
renames. For migration from the older full-prefix scheme, untracked or
history-less files may fall back to an existing YYYYMMDDHHMMSS filename prefix.
Markdown-like .txt notes are converted to .md.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from notes_archive_naming import (
    COMPACT_PREFIX_RE,
    FULL_PREFIX_RE,
    compact_prefix,
    kebab_case,
    strip_known_prefix,
    timestamp_from_full_prefix,
)

RESERVED_DOC_NAMES = {"AGENTS.md", "README.md"}
ARCHIVE_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class TimestampInfo:
    value: datetime
    source: str


@dataclass(frozen=True)
class Rename:
    source: Path
    target: Path
    timestamp: TimestampInfo
    day_index: int
    reasons: tuple[str, ...]


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def git_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"]))


def git_creation_timestamp(path: Path, root: Path) -> TimestampInfo | None:
    rel = path.relative_to(root).as_posix()
    try:
        output = run_git(["log", "--follow", "--format=%cI", "--", rel])
    except subprocess.CalledProcessError:
        return None
    lines = [line for line in output.splitlines() if line]
    if not lines:
        return None
    timestamps = [datetime.fromisoformat(line).astimezone(timezone.utc) for line in lines]
    return TimestampInfo(min(timestamps), "git history")


def filesystem_timestamp(path: Path) -> TimestampInfo:
    stat = path.stat()
    if hasattr(stat, "st_birthtime"):
        return TimestampInfo(datetime.fromtimestamp(stat.st_birthtime, timezone.utc), "filesystem birth time")
    return TimestampInfo(datetime.fromtimestamp(stat.st_mtime, timezone.utc), "filesystem mtime")


def timestamp_for(path: Path, root: Path) -> TimestampInfo:
    prefixed = timestamp_from_full_prefix(path.name)
    return (
        git_creation_timestamp(path, root)
        or (TimestampInfo(prefixed, "existing full filename prefix") if prefixed else None)
        or filesystem_timestamp(path)
    )


def archive_files(notes_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in notes_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in ARCHIVE_SUFFIXES
        and path.name not in RESERVED_DOC_NAMES
    )


def plan_renames(paths: list[Path], root: Path) -> list[Rename]:
    items: list[tuple[Path, TimestampInfo, str]] = []
    for source in paths:
        timestamp = timestamp_for(source, root)
        title = kebab_case(strip_known_prefix(source.stem))
        items.append((source, timestamp, title))

    day_groups: dict[str, list[tuple[Path, TimestampInfo, str]]] = {}
    for item in items:
        _source, timestamp, _title = item
        day = timestamp.value.astimezone(timezone.utc).strftime("%Y%m%d")
        day_groups.setdefault(day, []).append(item)

    planned: list[Rename] = []
    targets: set[Path] = set()
    for day in sorted(day_groups):
        day_items = sorted(
            day_groups[day],
            key=lambda item: (item[1].value.astimezone(timezone.utc), item[0].name),
        )
        for day_index, (source, timestamp, title) in enumerate(day_items, 1):
            prefix = compact_prefix(timestamp.value, day_index)
            target = source.with_name(f"{prefix}-{title}.md")
            if target in targets:
                raise SystemExit(f"internal collision while planning target: {target}")
            if target.exists() and target != source:
                raise SystemExit(f"target already exists; refusing ambiguous rename order: {target}")

            reasons: list[str] = []
            if not COMPACT_PREFIX_RE.match(source.name):
                if FULL_PREFIX_RE.match(source.name):
                    reasons.append("replace full UTC timestamp prefix with compact day counter")
                else:
                    reasons.append("add compact UTC date/counter prefix")
            elif not source.name.startswith(f"{prefix}-"):
                reasons.append("correct compact day counter prefix")
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


def commit_rename(rename: Rename, root: Path) -> None:
    source_rel = rename.source.relative_to(root).as_posix()
    target_rel = rename.target.relative_to(root).as_posix()
    if git_is_tracked(rename.source, root):
        subprocess.check_call(["git", "mv", "--", source_rel, target_rel], cwd=root)
        commit_paths = [source_rel, target_rel]
    else:
        rename.source.rename(rename.target)
        subprocess.check_call(["git", "add", "--", target_rel], cwd=root)
        commit_paths = [target_rel]

    iso = rename.timestamp.value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = iso
    env["GIT_COMMITTER_DATE"] = iso
    subprocess.check_call(
        ["git", "commit", "-m", f"Normalize note filename {rename.target.name}", "--", *commit_paths],
        cwd=root,
        env=env,
    )


def apply_renames(renames: list[Rename], root: Path) -> None:
    for rename in renames:
        commit_rename(rename, root)


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
        if path.name in RESERVED_DOC_NAMES:
            continue
        if path.suffix.lower() not in ARCHIVE_SUFFIXES:
            raise SystemExit(f"not an archive-note file: {path}")

    renames = plan_renames(paths, root)
    mode = "check" if args.check else "dry-run" if args.dry_run else "apply"
    print_plan(renames, root, mode, len(paths))

    if args.check and renames:
        print("Check failed: run without --check to apply these renames.")
        return 1
    if not args.dry_run:
        apply_renames(renames, root)
        if renames:
            print(f"Applied {len(renames)} rename(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
