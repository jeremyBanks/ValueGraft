#!/usr/bin/env python3
"""Normalize archive-note filenames in notes/.

Names become:

    YYYYMMDDHHMMSS-short-kebab-case-title.md

For tracked files, the timestamp is the first git commit timestamp for the file,
following renames. For untracked files, it falls back to filesystem birth time
when available, then mtime. Markdown-like .txt notes are converted to .md.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


CURRENT_PREFIX_RE = re.compile(r"^\d{14}-")
CURRENT_PREFIX_CAPTURE_RE = re.compile(r"^(\d{14})-")
OLD_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-")
SAFE_TITLE_RE = re.compile(r"[^a-z0-9-]+")
HYPHENS_RE = re.compile(r"-+")
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


def timestamp_from_existing_prefix(path: Path) -> TimestampInfo | None:
    match = CURRENT_PREFIX_CAPTURE_RE.match(path.name)
    if not match:
        return None
    timestamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return TimestampInfo(timestamp, "existing filename prefix")


def timestamp_for(path: Path, root: Path) -> TimestampInfo:
    return (
        git_creation_timestamp(path, root)
        or timestamp_from_existing_prefix(path)
        or filesystem_timestamp(path)
    )


def strip_known_prefix(stem: str) -> str:
    stem = CURRENT_PREFIX_RE.sub("", stem, count=1)
    stem = OLD_PREFIX_RE.sub("", stem, count=1)
    return stem


def kebab_case(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore")
    lowered = normalized.decode("ascii").lower()
    lowered = lowered.replace("_", "-").replace(" ", "-")
    lowered = SAFE_TITLE_RE.sub("-", lowered)
    lowered = HYPHENS_RE.sub("-", lowered)
    return lowered.strip("-") or "untitled"


def target_for(path: Path, root: Path) -> tuple[Path, TimestampInfo, tuple[str, ...]]:
    created = timestamp_for(path, root)
    prefix = created.value.strftime("%Y%m%d%H%M%S")
    title = kebab_case(strip_known_prefix(path.stem))
    target = path.with_name(f"{prefix}-{title}.md")

    reasons: list[str] = []
    if not CURRENT_PREFIX_RE.match(path.name):
        reasons.append("add UTC timestamp prefix")
    elif not path.name.startswith(f"{prefix}-"):
        reasons.append("correct UTC timestamp prefix")
    if strip_known_prefix(path.stem) != title:
        reasons.append("kebab-case title")
    if path.suffix.lower() != ".md":
        reasons.append("convert suffix to .md")
    if path.name != target.name and not reasons:
        reasons.append("normalize filename")

    return target, created, tuple(reasons)


def archive_files(notes_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in notes_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in ARCHIVE_SUFFIXES
        and path.name not in RESERVED_DOC_NAMES
    )


def available_target(base_target: Path, claimed: set[Path], source: Path) -> Path:
    if (not base_target.exists() or base_target == source) and base_target not in claimed:
        return base_target

    for index in range(2, 1000):
        candidate = base_target.with_name(f"{base_target.stem}-{index}{base_target.suffix}")
        if (not candidate.exists() or candidate == source) and candidate not in claimed:
            return candidate
    raise SystemExit(f"could not find an available collision suffix for {base_target}")


def plan_renames(paths: list[Path], root: Path) -> list[Rename]:
    planned: list[Rename] = []
    targets: set[Path] = set()
    for source in paths:
        base_target, timestamp, reasons = target_for(source, root)
        target = available_target(base_target, targets, source)
        if target != base_target:
            reasons = (*reasons, f"avoid name collision with suffix {target.stem.removeprefix(base_target.stem)}")
        if source == target:
            targets.add(target)
            continue
        targets.add(target)
        planned.append(Rename(source=source, target=target, timestamp=timestamp, reasons=reasons))
    return planned


def apply_renames(renames: list[Rename]) -> None:
    for rename in renames:
        rename.source.rename(rename.target)


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
        apply_renames(renames)
        if renames:
            print(f"Applied {len(renames)} rename(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
