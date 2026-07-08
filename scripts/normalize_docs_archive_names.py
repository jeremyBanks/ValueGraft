#!/usr/bin/env python3
"""Normalize archive-note filenames in docs/.

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
class Rename:
    source: Path
    target: Path


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def git_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"]))


def git_creation_timestamp(path: Path, root: Path) -> datetime | None:
    rel = path.relative_to(root).as_posix()
    try:
        output = run_git(["log", "--follow", "--format=%cI", "--", rel])
    except subprocess.CalledProcessError:
        return None
    lines = [line for line in output.splitlines() if line]
    if not lines:
        return None
    timestamps = [datetime.fromisoformat(line).astimezone(timezone.utc) for line in lines]
    return min(timestamps)


def filesystem_timestamp(path: Path) -> datetime:
    stat = path.stat()
    timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
    return datetime.fromtimestamp(timestamp, timezone.utc)


def timestamp_from_existing_prefix(path: Path) -> datetime | None:
    match = CURRENT_PREFIX_CAPTURE_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def timestamp_for(path: Path, root: Path) -> datetime:
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


def target_for(path: Path, root: Path) -> Path:
    created = timestamp_for(path, root)
    prefix = created.strftime("%Y%m%d%H%M%S")
    title = kebab_case(strip_known_prefix(path.stem))
    return path.with_name(f"{prefix}-{title}.md")


def archive_files(docs_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in docs_dir.iterdir()
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
        target = available_target(target_for(source, root), targets, source)
        if source == target:
            targets.add(target)
            continue
        targets.add(target)
        planned.append(Rename(source=source, target=target))
    return planned


def apply_renames(renames: list[Rename]) -> None:
    for rename in renames:
        rename.source.rename(rename.target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="specific archive-note files to normalize; defaults to docs/*.md and docs/*.txt",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="documentation archive directory; default: docs",
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
    docs_dir = (root / args.docs_dir).resolve()
    if args.paths:
        paths = [(root / path).resolve() if not path.is_absolute() else path for path in args.paths]
    else:
        paths = archive_files(docs_dir)

    for path in paths:
        if docs_dir not in path.parents:
            raise SystemExit(f"refusing to normalize file outside {docs_dir}: {path}")
        if path.name in RESERVED_DOC_NAMES:
            continue
        if path.suffix.lower() not in ARCHIVE_SUFFIXES:
            raise SystemExit(f"not an archive-note file: {path}")

    renames = plan_renames(paths, root)
    for rename in renames:
        print(f"{rename.source.relative_to(root)} -> {rename.target.relative_to(root)}")

    if args.check and renames:
        return 1
    if not args.dry_run:
        apply_renames(renames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
