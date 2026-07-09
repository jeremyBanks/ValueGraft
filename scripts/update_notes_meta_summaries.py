#!/usr/bin/env python3
"""Update daily notes meta-summaries, then the overall history summary."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from notes_archive_naming import DAILY_META_RE
from update_daily_meta_summary import DEFAULT_COMMAND, sha256_text


def git_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )


def source_days(notes_dir: Path) -> list[str]:
    days: set[str] = set()
    for path in notes_dir.glob("*.md"):
        if DAILY_META_RE.match(path.name):
            continue
        match = re.match(r"^(\d{8})", path.name)
        if match:
            days.add(match.group(1))
    return sorted(days)


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_text(path.read_text(encoding="utf-8"))


def run(cmd: list[str], root: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes-dir", type=Path, default=Path("notes"))
    parser.add_argument("--command", default=DEFAULT_COMMAND)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument(
        "--forbid-regex",
        action="append",
        default=[],
        help="case-insensitive regex forbidden in generated summaries; may be repeated",
    )
    parser.add_argument(
        "--dotenv",
        action="append",
        type=Path,
        default=None,
        help="dotenv file to read forbid-regex env vars from; defaults to .env and notes/.env",
    )
    parser.add_argument(
        "--no-default-forbid-regex",
        action="store_true",
        help="disable default secret-shaped forbid regexes",
    )
    parser.add_argument(
        "--max-forbid-attempts",
        type=int,
        default=5,
        help="summary retries before line-scrubbing forbidden output; default: 5",
    )
    args = parser.parse_args()

    root = git_root()
    notes_dir = (root / args.notes_dir).resolve()
    days = source_days(notes_dir)
    if not days:
        raise SystemExit("no ordinary note days found")

    changed_days: list[str] = []
    for day in days:
        note_path = notes_dir / f"{day}.md"
        before = file_hash(note_path)
        cmd = [
            "python3",
            "scripts/update_daily_meta_summary.py",
            day,
            "--notes-dir",
            args.notes_dir.as_posix(),
            "--command",
            args.command,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.force:
            cmd.append("--force")
        if args.no_commit or not args.dry_run:
            cmd.append("--no-commit")
        for pattern in args.forbid_regex:
            cmd.extend(["--forbid-regex", pattern])
        for dotenv in args.dotenv or []:
            cmd.extend(["--dotenv", dotenv.as_posix()])
        if args.no_default_forbid_regex:
            cmd.append("--no-default-forbid-regex")
        cmd.extend(["--max-forbid-attempts", str(args.max_forbid_attempts)])
        run(cmd, root)
        after = file_hash(note_path)
        if before != after:
            changed_days.append(day)

    overall_cmd = [
        "python3",
        "scripts/update_overall_meta_summary.py",
        "--notes-dir",
        args.notes_dir.as_posix(),
        "--command",
        args.command,
    ]
    if args.dry_run:
        overall_cmd.append("--dry-run")
    if args.force or changed_days:
        overall_cmd.append("--force")
    if args.no_commit or not args.dry_run:
        overall_cmd.append("--no-commit")
    for pattern in args.forbid_regex:
        overall_cmd.extend(["--forbid-regex", pattern])
    for dotenv in args.dotenv or []:
        overall_cmd.extend(["--dotenv", dotenv.as_posix()])
    if args.no_default_forbid_regex:
        overall_cmd.append("--no-default-forbid-regex")
    overall_cmd.extend(["--max-forbid-attempts", str(args.max_forbid_attempts)])
    if changed_days:
        print("daily summaries changed: " + ", ".join(changed_days), flush=True)
    else:
        print("no daily summaries changed", flush=True)
    run(overall_cmd, root)

    if not args.dry_run and not args.no_commit:
        paths = [notes_dir / f"{day}.md" for day in days]
        paths.extend(
            [
                root / "notes/README.md",
                root / "scripts/notes-daily-meta-manifest.json",
                root / "scripts/notes-overall-meta-manifest.json",
            ]
        )
        rels = []
        seen = set()
        for path in paths:
            if not path.exists():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            rels.append(rel)
        if rels:
            subprocess.check_call(["git", "add", "--", *rels], cwd=root)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--quiet", "--", *rels],
                cwd=root,
                check=False,
            )
            if diff.returncode == 1:
                subprocess.check_call(["git", "commit", "-m", "Update notes meta-summaries", "--", *rels], cwd=root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
