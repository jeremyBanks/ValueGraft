#!/usr/bin/env python3
"""Generate the overall notes history summary from daily meta-summaries."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from notes_archive_naming import DAILY_META_RE
from notes_summary_filters import resolve_forbid_patterns, run_filtered_summary_command
from update_daily_meta_summary import DEFAULT_COMMAND, deno_fmt, git_has_staged_changes, sha256_text

DEFAULT_MANIFEST = Path("scripts/notes-overall-meta-manifest.json")
DEFAULT_NOTE = Path("notes/README.md")


@dataclass(frozen=True)
class DailySummary:
    path: Path
    day: str
    text: str
    blob_id: str


def git_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )


def run_git(args: list[str], root: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def worktree_blob_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return run_git(["hash-object", "--", rel], root)


def daily_summary_paths(notes_dir: Path) -> list[Path]:
    return sorted(path for path in notes_dir.glob("*.md") if DAILY_META_RE.match(path.name))


def load_daily_summaries(notes_dir: Path, root: Path) -> list[DailySummary]:
    summaries: list[DailySummary] = []
    for path in daily_summary_paths(notes_dir):
        summaries.append(
            DailySummary(
                path=path,
                day=path.stem,
                text=path.read_text(encoding="utf-8"),
                blob_id=worktree_blob_id(path, root),
            )
        )
    return summaries


def day_label(day: str) -> str:
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", day)
    if not match:
        return day
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)} UTC"


def build_prompt(summaries: list[DailySummary], root: Path, note_path: Path) -> str:
    parts: list[str] = [
        f"""You are writing {note_path.relative_to(root).as_posix()}, the overall history summary for the ValueGraft research notes archive.

Input: the complete daily meta-summaries, in chronological order. Each source day begins with an explicit date header so you can understand the sequence. Use those dates for chronology, but do not copy those date headers into your own output and do not structure your output as one section per day.

Goal: help a future project agent understand the process and history of what happened from the beginning through the current state. Synthesize across days. Preserve the important pivots, methodology changes, empirical results, failures/corrections, terminology decisions, and remaining open questions. Conclude with a clear account of where the project generally stands now and what a future agent should check first.

Output: the standalone Markdown body. The caller will write your response to {note_path.relative_to(root).as_posix()}. Return only the Markdown document content: no preamble, no code fence, no tool-call syntax, no file-writing description, and no closing status note.

Preferred shape:

- a top-level `# ValueGraft Notes Overview` heading;
- one italicized opening paragraph summarizing the overall arc;
- a small number of thematic sections, not day-by-day sections;
- a concise current-state / handoff section at the end;
- bullets only where they make dense facts easier to scan.

Style: precise, readable prose. Avoid personal/emotional characterization, dramatic process labels, and publishing-platform logistics. Do not directly name sensitive-topic material; describe it generically as an interpretability tangent or notes-hygiene issue if needed.

# Chronological Daily Source Summaries
""",
    ]
    for summary in summaries:
        rel = summary.path.relative_to(root).as_posix()
        parts.append(
            f"""

## Source Day: {day_label(summary.day)}

Source file: {rel}

{summary.text.rstrip()}
"""
        )
    return "".join(parts)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {
            "version": 1,
            "description": "Overall notes meta-summary manifest. Staleness is keyed by ordered daily-summary git blob IDs.",
            "summary": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("summary"), dict):
        raise SystemExit(f"unsupported overall meta manifest format: {path}")
    return data


def write_manifest(path: Path, data: dict) -> bool:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def source_entries(summaries: list[DailySummary], root: Path) -> list[dict[str, str]]:
    return [
        {
            "day": summary.day,
            "note": summary.path.relative_to(root).as_posix(),
            "blob_id": summary.blob_id,
        }
        for summary in summaries
    ]


def entry_for(note_path: Path, summary: str, daily_summaries: list[DailySummary], root: Path) -> dict:
    return {
        "note": note_path.relative_to(root).as_posix(),
        "source_count": len(daily_summaries),
        "sources": source_entries(daily_summaries, root),
        "summary_hash": sha256_text(summary),
    }


def is_stale(note_path: Path, daily_summaries: list[DailySummary], manifest: dict, root: Path) -> bool:
    entry = manifest.get("summary", {})
    if not note_path.exists() or not entry:
        return True
    if entry.get("sources") != source_entries(daily_summaries, root):
        return True
    if entry.get("source_count") != len(daily_summaries):
        return True
    if entry.get("summary_hash") != sha256_text(note_path.read_text(encoding="utf-8")):
        return True
    return False


def commit_paths(root: Path, paths: list[Path], message: str) -> None:
    rels = [path.relative_to(root).as_posix() for path in paths]
    subprocess.check_call(["git", "add", "--", *rels], cwd=root)
    if git_has_staged_changes(root, paths):
        subprocess.check_call(["git", "commit", "-m", message, "--", *rels], cwd=root)


def print_plan(note_path: Path, summaries: list[DailySummary], stale: bool, root: Path) -> None:
    total_chars = sum(len(summary.text) for summary in summaries)
    print(
        f"overall meta-summary: note={note_path.relative_to(root)} "
        f"sources={len(summaries)} stale={stale}"
    )
    for summary in summaries:
        print(f"  {summary.day}: chars={len(summary.text)} blob={summary.blob_id[:12]}")
    print(f"source_chars={total_chars}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes-dir", type=Path, default=Path("notes"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--command", default=DEFAULT_COMMAND)
    parser.add_argument("--prompt-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-command", action="store_true")
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
    note_path = (root / args.note).resolve()
    manifest_path = (root / args.manifest).resolve()
    daily_summaries = load_daily_summaries(notes_dir, root)
    if not daily_summaries:
        raise SystemExit("no daily summaries found")

    manifest = load_manifest(manifest_path)
    stale = args.force or is_stale(note_path, daily_summaries, manifest, root)
    print_plan(note_path, daily_summaries, stale, root)
    prompt = build_prompt(daily_summaries, root, note_path)
    print(f"prompt_chars={len(prompt)}")

    if args.prompt_out:
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_out.write_text(prompt, encoding="utf-8")
        print(f"wrote prompt: {args.prompt_out}")

    if args.dry_run:
        return 0
    if not stale:
        print("overall meta-summary is current; no changes written")
        return 0
    if args.no_command:
        if not args.prompt_out:
            sys.stdout.write(prompt)
        return 0

    forbid_patterns = resolve_forbid_patterns(
        args.forbid_regex,
        root,
        include_defaults=not args.no_default_forbid_regex,
        dotenv_paths=args.dotenv,
    )
    summary = run_filtered_summary_command(
        args.command,
        prompt,
        forbid_patterns,
        args.max_forbid_attempts,
        "overall meta-summary",
    )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(summary, encoding="utf-8")
    deno_fmt(root, [note_path])
    formatted_summary = note_path.read_text(encoding="utf-8")
    manifest["summary"] = entry_for(note_path, formatted_summary, daily_summaries, root)
    manifest_changed = write_manifest(manifest_path, manifest)
    if manifest_changed:
        deno_fmt(root, [manifest_path])

    changed_paths = [note_path]
    if manifest_changed:
        changed_paths.append(manifest_path)
    if not args.no_commit:
        commit_paths(root, changed_paths, "Update overall notes meta-summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
