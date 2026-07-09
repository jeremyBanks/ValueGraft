#!/usr/bin/env python3
"""Generate UTC daily meta-summaries for notes/.

Daily meta-summaries are special archive files named:

    notes/YYYYMMDD.md

They synthesize the ordinary notes from that UTC day. Conversation summaries are
included in full unless unusually large; other long notes are included as
head/tail excerpts with an explicit omitted-middle marker. Regeneration is
driven by the set of source git blob IDs for the day, not by source filenames.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from notes_archive_naming import DAILY_META_RE

KIB = 1024
NON_CONVERSATION_THRESHOLD = 8 * KIB
NON_CONVERSATION_HEAD = 6 * KIB
NON_CONVERSATION_TAIL = 2 * KIB
CONVERSATION_THRESHOLD = 16 * KIB
CONVERSATION_HEAD = 12 * KIB
CONVERSATION_TAIL = 4 * KIB

DEFAULT_MANIFEST = Path("scripts/notes-daily-meta-manifest.json")
DEFAULT_COMMAND = "claude --print --model sonnet --no-session-persistence --permission-mode dontAsk --disallowedTools *"
RESERVED_NOTE_NAMES = {"AGENTS.md", "README.md"}


@dataclass(frozen=True)
class SourceNote:
    path: Path
    text: str
    blob_id: str
    kind: str
    shown_text: str
    omitted_chars: int


def git_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )


def run_git(args: list[str], root: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def yesterday_utc() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).strftime("%Y%m%d")


def validate_day(day: str) -> str:
    try:
        datetime.strptime(day, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected UTC day as YYYYMMDD, got {day!r}") from exc
    return day


def is_conversation_note(path: Path) -> bool:
    return "-conversation-" in path.name or path.name.endswith("-conversation.md")


def source_paths_for_day(notes_dir: Path, day: str) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(notes_dir.glob(f"{day}*.md")):
        if not path.is_file():
            continue
        if path.name in RESERVED_NOTE_NAMES:
            continue
        if DAILY_META_RE.match(path.name):
            continue
        paths.append(path)
    return paths


def worktree_blob_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return run_git(["hash-object", "--", rel], root)


def cap_text(text: str, threshold: int, head: int, tail: int, label: str) -> tuple[str, int]:
    if len(text) <= threshold:
        return text, 0
    omitted = len(text) - head - tail
    if omitted <= 0:
        return text, 0
    marker = f"\n\n[... {omitted} characters omitted from the middle of this {label} note ...]\n\n"
    return text[:head] + marker + text[-tail:], omitted


def load_source_note(path: Path, root: Path) -> SourceNote:
    text = path.read_text(encoding="utf-8")
    if is_conversation_note(path):
        kind = "conversation"
        shown, omitted = cap_text(
            text,
            CONVERSATION_THRESHOLD,
            CONVERSATION_HEAD,
            CONVERSATION_TAIL,
            kind,
        )
    else:
        kind = "ordinary"
        shown, omitted = cap_text(
            text,
            NON_CONVERSATION_THRESHOLD,
            NON_CONVERSATION_HEAD,
            NON_CONVERSATION_TAIL,
            kind,
        )
    return SourceNote(
        path=path,
        text=text,
        blob_id=worktree_blob_id(path, root),
        kind=kind,
        shown_text=shown,
        omitted_chars=omitted,
    )


def source_signature(sources: list[SourceNote]) -> str:
    # Sorted blob IDs make renames irrelevant while still changing on content edits.
    return "\n".join(sorted(source.blob_id for source in sources))


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt(day: str, sources: list[SourceNote], root: Path) -> str:
    parts: list[str] = [
        f"""You are writing notes/{day}.md, a UTC daily meta-summary for the ValueGraft research repository.

Input: every ordinary Markdown note whose archive filename begins with {day}. Conversation-summary notes are included in full unless unusually large. Other long notes are included as explicitly marked head/tail excerpts. Treat excerpts as incomplete source material and avoid overclaiming from omitted regions.

Output: the standalone Markdown body for a future project agent. The caller will
write your response to notes/{day}.md. Return only the Markdown document
content: no preamble, no code fence, no tool-call syntax, no file-writing
description, and no closing status note. Aim for synthesis, not a ledger.
Include:

- an italicized opening paragraph summarizing the UTC day in one or two sentences;
- the main research/workflow developments;
- decisions, terminology, or methodological clarifications that should persist;
- empirical results or observations, with caveats;
- outstanding risks, open questions, and likely next actions;
- a compact source map listing the input filenames and what each contributed.

Style: precise, readable prose. Use bullets only where they make dense facts easier to scan. Avoid personal/emotional characterization and dramatic process labels. Avoid publishing-platform logistics unless directly relevant to repository state. Do not mention timestamps beyond the UTC day. Do not directly name sensitive-topic material; describe it generically as an interpretability tangent or notes-hygiene issue if needed.

# Source Notes for {day} UTC
""",
    ]
    for source in sources:
        rel = source.path.relative_to(root).as_posix()
        omitted = (
            f"; omitted_middle_chars={source.omitted_chars}"
            if source.omitted_chars
            else ""
        )
        parts.append(
            f"""

## Source: {rel}

Source inclusion: {source.kind}; original_chars={len(source.text)}; shown_chars={len(source.shown_text)}{omitted}.

{source.shown_text.rstrip()}
"""
        )
    return "".join(parts)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {
            "version": 1,
            "description": "Daily notes meta-summary manifest. Staleness is keyed by sorted source git blob IDs, not source filenames.",
            "days": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("days"), dict):
        raise SystemExit(f"unsupported daily meta manifest format: {path}")
    return data


def write_manifest(path: Path, data: dict) -> bool:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def entry_for(day: str, note_path: Path, summary: str, sources: list[SourceNote], root: Path) -> dict:
    return {
        "note": note_path.relative_to(root).as_posix(),
        "source_count": len(sources),
        "source_blob_ids": sorted(source.blob_id for source in sources),
        "summary_hash": sha256_text(summary),
        "input_policy": {
            "ordinary_threshold_chars": NON_CONVERSATION_THRESHOLD,
            "ordinary_head_chars": NON_CONVERSATION_HEAD,
            "ordinary_tail_chars": NON_CONVERSATION_TAIL,
            "conversation_threshold_chars": CONVERSATION_THRESHOLD,
            "conversation_head_chars": CONVERSATION_HEAD,
            "conversation_tail_chars": CONVERSATION_TAIL,
        },
    }


def is_stale(day: str, note_path: Path, sources: list[SourceNote], manifest: dict) -> bool:
    entry = manifest.get("days", {}).get(day)
    if not note_path.exists() or not entry:
        return True
    if entry.get("source_blob_ids") != sorted(source.blob_id for source in sources):
        return True
    if entry.get("source_count") != len(sources):
        return True
    if entry.get("summary_hash") != sha256_text(note_path.read_text(encoding="utf-8")):
        return True
    return False


def run_summary_command(command: str, prompt: str) -> str:
    argv = shlex.split(command)
    if not argv:
        raise SystemExit("empty summary command")
    proc = subprocess.run(argv, input=prompt, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            f"daily meta-summary command failed with exit {proc.returncode}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip() + "\n"


def git_has_staged_changes(root: Path, paths: list[Path]) -> bool:
    rels = [path.relative_to(root).as_posix() for path in paths]
    proc = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *rels],
        cwd=root,
        check=False,
    )
    return proc.returncode == 1


def commit_paths(root: Path, paths: list[Path], message: str) -> None:
    rels = [path.relative_to(root).as_posix() for path in paths]
    subprocess.check_call(["git", "add", "--", *rels], cwd=root)
    if git_has_staged_changes(root, paths):
        subprocess.check_call(["git", "commit", "-m", message, "--", *rels], cwd=root)


def print_source_plan(day: str, note_path: Path, sources: list[SourceNote], stale: bool) -> None:
    full_prompt_chars = 0
    shown_chars = 0
    print(f"daily meta-summary: day={day} note={note_path} sources={len(sources)} stale={stale}")
    for source in sources:
        original = len(source.text)
        shown = len(source.shown_text)
        full_prompt_chars += original
        shown_chars += shown
        suffix = (
            f", omitted={source.omitted_chars}"
            if source.omitted_chars
            else ""
        )
        print(f"  {source.path.name}: kind={source.kind}, original={original}, shown={shown}{suffix}")
    print(f"source_chars_full={full_prompt_chars} source_chars_shown={shown_chars}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("day", nargs="?", type=validate_day, default=yesterday_utc())
    parser.add_argument("--notes-dir", type=Path, default=Path("notes"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--command", default=DEFAULT_COMMAND)
    parser.add_argument("--prompt-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-command", action="store_true", help="write or print the prompt without running Sonnet")
    parser.add_argument("--no-commit", action="store_true", help="write files but leave them uncommitted")
    args = parser.parse_args()

    root = git_root()
    notes_dir = (root / args.notes_dir).resolve()
    manifest_path = (root / args.manifest).resolve()
    note_path = notes_dir / f"{args.day}.md"
    sources = [load_source_note(path, root) for path in source_paths_for_day(notes_dir, args.day)]
    if not sources:
        raise SystemExit(f"no source notes found for UTC day {args.day}")

    manifest = load_manifest(manifest_path)
    stale = args.force or is_stale(args.day, note_path, sources, manifest)
    print_source_plan(args.day, note_path.relative_to(root), sources, stale)
    prompt = build_prompt(args.day, sources, root)
    print(f"prompt_chars={len(prompt)}")

    if args.prompt_out:
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_out.write_text(prompt, encoding="utf-8")
        print(f"wrote prompt: {args.prompt_out}")

    if args.dry_run:
        return 0
    if not stale:
        print("daily meta-summary is current; no changes written")
        return 0
    if args.no_command:
        if not args.prompt_out:
            sys.stdout.write(prompt)
        return 0

    summary = run_summary_command(args.command, prompt)
    notes_dir.mkdir(parents=True, exist_ok=True)
    note_path.write_text(summary, encoding="utf-8")
    manifest.setdefault("days", {})[args.day] = entry_for(args.day, note_path, summary, sources, root)
    manifest_changed = write_manifest(manifest_path, manifest)

    changed_paths = [note_path]
    if manifest_changed:
        changed_paths.append(manifest_path)
    if not args.no_commit:
        commit_paths(root, changed_paths, f"Update daily meta-summary for {args.day}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
