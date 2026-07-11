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
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from notes_archive_naming import DAILY_META_RE
from notes_rollup import with_sources_footer
from notes_summary_filters import resolve_forbid_patterns, run_filtered_summary_command
from summary_model import (
    DEFAULT_CODEX_REASONING,
    DEFAULT_PROVIDER,
    PROVIDERS,
    custom_command_provenance,
    resolve_spec,
    wrapper_command,
)

KIB = 1024
NON_CONVERSATION_THRESHOLD = 8 * KIB
NON_CONVERSATION_HEAD = 6 * KIB
NON_CONVERSATION_TAIL = 2 * KIB
CONVERSATION_THRESHOLD = 16 * KIB
CONVERSATION_HEAD = 12 * KIB
CONVERSATION_TAIL = 4 * KIB

DEFAULT_MANIFEST = Path("scripts/notes-daily-meta-manifest.json")
DEFAULT_COMMAND = wrapper_command(resolve_spec(DEFAULT_PROVIDER, None, DEFAULT_CODEX_REASONING))
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


def source_entries(sources: list[SourceNote], root: Path) -> list[dict[str, str]]:
    return [
        {
            "note": source.path.relative_to(root).as_posix(),
            "blob_id": source.blob_id,
        }
        for source in sorted(sources, key=lambda item: item.path.name)
    ]


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deno_fmt(root: Path, paths: list[Path]) -> None:
    rels = [path.relative_to(root).as_posix() for path in paths if path.exists()]
    if not rels:
        return
    proc = subprocess.run(
        ["deno", "fmt", "--", *rels],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"deno fmt failed with exit {proc.returncode}")


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
- a concise `**Participants/contributors:** ...` paragraph naming the users,
  assistant model identifiers, authors, and labeled subagents represented in
  the source notes; use only source-supported identities and never guess;
- the main research/workflow developments;
- decisions, terminology, or methodological clarifications that should persist;
- empirical results or observations, with caveats;
- outstanding risks, open questions, and likely next actions.

Style: precise, readable prose. Use bullets only where they make dense facts easier to scan. Avoid personal/emotional characterization and dramatic process labels. Avoid publishing-platform logistics unless directly relevant to repository state. Do not mention timestamps beyond the UTC day. Do not directly name sensitive-topic material; describe it generically as an interpretability tangent or notes-hygiene issue if needed.
Do not include a `Sources` section; the caller appends a standardized linked source list after your output.

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


def entry_for(
    day: str,
    note_path: Path,
    summary: str,
    sources: list[SourceNote],
    root: Path,
    summarizer: dict[str, str] | None = None,
) -> dict:
    entry = {
        "note": note_path.relative_to(root).as_posix(),
        "source_count": len(sources),
        "sources": source_entries(sources, root),
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
    if summarizer is not None:
        entry["summarizer"] = summarizer
    return entry


def is_stale(
    day: str,
    note_path: Path,
    sources: list[SourceNote],
    manifest: dict,
    root: Path,
    summarizer: dict[str, str] | None = None,
) -> bool:
    entry = manifest.get("days", {}).get(day)
    if not note_path.exists() or not entry:
        return True
    if entry.get("sources") != source_entries(sources, root):
        return True
    if entry.get("source_blob_ids") != sorted(source.blob_id for source in sources):
        return True
    if entry.get("source_count") != len(sources):
        return True
    if entry.get("summary_hash") != sha256_text(note_path.read_text(encoding="utf-8")):
        return True
    if summarizer is not None and entry.get("summarizer") != summarizer:
        return True
    return False


def remove_summary_if_single_source(
    day: str,
    note_path: Path,
    manifest_path: Path,
    manifest: dict,
    root: Path,
    no_commit: bool,
) -> int:
    changed_paths: list[Path] = []
    days = manifest.setdefault("days", {})
    if day in days:
        days.pop(day, None)
        write_manifest(manifest_path, manifest)
        deno_fmt(root, [manifest_path])
        changed_paths.append(manifest_path)
    if note_path.exists():
        rel = note_path.relative_to(root).as_posix()
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            subprocess.check_call(["git", "rm", "--", rel], cwd=root)
        else:
            note_path.unlink()
            subprocess.check_call(["git", "add", "-A", "--", rel], cwd=root)
        changed_paths.append(note_path)
    if changed_paths and not no_commit:
        commit_paths(root, changed_paths, f"Remove daily meta-summary for {day}")
    return 0


def can_refresh_footer_without_resummarizing(
    day: str,
    note_path: Path,
    sources: list[SourceNote],
    manifest: dict,
    summarizer: dict[str, str] | None = None,
) -> bool:
    entry = manifest.get("days", {}).get(day)
    if not note_path.exists() or not entry:
        return False
    if entry.get("source_count") != len(sources):
        return False
    if entry.get("source_blob_ids") != sorted(source.blob_id for source in sources):
        return False
    if entry.get("summary_hash") != sha256_text(note_path.read_text(encoding="utf-8")):
        return False
    if summarizer is not None and entry.get("summarizer") != summarizer:
        return False
    return True


def refresh_footer_without_resummarizing(
    day: str,
    note_path: Path,
    sources: list[SourceNote],
    manifest_path: Path,
    manifest: dict,
    root: Path,
    no_commit: bool,
    summarizer: dict[str, str] | None = None,
) -> int:
    note_path.write_text(
        with_sources_footer(note_path.read_text(encoding="utf-8"), note_path, [source.path for source in sources]),
        encoding="utf-8",
    )
    deno_fmt(root, [note_path])
    formatted_summary = note_path.read_text(encoding="utf-8")
    manifest.setdefault("days", {})[day] = entry_for(
        day,
        note_path,
        formatted_summary,
        sources,
        root,
        summarizer,
    )
    manifest_changed = write_manifest(manifest_path, manifest)
    changed_paths = [note_path]
    if manifest_changed:
        deno_fmt(root, [manifest_path])
        changed_paths.append(manifest_path)
    if not no_commit:
        commit_paths(root, changed_paths, f"Update daily meta-summary sources for {day}")
    return 0


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
    parser.add_argument("--command")
    parser.add_argument("--summary-provider", choices=PROVIDERS, default=DEFAULT_PROVIDER)
    parser.add_argument("--summary-model")
    parser.add_argument("--summary-reasoning", default=DEFAULT_CODEX_REASONING)
    parser.add_argument("--prompt-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-command", action="store_true", help="write or print the prompt without running Sonnet")
    parser.add_argument("--no-commit", action="store_true", help="write files but leave them uncommitted")
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

    if args.command:
        summarizer_provenance = custom_command_provenance(args.command)
    else:
        summary_spec = resolve_spec(args.summary_provider, args.summary_model, args.summary_reasoning)
        args.command = wrapper_command(summary_spec)
        summarizer_provenance = summary_spec.provenance()

    root = git_root()
    notes_dir = (root / args.notes_dir).resolve()
    manifest_path = (root / args.manifest).resolve()
    note_path = notes_dir / f"{args.day}.md"
    sources = [load_source_note(path, root) for path in source_paths_for_day(notes_dir, args.day)]
    if not sources:
        raise SystemExit(f"no source notes found for UTC day {args.day}")

    manifest = load_manifest(manifest_path)
    if len(sources) == 1:
        stale = note_path.exists() or args.day in manifest.get("days", {})
        print_source_plan(args.day, note_path.relative_to(root), sources, stale)
        print("single source note; no daily meta-summary should be generated")
        if args.dry_run:
            return 0
        return remove_summary_if_single_source(
            args.day,
            note_path,
            manifest_path,
            manifest,
            root,
            args.no_commit,
        )

    stale = args.force or is_stale(
        args.day,
        note_path,
        sources,
        manifest,
        root,
        summarizer_provenance,
    )
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
    if not args.force and can_refresh_footer_without_resummarizing(
        args.day,
        note_path,
        sources,
        manifest,
        summarizer_provenance,
    ):
        print("existing daily summary content is current; refreshing standardized source footer")
        return refresh_footer_without_resummarizing(
            args.day,
            note_path,
            sources,
            manifest_path,
            manifest,
            root,
            args.no_commit,
            summarizer_provenance,
        )
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
        f"daily meta-summary {args.day}",
    )
    summary = with_sources_footer(summary, note_path, [source.path for source in sources])
    notes_dir.mkdir(parents=True, exist_ok=True)
    note_path.write_text(summary, encoding="utf-8")
    deno_fmt(root, [note_path])
    formatted_summary = note_path.read_text(encoding="utf-8")
    manifest.setdefault("days", {})[args.day] = entry_for(
        args.day,
        note_path,
        formatted_summary,
        sources,
        root,
        summarizer_provenance,
    )
    manifest_changed = write_manifest(manifest_path, manifest)
    if manifest_changed:
        deno_fmt(root, [manifest_path])

    changed_paths = [note_path]
    if manifest_changed:
        changed_paths.append(manifest_path)
    if not args.no_commit:
        commit_paths(root, changed_paths, f"Update daily meta-summary for {args.day}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
