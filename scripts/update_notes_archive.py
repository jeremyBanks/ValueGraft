#!/usr/bin/env python3
"""Run the complete notes archive workflow through one command.

Stages run in dependency order: conversation summaries, archive filename
normalization, then daily/month/year/archive rollups.  Each stage can be turned
off, and one provider/model selection is propagated to every generated summary.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from summary_model import DEFAULT_CODEX_REASONING, DEFAULT_PROVIDER, PROVIDERS


def git_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )


def provider_args(args: argparse.Namespace, *, conversation: bool) -> list[str]:
    if args.summary_command:
        if conversation:
            return ["--command", *shlex.split(args.summary_command)]
        return ["--command", args.summary_command]
    result = [
        "--summary-provider",
        args.summary_provider,
        "--summary-reasoning",
        args.summary_reasoning,
    ]
    if args.summary_model:
        result.extend(["--summary-model", args.summary_model])
    return result


def filter_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for pattern in args.forbid_regex or []:
        result.extend(["--forbid-regex", pattern])
    for dotenv in args.dotenv or []:
        result.extend(["--dotenv", dotenv.as_posix()])
    if args.no_default_forbid_regex:
        result.append("--no-default-forbid-regex")
    return result


def build_stage_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    stages: list[tuple[str, list[str]]] = []
    if args.conversations:
        cmd = ["python3", "scripts/transcripts/update_conversation_notes.py", "update"]
        if args.force_small_continuations:
            cmd.append("--force-small-continuations")
        if args.no_subagent_finals:
            cmd.append("--no-subagent-finals")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.resummarize_all:
            cmd.append("--resummarize-all")
        cmd.extend(filter_args(args))
        # `--command` consumes the remainder of the conversation updater argv,
        # so provider/custom-command arguments must always come last.
        cmd.extend(provider_args(args, conversation=True))
        label = "conversation summaries full rebuild" if args.resummarize_all else "conversation summaries"
        stages.append((label, cmd))
    if args.normalize:
        cmd = ["python3", "scripts/normalize_notes_archive_names.py"]
        if args.dry_run:
            cmd.append("--dry-run")
        stages.append(("archive filename normalization", cmd))
    if args.rollups:
        cmd = ["python3", "scripts/update_notes_meta_summaries.py"]
        if args.force_rollups:
            cmd.append("--force")
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(filter_args(args))
        cmd.extend(provider_args(args, conversation=False))
        stages.append(("hierarchical rollups", cmd))
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update conversation summaries (default: enabled).",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize archive filenames and manifest paths (default: enabled).",
    )
    parser.add_argument(
        "--rollups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update daily/month/year/archive summaries (default: enabled).",
    )
    parser.add_argument("--summary-provider", choices=PROVIDERS, default=DEFAULT_PROVIDER)
    parser.add_argument("--summary-model")
    parser.add_argument("--summary-reasoning", default=DEFAULT_CODEX_REASONING)
    parser.add_argument(
        "--summary-command",
        help="Raw custom summary command shared by all summary stages; overrides provider/model flags.",
    )
    parser.add_argument("--force-small-continuations", action="store_true")
    parser.add_argument("--resummarize-all", action="store_true")
    parser.add_argument("--no-subagent-finals", action="store_true")
    parser.add_argument("--force-rollups", action="store_true")
    parser.add_argument("--forbid-regex", action="append", default=[])
    parser.add_argument("--dotenv", action="append", type=Path)
    parser.add_argument("--no-default-forbid-regex", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = git_root()
    stages = build_stage_commands(args)
    if not stages:
        print("No notes archive stages selected.")
        return 0
    for label, cmd in stages:
        print(f"\n== {label} ==", flush=True)
        print("+ " + shlex.join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
