#!/usr/bin/env python3
"""Summarize transcript shards with a command-line model.

The command is optional. Without --command this writes prompt files only.
With --command, the prompt is sent to the command on stdin and stdout is saved
as the shard summary.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROMPT_TEMPLATE = """\
You are summarizing a contiguous shard of mainline project conversation.

Write a concise but information-dense summary for a future agent. Capture:
- ideas and hypotheses raised
- methodology decisions and corrections
- concrete results and caveats
- operational lessons that affect future work
- handoff-relevant state at the shard boundary

Do not try to preserve every message. Use neutral, professional prose focused on
what changed and why. If the conversation established an intended writing form
for a deliverable, such as paper-style, blog-style, article-style, or
report-style, include that. Omit side logistics unless they directly affect
repository workflow.

Shard:

{shard_text}
"""


def run_command(command: list[str], prompt: str) -> str:
    proc = subprocess.run(
        command,
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "summarizer command failed with exit "
            f"{proc.returncode}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="Command to run; everything after --command is treated as argv.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.prompt_dir:
        args.prompt_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(args.shards_dir.glob("shard-*.md"))
    if not shards:
        raise SystemExit(f"No shard-*.md files found in {args.shards_dir}")

    for shard in shards:
        out = args.out_dir / f"{shard.stem}-summary.md"
        prompt = PROMPT_TEMPLATE.format(shard_text=shard.read_text(encoding="utf-8"))
        if args.prompt_dir:
            (args.prompt_dir / f"{shard.stem}-prompt.md").write_text(
                prompt,
                encoding="utf-8",
            )
        if not args.command:
            print(f"wrote prompt for {shard.name}")
            continue
        if out.exists() and not args.overwrite:
            print(f"skip existing {out}")
            continue
        summary = run_command(args.command, prompt)
        out.write_text(summary, encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
