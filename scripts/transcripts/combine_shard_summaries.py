#!/usr/bin/env python3
"""Combine per-shard transcript summaries into one draft file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def shard_number(path: Path) -> int:
    match = re.search(r"shard-(\d+)-summary$", path.stem)
    if not match:
        raise ValueError(f"Unexpected shard summary name: {path.name}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    paths = sorted(args.summary_dir.glob("shard-*-summary.md"))
    if not paths:
        raise SystemExit(f"No shard-*-summary.md files found in {args.summary_dir}")

    lines = [
        "# ValueGraft Mainline Conversation Summary Draft",
        "",
        "This draft combines per-shard summaries of the main Claude Code and Codex conversations.",
        "",
    ]
    for path in paths:
        lines.append(f"---\n\n## Shard {shard_number(path)}.\n")
        lines.append(path.read_text(encoding="utf-8").strip())
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"combined {len(paths)} summaries -> {args.out}")


if __name__ == "__main__":
    main()
