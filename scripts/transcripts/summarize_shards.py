#!/usr/bin/env python3
"""Summarize transcript segments with a command-line model.

The command is optional. Without --command this writes prompt files only.
With --command, the prompt is sent to the command on stdin and stdout is saved
as the segment summary.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROMPT_TEMPLATE = """\
You are summarizing a contiguous segment of mainline project conversation.

Write a concise but information-dense summary for a future agent. Capture:
- ideas and hypotheses raised
- methodology decisions and corrections
- concrete results and caveats
- operational lessons that affect future work
- handoff-relevant state at the conversation boundary

The first paragraph of your answer must be the italicized opening summary, or
at most two short italicized sentences, summarizing what this conversation
covers. Do
not put any title, heading, bold label, or preamble before that first italicized
paragraph. Then use short titled sections and prose paragraphs. Use bullets only
for compact lists of named results, rules, arms, or open questions; do not turn
the whole conversation into a bullet ledger.

Use neutral, professional prose focused on what changed and why. Preserve
priority and urgency when it affects future work, but express it as project
priority, blocking status, or required follow-up rather than participant mood.
Do not flatten importance: if emphasis changes what a future agent should do
first, record that priority as a project fact or required next action.
Do not describe participant emotions, temperament, or interpersonal tone. Never
use labels such as angry, frustrated, furious, annoyed, or upset. For example,
say "the user requested a status check and made kill verification a required
rule," not "the user was frustrated." Describe corrections, disagreements, and
requirements as project facts. Do not quote colorful or emotionally loaded user
phrasing; paraphrase it into neutral project terms. If the conversation
established an intended writing form for a deliverable, such as paper-style,
blog-style, article-style, or report-style, include that.

Omit side logistics unless they directly affect repository workflow. If the
transcript discusses arXiv, Zenodo, ACM, DOI, uploading, posting, author rights,
coauthor consent, or venue selection, omit those details entirely unless a
tracked repo artifact was changed; at most preserve the intended document style
or a concrete repo workflow change. For transcript-note style discussions,
record only the final durable style rule in general terms. Do not retell the
cleanup episode, mention prohibited words, or quote examples of language to
avoid.

Return only the note body. Do not include preambles such as "Ready to summarize."
Do not try to preserve every message.

{previous_context_block}

Conversation segment:

{shard_text}
"""

PREVIOUS_CONTEXT_TEMPLATE = """\
Brief context from the previous {platform} conversation summary:

{context}
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


def shard_platform(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    platforms = set()
    for line in text.splitlines():
        if line.startswith("# Conversation Chunk: "):
            rest = line.removeprefix("# Conversation Chunk: ")
            platform = rest.split("|", 1)[0].strip()
            if platform:
                platforms.add(platform)
    if len(platforms) == 1:
        return next(iter(platforms))
    if not platforms:
        return "unknown"
    return "mixed"


def context_from_summary(summary: str, max_chars: int) -> str:
    summary = summary.strip()
    if not summary:
        return ""
    first_para = summary.split("\n\n", 1)[0].strip()
    if first_para.startswith("*") and first_para.endswith("*") and len(first_para) <= max_chars:
        return first_para
    compact = " ".join(first_para.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rolling-context-chars",
        type=int,
        default=700,
        help="Carry this many chars from the previous same-source summary into the next prompt; 0 disables.",
    )
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

    previous_context_by_platform: dict[str, str] = {}

    for shard in shards:
        out = args.out_dir / f"{shard.stem}-summary.md"
        platform = shard_platform(shard)
        previous_context = ""
        if args.rolling_context_chars > 0 and platform != "mixed":
            previous_context = previous_context_by_platform.get(platform, "")
        previous_context_block = ""
        if previous_context:
            previous_context_block = PREVIOUS_CONTEXT_TEMPLATE.format(
                platform=platform,
                context=previous_context,
            ).strip()
        prompt = PROMPT_TEMPLATE.format(
            previous_context_block=previous_context_block,
            shard_text=shard.read_text(encoding="utf-8"),
        )
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
            if args.rolling_context_chars > 0 and platform != "mixed":
                previous_context_by_platform[platform] = context_from_summary(
                    out.read_text(encoding="utf-8"),
                    args.rolling_context_chars,
                )
            continue
        summary = run_command(args.command, prompt)
        out.write_text(summary, encoding="utf-8")
        if args.rolling_context_chars > 0 and platform != "mixed":
            previous_context_by_platform[platform] = context_from_summary(
                summary,
                args.rolling_context_chars,
            )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
