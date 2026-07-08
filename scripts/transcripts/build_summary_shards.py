#!/usr/bin/env python3
"""Build summarizer-facing transcript shards.

Inputs are the already filtered, timestamp-free Markdown transcript chunks.
The script sorts chunks by date/source/sequence and packs adjacent messages
into contiguous shard files, without exposing clock timestamps.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MessageBlock:
    label: str
    text: str


@dataclass
class ConversationChunk:
    platform: str
    date: str
    sequence: int
    source_key: str
    path: Path
    messages: list[MessageBlock]


def read_header_and_body(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}, text
    _, header_text, body = text.split("---", 2)
    header: dict[str, str] = {}
    for line in header_text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return header, body.lstrip()


def parse_messages(body: str) -> list[MessageBlock]:
    pieces = re.split(r"(?m)^## Message ", body)
    messages: list[MessageBlock] = []
    for piece in pieces[1:]:
        first, _, rest = piece.partition("\n\n")
        text = rest.strip()
        if not text:
            continue
        messages.append(MessageBlock(label="Message " + first.strip(), text=text))
    return messages


def load_chunks(filtered_root: Path) -> list[ConversationChunk]:
    chunks: list[ConversationChunk] = []
    for path in filtered_root.glob("*/*/*.md"):
        header, body = read_header_and_body(path)
        platform = header.get("platform", path.stem)
        date = header.get("date_utc", path.parent.name)
        sequence = int(header.get("sequence_in_date", path.stem.split("-", 1)[0]))
        source_key = "0-claude" if platform == "claude-code" else "1-codex"
        chunks.append(
            ConversationChunk(
                platform=platform,
                date=date,
                sequence=sequence,
                source_key=source_key,
                path=path,
                messages=parse_messages(body),
            )
        )
    return sorted(chunks, key=lambda c: (c.date, c.sequence, c.source_key, c.path.name))


def source_sort_key(platform: str) -> str:
    return "0-claude" if platform == "claude-code" else "1-codex"


def chunk_intro(chunk: ConversationChunk) -> str:
    return (
        f"\n\n# Conversation Chunk: {chunk.platform} | date {chunk.date} | "
        f"daily sequence {chunk.sequence:03d}\n\n"
    )


def render_message(block: MessageBlock) -> str:
    return f"## {block.label}\n\n{block.text.strip()}\n\n"


def flush_shard(out_dir: Path, shard_idx: int, parts: list[str], chunk_refs: list[str]) -> None:
    if not parts:
        return
    path = out_dir / f"shard-{shard_idx:03d}.md"
    header = [
        "# Transcript Summary Shard",
        "",
        f"Shard: {shard_idx:03d}",
        "",
        "This file is a contiguous mainline conversation slice for summarization.",
        "It intentionally contains dates and assistant model tags, but no clock times.",
        "",
        "Included chunks:",
        *[f"- {ref}" for ref in chunk_refs],
        "",
        "---",
        "",
    ]
    path.write_text("\n".join(header) + "".join(parts), encoding="utf-8")


def pack_chunks(
    chunks: list[ConversationChunk],
    out_dir: Path,
    shard_idx: int,
    target_chars: int,
) -> int:
    current: list[str] = []
    current_size = 0
    current_refs: list[str] = []
    active_ref: str | None = None

    for chunk in chunks:
        ref = f"{chunk.platform} {chunk.date} #{chunk.sequence:03d}"
        intro = chunk_intro(chunk)
        for message in chunk.messages:
            rendered = render_message(message)
            if current and current_size + len(rendered) > target_chars:
                flush_shard(out_dir, shard_idx, current, current_refs)
                shard_idx += 1
                current = []
                current_size = 0
                current_refs = []
                active_ref = None
            if active_ref != ref:
                current.append(intro)
                current_size += len(intro)
                current_refs.append(ref)
                active_ref = ref
            current.append(rendered)
            current_size += len(rendered)

    if current:
        flush_shard(out_dir, shard_idx, current, current_refs)
        shard_idx += 1
    return shard_idx


def build_shards(
    filtered_root: Path,
    out_dir: Path,
    target_chars: int,
    split_sources: bool,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 1
    chunks = load_chunks(filtered_root)
    if split_sources:
        platforms = sorted({chunk.platform for chunk in chunks}, key=source_sort_key)
        for platform in platforms:
            platform_chunks = [chunk for chunk in chunks if chunk.platform == platform]
            shard_idx = pack_chunks(platform_chunks, out_dir, shard_idx, target_chars)
    else:
        shard_idx = pack_chunks(chunks, out_dir, shard_idx, target_chars)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-root", type=Path, default=Path("/tmp/valuegraft_transcript_work/filtered"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/valuegraft_transcript_work/summary_shards"))
    parser.add_argument("--target-chars", type=int, default=180_000)
    parser.add_argument(
        "--combine-sources",
        action="store_true",
        help="Legacy mode: allow Claude Code and Codex chunks in the same shard.",
    )
    args = parser.parse_args()
    build_shards(args.filtered_root, args.out_dir, args.target_chars, not args.combine_sources)
    paths = sorted(args.out_dir.glob("shard-*.md"))
    print(f"wrote {len(paths)} shards to {args.out_dir}")
    for path in paths:
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
