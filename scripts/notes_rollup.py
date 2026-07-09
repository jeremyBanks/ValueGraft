#!/usr/bin/env python3
"""Shared helpers for generated notes rollups."""

from __future__ import annotations

import os
import re
from pathlib import Path

SOURCES_HEADING_RE = re.compile(r"(?m)^## Sources\s*$")


def strip_sources_footer(text: str) -> str:
    """Remove a final generated Sources section if one is present."""
    stripped = text.rstrip()
    matches = list(SOURCES_HEADING_RE.finditer(stripped))
    if not matches:
        return stripped
    last = matches[-1]
    return stripped[: last.start()].rstrip()


def markdown_link_target(source: Path, target: Path) -> str:
    rel = os.path.relpath(source, target.parent).replace(os.sep, "/")
    return rel


def sources_footer(target: Path, sources: list[Path]) -> str:
    ordered = sorted(sources, key=lambda path: path.name)
    lines = ["## Sources", ""]
    for source in ordered:
        link = markdown_link_target(source, target)
        lines.append(f"- [{source.name}]({link})")
    return "\n".join(lines)


def with_sources_footer(body: str, target: Path, sources: list[Path]) -> str:
    clean_body = strip_sources_footer(body)
    return clean_body + "\n\n" + sources_footer(target, sources) + "\n"
