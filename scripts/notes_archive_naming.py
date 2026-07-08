#!/usr/bin/env python3
"""Shared naming helpers for the notes archive."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone


FULL_PREFIX_RE = re.compile(r"^\d{14}-")
FULL_PREFIX_CAPTURE_RE = re.compile(r"^(\d{14})-")
COMPACT_PREFIX_RE = re.compile(r"^\d{8}[0-9A-Z]{2}-")
OLD_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-")
SAFE_TITLE_RE = re.compile(r"[^a-z0-9-]+")
HYPHENS_RE = re.compile(r"-+")
BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def archive_counter(index: int) -> str:
    if index < 1:
        raise ValueError(f"archive counter indexes are 1-based, got {index}")
    if index <= 99:
        return f"{index:02d}"

    offset = index - 100
    high = 10 + offset // 36
    low = offset % 36
    if high >= len(BASE36):
        raise ValueError(f"archive counter index too large for two base36 digits: {index}")
    return BASE36[high] + BASE36[low]


def compact_prefix(timestamp: datetime, index: int) -> str:
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%d") + archive_counter(index)


def timestamp_from_full_prefix(name: str) -> datetime | None:
    match = FULL_PREFIX_CAPTURE_RE.match(name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def strip_known_prefix(stem: str) -> str:
    stem = FULL_PREFIX_RE.sub("", stem, count=1)
    stem = COMPACT_PREFIX_RE.sub("", stem, count=1)
    stem = OLD_PREFIX_RE.sub("", stem, count=1)
    return stem


def kebab_case(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore")
    lowered = normalized.decode("ascii").lower()
    lowered = lowered.replace("_", "-").replace(" ", "-")
    lowered = SAFE_TITLE_RE.sub("-", lowered)
    lowered = HYPHENS_RE.sub("-", lowered)
    return lowered.strip("-") or "untitled"


def compact_alnum(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore")
    lowered = normalized.decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", lowered) or "unknown"


def participant_slug(participant: str) -> str:
    if participant == "User":
        return "user"

    value = participant.lower()
    if match := re.match(r"^gpt-(\d+)\.(\d+)", value):
        return f"gpt{match.group(1)}{match.group(2)}"
    if match := re.match(r"^claude-fable-(\d+)", value):
        return f"fable{match.group(1)}"
    if match := re.match(r"^claude-opus-(\d+)-(\d+)", value):
        return f"opus{match.group(1)}{match.group(2)}"
    if match := re.match(r"^claude-sonnet-(\d+)", value):
        return f"sonnet{match.group(1)}"

    return compact_alnum(participant)


def conversation_title(participants: list[str]) -> str:
    slugs: list[str] = []
    seen: set[str] = set()
    for participant in participants:
        slug = participant_slug(participant)
        if slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    if not slugs:
        return "conversation"
    return "conversation-" + "-".join(slugs)
