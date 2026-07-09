#!/usr/bin/env python3
"""Shared forbidden-output filtering for notes summary generators."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FORBID_REGEX = [
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bASIA[0-9A-Z]{16}\b",
    r"\bsk-or-v1-[A-Za-z0-9_-]{32,}\b",
    r"\brpa_[A-Za-z0-9]{32,}\b",
    r"\bhf_[A-Za-z0-9]{20,}\b",
]
FORBID_ENV_KEYS = ("VALUEGRAFT_NOTES_FORBID_REGEX", "NOTES_FORBID_REGEX")
DEFAULT_DOTENV_PATHS = (Path(".env"), Path("notes/.env"))

FORBIDDEN_RETRY_PROMPT = """\
A previous attempt at this summary used words or phrases that matched forbidden
output filters.

Forbidden matches seen across attempts so far:
{matches}

Rewrite the summary from scratch. Avoid these exact terms and closely similar
language. Refer to those subjects only with vague, generic phrasing and less
detail. Do not mention the filtering rule, the forbidden list, or the previous
attempt in the summary.

Original task:

{prompt}
"""


@dataclass(frozen=True)
class ForbiddenMatch:
    pattern: str
    text: str


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        values[key] = value
    return values


def dotenv_values(root: Path, paths: list[Path] | None = None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for path in paths or list(DEFAULT_DOTENV_PATHS):
        resolved = path if path.is_absolute() else root / path
        merged.update(parse_dotenv(resolved))
    return merged


def resolve_forbid_patterns(
    cli_patterns: list[str] | None,
    root: Path,
    *,
    include_defaults: bool = True,
    dotenv_paths: list[Path] | None = None,
) -> list[str]:
    patterns: list[str] = list(DEFAULT_FORBID_REGEX) if include_defaults else []
    values = dotenv_values(root, dotenv_paths)
    for key in FORBID_ENV_KEYS:
        value = os.environ.get(key) or values.get(key)
        if value:
            patterns.append(value)
    patterns.extend(pattern for pattern in cli_patterns or [] if pattern)
    return patterns


def forbidden_matches(text: str, forbidden_patterns: list[str]) -> list[ForbiddenMatch]:
    matches: list[ForbiddenMatch] = []
    for pattern in forbidden_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            snippet = match.group(0).strip()
            if snippet:
                matches.append(ForbiddenMatch(pattern=pattern, text=snippet))
    return matches


def display_forbidden_match(match: ForbiddenMatch) -> str:
    if match.pattern in DEFAULT_FORBID_REGEX:
        return f"[redacted token-like match for pattern: {match.pattern}]"
    return match.text


def merge_forbidden_matches(
    existing: list[ForbiddenMatch],
    new_matches: list[ForbiddenMatch],
) -> list[ForbiddenMatch]:
    seen = {(match.pattern, match.text.lower()) for match in existing}
    merged = list(existing)
    for match in new_matches:
        key = (match.pattern, match.text.lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(match)
    return merged


def retry_prompt_for_forbidden_matches(prompt: str, matches: list[ForbiddenMatch]) -> str:
    display_values: list[str] = []
    seen: set[str] = set()
    for match in matches:
        value = display_forbidden_match(match)
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        display_values.append(value)
    rendered = "\n".join(f"- {value}" for value in display_values)
    return FORBIDDEN_RETRY_PROMPT.format(matches=rendered, prompt=prompt)


def scrub_forbidden_lines(text: str, forbidden_patterns: list[str]) -> str:
    lines = text.splitlines()
    kept = [
        line
        for line in lines
        if not any(re.search(pattern, line, re.IGNORECASE) for pattern in forbidden_patterns)
    ]
    scrubbed = "\n".join(kept).strip() + "\n"
    for pattern in forbidden_patterns:
        scrubbed = re.sub(pattern, "", scrubbed, flags=re.IGNORECASE)
    return scrubbed.strip() + "\n"


def run_command(command: str, prompt: str) -> str:
    argv = shlex.split(command)
    if not argv:
        raise RuntimeError("empty summary command")
    proc = subprocess.run(argv, input=prompt, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"summary command failed with exit {proc.returncode}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip() + "\n"


def run_filtered_summary_command(
    command: str,
    prompt: str,
    forbidden_patterns: list[str],
    max_attempts: int,
    label: str,
) -> str:
    all_matches: list[ForbiddenMatch] = []
    current_prompt = prompt
    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        candidate = run_command(command, current_prompt)
        matches = forbidden_matches(candidate, forbidden_patterns)
        if not matches:
            return candidate
        all_matches = merge_forbidden_matches(all_matches, matches)
        print(
            f"forbidden {label} output in attempt {attempt}/{attempts}: "
            f"{len(matches)} match(es), {len(all_matches)} unique accumulated"
        )
        if attempt == attempts:
            scrubbed = scrub_forbidden_lines(candidate, forbidden_patterns)
            remaining = forbidden_matches(scrubbed, forbidden_patterns)
            if remaining:
                raise RuntimeError(
                    f"{label} still matches forbidden filters after scrub: "
                    f"{sorted({match.pattern for match in remaining})}"
                )
            print(f"scrubbed forbidden lines after {attempts} attempt(s): {label}")
            return scrubbed
        current_prompt = retry_prompt_for_forbidden_matches(prompt, all_matches)

    raise AssertionError("unreachable forbidden-filter retry state")
